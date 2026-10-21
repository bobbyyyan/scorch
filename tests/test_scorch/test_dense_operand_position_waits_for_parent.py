"""A dense operand's position waits for its parent's position, across loops.

A dense level's position is its parent's position times its own extent plus its own
coordinate: ``int pA2 = pA1 * A2_size + j;``.  The lowering keeps each such declaration
pending until the loops have bound the index variables it lists, its own and its
parent's, and then emits it.  That list is complete for a tensor with two dense levels.
With three, the declaration of level 2 lists ``j`` and ``i`` but reads ``pA1``, and
``pA1`` has a declaration of its own that waits for the index variable of level 0.

``scorch.einsum('ijk,jl,kl->il', A, B, C, format='ds')`` with a dense ``A`` and
coordinate ``B`` and ``C`` reached it at every extent up to 38.  ``einsum`` selects a
loop order, moves the operands to it, and selects again; below extent 39 the two
selections differ, so ``A`` is stored ``(k, i, j)`` and the loops run ``j, i, k, l``.
The loop over ``i`` got ``int pA2 = pA1 * A2_size + j;`` and the loop over ``k`` got
``pA0`` and ``pA1``: ``use of undeclared identifier 'pA1'``.  A dense level is random
access, so the loop order is legal (the scheduler puts no storage-order constraint on
an all-dense operand); the declarations were misplaced.

A pending declaration whose index variables are bound now also waits while a position
it reads has a pending declaration that cannot be emitted at this loop
(``_declarable_now`` in ``iter_lattice.py``).  A kernel whose declarations were
already in a valid order is emitted as before.

Four kinds of test:

* the rule itself, on hand-made pending tables;
* every loop order over a dense operand of rank 3 and of rank 4, lowered from a
  hand-built loop nest, so neither the scheduler nor Python's string hashing decides
  whether the case is reached;
* the 37 programs a scan of 9,061 found with the defect (extents 15 and 63), as the
  public call emits them, nothing built;
* the two public programs at extent 15, executed in a child process with
  ``PYTHONHASHSEED=0`` (the loop order ``einsum`` selects can follow the hash seed) and
  judged against a float64 ``torch.einsum``, with the result's own level arrays
  decoded and its stored entries counted.
"""

import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest
import torch

from scorch.compiler import llir
from scorch.compiler.cin import ForAll, IndexVar, TensorAssign, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.iter_lattice import _declarable_now, _parents_first
from tests.test_scorch.level_array_support import emitted_einsum_cpp

INT = llir.DataType.INT
# The C++ type of a position variable in emitted text; the tests read names and order.
POSITION_TYPE = r"(?:int|int64_t)"


def position(tensor: str, level: int, index: str) -> llir.VarInit:
    """The declaration of ``tensor``'s position at a dense ``level``, as the lowering
    builds it (``ModeIterator`` in ``iterator.py``)."""

    own = llir.Var(name=index, type=INT)
    if level == 0:
        value: llir.Expr = own
    else:
        value = llir.Add(
            left=llir.Mul(
                left=llir.Var(name=f"p{tensor}{level - 1}", type=INT),
                right=llir.Var(name=f"{tensor}{level}_size", type=INT),
            ),
            right=own,
        )
    return llir.VarInit(var=llir.Var(name=f"p{tensor}{level}", type=INT), value=value)


def pending_table(tensor: str, storage: str, loops: str, variables):
    """The table as the loops ``loops`` fill it: one declaration per dense level of a
    tensor stored ``storage``, met in loop order, listing its own index variable and
    its parent's."""

    table = {}
    for name in loops:
        level = storage.index(name)
        needed = [variables[name]] + ([variables[storage[level - 1]]] if level else [])
        table[position(tensor, level, name)] = needed
    return table


def released_per_loop(storage: str, loops: str) -> List[List[str]]:
    """The names declared at each loop of ``loops``, in emission order."""

    variables = {name: IndexVar(name) for name in storage}
    table: Dict[llir.VarInit, List[IndexVar]] = {}
    filled = pending_table("A", storage, loops, variables)
    by_loop = []
    for depth, name in enumerate(loops):
        level = storage.index(name)
        for stmt, needed in filled.items():
            if stmt.var.name == f"pA{level}":
                table[stmt] = needed
        ready = _declarable_now(table, {variables[n] for n in loops[: depth + 1]})
        for stmt in ready:
            del table[stmt]
        by_loop.append([stmt.var.name for stmt in _parents_first(list(ready))])
    assert not table
    return by_loop


def test_the_reported_order_declares_all_three_levels_in_the_innermost_loop():
    """``A`` stored ``(k, i, j)`` under loops ``j, i, k``: nothing can be declared
    before ``k`` is bound.  The old rule declared ``pA2`` in the loop over ``i``."""

    assert released_per_loop("kij", "jik") == [[], [], ["pA0", "pA1", "pA2"]]


def test_a_level_is_declared_at_the_first_loop_where_every_level_above_it_is_bound():
    """Every loop order over a dense tensor of rank 2, 3 and 4."""

    for storage in ("ab", "abc", "abcd"):
        for loops in itertools.permutations(storage):
            by_loop = released_per_loop(storage, "".join(loops))
            for level in range(len(storage)):
                first = max(loops.index(name) for name in storage[: level + 1])
                assert f"pA{level}" in by_loop[first], (storage, loops, by_loop)
            for names in by_loop:
                assert names == sorted(names), (storage, loops, by_loop)


def test_a_position_that_is_not_pending_holds_nothing_back():
    """A dense level below a compressed one reads the compressed level's iterator,
    which no declaration in the table provides: it is emitted as its two index
    variables are bound, as before."""

    i, j = IndexVar("i"), IndexVar("j")
    below_compressed = position("B", 1, "j")
    table = {below_compressed: [j, i]}
    assert _declarable_now(table, {j}) == []
    assert _declarable_now(table, {i, j}) == [below_compressed]


def test_tensors_do_not_hold_each_other_back():
    """``pB1`` waits for ``pB0``; ``pA0`` and ``pA1`` of another tensor do not."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    table = {
        position("B", 1, "j"): [j, k],
        position("A", 0, "i"): [i],
        position("A", 1, "j"): [j, i],
        position("B", 0, "k"): [k],
    }
    ready = _declarable_now(table, {i, j})
    assert [stmt.var.name for stmt in ready] == ["pA0", "pA1"]
    ready = _declarable_now(table, {i, j, k})
    assert [stmt.var.name for stmt in ready] == ["pB1", "pA0", "pA1", "pB0"]


# -- every loop order, lowered from a hand-built nest ---------------------------------


def copy_nest(storage: str, loops: str) -> ForAll:
    """``R[loops] = A[storage]``: a dense ``A`` stored ``storage`` and visited in the
    order ``loops``, copied into a dense ``R`` stored in the loops' own order."""

    variables = {name: IndexVar(name) for name in storage}
    extent = {name: 2 + level for level, name in enumerate(storage)}
    operand = TensorVar(
        "A", fmt="d" * len(storage), shape=tuple(extent[name] for name in storage)
    )
    result = TensorVar(
        "R", fmt="d" * len(storage), shape=tuple(extent[name] for name in loops)
    )
    stmt = TensorAssign(
        result[tuple(variables[name] for name in loops)],
        operand[tuple(variables[name] for name in storage)],
    )
    for name in reversed(loops):
        stmt = ForAll(variables[name], stmt)
    assert isinstance(stmt, ForAll)
    return stmt


def read_before_declared(source: str, tensor: str) -> List[str]:
    """The statements that read a position of ``tensor`` no enclosing block declares."""

    name = re.compile(rf"\bp{tensor}\d+\b")
    declaration = re.compile(rf"\s*{POSITION_TYPE} (p{tensor}\d+) = (.*);")
    scopes: List[set] = [set()]
    unresolved = []
    for line in source.splitlines():
        code = line.split("//")[0]
        declared = declaration.match(code)
        for read in name.findall(declared.group(2) if declared else code):
            if not any(read in scope for scope in scopes):
                unresolved.append(code.strip())
        if declared:
            scopes[-1].add(declared.group(1))
        for character in code:
            if character == "{":
                scopes.append(set())
            elif character == "}" and len(scopes) > 1:
                scopes.pop()
    return unresolved


def test_the_check_sees_the_reported_kernel():
    """The check above on the statements of the kernel that did not compile."""

    reported = """
    for (int pB0 = 0; pB0 < pB0_end; pB0 = pB1_end) {
      for (int64_t i = 0; i < A1_size; i++) {
        int pA2 = pA1 * A2_size + j;
        for (int pC0 = 0; pC0 < pC0_end; pC0 = pC1_end) {
          int pA0 = k;
          int pA1 = pA0 * A1_size + i;
          wksp.insert({i, l}, A_val[pA2] * B_val[pB1] * C_val[pC1]);
        }
      }
    }
    """
    assert read_before_declared(reported, "A") == ["int pA2 = pA1 * A2_size + j;"]
    fixed = reported.replace("        int pA2 = pA1 * A2_size + j;\n", "").replace(
        "int pA1 = pA0 * A1_size + i;",
        "int pA1 = pA0 * A1_size + i;\n          int pA2 = pA1 * A2_size + j;",
    )
    assert read_before_declared(fixed, "A") == []


@pytest.mark.parametrize("storage", ["kij", "abcd"])
def test_every_loop_order_over_a_dense_operand_declares_before_it_reads(storage):
    """6 loop orders at rank 3 and 24 at rank 4.  Before the fix an order failed when
    it bound some level after both of the two levels below it."""

    failures = []
    for loops in itertools.permutations(storage):
        source = LLIRLowerer().lower_llir(
            CINLowerer().lower_IndexStmt(copy_nest(storage, "".join(loops)))
        )
        declared = re.findall(rf"\b{POSITION_TYPE} (pA\d+) = ", source)
        unresolved = read_before_declared(source, "A")
        if unresolved or len(declared) != len(set(declared)):
            failures.append(("".join(loops), unresolved, declared))
        assert re.search(r"A_val\w*\[", source), source
    assert not failures, failures


def test_levels_bound_in_storage_order_are_declared_one_per_loop():
    """Loops in ``A``'s own storage order: each loop declares its own level, as it
    always did."""

    source = [
        line.strip()
        for line in LLIRLowerer()
        .lower_llir(CINLowerer().lower_IndexStmt(copy_nest("kij", "kij")))
        .splitlines()
    ]
    for level, index in enumerate("kij"):
        declarations = [
            number
            for number, line in enumerate(source)
            if re.match(rf"{POSITION_TYPE} pA{level} = ", line)
        ]
        if not declarations:
            # the innermost level may be read through a hoisted pointer instead
            assert level == 2
            continue
        (at,) = declarations
        loop = next(line for line in reversed(source[:at]) if line.startswith("for ("))
        assert loop.startswith(f"for (int64_t {index} = "), (level, loop)


# -- the programs found with the defect, as the public call emits them ------------------

# 9,061 programs emitted at extents 15 and 63 with hash seed 0: 37 had a kernel that read
# a position of a dense operand before its declaration.  36 are ``abc,bd,cd->ad`` with a
# dense ``A``, the layouts of ``B`` and ``C`` below and each of four result formats; one
# has a dense operand of rank 4.
FOUND_AT = {
    15: [("oo", "oo"), ("oo", "ss"), ("ss", "oo"), ("ss", "ss"), ("sd", "sd")],
    63: [("oo", "sd"), ("sd", "oo"), ("sd", "ss"), ("ss", "sd"), ("sd", "sd")],
}


def _found():
    for extent, layouts in FOUND_AT.items():
        for b, c in layouts:
            for result in ("ds", "sd", "ss", "oo"):
                yield "abc,bd,cd->ad", ("ddd", b, c), result, extent
    yield "ijkl,ka,lb->ijab", ("dddd", "ss", "sd"), "ddsd", 63


@pytest.mark.parametrize("expr,formats,result,extent", list(_found()))
def test_programs_found_with_the_defect_declare_before_they_read(
    monkeypatch, expr, formats, result, extent
):
    """Nothing is built.  Whether a run reaches the case is the scheduler's choice and
    can follow the hash seed; the hand-built nests above do not depend on either."""

    extents = {name: extent for name in expr if name.isalpha()}
    source = emitted_einsum_cpp(monkeypatch, expr, formats, extents, format=result)
    assert read_before_declared(source, "A") == []
    assert re.search(r"A_val\w*\[", source), "A is not read"


# -- the two public programs, executed --------------------------------------------------

EXPR = "ijk,jl,kl->il"
EXTENT = 15
ALL_IN_ONE_LOOP = re.compile(
    rf"{POSITION_TYPE} pA0 = (\w+);\s*"
    rf"{POSITION_TYPE} pA1 = pA0 \* A1_size \+ (\w+);\s*"
    rf"{POSITION_TYPE} pA2 = pA1 \* A2_size \+ (\w+);"
)


def _child(formats, dtype_name):
    """Run in the child: both inputs through the public call, judged; one JSON row."""

    import scorch
    from scorch import ops
    from tests.test_scorch.level_array_support import (
        assert_matches,
        logical_dense,
        stored,
    )

    dtype = getattr(torch, dtype_name)
    texts = []
    load = ops._load_validated_prepared_kernel

    def recording(prepared):
        texts.append(prepared.request.cpp_sources[-1])
        return load(prepared)

    ops._load_validated_prepared_kernel = recording
    terms = EXPR.split("->")[0].split(",")
    failures = []
    for kind, fill in (("filled", 0.55), ("hollow", 0.15)):
        try:
            generator = torch.Generator().manual_seed(11 + len(kind))
            denses = []
            for term, fmt in zip(terms, formats):
                shape = (EXTENT,) * len(term)
                if dtype.is_floating_point:
                    values = 0.5 + torch.rand(
                        shape, generator=generator, dtype=torch.float64
                    )
                else:
                    values = torch.randint(1, 100, shape, generator=generator).to(
                        torch.float64
                    )
                if set(fmt) != {"d"}:
                    values = values * (torch.rand(shape, generator=generator) < fill)
                    if kind == "hollow":
                        # no stored entry in the first, the middle and the last slice
                        values[[0, EXTENT // 2, EXTENT - 1]] = 0
                denses.append(values.to(dtype))
            operands = [
                stored(dense, fmt, name)
                for dense, fmt, name in zip(denses, formats, "ABC")
            ]
            result = scorch.einsum(EXPR, *operands, format="ds")
            expected = torch.einsum(EXPR, *[d.to(torch.float64) for d in denses])
            assert_matches(result, expected, dtype)
            # Every value is positive, so a cell is stored exactly when it is nonzero.
            decoded = logical_dense(result)
            assert int(result.values.numel()) == int((expected != 0).sum())
            assert torch.equal(decoded != 0, expected != 0)
            assert bool((expected != 0).any()), "the reference is all zero"
        except Exception as error:  # both inputs are reported, not the first
            failures.append(f"{kind}: {type(error).__name__}: {str(error)[:300]}")
    main = texts[-1] if texts else ""
    row = {
        "failures": failures,
        "kernels": len(texts),
        "all_in_one_loop": bool(ALL_IN_ONE_LOOP.search(main)),
        "read_before_declared": read_before_declared(main, "A"),
    }
    print("ROW " + json.dumps(row))


@pytest.mark.parametrize(
    "formats,dtype_name",
    [
        (("ddd", "oo", "oo"), "float64"),
        (("ddd", "oo", "oo"), "float32"),
        (("ddd", "oo", "oo"), "int64"),
        (("ddd", "oo", "ss"), "float64"),
    ],
)
def test_the_reported_programs_match_pytorch(formats, dtype_name):
    import scorch

    root = Path(__file__).resolve().parents[2]
    source = Path(scorch.__file__).resolve().parents[1]
    environment = dict(
        os.environ,
        PYTHONHASHSEED="0",
        PYTHONPATH=os.pathsep.join([str(source), str(root)]),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_scorch.test_dense_operand_position_waits_for_parent",
            json.dumps([list(formats), dtype_name]),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("ROW ")]
    assert completed.returncode == 0 and lines, completed.stderr[-3000:]
    row = json.loads(lines[-1][len("ROW ") :])
    assert not row["failures"], row["failures"]
    assert row["kernels"] >= 1
    assert not row["read_before_declared"], row["read_before_declared"]
    # The premise: at extent 15 and hash seed 0 the loops bind A's top level last, so
    # all three positions are declared in one loop.  If a scheduler change stores A in
    # the loops' order here, this fails; the lowering rule stays pinned by the tests
    # above, and this premise is then the line to update.
    assert row["all_in_one_loop"], "A is no longer visited against its storage order"


if __name__ == "__main__":
    formats_arg, dtype_arg = json.loads(sys.argv[1])
    _child(tuple(formats_arg), dtype_arg)
