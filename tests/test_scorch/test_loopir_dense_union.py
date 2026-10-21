"""Sums with a term present at every coordinate, and unions above the innermost loop.

Two loop forms of the new compiler, both on dense results:

* DENSE_UNION, the iteration of ``A + B`` where ``B`` stores every coordinate
  of the loop's dimension (a dense level) or does not depend on the loop
  variable and stores nothing sparse (``A[i,j] + x[i]`` with ``x`` dense).
  Every coordinate is a candidate; ``A``'s stored stream is followed in
  order, and ``A``'s term is its value where it stores the coordinate and
  zero elsewhere.
* a union above the innermost loop (``A + B`` with both operands storing
  ``i`` compressed): the loop descends at every coordinate either operand
  stores, and the child stream of an operand that does not store the
  coordinate is the empty segment.

Each program is judged against float64 dense semantics: every operand is
broadcast over the full index space, the terms are combined, and the
summed-over indices are summed.  The compiled kernel runs on both automatic
arms; the LoopIR oracle runs the same program.  Inputs include a slice that
stores nothing, and extents of zero and one.

A dense level stored below a level a union iterates (``sd + ds``, and
``dsd``/``ssd`` at rank 3) has a value only where that union's cursor
aligned: the read is guarded by the cursor's alignment, with zero elsewhere.

``REFUSED`` lists what stays refused, with its code: an invariant term
stored compressed (the coordinates it covers depend on an enclosing loop).
The check is strict both ways.

The second half of the file sums into sparse results (every result level
compressed) and runs each cell on both compilers.
"""

from __future__ import annotations

import itertools
import re
from typing import Dict, List, Tuple

import pytest
import torch

import scorch
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.nodes import LevelKind
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.stensor import STensor
from tests.test_scorch import level_array_support
from tests.test_scorch.test_high_rank_programs import (
    _NEW_REFUSALS,
    _SHIPPED_REFUSALS,
    Program,
    _judge,
)
from tests.test_scorch.test_high_rank_programs import _operand as _stored_operand
from tests.test_scorch.test_high_rank_programs import _refusal_code
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

# (expression, operand formats, result format); every term is added.
CELLS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = tuple(
    [
        (f"ij,ij->{out}", fmts, "d" * len(out))
        for fmts in (
            ("dd", "ds"),
            ("ds", "dd"),
            ("ss", "dd"),
            ("dd", "ss"),
            ("ss", "ds"),
            ("ds", "ss"),
            ("ss", "ss"),
            ("sd", "dd"),
            ("sd", "ds"),
            ("sd", "sd"),
        )
        for out in ("ij", "i", "j")
    ]
    + [
        ("ij,i->ij", ("ss", "d"), "dd"),
        ("ij,i->ij", ("ds", "d"), "dd"),
        ("ij,j->ij", ("ss", "d"), "dd"),
        ("ij,i->i", ("ss", "d"), "d"),
        ("ij,i->ij", ("ss", "s"), "dd"),
    ]
    + [
        (f"ijk,ijk->{out}", fmts, "d" * len(out))
        for fmts in (
            ("sss", "dss"),
            ("dss", "sss"),
            ("sss", "ddd"),
            ("sss", "sss"),
            ("dsd", "dss"),
            ("ssd", "sss"),
        )
        for out in ("ijk", "ik", "k")
    ]
    + [("ij,ij,ij->ij", ("ss", "ds", "dd"), "dd")]
    + [
        (f"ij,ij,ij->{out}", ("ss", "ss", "ss"), "d" * len(out))
        for out in ("ij", "i", "j")
    ]
    + [("ijk,ijk,ijk->ijk", ("sss", "sss", "dss"), "ddd")]
)

REFUSED: Dict[str, str] = {
    "ss,s ij,i->ij": "unsupported_union_operand",
}

# The last extents give 70 first-mode coordinates: at 16 rows per worker, a
# kernel whose outer loop is parallel runs it on several workers, which the
# small extents never do.
EXTENTS = (
    {"i": 4, "j": 5, "k": 3},
    {"i": 3, "j": 0, "k": 2},
    {"i": 1, "j": 1, "k": 1},
    {"i": 70, "j": 5, "k": 3},
)


@pytest.fixture(autouse=True)
def _torch_allows_the_teams_these_shapes_were_built_for(torch_threads):
    # The last extents give a parallel outer loop 70 // 16 = 4 workers.
    # torch's count must allow that, or a process started with MKL_NUM_THREADS=1 runs every kernel
    # here on one worker (tests/conftest.py, torch_threads).
    torch_threads(4, needs_team=False)


def _cell_id(cell) -> str:
    expr, fmts, _ = cell
    return f"{','.join(fmts)} {expr}"


def _sparse_cell_id(cell) -> str:
    return f"{_cell_id(cell)} [{cell[2]}]"


def _statement(
    expr: str,
    fmts: Tuple[str, ...],
    result_fmt: str,
    dtype: torch.dtype = torch.float32,
):
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    letters = "".join(dict.fromkeys("".join(terms)))
    variables = {letter: IndexVar(letter) for letter in letters}
    rhs = None
    for name, term, fmt in zip("ABD", terms, fmts):
        access = TensorVar(name, fmt=fmt, dtype=dtype)[
            tuple(variables[letter] for letter in term)
        ]
        rhs = access if rhs is None else CINBinaryOp(Operation.ADD, rhs, access)
    lhs = TensorVar("C", fmt=result_fmt, dtype=dtype)[
        tuple(variables[letter] for letter in output)
    ]
    statement = TensorAssign(
        lhs, rhs, op=Operation.ADD if set(letters) - set(output) else None
    )
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    return statement


# Shifts of the subset inputs (``_subset_keep``).  Coordinates 0, 1, 2 of a mode
# take the masks ``shift .. shift + 2`` modulo ``2**operands``, so the three
# shifts give every subset of up to three operands at any level of extent
# three or more.
SUBSET_SHIFTS = (0, 3, 5)


def _subset_keep(shape, position: int, operands: int, shift: int) -> torch.Tensor:
    """The cells operand ``position`` of ``operands`` stores in a subset input.

    Coordinate ``c`` of a mode is stored by the operands whose bit is set in
    ``(c + shift) mod 2**operands``, and an operand stores a cell when it
    stores each of the cell's coordinates.  Under a prefix some operands
    store, the next coordinate is therefore stored by each subset of them in
    turn, the empty one included: every case of every union runs, and in
    particular the cases in which one operand stores a coordinate another
    does not.  Random inputs leave those out above the last level: a whole
    slice of one operand is empty only by chance, and the ``gap`` and
    ``hollow`` inputs empty the same slices in every operand.
    """

    keep = torch.ones(shape, dtype=torch.bool)
    for mode, extent in enumerate(shape):
        mask = (torch.arange(extent) + shift) % (2**operands)
        view = [1] * len(shape)
        view[mode] = extent
        keep &= ((mask >> position) & 1).bool().reshape(view)
    return keep


# What the dense-result test stores in its operands: random cells, random cells
# with one slice emptied in every operand, and the subset inputs.
DENSE_INPUTS = ("random", "hollow", *(f"subsets+{shift}" for shift in SUBSET_SHIFTS))


def _operand(
    shape, seed: int, kind: str, position: int = 0, operands: int = 1
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5
    keep = torch.rand(shape, generator=generator) < 0.45
    if kind.startswith("subsets+"):
        keep = _subset_keep(shape, position, operands, int(kind.split("+")[1]))
    dense = values * keep
    if kind == "hollow" and len(shape) and shape[0] > 1:
        dense[1] = 0.0  # a slice this operand stores nothing under
    return dense.to(torch.float32)


def _levels(dense: torch.Tensor, fmt: str):
    """Level arrays built from the operand's own nonzeros (dense levels
    expand every coordinate under each stored parent)."""

    nonzeros = {tuple(int(t) for t in index) for index in dense.nonzero()}
    mode_indices, parents = [], [()]
    for level, kind in enumerate(fmt):
        if kind == "d":
            mode_indices.append([])
            parents = [p + (c,) for p in parents for c in range(dense.shape[level])]
            continue
        positions, coordinates, children = [0], [], []
        for parent in parents:
            stored = sorted({c[level] for c in nonzeros if c[:level] == parent})
            coordinates += stored
            children += [parent + (c,) for c in stored]
            positions.append(len(coordinates))
        mode_indices.append(
            [
                torch.tensor(positions, dtype=torch.int64),
                torch.tensor(coordinates, dtype=torch.int64),
            ]
        )
        parents = children
    values = torch.tensor([float(dense[c]) for c in parents], dtype=dense.dtype)
    return mode_indices, values


def _reference(expr: str, denses: List[torch.Tensor]) -> torch.Tensor:
    """Float64 dense semantics: broadcast every term over all indices, add,
    then sum the indices the result does not keep."""

    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    letters = "".join(dict.fromkeys("".join(terms)))
    sizes = {}
    for term, dense in zip(terms, denses):
        sizes.update(zip(term, dense.shape))
    total = torch.zeros([sizes[c] for c in letters], dtype=torch.float64)
    for term, dense in zip(terms, denses):
        view = dense.double().permute([term.index(c) for c in letters if c in term])
        total = total + view[tuple(slice(None) if c in term else None for c in letters)]
    return torch.einsum(f"{letters}->{output}", total)


def _refusal(cell, extents, arm):
    expr, fmts, result_fmt = cell
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    try:
        compile_cin_via_loopir(
            _statement(expr, fmts, result_fmt),
            tuple(extents[c] for c in output),
            tuple((tuple(extents[c] for c in term), torch.float32) for term in terms),
            compile_options=auto_options(arm),
        )
    except (LoopIRLoweringError, LoopIRTargetError) as error:
        return error.defect.code
    return None


def test_the_refusal_table_names_only_cells_of_this_file():
    assert set(REFUSED) <= {_cell_id(cell) for cell in CELLS}


@pytest.mark.parametrize("arm", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("cell", CELLS, ids=[_cell_id(cell) for cell in CELLS])
def test_dense_result_matches_dense_semantics(cell, arm):
    expr, fmts, result_fmt = cell
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    code = _refusal(cell, EXTENTS[0], arm)
    assert code == REFUSED.get(_cell_id(cell)), code
    if code is not None:
        return
    for number, (extents, kind) in enumerate(itertools.product(EXTENTS, DENSE_INPUTS)):
        denses = [
            _operand(
                tuple(extents[c] for c in term),
                31 * number + position,
                kind,
                position,
                len(terms),
            )
            for position, term in enumerate(terms)
        ]
        operands = []
        for dense, fmt, name in zip(denses, fmts, "ABD"):
            mode_indices, values = _levels(dense, fmt)
            operands.append(
                STensor.from_components(
                    tuple(dense.shape), fmt, mode_indices, values, name=name
                )
            )
        reference = _reference(expr, denses)
        result, kernel = execute_cin_via_loopir(
            _statement(expr, fmts, result_fmt),
            tuple(extents[c] for c in output),
            *operands,
            compile_options=auto_options(arm, jit=True),
        )
        torch.testing.assert_close(
            result.to_torch(in_place=False).double(),
            reference,
            atol=1e-5,
            rtol=1e-5,
            msg=lambda m: f"extents {extents}, {kind} input: {m}",
        )
        # The LoopIR oracle runs the same program over the same values.
        program = kernel.lowering.program
        inputs = {}
        for symbol, dense, fmt in zip(program.inputs, denses, fmts):
            nested = dense.double().tolist()
            if "s" in fmt:
                inputs[symbol] = LevelTensorStorage.from_dense(
                    nested,
                    tuple(dense.shape),
                    tuple(range(len(fmt))),
                    tuple(
                        LevelKind.DENSE if kind == "d" else LevelKind.COMPRESSED
                        for kind in fmt
                    ),
                )
            else:
                inputs[symbol] = nested
        oracle = run_program(
            program, inputs, {program.outputs[0]: tuple(reference.shape)}
        )[program.outputs[0]]
        torch.testing.assert_close(
            torch.tensor(oracle, dtype=torch.float64).reshape(reference.shape),
            reference,
            atol=1e-9,
            rtol=1e-9,
        )


def _program_text(cell) -> str:
    expr, fmts, result_fmt = cell
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    extents = EXTENTS[0]
    kernel = compile_cin_via_loopir(
        _statement(expr, fmts, result_fmt),
        tuple(extents[c] for c in output),
        tuple((tuple(extents[c] for c in term), torch.float32) for term in terms),
        compile_options=auto_options(False),
    )
    return kernel.program_text


def test_the_cells_exercise_their_loop_forms():
    """The cells carry the loop forms the module docstring names, where it
    names them; without this the table could pass by taking other routes."""

    def loops(text):
        return [
            (len(line) - len(line.lstrip()), match.group(1))
            for line in text.splitlines()
            for match in [re.match(r"\s*merged_(\w+?)_for ", line)]
            if match
        ]

    # A dense operand beside a sparse one: the inner loop visits every j.
    assert "dense_union" in {kind for _, kind in loops(_program_text(CELLS[0]))}
    # Both operands store i: the outer loop is a union above the inner one.
    ss_ss = next(c for c in CELLS if _cell_id(c) == "ss,ss ij,ij->ij")
    found = loops(_program_text(ss_ss))
    assert [kind for _, kind in found] == ["union", "union"], found
    # A sparse operand and a dense one, both levels: two dense unions.
    ss_dd = next(c for c in CELLS if _cell_id(c) == "ss,dd ij,ij->ij")
    found = loops(_program_text(ss_dd))
    assert [kind for _, kind in found] == ["dense_union", "dense_union"], found


# -- sums into sparse results -------------------------------------------------------
#
# The result stores a compressed coordinate exactly where some operand stores
# an entry under it (a dense operand level stores every coordinate under each
# stored parent), and a dense result level holds every cell of its block.  An
# operand coordinate whose child segments are all empty contributes nothing:
# the "hollow" input below stores such coordinates on purpose.

# A term over fewer indices than the result (``x[j]`` in ``A[i,j] + x[j]``)
# is added at every coordinate of the missing index, so a stored ``x[j]``
# stores column ``j`` in every row of a sparse result.  The new compiler
# computes the dense-result forms and the CSR-like forms whose term is dense
# (a dense loop or a DENSE_UNION visits every coordinate the term covers) and
# refuses the others; the shipped compiler refuses all of them.
INVARIANT_TERM_CELLS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("ij,j->ij", ("ss", "s"), "ss"),
    ("ij,j->ij", ("ds", "d"), "ds"),
    ("ij,j->ij", ("ds", "s"), "ds"),
    ("ij,i->ij", ("ss", "s"), "ss"),
    ("ij,i->ij", ("ss", "d"), "ss"),
    ("ij,i->ij", ("ds", "d"), "ds"),
    ("ij,i->ij", ("ss", "d"), "dd"),
    ("ij,j->ij", ("ss", "d"), "dd"),
    ("ij,j->ij", ("dd", "d"), "dd"),
    ("ijk,jk->ijk", ("sss", "ss"), "sss"),
    ("ijk,ik->ijk", ("dss", "dd"), "ddd"),
    ("ijk,ik->ijk", ("dds", "dd"), "dds"),
    ("ijk,ij->ijk", ("dds", "dd"), "dds"),
)

# Sums with a coordinate (COO) operand or result, which both compilers
# compute.  A COO level with a level below it repeats each coordinate once per
# entry and is walked run by run; the new compiler's unions walk it so (an
# aligned cursor steps over its run, an unaligned one's run is empty) since
# 2026-10-12, and refused it with ``unsupported_coordinate_merge`` before.
COORDINATE_SUM_CELLS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("i,i->i", ("o", "o"), "s"),
    ("i,i->i", ("o", "s"), "s"),
    ("i,i->i", ("o", "d"), "s"),
    ("i,i->i", ("o", "o"), "o"),
    ("i,i->i", ("o", "s"), "d"),
    ("i,i->i", ("o", "o"), "d"),
    ("i,i,i->i", ("o", "o", "s"), "s"),
    ("i,i,i->i", ("o", "d", "o"), "o"),
    ("i,i,i->i", ("o", "d", "s"), "d"),
    ("ij,ij->ij", ("oo", "oo"), "ss"),
    ("ij,ij->ij", ("oo", "oo"), "oo"),
    ("ij,ij->ij", ("ds", "oo"), "ds"),
    ("ij,ij->ij", ("ds", "oo"), "oo"),
    ("ij,ij->ij", ("oo", "ss"), "ss"),
    ("ij,ij->ij", ("oo", "dd"), "ss"),
    ("ij,ij->ij", ("oo", "oo"), "dd"),
    ("ij,ij->ij", ("ss", "ss"), "oo"),
    ("ijk,ijk->ijk", ("ooo", "sss"), "sss"),
    ("ijk,ijk->ijk", ("ooo", "ooo"), "ooo"),
    ("ijk,ijk->ijk", ("ooo", "ddd"), "ddd"),
    ("ij,ij,ij->ij", ("oo", "oo", "ss"), "ss"),
    ("ij,ij->ij", ("oo", "sd"), "ss"),
    ("ij,ij->ij", ("oo", "ss"), "ds"),
    ("ijk,ijk->ijk", ("ooo", "dss"), "dss"),
)

# Sums whose operand stores a dense level below a level a union iterates
# (``sds``): the dense level's position is arithmetic on the union-bound
# position above it, which is absent where that operand stores no ``i``, and
# the compressed level below walks the empty segment there.  The new compiler
# refused them all (``unsupported_union_dense_descent``) until 2026-10-12;
# the shipped compiler computes them.
UNION_DENSE_DESCENT_CELLS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("ijk,ijk->ijk", ("sds", "sds"), "sds"),
    ("ijk,ijk->ijk", ("sss", "sds"), "sss"),
    ("ijk,ijk->ijk", ("sds", "sds"), "sss"),
    ("ijk,ijk->ijk", ("sds", "sss"), "sds"),
    ("ijk,ijk->ijk", ("sds", "dss"), "sss"),
    ("ijk,ijk->ijk", ("sds", "ddd"), "sds"),
    ("ijk,ijk->ijk", ("sds", "sdd"), "sds"),
    ("ijk,ijk,ijk->ijk", ("sds", "sds", "sds"), "sds"),
    ("ijk,ijk,ijk->ijk", ("sds", "sss", "dss"), "sss"),
    ("ijkl,ijkl->ijkl", ("sdds", "sdds"), "sdds"),
    ("ijkl,ijkl->ijkl", ("sdsd", "ssss"), "ssss"),
    ("ijkl,ijkl->ijkl", ("sdsd", "sdsd"), "sdsd"),
    ("ijkl,ijkl->ijkl", ("sdsd", "sdsd"), "ssss"),
    ("ijkl,ijkl->ijkl", ("ssds", "ssds"), "ssds"),
    ("ijkl,ijkl->ijkl", ("sdss", "sdss"), "sdss"),
    ("ijkl,ijkl->ijkl", ("sdss", "ssss"), "dsss"),
)

SPARSE_CELLS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("i,i->i", ("s", "s"), "s"),
    ("i,i->i", ("s", "d"), "s"),
    ("i,i->i", ("d", "s"), "s"),
    *(
        ("ij,ij->ij", fmts, result)
        for fmts in (
            ("ss", "ss"),
            ("ss", "ds"),
            ("ds", "ss"),
            ("ss", "dd"),
            ("dd", "ss"),
            ("ss", "sd"),
            ("sd", "ss"),
            ("sd", "ds"),
            ("ds", "ds"),
            ("sd", "sd"),
        )
        for result in ("ss", "ds", "sd")
    ),
    ("ij,ij->ij", ("dd", "dd"), "ss"),
    ("ij,ij,ij->ij", ("ss", "ss", "ss"), "ss"),
    ("ij,ij,ij->ij", ("ss", "ss", "ss"), "ds"),
    ("ij,ij,ij->ij", ("ss", "ss", "ss"), "sd"),
    ("ij,ij,ij->ij", ("ss", "ds", "dd"), "ss"),
    ("ij,ij,ij->ij", ("sd", "ds", "ss"), "ss"),
    ("ijk,ijk,ijk->ijk", ("sss", "sss", "sss"), "sss"),
    ("ijk,ijk,ijk->ijk", ("sss", "dss", "ssd"), "sss"),
    ("ij,ij,ij->ij", ("ds", "ds", "ds"), "ds"),
    ("ij,ij,ij->ij", ("ds", "ds", "ss"), "ds"),
    ("ijk,ijk,ijk->ijk", ("dds", "dds", "dds"), "dds"),
    *(
        ("ijk,ijk->ijk", fmts, result)
        for fmts in (("sss", "sss"), ("sss", "dss"), ("dss", "ssd"))
        for result in ("sss", "dss", "dds", "ssd", "sdd", "sds")
    ),
    ("ijk,ijk->ijk", ("dss", "sss"), "sss"),
    ("ijk,ijk->ijk", ("sss", "ddd"), "sss"),
    ("ijk,ijk->ijk", ("sss", "ddd"), "dss"),
    ("ijk,ijk->ijk", ("dss", "dss"), "sss"),
    ("ijk,ijk->ijk", ("dsd", "dss"), "dss"),
    ("ijk,ijk->ijk", ("sdd", "sdd"), "sdd"),
    ("ijk,ijk->ijk", ("dsd", "dsd"), "dds"),
    ("ijkl,ijkl->ijkl", ("ssss", "ssss"), "ssss"),
    ("ijkl,ijkl->ijkl", ("ssss", "ssss"), "dsss"),
    ("ijkl,ijkl->ijkl", ("ssss", "dsss"), "ssss"),
    ("ijkl,ijkl->ijkl", ("ssss", "dsss"), "ddss"),
    ("ijkl,ijkl->ijkl", ("ssss", "dddd"), "ssss"),
    *INVARIANT_TERM_CELLS,
    *COORDINATE_SUM_CELLS,
    *UNION_DENSE_DESCENT_CELLS,
)

# Refusal codes, per compiler; a cell absent from a table is computed.
SPARSE_REFUSED_NEW: Dict[str, str] = {
    _sparse_cell_id(cell): code
    for cell, code in (
        (("ij,j->ij", ("ss", "s"), "ss"), "unsupported_union_operand"),
        (("ij,j->ij", ("ds", "s"), "ds"), "unsupported_program_shape"),
        (("ij,i->ij", ("ss", "s"), "ss"), "unsupported_union_operand"),
        (("ij,i->ij", ("ss", "d"), "ss"), "unsupported_program_shape"),
        (("ijk,ij->ijk", ("dds", "dd"), "dds"), "unsupported_program_shape"),
        (("ijk,jk->ijk", ("sss", "ss"), "sss"), "unsupported_union_operand"),
    )
}
# ken's 0029 (2026-10-12): the shipped compiler dropped the term (its body
# became ``C = A``) and now refuses every sum with such a term.
SPARSE_REFUSED_SHIPPED: Dict[str, str] = {
    _sparse_cell_id(cell): "unsupported_sum_addend_without_loop_variable"
    for cell in INVARIANT_TERM_CELLS
}

# Cells the shipped compiler fails on without a refusal, each with its cause, as
# strict expected failures: a cell that starts passing must leave the table.
SPARSE_SHIPPED_DEFECTS: Dict[str, str] = {
    # ``ss,ds,dd ij,ij,ij->ij [ss]`` sat here until ken's 0025 (2026-10-12):
    # the shipped kernel used an undeclared loop counter and did not build.
}

SPARSE_EXTENTS = (
    {"i": 4, "j": 5, "k": 3, "l": 3},
    {"i": 3, "j": 0, "k": 2, "l": 2},
    {"i": 0, "j": 3, "k": 2, "l": 2},
    {"i": 1, "j": 1, "k": 1, "l": 1},
    {"i": 70, "j": 5, "k": 3, "l": 3},  # several workers, as in ``EXTENTS``
)
SPARSE_VARIANTS = (
    "full",
    "sparse",
    "gap",
    "hollow",
    *(f"subsets+{shift}" for shift in SUBSET_SHIFTS),
)


def _sparse_inputs(cell, extents, variant: str, seed: int):
    """Operands built from explicit level arrays, the float64 reference, and
    the result's structure under the storage rule.

    A term over fewer indices than the result (``x[j]`` in ``A[i, j] + x[j]``)
    is broadcast over the others, values and storage alike: a stored ``x[j]``
    contributes an entry at every ``(i, j)``.
    """

    expr, fmts, _ = cell
    terms = expr.split("->")[0].split(",")
    operands, denses, indicators = [], [], []
    for position, (term, fmt) in enumerate(zip(terms, fmts)):
        shape = tuple(extents[c] for c in term)
        generator = torch.Generator().manual_seed(1000 * seed + position)
        values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5
        keep = (
            torch.ones(shape, dtype=torch.bool)
            if variant == "full"
            else torch.rand(shape, generator=generator) < 0.35
        )
        if variant.startswith("subsets+"):
            keep = _subset_keep(shape, position, len(terms), int(variant.split("+")[1]))
        if variant in ("gap", "hollow") and shape[0] > 1:
            keep[0] = False
            keep[-1] = False
        ghosts = None
        if variant == "hollow":
            ghosts = (torch.rand(shape, generator=generator) < 0.35) & ~keep
            if shape[0] > 1:
                ghosts[0] = True
        dense = values * keep
        if set(fmt) == {"o"}:
            # A coordinate tensor stores each kept entry once; it has no
            # segments, so the hollow variant stores nothing extra.
            operand = level_array_support.stored(dense, fmt, "ABD"[position])
            indicator = keep.to(torch.float64)
        else:
            operand, indicator = _stored_operand(
                dense, keep, fmt, "ABD"[position], ghosts
            )
        operands.append(operand)
        denses.append(dense)
        indicators.append(indicator)
    reference = _reference(expr, denses)
    structure = _reference(expr, indicators) > 0
    return operands, reference, structure


def _judge_coordinate_result(result, reference, structure) -> None:
    """A coordinate (COO) result: each implied cell stored once, in order.

    ``coordinate_entries`` checks one array per level, extents and strictly
    increasing lexicographic order; the stored set must equal ``structure``
    and each value its float64 reference cell.
    """

    coords, values = level_array_support.coordinate_entries(result)
    expected = structure.nonzero()
    assert coords.shape[0] == expected.shape[0] and torch.equal(
        coords.reshape(expected.shape), expected
    ), f"stored {coords.tolist()}, implied {expected.tolist()}"
    want = reference[tuple(expected.t())] if expected.numel() else values[:0]
    assert torch.allclose(
        values.to(torch.float64), want, rtol=1e-12, atol=1e-12
    ), f"values {values.tolist()}, expected {want.tolist()}"


def _sparse_outcome(cell, arm: bool, compiler: str):
    """None when every input is computed correctly, else the refusal code."""

    expr, fmts, result_fmt = cell
    output = expr.split("->")[1]
    refusals = _NEW_REFUSALS if compiler == "new" else _SHIPPED_REFUSALS
    shipped_options = CompileOptions.from_environment().with_regblock_enabled(arm)
    for number, extents in enumerate(SPARSE_EXTENTS):
        program = Program(
            family="sum",
            expr=expr,
            operand_formats=fmts,
            result_format=result_fmt,
            extents=tuple(extents.items()),
        )
        for seed, variant in enumerate(SPARSE_VARIANTS):
            operands, reference, structure = _sparse_inputs(
                cell, extents, variant, 10 * number + seed
            )
            statement = _statement(expr, fmts, result_fmt, torch.float64)
            shape = tuple(extents[c] for c in output)
            try:
                if compiler == "new":
                    result, _ = execute_cin_via_loopir(
                        statement,
                        shape,
                        *operands,
                        compile_options=auto_options(arm, jit=True),
                    )
                else:
                    result = scorch.ops.lower_and_exec_cin(
                        statement,
                        shape,
                        *operands,
                        compile_options=shipped_options,
                    )
            except refusals as error:
                return _refusal_code(error)
            try:
                if set(result_fmt) == {"o"}:
                    _judge_coordinate_result(result, reference, structure)
                else:
                    _judge(program, result, reference, structure)
            except AssertionError as error:
                raise AssertionError(
                    f"{compiler} compiler, extents {extents}, {variant} input: {error}"
                ) from error
    return None


def _sparse_parameters():
    parameters = []
    for cell in SPARSE_CELLS:
        name = _sparse_cell_id(cell)
        for compiler in ("new", "shipped"):
            reason = SPARSE_SHIPPED_DEFECTS.get(name) if compiler == "shipped" else None
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            parameters.append(
                pytest.param(cell, compiler, marks=marks, id=f"{name}-{compiler}")
            )
    return parameters


@pytest.mark.parametrize("arm", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("cell,compiler", _sparse_parameters())
def test_sparse_result_sum_matches_dense_semantics(cell, arm, compiler):
    refused = SPARSE_REFUSED_NEW if compiler == "new" else SPARSE_REFUSED_SHIPPED
    assert _sparse_outcome(cell, arm, compiler) == refused.get(_sparse_cell_id(cell))


_POSITION = re.compile(r"\bp[A-Z]\d+\b")
_DECLARED_POSITION = re.compile(r"\bint(?:64_t)?\s+(p[A-Z]\d+)\b")


def _positions_read_out_of_scope(source: str) -> List[str]:
    """Position variables a kernel names where no enclosing block declares them.

    The generated C++ holds one statement per line and declares a position
    (``pA2``: operand ``A``, level 2) with ``int``, in a statement or a
    ``for`` header.  A name read in a block that neither declares it nor is
    nested in one that does is a kernel that does not compile.
    """

    scopes: List[set] = [set()]
    found = []
    for number, line in enumerate(source.splitlines(), 1):
        code = line.split("//")[0].strip()
        while code.startswith("}"):
            scopes.pop()
            code = code[1:].strip()
        # ``std::min({i_A, i_B})`` opens and closes a brace on one line.
        net = code.count("{") - code.count("}")
        scopes.extend(set() for _ in range(max(net, 0)))
        scopes[-1].update(_DECLARED_POSITION.findall(code))
        for name in _POSITION.findall(code):
            if not any(name in scope for scope in scopes):
                found.append(f"{name} at line {number}")
        for _ in range(max(-net, 0)):
            scopes.pop()
    assert len(scopes) == 1, "unbalanced braces"
    return found


def test_the_scope_scan_reports_a_position_read_outside_its_block():
    source = """
    int pA0 = 0;
    if (i_A == i && i_B == i) {
      int pB2 = B2_pos[pB1];
      for (int64_t l = 0; l < A3_size; l++) {
        int pB3 = pB2 * B3_size + l;
      }
    } else if (i_A == i) {
      for (int pA2 = A2_pos[pA0]; pA2 < pA2_end; pA2++) {
        int k = std::min({k_A, k_B});
        int pB3 = pB2 * B3_size + l;
      }
    }
    """
    assert _positions_read_out_of_scope(source) == [
        "pB1 at line 4",
        "pB2 at line 11",
    ]


@pytest.mark.parametrize("arm", (False, True), ids=("plain", "blocked"))
def test_every_position_a_sum_kernel_reads_is_declared_in_scope(arm):
    """Every cell of this file, nothing compiled.

    Until 2026-10-12 a union case that drains one operand still computed the
    dense positions of the operand it does not iterate: ``sdsd + sdsd`` read
    ``pB2`` where only ``A`` has an entry, and the kernel did not build.
    """

    extents = SPARSE_EXTENTS[0]
    found = {}
    for cell in (*CELLS, *SPARSE_CELLS):
        expr, fmts, result_fmt = cell
        terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
        try:
            kernel = compile_cin_via_loopir(
                _statement(expr, fmts, result_fmt, torch.float64),
                tuple(extents[c] for c in output),
                tuple(
                    (tuple(extents[c] for c in term), torch.float64) for term in terms
                ),
                compile_options=auto_options(arm, jit=True),
            )
        except (LoopIRLoweringError, LoopIRTargetError):
            continue
        out_of_scope = _positions_read_out_of_scope(kernel.cpp_source)
        if out_of_scope:
            found[_sparse_cell_id(cell)] = out_of_scope
    assert not found


def test_the_sparse_refusal_tables_name_only_cells_of_this_file():
    names = {_sparse_cell_id(cell) for cell in SPARSE_CELLS}
    assert set(SPARSE_REFUSED_NEW) <= names
    assert set(SPARSE_REFUSED_SHIPPED) <= names
    assert set(SPARSE_SHIPPED_DEFECTS) <= names
