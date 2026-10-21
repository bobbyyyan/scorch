"""A tensor whose levels are all coordinate levels: a COO tensor.

A coordinate level stores one coordinate per stored entry and no position
array.  A tensor whose levels are all coordinate levels stores its entries in
lexicographic order.  As an operand, the new compiler walks it level by level
the way the shipped compiler does: level 0 visits each distinct coordinate once
and finds the end of its run of entries by scanning forward; a later level
walks the run its parent stands on; the last level walks entries one by one.
As a result, it is assembled the way the same result with every level
compressed is, one appended entry per stored cell in lexicographic order, and
each entry stores its coordinate on every level.

Three kinds of test.  The refusals that stay, each with its structured code: a
coordinate level beside a level of another kind, in an operand or in a result
(the shipped compiler refuses those tensors too), and a coordinate level
co-iterated with another sparse operand.  Text: every listed program with a
dense result emits byte for byte the C++ the shipped compiler emits for it, on
both settings of register blocking and on the scheduled and the unscheduled
route, so its runtime is the shipped kernel's; a program with a sparse result
does wherever the same program over compressed levels does.  Execution against
dense PyTorch in float64 at 1e-12 on operands that hold stored zeros, on an
empty operand, and on an operand whose only entry is a stored zero; a sparse
result must also store exactly the coordinates the operands' stored entries
imply, read from the result's own arrays.
"""

import re
from dataclasses import replace
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import pytest
import torch

from scorch import STensor
from scorch.compiler import llir
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir import lower_llir
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.lower_cin import lower_normalized_cin_to_loopir
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.nodes import LevelKind, MergeMode, ScalarType
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.compiler.loopir.verifier import LoopIRVerificationError, verify_program
from scorch.compiler.scheduler import Schedule

from tests.test_scorch.test_loopir_operand_conversions import (
    DTYPE,
    EXTENT,
    INPUTS,
    REFUSALS,
    level_arrays,
    operand_values,
    program,
    result_cells,
    shapes,
    stored,
    stored_cells,
)

Operand = Tuple[str, str, str]
Case = Tuple[List[Operand], str, str]


def options(arm: int, scheduled: bool) -> CompileOptions:
    plain = CompileOptions.from_environment().with_regblock_enabled(bool(arm))
    return replace(plain, requested_schedule=Schedule()) if scheduled else plain


def label(case: Case) -> str:
    operands, result_indices, result_format = case
    return (
        ",".join(fmt for _, fmt, _ in operands)
        + " "
        + ",".join(indices for _, _, indices in operands)
        + f"->{result_indices} [{result_format}]"
    )


def outcome(operands, result_indices, result_format, arm, scheduled=True):
    """``(kernel, None)``, or ``(None, "<type>/<code>")`` for a refusal."""

    result_shape, bindings = shapes(operands, result_indices)
    try:
        return (
            compile_cin_via_loopir(
                program(operands, result_indices, result_format),
                result_shape,
                bindings,
                compile_options=options(arm, scheduled),
            ),
            None,
        )
    except REFUSALS as error:
        defect = getattr(error, "defect", None)
        code = getattr(defect, "code", None) or getattr(error, "code", None)
        return None, f"{type(error).__name__}/{code}"


# -- operands ------------------------------------------------------------------


def coo(values: torch.Tensor, keep: torch.Tensor, name: str) -> STensor:
    """The COO tensor that stores exactly the cells ``keep``, zeros included."""

    cells = keep.nonzero()
    return STensor.from_coo(
        indices=cells.t().contiguous(),
        values=values[tuple(cells.t())],
        shape=tuple(values.shape),
        name=name,
    )


def operand(values: torch.Tensor, keep: torch.Tensor, fmt: str, name: str) -> STensor:
    if set(fmt) == {"o"}:
        return coo(values, keep, name)
    return stored(values, keep, fmt, name)


def cells_of(keep: torch.Tensor, fmt: str) -> torch.Tensor:
    """The cells an operand built by :func:`operand` stores."""

    return keep.clone() if set(fmt) == {"o"} else stored_cells(keep, fmt)


def inputs_of(operands: Sequence[Operand], kind: str):
    """``(tensors, dense values, stored-cell indicators)`` of one input kind.

    The ``random`` input of every operand with a stored level holds at least
    one stored zero: where the shared generator happens to keep no zero (a
    rank-1 operand has five cells), the first kept cell is set to zero.
    """

    tensors, references, indicators = [], [], []
    for position, (name, fmt, indices) in enumerate(operands):
        shape = tuple(EXTENT[letter] for letter in indices)
        values, keep = operand_values(kind, shape, position)
        cells = cells_of(keep, fmt)
        if kind == "random" and set(fmt) != {"d"}:
            if not bool((cells & (values == 0)).any()):
                values[tuple(cells.nonzero()[0].tolist())] = 0.0
            assert bool((cells & (values == 0)).any()), "no stored zero"
        tensors.append(operand(values, keep, fmt, name))
        if set(fmt) == {"o"}:
            # The premise of every test below: the COO operand stores the
            # kept cells and nothing else.
            assert tensors[-1].values.numel() == int(keep.sum())
        references.append(values)
        indicators.append(cells.to(DTYPE))
    if kind == "one stored zero":
        assert all(float(reference.abs().sum()) == 0.0 for reference in references)
    return tensors, references, indicators


def test_the_operand_builder_stores_what_it_is_given():
    values = torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0]], dtype=DTYPE)
    keep = torch.tensor([[True, True, False], [True, False, True]])
    tensor = coo(values, keep, "A")
    assert str(tensor.format) == "o,o"
    rows, columns = (level[0].tolist() for level in tensor.storage.index.mode_indices)
    assert (rows, columns) == ([0, 0, 1, 1], [0, 1, 0, 2])
    # The stored zero at (0, 0) is an entry.
    assert tensor.values.tolist() == [0.0, 2.0, 3.0, 4.0]
    empty = coo(
        torch.zeros(2, 3, dtype=DTYPE), torch.zeros(2, 3, dtype=torch.bool), "A"
    )
    assert [level[0].numel() for level in empty.storage.index.mode_indices] == [0, 0]
    assert empty.values.numel() == 0


# -- the refusals that stay ----------------------------------------------------

MIXED = "LoopIRLoweringError/unsupported_mixed_coordinate_hierarchy"
REFUSED = [
    # A coordinate level beside a level of another kind has no segment.
    ([("A", "do", "ij")], "ij", "dd", MIXED),
    ([("A", "od", "ij")], "ij", "dd", MIXED),
    ([("A", "so", "ij")], "i", "d", MIXED),
    ([("A", "os", "ij")], "i", "d", MIXED),
    ([("A", "soo", "ijk")], "ijk", "ddd", MIXED),
    ([("A", "doo", "ijk")], "k", "d", MIXED),
    # A result with a coordinate level beside a level of another kind.
    ([("A", "ss", "ij")], "ij", "so", MIXED),
    ([("A", "ss", "ij")], "ij", "os", MIXED),
    ([("A", "ss", "ij")], "ij", "do", MIXED),
    ([("A", "oo", "ij")], "ij", "od", MIXED),
    ([("A", "sss", "ijk")], "ijk", "oos", MIXED),
]


@pytest.mark.parametrize(
    "operands,result_indices,result_format,code",
    REFUSED,
    ids=[label(case[:3]) for case in REFUSED],
)
@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("scheduled", [True, False])
def test_what_is_not_lowered_is_refused_with_its_code(
    operands, result_indices, result_format, code, arm, scheduled
):
    kernel, refusal = outcome(operands, result_indices, result_format, arm, scheduled)
    assert kernel is None
    assert refusal == code


# -- the verifier's rules ------------------------------------------------------


def coordinate_copy(
    kinds: Sequence[LevelKind],
    merged: Optional[MergeMode] = None,
    *,
    merge_rows: bool = False,
):
    """A rank-2 input read by two nested sparse loops and stored densely.

    ``kinds`` are the input's two level kinds.  With ``merged`` the inner
    loop is a two-cursor merge of that mode of the input with itself; with
    ``merge_rows`` as well, the outer loop is that merge instead, over level 0.
    """

    builder = LoopIRBuilder()
    rows, columns = builder.dimension("i"), builder.dimension("j")
    dimensions = (rows.dimension, columns.dimension)
    source = builder.new_symbol_id()
    target = builder.new_symbol_id()
    tensors = (
        builder.tensor(
            source,
            "A",
            ScalarType.FLOAT64,
            dimensions,
            tuple(builder.level(kind, mode) for mode, kind in enumerate(kinds)),
        ),
        builder.tensor(
            target, "C", ScalarType.FLOAT64, dimensions, builder.dense_levels(2)
        ),
    )
    i, j = builder.new_index_id(), builder.new_index_id()
    outer_position, inner_position = (
        builder.new_position_id(),
        builder.new_position_id(),
    )
    outer = builder.sparse_cursor(
        builder.new_cursor_id(), source, 0, builder.root_position()
    )
    inner_cursor = builder.new_cursor_id()
    inner = builder.sparse_cursor(
        inner_cursor, source, 1, builder.position_value(outer_position)
    )
    store = builder.store(
        target,
        (builder.index_value(i), builder.index_value(j)),
        builder.cursor_value(
            # A cursor read under a union names the value of an absent entry.
            inner_cursor,
            (
                builder.float_const(0.0)
                if merged is MergeMode.UNION and not merge_rows
                else None
            ),
        ),
    )
    if merged is not None and not merge_rows:
        other = builder.sparse_cursor(
            builder.new_cursor_id(), source, 1, builder.position_value(outer_position)
        )
        inner_loop = builder.merged_sparse_for(
            merged, (inner, other), j, builder.block((store,))
        )
    else:
        inner_loop = builder.sparse_for(
            inner, inner_position, j, builder.block((store,))
        )
    if merged is not None and merge_rows:
        other_outer = builder.sparse_cursor(
            builder.new_cursor_id(), source, 0, builder.root_position()
        )
        outer_loop = builder.merged_sparse_for(
            merged,
            (outer, other_outer),
            i,
            builder.block((inner_loop,)),
            (outer_position, builder.new_position_id()),
        )
    else:
        outer_loop = builder.sparse_for(
            outer, outer_position, i, builder.block((inner_loop,))
        )
    body = builder.block((outer_loop,))
    return builder.program((rows, columns), tensors, (source,), (target,), body)


def defect_code(build) -> Optional[str]:
    try:
        verify_program(build())
    except LoopIRVerificationError as error:
        return error.defect.code
    return None


def test_the_verifier_admits_a_coo_input_walked_by_single_cursor_loops():
    both = (LevelKind.COORDINATE, LevelKind.COORDINATE)
    assert defect_code(lambda: coordinate_copy(both)) is None


@pytest.mark.parametrize(
    "kinds",
    [
        (LevelKind.COORDINATE, LevelKind.COMPRESSED),
        (LevelKind.COMPRESSED, LevelKind.COORDINATE),
        (LevelKind.DENSE, LevelKind.COORDINATE),
        (LevelKind.COORDINATE, LevelKind.DENSE),
    ],
)
def test_the_verifier_refuses_a_coordinate_level_beside_another_kind(kinds):
    assert defect_code(lambda: coordinate_copy(kinds)) == "unsupported_level_kind"


def test_the_verifier_admits_an_intersection_over_a_coordinate_level():
    both = (LevelKind.COORDINATE, LevelKind.COORDINATE)
    merged = MergeMode.INTERSECTION
    assert defect_code(lambda: coordinate_copy(both, merged)) is None


@pytest.mark.parametrize("merge_rows", [False, True], ids=["columns", "rows"])
def test_the_verifier_admits_a_union_over_a_coordinate_level(merge_rows):
    """A union walks a COO level run by run, as an intersection does.

    Level 0 of a COO tensor repeats each row once per entry: an aligned
    cursor steps over its run and an unaligned one's run is empty.  The last
    level stores each coordinate of a run once.  Both were refused until
    2026-10-12, when the target learned to walk runs under a union.
    """

    both = (LevelKind.COORDINATE, LevelKind.COORDINATE)
    merged = MergeMode.UNION
    assert (
        defect_code(lambda: coordinate_copy(both, merged, merge_rows=merge_rows))
        is None
    )


@pytest.mark.parametrize("kind", [LevelKind.SINGLETON])
def test_the_verifier_still_refuses_singleton_levels(kind):
    assert (
        defect_code(lambda: coordinate_copy((LevelKind.COMPRESSED, kind)))
        == "unsupported_level_kind"
    )


# -- text and execution --------------------------------------------------------

# Programs whose sparse operands are COO tensors, into a dense result.  The
# last field says whether the shipped compiler emits C++ for the program on
# the route named: on the scheduled route it refuses a dense result written in
# an order its plan rules forbid, and on the unscheduled route it emits text
# that reads a result position before declaring it.
READ: List[Tuple[List[Operand], str, str, Tuple[bool, bool]]] = [
    ([("A", "oo", "ij")], "i", "d", (True, True)),
    ([("A", "oo", "ij")], "j", "d", (True, True)),
    ([("A", "oo", "ij")], "ij", "dd", (True, True)),
    ([("A", "oo", "ij")], "ji", "dd", (False, False)),
    ([("A", "oo", "ij"), ("B", "d", "j")], "i", "d", (True, True)),
    ([("A", "oo", "ij"), ("B", "d", "i")], "j", "d", (True, True)),
    ([("A", "oo", "ij"), ("B", "dd", "jk")], "ik", "dd", (True, True)),
    ([("A", "dd", "ij"), ("B", "oo", "jk")], "ik", "dd", (False, True)),
    ([("A", "oo", "ij"), ("B", "ds", "jk")], "ik", "dd", (True, True)),
    ([("A", "ooo", "ijk")], "ijk", "ddd", (True, True)),
    ([("A", "ooo", "ijk")], "ij", "dd", (True, True)),
    ([("A", "ooo", "ijk")], "k", "d", (True, True)),
    ([("A", "ooo", "ijk")], "jk", "dd", (True, True)),
    ([("A", "ooo", "ijk"), ("B", "d", "k")], "ij", "dd", (True, True)),
    ([("A", "ooo", "ijk"), ("B", "dd", "kl")], "ijl", "ddd", (True, True)),
    ([("A", "o", "i")], "i", "d", (True, True)),
    ([("A", "o", "i"), ("B", "d", "i")], "i", "d", (True, True)),
    ([("A", "o", "i"), ("B", "dd", "ij")], "j", "d", (True, True)),
    # A product of two sparse operands co-iterates them: an intersection that
    # finds each COO operand's run at the merged coordinate and steps an
    # aligned COO cursor over its whole run.
    ([("A", "o", "i"), ("B", "s", "i")], "i", "d", (True, True)),
    ([("A", "o", "i"), ("B", "o", "i")], "i", "d", (True, True)),
    ([("A", "oo", "ij"), ("B", "ss", "ij")], "ij", "dd", (True, True)),
    ([("A", "ss", "ij"), ("B", "oo", "ij")], "ij", "dd", (True, True)),
    ([("A", "ds", "ij"), ("B", "oo", "ij")], "ij", "dd", (True, True)),
    ([("A", "oo", "ij"), ("B", "oo", "ij")], "ij", "dd", (True, True)),
    ([("A", "oo", "ij"), ("B", "ss", "ij")], "i", "d", (True, True)),
    ([("A", "oo", "ij"), ("B", "ss", "jk")], "ik", "dd", (True, True)),
    ([("A", "ss", "ij"), ("B", "oo", "jk")], "ik", "dd", (True, True)),
    ([("A", "ds", "ij"), ("B", "oo", "jk")], "ik", "dd", (True, True)),
    ([("A", "oo", "ij"), ("B", "oo", "jk")], "ik", "dd", (True, True)),
    ([("A", "ooo", "ijk"), ("B", "ss", "jk")], "i", "d", (True, True)),
    ([("A", "ooo", "ijk"), ("B", "ooo", "ijk")], "ijk", "ddd", (True, True)),
    ([("A", "ooo", "ijk"), ("B", "sss", "ijk")], "ij", "dd", (True, True)),
    ([("A", "oo", "ij"), ("B", "ss", "ij"), ("D", "d", "j")], "i", "d", (True, True)),
    (
        [("A", "oo", "ij"), ("B", "oo", "ij"), ("D", "ss", "ij")],
        "ij",
        "dd",
        (True, True),
    ),
]
READ_IDS = [label(case[:3]) for case in READ]


@pytest.mark.parametrize(
    "operands,result_indices,result_format,shipped", READ, ids=READ_IDS
)
@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("scheduled", [True, False])
def test_a_coo_program_emits_the_shipped_compilers_text(
    operands, result_indices, result_format, shipped, arm, scheduled
):
    kernel, refusal = outcome(operands, result_indices, result_format, arm, scheduled)
    assert kernel is not None, refusal
    assert kernel.conversion_sources == ()
    if not shipped[0 if scheduled else 1]:
        pytest.skip("the shipped compiler has no valid text for this program here")
    result_shape, bindings = shapes(operands, result_indices)
    assert kernel.cpp_source == legacy_generated_cpp(
        program(operands, result_indices, result_format),
        result_shape,
        bindings,
        compile_options=options(arm, scheduled),
    )


def test_every_listed_program_walks_a_coordinate_level():
    for operands, result_indices, result_format, _ in READ:
        kernel, refusal = outcome(operands, result_indices, result_format, 0)
        assert kernel is not None, refusal
        assert "Find iterator end for coordinate level" in kernel.cpp_source or all(
            len(fmt) == 1 for _, fmt, _ in operands if "o" in fmt
        )
        assert "_crd_tensor.size(0)" in kernel.cpp_source


# Extents past the affine tile widths the register-blocking plans carry (8 and
# 32), neither a multiple of one, so a tiled product runs whole tiles and a
# ragged tail.
LARGE_EXTENT = {"i": 37, "j": 41, "k": 70, "l": 19}
TILED = [
    ([("A", "oo", "ij"), ("B", "dd", "jk")], "ik", "dd"),
    ([("A", "ooo", "ijk"), ("B", "dd", "kl")], "ijl", "ddd"),
    ([("A", "oo", "ij"), ("B", "dd", "ik")], "jk", "dd"),
    ([("A", "o", "i"), ("B", "dd", "ij")], "ij", "dd"),
]


def test_the_register_blocking_plans_tile_the_listed_products(monkeypatch):
    for letter, extent in LARGE_EXTENT.items():
        monkeypatch.setitem(EXTENT, letter, extent)
    tiled = 0
    for operands, result_indices, result_format in TILED:
        kernel, refusal = outcome(operands, result_indices, result_format, 1)
        assert kernel is not None, refusal
        tiled += "kTile_" in kernel.cpp_source
    assert tiled, "no listed product is tiled, so the tiled test below tests nothing"


@pytest.mark.parametrize(
    "operands,result_indices,result_format",
    TILED,
    ids=[label(case) for case in TILED],
)
@pytest.mark.parametrize("arm", [0, 1])
def test_coo_products_match_pytorch_past_the_tile_width(
    operands, result_indices, result_format, arm, monkeypatch
):
    for letter, extent in LARGE_EXTENT.items():
        monkeypatch.setitem(EXTENT, letter, extent)
    test_coo_programs_match_pytorch(operands, result_indices, result_format, None, arm)


@pytest.mark.parametrize(
    "operands,result_indices,result_format,shipped", READ, ids=READ_IDS
)
@pytest.mark.parametrize("arm", [0, 1])
def test_coo_programs_match_pytorch(
    operands, result_indices, result_format, shipped, arm
):
    kernel, refusal = outcome(operands, result_indices, result_format, arm)
    assert kernel is not None, refusal
    expression = ",".join(indices for _, _, indices in operands) + "->" + result_indices
    result_shape, _ = shapes(operands, result_indices)
    for kind in INPUTS:
        tensors, references, _ = inputs_of(operands, kind)
        reference = torch.einsum(expression, *references)
        out, executed = execute_cin_via_loopir(
            program(operands, result_indices, result_format),
            result_shape,
            *tensors,
            compile_options=options(arm, True),
        )
        assert executed.cpp_source == kernel.cpp_source
        assert tuple(out.shape) == tuple(result_shape)
        got = out.values.reshape(tuple(result_shape)).to(torch.float64)
        assert torch.allclose(got, reference, rtol=1e-12, atol=1e-12), (
            kind,
            float((got - reference).abs().max()),
        )


# -- sparse results ------------------------------------------------------------

# Programs whose sparse operands are COO tensors, into each listed result
# layout that has a compressed level.
ASSEMBLED: List[Tuple[List[Operand], str, Tuple[str, ...]]] = [
    ([("A", "o", "i")], "i", ("s",)),
    ([("A", "oo", "ij")], "ij", ("ss", "ds", "sd")),
    ([("A", "oo", "ij")], "i", ("s",)),
    ([("A", "oo", "ij")], "j", ("s",)),
    ([("A", "ooo", "ijk")], "ijk", ("sss", "dss", "dds", "sds", "ssd", "dsd")),
    ([("A", "ooo", "ijk")], "ij", ("ss", "ds")),
    ([("A", "ooo", "ijk")], "ik", ("ss", "ds")),
    ([("A", "ooo", "ijk")], "jk", ("ss", "ds")),
    ([("A", "ooo", "ijk")], "k", ("s",)),
    ([("A", "oo", "ij"), ("B", "d", "j")], "i", ("s",)),
    ([("A", "oo", "ij"), ("B", "d", "j")], "ij", ("ss", "ds")),
    ([("A", "oo", "ij"), ("B", "dd", "ij")], "ij", ("ss", "ds")),
    ([("A", "oo", "ij"), ("B", "dd", "jk")], "ik", ("ss", "ds")),
    ([("A", "dd", "ij"), ("B", "oo", "jk")], "ik", ("ss", "ds")),
    ([("A", "oo", "ij"), ("B", "ds", "jk")], "ik", ("ds",)),
    ([("A", "ooo", "ijk"), ("B", "d", "k")], "ij", ("ss", "ds")),
    ([("A", "ooo", "ijk"), ("B", "dd", "kl")], "ijl", ("dss",)),
    # Intersections with a COO operand.
    ([("A", "o", "i"), ("B", "s", "i")], "i", ("s",)),
    ([("A", "oo", "ij"), ("B", "ss", "ij")], "ij", ("ss", "ds")),
    ([("A", "oo", "ij"), ("B", "oo", "ij")], "ij", ("ss", "ds")),
    ([("A", "oo", "ij"), ("B", "ss", "ij")], "i", ("s",)),
    ([("A", "oo", "ij"), ("B", "ss", "jk")], "ik", ("ss", "ds")),
    ([("A", "ds", "ij"), ("B", "oo", "jk")], "ik", ("ds",)),
    ([("A", "oo", "ij"), ("B", "oo", "jk")], "ik", ("ss", "ds")),
    ([("A", "ooo", "ijk"), ("B", "sss", "ijk")], "ijk", ("sss", "dss")),
    # Each row of a CSR operand intersected with a vector that has no row
    # index: the vector's cursor starts from its root on every row.  The
    # second program is the first one's all-compressed twin.
    ([("A", "ds", "ij"), ("B", "o", "j")], "ij", ("ds",)),
    ([("A", "ds", "ij"), ("B", "s", "j")], "ij", ("ds",)),
]
ASSEMBLED_CASES: List[Case] = [
    (operands, result_indices, result_format)
    for operands, result_indices, result_formats in ASSEMBLED
    for result_format in result_formats
]
ASSEMBLED_IDS = [label(case) for case in ASSEMBLED_CASES]


# The all-compressed twins of these two products are lowered by the hash
# workspace family, which reads position arrays and so does not lower a COO
# operand.  The COO program drains the sorted workspace instead, and that
# family keeps one result position counter the shipped text does not: the two
# texts differ by that one increment, which no statement reads.
HASH_WORKSPACE_TWINS = {"oo,ss ij,jk->ik [ss]", "oo,oo ij,jk->ik [ss]"}


def compressed_twin(operands: Sequence[Operand]) -> List[Operand]:
    """The same operands with every COO tensor stored all-compressed."""

    return [(name, fmt.replace("o", "s"), indices) for name, fmt, indices in operands]


def shipped_text(operands, result_indices, result_format, arm) -> Optional[str]:
    result_shape, bindings = shapes(operands, result_indices)
    try:
        return legacy_generated_cpp(
            program(operands, result_indices, result_format),
            result_shape,
            bindings,
            compile_options=options(arm, True),
        )
    except REFUSALS:
        return None


@pytest.mark.parametrize(
    "operands,result_indices,result_format", ASSEMBLED_CASES, ids=ASSEMBLED_IDS
)
@pytest.mark.parametrize("arm", [0, 1])
def test_a_coo_program_matches_the_shipped_text_wherever_its_twin_does(
    operands, result_indices, result_format, arm
):
    kernel, refusal = outcome(operands, result_indices, result_format, arm)
    assert kernel is not None, refusal
    assert kernel.conversion_sources == ()
    twin = compressed_twin(operands)
    twin_kernel, twin_refusal = outcome(twin, result_indices, result_format, arm)
    assert twin_kernel is not None, twin_refusal
    twin_shipped = shipped_text(twin, result_indices, result_format, arm)
    if twin_shipped is None or twin_kernel.cpp_source != twin_shipped:
        pytest.skip("the compressed twin's own text differs from the shipped one")
    shipped = shipped_text(operands, result_indices, result_format, arm)
    if label((operands, result_indices, result_format)) in HASH_WORKSPACE_TWINS:
        assert kernel.cpp_source != shipped
        assert kernel.cpp_source.replace("      pC0++;\n", "", 1) == shipped
        return
    assert kernel.cpp_source == shipped


def test_the_twin_comparison_is_not_vacuous():
    matched = 0
    for operands, result_indices, result_format in ASSEMBLED_CASES:
        twin = compressed_twin(operands)
        twin_kernel, _ = outcome(twin, result_indices, result_format, 0)
        twin_shipped = shipped_text(twin, result_indices, result_format, 0)
        matched += twin_kernel is not None and twin_kernel.cpp_source == twin_shipped
    assert matched >= 8, matched


@pytest.mark.parametrize(
    "operands,result_indices,result_format", ASSEMBLED_CASES, ids=ASSEMBLED_IDS
)
@pytest.mark.parametrize("arm", [0, 1])
def test_coo_programs_with_sparse_results_match_pytorch(
    operands, result_indices, result_format, arm
):
    kernel, refusal = outcome(operands, result_indices, result_format, arm)
    assert kernel is not None, refusal
    expression = ",".join(indices for _, _, indices in operands) + "->" + result_indices
    result_shape, _ = shapes(operands, result_indices)
    for kind in INPUTS:
        tensors, references, indicators = inputs_of(operands, kind)
        reference = torch.einsum(expression, *references)
        structure = torch.einsum(expression, *indicators) > 0
        _, expected_levels = level_arrays(structure, result_format)
        out, executed = execute_cin_via_loopir(
            program(operands, result_indices, result_format),
            result_shape,
            *tensors,
            compile_options=options(arm, True),
        )
        assert executed.cpp_source == kernel.cpp_source
        assert tuple(out.shape) == tuple(result_shape)
        got = result_cells(out, result_format, result_shape)
        assert torch.allclose(got, reference, rtol=1e-12, atol=1e-12), (
            kind,
            float((got - reference).abs().max()),
        )
        actual_levels = [
            (
                None
                if level_kind == "d"
                else tuple(t.reshape(-1).tolist() for t in arrays)
            )
            for level_kind, arrays in zip(result_format, out.storage.index.mode_indices)
        ]
        assert actual_levels == [
            None if level is None else (level[0], level[1]) for level in expected_levels
        ], kind


def test_every_random_input_holds_a_stored_zero():
    for operands, _, _ in [case[:3] for case in READ] + ASSEMBLED_CASES:
        _, references, indicators = inputs_of(operands, "random")
        for (_, fmt, _), reference, indicator in zip(operands, references, indicators):
            if set(fmt) != {"d"}:
                assert bool(((indicator > 0) & (reference == 0)).any())


# -- coordinate results --------------------------------------------------------

# Programs into a COO result: copies, products that keep every index,
# reductions appended in loop order, and reductions and contractions drained
# from the sorted workspace, at the root or once per coordinate of the loops
# above it.  The last field says whether the new compiler
# emits the shipped compiler's text for the program.  Where it does not, the
# shipped compiler either has no text (it refuses a reduction appended in loop
# order into a COO result) or the same program into the all-compressed result
# layout differs from it too, for the same reason: the shipped compiler drains
# a reduction through a temporary, or reserves the result from its shape.
COO_RESULT_TABLE: List[Tuple[List[Operand], str, str, bool]] = [
    ([("A", "o", "i")], "i", "o", True),
    ([("A", "s", "i")], "i", "o", True),
    ([("A", "d", "i")], "i", "o", True),
    ([("A", "oo", "ij")], "ij", "oo", True),
    ([("A", "ss", "ij")], "ij", "oo", True),
    ([("A", "ds", "ij")], "ij", "oo", True),
    ([("A", "sd", "ij")], "ij", "oo", False),
    ([("A", "dd", "ij")], "ij", "oo", False),
    ([("A", "ooo", "ijk")], "ijk", "ooo", True),
    ([("A", "sss", "ijk")], "ijk", "ooo", True),
    ([("A", "dss", "ijk")], "ijk", "ooo", True),
    ([("A", "ss", "ij"), ("B", "d", "j")], "ij", "oo", True),
    ([("A", "oo", "ij"), ("B", "d", "j")], "ij", "oo", True),
    ([("A", "ss", "ij"), ("B", "ss", "ij")], "ij", "oo", True),
    ([("A", "ss", "ij")], "i", "o", False),
    ([("A", "ss", "ij")], "j", "o", False),
    ([("A", "ds", "ij")], "i", "o", False),
    ([("A", "ds", "ij")], "j", "o", False),
    ([("A", "oo", "ij")], "i", "o", False),
    ([("A", "oo", "ij")], "j", "o", False),
    ([("A", "sss", "ijk")], "ij", "oo", False),
    ([("A", "sss", "ijk")], "ik", "oo", True),
    ([("A", "sss", "ijk")], "jk", "oo", False),
    ([("A", "sss", "ijk")], "k", "o", False),
    ([("A", "ooo", "ijk")], "ij", "oo", False),
    ([("A", "ooo", "ijk")], "ik", "oo", True),
    ([("A", "ooo", "ijk")], "jk", "oo", False),
    ([("A", "ooo", "ijk")], "k", "o", False),
    ([("A", "ss", "ij"), ("B", "d", "j")], "i", "o", False),
    ([("A", "oo", "ij"), ("B", "d", "j")], "i", "o", False),
    ([("A", "ss", "ij"), ("B", "dd", "jk")], "ik", "oo", True),
    ([("A", "oo", "ij"), ("B", "dd", "jk")], "ik", "oo", True),
    ([("A", "dd", "ij"), ("B", "ss", "jk")], "ik", "oo", True),
    ([("A", "ss", "ij"), ("B", "ss", "jk")], "ik", "oo", True),
    ([("A", "ds", "ij"), ("B", "ds", "jk")], "ik", "oo", True),
    ([("A", "ss", "ij"), ("B", "dd", "ik"), ("D", "dd", "jk")], "ij", "oo", False),
    ([("A", "oo", "ij"), ("B", "dd", "ik"), ("D", "dd", "jk")], "ij", "oo", False),
    # An index reduced below the result's own levels: one sorted workspace per
    # coordinate of the loops above it, drained into the result entry by
    # entry, each entry storing those loops' coordinates as well as its key.
    ([("A", "ss", "ij"), ("B", "ss", "ik")], "ij", "oo", True),
    ([("A", "oo", "ij"), ("B", "ss", "ik")], "ij", "oo", True),
    ([("A", "oo", "ij"), ("B", "oo", "ik")], "ij", "oo", True),
    ([("A", "oo", "ij"), ("B", "sd", "ik")], "ij", "oo", True),
    ([("A", "oo", "ij"), ("B", "ooo", "ijk")], "ik", "oo", True),
    ([("A", "oo", "ij"), ("B", "sdd", "ijk")], "ik", "oo", True),
    ([("A", "dss", "ijk"), ("B", "sss", "ijl")], "ijk", "ooo", True),
    ([("A", "dsd", "ijk"), ("B", "ooo", "ijl")], "ijk", "ooo", True),
    ([("A", "dsd", "ijk"), ("B", "dsd", "ijl")], "ijk", "ooo", True),
]
COO_RESULTS: List[Case] = [case[:3] for case in COO_RESULT_TABLE]
COO_RESULT_IDS = [label(case) for case in COO_RESULTS]


def compressed_result(result_format: str) -> str:
    """The layout a COO result is assembled as: every level compressed."""

    return "s" * len(result_format)


def loop_program(operands, result_indices, result_format):
    normalized = normalize_cin(
        program(operands, result_indices, result_format),
        compile_options=options(0, False),
    )
    return lower_normalized_cin_to_loopir(normalized).program


def output_of(loopir):
    return next(decl for decl in loopir.tensors if decl.symbol == loopir.outputs[0])


def test_a_coo_result_is_declared_with_coordinate_levels_and_verified():
    loopir = loop_program([("A", "ss", "ij")], "ij", "oo")
    verify_program(loopir)
    assert [level.kind for level in output_of(loopir).levels] == [
        LevelKind.COORDINATE,
        LevelKind.COORDINATE,
    ]
    assert lower_llir._has_coordinate_result(loopir)
    assert not lower_llir._has_coordinate_result(
        loop_program([("A", "oo", "ij")], "ij", "ss")
    )


def test_the_compressed_twin_changes_only_the_results_level_kinds():
    loopir = loop_program([("A", "oo", "ij")], "ij", "oo")
    twin = lower_llir._compressed_result_twin(loopir)
    declared, lowered = output_of(loopir), output_of(twin)
    assert [level.kind for level in lowered.levels] == [LevelKind.COMPRESSED] * 2
    assert [(level.node_id, level.mode) for level in lowered.levels] == [
        (level.node_id, level.mode) for level in declared.levels
    ]
    assert replace(lowered, levels=declared.levels) == declared
    assert twin.body is loopir.body
    for before, after in zip(loopir.tensors, twin.tensors):
        if before.symbol != loopir.outputs[0]:
            # The COO operand keeps its coordinate levels.
            assert after is before


def test_the_verifier_refuses_a_coo_result_stored_in_a_permuted_order():
    loopir = loop_program([("A", "ss", "ij")], "ij", "oo")
    declared = output_of(loopir)
    first, second = declared.levels
    permuted = replace(
        declared,
        levels=(replace(first, mode=second.mode), replace(second, mode=first.mode)),
    )
    swapped = replace(
        loopir,
        tensors=tuple(
            permuted if decl is declared else decl for decl in loopir.tensors
        ),
    )
    assert defect_code(lambda: verify_program(swapped)) == "unsupported_sparse_output"


def entry_appends(coordinates):
    """One appended entry of result ``C``: its value and the named coordinates."""

    return [
        llir.FunctionCallStmt(
            name=f"{array}.emplace_back",
            args=[llir.Var(name=argument, type=llir.DataType.NO_TYPE)],
        )
        for array, argument in [("C_values", "v"), *coordinates]
    ]


def assembly_refusal(statements) -> Optional[str]:
    declared = output_of(loop_program([("A", "ss", "ij")], "ij", "oo"))
    target = SimpleNamespace(result_decl=declared)
    try:
        lower_llir._require_coordinate_result_assembly(target, statements)
    except LoopIRTargetError as error:
        assert error.defect.code == "unsupported_sparse_output"
        return error.defect.message
    return None


WHOLE_ENTRY = [("C1_crd", "j"), ("C0_crd", "i")]


def test_a_whole_entry_of_a_coo_result_is_admitted():
    assert assembly_refusal(entry_appends(WHOLE_ENTRY)) is None
    # Names that only resemble a position array of the result are not one.
    assert (
        assembly_refusal(entry_appends([("C1_crd", "pC1"), ("C0_crd", "A1_pos")]))
        is None
    )


@pytest.mark.parametrize("name", ["C0_pos", "C1_pos.push_back", "C2_pos"])
def test_a_position_array_in_a_coo_results_assembly_fails_closed(name):
    statement = llir.FunctionCallStmt(
        name="scorch_vector_set" if "." not in name else name,
        args=[llir.Var(name=name.split(".")[0], type=llir.DataType.NO_TYPE)],
    )
    message = assembly_refusal([*entry_appends(WHOLE_ENTRY), statement])
    assert message is not None and "position array" in message


@pytest.mark.parametrize("missing", [0, 1])
def test_an_entry_without_one_levels_coordinate_fails_closed(missing):
    """What the shipped compiler emits for a rank-3 COO result drained two loops deep."""

    kept = [pair for pair in WHOLE_ENTRY if pair[0] != f"C{missing}_crd"]
    message = assembly_refusal(entry_appends(kept))
    assert message is not None and f"level {missing}" in message
    # The same omission inside a loop body is found there.
    loop = llir.WhileLoop(
        cond=llir.Var(name="more", type=llir.DataType.BOOL), body=entry_appends(kept)
    )
    nested = assembly_refusal([loop])
    assert nested is not None and f"level {missing}" in nested


def test_an_assembly_that_appends_no_entry_fails_closed():
    message = assembly_refusal([llir.BlankLine()])
    assert message is not None and "appends no entry" in message


@pytest.mark.parametrize(
    "operands,result_indices,result_format", COO_RESULTS, ids=COO_RESULT_IDS
)
@pytest.mark.parametrize("arm", [0, 1])
def test_a_coo_result_stores_no_position_array(
    operands, result_indices, result_format, arm
):
    kernel, refusal = outcome(operands, result_indices, result_format, arm)
    assert kernel is not None, refusal
    assert re.search(r"\bC\d+_pos\b", kernel.cpp_source) is None
    for level in range(len(result_format)):
        assert f"std::vector<int> C{level}_crd;" in kernel.cpp_source


@pytest.mark.parametrize(
    "operands,result_indices,result_format,same_text",
    COO_RESULT_TABLE,
    ids=COO_RESULT_IDS,
)
@pytest.mark.parametrize("arm", [0, 1])
def test_a_coo_result_emits_the_shipped_text_where_the_table_says_so(
    operands, result_indices, result_format, same_text, arm
):
    kernel, refusal = outcome(operands, result_indices, result_format, arm)
    assert kernel is not None, refusal
    shipped = shipped_text(operands, result_indices, result_format, arm)
    assert (shipped is not None and kernel.cpp_source == shipped) == same_text


@pytest.mark.parametrize(
    "operands,result_indices,result_format,same_text",
    COO_RESULT_TABLE,
    ids=COO_RESULT_IDS,
)
@pytest.mark.parametrize("arm", [0, 1])
def test_a_coo_result_emits_the_shipped_text_wherever_its_twin_does(
    operands, result_indices, result_format, same_text, arm
):
    """The twin is the same program into the all-compressed result layout."""

    twin_format = compressed_result(result_format)
    twin_kernel, twin_refusal = outcome(operands, result_indices, twin_format, arm)
    assert twin_kernel is not None, twin_refusal
    twin_shipped = shipped_text(operands, result_indices, twin_format, arm)
    if twin_kernel.cpp_source == twin_shipped:
        assert same_text


def test_the_table_of_coo_results_has_both_kinds_of_program():
    flags = [same_text for _, _, _, same_text in COO_RESULT_TABLE]
    assert flags.count(True) == 28
    assert flags.count(False) == 18


def coordinate_entries(result: STensor, logical_shape):
    """``(tuples, values)`` a COO result stores, read from its own arrays.

    Each level holds one coordinate array as long as the values array.
    """

    assert str(result.format).replace(",", "") == "o" * len(logical_shape)
    assert result.mode_order is None or list(result.mode_order) == sorted(
        result.mode_order
    )
    values = result.storage.value.reshape(-1).to(torch.float64)
    columns = []
    for arrays in result.storage.index.mode_indices:
        assert len(arrays) == 1
        column = arrays[0].reshape(-1).tolist()
        assert len(column) == values.numel()
        columns.append(column)
    assert len(columns) == len(logical_shape)
    return [tuple(entry) for entry in zip(*columns)], values


@pytest.mark.parametrize(
    "operands,result_indices,result_format", COO_RESULTS, ids=COO_RESULT_IDS
)
@pytest.mark.parametrize("arm", [0, 1])
def test_programs_with_coo_results_match_pytorch(
    operands, result_indices, result_format, arm
):
    kernel, refusal = outcome(operands, result_indices, result_format, arm)
    assert kernel is not None, refusal
    expression = ",".join(indices for _, _, indices in operands) + "->" + result_indices
    result_shape, _ = shapes(operands, result_indices)
    for kind in INPUTS:
        tensors, references, indicators = inputs_of(operands, kind)
        reference = torch.einsum(expression, *references)
        structure = torch.einsum(expression, *indicators) > 0
        out, executed = execute_cin_via_loopir(
            program(operands, result_indices, result_format),
            result_shape,
            *tensors,
            compile_options=options(arm, True),
        )
        assert executed.cpp_source == kernel.cpp_source
        assert tuple(out.shape) == tuple(result_shape)
        entries, values = coordinate_entries(out, result_shape)
        # Exactly the cells the operands' stored entries imply, each once, in
        # lexicographic order: a stored zero stays stored.
        assert entries == [tuple(cell) for cell in structure.nonzero().tolist()], kind
        got = torch.zeros(tuple(result_shape), dtype=torch.float64)
        for entry, value in zip(entries, values.tolist()):
            got[entry] = value
        assert torch.allclose(got, reference, rtol=1e-12, atol=1e-12), (
            kind,
            float((got - reference).abs().max()),
        )


def test_every_coo_result_input_holds_a_stored_zero():
    for operands, _, _ in COO_RESULTS:
        _, references, indicators = inputs_of(operands, "random")
        for (_, fmt, _), reference, indicator in zip(operands, references, indicators):
            if set(fmt) != {"d"}:
                assert bool(((indicator > 0) & (reference == 0)).any())


UNSCHEDULED = "LoopIRTargetError/unsupported_program_shape"


@pytest.mark.parametrize(
    "operands,result_indices,result_format",
    [
        ([("A", "ss", "ij")], "j", "o"),
        ([("A", "oo", "ij")], "j", "o"),
        ([("A", "sss", "ijk")], "jk", "oo"),
    ],
)
@pytest.mark.parametrize("arm", [0, 1])
def test_an_unscheduled_reduction_into_a_coo_result_is_refused_as_its_twin_is(
    operands, result_indices, result_format, arm
):
    """Without a schedule no sorted workspace exists for either result layout."""

    twin_format = compressed_result(result_format)
    refused = outcome(operands, result_indices, result_format, arm, scheduled=False)
    twin = outcome(operands, result_indices, twin_format, arm, scheduled=False)
    assert refused == (None, UNSCHEDULED)
    assert twin == (None, UNSCHEDULED)


# -- sums ----------------------------------------------------------------------


def sum_program(operands: Sequence[Operand], result_indices: str, result_format: str):
    """``C[result] (+)= A[...] + B[...]``, both operands over the same indices."""

    (name_a, fmt_a, indices), (name_b, fmt_b, _) = operands
    variables = {letter: IndexVar(letter) for letter in indices}
    index = tuple(variables[letter] for letter in indices)
    total = CINBinaryOp(
        Operation.ADD,
        TensorVar(name_a, fmt=fmt_a, dtype=DTYPE)[index],
        TensorVar(name_b, fmt=fmt_b, dtype=DTYPE)[index],
    )
    result = TensorVar("C", fmt=result_format, dtype=DTYPE)[
        tuple(variables[letter] for letter in result_indices)
    ]
    statement = (
        TensorAssign(result, total)
        if result_indices == indices
        else TensorAssign(result, total, op=Operation.ADD)
    )
    for letter in reversed(indices):
        statement = ForAll(variables[letter], statement)
    return statement


def sum_outcome(operands, result_indices, result_format, arm, scheduled=True):
    result_shape, bindings = shapes(operands, result_indices)
    try:
        return (
            compile_cin_via_loopir(
                sum_program(operands, result_indices, result_format),
                result_shape,
                bindings,
                compile_options=options(arm, scheduled),
            ),
            None,
        )
    except REFUSALS as error:
        defect = getattr(error, "defect", None)
        code = getattr(defect, "code", None) or getattr(error, "code", None)
        return None, f"{type(error).__name__}/{code}"


# A sum of two sparse operands assembled into a COO result: the union of their
# stored cells, each stored once.
COO_SUMS: List[Case] = [
    ([("A", "s", "i"), ("B", "s", "i")], "i", "o"),
    ([("A", "ss", "ij"), ("B", "ss", "ij")], "ij", "oo"),
    ([("A", "sss", "ijk"), ("B", "sss", "ijk")], "ijk", "ooo"),
    ([("A", "ssss", "ijkl"), ("B", "ssss", "ijkl")], "ijkl", "oooo"),
]


@pytest.mark.parametrize(
    "operands,result_indices,result_format",
    COO_SUMS,
    ids=[label(case).replace(",", "+", 1) for case in COO_SUMS],
)
@pytest.mark.parametrize("arm", [0, 1])
def test_sums_into_coo_results_match_pytorch(
    operands, result_indices, result_format, arm
):
    kernel, refusal = sum_outcome(operands, result_indices, result_format, arm)
    assert kernel is not None, refusal
    assert re.search(r"\bC\d+_pos\b", kernel.cpp_source) is None
    result_shape, _ = shapes(operands, result_indices)
    for kind in INPUTS:
        tensors, references, indicators = inputs_of(operands, kind)
        reference = references[0] + references[1]
        structure = (indicators[0] + indicators[1]) > 0
        out, executed = execute_cin_via_loopir(
            sum_program(operands, result_indices, result_format),
            result_shape,
            *tensors,
            compile_options=options(arm, True),
        )
        assert executed.cpp_source == kernel.cpp_source
        entries, values = coordinate_entries(out, result_shape)
        assert entries == [tuple(cell) for cell in structure.nonzero().tolist()], kind
        got = torch.zeros(tuple(result_shape), dtype=torch.float64)
        for entry, value in zip(entries, values.tolist()):
            got[entry] = value
        assert torch.allclose(got, reference, rtol=1e-12, atol=1e-12), kind


@pytest.mark.parametrize(
    "operands,result_indices,result_format",
    [
        ([("A", "oo", "ij"), ("B", "ss", "ij")], "ij", "dd"),
        ([("A", "oo", "ij"), ("B", "dd", "ij")], "ij", "dd"),
        ([("A", "oo", "ij"), ("B", "oo", "ij")], "ij", "oo"),
        ([("A", "oo", "ij"), ("B", "ss", "ij")], "i", "d"),
        ([("A", "ooo", "ijk"), ("B", "sss", "ijk")], "ijk", "sss"),
        ([("A", "ooo", "ijk"), ("B", "ooo", "ijk")], "ijk", "ooo"),
        ([("A", "o", "i"), ("B", "s", "i")], "i", "d"),
    ],
)
@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("scheduled", [True, False])
def test_a_sum_that_reads_a_coo_operand_matches_pytorch(
    operands, result_indices, result_format, arm, scheduled
):
    """A sum co-iterating a COO level: values to 1e-12, stored cells exact.

    Refused with ``unsupported_coordinate_merge`` until 2026-10-12.  A union
    now walks a COO level run by run: an aligned cursor steps over its run,
    an unaligned one's run is empty, and the one-cursor tails and one-sided
    drains scan and step runs too.  ``test_loopir_dense_union.py``'s
    ``COORDINATE_SUM_CELLS`` runs more layouts on both compilers.
    """

    kernel, refusal = sum_outcome(
        operands, result_indices, result_format, arm, scheduled
    )
    assert kernel is not None, refusal
    indices = operands[0][2]
    reduced = tuple(
        mode for mode, letter in enumerate(indices) if letter not in result_indices
    )
    result_shape, _ = shapes(operands, result_indices)
    for kind in INPUTS:
        tensors, references, indicators = inputs_of(operands, kind)
        total = references[0] + references[1]
        reference = total.sum(dim=reduced) if reduced else total
        out, _ = execute_cin_via_loopir(
            sum_program(operands, result_indices, result_format),
            result_shape,
            *tensors,
            compile_options=options(arm, scheduled),
        )
        if set(result_format) == {"o"}:
            entries, values = coordinate_entries(out, result_shape)
            structure = (indicators[0] + indicators[1]) > 0
            assert entries == [
                tuple(cell) for cell in structure.nonzero().tolist()
            ], kind
            got = torch.zeros(tuple(result_shape), dtype=torch.float64)
            for entry, value in zip(entries, values.tolist()):
                got[entry] = value
        else:
            got = out.to_torch().to(torch.float64)
        assert torch.allclose(got, reference, rtol=1e-12, atol=1e-12), kind


def test_the_execution_entry_wraps_a_result_in_every_level_kind_it_can_have():
    """A kernel's result is wrapped in the format its levels are declared with.

    The execution entry spells that format from the level kinds; a kind the
    verifier admits and this table lacks would be wrapped as another kind.
    """

    from scorch.compiler.loopir import pipeline, verifier

    assert set(pipeline._RESULT_LEVEL_LETTERS) == set(verifier._EXECUTABLE_LEVEL_KINDS)
    assert pipeline._RESULT_LEVEL_LETTERS[LevelKind.COORDINATE] == "o"
    assert pipeline._RESULT_LEVEL_LETTERS[LevelKind.COMPRESSED] == "s"
    assert pipeline._RESULT_LEVEL_LETTERS[LevelKind.DENSE] == "d"


def operand_position_refusal(statements) -> Optional[str]:
    """The refusal of a nest over a rank-2 COO operand ``A`` and a CSR operand ``B``."""

    loopir = loop_program([("A", "oo", "ij"), ("B", "ds", "jk")], "ik", "dd")
    decls = {decl.symbol: decl for decl in loopir.tensors}
    target = SimpleNamespace(decls=decls, _input_symbols=tuple(loopir.inputs))
    try:
        lower_llir._require_no_coordinate_operand_positions(target, statements)
    except LoopIRTargetError as error:
        assert error.defect.code == "unsupported_level_kind"
        return error.defect.message
    return None


def named(name: str):
    return llir.VarInit(
        var=llir.Var(name="x", type=llir.DataType.INT),
        value=llir.ArrayAccess(
            array=llir.Var(name=name, type=llir.DataType.NO_TYPE),
            index=llir.Var(name="p", type=llir.DataType.INT),
        ),
    )


@pytest.mark.parametrize("name", ["A0_pos", "A1_pos", "A2_pos"])
def test_a_position_array_of_a_coo_operand_in_the_nest_fails_closed(name):
    message = operand_position_refusal([named(name)])
    assert message is not None and name in message
    # Inside a loop body too.
    loop = llir.WhileLoop(
        cond=llir.Var(name="more", type=llir.DataType.BOOL), body=[named(name)]
    )
    assert operand_position_refusal([loop]) is not None


@pytest.mark.parametrize("name", ["A0_crd", "A1_crd", "B1_pos", "B1_crd", "A10_pos"])
def test_other_arrays_are_not_a_coo_operands_position_arrays(name):
    """The CSR operand's own position array is read by every nest over it."""

    assert operand_position_refusal([named(name)]) is None


def test_no_listed_program_reads_a_position_array_of_a_coo_operand():
    for operands, result_indices, result_format in (
        [case[:3] for case in READ] + ASSEMBLED_CASES + COO_RESULTS
    ):
        kernel, refusal = outcome(operands, result_indices, result_format, 0)
        assert kernel is not None, refusal
        for name, fmt, _ in operands:
            if set(fmt) == {"o"}:
                assert re.search(rf"\b{name}\d+_pos\b", kernel.cpp_source) is None
