"""The SpGEMM work estimate reads every position array inside its length.

The two-phase compressed-``Where`` kernels size their team, their per-thread workspace
count and their chunk from a work estimate, ``A_nnz * (B_nnz / B0_size + 1)``.  It read
each operand's stored count as ``X{L}_pos[X0_size]``, with ``L`` the operand's deepest
compressed level.  That is the stored count only for two-level CSR.  With compressed
levels ``1..L`` and ``L > 1``, level ``L``'s position array has one entry per stored
position of level ``L - 1``, plus one, so the read indexes it by the level-0 extent:
past its end when fewer positions are stored above it, and a wrong count otherwise
(dennis, region scan 82815: ``ijkl,ijkl->il`` with ``A`` in ``dssd`` read ``A2_pos[4]``
of a three-entry array).  The team, the workspace count and the chunk each make their
own read, so a read past the end can give them different values.

The estimate now follows the chain ``X{L}_pos[...X1_pos[X0_size]]``, which is the same
text for two-level CSR, and gives up (the caller takes the guarded ``A_nnz`` estimate)
when the compressed levels are not exactly ``1..L``.
"""

import re
from typing import List, Optional, Sequence, Tuple, cast

import pytest
import torch

from scorch.compiler import llir
from scorch.compiler.compressed_where_openmp_pass import (
    _spgemm_flop_work_estimate,
    _stored_count_chain,
)
from scorch.compiler.llir_traversal import LLIRTraversalContext
from tests.test_scorch import test_high_rank_programs as high_rank
from tests.test_scorch.level_array_support import emitted_einsum_cpp, level_arrays

TRAVERSAL = LLIRTraversalContext(stage="test", pass_name="spgemm_work_estimate")
BOUND = llir.Var(name="A0_size", type=llir.DataType.INT)


def _var(name: str, data_type: llir.DataType) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _end(operand: str, level: int) -> llir.VarInit:
    """``int pX{level}_end = X{level}_pos[pX{level - 1} + 1];``"""

    return llir.VarInit(
        _var(f"p{operand}{level}_end", llir.DataType.INT),
        llir.ArrayAccess(
            _var(f"{operand}{level}_pos", llir.DataType.PTR_INT),
            llir.Add(
                _var(f"p{operand}{level - 1}", llir.DataType.INT),
                llir.Literal(1, llir.DataType.INT),
            ),
        ),
    )


def _body(left: Sequence[int], right: Sequence[int]) -> List[llir.Stmt]:
    return [_end("A", level) for level in left] + [_end("B", level) for level in right]


def _chain(expr: llir.Expr) -> Tuple[List[str], str]:
    """The position arrays an estimate operand reads, outermost first, and the
    variable at the bottom of the chain."""

    names = []
    while type(expr) is llir.ArrayAccess:
        access = cast(llir.ArrayAccess, expr)
        names.append(cast(llir.Var, access.array).name)
        expr = access.index
    return names, cast(llir.Var, expr).name


def _operands(estimate) -> Tuple[llir.Expr, llir.Expr]:
    product = cast(llir.Mul, estimate.expr)
    left = cast(llir.Cast, product.left).expr
    fanout = cast(llir.Select, product.right).when_true
    right = cast(llir.BinOp, cast(llir.Add, fanout).left).left
    return left, right


def test_two_level_csr_keeps_its_text():
    estimate = _spgemm_flop_work_estimate(_body([1], [1]), BOUND, TRAVERSAL)
    assert estimate is not None
    assert estimate.text == (
        "(long)A1_pos[A0_size] * (B0_size > 0 ? B1_pos[B0_size] / B0_size + 1 : 1)"
    )
    left, right = _operands(estimate)
    assert _chain(left) == (["A1_pos"], "A0_size")
    assert _chain(right) == (["B1_pos"], "B0_size")


@pytest.mark.parametrize(
    ("left", "right"),
    [([1, 2], [1]), ([1], [1, 2]), ([1, 2], [1, 2]), ([1, 2, 3], [1, 2, 3])],
)
def test_a_deeper_operand_reads_its_count_through_every_level(left, right):
    estimate = _spgemm_flop_work_estimate(_body(left, right), BOUND, TRAVERSAL)
    assert estimate is not None
    left_names = [f"A{level}_pos" for level in reversed(left)]
    right_names = [f"B{level}_pos" for level in reversed(right)]
    left_expr, right_expr = _operands(estimate)
    assert _chain(left_expr) == (left_names, "A0_size")
    assert _chain(right_expr) == (right_names, "B0_size")
    left_text = "".join(f"{name}[" for name in left_names) + "A0_size" + "]" * len(left)
    right_text = (
        "".join(f"{name}[" for name in right_names) + "B0_size" + "]" * len(right)
    )
    assert estimate.text == (
        f"(long){left_text} * (B0_size > 0 ? {right_text} / B0_size + 1 : 1)"
    )
    for level in range(2, 4):
        assert f"A{level}_pos[A0_size]" not in estimate.text
        assert f"B{level}_pos[B0_size]" not in estimate.text


@pytest.mark.parametrize(
    ("left", "right"), [([2], [1]), ([1, 3], [1]), ([1], [2]), ([1], [1, 3])]
)
def test_a_dense_level_between_compressed_ones_takes_the_guarded_estimate(left, right):
    assert _spgemm_flop_work_estimate(_body(left, right), BOUND, TRAVERSAL) is None


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        ({1}, "X1_pos[X0_size]"),
        ({1, 2}, "X2_pos[X1_pos[X0_size]]"),
        ({1, 2, 3}, "X3_pos[X2_pos[X1_pos[X0_size]]]"),
        ({2}, None),
        ({1, 3}, None),
    ],
)
def test_the_chain_of_each_level_set(levels, expected: Optional[str]):
    chain = _stored_count_chain("X", levels, _var("X0_size", llir.DataType.INT64))
    assert (None if chain is None else chain[0]) == expected


@pytest.mark.parametrize(
    ("fmt", "old_read_past_the_end"),
    [("dssd", True), ("dss", True), ("dsss", False)],
)
def test_the_chain_is_the_stored_count_on_a_gap_input(fmt, old_read_past_the_end):
    """A gap input like dennis's: 3 stored (i, j) pairs, fewer than i's extent plus
    one.  The chain reads every array in bounds and gives the deepest compressed level's
    stored count.  The old read, that level's array at index 4, is past its end or a
    wrong count."""

    shape = (4, 3, 2, 2)[: len(fmt)]
    dense = torch.zeros(shape, dtype=torch.float64)
    dense[1, 2] = 1.0
    dense[3, 0] = 2.0
    dense[3, 1] = 3.0
    mode_indices, _ = level_arrays(dense, fmt)
    deepest = max(level for level, letter in enumerate(fmt) if letter == "s")
    count = shape[0]
    for level in range(1, deepest + 1):
        positions = mode_indices[level][0]
        assert 0 <= count < positions.numel()
        count = int(positions[count])
    assert count == mode_indices[deepest][1].numel()
    deepest_positions = mode_indices[deepest][0]
    assert (shape[0] >= deepest_positions.numel()) == old_read_past_the_end
    if not old_read_past_the_end:
        assert int(deepest_positions[shape[0]]) != count


# dennis's examples from region scan 82815 (job 82954), one per pattern, with the chains
# their estimates read.  An empty tuple is a program whose levels have no chain, which
# takes the guarded estimate.  Three are places in test_high_rank_programs' population,
# which seeds a program's inputs by its place.  3530 and 3620 are the scan's "wide"
# family (merge_class/vmscan's expressions over every layout, extents a=3 b=9 c=4 d=5),
# seeded by their number in the scan (dennis, job 83089).
WIDE_EXTENTS = (("a", 3), ("b", 9), ("c", 4), ("d", 5))
HIGH_RANK = [
    (
        228,
        "population",
        ("ijkl,ijkl->il", ("dssd", "dsdd"), "ds", None),
        ("A2_pos[A1_pos[A0_size]]",),
    ),
    (
        3620,
        "wide",
        ("abc,cd->abd", ("dss", "ds"), "dss", WIDE_EXTENTS),
        ("A2_pos[A1_pos[A0_size]]",),
    ),
    (3530, "wide", ("abc,cd->abd", ("dds", "ds"), "dss", WIDE_EXTENTS), ()),
    (
        1004,
        "population",
        ("bijk,bjk->bi", ("dsss", "dss"), "ds", None),
        ("A3_pos[A2_pos[A1_pos[A0_size]]]", "B2_pos[B1_pos[B0_size]]"),
    ),
    (187, "population", ("ijkl,jklm->im", ("ddss", "dsds"), "ds", None), ()),
]
HIGH_RANK_IDS = [str(program[0]) for program in HIGH_RANK]
# A deeper level's position array indexed by the level-0 extent: the old read.
OLD_READ = re.compile(r"\b(?P<tensor>[A-Z])([2-9])_pos\[(?P=tensor)0_size\]")


def _high_rank_program(monkeypatch, number, family, spec):
    expr, formats, result, extents = spec
    if family == "population":
        program = high_rank._POPULATION[number]
        assert (program.expr, program.operand_formats, program.result_format) == (
            expr,
            formats,
            result,
        ), program.name
        return program
    program = high_rank.Program(
        family=family,
        expr=expr,
        operand_formats=formats,
        result_format=result,
        extents=extents,
    )
    # Seeded by its number in the scan, as the harness seeds a population program by
    # its place.
    monkeypatch.setitem(high_rank._POPULATION_INDEX, program.name, number)
    return program


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize(
    ("number", "family", "spec", "chains"), HIGH_RANK, ids=HIGH_RANK_IDS
)
def test_the_programs_that_read_past_the_end_emit_the_chain(
    monkeypatch, blocking, number, family, spec, chains
):
    program = _high_rank_program(monkeypatch, number, family, spec)
    cpp = emitted_einsum_cpp(
        monkeypatch,
        program.expr,
        program.operand_formats,
        dict(program.extents),
        format=program.result_format,
        _compile_options=high_rank._options(blocking),
    )
    assert OLD_READ.search(cpp) is None, OLD_READ.search(cpp)
    for chain in chains:
        # Inside the team size and the chunk, the two clauses the estimate sizes.
        for clause in ("scorch_nthreads", "scorch_chunk"):
            sized = re.search(rf"{clause}\([^;]*{re.escape(chain)}", cpp)
            assert sized is not None, (clause, chain)
    if not chains:
        # The guarded estimate: no product of an operand count and a mean fanout.
        assert re.search(r"0_size \+ 1 : 1\)", cpp) is None


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize(
    ("number", "family", "spec", "chains"), HIGH_RANK, ids=HIGH_RANK_IDS
)
def test_the_programs_that_read_past_the_end_still_match_pytorch(
    monkeypatch, blocking, number, family, spec, chains
):
    program = _high_rank_program(monkeypatch, number, family, spec)
    outcome = high_rank._outcome(
        high_rank._run_shipped, program, blocking, high_rank._SHIPPED_REFUSALS
    )
    assert outcome == "correct", outcome
