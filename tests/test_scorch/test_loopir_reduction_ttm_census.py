"""Phase-7 cluster 2: the multi-compressed reduction / TTM frontier census.

Cluster 2 -- sparse reductions and TTM over multi-compressed receivers -- is
now **partly migrated**.  This module pins the representative matrix in both
automatic arms so the split between what the ordered-key sparse-workspace
vertical reaches and what it still does not is a locked fact rather than a
re-derived one.  It remains a deliberately finite frontier, not a claim to
enumerate every rank/layout combination: a level-general audit finds adjacent
``dss``/``sds`` reductions and mixed ``ds``/``sd`` factor variants that this
list does not name, and a migration must derive those itself.

The cells split three ways.

**Migrated (9 cells).**  Their automatic plan order is legal and the
ordered-key workspace target lowers them end to end, in both arms.  They span
rank-1, rank-2 and rank-3 keys, root-anchored and prefix-anchored regions,
single-cursor and merged producers, and dense and compressed second factors.
Their storage/oracle/PyTorch differentials live in
``test_loopir_ordered_key_workspace_target.py``; what is locked HERE is only
that they compile arm-invariantly, so a regression that un-migrates one is
caught by the census as well as by its own suite.

**Auto-tile blocked (2 cells).**  ``TTM dds x {dd,ss} -> dds`` is legal and
its target shape is supported, but the automatic origin also emits an affine
tile for it: ``Scheduler._select_index_vars_to_tile`` tiles every dense index
variable that does not appear in every access, and ``j`` qualifies for a
``dds`` receiver against a ``kl`` second factor.  A plan carrying BOTH a
sparse workspace and a tile has no replay contract -- the only implemented
workspace+tile composition is the dense reduce-out fusion -- so it stops at
``unsupported_schedule_auto_family``.  This is a schedule-composition
blocker, not a target or representation one.

**Auto-origin reorder-blocked (5 cells).**  ``Scheduler.select_loop_order``
ends with a forced reorder (``scheduler.py``, "ensure at least one free
variable appears after the last reduction variable"): when no free variable
follows the last reduction it moves the last free variable to the very end of
the loop order, with no legality check.  For these cells that permutation
violates the sparse operand's own parent dominance -- ``sss ijk->ij`` declared
``i,j,k`` becomes ``i,k,j`` while ``A``'s storage order is ``i,j,k`` -- so the
LoopIR route rejects the plan at ``loop_plan_legality``'s
``sparse_parent_dominance`` before any LoopIR admission decision is reached.

The explicit-order controls below sharpen what a LoopIR-only automatic-plan
repair would and would not buy.  A legal declared order does carry these cells
past LoopPlan -- so the block really is origin-specific -- but the program it
produces has an EMPTY workspace key: every result coordinate is bound above
the outermost reduction, leaving nothing to drain.  That ``K == 0`` shape is a
scalar-accumulator reduction, which no migrated family owns.  Repairing the
origin alone therefore moves the failure from LoopPlan to a later LoopIR seam
without migrating anything; the two must be decided together.
"""

import pytest
import torch

from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loop_plan_legality import InvalidSchedule
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir
from scorch.compiler.loopir.schedule_passes import SchedulePassError
from scorch.compiler.scheduler import Schedule
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options


def reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices):
    """``C[result] += A[operand]`` over one shared index nest."""

    order = operand_indices
    ivars = {name: IndexVar(name) for name in order}
    operand = TensorVar("A", fmt=operand_fmt, dtype=torch.float32)[
        tuple(ivars[name] for name in operand_indices)
    ]
    result = TensorVar("C", fmt=result_fmt, dtype=torch.float32)[
        tuple(ivars[name] for name in result_indices)
    ]
    stmt = TensorAssign(result, operand, op=Operation.ADD)
    for name in reversed(order):
        stmt = ForAll(ivars[name], stmt)
    return stmt


def ttm_cin(a_fmt, b_fmt, c_fmt):
    """``C[i,j,l] += A[i,j,k] * B[k,l]`` -- tensor times matrix."""

    ivars = {name: IndexVar(name) for name in "ijkl"}
    a = TensorVar("A", fmt=a_fmt, dtype=torch.float32)[
        ivars["i"], ivars["j"], ivars["k"]
    ]
    b = TensorVar("B", fmt=b_fmt, dtype=torch.float32)[ivars["k"], ivars["l"]]
    c = TensorVar("C", fmt=c_fmt, dtype=torch.float32)[
        ivars["i"], ivars["j"], ivars["l"]
    ]
    stmt = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for name in reversed("ijkl"):
        stmt = ForAll(ivars[name], stmt)
    return stmt


_SHAPE_3 = (4, 5, 6)
_SHAPE_2 = (4, 5)
_F32 = torch.float32


def _reduction_cell(name, operand_fmt, result_fmt, operand_indices, result_indices):
    shape = _SHAPE_3 if len(operand_indices) == 3 else _SHAPE_2
    result_shape = tuple(shape[operand_indices.index(c)] for c in result_indices)
    return (
        name,
        reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
        result_shape,
        ((shape, _F32),),
    )


def _ttm_cell(name, a_fmt, b_fmt, c_fmt):
    return (
        name,
        ttm_cin(a_fmt, b_fmt, c_fmt),
        (4, 5, 3),
        ((_SHAPE_3, _F32), ((6, 3), _F32)),
    )


# The plan order these cells receive violates the operand's parent dominance,
# so they never reach a LoopIR admission decision.  ``InvalidSchedule`` carries
# no ``defect.code``; the stage and reason are checked from its message.
REORDER_BLOCKED = [
    _reduction_cell("sss ijk->i", "sss", "s", "ijk", "i"),
    _reduction_cell("sss ijk->j", "sss", "s", "ijk", "j"),
    _reduction_cell("sss ijk->ij", "sss", "ss", "ijk", "ij"),
    _reduction_cell("ss ij->i", "ss", "s", "ij", "i"),
    _reduction_cell("ds ij->i", "ds", "s", "ij", "i"),
]

# Legal plan order, supported target shape: the ordered-key vertical lowers
# these end to end.  The ``(prefix, key rank)`` split each one exercises is
# recorded beside it because that -- not the layout spelling -- is what the
# migration is general over.
MIGRATED = [
    (*_reduction_cell("sss ijk->k", "sss", "s", "ijk", "k"), 0, 1),
    (*_reduction_cell("sss ijk->ik", "sss", "ss", "ijk", "ik"), 1, 1),
    (*_reduction_cell("sss ijk->jk", "sss", "ss", "ijk", "jk"), 0, 2),
    (*_reduction_cell("ss ij->j", "ss", "s", "ij", "j"), 0, 1),
    (*_reduction_cell("ds ij->j", "ds", "s", "ij", "j"), 0, 1),
    (*_ttm_cell("TTM sss x dd -> sss", "sss", "dd", "sss"), 2, 1),
    (*_ttm_cell("TTM sss x ss -> sss", "sss", "ss", "sss"), 2, 1),
    (*_ttm_cell("TTM dss x dd -> dss", "dss", "dd", "dss"), 2, 1),
    (*_ttm_cell("TTM dss x ss -> dss", "dss", "ss", "dss"), 2, 1),
]

# Legal plan order and a supported target shape, but the automatic origin
# also emits an affine tile, and no workspace+tile replay contract exists.
AUTO_TILE_BLOCKED = [
    (
        *_ttm_cell("TTM dds x dd -> dds", "dds", "dd", "dds"),
        "unsupported_schedule_auto_family",
    ),
    (
        *_ttm_cell("TTM dds x ss -> dds", "dds", "ss", "dds"),
        "unsupported_schedule_auto_family",
    ),
]


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    "cell", REORDER_BLOCKED, ids=[cell[0] for cell in REORDER_BLOCKED]
)
def test_reorder_blocked_cells_stop_at_parent_dominance(cell, arm):
    """The shared automatic origin's forced reorder makes the plan illegal.

    These stop before LoopIR gets a say on this origin.  The explicit-schedule
    control below proves that this is not an intrinsic program boundary -- and
    also that repairing the origin alone would not migrate them.
    """

    name, cin, result_shape, bindings = cell
    with pytest.raises(InvalidSchedule) as error:
        compile_cin_via_loopir(
            cin, result_shape, bindings, compile_options=auto_options(arm)
        )
    message = str(error.value)
    assert "stage=LoopPlan" in message, name
    assert "sparse_parent_dominance" in message, name


@pytest.mark.parametrize(
    ("cell", "loop_order", "exception", "expected_code"),
    [
        (
            _reduction_cell("sss ijk->ij", "sss", "ss", "ijk", "ij"),
            ("i", "j", "k"),
            LoopIRTargetError,
            "unsupported_program_shape",
        ),
        (
            _reduction_cell("ss ij->i", "ss", "s", "ij", "i"),
            ("i", "j"),
            LoopIRLoweringError,
            "unsupported_sparse_output_domain",
        ),
    ],
    ids=("sss-legal-explicit-order", "ss-legal-explicit-order"),
)
def test_auto_reorder_block_is_origin_specific_but_not_sufficient(
    cell, loop_order, exception, expected_code
):
    """A legal explicit order reaches a later LoopIR seam -- and stops there.

    Both controls bind every result coordinate ABOVE the outermost reduction,
    so the ordered workspace key is empty.  That is the exact reason a
    LoopIR-only automatic-plan repair is necessary but not sufficient for
    these cells: the legal program it would produce is a ``K == 0`` scalar
    accumulation that no migrated family owns.
    """

    name, cin, result_shape, bindings = cell
    options = CompileOptions.from_environment(
        environ={}, requested_schedule=Schedule(loop_order=loop_order)
    )
    with pytest.raises(exception) as error:
        compile_cin_via_loopir(cin, result_shape, bindings, compile_options=options)
    assert error.value.defect.code == expected_code, (name, error.value.defect)


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize("cell", MIGRATED, ids=[cell[0] for cell in MIGRATED])
def test_migrated_cells_compile_arm_invariantly(cell, arm):
    """Every migrated cell lowers to a complete kernel in both arms."""

    name, cin, result_shape, bindings, prefix, key_rank = cell
    kernel = compile_cin_via_loopir(
        cin, result_shape, bindings, compile_options=auto_options(arm)
    )
    assert "wksp.sort();" in kernel.cpp_source, name
    result_decl = next(
        decl
        for decl in kernel.lowering.program.tensors
        if decl.symbol == kernel.lowering.result_symbol
    )
    assert len(result_decl.levels) == prefix + key_rank, name


@pytest.mark.parametrize("cell", MIGRATED, ids=[cell[0] for cell in MIGRATED])
def test_migrated_cells_are_arm_source_identical(cell):
    name, cin, result_shape, bindings, _, _ = cell
    sources = {
        arm: compile_cin_via_loopir(
            cin, result_shape, bindings, compile_options=auto_options(arm)
        ).cpp_source
        for arm in (False, True)
    }
    assert sources[False] == sources[True], name


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    "cell", AUTO_TILE_BLOCKED, ids=[cell[0] for cell in AUTO_TILE_BLOCKED]
)
def test_auto_tile_blocked_cells_keep_their_schedule_code(cell, arm):
    """A workspace plan that also carries a tile has no replay contract.

    If a workspace+tile composition is ever implemented, this lock moves to
    whatever still occupies the seam rather than being deleted.
    """

    name, cin, result_shape, bindings, expected_code = cell
    with pytest.raises(SchedulePassError) as error:
        compile_cin_via_loopir(
            cin, result_shape, bindings, compile_options=auto_options(arm)
        )
    assert error.value.defect.code == expected_code, (name, error.value.defect)


def test_census_covers_the_declared_representative_matrix():
    """Pin the finite review matrix without claiming layout exhaustiveness."""

    assert len(REORDER_BLOCKED) == 5
    assert len(MIGRATED) == 9
    assert len(AUTO_TILE_BLOCKED) == 2
    names = (
        [cell[0] for cell in REORDER_BLOCKED]
        + [cell[0] for cell in MIGRATED]
        + [cell[0] for cell in AUTO_TILE_BLOCKED]
    )
    assert len(names) == len(set(names)) == 16
    # All six canonical TTM layouts named by the inherited review are present.
    ttm = [name for name in names if name.startswith("TTM ")]
    assert len(ttm) == 6
    assert len({name.split("->")[1].strip() for name in ttm}) == 3


def test_migrated_cells_span_every_key_rank_and_anchor():
    """The migration is general over the split, not over a layout list."""

    splits = {(cell[4], cell[5]) for cell in MIGRATED}
    assert {prefix for prefix, _ in splits} == {0, 1, 2}
    assert {key_rank for _, key_rank in splits} == {1, 2}
