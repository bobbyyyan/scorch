"""Phase-7 cluster 2: the multi-compressed reduction / TTM frontier census.

Cluster 2 -- sparse reductions and TTM over multi-compressed receivers -- is
migrated for every cell of the representative matrix below.  This module pins
that matrix in both automatic arms so that which family or plan decision
carries each cell is a locked fact rather than a re-derived one.  It remains a
deliberately finite frontier, not a claim to enumerate every rank/layout
combination: a level-general audit finds adjacent ``dss``/``sds`` reductions
and mixed ``ds``/``sd`` factor variants that this list does not name, and a
migration must derive those itself.

The cells split three ways; none of them is refused any more.

**Migrated (11 cells).**  Their automatic plan order is legal and the
ordered-key workspace target lowers them end to end, in both arms.  They span
rank-1, rank-2 and rank-3 keys, root-anchored and prefix-anchored regions,
single-cursor and merged producers, and dense and compressed second factors.
Their storage/oracle/PyTorch differentials live in
``test_loopir_ordered_key_workspace_target.py``; what is locked HERE is only
that they compile arm-invariantly, so a regression that un-migrates one is
caught by the census as well as by its own suite.  Two of them,
``TTM dds x {dd,ss} -> dds``, used to stop at
``unsupported_schedule_auto_family``: the automatic origin emitted an affine
tile over their non-dense receiver beside the sparse workspace, and it no
longer does (``AUTO_TILE_BLOCKED`` is empty and says why).

**Formerly reorder-blocked (5 cells).**  ``Scheduler.select_loop_order`` ends
with a forced reorder (``scheduler.py``, "ensure at least one free variable
appears after the last reduction variable"): when no free variable follows the
last reduction it moves the last free variable to the very end of the loop
order, with no legality check.  For ``sss ijk->{i,j,ij}``, ``ss ij->i`` and
``ds ij->i`` that permutation violates the sparse operand's own parent
dominance, and the plan used to be rejected at ``loop_plan_legality``'s
``sparse_parent_dominance``.  The automatic plan origin now repairs such an
order, and ``FORMER_REORDER_BLOCKED`` records the family each cell reaches.

**Permuted result (4 cells).**  ``ss ij->ji``, ``ds ij->ji``, ``sss ijk->ikj``
and ``sss ijk->ki`` store the result in an order the operand is not stored in,
so no loop order follows both tensors: the plan stopped at
``sparse_parent_dominance``, and the order the composition started from was
refused by ``result_storage_order`` too.  The plan now converts the operand to
the result's order (``LoopPlan.conversions``,
``test_loopir_operand_conversions.py``), and the main kernel reads the
converted operand in storage order.  The shipped compiler's plan origin has no
conversion and still stops at ``sparse_parent_dominance``.  Both facts are
locked below; the four run against dense PyTorch in
``test_loopir_ordered_key_workspace_target.py``, as the migrated cells do.
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
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    legacy_generated_cpp,
)
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


# No cell of this matrix stops at the plan's legality check any more.  The list
# is kept, empty, because the census counts it.  It first named
# ``sss ijk->{i,j,ij}``, ``ss ij->i`` and ``ds ij->i``, which moved when the
# bound-prefix family landed (``FORMER_REORDER_BLOCKED`` below records where
# each went, and ``test_loopir_bound_prefix_target`` owns the family), and then
# the four cells of ``PERMUTED_RESULT``, which moved when the plan could
# convert an operand.
REORDER_BLOCKED: list = []

# The result is stored in an order the operand is not stored in.  Each entry is
# the cell as ``_reduction_cell`` takes it (name, operand format, result format,
# operand indices, result indices) and the conversion the plan records for it:
# the mode order the operand is converted to and the layout it is converted into.
PERMUTED_RESULT = [
    (("ss ij->ji", "ss", "ss", "ij", "ji"), ((1, 0), "ss")),
    (("ds ij->ji", "ds", "ds", "ij", "ji"), ((1, 0), "ds")),
    (("sss ijk->ikj", "sss", "sss", "ijk", "ikj"), ((0, 2, 1), "sss")),
    (("sss ijk->ki", "sss", "ds", "ijk", "ki"), ((2, 1, 0), "sss")),
]

# Where the five cells this census used to lock as reorder-blocked now go, by
# family.  All five are admitted since 2026-10-11: ``ds ij->i`` (a compressed
# result level driven by a dense domain) and ``sss ijk->ij`` (a doubly-compressed
# receiver with no drained key) joined the bound-prefix family, and
# ``sss ijk->j`` -- whose result coordinate sits between two reductions -- is a
# rank-1 key of the ordered-key workspace region that starts at the outermost
# reduction.
FORMER_REORDER_BLOCKED = [
    (_reduction_cell("sss ijk->i", "sss", "s", "ijk", "i"), "bound_prefix"),
    (_reduction_cell("ss ij->i", "ss", "s", "ij", "i"), "bound_prefix"),
    (_reduction_cell("ds ij->i", "ds", "s", "ij", "i"), "bound_prefix"),
    (_reduction_cell("sss ijk->j", "sss", "s", "ijk", "j"), "ordered_key"),
    (_reduction_cell("sss ijk->ij", "sss", "ss", "ijk", "ij"), "bound_prefix"),
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
    # Moved here from AUTO_TILE_BLOCKED when the automatic origin stopped
    # emitting an affine tile over a non-dense receiver.  The tile was never
    # legal: it hoists the tile loop above the loops indexing the receiver's
    # dense prefix, so a ``dds`` receiver's compressed level no longer sees its
    # ``(i, j)`` parent prefix in lexicographic order.  Measured before the
    # change, legacy's public route raised ``TensorIndexError`` on these at every
    # extent, dtype, density and arm, and legacy's own lowering SEGFAULTED.
    (*_ttm_cell("TTM dds x dd -> dds", "dds", "dd", "dds"), 2, 1),
    (*_ttm_cell("TTM dds x ss -> dds", "dds", "ss", "dds"), 2, 1),
]

# The workspace+tile seam is EMPTY for this census's matrix.  It was two cells
# (``TTM dds x {dd,ss} -> dds``), and they are in MIGRATED above because the
# composition that blocked them -- a plan carrying both a sparse workspace and an
# affine tile -- is no longer constructed for a non-dense receiver.
#
# The seam itself is not gone.  Over the 1139-cell frontier it is still occupied,
# by ``ddd ijk->k [d]`` and ``ddd ijk->ik [dd]`` in arm 0 -- both DENSE receivers,
# where the legality rule is inert by construction and the tile is legitimate.
# Those two live outside this file's TTM matrix; the lock for them is
# ``test_dense_receivers_keep_their_automatic_tile`` below.
AUTO_TILE_BLOCKED: list = []


_PERMUTED_RESULT_IDS = [spec[0] for spec, _ in PERMUTED_RESULT]


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    ("spec", "conversion"), PERMUTED_RESULT, ids=_PERMUTED_RESULT_IDS
)
def test_permuted_result_cells_convert_the_operand(spec, conversion, arm):
    """The plan converts the operand to the result's order and the cell compiles.

    These four stopped at ``sparse_parent_dominance`` before LoopIR got a say:
    the result is permuted, so the order the composition started from was
    refused by ``result_storage_order`` too and there was no legal order to
    fall back to.  One conversion of the operand makes the result's order legal.
    """

    name, cin, result_shape, bindings = _reduction_cell(*spec)
    kernel = compile_cin_via_loopir(
        cin, result_shape, bindings, compile_options=auto_options(arm)
    )
    assert [(c.mode_order, c.format) for c in kernel.conversions] == [conversion], name
    assert len(kernel.conversion_sources) == 1, name
    # ``sss ijk->ki`` also reduces ``j`` between the result's two coordinates:
    # the ordered-key workspace region drains its key sorted.  The other three
    # are copies and reduce nothing.
    assert ("wksp.sort();" in kernel.cpp_source) == (name == "sss ijk->ki"), name


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    ("spec", "conversion"), PERMUTED_RESULT, ids=_PERMUTED_RESULT_IDS
)
def test_the_shipped_plan_origin_still_refuses_a_permuted_result(spec, conversion, arm):
    """The shipped compiler has no conversion: its plan stays illegal.

    So these four have no shipped text to compare the new compiler's with.
    ``InvalidSchedule`` carries no ``defect.code``; the stage and reason are
    read from its structured diagnostic.
    """

    name, cin, result_shape, bindings = _reduction_cell(*spec)
    with pytest.raises(InvalidSchedule) as error:
        legacy_generated_cpp(
            cin, result_shape, bindings, compile_options=auto_options(arm)
        )
    assert [d.code for d in error.value.diagnostics] == ["sparse_parent_dominance"]
    assert error.value.diagnostics[0].stage == "loop_plan", name


@pytest.mark.parametrize(
    ("cell", "loop_order"),
    [
        (_reduction_cell("sss ijk->ij", "sss", "ss", "ijk", "ij"), ("i", "j", "k")),
        (_reduction_cell("ds ij->i", "ds", "s", "ij", "i"), ("i", "j")),
    ],
    ids=("sss-legal-explicit-order", "ds-legal-explicit-order"),
)
def test_the_legal_explicit_order_reaches_the_bound_prefix_family(cell, loop_order):
    """A legal explicit order binds every result coordinate above the reduction.

    The ordered key is then empty, and the bound-prefix family owns the shape.
    These two controls used to stop at their own boundaries even so --
    ``ds ij->i`` because a dense domain drove a compressed result level,
    ``sss ijk->ij`` because the doubly-compressed receiver never reached the
    ordered-key branch -- and both boundaries moved on 2026-10-11: the dense-driven
    leaf appends when its reduction ran, and a doubly-compressed receiver with no
    drained key is classified by its split rather than by its layout.  The
    explicit order and the automatic one now agree on the family.
    """

    name, cin, result_shape, bindings = cell
    options = CompileOptions.from_environment(
        environ={}, requested_schedule=Schedule(loop_order=loop_order)
    )
    kernel = compile_cin_via_loopir(cin, result_shape, bindings, compile_options=options)
    assert "wksp" not in kernel.cpp_source, name
    assert "_reduction = 0.0;" in kernel.cpp_source, name


@pytest.mark.parametrize(
    ("cell", "family"),
    FORMER_REORDER_BLOCKED,
    ids=[cell[0][0] for cell in FORMER_REORDER_BLOCKED],
)
@pytest.mark.parametrize("arm", [False, True])
def test_the_former_reorder_blocked_cells_reach_their_family(cell, family, arm):
    """Each of the five now compiles, and the record says through which family."""

    name, cin, result_shape, bindings = cell
    kernel = compile_cin_via_loopir(
        cin, result_shape, bindings, compile_options=auto_options(arm)
    )
    if family == "bound_prefix":
        # The bound-prefix family accumulates into a scalar; no workspace
        # region of either representation participates.
        assert "wksp" not in kernel.cpp_source, name
        assert "_reduction = 0.0;" in kernel.cpp_source, name
    else:
        assert family == "ordered_key"
        assert "wksp.sort();" in kernel.cpp_source, name


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
def test_the_workspace_tile_seam_is_empty_for_this_matrix(arm):
    """The successor to ``test_auto_tile_blocked_cells_keep_their_schedule_code``.

    That lock's own docstring said it "moves to whatever still occupies the seam
    rather than being deleted".  Nothing in this census's matrix occupies it any
    more: the automatic origin no longer composes a sparse workspace with an
    affine tile, because it no longer emits an affine tile over a non-dense
    receiver at all.
    """

    assert AUTO_TILE_BLOCKED == []
    for name, cin, result_shape, bindings, _prefix, _key_rank in MIGRATED:
        if not name.startswith("TTM dds"):
            continue
        kernel = compile_cin_via_loopir(
            cin, result_shape, bindings, compile_options=auto_options(arm)
        )
        assert kernel.cpp_source, name


@pytest.mark.parametrize("arm", [False, True])
def test_dense_receivers_keep_their_automatic_tile(arm):
    """The legality rule must be INERT on a dense receiver.

    This is the neutrality half of the tile-legality change and the reason it can
    claim to be byte-neutral wherever tiling is legal: the guard fires only on a
    receiver it has positively identified as non-dense, so an all-dense receiver
    must still get exactly the tile it got before.
    """

    from scorch.compiler.scheduler import Scheduler

    dense_receiver = _reduction_cell("ddd ijk->ik", "ddd", "dd", "ijk", "ik")
    _name, cin, _result_shape, _bindings = dense_receiver
    scheduled = Scheduler.auto_schedule_plan(cin, regblock_enabled=arm)
    assert scheduled.verified_loop_plan.tiles, (
        "a dense receiver must keep its automatic affine tile; the non-dense "
        "guard is not allowed to reach it"
    )


def test_census_covers_the_declared_representative_matrix():
    """Pin the finite review matrix without claiming layout exhaustiveness."""

    assert len(REORDER_BLOCKED) == 0
    assert len(PERMUTED_RESULT) == 4
    assert len(FORMER_REORDER_BLOCKED) == 5
    # 9 -> 11 and 2 -> 0: the two ``TTM dds`` cells moved lists rather than
    # leaving the matrix, and so did the four permuted-result cells, so the
    # 20-name total below is unchanged.
    assert len(MIGRATED) == 11
    assert len(AUTO_TILE_BLOCKED) == 0
    names = (
        [cell[0] for cell in REORDER_BLOCKED]
        + _PERMUTED_RESULT_IDS
        + [cell[0][0] for cell in FORMER_REORDER_BLOCKED]
        + [cell[0] for cell in MIGRATED]
        + [cell[0] for cell in AUTO_TILE_BLOCKED]
    )
    assert len(names) == len(set(names)) == 20
    # All six canonical TTM layouts named by the inherited review are present.
    ttm = [name for name in names if name.startswith("TTM ")]
    assert len(ttm) == 6
    assert len({name.split("->")[1].strip() for name in ttm}) == 3


def test_migrated_cells_span_every_key_rank_and_anchor():
    """The migration is general over the split, not over a layout list."""

    splits = {(cell[4], cell[5]) for cell in MIGRATED}
    assert {prefix for prefix, _ in splits} == {0, 1, 2}
    assert {key_rank for _, key_rank in splits} == {1, 2}
