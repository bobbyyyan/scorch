"""Typed LoopIR scheduling passes: legality, purity, provenance, erasure.

The reorder and affine-tiling passes must be pure functions over verified
programs, fail closed with stable codes on everything outside the migrated
schedule families, mirror the legacy placement semantics exactly, retain
provenance to the unscheduled base program, and preserve semantics — the
oracle differentials here prove every iteration point of a scheduled
program is visited exactly once (exact integer-float counting) across
ragged, exact, oversized, and zero extents.
"""

from dataclasses import replace
import random

import pytest

from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.identity import IndexId
from scorch.compiler.loop_plan import (
    MAX_AFFINE_TILE_WIDTH,
    LoopPart,
    LoopPlacement,
    LoopPlan,
    LoopRef,
    LoopTile,
    PlacementKind,
)
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.lower_cin import lower_normalized_cin_to_loopir
from scorch.compiler.loopir.nodes import (
    BinaryOp as LoopIRBinaryOp,
    Block,
    DenseFor,
    TileId,
    TileInnerFor,
    TileOuterFor,
)
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.printer import canonical_program_dump, print_program
from scorch.compiler.loopir.schedule_passes import (
    ScheduledLoopIR,
    SchedulePassError,
    apply_affine_tile,
    apply_schedule_plan,
    apply_stack_tile,
    erase_schedule,
    reorder_loops,
    verify_scheduled_loopir,
)
from scorch.compiler.loopir.verifier import verify_program


def outermost_placement():
    return LoopPlacement(PlacementKind.OUTERMOST)


def affine_tile(index_id, width, placement=None, unroll=False, accum="direct"):
    return LoopTile(
        loop=LoopRef(index_id),
        width=width,
        placement=placement or outermost_placement(),
        parallel=False,
        kind="affine",
        accumulation=accum,
        unroll=unroll,
    )


def build_matmul_ikj():
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=Operation.ADD
    )
    return ForAll(i, ForAll(k, ForAll(j, assign))), (i, k, j)


def build_two_reduction_ijk():
    """C[i] += A[i,j] * B[i,k]: j and k are mutually unordered reductions."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="d")
    assign = TensorAssign(
        c[i], CINBinaryOp(Operation.MUL, a[i, j], b[i, k]), op=Operation.ADD
    )
    return ForAll(i, ForAll(j, ForAll(k, assign))), (i, j, k)


def build_spmv():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="ds")
    x = TensorVar("x", fmt="d")
    y = TensorVar("y", fmt="d")
    assign = TensorAssign(
        y[i], CINBinaryOp(Operation.MUL, a[i, j], x[j]), op=Operation.ADD
    )
    return ForAll(i, ForAll(j, assign)), (i, j)


def build_spmm_ijk():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(
        c[i, k], CINBinaryOp(Operation.MUL, a[i, j], b[j, k]), op=Operation.ADD
    )
    return ForAll(i, ForAll(j, ForAll(k, assign))), (i, j, k)


def build_union_add_to_csr():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    c = TensorVar("C", fmt="ds")
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]))
    return ForAll(i, ForAll(j, assign)), (i, j)


def lower(cin, planned_loop_order=None):
    return lower_normalized_cin_to_loopir(
        normalize_cin(cin), planned_loop_order=planned_loop_order
    )


def expect_code(code, call, *args, **kwargs):
    with pytest.raises(SchedulePassError) as error:
        call(*args, **kwargs)
    assert error.value.defect.code == code, error.value.defect
    return error.value.defect


def chain_types(program):
    types = []
    body = program.body
    while isinstance(body, Block) and len(body.statements) == 1:
        only = body.statements[0]
        if not hasattr(only, "body"):
            break
        types.append(type(only))
        body = only.body
    return types


# -- reorder ------------------------------------------------------------------


def test_reorder_identity_is_a_no_op():
    cin, (i, j, k) = build_two_reduction_ijk()
    lowering = lower(cin)
    result = reorder_loops(lowering.program, lowering.loop_index_ids)
    assert result is lowering.program


def test_reorder_permutes_and_preserves_semantics_exactly():
    cin, (i, j, k) = build_two_reduction_ijk()
    lowering = lower(cin)
    new_order = (i.index_id, k.index_id, j.index_id)
    reordered = reorder_loops(lowering.program, new_order)
    verify_program(reordered)
    assert reordered is not lowering.program
    # The input artifact is untouched.
    assert canonical_program_dump(lowering.program) == canonical_program_dump(
        lower(cin).program
    )
    a = [[float(1 + i_ * 5 + j_) for j_ in range(5)] for i_ in range(4)]
    b = [[float(1 + i_ * 7 + k_) for k_ in range(7)] for i_ in range(4)]
    inputs = {
        lowering.input_symbols[0]: a,
        lowering.input_symbols[1]: b,
    }
    shapes = {lowering.result_symbol: (4,)}
    base_out = run_program(lowering.program, inputs, shapes)
    reordered_out = run_program(reordered, inputs, shapes)
    assert base_out == reordered_out


def test_reorder_is_deterministic():
    cin, (i, j, k) = build_two_reduction_ijk()
    lowering = lower(cin)
    new_order = (i.index_id, k.index_id, j.index_id)
    first = reorder_loops(lowering.program, new_order)
    second = reorder_loops(lowering.program, new_order)
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert print_program(first) == print_program(second)


def test_reorder_rejects_incomplete_and_repeated_orders():
    cin, (i, j, k) = build_two_reduction_ijk()
    lowering = lower(cin)
    expect_code(
        "reorder_incomplete_order",
        reorder_loops,
        lowering.program,
        (i.index_id, j.index_id),
    )
    expect_code(
        "reorder_incomplete_order",
        reorder_loops,
        lowering.program,
        (i.index_id, j.index_id, j.index_id),
    )
    foreign = IndexId(10_000_001)
    expect_code(
        "reorder_incomplete_order",
        reorder_loops,
        lowering.program,
        (i.index_id, j.index_id, foreign),
    )
    expect_code(
        "reorder_invalid_order",
        reorder_loops,
        lowering.program,
        ([i.index_id.value], j.index_id, k.index_id),
    )


def test_reorder_rejects_breaking_sparse_parent_dominance():
    cin, (i, j) = build_spmv()
    lowering = lower(cin)
    expect_code(
        "reorder_sparse_dependency",
        reorder_loops,
        lowering.program,
        (j.index_id, i.index_id),
    )


def test_reorder_rejects_ordered_assembly_permutations():
    cin, (i, j) = build_union_add_to_csr()
    lowering = lower(cin)
    expect_code(
        "reorder_ordered_assembly",
        reorder_loops,
        lowering.program,
        (j.index_id, i.index_id),
    )
    # The identity order remains a legal no-op even over appends.
    assert reorder_loops(lowering.program, (i.index_id, j.index_id)) is (
        lowering.program
    )


def test_reorder_rejects_split_chains():
    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    tiled = apply_affine_tile(lowering.program, affine_tile(j.index_id, 4))
    expect_code(
        "reorder_split_chain",
        reorder_loops,
        tiled,
        (i.index_id, k.index_id, j.index_id),
    )


def test_reorder_enables_out_of_family_source_orders():
    """A plan may repair a CIN nest order the dense family cannot emit."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=Operation.ADD
    )
    cin = ForAll(i, ForAll(j, ForAll(k, assign)))
    planned = (i.index_id, k.index_id, j.index_id)
    lowering = lower(cin, planned_loop_order=planned)
    reordered = reorder_loops(lowering.program, planned)
    verify_program(reordered)
    direct_cin, _ = build_matmul_ikj()
    direct = lower(direct_cin)
    assert canonical_program_dump(reordered) == canonical_program_dump(direct.program)


# -- affine tiling ------------------------------------------------------------


def test_tile_outermost_splits_with_shared_identity_and_provenance():
    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    tiled = apply_affine_tile(lowering.program, affine_tile(j.index_id, 4, unroll=True))
    verify_program(tiled)
    assert chain_types(tiled) == [TileOuterFor, DenseFor, DenseFor, TileInnerFor]
    outer = tiled.body.statements[0]
    inner = outer.body.statements[0].body.statements[0].body.statements[0]
    assert type(outer) is TileOuterFor and type(inner) is TileInnerFor
    assert outer.tile == inner.tile
    assert outer.index == inner.index == j.index_id
    assert outer.dimension == inner.dimension
    assert outer.width == inner.width == 4
    assert inner.unroll is True


def test_tile_child_of_and_at_depth_mirror_legacy_placement():
    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    child_of_i = apply_affine_tile(
        lowering.program,
        affine_tile(
            j.index_id,
            4,
            placement=LoopPlacement(PlacementKind.CHILD_OF, parent=LoopRef(i.index_id)),
        ),
    )
    # i, j_out, k, j_in — the outer loop sits directly inside its parent.
    assert chain_types(child_of_i) == [
        DenseFor,
        TileOuterFor,
        DenseFor,
        TileInnerFor,
    ]
    at_depth_2 = apply_affine_tile(
        lowering.program,
        affine_tile(
            j.index_id,
            4,
            placement=LoopPlacement(PlacementKind.AT_DEPTH, depth=2),
        ),
    )
    assert chain_types(at_depth_2) == [
        DenseFor,
        DenseFor,
        TileOuterFor,
        TileInnerFor,
    ]


def test_tile_application_order_matches_legacy_sequencing():
    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    tiled = apply_affine_tile(lowering.program, affine_tile(i.index_id, 2))
    tiled = apply_affine_tile(tiled, affine_tile(j.index_id, 4))
    # Applied in order: i first, then j outermost above it —
    # j_out, i_out, i_in, k, j_in, exactly the legacy chain.
    assert chain_types(tiled) == [
        TileOuterFor,
        TileOuterFor,
        TileInnerFor,
        DenseFor,
        TileInnerFor,
    ]
    loops = []
    body = tiled.body
    while isinstance(body, Block) and hasattr(body.statements[0], "body"):
        loops.append(body.statements[0])
        body = body.statements[0].body
    assert loops[0].index == j.index_id
    assert loops[1].index == i.index_id
    assert loops[2].index == i.index_id
    assert loops[4].index == j.index_id


def test_tile_rejects_sparse_targets_missing_targets_and_double_splits():
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    expect_code(
        "tile_target_not_dense",
        apply_affine_tile,
        lowering.program,
        affine_tile(j.index_id, 4),
    )
    foreign = IndexId(10_000_002)
    expect_code(
        "tile_target_missing",
        apply_affine_tile,
        lowering.program,
        affine_tile(foreign, 4),
    )
    tiled = apply_affine_tile(lowering.program, affine_tile(k.index_id, 4))
    expect_code(
        "tile_target_already_split",
        apply_affine_tile,
        tiled,
        affine_tile(k.index_id, 2),
    )


def test_tile_rejects_illegal_placements():
    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    foreign = IndexId(10_000_003)
    expect_code(
        "tile_invalid_placement",
        apply_affine_tile,
        lowering.program,
        affine_tile(
            j.index_id,
            4,
            placement=LoopPlacement(PlacementKind.CHILD_OF, parent=LoopRef(foreign)),
        ),
    )
    expect_code(
        "tile_invalid_placement",
        apply_affine_tile,
        lowering.program,
        affine_tile(
            j.index_id,
            4,
            placement=LoopPlacement(PlacementKind.AT_DEPTH, depth=4),
        ),
    )
    # The origin loop must dominate the point loop.
    expect_code(
        "tile_invalid_placement",
        apply_affine_tile,
        lowering.program,
        affine_tile(
            i.index_id,
            2,
            placement=LoopPlacement(PlacementKind.AT_DEPTH, depth=2),
        ),
    )


def test_tile_rejects_unsupported_widths_and_families():
    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    expect_code(
        "tile_invalid_width",
        apply_affine_tile,
        lowering.program,
        affine_tile(j.index_id, 0),
    )
    expect_code(
        "tile_invalid_width",
        apply_affine_tile,
        lowering.program,
        affine_tile(j.index_id, MAX_AFFINE_TILE_WIDTH + 1),
    )
    expect_code(
        "tile_target_not_logical",
        apply_affine_tile,
        lowering.program,
        replace(
            affine_tile(j.index_id, 4),
            loop=LoopRef(j.index_id, LoopPart.OUTER),
        ),
    )
    expect_code(
        "unsupported_schedule_accumulation",
        apply_affine_tile,
        lowering.program,
        affine_tile(j.index_id, 4, accum="stack"),
    )
    expect_code(
        "unsupported_schedule_accumulation",
        apply_affine_tile,
        lowering.program,
        affine_tile(j.index_id, 4, accum="heap"),
    )
    panel = LoopTile(
        loop=LoopRef(j.index_id),
        width=4,
        placement=outermost_placement(),
        parallel=False,
        kind="panel",
        accumulation="direct",
        unroll=False,
    )
    expect_code(
        "unsupported_schedule_panel", apply_affine_tile, lowering.program, panel
    )
    parallel = LoopTile(
        loop=LoopRef(j.index_id),
        width=4,
        placement=outermost_placement(),
        parallel=True,
        kind="affine",
        accumulation="direct",
        unroll=False,
    )
    expect_code(
        "unsupported_schedule_parallel",
        apply_affine_tile,
        lowering.program,
        parallel,
    )


def test_direct_tile_pass_rejects_forged_fields_with_stable_diagnostics():
    cin, (_i, _k, j) = build_matmul_ikj()
    lowering = lower(cin)
    missing_loop = affine_tile(j.index_id, 4)
    object.__delattr__(missing_loop, "loop")
    expect_code(
        "invalid_schedule_tile",
        apply_affine_tile,
        lowering.program,
        missing_loop,
    )

    malformed_parent = replace(
        affine_tile(j.index_id, 4),
        placement=LoopPlacement(
            PlacementKind.CHILD_OF,
            parent=LoopRef(j.index_id),
        ),
    )
    object.__setattr__(malformed_parent.placement, "parent", "j")
    expect_code(
        "invalid_schedule_tile",
        apply_affine_tile,
        lowering.program,
        malformed_parent,
    )


def test_tile_is_deterministic_and_pure():
    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    before = canonical_program_dump(lowering.program)
    first = apply_affine_tile(lowering.program, affine_tile(j.index_id, 4))
    second = apply_affine_tile(lowering.program, affine_tile(j.index_id, 4))
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert canonical_program_dump(lowering.program) == before


# -- plan application ---------------------------------------------------------


def test_apply_schedule_plan_returns_full_provenance():
    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(affine_tile(j.index_id, 4),),
    )
    scheduled = apply_schedule_plan(lowering.program, plan)
    assert type(scheduled) is ScheduledLoopIR
    assert scheduled.base_program is lowering.program
    assert scheduled.plan is plan
    verify_program(scheduled.program)
    parts = [
        (entry.index, entry.part, entry.tile is not None) for entry in scheduled.loops
    ]
    assert parts == [
        (j.index_id, LoopPart.OUTER, True),
        (i.index_id, LoopPart.LOGICAL, False),
        (k.index_id, LoopPart.LOGICAL, False),
        (j.index_id, LoopPart.INNER, True),
    ]
    assert scheduled.loops[0].tile == scheduled.loops[3].tile
    verify_scheduled_loopir(scheduled)


def test_scheduled_artifact_snapshots_and_verifies_provenance_ownership():
    cin, (_i, _k, j) = build_matmul_ikj()
    lowering = lower(cin)
    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(affine_tile(j.index_id, 4),),
    )
    scheduled = apply_schedule_plan(lowering.program, plan)
    caller_owned = list(scheduled.loops)
    detached = ScheduledLoopIR(
        scheduled.base_program,
        scheduled.plan,
        scheduled.program,
        caller_owned,
    )
    caller_owned.clear()
    assert detached.loops == scheduled.loops
    verify_scheduled_loopir(detached)


def test_scheduled_artifact_rejects_cross_field_mismatches():
    cin, (_i, _k, j) = build_matmul_ikj()
    lowering = lower(cin)
    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(affine_tile(j.index_id, 4),),
    )
    scheduled = apply_schedule_plan(lowering.program, plan)

    expect_code(
        "scheduled_program_mismatch",
        verify_scheduled_loopir,
        replace(scheduled, program=scheduled.base_program),
    )
    expect_code(
        "scheduled_program_mismatch",
        verify_scheduled_loopir,
        replace(
            scheduled,
            plan=replace(
                plan,
                tiles=(replace(plan.tiles[0], width=8),),
            ),
        ),
    )
    expect_code(
        "scheduled_provenance_mismatch",
        verify_scheduled_loopir,
        replace(scheduled, loops=scheduled.loops[:-1]),
    )
    wrong_part = replace(scheduled.loops[0], part=LoopPart.LOGICAL)
    expect_code(
        "scheduled_provenance_mismatch",
        verify_scheduled_loopir,
        replace(scheduled, loops=(wrong_part, *scheduled.loops[1:])),
    )
    expect_code(
        "scheduled_base_not_unscheduled",
        verify_scheduled_loopir,
        replace(scheduled, base_program=scheduled.program),
    )


def test_scheduled_artifact_rejects_malformed_stored_state():
    cin, (_i, _k, _j) = build_matmul_ikj()
    lowering = lower(cin)
    scheduled = apply_schedule_plan(
        lowering.program,
        LoopPlan(loop_order=lowering.loop_index_ids),
    )
    object.__delattr__(scheduled, "loops")
    expect_code("invalid_scheduled_artifact", verify_scheduled_loopir, scheduled)


def test_scheduled_artifact_requires_exact_provenance_identities_and_enums():
    cin, (_i, _k, j) = build_matmul_ikj()
    lowering = lower(cin)
    scheduled = apply_schedule_plan(
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(affine_tile(j.index_id, 4),),
        ),
    )
    position = next(
        position
        for position, provenance in enumerate(scheduled.loops)
        if provenance.tile is not None
    )
    source = scheduled.loops[position]
    assert source.tile is not None

    missing_tile_value = TileId(source.tile.value)
    object.__delattr__(missing_tile_value, "value")
    missing_tile = replace(source, tile=missing_tile_value)

    extra_index_state = IndexId(source.index.value)
    object.__setattr__(extra_index_state, "shadow_value", source.index.value)
    extra_index = replace(source, index=extra_index_state)

    forged_part = object.__new__(LoopPart)
    object.__setattr__(forged_part, "_name_", "FORGED")
    object.__setattr__(forged_part, "_value_", "forged")
    noncanonical_part = replace(source, part=forged_part)

    for malformed in (missing_tile, extra_index, noncanonical_part):
        loops = list(scheduled.loops)
        loops[position] = malformed
        expect_code(
            "invalid_scheduled_artifact",
            verify_scheduled_loopir,
            replace(scheduled, loops=tuple(loops)),
        )


def test_scheduled_artifact_rejects_a_non_plan_with_artifact_diagnostic():
    cin, (_i, _k, _j) = build_matmul_ikj()
    lowering = lower(cin)
    scheduled = apply_schedule_plan(
        lowering.program,
        LoopPlan(loop_order=lowering.loop_index_ids),
    )
    object.__setattr__(scheduled, "plan", object())
    expect_code("invalid_scheduled_artifact", verify_scheduled_loopir, scheduled)


def test_plan_gate_rejects_unconsumed_tile_parts_and_placement_fields():
    cin, (i, _k, j) = build_matmul_ikj()
    lowering = lower(cin)
    base_tile = affine_tile(j.index_id, 4)
    expect_code(
        "tile_target_not_logical",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(
                replace(
                    base_tile,
                    loop=LoopRef(j.index_id, LoopPart.OUTER),
                ),
            ),
        ),
    )
    expect_code(
        "tile_invalid_placement",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(
                replace(
                    base_tile,
                    placement=LoopPlacement(
                        PlacementKind.OUTERMOST,
                        parent=LoopRef(i.index_id),
                    ),
                ),
            ),
        ),
    )


def test_resuming_ignores_non_schema_dataclass_state():
    cin, (_i, _k, _j) = build_matmul_ikj()
    program = lower(cin).program
    assert LoopIRBuilder.resuming(program).new_tile_id() == TileId(0)
    object.__setattr__(program.body, "ghost_tile", TileId(999))
    verify_program(program)
    assert LoopIRBuilder.resuming(program).new_tile_id() == TileId(0)


@pytest.mark.parametrize("provenance", ["auto", "tuned", "fallback"])
def test_apply_schedule_plan_provenance_gate(provenance):
    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    plan = LoopPlan(loop_order=lowering.loop_index_ids, provenance=provenance)
    expect_code(
        "unsupported_schedule_provenance",
        apply_schedule_plan,
        lowering.program,
        plan,
    )


def test_apply_schedule_plan_rejects_unmigrated_families():
    from scorch.compiler.loop_plan import (
        OperandRelayout,
        PanelBound,
        ResultTile,
    )

    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    order = lowering.loop_index_ids
    a_symbol = lowering.input_symbols[0]
    # Panels are a migrated family now: a bound with no panel tile to
    # consume it is a malformed panel plan, not an unmigrated one.
    expect_code(
        "invalid_schedule_panel",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=order,
            panel_bounds=(PanelBound(LoopRef(j.index_id), a_symbol, 1),),
        ),
    )
    expect_code(
        "unsupported_schedule_relayout",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=order,
            relayout=OperandRelayout(
                operand_id=a_symbol,
                pack_loop=LoopRef(j.index_id),
                panel_loop=LoopRef(k.index_id),
                scope_loop=LoopRef(k.index_id),
                row_loop=LoopRef(i.index_id),
                strip_width=4,
                access_indices=(k.index_id, j.index_id),
                operand_panel_level=0,
                operand_pack_level=1,
            ),
        ),
    )
    expect_code(
        "unsupported_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=order,
            result_tile=ResultTile(
                result_id=lowering.result_symbol,
                tile_loop=LoopRef(j.index_id),
                result_level=1,
                result_prefix=(i.index_id,),
                access_indices=(i.index_id, j.index_id),
            ),
        ),
    )
    expect_code(
        "unsupported_schedule_parallel",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(loop_order=order, parallel_loop=LoopRef(i.index_id)),
    )
    expect_code(
        "unsupported_schedule_accumulation",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=order,
            tiles=(affine_tile(j.index_id, 4, accum="heap"),),
        ),
    )


def test_apply_schedule_plan_requires_a_loop_plan():
    cin, _ = build_matmul_ikj()
    lowering = lower(cin)
    with pytest.raises(TypeError):
        apply_schedule_plan(lowering.program, object())


# -- erasure and exactly-once semantics --------------------------------------


def test_erase_schedule_restores_the_base_program():
    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(affine_tile(j.index_id, 4),),
    )
    scheduled = apply_schedule_plan(lowering.program, plan)
    erased = erase_schedule(scheduled.program)
    assert canonical_program_dump(erased) == canonical_program_dump(lowering.program)
    assert erase_schedule(lowering.program) is lowering.program


def test_scheduled_iteration_visits_every_point_exactly_once():
    """Counting semantics: all-ones inputs make each output cell count its
    visits, and exact integer float arithmetic makes the comparison exact."""

    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    for width in (1, 2, 3, 4, 5, 7):
        plan = LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(affine_tile(j.index_id, width),),
        )
        scheduled = apply_schedule_plan(lowering.program, plan)
        rows, inner, cols = 3, 4, 5
        ones_a = [[1.0] * inner for _ in range(rows)]
        ones_b = [[1.0] * cols for _ in range(inner)]
        inputs = {
            lowering.input_symbols[0]: ones_a,
            lowering.input_symbols[1]: ones_b,
        }
        shapes = {lowering.result_symbol: (rows, cols)}
        counted = run_program(scheduled.program, inputs, shapes)
        assert counted[lowering.result_symbol] == [
            [float(inner)] * cols for _ in range(rows)
        ]


def test_scheduled_oracle_matches_base_on_randomized_dimensions():
    """Randomized dims/widths; also tiles a reduction loop deliberately.

    The typed pass owns semantic legality only: splitting an ADD-reduction
    loop is exactly-once- and reassociation-legal, and the oracle proves it
    here, even though the legacy Schedule adapter refuses to request it
    (its stack accumulator machinery would be required on the legacy path).
    """

    rng = random.Random(20260723)
    cin, (i, j, k) = build_two_reduction_ijk()
    lowering = lower(cin)
    for trial in range(8):
        # Zero extents are covered by the SpMM case below, where the CSR
        # binding fixes the shapes a nested-list zero extent cannot express.
        rows = rng.randrange(1, 5)
        jdim = rng.randrange(1, 6)
        kdim = rng.randrange(1, 6)
        width = rng.choice((1, 2, 3, 4, 7))
        order = list(lowering.loop_index_ids)
        if trial % 2:
            order = [order[0], order[2], order[1]]
        plan = LoopPlan(
            loop_order=tuple(order),
            tiles=(affine_tile(k.index_id, width),),
        )
        scheduled = apply_schedule_plan(lowering.program, plan)
        a = [[float(rng.randrange(-4, 5)) for _ in range(jdim)] for _ in range(rows)]
        b = [[float(rng.randrange(-4, 5)) for _ in range(kdim)] for _ in range(rows)]
        inputs = {
            lowering.input_symbols[0]: a,
            lowering.input_symbols[1]: b,
        }
        shapes = {lowering.result_symbol: (rows,)}
        assert run_program(scheduled.program, inputs, shapes) == run_program(
            lowering.program, inputs, shapes
        )


def test_scheduled_spmm_tile_matches_base_oracle():
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    from scorch.compiler.loopir.levels import CsrMatrix

    dense_a = [
        [0.0, 2.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 4.0],
    ]
    csr = CsrMatrix.from_dense(dense_a)
    for n, width in ((2, 4), (4, 4), (6, 4), (0, 4), (6, 9)):
        plan = LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(affine_tile(k.index_id, width),),
        )
        scheduled = apply_schedule_plan(lowering.program, plan)
        b = [[float(1 + r * n + c) for c in range(n)] for r in range(4)]
        inputs = {
            lowering.input_symbols[0]: csr,
            lowering.input_symbols[1]: b,
        }
        shapes = {lowering.result_symbol: (3, n)}
        assert run_program(scheduled.program, inputs, shapes) == run_program(
            lowering.program, inputs, shapes
        )


def test_scheduled_chain_shape_guard():
    """A verified program outside the single-chain shape fails closed."""

    from scorch.compiler.loopir.build import LoopIRBuilder

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]))
    lowering = lower(ForAll(i, ForAll(j, assign)))
    program = lowering.program
    builder = LoopIRBuilder.resuming(program)
    outer = program.body.statements[0]
    inner = outer.body.statements[0]
    leaf = inner.body.statements[0]
    duplicate_leaf = builder.store(
        leaf.tensor,
        tuple(builder.index_value(index.index) for index in leaf.indices),
        builder.binary(
            LoopIRBinaryOp.ADD,
            builder.load(
                program.inputs[0],
                tuple(
                    builder.index_value(index.index) for index in leaf.value.lhs.indices
                ),
            ),
            builder.load(
                program.inputs[1],
                tuple(
                    builder.index_value(index.index) for index in leaf.value.rhs.indices
                ),
            ),
        ),
    )
    two_leaf_inner = builder.dense_for(
        inner.index, inner.dimension, builder.block((leaf, duplicate_leaf))
    )
    forged = builder.program(
        program.dimensions,
        program.tensors,
        program.inputs,
        program.outputs,
        builder.block(
            (
                builder.dense_for(
                    outer.index,
                    outer.dimension,
                    builder.block((two_leaf_inner,)),
                ),
            )
        ),
    )
    verify_program(forged)
    expect_code(
        "unsupported_schedule_shape",
        reorder_loops,
        forged,
        lowering.loop_index_ids,
    )


# -- stack accumulation (workspace regions) -----------------------------------


def stack_tile(index_id, width, placement=None, unroll=False):
    return affine_tile(
        index_id, width, placement=placement, unroll=unroll, accum="stack"
    )


def region_of(program):
    from scorch.compiler.loopir.nodes import WorkspaceRegion

    body = program.body
    while isinstance(body, Block) and len(body.statements) == 1:
        only = body.statements[0]
        if type(only) is WorkspaceRegion:
            return only
        if not hasattr(only, "body"):
            break
        body = only.body
    raise AssertionError("no workspace region in the chain")


def test_apply_stack_tile_builds_the_workspace_region_shape():
    from scorch.compiler.loopir.nodes import (
        SparseFor,
        StoreReduce,
        WorkspaceRead,
        WorkspaceReduce,
        WorkspaceRegion,
    )

    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    scheduled = apply_stack_tile(lowering.program, stack_tile(k.index_id, 4))
    verify_program(scheduled)
    # Chain: origin loop over k, dense row loop, then the region.
    assert chain_types(scheduled) == [TileOuterFor, DenseFor]
    region = region_of(scheduled)
    assert type(region) is WorkspaceRegion
    assert region.workspace.name == "wksp"
    assert region.workspace.tile == scheduled.body.statements[0].tile
    producer_first = region.producer.statements[0]
    assert type(producer_first) is SparseFor
    producer_point = producer_first.body.statements[0]
    assert type(producer_point) is TileInnerFor
    reduce_leaf = producer_point.body.statements[0]
    assert type(reduce_leaf) is WorkspaceReduce
    assert reduce_leaf.workspace == region.workspace.workspace
    consumer_point = region.consumer.statements[0]
    assert type(consumer_point) is TileInnerFor
    assert consumer_point.tile == producer_point.tile
    copy_out = consumer_point.body.statements[0]
    assert type(copy_out) is StoreReduce
    assert type(copy_out.value) is WorkspaceRead
    assert copy_out.value.workspace == region.workspace.workspace


def test_apply_stack_tile_is_pure_and_deterministic():
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    before = canonical_program_dump(lowering.program)
    first = apply_stack_tile(lowering.program, stack_tile(k.index_id, 4))
    second = apply_stack_tile(lowering.program, stack_tile(k.index_id, 4))
    assert canonical_program_dump(lowering.program) == before
    assert canonical_program_dump(first) == canonical_program_dump(second)
    region = region_of(first)
    assert LoopIRBuilder.resuming(first).new_workspace_id().value == (
        region.workspace.workspace.value + 1
    )


def test_apply_stack_tile_rejects_hostile_plan_primitives_fail_closed():
    class ExplosiveStr(str):
        def __eq__(self, other):
            raise RuntimeError("hostile string comparison escaped")

        __hash__ = str.__hash__

    class ExplosiveInt(int):
        def __eq__(self, other):
            raise RuntimeError("hostile integer comparison escaped")

        def __lt__(self, other):
            raise RuntimeError("hostile integer comparison escaped")

        def __gt__(self, other):
            raise RuntimeError("hostile integer comparison escaped")

        __hash__ = int.__hash__

    cin, (_, _, k) = build_spmm_ijk()
    lowering = lower(cin)
    valid = stack_tile(k.index_id, 4)
    malformed_tiles = (
        replace(valid, kind=ExplosiveStr("affine")),
        replace(valid, accumulation=ExplosiveStr("stack")),
        replace(valid, width=ExplosiveInt(4)),
        replace(
            valid,
            loop=LoopRef(IndexId(ExplosiveInt(k.index_id.value))),
        ),
    )
    for malformed in malformed_tiles:
        expect_code(
            "invalid_schedule_tile",
            apply_stack_tile,
            lowering.program,
            malformed,
        )

    malformed_plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(valid,),
        provenance=ExplosiveStr("explicit"),
    )
    expect_code(
        "invalid_schedule_plan",
        apply_schedule_plan,
        lowering.program,
        malformed_plan,
    )


def test_schedule_pass_rejects_deleted_plan_state_instead_of_erasing_tiles():
    cin, (_, _, k) = build_spmm_ijk()
    lowering = lower(cin)
    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(stack_tile(k.index_id, 4),),
        provenance="explicit",
    )
    object.__delattr__(plan, "tiles")

    expect_code(
        "invalid_schedule_plan",
        apply_schedule_plan,
        lowering.program,
        plan,
    )


def test_schedule_pass_rejects_forged_enum_members_before_dispatch():
    cin, (_, _, k) = build_spmm_ijk()
    lowering = lower(cin)

    forged_part = object.__new__(LoopPart)
    object.__setattr__(forged_part, "_name_", "FORGED")
    object.__setattr__(forged_part, "_value_", "forged")
    expect_code(
        "invalid_schedule_tile",
        apply_stack_tile,
        lowering.program,
        replace(
            stack_tile(k.index_id, 4),
            loop=LoopRef(k.index_id, forged_part),
        ),
    )

    forged_kind = object.__new__(PlacementKind)
    object.__setattr__(forged_kind, "_name_", "FORGED")
    object.__setattr__(forged_kind, "_value_", "forged")
    expect_code(
        "invalid_schedule_tile",
        apply_stack_tile,
        lowering.program,
        replace(
            stack_tile(k.index_id, 4),
            placement=LoopPlacement(forged_kind),
        ),
    )


def test_schedule_pass_diagnostics_handle_unrenderably_large_depths():
    huge_depth = 10**5000

    dense_cin, (_i, _k, j) = build_matmul_ikj()
    dense_lowering = lower(dense_cin)
    expect_code(
        "tile_invalid_placement",
        apply_affine_tile,
        dense_lowering.program,
        affine_tile(
            j.index_id,
            4,
            placement=LoopPlacement(PlacementKind.AT_DEPTH, depth=huge_depth),
        ),
    )

    sparse_cin, (_, _, k) = build_spmm_ijk()
    sparse_lowering = lower(sparse_cin)
    plan = LoopPlan(
        loop_order=sparse_lowering.loop_index_ids,
        tiles=(
            stack_tile(
                k.index_id,
                4,
                placement=LoopPlacement(
                    PlacementKind.AT_DEPTH,
                    depth=huge_depth,
                ),
            ),
        ),
        provenance="explicit",
    )
    expect_code(
        "tile_invalid_placement",
        apply_schedule_plan,
        sparse_lowering.program,
        plan,
    )


def test_apply_stack_tile_placements_resolve_against_the_prefix():
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    outermost = apply_stack_tile(lowering.program, stack_tile(k.index_id, 4))
    assert chain_types(outermost) == [TileOuterFor, DenseFor]
    child = apply_stack_tile(
        lowering.program,
        stack_tile(
            k.index_id,
            4,
            placement=LoopPlacement(PlacementKind.CHILD_OF, parent=LoopRef(i.index_id)),
        ),
    )
    assert chain_types(child) == [DenseFor, TileOuterFor]
    at_depth = apply_stack_tile(
        lowering.program,
        stack_tile(
            k.index_id, 4, placement=LoopPlacement(PlacementKind.AT_DEPTH, depth=1)
        ),
    )
    assert canonical_program_dump(at_depth) == canonical_program_dump(child)
    # The prefix above the region has exactly one loop, exactly as the
    # legacy prefix-of-Where rule: depth 2 is out of range.
    expect_code(
        "tile_invalid_placement",
        apply_stack_tile,
        lowering.program,
        stack_tile(
            k.index_id, 4, placement=LoopPlacement(PlacementKind.AT_DEPTH, depth=2)
        ),
    )
    # child_of may not name the reduction loop (it is inside the region).
    expect_code(
        "tile_invalid_placement",
        apply_stack_tile,
        lowering.program,
        stack_tile(
            k.index_id,
            4,
            placement=LoopPlacement(PlacementKind.CHILD_OF, parent=LoopRef(j.index_id)),
        ),
    )


def test_apply_stack_tile_mirrors_the_legacy_legality_boundary():
    # A pure elementwise program has no reduction to accumulate.
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    elementwise = lower(
        ForAll(
            i,
            ForAll(
                j, TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]))
            ),
        )
    )
    expect_code(
        "stack_tile_target_invalid",
        apply_stack_tile,
        elementwise.program,
        stack_tile(j.index_id, 4),
    )
    # The target must be the single trailing free loop after the last
    # reduction: the row loop of a matmul is not.
    cin, (i2, k2, j2) = build_matmul_ikj()
    matmul = lower(cin)
    expect_code(
        "stack_tile_target_invalid",
        apply_stack_tile,
        matmul.program,
        stack_tile(i2.index_id, 4),
    )
    # A reduction target is not a stack target either.
    expect_code(
        "stack_tile_target_invalid",
        apply_stack_tile,
        matmul.program,
        stack_tile(k2.index_id, 4),
    )
    # SpMV's trailing loop is the reduction itself: the workspace would
    # replace the chain root, exactly the legacy refusal.
    spmv_cin, (si, sj) = build_spmv()
    spmv = lower(spmv_cin)
    expect_code(
        "stack_tile_target_invalid",
        apply_stack_tile,
        spmv.program,
        stack_tile(sj.index_id, 4),
    )
    # C[k] += A[j, k] over order (j, k): the last reduction loop is the
    # chain root, so the region would replace it.
    rj, rk = IndexVar("j"), IndexVar("k")
    ra = TensorVar("A", fmt="dd")
    rc = TensorVar("C", fmt="d")
    root_cin = ForAll(
        rj, ForAll(rk, TensorAssign(rc[rk], ra[rj, rk], op=Operation.ADD))
    )
    root = lower(root_cin)
    expect_code(
        "stack_tile_root_scope",
        apply_stack_tile,
        root.program,
        stack_tile(rk.index_id, 4),
    )
    # Direct accumulation is the other pass's family.
    expect_code(
        "unsupported_schedule_accumulation",
        apply_stack_tile,
        lower(build_spmm_ijk()[0]).program,
        affine_tile(build_spmm_ijk()[1][2].index_id, 4),
    )


def test_apply_stack_tile_rejects_a_second_stack_transformation():
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    scheduled = apply_stack_tile(lowering.program, stack_tile(k.index_id, 4))
    # A region-terminated chain refuses every further stack transformation,
    # whatever its target: at most one stack accumulation exists.
    expect_code(
        "stack_tile_target_invalid",
        apply_stack_tile,
        scheduled,
        stack_tile(k.index_id, 2),
    )
    expect_code(
        "stack_tile_target_invalid",
        apply_stack_tile,
        scheduled,
        stack_tile(i.index_id, 2),
    )


def test_stack_plan_applies_through_apply_schedule_plan():
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(stack_tile(k.index_id, 4),),
    )
    scheduled = apply_schedule_plan(lowering.program, plan)
    verify_scheduled_loopir(scheduled)
    assert scheduled.base_program is lowering.program
    parts = [(entry.part, entry.tile is not None) for entry in scheduled.loops]
    # Documented order: prefix (origin + row), producer chain (reduction +
    # point), consumer chain (point).
    assert parts == [
        (LoopPart.OUTER, True),
        (LoopPart.LOGICAL, False),
        (LoopPart.LOGICAL, False),
        (LoopPart.INNER, True),
        (LoopPart.INNER, True),
    ]
    assert [entry.index for entry in scheduled.loops] == [
        k.index_id,
        i.index_id,
        j.index_id,
        k.index_id,
        k.index_id,
    ]


def test_stack_plan_composes_with_direct_tiles():
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(
            affine_tile(i.index_id, 2),
            stack_tile(
                k.index_id,
                4,
                placement=LoopPlacement(
                    PlacementKind.CHILD_OF,
                    parent=LoopRef(i.index_id, LoopPart.OUTER),
                ),
            ),
        ),
    )
    scheduled = apply_schedule_plan(lowering.program, plan)
    verify_scheduled_loopir(scheduled)
    assert chain_types(scheduled.program) == [TileOuterFor, TileOuterFor, TileInnerFor]


def test_stack_erasure_restores_the_base_program():
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    for tiles in (
        (stack_tile(k.index_id, 4),),
        (affine_tile(i.index_id, 2), stack_tile(k.index_id, 4)),
    ):
        plan = LoopPlan(loop_order=lowering.loop_index_ids, tiles=tiles)
        scheduled = apply_schedule_plan(lowering.program, plan)
        erased = erase_schedule(scheduled.program)
        assert canonical_program_dump(erased) == canonical_program_dump(
            lowering.program
        )


def test_stack_erasure_requires_the_exact_copy_out_form():
    from scorch.compiler.loopir.nodes import ReduceOp

    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    scheduled = apply_stack_tile(lowering.program, stack_tile(k.index_id, 4))
    region = region_of(scheduled)
    builder = LoopIRBuilder.resuming(scheduled)
    consumer_point = region.consumer.statements[0]
    copy_out = consumer_point.body.statements[0]
    doubled = builder.store_reduce(
        copy_out.tensor,
        copy_out.indices,
        ReduceOp.ADD,
        builder.binary(
            LoopIRBinaryOp.ADD,
            copy_out.value,
            builder.float_const(0.0),
        ),
    )
    object.__setattr__(consumer_point, "body", builder.block((doubled,)))
    verify_program(scheduled)
    expect_code("unsupported_schedule_shape", erase_schedule, scheduled)


def test_stack_scheduled_iteration_matches_the_base_oracle_exactly():
    """Reset, ragged-tail, and reduction semantics — not just visitation.

    All-ones counting inputs make every output cell count its contributing
    (j, k) pairs exactly (integer floats are exact), so a missing per-tile
    reset, a double copy-out, or a ragged-tail overshoot all change the
    result.
    """

    from scorch.compiler.loopir.levels import CsrMatrix

    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    dense_a = [
        [0.0, 2.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 4.0],
    ]
    csr = CsrMatrix.from_dense(dense_a)
    for n, width in ((2, 4), (4, 4), (6, 4), (5, 4), (0, 4), (6, 9), (6, 1)):
        plan = LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(stack_tile(k.index_id, width),),
        )
        scheduled = apply_schedule_plan(lowering.program, plan)
        b = [[float(1 + r * max(n, 1) + c) for c in range(n)] for r in range(4)]
        inputs = {
            lowering.input_symbols[0]: csr,
            lowering.input_symbols[1]: b,
        }
        shapes = {lowering.result_symbol: (3, n)}
        assert run_program(scheduled.program, inputs, shapes) == run_program(
            lowering.program, inputs, shapes
        )
        erased = erase_schedule(scheduled.program)
        assert run_program(erased, inputs, shapes) == run_program(
            lowering.program, inputs, shapes
        )


def test_stack_scheduled_carrier_rejects_forged_region_state():
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(stack_tile(k.index_id, 4),),
    )
    scheduled = apply_schedule_plan(lowering.program, plan)
    direct_plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(affine_tile(k.index_id, 4),),
    )
    direct = apply_schedule_plan(lowering.program, direct_plan)
    forged = ScheduledLoopIR(
        base_program=scheduled.base_program,
        plan=plan,
        program=direct.program,
        loops=direct.loops,
    )
    expect_code("scheduled_program_mismatch", verify_scheduled_loopir, forged)
    # A base program that already carries a region is not unscheduled.
    nested = ScheduledLoopIR(
        base_program=scheduled.program,
        plan=plan,
        program=scheduled.program,
        loops=scheduled.loops,
    )
    expect_code("scheduled_base_not_unscheduled", verify_scheduled_loopir, nested)


def test_reorder_rejects_region_chains():
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    scheduled = apply_stack_tile(lowering.program, stack_tile(k.index_id, 4))
    expect_code(
        "reorder_split_chain",
        reorder_loops,
        scheduled,
        lowering.loop_index_ids,
    )


# -- sparse panel windows (Phase-6 panel slice) --------------------------------

from scorch.compiler.loop_plan import PanelBound  # noqa: E402
from scorch.compiler.loopir.nodes import (  # noqa: E402
    PanelOuterFor,
    SparseFor,
    SparseWindowFor,
)
from scorch.compiler.loopir.schedule_passes import apply_panel_tile  # noqa: E402


def panel_tile(index_id, width, placement=None, unroll=True):
    # unroll=True mirrors the public TileSpec default; the legacy panel
    # lowering ignores the flag, so the typed family accepts and ignores it
    # the same way.
    return LoopTile(
        loop=LoopRef(index_id),
        width=width,
        placement=placement or outermost_placement(),
        parallel=False,
        kind="panel",
        accumulation="direct",
        unroll=unroll,
    )


def spmm_panel_parts(width=3, placement=None):
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    b_symbol = lowering.input_symbols[1]
    tile = panel_tile(j.index_id, width, placement=placement)
    bound = PanelBound(LoopRef(j.index_id), b_symbol, 0)
    parallel = LoopRef(i.index_id)
    return lowering, (i, j, k), tile, bound, parallel


def test_apply_panel_tile_builds_the_window_shape():
    lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts(width=3)
    windowed = apply_panel_tile(lowering.program, tile, bound, parallel)
    verify_program(windowed)
    assert chain_types(windowed) == [PanelOuterFor, DenseFor, SparseWindowFor, DenseFor]
    panel = windowed.body.statements[0]
    window = panel.body.statements[0].body.statements[0]
    assert panel.tile == window.tile
    assert panel.index == j.index_id == window.coord_index
    assert panel.width == 3
    assert panel.bound_tensor == bound.tensor_id
    assert panel.bound_level == 0
    base_window = lowering.program.body.statements[0].body.statements[0]
    assert window.cursor is base_window.cursor
    assert window.position == base_window.position


def test_apply_panel_tile_is_pure_and_deterministic():
    lowering, _ids, tile, bound, parallel = spmm_panel_parts(width=4)
    before = canonical_program_dump(lowering.program)
    first = apply_panel_tile(lowering.program, tile, bound, parallel)
    second = apply_panel_tile(lowering.program, tile, bound, parallel)
    assert canonical_program_dump(lowering.program) == before
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert print_program(first) == print_program(second)


def test_apply_panel_tile_accepts_and_consumes_both_unroll_values():
    lowering, (_i, j, _k), _tile, bound, parallel = spmm_panel_parts(width=4)
    unrolled = apply_panel_tile(
        lowering.program,
        panel_tile(j.index_id, 4, unroll=True),
        bound,
        parallel,
    )
    not_unrolled = apply_panel_tile(
        lowering.program,
        panel_tile(j.index_id, 4, unroll=False),
        bound,
        parallel,
    )
    verify_program(unrolled)
    verify_program(not_unrolled)
    # Legacy panels ignore the shared TileSpec compatibility field, so it
    # deliberately has no scheduled-LoopIR effect.
    assert canonical_program_dump(unrolled) == canonical_program_dump(not_unrolled)


def test_panel_canonical_dumps_are_stable_across_identity_histories():
    from scorch.compiler.identity import new_index_id, new_symbol_id

    lowering, _ids, tile, bound, parallel = spmm_panel_parts(width=5)
    first = apply_panel_tile(lowering.program, tile, bound, parallel)
    for _ in range(64):
        new_symbol_id()
        new_index_id()
    lowering2, _ids2, tile2, bound2, parallel2 = spmm_panel_parts(width=5)
    second = apply_panel_tile(lowering2.program, tile2, bound2, parallel2)
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert print_program(first) == print_program(second)


def test_apply_panel_tile_child_of_wraps_below_the_affine_origin():
    lowering, (i, j, k), _tile, bound, parallel = spmm_panel_parts()
    tiled = apply_affine_tile(lowering.program, affine_tile(k.index_id, 4))
    tile = panel_tile(
        j.index_id,
        3,
        placement=LoopPlacement(
            PlacementKind.CHILD_OF, parent=LoopRef(k.index_id, LoopPart.OUTER)
        ),
    )
    windowed = apply_panel_tile(tiled, tile, bound, parallel)
    assert chain_types(windowed) == [
        TileOuterFor,
        PanelOuterFor,
        DenseFor,
        SparseWindowFor,
        TileInnerFor,
    ]


def test_apply_panel_tile_target_legality_mirrors_legacy():
    # A dense loop is not a panel target.
    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    expect_code(
        "panel_target_invalid",
        apply_panel_tile,
        lowering.program,
        panel_tile(j.index_id, 3),
        PanelBound(LoopRef(j.index_id), lowering.input_symbols[1], 1),
        LoopRef(i.index_id),
    )

    # Ordered sparse assembly pins its nest.
    cin, (i, j) = build_union_add_to_csr()
    lowering = lower(cin)
    expect_code(
        "panel_target_invalid",
        apply_panel_tile,
        lowering.program,
        panel_tile(j.index_id, 3),
        PanelBound(LoopRef(j.index_id), lowering.input_symbols[0], 1),
        LoopRef(i.index_id),
    )

    # A merged coordinate loop is outside the single-cursor family.
    i2, j2 = IndexVar("i"), IndexVar("j")
    a2 = TensorVar("A", fmt="ds")
    b2 = TensorVar("B", fmt="ds")
    c2 = TensorVar("C", fmt="dd")
    union_dense = ForAll(
        i2,
        ForAll(
            j2,
            TensorAssign(
                c2[i2, j2], CINBinaryOp(Operation.ADD, a2[i2, j2], b2[i2, j2])
            ),
        ),
    )
    lowering = lower(union_dense)
    expect_code(
        "panel_target_invalid",
        apply_panel_tile,
        lowering.program,
        panel_tile(j2.index_id, 3),
        PanelBound(LoopRef(j2.index_id), lowering.input_symbols[0], 1),
        LoopRef(i2.index_id),
    )

    # A missing target and a split target keep the shared tile codes.
    lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts()
    expect_code(
        "tile_target_missing",
        apply_panel_tile,
        lowering.program,
        panel_tile(IndexId(999_999), 3),
        PanelBound(LoopRef(IndexId(999_999)), bound.tensor_id, 0),
        parallel,
    )
    tiled = apply_affine_tile(lowering.program, affine_tile(k.index_id, 4))
    expect_code(
        "tile_target_already_split",
        apply_panel_tile,
        tiled,
        panel_tile(k.index_id, 3),
        PanelBound(LoopRef(k.index_id), bound.tensor_id, 1),
        parallel,
    )

    # A workspace-region chain refuses panel composition.
    lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts()
    stacked = apply_stack_tile(
        lowering.program, affine_tile(k.index_id, 4, accum="stack")
    )
    expect_code(
        "panel_target_invalid", apply_panel_tile, stacked, tile, bound, parallel
    )

    # A second panel refuses composition.
    lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts()
    windowed = apply_panel_tile(lowering.program, tile, bound, parallel)
    expect_code(
        "invalid_schedule_panel", apply_panel_tile, windowed, tile, bound, parallel
    )

    # And the affine/stack passes refuse panel-scheduled chains.
    expect_code(
        "unsupported_schedule_shape",
        apply_affine_tile,
        windowed,
        affine_tile(k.index_id, 4),
    )
    expect_code(
        "unsupported_schedule_shape",
        apply_stack_tile,
        windowed,
        affine_tile(k.index_id, 4, accum="stack"),
    )
    expect_code(
        "reorder_split_chain",
        reorder_loops,
        windowed,
        (j.index_id, i.index_id, k.index_id),
    )


def test_apply_panel_tile_rejects_compressed_parent_windows():
    """DCSR: a window over a compressed-parent cursor is not migrated."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a = builder.new_symbol_id()
    y = builder.new_symbol_id()
    from scorch.compiler.loopir.nodes import LevelKind, ReduceOp, ScalarType

    decl_a = builder.tensor(
        a,
        "A",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.COMPRESSED, 0),
            builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    decl_y = builder.tensor(
        y, "y", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    row_cursor = builder.new_cursor_id()
    col_cursor = builder.new_cursor_id()
    row_position = builder.new_position_id()
    col_position = builder.new_position_id()
    inner = builder.sparse_for(
        builder.sparse_cursor(col_cursor, a, 1, builder.position_value(row_position)),
        col_position,
        col,
        builder.block(
            (
                builder.store_reduce(
                    y,
                    (builder.index_value(row),),
                    ReduceOp.ADD,
                    builder.cursor_value(col_cursor),
                ),
            )
        ),
    )
    outer = builder.sparse_for(
        builder.sparse_cursor(row_cursor, a, 0, builder.root_position()),
        row_position,
        row,
        builder.block((inner,)),
    )
    program = builder.program(
        (dim_i, dim_j), (decl_a, decl_y), (a,), (y,), builder.block((outer,))
    )
    expect_code(
        "panel_nested_compressed",
        apply_panel_tile,
        program,
        panel_tile(col, 3),
        PanelBound(LoopRef(col), a, 1),
        LoopRef(row),
    )


def test_apply_panel_tile_parallel_row_scope():
    lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts()
    expect_code(
        "panel_parallel_scope", apply_panel_tile, lowering.program, tile, bound, None
    )
    expect_code(
        "panel_parallel_scope",
        apply_panel_tile,
        lowering.program,
        tile,
        bound,
        LoopRef(k.index_id),
    )
    expect_code(
        "panel_parallel_scope",
        apply_panel_tile,
        lowering.program,
        tile,
        bound,
        LoopRef(i.index_id, LoopPart.OUTER),
    )

    # child_of below the row loop cannot surround the parallel row loop.
    lowering, (i, j, k), _tile, bound, parallel = spmm_panel_parts()
    tiled = apply_affine_tile(
        lowering.program,
        affine_tile(
            k.index_id,
            4,
            placement=LoopPlacement(PlacementKind.CHILD_OF, parent=LoopRef(i.index_id)),
        ),
    )
    below_row = panel_tile(
        j.index_id,
        3,
        placement=LoopPlacement(
            PlacementKind.CHILD_OF, parent=LoopRef(k.index_id, LoopPart.OUTER)
        ),
    )
    expect_code(
        "panel_parallel_scope", apply_panel_tile, tiled, below_row, bound, parallel
    )


def test_apply_panel_tile_placement_legality():
    lowering, (i, j, k), _tile, bound, parallel = spmm_panel_parts()
    expect_code(
        "panel_placement_invalid",
        apply_panel_tile,
        lowering.program,
        panel_tile(
            j.index_id, 3, placement=LoopPlacement(PlacementKind.AT_DEPTH, depth=0)
        ),
        bound,
        parallel,
    )
    expect_code(
        "panel_placement_invalid",
        apply_panel_tile,
        lowering.program,
        panel_tile(
            j.index_id,
            3,
            placement=LoopPlacement(PlacementKind.CHILD_OF, parent=LoopRef(i.index_id)),
        ),
        bound,
        parallel,
    )
    # child_of an affine origin that is not part of the chain.
    expect_code(
        "panel_placement_invalid",
        apply_panel_tile,
        lowering.program,
        panel_tile(
            j.index_id,
            3,
            placement=LoopPlacement(
                PlacementKind.CHILD_OF, parent=LoopRef(k.index_id, LoopPart.OUTER)
            ),
        ),
        bound,
        parallel,
    )


def test_apply_panel_tile_consumes_a_consistent_bound():
    from scorch.compiler.identity import SymbolId

    lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts()
    expect_code(
        "invalid_schedule_panel",
        apply_panel_tile,
        lowering.program,
        tile,
        PanelBound(LoopRef(k.index_id), bound.tensor_id, 0),
        parallel,
    )
    expect_code(
        "panel_bound_mismatch",
        apply_panel_tile,
        lowering.program,
        tile,
        PanelBound(LoopRef(j.index_id), SymbolId(999_998), 0),
        parallel,
    )
    # A's level 1 stores j but is compressed.
    expect_code(
        "panel_bound_mismatch",
        apply_panel_tile,
        lowering.program,
        tile,
        PanelBound(LoopRef(j.index_id), lowering.input_symbols[0], 1),
        parallel,
    )
    # B's level 1 is dense but stores k.
    expect_code(
        "panel_bound_mismatch",
        apply_panel_tile,
        lowering.program,
        tile,
        PanelBound(LoopRef(j.index_id), bound.tensor_id, 1),
        parallel,
    )
    expect_code(
        "panel_bound_mismatch",
        apply_panel_tile,
        lowering.program,
        tile,
        PanelBound(LoopRef(j.index_id), bound.tensor_id, 7),
        parallel,
    )


def test_apply_panel_tile_rejects_malformed_tiles_and_widths():
    lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts()
    expect_code(
        "invalid_schedule_panel",
        apply_panel_tile,
        lowering.program,
        affine_tile(j.index_id, 3),
        bound,
        parallel,
    )
    stack_panel = LoopTile(
        loop=LoopRef(j.index_id),
        width=3,
        placement=outermost_placement(),
        parallel=False,
        kind="panel",
        accumulation="stack",
        unroll=False,
    )
    expect_code(
        "invalid_schedule_panel",
        apply_panel_tile,
        lowering.program,
        stack_panel,
        bound,
        parallel,
    )
    parallel_panel = LoopTile(
        loop=LoopRef(j.index_id),
        width=3,
        placement=outermost_placement(),
        parallel=True,
        kind="panel",
        accumulation="direct",
        unroll=False,
    )
    expect_code(
        "invalid_schedule_panel",
        apply_panel_tile,
        lowering.program,
        parallel_panel,
        bound,
        parallel,
    )
    expect_code(
        "tile_invalid_width",
        apply_panel_tile,
        lowering.program,
        panel_tile(j.index_id, MAX_AFFINE_TILE_WIDTH + 1),
        bound,
        parallel,
    )
    forged = panel_tile(j.index_id, 3)
    object.__delattr__(forged, "width")
    expect_code(
        "invalid_schedule_tile",
        apply_panel_tile,
        lowering.program,
        forged,
        bound,
        parallel,
    )
    forged_bound = PanelBound(LoopRef(j.index_id), bound.tensor_id, 0)
    object.__setattr__(forged_bound, "level", "0")
    expect_code(
        "invalid_schedule_tile",
        apply_panel_tile,
        lowering.program,
        tile,
        forged_bound,
        parallel,
    )


def panel_plan(order, tile, bound, parallel):
    return LoopPlan(
        loop_order=order,
        tiles=(tile,) if not isinstance(tile, tuple) else tile,
        panel_bounds=(bound,),
        parallel_loop=parallel,
        provenance="explicit",
        tag="panel",
    )


def test_panel_plan_applies_through_apply_schedule_plan():
    lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts()
    order = lowering.loop_index_ids
    artifact = apply_schedule_plan(
        lowering.program, panel_plan(order, tile, bound, parallel)
    )
    verify_scheduled_loopir(artifact)
    assert chain_types(artifact.program) == [
        PanelOuterFor,
        DenseFor,
        SparseWindowFor,
        DenseFor,
    ]
    parts = [(prov.part, prov.tile is not None) for prov in artifact.loops]
    assert parts == [
        (LoopPart.OUTER, True),
        (LoopPart.LOGICAL, False),
        (LoopPart.INNER, True),
        (LoopPart.LOGICAL, False),
    ]
    assert artifact.loops[0].index == j.index_id
    assert artifact.loops[2].index == j.index_id


def test_panel_plan_composes_with_direct_tiles():
    lowering, (i, j, k), _tile, bound, parallel = spmm_panel_parts()
    order = lowering.loop_index_ids
    plan = LoopPlan(
        loop_order=order,
        tiles=(
            affine_tile(k.index_id, 4),
            panel_tile(
                j.index_id,
                3,
                placement=LoopPlacement(
                    PlacementKind.CHILD_OF, parent=LoopRef(k.index_id, LoopPart.OUTER)
                ),
            ),
        ),
        panel_bounds=(bound,),
        parallel_loop=parallel,
        provenance="explicit",
        tag="panel-child",
    )
    artifact = apply_schedule_plan(lowering.program, plan)
    verify_scheduled_loopir(artifact)
    assert chain_types(artifact.program) == [
        TileOuterFor,
        PanelOuterFor,
        DenseFor,
        SparseWindowFor,
        TileInnerFor,
    ]


def test_panel_plan_gate_failures():
    lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts()
    order = lowering.loop_index_ids

    # Two panels.
    two = LoopPlan(
        loop_order=order,
        tiles=(tile, panel_tile(k.index_id, 2)),
        panel_bounds=(bound, PanelBound(LoopRef(k.index_id), bound.tensor_id, 1)),
        parallel_loop=parallel,
    )
    expect_code("invalid_schedule_panel", apply_schedule_plan, lowering.program, two)

    # Panel before an affine tile.
    out_of_order = LoopPlan(
        loop_order=order,
        tiles=(tile, affine_tile(k.index_id, 4)),
        panel_bounds=(bound,),
        parallel_loop=parallel,
    )
    expect_code(
        "invalid_schedule_panel", apply_schedule_plan, lowering.program, out_of_order
    )

    # Bound-correspondence failures.
    for bounds in ((), (PanelBound(LoopRef(k.index_id), bound.tensor_id, 1),)):
        broken = LoopPlan(
            loop_order=order,
            tiles=(tile,),
            panel_bounds=bounds,
            parallel_loop=parallel,
        )
        expect_code(
            "invalid_schedule_panel", apply_schedule_plan, lowering.program, broken
        )

    # The mandatory parallel row loop.
    unparallel = LoopPlan(
        loop_order=order, tiles=(tile,), panel_bounds=(bound,), parallel_loop=None
    )
    expect_code(
        "panel_parallel_scope", apply_schedule_plan, lowering.program, unparallel
    )
    split_parallel = LoopPlan(
        loop_order=order,
        tiles=(tile,),
        panel_bounds=(bound,),
        parallel_loop=LoopRef(i.index_id, LoopPart.OUTER),
    )
    expect_code(
        "panel_parallel_scope", apply_schedule_plan, lowering.program, split_parallel
    )

    # child_of placements must name an outermost-placed affine tile.
    nested_parent = LoopPlan(
        loop_order=order,
        tiles=(
            affine_tile(
                k.index_id,
                4,
                placement=LoopPlacement(
                    PlacementKind.CHILD_OF, parent=LoopRef(i.index_id)
                ),
            ),
            panel_tile(
                j.index_id,
                3,
                placement=LoopPlacement(
                    PlacementKind.CHILD_OF, parent=LoopRef(k.index_id, LoopPart.OUTER)
                ),
            ),
        ),
        panel_bounds=(bound,),
        parallel_loop=parallel,
    )
    expect_code(
        "panel_placement_invalid", apply_schedule_plan, lowering.program, nested_parent
    )
    at_depth = LoopPlan(
        loop_order=order,
        tiles=(
            panel_tile(
                j.index_id, 3, placement=LoopPlacement(PlacementKind.AT_DEPTH, depth=0)
            ),
        ),
        panel_bounds=(bound,),
        parallel_loop=parallel,
    )
    expect_code(
        "panel_placement_invalid", apply_schedule_plan, lowering.program, at_depth
    )


def test_panel_erasure_restores_the_base_program():
    lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts()
    windowed = apply_panel_tile(lowering.program, tile, bound, parallel)
    erased = erase_schedule(windowed)
    assert canonical_program_dump(erased) == canonical_program_dump(lowering.program)
    assert chain_types(erased) == [DenseFor, SparseFor, DenseFor]

    tiled = apply_affine_tile(lowering.program, affine_tile(k.index_id, 4))
    both = apply_panel_tile(
        tiled,
        panel_tile(
            j.index_id,
            3,
            placement=LoopPlacement(
                PlacementKind.CHILD_OF, parent=LoopRef(k.index_id, LoopPart.OUTER)
            ),
        ),
        bound,
        parallel,
    )
    erased_both = erase_schedule(both)
    assert canonical_program_dump(erased_both) == canonical_program_dump(
        lowering.program
    )


def test_panel_scheduled_iteration_matches_the_base_oracle_exactly():
    """Exact integer-float counting across ragged, unit, exact, oversized,
    and huge widths, with an empty CSR row and disjoint supports: a missed
    or doubled window entry changes the counted result."""

    from scorch.compiler.loopir.levels import CsrMatrix

    a_dense = [
        [1.0, 0.0, 2.0, 0.0, 3.0],
        [0.0] * 5,
        [4.0, 5.0, 0.0, 0.0, 6.0],
        [0.0, 0.0, 0.0, 7.0, 0.0],
    ]
    b_dense = [[float(1 + r * 6 + c) for c in range(6)] for r in range(5)]
    for width in (1, 2, 3, 5, 7, MAX_AFFINE_TILE_WIDTH):
        lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts(width=width)
        base_out = run_program(
            lowering.program,
            {
                lowering.input_symbols[0]: CsrMatrix.from_dense(a_dense),
                lowering.input_symbols[1]: b_dense,
            },
            {lowering.result_symbol: (4, 6)},
        )
        windowed = apply_panel_tile(lowering.program, tile, bound, parallel)
        panel_out = run_program(
            windowed,
            {
                lowering.input_symbols[0]: CsrMatrix.from_dense(a_dense),
                lowering.input_symbols[1]: b_dense,
            },
            {lowering.result_symbol: (4, 6)},
        )
        assert panel_out == base_out


def test_panel_scheduled_oracle_matches_base_on_randomized_dimensions():
    from scorch.compiler.loopir.levels import CsrMatrix

    rng = random.Random(629)
    for _ in range(6):
        rows = rng.randrange(1, 6)
        inner = rng.randrange(1, 7)
        cols = rng.randrange(1, 5)
        width = rng.randrange(1, inner + 3)
        a_dense = [
            [
                float(rng.randrange(-3, 4)) if rng.random() < 0.5 else 0.0
                for _ in range(inner)
            ]
            for _ in range(rows)
        ]
        b_dense = [
            [float(rng.randrange(-3, 4)) for _ in range(cols)] for _ in range(inner)
        ]
        lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts(width=width)
        inputs = {
            lowering.input_symbols[0]: CsrMatrix.from_dense(a_dense),
            lowering.input_symbols[1]: b_dense,
        }
        shapes = {lowering.result_symbol: (rows, cols)}
        base_out = run_program(lowering.program, inputs, shapes)
        windowed = apply_panel_tile(lowering.program, tile, bound, parallel)
        assert run_program(windowed, inputs, shapes) == base_out


def test_panel_scheduled_carrier_rejects_forged_panel_state():
    lowering, (i, j, k), tile, bound, parallel = spmm_panel_parts()
    order = lowering.loop_index_ids
    artifact = apply_schedule_plan(
        lowering.program, panel_plan(order, tile, bound, parallel)
    )

    # Forged provenance: the panel origin's tile deleted.
    forged_loops = (
        replace(artifact.loops[0], tile=None),
        *artifact.loops[1:],
    )
    forged = ScheduledLoopIR(
        base_program=artifact.base_program,
        plan=artifact.plan,
        program=artifact.program,
        loops=forged_loops,
    )
    expect_code("scheduled_provenance_mismatch", verify_scheduled_loopir, forged)

    # A base program that already carries the panel is not unscheduled.
    forged_base = ScheduledLoopIR(
        base_program=artifact.program,
        plan=artifact.plan,
        program=artifact.program,
        loops=artifact.loops,
    )
    expect_code("scheduled_base_not_unscheduled", verify_scheduled_loopir, forged_base)

    # A plan whose panel width disagrees with the program cannot replay.
    wrong_plan = panel_plan(order, panel_tile(j.index_id, 9), bound, parallel)
    mismatched = ScheduledLoopIR(
        base_program=artifact.base_program,
        plan=wrong_plan,
        program=artifact.program,
        loops=artifact.loops,
    )
    expect_code("scheduled_program_mismatch", verify_scheduled_loopir, mismatched)
