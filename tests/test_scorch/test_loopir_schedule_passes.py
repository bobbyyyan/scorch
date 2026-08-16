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
    AUTO_ORIGIN_POLICY_SCHEMA,
    MAX_AFFINE_TILE_WIDTH,
    AutoOriginPolicy,
    LoopPart,
    LoopPlacement,
    LoopPlan,
    LoopRef,
    LoopTile,
    PlacementKind,
    WorkspaceInsertion,
    verify_loop_plan,
)
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import lower_normalized_cin_to_loopir
from scorch.compiler.loopir.nodes import (
    BinaryOp as LoopIRBinaryOp,
    Block,
    DenseFor,
    LevelKind,
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
    apply_sparse_workspace,
    apply_stack_tile,
    erase_schedule,
    reorder_loops,
    select_parallel_loop,
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


def build_dcsr_sparse_matmul():
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    a = TensorVar("A", fmt="ss")
    b = TensorVar("B", fmt="ss")
    c = TensorVar("C", fmt="ss")
    assign = TensorAssign(
        c[i, j],
        CINBinaryOp(Operation.MUL, a[i, k], b[k, j]),
        op=Operation.ADD,
    )
    return ForAll(i, ForAll(k, ForAll(j, assign))), (i, k, j)


def sparse_workspace_fixture():
    cin, (i, k, j) = build_dcsr_sparse_matmul()
    lowering = lower(cin)
    workspace = WorkspaceInsertion(
        reduction_loop=LoopRef(k.index_id),
        axis_loops=(LoopRef(j.index_id),),
        dense=False,
    )
    return lowering, workspace, (i, k, j)


def expect_code(code, call, *args, **kwargs):
    with pytest.raises(SchedulePassError) as error:
        call(*args, **kwargs)
    assert error.value.defect.code == code, error.value.defect
    return error.value.defect


def test_sparse_workspace_pass_preserves_semantics_and_erases_to_source():
    lowering, workspace, _ = sparse_workspace_fixture()
    scheduled = apply_sparse_workspace(lowering.program, workspace)
    assert canonical_program_dump(erase_schedule(scheduled)) == canonical_program_dump(
        lowering.program
    )

    left = [
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 3.0],
    ]
    right = [
        [1.0, 4.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 5.0, 2.0],
    ]
    kinds = (LevelKind.COMPRESSED, LevelKind.COMPRESSED)
    inputs = {
        lowering.program.inputs[0]: LevelTensorStorage.from_dense(
            left,
            (3, 3),
            (0, 1),
            kinds,
        ),
        lowering.program.inputs[1]: LevelTensorStorage.from_dense(
            right,
            (3, 3),
            (0, 1),
            kinds,
        ),
    }
    output_shapes = {lowering.program.outputs[0]: (3, 3)}
    base_result = run_program(lowering.program, inputs, output_shapes)
    scheduled_result = run_program(scheduled, inputs, output_shapes)
    assert scheduled_result == base_result
    # Two contributions cancel exactly at (0, 0); sparse workspace and
    # semantic accumulation both retain the explicit zero.
    assert 0.0 in scheduled_result[lowering.program.outputs[0]].values


@pytest.mark.parametrize("key_rank", [1, 2, 3])
def test_root_anchored_rank_k_sparse_workspace_erases_and_executes(key_rank):
    """A region above every producer loop has an intentionally empty prefix."""

    from tests.test_scorch.test_loopir_verifier import (
        build_rank_k_workspace_program,
    )

    _, scheduled, _, _ = build_rank_k_workspace_program(key_rank)
    erased = erase_schedule(scheduled)
    verify_program(erased)

    assert "sparse_workspace" not in print_program(erased)
    assert erase_schedule(erased) is erased

    shape = (2, 3, 2)[:key_rank]
    output_shapes = {scheduled.outputs[0]: shape}
    assert run_program(scheduled, {}, output_shapes) == run_program(
        erased, {}, output_shapes
    )


def test_root_rank_k_erasure_preserves_rotated_keys_and_contraction():
    """A non-symmetric reduction distinguishes key order from loop order."""

    from scorch.compiler.loopir.schedule_passes import _chain_provenance
    from tests.test_scorch.test_loopir_oracle import _level_entries
    from tests.test_scorch.test_loopir_verifier import (
        build_rank_k_workspace_program,
    )

    _, scheduled, _, drain = build_rank_k_workspace_program(
        2, key_permutation=(1, 0), contraction_extent=4
    )
    erased = erase_schedule(scheduled)
    provenance = _chain_provenance(scheduled)

    # Producer i/j/contraction loops execute first, followed by the one
    # composite drain loop, whose scheduling identity is its innermost key.
    assert len(provenance) == 4
    assert provenance[-1].index == drain.indices[-1]

    operand = [
        [
            [float(i * 100 + j * 10 + reduction) for reduction in range(4)]
            for j in range(3)
        ]
        for i in range(2)
    ]
    bindings = {scheduled.inputs[0]: operand}
    shapes = {scheduled.outputs[0]: (3, 2)}
    scheduled_result = run_program(scheduled, bindings, shapes)[scheduled.outputs[0]]
    erased_result = run_program(erased, bindings, shapes)[erased.outputs[0]]
    expected = [((j, i), sum(operand[i][j])) for j in range(3) for i in range(2)]

    assert scheduled_result == erased_result
    assert _level_entries(scheduled_result) == expected


def test_sparse_workspace_pass_handles_all_empty_inputs():
    lowering, workspace, _ = sparse_workspace_fixture()
    scheduled = apply_sparse_workspace(lowering.program, workspace)
    empty = LevelTensorStorage.from_dense(
        [[], []],
        (2, 0),
        (0, 1),
        (LevelKind.COMPRESSED, LevelKind.COMPRESSED),
    )
    result = run_program(
        scheduled,
        {
            lowering.program.inputs[0]: empty,
            lowering.program.inputs[1]: LevelTensorStorage.from_dense(
                [],
                (0, 4),
                (0, 1),
                (LevelKind.COMPRESSED, LevelKind.COMPRESSED),
            ),
        },
        {lowering.program.outputs[0]: (2, 4)},
    )
    assert result[lowering.program.outputs[0]].values == ()


def test_sparse_workspace_print_and_canonical_forms_are_structurally_active():
    lowering, workspace, _ = sparse_workspace_fixture()
    scheduled = apply_sparse_workspace(lowering.program, workspace)

    rendered = print_program(scheduled)
    assert "sparse_workspace_region" in rendered
    assert "sparse_workspace_insert(add)" in rendered
    assert "sparse_workspace_drain_for" in rendered
    assert "drained(w0)" in rendered

    canonical = canonical_program_dump(scheduled)
    assert '"schema":"scorch.loopir.canonical.v13"' in canonical
    assert '"kind":"sparse_workspace_region"' in canonical
    assert '"kind":"sparse_workspace_insert"' in canonical
    assert '"kind":"sparse_workspace_drain_for"' in canonical
    assert '"kind":"sparse_workspace_value"' in canonical
    assert "wksp" not in canonical
    assert canonical == canonical_program_dump(
        apply_sparse_workspace(lowering.program, workspace)
    )


def test_sparse_workspace_pass_validates_fact_and_exact_family():
    lowering, workspace, (_, _, j) = sparse_workspace_fixture()
    malformed = WorkspaceInsertion(
        workspace.reduction_loop,
        workspace.axis_loops,
        workspace.dense,
    )
    object.__setattr__(malformed, "axis_loops", None)
    expect_code(
        "invalid_schedule_plan",
        apply_sparse_workspace,
        lowering.program,
        malformed,
    )

    wrong_role = WorkspaceInsertion(
        reduction_loop=LoopRef(j.index_id),
        axis_loops=(LoopRef(j.index_id),),
        dense=False,
    )
    expect_code(
        "sparse_workspace_target_invalid",
        apply_sparse_workspace,
        lowering.program,
        wrong_role,
    )


def test_sparse_workspace_schedule_state_cannot_be_reordered_or_rebased():
    lowering, workspace, (i, k, j) = sparse_workspace_fixture()
    scheduled = apply_sparse_workspace(lowering.program, workspace)
    expect_code(
        "reorder_split_chain",
        reorder_loops,
        scheduled,
        (i.index_id, k.index_id, j.index_id),
    )
    artifact = ScheduledLoopIR(
        base_program=scheduled,
        plan=LoopPlan(loop_order=(i.index_id,)),
        program=scheduled,
        loops=(),
    )
    expect_code("scheduled_base_not_unscheduled", verify_scheduled_loopir, artifact)


@pytest.mark.parametrize("regblock_enabled", (False, True))
def test_sparse_workspace_scheduled_carrier_replays_deterministically(
    regblock_enabled,
):
    lowering, workspace, _ = sparse_workspace_fixture()
    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        workspace=workspace,
        auto_policy=AutoOriginPolicy(
            schema=AUTO_ORIGIN_POLICY_SCHEMA,
            regblock_enabled=regblock_enabled,
            tile_width=8 if regblock_enabled else 32,
        ),
        provenance="auto",
    )

    artifact = apply_schedule_plan(lowering.program, plan)
    verify_scheduled_loopir(artifact)
    replay = apply_schedule_plan(lowering.program, plan)
    assert replay.program == artifact.program
    assert canonical_program_dump(replay.program) == canonical_program_dump(
        artifact.program
    )


def test_sparse_workspace_rejects_an_extra_reduction_loop():
    lowering, workspace, _ = sparse_workspace_fixture()
    builder = LoopIRBuilder.resuming(lowering.program)
    extra_index = builder.new_index_id()
    extra_loop = builder.dense_for(
        extra_index,
        lowering.program.dimensions[0].dimension,
        lowering.program.body,
    )
    extra_program = replace(
        lowering.program,
        body=builder.block((extra_loop,)),
    )
    verify_program(extra_program)

    expect_code(
        "sparse_workspace_target_invalid",
        apply_sparse_workspace,
        extra_program,
        workspace,
    )


def test_merge_descent_position_prevents_child_reorder():
    lowering, _, (i, k, j) = sparse_workspace_fixture()
    expect_code(
        "reorder_sparse_dependency",
        reorder_loops,
        lowering.program,
        (i.index_id, j.index_id, k.index_id),
    )


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
    # A "heap" tile's split geometry is the direct split; the heap
    # accumulation fact is consumed by apply_result_tile, and the plan
    # gate refuses a heap tile with no result-tile fact to pair it.
    heap_split = apply_affine_tile(
        lowering.program, affine_tile(j.index_id, 4, accum="heap")
    )
    assert chain_types(heap_split) == chain_types(
        apply_affine_tile(lowering.program, affine_tile(j.index_id, 4))
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


@pytest.mark.parametrize("provenance", ["tuned", "fallback", "prebuilt", "AUTO"])
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


def test_apply_schedule_plan_admits_the_tile_free_auto_replay_contract():
    """A verified tile-free auto-replay order is a migrated plan family."""

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    cin = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                c[i, j],
                CINBinaryOp(Operation.MUL, a[i, j], b[i, j]),
            ),
        ),
    )
    lowering = lower(cin)
    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        auto_policy=AutoOriginPolicy(
            schema=AUTO_ORIGIN_POLICY_SCHEMA,
            regblock_enabled=False,
            tile_width=32,
        ),
        provenance="auto",
    )
    assert verify_loop_plan(cin, plan) is plan
    schedule = apply_schedule_plan(lowering.program, plan)
    assert schedule.plan is plan
    verify_program(schedule.program)


def test_apply_schedule_plan_enforces_auto_policy_provenance():
    """The consuming pass preserves the LoopPlan policy trust boundary."""

    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    policy = AutoOriginPolicy(
        schema=AUTO_ORIGIN_POLICY_SCHEMA,
        regblock_enabled=False,
        tile_width=32,
    )
    expect_code(
        "auto_origin_policy",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(loop_order=lowering.loop_index_ids, provenance="auto"),
    )
    expect_code(
        "auto_policy_provenance",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            auto_policy=policy,
            provenance="explicit",
        ),
    )


def test_apply_schedule_plan_rejects_the_tiled_auto_family():
    """Recorded heuristic tiles/workspace stay on the legacy path."""

    from scorch.compiler.loop_plan import WorkspaceInsertion

    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    order = lowering.loop_index_ids
    tiled = LoopPlan(
        loop_order=order,
        tiles=(
            LoopTile(
                loop=LoopRef(j.index_id),
                width=32,
                placement=LoopPlacement(PlacementKind.OUTERMOST),
                parallel=False,
                kind="affine",
                accumulation="direct",
                unroll=True,
            ),
        ),
        workspace=WorkspaceInsertion(
            reduction_loop=LoopRef(k.index_id),
            axis_loops=(LoopRef(j.index_id),),
            dense=True,
        ),
        auto_policy=AutoOriginPolicy(
            schema=AUTO_ORIGIN_POLICY_SCHEMA,
            regblock_enabled=False,
            tile_width=32,
        ),
        provenance="auto",
    )
    expect_code(
        "unsupported_schedule_auto_family",
        apply_schedule_plan,
        lowering.program,
        tiled,
    )
    # The tile-free sparse-workspace fact is now an admitted automatic
    # family; on this dense-output program the pass itself still fails
    # closed with its structural target code.
    workspace_only = LoopPlan(
        loop_order=order,
        workspace=WorkspaceInsertion(
            reduction_loop=LoopRef(k.index_id),
            axis_loops=(LoopRef(j.index_id),),
            dense=False,
        ),
        auto_policy=AutoOriginPolicy(
            schema=AUTO_ORIGIN_POLICY_SCHEMA,
            regblock_enabled=False,
            tile_width=32,
        ),
        provenance="auto",
    )
    expect_code(
        "sparse_workspace_target_invalid",
        apply_schedule_plan,
        lowering.program,
        workspace_only,
    )


def _stack_form_auto_plan(lowering, i, k, j, **overrides):
    """The admitted regblock stack-form automatic plan for matmul_ikj."""

    tile = LoopTile(
        loop=LoopRef(j.index_id),
        width=8,
        placement=LoopPlacement(PlacementKind.CHILD_OF, parent=LoopRef(i.index_id)),
        parallel=False,
        kind="affine",
        accumulation="direct",
        unroll=True,
    )
    fields = dict(
        loop_order=lowering.loop_index_ids,
        tiles=(tile,),
        workspace=WorkspaceInsertion(
            reduction_loop=LoopRef(k.index_id),
            axis_loops=(LoopRef(j.index_id),),
            dense=True,
        ),
        auto_policy=AutoOriginPolicy(
            schema=AUTO_ORIGIN_POLICY_SCHEMA,
            regblock_enabled=True,
            tile_width=8,
        ),
        provenance="auto",
    )
    fields.update(overrides)
    return LoopPlan(**fields)


def test_apply_schedule_plan_admits_the_regblock_stack_form():
    """The measured workspace+tile pair lowers through the stack-tile pass."""

    from scorch.compiler.loopir.nodes import WorkspaceRegion
    from scorch.compiler.loopir.printer import canonical_program_dump
    from scorch.compiler.loopir.schedule_passes import erase_schedule

    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    plan = _stack_form_auto_plan(lowering, i, k, j)
    schedule = apply_schedule_plan(lowering.program, plan)
    assert schedule.plan is plan
    verify_program(schedule.program)
    # The plan facts stay untouched: the recorded tile keeps its direct
    # accumulation while the program carries the stack workspace region.
    assert schedule.plan.tiles[0].accumulation == "direct"

    def find_region(node):
        stack = [node]
        while stack:
            current = stack.pop()
            if type(current) is WorkspaceRegion:
                return current
            stack.extend(getattr(current, "statements", ()))
            for attribute in ("body", "producer", "consumer"):
                child = getattr(current, attribute, None)
                if child is not None:
                    stack.append(child)
        return None

    assert find_region(schedule.program.body) is not None
    assert canonical_program_dump(
        erase_schedule(schedule.program)
    ) == canonical_program_dump(schedule.base_program)


def test_regblock_stack_form_shape_is_exact():
    """Every deviation from the measured stack form stays fail-closed."""

    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    base = _stack_form_auto_plan(lowering, i, k, j)

    def rejected(**overrides):
        expect_code(
            "unsupported_schedule_auto_family",
            apply_schedule_plan,
            lowering.program,
            _stack_form_auto_plan(lowering, i, k, j, **overrides),
        )

    expect_code(
        "auto_origin_policy",
        apply_schedule_plan,
        lowering.program,
        _stack_form_auto_plan(lowering, i, k, j, auto_policy=None),
    )
    rejected(
        auto_policy=AutoOriginPolicy(
            schema=AUTO_ORIGIN_POLICY_SCHEMA,
            regblock_enabled=False,
            tile_width=8,
        )
    )
    rejected(tiles=(replace(base.tiles[0], width=16),))
    rejected(tiles=(replace(base.tiles[0], unroll=False),))
    rejected(
        tiles=(
            replace(base.tiles[0], placement=LoopPlacement(PlacementKind.OUTERMOST)),
        )
    )
    rejected(
        tiles=(
            replace(
                base.tiles[0],
                placement=LoopPlacement(
                    PlacementKind.CHILD_OF, parent=LoopRef(k.index_id)
                ),
            ),
        )
    )
    rejected(workspace=replace(base.workspace, dense=False))
    rejected(workspace=replace(base.workspace, axis_loops=(LoopRef(k.index_id),)))
    rejected(workspace=None)


def _reduce_out_auto_plan(lowering, i, k, j, *, regblock_enabled, **overrides):
    """The admitted dense reduce-out automatic plan for matmul_ikj."""

    width = 8 if regblock_enabled else 32
    placement = (
        LoopPlacement(PlacementKind.CHILD_OF, parent=LoopRef(i.index_id))
        if regblock_enabled
        else LoopPlacement(PlacementKind.OUTERMOST)
    )

    def tile(loop):
        return LoopTile(
            loop=LoopRef(loop.index_id),
            width=width,
            placement=placement,
            parallel=False,
            kind="affine",
            accumulation="direct",
            unroll=True,
        )

    fields = dict(
        loop_order=lowering.loop_index_ids,
        tiles=(tile(k), tile(j)),
        workspace=WorkspaceInsertion(
            reduction_loop=LoopRef(k.index_id),
            axis_loops=(LoopRef(j.index_id),),
            dense=True,
        ),
        auto_policy=AutoOriginPolicy(
            schema=AUTO_ORIGIN_POLICY_SCHEMA,
            regblock_enabled=regblock_enabled,
            tile_width=width,
        ),
        provenance="auto",
    )
    fields.update(overrides)
    return LoopPlan(**fields)


@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_apply_schedule_plan_admits_the_reduce_out_form(regblock_enabled):
    """The strip-mined reduction plus copy-out region is a migrated family."""

    from scorch.compiler.loopir.nodes import (
        TileInnerFor,
        TileOuterFor,
        WorkspaceRegion,
    )
    from scorch.compiler.loopir.printer import canonical_program_dump
    from scorch.compiler.loopir.schedule_passes import erase_schedule

    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    plan = _reduce_out_auto_plan(lowering, i, k, j, regblock_enabled=regblock_enabled)
    schedule = apply_schedule_plan(lowering.program, plan)
    assert schedule.plan is plan
    verify_program(schedule.program)

    # Chain shape: the axis origin stacks LIFO above the reduction origin
    # (legacy add_tile order), around the arm placement anchor.
    chain = []
    node = schedule.program.body.statements[0]
    while type(node) in (DenseFor, TileOuterFor):
        chain.append(node)
        node = node.body.statements[0]
    kinds = [type(entry).__name__ for entry in chain]
    if regblock_enabled:
        assert kinds == ["DenseFor", "TileOuterFor", "TileOuterFor"]
        assert chain[1].index == j.index_id
        assert chain[2].index == k.index_id
    else:
        assert kinds == ["TileOuterFor", "TileOuterFor", "DenseFor"]
        assert chain[0].index == j.index_id
        assert chain[1].index == k.index_id
    region = node
    assert type(region) is WorkspaceRegion
    producer_top = region.producer.statements[0]
    assert type(producer_top) is TileInnerFor
    assert producer_top.index == k.index_id
    producer_point = producer_top.body.statements[0]
    assert type(producer_point) is TileInnerFor
    assert producer_point.index == j.index_id
    consumer_point = region.consumer.statements[0]
    assert type(consumer_point) is TileInnerFor
    assert consumer_point.index == j.index_id

    assert canonical_program_dump(
        erase_schedule(schedule.program)
    ) == canonical_program_dump(schedule.base_program)


@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize(
    "extent_class",
    [
        "zero_reduction",
        "zero_axis",
        "unit",
        "exact",
        "ragged",
        "oversized",
    ],
)
def test_reduce_out_oracle_executes_every_extent_class(regblock_enabled, extent_class):
    """Reset, producer, and copy-out semantics hold across tile boundaries."""

    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    plan = _reduce_out_auto_plan(lowering, i, k, j, regblock_enabled=regblock_enabled)
    artifact = apply_schedule_plan(lowering.program, plan)
    assert plan.auto_policy is not None
    width = plan.auto_policy.tile_width
    extents = {
        "zero_reduction": (0, 5),
        "zero_axis": (5, 0),
        "unit": (1, 1),
        "exact": (width, width),
        "ragged": (width + 1, width + 3),
        "oversized": (3, 5),
    }
    inner, cols = extents[extent_class]
    rows = 2
    a_values = [
        [float((row * 3 + red) % 7 - 3) for red in range(inner)] for row in range(rows)
    ]
    b_values = [
        [float((red * 5 + col) % 9 - 4) for col in range(cols)] for red in range(inner)
    ]
    bindings = {
        lowering.input_symbols[0]: a_values,
        lowering.input_symbols[1]: b_values,
    }
    shapes = {lowering.result_symbol: (rows, cols)}

    scheduled = run_program(artifact.program, bindings, shapes)
    base = run_program(artifact.base_program, bindings, shapes)
    erased_program = erase_schedule(artifact.program)
    erased = run_program(erased_program, bindings, shapes)
    expected = [
        [
            float(sum(a_values[row][red] * b_values[red][col] for red in range(inner)))
            for col in range(cols)
        ]
        for row in range(rows)
    ]
    assert scheduled == base == erased
    assert scheduled[lowering.result_symbol] == expected
    assert canonical_program_dump(erased_program) == canonical_program_dump(
        artifact.base_program
    )


@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_reduce_out_oracle_covers_multiple_reductions_and_four_tiles(
    regblock_enabled,
):
    """A rank-three output exercises prefix, two reductions, and axis tiles."""

    a, b, d, e, c = (IndexVar(name) for name in ("a", "b", "d", "e", "c"))
    left = TensorVar("A", fmt="dddd")
    right = TensorVar("B", fmt="ddd")
    result = TensorVar("C", fmt="ddd")
    cin = ForAll(
        a,
        ForAll(
            b,
            ForAll(
                d,
                ForAll(
                    e,
                    ForAll(
                        c,
                        TensorAssign(
                            result[a, b, c],
                            CINBinaryOp(
                                Operation.MUL,
                                left[a, b, d, e],
                                right[d, e, c],
                            ),
                            op=Operation.ADD,
                        ),
                    ),
                ),
            ),
        ),
    )
    lowering = lower(cin)
    width = 8 if regblock_enabled else 32
    placement = (
        LoopPlacement(PlacementKind.CHILD_OF, parent=LoopRef(a.index_id))
        if regblock_enabled
        else LoopPlacement(PlacementKind.OUTERMOST)
    )

    def tile(loop):
        return LoopTile(
            loop=LoopRef(loop.index_id),
            width=width,
            placement=placement,
            parallel=False,
            kind="affine",
            accumulation="direct",
            unroll=True,
        )

    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=tuple(tile(loop) for loop in (b, d, e, c)),
        workspace=WorkspaceInsertion(
            reduction_loop=LoopRef(e.index_id),
            axis_loops=(LoopRef(c.index_id),),
            dense=True,
        ),
        auto_policy=AutoOriginPolicy(
            schema=AUTO_ORIGIN_POLICY_SCHEMA,
            regblock_enabled=regblock_enabled,
            tile_width=width,
        ),
        provenance="auto",
    )
    artifact = apply_schedule_plan(lowering.program, plan)
    batch, rows, first_reduction, second_reduction, cols = (2, 3, 4, 5, 6)
    left_values = [
        [
            [
                [
                    float((batch_pos + row + first + second) % 7 - 3)
                    for second in range(second_reduction)
                ]
                for first in range(first_reduction)
            ]
            for row in range(rows)
        ]
        for batch_pos in range(batch)
    ]
    right_values = [
        [
            [float((first * 3 + second + col) % 9 - 4) for col in range(cols)]
            for second in range(second_reduction)
        ]
        for first in range(first_reduction)
    ]
    bindings = {
        lowering.input_symbols[0]: left_values,
        lowering.input_symbols[1]: right_values,
    }
    shapes = {lowering.result_symbol: (batch, rows, cols)}

    scheduled = run_program(artifact.program, bindings, shapes)
    base = run_program(artifact.base_program, bindings, shapes)
    erased_program = erase_schedule(artifact.program)
    erased = run_program(erased_program, bindings, shapes)
    expected = [
        [
            [
                float(
                    sum(
                        left_values[batch_pos][row][first][second]
                        * right_values[first][second][col]
                        for first in range(first_reduction)
                        for second in range(second_reduction)
                    )
                )
                for col in range(cols)
            ]
            for row in range(rows)
        ]
        for batch_pos in range(batch)
    ]
    assert scheduled == base == erased
    assert scheduled[lowering.result_symbol] == expected
    assert canonical_program_dump(erased_program) == canonical_program_dump(
        artifact.base_program
    )


def test_reduce_out_form_shape_is_exact():
    """Every deviation from the reduce-out form stays fail-closed."""

    cin, (i, k, j) = build_matmul_ikj()
    lowering = lower(cin)
    base = _reduce_out_auto_plan(lowering, i, k, j, regblock_enabled=False)

    def rejected(code="unsupported_schedule_auto_family", **overrides):
        expect_code(
            code,
            apply_schedule_plan,
            lowering.program,
            _reduce_out_auto_plan(
                lowering, i, k, j, regblock_enabled=False, **overrides
            ),
        )

    # A mixed-placement plan is not a policy arm.
    rejected(
        tiles=(
            base.tiles[0],
            replace(
                base.tiles[1],
                placement=LoopPlacement(
                    PlacementKind.CHILD_OF, parent=LoopRef(i.index_id)
                ),
            ),
        )
    )
    # A width outside the recorded policy is not the automatic decision.
    rejected(tiles=(replace(base.tiles[0], width=16), base.tiles[1]))
    rejected(tiles=(replace(base.tiles[0], unroll=False), base.tiles[1]))
    # Dropping the reduction or axis tile leaves an unmigrated shape.
    rejected(tiles=(base.tiles[1],))
    rejected(tiles=(base.tiles[0],))
    # The sparse-workspace family stays fail-closed.
    rejected(workspace=replace(base.workspace, dense=False))
    rejected(workspace=None)
    # The workspace fact must name the chain facts the pass consumes.
    rejected(
        code="reduce_out_shape_invalid",
        workspace=replace(base.workspace, reduction_loop=LoopRef(i.index_id)),
        tiles=(
            replace(base.tiles[0], loop=LoopRef(i.index_id)),
            base.tiles[1],
        ),
    )
    expect_code(
        "auto_origin_policy",
        apply_schedule_plan,
        lowering.program,
        _reduce_out_auto_plan(
            lowering, i, k, j, regblock_enabled=False, auto_policy=None
        ),
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
    # Relayout is a migrated family now: a relayout fact without its pack
    # and panel tiles to consume it is a malformed relayout plan, not an
    # unmigrated one.
    expect_code(
        "invalid_schedule_relayout",
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
    # Heap result tiles are a migrated family now: a result-tile fact
    # without its heap tile to consume it is a malformed heap plan, not an
    # unmigrated one — and a heap tile without its result-tile fact is
    # equally malformed.
    expect_code(
        "invalid_schedule_result_tile",
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
    # Explicit parallel-loop plans are a migrated family: the fact is
    # consumed into the scheduled program's abstract selection.
    from scorch.compiler.loopir.nodes import ParallelIntent, ParallelPart

    parallel_artifact = apply_schedule_plan(
        lowering.program,
        LoopPlan(loop_order=order, parallel_loop=LoopRef(i.index_id)),
    )
    verify_scheduled_loopir(parallel_artifact)
    stamped = parallel_artifact.program.parallel
    assert stamped is not None
    assert stamped.index == i.index_id
    assert stamped.part is ParallelPart.LOGICAL
    assert stamped.intent is ParallelIntent.EXPLICIT
    assert parallel_artifact.base_program.parallel is None
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=order,
            tiles=(affine_tile(j.index_id, 4, accum="heap"),),
        ),
    )
    # An accumulation kind outside the declared plan vocabulary fails the
    # structural gate before any family dispatch.
    expect_code(
        "invalid_schedule_plan",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=order,
            tiles=(affine_tile(j.index_id, 4, accum="hashed"),),
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


# -- the staged-operand relayout pass -----------------------------------------


def child_of_pack_placement(index_id):
    return LoopPlacement(
        PlacementKind.CHILD_OF, parent=LoopRef(index_id, LoopPart.OUTER)
    )


def spmm_relayout_parts(scope="panel", width=3, strip=4):
    from scorch.compiler.loop_plan import OperandRelayout

    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    b_symbol = lowering.input_symbols[1]
    pack = affine_tile(k.index_id, strip)
    panel = panel_tile(j.index_id, width, placement=child_of_pack_placement(k.index_id))
    bound = PanelBound(LoopRef(j.index_id), b_symbol, 0)
    scope_id = j.index_id if scope == "panel" else k.index_id
    relayout = OperandRelayout(
        operand_id=b_symbol,
        pack_loop=LoopRef(k.index_id),
        panel_loop=LoopRef(j.index_id),
        scope_loop=LoopRef(scope_id),
        row_loop=LoopRef(i.index_id),
        strip_width=strip,
        access_indices=(j.index_id, k.index_id),
        operand_panel_level=0,
        operand_pack_level=1,
    )
    return lowering, (i, j, k), pack, panel, bound, relayout


def relayout_plan(lowering, pack, panel, bound, relayout, parallel):
    return LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(pack, panel),
        panel_bounds=(bound,),
        relayout=relayout,
        parallel_loop=parallel,
        provenance="explicit",
        tag="relayout",
    )


def scheduled_relayout(scope="panel", width=3, strip=4):
    lowering, (i, j, k), pack, panel, bound, relayout = spmm_relayout_parts(
        scope, width, strip
    )
    plan = relayout_plan(lowering, pack, panel, bound, relayout, LoopRef(i.index_id))
    artifact = apply_schedule_plan(lowering.program, plan)
    return lowering, (i, j, k), plan, artifact


def relayout_chain_parts(program, scope):
    from scorch.compiler.loopir.nodes import (
        PanelOuterFor,
        RelayoutStage,
        SparseWindowFor,
    )

    pack = program.body.statements[0]
    assert type(pack) is TileOuterFor
    if scope == "panel":
        panel = pack.body.statements[0]
        assert type(panel) is PanelOuterFor
        stage = panel.body.statements[0]
        row = stage.body.statements[0]
    else:
        stage = pack.body.statements[0]
        panel = stage.body.statements[0]
        assert type(panel) is PanelOuterFor
        row = panel.body.statements[0]
    assert type(stage) is RelayoutStage
    assert type(row) is DenseFor
    window = row.body.statements[0]
    assert type(window) is SparseWindowFor
    point = window.body.statements[0]
    assert type(point) is TileInnerFor
    leaf = point.body.statements[0]
    return pack, panel, stage, row, window, point, leaf


@pytest.mark.parametrize("scope", ["panel", "pack"])
def test_relayout_plan_applies_through_apply_schedule_plan(scope):
    from scorch.compiler.loopir.nodes import RelayoutScope, StagedRead

    lowering, (i, j, k), plan, artifact = scheduled_relayout(scope)
    verify_scheduled_loopir(artifact)
    pack, panel, stage, row, window, point, leaf = relayout_chain_parts(
        artifact.program, scope
    )
    expected_scope = (
        RelayoutScope.PANEL if scope == "panel" else RelayoutScope.PACK_AXIS
    )
    assert stage.decl.scope is expected_scope
    assert stage.decl.operand == plan.relayout.operand_id
    assert stage.decl.panel == panel.tile == window.tile
    assert stage.decl.pack == pack.tile == point.tile
    staged = leaf.value.rhs
    assert type(staged) is StagedRead
    assert staged.relayout == stage.decl.relayout
    assert tuple(index.index for index in staged.indices) == (
        j.index_id,
        k.index_id,
    )
    # The staging region binds no loop: provenance lists exactly the five
    # chain loops in execution order.
    assert [(prov.index, prov.part) for prov in artifact.loops] == [
        (k.index_id, LoopPart.OUTER),
        (j.index_id, LoopPart.OUTER),
        (i.index_id, LoopPart.LOGICAL),
        (j.index_id, LoopPart.INNER),
        (k.index_id, LoopPart.INNER),
    ]


def test_apply_relayout_is_pure_and_deterministic():
    from scorch.compiler.loopir.build import LoopIRBuilder
    from scorch.compiler.loopir.schedule_passes import apply_relayout

    lowering, (i, j, k), pack, panel, bound, relayout = spmm_relayout_parts()
    plan = relayout_plan(lowering, pack, panel, bound, relayout, LoopRef(i.index_id))
    prescheduled = apply_schedule_plan(
        lowering.program,
        LoopPlan(
            loop_order=plan.loop_order,
            tiles=plan.tiles,
            panel_bounds=plan.panel_bounds,
            parallel_loop=plan.parallel_loop,
            provenance="explicit",
        ),
    ).program
    before = canonical_program_dump(prescheduled)
    first = apply_relayout(prescheduled, relayout)
    second = apply_relayout(prescheduled, relayout)
    assert canonical_program_dump(prescheduled) == before
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert print_program(first) == print_program(second)
    assert prescheduled.parallel is not None
    assert first.parallel == prescheduled.parallel
    # The fresh region identity continues deterministically.
    stage = relayout_chain_parts(first, "panel")[2]
    assert LoopIRBuilder.resuming(first).new_relayout_id().value == (
        stage.decl.relayout.value + 1
    )


def test_apply_relayout_requires_the_exact_family_shape():
    from scorch.compiler.loopir.schedule_passes import apply_relayout
    from dataclasses import replace as dc_replace

    lowering, (i, j, k), pack, panel, bound, relayout = spmm_relayout_parts()
    prescheduled = apply_schedule_plan(
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(pack, panel),
            panel_bounds=(bound,),
            parallel_loop=LoopRef(i.index_id),
            provenance="explicit",
        ),
    ).program

    with pytest.raises(TypeError):
        apply_relayout(prescheduled, object())

    # An unscheduled chain has no pack origin to stage against.
    expect_code(
        "invalid_schedule_relayout",
        apply_relayout,
        lowering.program,
        relayout,
    )
    # The strip width must match the pack split.
    expect_code(
        "invalid_schedule_relayout",
        apply_relayout,
        prescheduled,
        dc_replace(relayout, strip_width=8),
    )
    # The row loop must be the window's dense parent.
    expect_code(
        "invalid_schedule_relayout",
        apply_relayout,
        prescheduled,
        dc_replace(relayout, row_loop=LoopRef(k.index_id)),
    )
    # The scope must be the panel or pack loop.
    expect_code(
        "invalid_schedule_relayout",
        apply_relayout,
        prescheduled,
        dc_replace(relayout, scope_loop=LoopRef(i.index_id)),
    )
    # Derived loop parts are not logical loops.
    expect_code(
        "invalid_schedule_relayout",
        apply_relayout,
        prescheduled,
        dc_replace(relayout, pack_loop=LoopRef(k.index_id, LoopPart.OUTER)),
    )
    # The operand levels are validated against the declaration.
    expect_code(
        "invalid_schedule_relayout",
        apply_relayout,
        prescheduled,
        dc_replace(relayout, operand_pack_level=0),
    )
    # The staged operand must be the dense rank-2 input.
    expect_code(
        "invalid_schedule_relayout",
        apply_relayout,
        prescheduled,
        dc_replace(relayout, operand_id=lowering.input_symbols[0]),
    )
    # A malformed fact fails structurally before any chain work.
    hostile = dc_replace(relayout, access_indices=(j.index_id, object()))
    expect_code("invalid_schedule_relayout", apply_relayout, prescheduled, hostile)


def test_apply_relayout_redirection_is_unique_and_complete():
    from scorch.compiler.loopir.schedule_passes import (
        _collect_operand_loads,
        apply_relayout,
    )
    from scorch.compiler.loopir.nodes import Load
    from dataclasses import replace as dc_replace

    from tests.test_scorch.test_loopir_verifier import forge

    lowering, (i, j, k), pack, panel, bound, relayout = spmm_relayout_parts()
    prescheduled = apply_schedule_plan(
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(pack, panel),
            panel_bounds=(bound,),
            parallel_loop=LoopRef(i.index_id),
            provenance="explicit",
        ),
    ).program

    # Access indices that select nothing leave no read to redirect.
    expect_code(
        "relayout_target_missing",
        apply_relayout,
        prescheduled,
        dc_replace(relayout, access_indices=(k.index_id, j.index_id)),
    )

    # A second read occurrence of the operand makes redirection ambiguous.
    from scorch.compiler.loopir.build import LoopIRBuilder

    point = prescheduled.body.statements[0].body.statements[0]
    while not isinstance(point, TileInnerFor):
        point = point.body.statements[0]
    leaf = point.body.statements[0]
    b_load = leaf.value.rhs
    assert type(b_load) is Load
    # Extra instance state is outside the verified/canonical schema.  An
    # alias there must not turn one semantic read into two pass candidates.
    object.__setattr__(prescheduled.body, "ghost_load", b_load)
    verify_program(prescheduled)
    found_loads = _collect_operand_loads(prescheduled.body, b_load.tensor)
    assert len(found_loads) == 1 and found_loads[0] is b_load
    verify_program(apply_relayout(prescheduled, relayout))

    builder = LoopIRBuilder.resuming(prescheduled)
    second_load = builder.load(
        b_load.tensor,
        tuple(builder.index_value(index.index) for index in b_load.indices),
    )
    forge(
        leaf,
        value=builder.binary(
            LoopIRBinaryOp.MUL,
            leaf.value,
            second_load,
        ),
    )
    expect_code("relayout_ambiguous_access", apply_relayout, prescheduled, relayout)


@pytest.mark.parametrize("scope", ["panel", "pack"])
def test_relayout_erases_to_the_reordered_base(scope):
    lowering, _ids, _plan, artifact = scheduled_relayout(scope)
    erased = erase_schedule(artifact.program)
    assert canonical_program_dump(erased) == canonical_program_dump(lowering.program)


def test_relayout_oracle_differential_is_exact():
    import random

    from scorch.compiler.loopir.levels import CsrMatrix

    rng = random.Random(20260724)
    for scope in ("panel", "pack"):
        for width, strip in ((1, 1), (2, 3), (3, 4), (5, 7)):
            lowering, _ids, _plan, artifact = scheduled_relayout(
                scope, width=width, strip=strip
            )
            rows = rng.randrange(1, 6)
            inner = rng.randrange(1, 7)
            cols = rng.randrange(1, 8)
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
            bindings = {
                lowering.input_symbols[0]: CsrMatrix.from_dense(a_dense),
                lowering.input_symbols[1]: b_dense,
            }
            shapes = {lowering.result_symbol: (rows, cols)}
            scheduled_result = run_program(artifact.program, bindings, shapes)
            base_result = run_program(lowering.program, bindings, shapes)
            assert scheduled_result == base_result


def test_scheduling_passes_refuse_relayouted_chains():
    lowering, (i, j, k), _plan, artifact = scheduled_relayout("panel")
    expect_code(
        "unsupported_schedule_shape",
        reorder_loops,
        artifact.program,
        (i.index_id, j.index_id, k.index_id),
    )
    expect_code(
        "unsupported_schedule_shape",
        apply_affine_tile,
        artifact.program,
        affine_tile(i.index_id, 2),
    )
    expect_code(
        "unsupported_schedule_shape",
        apply_stack_tile,
        artifact.program,
        affine_tile(i.index_id, 2, accum="stack"),
    )


def test_relayouted_base_program_is_not_unscheduled():
    lowering, _ids, plan, artifact = scheduled_relayout("panel")
    forged = ScheduledLoopIR(
        base_program=artifact.program,
        plan=artifact.plan,
        program=artifact.program,
        loops=artifact.loops,
    )
    expect_code("scheduled_base_not_unscheduled", verify_scheduled_loopir, forged)


def test_relayout_plan_gate_requires_the_exact_family():
    from dataclasses import replace as dc_replace

    lowering, (i, j, k), pack, panel, bound, relayout = spmm_relayout_parts()
    parallel = LoopRef(i.index_id)

    # Both tiles are required to consume the fact.
    expect_code(
        "invalid_schedule_relayout",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(pack,),
            relayout=relayout,
            parallel_loop=parallel,
            provenance="explicit",
        ),
    )
    # The pack tile must sit at the strip width.
    expect_code(
        "invalid_schedule_relayout",
        apply_schedule_plan,
        lowering.program,
        relayout_plan(
            lowering,
            affine_tile(k.index_id, 8),
            panel,
            bound,
            relayout,
            parallel,
        ),
    )
    # The panel must be placed directly below the pack origin.
    expect_code(
        "invalid_schedule_relayout",
        apply_schedule_plan,
        lowering.program,
        relayout_plan(
            lowering,
            pack,
            panel_tile(j.index_id, 3),
            bound,
            relayout,
            parallel,
        ),
    )
    # The parallel loop must be the relayout's row loop.
    expect_code(
        "invalid_schedule_relayout",
        apply_schedule_plan,
        lowering.program,
        relayout_plan(
            lowering,
            pack,
            panel,
            bound,
            dc_replace(relayout, row_loop=LoopRef(j.index_id)),
            parallel,
        ),
    )
    # Relayout preflight owns the exact logical row/panel/pack order.
    expect_code(
        "invalid_schedule_relayout",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=(i.index_id, k.index_id, j.index_id),
            tiles=(pack, panel),
            panel_bounds=(bound,),
            relayout=relayout,
            parallel_loop=parallel,
            provenance="explicit",
        ),
    )
    # Logical access order may differ from physical panel/pack order for a
    # non-identity dense layout.  In this identity-layout fixture the swapped
    # tuple is structurally admissible but selects no direct operand read.
    expect_code(
        "relayout_target_missing",
        apply_schedule_plan,
        lowering.program,
        relayout_plan(
            lowering,
            pack,
            panel,
            bound,
            dc_replace(
                relayout,
                access_indices=(k.index_id, j.index_id),
            ),
            parallel,
        ),
    )
    expect_code(
        "invalid_schedule_relayout",
        apply_schedule_plan,
        lowering.program,
        relayout_plan(
            lowering,
            pack,
            panel,
            bound,
            dc_replace(relayout, operand_panel_level=1),
            parallel,
        ),
    )
    # Stack accumulation is a separate workspace family.  It previously
    # passed this gate, ran the stack pass, and failed later as a panel
    # target error.
    expect_code(
        "unsupported_schedule_accumulation",
        apply_schedule_plan,
        lowering.program,
        relayout_plan(
            lowering,
            affine_tile(k.index_id, 4, accum="stack"),
            panel,
            bound,
            relayout,
            parallel,
        ),
    )
    # Heap accumulation on the pack tile is migrated but requires the
    # plan's result-tile fact to compact the same pack loop.
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        relayout_plan(
            lowering,
            affine_tile(k.index_id, 4, accum="heap"),
            panel,
            bound,
            relayout,
            parallel,
        ),
    )


# -- the heap result-tile pass ------------------------------------------------

from tests.test_scorch.test_loopir_oracle import csr_from_dense  # noqa: E402


def heap_result_tile_fact(lowering, prefix, pack):
    from scorch.compiler.loop_plan import ResultTile

    return ResultTile(
        result_id=lowering.result_symbol,
        tile_loop=LoopRef(pack.index_id),
        result_level=1,
        result_prefix=(prefix.index_id,),
        access_indices=(prefix.index_id, pack.index_id),
    )


def heap_alone_plan(lowering, i, k, width=3):
    return LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(affine_tile(k.index_id, width, accum="heap"),),
        result_tile=heap_result_tile_fact(lowering, i, k),
        parallel_loop=LoopRef(i.index_id),
        provenance="explicit",
        tag="heap-alone",
    )


def scheduled_heap_alone(width=3):
    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    plan = heap_alone_plan(lowering, i, k, width)
    artifact = apply_schedule_plan(lowering.program, plan)
    return lowering, (i, j, k), plan, artifact


def build_ttm_abcd():
    """``Projected[a, b, d] += Core[a, b, c] * Factor[c, d]`` — rank-3 result.

    The multi-prefix heap representative: the compacted result has two
    dense prefix axes, so the family's derived chain gains one dense loop
    and the compact tile linearizes both prefix positions.
    """

    a, b, c, d = IndexVar("a"), IndexVar("b"), IndexVar("c"), IndexVar("d")
    core = TensorVar("Core", fmt="dds")
    factor = TensorVar("Factor", fmt="dd")
    out = TensorVar("Projected", fmt="ddd")
    assign = TensorAssign(
        out[a, b, d],
        CINBinaryOp(Operation.MUL, core[a, b, c], factor[c, d]),
        op=Operation.ADD,
    )
    return ForAll(a, ForAll(b, ForAll(c, ForAll(d, assign)))), (a, b, c, d)


def dds_storage(rows_dense, batch, rows, inner):
    """Bind a ``dds`` operand: two dense levels over one compressed leaf."""

    from scorch.compiler.loopir.levels import (
        CompressedLevel,
        DenseLevel,
        LevelTensorStorage,
    )

    offsets = [0]
    coords = []
    values = []
    for row in rows_dense:
        for column, entry in enumerate(row):
            if entry != 0.0:
                coords.append(column)
                values.append(entry)
        offsets.append(len(coords))
    return LevelTensorStorage(
        shape=(batch, rows, inner),
        modes=(0, 1, 2),
        levels=(
            DenseLevel(batch),
            DenseLevel(rows),
            CompressedLevel(tuple(offsets), tuple(coords)),
        ),
        values=tuple(values),
    )


def multi_prefix_result_tile_fact(lowering, prefix, pack):
    from scorch.compiler.loop_plan import ResultTile

    return ResultTile(
        result_id=lowering.result_symbol,
        tile_loop=LoopRef(pack.index_id),
        result_level=len(prefix),
        result_prefix=tuple(index.index_id for index in prefix),
        access_indices=tuple(index.index_id for index in prefix) + (pack.index_id,),
    )


def multi_prefix_heap_plan(lowering, prefix, pack, width=3):
    return LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(affine_tile(pack.index_id, width, accum="heap"),),
        result_tile=multi_prefix_result_tile_fact(lowering, prefix, pack),
        parallel_loop=LoopRef(prefix[0].index_id),
        provenance="explicit",
        tag="heap-multi-prefix",
    )


def scheduled_multi_prefix_heap(width=3):
    cin, (a, b, c, d) = build_ttm_abcd()
    lowering = lower(cin)
    plan = multi_prefix_heap_plan(lowering, (a, b), d, width)
    artifact = apply_schedule_plan(lowering.program, plan)
    return lowering, (a, b, c, d), plan, artifact


def heap_relayout_plan(lowering, pack, panel, bound, relayout, parallel, result_tile):
    return LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(replace(pack, accumulation="heap"), panel),
        panel_bounds=(bound,),
        relayout=relayout,
        result_tile=result_tile,
        parallel_loop=parallel,
        provenance="explicit",
        tag="heap-relayout",
    )


def test_heap_plan_applies_through_apply_schedule_plan():
    from scorch.compiler.loopir.nodes import ResultTileRegion, TiledReduce

    lowering, (i, j, k), plan, artifact = scheduled_heap_alone()
    verify_scheduled_loopir(artifact)
    pack = artifact.program.body.statements[0]
    assert type(pack) is TileOuterFor
    region = pack.body.statements[0]
    assert type(region) is ResultTileRegion
    assert region.decl.result == lowering.result_symbol
    assert region.decl.pack == pack.tile
    row = region.body.statements[0]
    assert type(row) is DenseFor and row.index == i.index_id
    point = row.body.statements[0].body.statements[0]
    assert type(point) is TileInnerFor and point.tile == pack.tile
    leaf = point.body.statements[0]
    assert type(leaf) is TiledReduce
    assert leaf.result_tile == region.decl.result_tile
    assert tuple(index.index for index in leaf.indices) == (
        i.index_id,
        k.index_id,
    )
    # The region binds no loop: provenance lists exactly the chain loops.
    assert [(prov.index, prov.part) for prov in artifact.loops] == [
        (k.index_id, LoopPart.OUTER),
        (i.index_id, LoopPart.LOGICAL),
        (j.index_id, LoopPart.LOGICAL),
        (k.index_id, LoopPart.INNER),
    ]


def test_multi_prefix_heap_plan_applies_through_apply_schedule_plan():
    """A rank-3 result yields one dense loop per prefix axis, in order."""

    from scorch.compiler.loopir.nodes import ResultTileRegion, TiledReduce

    lowering, (a, b, c, d), plan, artifact = scheduled_multi_prefix_heap()
    verify_scheduled_loopir(artifact)
    pack = artifact.program.body.statements[0]
    assert type(pack) is TileOuterFor
    region = pack.body.statements[0]
    assert type(region) is ResultTileRegion
    assert region.decl.result == lowering.result_symbol
    assert region.decl.pack == pack.tile
    batch = region.body.statements[0]
    assert type(batch) is DenseFor and batch.index == a.index_id
    row = batch.body.statements[0]
    assert type(row) is DenseFor and row.index == b.index_id
    point = row.body.statements[0].body.statements[0]
    assert type(point) is TileInnerFor and point.tile == pack.tile
    leaf = point.body.statements[0]
    assert type(leaf) is TiledReduce
    assert leaf.result_tile == region.decl.result_tile
    assert tuple(index.index for index in leaf.indices) == (
        a.index_id,
        b.index_id,
        d.index_id,
    )
    assert [(prov.index, prov.part) for prov in artifact.loops] == [
        (d.index_id, LoopPart.OUTER),
        (a.index_id, LoopPart.LOGICAL),
        (b.index_id, LoopPart.LOGICAL),
        (c.index_id, LoopPart.LOGICAL),
        (d.index_id, LoopPart.INNER),
    ]


def test_multi_prefix_heap_erases_to_the_reordered_base():
    lowering, _ids, _plan, artifact = scheduled_multi_prefix_heap()
    erased = erase_schedule(artifact.program)
    assert canonical_program_dump(erased) == canonical_program_dump(lowering.program)


def test_multi_prefix_heap_oracle_differential_is_exact():
    """The compact rank-3 tile computes exactly the direct reduction."""

    import random

    rng = random.Random(20260725)
    for width in (1, 2, 3, 5, 8):
        lowering, _ids, _plan, artifact = scheduled_multi_prefix_heap(width)
        batch = rng.randrange(1, 4)
        rows = rng.randrange(1, 4)
        inner = rng.randrange(1, 6)
        cols = rng.randrange(1, 7)
        # ``Core`` is ``dds``: the compressed leaf's segments enumerate the
        # (a, b) parent positions in row-major order.
        core_dense = [
            [
                float(rng.randrange(-3, 4)) if rng.random() < 0.5 else 0.0
                for _ in range(inner)
            ]
            for _ in range(batch * rows)
        ]
        factor = [
            [float(rng.randrange(-3, 4)) for _ in range(cols)] for _ in range(inner)
        ]
        bindings = {
            lowering.input_symbols[0]: dds_storage(core_dense, batch, rows, inner),
            lowering.input_symbols[1]: factor,
        }
        shapes = {lowering.result_symbol: (batch, rows, cols)}
        scheduled_result = run_program(artifact.program, bindings, shapes)
        erased_result = run_program(erase_schedule(artifact.program), bindings, shapes)
        assert scheduled_result == erased_result
        reference = [
            [
                [
                    sum(
                        core_dense[index * rows + row][red] * factor[red][col]
                        for red in range(inner)
                        if core_dense[index * rows + row][red] != 0.0
                    )
                    for col in range(cols)
                ]
                for row in range(rows)
            ]
            for index in range(batch)
        ]
        assert scheduled_result[lowering.result_symbol] == reference


@pytest.mark.parametrize(
    ("batch", "rows", "inner", "cols"),
    [
        (0, 3, 4, 5),
        (2, 0, 4, 5),
        (2, 3, 0, 5),
        (2, 3, 4, 0),
    ],
    ids=("zero-outer-prefix", "zero-inner-prefix", "zero-reduction", "zero-free"),
)
def test_multi_prefix_heap_oracle_covers_zero_extents(batch, rows, inner, cols):
    """Every zero-extent position preserves fresh-zero/copy-out semantics."""

    lowering, _ids, _plan, artifact = scheduled_multi_prefix_heap(width=3)
    core_dense = [
        [float((parent + red) % 5 - 2) for red in range(inner)]
        for parent in range(batch * rows)
    ]
    factor = [
        [float((red * 3 + col) % 7 - 3) for col in range(cols)] for red in range(inner)
    ]
    bindings = {
        lowering.input_symbols[0]: dds_storage(
            core_dense,
            batch,
            rows,
            inner,
        ),
        lowering.input_symbols[1]: factor,
    }
    shapes = {lowering.result_symbol: (batch, rows, cols)}
    scheduled_result = run_program(artifact.program, bindings, shapes)
    erased_result = run_program(erase_schedule(artifact.program), bindings, shapes)
    assert scheduled_result == erased_result
    reference = [
        [
            [
                sum(
                    core_dense[index * rows + row][red] * factor[red][col]
                    for red in range(inner)
                )
                for col in range(cols)
            ]
            for row in range(rows)
        ]
        for index in range(batch)
    ]
    assert scheduled_result[lowering.result_symbol] == reference


def test_multi_prefix_heap_keeps_physical_and_logical_orders_distinct():
    """The pass maps a physical prefix through the result's logical modes."""

    from scorch.compiler.loop_plan import ResultTile
    from scorch.compiler.loopir.nodes import LevelKind, ResultTileRegion
    from tests.test_scorch.test_loopir_verifier import forge

    cin, (a, b, c, d) = build_ttm_abcd()
    lowering = lower(cin)
    result_decl = next(
        decl
        for decl in lowering.program.tensors
        if decl.symbol == lowering.result_symbol
    )
    builder = LoopIRBuilder.resuming(lowering.program)
    forge(
        result_decl,
        levels=(
            builder.level(LevelKind.DENSE, 1),
            builder.level(LevelKind.DENSE, 0),
            builder.level(LevelKind.DENSE, 2),
        ),
    )
    result_tile = ResultTile(
        result_id=lowering.result_symbol,
        tile_loop=LoopRef(d.index_id),
        result_level=2,
        # A (1, 0, 2) physical mode order stores b then a before d.
        result_prefix=(b.index_id, a.index_id),
        # Tensor accesses remain in logical a, b, d order.
        access_indices=(a.index_id, b.index_id, d.index_id),
    )
    plan = LoopPlan(
        loop_order=(b.index_id, a.index_id, c.index_id, d.index_id),
        tiles=(affine_tile(d.index_id, 3, accum="heap"),),
        result_tile=result_tile,
        parallel_loop=LoopRef(b.index_id),
        provenance="explicit",
        tag="heap-physical-prefix",
    )
    artifact = apply_schedule_plan(lowering.program, plan)
    verify_scheduled_loopir(artifact)
    pack = artifact.program.body.statements[0]
    region = pack.body.statements[0]
    assert type(region) is ResultTileRegion
    physical_b = region.body.statements[0]
    physical_a = physical_b.body.statements[0]
    assert type(physical_b) is DenseFor and physical_b.index == b.index_id
    assert type(physical_a) is DenseFor and physical_a.index == a.index_id


def test_multi_prefix_heap_admits_every_prefix_parallel_anchor():
    """Any dense prefix loop is a legal heap anchor — the legacy envelope.

    The former outermost-prefix pin (§18.4 boundary 2) is lifted by the
    abstract selection: each admitted anchor is consumed into the
    scheduled program and a non-prefix anchor still fails closed.
    """

    from dataclasses import replace as dataclass_replace

    from scorch.compiler.loopir.nodes import ParallelDiscipline, ParallelPart

    cin, (a, b, c, d) = build_ttm_abcd()
    lowering = lower(cin)
    plan = multi_prefix_heap_plan(lowering, (a, b), d)
    for anchor in (a, b):
        anchored = dataclass_replace(plan, parallel_loop=LoopRef(anchor.index_id))
        artifact = apply_schedule_plan(lowering.program, anchored)
        verify_scheduled_loopir(artifact)
        stamped = artifact.program.parallel
        assert stamped is not None
        assert stamped.index == anchor.index_id
        assert stamped.part is ParallelPart.LOGICAL
        assert stamped.discipline is ParallelDiscipline.COMPACT_PARTITION
        if anchor is a:
            assert stamped.work.nnz is None
        else:
            assert stamped.work.nnz is not None
            assert stamped.work.nnz.tensor == lowering.input_symbols[0]
            assert stamped.work.nnz.level == 2
    reduction_anchor = dataclass_replace(plan, parallel_loop=LoopRef(c.index_id))
    defect = expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        reduction_anchor,
    )
    assert "dense result-prefix loop" in defect.message
    pack_anchor = dataclass_replace(plan, parallel_loop=LoopRef(d.index_id))
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        pack_anchor,
    )


def multi_prefix_heap_panel_plan(lowering, a, b, c, d, *, anchor):
    """The rank-3 TTM heap chain composed with a window over ``c``."""

    return LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(
            affine_tile(d.index_id, 3, accum="heap"),
            panel_tile(
                c.index_id,
                2,
                placement=LoopPlacement(
                    PlacementKind.CHILD_OF,
                    parent=LoopRef(d.index_id, LoopPart.OUTER),
                ),
            ),
        ),
        panel_bounds=(PanelBound(LoopRef(c.index_id), lowering.input_symbols[1], 0),),
        result_tile=multi_prefix_result_tile_fact(lowering, (a, b), d),
        parallel_loop=LoopRef(anchor.index_id),
        provenance="explicit",
        tag="heap-multi-prefix-panel",
    )


def test_multi_prefix_heap_composes_a_sparse_panel():
    """The rank-3 heap+panel chain applies with the dense-parent row anchor.

    Lifts §18.4 boundary 1: legacy admits the composition only with the
    window's CSR dense-parent row (the innermost prefix) as the parallel
    anchor, which the abstract selection can now express.  Any other
    prefix anchor keeps failing closed exactly as legacy does.
    """

    from scorch.compiler.loopir.nodes import (
        DenseFor as _DenseFor,
        PanelOuterFor,
        ParallelDiscipline,
        ParallelPart,
        ResultTileRegion,
        SparseWindowFor,
        TileInnerFor,
        TileOuterFor,
    )

    cin, (a, b, c, d) = build_ttm_abcd()
    lowering = lower(cin)
    plan = multi_prefix_heap_panel_plan(lowering, a, b, c, d, anchor=b)
    artifact = apply_schedule_plan(lowering.program, plan)
    verify_scheduled_loopir(artifact)
    stamped = artifact.program.parallel
    assert stamped is not None
    assert stamped.index == b.index_id
    assert stamped.part is ParallelPart.LOGICAL
    assert stamped.discipline is ParallelDiscipline.COMPACT_PARTITION
    pack = artifact.program.body.statements[0]
    assert type(pack) is TileOuterFor
    region = pack.body.statements[0]
    assert type(region) is ResultTileRegion
    panel = region.body.statements[0]
    assert type(panel) is PanelOuterFor
    loop_a = panel.body.statements[0]
    loop_b = loop_a.body.statements[0]
    window = loop_b.body.statements[0]
    point = window.body.statements[0]
    assert type(loop_a) is _DenseFor and loop_a.index == a.index_id
    assert type(loop_b) is _DenseFor and loop_b.index == b.index_id
    assert type(window) is SparseWindowFor
    assert type(point) is TileInnerFor
    erased = erase_schedule(artifact.program)
    assert canonical_program_dump(erased) == canonical_program_dump(
        artifact.base_program
    )
    outer_anchor = multi_prefix_heap_panel_plan(lowering, a, b, c, d, anchor=a)
    defect = expect_code(
        "panel_parallel_scope",
        apply_schedule_plan,
        lowering.program,
        outer_anchor,
    )
    assert "dense" in defect.message


@pytest.mark.parametrize("scope", ["panel", "pack"])
def test_heap_relayout_plan_applies_at_both_scopes(scope):
    from scorch.compiler.loopir.nodes import (
        PanelOuterFor,
        RelayoutStage,
        ResultTileRegion,
        TiledReduce,
    )

    lowering, (i, j, k), pack, panel, bound, relayout = spmm_relayout_parts(scope)
    plan = heap_relayout_plan(
        lowering,
        pack,
        panel,
        bound,
        relayout,
        LoopRef(i.index_id),
        heap_result_tile_fact(lowering, i, k),
    )
    artifact = apply_schedule_plan(lowering.program, plan)
    verify_scheduled_loopir(artifact)
    origin = artifact.program.body.statements[0]
    assert type(origin) is TileOuterFor
    region = origin.body.statements[0]
    assert type(region) is ResultTileRegion
    inner = region.body.statements[0]
    if scope == "pack":
        # PACK_AXIS: the staging region sits inside the result-tile region.
        assert type(inner) is RelayoutStage
        assert type(inner.body.statements[0]) is PanelOuterFor
    else:
        # PANEL: the staging region sits inside the panel origin.
        assert type(inner) is PanelOuterFor
        assert type(inner.body.statements[0]) is RelayoutStage
    leaves = [
        node for node in walk_stmts(artifact.program.body) if type(node) is TiledReduce
    ]
    assert len(leaves) == 1
    assert leaves[0].result_tile == region.decl.result_tile
    # Provenance stays loop-only across both transparent regions.
    assert [(prov.index, prov.part) for prov in artifact.loops] == [
        (k.index_id, LoopPart.OUTER),
        (j.index_id, LoopPart.OUTER),
        (i.index_id, LoopPart.LOGICAL),
        (j.index_id, LoopPart.INNER),
        (k.index_id, LoopPart.INNER),
    ]


def walk_stmts(root):
    from scorch.compiler.loopir.nodes import LoopIRNode

    pending = [root]
    while pending:
        value = pending.pop()
        yield value
        for child in vars(value).values():
            if isinstance(child, LoopIRNode):
                pending.append(child)
            elif type(child) is tuple:
                pending.extend(item for item in child if isinstance(item, LoopIRNode))


def test_apply_result_tile_is_pure_and_deterministic():
    from scorch.compiler.loopir.schedule_passes import apply_result_tile

    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    prescheduled = apply_schedule_plan(
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(affine_tile(k.index_id, 3),),
        ),
    ).program
    fact = heap_result_tile_fact(lowering, i, k)
    before = canonical_program_dump(prescheduled)
    first = apply_result_tile(prescheduled, fact)
    second = apply_result_tile(prescheduled, fact)
    assert canonical_program_dump(prescheduled) == before
    assert canonical_program_dump(first) == canonical_program_dump(second)
    region = first.body.statements[0].body.statements[0]
    assert LoopIRBuilder.resuming(first).new_result_tile_id().value == (
        region.decl.result_tile.value + 1
    )
    # Reapplying to an already-compacted chain fails closed: the region is
    # not a chain element for scheduling passes.
    expect_code("unsupported_schedule_shape", apply_result_tile, first, fact)


def test_apply_result_tile_requires_the_exact_family_shape():
    from dataclasses import replace as dc_replace

    from scorch.compiler.loopir.schedule_passes import apply_result_tile

    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    fact = heap_result_tile_fact(lowering, i, k)

    # An unsplit chain has no pack origin/point pair.
    expect_code(
        "invalid_schedule_result_tile",
        apply_result_tile,
        lowering.program,
        fact,
    )
    # A wrong tile loop cannot be the chain's schedule pair.
    prescheduled = apply_schedule_plan(
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(affine_tile(k.index_id, 3),),
        ),
    ).program
    preselected = apply_schedule_plan(
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(affine_tile(k.index_id, 3),),
            parallel_loop=LoopRef(i.index_id),
        ),
    ).program
    defect = expect_code(
        "invalid_schedule_result_tile",
        apply_result_tile,
        preselected,
        fact,
    )
    assert "parallel selection must run after" in defect.message
    expect_code(
        "invalid_schedule_result_tile",
        apply_result_tile,
        prescheduled,
        dc_replace(fact, tile_loop=LoopRef(j.index_id)),
    )
    # The prefix fact must be the chain's row loop.
    expect_code(
        "invalid_schedule_result_tile",
        apply_result_tile,
        prescheduled,
        dc_replace(
            fact,
            result_prefix=(j.index_id,),
            access_indices=(j.index_id, k.index_id),
        ),
    )
    # A multi-prefix fact is incompatible with this rank-2 program shape.
    expect_code(
        "invalid_schedule_result_tile",
        apply_result_tile,
        prescheduled,
        dc_replace(
            fact,
            result_prefix=(i.index_id, j.index_id),
            access_indices=(i.index_id, j.index_id, k.index_id),
            result_level=2,
        ),
    )
    # A non-fact type fails before anything runs.
    with pytest.raises(TypeError):
        apply_result_tile(prescheduled, object())


def test_apply_result_tile_redirection_is_unique_and_complete():
    from dataclasses import replace as dc_replace

    from scorch.compiler.loopir.schedule_passes import (
        _collect_result_writes,
        apply_result_tile,
    )

    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    prescheduled = apply_schedule_plan(
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            tiles=(affine_tile(k.index_id, 3),),
        ),
    ).program
    fact = heap_result_tile_fact(lowering, i, k)
    # Swapped access indices are caught by the exact fact pin before the
    # write scan: unlike operand reads, result writes are statements the
    # verifier's coordinate model fully pins, so the unique-occurrence
    # scan behind the pin is checked-property defense in depth.
    expect_code(
        "invalid_schedule_result_tile",
        apply_result_tile,
        prescheduled,
        dc_replace(fact, access_indices=(k.index_id, i.index_id)),
    )
    # The scan itself sees exactly the one leaf write in the verified
    # chain and nothing else.
    writes = _collect_result_writes(prescheduled.body, lowering.result_symbol)
    assert len(writes) == 1
    assert _collect_result_writes(prescheduled.body, lowering.input_symbols[0]) == []
    # Verifier-invisible instance state is not semantic pass input.  A
    # hidden alias of the write must neither create ambiguity nor be walked.
    object.__setattr__(prescheduled.body, "ghost_write", writes[0])
    verify_program(prescheduled)
    found_writes = _collect_result_writes(prescheduled.body, lowering.result_symbol)
    assert len(found_writes) == 1 and found_writes[0] is writes[0]
    # After the pass, no direct write of the result survives anywhere.
    compacted = apply_result_tile(prescheduled, fact)
    assert _collect_result_writes(compacted.body, lowering.result_symbol) == []


def test_heap_plan_gate_requires_the_exact_family():
    from dataclasses import replace as dc_replace

    cin, (i, j, k) = build_spmm_ijk()
    lowering = lower(cin)
    plan = heap_alone_plan(lowering, i, k)
    fact = plan.result_tile

    # The heap tile must be outermost and serial.
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        replace(
            plan,
            tiles=(
                affine_tile(
                    k.index_id,
                    3,
                    placement=LoopPlacement(PlacementKind.AT_DEPTH, depth=1),
                    accum="heap",
                ),
            ),
        ),
    )
    # Keeping the heap loop innermost is insufficient: the dense result
    # prefix must remain the first logical loop, before the reduction.
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        replace(
            plan,
            loop_order=(
                lowering.loop_index_ids[1],
                lowering.loop_index_ids[0],
                lowering.loop_index_ids[2],
            ),
        ),
    )
    parallel_heap = LoopTile(
        loop=LoopRef(k.index_id),
        width=3,
        placement=outermost_placement(),
        parallel=True,
        kind="affine",
        accumulation="heap",
        unroll=False,
    )
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        replace(plan, tiles=(parallel_heap,)),
    )
    # The result tile must compact the heap tile's loop.
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        replace(plan, result_tile=dc_replace(fact, tile_loop=LoopRef(j.index_id))),
    )
    # The heap tile targets the innermost logical loop.
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        replace(
            plan,
            loop_order=(
                lowering.loop_index_ids[0],
                lowering.loop_index_ids[2],
                lowering.loop_index_ids[1],
            ),
        ),
    )
    # The mandatory parallel anchor must be a dense result-prefix loop.
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        replace(plan, parallel_loop=None),
    )
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        replace(plan, parallel_loop=LoopRef(j.index_id)),
    )
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        replace(plan, parallel_loop=LoopRef(k.index_id, LoopPart.OUTER)),
    )
    # A second non-panel tile is outside the audited compositions.
    expect_code(
        "invalid_schedule_result_tile",
        apply_schedule_plan,
        lowering.program,
        replace(
            plan,
            tiles=(*plan.tiles, affine_tile(i.index_id, 2)),
        ),
    )

    # In the composed family, the panel is exactly the pack origin's child.
    # Check this before replay so a broader panel pass cannot accidentally
    # admit a heap lifetime that the target cannot complete.
    from scorch.compiler.loopir.schedule_passes import _check_heap_plan_family

    (
        panel_lowering,
        (panel_i, _panel_j, panel_k),
        pack,
        panel_tile,
        bound,
        relayout,
    ) = spmm_relayout_parts()
    misplaced_panel_plan = heap_relayout_plan(
        panel_lowering,
        pack,
        replace(panel_tile, placement=outermost_placement()),
        bound,
        relayout,
        LoopRef(panel_i.index_id),
        heap_result_tile_fact(panel_lowering, panel_i, panel_k),
    )
    expect_code(
        "invalid_schedule_result_tile",
        _check_heap_plan_family,
        misplaced_panel_plan,
    )


def test_heap_scheduled_iteration_counts_and_erases_exactly():
    """Counting differential + erasure: all-ones inputs count stored
    entries per output cell across exact/ragged/unit/oversized strips, and
    erasure restores the base canonical dump."""

    inner, cols = 7, 5
    a_dense = [
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0] * inner,
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0] * inner,
    ]
    stored_counts = [4.0, 0.0, 3.0, 7.0]
    for width in (1, 2, 3, 5, 8):
        lowering, (i, j, k), plan, artifact = scheduled_heap_alone(width)
        inputs = {
            lowering.input_symbols[0]: csr_from_dense(a_dense),
            lowering.input_symbols[1]: [[1.0] * cols for _ in range(inner)],
        }
        shapes = {lowering.result_symbol: (len(a_dense), cols)}
        counted = run_program(artifact.program, inputs, shapes)
        assert counted[lowering.result_symbol] == [
            [count] * cols for count in stored_counts
        ]
        erased = erase_schedule(artifact.program)
        assert canonical_program_dump(erased) == canonical_program_dump(
            artifact.base_program
        )


@pytest.mark.parametrize("scope", ["panel", "pack"])
def test_heap_relayout_composition_counts_and_erases_exactly(scope):
    inner, cols = 7, 5
    a_dense = [
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0] * inner,
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0] * inner,
    ]
    stored_counts = [4.0, 0.0, 3.0, 7.0]
    for width, strip in ((1, 2), (2, 5), (3, 3), (9, 8)):
        lowering, (i, j, k), pack, panel, bound, relayout = spmm_relayout_parts(
            scope, width, strip
        )
        plan = heap_relayout_plan(
            lowering,
            pack,
            panel,
            bound,
            relayout,
            LoopRef(i.index_id),
            heap_result_tile_fact(lowering, i, k),
        )
        artifact = apply_schedule_plan(lowering.program, plan)
        inputs = {
            lowering.input_symbols[0]: csr_from_dense(a_dense),
            lowering.input_symbols[1]: [[1.0] * cols for _ in range(inner)],
        }
        shapes = {lowering.result_symbol: (len(a_dense), cols)}
        counted = run_program(artifact.program, inputs, shapes)
        assert counted[lowering.result_symbol] == [
            [count] * cols for count in stored_counts
        ]
        erased = erase_schedule(artifact.program)
        assert canonical_program_dump(erased) == canonical_program_dump(
            artifact.base_program
        )


def test_scheduled_carrier_rejects_result_tile_region_bases():
    lowering, (i, j, k), plan, artifact = scheduled_heap_alone()
    forged = ScheduledLoopIR(
        base_program=artifact.program,
        plan=plan,
        program=artifact.program,
        loops=artifact.loops,
    )
    expect_code("scheduled_base_not_unscheduled", verify_scheduled_loopir, forged)


def test_rank_four_heap_plan_derives_three_prefix_loops():
    """Arbitrary positive prefix rank is derived, never enumerated."""

    a, b, c, d, e = (IndexVar(name) for name in "abcde")
    core = TensorVar("Core4", fmt="ddds")
    factor = TensorVar("Factor", fmt="dd")
    out = TensorVar("Out4", fmt="dddd")
    assign = TensorAssign(
        out[a, b, c, e],
        CINBinaryOp(Operation.MUL, core[a, b, c, d], factor[d, e]),
        op=Operation.ADD,
    )
    cin = ForAll(a, ForAll(b, ForAll(c, ForAll(d, ForAll(e, assign)))))
    lowering = lower(cin)
    from scorch.compiler.loop_plan import ResultTile

    plan = LoopPlan(
        loop_order=lowering.loop_index_ids,
        tiles=(affine_tile(e.index_id, 3, accum="heap"),),
        result_tile=ResultTile(
            result_id=lowering.result_symbol,
            tile_loop=LoopRef(e.index_id),
            result_level=3,
            result_prefix=(a.index_id, b.index_id, c.index_id),
            access_indices=(a.index_id, b.index_id, c.index_id, e.index_id),
        ),
        parallel_loop=LoopRef(a.index_id),
        provenance="explicit",
        tag="heap-rank4",
    )
    artifact = apply_schedule_plan(lowering.program, plan)
    verify_scheduled_loopir(artifact)
    erased = erase_schedule(artifact.program)
    assert canonical_program_dump(erased) == canonical_program_dump(lowering.program)


def test_rank_two_nonidentity_heap_oracle_differential_is_exact():
    """A (1, 0) physical result packs logical mode zero; execution is exact.

    The packed axis is the trailing physical level but logical mode zero, so
    the oracle's compact addressing and copy-out must key logical coordinates
    rather than nesting depth.
    """

    from scorch.compiler.loop_plan import ResultTile
    from scorch.compiler.loopir.nodes import LevelKind
    from tests.test_scorch.test_loopir_verifier import forge

    rng = random.Random(4321)
    for width in (1, 2, 3, 5):
        cin, (i, k, j) = build_matmul_ikj()
        lowering = lower(cin)
        result_decl = next(
            decl
            for decl in lowering.program.tensors
            if decl.symbol == lowering.result_symbol
        )
        builder = LoopIRBuilder.resuming(lowering.program)
        forge(
            result_decl,
            levels=(
                builder.level(LevelKind.DENSE, 1),
                builder.level(LevelKind.DENSE, 0),
            ),
        )
        plan = LoopPlan(
            loop_order=(j.index_id, k.index_id, i.index_id),
            tiles=(affine_tile(i.index_id, width, accum="heap"),),
            result_tile=ResultTile(
                result_id=lowering.result_symbol,
                tile_loop=LoopRef(i.index_id),
                result_level=1,
                result_prefix=(j.index_id,),
                access_indices=(i.index_id, j.index_id),
            ),
            parallel_loop=LoopRef(j.index_id),
            provenance="explicit",
            tag="heap-col-major",
        )
        artifact = apply_schedule_plan(lowering.program, plan)
        verify_scheduled_loopir(artifact)
        rows, inner, cols = 4, 3, 5
        a_values = [
            [float(rng.randrange(-3, 4)) for _ in range(inner)] for _ in range(rows)
        ]
        b_values = [
            [float(rng.randrange(-3, 4)) for _ in range(cols)] for _ in range(inner)
        ]
        bindings = {
            lowering.input_symbols[0]: a_values,
            lowering.input_symbols[1]: b_values,
        }
        shapes = {lowering.result_symbol: (rows, cols)}
        scheduled_result = run_program(artifact.program, bindings, shapes)
        erased_result = run_program(erase_schedule(artifact.program), bindings, shapes)
        assert scheduled_result == erased_result
        reference = [
            [
                sum(a_values[row][red] * b_values[red][col] for red in range(inner))
                for col in range(cols)
            ]
            for row in range(rows)
        ]
        assert scheduled_result[lowering.result_symbol] == reference


def test_erase_schedule_drops_a_parallel_selection():
    """The selection is schedule state; erasure restores the bare base."""

    from tests.test_scorch.test_loopir_verifier import (
        attach_selection,
        build_csr_spmv,
    )

    bare = build_csr_spmv()
    selected = build_csr_spmv()
    dim_i = selected.program.tensors[2].dimensions[0]
    attach_selection(
        selected,
        selected.row,
        rows=dim_i,
        nnz=selected.builder.sparse_work_source(selected.a, 1),
    )
    erased = erase_schedule(selected.program)
    assert erased.parallel is None
    assert canonical_program_dump(erased) == canonical_program_dump(bare.program)
    assert erase_schedule(bare.program) is bare.program


def test_parallel_selection_cannot_be_paired_with_a_fact_free_plan():
    """A carried selection may not silently survive a contradictory plan."""

    from tests.test_scorch.test_loopir_verifier import (
        attach_selection,
        build_vector_add,
    )

    fixture = build_vector_add()
    attach_selection(fixture, fixture.index, rows=fixture.dim)
    plan = LoopPlan(loop_order=(fixture.index,), provenance="explicit")
    defect = expect_code(
        "invalid_schedule_parallel",
        select_parallel_loop,
        fixture.program,
        plan,
    )
    assert "has no parallel-loop fact" in defect.message


def test_scheduled_carrier_rejects_a_parallel_base():
    """A base program carrying a selection is not an unscheduled base."""

    from scorch.compiler.loopir.nodes import (
        ParallelDiscipline,
        ParallelIntent,
        ParallelPart,
    )
    from tests.test_scorch.test_loopir_verifier import forge

    lowering, (i, j, k), plan, artifact = scheduled_heap_alone()
    base = artifact.base_program
    builder = LoopIRBuilder.resuming(base)
    row_dimension = next(
        decl for decl in base.tensors if decl.symbol == lowering.result_symbol
    ).dimensions[0]
    selection = builder.parallel_selection(
        i.index_id,
        ParallelPart.LOGICAL,
        ParallelDiscipline.RESULT_PARTITION,
        builder.parallel_work(
            row_dimension,
            builder.sparse_work_source(lowering.input_symbols[0], 1),
        ),
        ParallelIntent.EXPLICIT,
    )
    forge(base, parallel=selection)
    expect_code("scheduled_base_not_unscheduled", verify_scheduled_loopir, artifact)


def test_parallel_selection_is_execution_neutral():
    """The selection has no execution semantics; the oracle ignores it."""

    from tests.test_scorch.test_loopir_verifier import (
        attach_selection,
        build_csr_spmv,
    )
    from tests.test_scorch.test_loopir_verifier import CsrSpmvFixture

    def run(fixture: CsrSpmvFixture):
        dense = [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 5.0],
        ]
        bindings = {
            fixture.a: csr_from_dense(dense),
            fixture.x: [1.0, 2.0, 3.0],
        }
        shapes = {fixture.y: (4,)}
        return run_program(fixture.program, bindings, shapes)[fixture.y]

    bare = build_csr_spmv()
    selected = build_csr_spmv()
    dim_i = selected.program.tensors[2].dimensions[0]
    attach_selection(
        selected,
        selected.row,
        rows=dim_i,
        nnz=selected.builder.sparse_work_source(selected.a, 1),
    )
    assert run(selected) == run(bare)


def test_resuming_continues_past_selection_identities():
    from tests.test_scorch.test_loopir_verifier import (
        attach_selection,
        build_csr_spmv,
    )

    fixture = build_csr_spmv()
    dim_i = fixture.program.tensors[2].dimensions[0]
    selection = attach_selection(
        fixture,
        fixture.row,
        rows=dim_i,
        nnz=fixture.builder.sparse_work_source(fixture.a, 1),
    )
    continued = LoopIRBuilder.resuming(fixture.program)
    fresh = continued.block(())
    assert fresh.node_id.value > selection.node_id.value
    assert fresh.node_id.value > selection.work.node_id.value


def test_parallel_selection_rejects_unmigrated_anchor_kinds():
    """Merged and compressed coordinate anchors stay outside the family."""

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]))
    lowering = lower(ForAll(i, ForAll(j, assign)))
    defect = expect_code(
        "invalid_schedule_parallel",
        apply_schedule_plan,
        lowering.program,
        LoopPlan(
            loop_order=lowering.loop_index_ids,
            parallel_loop=LoopRef(j.index_id),
        ),
    )
    assert "dense logical loops and affine origin loops" in defect.message
