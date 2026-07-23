"""Typed LoopIR scheduling passes: legality, purity, provenance, erasure.

The reorder and affine-tiling passes must be pure functions over verified
programs, fail closed with stable codes on everything outside the migrated
schedule families, mirror the legacy placement semantics exactly, retain
provenance to the unscheduled base program, and preserve semantics — the
oracle differentials here prove every iteration point of a scheduled
program is visited exactly once (exact integer-float counting) across
ragged, exact, oversized, and zero extents.
"""

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
    LoopPart,
    LoopPlacement,
    LoopPlan,
    LoopRef,
    LoopTile,
    PlacementKind,
)
from scorch.compiler.loopir.lower_cin import lower_normalized_cin_to_loopir
from scorch.compiler.loopir.nodes import (
    BinaryOp as LoopIRBinaryOp,
    Block,
    DenseFor,
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
    erase_schedule,
    reorder_loops,
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
    expect_code(
        "unsupported_schedule_panel",
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
            tiles=(affine_tile(j.index_id, 4, accum="stack"),),
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
