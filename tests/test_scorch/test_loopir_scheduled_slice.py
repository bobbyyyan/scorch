"""The Phase-6 scheduled vertical slice: byte parity, execution, boundaries.

Every schedule the LoopIR strangler entry accepts is compared against the
production legacy scheduled route (``Scheduler.apply_schedule`` followed by
the legacy lowering of the verified ``ScheduledCIN``):

- a nineteen-member scheduled byte-parity grid (plus a result-bounded
  broadcast split) locks generated C++ equality across explicit reorders
  and affine ``accum='direct'`` tiles — every placement kind, unroll
  on/off, one and two splits, tile-i/tile-k over CSR SpMM, f32/f64, and N
  below/equal to/above/not divisible by the width, including zero extents;
- compiled shadow execution runs both pipelines on real tensors and
  requires bitwise-equal dense results plus PyTorch agreement;
- every unsupported schedule family fails closed with a stable code at the
  schedule-application boundary and the failure owns its stage record;
- the target lowering keeps its own fail-closed boundary for scheduled
  shapes it does not emit (splits over merged iteration or ordered
  assembly).
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
from scorch.compiler.compilation_context import (
    CompilationContext,
    CompilationContextError,
    CompilerStageId,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loop_plan import LoopPart
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import (
    compare_generated_sources,
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    execute_shadow,
)
from scorch.compiler.loopir.schedule_passes import SchedulePassError
from scorch.compiler.scheduler import Schedule, Scheduler, TileSpec
from scorch.stensor import STensor

F32 = torch.float32
F64 = torch.float64


def build_matmul(order=("i", "k", "j"), dtype=F32):
    ivs = {name: IndexVar(name) for name in ("i", "j", "k")}
    c = TensorVar("C", fmt="dd", dtype=dtype)
    a = TensorVar("A", fmt="dd", dtype=dtype)
    b = TensorVar("B", fmt="dd", dtype=dtype)
    stmt = TensorAssign(
        c[ivs["i"], ivs["j"]],
        CINBinaryOp(Operation.MUL, a[ivs["i"], ivs["k"]], b[ivs["k"], ivs["j"]]),
        op=Operation.ADD,
    )
    for name in reversed(order):
        stmt = ForAll(ivs[name], stmt)
    return stmt


def build_spmm(dtype=F32):
    ivs = {name: IndexVar(name) for name in ("i", "j", "k")}
    c = TensorVar("C", fmt="dd", dtype=dtype)
    a = TensorVar("A", fmt="ds", dtype=dtype)
    b = TensorVar("B", fmt="dd", dtype=dtype)
    stmt = TensorAssign(
        c[ivs["i"], ivs["k"]],
        CINBinaryOp(Operation.MUL, a[ivs["i"], ivs["j"]], b[ivs["j"], ivs["k"]]),
        op=Operation.ADD,
    )
    for name in reversed(("i", "j", "k")):
        stmt = ForAll(ivs[name], stmt)
    return stmt


def build_two_reduction(dtype=F32):
    ivs = {name: IndexVar(name) for name in ("i", "j", "k")}
    c = TensorVar("C", fmt="d", dtype=dtype)
    a = TensorVar("A", fmt="dd", dtype=dtype)
    b = TensorVar("B", fmt="dd", dtype=dtype)
    stmt = TensorAssign(
        c[ivs["i"]],
        CINBinaryOp(Operation.MUL, a[ivs["i"], ivs["j"]], b[ivs["i"], ivs["k"]]),
        op=Operation.ADD,
    )
    for name in reversed(("i", "j", "k")):
        stmt = ForAll(ivs[name], stmt)
    return stmt


def build_broadcast_row():
    """C[i, j] = a[i]: the j coordinate is driven by the result alone."""

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="d")
    c = TensorVar("C", fmt="dd")
    return ForAll(i, ForAll(j, TensorAssign(c[i, j], a[i])))


def build_union_add_to_dense():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]))
    return ForAll(i, ForAll(j, assign))


def build_dense_add_2d():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    return ForAll(
        i,
        ForAll(
            j,
            TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j])),
        ),
    )


def build_dense_add_3d():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="ddd")
    b = TensorVar("B", fmt="ddd")
    c = TensorVar("C", fmt="ddd")
    return ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(
                    c[i, j, k],
                    CINBinaryOp(Operation.ADD, a[i, j, k], b[i, j, k]),
                ),
            ),
        ),
    )


def build_vector_add():
    i = IndexVar("i")
    a = TensorVar("A", fmt="d")
    b = TensorVar("B", fmt="d")
    c = TensorVar("C", fmt="d")
    return ForAll(i, TensorAssign(c[i], CINBinaryOp(Operation.ADD, a[i], b[i])))


def tile(index_var, width, placement="outermost", unroll=False, accum="direct"):
    return TileSpec(index_var, width, placement=placement, accum=accum, unroll=unroll)


def scheduled_options(schedule):
    return CompileOptions.from_environment(requested_schedule=schedule)


MATMUL_BINDINGS = (((4, 5), F32), ((5, 6), F32))
MATMUL_BINDINGS_F64 = (((4, 5), F64), ((5, 6), F64))
SPMM_BINDINGS = (((4, 5), F32), ((5, 6), F32))

SCHEDULED_PARITY_GRID = [
    (
        "two-reduction identity order",
        build_two_reduction,
        Schedule(loop_order=("i", "j", "k")),
        (4,),
        (((4, 5), F32), ((4, 6), F32)),
    ),
    (
        "two-reduction real reorder",
        build_two_reduction,
        Schedule(loop_order=("i", "k", "j")),
        (4,),
        (((4, 5), F32), ((4, 6), F32)),
    ),
    (
        "matmul reorder from ijk source, f64",
        lambda: build_matmul(order=("i", "j", "k"), dtype=F64),
        Schedule(loop_order=("i", "k", "j")),
        (4, 6),
        MATMUL_BINDINGS_F64,
    ),
    (
        "matmul tile-j ragged",
        build_matmul,
        Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", 4),)),
        (4, 6),
        MATMUL_BINDINGS,
    ),
    (
        "matmul tile-j exact",
        build_matmul,
        Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", 4),)),
        (4, 4),
        (((4, 5), F32), ((5, 4), F32)),
    ),
    (
        "matmul tile-j width above extent",
        build_matmul,
        Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", 4),)),
        (4, 2),
        (((4, 5), F32), ((5, 2), F32)),
    ),
    (
        "matmul tile-j f64",
        lambda: build_matmul(dtype=F64),
        Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", 4),)),
        (4, 6),
        MATMUL_BINDINGS_F64,
    ),
    (
        "matmul tile-j unroll",
        build_matmul,
        Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", 4, unroll=True),)),
        (4, 6),
        MATMUL_BINDINGS,
    ),
    (
        "matmul tile-j child_of:i",
        build_matmul,
        Schedule(
            loop_order=("i", "k", "j"),
            tiles=(tile("j", 4, placement="child_of:i"),),
        ),
        (4, 6),
        MATMUL_BINDINGS,
    ),
    (
        "matmul tile-j at_depth:2",
        build_matmul,
        Schedule(
            loop_order=("i", "k", "j"),
            tiles=(tile("j", 4, placement="at_depth:2"),),
        ),
        (4, 6),
        MATMUL_BINDINGS,
    ),
    (
        "matmul two splits i2 and j4",
        build_matmul,
        Schedule(
            loop_order=("i", "k", "j"),
            tiles=(tile("i", 2), tile("j", 4)),
        ),
        (4, 6),
        MATMUL_BINDINGS,
    ),
    (
        "matmul tile-j zero extent",
        build_matmul,
        Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", 4),)),
        (4, 0),
        (((4, 5), F32), ((5, 0), F32)),
    ),
    (
        "spmm untiled scheduled",
        build_spmm,
        Schedule(loop_order=("i", "j", "k")),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm tile-k ragged",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(tile("k", 4),)),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm tile-k exact",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(tile("k", 4),)),
        (4, 4),
        (((4, 5), F32), ((5, 4), F32)),
    ),
    (
        "spmm tile-k width above extent",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(tile("k", 4),)),
        (4, 2),
        (((4, 5), F32), ((5, 2), F32)),
    ),
    (
        "spmm tile-k f64",
        lambda: build_spmm(dtype=F64),
        Schedule(loop_order=("i", "j", "k"), tiles=(tile("k", 4),)),
        (4, 6),
        (((4, 5), F64), ((5, 6), F64)),
    ),
    (
        "spmm tile-i ragged rows",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(tile("i", 3),)),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm tile-k zero rows",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(tile("k", 4),)),
        (0, 6),
        (((0, 5), F32), ((5, 6), F32)),
    ),
]


@pytest.mark.parametrize(
    "case",
    SCHEDULED_PARITY_GRID,
    ids=[case[0] for case in SCHEDULED_PARITY_GRID],
)
def test_scheduled_source_is_byte_identical_to_legacy(case):
    name, build, schedule, result_shape, bindings = case
    comparison = compare_generated_sources(
        build(),
        result_shape,
        bindings,
        compile_options=scheduled_options(schedule),
    )
    assert comparison.identical, f"{name} diverged from the legacy schedule"


def test_scheduled_structural_activation():
    """The scheduled kernels carry the exact legacy schedule structure."""

    schedule = Schedule(loop_order=("i", "j", "k"), tiles=(tile("k", 4),))
    kernel = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(schedule),
    )
    source = kernel.cpp_source
    assert "constexpr int kTile_k = 4;" in source
    assert "for (int64_t k_out = 0; k_out < B1_size; k_out += kTile_k)" in source
    assert "int64_t k = k_out + k_in;" in source
    assert "if (k >= B1_size)" in source
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(-1, ((B1_size + kTile_k - 1) / kTile_k)))"
        in source
    )
    assert "__builtin_prefetch" in source

    unrolled = compile_cin_via_loopir(
        build_matmul(),
        (4, 6),
        MATMUL_BINDINGS,
        compile_options=scheduled_options(
            Schedule(
                loop_order=("i", "k", "j"),
                tiles=(tile("j", 4, unroll=True),),
            )
        ),
    )
    assert "#pragma unroll" in unrolled.cpp_source

    tile_i = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(
            Schedule(loop_order=("i", "j", "k"), tiles=(tile("i", 3),))
        ),
    )
    # The nnz-aware row policy survives on the split row loop.
    assert (
        "num_threads(scorch_nthreads(A1_pos[A0_size], "
        "((A0_size + kTile_i - 1) / kTile_i)))" in tile_i.cpp_source
    )


def test_scheduled_artifact_carries_plan_and_provenance():
    schedule = Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", 4),))
    kernel = compile_cin_via_loopir(
        build_matmul(),
        (4, 6),
        MATMUL_BINDINGS,
        compile_options=scheduled_options(schedule),
    )
    scheduled = kernel.schedule
    assert scheduled is not None
    assert scheduled.plan.provenance == "explicit"
    assert scheduled.plan.tiles[0].width == 4
    parts = [(entry.part, entry.tile is not None) for entry in scheduled.loops]
    assert parts == [
        (LoopPart.OUTER, True),
        (LoopPart.LOGICAL, False),
        (LoopPart.LOGICAL, False),
        (LoopPart.INNER, True),
    ]
    # The unscheduled base program is retained and unscheduled.
    from scorch.compiler.loopir.printer import canonical_program_dump

    assert "tile_outer_for" not in canonical_program_dump(scheduled.base_program)
    assert "tile_outer_for" in canonical_program_dump(scheduled.program)


def test_scheduled_stage_sequence_and_failure_ownership():
    schedule = Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", 4),))
    options = scheduled_options(schedule)
    context = CompilationContext(options)
    compile_cin_via_loopir(
        build_matmul(),
        (4, 6),
        MATMUL_BINDINGS,
        compile_options=options,
        compilation_context=context,
    )
    stages = [record.stage_id for record in context.stage_run_records]
    assert stages == [
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION,
        CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION,
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        CompilerStageId.CIN_TO_LOOPIR_LOWERING,
        CompilerStageId.LOOPIR_SCHEDULE_APPLICATION,
        CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
        CompilerStageId.LLIR_TO_CPP_GENERATION,
    ]

    # A plan family the passes refuse fails the schedule-application stage:
    # no record is published for it and the context refuses later work,
    # naming the failed stage.
    from scorch.compiler.compilation_context import CompilationContextError

    failing = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(tile("k", 4, accum="stack"),),
    )
    failing_options = scheduled_options(failing)
    failing_context = CompilationContext(failing_options)
    with pytest.raises(SchedulePassError) as error:
        compile_cin_via_loopir(
            build_spmm(),
            (4, 6),
            SPMM_BINDINGS,
            compile_options=failing_options,
            compilation_context=failing_context,
        )
    assert error.value.defect.code == "unsupported_schedule_accumulation"
    completed = [record.stage_id for record in failing_context.stage_run_records]
    assert completed[-1] is CompilerStageId.CIN_TO_LOOPIR_LOWERING
    assert CompilerStageId.LOOPIR_SCHEDULE_APPLICATION not in completed
    with pytest.raises(CompilationContextError) as context_error:
        failing_context.begin_stage(
            CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
            compile_options=failing_options,
        )
    assert "loopir_schedule_application" in str(context_error.value)


@pytest.mark.parametrize(
    "schedule, code",
    [
        (
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(tile("k", 4, accum="stack"),),
            ),
            "unsupported_schedule_accumulation",
        ),
        (
            # Legacy heap accumulation also demands the parallel row loop;
            # the heap result tile is rejected before parallel selection.
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(tile("k", 4, accum="heap"),),
                parallel_loop="i",
            ),
            "unsupported_schedule_result_tile",
        ),
        (
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(
                    tile("k", 4),
                    TileSpec("j", 3, kind="panel", accum="direct"),
                ),
                parallel_loop="i",
            ),
            "unsupported_schedule_panel",
        ),
        (
            Schedule(loop_order=("i", "j", "k"), parallel_loop="i"),
            "unsupported_schedule_parallel",
        ),
        (
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(tile("k", 4, unroll=False),),
                parallel_loop="i",
            ),
            "unsupported_schedule_parallel",
        ),
    ],
)
def test_unsupported_schedule_families_fail_closed(schedule, code):
    with pytest.raises(SchedulePassError) as error:
        compile_cin_via_loopir(
            build_spmm(),
            (4, 6),
            SPMM_BINDINGS,
            compile_options=scheduled_options(schedule),
        )
    assert error.value.defect.code == code, error.value.defect


def test_empty_schedule_stays_on_the_auto_family_and_fails_closed():
    with pytest.raises(SchedulePassError) as error:
        compile_cin_via_loopir(
            build_spmm(),
            (4, 6),
            SPMM_BINDINGS,
            compile_options=scheduled_options(Schedule()),
        )
    assert error.value.defect.code == "unsupported_schedule_provenance"


def test_unscheduled_pipeline_is_unchanged_by_the_strangler_entry():
    kernel = compile_cin_via_loopir(build_matmul(), (4, 6), MATMUL_BINDINGS)
    assert kernel.schedule is None
    comparison = compare_generated_sources(build_matmul(), (4, 6), MATMUL_BINDINGS)
    assert comparison.identical


def test_tile_only_schedule_spelling_replays_legacy_canonically():
    schedule = Schedule(tiles=(tile("j", 4),))
    comparison = compare_generated_sources(
        build_matmul(),
        (4, 6),
        MATMUL_BINDINGS,
        compile_options=scheduled_options(schedule),
    )
    assert comparison.identical


def test_tile_only_shadow_freezes_the_policy_selected_order(monkeypatch):
    original = Scheduler.select_loop_order
    calls = []

    def select_once(*args, **kwargs):
        calls.append(None)
        if len(calls) > 1:
            raise AssertionError("verified shadow plan must not be selected again")
        return original(*args, **kwargs)

    monkeypatch.setattr(Scheduler, "select_loop_order", select_once)
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    assert_scheduled_shadow(
        build_matmul(),
        Schedule(tiles=(tile("j", 3),)),
        (3, 5),
        (dense_stensor(a, "A"), dense_stensor(b, "B")),
        a @ b,
    )
    assert len(calls) == 1


def test_broadcast_tile_bounds_from_the_result_like_legacy():
    """A split coordinate only the result drives is bounded and guarded by
    the result's dimension size, exactly as the legacy lattice resolves it."""

    schedule = Schedule(loop_order=("i", "j"), tiles=(tile("j", 4),))
    comparison = compare_generated_sources(
        build_broadcast_row(),
        (3, 6),
        (((3,), F32),),
        compile_options=scheduled_options(schedule),
    )
    assert comparison.identical
    assert "for (int64_t j_out = 0; j_out < C1_size" in comparison.loopir_cpp
    assert "if (j >= C1_size)" in comparison.loopir_cpp


def test_splits_over_merged_iteration_stay_uncompiled():
    """Verifier-legal splits over merged nests are outside the emitted
    families: the target boundary owns the fail-closed refusal."""

    schedule = Schedule(loop_order=("i", "j"), tiles=(tile("i", 2),))
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_union_add_to_dense(),
            (3, 4),
            (((3, 4), F32), ((3, 4), F32)),
            compile_options=scheduled_options(schedule),
        )
    assert error.value.defect.code == "unsupported_program_shape"
    assert "merged" in error.value.defect.message


# -- compiled execution -------------------------------------------------------


def dense_stensor(tensor, name):
    return STensor.from_torch(tensor, name).to_dense()


def csr_stensor(tensor, name):
    return STensor.from_torch(tensor, name).to_sparse("ds")


def assert_scheduled_shadow(
    cin,
    schedule,
    result_shape,
    args,
    reference,
    *,
    atol=1e-3,
    rtol=1e-3,
):
    loopir_result, legacy_result, comparison = execute_shadow(
        cin,
        result_shape,
        *args,
        compile_options=scheduled_options(schedule),
    )
    assert comparison.identical
    assert torch.equal(loopir_result.values, legacy_result.values)
    assert torch.allclose(
        loopir_result.values.reshape(result_shape),
        reference,
        atol=atol,
        rtol=rtol,
    )


def test_matmul_ragged_tile_shadow_execution():
    torch.manual_seed(2607)
    a = torch.randn(4, 5)
    b = torch.randn(5, 6)
    assert_scheduled_shadow(
        build_matmul(),
        Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", 4),)),
        (4, 6),
        (dense_stensor(a, "A"), dense_stensor(b, "B")),
        a @ b,
    )


def test_matmul_reordered_shadow_execution():
    torch.manual_seed(2608)
    a = torch.randn(5, 4)
    b = torch.randn(4, 3)
    assert_scheduled_shadow(
        build_matmul(order=("i", "j", "k")),
        Schedule(loop_order=("i", "k", "j")),
        (5, 3),
        (dense_stensor(a, "A"), dense_stensor(b, "B")),
        a @ b,
    )


@pytest.mark.parametrize(
    "schedule",
    [
        Schedule(
            loop_order=("i", "k", "j"),
            tiles=(tile("j", 3, placement="child_of:i"),),
        ),
        Schedule(
            loop_order=("i", "k", "j"),
            tiles=(tile("j", 3, placement="at_depth:2"),),
        ),
        Schedule(
            loop_order=("i", "k", "j"),
            tiles=(tile("j", 3, unroll=True),),
        ),
        Schedule(
            loop_order=("i", "k", "j"),
            tiles=(tile("i", 2), tile("j", 3)),
        ),
    ],
    ids=("child-of", "at-depth", "unroll", "two-splits"),
)
def test_matmul_schedule_placement_matrix_shadow_execution(schedule):
    torch.manual_seed(2618)
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    assert_scheduled_shadow(
        build_matmul(),
        schedule,
        (3, 5),
        (dense_stensor(a, "A"), dense_stensor(b, "B")),
        a @ b,
    )


@pytest.mark.parametrize("free_dim", [2, 4, 6])
def test_spmm_tile_k_shadow_execution_across_tile_regimes(free_dim):
    torch.manual_seed(2609 + free_dim)
    sparse = torch.randn(4, 5)
    sparse[sparse.abs() < 0.6] = 0.0
    sparse[2, :] = 0.0  # an empty CSR row
    dense = torch.randn(5, free_dim)
    assert_scheduled_shadow(
        build_spmm(),
        Schedule(loop_order=("i", "j", "k"), tiles=(tile("k", 4),)),
        (4, free_dim),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


def test_spmm_tile_i_shadow_execution():
    torch.manual_seed(2613)
    sparse = torch.randn(4, 5)
    sparse[sparse.abs() < 0.5] = 0.0
    dense = torch.randn(5, 6)
    assert_scheduled_shadow(
        build_spmm(),
        Schedule(loop_order=("i", "j", "k"), tiles=(tile("i", 3),)),
        (4, 6),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


def test_spmm_tile_k_float64_shadow_execution():
    torch.manual_seed(2614)
    sparse = torch.randn(3, 4, dtype=F64)
    sparse[sparse.abs() < 0.5] = 0.0
    dense = torch.randn(4, 5, dtype=F64)
    assert_scheduled_shadow(
        build_spmm(dtype=F64),
        Schedule(loop_order=("i", "j", "k"), tiles=(tile("k", 3),)),
        (3, 5),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
        atol=1e-10,
        rtol=1e-10,
    )


def test_matmul_zero_tile_extent_shadow_execution():
    a = torch.randn(3, 4)
    b = torch.empty(4, 0)
    assert_scheduled_shadow(
        build_matmul(),
        Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", 4),)),
        (3, 0),
        (dense_stensor(a, "A"), dense_stensor(b, "B")),
        a @ b,
    )


def test_matmul_tile_f64_execution_matches_torch_and_oracle():
    torch.manual_seed(2615)
    a = torch.randn(3, 4, dtype=torch.float64)
    b = torch.randn(4, 7, dtype=torch.float64)
    schedule = Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", 4),))
    result, kernel = execute_cin_via_loopir(
        build_matmul(dtype=F64),
        (3, 7),
        dense_stensor(a, "A"),
        dense_stensor(b, "B"),
        compile_options=scheduled_options(schedule),
    )
    assert kernel.schedule is not None
    assert torch.allclose(result.values.reshape(3, 7), a @ b, atol=1e-9, rtol=1e-9)
    from scorch.compiler.loopir.oracle import run_program

    lowering = kernel.lowering
    oracle_out = run_program(
        kernel.schedule.program,
        {
            lowering.input_symbols[0]: a.tolist(),
            lowering.input_symbols[1]: b.tolist(),
        },
        {lowering.result_symbol: (3, 7)},
    )[lowering.result_symbol]
    assert torch.allclose(
        result.values.reshape(3, 7),
        torch.tensor(oracle_out, dtype=torch.float64),
        atol=1e-9,
        rtol=1e-9,
    )


def test_randomized_scheduled_execution_matches_torch():
    torch.manual_seed(2616)
    import random as _random

    rng = _random.Random(20260723)
    for _ in range(3):
        rows = rng.randrange(1, 7)
        inner = rng.randrange(1, 7)
        cols = rng.randrange(1, 9)
        width = rng.choice((2, 3, 4))
        a = torch.randn(rows, inner)
        b = torch.randn(inner, cols)
        schedule = Schedule(loop_order=("i", "k", "j"), tiles=(tile("j", width),))
        result, _ = execute_cin_via_loopir(
            build_matmul(),
            (rows, cols),
            dense_stensor(a, "A"),
            dense_stensor(b, "B"),
            compile_options=scheduled_options(schedule),
        )
        assert torch.allclose(
            result.values.reshape(rows, cols), a @ b, atol=1e-3, rtol=1e-3
        )


def test_scheduled_relayout_prerequisite_uses_schedule_free_options(monkeypatch):
    torch.manual_seed(2617)
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 3, 4)
    a_st = STensor.from_torch(a, "A", mode_order=[1, 0, 2]).to_dense()
    b_st = STensor.from_torch(b, "B", mode_order=[1, 0, 2]).to_dense()
    observed = []
    original = STensor.change_mode_order

    def capture_options(self, mode_order, **kwargs):
        observed.append(
            (
                kwargs.get("_compile_options"),
                kwargs.get("_compilation_context"),
            )
        )
        return original(self, mode_order, **kwargs)

    monkeypatch.setattr(STensor, "change_mode_order", capture_options)
    options = scheduled_options(Schedule(loop_order=("i", "j", "k")))
    context = CompilationContext(options)
    result, _ = execute_cin_via_loopir(
        build_dense_add_3d(),
        (2, 3, 4),
        a_st,
        b_st,
        compile_options=options,
        _compilation_context=context,
    )
    assert torch.equal(result.to_torch(), a + b)
    assert CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION in tuple(
        record.stage_id for record in context.stage_run_records
    )
    assert observed
    for options, context in observed:
        assert options.requested_schedule is None
        assert context.compile_options is options


def test_scheduled_relayout_failure_retires_the_parent_compilation(monkeypatch):
    a = STensor.from_torch(
        torch.randn(2, 3, 4),
        "A",
        mode_order=[1, 0, 2],
    ).to_dense()
    b = STensor.from_torch(
        torch.randn(2, 3, 4),
        "B",
        mode_order=[1, 0, 2],
    ).to_dense()
    options = scheduled_options(Schedule(loop_order=("i", "j", "k")))
    context = CompilationContext(options)
    injected = RuntimeError("injected scheduled relayout failure")

    def fail_relayout(*_args, **_kwargs):
        raise injected

    monkeypatch.setattr(STensor, "change_mode_order", fail_relayout)
    with pytest.raises(RuntimeError) as error:
        execute_cin_via_loopir(
            build_dense_add_3d(),
            (2, 3, 4),
            a,
            b,
            compile_options=options,
            _compilation_context=context,
        )
    assert error.value is injected
    assert CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION not in tuple(
        record.stage_id for record in context.stage_run_records
    )
    with pytest.raises(CompilationContextError) as terminal:
        context.begin_stage(
            CompilerStageId.CIN_TO_LOOPIR_LOWERING,
            compile_options=options,
        )
    assert terminal.value.diagnostic.code == "failed_compilation"


@pytest.mark.parametrize("shape", [(3, 3), (3, 4)])
def test_scheduled_shadow_aligns_nonidentity_runtime_layouts(shape):
    rows, cols = shape
    a = torch.arange(rows * cols, dtype=F32).reshape(rows, cols)
    b = torch.arange(rows * cols, dtype=F32).reshape(rows, cols) * 3
    a_st = STensor.from_torch(a, "A", mode_order=[1, 0]).to_dense()
    b_st = STensor.from_torch(b, "B", mode_order=[1, 0]).to_dense()
    loopir, legacy, comparison = execute_shadow(
        build_dense_add_2d(),
        shape,
        a_st,
        b_st,
        compile_options=scheduled_options(Schedule(loop_order=("i", "j"))),
    )
    expected = a + b
    assert comparison.identical
    assert torch.equal(loopir.to_torch(), expected)
    assert torch.equal(legacy.to_torch(), expected)
    assert torch.equal(loopir.to_torch(), legacy.to_torch())


def test_scheduled_shadow_wraps_dense_result_rank_from_cin():
    a = torch.arange(5, dtype=F32)
    b = torch.arange(5, dtype=F32) * 2
    loopir, legacy, comparison = execute_shadow(
        build_vector_add(),
        (5,),
        dense_stensor(a, "A"),
        dense_stensor(b, "B"),
        compile_options=scheduled_options(Schedule(loop_order=("i",))),
    )
    assert comparison.identical
    assert str(loopir.format) == "d"
    assert str(legacy.format) == "d"
    assert torch.equal(loopir.to_torch(), a + b)
    assert torch.equal(legacy.to_torch(), a + b)
