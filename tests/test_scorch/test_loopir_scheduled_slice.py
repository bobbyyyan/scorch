"""The Phase-6 scheduled vertical slice: byte parity, execution, boundaries.

Every schedule the LoopIR strangler entry accepts is compared against the
production legacy scheduled route (``Scheduler.apply_schedule`` followed by
the legacy lowering of the verified ``ScheduledCIN``):

- a nineteen-member scheduled byte-parity grid (plus a result-bounded
  broadcast split) locks generated C++ equality across explicit reorders
  and affine ``accum='direct'`` tiles — every placement kind, unroll
  on/off, one and two splits, tile-i/tile-k over CSR SpMM, f32/f64, and N
  below/equal to/above/not divisible by the width, including zero extents;
- a thirteen-member stack byte-parity grid locks the ``accum='stack'``
  workspace family the same way — the legacy ``wksp[kTile]``
  producer/consumer shape over CSR SpMM and dense matmul, every placement
  kind, unroll, f32/f64, direct+stack composition, and ragged/exact/
  oversized/non-dividing/zero extents;
- a thirteen-member panel byte-parity grid locks the sparse coordinate-window
  family — the legacy tile-j windowed-CSR shape with its mandatory parallel
  row loop — across widths below/equal/above/not dividing the extent, unit
  and maximum constexpr widths, f32/f64, both supported placements
  (outermost and child_of an outermost affine origin, plus the
  both-outermost composition), and zero row/panel/free extents;
- compiled shadow execution runs both pipelines on real tensors and
  requires bitwise-equal dense results plus PyTorch agreement;
- every unsupported schedule family fails closed with a stable code at the
  schedule-application boundary and the failure owns its stage record;
- the target lowering keeps its own fail-closed boundary for scheduled
  shapes it does not emit (splits over merged iteration or ordered
  assembly).
"""

import os
from pathlib import Path
import subprocess
import sys

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


def build_dense_driver_before_csr():
    """C[i,k,j] = X[i,k] * A[i,j], with X owning the row-bound spelling."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    x = TensorVar("X", fmt="dd")
    a = TensorVar("A", fmt="ds")
    c = TensorVar("C", fmt="ddd")
    return ForAll(
        i,
        ForAll(
            k,
            ForAll(
                j,
                TensorAssign(
                    c[i, k, j],
                    CINBinaryOp(Operation.MUL, x[i, k], a[i, j]),
                ),
            ),
        ),
    )


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
        tiles=(TileSpec("k", 4, placement="outermost", parallel=True, unroll=False),),
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
    assert error.value.defect.code == "unsupported_schedule_parallel"
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
        # Stack accumulation left this list in the Phase-6 workspace slice,
        # sparse panel tiling in the panel slice, heap result tiles in the
        # heap slice, and explicit parallel_loop plans in the abstract
        # parallel-selection slice (their positive twins live in the anchor
        # parity grid).  Parallel tile selection is the remaining
        # unmigrated explicit-parallel spelling.
        (
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(
                    TileSpec(
                        "k", 4, placement="outermost", parallel=True, unroll=False
                    ),
                ),
            ),
            "unsupported_schedule_parallel",
        ),
        (
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(
                    TileSpec(
                        "k",
                        4,
                        placement="outermost",
                        accum="stack",
                        parallel=True,
                        unroll=False,
                    ),
                ),
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


# -- stack accumulation (workspace) slice -------------------------------------


def stack(index_var, width, placement="outermost", unroll=False):
    return tile(index_var, width, placement=placement, unroll=unroll, accum="stack")


STACK_PARITY_GRID = [
    (
        "spmm stack-k ragged",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(stack("k", 4),)),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm stack-k exact",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(stack("k", 4),)),
        (4, 4),
        (((4, 5), F32), ((5, 4), F32)),
    ),
    (
        "spmm stack-k width above extent",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(stack("k", 4),)),
        (4, 2),
        (((4, 5), F32), ((5, 2), F32)),
    ),
    (
        "spmm stack-k width not dividing extent",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(stack("k", 4),)),
        (4, 5),
        (((4, 5), F32), ((5, 5), F32)),
    ),
    (
        "spmm stack-k unroll",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(stack("k", 4, unroll=True),)),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm stack-k child_of:i",
        build_spmm,
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(stack("k", 4, placement="child_of:i"),),
        ),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm stack-k at_depth:1",
        build_spmm,
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(stack("k", 4, placement="at_depth:1"),),
        ),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm stack-k f64",
        lambda: build_spmm(dtype=F64),
        Schedule(loop_order=("i", "j", "k"), tiles=(stack("k", 4),)),
        (4, 6),
        (((4, 5), F64), ((5, 6), F64)),
    ),
    (
        "spmm stack-k zero free extent",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(stack("k", 4),)),
        (4, 0),
        (((4, 5), F32), ((5, 0), F32)),
    ),
    (
        "spmm stack-k zero rows",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(stack("k", 4),)),
        (0, 6),
        (((0, 5), F32), ((5, 6), F32)),
    ),
    (
        "matmul stack-j ragged",
        build_matmul,
        Schedule(loop_order=("i", "k", "j"), tiles=(stack("j", 4),)),
        (4, 6),
        MATMUL_BINDINGS,
    ),
    (
        "matmul stack-j f64",
        lambda: build_matmul(dtype=F64),
        Schedule(loop_order=("i", "k", "j"), tiles=(stack("j", 4),)),
        (4, 6),
        MATMUL_BINDINGS_F64,
    ),
    (
        "spmm direct-i plus stack-k",
        build_spmm,
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(
                tile("i", 2),
                stack("k", 4, placement="child_of:i_out"),
            ),
        ),
        (4, 6),
        SPMM_BINDINGS,
    ),
]


@pytest.mark.parametrize(
    "case",
    STACK_PARITY_GRID,
    ids=[case[0] for case in STACK_PARITY_GRID],
)
def test_stack_source_is_byte_identical_to_legacy(case):
    name, build, schedule, result_shape, bindings = case
    comparison = compare_generated_sources(
        build(),
        result_shape,
        bindings,
        compile_options=scheduled_options(schedule),
    )
    assert comparison.identical, f"{name} diverged from the legacy schedule"


def test_stack_structural_activation():
    """The stack kernels carry the exact legacy workspace structure."""

    schedule = Schedule(loop_order=("i", "j", "k"), tiles=(stack("k", 4, unroll=True),))
    kernel = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(schedule),
    )
    source = kernel.cpp_source
    # Allocation and zero-reset in one statement, per row iteration.
    assert "constexpr int kTile_k = 4;" in source
    assert "float wksp[kTile_k] = {};" in source
    # Producer: input-bounded point loop accumulating into the workspace,
    # with the sparse prefetch surviving the managed passes.
    assert "if (k >= B1_size) {" in source
    assert "wksp[k_in] += A_val[pA1] * B_val[pB1];" in source
    assert "__builtin_prefetch" in source
    # Consumer: the synthesized result-bounded copy-out loop.
    assert "// Lower consumer CIN" in source
    assert "if (k >= C1_size) {" in source
    assert "int64_t pC1 = pC0 * C1_size + k;" in source
    assert "C_values[pC1] += wksp[k_in];" in source
    # Both point loops carry the requested unroll preference.
    assert source.count("#pragma unroll") == 2
    # Zero-fill and the ceil-trip-count parallel policy on the origin loop.
    assert "scorch_zero_dense(C_values, C_capacity);" in source
    assert (
        "num_threads(scorch_nthreads(-1, ((B1_size + kTile_k - 1) / kTile_k)))"
        in source
    )
    # The scheduling artifact retains the region chain provenance:
    # origin, row, reduction, producer point, consumer point.
    assert kernel.schedule is not None
    parts = [entry.part for entry in kernel.schedule.loops]
    assert parts == [
        LoopPart.OUTER,
        LoopPart.LOGICAL,
        LoopPart.LOGICAL,
        LoopPart.INNER,
        LoopPart.INNER,
    ]


def test_stack_child_of_uses_the_nnz_aware_row_policy():
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(stack("k", 4, placement="child_of:i"),),
    )
    kernel = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(schedule),
    )
    assert "num_threads(scorch_nthreads(A1_pos[A0_size], A0_size))" in kernel.cpp_source


@pytest.mark.parametrize("free_dim", [2, 4, 6])
def test_spmm_stack_k_shadow_execution_across_tile_regimes(free_dim):
    torch.manual_seed(2619 + free_dim)
    sparse = torch.randn(4, 5)
    sparse[sparse.abs() < 0.6] = 0.0
    sparse[2, :] = 0.0  # an empty CSR row
    dense = torch.randn(5, free_dim)
    assert_scheduled_shadow(
        build_spmm(),
        Schedule(loop_order=("i", "j", "k"), tiles=(stack("k", 4),)),
        (4, free_dim),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


def test_spmm_stack_k_float64_shadow_execution():
    torch.manual_seed(2620)
    sparse = torch.randn(3, 4, dtype=F64)
    sparse[sparse.abs() < 0.5] = 0.0
    dense = torch.randn(4, 5, dtype=F64)
    assert_scheduled_shadow(
        build_spmm(dtype=F64),
        Schedule(loop_order=("i", "j", "k"), tiles=(stack("k", 3),)),
        (3, 5),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
        atol=1e-10,
        rtol=1e-10,
    )


def test_matmul_stack_shadow_execution_with_unroll():
    torch.manual_seed(2621)
    a = torch.randn(4, 5)
    b = torch.randn(5, 6)
    assert_scheduled_shadow(
        build_matmul(),
        Schedule(loop_order=("i", "k", "j"), tiles=(stack("j", 4, unroll=True),)),
        (4, 6),
        (dense_stensor(a, "A"), dense_stensor(b, "B")),
        a @ b,
    )


def test_stack_two_split_shadow_execution():
    torch.manual_seed(2622)
    sparse = torch.randn(4, 5)
    sparse[sparse.abs() < 0.5] = 0.0
    dense = torch.randn(5, 6)
    assert_scheduled_shadow(
        build_spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(
                tile("i", 2),
                stack("k", 4, placement="child_of:i_out"),
            ),
        ),
        (4, 6),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


def test_stack_zero_extent_shadow_execution():
    a = torch.randn(3, 4)
    b = torch.empty(4, 0)
    assert_scheduled_shadow(
        build_matmul(),
        Schedule(loop_order=("i", "k", "j"), tiles=(stack("j", 4),)),
        (3, 0),
        (dense_stensor(a, "A"), dense_stensor(b, "B")),
        a @ b,
    )


def test_randomized_stack_execution_matches_torch_and_oracle():
    torch.manual_seed(2623)
    import random as _random

    from scorch.compiler.loopir.oracle import run_program

    rng = _random.Random(20260724)
    for _ in range(3):
        rows = rng.randrange(1, 7)
        inner = rng.randrange(2, 7)
        cols = rng.randrange(1, 9)
        width = rng.choice((2, 3, 4))
        a = torch.randn(rows, inner)
        b = torch.randn(inner, cols)
        schedule = Schedule(loop_order=("i", "k", "j"), tiles=(stack("j", width),))
        result, kernel = execute_cin_via_loopir(
            build_matmul(),
            (rows, cols),
            dense_stensor(a, "A"),
            dense_stensor(b, "B"),
            compile_options=scheduled_options(schedule),
        )
        assert torch.allclose(
            result.values.reshape(rows, cols), a @ b, atol=1e-3, rtol=1e-3
        )
        assert kernel.schedule is not None
        lowering = kernel.lowering
        oracle_out = run_program(
            kernel.schedule.program,
            {
                lowering.input_symbols[0]: a.tolist(),
                lowering.input_symbols[1]: b.tolist(),
            },
            {lowering.result_symbol: (rows, cols)},
        )[lowering.result_symbol]
        assert torch.allclose(
            result.values.reshape(rows, cols),
            torch.tensor(oracle_out, dtype=torch.float32),
            atol=1e-3,
            rtol=1e-3,
        )


# -- sparse panel windows (Phase-6 panel slice) --------------------------------


def panel(index_var, width, placement="outermost", unroll=True):
    return TileSpec(
        index_var,
        width,
        placement=placement,
        kind="panel",
        accum="direct",
        unroll=unroll,
    )


def panel_schedule(width, tag, placement="outermost", extra_tiles=(), unroll=True):
    return Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            *extra_tiles,
            panel("j", width, placement=placement, unroll=unroll),
        ),
        tag=tag,
        parallel_loop="i",
    )


PANEL_PARITY_GRID = [
    (
        "spmm panel width below extent",
        build_spmm,
        panel_schedule(3, "p-below"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm panel unit width",
        build_spmm,
        panel_schedule(1, "p-unit"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm panel width equal to extent",
        build_spmm,
        panel_schedule(5, "p-equal"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm panel width above extent",
        build_spmm,
        panel_schedule(64, "p-above"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm panel width not dividing extent",
        build_spmm,
        panel_schedule(2, "p-ragged"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm panel maximum constexpr width",
        build_spmm,
        panel_schedule(2**31 - 1, "p-max"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm panel float64",
        lambda: build_spmm(dtype=F64),
        panel_schedule(3, "p-f64"),
        (4, 6),
        (((4, 5), F64), ((5, 6), F64)),
    ),
    (
        "spmm panel unroll compatibility false",
        build_spmm,
        panel_schedule(3, "p-unroll-false", unroll=False),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm panel child_of affine pack tile",
        build_spmm,
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(
                TileSpec("k", 4, placement="outermost", accum="direct", unroll=False),
                panel("j", 3, placement="child_of:k_out"),
            ),
            tag="p-child",
            parallel_loop="i",
        ),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm panel outermost over outermost affine tile",
        build_spmm,
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(
                TileSpec("k", 4, placement="outermost", accum="direct", unroll=False),
                panel("j", 3),
            ),
            tag="p-both-outermost",
            parallel_loop="i",
        ),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm panel zero rows",
        build_spmm,
        panel_schedule(3, "p-zero-rows"),
        (0, 6),
        (((0, 5), F32), ((5, 6), F32)),
    ),
    (
        "spmm panel zero panel extent",
        build_spmm,
        panel_schedule(3, "p-zero-cols"),
        (4, 6),
        (((4, 0), F32), ((0, 6), F32)),
    ),
    (
        "spmm panel zero free extent",
        build_spmm,
        panel_schedule(3, "p-zero-free"),
        (4, 0),
        (((4, 5), F32), ((5, 0), F32)),
    ),
]


@pytest.mark.parametrize(
    "case",
    PANEL_PARITY_GRID,
    ids=[case[0] for case in PANEL_PARITY_GRID],
)
def test_panel_source_is_byte_identical_to_legacy(case):
    name, build, schedule, result_shape, bindings = case
    comparison = compare_generated_sources(
        build(),
        result_shape,
        bindings,
        compile_options=scheduled_options(schedule),
    )
    assert comparison.identical, f"{name} diverged from the legacy schedule"


def test_panel_artifact_carries_plan_and_provenance():
    kernel = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(panel_schedule(3, "p-prov")),
    )
    scheduled = kernel.schedule
    assert scheduled is not None
    assert scheduled.plan.provenance == "explicit"
    panel_tiles = [t for t in scheduled.plan.tiles if t.kind == "panel"]
    assert len(panel_tiles) == 1 and panel_tiles[0].width == 3
    assert len(scheduled.plan.panel_bounds) == 1
    assert scheduled.plan.parallel_loop is not None
    parts = [(entry.part, entry.tile is not None) for entry in scheduled.loops]
    assert parts == [
        (LoopPart.OUTER, True),
        (LoopPart.LOGICAL, False),
        (LoopPart.INNER, True),
        (LoopPart.LOGICAL, False),
    ]
    from scorch.compiler.loopir.printer import canonical_program_dump

    assert "panel_outer_for" not in canonical_program_dump(scheduled.base_program)
    assert "panel_outer_for" in canonical_program_dump(scheduled.program)
    assert "sparse_window_for" in canonical_program_dump(scheduled.program)


@pytest.mark.parametrize("width", [1, 3, 64])
def test_spmm_panel_shadow_execution_across_window_regimes(width):
    torch.manual_seed(2623 + width)
    sparse = torch.randn(4, 5)
    sparse[sparse.abs() < 0.6] = 0.0
    sparse[2, :] = 0.0  # an empty CSR row
    dense = torch.randn(5, 6)
    assert_scheduled_shadow(
        build_spmm(),
        panel_schedule(width, f"p-shadow-{width}"),
        (4, 6),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


def test_spmm_panel_float64_shadow_execution():
    torch.manual_seed(2624)
    sparse = torch.randn(3, 4, dtype=F64)
    sparse[sparse.abs() < 0.5] = 0.0
    dense = torch.randn(4, 5, dtype=F64)
    assert_scheduled_shadow(
        build_spmm(dtype=F64),
        panel_schedule(3, "p-shadow-f64"),
        (3, 5),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
        atol=1e-10,
        rtol=1e-10,
    )


def test_spmm_panel_child_of_shadow_execution():
    torch.manual_seed(2625)
    sparse = torch.randn(4, 5)
    sparse[sparse.abs() < 0.5] = 0.0
    dense = torch.randn(5, 6)
    assert_scheduled_shadow(
        build_spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(
                TileSpec("k", 4, placement="outermost", accum="direct", unroll=False),
                panel("j", 3, placement="child_of:k_out"),
            ),
            tag="p-shadow-child",
            parallel_loop="i",
        ),
        (4, 6),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


def test_spmm_panel_zero_free_extent_shadow_execution():
    sparse = torch.randn(4, 5)
    sparse[sparse.abs() < 0.5] = 0.0
    dense = torch.empty(5, 0)
    assert_scheduled_shadow(
        build_spmm(),
        panel_schedule(3, "p-shadow-zero"),
        (4, 0),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


@pytest.mark.parametrize(
    ("sparse", "dense"),
    (
        (torch.empty(0, 5), torch.randn(5, 3)),
        (torch.empty(4, 0), torch.empty(0, 3)),
    ),
    ids=("zero-rows", "zero-panel-extent"),
)
def test_spmm_panel_zero_row_and_panel_extents_shadow_execution(sparse, dense):
    result_shape = (sparse.shape[0], dense.shape[1])
    assert_scheduled_shadow(
        build_spmm(),
        panel_schedule(3, f"p-shadow-{result_shape[0]}x{sparse.shape[1]}"),
        result_shape,
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


def test_panel_randomized_execution_matches_torch_and_oracle():
    torch.manual_seed(2626)
    import random as _random

    from scorch.compiler.loopir.levels import CsrMatrix
    from scorch.compiler.loopir.oracle import run_program

    rng = _random.Random(20260724)
    for round_index in range(3):
        rows = rng.randrange(1, 7)
        inner = rng.randrange(1, 7)
        cols = rng.randrange(1, 8)
        width = rng.choice((1, 2, 3, 5, 9))
        sparse = torch.randn(rows, inner)
        sparse[sparse.abs() < 0.5] = 0.0
        dense = torch.randn(inner, cols)
        schedule = panel_schedule(width, f"p-rand-{round_index}-{width}")
        result, kernel = execute_cin_via_loopir(
            build_spmm(),
            (rows, cols),
            csr_stensor(sparse, "A"),
            dense_stensor(dense, "B"),
            compile_options=scheduled_options(schedule),
        )
        assert torch.allclose(
            result.values.reshape(rows, cols),
            sparse @ dense,
            atol=1e-3,
            rtol=1e-3,
        )
        assert kernel.schedule is not None
        lowering = kernel.lowering
        oracle_out = run_program(
            kernel.schedule.program,
            {
                lowering.input_symbols[0]: CsrMatrix.from_dense(sparse.tolist()),
                lowering.input_symbols[1]: dense.tolist(),
            },
            {lowering.result_symbol: (rows, cols)},
        )[lowering.result_symbol]
        assert torch.allclose(
            result.values.reshape(rows, cols),
            torch.tensor(oracle_out, dtype=torch.float32),
            atol=1e-4,
            rtol=1e-4,
        )


def test_panel_schedule_validation_failure_owns_its_stage():
    """A panel schedule without its mandatory parallel row loop fails in
    the shared Scheduler validation with a terminal scheduling stage."""

    from scorch.compiler.diagnostics import InvalidSchedule, UnsupportedFeature

    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(panel("j", 3),),
        tag="p-invalid",
    )
    options = scheduled_options(schedule)
    context = CompilationContext(options)
    with pytest.raises((InvalidSchedule, UnsupportedFeature)):
        compile_cin_via_loopir(
            build_spmm(),
            (4, 6),
            SPMM_BINDINGS,
            compile_options=options,
            compilation_context=context,
        )
    completed = [record.stage_id for record in context.stage_run_records]
    assert CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION not in completed
    with pytest.raises(CompilationContextError):
        context.begin_stage(
            CompilerStageId.CIN_TO_LOOPIR_LOWERING, compile_options=options
        )


# -- the staged-operand relayout slice ----------------------------------------


def relayout_schedule(scope, tag, width=3, strip=4, accum="direct"):
    from scorch.compiler.scheduler import RelayoutSpec

    return Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec("k", strip, placement="outermost", accum=accum, unroll=False),
            panel("j", width, placement="child_of:k_out"),
        ),
        relayout=RelayoutSpec("B", "k", strip, scope_var=scope),
        tag=tag,
        parallel_loop="i",
    )


RELAYOUT_PARITY_GRID = [
    (
        "spmm relayout panel scope",
        build_spmm,
        relayout_schedule("j", "r-panel"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm relayout pack scope",
        build_spmm,
        relayout_schedule("k", "r-pack"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm relayout unit panel and strip",
        build_spmm,
        relayout_schedule("j", "r-unit", width=1, strip=1),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm relayout panel width above extent",
        build_spmm,
        relayout_schedule("j", "r-wide", width=64, strip=8),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm relayout strips not dividing extents",
        build_spmm,
        relayout_schedule("k", "r-ragged", width=2, strip=4),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm relayout float64 panel scope",
        lambda: build_spmm(dtype=F64),
        relayout_schedule("j", "r-f64"),
        (4, 6),
        (((4, 5), F64), ((5, 6), F64)),
    ),
    (
        "spmm relayout float64 pack scope",
        lambda: build_spmm(dtype=F64),
        relayout_schedule("k", "r-f64-pack"),
        (4, 6),
        (((4, 5), F64), ((5, 6), F64)),
    ),
    (
        "spmm relayout zero rows",
        build_spmm,
        relayout_schedule("j", "r-zero-rows"),
        (0, 6),
        (((0, 5), F32), ((5, 6), F32)),
    ),
    (
        "spmm relayout zero panel extent",
        build_spmm,
        relayout_schedule("k", "r-zero-cols"),
        (4, 6),
        (((4, 0), F32), ((0, 6), F32)),
    ),
    (
        "spmm relayout zero free extent",
        build_spmm,
        relayout_schedule("j", "r-zero-free"),
        (4, 0),
        (((4, 5), F32), ((5, 0), F32)),
    ),
    (
        "spmm relayout larger shapes pack scope",
        build_spmm,
        relayout_schedule("k", "r-large", width=5, strip=8),
        (7, 16),
        (((7, 9), F32), ((9, 16), F32)),
    ),
]


@pytest.mark.parametrize(
    "case",
    RELAYOUT_PARITY_GRID,
    ids=[case[0] for case in RELAYOUT_PARITY_GRID],
)
def test_relayout_source_is_byte_identical_to_legacy(case):
    name, build, schedule, result_shape, bindings = case
    comparison = compare_generated_sources(
        build(),
        result_shape,
        bindings,
        compile_options=scheduled_options(schedule),
    )
    assert comparison.identical, f"{name} diverged from the legacy schedule"


def test_relayout_artifact_carries_plan_and_provenance():
    from scorch.compiler.loopir.printer import canonical_program_dump

    kernel = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(relayout_schedule("j", "r-prov")),
    )
    scheduled = kernel.schedule
    assert scheduled is not None
    assert scheduled.plan.relayout is not None
    assert scheduled.plan.relayout.strip_width == 4
    parts = [(entry.part, entry.tile is not None) for entry in scheduled.loops]
    assert parts == [
        (LoopPart.OUTER, True),
        (LoopPart.OUTER, True),
        (LoopPart.LOGICAL, False),
        (LoopPart.INNER, True),
        (LoopPart.INNER, True),
    ]
    base_dump = canonical_program_dump(scheduled.base_program)
    scheduled_dump = canonical_program_dump(scheduled.program)
    assert "relayout_stage" not in base_dump and "staged_read" not in base_dump
    assert "relayout_stage" in scheduled_dump
    assert "staged_read" in scheduled_dump
    assert '"scope":"panel"' in scheduled_dump


def test_relayout_structural_activation_is_direct():
    """Every packed-storage component must be present in the generated
    source, asserted directly and never inferred from byte equality."""

    kernel = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(relayout_schedule("j", "r-act")),
    )
    source = kernel.cpp_source
    assert (
        "std::vector<float> packed_B_storage((size_t)kTile_j * (size_t)kTile_k);"
        in source
    )
    assert "float* __restrict__ packed_B = packed_B_storage.data();" in source
    assert "// Pack B j panel into contiguous j-major storage" in source
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads((j_out_end - "
        "j_out) * kTile_k, (j_out_end - j_out))) schedule(static)"
    ) in source
    assert (
        "packed_B[(j_pack - j_out) * kTile_k + k_pack] = "
        "B_val[j_pack * B1_size + k_packed];"
    ) in source
    assert "if (j < j_out || j >= j_out_end) {" in source
    assert "C_values[pC1] += A_val[pA1] * packed_B[(j - j_out) * kTile_k + k_in];" in (
        source
    )
    assert (
        "if (pA1 + 1 < pA1_end && A1_crd[pA1 + 1] >= j_out && "
        "A1_crd[pA1 + 1] < j_out_end) "
        "__builtin_prefetch(&packed_B[(A1_crd[pA1 + 1] - j_out) * kTile_k], 0, 1);"
    ) in source
    assert "B_val[pB1]" not in source

    kernel = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(relayout_schedule("k", "r-act-pack")),
    )
    source = kernel.cpp_source
    assert (
        "std::vector<float> packed_B_storage((size_t)B0_size * (size_t)kTile_k);"
        in source
    )
    assert "// Pack B full j axis into contiguous j-major storage" in source
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(B0_size * kTile_k, "
        "B0_size)) schedule(static)"
    ) in source
    assert "packed_B[j_pack * kTile_k + k_pack] = " in source
    assert "C_values[pC1] += A_val[pA1] * packed_B[j * kTile_k + k_in];" in source
    assert "__builtin_prefetch(&packed_B[A1_crd[pA1 + 1] * kTile_k], 0, 1);" in source
    assert "B_val[pB1]" not in source


@pytest.mark.parametrize("scope", ["j", "k"])
@pytest.mark.parametrize("width", [1, 3, 64])
def test_spmm_relayout_shadow_execution_across_window_regimes(scope, width):
    torch.manual_seed(2723 + width)
    sparse = torch.randn(4, 5)
    sparse[sparse.abs() < 0.6] = 0.0
    sparse[2, :] = 0.0  # an empty CSR row
    dense = torch.randn(5, 6)
    assert_scheduled_shadow(
        build_spmm(),
        relayout_schedule(scope, f"r-shadow-{scope}-{width}", width=width),
        (4, 6),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


def test_spmm_relayout_float64_shadow_execution():
    torch.manual_seed(2724)
    sparse = torch.randn(3, 4, dtype=F64)
    sparse[sparse.abs() < 0.5] = 0.0
    dense = torch.randn(4, 5, dtype=F64)
    assert_scheduled_shadow(
        build_spmm(dtype=F64),
        relayout_schedule("k", "r-shadow-f64", width=2, strip=3),
        (3, 5),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("scope", ["j", "k"])
def test_spmm_relayout_zero_extent_shadow_execution(scope):
    # Zero rows.
    dense = torch.randn(5, 6)
    assert_scheduled_shadow(
        build_spmm(),
        relayout_schedule(scope, f"r-shadow-zr-{scope}"),
        (0, 6),
        (csr_stensor(torch.zeros(0, 5), "A"), dense_stensor(dense, "B")),
        torch.zeros(0, 6),
    )
    # Zero panel extent.
    assert_scheduled_shadow(
        build_spmm(),
        relayout_schedule(scope, f"r-shadow-zp-{scope}"),
        (4, 6),
        (
            csr_stensor(torch.zeros(4, 0), "A"),
            dense_stensor(torch.zeros(0, 6), "B"),
        ),
        torch.zeros(4, 6),
    )
    # Zero free extent.
    sparse = torch.randn(4, 5)
    sparse[sparse.abs() < 0.5] = 0.0
    assert_scheduled_shadow(
        build_spmm(),
        relayout_schedule(scope, f"r-shadow-zf-{scope}"),
        (4, 0),
        (csr_stensor(sparse, "A"), dense_stensor(torch.zeros(5, 0), "B")),
        torch.zeros(4, 0),
    )


def test_relayout_randomized_execution_matches_torch_and_oracle():
    torch.manual_seed(2726)
    import random as _random

    from scorch.compiler.loopir.levels import CsrMatrix
    from scorch.compiler.loopir.oracle import run_program

    rng = _random.Random(20260725)
    for round_index in range(3):
        rows = rng.randrange(1, 7)
        inner = rng.randrange(1, 7)
        cols = rng.randrange(1, 8)
        width = rng.choice((1, 2, 3, 5, 9))
        strip = rng.choice((1, 2, 4, 7))
        scope = rng.choice(("j", "k"))
        sparse = torch.randn(rows, inner)
        sparse[sparse.abs() < 0.5] = 0.0
        dense = torch.randn(inner, cols)
        schedule = relayout_schedule(
            scope, f"r-rand-{round_index}-{scope}", width=width, strip=strip
        )
        result, kernel = execute_cin_via_loopir(
            build_spmm(),
            (rows, cols),
            csr_stensor(sparse, "A"),
            dense_stensor(dense, "B"),
            compile_options=scheduled_options(schedule),
        )
        assert torch.allclose(
            result.values.reshape(rows, cols),
            sparse @ dense,
            atol=1e-3,
            rtol=1e-3,
        )
        assert kernel.schedule is not None
        lowering = kernel.lowering
        oracle_out = run_program(
            kernel.schedule.program,
            {
                lowering.input_symbols[0]: CsrMatrix.from_dense(sparse.tolist()),
                lowering.input_symbols[1]: dense.tolist(),
            },
            {lowering.result_symbol: (rows, cols)},
        )[lowering.result_symbol]
        assert torch.allclose(
            result.values.reshape(rows, cols),
            torch.tensor(oracle_out, dtype=torch.float32),
            atol=1e-4,
            rtol=1e-4,
        )


def test_relayout_heap_composition_is_migrated():
    """The heap-accumulation relayout composition now routes through the
    typed heap family end to end (the boundary the earlier fail-closed
    lock guarded has moved; the heap parity grid below owns its bytes)."""

    kernel = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(
            relayout_schedule("k", "r-heap", accum="heap")
        ),
    )
    scheduled = kernel.schedule
    assert scheduled is not None
    assert scheduled.plan.result_tile is not None
    assert scheduled.plan.relayout is not None
    assert "tiled_C" in kernel.cpp_source
    assert "packed_B" in kernel.cpp_source


# -- the heap result-tile slice -----------------------------------------------


def heap_schedule(tag, strip=3):
    return Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec("k", strip, placement="outermost", accum="heap", unroll=False),
        ),
        tag=tag,
        parallel_loop="i",
    )


def dense_heap_schedule(tag, strip=3):
    return Schedule(
        loop_order=("i", "k", "j"),
        tiles=(
            TileSpec("j", strip, placement="outermost", accum="heap", unroll=False),
        ),
        tag=tag,
        parallel_loop="i",
    )


def heap_relayout_schedule(scope, tag, width=3, strip=4):
    from scorch.compiler.scheduler import RelayoutSpec

    return Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec("k", strip, placement="outermost", accum="heap", unroll=False),
            panel("j", width, placement="child_of:k_out"),
        ),
        relayout=RelayoutSpec("B", "k", strip, scope_var=scope),
        tag=tag,
        parallel_loop="i",
    )


def heap_panel_schedule(tag, width=3, strip=4):
    return Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec("k", strip, placement="outermost", accum="heap", unroll=False),
            panel("j", width, placement="child_of:k_out"),
        ),
        tag=tag,
        parallel_loop="i",
    )


def build_ttm(dtype=F32):
    """``Projected[a, b, d] += Core[a, b, c] * Factor[c, d]``.

    The multi-prefix heap representative: a rank-3 all-dense result whose
    compact tile linearizes two dense prefix axes.
    """

    ivs = {name: IndexVar(name) for name in ("a", "b", "c", "d")}
    out = TensorVar("Projected", fmt="ddd", dtype=dtype)
    core = TensorVar("Core", fmt="dds", dtype=dtype)
    factor = TensorVar("Factor", fmt="dd", dtype=dtype)
    stmt = TensorAssign(
        out[ivs["a"], ivs["b"], ivs["d"]],
        CINBinaryOp(
            Operation.MUL,
            core[ivs["a"], ivs["b"], ivs["c"]],
            factor[ivs["c"], ivs["d"]],
        ),
        op=Operation.ADD,
    )
    for name in reversed(("a", "b", "c", "d")):
        stmt = ForAll(ivs[name], stmt)
    return stmt


def ttm_heap_schedule(tag, strip=3):
    return Schedule(
        loop_order=("a", "b", "c", "d"),
        tiles=(
            TileSpec("d", strip, placement="outermost", accum="heap", unroll=False),
        ),
        tag=tag,
        parallel_loop="a",
    )


TTM_BINDINGS = (((3, 4, 5), F32), ((5, 6), F32))
TTM_BINDINGS_F64 = (((3, 4, 5), F64), ((5, 6), F64))

HEAP_PARITY_GRID = [
    (
        "dense matmul heap",
        build_matmul,
        dense_heap_schedule("h-dense", strip=4),
        (4, 6),
        MATMUL_BINDINGS,
    ),
    (
        "ttm multi-prefix heap exact strip",
        build_ttm,
        ttm_heap_schedule("h-ttm-exact", strip=3),
        (3, 4, 6),
        TTM_BINDINGS,
    ),
    (
        "ttm multi-prefix heap ragged strip",
        build_ttm,
        ttm_heap_schedule("h-ttm-ragged", strip=4),
        (3, 4, 6),
        TTM_BINDINGS,
    ),
    (
        "ttm multi-prefix heap unit strip",
        build_ttm,
        ttm_heap_schedule("h-ttm-unit", strip=1),
        (3, 4, 6),
        TTM_BINDINGS,
    ),
    (
        "ttm multi-prefix heap oversized strip",
        build_ttm,
        ttm_heap_schedule("h-ttm-over", strip=64),
        (3, 4, 6),
        TTM_BINDINGS,
    ),
    (
        "ttm multi-prefix heap f64",
        lambda: build_ttm(dtype=F64),
        ttm_heap_schedule("h-ttm-f64", strip=4),
        (3, 4, 6),
        TTM_BINDINGS_F64,
    ),
    (
        "ttm multi-prefix heap zero outer prefix",
        build_ttm,
        ttm_heap_schedule("h-ttm-zero-outer", strip=2),
        (0, 4, 6),
        (((0, 4, 5), F32), ((5, 6), F32)),
    ),
    (
        "ttm multi-prefix heap zero inner prefix",
        build_ttm,
        ttm_heap_schedule("h-ttm-zero-inner", strip=2),
        (3, 0, 6),
        (((3, 0, 5), F32), ((5, 6), F32)),
    ),
    (
        "ttm multi-prefix heap zero reduction extent",
        build_ttm,
        ttm_heap_schedule("h-ttm-zero-red", strip=2),
        (3, 4, 6),
        (((3, 4, 0), F32), ((0, 6), F32)),
    ),
    (
        "ttm multi-prefix heap zero free extent",
        build_ttm,
        ttm_heap_schedule("h-ttm-zero-free", strip=3),
        (3, 4, 0),
        (((3, 4, 5), F32), ((5, 0), F32)),
    ),
    (
        "spmm heap exact strip",
        build_spmm,
        heap_schedule("h-exact", strip=3),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm heap ragged strip",
        build_spmm,
        heap_schedule("h-ragged", strip=4),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm heap unit strip",
        build_spmm,
        heap_schedule("h-unit", strip=1),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm heap oversized strip",
        build_spmm,
        heap_schedule("h-wide", strip=64),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm heap float64",
        lambda: build_spmm(dtype=F64),
        heap_schedule("h-f64", strip=4),
        (4, 6),
        (((4, 5), F64), ((5, 6), F64)),
    ),
    (
        "spmm heap panel no relayout",
        build_spmm,
        heap_panel_schedule("h-panel"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm heap relayout panel scope",
        build_spmm,
        heap_relayout_schedule("j", "h-r-panel"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm heap relayout pack scope",
        build_spmm,
        heap_relayout_schedule("k", "h-r-pack"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm heap relayout float64 panel scope",
        lambda: build_spmm(dtype=F64),
        heap_relayout_schedule("j", "h-r-f64"),
        (4, 6),
        (((4, 5), F64), ((5, 6), F64)),
    ),
    (
        "spmm heap relayout float64 pack scope",
        lambda: build_spmm(dtype=F64),
        heap_relayout_schedule("k", "h-r-f64-pack"),
        (4, 6),
        (((4, 5), F64), ((5, 6), F64)),
    ),
    (
        "spmm heap zero rows",
        build_spmm,
        heap_schedule("h-zero-rows", strip=2),
        (0, 6),
        (((0, 5), F32), ((5, 6), F32)),
    ),
    (
        "spmm heap zero reduction extent",
        build_spmm,
        heap_schedule("h-zero-red", strip=2),
        (4, 6),
        (((4, 0), F32), ((0, 6), F32)),
    ),
    (
        "spmm heap zero free extent",
        build_spmm,
        heap_schedule("h-zero-free", strip=3),
        (4, 0),
        (((4, 5), F32), ((5, 0), F32)),
    ),
    (
        "spmm heap relayout larger shapes",
        build_spmm,
        heap_relayout_schedule("k", "h-r-large", width=5, strip=8),
        (7, 16),
        (((7, 9), F32), ((9, 16), F32)),
    ),
]


@pytest.mark.parametrize(
    "case",
    HEAP_PARITY_GRID,
    ids=[case[0] for case in HEAP_PARITY_GRID],
)
def test_heap_source_is_byte_identical_to_legacy(case):
    name, build, schedule, result_shape, bindings = case
    comparison = compare_generated_sources(
        build(),
        result_shape,
        bindings,
        compile_options=scheduled_options(schedule),
    )
    assert comparison.identical, f"{name} diverged from the legacy schedule"


def test_heap_artifact_carries_plan_and_provenance():
    from scorch.compiler.loopir.printer import canonical_program_dump

    kernel = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(heap_schedule("h-prov")),
    )
    scheduled = kernel.schedule
    assert scheduled is not None
    assert scheduled.plan.result_tile is not None
    assert scheduled.plan.result_tile.result_level == 1
    parts = [(entry.part, entry.tile is not None) for entry in scheduled.loops]
    assert parts == [
        (LoopPart.OUTER, True),
        (LoopPart.LOGICAL, False),
        (LoopPart.LOGICAL, False),
        (LoopPart.INNER, True),
    ]
    base_dump = canonical_program_dump(scheduled.base_program)
    scheduled_dump = canonical_program_dump(scheduled.program)
    assert "result_tile_region" not in base_dump
    assert "tiled_reduce" not in base_dump
    assert "result_tile_region" in scheduled_dump
    assert "tiled_reduce" in scheduled_dump


def test_heap_structural_activation_is_direct():
    """Every compact-tile component must be present in the generated
    source, asserted directly and never inferred from byte equality."""

    kernel = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(heap_schedule("h-act", strip=3)),
    )
    source = kernel.cpp_source
    assert (
        "std::vector<float> tiled_C_storage((size_t)C0_size * (size_t)kTile_k);"
        in source
    )
    assert "float* __restrict__ tiled_C = tiled_C_storage.data();" in source
    assert "// Initialize compact result tile for C" in source
    assert "// Copy compact result tile to C" in source
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads((C0_size) * "
        "kTile_k, (C0_size))) schedule(static)"
    ) in source
    assert "tiled_C[C_tile_init * kTile_k + k_tile_init] = 0.0f;" in source
    assert (
        "C_values[C_tile_copy * C1_size + k_copy_logical] = "
        "tiled_C[C_tile_copy * kTile_k + k_tile_copy];"
    ) in source
    assert "tiled_C[pC0 * kTile_k + k_in] += A_val[pA1] * B_val[pB1];" in source
    # The exactly-once copy-out discharges the whole-result zero fill.
    assert "scorch_zero_dense(C_values" not in source
    # The parallel row keeps the legacy explicit-parallel policy.
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(A1_pos[A0_size], "
        "A0_size)) schedule(dynamic, scorch_chunk(A0_size, A1_pos[A0_size]))"
    ) in source
    # The dead position resolve stays in place, exactly as legacy leaves it.
    assert "int pC1 = pC0 * C1_size + k;" in source

    kernel = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=scheduled_options(heap_relayout_schedule("k", "h-act-pack")),
    )
    source = kernel.cpp_source
    assert (
        "std::vector<float> tiled_C_storage((size_t)C0_size * (size_t)kTile_k);"
        in source
    )
    assert (
        "std::vector<float> packed_B_storage((size_t)B0_size * (size_t)kTile_k);"
        in source
    )
    assert "tiled_C[pC0 * kTile_k + k_in] += A_val[pA1] * packed_B[j * kTile_k" in (
        source
    )
    assert "scorch_zero_dense(C_values" not in source
    assert "B_val[pB1]" not in source


@pytest.mark.parametrize("width", [1, 3, 64])
def test_spmm_heap_shadow_execution_across_strip_regimes(width):
    torch.manual_seed(2911 + width)
    sparse = torch.randn(4, 5)
    sparse[sparse.abs() < 0.6] = 0.0
    sparse[2, :] = 0.0  # an empty CSR row
    dense = torch.randn(5, 6)
    assert_scheduled_shadow(
        build_spmm(),
        heap_schedule(f"h-shadow-{width}", strip=width),
        (4, 6),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


def test_dense_matmul_heap_shadow_execution():
    """Dense prefixes use the same legacy-derived parallel policy contract."""

    torch.manual_seed(2913)
    a = torch.randn(4, 5)
    b = torch.randn(5, 6)
    assert_scheduled_shadow(
        build_matmul(),
        dense_heap_schedule("h-dense-shadow", strip=4),
        (4, 6),
        (dense_stensor(a, "A"), dense_stensor(b, "B")),
        a @ b,
    )


def ttm_stensor(tensor, name):
    """Bind a ``dds`` operand: two dense levels over one compressed leaf.

    Built directly rather than through ``STensor.to_sparse('dds')``, whose
    rank-3 filter-zeros kernel emits a leaf position array sized for the
    innermost parent only (a pre-existing public-conversion defect outside
    this milestone; see the Phase-6 review's limitations).
    """

    from scorch.storage import TensorIndex, TensorStorage

    outer, inner, reduction = tensor.shape
    flat = tensor.reshape(outer * inner, reduction)
    stored = flat != 0
    positions = torch.zeros(outer * inner + 1, dtype=torch.int32)
    positions[1:] = torch.cumsum(stored.sum(1), 0).to(torch.int32)
    coordinates = torch.nonzero(stored)[:, 1].to(torch.int32)
    index = TensorIndex("dds", [[], [], [positions, coordinates]])
    storage = TensorStorage(
        index=index,
        value=flat[stored].contiguous(),
        shape=(outer, inner, reduction),
    )
    return STensor(name=name, storage=storage)


def _assert_ttm_multi_prefix_heap_shadow(
    *,
    width,
    batch,
    rows,
    inner,
    cols,
    dtype_name,
    seed,
):
    """Run one native rank-3 differential in an isolated process.

    The helper is module-level so the pytest parent can execute each new
    JIT-heavy case in a short-lived child.  Scorch's pre-existing macOS
    full-suite process already approaches libomp's pthread-key ceiling; keeping
    these additional extensions out of that long-lived process makes the
    canonical, unpartitioned suite a meaningful gate instead of moving the
    failure threshold.
    """

    dtype = {"float32": F32, "float64": F64}[dtype_name]
    torch.manual_seed(seed)
    core = torch.randn(batch, rows, inner, dtype=dtype)
    if core.numel():
        core[core.abs() < 0.6] = 0.0
    if batch > 1 and rows > 2:
        core[1, 2, :] = 0.0
    factor = torch.randn(inner, cols, dtype=dtype)
    tolerance = 1e-9 if dtype is F64 else 1e-3
    assert_scheduled_shadow(
        build_ttm(dtype=dtype),
        ttm_heap_schedule(f"h-ttm-shadow-{width}", strip=width),
        (batch, rows, cols),
        (ttm_stensor(core, "Core"), dense_stensor(factor, "Factor")),
        torch.einsum("abc,cd->abd", core, factor),
        atol=tolerance,
        rtol=tolerance,
    )


def _run_ttm_multi_prefix_heap_shadow_in_subprocess(**kwargs):
    """Keep each added native rank-3 case below macOS's process-local limit."""

    invocation = (
        "from tests.test_scorch.test_loopir_scheduled_slice import "
        "_assert_ttm_multi_prefix_heap_shadow as run\n"
        f"run(**{kwargs!r})\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", invocation],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "isolated rank-3 heap differential failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.parametrize("width", [1, 3, 4, 64])
def test_ttm_multi_prefix_heap_shadow_execution(width):
    """The rank-3 compact tile agrees bitwise with legacy and with Torch."""

    _run_ttm_multi_prefix_heap_shadow_in_subprocess(
        width=width,
        batch=3,
        rows=4,
        inner=5,
        cols=6,
        dtype_name="float32",
        seed=2941 + width,
    )


def test_ttm_multi_prefix_heap_float64_shadow_execution():
    _run_ttm_multi_prefix_heap_shadow_in_subprocess(
        width=4,
        batch=3,
        rows=4,
        inner=5,
        cols=6,
        dtype_name="float64",
        seed=2949,
    )


@pytest.mark.parametrize(
    ("batch", "rows", "inner", "cols"),
    [
        (0, 4, 5, 6),
        (3, 0, 5, 6),
        (3, 4, 0, 6),
        (3, 4, 5, 0),
    ],
    ids=("zero-outer-prefix", "zero-inner-prefix", "zero-reduction", "zero-free"),
)
def test_ttm_multi_prefix_heap_zero_extent_shadow_execution(batch, rows, inner, cols):
    """Every rank-3 zero-extent position executes through both native routes."""

    _run_ttm_multi_prefix_heap_shadow_in_subprocess(
        width=3,
        batch=batch,
        rows=rows,
        inner=inner,
        cols=cols,
        dtype_name="float32",
        seed=2951 + batch + rows + inner + cols,
    )


def test_ttm_multi_prefix_heap_structural_activation_is_direct():
    """Every rank-3 compact-tile component is asserted directly."""

    kernel = compile_cin_via_loopir(
        build_ttm(),
        (3, 4, 6),
        TTM_BINDINGS,
        compile_options=scheduled_options(ttm_heap_schedule("h-ttm-act", strip=3)),
    )
    source = kernel.cpp_source
    # The compact extent is the product of *both* dense prefix levels.
    assert (
        "std::vector<float> tiled_Projected_storage("
        "(size_t)(Projected0_size * Projected1_size) * (size_t)kTile_d);" in source
    )
    assert (
        "float* __restrict__ tiled_Projected = tiled_Projected_storage.data();"
        in source
    )
    assert "// Initialize compact result tile for Projected" in source
    assert "// Copy compact result tile to Projected" in source
    # The compact row is the linearized prefix position of the *last*
    # prefix level, not a rank-2 spelling.
    assert (
        "tiled_Projected[pProjected1 * kTile_d + d_in] += "
        "Core_val[pCore2] * Factor_val[pFactor1];" in source
    )
    assert (
        "Projected_values[Projected_tile_copy * Projected2_size + d_copy_logical] = "
        "tiled_Projected[Projected_tile_copy * kTile_d + d_tile_copy];" in source
    )
    assert (
        "for (int64_t Projected_tile_init = 0; "
        "Projected_tile_init < Projected0_size * Projected1_size; "
        "Projected_tile_init++) {" in source
    )
    # The exactly-once copy-out discharges the whole-result zero fill.
    assert "scorch_zero_dense(Projected_values" not in source
    # The parallel policy lands on the outermost dense prefix loop.
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(-1, Core0_size)) "
        "schedule(dynamic, scorch_chunk(Core0_size, -1))" in source
    )
    scheduled = kernel.schedule
    assert scheduled is not None
    assert scheduled.plan.result_tile is not None
    assert scheduled.plan.result_tile.result_level == 2
    assert len(scheduled.plan.result_tile.result_prefix) == 2


@pytest.mark.parametrize("scope", ["j", "k"])
def test_spmm_heap_relayout_shadow_execution(scope):
    torch.manual_seed(2917)
    sparse = torch.randn(4, 5)
    sparse[sparse.abs() < 0.6] = 0.0
    sparse[2, :] = 0.0  # an empty CSR row
    dense = torch.randn(5, 6)
    assert_scheduled_shadow(
        build_spmm(),
        heap_relayout_schedule(scope, f"h-r-shadow-{scope}", width=2, strip=4),
        (4, 6),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


def test_spmm_heap_float64_shadow_execution():
    torch.manual_seed(2919)
    sparse = torch.randn(4, 5, dtype=torch.float64)
    sparse[sparse.abs() < 0.6] = 0.0
    dense = torch.randn(5, 6, dtype=torch.float64)
    assert_scheduled_shadow(
        build_spmm(dtype=F64),
        heap_schedule("h-shadow-f64", strip=4),
        (4, 6),
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    ("result_shape", "bindings", "sparse_shape", "dense_shape"),
    [
        ((0, 6), (((0, 5), F32), ((5, 6), F32)), (0, 5), (5, 6)),
        ((4, 6), (((4, 0), F32), ((0, 6), F32)), (4, 0), (0, 6)),
        ((4, 0), (((4, 5), F32), ((5, 0), F32)), (4, 5), (5, 0)),
    ],
    ids=["zero-rows", "zero-reduction", "zero-free"],
)
def test_spmm_heap_zero_extent_shadow_execution(
    result_shape, bindings, sparse_shape, dense_shape
):
    torch.manual_seed(2921)
    sparse = torch.randn(*sparse_shape)
    if sparse.numel():
        sparse[sparse.abs() < 0.5] = 0.0
    dense = torch.randn(*dense_shape)
    assert_scheduled_shadow(
        build_spmm(),
        heap_schedule(f"h-zero-{result_shape}", strip=2),
        result_shape,
        (csr_stensor(sparse, "A"), dense_stensor(dense, "B")),
        sparse @ dense,
    )


def test_heap_randomized_execution_matches_torch_and_oracle():
    torch.manual_seed(2929)
    import random as _random

    from scorch.compiler.loopir.levels import CsrMatrix
    from scorch.compiler.loopir.oracle import run_program

    rng = _random.Random(20260726)
    for round_index in range(3):
        rows = rng.randrange(1, 7)
        inner = rng.randrange(1, 7)
        cols = rng.randrange(1, 8)
        strip = rng.choice((1, 2, 4, 7))
        composed = rng.random() < 0.5
        sparse = torch.randn(rows, inner)
        sparse[sparse.abs() < 0.5] = 0.0
        dense = torch.randn(inner, cols)
        if composed:
            schedule = heap_relayout_schedule(
                rng.choice(("j", "k")),
                f"h-rand-{round_index}",
                width=rng.choice((1, 2, 3, 9)),
                strip=strip,
            )
        else:
            schedule = heap_schedule(f"h-rand-{round_index}", strip=strip)
        result, kernel = execute_cin_via_loopir(
            build_spmm(),
            (rows, cols),
            csr_stensor(sparse, "A"),
            dense_stensor(dense, "B"),
            compile_options=scheduled_options(schedule),
        )
        assert torch.allclose(
            result.values.reshape(rows, cols),
            sparse @ dense,
            atol=1e-3,
            rtol=1e-3,
        )
        assert kernel.schedule is not None
        lowering = kernel.lowering
        oracle_out = run_program(
            kernel.schedule.program,
            {
                lowering.input_symbols[0]: CsrMatrix.from_dense(sparse.tolist()),
                lowering.input_symbols[1]: dense.tolist(),
            },
            {lowering.result_symbol: (rows, cols)},
        )[lowering.result_symbol]
        assert torch.allclose(
            result.values.reshape(rows, cols),
            torch.tensor(oracle_out, dtype=torch.float32),
            atol=1e-4,
            rtol=1e-4,
        )


# -- Explicit parallel anchors ------------------------------------------------


def ttm_heap_anchor_schedule(tag, anchor, strip=3, dtype_tag=""):
    return Schedule(
        loop_order=("a", "b", "c", "d"),
        tiles=(
            TileSpec("d", strip, placement="outermost", accum="heap", unroll=False),
        ),
        tag=tag,
        parallel_loop=anchor,
    )


def ttm_heap_panel_schedule(tag, anchor, width=2, strip=3):
    return Schedule(
        loop_order=("a", "b", "c", "d"),
        tiles=(
            TileSpec("d", strip, placement="outermost", accum="heap", unroll=False),
            panel("c", width, placement="child_of:d_out", unroll=False),
        ),
        tag=tag,
        parallel_loop=anchor,
    )


ANCHOR_PARITY_GRID = [
    (
        "matmul anchor i matches the auto row",
        build_matmul,
        Schedule(loop_order=("i", "k", "j"), parallel_loop="i", tag="anch-mm-i"),
        (4, 6),
        MATMUL_BINDINGS,
    ),
    (
        "matmul anchor j marks the inner free loop",
        build_matmul,
        Schedule(loop_order=("i", "k", "j"), parallel_loop="j", tag="anch-mm-j"),
        (4, 6),
        MATMUL_BINDINGS,
    ),
    (
        "spmm anchor i keeps the nnz-aware row policy",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), parallel_loop="i", tag="anch-sp-i"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "spmm anchor k marks the dense free loop",
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), parallel_loop="k", tag="anch-sp-k"),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "stack anchor i suppresses the origin auto gate",
        build_spmm,
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(stack("k", 4),),
            parallel_loop="i",
            tag="anch-st-i",
        ),
        (4, 6),
        SPMM_BINDINGS,
    ),
    (
        "tile-j anchor i marks the row over the origin",
        build_matmul,
        Schedule(
            loop_order=("i", "k", "j"),
            tiles=(tile("j", 4),),
            parallel_loop="i",
            tag="anch-tj-i",
        ),
        (4, 6),
        MATMUL_BINDINGS,
    ),
    (
        "tile-j anchor j redirects to the split origin",
        build_matmul,
        Schedule(
            loop_order=("i", "k", "j"),
            tiles=(tile("j", 4),),
            parallel_loop="j",
            tag="anch-tj-j",
        ),
        (4, 6),
        MATMUL_BINDINGS,
    ),
    (
        "ttm heap outer prefix anchor",
        build_ttm,
        ttm_heap_anchor_schedule("anch-ttm-a", "a"),
        (3, 4, 6),
        TTM_BINDINGS,
    ),
    (
        "ttm heap inner prefix anchor",
        build_ttm,
        ttm_heap_anchor_schedule("anch-ttm-b", "b"),
        (3, 4, 6),
        TTM_BINDINGS,
    ),
    (
        "ttm heap inner prefix anchor f64",
        lambda: build_ttm(dtype=F64),
        ttm_heap_anchor_schedule("anch-ttm-b64", "b"),
        (3, 4, 6),
        TTM_BINDINGS_F64,
    ),
    (
        "ttm heap panel dense-parent anchor",
        build_ttm,
        ttm_heap_panel_schedule("anch-ttm-pb", "b"),
        (3, 4, 6),
        TTM_BINDINGS,
    ),
]


@pytest.mark.parametrize(
    "case",
    ANCHOR_PARITY_GRID,
    ids=[case[0] for case in ANCHOR_PARITY_GRID],
)
def test_anchor_source_is_byte_identical_to_legacy(case):
    """Every admitted explicit anchor reproduces the legacy route's bytes."""

    name, build, schedule, result_shape, bindings = case
    comparison = compare_generated_sources(
        build(),
        result_shape,
        bindings,
        compile_options=scheduled_options(schedule),
    )
    assert comparison.identical, f"{name} diverged from the legacy anchor"


def test_merged_outer_anchor_keeps_the_legacy_row_only_policy():
    """Merged iterator state is deliberately invisible to work discovery."""

    comparison = compare_generated_sources(
        build_union_add_to_dense(),
        (4, 5),
        (((4, 5), F32), ((4, 5), F32)),
        compile_options=scheduled_options(
            Schedule(
                loop_order=("i", "j"),
                parallel_loop="i",
                tag="anch-merged-row",
            )
        ),
    )
    assert comparison.identical
    expected = (
        "num_threads(scorch_nthreads(-1, A0_size)) "
        "schedule(dynamic, scorch_chunk(A0_size, -1))"
    )
    assert expected in comparison.legacy_cpp
    assert expected in comparison.loopir_cpp


def test_sparse_work_requires_the_exact_legacy_dense_bound_driver():
    """An equal logical extent cannot substitute a different bound spelling."""

    comparison = compare_generated_sources(
        build_dense_driver_before_csr(),
        (4, 5, 6),
        (((4, 5), F32), ((4, 6), F32)),
        compile_options=scheduled_options(
            Schedule(
                loop_order=("i", "k", "j"),
                parallel_loop="i",
                tag="anch-dense-driver",
            )
        ),
    )
    assert comparison.identical
    expected = (
        "num_threads(scorch_nthreads(-1, X0_size)) "
        "schedule(dynamic, scorch_chunk(X0_size, -1))"
    )
    assert expected in comparison.legacy_cpp
    assert expected in comparison.loopir_cpp
    pragma = next(
        line for line in comparison.loopir_cpp.splitlines() if "scorch_nthreads" in line
    )
    assert "A1_pos" not in pragma


def test_ttm_heap_inner_anchor_structural_activation():
    """The lifted inner-prefix anchor moves the nnz-aware policy to ``b``."""

    comparison = compare_generated_sources(
        build_ttm(),
        (3, 4, 6),
        TTM_BINDINGS,
        compile_options=scheduled_options(
            ttm_heap_anchor_schedule("anch-ttm-act", "b")
        ),
    )
    source = comparison.loopir_cpp
    assert "num_threads(scorch_nthreads(Core2_pos[Core1_size], Core1_size))" in source
    assert "schedule(dynamic, scorch_chunk(Core1_size, Core2_pos[Core1_size]))" in (
        source
    )
    # The strip init/copy groups keep their anchor-independent static policy.
    assert (
        source.count(
            "scorch_nthreads((Projected0_size * Projected1_size) * kTile_d, "
            "(Projected0_size * Projected1_size))"
        )
        == 2
    )
    outer = compare_generated_sources(
        build_ttm(),
        (3, 4, 6),
        TTM_BINDINGS,
        compile_options=scheduled_options(
            ttm_heap_anchor_schedule("anch-ttm-act-a", "a")
        ),
    ).loopir_cpp
    assert "num_threads(scorch_nthreads(-1, Core0_size))" in outer
    assert outer != source


def _assert_explicit_anchor_shadow(*, kind, seed):
    """Run one native explicit-anchor differential in an isolated process."""

    torch.manual_seed(seed)
    if kind in ("ttm_heap_b", "ttm_heap_panel_b"):
        core = torch.randn(3, 4, 5, dtype=F32)
        core[core.abs() < 0.6] = 0.0
        core[1, 2, :] = 0.0
        factor = torch.randn(5, 6, dtype=F32)
        schedule = (
            ttm_heap_anchor_schedule("anch-sh-ttm-b", "b")
            if kind == "ttm_heap_b"
            else ttm_heap_panel_schedule("anch-sh-ttm-pb", "b")
        )
        assert_scheduled_shadow(
            build_ttm(),
            schedule,
            (3, 4, 6),
            (ttm_stensor(core, "Core"), dense_stensor(factor, "Factor")),
            torch.einsum("abc,cd->abd", core, factor),
        )
        return
    a_dense = torch.randn(4, 5, dtype=F32)
    a_dense[a_dense.abs() < 0.5] = 0.0
    a_dense[2, :] = 0.0
    b_dense = torch.randn(5, 6, dtype=F32)
    if kind == "spmm_k":
        schedule = Schedule(
            loop_order=("i", "j", "k"), parallel_loop="k", tag="anch-sh-sp-k"
        )
    else:
        assert kind == "stack_i"
        schedule = Schedule(
            loop_order=("i", "j", "k"),
            tiles=(stack("k", 4),),
            parallel_loop="i",
            tag="anch-sh-st-i",
        )
    assert_scheduled_shadow(
        build_spmm(),
        schedule,
        (4, 6),
        (csr_stensor(a_dense, "A"), dense_stensor(b_dense, "B")),
        a_dense @ b_dense,
    )


def _run_explicit_anchor_shadow_in_subprocess(**kwargs):
    """Keep each added native anchor case out of the long-lived parent."""

    invocation = (
        "from tests.test_scorch.test_loopir_scheduled_slice import "
        "_assert_explicit_anchor_shadow as run\n"
        f"run(**{kwargs!r})\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", invocation],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "isolated explicit-anchor differential failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


@pytest.mark.parametrize(
    "kind",
    ["ttm_heap_b", "ttm_heap_panel_b", "spmm_k", "stack_i"],
)
def test_explicit_anchor_shadow_execution(kind):
    """Each lifted or newly admitted anchor executes bitwise-equal to legacy."""

    _run_explicit_anchor_shadow_in_subprocess(kind=kind, seed=2026)
