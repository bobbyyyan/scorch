"""Behavioral coverage for compilation-local compiler-stage timing.

The compiler-stage records in this file stop at the frozen native build
request.  Cache lookup, native compilation/loading, and kernel execution are
deliberately outside the seam.  Managed LLIR pass records remain a separate,
nested observation owned by the same ``CompilationContext``.
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace
from typing import Callable, Optional, Tuple, cast

import pytest
import torch

import scorch  # type: ignore[import-untyped]
import scorch.compiler.cin_analysis as cin_analysis  # type: ignore[import-untyped]
import scorch.compiler.cin_lowerer as cin_lowerer_module  # type: ignore[import-untyped]
import scorch.compiler.compilation_context as context_module  # type: ignore[import-untyped]
import scorch.compiler.llir_pass_manager as llir_pass_manager  # type: ignore[import-untyped]
import scorch.compiler.schedule_lowerer as schedule_lowerer  # type: ignore[import-untyped]
import scorch.compiler.scheduler as scheduler_module  # type: ignore[import-untyped]
import scorch.ops as ops  # type: ignore[import-untyped]
import scorch.stensor as stensor_module  # type: ignore[import-untyped]
import scorch.utils as utils  # type: ignore[import-untyped]
from scorch.compiler import llir  # type: ignore[import-untyped]
from scorch.compiler import (  # type: ignore[import-untyped]
    compressed_where_openmp_pass as compressed_where_module,
)
from scorch.compiler.cin import (  # type: ignore[import-untyped]
    ForAll,
    IndexStmt,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
)
from scorch.compiler.cin_analysis import (  # type: ignore[import-untyped]
    canonical_cin_dump,
    normalize_cin,
)
from scorch.compiler.cin_lowerer import CINLowerer  # type: ignore[import-untyped]
from scorch.compiler.codegen import LLIRLowerer  # type: ignore[import-untyped]
from scorch.compiler.iterator import ModeIterator  # type: ignore[import-untyped]
from scorch.compiler.compilation_context import (  # type: ignore[import-untyped]
    CANONICAL_COMPILER_STAGES,
    CompilationContext,
    CompilationContextError,
    CompilerStageId,
    CompilerStageRunRecord,
    CompilerStageToken,
)
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
    canonical_cache_digest,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    InvalidSchedule,
    VerificationError,
)
from scorch.compiler.loop_plan import (  # type: ignore[import-untyped]
    LoopRef,
    ScheduledCIN,
    verify_scheduled_cin,
)
from scorch.compiler.llir_pass_manager import (  # type: ignore[import-untyped]
    DEBUG_LLIR_PASS_OPTIONS,
    LLIRPassOptions,
    PRODUCTION_LLIR_PASS_OPTIONS,
)
from scorch.compiler.llir_traversal import (  # type: ignore[import-untyped]
    LLIRTraversalContext,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
)
from scorch.compiler.torch_cpp_abi import (  # type: ignore[import-untyped]
    ResultTensorAssembler,
    TorchCppKernelABI,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    RelayoutSpec,
    Schedule,
    Scheduler,
    TileSpec,
)
from scorch.format import parse_format  # type: ignore[import-untyped]
from scorch.layout import TensorSpec  # type: ignore[import-untyped]
from scorch.stensor import STensor  # type: ignore[import-untyped]

_FULL_STAGE_SEQUENCE = [stage.value for stage in CANONICAL_COMPILER_STAGES]
_MANUAL_STAGE_SEQUENCE = [
    CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value,
    CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value,
    CompilerStageId.LEGACY_CIN_ADAPTATION.value,
    CompilerStageId.CIN_LOWERING.value,
    CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    CompilerStageId.LLIR_TO_CPP_GENERATION.value,
    CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY.value,
]
_AUTO_STAGE_SEQUENCE = [
    CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value,
    CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION.value,
    CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value,
    CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION.value,
    CompilerStageId.LEGACY_CIN_ADAPTATION.value,
    CompilerStageId.CIN_LOWERING.value,
    CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    CompilerStageId.LLIR_TO_CPP_GENERATION.value,
    CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY.value,
]
_EXPLICIT_STAGE_SEQUENCE = [
    CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value,
    CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION.value,
    CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value,
    CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION.value,
    CompilerStageId.LEGACY_CIN_ADAPTATION.value,
    CompilerStageId.CIN_LOWERING.value,
    CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    CompilerStageId.SCHEDULE_LOWERING.value,
    CompilerStageId.LLIR_TO_CPP_GENERATION.value,
    CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY.value,
]
_EINSUM_PREFIX_THROUGH_ADAPTER = _EXPLICIT_STAGE_SEQUENCE[:5]
_DIRECT_CIN_STAGE_SEQUENCE = [
    CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value,
    CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value,
    CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value,
    CompilerStageId.LEGACY_CIN_ADAPTATION.value,
    CompilerStageId.CIN_LOWERING.value,
    CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    CompilerStageId.LLIR_TO_CPP_GENERATION.value,
    CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY.value,
]


def _default_options(
    *,
    verify_cin: bool = False,
    llir_pass_options: LLIRPassOptions = PRODUCTION_LLIR_PASS_OPTIONS,
    requested_schedule: Optional[Schedule] = None,
    regblock_dual: bool = False,
) -> CompileOptions:
    environ = {} if regblock_dual else {"SCORCH_REGBLOCK_DUAL": "0"}
    return CompileOptions.from_environment(
        environ=environ,
        requested_schedule=requested_schedule,
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=verify_cin,
        llir_pass_options=llir_pass_options,
    )


def _explicit_options(
    *,
    verify_cin: bool = False,
    llir_pass_options: LLIRPassOptions = PRODUCTION_LLIR_PASS_OPTIONS,
) -> CompileOptions:
    return _default_options(
        verify_cin=verify_cin,
        llir_pass_options=llir_pass_options,
        requested_schedule=Schedule(loop_order=("i", "k", "j")),
    )


def _spmm_specs() -> Tuple[TensorSpec, TensorSpec]:
    return (
        TensorSpec("ds", (2, 3), name="A"),
        TensorSpec("dd", (3, 4), name="B"),
    )


def _all_coo_sddmm_specs() -> Tuple[TensorSpec, TensorSpec, TensorSpec]:
    return (
        TensorSpec("oo", (2, 3), name="Mask"),
        TensorSpec("dd", (2, 4), name="Query"),
        TensorSpec("dd", (3, 4), name="Key"),
    )


def _build_spmm_cin() -> ForAll:
    row, reduction, column = IndexVar("i"), IndexVar("k"), IndexVar("j")
    result = TensorVar("C", fmt="dd")
    left = TensorVar("A", fmt="ds")
    right = TensorVar("B", fmt="dd")
    assignment = TensorAssign(
        result[row, column],
        left[row, reduction] * right[reduction, column],
        op=Operation.ADD,
    )
    return ForAll(row, ForAll(reduction, ForAll(column, assignment)))


def _stage_values(context: CompilationContext) -> list[str]:
    return [record.stage_id.value for record in context.stage_run_records]


def _exact_stage_record_values(
    context: CompilationContext,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        tuple(getattr(record, field.name) for field in fields(type(record)))
        for record in context.stage_run_records
    )


def _isolate_compiler_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})


def _compile_explicit(
    monkeypatch: pytest.MonkeyPatch,
    options: CompileOptions,
    context: CompilationContext,
    *,
    prepared_builds: Optional[list[object]] = None,
) -> TensorSpec:
    _isolate_compiler_caches(monkeypatch)

    def stop_before_native(prepared: object) -> object:
        if prepared_builds is not None:
            prepared_builds.append(prepared)
        return object()

    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", stop_before_native)
    result = ops.einsum(
        "ik,kj->ij",
        *_spmm_specs(),
        compile_only=True,
        format="dd",
        _compile_options=options,
        _compilation_context=context,
    )
    assert isinstance(result, TensorSpec)
    return result


def _direct_lower(
    options: CompileOptions,
    context: Optional[CompilationContext],
) -> tuple[CINLowerer, object, str]:
    source = _build_spmm_cin()
    scheduled = Scheduler.apply_schedule(
        source,
        Schedule(loop_order=("i", "k", "j")),
        compile_options=options,
        compilation_context=context,
    )
    lowerer = CINLowerer(
        compile_options=options,
        compilation_context=context,
    )
    lowered = lowerer._lower_owned_IndexStmt(scheduled)
    cpp = LLIRLowerer(compile_options=options).lower_llir(lowered)
    return lowerer, lowered, cpp


def test_stage_identities_records_and_owner_are_typed_frozen_and_stable() -> None:
    assert CANONICAL_COMPILER_STAGES == (
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION,
        CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION,
        CompilerStageId.LEGACY_CIN_ADAPTATION,
        CompilerStageId.CIN_LOWERING,
        CompilerStageId.RESULT_ABI_ASSEMBLY,
        CompilerStageId.SCHEDULE_LOWERING,
        CompilerStageId.LLIR_TO_CPP_GENERATION,
        CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY,
    )
    assert _FULL_STAGE_SEQUENCE == [
        "frontend_validated_operation_construction",
        "cin_normalization_and_verification",
        "scheduling_and_loop_plan_construction",
        "legacy_cin_adaptation",
        "cin_lowering",
        "result_abi_assembly",
        "schedule_lowering",
        "llir_to_cpp_generation",
        "kernel_name_and_build_request_assembly",
    ]

    record = CompilerStageRunRecord(
        sequence_index=0,
        stage_id=CompilerStageId.CIN_LOWERING,
        nested_within=None,
        duration_ns=17,
    )
    assert record == replace(record, duration_ns=999_999)
    assert record != replace(record, sequence_index=1)
    with pytest.raises(FrozenInstanceError):
        record.stage_id = CompilerStageId.SCHEDULE_LOWERING  # type: ignore[misc]

    options = _default_options()
    context = CompilationContext(options)
    with pytest.raises(FrozenInstanceError):
        context.compile_options = options  # type: ignore[misc]
    token = context.begin_stage(
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        compile_options=options,
    )
    assert type(token) is CompilerStageToken
    with pytest.raises(FrozenInstanceError):
        token.stage_id = CompilerStageId.CIN_LOWERING  # type: ignore[misc]
    context.complete_stage(token)
    assert type(context.stage_run_records) is tuple
    assert context.stage_run_records[0].sequence_index == 0


def test_context_validation_rejects_foreign_options_tokens_and_nesting() -> None:
    options = _default_options()
    other = _default_options(verify_cin=True)
    with pytest.raises(CompilationContextError) as invalid_owner:
        CompilationContext("options")  # type: ignore[arg-type]
    assert invalid_owner.value.diagnostic.code == "invalid_compile_options"

    context = CompilationContext(options)
    with pytest.raises(CompilationContextError) as invalid_stage:
        context.begin_stage("cin_lowering", compile_options=options)  # type: ignore[arg-type]
    assert invalid_stage.value.diagnostic.code == "invalid_stage_id"
    with pytest.raises(CompilationContextError) as detached_options:
        context.begin_stage(
            CompilerStageId.CIN_LOWERING,
            compile_options=other,
        )
    assert detached_options.value.diagnostic.code == "detached_compile_options"
    with pytest.raises(CompilationContextError) as invalid_nesting:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
            nested_within=CompilerStageId.CIN_LOWERING,
        )
    assert invalid_nesting.value.diagnostic.code == "invalid_stage_nesting"
    with pytest.raises(CompilationContextError) as inactive_parent:
        context.begin_stage(
            CompilerStageId.RESULT_ABI_ASSEMBLY,
            compile_options=options,
            nested_within=CompilerStageId.CIN_LOWERING,
        )
    assert inactive_parent.value.diagnostic.code == "inactive_parent_stage"

    token = context.begin_stage(
        CompilerStageId.CIN_LOWERING,
        compile_options=options,
    )
    foreign = CompilationContext(options)
    with pytest.raises(CompilationContextError) as detached_token:
        foreign.complete_stage(token)
    assert detached_token.value.diagnostic.code == "detached_stage_token"
    context.complete_stage(token)
    with pytest.raises(CompilationContextError) as duplicate:
        context.complete_stage(token)
    assert duplicate.value.diagnostic.code == "completed_stage_token"


def test_context_enforces_strict_lifo_nesting_and_exact_token_identity() -> None:
    options = _default_options()
    context = CompilationContext(options)
    frontend = context.begin_stage(
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        compile_options=options,
    )
    with pytest.raises(CompilationContextError) as overlap:
        context.begin_stage(
            CompilerStageId.LLIR_TO_CPP_GENERATION,
            compile_options=options,
        )
    assert overlap.value.diagnostic.code == "overlapping_stage"
    context.complete_stage(frontend)

    lowering = context.begin_stage(
        CompilerStageId.CIN_LOWERING,
        compile_options=options,
    )
    assembly = context.begin_stage(
        CompilerStageId.RESULT_ABI_ASSEMBLY,
        compile_options=options,
        nested_within=CompilerStageId.CIN_LOWERING,
    )
    with pytest.raises(CompilationContextError) as parent_first:
        context.complete_stage(lowering)
    assert parent_first.value.diagnostic.code == "unbalanced_stage_stack"

    forged = replace(
        assembly,
        stage_id=CompilerStageId.LLIR_TO_CPP_GENERATION,
        started_ns=0,
    )
    with pytest.raises(CompilationContextError) as forged_token:
        context.complete_stage(forged)
    assert forged_token.value.diagnostic.code == "inactive_stage_token"
    assert context.is_stage_active(assembly)
    context.complete_stage(assembly)
    context.complete_stage(lowering)


def test_deterministic_clock_failure_retains_child_and_makes_owner_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter(range(0, 10_000, 100))
    monkeypatch.setattr(context_module, "perf_counter_ns", lambda: next(ticks))
    options = _default_options()
    context = CompilationContext(options)
    lowering = context.begin_stage(
        CompilerStageId.CIN_LOWERING,
        compile_options=options,
    )
    assembly = context.begin_stage(
        CompilerStageId.RESULT_ABI_ASSEMBLY,
        compile_options=options,
        nested_within=CompilerStageId.CIN_LOWERING,
    )
    context.complete_stage(assembly)
    context.fail_stage(lowering)

    records = context.stage_run_records
    assert [record.sequence_index for record in records] == [0]
    assert [record.stage_id for record in records] == [
        CompilerStageId.RESULT_ABI_ASSEMBLY
    ]
    assert [record.duration_ns for record in records] == [100]
    assert records[0].nested_within is CompilerStageId.CIN_LOWERING
    with pytest.raises(CompilationContextError) as later_work:
        context.begin_stage(
            CompilerStageId.LLIR_TO_CPP_GENERATION,
            compile_options=options,
        )
    assert later_work.value.diagnostic.code == "failed_compilation"


def test_cancelled_optional_stage_publishes_nothing_and_owner_remains_usable() -> None:
    options = _default_options()
    context = CompilationContext(options)
    cancelled = context.begin_stage(
        CompilerStageId.CIN_LOWERING,
        compile_options=options,
    )
    context.cancel_stage(cancelled)
    assert context.stage_run_records == ()
    with pytest.raises(CompilationContextError) as retired:
        context.complete_stage(cancelled)
    assert retired.value.diagnostic.code == "retired_stage_token"

    cpp = context.begin_stage(
        CompilerStageId.LLIR_TO_CPP_GENERATION,
        compile_options=options,
    )
    context.complete_stage(cpp)
    assert _stage_values(context) == [CompilerStageId.LLIR_TO_CPP_GENERATION.value]


def test_context_accepts_managed_pass_records_only_during_active_lowering() -> None:
    options = _default_options()
    context = CompilationContext(options)
    with pytest.raises(CompilationContextError) as inactive:
        context.record_llir_pass_runs((), compile_options=options)
    assert inactive.value.diagnostic.code == "inactive_cin_lowering"

    lowering = context.begin_stage(
        CompilerStageId.CIN_LOWERING,
        compile_options=options,
    )
    context.record_llir_pass_runs((), compile_options=options)
    context.complete_stage(lowering)


@pytest.mark.parametrize(
    ("verify_cin", "pass_options"),
    [
        (False, PRODUCTION_LLIR_PASS_OPTIONS),
        (True, DEBUG_LLIR_PASS_OPTIONS),
    ],
    ids=("production", "debug"),
)
def test_production_and_debug_record_the_exact_complete_stage_sequence(
    monkeypatch: pytest.MonkeyPatch,
    verify_cin: bool,
    pass_options: LLIRPassOptions,
) -> None:
    options = _explicit_options(
        verify_cin=verify_cin,
        llir_pass_options=pass_options,
    )
    context = CompilationContext(options)
    _compile_explicit(monkeypatch, options, context)

    assert _stage_values(context) == _EXPLICIT_STAGE_SEQUENCE
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(len(_EXPLICIT_STAGE_SEQUENCE))
    )
    assert all(
        type(record.duration_ns) is int and record.duration_ns >= 0
        for record in context.stage_run_records
    )
    assert [record.nested_within for record in context.stage_run_records] == [
        None,
        None,
        None,
        None,
        None,
        None,
        CompilerStageId.CIN_LOWERING,
        None,
        None,
        None,
    ]


def test_packed_storage_activation_is_independent_and_records_complete_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                4,
                placement="outermost",
                accum="direct",
                unroll=False,
            ),
            TileSpec(
                "j",
                3,
                placement="child_of:k_out",
                kind="panel",
                accum="direct",
            ),
        ),
        relayout=RelayoutSpec("B", "k", 4, scope_var="j"),
        tag="structured-packed-storage",
        parallel_loop="i",
    )
    schedule_snapshot = replace(schedule)
    schedule_identity = (schedule.cache_key, hash(schedule))
    options = _default_options(requested_schedule=schedule)
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    specs = _spmm_specs()
    specs_snapshot = tuple(spec.metadata for spec in specs)
    prepared_builds: list[utils._PreparedJITBuild] = []

    def record_prepared(prepared: utils._PreparedJITBuild) -> object:
        prepared_builds.append(prepared)
        return object()

    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", record_prepared)

    first_context = CompilationContext(options)
    first = ops.einsum(
        "ij,jk->ik",
        *specs,
        compile_only=True,
        format="dd",
        _compile_options=options,
        _compilation_context=first_context,
    )
    first_stage_records = first_context.stage_run_records
    first_pass_records = first_context.llir_pass_run_records
    ops._kernel_cache.clear()
    ops._einsum_dispatch_cache.clear()

    second_context = CompilationContext(options)
    second = ops.einsum(
        "ij,jk->ik",
        *specs,
        compile_only=True,
        format="dd",
        _compile_options=options,
        _compilation_context=second_context,
    )

    assert isinstance(first, TensorSpec)
    assert isinstance(second, TensorSpec)
    assert first == second
    assert first is not second
    assert len(prepared_builds) == 2
    assert prepared_builds[0] == prepared_builds[1]
    assert (
        prepared_builds[0].request.cpp_sources == prepared_builds[1].request.cpp_sources
    )
    expected = (
        "std::vector<float> packed_B_storage(" "(size_t) kTile_j * (size_t) kTile_k);"
    )
    assert (
        sum(source.count(expected) for source in prepared_builds[0].request.cpp_sources)
        == 1
    )
    assert prepared_builds[0].request.build_options is options.build
    assert prepared_builds[1].request.build_options is options.build
    assert first_context.compile_options is options
    assert second_context.compile_options is options
    assert _stage_values(first_context) == _EXPLICIT_STAGE_SEQUENCE
    assert _stage_values(second_context) == _EXPLICIT_STAGE_SEQUENCE
    assert [record.pass_name for record in first_pass_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.sequence_index for record in first_pass_records] == list(range(5))
    assert tuple(
        replace(record, duration_ns=0) for record in first_stage_records
    ) == tuple(
        replace(record, duration_ns=0) for record in second_context.stage_run_records
    )
    assert tuple(
        replace(record, duration_ns=0) for record in first_pass_records
    ) == tuple(
        replace(record, duration_ns=0)
        for record in second_context.llir_pass_run_records
    )
    assert all(record.duration_ns >= 0 for record in first_stage_records)
    assert all(record.duration_ns >= 0 for record in first_pass_records)
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert schedule == schedule_snapshot
    assert (schedule.cache_key, hash(schedule)) == schedule_identity
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity


def test_regblock_dual_path_records_both_schedule_and_lowering_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options(regblock_dual=True)
    context = CompilationContext(options)
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )

    result = ops.einsum(
        "ik,kj->ij",
        *_spmm_specs(),
        compile_only=True,
        format="dd",
        _compile_options=options,
        _compilation_context=context,
    )

    assert isinstance(result, TensorSpec)
    assert _stage_values(context) == [
        _FULL_STAGE_SEQUENCE[0],
        _FULL_STAGE_SEQUENCE[2],
        _FULL_STAGE_SEQUENCE[1],
        _FULL_STAGE_SEQUENCE[2],
        _FULL_STAGE_SEQUENCE[1],
        _FULL_STAGE_SEQUENCE[2],
        _FULL_STAGE_SEQUENCE[3],
        _FULL_STAGE_SEQUENCE[4],
        _FULL_STAGE_SEQUENCE[5],
        _FULL_STAGE_SEQUENCE[3],
        _FULL_STAGE_SEQUENCE[4],
        _FULL_STAGE_SEQUENCE[5],
        _FULL_STAGE_SEQUENCE[4],
        _FULL_STAGE_SEQUENCE[7],
        _FULL_STAGE_SEQUENCE[8],
    ]
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ] * 2


def test_regblock_dual_second_arm_failure_keeps_first_arm_and_suppresses_codegen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options(regblock_dual=True)
    context = CompilationContext(options)
    error = RuntimeError("injected second dual-arm failure")
    calls = 0
    original_factor = llir_pass_manager.hoist_loop_invariant_factors

    def fail_second_factor(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error
        return original_factor(*args, **kwargs)

    monkeypatch.setattr(
        llir_pass_manager,
        "hoist_loop_invariant_factors",
        fail_second_factor,
    )
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert _stage_values(context) == [
        _FULL_STAGE_SEQUENCE[0],
        _FULL_STAGE_SEQUENCE[2],
        _FULL_STAGE_SEQUENCE[1],
        _FULL_STAGE_SEQUENCE[2],
        _FULL_STAGE_SEQUENCE[1],
        _FULL_STAGE_SEQUENCE[2],
        _FULL_STAGE_SEQUENCE[3],
        _FULL_STAGE_SEQUENCE[4],
        _FULL_STAGE_SEQUENCE[5],
        _FULL_STAGE_SEQUENCE[3],
    ]
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
    ]


def test_regblock_dual_stitch_failure_has_failed_cin_stage_and_suppresses_codegen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options(regblock_dual=True)
    context = CompilationContext(options)
    error = RuntimeError("injected dual-path stitch failure")
    monkeypatch.setattr(ops, "_stitch_regblock_dual_path", _raise_same(error))
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert _stage_values(context) == [
        _FULL_STAGE_SEQUENCE[0],
        _FULL_STAGE_SEQUENCE[2],
        _FULL_STAGE_SEQUENCE[1],
        _FULL_STAGE_SEQUENCE[2],
        _FULL_STAGE_SEQUENCE[1],
        _FULL_STAGE_SEQUENCE[2],
        _FULL_STAGE_SEQUENCE[3],
        _FULL_STAGE_SEQUENCE[4],
        _FULL_STAGE_SEQUENCE[5],
        _FULL_STAGE_SEQUENCE[3],
        _FULL_STAGE_SEQUENCE[4],
        _FULL_STAGE_SEQUENCE[5],
    ]
    assert len(context.llir_pass_run_records) == 10


def test_regblock_dual_declined_stitch_records_only_artifact_producing_lowerings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options(regblock_dual=True)
    context = CompilationContext(options)
    monkeypatch.setattr(ops, "_stitch_regblock_dual_path", lambda *args: None)
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )

    result = ops.einsum(
        "ik,kj->ij",
        *_spmm_specs(),
        compile_only=True,
        format="dd",
        _compile_options=options,
        _compilation_context=context,
    )

    assert isinstance(result, TensorSpec)
    assert _stage_values(context) == [
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value,
        CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION.value,
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value,
        CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION.value,
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value,
        CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION.value,
        CompilerStageId.LEGACY_CIN_ADAPTATION.value,
        CompilerStageId.CIN_LOWERING.value,
        CompilerStageId.RESULT_ABI_ASSEMBLY.value,
        CompilerStageId.LEGACY_CIN_ADAPTATION.value,
        CompilerStageId.CIN_LOWERING.value,
        CompilerStageId.RESULT_ABI_ASSEMBLY.value,
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value,
        CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION.value,
        CompilerStageId.LEGACY_CIN_ADAPTATION.value,
        CompilerStageId.CIN_LOWERING.value,
        CompilerStageId.RESULT_ABI_ASSEMBLY.value,
        CompilerStageId.LLIR_TO_CPP_GENERATION.value,
        CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY.value,
    ]
    assert (
        sum(
            record.stage_id is CompilerStageId.CIN_LOWERING
            for record in context.stage_run_records
        )
        == 3
    )


def test_debug_verification_policy_is_not_enabled_by_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verification_calls: list[object] = []
    original_verify = cin_analysis.verify_cin

    def counting_verify(cin: object) -> None:
        verification_calls.append(cin)
        original_verify(cin)  # type: ignore[arg-type]

    monkeypatch.setattr(cin_analysis, "verify_cin", counting_verify)
    production = _default_options()
    normalize_cin(
        _build_spmm_cin(),
        compile_options=production,
        compilation_context=CompilationContext(production),
    )
    assert verification_calls == []

    debug = _default_options(
        verify_cin=True,
        llir_pass_options=DEBUG_LLIR_PASS_OPTIONS,
    )
    normalize_cin(
        _build_spmm_cin(),
        compile_options=debug,
        compilation_context=CompilationContext(debug),
    )
    assert len(verification_calls) == 1


def test_one_options_snapshot_routes_through_one_owner_without_rereads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _explicit_options()
    context = CompilationContext(options)
    routed_options: list[CompileOptions] = []
    original_begin = CompilationContext.begin_stage
    original_prepare = ops._prepare_jit_build

    def recording_begin(
        self: CompilationContext,
        stage_id: CompilerStageId,
        *,
        compile_options: CompileOptions,
        nested_within: Optional[CompilerStageId] = None,
    ) -> CompilerStageToken:
        routed_options.append(compile_options)
        return original_begin(
            self,
            stage_id,
            compile_options=compile_options,
            nested_within=nested_within,
        )

    def recording_prepare(*args: object, **kwargs: object) -> object:
        routed_options.append(kwargs["compile_options"])  # type: ignore[arg-type]
        return original_prepare(*args, **kwargs)

    def forbidden_snapshot(*args: object, **kwargs: object) -> CompileOptions:
        raise AssertionError("a compiler stage resnapshotted process state")

    environment_reads: list[str] = []
    context_reads: list[str] = []
    verification_calls: list[object] = []
    snapshotted_environment_keys = {
        "CXX",
        "DEVELOPER_DIR",
        "MACOSX_DEPLOYMENT_TARGET",
        "PATH",
        "SCORCH_JIT_TUNE_HOOKS",
        "SCORCH_REGBLOCK",
        "SCORCH_REGBLOCK_DUAL",
        "SCORCH_REGBLOCK_MAX_N",
        "SCORCH_REGBLOCK_T",
        "SCORCH_VERIFY_CIN",
        "SDKROOT",
        "TORCH_DONT_CHECK_COMPILER_ABI",
        "TORCH_NO_COMPILER_WRAPPER",
    }
    environment_type = type(os.environ)
    original_environment_get = environment_type.get
    original_environment_getitem = environment_type.__getitem__

    def recording_environment_get(
        self: object, key: str, default: object = None
    ) -> object:
        if key in snapshotted_environment_keys:
            environment_reads.append(key)
        return original_environment_get(  # type: ignore[arg-type, call-overload]
            self, key, default
        )

    def recording_environment_getitem(self: object, key: str) -> str:
        if key in snapshotted_environment_keys:
            environment_reads.append(key)
        return original_environment_getitem(self, key)  # type: ignore[arg-type]

    class RecordingContextVar:
        def __init__(self, name: str, variable: object) -> None:
            self._name = name
            self._variable = variable

        def get(self, *args: object) -> object:
            context_reads.append(self._name)
            return self._variable.get(*args)  # type: ignore[attr-defined, no-any-return]

        def set(self, value: object) -> object:
            return self._variable.set(value)  # type: ignore[attr-defined, no-any-return]

        def reset(self, token: object) -> None:
            self._variable.reset(token)  # type: ignore[attr-defined]

    monkeypatch.setattr(CompilationContext, "begin_stage", recording_begin)
    monkeypatch.setattr(ops, "_prepare_jit_build", recording_prepare)
    monkeypatch.setattr(
        CompileOptions,
        "from_environment",
        classmethod(forbidden_snapshot),
    )
    monkeypatch.setattr(
        cin_analysis,
        "verify_cin",
        lambda cin: verification_calls.append(cin),
    )
    monkeypatch.setenv("SCORCH_VERIFY_CIN", "1")
    monkeypatch.setenv("SCORCH_REGBLOCK", "1")
    monkeypatch.setenv("SCORCH_REGBLOCK_DUAL", "1")
    with (
        cin_analysis.full_cin_verification(True),
        scheduler_module.regblock_force(True),
        scheduler_module.schedule_force(Schedule(loop_order=("k", "i", "j"))),
    ):
        monkeypatch.setattr(environment_type, "get", recording_environment_get)
        monkeypatch.setattr(
            environment_type,
            "__getitem__",
            recording_environment_getitem,
        )
        monkeypatch.setattr(
            cin_analysis,
            "_VERIFY_CIN_CONTEXT",
            RecordingContextVar(
                "verify_cin",
                cin_analysis._VERIFY_CIN_CONTEXT,
            ),
        )
        monkeypatch.setattr(
            scheduler_module,
            "_REGBLOCK_FORCE",
            RecordingContextVar(
                "regblock",
                scheduler_module._REGBLOCK_FORCE,
            ),
        )
        monkeypatch.setattr(
            scheduler_module,
            "_SCHEDULE_FORCE",
            RecordingContextVar(
                "schedule",
                scheduler_module._SCHEDULE_FORCE,
            ),
        )
        _compile_explicit(monkeypatch, options, context)

    assert routed_options
    assert all(routed is options for routed in routed_options)
    assert environment_reads == []
    assert context_reads == []
    assert verification_calls == []
    assert _stage_values(context) == _EXPLICIT_STAGE_SEQUENCE


def test_existing_llir_pass_records_are_context_owned_ordered_and_unchanged() -> None:
    options = _explicit_options()
    untimed_lowerer, _, untimed_cpp = _direct_lower(options, None)
    context = CompilationContext(options)
    timed_lowerer, _, timed_cpp = _direct_lower(options, context)

    expected_names = [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert timed_lowerer.llir_pass_run_records == untimed_lowerer.llir_pass_run_records
    assert context.llir_pass_run_records == timed_lowerer.llir_pass_run_records
    assert [
        record.pass_name for record in context.llir_pass_run_records
    ] == expected_names
    assert [record.sequence_index for record in context.llir_pass_run_records] == list(
        range(len(expected_names))
    )
    assert timed_cpp == untimed_cpp

    cin_record = next(
        record
        for record in context.stage_run_records
        if record.stage_id is CompilerStageId.CIN_LOWERING
    )
    pass_duration = sum(
        record.duration_ns or 0 for record in context.llir_pass_run_records
    )
    assert cin_record.duration_ns >= pass_duration


def test_result_abi_barrier_remains_between_invariant_and_dynamic_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    original_factor = llir_pass_manager.LLIRPassManager.run_loop_invariant_factor_hoist
    original_dynamic = llir_pass_manager.LLIRPassManager.run_dynamic_vector_access
    original_begin = CompilationContext.begin_stage
    original_complete = CompilationContext.complete_stage

    def recording_factor(self: object, *args: object, **kwargs: object) -> object:
        events.append(("pass", "hoist_loop_invariant_factors"))
        return original_factor(self, *args, **kwargs)

    def recording_dynamic(self: object, *args: object, **kwargs: object) -> object:
        events.append(("pass", "rewrite_dynamic_vector_accesses"))
        return original_dynamic(self, *args, **kwargs)

    def recording_begin(
        self: CompilationContext,
        stage_id: CompilerStageId,
        **kwargs: object,
    ) -> CompilerStageToken:
        events.append(("begin", stage_id))
        return original_begin(self, stage_id, **kwargs)  # type: ignore[arg-type]

    def recording_complete(
        self: CompilationContext,
        token: CompilerStageToken,
    ) -> None:
        events.append(("complete", token.stage_id))
        original_complete(self, token)

    monkeypatch.setattr(
        llir_pass_manager.LLIRPassManager,
        "run_loop_invariant_factor_hoist",
        recording_factor,
    )
    monkeypatch.setattr(
        llir_pass_manager.LLIRPassManager,
        "run_dynamic_vector_access",
        recording_dynamic,
    )
    monkeypatch.setattr(CompilationContext, "begin_stage", recording_begin)
    monkeypatch.setattr(CompilationContext, "complete_stage", recording_complete)

    options = _explicit_options()
    context = CompilationContext(options)
    _direct_lower(options, context)
    factor = events.index(("pass", "hoist_loop_invariant_factors"))
    begin = events.index(("begin", CompilerStageId.RESULT_ABI_ASSEMBLY))
    complete = events.index(("complete", CompilerStageId.RESULT_ABI_ASSEMBLY))
    dynamic = events.index(("pass", "rewrite_dynamic_vector_accesses"))
    assert factor < begin < complete < dynamic
    record = next(
        record
        for record in context.stage_run_records
        if record.stage_id is CompilerStageId.RESULT_ABI_ASSEMBLY
    )
    assert record.nested_within is CompilerStageId.CIN_LOWERING


def test_result_abi_record_is_not_published_before_barrier_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _explicit_options()
    context = CompilationContext(options)
    error = RuntimeError("injected result/ABI artifact validation failure")
    monkeypatch.setattr(
        llir_pass_manager.LLIRPassManager,
        "validate_body_assembly_artifact",
        _raise_same(error),
    )
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]


def test_einsum_frontend_completes_before_a_nested_runtime_relayout_native_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    error = RuntimeError("stop nested relayout before native work")
    observed_at_native: list[list[str]] = []

    def stop_nested_relayout(prepared: object) -> object:
        observed_at_native.append(_stage_values(context))
        raise error

    monkeypatch.setattr(
        stensor_module, "_load_validated_prepared_kernel", stop_nested_relayout
    )
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", stop_nested_relayout)
    _isolate_compiler_caches(monkeypatch)
    transposed = scorch.STensor.from_torch(
        torch.ones(3, 2, 4),
        mode_order=[1, 0, 2],
    )

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ijk,ijk->ijk",
            transposed,
            TensorSpec("ddd", (3, 2, 4), name="B"),
            compile_only=True,
            format="ddd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert len(observed_at_native) == 1
    assert observed_at_native[0] == [
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value,
        *_MANUAL_STAGE_SEQUENCE,
    ]
    assert _stage_values(context) == observed_at_native[0]


def test_einsum_preserves_topological_then_selected_operand_relayout_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options(
        requested_schedule=Schedule(loop_order=("i", "j", "l", "k"))
    )
    context = CompilationContext(options)
    original_relayout = ops._with_compiler_mode_order
    requested_orders: list[tuple[str, tuple[int, ...]]] = []

    def recording_relayout(
        tensor: object,
        mode_order: object,
        compile_options: CompileOptions,
        compilation_context: Optional[CompilationContext] = None,
    ) -> object:
        requested_orders.append(
            (tensor.name, tuple(mode_order))  # type: ignore[attr-defined, arg-type]
        )
        return original_relayout(
            tensor,  # type: ignore[arg-type]
            mode_order,  # type: ignore[arg-type]
            compile_options,
            compilation_context,
        )

    monkeypatch.setattr(ops, "_with_compiler_mode_order", recording_relayout)
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )
    left = TensorSpec("ddd", (2, 3, 4), name="A", mode_order=(0, 2, 1))
    right = TensorSpec("dd", (4, 5), name="B", mode_order=(1, 0))

    result = ops.einsum(
        "ijk,kl->ijl",
        left,
        right,
        compile_only=True,
        format="ddd",
        _compile_options=options,
        _compilation_context=context,
    )

    assert isinstance(result, TensorSpec)
    assert [order for name, order in requested_orders if name == "B"] == [
        (0, 1),
        (1, 0),
    ]
    assert left.mode_order == (0, 2, 1)
    assert right.mode_order == (1, 0)
    assert _stage_values(context) == _EXPLICIT_STAGE_SEQUENCE


def test_timing_is_excluded_from_source_name_request_and_cache_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stepped_clock(step: int) -> Callable[[], int]:
        current = 0

        def now() -> int:
            nonlocal current
            current += step
            return current

        return now

    options = _explicit_options()
    first_context = CompilationContext(options)
    first_builds: list[object] = []
    monkeypatch.setattr(context_module, "perf_counter_ns", stepped_clock(7))
    first_result = _compile_explicit(
        monkeypatch,
        options,
        first_context,
        prepared_builds=first_builds,
    )
    first_records = first_context.stage_run_records
    monkeypatch.undo()

    second_context = CompilationContext(options)
    second_builds: list[object] = []
    monkeypatch.setattr(context_module, "perf_counter_ns", stepped_clock(701))
    second_result = _compile_explicit(
        monkeypatch,
        options,
        second_context,
        prepared_builds=second_builds,
    )
    assert first_result == second_result
    assert len(first_builds) == len(second_builds) == 1
    assert first_builds[0] == second_builds[0]
    first_prepared = first_builds[0]
    second_prepared = second_builds[0]
    assert isinstance(first_prepared, utils._PreparedJITBuild)
    assert isinstance(second_prepared, utils._PreparedJITBuild)
    assert tuple(field.name for field in fields(type(first_prepared))) == (
        "request",
        "cache_key",
        "so_path",
    )
    assert tuple(field.name for field in fields(type(first_prepared.request))) == (  # type: ignore[attr-defined]
        "name",
        "cpp_sources",
        "functions",
        "extra_cflags",
        "extra_ldflags",
        "build_directory",
        "build_options",
    )
    assert first_prepared.request.name == second_prepared.request.name  # type: ignore[attr-defined]
    assert first_prepared.request.cpp_sources == second_prepared.request.cpp_sources  # type: ignore[attr-defined]
    assert first_prepared.cache_key == second_prepared.cache_key  # type: ignore[attr-defined]
    assert first_prepared.request.build_options is options.build  # type: ignore[attr-defined]
    assert second_prepared.request.build_options is options.build  # type: ignore[attr-defined]
    assert first_context.compile_options is options
    assert second_context.compile_options is options

    canonical_cache_digest(options.cache_key)
    canonical_cache_digest(options.semantic_cache_key)
    with pytest.raises(TypeError):
        canonical_cache_digest((first_records[0],))
    with pytest.raises(TypeError):
        canonical_cache_digest((first_context,))
    assert tuple(replace(record, duration_ns=0) for record in first_records) == tuple(
        replace(record, duration_ns=0) for record in second_context.stage_run_records
    )
    assert [record.duration_ns for record in first_records] != [
        record.duration_ns for record in second_context.stage_run_records
    ]


def test_known_nnz_extent_is_independent_across_full_compilations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    specs = _all_coo_sddmm_specs()
    specs_snapshot = tuple(spec.metadata for spec in specs)
    prepared_builds: list[utils._PreparedJITBuild] = []

    def record_prepared(prepared: utils._PreparedJITBuild) -> object:
        prepared_builds.append(prepared)
        return object()

    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", record_prepared)
    first_context = CompilationContext(options)
    with scheduler_module.regblock_force(False):
        first = ops.einsum(
            "ij,ik,jk->ij",
            *specs,
            compile_only=True,
            format="oo",
            _compile_options=options,
            _compilation_context=first_context,
        )
    first_stage_records = first_context.stage_run_records
    first_pass_records = first_context.llir_pass_run_records
    ops._kernel_cache.clear()
    ops._einsum_dispatch_cache.clear()

    second_context = CompilationContext(options)
    with scheduler_module.regblock_force(False):
        second = ops.einsum(
            "ij,ik,jk->ij",
            *specs,
            compile_only=True,
            format="oo",
            _compile_options=options,
            _compilation_context=second_context,
        )

    assert isinstance(first, TensorSpec)
    assert isinstance(second, TensorSpec)
    assert first == second
    assert first is not second
    assert len(prepared_builds) == 2
    assert prepared_builds[0] == prepared_builds[1]
    assert (
        prepared_builds[0].request.cpp_sources == prepared_builds[1].request.cpp_sources
    )
    assert (
        sum(
            source.count("int64_t _known_nnz = A_values.size(0);")
            for source in prepared_builds[0].request.cpp_sources
        )
        == 1
    )
    assert any(
        "torch::empty({_known_nnz}, torch::kFloat32);" in source
        for source in prepared_builds[0].request.cpp_sources
    )
    assert prepared_builds[0].request.build_options is options.build
    assert prepared_builds[1].request.build_options is options.build
    assert first_context.compile_options is options
    assert second_context.compile_options is options
    assert _stage_values(first_context) == _AUTO_STAGE_SEQUENCE
    assert _stage_values(second_context) == _AUTO_STAGE_SEQUENCE
    assert tuple(
        replace(record, duration_ns=0) for record in first_stage_records
    ) == tuple(
        replace(record, duration_ns=0) for record in second_context.stage_run_records
    )
    assert tuple(
        replace(record, duration_ns=0) for record in first_pass_records
    ) == tuple(
        replace(record, duration_ns=0)
        for record in second_context.llir_pass_run_records
    )
    assert [record.pass_name for record in first_pass_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity


def test_compressed_base_load_is_independent_across_full_compilations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    specs = (
        TensorSpec("ds", (2, 3), name="A"),
        TensorSpec("ds", (3, 4), name="B"),
    )
    specs_snapshot = tuple(spec.metadata for spec in specs)
    prepared_builds: list[utils._PreparedJITBuild] = []

    def record_prepared(prepared: utils._PreparedJITBuild) -> object:
        prepared_builds.append(prepared)
        return object()

    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", record_prepared)
    first_context = CompilationContext(options)
    first = ops.einsum(
        "ik,kj->ij",
        *specs,
        compile_only=True,
        format="ds",
        _compile_options=options,
        _compilation_context=first_context,
    )
    first_stage_records = first_context.stage_run_records
    first_pass_records = first_context.llir_pass_run_records
    ops._kernel_cache.clear()
    ops._einsum_dispatch_cache.clear()

    second_context = CompilationContext(options)
    second = ops.einsum(
        "ik,kj->ij",
        *specs,
        compile_only=True,
        format="ds",
        _compile_options=options,
        _compilation_context=second_context,
    )

    assert isinstance(first, TensorSpec)
    assert isinstance(second, TensorSpec)
    assert first == second
    assert first is not second
    assert len(prepared_builds) == 2
    assert prepared_builds[0] == prepared_builds[1]
    assert prepared_builds[0].request.cpp_sources == (
        prepared_builds[1].request.cpp_sources
    )
    assert (
        sum(
            source.count("int64_t _base1 = _offset1[i];")
            for source in prepared_builds[0].request.cpp_sources
        )
        == 1
    )
    assert prepared_builds[0].request.build_options is options.build
    assert prepared_builds[1].request.build_options is options.build
    assert first_context.compile_options is options
    assert second_context.compile_options is options
    assert _stage_values(first_context) == _AUTO_STAGE_SEQUENCE
    assert _stage_values(second_context) == _AUTO_STAGE_SEQUENCE
    assert tuple(
        replace(record, duration_ns=0) for record in first_stage_records
    ) == tuple(
        replace(record, duration_ns=0) for record in second_context.stage_run_records
    )
    assert tuple(
        replace(record, duration_ns=0) for record in first_pass_records
    ) == tuple(
        replace(record, duration_ns=0)
        for record in second_context.llir_pass_run_records
    )
    assert [record.configuration_name for record in first_pass_records] == [
        "compressed_where_openmp",
        "count",
        "fill",
        "sparse_prefetch",
        "dense_pointer_hoist",
        "single_iteration_loop_elimination",
        "loop_invariant_factor_hoist",
        "dynamic_vector_access",
    ]
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity


def test_two_options_snapshots_have_independent_records_results_and_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = _explicit_options()
    debug = _explicit_options(
        verify_cin=True,
        llir_pass_options=DEBUG_LLIR_PASS_OPTIONS,
    )
    assert production is not debug
    assert production.cache_key != debug.cache_key
    production_context = CompilationContext(production)
    production_builds: list[object] = []
    first_result = _compile_explicit(
        monkeypatch,
        production,
        production_context,
        prepared_builds=production_builds,
    )
    first_stage_snapshot = production_context.stage_run_records
    first_pass_snapshot = production_context.llir_pass_run_records
    monkeypatch.undo()

    debug_context = CompilationContext(debug)
    debug_builds: list[object] = []
    second_result = _compile_explicit(
        monkeypatch,
        debug,
        debug_context,
        prepared_builds=debug_builds,
    )
    assert first_result == second_result
    assert first_result is not second_result
    assert production_context.compile_options is production
    assert debug_context.compile_options is debug
    assert production_context.stage_run_records == first_stage_snapshot
    assert production_context.llir_pass_run_records == first_pass_snapshot
    assert production_context.stage_run_records is not debug_context.stage_run_records
    assert all(
        first_record is not second_record
        for first_record, second_record in zip(
            production_context.stage_run_records,
            debug_context.stage_run_records,
        )
    )
    assert production_builds[0].request.cpp_sources == debug_builds[0].request.cpp_sources  # type: ignore[attr-defined]
    assert all(
        not record.verified_before and not record.verified_after
        for record in first_pass_snapshot
    )
    assert all(
        record.verified_before and record.verified_after
        for record in debug_context.llir_pass_run_records
    )


def test_caller_owned_cin_llir_and_first_result_are_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _explicit_options()
    source = _build_spmm_cin()
    source_before = canonical_cin_dump(source)
    context = CompilationContext(options)
    scheduled = Scheduler.apply_schedule(
        source,
        Schedule(loop_order=("i", "k", "j")),
        compile_options=options,
        compilation_context=context,
    )
    assert canonical_cin_dump(source) == source_before
    lowerer = CINLowerer(compile_options=options, compilation_context=context)
    lowered = lowerer._lower_owned_IndexStmt(scheduled)
    lowered_before = repr(lowered)
    LLIRLowerer(compile_options=options).lower_llir(lowered)
    assert repr(lowered) == lowered_before
    assert canonical_cin_dump(source) == source_before

    first_context = CompilationContext(options)
    first = _compile_explicit(monkeypatch, options, first_context)
    first_snapshot = first.metadata
    monkeypatch.undo()
    second_context = CompilationContext(options)
    _compile_explicit(monkeypatch, options, second_context)
    assert first.metadata == first_snapshot
    assert first_context.stage_run_records


def _raise_same(error: BaseException) -> Callable[..., object]:
    def failing(*args: object, **kwargs: object) -> object:
        raise error

    return failing


def _forbid_later_work(calls: list[str], name: str) -> Callable[..., object]:
    def forbidden(*args: object, **kwargs: object) -> object:
        calls.append(name)
        raise AssertionError(f"{name} ran after schedule verification failed")

    return forbidden


def _forbid_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str],
    boundaries: tuple[tuple[object, str, str], ...],
) -> None:
    for owner, attribute, name in boundaries:
        monkeypatch.setattr(
            owner,
            attribute,
            _forbid_later_work(calls, name),
        )


def _legality_identity(
    scheduled: ScheduledCIN,
    schedule: Schedule,
    options: CompileOptions,
    specs: Tuple[TensorSpec, TensorSpec],
) -> tuple[object, ...]:
    plan = scheduled.verified_loop_plan
    return (
        schedule.cache_key,
        hash(schedule),
        hash(plan),
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
        canonical_cache_digest(options.cache_key),
        ops._einsum_cache_key(
            "ik,kj->ij",
            specs,
            "dd",
            None,
            schedule,
            options,
        ),
        ops._codegen_kernel_cache_key(
            scheduled,
            None,
            schedule,
            options,
        ),
    )


def test_public_invalid_schedule_fails_its_stage_and_suppresses_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule(
        loop_order=("i", "k", "j"),
        parallel_loop="k",
        tag="invalid-parallel-reduction",
    )
    schedule_snapshot = replace(schedule)
    schedule_cache_key = schedule.cache_key
    options = _default_options(requested_schedule=schedule)
    options_fingerprint = options.cache_fingerprint
    context = CompilationContext(options)
    later_calls: list[str] = []

    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (CINLowerer, "_lower_owned_IndexStmt", "cin_lowering"),
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_codegen_kernel_cache_key", "codegen_cache_key"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(InvalidSchedule) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert type(failure.value) is InvalidSchedule
    assert _stage_values(context) == _EXPLICIT_STAGE_SEQUENCE[:3]
    assert [record.sequence_index for record in context.stage_run_records] == [
        0,
        1,
        2,
    ]
    assert all(
        _stage_values(context).count(stage) == 1
        for stage in _EXPLICIT_STAGE_SEQUENCE[:3]
    )
    assert context.llir_pass_run_records == ()
    assert later_calls == []
    assert schedule == schedule_snapshot
    assert schedule.cache_key == schedule_cache_key
    assert context.compile_options is options
    assert options.cache_fingerprint == options_fingerprint
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.LEGACY_CIN_ADAPTATION,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"
    assert CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION.value in str(
        suppressed.value
    )


def test_forged_scheduled_cin_fails_adapter_with_exact_prior_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule(loop_order=("i", "k", "j"), tag="direct-valid")
    options = _default_options(requested_schedule=schedule)
    context = CompilationContext(options)
    source = _build_spmm_cin()
    source_snapshot = canonical_cin_dump(source)
    scheduled = Scheduler.apply_schedule(
        source,
        schedule,
        compile_options=options,
        compilation_context=context,
    )
    plan = scheduled.verified_loop_plan
    plan_snapshot = replace(plan)
    plan_hash = hash(plan)
    normalized_snapshot = canonical_cin_dump(scheduled.normalized_cin)
    prior_records = context.stage_run_records
    prior_record_values = _exact_stage_record_values(context)
    assert _stage_values(context) == [
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value,
        CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION.value,
    ]

    analysis = cin_analysis.analyze_cin(scheduled.normalized_cin)
    assert len(analysis.reduction_index_ids) == 1
    forged_plan = replace(
        plan,
        parallel_loop=LoopRef(analysis.reduction_index_ids[0]),
    )
    forged = ScheduledCIN(scheduled.normalized_cin, forged_plan)
    with pytest.raises(InvalidSchedule) as direct_failure:
        verify_scheduled_cin(forged)
    assert direct_failure.value.diagnostics[0].code == "parallel_reduction"

    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (CINLowerer, "_lower_prepared_index_stmt", "cin_lowering"),
            (ResultTensorAssembler, "emit_final_assembly", "result_abi"),
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )

    lowerer = CINLowerer(
        compile_options=options,
        compilation_context=context,
    )
    with pytest.raises(InvalidSchedule) as timed_failure:
        lowered = lowerer._lower_owned_IndexStmt(forged)
        cpp = LLIRLowerer(compile_options=options).lower_llir(lowered)
        ops._prepare_generated_kernel_build(  # pragma: no cover
            options.build.preamble_source,
            cpp,
            options,
            context,
        )

    assert type(timed_failure.value) is type(direct_failure.value)
    assert timed_failure.value.args == direct_failure.value.args
    assert timed_failure.value.diagnostics == direct_failure.value.diagnostics
    assert timed_failure.value.__cause__ is direct_failure.value.__cause__ is None
    assert context.stage_run_records == prior_records
    assert _exact_stage_record_values(context) == prior_record_values
    assert [record.sequence_index for record in context.stage_run_records] == [0, 1]
    assert context.llir_pass_run_records == ()
    assert later_calls == []
    assert canonical_cin_dump(source) == source_snapshot
    assert canonical_cin_dump(scheduled.normalized_cin) == normalized_snapshot
    assert scheduled.verified_loop_plan is plan
    assert plan == plan_snapshot
    assert hash(plan) == plan_hash
    assert schedule is options.requested_schedule
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.CIN_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"
    assert CompilerStageId.LEGACY_CIN_ADAPTATION.value in str(suppressed.value)


def test_artifact_verification_does_not_read_environment_or_contextvars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule(loop_order=("i", "k", "j"), tag="exact-options")
    options = _default_options(requested_schedule=schedule)
    scheduled = Scheduler.apply_schedule(
        _build_spmm_cin(),
        schedule,
        compile_options=options,
    )
    analysis = cin_analysis.analyze_cin(scheduled.normalized_cin)
    forged = ScheduledCIN(
        scheduled.normalized_cin,
        replace(
            scheduled.verified_loop_plan,
            parallel_loop=LoopRef(analysis.reduction_index_ids[0]),
        ),
    )
    snapshotted_environment_keys = {
        "SCORCH_REGBLOCK",
        "SCORCH_REGBLOCK_DUAL",
        "SCORCH_VERIFY_CIN",
    }
    environment_type = type(os.environ)
    original_environment_get = environment_type.get

    def forbidden_environment_get(
        self: object, key: str, default: object = None
    ) -> object:
        if key in snapshotted_environment_keys:
            raise AssertionError(f"artifact verification read environment key {key}")
        return original_environment_get(  # type: ignore[arg-type, call-overload]
            self, key, default
        )

    def forbidden_snapshot(*args: object, **kwargs: object) -> CompileOptions:
        raise AssertionError("artifact verification resnapshotted process state")

    class ForbiddenContextVar:
        def __init__(self, name: str) -> None:
            self._name = name

        def get(self, *args: object) -> object:
            raise AssertionError(f"artifact verification read {self._name} ContextVar")

    monkeypatch.setenv("SCORCH_VERIFY_CIN", "1")
    monkeypatch.setenv("SCORCH_REGBLOCK", "1")
    monkeypatch.setenv("SCORCH_REGBLOCK_DUAL", "1")
    with (
        cin_analysis.full_cin_verification(True),
        scheduler_module.regblock_force(True),
        scheduler_module.schedule_force(
            Schedule(loop_order=("k", "i", "j"), tag="ignored-context")
        ),
    ):
        with monkeypatch.context() as verifier_patch:
            verifier_patch.setattr(
                CompileOptions,
                "from_environment",
                classmethod(forbidden_snapshot),
            )
            verifier_patch.setattr(
                environment_type,
                "get",
                forbidden_environment_get,
            )
            verifier_patch.setattr(
                cin_analysis,
                "_VERIFY_CIN_CONTEXT",
                ForbiddenContextVar("verify_cin"),
            )
            verifier_patch.setattr(
                scheduler_module,
                "_REGBLOCK_FORCE",
                ForbiddenContextVar("regblock"),
            )
            verifier_patch.setattr(
                scheduler_module,
                "_SCHEDULE_FORCE",
                ForbiddenContextVar("schedule"),
            )

            assert verify_scheduled_cin(scheduled) is scheduled
            lowerer = CINLowerer(compile_options=options)
            assert lowerer.compile_options is options
            with pytest.raises(InvalidSchedule):
                lowerer._prepare_scheduled_cin(
                    forged,
                    recurse=False,
                    ownership_transferred=False,
                )


def test_legality_verification_is_nonsemantic_and_plans_are_independent() -> None:
    source = _build_spmm_cin()
    source_snapshot = canonical_cin_dump(source)
    first_schedule = Schedule(
        loop_order=("i", "k", "j"),
        tiles=(TileSpec("j", 4, accum="direct", unroll=False),),
        tag="independent-width-four",
    )
    second_schedule = replace(
        first_schedule,
        tiles=(TileSpec("j", 8, accum="direct", unroll=False),),
        tag="independent-width-eight",
    )
    first_options = _default_options(requested_schedule=first_schedule)
    second_options = _default_options(requested_schedule=second_schedule)
    first_context = CompilationContext(first_options)
    second_context = CompilationContext(second_options)
    first = Scheduler.apply_schedule(
        source,
        first_schedule,
        compile_options=first_options,
        compilation_context=first_context,
    )
    second = Scheduler.apply_schedule(
        source,
        second_schedule,
        compile_options=second_options,
        compilation_context=second_context,
    )
    first_plan = first.verified_loop_plan
    second_plan = second.verified_loop_plan
    first_plan_snapshot = replace(first_plan)
    second_plan_snapshot = replace(second_plan)
    first_records = first_context.stage_run_records
    second_records = second_context.stage_run_records
    first_record_values = _exact_stage_record_values(first_context)
    second_record_values = _exact_stage_record_values(second_context)
    specs = _spmm_specs()

    first_identity = _legality_identity(
        first,
        first_schedule,
        first_options,
        specs,
    )
    second_identity = _legality_identity(
        second,
        second_schedule,
        second_options,
        specs,
    )

    assert verify_scheduled_cin(first) is first
    analysis = cin_analysis.analyze_cin(first.normalized_cin)
    forged = ScheduledCIN(
        first.normalized_cin,
        replace(
            first_plan,
            parallel_loop=LoopRef(analysis.reduction_index_ids[0]),
        ),
    )
    with pytest.raises(InvalidSchedule):
        verify_scheduled_cin(forged)
    assert verify_scheduled_cin(second) is second

    assert canonical_cin_dump(source) == source_snapshot
    assert first.verified_loop_plan is first_plan
    assert second.verified_loop_plan is second_plan
    assert first_plan == first_plan_snapshot
    assert second_plan == second_plan_snapshot
    assert first_plan != second_plan
    assert first_plan is not second_plan
    assert first_plan.tiles is not second_plan.tiles
    assert first_plan.tiles[0] is not second_plan.tiles[0]
    assert first_context.compile_options is first_options
    assert second_context.compile_options is second_options
    assert first_context.stage_run_records == first_records
    assert second_context.stage_run_records == second_records
    assert _exact_stage_record_values(first_context) == first_record_values
    assert _exact_stage_record_values(second_context) == second_record_values

    assert first_identity == _legality_identity(
        first,
        first_schedule,
        first_options,
        specs,
    )
    assert second_identity == _legality_identity(
        second,
        second_schedule,
        second_options,
        specs,
    )


@pytest.mark.parametrize(
    ("failure_site", "expected_prefix"),
    [
        ("normalization", _EXPLICIT_STAGE_SEQUENCE[:2]),
        ("scheduling", _EXPLICIT_STAGE_SEQUENCE[:3]),
        ("adapter", _EXPLICIT_STAGE_SEQUENCE[:4]),
        ("cin_pass", _EINSUM_PREFIX_THROUGH_ADAPTER),
        ("abi", _EINSUM_PREFIX_THROUGH_ADAPTER),
        ("kernel_abi_input_prologue", _EINSUM_PREFIX_THROUGH_ADAPTER),
        ("kernel_abi_validation", _EINSUM_PREFIX_THROUGH_ADAPTER),
        ("kernel_abi_extra_prologue", _EINSUM_PREFIX_THROUGH_ADAPTER),
        (
            "kernel_abi_function",
            [
                *_EINSUM_PREFIX_THROUGH_ADAPTER,
                CompilerStageId.RESULT_ABI_ASSEMBLY.value,
            ],
        ),
        ("schedule_lowering", _EXPLICIT_STAGE_SEQUENCE[:7]),
        ("cpp", _EXPLICIT_STAGE_SEQUENCE[:8]),
        ("request", _EXPLICIT_STAGE_SEQUENCE[:9]),
    ],
)
def test_stage_failures_raise_original_error_and_suppress_all_later_stages(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
    expected_prefix: list[str],
) -> None:
    options = _explicit_options()
    context = CompilationContext(options)
    error = RuntimeError(f"injected {failure_site} failure")
    later_calls: list[str] = []
    if failure_site == "normalization":
        monkeypatch.setattr(
            cin_analysis,
            "verify_cin_if_enabled",
            _raise_same(error),
        )
    elif failure_site == "scheduling":
        monkeypatch.setattr(
            Scheduler,
            "_apply_schedule_legacy",
            staticmethod(_raise_same(error)),
        )
    elif failure_site == "adapter":
        monkeypatch.setattr(
            cin_lowerer_module,
            "claim_legacy_cin_working_tree",
            _raise_same(error),
        )
    elif failure_site == "cin_pass":
        monkeypatch.setattr(
            llir_pass_manager,
            "hoist_loop_invariant_factors",
            _raise_same(error),
        )
    elif failure_site == "abi":
        monkeypatch.setattr(
            ResultTensorAssembler,
            "emit_final_assembly",
            _raise_same(error),
        )
    elif failure_site == "kernel_abi_input_prologue":
        monkeypatch.setattr(
            TorchCppKernelABI,
            "emit_input_prologue",
            _raise_same(error),
        )
    elif failure_site == "kernel_abi_validation":
        monkeypatch.setattr(
            TorchCppKernelABI,
            "emit_validation",
            _raise_same(error),
        )
    elif failure_site == "kernel_abi_extra_prologue":
        monkeypatch.setattr(
            TorchCppKernelABI,
            "emit_extra_tensor_prologue",
            _raise_same(error),
        )
    elif failure_site == "kernel_abi_function":
        monkeypatch.setattr(
            TorchCppKernelABI,
            "assemble_function",
            _raise_same(error),
        )
    elif failure_site == "schedule_lowering":
        monkeypatch.setattr(
            schedule_lowerer,
            "apply_schedule_to_llir",
            _raise_same(error),
        )
    elif failure_site == "cpp":
        monkeypatch.setattr(LLIRLowerer, "lower_llir", _raise_same(error))
    elif failure_site == "request":
        monkeypatch.setattr(ops, "_prepare_jit_build", _raise_same(error))
    else:  # pragma: no cover - the parameter list is exhaustive
        raise AssertionError(failure_site)
    _isolate_compiler_caches(monkeypatch)
    if failure_site.startswith("kernel_abi_"):
        _forbid_boundaries(
            monkeypatch,
            later_calls,
            (
                (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
                (LLIRLowerer, "lower_llir", "cpp_generation"),
                (ops, "_prepare_generated_kernel_build", "build_request"),
                (ops, "_load_validated_prepared_kernel", "native_load"),
            ),
        )
    else:
        monkeypatch.setattr(
            ops, "_load_validated_prepared_kernel", lambda prepared: object()
        )

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )
    assert failure.value is error
    assert _stage_values(context) == expected_prefix
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(len(expected_prefix))
    )

    if failure_site == "cin_pass":
        assert [record.pass_name for record in context.llir_pass_run_records] == [
            "insert_sparse_prefetch",
            "hoist_dense_pointers",
            "eliminate_single_iteration_loops",
        ]
    elif failure_site == "abi":
        assert [record.pass_name for record in context.llir_pass_run_records] == [
            "insert_sparse_prefetch",
            "hoist_dense_pointers",
            "eliminate_single_iteration_loops",
            "hoist_loop_invariant_factors",
        ]
    elif failure_site in {
        "kernel_abi_input_prologue",
        "kernel_abi_validation",
        "kernel_abi_extra_prologue",
    }:
        assert context.llir_pass_run_records == ()
        assert later_calls == []
        assert ops._kernel_cache == {}
        assert ops._einsum_dispatch_cache == {}
    elif failure_site == "kernel_abi_function":
        assert [record.pass_name for record in context.llir_pass_run_records] == [
            "insert_sparse_prefetch",
            "hoist_dense_pointers",
            "eliminate_single_iteration_loops",
            "hoist_loop_invariant_factors",
            "rewrite_dynamic_vector_accesses",
        ]
        assert later_calls == []
        assert ops._kernel_cache == {}
        assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.LLIR_TO_CPP_GENERATION,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_panel_bound_fails_schedule_stage_and_suppresses_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnknownPanelIndex(llir.Expr):
        pass

    schedule = Schedule(
        loop_order=("i", "k", "j"),
        tiles=(TileSpec("k", 2, kind="panel", accum="direct"),),
        parallel_loop="i",
        tag="malformed-structured-panel-bound",
    )
    schedule_snapshot = replace(schedule)
    schedule_hash = hash(schedule)
    schedule_cache_key = schedule.cache_key
    options = _default_options(requested_schedule=schedule)
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    context = CompilationContext(options)
    specs = _spmm_specs()
    observed_bounds: list[tuple[object, object]] = []
    original_matcher = schedule_lowerer.match_mode_position_bounds

    def forge_panel_begin(begin: object, end: object) -> llir.ArrayAccess:
        assert original_matcher(begin, end) is not None
        observed_bounds.append((begin, end))
        return llir.ArrayAccess(
            llir.Var("A1_pos", llir.DataType.PTR_INT),
            UnknownPanelIndex(),
        )

    monkeypatch.setattr(
        schedule_lowerer,
        "match_mode_position_bounds",
        forge_panel_begin,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *specs,
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    diagnostic = failure.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.stage == "schedule lowering"
    assert diagnostic.pass_name == "window_sparse_loop"
    assert diagnostic.node_type == "UnknownPanelIndex"
    assert diagnostic.path == ("root", "index")
    assert len(observed_bounds) == 1
    assert _stage_values(context) == _EXPLICIT_STAGE_SEQUENCE[:7]
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(7)
    )
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert later_calls == []
    assert context.compile_options is options
    assert options.requested_schedule is schedule
    assert schedule == schedule_snapshot
    assert hash(schedule) == schedule_hash
    assert schedule.cache_key == schedule_cache_key
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert specs[0].name == "A"
    assert specs[1].name == "B"
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.LLIR_TO_CPP_GENERATION,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"
    assert CompilerStageId.SCHEDULE_LOWERING.value in str(suppressed.value)


@pytest.mark.parametrize(
    ("malformation", "diagnostic_code", "node_type"),
    (
        ("field", "invalid_direct_init_var_type", "DataType"),
        ("child", "unknown_llir_node", "UnknownPackedExtent"),
        ("subclass", "unknown_llir_node", "UnknownDirectInit"),
    ),
)
def test_malformed_packed_storage_fails_schedule_owner_and_suppresses_later_work(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    diagnostic_code: str,
    node_type: str,
) -> None:
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                4,
                placement="outermost",
                accum="direct",
                unroll=False,
            ),
            TileSpec(
                "j",
                3,
                placement="child_of:k_out",
                kind="panel",
                accum="direct",
            ),
        ),
        relayout=RelayoutSpec("B", "k", 4, scope_var="j"),
        tag="malformed-packed-storage",
        parallel_loop="i",
    )
    schedule_snapshot = replace(schedule)
    schedule_identity = (schedule.cache_key, hash(schedule))
    options = _default_options(requested_schedule=schedule)
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    context = CompilationContext(options)
    specs = _spmm_specs()
    specs_snapshot = tuple(spec.metadata for spec in specs)
    original_builder = schedule_lowerer._packed_storage_declaration
    injected: list[tuple[str, llir.DataType]] = []

    class UnknownPackedExtent(llir.Expr):
        pass

    class UnknownDirectInit(llir.DirectInit):
        pass

    def build_malformed_packed_storage(
        *,
        storage_name: str,
        scalar_type: llir.DataType,
        stage_rows: str,
        stage_rows_type: llir.DataType,
        pack_tile_var: str,
    ) -> llir.DirectInit:
        declaration = original_builder(
            storage_name=storage_name,
            scalar_type=scalar_type,
            stage_rows=stage_rows,
            stage_rows_type=stage_rows_type,
            pack_tile_var=pack_tile_var,
        )
        injected.append((declaration.var.name, declaration.var.type))
        if malformation == "field":
            declaration.var.type = llir.DataType.VOID
            return declaration
        if malformation == "child":
            object.__setattr__(declaration, "args", (UnknownPackedExtent(),))
            return declaration
        return UnknownDirectInit(declaration.var, declaration.args)

    monkeypatch.setattr(
        schedule_lowerer,
        "_packed_storage_declaration",
        build_malformed_packed_storage,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ij,jk->ik",
            *specs,
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    diagnostic = failure.value.diagnostic
    assert injected == [
        ("packed_B_storage", llir.DataType.STD_VECTOR_FLOAT32),
    ]
    assert diagnostic.code == diagnostic_code
    assert diagnostic.stage == "schedule lowering"
    assert diagnostic.pass_name == "build_packed_storage"
    assert diagnostic.node_type == node_type
    assert _stage_values(context) == _EXPLICIT_STAGE_SEQUENCE[:7]
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(7)
    )
    assert all(record.duration_ns >= 0 for record in context.stage_run_records)
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.sequence_index for record in context.llir_pass_run_records] == list(
        range(5)
    )
    assert all(record.duration_ns >= 0 for record in context.llir_pass_run_records)
    assert later_calls == []
    assert context.compile_options is options
    assert options.requested_schedule is schedule
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert schedule == schedule_snapshot
    assert (schedule.cache_key, hash(schedule)) == schedule_identity
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert ops._kernel_cache == {}
    assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.LLIR_TO_CPP_GENERATION,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"
    assert CompilerStageId.SCHEDULE_LOWERING.value in str(suppressed.value)


@pytest.mark.parametrize(
    ("malformation", "diagnostic_code", "node_type", "diagnostic_path"),
    (
        (
            "field",
            "invalid_direct_init_var_type",
            "DataType",
            ("root", "var", "type"),
        ),
        (
            "child",
            "unknown_llir_node",
            "UnknownHeapResultExtent",
            ("root", "args", "[0]"),
        ),
        (
            "subclass",
            "unknown_llir_node",
            "UnknownHeapResultInit",
            ("root",),
        ),
    ),
)
def test_malformed_heap_result_storage_fails_schedule_owner_and_stops_later_work(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    diagnostic_code: str,
    node_type: str,
    diagnostic_path: tuple[str, ...],
) -> None:
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                4,
                placement="outermost",
                accum="heap",
                unroll=False,
            ),
        ),
        tag="malformed-heap-result-storage",
        parallel_loop="i",
    )
    schedule_snapshot = replace(schedule)
    schedule_identity = (schedule.cache_key, hash(schedule))
    options = _default_options(requested_schedule=schedule)
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    context = CompilationContext(options)
    specs = _spmm_specs()
    specs_snapshot = tuple(spec.metadata for spec in specs)
    original_builder = schedule_lowerer._heap_result_storage_declaration
    injected: list[tuple[str, llir.DataType, tuple[str, ...], str]] = []

    class UnknownHeapResultExtent(llir.Expr):
        pass

    class UnknownHeapResultInit(llir.DirectInit):
        pass

    def build_malformed_heap_result_storage(
        *,
        storage_name: str,
        scalar_type: llir.DataType,
        prefix_dimension_names: tuple[str, ...],
        tile_size_name: str,
    ) -> llir.DirectInit:
        declaration = original_builder(
            storage_name=storage_name,
            scalar_type=scalar_type,
            prefix_dimension_names=prefix_dimension_names,
            tile_size_name=tile_size_name,
        )
        injected.append(
            (
                storage_name,
                scalar_type,
                prefix_dimension_names,
                tile_size_name,
            )
        )
        if malformation == "field":
            declaration.var.type = llir.DataType.VOID
            return declaration
        if malformation == "child":
            object.__setattr__(declaration, "args", (UnknownHeapResultExtent(),))
            return declaration
        return UnknownHeapResultInit(declaration.var, declaration.args)

    monkeypatch.setattr(
        schedule_lowerer,
        "_heap_result_storage_declaration",
        build_malformed_heap_result_storage,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ij,jk->ik",
            *specs,
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    diagnostic = failure.value.diagnostic
    assert injected == [
        (
            "tiled_C_storage",
            llir.DataType.FLOAT32,
            ("C0_size",),
            "kTile_k",
        )
    ]
    assert diagnostic.code == diagnostic_code
    assert diagnostic.stage == "schedule lowering"
    assert diagnostic.pass_name == "build_heap_result_storage"
    assert diagnostic.node_type == node_type
    assert diagnostic.path == diagnostic_path
    assert _stage_values(context) == _EXPLICIT_STAGE_SEQUENCE[:7]
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(7)
    )
    assert all(record.duration_ns >= 0 for record in context.stage_run_records)
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.sequence_index for record in context.llir_pass_run_records] == list(
        range(5)
    )
    assert all(record.duration_ns >= 0 for record in context.llir_pass_run_records)
    assert later_calls == []
    assert context.compile_options is options
    assert options.requested_schedule is schedule
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert schedule == schedule_snapshot
    assert (schedule.cache_key, hash(schedule)) == schedule_identity
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert ops._kernel_cache == {}
    assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.LLIR_TO_CPP_GENERATION,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"
    assert CompilerStageId.SCHEDULE_LOWERING.value in str(suppressed.value)


def test_frontend_failure_records_nothing_and_never_enters_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _explicit_options()
    context = CompilationContext(options)
    normalization_calls: list[object] = []

    def forbidden_normalization(*args: object, **kwargs: object) -> object:
        normalization_calls.append(args)
        raise AssertionError("normalization ran after frontend failure")

    monkeypatch.setattr(ops, "normalize_cin", forbidden_normalization)
    with pytest.raises(Exception) as failure:
        ops.einsum(
            "invalid",
            *_spmm_specs(),
            compile_only=True,
            _compile_options=options,
            _compilation_context=context,
        )
    assert type(failure.value).__name__ == "CompileSpecError"
    assert context.stage_run_records == ()
    assert context.llir_pass_run_records == ()
    assert normalization_calls == []


def test_frontend_stage_failure_preserves_exact_error_and_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _explicit_options()
    context = CompilationContext(options)
    error = RuntimeError("injected frontend construction failure")
    normalization_calls: list[object] = []
    monkeypatch.setattr(STensor, "from_torch", _raise_same(error))
    monkeypatch.setattr(
        ops,
        "normalize_cin",
        lambda *args, **kwargs: normalization_calls.append(args),
    )

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            torch.ones(2, 3),
            TensorSpec("dd", (3, 4), name="B"),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert context.stage_run_records == ()
    assert context.llir_pass_run_records == ()
    assert normalization_calls == []
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


@pytest.mark.parametrize("failure_site", ("selection", "binding"))
def test_prealignment_scheduling_failure_keeps_only_completed_frontend(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    options = _explicit_options()
    context = CompilationContext(options)
    error = RuntimeError(f"injected prealignment {failure_site} failure")
    if failure_site == "selection":
        monkeypatch.setattr(
            Scheduler,
            "resolve_loop_order",
            staticmethod(_raise_same(error)),
        )
    else:
        monkeypatch.setattr(
            ops,
            "_bind_frontend_operand_mode_orders",
            _raise_same(error),
        )

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert _stage_values(context) == [
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value
    ]
    assert context.llir_pass_run_records == ()
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_native_load_failure_is_outside_the_completed_compiler_stage_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _explicit_options()
    context = CompilationContext(options)
    error = RuntimeError("injected native/cache failure")
    prepared_builds: list[utils._PreparedJITBuild] = []

    def fail_at_native_boundary(prepared: utils._PreparedJITBuild) -> object:
        assert _stage_values(context) == _EXPLICIT_STAGE_SEQUENCE
        assert utils._validate_prepared_jit_build(prepared) is prepared
        prepared_builds.append(prepared)
        raise error

    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", fail_at_native_boundary)
    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )
    assert failure.value is error
    assert _stage_values(context) == _EXPLICIT_STAGE_SEQUENCE
    assert len(prepared_builds) == 1
    prepared = prepared_builds[0]
    request = prepared.request
    assert request.build_options is options.build
    assert prepared.cache_key == (request.name, utils._request_cache_key(request))
    assert prepared.so_path == str(Path(request.build_directory) / f"{request.name}.so")
    assert type(request.cpp_sources) is tuple
    assert type(request.functions) is tuple
    with pytest.raises(FrozenInstanceError):
        prepared.so_path = "detached.so"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        request.name = "detached"  # type: ignore[misc]


def test_prepared_build_validation_failure_is_owned_by_kernel_request_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _explicit_options()
    context = CompilationContext(options)
    error = RuntimeError("injected prepared-build validation failure")
    loader_calls: list[object] = []
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(utils, "_validate_prepared_jit_build", _raise_same(error))
    monkeypatch.setattr(
        ops,
        "_load_validated_prepared_kernel",
        lambda prepared: loader_calls.append(prepared),
    )

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert _stage_values(context) == _EXPLICIT_STAGE_SEQUENCE[:-1]
    assert loader_calls == []
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_prepared_build_rejects_detached_request_cache_and_path_before_native_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    prepared = utils._prepare_jit_build(
        "kernel_timing_carrier",
        ["int timing_carrier;"],
        ["evaluate"],
        options.build.extra_cflags,
        options.build.extra_ldflags,
        compile_options=options,
    )
    monkeypatch.setattr(utils, "_so_cache", {})
    monkeypatch.setattr(
        utils,
        "_build_and_load_extension",
        lambda request: (_ for _ in ()).throw(
            AssertionError("invalid carrier reached native compilation")
        ),
    )
    monkeypatch.setattr(
        utils,
        "_load_extension_file",
        lambda name, path: (_ for _ in ()).throw(
            AssertionError("invalid carrier reached native loading")
        ),
    )

    with pytest.raises(ValueError, match="cache key"):
        utils._load_prepared_kernel(
            replace(prepared, cache_key=("detached", prepared.cache_key[1]))
        )
    with pytest.raises(ValueError, match="path"):
        utils._load_prepared_kernel(replace(prepared, so_path="/tmp/detached.so"))
    with pytest.raises(TypeError, match="build payload"):
        utils._load_prepared_kernel(
            replace(prepared, request=object())  # type: ignore[arg-type]
        )


def test_dynamic_failure_after_abi_keeps_assembly_and_preceding_pass_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _explicit_options()
    context = CompilationContext(options)
    error = RuntimeError("injected dynamic-vector failure")
    monkeypatch.setattr(
        llir_pass_manager,
        "rewrite_dynamic_vector_accesses",
        _raise_same(error),
    )
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert _stage_values(context) == [
        *_EINSUM_PREFIX_THROUGH_ADAPTER,
        CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    ]
    assert context.stage_run_records[-1].nested_within is CompilerStageId.CIN_LOWERING
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]


def test_malformed_assembled_call_fails_its_owner_and_suppresses_later_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _explicit_options()
    context = CompilationContext(options)
    original_assembly = ResultTensorAssembler.emit_final_assembly

    def emit_malformed_call(self: ResultTensorAssembler) -> list[llir.Stmt]:
        malformed = object.__new__(llir.FunctionCall)
        object.__setattr__(malformed, "name", "std::move")
        object.__setattr__(
            malformed,
            "args",
            [llir.Var("values", llir.DataType.STD_VECTOR_FLOAT32)],
        )
        return [
            *original_assembly(self),
            llir.VarInit(
                llir.Var("malformed_move", llir.DataType.NO_TYPE),
                malformed,
            ),
        ]

    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_final_assembly",
        emit_malformed_call,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value.diagnostic.code == "invalid_function_call_args"
    assert failure.value.diagnostic.stage == "LLIR rewrite"
    assert failure.value.diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert _stage_values(context) == [
        *_EINSUM_PREFIX_THROUGH_ADAPTER,
        CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    ]
    assert context.stage_run_records[-1].nested_within is CompilerStageId.CIN_LOWERING
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]
    assert later_calls == []
    assert context.compile_options is options
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_assembled_member_call_fails_owner_and_suppresses_later_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _explicit_options()
    context = CompilationContext(options)
    original_assembly = ResultTensorAssembler.emit_final_assembly
    malformed_indices: list[int] = []

    def emit_malformed_call(self: ResultTensorAssembler) -> list[llir.Stmt]:
        assembled = original_assembly(self)
        malformed_indices.append(len(assembled))
        malformed = object.__new__(llir.MemberCall)
        object.__setattr__(
            malformed,
            "base",
            llir.Var("values_torch", llir.DataType.TORCH_TENSOR),
        )
        object.__setattr__(malformed, "member", "data_ptr")
        object.__setattr__(malformed, "template_args", (llir.DataType.FLOAT32,))
        object.__setattr__(malformed, "args", [])
        return [
            *assembled,
            llir.VarInit(
                llir.Var("malformed_data", llir.DataType.PTR_FLOAT32),
                malformed,
            ),
        ]

    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_final_assembly",
        emit_malformed_call,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert malformed_indices and len(malformed_indices) == 1
    assert failure.value.diagnostic.code == "invalid_member_call_args"
    assert failure.value.diagnostic.path == (
        "root",
        "[31]",
        "value",
        "args",
    )
    assert failure.value.diagnostic.stage == "LLIR rewrite"
    assert failure.value.diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert _stage_values(context) == [
        *_EINSUM_PREFIX_THROUGH_ADAPTER,
        CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    ]
    assert context.stage_run_records[-1].nested_within is CompilerStageId.CIN_LOWERING
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(len(context.stage_run_records))
    )
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]
    assert [record.sequence_index for record in context.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
    ]
    assert later_calls == []
    assert context.compile_options is options
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_known_nnz_size_construction_failure_fails_result_abi_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    specs = _all_coo_sddmm_specs()
    specs_snapshot = tuple(spec.metadata for spec in specs)
    context = CompilationContext(options)
    error = RuntimeError("injected known-nnz result/ABI assembly failure")
    observed: list[tuple[str, llir.DataType, str, llir.DataType, str, int]] = []
    original_instrument = CINLowerer._instrument_body_assembler

    def instrument_failing_known_nnz(
        self: CINLowerer,
        body_assembler: cin_lowerer_module.LLIRBodyAssembler,
    ) -> cin_lowerer_module.LLIRBodyAssembler:
        def observe_then_fail(
            transformed_body: cin_lowerer_module.LLIRStatementListArtifact,
            compressed_output_parallel: bool,
        ) -> cin_lowerer_module.LLIRRewriteArtifact[list[llir.Stmt]]:
            assembled = body_assembler(
                transformed_body,
                compressed_output_parallel,
            )
            matches = [
                cast(llir.VarInit, candidate)
                for candidate in assembled.value
                if type(candidate) is llir.VarInit
                and cast(llir.VarInit, candidate).var.name == "_known_nnz"
            ]
            assert len(matches) == 1
            initializer = matches[0]
            assert type(initializer.value) is llir.MemberCall
            call = cast(llir.MemberCall, initializer.value)
            assert type(call.base) is llir.Var
            base = cast(llir.Var, call.base)
            assert call.template_args == ()
            assert len(call.args) == 1
            assert type(call.args[0]) is llir.Literal
            extent = cast(llir.Literal, call.args[0])
            observed.append(
                (
                    initializer.var.name,
                    initializer.var.type,
                    base.name,
                    base.type,
                    call.member,
                    cast(int, extent.value),
                )
            )
            raise error

        return original_instrument(self, observe_then_fail)

    monkeypatch.setattr(
        CINLowerer,
        "_instrument_body_assembler",
        instrument_failing_known_nnz,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (
                llir_pass_manager.LLIRPassManager,
                "run_dynamic_vector_access",
                "dynamic_vector",
            ),
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with scheduler_module.regblock_force(False):
        with pytest.raises(RuntimeError) as failure:
            ops.einsum(
                "ij,ik,jk->ij",
                *specs,
                compile_only=True,
                format="oo",
                _compile_options=options,
                _compilation_context=context,
            )

    assert failure.value is error
    assert observed == [
        (
            "_known_nnz",
            llir.DataType.INT64,
            "A_values",
            llir.DataType.TORCH_TENSOR,
            "size",
            0,
        )
    ]
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]
    assert [record.sequence_index for record in context.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
    ]
    assert all(record.duration_ns >= 0 for record in context.llir_pass_run_records)
    assert later_calls == []
    assert context.compile_options is options
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert ops._kernel_cache == {}
    assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_known_nnz_size_fails_dynamic_owner_and_suppresses_later_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    specs = _all_coo_sddmm_specs()
    specs_snapshot = tuple(spec.metadata for spec in specs)
    context = CompilationContext(options)
    original_dynamic = llir_pass_manager.LLIRPassManager.run_dynamic_vector_access
    injected: list[tuple[str, llir.DataType, str, llir.DataType, str, int]] = []

    def run_with_malformed_known_nnz(
        self: llir_pass_manager.LLIRPassManager,
        artifact: llir_pass_manager.LLIRRewriteArtifact[list[llir.Stmt]],
        pass_spec: llir_pass_manager.DynamicVectorAccessPassSpec,
    ) -> object:
        matches = [
            cast(llir.VarInit, candidate)
            for candidate in artifact.value
            if type(candidate) is llir.VarInit
            and cast(llir.VarInit, candidate).var.name == "_known_nnz"
        ]
        assert len(matches) == 1
        initializer = matches[0]
        assert type(initializer.value) is llir.MemberCall
        call = cast(llir.MemberCall, initializer.value)
        assert type(call.base) is llir.Var
        base = cast(llir.Var, call.base)
        assert len(call.args) == 1
        assert type(call.args[0]) is llir.Literal
        extent = cast(llir.Literal, call.args[0])
        injected.append(
            (
                initializer.var.name,
                initializer.var.type,
                base.name,
                base.type,
                call.member,
                cast(int, extent.value),
            )
        )
        object.__setattr__(call, "args", list(call.args))
        return original_dynamic(self, artifact, pass_spec)

    monkeypatch.setattr(
        llir_pass_manager.LLIRPassManager,
        "run_dynamic_vector_access",
        run_with_malformed_known_nnz,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with scheduler_module.regblock_force(False):
        with pytest.raises(LLIRTraversalError) as failure:
            ops.einsum(
                "ij,ik,jk->ij",
                *specs,
                compile_only=True,
                format="oo",
                _compile_options=options,
                _compilation_context=context,
            )

    diagnostic = failure.value.diagnostic
    assert injected == [
        (
            "_known_nnz",
            llir.DataType.INT64,
            "A_values",
            llir.DataType.TORCH_TENSOR,
            "size",
            0,
        )
    ]
    assert diagnostic.code == "invalid_member_call_args"
    assert diagnostic.path == ("root", "[22]", "value", "args")
    assert diagnostic.stage == "LLIR rewrite"
    assert diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert _stage_values(context) == [
        *_EINSUM_PREFIX_THROUGH_ADAPTER,
        CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    ]
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(len(context.stage_run_records))
    )
    assert all(record.duration_ns >= 0 for record in context.stage_run_records)
    assert context.stage_run_records[-1].nested_within is CompilerStageId.CIN_LOWERING
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]
    assert [record.sequence_index for record in context.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
    ]
    assert all(record.duration_ns >= 0 for record in context.llir_pass_run_records)
    assert later_calls == []
    assert context.compile_options is options
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert ops._kernel_cache == {}
    assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_tiled_stack_decl_fails_first_pass_and_suppresses_later_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule(
        loop_order=("i", "k", "j"),
        tiles=(
            TileSpec(
                index_var="j",
                width=4,
                placement="child_of:i",
                accum="stack",
            ),
        ),
        tag="malformed-tiled-stack-declaration",
    )
    options = _default_options(requested_schedule=schedule)
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    schedule_identity = (schedule.cache_key, hash(schedule))
    schedule_snapshot = replace(schedule)
    specs = _spmm_specs()
    specs_snapshot = tuple(spec.metadata for spec in specs)
    context = CompilationContext(options)
    original_lower_where = CINLowerer.lower_Where
    injected_declarations: list[tuple[str, str, llir.DataType]] = []

    def lower_with_malformed_declaration(
        self: CINLowerer,
        statement: Where,
    ) -> llir.Stmt | list[llir.Stmt]:
        lowered = original_lower_where(self, statement)
        declarations: list[llir.FixedStackArrayDecl] = []

        class StackDeclarationCollector(LLIRWalker):
            def visit_fixed_stack_array_decl(
                self,
                node: llir.FixedStackArrayDecl,
                path: tuple[str, ...],
            ) -> None:
                if node.name == "wksp":
                    declarations.append(node)
                super().visit_fixed_stack_array_decl(node, path)

        StackDeclarationCollector(
            LLIRTraversalContext(
                stage="test",
                pass_name="find_tiled_stack_declaration",
            )
        ).walk(lowered)
        for declaration in declarations:
            extent = cast(llir.Var, declaration.extent)
            injected_declarations.append((declaration.name, extent.name, extent.type))
            object.__setattr__(
                declaration,
                "extent",
                llir.Var("runtime_extent", llir.DataType.INT),
            )
        return lowered

    monkeypatch.setattr(
        CINLowerer,
        "lower_Where",
        lower_with_malformed_declaration,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (ResultTensorAssembler, "emit_final_assembly", "result_abi"),
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *specs,
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    diagnostic = failure.value.diagnostic
    assert injected_declarations == [
        ("wksp", "kTile_j", llir.DataType.CONSTEXPR_INT),
    ]
    assert diagnostic.code == "invalid_fixed_stack_array_extent"
    assert diagnostic.path == (
        "root",
        "[1]",
        "body",
        "[5]",
        "body",
        "[1]",
        "extent",
    )
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "insert_sparse_prefetch"
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(len(context.stage_run_records))
    )
    assert all(record.duration_ns >= 0 for record in context.stage_run_records)
    assert context.llir_pass_run_records == ()
    assert later_calls == []
    assert context.compile_options is options
    assert options.requested_schedule is schedule
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert schedule == schedule_snapshot
    assert (schedule.cache_key, hash(schedule)) == schedule_identity
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert ops._kernel_cache == {}
    assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


@pytest.mark.parametrize("route", ("dense_capacity", "known_nnz"))
def test_malformed_torch_empty_extent_fails_dynamic_owner_and_suppresses_later_stages(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
) -> None:
    specs: Tuple[TensorSpec, ...]
    if route == "dense_capacity":
        equation = "ik,kj->ij"
        specs = _spmm_specs()
        output_format = "dd"
        options = _explicit_options()
        expected_extent = "C_capacity"
    else:
        equation = "ij,ik,jk->ij"
        specs = _all_coo_sddmm_specs()
        output_format = "oo"
        options = _default_options()
        expected_extent = "_known_nnz"
    context = CompilationContext(options)
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    specs_snapshot = tuple(spec.metadata for spec in specs)
    original_initialization = ResultTensorAssembler.emit_value_array_init
    injected_extents: list[str] = []

    def emit_malformed_extent(self: ResultTensorAssembler) -> list[llir.Stmt]:
        statements = original_initialization(self)
        for statement in statements:
            if type(statement) is not llir.VarInit:
                continue
            initializer = cast(llir.VarInit, statement)
            if type(initializer.value) is not llir.FunctionCall:
                continue
            call = cast(llir.FunctionCall, initializer.value)
            if call.name != "torch::empty" or type(call.args[0]) is not llir.Array:
                continue
            extent = cast(llir.Array, call.args[0])
            child = cast(llir.Var, extent.values[0])
            injected_extents.append(child.name)
            object.__setattr__(extent, "data_type", "int64_t")
        return statements

    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_value_array_init",
        emit_malformed_extent,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    if route == "known_nnz":
        with scheduler_module.regblock_force(False):
            with pytest.raises(LLIRTraversalError) as failure:
                ops.einsum(
                    equation,
                    *specs,
                    compile_only=True,
                    format=output_format,
                    _compile_options=options,
                    _compilation_context=context,
                )
    else:
        with pytest.raises(LLIRTraversalError) as failure:
            ops.einsum(
                equation,
                *specs,
                compile_only=True,
                format=output_format,
                _compile_options=options,
                _compilation_context=context,
            )

    diagnostic = failure.value.diagnostic
    assert injected_extents == [expected_extent]
    assert diagnostic.code == "invalid_array_data_type"
    assert diagnostic.stage == "LLIR rewrite"
    assert diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert diagnostic.path[-4:] == ("value", "args", "[0]", "data_type")
    assert _stage_values(context) == [
        *_EINSUM_PREFIX_THROUGH_ADAPTER,
        CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    ]
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(6)
    )
    assert all(record.duration_ns >= 0 for record in context.stage_run_records)
    assert context.stage_run_records[-1].nested_within is CompilerStageId.CIN_LOWERING
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]
    assert [record.sequence_index for record in context.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
    ]
    assert all(record.duration_ns >= 0 for record in context.llir_pass_run_records)
    assert later_calls == []
    assert context.compile_options is options
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert ops._kernel_cache == {}
    assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


@pytest.mark.parametrize(
    ("malformation", "diagnostic_code", "node_type", "diagnostic_path"),
    (
        (
            "field",
            "invalid_direct_init_var_type",
            "DataType",
            ("root", "[13]", "var", "type"),
        ),
        (
            "subclass",
            "unknown_llir_node",
            "UnknownResultPositionInit",
            ("root", "[13]"),
        ),
    ),
)
def test_malformed_fixed_result_position_owner_fails_result_abi_and_stops_later_work(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    diagnostic_code: str,
    node_type: str,
    diagnostic_path: tuple[str, ...],
) -> None:
    specs = (TensorSpec("oo", (2, 3), name="A"),)
    specs_snapshot = tuple(spec.metadata for spec in specs)
    options = _default_options()
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    context = CompilationContext(options)
    original_initialization = ResultTensorAssembler.emit_level_indices_init
    injected: list[tuple[str, llir.DataType]] = []

    class UnknownResultPositionInit(llir.DirectInit):
        pass

    def emit_malformed_result_position(
        self: ResultTensorAssembler,
    ) -> list[llir.Stmt]:
        statements = original_initialization(self)
        for index, statement in enumerate(statements):
            if type(statement) is not llir.DirectInit:
                continue
            declaration = cast(llir.DirectInit, statement)
            if not declaration.var.name.endswith("_pos"):
                continue
            injected.append((declaration.var.name, declaration.var.type))
            if malformation == "field":
                declaration.var.type = llir.DataType.VOID
            else:
                statements[index] = UnknownResultPositionInit(
                    declaration.var,
                    declaration.args,
                )
            break
        return statements

    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_level_indices_init",
        emit_malformed_result_position,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ij->ij",
            *specs,
            compile_only=True,
            format="ds",
            _compile_options=options,
            _compilation_context=context,
        )

    diagnostic = failure.value.diagnostic
    assert injected == [("B1_pos", llir.DataType.STD_VECTOR_C_INT)]
    assert diagnostic.code == diagnostic_code
    assert diagnostic.stage == "LLIR rewrite"
    assert diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert diagnostic.node_type == node_type
    assert diagnostic.path == diagnostic_path
    assert _stage_values(context) == [
        *_EINSUM_PREFIX_THROUGH_ADAPTER,
        CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    ]
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(6)
    )
    assert all(record.duration_ns >= 0 for record in context.stage_run_records)
    assert context.stage_run_records[-1].nested_within is CompilerStageId.CIN_LOWERING
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]
    assert later_calls == []
    assert context.compile_options is options
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert ops._kernel_cache == {}
    assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


@pytest.mark.parametrize("malformation", ("forged_field", "unknown_subclass"))
def test_malformed_known_nnz_coordinate_owner_fails_result_abi_and_stops_later_work(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    specs = _all_coo_sddmm_specs()
    specs_snapshot = tuple(spec.metadata for spec in specs)
    options = _default_options()
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    context = CompilationContext(options)
    original_initialization = ResultTensorAssembler.emit_level_indices_init
    injected: list[tuple[str, str, llir.DataType]] = []

    class UnknownCoordinateExtent(llir.Var):
        pass

    def emit_malformed_coordinate_owner(
        self: ResultTensorAssembler,
    ) -> list[llir.Stmt]:
        statements = original_initialization(self)
        if self.known_nnz_var is None:
            return statements
        for statement in statements:
            if type(statement) is not llir.VarInit:
                continue
            initializer = cast(llir.VarInit, statement)
            if not initializer.var.name.endswith("_crd_torch"):
                continue
            if type(initializer.value) is not llir.FunctionCall:
                continue
            call = cast(llir.FunctionCall, initializer.value)
            if call.name != "torch::empty" or type(call.args[0]) is not llir.Array:
                continue
            extent = cast(llir.Array, call.args[0])
            child = cast(llir.Var, extent.values[0])
            injected.append((initializer.var.name, child.name, child.type))
            if malformation == "forged_field":
                object.__setattr__(extent, "data_type", "int64_t")
            else:
                object.__setattr__(
                    extent,
                    "values",
                    (
                        UnknownCoordinateExtent(
                            child.name,
                            child.type,
                        ),
                    ),
                )
            break
        return statements

    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_level_indices_init",
        emit_malformed_coordinate_owner,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with scheduler_module.regblock_force(False):
        with pytest.raises(LLIRTraversalError) as failure:
            ops.einsum(
                "ij,ik,jk->ij",
                *specs,
                compile_only=True,
                format="oo",
                _compile_options=options,
                _compilation_context=context,
            )

    diagnostic = failure.value.diagnostic
    assert injected == [
        ("D0_crd_torch", "_known_nnz", llir.DataType.INT64),
    ]
    if malformation == "forged_field":
        assert diagnostic.code == "invalid_array_data_type"
        assert diagnostic.path[-4:] == ("value", "args", "[0]", "data_type")
    else:
        assert diagnostic.code == "unknown_llir_node"
        assert diagnostic.node_type == "UnknownCoordinateExtent"
        assert diagnostic.path[-5:] == (
            "value",
            "args",
            "[0]",
            "values",
            "[0]",
        )
    assert diagnostic.stage == "LLIR rewrite"
    assert diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert _stage_values(context) == [
        *_EINSUM_PREFIX_THROUGH_ADAPTER,
        CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    ]
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(6)
    )
    assert all(record.duration_ns >= 0 for record in context.stage_run_records)
    assert context.stage_run_records[-1].nested_within is CompilerStageId.CIN_LOWERING
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]
    assert [record.sequence_index for record in context.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
    ]
    assert all(record.duration_ns >= 0 for record in context.llir_pass_run_records)
    assert later_calls == []
    assert context.compile_options is options
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert ops._kernel_cache == {}
    assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_final_qualified_name_fails_dynamic_owner_and_suppresses_later_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = _spmm_specs()
    options = _explicit_options()
    context = CompilationContext(options)
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    specs_snapshot = tuple(spec.metadata for spec in specs)
    original_initialization = ResultTensorAssembler.emit_value_array_init
    injected_names: list[tuple[str, str, llir.DataType]] = []

    def emit_malformed_dtype(self: ResultTensorAssembler) -> list[llir.Stmt]:
        statements = original_initialization(self)
        for statement in statements:
            if type(statement) is not llir.VarInit:
                continue
            initializer = cast(llir.VarInit, statement)
            if type(initializer.value) is not llir.FunctionCall:
                continue
            call = cast(llir.FunctionCall, initializer.value)
            if (
                call.name != "torch::empty"
                or type(call.args[1]) is not llir.QualifiedName
            ):
                continue
            dtype_name = cast(llir.QualifiedName, call.args[1])
            injected_names.append(
                (dtype_name.namespace, dtype_name.name, dtype_name.data_type)
            )
            malformed = object.__new__(llir.QualifiedName)
            object.__setattr__(malformed, "namespace", "torch::detail")
            object.__setattr__(malformed, "name", dtype_name.name)
            object.__setattr__(malformed, "data_type", dtype_name.data_type)
            initializer.value = llir.FunctionCall(
                name=call.name,
                args=(call.args[0], malformed),
            )
        return statements

    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_value_array_init",
        emit_malformed_dtype,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *specs,
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    diagnostic = failure.value.diagnostic
    assert injected_names == [
        ("torch", "kFloat32", llir.DataType.TORCH_SCALAR_TYPE),
    ]
    assert diagnostic.code == "invalid_qualified_name_namespace"
    assert diagnostic.stage == "LLIR rewrite"
    assert diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert diagnostic.path[-4:] == ("value", "args", "[1]", "namespace")
    assert _stage_values(context) == [
        *_EINSUM_PREFIX_THROUGH_ADAPTER,
        CompilerStageId.RESULT_ABI_ASSEMBLY.value,
    ]
    assert [record.sequence_index for record in context.stage_run_records] == list(
        range(6)
    )
    assert all(record.duration_ns >= 0 for record in context.stage_run_records)
    assert context.stage_run_records[-1].nested_within is CompilerStageId.CIN_LOWERING
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]
    assert [record.sequence_index for record in context.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
    ]
    assert all(record.duration_ns >= 0 for record in context.llir_pass_run_records)
    assert later_calls == []
    assert context.compile_options is options
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert ops._kernel_cache == {}
    assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_intermediate_qualified_name_fails_first_managed_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule(
        loop_order=("q", "r", "c"),
        tag="malformed-intermediate-qualified-name",
    )
    specs = (
        TensorSpec(
            "ds",
            (4, 5),
            name="SparseLeft",
            mode_order=(1, 0),
        ),
        TensorSpec("ds", (5, 3), name="SparseRight"),
    )
    options = _default_options(requested_schedule=schedule)
    context = CompilationContext(options)
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    schedule_identity = (schedule.cache_key, hash(schedule))
    specs_snapshot = tuple(spec.metadata for spec in specs)
    original_lower_outer_consumer = CINLowerer.lower_outer_ConsumerIndexStmt
    injected_names: list[tuple[str, str, llir.DataType]] = []

    def lower_with_malformed_dtype(
        self: CINLowerer,
        statement: IndexStmt,
    ) -> list[llir.Stmt]:
        lowered = original_lower_outer_consumer(self, statement)

        def inject(value: object) -> None:
            if injected_names:
                return
            if type(value) is llir.QualifiedName:
                qualified_name = cast(llir.QualifiedName, value)
                injected_names.append(
                    (
                        qualified_name.namespace,
                        qualified_name.name,
                        qualified_name.data_type,
                    )
                )
                object.__setattr__(qualified_name, "namespace", "torch::detail")
                return
            if isinstance(value, llir.Node):
                for child in vars(value).values():
                    inject(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    inject(child)

        inject(lowered)
        return lowered

    monkeypatch.setattr(
        CINLowerer,
        "lower_outer_ConsumerIndexStmt",
        lower_with_malformed_dtype,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (ResultTensorAssembler, "emit_final_assembly", "result_abi"),
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "rq,qc->rc",
            *specs,
            compile_only=True,
            format="ds",
            _compile_options=options,
            _compilation_context=context,
        )

    diagnostic = failure.value.diagnostic
    assert injected_names == [
        ("torch", "kInt", llir.DataType.TORCH_SCALAR_TYPE),
    ]
    assert diagnostic.code == "invalid_qualified_name_namespace"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "insert_sparse_prefetch"
    assert diagnostic.path[-1] == "namespace"
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert context.llir_pass_run_records == ()
    assert later_calls == []
    assert context.compile_options is options
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert (schedule.cache_key, hash(schedule)) == schedule_identity
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert ops._kernel_cache == {}
    assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_tiled_workspace_read_fails_cin_lowering_and_stops_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule(
        loop_order=("i", "k", "j"),
        tiles=(
            TileSpec(
                index_var="j",
                width=4,
                placement="child_of:i",
                accum="stack",
            ),
        ),
        tag="malformed-tiled-workspace-read",
    )
    options = _default_options(requested_schedule=schedule)
    context = CompilationContext(options)
    original_lower_consumer = CINLowerer.lower_ConsumerIndexStmt
    injected_reads: list[llir.Assign] = []

    def lower_with_malformed_workspace_read(
        self: CINLowerer,
        statement: IndexStmt,
    ) -> list[llir.Stmt]:
        lowered = original_lower_consumer(self, statement)
        matches: list[llir.Assign] = []

        class WorkspaceReadCollector(LLIRWalker):
            def visit_assign(
                self,
                node: llir.Assign,
                path: tuple[str, ...],
            ) -> None:
                if type(node.value) is llir.ArrayAccess:
                    value = cast(llir.ArrayAccess, node.value)
                    if (
                        type(value.array) is llir.Var
                        and cast(llir.Var, value.array).name == "wksp"
                    ):
                        matches.append(node)
                super().visit_assign(node, path)

        WorkspaceReadCollector(
            LLIRTraversalContext(
                stage="test",
                pass_name="find_tiled_workspace_read",
            )
        ).walk(lowered)
        for assignment in matches:
            original = cast(llir.ArrayAccess, assignment.value)
            malformed = object.__new__(llir.ArrayAccess)
            object.__setattr__(malformed, "array", original.array)
            object.__setattr__(malformed, "index", original.index)
            object.__setattr__(malformed, "tensor_access", object())
            assignment.value = malformed
            injected_reads.append(assignment)
        return lowered

    monkeypatch.setattr(
        CINLowerer,
        "lower_ConsumerIndexStmt",
        lower_with_malformed_workspace_read,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert len(injected_reads) == 1
    assert failure.value.diagnostic.code == "invalid_tensor_access_metadata"
    assert failure.value.diagnostic.stage == "LLIR transformation"
    assert failure.value.diagnostic.pass_name == "insert_sparse_prefetch"
    assert context.compile_options is options
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert context.llir_pass_run_records == ()
    assert later_calls == []
    assert context.compile_options is options
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_all_coo_coordinate_read_fails_cin_lowering_and_stops_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    original_transform = CINLowerer._transform_coo_loop_for_openmp
    injected_reads: list[llir.VarInit] = []

    def transform_with_malformed_coordinate_read(
        self: CINLowerer,
        statements: list[llir.Stmt],
    ) -> list[llir.Stmt]:
        transformed = original_transform(self, statements)

        def inject(value: object) -> None:
            if type(value) is llir.VarInit:
                initializer = cast(llir.VarInit, value)
                if type(initializer.value) is llir.ArrayAccess:
                    access = cast(llir.ArrayAccess, initializer.value)
                    if (
                        type(access.array) is llir.Var
                        and cast(llir.Var, access.array).name.endswith("0_crd")
                        and cast(llir.Var, access.array).type is llir.DataType.PTR_INT
                        and type(access.index) is llir.Var
                        and cast(llir.Var, access.index).type is llir.DataType.INT64
                    ):
                        malformed = object.__new__(llir.ArrayAccess)
                        object.__setattr__(malformed, "array", access.array)
                        object.__setattr__(malformed, "index", access.index)
                        object.__setattr__(malformed, "tensor_access", object())
                        initializer.value = malformed
                        injected_reads.append(initializer)
                        return
            if isinstance(value, llir.Node):
                for child in vars(value).values():
                    inject(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    inject(child)

        inject(transformed)
        return transformed

    monkeypatch.setattr(
        CINLowerer,
        "_transform_coo_loop_for_openmp",
        transform_with_malformed_coordinate_read,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with scheduler_module.regblock_force(False):
        with pytest.raises(LLIRTraversalError) as failure:
            ops.einsum(
                "ij,ik,jk->ij",
                *_all_coo_sddmm_specs(),
                compile_only=True,
                format="oo",
                _compile_options=options,
                _compilation_context=context,
            )

    assert len(injected_reads) == 1
    assert failure.value.diagnostic.code == "invalid_tensor_access_metadata"
    assert failure.value.diagnostic.stage == "LLIR transformation"
    assert failure.value.diagnostic.pass_name == "insert_sparse_prefetch"
    assert context.compile_options is options
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert context.llir_pass_run_records == ()
    assert later_calls == []
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_all_coo_end_bound_fails_cin_lowering_and_stops_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    original_transform = CINLowerer._transform_coo_loop_for_openmp
    injected_bounds: list[llir.Add] = []

    def transform_with_malformed_end_bound(
        self: CINLowerer,
        statements: list[llir.Stmt],
    ) -> list[llir.Stmt]:
        transformed = original_transform(self, statements)

        def inject(value: object) -> None:
            if type(value) is llir.VarInit:
                initializer = cast(llir.VarInit, value)
                if type(initializer.value) is llir.Add:
                    arithmetic = cast(llir.Add, initializer.value)
                    left = arithmetic.left
                    right = arithmetic.right
                else:
                    arithmetic = None
                    left = None
                    right = None
                if (
                    initializer.var.name.endswith("1_end")
                    and arithmetic is not None
                    and type(left) is llir.Var
                    and cast(llir.Var, left).type is llir.DataType.INT64
                    and type(right) is llir.Literal
                    and cast(llir.Literal, right).value == 1
                    and cast(llir.Literal, right).data_type is llir.DataType.INT64
                ):
                    object.__setattr__(arithmetic, "op", "-")
                    injected_bounds.append(arithmetic)
                    return
            if isinstance(value, llir.Node):
                for child in vars(value).values():
                    inject(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    inject(child)

        inject(transformed)
        return transformed

    monkeypatch.setattr(
        CINLowerer,
        "_transform_coo_loop_for_openmp",
        transform_with_malformed_end_bound,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with scheduler_module.regblock_force(False):
        with pytest.raises(LLIRTraversalError) as failure:
            ops.einsum(
                "ij,ik,jk->ij",
                *_all_coo_sddmm_specs(),
                compile_only=True,
                format="oo",
                _compile_options=options,
                _compilation_context=context,
            )

    assert len(injected_bounds) == 1
    assert failure.value.diagnostic.code == "invalid_add_operator"
    assert failure.value.diagnostic.stage == "LLIR transformation"
    assert failure.value.diagnostic.pass_name == "insert_sparse_prefetch"
    assert context.compile_options is options
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert context.llir_pass_run_records == ()
    assert later_calls == []
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_mode_iterator_coordinate_read_stops_all_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    original_post_init = ModeIterator.__post_init__
    injected_reads: list[llir.ArrayAccess] = []

    def post_init_with_malformed_coordinate_read(self: ModeIterator) -> None:
        original_post_init(self)
        value = self.coord_var_value_llir
        if (
            type(value) is llir.ArrayAccess
            and type(value.array) is llir.Var
            and value.array.name == "A1_crd"
            and value.array.type is llir.DataType.PTR_INT
            and type(value.index) is llir.Var
            and value.index.type is llir.DataType.INT
        ):
            malformed = object.__new__(llir.ArrayAccess)
            object.__setattr__(malformed, "array", value.array)
            object.__setattr__(malformed, "index", value.index)
            object.__setattr__(malformed, "tensor_access", object())
            self.coord_var_value_llir = malformed
            injected_reads.append(malformed)

    monkeypatch.setattr(
        ModeIterator,
        "__post_init__",
        post_init_with_malformed_coordinate_read,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert len(injected_reads) == 1
    assert failure.value.diagnostic.code == "invalid_tensor_access_metadata"
    assert failure.value.diagnostic.stage == "LLIR transformation"
    assert failure.value.diagnostic.pass_name == "insert_sparse_prefetch"
    assert context.compile_options is options
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert context.llir_pass_run_records == ()
    assert later_calls == []
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_mode_iterator_position_bound_stops_all_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    original_post_init = ModeIterator.__post_init__
    injected_bounds: list[llir.Add] = []

    def post_init_with_malformed_position_bound(self: ModeIterator) -> None:
        original_post_init(self)
        value = self.iterator_var_end_value_llir
        if (
            type(value) is llir.ArrayAccess
            and type(value.array) is llir.Var
            and value.array.name == "A1_pos"
            and value.array.type is llir.DataType.PTR_INT
            and type(value.index) is llir.Add
        ):
            malformed = cast(llir.Add, value.index)
            object.__setattr__(malformed, "op", "-")
            injected_bounds.append(malformed)

    monkeypatch.setattr(
        ModeIterator,
        "__post_init__",
        post_init_with_malformed_position_bound,
    )
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _compilation_context=context,
        )

    assert len(injected_bounds) == 1
    assert failure.value.diagnostic.code == "invalid_add_operator"
    assert failure.value.diagnostic.stage == "LLIR transformation"
    assert failure.value.diagnostic.pass_name == "insert_sparse_prefetch"
    assert context.compile_options is options
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert context.llir_pass_run_records == ()
    assert later_calls == []
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_malformed_compressed_base_load_fails_owner_and_stops_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    options_identity = (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    )
    specs = (
        TensorSpec("ds", (2, 3), name="A"),
        TensorSpec("ds", (3, 4), name="B"),
    )
    specs_snapshot = tuple(spec.metadata for spec in specs)
    context = CompilationContext(options)
    injected: list[tuple[str, llir.DataType, str, llir.DataType, str]] = []

    def base_loads(value: object) -> list[llir.VarInit]:
        matches: list[llir.VarInit] = []
        if type(value) is llir.VarInit:
            initializer = cast(llir.VarInit, value)
            access = initializer.value
            if (
                initializer.var.name.startswith("_base")
                and type(access) is llir.ArrayAccess
                and type(cast(llir.ArrayAccess, access).array) is llir.Var
                and cast(
                    llir.Var, cast(llir.ArrayAccess, access).array
                ).name.startswith("_offset")
            ):
                matches.append(initializer)
        if isinstance(value, llir.Node):
            for child in vars(value).values():
                matches.extend(base_loads(child))
        elif isinstance(value, (list, tuple)):
            for child in value:
                matches.extend(base_loads(child))
        return matches

    class InjectingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            matches = base_loads(value)
            if matches and not injected:
                initializer = matches[0]
                access = cast(llir.ArrayAccess, initializer.value)
                array = cast(llir.Var, access.array)
                index = cast(llir.Var, access.index)
                injected.append(
                    (
                        initializer.var.name,
                        initializer.var.type,
                        array.name,
                        array.type,
                        index.name,
                    )
                )
                forged = object.__new__(llir.ArrayAccess)
                object.__setattr__(forged, "array", array)
                object.__setattr__(forged, "index", index)
                object.__setattr__(forged, "tensor_access", object())
                initializer.value = forged
            super().walk(value)

    monkeypatch.setattr(compressed_where_module, "LLIRWalker", InjectingWalker)
    later_calls: list[str] = []
    _forbid_boundaries(
        monkeypatch,
        later_calls,
        (
            (
                llir_pass_manager.LLIRPassManager,
                "run_sparse_prefetch",
                "sparse_prefetch",
            ),
            (
                llir_pass_manager.LLIRPassManager,
                "run_dense_pointer_hoist",
                "dense_pointer_hoist",
            ),
            (
                llir_pass_manager.LLIRPassManager,
                "run_single_iteration_loop_elimination",
                "single_iteration",
            ),
            (
                llir_pass_manager.LLIRPassManager,
                "run_loop_invariant_factor_hoist",
                "factor_hoist",
            ),
            (
                llir_pass_manager.LLIRPassManager,
                "run_dynamic_vector_access",
                "dynamic_vector",
            ),
            (schedule_lowerer, "apply_schedule_to_llir", "schedule_lowering"),
            (LLIRLowerer, "lower_llir", "cpp_generation"),
            (ops, "_prepare_generated_kernel_build", "build_request"),
            (ops, "_load_validated_prepared_kernel", "native_load"),
        ),
    )
    _isolate_compiler_caches(monkeypatch)

    with pytest.raises(LLIRTraversalError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *specs,
            compile_only=True,
            format="ds",
            _compile_options=options,
            _compilation_context=context,
        )

    diagnostic = failure.value.diagnostic
    assert injected == [
        (
            "_base1",
            llir.DataType.INT64,
            "_offset1",
            llir.DataType.STD_VECTOR_INT,
            "i",
        )
    ]
    assert diagnostic.code == "invalid_tensor_access_metadata"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "transform_compressed_where_for_openmp"
    assert diagnostic.path[-2:] == ("value", "tensor_access")
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert [
        (record.pass_name, record.configuration_name)
        for record in context.llir_pass_run_records
    ] == [
        ("rewrite_result_writes", "count"),
        ("rewrite_result_writes", "fill"),
    ]
    assert [record.sequence_index for record in context.llir_pass_run_records] == [
        0,
        1,
    ]
    assert all(record.duration_ns >= 0 for record in context.llir_pass_run_records)
    assert later_calls == []
    assert context.compile_options is options
    assert tuple(spec.metadata for spec in specs) == specs_snapshot
    assert (
        options.cache_key,
        options.semantic_cache_key,
        options.cache_fingerprint,
    ) == options_identity
    assert ops._kernel_cache == {}
    assert ops._einsum_dispatch_cache == {}
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.SCHEDULE_LOWERING,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


@pytest.mark.parametrize(
    ("failed_mode", "expected_configurations"),
    [("count", []), ("fill", ["count"])],
)
def test_compressed_where_partial_failures_preserve_exact_nested_pass_prefix(
    monkeypatch: pytest.MonkeyPatch,
    failed_mode: str,
    expected_configurations: list[str],
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    error = VerificationError(f"injected compressed {failed_mode} failure")
    original_rewrite = llir_pass_manager.rewrite_result_writes

    def fail_selected_mode(value: object, pass_context: object) -> object:
        if getattr(pass_context, "mode", None) == failed_mode:
            raise error
        return original_rewrite(value, pass_context)  # type: ignore[arg-type]

    monkeypatch.setattr(
        llir_pass_manager,
        "rewrite_result_writes",
        fail_selected_mode,
    )
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )
    with pytest.raises(VerificationError) as failure:
        ops.einsum(
            "ik,kj->ij",
            TensorSpec("ds", (2, 3), name="A"),
            TensorSpec("ds", (3, 4), name="B"),
            compile_only=True,
            format="ds",
            _compile_options=options,
            _compilation_context=context,
        )
    assert failure.value is error
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert [
        record.configuration_name for record in context.llir_pass_run_records
    ] == expected_configurations
    assert all(
        record.pass_name == "rewrite_result_writes"
        for record in context.llir_pass_run_records
    )


def test_compressed_where_ordinary_fill_failure_preserves_completed_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    error = RuntimeError("injected ordinary compressed fill failure")
    original_rewrite = llir_pass_manager.rewrite_result_writes

    def fail_fill(value: object, pass_context: object) -> object:
        if getattr(pass_context, "mode", None) == "fill":
            raise error
        return original_rewrite(value, pass_context)  # type: ignore[arg-type]

    monkeypatch.setattr(llir_pass_manager, "rewrite_result_writes", fail_fill)
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )
    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            TensorSpec("ds", (2, 3), name="A"),
            TensorSpec("ds", (3, 4), name="B"),
            compile_only=True,
            format="ds",
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert [
        (record.pass_name, record.configuration_name)
        for record in context.llir_pass_run_records
    ] == [("rewrite_result_writes", "count")]


def test_compressed_where_ordinary_parent_failure_preserves_count_and_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    error = RuntimeError("injected ordinary compressed parent failure")
    original_record = llir_pass_manager._record

    def fail_parent_record(*args: object, **kwargs: object) -> object:
        descriptor = kwargs.get("descriptor")
        if descriptor is llir_pass_manager.COMPRESSED_WHERE_OPENMP_PASS:
            raise error
        return original_record(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(llir_pass_manager, "_record", fail_parent_record)
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )
    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            TensorSpec("ds", (2, 3), name="A"),
            TensorSpec("ds", (3, 4), name="B"),
            compile_only=True,
            format="ds",
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert _stage_values(context) == _EINSUM_PREFIX_THROUGH_ADAPTER
    assert [
        (record.pass_name, record.configuration_name)
        for record in context.llir_pass_run_records
    ] == [
        ("rewrite_result_writes", "count"),
        ("rewrite_result_writes", "fill"),
    ]


def test_compressed_where_success_retains_count_fill_and_all_seven_pass_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )
    result = ops.einsum(
        "ik,kj->ij",
        TensorSpec("ds", (2, 3), name="A"),
        TensorSpec("ds", (3, 4), name="B"),
        compile_only=True,
        format="ds",
        _compile_options=options,
        _compilation_context=context,
    )
    assert isinstance(result, TensorSpec)
    assert [record.configuration_name for record in context.llir_pass_run_records] == [
        "compressed_where_openmp",
        "count",
        "fill",
        "sparse_prefetch",
        "dense_pointer_hoist",
        "single_iteration_loop_elimination",
        "loop_invariant_factor_hoist",
        "dynamic_vector_access",
    ]
    assert [record.sequence_index for record in context.llir_pass_run_records] == list(
        range(8)
    )


def _runtime_compiler_inputs() -> tuple[STensor, STensor, STensor]:
    sparse = scorch.STensor.from_torch(
        torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0],
            ]
        ).to_sparse_csr(),
        "A",
    )
    dense = scorch.STensor.from_torch(torch.ones(3, 4), "B")
    vector = scorch.STensor.from_torch(torch.ones(3), "v")
    return sparse, dense, vector


@pytest.mark.parametrize(
    "boundary",
    (
        "spmv",
        "lower_and_exec_cin",
        "to_dense",
        "to_sparse",
        "change_mode_order",
    ),
)
def test_manual_generated_boundaries_record_only_the_stages_they_execute(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    error = RuntimeError(f"stop {boundary} at native boundary")
    sparse, dense, vector = _runtime_compiler_inputs()
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", _raise_same(error))
    monkeypatch.setattr(
        stensor_module, "_load_validated_prepared_kernel", _raise_same(error)
    )

    with pytest.raises(RuntimeError) as failure:
        if boundary == "spmv":
            ops.spmv(
                sparse,
                vector,
                _compile_options=options,
                _compilation_context=context,
            )
        elif boundary == "lower_and_exec_cin":
            ops.lower_and_exec_cin(
                _build_spmm_cin(),
                (2, 4),
                sparse,
                dense,
                _compile_options=options,
                _compilation_context=context,
            )
        elif boundary == "to_dense":
            sparse.to_dense(
                _compile_options=options,
                _compilation_context=context,
            )
        elif boundary == "to_sparse":
            dense.copy().to_sparse(
                "ds",
                _compile_options=options,
                _compilation_context=context,
            )
        elif boundary == "change_mode_order":
            scorch.STensor.from_torch(torch.ones(2, 3, 4)).change_mode_order(
                [1, 0, 2],
                _compile_options=options,
                _compilation_context=context,
            )
        else:  # pragma: no cover - the parameter list is exhaustive
            raise AssertionError(boundary)

    assert failure.value is error
    expected = (
        _DIRECT_CIN_STAGE_SEQUENCE
        if boundary == "lower_and_exec_cin"
        else _MANUAL_STAGE_SEQUENCE
    )
    assert _stage_values(context) == expected


def test_direct_cin_runtime_binding_failure_suppresses_legacy_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    error = RuntimeError("injected direct-CIN binding failure")
    sparse, dense, _ = _runtime_compiler_inputs()
    monkeypatch.setattr(ops, "_apply_mode_order_alignment", _raise_same(error))
    monkeypatch.setattr(
        ops,
        "_load_validated_prepared_kernel",
        lambda prepared: (_ for _ in ()).throw(
            AssertionError("lowering continued after direct-CIN binding failure")
        ),
    )

    with pytest.raises(RuntimeError) as failure:
        ops.lower_and_exec_cin(
            _build_spmm_cin(),
            (2, 4),
            sparse,
            dense,
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert _stage_values(context) == [
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value,
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value,
    ]
    assert context.llir_pass_run_records == ()


def test_direct_cin_runtime_binding_plan_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    error = RuntimeError("injected direct-CIN binding-plan failure")
    sparse, dense, _ = _runtime_compiler_inputs()
    relayout_calls: list[object] = []
    monkeypatch.setattr(ops, "_plan_mode_orders_to_loop_order", _raise_same(error))
    monkeypatch.setattr(
        ops,
        "_relayout_mode_order_args",
        lambda *args, **kwargs: relayout_calls.append(args),
    )

    with pytest.raises(RuntimeError) as failure:
        ops.lower_and_exec_cin(
            _build_spmm_cin(),
            (2, 4),
            sparse,
            dense,
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert _stage_values(context) == [
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value
    ]
    assert relayout_calls == []
    assert context.llir_pass_run_records == ()
    with pytest.raises(CompilationContextError) as suppressed:
        context.begin_stage(
            CompilerStageId.LEGACY_CIN_ADAPTATION,
            compile_options=options,
        )
    assert suppressed.value.diagnostic.code == "failed_compilation"


def test_direct_cin_actual_legacy_adapter_failure_keeps_frontend_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    error = RuntimeError("injected direct-CIN legacy adapter failure")
    sparse, dense, _ = _runtime_compiler_inputs()
    monkeypatch.setattr(
        cin_lowerer_module,
        "claim_legacy_cin_working_tree",
        _raise_same(error),
    )

    with pytest.raises(RuntimeError) as failure:
        ops.lower_and_exec_cin(
            _build_spmm_cin(),
            (2, 4),
            sparse,
            dense,
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert _stage_values(context) == [
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value,
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value,
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value,
    ]
    assert context.llir_pass_run_records == ()


def test_direct_cin_normalization_completes_before_nested_relayout_native_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    result_var = TensorVar("C", fmt="ddd")
    input_var = TensorVar("B", fmt="ddd")
    cin = ForAll(
        i,
        ForAll(
            j,
            ForAll(k, TensorAssign(result_var[i, j, k], input_var[i, j, k])),
        ),
    )
    options = _default_options()
    context = CompilationContext(options)
    error = RuntimeError("stop direct-CIN nested relayout")
    observed_at_native: list[list[str]] = []

    def stop_nested_relayout(prepared: object) -> object:
        observed_at_native.append(_stage_values(context))
        raise error

    monkeypatch.setattr(
        stensor_module, "_load_validated_prepared_kernel", stop_nested_relayout
    )
    runtime = scorch.STensor.from_torch(
        torch.ones(3, 2, 4),
        mode_order=[1, 0, 2],
    )

    with pytest.raises(RuntimeError) as failure:
        ops.lower_and_exec_cin(
            cin,
            (3, 2, 4),
            runtime,
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert observed_at_native == [
        [
            CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION.value,
            CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION.value,
            *_MANUAL_STAGE_SEQUENCE,
        ]
    ]
    assert _stage_values(context) == observed_at_native[0]


def test_stensor_add_constructs_one_owner_and_routes_one_options_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    contexts: list[CompilationContext] = []
    snapshot_calls: list[object] = []
    error = RuntimeError("stop STensor.__add__ at native boundary")
    original_post_init = CompilationContext.__post_init__

    def capture_context(context: CompilationContext) -> None:
        original_post_init(context)
        contexts.append(context)

    def snapshot_once(
        cls: type[CompileOptions], *args: object, **kwargs: object
    ) -> CompileOptions:
        snapshot_calls.append((args, kwargs))
        return options

    monkeypatch.setattr(CompilationContext, "__post_init__", capture_context)
    monkeypatch.setattr(
        CompileOptions,
        "from_environment",
        classmethod(snapshot_once),
    )
    monkeypatch.setattr(
        stensor_module, "_load_validated_prepared_kernel", _raise_same(error)
    )
    left = scorch.STensor.from_torch(torch.ones(2, 2), "left")
    right = scorch.STensor.from_torch(torch.ones(2, 2), "right")

    with pytest.raises(RuntimeError) as failure:
        left + right

    assert failure.value is error
    assert len(snapshot_calls) == 1
    assert len(contexts) == 1
    assert contexts[0].compile_options is options
    assert _stage_values(contexts[0]) == _MANUAL_STAGE_SEQUENCE


def test_matmul_routes_one_owner_through_the_auto_einsum_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    context = CompilationContext(options)
    error = RuntimeError("stop nested matmul compiler path")
    sparse, dense, _ = _runtime_compiler_inputs()
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", _raise_same(error))

    with pytest.raises(RuntimeError) as failure:
        ops.matmul(
            sparse,
            dense,
            format="dd",
            use_cache=False,
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    assert context.compile_options is options
    assert _stage_values(context) == _AUTO_STAGE_SEQUENCE


def test_dispatch_cache_hit_records_no_compiler_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    sparse, dense, _ = _runtime_compiler_inputs()
    fake_result = SimpleNamespace(
        storage=SimpleNamespace(
            index=SimpleNamespace(mode_indices=[[], []]),
            value=torch.ones(8),
        )
    )
    fake_module = SimpleNamespace(evaluate=lambda *args: fake_result)
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: fake_module
    )

    first_context = CompilationContext(options)
    first = ops.einsum(
        "ik,kj->ij",
        sparse,
        dense,
        format="dd",
        _compile_options=options,
        _compilation_context=first_context,
    )
    assert isinstance(first, STensor)
    assert _stage_values(first_context) == _AUTO_STAGE_SEQUENCE

    dispatch_hit_context = CompilationContext(options)
    second = ops.einsum(
        "ik,kj->ij",
        sparse,
        dense,
        format="dd",
        _compile_options=options,
        _compilation_context=dispatch_hit_context,
    )
    assert isinstance(second, STensor)
    assert dispatch_hit_context.stage_run_records == ()
    assert dispatch_hit_context.llir_pass_run_records == ()


def test_prebuilt_sddmm_rank_one_sparse_and_mode_order_noops_record_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scorch_ops  # type: ignore[import-not-found]

    options = _default_options()
    mask = scorch.STensor.from_torch(torch.eye(2).to_sparse_coo(), "S")
    left = scorch.STensor.from_torch(torch.ones(2, 2), "A")
    right = scorch.STensor.from_torch(torch.ones(2, 2), "B")
    fake_sddmm_result = SimpleNamespace(
        storage=SimpleNamespace(
            index=SimpleNamespace(mode_indices=mask._native_mode_indices()),
            value=torch.ones_like(mask.values),
        )
    )
    monkeypatch.setattr(
        scorch_ops,
        "sddmm_coo_float_prebuilt",
        lambda *args: fake_sddmm_result,
    )
    prebuilt_context = CompilationContext(options)

    result = ops.einsum(
        "ij,ik,jk->ij",
        mask,
        left,
        right,
        _compile_options=options,
        _compilation_context=prebuilt_context,
    )

    assert isinstance(result, STensor)
    assert prebuilt_context.stage_run_records == ()

    rank_one_context = CompilationContext(options)
    vector = scorch.STensor.from_torch(torch.tensor([0.0, 1.0, 0.0]))
    vector.to_sparse(
        _compile_options=options,
        _compilation_context=rank_one_context,
    )
    assert rank_one_context.stage_run_records == ()

    mode_noop_context = CompilationContext(options)
    dense = scorch.STensor.from_torch(torch.ones(2, 2))
    dense.change_mode_order(
        [0, 1],
        _compile_options=options,
        _compilation_context=mode_noop_context,
    )
    dense.change_mode_order(
        [1, 0],
        _compile_options=options,
        _compilation_context=mode_noop_context,
    )
    assert mode_noop_context.stage_run_records == ()


def test_kernel_cache_prebuilt_and_noop_paths_do_not_fabricate_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _explicit_options()
    _isolate_compiler_caches(monkeypatch)
    monkeypatch.setattr(
        ops, "_load_validated_prepared_kernel", lambda prepared: object()
    )

    first_context = CompilationContext(options)
    first = ops.einsum(
        "ik,kj->ij",
        *_spmm_specs(),
        compile_only=True,
        format="dd",
        _compile_options=options,
        _compilation_context=first_context,
    )
    assert isinstance(first, TensorSpec)
    assert _stage_values(first_context) == _EXPLICIT_STAGE_SEQUENCE

    kernel_hit_context = CompilationContext(options)
    second = ops.einsum(
        "ik,kj->ij",
        *_spmm_specs(),
        compile_only=True,
        format="dd",
        _compile_options=options,
        _compilation_context=kernel_hit_context,
    )
    assert isinstance(second, TensorSpec)
    assert _stage_values(kernel_hit_context) == _EXPLICIT_STAGE_SEQUENCE[:4]
    assert kernel_hit_context.llir_pass_run_records == ()

    sparse = scorch.STensor.from_torch(torch.eye(2).to_sparse_csr())
    dense = scorch.STensor.from_torch(torch.ones(2, 2))
    fake_result = SimpleNamespace(
        storage=SimpleNamespace(value=torch.arange(4, dtype=torch.float32))
    )
    resolved = SimpleNamespace(
        fn=object(),
        output_format=parse_format("dd"),
        symbol_name="test_prebuilt",
    )
    monkeypatch.setattr(
        ops,
        "resolve_prebuilt_matmul",
        lambda *args, **kwargs: resolved,
    )
    monkeypatch.setattr(
        ops,
        "execute_prebuilt_binary_kernel",
        lambda *args, **kwargs: (fake_result, (2, 2)),
    )
    prebuilt_options = _default_options()
    prebuilt_context = CompilationContext(prebuilt_options)
    output = ops.matmul(
        sparse,
        dense,
        _compile_options=prebuilt_options,
        _compilation_context=prebuilt_context,
    )
    assert tuple(output.shape) == (2, 2)
    assert prebuilt_context.stage_run_records == ()

    dense_tensor = scorch.STensor.from_torch(torch.ones(2, 2))
    noop_context = CompilationContext(options)
    copy = dense_tensor.to_dense(
        _compile_options=options,
        _compilation_context=noop_context,
    )
    assert copy is not dense_tensor
    assert noop_context.stage_run_records == ()


def test_direct_analysis_scheduler_lowerer_renderer_and_loader_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    source = _build_spmm_cin()
    source_before = canonical_cin_dump(source)
    normalized = normalize_cin(source, compile_options=options)
    scheduled = Scheduler.auto_schedule(source, compile_options=options)
    lowerer = CINLowerer(compile_options=options)
    lowered = lowerer.lower_IndexStmt(scheduled)
    cpp = LLIRLowerer(compile_options=options).lower_llir(lowered)
    assert canonical_cin_dump(source) == source_before
    assert normalized is not source
    assert lowerer.llir_pass_run_records
    assert cpp == LLIRLowerer().lower_llir(lowered)

    loaded = object()
    prepared: list[object] = []

    def standalone_load(request: object) -> object:
        prepared.append(request)
        return loaded

    monkeypatch.setattr(utils, "_load_validated_prepared_kernel", standalone_load)
    module = utils._load_kernel(
        "kernel_direct_compatibility",
        ["int direct_compatibility;"],
        ["evaluate"],
        list(options.build.extra_cflags),
        list(options.build.extra_ldflags),
        compile_options=options,
    )
    assert module is loaded
    assert len(prepared) == 1


def test_isolated_compilation_context_plumbing_p95_is_below_one_millisecond() -> None:
    options = _default_options()
    samples: list[int] = []
    for _ in range(300):
        context = CompilationContext(options)
        started = perf_counter_ns()
        token = context.begin_stage(
            CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
            compile_options=options,
        )
        context.complete_stage(token)
        assert context.stage_run_records[0].duration_ns >= 0
        samples.append(perf_counter_ns() - started)
    samples.sort()
    p95 = samples[int((len(samples) - 1) * 0.95)]
    assert p95 < 1_000_000
