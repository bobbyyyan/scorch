"""Focused lifecycle tests for canonical compiler-stage timing.

These tests pin the Phase-2 production/debug stage-timing seam: typed
immutable stage identities and records, one timing owner per compilation
routing one exact ``CompileOptions`` snapshot, deterministic completion-order
records with explicit nesting, unchanged nested LLIR pass records, fail-closed
partial records, non-semantic exclusion from every cache/name/build identity,
and honest cached/prebuilt short-circuits.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from time import perf_counter_ns
from typing import Any, List, Optional, Tuple

import pytest
import torch

import scorch  # type: ignore[import-untyped]
import scorch.compiler.cin_analysis as cin_analysis  # type: ignore[import-untyped]
import scorch.compiler.llir_pass_manager as llir_pass_manager  # type: ignore[import-untyped]
import scorch.compiler.stage_timing as stage_timing_module  # type: ignore[import-untyped]
import scorch.ops as ops  # type: ignore[import-untyped]
from scorch.compiler.cin import (  # type: ignore[import-untyped]
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_analysis import (  # type: ignore[import-untyped]
    canonical_cin_dump,
    normalize_cin,
)
from scorch.compiler.cin_lowerer import (  # type: ignore[import-untyped]
    CINLowerer,
    ResultTensorAssembler,
)
from scorch.compiler.codegen import LLIRLowerer  # type: ignore[import-untyped]
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
    canonical_cache_digest,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    VerificationError,
)
from scorch.compiler.llir_pass_manager import (  # type: ignore[import-untyped]
    DEBUG_LLIR_PASS_OPTIONS,
    LLIRPassOptions,
    PRODUCTION_LLIR_PASS_OPTIONS,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    Schedule,
    Scheduler,
)
from scorch.compiler.stage_timing import (  # type: ignore[import-untyped]
    CANONICAL_COMPILER_STAGES,
    CompilerStageId,
    CompilerStageRecord,
    CompilerStageTiming,
    CompilerStageTimingError,
    CompilerStageToken,
)
from scorch.layout import TensorSpec  # type: ignore[import-untyped]
from scorch.utils import _kernel_name  # type: ignore[import-untyped]


def _default_options(
    *,
    verify_cin: bool = False,
    llir_pass_options: LLIRPassOptions = PRODUCTION_LLIR_PASS_OPTIONS,
    requested_schedule: Optional[Schedule] = None,
    regblock_dual: bool = True,
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


def _build_spmm_source() -> ForAll:
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


def _stage_values(timing: CompilerStageTiming) -> List[str]:
    return [record.stage.value for record in timing.records]


def _stub_build(build_calls: Optional[list] = None) -> Any:
    def record_build(**kwargs: object) -> object:
        if build_calls is not None:
            build_calls.append(kwargs)
        return object()

    return record_build


def _einsum_spmm_specs() -> Tuple[TensorSpec, TensorSpec]:
    return (
        TensorSpec("ds", (2, 3), name="A"),
        TensorSpec("dd", (3, 4), name="B"),
    )


def _compile_spmm_direct(
    source: ForAll,
    options: CompileOptions,
    timing: Optional[CompilerStageTiming],
) -> tuple[CINLowerer, str, str]:
    scheduled = Scheduler.auto_schedule(
        source,
        compile_options=options,
        stage_timing=timing,
    )
    lowerer = CINLowerer(compile_options=options, stage_timing=timing)
    lowered = lowerer.lower_IndexStmt(scheduled)
    cpp = LLIRLowerer(compile_options=options).lower_llir(lowered)
    name = _kernel_name(
        options.build.preamble_source,
        cpp,
        compile_options=options,
    )
    return lowerer, cpp, name


def test_stage_identities_and_records_are_typed_immutable_stable_and_ordered() -> None:
    assert CANONICAL_COMPILER_STAGES == (
        CompilerStageId.FRONTEND_CONSTRUCTION,
        CompilerStageId.CIN_NORMALIZATION,
        CompilerStageId.SCHEDULING,
        CompilerStageId.CIN_LOWERING,
        CompilerStageId.RESULT_ABI_ASSEMBLY,
        CompilerStageId.SCHEDULE_LOWERING,
        CompilerStageId.CPP_GENERATION,
        CompilerStageId.KERNEL_NAME_ASSEMBLY,
    )
    assert [stage.value for stage in CANONICAL_COMPILER_STAGES] == [
        "frontend_construction",
        "cin_normalization",
        "scheduling",
        "cin_lowering",
        "result_abi_assembly",
        "schedule_lowering",
        "cpp_generation",
        "kernel_name_assembly",
    ]

    record = CompilerStageRecord(
        sequence_index=0,
        stage=CompilerStageId.CIN_LOWERING,
        nested_within=None,
        duration_ns=17,
    )
    with pytest.raises(FrozenInstanceError):
        record.stage = CompilerStageId.SCHEDULING  # type: ignore[misc]
    assert record == replace(record, duration_ns=999_999)
    assert record != replace(record, sequence_index=1)

    options = _default_options()
    timing = CompilerStageTiming(compile_options=options)
    with pytest.raises(FrozenInstanceError):
        timing.compile_options = options  # type: ignore[misc]
    token = timing.begin(
        CompilerStageId.FRONTEND_CONSTRUCTION,
        compile_options=options,
    )
    assert type(token) is CompilerStageToken
    with pytest.raises(FrozenInstanceError):
        token.stage = CompilerStageId.SCHEDULING  # type: ignore[misc]
    timing.commit(token)
    assert type(timing.records) is tuple
    assert [record.sequence_index for record in timing.records] == [0]


def test_owner_and_boundary_validation_fail_closed() -> None:
    options = _default_options()
    other = _default_options()

    with pytest.raises(CompilerStageTimingError) as bad_owner:
        CompilerStageTiming(compile_options="not options")  # type: ignore[arg-type]
    assert bad_owner.value.diagnostic.code == "invalid_compile_options"

    timing = CompilerStageTiming(compile_options=options)
    with pytest.raises(CompilerStageTimingError) as bad_stage:
        timing.begin("scheduling", compile_options=options)  # type: ignore[arg-type]
    assert bad_stage.value.diagnostic.code == "invalid_stage_id"

    with pytest.raises(CompilerStageTimingError) as detached:
        timing.begin(CompilerStageId.SCHEDULING, compile_options=other)
    assert detached.value.diagnostic.code == "detached_compile_options"

    token = timing.begin(CompilerStageId.SCHEDULING, compile_options=options)
    with pytest.raises(CompilerStageTimingError) as reentrant:
        timing.begin(CompilerStageId.SCHEDULING, compile_options=options)
    assert reentrant.value.diagnostic.code == "reentrant_stage"
    timing.commit(token)
    with pytest.raises(CompilerStageTimingError) as unbalanced:
        timing.commit(token)
    assert unbalanced.value.diagnostic.code == "unbalanced_stage_commit"
    with pytest.raises(CompilerStageTimingError) as bad_token:
        timing.commit("token")  # type: ignore[arg-type]
    assert bad_token.value.diagnostic.code == "invalid_stage_token"

    with pytest.raises(TypeError):
        CINLowerer(compile_options=options, stage_timing="timing")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CINLowerer(
            compile_options=other,
            stage_timing=CompilerStageTiming(compile_options=options),
        )
    with pytest.raises(TypeError):
        ops.einsum(
            "ik,kj->ij",
            *_einsum_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _stage_timing=CompilerStageTiming(compile_options=other),
        )


def _explicit_schedule_records(
    monkeypatch: pytest.MonkeyPatch,
    options: CompileOptions,
) -> CompilerStageTiming:
    timing = CompilerStageTiming(compile_options=options)
    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
    result = ops.einsum(
        "ik,kj->ij",
        *_einsum_spmm_specs(),
        compile_only=True,
        format="dd",
        _compile_options=options,
        _stage_timing=timing,
    )
    assert isinstance(result, TensorSpec)
    return timing


_EXPLICIT_SCHEDULE_SEQUENCE = [
    "frontend_construction",
    "cin_normalization",
    "scheduling",
    "result_abi_assembly",
    "schedule_lowering",
    "cin_lowering",
    "cpp_generation",
    "kernel_name_assembly",
]


def test_production_records_complete_canonical_stage_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule(loop_order=("i", "k", "j"))
    options = _default_options(requested_schedule=schedule)
    timing = _explicit_schedule_records(monkeypatch, options)

    assert _stage_values(timing) == _EXPLICIT_SCHEDULE_SEQUENCE
    assert {record.stage for record in timing.records} == set(CANONICAL_COMPILER_STAGES)
    assert [record.sequence_index for record in timing.records] == list(range(8))
    nesting = {record.stage: record.nested_within for record in timing.records}
    assert nesting[CompilerStageId.CIN_NORMALIZATION] is CompilerStageId.SCHEDULING
    assert nesting[CompilerStageId.RESULT_ABI_ASSEMBLY] is CompilerStageId.CIN_LOWERING
    assert nesting[CompilerStageId.SCHEDULE_LOWERING] is CompilerStageId.CIN_LOWERING
    assert nesting[CompilerStageId.FRONTEND_CONSTRUCTION] is None
    assert nesting[CompilerStageId.SCHEDULING] is None
    assert nesting[CompilerStageId.CIN_LOWERING] is None
    assert nesting[CompilerStageId.CPP_GENERATION] is None
    assert nesting[CompilerStageId.KERNEL_NAME_ASSEMBLY] is None
    assert all(
        type(record.duration_ns) is int and record.duration_ns >= 0
        for record in timing.records
    )


def test_debug_records_same_canonical_stage_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule(loop_order=("i", "k", "j"))
    options = _default_options(
        requested_schedule=schedule,
        verify_cin=True,
        llir_pass_options=DEBUG_LLIR_PASS_OPTIONS,
    )
    timing = _explicit_schedule_records(monkeypatch, options)
    assert _stage_values(timing) == _EXPLICIT_SCHEDULE_SEQUENCE


def test_timing_does_not_enable_debug_verification_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_calls: list[object] = []
    original_verify = cin_analysis.verify_cin

    def counting_verify(cin: object) -> object:
        verify_calls.append(cin)
        return original_verify(cin)

    monkeypatch.setattr(cin_analysis, "verify_cin", counting_verify)

    production = _default_options()
    production_timing = CompilerStageTiming(compile_options=production)
    normalize_cin(
        _build_spmm_source(),
        compile_options=production,
        stage_timing=production_timing,
    )
    assert verify_calls == []
    assert _stage_values(production_timing) == ["cin_normalization"]

    debug = _default_options(
        verify_cin=True,
        llir_pass_options=DEBUG_LLIR_PASS_OPTIONS,
    )
    debug_timing = CompilerStageTiming(compile_options=debug)
    normalize_cin(
        _build_spmm_source(),
        compile_options=debug,
        stage_timing=debug_timing,
    )
    assert len(verify_calls) == 1
    assert _stage_values(debug_timing) == ["cin_normalization"]


def test_llir_pass_records_remain_nested_ordered_and_unchanged() -> None:
    options = _default_options()
    untimed_lowerer, untimed_cpp, untimed_name = _compile_spmm_direct(
        _build_spmm_source(), options, None
    )
    timing = CompilerStageTiming(compile_options=options)
    timed_lowerer, timed_cpp, timed_name = _compile_spmm_direct(
        _build_spmm_source(), options, timing
    )

    untimed_records = untimed_lowerer.llir_pass_run_records
    timed_records = timed_lowerer.llir_pass_run_records
    assert untimed_records == timed_records
    assert [record.pass_name for record in timed_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert all(
        type(record.duration_ns) is int and record.duration_ns >= 0
        for record in timed_records
    )
    # The stage owner holds only stage records; nested pass records stay on
    # the lowerer exactly once, within the enclosing CIN-lowering stage.
    assert _stage_values(timing) == [
        "cin_normalization",
        "scheduling",
        "result_abi_assembly",
        "cin_lowering",
    ]
    pass_total = sum(record.duration_ns for record in timed_records)
    lowering_record = timing.records[-1]
    assert lowering_record.stage is CompilerStageId.CIN_LOWERING
    assert lowering_record.duration_ns >= pass_total
    assert timed_cpp == untimed_cpp
    assert timed_name == untimed_name


def test_result_abi_assembly_record_keeps_lazy_barrier_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    original_factor = llir_pass_manager.LLIRPassManager.run_loop_invariant_factor_hoist
    original_dynamic = llir_pass_manager.LLIRPassManager.run_dynamic_vector_access
    original_begin = CompilerStageTiming.begin
    original_commit = CompilerStageTiming.commit

    def record_factor(self: object, *args: object, **kwargs: object) -> object:
        events.append(("pass", "hoist_loop_invariant_factors"))
        return original_factor(self, *args, **kwargs)

    def record_dynamic(self: object, *args: object, **kwargs: object) -> object:
        events.append(("pass", "rewrite_dynamic_vector_accesses"))
        return original_dynamic(self, *args, **kwargs)

    def record_begin(
        self: CompilerStageTiming, stage: CompilerStageId, **kwargs: object
    ) -> object:
        events.append(("begin", stage))
        return original_begin(self, stage, **kwargs)

    def record_commit(self: CompilerStageTiming, token: object) -> None:
        events.append(("commit", token.stage))  # type: ignore[attr-defined]
        original_commit(self, token)

    monkeypatch.setattr(
        llir_pass_manager.LLIRPassManager,
        "run_loop_invariant_factor_hoist",
        record_factor,
    )
    monkeypatch.setattr(
        llir_pass_manager.LLIRPassManager,
        "run_dynamic_vector_access",
        record_dynamic,
    )
    monkeypatch.setattr(CompilerStageTiming, "begin", record_begin)
    monkeypatch.setattr(CompilerStageTiming, "commit", record_commit)

    options = _default_options()
    timing = CompilerStageTiming(compile_options=options)
    _compile_spmm_direct(_build_spmm_source(), options, timing)

    factor_index = events.index(("pass", "hoist_loop_invariant_factors"))
    begin_index = events.index(("begin", CompilerStageId.RESULT_ABI_ASSEMBLY))
    commit_index = events.index(("commit", CompilerStageId.RESULT_ABI_ASSEMBLY))
    dynamic_index = events.index(("pass", "rewrite_dynamic_vector_accesses"))
    assert factor_index < begin_index < commit_index < dynamic_index

    assembly_records = [
        record
        for record in timing.records
        if record.stage is CompilerStageId.RESULT_ABI_ASSEMBLY
    ]
    assert len(assembly_records) == 1
    assert assembly_records[0].nested_within is CompilerStageId.CIN_LOWERING


def test_one_snapshot_routes_through_one_timing_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    timing = CompilerStageTiming(compile_options=options)
    begin_options: list[object] = []
    original_begin = CompilerStageTiming.begin

    def record_begin(
        self: CompilerStageTiming,
        stage: CompilerStageId,
        *,
        compile_options: CompileOptions,
    ) -> object:
        begin_options.append(compile_options)
        return original_begin(self, stage, compile_options=compile_options)

    monkeypatch.setattr(CompilerStageTiming, "begin", record_begin)
    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    result = ops.einsum(
        "ik,kj->ij",
        *_einsum_spmm_specs(),
        compile_only=True,
        format="dd",
        _compile_options=options,
        _stage_timing=timing,
    )
    assert isinstance(result, TensorSpec)
    assert begin_options
    assert all(value is options for value in begin_options)
    assert timing.records


def test_no_timed_stage_rereads_environment_or_contextvars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    timing = CompilerStageTiming(compile_options=options)

    def forbidden_snapshot(*args: object, **kwargs: object) -> object:
        raise AssertionError("a timed stage resnapshotted process state")

    monkeypatch.setattr(
        CompileOptions,
        "from_environment",
        classmethod(forbidden_snapshot),
    )
    monkeypatch.setenv("SCORCH_VERIFY_CIN", "1")
    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    verify_calls: list[object] = []
    original_verify = cin_analysis.verify_cin

    def counting_verify(cin: object) -> object:
        verify_calls.append(cin)
        return original_verify(cin)

    monkeypatch.setattr(cin_analysis, "verify_cin", counting_verify)

    with cin_analysis.full_cin_verification(True):
        result = ops.einsum(
            "ik,kj->ij",
            *_einsum_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _stage_timing=timing,
        )
    assert isinstance(result, TensorSpec)
    assert timing.records
    assert verify_calls == []


def test_timing_records_are_non_semantic_and_outside_every_identity() -> None:
    options = _default_options()
    source_dump = canonical_cin_dump(_build_spmm_source())

    first_timing = CompilerStageTiming(compile_options=options)
    _, first_cpp, first_name = _compile_spmm_direct(
        _build_spmm_source(), options, first_timing
    )
    second_timing = CompilerStageTiming(compile_options=options)
    _, second_cpp, second_name = _compile_spmm_direct(
        _build_spmm_source(), options, second_timing
    )
    _, untimed_cpp, untimed_name = _compile_spmm_direct(
        _build_spmm_source(), options, None
    )

    assert first_cpp == second_cpp == untimed_cpp
    assert first_name == second_name == untimed_name
    assert canonical_cin_dump(_build_spmm_source()) == source_dump

    assert _stage_values(first_timing) == _stage_values(second_timing)
    assert first_timing.records != second_timing.records or (
        [record.duration_ns for record in first_timing.records]
        != [record.duration_ns for record in second_timing.records]
        or first_timing.records == second_timing.records
    )
    stripped_first = tuple(
        replace(record, duration_ns=0) for record in first_timing.records
    )
    stripped_second = tuple(
        replace(record, duration_ns=0) for record in second_timing.records
    )
    assert stripped_first == stripped_second

    # Records and owners cannot enter canonical cache digests or option keys.
    canonical_cache_digest(options.cache_key)
    canonical_cache_digest(options.semantic_cache_key)
    with pytest.raises(TypeError):
        canonical_cache_digest((first_timing.records[0],))
    with pytest.raises(TypeError):
        canonical_cache_digest((first_timing,))

    key_with_timing = ops._codegen_kernel_cache_key(
        Scheduler.auto_schedule(
            _build_spmm_source(),
            compile_options=options,
            stage_timing=CompilerStageTiming(compile_options=options),
        ),
        None,
        None,
        compile_options=options,
    )
    key_without_timing = ops._codegen_kernel_cache_key(
        Scheduler.auto_schedule(_build_spmm_source(), compile_options=options),
        None,
        None,
        compile_options=options,
    )
    assert key_with_timing == key_without_timing


def test_build_requests_are_identical_with_and_without_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()

    def run_once(timing: Optional[CompilerStageTiming]) -> dict[str, object]:
        build_calls: list[dict[str, object]] = []
        monkeypatch.setattr(ops, "_load_kernel", _stub_build(build_calls))
        monkeypatch.setattr(ops, "_kernel_cache", {})
        monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
        kwargs: dict[str, object] = {
            "compile_only": True,
            "format": "dd",
            "_compile_options": options,
        }
        if timing is not None:
            kwargs["_stage_timing"] = timing
        ops.einsum("ik,kj->ij", *_einsum_spmm_specs(), **kwargs)
        assert len(build_calls) == 1
        return build_calls[0]

    timed_call = run_once(CompilerStageTiming(compile_options=options))
    untimed_call = run_once(None)
    assert timed_call == untimed_call


def test_two_distinct_snapshots_have_independent_owners_and_results() -> None:
    production = _default_options()
    debug = _default_options(
        verify_cin=True,
        llir_pass_options=DEBUG_LLIR_PASS_OPTIONS,
    )
    assert production.cache_key != debug.cache_key

    production_timing = CompilerStageTiming(compile_options=production)
    production_lowerer, production_cpp, _ = _compile_spmm_direct(
        _build_spmm_source(), production, production_timing
    )
    production_records = production_timing.records
    production_pass_records = production_lowerer.llir_pass_run_records

    debug_timing = CompilerStageTiming(compile_options=debug)
    _, debug_cpp, _ = _compile_spmm_direct(_build_spmm_source(), debug, debug_timing)

    assert production_timing.records == production_records
    assert production_lowerer.llir_pass_run_records == production_pass_records
    assert debug_timing.records is not production_timing.records
    assert _stage_values(debug_timing) == _stage_values(production_timing)
    assert production_cpp == debug_cpp
    assert all(
        not record.verified_before and not record.verified_after
        for record in production_pass_records
    )

    with pytest.raises(CompilerStageTimingError) as crossed:
        debug_timing.begin(
            CompilerStageId.FRONTEND_CONSTRUCTION,
            compile_options=production,
        )
    assert crossed.value.diagnostic.code == "detached_compile_options"


def test_scheduling_failure_preserves_frontend_and_normalization_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule(loop_order=("i", "k", "j"))
    options = _default_options(requested_schedule=schedule)
    timing = CompilerStageTiming(compile_options=options)
    boom = RuntimeError("injected legacy scheduling failure")

    def failing_legacy(*args: object, **kwargs: object) -> object:
        raise boom

    monkeypatch.setattr(
        Scheduler,
        "_apply_schedule_legacy",
        staticmethod(failing_legacy),
    )
    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_einsum_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _stage_timing=timing,
        )
    assert failure.value is boom
    assert _stage_values(timing) == [
        "frontend_construction",
        "cin_normalization",
    ]


def test_pass_failure_preserves_stage_and_partial_pass_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options(regblock_dual=False)
    timing = CompilerStageTiming(compile_options=options)
    lowerers: list[CINLowerer] = []
    original_init = CINLowerer.__init__

    def record_init(self: CINLowerer, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        lowerers.append(self)

    boom = RuntimeError("injected invariant-factor failure")

    def failing_factor(*args: object, **kwargs: object) -> object:
        raise boom

    monkeypatch.setattr(CINLowerer, "__init__", record_init)
    monkeypatch.setattr(
        llir_pass_manager,
        "hoist_loop_invariant_factors",
        failing_factor,
    )
    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_einsum_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _stage_timing=timing,
        )
    assert failure.value is boom
    assert _stage_values(timing) == [
        "frontend_construction",
        "cin_normalization",
        "scheduling",
    ]
    assert len(lowerers) == 1
    assert [record.pass_name for record in lowerers[0].llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
    ]


def test_assembly_failure_suppresses_assembly_and_later_stage_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options(regblock_dual=False)
    timing = CompilerStageTiming(compile_options=options)
    lowerers: list[CINLowerer] = []
    original_init = CINLowerer.__init__

    def record_init(self: CINLowerer, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        lowerers.append(self)

    boom = RuntimeError("injected assembly failure")

    def failing_assembly(*args: object, **kwargs: object) -> object:
        raise boom

    monkeypatch.setattr(CINLowerer, "__init__", record_init)
    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_final_assembly",
        failing_assembly,
    )
    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_einsum_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _stage_timing=timing,
        )
    assert failure.value is boom
    assert _stage_values(timing) == [
        "frontend_construction",
        "cin_normalization",
        "scheduling",
    ]
    assert len(lowerers) == 1
    assert [record.pass_name for record in lowerers[0].llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]


def test_codegen_and_kernel_name_failures_suppress_only_later_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options(regblock_dual=False)
    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    codegen_boom = RuntimeError("injected codegen failure")

    def failing_codegen(self: object, *args: object, **kwargs: object) -> object:
        raise codegen_boom

    codegen_timing = CompilerStageTiming(compile_options=options)
    monkeypatch.setattr(LLIRLowerer, "lower_llir", failing_codegen)
    with pytest.raises(RuntimeError) as codegen_failure:
        ops.einsum(
            "ik,kj->ij",
            *_einsum_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _stage_timing=codegen_timing,
        )
    assert codegen_failure.value is codegen_boom
    assert _stage_values(codegen_timing) == [
        "frontend_construction",
        "cin_normalization",
        "scheduling",
        "result_abi_assembly",
        "cin_lowering",
    ]
    monkeypatch.undo()

    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
    name_boom = RuntimeError("injected kernel-name failure")

    def failing_name(*args: object, **kwargs: object) -> object:
        raise name_boom

    name_timing = CompilerStageTiming(compile_options=options)
    monkeypatch.setattr(ops, "_kernel_name", failing_name)
    with pytest.raises(RuntimeError) as name_failure:
        ops.einsum(
            "ik,kj->ij",
            *_einsum_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _stage_timing=name_timing,
        )
    assert name_failure.value is name_boom
    assert _stage_values(name_timing) == [
        "frontend_construction",
        "cin_normalization",
        "scheduling",
        "result_abi_assembly",
        "cin_lowering",
        "cpp_generation",
    ]


def test_normalization_failure_leaves_only_frontend_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    timing = CompilerStageTiming(compile_options=options)
    boom = RuntimeError("injected verification failure")

    def failing_verify(*args: object, **kwargs: object) -> object:
        raise boom

    monkeypatch.setattr(cin_analysis, "verify_cin_if_enabled", failing_verify)
    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    with pytest.raises(RuntimeError) as failure:
        ops.einsum(
            "ik,kj->ij",
            *_einsum_spmm_specs(),
            compile_only=True,
            format="dd",
            _compile_options=options,
            _stage_timing=timing,
        )
    assert failure.value is boom
    assert _stage_values(timing) == ["frontend_construction"]


def test_compressed_where_fill_failure_keeps_count_record_and_stage_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    timing = CompilerStageTiming(compile_options=options)
    lowerers: list[CINLowerer] = []
    original_init = CINLowerer.__init__

    def record_init(self: CINLowerer, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        lowerers.append(self)

    boom = VerificationError("injected fill failure")
    original_rewrite = llir_pass_manager.rewrite_result_writes

    def failing_fill(value: object, context: object) -> object:
        if getattr(context, "mode", None) == "fill":
            raise boom
        return original_rewrite(value, context)

    monkeypatch.setattr(CINLowerer, "__init__", record_init)
    monkeypatch.setattr(llir_pass_manager, "rewrite_result_writes", failing_fill)
    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    with pytest.raises(VerificationError) as failure:
        ops.einsum(
            "ik,kj->ij",
            TensorSpec("ds", (2, 3), name="A"),
            TensorSpec("ds", (3, 4), name="B"),
            compile_only=True,
            format="ds",
            _compile_options=options,
            _stage_timing=timing,
        )
    assert failure.value is boom
    assert _stage_values(timing) == [
        "frontend_construction",
        "cin_normalization",
        "scheduling",
    ]
    assert len(lowerers) == 1
    nested = lowerers[0].llir_pass_run_records
    assert [record.configuration_name for record in nested] == ["count"]
    assert [record.pass_name for record in nested] == ["rewrite_result_writes"]


def test_compressed_where_success_keeps_full_nested_records_and_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    timing = CompilerStageTiming(compile_options=options)
    lowerers: list[CINLowerer] = []
    original_init = CINLowerer.__init__

    def record_init(self: CINLowerer, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        lowerers.append(self)

    monkeypatch.setattr(CINLowerer, "__init__", record_init)
    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    result = ops.einsum(
        "ik,kj->ij",
        TensorSpec("ds", (2, 3), name="A"),
        TensorSpec("ds", (3, 4), name="B"),
        compile_only=True,
        format="ds",
        _compile_options=options,
        _stage_timing=timing,
    )
    assert isinstance(result, TensorSpec)
    assert _stage_values(timing) == [
        "frontend_construction",
        "cin_normalization",
        "scheduling",
        "result_abi_assembly",
        "cin_lowering",
        "cpp_generation",
        "kernel_name_assembly",
    ]
    assert len(lowerers) == 1
    assert [
        record.configuration_name for record in lowerers[0].llir_pass_run_records
    ] == [
        "compressed_where_openmp",
        "count",
        "fill",
        "sparse_prefetch",
        "dense_pointer_hoist",
        "single_iteration_loop_elimination",
        "loop_invariant_factor_hoist",
        "dynamic_vector_access",
    ]


def test_dual_path_records_both_scheduled_and_lowered_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()
    timing = CompilerStageTiming(compile_options=options)
    monkeypatch.setattr(ops, "_load_kernel", _stub_build())
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    result = ops.einsum(
        "ik,kj->ij",
        *_einsum_spmm_specs(),
        compile_only=True,
        format="dd",
        _compile_options=options,
        _stage_timing=timing,
    )
    assert isinstance(result, TensorSpec)
    assert _stage_values(timing) == [
        "frontend_construction",
        "cin_normalization",
        "scheduling",
        "cin_normalization",
        "scheduling",
        "result_abi_assembly",
        "cin_lowering",
        "result_abi_assembly",
        "cin_lowering",
        "cpp_generation",
        "kernel_name_assembly",
    ]


def test_direct_and_standalone_compatibility_apis_are_unchanged() -> None:
    options = _default_options()
    source = _build_spmm_source()

    normalized = normalize_cin(source, compile_options=options)
    scheduled = Scheduler.auto_schedule(source, compile_options=options)
    explicit = Scheduler.apply_schedule(
        source,
        Schedule(loop_order=("i", "k", "j")),
        compile_options=_default_options(
            requested_schedule=Schedule(loop_order=("i", "k", "j"))
        ),
    )
    lowerer = CINLowerer(compile_options=options)
    lowered = lowerer.lower_IndexStmt(scheduled)
    cpp = LLIRLowerer(compile_options=options).lower_llir(lowered)
    standalone_cpp = LLIRLowerer().lower_llir(lowered)

    assert normalized is not source
    assert type(explicit).__name__ == "ScheduledCIN"
    assert cpp == standalone_cpp
    assert lowerer.llir_pass_run_records

    # A direct boundary snapshot cannot silently pair with a foreign owner.
    timing = CompilerStageTiming(compile_options=options)
    with pytest.raises(CompilerStageTimingError) as detached:
        normalize_cin(_build_spmm_source(), stage_timing=timing)
    assert detached.value.diagnostic.code == "detached_compile_options"


def test_cached_and_prebuilt_paths_do_not_fabricate_stage_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real end-to-end compile needs the real process environment so the
    # isolated build child can resolve ninja and the toolchain.
    options = CompileOptions.from_environment()
    left = scorch.from_torch(torch.rand(8), "cache_left")
    right = scorch.from_torch(torch.rand(8), "cache_right")

    first_timing = CompilerStageTiming(compile_options=options)
    first = ops.einsum(
        "i,i->i",
        left,
        right,
        _compile_options=options,
        _stage_timing=first_timing,
    )
    assert torch.allclose(
        first.to_torch(),
        left.to_torch() * right.to_torch(),
        atol=1e-3,
        rtol=1e-3,
    )
    assert "cin_lowering" in _stage_values(first_timing)
    assert "cpp_generation" in _stage_values(first_timing)
    assert "kernel_name_assembly" in _stage_values(first_timing)

    # Dispatch-cache hit: the compiler does not run, so nothing is recorded.
    hit_timing = CompilerStageTiming(compile_options=options)
    ops.einsum(
        "i,i->i",
        left,
        right,
        _compile_options=options,
        _stage_timing=hit_timing,
    )
    assert hit_timing.records == ()

    # Kernel-cache hit with a cold dispatch cache: scheduling honestly reruns
    # while lowering, codegen, and kernel naming are skipped.
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
    kernel_hit_timing = CompilerStageTiming(compile_options=options)
    ops.einsum(
        "i,i->i",
        left,
        right,
        _compile_options=options,
        _stage_timing=kernel_hit_timing,
    )
    recorded = set(_stage_values(kernel_hit_timing))
    assert "frontend_construction" in recorded
    assert "scheduling" in recorded
    assert "cin_lowering" not in recorded
    assert "result_abi_assembly" not in recorded
    assert "cpp_generation" not in recorded
    assert "kernel_name_assembly" not in recorded

    # Prebuilt CSR-by-dense matmul: the generic compiler never runs.
    csr = scorch.from_torch((torch.rand(16, 16) < 0.4).float(), "prebuilt_a").to_sparse(
        "ds"
    )
    dense = torch.rand(16, 4)
    prebuilt_timing = CompilerStageTiming(compile_options=options)
    prebuilt = ops.matmul(
        csr,
        dense,
        _compile_options=options,
        _stage_timing=prebuilt_timing,
    )
    assert torch.allclose(
        prebuilt if isinstance(prebuilt, torch.Tensor) else prebuilt.to_torch(),
        csr.to_torch() @ dense,
        atol=1e-3,
        rtol=1e-3,
    )
    assert prebuilt_timing.records == ()


def test_deterministic_clock_monkeypatching_yields_exact_durations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter(range(0, 10_000, 250))
    monkeypatch.setattr(
        stage_timing_module,
        "perf_counter_ns",
        lambda: next(ticks),
    )
    options = _default_options()
    timing = CompilerStageTiming(compile_options=options)
    outer = timing.begin(
        CompilerStageId.SCHEDULING,
        compile_options=options,
    )
    inner = timing.begin(
        CompilerStageId.CIN_NORMALIZATION,
        compile_options=options,
    )
    timing.commit(inner)
    timing.commit(outer)
    assert [record.duration_ns for record in timing.records] == [250, 750]
    assert (
        timing.records[0].nested_within is CompilerStageId.SCHEDULING
        and timing.records[1].nested_within is None
    )


def test_stage_timing_overhead_is_bounded() -> None:
    options = _default_options()
    samples: List[int] = []
    for _ in range(200):
        timing = CompilerStageTiming(compile_options=options)
        started = perf_counter_ns()
        token = timing.begin(
            CompilerStageId.FRONTEND_CONSTRUCTION,
            compile_options=options,
        )
        timing.commit(token)
        samples.append(perf_counter_ns() - started)
    samples.sort()
    p95 = samples[int((len(samples) - 1) * 0.95)]
    assert p95 <= 1_000_000
