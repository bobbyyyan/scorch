from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from time import perf_counter_ns
from typing import List, Set, Tuple, cast

import pytest

from scorch.compiler import llir
import scorch.compiler.llir_pass_manager as pass_manager_module
from scorch.compiler.compressed_where_openmp_pass import (
    CompressedWhereOpenMPContext,
    CompressedWhereOpenMPResult,
)
from scorch.compiler.dynamic_vector_access_pass import (
    DYNAMIC_VECTOR_ACCESS_CONTEXT,
    DynamicVectorAccessContext,
    rewrite_dynamic_vector_accesses,
)
from scorch.compiler.llir_pass_manager import (
    COMPRESSED_WHERE_OPENMP_PASS,
    DEBUG_LLIR_PASS_OPTIONS,
    DYNAMIC_VECTOR_ACCESS_PASS,
    PRODUCTION_LLIR_PASS_OPTIONS,
    RESULT_WRITE_PASS,
    CompressedWhereOpenMPPassSpec,
    DynamicVectorAccessPassSpec,
    LLIRPassArtifact,
    LLIRPassArtifactType,
    LLIRPassContextType,
    LLIRPassDescriptor,
    LLIRPassManager,
    LLIRPassManagerError,
    LLIRPassOptions,
    LLIRPassPipelineResult,
    LLIRPassRunRecord,
    LLIRRewritePassSpec,
    LLIRRewritePipeline,
    ManagedCompressedWhereOpenMPResult,
    ResultWritePassSpec,
)
from scorch.compiler.llir_traversal import (
    LLIRTraversalError,
    LLIRWalker,
)
from scorch.compiler.result_write_pass import ResultWriteContext


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _result_context(mode: str = "count") -> ResultWriteContext:
    return ResultWriteContext(
        result_name="Result",
        compressed_levels=(1,),
        mode=cast("str", mode),
    )


def _compressed_context() -> CompressedWhereOpenMPContext:
    return CompressedWhereOpenMPContext(
        result_name="Result",
        compressed_levels=(1,),
        workspace_name="wksp",
        workspace_ctype="float",
    )


def _compatible_loop(body: List[llir.Stmt]) -> llir.ForLoop:
    return llir.ForLoop(
        init=llir.VarInit(_var("row", llir.DataType.INT), llir.Literal(0)),
        cond=llir.BinOp(
            "<",
            _var("row", llir.DataType.INT),
            _var("A0_size", llir.DataType.INT64),
        ),
        update=llir.Increment(_var("row", llir.DataType.INT)),
        body=body,
    )


def _compressed_source() -> List[llir.Stmt]:
    return [
        _compatible_loop(
            [
                llir.FunctionCallStmt(
                    "Result1_crd.push_back",
                    [_var("column")],
                ),
                llir.FunctionCallStmt(
                    "Result_values.push_back",
                    [_var("value")],
                ),
            ]
        )
    ]


def _mutable_ir_ids(value: object) -> Set[int]:
    mutable_ids: Set[int] = set()
    if isinstance(value, llir.Node):
        mutable_ids.add(id(value))
        for child in vars(value).values():
            mutable_ids.update(_mutable_ir_ids(child))
    elif isinstance(value, list):
        mutable_ids.add(id(value))
        for child in value:
            mutable_ids.update(_mutable_ir_ids(child))
    elif isinstance(value, tuple):
        for child in value:
            mutable_ids.update(_mutable_ir_ids(child))
    return mutable_ids


def _structural_snapshot(value: object) -> object:
    if isinstance(value, llir.Node):
        return (
            type(value).__name__,
            tuple(
                (name, _structural_snapshot(child))
                for name, child in sorted(vars(value).items())
            ),
        )
    if isinstance(value, list):
        return ("list", tuple(_structural_snapshot(child) for child in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_structural_snapshot(child) for child in value))
    return value


def _record(duration_ns: int) -> LLIRPassRunRecord:
    return LLIRPassRunRecord(
        sequence_index=0,
        pass_name=DYNAMIC_VECTOR_ACCESS_PASS.name,
        pass_version=DYNAMIC_VECTOR_ACCESS_PASS.version,
        input_artifact=DYNAMIC_VECTOR_ACCESS_PASS.input_artifact,
        output_artifact=DYNAMIC_VECTOR_ACCESS_PASS.output_artifact,
        context_type=DYNAMIC_VECTOR_ACCESS_PASS.context_type,
        configuration_name="dynamic_vector_access",
        diagnostic_stage="LLIR rewrite",
        diagnostic_pass_name="rewrite_dynamic_vector_accesses",
        verified_before=False,
        verified_after=False,
        duration_ns=duration_ns,
    )


def test_options_descriptors_specs_artifacts_results_and_manager_are_frozen() -> None:
    options = LLIRPassOptions()
    descriptor = DYNAMIC_VECTOR_ACCESS_PASS
    dynamic_spec = DynamicVectorAccessPassSpec()
    result_spec = ResultWritePassSpec(_result_context())
    compressed_spec = CompressedWhereOpenMPPassSpec(_compressed_context())
    artifact = LLIRPassArtifact([llir.BlankLine()])
    pipeline = LLIRRewritePipeline((dynamic_spec, result_spec))
    record = _record(1)
    pipeline_result = LLIRPassPipelineResult(artifact, (record,))
    compressed_result = ManagedCompressedWhereOpenMPResult(
        CompressedWhereOpenMPResult([], False),
        (record,),
    )
    manager = LLIRPassManager(options)

    frozen_updates: Tuple[Tuple[object, str, object], ...] = (
        (options, "record_timing", False),
        (descriptor, "version", 2),
        (dynamic_spec, "context", DYNAMIC_VECTOR_ACCESS_CONTEXT),
        (result_spec, "descriptor", DYNAMIC_VECTOR_ACCESS_PASS),
        (compressed_spec, "descriptor", DYNAMIC_VECTOR_ACCESS_PASS),
        (artifact, "value", []),
        (pipeline, "passes", ()),
        (record, "duration_ns", 2),
        (pipeline_result, "run_records", ()),
        (compressed_result, "run_records", ()),
        (manager, "options", DEBUG_LLIR_PASS_OPTIONS),
    )
    for value, name, replacement in frozen_updates:
        with pytest.raises(FrozenInstanceError):
            setattr(value, name, replacement)


def test_stable_pass_descriptors_expose_exact_boundary_types() -> None:
    assert DYNAMIC_VECTOR_ACCESS_PASS == LLIRPassDescriptor(
        name="rewrite_dynamic_vector_accesses",
        version=1,
        input_artifact=LLIRPassArtifactType.REWRITE_VALUE,
        output_artifact=LLIRPassArtifactType.REWRITE_VALUE,
        context_type=LLIRPassContextType.DYNAMIC_VECTOR_ACCESS,
    )
    assert RESULT_WRITE_PASS == LLIRPassDescriptor(
        name="rewrite_result_writes",
        version=1,
        input_artifact=LLIRPassArtifactType.REWRITE_VALUE,
        output_artifact=LLIRPassArtifactType.REWRITE_VALUE,
        context_type=LLIRPassContextType.RESULT_WRITE,
    )
    assert COMPRESSED_WHERE_OPENMP_PASS == LLIRPassDescriptor(
        name="transform_compressed_where_for_openmp",
        version=1,
        input_artifact=LLIRPassArtifactType.STATEMENT_LIST,
        output_artifact=LLIRPassArtifactType.COMPRESSED_WHERE_RESULT,
        context_type=LLIRPassContextType.COMPRESSED_WHERE_OPENMP,
    )


def test_explicit_pipeline_order_is_deterministic_and_observable() -> None:
    pipeline = LLIRRewritePipeline(
        (
            ResultWritePassSpec(_result_context("count")),
            DynamicVectorAccessPassSpec(),
            ResultWritePassSpec(_result_context("fill")),
        )
    )
    source = [llir.BlankLine()]

    managed = LLIRPassManager().run(LLIRPassArtifact(source), pipeline)

    assert [record.sequence_index for record in managed.run_records] == [0, 1, 2]
    assert [record.pass_name for record in managed.run_records] == [
        "rewrite_result_writes",
        "rewrite_dynamic_vector_accesses",
        "rewrite_result_writes",
    ]
    assert [record.configuration_name for record in managed.run_records] == [
        "count",
        "dynamic_vector_access",
        "fill",
    ]
    assert source is not managed.artifact.value
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(managed.artifact.value))


@pytest.mark.parametrize(
    "source",
    [
        llir.Literal(3),
        llir.BlankLine(),
        [llir.BlankLine(), [llir.Break()]],
        (llir.BlankLine(), (llir.Continue(),)),
    ],
)
def test_managed_dynamic_pass_preserves_scalar_and_container_roots(
    source: object,
) -> None:
    pipeline = LLIRRewritePipeline((DynamicVectorAccessPassSpec(),))

    managed = LLIRPassManager().run(
        LLIRPassArtifact(cast("object", source)),
        pipeline,
    )

    assert type(managed.artifact.value) is type(source)
    assert source is not managed.artifact.value
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(managed.artifact.value))


def test_two_managed_passes_do_not_alias_input_intermediate_or_output() -> None:
    source = [llir.BlankLine()]
    first = LLIRPassManager().run(
        LLIRPassArtifact(source),
        LLIRRewritePipeline((DynamicVectorAccessPassSpec(),)),
    )
    complete = LLIRPassManager().run(
        LLIRPassArtifact(source),
        LLIRRewritePipeline(
            (
                DynamicVectorAccessPassSpec(),
                ResultWritePassSpec(_result_context()),
            )
        ),
    )

    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(first.artifact.value))
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(complete.artifact.value))
    assert _mutable_ir_ids(first.artifact.value).isdisjoint(
        _mutable_ir_ids(complete.artifact.value)
    )


def test_managed_dynamic_rewrite_matches_direct_pass_and_is_idempotent() -> None:
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_values", llir.DataType.STD_VECTOR_FLOAT32)),
        llir.Assign(_var("out_values[p]"), _var("value")),
    ]
    pipeline = LLIRRewritePipeline((DynamicVectorAccessPassSpec(),))

    direct = rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
    once = LLIRPassManager().run(LLIRPassArtifact(source), pipeline).artifact.value
    twice = LLIRPassManager().run(LLIRPassArtifact(once), pipeline).artifact.value

    assert _structural_snapshot(direct) == _structural_snapshot(once)
    assert _structural_snapshot(once) == _structural_snapshot(twice)
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(once))
    assert _mutable_ir_ids(once).isdisjoint(_mutable_ir_ids(twice))


def test_managed_result_write_preserves_root_and_actual_repeat_contract() -> None:
    source = (
        llir.Assign(_var("scratch"), _var("keep")),
        (llir.RawStmt("opaque"),),
    )
    pipeline = LLIRRewritePipeline((ResultWritePassSpec(_result_context()),))

    once = LLIRPassManager().run(LLIRPassArtifact(source), pipeline).artifact.value
    twice = LLIRPassManager().run(LLIRPassArtifact(once), pipeline).artifact.value

    assert type(once) is tuple
    assert type(cast(Tuple[object, ...], once)[1]) is tuple
    assert _structural_snapshot(once) == _structural_snapshot(twice)
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(once))
    assert _mutable_ir_ids(once).isdisjoint(_mutable_ir_ids(twice))


def test_managed_compressed_where_preserves_applied_and_detached_noop() -> None:
    manager = LLIRPassManager()
    spec = CompressedWhereOpenMPPassSpec(_compressed_context())
    source = _compressed_source()

    applied = manager.run_compressed_where_openmp(LLIRPassArtifact(source), spec)
    noop_source: List[llir.Stmt] = [llir.BlankLine()]
    noop = manager.run_compressed_where_openmp(
        LLIRPassArtifact(noop_source),
        spec,
    )

    assert type(applied.result) is CompressedWhereOpenMPResult
    assert applied.result.applied is True
    assert noop.result.applied is False
    assert noop.result.statements is not noop_source
    assert _mutable_ir_ids(source).isdisjoint(
        _mutable_ir_ids(applied.result.statements)
    )
    assert _mutable_ir_ids(noop_source).isdisjoint(
        _mutable_ir_ids(noop.result.statements)
    )
    assert [record.pass_name for record in applied.run_records] == [
        "transform_compressed_where_for_openmp"
    ]


def test_managed_compressed_where_remains_single_use_and_non_idempotent() -> None:
    manager = LLIRPassManager()
    spec = CompressedWhereOpenMPPassSpec(_compressed_context())

    first = manager.run_compressed_where_openmp(
        LLIRPassArtifact(_compressed_source()),
        spec,
    )
    second = manager.run_compressed_where_openmp(
        LLIRPassArtifact(first.result.statements),
        spec,
    )

    assert first.result.applied is True
    assert second.result.applied is True
    assert _structural_snapshot(first.result.statements) != _structural_snapshot(
        second.result.statements
    )
    assert _mutable_ir_ids(first.result.statements).isdisjoint(
        _mutable_ir_ids(second.result.statements)
    )


class _UnknownStatement(llir.Stmt):
    pass


def test_managed_pass_preserves_original_structured_diagnostic() -> None:
    source = [cast(llir.Stmt, _UnknownStatement())]
    with pytest.raises(LLIRTraversalError) as direct_error:
        rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)

    with pytest.raises(LLIRTraversalError) as managed_error:
        LLIRPassManager().run(
            LLIRPassArtifact(source),
            LLIRRewritePipeline((DynamicVectorAccessPassSpec(),)),
        )

    assert managed_error.value.diagnostic == direct_error.value.diagnostic
    assert managed_error.value.diagnostic.pass_name == "rewrite_dynamic_vector_accesses"


def test_first_pass_failure_stops_later_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamic_called = False

    def unexpected_dynamic(
        value: object, context: DynamicVectorAccessContext
    ) -> object:
        nonlocal dynamic_called
        dynamic_called = True
        return value

    monkeypatch.setattr(
        pass_manager_module,
        "rewrite_dynamic_vector_accesses",
        unexpected_dynamic,
    )
    pipeline = LLIRRewritePipeline(
        (
            ResultWritePassSpec(_result_context()),
            DynamicVectorAccessPassSpec(),
        )
    )

    with pytest.raises(LLIRTraversalError) as error:
        LLIRPassManager().run(
            LLIRPassArtifact([cast(llir.Stmt, _UnknownStatement())]),
            pipeline,
        )

    assert error.value.diagnostic.pass_name == "rewrite_result_writes"
    assert dynamic_called is False


def test_unknown_pass_descriptor_artifact_and_context_combinations_fail_closed() -> (
    None
):
    manager = LLIRPassManager()
    source = LLIRPassArtifact([llir.BlankLine()])
    bad_descriptor = replace(DYNAMIC_VECTOR_ACCESS_PASS, version=2)
    bad_spec = DynamicVectorAccessPassSpec(descriptor=bad_descriptor)

    invalid_cases = (
        lambda: manager.run(
            source,
            LLIRRewritePipeline(cast(Tuple[LLIRRewritePassSpec, ...], (object(),))),
        ),
        lambda: manager.run(source, LLIRRewritePipeline((bad_spec,))),
        lambda: manager.run(
            source,
            LLIRRewritePipeline(
                (
                    DynamicVectorAccessPassSpec(
                        context=cast(DynamicVectorAccessContext, object())
                    ),
                )
            ),
        ),
        lambda: manager.run(
            cast(LLIRPassArtifact[object], object()),
            LLIRRewritePipeline(),
        ),
        lambda: manager.run_compressed_where_openmp(
            source,
            cast(CompressedWhereOpenMPPassSpec, object()),
        ),
    )
    for invalid in invalid_cases:
        with pytest.raises(LLIRPassManagerError):
            invalid()

    with pytest.raises(LLIRTraversalError) as artifact_error:
        manager.run(
            LLIRPassArtifact(cast(object, 3)),
            LLIRRewritePipeline((DynamicVectorAccessPassSpec(),)),
        )
    assert artifact_error.value.diagnostic.code == "invalid_llir_value"


def test_production_defaults_skip_extra_walks_and_debug_runs_pre_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager_walks: List[Tuple[str, str]] = []

    class RecordingWalker(LLIRWalker):
        def walk(self, value: object) -> None:
            manager_walks.append((self.context.stage, self.context.pass_name))
            super().walk(cast(object, value))

    monkeypatch.setattr(pass_manager_module, "LLIRWalker", RecordingWalker)
    artifact = LLIRPassArtifact([llir.BlankLine()])
    pipeline = LLIRRewritePipeline((DynamicVectorAccessPassSpec(),))

    production = LLIRPassManager(PRODUCTION_LLIR_PASS_OPTIONS).run(
        artifact,
        pipeline,
    )
    assert manager_walks == []
    assert production.run_records[0].verified_before is False
    assert production.run_records[0].verified_after is False

    debug = LLIRPassManager(DEBUG_LLIR_PASS_OPTIONS).run(artifact, pipeline)
    assert manager_walks == [
        ("LLIR rewrite", "rewrite_dynamic_vector_accesses"),
        ("LLIR rewrite", "rewrite_dynamic_vector_accesses"),
    ]
    assert debug.run_records[0].verified_before is True
    assert debug.run_records[0].verified_after is True


def test_timing_records_are_ordered_nonsemantic_and_optional() -> None:
    source = LLIRPassArtifact([llir.BlankLine()])
    pipeline = LLIRRewritePipeline((DynamicVectorAccessPassSpec(),))

    timed = LLIRPassManager().run(source, pipeline)
    untimed = LLIRPassManager(LLIRPassOptions(record_timing=False)).run(
        source,
        pipeline,
    )

    assert timed.run_records[0].duration_ns is not None
    assert cast(int, timed.run_records[0].duration_ns) >= 0
    assert untimed.run_records[0].duration_ns is None
    assert _record(1) == _record(999)
    shared_artifact = LLIRPassArtifact([llir.BlankLine()])
    assert LLIRPassPipelineResult(shared_artifact, (_record(1),)) == (
        LLIRPassPipelineResult(shared_artifact, (_record(999),))
    )


def _p95(samples: List[int]) -> int:
    ordered = sorted(samples)
    return ordered[int((len(ordered) - 1) * 0.95)]


def test_empty_and_one_pass_plumbing_p95_is_below_one_millisecond() -> None:
    samples = 2000
    source = [llir.BlankLine()]
    manager = LLIRPassManager(LLIRPassOptions(record_timing=False))
    empty = LLIRRewritePipeline()
    one = LLIRRewritePipeline((DynamicVectorAccessPassSpec(),))

    for _ in range(100):
        manager.run(LLIRPassArtifact(source), empty)
        manager.run(LLIRPassArtifact(source), one)

    empty_ns: List[int] = []
    one_ns: List[int] = []
    for _ in range(samples):
        started = perf_counter_ns()
        manager.run(LLIRPassArtifact(source), empty)
        empty_ns.append(perf_counter_ns() - started)

        started = perf_counter_ns()
        manager.run(LLIRPassArtifact(source), one)
        one_ns.append(perf_counter_ns() - started)

    assert _p95(empty_ns) <= 1_000_000
    assert _p95(one_ns) <= 1_000_000
