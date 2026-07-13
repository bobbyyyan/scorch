from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from time import perf_counter_ns
from typing import List, NoReturn, Set, Tuple, cast

import pytest

from scorch.compiler import llir
import scorch.compiler.llir_pass_manager as pass_manager_module
from scorch.compiler.compressed_where_openmp_pass import (
    CompressedWhereOpenMPContext,
    CompressedWhereOpenMPResult,
    transform_compressed_where_for_openmp,
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
    SPARSE_PREFETCH_PASS,
    CompressedWhereOpenMPPassSpec,
    DynamicVectorAccessPassSpec,
    LLIRPassArtifactType,
    LLIRPassContextType,
    LLIRPassDescriptor,
    LLIRPassManager,
    LLIRPassManagerError,
    LLIRPassOptions,
    LLIRPassRunRecord,
    LLIRRewriteArtifact,
    LLIRRewritePassResult,
    LLIRStatementListArtifact,
    LLIRStatementListPassResult,
    ManagedCompressedWhereOpenMPResult,
    ResultWritePassSpec,
    SparsePrefetchPassSpec,
)
from scorch.compiler.llir_traversal import (
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
)
from scorch.compiler.result_write_pass import (
    ResultWriteContext,
    ResultWriteMode,
    rewrite_result_writes,
)
from scorch.compiler.sparse_prefetch_pass import SparsePrefetchContext


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _result_context(mode: ResultWriteMode = "count") -> ResultWriteContext:
    return ResultWriteContext(
        result_name="Result",
        compressed_levels=(1,),
        mode=mode,
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


def test_all_manager_configuration_artifact_and_result_carriers_are_frozen() -> None:
    options = LLIRPassOptions()
    dynamic_spec = DynamicVectorAccessPassSpec()
    result_spec = ResultWritePassSpec(_result_context())
    compressed_spec = CompressedWhereOpenMPPassSpec(_compressed_context())
    sparse_prefetch_spec = SparsePrefetchPassSpec()
    rewrite_artifact = LLIRRewriteArtifact([llir.BlankLine()])
    statement_artifact = LLIRStatementListArtifact([llir.BlankLine()])
    record = _record(1)
    rewrite_result = LLIRRewritePassResult(rewrite_artifact, (record,))
    statement_result = LLIRStatementListPassResult(statement_artifact, (record,))
    compressed_result = ManagedCompressedWhereOpenMPResult(
        CompressedWhereOpenMPResult([], False),
        (record,),
    )
    manager = LLIRPassManager(options)

    frozen_updates: Tuple[Tuple[object, str, object], ...] = (
        (options, "record_timing", False),
        (DYNAMIC_VECTOR_ACCESS_PASS, "version", 2),
        (dynamic_spec, "context", DYNAMIC_VECTOR_ACCESS_CONTEXT),
        (result_spec, "descriptor", DYNAMIC_VECTOR_ACCESS_PASS),
        (compressed_spec, "descriptor", DYNAMIC_VECTOR_ACCESS_PASS),
        (sparse_prefetch_spec, "descriptor", DYNAMIC_VECTOR_ACCESS_PASS),
        (rewrite_artifact, "value", []),
        (statement_artifact, "statements", []),
        (record, "duration_ns", 2),
        (rewrite_result, "run_records", ()),
        (statement_result, "run_records", ()),
        (compressed_result, "run_records", ()),
        (manager, "options", DEBUG_LLIR_PASS_OPTIONS),
    )
    for value, name, replacement in frozen_updates:
        with pytest.raises(FrozenInstanceError):
            setattr(value, name, replacement)


def test_stable_descriptors_expose_exact_artifact_and_context_types() -> None:
    assert DYNAMIC_VECTOR_ACCESS_PASS == LLIRPassDescriptor(
        "rewrite_dynamic_vector_accesses",
        1,
        LLIRPassArtifactType.REWRITE_VALUE,
        LLIRPassArtifactType.REWRITE_VALUE,
        LLIRPassContextType.DYNAMIC_VECTOR_ACCESS,
    )
    assert RESULT_WRITE_PASS == LLIRPassDescriptor(
        "rewrite_result_writes",
        1,
        LLIRPassArtifactType.REWRITE_VALUE,
        LLIRPassArtifactType.REWRITE_VALUE,
        LLIRPassContextType.RESULT_WRITE,
    )
    assert COMPRESSED_WHERE_OPENMP_PASS == LLIRPassDescriptor(
        "transform_compressed_where_for_openmp",
        1,
        LLIRPassArtifactType.STATEMENT_LIST,
        LLIRPassArtifactType.COMPRESSED_WHERE_RESULT,
        LLIRPassContextType.COMPRESSED_WHERE_OPENMP,
    )
    assert SPARSE_PREFETCH_PASS == LLIRPassDescriptor(
        "insert_sparse_prefetch",
        1,
        LLIRPassArtifactType.STATEMENT_LIST,
        LLIRPassArtifactType.STATEMENT_LIST,
        LLIRPassContextType.SPARSE_PREFETCH,
    )


@pytest.mark.parametrize(
    "source",
    [
        llir.Literal(3),
        llir.BlankLine(),
        [llir.BlankLine(), [llir.Break()]],
        (llir.BlankLine(), (llir.Continue(),)),
    ],
)
def test_empty_manager_validates_detaches_and_preserves_every_root(
    source: object,
) -> None:
    managed = LLIRPassManager().run_empty(LLIRRewriteArtifact(cast(LLIRValue, source)))

    assert type(managed.artifact.value) is type(source)
    assert managed.run_records == ()
    assert source is not managed.artifact.value
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(managed.artifact.value))


def test_empty_manager_rejects_unknown_payload_instead_of_aliasing_it() -> None:
    with pytest.raises(LLIRTraversalError) as error:
        LLIRPassManager().run_empty(LLIRRewriteArtifact(cast(LLIRValue, object())))

    assert error.value.diagnostic.code == "invalid_llir_value"
    assert error.value.diagnostic.pass_name == "empty_pipeline"


def test_dynamic_runner_preserves_exact_root_and_detached_idempotence() -> None:
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_values", llir.DataType.STD_VECTOR_FLOAT32)),
        llir.Assign(_var("out_values[p]"), _var("value")),
    ]
    manager = LLIRPassManager()

    direct = rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
    once = manager.run_dynamic_vector_access(LLIRRewriteArtifact(source))
    twice = manager.run_dynamic_vector_access(once.artifact)

    assert _structural_snapshot(direct) == _structural_snapshot(once.artifact.value)
    assert _structural_snapshot(once.artifact.value) == _structural_snapshot(
        twice.artifact.value
    )
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(once.artifact.value))
    assert _mutable_ir_ids(once.artifact.value).isdisjoint(
        _mutable_ir_ids(twice.artifact.value)
    )
    assert once.run_records[0].configuration_name == "dynamic_vector_access"


def test_result_count_and_fill_are_independent_not_a_linear_pipeline() -> None:
    source: List[llir.Stmt] = [
        llir.Assign(_var("Result_values[pResult1]"), _var("value")),
        llir.Assign(_var("Result1_crd[pResult1]"), _var("coordinate")),
    ]
    manager = LLIRPassManager()

    count = manager.run_result_write(
        LLIRRewriteArtifact(source),
        ResultWritePassSpec(_result_context("count")),
    )
    fill = manager.run_result_write(
        LLIRRewriteArtifact(source),
        ResultWritePassSpec(_result_context("fill")),
    )
    count_again = manager.run_result_write(
        count.artifact,
        ResultWritePassSpec(_result_context("count")),
    )
    fill_after_count = manager.run_result_write(
        count.artifact,
        ResultWritePassSpec(_result_context("fill")),
    )

    assert count.run_records[0].configuration_name == "count"
    assert fill.run_records[0].configuration_name == "fill"
    assert _structural_snapshot(count.artifact.value) != _structural_snapshot(
        fill.artifact.value
    )
    assert _structural_snapshot(count.artifact.value) == _structural_snapshot(
        count_again.artifact.value
    )
    assert _structural_snapshot(fill_after_count.artifact.value) != (
        _structural_snapshot(fill.artifact.value)
    )
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(count.artifact.value))
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(fill.artifact.value))
    assert _mutable_ir_ids(count.artifact.value).isdisjoint(
        _mutable_ir_ids(fill.artifact.value)
    )
    assert _mutable_ir_ids(count.artifact.value).isdisjoint(
        _mutable_ir_ids(count_again.artifact.value)
    )


def test_result_runner_preserves_scalar_list_and_tuple_roots() -> None:
    manager = LLIRPassManager()
    spec = ResultWritePassSpec(_result_context())
    roots = (
        llir.Assign(_var("scratch"), _var("keep")),
        [llir.BlankLine(), [llir.Break()]],
        (llir.BlankLine(), (llir.Continue(),)),
    )

    for source in roots:
        result = manager.run_result_write(
            LLIRRewriteArtifact(cast(LLIRValue, source)),
            spec,
        )
        assert type(result.artifact.value) is type(source)
        assert _mutable_ir_ids(source).isdisjoint(
            _mutable_ir_ids(result.artifact.value)
        )


def test_compressed_runner_preserves_exact_applied_result_and_legal_noop() -> None:
    manager = LLIRPassManager()
    spec = CompressedWhereOpenMPPassSpec(_compressed_context())
    source = _compressed_source()

    applied = manager.run_compressed_where_openmp(
        LLIRStatementListArtifact(source), spec
    )
    noop_source: List[llir.Stmt] = [llir.BlankLine()]
    noop = manager.run_compressed_where_openmp(
        LLIRStatementListArtifact(noop_source), spec
    )

    assert type(applied.result) is CompressedWhereOpenMPResult
    assert applied.result.applied is True
    assert noop.result.applied is False
    assert _mutable_ir_ids(source).isdisjoint(
        _mutable_ir_ids(applied.result.statements)
    )
    assert _mutable_ir_ids(noop_source).isdisjoint(
        _mutable_ir_ids(noop.result.statements)
    )
    assert [record.pass_name for record in applied.run_records] == [
        "transform_compressed_where_for_openmp",
        "rewrite_result_writes",
        "rewrite_result_writes",
    ]
    assert [record.configuration_name for record in applied.run_records] == [
        "compressed_where_openmp",
        "count",
        "fill",
    ]
    assert [record.sequence_index for record in applied.run_records] == [0, 1, 2]
    assert [record.diagnostic_pass_name for record in applied.run_records] == [
        "transform_compressed_where_for_openmp",
        "transform_compressed_where_for_openmp",
        "transform_compressed_where_for_openmp",
    ]
    assert [record.pass_name for record in noop.run_records] == [
        "transform_compressed_where_for_openmp"
    ]


def test_compressed_runner_preserves_single_use_non_idempotent_contract() -> None:
    manager = LLIRPassManager()
    spec = CompressedWhereOpenMPPassSpec(_compressed_context())

    first = manager.run_compressed_where_openmp(
        LLIRStatementListArtifact(_compressed_source()), spec
    )
    second = manager.run_compressed_where_openmp(
        LLIRStatementListArtifact(first.result.statements), spec
    )

    assert first.result.applied is True
    assert second.result.applied is True
    assert _structural_snapshot(first.result.statements) != _structural_snapshot(
        second.result.statements
    )
    assert _mutable_ir_ids(first.result.statements).isdisjoint(
        _mutable_ir_ids(second.result.statements)
    )


def test_compressed_count_failure_stops_before_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modes: List[str] = []
    failure = LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code="synthetic_count_failure",
            message="count failed",
            path=("root",),
            node_type="RawStmt",
            stage="LLIR transformation",
            pass_name="transform_compressed_where_for_openmp",
        )
    )

    def fail_count(
        self: LLIRPassManager,
        artifact: object,
        pass_spec: ResultWritePassSpec,
    ) -> NoReturn:
        modes.append(pass_spec.context.mode)
        raise failure

    monkeypatch.setattr(LLIRPassManager, "run_result_write", fail_count)

    with pytest.raises(LLIRTraversalError) as error:
        LLIRPassManager().run_compressed_where_openmp(
            LLIRStatementListArtifact(_compressed_source()),
            CompressedWhereOpenMPPassSpec(_compressed_context()),
        )

    assert error.value is failure
    assert modes == ["count"]


class _UnknownStatement(llir.Stmt):
    pass


def test_each_runner_preserves_its_original_structured_diagnostic() -> None:
    source = [cast(llir.Stmt, _UnknownStatement())]
    manager = LLIRPassManager()

    with pytest.raises(LLIRTraversalError) as direct_dynamic:
        rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
    with pytest.raises(LLIRTraversalError) as managed_dynamic:
        manager.run_dynamic_vector_access(LLIRRewriteArtifact(source))
    assert managed_dynamic.value.diagnostic == direct_dynamic.value.diagnostic

    result_context = _result_context()
    with pytest.raises(LLIRTraversalError) as direct_result:
        rewrite_result_writes(source, result_context)
    with pytest.raises(LLIRTraversalError) as managed_result:
        manager.run_result_write(
            LLIRRewriteArtifact(source),
            ResultWritePassSpec(result_context),
        )
    assert managed_result.value.diagnostic == direct_result.value.diagnostic

    compressed_context = _compressed_context()
    with pytest.raises(LLIRTraversalError) as direct_compressed:
        transform_compressed_where_for_openmp(source, compressed_context)
    with pytest.raises(LLIRTraversalError) as managed_compressed:
        manager.run_compressed_where_openmp(
            LLIRStatementListArtifact(source),
            CompressedWhereOpenMPPassSpec(compressed_context),
        )
    assert managed_compressed.value.diagnostic == direct_compressed.value.diagnostic


def test_unknown_descriptor_artifact_spec_and_context_combinations_fail_closed() -> (
    None
):
    manager = LLIRPassManager()
    rewrite_artifact = LLIRRewriteArtifact([llir.BlankLine()])
    statement_artifact = LLIRStatementListArtifact([llir.BlankLine()])
    bad_descriptor = replace(DYNAMIC_VECTOR_ACCESS_PASS, version=2)

    invalid_calls = (
        lambda: manager.run_dynamic_vector_access(
            rewrite_artifact,
            DynamicVectorAccessPassSpec(descriptor=bad_descriptor),
        ),
        lambda: manager.run_dynamic_vector_access(
            cast(LLIRRewriteArtifact[List[llir.Stmt]], statement_artifact)
        ),
        lambda: manager.run_dynamic_vector_access(
            rewrite_artifact,
            cast(DynamicVectorAccessPassSpec, object()),
        ),
        lambda: manager.run_dynamic_vector_access(
            rewrite_artifact,
            DynamicVectorAccessPassSpec(
                context=cast(DynamicVectorAccessContext, object())
            ),
        ),
        lambda: manager.run_compressed_where_openmp(
            cast(LLIRStatementListArtifact, rewrite_artifact),
            CompressedWhereOpenMPPassSpec(_compressed_context()),
        ),
        lambda: manager.run_sparse_prefetch(
            cast(LLIRStatementListArtifact, rewrite_artifact),
            SparsePrefetchPassSpec(),
        ),
        lambda: manager.run_sparse_prefetch(
            statement_artifact,
            cast(SparsePrefetchPassSpec, object()),
        ),
        lambda: manager.run_sparse_prefetch(
            statement_artifact,
            SparsePrefetchPassSpec(context=cast(SparsePrefetchContext, object())),
        ),
    )
    for invalid in invalid_calls:
        with pytest.raises(LLIRPassManagerError):
            invalid()


def test_production_skips_extra_verification_and_debug_checks_all_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager_walks: List[Tuple[str, str]] = []

    class RecordingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            manager_walks.append((self.context.stage, self.context.pass_name))
            super().walk(value)

    monkeypatch.setattr(pass_manager_module, "LLIRWalker", RecordingWalker)
    rewrite_artifact = LLIRRewriteArtifact([llir.BlankLine()])
    result_spec = ResultWritePassSpec(_result_context())
    compressed_artifact = LLIRStatementListArtifact([llir.BlankLine()])
    compressed_spec = CompressedWhereOpenMPPassSpec(_compressed_context())

    production = LLIRPassManager(PRODUCTION_LLIR_PASS_OPTIONS)
    production_dynamic = production.run_dynamic_vector_access(rewrite_artifact)
    production_result = production.run_result_write(rewrite_artifact, result_spec)
    production_sparse_prefetch = production.run_sparse_prefetch(compressed_artifact)
    production_compressed = production.run_compressed_where_openmp(
        compressed_artifact, compressed_spec
    )
    assert manager_walks == []
    assert all(
        not record.verified_before and not record.verified_after
        for record in (
            *production_dynamic.run_records,
            *production_result.run_records,
            *production_sparse_prefetch.run_records,
            *production_compressed.run_records,
        )
    )

    debug = LLIRPassManager(DEBUG_LLIR_PASS_OPTIONS)
    debug_dynamic = debug.run_dynamic_vector_access(rewrite_artifact)
    debug_result = debug.run_result_write(rewrite_artifact, result_spec)
    debug_sparse_prefetch = debug.run_sparse_prefetch(compressed_artifact)
    debug_compressed = debug.run_compressed_where_openmp(
        compressed_artifact, compressed_spec
    )
    assert manager_walks == [
        ("LLIR rewrite", "rewrite_dynamic_vector_accesses"),
        ("LLIR rewrite", "rewrite_dynamic_vector_accesses"),
        ("LLIR rewrite", "rewrite_result_writes"),
        ("LLIR rewrite", "rewrite_result_writes"),
        ("LLIR transformation", "insert_sparse_prefetch"),
        ("LLIR transformation", "insert_sparse_prefetch"),
        ("LLIR transformation", "transform_compressed_where_for_openmp"),
        ("LLIR transformation", "transform_compressed_where_for_openmp"),
    ]
    assert all(
        record.verified_before and record.verified_after
        for record in (
            *debug_dynamic.run_records,
            *debug_result.run_records,
            *debug_sparse_prefetch.run_records,
            *debug_compressed.run_records,
        )
    )


def test_timing_records_are_ordered_nonsemantic_and_optional() -> None:
    artifact = LLIRRewriteArtifact([llir.BlankLine()])
    timed = LLIRPassManager().run_dynamic_vector_access(artifact)
    untimed = LLIRPassManager(
        LLIRPassOptions(record_timing=False)
    ).run_dynamic_vector_access(artifact)

    assert timed.run_records[0].duration_ns is not None
    assert cast(int, timed.run_records[0].duration_ns) >= 0
    assert untimed.run_records[0].duration_ns is None
    assert _record(1) == _record(999)
    shared = LLIRRewriteArtifact([llir.BlankLine()])
    assert LLIRRewritePassResult(shared, (_record(1),)) == LLIRRewritePassResult(
        shared, (_record(999),)
    )


def _p95(samples: List[int]) -> int:
    ordered = sorted(samples)
    return ordered[int((len(ordered) - 1) * 0.95)]


def test_empty_and_incremental_one_pass_plumbing_p95_is_below_one_ms() -> None:
    sample_count = 2000
    source = [llir.BlankLine()]
    manager = LLIRPassManager(LLIRPassOptions(record_timing=False))

    for _ in range(100):
        manager.run_empty(LLIRRewriteArtifact(source))
        rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
        manager.run_dynamic_vector_access(LLIRRewriteArtifact(source))

    empty_ns: List[int] = []
    incremental_ns: List[int] = []
    for sample in range(sample_count):
        started = perf_counter_ns()
        manager.run_empty(LLIRRewriteArtifact(source))
        empty_ns.append(perf_counter_ns() - started)

        if sample % 2:
            managed_start = perf_counter_ns()
            manager.run_dynamic_vector_access(LLIRRewriteArtifact(source))
            managed_elapsed = perf_counter_ns() - managed_start
            direct_start = perf_counter_ns()
            rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
            direct_elapsed = perf_counter_ns() - direct_start
        else:
            direct_start = perf_counter_ns()
            rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
            direct_elapsed = perf_counter_ns() - direct_start
            managed_start = perf_counter_ns()
            manager.run_dynamic_vector_access(LLIRRewriteArtifact(source))
            managed_elapsed = perf_counter_ns() - managed_start
        incremental_ns.append(managed_elapsed - direct_elapsed)

    assert _p95(empty_ns) <= 1_000_000
    assert _p95(incremental_ns) <= 1_000_000
