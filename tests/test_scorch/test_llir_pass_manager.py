from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from time import perf_counter_ns
from typing import Callable, List, NoReturn, Set, Tuple, cast

import pytest
import torch

from scorch.compiler import llir  # type: ignore[import-untyped]
import scorch.compiler.llir_pass_manager as pass_manager_module  # type: ignore[import-untyped]
import scorch.compiler.result_write_pass as result_write_module  # type: ignore[import-untyped]
from scorch.compiler.compile_options import CompileOptions  # type: ignore[import-untyped]
from scorch.compiler.compressed_where_openmp_pass import (  # type: ignore[import-untyped]
    CompressedWhereOpenMPContext,
    CompressedWhereOpenMPResult,
    transform_compressed_where_for_openmp,
)
from scorch.compiler.dynamic_vector_access_pass import (  # type: ignore[import-untyped]
    DYNAMIC_VECTOR_ACCESS_CONTEXT,
    DynamicVectorAccessContext,
    rewrite_dynamic_vector_accesses,
)
from scorch.compiler.dense_pointer_hoist_pass import (  # type: ignore[import-untyped]
    DensePointerHoistContext,
    hoist_dense_pointers,
)
from scorch.compiler.llir_pass_manager import (  # type: ignore[import-untyped]
    COMPRESSED_WHERE_OPENMP_PASS,
    CURRENT_LLIR_PASS_DESCRIPTORS,
    CURRENT_LLIR_PASSES,
    DEBUG_LLIR_PASS_OPTIONS,
    DENSE_POINTER_HOIST_PASS,
    DYNAMIC_VECTOR_ACCESS_PASS,
    LOOP_INVARIANT_FACTOR_HOIST_PASS,
    PRODUCTION_LLIR_PASS_OPTIONS,
    RESULT_WRITE_PASS,
    SINGLE_ITERATION_LOOP_ELIMINATION_PASS,
    SPARSE_PREFETCH_PASS,
    CompressedWhereOpenMPPassSpec,
    DensePointerHoistPassSpec,
    DynamicVectorAccessPassSpec,
    LLIRPassArtifactType,
    LLIRPassContextType,
    LLIRPassDescriptor,
    LLIRPassId,
    LLIRPassManager,
    LLIRPassManagerError,
    LLIRPassOptions,
    LLIRPassPartialFailure,
    LLIRPassPipeline,
    LLIRPassRunRecord,
    LLIRRewriteArtifact,
    LLIRRewritePassResult,
    LLIRStatementListArtifact,
    LLIRStatementListPassResult,
    LoopInvariantFactorHoistPassSpec,
    ManagedCompressedWhereOpenMPResult,
    ResultWritePassSpec,
    SingleIterationLoopEliminationPassSpec,
    SparsePrefetchPassSpec,
)
from scorch.compiler.llir_traversal import (  # type: ignore[import-untyped]
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
)
from scorch.compiler.identity import (  # type: ignore[import-untyped]
    AccessId,
    IndexId,
    SymbolId,
)
from scorch.compiler.torch_cpp_abi import ResultTensorAssembler  # type: ignore[import-untyped]
from scorch.format import LevelType
from scorch.compiler.loop_invariant_factor_pass import (  # type: ignore[import-untyped]
    LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
    LoopInvariantFactorHoistContext,
    hoist_loop_invariant_factors,
)
from scorch.compiler.result_write_pass import (  # type: ignore[import-untyped]
    ResultWriteContext,
    ResultWriteMode,
    rewrite_result_writes,
)
from scorch.compiler.single_iteration_loop_pass import (  # type: ignore[import-untyped]
    SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    SingleIterationLoopEliminationContext,
    eliminate_single_iteration_loops,
)
from scorch.compiler.sparse_prefetch_pass import (  # type: ignore[import-untyped]
    SparsePrefetchContext,
)


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


_RESULT_ID = SymbolId(41)


def _access(
    array: str,
    index: str,
    *,
    metadata: llir.TensorAccessMetadata | None = None,
) -> llir.ArrayAccess:
    return llir.ArrayAccess(
        array=_var(array),
        index=_var(index, llir.DataType.INT64),
        tensor_access=metadata,
    )


def _fixed_stack_array_decl() -> llir.FixedStackArrayDecl:
    return llir.FixedStackArrayDecl(
        name="wksp",
        element_type=llir.DataType.FLOAT32,
        extent=_var("kTile_k", llir.DataType.CONSTEXPR_INT),
        initializer=llir.Array([], llir.DataType.FLOAT32),
    )


def _result_metadata() -> llir.TensorAccessMetadata:
    return llir.TensorAccessMetadata(
        access_id=AccessId(43),
        tensor_id=_RESULT_ID,
        index_ids=(IndexId(47),),
        role=llir.TensorAccessRole.RESULT_WRITE,
    )


def _result_context(mode: ResultWriteMode = "count") -> ResultWriteContext:
    return ResultWriteContext(
        result_name="Result",
        result_id=_RESULT_ID,
        compressed_levels=(1,),
        mode=mode,
        value_pointer_type=llir.DataType.PTR_FLOAT32,
    )


def _compressed_context() -> CompressedWhereOpenMPContext:
    return CompressedWhereOpenMPContext(
        result_name="Result",
        result_id=_RESULT_ID,
        compressed_levels=(1,),
        result_assembler=ResultTensorAssembler(
            name="Result",
            level_types=(LevelType.DENSE, LevelType.COMPRESSED),
            dtype=torch.float32,
        ),
        workspace_name="wksp",
        workspace_ctype="float",
    )


def _dense_context() -> DensePointerHoistContext:
    return DensePointerHoistContext((("Input_val", "float"),))


def _compile_options(
    pass_options: LLIRPassOptions = PRODUCTION_LLIR_PASS_OPTIONS,
) -> CompileOptions:
    return CompileOptions.from_environment(
        environ={},
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
        llir_pass_options=pass_options,
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


def _compressed_offset_family(statements: List[llir.Stmt]) -> List[llir.Stmt]:
    """Select the generated W6--W9 level-one family in structural order."""

    selected: List[llir.Stmt] = []
    for statement in statements:
        if type(statement) is llir.DirectInit and statement.var.name in (
            "_count1",
            "_offset1",
        ):
            selected.append(statement)
            continue
        if type(statement) is llir.Assign:
            target = statement.var
            if (
                type(target) is llir.ArrayAccess
                and type(target.array) is llir.Var
                and target.array.name == "_offset1"
                and type(target.index) is llir.Literal
                and target.index.value == 0
            ):
                selected.append(statement)
            continue
        if type(statement) is llir.ForLoop:
            init = statement.init
            if type(init) is llir.VarInit and init.var.name == "_i":
                selected.append(statement)
            continue
        if type(statement) is llir.VarInit and statement.var.name == "_total1":
            selected.append(statement)
    return selected


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
    dense_context = _dense_context()
    dense_spec = DensePointerHoistPassSpec(dense_context)
    single_context = SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT
    single_spec = SingleIterationLoopEliminationPassSpec(single_context)
    factor_context = LOOP_INVARIANT_FACTOR_HOIST_CONTEXT
    factor_spec = LoopInvariantFactorHoistPassSpec(factor_context)
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
        (dense_context, "value_array_ctypes", ()),
        (dense_spec, "descriptor", DYNAMIC_VECTOR_ACCESS_PASS),
        (single_context, "traversal", object()),
        (single_spec, "descriptor", DYNAMIC_VECTOR_ACCESS_PASS),
        (factor_context, "traversal", object()),
        (factor_spec, "descriptor", DYNAMIC_VECTOR_ACCESS_PASS),
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
    assert DENSE_POINTER_HOIST_PASS == LLIRPassDescriptor(
        "hoist_dense_pointers",
        1,
        LLIRPassArtifactType.STATEMENT_LIST,
        LLIRPassArtifactType.STATEMENT_LIST,
        LLIRPassContextType.DENSE_POINTER_HOIST,
    )
    assert SINGLE_ITERATION_LOOP_ELIMINATION_PASS == LLIRPassDescriptor(
        "eliminate_single_iteration_loops",
        1,
        LLIRPassArtifactType.STATEMENT_LIST,
        LLIRPassArtifactType.STATEMENT_LIST,
        LLIRPassContextType.SINGLE_ITERATION_LOOP_ELIMINATION,
    )
    assert LOOP_INVARIANT_FACTOR_HOIST_PASS == LLIRPassDescriptor(
        "hoist_loop_invariant_factors",
        1,
        LLIRPassArtifactType.STATEMENT_LIST,
        LLIRPassArtifactType.STATEMENT_LIST,
        LLIRPassContextType.LOOP_INVARIANT_FACTOR_HOIST,
    )


def test_compile_options_assembles_one_frozen_typed_ordered_pipeline() -> None:
    compile_options = _compile_options()
    pipeline = LLIRPassPipeline.from_compile_options(compile_options)
    manager = LLIRPassManager.from_compile_options(compile_options)

    assert type(pipeline) is LLIRPassPipeline
    assert pipeline.compile_options is compile_options
    assert pipeline.pass_ids is compile_options.enabled_llir_passes
    assert pipeline.options is compile_options.verification.llir_pass_options
    assert pipeline.pass_ids == CURRENT_LLIR_PASSES
    assert pipeline.pass_descriptors == CURRENT_LLIR_PASS_DESCRIPTORS
    assert type(pipeline.pass_ids) is tuple
    assert type(pipeline.pass_descriptors) is tuple
    assert all(type(pass_id) is LLIRPassId for pass_id in pipeline.pass_ids)
    assert all(
        type(descriptor) is LLIRPassDescriptor
        for descriptor in pipeline.pass_descriptors
    )
    assert tuple(pass_id.value for pass_id in pipeline.pass_ids) == tuple(
        descriptor.name for descriptor in pipeline.pass_descriptors
    )

    assert manager.pipeline is not None
    assert manager.pipeline.compile_options is compile_options
    assert manager.pipeline.pass_ids is compile_options.enabled_llir_passes
    assert manager.pipeline.options is compile_options.verification.llir_pass_options
    assert manager.options is compile_options.verification.llir_pass_options

    with pytest.raises(FrozenInstanceError):
        pipeline.pass_ids = ()
    with pytest.raises(FrozenInstanceError):
        pipeline.compile_options = _compile_options()


def test_production_pipeline_runs_ordinary_passes_in_exact_order() -> None:
    compile_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(compile_options)
    source: List[llir.Stmt] = [llir.BlankLine()]
    assembly_modes: List[bool] = []

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assembly_modes.append(compressed_output_parallel)
        return LLIRRewriteArtifact(artifact.statements)

    result = manager.run_production_pipeline(
        LLIRStatementListArtifact(source),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )

    assert assembly_modes == [False]
    assert [record.pass_name for record in result.run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.configuration_name for record in result.run_records] == [
        "sparse_prefetch",
        "dense_pointer_hoist",
        "single_iteration_loop_elimination",
        "loop_invariant_factor_hoist",
        "dynamic_vector_access",
    ]
    assert [record.sequence_index for record in result.run_records] == list(range(5))
    assert result.compressed_output_parallel is False
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(result.artifact.value))


def test_fixed_stack_array_survives_repeated_production_pipeline_detached() -> None:
    compile_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(compile_options)
    source_decl = _fixed_stack_array_decl()
    source: List[llir.Stmt] = [source_decl]
    source_snapshot = _structural_snapshot(source)

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assert compressed_output_parallel is False
        return LLIRRewriteArtifact(artifact.statements)

    once = manager.run_production_pipeline(
        LLIRStatementListArtifact(source),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )
    twice = manager.run_production_pipeline(
        LLIRStatementListArtifact(once.artifact.value),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )

    assert _structural_snapshot(source) == source_snapshot
    assert _structural_snapshot(once.artifact.value) == source_snapshot
    assert _structural_snapshot(twice.artifact.value) == source_snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(once.artifact.value))
    assert _mutable_ir_ids(once.artifact.value).isdisjoint(
        _mutable_ir_ids(twice.artifact.value)
    )
    expected_passes = [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    for result in (once, twice):
        assert [record.pass_name for record in result.run_records] == expected_passes
        assert [record.sequence_index for record in result.run_records] == list(
            range(5)
        )
        assert all(record.duration_ns is not None for record in result.run_records)

    once_decl = cast(llir.FixedStackArrayDecl, once.artifact.value[0])
    twice_decl = cast(llir.FixedStackArrayDecl, twice.artifact.value[0])
    assert type(once_decl) is llir.FixedStackArrayDecl
    assert type(twice_decl) is llir.FixedStackArrayDecl
    assert once_decl == source_decl == twice_decl
    assert hash(once_decl) == hash(source_decl) == hash(twice_decl)
    assert once_decl.extent is not source_decl.extent
    assert once_decl.initializer is not source_decl.initializer
    assert twice_decl.extent is not once_decl.extent
    assert twice_decl.initializer is not once_decl.initializer


def test_workspace_pair_reads_survive_repeated_production_pipeline_noops() -> None:
    compile_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(compile_options)
    source: List[llir.Stmt] = [
        llir.ForLoopAuto(
            var=_var("it", llir.DataType.CONST_AUTO_REF),
            array=_var("workspace", llir.DataType.AUTO),
            body=[
                llir.VarInit(
                    _var("column", llir.DataType.INT64),
                    llir.ArrayAccess(
                        llir.MemberAccess(
                            _var("it", llir.DataType.CONST_AUTO_REF),
                            "first",
                        ),
                        llir.Literal(1, llir.DataType.INT64),
                    ),
                ),
                llir.VarInit(
                    _var("value", llir.DataType.FLOAT32),
                    llir.MemberAccess(
                        _var("it", llir.DataType.CONST_AUTO_REF),
                        "second",
                    ),
                ),
            ],
        )
    ]

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assert compressed_output_parallel is False
        return LLIRRewriteArtifact(artifact.statements)

    once = manager.run_production_pipeline(
        LLIRStatementListArtifact(source),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )
    twice = manager.run_production_pipeline(
        LLIRStatementListArtifact(once.artifact.value),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )

    assert _structural_snapshot(source) == _structural_snapshot(once.artifact.value)
    assert _structural_snapshot(once.artifact.value) == _structural_snapshot(
        twice.artifact.value
    )
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(once.artifact.value))
    assert _mutable_ir_ids(once.artifact.value).isdisjoint(
        _mutable_ir_ids(twice.artifact.value)
    )
    expected_passes = [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.pass_name for record in once.run_records] == expected_passes
    assert [record.pass_name for record in twice.run_records] == expected_passes
    assert [record.sequence_index for record in once.run_records] == list(range(5))
    assert all(record.duration_ns is not None for record in once.run_records)


def test_physical_workspace_copy_read_survives_repeated_production_pipeline() -> None:
    compile_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(compile_options)
    source_read = llir.ArrayAccess(
        _var("wksp", llir.DataType.PTR_FLOAT32),
        _var("k_in", llir.DataType.INT64),
    )
    source: List[llir.Stmt] = [
        llir.Assign(
            _access("C_values", "pC1"),
            source_read,
            op=llir.AssignOp.ADD_ASSIGN,
        )
    ]

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assert compressed_output_parallel is False
        return LLIRRewriteArtifact(artifact.statements)

    once = manager.run_production_pipeline(
        LLIRStatementListArtifact(source),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )
    twice = manager.run_production_pipeline(
        LLIRStatementListArtifact(once.artifact.value),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )

    assert _structural_snapshot(source) == _structural_snapshot(once.artifact.value)
    assert _structural_snapshot(once.artifact.value) == _structural_snapshot(
        twice.artifact.value
    )
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(once.artifact.value))
    assert _mutable_ir_ids(once.artifact.value).isdisjoint(
        _mutable_ir_ids(twice.artifact.value)
    )
    expected_passes = [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.pass_name for record in once.run_records] == expected_passes
    assert [record.pass_name for record in twice.run_records] == expected_passes
    assert [record.sequence_index for record in once.run_records] == list(range(5))
    assert all(record.duration_ns is not None for record in once.run_records)

    once_assignment = cast(llir.Assign, once.artifact.value[0])
    twice_assignment = cast(llir.Assign, twice.artifact.value[0])
    assert once_assignment.value == source_read
    assert twice_assignment.value == source_read
    assert type(once_assignment.value) is llir.ArrayAccess
    once_read = cast(llir.ArrayAccess, once_assignment.value)
    assert type(once_read.array) is llir.Var
    assert cast(llir.Var, once_read.array).type is llir.DataType.PTR_FLOAT32
    assert type(once_read.index) is llir.Var
    assert cast(llir.Var, once_read.index).type is llir.DataType.INT64
    assert once_read.tensor_access is None


def test_all_coo_coordinate_read_survives_repeated_production_pipeline() -> None:
    compile_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(compile_options)
    source_read = llir.ArrayAccess(
        _var("Mask0_crd", llir.DataType.PTR_INT),
        _var("pMask0", llir.DataType.INT64),
    )
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("_group_starts", llir.DataType.STD_VECTOR_C_INT)),
        llir.VarInit(
            _var("pMask0", llir.DataType.INT64),
            _var("_group_starts[_g]", llir.DataType.INT64),
        ),
        llir.VarInit(_var("r", llir.DataType.INT64), source_read),
    ]

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assert compressed_output_parallel is False
        return LLIRRewriteArtifact(artifact.statements)

    once = manager.run_production_pipeline(
        LLIRStatementListArtifact(source),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )
    twice = manager.run_production_pipeline(
        LLIRStatementListArtifact(once.artifact.value),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )

    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(once.artifact.value))
    assert _mutable_ir_ids(once.artifact.value).isdisjoint(
        _mutable_ir_ids(twice.artifact.value)
    )
    assert _structural_snapshot(once.artifact.value) == _structural_snapshot(
        twice.artifact.value
    )
    expected_passes = [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.pass_name for record in once.run_records] == expected_passes
    assert [record.sequence_index for record in once.run_records] == list(range(5))
    assert all(record.duration_ns is not None for record in once.run_records)

    once_group_begin = cast(llir.VarInit, once.artifact.value[1])
    twice_group_begin = cast(llir.VarInit, twice.artifact.value[1])
    assert cast(llir.Var, once_group_begin.value).name == "_group_starts.at(_g)"
    assert cast(llir.Var, twice_group_begin.value).name == "_group_starts.at(_g)"
    once_initializer = cast(llir.VarInit, once.artifact.value[2])
    twice_initializer = cast(llir.VarInit, twice.artifact.value[2])
    assert once_initializer.value == source_read
    assert twice_initializer.value == source_read
    assert type(once_initializer.value) is llir.ArrayAccess
    once_read = cast(llir.ArrayAccess, once_initializer.value)
    assert type(once_read.array) is llir.Var
    assert cast(llir.Var, once_read.array).type is llir.DataType.PTR_INT
    assert type(once_read.index) is llir.Var
    assert cast(llir.Var, once_read.index).type is llir.DataType.INT64
    assert once_read.tensor_access is None


def test_all_coo_end_bound_survives_repeated_detached_production_pipeline() -> None:
    compile_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(compile_options)
    source_bound = llir.Add(
        _var("pMask0", llir.DataType.INT64),
        llir.Literal(1, data_type=llir.DataType.INT64),
    )
    source: List[llir.Stmt] = [
        llir.VarInit(
            _var("pMask1_end", llir.DataType.INT64),
            source_bound,
        )
    ]

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assert compressed_output_parallel is False
        return LLIRRewriteArtifact(artifact.statements)

    once = manager.run_production_pipeline(
        LLIRStatementListArtifact(source),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )
    twice = manager.run_production_pipeline(
        LLIRStatementListArtifact(once.artifact.value),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )

    assert _structural_snapshot(source) == _structural_snapshot(once.artifact.value)
    assert _structural_snapshot(once.artifact.value) == _structural_snapshot(
        twice.artifact.value
    )
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(once.artifact.value))
    assert _mutable_ir_ids(once.artifact.value).isdisjoint(
        _mutable_ir_ids(twice.artifact.value)
    )
    expected_passes = [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    for result in (once, twice):
        assert [record.pass_name for record in result.run_records] == expected_passes
        assert [record.sequence_index for record in result.run_records] == list(
            range(5)
        )
        assert all(record.duration_ns is not None for record in result.run_records)

    once_initializer = cast(llir.VarInit, once.artifact.value[0])
    twice_initializer = cast(llir.VarInit, twice.artifact.value[0])
    assert once_initializer.value == source_bound
    assert twice_initializer.value == source_bound
    assert hash(cast(llir.Add, once_initializer.value)) == hash(source_bound)
    assert type(once_initializer.value) is llir.Add
    assert type(twice_initializer.value) is llir.Add
    once_bound = cast(llir.Add, once_initializer.value)
    twice_bound = cast(llir.Add, twice_initializer.value)
    assert type(once_bound.left) is llir.Var
    assert cast(llir.Var, once_bound.left).name == "pMask0"
    assert cast(llir.Var, once_bound.left).type is llir.DataType.INT64
    assert cast(llir.Var, once_bound.left).tensor_access is None
    assert type(once_bound.right) is llir.Literal
    assert cast(llir.Literal, once_bound.right).value == 1
    assert cast(llir.Literal, once_bound.right).data_type is llir.DataType.INT64
    assert once_bound is not source_bound
    assert once_bound.left is not source_bound.left
    assert once_bound.right is not source_bound.right
    assert twice_bound is not once_bound
    assert twice_bound.left is not once_bound.left
    assert twice_bound.right is not once_bound.right


def test_mode_iterator_position_bounds_and_coordinate_read_activate_managed_prefetch() -> (
    None
):
    compile_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(compile_options)
    source_begin = llir.ArrayAccess(
        _var("A1_pos", llir.DataType.PTR_INT),
        _var("pA0", llir.DataType.INT),
    )
    source_end = llir.ArrayAccess(
        _var("A1_pos", llir.DataType.PTR_INT),
        llir.Add(
            _var("pA0", llir.DataType.INT),
            llir.Literal(1, llir.DataType.INT),
        ),
    )
    source_read = llir.ArrayAccess(
        _var("A1_crd", llir.DataType.PTR_INT),
        _var("pA1", llir.DataType.INT),
    )
    sparse_loop = llir.ForLoop(
        init=llir.VarInit(
            _var("pA1", llir.DataType.INT),
            source_begin,
        ),
        cond=llir.BinOp(
            "<",
            _var("pA1", llir.DataType.INT),
            _var("pA1_end", llir.DataType.INT),
        ),
        update=llir.Increment(_var("pA1", llir.DataType.INT)),
        body=[
            llir.VarInit(_var("j", llir.DataType.INT), source_read),
            llir.ForLoop(
                init=llir.VarInit(
                    _var("k", llir.DataType.INT),
                    llir.Literal(0),
                ),
                cond=llir.BinOp(
                    "<",
                    _var("k", llir.DataType.INT),
                    _var("B1_size", llir.DataType.INT),
                ),
                update=llir.Increment(_var("k", llir.DataType.INT)),
                body=[
                    llir.VarInit(
                        _var("pB1", llir.DataType.INT),
                        llir.Add(
                            llir.Mul(
                                _var("j", llir.DataType.INT),
                                _var("B1_size", llir.DataType.INT),
                            ),
                            _var("k", llir.DataType.INT),
                        ),
                    ),
                    llir.Assign(
                        _access("C_val", "pC1"),
                        _var("B_val[pB1]"),
                    ),
                ],
            ),
        ],
    )
    source: List[llir.Stmt] = [
        llir.VarInit(_var("pA1_end", llir.DataType.INT), source_end),
        sparse_loop,
    ]

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assert compressed_output_parallel is False
        return LLIRRewriteArtifact(artifact.statements)

    once = manager.run_production_pipeline(
        LLIRStatementListArtifact(source),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )
    twice = manager.run_production_pipeline(
        LLIRStatementListArtifact(once.artifact.value),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )

    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(once.artifact.value))
    assert _mutable_ir_ids(once.artifact.value).isdisjoint(
        _mutable_ir_ids(twice.artifact.value)
    )
    expected_passes = [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    for result in (once, twice):
        assert [record.pass_name for record in result.run_records] == expected_passes
        assert [record.sequence_index for record in result.run_records] == list(
            range(5)
        )
        assert all(record.duration_ns is not None for record in result.run_records)

    once_end_init = cast(llir.VarInit, once.artifact.value[0])
    twice_end_init = cast(llir.VarInit, twice.artifact.value[0])
    once_loop = cast(llir.ForLoop, once.artifact.value[1])
    twice_loop = cast(llir.ForLoop, twice.artifact.value[1])
    once_prefetches = [
        cast(llir.RawStmt, statement).code
        for statement in once_loop.body
        if type(statement) is llir.RawStmt
    ]
    twice_prefetches = [
        cast(llir.RawStmt, statement).code
        for statement in twice_loop.body
        if type(statement) is llir.RawStmt
    ]
    expected_prefetch = (
        "if (pA1 + 1 < pA1_end) "
        "__builtin_prefetch(&B_val[A1_crd[pA1 + 1] * B1_size], 0, 1)"
    )
    assert once_prefetches == [expected_prefetch]
    assert twice_prefetches == [expected_prefetch, expected_prefetch]

    once_initializer = cast(llir.VarInit, once_loop.body[1])
    twice_initializer = cast(llir.VarInit, twice_loop.body[2])
    assert once_initializer.value == source_read
    assert twice_initializer.value == source_read
    assert once_initializer.value is not source_read
    assert twice_initializer.value is not once_initializer.value
    once_access = cast(llir.ArrayAccess, once_initializer.value)
    twice_access = cast(llir.ArrayAccess, twice_initializer.value)
    assert once_access.array is not source_read.array
    assert once_access.index is not source_read.index
    assert twice_access.array is not once_access.array
    assert twice_access.index is not once_access.index

    once_begin = cast(llir.ArrayAccess, cast(llir.VarInit, once_loop.init).value)
    twice_begin = cast(llir.ArrayAccess, cast(llir.VarInit, twice_loop.init).value)
    once_end = cast(llir.ArrayAccess, once_end_init.value)
    twice_end = cast(llir.ArrayAccess, twice_end_init.value)
    assert once_begin == source_begin == twice_begin
    assert once_end == source_end == twice_end
    assert once_begin is not source_begin
    assert twice_begin is not once_begin
    assert once_end is not source_end
    assert twice_end is not once_end
    assert once_begin.array is not source_begin.array
    assert once_begin.index is not source_begin.index
    assert twice_begin.array is not once_begin.array
    assert twice_begin.index is not once_begin.index
    assert once_end.array is not source_end.array
    assert once_end.index is not source_end.index
    assert twice_end.array is not once_end.array
    assert twice_end.index is not once_end.index
    source_add = cast(llir.Add, source_end.index)
    once_add = cast(llir.Add, once_end.index)
    twice_add = cast(llir.Add, twice_end.index)
    assert once_add.left is not source_add.left
    assert once_add.right is not source_add.right
    assert twice_add.left is not once_add.left
    assert twice_add.right is not once_add.right


def test_malformed_workspace_pair_read_stops_the_production_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = object.__new__(llir.MemberAccess)
    object.__setattr__(malformed, "base", _var("it"))
    object.__setattr__(malformed, "member", "first[0]")
    source = [llir.VarInit(_var("column"), malformed)]
    later_calls: List[str] = []

    def unexpected_dense_pointer_hoist(
        statements: List[llir.Stmt], context: DensePointerHoistContext
    ) -> List[llir.Stmt]:
        del statements, context
        later_calls.append("dense_pointer_hoist")
        raise AssertionError("later pass ran after malformed MemberAccess")

    monkeypatch.setattr(
        pass_manager_module,
        "hoist_dense_pointers",
        unexpected_dense_pointer_hoist,
    )
    manager = LLIRPassManager.from_compile_options(_compile_options())

    with pytest.raises(LLIRPassPartialFailure) as raised:
        manager.run_production_pipeline(
            LLIRStatementListArtifact(source),
            compressed_where_pass_spec=None,
            dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
            body_assembler=lambda artifact, _: LLIRRewriteArtifact(artifact.statements),
        )

    assert type(raised.value.failure) is LLIRTraversalError
    failure = cast(LLIRTraversalError, raised.value.failure)
    assert failure.diagnostic.code == "invalid_member_access_member"
    assert failure.diagnostic.path == (
        "root",
        "[0]",
        "value",
        "member",
    )
    assert failure.diagnostic.pass_name == "insert_sparse_prefetch"
    assert raised.value.completed_run_records == ()
    assert later_calls == []


def test_malformed_function_call_after_assembly_preserves_exact_pass_prefix() -> None:
    source = [llir.BlankLine()]
    source_snapshot = _structural_snapshot(source)
    malformed = object.__new__(llir.FunctionCall)
    object.__setattr__(malformed, "name", "std::move")
    object.__setattr__(
        malformed,
        "args",
        [_var("values", llir.DataType.STD_VECTOR_FLOAT32)],
    )

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assert compressed_output_parallel is False
        return LLIRRewriteArtifact(
            [
                *artifact.statements,
                llir.VarInit(_var("moved"), malformed),
            ]
        )

    manager = LLIRPassManager.from_compile_options(_compile_options())
    with pytest.raises(LLIRPassPartialFailure) as raised:
        manager.run_production_pipeline(
            LLIRStatementListArtifact(source),
            compressed_where_pass_spec=None,
            dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
            body_assembler=assemble_body,
        )

    assert type(raised.value.failure) is LLIRTraversalError
    failure = cast(LLIRTraversalError, raised.value.failure)
    assert failure.diagnostic.code == "invalid_function_call_args"
    assert failure.diagnostic.path == ("root", "[1]", "value", "args")
    assert failure.diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert [record.pass_name for record in raised.value.completed_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]
    assert [record.sequence_index for record in raised.value.completed_run_records] == [
        0,
        1,
        2,
        3,
    ]
    assert _structural_snapshot(source) == source_snapshot


def test_malformed_member_call_after_assembly_preserves_exact_pass_prefix() -> None:
    source = [llir.BlankLine()]
    source_snapshot = _structural_snapshot(source)
    malformed = object.__new__(llir.MemberCall)
    object.__setattr__(
        malformed,
        "base",
        _var("values_torch", llir.DataType.TORCH_TENSOR),
    )
    object.__setattr__(malformed, "member", "data_ptr")
    object.__setattr__(malformed, "template_args", (llir.DataType.FLOAT32,))
    object.__setattr__(malformed, "args", [])

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assert compressed_output_parallel is False
        return LLIRRewriteArtifact(
            [
                *artifact.statements,
                llir.VarInit(_var("values"), malformed),
            ]
        )

    manager = LLIRPassManager.from_compile_options(_compile_options())
    with pytest.raises(LLIRPassPartialFailure) as raised:
        manager.run_production_pipeline(
            LLIRStatementListArtifact(source),
            compressed_where_pass_spec=None,
            dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
            body_assembler=assemble_body,
        )

    assert type(raised.value.failure) is LLIRTraversalError
    failure = cast(LLIRTraversalError, raised.value.failure)
    assert failure.diagnostic.code == "invalid_member_call_args"
    assert failure.diagnostic.path == ("root", "[1]", "value", "args")
    assert failure.diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert [record.pass_name for record in raised.value.completed_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]
    assert [record.sequence_index for record in raised.value.completed_run_records] == [
        0,
        1,
        2,
        3,
    ]
    assert _structural_snapshot(source) == source_snapshot


def test_malformed_fixed_stack_array_after_assembly_preserves_pass_prefix() -> None:
    source = [llir.BlankLine()]
    source_snapshot = _structural_snapshot(source)
    malformed = object.__new__(llir.FixedStackArrayDecl)
    object.__setattr__(malformed, "name", "wksp")
    object.__setattr__(malformed, "element_type", llir.DataType.FLOAT32)
    object.__setattr__(malformed, "extent", _var("runtime_extent", llir.DataType.INT))
    object.__setattr__(
        malformed,
        "initializer",
        llir.Array([], llir.DataType.FLOAT32),
    )

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assert compressed_output_parallel is False
        return LLIRRewriteArtifact([*artifact.statements, malformed])

    manager = LLIRPassManager.from_compile_options(_compile_options())
    with pytest.raises(LLIRPassPartialFailure) as raised:
        manager.run_production_pipeline(
            LLIRStatementListArtifact(source),
            compressed_where_pass_spec=None,
            dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
            body_assembler=assemble_body,
        )

    assert type(raised.value.failure) is LLIRTraversalError
    failure = cast(LLIRTraversalError, raised.value.failure)
    assert failure.diagnostic.code == "invalid_fixed_stack_array_extent"
    assert failure.diagnostic.path == ("root", "[1]", "extent")
    assert failure.diagnostic.stage == "LLIR rewrite"
    assert failure.diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert [record.pass_name for record in raised.value.completed_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
    ]
    assert [
        record.configuration_name for record in raised.value.completed_run_records
    ] == [
        "sparse_prefetch",
        "dense_pointer_hoist",
        "single_iteration_loop_elimination",
        "loop_invariant_factor_hoist",
    ]
    assert [record.sequence_index for record in raised.value.completed_run_records] == [
        0,
        1,
        2,
        3,
    ]
    assert all(
        record.duration_ns is not None for record in raised.value.completed_run_records
    )
    assert _structural_snapshot(source) == source_snapshot


def test_production_pipeline_rejects_a_detached_compressed_options_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager_options = _compile_options()
    detached_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(manager_options)
    nested_calls: List[str] = []

    def unexpected_result_write(
        self: LLIRPassManager,
        artifact: LLIRRewriteArtifact[object],
        pass_spec: ResultWritePassSpec,
    ) -> NoReturn:
        del self, artifact
        nested_calls.append(pass_spec.context.mode)
        raise AssertionError("mixed snapshots reached a nested pass")

    monkeypatch.setattr(
        LLIRPassManager,
        "run_result_write",
        unexpected_result_write,
    )
    with pytest.raises(LLIRPassManagerError) as error:
        manager.run_production_pipeline(
            LLIRStatementListArtifact(_compressed_source()),
            compressed_where_pass_spec=CompressedWhereOpenMPPassSpec(
                replace(_compressed_context(), compile_options=detached_options)
            ),
            dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
            body_assembler=lambda artifact, _: LLIRRewriteArtifact(artifact.statements),
        )

    assert manager_options == detached_options
    assert manager_options is not detached_options
    assert error.value.diagnostic.code == "detached_compile_options"
    assert error.value.diagnostic.pass_name == COMPRESSED_WHERE_OPENMP_PASS.name
    assert nested_calls == []


def test_production_pipeline_rejects_malformed_compressed_context_before_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = LLIRPassManager.from_compile_options(_compile_options())
    orchestration_calls: List[str] = []

    def unexpected_compressed_pass(
        self: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: CompressedWhereOpenMPPassSpec,
    ) -> NoReturn:
        del self, artifact, pass_spec
        orchestration_calls.append("compressed_where_openmp")
        raise AssertionError("malformed context reached production work")

    monkeypatch.setattr(
        LLIRPassManager,
        "run_compressed_where_openmp",
        unexpected_compressed_pass,
    )
    malformed_spec = CompressedWhereOpenMPPassSpec(
        cast(CompressedWhereOpenMPContext, object())
    )
    with pytest.raises(LLIRPassManagerError) as error:
        manager.run_production_pipeline(
            LLIRStatementListArtifact(_compressed_source()),
            compressed_where_pass_spec=malformed_spec,
            dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
            body_assembler=lambda artifact, _: LLIRRewriteArtifact(artifact.statements),
        )

    assert error.value.diagnostic.code == "invalid_pass_context"
    assert error.value.diagnostic.pass_name == COMPRESSED_WHERE_OPENMP_PASS.name
    assert orchestration_calls == []


@pytest.mark.parametrize(
    ("pass_options", "verified"),
    [
        pytest.param(PRODUCTION_LLIR_PASS_OPTIONS, False, id="production"),
        pytest.param(DEBUG_LLIR_PASS_OPTIONS, True, id="debug"),
    ],
)
def test_production_pipeline_preserves_applied_compressed_order_and_policy(
    monkeypatch: pytest.MonkeyPatch,
    pass_options: LLIRPassOptions,
    verified: bool,
) -> None:
    compile_options = _compile_options(pass_options)
    manager = LLIRPassManager.from_compile_options(compile_options)
    nested_managers: List[LLIRPassManager] = []
    nested_source_ids: List[int] = []
    nested_modes: List[str] = []
    assembly_modes: List[bool] = []
    assembled_offset_family: List[List[llir.Stmt]] = []
    original_result_write = LLIRPassManager.run_result_write

    def observe_result_write(
        self: LLIRPassManager,
        artifact: LLIRRewriteArtifact[object],
        pass_spec: ResultWritePassSpec,
    ) -> LLIRRewritePassResult[object]:
        nested_managers.append(self)
        nested_source_ids.append(id(artifact.value))
        nested_modes.append(pass_spec.context.mode)
        assert pass_spec.context.compile_options is compile_options
        return original_result_write(self, artifact, pass_spec)

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assembly_modes.append(compressed_output_parallel)
        family = _compressed_offset_family(artifact.statements)
        assert [type(statement) for statement in family] == [
            llir.DirectInit,
            llir.DirectInit,
            llir.Assign,
            llir.ForLoop,
            llir.VarInit,
        ]
        assert [
            cast(llir.DirectInit, statement).var.name for statement in family[:2]
        ] == ["_count1", "_offset1"]
        assert [
            len(cast(llir.DirectInit, statement).args) for statement in family[:2]
        ] == [2, 1]
        prefix_loop = cast(llir.ForLoop, family[3])
        assert len(prefix_loop.body) == 1
        assert type(prefix_loop.body[0]) is llir.Assign
        assembled_offset_family.append(family)
        return LLIRRewriteArtifact(artifact.statements)

    monkeypatch.setattr(
        LLIRPassManager,
        "run_result_write",
        observe_result_write,
    )
    result = manager.run_production_pipeline(
        LLIRStatementListArtifact(_compressed_source()),
        compressed_where_pass_spec=CompressedWhereOpenMPPassSpec(
            replace(_compressed_context(), compile_options=compile_options)
        ),
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )

    assert nested_managers == [manager, manager]
    assert nested_modes == ["count", "fill"]
    assert len(nested_source_ids) == 2
    assert nested_source_ids[0] == nested_source_ids[1]
    assert assembly_modes == [True]
    assert [record.pass_name for record in result.run_records] == [
        "transform_compressed_where_for_openmp",
        "rewrite_result_writes",
        "rewrite_result_writes",
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.configuration_name for record in result.run_records] == [
        "compressed_where_openmp",
        "count",
        "fill",
        "sparse_prefetch",
        "dense_pointer_hoist",
        "single_iteration_loop_elimination",
        "loop_invariant_factor_hoist",
        "dynamic_vector_access",
    ]
    assert [record.sequence_index for record in result.run_records] == list(range(8))
    assert all(record.verified_before is verified for record in result.run_records)
    assert all(record.verified_after is verified for record in result.run_records)
    assert all(record.duration_ns is not None for record in result.run_records)
    assert result.compressed_output_parallel is True
    assert len(assembled_offset_family) == 1
    final_offset_family = _compressed_offset_family(result.artifact.value)
    assert _structural_snapshot(final_offset_family) == _structural_snapshot(
        assembled_offset_family[0]
    )
    assert _mutable_ir_ids(final_offset_family).isdisjoint(
        _mutable_ir_ids(assembled_offset_family[0])
    )


def test_production_pipeline_nested_fill_failure_preserves_count_and_stops_later_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(compile_options)
    nested_managers: List[LLIRPassManager] = []
    nested_source_ids: List[int] = []
    nested_modes: List[str] = []
    later_work: List[str] = []
    original_result_write = LLIRPassManager.run_result_write
    failure = LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code="synthetic_pipeline_fill_failure",
            message="fill failed in production orchestration",
            path=("root",),
            node_type="RawStmt",
            stage="LLIR transformation",
            pass_name="transform_compressed_where_for_openmp",
        )
    )

    def fail_fill(
        self: LLIRPassManager,
        artifact: LLIRRewriteArtifact[object],
        pass_spec: ResultWritePassSpec,
    ) -> LLIRRewritePassResult[object]:
        nested_managers.append(self)
        nested_source_ids.append(id(artifact.value))
        nested_modes.append(pass_spec.context.mode)
        if pass_spec.context.mode == "fill":
            raise failure
        return original_result_write(self, artifact, pass_spec)

    def unexpected_later_transform(source: object, context: object) -> NoReturn:
        del source, context
        later_work.append("pass")
        raise AssertionError("later production pass executed after nested failure")

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        del compressed_output_parallel
        later_work.append("body_assembler")
        return LLIRRewriteArtifact(artifact.statements)

    monkeypatch.setattr(LLIRPassManager, "run_result_write", fail_fill)
    for function_name in (
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ):
        monkeypatch.setattr(
            pass_manager_module,
            function_name,
            unexpected_later_transform,
        )

    with pytest.raises(LLIRPassPartialFailure) as error:
        manager.run_production_pipeline(
            LLIRStatementListArtifact(_compressed_source()),
            compressed_where_pass_spec=CompressedWhereOpenMPPassSpec(
                replace(_compressed_context(), compile_options=compile_options)
            ),
            dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
            body_assembler=assemble_body,
        )

    assert error.value.failure is failure
    assert nested_managers == [manager, manager]
    assert nested_modes == ["count", "fill"]
    assert len(nested_source_ids) == 2
    assert nested_source_ids[0] == nested_source_ids[1]
    assert later_work == []
    assert [
        (record.pass_name, record.configuration_name, record.sequence_index)
        for record in error.value.completed_run_records
    ] == [("rewrite_result_writes", "count", 0)]


def test_workspace_insert_name_failure_stops_pipeline_before_record_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(compile_options)
    malformed = llir.FunctionCallStmt(cast(str, object()), [_var("value")])
    source = [
        _compatible_loop(
            [
                llir.VarInit(
                    _var("wksp", llir.DataType.AUTO),
                    llir.FunctionCall(
                        "coo_workspace_1d<float, 1>", [llir.Literal(1024)]
                    ),
                ),
                malformed,
            ]
        )
    ]
    source_snapshot = _structural_snapshot(source)
    observed_failures: List[LLIRTraversalError] = []
    result_write_modes: List[str] = []
    later_work: List[str] = []
    original_compressed = LLIRPassManager.run_compressed_where_openmp

    def capture_compressed_failure(
        self: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        pass_spec: CompressedWhereOpenMPPassSpec,
    ) -> ManagedCompressedWhereOpenMPResult:
        try:
            return original_compressed(self, artifact, pass_spec)
        except LLIRTraversalError as failure:
            observed_failures.append(failure)
            raise

    def unexpected_result_write(
        self: LLIRPassManager,
        artifact: LLIRRewriteArtifact[object],
        pass_spec: ResultWritePassSpec,
    ) -> NoReturn:
        del self, artifact
        result_write_modes.append(pass_spec.context.mode)
        raise AssertionError("result-write work ran after workspace-name failure")

    def unexpected_later_transform(source: object, context: object) -> NoReturn:
        del source, context
        later_work.append("pass")
        raise AssertionError("later production pass ran after workspace-name failure")

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        del artifact, compressed_output_parallel
        later_work.append("body_assembler")
        raise AssertionError("body assembly ran after workspace-name failure")

    monkeypatch.setattr(
        LLIRPassManager,
        "run_compressed_where_openmp",
        capture_compressed_failure,
    )
    monkeypatch.setattr(LLIRPassManager, "run_result_write", unexpected_result_write)
    for function_name in (
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ):
        monkeypatch.setattr(
            pass_manager_module,
            function_name,
            unexpected_later_transform,
        )

    with pytest.raises(LLIRPassPartialFailure) as error:
        manager.run_production_pipeline(
            LLIRStatementListArtifact(source),
            compressed_where_pass_spec=CompressedWhereOpenMPPassSpec(
                replace(_compressed_context(), compile_options=compile_options)
            ),
            dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
            body_assembler=assemble_body,
        )

    assert len(observed_failures) == 1
    assert error.value.failure is observed_failures[0]
    diagnostic = observed_failures[0].diagnostic
    assert diagnostic.code == "invalid_workspace_insert_call_name"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "transform_compressed_where_for_openmp"
    # The workspace rewriter consumes the filtered, re-rooted work-body artifact.
    assert diagnostic.path == ("root", "[0]", "name")
    assert error.value.completed_run_records == ()
    assert result_write_modes == []
    assert later_work == []
    assert _structural_snapshot(source) == source_snapshot


def test_malformed_generated_phase_state_fails_owner_before_record_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(compile_options)
    original_phase_state = result_write_module._ResultWriteRewriter._phase_state
    later_work: List[str] = []

    def malformed_count_state(prefix: str, level: int) -> llir.Var:
        state = original_phase_state(prefix, level)
        if prefix == "_cnt":
            state.tensor_access = cast(llir.TensorAccessMetadata, "malformed")
        return state

    def unexpected_later_transform(source: object, context: object) -> NoReturn:
        del source, context
        later_work.append("pass")
        raise AssertionError("later production pass executed after state failure")

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        del compressed_output_parallel
        later_work.append("body_assembler")
        return LLIRRewriteArtifact(artifact.statements)

    monkeypatch.setattr(
        result_write_module._ResultWriteRewriter,
        "_phase_state",
        staticmethod(malformed_count_state),
    )
    for function_name in (
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ):
        monkeypatch.setattr(
            pass_manager_module,
            function_name,
            unexpected_later_transform,
        )

    with pytest.raises(LLIRPassPartialFailure) as error:
        manager.run_production_pipeline(
            LLIRStatementListArtifact(_compressed_source()),
            compressed_where_pass_spec=CompressedWhereOpenMPPassSpec(
                replace(_compressed_context(), compile_options=compile_options)
            ),
            dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
            body_assembler=assemble_body,
        )

    failure = cast(LLIRTraversalError, error.value.failure)
    diagnostic = failure.diagnostic
    assert diagnostic.code == "invalid_tensor_access_metadata"
    assert diagnostic.stage == "LLIR transformation"
    assert diagnostic.pass_name == "transform_compressed_where_for_openmp"
    assert diagnostic.path[-2:] == ("var", "tensor_access")
    assert error.value.completed_run_records == ()
    assert later_work == []


@pytest.mark.parametrize(
    ("failing_stage", "completed_pass_names"),
    [
        pytest.param("sparse_prefetch", (), id="sparse-prefetch"),
        pytest.param(
            "dense_pointer_hoist",
            ("insert_sparse_prefetch",),
            id="dense-pointer-hoist",
        ),
        pytest.param(
            "single_iteration_loop_elimination",
            ("insert_sparse_prefetch", "hoist_dense_pointers"),
            id="single-iteration",
        ),
        pytest.param(
            "loop_invariant_factor_hoist",
            (
                "insert_sparse_prefetch",
                "hoist_dense_pointers",
                "eliminate_single_iteration_loops",
            ),
            id="loop-invariant-factor",
        ),
        pytest.param(
            "body_assembler",
            (
                "insert_sparse_prefetch",
                "hoist_dense_pointers",
                "eliminate_single_iteration_loops",
                "hoist_loop_invariant_factors",
            ),
            id="body-assembler",
        ),
        pytest.param(
            "dynamic_vector_access",
            (
                "insert_sparse_prefetch",
                "hoist_dense_pointers",
                "eliminate_single_iteration_loops",
                "hoist_loop_invariant_factors",
            ),
            id="dynamic-vector-access",
        ),
    ],
)
def test_each_production_pipeline_failure_stops_later_positions(
    monkeypatch: pytest.MonkeyPatch,
    failing_stage: str,
    completed_pass_names: Tuple[str, ...],
) -> None:
    manager = LLIRPassManager.from_compile_options(_compile_options())
    calls: List[str] = []
    failure: Exception
    if failing_stage == "body_assembler":
        failure = AssertionError("synthetic body-assembler failure")
    else:
        failure = LLIRTraversalError(
            LLIRTraversalDiagnostic(
                code=f"synthetic_{failing_stage}_failure",
                message=f"{failing_stage} failed",
                path=("root",),
                node_type="BlankLine",
                stage="LLIR production pipeline",
                pass_name=failing_stage,
            )
        )

    def instrument_transform(
        stage: str,
        implementation: object,
    ) -> Callable[[List[llir.Stmt], object], List[llir.Stmt]]:
        typed_implementation = cast(
            Callable[[List[llir.Stmt], object], List[llir.Stmt]],
            implementation,
        )

        def instrumented(source: List[llir.Stmt], context: object) -> List[llir.Stmt]:
            calls.append(stage)
            if stage == failing_stage:
                raise failure
            return typed_implementation(source, context)

        return instrumented

    for stage, function_name in (
        ("sparse_prefetch", "insert_sparse_prefetch"),
        ("dense_pointer_hoist", "hoist_dense_pointers"),
        (
            "single_iteration_loop_elimination",
            "eliminate_single_iteration_loops",
        ),
        ("loop_invariant_factor_hoist", "hoist_loop_invariant_factors"),
        ("dynamic_vector_access", "rewrite_dynamic_vector_accesses"),
    ):
        original = getattr(pass_manager_module, function_name)
        monkeypatch.setattr(
            pass_manager_module,
            function_name,
            instrument_transform(stage, original),
        )

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assert compressed_output_parallel is False
        calls.append("body_assembler")
        if failing_stage == "body_assembler":
            raise failure
        return LLIRRewriteArtifact(artifact.statements)

    production_order = (
        "sparse_prefetch",
        "dense_pointer_hoist",
        "single_iteration_loop_elimination",
        "loop_invariant_factor_hoist",
        "body_assembler",
        "dynamic_vector_access",
    )
    with pytest.raises(LLIRPassPartialFailure) as error:
        manager.run_production_pipeline(
            LLIRStatementListArtifact([llir.BlankLine()]),
            compressed_where_pass_spec=None,
            dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
            body_assembler=assemble_body,
        )

    failure_index = production_order.index(failing_stage)
    assert calls == list(production_order[: failure_index + 1])
    assert error.value.failure is failure
    assert (
        tuple(record.pass_name for record in error.value.completed_run_records)
        == completed_pass_names
    )
    assert [
        record.sequence_index for record in error.value.completed_run_records
    ] == list(range(len(completed_pass_names)))


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
        llir.Assign(_access("out_values", "p"), _var("value")),
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


def test_dynamic_runner_detaches_structured_initializer_list_arguments() -> None:
    source: List[llir.Stmt] = [
        llir.VarInit(
            _var("Result_values_torch", llir.DataType.TORCH_TENSOR),
            llir.FunctionCall(
                "torch::empty",
                (
                    llir.Array(
                        (_var("Result_capacity", llir.DataType.INT64),),
                        llir.DataType.INT64,
                    ),
                    llir.QualifiedName(
                        "torch",
                        "kFloat32",
                        llir.DataType.TORCH_SCALAR_TYPE,
                    ),
                ),
            ),
        )
    ]
    manager = LLIRPassManager()

    once = manager.run_dynamic_vector_access(LLIRRewriteArtifact(source))
    twice = manager.run_dynamic_vector_access(once.artifact)
    once_initializer = cast(llir.VarInit, cast(List[llir.Stmt], once.artifact.value)[0])
    twice_initializer = cast(
        llir.VarInit,
        cast(List[llir.Stmt], twice.artifact.value)[0],
    )
    original_call = cast(llir.FunctionCall, cast(llir.VarInit, source[0]).value)
    once_call = cast(llir.FunctionCall, once_initializer.value)
    twice_call = cast(llir.FunctionCall, twice_initializer.value)
    original_array = cast(llir.Array, original_call.args[0])
    once_array = cast(llir.Array, once_call.args[0])
    twice_array = cast(llir.Array, twice_call.args[0])
    original_dtype = cast(llir.QualifiedName, original_call.args[1])
    once_dtype = cast(llir.QualifiedName, once_call.args[1])
    twice_dtype = cast(llir.QualifiedName, twice_call.args[1])

    assert original_array == once_array == twice_array
    assert original_array is not once_array
    assert once_array is not twice_array
    assert original_array.values[0] is not once_array.values[0]
    assert once_array.values[0] is not twice_array.values[0]
    assert original_dtype == once_dtype == twice_dtype
    assert original_dtype is not once_dtype
    assert once_dtype is not twice_dtype
    assert once_dtype.namespace == "torch"
    assert once_dtype.name == "kFloat32"
    assert once_dtype.data_type is llir.DataType.TORCH_SCALAR_TYPE
    cast(llir.Var, once_array.values[0]).name = "owned"
    assert cast(llir.Var, original_array.values[0]).name == "Result_capacity"
    assert cast(llir.Var, twice_array.values[0]).name == "Result_capacity"


def test_qualified_dtype_survives_repeated_production_pipeline_detached() -> None:
    compile_options = _compile_options()
    manager = LLIRPassManager.from_compile_options(compile_options)
    source: List[llir.Stmt] = [
        llir.VarInit(
            _var("tensor", llir.DataType.TORCH_TENSOR),
            llir.FunctionCall(
                "scorch_tensor_from_vector",
                (
                    llir.FunctionCall("std::move", (_var("values"),)),
                    llir.QualifiedName(
                        "torch",
                        "kFloat32",
                        llir.DataType.TORCH_SCALAR_TYPE,
                    ),
                ),
            ),
        )
    ]

    def assemble_body(
        artifact: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact[List[llir.Stmt]]:
        assert compressed_output_parallel is False
        return LLIRRewriteArtifact(artifact.statements)

    once = manager.run_production_pipeline(
        LLIRStatementListArtifact(source),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )
    twice = manager.run_production_pipeline(
        LLIRStatementListArtifact(once.artifact.value),
        compressed_where_pass_spec=None,
        dense_pointer_pass_spec=DensePointerHoistPassSpec(_dense_context()),
        body_assembler=assemble_body,
    )

    def dtype_leaf(value: object) -> llir.QualifiedName:
        initializer = cast(llir.VarInit, cast(List[llir.Stmt], value)[0])
        call = cast(llir.FunctionCall, initializer.value)
        return cast(llir.QualifiedName, call.args[1])

    original_dtype = dtype_leaf(source)
    once_dtype = dtype_leaf(once.artifact.value)
    twice_dtype = dtype_leaf(twice.artifact.value)
    assert original_dtype == once_dtype == twice_dtype
    assert original_dtype is not once_dtype
    assert once_dtype is not twice_dtype
    assert [record.pass_name for record in once.run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.sequence_index for record in once.run_records] == list(range(5))
    assert all(record.duration_ns >= 0 for record in once.run_records)


def test_result_count_and_fill_are_independent_not_a_linear_pipeline() -> None:
    source: List[llir.Stmt] = [
        llir.Assign(
            _access(
                "Result_values",
                "pResult1",
                metadata=_result_metadata(),
            ),
            _var("value"),
        ),
        llir.Assign(_access("Result1_crd", "pResult1"), _var("coordinate")),
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


def test_compressed_fill_failure_carries_only_completed_count_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modes: List[str] = []
    failure = LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code="synthetic_fill_failure",
            message="fill failed",
            path=("root",),
            node_type="RawStmt",
            stage="LLIR transformation",
            pass_name="transform_compressed_where_for_openmp",
        )
    )
    original_result_write = LLIRPassManager.run_result_write

    def fail_fill(
        self: LLIRPassManager,
        artifact: LLIRRewriteArtifact[object],
        pass_spec: ResultWritePassSpec,
    ) -> LLIRRewritePassResult[object]:
        modes.append(pass_spec.context.mode)
        if pass_spec.context.mode == "fill":
            raise failure
        return original_result_write(self, artifact, pass_spec)

    monkeypatch.setattr(LLIRPassManager, "run_result_write", fail_fill)

    with pytest.raises(LLIRPassPartialFailure) as error:
        LLIRPassManager().run_compressed_where_openmp(
            LLIRStatementListArtifact(_compressed_source()),
            CompressedWhereOpenMPPassSpec(_compressed_context()),
        )

    assert error.value.failure is failure
    assert modes == ["count", "fill"]
    assert [
        (record.pass_name, record.configuration_name, record.sequence_index)
        for record in error.value.completed_run_records
    ] == [("rewrite_result_writes", "count", 0)]


def test_compressed_result_epilogue_failure_carries_count_and_fill_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("synthetic compressed result epilogue failure")

    def fail_result_epilogue(self: ResultTensorAssembler) -> NoReturn:
        del self
        raise failure

    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_storage_epilogue",
        fail_result_epilogue,
    )

    with pytest.raises(LLIRPassPartialFailure) as error:
        LLIRPassManager().run_compressed_where_openmp(
            LLIRStatementListArtifact(_compressed_source()),
            CompressedWhereOpenMPPassSpec(_compressed_context()),
        )

    assert error.value.failure is failure
    assert [
        (record.pass_name, record.configuration_name, record.sequence_index)
        for record in error.value.completed_run_records
    ] == [
        ("rewrite_result_writes", "count", 0),
        ("rewrite_result_writes", "fill", 1),
    ]


def test_compressed_runner_rejects_malformed_nested_records_without_carrying_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_result_write = LLIRPassManager.run_result_write

    def corrupt_record(
        self: LLIRPassManager,
        artifact: LLIRRewriteArtifact[object],
        pass_spec: ResultWritePassSpec,
    ) -> LLIRRewritePassResult[object]:
        result = original_result_write(self, artifact, pass_spec)
        malformed = replace(result.run_records[0], pass_name="not_result_write")
        return replace(result, run_records=(malformed,))

    monkeypatch.setattr(LLIRPassManager, "run_result_write", corrupt_record)

    with pytest.raises(LLIRPassManagerError) as error:
        LLIRPassManager().run_compressed_where_openmp(
            LLIRStatementListArtifact(_compressed_source()),
            CompressedWhereOpenMPPassSpec(_compressed_context()),
        )

    assert error.value.diagnostic.code == "invalid_nested_pass_records"
    assert not isinstance(error.value, LLIRPassPartialFailure)


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

    dense_context = _dense_context()
    with pytest.raises(LLIRTraversalError) as direct_dense:
        hoist_dense_pointers(source, dense_context)
    with pytest.raises(LLIRTraversalError) as managed_dense:
        manager.run_dense_pointer_hoist(
            LLIRStatementListArtifact(source),
            DensePointerHoistPassSpec(dense_context),
        )
    assert managed_dense.value.diagnostic == direct_dense.value.diagnostic

    single_context = SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT
    with pytest.raises(LLIRTraversalError) as direct_single:
        eliminate_single_iteration_loops(source, single_context)
    with pytest.raises(LLIRTraversalError) as managed_single:
        manager.run_single_iteration_loop_elimination(
            LLIRStatementListArtifact(source),
            SingleIterationLoopEliminationPassSpec(single_context),
        )
    assert managed_single.value.diagnostic == direct_single.value.diagnostic

    factor_context = LOOP_INVARIANT_FACTOR_HOIST_CONTEXT
    with pytest.raises(LLIRTraversalError) as direct_factor:
        hoist_loop_invariant_factors(source, factor_context)
    with pytest.raises(LLIRTraversalError) as managed_factor:
        manager.run_loop_invariant_factor_hoist(
            LLIRStatementListArtifact(source),
            LoopInvariantFactorHoistPassSpec(factor_context),
        )
    assert managed_factor.value.diagnostic == direct_factor.value.diagnostic


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
        lambda: manager.run_result_write(
            cast(LLIRRewriteArtifact[object], statement_artifact),
            ResultWritePassSpec(_result_context()),
        ),
        lambda: manager.run_result_write(
            rewrite_artifact,
            cast(ResultWritePassSpec, object()),
        ),
        lambda: manager.run_result_write(
            rewrite_artifact,
            ResultWritePassSpec(context=cast(ResultWriteContext, object())),
        ),
        lambda: manager.run_result_write(
            rewrite_artifact,
            ResultWritePassSpec(
                context=_result_context(),
                descriptor=replace(RESULT_WRITE_PASS, version=2),
            ),
        ),
        lambda: manager.run_compressed_where_openmp(
            cast(LLIRStatementListArtifact, rewrite_artifact),
            CompressedWhereOpenMPPassSpec(_compressed_context()),
        ),
        lambda: manager.run_compressed_where_openmp(
            statement_artifact,
            cast(CompressedWhereOpenMPPassSpec, object()),
        ),
        lambda: manager.run_compressed_where_openmp(
            statement_artifact,
            CompressedWhereOpenMPPassSpec(
                context=cast(CompressedWhereOpenMPContext, object())
            ),
        ),
        lambda: manager.run_compressed_where_openmp(
            statement_artifact,
            CompressedWhereOpenMPPassSpec(
                context=_compressed_context(),
                descriptor=replace(COMPRESSED_WHERE_OPENMP_PASS, version=2),
            ),
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
        lambda: manager.run_dense_pointer_hoist(
            cast(LLIRStatementListArtifact, rewrite_artifact),
            DensePointerHoistPassSpec(_dense_context()),
        ),
        lambda: manager.run_dense_pointer_hoist(
            statement_artifact,
            cast(DensePointerHoistPassSpec, object()),
        ),
        lambda: manager.run_dense_pointer_hoist(
            statement_artifact,
            DensePointerHoistPassSpec(context=cast(DensePointerHoistContext, object())),
        ),
        lambda: manager.run_single_iteration_loop_elimination(
            cast(LLIRStatementListArtifact, rewrite_artifact),
            SingleIterationLoopEliminationPassSpec(),
        ),
        lambda: manager.run_single_iteration_loop_elimination(
            statement_artifact,
            cast(SingleIterationLoopEliminationPassSpec, object()),
        ),
        lambda: manager.run_single_iteration_loop_elimination(
            statement_artifact,
            SingleIterationLoopEliminationPassSpec(
                context=cast(SingleIterationLoopEliminationContext, object())
            ),
        ),
        lambda: manager.run_loop_invariant_factor_hoist(
            cast(LLIRStatementListArtifact, rewrite_artifact),
            LoopInvariantFactorHoistPassSpec(),
        ),
        lambda: manager.run_loop_invariant_factor_hoist(
            statement_artifact,
            cast(LoopInvariantFactorHoistPassSpec, object()),
        ),
        lambda: manager.run_loop_invariant_factor_hoist(
            statement_artifact,
            LoopInvariantFactorHoistPassSpec(
                context=cast(LoopInvariantFactorHoistContext, object())
            ),
        ),
        lambda: manager.run_loop_invariant_factor_hoist(
            statement_artifact,
            LoopInvariantFactorHoistPassSpec(
                descriptor=replace(LOOP_INVARIANT_FACTOR_HOIST_PASS, version=2)
            ),
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
    dense_spec = DensePointerHoistPassSpec(_dense_context())

    production = LLIRPassManager(PRODUCTION_LLIR_PASS_OPTIONS)
    production_dynamic = production.run_dynamic_vector_access(rewrite_artifact)
    production_result = production.run_result_write(rewrite_artifact, result_spec)
    production_sparse_prefetch = production.run_sparse_prefetch(compressed_artifact)
    production_dense = production.run_dense_pointer_hoist(
        compressed_artifact, dense_spec
    )
    production_single = production.run_single_iteration_loop_elimination(
        compressed_artifact
    )
    production_factor = production.run_loop_invariant_factor_hoist(compressed_artifact)
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
            *production_dense.run_records,
            *production_single.run_records,
            *production_factor.run_records,
            *production_compressed.run_records,
        )
    )

    debug = LLIRPassManager(DEBUG_LLIR_PASS_OPTIONS)
    debug_dynamic = debug.run_dynamic_vector_access(rewrite_artifact)
    debug_result = debug.run_result_write(rewrite_artifact, result_spec)
    debug_sparse_prefetch = debug.run_sparse_prefetch(compressed_artifact)
    debug_dense = debug.run_dense_pointer_hoist(compressed_artifact, dense_spec)
    debug_single = debug.run_single_iteration_loop_elimination(compressed_artifact)
    debug_factor = debug.run_loop_invariant_factor_hoist(compressed_artifact)
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
        ("LLIR transformation", "hoist_dense_pointers"),
        ("LLIR transformation", "hoist_dense_pointers"),
        ("LLIR transformation", "eliminate_single_iteration_loops"),
        ("LLIR transformation", "eliminate_single_iteration_loops"),
        ("LLIR transformation", "hoist_loop_invariant_factors"),
        ("LLIR transformation", "hoist_loop_invariant_factors"),
        ("LLIR transformation", "transform_compressed_where_for_openmp"),
        ("LLIR transformation", "transform_compressed_where_for_openmp"),
    ]
    assert all(
        record.verified_before and record.verified_after
        for record in (
            *debug_dynamic.run_records,
            *debug_result.run_records,
            *debug_sparse_prefetch.run_records,
            *debug_dense.run_records,
            *debug_single.run_records,
            *debug_factor.run_records,
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
