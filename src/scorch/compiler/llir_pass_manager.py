"""Minimal typed runners for the six extracted current-LLIR passes.

The production pass graph is intentionally not represented as one linear
pipeline.  Dynamic-vector rewriting is a top-level step over the assembled
function body.  Compressed ``Where`` lowering is a top-level step during outer
loop lowering, and it internally runs independent result-write ``count`` and
``fill`` transformations over the same original work body.  Sparse prefetching
and dense-pointer hoisting and single-iteration-loop elimination accept only the
recursively lowered statement list before the remaining inline optimization.
Dedicated runner methods preserve those different contracts and make
unsupported composition unrepresentable.

The configuration and artifact carriers are frozen, but the legacy LLIR
payloads remain mutable.  Every non-empty pass returns a detached tree through
its proven typed API.  ``run_empty`` uses the common identity rewriter so even
empty-manager plumbing validates and detaches its mutable payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from time import perf_counter_ns
from typing import TYPE_CHECKING, Generic, NoReturn, Optional, Tuple, cast

from . import llir
from .diagnostics import CompilerInvariantError
from .dynamic_vector_access_pass import (
    DYNAMIC_VECTOR_ACCESS_CONTEXT,
    DynamicVectorAccessConfig,
    DynamicVectorAccessContext,
    rewrite_dynamic_vector_accesses,
)
from .dense_pointer_hoist_pass import (
    DensePointerHoistContext,
    hoist_dense_pointers,
)
from .llir_traversal import (
    LLIRRewriter,
    LLIRRewriteValueT,
    LLIRTraversalContext,
    LLIRValue,
    LLIRWalker,
)
from .result_write_pass import ResultWriteContext, rewrite_result_writes
from .single_iteration_loop_pass import (
    SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
    SingleIterationLoopEliminationContext,
    eliminate_single_iteration_loops,
)
from .sparse_prefetch_pass import (
    SPARSE_PREFETCH_CONTEXT,
    SparsePrefetchContext,
    insert_sparse_prefetch,
)

if TYPE_CHECKING:
    from .compressed_where_openmp_pass import (
        CompressedWhereOpenMPContext,
        CompressedWhereOpenMPResult,
    )


class LLIRPassArtifactType(Enum):
    """Stable artifact categories named by pass descriptors and run records."""

    REWRITE_VALUE = "llir_rewrite_value"
    STATEMENT_LIST = "llir_statement_list"
    COMPRESSED_WHERE_RESULT = "compressed_where_openmp_result"


class LLIRPassContextType(Enum):
    """Stable explicit configuration categories for managed passes."""

    DYNAMIC_VECTOR_ACCESS = "dynamic_vector_access_context"
    RESULT_WRITE = "result_write_context"
    COMPRESSED_WHERE_OPENMP = "compressed_where_openmp_context"
    SPARSE_PREFETCH = "sparse_prefetch_context"
    DENSE_POINTER_HOIST = "dense_pointer_hoist_context"
    SINGLE_ITERATION_LOOP_ELIMINATION = "single_iteration_loop_elimination_context"


@dataclass(frozen=True)
class LLIRPassDescriptor:
    """Stable identity and exact artifact/configuration boundary of a pass."""

    name: str
    version: int
    input_artifact: LLIRPassArtifactType
    output_artifact: LLIRPassArtifactType
    context_type: LLIRPassContextType


DYNAMIC_VECTOR_ACCESS_PASS = LLIRPassDescriptor(
    name="rewrite_dynamic_vector_accesses",
    version=1,
    input_artifact=LLIRPassArtifactType.REWRITE_VALUE,
    output_artifact=LLIRPassArtifactType.REWRITE_VALUE,
    context_type=LLIRPassContextType.DYNAMIC_VECTOR_ACCESS,
)

RESULT_WRITE_PASS = LLIRPassDescriptor(
    name="rewrite_result_writes",
    version=1,
    input_artifact=LLIRPassArtifactType.REWRITE_VALUE,
    output_artifact=LLIRPassArtifactType.REWRITE_VALUE,
    context_type=LLIRPassContextType.RESULT_WRITE,
)

COMPRESSED_WHERE_OPENMP_PASS = LLIRPassDescriptor(
    name="transform_compressed_where_for_openmp",
    version=1,
    input_artifact=LLIRPassArtifactType.STATEMENT_LIST,
    output_artifact=LLIRPassArtifactType.COMPRESSED_WHERE_RESULT,
    context_type=LLIRPassContextType.COMPRESSED_WHERE_OPENMP,
)

SPARSE_PREFETCH_PASS = LLIRPassDescriptor(
    name="insert_sparse_prefetch",
    version=1,
    input_artifact=LLIRPassArtifactType.STATEMENT_LIST,
    output_artifact=LLIRPassArtifactType.STATEMENT_LIST,
    context_type=LLIRPassContextType.SPARSE_PREFETCH,
)

DENSE_POINTER_HOIST_PASS = LLIRPassDescriptor(
    name="hoist_dense_pointers",
    version=1,
    input_artifact=LLIRPassArtifactType.STATEMENT_LIST,
    output_artifact=LLIRPassArtifactType.STATEMENT_LIST,
    context_type=LLIRPassContextType.DENSE_POINTER_HOIST,
)

SINGLE_ITERATION_LOOP_ELIMINATION_PASS = LLIRPassDescriptor(
    name="eliminate_single_iteration_loops",
    version=1,
    input_artifact=LLIRPassArtifactType.STATEMENT_LIST,
    output_artifact=LLIRPassArtifactType.STATEMENT_LIST,
    context_type=LLIRPassContextType.SINGLE_ITERATION_LOOP_ELIMINATION,
)


@dataclass(frozen=True)
class DynamicVectorAccessPassSpec:
    """One immutable dynamic-vector pass configuration snapshot."""

    context: DynamicVectorAccessContext = DYNAMIC_VECTOR_ACCESS_CONTEXT
    descriptor: LLIRPassDescriptor = DYNAMIC_VECTOR_ACCESS_PASS


@dataclass(frozen=True)
class ResultWritePassSpec:
    """One immutable result-write count or fill configuration snapshot."""

    context: ResultWriteContext
    descriptor: LLIRPassDescriptor = RESULT_WRITE_PASS


@dataclass(frozen=True)
class CompressedWhereOpenMPPassSpec:
    """One immutable compressed-Where/OpenMP configuration snapshot."""

    context: CompressedWhereOpenMPContext
    descriptor: LLIRPassDescriptor = COMPRESSED_WHERE_OPENMP_PASS


@dataclass(frozen=True)
class SparsePrefetchPassSpec:
    """One immutable sparse-prefetch pass configuration snapshot."""

    context: SparsePrefetchContext = SPARSE_PREFETCH_CONTEXT
    descriptor: LLIRPassDescriptor = SPARSE_PREFETCH_PASS


@dataclass(frozen=True)
class DensePointerHoistPassSpec:
    """One immutable dense-pointer-hoist configuration snapshot."""

    context: DensePointerHoistContext
    descriptor: LLIRPassDescriptor = DENSE_POINTER_HOIST_PASS


@dataclass(frozen=True)
class SingleIterationLoopEliminationPassSpec:
    """One immutable single-iteration-loop configuration snapshot."""

    context: SingleIterationLoopEliminationContext = (
        SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT
    )
    descriptor: LLIRPassDescriptor = SINGLE_ITERATION_LOOP_ELIMINATION_PASS


@dataclass(frozen=True)
class LLIRPassOptions:
    """Immutable verification and instrumentation policy snapshot."""

    verify_before_pass: bool = False
    verify_after_pass: bool = False
    record_timing: bool = True


PRODUCTION_LLIR_PASS_OPTIONS = LLIRPassOptions()
DEBUG_LLIR_PASS_OPTIONS = LLIRPassOptions(
    verify_before_pass=True,
    verify_after_pass=True,
)


@dataclass(frozen=True)
class LLIRRewriteArtifact(Generic[LLIRRewriteValueT]):
    """Frozen owner label around a root-preserving legacy LLIR payload."""

    value: LLIRRewriteValueT


@dataclass(frozen=True)
class LLIRStatementListArtifact:
    """Frozen owner label for an exact current-LLIR statement-list root.

    Legacy payloads may contain nested list/tuple statement sequences even
    though the outer artifact category remains exactly a list.
    """

    statements: list[llir.Stmt]


@dataclass(frozen=True)
class LLIRPassRunRecord:
    """Non-semantic ordered instrumentation for one completed pass."""

    sequence_index: int
    pass_name: str
    pass_version: int
    input_artifact: LLIRPassArtifactType
    output_artifact: LLIRPassArtifactType
    context_type: LLIRPassContextType
    configuration_name: str
    diagnostic_stage: str
    diagnostic_pass_name: str
    verified_before: bool
    verified_after: bool
    duration_ns: Optional[int] = field(compare=False)


@dataclass(frozen=True)
class LLIRRewritePassResult(Generic[LLIRRewriteValueT]):
    """Exact root-preserving output plus non-semantic run information."""

    artifact: LLIRRewriteArtifact[LLIRRewriteValueT]
    run_records: Tuple[LLIRPassRunRecord, ...] = field(compare=False)


@dataclass(frozen=True)
class LLIRStatementListPassResult:
    """Exact statement-list output plus non-semantic run information."""

    artifact: LLIRStatementListArtifact
    run_records: Tuple[LLIRPassRunRecord, ...] = field(compare=False)


@dataclass(frozen=True)
class ManagedCompressedWhereOpenMPResult:
    """Exact compressed-Where result plus non-semantic run information."""

    result: CompressedWhereOpenMPResult
    run_records: Tuple[LLIRPassRunRecord, ...] = field(compare=False)


@dataclass(frozen=True)
class LLIRPassManagerDiagnostic:
    """Structured failure owned by runner configuration or typed dispatch."""

    code: str
    message: str
    sequence_index: int
    pass_name: str
    pass_version: int


class LLIRPassManagerError(CompilerInvariantError):
    """An unknown or mismatched runner configuration failed closed."""

    def __init__(self, diagnostic: LLIRPassManagerDiagnostic) -> None:
        self.diagnostic = diagnostic
        self.diagnostics = (diagnostic,)
        super().__init__(
            "stage=LLIR pass manager "
            f"pass={diagnostic.pass_name}@{diagnostic.pass_version}: "
            f"{diagnostic.code} at sequence[{diagnostic.sequence_index}]: "
            f"{diagnostic.message}"
        )


def _raise_manager_error(
    *,
    code: str,
    message: str,
    sequence_index: int,
    descriptor: Optional[LLIRPassDescriptor] = None,
) -> NoReturn:
    raise LLIRPassManagerError(
        LLIRPassManagerDiagnostic(
            code=code,
            message=message,
            sequence_index=sequence_index,
            pass_name=descriptor.name if descriptor is not None else "<manager>",
            pass_version=descriptor.version if descriptor is not None else 1,
        )
    )


def _validate_traversal_context(
    traversal: object,
    *,
    descriptor: LLIRPassDescriptor,
) -> LLIRTraversalContext:
    if (
        type(traversal) is not LLIRTraversalContext
        or type(cast(LLIRTraversalContext, traversal).stage) is not str
        or not cast(LLIRTraversalContext, traversal).stage
        or type(cast(LLIRTraversalContext, traversal).pass_name) is not str
        or not cast(LLIRTraversalContext, traversal).pass_name
    ):
        _raise_manager_error(
            code="invalid_pass_traversal_context",
            message="pass traversal stage and name must be non-empty strings",
            sequence_index=0,
            descriptor=descriptor,
        )
    return cast(LLIRTraversalContext, traversal)


def _validate_descriptor(
    actual: object,
    expected: LLIRPassDescriptor,
) -> None:
    if type(actual) is not LLIRPassDescriptor or actual != expected:
        _raise_manager_error(
            code="pass_descriptor_mismatch",
            message=f"pass specification requires descriptor {expected.name}@{expected.version}",
            sequence_index=0,
            descriptor=(
                cast(LLIRPassDescriptor, actual)
                if type(actual) is LLIRPassDescriptor
                else expected
            ),
        )


def _validate_dynamic_context(context: object) -> DynamicVectorAccessContext:
    if type(context) is not DynamicVectorAccessContext:
        _raise_manager_error(
            code="invalid_pass_context",
            message="dynamic-vector runner requires DynamicVectorAccessContext",
            sequence_index=0,
            descriptor=DYNAMIC_VECTOR_ACCESS_PASS,
        )
    typed_context = cast(DynamicVectorAccessContext, context)
    _validate_traversal_context(
        typed_context.traversal,
        descriptor=DYNAMIC_VECTOR_ACCESS_PASS,
    )
    config = typed_context.config
    if type(config) is not DynamicVectorAccessConfig:
        _raise_manager_error(
            code="invalid_pass_context",
            message="dynamic-vector runner requires DynamicVectorAccessConfig",
            sequence_index=0,
            descriptor=DYNAMIC_VECTOR_ACCESS_PASS,
        )
    if (
        type(config.vector_type_prefix) is not str
        or not config.vector_type_prefix
        or type(config.append_suffixes) is not tuple
        or type(config.deduplicate_suffixes) is not tuple
        or any(
            type(suffix) is not str or not suffix
            for suffix in (*config.append_suffixes, *config.deduplicate_suffixes)
        )
        or type(config.append_method) is not str
        or not config.append_method
        or type(config.checked_set_function) is not str
        or not config.checked_set_function
    ):
        _raise_manager_error(
            code="invalid_pass_context",
            message="dynamic-vector configuration has malformed policy fields",
            sequence_index=0,
            descriptor=DYNAMIC_VECTOR_ACCESS_PASS,
        )
    return typed_context


def _validate_sparse_prefetch_context(context: object) -> SparsePrefetchContext:
    if type(context) is not SparsePrefetchContext:
        _raise_manager_error(
            code="invalid_pass_context",
            message="sparse-prefetch runner requires SparsePrefetchContext",
            sequence_index=0,
            descriptor=SPARSE_PREFETCH_PASS,
        )
    typed_context = cast(SparsePrefetchContext, context)
    _validate_traversal_context(
        typed_context.traversal,
        descriptor=SPARSE_PREFETCH_PASS,
    )
    return typed_context


def _validate_dense_pointer_hoist_context(
    context: object,
) -> DensePointerHoistContext:
    if type(context) is not DensePointerHoistContext:
        _raise_manager_error(
            code="invalid_pass_context",
            message="dense-pointer runner requires DensePointerHoistContext",
            sequence_index=0,
            descriptor=DENSE_POINTER_HOIST_PASS,
        )
    typed_context = cast(DensePointerHoistContext, context)
    _validate_traversal_context(
        typed_context.traversal,
        descriptor=DENSE_POINTER_HOIST_PASS,
    )
    entries = cast(object, typed_context.value_array_ctypes)
    if type(entries) is not tuple:
        _raise_manager_error(
            code="invalid_pass_context",
            message="dense-pointer C-type mapping must be an immutable tuple",
            sequence_index=0,
            descriptor=DENSE_POINTER_HOIST_PASS,
        )
    seen_names: set[str] = set()
    for entry in cast(Tuple[object, ...], entries):
        if type(entry) is not tuple or len(entry) != 2:
            _raise_manager_error(
                code="invalid_pass_context",
                message="dense-pointer C-type mapping entries must be exact pairs",
                sequence_index=0,
                descriptor=DENSE_POINTER_HOIST_PASS,
            )
        name, c_type = cast(Tuple[object, object], entry)
        if type(name) is not str or not name or type(c_type) is not str or not c_type:
            _raise_manager_error(
                code="invalid_pass_context",
                message=(
                    "dense-pointer C-type mapping names and types must be "
                    "non-empty strings"
                ),
                sequence_index=0,
                descriptor=DENSE_POINTER_HOIST_PASS,
            )
        typed_name = cast(str, name)
        if typed_name in seen_names:
            _raise_manager_error(
                code="invalid_pass_context",
                message="dense-pointer C-type mapping names must be unique",
                sequence_index=0,
                descriptor=DENSE_POINTER_HOIST_PASS,
            )
        seen_names.add(typed_name)
    return typed_context


def _validate_single_iteration_loop_elimination_context(
    context: object,
) -> SingleIterationLoopEliminationContext:
    if type(context) is not SingleIterationLoopEliminationContext:
        _raise_manager_error(
            code="invalid_pass_context",
            message=(
                "single-iteration-loop runner requires "
                "SingleIterationLoopEliminationContext"
            ),
            sequence_index=0,
            descriptor=SINGLE_ITERATION_LOOP_ELIMINATION_PASS,
        )
    typed_context = cast(SingleIterationLoopEliminationContext, context)
    _validate_traversal_context(
        typed_context.traversal,
        descriptor=SINGLE_ITERATION_LOOP_ELIMINATION_PASS,
    )
    return typed_context


def _record(
    *,
    descriptor: LLIRPassDescriptor,
    traversal: LLIRTraversalContext,
    configuration_name: str,
    options: LLIRPassOptions,
    started_ns: Optional[int],
) -> LLIRPassRunRecord:
    duration_ns = perf_counter_ns() - started_ns if started_ns is not None else None
    return LLIRPassRunRecord(
        sequence_index=0,
        pass_name=descriptor.name,
        pass_version=descriptor.version,
        input_artifact=descriptor.input_artifact,
        output_artifact=descriptor.output_artifact,
        context_type=descriptor.context_type,
        configuration_name=configuration_name,
        diagnostic_stage=traversal.stage,
        diagnostic_pass_name=traversal.pass_name,
        verified_before=options.verify_before_pass,
        verified_after=options.verify_after_pass,
        duration_ns=duration_ns,
    )


@dataclass(frozen=True)
class LLIRPassManager:
    """Dedicated, fail-closed runner methods for current extracted passes."""

    options: LLIRPassOptions = PRODUCTION_LLIR_PASS_OPTIONS

    def _validate_options(self) -> None:
        if type(self.options) is not LLIRPassOptions or any(
            type(value) is not bool
            for value in (
                self.options.verify_before_pass,
                self.options.verify_after_pass,
                self.options.record_timing,
            )
        ):
            _raise_manager_error(
                code="invalid_pass_options",
                message="manager options must contain exact boolean policy fields",
                sequence_index=-1,
            )

    @staticmethod
    def _validate_rewrite_artifact(artifact: object) -> None:
        if type(artifact) is not LLIRRewriteArtifact:
            _raise_manager_error(
                code="invalid_pass_artifact",
                message="root-preserving runner requires LLIRRewriteArtifact",
                sequence_index=0,
            )

    def run_empty(
        self,
        artifact: LLIRRewriteArtifact[LLIRRewriteValueT],
    ) -> LLIRRewritePassResult[LLIRRewriteValueT]:
        """Validate and detach an artifact without recording a semantic pass."""

        self._validate_options()
        self._validate_rewrite_artifact(artifact)
        traversal = LLIRTraversalContext(
            stage="LLIR pass manager",
            pass_name="empty_pipeline",
        )
        detached = LLIRRewriter(traversal).rewrite(artifact.value)
        return LLIRRewritePassResult(LLIRRewriteArtifact(detached), ())

    def run_sparse_prefetch(
        self,
        artifact: LLIRStatementListArtifact,
        pass_spec: SparsePrefetchPassSpec = SparsePrefetchPassSpec(),
    ) -> LLIRStatementListPassResult:
        """Run sparse-prefetch insertion over an exact statement-list root."""

        self._validate_options()
        if type(artifact) is not LLIRStatementListArtifact:
            _raise_manager_error(
                code="invalid_pass_artifact",
                message=("sparse-prefetch runner requires LLIRStatementListArtifact"),
                sequence_index=0,
                descriptor=SPARSE_PREFETCH_PASS,
            )
        if type(pass_spec) is not SparsePrefetchPassSpec:
            _raise_manager_error(
                code="unknown_pass_spec",
                message=(
                    "sparse-prefetch runner requires its exact pass specification"
                ),
                sequence_index=0,
                descriptor=SPARSE_PREFETCH_PASS,
            )
        _validate_descriptor(pass_spec.descriptor, SPARSE_PREFETCH_PASS)
        context = _validate_sparse_prefetch_context(pass_spec.context)
        traversal = context.traversal
        started_ns = perf_counter_ns() if self.options.record_timing else None
        if self.options.verify_before_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, artifact.statements))
        output = insert_sparse_prefetch(artifact.statements, context)
        if self.options.verify_after_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, output))
        record = _record(
            descriptor=SPARSE_PREFETCH_PASS,
            traversal=traversal,
            configuration_name="sparse_prefetch",
            options=self.options,
            started_ns=started_ns,
        )
        return LLIRStatementListPassResult(
            LLIRStatementListArtifact(output),
            (record,),
        )

    def run_dense_pointer_hoist(
        self,
        artifact: LLIRStatementListArtifact,
        pass_spec: DensePointerHoistPassSpec,
    ) -> LLIRStatementListPassResult:
        """Run dense-pointer hoisting over an exact statement-list root."""

        self._validate_options()
        if type(artifact) is not LLIRStatementListArtifact:
            _raise_manager_error(
                code="invalid_pass_artifact",
                message="dense-pointer runner requires LLIRStatementListArtifact",
                sequence_index=0,
                descriptor=DENSE_POINTER_HOIST_PASS,
            )
        if type(pass_spec) is not DensePointerHoistPassSpec:
            _raise_manager_error(
                code="unknown_pass_spec",
                message="dense-pointer runner requires its exact pass specification",
                sequence_index=0,
                descriptor=DENSE_POINTER_HOIST_PASS,
            )
        _validate_descriptor(pass_spec.descriptor, DENSE_POINTER_HOIST_PASS)
        context = _validate_dense_pointer_hoist_context(pass_spec.context)
        traversal = context.traversal
        started_ns = perf_counter_ns() if self.options.record_timing else None
        if self.options.verify_before_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, artifact.statements))
        output = hoist_dense_pointers(artifact.statements, context)
        if self.options.verify_after_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, output))
        record = _record(
            descriptor=DENSE_POINTER_HOIST_PASS,
            traversal=traversal,
            configuration_name="dense_pointer_hoist",
            options=self.options,
            started_ns=started_ns,
        )
        return LLIRStatementListPassResult(
            LLIRStatementListArtifact(output),
            (record,),
        )

    def run_single_iteration_loop_elimination(
        self,
        artifact: LLIRStatementListArtifact,
        pass_spec: SingleIterationLoopEliminationPassSpec = (
            SingleIterationLoopEliminationPassSpec()
        ),
    ) -> LLIRStatementListPassResult:
        """Eliminate single-iteration loops over an exact statement-list root."""

        self._validate_options()
        if type(artifact) is not LLIRStatementListArtifact:
            _raise_manager_error(
                code="invalid_pass_artifact",
                message=(
                    "single-iteration-loop runner requires " "LLIRStatementListArtifact"
                ),
                sequence_index=0,
                descriptor=SINGLE_ITERATION_LOOP_ELIMINATION_PASS,
            )
        if type(pass_spec) is not SingleIterationLoopEliminationPassSpec:
            _raise_manager_error(
                code="unknown_pass_spec",
                message=(
                    "single-iteration-loop runner requires its exact pass "
                    "specification"
                ),
                sequence_index=0,
                descriptor=SINGLE_ITERATION_LOOP_ELIMINATION_PASS,
            )
        _validate_descriptor(
            pass_spec.descriptor,
            SINGLE_ITERATION_LOOP_ELIMINATION_PASS,
        )
        context = _validate_single_iteration_loop_elimination_context(pass_spec.context)
        traversal = context.traversal
        started_ns = perf_counter_ns() if self.options.record_timing else None
        if self.options.verify_before_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, artifact.statements))
        output = eliminate_single_iteration_loops(artifact.statements, context)
        if self.options.verify_after_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, output))
        record = _record(
            descriptor=SINGLE_ITERATION_LOOP_ELIMINATION_PASS,
            traversal=traversal,
            configuration_name="single_iteration_loop_elimination",
            options=self.options,
            started_ns=started_ns,
        )
        return LLIRStatementListPassResult(
            LLIRStatementListArtifact(output),
            (record,),
        )

    def run_dynamic_vector_access(
        self,
        artifact: LLIRRewriteArtifact[LLIRRewriteValueT],
        pass_spec: DynamicVectorAccessPassSpec = DynamicVectorAccessPassSpec(),
    ) -> LLIRRewritePassResult[LLIRRewriteValueT]:
        """Run the root-preserving dynamic-vector rewrite."""

        self._validate_options()
        self._validate_rewrite_artifact(artifact)
        if type(pass_spec) is not DynamicVectorAccessPassSpec:
            _raise_manager_error(
                code="unknown_pass_spec",
                message="dynamic-vector runner requires its exact pass specification",
                sequence_index=0,
                descriptor=DYNAMIC_VECTOR_ACCESS_PASS,
            )
        _validate_descriptor(pass_spec.descriptor, DYNAMIC_VECTOR_ACCESS_PASS)
        context = _validate_dynamic_context(pass_spec.context)
        traversal = context.traversal
        started_ns = perf_counter_ns() if self.options.record_timing else None
        if self.options.verify_before_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, artifact.value))
        output = rewrite_dynamic_vector_accesses(artifact.value, context)
        if self.options.verify_after_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, output))
        record = _record(
            descriptor=DYNAMIC_VECTOR_ACCESS_PASS,
            traversal=traversal,
            configuration_name="dynamic_vector_access",
            options=self.options,
            started_ns=started_ns,
        )
        return LLIRRewritePassResult(LLIRRewriteArtifact(output), (record,))

    def run_result_write(
        self,
        artifact: LLIRRewriteArtifact[LLIRRewriteValueT],
        pass_spec: ResultWritePassSpec,
    ) -> LLIRRewritePassResult[LLIRRewriteValueT]:
        """Run one independent root-preserving result-write mode."""

        self._validate_options()
        self._validate_rewrite_artifact(artifact)
        if type(pass_spec) is not ResultWritePassSpec:
            _raise_manager_error(
                code="unknown_pass_spec",
                message="result-write runner requires its exact pass specification",
                sequence_index=0,
                descriptor=RESULT_WRITE_PASS,
            )
        _validate_descriptor(pass_spec.descriptor, RESULT_WRITE_PASS)
        if type(pass_spec.context) is not ResultWriteContext:
            _raise_manager_error(
                code="invalid_pass_context",
                message="result-write runner requires ResultWriteContext",
                sequence_index=0,
                descriptor=RESULT_WRITE_PASS,
            )
        traversal = _validate_traversal_context(
            pass_spec.context.traversal,
            descriptor=RESULT_WRITE_PASS,
        )
        started_ns = perf_counter_ns() if self.options.record_timing else None
        if self.options.verify_before_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, artifact.value))
        output = rewrite_result_writes(artifact.value, pass_spec.context)
        if self.options.verify_after_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, output))
        record = _record(
            descriptor=RESULT_WRITE_PASS,
            traversal=traversal,
            configuration_name=pass_spec.context.mode,
            options=self.options,
            started_ns=started_ns,
        )
        return LLIRRewritePassResult(LLIRRewriteArtifact(output), (record,))

    def run_compressed_where_openmp(
        self,
        artifact: LLIRStatementListArtifact,
        pass_spec: CompressedWhereOpenMPPassSpec,
    ) -> ManagedCompressedWhereOpenMPResult:
        """Run compressed-Where without erasing its exact secondary result."""

        from .compressed_where_openmp_pass import (
            CompressedWhereOpenMPContext,
            _transform_compressed_where_for_openmp_managed,
        )

        self._validate_options()
        if type(artifact) is not LLIRStatementListArtifact:
            _raise_manager_error(
                code="invalid_pass_artifact",
                message="compressed-Where runner requires LLIRStatementListArtifact",
                sequence_index=0,
                descriptor=COMPRESSED_WHERE_OPENMP_PASS,
            )
        if type(pass_spec) is not CompressedWhereOpenMPPassSpec:
            _raise_manager_error(
                code="unknown_pass_spec",
                message="compressed-Where runner requires its exact pass specification",
                sequence_index=0,
                descriptor=COMPRESSED_WHERE_OPENMP_PASS,
            )
        _validate_descriptor(pass_spec.descriptor, COMPRESSED_WHERE_OPENMP_PASS)
        if type(pass_spec.context) is not CompressedWhereOpenMPContext:
            _raise_manager_error(
                code="invalid_pass_context",
                message="compressed-Where runner requires CompressedWhereOpenMPContext",
                sequence_index=0,
                descriptor=COMPRESSED_WHERE_OPENMP_PASS,
            )
        traversal = _validate_traversal_context(
            pass_spec.context.traversal,
            descriptor=COMPRESSED_WHERE_OPENMP_PASS,
        )
        started_ns = perf_counter_ns() if self.options.record_timing else None
        if self.options.verify_before_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, artifact.statements))
        execution = _transform_compressed_where_for_openmp_managed(
            artifact.statements,
            pass_spec.context,
            self.options,
        )
        result = execution.result
        if self.options.verify_after_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, result.statements))
        parent_record = _record(
            descriptor=COMPRESSED_WHERE_OPENMP_PASS,
            traversal=traversal,
            configuration_name="compressed_where_openmp",
            options=self.options,
            started_ns=started_ns,
        )
        expected_nested = ("count", "fill") if result.applied else ()
        actual_nested = tuple(
            record.configuration_name for record in execution.nested_run_records
        )
        if actual_nested != expected_nested or any(
            record.pass_name != RESULT_WRITE_PASS.name
            or record.pass_version != RESULT_WRITE_PASS.version
            for record in execution.nested_run_records
        ):
            _raise_manager_error(
                code="invalid_nested_pass_records",
                message=(
                    "compressed-Where composition must run independent count and "
                    "fill result-write passes exactly once"
                ),
                sequence_index=0,
                descriptor=COMPRESSED_WHERE_OPENMP_PASS,
            )
        nested_records = tuple(
            replace(record, sequence_index=index)
            for index, record in enumerate(execution.nested_run_records, start=1)
        )
        return ManagedCompressedWhereOpenMPResult(
            result,
            (parent_record, *nested_records),
        )
