"""Minimal typed orchestration for the extracted current-LLIR passes.

The current production pass graph is not one linear pipeline.  Dynamic-vector
rewriting is a top-level step over the assembled function body, compressed
``Where`` lowering is a top-level step during outer-loop lowering, and
result-write rewriting is composed internally by the compressed ``Where``
transform.  This module preserves those distinct call points while making each
managed invocation, its configuration, and its run information explicit.

Only the dynamic-vector and result-write passes share the root-preserving
``LLIRRewriteValueT -> LLIRRewriteValueT`` contract, so only those passes may be
placed in :class:`LLIRRewritePipeline`.  Compressed ``Where`` has a separate
typed runner method because it accepts an exact statement-list root and returns
the secondary ``applied`` result in :class:`CompressedWhereOpenMPResult`.

Artifacts and configuration carriers are frozen, but the legacy LLIR payload
is mutable.  Every non-empty managed execution relies on the underlying typed
pass contract to return a fully detached tree.  An empty pipeline performs only
artifact/configuration plumbing and intentionally does not clone its payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import TYPE_CHECKING, Generic, NoReturn, Optional, Tuple, Union, cast

from . import llir
from .diagnostics import CompilerInvariantError
from .dynamic_vector_access_pass import (
    DYNAMIC_VECTOR_ACCESS_CONTEXT,
    DynamicVectorAccessConfig,
    DynamicVectorAccessContext,
    rewrite_dynamic_vector_accesses,
)
from .llir_traversal import (
    LLIRRewriteValueT,
    LLIRTraversalContext,
    LLIRValue,
    LLIRWalker,
)
from .result_write_pass import ResultWriteContext, rewrite_result_writes

if TYPE_CHECKING:
    from .compressed_where_openmp_pass import (
        CompressedWhereOpenMPContext,
        CompressedWhereOpenMPResult,
    )


class LLIRPassArtifactType(Enum):
    """Stable artifact categories used by current-LLIR pass descriptors."""

    REWRITE_VALUE = "llir_rewrite_value"
    STATEMENT_LIST = "llir_statement_list"
    COMPRESSED_WHERE_RESULT = "compressed_where_openmp_result"


class LLIRPassContextType(Enum):
    """Stable explicit configuration categories for managed LLIR passes."""

    DYNAMIC_VECTOR_ACCESS = "dynamic_vector_access_context"
    RESULT_WRITE = "result_write_context"
    COMPRESSED_WHERE_OPENMP = "compressed_where_openmp_context"


@dataclass(frozen=True)
class LLIRPassDescriptor:
    """Stable identity and typed boundary of one managed pass."""

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


@dataclass(frozen=True)
class DynamicVectorAccessPassSpec:
    """One configured dynamic-vector pass invocation."""

    context: DynamicVectorAccessContext = DYNAMIC_VECTOR_ACCESS_CONTEXT
    descriptor: LLIRPassDescriptor = DYNAMIC_VECTOR_ACCESS_PASS


@dataclass(frozen=True)
class ResultWritePassSpec:
    """One configured count or fill result-write invocation."""

    context: ResultWriteContext
    descriptor: LLIRPassDescriptor = RESULT_WRITE_PASS


@dataclass(frozen=True)
class CompressedWhereOpenMPPassSpec:
    """One configured compressed-Where/OpenMP invocation."""

    context: CompressedWhereOpenMPContext
    descriptor: LLIRPassDescriptor = COMPRESSED_WHERE_OPENMP_PASS


LLIRRewritePassSpec = Union[DynamicVectorAccessPassSpec, ResultWritePassSpec]


@dataclass(frozen=True)
class LLIRPassOptions:
    """Immutable verification and instrumentation policy snapshot.

    Production records boundary timing but does not add extra full-tree walks;
    the typed passes retain their own always-on validation.  Tests and debug
    callers may request explicit full structural walks before and after every
    managed pass.
    """

    verify_before_pass: bool = False
    verify_after_pass: bool = False
    record_timing: bool = True


PRODUCTION_LLIR_PASS_OPTIONS = LLIRPassOptions()
DEBUG_LLIR_PASS_OPTIONS = LLIRPassOptions(
    verify_before_pass=True,
    verify_after_pass=True,
)


@dataclass(frozen=True)
class LLIRPassArtifact(Generic[LLIRRewriteValueT]):
    """Frozen owner label around a legacy mutable LLIR payload."""

    value: LLIRRewriteValueT


@dataclass(frozen=True)
class LLIRRewritePipeline:
    """Ordered root-preserving pass configuration for one boundary."""

    passes: Tuple[LLIRRewritePassSpec, ...] = ()


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
class LLIRPassPipelineResult(Generic[LLIRRewriteValueT]):
    """Typed rewrite artifact plus non-semantic ordered run information."""

    artifact: LLIRPassArtifact[LLIRRewriteValueT]
    run_records: Tuple[LLIRPassRunRecord, ...] = field(compare=False)


@dataclass(frozen=True)
class ManagedCompressedWhereOpenMPResult:
    """Exact compressed-Where result plus non-semantic run information."""

    result: CompressedWhereOpenMPResult
    run_records: Tuple[LLIRPassRunRecord, ...] = field(compare=False)


@dataclass(frozen=True)
class LLIRPassManagerDiagnostic:
    """Structured failure owned by pass configuration/dispatch."""

    code: str
    message: str
    sequence_index: int
    pass_name: str
    pass_version: int


class LLIRPassManagerError(CompilerInvariantError):
    """An unknown or mismatched managed-pass configuration failed closed."""

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
    sequence_index: int,
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
            sequence_index=sequence_index,
            descriptor=descriptor,
        )
    return cast(LLIRTraversalContext, traversal)


def _validate_dynamic_context(
    context: object,
    *,
    sequence_index: int,
    descriptor: LLIRPassDescriptor,
) -> DynamicVectorAccessContext:
    if type(context) is not DynamicVectorAccessContext:
        _raise_manager_error(
            code="invalid_pass_context",
            message="dynamic-vector pass requires DynamicVectorAccessContext",
            sequence_index=sequence_index,
            descriptor=descriptor,
        )
    typed_context = cast(DynamicVectorAccessContext, context)
    _validate_traversal_context(
        typed_context.traversal,
        sequence_index=sequence_index,
        descriptor=descriptor,
    )
    config = typed_context.config
    if type(config) is not DynamicVectorAccessConfig:
        _raise_manager_error(
            code="invalid_pass_context",
            message="dynamic-vector pass requires DynamicVectorAccessConfig",
            sequence_index=sequence_index,
            descriptor=descriptor,
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
            sequence_index=sequence_index,
            descriptor=descriptor,
        )
    return typed_context


def _validate_rewrite_pass(
    pass_spec: object, sequence_index: int
) -> LLIRRewritePassSpec:
    if type(pass_spec) is DynamicVectorAccessPassSpec:
        dynamic_spec = cast(DynamicVectorAccessPassSpec, pass_spec)
        if dynamic_spec.descriptor != DYNAMIC_VECTOR_ACCESS_PASS:
            _raise_manager_error(
                code="pass_descriptor_mismatch",
                message="dynamic-vector pass descriptor does not match its runner",
                sequence_index=sequence_index,
                descriptor=dynamic_spec.descriptor,
            )
        _validate_dynamic_context(
            dynamic_spec.context,
            sequence_index=sequence_index,
            descriptor=dynamic_spec.descriptor,
        )
        return dynamic_spec

    if type(pass_spec) is ResultWritePassSpec:
        result_spec = cast(ResultWritePassSpec, pass_spec)
        if result_spec.descriptor != RESULT_WRITE_PASS:
            _raise_manager_error(
                code="pass_descriptor_mismatch",
                message="result-write pass descriptor does not match its runner",
                sequence_index=sequence_index,
                descriptor=result_spec.descriptor,
            )
        if type(result_spec.context) is not ResultWriteContext:
            _raise_manager_error(
                code="invalid_pass_context",
                message="result-write pass requires ResultWriteContext",
                sequence_index=sequence_index,
                descriptor=result_spec.descriptor,
            )
        _validate_traversal_context(
            result_spec.context.traversal,
            sequence_index=sequence_index,
            descriptor=result_spec.descriptor,
        )
        return result_spec

    _raise_manager_error(
        code="unknown_pass_spec",
        message="rewrite pipeline contains an unknown pass specification",
        sequence_index=sequence_index,
    )


def _pass_traversal(pass_spec: LLIRRewritePassSpec) -> LLIRTraversalContext:
    if type(pass_spec) is DynamicVectorAccessPassSpec:
        return cast(DynamicVectorAccessPassSpec, pass_spec).context.traversal
    return cast(ResultWritePassSpec, pass_spec).context.traversal


def _configuration_name(pass_spec: LLIRRewritePassSpec) -> str:
    if type(pass_spec) is DynamicVectorAccessPassSpec:
        return "dynamic_vector_access"
    return cast(ResultWritePassSpec, pass_spec).context.mode


def _run_rewrite_pass(
    value: LLIRRewriteValueT,
    pass_spec: LLIRRewritePassSpec,
) -> LLIRRewriteValueT:
    if type(pass_spec) is DynamicVectorAccessPassSpec:
        dynamic_spec = cast(DynamicVectorAccessPassSpec, pass_spec)
        return rewrite_dynamic_vector_accesses(value, dynamic_spec.context)
    result_spec = cast(ResultWritePassSpec, pass_spec)
    return rewrite_result_writes(value, result_spec.context)


def _record(
    *,
    sequence_index: int,
    descriptor: LLIRPassDescriptor,
    traversal: LLIRTraversalContext,
    configuration_name: str,
    options: LLIRPassOptions,
    started_ns: Optional[int],
) -> LLIRPassRunRecord:
    duration_ns = perf_counter_ns() - started_ns if started_ns is not None else None
    return LLIRPassRunRecord(
        sequence_index=sequence_index,
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
    """Typed, fail-closed runner for the current extracted LLIR passes."""

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

    def run(
        self,
        artifact: LLIRPassArtifact[LLIRRewriteValueT],
        pipeline: LLIRRewritePipeline,
    ) -> LLIRPassPipelineResult[LLIRRewriteValueT]:
        """Run an explicit root-preserving pipeline and stop on first failure."""

        self._validate_options()
        if type(artifact) is not LLIRPassArtifact:
            _raise_manager_error(
                code="invalid_pass_artifact",
                message="rewrite runner requires LLIRPassArtifact",
                sequence_index=-1,
            )
        if (
            type(pipeline) is not LLIRRewritePipeline
            or type(pipeline.passes) is not tuple
        ):
            _raise_manager_error(
                code="invalid_pass_pipeline",
                message="rewrite pipeline must be a frozen tuple-backed configuration",
                sequence_index=-1,
            )

        checked_passes = tuple(
            _validate_rewrite_pass(pass_spec, index)
            for index, pass_spec in enumerate(pipeline.passes)
        )
        current = artifact.value
        records = []
        for index, pass_spec in enumerate(checked_passes):
            descriptor = pass_spec.descriptor
            traversal = _pass_traversal(pass_spec)
            started_ns = perf_counter_ns() if self.options.record_timing else None
            if self.options.verify_before_pass:
                LLIRWalker(traversal).walk(cast(LLIRValue, current))
            current = _run_rewrite_pass(current, pass_spec)
            if self.options.verify_after_pass:
                LLIRWalker(traversal).walk(cast(LLIRValue, current))
            records.append(
                _record(
                    sequence_index=index,
                    descriptor=descriptor,
                    traversal=traversal,
                    configuration_name=_configuration_name(pass_spec),
                    options=self.options,
                    started_ns=started_ns,
                )
            )

        return LLIRPassPipelineResult(
            artifact=LLIRPassArtifact(current),
            run_records=tuple(records),
        )

    def run_compressed_where_openmp(
        self,
        artifact: LLIRPassArtifact[list[llir.Stmt]],
        pass_spec: CompressedWhereOpenMPPassSpec,
    ) -> ManagedCompressedWhereOpenMPResult:
        """Run the non-root-preserving compressed-Where pass without erasure."""

        from .compressed_where_openmp_pass import (
            CompressedWhereOpenMPContext,
            transform_compressed_where_for_openmp,
        )

        self._validate_options()
        if type(artifact) is not LLIRPassArtifact:
            _raise_manager_error(
                code="invalid_pass_artifact",
                message="compressed-Where runner requires LLIRPassArtifact",
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
        if pass_spec.descriptor != COMPRESSED_WHERE_OPENMP_PASS:
            _raise_manager_error(
                code="pass_descriptor_mismatch",
                message="compressed-Where descriptor does not match its runner",
                sequence_index=0,
                descriptor=pass_spec.descriptor,
            )
        if type(pass_spec.context) is not CompressedWhereOpenMPContext:
            _raise_manager_error(
                code="invalid_pass_context",
                message="compressed-Where pass requires CompressedWhereOpenMPContext",
                sequence_index=0,
                descriptor=pass_spec.descriptor,
            )
        traversal = _validate_traversal_context(
            pass_spec.context.traversal,
            sequence_index=0,
            descriptor=pass_spec.descriptor,
        )

        started_ns = perf_counter_ns() if self.options.record_timing else None
        if self.options.verify_before_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, artifact.value))
        result = transform_compressed_where_for_openmp(
            artifact.value,
            pass_spec.context,
        )
        if self.options.verify_after_pass:
            LLIRWalker(traversal).walk(cast(LLIRValue, result.statements))
        record = _record(
            sequence_index=0,
            descriptor=pass_spec.descriptor,
            traversal=traversal,
            configuration_name="compressed_where_openmp",
            options=self.options,
            started_ns=started_ns,
        )
        return ManagedCompressedWhereOpenMPResult(result, (record,))
