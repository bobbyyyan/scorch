"""Compilation-local ownership for compiler timing instrumentation.

``CompilationContext`` is the design-canonical owner paired with one exact
``CompileOptions`` snapshot.  It publishes immutable tuple snapshots of
completed compiler stages and of the existing managed LLIR pass records.  The
records are observation only: this module is not referenced by semantic IR,
cache identity, generated names, emitted source, or native build requests.

Stage ordinals are reserved when work begins and a record is published only
after that work completes successfully.  This keeps nested result/ABI assembly
ordered inside CIN lowering while retaining the completed child record if later
CIN work fails.  Failed stages publish no record and ordinary exceptions pass
through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from time import perf_counter_ns
from typing import TYPE_CHECKING, NoReturn, Optional, Tuple

from .diagnostics import CompilerInvariantError

if TYPE_CHECKING:
    from .compile_options import CompileOptions
    from .llir_pass_manager import LLIRPassRunRecord


class CompilerStageId(Enum):
    """Stable identities for the current production compiler stages."""

    FRONTEND_VALIDATED_OPERATION_CONSTRUCTION = (
        "frontend_validated_operation_construction"
    )
    CIN_NORMALIZATION_AND_VERIFICATION = "cin_normalization_and_verification"
    SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION = "scheduling_and_loop_plan_construction"
    LEGACY_CIN_ADAPTATION = "legacy_cin_adaptation"
    CIN_LOWERING = "cin_lowering"
    RESULT_ABI_ASSEMBLY = "result_abi_assembly"
    SCHEDULE_LOWERING = "schedule_lowering"
    LLIR_TO_CPP_GENERATION = "llir_to_cpp_generation"
    KERNEL_NAME_AND_BUILD_REQUEST_ASSEMBLY = "kernel_name_and_build_request_assembly"


CANONICAL_COMPILER_STAGES: Tuple[CompilerStageId, ...] = tuple(CompilerStageId)


@dataclass(frozen=True)
class CompilationContextDiagnostic:
    """Structured diagnostic for invalid instrumentation ownership."""

    code: str
    message: str
    stage_id: Optional[CompilerStageId] = None


class CompilationContextError(CompilerInvariantError):
    """A compilation context was paired with invalid typed ownership."""

    def __init__(self, diagnostic: CompilationContextDiagnostic) -> None:
        self.diagnostic = diagnostic
        self.diagnostics = (diagnostic,)
        stage_name = (
            diagnostic.stage_id.value
            if diagnostic.stage_id is not None
            else "<context>"
        )
        super().__init__(
            "stage=compilation context "
            f"stage_id={stage_name}: {diagnostic.code}: {diagnostic.message}"
        )


def _raise_context_error(
    *,
    code: str,
    message: str,
    stage_id: Optional[CompilerStageId] = None,
) -> NoReturn:
    raise CompilationContextError(
        CompilationContextDiagnostic(
            code=code,
            message=message,
            stage_id=stage_id,
        )
    )


@dataclass(frozen=True)
class CompilerStageRunRecord:
    """One successfully completed, non-semantic compiler-stage observation."""

    sequence_index: int
    stage_id: CompilerStageId
    nested_within: Optional[CompilerStageId]
    duration_ns: int = field(compare=False)


@dataclass(frozen=True)
class CompilerStageToken:
    """Opaque immutable start marker consumed by its owning context once."""

    stage_id: CompilerStageId
    sequence_index: int
    nested_within: Optional[CompilerStageId]
    started_ns: int
    _owner_identity: object = field(repr=False, compare=False)
    _token_identity: object = field(repr=False, compare=False)


@dataclass(frozen=True, eq=False)
class CompilationContext:
    """One frozen owner for one compilation's options and timing records.

    The public label and all exposed snapshots are immutable.  Completed
    observations are appended internally by replacing private tuple fields;
    there is no mutable registry, global state, callback, or cache.
    """

    compile_options: "CompileOptions"
    _owner_identity: object = field(
        default_factory=object,
        init=False,
        repr=False,
        compare=False,
    )
    _next_stage_sequence_index: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
    )
    _completed_stage_token_identities: Tuple[object, ...] = field(
        default=(),
        init=False,
        repr=False,
        compare=False,
    )
    _retired_stage_token_identities: Tuple[object, ...] = field(
        default=(),
        init=False,
        repr=False,
        compare=False,
    )
    _active_stage_tokens: Tuple[CompilerStageToken, ...] = field(
        default=(),
        init=False,
        repr=False,
        compare=False,
    )
    _failed_stage_id: Optional[CompilerStageId] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _stage_run_records: Tuple[CompilerStageRunRecord, ...] = field(
        default=(),
        init=False,
        repr=False,
        compare=False,
    )
    _llir_pass_run_records: Tuple["LLIRPassRunRecord", ...] = field(
        default=(),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        from .compile_options import CompileOptions

        if type(self.compile_options) is not CompileOptions:
            _raise_context_error(
                code="invalid_compile_options",
                message="CompilationContext requires an exact CompileOptions snapshot",
            )

    def require_compile_options(
        self,
        compile_options: "CompileOptions",
        *,
        stage_id: Optional[CompilerStageId] = None,
    ) -> None:
        """Fail closed unless a stage routes this context's exact snapshot."""

        if compile_options is not self.compile_options:
            _raise_context_error(
                code="detached_compile_options",
                message=(
                    "a compilation stage must route the exact CompileOptions "
                    "snapshot owned by its CompilationContext"
                ),
                stage_id=stage_id,
            )

    def begin_stage(
        self,
        stage_id: CompilerStageId,
        *,
        compile_options: "CompileOptions",
        nested_within: Optional[CompilerStageId] = None,
    ) -> CompilerStageToken:
        """Reserve execution order and start one canonical compiler stage."""

        if type(stage_id) is not CompilerStageId:
            _raise_context_error(
                code="invalid_stage_id",
                message="begin_stage requires an exact CompilerStageId",
            )
        if nested_within is not None and type(nested_within) is not CompilerStageId:
            _raise_context_error(
                code="invalid_parent_stage_id",
                message="nested_within must be an exact CompilerStageId or None",
                stage_id=stage_id,
            )
        expected_parent = (
            CompilerStageId.CIN_LOWERING
            if stage_id is CompilerStageId.RESULT_ABI_ASSEMBLY
            else None
        )
        if nested_within is not expected_parent:
            _raise_context_error(
                code="invalid_stage_nesting",
                message=(
                    "only result/ABI assembly is nested, directly within "
                    "CIN lowering"
                ),
                stage_id=stage_id,
            )
        self.require_compile_options(compile_options, stage_id=stage_id)
        if self._failed_stage_id is not None:
            _raise_context_error(
                code="failed_compilation",
                message=(
                    "a failed compilation cannot begin later compiler work; "
                    f"the first failed stage was {self._failed_stage_id.value}"
                ),
                stage_id=stage_id,
            )
        if stage_id is CompilerStageId.RESULT_ABI_ASSEMBLY:
            if (
                len(self._active_stage_tokens) != 1
                or self._active_stage_tokens[-1].stage_id
                is not CompilerStageId.CIN_LOWERING
            ):
                _raise_context_error(
                    code="inactive_parent_stage",
                    message=(
                        "result/ABI assembly requires the current active stage "
                        "to be this compilation's CIN-lowering stage"
                    ),
                    stage_id=stage_id,
                )
        elif self._active_stage_tokens:
            _raise_context_error(
                code="overlapping_stage",
                message=(
                    "root compiler stages execute serially; only result/ABI "
                    "assembly may nest within CIN lowering"
                ),
                stage_id=stage_id,
            )
        sequence_index = self._next_stage_sequence_index
        object.__setattr__(
            self,
            "_next_stage_sequence_index",
            sequence_index + 1,
        )
        token = CompilerStageToken(
            stage_id=stage_id,
            sequence_index=sequence_index,
            nested_within=nested_within,
            started_ns=perf_counter_ns(),
            _owner_identity=self._owner_identity,
            _token_identity=object(),
        )
        object.__setattr__(
            self,
            "_active_stage_tokens",
            self._active_stage_tokens + (token,),
        )
        return token

    def _require_active_top_token(
        self,
        token: CompilerStageToken,
        *,
        operation: str,
    ) -> None:
        if type(token) is not CompilerStageToken:
            _raise_context_error(
                code="invalid_stage_token",
                message=(f"{operation} requires the exact token begin_stage returned"),
            )
        if token._owner_identity is not self._owner_identity:
            _raise_context_error(
                code="detached_stage_token",
                message=(
                    f"a stage token may be {operation} only by its owning context"
                ),
                stage_id=token.stage_id,
            )
        if token._token_identity in self._completed_stage_token_identities:
            _raise_context_error(
                code="completed_stage_token",
                message="a completed stage token cannot be consumed again",
                stage_id=token.stage_id,
            )
        if token._token_identity in self._retired_stage_token_identities:
            _raise_context_error(
                code="retired_stage_token",
                message="a failed or cancelled stage token cannot be consumed again",
                stage_id=token.stage_id,
            )
        if not self._active_stage_tokens or not any(
            active is token for active in self._active_stage_tokens
        ):
            _raise_context_error(
                code="inactive_stage_token",
                message=f"a stage token must be active before it can be {operation}",
                stage_id=token.stage_id,
            )
        if self._active_stage_tokens[-1] is not token:
            _raise_context_error(
                code="unbalanced_stage_stack",
                message=(
                    "compiler stages must be consumed in strict last-in, "
                    "first-out order"
                ),
                stage_id=token.stage_id,
            )

    def is_stage_active(self, token: CompilerStageToken) -> bool:
        """Whether the exact owning token is still present on this stack."""

        if type(token) is not CompilerStageToken:
            _raise_context_error(
                code="invalid_stage_token",
                message="is_stage_active requires the exact token begin_stage returned",
            )
        if token._owner_identity is not self._owner_identity:
            _raise_context_error(
                code="detached_stage_token",
                message="a stage token may be queried only by its owning context",
                stage_id=token.stage_id,
            )
        return any(active is token for active in self._active_stage_tokens)

    def complete_stage(self, token: CompilerStageToken) -> None:
        """Publish a record for one successful stage exactly once."""

        self._require_active_top_token(token, operation="completed")
        duration_ns = perf_counter_ns() - token.started_ns
        record = CompilerStageRunRecord(
            sequence_index=token.sequence_index,
            stage_id=token.stage_id,
            nested_within=token.nested_within,
            duration_ns=duration_ns,
        )
        object.__setattr__(
            self,
            "_completed_stage_token_identities",
            self._completed_stage_token_identities + (token._token_identity,),
        )
        object.__setattr__(
            self,
            "_active_stage_tokens",
            self._active_stage_tokens[:-1],
        )
        object.__setattr__(
            self,
            "_stage_run_records",
            tuple(
                sorted(
                    self._stage_run_records + (record,),
                    key=lambda item: item.sequence_index,
                )
            ),
        )

    def fail_stage(self, token: CompilerStageToken) -> None:
        """Retire one failed stage without publishing a record.

        The first failure makes the compilation terminal.  Callers re-raise the
        original exception after this bookkeeping, so instrumentation never
        replaces the compiler's existing failure contract.
        """

        self._require_active_top_token(token, operation="failed")
        if self._failed_stage_id is None:
            object.__setattr__(self, "_failed_stage_id", token.stage_id)
        object.__setattr__(
            self,
            "_retired_stage_token_identities",
            self._retired_stage_token_identities + (token._token_identity,),
        )
        object.__setattr__(
            self,
            "_active_stage_tokens",
            self._active_stage_tokens[:-1],
        )

    def cancel_stage(self, token: CompilerStageToken) -> None:
        """Retire an optional stage attempt that produced no compiler artifact."""

        self._require_active_top_token(token, operation="cancelled")
        object.__setattr__(
            self,
            "_retired_stage_token_identities",
            self._retired_stage_token_identities + (token._token_identity,),
        )
        object.__setattr__(
            self,
            "_active_stage_tokens",
            self._active_stage_tokens[:-1],
        )

    def record_llir_pass_runs(
        self,
        records: Tuple["LLIRPassRunRecord", ...],
        *,
        compile_options: "CompileOptions",
    ) -> None:
        """Append existing managed-pass records to compilation-wide ownership."""

        from .llir_pass_manager import LLIRPassRunRecord

        self.require_compile_options(
            compile_options,
            stage_id=CompilerStageId.CIN_LOWERING,
        )
        if (
            not self._active_stage_tokens
            or self._active_stage_tokens[-1].stage_id
            is not CompilerStageId.CIN_LOWERING
        ):
            _raise_context_error(
                code="inactive_cin_lowering",
                message=(
                    "managed LLIR pass records may be appended only while this "
                    "compilation's CIN-lowering stage is current"
                ),
                stage_id=CompilerStageId.CIN_LOWERING,
            )
        if type(records) is not tuple or any(
            type(record) is not LLIRPassRunRecord for record in records
        ):
            _raise_context_error(
                code="invalid_llir_pass_records",
                message="managed pass observations must be exact immutable records",
                stage_id=CompilerStageId.CIN_LOWERING,
            )
        start = len(self._llir_pass_run_records)
        appended = tuple(
            replace(record, sequence_index=start + offset)
            for offset, record in enumerate(records)
        )
        object.__setattr__(
            self,
            "_llir_pass_run_records",
            self._llir_pass_run_records + appended,
        )

    @property
    def stage_run_records(self) -> Tuple[CompilerStageRunRecord, ...]:
        """Completed compiler stages in deterministic execution-start order."""

        return tuple(
            replace(record, sequence_index=sequence_index)
            for sequence_index, record in enumerate(self._stage_run_records)
        )

    @property
    def llir_pass_run_records(self) -> Tuple["LLIRPassRunRecord", ...]:
        """Completed nested managed passes in exact execution order."""

        return self._llir_pass_run_records
