"""Typed production/debug timing for the canonical compiler stages.

One frozen :class:`CompilerStageTiming` owner is constructed per compilation
beside the exact :class:`CompileOptions` snapshot that the compilation routes
through every stage.  Stage call sites bracket their existing work with
``begin``/``commit``; a stage that fails commits nothing, so already completed
stage records are preserved exactly once in deterministic completion order and
every later stage is suppressed by the propagating original exception, matching
the managed LLIR pass partial-record policy.

Records mirror :class:`LLIRPassRunRecord`: frozen dataclasses ordered by
``sequence_index`` whose ``duration_ns`` comes from ``perf_counter_ns`` and is
excluded from equality.  The owner and its records are non-semantic
observation: they never enter cache keys, fingerprints, kernel names, emitted
C++, or build inputs.  Nested managed LLIR pass records remain owned by the
pass manager and ``CINLowerer`` and are deliberately not merged here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter_ns
from typing import TYPE_CHECKING, List, NoReturn, Optional, Tuple

from .diagnostics import CompilerInvariantError

if TYPE_CHECKING:
    from .compile_options import CompileOptions


class CompilerStageId(Enum):
    """Stable identities of the canonical timed compiler stages."""

    FRONTEND_CONSTRUCTION = "frontend_construction"
    CIN_NORMALIZATION = "cin_normalization"
    SCHEDULING = "scheduling"
    CIN_LOWERING = "cin_lowering"
    RESULT_ABI_ASSEMBLY = "result_abi_assembly"
    SCHEDULE_LOWERING = "schedule_lowering"
    CPP_GENERATION = "cpp_generation"
    KERNEL_NAME_ASSEMBLY = "kernel_name_assembly"


CANONICAL_COMPILER_STAGES: Tuple[CompilerStageId, ...] = (
    CompilerStageId.FRONTEND_CONSTRUCTION,
    CompilerStageId.CIN_NORMALIZATION,
    CompilerStageId.SCHEDULING,
    CompilerStageId.CIN_LOWERING,
    CompilerStageId.RESULT_ABI_ASSEMBLY,
    CompilerStageId.SCHEDULE_LOWERING,
    CompilerStageId.CPP_GENERATION,
    CompilerStageId.KERNEL_NAME_ASSEMBLY,
)


@dataclass(frozen=True)
class CompilerStageTimingDiagnostic:
    """Structured failure owned by the stage-timing seam itself."""

    code: str
    message: str
    stage_name: str


class CompilerStageTimingError(CompilerInvariantError):
    """A mismatched or misused stage-timing boundary failed closed."""

    def __init__(self, diagnostic: CompilerStageTimingDiagnostic) -> None:
        self.diagnostic = diagnostic
        self.diagnostics = (diagnostic,)
        super().__init__(
            "stage=compiler stage timing "
            f"stage_id={diagnostic.stage_name}: {diagnostic.code}: "
            f"{diagnostic.message}"
        )


def _raise_timing_error(
    *,
    code: str,
    message: str,
    stage: Optional[CompilerStageId] = None,
) -> NoReturn:
    raise CompilerStageTimingError(
        CompilerStageTimingDiagnostic(
            code=code,
            message=message,
            stage_name=stage.value if stage is not None else "<owner>",
        )
    )


@dataclass(frozen=True)
class CompilerStageRecord:
    """Non-semantic ordered instrumentation for one completed compiler stage."""

    sequence_index: int
    stage: CompilerStageId
    nested_within: Optional[CompilerStageId]
    duration_ns: int = field(compare=False)


@dataclass(frozen=True)
class CompilerStageToken:
    """Opaque begin marker consumed exactly once by the matching commit."""

    stage: CompilerStageId
    started_ns: int
    depth: int


@dataclass(frozen=True, eq=False)
class CompilerStageTiming:
    """One typed timing owner for one compilation's CompileOptions snapshot.

    The owner label is frozen; the internal record and open-stage payloads are
    append-only observation state, like the mutable payloads carried by the
    frozen LLIR pass artifacts.  ``begin`` validates that the timed stage holds
    the exact snapshot this owner was constructed with, so one owner cannot
    span two configurations and no timed stage can resnapshot environment or
    ``ContextVar`` state without failing closed.
    """

    compile_options: "CompileOptions"
    _open_stages: List[CompilerStageId] = field(
        default_factory=list, repr=False, compare=False
    )
    _records: List[CompilerStageRecord] = field(
        default_factory=list, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        from .compile_options import CompileOptions

        if type(self.compile_options) is not CompileOptions:
            _raise_timing_error(
                code="invalid_compile_options",
                message="stage timing requires an exact CompileOptions snapshot",
            )

    def begin(
        self,
        stage: CompilerStageId,
        *,
        compile_options: "CompileOptions",
    ) -> CompilerStageToken:
        """Open one canonical stage after proving exact snapshot identity."""

        if type(stage) is not CompilerStageId:
            _raise_timing_error(
                code="invalid_stage_id",
                message="stage timing requires an exact CompilerStageId",
            )
        if compile_options is not self.compile_options:
            _raise_timing_error(
                code="detached_compile_options",
                message=(
                    "a timed stage must route the exact CompileOptions snapshot "
                    "owned by this compilation's timing owner"
                ),
                stage=stage,
            )
        if stage in self._open_stages:
            _raise_timing_error(
                code="reentrant_stage",
                message="a canonical stage cannot nest within itself",
                stage=stage,
            )
        self._open_stages.append(stage)
        return CompilerStageToken(
            stage=stage,
            started_ns=perf_counter_ns(),
            depth=len(self._open_stages),
        )

    def commit(self, token: CompilerStageToken) -> None:
        """Record one completed stage; a failed stage never reaches commit."""

        if type(token) is not CompilerStageToken:
            _raise_timing_error(
                code="invalid_stage_token",
                message="stage timing requires the exact token begin returned",
            )
        if (
            not self._open_stages
            or self._open_stages[-1] is not token.stage
            or len(self._open_stages) != token.depth
        ):
            _raise_timing_error(
                code="unbalanced_stage_commit",
                message="commit must close the innermost open stage exactly once",
                stage=token.stage,
            )
        duration_ns = perf_counter_ns() - token.started_ns
        self._open_stages.pop()
        nested_within = self._open_stages[-1] if self._open_stages else None
        self._records.append(
            CompilerStageRecord(
                sequence_index=len(self._records),
                stage=token.stage,
                nested_within=nested_within,
                duration_ns=duration_ns,
            )
        )

    @property
    def records(self) -> Tuple[CompilerStageRecord, ...]:
        """Completed stage records in deterministic completion order."""

        return tuple(self._records)
