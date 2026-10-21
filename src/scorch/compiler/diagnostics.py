"""Compiler diagnostics raised at explicit stage boundaries.

These exception types intentionally carry no mutable compiler state.  Stage-specific
callers provide the compact node or schedule context in the error message until the
later structured diagnostic artifact is introduced.
"""

from dataclasses import dataclass
from typing import Optional

from ..exceptions import (
    ScorchTypeError,
    ShapeCapacityError,
    UnsupportedScheduleError,
)


class CompilerError(Exception):
    """Base class for failures owned by the Scorch compiler pipeline.

    ``code`` names the defect, as on :class:`scorch.exceptions.ScorchError`: set on a
    class with one cause, passed by the raise (``code="..."``) of a class with several,
    and otherwise ``None``.  An error that carries ``diagnostics`` has their codes too;
    :func:`scorch.exceptions.defect_code` reads both.
    """

    code: Optional[str] = None

    def __init__(self, *args: object, code: Optional[str] = None) -> None:
        super().__init__(*args)
        if code is not None:
            self.code = code


@dataclass(frozen=True)
class CompileOptionsDiagnostic:
    """One immutable failure while snapshotting compiler configuration."""

    code: str
    field: str
    message: str


class CompileOptionsError(CompilerError):
    """Compiler configuration was malformed or internally inconsistent."""

    def __init__(self, diagnostics: tuple[CompileOptionsDiagnostic, ...]) -> None:
        if type(diagnostics) is not tuple or not diagnostics:
            raise ScorchTypeError(
                "CompileOptionsError requires immutable diagnostics",
                code="internal_argument_type_invalid",
            )
        if any(
            type(diagnostic) is not CompileOptionsDiagnostic
            for diagnostic in diagnostics
        ):
            raise ScorchTypeError(
                "CompileOptionsError requires CompileOptionsDiagnostic values",
                code="internal_argument_type_invalid",
            )
        self.diagnostics = diagnostics
        first = diagnostics[0]
        super().__init__(
            "stage=compile options: " f"{first.code} for {first.field}: {first.message}"
        )


SPMV_REFUSAL_TEXT = "Explicit schedules are not yet threaded through the SpMV compiler"


class ScheduleNotThreadedThroughSpMVError(
    UnsupportedScheduleError, CompileOptionsError
):
    """An explicit schedule for a matrix-vector product, which SpMV cannot take.

    ``matmul`` sends every 2-D times 1-D product with a sparse matrix to the SpMV
    path (the prebuilt CSR kernel, or ``ops.spmv`` when that does not match), and
    ``ops.spmv`` can also be called directly.  Neither takes a schedule yet, so a
    schedule given to either, by the ``schedule=`` keyword, a ``schedule_force``
    context or ``_compile_options``, is refused with this error before any kernel is
    built: one class and one code for the one cause, at both entry points and in
    every spelling.

    Both of the types these refusals had before they had a code: an
    :class:`scorch.exceptions.UnsupportedScheduleError`, so a ``NotImplementedError``
    (``matmul``'s), and a :class:`CompileOptionsError` (``ops.spmv``'s, whose code
    was ``unsupported_requested_schedule``).  So an ``except`` clause written for
    either still catches it.  Its one diagnostic carries ``code`` on the field
    ``requested_schedule``, and its message is ``matmul``'s former text after the
    diagnostic's prefix.
    """

    code = "schedule_not_threaded_through_spmv"

    def __init__(self, message: str = SPMV_REFUSAL_TEXT) -> None:
        self.reason = message
        CompileOptionsError.__init__(
            self,
            (
                CompileOptionsDiagnostic(
                    code=self.code, field="requested_schedule", message=message
                ),
            ),
        )

    def __reduce__(self):  # type: ignore[no-untyped-def]
        # ``args`` holds the prefixed text, which the constructor would prefix again.
        return (type(self), (self.reason,))


class UnsupportedFeature(NotImplementedError, CompilerError):
    """A valid program requests a feature the current compiler cannot lower."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[object, ...] = (),
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        if code is not None:
            self.code = code


class InvalidSchedule(ValueError, CompilerError):
    """A scheduling decision is invalid for the normalized CIN program."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[object, ...] = (),
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        if code is not None:
            self.code = code


class VerificationError(CompilerError):
    """An artifact violates an invariant at a compiler stage boundary."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[object, ...] = (),
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        if code is not None:
            self.code = code


class ShapeCapacityVerificationError(ShapeCapacityError, VerificationError):
    """:class:`scorch.exceptions.ShapeCapacityError` from a compiler stage boundary."""


class CompilerInvariantError(CompilerError):
    """An internal compiler invariant was violated."""


class CodegenError(CompilerError):
    """Verified low-level IR cannot be emitted as C++."""
