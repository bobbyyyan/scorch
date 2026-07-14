"""Compiler diagnostics raised at explicit stage boundaries.

These exception types intentionally carry no mutable compiler state.  Stage-specific
callers provide the compact node or schedule context in the error message until the
later structured diagnostic artifact is introduced.
"""

from dataclasses import dataclass


class CompilerError(Exception):
    """Base class for failures owned by the Scorch compiler pipeline."""


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
            raise TypeError("CompileOptionsError requires immutable diagnostics")
        if any(
            type(diagnostic) is not CompileOptionsDiagnostic
            for diagnostic in diagnostics
        ):
            raise TypeError(
                "CompileOptionsError requires CompileOptionsDiagnostic values"
            )
        self.diagnostics = diagnostics
        first = diagnostics[0]
        super().__init__(
            "stage=compile options: " f"{first.code} for {first.field}: {first.message}"
        )


class UnsupportedFeature(CompilerError):
    """A valid program requests a feature the current compiler cannot lower."""


class InvalidSchedule(CompilerError):
    """A scheduling decision is invalid for the normalized CIN program."""


class VerificationError(CompilerError):
    """An artifact violates an invariant at a compiler stage boundary."""

    def __init__(self, message: str, *, diagnostics: tuple[object, ...] = ()) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class CompilerInvariantError(CompilerError):
    """An internal compiler invariant was violated."""


class CodegenError(CompilerError):
    """Verified low-level IR cannot be emitted as C++."""
