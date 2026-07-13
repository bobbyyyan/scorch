"""Compiler diagnostics raised at explicit stage boundaries.

These exception types intentionally carry no mutable compiler state.  Stage-specific
callers provide the compact node or schedule context in the error message until the
later structured diagnostic artifact is introduced.
"""


class CompilerError(Exception):
    """Base class for failures owned by the Scorch compiler pipeline."""


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
