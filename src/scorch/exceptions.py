"""Domain exceptions raised at Scorch's public tensor boundaries."""

import re
from typing import Optional, Sequence, Tuple


class ScorchError(Exception):
    """Base class for errors reported by Scorch."""


class TensorValidationError(ScorchError, ValueError):
    """A tensor or one of its value objects is internally inconsistent."""


class TensorFormatError(TensorValidationError):
    """A tensor format cannot be parsed or is structurally invalid."""


class TensorLayoutError(TensorValidationError):
    """Logical and physical layout metadata is inconsistent."""


class TensorIndexError(TensorValidationError):
    """Sparse index arrays violate their declared layout."""


class TensorStorageError(TensorValidationError):
    """Tensor values and sparse index storage are incompatible."""


class TensorDeviceError(TensorValidationError):
    """A tensor uses a device unsupported by Scorch."""


class TensorTypeError(ScorchError, TypeError):
    """A public tensor API received an object of the wrong type."""


class CompileSpecError(TensorValidationError):
    """A compile-only tensor specification is invalid."""


class UnsupportedDtypePromotionError(TensorValidationError):
    """Operands' value dtypes promote to a dtype Scorch does not store.

    int8 with uint8 promotes to int16 by PyTorch's rule; Scorch stores neither
    int16 nor a narrower type that holds both, so the operation is refused.
    ``code`` names the defect.
    """

    code = "unsupported_dtype_promotion"


class OperandDtypeMismatchError(TensorValidationError, RuntimeError):
    """A product's operands hold values of different dtypes.

    PyTorch raises ``RuntimeError`` for a mixed-dtype ``matmul``, ``mv``, ``mm``, ``@``
    and ``einsum`` and does not promote, so Scorch raises this, a ``RuntimeError`` as
    well, before it looks up or builds any kernel.  ``code`` names the defect.
    """

    code = "operand_dtype_mismatch"


class EinsumRepeatedIndexError(CompileSpecError):
    """One term of an einsum expression names an index twice (``"ii->i"``, ``"i->ii"``).

    An operand term that repeats a letter asks for the operand's diagonal, which no
    kernel computes yet, and a result term that repeats one has no meaning (PyTorch
    rejects it too).  ``scorch.einsum`` refuses both when it checks the expression's
    labels, before it selects a loop order or looks up a kernel.  ``operand`` is the
    position of the operand whose term repeats the letter, or ``None`` for the result
    term; ``index`` is the letter; ``code`` names the defect.
    """

    code = "einsum_repeated_index"

    def __init__(self, message: str, operand: Optional[int], index: str) -> None:
        super().__init__(message)
        self.operand = operand
        self.index = index

    def __reduce__(self):  # type: ignore[no-untyped-def]
        # ``args`` holds the message alone, so the default reduction would drop the rest.
        return (type(self), (str(self), self.operand, self.index))


class EinsumLoopOrderError(CompileSpecError):
    """The loop-order sort left some of an einsum expression's indices unplaced.

    ``utils.topo_sort_characters`` orders the index letters so that each operand is read
    in its storage order where the operands allow it, and gives up some of those orders
    where they contradict one another.  That resolution ends on every graph, so this is
    raised only if it ends with a cycle still in place, which would be a defect in it;
    the expression is refused rather than lowered with a partial order.  ``code`` names
    the defect.
    """

    code = "einsum_loop_order_unresolved"


class RepeatedCoordinateError(TensorIndexError):
    """A COO tensor stores one coordinate tuple at two positions.

    Scorch's kernels read every stored coordinate once and never sum two of them, so raw
    COO arrays, or a COO operand whose index arrays were written in place, that repeat a
    tuple are refused, and the caller sums the repeats explicitly with ``coalesce()``
    (Bobby's decision of 2026-10-12).  A torch COO tensor's repeats are summed on import
    instead, as torch defines them.  ``position`` is the later of the two positions,
    ``coordinate`` the repeated tuple, and ``code`` names the defect.  The native index
    check reports the same defect as text, which :func:`raise_if_native_repeat` turns
    into this error.
    """

    code = "coo_repeated_coordinate"

    def __init__(
        self, message: str, position: int, coordinate: Tuple[int, ...]
    ) -> None:
        super().__init__(message)
        self.position = position
        self.coordinate = coordinate

    def __reduce__(self):  # type: ignore[no-untyped-def]
        # ``args`` holds the message alone, so the default reduction would drop the rest.
        return (type(self), (str(self), self.position, self.coordinate))


COALESCE_ADVICE = "call coalesce() to sum repeated coordinates"


def format_coordinate(coordinate: Sequence[int]) -> str:
    """``(1, 2)``, and ``(2)`` for one level: the native check's spelling of a tuple."""
    return "(" + ", ".join(str(int(part)) for part in coordinate) + ")"


# csrc/native_abi.h, check_coo_strictly_ascending: its text is this pattern's contract.
_NATIVE_REPEAT = re.compile(
    r"([^\n]*?)COO coordinate \((-?\d+(?:, -?\d+)*)\) is stored at positions (\d+) and "
    r"(\d+); call coalesce\(\) to sum repeated coordinates \[coo_repeated_coordinate\]"
)


def raise_if_native_repeat(error: BaseException) -> None:
    """Raise the structured form of the native check's refusal of a repeated coordinate.

    A kernel's index check can only raise ``c10::Error``, which reaches Python as a
    ``RuntimeError`` carrying text.  Every Python call site of a kernel that validates a
    COO operand calls this inside ``except RuntimeError`` and re-raises the original
    error when it returns, so the refusal leaves Scorch as a
    :class:`RepeatedCoordinateError` and every other error is unchanged.  The ``try``
    costs nothing when nothing is raised (CPython 3.11+).
    """
    match = _NATIVE_REPEAT.search(str(error))
    if match is None:
        return
    prefix, digits, first, second = match.groups()
    raise RepeatedCoordinateError(
        f"{prefix}COO coordinate ({digits}) is stored at positions {first} and "
        f"{second}; {COALESCE_ADVICE}",
        int(second),
        tuple(int(part) for part in digits.split(", ")),
    ) from error
