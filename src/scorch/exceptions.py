"""Domain exceptions raised at Scorch's public tensor boundaries."""

import re
from dataclasses import FrozenInstanceError
from typing import Optional, Sequence, Tuple, Type, TypeVar


class ScorchError(Exception):
    """Base class for errors reported by Scorch.

    ``code`` names the defect.  A class with one cause sets it on the class.  A raise
    of a class that covers several causes passes it, ``code="..."``, and that
    instance's code is then the one passed; with none passed it stays ``None``, so a
    raise names its own cause.  Read a code with :func:`defect_code`, which also reads
    a compiler error's diagnostics and the exception an error was raised from.
    """

    code: Optional[str] = None

    def __init__(self, *args: object, code: Optional[str] = None) -> None:
        super().__init__(*args)
        if code is not None:
            self.code = code


# The built-in exception types Scorch raises, each with a Scorch variant that is still
# that type, so an ``except`` clause written for the built-in catches it, and that
# carries the raise's ``code``.
class ScorchTypeError(ScorchError, TypeError):
    """A ``TypeError`` raised by Scorch, with a ``code``."""


class ScorchValueError(ScorchError, ValueError):
    """A ``ValueError`` raised by Scorch, with a ``code``."""


class ScorchAssertionError(ScorchError, AssertionError):
    """A failed internal check, an ``AssertionError``, with a ``code``.

    Raised where an ``assert`` statement stood, so the check also runs under
    ``python -O`` and the failure names the check that failed.
    """


class ScorchNotImplementedError(ScorchError, NotImplementedError):
    """A ``NotImplementedError`` raised by Scorch, with a ``code``."""


class ScorchRuntimeError(ScorchError, RuntimeError):
    """A ``RuntimeError`` raised by Scorch, with a ``code``."""


class ScorchLookupError(ScorchError, LookupError):
    """A ``LookupError`` raised by Scorch, with a ``code``."""


class ScorchKeyError(ScorchError, KeyError):
    """A ``KeyError`` raised by Scorch, with a ``code``.

    A mapping's ``__getitem__`` raises it for a missing key: ``Mapping.get`` and ``in``
    catch ``KeyError``, so a narrower ``LookupError`` would break them.
    """


class ScorchImportError(ScorchError, ImportError):
    """An ``ImportError`` raised by Scorch, with a ``code``."""


class ScorchOSError(ScorchError, OSError):
    """An ``OSError`` raised by Scorch, with a ``code``."""


class ScorchFrozenInstanceError(ScorchError, FrozenInstanceError):
    """A write to a sealed value object, a ``FrozenInstanceError``, with a ``code``."""


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


class KeywordArgumentError(ScorchTypeError):
    """A product entry (``matmul``, ``einsum``, ``ops.spmv``) was given a keyword argument
    it does not take, or two spellings of one argument that disagree.

    A ``TypeError``, because Python and torch raise one for the same mistakes.  Each raise
    passes its code:

    * ``unknown_keyword_argument``: a name the entry does not take, with Python's own
      message for it;
    * ``format_keywords_conflict``: ``format=`` and its alias ``output_format=`` naming
      different formats.
    """


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


class EinsumScalarResultError(CompileSpecError):
    """An einsum expression with no index letter in any term (``"->"``, ``",->"``).

    Its result is a scalar, the product of rank-0 operands, which PyTorch computes.  A
    scalar result over at least one index (``"ij->"``, ``"i,i->"``) is computed: it is
    the sum over every index, a loop nest around one value.  With no index there is no
    loop, and neither compiler lowers a program that is not a loop nest.
    ``scorch.einsum`` refuses the expression when it checks the labels, before it selects
    a loop order or looks up a kernel.  ``code`` names the defect.
    """

    code = "einsum_scalar_result_unsupported"


class MatmulOperandRankError(CompileSpecError):
    """A ``matmul`` whose operand ranks its generated path does not compute yet.

    The generated path follows ``torch.matmul``'s rank rules for operands of rank one
    and two: matrix x matrix, vector x matrix and matrix x vector are computed, and
    vector x vector, whose result is a scalar, is refused by ``einsum`` by code until a
    scalar result is computed.  A batched product (an operand of rank three or more) is not
    computed yet.  Before the refusal had a code, ``matmul`` sent every such pair to
    ``einsum("ij,jk->ik")``, which raised a plain ``CompileSpecError`` on the rank, as
    it did for a vector first operand; this is a ``CompileSpecError`` too.  ``code``
    names the defect.
    """

    code = "matmul_operand_rank_unsupported"

    def __init__(self, ranks: Tuple[int, int]) -> None:
        self.ranks = ranks
        super().__init__(
            f"matmul of operands of ranks {self.ranks[0]} and {self.ranks[1]}: the "
            "generated path computes operands of rank one and two only"
        )


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


class UnsupportedScheduleError(ScorchError, NotImplementedError):
    """An explicit schedule was given for a product that no compiler path can schedule.

    A schedule reaches a product three ways: the ``schedule=`` keyword, a
    ``schedule_force`` context, or ``_compile_options`` whose ``requested_schedule`` is
    set.  Each spelling is refused the same way, before any kernel is looked up or
    built, rather than ignored.  Never raised itself: each refusal raises a subclass
    whose ``code`` names the defect, one class and one code per cause, whichever
    spelling and entry point the schedule came through:

    - :class:`ScheduleWithoutSparseOperandError`, below;
    - ``scorch.compiler.diagnostics.ScheduleNotThreadedThroughSpMVError``, kept there
      because it is a ``CompileOptionsError`` as well, which this module cannot import
      (``scorch.compiler.diagnostics`` imports this one).

    A ``NotImplementedError`` as well, which is what ``matmul``'s refusals raised before
    they had a code, so an ``except NotImplementedError`` clause still catches them.
    """

    code: str = ""


class ScheduleWithoutSparseOperandError(UnsupportedScheduleError):
    """An explicit schedule for a product of two dense operands.

    Two dense torch tensors, or two dense ``STensor`` operands, are multiplied by
    ``torch.matmul``; no Scorch compiler takes the product, so there is nothing a
    schedule could apply to.  The message is the text ``matmul`` raised before the
    refusal had a code.
    """

    code = "schedule_without_sparse_operand"

    def __init__(
        self,
        message: str = "Explicit compiler schedules currently require a sparse operand",
    ) -> None:
        super().__init__(message)


class DerivativeUnsupportedError(ScorchNotImplementedError):
    """A gradient was asked of a result whose operation has no derivative.

    torch's convention for such an operation: the forward pass computes, so inference
    with parameters that require grad keeps working, and the result's ``grad_fn``
    raises when a gradient is asked of it, naming what was to be differentiated.  Never
    raised itself: each refusal raises the subclass of its operation, whose ``code``
    names it.  A ``ScorchNotImplementedError``, which is what these refusals raised
    before they had a class of their own, so an ``except`` clause written for it or for
    ``NotImplementedError`` still catches them.
    """

    code: str = ""


class MatmulDerivativeError(DerivativeUnsupportedError):
    """``scorch.matmul``'s result has no derivative for its operands and keywords."""

    code = "matmul_derivative_unsupported"


class EinsumDerivativeError(DerivativeUnsupportedError):
    """``scorch.einsum``'s result has no derivative: the call is in a form the
    derivative does not read, it carries fused post-operations, or the compiler refused
    the einsum that computes an operand's gradient."""

    code = "einsum_derivative_unsupported"


class SparseLinearDerivativeError(DerivativeUnsupportedError):
    """``scorch.sparse_linear_fm``'s result has no derivative for its weight's mode
    order."""

    code = "sparse_linear_derivative_unsupported"


class RepeatedCoordinateError(TensorIndexError):
    """A COO tensor stores one coordinate tuple at two positions.

    Scorch's kernels read every stored coordinate once and never sum two of them, so raw
    COO arrays, or a COO operand whose index arrays were written in place, that repeat a
    tuple are refused, and the caller sums the repeats explicitly with ``coalesce()``
    (Bobby's decision of 2026-10-12).  A torch COO tensor's repeats are summed on import
    instead, as torch defines them.  ``position`` is the later of the two positions,
    ``coordinate`` the repeated tuple, and ``code`` names the defect.  The native index
    check reports the same defect as text, which :func:`raise_if_native_coded` turns
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


class IndexCapacityError(TensorValidationError, RuntimeError):
    """A tensor needs an index the kernels' 32-bit positions and coordinates cannot hold.

    Every kernel declares a level's positions, its loop variables and its coordinates as
    32-bit ``int``.  So each storage level of an operand or a result may hold at most
    ``2**31 - 1`` positions, and no extent may pass that either.  What counts is the
    number of positions a level stores, not the tensor's logical size: a 64,000 x 64,000
    CSR matrix with two million entries stores 64,001 row offsets and two million column
    coordinates, and is computed.  Raised before a kernel is built where the shapes and
    formats decide it, and from the native boundary's own check otherwise, whose text
    :func:`raise_if_native_coded` turns into this error.  A ``RuntimeError`` as well,
    which is what the native check raised before it had a code.  ``code`` names the
    defect.
    """

    code = "index_position_exceeds_int32"


class IndexExtentMismatchError(CompileSpecError, RuntimeError):
    """A product's operands give one index two different extents.

    A kernel bounds an index's loop by one operand's extent and reads every other operand
    that carries the index by it, so a shorter operand would be read past its end.
    ``einsum`` and ``spmv`` refuse an index whose operands give it different extents
    before any kernel is built, and the prebuilt kernels refuse the same operands in their
    native entry checks before they read anything, with this code in the text
    (:func:`raise_if_native_coded`).  ``matmul`` reaches one of the three on every route
    but two dense layouts, where it gives torch's refusal this class.  PyTorch raises
    ``RuntimeError`` for the same mistake in ``matmul``, ``mv``, ``mm``, ``@`` and
    ``einsum``, and so did the prebuilt kernels' native checks, so this is a
    ``RuntimeError`` as well as the ``CompileSpecError`` ``einsum`` raised.
    The compiler refuses the mismatch between known shapes before it builds a kernel, and
    a generated kernel whose extents are not all compiled in refuses it at its entry;
    :func:`raise_if_native_coded` turns that refusal's text into this error.  ``code``
    names the defect.
    """

    code = "einsum_index_extent_mismatch"


class GeneratedKernelCacheError(ScorchError, RuntimeError):
    """A kept generated kernel is not the kernel its call builds.

    Raised only with ``SCORCH_CHECK_KERNEL_CACHE_HITS=1`` (or the cell
    ``scorch.stensor._CHECK_GENERATED_KERNEL_HITS``), under which every hit of the
    conversion, addition and SpMV kernel cache also builds its kernel the full way
    and compares the two kernels (name and shared-object path).  A difference means
    the cache key left out something the lowering or the build reads.  ``code``
    names the defect.
    """

    code = "kept_kernel_differs_from_built"


INT64_MAX = (1 << 63) - 1


class ShapeCapacityError(Exception):
    """A shape that signed int64 cannot hold: one extent, or the product of all of them.

    Scorch holds every extent and every element count as a signed 64-bit integer, as
    torch's sizes and the native boundary's ``int64_t`` shapes do.  A shape is refused
    when one extent passes ``2**63 - 1``, or when the product of its extents does.  A
    shape with a zero extent holds no element, so its count is 0 whatever its other
    extents are, and it is not refused for its count.

    Never raised itself.  Each boundary raises a subclass that is also the type it
    raised before the refusal had a code: :class:`ShapeCapacityLayoutError` a
    :class:`TensorLayoutError`, :class:`ShapeCapacitySpecError` a
    :class:`CompileSpecError`, :class:`ShapeCapacityKernelError` a ``ValueError``, and
    the compiler's ``ShapeCapacityVerificationError`` a ``VerificationError``.  So an
    ``except`` clause written for that type still catches it, and this class catches
    the refusal at every boundary.  The message is the boundary's former text, then the
    extent at fault or the element count, its value and the bound.  ``code`` names the
    defect.  ``owner`` names the shape, ``mode`` the extent at fault (``None`` when the
    element count is), ``value`` that extent or the count, and ``bound`` the limit.
    """

    code: Optional[str] = "shape_exceeds_int64"
    owner: str = ""
    mode: Optional[int] = None
    value: int = 0
    bound: int = INT64_MAX

    def retyped(self, error_type: Type["_ShapeError"]) -> "_ShapeError":
        """This refusal, with the same text and fields, as ``error_type``."""
        error = error_type(*self.args)
        error.owner = self.owner
        error.mode = self.mode
        error.value = self.value
        error.bound = self.bound
        return error


_ShapeError = TypeVar("_ShapeError", bound=ShapeCapacityError)


class ShapeCapacityLayoutError(ShapeCapacityError, TensorLayoutError):
    """The refusal from a tensor layout, which every tensor construction builds."""


class ShapeCapacitySpecError(ShapeCapacityError, CompileSpecError):
    """The refusal from a compile-only spec or a runtime shape snapshot."""


class ShapeCapacityKernelError(ShapeCapacityError, ValueError):
    """The refusal from a kernel's ABI description of a known shape."""


def _integer_text(value: int) -> str:
    """``value`` in decimal, or the power of two it reaches when it is too long to read.

    A shape refused for one extent can hold any integer, and ``str`` refuses one past
    4,300 digits.
    """
    if value.bit_length() <= 128:
        return str(value)
    return f"2**{value.bit_length() - 1} or more"


def shape_capacity_error(
    error_type: Type[_ShapeError],
    message: str,
    owner: str,
    shape: Sequence[int],
    mode: Optional[int],
    value: int,
) -> _ShapeError:
    """The refusal of ``owner`` for extent ``mode`` of ``shape``, or for its count.

    ``mode`` is ``None`` when the element count ``value`` is at fault; ``shape`` is then
    every extent, each within int64.  ``message`` is the text the boundary raised before
    the refusal had a code, and the details follow it.
    """
    bound = f"the signed int64 bound {INT64_MAX} (2**63 - 1)"
    if mode is None:
        extents = ", ".join(str(extent) for extent in shape)
        detail = (
            f"shape ({extents}) holds {_integer_text(value)} elements, "
            f"more than {bound}"
        )
    else:
        detail = f"extent {mode} is {_integer_text(value)}, more than {bound}"
    error = error_type(f"{message}: {detail}")
    error.owner = owner
    error.mode = mode
    error.value = value
    return error


COALESCE_ADVICE = "call coalesce() to sum repeated coordinates"


def format_coordinate(coordinate: Sequence[int]) -> str:
    """``(1, 2)``, and ``(2)`` for one level: the native check's spelling of a tuple."""
    return "(" + ", ".join(str(int(part)) for part in coordinate) + ")"


# csrc/native_abi.h, check_coo_strictly_ascending: its text is this pattern's contract.
_NATIVE_REPEAT = re.compile(
    r"([^\n]*?)COO coordinate \((-?\d+(?:, -?\d+)*)\) is stored at positions (\d+) and "
    r"(\d+); call coalesce\(\) to sum repeated coordinates \[coo_repeated_coordinate\]"
)


# csrc/native_abi.h, SCORCH_INDEX_CAPACITY_CODE: every native refusal under the 32-bit
# position bound ends its line with this code.
_NATIVE_CAPACITY_CODE = " [index_position_exceeds_int32]"

# csrc/native_abi.h, SCORCH_EXTENT_MISMATCH_CODE: a prebuilt kernel's refusal of operands
# that give one index two extents ends its line with this code, and so do a generated
# kernel's (validate_jit_index_extents and the extra tensor check).
_NATIVE_EXTENT_CODE = " [einsum_index_extent_mismatch]"


def raise_if_native_coded(error: BaseException) -> None:
    """Raise the structured form of a native refusal whose text carries a code.

    A kernel's checks can only raise ``c10::Error``, which reaches Python as a
    ``RuntimeError`` carrying text.  Three refusals mark that text with a code: a repeated
    COO coordinate, which becomes :class:`RepeatedCoordinateError`, an index past the
    32-bit position bound, which becomes :class:`IndexCapacityError`, and operands that
    give one index two extents, which becomes :class:`IndexExtentMismatchError`.  Every
    Python call site of a kernel calls this inside ``except RuntimeError`` and re-raises
    the original error when it returns, so a coded refusal leaves Scorch as its
    structured error and every other error is unchanged.  The structured message is the
    native line without the code, which the error's ``code`` attribute carries.  The
    ``try`` costs nothing when nothing is raised (CPython 3.11+).
    """
    text = str(error)
    match = _NATIVE_REPEAT.search(text)
    if match is not None:
        prefix, digits, first, second = match.groups()
        raise RepeatedCoordinateError(
            f"{prefix}COO coordinate ({digits}) is stored at positions {first} and "
            f"{second}; {COALESCE_ADVICE}",
            int(second),
            tuple(int(part) for part in digits.split(", ")),
        ) from error
    if _NATIVE_CAPACITY_CODE in text:
        # The line that carries the code: with TORCH_SHOW_CPP_STACKTRACES a C++ backtrace
        # follows it.
        line = next(part for part in text.splitlines() if _NATIVE_CAPACITY_CODE in part)
        raise IndexCapacityError(line.replace(_NATIVE_CAPACITY_CODE, "")) from error
    if _NATIVE_EXTENT_CODE in text:
        line = next(part for part in text.splitlines() if _NATIVE_EXTENT_CODE in part)
        raise IndexExtentMismatchError(line.replace(_NATIVE_EXTENT_CODE, "")) from error


def defect_code(error: Optional[BaseException]) -> Optional[str]:
    """The code that names the defect ``error`` reports, or ``None`` if it has none.

    The error's own ``code`` comes first, then the first code among its
    ``diagnostics`` (the compiler's errors carry their codes there), then its
    ``defect``'s code (the new compiler's errors), then the same for the exception it
    was raised from (``raise ... from error``), and so on down that chain.
    """

    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        code = getattr(error, "code", None)
        if type(code) is str and code:
            return code
        for diagnostic in getattr(error, "diagnostics", None) or ():
            code = getattr(diagnostic, "code", None)
            if type(code) is str and code:
                return code
        code = getattr(getattr(error, "defect", None), "code", None)
        if type(code) is str and code:
            return code
        error = error.__cause__
    return None
