from __future__ import annotations

from array import array as _Array
from collections import deque as _Deque
from collections.abc import Mapping as _AbcMapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    ClassVar,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from .exceptions import TensorFormatError, TensorTypeError

if TYPE_CHECKING:
    from .compiler import llir


class LevelType(Enum):
    """Per-mode storage discipline for one tensor dimension.

    Scorch describes a sparse tensor's physical layout the TACO way: as an
    ordered, per-mode sequence of level types (one per dimension, in storage
    order). A single ``LevelType`` answers, for one mode, whether every
    coordinate is stored (dense) or only the nonzeros are (compressed /
    coordinate / singleton).

    There are four level types:

    - ``DENSE`` (``"d"``) -- no coordinates stored; the mode occupies its full
      extent and positions are computed arithmetically. Iterates every index
      ``0 .. size-1`` with a plain counted loop.
    - ``COMPRESSED`` (``"s"``) -- CSR-style segmented storage: a ``pos``
      (row-pointer) array indexing into a ``crd`` (coordinate) array. Only
      nonzeros are stored; iteration walks ``pos[parent] .. pos[parent+1]``.
    - ``COORDINATE`` (``"o"``) -- flat COO-style coordinate list; iteration
      reads stored coordinates directly over the position range.
    - ``SINGLETON`` (``"singleton"``) -- one coordinate per parent (the COO
      tail companion to a coordinate head). Declared in the type system but
      **not yet lowered** by codegen (see Notes).

    Notes
    -----
    ``SINGLETON`` is the only level type with **no single-character alias**;
    its enum value is the full word ``"singleton"``. The single-letter
    alphabet is ``d`` / ``s`` / ``o``. Note also that ``s`` and ``c`` are
    synonyms -- both parse to ``COMPRESSED`` (there is no ``c``-vs-``s``
    distinction), as do the words ``"sparse"`` and ``"compressed"``.

    ``SINGLETON`` is parsed consistently by the canonical format parser but is
    still reserved for future lowering; generated kernels currently support
    ``DENSE`` / ``COMPRESSED`` / ``COORDINATE``.

    Examples
    --------
    >>> import scorch
    >>> from scorch.format import LevelType
    >>> LevelType.DENSE.value
    'd'
    >>> LevelType.COMPRESSED.value
    's'

    See Also
    --------
    LevelFormat : One mode's format (a ``LevelType`` plus optional bit width).
    TensorFormat : The ordered per-mode list for a whole tensor.
    """

    DENSE = "d"
    COMPRESSED = "s"
    SINGLETON = "singleton"
    COORDINATE = "o"


# Canonical mapping from string aliases to LevelType.
_STR_TO_LEVEL_TYPE = {
    "dense": LevelType.DENSE,
    "d": LevelType.DENSE,
    "compressed": LevelType.COMPRESSED,
    "sparse": LevelType.COMPRESSED,
    "c": LevelType.COMPRESSED,
    "s": LevelType.COMPRESSED,
    "singleton": LevelType.SINGLETON,
    "single": LevelType.SINGLETON,
    "coordinate": LevelType.COORDINATE,
    "coord": LevelType.COORDINATE,
    "o": LevelType.COORDINATE,
}


def _actual_type_name(value: object) -> str:
    """Return the real type name without invoking caller-owned descriptors."""

    name = type.__dict__["__name__"].__get__(type(value), type)
    return str.__str__(name)


# ``collections.abc`` registers several concrete standard-library types
# virtually rather than by inheritance -- ``Mapping not in dict.__mro__`` and
# ``Sequence not in list.__mro__`` -- so an MRO-identity recognizer has to name
# them explicitly.  Keep the complete concrete Sequence set that the old ABC
# check accepted: some of these cannot contain valid level aliases, but
# recognition must not change which boundary reports that content error.
# Anything else must genuinely derive from the ABC; arbitrary third-party
# virtual registrations fail closed exactly like every other foreign
# lookalike here.
_MAPPING_BASES: Tuple[type, ...] = (_AbcMapping, dict, MappingProxyType)
_SEQUENCE_BASES: Tuple[type, ...] = (
    Sequence,
    list,
    tuple,
    range,
    str,
    bytes,
    bytearray,
    memoryview,
    _Deque,
    _Array,
)


def _derives_from(value_type: type, *bases: type) -> bool:
    """Whether ``value_type``'s real MRO contains any of ``bases``.

    This exists instead of ``issubclass``/``isinstance`` against the
    ``collections.abc`` ABCs.  ``ABCMeta``'s subclass check inserts the
    candidate into its positive and negative ``WeakSet`` caches, and those
    inserts call the candidate metaclass's ``__hash__`` and ``__eq__``.  A
    caller-owned metaclass therefore observed public format validation, and
    a raising one escaped it as a bare ``RuntimeError`` — including on the
    success path for a genuine ``Sequence`` subclass.

    Identity membership in the MRO, read through the base ``type``
    descriptor, answers the same question for every genuine subclass while
    running no caller-owned code.  Virtual (``register``-ed) subclasses are
    not recognized and therefore fail closed, which is exactly how this
    boundary already treats every other foreign lookalike.
    """

    try:
        mro = type.__dict__["__mro__"].__get__(value_type, type)
    except Exception:
        return False
    if type(mro) is not tuple:
        return False
    for entry in mro:
        for base in bases:
            if entry is base:
                return True
    return False


def _parse_level_type(s: str) -> LevelType:
    """Convert one canonicalized string alias to a :class:`LevelType`."""
    string_type = type(s)
    if string_type is str:
        text = s
        key = s.strip().lower()
    elif issubclass(string_type, str):
        # Bypass every subclass hook while preserving real ``str`` subclasses
        # as accepted aliases.  The base descriptors return exact strings, so
        # both lookup and any failure rendering below are callback-free.
        text = str.__str__(s)
        key = str.lower(str.strip(text))
    else:
        raise TensorTypeError(
            f"level format must be a string or LevelType, got {_actual_type_name(s)}"
        )
    try:
        return _STR_TO_LEVEL_TYPE[key]
    except KeyError as error:
        aliases = ", ".join(sorted(_STR_TO_LEVEL_TYPE))
        raise TensorFormatError(
            f"invalid level format {text!r}; expected one of: {aliases}"
        ) from error


@dataclass(frozen=True, init=False)
class LevelFormat:
    """The format of a single tensor mode: a level type plus optional bit width.

    A ``LevelFormat`` bundles one :class:`LevelType` -- ``DENSE``,
    ``COMPRESSED``, ``COORDINATE``, or ``SINGLETON`` -- with an optional index
    bit-width hint. A :class:`TensorFormat` is an ordered list of these, one
    per mode.

    Parameters
    ----------
    mode : str or LevelType
        The level type for this mode. May be a :class:`LevelType` directly, or
        a string alias parsed via the full alias table (so it accepts
        ``"dense"``/``"d"``; ``"compressed"``/``"sparse"``/``"c"``/``"s"``;
        ``"coordinate"``/``"coord"``/``"o"``; and ``"singleton"``/``"single"``).
        An unknown alias raises ``ValueError``.
    bit_width : int, optional
        Reserved index bit-width hint for optimization. Stored but not
        otherwise exercised by the mainline path.

    Notes
    -----
    ``s`` and ``c`` are synonyms for ``COMPRESSED`` -- do not read a semantic
    difference into them. ``str(level_format)`` returns the underlying
    ``LevelType`` value string, e.g. ``str(LevelFormat("d")) == "d"``.

    Examples
    --------
    >>> from scorch.format import LevelFormat, LevelType
    >>> LevelFormat("s").get_level_type() is LevelType.COMPRESSED
    True
    >>> str(LevelFormat(LevelType.DENSE))
    'd'

    See Also
    --------
    LevelType : The four per-mode storage disciplines.
    TensorFormat : The ordered per-mode list for a whole tensor.
    """

    _mode: LevelType
    _bit_width: Optional[int]

    def __init__(
        self,
        mode: Union[str, LevelType],
        bit_width: Optional[int] = None,
    ):
        mode_type = type(mode)
        parsed_mode: LevelType
        if mode_type is str:
            parsed_mode = _parse_level_type(cast(str, mode))
        elif mode_type is LevelType:
            parsed_mode = cast(LevelType, mode)
        elif issubclass(mode_type, str):
            parsed_mode = _parse_level_type(cast(str, mode))
        else:
            raise TensorTypeError(
                "level format mode must be a string alias or LevelType, "
                f"got {_actual_type_name(mode)}"
            )
        if bit_width is not None:
            bit_width_type = type(bit_width)
            if bit_width_type is int:
                canonical_bit_width = bit_width
            elif bit_width_type is bool or not issubclass(bit_width_type, int):
                raise TensorTypeError("level format bit_width must be an integer")
            else:
                # Validate the underlying integer without trusting subclass
                # comparison or conversion hooks.  Retain the caller's value
                # here for constructor compatibility; retaining boundaries
                # canonicalize it to the exact integer proven by this check.
                canonical_bit_width = int.__int__(bit_width)
            if canonical_bit_width <= 0:
                raise TensorFormatError("level format bit_width must be positive")
        object.__setattr__(self, "_mode", parsed_mode)
        object.__setattr__(self, "_bit_width", bit_width)

    def get_level_type(self) -> LevelType:
        return self._mode

    @property
    def bit_width(self) -> Optional[int]:
        """Optional storage-width hint for this level."""
        return self._bit_width

    def to_dict(self) -> Mapping[str, Any]:
        """Return a JSON-compatible canonical representation."""
        return {"type": self._mode.name.lower(), "bit_width": self._bit_width}

    def __str__(self):
        # return f'"{self._mode.value}"'
        return str(self._mode.value)

    def __repr__(self):
        return str(self)


class LevelPack:
    def __init__(self, level_type: LevelType, tensor: llir.Expr, mode: int, level: int):
        self.level_type = level_type
        self.tensor = tensor
        self.mode = mode
        self.level = level
        self.arrays = self.get_arrays(level_type, tensor, mode, level)

    @staticmethod
    def get_arrays(
        level_type: LevelType, tensor: llir.Expr, mode: int, level: int
    ) -> List[llir.Expr]:
        # TODO: implement this
        raise NotImplementedError


@dataclass(frozen=True, init=False)
class TensorFormat:
    """The whole-tensor physical layout: an ordered list of per-mode formats.

    A ``TensorFormat`` is the central format abstraction -- an ordered list of
    :class:`LevelFormat` objects, one per mode in storage order. The compiler
    pipeline keys kernel generation and iteration strategy off it. Reading the
    format left-to-right tells you how you descend from the outermost mode to
    the innermost. The familiar names map directly: CSR is ``"ds"`` (dense
    rows, compressed columns), COO is ``"oo"``, a dense matrix is ``"dd"``, and
    doubly-compressed CSR is ``"ss"``.

    Parameters
    ----------
    level_formats : LevelFormat or list of LevelFormat or list of str or str, optional
        The per-mode formats. Accepted shapes:

        - ``None`` -> empty format (order 0).
        - a single :class:`LevelFormat` -> a one-mode format.
        - a ``list`` of :class:`LevelFormat` -> used directly.
        - a ``list`` of ``str`` aliases -> each parsed to a ``LevelFormat``
          (e.g. ``["dense", "compressed"]`` for CSR).
        - a compact ``str`` such as ``"ds"`` -> one character per mode.
        - a long alias such as ``"dense"`` -> one mode.
        - a canonical comma string such as ``"d,s"`` -> one token per mode.

    Notes
    -----
    Parsing is centralized in :func:`parse_format`; constructor calls, compiler
    inputs, compact strings, long aliases, sequences, and the comma-delimited
    display form all use the same grammar.

    ``s`` and ``c`` are synonyms for ``COMPRESSED``. ``singleton`` constructs
    fine here but is not lowered by codegen (reserved / not-yet-supported).

    ``__str__`` is comma-joined (``str(TensorFormat("ds")) == "d,s"``), and
    that representation can be parsed again. :meth:`serialize` includes level
    bit widths and is the canonical cache/metadata form.

    The implicit / fill value is fixed at ``0.0`` (class attribute
    ``_fill_value``); non-zero fill values are not supported.

    Examples
    --------
    >>> from scorch.format import TensorFormat, LevelType
    >>> csr = TensorFormat("ds")                     # bare string -> CSR
    >>> csr == TensorFormat(["dense", "compressed"]) # list of aliases -> same
    True
    >>> str(csr)
    'd,s'
    >>> csr.get_order()
    2
    >>> csr.get_level_types() == [LevelType.DENSE, LevelType.COMPRESSED]
    True
    >>> csr.is_dense()
    False

    See Also
    --------
    LevelType : The four per-mode storage disciplines.
    LevelFormat : One mode's format.
    """

    _level_formats: Tuple[LevelFormat, ...]

    # Fill value default to 0.0
    # TODO: extend to support other fill values
    _fill_value: ClassVar[Optional[float]] = 0.0

    def __init__(
        self,
        level_formats: Optional[
            Union[
                LevelFormat,
                LevelType,
                Sequence[Union[LevelFormat, LevelType, str]],
                str,
            ]
        ] = None,
    ):
        try:
            normalized = _normalize_level_formats(level_formats)
        except (TensorFormatError, TensorTypeError):
            raise
        except Exception as error:
            # A genuine user-defined Sequence has to be consumed to obtain
            # its levels.  Its iteration/indexing hooks are outside the
            # callback-free *recognition* contract, but their exceptions must
            # not escape this public constructor as arbitrary builtins.
            raise TensorFormatError("tensor format sequence is malformed") from error
        object.__setattr__(self, "_level_formats", normalized)

    def get_level_formats(self) -> Tuple[LevelFormat, ...]:
        """Return immutable per-mode :class:`LevelFormat` values in storage order.

        Returns
        -------
        tuple of LevelFormat
            One entry per mode.
        """
        return self._level_formats

    def get_level_types(self) -> List[LevelType]:
        """Return the per-mode :class:`LevelType` list, in storage order.

        Returns
        -------
        list of LevelType
            The bare level type of each mode, e.g. ``[LevelType.DENSE,
            LevelType.COMPRESSED]`` for CSR.
        """
        return [level_format.get_level_type() for level_format in self._level_formats]

    def get_order(self) -> int:
        """Return the tensor's order (rank) -- the number of modes.

        Returns
        -------
        int
            ``len(get_level_formats())``.
        """
        return len(self.get_level_formats())

    def is_dense(self) -> bool:
        """Return ``True`` iff every mode is :attr:`LevelType.DENSE`.

        Used throughout the op layer to pick fast all-dense paths.

        Returns
        -------
        bool
            ``True`` only when all modes are ``DENSE`` (an empty format is
            vacuously dense).
        """
        return all(
            [
                level_format.get_level_type() == LevelType.DENSE
                for level_format in self._level_formats
            ]
        )

    def __str__(self):
        # return "TensorFormat({})".format(self._level_formats)
        # return str(self._level_formats)
        return ",".join([str(level_format) for level_format in self._level_formats])

    def __repr__(self):
        return f"TensorFormat({str(self)!r})"

    def to_dict(self) -> Mapping[str, Any]:
        """Return a JSON-compatible canonical representation."""
        return {
            "levels": [level.to_dict() for level in self._level_formats],
            "fill_value": self._fill_value,
        }

    def serialize(self) -> str:
        """Serialize deterministically for metadata and compiler cache keys."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorFormat":
        """Reconstruct a format from :meth:`to_dict`."""
        # Recognition is by real MRO, never by an ABC check: see
        # ``_derives_from``.  Reading a value once it has been recognized as
        # a genuine mapping/sequence is ordinary consumption and can invoke
        # its protocol hooks.  Translate any exception from that consumption
        # into this public boundary's domain error.
        try:
            if not _derives_from(type(data), *_MAPPING_BASES) or "levels" not in data:
                raise TensorFormatError(
                    "serialized tensor format must contain 'levels'"
                )
            fill_value = data.get("fill_value", 0.0)
            if fill_value != 0.0:
                raise TensorFormatError("only a zero tensor fill value is supported")
            levels = data["levels"]
            levels_type = type(levels)
            if not _derives_from(levels_type, *_SEQUENCE_BASES) or _derives_from(
                levels_type, str, bytes, bytearray
            ):
                raise TensorFormatError(
                    "serialized tensor format levels must be a list"
                )
            parsed = []
            for level in levels:
                if (
                    not _derives_from(type(level), *_MAPPING_BASES)
                    or "type" not in level
                ):
                    raise TensorFormatError(
                        "each serialized tensor level must contain a 'type'"
                    )
                parsed.append(LevelFormat(level["type"], level.get("bit_width")))
            return cls(parsed)
        except (TensorFormatError, TensorTypeError):
            raise
        except Exception as error:
            raise TensorFormatError("serialized tensor format is malformed") from error


FormatInput = Union[
    TensorFormat,
    LevelFormat,
    LevelType,
    Sequence[Union[LevelFormat, LevelType, str]],
    str,
]


def _tokenize_format_string(value: str) -> Tuple[str, ...]:
    value_type = type(value)
    if value_type is str:
        source = value
        text = value.strip().lower()
    elif issubclass(value_type, str):
        source = str.__str__(value)
        text = str.lower(str.strip(source))
    else:
        raise TensorTypeError("tensor format must be a string or sequence of levels")
    if not text:
        return tuple()
    if text in _STR_TO_LEVEL_TYPE:
        return (text,)
    if "," in text:
        tokens = tuple(token.strip() for token in text.split(","))
        if any(not token for token in tokens):
            raise TensorFormatError(f"invalid comma-separated tensor format {source!r}")
        return tokens
    return tuple(text)


def _normalize_level_formats(
    value: Optional[
        Union[
            LevelFormat,
            LevelType,
            Sequence[Union[LevelFormat, LevelType, str]],
            str,
        ]
    ],
) -> Tuple[LevelFormat, ...]:
    if value is None:
        return tuple()
    value_type = type(value)
    normalized_value: Sequence[Union[LevelFormat, LevelType, str]]
    if value_type is LevelFormat:
        return (cast(LevelFormat, value),)
    if value_type is LevelType:
        return (LevelFormat(cast(LevelType, value)),)
    if value_type is str:
        normalized_value = _tokenize_format_string(cast(str, value))
    elif value_type is list or value_type is tuple:
        normalized_value = cast(Sequence[Union[LevelFormat, LevelType, str]], value)
    elif issubclass(value_type, LevelFormat):
        return (cast(LevelFormat, value),)
    elif issubclass(value_type, str):
        normalized_value = _tokenize_format_string(cast(str, value))
    else:
        if _derives_from(value_type, bytes, bytearray) or not _derives_from(
            value_type, *_SEQUENCE_BASES
        ):
            raise TensorTypeError(
                "tensor format must be a string, level, or sequence of levels"
            )
        normalized_value = cast(Sequence[Union[LevelFormat, LevelType, str]], value)
    result: List[LevelFormat] = []
    for level in normalized_value:
        level_type = type(level)
        if level_type is LevelFormat:
            result.append(cast(LevelFormat, level))
        elif level_type is LevelType:
            result.append(LevelFormat(cast(LevelType, level)))
        elif level_type is str:
            result.append(LevelFormat(cast(str, level)))
        elif issubclass(level_type, LevelFormat):
            result.append(cast(LevelFormat, level))
        elif issubclass(level_type, str):
            result.append(LevelFormat(cast(str, level)))
        else:
            raise TensorTypeError(
                "tensor format levels must be LevelFormat, LevelType, or string; "
                f"got {_actual_type_name(level)}"
            )
    return tuple(result)


def parse_format(fmt: FormatInput) -> TensorFormat:
    """Canonical parser used by every public format-taking API."""

    # This is the overwhelmingly common internal call shape.  Keep it as the
    # same single exact-type comparison as the historical implementation;
    # subclass/foreign handling below is deliberately more defensive, but
    # should not tax every ordinary dispatch by running an MRO walk.
    if type(fmt) is TensorFormat:
        return fmt
    try:
        if issubclass(type(fmt), TensorFormat):
            return cast(TensorFormat, fmt)
        return TensorFormat(cast(Any, fmt))
    except (TensorFormatError, TensorTypeError):
        raise
    except Exception as error:
        raise TensorFormatError("tensor format is malformed") from error


def audit_format_state(tensor_format: object) -> Optional[List["LevelFormat"]]:
    """Structurally audit one format's stored state; rebuild its levels.

    ``TensorFormat`` and ``LevelFormat`` are frozen value objects, but
    ``object.__setattr__`` still forges or mutates their stored state, and a
    caller who keeps a reference to a format an object retained can change
    that object's declared layout after the fact.  This inspects the exact
    stored fields without invoking any overridable accessor -- key types are
    proven to be exact ``str`` before any comparison or hashing, so a hostile
    ``str`` subclass cannot run overloaded equality here -- and returns a
    freshly built level list.

    Returns ``None`` when the argument is not a ``TensorFormat`` or when its
    stored state is not exactly the expected base-value shape.  A subclass
    with that exact state is safe to canonicalize as a fresh base value; no
    subclass methods are invoked while inspecting it.
    """

    if not issubclass(type(tensor_format), TensorFormat):
        return None
    # Read the base class's actual instance dictionary through its descriptor.
    # ``object.__getattribute__(value, "__dict__")`` still honors a subclass's
    # overriding ``__dict__`` data descriptor, which could run caller code or
    # conceal extra state at this ownership boundary.
    state = TensorFormat.__dict__["__dict__"].__get__(tensor_format, TensorFormat)
    state_keys = tuple(state) if type(state) is dict else ()
    if (
        type(state) is not dict
        or len(state_keys) != 1
        or type(state_keys[0]) is not str
        or state_keys[0] != "_level_formats"
    ):
        return None
    levels = state["_level_formats"]
    if type(levels) is not tuple:
        return None

    owned_levels: List[LevelFormat] = []
    for level_format in levels:
        if not issubclass(type(level_format), LevelFormat):
            return None
        level_state = LevelFormat.__dict__["__dict__"].__get__(
            level_format, LevelFormat
        )
        level_keys = tuple(level_state) if type(level_state) is dict else ()
        if (
            type(level_state) is not dict
            or len(level_keys) != 2
            or any(type(key) is not str for key in level_keys)
            or set(level_keys) != {"_mode", "_bit_width"}
        ):
            return None
        mode = level_state["_mode"]
        bit_width = level_state["_bit_width"]
        if type(mode) is not LevelType:
            return None
        if bit_width is not None:
            bit_width_type = type(bit_width)
            if bit_width_type is bool or not issubclass(bit_width_type, int):
                return None
            # ``LevelFormat.__init__`` accepts any non-bool ``int`` subclass,
            # so an ``IntEnum`` width is a legally constructed format and must
            # survive this boundary rather than becoming a construction error.
            # Canonicalize it exactly the way every other retained scalar is
            # canonicalized: the base descriptor bypasses an overridden
            # ``__int__`` and yields an exact ``int``, so no caller arithmetic
            # is retained or executed past this point.
            bit_width = int.__int__(bit_width)
            if bit_width <= 0:
                return None
        owned_levels.append(LevelFormat(mode, bit_width=bit_width))
    return owned_levels


def owned_format(tensor_format: TensorFormat) -> TensorFormat:
    """Detach one format on the way *in* to a retaining object.

    This is the construction-side half of the format-ownership boundary: an
    object that will retain a format rebuilds it, so a caller who keeps the
    original cannot mutate the retained metadata afterwards.  It also breaks
    the sharing that let one process-global memoized format become several
    unrelated tensors' declared layout.

    Structurally canonical subclasses remain accepted, but are normalized to
    the base value type.  Anything malformed fails closed: retaining it would
    preserve mutable caller-owned state at the ownership boundary.

    The read side is deliberately untouched -- rebuilding on every ``.format``
    access measured as a double-digit regression on the live dispatch path.
    """

    owned_levels = audit_format_state(tensor_format)
    if owned_levels is None:
        raise TensorFormatError("tensor format has malformed stored state")
    return TensorFormat(owned_levels)
