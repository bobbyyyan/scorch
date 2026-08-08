from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    List,
    Mapping,
    Optional,
    Sequence,
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


def _parse_level_type(s: str) -> LevelType:
    """Convert one canonicalized string alias to a :class:`LevelType`."""
    if not isinstance(s, str):
        raise TensorTypeError(
            f"level format must be a string or LevelType, got {type(s).__name__}"
        )
    key = s.strip().lower()
    try:
        return _STR_TO_LEVEL_TYPE[key]
    except KeyError as error:
        aliases = ", ".join(sorted(_STR_TO_LEVEL_TYPE))
        raise TensorFormatError(
            f"invalid level format {s!r}; expected one of: {aliases}"
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
        parsed_mode = _parse_level_type(mode) if isinstance(mode, str) else mode
        if not isinstance(parsed_mode, LevelType):
            raise TensorTypeError(
                "level format mode must be a string alias or LevelType, "
                f"got {type(mode).__name__}"
            )
        if bit_width is not None:
            if isinstance(bit_width, bool) or not isinstance(bit_width, int):
                raise TensorTypeError("level format bit_width must be an integer")
            if bit_width <= 0:
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
        object.__setattr__(
            self, "_level_formats", _normalize_level_formats(level_formats)
        )

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
        if not isinstance(data, Mapping) or "levels" not in data:
            raise TensorFormatError("serialized tensor format must contain 'levels'")
        fill_value = data.get("fill_value", 0.0)
        if fill_value != 0.0:
            raise TensorFormatError("only a zero tensor fill value is supported")
        levels = data["levels"]
        if not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
            raise TensorFormatError("serialized tensor format levels must be a list")
        parsed = []
        for level in levels:
            if not isinstance(level, Mapping) or "type" not in level:
                raise TensorFormatError(
                    "each serialized tensor level must contain a 'type'"
                )
            parsed.append(LevelFormat(level["type"], level.get("bit_width")))
        return cls(parsed)


FormatInput = Union[
    TensorFormat,
    LevelFormat,
    LevelType,
    Sequence[Union[LevelFormat, LevelType, str]],
    str,
]


def _tokenize_format_string(value: str) -> Tuple[str, ...]:
    text = value.strip().lower()
    if not text:
        return tuple()
    if text in _STR_TO_LEVEL_TYPE:
        return (text,)
    if "," in text:
        tokens = tuple(token.strip() for token in text.split(","))
        if any(not token for token in tokens):
            raise TensorFormatError(f"invalid comma-separated tensor format {value!r}")
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
    if isinstance(value, LevelFormat):
        return (value,)
    if isinstance(value, LevelType):
        return (LevelFormat(value),)
    if isinstance(value, str):
        value = _tokenize_format_string(value)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TensorTypeError(
            "tensor format must be a string, level, or sequence of levels"
        )
    result = []
    for level in value:
        if isinstance(level, LevelFormat):
            result.append(level)
        elif isinstance(level, (LevelType, str)):
            result.append(LevelFormat(level))
        else:
            raise TensorTypeError(
                "tensor format levels must be LevelFormat, LevelType, or string; "
                f"got {type(level).__name__}"
            )
    return tuple(result)


def parse_format(fmt: FormatInput) -> TensorFormat:
    """Canonical parser used by every public format-taking API."""
    return fmt if isinstance(fmt, TensorFormat) else TensorFormat(fmt)


MAX_FORMAT_BIT_WIDTH = (1 << 63) - 1


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

    Returns ``None`` when the argument is not an exact ``TensorFormat`` or
    when its stored state is not exactly the expected shape.  Callers that owe
    the strict public contract raise on ``None``; construction boundaries
    treat it as "nothing safe to rebuild" and keep the value they were given,
    which preserves today's acceptance of ``TensorFormat`` subclasses.
    """

    if type(tensor_format) is not TensorFormat:
        return None
    try:
        state = object.__getattribute__(tensor_format, "__dict__")
    except AttributeError:
        return None
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
        if type(level_format) is not LevelFormat:
            return None
        try:
            level_state = object.__getattribute__(level_format, "__dict__")
        except AttributeError:
            return None
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
        if bit_width is not None and (
            type(bit_width) is not int
            or bit_width <= 0
            or bit_width > MAX_FORMAT_BIT_WIDTH
        ):
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

    Fail-open by design: anything this cannot prove structurally exact is
    returned unchanged, so no previously accepted argument becomes an error.
    The read side is deliberately untouched -- rebuilding on every ``.format``
    access measured as a double-digit regression on the live dispatch path.
    """

    owned_levels = audit_format_state(tensor_format)
    if owned_levels is None:
        return tensor_format
    return TensorFormat(owned_levels)
