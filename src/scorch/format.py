from __future__ import annotations

from array import array as _Array
from collections import deque as _Deque
from collections.abc import Mapping as _AbcMapping, Sequence
from dataclasses import FrozenInstanceError, dataclass
from enum import Enum
import json

# Mapping/Sequence come from collections.abc, not typing: these are used for
# runtime isinstance checks on a per-call path, and typing's generic aliases
# route isinstance through __subclasscheck__ at 153 ns against 73 ns for the
# abc. `from __future__ import annotations` above means the annotations that
# also use these names are never evaluated at runtime.
from collections.abc import Mapping
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch  # noqa: F401  -- loads libtorch, which scorch_ops links against
import scorch_ops as _native  # type: ignore[import-not-found]

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


# The two functions below read only a letter string, one letter per storage
# level ("d" dense, "s" compressed).  Both compilers use them: the release
# einsum's operand relayout (scorch.ops._einsum_operand_relayout) and the new
# compiler's operand conversion module, which imports them from here so that a
# release module never has to import that compiler's package.  (The new
# compiler's package is not named here: the neutrality suite searches release
# modules for its name.)


def stores_only_its_entries(format_letters: str) -> bool:
    """Whether a tensor in this layout stores no cell beyond the entries put in it.

    A dense level stores every cell under each stored parent.  Above every
    compressed level that adds parents with nothing under them and no entry;
    below a compressed level it adds entries.
    """

    return "sd" not in format_letters


def conversion_format(format_letters: str) -> str:
    """The layout of an operand of this layout after a conversion.

    Its own layout on the new level order when that stores only the operand's
    entries (CSR becomes CSC), and every level compressed otherwise.
    """

    if stores_only_its_entries(format_letters):
        return format_letters
    return "s" * len(format_letters)


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


# --------------------------------------------------------------------------- #
# Read-only stored state
#
# LevelFormat and TensorFormat here, and TensorLayout and TensorMetadata in layout.py, keep
# their fields in a base made by ``scorch_ops._sealed_type`` (csrc/sealed_values.cpp, which
# says why this has to be C). Each field is read-only -- ``object.__setattr__`` and
# ``object.__delattr__`` refuse it -- and an instance has no ``__dict__`` to take any other
# attribute. The constructor validates its arguments and then sets every field at once with
# ``_seal``, which refuses an object whose fields are already set. So a value object cannot
# change once it is built, and any number of tensors and caches may share one.
#
# The value classes stay dataclasses (init=False, with the base's descriptors as the
# fields), so equality, hashing, repr and ``dataclasses.fields``/``replace`` are generated
# exactly as before. Pickling and copying rebuild through the public constructor.
# --------------------------------------------------------------------------- #
_seal = _native._seal
_sealed_new = _native._sealed_new
_sealed_type = _native._sealed_type


def _field_unset(value: object, descriptor: Any) -> bool:
    """Whether the field ``descriptor`` reads is unset: an object made by ``__new__``
    without its constructor, which only unpickling does on purpose."""
    try:
        descriptor.__get__(value)
    except AttributeError:
        return True
    return False


def _reduce_value(value: object, args: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """``__reduce__`` of a value type: rebuild through the public constructor, which
    validates what the pickle holds again. A subclass's own instance state goes along.
    """
    state = getattr(value, "__dict__", None)
    if state:
        return (type(value), args, dict(state))
    return (type(value), args)


def _same_derived(stored: Any, derived: Any) -> bool:
    """Whether a derived field an older pickle stored equals the one the constructor
    derived. The derived fields are tuples of LevelType members, bools and strs."""
    if type(stored) is not type(derived):
        return False
    if type(derived) is tuple:
        return len(stored) == len(derived) and all(
            a is b for a, b in zip(stored, derived)
        )
    return bool(stored == derived)


def _restore_value(
    value: object,
    state: Any,
    init: Any,
    descriptors: Tuple[Tuple[str, Any], ...],
    error: type,
    derived: Tuple[Tuple[str, Any], ...] = (),
) -> None:
    """``__setstate__`` of a value type. ``state`` is a subclass's instance state, or, in a
    pickle written before the fields were read-only, the fields themselves: those go
    through the base class's constructor (``descriptors`` names its arguments). Trees
    before that also memoized derived fields in the instance state; ``derived`` names
    them, and each one the pickle holds must equal what the constructor derived, so a
    pickle of a value whose memo had gone stale fails closed. The fields of an existing
    value are never changed, so a field in the state of an object already built is
    refused."""
    if state is None:
        return
    if type(state) is not dict:
        raise error("pickled value has malformed state")
    names = tuple(name for name, _ in descriptors)
    derived_names = tuple(name for name, _ in derived)
    if _field_unset(value, descriptors[0][1]):
        try:
            arguments = [state[name] for name in names]
        except KeyError as missing:
            raise error(f"pickled value is missing {missing.args[0]!r}") from None
        init(value, *arguments)
        for name, descriptor in derived:
            if name in state and not _same_derived(
                state[name], descriptor.__get__(value)
            ):
                raise error(f"pickled value's {name!r} disagrees with its other fields")
        state = {
            key: item
            for key, item in state.items()
            if key not in names and key not in derived_names
        }
    if not state:
        return
    instance_state = getattr(value, "__dict__", None)
    if type(instance_state) is not dict or any(
        name in state for name in names + derived_names
    ):
        raise error("pickled value holds state its type does not")
    instance_state.update(state)


if TYPE_CHECKING:
    # The bases are made at import time, so a type checker sees stand-ins: these name
    # the fields the base holds beyond the declared ones, which must not be annotated
    # on the dataclass itself (they would become dataclass fields).
    _LevelFormatFields = object

    class _TensorFormatFields:
        _level_types: Tuple[LevelType, ...]
        _is_dense: bool
        _str: str
        _canonical: bool

else:
    _LevelFormatFields = _sealed_type("LevelFormatFields", ("_mode", "_bit_width"))
    _TensorFormatFields = _sealed_type(
        "TensorFormatFields",
        ("_level_formats", "_level_types", "_is_dense", "_str", "_canonical"),
    )
_LEVEL_MODE = _LevelFormatFields.__dict__["_mode"]
_LEVEL_BIT_WIDTH = _LevelFormatFields.__dict__["_bit_width"]


@dataclass(frozen=True, init=False)
class LevelFormat(_LevelFormatFields):
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

    # Read-only fields held by LevelFormatFields: see "Read-only stored state" above.
    __slots__ = ()

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
        if not _seal(self, parsed_mode, bit_width):
            raise FrozenInstanceError("cannot assign to field '_mode'")

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

    def __reduce__(self):
        return _reduce_value(self, (self._mode, self._bit_width))

    def __setstate__(self, state):
        _restore_value(
            self,
            state,
            LevelFormat.__init__,
            (("_mode", _LEVEL_MODE), ("_bit_width", _LEVEL_BIT_WIDTH)),
            TensorFormatError,
        )


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


_FORMAT_LEVELS = _TensorFormatFields.__dict__["_level_formats"]
# The derived fields that trees before the read-only fields memoized in a format's instance
# state on first use, and so wrote into its pickles (864745d did; 154320f kept them out).
_FORMAT_MEMOS = tuple(
    (name, _TensorFormatFields.__dict__[name])
    for name in ("_level_types", "_is_dense", "_str")
)

# str(LevelFormat(t)) for each level type, so a format derives its string without a call
# per level.
_LEVEL_TEXT = {level_type: str(level_type.value) for level_type in LevelType}


def _format_fields(levels: Tuple[LevelFormat, ...]) -> Tuple[Any, ...]:
    """A format's stored state: its levels, and what the op layer asks of them on every
    call (the level types, whether every mode is dense, the string form), derived once
    here because nothing can change them afterwards. The last field says whether the
    format is already what ``owned_format`` would make of it: exact LevelFormat levels
    with exact-int (or no) bit widths."""
    canonical = True
    level_types = []
    texts = []
    for level in levels:
        if type(level) is LevelFormat:
            mode = level._mode
            level_types.append(mode)
            texts.append(_LEVEL_TEXT[mode])
            bit_width = level._bit_width
            if bit_width is not None and type(bit_width) is not int:
                canonical = False
        else:
            # A subclass answers for itself, as it did when these were derived lazily.
            level_types.append(level.get_level_type())
            texts.append(str(level))
            canonical = False
    derived = tuple(level_types)
    return (
        levels,
        derived,
        all(level_type == LevelType.DENSE for level_type in derived),
        ",".join(texts),
        canonical,
    )


# Formats parsed from text, keyed by the exact string: the fields the constructor derives,
# and the instance parse_format hands out. Nothing in a format can change once it is built
# (see "Read-only stored state"), so every format made from one string may hold the same
# levels, and every caller naming the string may hold the same format. Deriving the fields
# at construction made TensorFormat("ds") cost 2.10 us against 1.53 us when they were
# derived lazily (mkt1, job 17813665); a hit here skips the parse and the derivation.
# Bounded like the layout cache (layout.py): a program names a handful of format strings,
# and one that names more is served the ordinary way.
_TEXT_FIELDS: Dict[str, Tuple[Any, ...]] = {}
_TEXT_FORMATS: Dict[str, "TensorFormat"] = {}
_TEXT_CACHE_MAX = 1024


@dataclass(frozen=True, init=False)
class TensorFormat(_TensorFormatFields):
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

    # Read-only fields held by TensorFormatFields: see "Read-only stored state" above. The
    # only declared field, and so the only one equality and hashing see, is _level_formats;
    # the others are derived from it at construction (_format_fields).
    __slots__ = ()

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
        # Only an exact str is looked up: a str subclass could answer __hash__ and __eq__.
        text = level_formats if type(level_formats) is str else None
        fields = None if text is None else _TEXT_FIELDS.get(text)
        if fields is None:
            try:
                normalized = _normalize_level_formats(level_formats)
            except (TensorFormatError, TensorTypeError):
                raise
            except Exception as error:
                # A genuine user-defined Sequence has to be consumed to obtain
                # its levels.  Its iteration/indexing hooks are outside the
                # callback-free *recognition* contract, but their exceptions must
                # not escape this public constructor as arbitrary builtins.
                raise TensorFormatError(
                    "tensor format sequence is malformed"
                ) from error
            try:
                fields = _format_fields(normalized)
            except Exception as error:
                # Only a LevelFormat subclass's own accessors run here.
                raise TensorFormatError("tensor format levels are malformed") from error
            if text is not None and len(_TEXT_FIELDS) < _TEXT_CACHE_MAX:
                _TEXT_FIELDS[text] = fields
        if not _seal(self, *fields):
            raise FrozenInstanceError("cannot assign to field '_level_formats'")

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
        return list(self.level_types)

    @property
    def level_types(self) -> Tuple[LevelType, ...]:
        """The per-mode level types as a tuple, derived at construction.

        For the construction paths, which ask for them several times per tensor;
        ``get_level_types`` returns a fresh list for callers that may modify it.
        """
        return self._level_types

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
        # Derived at construction: the op layer asks this three times per matmul. Not a
        # declared field, so it takes no part in __eq__ or __hash__.
        return self._is_dense

    def __str__(self):
        # Derived at construction for the same reason as is_dense: this is on a per-call
        # path (kernel resolution keys on it). A str is also cheaper to hash than the
        # underlying tuple of LevelFormat, since CPython caches string hashes and hashing
        # the tuple recurses through every level.
        return self._str

    def __reduce__(self):
        # A format of exact levels without bit widths is pickled as its text, which parses
        # back to it: loading one then costs a lookup in _TEXT_FIELDS rather than a
        # LevelType, a LevelFormat and a validation per level, and the pickle is smaller.
        if (
            type(self) is TensorFormat
            and self._canonical
            and all(level._bit_width is None for level in self._level_formats)
        ):
            return (TensorFormat, (self._str,))
        return _reduce_value(self, (self._level_formats,))

    def __setstate__(self, state):
        _restore_value(
            self,
            state,
            TensorFormat.__init__,
            (("_level_formats", _FORMAT_LEVELS),),
            TensorFormatError,
            _FORMAT_MEMOS,
        )

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


# parse_format shares one format per string (_TEXT_FORMATS above). The performance line
# had such a memo, because "`matmul` alone parses a format three times per call, always on
# the same constant strings". On this tree a warm `matmul` constructs no formats either
# way (the plan path and the compile-options policy hold parsed formats upstream); the
# memo serves tensor construction, where `from_torch(...).to_sparse("ds")` parses about
# five times. The refactor had removed it because a retained format could be rewritten
# through `object.__setattr__`, and `TensorVar` retains what `parse_format` returns, so
# one corrupted format became every later caller's (1300 test failures when the two sides
# were first combined). A format's fields are read-only now, so that cannot happen.


def parse_format(fmt: FormatInput) -> TensorFormat:
    """Canonical parser used by every public format-taking API."""

    # This is the overwhelmingly common internal call shape.  Keep it as the
    # same single exact-type comparison as the historical implementation;
    # subclass/foreign handling below is deliberately more defensive, but
    # should not tax every ordinary dispatch by running an MRO walk.
    if type(fmt) is TensorFormat:
        return fmt
    text = fmt if type(fmt) is str else None
    if text is not None:
        shared = _TEXT_FORMATS.get(text)
        if shared is not None:
            return shared
    try:
        if issubclass(type(fmt), TensorFormat):
            return cast(TensorFormat, fmt)
        parsed = TensorFormat(cast(Any, fmt))
    except (TensorFormatError, TensorTypeError):
        raise
    except Exception as error:
        raise TensorFormatError("tensor format is malformed") from error
    if text is not None and len(_TEXT_FORMATS) < _TEXT_CACHE_MAX:
        _TEXT_FORMATS[text] = parsed
    return parsed


def audit_format_state(tensor_format: object) -> Optional[List["LevelFormat"]]:
    """The levels of a format, each an exact ``LevelFormat`` with an exact-int (or no)
    bit width; ``None`` when the argument is not a ``TensorFormat``, or is one that was
    never constructed (made by ``__new__`` alone, so its fields are unset).

    A format's fields are read-only and set only by its constructor, after validation
    (see "Read-only stored state"), so there is no stored state to audit any more: this
    only reads it. It reads through the base's own descriptors, so a subclass's accessors
    are not consulted and none of its code runs. A level that is a ``LevelFormat``
    subclass, or whose bit width is an ``int`` subclass (``LevelFormat`` accepts an
    ``IntEnum`` width), is rebuilt as an exact value; the base ``int`` descriptor yields an
    exact ``int`` without running an overridden ``__int__``.
    """

    if not issubclass(type(tensor_format), TensorFormat):
        return None
    try:
        levels = _FORMAT_LEVELS.__get__(tensor_format)
    except AttributeError:
        return None
    owned_levels: List[LevelFormat] = []
    for level_format in levels:
        try:
            mode = _LEVEL_MODE.__get__(level_format)
            bit_width = _LEVEL_BIT_WIDTH.__get__(level_format)
        except AttributeError:
            return None
        if type(level_format) is LevelFormat and (
            bit_width is None or type(bit_width) is int
        ):
            owned_levels.append(level_format)
        else:
            owned_levels.append(
                LevelFormat(
                    mode,
                    bit_width=None if bit_width is None else int.__int__(bit_width),
                )
            )
    return owned_levels


def owned_format(tensor_format: TensorFormat) -> TensorFormat:
    """The format an object retains when it is handed ``tensor_format``.

    A format cannot change once built (see "Read-only stored state"), so an exact
    ``TensorFormat`` of exact levels is retained as it is, and any number of objects may
    share it. A subclass, or a format with subclass levels or an ``int``-subclass bit
    width, is normalized to an exact value of the base type, so that no retained format
    carries caller-defined behaviour. A format that was never constructed fails closed.
    """

    if type(tensor_format) is TensorFormat:
        try:
            if tensor_format._canonical:
                return tensor_format
        except AttributeError:
            raise TensorFormatError(
                "tensor format has malformed stored state"
            ) from None
    owned_levels = audit_format_state(tensor_format)
    if owned_levels is None:
        raise TensorFormatError("tensor format has malformed stored state")
    return TensorFormat(owned_levels)
