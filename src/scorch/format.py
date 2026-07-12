from enum import Enum
from typing import Optional, Union, List

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

    ``SINGLETON`` constructs fine in a ``TensorFormat`` but is **reserved /
    not-yet-supported** end-to-end: ``utils.parse_format`` rejects it (so it
    cannot be an op ``output_format``), and the lowering path branches only on
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
    """Convert a string alias to a LevelType, or raise ValueError."""
    try:
        return _STR_TO_LEVEL_TYPE[s]
    except KeyError:
        raise ValueError(f"Invalid format string: {s}")


class LevelFormat(object):
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
        if isinstance(mode, str):
            mode = _parse_level_type(mode)
        assert isinstance(mode, LevelType)
        self._mode = mode
        self._bit_width = bit_width

    def get_level_type(self) -> LevelType:
        return self._mode

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


class TensorFormat(object):
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
        - a bare ``str`` -> **split one character per mode** via ``list(...)``
          (e.g. ``"ds"`` -> ``["d", "s"]``).

    Notes
    -----
    **Bare strings are split per character.** A bare ``str`` is only for the
    single-letter alphabet ``d`` / ``s`` / ``c`` / ``o``. Multi-character
    aliases in a bare string are mis-parsed: ``TensorFormat("singleton")``
    splits into ``["s", "i", "n", ...]`` and raises ``ValueError`` on ``"i"``.
    To use a long-form alias (including ``singleton``) pass a **list**, e.g.
    ``TensorFormat(["coordinate", "singleton"])``.

    ``s`` and ``c`` are synonyms for ``COMPRESSED``. ``singleton`` constructs
    fine here but is not lowered by codegen (reserved / not-yet-supported).

    ``__str__`` is **comma-joined** (``str(TensorFormat("ds")) == "d,s"``),
    which differs from the char-per-mode input string ``"ds"`` -- the
    round-trip is not identical.

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

    _level_formats: List[LevelFormat]

    # Fill value default to 0.0
    # TODO: extend to support other fill values
    _fill_value: Optional[float] = 0.0

    def __init__(
        self,
        level_formats: Optional[
            Union[LevelFormat, List[LevelFormat], List[str], str]
        ] = None,
    ):
        if level_formats is None:
            self._level_formats = []
        elif isinstance(level_formats, LevelFormat):
            self._level_formats = [level_formats]
        else:
            if isinstance(level_formats, str):
                level_formats = list(level_formats)
            self._level_formats = [
                lf if isinstance(lf, LevelFormat) else LevelFormat(mode=_parse_level_type(lf))
                for lf in level_formats
            ]

    def get_level_formats(self) -> List[LevelFormat]:
        """Return the per-mode :class:`LevelFormat` list, in storage order.

        Returns
        -------
        list of LevelFormat
            One entry per mode.
        """
        assert self._level_formats is not None, "level_formats is None"
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
        return str(self)

    def __eq__(self, other):
        return self._level_formats == other._level_formats

    def __ne__(self, other):
        return not self.__eq__(other)
