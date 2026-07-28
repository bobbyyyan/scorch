"""Format-neutral level storage for the production LoopIR oracle.

Promoted from the Phase-3.5 spike (its level-storage and CSR
container modules) for the Phase-5 sparse vertical slice.  The
oracle's execution core consumes only this small interface — ``segment``,
``coordinate_at``, and ``leaf_value`` on a validated
:class:`LevelTensorStorage` — so its traversal is defined over generic
physical levels rather than any one container.  CSR is exactly one adapter
(:class:`CsrMatrix` plus :func:`from_csr` on the input side,
:class:`CsrOutputBuilder` on the assembly side); DCSR, CSC, CSF-like, and
any other DENSE/COMPRESSED level composition with any physical-to-logical
mode permutation bind through the same storage class.

Canonical means: exact int/float scalars; per-level consistency between the
parent position count and the child level's storage (``seg_offsets`` of
length ``parents + 1``, starting at zero, nondecreasing; per-segment
coordinates strictly increasing and inside the stored dimension's extent);
dense extents equal to the stored dimension's extent; and a value stream
owned by the leaf level with exactly one scalar per leaf position.
Everything unexpected raises :class:`LevelStorageError` rather than being
coerced.  This module is Torch-free and is loaded only by the oracle and
dedicated tests, never by production compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, cast

from .nodes import LevelKind

MAX_LEVEL_STORAGE_RANK = 64


class LevelStorageError(Exception):
    """A storage construction or access violated the canonical contract."""


def _stored_field(value: object, name: str, what: str) -> object:
    """Read one exact stored field without leaking forged-state exceptions."""

    state = getattr(value, "__dict__", None)
    if type(state) is not dict or name not in state:
        raise LevelStorageError(f"{what} is missing stored field {name!r}")
    return state[name]


def _expect_exact_int(value: object, what: str) -> int:
    if type(value) is not int:
        raise LevelStorageError(
            f"{what} must be an exact int, got {type(value).__name__}"
        )
    return value


class CsrFormatError(Exception):
    """A container construction violated the canonical CSR contract."""


@dataclass(frozen=True)
class CsrMatrix:
    """One canonical, immutable rank-2 dense-by-compressed matrix.

    Explicit zero values are permitted (a UNION add can cancel to an exact
    zero at an overlapping coordinate); ``from_dense`` never manufactures
    them.
    """

    n_rows: int
    n_cols: int
    indptr: Tuple[int, ...]
    indices: Tuple[int, ...]
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        def expect_int(value: object, what: str) -> int:
            if type(value) is not int:
                raise CsrFormatError(
                    f"{what} must be an exact int, got {type(value).__name__}"
                )
            return value

        n_rows = expect_int(self.n_rows, "n_rows")
        n_cols = expect_int(self.n_cols, "n_cols")
        if n_rows < 0 or n_cols < 0:
            raise CsrFormatError("matrix dimensions must be nonnegative")
        for name, stream in (
            ("indptr", self.indptr),
            ("indices", self.indices),
            ("values", self.values),
        ):
            if type(stream) is not tuple:
                raise CsrFormatError(f"{name} must be an owned tuple")
        if len(self.indptr) != n_rows + 1:
            raise CsrFormatError(f"indptr length {len(self.indptr)} for {n_rows} rows")
        previous = 0
        for position, offset in enumerate(self.indptr):
            expect_int(offset, f"indptr[{position}]")
            if position == 0:
                if offset != 0:
                    raise CsrFormatError("indptr must start at zero")
            elif offset < previous:
                raise CsrFormatError("indptr must be nondecreasing")
            previous = offset
        if len(self.indices) != self.indptr[-1]:
            raise CsrFormatError("indptr must terminate at the stored-coordinate count")
        if len(self.values) != len(self.indices):
            raise CsrFormatError("indices and values must have equal length")
        for row in range(n_rows):
            last = -1
            for position in range(self.indptr[row], self.indptr[row + 1]):
                column = expect_int(self.indices[position], f"indices[{position}]")
                if not 0 <= column < n_cols:
                    raise CsrFormatError(
                        f"column {column} outside [0, {n_cols}) in row {row}"
                    )
                if column <= last:
                    raise CsrFormatError(
                        f"row {row} columns must be strictly increasing"
                    )
                last = column
        for position, value in enumerate(self.values):
            if type(value) is not float:
                raise CsrFormatError(f"values[{position}] must be an exact float")

    @classmethod
    def from_dense(cls, rows: Sequence[Sequence[float]]) -> "CsrMatrix":
        """Build a canonical CSR container from a rectangular dense grid."""

        if type(rows) not in (list, tuple):
            raise CsrFormatError("dense input must be a list or tuple of rows")
        n_rows = len(rows)
        n_cols = 0
        for row in rows:
            if type(row) not in (list, tuple):
                raise CsrFormatError("dense rows must be lists or tuples")
            n_cols = max(n_cols, len(row))
        for row in rows:
            if len(row) != n_cols:
                raise CsrFormatError("dense input is ragged")
        indptr: List[int] = [0]
        indices: List[int] = []
        values: List[float] = []
        for row in rows:
            for column, entry in enumerate(row):
                if type(entry) is not float and type(entry) is not int:
                    raise CsrFormatError("dense entries must be numeric")
                try:
                    value = float(entry)
                except (OverflowError, TypeError, ValueError) as error:
                    raise CsrFormatError(
                        "dense entries must be representable numeric values"
                    ) from error
                if value != 0.0:
                    indices.append(column)
                    values.append(value)
            indptr.append(len(indices))
        return cls(
            n_rows=n_rows,
            n_cols=n_cols,
            indptr=tuple(indptr),
            indices=tuple(indices),
            values=tuple(values),
        )

    def to_dense(self) -> List[List[float]]:
        """Materialize the stored matrix as nested lists."""

        dense = [[0.0] * self.n_cols for _ in range(self.n_rows)]
        for row in range(self.n_rows):
            for position in range(self.indptr[row], self.indptr[row + 1]):
                dense[row][self.indices[position]] = self.values[position]
        return dense


@dataclass(frozen=True)
class DenseLevel:
    """One DENSE physical level: positions are ``parent * extent + coord``."""

    extent: int


@dataclass(frozen=True)
class CompressedLevel:
    """One COMPRESSED physical level: segmented positions plus coordinates.

    ``seg_offsets[parent] .. seg_offsets[parent + 1]`` is the half-open
    child-position range selected by one dominating parent position;
    ``coords[position]`` is the stored coordinate at one child position.
    """

    seg_offsets: Tuple[int, ...]
    coords: Tuple[int, ...]


_LEVEL_KINDS = {DenseLevel: LevelKind.DENSE, CompressedLevel: LevelKind.COMPRESSED}


@dataclass(frozen=True)
class LevelTensorStorage:
    """One canonical, immutable bound tensor behind the level interface.

    ``shape`` is the logical shape in mode order; ``modes`` maps each
    physical level to the logical mode it stores; ``levels`` are the
    physical levels in storage order; ``values`` is the scalar stream owned
    by the leaf (last physical) level, one value per leaf position.
    """

    shape: Tuple[int, ...]
    modes: Tuple[int, ...]
    levels: Tuple[object, ...]
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        for field_name in ("shape", "modes", "levels", "values"):
            _stored_field(self, field_name, "LevelTensorStorage")
        if type(self.shape) is not tuple or not self.shape:
            raise LevelStorageError("shape must be a nonempty owned tuple")
        for position, extent in enumerate(self.shape):
            if _expect_exact_int(extent, f"shape[{position}]") < 0:
                raise LevelStorageError("shape extents must be nonnegative")
        rank = len(self.shape)
        if rank > MAX_LEVEL_STORAGE_RANK:
            raise LevelStorageError(
                f"rank {rank} exceeds the level-storage limit "
                f"{MAX_LEVEL_STORAGE_RANK}"
            )
        if type(self.modes) is not tuple or len(self.modes) != rank:
            raise LevelStorageError(f"modes must be an owned tuple of length {rank}")
        for position, mode in enumerate(self.modes):
            _expect_exact_int(mode, f"modes[{position}]")
        if sorted(self.modes) != list(range(rank)):
            raise LevelStorageError(
                f"modes must be a permutation of the logical modes, got {self.modes}"
            )
        if type(self.levels) is not tuple or len(self.levels) != rank:
            raise LevelStorageError(f"levels must be an owned tuple of length {rank}")
        counts: List[int] = []
        parent_count = 1
        for number, level in enumerate(self.levels):
            extent = self.shape[self.modes[number]]
            if type(level) is DenseLevel:
                declared = _expect_exact_int(
                    _stored_field(level, "extent", f"levels[{number}]"),
                    f"levels[{number}].extent",
                )
                if declared != extent:
                    raise LevelStorageError(
                        f"levels[{number}] extent {declared} does not match "
                        f"dimension extent {extent}"
                    )
                parent_count = parent_count * extent
            elif type(level) is CompressedLevel:
                offsets = _stored_field(level, "seg_offsets", f"levels[{number}]")
                if type(offsets) is not tuple:
                    raise LevelStorageError(
                        f"levels[{number}].seg_offsets must be an owned tuple"
                    )
                if len(offsets) != parent_count + 1:
                    raise LevelStorageError(
                        f"levels[{number}] has {len(offsets)} segment offsets "
                        f"for {parent_count} parent positions"
                    )
                previous = 0
                for position, offset in enumerate(offsets):
                    _expect_exact_int(
                        offset, f"levels[{number}].seg_offsets[{position}]"
                    )
                    if position == 0:
                        if offset != 0:
                            raise LevelStorageError(
                                f"levels[{number}].seg_offsets must start at zero"
                            )
                    elif offset < previous:
                        raise LevelStorageError(
                            f"levels[{number}].seg_offsets must be nondecreasing"
                        )
                    previous = offset
                coords = _stored_field(level, "coords", f"levels[{number}]")
                if type(coords) is not tuple:
                    raise LevelStorageError(
                        f"levels[{number}].coords must be an owned tuple"
                    )
                if len(coords) != offsets[-1]:
                    raise LevelStorageError(
                        f"levels[{number}].seg_offsets must terminate at the "
                        "stored-coordinate count"
                    )
                for parent in range(parent_count):
                    last = -1
                    for position in range(offsets[parent], offsets[parent + 1]):
                        coord = _expect_exact_int(
                            coords[position], f"levels[{number}].coords[{position}]"
                        )
                        if not 0 <= coord < extent:
                            raise LevelStorageError(
                                f"levels[{number}] coordinate {coord} outside "
                                f"[0, {extent})"
                            )
                        if coord <= last:
                            raise LevelStorageError(
                                f"levels[{number}] segment {parent} coordinates "
                                "must be strictly increasing"
                            )
                        last = coord
                parent_count = len(coords)
            else:
                raise LevelStorageError(
                    f"levels[{number}] has unsupported storage class "
                    f"{type(level).__name__}"
                )
            counts.append(parent_count)
        if type(self.values) is not tuple:
            raise LevelStorageError("values must be an owned tuple")
        if len(self.values) != parent_count:
            raise LevelStorageError(
                f"{len(self.values)} values for {parent_count} leaf positions"
            )
        for position, value in enumerate(self.values):
            if type(value) is not float:
                raise LevelStorageError(f"values[{position}] must be an exact float")
        object.__setattr__(self, "_position_counts", tuple(counts))

    def validate(self) -> None:
        """Recheck canonical stored state, including deliberately forged state."""

        self.__post_init__()

    def snapshot(self) -> "LevelTensorStorage":
        """Return a validated deep structural snapshot detached from its caller."""

        shape = _stored_field(self, "shape", "LevelTensorStorage")
        modes = _stored_field(self, "modes", "LevelTensorStorage")
        levels = _stored_field(self, "levels", "LevelTensorStorage")
        values = _stored_field(self, "values", "LevelTensorStorage")
        for name, owned in (
            ("shape", shape),
            ("modes", modes),
            ("levels", levels),
            ("values", values),
        ):
            if type(owned) is not tuple:
                raise LevelStorageError(f"{name} must be an owned tuple")
        copied_levels: List[object] = []
        for number, level in enumerate(cast(Tuple[object, ...], levels)):
            if type(level) is DenseLevel:
                copied_levels.append(
                    DenseLevel(
                        cast(
                            int,
                            _stored_field(level, "extent", f"levels[{number}]"),
                        )
                    )
                )
            else:
                if type(level) is not CompressedLevel:
                    raise LevelStorageError(
                        f"levels[{number}] has unsupported storage class "
                        f"{type(level).__name__}"
                    )
                offsets = _stored_field(level, "seg_offsets", f"levels[{number}]")
                coords = _stored_field(level, "coords", f"levels[{number}]")
                if type(offsets) is not tuple:
                    raise LevelStorageError(
                        f"levels[{number}].seg_offsets must be an owned tuple"
                    )
                if type(coords) is not tuple:
                    raise LevelStorageError(
                        f"levels[{number}].coords must be an owned tuple"
                    )
                copied_levels.append(
                    CompressedLevel(
                        cast(Tuple[int, ...], offsets),
                        cast(Tuple[int, ...], coords),
                    )
                )
        return LevelTensorStorage(
            shape=cast(Tuple[int, ...], shape),
            modes=cast(Tuple[int, ...], modes),
            levels=tuple(copied_levels),
            values=cast(Tuple[float, ...], values),
        )

    @property
    def kinds(self) -> Tuple[LevelKind, ...]:
        """Per-physical-level kinds, for binding-time declaration checks."""

        levels = _stored_field(self, "levels", "LevelTensorStorage")
        if type(levels) is not tuple:
            raise LevelStorageError("levels must be an owned tuple")
        try:
            return tuple(_LEVEL_KINDS[type(level)] for level in levels)
        except KeyError as error:
            unsupported = error.args[0]
            raise LevelStorageError(
                f"unsupported storage class {unsupported.__name__}"
            ) from error

    def _parent_count(self, level: int) -> int:
        counts = _stored_field(self, "_position_counts", "LevelTensorStorage")
        if type(counts) is not tuple:
            raise LevelStorageError("position counts must be an owned tuple")
        if level == 0:
            return 1
        if not 0 < level <= len(counts):
            raise LevelStorageError(
                f"no parent-position count for physical level {level}"
            )
        count = _expect_exact_int(counts[level - 1], "parent-position count")
        if count < 0:
            raise LevelStorageError("parent-position count must be nonnegative")
        return count

    def segment(self, level: int, parent_position: int) -> Tuple[int, int]:
        """Half-open child-position range one parent position dominates."""

        level = _expect_exact_int(level, "level")
        parent_position = _expect_exact_int(parent_position, "parent_position")
        levels = _stored_field(self, "levels", "LevelTensorStorage")
        if type(levels) is not tuple:
            raise LevelStorageError("levels must be an owned tuple")
        if not 0 <= level < len(levels):
            raise LevelStorageError(f"no level {level} in rank-{len(levels)}")
        stored = levels[level]
        if type(stored) is not CompressedLevel:
            raise LevelStorageError(f"level {level} has no stored segments")
        if not 0 <= parent_position < self._parent_count(level):
            raise LevelStorageError(
                f"parent position {parent_position} outside "
                f"[0, {self._parent_count(level)}) at level {level}"
            )
        offsets = _stored_field(stored, "seg_offsets", f"levels[{level}]")
        if type(offsets) is not tuple:
            raise LevelStorageError(
                f"levels[{level}].seg_offsets must be an owned tuple"
            )
        if parent_position + 1 >= len(offsets):
            raise LevelStorageError(
                f"levels[{level}].seg_offsets has no segment for parent "
                f"position {parent_position}"
            )
        start = _expect_exact_int(
            offsets[parent_position],
            f"levels[{level}].seg_offsets[{parent_position}]",
        )
        end = _expect_exact_int(
            offsets[parent_position + 1],
            f"levels[{level}].seg_offsets[{parent_position + 1}]",
        )
        coords = _stored_field(stored, "coords", f"levels[{level}]")
        if type(coords) is not tuple:
            raise LevelStorageError(f"levels[{level}].coords must be an owned tuple")
        if not 0 <= start <= end <= len(coords):
            raise LevelStorageError(
                f"levels[{level}] segment {parent_position} has invalid bounds "
                f"({start}, {end}) for {len(coords)} coordinates"
            )
        return (start, end)

    def coordinate_at(self, level: int, position: int) -> int:
        """The stored coordinate at one compressed-level position."""

        level = _expect_exact_int(level, "level")
        position = _expect_exact_int(position, "position")
        levels = _stored_field(self, "levels", "LevelTensorStorage")
        if type(levels) is not tuple:
            raise LevelStorageError("levels must be an owned tuple")
        if not 0 <= level < len(levels):
            raise LevelStorageError(f"no level {level} in rank-{len(levels)}")
        stored = levels[level]
        if type(stored) is not CompressedLevel:
            raise LevelStorageError(f"level {level} has no stored coordinates")
        coords = _stored_field(stored, "coords", f"levels[{level}]")
        if type(coords) is not tuple:
            raise LevelStorageError(f"levels[{level}].coords must be an owned tuple")
        if not 0 <= position < len(coords):
            raise LevelStorageError(
                f"position {position} outside [0, {len(coords)}) " f"at level {level}"
            )
        coordinate = _expect_exact_int(
            coords[position], f"levels[{level}].coords[{position}]"
        )
        shape = _stored_field(self, "shape", "LevelTensorStorage")
        modes = _stored_field(self, "modes", "LevelTensorStorage")
        if type(shape) is not tuple or type(modes) is not tuple or level >= len(modes):
            raise LevelStorageError("shape and modes must be owned rank-matched tuples")
        mode = _expect_exact_int(modes[level], f"modes[{level}]")
        if not 0 <= mode < len(shape):
            raise LevelStorageError(f"modes[{level}] is outside the logical rank")
        extent = _expect_exact_int(shape[mode], f"shape[{mode}]")
        if not 0 <= coordinate < extent:
            raise LevelStorageError(
                f"levels[{level}] coordinate {coordinate} outside [0, {extent})"
            )
        return coordinate

    def leaf_value(self, position: int) -> float:
        """The scalar the value-bearing leaf level owns at one position."""

        position = _expect_exact_int(position, "position")
        values = _stored_field(self, "values", "LevelTensorStorage")
        if type(values) is not tuple:
            raise LevelStorageError("values must be an owned tuple")
        if not 0 <= position < len(values):
            raise LevelStorageError(
                f"leaf position {position} outside [0, {len(values)})"
            )
        value = values[position]
        if type(value) is not float:
            raise LevelStorageError(f"values[{position}] must be an exact float")
        return value

    @classmethod
    def from_dense(
        cls,
        dense: object,
        shape: Sequence[int],
        modes: Sequence[int],
        kinds: Sequence[LevelKind],
    ) -> "LevelTensorStorage":
        """Build canonical storage from a logical nested-sequence tensor.

        ``dense`` nests in logical mode order; ``shape`` is explicit so
        zero-extent modes stay representable.  Compressed levels prune
        subtrees with no stored leaf entry (exact zeros are dropped, as in
        ``CsrMatrix.from_dense``); dense levels materialize every child.
        """

        for name, metadata in (
            ("shape", shape),
            ("modes", modes),
            ("kinds", kinds),
        ):
            if type(metadata) not in (list, tuple):
                raise LevelStorageError(f"{name} must be an owned list or tuple")
        shape = tuple(_expect_exact_int(extent, "shape entry") for extent in shape)
        modes = tuple(_expect_exact_int(mode, "modes entry") for mode in modes)
        kinds = tuple(kinds)
        rank = len(shape)
        if rank == 0 or any(extent < 0 for extent in shape):
            raise LevelStorageError("shape must contain nonnegative extents")
        if rank > MAX_LEVEL_STORAGE_RANK:
            raise LevelStorageError(
                f"rank {rank} exceeds the level-storage limit "
                f"{MAX_LEVEL_STORAGE_RANK}"
            )
        if len(modes) != rank or len(kinds) != rank:
            raise LevelStorageError("shape, modes, and kinds must agree on rank")
        if sorted(modes) != list(range(rank)):
            raise LevelStorageError(
                f"modes must be a permutation of the logical modes, got {modes}"
            )
        for position, kind in enumerate(kinds):
            if type(kind) is not LevelKind:
                raise LevelStorageError(
                    f"kinds[{position}] must be a LevelKind member, got "
                    f"{type(kind).__name__}"
                )
            if kind not in (LevelKind.DENSE, LevelKind.COMPRESSED):
                raise LevelStorageError(f"from_dense cannot build {kind.value} levels")
        all_values: Dict[Tuple[int, ...], float] = {}
        support: Dict[Tuple[int, ...], None] = {}

        def walk(layer: object, logical: Tuple[int, ...]) -> None:
            depth = len(logical)
            if depth == rank:
                if type(layer) is not float and type(layer) is not int:
                    raise LevelStorageError("dense entries must be numeric")
                try:
                    value = float(layer)
                except (OverflowError, TypeError, ValueError) as error:
                    raise LevelStorageError(
                        "dense entries must be representable numeric values"
                    ) from error
                physical = tuple(logical[m] for m in modes)
                all_values[physical] = value
                if value != 0.0:
                    support[physical] = None
                return
            if type(layer) not in (list, tuple):
                raise LevelStorageError(
                    f"dense input is ragged or mis-shaped at logical mode {depth}"
                )
            owned_layer = cast(Sequence[object], layer)
            if len(owned_layer) != shape[depth]:
                raise LevelStorageError(
                    f"dense input is ragged or mis-shaped at logical mode {depth}"
                )
            for coordinate, child in enumerate(owned_layer):
                walk(child, logical + (coordinate,))

        walk(dense, ())
        prefixes: List[Tuple[int, ...]] = [()]
        levels: List[object] = []
        for number, kind in enumerate(kinds):
            extent = shape[modes[number]]
            if kind is LevelKind.DENSE:
                levels.append(DenseLevel(extent))
                prefixes = [
                    prefix + (coordinate,)
                    for prefix in prefixes
                    for coordinate in range(extent)
                ]
            elif kind is LevelKind.COMPRESSED:
                seg_offsets = [0]
                expanded: List[Tuple[int, ...]] = []
                for prefix in prefixes:
                    present = sorted(
                        {key[number] for key in support if key[:number] == prefix}
                    )
                    expanded.extend(prefix + (coordinate,) for coordinate in present)
                    seg_offsets.append(len(expanded))
                levels.append(
                    CompressedLevel(
                        tuple(seg_offsets),
                        tuple(prefix[number] for prefix in expanded),
                    )
                )
                prefixes = expanded
            else:
                raise LevelStorageError(f"from_dense cannot build {kind.value} levels")
        values = tuple(all_values[prefix] for prefix in prefixes)
        return cls(shape=shape, modes=modes, levels=tuple(levels), values=values)

    def to_dense(self) -> object:
        """Materialize the stored tensor as nested lists in logical order."""

        self.validate()

        def zeros(depth: int) -> Any:
            if depth == len(self.shape) - 1:
                return [0.0] * self.shape[depth]
            return [zeros(depth + 1) for _ in range(self.shape[depth])]

        dense: Any = zeros(0)
        prefixes: List[Tuple[int, ...]] = [()]
        for number, level in enumerate(self.levels):
            if type(level) is DenseLevel:
                prefixes = [
                    prefix + (coordinate,)
                    for prefix in prefixes
                    for coordinate in range(level.extent)
                ]
            else:
                if type(level) is not CompressedLevel:
                    raise LevelStorageError(
                        f"levels[{number}] has unsupported storage class "
                        f"{type(level).__name__}"
                    )
                expanded: List[Tuple[int, ...]] = []
                for parent, prefix in enumerate(prefixes):
                    start, end = (
                        level.seg_offsets[parent],
                        level.seg_offsets[parent + 1],
                    )
                    expanded.extend(
                        prefix + (level.coords[position],)
                        for position in range(start, end)
                    )
                prefixes = expanded
        for position, prefix in enumerate(prefixes):
            logical = [0] * len(self.shape)
            for number, coordinate in enumerate(prefix):
                logical[self.modes[number]] = coordinate
            target = dense
            for coordinate in logical[:-1]:
                target = target[coordinate]
            target[logical[-1]] = self.values[position]
        return dense


def from_csr(matrix: CsrMatrix) -> LevelTensorStorage:
    """The CSR input adapter: one canonical container, generic storage out."""

    if type(matrix) is not CsrMatrix:
        raise LevelStorageError(
            f"from_csr needs a CsrMatrix, got {type(matrix).__name__}"
        )
    try:
        snapshot = CsrMatrix(
            n_rows=cast(int, _stored_field(matrix, "n_rows", "CsrMatrix")),
            n_cols=cast(int, _stored_field(matrix, "n_cols", "CsrMatrix")),
            indptr=cast(Tuple[int, ...], _stored_field(matrix, "indptr", "CsrMatrix")),
            indices=cast(
                Tuple[int, ...], _stored_field(matrix, "indices", "CsrMatrix")
            ),
            values=cast(
                Tuple[float, ...], _stored_field(matrix, "values", "CsrMatrix")
            ),
        )
    except (CsrFormatError, TypeError) as error:
        raise LevelStorageError(f"invalid CsrMatrix: {error}") from error
    return LevelTensorStorage(
        shape=(snapshot.n_rows, snapshot.n_cols),
        modes=(0, 1),
        levels=(
            DenseLevel(snapshot.n_rows),
            CompressedLevel(snapshot.indptr, snapshot.indices),
        ),
        values=snapshot.values,
    )


class CsrOutputBuilder:
    """The CSR assembly adapter: order-checked appends into one CSR output."""

    def __init__(self, name: str, shape: Tuple[int, ...]) -> None:
        if type(name) is not str or not name:
            raise LevelStorageError("CSR output name must be a nonempty str")
        if (
            type(shape) is not tuple
            or len(shape) != 2
            or any(type(extent) is not int or extent < 0 for extent in shape)
        ):
            raise LevelStorageError(
                "CSR output shape must be a rank-2 tuple of nonnegative exact ints"
            )
        self.name = name
        self.n_rows, self.n_cols = shape
        self.rows: List[int] = []
        self.columns: List[int] = []
        self.values: List[float] = []

    def append(self, coords: Tuple[int, ...], value: float) -> None:
        if (
            type(coords) is not tuple
            or len(coords) != 2
            or any(type(coord) is not int for coord in coords)
        ):
            raise LevelStorageError("CSR append coordinates must be two exact ints")
        if type(value) is not float:
            raise LevelStorageError("CSR append value must be an exact float")
        row, column = coords
        if not 0 <= row < self.n_rows or not 0 <= column < self.n_cols:
            raise LevelStorageError(
                f"append to {self.name} at {coords} escapes shape "
                f"({self.n_rows}, {self.n_cols})"
            )
        if self.rows and (row, column) <= (self.rows[-1], self.columns[-1]):
            raise LevelStorageError(
                f"appends to {self.name} must be lexicographically increasing; "
                f"got {coords} after ({self.rows[-1]}, {self.columns[-1]})"
            )
        self.rows.append(row)
        self.columns.append(column)
        self.values.append(value)

    def finish(self) -> CsrMatrix:
        indptr: List[int] = [0]
        position = 0
        for row in range(self.n_rows):
            while position < len(self.rows) and self.rows[position] == row:
                position += 1
            indptr.append(position)
        return CsrMatrix(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            indptr=tuple(indptr),
            indices=tuple(self.columns),
            values=tuple(self.values),
        )


@dataclass(frozen=True)
class LevelTensor:
    """One canonical, immutable level-stored tensor of DENSE/COMPRESSED levels.

    ``positions``/``coordinates`` hold one entry per physical level:
    compressed levels carry their position and coordinate tuples, dense
    levels carry ``None`` for both (dense storage is implicit).  Explicit
    zero values are permitted, exactly as for :class:`CsrMatrix`.
    """

    shape: Tuple[int, ...]
    level_kinds: Tuple[LevelKind, ...]
    positions: Tuple[Any, ...]
    coordinates: Tuple[Any, ...]
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "shape",
            "level_kinds",
            "positions",
            "coordinates",
            "values",
        ):
            _stored_field(self, field_name, "LevelTensor")
        if type(self.shape) is not tuple or not self.shape:
            raise LevelStorageError("LevelTensor.shape must be a nonempty tuple")
        if len(self.shape) > MAX_LEVEL_STORAGE_RANK:
            raise LevelStorageError(
                f"rank {len(self.shape)} exceeds the level-storage limit "
                f"{MAX_LEVEL_STORAGE_RANK}"
            )
        for level, extent in enumerate(self.shape):
            if _expect_exact_int(extent, f"shape[{level}]") < 0:
                raise LevelStorageError("LevelTensor extents must be nonnegative")
        rank = len(self.shape)
        if (
            type(self.level_kinds) is not tuple
            or len(self.level_kinds) != rank
            or any(
                type(kind) is not LevelKind
                or (kind is not LevelKind.DENSE and kind is not LevelKind.COMPRESSED)
                for kind in self.level_kinds
            )
        ):
            raise LevelStorageError(
                "LevelTensor needs one DENSE/COMPRESSED kind per level"
            )
        if (
            type(self.positions) is not tuple
            or type(self.coordinates) is not tuple
            or len(self.positions) != rank
            or len(self.coordinates) != rank
        ):
            raise LevelStorageError(
                "LevelTensor position and coordinate streams must match its rank"
            )

        parent_count = 1
        for level, kind in enumerate(self.level_kinds):
            offsets = self.positions[level]
            coords = self.coordinates[level]
            if kind is LevelKind.DENSE:
                if offsets is not None or coords is not None:
                    raise LevelStorageError(
                        f"dense level {level} must use implicit storage"
                    )
                parent_count *= self.shape[level]
                continue
            if type(offsets) is not tuple or type(coords) is not tuple:
                raise LevelStorageError(
                    f"compressed level {level} needs owned position and "
                    "coordinate tuples"
                )
            if len(offsets) != parent_count + 1:
                raise LevelStorageError(
                    f"compressed level {level} has {len(offsets)} offsets for "
                    f"{parent_count} parents"
                )
            previous = 0
            for position, offset in enumerate(offsets):
                _expect_exact_int(offset, f"positions[{level}][{position}]")
                if position == 0 and offset != 0:
                    raise LevelStorageError(
                        f"compressed level {level} positions must start at zero"
                    )
                if offset < previous:
                    raise LevelStorageError(
                        f"compressed level {level} positions must be nondecreasing"
                    )
                previous = offset
            if offsets[-1] != len(coords):
                raise LevelStorageError(
                    f"compressed level {level} positions must terminate at "
                    "the coordinate count"
                )
            for parent in range(parent_count):
                last = -1
                for position in range(offsets[parent], offsets[parent + 1]):
                    coord = _expect_exact_int(
                        coords[position],
                        f"coordinates[{level}][{position}]",
                    )
                    if not 0 <= coord < self.shape[level]:
                        raise LevelStorageError(
                            f"coordinate {coord} escapes level {level} extent "
                            f"{self.shape[level]}"
                        )
                    if coord <= last:
                        raise LevelStorageError(
                            f"compressed level {level} coordinates must be "
                            "strictly increasing per segment"
                        )
                    last = coord
            parent_count = len(coords)
        if type(self.values) is not tuple or len(self.values) != parent_count:
            raise LevelStorageError(
                "LevelTensor needs one owned value per leaf position"
            )
        for position, value in enumerate(self.values):
            if type(value) is not float:
                raise LevelStorageError(
                    f"LevelTensor.values[{position}] must be an exact float"
                )


class LevelOutputBuilder:
    """Level-general ordered assembly: order-checked appends, level storage.

    Admits any rank of DENSE and COMPRESSED levels with at least one
    COMPRESSED level.  Appends must be lexicographically strictly
    increasing.  ``finish`` derives the per-level position/coordinate
    storage from the append stream alone and enforces the dense-suffix
    contract: every level after the last compressed level is dense storage
    with no skip representation, so each materialized position of the last
    compressed level must receive its complete Cartesian block of trailing
    dense coordinates.
    """

    def __init__(
        self,
        name: str,
        shape: Tuple[int, ...],
        level_kinds: Tuple[LevelKind, ...],
    ) -> None:
        if type(name) is not str or not name:
            raise LevelStorageError("output name must be a nonempty str")
        if (
            type(shape) is not tuple
            or not shape
            or any(type(extent) is not int or extent < 0 for extent in shape)
        ):
            raise LevelStorageError(
                "output shape must be a nonempty tuple of nonnegative exact ints"
            )
        if (
            type(level_kinds) is not tuple
            or len(level_kinds) != len(shape)
            or any(type(kind) is not LevelKind for kind in level_kinds)
        ):
            raise LevelStorageError(
                "output level kinds must be one LevelKind per storage level"
            )
        if any(
            kind is not LevelKind.DENSE and kind is not LevelKind.COMPRESSED
            for kind in level_kinds
        ):
            raise LevelStorageError(
                "ordered assembly supports DENSE and COMPRESSED levels only"
            )
        if not any(kind is LevelKind.COMPRESSED for kind in level_kinds):
            raise LevelStorageError(
                "ordered assembly requires at least one compressed level"
            )
        self.name = name
        self.shape = shape
        self.level_kinds = level_kinds
        self.entries: List[Tuple[Tuple[int, ...], float]] = []

    def append(self, coords: Tuple[int, ...], value: float) -> None:
        if (
            type(coords) is not tuple
            or len(coords) != len(self.shape)
            or any(type(coord) is not int for coord in coords)
        ):
            raise LevelStorageError(
                f"append coordinates must be {len(self.shape)} exact ints"
            )
        if type(value) is not float:
            raise LevelStorageError("append value must be an exact float")
        for level, (coord, extent) in enumerate(zip(coords, self.shape)):
            if not 0 <= coord < extent:
                raise LevelStorageError(
                    f"append to {self.name} at {coords} escapes extent "
                    f"{extent} at level {level}"
                )
        if self.entries and coords <= self.entries[-1][0]:
            raise LevelStorageError(
                f"appends to {self.name} must be lexicographically "
                f"increasing; got {coords} after {self.entries[-1][0]}"
            )
        self.entries.append((coords, value))

    def finish(self) -> LevelTensor:
        rank = len(self.shape)
        last_compressed = max(
            level
            for level, kind in enumerate(self.level_kinds)
            if kind is LevelKind.COMPRESSED
        )
        suffix_extents = self.shape[last_compressed + 1 :]
        suffix_cells = 1
        for extent in suffix_extents:
            suffix_cells *= extent

        # The dense-suffix contract: entries group into complete, in-order
        # Cartesian blocks per materialized last-compressed position.
        if suffix_extents:
            if suffix_cells == 0:
                # No coordinate tuple can address a zero-extent dense suffix,
                # so the canonical append stream is necessarily empty.  Keep
                # the empty sparse structure representable without dividing
                # or stepping by zero.
                if self.entries:
                    raise LevelStorageError(
                        f"assembly of {self.name} contains entries beneath a "
                        "zero-extent trailing dense level"
                    )
                suffix_extents = ()
            else:
                if len(self.entries) % suffix_cells != 0:
                    raise LevelStorageError(
                        f"assembly of {self.name} left an incomplete trailing "
                        "dense block"
                    )

        if suffix_extents:

            def _expected_suffix(offset: int) -> Tuple[int, ...]:
                coordinates = [0] * len(suffix_extents)
                for level in range(len(suffix_extents) - 1, -1, -1):
                    offset, coordinates[level] = divmod(offset, suffix_extents[level])
                return tuple(coordinates)

            for block_start in range(0, len(self.entries), suffix_cells):
                block = self.entries[block_start : block_start + suffix_cells]
                head = block[0][0][: last_compressed + 1]
                for offset, (coords, _) in enumerate(block):
                    if coords[: last_compressed + 1] != head:
                        raise LevelStorageError(
                            f"assembly of {self.name} split a dense block "
                            "across materialized parents"
                        )
                    if coords[last_compressed + 1 :] != _expected_suffix(offset):
                        raise LevelStorageError(
                            f"assembly of {self.name} skipped a trailing " "dense cell"
                        )

        # Derive per-level compressed storage from the ordered stream.
        positions: List[Any] = [None] * rank
        coordinates: List[Any] = [None] * rank
        # Parent-position keys per level: the stored prefix that owns one
        # position of level l (dense levels contribute their coordinate,
        # compressed levels contribute their materialized segment).
        for level, kind in enumerate(self.level_kinds):
            if kind is not LevelKind.COMPRESSED:
                continue
            level_coords: List[int] = []
            level_pos: List[int] = [0]
            previous_prefix: Any = None
            for coords, _ in self.entries:
                prefix_key = coords[: level + 1]
                if prefix_key != previous_prefix:
                    level_coords.append(coords[level])
                    previous_prefix = prefix_key
            # One position segment per materialized parent position, in
            # order; parents the stream never touched own empty segments.
            segment_counts: Dict[Any, int] = {}
            previous_prefix = None
            for coords, _ in self.entries:
                parent_key = self._parent_key(coords, level)
                prefix_key = coords[: level + 1]
                if prefix_key != previous_prefix:
                    segment_counts[parent_key] = segment_counts.get(parent_key, 0) + 1
                    previous_prefix = prefix_key
            running = 0
            for parent_key in self._materialized_parents(level):
                running += segment_counts.get(parent_key, 0)
                level_pos.append(running)
            positions[level] = tuple(level_pos)
            coordinates[level] = tuple(level_coords)

        return LevelTensor(
            shape=self.shape,
            level_kinds=self.level_kinds,
            positions=tuple(positions),
            coordinates=tuple(coordinates),
            values=tuple(value for _, value in self.entries),
        )

    def _parent_key(self, coords: Tuple[int, ...], level: int) -> Tuple[int, ...]:
        """The materialized-parent key of one entry at ``level``.

        Compressed ancestors contribute their coordinate (their stored
        segment identity); dense ancestors contribute their coordinate as
        an implicit dense position component.  The key is simply the
        coordinate prefix — parent-position identity and prefix identity
        coincide because every ancestor position is reached through its
        coordinates.
        """

        return coords[:level]

    def _materialized_parents(self, level: int) -> List[Tuple[int, ...]]:
        """Every parent position of ``level`` in storage order.

        Walk levels above ``level``: a dense ancestor multiplies the
        current parents by its full extent (dense storage materializes
        every cell, including cells the stream never touched); a
        compressed ancestor materializes exactly the distinct stored
        prefixes, which the lexicographic append order already lists in
        storage order.
        """

        parents: List[Tuple[int, ...]] = [()]
        for ancestor in range(level):
            if self.level_kinds[ancestor] is LevelKind.DENSE:
                parents = [
                    prefix + (coord,)
                    for prefix in parents
                    for coord in range(self.shape[ancestor])
                ]
            else:
                seen: set = set()
                stored: List[Tuple[int, ...]] = []
                for coords, _ in self.entries:
                    prefix = coords[: ancestor + 1]
                    if prefix not in seen:
                        seen.add(prefix)
                        stored.append(prefix)
                parents = stored
        return parents

    def _parent_position_count(self, level: int) -> int:
        return len(self._materialized_parents(level))
