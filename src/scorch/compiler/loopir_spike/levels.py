"""Format-neutral level storage for the Phase-3.5 LoopIR spike.

The interpreter's execution core consumes only this small interface —
``segment``, ``coordinate_at``, and ``leaf_value`` on a validated
:class:`LevelTensorStorage` — so its traversal is defined over generic
physical levels rather than any one container.  CSR is exactly one adapter
(:func:`from_csr` on the input side, :class:`CsrOutputBuilder` on the
assembly side); DCSR, CSC, CSF-like, and any other DENSE/COMPRESSED level
composition with any physical-to-logical mode permutation bind through the
same storage class.

Canonical means: exact int/float scalars; per-level consistency between the
parent position count and the child level's storage (``seg_offsets`` of
length ``parents + 1``, starting at zero, nondecreasing; per-segment
coordinates strictly increasing and inside the stored dimension's extent);
dense extents equal to the stored dimension's extent; and a value stream
owned by the leaf level with exactly one scalar per leaf position.
Everything unexpected raises :class:`LevelStorageError` rather than being
coerced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .csr import CsrMatrix
from .nodes import LevelKind


class LevelStorageError(Exception):
    """A storage construction or access violated the canonical contract."""


def _expect_exact_int(value: object, what: str) -> int:
    if type(value) is not int:
        raise LevelStorageError(
            f"{what} must be an exact int, got {type(value).__name__}"
        )
    return value


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
        if type(self.shape) is not tuple or not self.shape:
            raise LevelStorageError("shape must be a nonempty owned tuple")
        for position, extent in enumerate(self.shape):
            if _expect_exact_int(extent, f"shape[{position}]") < 0:
                raise LevelStorageError("shape extents must be nonnegative")
        rank = len(self.shape)
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
                declared = _expect_exact_int(level.extent, f"levels[{number}].extent")
                if declared != extent:
                    raise LevelStorageError(
                        f"levels[{number}] extent {declared} does not match "
                        f"dimension extent {extent}"
                    )
                parent_count = parent_count * extent
            elif type(level) is CompressedLevel:
                offsets = level.seg_offsets
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
                coords = level.coords
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

    @property
    def kinds(self) -> Tuple[LevelKind, ...]:
        """Per-physical-level kinds, for binding-time declaration checks."""

        return tuple(_LEVEL_KINDS[type(level)] for level in self.levels)

    def _parent_count(self, level: int) -> int:
        counts: Tuple[int, ...] = getattr(self, "_position_counts")
        return counts[level - 1] if level > 0 else 1

    def segment(self, level: int, parent_position: int) -> Tuple[int, int]:
        """Half-open child-position range one parent position dominates."""

        if not 0 <= level < len(self.levels):
            raise LevelStorageError(f"no level {level} in rank-{len(self.levels)}")
        stored = self.levels[level]
        if type(stored) is not CompressedLevel:
            raise LevelStorageError(f"level {level} has no stored segments")
        if not 0 <= parent_position < self._parent_count(level):
            raise LevelStorageError(
                f"parent position {parent_position} outside "
                f"[0, {self._parent_count(level)}) at level {level}"
            )
        return (
            stored.seg_offsets[parent_position],
            stored.seg_offsets[parent_position + 1],
        )

    def coordinate_at(self, level: int, position: int) -> int:
        """The stored coordinate at one compressed-level position."""

        if not 0 <= level < len(self.levels):
            raise LevelStorageError(f"no level {level} in rank-{len(self.levels)}")
        stored = self.levels[level]
        if type(stored) is not CompressedLevel:
            raise LevelStorageError(f"level {level} has no stored coordinates")
        if not 0 <= position < len(stored.coords):
            raise LevelStorageError(
                f"position {position} outside [0, {len(stored.coords)}) "
                f"at level {level}"
            )
        return stored.coords[position]

    def leaf_value(self, position: int) -> float:
        """The scalar the value-bearing leaf level owns at one position."""

        if not 0 <= position < len(self.values):
            raise LevelStorageError(
                f"leaf position {position} outside [0, {len(self.values)})"
            )
        return self.values[position]

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

        shape = tuple(_expect_exact_int(extent, "shape entry") for extent in shape)
        modes = tuple(_expect_exact_int(mode, "modes entry") for mode in modes)
        kinds = tuple(kinds)
        rank = len(shape)
        if len(modes) != rank or len(kinds) != rank:
            raise LevelStorageError("shape, modes, and kinds must agree on rank")
        entries: Dict[Tuple[int, ...], float] = {}

        def walk(layer: object, logical: Tuple[int, ...]) -> None:
            depth = len(logical)
            if depth == rank:
                if type(layer) is not float and type(layer) is not int:
                    raise LevelStorageError("dense entries must be numeric")
                value = float(layer)
                if value != 0.0:
                    entries[tuple(logical[m] for m in modes)] = value
                return
            if not isinstance(layer, (list, tuple)) or len(layer) != shape[depth]:
                raise LevelStorageError(
                    f"dense input is ragged or mis-shaped at logical mode {depth}"
                )
            for coordinate, child in enumerate(layer):
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
                        {key[number] for key in entries if key[:number] == prefix}
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
        values = tuple(entries.get(prefix, 0.0) for prefix in prefixes)
        return cls(shape=shape, modes=modes, levels=tuple(levels), values=values)

    def to_dense(self) -> object:
        """Materialize the stored tensor as nested lists in logical order."""

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
    return LevelTensorStorage(
        shape=(matrix.n_rows, matrix.n_cols),
        modes=(0, 1),
        levels=(
            DenseLevel(matrix.n_rows),
            CompressedLevel(matrix.indptr, matrix.indices),
        ),
        values=matrix.values,
    )


class CsrOutputBuilder:
    """The CSR assembly adapter: order-checked appends into one CSR output."""

    def __init__(self, name: str, shape: Tuple[int, ...]) -> None:
        self.name = name
        self.n_rows, self.n_cols = shape
        self.rows: List[int] = []
        self.columns: List[int] = []
        self.values: List[float] = []

    def append(self, coords: Tuple[int, ...], value: float) -> None:
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
