"""Immutable sparse index and payload storage value objects."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

# Mapping/Sequence come from collections.abc, not typing: these are used for
# runtime isinstance checks on a per-call path, and typing's generic aliases
# route isinstance through __subclasscheck__ at 153 ns against 73 ns for the
# abc. `from __future__ import annotations` above means the annotations that
# also use these names are never evaluated at runtime.
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional, Tuple

import torch

from .exceptions import (
    TensorDeviceError,
    TensorIndexError,
    TensorLayoutError,
    TensorStorageError,
    TensorTypeError,
)
from .format import FormatInput, LevelType, TensorFormat, parse_format
from .layout import TensorLayout, validate_runtime_contract

IndexModes = Tuple[Tuple[torch.Tensor, ...], ...]
_INDEX_DTYPES = (torch.int32, torch.int64)


def _tensor_value_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and left.device == right.device
        and left.layout == right.layout
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left, right)
    )


def _expected_level_arity(level_type: LevelType) -> int:
    if level_type == LevelType.DENSE:
        return 0
    if level_type == LevelType.COMPRESSED:
        return 2
    if level_type in (LevelType.COORDINATE, LevelType.SINGLETON):
        return 1
    raise TensorIndexError(f"unsupported level type {level_type}")


def _normalize_mode_indices(
    mode_indices: Sequence[Sequence[torch.Tensor]], tensor_format: TensorFormat
) -> IndexModes:
    if isinstance(mode_indices, (str, bytes)) or not isinstance(mode_indices, Sequence):
        raise TensorTypeError("mode_indices must be a sequence with one entry per mode")
    if len(mode_indices) != tensor_format.get_order():
        raise TensorIndexError(
            f"mode_indices has rank {len(mode_indices)}, expected "
            f"{tensor_format.get_order()} for format {tensor_format}"
        )

    normalized = []
    for mode, (arrays, level_type) in enumerate(
        zip(mode_indices, tensor_format.get_level_types())
    ):
        if isinstance(arrays, (str, bytes)) or not isinstance(arrays, Sequence):
            raise TensorTypeError(f"mode_indices[{mode}] must be a sequence")
        expected = _expected_level_arity(level_type)
        if len(arrays) != expected:
            raise TensorIndexError(
                f"mode {mode} ({level_type.name.lower()}) requires {expected} "
                f"index arrays, got {len(arrays)}"
            )
        normalized_arrays = []
        for slot, index in enumerate(arrays):
            if not isinstance(index, torch.Tensor):
                raise TensorTypeError(
                    f"mode_indices[{mode}][{slot}] must be a torch.Tensor"
                )
            if index.layout != torch.strided:
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] must use strided storage"
                )
            if index.device.type != "cpu":
                raise TensorDeviceError(
                    f"mode_indices[{mode}][{slot}] must be on CPU, got {index.device}"
                )
            if index.dtype not in _INDEX_DTYPES:
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] must use int32 or int64, "
                    f"got {index.dtype}"
                )
            if index.dim() != 1:
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] must be one-dimensional"
                )
            if not index.is_contiguous():
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] must be contiguous"
                )
            # Index coordinates are structural data.  Own an independent copy so
            # caller mutation cannot invalidate a previously validated object.
            normalized_arrays.append(index.detach().clone())
        normalized.append(tuple(normalized_arrays))
    return tuple(normalized)


@dataclass(frozen=True, init=False)
class TensorLevelIndex:
    """Validated immutable index arrays for one physical storage level."""

    _indices: Tuple[torch.Tensor, ...]
    _level_type: LevelType

    def __init__(self, indices: Sequence[torch.Tensor], level_type: LevelType) -> None:
        if not isinstance(level_type, LevelType):
            raise TensorTypeError("level_type must be a LevelType")
        tensor_format = TensorFormat([level_type])
        normalized = _normalize_mode_indices([indices], tensor_format)[0]
        object.__setattr__(self, "_indices", normalized)
        object.__setattr__(self, "_level_type", level_type)

    @property
    def indices(self) -> List[torch.Tensor]:
        return [index.clone() for index in self._indices]

    @property
    def level_type(self) -> LevelType:
        return self._level_type

    def __str__(self) -> str:
        return (
            f"TensorLevelIndex(level_type={self._level_type}, "
            f"indices={self._indices})"
        )

    __repr__ = __str__


@dataclass(frozen=True, init=False, eq=False)
class TensorIndex:
    """Immutable compatibility descriptor for format and sparse index arrays.

    Shape-dependent checks are completed when the descriptor is combined with a
    :class:`TensorLayout` in :class:`SparseStorage`. Caller containers and index
    tensors are copied, so later caller mutation cannot invalidate the descriptor;
    they are never silently cast or replaced in the caller.
    """

    _format: TensorFormat
    _mode_indices: IndexModes
    _mode_order: Tuple[int, ...]
    _index_dtype: torch.dtype

    def __init__(
        self,
        tensor_format: FormatInput,
        mode_indices: Sequence[Sequence[torch.Tensor]],
        mode_order: Optional[Sequence[int]] = None,
        index_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if tensor_format is None:
            raise TensorIndexError(
                "tensor_format is required; use TensorFormat() for a scalar"
            )
        tensor_format = parse_format(tensor_format)
        normalized = _normalize_mode_indices(mode_indices, tensor_format)
        if mode_order is None:
            order = tuple(range(tensor_format.get_order()))
        else:
            if isinstance(mode_order, (str, bytes)) or not isinstance(
                mode_order, Sequence
            ):
                raise TensorTypeError("mode_order must be a sequence of integers")
            order = tuple(mode_order)
        if any(isinstance(mode, bool) or not isinstance(mode, int) for mode in order):
            raise TensorTypeError("mode_order entries must be integers")
        if len(order) != tensor_format.get_order() or sorted(order) != list(
            range(tensor_format.get_order())
        ):
            raise TensorIndexError(
                f"mode_order must be a permutation of range({tensor_format.get_order()})"
            )

        observed_dtypes = {index.dtype for arrays in normalized for index in arrays}
        if len(observed_dtypes) > 1:
            raise TensorIndexError(
                "all sparse index arrays must use one common index dtype"
            )
        observed = next(iter(observed_dtypes), None)
        declared = (
            torch.int32
            if index_dtype is None and observed is None
            else (observed if index_dtype is None else index_dtype)
        )
        if declared not in _INDEX_DTYPES:
            raise TensorIndexError(
                f"index_dtype must be torch.int32 or torch.int64, got {declared}"
            )
        if observed is not None and observed != declared:
            raise TensorIndexError(
                f"declared index dtype {declared} does not match arrays of dtype {observed}"
            )
        object.__setattr__(self, "_format", tensor_format)
        object.__setattr__(self, "_mode_indices", normalized)
        object.__setattr__(self, "_mode_order", order)
        object.__setattr__(self, "_index_dtype", declared)

    @classmethod
    def _from_layout(
        cls, layout: TensorLayout, mode_indices: IndexModes
    ) -> "TensorIndex":
        index = object.__new__(cls)
        object.__setattr__(index, "_format", layout.format)
        object.__setattr__(index, "_mode_indices", mode_indices)
        object.__setattr__(index, "_mode_order", layout.permutation)
        object.__setattr__(index, "_index_dtype", layout.index_dtype)
        return index

    @property
    def format(self) -> TensorFormat:
        return self._format

    @property
    def mode_indices(self) -> List[List[torch.Tensor]]:
        """Return a deep defensive copy of the structural index arrays."""
        return [[index.clone() for index in arrays] for arrays in self._mode_indices]

    @property
    def mode_order(self) -> List[int]:
        """Return a defensive copy of the physical-to-logical permutation."""
        return list(self._mode_order)

    @property
    def index_dtype(self) -> torch.dtype:
        return self._index_dtype

    def __str__(self) -> str:
        return f"TensorIndex({self.format})"

    __repr__ = __str__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorIndex):
            return False
        return (
            self._format == other._format
            and self._mode_order == other._mode_order
            and self._index_dtype == other._index_dtype
            and len(self._mode_indices) == len(other._mode_indices)
            and all(
                len(left) == len(right)
                and all(_tensor_value_equal(a, b) for a, b in zip(left, right))
                for left, right in zip(self._mode_indices, other._mode_indices)
            )
        )

    __hash__: Any = None

    def copy(self) -> "TensorIndex":
        copied = tuple(
            tuple(index.clone().detach() for index in arrays)
            for arrays in self._mode_indices
        )
        return TensorIndex._from_layout(
            TensorLayout.from_physical_shape(
                # Only format/permutation/dtype are consumed by _from_layout.
                tuple(1 for _ in self._mode_order),
                self._format,
                self._mode_order,
                self._index_dtype,
            ),
            copied,
        )

    def get_mode_index(self, mode: int) -> List[torch.Tensor]:
        if mode < 0:
            raise TensorIndexError(f"mode {mode} is outside tensor rank")
        try:
            return [index.clone() for index in self._mode_indices[mode]]
        except IndexError as error:
            raise TensorIndexError(f"mode {mode} is outside tensor rank") from error

    def get_mode_indices(self) -> List[List[torch.Tensor]]:
        return self.mode_indices

    def get_order(self) -> int:
        return self._format.get_order()

    def get_format(self) -> TensorFormat:
        return self._format

    def get_mode_order(self) -> List[int]:
        return self.mode_order

    def get_level_type(self, mode: int) -> LevelType:
        if mode < 0:
            raise TensorIndexError(f"mode {mode} is outside tensor rank")
        try:
            return self._format.get_level_types()[mode]
        except IndexError as error:
            raise TensorIndexError(f"mode {mode} is outside tensor rank") from error

    def get_level_types(self) -> List[LevelType]:
        return self._format.get_level_types()

    def get_size(self, mode: int) -> int:
        arrays = self.get_mode_index(mode)
        if not arrays:
            raise TensorIndexError(f"dense mode {mode} has no explicit index size")
        return arrays[0].numel()

    def get_sizes(self) -> List[int]:
        return [self.get_size(mode) for mode in range(self.get_order())]


def _check_coordinate_bounds(coordinate: torch.Tensor, extent: int, label: str) -> None:
    if coordinate.numel() == 0:
        return
    minimum = int(coordinate.min().item())
    maximum = int(coordinate.max().item())
    if minimum < 0 or maximum >= extent:
        raise TensorIndexError(
            f"{label} contains coordinate range [{minimum}, {maximum}] outside [0, {extent})"
        )


def _check_sorted_within_parents(
    positions: torch.Tensor, coordinates: torch.Tensor, mode: int
) -> None:
    """Every parent's coordinates must be nondecreasing within that parent.

    Whole-array, in a handful of kernels. The obvious formulation of this predicate --
    "no entry is smaller than the one before it" -- is wrong, because coordinates
    restart at every parent, so a real CSR descends at nearly every row boundary. The
    boundaries are exactly the interior entries of ``positions``, so masking those
    positions out of the comparison leaves precisely the descents that occur *inside* a
    parent.

    This replaces a Python loop over parents that sliced the coordinates, launched a
    comparison and synced on ``.item()`` once per parent: 3.7 us per row, so 74 ms to
    wrap a 20,000-row CSR, and it ran on every STensor built over a compressed level --
    including every generated kernel's own output. Measured on ``from_torch`` over a CSR,
    all else equal, it makes the whole wrap 11.6x faster at 128 rows, 69x at 1,000, 243x
    at 20,000 and 399x at 100,000 on the x86 host (12.3x / 65x / 132x / 262x on an M5),
    with identical verdicts -- a differential test against the old implementation, kept
    verbatim as the oracle, pins that.

    The message names the offending parent, so the reporting path recovers it. That
    costs two more kernels and only runs when the storage is actually malformed.
    """
    count = coordinates.numel()
    if count < 2:
        return
    descends = coordinates[1:] < coordinates[:-1]
    # A descent at index i compares coordinates[i+1] against coordinates[i], so a
    # parent starting at position p makes index p-1 a boundary. Interior entries of
    # `positions` outside [1, count-1] describe empty parents at either end, which have
    # no comparison to exempt.
    starts = positions[1:-1].to(torch.long)
    starts = starts[(starts >= 1) & (starts <= count - 1)]
    if starts.numel():
        descends[starts - 1] = False
    if not bool(descends.any()):
        return
    # Parents are contiguous and ordered, so the first offending position lies in the
    # first offending parent -- which is the one the row loop reported.
    first = int(torch.nonzero(descends)[0].item())
    boundaries = positions.to(torch.long)
    parent = int(torch.searchsorted(boundaries, first, right=True).item()) - 1
    raise TensorIndexError(
        f"compressed mode {mode} coordinates must be sorted within parent {parent}"
    )


def _validate_index_storage(
    layout: TensorLayout, mode_indices: IndexModes, values: torch.Tensor
) -> None:
    nnz = values.numel()
    level_types = layout.format.get_level_types()

    for mode, arrays in enumerate(mode_indices):
        for slot, index in enumerate(arrays):
            if index.dtype != layout.index_dtype:
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] has dtype {index.dtype}, "
                    f"expected layout index dtype {layout.index_dtype}"
                )
            if index.device != values.device:
                raise TensorDeviceError(
                    f"mode_indices[{mode}][{slot}] and values must share a device"
                )

    if all(level_type == LevelType.DENSE for level_type in level_types):
        expected = math.prod(layout.physical_shape)
        if nnz != expected:
            raise TensorStorageError(
                f"dense values contain {nnz} elements, expected {expected} for shape "
                f"{layout.physical_shape}"
            )
        return

    if level_types and all(
        level_type == LevelType.COORDINATE for level_type in level_types
    ):
        coordinate_values = []
        for mode, arrays in enumerate(mode_indices):
            coordinate = arrays[0]
            if coordinate.numel() != nnz:
                raise TensorStorageError(
                    f"COO mode {mode} has {coordinate.numel()} coordinates, "
                    f"but values has {nnz} elements"
                )
            _check_coordinate_bounds(
                coordinate, layout.physical_shape[mode], f"COO mode {mode}"
            )
            coordinate_values.append(coordinate.tolist())
        for position in range(1, nnz):
            previous = tuple(
                coordinate_values[mode][position - 1]
                for mode in range(len(coordinate_values))
            )
            current = tuple(
                coordinate_values[mode][position]
                for mode in range(len(coordinate_values))
            )
            if current < previous:
                raise TensorIndexError(
                    "COO coordinates must be lexicographically ordered"
                )
        return

    parent_positions = 1
    for mode, (level_type, arrays, extent) in enumerate(
        zip(level_types, mode_indices, layout.physical_shape)
    ):
        if level_type == LevelType.DENSE:
            parent_positions *= extent
            continue
        if level_type == LevelType.COMPRESSED:
            positions, coordinates = arrays
            expected_positions = parent_positions + 1
            if positions.numel() != expected_positions:
                raise TensorIndexError(
                    f"compressed mode {mode} position array has {positions.numel()} "
                    f"elements, expected {expected_positions}"
                )
            if positions.numel() == 0 or int(positions[0].item()) != 0:
                raise TensorIndexError(
                    f"compressed mode {mode} position array must start at zero"
                )
            if positions.numel() > 1 and bool(
                torch.any(positions[1:] < positions[:-1]).item()
            ):
                raise TensorIndexError(
                    f"compressed mode {mode} position array must be nondecreasing"
                )
            terminal = int(positions[-1].item())
            if terminal != coordinates.numel():
                raise TensorIndexError(
                    f"compressed mode {mode} terminal position {terminal} does not "
                    f"match coordinate count {coordinates.numel()}"
                )
            _check_coordinate_bounds(coordinates, extent, f"compressed mode {mode}")
            _check_sorted_within_parents(positions, coordinates, mode)
            parent_positions = coordinates.numel()
            continue
        coordinate = arrays[0]
        _check_coordinate_bounds(coordinate, extent, f"mode {mode}")
        if level_type == LevelType.SINGLETON and coordinate.numel() != parent_positions:
            raise TensorIndexError(
                f"singleton mode {mode} must contain one coordinate per parent"
            )
        if level_type == LevelType.COORDINATE and coordinate.numel() != nnz:
            raise TensorStorageError(
                f"coordinate mode {mode} has {coordinate.numel()} entries, "
                f"but values has {nnz} elements"
            )
        parent_positions = coordinate.numel()

    if parent_positions != nnz:
        raise TensorStorageError(
            f"layout describes {parent_positions} stored positions, but values has {nnz} elements"
        )


@dataclass(frozen=True, init=False, eq=False)
class SparseStorage:
    """Frozen sparse/dense payload storage tied to one canonical layout."""

    _layout: TensorLayout
    _mode_indices: IndexModes
    _value: torch.Tensor

    def __init__(
        self,
        layout: TensorLayout,
        value: torch.Tensor,
        mode_indices: Optional[Sequence[Sequence[torch.Tensor]]] = None,
        index: Optional[TensorIndex] = None,
    ) -> None:
        if not isinstance(layout, TensorLayout):
            raise TensorTypeError("SparseStorage layout must be a TensorLayout")
        if not isinstance(value, torch.Tensor):
            raise TensorTypeError("SparseStorage value must be a torch.Tensor")
        if value.layout != torch.strided:
            raise TensorStorageError("values must use strided storage")
        if value.device.type != "cpu":
            raise TensorDeviceError(f"Scorch values must be on CPU, got {value.device}")
        if value.dim() != 1:
            raise TensorStorageError("values must be a flat one-dimensional tensor")
        if not value.is_contiguous():
            raise TensorStorageError("values must be contiguous")
        if value.is_neg() or value.is_conj():
            raise TensorStorageError(
                "values must resolve lazy negative and conjugate view bits"
            )
        validate_runtime_contract(layout.format, value.dtype)
        if (mode_indices is None) == (index is None):
            raise TensorStorageError(
                "provide exactly one of mode_indices or a TensorIndex"
            )
        if index is not None:
            if not isinstance(index, TensorIndex):
                raise TensorTypeError("index must be a TensorIndex")
            if index.format != layout.format:
                raise TensorIndexError("index format does not match storage layout")
            if tuple(index.mode_order) != layout.permutation:
                raise TensorIndexError("index mode_order does not match storage layout")
            if index.index_dtype != layout.index_dtype:
                raise TensorIndexError("index dtype does not match storage layout")
            normalized = index._mode_indices
        else:
            normalized = _normalize_mode_indices(mode_indices, layout.format)  # type: ignore[arg-type]
        # Keep tensor metadata independent from the caller while retaining normal
        # tensor payload aliasing semantics for element updates.
        owned_value = value.detach()
        _validate_index_storage(layout, normalized, owned_value)
        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "_mode_indices", normalized)
        object.__setattr__(self, "_value", owned_value)

    @property
    def layout(self) -> TensorLayout:
        return self._layout

    @property
    def has_index(self) -> bool:
        return True

    @property
    def value(self) -> torch.Tensor:
        return self._value.detach()

    @property
    def values(self) -> torch.Tensor:
        return self._value.detach()

    @property
    def index(self) -> TensorIndex:
        return TensorIndex._from_layout(self._layout, self._mode_indices)

    @property
    def _index(self) -> TensorIndex:
        """Compatibility view used by older generated-kernel tests."""
        return self.index

    @property
    def mode_indices(self) -> List[List[torch.Tensor]]:
        return [[index.clone() for index in arrays] for arrays in self._mode_indices]

    def _native_mode_indices(self) -> List[List[torch.Tensor]]:
        """Return internal arrays for trusted Scorch-to-native calls only."""
        return [list(arrays) for arrays in self._mode_indices]

    def validate(self) -> None:
        _validate_index_storage(self._layout, self._mode_indices, self._value)

    def copy(self) -> "SparseStorage":
        indices = tuple(
            tuple(index.clone().detach() for index in arrays)
            for arrays in self._mode_indices
        )
        return SparseStorage(
            self._layout,
            self._value.clone(),
            mode_indices=indices,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SparseStorage)
            and self._layout == other._layout
            and _tensor_value_equal(self._value, other._value)
            and all(
                len(left) == len(right)
                and all(_tensor_value_equal(a, b) for a, b in zip(left, right))
                for left, right in zip(self._mode_indices, other._mode_indices)
            )
        )

    __hash__: Any = None

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "layout": self._layout.to_dict(),
            "values": {
                "dtype": str(self._value.dtype).removeprefix("torch."),
                "data": self._value.tolist(),
            },
            "mode_indices": [
                [
                    {
                        "dtype": str(index.dtype).removeprefix("torch."),
                        "data": index.tolist(),
                    }
                    for index in arrays
                ]
                for arrays in self._mode_indices
            ],
        }

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SparseStorage":
        if not isinstance(data, Mapping):
            raise TensorTypeError("serialized storage must be a mapping")
        try:
            layout = TensorLayout.from_dict(data["layout"])
            values_record = data["values"]
            index_records = data["mode_indices"]
            value_dtype = getattr(torch, values_record["dtype"])
            values = torch.tensor(values_record["data"], dtype=value_dtype)
            mode_indices = []
            for arrays in index_records:
                mode_indices.append(
                    [
                        torch.tensor(
                            record["data"], dtype=getattr(torch, record["dtype"])
                        )
                        for record in arrays
                    ]
                )
        except (
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            OverflowError,
            AttributeError,
        ) as error:
            raise TensorStorageError("serialized storage is malformed") from error
        return cls(layout, values, mode_indices=mode_indices)

    def __str__(self) -> str:
        return f"SparseStorage(shape={self.layout.physical_shape}, format={self.layout.format})"

    __repr__ = __str__


class TensorStorage(SparseStorage):
    """Compatibility constructor that still produces validated SparseStorage.

    New code should pass an explicit ``layout`` to :class:`SparseStorage`. The
    legacy ``index/value/shape`` spelling remains accepted only when all three
    components are present, so it can no longer create partial storage.
    """

    def __init__(
        self,
        index: Optional[TensorIndex] = None,
        value: Optional[torch.Tensor] = None,
        shape: Optional[Sequence[int]] = None,
        *,
        layout: Optional[TensorLayout] = None,
        mode_indices: Optional[Sequence[Sequence[torch.Tensor]]] = None,
    ) -> None:
        if layout is None:
            missing = [
                field
                for field, item in (
                    ("index", index),
                    ("value", value),
                    ("shape", shape),
                )
                if item is None
            ]
            if missing:
                raise TensorStorageError(
                    "TensorStorage requires a layout or complete index/value/shape; "
                    f"missing {', '.join(missing)}"
                )
            if not isinstance(index, TensorIndex):
                raise TensorTypeError("index must be a TensorIndex")
            layout = TensorLayout.from_physical_shape(
                shape, index.format, index.mode_order, index.index_dtype  # type: ignore[arg-type]
            )
        elif shape is not None:
            if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
                raise TensorTypeError("TensorStorage shape must be a sequence")
            if tuple(shape) != layout.physical_shape:
                raise TensorLayoutError(
                    f"TensorStorage shape {tuple(shape)} does not match layout "
                    f"physical shape {layout.physical_shape}"
                )
        if value is None:
            raise TensorStorageError("TensorStorage value is required")
        super().__init__(layout, value, mode_indices=mode_indices, index=index)


class TensorStorageView(SparseStorage):
    """Reserved validated secondary-index storage type."""
