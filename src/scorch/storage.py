"""Immutable sparse index and payload storage value objects."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
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

# Resolved once. The extension is built from this same tree, but an editable install
# whose extension has not been rebuilt will not have the entry point, and validation
# must keep working there -- so its absence is a missing optimization, not an error.
try:  # pragma: no cover - exercised by whichever branch this import takes
    import scorch_ops as _native_ops

    _NATIVE_SCREEN = getattr(_native_ops, "abi_screen_compressed_level", None)
    _NATIVE_BOUNDS = getattr(_native_ops, "abi_screen_bounds_level", None)
    _NATIVE_LEX = getattr(_native_ops, "abi_screen_lex_levels", None)
except Exception:
    _NATIVE_SCREEN = None
    _NATIVE_BOUNDS = None
    _NATIVE_LEX = None

# How long a scan has to be before a screen splits it, for the screens *this* module
# calls. The native call boundary keeps its own, higher value (SCORCH_ABI_VALIDATE_GRAIN,
# a million) because a memo there means each array is scanned about once, so the scan is
# not hot; here every wrap scans fresh arrays.
#
# The screens split with at::parallel_reduce, which opens a region at torch's full thread
# width and then hands work to min(team, ceil(nnz / grain)) of those threads. So this
# value does two things: below it the scan stays serial, and above it it bounds how many
# workers share the array. Opening the region is the fixed cost -- ~3-5 us at four torch
# threads, ~10-13 us at thirty-two -- and it is why the value cannot be small.
#
# 65536 cannot regress anything, and that is structural rather than lucky: below the
# threshold no split happens, so the code is the same code it was. Measured over 9 cells
# from 8k to 1.6M nonzeros x {4, 32} torch threads on redwood and {4, 8} on an M5, the five
# sub-threshold cells span 0.2-3.8%, which is this measurement's noise floor rather than an
# effect. The four cells that do split are all wins: 1.45-1.59x at 480k, 1.07-2.92x at 98k
# COO, 2.95-8.54x at 1M COO.
#
# What fixes the value from below is that a *lower* threshold does regress, because there
# the code genuinely differs: 16384 costs 2.5-3.6x at 20k nonzeros and 4096 costs 3.0-5.7x
# at 8k, where the scan is a few microseconds and the region is 3-13 us.
#
# SCORCH_WRAP_VALIDATE_GRAIN overrides it, which is what lets the sweep run every value
# against one binary.
_WRAP_GRAIN = int(os.environ.get("SCORCH_WRAP_VALIDATE_GRAIN", 65536))
# Whether the O(nnz) structural walk runs on index arrays *Scorch's own compiler just
# emitted*. Off in release, on under test.
#
# This is PyTorch's design for the same problem, deliberately:
# `torch.sparse.check_sparse_tensor_invariants` exists, defaults to disabled, and warns
# that disabled means "memory errors (e.g. SEGFAULT) will occur when operating on a
# sparse tensor which violates the invariants". The reasoning transfers exactly. Our
# kernels take `data_ptr<int>()` and do unchecked pointer arithmetic, so a coordinate past
# the extent is an out-of-bounds access in C++, not an exception -- which is why arrays a
# *caller* supplies are always walked, no flag involved.
#
# A generated result is different in kind. Its index arrays were produced microseconds
# earlier by our own codegen, which allocates every output level with `torch::empty` sized
# from a counted extent and fills it; validating them re-derives a fact the compiler
# already established. Measured, that re-derivation is 35-41% of a wrap.
#
# What the flag buys is that the fact stays checked where a mistake would be found. The
# risk of trusting the compiler is a bug in lowering or a new codegen path producing a
# malformed index that reaches a kernel as raw pointers, and the LoopIR migration is
# actively changing that layer -- so `tests/conftest.py` turns this ON for the whole suite.
# A compiler bug is then a structured error in CI rather than a segfault in someone's run.
#
# Mutable cell rather than a plain bool so a test or a benchmark can flip it in one
# process; `SCORCH_VALIDATE_KERNEL_RESULTS=1` sets it at import.
_VALIDATE_KERNEL_RESULTS = [
    os.environ.get("SCORCH_VALIDATE_KERNEL_RESULTS", "0") not in ("0", "false", "False")
]
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
    mode_indices: Sequence[Sequence[torch.Tensor]],
    tensor_format: TensorFormat,
    copy: bool = True,
) -> IndexModes:
    """Validate one index array per level and return them as nested tuples.

    ``copy=False`` adopts the caller's arrays instead of copying them, and is only for
    arrays this process just produced and holds the sole reference to -- a kernel's
    freshly allocated output. Every structural check still runs; the only thing that
    changes is who owns the buffer. See ``TensorIndex.__init__``'s ``_adopt``.
    """
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
            # caller mutation cannot invalidate a previously validated object -- unless
            # the caller has handed over ownership outright (see `copy`).
            normalized_arrays.append(
                index.detach() if not copy else index.detach().clone()
            )
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
        *,
        _adopt: bool = False,
    ) -> None:
        """Describe a format and its index arrays, copying the arrays.

        ``_adopt`` is internal and takes the arrays as they are instead of copying
        them. It is for one situation: arrays a kernel just allocated for its own
        result, which nothing else references. Every validation below still runs.
        Passing it for arrays a caller might still mutate, or that another tensor
        shares, breaks this class's immutability -- which is why it is keyword-only,
        underscored, and used in exactly one helper (`_wrap_generated_result`).
        """
        if tensor_format is None:
            raise TensorIndexError(
                "tensor_format is required; use TensorFormat() for a scalar"
            )
        tensor_format = parse_format(tensor_format)
        normalized = _normalize_mode_indices(mode_indices, tensor_format, not _adopt)
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
    # One native pass instead of a min reduction, a max reduction and two syncs. As
    # everywhere here, only "no violation" is a claim: when the screen declines, the
    # reductions run and report the range they found, so the message is unchanged.
    if _screen_bounds(coordinate, extent) is False:
        return
    minimum = int(coordinate.min().item())
    maximum = int(coordinate.max().item())
    if minimum < 0 or maximum >= extent:
        raise TensorIndexError(
            f"{label} contains coordinate range [{minimum}, {maximum}] outside [0, {extent})"
        )


def _screen_bounds(coordinate: torch.Tensor, extent: int) -> bool:
    """``False`` means every coordinate is in ``[0, extent)``. See ``_screen_...`` below."""
    screen = _NATIVE_BOUNDS
    if screen is None:
        return True
    try:
        return bool(screen(coordinate, int(extent), _WRAP_GRAIN))
    except Exception:
        return True


def _screen_lex_levels(levels: Sequence[torch.Tensor], count: int) -> bool:
    """``False`` means the COO coordinates ascend lexicographically across levels.

    Same one-directional contract as the other screens. When this clears a tensor,
    the Python loop below it -- which materializes every index array with ``tolist``
    and then compares a tuple per nonzero -- is skipped entirely.
    """
    screen = _NATIVE_LEX
    if screen is None:
        return True
    try:
        return bool(screen(list(levels), int(count), _WRAP_GRAIN))
    except Exception:
        return True


def _screen_compressed_level(
    positions: torch.Tensor, coordinates: torch.Tensor, extent: int
) -> bool:
    """Ask the native extension whether this level might be malformed.

    ``True`` means "check it yourself" -- either the screen found something or it
    cannot serve this input, including when the extension is absent entirely. Only
    ``False`` is a claim, and the claim is one-directional: no violation exists. That
    asymmetry is what makes this safe to consult, and it is the screen's documented
    contract in ``csrc/native_abi.h``, where it has been guarding the native call
    boundary for the same checks.

    Failing to ``True`` on any surprise keeps a validator a validator: the worst a
    broken or missing screen can do is make construction as slow as it used to be.
    """
    screen = _NATIVE_SCREEN
    if screen is None:
        return True
    try:
        return bool(screen(positions, coordinates, int(extent), True, _WRAP_GRAIN))
    except Exception:
        return True


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
        coordinates_per_mode = []
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
            coordinates_per_mode.append(coordinate)
        # The Python loop below is the most expensive thing in this file: it
        # materializes every mode's index array with `tolist` and then builds and
        # compares two tuples per NONZERO -- 0.40 us each, so 159 ms to wrap a
        # 400,000-nonzero COO tensor. The native screen makes the same comparison in
        # one thread-split pass, and clears the tensor without allocating anything.
        # Only when it declines does any of that happen, and then it reports.
        if _screen_lex_levels(coordinates_per_mode, nnz) is False:
            return
        coordinate_values = [c.tolist() for c in coordinates_per_mode]
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
            # One native pass stands in for the three whole-array torch checks
            # below -- positions nondecreasing, coordinates in range, coordinates
            # ascending inside each parent. It reports "no violation exists" or "go
            # and look", never "this is fine" about something that is not, so when it
            # clears the level the three checks below cannot fail and are skipped;
            # when it does not, they run exactly as they always did and remain the
            # only thing that reports. Diagnostics, and the order they fire in, are
            # therefore unchanged either way. See csrc/native_abi.h.
            suspect = _screen_compressed_level(positions, coordinates, extent)
            if suspect and (
                positions.numel() > 1
                and bool(torch.any(positions[1:] < positions[:-1]).item())
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
            if suspect:
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


def tensor_version(array: torch.Tensor) -> int:
    """``array._version``, or 0 for a tensor that does not track one.

    Inference-mode tensors have their version counter disabled and ``_version`` *raises* on
    them, so this has to be asked rather than assumed. The native validator already does
    exactly this -- ``abi_version_of`` in ``csrc/native_abi.h`` -- and its comment states the
    consequence of not doing it: a ``with torch.inference_mode():`` block turns every matmul
    into an exception. This is the Python half of the same guard, added after the ABI guard
    suite caught the Python half missing while the native half passed.

    Collapsing the version to 0 leaves the stamp unable to see an in-place write to an
    inference tensor that preserves both ``data_ptr`` and ``numel``. That is one more blind
    spot beside the ones the stamp already has -- a write through a raw pointer or a numpy
    view bumps no counter either -- and it buys keeping the O(nnz) validation skipped inside
    ``inference_mode``, which is where an inference workload spends all of its time. Failing
    closed to always-revalidate there would put the cost back on the hot path it was removed
    from, for a mutation pattern nothing else in Scorch catches either.
    """
    return 0 if array.is_inference() else array._version


def _validation_stamp(
    mode_indices: IndexModes, value: torch.Tensor
) -> Tuple[Tuple[Tuple[int, int, int], ...], Tuple[int, int, int]]:
    """A cheap summary of everything ``_validate_index_storage`` reads.

    Equal stamps mean the validator would reach the same verdict, so the answer can
    be reused instead of recomputed -- and recomputing is O(nnz), which is why it is
    worth summarizing at all.

    Identity is not sufficient on its own: a tensor's contents can change in place
    under a stable ``id``, and a freed tensor's address can be handed to a later one.
    ``data_ptr`` pins the buffer, ``_version`` moves on every in-place torch
    operation against it, and ``numel`` catches a ``resize_``. This is the same
    triple, for the same reason, that a cached call plan records (see plan.py).

    It does not see a write through a raw pointer or a numpy view, which bumps no
    version counter. Neither does anything else in Scorch: validation happens when a
    tensor is built or reassembled, and such a write happens in between without
    telling anyone. The stamp is therefore exactly as strong as the check it skips.
    """
    return (
        tuple(
            (array.data_ptr(), tensor_version(array), array.numel())
            for arrays in mode_indices
            for array in arrays
        ),
        (value.data_ptr(), tensor_version(value), value.numel()),
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
        *,
        _trusted_index: bool = False,
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
        # The cheap per-array checks -- dtype, rank, contiguity, device, one array per
        # level -- have already run inside `_normalize_mode_indices` for every caller,
        # trusted or not. They are O(1) each and adopting a kernel's arrays depends on
        # them. What `_trusted_index` skips is only the O(nnz) structural walk.
        if not _trusted_index or _VALIDATE_KERNEL_RESULTS[0]:
            _validate_index_storage(layout, normalized, owned_value)
        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "_mode_indices", normalized)
        object.__setattr__(self, "_value", owned_value)
        # What was just validated, so that `STensor._set_state` -- which every
        # constructor and every in-place structural change funnels through, right
        # after this one -- does not walk the index arrays a second time. For a trusted
        # index with the flag off the stamp records "these arrays are not to be walked"
        # rather than "these arrays were walked and passed"; both mean the same thing to
        # every reader of it, and an explicit `.validate()` still forces the walk.
        object.__setattr__(self, "_checked", _validation_stamp(normalized, owned_value))

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
        """Re-run every structural invariant check, unconditionally."""
        _validate_index_storage(self._layout, self._mode_indices, self._value)
        object.__setattr__(
            self, "_checked", _validation_stamp(self._mode_indices, self._value)
        )

    def validate_unless_already_checked(self) -> None:
        """Validate, unless these exact arrays have already passed unmutated.

        For the internal path: a storage reaches ``STensor._set_state`` immediately
        after ``SparseStorage.__init__`` validated it, and repeating an O(nnz) walk
        over index arrays nothing has touched since is the single largest cost of
        wrapping a large matrix (74 ms of the 78 ms it took to wrap a 20k-row CSR,
        half of it this second walk). :func:`_validation_stamp` says whether the
        verdict could possibly have changed; where it could, the full check runs.
        """
        current = _validation_stamp(self._mode_indices, self._value)
        if getattr(self, "_checked", None) == current:
            return
        _validate_index_storage(self._layout, self._mode_indices, self._value)
        object.__setattr__(self, "_checked", current)

    def __getstate__(self) -> dict:
        """The state to pickle or copy: everything except the validation stamp.

        A stamp records buffer addresses and version counters in *this* process, so it
        means nothing anywhere else. Dropping it makes an unpickled or copied storage
        validate itself once on first use, which is the same thing that happens to
        storage assembled any other way.
        """
        state = dict(self.__dict__)
        state.pop("_checked", None)
        return state

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
