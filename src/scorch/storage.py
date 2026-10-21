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
from collections.abc import Container, Iterator, Mapping, Sequence
from typing import Any, List, Optional, Tuple

import torch

from .exceptions import (
    COALESCE_ADVICE,
    RepeatedCoordinateError,
    ScorchAssertionError,
    TensorDeviceError,
    TensorIndexError,
    TensorLayoutError,
    TensorStorageError,
    TensorTypeError,
    format_coordinate,
)
from .format import (
    FormatInput,
    LevelType,
    TensorFormat,
    owned_format,
    parse_format,
)
from .layout import (
    TensorLayout,
    _normalize_shape,
    owned_layout,
    validate_runtime_contract,
)

IndexModes = Tuple[Tuple[torch.Tensor, ...], ...]
# What ``_index_components`` returns: a format, the index arrays audited against it, the
# physical-to-logical mode order and the index dtype.
AuditedIndex = Tuple[TensorFormat, IndexModes, Tuple[int, ...], torch.dtype]

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
    raise TensorIndexError(
        f"unsupported level type {level_type}", code="level_type_unknown"
    )


def _normalize_mode_indices(
    mode_indices: Sequence[Sequence[torch.Tensor]],
    tensor_format: TensorFormat,
    copy: bool = True,
) -> IndexModes:
    """Validate one index array per level and return them as nested tuples.

    ``copy=False`` adopts the caller's arrays instead of copying them. Every structural
    check still runs; the only thing that changes is who owns the buffer. See
    ``TensorIndex.__init__``'s ``_adopt`` for when that is allowed.
    """
    if isinstance(mode_indices, (str, bytes)) or not isinstance(mode_indices, Sequence):
        raise TensorTypeError(
            "mode_indices must be a sequence with one entry per mode",
            code="mode_indices_type_invalid",
        )
    if len(mode_indices) != tensor_format.get_order():
        raise TensorIndexError(
            f"mode_indices has rank {len(mode_indices)}, expected "
            f"{tensor_format.get_order()} for format {tensor_format}",
            code="mode_indices_rank_mismatch",
        )

    normalized = []
    for mode, (arrays, level_type) in enumerate(
        zip(mode_indices, tensor_format.level_types)
    ):
        if isinstance(arrays, (str, bytes)) or not isinstance(arrays, Sequence):
            raise TensorTypeError(
                f"mode_indices[{mode}] must be a sequence",
                code="mode_indices_type_invalid",
            )
        expected = _expected_level_arity(level_type)
        if len(arrays) != expected:
            raise TensorIndexError(
                f"mode {mode} ({level_type.name.lower()}) requires {expected} "
                f"index arrays, got {len(arrays)}",
                code="mode_indices_arity_mismatch",
            )
        normalized_arrays = []
        for slot, index in enumerate(arrays):
            index_type = type(index)
            if index_type is torch.Tensor:
                # An exact torch.Tensor has no subclass whose overrides a read could
                # reach, so it needs no base-class view of itself: making one is a new
                # tensor object per array per audit. The reads below that are methods
                # go through the type, so an attribute set on the instance cannot
                # answer for them, and the properties are data descriptors, which an
                # instance cannot shadow. What is retained is the detached tensor made
                # at the end of this loop body, never the caller's object.
                base = index
            else:
                try:
                    mro = type.__getattribute__(index_type, "__mro__")
                except Exception as error:
                    raise TensorTypeError(
                        f"mode_indices[{mode}][{slot}] must be a torch.Tensor",
                        code="mode_indices_type_invalid",
                    ) from error
                if not any(base is torch.Tensor for base in mro):
                    raise TensorTypeError(
                        f"mode_indices[{mode}][{slot}] must be a torch.Tensor",
                        code="mode_indices_type_invalid",
                    )
                try:
                    base = torch.Tensor.as_subclass(index, torch.Tensor)
                except Exception as error:
                    raise TensorIndexError(
                        f"mode_indices[{mode}][{slot}] is malformed",
                        code="index_array_malformed",
                    ) from error
                if type(base) is not torch.Tensor:
                    raise TensorIndexError(
                        f"mode_indices[{mode}][{slot}] is malformed",
                        code="index_array_malformed",
                    )
            if base.layout != torch.strided:
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] must use strided storage",
                    code="index_array_layout_invalid",
                )
            if base.device.type != "cpu":
                raise TensorDeviceError(
                    f"mode_indices[{mode}][{slot}] must be on CPU, got {base.device}",
                    code="device_not_cpu_unsupported",
                )
            if base.dtype not in _INDEX_DTYPES:
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] must use int32 or int64, "
                    f"got {base.dtype}",
                    code="index_dtype_invalid",
                )
            if torch.Tensor.dim(base) != 1:
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] must be one-dimensional",
                    code="index_array_layout_invalid",
                )
            if not torch.Tensor.is_contiguous(base):
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] must be contiguous",
                    code="index_array_layout_invalid",
                )
            # Index coordinates are structural data. Own an independent copy so
            # caller mutation cannot invalidate a previously validated object -- unless
            # the arrays are being adopted (see `copy`). A malformed tensor fails closed.
            try:
                detached = torch.Tensor.detach(base)
                owned = torch.Tensor.clone(detached) if copy else detached
            except Exception as error:
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] is malformed",
                    code="index_array_malformed",
                ) from error
            if type(owned) is not torch.Tensor:
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] is malformed",
                    code="index_array_malformed",
                )
            normalized_arrays.append(owned)
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


def _index_components(
    tensor_format: FormatInput,
    mode_indices: Sequence[Sequence[torch.Tensor]],
    mode_order: Optional[Sequence[int]],
    index_dtype: Optional[torch.dtype],
    copy: bool,
) -> AuditedIndex:
    """Check a format with its index arrays; return the four values an index holds.

    These are the checks ``TensorIndex.__init__`` makes, in its order and with its
    messages, as a function: the format, the audited arrays (copied, or the caller's
    own when ``copy`` is false), the physical-to-logical order and the index dtype.

    It is a function so that code inside Scorch that is about to build a storage
    (``STensor._from_parts``, ``stensor._wrap_generated_result``) can make these checks
    once and hand the arrays over. Going through a ``TensorIndex`` instead audits
    twice: ``SparseStorage`` cannot know that an index object it is handed was made a
    moment ago by its own caller, so it audits the object's stored state and every
    array again, and it copies the arrays of an index that did not adopt them a second
    time. That audit is right for an index a user kept; for one that never left the
    constructing expression it found nothing, at 75 Python calls of the 249 in a
    ``STensor.from_csr``.
    """
    if tensor_format is None:
        raise TensorIndexError(
            "tensor_format is required; use TensorFormat() for a scalar",
            code="tensor_format_missing",
        )
    # Direct index construction retains the format through
    # ``format.owned_format``: an exact format is kept as it is (it cannot
    # change once built), a subclass is normalized, and a format that was
    # never constructed fails closed. ``TensorIndex._from_layout`` needs neither: a
    # ``TensorLayout`` already owns the format it hands over.
    tensor_format = owned_format(parse_format(tensor_format))
    normalized = _normalize_mode_indices(mode_indices, tensor_format, copy)
    if mode_order is None:
        order = tuple(range(tensor_format.get_order()))
    else:
        if isinstance(mode_order, (str, bytes)) or not isinstance(mode_order, Sequence):
            raise TensorTypeError(
                "mode_order must be a sequence of integers",
                code="mode_order_type_invalid",
            )
        supplied_order = tuple(mode_order)
        if any(
            type(mode) is bool or not issubclass(type(mode), int)
            for mode in supplied_order
        ):
            raise TensorTypeError(
                "mode_order entries must be integers", code="mode_order_type_invalid"
            )
        order = tuple(int.__int__(mode) for mode in supplied_order)
    if len(order) != tensor_format.get_order() or sorted(order) != list(
        range(tensor_format.get_order())
    ):
        raise TensorIndexError(
            f"mode_order must be a permutation of range({tensor_format.get_order()})",
            code="mode_order_not_permutation",
        )

    observed_dtypes = {index.dtype for arrays in normalized for index in arrays}
    if len(observed_dtypes) > 1:
        raise TensorIndexError(
            "all sparse index arrays must use one common index dtype",
            code="index_dtype_mixed",
        )
    observed = next(iter(observed_dtypes), None)
    declared = (
        torch.int32
        if index_dtype is None and observed is None
        else (observed if index_dtype is None else index_dtype)
    )
    if type(declared) is not torch.dtype:
        raise TensorIndexError(
            "index_dtype must be torch.int32 or torch.int64, got "
            f"{type(declared).__name__}",
            code="index_dtype_invalid",
        )
    if declared not in _INDEX_DTYPES:
        raise TensorIndexError(
            f"index_dtype must be torch.int32 or torch.int64, got {declared}",
            code="index_dtype_invalid",
        )
    if observed is not None and observed != declared:
        raise TensorIndexError(
            f"declared index dtype {declared} does not match arrays of dtype {observed}",
            code="index_dtype_mismatch",
        )
    return tensor_format, normalized, order, declared


@dataclass(frozen=True, init=False, eq=False)
class TensorIndex:
    """Immutable compatibility descriptor for format and sparse index arrays.

    Shape-dependent checks are completed when the descriptor is combined with a
    :class:`TensorLayout` in :class:`SparseStorage`. Caller containers and index
    tensors are copied, so later caller mutation cannot invalidate the descriptor;
    they are never silently cast or replaced in the caller. The exception is an index
    built to adopt its arrays (see ``_adopt``), the way the conversion of a torch
    sparse tensor (``STensor.from_torch`` / ``from_csr`` / ``from_coo``) shares that
    tensor's arrays as torch itself does.

    Whether the arrays were adopted is recorded (``_adopted``, a slot, so the instance
    ``__dict__`` keeps exactly the four declared fields that every ownership audit checks)
    and set only by that constructor argument. A ``SparseStorage`` built from an index
    shares the arrays of an adopted one, because sharing is the point of adopting them, and
    copies those of any other, so that two storages built from one index never share arrays.
    """

    __slots__ = ("_adopted", "__dict__", "__weakref__")

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
        them. Every validation below still runs. Scorch's own constructions make the
        same choice without an index object, as ``_index_components(..., copy=False)``
        (``STensor._from_parts(..., adopt=True)``). It is for two situations:

        * Arrays a kernel just allocated for its own result, which nothing else
          references (`_wrap_generated_result`).
        * The arrays of a torch sparse tensor being converted (`STensor.from_torch`,
          `from_csr`, `from_coo`). torch does not copy them either:
          ``torch.sparse_csr_tensor(crow, col, values)`` keeps the caller's own
          ``crow`` and ``col``, and a later write to them shows up in the tensor. So
          the STensor shares them with the torch tensor, and a write the caller makes
          through torch reaches both -- which is what the caller of a torch tensor
          already has. Copying them cost two O(nnz) copies per conversion and, because
          the copies were new arrays every time, defeated every verdict and narrowed
          copy the native call boundary keeps per array (`native_abi.h`), so each call
          also re-walked and re-narrowed them: a drop-in caller's matmul ran 2-4x slower
          than MKL's on mkt1 although the kernel was faster.

        Safety does not rest on nobody writing: the native call boundary re-checks
        every index array per call, keyed on the array's identity and version counter,
        so a write made through torch is re-validated at the next kernel call, and so
        is a cached plan's (plan.h). What sharing gives up is what torch gives up: a
        write through a raw pointer or a numpy view bumps no version counter. Nothing
        in Scorch writes into an index array it was given.
        """
        tensor_format, normalized, order, declared = _index_components(
            tensor_format, mode_indices, mode_order, index_dtype, not _adopt
        )
        object.__setattr__(self, "_format", tensor_format)
        object.__setattr__(self, "_mode_indices", normalized)
        object.__setattr__(self, "_mode_order", order)
        object.__setattr__(self, "_index_dtype", declared)
        object.__setattr__(self, "_adopted", _adopt is True)

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
            raise TensorIndexError(
                f"mode {mode} is outside tensor rank", code="mode_out_of_range"
            )
        try:
            return self._format.get_level_types()[mode]
        except IndexError as error:
            raise TensorIndexError(
                f"mode {mode} is outside tensor rank", code="mode_out_of_range"
            ) from error

    def get_level_types(self) -> List[LevelType]:
        return self._format.get_level_types()

    def get_size(self, mode: int) -> int:
        arrays = self.get_mode_index(mode)
        if not arrays:
            raise TensorIndexError(f"dense mode {mode} has no explicit index size")
        return arrays[0].numel()

    def get_sizes(self) -> List[int]:
        return [self.get_size(mode) for mode in range(self.get_order())]


_ADOPTED_SLOT = TensorIndex.__dict__["_adopted"]


def _index_adopted(index: TensorIndex) -> bool:
    """Whether ``index`` was built to adopt its arrays. Read through the slot descriptor so
    a subclass cannot answer for it; an index that never recorded it (``_from_layout``, a
    view of a storage's own arrays) did not adopt them. Anything that is not a TensorIndex
    (the descriptor raises TypeError) did not either, and is left to
    ``_owned_tensor_index_components``, which rejects it with a TensorTypeError."""
    try:
        return _ADOPTED_SLOT.__get__(index, TensorIndex) is True
    except (AttributeError, TypeError):
        return False


def _owned_tensor_index_components(
    index: object,
    *,
    own_arrays: bool = True,
) -> Tuple[TensorFormat, IndexModes, Tuple[int, ...], torch.dtype]:
    """Audit and detach a retained ``TensorIndex`` without virtual reads."""

    index_type = type(index)
    try:
        mro = type.__getattribute__(index_type, "__mro__")
    except Exception as error:
        raise TensorTypeError(
            "index must be a TensorIndex", code="index_type_invalid"
        ) from error
    if not any(base is TensorIndex for base in mro):
        raise TensorTypeError("index must be a TensorIndex", code="index_type_invalid")
    try:
        state = TensorIndex.__dict__["__dict__"].__get__(index, TensorIndex)
    except Exception as error:
        raise TensorIndexError(
            "index has malformed stored state", code="value_object_state_corrupt"
        ) from error
    expected = {"_format", "_mode_indices", "_mode_order", "_index_dtype"}
    if (
        type(state) is not dict
        or any(type(key) is not str for key in state)
        or set(state) != expected
        or type(state["_mode_indices"]) is not tuple
        or any(type(arrays) is not tuple for arrays in state["_mode_indices"])
        or type(state["_mode_order"]) is not tuple
    ):
        raise TensorIndexError(
            "index has malformed stored state", code="value_object_state_corrupt"
        )

    tensor_format = owned_format(state["_format"])
    supplied_order = state["_mode_order"]
    if any(
        type(mode) is bool or not issubclass(type(mode), int) for mode in supplied_order
    ):
        raise TensorIndexError(
            "index has malformed stored state", code="value_object_state_corrupt"
        )
    order = tuple(int.__int__(mode) for mode in supplied_order)
    index_dtype = state["_index_dtype"]
    if type(index_dtype) is not torch.dtype:
        raise TensorIndexError(
            "index has malformed stored state", code="value_object_state_corrupt"
        )
    normalized = (
        _normalize_mode_indices(state["_mode_indices"], tensor_format)
        if own_arrays
        else state["_mode_indices"]
    )
    return tensor_format, normalized, order, index_dtype


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
            f"{label} contains coordinate range [{minimum}, {maximum}] outside [0, {extent})",
            code="coordinate_out_of_bounds",
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
    """``False`` means the COO coordinates ascend strictly lexicographically across
    levels: no tuple is smaller than, or equal to, the one before it.

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


def _first_coo_order_defect(
    levels: Sequence[torch.Tensor], count: int
) -> Optional[Tuple[int, bool]]:
    """The first position whose coordinate tuple is not strictly greater than the one
    before it, with True when it is equal (one coordinate stored twice) and False when it
    is smaller; None when the tuples strictly ascend.

    The Python loop below is the most expensive thing in this file: it materializes every
    level's index array with `tolist` and then builds and compares two tuples per NONZERO
    -- 0.40 us each, so 159 ms over a 400,000-nonzero COO tensor. The native screen makes
    the same comparison in one thread-split pass and clears the tensor without allocating
    anything. Only when it declines does any of that happen, and then it reports.
    """
    if _screen_lex_levels(levels, count) is False:
        return None
    coordinate_values = [level.tolist() for level in levels]
    for position in range(1, count):
        previous = tuple(values[position - 1] for values in coordinate_values)
        current = tuple(values[position] for values in coordinate_values)
        if current <= previous:
            return position, current == previous
    return None


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


def _first_unsorted_parent(
    positions: torch.Tensor, coordinates: torch.Tensor
) -> Optional[int]:
    """The first parent whose coordinates are not strictly increasing, or None.

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

    It returns the offending parent so that the error can name it. That costs two more
    kernels and only runs when the storage is actually malformed.
    """
    count = coordinates.numel()
    if count < 2:
        return None
    # Strict: a repeated coordinate within a parent is as invalid as a descent (the
    # compiler's co-iteration and torch's own CSR invariant both assume distinct ones).
    descends = coordinates[1:] <= coordinates[:-1]
    # A descent at index i compares coordinates[i+1] against coordinates[i], so a
    # parent starting at position p makes index p-1 a boundary. Interior entries of
    # `positions` outside [1, count-1] describe empty parents at either end, which have
    # no comparison to exempt.
    starts = positions[1:-1].to(torch.long)
    starts = starts[(starts >= 1) & (starts <= count - 1)]
    if starts.numel():
        descends[starts - 1] = False
    if not bool(descends.any()):
        return None
    # Parents are contiguous and ordered, so the first offending position lies in the
    # first offending parent -- which is the one the row loop reported.
    first = int(torch.nonzero(descends)[0].item())
    boundaries = positions.to(torch.long)
    return int(torch.searchsorted(boundaries, first, right=True).item()) - 1


def _validate_index_storage(
    layout: TensorLayout,
    mode_indices: IndexModes,
    values: torch.Tensor,
    *,
    allow_unordered: bool = False,
    repair_coo: bool = False,
) -> bool:
    """Raise unless the index arrays describe well-formed storage for ``values``.

    ``allow_unordered`` is for a caller's arrays (see ``SparseStorage.__init__``): a
    compressed level whose coordinates are out of order or repeated inside a parent is
    then reported by returning True instead of raised, provided every level is dense or
    compressed and nothing else is wrong. The caller replaces the storage with its
    canonical form (``_canonical_storage``). Every other defect still raises, and so
    does an unordered level in any other format. Returns False for well-formed storage.

    A COO tensor whose coordinates repeat or descend is never repaired on construction:
    a repeat raises ``RepeatedCoordinateError``. ``repair_coo`` is for
    ``STensor.coalesce()`` alone, the call that sums repeats on request: such COO
    coordinates are then reported by returning True instead.
    """
    nnz = values.numel()
    level_types = layout.format.level_types

    for mode, arrays in enumerate(mode_indices):
        for slot, index in enumerate(arrays):
            if index.dtype != layout.index_dtype:
                raise TensorIndexError(
                    f"mode_indices[{mode}][{slot}] has dtype {index.dtype}, "
                    f"expected layout index dtype {layout.index_dtype}",
                    code="index_dtype_mismatch",
                )
            if index.device != values.device:
                raise TensorDeviceError(
                    f"mode_indices[{mode}][{slot}] and values must share a device",
                    code="index_values_device_mismatch",
                )

    if all(level_type == LevelType.DENSE for level_type in level_types):
        expected = math.prod(layout.physical_shape)
        if nnz != expected:
            raise TensorStorageError(
                f"dense values contain {nnz} elements, expected {expected} for shape "
                f"{layout.physical_shape}",
                code="values_count_mismatch",
            )
        return False

    if level_types and all(
        level_type == LevelType.COORDINATE for level_type in level_types
    ):
        coordinates_per_mode = []
        for mode, arrays in enumerate(mode_indices):
            coordinate = arrays[0]
            if coordinate.numel() != nnz:
                raise TensorStorageError(
                    f"COO mode {mode} has {coordinate.numel()} coordinates, "
                    f"but values has {nnz} elements",
                    code="values_count_mismatch",
                )
            _check_coordinate_bounds(
                coordinate, layout.physical_shape[mode], f"COO mode {mode}"
            )
            coordinates_per_mode.append(coordinate)
        defect = _first_coo_order_defect(coordinates_per_mode, nnz)
        if defect is None:
            return False
        if repair_coo:
            return True
        position, repeated = defect
        if not repeated:
            raise TensorIndexError(
                "COO coordinates must be lexicographically ordered",
                code="coo_coordinates_unordered",
            )
        # A COO tensor is never repaired, unlike an unordered compressed level: Scorch
        # refuses a stored repeat and the caller sums it (RepeatedCoordinateError).
        repeat = tuple(int(c[position].item()) for c in coordinates_per_mode)
        raise RepeatedCoordinateError(
            f"COO coordinate {format_coordinate(repeat)} is stored at positions "
            f"{position - 1} and {position}; {COALESCE_ADVICE}",
            position,
            repeat,
        )

    unordered = False
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
                    f"elements, expected {expected_positions}",
                    code="compressed_positions_length_invalid",
                )
            # One native pass stands in for every check below -- the first position
            # is zero, positions nondecreasing, the last position equals the
            # coordinate count, coordinates in range, coordinates ascending inside
            # each parent. It reports "no violation exists" or "go and look", never
            # "this is fine" about something that is not, so when it clears the level
            # the checks below cannot fail and are skipped; when it does not, they run
            # exactly as they always did, in the same order, and remain the only thing
            # that reports. Diagnostics are therefore unchanged either way. Its verdict
            # is kept per array identity and version (csrc/native_abi.h), so a level
            # built again around arrays that already passed is not rescanned.
            suspect = _screen_compressed_level(positions, coordinates, extent)
            if suspect and (positions.numel() == 0 or int(positions[0].item()) != 0):
                raise TensorIndexError(
                    f"compressed mode {mode} position array must start at zero",
                    code="compressed_positions_start_nonzero",
                )
            if suspect and (
                positions.numel() > 1
                and bool(torch.any(positions[1:] < positions[:-1]).item())
            ):
                raise TensorIndexError(
                    f"compressed mode {mode} position array must be nondecreasing",
                    code="compressed_positions_decreasing",
                )
            if suspect:
                terminal = int(positions[-1].item())
                if terminal != coordinates.numel():
                    raise TensorIndexError(
                        f"compressed mode {mode} terminal position {terminal} does "
                        f"not match coordinate count {coordinates.numel()}",
                        code="compressed_positions_terminal_mismatch",
                    )
            if suspect:
                _check_coordinate_bounds(coordinates, extent, f"compressed mode {mode}")
                parent = _first_unsorted_parent(positions, coordinates)
                # An unordered level can be repaired only where the canonical form is
                # defined for every level: dense and compressed ones (_canonical_index).
                # Asked here, so that well-formed storage never pays for the question.
                if parent is not None and not (
                    allow_unordered
                    and all(
                        level in (LevelType.DENSE, LevelType.COMPRESSED)
                        for level in level_types
                    )
                ):
                    raise TensorIndexError(
                        f"compressed mode {mode} coordinates must be strictly "
                        f"increasing within parent {parent}",
                        code="compressed_coordinates_unordered",
                    )
                unordered = unordered or parent is not None
            parent_positions = coordinates.numel()
            continue
        coordinate = arrays[0]
        _check_coordinate_bounds(coordinate, extent, f"mode {mode}")
        if level_type == LevelType.SINGLETON and coordinate.numel() != parent_positions:
            raise TensorIndexError(
                f"singleton mode {mode} must contain one coordinate per parent",
                code="singleton_coordinates_count_mismatch",
            )
        if level_type == LevelType.COORDINATE and coordinate.numel() != nnz:
            raise TensorStorageError(
                f"coordinate mode {mode} has {coordinate.numel()} entries, "
                f"but values has {nnz} elements",
                code="values_count_mismatch",
            )
        parent_positions = coordinate.numel()

    if parent_positions != nnz:
        raise TensorStorageError(
            f"layout describes {parent_positions} stored positions, but values has {nnz} elements",
            code="values_count_mismatch",
        )
    return unordered


def _sorted_distinct(
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Sort entries on their row-major linear index ``key`` and number the distinct keys.

    Returns the distinct keys in ascending order, the sorting permutation, each sorted
    entry's distinct number, and the number of distinct keys. Stable, so entries at one
    coordinate keep their order and are summed in it. ``key`` is not empty.
    """
    key, order = torch.sort(key, stable=True)
    starts = torch.ones(key.numel(), dtype=torch.bool)
    if key.numel() > 1:
        starts[1:] = key[1:] != key[:-1]
    entry = torch.cumsum(starts.to(torch.int64), 0) - 1
    return key[starts], order, entry, int(entry[-1].item()) + 1


def level_positions(
    level_types: Sequence[LevelType],
    mode_indices: IndexModes,
    extents: Sequence[int],
    level: int,
) -> int:
    """How many positions storage level ``level`` holds; level ``-1`` is the root, one.

    A dense level holds its parent's positions times its extent; a compressed,
    coordinate or singleton level one per coordinate it stores."""

    count = 1
    for kind, arrays, extent in zip(level_types[: level + 1], mode_indices, extents):
        count = count * int(extent) if kind == LevelType.DENSE else arrays[-1].numel()
    return count


def _records_its_parent(level_types: Sequence[LevelType], level: int) -> bool:
    """Whether the index arrays say which position of the level above each position of
    storage level ``level`` belongs to.

    A dense level's parent follows from the position (position = parent * extent +
    coordinate) and a compressed level's from its ``pos`` array; a singleton level stores
    one coordinate per position of its parent, so a position's parent is the same
    position.  A coordinate level stores one coordinate per stored value: in the leading
    run of coordinate levels (COO) every level does, so the parent there is the same
    position too (level 0's is the root), but below a dense, compressed or singleton
    level nothing records which parent a coordinate belongs to.  This is the one rule
    for every walk over a storage's levels (:func:`_walk_up`, :func:`positions_beneath`).
    It does not lean on ``validate_runtime_contract`` refusing singleton levels today: a
    coordinate level below a singleton one is refused here for what its arrays hold.
    """

    if level_types[level] != LevelType.COORDINATE:
        return True
    return all(kind == LevelType.COORDINATE for kind in level_types[:level])


def _parents_recorded(level_types: Sequence[LevelType], level: int, top: int) -> bool:
    """Whether a walk from storage level ``level`` up to level ``top`` finds every parent
    it needs: those of levels ``top + 1 .. level``."""

    return all(
        _records_its_parent(level_types, current)
        for current in range(top + 1, level + 1)
    )


def _walk_up(
    level_types: Sequence[LevelType],
    mode_indices: IndexModes,
    extents: Sequence[int],
    level: int,
    top: int,
    wanted: Optional[Container[int]] = None,
    positions: Optional[torch.Tensor] = None,
) -> Iterator[Tuple[int, torch.Tensor]]:
    """The coordinates of positions of storage level ``level`` at each level from
    ``level`` up to ``top``, deepest first: yields ``(current, coordinates)`` for each
    ``current`` in ``wanted`` (every level when ``None``), ``coordinates`` int64 with one
    entry per walked position.  The positions walked are ``positions`` (int64 indices
    into level ``level``), or every position of the level in order when ``None``; then a
    yielded tensor may be the storage's own array, which the caller must not write.

    The one walk up a storage's levels: :func:`_walked_coordinates` (each stored value's
    full coordinate, for ``stored_coordinates`` and ``_canonical_index``) and
    :func:`level_keys` (``result_bound``'s keys) read it.  A dense level's coordinate and parent follow from the
    position arithmetically; a compressed level's coordinate is ``crd[position]`` and its
    parent the segment of ``pos`` holding the position; a singleton or coordinate level's
    coordinate is ``crd[position]`` and its parent the same position one level up.  The
    caller checks :func:`_parents_recorded` first: a walk that needs a parent no array
    records raises ScorchAssertionError (``storage_parent_unrecorded``).
    """

    position = positions  # None: every position of ``current``, in order
    for current in range(level, top - 1, -1):
        kind = level_types[current]
        if kind == LevelType.DENSE:
            extent = int(extents[current])
            if position is None:
                position = torch.arange(
                    level_positions(level_types, mode_indices, extents, current),
                    dtype=torch.int64,
                )
            if wanted is None or current in wanted:
                yield current, position % extent
            if current == top:
                return
            position = torch.div(position, extent, rounding_mode="floor")
            continue
        if wanted is None or current in wanted:
            coordinates = mode_indices[current][-1]
            here = coordinates if position is None else coordinates[position]
            yield current, here.to(torch.int64)
        if current == top:
            return
        if kind == LevelType.COMPRESSED:
            pos = mode_indices[current][0]
            lengths = (pos[1:] - pos[:-1]).to(torch.int64)
            above = level_positions(level_types, mode_indices, extents, current - 1)
            owner = torch.repeat_interleave(
                torch.arange(above, dtype=torch.int64), lengths
            )
            position = owner if position is None else owner[position]
        elif not _records_its_parent(level_types, current):
            raise ScorchAssertionError(
                f"storage level {current} is a coordinate level below a "
                f"{level_types[current - 1].name.lower()} level: no index array records "
                "the parent of its positions, and the caller did not check "
                "(_parents_recorded)",
                code="storage_parent_unrecorded",
            )


def level_keys(
    level_types: Sequence[LevelType],
    mode_indices: IndexModes,
    extents: Sequence[int],
    strides: Mapping[int, int],
    level: int,
) -> Optional[torch.Tensor]:
    """For every position of storage level ``level``, as int64: the sum over the levels
    in ``strides`` (each at most ``level``) of the position's coordinate at that level
    times the level's stride, read by :func:`_walk_up` from ``level`` to the highest
    level named.  ``None`` when that walk needs a parent no array records (a coordinate
    level below a dense, compressed or singleton one, :func:`_records_its_parent`).

    The caller keeps the sum inside int64 (a layout's element count fits one,
    layout._normalize_shape).  ``result_bound`` reads the operands' keys of an einsum
    term this way.
    """

    count = level_positions(level_types, mode_indices, extents, level)
    key = torch.zeros(count, dtype=torch.int64)
    if not strides:
        return key
    top = min(strides)
    if not _parents_recorded(level_types, level, top):
        return None
    for current, coordinates in _walk_up(
        level_types, mode_indices, extents, level, top, wanted=strides
    ):
        key += coordinates * strides[current]
    return key


def positions_beneath(
    level_types: Sequence[LevelType],
    mode_indices: IndexModes,
    extents: Sequence[int],
    level: int,
    depth: int,
) -> Optional[torch.Tensor]:
    """For every position of storage level ``level`` (``-1``: the root), as int64: how
    many positions of level ``depth`` (at least ``level``) lie beneath it.  ``None`` when
    a level between them has a parent no array records (a coordinate level below a
    dense, compressed or singleton one, :func:`_records_its_parent`).

    The positions beneath one position form a range at every level below it, carried
    down level by level: a dense level multiplies both ends by its extent, a compressed
    level maps them through ``pos``, a coordinate or singleton level keeps them (one
    position per parent position), and a coordinate or singleton level 0 holds every
    one of its positions beneath the root."""

    start = torch.arange(
        level_positions(level_types, mode_indices, extents, level), dtype=torch.int64
    )
    end = start + 1
    for current in range(level + 1, depth + 1):
        kind = level_types[current]
        if kind == LevelType.DENSE:
            start = start * int(extents[current])
            end = end * int(extents[current])
        elif kind == LevelType.COMPRESSED:
            positions = mode_indices[current][0].to(torch.int64)
            start, end = positions[start], positions[end]
        elif current == 0:
            # Beneath the root's one position: start is [0] and end [1] here.
            end = end * mode_indices[0][-1].numel()
        elif not _records_its_parent(level_types, current):
            return None
    return end - start


def structure_of(
    storage: "SparseStorage",
) -> Tuple[Tuple[LevelType, ...], IndexModes, Tuple[int, ...], Tuple[int, ...]]:
    """A storage's level kinds, index arrays, extents and permutation, in storage order
    (level ``l`` holds logical mode ``permutation[l]``).  The arrays are the storage's
    own, not copies: for analyses that only read them (``result_bound``)."""

    layout = storage._layout
    return (
        tuple(layout.format.level_types),
        storage._mode_indices,
        tuple(int(extent) for extent in layout.physical_shape),
        tuple(layout.permutation),
    )


def structure_stamp(storage: "SparseStorage") -> Tuple[Tuple[int, int, int], ...]:
    """``(data_ptr, version, numel)`` of each index array of ``storage``: equal stamps on
    the same live storage mean the same structure, for the reasons, and with the blind
    spots, :func:`_validation_stamp` gives for the same triple."""

    return tuple(
        (array.data_ptr(), tensor_version(array), array.numel())
        for arrays in storage._mode_indices
        for array in arrays
    )


def _canonical_coo_index(
    mode_indices: IndexModes,
    extents: Sequence[int],
    index_dtype: torch.dtype,
    count: int,
) -> Tuple[IndexModes, int, Optional[torch.Tensor]]:
    """``_canonical_index`` for a COO layout, whose every level stores one coordinate per
    entry, so that an entry's full coordinate is read straight from the levels."""
    if count == 0:
        return tuple((torch.zeros(0, dtype=index_dtype),) for _ in extents), 0, None
    key = torch.zeros(count, dtype=torch.int64)
    for arrays, extent in zip(mode_indices, extents):
        key = key * extent + arrays[0].to(torch.int64)
    key, order, entry, unique_count = _sorted_distinct(key)
    rebuilt: List[Tuple[torch.Tensor, ...]] = [()] * len(extents)
    for level in reversed(range(len(extents))):
        rebuilt[level] = ((key % extents[level]).to(index_dtype).contiguous(),)
        key = torch.div(key, extents[level], rounding_mode="floor")
    destination = torch.empty(count, dtype=torch.int64)
    destination[order] = entry
    return tuple(rebuilt), unique_count, destination


def _walked_coordinates(
    level_types: Sequence[LevelType],
    mode_indices: IndexModes,
    extents: Sequence[int],
    count: int,
) -> List[torch.Tensor]:
    """Each stored value's coordinate at every level, as one int64 tensor per level
    (physical order), entry ``p`` of each the coordinate of the value at position
    ``p``; the caller has checked the hierarchy (``stored_coordinates``).

    Walks up from each value's own position (:func:`_walk_up`, the walk every reader of
    a storage's levels shares): a dense level's coordinate and parent follow from the
    position arithmetically; a compressed level's come from its arrays; a singleton
    level, and a coordinate level below coordinate levels, hold one coordinate per
    position of the level above, so a position's parent is the position itself.  The
    positions walked are ``0 .. count - 1`` (``count`` stored values), so every tensor
    returned is a new one, never an index array of the storage.
    """

    coordinate_at: List[torch.Tensor] = [torch.zeros(0, dtype=torch.int64)] * len(
        level_types
    )
    for level, coordinates in _walk_up(
        level_types,
        mode_indices,
        extents,
        len(level_types) - 1,
        0,
        positions=torch.arange(count, dtype=torch.int64),
    ):
        coordinate_at[level] = coordinates
    return coordinate_at


def stored_coordinates(
    layout: TensorLayout, mode_indices: IndexModes, count: int
) -> List[torch.Tensor]:
    """The coordinate of every stored value, one int64 tensor per physical level.

    Entry ``p`` of the tensor for level ``k`` is the coordinate, along the physical
    mode of level ``k``, of the value at storage position ``p``.  Read from the index
    arrays without writing them, for every hierarchy whose arrays say which parent each
    entry belongs to: any sequence of dense, compressed and singleton levels, below a
    leading run of coordinate (COO) levels.  A coordinate level below a dense,
    compressed or singleton one holds one coordinate per stored value and nothing that
    places it under a parent, so that hierarchy is refused with a code, never guessed
    at.  ``count`` is the number of stored values.
    """

    level_types = layout.format.level_types
    extents = layout.physical_shape
    if not _parents_recorded(level_types, len(level_types) - 1, 0):
        raise TensorStorageError(
            f"the stored coordinates of format {layout.format} are not computed: a "
            "coordinate level below a dense, compressed or singleton level does not "
            "say which parent each of its entries belongs to",
            code="stored_coordinates_format_unsupported",
        )
    if count == 0:
        return [torch.zeros(0, dtype=torch.int64) for _ in level_types]
    if all(level == LevelType.COORDINATE for level in level_types):
        return [arrays[0].to(torch.int64) for arrays in mode_indices]
    return _walked_coordinates(level_types, mode_indices, extents, count)


def _canonical_index(
    layout: TensorLayout, mode_indices: IndexModes, value_count: int
) -> Tuple[IndexModes, int, Optional[torch.Tensor]]:
    """The canonical form of an index whose levels are all dense or compressed, or all
    coordinate (COO).

    Every parent's coordinates sorted and each coordinate stored once: for an index with a
    compressed level whose coordinates are out of order or repeated inside a parent, which
    torch accepts in a CSR tensor and sums in every operation on it, but which Scorch's
    kernels cannot read, since co-iterating two sparse operands assumes distinct,
    increasing coordinates; and for COO levels that store a coordinate twice or out of
    order, which Scorch refuses and ``STensor.coalesce()`` repairs. Every stored entry's
    full coordinate is recovered, the entries are sorted lexicographically, and the levels
    are rebuilt in the same format from the distinct coordinates.

    Returns the rebuilt index arrays, the number of values they store, and the map from
    each of the given index's value positions to its position in the rebuilt one
    (``None`` when nothing is stored), so that ``values`` in the given layout become
    ``zeros(count).index_add_(0, map, values)`` in the new one: entries at one coordinate
    are summed, and an explicitly stored zero is kept, as torch's ``coalesce`` does. The
    given arrays are read and never written, and the rebuilt ones share nothing with them.

    Only called once a defect has been found, so a well-formed index never pays for it.
    """
    level_types = layout.format.level_types
    extents = layout.physical_shape
    index_dtype = layout.index_dtype
    count = value_count

    def arange(n: int) -> torch.Tensor:
        return torch.arange(n, dtype=torch.int64)

    if level_types and all(level == LevelType.COORDINATE for level in level_types):
        return _canonical_coo_index(mode_indices, extents, index_dtype, count)
    if count == 0:
        # Nothing is stored below some level, so there is nothing to sort or sum: the
        # canonical form of an empty tensor stores no coordinates at all. (Any extent of
        # zero leads here, which keeps the divisions below away from it.)
        empty: List[Tuple[torch.Tensor, ...]] = []
        parent_count = 1
        for level_type, extent in zip(level_types, extents):
            if level_type == LevelType.DENSE:
                empty.append(())
                parent_count *= extent
            else:
                empty.append(
                    (
                        torch.zeros(parent_count + 1, dtype=index_dtype),
                        torch.zeros(0, dtype=index_dtype),
                    )
                )
                parent_count = 0
        return tuple(empty), parent_count, None

    coordinate_at = _walked_coordinates(level_types, mode_indices, extents, count)

    # Sort on the row-major linear index. It cannot overflow: a layout's element count
    # is checked to fit in a signed int64 (layout._normalize_shape).
    key = torch.zeros(count, dtype=torch.int64)
    for level, extent in enumerate(extents):
        key = key * extent + coordinate_at[level]
    key, order, entry, unique_count = _sorted_distinct(key)

    # Rebuild the levels from the sorted, distinct coordinates.
    unique_at: List[torch.Tensor] = [arange(0)] * len(level_types)
    for level in reversed(range(len(level_types))):
        extent = extents[level]
        unique_at[level] = key % extent
        key = torch.div(key, extent, rounding_mode="floor")
    rebuilt: List[Tuple[torch.Tensor, ...]] = []
    new_prefix = torch.zeros(unique_count, dtype=torch.bool)
    new_prefix[0] = True
    position = torch.zeros(unique_count, dtype=torch.int64)
    parent_count = 1
    for level_type, extent, coordinate in zip(level_types, extents, unique_at):
        if unique_count > 1:
            new_prefix[1:] |= coordinate[1:] != coordinate[:-1]
        if level_type == LevelType.DENSE:
            rebuilt.append(())
            position = position * extent + coordinate
            parent_count *= extent
            continue
        # One stored coordinate per distinct prefix through this level, grouped under
        # the position of its parent.
        kept_parent = position[new_prefix]
        kept_coordinate = coordinate[new_prefix]
        level_positions = torch.zeros(parent_count + 1, dtype=torch.int64)
        level_positions[1:] = torch.cumsum(
            torch.bincount(kept_parent, minlength=parent_count), 0
        )
        rebuilt.append(
            (
                level_positions.to(index_dtype),
                kept_coordinate.to(index_dtype).contiguous(),
            )
        )
        position = torch.cumsum(new_prefix.to(torch.int64), 0) - 1
        parent_count = kept_coordinate.numel()
    # `position` is each distinct entry's value position in the rebuilt index; `entry`,
    # in sorted order, is each given value's distinct entry.
    destination = torch.empty(count, dtype=torch.int64)
    destination[order] = position[entry]
    return tuple(rebuilt), parent_count, destination


def repeated_coordinate_in(
    indices: torch.Tensor, what: str, remedy: str
) -> RepeatedCoordinateError:
    """The error for COO coordinates ``indices`` (``[rank, nnz]``) that store a tuple more
    than once: it names the smallest repeated tuple and the first two positions storing it.
    Only built once a repeat is known to exist."""
    if indices.shape[0] == 0:  # rank 0: every entry is the one coordinate ()
        coordinate: Tuple[int, ...] = ()
        positions = [0, 1]
    else:
        distinct, inverse, counts = torch.unique(
            indices, dim=1, return_inverse=True, return_counts=True
        )
        repeated = int(torch.nonzero(counts > 1)[0].item())
        positions = torch.nonzero(inverse == repeated).flatten()[:2].tolist()
        coordinate = tuple(int(part) for part in distinct[:, repeated].tolist())
    return RepeatedCoordinateError(
        f"{what} store coordinate {format_coordinate(coordinate)} at positions "
        f"{positions[0]} and {positions[1]}; {COALESCE_ADVICE} ({remedy})",
        int(positions[1]),
        coordinate,
    )


def _canonical_storage(
    layout: TensorLayout, mode_indices: IndexModes, values: torch.Tensor
) -> Tuple[IndexModes, torch.Tensor]:
    """A private copy of the storage in canonical form (``_canonical_index``), values
    included: entries at one coordinate summed."""
    index, count, destination = _canonical_index(layout, mode_indices, values.numel())
    canonical_values = torch.zeros(count, dtype=values.dtype)
    if destination is not None:
        canonical_values.index_add_(0, destination, values)
    return index, canonical_values


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

    # `_checked` (the validation stamp, see validate_unless_already_checked) is a slot, so
    # the instance __dict__ holds exactly {_layout, _mode_indices, _value}: the state
    # `_owned_sparse_storage` audits. A stamp is never carried across that audit -- the
    # storage it rebuilds computes its own -- so keeping it out of the audited state does
    # not let one forged stamp skip a walk.
    __slots__ = ("_checked", "__dict__", "__weakref__")

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
        _audited_index: Optional[AuditedIndex] = None,
    ) -> None:
        # ``_audited_index`` is internal, like ``_trusted_index``: the four values that
        # ``_index_components`` returned to the caller's own frame a moment ago. The
        # arrays are retained as they are, with no second audit, and the layout is held
        # to the format, mode order and index dtype they were audited against, exactly
        # as it is held to an index object's further down. It is how a construction
        # inside Scorch audits its index arrays once (see ``_index_components``); a
        # caller that holds an index object passes ``index=`` and gets the full audit.
        #
        # A caller's layout passes through owned_layout: an exact TensorLayout, which
        # cannot change once built, is kept as it is, and a subclass is rebuilt as an
        # exact one. A trusted construction (a kernel result, assembled by scorch around
        # a layout scorch built) keeps the object it is handed without that step, since
        # STensor._set_state requires the metadata and the storage to name the same
        # layout object. So does one that brings ``_audited_index``, and there the
        # layout has to be an exact TensorLayout already: a subclass is refused, not
        # rebuilt, because the caller's metadata names the object it handed over.
        if _audited_index is not None:
            if type(layout) is not TensorLayout:
                raise TensorTypeError(
                    "layout must be a TensorLayout", code="layout_type_invalid"
                )
        elif not _trusted_index:
            layout = owned_layout(layout)
        value_type = type(value)
        if value_type is torch.Tensor:
            # As for an index array in ``_normalize_mode_indices``: an exact tensor needs
            # no base-class view, the method reads below go through the type, and what
            # is retained is the detached tensor made further down.
            base_value = value
        else:
            try:
                value_mro = type.__getattribute__(value_type, "__mro__")
            except Exception as error:
                raise TensorTypeError(
                    "SparseStorage value must be a torch.Tensor",
                    code="values_type_invalid",
                ) from error
            if not any(base is torch.Tensor for base in value_mro):
                raise TensorTypeError(
                    "SparseStorage value must be a torch.Tensor",
                    code="values_type_invalid",
                )
            try:
                base_value = torch.Tensor.as_subclass(value, torch.Tensor)
            except Exception as error:
                raise TensorStorageError(
                    "values tensor is malformed", code="values_malformed"
                ) from error
            if type(base_value) is not torch.Tensor:
                raise TensorStorageError(
                    "values tensor is malformed", code="values_malformed"
                )
        if base_value.layout != torch.strided:
            raise TensorStorageError(
                "values must use strided storage", code="values_layout_invalid"
            )
        if base_value.device.type != "cpu":
            raise TensorDeviceError(
                f"Scorch values must be on CPU, got {base_value.device}",
                code="device_not_cpu_unsupported",
            )
        if torch.Tensor.dim(base_value) != 1:
            raise TensorStorageError(
                "values must be a flat one-dimensional tensor",
                code="values_shape_invalid",
            )
        if not torch.Tensor.is_contiguous(base_value):
            raise TensorStorageError(
                "values must be contiguous", code="values_layout_invalid"
            )
        if torch.Tensor.is_neg(base_value) or torch.Tensor.is_conj(base_value):
            raise TensorStorageError(
                "values must resolve lazy negative and conjugate view bits",
                code="values_layout_invalid",
            )
        validate_runtime_contract(layout.format, base_value.dtype)
        if (
            (mode_indices is not None)
            + (index is not None)
            + (_audited_index is not None)
        ) != 1:
            raise TensorStorageError(
                "provide exactly one of mode_indices or a TensorIndex",
                code="storage_arguments_conflict",
            )
        if _audited_index is not None:
            index_format, normalized, index_order, index_dtype = _audited_index
        elif index is not None:
            # Arrays the index adopted on purpose (`_adopt`: a torch tensor's arrays, which
            # the STensor shares as torch does, or a kernel result's) and a kernel result's
            # trusted index are shared and re-audited -- type, dtype, layout, rank,
            # contiguity -- without a copy. Any other index's arrays are copied, so two
            # storages built from one index, or a storage and the index it came from, never
            # share arrays: a write through one cannot reach the other.
            share = _trusted_index or _index_adopted(index)
            (
                index_format,
                held,
                index_order,
                index_dtype,
            ) = _owned_tensor_index_components(index, own_arrays=not share)
            normalized = (
                _normalize_mode_indices(held, index_format, copy=False)
                if share
                else held
            )
        else:
            normalized = _normalize_mode_indices(mode_indices, layout.format)  # type: ignore[arg-type]
        if mode_indices is None:
            # What an index object declares, or what the arrays were audited against,
            # has to be what the layout says. The identity test comes first because a
            # construction inside Scorch builds its layout from the format its arrays
            # were audited against, which is then the same object or an equal one.
            if index_format is not layout.format and index_format != layout.format:
                raise TensorIndexError(
                    "index format does not match storage layout",
                    code="index_layout_mismatch",
                )
            if index_order != layout.permutation:
                raise TensorIndexError(
                    "index mode_order does not match storage layout",
                    code="index_layout_mismatch",
                )
            if index_dtype != layout.index_dtype:
                raise TensorIndexError(
                    "index dtype does not match storage layout",
                    code="index_layout_mismatch",
                )
        # Keep tensor metadata independent from the caller while retaining normal
        # tensor payload aliasing semantics for element updates.
        owned_value = torch.Tensor.detach(base_value)
        if type(owned_value) is not torch.Tensor:
            raise TensorStorageError(
                "values tensor is malformed", code="values_malformed"
            )
        # The cheap per-array checks -- dtype, rank, contiguity, device, one array per
        # level -- have already run inside `_normalize_mode_indices` for every caller,
        # trusted or not. They are O(1) each and adopting a kernel's arrays depends on
        # them. What `_trusted_index` skips is only the O(nnz) structural walk.
        if not _trusted_index:
            # A caller's arrays may hold a compressed level whose coordinates are out of
            # order or repeated inside a parent, which torch accepts in a CSR tensor and
            # sums. The storage then holds a sorted, repeat-summed private copy instead
            # (_canonical_storage), so it shares no arrays with the caller and a later
            # write to theirs does not reach it, as with an uncoalesced COO tensor, which
            # `from_torch` coalesces into a new one. A kernel's result is never repaired:
            # the branch below makes the same defect in one an error, because it means
            # the compiler emitted something wrong.
            if _validate_index_storage(
                layout, normalized, owned_value, allow_unordered=True
            ):
                normalized, owned_value = _canonical_storage(
                    layout, normalized, owned_value
                )
                _validate_index_storage(layout, normalized, owned_value)
        elif _VALIDATE_KERNEL_RESULTS[0]:
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
        # The constructor detaches both retaining graphs. Passing the owned
        # values directly avoids cloning them once here and a second time in
        # their canonicalizers.
        return SparseStorage(
            self._layout,
            self._value.clone(),
            mode_indices=self._mode_indices,
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


def _owned_sparse_storage(storage: object) -> SparseStorage:
    """Detach a storage's structural graph while preserving payload aliasing."""

    storage_type = type(storage)
    try:
        mro = type.__getattribute__(storage_type, "__mro__")
    except Exception as error:
        raise TensorTypeError(
            "storage must be a SparseStorage", code="storage_type_invalid"
        ) from error
    if not any(base is SparseStorage for base in mro):
        raise TensorTypeError(
            "storage must be a SparseStorage", code="storage_type_invalid"
        )
    try:
        state = SparseStorage.__dict__["__dict__"].__get__(storage, SparseStorage)
    except Exception as error:
        raise TensorStorageError(
            "storage has malformed stored state", code="value_object_state_corrupt"
        ) from error
    expected = {"_layout", "_mode_indices", "_value"}
    if (
        type(state) is not dict
        or any(type(key) is not str for key in state)
        or set(state) != expected
        or type(state["_mode_indices"]) is not tuple
        or any(type(arrays) is not tuple for arrays in state["_mode_indices"])
    ):
        raise TensorStorageError(
            "storage has malformed stored state", code="value_object_state_corrupt"
        )
    return SparseStorage(
        state["_layout"],
        state["_value"],
        mode_indices=state["_mode_indices"],
    )


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
            index_format, _, index_order, index_dtype = _owned_tensor_index_components(
                index, own_arrays=False
            )
            layout = TensorLayout.from_physical_shape(
                shape,  # type: ignore[arg-type]
                index_format,
                index_order,
                index_dtype,
            )
        elif shape is not None:
            layout = owned_layout(layout)
            declared_shape = _normalize_shape(shape, "TensorStorage shape")
            if declared_shape != layout.physical_shape:
                raise TensorLayoutError(
                    f"TensorStorage shape {declared_shape} does not match layout "
                    f"physical shape {layout.physical_shape}"
                )
        if value is None:
            raise TensorStorageError("TensorStorage value is required")
        super().__init__(layout, value, mode_indices=mode_indices, index=index)


class TensorStorageView(SparseStorage):
    """Reserved validated secondary-index storage type."""
