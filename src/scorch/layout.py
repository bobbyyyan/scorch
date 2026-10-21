"""Immutable tensor layout, metadata, and compile-only specifications."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math

# Mapping/Sequence come from collections.abc, not typing: these are used for
# runtime isinstance checks on a per-call path, and typing's generic aliases
# route isinstance through __subclasscheck__ at 153 ns against 73 ns for the
# abc. `from __future__ import annotations` above means the annotations that
# also use these names are never evaluated at runtime.
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch

from .exceptions import (
    CompileSpecError,
    OperandDtypeMismatchError,
    ScorchFrozenInstanceError,
    ShapeCapacityLayoutError,
    ShapeCapacitySpecError,
    TensorDeviceError,
    TensorLayoutError,
    TensorTypeError,
    TensorValidationError,
    UnsupportedDtypePromotionError,
    defect_code,
    shape_capacity_error,
)
from .format import (
    FormatInput,
    LevelType,
    TensorFormat,
    _field_unset,
    _reduce_value,
    _restore_value,
    _seal,
    _sealed_new,
    _sealed_type,
    owned_format,
    parse_format,
)

ShapeLike = Sequence[int]
_MAX_INT64 = (1 << 63) - 1
_INDEX_DTYPES = (torch.int32, torch.int64)
SUPPORTED_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.int8,
    torch.uint8,
)


def validate_runtime_contract(tensor_format: TensorFormat, dtype: torch.dtype) -> None:
    """Reject layouts and value dtypes that Scorch cannot execute safely."""
    if dtype not in SUPPORTED_VALUE_DTYPES:
        raise TensorValidationError(
            f"Scorch tensor dtype {dtype} is not supported", code="dtype_unsupported"
        )
    # A membership test, not a generator over the levels: this runs for every tensor's
    # metadata, and level_types holds the exact LevelType members.
    if LevelType.SINGLETON in tensor_format.level_types:
        raise TensorLayoutError(
            "singleton levels are not supported by the Scorch runtime",
            code="singleton_level_unsupported",
        )


def common_value_dtype(*dtypes: torch.dtype, operation: str) -> torch.dtype:
    """The value dtype the result of ``operation`` on operands of ``dtypes`` stores.

    PyTorch's promotion rule (``torch.promote_types``), so that no operand's values
    are rounded or truncated on the way into the result: an int64 matrix times a
    float64 vector is float64, not int64.  For operands of one dtype it is that
    dtype.  Promotion can name a type Scorch does not store (int8 with uint8
    promotes to int16); such a pair is refused.
    """

    common = dtypes[0]
    for dtype in dtypes[1:]:
        common = torch.promote_types(common, dtype)
    if common not in SUPPORTED_VALUE_DTYPES:
        raise UnsupportedDtypePromotionError(
            f"{operation} of operands with value dtypes "
            f"{', '.join(str(dtype) for dtype in dtypes)} needs {common}, "
            "which Scorch does not store"
        )
    return common


def require_same_value_dtype(*dtypes: torch.dtype, operation: str) -> torch.dtype:
    """The one value dtype of a product's operands, or ``OperandDtypeMismatchError``.

    PyTorch does not promote the operands of a product (``matmul``, ``mv``, ``mm``,
    ``@``, ``einsum``): operands of different dtypes raise ``RuntimeError``.  Sums
    promote instead (:func:`common_value_dtype`).
    """

    if any(dtype != dtypes[0] for dtype in dtypes[1:]):
        raise OperandDtypeMismatchError(
            f"{operation} operands must have the same value dtype, got "
            f"{', '.join(str(dtype) for dtype in dtypes)}"
        )
    return dtypes[0]


def _normalize_shape(shape: ShapeLike, field: str) -> Tuple[int, ...]:
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise TensorTypeError(
            f"{field} must be a sequence of integers", code="shape_type_invalid"
        )
    normalized: list[int] = []
    for mode, extent in enumerate(shape):
        extent_type = type(extent)
        if extent_type is bool or not issubclass(extent_type, int):
            raise TensorTypeError(
                f"{field}[{mode}] must be an integer, got {type(extent).__name__}",
                code="shape_type_invalid",
            )
        # Do not retain or execute arithmetic supplied by an ``int``
        # subclass after this boundary.  Calling the base descriptor directly
        # both bypasses an overridden ``__int__`` and produces an exact int.
        extent = int.__int__(extent)
        if extent < 0:
            raise TensorLayoutError(
                f"{field}[{mode}] must be nonnegative", code="shape_extent_negative"
            )
        if extent > _MAX_INT64:
            raise shape_capacity_error(
                ShapeCapacityLayoutError,
                f"{field}[{mode}] exceeds signed int64",
                field,
                normalized,
                mode,
                extent,
            )
        normalized.append(extent)
    # The count of every extent, multiplied exactly: a zero extent holds no element
    # whatever the extents before it multiply to.
    if 0 not in normalized:
        count = math.prod(normalized)
        if count > _MAX_INT64:
            raise shape_capacity_error(
                ShapeCapacityLayoutError,
                f"{field} element count exceeds signed int64",
                field,
                normalized,
                None,
                count,
            )
    return tuple(normalized)


def _normalize_permutation(
    permutation: Optional[Sequence[int]], rank: int
) -> Tuple[int, ...]:
    if permutation is None:
        return tuple(range(rank))
    if isinstance(permutation, (str, bytes)) or not isinstance(permutation, Sequence):
        raise TensorTypeError(
            "layout permutation must be a sequence of integers",
            code="mode_order_type_invalid",
        )
    supplied = tuple(permutation)
    if any(type(mode) is bool or not issubclass(type(mode), int) for mode in supplied):
        raise TensorTypeError(
            "layout permutation entries must be integers",
            code="mode_order_type_invalid",
        )
    result = tuple(int.__int__(mode) for mode in supplied)
    if len(result) != rank:
        raise TensorLayoutError(
            f"layout permutation has rank {len(result)}, expected {rank}",
            code="mode_order_not_permutation",
        )
    if sorted(result) != list(range(rank)):
        raise TensorLayoutError(
            f"layout permutation must contain each mode in range({rank}) once",
            code="mode_order_not_permutation",
        )
    return result


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _parse_dtype(value: Union[str, torch.dtype], field: str) -> torch.dtype:
    if type(value) is torch.dtype:
        return value
    if issubclass(type(value), str):
        name = str.removeprefix(str.__str__(value), "torch.")
        parsed = getattr(torch, name, None)
        if type(parsed) is torch.dtype:
            return parsed
    raise TensorTypeError(
        f"{field} must be a torch.dtype, got {type(value).__name__}",
        code="dtype_type_invalid",
    )


# Shared TensorLayout instances, keyed by the arguments that produced them. A layout
# cannot change once built (format.py, "Read-only stored state"), so every tensor built
# from equal arguments may hold the same one. Bounded so a caller cycling through shapes
# cannot grow it without limit; the real key space for a given model is a handful of
# (shape, format) pairs.
_LAYOUT_CACHE: dict = {}
_LAYOUT_CACHE_MAX = 4096

_LAYOUT_FIELDS = (
    "logical_shape",
    "physical_shape",
    "format",
    "permutation",
    "index_dtype",
)
if TYPE_CHECKING:
    # The bases are made at import time, so a type checker sees them as plain objects.
    _TensorLayoutFields = object
    _TensorMetadataFields = object
else:
    _TensorLayoutFields = _sealed_type("TensorLayoutFields", _LAYOUT_FIELDS)
    _TensorMetadataFields = _sealed_type(
        "TensorMetadataFields",
        ("name", "dtype", "device", "layout", "requires_grad"),
    )
_LAYOUT_DESCRIPTORS = tuple(
    (name, _TensorLayoutFields.__dict__[name]) for name in _LAYOUT_FIELDS
)


@dataclass(frozen=True, init=False)
class TensorLayout(_TensorLayoutFields):
    """Canonical mapping between logical modes and their physical storage."""

    # Read-only fields held by TensorLayoutFields: see format.py, "Read-only stored
    # state". index_dtype's default is __init__'s: a default here would be a class
    # attribute hiding the field.
    __slots__ = ()

    logical_shape: Tuple[int, ...]
    physical_shape: Tuple[int, ...]
    format: TensorFormat
    permutation: Tuple[int, ...]
    index_dtype: torch.dtype

    def __init__(
        self,
        logical_shape: ShapeLike,
        physical_shape: ShapeLike,
        format: FormatInput,
        permutation: Sequence[int],
        index_dtype: torch.dtype = torch.int32,
    ) -> None:
        logical_shape = _normalize_shape(logical_shape, "logical_shape")
        physical_shape = _normalize_shape(physical_shape, "physical_shape")
        # The layout is the tensor's single authoritative format holder. See
        # ``format.owned_format``: an exact format is held as it is, a subclass is
        # normalized, and a format never constructed fails closed.
        tensor_format = owned_format(parse_format(format))
        rank = len(logical_shape)
        permutation = _normalize_permutation(permutation, rank)
        index_dtype = _parse_dtype(index_dtype, "index_dtype")
        if len(physical_shape) != rank:
            raise TensorLayoutError(
                "logical_shape and physical_shape must have the same rank",
                code="physical_shape_mismatch",
            )
        if tensor_format.get_order() != rank:
            raise TensorLayoutError(
                f"format rank {tensor_format.get_order()} does not match shape rank {rank}",
                code="format_rank_mismatch",
            )
        expected_physical = tuple(logical_shape[mode] for mode in permutation)
        if physical_shape != expected_physical:
            raise TensorLayoutError(
                "physical_shape must equal logical_shape reordered by permutation: "
                f"expected {expected_physical}, got {physical_shape}",
                code="physical_shape_mismatch",
            )
        if index_dtype not in _INDEX_DTYPES:
            raise TensorLayoutError(
                "index_dtype must be torch.int32 or torch.int64, " f"got {index_dtype}",
                code="index_dtype_invalid",
            )
        if not _seal(
            self, logical_shape, physical_shape, tensor_format, permutation, index_dtype
        ):
            raise ScorchFrozenInstanceError(
                "cannot assign to field 'logical_shape'",
                code="value_object_already_sealed",
            )

    def __reduce__(self):
        # An exact layout loads through from_physical_shape (the logical shape follows
        # from the physical one and the permutation), which serves an equal layout from
        # the layout cache and validates the arguments when it is not there.
        if type(self) is TensorLayout:
            return (
                TensorLayout.from_physical_shape,
                (self.physical_shape, self.format, self.permutation, self.index_dtype),
            )
        return _reduce_value(
            self,
            (
                self.logical_shape,
                self.physical_shape,
                self.format,
                self.permutation,
                self.index_dtype,
            ),
        )

    def __setstate__(self, state):
        _restore_value(
            self, state, TensorLayout.__init__, _LAYOUT_DESCRIPTORS, TensorLayoutError
        )

    @classmethod
    def from_logical_shape(
        cls,
        logical_shape: ShapeLike,
        tensor_format: FormatInput,
        permutation: Optional[Sequence[int]] = None,
        index_dtype: torch.dtype = torch.int32,
    ) -> "TensorLayout":
        logical = _normalize_shape(logical_shape, "logical_shape")
        order = _normalize_permutation(permutation, len(logical))
        physical = tuple(logical[mode] for mode in order)
        return cls(logical, physical, parse_format(tensor_format), order, index_dtype)

    @classmethod
    def from_physical_shape(
        cls,
        physical_shape: ShapeLike,
        tensor_format: FormatInput,
        permutation: Optional[Sequence[int]] = None,
        index_dtype: torch.dtype = torch.int32,
    ) -> "TensorLayout":
        # Every STensor construction lands here, and a TensorLayout cannot change once
        # built, so equal arguments can share one instance. On a small SpMM the
        # normalize-and-validate work below is ~11% of the whole call, repeated
        # identically on every iteration of a training loop. Only the exact class is
        # cached, since the key does not name the class.
        #
        # A key build that raises is not an error: an argument too malformed to hash is
        # left to the normal path below, which reports it with the right exception type.
        key = None
        cached = None
        if cls is TensorLayout:
            try:
                key = (
                    tuple(physical_shape),
                    tensor_format,
                    None if permutation is None else tuple(permutation),
                    index_dtype,
                )
                cached = _LAYOUT_CACHE.get(key)
            except TypeError:
                key = None
        if cached is not None:
            return cached

        physical = _normalize_shape(physical_shape, "physical_shape")
        order = _normalize_permutation(permutation, len(physical))
        logical = [0] * len(physical)
        for physical_mode, logical_mode in enumerate(order):
            logical[logical_mode] = physical[physical_mode]
        layout = cls(
            tuple(logical), physical, parse_format(tensor_format), order, index_dtype
        )
        if key is not None and len(_LAYOUT_CACHE) < _LAYOUT_CACHE_MAX:
            _LAYOUT_CACHE[key] = layout
        return layout

    @property
    def rank(self) -> int:
        return len(self.logical_shape)

    @property
    def logical_to_physical(self) -> Tuple[int, ...]:
        inverse = [0] * self.rank
        for physical_mode, logical_mode in enumerate(self.permutation):
            inverse[logical_mode] = physical_mode
        return tuple(inverse)

    @property
    def element_count(self) -> int:
        return math.prod(self.physical_shape)

    def with_permutation(self, permutation: Sequence[int]) -> "TensorLayout":
        return TensorLayout.from_logical_shape(
            self.logical_shape,
            self.format,
            permutation,
            self.index_dtype,
        )

    def with_format(
        self, tensor_format: FormatInput, index_dtype: Optional[torch.dtype] = None
    ) -> "TensorLayout":
        return TensorLayout(
            self.logical_shape,
            self.physical_shape,
            parse_format(tensor_format),
            self.permutation,
            self.index_dtype if index_dtype is None else index_dtype,
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "logical_shape": list(self.logical_shape),
            "physical_shape": list(self.physical_shape),
            "format": self.format.to_dict(),
            "permutation": list(self.permutation),
            "index_dtype": _dtype_name(self.index_dtype),
        }

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorLayout":
        if not isinstance(data, Mapping):
            raise TensorTypeError("serialized layout must be a mapping")
        try:
            return cls(
                logical_shape=tuple(data["logical_shape"]),
                physical_shape=tuple(data["physical_shape"]),
                format=TensorFormat.from_dict(data["format"]),
                permutation=tuple(data["permutation"]),
                index_dtype=_parse_dtype(data["index_dtype"], "index_dtype"),
            )
        except KeyError as error:
            raise TensorLayoutError(
                f"serialized layout is missing {error.args[0]!r}"
            ) from error
        except (TypeError, ValueError, RuntimeError, OverflowError) as error:
            raise TensorLayoutError("serialized layout is malformed") from error


def owned_layout(layout: object) -> TensorLayout:
    """The layout an object retains when it is handed ``layout``.

    A layout cannot change once built (format.py, "Read-only stored state"), so an exact
    ``TensorLayout`` is retained as it is, and any number of storages and metadata may
    share it. A subclass is rebuilt as an exact layout from the base's fields, read through
    the base's own descriptors so that none of the subclass's code runs; the constructor
    validates them again. A layout that was never constructed fails closed.
    """

    layout_type = type(layout)
    # The constructor sets every field at once, so one unset field means all are.
    if layout_type is TensorLayout:
        if _field_unset(layout, _LAYOUT_DESCRIPTORS[0][1]):
            raise TensorLayoutError(
                "layout has malformed stored state", code="value_object_state_corrupt"
            )
        return cast(TensorLayout, layout)
    try:
        is_layout = issubclass(layout_type, TensorLayout)
    except TypeError:
        is_layout = False
    if not is_layout:
        raise TensorTypeError(
            "layout must be a TensorLayout", code="layout_type_invalid"
        )
    if _field_unset(layout, _LAYOUT_DESCRIPTORS[0][1]):
        raise TensorLayoutError(
            "layout has malformed stored state", code="value_object_state_corrupt"
        )
    fields = [descriptor.__get__(layout) for _, descriptor in _LAYOUT_DESCRIPTORS]
    try:
        return TensorLayout(*fields)
    except (TensorTypeError, TensorLayoutError):
        raise
    except Exception as error:
        raise TensorLayoutError(
            "layout has malformed stored state", code="value_object_state_corrupt"
        ) from error


_METADATA_FIELDS = ("name", "dtype", "device", "layout", "requires_grad")
_METADATA_DESCRIPTORS = tuple(
    (name, _TensorMetadataFields.__dict__[name]) for name in _METADATA_FIELDS
)


def _metadata_fields(
    name: Any,
    dtype: Any,
    device: Any,
    layout: Any,
    requires_grad: Any,
    *,
    own_layout: bool,
) -> Tuple[Any, ...]:
    """Validate metadata's arguments; return its fields in declaration order."""
    if not issubclass(type(name), str):
        raise TensorLayoutError(
            "tensor name must be a non-empty string", code="tensor_name_invalid"
        )
    # Bypass a subclass override and retain an exact immutable ``str``.
    name = str.strip(name)
    if not name:
        raise TensorLayoutError(
            "tensor name must be a non-empty string", code="tensor_name_invalid"
        )
    dtype = _parse_dtype(dtype, "dtype")
    try:
        if type(device) is torch.device:
            pass
        elif issubclass(type(device), str):
            device = torch.device(str.__str__(device))
        else:
            raise TypeError
    except Exception as error:
        raise TensorDeviceError(
            f"invalid tensor device of type {type(device).__name__}",
            code="device_type_invalid",
        ) from error
    if device.type != "cpu":
        raise TensorDeviceError(
            f"Scorch runtime tensors must be on CPU, got {device}",
            code="device_not_cpu_unsupported",
        )
    if own_layout:
        layout = owned_layout(layout)
    elif type(layout) is not TensorLayout:
        raise TensorTypeError(
            "owned metadata layout must be a TensorLayout", code="layout_type_invalid"
        )
    validate_runtime_contract(layout.format, dtype)
    if type(requires_grad) is not bool:
        raise TensorTypeError(
            "requires_grad must be a bool", code="requires_grad_type_invalid"
        )
    return (name, dtype, device, layout, requires_grad)


@dataclass(frozen=True, init=False)
class TensorMetadata(_TensorMetadataFields):
    """All non-payload runtime tensor metadata in one immutable value."""

    # Read-only fields held by TensorMetadataFields: see format.py, "Read-only stored
    # state". requires_grad's default is __init__'s, as index_dtype's is TensorLayout's.
    __slots__ = ()

    name: str
    dtype: torch.dtype
    device: torch.device
    layout: TensorLayout
    requires_grad: bool

    def __init__(
        self,
        name: str,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        layout: TensorLayout,
        requires_grad: bool = False,
    ) -> None:
        fields = _metadata_fields(
            name, dtype, device, layout, requires_grad, own_layout=True
        )
        if not _seal(self, *fields):
            raise ScorchFrozenInstanceError(
                "cannot assign to field 'name'", code="value_object_already_sealed"
            )

    @classmethod
    def _from_owned_layout(
        cls,
        name: str,
        dtype: torch.dtype,
        device: torch.device,
        layout: TensorLayout,
        requires_grad: bool = False,
    ) -> "TensorMetadata":
        """Build metadata around a layout the caller already owns.

        Public construction passes its layout through ``owned_layout``. A runtime tensor's
        metadata and storage hold one layout object between them; this private boundary
        builds the metadata around the storage's, which is an exact ``TensorLayout`` by
        construction, without that step.
        """

        return _sealed_new(
            cls,
            *_metadata_fields(
                name, dtype, device, layout, requires_grad, own_layout=False
            ),
        )

    def __reduce__(self):
        return _reduce_value(
            self,
            (self.name, self.dtype, self.device, self.layout, self.requires_grad),
        )

    def __setstate__(self, state):
        _restore_value(
            self,
            state,
            TensorMetadata.__init__,
            _METADATA_DESCRIPTORS,
            TensorLayoutError,
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "dtype": _dtype_name(self.dtype),
            "device": str(self.device),
            "layout": self.layout.to_dict(),
            "requires_grad": self.requires_grad,
        }

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorMetadata":
        if not isinstance(data, Mapping):
            raise TensorTypeError("serialized metadata must be a mapping")
        try:
            return cls(
                name=data["name"],
                dtype=_parse_dtype(data["dtype"], "dtype"),
                device=torch.device(data["device"]),
                layout=TensorLayout.from_dict(data["layout"]),
                requires_grad=data.get("requires_grad", False),
            )
        except KeyError as error:
            raise TensorLayoutError(
                f"serialized metadata is missing {error.args[0]!r}"
            ) from error
        except (TypeError, RuntimeError) as error:
            raise TensorLayoutError("serialized metadata is malformed") from error


@dataclass(frozen=True, init=False)
class TensorSpec:
    """Immutable, payload-free tensor description used only for compilation."""

    metadata: TensorMetadata

    def __init__(
        self,
        tensor_format: FormatInput,
        shape: ShapeLike,
        *,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        mode_order: Optional[Sequence[int]] = None,
        index_dtype: torch.dtype = torch.int32,
        name: str = "compile_spec",
    ) -> None:
        try:
            parsed_format = parse_format(tensor_format)
            parsed_dtype = _parse_dtype(dtype, "dtype")
            if any(
                level_type == LevelType.SINGLETON
                for level_type in parsed_format.get_level_types()
            ):
                raise CompileSpecError(
                    "singleton levels are not supported by the compiler"
                )
            if parsed_dtype not in SUPPORTED_VALUE_DTYPES:
                raise CompileSpecError(
                    f"compiler tensor dtype {parsed_dtype} is not supported"
                )
            layout = TensorLayout.from_logical_shape(
                shape, parsed_format, mode_order, index_dtype
            )
            metadata = TensorMetadata(
                name, parsed_dtype, torch.device(device), layout, False
            )
        except ShapeCapacityLayoutError as error:
            # The clause below would keep the text and drop the code.
            raise error.retyped(ShapeCapacitySpecError) from error
        except (
            TensorLayoutError,
            TensorDeviceError,
            TensorTypeError,
            TensorValidationError,
            TypeError,
            RuntimeError,
        ) as error:
            raise CompileSpecError(
                str(error), code=defect_code(error) or "tensor_spec_invalid"
            ) from error
        object.__setattr__(self, "metadata", metadata)

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def dtype(self) -> torch.dtype:
        return self.metadata.dtype

    @property
    def device(self) -> torch.device:
        return self.metadata.device

    @property
    def layout(self) -> TensorLayout:
        return self.metadata.layout

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.layout.physical_shape

    @property
    def logical_shape(self) -> Tuple[int, ...]:
        return self.layout.logical_shape

    @property
    def physical_shape(self) -> Tuple[int, ...]:
        return self.layout.physical_shape

    @property
    def format(self) -> TensorFormat:
        return self.layout.format

    @property
    def mode_order(self) -> Tuple[int, ...]:
        return self.layout.permutation

    @property
    def index_dtype(self) -> torch.dtype:
        return self.layout.index_dtype

    def dim(self) -> int:
        return self.layout.rank

    def with_mode_order(
        self,
        mode_order: Sequence[int],
        *,
        tensor_format: Optional[TensorFormat] = None,
    ) -> "TensorSpec":
        """This spec under another storage order, and optionally another format.

        ``tensor_format`` is set only by einsum's operand relayout
        (``scorch.ops._einsum_operand_relayout``), for an operand it converts.
        """
        new_layout = TensorLayout.from_logical_shape(
            self.logical_shape,
            self.format if tensor_format is None else tensor_format,
            mode_order,
            self.index_dtype,
        )
        spec = object.__new__(TensorSpec)
        object.__setattr__(spec, "metadata", replace(self.metadata, layout=new_layout))
        return spec

    def to_dict(self) -> Mapping[str, Any]:
        return {"kind": "tensor_spec", "metadata": self.metadata.to_dict()}

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TensorSpec":
        if not isinstance(data, Mapping) or data.get("kind") != "tensor_spec":
            raise CompileSpecError("serialized TensorSpec has an invalid kind")
        if "metadata" not in data:
            raise CompileSpecError("serialized TensorSpec is missing metadata")
        metadata = TensorMetadata.from_dict(data["metadata"])
        if metadata.requires_grad:
            raise CompileSpecError("TensorSpec metadata cannot require gradients")
        return cls(
            metadata.layout.format,
            metadata.layout.logical_shape,
            dtype=metadata.dtype,
            device=metadata.device,
            mode_order=metadata.layout.permutation,
            index_dtype=metadata.layout.index_dtype,
            name=metadata.name,
        )
