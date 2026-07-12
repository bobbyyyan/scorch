"""Immutable tensor layout, metadata, and compile-only specifications."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch

from .exceptions import (
    CompileSpecError,
    TensorDeviceError,
    TensorLayoutError,
    TensorTypeError,
    TensorValidationError,
)
from .format import FormatInput, LevelType, TensorFormat, parse_format

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
        raise TensorValidationError(f"Scorch tensor dtype {dtype} is not supported")
    if any(
        level_type == LevelType.SINGLETON
        for level_type in tensor_format.get_level_types()
    ):
        raise TensorLayoutError(
            "singleton levels are not supported by the Scorch runtime"
        )


def _normalize_shape(shape: ShapeLike, field: str) -> Tuple[int, ...]:
    if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence):
        raise TensorTypeError(f"{field} must be a sequence of integers")
    normalized = []
    product = 1
    for mode, extent in enumerate(shape):
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TensorTypeError(
                f"{field}[{mode}] must be an integer, got {type(extent).__name__}"
            )
        if extent < 0:
            raise TensorLayoutError(f"{field}[{mode}] must be nonnegative")
        if extent > _MAX_INT64:
            raise TensorLayoutError(f"{field}[{mode}] exceeds signed int64")
        if extent and product > _MAX_INT64 // extent:
            raise TensorLayoutError(f"{field} element count exceeds signed int64")
        product *= extent
        normalized.append(extent)
    return tuple(normalized)


def _normalize_permutation(
    permutation: Optional[Sequence[int]], rank: int
) -> Tuple[int, ...]:
    if permutation is None:
        return tuple(range(rank))
    if isinstance(permutation, (str, bytes)) or not isinstance(permutation, Sequence):
        raise TensorTypeError("layout permutation must be a sequence of integers")
    result = tuple(permutation)
    if any(isinstance(mode, bool) or not isinstance(mode, int) for mode in result):
        raise TensorTypeError("layout permutation entries must be integers")
    if len(result) != rank:
        raise TensorLayoutError(
            f"layout permutation has rank {len(result)}, expected {rank}"
        )
    if sorted(result) != list(range(rank)):
        raise TensorLayoutError(
            f"layout permutation must contain each mode in range({rank}) once"
        )
    return result


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _parse_dtype(value: Union[str, torch.dtype], field: str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        name = value.removeprefix("torch.")
        parsed = getattr(torch, name, None)
        if isinstance(parsed, torch.dtype):
            return parsed
    raise TensorTypeError(f"{field} must be a torch.dtype, got {value!r}")


@dataclass(frozen=True)
class TensorLayout:
    """Canonical mapping between logical modes and their physical storage."""

    logical_shape: Tuple[int, ...]
    physical_shape: Tuple[int, ...]
    format: TensorFormat
    permutation: Tuple[int, ...]
    index_dtype: torch.dtype = torch.int32

    def __post_init__(self) -> None:
        logical_shape = _normalize_shape(self.logical_shape, "logical_shape")
        physical_shape = _normalize_shape(self.physical_shape, "physical_shape")
        tensor_format = parse_format(self.format)
        rank = len(logical_shape)
        permutation = _normalize_permutation(self.permutation, rank)
        index_dtype = _parse_dtype(self.index_dtype, "index_dtype")
        if len(physical_shape) != rank:
            raise TensorLayoutError(
                "logical_shape and physical_shape must have the same rank"
            )
        if tensor_format.get_order() != rank:
            raise TensorLayoutError(
                f"format rank {tensor_format.get_order()} does not match shape rank {rank}"
            )
        expected_physical = tuple(logical_shape[mode] for mode in permutation)
        if physical_shape != expected_physical:
            raise TensorLayoutError(
                "physical_shape must equal logical_shape reordered by permutation: "
                f"expected {expected_physical}, got {physical_shape}"
            )
        if index_dtype not in _INDEX_DTYPES:
            raise TensorLayoutError(
                "index_dtype must be torch.int32 or torch.int64, " f"got {index_dtype}"
            )
        object.__setattr__(self, "logical_shape", logical_shape)
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "format", tensor_format)
        object.__setattr__(self, "permutation", permutation)
        object.__setattr__(self, "index_dtype", index_dtype)

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
        physical = _normalize_shape(physical_shape, "physical_shape")
        order = _normalize_permutation(permutation, len(physical))
        logical = [0] * len(physical)
        for physical_mode, logical_mode in enumerate(order):
            logical[logical_mode] = physical[physical_mode]
        return cls(
            tuple(logical), physical, parse_format(tensor_format), order, index_dtype
        )

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


@dataclass(frozen=True)
class TensorMetadata:
    """All non-payload runtime tensor metadata in one immutable value."""

    name: str
    dtype: torch.dtype
    device: torch.device
    layout: TensorLayout
    requires_grad: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TensorLayoutError("tensor name must be a non-empty string")
        dtype = _parse_dtype(self.dtype, "dtype")
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError) as error:
            raise TensorDeviceError(f"invalid tensor device {self.device!r}") from error
        if device.type != "cpu":
            raise TensorDeviceError(
                f"Scorch runtime tensors must be on CPU, got {device}"
            )
        if not isinstance(self.layout, TensorLayout):
            raise TensorTypeError("metadata layout must be a TensorLayout")
        validate_runtime_contract(self.layout.format, dtype)
        if not isinstance(self.requires_grad, bool):
            raise TensorTypeError("requires_grad must be a bool")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "device", device)

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
            if any(
                level_type == LevelType.SINGLETON
                for level_type in parsed_format.get_level_types()
            ):
                raise CompileSpecError(
                    "singleton levels are not supported by the compiler"
                )
            if dtype not in SUPPORTED_VALUE_DTYPES:
                raise CompileSpecError(
                    f"compiler tensor dtype {dtype} is not supported"
                )
            layout = TensorLayout.from_logical_shape(
                shape, parsed_format, mode_order, index_dtype
            )
            metadata = TensorMetadata(name, dtype, torch.device(device), layout, False)
        except (
            TensorLayoutError,
            TensorDeviceError,
            TensorTypeError,
            TensorValidationError,
            TypeError,
            RuntimeError,
        ) as error:
            raise CompileSpecError(str(error)) from error
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

    def with_mode_order(self, mode_order: Sequence[int]) -> "TensorSpec":
        new_layout = TensorLayout.from_logical_shape(
            self.logical_shape, self.format, mode_order, self.index_dtype
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
