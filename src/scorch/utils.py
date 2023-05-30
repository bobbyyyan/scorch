from itertools import chain
from pathlib import Path
from typing import List, Dict, Any, Iterable, Union

import torch

from .format import TensorFormat, LevelFormat, LevelType
from .compiler.llir import DataType

PROJECT_ROOT_DIR = Path(__file__)
while not (PROJECT_ROOT_DIR / "setup.py").exists():
    PROJECT_ROOT_DIR = PROJECT_ROOT_DIR.parent


def parse_format(fmt: Union[List[str], str, TensorFormat]) -> TensorFormat:
    """Convert a list of format strings to a TensorFormat.

    Args:
        fmt (List[str]): List of format strings.

    Returns:
        TensorFormat: TensorFormat object.
    """
    if isinstance(fmt, TensorFormat):
        return fmt
    if isinstance(fmt, str):
        fmt = list(fmt)
    level_formats = []
    for format_str in fmt:
        if format_str in ["dense", "d"]:
            level_formats.append(LevelFormat(mode=LevelType.DENSE))
        elif format_str in ["compressed", "sparse", "c", "s"]:
            level_formats.append(LevelFormat(mode=LevelType.COMPRESSED))
        elif format_str in ["coordinate", "coord", "o"]:
            level_formats.append(LevelFormat(mode=LevelType.COORDINATE))
        else:
            raise ValueError(f"Invalid format string: {format_str}")
    return TensorFormat(level_formats=level_formats)


PYTORCH_DTYPE_TO_C_PYTORCH_DTYPE: Dict[torch.dtype, str] = {
    torch.float32: "torch::kFloat32",
    torch.float64: "torch::kFloat64",
    torch.int32: "torch::kInt32",
    torch.int64: "torch::kInt64",
    torch.int8: "torch::kInt8",
    torch.uint8: "torch::kUInt8",
}

PYTORCH_DTYPE_TO_DATATYPE: Dict[torch.dtype, DataType] = {
    torch.float32: DataType.TORCH_FLOAT32,
    torch.float64: DataType.TORCH_FLOAT64,
    torch.int32: DataType.TORCH_INT32,
    torch.int64: DataType.TORCH_INT64,
    torch.int8: DataType.TORCH_INT8,
    torch.uint8: DataType.TORCH_UINT8,
}

PYTORCH_DTYPE_TO_C_DATATYPE: Dict[torch.dtype, DataType] = {
    torch.float32: DataType.FLOAT32,
    torch.float64: DataType.FLOAT64,
    torch.int32: DataType.INT32,
    torch.int64: DataType.INT64,
    torch.int8: DataType.INT8,
    torch.uint8: DataType.UINT8,
}


def dtype_to_c_datatype(dtype: torch.dtype) -> DataType:
    """Convert a pytorch dtype to a C++ DataType.

    Args:
        dtype (torch.dtype): Pytorch dtype.

    Returns:
        DataType: C++ DataType object.
    """
    return PYTORCH_DTYPE_TO_C_DATATYPE[dtype]


def dtype_to_datatype(dtype: torch.dtype) -> DataType:
    """Convert a pytorch dtype to a DataType.

    Args:
        dtype (torch.dtype): Pytorch dtype.

    Returns:
        DataType: DataType object.
    """
    return PYTORCH_DTYPE_TO_DATATYPE[dtype]


def get_pytorch_c_dtype_str(dtype: torch.dtype) -> str:
    """Get the C++ pytorch dtype string for a given pytorch dtype.

    Args:
        dtype (torch.dtype): Pytorch dtype.

    Returns:
        str: C++ pytorch dtype string.
    """
    return PYTORCH_DTYPE_TO_C_PYTORCH_DTYPE[dtype]


def flatten_2d_list(lst: Iterable[List[Any]]) -> List[Any]:
    return list(chain(*lst))
