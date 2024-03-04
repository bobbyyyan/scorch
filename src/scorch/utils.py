from collections import defaultdict, deque
from itertools import chain
from pathlib import Path
from typing import List, Dict, Any, Iterable, Union, Optional

import torch
from torch.utils.cpp_extension import load_inline, load

from .compiler.llir import DataType
from .format import TensorFormat, LevelFormat, LevelType

PROJECT_ROOT_DIR = Path(__file__)
# while not (PROJECT_ROOT_DIR / "setup.py").exists():
#     PROJECT_ROOT_DIR = PROJECT_ROOT_DIR.parent

import time


def load_to_kernel_cache(
    kernel_name: str, kernel_cache: Dict, kernel_code_filename: Optional[str]
) -> None:
    """Load a kernel to the kernel cache.

    Args:
        kernel_name (str): Name of the kernel.
        kernel_cache (Dict): Kernel cache.
        kernel_code_filename (str): Filename of the kernel code.
    """

    if kernel_code_filename is None:
        kernel_code_filename = f"{kernel_name}.cpp"

    # Read header_cpp_code from csrc/header.cpp
    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    with open(PROJECT_ROOT_DIR / f"csrc/{kernel_code_filename}", "r") as f:
        cpp_code = f.read()

    # Load special kernels
    start_time = time.time()
    module = load_inline(
        name=kernel_name,
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=["-O3", "-march=native", "-ffast-math", "-fno-signed-zeros"],
        build_directory=PROJECT_ROOT_DIR / "build",
    )
    end_time = time.time()
    print(f"Loading {kernel_name} took {end_time - start_time} s")

    kernel_cache[kernel_name] = module


def topo_sort_characters(s, priority=""):
    # Split the string into substrings
    substrings = s.split(",")

    # Create a directed graph
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    nodes = set()

    # Initialize in_degree for all characters
    for substring in substrings:
        for char in substring:
            nodes.add(char)
            if char not in in_degree:
                in_degree[char] = 0

    # Add edges to the graph
    for substring in substrings:
        for i in range(len(substring) - 1):
            if substring[i + 1] not in graph[substring[i]]:
                graph[substring[i]].append(substring[i + 1])
                in_degree[substring[i + 1]] += 1

    # Run topological sort
    zero_in_degree_nodes = [node for node in nodes if in_degree[node] == 0]
    zero_in_degree_nodes.sort(
        key=lambda x: priority.index(x) if x in priority else float("inf")
    )
    zero_in_degree_nodes = deque(zero_in_degree_nodes)

    result = []

    while zero_in_degree_nodes:
        node = zero_in_degree_nodes.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_nodes.append(neighbor)

        # Sort newly zero in-degree nodes according to priority
        zero_in_degree_nodes = list(zero_in_degree_nodes)
        zero_in_degree_nodes.sort(
            key=lambda x: priority.index(x) if x in priority else float("inf")
        )
        zero_in_degree_nodes = deque(zero_in_degree_nodes)

    if len(result) < len(nodes):
        # The graph contains a cycle, so it's not possible to sort the nodes
        raise ValueError("The input string contains a contradiction.")

    return result


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
