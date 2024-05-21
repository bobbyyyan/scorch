import math
from collections import defaultdict, deque
from itertools import chain
from pathlib import Path
from typing import List, Dict, Any, Iterable, Union, Optional

import torch
from torch.utils.cpp_extension import load_inline, load

from .compiler.llir import DataType
from .format import TensorFormat, LevelFormat, LevelType

PROJECT_ROOT_DIR = Path(__file__)
while not (PROJECT_ROOT_DIR / "setup.py").exists():
    PROJECT_ROOT_DIR = PROJECT_ROOT_DIR.parent

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
        # extra_cflags=["-O3", "-march=native", "-ffast-math", "-fno-signed-zeros"],
        extra_cflags=["-O3", "-ffast-math", "-fno-signed-zeros"],
        build_directory=PROJECT_ROOT_DIR / "build",
    )
    end_time = time.time()
    print(f"Loading {kernel_name} took {end_time - start_time} s")

    kernel_cache[kernel_name] = module


def topo_sort_characters(substrings, tensors):
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

    def topo_sort():
        zero_in_degree_nodes = [node for node in nodes if in_degree[node] == 0]
        zero_in_degree_nodes = deque(zero_in_degree_nodes)

        result_ = []
        while zero_in_degree_nodes:
            node = zero_in_degree_nodes.popleft()
            result_.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree_nodes.append(neighbor)

            # Sort newly zero in-degree nodes according to priority
            zero_in_degree_nodes = list(zero_in_degree_nodes)
            zero_in_degree_nodes = deque(zero_in_degree_nodes)
        return result_

    # Run topological sort
    result = topo_sort()

    new_substrings = {}
    if len(result) < len(nodes):
        inverted_edges = resolve_cycles(nodes, graph, in_degree, substrings, tensors)
        # Re-run topo_sort with new graph, in_degree
        result = topo_sort()

        if len(result) < len(nodes):
            raise ValueError("resolve_cycles did not resolve cycles!")

        for tensor_index, edges_to_invert in inverted_edges.items():
            tensor_index_var_strs = list(substrings[tensor_index])
            tensor_edges = [(substrings[tensor_index][i], substrings[tensor_index][i + 1])
                            for i in range(len(substrings[tensor_index]) - 1)]
            for edge in edges_to_invert:
                tensor_edges.remove((edge[0], edge[1]))
                tensor_edges.append((edge[1], edge[0]))

            adj_list = {}
            reverse_adj_list = {}
            for start_edge, end_edge in tensor_edges:
                adj_list[start_edge] = end_edge
                reverse_adj_list[end_edge] = start_edge

            start_node = next(node for node in tensor_index_var_strs if node not in reverse_adj_list)
            new_substr = []
            current_node = start_node

            while current_node in adj_list:
                new_substr.append(current_node)
                current_node = adj_list[current_node]
            new_substr.append(current_node)

            new_substrings[tensor_index] = ''.join(new_substr)

    return result, new_substrings


def resolve_cycles(nodes, graph, in_degree, substrings, tensors):
    # Finds a cycle in the graph (if any) and then returns cycle's edges
    def find_cycle():
        visited = set()
        stack = []
        in_stack = set()

        def dfs(curr_node):
            visited.add(curr_node)
            stack.append(curr_node)
            in_stack.add(curr_node)

            for neighbor in graph[curr_node]:
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:  # If cycle edges were found, propagate them up
                        return result
                elif neighbor in in_stack:
                    # Cycle found, collect the cycle edges
                    cycle_start = stack.index(neighbor)
                    edges = [(stack[i], stack[i + 1]) for i in range(cycle_start, len(stack) - 1)]
                    edges.append((stack[-1], neighbor))
                    return edges

            stack.pop()
            in_stack.remove(curr_node)
            return []

        for node in nodes:
            if node not in visited:
                cycle_edges_ = dfs(node)
                if cycle_edges_:
                    return cycle_edges_

        return []

    # Finds the cheapest edge to invert based on tensor shape and already inverted edges
    def invert_cheapest_edge(edges):
        inverted_tensor_indices_set = {idx for edge in inverted_edges for idx in inverted_edge_to_tensor_indices[edge]}
        inverted_tensor_indices = list(inverted_tensor_indices_set)

        # Find edge associated with the smallest tensor, tiebreaker goes to result tensor
        def edge_cost(edge):
            tensor_indices_ = edges_to_tensor_indices[edge]
            return (0, -max(tensor_indices_)) \
                if set(tensor_indices_).issubset(set(inverted_tensor_indices)) \
                else (sum(tensor_index_to_size[index] for index in tensor_indices_), -max(tensor_indices_))

        min_cost_edge = min(edges, key=edge_cost)
        min_cost_tensor_indices = edges_to_tensor_indices[min_cost_edge]
        inverted_edge_to_tensor_indices[min_cost_edge] = min_cost_tensor_indices
        return min_cost_edge, min_cost_tensor_indices

    inverted_edges = defaultdict(list)
    inverted_edge_to_tensor_indices = {}
    edges_to_tensor_indices = defaultdict(list)

    # Build dictionary from edges in graph to indices of tensors they appear in
    for tensor_index, substring in enumerate(substrings):
        for i in range(len(substring) - 1):
            edges_to_tensor_indices[(substring[i], substring[i + 1])].append(tensor_index)

    # Build dictionary from tensor index to size using tensor shapes for operands and result
    tensor_index_to_size = {}
    for i in range(len(tensors)):
        tensor_index_to_size[i] = math.prod(tensors[i].shape)

    result_size = 1
    for char in substrings[-1]:
        shape_index = substrings[0].find(char)
        if shape_index == -1:
            shape_index = substrings[1].find(char)
            result_size *= tensors[1].shape[shape_index]
        else:
            result_size *= tensors[0].shape[shape_index]

    tensor_index_to_size[len(tensors)] = result_size

    # Loop while there are cycles, resolving them by inverting edges and storing them in inverted_edges
    while True:
        cycle_edges = find_cycle()
        if not cycle_edges:
            break
        edge_to_invert, tensor_indices = invert_cheapest_edge(cycle_edges)
        graph[edge_to_invert[0]].remove(edge_to_invert[1])
        graph[edge_to_invert[1]].append(edge_to_invert[0])
        in_degree[edge_to_invert[0]] += 1
        in_degree[edge_to_invert[1]] -= 1
        for tensor_index in tensor_indices:
            inverted_edges[tensor_index].append(edge_to_invert)

    return inverted_edges


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
