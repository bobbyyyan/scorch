import functools
import glob
import hashlib
import math
import os
import platform
import subprocess
import time
from collections import defaultdict, deque
from importlib import resources
from itertools import chain
from typing import List, Dict, Any, Iterable, Optional

import torch
from torch.utils.cpp_extension import load_inline

from .compiler.llir import DataType
from .format import parse_format  # noqa: F401 - compatibility re-export

_NATIVE_RESOURCES = resources.files("scorch").joinpath("csrc")


def native_resource_text(filename: str) -> str:
    """Read a packaged native source or header as UTF-8 text."""
    if filename in {"", ".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError("filename must name a file directly under scorch/csrc")
    return _NATIVE_RESOURCES.joinpath(filename).read_text(encoding="utf-8")


def _policy_header_text() -> str:
    """Return packaged policy text, with optional local tuning overrides first."""
    base = native_resource_text("scorch_policy.h")
    tuned = _NATIVE_RESOURCES.joinpath("scorch_policy_tuned.h")
    tuned_text = tuned.read_text(encoding="utf-8") if tuned.is_file() else ""
    return tuned_text + base


def jit_preamble_text() -> str:
    """Return the complete packaged C++ preamble used by generated kernels.

    ``load_inline`` writes its source into a separate cache directory, so quote
    includes in ``header.h`` cannot reliably find adjacent installed headers.
    Expanding the packaged policy header into the template keeps each generated
    translation unit self-contained and also works for non-filesystem resource
    loaders.
    """
    template = native_resource_text("header.h")
    abi_include = '#include "native_abi.h"'
    if template.count(abi_include) != 1:
        raise RuntimeError("packaged header.h must include native_abi.h once")
    template = template.replace(abi_include, native_resource_text("native_abi.h"), 1)

    include = '#include "scorch_policy.h"'
    if template.count(include) != 1:
        raise RuntimeError("packaged header.h must include scorch_policy.h once")
    return template.replace(include, _policy_header_text(), 1)


def _kernel_name(*sources: str) -> str:
    """Deterministic name from kernel source so torch's disk cache persists.

    Includes torch version in the hash so a PyTorch upgrade invalidates all
    cached .so files (they link against libtorch). ``jit_preamble_text`` expands
    the policy header into the source, so a local policy retune is covered too.
    """
    keyed = "".join(sources) + torch.__version__
    if os.environ.get("SCORCH_JIT_TUNE_HOOKS"):
        # Instrumented sweep kernels carry extra getenv branches (same text, -D flag),
        # so key them apart from clean kernels to avoid serving one for the other.
        keyed += "|scorch_tune_hooks"
    h = hashlib.md5(keyed.encode()).hexdigest()[:12]
    return f"kernel_{h}"


_so_cache: dict = {}


def _load_kernel(name: str, cpp_sources, functions, extra_cflags, extra_ldflags):
    """Load a compiled kernel, using a persistent .so cache when possible.

    PyTorch's JIT_EXTENSION_VERSIONER is in-memory only, so load_inline
    always recompiles on the first call in each process (~7s).  We bypass
    it by checking if the .so already exists on disk and loading it directly.
    """
    if name in _so_cache:
        return _so_cache[name]

    import importlib.util
    from torch.utils.cpp_extension import _get_build_directory, load_inline

    build_dir = _get_build_directory(name, verbose=False)
    so_path = os.path.join(build_dir, f"{name}.so")

    if os.path.isfile(so_path):
        # .so exists — load directly without invoking ninja
        spec = importlib.util.spec_from_file_location(name, so_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = load_inline(
            name=name,
            cpp_sources=cpp_sources,
            functions=functions,
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
        )

    _so_cache[name] = module
    return module


@functools.lru_cache(maxsize=1)
def _macos_libcxx_include() -> Optional[str]:
    """The libc++ headers belonging to the SDK the active toolchain will use.

    They have to come from the *active* developer directory. Pinning them to
    CommandLineTools, which this used to do, breaks every generated kernel on a host
    whose ``xcode-select -p`` is Xcode: the compiler torch invokes is then Xcode's
    clang, and handing it a different toolchain's libc++ fails inside the headers
    themselves ("reference to unresolved using declaration" in
    ``<__type_traits/is_trivially_copyable.h>``). The prebuilt extension was never
    affected because ``scorch_build.py`` does not add this flag, so the breakage
    looked like a codegen defect rather than a toolchain one.

    ``SCORCH_MACOS_SDK`` overrides the lookup. ``None`` means no candidate has the
    headers, and then no flag is added at all -- clang's own default include path is
    correct on a consistent toolchain, and a wrong ``-isystem`` is worse than none.

    Cached because ``get_extra_cflags`` runs once per generic ``matmul``/``einsum``
    call, and this spawns a process. A mid-process ``xcode-select`` switch therefore
    is not picked up; set ``SCORCH_MACOS_SDK`` if that is ever wanted.
    """
    override = os.environ.get("SCORCH_MACOS_SDK")
    if override:
        candidates = [override]
    else:
        candidates = []
        try:
            found = subprocess.run(
                ["xcrun", "--show-sdk-path"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if found.returncode == 0:
                candidates.append(found.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass  # no xcrun, or it hung: fall through to the fixed location
        candidates.append("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")

    for sdk in candidates:
        if not sdk:
            continue
        include = os.path.join(sdk, "usr", "include", "c++", "v1")
        if os.path.isdir(include):
            return include
    return None


def get_extra_cflags(base_flags: Optional[List[str]] = None) -> List[str]:
    """Get platform-specific extra compiler flags for torch cpp_extension.

    On macOS, adds the active SDK's C++ standard library include path needed
    for compilation, plus the OpenMP flags.

    Args:
        base_flags: Base compiler flags to include. Defaults to ["-O3"].

    Returns:
        List of compiler flags including platform-specific additions.
    """
    if base_flags is None:
        base_flags = ["-O3", "-march=native", "-ffast-math", "-funroll-loops"]
    flags = list(base_flags)

    if platform.system() == "Darwin":
        # macOS needs an explicit C++ stdlib include path for torch's JIT compile,
        # and it must be the active toolchain's (see _macos_libcxx_include).
        libcxx = _macos_libcxx_include()
        if libcxx is not None:
            flags.append(f"-isystem{libcxx}")

        # OpenMP flags for macOS - use PyTorch's bundled libomp to avoid runtime conflicts
        flags.extend(["-Xpreprocessor", "-fopenmp"])

        # Add OpenMP header path from Homebrew (headers only)
        for header_path in [
            "/opt/homebrew/opt/libomp/include",
            "/usr/local/opt/libomp/include",
        ]:
            if os.path.exists(header_path):
                flags.append(f"-I{header_path}")
                break
    else:
        # Linux: standard OpenMP support
        flags.append("-fopenmp")

    # Install-time autotune: build JIT kernels with the SCORCH_TUNE_HOOKS sweep hooks
    # so the codegen thread/chunk policy is tunable in-process too (mirrors the
    # native build's SCORCH_BUILD_TUNE_HOOKS mode). Off in the shipped path.
    if os.environ.get("SCORCH_JIT_TUNE_HOOKS"):
        flags.append("-DSCORCH_TUNE_HOOKS")

    return flags


def get_extra_ldflags() -> List[str]:
    """Get platform-specific extra linker flags for torch cpp_extension.

    On macOS, links against PyTorch's bundled libomp to avoid runtime conflicts
    with Homebrew's libomp.

    Returns:
        List of linker flags.
    """
    ldflags = []

    if platform.system() == "Darwin":
        # Link against PyTorch's bundled libomp to avoid runtime conflicts
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
        torch_omp = os.path.join(torch_lib_path, "libomp.dylib")

        if os.path.exists(torch_omp):
            # Use full path to avoid linker finding Homebrew's libomp
            ldflags.append(torch_omp)
            # Add rpath so it finds the right library at runtime
            ldflags.append(f"-Wl,-rpath,{torch_lib_path}")
        else:
            # Fall back to Homebrew's libomp
            for lib_path in [
                "/opt/homebrew/opt/libomp/lib",
                "/usr/local/opt/libomp/lib",
            ]:
                if os.path.exists(lib_path):
                    ldflags.extend(["-lomp", f"-L{lib_path}"])
                    break
            else:
                ldflags.append("-lomp")
    else:
        # Linux: link against PyTorch's bundled libgomp to avoid
        # dual-runtime conflicts (PyTorch's copy vs system copy)
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
        gomp_libs = glob.glob(os.path.join(torch_lib_path, "libgomp*.so*"))

        if gomp_libs:
            ldflags.extend([gomp_libs[0], f"-Wl,-rpath,{torch_lib_path}"])
        else:
            # Fallback: use system OpenMP (no bundled libgomp = no conflict)
            ldflags.append("-fopenmp")

    return ldflags


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

    header_cpp_code = jit_preamble_text()
    cpp_code = native_resource_text(kernel_code_filename)

    # Load special kernels
    start_time = time.time()
    module = load_inline(
        name=kernel_name,
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=get_extra_cflags(
            ["-O3", "-march=native", "-ffast-math", "-fno-signed-zeros"]
        ),
        extra_ldflags=get_extra_ldflags(),
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

    if len(result) < len(nodes):
        resolve_cycles(nodes, graph, in_degree, substrings, tensors)
        # Re-run topo_sort with new graph, in_degree
        result = topo_sort()

        if len(result) < len(nodes):
            raise ValueError("resolve_cycles did not resolve cycles!")

    return result


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
                    edges = [
                        (stack[i], stack[i + 1])
                        for i in range(cycle_start, len(stack) - 1)
                    ]
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
        inverted_tensor_indices = list(inverted_edges.keys())

        # Find edge associated with the smallest tensor, tiebreaker goes to result tensor. Represents cost as a tuple.
        def edge_cost(edge):
            edge_tensor_indices = edges_to_tensor_indices[edge]
            return (
                (0, -(max(edge_tensor_indices)))
                if set(edge_tensor_indices).issubset(set(inverted_tensor_indices))
                else (
                    sum(tensor_index_to_size[index] for index in edge_tensor_indices),
                    -max(edge_tensor_indices),
                )
            )

        min_cost_edge = min(edges, key=edge_cost)
        min_cost_tensor_indices = edges_to_tensor_indices[min_cost_edge]
        return min_cost_edge, min_cost_tensor_indices

    inverted_edges = defaultdict(list)
    edges_to_tensor_indices = defaultdict(list)

    # Build dictionary from edges (tuples) in graph to indices of tensors they appear in
    for tensor_index, substring in enumerate(substrings):
        for i in range(len(substring) - 1):
            edges_to_tensor_indices[(substring[i], substring[i + 1])].append(
                tensor_index
            )

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
        in_degree[edge_to_invert[1]] -= 1

        # If the cycle is just two edges, we will remove one edge, not invert it
        if edge_to_invert[0] not in graph[edge_to_invert[1]]:
            graph[edge_to_invert[1]].append(edge_to_invert[0])
            in_degree[edge_to_invert[0]] += 1

        for tensor_index in tensor_indices:
            inverted_edges[tensor_index].append(edge_to_invert)


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
