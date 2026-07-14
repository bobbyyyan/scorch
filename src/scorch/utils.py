import hashlib
import importlib.util
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from importlib import resources
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import torch
from .compiler.llir import DataType
from .format import parse_format  # noqa: F401 - compatibility re-export

if TYPE_CHECKING:
    from .compiler.compile_options import CompileOptions, KernelBuildOptions

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


def _resolve_compile_options(
    compile_options: Optional["CompileOptions"] = None,
) -> "CompileOptions":
    """Resolve one typed snapshot at a direct utility compilation boundary."""

    from .compiler.compile_options import CompileOptions

    if compile_options is None:
        return CompileOptions.from_environment()
    if type(compile_options) is not CompileOptions:
        raise TypeError("compile_options must be a CompileOptions instance")
    return compile_options


def _kernel_name(
    *sources: str, compile_options: Optional["CompileOptions"] = None
) -> str:
    """Deterministic name from kernel source so torch's disk cache persists.

    Includes torch version in the hash so a PyTorch upgrade invalidates all
    cached .so files (they link against libtorch). ``jit_preamble_text`` expands
    the policy header into the source, so a local policy retune is covered too.
    """
    options = _resolve_compile_options(compile_options)
    keyed = "".join(sources) + options.build.torch_version
    if options.build.jit_tune_hooks:
        # Instrumented sweep kernels carry extra getenv branches (same text, -D flag),
        # so key them apart from clean kernels to avoid serving one for the other.
        keyed += "|scorch_tune_hooks"
    h = hashlib.md5(keyed.encode()).hexdigest()[:12]
    return f"kernel_{h}"


_so_cache: Dict[tuple[str, tuple[object, ...]], Any] = {}


@dataclass(frozen=True)
class _JITBuildRequest:
    """Detached, typed payload for one out-of-process Torch extension build."""

    name: str
    cpp_sources: Tuple[str, ...]
    functions: Tuple[str, ...]
    extra_cflags: Tuple[str, ...]
    extra_ldflags: Tuple[str, ...]
    build_directory: str
    build_options: "KernelBuildOptions"


def _validate_jit_build_request(request: object) -> _JITBuildRequest:
    from .compiler.compile_options import KernelBuildOptions

    if type(request) is not _JITBuildRequest:
        raise TypeError("JIT build payload must be an exact _JITBuildRequest")
    typed_request = request
    if type(typed_request.build_options) is not KernelBuildOptions:
        raise TypeError("JIT build payload must own exact KernelBuildOptions")
    string_fields = (
        ("name", typed_request.name),
        ("build_directory", typed_request.build_directory),
    )
    for field_name, string_value in string_fields:
        if type(string_value) is not str or not string_value:
            raise TypeError(f"JIT build {field_name} must be a non-empty string")
    sequence_fields = (
        ("cpp_sources", typed_request.cpp_sources),
        ("functions", typed_request.functions),
        ("extra_cflags", typed_request.extra_cflags),
        ("extra_ldflags", typed_request.extra_ldflags),
    )
    for field_name, sequence_value in sequence_fields:
        if type(sequence_value) is not tuple or any(
            type(item) is not str or not item for item in sequence_value
        ):
            raise TypeError(f"JIT build {field_name} must be an exact tuple of strings")
    if not typed_request.cpp_sources or not typed_request.functions:
        raise ValueError("JIT build sources and functions must be non-empty")
    if typed_request.extra_cflags not in (
        typed_request.build_options.extra_cflags,
        typed_request.build_options.special_kernel_cflags,
    ):
        raise ValueError("JIT build compiler flags disagree with KernelBuildOptions")
    if typed_request.extra_ldflags != typed_request.build_options.extra_ldflags:
        raise ValueError("JIT build linker flags disagree with KernelBuildOptions")
    if not os.path.isabs(typed_request.build_directory):
        raise ValueError("JIT build directory must be absolute")
    return typed_request


def _verify_snapshotted_build_runtime(request: _JITBuildRequest) -> None:
    """Fail closed if the build child does not match its frozen ABI snapshot."""

    import platform
    import sysconfig

    from .compiler.compile_options import CompilerWrapperPolicy

    options = request.build_options
    current_torch_path = torch.__file__
    current_torch_root = os.path.dirname(current_torch_path)
    current_torch_include = os.path.join(current_torch_root, "include")
    current_torch_include_paths = (
        current_torch_include,
        os.path.join(current_torch_include, "torch", "csrc", "api", "include"),
    )
    current_torch_library_path = os.path.join(current_torch_root, "lib")
    current_cache_tag = sys.implementation.cache_tag
    current_python_include = sysconfig.get_path("include", scheme="posix_prefix")
    observed = (
        ("torch_version", str(torch.__version__), options.torch_version),
        ("torch_path", current_torch_path, options.torch_path),
        (
            "torch_include_paths",
            current_torch_include_paths,
            options.torch_include_paths,
        ),
        (
            "torch_library_path",
            current_torch_library_path,
            options.torch_library_path,
        ),
        (
            "torch_cxx11_abi",
            bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            options.torch_cxx11_abi,
        ),
        ("python_executable", sys.executable, options.python_executable),
        ("python_version", platform.python_version(), options.python_version),
        ("python_cache_tag", current_cache_tag, options.python_cache_tag),
        (
            "python_include_path",
            current_python_include,
            options.python_include_path,
        ),
    )
    mismatches = [name for name, current, expected in observed if current != expected]

    wrapper_disabled = bool(os.environ.get("TORCH_NO_COMPILER_WRAPPER"))
    if options.compiler_wrapper_policy is CompilerWrapperPolicy.DISABLED:
        if not wrapper_disabled:
            mismatches.append("compiler_wrapper_policy")
    else:
        if wrapper_disabled:
            mismatches.append("compiler_wrapper_policy")
        observed_wrapper_name: Optional[str] = None
        observed_wrapper_path: Optional[str] = None
        for wrapper_name in ("ccache", "sccache"):
            resolved_wrapper = shutil.which(wrapper_name)
            if resolved_wrapper is not None:
                observed_wrapper_name = wrapper_name
                observed_wrapper_path = os.path.abspath(resolved_wrapper)
                break
        if observed_wrapper_name != options.compiler_wrapper_name:
            mismatches.append("compiler_wrapper_name")
        if observed_wrapper_path != options.compiler_wrapper_path:
            mismatches.append("compiler_wrapper_path")
    if mismatches:
        raise RuntimeError(
            "JIT build runtime differs from CompileOptions snapshot: "
            + ", ".join(mismatches)
        )


def _run_jit_build_request() -> None:
    """Child-process entry point; build without mutating the parent environment."""

    from torch.utils.cpp_extension import load_inline

    request = _validate_jit_build_request(pickle.load(sys.stdin.buffer))
    _verify_snapshotted_build_runtime(request)
    load_inline(
        name=request.name,
        cpp_sources=list(request.cpp_sources),
        functions=list(request.functions),
        extra_cflags=list(request.extra_cflags),
        extra_ldflags=list(request.extra_ldflags),
        build_directory=request.build_directory,
    )


def _request_cache_key(request: _JITBuildRequest) -> tuple[object, ...]:
    from .compiler.compile_options import canonical_cache_digest

    content_digest = canonical_cache_digest(
        (
            request.cpp_sources,
            request.functions,
            request.extra_cflags,
            request.extra_ldflags,
        )
    )
    return request.build_options.cache_key + (content_digest,)


def _build_identity_digest(
    cpp_sources: Sequence[str],
    functions: Sequence[str],
    extra_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    build_options: "KernelBuildOptions",
) -> str:
    from .compiler.compile_options import canonical_cache_digest

    return canonical_cache_digest(
        (
            tuple(cpp_sources),
            tuple(functions),
            tuple(extra_cflags),
            tuple(extra_ldflags),
            build_options.cache_key,
        )
    )


def _load_extension_file(name: str, so_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load compiled extension {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_and_load_extension(request: _JITBuildRequest) -> Any:
    """Compile in a child with frozen process state, then import the result."""

    request = _validate_jit_build_request(request)
    os.makedirs(request.build_directory, exist_ok=True)
    toolchain_directory = _prepare_compiler_toolchain(request)
    so_path = os.path.join(request.build_directory, f"{request.name}.so")
    if not os.path.isfile(so_path):
        subprocess.run(
            [
                request.build_options.python_executable,
                "-P",
                "-c",
                "from scorch.utils import _run_jit_build_request; "
                "_run_jit_build_request()",
            ],
            input=pickle.dumps(request),
            env=_jit_build_environment_from_request(
                request,
                toolchain_directory=toolchain_directory,
            ),
            check=True,
        )
    if not os.path.isfile(so_path):
        raise RuntimeError(f"JIT build did not produce expected extension {so_path}")
    return _load_extension_file(request.name, so_path)


def _prepare_compiler_toolchain(request: _JITBuildRequest) -> Optional[str]:
    """Pin a bare compiler spelling to the snapshotted absolute executable."""

    build_options = request.build_options
    if os.path.isabs(build_options.cxx_compiler):
        return None
    toolchain_directory = os.path.join(
        request.build_directory,
        ".scorch-toolchain",
    )
    os.makedirs(toolchain_directory, exist_ok=True)
    launcher = os.path.join(toolchain_directory, build_options.cxx_compiler)
    if not os.path.lexists(launcher):
        try:
            os.symlink(build_options.cxx_compiler_path, launcher)
        except FileExistsError:
            # A concurrent identical build may have installed the same launcher.
            pass
    if not os.path.islink(launcher) or os.readlink(launcher) != (
        build_options.cxx_compiler_path
    ):
        raise RuntimeError(
            "JIT compiler launcher conflicts with the CompileOptions snapshot"
        )
    return toolchain_directory


def _jit_build_environment_from_request(
    request: _JITBuildRequest,
    *,
    toolchain_directory: Optional[str],
) -> Dict[str, str]:
    """Build a child environment without requiring a second option snapshot."""

    build_options = request.build_options
    search_path = build_options.executable_search_path
    if toolchain_directory is not None:
        search_path = toolchain_directory + os.pathsep + search_path
    environment = {
        "PATH": search_path,
        "PYTHONPATH": build_options.scorch_python_path,
    }
    from .compiler.compile_options import CompilerWrapperPolicy

    if build_options.compiler_wrapper_policy is CompilerWrapperPolicy.DISABLED:
        environment["TORCH_NO_COMPILER_WRAPPER"] = "1"
    if build_options.cxx_compiler_from_environment:
        environment["CXX"] = build_options.cxx_compiler
    darwin_toolchain = build_options.darwin_toolchain
    if darwin_toolchain is not None:
        environment["DEVELOPER_DIR"] = darwin_toolchain.developer_dir
        environment["SDKROOT"] = darwin_toolchain.sdk_root
        if darwin_toolchain.deployment_target is not None:
            environment["MACOSX_DEPLOYMENT_TARGET"] = darwin_toolchain.deployment_target
    return environment


def _load_kernel(
    name: str,
    cpp_sources: Sequence[str],
    functions: Sequence[str],
    extra_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    *,
    compile_options: Optional["CompileOptions"] = None,
) -> Any:
    """Load a compiled kernel, using a persistent .so cache when possible.

    PyTorch's JIT_EXTENSION_VERSIONER is in-memory only, so load_inline
    always recompiles on the first call in each process (~7s).  We bypass
    it by checking if the .so already exists on disk and loading it directly.
    """
    options = _resolve_compile_options(compile_options)
    cpp_sources_snapshot = tuple(cpp_sources)
    functions_snapshot = tuple(functions)
    cflags_snapshot = tuple(extra_cflags)
    ldflags_snapshot = tuple(extra_ldflags)
    if cflags_snapshot != options.build.extra_cflags:
        raise ValueError("extra_cflags must match the CompileOptions snapshot")
    if ldflags_snapshot != options.build.extra_ldflags:
        raise ValueError("extra_ldflags must match the CompileOptions snapshot")

    from torch.utils.cpp_extension import _get_build_directory

    build_root = _get_build_directory(name, verbose=False)
    build_digest = _build_identity_digest(
        cpp_sources_snapshot,
        functions_snapshot,
        cflags_snapshot,
        ldflags_snapshot,
        options.build,
    )
    build_dir = os.path.join(build_root, f"scorch_{build_digest}")
    so_path = os.path.join(build_dir, f"{name}.so")
    request = _validate_jit_build_request(
        _JITBuildRequest(
            name=name,
            cpp_sources=cpp_sources_snapshot,
            functions=functions_snapshot,
            extra_cflags=cflags_snapshot,
            extra_ldflags=ldflags_snapshot,
            build_directory=build_dir,
            build_options=options.build,
        )
    )
    cache_key = (name, _request_cache_key(request))
    if cache_key in _so_cache:
        return _so_cache[cache_key]

    if os.path.isfile(so_path):
        module = _load_extension_file(name, so_path)
    else:
        module = _build_and_load_extension(request)

    _so_cache[cache_key] = module
    return module


def get_extra_cflags(
    base_flags: Optional[List[str]] = None,
    *,
    compile_options: Optional["CompileOptions"] = None,
) -> List[str]:
    """Get platform-specific extra compiler flags for torch cpp_extension.

    On macOS, adds the C++ standard library include path needed for compilation.

    Args:
        base_flags: Base compiler flags to include. Defaults to ["-O3"].

    Returns:
        List of compiler flags including platform-specific additions.
    """
    options = _resolve_compile_options(compile_options)
    default_base = ("-O3", "-march=native", "-ffast-math", "-funroll-loops")
    if base_flags is None:
        return list(options.build.direct_extension_cflags)

    # Compatibility callers may replace only the optimization prefix. Target,
    # OpenMP, and instrumentation flags still come from the same snapshot.
    return list(base_flags) + list(
        options.build.direct_extension_cflags[len(default_base) :]
    )


def get_extra_ldflags(
    *, compile_options: Optional["CompileOptions"] = None
) -> List[str]:
    """Get platform-specific extra linker flags for torch cpp_extension.

    On macOS, links against PyTorch's bundled libomp to avoid runtime conflicts
    with Homebrew's libomp.

    Returns:
        List of linker flags.
    """
    options = _resolve_compile_options(compile_options)
    return list(options.build.extra_ldflags)


def load_to_kernel_cache(
    kernel_name: str,
    kernel_cache: Dict,
    kernel_code_filename: Optional[str],
    *,
    compile_options: Optional["CompileOptions"] = None,
) -> None:
    """Load a kernel to the kernel cache.

    Args:
        kernel_name (str): Name of the kernel.
        kernel_cache (Dict): Kernel cache.
        kernel_code_filename (str): Filename of the kernel code.
    """

    if kernel_code_filename is None:
        kernel_code_filename = f"{kernel_name}.cpp"

    options = _resolve_compile_options(compile_options)
    header_cpp_code = options.build.preamble_source
    cpp_code = native_resource_text(kernel_code_filename)
    build_name = kernel_name

    # Load special kernels. Their historical no-signed-zeros policy is distinct
    # from generated kernels and therefore has its own frozen flag tuple.
    start_time = time.time()
    from torch.utils.cpp_extension import _get_build_directory

    build_root = _get_build_directory(build_name, verbose=False)
    build_digest = _build_identity_digest(
        (header_cpp_code, cpp_code),
        ("evaluate",),
        options.build.special_kernel_cflags,
        options.build.extra_ldflags,
        options.build,
    )
    request = _validate_jit_build_request(
        _JITBuildRequest(
            name=build_name,
            cpp_sources=(header_cpp_code, cpp_code),
            functions=("evaluate",),
            extra_cflags=options.build.special_kernel_cflags,
            extra_ldflags=options.build.extra_ldflags,
            build_directory=os.path.join(build_root, f"scorch_{build_digest}"),
            build_options=options.build,
        )
    )
    cache_key = (build_name, _request_cache_key(request))
    if cache_key in _so_cache:
        module = _so_cache[cache_key]
    else:
        module = _build_and_load_extension(request)
        _so_cache[cache_key] = module
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
