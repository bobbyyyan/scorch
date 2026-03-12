import time
from pathlib import Path
from typing import Any, Union, Sequence, Optional, List

import torch

from .compiler.cin import (
    IndexVar,
    TensorVar,
    ForAll,
    Workspace,
    Where,
    TensorAssign,
    Operation,
    IndexStmt,
)
from .compiler.cin_lowerer import CINLowerer
from .compiler.codegen import LLIRLowerer
from .compiler.scheduler import Scheduler
from .format import TensorFormat, LevelFormat, LevelType
from .prebuilt_kernels import execute_prebuilt_binary_kernel, resolve_prebuilt_matmul
from .storage import TensorIndex
from .stensor import STensor
from .utils import parse_format, topo_sort_characters, load_to_kernel_cache, get_extra_cflags, get_extra_ldflags, _kernel_name, _load_kernel

PROJECT_ROOT_DIR = Path(__file__)
while not (PROJECT_ROOT_DIR / "setup.py").exists():
    PROJECT_ROOT_DIR = PROJECT_ROOT_DIR.parent

_kernel_cache = {}
_einsum_dispatch_cache = {}

# start_time = time.time()
# # Register custom classes
# load(
#     name="pybind",
#     sources=[str(PROJECT_ROOT_DIR / "csrc/pybind.cpp")],
#     build_directory=PROJECT_ROOT_DIR / "build",
# )
# end_time = time.time()
# compile_time = end_time - start_time
# print(f"Pybind load time: {compile_time:.5f} seconds")
#
# load_to_kernel_cache("spmm_csr", _kernel_cache, "spmm-csr.cpp")
# load_to_kernel_cache("spmm_csr_ones", _kernel_cache, "spmm-csr-ones.cpp")


def spmv(
    a: STensor,
    b: STensor,
    output_format: Optional[Union[TensorFormat, str, List[str]]] = None,
    **kwargs,
) -> STensor:
    if output_format is None:
        output_format = parse_format("d")
    elif not isinstance(output_format, TensorFormat):
        output_format = parse_format(output_format)

    y = TensorVar("y", fmt=output_format)
    A = TensorVar("A", fmt=a.format)
    x = TensorVar("x", fmt=b.format)

    i = IndexVar("i")
    j = IndexVar("j")

    workspace = Workspace(
        name="wksp",
        dim=0,
    )

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                j,
                TensorAssign(
                    workspace.get_default_access(), A[i, j] * x[j], op=Operation.ADD
                ),
            ),
            consumer=TensorAssign(
                y[i],
                workspace.get_default_access(),
            ),
        ),
    )

    lowerer = CINLowerer()
    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
    llir_lowerer = LLIRLowerer()
    cpp_code = llir_lowerer.lower_llir(lowered_llir)

    # Read header_cpp_code from csrc/header.cpp
    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    # start_time = time.time()
    module = _load_kernel(
        name=_kernel_name(header_cpp_code, cpp_code),
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=get_extra_cflags(),
        extra_ldflags=get_extra_ldflags(),
    )
    # end_time = time.time()

    # compile_time = end_time - start_time
    #  Print kernel compile time to 5 decimal places
    # print(f"Kernel compile time: {compile_time:.5f} seconds")

    result_shape = (a.shape[0],)
    args = [result_shape]

    for tensor in [a, b]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor.index.mode_indices)  # type: ignore
        args.append(tensor.values)  # type: ignore

    start_time = time.time()
    result_cpp = module.evaluate(*args)
    end_time = time.time()
    eval_time = end_time - start_time
    if "time_dict" in kwargs:
        time_dict = kwargs["time_dict"]
        time_dict["eval_time"] = eval_time
    # m

    result = STensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp.storage.index.mode_indices,
            tensor_format=output_format,
        ),
        value=result_cpp.storage.value,
    )

    return result


def matmul_wksp(
    a: Union[torch.Tensor, STensor],
    b: Union[torch.Tensor, STensor],
    output_format: Optional[Union[TensorFormat, str, List[str]]] = None,
    **kwargs,
) -> STensor:
    if isinstance(a, torch.Tensor):
        a = STensor.from_torch(a).to_sparse()
    if isinstance(b, torch.Tensor):
        b = STensor.from_torch(b).to_sparse()

    if output_format is None:
        output_format = parse_format("ds")
    elif not isinstance(output_format, TensorFormat):
        output_format = parse_format(output_format)

    C = TensorVar("C", fmt=output_format)
    A = TensorVar("A", fmt=a.format)
    B = TensorVar("B", fmt=b.format)

    workspace = Workspace(
        name="wksp",
        dim=1,
    )

    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[j],
                        A[i, k] * B[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    C[i, j],
                    workspace[j],
                ),
            ),
        ),
    )

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    cpp_code = llir_lowerer.lower_llir(lowered_llir)

    # print("\n C++ CODE:\n")
    # print(cpp_code)

    # Read header_cpp_code from csrc/header.cpp
    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    # start_time = time.time()
    module = _load_kernel(
        name=_kernel_name(header_cpp_code, cpp_code),
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=get_extra_cflags(),
        extra_ldflags=get_extra_ldflags(),
    )
    # end_time = time.time()

    # compile_time = end_time - start_time
    #  Print kernel compile time to 5 decimal places
    # print(f"Kernel compile time: {compile_time:.5f} seconds")

    result_shape = (a.shape[0], b.shape[1])
    args = [result_shape]

    for tensor in [a, b]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor.index.mode_indices)  # type: ignore
        args.append(tensor.values)  # type: ignore

    start_time = time.time()
    result_cpp = module.evaluate(*args)
    end_time = time.time()
    eval_time = end_time - start_time
    if "time_dict" in kwargs:
        time_dict = kwargs["time_dict"]
        time_dict["eval_time"] = eval_time
    # print("Time taken for evaluate:", eval_time)

    result = STensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp.storage.index.mode_indices,
            tensor_format=output_format,
        ),
        value=result_cpp.storage.value,
    )

    return result


def matmul(
    a: Union[torch.Tensor, STensor],
    b: Union[torch.Tensor, STensor],
    **kwargs: Any,
) -> Union[torch.Tensor, STensor]:
    """Matrix multiplication."""

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.is_sparse and b.is_sparse and a.layout == torch.sparse_coo and b.layout == torch.sparse_coo:
            a = a.to_sparse_csr()
            b = b.to_sparse_csr()
        if a.is_sparse or a.is_sparse_csr or b.is_sparse or b.is_sparse_csr:
            a = STensor.from_torch(a)
            b = STensor.from_torch(b)
        else:
            return torch.matmul(a, b)

    if isinstance(a, torch.Tensor):
        a = STensor.from_torch(a)
    if isinstance(b, torch.Tensor):
        b = STensor.from_torch(b)

    # Dense STensor x dense STensor: use torch dense matmul directly.
    # This is both faster and more reliable than lowering through sparse
    # scheduling paths for fully dense operands.
    if a.format.is_dense() and b.format.is_dense():
        start_time = time.time()
        result_torch = torch.matmul(
            a.to_torch(in_place=False),
            b.to_torch(in_place=False),
        )
        end_time = time.time()
        eval_time = end_time - start_time
        if "time_dict" in kwargs:
            time_dict = kwargs["time_dict"]
            time_dict["eval_time"] = eval_time

        output_format_kw = kwargs.get("format", None)
        if output_format_kw is None:
            return result_torch

        output_format = parse_format(output_format_kw)
        if output_format.is_dense():
            return result_torch

        return STensor.from_torch(result_torch).to_sparse(output_format)

    use_cache = kwargs.get("use_cache", True)
    time_dict = kwargs.get("time_dict", None)
    requested_output_format = kwargs.get("format", kwargs.get("output_format", None))

    if a.dim() == 2 and b.dim() == 1:
        default_mode_order = [0, 1]
        if (not a.format.is_dense()) and a.storage.index.mode_order != default_mode_order:
            a = a.copy()
            a.change_mode_order(default_mode_order)

        if use_cache:
            resolved = resolve_prebuilt_matmul(
                a, b, output_format=requested_output_format
            )
            if resolved is not None:
                result_cpp, result_shape = execute_prebuilt_binary_kernel(
                    resolved.fn, a, b, time_dict=time_dict
                )
                result = STensor(
                    shape=result_shape,
                    index=TensorIndex(
                        mode_indices=result_cpp.storage.index.mode_indices,
                        tensor_format=resolved.output_format,
                    ),
                    value=result_cpp.storage.value,
                )
                if result.format.is_dense():
                    return result.to_torch()
                return result

        spmv_kwargs = dict(kwargs)
        if "format" in spmv_kwargs and "output_format" not in spmv_kwargs:
            spmv_kwargs["output_format"] = spmv_kwargs.pop("format")
        return spmv(a, b, **spmv_kwargs)

    # Normalize sparse 2D operands to canonical mode order before dispatch.
    # This keeps fast kernels correct and avoids known non-default mode-order
    # issues in the generic path.
    if a.dim() == 2 and b.dim() == 2:
        default_mode_order = [0, 1]
        has_non_default_mode_order = (
            a.storage.index.mode_order != default_mode_order
            or b.storage.index.mode_order != default_mode_order
        )
        has_sparse_input = (not a.format.is_dense()) or (not b.format.is_dense())
        if has_non_default_mode_order and has_sparse_input:
            if a.storage.index.mode_order != default_mode_order:
                a = a.copy()
                a.change_mode_order(default_mode_order)
            if b.storage.index.mode_order != default_mode_order:
                b = b.copy()
                b.change_mode_order(default_mode_order)

    if use_cache:
        resolved = resolve_prebuilt_matmul(
            a, b, output_format=requested_output_format
        )
        if resolved is not None:
            result_cpp, result_shape = execute_prebuilt_binary_kernel(
                resolved.fn, a, b, time_dict=time_dict
            )
            result = STensor(
                shape=result_shape,
                index=TensorIndex(
                    mode_indices=result_cpp.storage.index.mode_indices,
                    tensor_format=resolved.output_format,
                ),
                value=result_cpp.storage.value,
            )
        else:
            result = einsum("ij,jk->ik", a, b, **kwargs)
    else:
        result = einsum("ij,jk->ik", a, b, **kwargs)

    if isinstance(result, STensor) and result.format.is_dense():
        result = result.to_torch()

    return result


def einsum(
    expression: str,
    *tensors: Optional[Union[torch.Tensor, STensor]],
    compile_only: Optional[bool] = False,
    **kwargs: Any,
) -> STensor:
    # e.g. expression might be e.g. "i,i->i" and "ij,ij->ij" for
    # elementwise multiplication or "ik,kj->ij" for matrix multiplication

    # # If any of the tensors have the same name, rename them
    # tensor_names = [tensor.name for tensor in tensors]
    # tensor_name_counts = {name: tensor_names.count(name) for name in tensor_names}
    # for i, tensor in enumerate(tensors):
    #     if tensor_name_counts[tensor.name] > 1:
    #         tensor.name = tensor.name + str(i)

    # Convert all torch.Tensor inputs to STensor early
    tensors = tuple(
        STensor.from_torch(t) if isinstance(t, torch.Tensor) else t for t in tensors
    )

    # ── Fast dispatch cache ─────────────────────────────────────────────
    # On a cache hit, skip the entire scheduling pipeline (select_loop_order
    # + auto_schedule) which dominates wall-clock time for cached kernels.
    _dispatch_key = None
    if not compile_only and "_post_ops" not in kwargs:
        _dispatch_key = (
            expression,
            tuple(str(t.format) for t in tensors),
            tuple(t.dtype for t in tensors),
            kwargs.get("format", None),
            tuple(kwargs.get("output_mode_order", ())) if kwargs.get("output_mode_order") else None,
        )
        _cached = _einsum_dispatch_cache.get(_dispatch_key)
        if _cached is not None:
            _module = _cached[0]
            _output_fmt = _cached[1]
            _temp_mo = _cached[2]
            _final_mo = _cached[3]
            _input_mos = _cached[4]
            _input_idx_strs = _cached[5]
            _result_idx_strs = _cached[6]

            # Set correct mode orders on input tensors
            for _t, _mo in zip(tensors, _input_mos):
                if _t.storage.index.mode_order != _mo:
                    _t.change_mode_order(_mo)

            # Compute result shape from expression + current tensor shapes
            _idx_to_size: dict = {}
            for _idxs, _t in zip(_input_idx_strs, tensors):
                for _i, _s in enumerate(_idxs):
                    if _s not in _idx_to_size:
                        _idx_to_size[_s] = _t.shape[_i]
            _result_shape = tuple(_idx_to_size[_s] for _s in _result_idx_strs)

            # Build args and evaluate
            _args: List[Any] = [_result_shape]
            for _t in tensors:
                _args.append(_t.shape)
                _args.append(_t.index.mode_indices)
                _args.append(_t.values)

            _t0 = time.time()
            _result_cpp = _module.evaluate(*_args)
            _eval_time = time.time() - _t0

            _result = STensor(
                shape=_result_shape,
                index=TensorIndex(
                    mode_indices=_result_cpp.storage.index.mode_indices,
                    tensor_format=_output_fmt,
                    mode_order=_temp_mo if _temp_mo else _final_mo,
                ),
                value=_result_cpp.storage.value,
            )

            if "time_dict" in kwargs:
                kwargs["time_dict"]["eval_time"] = _eval_time

            if _temp_mo:
                _result.change_mode_order(_final_mo)

            return _result
    # ── End fast dispatch ────────────────────────────────────────────────

    # unique_index_strs should be a list of unique index strings
    # e.g. ["i", "j", "k"]
    unique_index_strs = list(expression.replace(",", "").replace("->", ""))
    # Make sure the index strings are unique, keeping the order
    unique_index_strs = list(dict.fromkeys(unique_index_strs))
    input_index_strs = [list(x) for x in expression.split("->")[0].split(",")]
    result_index_strs = list(expression.split("->")[1])

    # Reorder input index strings by each tensor's mode_order
    input_index_strs_sorted = [
        [input_index_strs[i][idx] for idx in tensors[i].storage.index.mode_order]
        for i in range(len(tensors))
    ]

    # Reorder result index strings by output_mode_order if provided
    output_mode_order = kwargs.get("output_mode_order", None)
    result_index_strs_sorted = (
        [result_index_strs[i] for i in output_mode_order]
        if output_mode_order
        else result_index_strs
    )

    # Build concatenated substrings for topo_sort_characters
    index_strs_concat = (
        ["".join(s) for s in input_index_strs_sorted]
        + ["".join(result_index_strs_sorted)]
    )
    index_strs_by_schedule = topo_sort_characters(index_strs_concat, tensors)

    # Create a list of IndexVar objects, and a dict mapping index strings
    # to IndexVar objects
    index_vars = [IndexVar(index_str) for index_str in unique_index_strs]
    index_var_dict = {index_var.name: index_var for index_var in index_vars}

    # Compute temp_mode_order: the scheduler-recommended mode order for result
    index_str_to_mode_index = {s: i for i, s in enumerate(result_index_strs)}
    temp_mode_order = [
        index_str_to_mode_index[s]
        for s in index_strs_by_schedule
        if s in result_index_strs
    ]
    final_mode_order = output_mode_order if output_mode_order else temp_mode_order

    # Change input tensor mode orders to match schedule
    for tensor_index, input_index_str in enumerate(input_index_strs):
        new_mode_order = []
        str_to_mode = {s: i for i, s in enumerate(input_index_str)}
        for s in index_strs_by_schedule:
            if s in str_to_mode:
                new_mode_order.append(str_to_mode[s])
        tensors[tensor_index].change_mode_order(new_mode_order)

    # Create a mapping from each index string to the list of LevelFormats
    # of the levels it indexes into each input tensor
    index_str_to_level_formats = {}
    tensors_new = []
    for sorted_index_strs, tensor in zip(input_index_strs_sorted, tensors):
        assert isinstance(tensor, STensor), "Input tensor is not a Scorch Tensor"
        tensors_new.append(tensor)

        for i, index_str in enumerate(sorted_index_strs):
            if index_str not in index_str_to_level_formats:
                index_str_to_level_formats[index_str] = []
            index_str_to_level_formats[index_str].append(
                tensor.format.get_level_formats()[i]
            )

    tensors = tensors_new

    # Create TensorVar's for each tensor
    tensor_vars = []
    tensor_names_available = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    output_tensor_dtype = None
    for i, tensor in enumerate(tensors):
        if isinstance(tensor, STensor):
            tensor_name = tensor_names_available.pop(0)
            tensor_vars.append(
                TensorVar(
                    name=tensor_name,
                    fmt=tensor.format,
                    dtype=tensor.dtype,
                    mode_order=tensor.storage.index.mode_order,
                )
            )
            if output_tensor_dtype is None:
                output_tensor_dtype = tensor.dtype
            else:
                assert (
                    output_tensor_dtype == tensor.dtype
                ), "All tensors must have the same dtype"

    # Get output format from kwargs
    output_format = kwargs.get("format", None)

    # If output format is not specified, do sparse for all levels first
    if output_format is None:
        # Use format inference rules to infer the optimal format of the output
        # tensor
        # The format inference rules are decided on a per-level basis:
        # 1. Let the index variable indexing into the level be called i
        # 2. If the index variable is used to index into any input tensor's sparse
        #    dimension and multiplied with any other tensor, then the level is
        #    sparse
        # 3. If the index variable is used to index into any input tensor's dense
        #    dimension and added with any other tensor, then the level is dense
        # 4. Otherwise, the level is compressed
        # i.e sparse * anything = sparse
        #     dense + anything = dense
        # Note that LevelType.COMPRESSED and LevelType.COORDINATE are both "sparse"
        # levels
        # To break ties, we use the following priority: we always prefer coordinate
        # over compressed

        # Create a list of LevelFormat objects
        output_level_formats = []
        for index_str in result_index_strs:
            level_format = LevelFormat(LevelType.DENSE)
            # Use the index_str_to_level_formats to get the list of LevelFormats
            # of the levels it indexes into each input tensor
            level_formats: List[LevelFormat] = index_str_to_level_formats[index_str]
            # If any of them is sparse, then the output level is sparse
            if any(
                level_format.get_level_type() == LevelType.COMPRESSED
                for level_format in level_formats
            ):
                level_format = LevelFormat(LevelType.COMPRESSED)
            # If any of them is coordinate, then the output level is coordinate format
            elif any(
                level_format.get_level_type() == LevelType.COORDINATE
                for level_format in level_formats
            ):
                level_format = LevelFormat(LevelType.COORDINATE)

            output_level_formats.append(level_format)

        # Make sure that the output format doesn't have a sparse level preceding
        # a dense level
        # If it does, then we need to make the preceding level dense as well
        # e.g. if the output level formats are [sparse, dense, dense], then we
        # need to make it [dense, dense, dense]
        # TODO: unless we are dealing with block tensors
        for i in range(len(output_level_formats) - 1, 0, -1):
            if (
                output_level_formats[i].get_level_type() == LevelType.DENSE
                and output_level_formats[i - 1].get_level_type() != LevelType.DENSE
            ):
                output_level_formats[i - 1] = LevelFormat(LevelType.DENSE)

        # For SDDMM-like patterns (sparse output with reduction variables
        # where an input tensor mirrors the output sparsity), use all-COO
        # output to enable the scalar-accum codegen path.  This gives
        # optimal loop order (reduction innermost), no workspace, and SIMD
        # vectorization of the dense reduction.
        _has_sparse_output = any(
            lf.get_level_type() in (LevelType.COMPRESSED, LevelType.COORDINATE)
            for lf in output_level_formats
        )
        if _has_sparse_output:
            _result_set = set(result_index_strs)
            _reduction_strs = [s for s in unique_index_strs if s not in _result_set]
            if _reduction_strs:
                # Check if any input tensor has the same index variables as
                # the output and contains sparse levels (SDDMM pattern).
                for inp_strs, tensor in zip(input_index_strs, tensors):
                    if set(inp_strs) == _result_set:
                        _inp_level_types = tensor.format.get_level_types()
                        if any(
                            lt in (LevelType.COMPRESSED, LevelType.COORDINATE)
                            for lt in _inp_level_types
                        ):
                            output_level_formats = [
                                LevelFormat(LevelType.COORDINATE)
                                for _ in output_level_formats
                            ]
                            break

        output_format = TensorFormat(output_level_formats)
        # print(f"\nUnspecified output format, using inferred {output_format}")
    else:
        output_format = parse_format(output_format)

    # Create the result TensorVar
    assert output_tensor_dtype is not None, "Output tensor type is not defined"
    result_tensor_var = TensorVar(
        name=tensor_names_available.pop(0),
        fmt=output_format,
        dtype=output_tensor_dtype,
        mode_order=temp_mode_order,
    )

    # Build RHS expression: product of all tensor accesses
    assert index_var_dict, "index_var_dict is empty"
    rhs_expr = None
    for i, tensor_var in enumerate(tensor_vars):
        indices = [index_var_dict[s] for s in input_index_strs[i]]
        access = tensor_var[indices[0]] if len(indices) == 1 else tensor_var[tuple(indices)]
        rhs_expr = access if rhs_expr is None else rhs_expr * access

    # Build LHS access and create assignment
    assert result_tensor_var is not None, "result_tensor_var is not defined"
    lhs_indices = [index_var_dict[s] for s in result_index_strs]
    lhs_key = lhs_indices[0] if len(lhs_indices) == 1 else tuple(lhs_indices)
    result_tensor_var[lhs_key] = rhs_expr

    # Wrap in nested ForAll loops (outermost first in schedule, built inside-out)
    cin_stmt = result_tensor_var._assignment
    for index_str in reversed(index_strs_by_schedule):
        cin_stmt = ForAll(index_var_dict[index_str], cin_stmt)

    # print("CIN:\n", cin_stmt)

    # Align input tensor mode orders with the selected loop order to keep
    # parent-child level traversal valid during lowering for non-canonical
    # schedules.
    selected_loop_order = Scheduler.select_loop_order(cin_stmt)
    selected_loop_order_names = [index_var.name for index_var in selected_loop_order]
    for tensor_index, input_index_str in enumerate(input_index_strs):
        desired_index_strs = [
            index_str for index_str in selected_loop_order_names if index_str in input_index_str
        ]
        desired_mode_order = [input_index_str.index(index_str) for index_str in desired_index_strs]
        if tensors[tensor_index].storage.index.mode_order != desired_mode_order:
            tensors[tensor_index].change_mode_order(desired_mode_order)
        if tensor_vars[tensor_index].mode_order != desired_mode_order:
            tensor_vars[tensor_index].mode_order = desired_mode_order

    cin_stmt = Scheduler.auto_schedule(cin_stmt)

    # print("Auto-scheduled CIN:\n", cin_stmt)

    # Extract PostOps for fused kernel compilation
    _post_ops = kwargs.get("_post_ops", None)
    _post_ops_tensors = kwargs.get("_post_ops_tensors", None)
    _cache_key_suffix = str(_post_ops) if _post_ops else ""
    _kernel_cache_key = str(cin_stmt) + _cache_key_suffix

    if _kernel_cache_key in _kernel_cache:
        # print(f"Using cached kernel for {cin_stmt}")
        module = _kernel_cache[_kernel_cache_key]
    else:
        lowerer = CINLowerer(post_ops=_post_ops)

        lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

        llir_lowerer = LLIRLowerer()

        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        # print("\n\n", cpp_code)

        # Read header_cpp_code from csrc/header.cpp
        with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
            header_cpp_code = f.read()

        module = _load_kernel(
            name=_kernel_name(header_cpp_code, cpp_code),
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=get_extra_cflags(),
            extra_ldflags=get_extra_ldflags(),
        )

        _kernel_cache[_kernel_cache_key] = module

    # Populate the dispatch cache so future calls skip scheduling entirely.
    if _dispatch_key is not None:
        _einsum_dispatch_cache[_dispatch_key] = (
            module,
            output_format,
            temp_mode_order,
            final_mode_order,
            [list(t.storage.index.mode_order) for t in tensors],
            input_index_strs,
            result_index_strs,
        )

    if compile_only:
        return STensor("Compile only")

    # Create a mapping from each index string to the size of the dimension
    # it indexes
    index_str_to_size = {}
    for index_strs, tensor in zip(input_index_strs, tensors):
        assert isinstance(tensor, STensor)
        for i, index_str in enumerate(index_strs):
            if index_str not in index_str_to_size:
                index_str_to_size[index_str] = tensor.shape[i]

    # Get the result shape from the expression, using index_str_to_size
    result_shape = tuple(
        [index_str_to_size[index_str] for index_str in result_index_strs]
    )

    # Call module.evaluate with the output shape,and the mode indices and values of each tensor
    args: Sequence[Any] = [result_shape]
    for tensor in tensors:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor.index.mode_indices)  # type: ignore
        args.append(tensor.values)  # type: ignore

    # Append extra tensors for PostOps (bias, scale, etc.)
    if _post_ops_tensors:
        for extra_t in _post_ops_tensors:
            args.append(extra_t)  # type: ignore

    start_time = time.time()
    result_cpp = module.evaluate(*args)
    end_time = time.time()
    eval_time = end_time - start_time
    # print("Time taken for evaluate:", eval_time)

    result = STensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp.storage.index.mode_indices,
            tensor_format=output_format,
            mode_order=temp_mode_order if temp_mode_order else final_mode_order,
        ),
        value=result_cpp.storage.value,
    )

    if "time_dict" in kwargs:
        time_dict = kwargs["time_dict"]
        time_dict["eval_time"] = eval_time

    # Convert to final mode order if it differs from temporary mode order
    if temp_mode_order:
        result.change_mode_order(final_mode_order)

    return result


def _align_mode_orders_to_loop_order(
    cin_stmt: IndexStmt, args: tuple
) -> None:
    """Align input tensor mode orders to the CIN loop order.

    The lowerer requires parent physical levels to be iterated before child
    levels.  When a tensor's mode_order doesn't match the loop nesting, the
    generated code references coordinate variables before they are defined.
    This mirrors the alignment that ``einsum`` performs (ops.py L581-591).

    Mutates TensorVar.mode_order in *cin_stmt* and calls
    ``STensor.change_mode_order`` on the corresponding *args* entries.
    """
    # 1. Extract loop order
    loop_order_names: List[str] = []
    curr: IndexStmt = cin_stmt
    while isinstance(curr, ForAll):
        loop_order_names.append(curr.index_var.name)
        curr = curr.stmt

    if not loop_order_names:
        return

    # 2. Get RHS tensor accesses (left-to-right order matches *args*)
    rhs_accesses = cin_stmt.get_rhs_tensor_accesses()
    if len(rhs_accesses) != len(args):
        return  # can't align if we don't have a 1:1 mapping

    for ta, stensor in zip(rhs_accesses, args):
        tv = ta.get_tensor()
        index_var_names = [iv.name for iv in ta.get_index_vars()]
        # Filter loop order to vars present in this tensor
        desired_names = [n for n in loop_order_names if n in index_var_names]
        desired_mode_order = [index_var_names.index(n) for n in desired_names]

        # Skip when tiling/broadcasting causes a rank mismatch
        if len(desired_mode_order) != len(tv.mode_order):
            continue
        if list(tv.mode_order) != desired_mode_order:
            tv.mode_order = desired_mode_order
            if stensor.has_index and stensor.shape is not None:
                stensor.change_mode_order(desired_mode_order)

    # 3. Also align the output tensor
    if isinstance(curr, TensorAssign):
        lhs_tv = curr.lhs.get_tensor()
        lhs_names = [iv.name for iv in curr.lhs.get_index_vars()]
        desired_names = [n for n in loop_order_names if n in lhs_names]
        desired_mode_order = [lhs_names.index(n) for n in desired_names]
        if len(desired_mode_order) == len(lhs_tv.mode_order) and list(lhs_tv.mode_order) != desired_mode_order:
            lhs_tv.mode_order = desired_mode_order


def lower_and_exec_cin(
    cin_stmt: IndexStmt, result_shape: Sequence[int], *args: STensor, **kwargs
) -> STensor:
    """Lower a CIN statement to LLIR then codegen and call on the input tensors.

    Args:
        cin_stmt (IndexStmt): CIN statement to lower.
        result_shape (Sequence[int]): Shape of the result tensor.
        *args (STensor): Input tensors.

    Returns:
        STensor: Output tensor.
    """
    _align_mode_orders_to_loop_order(cin_stmt, args)

    # Lower to LLIR
    lowerer = CINLowerer()
    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
    llir_lowerer = LLIRLowerer()
    cpp_code = llir_lowerer.lower_llir(lowered_llir)
    # print(cpp_code)
    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    module = _load_kernel(
        name=_kernel_name(header_cpp_code, cpp_code),
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=get_extra_cflags(),
        extra_ldflags=get_extra_ldflags(),
    )

    module_args: List[Any] = [result_shape]

    for arg in args:
        module_args.append(arg.shape)
        module_args.append(arg.index.mode_indices)
        module_args.append(arg.values)

    start_time = time.time()
    result_cpp = module.evaluate(*module_args)
    end_time = time.time()
    eval_time = end_time - start_time
    if "time_dict" in kwargs:
        kwargs["time_dict"]["eval_time"] = eval_time

    result = STensor(
        shape=tuple(result_shape),
        index=TensorIndex(
            mode_indices=result_cpp.storage.index.mode_indices,
            tensor_format="dd",
        ),
        value=result_cpp.storage.value,
    )

    return result


def precompile_kernels():
    DS = STensor(index=TensorIndex(tensor_format="ds"))
    DD = STensor(index=TensorIndex(tensor_format="dd"))
    OO = STensor(index=TensorIndex(tensor_format="oo"))

    einsum("ik,kj->ij", DS, DD, compile_only=True, format="dd")
    einsum("ik,kj->ij", OO, DD, compile_only=True, format="dd")
    einsum("ik,kj->ij", OO, DS, compile_only=True, format="dd")
    einsum("ik,kj->ij", DS, DS, compile_only=True, format="dd")
    einsum("ik,kj->ij", DS, DS, compile_only=True, format="ds")

    print("Precompiled kernels.")
