import time
from pathlib import Path
from typing import Any, Union, Sequence, Optional, List

import torch
from torch.utils.cpp_extension import load, load_inline

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
from .storage import TensorIndex
from .tensor import Tensor
from .utils import parse_format, topo_sort_characters

PROJECT_ROOT_DIR = Path(__file__)
while not (PROJECT_ROOT_DIR / "setup.py").exists():
    PROJECT_ROOT_DIR = PROJECT_ROOT_DIR.parent

_kernel_cache = {}

# Register custom classes
load(
    name="pybind",
    sources=[str(PROJECT_ROOT_DIR / "csrc/pybind.cpp")],
)

# Read header_cpp_code from csrc/header.cpp
with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
    header_cpp_code = f.read()

with open(PROJECT_ROOT_DIR / "csrc/spmm-csr.cpp", "r") as f:
    cpp_code = f.read()

# Load special kernels
spmm_csr = load_inline(
    name="kernel",
    cpp_sources=[header_cpp_code, cpp_code],
    functions=["evaluate"],
    extra_cflags=["-O3", "-march=native", "-ffast-math", "-fno-signed-zeros"],
)

_kernel_cache["spmm_csr"] = spmm_csr


def spmv(
    a: Tensor,
    b: Tensor,
    output_format: Optional[Union[TensorFormat, str, List[str]]] = None,
    **kwargs,
) -> Tensor:
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
    module = torch.utils.cpp_extension.load_inline(
        name="kernel",
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=["-O3"],
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

    result = Tensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format=output_format,
        ),
        value=result_cpp._storage._value,
    )

    return result


def matmul_wksp(
    a: Union[torch.Tensor, Tensor],
    b: Union[torch.Tensor, Tensor],
    output_format: Optional[Union[TensorFormat, str, List[str]]] = None,
    **kwargs,
) -> Tensor:
    if isinstance(a, torch.Tensor):
        a = Tensor.from_torch(a).to_sparse()
    if isinstance(b, torch.Tensor):
        b = Tensor.from_torch(b).to_sparse()

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
    module = torch.utils.cpp_extension.load_inline(
        name="kernel",
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=["-O3"],
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

    result = Tensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format=output_format,
        ),
        value=result_cpp._storage._value,
    )

    return result


def matmul(
    a: Union[torch.Tensor, Tensor],
    b: Union[torch.Tensor, Tensor],
    **kwargs: Any,
) -> Tensor:
    """Perform a matrix multiplication."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        result = torch.matmul(a, b)
        return Tensor.from_torch(result)

    if isinstance(a, torch.Tensor):
        a = Tensor.from_torch(a)
    if isinstance(b, torch.Tensor):
        b = Tensor.from_torch(b)

    if a.dim() == 2 and b.dim() == 1:
        return spmv(a, b, **kwargs)

    if str(a.format) == "d,s" and str(b.format) == "d,d":
        result_shape = (a.shape[0], b.shape[1])
        args = [result_shape]

        for tensor in [a, b]:
            args.append(tensor.shape)  # type: ignore
            args.append(tensor.index.mode_indices)  # type: ignore
            args.append(tensor.values)  # type: ignore

        spmm_csr = _kernel_cache["spmm_csr"]

        start_time = time.time()
        result_cpp = spmm_csr.evaluate(*args)
        end_time = time.time()
        eval_time = end_time - start_time
        if "time_dict" in kwargs:
            kwargs["time_dict"]["eval_time"] = eval_time

        result = Tensor(
            shape=result_shape,
            index=TensorIndex(
                mode_indices=result_cpp._storage._index.mode_indices,
                tensor_format="dd",
            ),
            value=result_cpp._storage._value,
        )

        return result

    return einsum("ij,jk->ik", a, b, **kwargs)


def einsum(
    expression: str,
    *tensors: Optional[Union[torch.Tensor, Tensor]],
    compile_only: Optional[bool] = False,
    **kwargs: Any,
) -> Tensor:
    # e.g. expression might be e.g. "i,i->i" and "ij,ij->ij" for
    # elementwise multiplication or "ik,kj->ij" for matrix multiplication

    # # If any of the tensors have the same name, rename them
    # tensor_names = [tensor.name for tensor in tensors]
    # tensor_name_counts = {name: tensor_names.count(name) for name in tensor_names}
    # for i, tensor in enumerate(tensors):
    #     if tensor_name_counts[tensor.name] > 1:
    #         tensor.name = tensor.name + str(i)

    # unique_index_strs should be a list of unique index strings
    # e.g. ["i", "j", "k"]
    unique_index_strs = list(expression.replace(",", "").replace("->", ""))
    # Make sure the index strings are unique, keeping the order
    unique_index_strs = list(dict.fromkeys(unique_index_strs))
    result_index_strs = list(expression.split("->")[1])
    input_index_strs_concat = expression.split("->")[0]
    index_strs_by_schedule = topo_sort_characters(
        input_index_strs_concat, priority=result_index_strs
    )
    input_index_strs = [list(x) for x in expression.split("->")[0].split(",")]
    # Create a list of IndexVar objects, and a dict mapping index strings
    # to IndexVar objects
    index_vars = [IndexVar(index_str) for index_str in unique_index_strs]
    index_var_dict = {index_var.name: index_var for index_var in index_vars}

    # Create a mapping from each index string to the list of LevelFormats
    # of the levels it indexes into each input tensor
    index_str_to_level_formats = {}
    for index_strs, tensor in zip(input_index_strs, tensors):
        assert isinstance(tensor, Tensor), "Input tensor is not a Scorch Tensor"

        for i, index_str in enumerate(index_strs):
            if index_str not in index_str_to_level_formats:
                index_str_to_level_formats[index_str] = []
            index_str_to_level_formats[index_str].append(
                tensor.format.get_level_formats()[i]
            )

    # Create TensorVar's for each tensor
    tensor_vars = []
    tensor_names_available = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i, tensor in enumerate(tensors):
        if isinstance(tensor, Tensor):
            tensor_name = tensor_names_available.pop(0)
            tensor_vars.append(TensorVar(name=tensor_name, fmt=tensor.format))

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

        output_format = TensorFormat(output_level_formats)
        print(f"\nUnspecified output format, using inferred {output_format}")
    else:
        output_format = parse_format(output_format)

    # Create the result TensorVar
    result_tensor_var = TensorVar(name=tensor_names_available.pop(0), fmt=output_format)

    # Generate the python code for constructing the TensorAccess's, and TensorAssign and execute it
    assert index_var_dict, "index_var_dict is empty"
    rhs = ""
    for i, tensor_var in enumerate(tensor_vars):
        inside = ", ".join(
            [f'index_var_dict["{index_str}"]' for index_str in input_index_strs[i]]
        )
        rhs += f"tensor_vars[{i}][{inside}]"
        if i < len(tensor_vars) - 1:
            rhs += " * "
    lhs_inside = ", ".join(
        [f'index_var_dict["{index_str}"]' for index_str in result_index_strs]
    )
    assert result_tensor_var is not None, "result_tensor_var is not defined"
    code = f"result_tensor_var[{lhs_inside}] = {rhs}"

    exec(code)

    # print("result_tensor_var._assignment:", result_tensor_var._assignment)

    # Generate the python code for constructing the ForAll's and execute it
    rhs = "result_tensor_var._assignment"
    assert ForAll is not None, "ForAll is not imported"
    for i, index_str in enumerate(index_strs_by_schedule[::-1]):
        rhs = f'ForAll(index_var_dict["{index_str}"], {rhs})'

    cin_stmt = eval(rhs)

    print("CIN:\n", cin_stmt)

    cin_stmt = Scheduler.auto_schedule(cin_stmt)

    print("Auto-scheduled CIN:\n", cin_stmt)

    if str(cin_stmt) in _kernel_cache:
        # print(f"Using cached kernel for {cin_stmt}")
        module = _kernel_cache[str(cin_stmt)]
    else:
        lowerer = CINLowerer()

        lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

        llir_lowerer = LLIRLowerer()

        cpp_code = llir_lowerer.lower_llir(lowered_llir)

        print("\n\n", cpp_code)

        # Read header_cpp_code from csrc/header.cpp
        with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
            header_cpp_code = f.read()

        # # Write "#include <torch/extension.h>" + "\n" + '#include "header.cpp"' + "\n" + cpp_code to a file
        # # and PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("evaluate", &evaluate); }
        # with open(PROJECT_ROOT_DIR / "csrc/kernel.cpp", "w") as f:
        #     f.write("#include <torch/extension.h>\n")
        #     f.write('#include "header.cpp"\n')
        #     f.write(cpp_code)
        #     f.write(
        #         f"""
        #         PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
        #           m.def("evaluate", &evaluate);
        #             pybind11::class_<Tensor>(m, "Tensor")
        #               .def(pybind11::init<>())
        #               .def_readwrite("_storage", &Tensor::_storage);
        #             pybind11::class_<TensorStorage>(m, "TensorStorage")
        #               .def(pybind11::init<>())
        #               .def_readwrite("_value", &TensorStorage::_value)
        #               .def_readwrite("_index", &TensorStorage::_index);
        #             pybind11::class_<TensorIndex>(m, "TensorIndex")
        #               .def(pybind11::init<>())
        #               .def_readwrite("mode_indices", &TensorIndex::mode_indices);
        #         }}
        #         """
        #     )
        #
        # # Load the C++ code using PyTorch's C++ extension
        # module = torch.utils.cpp_extension.load(
        #     name="kernel",
        #     sources=[str(PROJECT_ROOT_DIR / "csrc/kernel.cpp")],
        # )

        module = torch.utils.cpp_extension.load_inline(
            name="kernel",
            cpp_sources=[header_cpp_code, cpp_code],
            functions=["evaluate"],
            extra_cflags=["-O3"],
        )

        _kernel_cache[str(cin_stmt)] = module

        print(f"Cached kernel for {cin_stmt}")

    if compile_only:
        return None

    # Create a mapping from each index string to the size of the dimension
    # it indexes
    index_str_to_size = {}
    for index_strs, tensor in zip(input_index_strs, tensors):
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
        args.append(tensor._storage._index.mode_indices)  # type: ignore
        args.append(tensor._storage.value)  # type: ignore

    start_time = time.time()
    result_cpp = module.evaluate(*args)
    end_time = time.time()
    eval_time = end_time - start_time
    # print("Time taken for evaluate:", eval_time)

    result = Tensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format=output_format,
        ),
        value=result_cpp._storage._value,
    )

    if "time_dict" in kwargs:
        time_dict = kwargs["time_dict"]
        time_dict["eval_time"] = eval_time

    return result


def lower_and_exec_cin(
    cin_stmt: IndexStmt, result_shape: Sequence[int], *args: Tensor, **kwargs
) -> Tensor:
    """Lower a CIN statement to LLIR then codegen and call on the input tensors.

    Args:
        cin_stmt (IndexStmt): CIN statement to lower.
        result_shape (Sequence[int]): Shape of the result tensor.
        *args (Tensor): Input tensors.

    Returns:
        Tensor: Output tensor.
    """
    # Lower to LLIR
    lowerer = CINLowerer()
    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
    llir_lowerer = LLIRLowerer()
    cpp_code = llir_lowerer.lower_llir(lowered_llir)
    print(cpp_code)
    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    module = torch.utils.cpp_extension.load_inline(
        name="kernel",
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=["-O3"],
    )

    module_args = [result_shape]

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

    result = Tensor(
        shape=tuple(result_shape),
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format="dd",
        ),
        value=result_cpp._storage._value,
    )

    return result


def precompile_kernels():
    DS = Tensor(index=TensorIndex(tensor_format="ds"))
    DD = Tensor(index=TensorIndex(tensor_format="dd"))
    OO = Tensor(index=TensorIndex(tensor_format="oo"))

    einsum("ik,kj->ij", DS, DD, compile_only=True, format="dd")
    einsum("ik,kj->ij", OO, DD, compile_only=True, format="dd")
    einsum("ik,kj->ij", OO, DS, compile_only=True, format="dd")
    einsum("ik,kj->ij", DS, DS, compile_only=True, format="dd")
    einsum("ik,kj->ij", DS, DS, compile_only=True, format="ds")

    print("Precompiled kernels.")
