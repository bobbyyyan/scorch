from pathlib import Path
from typing import Any, Union, Sequence

import torch
from torch.utils.cpp_extension import load

from src.taco_torch.compiler.cin import IndexVar, TensorVar, ForAll
from src.taco_torch.compiler.cin_lowerer import CINLowerer
from src.taco_torch.compiler.codegen import LLIRLowerer
from src.taco_torch.format import TensorFormat, LevelFormat, LevelType
from src.taco_torch.storage import TensorIndex
from src.taco_torch.tensor import TacoTensor

PROJECT_ROOT_DIR = Path(__file__)
while not (PROJECT_ROOT_DIR / "setup.py").exists():
    PROJECT_ROOT_DIR = PROJECT_ROOT_DIR.parent

ops_cpp = load(
    name="ops_cpp",
    sources=[str(PROJECT_ROOT_DIR / "csrc/ops.cpp")],
)


def test_cpp_ext_rand_matrix():
    print("random tensor:")
    print(ops_cpp.get_rand_matrix(7, 8))


def test_cpp_ext_rand_matrix_tt():
    print("random tt tensor:")
    rand_tt_tensor: TacoTensor = ops_cpp.get_rand_matrix_tt(7, 8)
    print(rand_tt_tensor._storage._value)  # type: ignore


def test_cpp_ext_sparse_vector_mul():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5])

    sp_vector_a = TacoTensor.from_torch(tensor_a_torch, "a").to_sparse()
    sp_vector_b = TacoTensor.from_torch(tensor_b_torch, "b").to_sparse()

    result = einsum("i,i->i", sp_vector_a, sp_vector_b)

    print("\nResult shape:", result.shape)
    print("Result format:", result.format)
    print("Result index:", result.index.mode_indices)
    print("Result values:", result.values)


def einsum(
    expression: str,
    *tensors: Union[torch.Tensor, TacoTensor],
    **kwargs: Any,
) -> TacoTensor:
    """Perform a tensor contraction using the TACO compiler."""
    # e.g. expression might be e.g. "i,i->i" and "ij,ij->ij" for
    # elementwise multiplication or "ik,kj->ij" for matrix multiplication

    # unique_index_strs should be a list of unique index strings
    # e.g. ["i", "j", "k"]
    unique_index_strs = list(set(expression.replace(",", "").replace("->", "")))
    result_index_strs = list(expression.split("->")[1])
    input_index_strs = [list(x) for x in expression.split("->")[0].split(",")]
    # Create a list of IndexVar objects, and a dict mapping index strings
    # to IndexVar objects
    index_vars = [IndexVar(index_str) for index_str in unique_index_strs]
    index_var_dict = {index_var.name: index_var for index_var in index_vars}

    # Create a mapping from each index string to the size of the dimension
    # it indexes
    index_str_to_size = {}
    for index_strs, tensor in zip(input_index_strs, tensors):
        for i, index_str in enumerate(index_strs):
            if index_str not in index_str_to_size:
                index_str_to_size[index_str] = tensor.shape[i]

    # Create TensorVar's for each tensor
    tensor_vars = []
    for tensor in tensors:
        if isinstance(tensor, TacoTensor):
            tensor_vars.append(TensorVar(name=tensor.name, fmt=tensor.format))

    # Get output format from kwargs
    output_format = kwargs.get("format", None)

    # If output format is not specified, do sparse for all levels first
    if output_format is None:
        # Create a list of LevelFormat objects
        level_formats = []
        for index_str in result_index_strs:
            level_formats.append(LevelFormat(LevelType.COMPRESSED))
        output_format = TensorFormat(level_formats)

    # Create the result TensorVar
    result_tensor_var = TensorVar(name="result", fmt=output_format)

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
    for i, index_str in enumerate(unique_index_strs[::-1]):
        rhs = f'ForAll(index_var_dict["{index_str}"], {rhs})'
    cin_stmt = eval(rhs)

    # print("cin_stmt:", cin_stmt)

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    cpp_code = llir_lowerer.lower_llir(lowered_llir)

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
    #             m.def("evaluate", &evaluate);
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
    )

    # Get the result shape from the expression, using index_str_to_size
    result_shape = tuple(
        [index_str_to_size[index_str] for index_str in result_index_strs]
    )

    # Call module.evaluate with the output shape,and the mode indices and values of each tensor
    args: Sequence[Any] = [result_shape]
    for tensor in tensors:
        args.append(tensor._storage._index.mode_indices)  # type: ignore
        args.append(tensor._storage.value)  # type: ignore

    result_cpp = module.evaluate(*args)
    result = TacoTensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format=output_format,
        ),
        value=result_cpp._storage._value,
    )

    return result
