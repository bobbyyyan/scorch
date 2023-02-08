import torch
from torch.utils.cpp_extension import load

from pathlib import Path
from typing import Any, Union

from src.taco_torch.compiler.cin import IndexVar
from src.taco_torch.format import TensorFormat, LevelFormat, LevelType
from src.taco_torch.storage import TensorStorage, TensorIndex
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
    print(rand_tt_tensor._storage._value)


def test_cpp_ext_sparse_vector_mul():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0])

    sp_vector_a = TacoTensor.from_torch(tensor_a_torch, "a").to_sparse()
    sp_vector_b = TacoTensor.from_torch(tensor_b_torch, "b").to_sparse()

    sp_vector_result = TacoTensor.einsum("i,i->i", sp_vector_a, sp_vector_b)

    # i = IndexVar("i")
    #
    # sp_vector_result = TacoTensor(name="result")

    # sp_vector_result[i] = sp_vector_a[i] * sp_vector_b[i]

    tensor_result = tensor_a_torch * tensor_b_torch

    # csr_matrix[i, j] = TacoTensor.sum(j, csr_matrix_a[i, k] * csr_matrix_b[k, j])
    # csr_matrix[i, j] = csr_matrix_a[i, k] * csr_matrix_b[k, j] + sp_vector_d[k]

    # sp_vector_result_cpp: ops_cpp.TacoTensor = ops_cpp.elemwise_vector_mul_sss(
    #     sp_vector_a.shape,
    #     sp_vector_a.storage.index.mode_indices,
    #     sp_vector_a.values,
    #     sp_vector_b.storage.index.mode_indices,
    #     sp_vector_b.values,
    # )
    #
    # sp_vector_result = TacoTensor(
    #     index=TensorIndex(
    #         mode_indices=sp_vector_result_cpp._storage._index.mode_indices,
    #     ),
    #     value=sp_vector_result_cpp._storage._value,
    # )

    print(sp_vector_result)
    print(sp_vector_result.storage.index.mode_indices)
    print(sp_vector_result.values)


def taco_einsum(
    expression: str,
    *tensors: Union[torch.Tensor, TacoTensor],
    **kwargs: Any,
) -> TacoTensor:
    """Perform a tensor contraction using the TACO compiler."""
    print(f"Evaluating expression: {expression}")
    print("Tensors:", tensors)
    raise NotImplementedError("TACO einsum is not implemented yet.")


def mul(src: TacoTensor, other: TacoTensor):
    """Multiply two tensors.
    e.g. `mul(a, b)` is equivalent to `a * b`.
    """
    # TODO: Lower to TACO IR
    # TODO: Compile to C++ code
    # TODO: (Inline) load C++ code using PyTorch's C++ extension
    # TODO: Call C++ code
    # TODO: Return result
    # ttensor = ops_cpp.TacoTensor

    result = ops_cpp.elemwise_mul(
        src._storage._index.mode_indices,
        src._storage.value,
        other._storage._index.mode_indices,
        other._storage.value,
    )

    print("ops_cpp.TacoTensor", ops_cpp.TacoTensor)

    return TacoTensor(
        storage=TensorStorage(
            index=TensorIndex(
                mode_indices=result._storage._index.mode_indices,
            ),
            value=result._storage._value,
        )
    )

    raise NotImplementedError("TACO mul is not implemented yet.")


def add(src, other):
    # print(ops_cpp.add)
    # TODO: implement this
    # return TacoTensor(
    #     value=ops_cpp.add(src.value, other.value),
    # )
    # return ops_cpp.add(src, other)

    # Lower to TACO IR
    # Compile to C++ code
    # (Inline) load C++ code using PyTorch's C++ extension
    # Call C++ code
    # Return result

    if isinstance(other, TacoTensor):
        raise NotImplementedError("TACO add is not implemented yet.")


TacoTensor.add = add
TacoTensor.__add__ = add
TacoTensor.__mul__ = mul
