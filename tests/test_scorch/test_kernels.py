import torch

from src.scorch import Tensor, einsum, TensorFormat


def test_elemwise_vector_mul_sss():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5])

    sp_vector_a = Tensor.from_torch(tensor_a_torch, "a").to_sparse()
    sp_vector_b = Tensor.from_torch(tensor_b_torch, "b").to_sparse()

    result = einsum("i,i->i", sp_vector_a, sp_vector_b)

    print("\nResult shape:", result.shape)
    print("Result format:", result.format)
    print("Result index:", result.index.mode_indices)
    print("Result values:", result.values)


def test_elemwise_vector_add_sss():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5])

    sp_vector_a = Tensor.from_torch(tensor_a_torch, "a").to_sparse()
    sp_vector_b = Tensor.from_torch(tensor_b_torch, "b").to_sparse()

    result = sp_vector_a + sp_vector_b

    print("\nResult shape:", result.shape)
    print("Result format:", result.format)
    print("Result index:", result.index.mode_indices)
    print("Result values:", result.values)


def test_elemwise_vector_mul_dss():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4, 5, 0])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5, 0, 14])

    sp_vector_a = Tensor.from_torch(tensor_a_torch, "a").to_sparse()
    sp_vector_b = Tensor.from_torch(tensor_b_torch, "b").to_sparse()

    result = einsum("i,i->i", sp_vector_a, sp_vector_b, format=TensorFormat("d"))

    print("\nResult shape:", result.shape)
    print("Result format:", result.format)
    print("Result index:", result.index.mode_indices)
    print("Result values:", result.values)


def test_elemwise_matrix_mul_ss_ss_ss():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5])

    sp_vector_a = Tensor.from_torch(tensor_a_torch, "a").to_sparse()
    sp_vector_b = Tensor.from_torch(tensor_b_torch, "b").to_sparse()

    result = einsum("i,i->i", sp_vector_a, sp_vector_b)

    print("\nResult shape:", result.shape)
    print("Result format:", result.format)
    print("Result index:", result.index.mode_indices)
    print("Result values:", result.values)
