import torch

from src.scorch import Tensor, einsum, TensorFormat


def test_elemwise_vector_mul_sss():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5])

    sp_vector_a = Tensor.from_torch(tensor_a_torch).to_sparse()
    sp_vector_b = Tensor.from_torch(tensor_b_torch).to_sparse()

    result = einsum("i,i->i", sp_vector_a, sp_vector_b)

    assert result.shape == (12,)
    assert len(result.index.mode_indices) == 1

    mode_index = result.index.mode_indices[0]
    assert mode_index[0].tolist() == [0, 5]
    assert mode_index[1].tolist() == [0, 2, 4, 10, 11]

    assert result.values.tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]


def test_elemwise_vector_add_sss():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5])

    sp_vector_a = Tensor.from_torch(tensor_a_torch).to_sparse()
    sp_vector_b = Tensor.from_torch(tensor_b_torch).to_sparse()

    result = sp_vector_a + sp_vector_b

    assert result.shape == (12,)
    assert len(result.index.mode_indices) == 1

    mode_index = result.index.mode_indices[0]
    assert mode_index[0].tolist() == [0, 9]
    assert mode_index[1].tolist() == [0, 1, 2, 3, 4, 6, 8, 10, 11]

    assert result.values.tolist() == [3.0, 2.0, 4.0, 2.0, 5.0, 4.0, 5.0, 9.0, 6.5]


def test_elemwise_vector_add_sss_2():
    tensor_a_torch = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    tensor_b_torch = torch.Tensor([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    tensor_c_torch = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12])

    sp_vector_a = Tensor.from_torch(tensor_a_torch).to_sparse()
    sp_vector_b = Tensor.from_torch(tensor_b_torch).to_sparse()
    sp_vector_c = Tensor.from_torch(tensor_c_torch).to_sparse()

    result = sp_vector_a + sp_vector_b + sp_vector_c

    assert result.shape == (12,)
    assert len(result.index.mode_indices) == 1

    mode_index = result.index.mode_indices[0]
    assert mode_index[0].tolist() == [0, 3]
    assert mode_index[1].tolist() == [0, 1, 11]

    assert result.values.tolist() == [1.0, 2.0, 12.0]


def test_elemwise_vector_mul_dss():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4, 5, 0])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5, 0, 14])

    sp_vector_a = Tensor.from_torch(tensor_a_torch).to_sparse()
    sp_vector_b = Tensor.from_torch(tensor_b_torch).to_sparse()

    result = einsum("i,i->i", sp_vector_a, sp_vector_b, format=TensorFormat("d"))

    assert result.shape == (14,)
    assert len(result.index.mode_indices) == 1

    assert result.values.tolist() == [
        2.0,
        0.0,
        4.0,
        0.0,
        6.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.0,
        10.0,
        0.0,
        0.0,
    ]


def test_elemwise_matrix_mul_ss_ss_ss():
    # # Generate a random sparse 10x10 torch tensor
    # tensor_a_torch = torch.rand(10, 10)
    # tensor_a_torch[torch.rand(10, 10) > 0.5] = 0
    # tensor_b_torch = torch.rand(10, 10)
    # tensor_b_torch[torch.rand(10, 10) > 0.5] = 0
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse()
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse()

    result = einsum("ij,ij->ij", a_sparse, b_sparse)

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[0][0].tolist() == [0, 5]
    assert result.index.mode_indices[0][1].tolist() == [0, 1, 2, 3, 4]

    assert result.index.mode_indices[1][0].tolist() == [0, 1, 2, 3, 4, 5]
    assert result.index.mode_indices[1][1].tolist() == [0, 1, 2, 3, 4]

    assert result.values.tolist() == [1.0, 4.0, 9.0, 16.0, 25.0]
