import torch

from scorch import Tensor
from scorch.storage import TensorIndex


def test_2d_ss_oo():
    # Test converting a COO matrix to a DCSR (sparse, sparse) matrix
    matrix = Tensor(
        shape=(5, 5),
        index=TensorIndex(
            tensor_format="oo",
            mode_indices=[
                [torch.tensor([0, 1, 2, 2, 3, 4])],
                [torch.tensor([0, 1, 2, 3, 3, 4])],
            ],
        ),
        value=torch.tensor([1, 2, 3, 4, 0, 5]),
    )
    matrix.to_sparse("ss")

    assert len(matrix.index.mode_indices) == 2

    assert matrix.index.mode_indices[0][0].tolist() == [0, 4]
    assert matrix.index.mode_indices[0][1].tolist() == [0, 1, 2, 4]
    assert matrix.index.mode_indices[1][0].tolist() == [0, 1, 2, 4, 5]
    assert matrix.index.mode_indices[1][1].tolist() == [0, 1, 2, 3, 4]

    assert matrix.values.tolist() == [1, 2, 3, 4, 5]


def test_2d_ds_oo_random():
    torch_tensor = torch.rand(10, 10)
    # Randomly sparsify
    torch_tensor[torch.rand(10, 10) < 0.9] = 0
    scorch_tensor = Tensor.from_torch(torch_tensor).to_sparse("oo")
    scorch_tensor = scorch_tensor.to_sparse("ds")

    assert torch.allclose(torch_tensor, scorch_tensor.to_torch())


def test_2d_ds_oo_2():
    matrix = Tensor.from_coo(
        indices=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 5, 5, 6, 9],
                              [6, 0, 9, 2, 9, 0, 6, 0, 7, 1, 5]]),
        values=torch.tensor([0.5713, 0.4238, 0.6953, 0.9270, 0.1595, 0.9325, 0.2542,
                             0.6638, 0.8833, 0.2976, 0.8072]),
        shape=(10, 10),
    )
    matrix.to_sparse("ds")

    assert len(matrix.index.mode_indices) == 2

    assert matrix.index.mode_indices[1][0].tolist() == [0, 1, 3, 5, 7, 7, 9, 10, 10, 10, 11]
    assert matrix.index.mode_indices[1][1].tolist() == [6, 0, 9, 2, 9, 0, 6, 0, 7, 1, 5]

    assert matrix.values.tolist() == [0.5713, 0.4238, 0.6953, 0.9270, 0.1595, 0.9325, 0.2542,
                                      0.6638, 0.8833, 0.2976, 0.8072]


def test_2d_ds_oo():
    # Test converting a COO matrix to a CSR (dense, sparse) matrix
    matrix = Tensor(
        shape=(5, 5),
        index=TensorIndex(
            tensor_format="oo",
            mode_indices=[
                [torch.tensor([0, 1, 2, 2, 3, 4])],
                [torch.tensor([0, 1, 2, 3, 3, 4])],
            ],
        ),
        value=torch.tensor([1, 2, 3, 4, 0, 5]),
    )
    matrix.to_sparse("ds")

    assert len(matrix.index.mode_indices) == 2

    assert matrix.index.mode_indices[1][0].tolist() == [0, 1, 2, 4, 4, 5]
    assert matrix.index.mode_indices[1][1].tolist() == [0, 1, 2, 3, 4]

    assert matrix.values.tolist() == [1, 2, 3, 4, 5]


def test_2d_oo_dd():
    matrix = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    matrix = Tensor.from_torch(matrix)

    matrix = matrix.to_sparse("oo")

    assert len(matrix.index.mode_indices) == 2

    assert matrix.index.mode_indices[0][0].tolist() == [0, 1, 2, 3, 4]

    assert matrix.index.mode_indices[1][0].tolist() == [0, 1, 2, 3, 4]

    assert matrix.values.tolist() == [1, 2, 3, 4, 5]


def test_2d_dd_ds():
    # Test converting a CSR matrix to a dense matrix
    matrix = Tensor(
        shape=(5, 5),
        index=TensorIndex(
            tensor_format="oo",
            mode_indices=[
                [torch.tensor([0, 1, 2, 2, 3, 4])],
                [torch.tensor([0, 1, 2, 3, 3, 4])],
            ],
        ),
        value=torch.tensor([1, 2, 3, 4, 0, 5]),
    )
    csr_matrix = matrix.to_sparse("ds")

    dense_matrix = csr_matrix.to_torch(in_place=False)


def test_2d_dd_oo():
    # Test converting a COO matrix to a dense matrix
    matrix = Tensor(
        shape=(5, 5),
        index=TensorIndex(
            tensor_format="oo",
            mode_indices=[
                [torch.tensor([0, 1, 2, 2, 3, 4])],
                [torch.tensor([0, 1, 2, 3, 3, 4])],
            ],
        ),
        value=torch.tensor([1, 2, 3, 4, 0, 5]),
    )
    matrix.to_sparse("dd")

    assert matrix.values.tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0,
        4.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        5.0,
    ]

    print(matrix.values.tolist())


def test_2d_ss_dd():
    # Test converting a dense matrix to a sparse matrix
    # matrix = torch.rand(5, 5)
    # Sparsify the matrix
    # matrix[torch.rand(5, 5) > 0.5] = 0
    matrix = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 4, 5],
            [0, 0, 0, 6, 7],
            [0, 0, 0, 0, 8],
        ]
    )
    matrix = Tensor.from_torch(matrix)
    matrix.to_sparse("ss")

    assert len(matrix.index.mode_indices) == 2

    assert matrix.index.mode_indices[0][0].tolist() == [0, 5]
    assert matrix.index.mode_indices[0][1].tolist() == [0, 1, 2, 3, 4]
    assert matrix.index.mode_indices[1][0].tolist() == [0, 1, 2, 5, 7, 8]
    assert matrix.index.mode_indices[1][1].tolist() == [0, 1, 2, 3, 4, 3, 4, 4]

    assert matrix.values.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
