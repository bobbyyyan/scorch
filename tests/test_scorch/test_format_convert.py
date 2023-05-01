import torch

from src.scorch import Tensor
from src.scorch.storage import TensorIndex
from src.scorch.utils import get_format_from_list


def test_2d_ss_oo():
    # Test converting a COO matrix to a DCSR (sparse, sparse) matrix
    matrix = Tensor(
        shape=(5, 5),
        index=TensorIndex(
            tensor_format=get_format_from_list(["o", "o"]),
            mode_indices=[
                [torch.tensor([0, 1, 2, 2, 3, 4])],
                [torch.tensor([0, 1, 2, 3, 3, 4])],
            ],
        ),
        value=torch.tensor([1, 2, 3, 4, 0, 5]),
    )
    matrix.to_sparse(get_format_from_list(["s", "s"]))

    assert len(matrix.index.mode_indices) == 2

    assert matrix.index.mode_indices[0][0].tolist() == [0, 4]
    assert matrix.index.mode_indices[0][1].tolist() == [0, 1, 2, 4]
    assert matrix.index.mode_indices[1][0].tolist() == [0, 1, 2, 4, 5]
    assert matrix.index.mode_indices[1][1].tolist() == [0, 1, 2, 3, 4]

    assert matrix.values.tolist() == [1, 2, 3, 4, 5]


def test_2d_ds_oo():
    # Test converting a COO matrix to a CSR (dense, sparse) matrix
    matrix = Tensor(
        shape=(5, 5),
        index=TensorIndex(
            tensor_format=get_format_from_list(["o", "o"]),
            mode_indices=[
                [torch.tensor([0, 1, 2, 2, 3, 4])],
                [torch.tensor([0, 1, 2, 3, 3, 4])],
            ],
        ),
        value=torch.tensor([1, 2, 3, 4, 0, 5]),
    )
    matrix.to_sparse(get_format_from_list(["d", "s"]))

    assert len(matrix.index.mode_indices) == 2

    assert matrix.index.mode_indices[1][0].tolist() == [0, 1, 2, 4, 4, 5]
    assert matrix.index.mode_indices[1][1].tolist() == [0, 1, 2, 3, 4]

    assert matrix.values.tolist() == [1, 2, 3, 4, 5]


def test_2d_dd_oo():
    # Test converting a COO matrix to a dense matrix
    matrix = Tensor(
        shape=(5, 5),
        index=TensorIndex(
            tensor_format=get_format_from_list(["o", "o"]),
            mode_indices=[
                [torch.tensor([0, 1, 2, 2, 3, 4])],
                [torch.tensor([0, 1, 2, 3, 3, 4])],
            ],
        ),
        value=torch.tensor([1, 2, 3, 4, 0, 5]),
    )
    matrix.to_sparse(get_format_from_list(["d", "d"]))


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
    matrix.to_sparse(get_format_from_list(["s", "s"]))

    assert len(matrix.index.mode_indices) == 2

    assert matrix.index.mode_indices[0][0].tolist() == [0, 5]
    assert matrix.index.mode_indices[0][1].tolist() == [0, 1, 2, 3, 4]
    assert matrix.index.mode_indices[1][0].tolist() == [0, 1, 2, 5, 7, 8]
    assert matrix.index.mode_indices[1][1].tolist() == [0, 1, 2, 3, 4, 3, 4, 4]

    assert matrix.values.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
