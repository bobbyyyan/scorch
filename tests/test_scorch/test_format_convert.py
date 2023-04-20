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
    matrix = Tensor.from_torch(torch.randn(5, 5))
    matrix.to_sparse(get_format_from_list(["s", "s"]))
