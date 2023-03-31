import torch

from src.scorch.compiler.cin import IndexVar, TensorVar, ForAll
from src.scorch.compiler.cin_lowerer import CINLowerer
from src.scorch.compiler.codegen import LLIRLowerer
from src.scorch import Tensor, einsum, TensorFormat
from src.scorch.storage import TensorIndex
from src.scorch.utils import get_format_from_list


def test_2d_ss_oo():
    # Test converting a COO matrix to a sparse-sparse matrix
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
