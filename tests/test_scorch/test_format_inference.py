import torch

import scorch
from scorch import STensor, einsum


def test_spmm_oo_dd_fmtinf():
    n = 64
    tensor_a_torch = torch.rand(n, n)
    tensor_b_torch = torch.rand(n, n)

    a_scorch = STensor.from_torch(tensor_a_torch, "A").to_sparse("oo")
    b_scorch = STensor.from_torch(tensor_b_torch, "B").to_dense()

    # result = einsum("ik,kj->ij", a_scorch, b_scorch)
    result = scorch.matmul(a_scorch, b_scorch)

    result_torch = torch.matmul(tensor_a_torch, tensor_b_torch)

    if isinstance(result, torch.Tensor):
        assert torch.allclose(result, result_torch)
    elif isinstance(result, STensor):
        assert torch.allclose(result.to_torch(), result_torch)
    else:
        raise ValueError(f"Unexpected result type: {type(result)}")


def test_spmm_ds_dd_fmtinf():
    n = 64
    tensor_a_torch = torch.rand(n, n)
    tensor_b_torch = torch.rand(n, n)

    a_scorch = STensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_scorch = STensor.from_torch(tensor_b_torch, "B").to_dense()

    # result = einsum("ik,kj->ij", a_scorch, b_scorch)
    result = scorch.matmul(a_scorch, b_scorch)

    result_torch = torch.matmul(tensor_a_torch, tensor_b_torch)

    if isinstance(result, torch.Tensor):
        assert torch.allclose(result, result_torch)
    elif isinstance(result, STensor):
        assert torch.allclose(result.to_torch(), result_torch)
    else:
        raise ValueError(f"Unexpected result type: {type(result)}")
