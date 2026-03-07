import pytest
import torch

import scorch
import scorch.ops as scorch_ops
from scorch import STensor


@pytest.mark.parametrize("use_cache", [True, False])
def test_sparse_stensor_spmm_does_not_call_torch_matmul(use_cache, monkeypatch):
    torch.manual_seed(0)
    n = 32
    a_torch = torch.rand(n, n) * (torch.rand(n, n) < 0.1)
    b_torch = torch.rand(n, n)
    expected = torch.matmul(a_torch, b_torch)

    a = STensor.from_torch(a_torch, "A").to_sparse("ds")
    b = STensor.from_torch(b_torch, "B").to_dense()

    def _forbidden_matmul(*args, **kwargs):
        raise AssertionError("torch.matmul should not be used for sparse STensor matmul")

    monkeypatch.setattr(scorch_ops.torch, "matmul", _forbidden_matmul)

    out = scorch.matmul(a, b, format="dd", use_cache=use_cache)
    out_torch = out if isinstance(out, torch.Tensor) else out.to_torch()
    assert torch.allclose(out_torch, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("use_cache", [True, False])
def test_sparse_stensor_spgemm_does_not_call_torch_matmul(use_cache, monkeypatch):
    torch.manual_seed(1)
    n = 32
    a_torch = torch.rand(n, n) * (torch.rand(n, n) < 0.05)
    b_torch = torch.rand(n, n) * (torch.rand(n, n) < 0.05)
    expected = torch.matmul(a_torch, b_torch)

    a = STensor.from_torch(a_torch, "A").to_sparse("ds")
    b = STensor.from_torch(b_torch, "B").to_sparse("ds")

    def _forbidden_matmul(*args, **kwargs):
        raise AssertionError("torch.matmul should not be used for sparse STensor matmul")

    monkeypatch.setattr(scorch_ops.torch, "matmul", _forbidden_matmul)

    out = scorch.matmul(a, b, format="dd", use_cache=use_cache)
    out_torch = out if isinstance(out, torch.Tensor) else out.to_torch()
    assert torch.allclose(out_torch, expected, atol=1e-3, rtol=1e-3)
