import pytest
import torch

import scorch
from scorch import STensor


def _run_dense_stensor_matmul(mode_order_a, mode_order_b, use_cache: bool):
    torch.manual_seed(0)
    n = 16
    a_torch = torch.rand(n, n)
    b_torch = torch.rand(n, n)

    a = STensor.from_torch(a_torch, "A", mode_order=mode_order_a)
    b = STensor.from_torch(b_torch, "B", mode_order=mode_order_b)

    out = scorch.matmul(a, b, format="dd", use_cache=use_cache)
    out_torch = out if isinstance(out, torch.Tensor) else out.to_torch()

    expected = torch.matmul(a_torch, b_torch)
    assert torch.allclose(out_torch, expected, atol=1e-3, rtol=1e-3)


@torch.no_grad()
@pytest.mark.parametrize("use_cache", [True, False])
def test_known_gap_dense_stensor_matmul_default_mode_order(use_cache):
    _run_dense_stensor_matmul(mode_order_a=[0, 1], mode_order_b=[0, 1], use_cache=use_cache)


@torch.no_grad()
@pytest.mark.parametrize("use_cache", [True, False])
def test_known_gap_dense_stensor_matmul_transposed_mode_order(use_cache):
    _run_dense_stensor_matmul(mode_order_a=[1, 0], mode_order_b=[0, 1], use_cache=use_cache)
