import pytest
import torch

import scorch
from scorch import STensor


def _make_sparse_matrix(n: int, keep_prob: float = 0.1) -> torch.Tensor:
    dense = torch.rand(n, n)
    mask = torch.rand(n, n) < keep_prob
    return dense * mask


def _run_matmul(
    a_torch: torch.Tensor,
    b_torch: torch.Tensor,
    fmt_a: str,
    fmt_b: str,
    mode_order_a,
    mode_order_b,
    output_fmt: str = "dd",
    use_cache: bool = True,
) -> torch.Tensor:
    a = STensor.from_torch(a_torch, "A", mode_order=mode_order_a)
    b = STensor.from_torch(b_torch, "B", mode_order=mode_order_b)

    if fmt_a != "dd":
        a = a.to_sparse(fmt_a)
    if fmt_b != "dd":
        b = b.to_sparse(fmt_b)

    out = scorch.matmul(a, b, format=output_fmt, use_cache=use_cache)
    return out if isinstance(out, torch.Tensor) else out.to_torch()


@pytest.mark.parametrize("use_cache", [True, False])
def test_gap_spmm_ds_dd_transposed_mode_order_wrong_result(use_cache):
    torch.manual_seed(0)
    n = 32
    a_torch = _make_sparse_matrix(n, keep_prob=0.1)
    b_torch = torch.rand(n, n)

    result = _run_matmul(
        a_torch=a_torch,
        b_torch=b_torch,
        fmt_a="ds",
        fmt_b="dd",
        mode_order_a=[1, 0],
        mode_order_b=[1, 0],
        output_fmt="dd",
        use_cache=use_cache,
    )

    expected = torch.matmul(a_torch, b_torch)
    assert torch.allclose(result, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("use_cache", [True, False])
def test_gap_spmm_oo_dd_mixed_mode_order_wrong_result(use_cache):
    torch.manual_seed(1)
    n = 24
    a_torch = _make_sparse_matrix(n, keep_prob=0.1)
    b_torch = torch.rand(n, n)

    result = _run_matmul(
        a_torch=a_torch,
        b_torch=b_torch,
        fmt_a="oo",
        fmt_b="dd",
        mode_order_a=[1, 0],
        mode_order_b=[0, 1],
        output_fmt="ds",
        use_cache=use_cache,
    )

    expected = torch.matmul(a_torch, b_torch)
    assert torch.allclose(result, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("use_cache", [True, False])
def test_gap_spgemm_ds_ds_transposed_mode_order_wrong_result(use_cache):
    torch.manual_seed(2)
    n = 32
    a_torch = _make_sparse_matrix(n, keep_prob=0.05)
    b_torch = _make_sparse_matrix(n, keep_prob=0.05)

    result = _run_matmul(
        a_torch=a_torch,
        b_torch=b_torch,
        fmt_a="ds",
        fmt_b="ds",
        mode_order_a=[1, 0],
        mode_order_b=[1, 0],
        output_fmt="dd",
        use_cache=use_cache,
    )

    expected = torch.matmul(a_torch, b_torch)
    assert torch.allclose(result, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("use_cache", [True, False])
def test_gap_spgemm_oo_oo_transposed_mode_order_wrong_result(use_cache):
    torch.manual_seed(3)
    n = 24
    a_torch = _make_sparse_matrix(n, keep_prob=0.05)
    b_torch = _make_sparse_matrix(n, keep_prob=0.05)

    result = _run_matmul(
        a_torch=a_torch,
        b_torch=b_torch,
        fmt_a="oo",
        fmt_b="oo",
        mode_order_a=[1, 0],
        mode_order_b=[1, 0],
        output_fmt="dd",
        use_cache=use_cache,
    )

    expected = torch.matmul(a_torch, b_torch)
    assert torch.allclose(result, expected, atol=1e-3, rtol=1e-3)
