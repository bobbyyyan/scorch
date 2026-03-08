import torch

from scorch import STensor
from scorch.prebuilt_kernels import resolve_prebuilt_matmul


def test_resolve_prebuilt_spmm_float64_symbol():
    torch.manual_seed(0)
    n = 16
    a_torch = (torch.rand(n, n, dtype=torch.float64) * (torch.rand(n, n) < 0.2)).contiguous()
    b_torch = torch.rand(n, n, dtype=torch.float64)

    a = STensor.from_torch(a_torch.to_sparse_csr())
    b = STensor.from_torch(b_torch).to_dense()

    resolved = resolve_prebuilt_matmul(a, b)
    assert resolved is not None
    assert resolved.symbol_name == "prebuilt_spmm_csr_f64"


def test_resolve_prebuilt_spmv_int64_symbol():
    a_torch = torch.tensor(
        [
            [1, 0, 2, 0],
            [0, 3, 0, 4],
            [0, 0, 0, 0],
            [5, 0, 0, 6],
        ],
        dtype=torch.int64,
    )
    x_torch = torch.tensor([1, 2, 3, 4], dtype=torch.int64)

    a = STensor.from_torch(a_torch).to_sparse("ds")
    x = STensor.from_torch(x_torch).to_dense()

    resolved = resolve_prebuilt_matmul(a, x)
    assert resolved is not None
    assert resolved.symbol_name == "prebuilt_spmv_csr_i64"


def test_resolve_prebuilt_skips_output_format_mismatch():
    torch.manual_seed(1)
    n = 8
    a_torch = torch.rand(n, n, dtype=torch.float32) * (torch.rand(n, n) < 0.3)
    b_torch = torch.rand(n, n, dtype=torch.float32) * (torch.rand(n, n) < 0.3)

    a = STensor.from_torch(a_torch).to_sparse("ds")
    b = STensor.from_torch(b_torch).to_sparse("ds")

    resolved = resolve_prebuilt_matmul(a, b, output_format="dd")
    assert resolved is None
