import pytest
import torch

from scorch import STensor
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.ops import lower_and_exec_cin


def _build_spmm_cin(
    fmt_out: str,
    fmt_a: str,
    fmt_b: str,
    mode_order_out,
    mode_order_a,
    mode_order_b,
):
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    C = TensorVar("C", fmt=fmt_out, mode_order=mode_order_out)
    A = TensorVar("A", fmt=fmt_a, mode_order=mode_order_a)
    B = TensorVar("B", fmt=fmt_b, mode_order=mode_order_b)

    stmt = TensorAssign(C[i, k], A[i, j] * B[j, k], op=Operation.ADD)
    return ForAll(i, ForAll(j, ForAll(k, stmt)))


def _build_broadcast_rhs_vec_cin(
    fmt_out: str,
    fmt_a: str,
    fmt_b: str,
    mode_order_out,
    mode_order_a,
):
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    C = TensorVar("C", fmt=fmt_out, mode_order=mode_order_out)
    A = TensorVar("A", fmt=fmt_a, mode_order=mode_order_a)
    b = TensorVar("b", fmt=fmt_b, mode_order=[0])

    stmt = TensorAssign(C[i, k], A[i, j] * b[j], op=Operation.ADD)
    return ForAll(i, ForAll(j, ForAll(k, stmt)))


def _build_broadcast_lhs_vec_cin(
    fmt_out: str,
    fmt_a: str,
    fmt_b: str,
    mode_order_out,
    mode_order_b,
):
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    C = TensorVar("C", fmt=fmt_out, mode_order=mode_order_out)
    a = TensorVar("a", fmt=fmt_a, mode_order=[0])
    B = TensorVar("B", fmt=fmt_b, mode_order=mode_order_b)

    stmt = TensorAssign(C[i, k], a[i] * B[j, k], op=Operation.ADD)
    return ForAll(i, ForAll(j, ForAll(k, stmt)))


@pytest.mark.xfail(
    reason="Known gap: mode-order interactions can fail for transposed sparse/dense SpMM",
    strict=False,
)
def test_known_gap_spmm_transposed_mode_order():
    torch.manual_seed(0)
    n = 16
    a_torch = torch.rand(n, n)
    b_torch = torch.rand(n, n) * (torch.rand(n, n) > 0.9)

    a = STensor.from_torch(a_torch, "A", mode_order=[1, 0]).to_dense()
    b = STensor.from_torch(b_torch, "B", mode_order=[1, 0]).to_sparse("ds")

    cin_stmt = _build_spmm_cin(
        fmt_out="dd",
        fmt_a="dd",
        fmt_b="ds",
        mode_order_out=[1, 0],
        mode_order_a=[1, 0],
        mode_order_b=[1, 0],
    )
    result = lower_and_exec_cin(cin_stmt, (n, n), a, b)
    expected = torch.matmul(a_torch, b_torch)
    assert torch.allclose(result.to_torch(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.xfail(
    reason="Known gap: 3-loop broadcasted reduction C[i,k]+=A[i,j]*b[j] has unsupported sparse paths",
    strict=False,
)
def test_known_gap_broadcast_rhs_vector():
    torch.manual_seed(1)
    n = 16
    a_torch = torch.rand(n, n) * (torch.rand(n, n) > 0.9)
    b_torch = torch.rand(n) * (torch.rand(n) > 0.9)

    a = STensor.from_torch(a_torch, "A", mode_order=[0, 1]).to_sparse("ds")
    b = STensor.from_torch(b_torch, "b").to_sparse("s")

    cin_stmt = _build_broadcast_rhs_vec_cin(
        fmt_out="dd",
        fmt_a="ds",
        fmt_b="s",
        mode_order_out=[0, 1],
        mode_order_a=[0, 1],
    )
    result = lower_and_exec_cin(cin_stmt, (n, n), a, b)

    row = torch.matmul(a_torch, b_torch).reshape(n, 1)
    expected = row.repeat(1, n)
    assert torch.allclose(result.to_torch(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.xfail(
    reason="Known gap: 3-loop broadcasted reduction C[i,k]+=a[i]*B[j,k] with sparse vector is unstable",
    strict=False,
)
def test_known_gap_broadcast_lhs_vector():
    torch.manual_seed(2)
    n = 16
    a_torch = torch.rand(n) * (torch.rand(n) > 0.9)
    b_torch = torch.rand(n, n)

    a = STensor.from_torch(a_torch, "a").to_sparse("s")
    b = STensor.from_torch(b_torch, "B", mode_order=[1, 0]).to_dense()

    cin_stmt = _build_broadcast_lhs_vec_cin(
        fmt_out="dd",
        fmt_a="s",
        fmt_b="dd",
        mode_order_out=[1, 0],
        mode_order_b=[1, 0],
    )
    result = lower_and_exec_cin(cin_stmt, (n, n), a, b)

    expected = a_torch.reshape(n, 1) * b_torch.sum(dim=0).reshape(1, n)
    assert torch.allclose(result.to_torch(), expected, atol=1e-4, rtol=1e-4)
