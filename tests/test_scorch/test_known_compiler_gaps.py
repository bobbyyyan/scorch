import pytest
import torch

from scorch import STensor
from scorch.compiler.cin import (
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_analysis import canonical_cin_dump
from scorch.compiler.compile_options import CompileOptions
from scorch.exceptions import TensorIndexError, defect_code
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
    cin_before = canonical_cin_dump(cin_stmt)
    input_mode_orders = (a.mode_order, b.mode_order)
    result = lower_and_exec_cin(cin_stmt, (n, n), a, b)
    expected = torch.matmul(a_torch, b_torch)
    assert torch.allclose(result.to_torch(), expected, atol=1e-4, rtol=1e-4)
    assert canonical_cin_dump(cin_stmt) == cin_before
    assert (a.mode_order, b.mode_order) == input_mode_orders


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


@pytest.mark.xfail(
    raises=TensorIndexError,
    strict=True,
    reason="a result stored sd with a reduction loop between its compressed row level "
    "and its dense leaf appends the row once per iteration of that loop; the "
    "result's validation rejects it (compressed_coordinates_unordered)",
)
@pytest.mark.parametrize("arm", [0, 1])
def test_known_gap_sd_result_with_a_reduction_below_its_compressed_level(
    monkeypatch, arm
):
    """A = B C with A stored sd (compressed rows, dense columns), B and C stored ds,
    the loop order i, k, j and a plain assignment (codegen/test_matmul_operations.py's
    test_spmm_sd_ds_ds, which only lowers it), executed in both automatic arms.  The
    kernel appends row i of A inside the loop over k, once per nonzero B[i, k], so the
    row level holds repeated coordinates.  The lowering refuses a reduction loop
    around the append only when the result's last level is compressed or coordinate
    (unscheduled_sparse_result_reduction); here the last level is dense.  The shipped
    compiler the series starts from fails the same way.  When the lowering computes
    this program, or refuses it with a code, this test fails under strict=True:
    update it then.  Compared with a float64 reference to a relative 1e-12; every
    operand entry is nonnegative, so no sum cancels."""
    n, inner, m = 7, 6, 5
    generator = torch.Generator().manual_seed(16)
    b = torch.rand(n, inner, generator=generator, dtype=torch.float64)
    b[b < 0.5] = 0.0
    b[1] = 0.0
    c = torch.rand(inner, m, generator=generator, dtype=torch.float64)
    c[c < 0.4] = 0.0

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    A = TensorVar("A", fmt=["sparse", "dense"], shape=(n, m), dtype=torch.float64)
    B = TensorVar("B", fmt=["dense", "sparse"], shape=(n, inner), dtype=torch.float64)
    C = TensorVar("C", fmt=["dense", "sparse"], shape=(inner, m), dtype=torch.float64)
    A[i, j] = B[i, k] * C[k, j]
    cin_stmt = ForAll(i, ForAll(k, ForAll(j, A._assignment)))

    monkeypatch.setenv("SCORCH_REGBLOCK", str(arm))
    assert CompileOptions.from_environment().scheduler.regblock_enabled == bool(arm)
    try:
        result = lower_and_exec_cin(
            cin_stmt,
            (n, m),
            STensor.from_torch(b).to_sparse("ds"),
            STensor.from_torch(c).to_sparse("ds"),
        )
    except TensorIndexError as error:
        # Only this rejection is the gap; any other code fails the test (an
        # AssertionError is not the xfail's TensorIndexError).
        assert defect_code(error) == "compressed_coordinates_unordered", error
        raise

    got = result.to_torch()
    got = got if got.layout == torch.strided else got.to_dense()
    assert got.shape == (n, m)
    assert torch.isfinite(got).all(), got
    assert torch.allclose(got, b @ c, rtol=1e-12, atol=0.0), (got, b @ c)
