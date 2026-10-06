"""Rows too wide for a whole-row partition, computed in segments.

A row holding more than 1/64 of a matrix's nonzeros (on matrices of at least 150K nonzeros) is a
wide row: the plan finds those once (``SpmmCsrPlan.wide_rows``), and on a call that runs a
specialised row loop on more than one worker the v2 kernels compute each wide row in segments,
every segment with the same loop as an ordinary row, and sum them afterwards in segment order.
These pin:

* the product matches torch's at every width the specialised loops cover (1, 2, 3, 4, 8, 16, 32
  and 64), in both dtypes, with wide rows at the start, in the middle and last;
* the plan reports exactly the rows over the threshold, and none on a matrix without one;
* the answer does not depend on the thread count -- the segments are a function of the matrix --
  so two thread counts give bit-identical outputs.
"""
import pytest
import torch

import scorch
from scorch import STensor
from scorch.plan import plans_of

pytest.importorskip("scorch_ops")


def _wide_matrix(dtype, seed=0):
    """3000 rows of 0-60 nonzeros, rows 0, 1500 and 2999 widened to 40K, 30K and 12K nonzeros."""
    g = torch.Generator().manual_seed(seed)
    rows, cols = 3000, 60000
    lengths = torch.randint(0, 61, (rows,), generator=g)
    lengths[0], lengths[1500], lengths[2999] = 40000, 30000, 12000
    crow = torch.zeros(rows + 1, dtype=torch.long)
    crow[1:] = torch.cumsum(lengths, 0)
    col = torch.cat([torch.randperm(cols, generator=g)[: int(n)].sort().values for n in lengths])
    vals = torch.randn(int(crow[-1]), generator=g, dtype=torch.float64).to(dtype)
    return torch.sparse_csr_tensor(crow, col, vals, size=(rows, cols))


def _planned(st, b, calls=3):
    out = None
    for _ in range(calls):  # plans install on the second call
        out = scorch.matmul(st, b)
    return out


def _plans(st):
    return list(plans_of(st).values())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 8, 16, 32, 64])
def test_wide_rows_match_torch(dtype, width):
    a = _wide_matrix(dtype)
    assert a._nnz() >= 150000
    b = torch.randn(a.shape[1], width, generator=torch.Generator().manual_seed(width), dtype=dtype)
    st = STensor.from_torch(a)
    got = _planned(st, b)
    want = torch.sparse.mm(a, b)
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3)
    plans = _plans(st)
    assert plans and any(p.served > 0 for p in plans)


def test_plan_finds_exactly_the_wide_rows():
    a = _wide_matrix(torch.float32)
    st = STensor.from_torch(a)
    _planned(st, torch.randn(a.shape[1], 4))
    nnz = a._nnz()
    crow = a.crow_indices()
    wide = sorted(i for i in range(a.shape[0]) if int(crow[i + 1] - crow[i]) > nnz // 64)
    assert wide == [0, 1500, 2999]
    plans = _plans(st)
    assert any(list(p.wide_rows) == wide for p in plans)
    # Every wide row is at least one segment, and these are longer than any one segment can be.
    assert all(p.wide_segments > len(wide) for p in plans)


def test_no_wide_row_no_segments():
    g = torch.Generator().manual_seed(5)
    rows, cols, deg = 4000, 5000, 50
    crow = torch.arange(rows + 1) * deg
    col = torch.randint(0, cols, (rows * deg,), generator=g)
    a = torch.sparse_coo_tensor(
        torch.stack([torch.repeat_interleave(torch.arange(rows), deg), col]),
        torch.randn(rows * deg, generator=g), (rows, cols)).coalesce().to_sparse_csr()
    st = STensor.from_torch(a)
    _planned(st, torch.randn(cols, 4))
    assert all(len(p.wide_rows) == 0 and p.wide_segments == 0 for p in _plans(st))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_one_operand_two_widths(dtype):
    """Segments are cut per free dimension: two widths on one operand are two plans, each right."""
    a = _wide_matrix(dtype, seed=2)
    st = STensor.from_torch(a)
    g = torch.Generator().manual_seed(7)
    b1 = torch.randn(a.shape[1], 1, generator=g, dtype=dtype)
    b32 = torch.randn(a.shape[1], 32, generator=g, dtype=dtype)
    for _ in range(3):
        got1, got32 = scorch.matmul(st, b1), scorch.matmul(st, b32)
    torch.testing.assert_close(got1, torch.sparse.mm(a, b1), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(got32, torch.sparse.mm(a, b32), atol=1e-3, rtol=1e-3)
    assert len(_plans(st)) == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [1, 4, 32])
def test_answer_does_not_depend_on_thread_count(dtype, width):
    a = _wide_matrix(dtype, seed=1)
    b = torch.randn(a.shape[1], width, generator=torch.Generator().manual_seed(3), dtype=dtype)
    before = torch.get_num_threads()
    if before < 4:
        pytest.skip("needs 4 threads")
    try:
        torch.set_num_threads(2)
        two = _planned(STensor.from_torch(a), b)
        torch.set_num_threads(before)
        many = _planned(STensor.from_torch(a), b)
    finally:
        torch.set_num_threads(before)
    assert torch.equal(two, many)
