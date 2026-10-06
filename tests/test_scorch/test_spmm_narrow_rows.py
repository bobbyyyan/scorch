"""Products whose rows are short or empty: the work the thread policy is told for them and the
float64 width-1 row loop.

``scorch_spmm_work`` is the work the drop-in SpMM states to ``scorch_spmm_nthreads``: a nonzero is
credited its width floored at 16 (its gather touches a whole cache line), an empty row the workers
zero its output width. Rows themselves are not credited (see the function's comment for the
measurement). These pin, through the exported function (never a re-derivation), the formula at
narrow and wide widths, and that a matrix of mostly empty rows and a few dozen nonzeros
(flowmeter5_B: 9669 rows, 24 nonzeros) stays on one worker, where a 5 us call does not pay for
waking the pool.

The float64 width-1 loop (``scorch_spmm_rows_short_k1_f64``) sends a row of eight or more nonzeros
to the exact-width kernel and a shorter one to a scalar sum, and zeroes a run of empty rows with
one memset. The correctness cases put rows of every length class -- empty runs at the start, in
the middle and at the end, rows of 1-7 nonzeros, and rows of 8, 9 and 40 -- into one matrix,
against torch's own product.
"""
import pytest
import torch

import scorch
from scorch import STensor

scorch_ops = pytest.importorskip("scorch_ops")
if not hasattr(scorch_ops, "scorch_spmm_work"):
    pytest.skip("scorch_ops predates scorch_spmm_work", allow_module_level=True)

GRAIN = 150000  # SCORCH_GRAIN_SPMM


@pytest.mark.parametrize("width", [1, 2, 4, 15, 16, 32, 256])
def test_formula(width):
    w = max(width, 16)
    assert scorch_ops.scorch_spmm_work(1000, 300, width, width) == 1000 * w + 300 * width


def test_a_nearly_empty_matrix_stays_on_one_worker():
    rows, nnz, empty, pool = 9669, 24, 9645, 24  # flowmeter5_B
    work = scorch_ops.scorch_spmm_work(nnz, empty, 1, 1)
    assert work < GRAIN
    assert scorch_ops.scorch_spmm_nthreads(work, rows, pool, nnz, nnz, True) == 1


def _mixed_rows(cols, seed):
    """A CSR whose rows cover every length class the width-1 loops branch on."""
    g = torch.Generator().manual_seed(seed)
    lengths = [0, 0, 0, 1, 2, 0, 3, 4, 5, 6, 7, 8, 0, 9, 40, 1, 0, 0, 7, 8, 2, 0]
    lengths = lengths * 37 + [0, 0, 0]
    crow = [0]
    cols_list = []
    for n in lengths:
        cols_list.append(torch.randperm(cols, generator=g)[:n].sort().values)
        crow.append(crow[-1] + n)
    col = torch.cat(cols_list)
    vals = torch.randn(len(col), generator=g, dtype=torch.float64)
    return torch.tensor(crow), col, vals, len(lengths)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [1, 2])
def test_mixed_row_lengths_match_torch(dtype, width):
    cols = 500
    crow, col, vals, rows = _mixed_rows(cols, seed=width)
    a = torch.sparse_csr_tensor(crow, col, vals.to(dtype), size=(rows, cols))
    b = torch.randn(cols, width, generator=torch.Generator().manual_seed(7), dtype=dtype)
    got = scorch.matmul(STensor.from_torch(a), b)
    want = torch.sparse.mm(a, b)
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3)
