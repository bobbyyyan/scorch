"""Products whose rows are short or empty: the work the thread policy is told for them, the
float64 width-1 row loop, and how runs of empty rows are zeroed.

``scorch_spmm_work`` is the work the drop-in SpMM states to ``scorch_spmm_nthreads``: a nonzero is
credited its width floored at 16 (its gather touches a whole cache line), an empty row the workers
zero its output width. Rows themselves are not credited (see the function's comment for the
measurement). These pin, through the exported function (never a re-derivation), the formula at
narrow and wide widths, and that a matrix of mostly empty rows and a few dozen nonzeros
(flowmeter5_B: 9669 rows, 24 nonzeros) stays on one worker, where a 5 us call does not pay for
waking the pool.

The float64 width-1 loop (``scorch_spmm_rows_short_k1_f64``) sends a row of eight or more nonzeros
to the exact-width kernel and a shorter one to a scalar sum, and zeroes a run of empty rows as it
walks to the run's end. The correctness cases put rows of every length class -- empty runs at the start, in
the middle and at the end, rows of 1-7 nonzeros, and rows of 8, 9 and 40 -- into one matrix,
against torch's own product.

Below five vectors a row, a run's empty rows are zeroed by the walk that finds its end, each row
written with the stores that write a computed row, until the run has covered 256 bytes or eight
rows, whichever is more rows; the rest of a longer run is found eight row bounds at a time and
zeroed with one memset. Wider rows take one memset per run. The switch is at 64 rows for float32
width 1, 32 for float64 width 1 and float32 width 2, 21 for float32 width 3, 16 for float64 width 2,
10 for float64 width 3 and 8 above that. The run-length cases put runs of 1 to 1000 empty rows
between written rows at widths 1 to 32 -- on both sides of each of those switches, and ending at
every position inside an eight-bound step -- so the runs span 4 bytes to past 100 KB, and the output
is first filled with a product of the same size so a missed zero would read back as a stale value
rather than as fresh-page zeros. The nearly empty
case runs on several workers from width 2 up, so there runs also cross chunk boundaries; at width 1
its empty rows do not credit enough work for a second worker.
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


def test_a_nearly_empty_matrix_stays_on_one_worker(torch_threads):
    # At a pool of 24, which torch's count must allow, or "one worker" holds for every product.
    torch_threads(24)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8])
def test_nearly_empty_matches_torch(dtype, width, torch_threads):
    # More rows than nonzeros and empty runs of every length. From width 2 up the empty rows credit
    # enough work for several workers, so runs also cross chunk boundaries; the policy is asked,
    # so a change to it that left this on one worker fails here rather than silently narrowing it.
    # The policy is asked at a pool of 24, which torch's count must allow.
    torch_threads(24)
    rows, cols, nnz = 40000, 3000, 5000
    g = torch.Generator().manual_seed(3)
    row = torch.randint(0, rows, (nnz,), generator=g)
    col = torch.randint(0, cols, (nnz,), generator=g)
    a = torch.sparse_coo_tensor(torch.stack([row, col]), torch.randn(nnz, generator=g, dtype=dtype),
                                (rows, cols)).coalesce().to_sparse_csr()
    ip = a.crow_indices()
    empty = int((ip[1:] == ip[:-1]).sum())
    work = scorch_ops.scorch_spmm_work(a._nnz(), empty, width, width)
    workers = scorch_ops.scorch_spmm_nthreads(work, rows, 24, a._nnz() * width, a._nnz(), True)
    assert (workers > 1) == (width > 1)
    b = torch.randn(cols, width, generator=g, dtype=dtype)
    got = scorch.matmul(STensor.from_torch(a), b)
    torch.testing.assert_close(got, torch.sparse.mm(a, b), atol=1e-3, rtol=1e-3)


def _runs_matrix(cols, runs, seed):
    """Rows of one nonzero separated by runs of empty rows of the given lengths, at both ends too."""
    g = torch.Generator().manual_seed(seed)
    crow, col = [0], []
    for r in runs:
        crow.extend([crow[-1]] * r)                     # r empty rows
        col.append(int(torch.randint(0, cols, (1,), generator=g)))
        crow.append(crow[-1] + 1)                        # one written row
    crow.extend([crow[-1]] * runs[0])                    # an empty run at the end
    vals = torch.randn(len(col), generator=g, dtype=torch.float64)
    return torch.tensor(crow), torch.tensor(col), vals, len(crow) - 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8, 16, 32])
def test_runs_of_empty_rows_of_every_size_are_zeroed(dtype, width):
    cols = 300
    runs = [1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 31, 32, 33, 40, 1, 40, 2, 100, 257, 1000, 17, 18, 19,
            20, 21, 22, 23, 24, 10, 11, 12, 63, 64, 65, 72, 73]
    crow, col, vals, rows = _runs_matrix(cols, runs, seed=width)
    a = torch.sparse_csr_tensor(crow, col, vals.to(dtype), size=(rows, cols))
    b = torch.randn(cols, width, generator=torch.Generator().manual_seed(11), dtype=dtype)
    # Leave a same-sized output of nonzeros behind for the allocator to hand back.
    full = torch.sparse_csr_tensor(torch.arange(rows + 1), torch.zeros(rows, dtype=torch.long),
                                   torch.ones(rows, dtype=dtype), size=(rows, cols))
    stale = scorch.matmul(STensor.from_torch(full), b + 2)
    assert bool((stale != 0).all())
    del stale
    got = scorch.matmul(STensor.from_torch(a), b)
    want = torch.sparse.mm(a, b)
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3)
    empty = (crow[1:] == crow[:-1])
    assert bool((got[empty] == 0).all())
