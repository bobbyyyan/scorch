"""The row ceiling on torch's full-width team: one worker per row, not one per 16 rows.

A drop-in SpMM call with more than one worker runs, under libgomp, on torch's intra-op team via
at::parallel_for, which is always the full width: every pool thread wakes whatever the count, and
the count only decides how many of them take rows. The 16-rows-per-worker ceiling exists to bound
how wide a team a call opens, so on that path it bounds nothing it was there to bound, and it held
a 71-row product with 212536 nonzeros (kl02) to 4 workers of 24. ``scorch_spmm_nthreads`` takes
``full_team`` from the call site; the composition adoption then allows ``min(pool, rows)``.

What these pin down, through the exported resolver (never a re-derivation of the policy):

* ``full_team=False`` -- the default, and every caller that is not on the full team -- resolves
  exactly as before;
* on the full team a few-row product above the grain gets one worker per row up to the pool;
* the change cannot reach a product with at least 16 rows per pool thread (every GCN and
  autoencoder layer in the workload harness), nor one below the grain, which stays inline;
* the count never exceeds the rows, the pool, or the processors.
"""
import pytest
import torch

scorch_ops = pytest.importorskip("scorch_ops")

ROWS_PER_THREAD = 16  # SCORCH_ROWS_PER_THREAD, the ceiling the full team lifts


def _n(rows, nnz, k, pool, full_team):
    return scorch_ops.scorch_spmm_nthreads(nnz * max(k, 16), rows, pool, nnz * k, nnz, full_team)


def _procs():
    # The resolver caps every count at scorch_thread_ceiling(): the processors the process may run on,
    # or, once a thread setting is in force, the smaller of those and torch's count. Each test below
    # first sets torch's count to the pool it simulates (torch_threads), so this is at least that pool,
    # capped at the processors.
    return _n(1 << 20, 1 << 26, 64, 1 << 10, False)


# kl02 (71 rows, 212536 nonzeros) and a 64-row pruned ResNet-50 layer, the two few-row shapes the
# ceiling was measured on; and a 300-row product, between the old ceiling and the pool's 16x.
FEW_ROW = [(71, 212536, 2), (71, 212536, 32), (64, 18432, 8), (300, 400000, 4)]


@pytest.mark.parametrize("rows,nnz,k", FEW_ROW)
def test_off_the_full_team_nothing_changes(rows, nnz, k, torch_threads):
    pool = 24
    torch_threads(pool)
    assert _n(rows, nnz, k, pool, False) == scorch_ops.scorch_spmm_nthreads(
        nnz * max(k, 16), rows, pool, nnz * k, nnz)


@pytest.mark.parametrize("rows,nnz,k", FEW_ROW)
def test_on_the_full_team_a_few_row_product_gets_a_worker_per_row(rows, nnz, k, torch_threads):
    pool = 24
    torch_threads(pool)
    assert _n(rows, nnz, k, pool, True) == min(pool, rows, _procs())


def test_the_ceiling_binds_where_it_was_measured(torch_threads):
    # kl02 at k=2: one grain of real arithmetic, so the work-based raise cannot lift it, and the
    # 16-row ceiling holds it to 71 // 16 = 4. At k=32 the raise already reaches the pool, which
    # is why this is pinned at k=2 and not asserted for every shape above.
    pool = 24
    torch_threads(pool)
    if min(pool, _procs()) <= 71 // ROWS_PER_THREAD:
        pytest.skip("this host has too few processors for the ceiling to bind")
    assert _n(71, 212536, 2, pool, False) == 71 // ROWS_PER_THREAD
    assert _n(71, 212536, 2, pool, True) == min(pool, _procs())


@pytest.mark.parametrize("rows,nnz,k", [(2708, 13264, 16), (19717, 108365, 3), (1024, 80282, 256)])
@pytest.mark.parametrize("pool", [6, 24, 32])
def test_cannot_reach_products_with_sixteen_rows_per_pool_thread(rows, nnz, k, pool, torch_threads):
    torch_threads(pool)
    # Cora, PubMed, and an autoencoder layer at 0.9 sparsity: rows >= 16 * pool, so the old
    # ceiling already allowed the pool and lifting it changes nothing.
    assert rows >= ROWS_PER_THREAD * pool
    assert _n(rows, nnz, k, pool, True) == _n(rows, nnz, k, pool, False)


def test_a_product_below_the_grain_stays_inline(torch_threads):
    torch_threads(24)  # the pool below: with torch at 1, "inline" would hold for every product
    # 64 rows, 1024 nonzeros, k=1: well under one grain of work, so the adoption does not fire
    # and the call runs on the launching thread on either path.
    assert _n(64, 1024, 1, 24, False) == 1
    assert _n(64, 1024, 1, 24, True) == 1


@pytest.mark.parametrize("rows,nnz,k", FEW_ROW + [(3, 900000, 4), (1, 50000, 16)])
@pytest.mark.parametrize("pool", [1, 2, 6, 24, 64])
def test_never_more_workers_than_rows_pool_or_processors(rows, nnz, k, pool, torch_threads):
    torch_threads(pool)
    n = _n(rows, nnz, k, pool, True)
    assert 1 <= n <= max(1, min(rows, pool, _procs()))


def test_the_widened_count_computes_the_right_product():
    # A 40-row product with long rows, through the drop-in path: the answer must not depend on
    # how many workers took rows.
    import scorch
    from scorch import STensor

    g = torch.Generator().manual_seed(3)
    dense = torch.rand(40, 3000, generator=g) * (torch.rand(40, 3000, generator=g) < 0.5)
    A = STensor.from_csr(dense.to_sparse_csr(), "A")
    B = torch.rand(3000, 8, generator=g)
    got = scorch.matmul(A, B)
    got = got if isinstance(got, torch.Tensor) else got.to_torch()
    assert torch.allclose(got, dense @ B, atol=1e-3, rtol=1e-3)
