"""The validation memo sweeps out dead arrays in amortized O(1) per insert, and bounded.

Every insert used to sweep the whole memo, so a program that kept many structures alive and met
each for the first time in turn paid O(live entries) per call (csrc/native_abi.h,
abi_bound_map). Now a sweep runs once the inserts since the last one reach what it left (at
least 32), and for the int32 copies of int64 index arrays also once the bytes copied since reach
the live copies' bytes (at least 1 MiB). These tests hold the bounds that replaced sweeping on
every insert: dead entries and dead copies never pile up.
"""

import gc

import torch

import scorch
import scorch_ops

SWEEP_FLOOR = 32
SWEEP_BYTES = 1 << 20


def _int64_csr(rows, cols, nnz_per_row, seed):
    g = torch.Generator().manual_seed(seed)
    crow = torch.arange(rows + 1, dtype=torch.int64) * nnz_per_row
    # Strictly increasing within each row (torch's CSR invariant): cumulative gaps of
    # 1..cols // nnz_per_row, so the last coordinate stays below cols.
    gaps = torch.randint(1, cols // nnz_per_row + 1, (rows, nnz_per_row), generator=g)
    col = (gaps.cumsum(dim=1) - 1).reshape(-1)
    values = torch.rand(rows * nnz_per_row, generator=g)
    return torch.sparse_csr_tensor(crow, col, values, size=(rows, cols))


def _first_call(a, b):
    torch.testing.assert_close(
        scorch.matmul(a, b), a.to_dense() @ b, rtol=1e-5, atol=1e-5
    )


def test_dead_small_copies_are_swept_by_count():
    scorch_ops.abi_memo_clear()
    b = torch.rand(30, 4)
    peak = 0
    for seed in range(500):
        _first_call(_int64_csr(40, 30, 3, seed), b)  # dies after its call
        gc.collect()
        peak = max(peak, scorch_ops.abi_narrow_held()[0])
    # Two int32 copies per structure; a sweep once the inserts since the last reach 32.
    assert peak <= SWEEP_FLOOR + 2, f"{peak} copies held for at most one live structure"


def test_dead_large_copies_are_swept_by_bytes():
    """A copy past the byte floor triggers a sweep on its own insert, so a program replacing
    one large structure per call holds at most one dead copy, as when every insert swept.
    """
    scorch_ops.abi_memo_clear()
    b = torch.rand(300, 2)
    one_copy = 3000 * 100 * 4  # the column array's int32 copy: 1.2 MB
    assert one_copy > SWEEP_BYTES
    peak = 0
    for seed in range(12):
        _first_call(_int64_csr(3000, 300, 100, seed), b)
        gc.collect()
        peak = max(peak, scorch_ops.abi_narrow_held()[1])
    assert peak <= 2 * one_copy + 2 * (3000 + 1) * 4 * 2, f"{peak} bytes of copies held"


def test_live_structures_stay_memoized_and_dead_ones_do_not_outnumber_them():
    scorch_ops.abi_memo_clear()
    b = torch.rand(30, 4)
    kept = [_int64_csr(40, 30, 3, seed) for seed in range(300)]
    for a in kept:
        _first_call(a, b)
    held = scorch_ops.abi_narrow_held()[0]
    assert held >= 2 * len(kept), "a live structure's copies were dropped"
    for a in kept:  # every copy is found again: nothing is added
        _first_call(a, b)
    assert scorch_ops.abi_narrow_held()[0] == held
    del kept
    gc.collect()
    for seed in range(1000, 1000 + 2 * held):
        _first_call(_int64_csr(40, 30, 3, seed), b)
        gc.collect()
    # The 600 dead copies are gone once as many inserts have followed the sweep that kept them.
    assert scorch_ops.abi_narrow_held()[0] <= 2 * SWEEP_FLOOR + 2
