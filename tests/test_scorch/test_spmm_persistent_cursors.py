"""Partition mode 3's cursors persist across calls, one array per launching thread, and each
worker writes its own range into them (scorch_spmm_gen_cursor in spmm.h). What can go wrong
with that is specific, so the tests aim at it:

  * a cursor left by an EARLIER call must never be taken as this call's. The dangerous case is a
    call with more workers followed by calls with fewer and then more again: entries the narrow
    calls never touched still hold an old range, and only the call number on the line keeps a
    thief out of it. So the worker count changes on every call, in both directions, and the
    pool is made to grow mid-sequence;
  * a row nobody claims would not show up as a wrong number if the output buffer happened to
    hold last call's answer for it -- the allocator hands the same block back. So B is fresh on
    every call, and the test first checks that consecutive answers really do differ, which is
    what makes a stale row detectable at all;
  * every row is computed by exactly one worker with arithmetic that does not depend on which
    one, so the answer is BIT-identical across worker counts, not merely close.

Calls that resolve to one worker never touch the cursors, so the test also counts the calls that
resolved to mode 3 with more than one worker, using the exported policy -- a sequence that
happened to run everything on one worker would otherwise pass without testing anything.
"""
import hashlib
import os
import threading

import numpy as np
import pytest
import torch

scorch_ops = pytest.importorskip("scorch_ops")

pytestmark = [
    pytest.mark.skipif(
        not hasattr(scorch_ops, "scorch_spmm_partition_mode"),
        reason="scorch_ops predates the exported partition policy",
    ),
    pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="one CPU: every call runs on one worker"),
]

# Up, down, up past the first maximum (so the pool grows after it has been used), and through
# counts that do and do not divide the row count.
_WORKERS = [8, 2, 64, 3, 1, 17, 5, 96, 4, 32, 2, 96]


def _matrix(kind, seed):
    g = np.random.default_rng(seed)
    if kind == "skewed":  # a few rows carry most of the nonzeros
        rows, cols = 3000, 700
        deg = g.integers(0, 6, rows)
        deg[g.choice(rows, 12, replace=False)] = g.integers(300, 700, 12)
    elif kind == "uniform":
        rows, cols = 4096, 512
        deg = np.full(rows, 9)
    elif kind == "fewer_rows_than_workers":
        rows, cols = 5, 64
        deg = np.array([3, 0, 64, 1, 7])
    elif kind == "empty_rows":
        rows, cols = 2000, 300
        deg = np.where(g.random(rows) < 0.6, 0, g.integers(1, 40, rows))
    else:
        raise ValueError(kind)
    deg = np.minimum(deg, cols)
    indptr = np.zeros(rows + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(deg)
    idx = np.concatenate([np.sort(g.choice(cols, int(d), replace=False)) for d in deg]).astype(np.int32)
    vals = g.standard_normal(int(indptr[-1]))
    return rows, cols, torch.from_numpy(indptr), torch.from_numpy(idx), vals


def _call(dt, rows, cols, ipt, ixt, vt, B, k, workers, atparallel):
    sym = scorch_ops.spmm_csr_float_v2 if dt == torch.float32 else scorch_ops.spmm_csr_double_v2
    out = sym([rows, k], [rows, cols], [[], [ipt, ixt]], vt, [cols, k], [[], []], B.reshape(-1),
              256, workers, atparallel)
    return out.storage.value.view(rows, k).clone()


def _mode3_multiworker(rows, nnz, k, esz, workers):
    work = nnz * max(k, 16)
    nt = scorch_ops.scorch_spmm_nthreads(work, rows, workers, nnz * k, nnz)
    return nt > 1 and scorch_ops.scorch_spmm_partition_mode(rows, nnz, k, k, esz, nt, workers) == 3


@pytest.mark.parametrize("dt", [torch.float32, torch.float64])
@pytest.mark.parametrize("atparallel", [True, False])
def test_changing_worker_counts_give_the_same_bits_and_the_right_answer(dt, atparallel):
    torch.manual_seed(0)
    exercised = calls = 0
    for kind in ("skewed", "uniform", "fewer_rows_than_workers", "empty_rows"):
        rows, cols, ipt, ixt, vals = _matrix(kind, seed=len(kind))
        vt = torch.from_numpy(vals).to(dt)
        A = torch.sparse_csr_tensor(ipt.long(), ixt.long(), vt, size=(rows, cols))
        nnz = int(ipt[-1])
        for k in (1, 2, 3, 4, 8, 32):
            B = torch.randn(cols, k, dtype=dt)
            first = None
            for workers in _WORKERS:
                got = _call(dt, rows, cols, ipt, ixt, vt, B, k, workers, atparallel)
                h = hashlib.sha256(got.numpy().tobytes()).hexdigest()
                if first is None:
                    first = h
                    ref = torch.sparse.mm(A, B)
                    denom = max(float(ref.norm()), 1e-30)
                    assert float((got - ref).norm()) / denom < 1e-5, (kind, k)
                assert h == first, ("answer depends on the worker count", kind, k, workers)
                exercised += _mode3_multiworker(rows, nnz, k, dt.itemsize, workers)
                calls += 1
    # Non-vacuity: a good share of these calls must actually have gone through the cursors
    # (138 of 288 on a 18-core M5, more on wider hosts).
    assert exercised >= calls // 4, (exercised, calls)


@pytest.mark.parametrize("atparallel", [True, False])
def test_a_fresh_B_every_call_is_answered_fresh_every_call(atparallel):
    rows, cols, ipt, ixt, vals = _matrix("skewed", seed=3)
    vt = torch.from_numpy(vals).float()
    A = torch.sparse_csr_tensor(ipt.long(), ixt.long(), vt, size=(rows, cols))
    nonempty = (ipt[1:] - ipt[:-1]) > 0
    torch.manual_seed(1)
    prev = None
    for workers in _WORKERS * 2:
        B = torch.randn(cols, 4)
        got = _call(torch.float32, rows, cols, ipt, ixt, vt, B, 4, workers, atparallel)
        ref = torch.sparse.mm(A, B)
        assert float((got - ref).norm() / ref.norm()) < 1e-5, workers
        if prev is not None:
            # What makes a skipped row visible: no non-empty row may keep its old value.
            assert bool((got[nonempty] != prev[nonempty]).any(dim=1).all())
        prev = got


def test_launchers_on_several_threads_each_get_their_own_cursors():
    mats = [_matrix(kind, seed=s) for s, kind in enumerate(("skewed", "uniform", "empty_rows", "skewed"))]
    errors = []

    def run(i):
        rows, cols, ipt, ixt, vals = mats[i]
        vt = torch.from_numpy(vals).float()
        A = torch.sparse_csr_tensor(ipt.long(), ixt.long(), vt, size=(rows, cols))
        gen = torch.Generator().manual_seed(100 + i)
        for workers in _WORKERS:
            B = torch.randn(cols, 8, generator=gen)
            got = _call(torch.float32, rows, cols, ipt, ixt, vt, B, 8, workers, True)
            ref = torch.sparse.mm(A, B)
            if float((got - ref).norm() / ref.norm()) >= 1e-5:
                errors.append((i, workers))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(mats))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
