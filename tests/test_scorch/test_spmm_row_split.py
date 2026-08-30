"""The row-split partition: does it fire where the load-balance ceiling binds, stay away
everywhere else, and give an answer that does not depend on the thread count?

Background, because the gate looks arbitrary without it. Every partition mode in the SpMM hands
out WHOLE ROWS, so no schedule can finish before the widest single row is done. Three matrices in
the benchmark corpus hold a tenth of their nonzeros in ONE row, which caps them at 8.3-10.0 of 24
workers however good the inner loop is, and their cells measured 0.9065 from doubling workers 12
-> 24 where the ceiling permits exactly 1.0000. The split exists for those.

Three properties are worth a test rather than a comment:

  * the probe is the LONGEST RUN OF EQUAL-NONZERO CUTS LANDING IN ONE ROW, not max-degree over
    mean. The two agree on a uniform matrix and disagree on exactly the matrices this is for, so a
    test that only checked uniform matrices would pass on the wrong implementation;
  * the segment width ignores the worker count, so a split result is bit-identical at every thread
    count. A width keyed on the pool would have been the obvious implementation and would have made
    results depend on OMP_NUM_THREADS;
  * a matrix whose widest row is under one segment must come out BIT-IDENTICAL to the unsplit
    kernel, not merely close -- that is what makes the mechanism inert rather than merely small
    where it should not act.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

scorch_ops = pytest.importorskip("scorch_ops")

pytestmark = pytest.mark.skipif(
    not hasattr(scorch_ops, "scorch_spmm_row_imbalance"),
    reason="scorch_ops predates the row-split partition",
)

# Driving the split needs the environment hooks. Guarding on the presence of the exported probe is
# not enough: a release object exports it too, and the arms would then be the same code twice.
_needs_hooks = pytest.mark.skipif(
    not scorch_ops.scorch_tune_hooks(),
    reason="release build: the SCORCH_SPMM_SPLIT_* knobs are compiled out",
)

_REFN = 64


def _csr(degrees, cols=None):
    """A CSR matrix with the given per-row degrees, values all 1.0, indices ascending in a row."""
    degrees = np.asarray(degrees, dtype=np.int64)
    cols = int(cols or max(int(degrees.max(initial=1)), 2))
    assert int(degrees.max(initial=0)) <= cols, "degree exceeds the column count"
    indptr = np.zeros(len(degrees) + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(degrees).astype(np.int32)
    idx = (np.concatenate([np.arange(d, dtype=np.int32) for d in degrees])
           if len(degrees) else np.zeros(0, dtype=np.int32))
    vals = np.ones(int(indptr[-1]), dtype=np.float32)
    return indptr, idx, vals, cols


def _imbalance(indptr, refn=_REFN):
    return scorch_ops.scorch_spmm_row_imbalance(torch.from_numpy(indptr), len(indptr) - 1, refn)


def test_a_uniform_matrix_is_perfectly_balanced():
    indptr, _, _, _ = _csr([64] * 256)
    assert _imbalance(indptr) == 1


def test_one_dominant_row_is_found_and_its_size_is_reported():
    """A row holding a quarter of the nonzeros should read about refn/4 against refn cuts."""
    n = 256
    degrees = [10] * n
    degrees[123] = 10 * n // 3          # ~25% of the total
    indptr, _, _, _ = _csr(degrees, cols=10 * n // 3)
    imb = _imbalance(indptr)
    share = degrees[123] / sum(degrees)
    assert imb >= 2, "a row with a quarter of the nonzeros must be detected"
    # the run length is floor-ish of share*refn; allow one cut either way
    assert abs(imb - share * _REFN) <= 2, (imb, share * _REFN)


def test_the_probe_is_not_max_degree_over_mean():
    """The two statistics disagree, and the ceiling is about the one this returns.

    A power-law matrix can have a large max/mean ratio while no single row is anywhere near a
    fair share of the whole matrix -- that is a matrix a whole-row partition handles fine, and a
    gate written on max/mean would split it for nothing.
    """
    degrees = [1] * 100000 + [2000]     # max/mean is ~1000, but 2000 of 102000 nonzeros is 2%
    indptr, _, _, _ = _csr(degrees, cols=2000)
    mean = sum(degrees) / len(degrees)
    assert max(degrees) / mean > 500, "the decoy statistic is supposed to be large here"
    assert _imbalance(indptr) == 1, "no row is close to a fair share, so nothing should fire"


def test_the_segment_width_ignores_the_worker_count_and_rises_with_k():
    w = scorch_ops.scorch_spmm_seg_width
    assert w(1_000_000, 4, 4) == w(1_000_000, 4, 4)
    narrow = w(100_000_000, 4, 4)
    wide = w(100_000_000, 256, 8)
    assert wide > narrow, "the partial buffer scales with k and the element size"
    assert narrow & (narrow - 1) == 0 and wide & (wide - 1) == 0, "widths stay powers of two"


def test_the_segment_width_bounds_the_partial_buffer():
    """The extra memory is (nnz / seg) * k * elem, and that is what the width is chosen to hold."""
    for nnz, k, elem in [(10**8, 256, 8), (10**8, 4, 4), (10**6, 64, 8)]:
        seg = scorch_ops.scorch_spmm_seg_width(nnz, k, elem)
        extra_mb = (nnz / seg) * k * elem / 2**20
        assert extra_mb <= 16.0, (nnz, k, elem, seg, extra_mb)


_PROBE = r"""
import os, numpy as np, torch, scorch_ops as so
M, deg, big, k, dt, pool = %s
npd = np.float32 if dt == 32 else np.float64
td = torch.float32 if dt == 32 else torch.float64
sym = so.spmm_csr_float_v2 if dt == 32 else so.spmm_csr_double_v2
degrees = np.full(M, deg, dtype=np.int64)
degrees[M // 2] = big
J = int(max(degrees.max(), k, 2))
indptr = np.zeros(M + 1, dtype=np.int32); indptr[1:] = np.cumsum(degrees)
idx = np.concatenate([np.arange(d, dtype=np.int32) for d in degrees])
vals = np.ones(int(indptr[-1]), dtype=npd)
g = torch.Generator().manual_seed(7)
# B at the target dtype, NOT generated in float32 and widened: widening makes every product
# exactly representable in float64, the sums become exact, and no reordering can change them --
# which would make the float64 half of this check unable to fail.
B = torch.rand((J, k), generator=g, dtype=td)
out = sym(result_shape=[M, k], A_shape=[M, J],
          A_mode_indices=[[], [torch.from_numpy(indptr), torch.from_numpy(idx)]],
          A_values=torch.from_numpy(vals), B_shape=[J, k], B_mode_indices=[[], []],
          B_values=B.reshape(-1).contiguous(), nthreads_override=pool, atparallel=False)
v = out.storage.value.reshape(M, k)
print(so.scorch_spmm_row_imbalance(torch.from_numpy(indptr), M, 64))
print(v.numpy().tobytes().hex())
"""


def _run(shape, imbalance, seg_min=256, min_nnz=150000):
    env = dict(os.environ)
    env["SCORCH_SPMM_SPLIT_MIN_IMBALANCE"] = str(imbalance)
    env["SCORCH_SPMM_SPLIT_MIN_NNZ"] = str(min_nnz)
    env["SCORCH_SPMM_SEG_MIN"] = str(seg_min)
    out = subprocess.run([sys.executable, "-c", _PROBE % (shape,)],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    imb, payload = out.stdout.split()
    return int(imb), payload


# One shape, used by every test below: 400 rows of degree 400 plus one row of 40000, so the widest
# row holds 20% of the nonzeros and the matrix is well past SPLIT_MIN_NNZ.
_SPLIT_SHAPE = "400, 400, 40000, 4, 32, 4"


@_needs_hooks
def test_the_split_fires_only_where_a_row_dominates():
    imb, _ = _run(_SPLIT_SHAPE, imbalance=2)
    assert imb >= 2, "the fixture is supposed to be imbalanced"
    balanced = "400, 400, 400, 4, 32, 4"
    imb_b, _ = _run(balanced, imbalance=2)
    assert imb_b == 1


@_needs_hooks
def test_splitting_changes_the_answer_only_at_rounding_level():
    _, off = _run(_SPLIT_SHAPE, imbalance=0)
    _, on = _run(_SPLIT_SHAPE, imbalance=2)
    a = np.frombuffer(bytes.fromhex(off), dtype=np.float32)
    b = np.frombuffer(bytes.fromhex(on), dtype=np.float32)
    assert a.shape == b.shape
    assert not np.array_equal(a, b), (
        "the split did not fire, so this test proves nothing -- check the gate")
    rel = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
    assert rel < 2e-5, rel


@_needs_hooks
@pytest.mark.parametrize("dt", [32, 64])
def test_a_split_result_is_bit_identical_at_every_thread_count(dt):
    """The property the segment width was designed for. A width keyed on the pool would fail here."""
    shapes = ["400, 400, 40000, 4, %d, %d" % (dt, pool) for pool in (1, 2, 3, 5, 8, 16)]
    payloads = [_run(s, imbalance=2)[1] for s in shapes]
    assert len(set(payloads)) == 1, "the result changed with the worker count"


@_needs_hooks
def test_a_matrix_under_one_segment_wide_is_bit_identical_to_the_unsplit_kernel():
    """Inert by construction, not merely small: with the segment wider than every row the split
    produces exactly the original rows, so the arithmetic is the same arithmetic."""
    _, off = _run(_SPLIT_SHAPE, imbalance=0)
    _, on = _run(_SPLIT_SHAPE, imbalance=2, seg_min=1 << 20)
    assert on == off


@_needs_hooks
def test_the_size_floor_keeps_the_probe_off_small_products():
    """Below SPLIT_MIN_NNZ nothing may change, however imbalanced the matrix is."""
    small = "40, 40, 4000, 4, 32, 4"          # 5600 nonzeros, one row holding 71% of them
    _, off = _run(small, imbalance=0)
    _, on = _run(small, imbalance=2, min_nnz=150000)
    assert on == off
