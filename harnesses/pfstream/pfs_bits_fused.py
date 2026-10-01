#!/usr/bin/env python3
"""Correctness for PFSTREAM on the FUSED Linear path: degenerate inputs first, then bit-identity.

A SEPARATE FILE FROM pfs_bits.py ON PURPOSE. pfs_bits.py produced the completed correctness
artefact for the drop-in cell (2120 cells, four builds agreeing, 2120 distinct digests). Extending
it would change the producer of a sealed result, so this is a second file and the first one is left
byte-stable.

WHY BIT-IDENTITY IS STILL THE RIGHT CHECK. A prefetch is a pure hint: it changes no arithmetic and
no reduction order, so every object here must produce byte-identical output rather than output
within a tolerance. allclose would pass a change that quietly altered the accumulation order and so
would not be testing the claim.

WHAT THIS PATH DOES THAT THE DROP-IN DOES NOT, and therefore what has to be covered here:

1. A STRUCTURALLY-EMPTY OUTPUT ROW IS NOT ZERO. spmm.h:6333-6337 short-circuits a row with no
   nonzeros to act(bias[o]) repeated across the row, skipping the register kernel entirely. On the
   drop-in path an empty row is zeros. So the empty-row case here is both a branch the drop-in has
   no analogue for AND a stronger positive control for the digest, since its output is nonzero by
   construction.

2. THE REGISTER BLOCK IS REACHED AT EVERY WIDTH 1..32, because this path has no exact-width branch
   and no half-vector branch. Widths 1, 2 and 3 therefore reach (nvec 1, masked tail), which the
   drop-in path cannot reach at any width, so they are covered here and nowhere else.

3. k >= 33 LEAVES narrow_k AND TAKES scorch_spmm_row_regtile. That is the timing cell's negative
   control, so its correctness has to be established too -- a control whose own output is wrong
   bounds nothing.

4. THE CLAMP'S BOUND IS A DIFFERENT QUANTITY. This path passes A1_pos[A0_size] (spmm.h:6157); the
   drop-in passes the redirect-aware A1_pos[SCORCH_SPMM_ROW_UNITS] (3558-3563). They coincide
   unless SCORCH_SPMM_SPLIT_ENABLED is 1, which needs SCORCH_TUNE_HOOKS or
   SCORCH_SPMM_SPLIT_MIN_IMBALANCE > 0, and that ships at 0L. The hazard pfs_bits.py was built
   around is unchanged in form: an all-empty matrix makes the bound 0, and the run-ahead
   `pA + PFD < pA_limit ? pA + PFD : pA_limit - 1` indexes A1_crd[-1] at exactly that value.

THE TOLERANCE QUESTION THE EPILOGUE RAISES, ANSWERED BY MEASUREMENT RATHER THAN BY ASSERTION. The
drop-in's tolerance is scaled to |A| @ |B| because the error is set by the accumulation TERMS and
not by the result -- a cancelling row has a tiny result and a large error bound. Folding in a bias
and an activation could in principle break that relationship. It does not, and the reason is that
both activations are NON-EXPANSIVE: |act(a) - act(b)| <= L|a - b| with L <= 1 for identity (L=1),
relu (L=1, and it truncates toward zero) and sigmoid (L=1/4). So a bound valid on the
pre-activation value is valid on the post-activation value, and the scale is |A| @ |B| + |bias|.
L is CHECKED NUMERICALLY below rather than taken from calculus, because the claim that matters is
about the implementation's activation and not about the mathematical function.
"""
import argparse, csv, hashlib, os, sys, zlib
ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--keys", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--build", required=True)
ap.add_argument("--ks", default="1,2,3,4,8,12,16,20,32,33")
ap.add_argument("--acts", default="0,1,2")
ap.add_argument("--limit", type=int, default=0, help="0 = every key")
A = ap.parse_args()
import numpy as np, torch, scorch_ops as so
if so.scorch_tune_hooks():
    sys.exit("REFUSE: hooks object; the arms here are builds")
if not hasattr(so, "spmm_csr_linear_fused_float"):
    sys.exit("REFUSE: this object has no spmm_csr_linear_fused_float. Reporting zero mismatches "
             "over zero cells is how a failure to measure reads as a pass.")
SYM = so.spmm_csr_linear_fused_float
KS = [int(x) for x in A.ks.split(",")]
ACTS = [int(x) for x in A.acts.split(",")]
if any(a not in (0, 1, 2) for a in ACTS):
    sys.exit("REFUSE: --acts must be a subset of 0,1,2; the native symbol raises otherwise.")
POOL = torch.get_num_threads()
try:
    import scipy.sparse as sp
except ImportError:
    sys.exit("REFUSE: scipy is required. A Python loop over nonzeros cannot run on the large "
             "matrices, and skipping the reference there leaves the check vacuous on exactly the "
             "matrices where the prefetch matters most.")


def _act_np(x, act):
    if act == 0: return x
    if act == 1: return np.maximum(x, 0.0)
    return 1.0 / (1.0 + np.exp(-x))


def _lipschitz(act, rng):
    """Largest observed |act(a)-act(b)| / |a-b| over random pairs spanning the useful range.

    Measured, not asserted: the tolerance argument above rests on this being <= 1, and the claim
    that matters is about the activation as implemented here, not about the textbook function.
    """
    a = rng.uniform(-40, 40, size=200000)
    b = a + rng.uniform(-1, 1, size=a.size) * 1e-3
    d = np.abs(b - a)
    keep = d > 0
    return float(np.max(np.abs(_act_np(b[keep], act) - _act_np(a[keep], act)) / d[keep]))


_rng = np.random.default_rng(20261001)
_L = {a: _lipschitz(a, _rng) for a in ACTS}
for a in sorted(_L):
    print("  act %d measured Lipschitz constant %.6f %s"
          % (a, _L[a], "OK (<= 1, so a pre-activation bound carries)" if _L[a] <= 1.0 + 1e-9
             else "REFUSE-WORTHY"), flush=True)
_bad_L = [a for a in _L if _L[a] > 1.0 + 1e-9]
if _bad_L:
    sys.exit("REFUSE: act %s measured a Lipschitz constant above 1, so the tolerance scaled to the "
             "pre-activation terms does not bound the post-activation error and every tolerance "
             "below is unjustified." % _bad_L)


def _ref(M, J, indptr, indices, vals, Bd, bias, act):
    """act(A @ B + bias[:, None]) in float64."""
    Acsr = sp.csr_matrix((np.asarray(vals, dtype=np.float64),
                          np.asarray(indices, dtype=np.int64),
                          np.asarray(indptr, dtype=np.int64)), shape=(M, J))
    return _act_np(Acsr @ Bd + np.asarray(bias, dtype=np.float64)[:, None], act)


def run_cell(M, J, indptr, indices, vals, k, act, seed, pos_bias=False):
    """pos_bias forces a strictly positive bias, and the degenerate cases need it.

    An empty row's output IS act(bias), so with relu and a negative bias the whole cell is zeros --
    and a digest over an all-zero result agrees across four builds for the wrong reason. One of 126
    degenerate cells hit exactly that (`empty`, k=33, act=1, both biases negative), which is a
    property of the seed rather than of the code, so leaving it to the seed means the epilogue
    branch is proven to RUN only by luck. Real matrices keep the signed bias, so relu's truncation
    branch is still exercised where the rows are not empty.
    """
    g = torch.Generator().manual_seed(seed)
    B = torch.rand((J, k), generator=g).to(torch.float32)
    bias = torch.rand((M,), generator=g).to(torch.float32)
    bias = bias + 0.5 if pos_bias else bias * 2.0 - 1.0
    ipt = torch.from_numpy(np.asarray(indptr, dtype=np.int32))
    ixt = torch.from_numpy(np.asarray(indices, dtype=np.int32))
    vt = torch.from_numpy(np.asarray(vals, dtype=np.float32))
    r = SYM(result_shape=[M, k], A_shape=[M, J], A_mode_indices=[[], [ipt, ixt]],
            A_values=vt, B_shape=[J, k], B_mode_indices=[[], []],
            B_values=B.reshape(-1).contiguous(), bias_values=bias, act=act,
            nthreads_override=POOL, atparallel=False)
    # The native symbol returns scorch_ops.Tensor; the torch tensor is .storage.value. Asserted
    # rather than probed, because a digest over the wrong object is a digest that always agrees.
    if not (hasattr(r, "storage") and hasattr(r.storage, "value")):
        sys.exit("REFUSE: the native return has no storage.value; a digest over anything else "
                 "would agree across builds for the wrong reason.")
    v = r.storage.value.detach().cpu().numpy()
    if v.size != M * k:
        sys.exit("REFUSE: result has %d entries, expected %d -- the accessor is wrong and a digest "
                 "over it proves nothing." % (v.size, M * k))
    got = v.reshape(M, k).astype(np.float64)
    Bd = B.numpy().astype(np.float64)
    ref = _ref(M, J, indptr, indices, vals, Bd, bias.numpy(), act)
    # Scaled to the TERMS, not the result: a cancelling row has a tiny result and a large error
    # bound, so a result-relative tolerance fails on exactly those rows. Plus |bias|, which enters
    # the sum before the activation. Valid post-activation because L <= 1, measured above.
    scale = (_ref(M, J, indptr, np.asarray(indices), np.abs(np.asarray(vals, dtype=np.float64)),
                  np.abs(Bd), np.abs(bias.numpy()), 0))
    tol = 1e-3 + 1e-3 * np.abs(scale)
    off = np.abs(got - ref) - tol
    if np.any(off > 0):
        # The element that actually FAILED -- the largest excess over its OWN tolerance -- not the
        # largest absolute difference. Those are different elements whenever the tolerance varies.
        bad = int(np.argmax(off))
        sys.exit("REFUSE: build %s k=%d act=%d element %d: got %r reference %r tolerance %r "
                 "(excess %r). A prefetch changes no arithmetic."
                 % (A.build, k, act, bad, got.reshape(-1)[bad], ref.reshape(-1)[bad],
                    tol.reshape(-1)[bad], off.reshape(-1)[bad]))
    return hashlib.sha256(v.tobytes()).hexdigest()[:32], v


# PREMISE cases make the clamp's bound take the value that would index backwards; CLAMP cases make
# the run-ahead actually clip. The fused path adds a third kind: EPILOGUE, where the row never
# enters the register kernel and the output comes from act(bias) alone.
DEGEN = [
    ("empty",       2, 4, [0, 0, 0],       [],           []),          # PREMISE: bound 0
    ("one_nz",      2, 4, [0, 1, 1],       [2],          [1.5]),       # PREMISE: bound 1
    ("two_nz",      1, 4, [0, 2],          [0, 3],       [1.0, 2.0]),  # CLAMP: bound 2, pf_=1
    ("row_of_2",    3, 4, [0, 2, 4, 6],    [0, 1, 1, 2, 2, 3],
                                           [1., 2., 3., 4., 5., 6.]),  # CLAMP: 2-nonzero rows
    ("row_of_3",    2, 4, [0, 3, 6],       [0, 1, 2, 1, 2, 3],
                                           [1., 2., 3., 4., 5., 6.]),  # CLAMP: 3-nonzero rows
    ("empty_rows",  4, 4, [0, 0, 2, 2, 2], [1, 3],       [7., 8.]),    # EPILOGUE: interior empties
    ("all_empty",   3, 4, [0, 0, 0, 0],    [],           []),          # EPILOGUE: every row act(b)
]

out = open(A.csv, "w", newline="")
w = csv.DictWriter(out, fieldnames=["build", "case", "key", "k", "act", "rows", "nnz", "sha"])
w.writeheader()
keys = [l.strip() for l in open(A.keys) if l.strip()]
if A.limit:
    keys = keys[:A.limit]
try:
    from csrresolve import resolve_all, describe
except ImportError:
    sys.exit("REFUSE: csrresolve.py is not importable next to this script; falling back to an "
             "exact-path lookup is what silently dropped 50 of 298 keys on MKT.")
_paths, _how, _bad = resolve_all(A.cache, keys)
for _l in describe(_how, _bad, "corpus keys"):
    print(_l, flush=True)
if _bad:
    sys.exit("REFUSE: %d of %d keys unresolved under %s (first %s)."
             % (len(_bad), len(keys), A.cache, _bad[0][0]))

n_degen = n_real = 0
nz_degen = nz_real = 0
print("DEGENERATE CASES FIRST -- the script stops here if any fails, because every digest below "
      "would otherwise be taken on a process that already read out of bounds.", flush=True)
for name, M, J, ip, ix, vl in DEGEN:
    for k in KS:
        for act in ACTS:
            sha, v = run_cell(M, J, ip, ix, vl, k, act,
                              zlib.crc32(("%s|%d|%d" % (name, k, act)).encode()) % (2**31),
                              pos_bias=True)
            w.writerow(dict(build=A.build, case="degen", key=name, k=k, act=act, rows=M,
                            nnz=len(vl), sha=sha))
            n_degen += 1
            nz_degen += int(np.any(v != 0))
print("  %d degenerate cell(s) passed, %d with a nonzero result" % (n_degen, nz_degen), flush=True)
# ASSERTED, not reported. With a strictly positive bias every degenerate cell is nonzero by
# construction, so any shortfall means a cell went vacuous and its digest proves nothing -- and a
# count that is only printed is a count nobody checks.
if nz_degen != n_degen:
    sys.exit("REFUSE: %d of %d degenerate cells produced an all-zero result. Their digests agree "
             "across builds for the wrong reason, and with pos_bias every one of them should be "
             "nonzero." % (n_degen - nz_degen, n_degen))
for key in keys:
    z = np.load(_paths[key])
    M, J = int(z["shape"][0]), int(z["shape"][1])
    ip, ix, vl = z["indptr"], z["indices"], z["data"]
    nnz = int(ip[-1])
    for k in KS:
        for act in ACTS:
            sha, v = run_cell(M, J, ip, ix, vl, k, act,
                              zlib.crc32(("%s|%d|%d" % (key, k, act)).encode()) % (2**31))
            w.writerow(dict(build=A.build, case="real", key=key, k=k, act=act, rows=M, nnz=nnz,
                            sha=sha))
            n_real += 1
            nz_real += int(np.any(v != 0))
    out.flush()
out.close()
# The sentinel carries the counts, so a comparator can cross-check the cell set it was HANDED
# against the cell set this producer DECLARED. Checking that four builds cover the same cells
# rather than the declared ones is what called a run truncated at matrix 27 of 418 CORRECT.
with open(A.csv + ".done", "w") as _h:
    _h.write("%d %d %d\n" % (n_degen + n_real, n_real, n_degen))
print("wrote %s (%d degenerate + %d real = %d cells)"
      % (A.csv, n_degen, n_real, n_degen + n_real), flush=True)
# Beside the digest count, the count that shows the digests are not over zeros. relu sends about
# half of a symmetric pre-activation to exactly 0, so this number is load-bearing here in a way it
# is not on the drop-in path.
print("  NONZERO RESULTS: %d of %d degenerate, %d of %d real -- a digest over an all-zero result "
      "agrees across builds for the wrong reason" % (nz_degen, n_degen, nz_real, n_real), flush=True)
