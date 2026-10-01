#!/usr/bin/env python3
"""Correctness for the PFSTREAM clamp: degenerate inputs first, then bit-identity on real ones.

WHY BIT-IDENTITY IS THE RIGHT CHECK. A prefetch is a pure hint -- it changes no arithmetic and no
reduction order -- so every object here must produce byte-identical output, not merely output
within a tolerance. allclose would pass a change that quietly altered the accumulation order and
would therefore not be testing the claim. Bit-identity is also free.

WHY THE DEGENERATE CASES COME FIRST AND THE SCRIPT STOPS IF THEY FAIL. Under PFSTREAM the
run-ahead is `pA + PFD < pA_limit ? pA + PFD : pA_limit - 1`, so pA_limit == 0 indexes A1_crd[-1]:
an out-of-bounds read backwards, and the ONE value of the bound that does it. pA_limit is
SCORCH_SPMM_NNZ_TOTAL = `ROW_UNITS > 0 ? A1_pos[ROW_UNITS] : 0`, which is 0 for a matrix with no
nonzeros, and in the shipped build that 0 reaches only work measures and the chunk rule, where the
sentinel is the point. PFSTREAM gives the same expression a second meaning as an array index.

What is supposed to keep it unreachable is that the row loop cannot execute when the limit is 0.
That is a premise about the LOOP, not about the bound, and `pA_end <= pA_limit` does not imply it
-- pA_end >= 2 inside the body gives nnz >= 2 only if pA_limit == nnz is already assumed, which is
the thing being assumed. So it is checked by running it: an empty matrix, a one-nonzero matrix, and
rows of exactly 2 and 3 nonzeros, which are also the rows the row-clipped form serves worst (no
prefetch at all, and exactly one, respectively).
"""
import argparse, csv, hashlib, os, sys, zlib
ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--keys", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--build", required=True)
ap.add_argument("--ks", default="2,4,8,16,32")
ap.add_argument("--limit", type=int, default=0, help="0 = every key")
A = ap.parse_args()
import numpy as np, torch, scorch_ops as so
if so.scorch_tune_hooks():
    sys.exit("REFUSE: hooks object; the arms here are builds")
KS = [int(x) for x in A.ks.split(",")]
SYM = so.spmm_csr_float_v2
POOL = torch.get_num_threads()
# The dense reference. scipy is REQUIRED rather than optional: a Python loop over nonzeros cannot
# run on the large matrices, and silently skipping the reference there would leave the identity
# check vacuous on exactly the matrices where the prefetch matters most.
try:
    import scipy.sparse as _sp
except ImportError:
    sys.exit("REFUSE: scipy is not importable. Without it the dense reference cannot run on the "
             "large matrices, and a bit-identity check with no reference passes trivially if "
             "every object returns zeros.")


def _REF(M, J, indptr, indices, vals, Bd):
    Acsr = _sp.csr_matrix((np.asarray(vals, dtype=np.float64),
                           np.asarray(indices, dtype=np.int64),
                           np.asarray(indptr, dtype=np.int64)), shape=(M, J))
    return np.asarray(Acsr @ Bd)


def run(M, J, indptr, indices, vals, k, seed):
    g = torch.Generator().manual_seed(seed)
    B = torch.rand((J, k), generator=g)
    ipt = torch.from_numpy(np.asarray(indptr, dtype=np.int32))
    ixt = torch.from_numpy(np.asarray(indices, dtype=np.int32))
    vt = torch.from_numpy(np.asarray(vals, dtype=np.float32))
    r = SYM(nthreads_override=POOL, atparallel=False,
            result_shape=[M, k], A_shape=[M, J], A_mode_indices=[[], [ipt, ixt]],
            A_values=vt, B_shape=[J, k], B_mode_indices=[[], []],
            B_values=B.reshape(-1).contiguous())
    # The native symbol returns scorch_ops' own tensor type; the dense values live at
    # storage.value. Asserted rather than probed, because a digest taken over the wrong object
    # compares EQUAL across every arm and the whole check reads as a pass.
    if not (hasattr(r, "storage") and hasattr(r.storage, "value")):
        sys.exit("REFUSE: the native return has no storage.value; a digest over anything else "
                 "would compare equal across every arm and the check would read as a pass")
    v = r.storage.value.detach().cpu().numpy()
    v = np.ascontiguousarray(np.asarray(v, dtype=np.float32)).reshape(-1)
    if v.size != M * k:
        sys.exit("REFUSE: result has %d entries, expected %d -- the accessor is wrong and a digest "
                 "over the wrong bytes would compare equal across every arm" % (v.size, M * k))
    # POSITIVE CONTROL for the digest. Bit-identity across four objects is satisfied trivially if
    # every object returns zeros, so each reading carries a dense reference and the count of
    # NONZERO results is reported beside the count of matching digests.
    Bd = B.numpy().astype(np.float64)
    ref = _REF(M, J, indptr, indices, vals, Bd)
    got = v.reshape(M, k).astype(np.float64)
    # THE TOLERANCE SCALES WITH THE ACCUMULATION, not with the result. The kernel sums in float32
    # and the reference in float64, so the error bound is set by the magnitude of the terms summed
    # -- |A| @ |B| -- and not by |A @ B|, which can be near zero on a cancelling row while the
    # terms are large. A tolerance relative to the RESULT therefore fails on exactly the rows where
    # float32 has the most cancellation and nothing is wrong. This is the standard float32
    # accumulation bound, n * eps * sum|terms|, with n absorbed into a generous rtol; a prefetch
    # that read the wrong address corrupts a value outright, so the check still has all the power
    # it needs.
    scale = _REF(M, J, indptr, np.asarray(indices), np.abs(np.asarray(vals, dtype=np.float64)),
                 np.abs(Bd))
    tol = 1e-3 + 1e-3 * np.abs(scale)
    off = np.abs(got - ref) - tol
    if np.any(off > 0):
        # Report the element that actually FAILED -- the largest excess over its own tolerance --
        # not the largest absolute difference. Those are different elements whenever the tolerance
        # varies across the array, and the first version printed the second: a 1.1e-6 relative
        # difference on the largest entry, which passes, while some near-zero entry was the
        # violator.
        bad = int(np.argmax(off))
        sys.exit("REFUSE: result disagrees with the dense reference at flat index %d: got %r, "
                 "reference %r, tolerance %r (excess %r). A prefetch changes no arithmetic."
                 % (bad, got.reshape(-1)[bad], ref.reshape(-1)[bad],
                    tol.reshape(-1)[bad], off.reshape(-1)[bad]))
    return hashlib.sha256(v.tobytes()).hexdigest()[:32], v


out = open(A.csv, "w", newline="")
w = csv.DictWriter(out, fieldnames=["build", "case", "key", "k", "rows", "nnz", "sha"])
w.writeheader()

# ---------------------------------------------------------------- degenerate cases, asserted first
#
# `nnz` here is the matrix's total, which is exactly the pA_limit the clamp reads.
# TWO DIFFERENT CLAIMS, and the table says which cell carries which, because an earlier version of
# this header called all six "the clamp's edge values" and that is true of four.
#
#   PREMISE cells reach the clamp's edge BOUNDS without reaching the clamp. They are the witnesses
#   for the reachability premise -- the thing that keeps pA_limit == 0 from indexing A1_crd[-1] --
#   and `empty` is that premise's only witness. `empty` never calls the register-block kernel at
#   all: no row has nonzeros and the empty-row short-circuit continues first. `one_nz` calls it but
#   `pA + 1 < pA_end` is false at one nonzero, so only the single-nonzero tail runs and the tail
#   issues no prefetch.
#
#   CLAMP cells actually index through it. The smallest bound the clamp ever READS is pA_limit == 2
#   in two_nz, where `0 + 2 < 2` is false and pf_ = pA_limit - 1 = 1 names the last element of a
#   two-element array -- the in-bounds boundary, one past which is the bug.
DEGEN = [
    ("empty",      2, 4, [0, 0, 0],       [],              []),              # PREMISE: limit 0
    ("one_nz",     2, 4, [0, 1, 1],       [2],             [1.5]),           # PREMISE: limit 1
    ("two_nz",     1, 4, [0, 2],          [0, 3],          [1.0, 2.0]),      # CLAMP: limit 2, pf_=1
    ("row_of_2",   3, 4, [0, 2, 4, 6],    [0, 1, 1, 2, 2, 3],
     [1., 2., 3., 4., 5., 6.]),                                        # CLAMP: row-clip 0 prefetch
    ("row_of_3",   2, 4, [0, 3, 6],       [0, 1, 2, 1, 2, 3],
     [1., 2., 3., 4., 5., 6.]),                                        # CLAMP: row-clip 1 prefetch
    ("empty_rows", 4, 4, [0, 0, 2, 2, 2], [1, 3], [7.0, 8.0]),         # CLAMP: interior empty rows
]
print("DEGENERATE CASES (build %s): PREMISE cells reach the clamp's edge BOUNDS without\n  reaching the clamp; CLAMP cells index through it." % A.build, flush=True)
nz = 0
for name, M, J, ip, ix, vl in DEGEN:
    for k in KS:
        sha, v = run(M, J, ip, ix, vl, k, zlib.crc32(("%s|%d" % (name, k)).encode()) % 2**31)
        nz += int(np.count_nonzero(v) > 0)
        w.writerow(dict(build=A.build, case="degen", key=name, k=k, rows=M,
                        nnz=len(ix), sha=sha))
    print("  %-11s rows=%d nnz=%d  OK at all %d widths" % (name, M, len(ix), len(KS)), flush=True)
out.flush()

# ---------------------------------------------------------------- real matrices, bit-identity
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from csrresolve import resolve_all, describe
except ImportError:
    sys.exit("REFUSE: csrresolve.py is not importable next to this script; resolving corpus keys "
             "any other way is what silently dropped 50 of 298 keys on MKT")
keys = [l.strip() for l in open(A.keys) if l.strip()]
if A.limit:
    keys = keys[:A.limit]
paths, how, bad = resolve_all(A.cache, keys)
for line in describe(how, bad, "corpus keys"):
    print(line, flush=True)
if bad:
    sys.exit("REFUSE: %d of %d keys unresolved (first %s)" % (len(bad), len(keys), bad[0][0]))

n, nzr = 0, 0
for key in keys:
    z = np.load(paths[key])
    M, J = int(z["shape"][0]), int(z["shape"][1])
    nnz = int(z["indptr"][-1])
    for k in KS:
        sha, v = run(M, J, z["indptr"], z["indices"], z["data"], k,
                     zlib.crc32(("%s|%d" % (key, k)).encode()) % 2**31)
        nzr += int(np.count_nonzero(v) > 0)
        w.writerow(dict(build=A.build, case="real", key=key, k=k, rows=M, nnz=nnz, sha=sha))
        n += 1
    out.flush()
out.close()
# A COMPLETION SENTINEL, written only after the CSV is closed and carrying the row count it closed
# with. The comparator REQUIRES it, which is why it is written here and not by a shell wrapper: an
# exit code recorded by whatever shell happened to launch the producer does not exist in a clean
# checkout, and a check that skips when its input is missing reports an unrecorded exit as correct
# -- the same shape, one level down, as the comparator defect this sentinel was added to close.
with open(A.csv + ".done", "w") as _h:
    _h.write("%d %d %d\n" % (n + len(DEGEN) * len(KS), n, len(DEGEN) * len(KS)))
print("wrote %s (%d real rows + %d degenerate)" % (A.csv, n, len(DEGEN) * len(KS)), flush=True)
# Beside the digest count, the count that shows the digests are not over zeros.
print("  NONZERO RESULTS: %d of %d degenerate, %d of %d real -- a digest over an all-zero result "
      "matches across every arm, so this is what keeps the identity check from being vacuous"
      % (nz, len(DEGEN) * len(KS), nzr, n), flush=True)
