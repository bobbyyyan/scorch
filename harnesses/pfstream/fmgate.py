#!/usr/bin/env python3
"""The PFSTREAM screen's FUSED-LINEAR cell: four builds over the fused feature-major path.

WHY A SECOND DRIVER RATHER THAN A FLAG ON mrgate.py. mrgate.py is the instrument for two runs that
are in flight, and its digest is recorded in the pre-registration. Editing it mid-flight would
change the instrument under a live measurement. It also has no fused entry point to switch to: the
fused symbol takes a bias and an activation that the v2 signature does not have.

WHAT THIS PATH IS. spmm_csr_linear_fused_float computes Y[out,batch] = act(W @ X[in,batch] +
bias[:,None]) in one parallel region, with the bias and activation folded into the row epilogue.
The narrow-k dimension is therefore the BATCH, not a feature count. There is no float64 variant --
spmm_csr_linear_fused_double does not exist -- so this driver refuses any other dtype rather than
silently measuring something else.

THE CORPUS IS THE SAME SuiteSparse SAMPLE, with the matrix standing in as the weight and k as the
batch. That is artificial as a workload and correct as a kernel measurement, and it is deliberate:
the same matrices make this cell comparable to the drop-in screen's cell rather than only to
itself.

k=2 IS AN ARM HERE AND NOT A CONTROL, WHICH IS THE OPPOSITE OF THE DROP-IN PATH. On the drop-in,
float32 k in {1,2,3} routes to scorch_spmm_row_narrow_exact and the register block is bypassed, so
k=2 is a free negative control. The fused path has NO exact-width branch and NO half-vector branch
-- verified by zero references to narrow_exact, exact_width, halfvec, HALFVEC or NARROWK anywhere
inside spmm_csr_linear_fused_float -- and its own constants are hardcoded rather than derived:

    narrow_k = (B1_size >= 1 && B1_size <= 32)      spmm.h:6242
    nvec     = (B1_size + 7) / 8                    spmm.h:6243
    full_last = (mlast == 8)                        spmm.h:6245

So the register block is reached at EVERY width 1..32 on this path, including 1, 2 and 3, and those
widths reach (nvec 1, MASKED tail) -- instantiations neither the float32 drop-in screen nor the
float64 one can reach at all. Treating "the register block is unreachable at narrow k" as a property
of the kernel rather than of one entry point is a mistake this file exists partly to correct.

AND THAT MAKES THIS CELL CLOSE THE MASKED-TAIL AXIS, which the drop-in screen does not. The
formula is the same on both paths -- full_last = (mlast == 8) -- but the reachable DOMAIN is not,
because the v2 core loses widths 1-3 to narrow_exact and width 4 to halfvec while the fused path
loses nothing. Derived over all 32 widths: v2 can reach a masked tail at 24 of them, the fused path
at 28, and widths 1, 2, 3 and 4 are masked-and-reachable here while unreachable on v2 at all. So
even the drop-in screen's own width set {2,4,8,16,32} reaches a masked tail on this path, at k=2
(mlast 2) and k=4 (mlast 4). Same formula, different domain -- and "the fused path has the same
full_last rule, so it cannot close that axis either" is the inference that does not follow.

THE TWO PATHS HAND THE CLAMP DIFFERENT QUANTITIES, AND IN THIS BUILD CLASS THEY COINCIDE. The fused
path computes its own `const int total_nnz = A1_pos[A0_size]` at spmm.h:6157; the v2 core passes
SCORCH_SPMM_NNZ_TOTAL, which is A1_pos[SCORCH_SPMM_ROW_UNITS] and therefore redirect-aware
(spmm.h:3558-3563). They differ only when SCORCH_SPMM_SPLIT_ENABLED is 1, which needs
SCORCH_TUNE_HOOKS or SCORCH_SPMM_SPLIT_MIN_IMBALANCE > 0 -- and that ships at 0L, so in the four
screen builds ROW_UNITS is A0_size and both paths pass A1_pos[A0_size]. The difference is
structural and LATENT here, which is why this driver does not try to correct for it: it would be
correcting for something that cannot differ in the objects being measured.

THE NEGATIVE CONTROL IS THEREFORE k=64, where narrow_k is false and the row takes
scorch_spmm_row_regtile<float> instead, so PFSTREAM is unreachable. Two controls with different
jobs, and they are not substitutes:
  * k=64 is a width where the arm provably cannot fire, so it bounds false positives from the
    instrument -- calibration, process, build drift. Its floor MAGNITUDE is measured on a different
    kernel, so it does not transfer to the narrow widths.
  * shpb against ship at each width is the per-width build floor, and that is the one an effect at
    a given width must clear.

act IS FIXED AT 1 (relu) AND THAT IS ONE VALUE ON PURPOSE. The activation is a scalar epilogue
applied to the row after it is written; all three values read and write the same bytes and differ
only in the scalar op, so it is not an axis a prefetch clamp inside the row kernel can interact
with. Sweeping it would add a third of the wall time for a factor the mechanism cannot touch.

THE MKL COLUMN IS A DRIFT REFERENCE, NOT A BASELINE, and the arm is named mklref so the schema
carries that. torch.sparse.mm does the SpMM without the bias and activation, so ours/mklref differs
from 1 by the epilogue as well as by the kernel and IS NOT A SPEEDUP. It is here for the one job
mrgate uses it for: a fixed workload timed in the same process as ours in every cell, so ours
against mklref cancels process-level drift between builds even where ours-here against ours-there
does not. The fused path's standing result against MKL was measured elsewhere, on the autoencoder
workload with the bias and activation included on both sides.
"""
import argparse, csv, os, sys, time, zlib
ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--keys", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--dtype", default="float32", help="float32 only; there is no fused double symbol")
# Each width names the (nvec, tail) instantiation it exists to reach, and the set covers nvec 1-4
# with both tail states at nvec 1, 2 and 3. The one gap is (nvec 4, masked), which needs k in
# 25..31 -- stated rather than closed, because an eighth width is already 6 hours of wall time and
# the masked tail is exercised at three other vector counts.
#   2  nvec 1 MASKED(2)   unreachable on the drop-in path at all
#   4  nvec 1 MASKED(4)   unreachable on the drop-in path at all
#   8  nvec 1 full
#  12  nvec 2 MASKED(4)
#  16  nvec 2 full
#  20  nvec 3 MASKED(4)
#  32  nvec 4 full
#  64  NEGATIVE CONTROL -- narrow_k false, regblock and so PFSTREAM unreachable
ap.add_argument("--ks", default="2,4,8,12,16,20,32,64")
ap.add_argument("--act", type=int, default=1, help="0 identity, 1 relu, 2 sigmoid; see the docstring")
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--target-ms", type=float, default=15.0)
ap.add_argument("--pass-tag", default="p1")
ap.add_argument("--build", required=True, help="label recorded in every row: which build this is")
ap.add_argument("--slice", default="all", help="label recorded in every row: which slice this is")
A = ap.parse_args()
if A.dtype != "float32":
    sys.exit("REFUSE: --dtype %s. spmm_csr_linear_fused_double does not exist, so any other dtype "
             "would silently measure a different path." % A.dtype)
if A.act not in (0, 1, 2):
    sys.exit("REFUSE: --act %d; the native symbol accepts 0, 1 or 2 and raises otherwise." % A.act)
import numpy as np, torch, scorch_ops as so
if so.scorch_tune_hooks():
    sys.exit("REFUSE: this is a hooks object. Against MKL that charges us ~19 microseconds of "
             "getenv a call and manufactures a deficit on short kernels; the arms here are builds.")
if not hasattr(so, "spmm_csr_linear_fused_float"):
    sys.exit("REFUSE: this object has no spmm_csr_linear_fused_float. A driver whose symbol is "
             "absent would otherwise report SKIP on every cell and exit 0, which is the shape that "
             "makes a failure to measure read as a null result.")
SYM = so.spmm_csr_linear_fused_float
KS = [int(x) for x in A.ks.split(",")]
POOL = torch.get_num_threads()
ARMS = ["ours", "oursb", "mklref"]
# Widths above this take the ordinary path, where the register block -- and so PFSTREAM -- is
# unreachable. Read from spmm.h:6242 rather than assumed, and recorded per row so the analysis does
# not have to restate it. A harness that recomputes a production predicate is wrong silently.
FUSED_NARROW_HI = 32


def calibrate(fn):
    """How many calls a rep should make. Separated from the timing on purpose -- see `timeit`."""
    fn(); fn(); fn()
    cals = []
    for _ in range(5):
        t0 = time.perf_counter(); fn(); cals.append(time.perf_counter() - t0)
    cals.sort()
    return max(1, int((A.target_ms * 1e-3) / max(cals[len(cals) // 2], 1e-9)))


def timeit(fn, batch):
    """Min over `reps` batched means, at a batch the CALLER fixes.

    Identical discipline to mrgate.py, and for the reason recorded there: calibrating inside the
    timer gave each arm its own batch, a small batch reads systematically slower than a large one,
    and two arms running IDENTICAL code came out up to 3x apart. The batch is fixed once per cell
    and handed to every arm running the same code, so batch divergence between an arm and its own
    floor is zero by construction. mklref is calibrated separately, since it is a different
    operation at a different speed and sharing a batch would put one of the two in a useless window.
    """
    b = float("inf")
    for _ in range(A.reps):
        t0 = time.perf_counter()
        for _ in range(batch): fn()
        b = min(b, (time.perf_counter() - t0) / batch)
    return b * 1e3, batch


keys = [l.strip() for l in open(A.keys) if l.strip()]
# The corpus key -> cached path resolution lives in ONE place. A consumer that cannot import it
# must refuse: falling back to os.path.exists is the exact bug that had MKT reading 248 of 298 keys.
try:
    from csrresolve import resolve_all, describe
except ImportError:
    sys.exit("REFUSE: csrresolve.py is not importable next to this script. Every consumer resolving "
             "corpus keys its own way is what silently dropped 50 of 298 keys on MKT -- so this "
             "refuses rather than falling back to an exact-path lookup.")
_paths, _how, _bad = resolve_all(A.cache, keys)
for _l in describe(_how, _bad, "corpus keys"):
    print(_l, flush=True)
if _bad:
    sys.exit("REFUSE: %d of %d keys unresolved under %s (first %s). Resolution is exact-path then "
             "unique basename; an unresolved key here means the cache genuinely lacks it or two "
             "matrices share a basename." % (len(_bad), len(keys), A.cache, _bad[0][0]))
out = open(A.csv, "w", newline="")
w = csv.DictWriter(out, fieldnames=["build", "slice", "key", "k", "rows", "nnz", "deg", "act",
                                    "regblock", "nvec", "tail", "arm", "ms", "batch"])
w.writeheader()
print("build %s slice %s path fused_float act %d pool %d %d matrices %d widths"
      % (A.build, A.slice, A.act, POOL, len(keys), len(KS)), flush=True)
nrow = 0
for key in keys:
    z = np.load(_paths[key])
    M, J = int(z["shape"][0]), int(z["shape"][1])
    ipt = torch.from_numpy(z["indptr"].astype(np.int32))
    ixt = torch.from_numpy(z["indices"].astype(np.int32))
    vt = torch.from_numpy(z["data"].astype(np.float32))
    nnz = int(z["indptr"][-1])
    Acsr = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
    for k in KS:
        g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
        B = torch.rand((J, k), generator=g).to(torch.float32)
        bias = torch.rand((M,), generator=g).to(torch.float32)
        native = dict(result_shape=[M, k], A_shape=[M, J], A_mode_indices=[[], [ipt, ixt]],
                      A_values=vt, B_shape=[J, k], B_mode_indices=[[], []],
                      B_values=B.reshape(-1).contiguous(), bias_values=bias, act=A.act)

        def ours():
            return SYM(nthreads_override=POOL, atparallel=False, **native)

        fns = {"ours": ours, "oursb": ours, "mklref": lambda: torch.sparse.mm(Acsr, B)}
        # One batch for every arm that runs OUR code, so an arm and its own floor cannot differ in
        # the one parameter that biases the reading. mklref gets its own; it is a different
        # operation at a different speed.
        try:
            batch_ours = calibrate(ours)
            batch_ref = calibrate(fns["mklref"])
        except Exception as exc:
            print("  SKIP %s k=%d calibration: %s: %s"
                  % (key, k, type(exc).__name__, exc), flush=True)
            continue
        # Which register-block instantiation this width selects, derived from spmm.h's own
        # constants and recorded so the analysis reads it rather than recomputing it.
        rb = 1 <= k <= FUSED_NARROW_HI
        nvec = (k + 7) // 8 if rb else 0
        tail = ("full" if (k - 8 * (nvec - 1)) == 8 else "masked") if rb else "n/a"
        order = list(ARMS)
        np.random.default_rng(zlib.crc32(("%s|%d|%s|%s" % (key, k, A.pass_tag, A.build))
                                         .encode())).shuffle(order)
        for arm in order:
            try: ms, batch = timeit(fns[arm], batch_ref if arm == "mklref" else batch_ours)
            except Exception as exc:
                print("  SKIP %s k=%d %s: %s" % (key, k, arm, type(exc).__name__), flush=True)
                continue
            w.writerow(dict(build=A.build, slice=A.slice, key=key, k=k, rows=M, nnz=nnz,
                            deg="%.2f" % (nnz / max(M, 1)), act=A.act,
                            regblock=int(rb), nvec=nvec, tail=tail, arm=arm,
                            ms="%.6f" % ms, batch=batch))
            nrow += 1
    out.flush()
out.close()
print("wrote %s (%d rows)" % (A.csv, nrow), flush=True)
