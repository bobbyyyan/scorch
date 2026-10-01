#!/usr/bin/env python3
"""The multi-row register block's pre-registered gate: a compiled-in run on both x86 hosts.

scorch_policy.h ships SCORCH_SPMM_MULTIROW_ROWS at 0 "until a compiled-in three-build on both
hosts agrees with the hooked grid above". This is that build. The arms are therefore BUILDS, not
knobs, and this refuses a hooks object: against MKL a hooks object charges us ~19 microseconds of
getenv a call that MKL does not pay, which is enough to invert a short cell.

Because the arms are builds, drift between them is the threat. Two defences: the caller runs this in
slices and rotates which build leads each slice, and MKL is timed in the same process as ours in
every cell, so ours/MKL is immune to drift between builds even when ours-here/ours-there is not.

This file records only INPUTS -- dtype, width, rows, nonzeros -- and the build label. It does not
compute whether the multi-row kernel serves a cell. That predicate lives in spmm.h, and a harness
that restates a production policy is wrong silently and flatteringly. The analysis predicts firing
from the rule and then checks the prediction against measured movement between the builds, which
tests what the binary did rather than what it was meant to do.

`oursb` is `ours` again -- identical code, identical argument -- and is the noise floor.
"""
import argparse, csv, os, sys, time, zlib
ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--keys", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--dtype", default="float32")
ap.add_argument("--ks", default="1,2,4,8,16,64")
ap.add_argument("--reps", type=int, default=13)
ap.add_argument("--target-ms", type=float, default=25.0)
ap.add_argument("--pass-tag", default="p1")
ap.add_argument("--build", required=True, help="label recorded in every row: which build this is")
ap.add_argument("--slice", default="all", help="label recorded in every row: which slice this is")
A = ap.parse_args()
import numpy as np, torch, scorch_ops as so
if so.scorch_tune_hooks():
    sys.exit("REFUSE: this is a hooks object. Against MKL that charges us ~19 microseconds of "
             "getenv a call and manufactures a deficit on short kernels; the arms here are builds.")
NPDT = np.float32 if A.dtype == "float32" else np.float64
TDT = torch.float32 if A.dtype == "float32" else torch.float64
SYM = so.spmm_csr_float_v2 if A.dtype == "float32" else so.spmm_csr_double_v2
KS = [int(x) for x in A.ks.split(",")]
POOL = torch.get_num_threads()
ARMS = ["ours", "oursb", "mkl"]

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

    The batch has to be an argument, and this is the third version of that decision.

    Calibrating inside the timer gave each arm its own batch. A small batch reads systematically
    slower than a large one, so an arm that happened to draw a slow calibration call carried the
    penalty for the whole cell, and two arms running IDENTICAL code came out up to 3x apart.
    Grouping redwood's same-code floor by how far the two arms' batches had diverged made it plain:
    within 1.2x of each other the 5-95 spread is 13.6%, at 1.2-2x 19.9%, at 2-5x 36.6%, beyond 5x
    78.2%, with the median at 1.0000 throughout -- variance, not bias, and a floor looser than the
    hardware.

    Averaging the calibration does not fix it. The minimum of five calls sets the batch far too
    large whenever the fastest call is much faster than a typical one, tripling the runtime for no
    precision. The median of five still left the two same-code arms 4.4x apart at p95 and 29.6x at
    worst, because the variability is in the kernel and not in the estimator.

    So the batch is fixed ONCE per cell by the caller and handed to every arm running the same code.
    Then batch divergence between an arm and its own floor is zero by construction rather than
    small on average. MKL is calibrated separately, since it can be twenty times a different speed
    and sharing a batch with it would put one of the two in a useless window.
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
w = csv.DictWriter(out, fieldnames=["build", "slice", "key", "k", "rows", "nnz", "deg",
                                    "arm", "ms", "batch"])
w.writeheader()
print("build %s slice %s dtype %s pool %d %d matrices %d widths"
      % (A.build, A.slice, A.dtype, POOL, len(keys), len(KS)), flush=True)
nrow = 0
for key in keys:
    z = np.load(_paths[key])
    M, J = int(z["shape"][0]), int(z["shape"][1])
    ipt = torch.from_numpy(z["indptr"].astype(np.int32))
    ixt = torch.from_numpy(z["indices"].astype(np.int32))
    vt = torch.from_numpy(z["data"].astype(NPDT))
    nnz = int(z["indptr"][-1])
    Acsr = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
    for k in KS:
        g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
        B = torch.rand((J, k), generator=g).to(TDT)
        native = dict(result_shape=[M, k], A_shape=[M, J], A_mode_indices=[[], [ipt, ixt]],
                      A_values=vt, B_shape=[J, k], B_mode_indices=[[], []],
                      B_values=B.reshape(-1).contiguous())
        def ours():
            return SYM(nthreads_override=POOL, atparallel=False, **native)
        fns = {"ours": ours, "oursb": ours, "mkl": lambda: torch.sparse.mm(Acsr, B)}
        # One batch for every arm that runs OUR code, so an arm and its own floor cannot differ in
        # the one parameter that biases the reading. MKL gets its own; it can be twenty times a
        # different speed.
        try:
            batch_ours = calibrate(ours)
            batch_mkl = calibrate(fns["mkl"])
        except Exception as exc:
            print("  SKIP %s k=%d calibration: %s" % (key, k, type(exc).__name__), flush=True)
            continue
        order = list(ARMS)
        np.random.default_rng(zlib.crc32(("%s|%d|%s|%s" % (key, k, A.pass_tag, A.build))
                                         .encode())).shuffle(order)
        for arm in order:
            try: ms, batch = timeit(fns[arm], batch_mkl if arm == "mkl" else batch_ours)
            except Exception as exc:
                print("  SKIP %s k=%d %s: %s" % (key, k, arm, type(exc).__name__), flush=True)
                continue
            w.writerow(dict(build=A.build, slice=A.slice, key=key, k=k, rows=M, nnz=nnz,
                            deg="%.2f" % (nnz / max(M, 1)), arm=arm,
                            ms="%.6f" % ms, batch=batch))
            nrow += 1
    out.flush()
out.close()
print("wrote %s (%d rows)" % (A.csv, nrow), flush=True)
