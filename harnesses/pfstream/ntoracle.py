#!/usr/bin/env python3
"""The per-cell OPTIMAL worker count, measured, so any thread rule can be scored without building it.

WHY THIS AND NOT ANOTHER BUILD. A rule is a function from (nnz, rows, k, pool) to a worker count.
Building one object per candidate rule prices one function at a time; measuring the time at EVERY
reachable worker count prices the whole family at once, because a rule's cost on a cell is just
the measured time at the count it names, relative to the best count on that cell. That makes
registered prediction 1 -- that no single linear grain matches the per-width optima -- answerable
from one run, and it makes the comparison of the shipped rule, the ADOPT_GRAIN rule and the sqrt
rule a reading rather than three builds.

THE OVERRIDE OFTEN DOES NOT DELIVER THE COUNT, and this is written the way it is because the
first version of this file asserted that it does. Two clauses stop it. The adoption branch is
gated -- `if (nthreads_override > 0 && gate_work >= grain)` with gate_work = nnz*max(k,16) -- so
below the gate the override is purely a cap and the resolved count is the base policy count. And
the final cap DECLINES when the resolved count reaches twice the caller's pool, to preserve the
recruit that makes the kernel launch its own team, so a large-policy cell cannot be lowered at
all. Asked of the object over the tracked 72-key corpus at pool 6 and eight widths, 231 of 576
cells (40.1%) resolve to ONE count for every override in 1..6 -- 141 of them pinned at 1 by the
policy floor and 90 pinned at 18 by the recruit decline.

So this sweeps the OVERRIDE and reports the RESOLVED count, and the analysis groups by resolved.
Two consequences, one of which is a gain. Cells with a single reachable count are excluded: the
sweep would otherwise time one configuration six times and a "regret" there would be noise, and a
rule compared to that argmin would be compared to a coin flip. And where two overrides resolve to
the SAME count, those readings time the same configuration with a different argument, which is a
free same-object A/A floor at the cell level -- so the noise the first version would have reported
as effect becomes the thing the effect is scored against.

The reported time per (cell, count) is the minimum over `reps` batched means at a batch fixed by
one calibration at the pool -- the SAME batch for every count on that cell, because the batch is
the one parameter that biases a reading and a per-count batch would make the count and the batch
vary together.
"""
import argparse, collections, csv, math, os, sys, time, zlib
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--keys", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--dtype", default="float32")
ap.add_argument("--ks", default="2,4,8,12,16,20,24,32")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--target-ms", type=float, default=10.0)
ap.add_argument("--threads", default="", help="comma list; default 1..pool")
A = ap.parse_args()
import numpy as np, torch, scorch_ops as so
if so.scorch_tune_hooks():
    sys.exit("REFUSE: hooks object; the per-call getenv tax lands on exactly the short kernels "
             "this is about.")
NPDT = np.float32 if A.dtype == "float32" else np.float64
TDT = torch.float32 if A.dtype == "float32" else torch.float64
SYM = so.spmm_csr_float_v2 if A.dtype == "float32" else so.spmm_csr_double_v2
KS = [int(x) for x in A.ks.split(",")]
POOL = torch.get_num_threads()
TS = [int(x) for x in A.threads.split(",")] if A.threads else [0] + list(range(1, POOL + 1))
# EACH OVERRIDE IS TIMED TWICE, so every reachable count carries a same-object repeat and the
# cell-level A/A floor exists on every live cell. Relying on two overrides happening to resolve
# to the same count does not work: over the tracked corpus at pool 6, overrides 1..6 give a
# distinct-count histogram of {1: 231, 4: 18, 6: 327}, so the accidental supply is 54 pairs on
# 18 cells -- 3.1% of cells -- and 56.8% of cells supply none at all. Override 0 is included in
# the sweep because it is a real configuration: the standalone and panel paths pass no override.
REPEAT = 2


def read_smtx(path):
    with open(path) as fh:
        head = fh.readline(); ipt = fh.readline(); idx = fh.readline()
    rows, cols, nnz = (int(x) for x in head.replace(",", " ").split())
    indptr = np.array(ipt.split(), dtype=np.int64)
    indices = np.array(idx.split(), dtype=np.int64)
    if len(indptr) != rows + 1 or int(indptr[-1]) != nnz or len(indices) != nnz:
        raise ValueError("%s: truncated -- it parses cleanly and reads as a sparser matrix" % path)
    return indptr, indices, rows, cols


def load(key):
    p = os.path.join(A.root, key + ".smtx")
    if os.path.exists(p):
        return read_smtx(p)
    p = os.path.join(A.root, key + ".npz")
    z = np.load(p)
    s = z["shape"]
    return z["indptr"], z["indices"], int(s[0]), int(s[1])


keys = [l.strip() for l in open(A.keys) if l.strip() and not l.startswith("#")]
loaded, bad = {}, []
for key in keys:
    try:
        loaded[key] = load(key)
    except Exception as exc:
        bad.append((key, type(exc).__name__))
if bad:
    sys.exit("REFUSE: %d of %d keys unresolved (first %s); a silently shorter corpus is a "
             "different experiment." % (len(bad), len(keys), bad[0][0]))
print("corpus %d keys  pool %d  threads %s  widths %s" % (len(keys), POOL, TS, KS), flush=True)

out = open(A.csv, "w", newline="")
# `policy` and `cap_rows` are recorded from the OBJECT, not recomputed downstream. The analyser
# needs the base policy count to say whether a rule could have reached the measured optimum at
# all, and a harness that recomputes a production policy is wrong silently and flatteringly --
# an earlier draft of the analyser reimplemented it and disagreed with the object on 16% of cells,
# always in the same direction.
# `res_ovr0` IS NOT THE POLICY COUNT and is named for what it is. With override <= 0 the final
# cap is `nt_cap = omp_get_max_threads()` (scorch_policy.h:2203), so the reading is the policy
# count CAPPED BY THE PROCESS POOL, with the >= 2*pool decline punching through for large values.
# Measured at pool 6 over the tracked corpus, 342 of 576 cells (59.4%) have a saturating count
# strictly above it, gaps up to 17. Recording it is right -- the standalone and panel paths pass
# no override, so it is a configuration to TIME -- but calling it `policy` and reasoning from it
# as a lower bound is not, and a bound taken from it is one-signed too small.
# `saturating` is the count at an override far above anything, recorded so the gap is visible in
# the data rather than having to be rederived. Both carry `pool`, because a number derived from a
# policy query is not a number until its pool is attached.
w = csv.DictWriter(out, fieldnames=["key", "k", "rows", "cols", "nnz", "deg", "nthreads",
                                    "resolved", "res_ovr0", "saturating", "cap_rows", "pool",
                                    "ms", "batch"])
w.writeheader()
nrow = 0
CELLN = collections.Counter()
for key in keys:
    indptr, indices, M, J = loaded[key]
    nnz = int(indptr[-1])
    ipt = torch.from_numpy(np.ascontiguousarray(indptr, dtype=np.int32))
    ixt = torch.from_numpy(np.ascontiguousarray(indices, dtype=np.int32))
    g0 = torch.Generator().manual_seed(zlib.crc32(key.encode()) % (2 ** 31))
    vt = (torch.rand(nnz, generator=g0) - 0.5).to(TDT)
    for k in KS:
        g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2 ** 31))
        B = torch.rand((J, k), generator=g).to(TDT)
        native = dict(result_shape=[M, k], A_shape=[M, J], A_mode_indices=[[], [ipt, ixt]],
                      A_values=vt, B_shape=[J, k], B_mode_indices=[[], []],
                      B_values=B.reshape(-1).contiguous())

        def at(nt):
            return lambda: SYM(nthreads_override=nt, atparallel=False, **native)

        # ONE calibration, at the pool, shared by every count on this cell.
        f = at(POOL)
        try:
            f(); f(); f()
            # Estimate the per-call cost from a WARM BATCH, not from single calls. A single call
            # carries cold overhead the batched loop does not pay, which inflates the estimate
            # and shrinks the batch by the same factor.
            probe = 32
            t0 = time.perf_counter()
            for _ in range(probe):
                f()
            per = (time.perf_counter() - t0) / probe
            batch = max(1, int((A.target_ms * 1e-3) / max(per, 1e-9)))
        except Exception as exc:
            print("  SKIP %s k=%d calibration: %s" % (key, k, type(exc).__name__), flush=True)
            continue
        # Counts visited in a rotated order, with the offset taken from a COUNTER over cells
        # rather than from a hash of the key. A hash balances position only in expectation, and
        # in the 36-cell per-width window the analysis actually reports that is not balanced at
        # all -- one count led 11 times and another once against an ideal of 6. A counter makes
        # each count lead within one of an equal share inside every width's own window.
        off = CELLN[k] % len(TS)
        CELLN[k] += 1
        for nt in (TS[off:] + TS[:off]) * REPEAT:
            fn = at(nt)
            res = so.scorch_spmm_nthreads(nnz * max(k, 16), M, nt, nnz * k, nnz)
            pol = so.scorch_spmm_nthreads(nnz * max(k, 16), M, 0, nnz * k, nnz)
            sat = so.scorch_spmm_nthreads(nnz * max(k, 16), M, 1 << 20, nnz * k, nnz)
            best = float("inf")
            try:
                for _ in range(A.reps):
                    t0 = time.perf_counter()
                    for _ in range(batch):
                        fn()
                    dt = (time.perf_counter() - t0) / batch
                    best = min(best, dt)
            except Exception as exc:
                print("  SKIP %s k=%d nt=%d: %s" % (key, k, nt, type(exc).__name__), flush=True)
                continue
            w.writerow(dict(key=key, k=k, rows=M, cols=J, nnz=nnz,
                            deg=round(nnz / max(M, 1), 2), nthreads=nt, resolved=res,
                            res_ovr0=pol, saturating=sat, cap_rows=max(1, M // 16),
                            pool=POOL,
                            ms="%.6f" % (best * 1e3), batch=batch))
            nrow += 1
        out.flush()
print("wrote %d rows to %s" % (nrow, A.csv), flush=True)
