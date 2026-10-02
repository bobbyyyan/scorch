#!/usr/bin/env python3
"""The per-cell OPTIMAL worker count, measured, so any thread rule can be scored without building it.

WHY THIS AND NOT ANOTHER BUILD. A rule is a function from (nnz, rows, k, pool) to a worker count.
Building one object per candidate rule prices one function at a time; measuring the time at EVERY
reachable worker count prices the whole family at once, because a rule's cost on a cell is just
the measured time at the count it names, relative to the best count on that cell. That makes
registered prediction 1 -- that no single linear grain matches the per-width optima -- answerable
from one run, and it makes the comparison of the shipped rule, the ADOPT_GRAIN rule and the sqrt
rule a reading rather than three builds.

nthreads_override DOES deliver the count. With SCORCH_SPMM_NT_CAP = -2 a positive override is a
hard cap below the base count as well as a floor the adoption raises to, so the resolved count is
the override wherever rows/16 allows it. Every row records what the object said it resolved to,
asked of the object, so a cell where the override did NOT deliver is visible rather than assumed.

The reported time per (cell, count) is the minimum over `reps` batched means at a batch fixed by
one calibration at the pool -- the SAME batch for every count on that cell, because the batch is
the one parameter that biases a reading and a per-count batch would make the count and the batch
vary together.
"""
import argparse, csv, math, os, sys, time, zlib
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
TS = [int(x) for x in A.threads.split(",")] if A.threads else list(range(1, POOL + 1))


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
w = csv.DictWriter(out, fieldnames=["key", "k", "rows", "cols", "nnz", "deg", "nthreads",
                                    "resolved", "ms", "batch"])
w.writeheader()
nrow = 0
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
            c = sorted(time.perf_counter() - t for t in
                       [(lambda: (time.perf_counter(), f())[0])() for _ in range(5)])
            batch = max(1, int((A.target_ms * 1e-3) / max(c[2], 1e-9)))
        except Exception as exc:
            print("  SKIP %s k=%d calibration: %s" % (key, k, type(exc).__name__), flush=True)
            continue
        # Counts visited in a rotated order per cell, so a count does not always sit at the
        # warm or cold end of the cell's own sequence.
        off = zlib.crc32(("%s|%d" % (key, k)).encode()) % len(TS)
        for nt in TS[off:] + TS[:off]:
            fn = at(nt)
            res = so.scorch_spmm_nthreads(nnz * max(k, 16), M, nt, nnz * k, nnz)
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
                            ms="%.6f" % (best * 1e3), batch=batch))
            nrow += 1
        out.flush()
print("wrote %d rows to %s" % (nrow, A.csv), flush=True)
