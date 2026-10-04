#!/usr/bin/env python3
"""Decompose a short SpMM call: where do ours' extra microseconds go, against MKL's?

Cells: screen-sample keys whose prior MKL time at k=2 fell in [lo, hi) microseconds -- the band
holding most of the deficit. Arms, all in this one process:
  ours_kN   the harness call (keyword arguments) with nthreads_override=N, N in 1..32
  ours_pos  the same call with positional arguments at N=32 (prices keyword parsing)
  mkl_tN    torch.sparse.mm with torch.set_num_threads(N)
Fixed costs: the same arms on a 1x1 matrix with one nonzero, where the arithmetic is nil.

usage: decomp.py <cache_dir> <sample_dir> <prior_csv_dir> [lo_us hi_us per_corpus]
"""
import csv, glob, math, os, sys, time, zlib
import numpy as np, torch
import scorch_ops as so
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csrresolve import resolve_all

cache, sdir, prior = sys.argv[1], sys.argv[2], sys.argv[3]
lo, hi = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (10.0, 20.0)
per = int(sys.argv[6]) if len(sys.argv) > 6 else 12
POOL = torch.get_num_threads()
SYM = so.spmm_csr_float_v2
NS = [1, 2, 4, 8, 16, 32]
print("pool", POOL, "object", so.__file__, flush=True)

def tmin(fn, reps=7, target=0.01):
    fn(); fn(); fn()
    c = []
    for _ in range(5):
        t0 = time.perf_counter(); fn(); c.append(time.perf_counter() - t0)
    b = max(1, int(target / max(sorted(c)[2], 1e-9)))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(b): fn()
        best = min(best, (time.perf_counter() - t0) / b)
    return best * 1e6

def arms(ipt, ixt, vt, M, J, k, seed):
    g = torch.Generator().manual_seed(seed)
    B = torch.rand((J, k), generator=g)
    Bv = B.reshape(-1).contiguous()
    A = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
    kw = dict(result_shape=[M, k], A_shape=[M, J], A_mode_indices=[[], [ipt, ixt]], A_values=vt,
              B_shape=[J, k], B_mode_indices=[[], []], B_values=Bv)
    out = {}
    for n in NS:
        out["ours_k%d" % n] = tmin(lambda: SYM(nthreads_override=n, atparallel=False, **kw))
    out["ours_pos"] = tmin(lambda: SYM([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv,
                                       256, POOL, False))
    for n in (1, 4, 32):
        torch.set_num_threads(n)
        out["mkl_t%d" % n] = tmin(lambda: torch.sparse.mm(A, B))
    torch.set_num_threads(POOL)
    out["empty"] = tmin(lambda: torch.empty((M, k)))
    return out

# Fixed costs on a 1x1 matrix.
one = (torch.tensor([0, 1], dtype=torch.int32), torch.tensor([0], dtype=torch.int32),
       torch.tensor([1.0]))
for k in (2, 8):
    r = arms(*one, 1, 1, k, 1)
    print("FIXED 1x1 k=%d  " % k + "  ".join("%s %.2f" % kv for kv in r.items()), flush=True)

# Band cells from the prior full-collection timing of the shipped build.
for corpus in ("ss", "dlmc"):
    keys = [l.strip() for l in open(os.path.join(sdir, "screen_%s.txt" % corpus)) if l.strip()]
    mk = {}
    for p in glob.glob(os.path.join(prior, "%s_float32_ship_s*.csv" % corpus)):
        for r in csv.DictReader(open(p)):
            if r["arm"] == "mkl" and r["k"] == "2": mk[r["key"]] = float(r["ms"]) * 1e3
    band = [k for k in keys if lo <= mk.get(k, -1) < hi]
    pick = band[:: max(1, len(band) // per)][:per]
    print("%s: %d of %d sample keys in MKL band [%g,%g) us; probing %d" % (corpus, len(band), len(keys), lo, hi, len(pick)), flush=True)
    paths, _, bad = resolve_all(os.path.join(cache, corpus), pick)
    res = {}
    for key in pick:
        z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
        ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
        vt = torch.from_numpy(z["data"].astype(np.float32)); nnz = int(z["indptr"][-1])
        for k in (2, 8):
            r = arms(ipt, ixt, vt, M, J, k, zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
            res.setdefault(k, []).append(r)
            best_n = min(NS, key=lambda n: r["ours_k%d" % n])
            print("  %-60s M=%6d nnz=%8d k=%d  ours32 %.1f best ours %.1f@%d  mkl32 %.1f mkl1 %.1f  pos %.1f"
                  % (key[-60:], M, nnz, k, r["ours_k32"], r["ours_k%d" % best_n], best_n, r["mkl_t32"], r["mkl_t1"], r["ours_pos"]), flush=True)
    med = lambda xs: sorted(xs)[len(xs) // 2]
    for k, rs in sorted(res.items()):
        names = list(rs[0])
        print("%s k=%d MEDIAN us: " % (corpus, k) + "  ".join("%s %.1f" % (a, med([r[a] for r in rs])) for a in names))
        bo = [min(r["ours_k%d" % n] for n in NS) for r in rs]
        print("%s k=%d  oracle-thread ours beats mkl32 on %d/%d; ours32 beats mkl32 on %d/%d"
              % (corpus, k, sum(b < r["mkl_t32"] for b, r in zip(bo, rs)), len(rs),
                 sum(r["ours_k32"] < r["mkl_t32"] for r in rs), len(rs)))
print("DECOMP DONE")
