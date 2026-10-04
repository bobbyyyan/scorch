#!/usr/bin/env python3
"""Small products: one worker inline vs the full team, ours and MKL, warm and cold.

A 32-wide libgomp region costs ~6 us before any worker starts on MKT, and PyTorch's MKL runs
on the same libgomp pool. So for a product whose whole single-thread time is a few us, the
team cannot pay for itself -- and MKL, at 12.5 us on 512-row DLMC layers where ours takes 15,
may simply not be using it. Per cell: ours with nthreads_override=P (production) and =1
(inline, no team), MKL with torch's pool at P and at 1; warm best-of-batches and cold median.

usage: serial_vs_team.py <cache_dir> <sample_dir> <out_csv> [ks] [max_nnz]
"""
import csv, os, sys, time, zlib, statistics as st
import numpy as np, torch, scorch_ops as so
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
cache, sdir, out = sys.argv[1:4]; KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "2,3,4,8,16,32").split(",")]
MAXNNZ = int(sys.argv[5]) if len(sys.argv) > 5 else 400000
P = torch.get_num_threads(); F = torch.rand(256 << 18); REP = 15
def tmin(fn, reps=5, target=0.004):
    fn(); fn(); c = []
    for _ in range(3):
        t0 = time.perf_counter(); fn(); c.append(time.perf_counter() - t0)
    b = max(2, int(target / max(sorted(c)[1], 1e-9))); best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(b): fn()
        best = min(best, (time.perf_counter() - t0) / b)
    return best * 1e6
def tcold(fn, Bv):
    fn(); xs = []
    for _ in range(REP):
        float(F.sum()); Bv.mul_(1.0)
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    return st.median(xs) * 1e6
w = csv.writer(open(out, "w")); w.writerow(["corpus", "key", "k", "rows", "nnz", "arm", "us"])
for corpus in ("ss", "dlmc"):
    keys = [l.strip() for l in open(os.path.join(sdir, "screen_%s.txt" % corpus)) if l.strip()]
    paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
    for key in keys:
        z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1]); nnz = int(z["indptr"][-1])
        if nnz > MAXNNZ: continue
        ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
        vt = torch.from_numpy(z["data"].astype(np.float32))
        A = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
        for k in KS:
            g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
            B = torch.rand((J, k), generator=g); Bv = B.reshape(-1).contiguous()
            ours = lambda n: (lambda: so.spmm_csr_float_v2([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv, 256, n, True))
            mkl = lambda: torch.sparse.mm(A, B)
            res = {}
            for n in (P, 1):
                res["ours_n%d" % n] = tmin(ours(n)); res["ours_n%d_cold" % n] = tcold(ours(n), Bv)
            for n in (P, 1):
                torch.set_num_threads(n)
                res["mkl_n%d" % n] = tmin(mkl); res["mkl_n%d_cold" % n] = tcold(mkl, Bv)
            torch.set_num_threads(P)
            for a, v in res.items(): w.writerow([corpus, key, k, M, nnz, a, "%.3f" % v])
    print(corpus, "done", flush=True)
