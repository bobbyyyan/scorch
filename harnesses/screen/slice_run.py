#!/usr/bin/env python3
"""One slice of a full-collection A/B: for each key in <keys_file> and each width, our SpMM
(production launch path) warm (best of 5 back-to-back batches) and cold (median of REP calls,
each after A is evicted and B touched -- see pipe_cold.py). With --mkl, MKL the same way.

usage: slice_run.py <cache_dir> <corpus> <keys_file> <out_csv> <ks> [--mkl]
"""
import csv, os, sys, time, zlib, statistics as st
import numpy as np, torch, scorch_ops as so
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
cache, corpus, keysf, out, ks = sys.argv[1:6]; KS = [int(x) for x in ks.split(",")]; MKL = "--mkl" in sys.argv
P = torch.get_num_threads(); REP = 11; F = torch.rand(256 << 18)
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
keys = [l.strip() for l in open(keysf) if l.strip()]
paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
w = csv.writer(open(out, "w")); w.writerow(["corpus", "key", "k", "rows", "nnz", "arm", "us"])
for key in keys:
    if key not in paths: w.writerow([corpus, key, 0, 0, 0, "unresolved", 0]); continue
    z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
    ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
    vt = torch.from_numpy(z["data"].astype(np.float32)); nnz = int(z["indptr"][-1])
    A = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J)) if MKL else None
    for k in KS:
        g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
        B = torch.rand((J, k), generator=g); Bv = B.reshape(-1).contiguous()
        ours = lambda: so.spmm_csr_float_v2([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv, 256, P, True)
        res = {"ours": tmin(ours), "ours_cold": tcold(ours, Bv)}
        if MKL:
            mkl = lambda: torch.sparse.mm(A, B)
            res["mkl"] = tmin(mkl); res["mkl_cold"] = tcold(mkl, Bv)
        for a, v in res.items(): w.writerow([corpus, key, k, M, nnz, a, "%.3f" % v])
