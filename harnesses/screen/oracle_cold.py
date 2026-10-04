#!/usr/bin/env python3
"""Thread-count ceiling in the cold-cache regime (see pipe_cold.py): per cell, ours on the
production launch path with the worker count capped at n, and MKL at the pool, each timed
after A is evicted and B is touched in parallel. If the best n per cell closes the cold
deficit, the worker-count policy (calibrated in warm loops) is the lever.

usage: oracle_cold.py <cache_dir> <sample_dir> <out_csv> [ks] [flush_mb]
"""
import csv, os, sys, time, zlib
import numpy as np, torch, scorch_ops as so
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
cache, sdir, out = sys.argv[1:4]; KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "2,3,4,8").split(",")]
FLUSH_MB = int(sys.argv[5]) if len(sys.argv) > 5 else 256
POOL = torch.get_num_threads(); NS = [1, 2, 4, 8, 16, 32]; REP = 21
F = torch.rand(FLUSH_MB << 18)
def tcold(fn, Bv):
    fn(); xs = []
    for _ in range(REP):
        float(F.sum()); Bv.mul_(1.0)
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    xs.sort(); return xs[REP // 2] * 1e6
w = csv.writer(open(out, "w")); w.writerow(["corpus", "key", "k", "rows", "nnz", "arm", "us"])
for corpus in ("ss", "dlmc"):
    keys = [l.strip() for l in open(os.path.join(sdir, "screen_%s.txt" % corpus)) if l.strip()]
    paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
    for key in keys:
        z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
        ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
        vt = torch.from_numpy(z["data"].astype(np.float32)); nnz = int(z["indptr"][-1])
        A = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
        for k in KS:
            g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
            B = torch.rand((J, k), generator=g); Bv = B.reshape(-1).contiguous()
            arms = [("policy", POOL)] + [("n%d" % n, n) for n in NS]
            for name, n in arms:
                us = tcold(lambda: so.spmm_csr_float_v2([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv, 256, n, True), Bv)
                w.writerow([corpus, key, k, M, nnz, name, "%.3f" % us])
            w.writerow([corpus, key, k, M, nnz, "mkl", "%.3f" % tcold(lambda: torch.sparse.mm(A, B), Bv)])
    print(corpus, "done", flush=True)
