#!/usr/bin/env python3
"""Cold-cache screen: what one SpMM costs inside a model, where other layers have run since the
last call on the same matrix. Before every timed call a parallel read of a buffer larger than the
last-level cache evicts A (and everything else); then B is touched in parallel, as the op that
produced it would have, so B starts warm and A starts cold. Per cell, ours (production launch
path) and MKL: median of REP such calls. Also the warm isolated time (same as pipe_screen.py).

usage: pipe_cold.py <cache_dir> <sample_dir> <out_csv> [ks] [flush_mb]
"""
import csv, os, sys, time, zlib, numpy as np, torch, scorch_ops as so
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
cache, sdir, out = sys.argv[1:4]; KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "2,3,4,8,16,32").split(",")]
FLUSH_MB = int(sys.argv[5]) if len(sys.argv) > 5 else 256
P = torch.get_num_threads(); REP = 21
F = torch.rand(FLUSH_MB << 18)          # FLUSH_MB MiB of float32
def tmin(fn, reps=5, target=0.006):
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
        float(F.sum()); Bv.mul_(1.0)    # evict, then warm B the way its producer would
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
            assert Bv.data_ptr() == B.data_ptr()
            ours = lambda: so.spmm_csr_float_v2([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv, 256, P, True)
            mkl = lambda: torch.sparse.mm(A, B)
            res = {"ours": tmin(ours), "mkl": tmin(mkl), "ours_cold": tcold(ours, Bv), "mkl_cold": tcold(mkl, Bv)}
            for a, v in res.items(): w.writerow([corpus, key, k, M, nnz, a, "%.3f" % v])
    print(corpus, "done", flush=True)
