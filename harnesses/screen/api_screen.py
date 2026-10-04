#!/usr/bin/env python3
"""The product as a user calls it: scorch.matmul(STensor, dense) with no keyword arguments, which
after its first call on an operand is served by a cached native plan (plan.py), against
torch.sparse.mm on the same CSR. Also the legacy pybind entry the other screens time, so the
difference between what they measure and what a user pays is itself measured. Per cell: warm
(best of batches, back to back) and cold (median of REP calls, each after A is evicted and B
touched in parallel, as pipe_cold.py). Every cell checks scorch.matmul's answer against MKL's.

SCREEN_DTYPE=float64 runs double. Run with PYTHONPATH=<tree>/src so `scorch` and `scorch_ops` come
from the tree under test.

usage: api_screen.py <cache_dir> <sample_dir> <out_csv> [ks]
"""
import csv, os, sys, time, zlib, statistics as st
import numpy as np, torch, scorch, scorch_ops as so
from scorch import STensor
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
DT = {"float32": (torch.float32, np.float32, so.spmm_csr_float_v2),
      "float64": (torch.float64, np.float64, so.spmm_csr_double_v2)}[os.environ.get("SCREEN_DTYPE", "float32")]
cache, sdir, out = sys.argv[1:4]; KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "2,3,4,8,16,32").split(",")]
P = torch.get_num_threads(); REP = 21; F = torch.rand(256 << 18)
print("scorch from", scorch.__file__, "| scorch_ops from", so.__file__, flush=True)
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
        float(F.sum()); Bv.mul_(1.0)
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    return st.median(xs) * 1e6
w = csv.writer(open(out, "w")); w.writerow(["corpus", "key", "k", "rows", "nnz", "arm", "us"])
bad = 0
for corpus in ("ss", "dlmc"):
    keys = [l.strip() for l in open(os.path.join(sdir, "screen_%s.txt" % corpus)) if l.strip()]
    paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
    for key in keys:
        z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
        ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
        vt = torch.from_numpy(z["data"].astype(DT[1])); nnz = int(z["indptr"][-1])
        A = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
        St = STensor.from_csr(torch.sparse_csr_tensor(ipt.long(), ixt.long(), vt, size=(M, J)), "A")
        for k in KS:
            g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
            B = torch.rand((J, k), generator=g, dtype=DT[0]); Bv = B.reshape(-1).contiguous()
            api = lambda: scorch.matmul(St, B)
            legacy = lambda: DT[2]([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv, 256, P, True)
            mkl = lambda: torch.sparse.mm(A, B)
            got, ref = api(), mkl()
            if not isinstance(got, torch.Tensor) or got.shape != ref.shape or \
               float((got - ref).norm()) > 1e-4 * max(float(ref.norm()), 1e-30):
                bad += 1; print("MISMATCH", corpus, key, k, type(got).__name__, flush=True); continue
            res = {"api": tmin(api), "legacy": tmin(legacy), "mkl": tmin(mkl),
                   "api_cold": tcold(api, Bv), "mkl_cold": tcold(mkl, Bv)}
            for a, v in res.items(): w.writerow([corpus, key, k, M, nnz, a, "%.3f" % v])
    print(corpus, "done; mismatches", bad, flush=True)
