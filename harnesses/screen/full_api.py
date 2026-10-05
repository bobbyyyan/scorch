#!/usr/bin/env python3
"""One slice of the full-collection board as a user calls it: scorch.matmul(STensor, dense), no
keyword arguments, served by the operand's plan after its first call, against torch.sparse.mm on
the same CSR, in the same process. Per cell: warm (best of 5 back-to-back batches) and cold
(median of REP calls, each after a 256 MB sweep evicts A and B is touched -- as pipe_cold.py).
Every cell checks scorch.matmul's answer against MKL's first; a mismatch is written as an arm
named "mismatch" and the cell is not timed, so a reader cannot count it as a win.

SCREEN_DTYPE=float64 runs double. Run with PYTHONPATH=<tree>/src so `scorch` and `scorch_ops`
come from the tree under test.

usage: full_api.py <cache_dir> <corpus> <keys_file> <out_csv> <ks>
"""
import csv, os, sys, time, zlib, statistics as st
import numpy as np, torch, scorch
from scorch import STensor
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
TDT, NDT = {"float32": (torch.float32, np.float32), "float64": (torch.float64, np.float64)}[
    os.environ.get("SCREEN_DTYPE", "float32")]
cache, corpus, keysf, out, ks = sys.argv[1:6]; KS = [int(x) for x in ks.split(",")]
REP = 11; F = torch.rand(256 << 18)
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
    vt = torch.from_numpy(z["data"].astype(NDT)); nnz = int(z["indptr"][-1])
    A = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
    St = STensor.from_csr(torch.sparse_csr_tensor(ipt.long(), ixt.long(), vt, size=(M, J)), "A")
    for k in KS:
        g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
        B = torch.rand((J, k), generator=g, dtype=TDT); Bv = B.reshape(-1)
        api = lambda: scorch.matmul(St, B)
        mkl = lambda: torch.sparse.mm(A, B)
        got, ref = api(), mkl()
        if not isinstance(got, torch.Tensor) or got.shape != ref.shape or \
           float((got - ref).norm()) > 1e-4 * max(float(ref.norm()), 1e-30):
            w.writerow([corpus, key, k, M, nnz, "mismatch", 0]); continue
        res = {"api": tmin(api), "mkl": tmin(mkl), "api_cold": tcold(api, Bv), "mkl_cold": tcold(mkl, Bv)}
        for a, v in res.items(): w.writerow([corpus, key, k, M, nnz, a, "%.3f" % v])
