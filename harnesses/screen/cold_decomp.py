#!/usr/bin/env python3
"""Split a cold call's extra time into cold DATA and cold CODE, for scorch.matmul and for MKL.

full_api.py's cold arm sweeps 256 MB before every call, which evicts the matrix, B, the output's
pages -- and every instruction and every runtime structure the call path touches. On a small
matrix the arithmetic is a few microseconds, so what a cold call costs is mostly the second kind.
Per cell, each arm is timed four ways, each the median of REP calls:

  warm       back to back, nothing in between
  cold       256 MB sweep, touch B, call                         (as full_api.py)
  data_cold  256 MB sweep, one call on an UNRELATED small matrix through the same entry point
             and width, touch B, call -- the code path, the thread team and the allocator are
             warm again, the matrix is not
  a_warm     256 MB sweep, one call on THIS product (A and the code come back into whatever cache
             holds them), touch B, call

so cold - data_cold is what cold code and runtime state cost the call, data_cold - a_warm is what
reading A from memory costs it, and a_warm - warm is what B and the output cost after a touch.
The unrelated matrix is a fixed 64 x 64 CSR with 4 nonzeros a row: the same dispatch, a
different plan, nothing of A's in cache.

usage: cold_decomp.py <cache_dir> <corpus> <keys_file> <out_csv> <ks>
"""
import csv, os, sys, time, zlib, statistics as st
import numpy as np, torch, scorch
from scorch import STensor
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
TDT, NDT = {"float32": (torch.float32, np.float32), "float64": (torch.float64, np.float64)}[
    os.environ.get("SCREEN_DTYPE", "float32")]
cache, corpus, keysf, out, ks = sys.argv[1:6]; KS = [int(x) for x in ks.split(",")]
REP = 15; F = torch.rand(256 << 18)
g0 = torch.Generator().manual_seed(7)
dm = torch.zeros(64, 64, dtype=TDT)
for i in range(64):
    dm[i, torch.randperm(64, generator=g0)[:4]] = 1.0
dcsr = dm.to_sparse_csr()
D_mkl = torch.sparse_csr_tensor(dcsr.crow_indices().int(), dcsr.col_indices().int(), dcsr.values(), size=(64, 64))
D_api = STensor.from_csr(dcsr, "D")
def med(fn, pre):
    fn(); xs = []
    for _ in range(REP):
        pre(); t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    return st.median(xs) * 1e6
keys = [l.strip() for l in open(keysf) if l.strip()]
paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
w = csv.writer(open(out, "w")); w.writerow(["corpus", "key", "k", "rows", "nnz", "arm", "us"])
for key in keys:
    if key not in paths: continue
    z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
    ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
    vt = torch.from_numpy(z["data"].astype(NDT)); nnz = int(z["indptr"][-1])
    A = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
    St = STensor.from_csr(torch.sparse_csr_tensor(ipt.long(), ixt.long(), vt, size=(M, J)), "A")
    for k in KS:
        g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
        B = torch.rand((J, k), generator=g, dtype=TDT); Bv = B.reshape(-1)
        Bd = torch.rand((64, k), dtype=TDT)
        for name, fn, dfn in (("api", lambda: scorch.matmul(St, B), lambda: scorch.matmul(D_api, Bd)),
                              ("mkl", lambda: torch.sparse.mm(A, B), lambda: torch.sparse.mm(D_mkl, Bd))):
            dfn(); dfn(); dfn()
            res = {"warm": med(fn, lambda: None),
                   "cold": med(fn, lambda: (float(F.sum()), Bv.mul_(1.0))),
                   "data_cold": med(fn, lambda: (float(F.sum()), dfn(), Bv.mul_(1.0))),
                   "a_warm": med(fn, lambda: (float(F.sum()), fn(), Bv.mul_(1.0)))}
            for r, v in res.items(): w.writerow([corpus, key, k, M, nnz, "%s_%s" % (name, r), "%.3f" % v])
