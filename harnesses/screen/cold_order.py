#!/usr/bin/env python3
"""Does a cold call's time depend on what ran before the 256 MB sweep?

full_api.py times, per cell, our back-to-back loop, then MKL's, then our cold calls, then MKL's,
so our cold calls always follow MKL's loop and MKL's always follow ours. cold_decomp.py times each
library's cold calls after its OWN loop, and on the same build and node it read the large-matrix
cold cells the other way round: ours 588 -> 470 us on Ge99H100 at k=1, MKL 457 -> 537. This times
each library's cold call (median of REP, each after the sweep and a touch of B) after each kind of
preceding activity, in one process:

  <lib>_after_mkl   a back-to-back loop of MKL first        (our cold arm's position on the board)
  <lib>_after_api   a back-to-back loop of scorch first     (MKL's cold arm's position on the board)
  <lib>_alternate   reps alternate: sweep, scorch; sweep, MKL; ... -- each call follows the other
                    library's cold call, the same for both

usage: cold_order.py <cache_dir> <corpus> <keys_file> <out_csv> <ks>
"""
import csv, os, sys, time, zlib, statistics as st
import numpy as np, torch, scorch
from scorch import STensor
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settle import settle
TDT, NDT = {"float32": (torch.float32, np.float32), "float64": (torch.float64, np.float64)}[
    os.environ.get("SCREEN_DTYPE", "float32")]
cache, corpus, keysf, out, ks = sys.argv[1:6]; KS = [int(x) for x in ks.split(",")]
REP = 11; F = torch.rand(256 << 18)
def loop(fn, n=60):
    for _ in range(n): fn()
def cold(fn, Bv):
    xs = []
    for _ in range(REP):
        float(F.sum()); Bv.mul_(1.0)
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    return st.median(xs) * 1e6
keys = [l.strip() for l in open(keysf) if l.strip()]
paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
settle()  # the process's first parallel work runs slow; see settle.py
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
        api = lambda: scorch.matmul(St, B)
        mkl = lambda: torch.sparse.mm(A, B)
        api(); mkl()
        res = {}
        for pre_name, pre in (("after_mkl", mkl), ("after_api", api)):
            for name, fn in (("api", api), ("mkl", mkl)):
                loop(pre); res["%s_%s" % (name, pre_name)] = cold(fn, Bv)
        xa, xm = [], []
        for _ in range(REP):
            float(F.sum()); Bv.mul_(1.0); t0 = time.perf_counter(); api(); xa.append(time.perf_counter() - t0)
            float(F.sum()); Bv.mul_(1.0); t0 = time.perf_counter(); mkl(); xm.append(time.perf_counter() - t0)
        res["api_alternate"] = st.median(xa) * 1e6; res["mkl_alternate"] = st.median(xm) * 1e6
        for a, v in res.items(): w.writerow([corpus, key, k, M, nnz, a, "%.3f" % v])
