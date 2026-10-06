#!/usr/bin/env python3
"""Where a nearly empty product's time goes: fewer nonzeros than rows, so the output and the row
loop dominate and the arithmetic does not.

Run once per pool size (OMP_NUM_THREADS sets it at start-up; changing it in-process would resize
libgomp's pool between arms). Per cell, best of 5 back-to-back batches:

  api      scorch.matmul(STensor, B): the drop-in path, plan cache live
  direct   spmm_csr_float_v2 / _double_v2 on the same CSR, production launch (atparallel)
  mkl      torch.sparse.mm on the same CSR
  zero     torch.zeros(rows, k): the output's allocation and zero fill alone, a floor both pay

the worker count our resolver gives the cell, and each arm's minor page faults per call over 50
back-to-back calls (getrusage): a fresh output buffer the allocator maps per call shows here.
Pool 1 against pool P separates MKL's threading from its per-row cost; api minus direct is the
Python dispatch.

usage: near_empty.py <cache_dir> <corpus> <keys_file> <out_csv> <ks>
SCREEN_DTYPE=float64 runs double.
"""
import csv, os, resource, sys, time, zlib
import numpy as np, torch, scorch, scorch_ops as so
from scorch import STensor
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csrresolve import resolve_all
from settle import settle
TDT, NDT, KERN = {"float32": (torch.float32, np.float32, so.spmm_csr_float_v2),
                  "float64": (torch.float64, np.float64, so.spmm_csr_double_v2)}[
    os.environ.get("SCREEN_DTYPE", "float32")]
cache, corpus, keysf, out, ks = sys.argv[1:6]; KS = [int(x) for x in ks.split(",")]
P = torch.get_num_threads()


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


keys = [l.strip() for l in open(keysf) if l.strip()]
paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
settle()  # the process's first parallel work runs slow; see settle.py
w = csv.writer(open(out, "w")); w.writerow(["corpus", "key", "k", "rows", "nnz", "pool", "arm", "us"])
for key in keys:
    if key not in paths: continue
    z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
    ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
    vt = torch.from_numpy(z["data"].astype(NDT)); nnz = int(z["indptr"][-1])
    A = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
    St = STensor.from_csr(torch.sparse_csr_tensor(ipt.long(), ixt.long(), vt, size=(M, J)), "A")
    for k in KS:
        g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
        B = torch.rand((J, k), generator=g, dtype=TDT); Bv = B.reshape(-1).contiguous()
        arms = {"api": lambda: scorch.matmul(St, B),
                "direct": lambda: KERN([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv, 256, P, True),
                "mkl": lambda: torch.sparse.mm(A, B),
                "zero": lambda: torch.zeros(M, k, dtype=TDT)}
        ref = torch.sparse.mm(A, B); got = arms["api"]()
        if float((got - ref).norm()) > 1e-4 * max(float(ref.norm()), 1e-30):
            w.writerow([corpus, key, k, M, nnz, P, "mismatch", 0]); continue
        res = {a: tmin(f) for a, f in arms.items()}
        for a, f in arms.items():                    # minor page faults per call, warm
            f0 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
            for _ in range(50): f()
            res[a + "_faults"] = (resource.getrusage(resource.RUSAGE_SELF).ru_minflt - f0) / 50
        res["workers"] = so.scorch_spmm_nthreads(nnz * max(k, 16), M, P, nnz * k, nnz)
        for a, v in res.items(): w.writerow([corpus, key, k, M, nnz, P, a, "%.3f" % v])
