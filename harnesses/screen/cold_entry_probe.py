#!/usr/bin/env python3
"""Where the production path's cold-cache tax lives. One process, one matrix at a time; every timed
call follows a flush of everything (256 MB parallel read) and a parallel touch of B, as pipe_cold.py.
Arms, interleaved per repetition so drift cancels:
  api      scorch.matmul(St, B)                          -- what a user calls
  run      plan.run(values, B, nthreads, atparallel)     -- the native plan alone, args prebound
  legacy   the nested-list pybind entry the screens time
  mkl      torch.sparse.mm(A, B)
  nop      a Python call into a trivial C function (len) -- the floor of any Python->C call cold
Each arm is also timed warm (best of 5 batches of 200, back to back) under its plain name; cold rows
are <arm>_cold. api - run is the Python dispatch in scorch.matmul; run - legacy is the plan's own native cost.

usage: cold_entry_probe.py <cache_dir> <sample_dir> <out_csv> [ks] [n_per_corpus]
"""
import csv, os, sys, time, zlib, statistics as st
import numpy as np, torch, scorch, scorch_ops as so
from scorch import STensor
from scorch import ops as sops
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
cache, sdir, out = sys.argv[1:4]; KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "2").split(",")]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 20
P = torch.get_num_threads(); REP = 21; F = torch.rand(256 << 18)
w = csv.writer(open(out, "w")); w.writerow(["corpus", "key", "k", "rows", "nnz", "arm", "us"])
for corpus in ("ss", "dlmc"):
    keys = [l.strip() for l in open(os.path.join(sdir, "screen_%s.txt" % corpus)) if l.strip()]
    paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
    done = 0
    for key in keys:
        z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1]); nnz = int(z["indptr"][-1])
        if nnz > 400000: continue
        ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
        vt = torch.from_numpy(z["data"].astype(np.float32))
        A = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
        St = STensor.from_csr(torch.sparse_csr_tensor(ipt.long(), ixt.long(), vt, size=(M, J)), "A")
        for k in KS:
            g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
            B = torch.rand((J, k), generator=g); Bv = B.reshape(-1).contiguous()
            for _ in range(3): scorch.matmul(St, B)
            plans = St.__dict__.get("_scorch_plans") or next(v for kk, v in St.__dict__.items() if "plan" in kk.lower() and isinstance(v, dict))
            plan = next(p for p in plans.values() if p is not None)
            vals, nt, atp = St._storage._value, sops._PLAN_NTHREADS(), sops._PLAN_ATPARALLEL
            arms = {"api": lambda: scorch.matmul(St, B), "run": lambda: plan.run(vals, B, nt, atp),
                    "legacy": lambda: so.spmm_csr_float_v2([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv, 256, P, True),
                    "mkl": lambda: torch.sparse.mm(A, B), "nop": lambda: len(ipt)}
            assert arms["run"]() is not None
            xs = {a: [] for a in arms}
            for _ in range(REP):
                for a, fn in arms.items():
                    float(F.sum()); Bv.mul_(1.0)
                    t0 = time.perf_counter(); fn(); xs[a].append(time.perf_counter() - t0)
            for a, v in xs.items(): w.writerow([corpus, key, k, M, nnz, a + "_cold", "%.3f" % (st.median(v) * 1e6)])
            for a, fn in arms.items():           # warm: best of 5 batches, back to back
                for _ in range(3): fn()
                best = 1e9
                for _ in range(5):
                    t0 = time.perf_counter()
                    for _ in range(200): fn()
                    best = min(best, (time.perf_counter() - t0) / 200)
                w.writerow([corpus, key, k, M, nnz, a, "%.3f" % (best * 1e6)])
        done += 1
        if done >= N: break
    print(corpus, "done", flush=True)
