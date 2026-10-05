#!/usr/bin/env python3
"""Does a multi-worker SpMM call want more of the team it has already woken?

Under libgomp every call with more than one worker runs on torch's full-width team (cc1bef5): the
whole pool wakes and only the first `nthreads` members take rows. So the cost of an extra ACTIVE
worker is no longer a thread wake-up, and the worker count's row ceiling (rows/16) was chosen when
it was. This times the production launch path (atparallel, nthreads_override = the pool) at the
count the policy resolves and at the count forced to the pool, in one process, ABBA per cell:

  pol / pool   SCORCH_SPMM_NT_FORCE=0 (no force; the policy's count) / =P (the pool)

each as warm (best of 5 back-to-back batches) and cold (median of REP calls after a 256 MB sweep
and a touch of B, as pipe_cold.py). Both arms set the same variable, so they look up the same
number of names. The count each arm resolved is asked of the exported resolver, under that arm's
environment, and written down -- a cell whose two counts agree is an A/A, not a comparison.
MKL warm and cold once per cell, for the reading against MKL. Needs a build with
SCORCH_BUILD_TUNE_HOOKS=1 (the force is a hook).

usage: thread_force.py <cache_dir> <sample_dir> <out_csv> [ks]
SCREEN_DTYPE=float64 runs double.
"""
import csv, os, sys, time, zlib, numpy as np, torch, scorch_ops as so
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
DT = {"float32": (torch.float32, np.float32, so.spmm_csr_float_v2),
      "float64": (torch.float64, np.float64, so.spmm_csr_double_v2)}[os.environ.get("SCREEN_DTYPE", "float32")]
cache, sdir, out = sys.argv[1:4]; KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,2,4,8,16,32").split(",")]
P = torch.get_num_threads(); REP = 11
F = torch.rand(256 << 18)
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
    xs.sort(); return xs[REP // 2] * 1e6
ARMS = {"pol": "0", "pool": str(P)}
w = csv.writer(open(out, "w")); w.writerow(["corpus", "key", "k", "rows", "nnz", "arm", "us"])
for corpus in ("ss", "dlmc"):
    keys = [l.strip() for l in open(os.path.join(sdir, "screen_%s.txt" % corpus)) if l.strip()]
    paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
    for key in keys:
        if key not in paths: w.writerow([corpus, key, 0, 0, 0, "unresolved", 0]); continue
        z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
        ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
        vt = torch.from_numpy(z["data"].astype(DT[1])); nnz = int(z["indptr"][-1])
        A = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
        for k in KS:
            g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
            B = torch.rand((J, k), generator=g, dtype=DT[0]); Bv = B.reshape(-1).contiguous()
            ours = lambda: DT[2]([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv, 256, P, True)
            mkl = lambda: torch.sparse.mm(A, B)
            ref = mkl(); res = {}
            for i, arm in enumerate(("pol", "pool", "pool", "pol")):
                os.environ["SCORCH_SPMM_NT_FORCE"] = ARMS[arm]
                if i < 2:
                    res["n_" + arm] = so.scorch_spmm_nthreads(nnz * max(k, 16), M, P, nnz * k, nnz)
                    got = ours().storage.value.view(M, k)
                    if float((got - ref).norm()) > 1e-3 * max(float(ref.norm()), 1e-30): res["mismatch_" + arm] = 1
                tag = "%s%d" % (arm, 1 if i < 2 else 2)
                res[tag] = tmin(ours); res[tag + "_cold"] = tcold(ours, Bv)
            os.environ["SCORCH_SPMM_NT_FORCE"] = "0"
            res["mkl"] = tmin(mkl); res["mkl_cold"] = tcold(mkl, Bv)
            for a, v in res.items(): w.writerow([corpus, key, k, M, nnz, a, "%.3f" % v])
    print(corpus, "done", flush=True)
