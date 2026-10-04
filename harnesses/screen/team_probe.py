#!/usr/bin/env python3
"""Launch cost by team width: ours at N threads and MKL at N threads on band cells, one process.

Run it under different OpenMP wait policies (separate processes) to see whether the cost of a
wide team is waking sleeping workers or something the workers do once awake.

usage: team_probe.py <cache_dir> <sample_dir> <prior_csv_dir> [keys_per_corpus]
"""
import csv, glob, os, sys, time, zlib
import numpy as np, torch, scorch_ops as so
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
cache, sdir, prior = sys.argv[1:4]; per = int(sys.argv[4]) if len(sys.argv) > 4 else 6
POOL = torch.get_num_threads(); NS = [4, 8, 12, 16, 20, 24, 28, 32]
def tmin(fn, reps=7, target=0.01):
    fn(); fn(); fn(); c = []
    for _ in range(5):
        t0 = time.perf_counter(); fn(); c.append(time.perf_counter() - t0)
    b = max(1, int(target / max(sorted(c)[2], 1e-9))); best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(b): fn()
        best = min(best, (time.perf_counter() - t0) / b)
    return best * 1e6
med = lambda xs: sorted(xs)[len(xs) // 2]
for corpus in ("dlmc", "ss"):
    keys = [l.strip() for l in open(os.path.join(sdir, "screen_%s.txt" % corpus)) if l.strip()]
    mk = {}
    for p in glob.glob(os.path.join(prior, "%s_float32_ship_s*.csv" % corpus)):
        for r in csv.DictReader(open(p)):
            if r["arm"] == "mkl" and r["k"] == "2": mk[r["key"]] = float(r["ms"]) * 1e3
    band = [k for k in keys if 10 <= mk.get(k, -1) < 20]; pick = band[:: max(1, len(band) // per)][:per]
    paths, _, _ = resolve_all(os.path.join(cache, corpus), pick)
    for k in (2, 8):
        res = {}
        for key in pick:
            z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
            ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
            vt = torch.from_numpy(z["data"].astype(np.float32))
            g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
            B = torch.rand((J, k), generator=g); Bv = B.reshape(-1).contiguous(); A = torch.sparse_csr_tensor(ipt, ixt, vt, size=(M, J))
            for n in NS:
                res.setdefault("ours%d" % n, []).append(tmin(lambda: so.spmm_csr_float_v2([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv, 256, n, False)))
            for n in (16, 32):
                torch.set_num_threads(n); res.setdefault("mkl%d" % n, []).append(tmin(lambda: torch.sparse.mm(A, B)))
            torch.set_num_threads(POOL)
        print("%s k=%d n=%d " % (corpus, k, len(pick)) + " ".join("%s %.1f" % (a, med(v)) for a, v in res.items()), flush=True)
