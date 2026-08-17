#!/usr/bin/env python3
"""Roofline attribution from phase0.jsonl.

Machine ceilings measured on this host with bwcal (32 threads, best of 5):
  pure read            55.3 GB/s
  copy  (2R+1W)        69.7 GB/s
  triad (3R+1W)        66.1 GB/s
  random 64B-line gather 50.1 GB/s
So ~50 GB/s is the right ceiling for a scattered-gather kernel and ~66-70 GB/s for
streaming. `r412e` is LONGEST_LAT_CACHE.MISS: every core-originated request that
misses L3, hardware prefetches included. It counts line FILLS, not writebacks, so
measured bytes = misses*64 is DRAM *read* traffic; C's writeback is added by model.
"""
import json
import sys
from collections import defaultdict

LINE = 64.0
BW_GATHER = 50.1e9
BW_STREAM = 66.1e9

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = defaultdict(dict)
for r in rows:
    by[(r["matrix"], r["n"])][r["arm"]] = r

hdr = (f"{'cell':26s} {'arm':11s} {'ms':>9s} {'thr':>3s} "
       f"{'fill_GB':>8s} {'compul_GB':>9s} {'ampl':>6s} {'GB/s':>6s} {'%peak':>6s} "
       f"{'IPC':>5s} {'cyc/nnz':>8s} {'walks/nnz':>9s} {'MLP':>5s} {'vs_mkl':>7s}")
print(hdr)
print("-" * len(hdr))

for (mat, n) in sorted(by, key=lambda k: (k[0], k[1])):
    arms = by[(mat, n)]
    mkl = arms.get("mkl32")
    for arm in ("mkl32", "sc_off", "sc_analytic", "sc_balanced"):
        r = arms.get(arm)
        if not r:
            continue
        c = r["counters"]
        reps = r["reps"]
        fill = c["r412e"] * LINE / reps
        compul = r["compulsory_read"]
        # C writeback is not a counted fill; add it to get total DRAM bytes moved.
        total = fill + r["c_bytes"]
        t = r["t_med"]
        gbs = total / t / 1e9
        cyc = c["cycles"] / reps
        ipc = c["instructions"] / c["cycles"]
        walks = c["r0e12"] / reps / r["nnz"]
        mlp = c["r0820"] / c["cycles"]
        vs = (mkl["t_med"] / t) if mkl else float("nan")
        print(f"{mat + '@' + str(n):26s} {arm:11s} {t*1e3:9.3f} {r['threads']:3d} "
              f"{fill/1e9:8.3f} {compul/1e9:9.3f} {fill/compul:6.2f} {gbs:6.1f} "
              f"{100*total/t/BW_GATHER:6.1f} {ipc:5.2f} "
              f"{cyc/r['nnz']:8.2f} {walks:9.4f} {mlp:5.2f} {vs:7.3f}")
    print()
