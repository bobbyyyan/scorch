#!/usr/bin/env python3
"""Where the per-call cost of stealing sits, warm: Python-timed calls paired with the probe
build's records of the same calls -- row-split and cursor setup, setup-to-launch, the parallel
region, and its join -- for stealing on and off in blocks alternating inside one process.
usage: probe_setup.py <cache_dir> <trace_keys_file> [blocks] [calls_per_block]"""
import os, sys, time, zlib, statistics as st, numpy as np, torch, scorch_ops as so
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
cache, keysf = sys.argv[1:3]; NB = int(sys.argv[3]) if len(sys.argv) > 3 else 8; NC = int(sys.argv[4]) if len(sys.argv) > 4 else 60
ARMS = [tuple(map(int, a.split(":"))) for a in (sys.argv[5] if len(sys.argv) > 5 else "1:0,0:0").split(",")]
P = torch.get_num_threads()
cells = [l.strip().rsplit("|", 1) for l in open(keysf) if l.strip()]
print("%-34s %3s %4s | %7s %7s | %6s %7s %7s %6s %7s | %s" % ("cell", "k", "arm", "py_mean", "py_med", "setup", "prelnch", "region", "join", "outside", "outside = py - setup - prelaunch - region"))
for key, k in cells:
    k = int(k); corpus = "dlmc" if key.count("/") >= 3 and not key.startswith(("LAW", "SNAP", "Pajek", "QY", "Rajat")) else "ss"
    paths, _, _ = resolve_all(os.path.join(cache, corpus), [key])
    z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
    ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
    vt = torch.from_numpy(z["data"].astype(np.float32))
    Bv = torch.rand(J * k, generator=torch.Generator().manual_seed(zlib.crc32(key.encode()) % (2**31)))
    fn = lambda: so.spmm_csr_float_v2([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv, 256, P, True)
    acc = {a: [] for a in ARMS}
    for blk in range(NB):
        for arm in (ARMS if blk % 2 == 0 else ARMS[::-1]):
            so.probe_set(arm[0], 1, arm[1]); fn(); so.probe_set(arm[0], 1, arm[1])
            py = []
            for _ in range(NC):
                t0 = time.perf_counter(); fn(); py.append((time.perf_counter() - t0) * 1e6)
            d = so.probe_dump()
            calls = {int(r[1]): r for r in d if r[0] == -1}
            exits = {}
            for r in d:
                if r[0] >= 0: exits[int(r[1])] = max(exits.get(int(r[1]), 0.0), r[4])
            for i, u in enumerate(py):
                c = calls[i]
                acc[arm].append((u, c[5] / 1e3, c[6] / 1e3, c[2] / 1e3, (c[2] - exits[i]) / 1e3))
    for arm in ARMS:
        a = acc[arm]; m = lambda j: st.mean(x[j] for x in a)
        print("%-34s %3d %4s | %7.2f %7.2f | %6.2f %7.2f %7.2f %6.2f %7.2f |" % (key[-34:], k, "%d:%d" % arm, m(0), st.median(x[0] for x in a),
              m(1), m(2), m(3), m(4), m(0) - m(1) - m(2) - m(3)))
