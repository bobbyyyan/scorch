#!/usr/bin/env python3
"""Stealing on vs off in ONE binary (runtime switch in a probe build), so both arms run the same
machine code. Phase 1 times every sample cell warm (back-to-back calls) and cold (A evicted,
B touched), the two arms interleaved inside one process. Phase 2 traces chosen cells: per call
and per worker, when it entered, finished its own range, exited, and how many rows it stole.

usage: probe_steal.py <cache_dir> <sample_dir> <out_prefix> [ks] [trace_keys_file]
"""
import csv, os, sys, time, zlib, statistics as stt
import numpy as np, torch, scorch_ops as so
sys.path.insert(0, "/matx/u/bobbyy/pfsstage")
from csrresolve import resolve_all
cache, sdir, outp = sys.argv[1:4]; KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "2,3,4,8,16,32").split(",")]
TRACE = set(l.strip() for l in open(sys.argv[5])) if len(sys.argv) > 5 and sys.argv[5] != "-" else set()
ARMS = [tuple(map(int, a.split(":"))) for a in (sys.argv[6] if len(sys.argv) > 6 else "1:0,0:0").split(",")]
import hashlib
NBAD = 0
P = torch.get_num_threads(); F = torch.rand(256 << 18); REP = 21
def warm(fn, arm):
    so.probe_set(arm[0], 0, arm[1]); fn(); fn(); c = []
    for _ in range(3):
        t0 = time.perf_counter(); fn(); c.append(time.perf_counter() - t0)
    return max(2, int(0.003 / max(sorted(c)[1], 1e-9)))
def tbatch(fn, b):
    t0 = time.perf_counter()
    for _ in range(b): fn()
    return (time.perf_counter() - t0) / b * 1e6
w = csv.writer(open(outp + "_ab.csv", "w")); w.writerow(["corpus", "key", "k", "rows", "nnz", "arm", "us"])
tw = csv.writer(open(outp + "_trace.csv", "w"))
tw.writerow(["corpus", "key", "k", "regime", "steal", "call", "worker", "t_entry", "t_own", "t_exit", "rows_own", "rows_stolen", "steals", "scanned"])
for corpus in ("ss", "dlmc"):
    keys = [l.strip() for l in open(os.path.join(sdir, "screen_%s.txt" % corpus)) if l.strip()]
    paths, _, _ = resolve_all(os.path.join(cache, corpus), keys)
    for key in keys:
        z = np.load(paths[key]); M, J = int(z["shape"][0]), int(z["shape"][1])
        ipt = torch.from_numpy(z["indptr"].astype(np.int32)); ixt = torch.from_numpy(z["indices"].astype(np.int32))
        vt = torch.from_numpy(z["data"].astype(np.float32)); nnz = int(z["indptr"][-1])
        for k in KS:
            g = torch.Generator().manual_seed(zlib.crc32(("%s|%d" % (key, k)).encode()) % (2**31))
            Bv = torch.rand(J * k, generator=g)
            fn = lambda: so.spmm_csr_float_v2([M, k], [M, J], [[], [ipt, ixt]], vt, [J, k], [[], []], Bv, 256, P, True)
            hs = set()
            for arm in ARMS:
                so.probe_set(arm[0], 0, arm[1]); hs.add(hashlib.sha256(fn().storage.value.numpy().tobytes()).hexdigest())
            if len(hs) != 1: NBAD += 1; print("BIT MISMATCH", corpus, key, k, flush=True)
            b = warm(fn, ARMS[0]); best = {a: 1e9 for a in ARMS}
            for r in range(6):
                for arm in (ARMS if r % 2 == 0 else ARMS[::-1]):
                    so.probe_set(arm[0], 0, arm[1]); fn(); best[arm] = min(best[arm], tbatch(fn, b))
            cold = {a: [] for a in ARMS}
            for r in range(REP):
                for arm in (ARMS if r % 2 == 0 else ARMS[::-1]):
                    so.probe_set(arm[0], 0, arm[1]); float(F.sum()); Bv.mul_(1.0)
                    t0 = time.perf_counter(); fn(); cold[arm].append((time.perf_counter() - t0) * 1e6)
            for arm in ARMS:
                nm = "s%dc%d" % arm
                w.writerow([corpus, key, k, M, nnz, nm + "_warm", "%.3f" % best[arm]])
                w.writerow([corpus, key, k, M, nnz, nm + "_cold", "%.3f" % stt.median(cold[arm])])
            if "%s|%d" % (key, k) in TRACE:
                for regime in ("warm", "cold"):
                    for arm in (1, 0):
                        so.probe_set(arm, 1)
                        for _ in range(200 if regime == "warm" else 60):
                            if regime == "cold": float(F.sum()); Bv.mul_(1.0)
                            fn()
                        for rec in so.probe_dump():
                            tw.writerow([corpus, key, k, regime, arm] + ["%.0f" % rec[1], "%d" % rec[0]] + ["%.0f" % v for v in (rec[2], rec[3], rec[4])] + ["%d" % v for v in rec[5:]])
                so.probe_set(1, 0)
    print(corpus, "done; bit mismatches so far", NBAD, flush=True)
