#!/usr/bin/env python3
"""Is the gap to MKL a fixed per-call cost or proportional to work?

Bins cells by MKL time and prints median ours, median MKL, median (ours - MKL) in microseconds,
and median ours/MKL. A fixed per-call cost shows as a roughly constant difference in us with the
ratio shrinking toward 1 as time grows; a per-unit-of-work cost shows as a constant ratio.

usage: gap_shape.py <csv_glob_prefix> [k ...]     e.g. /scr/.../dlmc_float32_ship_s 2 8
"""
import csv, glob, math, sys
pre = sys.argv[1]; ks = [int(x) for x in sys.argv[2:]] or [2, 3, 8]
t = {}
for p in sorted(glob.glob(pre + "*.csv")):
    for r in csv.DictReader(open(p)):
        t.setdefault((r["key"], int(r["k"])), {})[r["arm"]] = (float(r["ms"]), int(r["nnz"]), int(r["rows"]))
med = lambda xs: sorted(xs)[len(xs) // 2]
for k in ks:
    cells = [(v["mkl"][0] * 1e3, v["ours"][0] * 1e3, v["ours"][1], v["ours"][2]) for (key, kk), v in t.items()
             if kk == k and "mkl" in v and "ours" in v]
    print("k=%d  n=%d   bins by MKL time (us)" % (k, len(cells)))
    print("  %-12s %5s %9s %9s %9s %7s %8s %9s" % ("mkl_us", "n", "ours_us", "mkl_us", "diff_us", "ours/mkl", "behind", "nnz_med"))
    edges = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1e9]
    for lo, hi in zip(edges, edges[1:]):
        b = [c for c in cells if lo <= c[0] < hi]
        if not b: continue
        print("  %5g-%-6g %5d %9.1f %9.1f %9.1f %7.3f %7.1f%% %9d" % (lo, hi, len(b), med([c[1] for c in b]),
              med([c[0] for c in b]), med([c[1] - c[0] for c in b]), med([c[1] / c[0] for c in b]),
              100 * sum(c[1] > c[0] for c in b) / len(b), med([c[2] for c in b])))
