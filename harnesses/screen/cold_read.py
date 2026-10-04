#!/usr/bin/env python3
"""Behind-MKL share per width, warm (isolated loop) vs cold (A evicted before every call).
usage: cold_read.py <csv> <label>"""
import csv, math, sys
t = {}
for r in csv.DictReader(open(sys.argv[1])):
    t.setdefault((r["corpus"], r["key"], int(r["k"])), {})[r["arm"]] = float(r["us"])
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
for corp in ("ss", "dlmc"):
    for k in sorted({k for (c, _, k) in t if c == corp}):
        cells = [v for (c, _, kk), v in t.items() if c == corp and kk == k]
        w = [v["mkl"] / v["ours"] for v in cells]; c = [v["mkl_cold"] / v["ours_cold"] for v in cells]
        print("%-6s %-4s k=%-2d n=%3d  warm: gmean %.3f behind %5.1f%%  |  cold: gmean %.3f behind %5.1f%%  | cold/warm time ours %.2f mkl %.2f" % (
            sys.argv[2], corp, k, len(cells), gm(w), 100.0 * sum(x < 1 for x in w) / len(w), gm(c), 100.0 * sum(x < 1 for x in c) / len(c),
            gm([v["ours_cold"] / v["ours"] for v in cells]), gm([v["mkl_cold"] / v["mkl"] for v in cells])))
