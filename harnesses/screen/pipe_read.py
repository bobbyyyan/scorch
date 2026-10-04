#!/usr/bin/env python3
"""Summarise pipe_screen.py: share of cells behind MKL, isolated vs in-pipeline. usage: pipe_read.py <csv> [label]"""
import csv, math, sys
t = {}
for r in csv.DictReader(open(sys.argv[1])):
    t.setdefault((r["corpus"], r["key"], int(r["k"])), {})[r["arm"]] = float(r["us"])
lab = sys.argv[2] if len(sys.argv) > 2 else ""
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
for c in ("ss", "dlmc"):
    for k in sorted({k for (cc, _, k) in t if cc == c}):
        v = [x for (cc, _, kk), x in t.items() if cc == c and kk == k]
        iso = [x["mkl"] / x["ours"] for x in v]
        pipe = [max(x["mkl_pipe"], 0.5) / max(x["ours_pipe"], 0.5) for x in v]
        big = sum(x["ours_pipe"] - x["ours"] > 100 for x in v)
        print("%s %-4s k=%-2d n=%3d  isolated: gmean %.3f behind %5.1f%%  |  in-pipeline: gmean %.3f behind %5.1f%%  | ours pays >100us extra in pipeline: %d"
              % (lab, c, k, len(v), gm(iso), 100 * sum(x < 1 for x in iso) / len(iso), gm(pipe), 100 * sum(x < 1 for x in pipe) / len(pipe), big))
