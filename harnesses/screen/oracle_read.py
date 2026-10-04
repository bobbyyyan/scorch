#!/usr/bin/env python3
"""Summarise oracle_threads.py output. usage: oracle_read.py <csv>"""
import csv, math, sys
t = {}
for r in csv.DictReader(open(sys.argv[1])):
    t.setdefault((r["corpus"], r["key"], int(r["k"])), {})[r["arm"]] = float(r["us"])
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
NS = ["n1", "n2", "n4", "n8", "n16", "n32"]
for c in ("ss", "dlmc"):
    for k in sorted({k for (cc, _, k) in t if cc == c}):
        cells = [v for (cc, _, kk), v in t.items() if cc == c and kk == k]
        pol = [v["mkl"] / v["policy"] for v in cells]
        best = [v["mkl"] / min(v[n] for n in NS) for v in cells]
        wins = {n: sum(min(NS, key=lambda m: v[m]) == n for v in cells) for n in NS}
        print("%-4s k=%d n=%3d  policy: gmean %.3f behind %5.1f%%  |  best width: gmean %.3f behind %5.1f%%  | best width counts %s"
              % (c, k, len(cells), gm(pol), 100 * sum(x < 1 for x in pol) / len(pol), gm(best),
                 100 * sum(x < 1 for x in best) / len(best), " ".join("%s:%d" % kv for kv in wins.items())))
