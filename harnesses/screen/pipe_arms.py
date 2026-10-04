#!/usr/bin/env python3
"""Per-cell base vs candidate from two pipe_screen.py CSVs, isolated (ours) and in-pipeline (ours_pipe).
r = base_us / cand_us (>1 = candidate faster). usage: pipe_arms.py <base.csv> <cand.csv> [show]"""
import csv, math, sys
def load(p):
    t = {}
    for r in csv.DictReader(open(p)):
        t.setdefault((r["corpus"], r["key"], int(r["k"])), dict(rows=int(r["rows"]), nnz=int(r["nnz"])))[r["arm"]] = float(r["us"])
    return t
b, c = load(sys.argv[1]), load(sys.argv[2]); show = int(sys.argv[3]) if len(sys.argv) > 3 else 8
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
for arm in [a for a in ("ours", "ours_pipe", "ours_cold") if any(a in v for v in b.values())]:
    for corp in ("ss", "dlmc"):
        for k in sorted({k for (cc, _, k) in b if cc == corp}):
            cells = [(b[x][arm] / c[x][arm], x) for x in b if x[0] == corp and x[2] == k and x in c and b[x][arm] > 0.5 and c[x][arm] > 0.5]
            r = [v for v, _ in cells]
            print("%-9s %-4s k=%-2d gmean %.3f  slower>5%% %3d  slower>10%% %3d  min %.3f" % (arm, corp, k, gm(r), sum(v < 0.95 for v in r), sum(v < 0.90 for v in r), min(r)))
worst = sorted((b[x]["ours"] / c[x]["ours"], x) for x in b if x in c)[:show]
print("worst cells (isolated):")
for v, x in worst:
    print("  %.3f  %-4s k=%-2d rows %8d nnz %9d  base %8.1f us cand %8.1f us  %s" % (v, x[0], x[2], b[x]["rows"], b[x]["nnz"], b[x]["ours"], c[x]["ours"], x[1][-40:]))
