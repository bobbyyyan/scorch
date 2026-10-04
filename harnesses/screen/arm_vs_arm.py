#!/usr/bin/env python3
"""Per-cell candidate vs base on our own kernel, from two screen CSVs (mrgate format).

r = base_ours_ms / cand_ours_ms (>1 = candidate faster). The noise floor per width is each
arm's own same-process repeat (oursb/ours); a cell is SLOWER BEYOND NOISE when log r is below
minus the larger of the two arms' p95 same-code spreads at that width.

usage: arm_vs_arm.py <base.csv> <cand.csv> [label]
"""
import csv, math, sys
def load(p):
    t = {}
    for r in csv.DictReader(open(p)):
        t.setdefault((r["key"], int(r["k"])), {})[r["arm"]] = float(r["ms"])
    return t
b, c = load(sys.argv[1]), load(sys.argv[2]); lab = sys.argv[3] if len(sys.argv) > 3 else ""
def p95(t, k):
    xs = sorted(abs(math.log(v["oursb"] / v["ours"])) for (kk, w), v in t.items() if w == k and "oursb" in v)
    return xs[int(0.95 * (len(xs) - 1))]
for k in sorted({w for _, w in b}):
    cells = [(key, b[(key, w)]["ours"] / c[(key, w)]["ours"]) for (key, w) in b if w == k and (key, w) in c]
    fl = max(p95(b, k), p95(c, k))
    r = [x for _, x in cells]; g = math.exp(sum(map(math.log, r)) / len(r))
    slow = [(x, key) for key, x in cells if math.log(x) < -fl]
    print("%s k=%-2d n=%3d gmean %.3f  min %.3f  slower %d  slower-beyond-noise(p95 %.1f%%) %d %s"
          % (lab, k, len(r), g, min(r), sum(x < 1 for x in r), 100 * (math.exp(fl) - 1), len(slow),
             " ".join("%s:%.3f" % (key.split("/")[-1][:24], x) for x, key in sorted(slow)[:4])))
