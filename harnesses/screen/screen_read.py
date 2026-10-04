#!/usr/bin/env python3
"""Read a screen CSV (mrgate.py format): per collection and width, how often we lose to MKL.

ratio = mkl_ms / ours_ms (>1 = we are faster). The noise floor is the same-code pair
oursb/ours from the same process: a cell only counts as BEHIND BEYOND NOISE if mkl/ours is
below 1 by more than that width's p95 same-code spread.

usage: screen_read.py <csv> [<csv> ...]   (collection taken from the basename: dlmc* or ss*)
"""
import csv, math, sys

t = {}
for p in sys.argv[1:]:
    with open(p) as fh:
        for r in csv.DictReader(fh):
            c = "dlmc" if p.split("/")[-1].startswith("dlmc") else "ss"
            t.setdefault((c, r["key"], int(r["k"])), {})[r["arm"]] = float(r["ms"])
if not t:
    sys.exit("REFUSE: no rows read")
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
print("%-5s %3s %4s %7s %8s %8s %9s %s" % ("coll", "k", "n", "gmean", "behind", "beyond", "p95noise", "worst"))
for c in ("ss", "dlmc"):
    for k in sorted({kk for (cc, _, kk) in t if cc == c}):
        cells = [(key, v) for (cc, key, kk), v in t.items() if cc == c and kk == k
                 and all(a in v for a in ("ours", "oursb", "mkl"))]
        if not cells:
            continue
        r = [v["mkl"] / v["ours"] for _, v in cells]
        noise = sorted(abs(math.log(v["oursb"] / v["ours"])) for _, v in cells)
        p95 = noise[min(len(noise) - 1, int(0.95 * len(noise)))]
        beyond = sum(math.log(x) < -p95 for x in r)
        worst = min(zip(r, [kk for kk, _ in cells]))
        print("%-5s %3d %4d %7.3f %7.1f%% %7.1f%% %8.1f%%  %.3f %s"
              % (c, k, len(r), gm(r), 100 * sum(x < 1 for x in r) / len(r),
                 100 * beyond / len(r), 100 * (math.exp(p95) - 1), worst[0], worst[1]))
