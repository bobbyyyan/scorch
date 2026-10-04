#!/usr/bin/env python3
"""Arm vs base from a probe_steal.py *_ab.csv: per corpus/width/regime, gmean of base_us/arm_us
(>1 = arm faster) and how many cells the arm is slower by >5% and >10%.
usage: probe_ab_read.py <ab.csv> <base_arm> <arm> [<arm> ...]   (arm names like s1c0)"""
import csv, math, sys, collections
t = collections.defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    t[(r["corpus"], r["key"], int(r["k"]))][r["arm"]] = float(r["us"])
base = sys.argv[2]; gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
for arm in sys.argv[3:]:
    print("==== %s vs %s (>1 = %s faster)" % (arm, base, arm))
    for corp in ("ss", "dlmc"):
        for k in sorted({k for (c, _, k) in t if c == corp}):
            line = "%-4s k=%-2d" % (corp, k)
            for reg in ("warm", "cold"):
                r = [v[base + "_" + reg] / v[arm + "_" + reg] for (c, _, kk), v in t.items() if c == corp and kk == k]
                line += "  %s gmean %.3f slower>5%% %3d >10%% %3d min %.3f |" % (reg, gm(r), sum(x < 0.95 for x in r), sum(x < 0.90 for x in r), min(r))
            print(line)
