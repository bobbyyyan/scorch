#!/usr/bin/env python3
"""Summarise a kernel-variant sweep from spmm_micro output.

  analyze_micro.py <micro.txt> [--min-gain 1.02]

A row-kernel change ships only if it is neutral-or-better across the whole grid, so
what matters is not the mean speedup but the WORST cell — and whether that worst cell
is outside its own A/A noise floor. Each cell's floor comes from the `aa` arm, which
is `base` entered under a second name.
"""
import re
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
cells = []          # (matrix, N, {variant: ms}, floor)
cur = None
vals = {}
for line in open(path):
    m = re.match(r"^### (\S+) N=(\d+) threads=(\d+)", line)
    if m:
        if cur and vals:
            cells.append((cur, dict(vals)))
        cur, vals = (m.group(1), int(m.group(2)), int(m.group(3))), {}
        continue
    m = re.match(r"^MICRO\s+(\S+)\s+med_ms=\s*([0-9.]+)\s+vs_base=\s*([0-9.]+)\s+relerr=(\S+)", line)
    if m and cur:
        vals[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
if cur and vals:
    cells.append((cur, dict(vals)))

if not cells:
    raise SystemExit("no cells parsed")

variants = [v for v in cells[0][1] if v not in ("base",)]
print(f"{len(cells)} cells parsed from {path}\n")

# per-variant: geomean, worst cell, count of cells outside the floor in each direction
hdr = (f"{'variant':10s} {'geomean':>8s} {'min':>7s} {'max':>7s} "
       f"{'wins>floor':>11s} {'loss>floor':>11s} {'worst cell':>26s}")
print(hdr)
print("-" * len(hdr))
summary = {}
for v in variants:
    gains, wins, losses, worst, worstcell = [], 0, 0, 9e9, None
    for (key, d) in cells:
        if v not in d or "base" not in d or "aa" not in d:
            continue
        g = d[v][1]                       # vs_base as reported by the harness
        floor = abs(d["aa"][1] - 1.0)      # |aa/base - 1| for this cell
        gains.append(g)
        if g > 1.0 + max(floor, 0.005):
            wins += 1
        elif g < 1.0 - max(floor, 0.005):
            losses += 1
        if g < worst:
            worst, worstcell = g, key
    if not gains:
        continue
    gm = statistics.geometric_mean(gains)
    summary[v] = (gm, min(gains), max(gains), wins, losses, worstcell)
    wc = f"{worstcell[0]}@{worstcell[1]}" if worstcell else "-"
    print(f"{v:10s} {gm:8.4f} {min(gains):7.4f} {max(gains):7.4f} "
          f"{wins:11d} {losses:11d} {wc:>26s}")

print("\n## the A/A floor itself, so the thresholds above are legible")
floors = [abs(d['aa'][1] - 1.0) for (_, d) in cells if 'aa' in d]
floors.sort()
print(f"  median {statistics.median(floors)*100:.2f}%  "
      f"p90 {floors[int(0.9*len(floors))]*100:.2f}%  max {floors[-1]*100:.2f}%")

# The candidate has to be neutral-or-better EVERYWHERE, so print its losing cells.
for v in ("nm_d16", "d16T0", "nomask", "nopf"):
    if v not in summary:
        continue
    bad = []
    for (key, d) in cells:
        if v not in d or "aa" not in d:
            continue
        floor = abs(d["aa"][1] - 1.0)
        if d[v][1] < 1.0 - max(floor, 0.005):
            bad.append((d[v][1], key, floor))
    print(f"\n## {v}: cells below 1.0 by more than their floor ({len(bad)})")
    for g, key, floor in sorted(bad)[:14]:
        print(f"   {g:6.4f}  {key[0]}@{key[1]}  (floor {floor*100:.2f}%)")

maxrel = max(d[v][2] for (_, d) in cells for v in d)
print(f"\nmax relative error over every variant and cell: {maxrel:.2e}")
