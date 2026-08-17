#!/usr/bin/env python3
"""base-vs-candidate table from ab.jsonl.

Per cell, per arm: median across rounds of the per-round median time. The `mkl32`
arm is byte-identical in both trees, so `mkl_base/mkl_cand` is this cell's
cross-process noise floor — a scorch change smaller than that is not a result.
"""
import json
import statistics
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
t = defaultdict(list)  # (matrix, n, arm, tree) -> [per-round median]
for r in rows:
    t[(r["matrix"], r["n"], r["arm"], r["tree"])].append(r["t_med"])


def med(*key):
    v = t.get(key)
    return statistics.median(v) if v else None


cells = sorted({(r["matrix"], r["n"]) for r in rows}, key=lambda c: (c[0], c[1]))
hdr = (f"{'cell':22s} {'mkl_ms':>9s} {'off_base':>9s} {'off_cand':>9s} {'gain':>6s} "
       f"{'bal_base':>9s} {'bal_cand':>9s} {'gain':>6s} "
       f"{'best/mkl':>9s} {'was':>7s} {'floor':>6s} {'n':>2s}")
print(hdr)
print("-" * len(hdr))

improved = []
for mat, n in cells:
    mb, mc = med(mat, n, "mkl32", "base"), med(mat, n, "mkl32", "cand")
    ob, oc = med(mat, n, "sc_off", "base"), med(mat, n, "sc_off", "cand")
    bb, bc = med(mat, n, "sc_balanced", "base"), med(mat, n, "sc_balanced", "cand")
    if None in (mb, mc, ob, oc):
        continue
    floor = abs(mb / mc - 1.0)
    mkl = min(mb, mc)
    best_cand = min(x for x in (oc, bc) if x is not None)
    best_base = min(x for x in (ob, bb) if x is not None)
    og = ob / oc
    bg = (bb / bc) if (bb and bc) else float("nan")
    print(f"{mat + '@' + str(n):22s} {mkl*1e3:9.3f} {ob*1e3:9.3f} {oc*1e3:9.3f} {og:6.2f} "
          f"{(bb or 0)*1e3:9.3f} {(bc or 0)*1e3:9.3f} {bg:6.2f} "
          f"{mkl/best_cand:9.3f} {mkl/best_base:7.3f} {floor*100:5.1f}% "
          f"{len(t[(mat, n, 'mkl32', 'base')]):2d}")
    improved.append((mat, n, mkl / best_base, mkl / best_cand, floor))

print()
gm_b = 1.0
gm_c = 1.0
for _, _, b, c, _ in improved:
    gm_b *= b
    gm_c *= c
k = len(improved)
print(f"geomean vs MKL over {k} cells:  base {gm_b ** (1/k):.3f}x  ->  cand {gm_c ** (1/k):.3f}x")
below = [(m, n, c) for m, n, _, c, _ in improved if c < 1.0]
print(f"cells still below parity: {len(below)}/{k}")
for m, n, c in sorted(below, key=lambda x: x[2]):
    print(f"   {m}@{n}: {c:.3f}x")
