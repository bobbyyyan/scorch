#!/usr/bin/env python3
"""Does adjacent-row column overlap predict whether column-panel blocking helps?

  probe_overlap_vs_outcome.py <grid_dir> [group ...]

The proposal (ADAPTIVE_SPMM_TILING.md §9.7) is a third gate condition: only tile when
consecutive rows do NOT already share columns, because when they do, the row-at-a-time
kernel already finds the B rows in cache and tile-j pays P-fold output re-traffic to
recover reuse it never gains. It was backed by nine matrices on one host, against
outcomes measured earlier — not enough to change a gate on.

This gets a much larger validation set for free. In the grid CSVs, the `balanced` level
probes every candidate INCLUDING v2 and reports which it chose: its route is therefore
the measured verdict for that cell, on the same machine, in the same run. So join the
route against the feature and see whether a threshold separates them.

Every feature here is computed from a bounded sample, at the same cost class as the
samplers already in tiling.py — nothing needs a tocsc() or a full scan.
"""
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase0_attrib as P  # noqa: E402
import scorch  # noqa: E402
from scorch import tiling  # noqa: E402


def adjacent_overlap(csr, nsamp=512, seed=0):
    """Mean |cols(i) ∩ cols(i+1)| / min(deg_i, deg_{i+1}) over sampled row pairs."""
    M = csr.shape[0]
    if M < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(M - 1, min(nsamp, M - 1), replace=False)
    ip, ind = csr.indptr, csr.indices
    vals = []
    for i in idx:
        a = ind[ip[i]:ip[i + 1]]
        b = ind[ip[i + 1]:ip[i + 2]]
        if a.size == 0 or b.size == 0:
            continue
        vals.append(np.intersect1d(a, b).size / min(a.size, b.size))
    return float(np.mean(vals)) if vals else 0.0


def window_unique(csr, W=64, nsamp=128, seed=0):
    """Distinct columns / total nonzeros over W consecutive rows. 1.0 means the row
    sweep gets no reuse at all; low means it already reuses B rows."""
    M = csr.shape[0]
    if M <= W:
        W = max(2, M // 2)
    rng = np.random.default_rng(seed)
    starts = rng.choice(max(1, M - W), min(nsamp, max(1, M - W)), replace=False)
    ip, ind = csr.indptr, csr.indices
    vals = []
    for s in starts:
        seg = ind[ip[s]:ip[s + W]]
        if seg.size:
            vals.append(np.unique(seg).size / seg.size)
    return float(np.mean(vals)) if vals else 0.0


def main():
    grid_dir = sys.argv[1]
    groups = sys.argv[2:] or ["main", "ss-tiling", "ss-quick", "wide"]

    rows = []
    for g in groups:
        path = os.path.join(grid_dir, f"cand_{g}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                r["group"] = g
                rows.append(r)
    if not rows:
        raise SystemExit("no candidate CSVs found")

    by = defaultdict(dict)
    for r in rows:
        by[(r["group"], r["matrix"], int(r["N"]))][r["arm"]] = r

    # Only cells where blocking was even considered carry information about the gate.
    cells = []
    for key, arms in by.items():
        bal = arms.get("sc_balanced")
        off = arms.get("sc_off")
        if not bal or not off:
            continue
        try:
            t_bal, t_off = float(bal["ms"]), float(off["ms"])
        except ValueError:
            continue
        route = bal.get("route") or "v2"
        cells.append((key, route, t_off / t_bal, bal, off))

    feats = {}
    print(f"{'matrix':22s} {'deg/col':>8s} {'span':>6s} {'overlap':>8s} {'win64':>6s}")
    print("-" * 56)
    for mat in sorted({k[1] for (k, _, _, _, _) in cells}):
        try:
            csr = P.load_matrix(mat)
        except SystemExit:
            continue
        J = csr.shape[1]
        st = P.H.to_st(csr)
        feats[mat] = dict(
            deg=csr.nnz / max(1, J),
            span=tiling._locality_ratio(st, J),
            overlap=adjacent_overlap(csr),
            win64=window_unique(csr),
        )
        f = feats[mat]
        print(f"{mat:22s} {f['deg']:8.1f} {f['span']:6.3f} {f['overlap']:8.3f} "
              f"{f['win64']:6.3f}")
        del csr, st

    print("\n## cells where `balanced` chose a tiled route (tiling measurably won)")
    print(f"{'cell':24s} {'route':>10s} {'off/bal':>8s} {'overlap':>8s} {'span':>6s} "
          f"{'elig':>5s} {'scat':>5s}")
    tiled, untiled = [], []
    for (key, route, gain, bal, off) in sorted(cells, key=lambda c: c[0]):
        mat = key[1]
        if mat not in feats:
            continue
        ov = feats[mat]["overlap"]
        rec = (gain, ov, key, route)
        if route and route not in ("v2", ""):
            tiled.append(rec)
            print(f"{mat + '@' + str(key[2]):24s} {route:>10s} {gain:8.3f} {ov:8.3f} "
                  f"{feats[mat]['span']:6.3f} {bal.get('eligible',''):>5s} "
                  f"{bal.get('scattered',''):>5s}")
        else:
            untiled.append(rec)

    print(f"\ntiled cells: {len(tiled)}   v2 cells: {len(untiled)}")
    if tiled:
        print(f"overlap on tiled-route cells:  min {min(t[1] for t in tiled):.3f} "
              f"max {max(t[1] for t in tiled):.3f}")
    # Only v2 cells that were ELIGIBLE tell us anything: an ineligible cell never
    # reached the decision, so it is not evidence about the third condition.
    elig_v2 = [u for u in untiled
               if by[u[2]].get("sc_balanced", {}).get("eligible") in ("True", "true", "1")]
    if elig_v2:
        print(f"overlap on eligible-but-v2 cells: min {min(u[1] for u in elig_v2):.3f} "
              f"max {max(u[1] for u in elig_v2):.3f}  (n={len(elig_v2)})")

    print("\n## threshold sweep: would `overlap < THR` reproduce the measured routes?")
    print(f"{'THR':>5s} {'tiled kept':>11s} {'v2 correctly rejected':>22s} {'agree':>7s}")
    pool = [(g, ov, True) for (g, ov, _, _) in tiled] + \
           [(g, ov, False) for (g, ov, _, _) in elig_v2]
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        keep = sum(1 for (_, ov, t) in pool if t and ov < thr)
        rej = sum(1 for (_, ov, t) in pool if not t and ov >= thr)
        n_t = sum(1 for (_, _, t) in pool if t)
        n_v = len(pool) - n_t
        agree = (keep + rej) / len(pool) if pool else float("nan")
        print(f"{thr:5.1f} {keep:5d}/{n_t:<5d} {rej:11d}/{n_v:<10d} {agree:7.3f}")


if __name__ == "__main__":
    main()
