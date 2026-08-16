#!/usr/bin/env python3
"""Per-cell before/after table from the bench_spmm_vs_mkl CSVs of two trees.

  analyze_grid.py <out_dir> [group ...]

Reads <out_dir>/base_<group>.csv and <out_dir>/cand_<group>.csv. Every row is one
(matrix, N, arm); the arms are the two MKL index widths, ATen's COO SpMM, scorch at
each autotune level, and an A/A control that reruns level `off` under another name.

Three things this reports that a single geomean cannot:

* per-cell vs-MKL before and after, against the FASTER MKL arm, with that cell's own
  A/A noise floor beside it;
* every (level, cell) pair where the selector is slower than plain `off` by more than
  the floor, split by whether it actually routed to a tiled kernel or ran `off`'s own
  machine code (the latter is dispatch noise, not a scheduling mistake);
* for cells still below parity, how much of the gap is the native kernel and how much
  is fixed per-call overhead, from the harness's kernel-only timing.
"""
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

LEVELS = ("off", "analytic", "balanced", "max", "learned")


def load(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["matrix"], int(r["N"]), r["arm"])
            out[key] = r
    return out


def fnum(r, field):
    if r is None:
        return None
    v = r.get(field, "")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def analyse(out_dir, groups):
    grand = defaultdict(lambda: defaultdict(list))
    regressions = []
    below = []
    relerrs = []

    for group in groups:
        base = load(os.path.join(out_dir, f"base_{group}.csv"))
        cand = load(os.path.join(out_dir, f"cand_{group}.csv"))
        if not base or not cand:
            print(f"\n## {group}: missing ({'base' if not base else 'cand'} csv absent)")
            continue
        cells = sorted({(m, n) for (m, n, a) in base}, key=lambda c: (c[0], c[1]))

        print(f"\n## {group} — {len(cells)} cells")
        hdr = (f"{'cell':24s} {'mkl_ms':>10s} | {'base_ms':>9s} {'b/mkl':>6s} | "
               f"{'cand_ms':>9s} {'c/mkl':>6s} {'gain':>6s} {'route':>10s} | "
               f"{'floor':>6s} {'relerr':>8s}")
        print(hdr)
        print("-" * len(hdr))

        for (mat, n) in cells:
            def arm(tbl, a):
                return tbl.get((mat, n, a))

            mkl = min(
                [x for x in (fnum(arm(base, "mkl_csr32"), "ms"),
                             fnum(arm(base, "mkl_csr"), "ms"),
                             fnum(arm(cand, "mkl_csr32"), "ms"),
                             fnum(arm(cand, "mkl_csr"), "ms")) if x] or [None])
            if not mkl:
                continue
            # cross-tree floor from the byte-identical MKL arm, plus the in-process A/A
            f_cross = None
            b32, c32 = fnum(arm(base, "mkl_csr32"), "ms"), fnum(arm(cand, "mkl_csr32"), "ms")
            if b32 and c32:
                f_cross = abs(b32 / c32 - 1.0)
            f_aa = fnum(arm(cand, "aa_control"), "noise_floor") or 0.0
            floor = max(f_cross or 0.0, f_aa)

            def best(tbl):
                cands = [(fnum(arm(tbl, "sc_" + lv), "ms"), lv) for lv in LEVELS]
                cands = [(t, lv) for (t, lv) in cands if t]
                return min(cands) if cands else (None, None)

            bt, blv = best(base)
            ct, clv = best(cand)
            if not bt or not ct:
                continue
            crow = arm(cand, "sc_" + clv)
            route = (crow.get("route") or "?") if crow else "?"
            re_ = fnum(crow, "relerr")
            if re_ is not None:
                relerrs.append((re_, mat, n, clv))
            print(f"{mat + '@' + str(n):24s} {mkl:10.4f} | {bt:9.4f} {mkl/bt:6.3f} | "
                  f"{ct:9.4f} {mkl/ct:6.3f} {bt/ct:6.3f} {route:>10s} | "
                  f"{floor*100:5.1f}% {(re_ if re_ is not None else float('nan')):8.1e}")

            grand[group]["base"].append(mkl / bt)
            grand[group]["cand"].append(mkl / ct)
            for lv in LEVELS:
                for tag, tbl in (("base", base), ("cand", cand)):
                    t = fnum(arm(tbl, "sc_" + lv), "ms")
                    if t:
                        grand[group][f"{tag}:{lv}"].append(mkl / t)

            # selector regressions vs plain `off`, in the candidate tree
            off = fnum(arm(cand, "sc_off"), "ms")
            if off:
                for lv in LEVELS[1:]:
                    r = arm(cand, "sc_" + lv)
                    t = fnum(r, "ms")
                    if not t:
                        continue
                    ratio = off / t
                    if ratio < 1.0 - max(floor, 0.02):
                        regressions.append((ratio, group, mat, n, lv,
                                            (r.get("route") or "?"), floor))
            if mkl / ct < 1.0 - floor:
                km = fnum(crow, "kernel_ms")
                below.append((mkl / ct, group, mat, n, clv, ct, km, mkl, floor))

    print("\n\n## geomean vs the faster MKL arm, per group and level")
    hdr = f"{'group':12s} {'cells':>5s} " + " ".join(f"{lv:>10s}" for lv in LEVELS) + f" {'best':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for group, d in grand.items():
        for tag in ("base", "cand"):
            cols = " ".join(f"{geomean(d[f'{tag}:{lv}']):10.3f}" for lv in LEVELS)
            print(f"{group + ' ' + tag:12s} {len(d[tag]):5d} {cols} {geomean(d[tag]):10.3f}")

    allb = [x for d in grand.values() for x in d["base"]]
    allc = [x for d in grand.values() for x in d["cand"]]
    print(f"\nPOOLED over {len(allb)} cells: base {geomean(allb):.3f}x -> "
          f"cand {geomean(allc):.3f}x")

    print("\n## selector regressions in the candidate (level slower than `off`)")
    if not regressions:
        print("  none beyond the per-cell noise floor")
    else:
        print(f"{'off/level':>9s} {'group':11s} {'cell':22s} {'level':9s} {'route':>10s} {'floor':>6s}")
        for ratio, group, mat, n, lv, route, floor in sorted(regressions):
            print(f"{ratio:9.3f} {group:11s} {mat + '@' + str(n):22s} {lv:9s} "
                  f"{route:>10s} {floor*100:5.1f}%")

    print("\n## cells still below parity with MKL")
    if not below:
        print("  none")
    else:
        print(f"{'c/mkl':>6s} {'group':11s} {'cell':22s} {'level':9s} "
              f"{'ms':>9s} {'kernel':>9s} {'fixed_us':>9s} {'mkl_ms':>9s} {'kern/mkl':>9s}")
        for v, group, mat, n, lv, ct, km, mkl, floor in sorted(below):
            fixed = (ct - km) * 1e3 if km else float("nan")
            kr = (mkl / km) if km else float("nan")
            print(f"{v:6.3f} {group:11s} {mat + '@' + str(n):22s} {lv:9s} "
                  f"{ct:9.4f} {(km if km else float('nan')):9.4f} {fixed:9.1f} "
                  f"{mkl:9.4f} {kr:9.3f}")

    if relerrs:
        relerrs.sort(reverse=True)
        print(f"\n## correctness: {len(relerrs)} cells vs a float64 reference, "
              f"max relerr {relerrs[0][0]:.2e} ({relerrs[0][1]}@{relerrs[0][2]}), "
              f"median {statistics.median(r[0] for r in relerrs):.2e}")


if __name__ == "__main__":
    d = sys.argv[1]
    gs = sys.argv[2:] or ["main", "ss-tiling", "ss-quick", "wide"]
    analyse(d, gs)
