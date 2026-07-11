#!/usr/bin/env python3
r"""tiling_selector_validate.py — parse an oracle log (tiling_autotuner_<plat>.log)
into ground truth, then validate the selector (static pick + probe candidates)
against it. Optionally grid-fit the cache constants (L2c, L3/SLC) to the machine.

Usage:
  python tiling_selector_validate.py bench_results/tiling_autotuner_darwin.log
  python tiling_selector_validate.py <log> --fit     # grid-search best constants
"""
from __future__ import annotations
import sys, re, math, itertools

# the canonical selector lives in tiling_selector.py (the thing Phase B ports to
# production); import it so there is one source of truth.
from tiling_selector import select_static, candidates  # noqa: E402


# ---------------------------------------------------------------------------
# oracle-log parser
# ---------------------------------------------------------------------------
HDR = re.compile(r"#{5,}\s+(\S+).*M=(\d+) J=(\d+) nnz=(\d+).*W\*=(\d+)")
NROW = re.compile(r"N=\s*(\d+)\s+ORACLE=")
MEAS = re.compile(r"measured:\s+(.*?)\s+relerr")


def parse_log(path):
    gt = []
    name = M = J = nnz = Ws = None
    curN = None
    with open(path) as f:
        for line in f:
            m = HDR.search(line)
            if m:
                name, M, J, nnz, Ws = m.group(1), int(m.group(2)), int(m.group(3)), \
                    int(m.group(4)), int(m.group(5))
                continue
            m = NROW.search(line)
            if m:
                curN = int(m.group(1)); continue
            m = MEAS.search(line)
            if m and curN is not None:
                g = {}
                for tok in m.group(1).split():
                    if "=" in tok:
                        k, v = tok.split("="); g[k] = float(v)
                gt.append((name, M, J, nnz, Ws, curN, g))
                curN = None
    return gt


def evaluate(gt, *, L2c, L3, ik_ok=True, verbose=False):
    gm = lambda xs: math.exp(sum(math.log(max(1e-9, x)) for x in xs) / len(xs))
    sr, pr, sh, ph = [], [], 0, 0
    if verbose:
        print(f"{'matrix':10s}{'W*/J':>6s}{'N':>7s} | {'oracle':>9s}{'gf':>5s} | "
              f"{'static':>9s}{'r':>6s} | probe(cands)          r")
    for name, M, J, nnz, Ws, N, g in gt:
        oracle = max(g, key=g.get); og = g[oracle]
        st = select_static(M, J, nnz, Ws, N, L2c=L2c, L3=L3, ik_ok=ik_ok)
        sg = g.get(st, 0.0); s = sg / og
        cs = candidates(M, J, nnz, Ws, N, L2c=L2c, L3=L3)
        pg = max(g.get(c, 0.0) for c in cs); p = pg / og
        sr.append(s); pr.append(p); sh += s >= 0.97; ph += p >= 0.97
        if verbose:
            print(f"{name:10s}{Ws/J:6.2f}{N:7d} | {oracle:>9s}{og:5.0f} | "
                  f"{st:>9s}{s:6.2f} | {'+'.join(c.replace('tile-','') for c in cs):20s}{p:5.2f}")
    return gm(sr), sh, gm(pr), ph, len(gt)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    path = sys.argv[1]
    do_fit = "--fit" in sys.argv
    gt = parse_log(path)
    print(f"parsed {len(gt)} cells from {path}\n")

    MB = 1024 * 1024
    if do_fit:
        best = None
        for L2c in [2, 4, 8, 12, 16, 24]:
            for L3 in [16, 24, 32, 36, 48, 64, 96]:
                for ik in (True, False):
                    sgm, sh, pgm, ph, n = evaluate(gt, L2c=L2c*MB, L3=L3*MB, ik_ok=ik)
                    key = (sgm, pgm)
                    if best is None or key > best[0]:
                        best = (key, L2c, L3, ik, sgm, sh, pgm, ph)
        _, L2c, L3, ik, sgm, sh, pgm, ph = best
        print(f"BEST FIT: L2c={L2c}MB L3/SLC={L3}MB ik_ok={ik}")
        print(f"  static geomean {sgm:.3f} ({sh}/{len(gt)} within 3%) | "
              f"probe geomean {pgm:.3f} ({ph}/{len(gt)} within 3%)\n")
        evaluate(gt, L2c=L2c*MB, L3=L3*MB, ik_ok=ik, verbose=True)
    else:
        # default M5 constants
        L2c = int(sys.argv[sys.argv.index("--l2c")+1]) if "--l2c" in sys.argv else 16
        L3 = int(sys.argv[sys.argv.index("--l3")+1]) if "--l3" in sys.argv else 24
        ik = "--noik" not in sys.argv
        sgm, sh, pgm, ph, n = evaluate(gt, L2c=L2c*MB, L3=L3*MB, ik_ok=ik, verbose=True)
        print(f"\nL2c={L2c}MB L3/SLC={L3}MB ik_ok={ik}")
        print(f"STATIC geomean {sgm:.3f} ({sh}/{n} within 3%)")
        print(f"PROBE  geomean {pgm:.3f} ({ph}/{n} within 3%)")


if __name__ == "__main__":
    main()
