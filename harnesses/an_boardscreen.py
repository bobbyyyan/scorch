#!/usr/bin/env python3
"""A cheap screen for board contamination: does a pass's own wall time predict its ours/mkl ratio?

WHY THIS EXISTS. chain210's pre-registered assay-validity check (P4) failed -- the pooled per-width
median ours/mkl did not reproduce between two runs of the same byte-identical objects, worst
deviation 4.34%. Decomposing it put the shift on OUR side (-0.38% to -4.47% across widths) while
MKL moved the other way at k>=8, so it is not "a quieter night". What identified which run was at
fault is this regression, and it needs no mechanism: inside a contaminated run, a slower pass has a
systematically worse ratio.

WHAT IT FOUND, on three sealed boards:

  board                       object-passes   wall spread   r(wall,S) by k=2/4/8/16/32
  chain207 DLMC                    12            3.87%      0.790 -0.045 0.411 0.958 0.878
  chain210 DLMC                    12            0.67%     -0.001  0.073 -0.055 -0.349 -0.406
  chain206 SuiteSparse             18            0.66%     -0.050 -0.070 -0.101 -0.171 -0.049

Three of five widths clear the two-sided 5% critical value in chain207 and none in either other
board. So chain207's board is the contaminated one and chain206's SuiteSparse board stands -- which
matters because chain206 is where the 14-cell SuiteSparse count comes from.

THE CHEAP PART. The wall-time spread alone separates the three boards 3.87% against 0.66-0.67%, and
it is computable from the stage log before any CSV is read. Report it beside every board. The
regression is the confirmation; the spread is the screen.

WHAT IT CANNOT DO. It detects a perturbation that VARIES across passes. A perturbation constant for
a whole run shifts every pass equally, leaves no within-run slope, and is invisible here -- the
spread would read clean. So a clean screen is necessary and not sufficient, and the only thing that
catches a constant perturbation is a second run, which is what P4 is for.

Takes the ledger root as argv[1]; the CSV prefix is derived from what is there, not assumed.
"""
import csv, glob, math, os, statistics, sys, collections

med = statistics.median
# Two-sided 5% critical values for Pearson r, by n.
CRIT = {6:0.811, 8:0.707, 10:0.632, 12:0.576, 14:0.532, 16:0.497, 18:0.468, 20:0.444, 24:0.404, 30:0.361}


def crit(n):
    ks = sorted(CRIT)
    return CRIT[min(ks, key=lambda k: abs(k - n))]


def hms(s):
    return sum(int(x) * f for x, f in zip(s.split(":"), (3600, 60, 1)))


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if not sx or not sy:
        return float("nan"), float("nan")
    r = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    return r, 100.0 * slope * 100.0 / my          # percent of the mean ratio per 100 s


def screen(d):
    stages = glob.glob(os.path.join(d, "*_stage.log"))
    if len(stages) != 1:
        sys.exit("REFUSE: %d files match *_stage.log in %s -- no unambiguous source of per-pass "
                 "wall times, and this screen is entirely about them." % (len(stages), d))
    st, dn = {}, {}
    for ln in open(stages[0]):
        t = ln.split()
        if len(t) >= 5 and t[0] == "---" and t[3] == "start":
            st[(t[1], t[2])] = hms(t[4])
        elif len(t) >= 5 and t[0] == "---" and t[3] == "done":
            dn[(t[1], t[2])] = hms(t[4])
    W = {}
    for k in st:
        if k not in dn:
            continue
        w = dn[k] - st[k] + (86400 if dn[k] < st[k] else 0)
        if not (0 < w < 14400):
            sys.exit("REFUSE: %s %s spans %d s; the stage log's clock is not seconds-of-day here."
                     % (k[0], k[1], w))
        W[k] = w
    if len(W) < 6:
        sys.exit("REFUSE: only %d completed object-passes; a correlation on fewer than 6 points has "
                 "no critical value worth quoting." % len(W))

    csvs = [f for f in glob.glob(os.path.join(d, "*_float32_p*.csv"))]
    if not csvs:
        sys.exit("REFUSE: no *_float32_p*.csv under %s -- an empty glob makes every correlation nan, "
                 "which reads as 'no contamination found'." % d)
    per = {}
    for f in csvs:
        base = os.path.basename(f)
        parts = base.split("_")
        b, p = parts[1], "p" + base.rsplit("_p", 1)[1].split(".")[0]
        acc = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in csv.DictReader(open(f)):
            try:
                ms = float(r["ms"])
            except (TypeError, ValueError, KeyError):
                continue
            if ms > 0:
                acc[int(r["k"])][r["arm"]].append(ms)
        per[(p, b)] = acc

    ws = sorted(W.values())
    spread = 100.0 * (ws[-1] / ws[0] - 1)
    print("%s" % d)
    print("  %d completed object-passes; wall %d to %d s, median %d s, SPREAD %.2f%%"
          % (len(W), ws[0], ws[-1], med(ws), spread))
    print("  CV %.3f%%" % (100.0 * statistics.stdev(ws) / statistics.mean(ws)) if len(ws) > 1 else "")
    print()
    print("  %-5s %11s %7s %15s %9s" % ("k", "r(wall,S)", "n", "slope %/100s", "vs crit"))
    print("  " + "-" * 52)
    flagged = 0
    ks = sorted({k for a in per.values() for k in a})
    for k in ks:
        xs, ys = [], []
        for key in per:
            if key not in W or k not in per[key]:
                continue
            o, m = per[key][k].get("ours"), per[key][k].get("mkl")
            if not o or not m:
                continue
            xs.append(W[key]); ys.append(med(o) / med(m))
        if len(xs) < 6:
            continue
        r, sl = pearson(xs, ys)
        c = crit(len(xs))
        hit = abs(r) >= c
        flagged += 1 if hit else 0
        print("  k=%-3d %11.3f %7d %15.4f %9s" % (k, r, len(xs), sl, "OVER" if hit else "-"))
    print()
    print("  %d of %d widths clear |r| >= %.3f (two-sided 5%%, n=%d)" % (flagged, len(ks), crit(len(W)), len(W)))
    if flagged >= 2:
        print("  SCREEN: FLAGGED -- a pass's own wall time predicts its ratio at %d widths. Treat this" % flagged)
        print("  board's behind-set as contaminated by a pass-varying perturbation until replicated.")
    elif flagged == 1:
        print("  SCREEN: one width over critical. At 5%% per test that is within multiplicity on %d" % len(ks))
        print("  widths (expected %.2f), so this is not a flag on its own." % (0.05 * len(ks)))
    else:
        print("  SCREEN: clean on this axis. NECESSARY, NOT SUFFICIENT -- a perturbation constant")
        print("  across a whole run shifts every pass equally, leaves no within-run slope, and is")
        print("  invisible here. Only a second run catches that.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: an_boardscreen.py <ledger dir> [<ledger dir> ...]")
    for i, a in enumerate(sys.argv[1:]):
        if i:
            print()
        screen(a)
