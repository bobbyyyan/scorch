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

THE STATISTIC IS max|r| WITH A PERMUTATION p, NOT A COUNT OF WIDTHS. The first version of this
file flagged on "two or more widths above the per-width critical value". Permuting the twelve pass
labels -- which carries the width-to-width dependence exactly, since all five widths are recomputed
from the same permuted assignment -- puts that rule's own false-positive rate at 0.0441 to 0.1051
across three boards of the SAME design, against the 0.0226 independence would give. P(at least one
width over critical) is 0.111 to 0.197. So a width count is not evidence at n=12, there is no
multiplicity constant to hardcode, and the rate has to come from each board's own pass structure.
max|r| with its permutation p is both correct and much stronger: chain207 reads p = 0.00011 at
|r| = 0.963, chain210 p = 0.667 at 0.263, chain206 p = 0.894 at 0.204.

A CLEAN SCREEN IS A BOUND, NOT A CLEARANCE. "chain206 stands" is a detection claim. The defensible
form multiplies each board's wall-time spread in SECONDS by a reference sensitivity -- chain207's
measured worst slope, 11.57% per 100 s at k=32 -- to get how much pass-varying contamination the
spread admits: chain207 6.32%, chain210 1.04%, chain206 1.40%. chain206's bound sits BELOW the 2%
P4 gate, which is the useful statement. Note that chain206's percentage spread equals chain210's at
0.66% while its bound is 35% wider, because its passes are longer: seconds times slope is the
comparable quantity across boards and a percentage spread is not.

WHAT IT CANNOT DO. It detects a perturbation that VARIES across passes. A perturbation constant for
a whole run shifts every pass equally, leaves no within-run slope, and is invisible here -- the
spread would read clean. So a clean screen is necessary and not sufficient, and the only thing that
catches a constant perturbation is a second run, which is what P4 is for.

Takes the ledger root as argv[1]; the CSV prefix is derived from what is there, not assumed.
"""
import collections, csv, glob, math, os, random, statistics, sys

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
    # The slope is reported as a percentage of the mean ratio, which is undefined on a CENTRED
    # series (mean 0). The within-object check needs only r, so return nan rather than dividing.
    pct = 100.0 * slope * 100.0 / my if abs(my) > 1e-12 else float("nan")
    return r, pct                                  # percent of the mean ratio per 100 s


def perm_max_r(W, per, ks, draws, seed):
    """P(max|r| over widths >= observed) under permutation of the PASS LABELS.

    Permuting pass labels rather than resampling widths is what carries the dependence: all five
    widths are recomputed from the same permuted assignment, exactly as in the observed statistic.
    """
    keys = [k for k in per if k in W]
    walls = [W[k] for k in keys]
    series = {}
    for k in ks:
        ys = []
        ok = True
        for key in keys:
            o = per[key].get(k, {}).get("ours")
            m = per[key].get(k, {}).get("mkl")
            if not o or not m:
                ok = False; break
            ys.append(med(o) / med(m))
        if ok:
            series[k] = ys
    if not series:
        return float("nan"), float("nan"), {}
    obs = {k: abs(pearson(walls, series[k])[0]) for k in series}
    mobs = max(obs.values())
    rng = random.Random(seed)
    hits = 0
    idx = list(range(len(walls)))
    for _ in range(draws):
        rng.shuffle(idx)
        pw = [walls[i] for i in idx]
        if max(abs(pearson(pw, series[k])[0]) for k in series) >= mobs:
            hits += 1
    return mobs, (hits + 1.0) / (draws + 1.0), obs


def screen(d, draws=20000, seed=20261001, ref_slope=11.57):
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
        if (p, b) in per:
            sys.exit("REFUSE: %s and an earlier file both map to key (%s, %s). The key comes from "
                     "the filename's second field, so a board with two files per pass per object "
                     "would silently overwrite one and halve the sample." % (base, p, b))
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
    mx, pp, obs = perm_max_r(W, per, ks, draws, seed)
    bound = (ws[-1] - ws[0]) * ref_slope / 100.0
    print("  max|r| = %.3f at k=%s ; permutation p = %.5f (%d draws, pass labels permuted so the"
          % (mx, [k for k in obs if abs(obs[k]) == mx][0] if obs else "?", pp, draws))
    print("    width-to-width dependence and the five-width maximum are both in the null)")
    print("  P(>=1 width over the per-width critical value) under the same null: see the docstring --")
    print("    it runs 0.111 to 0.197 on boards of this design, so a WIDTH COUNT is not evidence here.")
    print()
    print("  ADMISSIBLE CONTAMINATION, which is what a clean screen actually licenses:")
    print("    wall spread %.1f s x %.2f%%/100s (chain207's measured worst slope) = %.2f%% in the ratio"
          % (ws[-1] - ws[0], ref_slope, bound))
    print("    Compare against the 2% P4 reproducibility gate. Seconds x slope is the comparable")
    print("    quantity across boards; a PERCENTAGE spread is not, since longer passes admit more.")
    print()
    if pp <= 0.01:
        print("  SCREEN: FLAGGED at p = %.5f. A pass's own wall time predicts its ratio. Treat this" % pp)
        print("  board's behind-set as contaminated by a pass-varying perturbation until replicated.")
    elif pp <= 0.05:
        print("  SCREEN: BORDERLINE at p = %.5f. Not a clearance and not a flag; replicate." % pp)
    else:
        print("  SCREEN: no pass-varying signal (p = %.5f). The claim is the BOUND above, %.2f%%," % (pp, bound))
        print("  not that the board is clean. And it is bounded only against a perturbation that")
        print("  VARIES across passes: one constant for a whole run shifts every pass equally, leaves")
        print("  no within-run slope, and is invisible here. Only a second run catches that.")

    # Between-object confound: on a multi-object board the pooled slope could be picking up a
    # systematic difference BETWEEN objects rather than a within-run perturbation. Measured, not
    # assumed -- it moved nothing by more than 0.06 on either board checked.
    objs = {b for (_, b) in per}
    if len(objs) > 1:
        print()
        print("  WITHIN-OBJECT CHECK (%d objects): x and y centred inside each object first, so a" % len(objs))
        print("  between-object difference in wall time and ratio cannot produce the slope.")
        print("    %-5s %10s %10s" % ("k", "pooled r", "within r"))
        for k in ks:
            xs, ys, gs = [], [], []
            for key in per:
                if key not in W or k not in per[key]: continue
                o, m = per[key][k].get("ours"), per[key][k].get("mkl")
                if not o or not m: continue
                xs.append(W[key]); ys.append(med(o) / med(m)); gs.append(key[1])
            if len(xs) < 6: continue
            r0, _ = pearson(xs, ys)
            cx, cy = [], []
            for g in set(gs):
                ix = [i for i in range(len(xs)) if gs[i] == g]
                if len(ix) < 2: continue
                mxg = sum(xs[i] for i in ix) / len(ix); myg = sum(ys[i] for i in ix) / len(ix)
                for i in ix:
                    cx.append(xs[i] - mxg); cy.append(ys[i] - myg)
            r1, _ = pearson(cx, cy) if len(cx) >= 6 else (float("nan"), 0)
            print("    k=%-3d %10.3f %10.3f" % (k, r0, r1))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: an_boardscreen.py <ledger dir> [<ledger dir> ...]")
    for i, a in enumerate(sys.argv[1:]):
        if i:
            print()
        screen(a)
