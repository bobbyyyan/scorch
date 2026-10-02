#!/usr/bin/env python3
"""Score thread rules against the measured per-cell optimum.

A RULE NAMES A RESOLVED COUNT, NOT AN OVERRIDE. The sweep varies the override; the object decides
what that resolves to, and two different overrides frequently resolve to the same count. So
everything here is grouped by the RESOLVED count the object reported.

WHAT IS EXCLUDED AND WHY. A cell with one reachable count is excluded. The sweep timed one
configuration several times there, so an argmin over those readings is an argmin over noise and
any rule compared to it is compared to a coin flip. 40.1% of cells on the M5 at pool 6 are like
this -- the policy floor pins 141 of them at 1 below the adoption gate, and the final cap's
recruit decline pins 90 at 18 because their policy count reaches twice the caller's pool.

AND WHAT THAT BUYS. Where two overrides resolve to the SAME count, those readings time the same
configuration with a different argument: a free same-object A/A floor at the cell level. It is
printed beside the effect, so a regret is read against the noise measured on its own cell rather
than against an assumption.

THE SPLIT IS FIXED HERE AND NOT CHOSEN AFTER LOOKING: keys sorted, EVEN indices FIT, ODD TEST.
Every constant any rule needs is estimated on FIT; every reported number is from TEST.

ONE RESTATEMENT OF PRODUCTION, FLAGGED. `policy` and `cap_rows` come from the object, recorded by
the sweep. The adoption candidate for an arbitrary grain G cannot -- G is compiled in -- so
rule_adopt restates `min(pool, cap_rows, max(1, nnz*k//G))` and is validated at G=150000 against
the adgG object by a separate call. An earlier draft reimplemented the POLICY count too, which
is the defect this comment exists to record: a harness that recomputes a production policy is
wrong silently and flatteringly.

A FIGURE WITHDRAWN. That draft was reported as disagreeing with the object on 92 of 576 cells
(16.0%), one-signed. It does not. Re-measured against the DEFAULT object, the restated policy
count disagrees on 0 of 576 cells and the signed-difference histogram is empty. The 16% was a
measurement of an experimental SCORCH_SPMM_BASE_WORK_TRUE=1 build that an in-place two-arm
script had left in the worktree -- which the conda env's editable .pth makes the import default
for the whole environment -- so it described that object, not any restatement error. Do not
quote it. The structural objection stands on its own: a restatement can drift from production
without any run noticing, so it is validated against the object here rather than trusted.
"""
import argparse, csv, collections, math, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--percore-us", type=float, required=True,
                help="independently measured per-worker team cost for THIS host, microseconds")
ap.add_argument("--grain", type=int, default=150000, help="base grain, for the adopt family only")
A = ap.parse_args()

ms = collections.defaultdict(lambda: collections.defaultdict(list))   # (key,k) -> {resolved:[ms]}
meta, pol, caprows, pool = {}, {}, {}, None
sat = {}
shipped_res = {}
for r in csv.DictReader(open(A.csv)):
    key, k = r["key"], int(r["k"])
    ms[(key, k)][int(r["resolved"])].append(float(r["ms"]))
    meta[key] = (int(r["rows"]), int(r["nnz"]))
    # Accept either column name. A CSV written before the rename carries `policy`, which is the
    # resolved count at override 0 -- the policy count CAPPED BY THE POOL, not the policy count.
    pol[(key, k)] = int(r.get("res_ovr0") or r["policy"])
    sat[(key, k)] = int(r.get("res_ovr_max") or r.get("saturating") or 0) or None
    caprows[(key, k)] = int(r["cap_rows"])
    pool = int(r["pool"])
    if int(r["nthreads"]) == pool:
        shipped_res[(key, k)] = int(r["resolved"])
keys = sorted({key for (key, _) in ms})
ks = sorted({k for (_, k) in ms})
FIT, TEST = set(keys[0::2]), set(keys[1::2])
gm = lambda v: math.exp(st.median([math.log(x) for x in v])) if v else float("nan")

live = {c for c in ms if len(ms[c]) >= 2}
# THE REGISTERED 33288-ROW NEGATIVE CONTROL IS WITHDRAWN. It was registered on the ground that
# sqrt(T1/c) exceeds min(pool, rows/16) there so the clamp binds and the rule is provably inert.
# Measured, 57 of that stratum's 72 cells are flat at resolved 18 -- inert through the final
# cap's recruit DECLINE, not through the clamp -- so the control cannot distinguish its
# registered reason from a harness fault, which is the one thing it existed to do. Excluding
# flat cells empties it rather than cleaning it. What replaces it is printed below with its own
# n, so an empty control reads as empty instead of as a pass.
# MY FIRST REPLACEMENT CONTROL WAS DEGENERATE: `max(reachable) >= clamp` selected 345 of 345
# live cells, because override 0 can resolve above the pool so the reachable maximum always
# reaches the clamp. A control that selects everything discriminates nothing -- the same failure
# as the stratum control it replaced, one definition later. The discriminating quantity is
# whether the measured OPTIMUM sits at or above the clamp, which is where a rule naming a count
# at the clamp is forced to read the floor. Printed per width as `argmin>=cap` alongside the
# strictly-greater count, so a zero reads as "the clamp never excludes the optimum" -- itself a
# clean result -- rather than as a pass.
# computed at the print site below, after best() exists
print("%d keys (%d fit / %d test)  widths %s  pool %d  per-worker cost %.2f us"
      % (len(keys), len(FIT), len(TEST), ks, pool, A.percore_us))
print("cells: %d total, %d with >=2 reachable counts (LIVE), %d excluded as single-configuration"
      % (len(ms), len(live), len(ms) - len(live)))
print("  reachable-count histogram: %s"
      % dict(sorted(collections.Counter(len(ms[c]) for c in ms).items())))

print()
print("THE CELL-LEVEL A/A FLOOR: the same resolved count timed more than once on the same object.")
print("With REPEAT=2 in the sweep this exists on every live cell by design. Before REPEAT it came")
print("only from two overrides happening to resolve alike, which supplied 3.1% of cells.")
print("%4s %8s %10s %10s" % ("k", "n pairs", "median", "p95"))
floor95 = {}
for k in ks:
    sp = [max(v) / min(v) - 1.0 for (key, kk), d in ms.items() if kk == k and (key, kk) in live
          for v in d.values() if len(v) >= 2]
    if not sp:
        floor95[k] = float("nan"); print("%4d %8d %10s %10s" % (k, 0, "-", "-")); continue
    sp.sort()
    floor95[k] = sp[min(len(sp) - 1, int(0.95 * len(sp)))] * 100
    print("%4d %8d %9.3f%% %9.3f%%" % (k, len(sp), st.median(sp) * 100, floor95[k]))


def best(c):
    d = ms[c]
    n = min(d, key=lambda n: min(d[n]))
    return n, min(d[n])


def best_argmin_at_or_above_cap(c):
    return best(c)[0] >= max(1, min(pool, caprows[c]))


print()
print("REPLACEMENT CONTROL, with its n so an empty one cannot read as a pass: LIVE cells whose")
print("measured OPTIMUM sits at or above the clamp min(pool, rows/16). A rule naming a count at")
print("the clamp is forced to read the floor there.")
clampbound = {c for c in live if best_argmin_at_or_above_cap(c)}
print("  live cells %d   optimum AT or above the clamp: %d (%.1f%%)   strictly above: %d"
      % (len(live), len(clampbound), 100.0 * len(clampbound) / max(len(live), 1),
         sum(1 for c in live if best(c)[0] > max(1, min(pool, caprows[c])))))
if len(clampbound) < 20:
    print("  NOTE: fewer than 20 such cells. A quiet reading from this control means little, and")
    print("        a count of zero strictly above the clamp means the clamp never excludes the")
    print("        optimum -- which is a result about the clamp, not a pass for the rule.")


def at(c, n):
    """Time at the nearest REACHABLE count to n, plus whether a snap was needed."""
    d = ms[c]
    if n in d:
        return min(d[n]), False
    nn = min(d, key=lambda x: (abs(x - n), x))
    return min(d[nn]), True


# ---------- FIT: the T1 model. Its own refutation check comes first.
print()
print("THE T1 MODEL AND ITS OWN REFUTATION CHECK (prereg amendment: 'three host constants').")
print("T1 = a*nnz + b*nnz*k treats a as a CONSTANT. Fit per nnz decade on the FIT half, at the")
print("resolved count 1 only. If a varies by orders of magnitude the model has no constant a,")
print("and since n* = sqrt(T1/c) a factor f in T1 is sqrt(f) in the count.")
bands = collections.defaultdict(list)
for key in sorted(FIT):
    M, nnz = meta[key]
    for k in ks:
        c = (key, k)
        if c in ms and 1 in ms[c]:
            bands[int(math.log10(max(nnz, 1)))].append((nnz, k, min(ms[c][1]) * 1000.0))


def fit(pts):
    s11 = sum(p[0] * p[0] for p in pts); s22 = sum((p[0] * p[1]) ** 2 for p in pts)
    s12 = sum(p[0] * p[0] * p[1] for p in pts)
    sy1 = sum(p[0] * p[2] for p in pts); sy2 = sum(p[0] * p[1] * p[2] for p in pts)
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-30:
        return float("nan"), float("nan")
    return (sy1 * s22 - sy2 * s12) / det, (sy2 * s11 - sy1 * s12) / det


print("%10s %6s %14s %14s" % ("nnz decade", "n", "a (us/nnz)", "b (us/unit)"))
avals = []
for d in sorted(bands):
    a_, b_ = fit(bands[d])
    if a_ == a_:
        avals.append(a_)
    print("%10s %6d %14.3e %14.3e" % ("1e%d" % d, len(bands[d]), a_, b_))
allpts = [p for v in bands.values() for p in v]
A_, B_ = fit(allpts)
print("  pooled over all bands: a = %.3e  b = %.3e" % (A_, B_))
if len(avals) >= 2:
    sp = max(avals) / min(avals)
    print("  SPREAD IN a ACROSS BANDS: %.0fx  ->  up to %.1fx in n* via sqrt. %s"
          % (sp, math.sqrt(sp), "a is NOT a constant; the model's premise fails as written."
             if sp > 4 else "within a factor of four, so a single a is defensible."))

T1m = lambda nnz, k: max(A_ * nnz + B_ * nnz * k, 1e-6)
cap = lambda c: max(1, min(pool, caprows[c]))


def r_shipped(c):
    return shipped_res.get(c, pol[c])


def r_policy(c):
    return pol[c]


def r_adopt(c, G):
    key, k = c
    M, nnz = meta[key]
    cand = min(pool, caprows[c], max(1, (nnz * k) // G))
    return max(1, min(max(pol[c], cand), pool))


def r_sqrt_meas(c):
    """The FORM, given T1 measured. Upper bound on what the form can do; cannot ship."""
    if 1 not in ms[c]:
        return None
    t1 = min(ms[c][1]) * 1000.0
    return max(1, min(cap(c), int(round(math.sqrt(t1 / A.percore_us)))))


def r_sqrt_model(c):
    key, k = c
    M, nnz = meta[key]
    return max(1, min(cap(c), int(round(math.sqrt(T1m(nnz, k) / A.percore_us)))))


# ---------- FIT: best single G
cands = [600, 2000, 5000, 20000, 50000, 150000, 400000]
bestG, bestr = None, None
for G in cands:
    rg = [at((key, k), r_adopt((key, k), G))[0] / best((key, k))[1]
          for key in sorted(FIT) for k in ks if (key, k) in live]
    r = gm(rg)
    if bestr is None or r < bestr:
        bestG, bestr = G, r
print()
print("FIT half: best single ADOPT_GRAIN is G = %d (fit regret %.4f)" % (bestG, bestr))

RULES = [("shipped", r_shipped), ("policy", r_policy),
         ("adopt G=%d" % bestG, lambda c: r_adopt(c, bestG)),
         ("sqrt/measT1", r_sqrt_meas), ("sqrt/modelT1", r_sqrt_model)]
print()
print("REGRET'S DENOMINATOR HAS A WINNER'S CURSE AND IT IS LARGE. `best(c)` is the MINIMUM over")
print("roughly twelve readings -- about six reachable counts at two repeats each -- so it sits")
print("below the true best count's time even when every count is identical, and every regret")
print("below is inflated by that amount. The size is not a guess: on the same sweep, production's")
print("own count against `best` reads 1.1856 at k=2 while a permutation null that shuffles each")
print("cell's readings among its counts reads 1.1818, an excess of 1.0032. At the other widths")
print("the excess is BELOW 1 (0.886-0.953). So on this host the apparent per-cell headroom is")
print("essentially all selection, and `there is N% to recover` does not survive it.")
print()
print("WHAT STILL STANDS AND WHAT DOES NOT. Rule-against-rule comparisons share this denominator,")
print("so the bias largely cancels in their DIFFERENCE and the ordering of the columns below is")
print("still informative. What does not survive is any absolute reading of a regret figure as")
print("recoverable time, including `the sqrt rule reaches 1.0000`: a rule that lands on a")
print("noise-selected argmin has fitted the noise, which is the definition of the overfitting")
print("this project is supposed to refuse. The NULL row below is the regret a rule choosing")
print("counts at random earns on shuffled readings, and no column is evidence unless it beats it.")
print()
print("TEST HALF, LIVE CELLS ONLY. Regret = time(rule's count) / time(best measured count).")
print("Paired with the share above 1.10, because a central tendency cannot see a tail.")
hdr = "%4s %5s" % ("k", "n")
for nm, _ in RULES:
    hdr += " | %-13s" % nm
print(hdr)
print("%4s %5s" % ("", "") + "".join(" | %6s %6s" % ("regret", ">1.10") for _ in RULES))
snaps = collections.Counter()
for k in ks:
    cells = [(key, k) for key in sorted(TEST) if (key, k) in live]
    row = "%4d %5d" % (k, len(cells))
    for nm, f in RULES:
        rg = []
        for c in cells:
            n = f(c)
            if n is None:
                continue
            t, sn = at(c, n)
            if sn:
                snaps[nm] += 1
            rg.append(t / best(c)[1])
        row += " | %6.4f %5.1f%%" % (gm(rg), 100.0 * sum(1 for x in rg if x > 1.10) / max(len(rg), 1))
    print(row)
print("  counts snapped to the nearest reachable value: %s" % dict(snaps))

# THE NULL ROW. Within a cell, readings are exchangeable among counts under the hypothesis that
# the count does not matter; shuffling them and recomputing regret for an arbitrary count gives
# the regret that buys nothing. Computed per width on the same cells, with the same `best`.
import random as _random
print()
print("THE NULL, same cells and same denominator: regret of an arbitrary count on SHUFFLED")
print("readings. A rule is evidence only to the extent it beats this.")
print("%4s %5s %10s %10s   %s" % ("k", "n", "null", "best rule", "beats null by"))
for k in ks:
    cells = [(key, k) for key in sorted(TEST) if (key, k) in live]
    rng = _random.Random(20261002 + k)
    nullv = []
    for c in cells:
        d = ms[c]
        flat = [x for v in d.values() for x in v]
        sizes = [(n, len(v)) for n, v in sorted(d.items())]
        for _ in range(40):
            rng.shuffle(flat)
            i, sh = 0, {}
            for n, cnt in sizes:
                sh[n] = flat[i:i + cnt]
                i += cnt
            pick = rng.choice(sorted(sh))
            bmin = min(min(v) for v in sh.values() if v)
            if bmin > 0 and sh[pick]:
                nullv.append(min(sh[pick]) / bmin)
    if not nullv:
        print("%4d %5d   no shuffleable cell at this width, so no null" % (k, len(cells)))
        continue
    rules_r = []
    for nm, f in RULES:
        rg = [at(c, f(c))[0] / best(c)[1] for c in cells if f(c) is not None]
        if rg:
            rules_r.append((gm(rg), nm))
    nz = gm(nullv)
    br, bn = min(rules_r) if rules_r else (float("nan"), "-")
    print("%4d %5d %10.4f %10.4f   %s is %.4f below the null"
          % (k, len(cells), nz, br, bn, nz - br))

print()
print("=" * 78)
print("EVERY REGISTERED PREDICTION OVER THESE NUMBERS, EACH CLAUSE SCORED SEPARATELY.")
print("A registration holding N predictions over one measured quantity must name all N: a report")
print("that names only the one that passes lets the prose choose the verdict. And a conjunction")
print("must score each clause -- 'at least 80% AND no cell worse than 2' can pass on its share")
print("and fail on its tail.")
print("=" * 78)
P2_WIDTHS = [4, 12, 16, 20, 24]
pooled_d, byw = [], {}
for k in ks:
    cells = [(key, k) for key in sorted(TEST) if (key, k) in live]
    d = [abs(r_sqrt_meas(c) - best(c)[0]) for c in cells if r_sqrt_meas(c) is not None]
    byw[k] = d
    pooled_d.extend(d)

# THE REGISTERED DENOMINATOR IS AMBIGUOUS, SO EVERY DEFENSIBLE ONE IS REPORTED.
# Prediction 3 says "at least 80% of (matrix, width) cells" over "the 72-key corpus", which
# admits four readings: all 576 cells, the 345 live ones, the 288 test ones, the 160 live-test
# ones. The registration predates knowing that 231 of 576 cells can reach only one thread
# count, so "cells" could not then have meant what it now has to mean -- a cell with no
# measurable argmin cannot be within 1 of it.
#   This particular rule is sqrt(T1_measured/c): T1 is measured per cell and c is the host's
# per-worker team cost from a separate probe. NO constant is fitted on these cells, so the
# fit/test split is not protecting anything here and the all-live denominator is not the
# optimistic reading -- it is just the larger one.
pooled_live, byw_live = [], {}
for k in ks:
    cells = [(key, k) for key in sorted(FIT | TEST) if (key, k) in live]
    d = [abs(r_sqrt_meas(c) - best(c)[0]) for c in cells if r_sqrt_meas(c) is not None]
    byw_live[k] = d
    pooled_live.extend(d)
print()
print("PREDICTION 2 -- registered scope: k in %s, 'within +/-1 at every one of those widths'." % P2_WIDTHS)
print("The registration names no share, so the verdict prints at the same three thresholds")
print("prediction 1 gets; both were written at the same time and have the same hole.")
for k in P2_WIDTHS:
    d = byw.get(k, [])
    if d:
        print("   k=%-3d n=%-3d within 1: %5.1f%%" % (k, len(d), 100.0 * sum(1 for x in d if x <= 1) / len(d)))
for thr in (50.0, 80.0, 100.0):
    shares = [100.0 * sum(1 for x in byw[k] if x <= 1) / len(byw[k]) for k in P2_WIDTHS if byw.get(k)]
    ok = shares and min(shares) >= thr
    print("   threshold %5.0f%%: prediction 2 %s%s" % (thr, "HOLDS" if ok else "FAILS",
          "" if ok else "  (lowest width %.1f%%)" % min(shares)))
print()
print("PREDICTION 3 -- registered as a CONJUNCTION over all (matrix,width) cells:")
w1 = 100.0 * sum(1 for x in pooled_d if x <= 1) / max(len(pooled_d), 1)
gt2 = sum(1 for x in pooled_d if x > 2)
print("   clause A  'within +/-1 on at least 80%% of cells'  pooled %.1f%% of n=%d  ->  %s"
      % (w1, len(pooled_d), "HOLDS" if w1 >= 80 else "FAILS"))
print("   clause B  'and no cell off by more than 2'         %d cell(s) off by >2  ->  %s"
      % (gt2, "HOLDS" if gt2 == 0 else "FAILS"))
if gt2:
    print("       by width: %s" % {k: sum(1 for x in v if x > 2) for k, v in byw.items()
                                    if any(x > 2 for x in v)})
print("   NOTE: clause A is POOLED as registered; per width, %s sit below 80%%."
      % [k for k, v in byw.items() if v and 100.0 * sum(1 for x in v if x <= 1) / len(v) < 80])
wL = 100.0 * sum(1 for x in pooled_live if x <= 1) / max(len(pooled_live), 1)
gt2L = sum(1 for x in pooled_live if x > 2)
print("   SAME CLAUSES ON THE OTHER DEFENSIBLE DENOMINATOR -- all live cells, both halves:")
print("     clause A  %5.1f%% of n=%-4d -> %s      clause B  %d off by >2 -> %s"
      % (wL, len(pooled_live), "HOLDS" if wL >= 80 else "FAILS",
         gt2L, "HOLDS" if gt2L == 0 else "FAILS"))
print("     headroom above the registered 80%%: %+.1f points on n=%d, %+.1f on n=%d."
      % (w1 - 80.0, len(pooled_d), wL - 80.0, len(pooled_live)))
if (w1 >= 80) != (wL >= 80):
    print("     THE VERDICT TURNS ON THE DENOMINATOR. Clause A is not decided by this run.")
else:
    print("     Clause A reads the same way under both, so the ambiguity does not decide it.")
print()
print("PER-WIDTH DETAIL, which is not any registered statistic:")
print("The four columns are exhaustive and are asserted to sum to 100%.")
print("%4s %6s %9s %9s %9s %9s" % ("k", "n", "exact", "off by 1", "off by 2", "off by >2"))
for k in ks:
    cells = [(key, k) for key in sorted(TEST) if (key, k) in live]
    d = [abs(r_sqrt_meas(c) - best(c)[0]) for c in cells if r_sqrt_meas(c) is not None]
    if not d:
        continue
    f = [100.0 * sum(1 for x in d if x == j) / len(d) for j in (0, 1, 2)]
    f.append(100.0 * sum(1 for x in d if x > 2) / len(d))
    assert abs(sum(f) - 100.0) < 1e-6, "the four columns must partition the cells"
    print("%4d %6d %8.1f%% %8.1f%% %8.1f%% %8.1f%%" % (k, len(d), f[0], f[1], f[2], f[3]))

print()
print("REGISTERED PREDICTION 1, scored as registered: PER CELL, within +/-1, not modal.")
print("THE SHARE THRESHOLD IS NOT REGISTERED, so the verdict is printed at three of them rather")
print("than at one of my choosing. The registration names a per-width quantity -- within +/-1 of")
print("the measured optimum -- and gives no aggregation rule; an earlier draft borrowed the 80%")
print("that belongs to prediction 3, which is a decision rule supplied by the instrument.")
print("ITS NULL IS PRINTED BESIDE IT, as the count of cells where NO tested G lands within 1 of")
print("the measured argmin. Prediction 1 passes on those for a reason that has nothing to do with")
print("linear-versus-sqrt -- the family simply cannot reach the optimum there -- so a width whose")
print("no-G-can count is a large share of n cannot inform the comparison either way.")
print("%4s %6s %12s %12s   %s" % ("k", "n", "no-G-can", "argmin>cap", "share within 1 by G"))
for k in ks:
    cells = [(key, k) for key in sorted(TEST) if (key, k) in live]
    if not cells:
        continue
    # THE NULL IS COMPUTED DIRECTLY OVER THE LADDER, not from a policy lower bound. The earlier
    # form subtracted the override-0 reading, which is pool-capped and smaller than the count
    # that actually bounds the rule from below on 59.4% of cells -- so the subtraction was
    # smaller and the null UNDER-counted, one-signed toward reporting that prediction 1's pass
    # is informative. Asking the ladder whether ANY tested G lands within 1 needs no bound at
    # all and is exact for the family actually tested.
    nullc = sum(1 for c in cells
                if min(abs(r_adopt(c, G) - best(c)[0]) for G in cands) > 1)
    overc = sum(1 for c in cells if best(c)[0] > cap(c))
    shares = []
    for G in cands:
        ok = sum(1 for c in cells if abs(r_adopt(c, G) - best(c)[0]) <= 1)
        shares.append(100.0 * ok / len(cells))
    print("%4d %6d %11d %12d   %s" % (k, len(cells), nullc, overc,
          " ".join("%.0f%%" % x for x in shares)))
print("  G columns, in order: %s" % cands)
print()
print("VERDICT AT THREE THRESHOLDS. Prediction 1 fails -- and the sqrt form is then unnecessary --")
print("if some single G clears the threshold at EVERY width. Read the null columns first: a width")
print("whose pol-argmin>=2 count is a large share of n cannot inform the comparison either way.")
for thr in (50.0, 80.0, 100.0):
    winners = []
    for G in cands:
        if all(100.0 * sum(1 for c in [(key, k) for key in sorted(TEST) if (key, k) in live]
                           if abs(r_adopt(c, G) - best(c)[0]) <= 1)
               / max(len([(key, k) for key in sorted(TEST) if (key, k) in live]), 1) >= thr
               for k in ks if [(key, k) for key in sorted(TEST) if (key, k) in live]):
            winners.append(G)
    print("  threshold %5.0f%%: %s" % (thr, ("prediction 1 FAILS, G in %s clears every width" % winners)
                                       if winners else "prediction 1 HOLDS, no single G clears every width"))
