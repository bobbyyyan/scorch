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
the adgG object by a separate call. An earlier draft reimplemented the POLICY count too and
disagreed with the object on 16% of cells, always in one direction.
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
    sat[(key, k)] = int(r["saturating"]) if r.get("saturating") else None
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
clampbound = {c for c in live if max(ms[c]) >= max(1, min(pool, caprows[c]))}
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


print()
print("REPLACEMENT CONTROL, with its n so an empty one cannot read as a pass: LIVE cells whose")
print("largest reachable count already equals the clamp min(pool, rows/16). A rule cannot ask for")
print("more there, so every rule naming a count at or above the clamp must read the floor.")
print("  live cells: %d   of those, clamp-bound: %d (%.1f%%)   flat cells excluded: %d"
      % (len(live), len(clampbound), 100.0 * len(clampbound) / max(len(live), 1), len(ms) - len(live)))
if len(clampbound) < 20:
    print("  WARNING: fewer than 20 clamp-bound live cells, so this control is underpowered and")
    print("           a quiet reading from it means little.")


def best(c):
    d = ms[c]
    n = min(d, key=lambda n: min(d[n]))
    return n, min(d[n])


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

print()
print("REGISTERED PREDICTION 2, scored as registered: |rule count - measured argmin| <= 1.")
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
