#!/usr/bin/env python3
"""Score thread rules against the measured per-cell optimum.

THE SPLIT IS FIXED HERE AND NOT CHOSEN AFTER LOOKING. Keys are sorted; EVEN indices are the FIT
half and ODD indices the TEST half. Every constant any rule needs is estimated on the FIT half
only, and every number reported as a result comes from the TEST half. A rule whose constants were
fitted on the cells it is scored on is not a rule, it is a lookup table.

THE REGRET of a rule on a cell is time(the count the rule names) / time(the best count measured
on that cell). It is >= 1 by construction, and 1.0 means the rule picked an optimal count. The
geometric median of regret over cells is the figure of merit; the share of cells where regret
exceeds 1.10 is the harm figure, because a pooled central tendency cannot see a tail and this
lever's whole problem is a tail.

FOUR RULES, three of which already exist in the library:
  shipped      what the default object resolves to with override = pool
  adopt_grain  the same with SCORCH_SPMM_ADOPT_GRAIN = G, best single G chosen ON THE FIT HALF
  sqrt         n* = clamp(round(sqrt(T1/c)), 1, min(pool, rows/16)), T1 = a*nnz + b*nnz*k
  oracle       the measured argmin, which is regret 1.0 by definition and is the ceiling
"""
import argparse, csv, collections, math, os, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--pool", type=int, default=6)
ap.add_argument("--percore-us", type=float, default=4.75,
                help="independently measured per-worker team cost for this host")
ap.add_argument("--rpt", type=int, default=16, help="SCORCH_ROWS_PER_THREAD")
ap.add_argument("--grain", type=int, default=150000)
A = ap.parse_args()

t = collections.defaultdict(dict)        # (key,k) -> {nt: ms}
meta = {}
resolved = collections.defaultdict(dict)
for r in csv.DictReader(open(A.csv)):
    key, k, nt = r["key"], int(r["k"]), int(r["nthreads"])
    t[(key, k)][nt] = float(r["ms"])
    resolved[(key, k)][nt] = int(r["resolved"])
    meta[key] = (int(r["rows"]), int(r["nnz"]))
keys = sorted({key for (key, _) in t})
ks = sorted({k for (_, k) in t})
FIT = set(keys[0::2])
TEST = set(keys[1::2])
print("%d keys (%d fit / %d test)  widths %s  pool %d  per-worker cost %.2f us"
      % (len(keys), len(FIT), len(TEST), ks, A.pool, A.percore_us))

gm = lambda v: math.exp(st.median([math.log(x) for x in v])) if v else float("nan")
cap = lambda M: max(1, min(A.pool, M // A.rpt))


def best(key, k):
    d = t[(key, k)]
    n = min(d, key=lambda n: d[n])
    return n, d[n]


# ---- did the override deliver? A cell where it did not is not evidence about a count.
undeliv = sum(1 for (key, k) in t for nt in t[(key, k)]
              if resolved[(key, k)][nt] != min(nt, cap(meta[key][0])) )
print("cells where the resolved count != min(requested, rows/%d): %d of %d  (a nonzero count here"
      % (A.rpt, undeliv, sum(len(v) for v in t.values())))
print("  means some readings are not at the count they are labelled with)")

# ---- FIT: the T1 model, on the fit half only
xs, ys = [], []
for key in sorted(FIT):
    M, nnz = meta[key]
    for k in ks:
        if 1 in t[(key, k)]:
            xs.append((nnz, nnz * k)); ys.append(t[(key, k)][1] * 1000.0)
n = len(xs)
s1 = sum(x[0] for x in xs); s2 = sum(x[1] for x in xs)
s11 = sum(x[0] * x[0] for x in xs); s22 = sum(x[1] * x[1] for x in xs)
s12 = sum(x[0] * x[1] for x in xs)
sy1 = sum(x[0] * y for x, y in zip(xs, ys)); sy2 = sum(x[1] * y for x, y in zip(xs, ys))
det = s11 * s22 - s12 * s12
a_ = (sy1 * s22 - sy2 * s12) / det
b_ = (sy2 * s11 - sy1 * s12) / det
print()
print("FIT half, single-thread model T1 = a*nnz + b*nnz*k   (microseconds, n=%d)" % n)
print("  a = %.4e us per nonzero (k-independent)   b = %.4e us per nonzero-unit" % (a_, b_))
print("  implied: at k=2 the k-independent term is %.0f%% of T1 for the median fit matrix"
      % (100.0 * a_ / (a_ + 2 * b_)))

T1 = lambda nnz, k: max(a_ * nnz + b_ * nnz * k, 1e-6)


def rule_sqrt(key, k):
    M, nnz = meta[key]
    return max(1, min(cap(M), int(round(math.sqrt(T1(nnz, k) / A.percore_us)))))


def rule_shipped(key, k):
    return min(resolved[(key, k)].get(A.pool, A.pool), A.pool)


def rule_adopt(key, k, G):
    M, nnz = meta[key]
    pol = min(cap(M), max(1, (nnz * max(k, 16)) // A.grain))
    cand = min(A.pool, M // A.rpt, max(1, (nnz * k) // G))
    return max(1, min(max(pol, cand), A.pool))


# ---- FIT: choose G on the fit half
cands = [600, 2000, 5000, 10000, 20000, 50000, 150000, 400000]
bestG, bestr = None, None
for G in cands:
    rg = []
    for key in sorted(FIT):
        for k in ks:
            if not t[(key, k)]:
                continue
            nt = rule_adopt(key, k, G)
            if nt in t[(key, k)]:
                rg.append(t[(key, k)][nt] / best(key, k)[1])
    r = gm(rg)
    if bestr is None or r < bestr:
        bestG, bestr = G, r
print("  best single ADOPT_GRAIN on the fit half: G = %d (fit regret %.4f)" % (bestG, bestr))
print("  fitted c that would make sqrt match the fit half is reported below for convergence")

print()
print("TEST HALF -- every number below is held out. Regret = time(rule) / time(best measured).")
print("%4s %6s | %-22s %-22s %-22s" % ("k", "n", "shipped", "adopt_grain G=%d" % bestG, "sqrt"))
print("%4s %6s | %9s %11s %9s %11s %9s %11s" % ("", "", "regret", ">1.10", "regret", ">1.10",
                                                 "regret", ">1.10"))
agree = collections.Counter()
for k in ks:
    cols = []
    for name in ("shipped", "adopt", "sqrt"):
        rg = []
        for key in sorted(TEST):
            if not t[(key, k)]:
                continue
            nt = (rule_shipped(key, k) if name == "shipped" else
                  rule_adopt(key, k, bestG) if name == "adopt" else rule_sqrt(key, k))
            if nt in t[(key, k)]:
                rg.append(t[(key, k)][nt] / best(key, k)[1])
        cols.append((gm(rg), 100.0 * sum(1 for x in rg if x > 1.10) / max(len(rg), 1), len(rg)))
    print("%4d %6d | %9.4f %10.1f%% %9.4f %10.1f%% %9.4f %10.1f%%"
          % (k, cols[0][2], cols[0][0], cols[0][1], cols[1][0], cols[1][1],
             cols[2][0], cols[2][1]))

print()
print("REGISTERED PREDICTION 2: the sqrt rule's count agrees with the measured argmin within +/-1.")
print("%4s %8s %10s %10s %10s" % ("k", "n", "within 0", "within 1", "off by >2"))
for k in ks:
    d = []
    for key in sorted(TEST):
        if not t[(key, k)]:
            continue
        d.append(abs(rule_sqrt(key, k) - best(key, k)[0]))
    if not d:
        continue
    print("%4d %8d %9.1f%% %9.1f%% %9.1f%%" % (k, len(d),
          100.0 * sum(1 for x in d if x == 0) / len(d),
          100.0 * sum(1 for x in d if x <= 1) / len(d),
          100.0 * sum(1 for x in d if x > 2) / len(d)))

print()
print("REGISTERED PREDICTION 1: no single linear G matches the per-width optimum within +/-1.")
print("The measured modal argmin by width, against what each G names for the median cell:")
print("%4s %12s   %s" % ("k", "modal argmin", "  ".join("G=%d" % G for G in cands)))
for k in ks:
    arg = collections.Counter(best(key, k)[0] for key in sorted(TEST) if t[(key, k)])
    if not arg:
        continue
    modal = arg.most_common(1)[0][0]
    row = []
    for G in cands:
        v = collections.Counter(rule_adopt(key, k, G) for key in sorted(TEST) if t[(key, k)])
        row.append(v.most_common(1)[0][0])
    print("%4d %12d   %s" % (k, modal, "  ".join("%5d" % x for x in row)))
print()
print("A G 'matches' a width if its modal count equals the measured modal argmin. Prediction 1")
print("fails if any single column matches at EVERY width.")
