"""The PRE-REGISTERED k=2 test, printed as its own named output.

THE REGISTRATION, quoted from PREREG_thread_sizing.md's pending-work clause: "if MKT's k=2
curve is flat in worker count rather than rising, the deficit there is bandwidth saturation and
the thread rule is not MKT's fix, and saying so is the result."

WHY THIS IS A SEPARATE SCRIPT FROM an_ntoracle.py. That analysis reports REGRET -- time at the
rule's count over time at the best measured count -- which is the right statistic for choosing a
rule and the wrong one for this test. Regret is near 1 both when the curve is flat (every count
is as good as the best) and when a rule happens to pick the optimum of a steep curve. The
registered quantity is the SHAPE of time against worker count, and a shape cannot be recovered
from a ratio to its own minimum. Printing it separately also means the registered test cannot be
reported by whichever number came out agreeable, which is a failure this project has on record.

A WITHDRAWN CALIBRATION, FIRST, BECAUSE IT WAS THIS SCRIPT'S ORIGINAL JUSTIFICATION. The figure
"at k=2 one worker beats six by 3.06x, monotone (0.0073 / 0.0109 / 0.0154 / 0.0222 ms at
t=1/2/4/6)" labelled its arms by the OVERRIDE ARGUMENT, not by the worker count the object
actually resolved. Those are not the same thing and the gap is not small: on 231 of 576 M5 cells
every override from 1 to 6 resolves to ONE count -- 141 pinned at the policy floor and 90 pinned
at 18 by the recruit decline -- so an arm labelled "t=1" frequently ran the same team as the arm
labelled "t=6". That is the defect this whole sweep exists to remove, which is why the sweep
records `resolved` beside `nthreads`. The 3.06x figure must not be quoted and is not used below.

HOW FLAT IS DECIDED, on two bounds that do not depend on any k=2 curve:

    >= 1.20   RISING   -- team width is a real cost at k=2 on this host
    <= 1.10   FLAT     -- it is not, and the thread rule is not this host's fix
    between   NEITHER  -- reported as indeterminate, not rounded to whichever side is nearer

The lower bound is the noise: 1.10 sits above the widest per-width cell-level A/A floor median
this corpus has produced (1.02-1.61%), so FLAT cannot be returned by noise alone. The upper bound
is the thing to be explained: the measured per-cell deficit against MKL is 1.16-1.90x, and a
mechanism offered as its cause has to produce an effect at least the size of the effect. Between
them is neither -- real but too small to be the cause -- and that is reported as such.

THESE BOUNDS ARE SENSITIVE AT THE LOW END AND THE SENSITIVITY IS PRINTED. The M5 reads 1.0504 at
k=2, which is FLAT under any ceiling at or above 1.06 and NEITHER under a tighter one. The median
is therefore printed beside the verdict in every case, so a reader can apply a different ceiling
without rerunning anything.

A CELL CONTRIBUTES ONLY IF IT CAN VARY. 231 of 576 cells on the M5 resolve to one thread count
for every override, so their ratio is exactly 1.000 by construction and pooling them drags any
median toward FLAT -- which is the verdict this script could wrongly return. They are excluded
and counted, and the excluded count is printed beside the verdict so that an exclusion large
enough to empty the test is visible rather than silent.
"""
import argparse
import collections
import csv
import math
import statistics as st

RISING, FLAT = 1.20, 1.10

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="an ntoracle.py sweep")
ap.add_argument("--ks", default="", help="widths to report; default every width present")
A = ap.parse_args()

rows = list(csv.DictReader(open(A.csv)))
if not rows:
    raise SystemExit("REFUSE: %s holds no rows, so a flat verdict would be vacuous" % A.csv)

# so_sha names the object that answered. It is read rather than assumed because the whole
# reason the column exists is that an in-place build in the tree this environment imports from
# silently becomes every process's default.
shas = sorted({r.get("so_sha", "") for r in rows})
pools = sorted({r.get("pool", "") for r in rows})
print("object(s): %s     pool(s): %s" % (shas, pools))
if len(shas) > 1:
    print("REFUSE: %d distinct objects in one sweep. A curve pooled over two objects is not a"
          % len(shas))
    print("        curve; the layout artefact between objects is 1-2.6% at narrow k.")
    raise SystemExit(1)

# (key, k) -> resolved count -> [ms]
cell = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    try:
        cell[(r["key"], int(r["k"]))][int(r["resolved"])].append(float(r["ms"]))
    except (KeyError, ValueError):
        continue

# PRODUCTION'S OWN COUNT, taken from the override-0 rows, which is what a caller that sets no
# override gets. Recorded per cell rather than assumed, and noted as the one reading that is not
# reproducible across processes: `nt_cap = override > 0 ? override : omp_get_max_threads()`, so
# override 0 reads ambient OpenMP state. Within this sweep it is the right label because the
# object that answered and the kernel that was timed were in the same process.
prod_count = {}
for r in rows:
    try:
        if int(r["nthreads"]) == 0:
            prod_count[(r["key"], int(r["k"]))] = int(r["resolved"])
    except (KeyError, ValueError):
        continue

ks = ([int(x) for x in A.ks.split(",")] if A.ks
      else sorted({k for (_key, k) in cell}))

print()
print("TWO CANDIDATE STATISTICS, BOTH REPORTED, BECAUSE THEY DISAGREE AND THE CHOICE DECIDES")
print("THE VERDICT. On the pooled M5 curve the endpoint ratio reads 0.9615 (FLAT) and max/min")
print("reads 1.2362 (RISING). An endpoint ratio cannot see a non-monotone shape at all: a curve")
print("that falls 19% and rises 19% returns to its start and reads flat, which is exactly the")
print("shallow U this sweep measures. max/min sees any spread but cannot say whether the spread")
print("is a trend or one bad count. Reporting one of them alone would be choosing the answer.")
print()
print("EACH STATISTIC CARRIES ITS OWN NULL, COMPUTED THE WAY THE STATISTIC IS COMPUTED. The")
print("cell-level A/A floor (1.02-1.61%) is a TWO-reading same-count comparison, so it bounds a")
print("two-point endpoint ratio and does NOT bound an extreme over ~12 draws -- the null of an")
print("extreme widens with the number of draws. So the null here is a PERMUTATION within each")
print("cell: its readings are exchangeable among counts under the hypothesis that the count does")
print("not matter, so shuffling them and recomputing the same statistic gives that statistic's")
print("own floor on that cell's own noise. Reported as the null's median, beside the observed.")


def perm_null(counts, stat, nperm=200, seed=20261002):
    """Median of `stat` over permutations that shuffle a cell's readings among its counts."""
    import random
    flat = [x for v in counts.values() for x in v]
    sizes = [(c, len(v)) for c, v in sorted(counts.items())]
    rng = random.Random(seed)
    out = []
    for _ in range(nperm):
        rng.shuffle(flat)
        i, shuf = 0, {}
        for c, n in sizes:
            shuf[c] = flat[i:i + n]
            i += n
        v = stat(shuf)
        if v is not None:
            out.append(v)
    return st.median(out) if out else float("nan")


def stat_endpoint(counts):
    lo, hi = min(counts), max(counts)
    a, b = st.median(counts[lo]), st.median(counts[hi])
    return (b / a) if a > 0 else None


def stat_maxmin(counts):
    meds = [st.median(v) for v in counts.values() if v]
    return (max(meds) / min(meds)) if meds and min(meds) > 0 else None


def shape(counts):
    """Where the minimum sits: the registration's dichotomy presupposes monotonicity."""
    cs = sorted(counts)
    meds = [st.median(counts[c]) for c in cs]
    i = meds.index(min(meds))
    if i == 0:
        return "rising" if meds[-1] > meds[0] else "flatish"
    if i == len(meds) - 1:
        return "falling"
    return "interior-min"


STATS = (("endpoint hi/lo", stat_endpoint), ("max/min", stat_maxmin))
print()
print("%4s %6s %6s  %-15s %9s %9s %9s %9s %9s %s"
      % ("k", "live", "flat", "statistic", "median", "null", "excess", "p25", "p75", "verdict"))
verdicts = {}
for k in ks:
    cells = [(ck, c) for ck, c in cell.items() if ck[1] == k]
    live = [(ck, c) for ck, c in cells if len(c) >= 2]
    n_flat = len(cells) - len(live)
    if not live:
        print("%4d %6d %6d   NO CELL AT THIS WIDTH CAN VARY ITS WORKER COUNT, so no verdict"
              % (k, 0, n_flat))
        verdicts[k] = {name: "unable" for name, _ in STATS}
        continue
    verdicts[k] = {}
    for name, fn in STATS:
        obs = sorted(x for x in (fn(c) for _ck, c in live) if x is not None)
        nul = sorted(x for x in (perm_null(c, fn) for _ck, c in live)
                     if x == x)
        if not obs:
            verdicts[k][name] = "unable"
            continue
        med = st.median(obs)
        v = "RISING" if med >= RISING else ("FLAT" if med <= FLAT else "NEITHER")
        verdicts[k][name] = v
        nv = st.median(nul) if nul else float("nan")
        print("%4d %6d %6d  %-15s %9.4f %9.4f %8.1f%% %9.4f %9.4f %s"
              % (k, len(obs), n_flat, name, med, nv,
                 100.0 * (med / nv - 1.0) if nv == nv and nv > 0 else float("nan"),
                 obs[int(0.25 * (len(obs) - 1))], obs[int(0.75 * (len(obs) - 1))], v))

print()
print("WHERE EACH CELL'S MINIMUM SITS. The registration asks whether the curve is \"flat ... rather")
print("than rising\", a dichotomy that presupposes a monotone curve. A cell whose minimum is")
print("INTERIOR is neither, and forcing it to one side is the error both statistics above make in")
print("opposite directions. The census decides whether the registered question is even well posed")
print("on this host, so it is printed before any verdict is read.")
print("  %4s %7s %9s %9s %9s %9s" % ("k", "live", "rising", "falling", "interior", "flatish"))
for k in ks:
    live = [c for ck, c in cell.items() if ck[1] == k and len(c) >= 2]
    if not live:
        continue
    cnt = collections.Counter(shape(c) for c in live)
    print("  %4d %7d %9d %9d %9d %9d"
          % (k, len(live), cnt["rising"], cnt["falling"], cnt["interior-min"], cnt["flatish"]))

print()
print("THE QUANTITY THE DECISION ACTUALLY NEEDS, which is neither of the two above: how much")
print("better the best reachable count is than the count PRODUCTION resolves to. That is the")
print("headroom a thread rule could recover, and it is well defined whatever the shape.")
print()
print("IT CARRIES THE LARGEST SELECTION BIAS OF THE THREE AND SO IT GETS THE SAME NULL. `best`")
print("is the MINIMUM of about six noisy per-count medians, and a minimum over six draws sits")
print("below the truth even when every count is identical -- so this statistic reads above 1")
print("on pure noise, by construction, and the raw value is not the headroom. The permutation")
print("null is the winner's curse itself, measured on this cell's own noise, and the EXCESS")
print("over it is the part a thread rule could actually recover.")
print("  %4s %7s %9s %9s %9s %9s %9s" % ("k", "cells", "median", "null", "excess", "p75", "p95"))
excess_by_k = {}
for k in ks:
    gains, nulls = [], []
    for ck, c in cell.items():
        if ck[1] != k or len(c) < 2:
            continue
        prod = prod_count.get(ck)
        if prod is None or prod not in c:
            continue

        def gain(counts, _p=prod):
            if _p not in counts or not counts[_p]:
                return None
            b = min(st.median(v) for v in counts.values() if v)
            cu = st.median(counts[_p])
            return (cu / b) if b > 0 and cu > 0 else None

        g = gain(c)
        if g is None:
            continue
        gains.append(g)
        nulls.append(perm_null(c, gain))
    if not gains:
        print("  %4d       0   no cell has a reading at the count production resolves to" % k)
        continue
    gs, ns = sorted(gains), sorted(x for x in nulls if x == x)
    med, nul = st.median(gs), (st.median(ns) if ns else float("nan"))
    exc = med / nul if nul == nul and nul > 0 else float("nan")
    excess_by_k[k] = exc
    print("  %4d %7d %9.4f %9.4f %9.4f %9.4f %9.4f"
          % (k, len(gs), med, nul, exc,
             gs[int(0.75 * (len(gs) - 1))], gs[int(0.95 * (len(gs) - 1))]))
print("  'excess' is median/null. At 1.00 the apparent headroom is entirely the winner's curse")
print("  and a thread rule has nothing to recover; above 1.00 it is the recoverable part.")
print()
print("  AN EXCESS BELOW 1.00 IS NOT AN ABSENCE OF RESULT, IT IS THE SHIPPED PATH WINNING. It")
print("  says production's count beats an ARBITRARY count by (1 - excess) of the null, so the")
print("  rule is measurably better than chance rather than merely not leaving anything behind.")
print("  Printed with its sign, because reporting it as 'no headroom' inverts a positive result:")
for k in ks:
    e = excess_by_k.get(k)
    if e is None or e != e:
        continue
    if e < 1.0:
        print("    k=%-3d production beats an arbitrary count by %5.1f%% of the null (excess %.4f)"
              % (k, 100.0 * (1.0 - e), e))
    else:
        print("    k=%-3d an arbitrary count beats production by %5.1f%% -- recoverable (excess %.4f)"
              % (k, 100.0 * (e - 1.0), e))

print()
print("THE THRESHOLDS ARE FIXED IN THIS FILE, NOT CHOSEN HERE: >=%.2f RISING, <=%.2f FLAT,"
      % (RISING, FLAT))
print("between is NEITHER. The lower bound is the noise -- the widest per-width cell-level A/A")
print("floor median this corpus has produced is 1.61%, so FLAT is not reachable by noise. The")
print("upper bound is the effect to be explained: the measured per-cell deficit is 1.16-1.90x,")
print("and a proposed cause must be at least the size of the effect. The 3.06x M5 figure that")
print("once justified these bounds is WITHDRAWN -- it labelled arms by the override argument,")
print("which on 231 of 576 cells delivers one worker count for every override from 1 to 6.")

print()
print("THE CURVE ITSELF at the narrowest width, pooled as a geometric mean over cells, because a")
print("verdict on a median hides whether the shape is monotone -- and monotonicity is what the")
print("team-width account predicts and bandwidth saturation does not.")
k0 = ks[0]
bycount = collections.defaultdict(list)
for ck, counts in cell.items():
    if ck[1] != k0 or len(counts) < 2:
        continue
    base = st.median(counts[min(counts)])
    if base <= 0:
        continue
    for c, v in counts.items():
        bycount[c].append(st.median(v) / base)
print("k=%d, each count's time relative to that cell's own smallest reachable count:" % k0)
print("  %6s %7s %10s" % ("count", "cells", "geomean"))
prev, mono = None, True
for c in sorted(bycount):
    v = bycount[c]
    g = math.exp(sum(math.log(x) for x in v) / len(v))
    print("  %6d %7d %10.4f" % (c, len(v), g))
    if prev is not None and g < prev - 1e-9:
        mono = False
    prev = g
print("  monotone non-decreasing in worker count: %s" % mono)
print("  (a rising-and-monotone curve is the team-width account; a flat or U-shaped one is not,")
print("   and the counts are NOT a common support -- each cell contributes only its own")
print("   reachable counts, so a count with few cells is a different population, not a trend.)")

print()
print("REGISTERED CONCLUSION. The registration offers two outcomes and this host's measured")
print("shape is neither, so the conclusion is stated in three parts rather than forced.")
v2 = verdicts.get(2, {})
ve, vm = v2.get("endpoint hi/lo", "unable"), v2.get("max/min", "unable")
if ve == "unable" and vm == "unable":
    print("  NO VERDICT: no k=2 cell on this host can vary its worker count, so this sweep cannot")
    print("  address the registration at all. Printed because an absent verdict must not be read")
    print("  as a flat one.")
else:
    print("  1. THE TWO STATISTICS: endpoint hi/lo says %s, max/min says %s." % (ve, vm))
    if ve != vm:
        print("     They DISAGREE, so no single verdict is available from this instrument and the")
        print("     registered dichotomy cannot be settled by picking one. Each is reported above")
        print("     against its own permutation null.")
    else:
        print("     They AGREE, so the verdict does not depend on which was chosen.")
    print("  2. THE SHAPE: see the census above. The registration's words -- flat rather than")
    print("     rising -- presuppose a monotone curve, and a cell with an INTERIOR minimum is")
    print("     neither. Where interior minima dominate, the registered question is not well")
    print("     posed on this host and reporting that is the honest result.")
    print("  3. WHAT IT MEANS FOR THE THREAD RULE, which is what the registration was for: read")
    print("     the headroom table. A thread rule can only recover the gap between production's")
    print("     count and the best reachable one. If that median is at the floor, the rule has")
    print("     nothing to recover here whatever the curve's shape, and if it is well above the")
    print("     floor the rule has headroom even on a curve that reads flat end to end.")
