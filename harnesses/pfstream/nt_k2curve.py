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
    print("        curve; the layout artefact between objects is 1-2.6%% at narrow k.")
    raise SystemExit(1)

# (key, k) -> resolved count -> [ms]
cell = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    try:
        cell[(r["key"], int(r["k"]))][int(r["resolved"])].append(float(r["ms"]))
    except (KeyError, ValueError):
        continue

ks = ([int(x) for x in A.ks.split(",")] if A.ks
      else sorted({k for (_key, k) in cell}))

print()
print("TIME AGAINST RESOLVED WORKER COUNT. The per-cell ratio is time(largest reachable count)")
print("over time(smallest reachable count), each the median of its repeats.")
print("%4s %6s %6s %9s %9s %9s %9s %s"
      % ("k", "live", "flat", "median", "p25", "p75", "max", "verdict"))

verdicts = {}
for k in ks:
    cells = [(ck, counts) for ck, counts in cell.items() if ck[1] == k]
    ratios, n_flat = [], 0
    for _ck, counts in cells:
        if len(counts) < 2:
            n_flat += 1
            continue
        lo, hi = min(counts), max(counts)
        t_lo, t_hi = st.median(counts[lo]), st.median(counts[hi])
        if t_lo > 0:
            ratios.append(t_hi / t_lo)
    if not ratios:
        print("%4d %6d %6d   NO CELL AT THIS WIDTH CAN VARY ITS WORKER COUNT, so no verdict"
              % (k, 0, n_flat))
        verdicts[k] = "unable"
        continue
    rs = sorted(ratios)
    med = st.median(rs)
    v = "RISING" if med >= RISING else ("FLAT" if med <= FLAT else "NEITHER")
    verdicts[k] = v
    print("%4d %6d %6d %9.4f %9.4f %9.4f %9.4f %s"
          % (k, len(rs), n_flat, med,
             rs[int(0.25 * (len(rs) - 1))], rs[int(0.75 * (len(rs) - 1))], rs[-1], v))

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
print("REGISTERED CONCLUSION, stated as the registration requires:")
v2 = verdicts.get(2, "unable")
if v2 == "FLAT":
    print("  k=2 is FLAT in worker count on this host. The deficit here is NOT team width, the")
    print("  thread rule is not this host's fix, and that is the result rather than a setback.")
elif v2 == "RISING":
    print("  k=2 RISES with worker count on this host, so team width is a real cost at k=2 here")
    print("  and the thread rule is a candidate. It is not thereby shown to CLOSE the deficit:")
    print("  that needs the regret table in an_ntoracle.py and a measured ours/mkl.")
elif v2 == "unable":
    print("  NO VERDICT: no k=2 cell on this host can vary its worker count, so this sweep")
    print("  cannot address the registration at all. Printed because an absent verdict must")
    print("  not be read as a flat one.")
else:
    print("  k=2 is INDETERMINATE on this host: between the registered thresholds. Reported as")
    print("  indeterminate rather than rounded to the nearer side.")
