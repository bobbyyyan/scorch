#!/usr/bin/env python3
"""Is a within-pass bootstrap CI an adequate floor for this board? The k=2 control answers.

THE CRITERION, STATED BEFORE THE NUMBERS EXIST. an_paired_c211.py resamples KEYS within one
pass. That holds the pass fixed, so the interval it reports cannot see variation between
passes -- and a pass is a separate process on a shared host, which is a boundary an A/A over
keys does not cross. The board carries its own test of this: k=2 is a width where the arm is
routed away from the register-block kernel and provably cannot fire, so every pass's k=2
displacement is floor, and the spread of those displacements across passes is a floor
estimate that DOES cross the boundary.

  ADEQUATE    the k=2 across-pass spread sits within the typical within-pass half-width. The
              per-pass intervals already contain the pass-level variation and may be read as
              published.
  UNDERSTATED the spread exceeds it. Then every per-pass interval understates the floor by
              the measured ratio, and a per-pass interval is readable only for effects larger
              than the across-pass spread. The ratio is the number to publish, not a verdict.

Registered consequence either way: this cannot change an arm verdict whose effect exceeds
the across-pass spread by a wide margin, and it is not licence to re-read a verdict it
contradicts. It bounds what the INTERVALS mean, and nothing else.

Usage:  an_c211_acrosspass.py <dir-with-dl_*_float32_p*.csv> [pass-tag ...]
"""
import csv
import os
import statistics as st
import sys

KS = (2, 4, 8, 16, 32)
ARM, FLOOR = "deep", "bsb_"

if len(sys.argv) < 2:
    sys.exit(__doc__)
D = sys.argv[1]
TAGS = sys.argv[2:] or ["p1", "p2", "p3", "p4", "p5", "p6"]


def load(obj, tag):
    p = os.path.join(D, "dl_%s_float32_%s.csv" % (obj, tag))
    if not os.path.exists(p):
        return None
    out = {}
    with open(p) as h:
        for r in csv.DictReader(h):
            if r["arm"] != "ours":
                continue
            key = (r["key"], int(r["k"]))
            if key in out:
                sys.exit("REFUSE: %s repeats cell %s for arm ours" % (p, key))
            out[key] = float(r["ms"])
    return out


def median_ratio(a, f, k):
    """The same estimand an_paired_c211 reports: median of the paired per-cell ratio."""
    vals = [a[(key, kk)] / f[(key, kk)] for (key, kk) in a
            if kk == k and (key, kk) in f and a[(key, kk)] > 0 and f[(key, kk)] > 0]
    return (st.median(vals), len(vals)) if vals else (None, 0)


rows, missing = {}, []
for tag in TAGS:
    a, f = load(ARM, tag), load(FLOOR, tag)
    if a is None or f is None:
        missing.append(tag)
        continue
    for k in KS:
        m, n = median_ratio(a, f, k)
        if m is not None:
            rows.setdefault(k, []).append((tag, 100 * (m - 1), n))

if missing:
    print("passes ABSENT and excluded: %s" % ", ".join(missing))
present = sorted(set(t for v in rows.values() for t, _d, _n in v))
print("passes read: %d (%s)" % (len(present), ", ".join(present)))
if len(present) < 3:
    sys.exit("REFUSE: %d pass(es) is not a spread -- at least 3 are needed before an "
             "across-pass range says anything about the floor" % len(present))

print()
print("PER-PASS displacement of %s/%s, in percent. k=2 is the row that is floor by "
      "construction." % (ARM, FLOOR))
print("  %-4s %-7s %s" % ("k", "n/pass", "  ".join("%7s" % t for t in present)))
for k in KS:
    v = dict((t, d) for t, d, _n in rows.get(k, []))
    ns = set(n for _t, _d, n in rows.get(k, []))
    print("  %-4d %-7s %s" % (k, ns.pop() if len(ns) == 1 else "VARIES",
                              "  ".join("%+7.2f" % v[t] if t in v else "      -"
                                        for t in present)))

print()
print("ACROSS-PASS spread per width, against the within-pass half-width an_paired_c211 reports.")
print("  The k=2 half-width is ~0.20 pp on every pass of this board; it is quoted from that")
print("  report rather than recomputed here, so this file does not restate its estimator.")
HW_K2 = 0.20
sp = {}
for k in KS:
    d = [x for _t, x, _n in rows.get(k, [])]
    if len(d) < 3:
        continue
    sp[k] = (max(d) - min(d), st.pstdev(d), min(d), max(d))
    print("  k=%-3d spread %5.2f pp  sd %5.2f pp  range [%+6.2f, %+6.2f]"
          % (k, sp[k][0], sp[k][1], sp[k][2], sp[k][3]))

print()
if 2 not in sp:
    sys.exit("REFUSE: no k=2 row, so the board's own floor width is absent and nothing here "
             "is a floor statement")
s2 = sp[2][0]
ratio = s2 / HW_K2
verdict = "ADEQUATE" if s2 <= HW_K2 else "UNDERSTATED"
print("VERDICT %s. The k=2 across-pass spread is %.2f pp against a within-pass half-width of"
      % (verdict, s2))
print("  %.2f pp, a factor of %.1f. k=2 is the width where the arm provably cannot fire, so"
      % (HW_K2, ratio))
print("  that spread is floor and not effect.")
if verdict == "UNDERSTATED":
    print("  CONSEQUENCE, as registered: every per-pass interval on this board understates the")
    print("  floor by about that factor, and a per-pass interval is readable only for effects")
    print("  larger than %.2f pp. It does not move any arm verdict here -- the smallest arm" % s2)
    sm = min(min(abs(x) for _t, x, _n in rows[k]) for k in KS if k != 2 and k in rows)
    print("  effect at any width and pass is %.2f pp, which is %.0fx the spread." % (sm, sm / s2))
    print("  What it does retire is reading a single pass's interval at the 1 pp scale.")
print()
# p4 is the lowest reading at EVERY width and p2 the highest at most of them, which looks
# like a single multiplicative pass-level factor. k=2 is a floor read on the SAME pass, so
# subtracting it is the direct test of that story -- and a plausible mechanical cause has to
# be measured rather than named, because the last two times one was named it was wrong.
print("IS IT ONE PASS-LEVEL FACTOR? k=2 is floor on the same pass, so subtracting it removes a")
print("  common multiplicative term if there is one. Reported per width, because a factor that")
print("  explains some widths and not others is not that factor.")
k2 = dict((t, d) for t, d, _n in rows.get(2, []))
helped = []
for k in KS:
    if k == 2 or k not in sp:
        continue
    raw = [x for _t, x, _n in rows[k]]
    adj = [x - k2[t] for t, x, _n in rows[k] if t in k2]
    if len(adj) != len(raw):
        continue
    r, a_ = max(raw) - min(raw), max(adj) - min(adj)
    helped.append(a_ < r)
    print("  k=%-3d spread %5.2f pp raw -> %5.2f pp after subtracting that pass's k=2  (%s)"
          % (k, r, a_, "narrower" if a_ < r else "no narrower"))
if all(helped):
    print("  Every width narrows, so one pass-level term is consistent with all of them and the")
    print("  k=2-adjusted spread is the tighter floor.")
elif any(helped):
    print("  MIXED: %d of %d widths narrow. A term that explains some widths and not others is"
          % (sum(helped), len(helped)))
    print("  not a single pass-level factor, so it is NOT adopted and the unadjusted spread")
    print("  above stands as the floor. The table is the honest form of this.")
else:
    print("  No width narrows, so the pass-level story is not supported at all.")

print()
print("WHAT THIS IS NOT: a positive control. It bounds how wide the floor is, not whether the")
print("  instrument can see an effect of a given size. Those are different questions and this")
print("  board still has no injected effect.")
