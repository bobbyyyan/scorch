#!/usr/bin/env python3
"""The paired arm-vs-floor contrast for a three-object board, with a SEALED bootstrap index matrix.

TWO METHOD POINTS, both of which changed the answer's standing rather than its sign.

THE CONTRAST IS PAIRED, because the control depends on it. Comparing two separately-bootstrapped
marginal intervals -- one for deep/base, one for bsb_/base -- is conservative by an unknown factor:
non-overlap implies a difference at a stricter level than 0.05, and overlap implies NOTHING about
absence. On a width where the effect is large that conservatism is free. On the NEGATIVE CONTROL it
is not, because an under-powered test passes too easily, and passing too easily is the one failure
mode a control cannot have. Both arms are measured on the same keys against the same denominator,
so the paired per-cell contrast is log(deep/base) - log(bsb_/base) = log(deep/bsb_): base's
per-cell noise cancels exactly instead of being carried twice. The estimand is unchanged; this is
the contrast the claim is about.

THE INTERVAL IS PINNED BY AN INDEX MATRIX, not by a seed. A seed plus a resample count does not
reproduce a bootstrap bound -- the draws depend on the RNG implementation, on the row order the
resampler sees, and on whatever consumed the stream first. So the index matrix is built
explicitly, digested, and the digest is printed beside every interval. The serialisation is named
with the digest, because a digest without its serialisation emits only false negatives: int32,
C order, shape (NBOOT, n), little-endian, keys sorted lexicographically before indexing.
"""
import csv, hashlib, math, os, re, sys
import numpy as np

OUT = []
_real_print = print


def print(*a, derived=False, **kw):   # noqa: A001 -- tee, so the duplicate scan reads the REPORT
    """`derived=True` marks a line that INTENTIONALLY restates a table value in another form.

    The duplicate scan skips those. The flag lives at the PRODUCER rather than in a list of
    expected strings kept beside the scan, because a typed list can only excuse what its author
    thought of, and it rots silently when a message is reworded. It also fails in the safe
    direction: forgetting the flag adds noise, it does not hide a repeat.
    """
    line = " ".join(str(x) for x in a)
    OUT.append((line, derived))
    _real_print(line, **kw)

if len(sys.argv) < 2:
    sys.exit("usage: an_paired_c211.py <dir-with-dl_*_float32_p*.csv> [pass-tag]")
D = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "p1"
KS = (2, 4, 8, 16, 32)
NBOOT = 2000
SEED = 20261001
ARM, FLOOR, DEN = "deep", "bsb_", "base"


def load(obj):
    p = os.path.join(D, "dl_%s_float32_%s.csv" % (obj, TAG))
    if not os.path.exists(p):
        sys.exit("REFUSE: %s is absent" % p)
    out = {}
    with open(p) as h:
        for r in csv.DictReader(h):
            if r["arm"] != "ours":
                continue
            k = (r["key"], int(r["k"]))
            if k in out:
                sys.exit("REFUSE: %s repeats cell %s for arm ours" % (p, k))
            out[k] = float(r["ms"])
    return out


A, F, B = load(ARM), load(FLOOR), load(DEN)
print("loaded %s/%s/%s: %d / %d / %d cells at arm=ours" % (ARM, FLOOR, DEN, len(A), len(F), len(B)))

print()
print("PAIRED CONTRAST %s/%s per width -- base cancels, so this is the arm against its own floor."
      % (ARM, FLOOR))
print("  %-4s %8s %24s %14s %7s  %s"
      % ("k", "n", "deep/bsb_  [95% CI]", "contains 1?", "frac<1", "index-matrix sha256"))
verdicts, ci = {}, {}
for k in KS:
    keys = sorted(key for (key, kk) in A if kk == k)
    keys = [key for key in keys if (key, k) in F and A[(key, k)] > 0 and F[(key, k)] > 0]
    if not keys:
        sys.exit("REFUSE: no paired cells at k=%d" % k)
    v = np.array([math.log(A[(key, k)] / F[(key, k)]) for key in keys], dtype=np.float64)
    n = len(v)
    # The index matrix, built explicitly and digested. Width is in the seed so the five matrices
    # are distinct and each is reproducible on its own.
    rng = np.random.Generator(np.random.PCG64(SEED + k))
    idx = rng.integers(0, n, size=(NBOOT, n), dtype=np.int32)
    dig = hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest()[:24]
    meds = np.median(v[idx], axis=1)
    lo, hi = np.exp(np.percentile(meds, [2.5, 97.5]))
    point = math.exp(float(np.median(v)))
    inside = lo <= 1.0 <= hi
    # WHERE 1 SITS IN THE INTERVAL, not merely whether it is in it. A paired test is tight enough
    # that "contains 1" can be a bare pass, and a control that passes at the edge is a different
    # statement from one that passes in the middle. Reported as the fraction of resampled medians
    # below 1, which is the one-sided p against the null directly.
    frac = float(np.mean(meds < 0.0))
    verdicts[k] = (inside, frac, point)
    ci[k] = (lo, hi)
    print("  %-4d %8d %8.4f [%.4f,%.4f] %14s  %5.3f  %s"
          % (k, n, point, lo, hi, "CONTAINS 1" if inside else "excludes 1", frac, dig))

print()
print("Serialisation of each digest: int32, C order, shape (%d, n), little-endian; keys sorted"
      % NBOOT)
print("lexicographically before indexing; np.random.PCG64(%d + k). A rerun whose digest differs"
      % SEED)
print("is not reproducing these intervals and its numbers do not substitute for them.")
print()
# k=2 is the NEGATIVE CONTROL: the arm is routed away from regblock there, so it cannot fire.
#
# IT REPORTS A MAGNITUDE, NOT A VERDICT AGAINST 1. Testing a control's displacement for
# "contains 1" makes the verdict depend on the interval width, so a NOISIER control passes more
# easily -- the same inversion as using two conservative marginal intervals, one layer down and
# now in the criterion rather than the instrument. An earlier version of this block printed
# "CONTAINS 1" and then called the pass marginal because 95.9% of resampled medians sat below 1.
# That describes the distance from a reference point the control does not pose.
#
# AND THE MAGNITUDE IS NOT EXPLAINED BY CITING A PRIOR RUN'S ARTEFACT RANGE. A previous version
# attributed it to chain185's 1-2.6% cross-object spread. That is 37 SuiteSparse matrices on
# another host, a hundredth of this collection, and the citation would import a magnitude from an
# unmatched population -- inflating the floor from the 0.18% measured HERE to 2.6% on nothing this
# run supports. It is also nearly vacuous as an explanation: a 1-2.6% range predicts five to
# fourteen times MORE than observed, and almost any small number is "consistent with" it. This
# cell IS that artefact, measured directly on these objects at n=3820 with an interval, which is
# what chain185's own lesson says to do.
if 2 in verdicts:
    _, frac2, point2 = verdicts[2]
    lo2, hi2 = ci[2]
    disp = 100 * (point2 - 1)
    band = (100 * (lo2 - 1), 100 * (hi2 - 1))
    bound = max(abs(band[0]), abs(band[1]))
    eff = [abs(100 * (verdicts[k][2] - 1)) for k in KS if k != 2]
    print("NEGATIVE CONTROL k=2, reported as a magnitude:")
    print("  displacement %+.2f%%, interval [%+.2f%%, %+.2f%%], at a width where the arm is routed"
          % (disp, band[0], band[1]), derived=True)
    print("  away from the register-block kernel and provably cannot fire -- same %d keys, same"
          % len(keys))
    print("  calibration, same passes as every effect above.")
    print("  Against the effects it has to be small relative to (%.2f%% to %.2f%%):"
          % (min(eff), max(eff)), derived=True)
    print("    vs the point displacement %.2f%%   : %.0fx to %.0fx smaller"
          % (abs(disp), min(eff) / abs(disp), max(eff) / abs(disp)), derived=True)
    print("    vs the interval bound    %.2f%%   : %.0fx to %.0fx smaller   <- the conservative form"
          % (bound, min(eff) / bound, max(eff) / bound), derived=True)
    print("  This bounds FALSE POSITIVES. It does NOT bound false negatives; that needs a positive")
    print("  control, which injects a known effect, and this board has none.")
    print("  SIGN: the arm is SLOWER at every width where it can fire and this reads FASTER, so the")
    print("  displacement has the wrong sign to be the arm. That argument needs no artefact range.")

# AN INTERNAL-DUPLICATE SCAN OVER THE PRINTED OUTPUT, which is the only version that catches the
# bug it is named for. A leaked loop variable produces a number that is plausible in isolation and
# DUPLICATED IN CONTEXT: this report printed k=32's 8.13% as k=2's displacement, and the tell was
# that it equalled another row.
#
# A first version of this scan compared the STORED statistics and reported "all distinct" -- which
# it would have done on the buggy run too, because the leaked value was computed at print time from
# a variable that had fallen out of the loop and was never stored under k=2. So the scan has to run
# over the TEXT, which is also the cheaper check: one pass over output, no review of the code that
# produced it.
print()
# NORMALISED ACROSS REPRESENTATIONS, because the historical bug printed 8.13% where the table held
# 1.0813 -- the SAME value in ratio and percent form, and two different decimal STRINGS. A literal
# string scan reports "all distinct" on it, so the scan is keyed on the ratio form and every number
# is also offered under the percent-of-excess transform x -> 1 + x/100.
# SIX decimals, not four, and matches must come from DIFFERENT lines. At four the percent
# transform collided 1.0772 with 1.0813 -- both map to 1.0108 -- which are the interval endpoint
# and the point estimate on the SAME row, so the scan invented a duplicate out of its own rounding.
# Six separates them (1.010772 vs 1.010813), and the different-line requirement is the real
# predicate anyway: a leaked loop variable duplicates a value ACROSS rows, so a same-line repeat is
# never the signal.
def _canon(x):
    return round(x, 6)


_nums = {}
for _i, (_ln, _dv) in enumerate(OUT):
    if _dv:
        continue
    for _m in re.finditer(r"(?<![\w.])[-+]?\d+\.\d+(?![\w])", _ln):
        _x = float(_m.group(0))
        for _form in {_canon(_x), _canon(1.0 + _x / 100.0)}:
            _nums.setdefault(_form, []).append((_i, _m.group(0), _ln.strip()[:56]))
_dups = {}
for _v, _w in _nums.items():
    if len({_e[0] for _e in _w}) > 1:          # must span at least two distinct lines
        _dups[_v] = sorted({"%s  [line %d: %s]" % (_e[1], _e[0], _e[2]) for _e in _w})
if _dups:
    print("INTERNAL-DUPLICATE SCAN over the printed report: %d repeated value(s)." % len(_dups))
    for v, w in sorted(_dups.items(), key=lambda kv: -len(kv[1])):
        print("    ratio form %.4f appears %d times:" % (v, len(w)))
        for _l in w:
            print("        %s" % _l)
    print("  Exact repetition of a continuous statistic across rows is the signature of a leaked")
    print("  loop variable or a copied cell. Inspect each before quoting it. Some repeats are")
    print("  legitimate (two widths sharing a floor), so this REPORTS rather than refuses.")
else:
    print("INTERNAL-DUPLICATE SCAN over the printed report: no value repeats across lines under")
    print("  either the ratio form or the percent-of-excess form. %d lines marked `derived` are"
          % sum(1 for _, _d in OUT if _d))
    print("  skipped -- they restate a table value in percent form on purpose.")
