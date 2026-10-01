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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfs_rowguard as _rowguard
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
# Everything compared against a displacement is compared at the precision it is PRINTED at.
# A guard tighter than that lets a value print as zero beside a number derived by dividing
# by it. `%.2f` of a percentage rounds at 0.005.
PRINTED_ZERO = 0.005


def _sign(pp):
    """-1 / 0 / +1 at the PRINTED precision. One place, so no caller can treat the zero case
    on one line and omit it on the next."""
    if abs(pp) < PRINTED_ZERO:
        return 0
    return 1 if pp > 0 else -1


def sign_argument(arm_pps, ctl_pp):
    """The SIGN paragraph as a pure function of (arm displacements, control displacement).

    Extracted because the block was module-level straight-line code with no selftest, and the
    defect it replaced was a sentence true only on the pass it was written for. An unexercised
    branch is the same thing with a condition in front of it. -> (token, arm_word, ctl_word).
    """
    signs = set(_sign(x) for x in arm_pps)
    signs.discard(0)
    arm_word = ("SLOWER" if signs == {1} else "FASTER" if signs == {-1}
                else "NO EFFECT AT ANY WIDTH" if not signs else "MIXED")
    cs = _sign(ctl_pp)
    ctl_word = "FASTER" if cs < 0 else "SLOWER" if cs > 0 else "ZERO"
    if cs == 0:
        return "CONTROL_ZERO", arm_word, ctl_word
    if arm_word in ("MIXED", "NO EFFECT AT ANY WIDTH"):
        return "ARM_HAS_NO_SINGLE_SIGN", arm_word, ctl_word
    if (cs == 1) != (signs == {1}):
        return "OPPOSITE_SIGN_AVAILABLE", arm_word, ctl_word
    return "SAME_SIGN_NOT_AVAILABLE", arm_word, ctl_word


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

# THE CALL SITE. pfs_rowguard was written for a partial-write that produced plausible medians
# on a quarter of the corpus, and it shipped with no callers -- the only references to it in
# the tree were in its own selftest, so the defect it was written for would have been read
# exactly as before. Called here, before any estimate is formed, because line 113 below
# intersects the arms silently: a shrunken intersection reaches the bootstrap as a smaller n
# and printing n does not help a reader who does not already know the expected value, which
# is the condition that failed the first time.
try:
    _cen = _rowguard.check({ARM: A, FLOOR: F, DEN: B})
except _rowguard.Incomplete as _e:
    # Structured, not a traceback. The house rule is that an unsupported shape exits with a
    # structured defect code and never an unstructured exception, and a traceback from inside
    # an imported guard is exactly that -- it also buries the sentence that names the file to
    # wait for under a stack.
    sys.exit("REFUSE: %s" % _e)
print("row guard: the three arms cover the same cells -- %s"
      % ", ".join("k=%d n=%d" % (k, _cen[k]) for k in sorted(_cen)))

print()
print("PAIRED CONTRAST %s/%s per width -- base cancels, so this is the arm against its own floor."
      % (ARM, FLOOR))
print("  %-4s %8s %24s %14s %7s  %s"
      % ("k", "n", "deep/bsb_  [95% CI]", "contains 1?", "n<1/boot", "index-matrix sha256"))
verdicts, ci, MACHINE = {}, {}, []
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
    # THE COUNT, not a fraction. The estimate's resolution is 1/NBOOT, so the smallest nonzero
    # value this can take is 0.0005 and "0.000" means "0 of 2000" -- p < 1/2000 and nothing finer.
    # A decimal hides that resolution, reads as an exact zero probability, and keeps the resample
    # count out of sight at exactly the point where a reader would ask whether it was enough.
    # The display's floor and the estimate's floor are different quantities, and the display is the
    # one being read.
    nlt = int(np.count_nonzero(meds < 0.0))
    verdicts[k] = (inside, nlt, point)
    ci[k] = (lo, hi)
    # Emitted so a downstream analysis READS this half-width rather than transcribing it to
    # two significant figures. A hand-copied 0.20 bounded a published factor's precision by
    # its own rounding (6.1 at 0.25, 10.1 at 0.15), and a scalar also cannot carry the
    # per-pass per-width variation that a uniformity claim about it needs.
    MACHINE.append("HALFWIDTH k=%d pass=%s halfwidth_pp=%.6f point_pp=%.6f n=%d"
                   % (k, TAG, 100 * (hi - lo) / 2.0, 100 * (point - 1), n))
    print("  %-4d %8d %8.4f [%.4f,%.4f] %14s %9s  %s"
          % (k, n, point, lo, hi, "CONTAINS 1" if inside else "excludes 1",
             "%d/%d" % (nlt, NBOOT), dig))

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
    _, _nlt2, point2 = verdicts[2]   # the count is in the table; the control reports a magnitude
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
    # c211 k=2 p4 landed at exactly 1.0000 and divided by itself here, crashing the reporter
    # with ZeroDivisionError after the interval had printed. An exactly-zero displacement is
    # NOT "perfect agreement" -- that reading was wrong and the mechanism is order-statistic
    # tie placement. In that cell the ratios take 3818 distinct values over 3820 cells, so the
    # data is not coarse; exactly three cells have arm == floor bit-for-bit and they sit at
    # ranks 1909, 1910 and 1911, while the median at n=3820 is the mean of ranks 1910 and 1911
    # -- both inside that run. The neighbours are -0.0093 pp at rank 1908 and +0.0067 pp at
    # 1912, so the honest characterisation is |displacement| < 0.01 pp and the exact zero is
    # cosmetic. The ratio is then unbounded above, which is a statement and not a number, so
    # the interval bound below is used: it always exists and is the conservative form anyway.
    #
    # THIS BRANCH'S REACHABILITY IS A PROPERTY OF THE MEDIAN. One of thirty (width, pass)
    # cells on that board reaches it, and under a geometric or trimmed mean nothing would land
    # on exactly 1.0, so the branch and its plant would go from live to dead with nothing
    # failing. an_paired_c211_selftest.py asserts the estimand is still an order statistic for
    # that reason.
    # PRINTED_ZERO, not 1e-9. The value is printed `%.2f%%`, so anything under 0.005 prints
    # as 0.00% -- and the old 1e-9 guard let a displacement of 0.001 print "0.00%" beside a
    # finite ratio with a 1e-3 divisor, inflating it 1000x against a 1.00 displacement. The
    # crash was fixed; the misleading-number case the comment was reasoning about was still
    # reachable across six decades. `<=` because exactly 0.005 rounds to 0.01 but exactly
    # 1e-9 must not take the number branch.
    if abs(disp) < PRINTED_ZERO:
        print("    vs the point displacement %.2f%%   : the point is zero to the printed "
              "precision, so this ratio is unbounded above and is not a number. The "
              "interval bound below is the form that always exists." % abs(disp),
              derived=True)
    else:
        print("    vs the point displacement %.2f%%   : %.0fx to %.0fx smaller"
              % (abs(disp), min(eff) / abs(disp), max(eff) / abs(disp)), derived=True)
    print("    vs the interval bound    %.2f%%   : %.0fx to %.0fx smaller   <- the conservative form"
          % (bound, min(eff) / bound, max(eff) / bound), derived=True)
    print("  This bounds FALSE POSITIVES. It does NOT bound false negatives; that needs a positive")
    print("  control, which injects a known effect, and this board has none.")
    # THIS WAS A CONSTANT SENTENCE ASSERTING A MEASURED SIGN. It read "this reads FASTER, so
    # the displacement has the wrong sign to be the arm" on every pass, including c211 p2/p3/p5
    # where the control's displacement is +1.34% / +0.51% / +0.71% -- the SAME direction as the
    # arm, so the argument it claimed was exactly unavailable. The sign now comes from the
    # data, and when it agrees with the arm the report says the argument is NOT available
    # rather than making it anyway.
    # An arm point ratio of exactly 1.0 -- no effect at all -- used to bin as -1, FASTER,
    # which would flip arm_signs off {1} and route the report to "the arm's direction is
    # MIXED, only the magnitude argument is available" on the strength of a width where
    # nothing happened. The control's sign got a zero case on the very next line and the
    # arm's did not, inside four lines.
    # ONE implementation, in sign_argument(). The inline version of this block was
    # module-level straight-line code with no test over its branches, which is the same
    # defect as the constant sentence it replaced, wearing a condition.
    tok, arm_word, ctl_word = sign_argument(
        [100 * (verdicts[k][2] - 1) for k in KS if k != 2], disp)
    if tok == "CONTROL_ZERO":
        print("  SIGN: the control's point displacement is zero to the printed precision, so "
              "there is no sign to compare. Only the magnitude argument above is available.")
    elif tok == "ARM_HAS_NO_SINGLE_SIGN":
        print("  SIGN: the arm's own direction is %s, so there is no single arm sign for the "
              "control to disagree with. Only the magnitude argument above is available."
              % arm_word)
    elif tok == "OPPOSITE_SIGN_AVAILABLE":
        print("  SIGN: the arm is %s at every width where it can fire and the control reads "
              "%s, so the displacement has the wrong sign to be the arm. That argument needs "
              "no artefact range." % (arm_word, ctl_word))
    else:
        print("  SIGN: NOT AVAILABLE. The arm is %s at every width where it can fire and the "
              "control reads %s too -- the SAME direction. The sign argument is therefore not "
              "available on this pass and only the magnitude argument above stands."
              % (arm_word, ctl_word))

# A SURVIVING-CONVERSION-SPECIFIER SCAN. Twice in one session a format string reached output
# unsubstituted -- a literal `%+.0f s`, and a doubled `%%` printed outside a format operation -- and
# both times in a line added to EXPLAIN a number rather than to compute one. Explanatory prints are
# written last, are exercised by no check, and fail in a way that looks cosmetic, so they are a
# population a scan covers and review does not.
#
# THE SPACE FLAG IS DELIBERATELY EXCLUDED. `%[-+ #0]*...` -- the textbook form -- false-positives on
# ordinary prose here: "against 0.3% identical-source" reads as `% i` and "95.9% of resampled" as
# `% o`, both legitimate percent signs followed by a word. Dropping the space flag costs the ability
# to catch a real `% d`, which is rare, and buys zero false positives on this harness's own output.
# `%%` is added as a separate alternative because it matches no conversion class.
_SPEC = re.compile(r"%[-+#0]*[0-9]*\.?[0-9]*[sdfgeiouxX]|%%")
_bad_fmt = [(i, l) for i, (l, _d) in enumerate(OUT) if _SPEC.search(l)]
if _bad_fmt:
    for _i, _l in _bad_fmt:
        _real_print("  line %d: %s" % (_i, _l.strip()[:96]))
    sys.exit("REFUSE: %d printed line(s) carry an unsubstituted conversion specifier, so a number "
             "this report claims to state was never formatted into it." % len(_bad_fmt))

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
# string scan reports "all distinct" on it, so every number is also offered under
# `x -> 1 + x / 100`, which READS A CANDIDATE PERCENTAGE BACK AS A RATIO. The direction matters and
# naming it badly cost a peer a wrong inference: the transform DIVIDES, so its slope is 1/100 and
# it compresses by a hundred. Applying it to a number that is already a ratio is the spurious case,
# and that is exactly what produced the four-decimal collision described below.
# SIX decimals, not four, and matches must come from DIFFERENT lines. At four decimals the
# read-a-percentage-back-as-a-ratio form collided 1.0772 with 1.0813, the interval endpoint and the
# point estimate on the SAME row: 1 + 1.0772/100 = 1.010772 and 1 + 1.0813/100 = 1.010813 both
# round to 1.0108, because an input gap of 0.0041 compresses to 0.000041, under the 0.00005 that
# four decimals can resolve. So the scan invented a duplicate out of its own rounding. Six decimals
# separates them, and the different-line requirement is the real predicate anyway: a leaked loop
# variable duplicates a value ACROSS rows, and a rounding collision does not.
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
    print("  (Values are compared as written AND as `1 + x/100`, so a percentage matches the ratio")
    print("   it restates.) Exact repetition of a continuous statistic across rows is the signature")
    print("  of a leaked loop variable or a copied cell. Inspect each before quoting. Some repeats")
    print("  are legitimate (two widths sharing a floor), so this REPORTS rather than refuses.")
else:
    print("INTERNAL-DUPLICATE SCAN over the printed report: no value repeats across lines, either")
    print("  as written or read back as a ratio. %d lines marked `derived` are"
          % sum(1 for _, _d in OUT if _d))
    print("  skipped -- they restate a table value in percent form on purpose.")

# MACHINE-READABLE TAIL, after the duplicate scan so it is outside the prose it would otherwise
# trip. These lines exist so a downstream analysis reads this estimator's own half-widths rather
# than transcribing them; `_real_print` because the duplicate scan has already run and these
# deliberately restate table values.
for _m in MACHINE:
    _real_print(_m)
