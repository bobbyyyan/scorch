#!/usr/bin/env python3
"""Does pass-to-pass variation exceed what the within-pass bootstrap already measures?

THE CRITERION, AND ITS NULL VALUE, BOTH STATED BEFORE THE NUMBERS. an_paired_c211.py
resamples KEYS inside one pass, which holds the pass fixed -- and a pass is a separate
process on a shared host, a boundary an A/A over keys does not cross. The board carries its
own test, because k=2 is a width where the arm is routed away from the register-block kernel
and provably cannot fire, so every pass's k=2 displacement is floor.

  ADEQUATE    the across-pass scatter is consistent with within-pass sampling alone. Then
              the per-pass intervals already contain it and may be read as published.
  UNDERSTATED it is larger. The number to publish is the ratio of the across-pass SD to the
              within-pass SE, beside its NULL value, plus the implied pass-level SD.

THE FIRST VERSION OF THIS FILE WAS SELF-VOIDING AND IS THE REASON THE NULL IS NOW STATED.
It compared `max - min` over the passes, a RANGE, against a 95% bootstrap HALF-WIDTH. Those
are different quantities: at the null each pass's median is a draw with SE = HW/1.96, so for
n=6 the expected range is 2.538 SE = 1.295 HW. `ADEQUATE` required range <= HW, which a
PERFECTLY ADEQUATE floor fails 73.7% of the time -- so one branch was nearly unreachable and
a verdict one branch cannot reach is not a test. Simulated here at 200000 draws, seed 20261001.

That is the same defect as the f32 screen's positive control, which injected an effect AT the
bound and therefore detected it 2.5% of the time, and it was committed to pfs_criterion.py as
a refusal four commits before this file repeated it. A criterion must report where its own
statistic sits under no effect, which is why `null_check` below runs every time rather than
being a comment.

The comparison is now like-for-like and standard: s_obs over the passes estimates
sqrt(sigma_pass^2 + SE_within^2), so under sigma_pass = 0 the variance ratio
(n-1) s_obs^2 / SE_within^2 is chi-square on n-1 df. The sample SD uses ddof=1; `pstdev`
treats six passes as the population and understates sigma by 8.7% at n=6 (measured below),
which was cosmetic while the verdict used the range and is load-bearing now.

Half-widths are READ from an_paired_c211's own `HALFWIDTH` lines, not transcribed. A
hand-copied two-significant-figure 0.20 bounded the published factor's precision by its own
rounding, and a scalar cannot carry the per-pass per-width variation a uniformity claim needs.

Registered consequence either way: this bounds what the INTERVALS mean and nothing else. It
cannot move an arm verdict whose effect exceeds the across-pass scatter by a wide margin, and
it is not licence to re-read a verdict it contradicts.

Usage:  an_c211_acrosspass.py <dir-with-HALFWIDTH-lines> [file ...]
        Each file is an an_paired_c211.py report for one pass. With none, every
        an_c211_p*.txt in the directory is read.
"""
import glob as globmod
import math
import os
import random
import re
import statistics as st
import sys

KS = (2, 4, 8, 16, 32)
CI_Z = 1.959963984540054
ALPHA = 0.05
NSIM = 200000
SEED = 20261001
HW_RX = re.compile(r"^HALFWIDTH k=(\d+) pass=(\S+) halfwidth_pp=([-\d.]+) point_pp=([-\d.]+) "
                   r"n=(\d+)\s*$")

if len(sys.argv) < 2:
    sys.exit(__doc__)
D = sys.argv[1]
FILES = sys.argv[2:] or sorted(globmod.glob(os.path.join(D, "an_c211_p*.txt")))


def null_check():
    """Where the statistic sits under NO pass-level effect. Printed every run, by construction.

    Two statistics: the range-over-half-width the first version of this file used, and the
    SD-over-SE the verdict now uses. The first is reported so the correction is visible rather
    than silently substituted.
    """
    random.seed(SEED)
    out = {}
    for n in (3, 4, 5, 6):
        rng, sd, psd = [], [], []
        for _ in range(NSIM // 4):
            d = [random.gauss(0, 1) for _ in range(n)]
            rng.append(max(d) - min(d))
            sd.append(st.stdev(d))
            psd.append(st.pstdev(d))
        out[n] = (st.mean(rng) / CI_Z, sum(1 for r in rng if r > CI_Z) / len(rng),
                  st.mean(sd), st.mean(psd))
    return out


def chi2_sf(x, df):
    """Upper tail of chi-square. df is small and even/odd both occur, so the series is exact
    for integer df via the regularised upper incomplete gamma at half-integer a."""
    a = df / 2.0
    if x <= 0:
        return 1.0
    # Lentz continued fraction for Q(a, x), adequate and stable for the df and x here.
    if x < a + 1:
        # series for P(a,x), then complement
        term = 1.0 / a
        tot = term
        k = 0
        while k < 1000:
            k += 1
            term *= x / (a + k)
            tot += term
            if abs(term) < abs(tot) * 1e-15:
                break
        return 1.0 - tot * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


rows = {}
for f in FILES:
    if not os.path.exists(f):
        sys.exit("REFUSE: %s is absent" % f)
    seen = 0
    for line in open(f):
        m = HW_RX.match(line)
        if m:
            seen += 1
            k, tag, hw, pt, n = (int(m.group(1)), m.group(2), float(m.group(3)),
                                 float(m.group(4)), int(m.group(5)))
            rows.setdefault(k, {})[tag] = (pt, hw, n)
    if not seen:
        sys.exit("REFUSE: %s has no HALFWIDTH line. Either it is not an an_paired_c211 report "
                 "or it predates the machine-readable tail, and a silent skip here would read "
                 "as a pass with fewer passes" % f)

tags = sorted(set(t for v in rows.values() for t in v))
print("passes read: %d (%s) from %d file(s)" % (len(tags), ", ".join(tags), len(FILES)))
NP = len(tags)
if NP < 3:
    sys.exit("REFUSE: %d pass(es) is not a spread -- at least 3 are needed before an "
             "across-pass statistic says anything about the floor" % NP)

NULL = null_check()
nr, np_, nsd, npsd = NULL[NP] if NP in NULL else NULL[6]
print()
print("NULL CALIBRATION at n=%d passes, %d draws, seed %d. No pass-level effect at all:"
      % (NP, NSIM // 4, SEED))
print("  range / half-width      expected %.4f, and P(range > half-width) = %.4f" % (nr, np_))
print("    <- the statistic the FIRST version of this file used, with ADEQUATE requiring")
print("       <= 1.0. A correct floor failed that %.1f%% of the time." % (100 * np_))
print("  sample SD / within SE   expected %.4f   <- the statistic used below" % nsd)
print("  pstdev understates the SD by %.1f%% at this n, which is why ddof=1 is used"
      % (100 * (1 - npsd / nsd)))

print()
print("PER-PASS displacement of deep/bsb_, percent. k=2 is floor by construction.")
print("  %-4s %s" % ("k", "  ".join("%7s" % t for t in tags)))
for k in KS:
    v = rows.get(k, {})
    print("  %-4d %s" % (k, "  ".join("%+7.2f" % v[t][0] if t in v else "      -"
                                      for t in tags)))

print()
print("ACROSS-PASS scatter against within-pass sampling. SE_within is the MEAN of the passes'")
print("  own half-widths divided by %.4f, read from their reports and not transcribed." % CI_Z)
res = {}
for k in KS:
    v = rows.get(k, {})
    if len(v) != NP:
        continue
    pts = [v[t][0] for t in tags]
    hws = [v[t][1] for t in tags]
    s_obs = st.stdev(pts)
    se_w = st.mean(hws) / CI_Z
    ratio = s_obs / se_w if se_w > 0 else float("inf")
    x2 = (NP - 1) * (s_obs ** 2) / (se_w ** 2) if se_w > 0 else float("inf")
    pv = chi2_sf(x2, NP - 1)
    s_pass = math.sqrt(max(0.0, s_obs ** 2 - se_w ** 2))
    res[k] = (s_obs, se_w, ratio, x2, pv, s_pass, min(hws), max(hws))
    print("  k=%-3d SD %5.3f  SE_within %5.3f  ratio %5.2f (null %.2f)  chi2 %8.1f  p %-9.2g"
          "  pass-level SD %5.3f" % (k, s_obs, se_w, ratio, nsd, x2, pv, s_pass))
    print("        the passes' own half-widths span %.3f to %.3f pp, so the mean above is not "
          "a uniform value" % (min(hws), max(hws)))

print()
if 2 not in res:
    sys.exit("REFUSE: no complete k=2 row, so the board's own provably-inert width is absent "
             "and nothing here is a floor statement")
s_obs, se_w, ratio, x2, pv, s_pass, _lo, _hi = res[2]
adequate = pv >= ALPHA
print("VERDICT %s at k=2, the width where the arm provably cannot fire."
      % ("ADEQUATE" if adequate else "UNDERSTATED"))
print("  across-pass SD %.3f pp against a within-pass SE of %.3f pp: ratio %.2f, whose value"
      % (s_obs, se_w, ratio))
print("  under NO pass-level effect is %.2f. chi2 = %.1f on %d df, p = %.2g."
      % (nsd, x2, NP - 1, pv))
if not adequate:
    bound = CI_Z * s_pass
    print("  The implied pass-level SD is %.3f pp, a 95%% pass-level band of +-%.2f pp against"
          % (s_pass, bound))
    print("  the +-%.2f pp a single pass reports." % (CI_Z * se_w))
    print("  CONSEQUENCE, as registered: a per-pass interval is readable only for effects")
    print("  larger than about %.2f pp." % bound)
    arm_ks = [k for k in KS if k != 2 and k in res]
    sm = min(abs(rows[k][t][0]) for k in arm_ks for t in tags)
    print("  It moves no arm verdict here: the smallest arm displacement at any width and pass")
    print("  is %.2f pp, which is %.1fx that bound." % (sm, sm / bound if bound > 0 else 0))
    print("  Arm mean across passes: %s"
          % ", ".join("k=%d %+.2f" % (k, st.mean([rows[k][t][0] for t in tags]))
                      for k in arm_ks))

print()
# `all([])` is True, so an empty `helped` printed the STRONGEST of the three conclusions on
# zero widths -- a positive claim indistinguishable from no evidence, and the `any()` branch
# that would have said "0 of 0" was unreachable because `all([])` wins first.
print("IS IT ONE PASS-LEVEL FACTOR? k=2 is floor on the same pass, so subtracting it removes a")
print("  common multiplicative term if there is one. Per width, because a term explaining some")
print("  widths and not others is not that term.")
k2 = {t: rows[2][t][0] for t in tags if t in rows.get(2, {})}
helped = []
for k in KS:
    if k == 2 or k not in res:
        continue
    raw = [rows[k][t][0] for t in tags]
    adj = [rows[k][t][0] - k2[t] for t in tags if t in k2]
    if len(adj) != len(raw):
        continue
    a_, b_ = st.stdev(raw), st.stdev(adj)
    helped.append(b_ < a_)
    print("  k=%-3d SD %5.3f raw -> %5.3f after subtracting that pass's k=2  (%s)"
          % (k, a_, b_, "narrower" if b_ < a_ else "no narrower"))
if not helped:
    print("  NO WIDTH QUALIFIED, so there is nothing to conclude. Reported rather than passed:")
    print("  `all([])` is True and would otherwise have printed the strongest of these three")
    print("  conclusions on an empty set.")
elif all(helped):
    print("  All %d widths narrow, so one pass-level term is consistent with every one of them."
          % len(helped))
elif any(helped):
    print("  MIXED: %d of %d widths narrow. A term explaining some and not others is not a"
          % (sum(helped), len(helped)))
    print("  single pass-level factor, so it is NOT adopted and the unadjusted scatter stands.")
else:
    print("  No width narrows, so the pass-level story is not supported at all.")

print()
print("WHAT THIS IS NOT: a positive control. It bounds how wide the floor is, not whether the")
print("  instrument can see an effect of a given size. This board still injects nothing.")
