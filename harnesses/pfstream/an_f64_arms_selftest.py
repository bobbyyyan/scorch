"""Self-test for the paired bootstrap added in response to finding 114.

THE DEFECT IT PINS. `an_f64_arms.py` compared a pooled median against a PER-CELL p05-p95 band.
A median's sampling spread narrows as 1/sqrt(n), so that boundary is ~sqrt(n) too wide for the
quantity it bounds. The consequence is one-directional: OUTSIDE verdicts stay safe, while
"inside" stops being evidence of absence.

So the control below is the case the defect hides: an effect that is REAL, CONSISTENT, and
SMALLER than a single cell's fluctuation. The band must call it inside; the paired CI must
exclude zero. A test that only confirmed the two agree where they already agree would not
cover this.
"""
import math
import random
import sys

sys.path.insert(0, ".")
from an_f64_arms import boot_median_ci, pct  # noqa: E402

FAIL = []


def check(name, cond, detail):
    print("  %-52s %s   %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        FAIL.append(name)


def main():
    n, EFFECT, CELL_SD = 392, -0.0050, 0.0200   # -0.50% real, 2.00% per-cell noise
    rng = random.Random(7)
    # floor = object swap only; arm = object swap PLUS a constant real effect. Paired per cell,
    # so the shared per-cell term must cancel -- that is what the estimator is for.
    shared = [rng.gauss(0.010, CELL_SD) for _ in range(n)]
    floor = [shared[i] + rng.gauss(0, 0.002) for i in range(n)]
    arm = [shared[i] + EFFECT + rng.gauss(0, 0.002) for i in range(n)]
    d = sorted(arm[i] - floor[i] for i in range(n))

    print("CONTROL: effect %+.2f%%, per-cell sd %.2f%%, n=%d" % (100 * EFFECT, 100 * CELL_SD, n))
    lo, hi = pct(sorted(floor), 0.05), pct(sorted(floor), 0.95)
    am = pct(sorted(arm), 0.5)
    inside = lo <= am <= hi
    check("the per-cell band calls this real effect 'inside'", inside,
          "arm median %+.3f%% in band [%+.3f%%, %+.3f%%]" % (100 * am, 100 * lo, 100 * hi))

    blo, bhi, dig, nn = boot_median_ci(d)
    check("the paired CI excludes zero", bhi < 0,
          "median %+.3f%% CI [%+.3f%%, %+.3f%%]" % (100 * pct(d, 0.5), 100 * blo, 100 * bhi))
    check("the paired CI covers the injected effect", blo <= EFFECT <= bhi,
          "injected %+.3f%%" % (100 * EFFECT))

    # THE sqrt(n) CLAIM, AND MY FIRST VERSION OF THIS CHECK WAS WRONG. It divided the floor's
    # per-cell band by the PAIRED CI and expected sqrt(n) ~ 19.8; it read 147.5. The pairing
    # removes the shared per-cell term, so that ratio carries sqrt(n) TIMES the variance pairing
    # removes -- two narrowings, not one. The sqrt(n) factor is only between a quantity's
    # per-cell spread and the sampling spread of ITS OWN median, so the check must use the
    # floor's own bootstrapped median. For a normal, band = 3.29s and CI = 4.91s/sqrt(n), so the
    # ratio is 0.67*sqrt(n) = 13.3 at n=392, not sqrt(n) itself.
    flo, fhi, _, _ = boot_median_ci(sorted(floor))
    same = (hi - lo) / (fhi - flo)
    pred = 0.67 * math.sqrt(n)
    check("band is ~0.67*sqrt(n) wider than that SAME median's CI",
          0.5 * pred <= same <= 2.0 * pred,
          "ratio %.1f vs predicted %.1f" % (same, pred))

    # and the pairing gain, reported separately because it is a DIFFERENT narrowing
    gain = (fhi - flo) / (bhi - blo)
    check("pairing narrows the CI further by removing the shared term", gain > 2.0,
          "paired CI is %.1fx narrower than the unpaired median CI" % gain)

    # NEGATIVE CONTROL, and it must be outside the population it certifies: no effect at all.
    arm0 = [shared[i] + rng.gauss(0, 0.002) for i in range(n)]
    d0 = sorted(arm0[i] - floor[i] for i in range(n))
    l0, h0, _, _ = boot_median_ci(d0)
    check("no-effect control: paired CI spans zero", l0 <= 0 <= h0,
          "CI [%+.3f%%, %+.3f%%]" % (100 * l0, 100 * h0))

    # the digest must PIN the resample: same seed same digest, different seed different digest
    _, _, dig2, _ = boot_median_ci(d, seed=20261002)
    _, _, dig3, _ = boot_median_ci(d, seed=99)
    check("index digest is stable under the same seed", dig == dig2, dig)
    check("index digest changes with the seed", dig2 != dig3, "%s vs %s" % (dig2, dig3))

    # and a guard that fires on its own input: too few cells must REFUSE, not return a bound
    r = boot_median_ci([0.1] * 5)
    check("n<8 returns no CI rather than a bound", r[0] is None, "returned %s" % (r[0],))

    print()
    if FAIL:
        print("SELFTEST FAILED: %s" % ", ".join(FAIL))
        return 1
    print("SELFTEST PASSED: 9 checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
