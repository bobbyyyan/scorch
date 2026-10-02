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
from an_f64_arms import arm_paired_d, boot_median_ci, paired_d, pct  # noqa: E402

FAIL = []
RAN = []


def check(name, cond, detail):
    RAN.append(name)
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

    blo, bhi, vdig, idig, nn = boot_median_ci(d)
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
    flo, fhi, _, _, _ = boot_median_ci(sorted(floor))
    same = (hi - lo) / (fhi - flo)
    pred = 0.67 * math.sqrt(n)
    check("band is ~0.67*sqrt(n) wider than that SAME median's CI",
          0.5 * pred <= same <= 2.0 * pred,
          "ratio %.1f vs predicted %.1f" % (same, pred))

    # and the pairing gain, reported separately because it is a DIFFERENT narrowing
    gain = (fhi - flo) / (bhi - blo)
    # THE THRESHOLD IS DELIBERATELY LOOSE AND THE HEADLINE NUMBER MUST NOT BE QUOTED. In this
    # control `floor` and `arm` share `shared[i]` EXACTLY, so the shared term cancels exactly and
    # the realized gain is a property of the construction. The real object-swap artefact is not
    # identically shared between the arm pair and the floor pair, so the gain on DLMC will be
    # smaller by however much it fails to be shared -- which is not a defect.
    check("pairing narrows the CI further by removing the shared term", gain > 2.0,
          "%.1fx here, CONSTRUCTION-DEPENDENT, not a property of the method" % gain)

    # NEGATIVE CONTROL, and it must be outside the population it certifies: no effect at all.
    arm0 = [shared[i] + rng.gauss(0, 0.002) for i in range(n)]
    d0 = sorted(arm0[i] - floor[i] for i in range(n))
    l0, h0, _, _, _ = boot_median_ci(d0)
    check("no-effect control: paired CI spans zero", l0 <= 0 <= h0,
          "CI [%+.3f%%, %+.3f%%]" % (100 * l0, 100 * h0))

    # the digest must PIN the resample: same seed same digest, different seed different digest
    _, _, _, idig_same, _ = boot_median_ci(d, seed=20261002)
    check("index digest is stable under the same seed", idig == idig_same, idig)
    _, _, _, idig2, _ = boot_median_ci(d, seed=20261002)
    _, _, _, idig3, _ = boot_median_ci(d, seed=99)
    check("index digest changes with the seed", idig2 != idig3,
          "%s vs %s" % (idig2, idig3))

    # THE ASSERTION THAT WAS MISSING, AND IT FAILED ON THE PREVIOUS VERSION. The old digest was a
    # pure function of (seed, nboot, n), so a substituted dataset reproduced it exactly: this
    # check is the one that catches it, and determinism alone does not imply it.
    big = sorted(x + 0.095 for x in d)
    _, _, vdig_b, idig_b, _ = boot_median_ci(big)
    check("SAME n, DIFFERENT data -> different DATA digest", vdig != vdig_b,
          "%s vs %s" % (vdig, vdig_b))
    check("and the index digest is correctly UNCHANGED by the data", idig == idig_b,
          "%s -- resample depends on (seed, nboot, n) alone, as intended" % idig)

    # and a guard that fires on its own input: too few cells must REFUSE, not return a bound
    r = boot_median_ci([0.1] * 5)
    check("n<8 returns no CI rather than a bound", r[0] is None, "returned %s" % (r[0],))

    # PRODUCTION'S PAIRING LINE, which the checks above never reached: they used a plain
    # difference, not even production's functional form. The material mutation is the ROLE
    # assignment -- swap arm and floor and every verdict's sign flips with nothing firing -- so
    # the synthetic `med` below is ASYMMETRIC and a swap must change the sign.
    pop = [("m%d" % i, 2) for i in range(40)]
    synth = {}
    for i, (key, w) in enumerate(pop):
        synth[(key, w, "ship")] = 100.0
        synth[(key, w, "shpb")] = 101.0 + 0.01 * i      # floor: ~+1% slower than ship
        synth[(key, w, "nopf")] = 97.0 + 0.01 * i       # arm:   ~-3% faster than ship
    # THE CHECK THAT COVERS THE CALL SITE: it consumes arm_paired_d, where the three tags live
    # exactly once, and asserts a SIGNED value. Swap the literals in that wrapper and this fails.
    dd = arm_paired_d(synth, pop)
    check("arm_paired_d reads the SIGNED known value (negative = arm faster)",
          bool(dd) and abs(pct(dd, 0.5) - (-0.04)) < 1e-9,
          "median %+.4f, expected -0.0400 (= arm -3%% minus floor +1%%)" % pct(dd, 0.5))
    # AND THIS ONE DOES NOT COVER IT, which is now stated rather than implied. paired_d returns
    # (a-f)/b, so swapping its two tag arguments is an exact negation for every dataset: the
    # property is real and cheap, but it shows a swap would be VISIBLE, not that none occurs.
    #
    # BOTH OPERANDS ARE EXPLICIT, AND THAT IS THE WHOLE POINT. An earlier version compared the
    # explicit swap against `dd` -- the WRAPPER's output -- so the assertion was
    # `paired_d(swapped) + arm_paired_d(...) == 0`, which depends on the wrapper and duly FAILED
    # under a planted swap. An identity cannot fail, so the label contradicted the behaviour in
    # both directions at once: "holds unconditionally" printed beside a FAIL, and "covers
    # nothing" when it had demonstrably covered something. A reader would have concluded that
    # antisymmetry broke, which cannot happen. A label about an assertion's NATURE is itself a
    # claim and has to match the assertion's FORM.
    fwd = paired_d(synth, pop, "nopf", "shpb", "ship")
    sw = paired_d(synth, pop, "shpb", "nopf", "ship")
    check("paired_d is antisymmetric in its two tags (an IDENTITY, not coverage)",
          abs(pct(sw, 0.5) + pct(fwd, 0.5)) < 1e-12,
          "%+.4f vs %+.4f -- holds unconditionally, covers nothing"
          % (pct(fwd, 0.5), pct(sw, 0.5)))
    miss = dict(synth)
    del miss[(pop[0][0], 2, "shpb")]
    check("a cell missing one of the three tags is dropped, not defaulted",
          len(arm_paired_d(miss, pop)) == len(dd) - 1,
          "%d cells vs %d" % (len(arm_paired_d(miss, pop)), len(dd)))

    print()
    if FAIL:
        print("SELFTEST FAILED %d of %d: %s" % (len(FAIL), len(RAN), ", ".join(FAIL)))
        return 1
    print("SELFTEST PASSED: %d checks" % len(RAN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
