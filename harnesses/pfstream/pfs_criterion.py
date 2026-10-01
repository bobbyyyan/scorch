"""The beyond-floor criterion as an OBJECT that refuses an incoherent registration.

WHY THIS IS CODE AND NOT PROSE. Two registration failures in one hour, both in a
pre-registration document that read as careful:

  * amendment 18 registered a POSITIVE control and no negative one. The criterion then
    returned a confident verdict on every arm, and the floor arm -- two objects with
    byte-identical .text, no arm in it at any width -- failed the criterion too. A test
    its own no-effect case fails is not a test, and nothing in the document asked.
  * amendment 19 registered "beyond floor = CI entirely outside +-b" together with
    "positive control: inject at b". Those contradict: injecting AT the bound centres the
    estimate on the bound, so detection needs it to exceed the bound by a full half-width.
    Measured at 200000 draws, detection is 2.45% -- so the control fails 39 times in 40 on
    a CORRECT analysis, and the registered consequence would then void almost every band.

Neither was a careless number. The first is a missing presence check and the second is
arithmetic between two clauses of one paragraph. Both mechanise, so they are mechanised
here instead of being resolved to more carefully next time.

THE TWO WAYS A CONTROL'S MAGNITUDE CAN BE WRONG, and they are a pair:

  SELF-VOIDING   delta <= b. Detection 2.5% at delta == b. The control fails on a correct
                 instrument, so a passing analysis is discarded.
  SELF-PASSING   delta set to b + k*h using the band's OWN half-width. Detection 97.5% at
                 k=2 whenever the interval is well formed, so the control passes without
                 reporting anything about whether the instrument's noise is small enough.

The general form covers both: a control whose magnitude is a function of the instrument's
own noise reports nothing about that noise. So `delta` must be an absolute magnitude
anchored OUTSIDE the instrument -- here 3.0%, from a prior measurement of this same lever
at 2.9-14.2% against a measured floor bound of 1.489%.
"""
import math

BEYOND, WITHIN, INDETERMINATE = "BEYOND_FLOOR", "WITHIN_FLOOR", "INDETERMINATE"
UNDERPOWERED = "UNDERPOWERED"


class Incoherent(Exception):
    """A registration that cannot answer the question it poses."""


def detect_prob(delta, b, h, ci_z=1.959963984540054):
    """P(CI entirely above +b) when the true effect is `delta` and the half-width is `h`.

    Normal, h = ci_z * SE. Detection is `estimate - h > b`, so the margin in standard
    errors is (delta - b)/SE - ci_z = ci_z*((delta-b)/h - 1). At delta == b that is
    -ci_z and the probability is 1 - 0.975 = 2.5%, independent of b and of SE -- which is
    why the self-voiding case cannot be fixed by choosing a different bound.
    """
    if h <= 0:
        raise Incoherent("half-width %r is not positive, so no detection probability "
                         "exists" % (h,))
    z = ci_z * ((delta - b) / h - 1.0)
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


class BeyondFloorCriterion:
    """Construction refuses an incoherent registration; verdicts refuse an unvalidated one.

    `bands` is [(label, n)]. `floor_bound` is b, measured on THIS run's floor arm -- a
    pooled floor cannot be scaled into a per-band one (measured 13.2x apart on the f32
    screen), and a floor imported from another run is another run's artefact range.
    """

    MIN_DETECT = 0.80          # declared, not implied: below this a band is UNDERPOWERED

    def __init__(self, floor_bound, injection, bands, ci_z=1.959963984540054):
        if floor_bound <= 0:
            raise Incoherent("floor bound %r is not positive; it must be MEASURED on this "
                             "run's own floor arm" % (floor_bound,))
        if injection <= floor_bound:
            raise Incoherent(
                "SELF-VOIDING: the injected effect %.4f does not exceed the floor bound "
                "%.4f, but the criterion defines 'beyond floor' as a CI entirely outside "
                "the bound -- so the injected effect is floor-sized BY THE CRITERION'S OWN "
                "DEFINITION. Detection would be %.2f%% at best (2.5%% at equality), so the "
                "control would fail on a correct analysis. Anchor the injection outside the "
                "instrument instead." % (injection, floor_bound,
                                         100 * detect_prob(injection, floor_bound, 1e-9)))
        if not bands:
            raise Incoherent("no bands declared, so a per-band criterion has nothing to "
                             "range over and would report a pooled verdict under a "
                             "per-band name")
        self.b, self.delta, self.bands, self.z = floor_bound, injection, list(bands), ci_z
        self._neg = None       # the negative control's result; None means NOT RUN
        self._pow = {}

    # ---------------------------------------------------------------- the two controls
    def run_positive(self, halfwidths):
        """halfwidths: {band_label: h}. -> per-band power table. Declares nothing by fiat."""
        missing = [lab for lab, _n in self.bands if lab not in halfwidths]
        if missing:
            raise Incoherent("no half-width for band(s) %s, so their power is unstated "
                             "rather than measured" % missing)
        rows = []
        for lab, n in self.bands:
            h = halfwidths[lab]
            p = detect_prob(self.delta, self.b, h)
            self._pow[lab] = p
            rows.append((lab, n, h, self.b + 2 * h, (self.delta - (self.b + 2 * h)) / h, p))
        return rows

    def run_negative(self, floor_arm_verdicts):
        """floor_arm_verdicts: labels produced by applying this criterion to the FLOOR arm.

        Amendment 19's failure in one line: if the no-effect arm produces any BEYOND_FLOOR
        label, the criterion is invalid for this run and no arm verdict from it is
        readable. Recorded here rather than argued afterwards.
        """
        bad = [v for v in floor_arm_verdicts if v == BEYOND]
        self._neg = (len(bad) == 0, len(bad), len(list(floor_arm_verdicts)))
        return self._neg

    # ------------------------------------------------------------------- the verdict
    def label(self, band, point, lo, hi):
        """One of BEYOND/WITHIN/INDETERMINATE, plus UNDERPOWERED when power is unproven.

        Refuses before the negative control has been RUN -- not merely declared. A
        criterion whose no-effect case was never exercised produces a confident verdict on
        every arm, which is the exact shape of amendment 19's failure.
        """
        if self._neg is None:
            raise Incoherent("the negative control has not been RUN. Declaring one is not "
                             "running one; a criterion whose no-effect case was never "
                             "exercised answers confidently on every arm")
        ok, nbad, ntot = self._neg
        if not ok:
            raise Incoherent("the negative control FAILED: the floor arm produced %d "
                             "BEYOND_FLOOR label(s) of %d. The criterion is invalid for "
                             "this run and no arm verdict from it is readable"
                             % (nbad, ntot))
        if lo > hi:
            raise Incoherent("interval [%r, %r] is inverted" % (lo, hi))
        if lo > self.b or hi < -self.b:
            v = BEYOND
        elif -self.b <= lo and hi <= self.b:
            v = WITHIN
        else:
            v = INDETERMINATE
        p = self._pow.get(band)
        if p is None:
            raise Incoherent("band %r has no power measurement; run_positive() first, or "
                             "its verdict carries an unstated adequacy assumption" % (band,))
        if v != BEYOND and p < self.MIN_DETECT:
            # LABEL, never VOID. A label that voids its own analysis gets argued away
            # instead of reported, which is why this is a suffix and not an exception.
            return "%s/%s AT %.2f%% (power %.1f%%)" % (v, UNDERPOWERED, self.delta, 100 * p)
        return v
