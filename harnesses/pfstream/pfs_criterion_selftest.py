"""Plants for pfs_criterion. Each reproduces a registration failure that actually happened.

Coverage is established the way pfs_eta's is: the refusal population is extracted from the
target's own AST (every `raise Incoherent` site) and differenced against a line tracer, so
a refusal added later with no plant fails this file.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "pfs_criterion.py")
sys.path.insert(0, HERE)
import pfs_criterion as C                                            # noqa: E402

BANDS = [("0-1e4", 129), ("1e4-1e5", 181), ("1e5-1e6", 67), ("1e6-1e7", 36)]
HW = {"0-1e4": 0.2191, "1e4-1e5": 0.2989, "1e5-1e6": 0.2621, "1e6-1e7": 0.4129}
B, DELTA = 1.489, 3.0

FAIL, CHECKS, HIT = [], 0, set()


def _tracer(frame, event, arg):
    if frame.f_code.co_filename == TARGET:
        if event == "line":
            HIT.add(frame.f_lineno)
        return _tracer
    return None


def raise_sites():
    t = ast.parse(open(TARGET).read())
    return set(n.lineno for n in ast.walk(t) if isinstance(n, ast.Raise))


def expect_refusal(label, fn, want):
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except C.Incoherent as e:
        if want not in str(e):
            FAIL.append("%s: refused but without %r: %s" % (label, want, e))
    except Exception as e:                                           # noqa: BLE001
        FAIL.append("%s: raised %s, not Incoherent: %s" % (label, type(e).__name__, e))
    else:
        FAIL.append("%s: did NOT refuse" % label)


def ok_crit():
    c = C.BeyondFloorCriterion(B, DELTA, BANDS)
    c.run_positive(HW)
    return c


# -------------------------------------------------------- the two failures that happened
def t_self_voiding_injection():
    """AMENDMENT 19'S FAILURE. Injecting at the bound detects 2.5% of the time, so the
    control fails on a correct analysis. Construction refuses it and names the number."""
    expect_refusal("self-voiding/equal", lambda: C.BeyondFloorCriterion(B, B, BANDS),
                   "SELF-VOIDING")
    expect_refusal("self-voiding/below", lambda: C.BeyondFloorCriterion(B, B * 0.5, BANDS),
                   "floor-sized BY THE CRITERION'S OWN DEFINITION")


def t_negative_control_must_be_RUN_not_declared():
    """AMENDMENT 18'S FAILURE. A criterion whose no-effect case was never exercised
    answers confidently on every arm. Verdicts refuse until run_negative() has been
    called -- declaring one is not running one."""
    c = C.BeyondFloorCriterion(B, DELTA, BANDS)
    c.run_positive(HW)
    expect_refusal("neg-not-run", lambda: c.label("0-1e4", 5.0, 4.0, 6.0),
                   "has not been RUN")


def t_negative_control_failing_invalidates():
    """And when the floor arm DOES produce a beyond-floor label, the criterion says no
    arm verdict is readable -- the conclusion amendment 19 had to reach by argument."""
    c = ok_crit()
    ok, nbad, ntot = c.run_negative([C.WITHIN, C.BEYOND, C.INDETERMINATE])
    global CHECKS
    CHECKS += 1
    if ok or nbad != 1 or ntot != 3:
        FAIL.append("neg-fail: run_negative reported %r" % ((ok, nbad, ntot),))
    expect_refusal("neg-failed", lambda: c.label("0-1e4", 5.0, 4.0, 6.0),
                   "negative control FAILED")


# ------------------------------------------------------------------ the arithmetic
def t_detect_prob_matches_the_simulation():
    """The three numbers the simulation produced, from the closed form: 2.5 / 50 / 97.5.

    The value loop is LOAD-BEARING: measured over the mutation set in
    pfs_criterion_mutations.py it is the SOLE catcher of a narrowed ci_z, and one of two
    catchers of a dropped offset in detect_prob. The invariance check below reads as the
    more structural of the two and catches strictly less -- it catches nothing the value
    loop does not. Do not infer the attribution from how a check reads; that is the error
    this docstring is replacing, and the first version of this very sentence got it
    wrong too (it credited the value loop with being sole catcher of the offset).
    """
    global CHECKS
    for mult, want in ((0, 0.025), (1, 0.5), (2, 0.975)):
        CHECKS += 1
        h = 0.4
        got = C.detect_prob(B + mult * h, B, h)
        if abs(got - want) > 5e-4:
            FAIL.append("detect-prob-value: at b+%dh got %.6f, expected %.4f"
                        % (mult, got, want))
    CHECKS += 1
    # Assumption-free in b and in h at delta == b, which is why the self-voiding case
    # cannot be repaired by moving the bound. SCOPE: that is ALL this covers. It does
    # not cover a wrong constant -- dropping the `- 1.0` from detect_prob leaves the
    # probability flat in b and h too (0.5 rather than 0.025, spread 0.000e+00 under
    # both formulas), so this check passes and the value loop above is what fails.
    # Measured status over the 8 mutations in pfs_criterion_mutations.py: this assertion
    # catches exactly one (`bound-not-subtracted`) and the value loop catches that one as
    # well, so it is strictly dominated there and is the one of the pair that could go.
    # Kept because the mutation set is hand-written and 'dominated over 8 mutations I
    # thought of' is not 'dominated'.
    vals = [C.detect_prob(x, x, h) for x in (0.1, 1.489, 50.0) for h in (0.01, 0.4, 9.0)]
    if max(vals) - min(vals) > 1e-9:
        FAIL.append("detect-prob-invariance: at delta==b the probability moved with b "
                    "or h: %s" % vals)


def t_underpowered_labels_and_does_not_void():
    """The structural fix: a band with inadequate power gets a LABEL, and other bands
    keep their verdicts. A label that voids its own analysis gets argued away."""
    global CHECKS
    c = C.BeyondFloorCriterion(B, DELTA, BANDS)
    rows = c.run_positive(dict(HW, **{"1e6-1e7": 3.0}))      # one band made very wide
    c.run_negative([C.WITHIN] * 4)
    CHECKS += 1
    lab = c.label("1e6-1e7", 0.0, -0.5, 0.5)
    if "UNDERPOWERED" not in lab or "WITHIN_FLOOR" not in lab:
        FAIL.append("underpowered: expected a labelled WITHIN verdict, got %r" % lab)
    CHECKS += 1
    if c.label("0-1e4", 5.0, 4.0, 6.0) != C.BEYOND:
        FAIL.append("underpowered: a wide band voided another band's verdict")
    CHECKS += 1
    # and a BEYOND verdict is never suffixed -- power only qualifies a non-detection
    if "UNDERPOWERED" in c.label("1e6-1e7", 9.0, 8.0, 10.0):
        FAIL.append("underpowered: suffixed a BEYOND_FLOOR verdict, where low power is "
                    "not a caveat")
    CHECKS += 1
    margins = {r[0]: r[4] for r in rows}
    if abs(margins["0-1e4"] - (DELTA - (B + 2 * HW["0-1e4"])) / HW["0-1e4"]) > 1e-9:
        FAIL.append("positive: margin in half-widths is wrong for 0-1e4")


def t_the_bound_is_inclusive():
    """An interval exactly on +-b is WITHIN, not INDETERMINATE.

    Found by pfs_criterion_mutations.py, not by reading: tightening both comparisons in
    `label` to strict changed the verdict on a touching interval and no assertion noticed.
    It follows from the registration -- BEYOND is "entirely outside the bound", so an
    interval whose endpoint IS the bound is not entirely outside it, and the complement
    has to be inclusive or a touching interval belongs to neither side.
    """
    global CHECKS
    c = ok_crit()
    c.run_negative([C.WITHIN] * 4)
    CHECKS += 1
    v = c.label("0-1e4", 0.0, -B, B)
    if v != C.WITHIN:
        FAIL.append("inclusive-bound: [-b, +b] exactly gave %r, not WITHIN" % (v,))
    CHECKS += 1
    # The input that FAILS the check above, so its passing is not vacuous: one ulp past
    # the bound must leave WITHIN.
    v = c.label("0-1e4", 0.0, -B, B + 1e-9)
    if v != C.INDETERMINATE:
        FAIL.append("inclusive-bound: a hair past +b gave %r, not INDETERMINATE" % (v,))


def t_structural_refusals():
    expect_refusal("no-bands", lambda: C.BeyondFloorCriterion(B, DELTA, []), "no bands")
    expect_refusal("bad-bound", lambda: C.BeyondFloorCriterion(0.0, DELTA, BANDS),
                   "must be MEASURED")
    expect_refusal("missing-hw", lambda: C.BeyondFloorCriterion(B, DELTA, BANDS)
                   .run_positive({"0-1e4": 0.2}), "no half-width for band")
    c = ok_crit()
    c.run_negative([C.WITHIN])
    expect_refusal("inverted", lambda: c.label("0-1e4", 0.0, 1.0, -1.0), "inverted")
    expect_refusal("no-power", lambda: c.label("nosuch", 0.0, -1.0, 1.0),
                   "no power measurement")
    expect_refusal("bad-h", lambda: C.detect_prob(3.0, 1.0, 0.0), "not positive")


def t_f32_margins_are_the_published_table():
    """The table the peer computed and I reproduced, as a regression on the arithmetic."""
    global CHECKS
    c = C.BeyondFloorCriterion(B, DELTA, BANDS)
    got = {r[0]: (round(r[3], 3), round(r[4], 1)) for r in c.run_positive(HW)}
    want = {"0-1e4": (1.927, 4.9), "1e4-1e5": (2.087, 3.1),
            "1e5-1e6": (2.013, 3.8), "1e6-1e7": (2.315, 1.7)}
    CHECKS += 1
    if got != want:
        FAIL.append("f32-margins: got %s, expected %s" % (got, want))


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t_")]


def main():
    sys.settrace(_tracer)
    try:
        for fn in TESTS:
            try:
                fn()
            except Exception as e:                                   # noqa: BLE001
                FAIL.append("%s CRASHED: %s: %s" % (fn.__name__, type(e).__name__, e))
    finally:
        sys.settrace(None)
    want = raise_sites()
    missed = sorted(want - HIT)
    src = open(TARGET).read().split("\n")
    print("pfs_criterion selftest: %d plants, %d assertions, %d raise sites in the AST"
          % (len(TESTS), CHECKS, len(want)))
    print("  raise sites reached: %d of %d" % (len(want & HIT), len(want)))
    rc = 0
    if missed:
        rc = 1
        print("  UNCOVERED raise sites -- a plant is missing for each:")
        for ln in missed:
            print("    pfs_criterion.py:%d  %s" % (ln, src[ln - 1].strip()[:76]))
    for f in FAIL:
        rc = 1
        print("  FAIL " + f)
    if len(HIT) < 30:
        rc = 1
        print("  REFUSE: the tracer saw only %d lines, so the ratio above is vacuous"
              % len(HIT))
    print("  " + ("OK" if rc == 0 else "FAILED"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
