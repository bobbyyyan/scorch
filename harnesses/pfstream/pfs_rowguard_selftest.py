"""Plants for pfs_rowguard. The first one is the read that happened."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfs_rowguard as G                                            # noqa: E402

FAIL, CHECKS = [], 0


def cells(n_per_width, widths=(2, 4, 6, 8, 10, 12, 16), prefix="m"):
    return dict(((("%s%d" % (prefix, i)), k), 1.0)
                for k in widths for i in range(n_per_width))


def expect_refusal(label, fn, *must):
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except G.Incomplete as e:
        for m in must:
            CHECKS += 1
            if m not in str(e):
                FAIL.append("%s: refused but without %r: %s" % (label, m, e))
    except Exception as e:                                          # noqa: BLE001
        FAIL.append("%s: raised %s, not Incomplete: %s" % (label, type(e).__name__, e))
    else:
        FAIL.append("%s: did NOT refuse" % label)


def expect_ok(label, fn):
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except Exception as e:                                          # noqa: BLE001
        FAIL.append("%s: refused a complete pass: %s" % (label, e))


def t_the_read_that_happened():
    """f64 round 1: ship/shpb/strm at 418 per width, nopf mid-write at 100. The medians on
    the short arm were plausible and on a quarter of the corpus."""
    by = {a: cells(418) for a in ("ship", "shpb", "strm")}
    by["nopf"] = cells(100)
    expect_refusal("nopf-mid-write", lambda: G.check(by), "nopf=700", "WAIT for those file(s)")


def t_a_complete_pass_passes():
    """NON-VACUITY. Without this, 'it refuses the truncated pass' is satisfied by a guard that
    refuses everything."""
    expect_ok("complete", lambda: G.check({a: cells(418) for a in
                                           ("ship", "shpb", "strm", "nopf")}))


def t_equal_totals_different_cells():
    """Totals can agree while the censuses differ -- a pass that skipped one width and
    over-covered another. A total-only check cannot see it."""
    by = {"ship": cells(418)}
    odd = cells(418)
    for i in range(418):
        odd.pop(("m%d" % i, 2), None)
    for i in range(418):
        odd[("extra%d" % i, 4)] = 1.0
    by["strm"] = odd
    expect_refusal("same-total-different-census", lambda: G.check(by),
                   "different per-width censuses")


def t_one_arm_cannot_check_itself():
    """A single arm has no sibling to compare against, so the guard must refuse rather than
    pass -- otherwise it passes on an empty file."""
    expect_refusal("single-arm", lambda: G.check({"ship": cells(418)}),
                   "cannot be checked against its siblings")
    expect_refusal("single-arm-empty", lambda: G.check({"ship": {}}),
                   "cannot be checked against its siblings")
    expect_ok("single-arm-with-expected",
              lambda: G.check({"ship": cells(418)}, expected_total=418 * 7))


def t_all_arms_equally_truncated_is_the_blind_spot():
    """STATED, because the guard cannot see it: if every arm is truncated the same way, the
    counts agree and this passes. That is why expected_total exists, and why its absence is
    a limit rather than a bug."""
    expect_ok("all-truncated-passes", lambda: G.check({a: cells(100) for a in
                                                       ("ship", "shpb", "strm", "nopf")}))
    expect_refusal("expected-total-catches-it",
                   lambda: G.check({a: cells(100) for a in ("ship", "shpb")},
                                   expected_total=418 * 7),
                   "against an expected", "ship, shpb")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t_")]


def main():
    for fn in TESTS:
        try:
            fn()
        except Exception as e:                                       # noqa: BLE001
            FAIL.append("%s CRASHED: %s: %s" % (fn.__name__, type(e).__name__, e))
    print("pfs_rowguard selftest: %d plants, %d assertions" % (len(TESTS), CHECKS))
    rc = 0
    for f in FAIL:
        rc = 1
        print("  FAIL " + f)
    if CHECKS < 12:
        rc = 1
        print("  REFUSE: only %d assertions ran" % CHECKS)
    print("  " + ("OK" if rc == 0 else "FAILED"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
