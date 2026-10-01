"""Plants for pfs_sharedtail, including the real cells and the tail the first version missed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfs_sharedtail as S                                          # noqa: E402

FAIL, CHECKS = [], 0


def eq(label, got, want):
    global CHECKS
    CHECKS += 1
    if got != want:
        FAIL.append("%s: got %r, expected %r" % (label, got, want))


def grid(n=50, k=8, base=1.0):
    return {(("m%d" % i), k): base for i in range(n)}


def t_denominator_too_fast_is_found():
    """higgs-twitter_retweet at k=10: ship 0.2132 ms, arms 4.49 / 4.30 / 4.53 -> ~21x."""
    by = {"ship": grid(), "shpb": grid(), "strm": grid(), "nopf": grid()}
    by["ship"][("m7", 8)] = 0.2132
    for a, v in (("shpb", 4.4922), ("strm", 4.3030), ("nopf", 4.5272)):
        by[a][("m7", 8)] = v
    got = S.shared_tail(by, "ship")
    eq("fast-found", sorted(got), [("m7", 8)])
    eq("fast-direction", got[("m7", 8)][0], "denominator-too-fast")


def t_denominator_too_slow_is_found_and_was_the_blind_tail():
    """DIMACS10/nc2010 at k=16: all three arms 0.10x because ship was anomalously SLOW. The
    first detector tested only min(r) > thr and could not see this -- and it was the LARGEST
    deviation on the screen, so the blind tail held the biggest cell."""
    by = {"ship": grid(), "shpb": grid(), "strm": grid(), "nopf": grid()}
    by["ship"][("m3", 8)] = 5.03892
    for a, r in (("shpb", 0.112), ("strm", 0.107), ("nopf", 0.103)):
        by[a][("m3", 8)] = 5.03892 * r
    got = S.shared_tail(by, "ship")
    eq("slow-found", sorted(got), [("m3", 8)])
    eq("slow-direction", got[("m3", 8)][0], "denominator-too-slow")
    global CHECKS
    CHECKS += 1
    one_sided = {c for c, v in got.items() if v[0] == "denominator-too-fast"}
    if one_sided:
        FAIL.append("slow-blindness: the one-sided subset is non-empty on a too-slow cell, so "
                    "the plant is not exercising the tail it claims")


def t_one_arm_disagreeing_is_not_shared():
    """A real arm effect moves ONE arm. The detector must not claim it."""
    by = {"ship": grid(), "shpb": grid(), "strm": grid(), "nopf": grid()}
    by["strm"][("m5", 8)] = 30.0            # one arm only, 30x
    eq("single-arm-not-shared", sorted(S.shared_tail(by, "ship")), [])
    # and the complementary input: all three move -> it IS shared
    for a in ("shpb", "strm", "nopf"):
        by[a][("m5", 8)] = 30.0
    eq("all-arms-shared", sorted(S.shared_tail(by, "ship")), [("m5", 8)])


def t_a_clean_grid_finds_nothing():
    """NON-VACUITY. Without this, 'it finds the anomalies' is satisfied by a detector that
    flags everything."""
    by = {"ship": grid(), "shpb": grid(), "strm": grid(), "nopf": grid()}
    eq("clean-empty", sorted(S.shared_tail(by, "ship")), [])
    global CHECKS
    CHECKS += 1
    if len(by["ship"]) < 10:
        FAIL.append("clean-empty: the grid has %d cells, so an empty result is near-vacuous"
                    % len(by["ship"]))


def t_sensitivity_pairs_the_two_counts():
    """A bare count is uninformative if the threshold was unreachable. The sweep must show the
    two-sided count exceeding the one-sided one where a too-slow cell exists."""
    by = {"ship": grid(), "shpb": grid(), "strm": grid(), "nopf": grid()}
    by["ship"][("m1", 8)] = 0.05                      # too fast -> both forms see it
    for a in ("shpb", "strm", "nopf"):
        by[a][("m1", 8)] = 1.0
        by[a][("m2", 8)] = 0.05                       # too slow -> only two-sided sees it
    rows = S.sensitivity(by, "ship", thrs=(1.20,))
    thr, two, one, widths = rows[0]
    eq("sweep-two", two, 2)
    eq("sweep-one", one, 1)
    eq("sweep-widths", widths, [8])


def t_refuses_a_degenerate_call():
    """No arms besides the denominator would return empty on any input, and thr <= 1 is not a
    ratio. Both must raise rather than produce a clean-looking zero."""
    global CHECKS
    for label, fn in (("no-arms", lambda: S.shared_tail({"ship": grid()}, "ship")),
                      ("thr-one", lambda: S.shared_tail({"ship": grid(), "a": grid()},
                                                        "ship", thr=1.0))):
        CHECKS += 1
        try:
            fn()
        except ValueError:
            pass
        else:
            FAIL.append("%s: did NOT raise" % label)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t_")]


def main():
    for fn in TESTS:
        try:
            fn()
        except Exception as e:                                       # noqa: BLE001
            FAIL.append("%s CRASHED: %s: %s" % (fn.__name__, type(e).__name__, e))
    print("pfs_sharedtail selftest: %d plants, %d assertions" % (len(TESTS), CHECKS))
    rc = 0
    for f in FAIL:
        rc = 1
        print("  FAIL " + f)
    if CHECKS < 14:
        rc = 1
        print("  REFUSE: only %d assertions ran" % CHECKS)
    print("  " + ("OK" if rc == 0 else "FAILED"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
