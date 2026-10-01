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


def _mixed_population():
    """One cell of each kind, so an equivalence check has something to disagree about."""
    by = {"ship": grid(n=6), "shpb": grid(n=6), "strm": grid(n=6), "nopf": grid(n=6)}
    # m0: every arm slower -> denominator too fast, binding = min = 1.30
    for a, r in (("shpb", 1.30), ("strm", 1.44), ("nopf", 1.55)):
        by[a][("m0", 8)] = r
    # m1: every arm faster -> denominator too slow, binding = 1/max = 1/0.40 = 2.50
    for a, r in (("shpb", 0.25), ("strm", 0.40), ("nopf", 0.31)):
        by[a][("m1", 8)] = r
    # m2: arms straddle 1 -> shared at no threshold, binding = 1.0 even though an arm reads 1.5
    for a, r in (("shpb", 0.50), ("strm", 1.50), ("nopf", 1.05)):
        by[a][("m2", 8)] = r
    return by


def t_binding_is_the_threshold_where_shared_tail_flips():
    """Tie the new quantity to the existing detector MECHANICALLY, not by restating its rule.

    For thr >= 1, `cell in shared_tail(thr)` must equal `binding(cell) > thr`. Checked over a
    threshold grid that straddles all three cells' binding values, so each cell crosses.
    """
    by = _mixed_population()
    b = S.binding_thresholds(by, "ship")
    eq("binding-fast", round(b[("m0", 8)], 6), 1.30)
    eq("binding-slow", round(b[("m1", 8)], 6), 2.5)
    eq("binding-straddle", b[("m2", 8)], 1.0)
    disagree = []
    for thr in (1.01, 1.2, 1.29, 1.30, 1.31, 1.5, 2.0, 2.49, 2.5, 2.51, 3.0):
        inset = set(S.shared_tail(by, "ship", thr))
        predicted = {c for c, v in b.items() if v > thr}
        if inset != predicted:
            disagree.append((thr, sorted(inset), sorted(predicted)))
    eq("binding-flips-where-shared_tail-flips", disagree, [])


def t_the_max_arm_ratio_form_is_not_equivalent():
    """ABLATION: the form I published twice disagrees, so the check above is not vacuous.

    `max(rs) if min(rs) > 1 else 1/min(rs)` ignores whether the cell is SHARED. On the
    straddling cell it claims reach 2.0 -- a threshold at which the cell is in no tail at all.
    """
    by = _mixed_population()

    def wrong(rs):
        return max(rs) if min(rs) > 1.0 else 1.0 / min(rs)

    rs = (0.50, 1.50, 1.05)
    eq("ablation-wrong-value", wrong(rs), 2.0)
    eq("ablation-right-value", S.binding(rs), 1.0)
    thr = 1.5
    eq("ablation-cell-not-shared", ("m2", 8) in S.shared_tail(by, "ship", thr), False)
    eq("ablation-wrong-would-claim-shared", wrong(rs) > thr, True)
    eq("ablation-right-agrees", S.binding(rs) > thr, False)


def t_binding_refuses_an_empty_arm_list():
    """Own input: no arms is not a binding threshold of 1, it is unanswerable."""
    global CHECKS
    CHECKS += 1
    try:
        S.binding([])
    except ValueError:
        pass
    else:
        FAIL.append("binding-empty: did NOT raise")


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
    if CHECKS < 24:
        rc = 1
        print("  REFUSE: only %d assertions ran" % CHECKS)
    print("  " + ("OK" if rc == 0 else "FAILED"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
