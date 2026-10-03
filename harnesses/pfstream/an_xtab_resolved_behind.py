"""Cross-tabulate production's resolved worker count against being behind MKL, at one width.

WHY A CROSS-TABULATION AND NOT A LOOKUP. PREREG_thread_sizing amendment 21(b) asked whether the
nine `<=2`-worker cells intersect the 328-cell deficit set, found one of nine present in the
screen's corpus, and could conclude nothing: nine keys drawn from 72 against a 20-key shared set
have expected overlap 9 x 20/72 = 2.5, so observing 1 is unremarkable. Conditioning on the
predictor keeps one cell; conditioning on nothing keeps twenty. This script does the latter.

WHAT IT CANNOT DO, STATED FIRST BECAUSE THE MARGINAL IS EXTREME. 328 of 392 keys are behind at
k=2 = 83.7%, so on a 20-key shared set roughly 17 will be behind whatever the association is. The
registered reading (amendment 22(d)) is therefore: ONLY a table in which EVERY non-behind key sits
at `>=8` counts as support, because any other arrangement is reachable by chance at this marginal.
The script prints the marginal, the cell counts, and a two-sided Fisher exact p on the collapsed
2x2 so the reader can see how little the table can carry -- not to license a verdict at 0.05.

THE BEHIND SET IS AN INPUT, NOT RECOMPUTED HERE. It comes from `an_f64_vs_mkl.py --dump-behind`,
which is the instrument that defines it, so this script cannot disagree with the published board
about what "behind" means. A key absent from that file is NOT-behind only if it is also in the
screen's corpus, which is why the corpus list is a separate required argument: a key the screen
never measured is unmeasured, not ahead, and collapsing those two is how an absence becomes a zero.
"""
import argparse
import collections
import csv
import math


def fisher_2x2(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]], by enumeration. Exact, no SciPy."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def p_of(x):
        return (math.comb(r1, x) * math.comb(n - r1, c1 - x)) / math.comb(n, c1)

    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    obs = p_of(a)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= obs * (1 + 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, help="the thread-count sweep CSV (key,k,resolved,...)")
    ap.add_argument("--behind", required=True, help="keys behind in EVERY reading, from "
                    "an_f64_vs_mkl.py --dump-behind")
    ap.add_argument("--corpus", required=True, help="every key the screen measured; a key here but "
                    "not in --behind is NOT-behind, a key in neither is UNMEASURED")
    ap.add_argument("--k", type=int, required=True, help="the width. REQUIRED: the behind set and "
                    "the resolved count both differ by width, and a default would pick one")
    A = ap.parse_args()

    behind = set(l.strip() for l in open(A.behind) if l.strip())
    corpus = set(l.strip() for l in open(A.corpus) if l.strip())
    if not behind or not corpus:
        raise SystemExit("REFUSE: --behind or --corpus is empty, so every cell below would be 0")
    if not behind <= corpus:
        raise SystemExit("REFUSE: %d behind keys are outside the corpus list, so 'not behind' "
                         "would not be the complement of 'behind' within it"
                         % len(behind - corpus))

    # Production's own count. `res_ovr0` is a per-cell CONSTANT -- the count a caller that sets no
    # override gets -- so it is read from every row of the cell and asserted consistent rather than
    # taken from whichever row happens to be first. A first version filtered on `probe_ovr == 0`
    # instead; if that column never holds 0 the filter yields an EMPTY table, which at this
    # marginal would print as a clean "all shared keys are behind, NOT SCOREABLE" and read like a
    # measurement.
    # LIVENESS IS NOT OPTIONAL HERE, and getting it wrong was the defect that prompted this
    # paragraph. A cell reaching ONE distinct worker count has no argmin, so the headroom statistic
    # is UNDEFINED there -- not small. Grouping every shared key by its resolved count silently
    # widens the predicate to a superset of the population the effect was measured on: the nine
    # live `<=2` cells of the 72-key corpus, of which exactly ONE is in the shared set. A table
    # whose `<=2` cell held seven keys therefore had apparent power 7 and real power 1, and a
    # verdict read off the four non-behind keys in it was read off cells where no headroom can
    # exist. Liveness is counted the same way `nt_k2curve.py` counts it -- distinct `resolved`
    # values for the cell across every override row -- and reported per group rather than applied
    # silently, because n=1 labelled n=1 is worth more than n=7 carrying six undefined cells.
    seen = collections.defaultdict(set)
    reach = collections.defaultdict(set)
    rows = 0
    for r in csv.DictReader(open(A.oracle)):
        rows += 1
        try:
            if int(r["k"]) != A.k:
                continue
            seen[r["key"]].add(int(r["res_ovr0"]))
            reach[r["key"]].add(int(r["resolved"]))
        except (KeyError, TypeError, ValueError):
            continue
    live = {k for k, v in reach.items() if len(v) >= 2}
    bad = {k: sorted(v) for k, v in seen.items() if len(v) != 1}
    if bad:
        raise SystemExit("REFUSE: %d key(s) carry more than one res_ovr0 at k=%d, so production's "
                         "count is not a per-cell constant here and the grouping below would "
                         "depend on row order. First: %r" % (len(bad), A.k, sorted(bad.items())[0]))
    res = {k: next(iter(v)) for k, v in seen.items()}

    oracle_keys = set(res)
    shared = sorted(oracle_keys & corpus)
    print("ORACLE      %s: %d rows, %d keys with a production count at k=%d"
          % (A.oracle, rows, len(oracle_keys), A.k))
    print("SCREEN      corpus %d keys, behind-in-all at k=%d %d (%.1f%% -- THE MARGINAL)"
          % (len(corpus), A.k, len(behind), 100.0 * len(behind) / len(corpus)))
    print("SHARED      %d keys carry BOTH a production count and a screen verdict" % len(shared))
    if not shared:
        raise SystemExit("REFUSE: the two corpora share no key, so the table would be empty by "
                         "construction and not by measurement. Compare a sample key from each "
                         "side: an unnormalised prefix is the usual cause.")

    GRP = (("<=2", lambda n: n <= 2), ("3-7", lambda n: 3 <= n <= 7), (">=8", lambda n: n >= 8))
    print("LIVENESS    %d of %d oracle keys at k=%d reach >=2 counts; among the %d shared, %d live"
          % (len(live), len(oracle_keys), A.k, len(shared), sum(1 for x in shared if x in live)))

    def build(pop, label):
        tab = collections.Counter()
        for key in pop:
            g = next(name for name, pred in GRP if pred(res[key]))
            tab[(g, "behind" if key in behind else "not")] += 1
        print()
        print("  %s" % label)
        print("  %-8s %8s %8s %8s   %s" % ("group", "behind", "not", "total", "behind share"))
        for name, _p in GRP:
            b, n_ = tab[(name, "behind")], tab[(name, "not")]
            t = b + n_
            print("  %-8s %8d %8d %8d   %s"
                  % (name, b, n_, t, ("%.1f%%" % (100.0 * b / t)) if t else "n=0, no share"))
        tb = sum(tab[(g, "behind")] for g, _ in GRP)
        tn = sum(tab[(g, "not")] for g, _ in GRP)
        print("  %-8s %8d %8d %8d" % ("TOTAL", tb, tn, tb + tn))
        return tab, tb, tn

    _, _, _ = build(shared, "ALL SHARED KEYS -- NOT the population the effect was measured on; "
                            "printed only so the widening is visible")
    livshared = [x for x in shared if x in live]
    if not livshared:
        raise SystemExit("REFUSE: no shared key is live at k=%d, so the effect's population is "
                         "empty here and every share below would be undefined rather than small"
                         % A.k)
    tab, tb, tn = build(livshared, "LIVE SHARED KEYS ONLY -- THIS is the effect's population, "
                                   "because a cell reaching one count has no argmin")

    # The registered reading, applied rather than described.
    nonb = [(g, tab[(g, "not")]) for g, _ in GRP if tab[(g, "not")]]
    print()
    print("REGISTERED READING (amendment 22(d)): support ONLY if every non-behind key sits at >=8.")
    if tn == 0:
        print("  NOT SCOREABLE: all %d shared keys are behind, so the table has one column and no"
              % tb)
        print("  arrangement of it can discriminate. This is the marginal doing the work.")
    else:
        off = [(g, c) for g, c in nonb if g != ">=8"]
        print("  non-behind keys by group: %s" % ", ".join("%s=%d" % x for x in nonb))
        print("  -> %s" % ("SUPPORT: every non-behind key is at >=8" if not off else
                           "NOT SUPPORT: %d non-behind key(s) outside >=8 (%s)"
                           % (sum(c for _g, c in off), ", ".join("%s=%d" % x for x in off))))

    # Collapsed to 2x2 so an exact p exists, with its own caveat beside it.
    a = tab[("<=2", "behind")] + tab[("3-7", "behind")]
    b = tab[("<=2", "not")] + tab[("3-7", "not")]
    c = tab[(">=8", "behind")]
    d = tab[(">=8", "not")]
    if (a + b) and (c + d) and (a + c) and (b + d):
        p = fisher_2x2(a, b, c, d)
        print()
        print("  Collapsed 2x2 (<=7 workers vs >=8) [[%d, %d], [%d, %d]], Fisher exact two-sided "
              "p = %.4f" % (a, b, c, d, p))
        print("  Printed because a table with no test invites one to be read off it by eye. It is")
        print("  NOT a verdict at 0.05: the registered reading above is the verdict, and this p is")
        print("  the margin by which the marginal alone could have produced the table.")
    else:
        print()
        print("  No exact p: a row or column margin is zero, so the test is undefined. The counts")
        print("  above are the whole of the result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
