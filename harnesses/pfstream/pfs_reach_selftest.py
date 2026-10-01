"""Plants for pfs_reach, covering the degeneracies a real corpus cannot reach.

The identities in `reach_of` are asserted on every matrix it sees, but a corpus only ever supplies
matrices that exist: no all-empty case, no single-row case, nothing with zero nonzeros. So the
assertions have never run against the three boundary shapes they are most likely to be wrong on.
These plants supply them directly as row-length vectors, which is the checker's suggestion and
costs nothing.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfs_reach as P                                                 # noqa: E402

FAIL, CHECKS = [], 0


def eq(label, got, want):
    global CHECKS
    CHECKS += 1
    if got != want:
        FAIL.append("%s: got %r, expected %r" % (label, got, want))


def indptr(lens):
    return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def call(lens):
    """(G, S, added, reach, n_ge2, n_odd) for a row-length vector."""
    return P.reach_of(indptr(lens))


def t_the_loop_body_counts_match_a_literal_simulation():
    """The closed forms are algebra; _simulate runs the kernel's loop and guard. Over every
    length 0..40 they must agree, which is what makes the closed forms usable at all."""
    for L in range(0, 41):
        it, pf = P._simulate(L)
        eq("iters L=%d" % L, it, L // 2)
        eq("ship-pf L=%d" % L, pf, max(L - 1, 0) // 2)
        eq("added L=%d" % L, it - pf, 1 if (L >= 2 and L % 2 == 0) else 0)


def t_all_empty_rows():
    """Every row length 0: no prefetch site, so reach is undefined rather than zero."""
    G, S, added, reach, n_ge2, n_odd = call([0] * 7)
    eq("empty-G", G, 0)
    eq("empty-added", added, 0)
    eq("empty-nge2", n_ge2, 0)
    eq("empty-nodd", n_odd, 0)
    CHECKS_local = np.isnan(reach)
    eq("empty-reach-is-nan", bool(CHECKS_local), True)


def t_a_single_row():
    """rows == 1 is the shape the gate's `rows <= 1` clause is about, and a shape no corpus
    key supplies. A single row of 4 gives two iterations and one declined prefetch."""
    G, S, added, reach, n_ge2, n_odd = call([4])
    eq("one-row-G", G, 2)
    eq("one-row-S", S, 1)
    eq("one-row-added", added, 1)
    eq("one-row-reach", reach, 0.5)
    eq("one-row-nodd", n_odd, 0)


def t_every_long_row_odd_is_the_reach_zero_control():
    """The 154-matrix control group's shape: G > 0, added == 0, so reach is EXACTLY 0 and the
    2*P/d approximation is exact rather than undefined."""
    G, S, added, reach, n_ge2, n_odd = call([5] * 10)
    eq("odd-G", G, 20)
    eq("odd-added", added, 0)
    eq("odd-reach", reach, 0.0)
    eq("odd-nodd", n_odd, 10)
    eq("odd-exact-identity", 2 * G, sum([5] * 10) - n_odd)


def t_every_row_even_puts_the_error_factor_at_exactly_one():
    """n_odd == 0 is the atom at 1.0 in the error factor, and the only shape where 2*P/d is
    exact for a nonzero reach. 4.86% of SuiteSparse sits here; a p95 one rounding below it does
    not."""
    lens = [6] * 9
    G, S, added, reach, n_ge2, n_odd = call(lens)
    eq("even-nodd", n_odd, 0)
    eq("even-factor-is-one", 1.0 - n_odd / sum(lens), 1.0)
    eq("even-added", added, 9)
    eq("even-reach", reach, 9 / G)


def t_singletons_and_empties_are_excluded_from_added():
    """L == 1 contributes a row but no iteration; L == 0 contributes neither. Both must stay out
    of `added`, or a corpus of near-diagonal matrices inflates it."""
    G, S, added, reach, n_ge2, n_odd = call([0, 1, 0, 1, 2])
    eq("mix-G", G, 1)
    eq("mix-added", added, 1)
    eq("mix-nge2", n_ge2, 1)
    eq("mix-nodd", n_odd, 2)
    eq("mix-identity", 2 * G, 4 - n_odd)


def t_the_exact_form_equals_the_ratio_on_every_plant():
    """reach == 2*added/(nnz - n_odd) wherever G > 0, which is the identity the corpus driver
    asserts. Checked here on shapes the corpus cannot supply."""
    for lens in ([4], [5] * 10, [6] * 9, [0, 1, 0, 1, 2], [2, 3, 4, 5, 6], [1] * 5 + [2]):
        G, S, added, reach, n_ge2, n_odd = call(lens)
        nnz = sum(lens)
        if G == 0:
            continue
        eq("exact %r" % (lens,), round(reach, 12), round(2.0 * added / (nnz - n_odd), 12))


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("t_")]


def main():
    for fn in TESTS:
        try:
            fn()
        except Exception as e:                                        # noqa: BLE001
            FAIL.append("%s CRASHED: %s: %s" % (fn.__name__, type(e).__name__, e))
    print("pfs_reach selftest: %d plants, %d assertions" % (len(TESTS), CHECKS))
    rc = 0
    for f in FAIL:
        rc = 1
        print("  FAIL " + f)
    if CHECKS < 150:
        rc = 1
        print("  REFUSE: only %d assertions ran" % CHECKS)
    print("  " + ("OK" if rc == 0 else "FAILED"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
