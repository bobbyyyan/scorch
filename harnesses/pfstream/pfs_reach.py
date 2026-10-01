"""How much of the register-block prefetch stream does PFSTREAM actually change?

THE QUANTITY, DERIVED FROM THE KERNEL. The register-block body is

    for (; pA + 1 < pA_end; pA += 2) { ... }                       spmm.h:2280

and inside it the two arms are, verbatim:

    PFSTREAM:  pf_ = pA + PFD < pA_limit ? pA + PFD : pA_limit - 1;   // unconditional prefetch
    ship:      if (pA + PFD < pA_end) __builtin_prefetch(...);        // conditional

PFD ships at 2. So over a row of length L the loop runs floor(L/2) times; PFSTREAM issues a
prefetch on every one of those iterations, and ship issues one only when pA + 2 < pA_end. The
difference is therefore ONE prefetch per even-length row and zero per odd-length row -- PFSTREAM
is not a different run-ahead, it is one extra cross-row prefetch at the tail of every even row.

    G     = sum floor(L_i / 2)        guard sites = PFSTREAM's prefetch count
    S     = sum floor((L_i - 1) / 2)  ship's prefetch count
    added = G - S                     = #{i : L_i even, L_i >= 2}
    reach = added / G

`reach` is a ceiling on how much PFSTREAM can matter at all, computed from sparsity structure
with no timing. It is a property of the matrix, and because the narrow-k routing decisions are
per-CALL (SCORCH_NARROWK_EXACT_MINDEG ships at 0, so there is no per-row degree floor), at any
width where the register block runs it runs on every row and this reach applies to all of them.

OWN-INPUT CHECK. The closed forms above are my algebra, not the kernel's. `_simulate` runs the
actual loop and guard for every L in a range and counts; the driver REFUSES if the closed form
disagrees anywhere. A closed form that silently differs from the loop would make every number
below plausible and wrong.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csrresolve                                                     # noqa: E402

PFD = 2


def _simulate(L):
    """(iterations, ship_prefetches) by running the kernel's loop literally."""
    pA_begin, pA_end = 0, L
    it = pf = 0
    pA = pA_begin
    while pA + 1 < pA_end:
        it += 1
        if pA + PFD < pA_end:
            pf += 1
        pA += 2
    return it, pf


def _check_closed_forms(hi=200):
    bad = []
    for L in range(0, hi):
        it, pf = _simulate(L)
        if it != L // 2 or pf != max(L - 1, 0) // 2:
            bad.append(("iters/pf", L, it, L // 2, pf, max(L - 1, 0) // 2))
        added_closed = 1 if (L >= 2 and L % 2 == 0) else 0
        if it - pf != added_closed:
            bad.append(("added", L, it - pf, added_closed))
    return bad


def _q(v, q):
    """A quantile, printed so a value NEAR a boundary cannot read as the boundary.

    A factor of `1 - n_odd/nnz` is exactly 1 only when a matrix has no odd-length row at all, and
    that is a real atom worth distinguishing from a value a rounding away from it. Printed at four
    decimals, SuiteSparse's p95 read 1.0000 and looked like the atom; it is 0.9999968283, and the
    atom (4.86% of that population) does not begin until quantile 0.9521. So a value within 5e-5
    of 0 or 1 is printed with enough digits to show it is not there, and an exact hit is marked.
    """
    import numpy as _np
    x = float(_np.quantile(v, q))
    if x == 1.0 or x == 0.0:
        return "%.4f(exact)" % x
    if abs(x - 1.0) < 5e-5 or abs(x) < 5e-5:
        return "%.10f" % x
    return "%.4f" % x


def reach_of(indptr):
    """(G, S, added, reach, n_rows_reaching_loop, n_odd) from CSR row pointers.

    `reach` has an EXACT closed form, which is worth having because the approximation below is
    otherwise tempting to check by comparing aggregates of different kinds. Since
    floor(L/2) = (L - (L mod 2))/2,

        G = (nnz - n_odd)/2   and   reach = added/G = 2*added/(nnz - n_odd)

    exactly, per matrix. The form `2*P/d = 2*added/nnz` is that with n_odd dropped, so its error
    is exactly the factor `1 - n_odd/nnz` -- the share of nonzeros sitting in odd-length rows'
    remainders. Both identities are asserted here, which makes them implementation controls on
    this function rather than facts about any corpus.
    """
    L = np.diff(indptr.astype(np.int64))
    nnz = int(L.sum())
    G = int((L // 2).sum())
    S = int((np.maximum(L - 1, 0) // 2).sum())
    added = int((((L % 2) == 0) & (L >= 2)).sum())
    n_odd = int(((L % 2) == 1).sum())
    if G - S != added:
        raise AssertionError("G-S=%d but added=%d" % (G - S, added))
    if 2 * G != nnz - n_odd:
        raise AssertionError("2G=%d but nnz-n_odd=%d" % (2 * G, nnz - n_odd))
    return G, S, added, (added / G if G else float("nan")), int((L >= 2).sum()), n_odd


def main():
    bad = _check_closed_forms()
    if bad:
        sys.exit("REFUSE: closed form disagrees with the simulated loop: %r" % bad[:5])
    print("own-input check: closed forms match the simulated loop for every L in 0..199")
    print("  L=2 -> %r, L=3 -> %r, L=4 -> %r, L=5 -> %r (iterations, ship prefetches)"
          % (_simulate(2), _simulate(3), _simulate(4), _simulate(5)))
    print()

    cache, keyfile = sys.argv[1], sys.argv[2]
    keys = csrresolve.read_keys(keyfile)
    paths, how, unresolved = csrresolve.resolve_all(cache, keys)
    for ln in csrresolve.describe(how, unresolved, "corpus keys"):
        print(ln)
    if not paths:
        sys.exit("REFUSE: zero keys resolved, so every number below would be vacuous")

    rows = []
    for key in keys:
        p = paths.get(key)
        if p is None:
            continue
        with np.load(p) as z:
            if "indptr" in z:
                indptr = z["indptr"]
                nnz = int(indptr[-1])
            else:
                A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]))
                indptr, nnz = A.indptr, A.nnz
        G, S, added, reach, nr, n_odd = reach_of(indptr)
        nrows = len(indptr) - 1
        rows.append((key, nrows, nnz, G, S, added, reach,
                     nnz / nrows if nrows else float("nan"), n_odd))

    print()
    print("matrices measured: %d" % len(rows))

    # A matrix whose every row has length 0 or 1 never enters the register block's loop body, so
    # it has no prefetch sites and no reach -- not a reach of zero, an undefined one. Reporting
    # these separately is the point: they are the matrices on which the whole prefetch question,
    # ship and PFSTREAM alike, is inert, and averaging a NaN over them hid them completely.
    nosites = [x for x in rows if x[3] == 0]
    have = [x for x in rows if x[3] > 0]
    print("matrices with ZERO prefetch sites (every row shorter than 2 nonzeros): %d of %d"
          % (len(nosites), len(rows)))
    for key, nrows, nnz, G, S, added, reach, deg, _o in nosites[:6]:
        print("    %-46s rows %8d nnz %9d deg %6.3f" % (key[-46:], nrows, nnz, deg))
    if not have:
        sys.exit("REFUSE: no matrix has a prefetch site, so reach is undefined everywhere")
    allrows = list(rows)
    rows = have
    r = np.array([x[6] for x in rows])
    d = np.array([x[7] for x in rows])
    addpn = np.array([x[5] / x[2] for x in rows])
    print("reach = added prefetches / PFSTREAM prefetch sites, over %d matrices:" % len(r))
    for q, lab in ((0.0, "min"), (0.10, "p10"), (0.50, "median"), (0.90, "p90"),
                   (0.99, "p99"), (1.0, "max")):
        print("  %-7s %8.4f%%" % (lab, 100.0 * np.quantile(r, q)))
    print("  mean    %8.4f%%" % (100.0 * r.mean()))
    print()
    print("added prefetches per NONZERO, which is the cost axis the kernel pays on:")
    for q, lab in ((0.50, "median"), (0.90, "p90"), (1.0, "max")):
        print("  %-7s %8.4f per 100 nonzeros" % (lab, 100.0 * np.quantile(addpn, q)))
    print()
    print("EXACT FORM: reach = 2*added/(nnz - n_odd), since G = (nnz - n_odd)/2. The familiar")
    print("2*P/d = 2*added/nnz is that with n_odd dropped, so it UNDERSTATES reach by exactly")
    print("the factor 1 - n_odd/nnz, per matrix -- no comparison of aggregates required. Chasing")
    print("that error by dividing a per-matrix median reach by a mean degree mixes estimator")
    print("kinds and gave three different answers (0.81, 0.81, 0.87); the per-matrix ratio is")
    print("the only form comparable to a per-matrix measured reach.")
    print()
    print("THREE DEGENERACY CLASSES, which the exact form separates and the ratio does not:")
    cls0 = [x for x in allrows if x[3] == 0]
    cls1 = [x for x in allrows if x[3] > 0 and x[5] == 0]
    cls2 = [x for x in allrows if x[3] > 0 and x[5] > 0]
    print("  G == 0            %5d   reach is 0/0, UNDEFINED; the factor is still defined"
          % len(cls0))
    print("  G > 0, added == 0 %5d   reach is EXACTLY 0 and so is the prediction, so the"
          % len(cls1))
    print("                            approximation is EXACT here; the RATIO is 0/0 = NaN")
    print("  G > 0, added > 0  %5d   ratio defined, and equal to the factor by the identity"
          % len(cls2))
    if len(cls0) + len(cls1) + len(cls2) != len(allrows):
        raise AssertionError("the three classes do not partition %d matrices" % len(allrows))

    # The factor and the ratio are the SAME random variable wherever added > 0, so their medians
    # cannot differ on a common population. Asserting that is free and it fails the instant one
    # side is computed over a different set -- which is exactly what happened when the factor was
    # taken over G > 0 (4271 matrices) and the ratio over reach > 0 (4117), printing 0.9016 and
    # 0.9063 as one answer. No amount of care in the prose catches that; this does.
    ef = np.array([1.0 - x[8] / x[2] for x in cls2])
    ratio = np.array([(2.0 * (x[5] / x[1]) / (x[2] / x[1])) / x[6] for x in cls2])
    if abs(float(np.median(ef)) - float(np.median(ratio))) > 1e-12:
        raise AssertionError(
            "factor median %.12f != ratio median %.12f on the added>0 population, so one of "
            "them is over a different set" % (np.median(ef), np.median(ratio)))
    print()
    print("ON THE %d MATRICES WHERE THE RATIO IS DEFINED (G > 0 and added > 0):" % len(cls2))
    print("  error factor 1 - n_odd/nnz   p5 %s  median %s  p95 %s"
          % tuple(_q(ef, q) for q in (0.05, 0.50, 0.95)))
    print("  predicted/measured           p5 %s  median %s  p95 %s"
          % tuple(_q(ratio, q) for q in (0.05, 0.50, 0.95)))
    print("  ASSERTED equal medians (they are one random variable where added > 0)")
    print("  so 2*P/d understates the median reach by %.2f%% ON THIS POPULATION"
          % (100.0 * (1 - np.median(ef))))
    efg = np.array([1.0 - x[8] / x[2] for x in allrows if x[3] > 0 and x[2]])
    same = len(cls2) == efg.size
    print("  the factor median over ALL %d with G > 0 is %.4f" % (efg.size, np.median(efg)))
    if same:
        print("    (the same population here, so necessarily identical)")
    else:
        print("    which differs because the added == 0 matrices are, by construction, those")
        print("    whose long rows are all ODD, so they sit at the low end of the factor")
    atom = sum(1 for x in cls2 if x[8] == 0)
    n2 = max(len(cls2), 1)
    # The quantile at which the atom STARTS, under this file's index convention (n-1)q: the atom
    # occupies 0-indexed ranks n-atom .. n-1, so it starts at q = (n-atom)/(n-1). The tempting
    # 1 - atom/n is NOT that -- it gives 0.9514 here where the true threshold is 0.9517, and a
    # bisection on the real data confirms q=0.9514 still returns 0.9999995 rather than 1.
    qstart = (n2 - atom) / (n2 - 1) if n2 > 1 and atom else float("nan")
    print("  matrices with n_odd == 0 exactly, where the factor is 1 and not merely near it:")
    print("    %d of %d = %.2f%%; the atom at 1.0 begins at quantile %s"
          % (atom, len(cls2), 100.0 * atom / n2,
             ("%.6f" % qstart) if atom else "none (no such matrix)"))
    print("Spearman-free check -- reach by mean-degree band:")
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 1 << 30)):
        m = (d >= lo) & (d < hi)
        if m.sum():
            print("  mean degree [%5s,%5s): n %3d   median reach %7.4f%%   max %7.4f%%"
                  % (lo, hi if hi < 1 << 29 else "inf", int(m.sum()),
                     100.0 * np.median(r[m]), 100.0 * r[m].max()))
    print()
    print("the ten matrices PFSTREAM reaches furthest:")
    for key, nrows, nnz, G, S, added, reach, deg, _o in sorted(rows, key=lambda x: -x[6])[:10]:
        print("  %-42s rows %8d nnz %10d deg %7.2f  added %8d of %9d = %6.3f%%"
              % (key[-42:], nrows, nnz, deg, added, G, 100.0 * reach))
    print()
    emit_csv(allrows, os.environ.get("PFS_REACH_CSV", "pfs_reach.csv"))
    print()
    print("the ten it reaches least:")
    for key, nrows, nnz, G, S, added, reach, deg, _o in sorted(rows, key=lambda x: x[6])[:10]:
        print("  %-42s rows %8d nnz %10d deg %7.2f  added %8d of %9d = %6.4f%%"
              % (key[-42:], nrows, nnz, deg, added, G, 100.0 * reach))


def emit_csv(rows, path):
    """Per-matrix reach, so a timing run can be joined against the structure that predicts it."""
    with open(path, "w") as fh:
        fh.write("key,rows,nnz,mean_degree,sites,ship_prefetches,added,reach,n_odd\n")
        for key, nrows, nnz, G, S, added, reach, deg, n_odd in rows:
            fh.write("%s,%d,%d,%.6f,%d,%d,%d,%s,%d\n"
                     % (key, nrows, nnz, deg, G, S, added,
                        ("%.8f" % reach) if G else "undefined", n_odd))
    print()
    print("wrote %s (%d rows, reach 'undefined' where there are no prefetch sites)"
          % (path, len(rows)))


if __name__ == "__main__":
    main()
