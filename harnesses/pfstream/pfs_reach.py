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
    print("that error by dividing a per-matrix median reach by a mean degree mixes three")
    print("estimator kinds and gave three different answers (0.81, 0.87); the per-matrix ratio")
    print("below is the only form comparable to a per-matrix measured reach.")
    ef = np.array([1.0 - x[8] / x[2] for x in rows if x[2]])
    print("  error factor 1 - n_odd/nnz:  p5 %.4f  median %.4f  p95 %.4f  (n=%d)"
          % (np.quantile(ef, 0.05), np.median(ef), np.quantile(ef, 0.95), ef.size))
    print("  so 2*P/d understates the median reach here by %.2f%%" % (100.0 * (1 - np.median(ef))))
    print("  where reach == 0 both sides are 0, so the approximation is EXACT, not undefined")
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
