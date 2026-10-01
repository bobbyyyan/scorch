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
    """(G, S, added, reach, n_rows_reaching_loop) from CSR row pointers."""
    L = np.diff(indptr.astype(np.int64))
    G = int((L // 2).sum())
    S = int((np.maximum(L - 1, 0) // 2).sum())
    added = int((((L % 2) == 0) & (L >= 2)).sum())
    if G - S != added:
        raise AssertionError("G-S=%d but added=%d" % (G - S, added))
    return G, S, added, (added / G if G else float("nan")), int((L >= 2).sum())


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
        G, S, added, reach, nr = reach_of(indptr)
        nrows = len(indptr) - 1
        rows.append((key, nrows, nnz, G, S, added, reach,
                     nnz / nrows if nrows else float("nan")))

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
    for key, nrows, nnz, G, S, added, reach, deg in nosites[:6]:
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
    print("reach is ~ P(row length even) / mean_degree, so it is largest on the sparsest rows.")
    print("Spearman-free check -- reach by mean-degree band:")
    for lo, hi in ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 1 << 30)):
        m = (d >= lo) & (d < hi)
        if m.sum():
            print("  mean degree [%5s,%5s): n %3d   median reach %7.4f%%   max %7.4f%%"
                  % (lo, hi if hi < 1 << 29 else "inf", int(m.sum()),
                     100.0 * np.median(r[m]), 100.0 * r[m].max()))
    print()
    print("the ten matrices PFSTREAM reaches furthest:")
    for key, nrows, nnz, G, S, added, reach, deg in sorted(rows, key=lambda x: -x[6])[:10]:
        print("  %-42s rows %8d nnz %10d deg %7.2f  added %8d of %9d = %6.3f%%"
              % (key[-42:], nrows, nnz, deg, added, G, 100.0 * reach))
    print()
    emit_csv(allrows, os.environ.get("PFS_REACH_CSV", "pfs_reach.csv"))
    print()
    print("the ten it reaches least:")
    for key, nrows, nnz, G, S, added, reach, deg in sorted(rows, key=lambda x: x[6])[:10]:
        print("  %-42s rows %8d nnz %10d deg %7.2f  added %8d of %9d = %6.4f%%"
              % (key[-42:], nrows, nnz, deg, added, G, 100.0 * reach))


def emit_csv(rows, path):
    """Per-matrix reach, so a timing run can be joined against the structure that predicts it."""
    with open(path, "w") as fh:
        fh.write("key,rows,nnz,mean_degree,sites,ship_prefetches,added,reach\n")
        for key, nrows, nnz, G, S, added, reach, deg in rows:
            fh.write("%s,%d,%d,%.6f,%d,%d,%d,%s\n"
                     % (key, nrows, nnz, deg, G, S, added,
                        ("%.8f" % reach) if G else "undefined"))
    print()
    print("wrote %s (%d rows, reach 'undefined' where there are no prefetch sites)"
          % (path, len(rows)))


if __name__ == "__main__":
    main()
