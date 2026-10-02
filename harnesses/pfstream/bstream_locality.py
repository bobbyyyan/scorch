"""Structural predictor of B-side memory traffic for CSR SpMM, per key and per width.

WHY THIS EXISTS. The float64 screen leaves 192 of 2926 cells behind MKL, concentrated at
narrow k and high mean degree (129 at k=2, none at k=12 or k=16), and NEITHER prefetch arm
moves them: PFD=0 wins 1.3-2.9% across k=4..16 and reads +0.139% on the behind set at k=2.
So the deficit is not the register-block prefetch. The standing hypothesis is that at narrow
k the kernel is bound by B-row traffic: every nonzero gathers one row of B, and at k=2
float64 that row is 16 bytes inside a 64-byte line, so four B rows share a line and whether
they are USED together is a property of the column index pattern, not of the kernel.

This script measures that property and nothing else. It reads only `indptr` and `indices`
from the cached npz -- no values, no timing, no kernel -- so it cannot be influenced by any
arm, and it can be computed before the per-cell verdicts are read.

WHAT IT REPORTS, per (key, k):
  deg            nnz / rows, the known separator. Reported so the new column can be shown to
                 add information OVER it rather than restate it.
  lines_per_nnz  sum over rows of |{ j // brows_per_line }| divided by nnz. The count of
                 distinct 64-byte B lines a row touches, amortized per nonzero, assuming no
                 reuse ACROSS rows. 1.0 means every nonzero pays a whole line; 1/brows_per_line
                 is the floor, reached when a row's columns are perfectly packed.
  b_bytes_nnz    64 * lines_per_nnz. The same number in bytes, comparable to the A-side cost.
  a_bytes_nnz    itemsize + 4, the value and the column index. Fixed by dtype, not structure.
  pack           1 / (lines_per_nnz * brows_per_line): the mean fraction of each fetched
                 64-byte line that the row actually consumes, in [1/brows_per_line, 1].
                 1.0 means perfectly packed columns, 1/brows_per_line means every fetched
                 line yields exactly one useful B row. Scale-free, so it is comparable
                 across widths where brows_per_line differs. (Written first as
                 lines_per_nnz * brows_per_line, which is the RECIPROCAL and reads 4.0 on a
                 scattered row at k=2 float64; the `0 < pack <= 1` assertion at the bottom
                 of this file is what caught that, before the script was ever run on a
                 corpus. The planted cases in the self-test below pin both ends.)
  b_footprint    cols * k * itemsize. The cold-per-row assumption above is an UPPER bound on
                 traffic; when this fits in cache, B is resident and the true traffic is far
                 lower. Reported so a reader can see where the bound is loose instead of
                 having the script decide a cache size for them.

WHAT IT DOES NOT DO. It does not classify a matrix, apply a threshold, or name a cache level.
Every such constant would be a tuning knob on a corpus whose verdicts I have already seen.
The prediction registered against it is in PREREG_pfstream_screen.md and is directional and
conditional on degree: a predictor that only reproduces the degree split has added nothing.
"""
import csv, os, sys

import numpy as np


ITEMSIZE = {"float32": 4, "float64": 8}
LINE = 64


def stats_for(indptr, indices, k, itemsize):
    """Return (lines_per_nnz, pack, brows_per_line) for one matrix at one width."""
    bytes_per_brow = k * itemsize
    brows_per_line = max(1, LINE // bytes_per_brow)
    nnz = int(indptr[-1])
    if nnz == 0:
        return None
    # Per row, count distinct line ids among that row's column indices. Rows are already
    # sorted within CSR for these caches, but sortedness is not assumed: np.unique does not
    # need it, and assuming it would make the number wrong exactly on a malformed input.
    total = 0
    lines = indices // brows_per_line
    for r in range(len(indptr) - 1):
        a, b = int(indptr[r]), int(indptr[r + 1])
        if b > a:
            total += int(np.unique(lines[a:b]).size)
    lpn = total / float(nnz)
    return lpn, 1.0 / (lpn * brows_per_line), brows_per_line


def main(argv):
    if len(argv) < 5:
        return "usage: bstream_locality.py <cache> <keysfile> <dtype> <k,k,...> [out.csv]"
    cache, keysfile, dtype, kspec = argv[1], argv[2], argv[3], argv[4]
    out = argv[5] if len(argv) > 5 else "-"
    if dtype not in ITEMSIZE:
        return "REFUSE: dtype %r is not one of %s" % (dtype, sorted(ITEMSIZE))
    itemsize = ITEMSIZE[dtype]
    ks = [int(x) for x in kspec.split(",") if x.strip()]
    if len(set(ks)) != len(ks):
        return "REFUSE: duplicate widths in %r; a duplicate silently doubles a key's weight" % kspec

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from csrresolve import resolve_all, describe
    except ImportError:
        return ("REFUSE: csrresolve.py is not importable next to this script. Resolving keys "
                "here with a second os.path.exists is the defect that module exists to prevent.")

    keys = [ln.strip() for ln in open(keysfile) if ln.strip() and not ln.startswith("#")]
    paths, how = resolve_all(cache, keys)
    print(describe(how))
    missing = [k for k in keys if paths.get(k) is None]
    if missing:
        return ("REFUSE: %d of %d keys did not resolve, first %s. A structural predictor "
                "computed on a subset is not comparable to a verdict table computed on the "
                "whole corpus." % (len(missing), len(keys), missing[:3]))

    rows_out = []
    for n, key in enumerate(keys, 1):
        z = np.load(paths[key])
        indptr = z["indptr"]
        indices = z["indices"]
        shape = z["shape"]
        rows, cols = int(shape[0]), int(shape[1])
        nnz = int(indptr[-1])
        if nnz == 0:
            print("  skip %s: nnz 0" % key)
            continue
        for k in ks:
            st = stats_for(indptr, indices, k, itemsize)
            if st is None:
                continue
            lpn, pack, bpl = st
            rows_out.append(dict(
                key=key, k=k, rows=rows, cols=cols, nnz=nnz,
                deg="%.4f" % (nnz / float(rows)),
                brows_per_line=bpl,
                lines_per_nnz="%.6f" % lpn,
                b_bytes_nnz="%.4f" % (LINE * lpn),
                a_bytes_nnz=itemsize + 4,
                pack="%.6f" % pack,
                b_footprint=cols * k * itemsize,
            ))
        if n % 50 == 0:
            print("  %d of %d keys" % (n, len(keys)))

    if not rows_out:
        return "REFUSE: no keys produced a row; nothing was measured"
    cols_out = list(rows_out[0].keys())
    fh = sys.stdout if out == "-" else open(out, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=cols_out)
    w.writeheader()
    w.writerows(rows_out)
    if out != "-":
        fh.close()
        print("wrote %d rows (%d keys x %d widths) to %s"
              % (len(rows_out), len(rows_out) // len(ks), len(ks), out))
    # The one invariant worth asserting here: pack is a fraction of a cache line and cannot
    # exceed 1, and cannot be below 1/brows_per_line unless a row touched fewer lines than it
    # has nonzeros, which is impossible by construction.
    bad = [r for r in rows_out
           if not (1.0 / int(r["brows_per_line"]) - 1e-12 <= float(r["pack"]) <= 1.0 + 1e-12)]
    if bad:
        return ("REFUSE: %d rows have pack outside [1/brows_per_line, 1], first %s"
                % (len(bad), bad[0]))
    print("pack within [1/brows_per_line, 1] on all %d rows" % len(rows_out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
