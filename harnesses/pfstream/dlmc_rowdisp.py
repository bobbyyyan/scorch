#!/usr/bin/env python3
"""Row-dispersion reader for the DLMC collection.

Condition 4 of PREREG_multirow_halfvec.md stratifies on row dispersion -- the
coefficient of variation of the per-row nonzero count. That quantity is not in any
ledger, so it has to be read off the matrices themselves.

A .smtx file is three lines: "rows, cols, nnz", then the CSR row pointer, then the
column indices. The row pointer alone gives every row's nonzero count, so this reads
two lines per file and never touches the indices.

The guard, and exactly what it does and does not cover. Every key's rows and nnz are
checked against the same key's `rows` and `nnz` columns in a results ledger, because a
substituted or differently-ordered corpus is invisible in a dispersion number -- it just
reads as a different distribution. A key that disagrees is reported and excluded, and a
nonzero exclusion count invalidates the run rather than shrinking it quietly.

But note which column that guard actually certifies. Once rows and nnz are validated and
the pointer is checked to sum to nnz, `mean_deg` is nnz/rows exactly -- a function of the
two validated scalars, carrying no information the ledger did not already hold. The three
columns anything here acts on -- row_cv, max_over_mean, empty_rows -- all depend on the
SHAPE of the row-length distribution, and no external record of row lengths exists in
this project to check that against. A corpus with correct per-key rows and nnz but
corrupted row pointers would pass the dimension check and produce wrong dispersion.

So the shape is constrained from inside the file instead, by the one header field that is
otherwise unused: every row's nonzero count must lie in [0, cols], since a row's entries
occupy distinct columns. That is free, it is a real constraint on the distribution rather
than on its mean, and it fires on precisely the corruption the dimension check cannot
see. It is weaker than an independent record and is not a substitute for one.

A ledger may qualify its keys with a leading component the filesystem path does not
carry -- chain207's DLMC keys begin "dlmc/" while a path relative to ~/dlmc does not,
and the redwood driver's key file is named "nopfx" for exactly that reason. So the
prefix is detected by trying both forms and reporting which one matched and how many
keys it hit. Detection that prints its own result is safe; silent detection is how a
corpus gets substituted without anyone noticing.

Usage: dlmc_rowdisp.py <dlmc-root> <ledger-dir> [out.csv]
Both paths are arguments. Neither is defaulted: a hardcoded ledger path has been the
cause of four silent-null harnesses in this project.
"""
import csv
import glob
import math
import os
import sys


def row_counts(path):
    """rows, cols, nnz, and the per-row nonzero counts, from the first two lines."""
    with open(path) as fh:
        head = fh.readline()
        ptr_line = fh.readline()
    rows, cols, nnz = (int(x) for x in head.replace(",", " ").split())
    ptr = [int(x) for x in ptr_line.split()]
    if len(ptr) != rows + 1:
        raise ValueError("row pointer has %d entries, header says %d rows"
                         % (len(ptr), rows + 1))
    counts = [ptr[i + 1] - ptr[i] for i in range(rows)]
    # The shape constraint. A row's nonzeros sit in distinct columns, so no count may
    # exceed cols, and a negative count means a non-monotone pointer. Neither is implied
    # by the header agreeing with the ledger.
    worst = max(counts) if counts else 0
    if worst > cols:
        raise ValueError("a row holds %d nonzeros but the matrix has %d columns" % (worst, cols))
    if min(counts, default=0) < 0:
        raise ValueError("row pointer is not monotone: smallest row length is %d"
                         % min(counts))
    return rows, cols, nnz, counts


def dispersion(counts):
    n = len(counts)
    total = sum(counts)
    mean = total / n
    if mean == 0.0:
        # Every row is empty exactly when the mean is zero, so the count is n, not 0.
        # Unreachable on this corpus -- the smallest key has 6 nonzeros -- but a zero
        # here would read as "no empty rows" on the one input where all of them are.
        return mean, float("nan"), float("nan"), n
    var = sum((c - mean) ** 2 for c in counts) / n
    cv = math.sqrt(var) / mean
    return mean, cv, max(counts) / mean, sum(1 for c in counts if c == 0)


def ledger_dims(ledger_dir):
    """key -> (rows, nnz) from every CSV in the ledger that carries those columns.

    Comment lines are dropped before parsing. A ledger CSV may carry a `#` preamble, and
    `csv.DictReader` then takes the comment text as the field names, so the file is
    skipped as "lacking the columns" while sitting right there holding them. Pointed at a
    directory of only such files the reader fails closed, which is correct; pointed at a
    directory holding both kinds it would quietly use the smaller population. Every file
    that IS skipped is now named, because a silently smaller key count is the failure this
    whole guard exists to prevent.
    """
    dims, conflict, skipped = {}, [], []
    for path in sorted(glob.glob(os.path.join(ledger_dir, "*.csv"))):
        with open(path) as fh:
            lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        rdr = csv.DictReader(lines)
        if True:
            if not rdr.fieldnames or not {"key", "rows", "nnz"} <= set(rdr.fieldnames):
                skipped.append((os.path.basename(path), rdr.fieldnames))
                continue
            for rec in rdr:
                got = (int(rec["rows"]), int(rec["nnz"]))
                was = dims.setdefault(rec["key"], got)
                if was != got:
                    conflict.append((rec["key"], was, got))
    if skipped:
        print("   CSVs in the ledger WITHOUT key/rows/nnz, named rather than dropped silently:")
        for nm, flds in skipped:
            print("      %s  fields=%r" % (nm, (flds or [])[:3]))
    return dims, conflict


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    root, ledger_dir = argv[1], argv[2]
    out = argv[3] if len(argv) > 3 else None

    dims, conflict = ledger_dims(ledger_dir)
    if conflict:
        print("LEDGER DISAGREES WITH ITSELF on %d key(s); first: %r" % (len(conflict), conflict[0]))
        return 1
    print("ledger %s: %d keys carry rows and nnz" % (ledger_dir, len(dims)))
    if not dims:
        print("EXIT: the ledger supplied no dimensions, so nothing can be checked")
        return 1

    files = sorted(glob.glob(os.path.join(root, "**", "*.smtx"), recursive=True))
    print("corpus %s: %d .smtx files" % (root, len(files)))
    if not files:
        print("EXIT: the corpus glob matched nothing")
        return 1

    rel = [os.path.relpath(f, root)[: -len(".smtx")] for f in files]
    cands = [("", ""), (os.path.basename(os.path.normpath(root)) + "/", "basename of the root")]
    hits = [(sum(1 for r in rel if pre + r in dims), pre, why) for pre, why in cands]
    for n, pre, why in hits:
        print("  key prefix %-8r (%s): matches %d of %d ledger keys"
              % (pre, why or "path relative to the root, as-is", n, len(dims)))
    best, prefix, _ = max(hits)
    if best == 0:
        print("EXIT: no prefix matched a single ledger key -- wrong corpus or wrong ledger")
        return 1
    print("  using prefix %r" % prefix)

    recs, mismatch, unread, nokey = [], [], [], 0
    for path in files:
        key = prefix + os.path.relpath(path, root)[: -len(".smtx")]
        try:
            rows, cols, nnz, counts = row_counts(path)
        except Exception as exc:                                  # noqa: BLE001
            unread.append((key, str(exc)))
            continue
        if sum(counts) != nnz:
            unread.append((key, "row pointer sums to %d, header says %d" % (sum(counts), nnz)))
            continue
        if key not in dims:
            nokey += 1
            continue
        if dims[key] != (rows, nnz):
            mismatch.append((key, dims[key], (rows, nnz)))
            continue
        mean, cv, mx, empty = dispersion(counts)
        recs.append(dict(key=key, rows=rows, cols=cols, nnz=nnz, mean_deg=mean,
                         row_cv=cv, max_over_mean=mx, empty_rows=empty))

    print("read and dimension-checked: %d" % len(recs))
    print("  in the corpus but not in the ledger (not an error, the ledger is a subset): %d" % nokey)
    print("  UNREADABLE or internally inconsistent: %d" % len(unread))
    for k, why in unread[:5]:
        print("    %s: %s" % (k, why))
    print("  DIMENSION MISMATCH against the ledger: %d" % len(mismatch))
    for k, was, got in mismatch[:5]:
        print("    %s: ledger (rows,nnz)=%r  file %r" % (k, was, got))

    missing = sorted(set(dims) - {r["key"] for r in recs} - {k for k, _ in mismatch})
    print("  ledger keys with no matching file: %d" % len(missing))
    for k in missing[:5]:
        print("    %s" % k)

    ok = not unread and not mismatch and not missing
    print("GUARD: %s" % ("every ledger key read and agrees on rows and nnz"
                         if ok else "FAILED -- the numbers below are NOT trustworthy"))

    if recs:
        cvs = sorted(r["row_cv"] for r in recs if r["row_cv"] == r["row_cv"])
        qs = [(q, cvs[int(round(q * (len(cvs) - 1)))]) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
        print("row_cv over %d keys: %s" % (len(cvs), "  ".join("q%g=%.4f" % (q, v) for q, v in qs)))
        print("  (quantile index = int(round(q*(n-1))); at even n that is the lower middle")
        print("   for n = 2 mod 4 and the upper for n = 0 mod 4 -- stated because it varies)")

    if out and not ok:
        print("NOT WRITING %s: a file on disk outlives the warning printed above it" % out)
    elif out and recs:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        print("wrote %s" % out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
