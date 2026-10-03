#!/usr/bin/env python3
"""Row-dispersion reader for the DLMC collection.

Condition 4 of PREREG_multirow_halfvec.md stratifies on row dispersion -- the
coefficient of variation of the per-row nonzero count. That quantity is not in any
ledger, so it has to be read off the matrices themselves.

A .smtx file is three lines: "rows, cols, nnz", then the CSR row pointer, then the
column indices. The row pointer alone gives every row's nonzero count, so this reads
two lines per file and never touches the indices.

The guard that makes the output trustworthy: every key's rows and nnz are checked
against the same key's `rows` and `nnz` columns in a results ledger. A substituted or
differently-ordered corpus is invisible in a dispersion number -- it just reads as a
different distribution -- so the only thing that separates a real reading from a wrong
corpus is agreement with an independent record of the same matrices. A key whose rows
or nnz disagree is reported and excluded; a nonzero exclusion count invalidates the run
rather than shrinking it quietly.

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
    return rows, cols, nnz, counts


def dispersion(counts):
    n = len(counts)
    total = sum(counts)
    mean = total / n
    if mean == 0.0:
        return mean, float("nan"), float("nan"), 0
    var = sum((c - mean) ** 2 for c in counts) / n
    cv = math.sqrt(var) / mean
    return mean, cv, max(counts) / mean, sum(1 for c in counts if c == 0)


def ledger_dims(ledger_dir):
    """key -> (rows, nnz) from every CSV in the ledger that carries those columns."""
    dims, conflict = {}, []
    for path in sorted(glob.glob(os.path.join(ledger_dir, "*.csv"))):
        with open(path) as fh:
            rdr = csv.DictReader(fh)
            if not rdr.fieldnames or not {"key", "rows", "nnz"} <= set(rdr.fieldnames):
                continue
            for rec in rdr:
                got = (int(rec["rows"]), int(rec["nnz"]))
                was = dims.setdefault(rec["key"], got)
                if was != got:
                    conflict.append((rec["key"], was, got))
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
