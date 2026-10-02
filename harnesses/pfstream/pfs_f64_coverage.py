"""Report the f64 screen's COVERAGE, deliberately without reading a single timing.

WHY A SEPARATE PROGRAM FROM THE ANALYSIS. The pre-registration for this screen says the
within-class agreement number of the alignment floor is read before any arm timing, because that
number decides whether a contrast is reported as a matched headline or only as a bound. Coverage
is not an arm timing: how many requested keys resolved, and how the corpus spreads over the
nonzero bands, are properties of the input. Putting them in a program that never touches the `ms`
column makes the separation checkable rather than promised -- the refusal below fires if `ms` is
read at all.

WHY THE BANDS MATTER. The A/A floor is an inverted U in matrix size: p95 near 38% at 1e4-1e6
nonzeros against about 6% at both ends. A floor pooled over bands bounds neither end, so band is
a reported axis, and an arm cell in a band the floor does not cover has no floor behind it.
"""
import collections
import csv
import os
import sys

BANDS = ((1e7, ">=1e7"), (1e6, "1e6-1e7"), (1e5, "1e5-1e6"),
         (1e4, "1e4-1e5"), (1e3, "1e3-1e4"))
FORBIDDEN = "ms"          # the arm timing; this program must not read it


def band(nnz):
    for lo, lab in BANDS:
        if nnz >= lo:
            return lab
    return "<1e3"


def band_order(lab):
    labs = ["<1e3"] + [b[1] for b in reversed(BANDS)]
    return labs.index(lab)


def read_coverage(path):
    """Key -> nnz, plus the arm names seen and what was skipped. Never reads FORBIDDEN."""
    nnz, arms, widths, builds, skipped = {}, collections.Counter(), set(), set(), 0
    with open(path) as fh:
        rd = csv.DictReader(fh)
        if FORBIDDEN not in rd.fieldnames:
            sys.exit("REFUSE: %s has no %r column, so this is not the schema this program was "
                     "written against and its column names cannot be trusted" % (path, FORBIDDEN))
        for row in rd:
            try:
                nnz[row["key"]] = int(row["nnz"])
                widths.add(int(row["k"]))
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            arms[row["arm"]] += 1
            builds.add(row["build"])
    return nnz, arms, widths, builds, skipped


def main(argv):
    if len(argv) < 2:
        return "usage: pfs_f64_coverage.py <csv-dir> [requested-keys-file]"
    d = argv[1]
    requested = argv[2] if len(argv) > 2 else None
    names = sorted(f for f in os.listdir(d) if f.endswith(".csv"))
    if not names:
        return "REFUSE: no CSV files in %s, so every count below would be zero" % d
    print("pass CSVs: %d in %s" % (len(names), d))

    per_file = {}
    for n in names:
        per_file[n] = read_coverage(os.path.join(d, n))
    # Every pass must see the same key set, or the per-build comparison is over different corpora.
    keysets = {n: frozenset(v[0]) for n, v in per_file.items()}
    distinct = set(keysets.values())
    print("distinct key sets across the %d pass files: %d" % (len(names), len(distinct)))
    if len(distinct) != 1:
        sizes = sorted({len(k) for k in distinct})
        base = max(distinct, key=len)
        for n in sorted(keysets):
            miss = base - keysets[n]
            if miss:
                print("  %s is missing %d key(s), e.g. %s"
                      % (n, len(miss), sorted(miss)[:4]))
        return ("REFUSE: the passes do not share one key set (sizes %s). A per-build contrast "
                "across passes with different corpora compares different problems." % sizes)

    nnz, arms, widths, builds, skipped = per_file[names[0]]
    tot_skipped = sum(v[4] for v in per_file.values())
    print("keys %d   widths %s   builds %s" % (len(nnz), sorted(widths), sorted(builds)))
    print("arm labels and row counts (names only, no timings): %s" % dict(arms))
    print("rows skipped for unparseable key/nnz/k, summed over files: %d" % tot_skipped)

    if requested:
        want = [ln.strip() for ln in open(requested) if ln.strip()]
        got = set(nnz)
        unresolved = [k for k in want if k not in got]
        print()
        print("RESOLVED OF REQUESTED: %d of %d" % (len(want) - len(unresolved), len(want)))
        if unresolved:
            print("  unresolved %d, e.g. %s" % (len(unresolved), unresolved[:6]))
            print("  ^ a headline must travel with this ratio: the denominator is the corpus the")
            print("    claim names, and the numerator is what was measured.")
        else:
            print("  every requested key resolved")

    print()
    print("BAND DISTRIBUTION of the measured corpus")
    by = collections.Counter(band(v) for v in nnz.values())
    for lab in sorted(by, key=band_order):
        ks = [k for k, v in nnz.items() if band(v) == lab]
        lo, hi = min(nnz[k] for k in ks), max(nnz[k] for k in ks)
        print("  %-9s %4d keys  nnz %d..%d" % (lab, by[lab], lo, hi))
    print("  total %d, and %d cells per build at %d widths" % (sum(by.values()), len(nnz) * len(widths), len(widths)))
    empty = [lab for _lo, lab in list(BANDS) + [(0, "<1e3")] if lab not in by]
    if empty:
        print("  BANDS WITH NO KEYS: %s -- no claim about those sizes is supported here" % empty)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
