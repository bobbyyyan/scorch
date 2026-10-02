"""How fast does a quantifier-over-repetitions estimator drift with the repetition count?

WHY THIS EXISTS. `behind in every reading` is an INTERSECTION over readings, so it shrinks
monotonically as readings are added, and `ahead in every reading` does too. An estimator defined
by a quantifier over repetitions silently encodes the repetition count in its own value, and the
drift is always toward the null. So two such counts are comparable only at equal N.

That matters for a specific open question: a deficit of 192 cells recorded elsewhere against 0
on a 16-reading screen could be a host difference OR could be the same host read at low N. This
measures the slope on passes already on disk, which separates those without a new measurement.

WHAT IT CANNOT DO, stated because the limit is the point. It cannot reproduce a count larger than
this dataset's mixed set: a cell ahead in all 16 readings is ahead in each one individually, so no
sub-selection reaches past it. The output prints that ceiling beside every row.
"""
import argparse
import collections
import csv
import itertools
import os
import random
import statistics
import sys


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("csvdir")
    ap.add_argument("--ns", default="1,2,4,8,12,16")
    ap.add_argument("--subsets", type=int, default=40, help="random pass-subsets per N")
    ap.add_argument("--seed", type=int, default=20261002)
    A = ap.parse_args(argv[1:])

    files = sorted(f for f in os.listdir(A.csvdir) if f.endswith(".csv"))
    if not files:
        return "REFUSE: no CSV in %s" % A.csvdir
    # (key, k) -> {pass-file: ours/mkl}
    # PARSED THE WAY an_f64_vs_mkl.py PARSES IT, deliberately: `arm` is "ours" or "mkl" and the
    # two are paired WITHIN one file, so a ratio never crosses a pass boundary. A tool that
    # re-derived this differently would answer a different question and look like it agreed.
    cell = collections.defaultdict(dict)
    unpaired = 0
    for f in files:
        by = {}
        for r in csv.DictReader(open(os.path.join(A.csvdir, f))):
            try:
                by[(r["key"], int(r["k"]), r["arm"])] = float(r["ms"])
            except (KeyError, TypeError, ValueError):
                continue
        for (key, k, a), t in by.items():
            if a != "ours":
                continue
            m = by.get((key, k, "mkl"))
            if m and t > 0 and m > 0:
                cell[(key, k)][f] = t / m
            else:
                unpaired += 1
    if not cell:
        return "REFUSE: no (ours, mkl) pair parsed, so every count below would be 0"
    print("ratios built %d, rows with no mkl partner %d"
          % (sum(len(v) for v in cell.values()), unpaired))

    full = sorted(files)
    nfull = len(full)
    print("csvdir %s" % A.csvdir)
    print("pass-files %d   cells %d" % (nfull, len(cell)))
    complete = [c for c, d in cell.items() if len(d) == nfull]
    print("cells with all %d readings: %d of %d (only these are used, so N is exact)"
          % (nfull, len(complete), len(cell)))
    if not complete:
        return "REFUSE: no cell carries all readings, so no N is exact"

    # THE CEILING, printed before the table so no row can be read as reaching past it.
    behind_any = sum(1 for c in complete if any(cell[c][f] > 1.0 for f in full))
    ahead_all = sum(1 for c in complete if all(cell[c][f] < 1.0 for f in full))
    print("CEILING: %d cells are ahead in ALL %d readings, so no sub-selection can call them"
          % (ahead_all, nfull))
    print("         behind. The most any N can report is %d." % (len(complete) - ahead_all))
    print("         (cells behind in at least ONE reading: %d)" % behind_any)
    print()
    rng = random.Random(A.seed)
    print("  %3s %9s %9s %9s %9s %9s   %s"
          % ("N", "subsets", "min", "median", "max", "mean", "behind-in-all over pass-subsets"))
    for n in [int(x) for x in A.ns.split(",")]:
        if n > nfull:
            print("  %3d   SKIPPED: only %d pass-files exist" % (n, nfull))
            continue
        combos = list(itertools.combinations(range(nfull), n))
        if len(combos) > A.subsets:
            combos = [tuple(rng.sample(range(nfull), n)) for _ in range(A.subsets)]
        vals = []
        for cb in combos:
            sub = [full[i] for i in cb]
            vals.append(sum(1 for c in complete if all(cell[c][f] > 1.0 for f in sub)))
        print("  %3d %9d %9d %9.1f %9d %9.2f"
              % (n, len(vals), min(vals), statistics.median(vals), max(vals),
                 sum(vals) / len(vals)))
    print()
    print("  The count can only FALL as N rises, by construction: an intersection over more")
    print("  readings is a subset. A difference between two such counts measured at different N")
    print("  is therefore not evidence of anything until N is equated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
