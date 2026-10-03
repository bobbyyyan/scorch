"""The A/A floor's band, computed by ONE script over ledgers with different schemas.

WHY THIS EXISTS. The floor is quoted in three screens -- MKT float32/DLMC, MKT float64/SuiteSparse,
redwood float64/SuiteSparse -- and the fourth cell of that 2x2 (redwood float32/DLMC) was sealed
months earlier under a different column layout. Reading the sealed numbers out of their own
analysis files would compare two FUNCTIONALS, not two hosts: `an_f64_arms.py` takes the per-cell
median over readings for each build and then the p05/p95 of the signed relative difference over
cells, and nothing guarantees an older analyser did the same. So this script recomputes every cell
from the raw rows with the one definition, and REFUSES on a file that cannot supply it.

The functional, stated once so it cannot drift between invocations:
    per cell (key, k):  m[b] = median over all rows of build b, chosen arm
    per cell:           d = (m[B] - m[A]) / m[A]
    per width:          median(d), p05(d), p95(d), and the count of d < 0
This matches `an_f64_arms.py`'s `rel(a, b, pop)` followed by `pct(.., 0.05/0.5/0.95)` exactly.

WHAT A FLOOR PAIR IS, and why the pair name is an argument rather than a default: the two builds
must be the same configuration built separately, so their `.text` is identical and only the object
differs. `pfs_build.sh` REFUSES when its ship/shpb pair disagrees in `.text`, and the sealed
ledgers record the same construction (chain207 `aa`/`bb` "two builds with byte-identical .text in
separate processes"; chain211 `base`/`bsb_` "that pair IS the floor"). A pair that is NOT of this
construction measures something else, and this script cannot tell from the rows alone -- so it
prints the pair it was given beside every number and never guesses one.
"""
import argparse
import collections
import csv
import glob
import os
import statistics as st

NEED = {"build", "key", "k", "arm", "ms"}


def pct(v, q):
    if not v:
        return float("nan")
    i = int(q * (len(v) - 1))
    return v[i]


def load(paths, arm, keep=None, strip=""):
    """rows -> {(key, k, build): [ms, ...]}, plus a per-file census so an empty read cannot pass."""
    acc = collections.defaultdict(list)
    census = []
    for p in paths:
        with open(p, newline="") as fh:
            rd = csv.DictReader(fh)
            hdr = set(rd.fieldnames or [])
            if not NEED <= hdr:
                raise SystemExit("REFUSE: %s lacks %s, so the floor functional is not computable "
                                 "from it" % (os.path.basename(p), ",".join(sorted(NEED - hdr))))
            n = kept = armhit = 0
            for r in rd:
                n += 1
                if r.get("arm") != arm:
                    continue
                armhit += 1
                try:
                    ms = float(r["ms"])
                    k = int(r["k"])
                except (TypeError, ValueError):
                    continue
                if ms <= 0:
                    continue
                key = r["key"]
                if strip and key.startswith(strip):
                    key = key[len(strip):]
                if keep is not None and key not in keep:
                    continue
                acc[(key, k, r["build"])].append(ms)
                kept += 1
            census.append((os.path.basename(p), n, kept, armhit))
    return acc, census


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding the CSVs")
    ap.add_argument("--glob", default="*.csv")
    ap.add_argument("--pair", required=True, help="A,B -- the two builds of the floor pair")
    ap.add_argument("--arm", default="ours", help="which arm's timings (default: ours)")
    ap.add_argument("--label", default="", help="what cell of the 2x2 this is")
    ap.add_argument("--keys", default="", help="restrict to these keys, one per line. THE BAND IS "
                    "A SPREAD OVER CELLS, so it depends on the corpus's size mix and two cells of "
                    "the 2x2 are not comparable unless the key SET matches, not merely the corpus "
                    "name. Without this the script prints the band over whatever is present and "
                    "says so.")
    ap.add_argument("--strip-prefix", default="", help="drop this prefix from the CSV's key column "
                    "before matching --keys; the sealed ledgers carry 'dlmc/' and the screen lists "
                    "do not, and an unnormalised join reads as a legitimate zero-overlap refusal")
    ap.add_argument("--widths", default="", help="restrict to these widths, comma separated; use "
                    "when the two cells were swept at different width sets")
    A = ap.parse_args(argv)
    a, b = [x.strip() for x in A.pair.split(",")]

    paths = sorted(glob.glob(os.path.join(A.dir, A.glob)))
    if not paths:
        raise SystemExit("REFUSE: %s/%s matched no file" % (A.dir, A.glob))
    keep = None
    if A.keys:
        keep = set(l.strip() for l in open(A.keys) if l.strip())
        if not keep:
            raise SystemExit("REFUSE: %s is empty, so restricting to it would drop every cell"
                             % A.keys)
    acc, census = load(paths, A.arm, keep, A.strip_prefix)

    print("LABEL          %s" % (A.label or "(unlabelled)"))
    print("DIR            %s" % A.dir)
    print("FLOOR PAIR     %s vs %s        ARM  %s" % (b, a, A.arm))
    print("FILES          %d matched %r" % (len(paths), A.glob))
    tot = sum(c[1] for c in census)
    kep = sum(c[2] for c in census)
    ahit = sum(c[3] for c in census)
    print("ROWS           %d read, %d carrying arm=%r, %d surviving the key/shape filters"
          % (tot, ahit, A.arm, kep))
    if ahit == 0:
        raise SystemExit("REFUSE: no row carries arm=%r, so every band below would be empty. The "
                         "arm column is the cause here, NOT the key list -- %d rows were read."
                         % (A.arm, tot))
    if kep == 0:
        raise SystemExit("REFUSE: %d rows carry arm=%r but none survived the key filter, so the "
                         "cause is the JOIN and not the arm. Check --strip-prefix: the sealed "
                         "ledgers key on 'dlmc/<model>/...' and the screen lists on '<model>/...'."
                         % (ahit, A.arm))

    # The builds actually present. A pair name that matches nothing is the way this reads as a
    # pass on the wrong data, so it is a REFUSAL and the available names are printed with it.
    have = sorted({kk[2] for kk in acc})
    print("BUILDS PRESENT %s" % ", ".join("%s" % x for x in have))
    missing = [x for x in (a, b) if x not in have]
    if missing:
        raise SystemExit("REFUSE: build(s) %s absent; the pair named cannot be the floor here"
                         % ", ".join(missing))

    if keep is not None:
        got = {kk[0] for kk in acc}
        print("KEY SET        %d requested, %d matched%s" % (len(keep), len(got),
              ("  (prefix %r stripped)" % A.strip_prefix) if A.strip_prefix else ""))
        if not got:
            raise SystemExit("REFUSE: the key list matched 0 rows. A zero-overlap join is the way "
                             "a prefix mismatch reads as a legitimate empty result -- compare a "
                             "sample key from each side before concluding the corpora differ.")
        if len(got) < 0.5 * len(keep):
            print("  WARNING: under half the requested keys are present, so this band is over a "
                  "DIFFERENT population than the list names and is not comparable cell-to-cell.")
    else:
        print("KEY SET        unrestricted -- the band below is over whatever this ledger holds,")
        print("               which is NOT comparable with another cell of the 2x2 unless the two")
        print("               key sets happen to coincide. Pass --keys to make it comparable.")

    med = {kk: st.median(v) for kk, v in acc.items()}
    widths = sorted({kk[1] for kk in med})
    if A.widths:
        want = set(int(x) for x in A.widths.split(","))
        skipped = [w for w in widths if w not in want]
        widths = [w for w in widths if w in want]
        if not widths:
            raise SystemExit("REFUSE: --widths %s selected none of %s" % (A.widths, skipped))
        if skipped:
            print("WIDTHS DROPPED %s  (present here, absent from the cell being compared)"
                  % ", ".join(str(w) for w in skipped))
    print("WIDTHS         %s" % ", ".join(str(w) for w in widths))
    print()
    print("THE BAND, per width. 'cells' is the number carrying BOTH builds -- printed because a")
    print("band over few cells is not comparable with one over many, and an empty width would")
    print("otherwise show as a row of nan that reads like a quiet zero.")
    print("  %4s %7s %10s %10s %10s %10s %9s"
          % ("k", "cells", "median", "p05", "p95", "width pp", "better"))
    out = {}
    for w in widths:
        keys = sorted({kk[0] for kk in med if kk[1] == w})
        d = []
        for key in keys:
            x, y = med.get((key, w, b)), med.get((key, w, a))
            if x and y and y > 0:
                d.append((x - y) / y)
        d.sort()
        if not d:
            print("  %4d %7d   no cell carries both builds at this width" % (w, 0))
            continue
        lo, hi = pct(d, 0.05), pct(d, 0.95)
        out[w] = (len(d), pct(d, 0.5), lo, hi)
        print("  %4d %7d %9.3f%% %9.3f%% %9.3f%% %9.3f %5d/%-4d"
              % (w, len(d), 100 * pct(d, 0.5), 100 * lo, 100 * hi, 100 * (hi - lo),
                 sum(1 for x in d if x < 0), len(d)))

    # The two functionals the 2x2 separates, named so a reader cannot take one for the other.
    if out:
        locs = [v[1] for v in out.values()]
        disps = [100 * (v[3] - v[2]) for v in out.values()]
        print()
        print("THE TWO FUNCTIONALS, which is the whole point of the 2x2: LOCATION is where the")
        print("band sits and DISPERSION is how wide it is. They are reported separately because")
        print("the three earlier screens had them tracking DIFFERENT factors -- location moving")
        print("with dtype/corpus and dispersion with host -- and a single summary number hides")
        print("which of the two a new cell agrees with.")
        print("  LOCATION    median across widths %+.3f%%   range %+.3f%% to %+.3f%%   %s"
              % (100 * st.median(locs), 100 * min(locs), 100 * max(locs),
                 "all positive" if min(locs) > 0 else
                 ("all negative" if max(locs) < 0 else "STRADDLES ZERO")))
        print("  DISPERSION  p05-p95 width %.2f to %.2f pp across %d widths"
              % (min(disps), max(disps), len(disps)))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
