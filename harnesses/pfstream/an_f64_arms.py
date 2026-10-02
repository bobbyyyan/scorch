"""Do the prefetch arms move the float64 narrow-k deficit, judged against this screen's OWN floor.

THE FLOOR IS IN THE SCREEN, NOT IMPORTED. `shpb` is a second build at the IDENTICAL
configuration as `ship` -- same source, same defines, a different object at an equal-length build
path. So ship-vs-shpb is an inter-object A/A measured on these exact cells, at these exact
widths, on this host, in this run. That is the right floor for an arm whose macro changes emitted
code, and it needs no constant carried from another host or another evening. The 16-member padded
family on redwood answers a different question (how placement alone spreads within one alignment
class) and its numbers are not a bound for MKT cells.

WHY MEDIAN OVER PASSES. Each build ran four passes in a Latin square, so each (cell, build) has
four readings at four different wall-clock positions. The median of four is used rather than the
mean so one pass landing next to a co-tenant does not set the value.

WHAT WOULD COUNT. An arm is a candidate on the deficit set only if its improvement over ship
exceeds the ship-vs-shpb spread on the SAME cells. Reporting an arm effect without the floor
beside it on the same population is the error this file exists to avoid.
"""
import collections
import csv
import os
import statistics
import sys

BANDS = ((1e7, ">=1e7"), (1e6, "1e6-1e7"), (1e5, "1e5-1e6"),
         (1e4, "1e4-1e5"), (1e3, "1e3-1e4"))


def band(nnz):
    for lo, lab in BANDS:
        if nnz >= lo:
            return lab
    return "<1e3"


def pct(v, q):
    if not v:
        return None
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def main(argv):
    if len(argv) < 2:
        return "usage: an_f64_arms.py <csv-dir>"
    d = argv[1]
    # (key, k, build) -> [ms per pass]; and the mkl reading per (key, k, build)
    ours = collections.defaultdict(list)
    mkl = collections.defaultdict(list)
    nnz, degree = {}, {}
    skipped = collections.Counter()
    for f in sorted(os.listdir(d)):
        if not f.endswith(".csv"):
            continue
        with open(os.path.join(d, f)) as fh:
            for r in csv.DictReader(fh):
                try:
                    key, k, b, a, t = r["key"], int(r["k"]), r["build"], r["arm"], float(r["ms"])
                    nnz[key] = int(r["nnz"])
                    degree[key] = float(r["deg"])
                except (KeyError, TypeError, ValueError):
                    skipped["unparseable"] += 1
                    continue
                if t <= 0:
                    skipped["non-positive"] += 1
                    continue
                if a == "ours":
                    ours[(key, k, b)].append(t)
                elif a == "mkl":
                    mkl[(key, k, b)].append(t)
    builds = sorted({b for (_k, _w, b) in ours})
    print("builds %s   skipped %s" % (builds, dict(skipped) or "{}"))
    for need in ("ship", "shpb"):
        if need not in builds:
            return "REFUSE: build %r absent; the floor pair is ship vs shpb" % need
    npass = collections.Counter(len(v) for v in ours.values())
    print("passes per (cell, build): %s" % dict(npass))

    med = {ck: statistics.median(v) for ck, v in ours.items()}
    medm = {ck: statistics.median(v) for ck, v in mkl.items()}

    cells = sorted({(k, w) for (k, w, _b) in ours})
    # the deficit set: behind MKL in every reading of every build
    behind = set()
    for (key, w) in cells:
        rs = []
        for b in builds:
            o, m = ours.get((key, w, b)), mkl.get((key, w, b))
            if o and m and len(o) == len(m):
                rs += [x / y for x, y in zip(o, m)]
        if rs and all(r > 1.0 for r in rs):
            behind.add((key, w))
    print("cells %d   deficit cells (behind MKL in every reading) %d" % (len(cells), len(behind)))

    def rel(a, b, pop):
        """Signed relative change of build a against build b, per cell, over a population."""
        out = []
        for (key, w) in pop:
            x, y = med.get((key, w, a)), med.get((key, w, b))
            if x and y and y > 0:
                out.append((x - y) / y)
        return sorted(out)

    arms = [b for b in builds if b not in ("ship", "shpb")]
    pops = [("ALL CELLS", cells), ("DEFICIT CELLS", sorted(behind)),
            ("DEFICIT, k=2", sorted(c for c in behind if c[1] == 2)),
            ("DEFICIT, deg>=5", sorted(c for c in behind if degree.get(c[0], 0) >= 5))]
    for label, pop in pops:
        if not pop:
            continue
        print()
        print("%s  (n=%d)" % (label, len(pop)))
        fl = rel("shpb", "ship", pop)
        print("  %-18s %6s %9s %9s %9s %9s" % ("contrast", "n", "median", "p05", "p95", "better")
              )
        print("  %-18s %6d %8.3f%% %8.3f%% %8.3f%% %8s   <- FLOOR, same config, 2 objects"
              % ("shpb vs ship", len(fl), 100 * pct(fl, 0.5), 100 * pct(fl, 0.05),
                 100 * pct(fl, 0.95), "%d/%d" % (sum(1 for x in fl if x < 0), len(fl))))
        for a in arms:
            v = rel(a, "ship", pop)
            if not v:
                continue
            print("  %-18s %6d %8.3f%% %8.3f%% %8.3f%% %8s"
                  % ("%s vs ship" % a, len(v), 100 * pct(v, 0.5), 100 * pct(v, 0.05),
                     100 * pct(v, 0.95), "%d/%d" % (sum(1 for x in v if x < 0), len(v))))
        print("  (negative = faster than ship. 'better' counts cells where the arm is faster;")
        print("   the floor row's own count is the null -- a coin-flip there is ~half.)")

    # ---- the decisive question, stated as a comparison and not a verdict.
    print()
    print("DOES ANY ARM CLEAR ITS OWN FLOOR ON THE DEFICIT SET?")
    pop = sorted(behind)
    fl = rel("shpb", "ship", pop)
    if not fl:
        return "REFUSE: the floor pair has no cells on the deficit set"
    fl_med, fl_p05 = pct(fl, 0.5), pct(fl, 0.05)
    print("  floor median %.3f%%, floor p05 %.3f%% (the floor's own best-case tail)"
          % (100 * fl_med, 100 * fl_p05))
    for a in arms:
        v = rel(a, "ship", pop)
        if not v:
            continue
        m = pct(v, 0.5)
        beats = sum(1 for x in v if x < fl_p05)
        print("  %-5s median %+.3f%%  vs floor median %+.3f%%  ->  %s"
              % (a, 100 * m, 100 * fl_med,
                 "INSIDE the floor" if abs(m) <= abs(fl_med) else
                 ("faster beyond the floor median" if m < fl_med else "slower beyond the floor median")))
        print("        cells beyond the floor's p05 tail: %d of %d (%.1f%%)"
              % (beats, len(v), 100.0 * beats / len(v)))
    print()
    print("  A median inside the floor is not evidence of nothing -- it is evidence that this")
    print("  screen cannot resolve the arm on this population, which is a different statement.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
