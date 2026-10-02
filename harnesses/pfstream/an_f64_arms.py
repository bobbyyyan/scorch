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
        # A MEAN BESIDE THE MEDIAN, because a median over cells has a 50% breakdown point and
        # an arm that treats a minority of cells is invisible to it however large the effect
        # where it fires. That is not hypothetical here: on a different lever in this directory
        # the pooled median read 0.02-1.01% and the mean read 10-19% on the same readings,
        # because the treated share was 26%. The two together say which case this is -- they
        # agree when the treatment is broad and diverge when it is not.
        print("  %-18s %6s %9s %9s %9s %9s %9s" % ("contrast", "n", "median", "MEAN", "p05",
                                                    "p95", "better"))
        amean = lambda v: (sum(v) / len(v)) if v else float("nan")
        print("  %-18s %6d %8.3f%% %8.3f%% %8.3f%% %8.3f%% %8s   <- FLOOR, same config, 2 objects"
              % ("shpb vs ship", len(fl), 100 * pct(fl, 0.5), 100 * amean(fl),
                 100 * pct(fl, 0.05), 100 * pct(fl, 0.95),
                 "%d/%d" % (sum(1 for x in fl if x < 0), len(fl))))
        for a in arms:
            v = rel(a, "ship", pop)
            if not v:
                continue
            md, mn = 100 * pct(v, 0.5), 100 * amean(v)
            print("  %-18s %6d %8.3f%% %8.3f%% %8.3f%% %8.3f%% %8s%s"
                  % ("%s vs ship" % a, len(v), md, mn, 100 * pct(v, 0.05),
                     100 * pct(v, 0.95), "%d/%d" % (sum(1 for x in v if x < 0), len(v)),
                     "   <- median and mean DISAGREE in sign" if md * mn < 0 else ""))
        print("  (negative = faster than ship. 'better' counts cells where the arm is faster;")
        print("   the floor row's own count is the null -- a coin-flip there is ~half.)")

    # ---- PER WIDTH, because the recorded float32 result is that this effect FADES with width
    # and reverses somewhere above the widths measured here. A pooled number hides that shape,
    # and the shape is the whole question for whether the macro can ship.
    print()
    print("PER WIDTH over all cells (negative = faster than ship)")
    omitted = []
    widths = sorted({w for (_k, w) in cells})
    print("  %-6s %10s %10s %10s %12s %12s" % ("k", "floor", "nopf", "strm", "nopf better", "strm better"))
    for w in widths:
        pop = [c for c in cells if c[1] == w]
        fl = rel("shpb", "ship", pop)
        vn = rel("nopf", "ship", pop) if "nopf" in builds else []
        vs = rel("strm", "ship", pop) if "strm" in builds else []
        # The FLOOR column is the one the other two are read against, and it was the only one
        # without a guard. The up-front requirement that shpb exists is satisfied by one row
        # anywhere in the corpus; this table needs it present PER WIDTH, and a pass that dies
        # partway -- which is how a long run ends -- leaves a width with ship cells and no shpb
        # cells. Unguarded that is a TypeError mid-table, not a structured refusal.
        if not fl:
            # NOT "REFUSE". In this project REFUSE marks a fail-closed exit, and every other
            # occurrence in these files is returned from main into sys.exit and leaves status 1.
            # This path prints a row and continues, so the process still exits 0 -- wearing the
            # same token would make a skipped width indistinguishable from an aborted run to
            # anyone grepping logs for it.
            print("  k=%-4d %9s   OMITTED: no shpb/ship pair at this width, so the floor the two"
                  % (w, "-"))
            print("            arm columns are read against does not exist here; %d cell(s) have"
                  % len(pop))
            print("            a ship reading. No arm figure is reported for k=%d." % w)
            omitted.append(w)
            continue
        print("  k=%-4d %9.3f%% %9.3f%% %9.3f%% %11s %11s"
              % (w, 100 * pct(fl, 0.5),
                 100 * pct(vn, 0.5) if vn else float("nan"),
                 100 * pct(vs, 0.5) if vs else float("nan"),
                 "%d/%d" % (sum(1 for x in vn if x < 0), len(vn)) if vn else "-",
                 "%d/%d" % (sum(1 for x in vs if x < 0), len(vs)) if vs else "-"))
    if omitted:
        print("  WIDTHS OMITTED FOR WANT OF A FLOOR: %s -- the table above is not complete over"
              % omitted)
        print("  the grid, and this line is the only record of that.")
    print("  The floor column is the null for the two beside it. The recorded float32 result is")
    print("  that removing the prefetch wins 14.6%% at k=4 and fades to 0.56%% by k=16, and that it")
    print("  HARMS 115 matrices at k=32 -- a width absent from this grid, so this table cannot")
    print("  see the side of the trade that blocks shipping.")

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
