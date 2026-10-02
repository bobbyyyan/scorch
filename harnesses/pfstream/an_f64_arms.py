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
import math
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


def boot_median_ci(vals, nboot=4000, seed=20261002, alpha=0.05):
    """Percentile bootstrap CI for the MEDIAN of `vals`, with digests of BOTH data and resample.

    REPRODUCING A BOUND NEEDS THE VALUES AND THE INDICES, and an earlier version digested only
    the indices. That digest is a pure function of (seed, nboot, n): a -0.50% dataset and a
    +9.00% dataset, both n=392 at the default seed, returned the same sixteen characters. So it
    pinned the resample and not the bound, a substituted dataset reproduced it exactly, and a
    seven-width table at one n printed the same string on every row -- which reads as a constant.

    Returns (lo, hi, vdig, idig, n). `vdig` digests the data the bound is OF; `idig` digests the
    resample. Quote the bound with both: vdig alone cannot show the resample was the registered
    one, and idig alone cannot show the data was.
    """
    import hashlib
    import random
    n = len(vals)
    if n < 8:
        return None, None, None, None, n
    vh = hashlib.sha256(("|".join("%.17g" % v for v in vals)).encode())
    rng = random.Random(seed)
    ih = hashlib.sha256()
    meds = []
    for _ in range(nboot):
        idx = [rng.randrange(n) for _ in range(n)]
        ih.update((",".join(map(str, idx))).encode())
        r = sorted(vals[i] for i in idx)
        meds.append(r[n // 2] if n % 2 else 0.5 * (r[n // 2 - 1] + r[n // 2]))
    meds.sort()
    lo = meds[int(alpha / 2 * (nboot - 1))]
    hi = meds[int((1 - alpha / 2) * (nboot - 1))]
    return lo, hi, vh.hexdigest()[:12], ih.hexdigest()[:12], n


def paired_d(med, pop, arm, floor, base):
    """Per-cell paired difference: (arm vs base) - (floor vs base), on cells carrying all three.

    EXTRACTED SO IT CAN BE TESTED. Inline, the production construction had no assertion
    coverage, and the material mutation is not an arithmetic typo -- swapping `arm` and `floor`
    flips the sign of every verdict and no check would fire. The self-test calls this with an
    ASYMMETRIC synthetic `med`, so a swapped role changes the sign and an assertion notices.
    """
    out = []
    for (key, w) in pop:
        a = med.get((key, w, arm))
        f = med.get((key, w, floor))
        b = med.get((key, w, base))
        if a and f and b and b > 0:
            out.append((a - b) / b - (f - b) / b)
    return sorted(out)


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
    signrows = []
    widths = sorted({w for (_k, w) in cells})
    # THE FLOOR COLUMN IS A MEDIAN, SO IT IS A LOCATION AND NOT A WIDTH. Two objects with the
    # SAME configuration differ by a systematic offset here (the layout artefact), so comparing
    # |arm| against |floor median| treats an offset as a noise band -- which is how a committed
    # amendment came to call an arm of -0.973% "outside" a floor of +1.381%. The floor's p05/p95
    # print beside its median so a magnitude claim has a width to be read against.
    #
    # AND THE SIGN TEST NEEDS THE FLOOR PAIR'S OWN cells-better, not 0.5. A pooled effect inside
    # the floor's band can still be directionally real, and cells-better is the statistic that
    # sees it -- but only if the same-config pair sits near half. If the pair that differs by
    # nothing but its object ALSO reads 75% better, that share is the artefact, not an effect.
    print("  %-5s %8s %8s %8s %8s %8s | %10s %10s %10s | %s"
          % ("k", "floor", "fl p05", "fl p95", "nopf", "strm",
             "nopf b/n", "strm b/n", "FLOOR b/n", "nopf vs band"))
    print("  'nopf vs band' is the only magnitude verdict this table supports: the arm's median")
    print("  against the floor pair's p05-p95 BAND. Chosen after the median-magnitude comparison")
    print("  was found wrong in a committed amendment, and labelled as post-hoc for that reason;")
    print("  it is the reading of the standing rule that nothing inside the noise floor counts.")
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
        lo, hi = pct(fl, 0.05), pct(fl, 0.95)
        if vn:
            mn = pct(vn, 0.5)
            if mn < lo:
                vd = "FASTER, outside"
            elif mn > hi:
                vd = "SLOWER, outside"
            else:
                vd = "inside, no verdict"
            vd += " (%+.3f vs %+.3f)" % (100 * mn, 100 * (lo if mn < lo else hi))
        else:
            vd = "-"
        print("  k=%-4d %7.3f%% %7.3f%% %7.3f%% %7.3f%% %7.3f%% | %10s %10s %10s | %s"
              % (w, 100 * pct(fl, 0.5), 100 * lo, 100 * hi,
                 100 * pct(vn, 0.5) if vn else float("nan"),
                 100 * pct(vs, 0.5) if vs else float("nan"),
                 "%d/%d" % (sum(1 for x in vn if x < 0), len(vn)) if vn else "-",
                 "%d/%d" % (sum(1 for x in vs if x < 0), len(vs)) if vs else "-",
                 "%d/%d" % (sum(1 for x in fl if x < 0), len(fl)), vd))
        signrows.append((w, fl, vn, vs))
    if omitted:
        print("  WIDTHS OMITTED FOR WANT OF A FLOOR: %s -- the table above is not complete over"
              % omitted)
        print("  the grid, and this line is the only record of that.")
    print()
    print("  THE PAIRED TEST, WHICH IS THE ONE THAT ANSWERS WHETHER THE ARM DIFFERS AT ALL.")
    print("  The band column beside it compares a POOLED MEDIAN against a PER-CELL dispersion:")
    print("  pct(fl, 0.05) is the spread over cells, pct(vn, 0.5) is a median over n of them, and")
    print("  a median's sampling spread narrows as 1/sqrt(n) -- so that boundary is about")
    print("  sqrt(n) ~ 20x too wide for the quantity it bounds. Its OUTSIDE verdicts are therefore")
    print("  conservative (safe, and understated), and its 'inside' is NOT evidence of absence:")
    print("  it says only that the pooled effect is smaller than a typical single-cell")
    print("  fluctuation, which nearly any real sub-percent effect would be.")
    print()
    print("  So the test below is PAIRED PER CELL -- d = (arm vs ship) - (floor vs ship) on the")
    print("  same cell -- which removes the object-change artefact cell by cell instead of")
    print("  subtracting two pooled locations, and carries a bootstrap CI on its own median.")
    print("  %-5s %6s %10s %10s %10s %-22s %s"
          % ("k", "n", "median d", "CI95 lo", "CI95 hi", "paired verdict",
             "v:data i:resample"))
    for (w, _fl, _vn, _vs) in signrows:
        pop = [c for c in cells if c[1] == w]
        d = paired_d(med, pop, "nopf", "shpb", "ship")
        if not d:
            print("  k=%-5d %6s   NO PAIRED CELL: needs nopf, shpb and ship on one cell" % (w, "-"))
            continue
        lo, hi, vdig, idig, n = boot_median_ci(d)
        m = pct(d, 0.5)
        if lo is None:
            vd, vdig, idig = "n < 8, no CI", "-", "-"
        elif hi < 0:
            vd = "FASTER than floor"
        elif lo > 0:
            vd = "SLOWER than floor"
        else:
            vd = "CI spans 0, no diff"
        print("  k=%-5d %6d %9.3f%% %9.3f%% %9.3f%% %-22s %s"
              % (w, n, 100 * m, 100 * lo if lo is not None else float("nan"),
                 100 * hi if hi is not None else float("nan"), vd,
                 "v:%s i:%s" % (vdig, idig)))
    print("  The CI is a percentile bootstrap on the median, 4000 resamples, seed 20261002. The")
    print("  last column digests the DATA (v) and the RESAMPLE (i) separately. Digesting only the")
    print("  indices, as an earlier version did, pins a pure function of (seed, nboot, n): two")
    print("  datasets with different effects gave the same string, so a substituted dataset")
    print("  reproduced it exactly. Reproducing a bound needs the values AND the indices.")
    print()
    print("  AND THE TWO COLUMNS ANSWER DIFFERENT QUESTIONS, which one verdict string had merged.")
    print("  The paired CI is a TEST OF EXISTENCE: does this arm differ from a mere object swap.")
    print("  The band is a DECISION RULE ABOUT ACTIONABILITY: an effect smaller than the per-cell")
    print("  artefact between two same-config objects is not worth shipping even if it is real.")
    print("  This project's standing rule -- nothing inside the noise floor counts -- is the")
    print("  second. Labelling it 'no verdict' was wrong: it is a verdict, about actionability,")
    print("  and it is silent about existence. Read the two columns together, never one alone.")
    print()
    print("  THE SIGN TEST, read against the FLOOR PAIR and not against 0.5. cells-better as a")
    print("  z against Binomial(n, 0.5) is only valid if the same-config pair sits near half; the")
    print("  floor column's own z is printed first for exactly that reason, and where it is large")
    print("  the arm's z beside it is measuring the object asymmetry, not the arm.")
    print("  %-5s %22s %22s %22s" % ("k", "FLOOR z (the null)", "nopf z", "strm z"))
    for (w, fl, vn, vs) in signrows:
        def _z(v):
            if not v:
                return None
            n = len(v)
            return ((sum(1 for x in v if x < 0) - n / 2.0) / math.sqrt(n / 4.0)) if n else None
        zf, zn, zs = _z(fl), _z(vn), _z(vs)
        def _f(z):
            return ("%+.1f sd" % z) if z is not None else "-"
        flag = ""
        if zf is not None and abs(zf) >= 2.0:
            flag = "  <- FLOOR ITSELF OFF HALF, arm z unreadable"
        print("  k=%-5d %22s %22s %22s%s" % (w, _f(zf), _f(zn), _f(zs), flag))
    print()
    print("  The floor column is the null for the two beside it. The recorded float32 result is")
    print("  that removing the prefetch wins 14.6% at k=4 and fades to 0.56% by k=16, and that it")
    print("  HARMS 115 matrices at k=32. That harm was measured on SUITESPARSE; whether this")
    print("  grid can see it depends on the grid, so the condition is tested rather than asserted:")
    if 32 in widths:
        pop32 = [c for c in cells if c[1] == 32]
        fl32 = rel("shpb", "ship", pop32)
        vn32 = rel("nopf", "ship", pop32) if "nopf" in builds else []
        if vn32 and fl32:
            fb = sum(1 for x in fl32 if x < 0)
            nb = sum(1 for x in vn32 if x < 0)
            print("    k=32 IS in this grid (%d cells): nopf reads %+.3f%% against a floor whose"
                  % (len(vn32), 100 * pct(vn32, 0.5)))
            print("    median is %+.3f%% and whose p05/p95 band is %+.3f%% to %+.3f%%. The MAGNITUDE"
                  % (100 * pct(fl32, 0.5), 100 * pct(fl32, 0.05), 100 * pct(fl32, 0.95)))
            print("    comparison is |%.3f| vs |%.3f|, so the pooled effect is INSIDE the floor and"
                  % (100 * pct(vn32, 0.5), 100 * pct(fl32, 0.5)))
            print("    carries NO sign from its magnitude. Cells-better: nopf %d/%d against the"
                  % (nb, len(vn32)))
            print("    floor pair's own %d/%d -- compare those two, never nopf against 0.5."
                  % (fb, len(fl32)))
            print("    The recorded harm is large enough to have been recorded over 115 matrices,")
            print("    so an effect inside this floor establishes that it DOES NOT REPRODUCE here;")
            print("    that is what a corpus-specificity claim needs, and it is all it may use.")
            print("    Either way this")
            print("    table DOES see the side of the trade that blocks shipping, so the sentence")
            print("    above must not be read as an excuse for not looking.")
        else:
            print("    k=32 is in the grid but lacks a nopf or shpb reading, so it is still unseen.")
    else:
        print("    k=32 is ABSENT from this grid, so this table cannot see the side of the trade")
        print("    that blocks shipping, and no conclusion about it may be drawn from these rows.")

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
