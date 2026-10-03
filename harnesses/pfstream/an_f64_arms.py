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


def arm_paired_d(med, pop):
    """THE SCREEN'S CONTRAST, and the only place its three tags appear.

    The tags live here because a test that re-derives them cannot cover them. The previous
    self-test passed its own literals and asserted that swapping the two tag arguments flips the
    sign -- but `paired_d` returns `(a - f) / b`, so that antisymmetry is an ALGEBRAIC IDENTITY
    holding for every dataset and every assignment. It proved a swap would be VISIBLE; it could
    not prove production performs none, because the call site's literals were never consumed by
    the test. Planting the swap at the call site left the suite reporting 14 checks PASSED while
    every verdict's sign was reversed.

    So the contrast is named once and both production and the self-test call THIS. The test
    asserts a SIGNED known value on it, which a swapped literal here flips and fails.
    """
    return paired_d(med, pop, "nopf", "shpb", "ship")


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
            # THE EDGE REPORTED MUST BE THE ONE THE EFFECT IS HEADING TOWARD. The first version
            # printed `hi` whenever the verdict was "inside", so a NEGATIVE arm median was shown
            # against the POSITIVE edge -- "-1.507 vs +3.033" when the binding comparison is
            # -1.507 against -3.299. A reader would take the wrong number as the bar.
            if mn < lo:
                vd, edge = "FASTER, outside", lo
            elif mn > hi:
                vd, edge = "SLOWER, outside", hi
            else:
                vd, edge = "inside, no verdict", (lo if mn < 0 else hi)
            vd += " (%+.3f vs %+.3f, margin %.3f pp)" % (100 * mn, 100 * edge,
                                                         abs(100 * (mn - edge)))
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
    print("  same cell, with a bootstrap CI on its own median. WHAT IT REMOVES IS `ship`, NOT THE")
    print("  ARTEFACT, and the earlier claim that it 'removes the object-change artefact cell by")
    print("  cell' is WITHDRAWN. Writing each object's offset relative to a Platonic time as o_X:")
    print("      (arm - ship)/ship   ~ o_arm  - o_ship")
    print("      (floor - ship)/ship ~ o_shpb - o_ship")
    print("      d                   ~ o_arm  - o_shpb")
    print("  Three DISTINCT objects, so subtracting shpb's offset cannot cancel the arm's: d trades")
    print("  one object artefact for a DIFFERENCE of two, which has larger spread, not zero.")
    print("  MEASURED, on this screen's own registered negative control: at float64 k=2 the PFD")
    print("  macro provably cannot change executed code (narrow_exact owns double k=1..3), so the")
    print("  arm is inert and d there is PURE o_nopf - o_shpb. It reads +0.429% with a CI of")
    print("  [+0.274, +0.611] -- EXCLUDING ZERO. So the residual this test leaves behind is about")
    print("  0.43% on this grid, and any width whose |d| is at or below that is not readable.")
    print("  %-5s %6s %10s %10s %10s %-22s %s"
          % ("k", "n", "median d", "CI95 lo", "CI95 hi", "paired verdict",
             "v:data i:resample"))
    for (w, _fl, _vn, _vs) in signrows:
        pop = [c for c in cells if c[1] == w]
        d = arm_paired_d(med, pop)
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
            # THIS VERDICT USED TO BE A LITERAL. The line below once read "...so the pooled
            # effect is INSIDE the floor", with "INSIDE the floor" hardcoded in the format string
            # and NO conditional between it and the block head -- so it printed inside whatever
            # the numbers were. It also stated the |arm median| vs |floor median| comparison that
            # amendment 30 withdrew, with the correct operands printed on the line directly
            # above. It survived because at k=32 on DLMC the wrong comparison and the right one
            # AGREED: magnitude said inside with margin 0.408 pp, the band said inside with
            # margin 0.009 pp. Same verdict, margins 45x apart. TWO METHODS AGREEING ON A VERDICT
            # IS NOT EVIDENCE THEY AGREE -- the margin is where the disagreement shows.
            #
            # The dangling "Either way this" four lines down was the fossil of the branch that
            # used to be here: the if/else was collapsed, the `inside` text kept, and the
            # connective kept with it. A stranded "either way" / "in both cases" / "whichever
            # holds" is a grep-able signature of a conditional that no longer exists.
            m32, lo32, hi32 = pct(vn32, 0.5), pct(fl32, 0.05), pct(fl32, 0.95)
            if m32 < lo32:
                v32, e32 = "OUTSIDE the band, FASTER", lo32
            elif m32 > hi32:
                v32, e32 = "OUTSIDE the band, SLOWER", hi32
            else:
                v32, e32 = "INSIDE the band", (lo32 if m32 < 0 else hi32)
            print("    median is %+.3f%% and its p05/p95 band is %+.3f%% to %+.3f%%."
                  % (100 * pct(fl32, 0.5), 100 * lo32, 100 * hi32))
            print("    BAND verdict: %+.3f%% vs edge %+.3f%% (margin %.3f pp) -> %s."
                  % (100 * m32, 100 * e32, abs(100 * (m32 - e32)), v32))
            print("    Cells-better: nopf %d/%d against the floor pair's own %d/%d -- compare"
                  % (nb, len(vn32), fb, len(fl32)))
            print("    those two, never nopf against 0.5.")
            if v32 == "INSIDE the band":
                print("    The recorded harm is large enough to have been recorded over 115")
                print("    matrices, so an effect inside the band establishes that it DOES NOT")
                print("    REPRODUCE here -- which is what a corpus-specificity claim needs, and")
                print("    all it may use. It does NOT license a claim that this arm wins at k=32.")
            elif v32 == "OUTSIDE the band, SLOWER":
                print("    The arm is SLOWER outside the band, which is the same SIGN as the")
                print("    recorded SuiteSparse harm. That CORROBORATES the harm on a second")
                print("    corpus and is the one outcome that argues against shipping outright.")
            else:
                print("    The arm is FASTER outside the band, the OPPOSITE sign to the recorded")
                print("    harm. That makes the harm corpus-specific rather than width-specific,")
                print("    and it still does not clear it to ship: the SuiteSparse harm is real")
                print("    on SuiteSparse, and 115 regressed matrices is a regression.")
            print("    Whichever of those three printed, this table DOES see the side of the")
            print("    trade that blocks shipping, so the sentence above must not be read as an")
            print("    excuse for not looking.")
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
    # THIS BLOCK STILL USED THE WITHDRAWN COMPARISON. It read `abs(m) <= abs(fl_med)` -- an arm's
    # LOCATION against the magnitude of the floor's LOCATION -- which amendment 30 withdrew after
    # it called an arm of -0.973% "outside" a floor of +1.381%. Amendment 31 fixed the per-width
    # table and left this one, so the withdrawn method survived in a second code path for two
    # amendments. The floor's median is where the object-swap artefact SITS; its p05/p95 is how
    # wide it is, and only the width can say whether an effect is resolvable.
    fl_med, fl_p05, fl_p95 = pct(fl, 0.5), pct(fl, 0.05), pct(fl, 0.95)
    print("  floor median %+.3f%%, band p05 %+.3f%% to p95 %+.3f%% (width %.3f pp)"
          % (100 * fl_med, 100 * fl_p05, 100 * fl_p95, 100 * (fl_p95 - fl_p05)))
    print("  The BAND decides actionability, not the median: a median is a location and cannot")
    print("  bound an effect. Read the paired CI below for whether the effect EXISTS.")
    for a in arms:
        v = rel(a, "ship", pop)
        if not v:
            continue
        m = pct(v, 0.5)
        beats = sum(1 for x in v if x < fl_p05)
        if m < fl_p05:
            vd, edge = "FASTER, outside the band", fl_p05
        elif m > fl_p95:
            vd, edge = "SLOWER, outside the band", fl_p95
        else:
            vd, edge = "inside the band, not actionable", (fl_p05 if m < 0 else fl_p95)
        print("  %-5s median %+.3f%% vs band edge %+.3f%% (margin %.3f pp)  ->  %s"
              % (a, 100 * m, 100 * edge, abs(100 * (m - edge)), vd))
        print("        cells beyond the floor's p05 tail: %d of %d (%.1f%%)"
              % (beats, len(v), 100.0 * beats / len(v)))
        d = paired_d(med, pop, a, "shpb", "ship") if a == "nopf" else []
        if d:
            lo_, hi_, vdg, idg, nn_ = boot_median_ci(d)
            if lo_ is not None:
                print("        PAIRED vs floor, n=%d: median %+.3f%% CI [%+.3f%%, %+.3f%%] -> %s"
                      "  v:%s i:%s"
                      % (nn_, 100 * pct(d, 0.5), 100 * lo_, 100 * hi_,
                         "EXISTS, faster" if hi_ < 0 else
                         ("EXISTS, slower" if lo_ > 0 else "CI spans 0"), vdg, idg))
    print()
    print("  An effect inside the band is not evidence of nothing -- it is evidence that it is")
    print("  smaller than the artefact between two same-config objects, so it is not worth")
    print("  shipping even where the paired CI says it is real. Those are different statements")
    print("  and both lines above are printed so neither can stand in for the other.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
