"""Where float64 SpMM stands against MKL, using FOUR OBJECTS as the control for placement.

WHY FOUR OBJECTS AND NOT ONE. The alignment floor measured on the 16-member padded family says
two objects built from IDENTICAL SOURCE, in the SAME alignment class, differ by a median of
1.3-5.2% per cell and a p95 of 5.7-50.7% depending on nonzero band. So a single object's reading
that a given matrix is a few percent behind MKL is not evidence that the kernel is behind on that
matrix -- it is within the spread of where the linker happened to put the loop.

This screen measured four objects (ship, shpb, strm, nopf) over four passes in a Latin square, so
every cell has 16 readings of ours and 16 of MKL. Requiring the SIGN to agree across all of them
is a control for placement that needs no floor constant: if a cell is behind MKL in every object
and every pass, the deficit is not an artefact of one object's layout or one pass's position.

THE THREE OUTCOMES ARE COUNTED SEPARATELY, NEVER DERIVED. Computing `mixed = n - ahead - behind`
makes the total an identity, so a misrouted cell still sums to n and the error is invisible. Each
cell is classified once and the three counts are asserted to sum to the cell total.

READ BY BAND AND WIDTH, NOT POOLED. A pooled ratio hides exactly the regions a "faster
everywhere" claim is about.
"""
import collections
import csv
import os
import sys

BANDS = ((1e7, ">=1e7"), (1e6, "1e6-1e7"), (1e5, "1e5-1e6"),
         (1e4, "1e4-1e5"), (1e3, "1e3-1e4"))
BAND_ORDER = ["<1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", "1e6-1e7", ">=1e7"]


def band(nnz):
    for lo, lab in BANDS:
        if nnz >= lo:
            return lab
    return "<1e3"


def geomean(vals):
    import math
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def pct(v, q):
    if not v:
        return None
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[i]


def main(argv):
    if len(argv) < 2:
        return "usage: an_f64_vs_mkl.py <csv-dir>"
    d = argv[1]
    names = sorted(f for f in os.listdir(d) if f.endswith(".csv"))
    if not names:
        return "REFUSE: no CSVs in %s" % d
    # (key, k) -> list of (build, pass, ours_ms, mkl_ms)
    cell = collections.defaultdict(list)
    nnz = {}
    rows_of = {}
    degree_of = {}
    skipped = collections.Counter()
    passes, builds = set(), set()
    for n in names:
        with open(os.path.join(d, n)) as fh:
            rows = list(csv.DictReader(fh))
        by = {}
        for r in rows:
            try:
                by[(r["key"], int(r["k"]), r["arm"])] = float(r["ms"])
                nnz[r["key"]] = int(r["nnz"])
                rows_of[r["key"]] = int(r["rows"])
                degree_of[r["key"]] = float(r["deg"])
            except (KeyError, TypeError, ValueError):
                skipped["unparseable"] += 1
        bset = {r["build"] for r in rows}
        if len(bset) != 1:
            return "REFUSE: %s mixes builds %s; a per-object control needs one build per file" % (n, bset)
        b = bset.pop()
        builds.add(b)
        tag = n.replace(".csv", "")
        passes.add(tag)
        for (key, k, a), t in by.items():
            if a != "ours":
                continue
            m = by.get((key, k, "mkl"))
            if m and t > 0 and m > 0:
                cell[(key, k)].append((b, tag, t, m))
            else:
                skipped["no mkl partner or non-positive"] += 1
    print("files %d   builds %s   pass-files %d" % (len(names), sorted(builds), len(passes)))
    print("cells %d   skipped %s" % (len(cell), dict(skipped) or "{}"))
    if not cell:
        return "REFUSE: no cells with both arms present"

    nread = collections.Counter(len(v) for v in cell.values())
    print("readings per cell: %s" % dict(nread))
    full = max(nread)
    print("cells with the full %d readings: %d of %d" % (full, nread[full], len(cell)))

    # ---- classify every cell once.
    AHEAD, BEHIND, MIXED = "ahead", "behind", "mixed"
    verdict, ratios = {}, {}
    for ck, obs in cell.items():
        rs = [t / m for (_b, _p, t, m) in obs]
        ratios[ck] = rs
        # ratio = ours/mkl, so < 1 means ours is FASTER.
        if all(r < 1.0 for r in rs):
            verdict[ck] = AHEAD
        elif all(r > 1.0 for r in rs):
            verdict[ck] = BEHIND
        else:
            verdict[ck] = MIXED
    counts = collections.Counter(verdict.values())
    tot = counts[AHEAD] + counts[BEHIND] + counts[MIXED]
    if tot != len(cell):
        return "REFUSE: the three verdicts sum to %d but there are %d cells" % (tot, len(cell))
    print()
    print("CELL VERDICTS over %d readings each (sign must agree in ALL readings)" % full)
    print("  ahead in every reading  %5d  (%.2f%%)" % (counts[AHEAD], 100.0 * counts[AHEAD] / tot))
    print("  behind in every reading %5d  (%.2f%%)" % (counts[BEHIND], 100.0 * counts[BEHIND] / tot))
    print("  mixed sign              %5d  (%.2f%%)" % (counts[MIXED], 100.0 * counts[MIXED] / tot))
    print("  counted independently and asserted to sum to %d" % tot)

    # ---- the deficit set, by band and width.
    print()
    print("BEHIND-IN-EVERY-READING, by band and width (the deficit set a 'faster everywhere'")
    print("claim has to close; mixed cells are inside the object/pass spread, not wins)")
    widths = sorted({k for (_key, k) in cell})
    hdr = "  %-9s" % "band" + "".join("%7s" % ("k=%d" % w) for w in widths) + "%8s" % "total"
    print(hdr)
    for lab in BAND_ORDER:
        row = "  %-9s" % lab
        tt = 0
        any_cell = False
        for w in widths:
            n = sum(1 for (key, k), v in verdict.items()
                    if k == w and band(nnz[key]) == lab and v == BEHIND)
            d_ = sum(1 for (key, k) in verdict if k == w and band(nnz[key]) == lab)
            if d_:
                any_cell = True
            row += "%7s" % (n if d_ else "-")
            tt += n
        print(row + "%8d" % tt if any_cell else row + "%8s" % "-")
    print("  (every width in the grid is a column above, so the row totals close on %d with no"
          % counts[BEHIND])
    print("   width omitted -- read the per-width totals rather than subtracting.)")
    print("  %-9s" % "TOTAL" + "".join(
        "%7d" % sum(1 for (key, k), v in verdict.items() if k == w and v == BEHIND)
        for w in widths) + "%8d" % counts[BEHIND])

    # ---- how far behind, for the cells that are.
    print()
    print("MARGIN on the behind set (max over readings of ours/mkl, so the worst object)")
    marg = collections.defaultdict(list)
    for ck, v in verdict.items():
        if v == BEHIND:
            marg[band(nnz[ck[0]])].append(max(ratios[ck]))
    print("  %-9s %6s %9s %9s %9s" % ("band", "cells", "median", "p95", "max"))
    for lab in BAND_ORDER:
        vv = sorted(marg.get(lab, []))
        if not vv:
            continue
        print("  %-9s %6d %8.3fx %8.3fx %8.3fx"
              % (lab, len(vv), pct(vv, 0.5), pct(vv, 0.95), vv[-1]))

    # ---- STRUCTURE of the deficit set against the rest, because a fix has to key on
    # something the matrices share. Printed as a comparison, never as the behind set alone:
    # a property is only a candidate if it SEPARATES behind from ahead.
    print()
    # THE COLUMN'S IDENTITY IS CHECKED BEFORE ANY NUMBER RESTS ON IT. Three quantities are in
    # the CSV -- nnz, rows, deg -- and one identity relates them. Every structural figure below
    # is a quantile of `deg`, so if that column were nnz/cols, or a mean over non-empty rows
    # only, or a count rather than a mean, the separation would print identically and read just
    # as convincing. An earlier version of this function built the two components into a dict
    # and never read it, which is the shape of having the check available and not making it.
    worst, worst_key = 0.0, None
    checked, unchecked = 0, 0
    for key in sorted(nnz):
        r = rows_of.get(key)
        if not r or key not in degree_of:
            unchecked += 1
            continue
        want = nnz[key] / float(r)
        got = degree_of[key]
        checked += 1
        # ABSOLUTE, not relative. The field is printed to two decimals, so its error is bounded
        # by half the last place (0.005) regardless of magnitude, while the RELATIVE error is
        # unbounded as the true value approaches zero. A relative criterion here refused on
        # Hamrle3_b, whose deg reads 0.00 against a true 0.000304 -- a correct column failing a
        # criterion that did not match its precision. The corpus really does contain such
        # matrices: LF10000_B has one nonzero in 19998 rows.
        d = abs(got - want)
        if d > worst:
            worst, worst_key = d, key
    print("DEGREE COLUMN IDENTITY: deg == nnz/rows over %d keys (%d unchecked)"
          % (checked, unchecked))
    if worst_key is None:
        sys.exit("REFUSE: the identity could not be evaluated on a single key, so every degree "
                 "quantile below would rest on an unchecked column")
    print("  worst ABSOLUTE discrepancy %.3e on %s (deg %.6f, nnz/rows %.6f)"
          % (worst, worst_key, degree_of[worst_key], nnz[worst_key] / float(rows_of[worst_key])))
    TOL = 0.005 + 1e-9        # half the last decimal place of a two-decimal field
    if worst > TOL:
        sys.exit("REFUSE: the deg column is not nnz/rows to within %.4f (worst %.3e on %s), so it "
                 "does not mean what the structure table below would claim it means"
                 % (TOL, worst, worst_key))
    print("  within %.4f, which is half the last place of a two-decimal field, so the column IS"
          % TOL)
    print("  nnz/rows. The table below nonetheless uses nnz/rows computed from its components,")
    print("  because rounding to two decimals makes the small-degree tail unusable: a true")
    print("  0.000304 prints as 0.00, and that tail is where the ahead set lives at k=2.")
    degree_of = {k: nnz[k] / float(rows_of[k]) for k in nnz if rows_of.get(k)}
    # The smallest degree in the corpus, printed because an extreme value in the column the
    # separation rests on is the strongest argument for showing its components.
    lo = min(degree_of, key=lambda k: degree_of[k])
    print("  smallest degree in the corpus: %s deg %.4f = %d nnz / %d rows"
          % (lo, degree_of[lo], nnz[lo], rows_of[lo]))
    print()
    print("STRUCTURE: the behind set against the ahead set, same widths")
    print("  %-6s %-7s %7s %9s %9s %9s %9s" % ("k", "set", "cells", "med deg", "p10 deg", "p90 deg", "med nnz"))
    for w in widths:
        for lab, want in (("behind", BEHIND), ("ahead", AHEAD), ("mixed", MIXED)):
            sel = [ck for ck, v in verdict.items() if ck[1] == w and v == want]
            if not sel:
                continue
            ds = sorted(degree_of[ck[0]] for ck in sel if ck[0] in degree_of)
            ns = sorted(nnz[ck[0]] for ck in sel)
            if not ds:
                continue
            print("  k=%-4d %-7s %7d %9.2f %9.2f %9.2f %9d"
                  % (w, lab, len(sel), pct(ds, 0.5), pct(ds, 0.10), pct(ds, 0.90), pct(ns, 0.5)))
        print()

    print("THE 20 WORST CELLS (largest max-over-readings ours/mkl), with structure")
    worst = sorted(((max(ratios[ck]), ck) for ck, v in verdict.items() if v == BEHIND),
                   reverse=True)[:20]
    print("  %-26s %4s %10s %9s %8s %8s" % ("key", "k", "nnz", "rows", "deg", "worst"))
    for r, ck in worst:
        key, w = ck
        print("  %-26s %4d %10d %9s %8.2f %7.3fx"
              % (key[-26:], w, nnz[key], rows_of.get(key, "?"), degree_of.get(key, 0), r))

    # ---- pooled position, reported last and labelled as pooled.
    print()
    print("POOLED ours/mkl geomean over every reading, by band (a summary, not a per-cell claim)")
    pool = collections.defaultdict(list)
    for ck, rs in ratios.items():
        pool[band(nnz[ck[0]])] += rs
    for lab in BAND_ORDER:
        vv = pool.get(lab)
        if not vv:
            continue
        g = geomean(vv)
        print("  %-9s readings %6d  geomean ours/mkl %.4f  (= %.3fx %s)"
              % (lab, len(vv), g, 1.0 / g, "ours faster" if g < 1 else "MKL faster"))
    allr = [r for v in ratios.values() for r in v]
    g = geomean(allr)
    print("  %-9s readings %6d  geomean ours/mkl %.4f  (= %.3fx ours faster)"
          % ("ALL", len(allr), g, 1.0 / g))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
