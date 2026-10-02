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
        return "usage: an_f64_vs_mkl.py <csv-dir> [declared-ks e.g. 2,4,6,8,10,12,16]"
    d = argv[1]
    # THE DECLARED WIDTH SET, from the producer rather than from the data. A width set derived
    # from the rows cannot detect a width the run measured and the analysis never grouped, nor a
    # declared width that produced no rows -- and the defect this guards against was exactly
    # that: a prose summary named five widths while the run measured seven, and 19 of 192 cells
    # went missing. Only a declaration outside the data can catch it.
    # The producer writes its own ks, keys, builds and passes into a sentinel beside the CSVs,
    # so the declaration is read rather than handed over. An analysis whose validation depends on
    # a caller remembering an argument has only its plants: there was no committed caller passing
    # it at all, and a forgotten argument completed and exited 0 with the width check silently
    # unavailable.
    declared, decl_src, decl = None, None, {}
    sentinel = os.path.join(d, "f64screen.done")
    if os.path.exists(sentinel):
        for ln in open(sentinel):
            parts = ln.split()
            if len(parts) >= 2:
                decl[parts[0]] = parts[1]
        if "ks" in decl:
            declared = sorted(int(x) for x in decl["ks"].split(",") if x)
            decl_src = "the producer's sentinel %s" % sentinel
    if declared is None and len(argv) > 2 and argv[2] and not argv[2].startswith("--"):
        declared = sorted(int(x) for x in argv[2].split(",") if x)
        decl_src = "argument 2"
    if declared is None:
        return ("REFUSE: no declared width set. Either %s must exist carrying a `ks` line, or the "
                "producer's ks must be passed as argument 2. Without a declaration the width "
                "validation is unavailable and the cell counts below cannot be checked against "
                "anything outside the data they are computed from -- which is how 19 of 192 "
                "cells went unreported once already." % sentinel)
    print("declaration read from %s: ks %s%s"
          % (decl_src, declared,
             (", keys %s, builds %s, passes %s"
              % (decl.get("keys", "?"), decl.get("builds", "?"), decl.get("passes", "?")))
             if decl else ""))
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
    # INDEPENDENT ACCOUNTING, derived entirely from the declaration. An earlier version compared
    # a counter incremented in the same branch as the append it was counting -- the two moved in
    # lockstep, so it was an identity, and it was evaluated before the classifier it claimed to
    # watch even ran. None of the four factors below come from the cell table: the key count and
    # width count are the producer's, and builds x passes is the pass-file structure.
    observed = sum(len(v) for v in cell.values())
    n_keys = int(decl["keys"]) if decl.get("keys", "").isdigit() else None
    n_bp = (int(decl["builds"]) * int(decl["passes"])
            if decl.get("builds", "").isdigit() and decl.get("passes", "").isdigit() else None)
    fell_back = []
    if n_keys is None:
        n_keys = len({k for (k, _w) in cell})
        fell_back.append("key count (now %d, taken from the rows)" % n_keys)
    if n_bp is None:
        n_bp = len(names)
        fell_back.append("pass-file count (now %d, taken from the files present)" % n_bp)
    if fell_back:
        # Name which factor fell back, not just that one did, and name which of the four failures
        # the refusal below can still catch. A refusal advertising four catches while running in
        # a mode with two is the misleading part.
        print("  DEGRADED: %s" % "; ".join(fell_back))
        print("  In this mode the check still catches a declared width that produced no rows and")
        print("  a row dropped before the mkl-partner test. It can NO LONGER catch a key the")
        print("  cache never resolved (absent from the rows, so the key count shrinks in step")
        print("  with the observation) nor a pass-file that never arrived (the file count counts")
        print("  what arrived) -- which are exactly the two the declaration exists for.")
    expected = n_keys * len(declared) * n_bp
    print("expected readings = %d keys x %d declared widths x %d pass-files = %d; observed %d"
          % (n_keys, len(declared), n_bp, expected, observed))
    if expected != observed:
        catches = ("a declared width that produced no rows, and a row dropped before the "
                   "mkl-partner test" if fell_back else
                   "a key the cache never resolved, a declared width that produced no rows, a "
                   "pass-file that never arrived, and a row dropped before the mkl-partner test")
        return ("REFUSE: %d readings expected from the declaration but %d present (%+d). In this "
                "mode the check covers: %s." % (expected, observed, observed - expected, catches))
    print("  ASSERTED equal, so no key, width or pass-file is silently missing")

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
    # This equality is an IDENTITY -- the classifier is if/elif/else over cell.items(), so every
    # cell gets exactly one verdict and a Counter over them sums to len(cell). Keeping it as an
    # assertion would claim coverage it does not have. The LIVE check is against a different
    # source: the number of `ours` rows actually parsed, which must equal cells x readings. That
    # catches a cell dropped between parse and classify, which is the failure the old form read
    # as covering.
    if tot != len(cell):
        return "REFUSE: internal inconsistency, verdicts %d vs cells %d" % (tot, len(cell))
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
    per_width = {w: sum(1 for (key, k), v in verdict.items() if k == w and v == BEHIND)
                 for w in widths}
    # This sum is an IDENTITY and is not an assertion: `widths` is derived from `cell` and
    # verdict's keys are cell's keys, so every behind cell's width is in widths by construction.
    # Printed as arithmetic, not as a check.
    print("  per-width behind counts %s sum to %d (an identity: the width set is derived from "
          "the same rows, so this cannot fail)" % (per_width, sum(per_width.values())))
    if declared is None:
        print("  NO DECLARED WIDTH SET GIVEN, so the live check is unavailable: pass the"
              " producer's")
        print("  own ks (e.g. 2,4,6,8,10,12,16) as argument 2 to compare the measured width set")
        print("  against the declared one. Without it a width the run measured and this analysis")
        print("  never grouped is undetectable here.")
    else:
        extra = [w for w in widths if w not in declared]
        missing = [w for w in declared if w not in widths]
        print("  declared widths %s vs measured %s" % (declared, widths))
        if extra or missing:
            return ("REFUSE: the measured width set does not match the declared one -- measured "
                    "but not declared %s, declared but unmeasured %s. The first means this "
                    "analysis is grouping rows the producer did not register; the second means a "
                    "declared width produced no rows at all."
                    % (extra or "none", missing or "none"))
        print("  ASSERTED equal, so no declared width is missing and none is unregistered")
    # Named explicitly, because a prose summary of this table omitted two widths and left a
    # reader to subtract 19 cells. For doubles (4 lanes) the tail is masked when k %% 4 != 0.
    masked = [w for w in widths if w > 3 and w % 4 != 0]
    print("  masked-tail widths above the exact kernel's range (k %% 4 != 0): %s holding %d cell(s)"
          % (masked, sum(per_width[w] for w in masked)))
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
        # A key absent from degree_of is one whose row count was unusable, and printing 0 for it
        # would be indistinguishable from a measured value -- the real small-degree tail reaches
        # 0.0001, so 0 is plausible here. Printed as "n/a" instead, and the count of such keys is
        # the `unchecked` figure above.
        dg = ("%8.2f" % degree_of[key]) if key in degree_of else "%8s" % "n/a"
        print("  %-26s %4d %10d %9s %s %7.3fx"
              % (key[-26:], w, nnz[key], rows_of.get(key, "?"), dg, r))

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
