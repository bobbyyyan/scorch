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
