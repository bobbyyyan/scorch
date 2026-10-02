"""The alignment floor's WITHIN-class agreement, which decides headline versus bound.

WHAT THE FAMILY IS. 16 objects built from identical source by re-running one arm's own link with
a .text pad of 0,16,...,240 bytes. Identical source means any difference between them is
placement, not code. The pad fixes the hot loop's address mod 64, so the members fall into four
alignment classes: class(m) = 16(1+m) mod 64 for member index m = pad/16.

WHY WITHIN-CLASS IS THE NUMBER. An arm sits in exactly ONE alignment class, fixed by the
compiler, not drawn from a distribution. So the right floor for an arm-vs-arm contrast where both
arms share a class is the spread between two SAME-class objects. If the arms sit in different
classes the contrast additionally carries the between-class term, and then the honest report is a
bound rather than a matched headline.

THE STATISTIC IS n-INVARIANT ON PURPOSE. A range inflates 1.93x from n=3 to n=12 on sample size
alone, so a range would make the floor an artefact of how many members a class has. The c4
correction makes the sample standard deviation unbiased at every n (measured 0.999 at n = 3, 6,
12, 24). Reported as a coefficient of variation so it is comparable across cells of wildly
different absolute time.

BAND IS A REPORTED AXIS, NEVER POOLED AWAY. The A/A floor is an inverted U in matrix size -- p95
near 38% at 1e4-1e6 nonzeros against about 6% at both ends -- so a floor pooled over bands bounds
neither end.

WHAT THIS PROGRAM CANNOT SEPARATE. In a cyclic step-M/S schedule the between-class term is
confounded with wall-clock drift, because the class means sit at distinct mean positions (range 3
of 15 slots for M=16, S=4) and the leader of every slice belongs to one class. A between-class
number from such a run is therefore an upper bound on the true class effect, not an estimate of
it, and this program says so beside the number instead of pooling it silently. The schedule's own
balance is re-certified here from the data, on the window actually analysed.
"""
import collections
import csv
import math
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


def c4(n):
    """Bias correction for the sample standard deviation at sample size n."""
    if n < 2:
        return float("nan")
    return math.sqrt(2.0 / (n - 1)) * math.gamma(n / 2.0) / math.gamma((n - 1) / 2.0)


def corrected_cv(vals):
    """Unbiased sd over mean. None when the sample is too small or the mean is not positive."""
    n = len(vals)
    if n < 2:
        return None
    mu = sum(vals) / n
    if mu <= 0:
        return None
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return (math.sqrt(var) / c4(n)) / mu


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def alignment_class(m):
    return (16 * (1 + m)) % 64


def load(d, manifest=None):
    """Rows keyed by (key, k, member, arm) -> ms. Reports what it skipped and from where."""
    if manifest:
        paths = []
        for ln in open(manifest):
            if ln.startswith("#") or not ln.strip():
                continue
            paths.append(ln.split()[-1])
        print("input: %d CSV(s) named by manifest %s" % (len(paths), manifest))
    else:
        paths = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.startswith("floor_") and f.endswith(".csv")]
        glob_n = len(paths)
        print("input: %d CSV(s) by glob over %s" % (glob_n, d))
        print("  ^ no manifest given, so this is the whole directory. If it holds more than one")
        print("    run's files, pass the run's manifest instead; nothing partitions a glob.")
    if not paths:
        sys.exit("REFUSE: no input CSVs, so every number below would be computed over nothing")

    ms = {}
    nnz = {}
    slice_of = {}
    skipped = collections.Counter()
    for p in paths:
        if not os.path.exists(p):
            skipped["missing file"] += 1
            continue
        with open(p) as fh:
            for row in csv.DictReader(fh):
                try:
                    tag = row["build"]
                    m = int(tag.lstrip("p")) // 16
                    k = int(row["k"])
                    t = float(row["ms"])
                    nz = int(row["nnz"])
                except (KeyError, TypeError, ValueError):
                    skipped["unparseable row"] += 1
                    continue
                if t <= 0:
                    skipped["non-positive ms"] += 1
                    continue
                ms[(row["key"], k, m, row["arm"])] = t
                nnz[row["key"]] = nz
                slice_of[(row["key"], m)] = row["slice"]
    print("  rows loaded %d, skipped %s" % (len(ms), dict(skipped) or "{}"))
    if not ms:
        sys.exit("REFUSE: no usable rows")
    return ms, nnz, slice_of


def main(argv):
    if len(argv) < 2:
        return "usage: an_floor_within.py <csv-dir> [manifest] [--arm ours]"
    d = argv[1]
    manifest = argv[2] if len(argv) > 2 and not argv[2].startswith("--") else None
    arm = argv[argv.index("--arm") + 1] if "--arm" in argv else "ours"
    ms, nnz, slice_of = load(d, manifest)

    members = sorted({m for (_k, _w, m, _a) in ms})
    arms = sorted({a for (_k, _w, _m, a) in ms})
    widths = sorted({w for (_k, w, _m, _a) in ms})
    print("members %d %s" % (len(members), members))
    print("arms %s   widths %s" % (arms, widths))
    if arm not in arms:
        return "REFUSE: arm %r not present; arms are %s" % (arm, arms)
    cls = collections.defaultdict(list)
    for m in members:
        cls[alignment_class(m)].append(m)
    print("classes -> members %s" % dict(cls))

    # ---- re-certify the schedule's balance on the analysed window, from the data itself.
    print()
    print("SCHEDULE BALANCE RE-CERTIFIED ON THIS WINDOW")
    seen = collections.defaultdict(set)
    for (key, m), sl in slice_of.items():
        seen[m].add(sl)
    nsl = {m: len(v) for m, v in seen.items()}
    print("  slices each member appears in: %s"
          % ("all %d for every member" % len(set().union(*seen.values()))
             if len(set(nsl.values())) == 1 else nsl))

    # ---- the number.
    print()
    print("WITHIN-CLASS AGREEMENT, coefficient of variation over same-class members")
    print("(c4-corrected sd / mean, per (key, width, class); n = members in the class)")
    rows_by_band = collections.defaultdict(list)
    per_class = collections.defaultdict(list)
    within_med = {}
    n_cells = 0
    short = collections.Counter()
    for key in sorted(nnz):
        for w in widths:
            for c, mem in cls.items():
                vals = [ms[(key, w, m, arm)] for m in mem if (key, w, m, arm) in ms]
                if len(vals) < len(mem):
                    short[len(vals)] += 1
                cv = corrected_cv(vals)
                if cv is None:
                    continue
                n_cells += 1
                rows_by_band[band(nnz[key])].append(cv)
                per_class[c].append(cv)
    print("  cells contributing %d; classes short of full membership: %s"
          % (n_cells, dict(short) or "none"))
    print()
    print("  %-9s %6s %9s %9s %9s" % ("band", "cells", "median", "p95", "max"))
    for lab in BAND_ORDER:
        v = sorted(rows_by_band.get(lab, []))
        if not v:
            print("  %-9s %6s %9s %9s %9s" % (lab, 0, "-", "-", "-"))
            continue
        print("  %-9s %6d %8.3f%% %8.3f%% %8.3f%%"
              % (lab, len(v), 100 * pct(v, 0.5), 100 * pct(v, 0.95), 100 * v[-1]))
        within_med[lab] = pct(v, 0.5)
    allv = sorted(x for v in rows_by_band.values() for x in v)
    print("  %-9s %6d %8.3f%% %8.3f%% %8.3f%%   <- pooled, which bounds NEITHER end"
          % ("POOLED", len(allv), 100 * pct(allv, 0.5), 100 * pct(allv, 0.95), 100 * allv[-1]))

    print()
    print("  per class (pooled over bands, so read the band table above for the real shape)")
    for c in sorted(per_class):
        v = sorted(per_class[c])
        print("    class %-3d cells %5d  median %6.3f%%  p95 %6.3f%%" % (c, len(v), 100 * pct(v, 0.5), 100 * pct(v, 0.95)))

    # ---- THE MATCHED FUNCTIONAL. The floor's magnitude depends mostly on WHICH cell statistic
    # is formed, not on the data: on this branch the same full-collection comparison gave pooled
    # p95 of 0.0569, 0.0851, 0.1970 and 0.4023 for four different cell statistics -- a 7.1x
    # spread. So a floor has to be the same functional as the contrast it bounds, structurally
    # rather than by argument. The coefficient of variation above is a dispersion over FOUR
    # objects; an arm contrast is a ratio between TWO objects. Those are different functionals,
    # so the CV is NOT the matched floor for an arm comparison -- this is.
    print()
    print("MATCHED FUNCTIONAL: the ratio between TWO same-class objects, which is the form an")
    print("arm contrast takes. Every unordered same-class pair contributes one |ratio - 1|.")
    pair_by_band = collections.defaultdict(list)
    npairs = 0
    for key in sorted(nnz):
        for w in widths:
            for c, mem in cls.items():
                have = [m for m in mem if (key, w, m, arm) in ms]
                for i in range(len(have)):
                    for j in range(i + 1, len(have)):
                        a_, b_ = ms[(key, w, have[i], arm)], ms[(key, w, have[j], arm)]
                        if a_ > 0 and b_ > 0:
                            pair_by_band[band(nnz[key])].append(abs(a_ / b_ - 1.0))
                            npairs += 1
    print("  pairs contributing %d (%d same-class pairs per (key,width,class) at n=4)" % (npairs, 6))
    print("  %-9s %6s %9s %9s %9s" % ("band", "pairs", "median", "p95", "max"))
    for lab in BAND_ORDER:
        v = sorted(pair_by_band.get(lab, []))
        if not v:
            continue
        print("  %-9s %6d %8.3f%% %8.3f%% %8.3f%%"
              % (lab, len(v), 100 * pct(v, 0.5), 100 * pct(v, 0.95), 100 * v[-1]))
    allp = sorted(x for v in pair_by_band.values() for x in v)
    print("  %-9s %6d %8.3f%% %8.3f%% %8.3f%%   <- pooled" % ("POOLED", len(allp),
          100 * pct(allp, 0.5), 100 * pct(allp, 0.95), 100 * allp[-1]))
    print("  This is the number to compare an arm's two-object contrast against. The CV table")
    print("  above answers a different question -- how far four objects of one class spread --")
    print("  and the two are not interchangeable.")

    # ---- the between-class term, labelled as the bound it is.
    print()
    print("BETWEEN-CLASS TERM (an upper bound, not an estimate: in a cyclic schedule the class")
    print("means sit at different mean wall-clock positions, so drift and class are not separable)")
    btw_by_band = collections.defaultdict(list)
    for key in sorted(nnz):
        for w in widths:
            cmeans = []
            for c, mem in cls.items():
                vals = [ms[(key, w, m, arm)] for m in mem if (key, w, m, arm) in ms]
                if vals:
                    cmeans.append(sum(vals) / len(vals))
            if len(cmeans) == len(cls) and min(cmeans) > 0:
                btw_by_band[band(nnz[key])].append((max(cmeans) - min(cmeans)) / (sum(cmeans) / len(cmeans)))
    print("  %-9s %6s %9s %9s" % ("band", "cells", "median", "p95"))
    for lab in BAND_ORDER:
        v = sorted(btw_by_band.get(lab, []))
        if not v:
            print("  %-9s %6s %9s %9s" % (lab, 0, "-", "-"))
            continue
        print("  %-9s %6d %8.3f%% %8.3f%%" % (lab, len(v), 100 * pct(v, 0.5), 100 * pct(v, 0.95)))

    # ---- the process-level A/A, as the tighter reference it is.
    if "oursb" in arms:
        print()
        print("FOR CONTRAST, the SAME-PROCESS repeat (ours vs oursb, one object): this is a")
        print("process-level A/A and does NOT bound inter-object variation, so it is the wrong")
        print("floor for an arm whose macro changes emitted code. Reported to show the gap.")
        rep = collections.defaultdict(list)
        rep_med = {}
        for (key, w, m, a), t in ms.items():
            if a != "ours":
                continue
            u = ms.get((key, w, m, "oursb"))
            if u and t > 0:
                rep[band(nnz[key])].append(abs(u - t) / ((u + t) / 2.0))
        print("  %-9s %6s %9s %9s" % ("band", "cells", "median", "p95"))
        for lab in BAND_ORDER:
            v = sorted(rep.get(lab, []))
            if not v:
                continue
            print("  %-9s %6d %8.3f%% %8.3f%%" % (lab, len(v), 100 * pct(v, 0.5), 100 * pct(v, 0.95)))
            rep_med[lab] = pct(v, 0.5)
    else:
        rep_med = {}
    print()
    print("THE GAP, PER BAND, because the ratio between the two IS the finding and two separate")
    print("ranges cannot produce it (a published '3-10x' was not derivable from them):")
    print("  %-9s %12s %12s %9s" % ("band", "within-class", "same-process", "ratio"))
    ratios = []
    for lab in BAND_ORDER:
        w_ = within_med.get(lab)
        r_ = rep_med.get(lab)
        if w_ is None or r_ is None or r_ <= 0:
            continue
        ratios.append(w_ / r_)
        print("  %-9s %11.3f%% %11.3f%% %8.2fx" % (lab, 100 * w_, 100 * r_, w_ / r_))
    # A band count beside the range, because a single-band table printed "3.51x to 3.51x" and
    # read as a result: the assignment that fills this dict had been placed one indent level
    # outside its loop, so it ran once with the loop variable left at its last value.
    print("  bands contributing %d of %d" % (len(ratios), len(BAND_ORDER)))
    if len(ratios) < 2:
        print("  REFUSING to state a range over %d band(s); the per-band pairing is the finding"
              % len(ratios))
    elif ratios:
        print("  so the inter-object floor is %.2fx to %.2fx the same-process floor by band"
              % (min(ratios), max(ratios)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
