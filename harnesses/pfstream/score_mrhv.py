#!/usr/bin/env python3
"""Score the multirow x half-vector screen against PREREG_multirow_halfvec.md.

Written before any of its data existed, which is the only form of pre-registration that
cannot be talked out of. Every reduction, floor, index and population below is the one
the prereg registers as amended through amendment 7.

The design. Four objects, all compiled at one fixed absolute path so the build path
cannot differ between them:

    mr0, mr0b   byte-identical controls -- their contrast is the A/A floor
    mrhv        multirow x half-vector, SCORCH_SPMM_MULTIROW_ROWS_F32=2
    mrhv4       the same with ROWS_F32=4, the firing check

Four passes rotate the object order so no object is always measured late. One pass file
holds exactly one reading per (key, width, arm, object), so the pass index is the only
valid pairing between objects -- the `batch` column is an inner repetition count that
differs by arm (hundreds for ours, tens for mkl) and pairing on it is meaningless.

Reductions. Every quantity scored here is a ratio, so every reduction is the geometric
middle exp(median(log(.))) -- amendment 6(c). Floors are the p95 of the |log| A/A
displacement, reported with the order-statistic index and the convention beside the
value, because `pct`-style indices have no single convention at even n -- amendment 5.

Usage: score_mrhv.py <csv-dir> <chain207-ledger-dir> [dtype] [prefix]
"""
import csv
import glob
import math
import os
import statistics
import random
import sys
import collections

FLOOR_BUILDS = ("mr0", "mr0b")
ARM, FIRE = "mrhv", "mrhv4"
BUILDS = (FLOOR_BUILDS[0], FLOOR_BUILDS[1], ARM, FIRE)


def geo(v):
    """The geometric middle. exp of the median of the logs -- an_c207.py:57."""
    return math.exp(statistics.median([math.log(x) for x in v]))


def fmt_sep(lo, hi):
    """Format a bootstrap interval with its WIDTH, always.

    `[+2.00, +2.00]` reads as precision. `width 0.00` reads as broken. They are the same
    interval. A degenerate bootstrap -- one where the resampling has nothing to resample
    over, as happens when a verification plant is a uniform rescale of the very quantity
    it is compared against -- collapses every replicate to one value, and that is a notice
    rather than a result. It was read as reassurance twice in this project before anyone
    noticed, because zero is legible as a number and invisible as a pair of endpoints that
    happen to match. So the width is printed by every call site, not by some of them.
    """
    return "[%+6.2f, %+6.2f] pp of S-1  width %5.2f%s" % (
        100 * lo, 100 * hi, 100 * (hi - lo),
        "  <- DEGENERATE, the resampling had nothing to vary" if hi - lo < 1e-12 else "")


def boot_sep(arm, flo, seed, B=2000, stat="tail"):
    """CI for functional(arm) - functional(floor) by resampling CELLS, paired in both.

    `stat` picks the functional, and the choice is not cosmetic -- it has to match the
    SIDEDNESS of the effect:

      "tail"    p95 of the per-cell |log| displacement. Two-sided. Correct for a
                difference that may go either way: condition 1 (a firing difference has
                no preferred sign) and condition 2 (an equivalence test).

      "center"  the geometric middle of the per-cell ratio, compared on the log scale.
                One-sided and monotone in a uniform speedup. This is what amendment 2
                registers for condition 3 secondary.

    Using "tail" for a one-sided effect is a real defect and not a conservative choice.
    A uniform speedup divides every cell's ratio by the effect, which pulls the cells
    sitting above 1 DOWN through 1 and out of the two-sided tail before the far side
    regrows, so p95(|log|) is NON-MONOTONE in the effect with its minimum near the
    effect sizes this screen exists to detect. Measured on chain207's own A/A noise over
    the 51 cells, on the SAME construction paired() builds here -- the object pair on
    the `ours` column, whose floor p95 is 39.77% of S-1:

        plant   x1.00   x1.02   x1.05   x1.10   x1.15   x1.25   x1.50
        p95     39.77   37.84   33.90   30.10   36.02   47.85   77.41
        verdict inside  inside  inside  inside  inside  inside  DISPL

    The statistic falls below its own no-effect value from x1.02 through x1.10 and does
    not fire until a x1.50 speedup. The registered mechanism is 7.7-12.4%, so detecting
    it would have needed an effect four to six times larger than the one sought.

    Which A/A construction this is measured on matters, and four of them share the name.
    Over the same 51 cells at k=4, p95 of the per-cell |log|: the object pair on `ours`
    is 39.77% of S-1, on `oursb` 33.29%, an arm pair inside one build 21.34%, and the
    object pair on S = ours/mkl 19.14% -- the last being the sealed analyser's form,
    which divides MKL out of both sides so common-mode noise cancels. paired() builds
    the first. An earlier version of this table was computed on the last and understated
    the defect threefold.

    And the plant above is a UNIFORM rescale, which is degenerate for the corrected
    statistic: arm = floor/g makes every bootstrap replicate exactly g-1 and the interval
    collapses to a point. It is a valid probe of the BROKEN statistic precisely because
    that one is not invariant to a uniform shift -- that invariance failure is the defect.
    For the corrected statistic, power has to be measured with independent numerators;
    amendment 11 does that and finds the registered range detected at +6.86 pp or better.

    A bare comparison of two p95 point estimates is a coin flip when nothing is
    happening: two p95s of the same noise distribution differ by a fraction of a
    percentage point in either direction, so `arm_p95 > floor_p95` fires about half the
    time on a null. The plant in this file's test caught exactly that -- it declared the
    macro live at k=2, where it is provably inert and the planted effect was zero.

    Resampling is on the cell, with the same resampled cells used for both quantities,
    because the arm displacement and the floor displacement are measured on the same
    cell and are not independent of each other.

    The index matrix is digested and printed. A bootstrap bound is not reproducible from
    a seed and a count alone -- the indices are what pin it.
    """
    import hashlib
    n = len(arm)
    rng = random.Random(seed)
    idx = [[rng.randrange(n) for _ in range(n)] for _ in range(B)]
    h = hashlib.sha256(repr(idx).encode()).hexdigest()[:16]
    seps = []
    for row in idx:
        if stat == "center":
            # log of the floor's middle minus log of the arm's middle: positive when the
            # arm is FASTER than an A/A contrast is, which is the one-sided question.
            a = math.log(geo([arm[i] for i in row]))
            f = math.log(geo([flo[i] for i in row]))
            seps.append(math.expm1(f - a))
        else:
            a = p95_upper([arm[i] for i in row])[0]
            f = p95_upper([flo[i] for i in row])[0]
            seps.append((math.exp(a) - 1.0) - (math.exp(f) - 1.0))
    seps.sort()
    lo = seps[int(0.025 * (B - 1))]
    hi = seps[int(0.975 * (B - 1))]
    return lo, hi, h, n


def p95_upper(d):
    """p95 as the UPPER order statistic, index ceil(0.95*(n-1)), printed with its index."""
    s = sorted(d)
    i = math.ceil(0.95 * (len(s) - 1))
    return s[i], i, len(s)


def load(csv_dir, dtype, prefix):
    """(build, pass, key, k, arm) -> ms, plus the per-key dimensions."""
    obs, dims, files = {}, {}, []
    for path in sorted(glob.glob(os.path.join(csv_dir, "%s_%s_*_p*.csv" % (prefix, dtype)))):
        stem = os.path.basename(path)[: -len(".csv")]
        pas = stem.rsplit("_p", 1)[1]
        n = 0
        for r in csv.DictReader(open(path)):
            obs[(r["build"], pas, r["key"], int(r["k"]), r["arm"])] = float(r["ms"])
            dims[r["key"]] = (int(r["rows"]), int(r["nnz"]), float(r["deg"]))
            n += 1
        files.append((os.path.basename(path), n))
    return obs, dims, files


def behind_set(ledger_dir):
    """The sign-only 51, and the ledger's FULL key set at k=4.

    The full set is returned because the scorer's own key list is derived from the CSVs
    it is certifying: a key missing from every object in every pass is not "short", it is
    simply absent from `keys` and from `dims`, and no per-cell completeness check can see
    it. The ledger is an independent record of which matrices the corpus holds, so the
    comparison costs nothing and closes the one direction the completeness guard cannot.

    Returns (the 51, every distinct key at k=4).
    """
    R = collections.defaultdict(lambda: collections.defaultdict(dict))
    for p in sorted(glob.glob(os.path.join(ledger_dir, "dl_*_float32_p*.csv"))):
        b = "aa" if "_aa_" in p else "bb"
        for r in csv.DictReader(open(p)):
            if int(r["k"]) == 4:
                R[(r["key"], b)][p][r["arm"]] = float(r["ms"])
    out = []
    for key in sorted({k for k, _ in R}):
        v = []
        for b in ("aa", "bb"):
            d = R.get((key, b), {})
            v += [d[p]["ours"] / d[p]["mkl"] for p in d if "ours" in d[p] and "mkl" in d[p]]
        if len(v) == 12 and all(x > 1.0 for x in v):
            out.append(key)
    return out, sorted({k for k, _ in R})


def paired(obs, keys, k, arm, lo, hi, passes):
    """Per cell, the geometric middle of the hi/lo ratio paired on the pass index."""
    out = {}
    for key in keys:
        r = []
        for pas in passes:
            a = obs.get((lo, pas, key, k, arm))
            b = obs.get((hi, pas, key, k, arm))
            if a and b:
                r.append(b / a)
        if len(r) == len(passes):
            out[key] = geo(r)
    return out


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    csv_dir, ledger = argv[1], argv[2]
    dtype = argv[3] if len(argv) > 3 else "float32"
    prefix = argv[4] if len(argv) > 4 else "dlmc"

    obs, dims, files = load(csv_dir, dtype, prefix)
    print("== input")
    for nm, n in files:
        print("   %-34s %7d rows" % (nm, n))
    if not obs:
        print("EXIT: no observation was read. Nothing below would mean anything.")
        return 1

    passes = sorted({p for _, p, _, _, _ in obs})
    widths = sorted({k for _, _, _, k, _ in obs})
    arms = sorted({a for _, _, _, _, a in obs})
    builds = sorted({b for b, _, _, _, _ in obs})
    keys = sorted({k for _, _, k, _, _ in obs})
    print("   builds %s | passes %s | widths %s | arms %s | keys %d"
          % (builds, passes, widths, arms, len(keys)))

    # --- completeness. A total is not a completeness check; a per-cell minimum is.
    want = len(BUILDS) * len(passes)
    per = collections.Counter()
    for (b, p, key, k, a) in obs:
        if a == "ours":
            per[(key, k)] += 1
    short = [c for c, n in per.items() if n != want]
    print("== completeness")
    print("   cells (key,width) with an `ours` reading: %d" % len(per))
    print("   readings per cell: want %d (4 objects x %d passes), min %d, short cells %d"
          % (want, len(passes), min(per.values()) if per else 0, len(short)))
    missing = [b for b in BUILDS if b not in builds]
    if missing or short or len(passes) < 4:
        print("   GUARD FAILED: missing objects %s, %d short cells, %d passes."
              % (missing or "none", len(short), len(passes)))
        print("EXIT: the screen is not complete. No table is printed, because a table")
        print("      printed from a partial rotation reads as a result and is not one.")
        return 1
    print("   GUARD: every cell carries all 4 objects in all %d passes" % len(passes))

    # --- the behind set, matched into this corpus by an explicit, reported prefix strip
    led, led_all = behind_set(ledger)
    strip = lambda s: s.split("/", 1)[1] if s.startswith("dlmc/") else s
    b51 = [strip(k) for k in led]
    led_keys = {strip(k) for k in led_all}
    absent = sorted(led_keys - set(keys))
    extra = sorted(set(keys) - led_keys)
    print("   ledger corpus at k=4: %d distinct keys; this run carries %d"
          % (len(led_keys), len(keys)))
    print("   in the ledger but ABSENT from every object in every pass: %d" % len(absent))
    for k in absent[:5]:
        print("      %s" % k)
    print("   in this run but not in the ledger: %d" % len(extra))
    if absent:
        print("EXIT: a key absent from the whole run is invisible to the per-cell")
        print("      completeness check -- it is not short, it is missing -- so the")
        print("      populations below would silently be narrower than they claim.")
        return 1
    hit = [k for k in b51 if k in set(keys)]
    print("== populations")
    print("   sign-only behind set from the ledger: %d keys (every one of 12 readings > 1.0)" % len(led))
    print("   resolved into this corpus after stripping a leading 'dlmc/': %d of %d" % (len(hit), len(b51)))
    if len(hit) != len(b51):
        print("EXIT: the behind set does not resolve. A partial match silently rescores")
        print("      a different population, which is invisible in the result.")
        return 1
    rest = [k for k in keys if k not in set(hit)]
    print("   condition 4 population, the cells not in that set: %d" % len(rest))

    for arm in ("ours", "oursb"):
        print("\n" + "=" * 78)
        print("ARM COLUMN %r" % arm)
        print("=" * 78)

        # ---------------- the floor, per width, from the byte-identical pair
        print("\n-- the A/A floor, %s vs %s, byte-identical objects" % FLOOR_BUILDS)
        floor = {}
        for k in widths:
            d = paired(obs, keys, k, arm, FLOOR_BUILDS[0], FLOOR_BUILDS[1], passes)
            if not d:
                continue
            dev = [abs(math.log(x)) for x in d.values()]
            v, i, n = p95_upper(dev)
            floor[k] = math.exp(v) - 1.0
            print("   k=%-3d n=%4d  p95 |log| displacement = %6.2f%% of S-1   (upper order"
                  " statistic, index %d of %d)" % (k, n, 100 * floor[k], i, n))

        # ---------------- condition 1, the firing check
        print("\n-- CONDITION 1  the macro is live at k=4: %s vs %s" % (FIRE, ARM))
        print("   admission is ROWS-independent beyond multirow > 1 (spmm.h:4410-4425), so a")
        print("   difference between ROWS=4 and ROWS=2 beyond the floor establishes that the")
        print("   template is instantiated and taken at this width. %s vs %s cannot serve:"
              % (ARM, FLOOR_BUILDS[0]))
        print("   that contrast IS the effect under test.")
        print("   k=2 is this criterion's OWN null: multirow_halfvec is inert where the exact")
        print("   kernel owns k, so a firing there is a false positive and calibrates the rest.")
        for k in (2, 4):
            if k not in widths:
                continue
            d = paired(obs, keys, k, arm, ARM, FIRE, passes)
            fd = paired(obs, keys, k, arm, FLOOR_BUILDS[0], FLOOR_BUILDS[1], passes)
            cells = sorted(set(d) & set(fd))
            if not cells:
                continue
            av = [abs(math.log(d[c])) for c in cells]
            fv = [abs(math.log(fd[c])) for c in cells]
            lo, hi, h, n = boot_sep(av, fv, seed=97 + k)
            print("   k=%-3d n=%4d  geomean %s/%s = %.4f   separation p95(arm)-p95(floor)"
                  " = %s  -> %s"
                  % (k, n, FIRE, ARM, geo([d[c] for c in cells]), fmt_sep(lo, hi),
                     "FIRES" if lo > 0 else "inside the floor"))
            print("          index matrix sha %s, B=2000, resampled on the cell" % h)

        # ---------------- condition 2, k=2 neutrality
        print("\n-- CONDITION 2  k=2 neutrality: the macro is inert where the exact kernel owns k")
        if 2 in widths:
            d = paired(obs, keys, 2, arm, FLOOR_BUILDS[0], ARM, passes)
            fd = paired(obs, keys, 2, arm, FLOOR_BUILDS[0], FLOOR_BUILDS[1], passes)
            cells = sorted(set(d) & set(fd))
            av = [abs(math.log(d[c])) for c in cells]
            fv = [abs(math.log(fd[c])) for c in cells]
            lo, hi, h, n = boot_sep(av, fv, seed=211)
            print("   n=%4d  geomean %s/%s = %.4f" % (n, ARM, FLOOR_BUILDS[0],
                                                      geo([d[c] for c in cells])))
            print("   separation p95(arm) - p95(floor) = %s" % fmt_sep(lo, hi))
            print("          index matrix sha %s, B=2000, resampled on the cell" % h)
            print("   EQUIVALENCE, not failure to detect: neutrality is the interval CONTAINING")
            print("   zero, and that is only evidence of neutrality because the same construction")
            print("   separates the k=4 effect, so it is not merely underpowered.")
            print("   -> %s" % ("NEUTRAL" if lo <= 0.0 else
                                "NOT neutral: rule out the layout artefact first, it is"
                                " largest at the narrowest width"))

        # ---------------- condition 3 primary, the gain curve, no threshold
        print("\n-- CONDITION 3 PRIMARY  the gain curve on the behind set, no threshold applied")
        print("   amendment 2: a threshold on this number can only return zero, so the primary")
        print("   reports the curve and lets its shape answer. Both units are printed, because")
        print("   the mechanism's reachability was verified as FACTORS (x1.1008 and so on) and")
        print("   a factor may only be set against the S-1 column.")
        print("   The last column is the A/A GEOMEAN on the same cells -- the functional")
        print("   matched to the one beside it. The per-cell p95 floor that used to sit here")
        print("   was a pooled centre set against a per-cell tail over a different")
        print("   population, and it printed a verdict in a block that decides nothing.")
        print("   %-5s %5s  %-14s  %-14s  %-16s %s"
              % ("k", "n", "speedup factor", "pct of S-1", "pct of 1-1/S", "A/A geomean"))
        for k in widths:
            d = paired(obs, hit, k, arm, FLOOR_BUILDS[0], ARM, passes)
            if not d:
                continue
            r = geo(list(d.values()))            # arm / control
            g = 1.0 / r                          # the speedup as a factor
            aa = paired(obs, hit, k, arm, FLOOR_BUILDS[0], FLOOR_BUILDS[1], passes)
            print("   k=%-3d %5d  x%-13.4f  %+13.2f%%  %+15.2f%%  %.4f"
                  % (k, len(d), g, 100 * (g - 1.0), 100 * (1.0 - r),
                     geo(list(aa.values())) if aa else float("nan")))

        # ---------------- condition 3 secondary, displacement against its own floor
        print("\n-- CONDITION 3 SECONDARY  k=4 geomean displacement vs the geomean's own floor")
        print("   amendment 2 registers this as the k=4 GEOMEAN against the geomean's own")
        print("   floor from the mr0/mr0b pair, and amendment 10 corrects the implementation:")
        print("   it had used the two-sided p95 of per-cell |log|, which is non-monotone in a")
        print("   one-sided effect and reads 'inside the floor' across the whole registered")
        print("   mechanism range. The functional below is the registered one.")
        if 4 in widths:
            d = paired(obs, hit, 4, arm, FLOOR_BUILDS[0], ARM, passes)
            fd = paired(obs, hit, 4, arm, FLOOR_BUILDS[0], FLOOR_BUILDS[1], passes)
            cells = sorted(set(d) & set(fd))
            dv = [d[c] for c in cells]
            fv = [fd[c] for c in cells]
            lo, hi, h, n = boot_sep(dv, fv, seed=409, stat="center")
            print("   n=%3d  arm geomean %.4f   A/A geomean %.4f" % (n, geo(dv), geo(fv)))
            print("   the arm's speedup over the A/A contrast = %s" % fmt_sep(lo, hi))
            print("          index matrix sha %s, B=2000, resampled on the cell, stat=center" % h)
            print("   -> %s" % ("DISPLACED beyond the floor" if lo > 0 else "inside the floor"))

        # ---------------- condition 3 descriptive
        print("\n-- CONDITION 3 DESCRIPTIVE  parity counts beside flip counts")
        print("   amendment 1(b): a count is a threshold in disguise, so these are descriptive")
        print("   only and decide nothing. The flip count is printed beside them because two")
        print("   byte-identical builds disagreed on 25 of 51 verdicts at the old pass line.")
        if 4 in widths:
            d = paired(obs, hit, 4, arm, FLOOR_BUILDS[0], ARM, passes)
            fd = paired(obs, hit, 4, arm, FLOOR_BUILDS[0], FLOOR_BUILDS[1], passes)
            better = sum(1 for v in d.values() if v < 1.0)
            flip = sum(1 for v in fd.values() if v < 1.0)
            print("   of %d cells, the arm is faster than the control on %d; the two"
                  " byte-identical builds disagree on %d" % (len(d), better, flip))

        # ---------------- condition 4, no stratification
        print("\n-- CONDITION 4  no regression off the behind set, UNSTRATIFIED (amendment 7)")
        print("   the row-dispersion stratifier is withdrawn: it sorts cells by their own A/A")
        print("   floor (2.7x across terciles) and predicts the outcome (rho -0.62). Both of")
        print("   those are chain207 6-pass figures, cited as the GROUNDS for a withdrawal and")
        print("   not as thresholds -- per amendment 9(4) no chain207 floor transfers to a")
        print("   4-pass run, and nothing here compares against either number.")
        print("   The null is MEASURED, not asserted. The floor is a p95 of |log|, so 5% of")
        print("   cells exceed it on EITHER side and only about half of those on the upper")
        print("   side the count tests -- the printed 5.00%% was roughly twice the right")
        print("   number, in the direction that makes a clean run look like a regression.")
        print("   So the A/A pair's own one-sided exceedance is computed per width and")
        print("   printed beside the arm's, which needs no distributional assumption.")
        for k in widths:
            d = paired(obs, rest, k, arm, FLOOR_BUILDS[0], ARM, passes)
            if not d:
                continue
            f = floor.get(k, float("nan"))
            aa = paired(obs, rest, k, arm, FLOOR_BUILDS[0], FLOOR_BUILDS[1], passes)
            null = (100.0 * sum(1 for v in aa.values() if math.log(v) > math.log(1.0 + f))
                    / len(aa)) if aa else float("nan")
            worse = [(key, v) for key, v in d.items() if math.log(v) > math.log(1.0 + f)]
            wk, wv = max(d.items(), key=lambda kv: kv[1])
            print("   k=%-3d n=%4d  geomean arm/control %.4f  cells beyond the floor %4d ="
                  " %5.2f%%  vs the A/A pair's own %5.2f%% at the same floor"
                  % (k, len(d), geo(list(d.values())), len(worse),
                     100 * len(worse) / len(d), null))
            print("          worst cell %s  %.4f  rows=%d nnz=%d deg=%.1f"
                  % (wk, wv, dims[wk][0], dims[wk][1], dims[wk][2]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
