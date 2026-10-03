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


def boot_sep(arm, flo, seed, B=2000):
    """CI for p95(arm) - p95(floor) by resampling CELLS, which are paired in both.

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
    """The sign-only 51: every one of 12 readings at k=4 above 1.0. No median anywhere."""
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
    return out


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
    led = behind_set(ledger)
    strip = lambda s: s.split("/", 1)[1] if s.startswith("dlmc/") else s
    b51 = [strip(k) for k in led]
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
                  " = [%+5.2f, %+5.2f] pp of S-1  -> %s"
                  % (k, n, FIRE, ARM, geo([d[c] for c in cells]), 100 * lo, 100 * hi,
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
            print("   separation p95(arm) - p95(floor) = [%+5.2f, %+5.2f] pp of S-1" % (100 * lo, 100 * hi))
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
        print("   %-5s %5s  %-14s  %-14s  %-16s %s"
              % ("k", "n", "speedup factor", "pct of S-1", "pct of 1-1/S", "floor, of S-1"))
        for k in widths:
            d = paired(obs, hit, k, arm, FLOOR_BUILDS[0], ARM, passes)
            if not d:
                continue
            r = geo(list(d.values()))            # arm / control
            g = 1.0 / r                          # the speedup as a factor
            f = floor.get(k, float("nan"))
            print("   k=%-3d %5d  x%-13.4f  %+13.2f%%  %+15.2f%%  %7.2f%%   %s"
                  % (k, len(d), g, 100 * (g - 1.0), 100 * (1.0 - r), 100 * f,
                     "beyond" if (g - 1.0) > f else "inside"))

        # ---------------- condition 3 secondary, displacement against its own floor
        print("\n-- CONDITION 3 SECONDARY  k=4 displacement vs the floor at the same n")
        if 4 in widths:
            d = paired(obs, hit, 4, arm, FLOOR_BUILDS[0], ARM, passes)
            fd = paired(obs, hit, 4, arm, FLOOR_BUILDS[0], FLOOR_BUILDS[1], passes)
            cells = sorted(set(d) & set(fd))
            dv = [abs(math.log(d[c])) for c in cells]
            fv = [abs(math.log(fd[c])) for c in cells]
            a, ai, an = p95_upper(dv)
            b, bi, bn = p95_upper(fv)
            print("   arm   n=%3d  p95 = %6.2f%% of S-1  (upper order statistic, index %d of %d)"
                  % (an, 100 * (math.exp(a) - 1), ai, an))
            print("   floor n=%3d  p95 = %6.2f%% of S-1  (upper order statistic, index %d of %d)"
                  % (bn, 100 * (math.exp(b) - 1), bi, bn))
            lo, hi, h, n = boot_sep(dv, fv, seed=409)
            print("   separation = [%+.2f, %+.2f] pp of S-1   index matrix sha %s, B=2000"
                  % (100 * lo, 100 * hi, h))
            print("   -> %s" % ("DISPLACED beyond the floor" if lo > 0 else
                                "inside the floor"))

        # ---------------- condition 3 descriptive
        print("\n-- CONDITION 3 DESCRIPTIVE  parity counts beside flip counts")
        print("   amendment 1(b): a count is a threshold in disguise, so these are descriptive")
        print("   only and decide nothing. The flip count is printed beside them because two")
        print("   byte-identical builds disagreed on 25 of 51 verdicts at the old pass line.")
        if 4 in widths:
            ctl = paired(obs, hit, 4, "ours", FLOOR_BUILDS[0], FLOOR_BUILDS[0], passes)
            d = paired(obs, hit, 4, arm, FLOOR_BUILDS[0], ARM, passes)
            fd = paired(obs, hit, 4, arm, FLOOR_BUILDS[0], FLOOR_BUILDS[1], passes)
            better = sum(1 for v in d.values() if v < 1.0)
            flip = sum(1 for v in fd.values() if v < 1.0)
            print("   of %d cells, the arm is faster than the control on %d; the two"
                  " byte-identical builds disagree on %d" % (len(d), better, flip))

        # ---------------- condition 4, no stratification
        print("\n-- CONDITION 4  no regression off the behind set, UNSTRATIFIED (amendment 7)")
        print("   the row-dispersion stratifier is withdrawn: it sorts cells by their own A/A")
        print("   floor (2.7x across terciles) and predicts the outcome (rho -0.62).")
        for k in widths:
            d = paired(obs, rest, k, arm, FLOOR_BUILDS[0], ARM, passes)
            if not d:
                continue
            f = floor.get(k, float("nan"))
            worse = [(key, v) for key, v in d.items() if math.log(v) > math.log(1.0 + f)]
            wk, wv = max(d.items(), key=lambda kv: kv[1])
            print("   k=%-3d n=%4d  geomean arm/control %.4f  cells beyond the floor %4d = %5.2f%%"
                  " (null 5.00%% by the floor's own definition)"
                  % (k, len(d), geo(list(d.values())), len(worse), 100 * len(worse) / len(d)))
            print("          worst cell %s  %.4f  rows=%d nnz=%d deg=%.1f"
                  % (wk, wv, dims[wk][0], dims[wk][1], dims[wk][2]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
