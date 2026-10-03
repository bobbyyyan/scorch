"""Conditions 2 and 4 of PREREG_thread_sizing: how much of the deficit a thread rule can REACH.

WRITTEN BEFORE THE SWEEP IT SCORES REPORTED. Job 17698209 sweeps all 392 float32 DLMC keys; this
script is the only thing that computes its reach shares, and every definition below is fixed by
amendments 21(c), 22, 24 and 26 rather than chosen here.

THE STATISTIC, with all three of its degrees of freedom named, because fixing one hid the next
twice already:
  UNITS      -- a share, read against the deficit's OWN prevalence (328 of 392 = 83.7%), not
                against an absolute threshold. Amendment 22(b).
  DENOMINATOR-- ALL keys of the half, with non-live keys INCLUDED. A non-live key reaches one
                worker count so a thread rule cannot act on it, but it can perfectly well be
                behind MKL, and the denominator counts what must be EXPLAINED. Amendment 24.
  NUMERATOR  -- keys that resolve to <= 2 workers AND are LIVE. The numerator counts where the
                mechanism can ACT, and a cell with one reachable count has no argmin, so the
                headroom statistic is undefined there rather than small. Amendment 26.
The two halves take DIFFERENT liveness predicates by design; that is the asymmetry, not an
inconsistency. Every alternative numerator and denominator is printed beside the registered one so
a later reader can recover any reading rather than having to trust this docstring.

CONDITION 2's BANDS, in units of the explained quantity (amendment 22(b)):
  s328 >= 0.50          -- reaches the majority; a causal claim is available
  0.25 <= s328 < 0.50   -- a share with NO causal claim; 1 - s328 needs another mechanism
  s328 <  0.25          -- a BOUND ON REACH, not a pass

CONDITION 4 (amendments 22(c), 24, 26): the statistic is s328 - s64 over the 64 keys that are NOT
behind -- the clean control, same corpus, same run, same instrument, OUTSIDE the population being
certified. A bootstrap CI containing zero means the share is a property of the corpus and not of
being behind MKL. The live fraction is reported per half, and a difference above 5 percentage
points makes the DENOMINATOR carry a composition effect; the pinned numerator is what keeps the
NUMERATOR from carrying one.
"""
import argparse
import collections
import csv
import hashlib
import random


def shares(keys, res, live):
    """Returns the registered share and every alternative reading of it."""
    present = [k for k in keys if k in res]
    if not present:
        return None
    n = len(present)
    all_le2 = [k for k in present if res[k] <= 2]
    liv = [k for k in present if k in live]
    liv_le2 = [k for k in present if k in live and res[k] <= 2]
    return {
        "n": n, "absent": len(keys) - n,
        "registered": len(liv_le2) / n,          # amendment 26: live & <=2 over ALL keys
        "all_le2_over_all": len(all_le2) / n,    # the unpinned reading, 3.11x on the 72-key corpus
        "live_le2_over_live": (len(liv_le2) / len(liv)) if liv else float("nan"),
        "n_all_le2": len(all_le2), "n_live": len(liv), "n_live_le2": len(liv_le2),
        "live_frac": len(liv) / n,
        "members": sorted(liv_le2),
    }


def boot_diff(a_keys, b_keys, res, live, nboot=4000, seed=20261002, alpha=0.05):
    """Bootstrap CI on sA - sB, resampling KEYS within each half with replacement.

    Keys are the sampling unit because they are what was drawn: the sweep measures every key at
    every width, so a cell is not an independent draw from the corpus and resampling cells would
    understate the interval. Returns (lo, hi, vdig, idig, nA, nB) -- a value digest AND an index
    digest, because an index-only digest is a pure function of (seed, nboot, n) and reproduces
    across completely different data, which is how a bound got published that nothing could check.
    """
    A = [k for k in a_keys if k in res]
    B = [k for k in b_keys if k in res]
    if len(A) < 8 or len(B) < 8:
        return None
    rnd = random.Random(seed)
    idx, out = [], []

    def s(sample):
        return sum(1 for k in sample if k in live and res[k] <= 2) / len(sample)

    for _ in range(nboot):
        ia = [rnd.randrange(len(A)) for _ in range(len(A))]
        ib = [rnd.randrange(len(B)) for _ in range(len(B))]
        idx.append((tuple(ia), tuple(ib)))
        out.append(s([A[i] for i in ia]) - s([B[i] for i in ib]))
    out.sort()
    lo = out[int((alpha / 2) * (nboot - 1))]
    hi = out[int((1 - alpha / 2) * (nboot - 1))]
    vdig = hashlib.sha256("|".join("%.17g" % x for x in out).encode()).hexdigest()[:12]
    idig = hashlib.sha256(repr(idx).encode()).hexdigest()[:12]
    return lo, hi, vdig, idig, len(A), len(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, help="the 392-key ntoracle sweep CSV")
    ap.add_argument("--behind", required=True, help="the 328 keys behind in EVERY reading at --k, "
                    "from an_f64_vs_mkl.py --dump-behind")
    ap.add_argument("--corpus", required=True, help="all 392 keys the screen measured")
    ap.add_argument("--k", type=int, required=True, help="the width. REQUIRED: the behind set and "
                    "the resolved count both differ by width")
    A = ap.parse_args()

    behind = set(l.strip() for l in open(A.behind) if l.strip())
    corpus = set(l.strip() for l in open(A.corpus) if l.strip())
    if not behind or not corpus:
        raise SystemExit("REFUSE: --behind or --corpus is empty; every share would be 0/0")
    if not behind <= corpus:
        raise SystemExit("REFUSE: %d behind keys lie outside the corpus list, so the control half "
                         "is not 'the corpus minus the deficit'" % len(behind - corpus))
    control = corpus - behind
    if not control:
        raise SystemExit("REFUSE: the corpus IS the behind set, so there is no control half and "
                         "condition 4 is undefined rather than null")

    seen = collections.defaultdict(set)
    reach = collections.defaultdict(set)
    rows = 0
    sha = collections.Counter()
    for r in csv.DictReader(open(A.oracle)):
        rows += 1
        sha[r.get("so_sha", "").strip()] += 1
        try:
            if int(r["k"]) != A.k:
                continue
            seen[r["key"]].add(int(r["res_ovr0"]))
            reach[r["key"]].add(int(r["resolved"]))
        except (KeyError, TypeError, ValueError):
            continue
    blank = sha.pop("", 0)
    if len(sha) > 1:
        raise SystemExit("REFUSE: %d distinct objects in one sweep; a share pooled over two "
                         "objects is not a share of either" % len(sha))
    bad = {k: sorted(v) for k, v in seen.items() if len(v) != 1}
    if bad:
        raise SystemExit("REFUSE: %d key(s) carry more than one res_ovr0 at k=%d, so production's "
                         "count is not a per-cell constant. First: %r"
                         % (len(bad), A.k, sorted(bad.items())[0]))
    res = {k: next(iter(v)) for k, v in seen.items()}
    live = {k for k, v in reach.items() if len(v) >= 2}

    print("ORACLE     %s: %d rows, object(s) %s%s"
          % (A.oracle, rows, (sorted(sha)[0][:16] if sha else "UNKNOWN"),
             ("   [%d rows carry NO so_sha]" % blank) if blank else ""))
    print("WIDTH      k=%d   keys with a production count: %d" % (A.k, len(res)))
    print("HALVES     deficit %d, control %d, corpus %d  (asserted: deficit is a subset)"
          % (len(behind), len(control), len(corpus)))

    out = {}
    for lab, keys in (("DEFICIT (s328)", behind), ("CONTROL (s64)", sorted(control))):
        d = shares(keys, res, live)
        print()
        if d is None:
            print("  %s: REFUSE-worthy -- no key of this half is in the sweep, so its share would"
                  % lab)
            print("  be 0/0 and would print as a measured zero. Not scored.")
            continue
        out[lab] = d
        print("  %s   n=%d present, %d ABSENT from the sweep (unmeasured, NOT excluded)"
              % (lab, d["n"], d["absent"]))
        print("    REGISTERED  live & <=2 over ALL keys of the half : %3d/%-3d = %.4f"
              % (d["n_live_le2"], d["n"], d["registered"]))
        print("    alternative all <=2        over all keys         : %3d/%-3d = %.4f  (%.2fx)"
              % (d["n_all_le2"], d["n"], d["all_le2_over_all"],
                 (d["all_le2_over_all"] / d["registered"]) if d["registered"] else float("inf")))
        print("    alternative live & <=2     over LIVE keys        : %3d/%-3d = %.4f"
              % (d["n_live_le2"], d["n_live"], d["live_le2_over_live"]))
        print("    live fraction of the half                        : %3d/%-3d = %.4f"
              % (d["n_live"], d["n"], d["live_frac"]))
        print("    non-live cells inside the <=2 group              : %3d of %d"
              % (d["n_all_le2"] - d["n_live_le2"], d["n_all_le2"]))

    if "DEFICIT (s328)" not in out:
        raise SystemExit("REFUSE: the deficit half is empty, so condition 2 is undefined")
    s328 = out["DEFICIT (s328)"]["registered"]
    print()
    print("CONDITION 2, bands fixed in amendment 22(b) and read on the REGISTERED numerator:")
    if s328 >= 0.50:
        v = "reaches the MAJORITY of the deficit; a causal claim is available"
    elif s328 >= 0.25:
        v = ("a SHARE WITH NO CAUSAL CLAIM: reaches %.1f%% of the deficit, and the remaining "
             "%.1f%% needs a different mechanism" % (100 * s328, 100 * (1 - s328)))
    else:
        v = ("a BOUND ON REACH, NOT A PASS: at most %.1f%% of the deficit, so thread sizing "
             "cannot be its principal cause whatever the per-cell excess" % (100 * s328))
    print("  s328 = %.4f  ->  %s" % (s328, v))
    print("  Against the deficit's own prevalence of 83.7%%, this mechanism reaches %.2fx less."
          % (0.837 / s328) if s328 > 0 else "  s328 is zero; the mechanism reaches nothing.")

    if "CONTROL (s64)" in out:
        s64 = out["CONTROL (s64)"]["registered"]
        lf = abs(out["DEFICIT (s328)"]["live_frac"] - out["CONTROL (s64)"]["live_frac"])
        print()
        print("CONDITION 4: s328 - s64 = %.4f - %.4f = %+.4f" % (s328, s64, s328 - s64))
        b = boot_diff(sorted(behind), sorted(control), res, live)
        if b is None:
            print("  NOT SCOREABLE: a half has under 8 present keys, so a bootstrap CI on the")
            print("  difference would be narrower than the resampling can support.")
        else:
            lo, hi, vdig, idig, na, nb = b
            spans = lo <= 0.0 <= hi
            print("  bootstrap 95%% CI [%+.4f, %+.4f]  nA=%d nB=%d  values %s indices %s"
                  % (lo, hi, na, nb, vdig, idig))
            print("  -> %s" % ("CONTAINS ZERO: the <=2 share is a property of the CORPUS, not of "
                               "being behind MKL. Condition 4's registered verdict applies -- the "
                               "mechanism does not track the deficit whatever its per-cell excess."
                               if spans else
                               ("EXCLUDES ZERO, %s: the <=2 share is %s among deficit keys."
                                % (("NEGATIVE" if hi < 0 else "POSITIVE"),
                                   ("LOWER" if hi < 0 else "HIGHER")))))
        print("  live fractions %.4f (deficit) vs %.4f (control), differing by %.1f pp -> %s"
              % (out["DEFICIT (s328)"]["live_frac"], out["CONTROL (s64)"]["live_frac"], 100 * lf,
                 "within the 5 pp tripwire" if lf <= 0.05 else
                 "ABOVE the 5 pp tripwire: the DENOMINATOR carries a composition effect shaped "
                 "like the association under test, so the difference is not a difference of "
                 "prevalences. The registered numerator keeps the NUMERATOR clean; this is the "
                 "denominator's."))
    else:
        print()
        print("CONDITION 4: NOT SCOREABLE -- the control half contributed no key to the sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
