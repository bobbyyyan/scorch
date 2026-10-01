#!/usr/bin/env python3
"""Where does V6's asymptotic win stop paying as the row degree falls?

THE QUESTION. The 11-arm probe put V6 (4 accumulator chains) at -21.259% on the P-core and
-4.032% on the E-core over a single row of 4096 nonzeros. DLMC's per-ROW length distribution, measured from the
CSR row pointers over all 3846 files, reads p10 11, p25 32, median 103, p75 208, p90 278 over the
3255080 rows that reach the dispatch -- so that row is 40x the median production ROW. (Place the
bracket against THIS and not against chain210's `deg`, which is a per-matrix mean; the two agree
within 1% at the median on DLMC and differ by a quarter at p25.) V6 buys a per-NONZERO gain with a per-ROW cost: four chains
fold in two dependent vaddps levels where two chains fold in one. Below some degree the trade
inverts and V6 is a regression, and the project's performance convention makes a regression a
blocker however large the asymptotic win is.

THE ESTIMATOR, registered in sweep_degree.sh before the sweep ran. The probe reports
nspn = t_call / (inner*n), so t_call = nspn*n. Per arm, least squares on

    t_call(n) = a + b*n

b is the per-nonzero cost; a is the per-call constant = function-call overhead + fold. The call
overhead is identical across arms, so it cancels in a difference of intercepts:

    delta_a = a(V6) - a(V0)        the extra fold cost alone
    n* = delta_a / (b(V0) - b(V6)) the overhead-free break-even degree

Production has no per-row function call -- it has an inlined row loop -- so the raw measured
ratio at small n UNDERSTATES V6's penalty, because the shared overhead sits in both numerator
and denominator and pulls the ratio toward 1 exactly where the penalty is. Both are printed: the
raw ratio, which needs no model, and the fit, which removes an overhead production does not pay.

REFUSALS. A CSV must carry a '# binary_sha256' matching its own ATTEST-<digest>.txt with
asmcheck_exit 0; the ladder must have at least 4 points for a fit; and the fit is reported with
its R^2 and residuals so a curved t_call(n) is visible rather than silently linearised.
"""
import collections, csv, glob, math, os, re, statistics, sys
# A quarantined artefact (chain194's hv_ex4b_float64_p2.contaminated.csv) is sealed
# evidence that this file's glob admits. LC refuses rather than letting it join a
# recount silently, or be dropped silently, which are both decisions for a reader.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ledger_csvs as LC

# ---- each arm's MACHINE body width, derived from the attested disassembly.
# WHY THIS IS NOT READ FROM THE SOURCE. probe.cpp's V6 steps `p += 4`, but -funroll-loops doubled
# it: the hot loop emits EIGHT FMAs, so the machine body consumes eight nonzeros. V0 steps p += 2
# and also emits eight; V7 steps p += 8 and ALSO emits eight, so the compiler doubled V0 four
# times, V6 twice and V7 not at all. Taking the width from the source would understate V0 by 4x
# and V6 by 2x while getting V7 right -- in three different directions.
#
# WHY IT MATTERS FOR THE FIT. Below its machine body width an arm runs only the scalar remainder,
# which has ONE accumulator chain and a different per-nonzero cost. That is a DIFFERENT KERNEL
# from the one being fitted, and a single line fitted across the boundary absorbs the difference
# into the intercept -- which is the estimand. The peer simulated it with a +0.20 ns/nnz remainder
# penalty (57% of b): the full ladder returns n* = 7.23 against a truth of 4.71, a 54% inflation
# far outside the noise interquartile range, and UPWARD -- so the error manufactures a degree
# guard that V6 may not need. Restricting to n >= body recovers 4.72 exactly.
#
# The excluded degrees are reported as raw ratios instead, which need no model and are the honest
# answer at a degree where the remainder IS what production would run.
MEMOP = re.compile(r"-?(0x[0-9a-f]+)?\(%[a-z0-9]+(,%[a-z0-9]+(,[0-9])?)?\)")
EPILOGUE = ("ret", "pop", "push", "call")
SYMS = {"_Z2v0RK4Args": "V0", "_Z3v0bRK4Args": "V0b", "_Z2v1RK4Args": "V1", "_Z2v2RK4Args": "V2",
        "_Z2v3RK4Args": "V3", "_Z2v4RK4Args": "V4", "_Z3v4bRK4Args": "V4b",
        "_Z3v4cRK4Args": "V4c", "_Z2v5RK4Args": "V5", "_Z2v6RK4Args": "V6", "_Z2v7RK4Args": "V7"}


def body_widths(asm_path):
    """FMA count of each arm's innermost loop = nonzeros consumed per machine body."""
    fn, cur = collections.defaultdict(list), None
    for ln in open(asm_path):
        m = re.match(r"^([0-9a-f]+) <(\S+)>:", ln)
        if m:
            cur = SYMS.get(m.group(2))
            continue
        if cur is None:
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$", ln.rstrip())
        if m:
            fn[cur].append((int(m.group(1), 16), m.group(2), m.group(3).split("#")[0].strip()))
    out = {}
    for arm, ins in fn.items():
        addrs = {a for a, _, _ in ins}
        sp = set()
        for a, mn, ops in ins:
            if not mn.startswith("j"):
                continue
            mm = re.match(r"^([0-9a-f]+)", ops)
            if mm and int(mm.group(1), 16) in addrs and int(mm.group(1), 16) < a:
                sp.add((int(mm.group(1), 16), a))
        cand = [(t, a) for t, a in sorted(sp, key=lambda x: x[1] - x[0])
                if not any(mn.startswith(EPILOGUE) for x, mn, _ in ins if t <= x <= a)]
        best = 0
        for t, a in cand:
            nf = sum(1 for x, mn, _ in ins if t <= x <= a and mn.startswith("vfmadd"))
            best = max(best, nf)
        for t, a in cand:
            nf = sum(1 for x, mn, _ in ins if t <= x <= a and mn.startswith("vfmadd"))
            if nf == best and best > 0:
                out[arm] = best
                break
    return out



KEEP = ("V0", "V0b", "V6", "V7", "V1", "V5")
red = lambda v: math.exp(statistics.median([math.log(x) for x in v]))
dev = lambda x: math.exp(abs(math.log(x))) - 1.0


def read(path):
    binsha, rows = None, collections.defaultdict(list)
    for ln in open(path):
        t = ln.split()
        if ln.startswith("#"):
            if len(t) == 3 and t[1] == "binary_sha256":
                binsha = t[2]
            continue
        break
    for r in csv.DictReader(l for l in open(path) if not l.startswith("#")):
        try:
            rows[r["arm"]].append(float(r["ns_per_nnz"]))
        except (KeyError, TypeError, ValueError):
            continue
    return binsha, rows


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: an_degree.py <dir-holding-pbdeg_n*_core*.csv-and-ATTEST-*.txt>")
    D = os.path.abspath(sys.argv[1])
    cores = {}
    for f in sorted(LC.glob_clean(D, "pbdeg_n*_core*.csv", label="an_degree")):
        m = re.search(r"pbdeg_n(\d+)_core(\d+)\.csv$", f)
        n, core = int(m.group(1)), int(m.group(2))
        binsha, rows = read(f)
        if binsha is None:
            sys.exit("REFUSE: %s carries no '# binary_sha256'." % f)
        att = os.path.join(D, "ATTEST-%s.txt" % binsha[:16])
        if not os.path.exists(att):
            sys.exit("REFUSE: %s names binary %s with no attestation at %s." % (f, binsha[:16], att))
        if not any(l.strip() == "asmcheck_exit 0" for l in open(att)):
            sys.exit("REFUSE: %s does not record asmcheck_exit 0." % att)
        if not rows.get("V0"):
            sys.exit("REFUSE: %s has no V0 readings." % f)
        cores.setdefault(core, {})[n] = {a: red(v) for a, v in rows.items() if v}
        cores[core][n]["__floor__"] = _floor(rows)
    if not cores:
        sys.exit("REFUSE: no pbdeg_n*_core*.csv found under %s" % D)

    asms = sorted(glob.glob(os.path.join(D, "probe-*.asm")))
    if len(asms) != 1:
        sys.exit("REFUSE: expected exactly one probe-<digest>.asm beside the data so each arm's\n"
                 "  machine body width is DERIVED rather than assumed; found %d. Without it the\n"
                 "  fit cannot know which ladder points run only the scalar remainder, and the\n"
                 "  peer's simulation puts that error at +54%% on n*, upward." % len(asms))
    BW = body_widths(asms[0])
    TAILS = arm_tails(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe.cpp"))
    for a in ("V0", "V6", "V7"):
        if a not in BW or BW[a] < 2:
            sys.exit("REFUSE: no machine body width for %s in %s." % (a, asms[0]))
    print("MACHINE BODY WIDTHS from %s (FMAs in the innermost loop = nonzeros per body):"
          % os.path.basename(asms[0]))
    print("  " + "   ".join("%s=%d" % (a, BW[a]) for a in sorted(BW)))
    print("  probe.cpp steps V0 by 2, V6 by 4, V7 by 8, so -funroll-loops doubled V0 four times,")
    print("  V6 twice and V7 not at all. The source would mis-state all three, in three")
    print("  different directions.")

    for core in sorted(cores):
        S = cores[core]
        ns = sorted(S)
        print()
        print("=" * 100)
        print("cpu%d -- %d degrees: %s" % (core, len(ns), " ".join(str(n) for n in ns)))
        print("  (floor = p95 of per-repetition dev(V0/V0b), computed separately at every degree,")
        print("   because the A/A floor is not degree-invariant: a shorter row is a shorter timed")
        print("   call and noise does not scale with it.)")
        print()
        print("  %5s %10s %10s %10s %9s %9s %9s %8s" %
              ("deg", "V0 ns/nnz", "V6 ns/nnz", "V7 ns/nnz", "V6/V0", "V7/V0", "V7/V6", "floor"))
        print("  " + "-" * 92)
        for n in ns:
            r = S[n]
            fl = r["__floor__"]
            def contam(a):
                """Nonzeros of an n-nonzero row that this arm runs outside a full body.

                '!' SINGLE-CHAIN: the arm has a scalar `++p` remainder, so `n mod step`
                    nonzeros ran in ONE accumulator chain. Production's deep kernel has no such
                    loop -- its remainder runs two streams and it static_asserts that it does --
                    so the ratio at this degree measures a loop that does not ship.
                '?' DROPPED WORK: the arm has NO remainder, so `n mod step` nonzeros are not
                    computed at all while the harness still divides by n. That is a work
                    mismatch, not an ILP difference, and it makes the ratio incomparable rather
                    than merely unrepresentative.
                """
                st = TAILS.get(a)
                if st is None:
                    # FAIL CLOSED, like the machine-body lookup above. An arm absent from TAILS
                    # would otherwise read as uncontaminated and print unmarked -- the same
                    # inertness mode as the .upper() defect that produced V4B/V4C. `is None`
                    # rather than falsiness, so a step of 0 is a refusal and not an absence.
                    sys.exit("REFUSE: no source step for arm %r -- probe.cpp parse missed it, so "
                             "the contamination mark would be silently inert for this arm" % a)
                step, has_scalar = st
                left = n % step
                if not left:
                    return ""
                return "!" if has_scalar else "?"

            def rat(a):
                if a not in r or "V0" not in r:
                    return "       -"
                x = r[a] / r["V0"]
                c = contam(a) or contam("V0")
                if c:
                    return "%8.4f%s" % (x, c)
                mark = "" if dev(x) > fl else "~"
                return "%8.4f%s" % (x, mark)
            if "V6" in r and "V7" in r:
                v76 = "%8.4f%s" % (r["V7"] / r["V6"], contam("V7") or contam("V6"))
            else:
                v76 = "       -"
            print("  %5d %10.5f %10.5f %10.5f %9s %9s %9s %8.5f" %
                  (n, r.get("V0", float("nan")), r.get("V6", float("nan")),
                   r.get("V7", float("nan")), rat("V6"), rat("V7"), v76, fl))
        print("  '~' marks a ratio whose deviation from 1 is INSIDE that degree's own A/A floor.")
        for sym, what, why in (
                ("!", "ran part of the row in probe.cpp's ONE-chain scalar remainder",
                 "production's deep kernel has a two-stream remainder and static_asserts it, so\n"
                 "      these ratios measure a loop that does not ship and must not price a guard"),
                ("?", "DID NOT COMPUTE part of the row (no remainder loop) while the harness\n"
                      "      still divided by n",
                 "a work mismatch rather than an ILP difference, so these ratios are\n"
                 "      incomparable, not merely unrepresentative")):
            # EVERY ARM PRESENT IN THE DATA, not the three the table prints. The arms with no
            # remainder loop at all are V4, V4b, V4c and V5, and a scan over ("V0","V6","V7")
            # excludes all four -- so the '?' class would have been structurally unable to report
            # a hit while printing a legend that promised it could. They ARE in this dataset: the
            # sweep timed eleven arms at every degree.
            present = sorted({a for nn in ns for a in S[nn] if not a.startswith("__")})
            # SELF-CHECK THE SCOPE. contam() refuses on an unknown arm, but it is only called for
            # the three arms the table prints; these loops cover all of them and used to `continue`
            # past an arm missing from TAILS. Under a name mismatch -- the .upper() defect produced
            # V4B/V4C where the CSV says V4b/V4c -- `present` would carry the arm, TAILS would not,
            # and the scan would skip it while the scope line below printed a SHORTER class list
            # with nothing saying two arms went unmatched. The statement added to make inertness
            # visible would itself have under-reported. Refuse instead.
            missing = [a for a in present if a not in TAILS]
            if missing:
                sys.exit("REFUSE: %d arm(s) timed but absent from the probe.cpp parse: %s. The "
                         "contamination scan would skip them and the scope line would under-report."
                         % (len(missing), " ".join(missing)))
            hits = []
            for n in ns:
                for a in present:
                    st = TAILS[a]          # refused above if absent, so no shrug here
                    if n % st[0] == 0:
                        continue
                    if ("!" if st[1] else "?") == sym:
                        hits.append("%s n=%d (%d of %d nonzeros)" % (a, n, n % st[0], n))
            if hits:
                print("  '%s' an arm in the ratio %s --" % (sym, what))
                print("      %s." % why)
                print("      %s" % "; ".join(hits))

        # Name the SCOPE the marks were evaluated over, not just the count. A legend that
        # reports zero hits is indistinguishable from a legend that cannot produce one, and the
        # difference here was the arm list: over ("V0","V6","V7") the '?' class is inert by
        # construction, because every no-remainder arm is outside it.
        present = sorted({a for nn in ns for a in S[nn] if not a.startswith("__")})
        missing = [a for a in present if a not in TAILS]
        if missing:
            sys.exit("REFUSE: %d timed arm(s) absent from the probe.cpp parse: %s"
                     % (len(missing), " ".join(missing)))
        noremain = [a for a in present if not TAILS[a][1]]
        print("  marks evaluated over %d arms x %d degrees: %s"
              % (len(present), len(ns), " ".join(present)))
        print("      arms with NO remainder loop (the '?' class): %s"
              % (" ".join(noremain) or "none in this dataset"))
        # The FULL extracted table, not three of thirteen. Printing the whole key set is what
        # caught the .upper() mismatch; a reader cannot apply that check to a list they are not
        # shown, and the three arms the table happens to print are the three least likely to drift.
        print("      source steps, all %d arms parsed from probe.cpp: %s"
              % (len(TAILS), ", ".join("%s=%d%s" % (a, TAILS[a][0], "" if TAILS[a][1] else "*")
                                       for a in sorted(TAILS))))
        print("      (* = no remainder loop; those arms drop `n mod step` nonzeros rather than")
        print("       running them in one chain)")
        bad = [(n, S[n]["__floor__"]) for n in ns
               if not math.isnan(S[n]["__floor__"]) and S[n]["__floor__"] > 0.005]
        if bad:
            print()
            print("  FLOOR QUALITY: %d degree(s) have an A/A floor above 0.5%%: %s"
                  % (len(bad), ", ".join("n=%d %.3f%%" % (n, f * 100) for n, f in bad)))
            print("  The peer simulated n* against noise: at 0.075% the interquartile range is")
            print("  [4.45, 4.95] around a truth of 4.71, at 0.5% it is [2.98, 6.30], and at 2% it")
            print("  CROSSES ZERO. So a degree whose own floor is above 0.5% contributes an")
            print("  interval wider than the answer, and n* below should be read as a direction")
            print("  rather than a number.")
        else:
            print("  FLOOR QUALITY: every degree's A/A floor is at or below 0.5%, so n* is")
            print("  resolvable (the peer's simulation gives an IQR of about +/-0.25 at 0.075%).")

        # ---- THE MODEL-FREE CROSSING. This is now the PRIMARY result and the fit below is
        # reported only so its failure is visible.
        #
        # WHY THE PER-DEGREE RATIO IS CLEAN AND THE CROSS-DEGREE FIT IS NOT. At a fixed n both
        # arms see the same row length, the same index array, the same B, and interleaved
        # repetitions -- so V6/V0 at that n holds everything fixed except the accumulator count.
        # Across n it does not: the number of DISTINCT B rows touched is about min(n, nrowsB), so
        # the working set grows with the degree. V0's own ns/nnz is non-monotone in n for exactly
        # that reason -- it falls to a minimum around n=16-48 and then RISES as B leaves L1 -- and
        # a model of the form a + b*n has no term for it. The fit absorbs the cache curve into the
        # intercept, which is the estimand, and that is fatal rather than inconvenient.
        #
        # So the break-even is read off the ratios directly: the first degree at or above each
        # arm's machine body width where the ratio crosses 1, bracketed by its neighbours. No
        # model, no intercept, no extrapolation.
        print()
        print("  THE MODEL-FREE BREAK-EVEN: the degree at which the ratio crosses 1")
        for cand in ("V6", "V7"):
            w = BW[cand]
            seq = [(n, S[n][cand] / S[n]["V0"], S[n]["__floor__"])
                   for n in ns if cand in S[n] and "V0" in S[n] and n >= w]
            cross = None
            for i in range(1, len(seq)):
                if seq[i - 1][1] > 1.0 >= seq[i][1]:
                    cross = (seq[i - 1], seq[i])
                    break
            if cross is None:
                side = "below 1 at every measured degree >= %d" % w if seq and seq[0][1] <= 1.0 \
                    else "above 1 at every measured degree >= %d" % w
                print("    %s: %s, so the crossing is outside the ladder." % (cand, side))
                continue
            (na, ra, fa), (nb, rb, fb) = cross
            print("    %s: crosses between degree %d (ratio %.4f) and degree %d (ratio %.4f)"
                  % (cand, na, ra, nb, rb))
            print("        linear interpolation in log(n) puts it at degree %.0f, but the bracket"
                  % math.exp(math.log(na) + (math.log(nb) - math.log(na)) * (ra - 1.0) / (ra - rb)))
            print("        [%d, %d] is the honest answer -- the ladder has no point between them."
                  % (na, nb))
            qs = [q for q, v in DLMC_ROWLEN if v < nb]
            print("        DLMC per-ROW quantiles below the upper bracket: %s"
                  % (", ".join("%s=%.0f" % (q, v) for q, v in DLMC_ROWLEN if v < nb) or "none"))
            share = _nnz_share_below(nb)
            if share is None:
                print("        nonzero share below %d: NOT MEASURED (off the an_rowdist ladder),"
                      " so the guard's cost is not priced here." % nb)
            else:
                print("        nonzero share in rows below %d: %.2f%% of the collection -- what a"
                      % (nb, share))
                print("        guard at the bracket's top forgoes. Those rows are not regressions;")
                print("        they run today's kernel.")
            if len(qs) >= 3:
                print("        ABOVE THE MEDIAN ROW: most of the collection's rows are below the"
                      " crossing.")
            elif qs:
                print("        So %s REGRESSES on the rows below the bracket and ships only behind"
                      % cand)
                print("        a row-length guard that provably cannot fire above it.")
            else:
                print("        Below the per-row p10, so %s pays on every decile and needs no"
                      " guard here." % cand)
            best = min(seq, key=lambda x: x[1])
            print("        best measured ratio %.4f at degree %d" % (best[1], best[0]))

        # ---- the registered fit
        print()
        print("  THE REGISTERED FIT  t_call(n) = a + b*n,  overhead cancelling in a difference")
        print("  %5s %13s %13s %9s %11s" % ("arm", "b (ns/nnz)", "a (ns/call)", "R^2", "max resid"))
        print("  " + "-" * 60)
        fits = {}
        for a in ("V0", "V6", "V7"):
            w = BW[a]
            pts = [(n, S[n][a] * n) for n in ns if a in S[n] and n >= w]
            drop = [n for n in ns if a in S[n] and n < w]
            if drop:
                print("  %5s excluding n=%s: below its %d-nonzero machine body, so those degrees"
                      % (a, ",".join(str(x) for x in drop), w))
                print("        run the scalar remainder -- a different kernel, with a different b.")
            if len(pts) < 4:
                print("  %5s  only %d points, a fit is refused" % (a, len(pts)))
                continue
            fits[a] = _lsq(pts)
            b, c, r2, mr = fits[a]
            print("  %5s %13.6f %13.4f %9.5f %11.4f" % (a, b, c, r2, mr))
        print()
        if "V0" in fits and "V6" in fits:
            for cand in ("V6", "V7"):
                if cand not in fits:
                    continue
                db = fits["V0"][0] - fits[cand][0]
                da = fits[cand][1] - fits["V0"][1]
                print("  %s: per-nonzero gain b(V0)-b(%s) = %+.6f ns/nnz" % (cand, cand, db))
                print("      extra per-row cost a(%s)-a(V0) = %+.4f ns  <- call overhead cancels"
                      % (cand, da))
                mr_pair = max(fits["V0"][3], fits[cand][3])
                if abs(da) < mr_pair:
                    print("      FIT REFUSED FOR THIS PURPOSE: |a(%s)-a(V0)| = %.4f ns is SMALLER"
                          % (cand, abs(da)))
                    print("      than the fits' own worst residual, %.4f ns. The intercept"
                          % mr_pair)
                    print("      difference IS the estimand, so a residual larger than it means")
                    print("      the straight line is absorbing structure that belongs to the")
                    print("      estimand -- here the B working set growing with the degree, which")
                    print("      makes V0's ns/nnz non-monotone. R^2 near 1 does NOT rescue this:")
                    print("      t_call is dominated by the b*n term over a 1000x range of n, so")
                    print("      R^2 measures the slope's fit and says nothing about the intercept.")
                    print("      Read the model-free crossing above instead.")
                    print()
                    continue
                if db > 0 and da > 0:
                    nstar = da / db
                    print("      BREAK-EVEN DEGREE n* = %.1f" % nstar)
                    print("      %s" % _verdict(nstar))
                elif db <= 0:
                    print("      no per-nonzero gain to amortise: %s does not pay at ANY degree "
                          "on this core." % cand)
                else:
                    print("      the extra per-row cost is <= 0, so %s pays at every degree and "
                          "needs no degree guard on this core." % cand)
                print()


# TWO DIFFERENT POPULATIONS, and only one of them bounds a per-row guard.
#
# The probe times a kernel on a single row of n nonzeros, and the production gate reads a row
# length, so the crossing bracket is a PER-ROW quantity. It must be placed against the per-row
# length distribution. chain210's `deg` column is nnz/rows, a PER-MATRIX mean, and this file used
# to place the bracket against that -- the wrong population, invisible at the median because on
# DLMC the two happen to agree there to within 1% (102.40 against 103) and wrong by a quarter at
# p25 (25.60 against 32). Both are kept, each under its own name, so the substitution cannot
# recur silently.
#
# Measured by harnesses/an_rowdist.py over all 3846 .smtx files, 3402920 rows, 475663122
# nonzeros, 0 refused. Quantiles are over the 3255080 NONEMPTY rows, because the empty ones never
# reach either kernel (the `pA_begin == pA_end` branch short-circuits them above the dispatch) and
# including them shifts every quantile toward a population the gate is never asked about.
DLMC_ROWLEN = [("p10", 11.0), ("p25", 32.0), ("median", 103.0), ("p75", 208.0), ("p90", 278.0)]
DLMC_MEANDEG = [("p10", 10.24), ("p25", 25.60), ("median", 102.40), ("p75", 204.80),
                ("p90", 307.20)]

# Share of the collection's ARITHMETIC in rows shorter than the threshold -- what a per-row guard
# sends to the fallback and that therefore gets no speedup. Not a regression: those rows run
# exactly today's kernel. This is the number that prices a guard, and it is far smaller than the
# row count because short rows are short.
DLMC_NNZ_BELOW = [(4, 0.03), (8, 0.16), (12, 0.46), (16, 0.80), (24, 1.56),
                  (32, 2.60), (48, 4.57), (64, 7.19), (128, 18.58)]


def _nnz_share_below(n):
    """Measured nonzero share below n, or None when n is off the measured ladder."""
    for t, pct in DLMC_NNZ_BELOW:
        if t == n:
            return pct
    return None



def arm_tails(probe_path):
    """Per-arm SOURCE step and remainder shape, read out of probe.cpp.

    The quantity that decides whether a degree is contaminated is `n mod SOURCE step`, not
    `n < machine body`. The machine body is what -funroll-loops produced; the SOURCE step is what
    decides how many nonzeros are left over, because gcc's leftover handling falls back to a
    rolled copy of the source loop and only then to the source's own scalar remainder. The two
    disagree on this ladder: every arm's body is 8, but V7 steps 8 while V6 steps 4, so V7 runs
    its single-chain remainder at 4, 6 AND 12 where V6 runs one only at 6. A `n < body` test marks
    4 and 6 for every arm and misses V7 at n=12, where a third of the row ran in one chain.

    Returns {ARM: (step, has_scalar_remainder)}. Both are read from the file so neither can drift
    out of agreement with the source the way a hardcoded table would.

    TWO CRITERIA, both legitimate, answering different questions -- do not collapse them:
      n mod SOURCE step != 0   some nonzeros ran outside a full body: in ONE chain if the arm has
                               a scalar remainder, or not at all if it has none. This is what
                               contaminates a ratio as a measurement of the chain count, and it
                               is what the table's '!' and '?' marks use.
      n < MACHINE body         the unrolled body never ran, so the whole degree executed a
                               different code path with different loop overhead per nonzero.
                               This is what the linear fit excludes on, and correctly: a fit of
                               a per-nonzero slope must not mix two code paths.
    They select different sets. On this ladder the first marks V6 at 6 and V7 at 4, 6 and 12;
    the second marks every arm at 4 and 6.
    """
    txt = open(probe_path).read()
    out = {}
    for m in re.finditer(r"NOINLINE\s+\w+\s+(v\d+[a-z]?)\s*\(", txt):
        name = m.group(1)
        # body of the function: from the opening brace to the first line-start '}'
        start = txt.index("{", m.end() - 1)
        end = txt.index("\n}", start)
        body = txt[start:end]
        steps = [int(x) for x in re.findall(r"for\s*\(;\s*p\s*\+\s*(\d+)\s*<=", body)]
        scalar = bool(re.search(r"for\s*\(;\s*p\s*<\s*\w+;\s*\+\+p\s*\)", body))
        if steps:
            # Capitalise the first letter only: v0 -> V0, v0b -> V0b, v4c -> V4c. An .upper()
            # here produced V4B and V4C, which match no row in ARMS[] and would have made the
            # contamination check silently inert for those arms rather than wrong out loud.
            out[name[0].upper() + name[1:]] = (max(steps), scalar)
    return out

def _verdict(nstar):
    """Place the break-even against DLMC's per-ROW length distribution."""
    below = [q for q, v in DLMC_ROWLEN if v < nstar]
    if not below:
        return ("below p10 (11 nonzeros), so the arm pays on every decile of DLMC's rows and "
                "ships without a row-length guard -- on this core type.")
    if len(below) >= 3:
        return ("above the median ROW (103 nonzeros), so the asymptotic win is unreachable for "
                "most of the collection's rows and the lever is effectively closed here.")
    return ("between %s and the next per-row quantile, so the arm REGRESSES on the rows below it "
            "and ships only behind a row-length guard constructed to be inert there." % below[-1])


def _floor(rows):
    a, b = rows.get("V0", []), rows.get("V0b", [])
    m = min(len(a), len(b))
    if m < 5:
        return float("nan")
    d = sorted(dev(a[i] / b[i]) for i in range(m) if b[i] > 0)
    return d[int(0.95 * (len(d) - 1))]


def _lsq(pts):
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    den = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / den
    c = (sy - b * sx) / n
    ybar = sy / n
    sst = sum((p[1] - ybar) ** 2 for p in pts)
    sse = sum((p[1] - (c + b * p[0])) ** 2 for p in pts)
    mr = max(abs(p[1] - (c + b * p[0])) for p in pts)
    return b, c, (1.0 - sse / sst if sst > 0 else float("nan")), mr


if __name__ == "__main__":
    main()
