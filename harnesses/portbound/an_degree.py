#!/usr/bin/env python3
"""Where does V6's asymptotic win stop paying as the row degree falls?

THE QUESTION. The 11-arm probe put V6 (4 accumulator chains) at -21.259% on the P-core and
-4.032% on the E-core over a single row of 4096 nonzeros. chain210's own `deg` column over the
3820 DLMC keys reads p10 10.24, p25 25.60, median 102.40, p75 204.80, p90 307.20 -- so that row
is 40x the median production row. V6 buys a per-NONZERO gain with a per-ROW cost: four chains
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
    for f in sorted(glob.glob(os.path.join(D, "pbdeg_n*_core*.csv"))):
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
            def rat(a):
                if a not in r or "V0" not in r:
                    return "       -"
                x = r[a] / r["V0"]
                mark = "" if dev(x) > fl else "~"
                return "%8.4f%s" % (x, mark)
            v76 = "%8.4f" % (r["V7"] / r["V6"]) if ("V6" in r and "V7" in r) else "       -"
            print("  %5d %10.5f %10.5f %10.5f %9s %9s %9s %8.5f" %
                  (n, r.get("V0", float("nan")), r.get("V6", float("nan")),
                   r.get("V7", float("nan")), rat("V6"), rat("V7"), v76, fl))
        print("  '~' marks a ratio whose deviation from 1 is INSIDE that degree's own A/A floor.")

        # ---- the registered fit
        print()
        print("  THE REGISTERED FIT  t_call(n) = a + b*n,  overhead cancelling in a difference")
        print("  %5s %13s %13s %9s %11s" % ("arm", "b (ns/nnz)", "a (ns/call)", "R^2", "max resid"))
        print("  " + "-" * 60)
        fits = {}
        for a in ("V0", "V6", "V7"):
            pts = [(n, S[n][a] * n) for n in ns if a in S[n]]
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


DLMC = [("p10", 10.24), ("p25", 25.60), ("median", 102.40), ("p75", 204.80), ("p90", 307.20)]


def _verdict(nstar):
    """Place the break-even against chain210's own DLMC degree distribution."""
    below = [q for q, v in DLMC if v < nstar]
    if not below:
        return ("below p10 (10.24), so V6 pays on every decile of the DLMC corpus and ships "
                "without a degree guard -- on this core type.")
    if len(below) >= 3:
        return ("above the DLMC median (102.40), so the asymptotic win is unreachable for most "
                "of the corpus and the lever is effectively closed here.")
    return ("between %s and the next DLMC quantile, so V6 REGRESSES on the keys below it and "
            "ships only behind a degree guard constructed to be inert there." % below[-1])


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
