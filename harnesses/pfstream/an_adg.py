#!/usr/bin/env python3
"""Read the ADOPT_GRAIN three-build CSVs: does narrowing the adopted team beat MKL, and where
does it regress?

TWO FLOORS, BOUNDING DIFFERENT THINGS, AND THEY ARE NOT INTERCHANGEABLE.
  oursb/ours within one CSV holds the OBJECT fixed and bounds timing only.
  adgb/adg0 across CSVs varies the object at one configuration and bounds placement.
The arm is a separate compile, so the arm-vs-floor comparison must use the PLACEMENT floor. A
reading scored against the timing floor alone would make almost every column look significant.

The harm criterion is PER-CELL, not a median. A pooled median cannot see a few per cent of cells
at a ten per cent threshold -- it is the wrong instrument, silent rather than weak.
"""
import argparse, csv, glob, math, os, statistics as st, collections

ap = argparse.ArgumentParser()
ap.add_argument("--csvdir", required=True)
ap.add_argument("--harm", type=float, default=0.10, help="per-cell slowdown counted as harm")
A = ap.parse_args()

cells = collections.defaultdict(dict)   # (build,key,k) -> {arm: ms}
meta = {}
nt = {}
backends = set()
for p in sorted(glob.glob(os.path.join(A.csvdir, "*.csv"))):
    for r in csv.DictReader(open(p)):
        key, k, b = r["key"], int(r["k"]), r["build"]
        cells[(b, key, k)][r["arm"]] = float(r["ms"])
        backends.add(r.get("refbackend", "unrecorded"))
        meta[key] = (int(r["rows"]), int(r["nnz"]), float(r["deg"]))
        nt[(b, key, k)] = int(r["nthreads"])
builds = sorted({b for (b, _, _) in cells})
ks = sorted({k for (_, _, k) in cells})
keys = sorted({key for (_, key, _) in cells})
print("builds %s   %d widths   %d matrices   %d cells" % (builds, len(ks), len(keys), len(cells)))
if not set(builds) >= {"adg0", "adgb", "adgG"}:
    raise SystemExit("REFUSE: need adg0, adgb and adgG; got %s. Without the floor PAIR the arm "
                     "has nothing to clear." % builds)

gm = lambda v: math.exp(st.median([math.log(x) for x in v])) if v else float("nan")


def ratio(b1, b2, k, arm="ours"):
    out = []
    for key in keys:
        c1, c2 = cells.get((b1, key, k), {}), cells.get((b2, key, k), {})
        if arm in c1 and arm in c2:
            out.append(c1[arm] / c2[arm])
    return out


print()
print("THE PLACEMENT FLOOR, one configuration built twice. Every arm reading must clear this.")
print("%4s %7s %9s %9s %9s" % ("k", "n", "adgb/adg0", "med|dev|%", "p95|dev|%"))
floor_p95 = {}
for k in ks:
    v = ratio("adgb", "adg0", k)
    dev = sorted(max(x, 1.0 / x) - 1.0 for x in v)
    p95 = dev[min(len(dev) - 1, int(0.95 * len(dev)))] * 100
    floor_p95[k] = p95
    print("%4d %7d %9.4f %9.4f %9.4f" % (k, len(v), gm(v), st.median(dev) * 100, p95))

print()
print("THE TIMING FLOOR for contrast, oursb/ours inside one object -- bounds a DIFFERENT thing")
for k in ks:
    v = [c["oursb"] / c["ours"] for (b, key, kk), c in cells.items()
         if kk == k and b == "adg0" and "oursb" in c and "ours" in c]
    dev = sorted(max(x, 1.0 / x) - 1.0 for x in v)
    print("     k=%-3d n=%-4d %.4f  p95 dev %.4f%%" % (k, len(v), gm(v),
          dev[min(len(dev) - 1, int(0.95 * len(dev)))] * 100))

print()
print("THE ARM: adgG/adg0 on our own kernel. Below 1.0 = the narrower adopted team is FASTER.")
print("%4s %7s %10s %12s %10s   %s" % ("k", "n", "adgG/adg0", "vs floor p95", "n_changed",
                                        "verdict"))
for k in ks:
    v = ratio("adgG", "adg0", k)
    r = gm(v)
    changed = sum(1 for key in keys
                  if nt.get(("adgG", key, k)) != nt.get(("adg0", key, k)))
    eff = (max(r, 1 / r) - 1) * 100
    verdict = "readable" if eff > floor_p95[k] else "INSIDE THE FLOOR"
    print("%4d %7d %10.4f %12.2f%% %10d   %s (effect %.2f%% vs floor %.2f%%)"
          % (k, len(v), r, floor_p95[k], changed, verdict, eff, floor_p95[k]))

print()
print("CONDITIONED ON WHETHER THE OBJECTS ACTUALLY RESOLVED DIFFERENT COUNTS.")
print("The cells where the count is identical are a NEGATIVE CONTROL: the two objects differ by")
print("a constant that cannot have been read, so a reading there is instrument, not effect.")
print("%4s %9s %10s %9s %10s" % ("k", "n changed", "adgG/adg0", "n same", "adgG/adg0"))
for k in ks:
    ch, sa = [], []
    for key in keys:
        c1, c0 = cells.get(("adgG", key, k), {}), cells.get(("adg0", key, k), {})
        if "ours" not in c1 or "ours" not in c0:
            continue
        (ch if nt.get(("adgG", key, k)) != nt.get(("adg0", key, k)) else sa).append(
            c1["ours"] / c0["ours"])
    print("%4d %9d %10.4f %9d %10.4f" % (k, len(ch), gm(ch) if ch else float("nan"),
                                          len(sa), gm(sa) if sa else float("nan")))

print()
# The reference arm is labelled by the backend the CSV records, not by the column's old name. A
# host without MKL dispatches torch.sparse.mm to PyTorch's native CSR path, which reads an order
# of magnitude slower than our kernel -- presenting that as "ours/mkl" would be a fabricated win.
BK = ", ".join(sorted(backends))
if backends == {"mkl"}:
    print("AGAINST MKL, which is the actual question. Below 1.0 = we are faster.")
else:
    print("AGAINST THE REFERENCE ARM, backend = %s. THIS IS NOT AN MKL COMPARISON." % BK)
    print("  torch.sparse.mm without MKL runs PyTorch's native CSR path. The ratios below are")
    print("  real but they do not bear on the MKL question, and no MKL claim may cite them.")
print("%4s %10s %10s %10s   %8s %8s" % ("k", "adg0/ref", "adgG/ref", "improvement",
                                         "ahead_0", "ahead_G"))
for k in ks:
    r0, rG = [], []
    for key in keys:
        c0, cG = cells.get(("adg0", key, k), {}), cells.get(("adgG", key, k), {})
        rc = "ref" if "ref" in c0 or "ref" in cG else "mkl"
        if rc in c0 and "ours" in c0:
            r0.append(c0["ours"] / c0[rc])
        if rc in cG and "ours" in cG:
            rG.append(cG["ours"] / cG[rc])
    a0 = 100.0 * sum(1 for x in r0 if x < 1) / max(len(r0), 1)
    aG = 100.0 * sum(1 for x in rG if x < 1) / max(len(rG), 1)
    print("%4d %10.4f %10.4f %10.4f   %7.1f%% %7.1f%%"
          % (k, gm(r0), gm(rG), gm(r0) / gm(rG), a0, aG))

print()
print("THE HARM CRITERION, per cell, against the PLACEMENT floor's own A/A gate.")
print("A cell counts as harmed when adgG is more than %.0f%% slower than adg0 AND the floor pair"
      % (A.harm * 100))
print("on that same cell agrees within %.0f%%." % (A.harm * 100 / 2))
print("%4s %8s %8s %9s %9s" % ("k", "gated", "harmed", "share", "helped>10%"))
for k in ks:
    gated = harmed = helped = 0
    for key in keys:
        cG, c0, cb = (cells.get((b, key, k), {}) for b in ("adgG", "adg0", "adgb"))
        if not all("ours" in c for c in (cG, c0, cb)):
            continue
        aa = abs(cb["ours"] / c0["ours"] - 1.0)
        if aa > A.harm / 2:
            continue
        gated += 1
        rr = cG["ours"] / c0["ours"]
        if rr > 1 + A.harm:
            harmed += 1
        if rr < 1 - A.harm:
            helped += 1
    print("%4d %8d %8d %8.2f%% %9d" % (k, gated, harmed,
          100.0 * harmed / max(gated, 1), helped))
