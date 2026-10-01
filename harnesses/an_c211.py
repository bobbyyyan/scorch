#!/usr/bin/env python3
"""chain211: does the four-chain deep SpMM kernel beat the kernel it would replace, on all of DLMC?

Takes the tree root as argv[1] and REFUSES without it, so no ledger path is hardcoded.

THE DESIGN. Three objects over six passes, all six permutations so each object sits in each
position exactly twice (an unbalanced rotation reads the later object systematically differently).
  base   SCORCH_NARROWK_DEEP_UNROLL=0, the kernel that ships
  bsb_   BYTE-IDENTICAL copy of base at an equal-length path: zero code difference, two process
         boundaries. This is the cross-object A/A floor and it is not optional -- the
         two-objects-same-decisions artefact is 1-2.6% on narrow-k SpMM, largest at the narrowest
         width, which is the same order as the effect under test.
  deep   SCORCH_NARROWK_DEEP_UNROLL=4
Each file holds three ARMS (ours, oursb, mkl) inside one build: `oursb` is `ours` again, so it is
the within-process floor, and `mkl` needs no build of ours at all.

EVERY ESTIMATOR AND FLOOR WAS FIXED BEFORE THE DATA. See harnesses/portbound/
PREREG_c211_residual.md, amendments 1-3. In particular:
  * `degree` is exactly nnz/rows, so {log nnz, log rows, log degree} has RANK TWO and must never be
    fitted together. The check below refuses if its own positive control stops working.
  * at a fixed width B's footprint is J up to a constant, and log J correlates +0.41 with log nnz,
    so omitting J makes the footprint and per-row hypotheses load one coefficient with one sign.
  * the footprint hypothesis is a THRESHOLD at an L1 boundary, not a slope: the only cache boundary
    this board crosses is L1 (largest B is 576 KB against a 2 MB P-core L2), and at k=8 a log J
    slope would spend most of its leverage inside L1 where the mechanism does not vary.
"""
import collections, csv, glob, math, os, statistics, sys
import numpy as np

if len(sys.argv) < 2:
    sys.exit("usage: an_c211.py <tree-root> [dims.csv]\n"
             "REFUSING rather than defaulting: a hardcoded ledger path is a defect caught three "
             "times in this project.")
ROOT = sys.argv[1]
DIMS = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
    "~/.cache/scorch-codex/verifier-dlmc-dims/dlmc_dims.csv")

OBJECTS = ("base", "bsb_", "deep")
PASSES = range(1, 7)
KS = (2, 4, 8, 16, 32)
EXPECT_KEYS, EXPECT_ROWS = 3820, 57300

red = lambda v: math.exp(statistics.median([math.log(x) for x in v]))
dev = lambda x: math.exp(abs(math.log(x))) - 1.0
p95 = lambda d: sorted(d)[int(0.95 * (len(d) - 1))] if d else float("nan")

# ---------------------------------------------------------------- load, with our own refusals
# The stage script already refuses a short file, but that protects the RUN. An analyser that
# trusts the producer's check is a harness restating production, which is wrong silently.
files = {}
for ob in OBJECTS:
    for p in PASSES:
        f = os.path.join(ROOT, "dl_%s_float32_p%d.csv" % (ob, p))
        if not os.path.exists(f):
            sys.exit("REFUSE: missing object-pass %s p%d (%s)" % (ob, p, f))
        n = sum(1 for _ in open(f)) - 1
        if n != EXPECT_ROWS:
            sys.exit("REFUSE: %s has %d data rows, expected %d (3820 keys x 5 widths x 3 arms)"
                     % (os.path.basename(f), n, EXPECT_ROWS))
        files[(ob, p)] = f
extra = set(glob.glob(os.path.join(ROOT, "dl_*_float32_p*.csv"))) - set(files.values())
if extra:
    sys.exit("REFUSE: %d CSV(s) present beyond the 18 declared: %s"
             % (len(extra), " ".join(sorted(os.path.basename(x) for x in extra))))

# A DUPLICATED PASS PASSES EVERY CHECK ABOVE and biases the floor DOWNWARD, which manufactures
# findings. A byte-copy of p2 as p3 has the right row count, build label and key set; then
# ratio()'s six-pass requirement is met by five independent observations plus a repeat, and the
# p95 of dev(bsb_/base) tightens because the duplicated pass contributes zero variance to the pair
# it appears in. A tighter floor makes more cells read as beyond-floor. Only provenance separates
# a run from a duplicate, so digest all 18 and print them rather than asserting the check ran.
import hashlib
digs = {}
for key, f in sorted(files.items()):
    digs[key] = hashlib.sha256(open(f, "rb").read()).hexdigest()
by_dig = collections.defaultdict(list)
for key, d in digs.items():
    by_dig[d].append(key)
dupes = {d: v for d, v in by_dig.items() if len(v) > 1}
if dupes:
    for d, v in dupes.items():
        print("  DUPLICATE %s : %s" % (d[:16], " ".join("%s_p%d" % kv for kv in sorted(v))))
    sys.exit("REFUSE: %d set(s) of byte-identical object-passes. A duplicated pass biases the "
             "cross-object floor low and manufactures beyond-floor cells." % len(dupes))
print("LOADED 18 object-passes, each exactly %d data rows; all 18 digests distinct" % EXPECT_ROWS)
print("  " + "  ".join("%s_p%d=%s" % (o, p, digs[(o, p)][:8])
                       for o in OBJECTS for p in PASSES))

T = collections.defaultdict(dict)        # (obj,key,k) -> pass -> arm -> ms
META = {}                                # key -> (rows, nnz)
ARMS = ("ours", "oursb", "mkl")
# COUNT, INDEX AND CONTENT ARE INDEPENDENT AXES. The row count alone is satisfied by a file with a
# k=7 row, or with "ours" twice for one cell and "oursb" absent -- those rows load and are then
# silently never read back, because every use of KS and ARMS downstream is iteration and not a
# membership test. So validate the (key, k, arm) multiset per file: exactly EXPECT_ROWS entries,
# no duplicate, every k in KS, and the arm set EQUAL to the three (bidirectional -- a missing arm
# and an extra arm are different defects and both must refuse).
for (ob, p), f in sorted(files.items()):
    seen = set()
    nrow = 0
    ks_here, arms_here = set(), set()
    for r in csv.DictReader(open(f)):
        if r["build"] != ob:
            sys.exit("REFUSE: %s contains build label %r; a mislabelled file would silently "
                     "cross-assign arms" % (os.path.basename(f), r["build"]))
        key, k, arm = r["key"], int(r["k"]), r["arm"]
        nrow += 1
        if (key, k, arm) in seen:
            sys.exit("REFUSE: %s repeats cell (%s, k=%d, %s); a repeated arm overwrites its own "
                     "reading and the row count cannot see it"
                     % (os.path.basename(f), key, k, arm))
        seen.add((key, k, arm))
        ks_here.add(k); arms_here.add(arm)
        rn = (int(r["rows"]), int(r["nnz"]))
        if key in META:
            if META[key] != rn:
                sys.exit("REFUSE: %s gives %s rows/nnz %s against %s seen earlier; a staging "
                         "error confined to one object would otherwise validate against whichever "
                         "file was read first" % (os.path.basename(f), key, rn, META[key]))
        else:
            META[key] = rn
        try:
            ms = float(r["ms"])
        except (TypeError, ValueError):
            continue
        if not (0 < ms < float("inf")):
            continue
        T[(ob, key, k)].setdefault(p, {})[arm] = ms
    if nrow != EXPECT_ROWS:
        sys.exit("REFUSE: %s parsed %d rows, expected %d" % (os.path.basename(f), nrow, EXPECT_ROWS))
    bad_k = ks_here - set(KS)
    if bad_k:
        sys.exit("REFUSE: %s contains width(s) %s outside the declared %s; those rows load and "
                 "are then never read" % (os.path.basename(f), sorted(bad_k), list(KS)))
    if ks_here != set(KS):
        sys.exit("REFUSE: %s is missing width(s) %s" % (os.path.basename(f), sorted(set(KS) - ks_here)))
    if arms_here != set(ARMS):
        sys.exit("REFUSE: %s arm set is %s, expected exactly %s"
                 % (os.path.basename(f), sorted(arms_here), sorted(ARMS)))
keys = sorted(META)
if len(keys) != EXPECT_KEYS:
    sys.exit("REFUSE: %d distinct keys, expected %d" % (len(keys), EXPECT_KEYS))
print("VALIDATED per file: %d unique (key,k,arm) cells, widths == %s, arms == %s, rows/nnz "
      "consistent across all 18" % (EXPECT_ROWS, list(KS), sorted(ARMS)))

# ---------------------------------------------------------------- join J, refusing on any gap
J = {}
with open(DIMS) as h:
    rdr = csv.reader(l for l in h if not l.startswith("#"))
    ix = {n: i for i, n in enumerate(next(rdr))}
    for row in rdr:
        kk = row[ix["key"]]
        v = (int(row[ix["rows"]]), int(row[ix["cols"]]), int(row[ix["nnz"]]))
        # A REPEATED KEY IN THE DIMS FILE IS THE ONE THING NOT CHECKED ANYWHERE. Assignment into
        # a dict silently keeps the LAST occurrence, so a file carrying a key twice with different
        # cols would join a J value that no line of the file announces as authoritative, and every
        # downstream coefficient would be computed from it. The board's own key count is pinned
        # separately, so the dims file's SIZE needs no assertion -- a larger dims file covering
        # more of the collection joins correctly and is harmless -- but a contradictory duplicate
        # inside it is not harmless and is invisible.
        if kk in J and J[kk] != v:
            sys.exit("REFUSE: dims file gives %s two different rows/cols/nnz, %s and %s; a dict "
                     "assignment would silently keep the last" % (kk, J[kk], v))
        J[kk] = v
miss = [k for k in keys if k not in J]
if miss:
    sys.exit("REFUSE: %d key(s) absent from the dims join; a partial join is invisible in every "
             "coefficient it produces: %s" % (len(miss), " ".join(miss[:5])))
bad = [k for k in keys if (J[k][0], J[k][2]) != META[k]]
if bad:
    sys.exit("REFUSE: %d key(s) disagree with the board on rows/nnz -- the dims file is not this "
             "collection: %s" % (len(bad), " ".join(bad[:5])))
print("JOINED J for all %d keys; rows and nnz agree with the board on every one" % len(keys))

# ---------------------------------------------------------------- per-cell ratios and floors
def ratio(o_num, a_num, o_den, a_den, key, k):
    """exp(median(log r)) over the passes where BOTH sides are present, requiring all six."""
    rs = []
    for p in PASSES:
        a = T[(o_num, key, k)].get(p, {})
        b = T[(o_den, key, k)].get(p, {})
        if a_num in a and a_den in b and b[a_den] > 0:
            rs.append(a[a_num] / b[a_den])
    return red(rs) if len(rs) == 6 else None

print()
print("FLOORS, per width. The cross-object A/A (bsb_/base, byte-identical code in two processes)")
print("is the floor the effect must clear, because deep-vs-base crosses a process AND an object.")
print("  %-34s %s   (value/n)" % ("", "   ".join("k=%-11d" % k for k in KS)))
FLOOR, ARMFLOOR = {}, {}
for lbl, fn, store in (("within-process  oursb/ours (base)",
                        lambda key, k: ratio("base", "oursb", "base", "ours", key, k), ARMFLOOR),
                       ("cross-object    bsb_/base  (ours)",
                        lambda key, k: ratio("bsb_", "ours", "base", "ours", key, k), FLOOR)):
    out = []
    for k in KS:
        d = [dev(r) for key in keys if (r := fn(key, k)) is not None]
        store[k] = p95(d)
        # Print the floor's OWN n. The effect table carries an n column and this one did not, so
        # the two halves of every "x floor" figure were documented to different standards: a p95
        # over a handful of surviving cells looks identical to a p95 over 3820.
        out.append("%.4f/%d" % (store[k], len(d)))
    print("  %-34s %s" % (lbl, "   ".join("%-13s" % x for x in out)))

print()
print("THE EFFECT: deep/base on our own kernel, per width, against the cross-object floor.")
print("  %-4s %10s %10s %9s %8s %8s %s" % ("k", "deep/base", "floor", "x floor", "n", "NVEC", "verdict"))
EFF = {}
for k in KS:
    vals = {key: r for key in keys if (r := ratio("deep", "ours", "base", "ours", key, k)) is not None}
    EFF[k] = vals
    m = red(list(vals.values()))
    fl = FLOOR[k]
    nvec = math.ceil(k / 8)
    v = "inside floor" if dev(m) <= fl else ("SLOWER" if m > 1 else "faster")
    print("  %-4d %10.4f %10.4f %9.1f %8d %8d %s"
          % (k, m, fl, dev(m) / fl if fl else float("nan"), len(vals), nvec, v))

print()
print("THE BOARD: ours/mkl per object, per width. Below 1 means Scorch is faster.")
print("  %-6s %s" % ("object", "   ".join("k=%-6d" % k for k in KS)))
for ob in OBJECTS:
    cells = []
    for k in KS:
        v = [r for key in keys if (r := ratio(ob, "ours", ob, "mkl", key, k)) is not None]
        cells.append("%.4f" % red(v) if v else "-")
    print("  %-6s %s" % (ob, "   ".join("%-8s" % c for c in cells)))

# ---------------------------------------------------------------- the registered fits
print()
print("=" * 100)
print("REGISTERED ANALYSIS of the deep/base effect. y = log(deep/base) per key.")
lnv = {k: np.log(np.array([J[k][2] for k in keys], float)) for k in (0,)}  # placeholder, see below
ln_all = np.log(np.array([J[key][2] for key in keys], float))
lr_all = np.log(np.array([J[key][0] for key in keys], float))
lc_all = np.log(np.array([J[key][1] for key in keys], float))

# P0's positive control, run on THIS data rather than assumed from the prereg.
X2 = np.column_stack([ln_all, lr_all, np.ones_like(ln_all)])
def collin(y):
    b, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
    r = y - X2 @ b
    rss = float(r @ r); tss = float(((y - y.mean()) ** 2).sum())
    return rss, tss, (rss < 1e-18 * max(tss, 1.0))
rj, tj, zj = collin(lc_all)
rd, td, zd = collin(ln_all - lr_all)
print("P0 collinearity, with its positive control:")
print("  log J      residual SS %10.1f  total SS %9.1f  VIF %6.3f" % (rj, tj, tj / rj))
print("  log degree %s  (MUST read collinear: degree is nnz/rows)"
      % ("~0 (< 1e-18 x total SS)" if zd else "residual SS %.3g -- NOT collinear" % rd))
if zj:
    sys.exit("REFUSE: log J reads collinear here; the repair separating the per-row and footprint "
             "families is unavailable on this data")
if not zd:
    sys.exit("REFUSE: log degree did not read collinear, but it is nnz/rows by construction. The "
             "check is broken and its verdict on log J carries no warrant.")

def ols(y, cols, names):
    X = np.column_stack(cols + [np.ones_like(y)])
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    s2 = float(resid @ resid) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * s2)
    return list(zip(names, b[:len(names)], se[:len(names)]))

STRAT = [key for key in keys if J[key][0] == 512]
print()
print("PRIMARY: stratified on rows==512 (%d keys, %.1f%% of the board), rows constant by"
      % (len(STRAT), 100 * len(STRAT) / len(keys)))
print("construction so the J slope reads directly with nothing partialled. Its J levels are")
sj = collections.Counter(J[key][1] for key in STRAT)
print("  " + "  ".join("%d:%d" % (j, sj[j]) for j in sorted(sj))
      + "  -- effectively a 512-vs-2048 contrast with small wings; a real slope, few levels.")
print()
print("  %-4s %22s %22s %10s" % ("k", "slope(log J) [strat]", "threshold J*k>=12288", "n"))
for k in KS:
    sub = [key for key in STRAT if key in EFF[k]]
    if len(sub) < 50:
        print("  %-4d %22s %22s %10d" % (k, "n too small", "-", len(sub))); continue
    y = np.log(np.array([EFF[k][key] for key in sub]))
    lj = np.log(np.array([J[key][1] for key in sub], float))
    nm, b, se = ols(y, [lj], ["log J"])[0]
    hi = np.array([J[key][1] * k >= 12288 for key in sub])
    d = (float(np.exp(y[hi].mean()) - np.exp(y[~hi].mean()))
         if hi.any() and (~hi).any() else float("nan"))
    print("  %-4d %10.4f +/- %7.4f %14.4f  (n_hi=%4d) %10d"
          % (k, b, se, d, int(hi.sum()), len(sub)))

print()
print("ROBUSTNESS: unstratified y ~ log(nnz) + log(rows) + log(J), rank 4 as checked above.")
print("  per-row family predicts slope(log J)=0; footprint predicts <0; a flat penalty predicts all 0.")
print("  %-4s %18s %18s %18s" % ("k", "slope(log nnz)", "slope(log rows)", "slope(log J)"))
for k in KS:
    sub = [key for key in keys if key in EFF[k]]
    y = np.log(np.array([EFF[k][key] for key in sub]))
    cols = [np.log(np.array([J[key][2] for key in sub], float)),
            np.log(np.array([J[key][0] for key in sub], float)),
            np.log(np.array([J[key][1] for key in sub], float))]
    out = ols(y, cols, ["log nnz", "log rows", "log J"])
    print("  %-4d %s" % (k, " ".join("%9.4f+/-%-7.4f" % (b, se) for _, b, se in out)))
