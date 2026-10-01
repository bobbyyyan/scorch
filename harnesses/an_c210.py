#!/usr/bin/env python3
"""chain210: does chain207's DLMC behind-set replicate across RUNS, with the objects held fixed?

PRE-REGISTERED. This file is committed before chain210 has produced a single complete object-pass,
so every choice below -- estimand, floor, population, the verdict partition and the three timing
hypotheses -- is fixed in advance of the data. If it is amended, the amendment says so and records
how many object-passes existed at the time.

THE QUESTION. chain207 used a replication PAIR (identical .text, distinct object files) to defeat
the one-object churn its own stage script documents: over SuiteSparse the same three objects gave
behind counts of 79/62/74, union 139, only 22 in all three. The pair fixed that. It does not fix
run-to-run churn: chain206's aa/bb gives 14 residues and chain196's c0/c0b gives 18 over the SAME
4584 matrices, union 25 and intersection 7, while the two pairs agree on the pooled k=4 floor to a
ratio of 1.0026 and on WHICH MATRIX is unstable at only Pearson 0.282. chain210 is a second run of
chain207's grid with byte-identical objects, which isolates the run with machine code held fixed --
a cleaner contrast than chain206-against-chain196, where objects and run both varied.

ESTIMAND, identical to an_c207.py so the two runs are comparable by construction:
  S = exp(median over passes of log(ours_p / mkl_p)) per cell, PAIRED at the pass level. Ours and
  MKL are timed in the same pass, so the pairing is what cancels the run-wide component; an
  unpaired median(ours)/median(mkl) is a different and looser statistic (it moved a k=4 gradient
  from 21.43% to 20.11% and inflated a band deviation from 6.57% to 7.77%).
  dev(x) = exp|ln x| - 1. Floor = p95 by nearest rank, per width, of dev(S_aa / S_bb).
  A cell is BEHIND MKL iff S > 1 on BOTH builds and dev(S) > floor on BOTH.

POPULATION. 3820 keys x 5 widths = 19100 cells. The 3820 is not a magic number: chain209's sealed
dimension table holds 3846 DLMC matrices and exactly 26 have nnz == 0 (all in
transformer/variational_dropout/0.98, pruned to nothing), so {nnz > 0} equals the timed set exactly.
chain210 drops the same 26 for the same deterministic reason, which is why both runs are 19100.
"""
import collections, csv, glob, hashlib, math, os, re, statistics, sys

C210 = sys.argv[1] if len(sys.argv) > 1 else "."
C207 = os.path.expanduser("~/.cache/scorch-codex/spmm-beat-mkl/chain207-dlmc-redwood")
DIMS = os.path.expanduser("~/.cache/scorch-codex/spmm-beat-mkl/chain209-dlmc-dims/dlmc_dims.csv")
KS, PS, BUILDS = (2, 4, 8, 16, 32), (1, 2, 3, 4, 5, 6), ("aa", "bb")
EXPECT_FILES, EXPECT_ROWS, EXPECT_CELLS = 12, 57300, 19100
DIMS_SHA = "ff167f71d50abf74a07c054d042f2e7de6b7b0eb32f03a871b2e298f50bb1ee5"
C207_TEXT, C207_WA, C207_WB = "a79b5d20b322c5ac", "7f8f78d58a8784b1", "70c9f5473e498f50"

red = lambda v: math.exp(statistics.median([math.log(x) for x in v]))
dev = lambda x: math.exp(abs(math.log(x))) - 1.0
p95 = lambda d: sorted(d)[int(0.95 * (len(d) - 1))] if d else float("nan")

def load(d, pat):
    """-> MS[(build, cell, arm)][pass] = ms, plus the per-(build,pass) row counts."""
    MS, seen, NNZ = collections.defaultdict(dict), collections.Counter(), {}
    files = sorted(glob.glob(os.path.join(d, pat)))
    for f in files:
        b = os.path.basename(f).split("_")[1]
        p = int(f.rsplit("_p", 1)[1].split(".")[0])
        for r in csv.DictReader(open(f)):
            try: ms = float(r["ms"])
            except (TypeError, ValueError): continue
            if not (0 < ms < float("inf")): continue
            MS[(b, (r["key"], int(r["k"])), r["arm"])][p] = ms
            NNZ[(r["key"], int(r["k"]))] = int(r["nnz"])
            seen[(b, p)] += 1
    return files, MS, seen, NNZ

def board(MS):
    cells = sorted({c for (b, c, a) in MS if b == BUILDS[0]})
    def S(b, c):
        o, m = MS.get((b, c, "ours"), {}), MS.get((b, c, "mkl"), {})
        ps = [p for p in PS if p in o and p in m and m[p] > 0]
        return red([o[p] / m[p] for p in ps]) if len(ps) == 6 else None
    Sa = {c: S("aa", c) for c in cells}
    Sb = {c: S("bb", c) for c in cells}
    ok = [c for c in cells if Sa[c] and Sb[c]]
    FR = {k: p95([dev(Sa[c] / Sb[c]) for c in ok if c[1] == k]) for k in KS}
    behind = {c for c in ok if Sa[c] > 1 and Sb[c] > 1
              and dev(Sa[c]) > FR[c[1]] and dev(Sb[c]) > FR[c[1]]}
    inst = {c: dev(Sa[c] / Sb[c]) for c in ok}
    return Sa, Sb, ok, FR, behind, inst

# ---------------------------------------------------------------- GUARDS
# Each asserts on the quantity it is about, and each has an input that fails it. A guard that can
# only pass is the dominant failure mode in this milestone: a table that could print only 0, a CRC
# manifest under `shasum -c`, a glob that ate its corpus. REFUSE rather than report on partial data
# -- chain207's pass 1 is the FASTEST of its twelve (22m50s against a 23m32.5s median), so a partial
# run is biased fast AND position-1-heavy, since the rotation balances only over all twelve.
print("GUARDS")
files, MS, seen, NNZ = load(C210, "dl_*_float32_p*.csv")
if len(files) != EXPECT_FILES:
    sys.exit("  REFUSE: %d input CSVs, expected %d. chain210 is incomplete; a partial run is "
             "biased fast and position-1-heavy and must not be reduced." % (len(files), EXPECT_FILES))
idx = {b: sorted({p for (bb, p) in seen if bb == b}) for b in BUILDS}
assert idx["aa"] == idx["bb"] == list(PS), "  REFUSE: pass-index sets differ or are not 1..6: %s" % idx
bad = {k: v for k, v in seen.items() if v != EXPECT_ROWS}
if bad: sys.exit("  REFUSE: object-passes with wrong row count (expected %d): %s" % (EXPECT_ROWS, bad))
print("  G1  %d CSVs, both builds carry passes %s, every object-pass exactly %d rows"
      % (len(files), idx["aa"], EXPECT_ROWS))

# G2: the dimension table, by digest, and the population predicate derived from it rather than
# assumed. 19100 must come from {nnz > 0}, not from a literal.
got = hashlib.sha256(open(DIMS, "rb").read()).hexdigest()
assert got == DIMS_SHA, "  REFUSE: dlmc_dims.csv digest %s != %s" % (got[:16], DIMS_SHA[:16])
DIM = {r["key"]: (int(r["rows"]), int(r["cols"]), int(r["nnz"])) for r in csv.DictReader(open(DIMS))}
pos = {k for k, v in DIM.items() if v[2] > 0}
keys210 = {c[0] for c in NNZ}
assert pos == keys210, ("  REFUSE: {nnz>0} != timed set (|nnz>0|=%d |timed|=%d sym-diff=%d)"
                        % (len(pos), len(keys210), len(pos ^ keys210)))
assert len(NNZ) == EXPECT_CELLS, "  REFUSE: %d cells, expected %d" % (len(NNZ), EXPECT_CELLS)
print("  G2  dims table binds by digest; {nnz>0} == timed set exactly (%d keys); %d cells"
      % (len(pos), len(NNZ)))

# G3: bind the OBJECTS. chain207's premise check accepts ANY replication pair; this asserts it was
# chain207's pair. 64-bit fields, which is what the stage log records -- not a full SHA-256.
log = open(os.path.join(C210, "c210_stage.log")).read()
tx = re.findall(r"^\s+(aa|bb)\s+\.text\s+([0-9a-f]{16})\s+whole\s+([0-9a-f]{16})", log, re.M)
assert len(tx) == 2, "  REFUSE: stage log does not record two object digests: %r" % (tx,)
dd = {b: (t, w) for b, t, w in tx}
assert dd["aa"][0] == dd["bb"][0] == C207_TEXT, "  REFUSE: .text is not chain207's: %r" % (dd,)
assert dd["aa"][1] == C207_WA and dd["bb"][1] == C207_WB, "  REFUSE: whole-file digests: %r" % (dd,)
print("  G3  objects bound to chain207's pair: .text %s, aa %s, bb %s"
      % (C207_TEXT, C207_WA, C207_WB))

# G4: the chain207 side, loaded and checked to reproduce its sealed headline of 10 residues. If it
# does not, the comparison is broken and nothing below is interpretable.
f7, MS7, seen7, NNZ7 = load(C207, "dl_*_float32_p*.csv")
assert len(f7) == EXPECT_FILES, "  REFUSE: chain207 has %d CSVs" % len(f7)
Sa7, Sb7, ok7, FR7, B7, I7 = board(MS7)
if len(B7) != 10:
    sys.exit("  REFUSE: chain207 reproduces %d residues, not its sealed 10 -- the estimand or the "
             "ledger has moved and the comparison is void." % len(B7))
print("  G4  chain207 side reproduces its sealed headline: %d residues over %d cells" % (len(B7), len(ok7)))

Sa0, Sb0, ok0, FR0, B0, I0 = board(MS)
print()
print("FLOORS, p95 dev by nearest rank, per width")
print("  %-10s %s" % ("", "  ".join("k=%-7d" % k for k in KS)))
print("  %-10s %s" % ("chain207", "  ".join("%-9.4f" % FR7[k] for k in KS)))
print("  %-10s %s" % ("chain210", "  ".join("%-9.4f" % FR0[k] for k in KS)))

# ---------------------------------------------------------------- P0, the headline
print()
print("P0  the symmetric statistic, over all %d cells, computed independently on each run" % EXPECT_CELLS)
inter, union = B7 & B0, B7 | B0
print("  chain207 %d (byk %s)   chain210 %d (byk %s)"
      % (len(B7), "/".join(str(sum(1 for c in B7 if c[1] == k)) for k in KS),
         len(B0), "/".join(str(sum(1 for c in B0 if c[1] == k)) for k in KS)))
print("  intersection %d   union %d   Jaccard %.4f   only-207 %d   only-210 %d"
      % (len(inter), len(union), len(inter) / float(len(union)) if union else float("nan"),
         len(B7 - B0), len(B0 - B7)))
print("  SuiteSparse analogue: 14 and 18, intersection 7, union 25, Jaccard 0.2800")
for c in sorted(inter, key=lambda c: (c[1], c[0])):
    print("    REPLICATED k=%-3d S207 %.4f S210 %.4f  %s" % (c[1], Sa7[c], Sa0[c], c[0]))

# ---------------------------------------------------------------- P1, conditioned on run 1
print()
print("P1  survivors among chain207's 10, with the verdict fixed in advance")
surv = sorted(B7 & B0, key=lambda c: (c[1], c[0]))
n = len(surv)
V = {0: "0-2  the DLMC column is essentially selection on noise", 3: "3  consistent with the "
     "SuiteSparse rate at its low end; mostly churn", 4: "4-5  the SuiteSparse transfer rate "
     "reproduces on DLMC", 6: "6-8  DLMC is MORE reproducible than SuiteSparse",
     9: "9-10 the DLMC column is stable; the SuiteSparse extrapolation does not transfer"}
key = 0 if n <= 2 else 3 if n == 3 else 4 if n <= 5 else 6 if n <= 8 else 9
print("  survivors %d of 10 -> VERDICT %s" % (n, V[key]))
print("  (registered: n=10 cannot discriminate a transfer rate of 0.389 from 0.500. This tests "
      "nearly-all-noise against nearly-all-signal, which is what the two failure zones encode.)")
print()
print("P2' union SIZE. Predicted 14-18 from the SuiteSparse ratios 25/14=1.79 and 25/18=1.39 on a "
      "base of 10.")
print("  union %d -> %s" % (len(union), "WITHIN prediction" if 14 <= len(union) <= 18
      else "outside prediction but inside [12,21]" if 12 <= len(union) <= 21 else "FAILS P2'"))

# ---------------------------------------------------------------- P3, promoted to primary
print()
print("P3  per-matrix instability agreement at k=4, RUN isolated with objects held fixed")
common = sorted(set(I7) & set(I0))
k4 = [c for c in common if c[1] == 4]
xs, ys = [I7[c] for c in k4], [I0[c] for c in k4]
def pearson(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    return (sum((x - ma) * (y - mb) for x, y in zip(a, b))
            / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)))
def spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(a))}; rb = {v: i for i, v in enumerate(sorted(b))}
    return pearson([ra[x] for x in a], [rb[y] for y in b])
pe, sp = pearson(xs, ys), spearman(xs, ys)
a7 = {c for c in k4 if I7[c] > FR7[4]}; a0 = {c for c in k4 if I0[c] > FR0[4]}
exp = len(a7) * len(a0) / float(len(k4))
print("  n=%d matrices;  Pearson %.3f   Spearman %.3f" % (len(k4), pe, sp))
print("  above own floor: 207 %d, 210 %d ; shared %d vs %.1f expected (%.1fx enrichment)"
      % (len(a7), len(a0), len(a7 & a0), exp, len(a7 & a0) / exp if exp else float("nan")))
print("  VERDICT: %s" % ("<=0.40 -- CONFIRMS the run-specific reading on a cleaner design" if pe <= 0.40
      else ">=0.60 -- part of what was attributed to the RUN is object-layout artefact, and the "
           "7-of-25 SuiteSparse intersection needs re-reading" if pe >= 0.60
      else "between 0.40 and 0.60 -- INDETERMINATE, reported as such"))
print("  pooled floors agree within 1%%? %s" % ", ".join(
    "k=%d %.2f%%" % (k, 100 * dev(FR7[k] / FR0[k])) for k in KS))

# ---------------------------------------------------------------- P4, assay validity
print()
print("P4  ASSAY VALIDITY (not a negative control -- it requires a quantity to be PRESENT and "
      "stable). Pooled per-width median S must reproduce within 2%.")
worst = 0.0
for k in KS:
    m7 = red([Sa7[c] for c in ok7 if c[1] == k]); m0 = red([Sa0[c] for c in ok0 if c[1] == k])
    dv = 100 * dev(m7 / m0); worst = max(worst, dv)
    print("  k=%-3d chain207 %.4f  chain210 %.4f  dev %.2f%%" % (k, m7, m0, dv))
print("  worst %.2f%% -> %s" % (worst, "PASSES; the two runs measure the same thing" if worst <= 2.0
      else "FAILS -- this invalidates the COMPARISON, not the hypothesis, and P0-P3 are void"))

# ---------------------------------------------------------------- timing, H1/H2/H3
print()
print("TIMING. chain207: 12 passes, median 1412.5 s, stdev 14.4 s, CV 1.020%. Its pass 1 is the "
      "FASTEST of twelve.")
ev = {}
for l in log.splitlines():
    m = re.match(r"\s+--- p(\d) (aa|bb) (start|done)\s+(\d\d):(\d\d):(\d\d)", l)
    if m:
        p, b, k, h, mi, s = m.groups()
        ev.setdefault((int(p), b), {})[k] = int(h) * 3600 + int(mi) * 60 + int(s)
durs = []
for key2 in sorted(ev):
    d2 = ev[key2]
    if "start" in d2 and "done" in d2:
        t = d2["done"] - d2["start"]
        durs.append((key2, t + 86400 if t < 0 else t))
if durs:
    v = [t for _, t in durs]
    print("  chain210 per-pass: %s" % " ".join("%dm%02ds" % (t // 60, t % 60) for t in v))
    print("  n=%d median %.1f s  stdev %.1f s  CV %.3f%%"
          % (len(v), statistics.median(v), statistics.stdev(v) if len(v) > 1 else 0,
             100 * statistics.stdev(v) / statistics.median(v) if len(v) > 1 else 0))
    if len(v) == 12:
        tot = sum(v)
        print("  H1 chain207 absolute pace 1412.5 s/pass -> predicted total %d s" % (12 * 1412.5))
        print("  H2 chain207 shape x -1.61%%  1389.8 s/pass -> predicted total %d s" % (12 * 1389.8))
        print("  H3 no pass-1 effect       1348.0 s/pass -> predicted total %d s" % (12 * 1348))
        print("  OBSERVED total %d s. 12-pass noise scale 14.4*sqrt(12) = %.1f s" % (tot, 14.4 * 12 ** 0.5))
        best = min(("H1", 12 * 1412.5), ("H2", 12 * 1389.8), ("H3", 12 * 1348.0),
                   key=lambda z: abs(z[1] - tot))
        print("  nearest hypothesis: %s (|delta| %.0f s = %.1f noise units)"
              % (best[0], abs(best[1] - tot), abs(best[1] - tot) / (14.4 * 12 ** 0.5)))
print()
print("Floors, counts and verdicts above are the pre-registered ones. Nothing in this file selects "
      "a threshold after seeing a number.")
