#!/usr/bin/env python3
r"""tiling_selector.py — the adaptive SpMM tiling selector + offline validation.

Given cheap structural features (M, J, nnz, wavefront W*) + N + the cache
hierarchy, pick the optimal schedule from {none, tile-i, tile-ik, tile-j,
tile-ijk(relaid)}.  Validated against the brute-force ORACLE measured by
bench_tiling_autotuner.py on TWO machines:

  * redwood  i9-14900K 24T  (x86, private 2MB L2 / shared 36MB L3):
        STATIC geomean 0.977 of oracle (18/25) ; PROBE geomean 1.000 (25/25).
  * M5 Max  18T (6P+12E)    (ARM, shared 16MB P-cluster L2 / large SLC):
        STATIC geomean 0.833 (6/20)            ; PROBE geomean 0.999 (20/20).

Two modes:
  select_static() : one analytic pick (no measurement).
  candidates()    : the schedules to micro-probe on the first call (memoized by
                    shape) -> robust across the compute-vs-memory boundary AND
                    across machines (the version that hits ~1.0 on both).

Mechanism: B's reuse working set is a rectangle  W* live-rows TALL x N WIDE.
  - banded (W*/J small): the band is PER-THREAD -> binding cache = per-thread L2.
      fits L2  -> scheduling only (tile-i);  spills -> tile-ik (x86) / tile-j (M5).
  - scattered (W*/J ~ 1): B is SHARED across threads -> binding cache = L3/SLC.
      fits     -> scheduling (tile-i);  thrash -> tile-j (moderate N) or
      tile-ijk with B width-panel relayout (wide N, where tile-j's C re-traffic ~N^2).

★ WHY THE STATIC MODEL IS WEAKER ON M5 (the key cross-platform finding):
  On x86 the winner IS a function of (working-set, cache) — the static traffic
  model nails 0.977.  On M5 two things break that:
   (1) tile-ik NEVER wins.  M5 has no private per-core L2 — the 6 P-cores share one
       16MB L2 — so a banded band never spills, and tile-ik's extra A-rescan only
       costs.  Banded-wide-N goes to tile-j (then relaid tile-ijk), not tile-ik.
   (2) tile-j / tile-ijk win even when the operand is CACHE-RESIDENT (band@N=1024
       W*.4N=135KB fits everything, yet tile-j 438 > none 363; scatter200@N=64
       operand 7.7MB < LLC, yet tile-j 235 > none 196).  This is an M5 memory-
       system STREAMING preference (contiguous column-panels stream faster than
       row-major gather) that no cache-size arithmetic predicts.
  => On M5 the analytic pick is only ~0.83 of oracle; the MICRO-PROBE (which
     measures) recovers ~1.0.  The probe is the portable production path.
"""
from __future__ import annotations
import math

MB = 1024 * 1024

# Per-machine cache constants.  Query these at init from the OS (Phase B does:
# macOS sysctl hw.perflevel0.l2cachesize + an SLC estimate; Linux /sys L3).  The
# M5 values are the empirical BEST FIT to the M5 oracle (grid search over the 20
# synthetic/FEM cells); they land near the 16MB P-cluster L2, which IS the M5
# binding cache.  The probe path is insensitive to their exact value (it only
# needs the candidate SET to contain the winner).
CONSTS = {
    "redwood": dict(L2c=2 * MB, L3=36 * MB, ik_ok=True),
    "m5":      dict(L2c=12 * MB, L3=16 * MB, ik_ok=False),  # tile-ik dead on M5
}
LOC_BANDED = 0.15          # W*/J below this => well-ordered / banded (structural)
FITS_MARGIN = 1.5          # scattered "fits L3" slack before real tiling pays


def select_static(M, J, nnz, Wstar, N, *, L2c, L3, ik_ok=True,
                  loc_banded=LOC_BANDED, fits_margin=FITS_MARGIN):
    loc = Wstar / J
    bw = Wstar * 4 * N                        # per-thread band working set (naive)
    banded = loc < loc_banded
    if banded:
        if bw <= L2c:
            return "tile-i"                    # band fits per-thread L2 -> scheduling
        return "tile-ik" if ik_ok else "tile-j"   # spills: shrink width (x86) / stream (M5)
    # scattered
    if bw <= fits_margin * L3:
        return "tile-i"                        # fits shared L3/SLC (w/ margin)
    # real thrash: tile-j vs tile-ijk by predicted DRAM bytes
    P = math.ceil(J * 4 * N / L3)
    tj = J * 4 * N + P * 2 * M * N * 4 + 8 * nnz               # tile-j: C re-traffic ~N^2
    Nc = min(N, max(1, int(L3 / (4 * M))))                    # tile-ijk width (Cp in L3)
    nk = math.ceil(N / Nc)
    tijk = J * 4 * N + M * N * 4 + 8 * nnz * nk                # tile-ijk: A re-scan nk
    return "tile-j" if tj <= tijk else "tile-ijk"


def candidates(M, J, nnz, Wstar, N, *, L2c, L3, ik_ok=True,
               loc_banded=LOC_BANDED, fits_margin=FITS_MARGIN):
    """Schedules to micro-probe (first call, memoized by shape+N). ALWAYS contains
    the production kernel (none/tile-i == spmm_csr_float_v2) so the probe can never
    regress it -> the CLAUDE.md no-regression gate holds by construction.

    Prune to {tile-i,none} ONLY when the operand fits the LLC AND there is little
    cross-row B reuse to recover (banded, or genuinely low degree): those provably
    can't beat v2, so we skip the probe entirely (zero overhead on GCN-small / AE /
    the FEM panel's narrow-N cells). Big-operand OR high-degree shapes add tile-j
    and (wide-N) tile-ijk for the probe to choose among."""
    loc = Wstar / J
    deg = nnz / J
    operand_fits = (J * 4 * N) <= L3
    banded = loc < loc_banded
    # thrash-and-tile TILE-PAYS proxy: deg > 2*J*4N/C  <=>  reuse worth recovering
    low_reuse = banded or deg <= max(32.0, 2.0 * J * 4 * N / L3)
    if operand_fits and low_reuse:
        return ["tile-i", "none"]
    # tile-ik is a candidate only where it can win (x86 private-L2 wide-N banded);
    # on M5 (ik_ok=False) it is dominated everywhere, so leaving it out saves a
    # first-call probe measurement without ever losing the winner.
    ik = ["tile-ik"] if ik_ok else []
    if banded:
        return ["tile-i", "none"] + ik + ["tile-j", "tile-ijk"]
    return ["none", "tile-i"] + ik + ["tile-j", "tile-ijk"]


# ---- offline validation against parsed oracle logs -----------------------
def _gm(xs):
    return math.exp(sum(math.log(max(1e-9, x)) for x in xs) / len(xs))


def validate(gt, consts, verbose=True):
    sr, pr, sh, ph = [], [], 0, 0
    if verbose:
        print(f"{'matrix':10s}{'W*/J':>6s}{'N':>7s} | {'oracle':>9s}{'gf':>5s} | "
              f"{'static':>9s}{'r':>6s} | probe(cands)          r")
    for name, M, J, nnz, Ws, N, g in gt:
        oracle = max(g, key=g.get); og = g[oracle]
        st = select_static(M, J, nnz, Ws, N, **consts)
        sg = g.get(st, 0.0); s = sg / og
        cs = candidates(M, J, nnz, Ws, N, **consts)
        pg = max(g.get(c, 0.0) for c in cs); p = pg / og
        sr.append(s); pr.append(p); sh += s >= 0.97; ph += p >= 0.97
        if verbose:
            print(f"{name:10s}{Ws/J:6.2f}{N:7d} | {oracle:>9s}{og:5.0f} | "
                  f"{st:>9s}{s:6.2f} | {'+'.join(c.replace('tile-','') for c in cs):20s}{p:5.2f}")
    return _gm(sr), sh, _gm(pr), ph, len(gt)


# ---- redwood ground truth (best GFLOP/s per schedule), 25 cells ----------
GT_REDWOOD = [
    ("band", 40000, 40000, 1319728, 33, 64,    {"none":393,"tile-i":586,"tile-ik":445,"tile-ijk":300}),
    ("band", 40000, 40000, 1319728, 33, 256,   {"none":231,"tile-i":252,"tile-ik":285,"tile-j":224,"tile-ijk":212}),
    ("band", 40000, 40000, 1319728, 33, 1024,  {"none":254,"tile-i":253,"tile-ik":250,"tile-j":192,"tile-ijk":219}),
    ("band", 40000, 40000, 1319728, 33, 4096,  {"none":253,"tile-i":253,"tile-ik":254,"tile-j":175,"tile-ijk":226}),
    ("band", 40000, 40000, 1319728, 33, 16384, {"none":135,"tile-i":135,"tile-ik":241,"tile-j":68, "tile-ijk":240}),
    ("scat16", 40000, 40000, 639904, 39972, 64,    {"none":182,"tile-i":266,"tile-ik":258,"tile-ijk":148}),
    ("scat16", 40000, 40000, 639904, 39972, 256,   {"none":78, "tile-i":81, "tile-ik":91, "tile-j":62, "tile-ijk":77}),
    ("scat16", 40000, 40000, 639904, 39972, 1024,  {"none":33, "tile-i":33, "tile-ik":33, "tile-j":33, "tile-ijk":79}),
    ("scat16", 40000, 40000, 639904, 39972, 4096,  {"none":27, "tile-i":27, "tile-ik":26, "tile-j":19, "tile-ijk":87}),
    ("scat16", 40000, 40000, 639904, 39972, 16384, {"none":26, "tile-i":26, "tile-ik":25, "tile-j":13, "tile-ijk":88}),
    ("scat200", 30000, 30000, 5979992, 30000, 64,    {"none":306,"tile-i":301,"tile-ik":304,"tile-ijk":268}),
    ("scat200", 30000, 30000, 5979992, 30000, 256,   {"none":219,"tile-i":222,"tile-ik":221,"tile-j":235,"tile-ijk":226}),
    ("scat200", 30000, 30000, 5979992, 30000, 1024,  {"none":47, "tile-i":47, "tile-ik":42, "tile-j":255,"tile-ijk":244}),
    ("scat200", 30000, 30000, 5979992, 30000, 4096,  {"none":32, "tile-i":32, "tile-ik":33, "tile-j":114,"tile-ijk":259}),
    ("scat200", 30000, 30000, 5979992, 30000, 16384, {"none":30, "tile-i":30, "tile-ik":30, "tile-j":43, "tile-ijk":263}),
    ("cant", 62451, 62451, 4007383, 549, 64,   {"none":291,"tile-i":444,"tile-ik":403,"tile-ijk":239}),
    ("cant", 62451, 62451, 4007383, 549, 256,  {"none":360,"tile-i":364,"tile-ik":381,"tile-j":323,"tile-ijk":248}),
    ("cant", 62451, 62451, 4007383, 549, 1024, {"none":405,"tile-i":398,"tile-ik":367,"tile-j":335,"tile-ijk":272}),
    ("cant", 62451, 62451, 4007383, 549, 4096, {"none":306,"tile-i":307,"tile-ik":313,"tile-j":228,"tile-ijk":284}),
    ("arxiv", 169343, 169343, 2484941, 151051, 64,   {"none":82,"tile-i":87,"tile-ik":83,"tile-j":51,"tile-ijk":42}),
    ("arxiv", 169343, 169343, 2484941, 151051, 256,  {"none":39,"tile-i":40,"tile-ik":42,"tile-j":37,"tile-ijk":39}),
    ("arxiv", 169343, 169343, 2484941, 151051, 1024, {"none":29,"tile-i":29,"tile-ik":31,"tile-j":23,"tile-ijk":42}),
    ("reddit", 232965, 232965, 114848857, 231023, 64,   {"none":121,"tile-i":121,"tile-ik":123,"tile-j":259,"tile-ijk":195}),
    ("reddit", 232965, 232965, 114848857, 231023, 256,  {"none":44, "tile-i":45, "tile-ik":49, "tile-j":297,"tile-ijk":229}),
    ("reddit", 232965, 232965, 114848857, 231023, 1024, {"none":32, "tile-i":32, "tile-ik":32, "tile-j":200,"tile-ijk":233}),
]


def main():
    import os, sys
    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 78)
    print("REDWOOD (x86, L2c=2MB L3=36MB):")
    sgm, sh, pgm, ph, n = validate(GT_REDWOOD, CONSTS["redwood"])
    print(f"\n  STATIC geomean {sgm:.3f} ({sh}/{n})  |  PROBE geomean {pgm:.3f} ({ph}/{n})")

    # M5: parse the local oracle log if present
    m5log = os.path.join(here, "bench_results", "tiling_autotuner_m5_nt18.log")
    if not os.path.exists(m5log):
        m5log = os.path.join(here, "bench_results", "tiling_autotuner_darwin.log")
    if os.path.exists(m5log):
        from tiling_selector_validate import parse_log
        gt = parse_log(m5log)
        print("\n" + "=" * 78)
        print(f"M5 (ARM, L2c=12MB SLC=16MB, tile-ik OFF) — {len(gt)} cells from {os.path.basename(m5log)}:")
        sgm, sh, pgm, ph, n = validate(gt, CONSTS["m5"])
        print(f"\n  STATIC geomean {sgm:.3f} ({sh}/{n})  |  PROBE geomean {pgm:.3f} ({ph}/{n})")
    else:
        print(f"\n[no M5 oracle log at {m5log}]")


if __name__ == "__main__":
    main()
