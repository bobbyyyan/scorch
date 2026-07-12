#!/usr/bin/env python3
r"""collect_autotune_data.py — Phase-2 (learned autotune level) data collection.

For each (matrix, N) in a diverse per-machine workload set, TIME EVERY SHIPPED
candidate the production probe would consider — {v2, tile-j@Jc-sweep, tile-ijk@params}
— with the exact production thread policy, and emit one CSV row per candidate:

    (cheap structural features)  ->  measured median time / GFLOP/s  (the label)

The learned level (see autotune-levels/01-phase2-learned-design.md) trains a cost model
on these rows to PREDICT each candidate's time in O(1) and argmin — matching the
balanced/max probe's QUALITY at analytic COST. So the labels here MUST be the SHIPPED
kernels (scorch_ops.spmm_csr_float_{v2,tilej,tileijk}) invoked exactly as ops.py /
tiling.py invoke them, and the candidate set / features MUST be produced by reusing
tiling.py's own helpers (train/inference parity).

Key fidelity choices:
  * Thread policy = production: v2 via spmm_csr_float_v2(nthreads_override=torch
    .get_num_threads(), atparallel=True); tile-j/tile-ijk via the raw-omp nt =
    torch.get_num_threads() that maybe_dispatch passes through (see ops.py ~L391-419).
  * Timing = the interleaved rotated-rounds median (the cold-first / hybrid-turbo
    gotcha from spmm_tiling_study — a fixed order gives later configs a phantom win).
  * Correctness = float64 relative-Frobenius-norm vs a scipy reference, once per
    (matrix, N); a row is dropped + logged if relerr > 1e-2.
  * Features are the SAME cheap, O(1)/sampled quantities tiling.py can compute at
    INFERENCE (M, J, nnz, degree, N, C, locality span-ratio, degree_cv, + analytic
    per-schedule bytes) — NO true wavefront (that is O(nnz), not inference-cheap).
  * N is capped per matrix by an output+operand memory budget; every cap is LOGGED
    (no silent truncation).

Env:
  ATD_OUT           output CSV (default bench/bench_results/autotune_train_<machine>.csv)
  ATD_MATS          comma list to restrict the matrix set (default: all available)
  ATD_MEM_BUDGET_GB per-matmul dense-memory budget for the N cap (default 10)
  ATD_QUICK=1       smoke mode: tiny synthetic set, few N
  SCORCH_LLC_BYTES  override queried LLC (feature C); otherwise tiling.query_llc()
"""
from __future__ import annotations
import os
import sys
import time
import math
import statistics
import random
import csv as csvmod
import platform

# redwood-only HOME/MPLCONFIGDIR forcing (harmless if the dir is absent, e.g. M5)
if os.path.isdir("/scratch/bobbyy"):
    os.environ.setdefault("HOME", "/scratch/bobbyy")
    os.environ.setdefault("MPLCONFIGDIR", "/scratch/bobbyy/.mplcache")

import numpy as np
import scipy.sparse
import scipy.io
import torch

os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scorch  # noqa: E402  (loads scorch_ops)
import scorch_ops as _ops  # noqa: E402
from scorch import tiling as T  # noqa: E402  (reuse the production selector helpers)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SS_ROOTS = ["/scratch/suitesparse",
            os.path.expanduser("~/.cache/scorch/ss_full_envelope"),
            os.path.expanduser("~/.cache/scorch_suitesparse")]

C_LLC = T.query_llc()
MACHINE = T._machine_id()
NT_HOST = torch.get_num_threads()          # production _MATCH_HOST_THREADS policy
MEM_BUDGET = float(os.environ.get("ATD_MEM_BUDGET_GB", "10")) * (1 << 30)
QUICK = os.environ.get("ATD_QUICK", "0") == "1"


# ------------------------------------------------------------------ timing ----
def timed(thunks, warmup=2, min_rounds=3, max_rounds=7, budget=2.5):
    """Interleaved rotated-rounds median (fair across turbo/thread migration)."""
    K = len(thunks)
    for _ in range(warmup):
        for th in thunks:
            th()
    Tt = [[] for _ in range(K)]
    order = list(range(K))
    rng = random.Random(7)
    r = 0
    while r < max_rounds:
        rng.shuffle(order)
        for idx in order:
            t0 = time.perf_counter()
            thunks[idx]()
            Tt[idx].append(time.perf_counter() - t0)
        r += 1
        if r >= min_rounds and sum(sum(t) for t in Tt) >= budget:
            break
    return [statistics.median(t) for t in Tt]


# ------------------------------------------------------------- matrix zoo -----
def m_scatter(M, deg, seed=0):
    """Uniform-random columns, UNIFORM degree — the appu / m_scatter tail (needs a
    small Jc: no degree skew, no locality => the full panel IS the working set)."""
    rng = np.random.default_rng(seed)
    indptr = np.arange(0, (M + 1) * deg, deg, dtype=np.int64)
    cols = rng.integers(0, M, size=M * deg, dtype=np.int64)
    data = rng.random(M * deg, dtype=np.float32)
    x = scipy.sparse.csr_matrix((data, cols, indptr), shape=(M, M))
    x.sum_duplicates(); x.sort_indices(); return x


def m_powerlaw(M, avg_deg, seed=0, alpha=2.0):
    """Uniform-random columns, POWER-LAW degree — same (scattered) locality as
    m_scatter but SKEWED degree, so a small hot-B-set keeps the large Jc fine
    (reddit-like). The discriminator degree_cv must separate this from m_scatter."""
    rng = np.random.default_rng(seed)
    raw = rng.pareto(alpha, M) + 1.0
    degs = np.clip(np.round(raw * avg_deg / raw.mean()), 1, M // 2).astype(np.int64)
    nnz = int(degs.sum())
    indptr = np.zeros(M + 1, dtype=np.int64)
    np.cumsum(degs, out=indptr[1:])
    cols = rng.integers(0, M, size=nnz, dtype=np.int64)
    data = rng.random(nnz, dtype=np.float32)
    x = scipy.sparse.csr_matrix((data, cols, indptr), shape=(M, M))
    x.sum_duplicates(); x.sort_indices(); return x


def m_band(M, bw, seed=0):
    """Banded / well-ordered — high locality, v2 already streams the band from cache."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in range(M):
        lo = max(0, i - bw); hi = min(M, i + bw + 1)
        c = np.arange(lo, hi); rows.append(np.full(c.size, i)); cols.append(c)
    r = np.concatenate(rows); c = np.concatenate(cols)
    d = rng.random(r.size, dtype=np.float32)
    x = scipy.sparse.csr_matrix((d, (r, c)), shape=(M, M)); x.sort_indices(); return x


def m_mix(M, deg, p_scatter, seed=0, bw=64):
    """Locality-sweep: fraction p_scatter of each row's entries are uniform-random,
    the rest banded. Dials W*/J from banded (p=0) to scattered (p=1)."""
    rng = np.random.default_rng(seed)
    n_sc = int(round(deg * p_scatter)); n_bd = deg - n_sc
    indptr = np.arange(0, (M + 1) * deg, deg, dtype=np.int64)
    cols = np.empty(M * deg, dtype=np.int64)
    for i in range(M):
        base = i * deg
        if n_bd:
            lo = max(0, i - bw); hi = min(M, i + bw + 1)
            cols[base:base + n_bd] = rng.integers(lo, hi, size=n_bd)
        if n_sc:
            cols[base + n_bd:base + deg] = rng.integers(0, M, size=n_sc)
    data = rng.random(M * deg, dtype=np.float32)
    x = scipy.sparse.csr_matrix((data, cols, indptr), shape=(M, M))
    x.sum_duplicates(); x.sort_indices(); return x


def m_reddit_direct():
    g = np.load(os.path.join(REPO, "data/reddit/raw/reddit_graph.npz"))
    n = int(g["shape"][0])
    row = g["row"].astype(np.int64); col = g["col"].astype(np.int64)
    A = scipy.sparse.csr_matrix((np.ones(row.size, np.float32), (row, col)), shape=(n, n))
    A = A + scipy.sparse.eye(n, dtype=np.float32, format="csr")
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D = scipy.sparse.diags(dinv.astype(np.float32))
    Ahat = (D @ A @ D).tocsr().astype(np.float32)
    Ahat.sort_indices(); return Ahat


def m_gcn(name):
    if name == "reddit":
        return m_reddit_direct()
    import bench_gcn as G
    ds = G.load_dataset(name)
    a = G.compute_normalized_adj(ds.edge_index, ds.num_nodes)[0]
    x = scipy.sparse.csr_matrix(
        (a.values().numpy().astype(np.float32), a.col_indices().numpy(),
         a.crow_indices().numpy().astype(np.int64)), shape=(ds.num_nodes, ds.num_nodes))
    x.sort_indices(); return x


def m_suitesparse(name):
    for root in SS_ROOTS:
        for p in (os.path.join(root, name, name + ".mtx"),
                  os.path.join(root, name + ".mtx")):
            if os.path.exists(p):
                x = scipy.sparse.csr_matrix(scipy.io.mmread(p), dtype=np.float32)
                x.sort_indices(); return x
    return None


# --------------------------------------------------------------- features -----
def locality_ratio(csr, nsamp=64):
    """Cheap sampled locality proxy = mean(colspan)/J over sampled non-empty rows
    (the float behind tiling._scattered's boolean; the W* stand-in)."""
    pos = csr.indptr; crd = csr.indices; M, J = csr.shape
    if M <= 0 or crd.size == 0:
        return 0.0
    rng = np.random.default_rng(0)
    ridx = rng.integers(0, M, size=min(nsamp, M))
    b = pos[ridx]; e = pos[ridx + 1]
    nz = e > b
    if not nz.any():
        return 0.0
    b = b[nz]; e = e[nz]
    span = (crd[e - 1].astype(np.int64) - crd[b].astype(np.int64)).astype(np.float64).mean()
    return float(span / max(1, J))


def degree_cv(csr, nsamp=4096):
    """Degree-skew = std/mean of per-row degrees (THE discriminator between
    power-law skewed graphs -- large Jc fine -- and uniform-random -- small Jc)."""
    d = np.diff(csr.indptr).astype(np.float64)
    if d.size > nsamp:
        d = np.random.default_rng(0).choice(d, size=nsamp, replace=False)
    m = d.mean()
    if m <= 0:
        return 0.0
    return float(d.std() / m)


def analytic_bytes(kind, M, J, nnz, N, Jc, Nc, C):
    """Inference-cheap predicted DRAM bytes for a candidate (mirrors tiling's own
    byte model — NO wavefront). Used as a model feature so worst-case the tree
    reproduces the analytic decision and only learns the residual."""
    BN = 4.0 * N
    Cwr = M * BN
    A = 8.0 * nnz
    if kind == "v2":
        # bracket: optimistic (B fits, streamed once) .. pessimistic (fully
        # scattered, each nnz pulls a fresh B row). The tree interpolates via
        # locality / degree_cv.
        return J * BN + Cwr + A
    if kind == "tilej":
        P = max(1, -(-J // max(1, Jc)))                 # ceil(J/Jc)
        return J * BN + P * 2 * Cwr + A + P * M * 4
    # tileijk
    nk = max(1, -(-N // max(1, Nc)))                    # ceil(N/Nc)
    return J * BN + Cwr + A * nk + 2 * J * BN


FIELDS = [
    "machine_id", "matrix", "family", "sig", "M", "J", "nnz", "N",
    "degree", "locality", "degree_cv", "C_llc",
    "operand", "thrash_ratio", "output_bytes", "A_bytes", "log_nnz", "log_M", "log_J",
    "kind", "is_v2", "is_tilej", "is_tileijk", "cand_Jc", "cand_Nc",
    "cand_P", "cand_nk", "cand_bytes",
    "time_s", "gflops", "is_oracle", "relerr", "nt_host",
]


# ------------------------------------------------------------- kernel calls ---
def _stensors(csr, B):
    A = scorch.STensor.from_torch(torch.sparse_csr_tensor(
        torch.from_numpy(csr.indptr.astype(np.int64)),
        torch.from_numpy(csr.indices.astype(np.int64)),
        torch.from_numpy(csr.data.astype(np.float32)), size=csr.shape))
    Bt = scorch.STensor.from_torch(B)
    return A, Bt


def _extract(out, rshape):
    return out.storage.value.reshape(rshape)


def candidate_set(M, J, N, C):
    """The candidates the production probe would time, enriched with a couple extra
    tile-j Jc rungs to give the width-response curve more training signal. Returns
    list of (kind, Jc, Nc)."""
    base = T._panel_width(N, C)
    # ladder (base,/2,/4,/8) + brackets (*2, /16) for curve signal; clamp+dedup.
    jcs = sorted({min(J, max(16, base * m)) if m >= 1 else max(16, int(base * m))
                  for m in (2, 1, 0.5, 0.25, 0.125, 0.0625)})
    jcs = [jc for jc in jcs if jc < J] or [min(J, base)]
    cands = [("v2", 0, 0)]
    for jc in jcs:
        cands.append(("tilej", int(jc), 0))
    if T._HAS_TILEIJK and N >= T._NIJK_MIN:
        Nc, Jci = T._ijk_params(N, M, J, C)
        if Nc < N:
            cands.append(("tileijk", int(Jci), int(Nc)))
    return cands


def eval_matrix(matrix, family, csr, N, writer, loc, dcv):
    M, J = csr.shape
    nnz = int(csr.nnz)
    rs = [M, N]
    torch.manual_seed(0)
    B = torch.rand(J, N, dtype=torch.float32)
    ref = torch.from_numpy(csr.astype(np.float64) @ B.double().numpy())
    rn = ref.norm().item() + 1e-30
    A, Bt = _stensors(csr, B)
    ashape, ami, av = A.shape, A.index.mode_indices, A.values
    bshape, bmi, bv = Bt.shape, Bt.index.mode_indices, Bt.values

    def v2_thunk():
        return _ops.spmm_csr_float_v2(rs, ashape, ami, av, bshape, bmi, bv,
                                      nthreads_override=NT_HOST, atparallel=True)

    def tj_thunk(jc):
        return _ops.spmm_csr_float_tilej(rs, ashape, ami, av, bshape, bmi, bv, jc, NT_HOST)

    def ijk_thunk(nc, jc):
        return _ops.spmm_csr_float_tileijk(rs, ashape, ami, av, bshape, bmi, bv, nc, jc, NT_HOST)

    cands = candidate_set(M, J, N, C_LLC)
    thunks, metas = [], []
    for kind, jc, nc in cands:
        if kind == "v2":
            thunks.append(v2_thunk)
        elif kind == "tilej":
            thunks.append(lambda jc=jc: tj_thunk(jc))
        else:
            thunks.append(lambda nc=nc, jc=jc: ijk_thunk(nc, jc))
        metas.append((kind, jc, nc))

    # correctness: check v2 + first tile-j + tile-ijk (if present) against ref
    worst = 0.0
    for i, (kind, jc, nc) in enumerate(metas):
        if kind == "v2" or (kind == "tilej" and metas[i - 1][0] == "v2") or kind == "tileijk":
            out = thunks[i]()
            worst = max(worst, (_extract(out, rs).double() - ref).norm().item() / rn)
    if worst > 1e-2:
        print(f"  [BAD relerr {worst:.1e}] {matrix} N={N} — DROPPED", flush=True)
        return 0

    meds = timed(thunks)
    gfs = [2.0 * nnz * N / t / 1e9 for t in meds]
    oidx = int(np.argmax(gfs))

    operand = J * 4.0 * N
    base_row = dict(
        machine_id=MACHINE, matrix=matrix, family=family, sig="",
        M=M, J=J, nnz=nnz, N=N, degree=nnz / max(1, J), locality=loc, degree_cv=dcv,
        C_llc=C_LLC, operand=operand, thrash_ratio=operand / C_LLC,
        output_bytes=M * 4.0 * N, A_bytes=8.0 * nnz,
        log_nnz=math.log(max(1, nnz)), log_M=math.log(max(1, M)), log_J=math.log(max(1, J)),
        relerr=worst, nt_host=NT_HOST,
    )
    for i, (kind, jc, nc) in enumerate(metas):
        P = max(1, -(-J // max(1, jc))) if kind == "tilej" else 0
        nk = max(1, -(-N // max(1, nc))) if kind == "tileijk" else 0
        row = dict(base_row)
        row.update(
            kind=kind, is_v2=int(kind == "v2"), is_tilej=int(kind == "tilej"),
            is_tileijk=int(kind == "tileijk"), cand_Jc=jc, cand_Nc=nc,
            cand_P=P, cand_nk=nk,
            cand_bytes=analytic_bytes(kind, M, J, nnz, N, jc, nc, C_LLC),
            time_s=meds[i], gflops=gfs[i], is_oracle=int(i == oidx),
        )
        writer.writerow(row)
    win = metas[oidx]
    print(f"  N={N:5d} nnz={nnz} deg={nnz/max(1,J):.0f} loc={loc:.2f} cv={dcv:.2f} "
          f"| oracle={win[0]}{'@Jc'+str(win[1]) if win[0]=='tilej' else ('@Nc'+str(win[2]) if win[0]=='tileijk' else '')} "
          f"{gfs[oidx]:.0f} GF/s  (v2={gfs[0]:.0f})", flush=True)
    return len(metas)


# --------------------------------------------------------------- N capping ----
def n_cap(M, J, Ns):
    """Keep N values whose dense footprint (B operand + output + a tile-ijk strip)
    fits the memory budget. Returns (kept_Ns, dropped_Ns)."""
    kept, dropped = [], []
    for N in Ns:
        bytes_needed = (J * 4.0 * N) + (M * 4.0 * N) * 2   # B + C + ~relaid headroom
        (kept if bytes_needed <= MEM_BUDGET else dropped).append(N)
    return kept, dropped


# ------------------------------------------------------------------- grid -----
def build_grid():
    """(family, name, builder, [Ns]). Synthetics span the full N sweep (incl. the
    wide-B tile-ijk tail); real graphs cover the narrow GCN regime (their size
    caps N)."""
    FULL = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    NARROW = [64, 128, 256, 512, 1024]
    SMALL = [64, 128, 256]
    if QUICK:
        return [
            ("scatter", "scatter_deg200", lambda: m_scatter(8000, 200), [256, 1024, 2048]),
            ("powerlaw", "powerlaw_avg200", lambda: m_powerlaw(8000, 200), [256, 1024, 2048]),
            ("band", "band_bw16", lambda: m_band(8000, 16), [256, 1024]),
        ]
    grid = []
    # --- synthetic: uniform-random (appu tail) degree grid ---
    for deg in (16, 50, 120, 200, 500):
        grid.append(("scatter", f"scatter_deg{deg}",
                     lambda d=deg: m_scatter(30000 if d <= 200 else 16000, d), FULL))
    # --- synthetic: power-law degree (skewed; large-Jc-fine) ---
    for ad in (50, 200):
        grid.append(("powerlaw", f"powerlaw_avg{ad}",
                     lambda a=ad: m_powerlaw(30000, a), FULL))
    # --- synthetic: banded (v2 wins) ---
    for bw in (4, 16, 64):
        grid.append(("band", f"band_bw{bw}", lambda b=bw: m_band(40000, b), FULL))
    # --- synthetic: locality sweep at two degrees ---
    for deg in (50, 200):
        for p in (0, 25, 50, 75, 100):
            grid.append(("mix", f"mix_p{p}_deg{deg}",
                         lambda d=deg, pp=p: m_mix(30000, d, pp / 100.0), FULL))
    # --- real graphs (via ogb / direct reddit) ---
    grid.append(("real_graph", "reddit", lambda: m_gcn("reddit"), NARROW))
    grid.append(("real_graph", "ogbn-arxiv", lambda: m_gcn("ogbn-arxiv"), NARROW))
    grid.append(("real_graph", "ogbn-products", lambda: m_gcn("ogbn-products"), SMALL))
    # --- SuiteSparse: FEM (v2) + social/web (scattered) ---
    for nm, fam in [("cant", "fem"), ("pwtk", "fem"), ("pdb1HYS", "fem"),
                    ("bcsstk17", "fem"), ("bcsstk30", "fem"),
                    ("ca-CondMat", "social"), ("ca-AstroPh", "social"),
                    ("as-735", "social"), ("email-Enron", "social")]:
        grid.append((fam, nm, lambda n=nm: m_suitesparse(n), NARROW + [2048, 4096]))
    return grid


def main():
    out = os.environ.get("ATD_OUT") or os.path.join(
        REPO, "bench", "bench_results", f"autotune_train_{platform.system().lower()}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    restrict = set(x for x in os.environ.get("ATD_MATS", "").split(",") if x)
    print(f"[cfg] machine={MACHINE} platform={platform.system()} NT_HOST={NT_HOST} "
          f"C_LLC={C_LLC/1e6:.0f}MB mem_budget={MEM_BUDGET/1e9:.0f}GB "
          f"NIJK_MIN={T._NIJK_MIN} quick={QUICK}", flush=True)
    print(f"[cfg] out={out}", flush=True)

    grid = build_grid()
    n_rows = 0
    n_cells = 0
    skipped, capped = [], []
    with open(out, "w", newline="") as f:
        writer = csvmod.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for family, name, builder, Ns in grid:
            if restrict and name not in restrict:
                continue
            try:
                csr = builder()
            except Exception as ex:
                print(f"[skip] {name}: {type(ex).__name__}: {ex}", flush=True)
                skipped.append((name, f"{type(ex).__name__}: {ex}")); continue
            if csr is None:
                print(f"[skip] {name}: not found in SS roots", flush=True)
                skipped.append((name, "not found")); continue
            M, J = csr.shape
            kept, dropped = n_cap(M, J, Ns)
            if dropped:
                capped.append((name, dropped))
                print(f"[cap] {name} M={M} J={J}: dropped N={dropped} (>{MEM_BUDGET/1e9:.0f}GB)", flush=True)
            if not kept:
                print(f"[skip] {name}: all N over budget", flush=True)
                skipped.append((name, "all N over budget")); continue
            loc = locality_ratio(csr); dcv = degree_cv(csr)
            print(f"\n##### [{family}] {name}  M={M} J={J} nnz={csr.nnz} "
                  f"({csr.nnz/M:.0f}/row) loc={loc:.3f} cv={dcv:.2f} #####", flush=True)
            for N in kept:
                try:
                    n_rows += eval_matrix(name, family, csr, N, writer, loc, dcv)
                    n_cells += 1
                    f.flush()
                except Exception as ex:
                    print(f"  [err] {name} N={N}: {type(ex).__name__}: {ex}", flush=True)
                    skipped.append((f"{name}@N={N}", f"{type(ex).__name__}: {ex}"))
    print(f"\n===== DONE: {n_rows} rows over {n_cells} (matrix,N) cells -> {out} =====", flush=True)
    if capped:
        print(f"[coverage] N-capped matrices: {capped}", flush=True)
    if skipped:
        print(f"[coverage] skipped: {skipped}", flush=True)


if __name__ == "__main__":
    main()
