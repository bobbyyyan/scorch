#!/usr/bin/env python3
r"""bench_spmm_vs_mkl.py — Scorch's adaptive SpMM tiling selector vs PyTorch / MKL.

For every (matrix, N) cell it times CSR x dense -> dense with:

  mkl_csr      torch.sparse.mm(A_csr_i64, B)  -> MKL sparse, int64 indices (what a
                                                 PyTorch user gets from from_scipy)
  mkl_csr32    torch.sparse.mm(A_csr_i32, B)  -> MKL sparse, int32 indices (MKL's
                                                 native index width: its best shot)
  torch_coo    torch.sparse.mm(A_coo, B)      -> ATen's own COO SpMM
  sc_off       scorch.matmul, autotune "off"       -> pure spmm_csr_float_v2, no tiling
  sc_analytic  scorch.matmul, autotune "analytic"  -> cost-model pick, no probe (DEFAULT)
  sc_balanced  scorch.matmul, autotune "balanced"  -> first-call micro-probe, memoized
  sc_max       scorch.matmul, autotune "max"       -> probe + persistent on-disk cache
  sc_learned   scorch.matmul, autotune "learned"   -> offline-trained GBT cost model
  aa_control   identical code to sc_off, timed as a separate arm -> noise floor

Dense SGEMM (A_dense @ B) is timed in a SEPARATE pass, never interleaved: it is
10-100x slower than the sparse arms and evicting the whole LLC right before a
sub-millisecond sparse kernel is a systematic bias, not noise.

Methodology (the parts that matter on a shared box):
  * RANDOM-PERMUTATION INTERLEAVE. Every round runs each arm exactly once in a fresh
    seeded random order. Merely *rotating* a fixed list preserves arm-to-arm
    adjacency, so a heavy arm keeps polluting the cache for whichever arm follows it;
    permuting breaks that. Thermal drift and contention then hit all arms equally.
  * A/A NOISE FLOOR. sc_off is entered twice under different names, running identical
    code. |aa_control/sc_off - 1| is that cell's measurement noise; no speedup or
    slowdown smaller than it is real.
  * The autotune level is switched OUTSIDE the timed region (per-arm prep), so no
    context-manager overhead lands in any arm's time.
  * Per-arm warmup before timing, so the balanced/max/learned first-call probe and
    any lazy allocation are paid outside the timed region.
  * Median over --reps rounds.
  * Two numbers per scorch arm: END-TO-END (what a user's call costs, including
    Python dispatch) and KERNEL-ONLY (the native kernel's own time, from matmul's
    time_dict), so kernel quality and dispatch overhead can be told apart.
  * Correctness: every arm's result is compared against a float64 scipy reference.

It also records what the selector DECIDED at each level (route + tile params) and the
cheap structural features that decision was made from, so the numbers can be read
next to the mechanism.

Usage (redwood, isolated pinned tree):
  PYTHONPATH=<tree>/src python bench/bench_spmm_vs_mkl.py \
      --group main --ns 32 128 512 --reps 15 --csv out.csv
"""
from __future__ import annotations

import argparse
import csv as csvmod
import gc
import math
import os
import platform
import random
import statistics
import sys
import time

# redwood-only HOME/MPLCONFIGDIR forcing (harmless elsewhere)
if os.path.isdir("/scratch/bobbyy"):
    os.environ.setdefault("HOME", "/scratch/bobbyy")
    os.environ.setdefault("MPLCONFIGDIR", "/scratch/bobbyy/.mplcache")

import numpy as np
import scipy.io
import scipy.sparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scorch  # noqa: E402
from scorch import tiling  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

SS_ROOTS = [
    "/scratch/suitesparse",
    os.path.expanduser("~/.cache/scorch/ss_full_envelope"),
    os.path.expanduser("~/.cache/scorch_suitesparse"),
]

# Where the GNN graph data lives. We only ever READ from these.
DATA_ROOTS = [
    os.path.join(REPO, "data"),
    "/scratch/bobbyy/scorch/data",
]


# --------------------------------------------------------------------------- #
# matrix loaders
# --------------------------------------------------------------------------- #
def m_suitesparse(name):
    for root in SS_ROOTS:
        for p in (os.path.join(root, name, name + ".mtx"), os.path.join(root, name + ".mtx")):
            if os.path.exists(p):
                x = scipy.sparse.csr_matrix(scipy.io.mmread(p), dtype=np.float32)
                x.sort_indices()
                return x
    return None


def m_band(M, bw, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in range(M):
        lo, hi = max(0, i - bw), min(M, i + bw + 1)
        c = np.arange(lo, hi)
        rows.append(np.full(c.size, i))
        cols.append(c)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    d = rng.random(r.size, dtype=np.float32)
    x = scipy.sparse.csr_matrix((d, (r, c)), shape=(M, M))
    x.sort_indices()
    return x


def m_scatter(M, deg, seed=0):
    rng = np.random.default_rng(seed)
    indptr = np.arange(0, (M + 1) * deg, deg, dtype=np.int64)
    cols = rng.integers(0, M, size=M * deg, dtype=np.int64)
    data = rng.random(M * deg, dtype=np.float32)
    x = scipy.sparse.csr_matrix((data, cols, indptr), shape=(M, M))
    x.sum_duplicates()
    x.sort_indices()
    return x


def _norm_adj_from_edges(row, col, n):
    """GCN-normalized adjacency  D~^-1/2 (A+I) D~^-1/2  from an edge list."""
    A = scipy.sparse.csr_matrix((np.ones(row.size, np.float32), (row, col)), shape=(n, n))
    A = A + scipy.sparse.eye(n, dtype=np.float32, format="csr")
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D = scipy.sparse.diags(dinv.astype(np.float32))
    Ahat = (D @ A @ D).tocsr().astype(np.float32)
    Ahat.sort_indices()
    return Ahat


def _data_path(*parts):
    for root in DATA_ROOTS:
        p = os.path.join(root, *parts)
        if os.path.exists(p):
            return p
    return None


def m_reddit():
    p = _data_path("reddit", "raw", "reddit_graph.npz")
    if p is None:
        return None
    g = np.load(p)
    n = int(g["shape"][0])
    return _norm_adj_from_edges(g["row"].astype(np.int64), g["col"].astype(np.int64), n)


def m_ogbn(name):
    """ogbn-arxiv / ogbn-products straight from the OGB csv.gz files (no ogb import)."""
    stem = name.replace("-", "_")
    edge = _data_path(name, stem, "raw", "edge.csv.gz")
    nodes = _data_path(name, stem, "raw", "num-node-list.csv.gz")
    if edge is None or nodes is None:
        return None
    import gzip

    with gzip.open(nodes, "rt") as fh:
        n = int(fh.readline().strip())
    cache = os.path.join(
        os.environ.get("SPMM_MKL_CACHE", "/tmp"), f"{stem}_edges.npy"
    )
    if os.path.exists(cache):
        e = np.load(cache)
    else:
        e = np.loadtxt(edge, delimiter=",", dtype=np.int64)
        try:
            np.save(cache, e)
        except Exception:
            pass
    row = np.concatenate([e[:, 0], e[:, 1]])
    col = np.concatenate([e[:, 1], e[:, 0]])
    return _norm_adj_from_edges(row, col, n)


def m_planetoid(name):
    """cora / citeseer / pubmed from the Planetoid graph pickle."""
    p = _data_path(name, name, "raw", f"ind.{name}.graph") or _data_path(
        name, "raw", f"ind.{name}.graph"
    )
    if p is None:
        return None
    import pickle

    with open(p, "rb") as fh:
        graph = pickle.load(fh, encoding="latin1")
    n = len(graph)
    rows, cols = [], []
    for k, vs in graph.items():
        for v in vs:
            rows.append(k)
            cols.append(v)
    return _norm_adj_from_edges(np.asarray(rows, np.int64), np.asarray(cols, np.int64), n)


# --------------------------------------------------------------------------- #
# grids
# --------------------------------------------------------------------------- #
GROUPS = {
    # the headline grid: real graphs + FEM/circuit + synthetic regime anchors
    "main": [
        # --- real GNN adjacencies (the workloads scorch actually ships for) ---
        ("gcn:cora", lambda: m_planetoid("cora")),
        ("gcn:citeseer", lambda: m_planetoid("citeseer")),
        ("gcn:pubmed", lambda: m_planetoid("pubmed")),
        ("gcn:ogbn-arxiv", lambda: m_ogbn("ogbn-arxiv")),
        ("gcn:reddit", m_reddit),
        # --- SuiteSparse: FEM / structural / circuit (gate should stay inert) ---
        ("ss:bcsstk17", lambda: m_suitesparse("bcsstk17")),
        ("ss:scircuit", lambda: m_suitesparse("scircuit")),
        ("ss:cop20k_A", lambda: m_suitesparse("cop20k_A")),
        ("ss:webbase-1M", lambda: m_suitesparse("webbase-1M")),
        ("ss:pdb1HYS", lambda: m_suitesparse("pdb1HYS")),
        ("ss:consph", lambda: m_suitesparse("consph")),
        ("ss:pwtk", lambda: m_suitesparse("pwtk")),
        ("ss:ct20stif", lambda: m_suitesparse("ct20stif")),
        ("ss:thermal2", lambda: m_suitesparse("thermal2")),
        # --- SuiteSparse: high-degree / scattered (where tiling can fire) ---
        ("ss:nd24k", lambda: m_suitesparse("nd24k")),
        ("ss:mouse_gene", lambda: m_suitesparse("mouse_gene")),
        ("ss:ca-AstroPh", lambda: m_suitesparse("ca-AstroPh")),
        ("ss:email-Enron", lambda: m_suitesparse("email-Enron")),
        # --- synthetic regime anchors ---
        ("syn:band16", lambda: m_band(40000, 16)),
        ("syn:scatter16", lambda: m_scatter(40000, 16)),
        ("syn:scatter200", lambda: m_scatter(30000, 200)),
        ("syn:scatter200-big", lambda: m_scatter(300000, 200)),
    ],
    # the wide-B tail, where tile-ijk (width-panel relayout) enters the candidate set
    "wide": [
        ("syn:scatter200", lambda: m_scatter(20000, 200)),
        ("syn:scatter120", lambda: m_scatter(30000, 120, seed=2)),
        ("ss:mouse_gene", lambda: m_suitesparse("mouse_gene")),
        ("gcn:reddit", m_reddit),
    ],
    "smoke": [
        ("syn:scatter200-big", lambda: m_scatter(300000, 200)),
        ("syn:band16", lambda: m_band(20000, 16)),
    ],
}

# The canonical SuiteSparse sets already used elsewhere in bench/, so these numbers
# line up with the existing SpMM studies instead of a fresh ad-hoc selection.
#
# "ss-tiling": bench/bench_spmm_tiling.py DEFAULT_MATRICES — the 20-matrix SpMM
# tiling-study envelope, 0.44M..119M nnz, deliberately spanning short-row circuit /
# web graphs through dense-block structural matrices. This is the set the tiling
# strategy study was derived on, so it is the right one for a selector benchmark.
SS_TILING = [
    "bcsstk17",         # 0.44M  FEM, small
    "scircuit",         # 0.96M  circuit, ~6 nnz/row (irregular, short)
    "mac_econ_fwd500",  # 1.27M  economics, short rows
    "cage12",           # 2.03M  DNA, short rows
    "rma10",            # 2.37M  CFD
    "cop20k_A",         # 2.72M  accelerator, ~22 nnz/row
    "webbase-1M",       # 3.11M  web graph, power-law, 1M rows, ~3 nnz/row
    "cant",             # 4.07M  FEM cantilever
    "pdb1HYS",          # 4.38M  protein, ~120 nnz/row
    "consph",           # 6.09M  FEM concentric spheres
    "shipsec1",         # 7.95M  FEM ship section
    "thermal2",         # 9.81M  thermal FEM, 1.2M rows, short
    "crankseg_1",       # 10.7M  structural, ~202 nnz/row (dense blocks)
    "pwtk",             # 11.9M  pressurized wind tunnel
    "nd24k",            # 28.8M  2-D/3-D, ~400 nnz/row (dense)
    "mouse_gene",       # 29.0M  gene network, ~643 nnz/row (very dense rows)
    "inline_1",         # 37.3M  structural, large
    "ldoor",            # 47.5M  structural, large
    "audikw_1",         # 78.6M  automotive crankshaft, very large
    "Flan_1565",        # 119M   steel flange, the giant
]

# "ss-quick": bench/_utils.py _GENERAL_MATRICES — the curated 21-matrix set that
# bench_spmm.py --matrix-set quick runs. Skewed small (1K..12M nnz), so it mostly
# probes the cache-resident regime where the selector is inert by design.
SS_QUICK = [
    "arc130", "494_bus", "ash292", "bcspwr06", "bcspwr09", "bcsstk08",
    "bcsstk14", "bcsstk13", "bcsstk15", "bcsstk16", "bcsstk17",
    "bcsstk33", "bcsstk29", "crystk02", "bcsstk31", "crystk03",
    "bcsstk30", "ct20stif", "gupta2", "pre2", "pkustk11",
]

GROUPS["ss-tiling"] = [
    (f"ss:{n}", (lambda n=n: m_suitesparse(n))) for n in SS_TILING
]
GROUPS["ss-quick"] = [
    (f"ss:{n}", (lambda n=n: m_suitesparse(n))) for n in SS_QUICK
]


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def to_st(csr):
    return scorch.STensor.from_torch(
        torch.sparse_csr_tensor(
            torch.from_numpy(csr.indptr.astype(np.int64)),
            torch.from_numpy(csr.indices.astype(np.int64)),
            torch.from_numpy(csr.data.astype(np.float32)),
            size=csr.shape,
        )
    )


def _noop():
    pass


def build_arms(csr, B, A_st, B_st, args):
    """Return {name: (prep, run)}; prep is untimed, run materializes a full result."""
    A_csr64 = torch.sparse_csr_tensor(
        torch.from_numpy(csr.indptr.astype(np.int64)),
        torch.from_numpy(csr.indices.astype(np.int64)),
        torch.from_numpy(csr.data.astype(np.float32)),
        size=csr.shape,
    )
    A_csr32 = torch.sparse_csr_tensor(
        torch.from_numpy(csr.indptr.astype(np.int32)),
        torch.from_numpy(csr.indices.astype(np.int32)),
        torch.from_numpy(csr.data.astype(np.float32)),
        size=csr.shape,
    )
    A_coo = None
    if not args.no_coo and csr.nnz <= args.coo_nnz_cap:
        coo = csr.tocoo()
        A_coo = torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)),
            torch.from_numpy(coo.data.astype(np.float32)),
            size=csr.shape,
        ).coalesce()

    arms = {
        "mkl_csr": (_noop, lambda: torch.sparse.mm(A_csr64, B)),
        "mkl_csr32": (_noop, lambda: torch.sparse.mm(A_csr32, B)),
    }
    if not args.no_coo and csr.nnz <= args.coo_nnz_cap:
        arms["torch_coo"] = (_noop, lambda: torch.sparse.mm(A_coo, B))

    def sc_arm(level, td):
        def prep(level=level):
            tiling.set_autotune(level)

        def run():
            return scorch.matmul(A_st, B_st, time_dict=td)

        return prep, run

    kernel_td = {}
    for lvl in args.levels:
        td = {}
        kernel_td["sc_" + lvl] = td
        arms["sc_" + lvl] = sc_arm(lvl, td)
    td_aa = {}
    kernel_td["aa_control"] = td_aa
    arms["aa_control"] = sc_arm("off", td_aa)
    return arms, kernel_td


def permuted_interleave(arms, reps, warmup, seed, kernel_td):
    """Random-permutation interleaved timing. Returns (median_s, median_kernel_s)."""
    names = list(arms)
    for n in names:
        prep, run = arms[n]
        prep()
        for _ in range(warmup):
            run()
    samples = {n: [] for n in names}
    ksamples = {n: [] for n in names}
    rng = random.Random(seed)
    for _ in range(reps):
        order = names[:]
        rng.shuffle(order)
        for n in order:
            prep, run = arms[n]
            prep()
            t0 = time.perf_counter()
            out = run()
            dt = time.perf_counter() - t0
            del out
            samples[n].append(dt)
            td = kernel_td.get(n)
            if td and "eval_time" in td:
                ksamples[n].append(td["eval_time"])
    med = {n: statistics.median(v) for n, v in samples.items()}
    kmed = {
        n: (statistics.median(v) if v else float("nan")) for n, v in ksamples.items()
    }
    return med, kmed


def time_alone(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
        del out
    return statistics.median(ts)


def as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to_dense() if x.layout != torch.strided else x
    return x.to_torch()


# --------------------------------------------------------------------------- #
# features / decisions (report the mechanism next to the numbers)
# --------------------------------------------------------------------------- #
def cheap_features(A_st, csr, N, C):
    M, J = csr.shape
    nnz = csr.nnz
    operand = J * 4.0 * N
    out = {
        "M": M,
        "J": J,
        "nnz": nnz,
        "deg_per_col": nnz / max(J, 1),
        "deg_per_row": nnz / max(M, 1),
        "operand_bytes": operand,
        "operand_over_C": operand / C,
        "deg_floor_needed": max(tiling._DEG_FLOOR, 2.0 * operand / C),
        "eligible": tiling._eligible(J, nnz, N, C),
    }
    for key, fn in (
        ("locality", lambda: tiling._locality_ratio(A_st, J)),
        ("degree_cv", lambda: tiling._degree_cv(A_st)),
        ("scattered", lambda: tiling._scattered(A_st, J)),
    ):
        try:
            out[key] = fn()
        except Exception:
            out[key] = float("nan")
    return out


def decision_for(A_st, B_st, level):
    """Prime the selector at `level` and report the route it memoized."""
    tiling._decision.clear()
    prev = tiling.get_autotune()
    try:
        tiling.set_autotune(level)
        scorch.matmul(A_st, B_st)
    finally:
        tiling.set_autotune(prev)
    vals = list(tiling._decision.values())
    return vals[0] if vals else ("v2", None)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="main", choices=sorted(GROUPS))
    ap.add_argument("--matrices", nargs="*", default=None, help="subset by name")
    ap.add_argument("--ns", nargs="*", type=int, default=[32, 128, 512])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--levels", nargs="*", default=["off", "analytic", "balanced", "max", "learned"]
    )
    ap.add_argument("--threads", type=int, default=0, help="0 => torch default")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dense", action="store_true", help="also time A_dense @ B, separately")
    ap.add_argument("--no-coo", action="store_true")
    ap.add_argument("--coo-nnz-cap", type=float, default=2.5e7,
                    help="skip torch_coo above this nnz (it is ~10x slower)")
    ap.add_argument("--ref-rows", type=int, default=8192,
                    help="rows sampled for the float64 correctness reference")
    ap.add_argument("--dense-bytes-cap", type=float, default=4e9)
    ap.add_argument("--b-bytes-cap", type=float, default=2.0e9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    C = tiling.query_llc()
    print(f"host          : {platform.node()}  {platform.machine()}")
    print(f"torch         : {torch.__version__}   MKL={torch.backends.mkl.is_available()}")
    print(f"threads       : torch={torch.get_num_threads()}  cpu_count={os.cpu_count()}")
    print(f"scorch        : {scorch.__file__}")
    print(f"gate LLC C    : {C / 1e6:.1f} MB")
    print(
        f"gate consts   : DEG_FLOOR={tiling._DEG_FLOOR}  LOC_MIN={tiling._LOC_MIN}  "
        f"NIJK_MIN={tiling._NIJK_MIN}  LOC_NSAMP={tiling._LOC_NSAMP}  "
        f"CV_NSAMP={tiling._CV_NSAMP}"
    )
    print(f"learned model : {tiling._learned_model_path()}")
    print(f"levels        : {args.levels}   default={tiling.get_autotune()}")
    print(f"reps          : {args.reps} (warmup {args.warmup}), random-permutation interleave")
    print()

    grid = GROUPS[args.group]
    if args.matrices:
        want = set(args.matrices)
        grid = [(n, f) for n, f in grid if n in want or n.split(":", 1)[-1] in want]

    rows = []
    csv_fh = open(args.csv, "w", newline="") if args.csv else None
    writer = None

    for name, loader in grid:
        try:
            csr = loader()
        except Exception as ex:
            print(f"[skip {name}] loader failed: {type(ex).__name__}: {ex}")
            continue
        if csr is None:
            print(f"[skip {name}] not found")
            continue
        M, J = csr.shape
        print(f"\n=== {name}  M={M} J={J} nnz={csr.nnz} deg={csr.nnz / max(M,1):.1f} ===")
        A_st = to_st(csr)
        for N in args.ns:
            if J * N * 4 > args.b_bytes_cap:
                print(f"  N={N:5d}  [skip: B is {J*N*4/1e9:.1f} GB]")
                continue
            g = torch.Generator().manual_seed(args.seed)
            B = torch.rand(J, N, dtype=torch.float32, generator=g)
            B_st = scorch.STensor.from_torch(B)
            # float64 scipy reference. For big matrices a full reference is a
            # single-threaded 100+ GFLOP job, so check a RANDOM ROW SAMPLE instead:
            # A[rows] @ B against out[rows]. Random (not the first block) so a bug
            # anywhere in the row space is still caught.
            if M > args.ref_rows:
                rrng = np.random.default_rng(args.seed)
                ref_rows = np.sort(rrng.choice(M, args.ref_rows, replace=False))
            else:
                ref_rows = None
            csr_ref = csr if ref_rows is None else csr[ref_rows]
            ref = csr_ref.astype(np.float64) @ B.numpy().astype(np.float64)
            refn = np.linalg.norm(ref) + 1e-30

            feats = cheap_features(A_st, csr, N, C)
            routes = {}
            for lvl in args.levels:
                if lvl == "off":
                    routes[lvl] = ("v2", None)
                    continue
                try:
                    routes[lvl] = decision_for(A_st, B_st, lvl)
                except Exception as ex:
                    routes[lvl] = (f"ERR:{type(ex).__name__}", None)
            tiling._decision.clear()

            arms, kernel_td = build_arms(csr, B, A_st, B_st, args)

            # correctness pre-pass (also re-primes each arm's memo)
            errs = {}
            for an, (prep, run) in arms.items():
                prep()
                try:
                    # Row-sample FIRST, then widen to float64. Widening the whole
                    # output first would materialize an 8*M*N buffer -- 6.4 GB on
                    # Flan_1565 at N=512 -- per arm, for a check that only ever
                    # looks at ref_rows of it.
                    r = as_tensor(run()).reshape(M, N)
                    if ref_rows is not None:
                        r = r[ref_rows]
                    errs[an] = float(
                        np.linalg.norm(r.double().numpy() - ref) / refn
                    )
                except Exception as ex:
                    errs[an] = float("nan")
                    print(f"  [{an}] FAILED: {type(ex).__name__}: {ex}")
            gc.collect()

            t, tk = permuted_interleave(arms, args.reps, args.warmup, args.seed, kernel_td)
            base_mkl = min(
                x for k, x in t.items() if k.startswith("mkl_") and math.isfinite(x)
            )
            base_off = t["sc_off"]
            noise = abs(t["aa_control"] / base_off - 1.0)

            print(
                f"  N={N:5d}  operand={feats['operand_bytes']/1e6:8.1f}MB "
                f"({feats['operand_over_C']:5.2f}xC)  deg/col={feats['deg_per_col']:7.1f} "
                f"(need>{feats['deg_floor_needed']:7.1f})  loc={feats['locality']:.3f}  "
                f"cv={feats['degree_cv']:.2f}  eligible={feats['eligible']}  "
                f"scattered={feats['scattered']}  A/A-noise={noise*100:.1f}%"
            )
            order = ["mkl_csr", "mkl_csr32", "torch_coo"]
            order += ["sc_" + l for l in args.levels] + ["aa_control"]
            for an in order:
                if an not in t:
                    continue
                lvl = an[3:] if an.startswith("sc_") else None
                rt = routes.get(lvl, ("", None))
                rtxt = f"{rt[0]}{'' if rt[1] is None else ':' + str(rt[1])}" if lvl else ""
                kt = tk.get(an, float("nan"))
                ktxt = f"kern={kt*1e3:8.3f}" if math.isfinite(kt) else " " * 13
                print(
                    f"       {an:12s} {t[an]*1e3:9.3f} ms  {ktxt}  "
                    f"vs_mkl={base_mkl/t[an]:6.2f}x  vs_off={base_off/t[an]:6.2f}x  "
                    f"relerr={errs[an]:8.1e}  {rtxt}"
                )
                rec = dict(
                    matrix=name,
                    N=N,
                    arm=an,
                    level=lvl or "",
                    route=rt[0] if lvl else "",
                    route_param=("" if rt[1] is None else str(rt[1])),
                    ms=t[an] * 1e3,
                    kernel_ms=kt * 1e3 if math.isfinite(kt) else "",
                    vs_mkl=base_mkl / t[an],
                    vs_off=base_off / t[an],
                    relerr=errs[an],
                    noise_floor=noise,
                    threads=torch.get_num_threads(),
                    llc_bytes=C,
                    **feats,
                )
                rows.append(rec)
                if csv_fh is not None:
                    if writer is None:
                        writer = csvmod.DictWriter(csv_fh, fieldnames=list(rec))
                        writer.writeheader()
                    writer.writerow(rec)
                    csv_fh.flush()

            # dense SGEMM, timed alone (never interleaved: it evicts the whole LLC)
            if args.dense and M * J * 4 <= args.dense_bytes_cap:
                A_dense = torch.from_numpy(csr.toarray())
                td = time_alone(lambda: A_dense @ B, max(3, args.reps // 3), 1)
                print(
                    f"       {'dense_mkl':12s} {td*1e3:9.3f} ms  "
                    f"{' '*13}  vs_mkl={base_mkl/td:6.2f}x  vs_off={base_off/td:6.2f}x"
                    "   [separate pass]"
                )
                rec = dict(
                    matrix=name, N=N, arm="dense_mkl", level="", route="", route_param="",
                    ms=td * 1e3, kernel_ms="", vs_mkl=base_mkl / td, vs_off=base_off / td,
                    relerr=float("nan"), noise_floor=noise,
                    threads=torch.get_num_threads(), llc_bytes=C, **feats,
                )
                rows.append(rec)
                if csv_fh is not None and writer is not None:
                    writer.writerow(rec)
                    csv_fh.flush()
                del A_dense

            del arms, B, B_st
            gc.collect()

    if csv_fh is not None:
        csv_fh.close()
        print(f"\nwrote {args.csv}")

    # ---- summary ---------------------------------------------------------- #
    def geo(xs):
        xs = [x for x in xs if x and math.isfinite(x) and x > 0]
        return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")

    print("\n" + "=" * 82)
    print("GEOMEAN SPEEDUP (vs best MKL sparse arm, and vs scorch level 'off')")
    for an in sorted({r["arm"] for r in rows}):
        xs = [r["vs_mkl"] for r in rows if r["arm"] == an]
        ys = [r["vs_off"] for r in rows if r["arm"] == an]
        print(f"  {an:12s} vs_mkl {geo(xs):6.3f}x   vs_off {geo(ys):6.3f}x   n={len(xs)}")

    print("\nROUTES CHOSEN (level -> route -> count)")
    for lvl in args.levels:
        cnt = {}
        for r in rows:
            if r["level"] == lvl:
                cnt[r["route"]] = cnt.get(r["route"], 0) + 1
        print(f"  {lvl:10s} {cnt}")

    print("\nNO-REGRESSION CHECK (each scorch level vs level 'off', per cell,")
    print("tolerance = max(3%, that cell's A/A noise floor))")
    for k in sorted({r["arm"] for r in rows if r["arm"].startswith("sc_")}):
        if k == "sc_off":
            continue
        cells = [r for r in rows if r["arm"] == k]
        if not cells:
            continue
        w = min(cells, key=lambda r: r["vs_off"])
        tol = 1.0 - max(0.03, w["noise_floor"])
        flag = "OK" if w["vs_off"] >= tol else "REGRESSION"
        print(
            f"  {k:12s} worst {w['vs_off']:.3f}x on {w['matrix']}@N={w['N']} "
            f"(route={w['route']}, A/A noise={w['noise_floor']*100:.1f}%)  {flag}"
        )

    bad = [r for r in rows if isinstance(r["relerr"], float) and math.isfinite(r["relerr"]) and r["relerr"] > 1e-4]
    print(f"\ncorrectness: {len(bad)} cells with relerr > 1e-4")
    for r in bad[:20]:
        print(f"  {r['matrix']}@N={r['N']} {r['arm']} relerr={r['relerr']:.2e}")


if __name__ == "__main__":
    main()
