"""Adaptive SpMM tiling selector (runtime dispatch for the drop-in CSR SpMM).

Derived + validated offline (bench/tiling_selector.py + bench_tiling_autotuner.py)
on redwood x86 (probe geomean 1.000 / 25 cells) and Apple M5 (0.999 / 20 cells).
This is the production wiring: it routes scorch.matmul's prebuilt CSR@dense path
(spmm_csr_float_v2) to a column-panel kernel (spmm_csr_float_tilej) — or, for the
very-wide-B tail, the width-panel-relaid 3D kernel (spmm_csr_float_tileijk) — ONLY
on shapes where tiling provably beats v2 (the high-degree, operand-over-LLC thrash
regime: reddit/products-class graphs and the general-library wide-B case), and to
v2 (byte-unchanged) everywhere else.

Design (the CLAUDE.md no-regression gate, by construction):
  1. CHEAP O(1) PRE-FILTER (no wavefront W*, which is O(nnz)): from CSR metadata
     alone (J, nnz, degree, N) + the queried LLC size, a shape is tile-ELIGIBLE
     iff the dense operand thrashes the LLC (J*4N > C) AND the degree is high enough
     that column-blocking recovers more B-reuse than its output re-traffic costs
     (the thrash-and-tile rule: deg > max(DEG_FLOOR, 2*J*4N/C)). Everything else
     (all GCN-small/AE/FEM-panel/arxiv shapes) is INELIGIBLE -> v2, zero overhead.
  2. FIRST-CALL MICRO-PROBE: for an eligible (matrix-signature, N), measure the
     candidate set once, memoize the winner (+ its tile params). Subsequent calls
     route to the winner directly. Because v2 is ALWAYS a probed candidate, the
     memoized choice is never slower than v2 -> no regression even if the analytic
     pre-filter is imprecise (it only decides WHICH candidates to probe, never
     forces a tiled kernel). The probe cost (a few extra kernel calls) is bounded
     to big, reused graphs that amortize it over an epoch.

     Candidate set: {v2, tile-j} for the moderate-N thrash regime; {v2, tile-j,
     tile-ijk} once N is WIDE (>= NIJK_MIN). tile-j's output re-traffic grows ~N^2
     (it re-streams C P=J*4N/C times), so at wide N it erodes; tile-ijk relays B
     into contiguous Nc-wide width-panels so its C-traffic is ~N (linear) at the
     cost of an O(J*N) relayout + re-scanning A nk=N/Nc times. That relayout is
     done INSIDE the kernel per call, so the probe times the FULL cost (relayout +
     compute) honestly against tile-j and v2 — no scorch workload has N>=512, so
     tile-ijk is provably NEUTRAL on everything current and only ever a WIN on the
     general-library wide-B case the probe actually measures a speedup for.

Env:
  SCORCH_TILING=0          disable entirely (pure v2 — the pre-selector baseline).
  SCORCH_TILING_PROBE=0    skip the probe; use the analytic pick directly (for A/B).
  SCORCH_LLC_BYTES=<n>     override the queried last-level cache size.
  SCORCH_TILING_NIJK_MIN=<n>  free-dim width at/above which tile-ijk enters the
                              probe (default 512; > every GCN/AE N so it is inert
                              on current workloads).
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
from typing import Optional, Tuple

import torch
import scorch_ops as _ops

_ENABLED = os.environ.get("SCORCH_TILING", "1") == "1"
_PROBE = os.environ.get("SCORCH_TILING_PROBE", "1") == "1"
# Degree floor for tile-j eligibility. Sits ABOVE the shapes where column-blocking
# loses to v2 — sparse-AE weights (deg~=0.01*out, ~41 for the widest stl10 layer)
# and ogbn-products (deg~52; its 2.5GB operand = ~100 LLC-panels, so the panel
# re-traffic swamps the recovered reuse) — and BELOW the graphs where it wins big
# (reddit deg~493, high-degree scattered deg>=199). This keeps AE/products/arxiv/
# small-GCN from even being probed; the probe is still the ultimate no-regression
# safety net for anything that slips through. Env-overridable for A/B.
_DEG_FLOOR = float(os.environ.get("SCORCH_TILING_DEG_FLOOR", "64"))

# Free-dim width at/above which tile-ijk (B width-panel relayout) joins the probe.
# Default 512 sits ABOVE every current scorch workload (GCN hidden dims 16-256, AE
# batch 256), so tile-ijk is provably inert on all of them and only ever enters the
# probe on the general-library wide-B regime. Env-overridable for A/B.
_NIJK_MIN = int(os.environ.get("SCORCH_TILING_NIJK_MIN", "512"))

_HAS_TILEJ = hasattr(_ops, "spmm_csr_float_tilej")
_HAS_TILEIJK = hasattr(_ops, "spmm_csr_float_tileijk")

# memo: signature -> ("v2", None) | ("tilej", Jc) | ("tileijk", (Nc, Jc))
_decision: dict = {}
_llc_bytes: Optional[int] = None


def query_llc() -> int:
    """Effective last-level cache in bytes, queried from the OS (NO hardcoded
    constant; env override wins). macOS: the P-cluster L2 (hw.perflevel0
    .l2cachesize) — on M5 that 16MB shared L2 is the binding cache for the SpMM
    (Apple does not expose the SLC). Linux: L3 from sysfs. Falls back to a safe
    default if the query fails."""
    global _llc_bytes
    if _llc_bytes is not None:
        return _llc_bytes
    env = os.environ.get("SCORCH_LLC_BYTES")
    if env:
        _llc_bytes = int(env)
        return _llc_bytes
    val = None
    try:
        if platform.system() == "Darwin":
            for key in ("hw.perflevel0.l2cachesize", "hw.l2cachesize"):
                try:
                    out = subprocess.check_output(["sysctl", "-n", key],
                                                  stderr=subprocess.DEVNULL).strip()
                    if out:
                        val = int(out)
                        break
                except Exception:
                    continue
        else:  # Linux: largest cache level in sysfs (L3 if present)
            best = 0
            base = "/sys/devices/system/cpu/cpu0/cache"
            if os.path.isdir(base):
                for entry in os.listdir(base):
                    sz = os.path.join(base, entry, "size")
                    if os.path.isfile(sz):
                        s = open(sz).read().strip()
                        mult = 1024 if s.endswith("K") else (1024 * 1024 if s.endswith("M") else 1)
                        n = int(s.rstrip("KM")) * mult
                        best = max(best, n)
            val = best or None
    except Exception:
        val = None
    _llc_bytes = val or (16 * 1024 * 1024 if platform.system() == "Darwin" else 36 * 1024 * 1024)
    return _llc_bytes


def _panel_width(N: int, C: int) -> int:
    """Contraction-panel width Jc so a panel's B-rows (Jc*4N bytes) fit the LLC."""
    return max(256, int(C // (4 * N)))


def _ijk_params(N: int, M: int, J: int, C: int) -> Tuple[int, int]:
    """Free-dim strip width Nc and contraction-panel width Jc for tile-ijk, from
    the validated cost model (bench_tiling_autotuner.model_costs). Nc is bounded so
    the cache-resident output panel Cp (M*Nc bytes) plus a B column-panel both fit
    the LLC; Jc then sizes the B panel to ~C. Nc is rounded down to a multiple of 16
    (SIMD width), floored at 16, and capped at N."""
    bpanel = min(float(J), C / (4.0 * 64))
    denom = 4.0 * (M + bpanel)
    nc = int(C / denom) if denom > 0 else N
    nc = max(16, (nc // 16) * 16)
    nc = min(N, nc)
    jc = min(J, max(256, int(C / (4.0 * max(1, nc)))))
    return nc, jc


def _signature(a, N: int) -> tuple:
    """Cheap, content-stable key for the sparse operand + free dim. Samples a few
    indptr/indices entries so distinct matrices with equal (M,J,nnz) don't collide;
    stable across re-wrapped STensors of the same CSR (memo survives per-call
    re-wrapping in benchmarks/training loops)."""
    idx = a.index.mode_indices
    pos, crd = idx[1][0], idx[1][1]
    nnz = int(crd.numel())
    M = int(pos.numel()) - 1
    J = int(a.shape[1])
    # sample without materializing: a couple of interior entries
    def s(t, i):
        n = t.numel()
        return int(t[i % n].item()) if n else 0
    return (M, J, nnz, N, s(pos, M // 3), s(pos, 2 * M // 3),
            s(crd, nnz // 3), s(crd, 2 * nnz // 3))


_LOC_MIN = float(os.environ.get("SCORCH_TILING_LOC_MIN", "0.3"))
_LOC_NSAMP = 64


def _eligible(J: int, nnz: int, N: int, C: int) -> bool:
    """Cheap O(1) pre-filter: tile-j can only beat v2 when B thrashes the LLC
    (operand > C) AND there is enough per-column reuse (degree) to recover more
    than the panel re-traffic costs (thrash-and-tile). Degree alone can't tell a
    well-ordered high-degree matrix (FEM) from a scattered one — that needs the
    locality proxy (_scattered), applied only after this passes."""
    if not (_HAS_TILEJ and _ENABLED):
        return False
    operand = J * 4 * N
    if operand <= C:
        return False
    deg = nnz / max(1, J)
    return deg > max(_DEG_FLOOR, 2.0 * operand / C)


def _scattered(a, J: int) -> bool:
    """Cheap sampled LOCALITY proxy (stands in for the O(nnz) wavefront W*). CSR
    rows are column-sorted, so a row's column span is crd[last]-crd[first] in O(1);
    the mean span/J over ~64 sampled rows is ~1 for a scattered matrix (reddit
    0.95, random 0.99) and ~0 for a well-ordered/banded one (FEM cant 0.008,
    band 0.001). tile-j's cross-row B-reuse only exists when access is scattered;
    this keeps FEM/banded matrices (where v2 already streams the band from cache)
    off the tile-j path. Uses a private RNG so it never perturbs global torch RNG."""
    idx = a.index.mode_indices
    pos, crd = idx[1][0], idx[1][1]
    M = int(pos.numel()) - 1
    if M <= 0:
        return False
    n = min(_LOC_NSAMP, M)
    g = torch.Generator().manual_seed(0)
    ridx = torch.randint(0, M, (n,), generator=g)
    b = pos[ridx].to(torch.long)
    e = pos[ridx + 1].to(torch.long)
    nz = e > b
    if not bool(nz.any()):
        return False
    b = b[nz]
    e = e[nz]
    span = (crd[e - 1].to(torch.long) - crd[b].to(torch.long)).float().mean().item()
    return (span / max(1, J)) > _LOC_MIN


def is_candidate(a, b) -> bool:
    """Cheapest possible O(1) pre-gate, called by ops.matmul BEFORE building the
    dispatch closure — so an ineligible shape (all GCN-small/AE/FEM/arxiv) pays
    only a few int comparisons and returns to the byte-identical v2 path with no
    closure, no dict, no signature. This is what makes the wiring provably neutral
    on everything the selector does not touch."""
    if not (_HAS_TILEJ and _ENABLED):
        return False
    if b.dim() != 2:
        return False
    J = int(a.shape[1])
    N = int(b.shape[1])
    nnz = int(a.index.mode_indices[1][1].numel())
    return _eligible(J, nnz, N, query_llc())


def _tilej_args(a, b, result_shape, Jc, nthreads):
    return ([result_shape, a.shape, a.index.mode_indices, a.values,
             b.shape, b.index.mode_indices, b.values, Jc, nthreads])


def _tileijk_args(a, b, result_shape, Nc, Jc, nthreads):
    return ([result_shape, a.shape, a.index.mode_indices, a.values,
             b.shape, b.index.mode_indices, b.values, Nc, Jc, nthreads])


def _dispatch_decision(a, b, result_shape, kind, param, nt):
    """Run the memoized winner. Returns (result_cpp, True) for a tiled kernel or
    None (== 'use the caller's byte-identical v2 path') for kind 'v2'."""
    if kind == "tilej":
        return _ops.spmm_csr_float_tilej(*_tilej_args(a, b, result_shape, param, nt)), True
    if kind == "tileijk":
        Nc, Jc = param
        return _ops.spmm_csr_float_tileijk(
            *_tileijk_args(a, b, result_shape, Nc, Jc, nt)), True
    return None  # v2


def _ijk_beats_tilej_bytes(N, M, J, nnz, Jc_tj, Nc, C):
    """Analytic (_PROBE=0) tiebreak: predicted DRAM bytes for tile-ijk (INCLUDING
    the one-shot O(J*N) relayout: read B + write the relaid strips) vs tile-j (which
    re-streams C P=ceil(J/Jc) times, so ~N^2). Mirrors bench_tiling_autotuner
    .model_costs; used only when the probe is disabled."""
    BN = 4.0 * N
    Cwr = M * BN
    A = 8.0 * nnz
    P = max(1, -(-J // max(1, Jc_tj)))            # ceil(J / Jc_tj)
    tj = J * BN + P * 2 * Cwr + A + P * M * 4
    nk = max(1, -(-N // max(1, Nc)))              # ceil(N / Nc)
    ijk = J * BN + Cwr + A * nk + 2 * J * BN      # compute + relayout
    return ijk < tj


def maybe_dispatch(a, b, result_shape, v2_fn, nthreads: Optional[int],
                   time_dict: Optional[dict] = None):
    """Return (result_cpp, used_tiled: bool) or None to signal 'use the caller's
    normal v2 path'. Only ever returns a tiled kernel (tile-j / tile-ijk) when it
    has been MEASURED (or the analytic pick under SCORCH_TILING_PROBE=0) to be the
    right choice; v2 is always a probe candidate, so the memoized route is never
    slower than v2 -> no regression by construction.

    When a tiled kernel is returned and `time_dict` is supplied, its "eval_time" is
    populated (the winning kernel's measured/timed duration) so the tiled route
    honors the same timing contract as the caller's v2 fallthrough
    (execute_prebuilt_binary_kernel). When this returns None the caller runs v2 and
    populates time_dict itself."""
    if not (_HAS_TILEJ and _ENABLED):
        return None
    if b.dim() != 2:
        return None
    J = int(a.shape[1])
    N = int(b.shape[1])
    M = int(a.shape[0])
    idx = a.index.mode_indices
    nnz = int(idx[1][1].numel())
    C = query_llc()
    if not _eligible(J, nnz, N, C):
        return None

    nt = nthreads if nthreads is not None else -1
    Jc = _panel_width(N, C)
    sig = _signature(a, N)

    def _timed_dispatch(kind, param):
        """Run the memoized winner, recording its wall time into time_dict."""
        t0 = time.perf_counter()
        out = _dispatch_decision(a, b, result_shape, kind, param, nt)
        if time_dict is not None:
            time_dict["eval_time"] = time.perf_counter() - t0
        return out

    cached = _decision.get(sig)
    if cached is not None:
        return _timed_dispatch(cached[0], cached[1])

    # Locality gate (first time only, then memoized): well-ordered/banded matrices
    # (FEM) pass the degree filter but have no cross-row B-reuse for the tile path
    # to recover — v2 already streams their band from cache. Route them to v2.
    if not _scattered(a, J):
        _decision[sig] = ("v2", None)
        return None

    # tile-ijk (B width-panel relayout) joins the candidate set only once N is wide
    # enough that tile-j's ~N^2 output re-traffic erodes (>= NIJK_MIN, above every
    # current workload) AND the width-strip actually splits N (Nc < N -> >1 strip).
    ijk: Optional[Tuple[int, int]] = None
    if _HAS_TILEIJK and N >= _NIJK_MIN:
        Nc, Jc_ijk = _ijk_params(N, M, J, C)
        if Nc < N:
            ijk = (Nc, Jc_ijk)

    if not _PROBE:
        # analytic pick: eligibility implies tile-j > v2; choose tile-ijk over
        # tile-j only if the byte model (relayout included) says it is cheaper.
        if ijk is not None and _ijk_beats_tilej_bytes(N, M, J, nnz, Jc, ijk[0], C):
            _decision[sig] = ("tileijk", ijk)
        else:
            _decision[sig] = ("tilej", Jc)
        return _timed_dispatch(_decision[sig][0], _decision[sig][1])

    # first-call micro-probe: time every candidate, keep the fastest's result.
    # Because the relayout runs INSIDE spmm_csr_float_tileijk, timing that call
    # accounts for the relayout honestly against tile-j and v2.
    def _time(fn, out_holder):
        fn()  # warmup (also fills caches / builds any relaid buffer once)
        best = float("inf")
        r = None
        for _ in range(2):
            t0 = time.perf_counter()
            r = fn()
            best = min(best, time.perf_counter() - t0)
        out_holder[0] = r
        return best

    cands = [
        ("v2", None, lambda: v2_fn(nthreads)),
        ("tilej", Jc,
         lambda: _ops.spmm_csr_float_tilej(*_tilej_args(a, b, result_shape, Jc, nt))),
    ]
    if ijk is not None:
        cands.append(("tileijk", ijk,
                      lambda: _ops.spmm_csr_float_tileijk(
                          *_tileijk_args(a, b, result_shape, ijk[0], ijk[1], nt))))

    best_t = float("inf")
    best_kind, best_param, best_out = "v2", None, None
    for kind, param, fn in cands:
        holder = [None]
        t = _time(fn, holder)
        if t < best_t:
            best_t, best_kind, best_param, best_out = t, kind, param, holder[0]

    _decision[sig] = (best_kind, best_param)
    if best_kind == "v2":
        return None  # caller runs v2 + populates time_dict itself
    if time_dict is not None:
        time_dict["eval_time"] = best_t  # the winning kernel's measured time
    return best_out, True
