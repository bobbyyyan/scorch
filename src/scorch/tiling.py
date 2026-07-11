"""Adaptive SpMM tiling selector (runtime dispatch for the drop-in CSR SpMM).

Derived + validated offline (bench/tiling_selector.py + bench_tiling_autotuner.py)
on redwood x86 (probe geomean 1.000 / 25 cells) and Apple M5 (0.999 / 20 cells).
This is the production wiring: it routes scorch.matmul's prebuilt CSR@dense path
(spmm_csr_float_v2) to the column-panel kernel spmm_csr_float_tilej ONLY on shapes
where tiling provably beats v2 — the high-degree, operand-over-LLC thrash regime
(reddit/products-class graphs) — and to v2 (byte-unchanged) everywhere else.

Design (the CLAUDE.md no-regression gate, by construction):
  1. CHEAP O(1) PRE-FILTER (no wavefront W*, which is O(nnz)): from CSR metadata
     alone (J, nnz, degree, N) + the queried LLC size, a shape is tile-j-ELIGIBLE
     iff the dense operand thrashes the LLC (J*4N > C) AND the degree is high enough
     that column-blocking recovers more B-reuse than its output re-traffic costs
     (the thrash-and-tile rule: deg > max(DEG_FLOOR, 2*J*4N/C)). Everything else
     (all GCN-small/AE/FEM-panel/arxiv shapes) is INELIGIBLE -> v2, zero overhead.
  2. FIRST-CALL MICRO-PROBE: for an eligible (matrix-signature, N), measure v2 vs
     tile-j once, memoize the winner (+ its Jc). Subsequent calls route to the
     winner directly. Because v2 is always a probed candidate, the memoized choice
     is never slower than v2 -> no regression even if the analytic pre-filter is
     imprecise (it only decides WHETHER to probe, never forces tile-j). The probe
     cost (a few extra kernel calls) is bounded to big, reused graphs that amortize
     it over an epoch.

Env:
  SCORCH_TILING=0        disable entirely (pure v2 — the pre-selector baseline).
  SCORCH_TILING_PROBE=0  skip the probe; use the analytic pick directly (for A/B).
  SCORCH_LLC_BYTES=<n>   override the queried last-level cache size.
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

_HAS_TILEJ = hasattr(_ops, "spmm_csr_float_tilej")

# memo: signature -> ("v2", None) or ("tilej", Jc)
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


def maybe_dispatch(a, b, result_shape, v2_fn, nthreads: Optional[int]):
    """Return (result_cpp, used_tilej: bool) or None to signal 'use the caller's
    normal v2 path'. Only ever returns tile-j when it has been measured (or the
    analytic pick under SCORCH_TILING_PROBE=0) to be the right choice."""
    if not (_HAS_TILEJ and _ENABLED):
        return None
    if b.dim() != 2:
        return None
    J = int(a.shape[1])
    N = int(b.shape[1])
    idx = a.index.mode_indices
    nnz = int(idx[1][1].numel())
    C = query_llc()
    if not _eligible(J, nnz, N, C):
        return None

    nt = nthreads if nthreads is not None else -1
    Jc = _panel_width(N, C)
    sig = _signature(a, N)
    cached = _decision.get(sig)
    if cached is not None:
        kind, jc = cached
        if kind == "v2":
            return None
        return _ops.spmm_csr_float_tilej(*_tilej_args(a, b, result_shape, jc, nt)), True

    # Locality gate (first time only, then memoized): well-ordered/banded matrices
    # (FEM) pass the degree filter but have no cross-row B-reuse for tile-j to
    # recover — v2 already streams their band from cache. Route them to v2.
    if not _scattered(a, J):
        _decision[sig] = ("v2", None)
        return None

    if not _PROBE:
        # analytic pick: eligibility already implies tile-j is favored.
        _decision[sig] = ("tilej", Jc)
        return _ops.spmm_csr_float_tilej(*_tilej_args(a, b, result_shape, Jc, nt)), True

    # first-call micro-probe: time v2 vs tile-j, keep the winner's result.
    def _time(fn, out_holder):
        fn()  # warmup (also fills caches)
        best = float("inf")
        r = None
        for _ in range(2):
            t0 = time.perf_counter()
            r = fn()
            best = min(best, time.perf_counter() - t0)
        out_holder[0] = r
        return best

    v2_out = [None]
    tj_out = [None]
    t_v2 = _time(lambda: v2_fn(nthreads), v2_out)
    t_tj = _time(lambda: _ops.spmm_csr_float_tilej(*_tilej_args(a, b, result_shape, Jc, nt)), tj_out)
    if t_tj < t_v2:
        _decision[sig] = ("tilej", Jc)
        return tj_out[0], True
    _decision[sig] = ("v2", None)
    return None
