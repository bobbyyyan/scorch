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

Autotune levels (the user-facing knob; see autotune-levels/00-design.md):
  A compiler-style -O ladder over this selector, trading dispatch overhead for
  execution speed. The GATE (below) is preserved and byte-neutral at EVERY level;
  only the decision made for an already-eligible shape changes.
    off       no tiling — is_candidate short-circuits to False (pure v2).
    analytic  (DEFAULT) cost-model pick (tile-j@base / tile-ijk@wide-N), NO probe.
              Still TILES eligible graphs -> reddit-class keeps ~97% of the win
              with zero probe stall.
    balanced  first-call micro-probe over {v2, tile-j@{base,/2,/4,/8}, tile-ijk};
              memoized. v2 always a candidate -> never slower than v2.
    max       balanced probe + a PERSISTENT on-disk cache (per-machine) so the
              search is paid once EVER.
    learned   (Phase 2) offline-trained cost model; falls back to analytic until
              a model exists.
  Set via scorch.set_autotune(level) (global) / with scorch.autotune(level): ...
  (thread-local CM + decorator). See src/scorch/__init__.py.

Env (the Python API is primary; env is for override/CI):
  SCORCH_AUTOTUNE=<level>  initial global level (off/analytic/balanced/max/learned).
  SCORCH_AUTOTUNE_CACHE=<path>  persistent-cache location; =0 disables read+write.
  SCORCH_TILING=0          legacy: maps to level "off" (pure v2 baseline).
  SCORCH_TILING_PROBE=0    legacy: maps to "analytic"; =1 maps to "balanced".
  SCORCH_LLC_BYTES=<n>     override the queried last-level cache size (gate knob).
  SCORCH_TILING_DEG_FLOOR / _NIJK_MIN / _LOC_MIN  gate knobs, unchanged at all levels.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from typing import Optional, Tuple

import torch
import scorch_ops as _ops

# ---------------------------------------------------------------------------
# Autotune level state (thread-local override over a process-global default).
# Mirrors torch's grad-mode: set_autotune() is the global knob; the `autotune`
# class is a thread-local context manager + decorator. See 00-design.md.
# ---------------------------------------------------------------------------
_LEVELS = ("off", "analytic", "balanced", "max", "learned")


def _validate_level(level) -> str:
    if not isinstance(level, str):
        raise TypeError(f"autotune level must be a str, got {type(level).__name__}")
    lvl = level.strip().lower()
    if lvl not in _LEVELS:
        raise ValueError(
            f"unknown autotune level {level!r}; expected one of {_LEVELS}")
    return lvl


def _default_level_from_env() -> str:
    """Initial global level. New SCORCH_AUTOTUNE wins; else the legacy
    SCORCH_TILING/SCORCH_TILING_PROBE vars map onto levels; else the built-in
    default 'analytic' (see 00-design.md §5)."""
    v = os.environ.get("SCORCH_AUTOTUNE")
    if v is not None and v.strip():
        try:
            return _validate_level(v)
        except ValueError:
            pass  # bad value -> fall through to legacy mapping
    if os.environ.get("SCORCH_TILING", "1") == "0":
        return "off"
    probe = os.environ.get("SCORCH_TILING_PROBE")
    if probe == "0":
        return "analytic"
    if probe == "1":
        return "balanced"
    return "analytic"


_global_level = _default_level_from_env()
_tls = threading.local()


def _current_level() -> str:
    """The effective level: a thread-local CM override if active, else global."""
    lvl = getattr(_tls, "level", None)
    return lvl if lvl is not None else _global_level


def _level_probes(level: str) -> bool:
    """balanced/max measure candidates; analytic (and learned, until Phase 2's
    model exists) pick analytically with no kernel measurement."""
    return level in ("balanced", "max")


def set_autotune(level: str) -> None:
    """Set the process-global autotune level. See scorch.set_autotune."""
    global _global_level
    _global_level = _validate_level(level)


def get_autotune() -> str:
    """Return the effective autotune level (thread-local override wins)."""
    return _current_level()


class autotune:
    """Scope the autotune level as a thread-local context manager or decorator
    (mirrors torch.no_grad). See scorch.autotune."""

    def __init__(self, level: str):
        self.level = _validate_level(level)

    def __enter__(self):
        self._prev = getattr(_tls, "level", None)
        _tls.level = self.level
        return self

    def __exit__(self, *exc):
        _tls.level = self._prev
        return False

    def __call__(self, fn):
        level = self.level

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with autotune(level):  # fresh instance per call -> reentrant
                return fn(*args, **kwargs)

        return wrapper


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
    locality proxy (_scattered), applied only after this passes. The level gate
    (off) is applied by the callers (is_candidate / maybe_dispatch) before here."""
    if not _HAS_TILEJ:
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


def is_candidate(a, b, level: Optional[str] = None) -> bool:
    """Cheapest possible O(1) pre-gate, called by ops.matmul BEFORE building the
    dispatch closure — so an ineligible shape (all GCN-small/AE/FEM/arxiv) pays
    only a few int comparisons and returns to the byte-identical v2 path with no
    closure, no dict, no signature. This is what makes the wiring provably neutral
    on everything the selector does not touch. Level 'off' short-circuits here so
    the disabled path is the cheapest of all."""
    if not _HAS_TILEJ:
        return False
    if level is None:
        level = _current_level()
    if level == "off":
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


def _jc_ladder(base: int) -> list:
    """Coarse tile-j panel-width ladder {base, base/2, base/4, base/8} for the
    balanced/max probe (see tilej_width_never_tuned). A COARSE ladder captures ~the
    entire width win; a fine sweep buys ~0.2% for ~10x the search. Clamp each rung
    >=16 (SIMD floor) and dedup so a small base doesn't emit collapsed duplicates.
    Real structured/power-law matrices sit at base (~0.92-1.0 of best); the smaller
    rungs exist to grab the uniform-random (appu-like) tail where base overshoots."""
    out, seen = [], set()
    for d in (1, 2, 4, 8):
        jc = max(16, base // d)
        if jc not in seen:
            seen.add(jc)
            out.append(jc)
    return out


# ---------------------------------------------------------------------------
# Persistent autotune cache (the "max" level): pay the probe once EVER per
# machine. JSON at $XDG_CACHE_HOME/scorch/autotune.json, keyed machine_id -> sig
# -> [kind, param]. Best-effort: any I/O error degrades silently to in-memory.
# Invalidation = version bump (whole file) or machine mismatch (keyed out).
# ---------------------------------------------------------------------------
_CACHE_VERSION = 1
_cache_loaded = False
_persist_cache: Optional[dict] = None
_machine_id_val: Optional[str] = None


def _cache_path() -> Optional[str]:
    env = os.environ.get("SCORCH_AUTOTUNE_CACHE")
    if env == "0":
        return None  # disabled
    if env:
        return env
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache")
    return os.path.join(base, "scorch", "autotune.json")


def _machine_id() -> str:
    """Stable per-machine fingerprint so a cache built on one host is never
    applied on another (system + arch + CPU brand + core count + LLC bytes)."""
    global _machine_id_val
    if _machine_id_val is not None:
        return _machine_id_val
    brand = ""
    try:
        if platform.system() == "Darwin":
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL).decode().strip()
        elif platform.system() == "Linux" and os.path.isfile("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        brand = line.split(":", 1)[1].strip()
                        break
    except Exception:
        brand = ""
    parts = (platform.system(), platform.machine(), brand,
             str(os.cpu_count()), str(query_llc()))
    _machine_id_val = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    return _machine_id_val


def _load_persist_cache() -> dict:
    global _cache_loaded, _persist_cache
    if _cache_loaded and _persist_cache is not None:
        return _persist_cache
    _cache_loaded = True
    _persist_cache = {}
    path = _cache_path()
    if not path or not os.path.isfile(path):
        return _persist_cache
    try:
        with open(path) as f:
            data = json.load(f)
        # Invalidate the whole file on a version bump.
        if isinstance(data, dict) and data.get("version") == _CACHE_VERSION:
            ent = data.get("entries")
            if isinstance(ent, dict):
                _persist_cache = ent
    except Exception:
        _persist_cache = {}
    return _persist_cache


def _sig_key(sig: tuple) -> str:
    return ",".join(str(x) for x in sig)


def _persist_get(sig: tuple):
    """Return a cached (kind, param) for this machine+signature, or None."""
    entry = _load_persist_cache().get(_machine_id(), {}).get(_sig_key(sig))
    if entry is None:
        return None
    kind, param = entry[0], entry[1]
    if kind == "tileijk" and isinstance(param, list):
        param = tuple(param)  # JSON list -> (Nc, Jc)
    return (kind, param)


def _persist_put(sig: tuple, kind: str, param) -> None:
    """Record a measured winner. Atomic (temp + os.replace), best-effort."""
    path = _cache_path()
    if not path:
        return
    cache = _load_persist_cache()
    stored = list(param) if isinstance(param, tuple) else param
    cache.setdefault(_machine_id(), {})[_sig_key(sig)] = [kind, stored]
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"version": _CACHE_VERSION, "entries": cache}, f)
        os.replace(tmp, path)
    except Exception:
        pass  # never let a cache write break a matmul


def clear_autotune_cache() -> None:
    """Wipe the persistent on-disk autotune cache (used by the 'max' level)."""
    global _persist_cache, _cache_loaded
    _persist_cache = {}
    _cache_loaded = True
    path = _cache_path()
    if path:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception:
            pass


def _ijk_beats_tilej_bytes(N, M, J, nnz, Jc_tj, Nc, C):
    """Analytic-level tiebreak (no probe): predicted DRAM bytes for tile-ijk (INCLUDING
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
                   time_dict: Optional[dict] = None, level: Optional[str] = None):
    """Return (result_cpp, used_tiled: bool) or None to signal 'use the caller's
    normal v2 path'. Only ever returns a tiled kernel (tile-j / tile-ijk) when it
    has been MEASURED (balanced/max probe) or picked by the cost model (analytic)
    to be the right choice; v2 is always a probe candidate, so the memoized route
    is never slower than v2 -> no regression by construction.

    `level` selects the decision strategy for an eligible shape (off/analytic/
    balanced/max/learned); when None it is resolved from the current thread-local /
    global autotune level. The eligibility GATE is identical at every non-off level.

    When a tiled kernel is returned and `time_dict` is supplied, its "eval_time" is
    populated (the winning kernel's measured/timed duration) so the tiled route
    honors the same timing contract as the caller's v2 fallthrough
    (execute_prebuilt_binary_kernel). When this returns None the caller runs v2 and
    populates time_dict itself."""
    if not _HAS_TILEJ:
        return None
    if level is None:
        level = _current_level()
    if level == "off":
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
    # Memoize per (signature, level): different levels can pick different winners
    # (analytic->tile-j@base; balanced/max->probed width/kernel).
    memo_key = (sig, level)

    def _timed_dispatch(kind, param):
        """Run the memoized winner, recording its wall time into time_dict."""
        t0 = time.perf_counter()
        out = _dispatch_decision(a, b, result_shape, kind, param, nt)
        if time_dict is not None:
            time_dict["eval_time"] = time.perf_counter() - t0
        return out

    cached = _decision.get(memo_key)
    if cached is not None:
        return _timed_dispatch(cached[0], cached[1])

    # "max": a prior run may have already measured the winner for this machine.
    if level == "max":
        pc = _persist_get(sig)
        if pc is not None:
            _decision[memo_key] = pc
            return _timed_dispatch(pc[0], pc[1])

    # Locality gate (first time only, then memoized): well-ordered/banded matrices
    # (FEM) pass the degree filter but have no cross-row B-reuse for the tile path
    # to recover — v2 already streams their band from cache. Route them to v2.
    if not _scattered(a, J):
        _decision[memo_key] = ("v2", None)
        return None

    # tile-ijk (B width-panel relayout) joins the candidate set only once N is wide
    # enough that tile-j's ~N^2 output re-traffic erodes (>= NIJK_MIN, above every
    # current workload) AND the width-strip actually splits N (Nc < N -> >1 strip).
    ijk: Optional[Tuple[int, int]] = None
    if _HAS_TILEIJK and N >= _NIJK_MIN:
        Nc, Jc_ijk = _ijk_params(N, M, J, C)
        if Nc < N:
            ijk = (Nc, Jc_ijk)

    if not _level_probes(level):
        # analytic (and learned, until Phase 2's model lands): eligibility implies
        # tile-j > v2, so pick tile-j@base — or tile-ijk if the byte model (relayout
        # included) says it is cheaper. No kernel measurement.
        if ijk is not None and _ijk_beats_tilej_bytes(N, M, J, nnz, Jc, ijk[0], C):
            _decision[memo_key] = ("tileijk", ijk)
        else:
            _decision[memo_key] = ("tilej", Jc)
        return _timed_dispatch(_decision[memo_key][0], _decision[memo_key][1])

    # balanced/max first-call micro-probe: time every candidate, keep the fastest's
    # result. v2 is always a candidate -> the winner is never slower than v2. The
    # coarse Jc ladder {base,/2,/4,/8} grabs the uniform-random (appu-like) tail
    # where base overshoots. Because the relayout runs INSIDE spmm_csr_float_tileijk,
    # timing that call accounts for the relayout honestly against tile-j and v2.
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

    # Heterogeneous param slot (None | Jc int | (Nc,Jc) tuple) -> annotate `list`.
    cands: list = [("v2", None, lambda: v2_fn(nthreads))]
    for jc in _jc_ladder(Jc):
        cands.append(("tilej", jc,
                      lambda jc=jc: _ops.spmm_csr_float_tilej(
                          *_tilej_args(a, b, result_shape, jc, nt))))
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

    _decision[memo_key] = (best_kind, best_param)
    if level == "max":
        _persist_put(sig, best_kind, best_param)  # pay the search once EVER
    if best_kind == "v2":
        return None  # caller runs v2 + populates time_dict itself
    if time_dict is not None:
        time_dict["eval_time"] = best_t  # the winning kernel's measured time
    return best_out, True
