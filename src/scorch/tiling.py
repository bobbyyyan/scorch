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
    analytic  (DEFAULT) cost-model pick (tile-j@base / tile-ijk@wide-N), then ONE
              v2-confirm before the pick is memoized: 6 kernel invocations against
              balanced's 18. It used to commit unmeasured, which shipped six
              tiled-route regressions over a 236-cell grid (worst 0.373x of untiled);
              balanced, same gate but timing v2, had none. Still TILES eligible
              graphs -> reddit-class keeps the win.
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
  SCORCH_AUTOTUNE_CONFIRM=0  skip the one-shot v2-confirm (restores the pre-2026-08
                             unmeasured analytic/learned behaviour; for A/B only).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import platform
import subprocess
import threading
import time
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
import scorch_ops as _ops

from .compiler.scheduler import RelayoutSpec, Schedule, TileSpec
from .plan import invalidate_all as _plan_invalidate_all

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
        raise ValueError(f"unknown autotune level {level!r}; expected one of {_LEVELS}")
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
    """The effective level: a thread-local CM override if active, else global.

    Read through ``__dict__.get`` rather than ``getattr(_tls, "level", None)``, and
    not as a style choice: with no override active -- the overwhelmingly common case,
    since an override only exists inside a ``set_autotune`` context manager --
    ``getattr`` with a default raises and catches an AttributeError internally, which
    costs 0.132 us against 0.029 us for the dict lookup (M5, 300k calls, minimum of
    5 batches). Every prebuilt CSR@dense product reads this through ``tiling_gate``,
    so the 0.10 us is on the dispatch path of every SpMM the library serves.

    ``threading.local``'s ``__dict__`` is the per-thread mapping, so this returns
    exactly what the ``getattr`` returned, in every thread, set or unset."""
    lvl = _tls.__dict__.get("level")
    return lvl if lvl is not None else _global_level


def _level_probes(level: str) -> bool:
    """Does this level SEARCH the candidate ladder?

    balanced/max time every candidate; analytic and learned pick one from a cost
    model. Both kinds then measure the pick against v2 once (see maybe_dispatch), so
    this is about search breadth, not about whether any kernel is timed."""
    return level in ("balanced", "max")


def set_autotune(level: str) -> None:
    """Set the process-global autotune level for the SpMM tiling selector.

    Selects, for the whole process, how ``scorch.matmul``'s drop-in
    **CSR-sparse x dense** SpMM path is dispatched. Autotuning affects *only*
    that path — it does not touch einsum, other operations, or the general JIT
    compiler. On shapes the selector does not target (essentially all GCN,
    autoencoder, attention and FEM workloads, where the dense operand fits the
    last-level cache) every level dispatches to the byte-identical default
    kernel, so the level is a no-op there by design.

    The level is a compiler-style ``-O`` ladder trading dispatch overhead for
    execution speed. The eligibility gate is identical at every non-``off``
    level; only the decision made for an already-eligible shape changes:

    - ``"off"`` — no tiling; the pure default-kernel baseline.
    - ``"analytic"`` — **default.** Cost-model pick, confirmed against the default
      kernel once per shape, so it can never be slower than not tiling.
    - ``"balanced"`` — first-call micro-probe over the candidate kernels,
      memoized in-process; the default kernel is always a candidate, so the
      pick is never slower than the baseline.
    - ``"max"`` — the ``balanced`` probe plus a persistent on-disk cache, so the
      search is paid once per machine.
    - ``"learned"`` — experimental; an offline-trained cost model. Falls back to
      ``analytic`` unless a per-machine model file is installed.

    This is the process-global default; use :class:`autotune` for a
    thread-local, scoped override, and :func:`get_autotune` to read back the
    effective level.

    Parameters
    ----------
    level : str
        One of ``"off"``, ``"analytic"``, ``"balanced"``, ``"max"``,
        ``"learned"``. Case- and whitespace-insensitive.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``level`` is a string but not one of the allowed levels.
    TypeError
        If ``level`` is not a string.

    See Also
    --------
    get_autotune : Read the effective level.
    autotune : Scoped (thread-local) context-manager / decorator override.
    clear_autotune_cache : Wipe the persistent ``"max"`` cache.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> scorch.set_autotune("max")
    >>> scorch.get_autotune()
    'max'
    >>> scorch.set_autotune("off")   # pure baseline, no tiling
    """
    global _global_level
    _global_level = _validate_level(level)
    # Cached call plans carry this selector's verdict, so a policy change retires
    # them (scorch.plan.GENERATION).
    _plan_invalidate_all()


def get_autotune() -> str:
    """Return the effective SpMM autotune level currently in force.

    Resolves the level exactly as the dispatcher does: an active thread-local
    override (from an :class:`autotune` context manager or decorator) wins;
    otherwise the process-global level set by :func:`set_autotune` (or its
    environment / built-in default, ``"analytic"``) is returned.

    Returns
    -------
    str
        One of ``"off"``, ``"analytic"``, ``"balanced"``, ``"max"``,
        ``"learned"``.

    See Also
    --------
    set_autotune : Set the process-global level.
    autotune : Scoped (thread-local) override.

    Examples
    --------
    >>> import scorch
    >>> scorch.set_autotune("balanced")
    >>> with scorch.autotune("max"):
    ...     scorch.get_autotune()      # the scoped override wins
    'max'
    >>> scorch.get_autotune()          # back to the global level
    'balanced'
    """
    return _current_level()


class autotune:
    """Scope the SpMM autotune level, as a context manager or a decorator.

    Mirrors ``torch.no_grad()``: constructing ``autotune(level)`` and entering
    it installs a **thread-local** override of the autotune level for the
    duration of the block, restoring the previous value on exit. The override is
    per-thread and nests correctly; it does not affect other threads or the
    process-global default set by :func:`set_autotune`.

    Like the rest of the autotune API, this affects only ``scorch.matmul``'s
    CSR-sparse x dense SpMM dispatch; see :func:`set_autotune` for the level
    ladder and scope.

    Two usage forms:

    - **Context manager** — ``with scorch.autotune("max"): ...`` sets the
      override for the block and restores the prior level on exit.
    - **Decorator** — ``@scorch.autotune("balanced")`` on a function runs each
      call inside a fresh ``autotune(level)`` scope (reentrant).

    Parameters
    ----------
    level : str
        One of ``"off"``, ``"analytic"``, ``"balanced"``, ``"max"``,
        ``"learned"``. Validated at construction (case- / whitespace-
        insensitive).

    Raises
    ------
    ValueError
        If ``level`` is a string but not an allowed level.
    TypeError
        If ``level`` is not a string.

    See Also
    --------
    set_autotune : Set the process-global level.
    get_autotune : Read the effective level.

    Examples
    --------
    >>> import torch
    >>> import scorch
    >>> A = scorch.from_torch(
    ...     torch.tensor([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]]), "A"
    ... )
    >>> B = torch.randn(3, 128)
    >>> with scorch.autotune("max"):     # scoped override
    ...     C = scorch.matmul(A, B)
    >>> @scorch.autotune("balanced")     # decorator form
    ... def run(A, B):
    ...     return scorch.matmul(A, B)
    """

    def __init__(self, level: str):
        self.level = _validate_level(level)

    def __enter__(self):
        self._prev = getattr(_tls, "level", None)
        _tls.level = self.level
        _plan_invalidate_all()  # see set_autotune
        return self

    def __exit__(self, *exc):
        _tls.level = self._prev
        _plan_invalidate_all()
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
                    out = subprocess.check_output(
                        ["sysctl", "-n", key], stderr=subprocess.DEVNULL
                    ).strip()
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
                        mult = (
                            1024
                            if s.endswith("K")
                            else (1024 * 1024 if s.endswith("M") else 1)
                        )
                        n = int(s.rstrip("KM")) * mult
                        best = max(best, n)
            val = best or None
    except Exception:
        val = None
    _llc_bytes = val or (
        16 * 1024 * 1024 if platform.system() == "Darwin" else 36 * 1024 * 1024
    )
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
    idx = a._native_mode_indices()
    pos, crd = idx[1][0], idx[1][1]
    nnz = int(crd.numel())
    M = int(pos.numel()) - 1
    J = int(a.shape[1])

    # sample without materializing: a couple of interior entries
    def s(t, i):
        n = t.numel()
        return int(t[i % n].item()) if n else 0

    return (
        M,
        J,
        nnz,
        N,
        s(pos, M // 3),
        s(pos, 2 * M // 3),
        s(crd, nnz // 3),
        s(crd, 2 * nnz // 3),
    )


_LOC_MIN = float(os.environ.get("SCORCH_TILING_LOC_MIN", "0.3"))
_LOC_NSAMP = 64


def _operand_over_cache(J: int, N: int, C: int) -> bool:
    """Does the dense operand overflow the last-level cache?

    The physics boundary both gates rest on -- if B fits in cache, streaming it costs
    nothing extra and no tiling can help. Split out because it needs no nnz, which
    lets ``is_candidate`` answer the 99% case (every GCN-small, autoencoder and
    attention shape) on two int operations, without the attribute chain and the pybind
    ``numel()`` call that reading nnz off the index arrays costs."""
    if not _HAS_TILEJ:
        return False
    return J * 4 * N > C


def _eligible(J: int, nnz: int, N: int, C: int) -> bool:
    """Cheap O(1) pre-filter: tile-j can only beat v2 when B thrashes the LLC
    (operand > C) AND there is enough per-column reuse (degree) to recover more
    than the panel re-traffic costs (thrash-and-tile). Degree alone can't tell a
    well-ordered high-degree matrix (FEM) from a scattered one — that needs the
    locality proxy (_scattered), applied only after this passes. The level gate
    (off) is applied by the callers (is_candidate / maybe_dispatch) before here."""
    return _operand_over_cache(J, N, C) and _degree_pays(J, nnz, N, C)


def _degree_pays(J: int, nnz: int, N: int, C: int) -> bool:
    """The nnz-dependent half of the analytic gate: is there enough per-column reuse
    (degree) to recover more than the panel re-traffic costs?

    Assumes the operand-over-cache boundary has already passed, so a caller that
    tested it -- ``is_candidate`` does, to avoid reading nnz on the 99% that fail --
    is not made to test it twice."""
    return nnz / max(1, J) > max(_DEG_FLOOR, 2.0 * (J * 4 * N) / C)


def _locality_ratio(a, J: int) -> float:
    """Cheap sampled LOCALITY proxy in [0,1] (stands in for the O(nnz) wavefront W*).
    CSR rows are column-sorted, so a row's column span is crd[last]-crd[first] in
    O(1); the mean span/J over ~64 sampled rows is ~1 for a scattered matrix (reddit
    0.95, random 0.99) and ~0 for a well-ordered/banded one (FEM cant 0.008, band
    0.001). This is BOTH the analytic gate (_scattered) and a learned-model feature.
    Uses a private RNG so it never perturbs global torch RNG."""
    idx = a._native_mode_indices()
    pos, crd = idx[1][0], idx[1][1]
    M = int(pos.numel()) - 1
    if M <= 0:
        return 0.0
    n = min(_LOC_NSAMP, M)
    g = torch.Generator().manual_seed(0)
    ridx = torch.randint(0, M, (n,), generator=g)
    b = pos[ridx].to(torch.long)
    e = pos[ridx + 1].to(torch.long)
    nz = e > b
    if not bool(nz.any()):
        return 0.0
    b = b[nz]
    e = e[nz]
    span = (crd[e - 1].to(torch.long) - crd[b].to(torch.long)).float().mean().item()
    return float(span / max(1, J))


def _scattered(a, J: int) -> bool:
    """tile-j's cross-row B-reuse only exists when access is scattered; this keeps
    FEM/banded matrices (where v2 already streams the band from cache) off the
    tile-j path. The analytic/balanced/max locality gate; learned relaxes it."""
    return _locality_ratio(a, J) > _LOC_MIN


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
    C = query_llc()
    # Answer the 99% here, BEFORE touching an index array: both gates below start with
    # this same test, and nnz is only needed if it passes. Reading nnz costs an
    # attribute chain plus a pybind numel() call -- about a quarter of this function --
    # and every GCN-small / AE / attention shape fails on the two int operations above
    # it. The gates re-test the boundary because maybe_dispatch also calls them with
    # nnz already in hand for the signature.
    if not _operand_over_cache(J, N, C):
        return False
    # learned widens the gate (operand>C only) ONLY when opted-in (SCORCH_AUTOTUNE_WIDEN=1)
    # AND a per-machine model is loaded. DEFAULT: learned uses the analytic gate (no
    # widening) and only improves the within-gate pick.
    if level == "learned" and _LEARNED_WIDEN and _load_learned_model() is not None:
        return _eligible_learned(J, N, C)
    nnz = int(a.storage._mode_indices[1][1].numel())
    return _degree_pays(J, nnz, N, C)


def decided(a, n: int, level: Optional[str] = None, baseline_tag: str = "v2"):
    """The winner this selector has already memoized for ``a @ B`` at free dim ``n``
    against baseline ``baseline_tag``, as ``(kind, param)``, or ``None`` if it has
    not decided one.

    A read of the memo, never a decision: no gate, no probe, no cost model, no
    write. ``ops.matmul`` uses it to hand the memoized verdict to a call plan
    (plan.py), so a planned call runs exactly the kernel the selector chose for
    that shape — including its measured panel widths — without consulting the
    selector again. ``None`` covers every case where the ordinary path would have
    run v2: level off, an ineligible shape, or a first call whose probe has not
    landed yet."""
    if not _HAS_TILEJ:
        return None
    if level is None:
        level = _current_level()
    if level == "off":
        return None
    return _decision.get((_signature(a, int(n)), level, baseline_tag))


def _tilej_args(a, b, result_shape, Jc, nthreads):
    return [
        result_shape,
        a.shape,
        a._native_mode_indices(),
        a.values,
        b.shape,
        b._native_mode_indices(),
        b.values,
        Jc,
        nthreads,
    ]


def _tileijk_args(a, b, result_shape, Nc, Jc, nthreads):
    return [
        result_shape,
        a.shape,
        a._native_mode_indices(),
        a.values,
        b.shape,
        b._native_mode_indices(),
        b.values,
        Nc,
        Jc,
        nthreads,
    ]


def _compiler_structural_name(field: str, value: object) -> str:
    """Validate one user-provided tensor or logical-index name."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_tuner_width(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unique_tuner_widths(name: str, values: Iterable[int]) -> Tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of positive integers")
    try:
        candidates = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of positive integers") from exc
    if not candidates:
        raise ValueError(f"{name} must contain at least one width")

    widths = []
    seen = set()
    for value in candidates:
        width = _positive_tuner_width(f"{name} entry", value)
        if width not in seen:
            seen.add(width)
            widths.append(width)
    return tuple(widths)


def compiler_schedule_search_space(
    nc_values: Iterable[int],
    jc_values: Iterable[int],
    *,
    panel_var: str,
    free_var: str,
) -> Tuple[tuple, ...]:
    """Build the opt-in compiler tile-ijk tuner choice space.

    The returned immutable choices form the Cartesian product of ``Nc``, ``Jc``,
    operand staging scope, and result accumulation placement. Staging is scoped
    to either the sparse panel tile (``panel_var``) or the enclosing dense free-axis
    tile (``free_var``); accumulation is either direct to the result or through a
    heap-backed compact result tile. Each choice can be passed to
    :func:`schedule_from_tuner_choice`.

    This helper only describes compiler schedules. It does not participate in the
    production native selector, its persistent cache, or its learned-model features.
    Duplicate widths are removed while preserving the caller's order.
    """
    panel_var = _compiler_structural_name("panel_var", panel_var)
    free_var = _compiler_structural_name("free_var", free_var)
    if panel_var == free_var:
        raise ValueError("panel_var and free_var must name distinct index variables")

    ncs = _unique_tuner_widths("nc_values", nc_values)
    jcs = _unique_tuner_widths("jc_values", jc_values)
    return tuple(
        ("tileijk", (nc, jc, scope_var, accum))
        for nc in ncs
        for jc in jcs
        for scope_var in (panel_var, free_var)
        for accum in ("direct", "heap")
    )


def schedule_from_tuner_choice(
    choice: tuple,
    *,
    row_var: str,
    panel_var: str,
    free_var: str,
    packed_operand: str,
) -> Optional[Schedule]:
    """Translate a normalized tile-ijk tuner choice into a compiler schedule.

    The structural names are deliberately required: the compiler owns the mapping
    from tuner decisions to tensor accesses and index variables. The legacy
    ``("tileijk", (Nc, Jc))`` form remains an alias for panel-scoped staging with
    direct result accumulation. The compiler-only extended form is
    ``("tileijk", (Nc, Jc, scope_var, accum))``, where ``scope_var`` must be the
    supplied panel or free logical variable and ``accum`` is ``"direct"`` or
    ``"heap"``. Non-tile-ijk choices have no compiler adapter yet and return
    ``None``. This helper is opt-in and is not used by production dispatch.
    """
    structural_names = {
        "row_var": row_var,
        "panel_var": panel_var,
        "free_var": free_var,
        "packed_operand": packed_operand,
    }
    for field, value in structural_names.items():
        _compiler_structural_name(field, value)
    index_names = (row_var, panel_var, free_var)
    if len(set(index_names)) != len(index_names):
        raise ValueError("row_var, panel_var, and free_var must be distinct")

    if not isinstance(choice, tuple):
        raise TypeError("tuner choice must be a (kind, parameter) tuple")
    if len(choice) != 2:
        raise ValueError("tuner choice must contain exactly (kind, parameter)")

    kind, parameter = choice
    if not isinstance(kind, str):
        raise TypeError("tuner choice kind must be a string")
    if kind not in ("v2", "tilej", "tileijk"):
        raise ValueError(
            "unknown tuner choice kind "
            f"{kind!r}; expected 'v2', 'tilej', or 'tileijk'"
        )

    if kind == "v2":
        if parameter is not None:
            raise ValueError("normalized 'v2' choice must use parameter None")
        return None

    if kind == "tilej":
        _positive_tuner_width("tilej Jc", parameter)
        return None

    if not isinstance(parameter, tuple):
        raise TypeError(
            "normalized 'tileijk' parameter must be an "
            "(Nc, Jc) or (Nc, Jc, scope_var, accum) tuple"
        )
    if len(parameter) not in (2, 4):
        raise ValueError(
            "normalized 'tileijk' parameter must contain exactly (Nc, Jc) or "
            "(Nc, Jc, scope_var, accum)"
        )
    nc = _positive_tuner_width("tileijk Nc", parameter[0])
    jc = _positive_tuner_width("tileijk Jc", parameter[1])
    scope_var = panel_var
    accum = "direct"
    if len(parameter) == 4:
        scope_var = _compiler_structural_name("tileijk scope_var", parameter[2])
        if scope_var not in (panel_var, free_var):
            raise ValueError(
                "tileijk scope_var must name the supplied panel_var or free_var"
            )
        accum = parameter[3]
        if not isinstance(accum, str):
            raise TypeError("tileijk accum must be a string")
        if accum not in ("direct", "heap"):
            raise ValueError("tileijk accum must be 'direct' or 'heap'")

    return Schedule(
        loop_order=(row_var, panel_var, free_var),
        tiles=(
            TileSpec(
                free_var,
                nc,
                placement="outermost",
                accum=accum,
                unroll=False,
            ),
            TileSpec(
                panel_var,
                jc,
                placement=f"child_of:{free_var}_out",
                kind="panel",
                accum="direct",
            ),
        ),
        relayout=RelayoutSpec(
            operand=packed_operand,
            pack_var=free_var,
            strip_width=nc,
            scope_var=scope_var,
        ),
        tag="tuner-tileijk",
        parallel_loop=row_var,
    )


def _dispatch_decision(a, b, result_shape, kind, param, nt, epilogue=None):
    """Run the memoized winner. Returns (result, True) for a tiled kernel or
    None (== "use the caller's byte-identical baseline path") for kind 'v2'.

    ``result`` is the native result object, or whatever ``epilogue`` returns when
    the caller supplied one."""
    if kind == "tilej":
        out = _ops.spmm_csr_float_tilej(*_tilej_args(a, b, result_shape, param, nt))
    elif kind == "tileijk":
        Nc, Jc = param
        out = _ops.spmm_csr_float_tileijk(
            *_tileijk_args(a, b, result_shape, Nc, Jc, nt)
        )
    else:
        return None  # v2
    return (out if epilogue is None else epilogue(out)), True


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
            os.path.expanduser("~"), ".cache"
        )
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
            brand = (
                subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        elif platform.system() == "Linux" and os.path.isfile("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        brand = line.split(":", 1)[1].strip()
                        break
    except Exception:
        brand = ""
    parts = (
        platform.system(),
        platform.machine(),
        brand,
        str(os.cpu_count()),
        str(query_llc()),
    )
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


def _sig_key(sig: tuple, baseline_tag: str = "v2") -> str:
    """Persistent-cache key for a shape signature under one baseline.

    The default baseline keeps its historical unprefixed key, so a cache written
    before other baselines existed stays readable."""
    key = ",".join(str(x) for x in sig)
    return key if baseline_tag == "v2" else f"{baseline_tag}|{key}"


def _persist_get(sig: tuple, baseline_tag: str = "v2"):
    """Return a cached (kind, param) for this machine+signature+baseline, or None."""
    entry = (
        _load_persist_cache().get(_machine_id(), {}).get(_sig_key(sig, baseline_tag))
    )
    if entry is None:
        return None
    kind, param = entry[0], entry[1]
    if kind == "tileijk" and isinstance(param, list):
        param = tuple(param)  # JSON list -> (Nc, Jc)
    return (kind, param)


def _persist_put(sig: tuple, kind: str, param, baseline_tag: str = "v2") -> None:
    """Record a measured winner. Atomic (temp + os.replace), best-effort."""
    path = _cache_path()
    if not path:
        return
    cache = _load_persist_cache()
    stored = list(param) if isinstance(param, tuple) else param
    cache.setdefault(_machine_id(), {})[_sig_key(sig, baseline_tag)] = [kind, stored]
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"version": _CACHE_VERSION, "entries": cache}, f)
        os.replace(tmp, path)
    except Exception:
        pass  # never let a cache write break a matmul


def clear_autotune_cache() -> None:
    """Wipe the persistent on-disk autotune cache used by the ``"max"`` level.

    The ``"max"`` level records each measured SpMM probe winner in a per-machine
    JSON cache (by default under ``$XDG_CACHE_HOME/scorch/autotune.json``) so the
    search is paid only once per machine. This deletes that file and resets the
    in-memory copy, forcing the next ``"max"`` dispatch to re-probe.

    Best-effort: a missing cache file is fine and any I/O error is swallowed so a
    failed wipe never breaks a matmul. It does **not** clear the in-process
    per-shape decision memo (that lives for the process lifetime) nor the
    compiled ``.so`` kernel cache.

    Returns
    -------
    None

    See Also
    --------
    set_autotune : Select the ``"max"`` level that writes this cache.

    Examples
    --------
    >>> import scorch
    >>> scorch.clear_autotune_cache()
    """
    global _persist_cache, _cache_loaded
    _persist_cache = {}
    _cache_loaded = True
    _plan_invalidate_all()  # see set_autotune
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
    P = max(1, -(-J // max(1, Jc_tj)))  # ceil(J / Jc_tj)
    tj = J * BN + P * 2 * Cwr + A + P * M * 4
    nk = max(1, -(-N // max(1, Nc)))  # ceil(N / Nc)
    ijk = J * BN + Cwr + A * nk + 2 * J * BN  # compute + relayout
    return ijk < tj


# ---------------------------------------------------------------------------
# Learned level (Phase 2): an offline-trained gradient-boosted cost model predicts
# each candidate's runtime in O(1); we argmin over the SAME probe candidate set with
# a v2 floor. Falls back to analytic when no per-machine model is present. See
# autotune-levels/01-phase2-learned-design.md. The featurize + tree walker here are
# the CANONICAL definitions; bench/train_autotune_model.py IMPORTS them, so training
# and inference are byte-identical (no train/serve drift possible).
# ---------------------------------------------------------------------------

# Feature vector order (a candidate row). All quantities are cheap / inference-
# computable from CSR metadata + a fixed-size sample; no O(nnz) wavefront.
_FEATURES = (
    "f_log_degree",
    "f_locality",
    "f_degree_cv",
    "f_log2_N",
    "f_thrash",
    "f_log_nnz",
    "f_log_M",
    "f_log_J",
    "f_log_operand",
    "f_log_output",
    "f_is_tilej",
    "f_is_tileijk",
    "f_jc_frac",
    "f_log_P",
    "f_log_nk",
    "f_log_cand_bytes",
    "f_cand_over_output",
)

_LEARNED_MARGIN = float(os.environ.get("SCORCH_AUTOTUNE_MARGIN", "0.03"))
# Confirming a tiled pick against v2 is now unconditional at every non-probing level
# (see maybe_dispatch), so this is the ESCAPE HATCH rather than the opt-in it used to
# be: SCORCH_AUTOTUNE_CONFIRM=0 restores the old unmeasured behaviour, which is what
# the confirm's one-time cost is measured against.
_CONFIRM_TILED = os.environ.get("SCORCH_AUTOTUNE_CONFIRM", "1") != "0"
# Retained so existing callers and tests that force confirming on keep working; it no
# longer changes behaviour, because confirming is the default.
_LEARNED_CONFIRM = os.environ.get("SCORCH_AUTOTUNE_CONFIRM", "0") == "1"
_CV_NSAMP = int(os.environ.get("SCORCH_TILING_CV_NSAMP", "4096"))
# Gate policy for learned. DEFAULT = WIDEN (operand>C only, relax degree+locality).
# The data showed keep-gate (analytic gate) is NOT worth it -- it barely beats analytic
# on M5 (0.917 vs 0.876) and LOSES on redwood (0.873 vs 0.879) -- because analytic's
# DEG_FLOOR=64 EXCLUDES mid-degree scattered matrices (deg 16-64) where tiling genuinely
# wins (scatter_deg50: tile-j 1.7x over v2). Widening catches those analytic
# false-negatives; a tiled pick in the widened-only region (operand>C but analytic gate
# rejects) is CONFIRMED vs v2 (keep the faster) -> provably no-regression there.
# The analytic-eligible region used to be exempt from that confirm, on the theory that
# the model plus its v2 floor was reliable inside the gate; a 236-cell grid showed it
# is not (inline_1 at N=512 passes the ordinary gate and the model still shipped
# 0.385x of untiled), so the confirm is now unconditional. Result: 0.972 (M5) /
# 0.977 (redwood) of oracle vs analytic 0.876/0.879. SCORCH_AUTOTUNE_WIDEN=0 reverts
# to the analytic gate.
_LEARNED_WIDEN = os.environ.get("SCORCH_AUTOTUNE_WIDEN", "1") == "1"


def _cand_bytes(kind, M, J, nnz, N, Jc, Nc) -> float:
    """Predicted DRAM bytes for a candidate (mirrors tiling's byte model; a model
    feature, so the tree can reproduce the analytic decision and learn the residual)."""
    BN = 4.0 * N
    Cwr = M * BN
    A = 8.0 * nnz
    if kind == "tilej":
        P = max(1, -(-int(J) // max(1, int(Jc))))
        return J * BN + P * 2 * Cwr + A + P * M * 4
    if kind == "tileijk":
        nk = max(1, -(-int(N) // max(1, int(Nc))))
        return J * BN + Cwr + A * nk + 2 * J * BN
    return J * BN + Cwr + A  # v2


def _featurize(M, J, nnz, N, C, locality, degree_cv, kind, Jc, Nc):
    """Canonical feature vector (order == _FEATURES) for one candidate. Pure function
    of cheap inputs; imported by the offline trainer so train==serve by construction."""
    Jf = max(1.0, float(J))
    Nf = max(1.0, float(N))
    Mf = max(1.0, float(M))
    nnzf = max(1.0, float(nnz))
    degree = nnzf / Jf
    operand = Jf * 4.0 * Nf
    output = Mf * 4.0 * Nf
    cb = max(1.0, _cand_bytes(kind, Mf, Jf, nnzf, Nf, Jc, Nc))
    P = max(1, -(-int(J) // max(1, int(Jc)))) if kind == "tilej" else 0
    nk = max(1, -(-int(N) // max(1, int(Nc)))) if kind == "tileijk" else 0
    return [
        math.log(max(1e-6, degree)),
        float(locality),
        float(degree_cv),
        math.log2(Nf),
        operand / C,
        math.log(nnzf),
        math.log(Mf),
        math.log(Jf),
        math.log(max(1.0, operand)),
        math.log(output),
        float(kind == "tilej"),
        float(kind == "tileijk"),
        float(Jc) / Jf,
        math.log1p(P),
        math.log1p(nk),
        math.log(cb),
        cb / output,
    ]


def _build_stacked(spec: dict) -> dict:
    """Per-tree JSON -> stacked padded (T, maxnodes) numpy arrays for the fast walker."""
    trees = spec["trees"]
    T = len(trees)
    maxn = max(len(t["feature"]) for t in trees)
    feat = np.zeros((T, maxn), dtype=np.int64)
    thr = np.zeros((T, maxn), dtype=np.float64)
    left = np.zeros((T, maxn), dtype=np.int64)
    right = np.zeros((T, maxn), dtype=np.int64)
    val = np.zeros((T, maxn), dtype=np.float64)
    maxdepth = 0
    for i, t in enumerate(trees):
        n = len(t["feature"])
        feat[i, :n] = t["feature"]
        thr[i, :n] = t["threshold"]
        left[i, :n] = t["left"]
        right[i, :n] = t["right"]
        val[i, :n] = t["value"]
        d = 0
        stack = [(0, 0)]
        while stack:
            node, dd = stack.pop()
            if t["left"][node] == -1:
                d = max(d, dd)
            else:
                stack.append((t["left"][node], dd + 1))
                stack.append((t["right"][node], dd + 1))
        maxdepth = max(maxdepth, d)
    return dict(
        T=T,
        feat=feat,
        thr=thr,
        left=left,
        right=right,
        val=val,
        init=float(spec["init"]),
        lr=float(spec["learning_rate"]),
        maxdepth=maxdepth,
        feature_names=list(spec["feature_names"]),
    )


def _walker_predict(stacked: dict, X) -> "np.ndarray":
    """FAST pure-numpy GBT inference (== sklearn.predict). Level-synchronous over a
    (K candidates x T trees) node-index matrix: only `maxdepth` python iterations,
    each a few numpy ops on a K*T array (~50-200us for K~7, T~400). Memoized per
    signature upstream, so this is a first-call-only cost.

    X is cast to float32 BEFORE the threshold compares: sklearn's decision trees store
    X as float32 internally, so a float64 compare near a split boundary would take a
    different branch and desync the walker from sklearn.predict (the export we ship)."""
    X = np.asarray(X, dtype=np.float32)
    K = X.shape[0]
    T = stacked["T"]
    feat = stacked["feat"]
    thr = stacked["thr"]
    left = stacked["left"]
    right = stacked["right"]
    val = stacked["val"]
    trange = np.arange(T)[None, :]
    krange = np.arange(K)[:, None]
    state = np.zeros((K, T), dtype=np.int64)
    for _ in range(stacked["maxdepth"]):
        nf = feat[trange, state]
        nthr = thr[trange, state]
        safe_nf = np.where(nf >= 0, nf, 0)
        xval = X[krange, safe_nf]
        goleft = xval <= nthr
        nxt = np.where(goleft, left[trange, state], right[trange, state])
        state = np.where(nf < 0, state, nxt)
    return stacked["init"] + stacked["lr"] * val[trange, state].sum(axis=1)


# --- per-machine model loading (lazy, machine-keyed; cache-dir -> models-dir -> None)
_LEARNED_VERSION = 1
_learned_loaded = False
_learned_model: Optional[dict] = None


def _user_cache_dir() -> str:
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
    return os.path.join(base, "scorch")


def _learned_model_path() -> Optional[str]:
    env = os.environ.get("SCORCH_AUTOTUNE_MODEL")
    if env == "0":
        return None  # learned disabled -> analytic fallback
    if env:
        return env  # explicit override
    mid = _machine_id()
    cand = os.path.join(_user_cache_dir(), f"autotune_model_{mid}.json")
    if os.path.isfile(cand):
        return cand
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cand2 = os.path.join(repo, "autotune-levels", "models", f"model_{mid}.json")
    if os.path.isfile(cand2):
        return cand2
    return None


def _load_learned_model() -> Optional[dict]:
    """The stacked model for THIS machine, or None (-> analytic fallback). Cached.
    Rejected on version bump, machine mismatch, or feature-schema drift."""
    global _learned_loaded, _learned_model
    if _learned_loaded:
        return _learned_model
    _learned_loaded = True
    path = _learned_model_path()
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            spec = json.load(f)
        if (
            spec.get("version") != _LEARNED_VERSION
            or spec.get("machine_id") != _machine_id()
            or list(spec.get("feature_names", [])) != list(_FEATURES)
        ):
            return None
        _learned_model = _build_stacked(spec)
    except Exception:
        _learned_model = None  # any error -> silent analytic fallback
    return _learned_model


def _reset_learned_model_cache() -> None:
    """Force the next _load_learned_model() to re-read from disk (tests / retrain)."""
    global _learned_loaded, _learned_model
    _learned_loaded = False
    _learned_model = None


def _degree_cv(a) -> float:
    """Degree-skew std/mean over ~4096 sampled rows (the discriminator between
    power-law skewed graphs -- large Jc fine -- and uniform-random -- small Jc).
    Private RNG; O(sample) from indptr, never O(nnz)."""
    pos = a.storage._mode_indices[1][0]
    M = int(pos.numel()) - 1
    if M <= 0:
        return 0.0
    n = min(_CV_NSAMP, M)
    g = torch.Generator().manual_seed(1)
    ridx = torch.randint(0, M, (n,), generator=g)
    degs = (pos[ridx + 1].to(torch.long) - pos[ridx].to(torch.long)).float()
    m = degs.mean().item()
    if m <= 0:
        return 0.0
    return float(degs.std(unbiased=False).item() / m)


def _eligible_learned(J: int, N: int, C: int) -> bool:
    """Widened gate for the learned level: keep ONLY the operand>C prefilter (the
    physics boundary -- B fits cache => no tiling helps -- and the cost boundary --
    operand>C => a >=16-36MB B-stream => a matmul big enough to absorb the O(1)
    prediction). Relax the degree floor + locality pre-exclusion; the model + v2 floor
    route the newly-admitted products/arxiv/FEM/banded shapes. Every sub-ms GCN-small/
    AE/attention shape fails operand>C at int-comparison cost => byte-neutral 99%."""
    return _operand_over_cache(J, N, C)


def _learned_decide(a, b, M, J, N, nnz, C, model):
    """Predict each candidate's runtime, argmin with a v2 floor. Returns (kind, param)
    where param is None | Jc | (Nc, Jc). Measurement-free (the 'analytic cost')."""
    Jc0 = _panel_width(N, C)
    loc = _locality_ratio(a, J)
    dcv = _degree_cv(a)
    cands = [("v2", 0, 0)]
    for jc in _jc_ladder(Jc0):
        cands.append(("tilej", int(jc), 0))
    if _HAS_TILEIJK and N >= _NIJK_MIN:
        Nc, Jci = _ijk_params(N, M, J, C)
        if Nc < N:
            cands.append(("tileijk", int(Jci), int(Nc)))
    X = np.array(
        [_featurize(M, J, nnz, N, C, loc, dcv, k, jc, nc) for (k, jc, nc) in cands],
        dtype=np.float64,
    )
    pred = _walker_predict(model, X)  # predicted log-time (lower = faster)
    v2_pred = float(pred[0])
    # v2 floor: only leave v2 for a tiled kernel the model predicts BEATS v2 by margin.
    best = None
    for i in range(1, len(cands)):
        if best is None or pred[i] < pred[best]:
            best = i
    if best is not None and float(pred[best]) < v2_pred + math.log(
        max(1e-9, 1.0 - _LEARNED_MARGIN)
    ):
        k, jc, nc = cands[best]
        return ("tilej", jc) if k == "tilej" else ("tileijk", (nc, jc))
    return ("v2", None)


def _interleaved_times(fns, rounds: int = 2) -> list:
    """Time each candidate `rounds` times with the candidates INTERLEAVED, and
    return the best (minimum) seconds for each, in input order.

    Timing one candidate to completion before starting the next -- which is what
    this replaces -- hands whatever runs later a machine the earlier candidates
    warmed: on a turbo or hybrid part the clock is still ramping and the OpenMP
    team has not settled. It mattered here in a specific way: the caller's
    baseline is always the first candidate, so the safe fallback was the one arm
    that never ran on a warm machine, in the routine whose whole purpose is to
    guarantee we never lose to it.

    Every candidate is warmed before any is timed, then each round runs them all,
    rotating the start so no candidate keeps the first slot. The call budget is
    unchanged: one warmup plus `rounds` timings per candidate, exactly what
    timing them one at a time spent.

    Results are deliberately not returned. Retaining one output per candidate
    would multiply peak memory by the candidate count, and at a wide free
    dimension a single output is hundreds of megabytes -- reddit at N=1024 is
    950 MB. The caller re-runs the winner for its output instead, which costs one
    call against a budget of three per candidate.
    """
    n = len(fns)
    best = [float("inf")] * n
    for fn in fns:
        fn()
    for r in range(rounds):
        for i in range(n):
            j = (i + r) % n
            t0 = time.perf_counter()
            fns[j]()
            dt = time.perf_counter() - t0
            if dt < best[j]:
                best[j] = dt
    return best


def _clears_noise(t_tiled: float, t_first: float, t_last: float) -> bool:
    """Does a tiled candidate beat the baseline by more than the measurement can
    explain?

    `t_first` and `t_last` are two timings of the SAME baseline function with the
    same arguments, entered at opposite ends of the candidate list -- an A/A
    control, measured on this machine, for this shape, in this probe. Their ratio
    is what this cell's noise floor actually is; the house rule is that nothing
    inside the noise floor counts, and this is where a selector can apply it
    without a tuned constant, because it measures the floor rather than assuming
    one.

    The comparison is deliberately doubly conservative: the baseline is the
    *faster* of the two identical arms, and the tiled candidate must beat it by
    more than the *spread* between them. So a cell whose true margin is inside
    the floor fails closed to the baseline. That trades an unmeasurable win --
    which we could not have claimed anyway -- against an unbounded regression:
    the probe memoizes its verdict permanently, so a coin flip decided by noise
    is not a one-call mistake, it is every subsequent call on that shape.
    """
    base = t_first if t_first < t_last else t_last
    spread = (t_last / t_first) if t_last > t_first else (t_first / t_last)
    return t_tiled * spread < base


def _confirm_vs_baseline(
    a, b, result_shape, kind, param, nt, baseline_fn, nthreads, epilogue=None
):
    """One-shot confirm: time {predicted winner, the caller's baseline} once each and
    keep the faster.

    Keeps a level that does not run the full ladder probe no-slower than *that
    caller's own baseline*, at 9 kernel invocations (3 entries x 1 warmup + 2
    timed; the baseline is entered twice as its own control) against the probe's
    22. Memoized by the caller, so a shape pays this once.

    ``epilogue`` is timed as part of the tiled candidate. A caller whose baseline
    folds an elementwise tail into the SpMM -- ``scorch.compile``'s fused
    SpMM+bias+act -- must pay for that tail on the tiled side too, since the tiled
    kernels have no fused epilogue and run it as a separate pass over the output.
    Timing the bare tiled kernel against a fused baseline would credit the tiled
    kernel with work it did not do."""
    if kind == "tilej":
        run = lambda: _ops.spmm_csr_float_tilej(
            *_tilej_args(a, b, result_shape, param, nt)
        )
    else:
        run = lambda: _ops.spmm_csr_float_tileijk(
            *_tileijk_args(a, b, result_shape, param[0], param[1], nt)
        )
    win = run if epilogue is None else (lambda: epilogue(run()))

    def base():
        return baseline_fn(nthreads)

    # The baseline is entered TWICE, at both ends, and the three are interleaved.
    # Two reasons. Order: timing the tiled candidate first and the baseline second
    # -- which is what this used to do -- gave the baseline the warmer machine, so
    # this level was biased the opposite way from the ladder probe below, and
    # neither bias was measured. Floor: two timings of the same baseline are an A/A
    # control, and without one there is no way to tell a real 3% win from noise
    # before memoizing the verdict for good. See _interleaved_times, _clears_noise.
    t_first, t_win, t_last = _interleaved_times([base, win, base])

    # The "v2" sentinel keeps its historical name at every level of this module and
    # in the on-disk cache; it means "run the caller's own baseline path", which is
    # the drop-in SpMM for `matmul` and the fused kernel for `scorch.compile`.
    if _clears_noise(t_win, t_first, t_last):
        return (kind, param)
    return ("v2", None)


def maybe_dispatch(
    a,
    b,
    result_shape,
    baseline_fn,
    nthreads: Optional[int],
    time_dict: Optional[dict] = None,
    level: Optional[str] = None,
    epilogue: Optional[Callable] = None,
    baseline_tag: str = "v2",
):
    """Return (result, used_tiled: bool) or None to signal "use the caller's own
    baseline path". Only ever returns a tiled kernel (tile-j / tile-ijk) when it
    has been MEASURED (balanced/max probe, or the one-shot confirm) to beat
    `baseline_fn` on this shape by more than that measurement's own noise floor.

    The baseline is entered as a candidate twice, at both ends of the list, and
    every candidate is timed interleaved. So the baseline is never the only arm
    timed on a cold machine, and the two identical baseline arms give the probe an
    A/A control for this exact cell -- which is what "beat it" is measured
    against. A tiled candidate whose win is inside that floor fails closed to the
    baseline. Being a candidate is not on its own enough to make the guarantee
    true: the verdict is memoized for the process, and at "max" persisted for the
    machine, so a margin decided by noise is not one bad call but every later call
    on that shape.

    `baseline_fn(nthreads)` is the caller's own alternative, and `baseline_tag`
    names it. Two callers dispatch this kernel family and they are asking different
    questions: `ops.matmul`'s baseline is the drop-in SpMM `spmm_csr_float_v2`,
    while `scorch.compile`'s fused path baseline is a fused SpMM+bias+act kernel,
    which folds the elementwise tail into the SpMM's row epilogue and so is faster
    than the drop-in SpMM plus a separate pass. "tile-j beats the drop-in SpMM" does
    NOT imply "tile-j plus a separate tail beats the fused kernel", so each verdict
    is memoized under its own `baseline_tag`; sharing one entry would let either
    caller run a kernel that lost its own comparison.

    `epilogue`, when supplied, is applied to every tiled candidate's result -- inside
    the timed region during the probe/confirm, and on the memoized dispatch -- and
    its return value is what this function hands back. It exists so a fused caller's
    tail is charged to the tiled side, which has no fused epilogue of its own.

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
    idx = a._native_mode_indices()
    nnz = int(idx[1][1].numel())
    C = query_llc()
    # learned uses the WIDENED gate (operand>C only) iff opted-in AND a model is loaded;
    # by default it uses the analytic gate + the _scattered pre-gate below (so
    # banded/FEM/low-degree -> v2 exactly like analytic). Model-absent -> analytic
    # branch. Either way the 99% (operand<=C) short-circuits to v2 here.
    learned_model = _load_learned_model() if level == "learned" else None
    if learned_model is not None and _LEARNED_WIDEN:
        if not _eligible_learned(J, N, C):
            return None
    elif not _eligible(J, nnz, N, C):
        return None

    nt = nthreads if nthreads is not None else -1
    Jc = _panel_width(N, C)
    sig = _signature(a, N)
    # Memoize per (signature, level, baseline): different levels can pick different
    # winners (analytic->tile-j@base; balanced/max->probed width/kernel), and a
    # different baseline is a different question (see the docstring).
    memo_key = (sig, level, baseline_tag)

    def _timed_dispatch(kind, param):
        """Run the memoized winner, recording its wall time into time_dict."""
        t0 = time.perf_counter()
        out = _dispatch_decision(a, b, result_shape, kind, param, nt, epilogue)
        if time_dict is not None:
            time_dict["eval_time"] = time.perf_counter() - t0
        return out

    cached = _decision.get(memo_key)
    if cached is not None:
        return _timed_dispatch(cached[0], cached[1])

    # "max": a prior run may have already measured the winner for this machine.
    if level == "max":
        pc = _persist_get(sig, baseline_tag)
        if pc is not None:
            _decision[memo_key] = pc
            return _timed_dispatch(pc[0], pc[1])

    # "learned" (model present): predict every candidate's runtime in O(1) and argmin
    # with a predicted v2 floor, then confirm that pick against v2 by measurement once.
    # The gate was WIDENED above, so this skips the analytic _scattered pre-exclusion;
    # the model plus the confirm handle the newly-admitted products/arxiv/FEM/banded
    # shapes (they land on v2 unless a tiled kernel is measured to win). Memoized per
    # (sig, level). SCORCH_AUTOTUNE_CONFIRM=0 removes the confirm, for A/B only.
    if learned_model is not None:
        # DEFAULT (not widened): apply the analytic locality gate first so banded/FEM/
        # low-degree shapes route to v2 exactly like analytic -> the model only decides
        # in the region where it is reliable (scattered high-degree: reddit/scatter/
        # power-law -> big wins).
        if not _LEARNED_WIDEN and not _scattered(a, J):
            _decision[memo_key] = ("v2", None)
            return None
        kind, param = _learned_decide(a, b, M, J, N, nnz, C, learned_model)
        # When WIDENED, a tiled pick in the widened-only region (analytic gate would
        # reject) extrapolates into v2-territory, so confirm it against v2 once
        # (guaranteed no-regression; memoized). Data-driven: without this, redwood FEM
        # regressed to 0.72 of oracle. SCORCH_AUTOTUNE_CONFIRM=1 forces confirm always.
        if kind != "v2":
            # Confirm EVERY tiled pick, not just the ones in the widened-only region.
            # Restricting the confirm to picks the analytic gate would have rejected
            # left a hole exactly where the model is most confident and still wrong:
            # inline_1 at N=512 passes the ordinary gate (eligible and scattered), so
            # no confirm ran, and the model's tile-ijk pick shipped at 0.385x of
            # untiled — the same cell and the same mistake analytic made. Measured over
            # 236 cells, this hole accounted for 3 of learned's 4 regressions.
            if _CONFIRM_TILED:
                kind, param = _confirm_vs_baseline(
                    a,
                    b,
                    result_shape,
                    kind,
                    param,
                    nt,
                    baseline_fn,
                    nthreads,
                    epilogue,
                )
        _decision[memo_key] = (kind, param)
        if kind == "v2":
            return None
        return _timed_dispatch(kind, param)

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
        # analytic: the byte model picks tile-j@base, or tile-ijk when it says the
        # relayout pays for itself.
        if ijk is not None and _ijk_beats_tilej_bytes(N, M, J, nnz, Jc, ijk[0], C):
            kind, param = "tileijk", ijk
        else:
            kind, param = "tilej", Jc
        # ...and then that pick is CONFIRMED against v2, once, before being memoized.
        #
        # The premise this level used to rest on — "passing the gate implies tile-j
        # beats v2, so nothing needs timing" — is false, and measurably so. Over a
        # 236-cell grid on redwood, analytic shipped six tiled-route regressions, the
        # worst a 2.68x slowdown (audikw_1 at N=128, 0.373x of untiled, against a 1.9%
        # noise floor), while `balanced` — same gate, but it times v2 — had none. The
        # gate cannot be tightened out of this with the features on hand: span proxy
        # 0.823 both loses (crankseg_1) and wins (mouse_gene), and degree ~200 likewise
        # (crankseg_1 loses, scatter200 wins).
        #
        # So the default level now measures, once per shape, and only two candidates:
        # six kernel invocations against the eighteen a full ladder probe costs. That
        # makes every non-off level no-regression-vs-v2 by construction rather than by
        # the gate happening to be right. What still separates analytic from balanced
        # is the search: analytic checks one width, balanced searches the ladder.
        if _CONFIRM_TILED:
            kind, param = _confirm_vs_baseline(
                a, b, result_shape, kind, param, nt, baseline_fn, nthreads, epilogue
            )
        _decision[memo_key] = (kind, param)
        if kind == "v2":
            return None
        return _timed_dispatch(kind, param)

    # balanced/max first-call micro-probe: time every candidate, keep the fastest's
    # result. v2 is always a candidate -> the winner is never slower than v2. The
    # coarse Jc ladder {base,/2,/4,/8} grabs the uniform-random (appu-like) tail
    # where base overshoots. Because the relayout runs INSIDE spmm_csr_float_tileijk,
    # timing that call accounts for the relayout honestly against tile-j and v2.
    def _tailed(fn):
        """Charge the caller's elementwise tail to a tiled candidate's clock."""
        return fn if epilogue is None else (lambda: epilogue(fn()))

    def _baseline():
        return baseline_fn(nthreads)

    # Heterogeneous param slot (None | Jc int | (Nc,Jc) tuple) -> annotate `list`.
    # The baseline needs no `_tailed`: a fused baseline already folds its own tail.
    cands: list = [("v2", None, _baseline)]
    for jc in _jc_ladder(Jc):
        cands.append(
            (
                "tilej",
                jc,
                _tailed(
                    lambda jc=jc: _ops.spmm_csr_float_tilej(
                        *_tilej_args(a, b, result_shape, jc, nt)
                    )
                ),
            )
        )
    if ijk is not None:
        cands.append(
            (
                "tileijk",
                ijk,
                _tailed(
                    lambda: _ops.spmm_csr_float_tileijk(
                        *_tileijk_args(a, b, result_shape, ijk[0], ijk[1], nt)
                    )
                ),
            )
        )

    # The baseline again, at the far end of the list: the same function with the
    # same arguments as cands[0]. Two identical arms at opposite ends are this
    # cell's A/A control, so the probe measures its own noise floor instead of
    # assuming one, and they bracket the tiled candidates so no candidate is
    # compared against a baseline timed only on a cold machine.
    cands.append(("v2", None, _baseline))

    # Interleaved, not one candidate at a time -- see _interleaved_times.
    times = _interleaved_times([fn for _, _, fn in cands])
    t_first, t_last = times[0], times[-1]

    # The winner is chosen among the tiled candidates only; both baseline entries
    # are controls, and either one winning means the same thing (route to v2).
    best_i, best_t = -1, float("inf")
    for i in range(1, len(cands) - 1):
        if times[i] < best_t:
            best_i, best_t = i, times[i]

    if best_i < 0 or not _clears_noise(best_t, t_first, t_last):
        # Inside the floor, or no tiled candidate at all: fail closed to the
        # baseline. A verdict here is memoized for the process (and for the
        # machine, at "max"), so guessing from noise is not a one-call error.
        best_kind, best_param = "v2", None
    else:
        best_kind, best_param = cands[best_i][0], cands[best_i][1]

    _decision[memo_key] = (best_kind, best_param)
    if level == "max":
        # pay the search once EVER
        _persist_put(sig, best_kind, best_param, baseline_tag)
    if best_kind == "v2":
        return None  # caller runs its own baseline + populates time_dict itself
    if time_dict is not None:
        time_dict["eval_time"] = best_t  # the winning kernel's measured time
    # Re-run the winner for its output rather than retaining every candidate's:
    # one dense output per candidate is hundreds of megabytes at a wide free
    # dimension. One call on a budget of three per candidate, and only on the
    # tiled branch -- when the baseline wins, the caller runs it anyway.
    return cands[best_i][2](), True
