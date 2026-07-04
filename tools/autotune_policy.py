#!/usr/bin/env python
"""Per-host autotune for scorch's OpenMP parallel policy (Phase 4b).

Measures THIS build machine and writes ``csrc/scorch_policy_tuned.h`` with the
thread-cap / schedule-chunk constants that best fit its cores, then rebuilds a
clean ``scorch_ops``. When the autotune is skipped (CI, cross-compile, plain
``pip install``), the redwood-tuned defaults baked into ``csrc/scorch_policy.h``
apply and everything still compiles — the tuned header is optional.

Why a machine measurement at all: the policy SHAPE (throttle small products,
adaptive chunk) is machine-stable, but the CONSTANTS are not. Redwood's hybrid
P+E i9 wants ~8 threads for an ultra-sparse product; an Apple M5 wants 1. Same
formula, different grain. This tool re-fits the grain per host.

Flow
----
1. Build an INSTRUMENTED ``scorch_ops`` (``-DSCORCH_TUNE_HOOKS``) so the prebuilt
   kernels' thread count and schedule chunk can be forced per-run via the
   ``SCORCH_TUNE_THREADS`` / ``SCORCH_TUNE_CHUNK`` env vars — no rebuild per cell.
   The shipped library never sets that define, so the hooks compile out.
2. Sweep a synthetic panel across a threads x chunk grid, BACK-TO-BACK in one
   process (never cross-time; median of repeats), recording runtime per cell.
   This is the methodology validated on redwood — cross-time comparisons on a
   shared box are unreliable, so we only ever compare cells measured adjacently.
3. Offline-fit the constants: grid-search (grain, rows/thread, chunks/thread,
   chunk bounds) against the measured grid (each candidate's predicted cell ->
   its measured runtime) and maximise the per-matrix geomean speedup. The
   winning fit is compared to the shipped defaults on the SAME measurements.
4. Only if the fit beats the defaults by a margin, write the tuned header.
   Otherwise keep the defaults (the sweep is only worthwhile if it wins).
5. Rebuild a CLEAN ``scorch_ops`` (no hooks). JIT kernels re-tune for free —
   ``utils._kernel_name`` folds the policy-header text into its cache hash, so
   rewriting the constants invalidates stale ``.so`` files automatically.

Panel: real SuiteSparse vs synthetic
------------------------------------
Two data sources feed the sweep:

  * REAL SuiteSparse matrices (preferred) — pass ``--matrices PATH`` (a dir of
    ``<name>/<name>.mtx``; also honors env ``SCORCH_SUITESPARSE`` and auto-detects
    ``/scratch/suitesparse``), or ``--download N`` to fetch a stratified sample if
    you don't have the collection. A stratified panel (across nnz bands AND
    nnz/row, incl. the ultra-sparse 1-4 nnz/row tail) is what makes tuning the
    work GRAINS trustworthy — see the fidelity caveat below.
  * SYNTHETIC uniform-random (fallback) — used when no dataset is present and
    ``--download`` wasn't given, so the tool still runs anywhere. Faithful for the
    CHUNK / topology knobs, NOT for the grains.

Usage
-----
    python tools/autotune_policy.py                       # auto: real if a dataset
                                                          #   is found, else synthetic
    python tools/autotune_policy.py --matrices /scratch/suitesparse   # real panel
    python tools/autotune_policy.py --download 40         # fetch ~40 real matrices, tune
    python tools/autotune_policy.py --matrices DIR --dry-run   # sweep + fit, write nothing
    python tools/autotune_policy.py --freeze-grains       # tune only chunk/topology knobs
    python tools/autotune_policy.py --reps 7 --rounds 3   # more measurement budget
    python tools/autotune_policy.py --kernels spmspm      # tune one kernel only

``--sweep-worker <out.json>`` is the internal per-host measurement subprocess
(runs inside the instrumented build); you normally don't call it directly.

CAVEAT — panel fidelity (validated on redwood 2026-07-03)
---------------------------------------------------------
The SYNTHETIC panel is uniform-random (no SuiteSparse dependency), a faithful proxy
for the CHUNK / topology knobs but NOT for the work GRAINS: uniform data lacks the
skewed ultra-sparse tail that the SuiteSparse-validated grains (spmspm 3000)
protect, so a full synthetic fit tends to pick a smaller grain (more threads) that
can regress real workloads. On redwood the full synthetic fit wanted grain 1000
(+15%) but `--freeze-grains` — tuning only chunk sizing — gave a defensible +6.5%
without touching the grains. A REAL-matrix panel (`--matrices`/`--download`) is
stronger evidence — it sees the skewed tail — but is still not a blank check for a
grain change: a modest panel's unweighted geomean can be dominated by a few
low-work/many-row outliers and pick a smaller grain that OVER-THREADS the mid-band
into a hybrid CPU's E-core cliff. (Observed on redwood 2026-07-04: a 40-matrix real
fit wanted spmspm grain 3000->500 for +8.8% panel geomean, yet per-matrix it
regressed the mid-band up to 2.4x; ~all the win was the topology knobs, and grain
3000 was actually best at the default topology.) So: prefer a real panel over
synthetic; inspect the per-matrix breakdown (or use a larger, distribution-faithful
`--panel-size`) before trusting a grain drop; `--freeze-grains` remains the
conservative choice.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POLICY_H = REPO / "csrc" / "scorch_policy.h"
TUNED_H = REPO / "csrc" / "scorch_policy_tuned.h"

# --- SuiteSparse real-matrix panel (Phase 4b "option 1") ----------------------
# The synthetic panel is a faithful proxy for the chunk/topology knobs but NOT
# for the work grains (uniform-random data lacks SuiteSparse's skewed ultra-sparse
# tail). A real-matrix panel — the collection, present locally or downloaded — lets
# the fit tune the grains trustworthily. These bound the panel to the SAME envelope
# as the validated 1714-matrix redwood sweep (bench_spmspm_all_after.py).
MAX_DIM = 100_000            # skip matrices whose larger dim reaches this
MAX_NNZ = 10_000_000         # skip matrices whose (symmetric-expanded) nnz reaches this
DEFAULT_PANEL_SIZE = 40      # target stratified real-matrix panel size (~30-60)
DEFAULT_DOWNLOAD_N = 40      # default sample size for --download
DEFAULT_MATRIX_CACHE = "~/.cache/scorch/suitesparse"

# Stratification bands. nnz is log-decade; nnz/row spans ultra-sparse -> dense-few-row.
# The whole point of a real panel is the ultra-sparse tail, so nnz/row band 0 (<2)
# and 1 (2-4) — the "1-4 nnz/row" regime the grain protects — get their own buckets.
_NNZ_BAND_EDGES = (1e2, 1e3, 1e4, 1e5, 1e6)      # -> 6 bands: <1e2 .. >1e6
_ZPR_BAND_EDGES = (2.0, 4.0, 16.0, 64.0)          # -> 5 bands: <2 .. >64 nnz/row

# One selected matrix. ``ref`` is either an .mtx Path (discovery) or an ssgetpy
# Matrix object (download); both carry name/rows/nnz/zpr so _stratify is shared.
Cand = namedtuple("Cand", "name ref rows nnz zpr sym")

# Shipped redwood-tuned defaults (must mirror csrc/scorch_policy.h). The fit is
# always compared against these; they are also the fallback when tuning is off.
DEFAULTS = {
    "SCORCH_GRAIN_SPMSPM": 3000,
    "SCORCH_GRAIN_SPMM": 150000,
    "SCORCH_ROWS_PER_THREAD": 16,
    "SCORCH_CHUNKS_PER_THREAD": 7,
    "SCORCH_CHUNK_MIN": 4,
    "SCORCH_CHUNK_MAX": 64,
}


# ----------------------------------------------------- SuiteSparse real panel
# Discovery + header pre-filter + stratified sampling, mirroring the validated
# redwood driver (/scratch/bobbyy/spmspm_sweep/bench_spmspm_all_after.py). The
# real panel is what makes a grain change trustworthy (vs the synthetic caveat).

_AUX_SUFFIXES = ("_b", "_x", "_rhs", "_coord", "_skew", "_ans")


def _is_primary_mtx(mtx):
    """True unless the file is an auxiliary RHS/solution (e.g. <name>_b.mtx)."""
    stem = mtx.stem.lower()
    return not any(stem.endswith(s) for s in _AUX_SUFFIXES)


def _discover_mtx(root):
    """Primary .mtx files under ``root``. Canonical SuiteSparse layout is
    ``<name>/<name>.mtx`` (the verified redwood layout: glob ``*/*.mtx`` with the
    file stem equal to its dir name, which auto-skips ``_b``/``_x`` aux files).
    Falls back to a flat dir of ``*.mtx`` and, last, a deep scan (downloader
    nesting) — always filtering out auxiliary files by suffix."""
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    found = {}
    for mtx in root.glob("*/*.mtx"):            # canonical <name>/<name>.mtx
        if mtx.stem == mtx.parent.name:
            found[str(mtx)] = mtx
    for mtx in root.glob("*.mtx"):              # flat dir of <name>.mtx
        if _is_primary_mtx(mtx):
            found.setdefault(str(mtx), mtx)
    if not found:                               # deep fallback (e.g. group nesting)
        for mtx in root.glob("**/*.mtx"):
            if _is_primary_mtx(mtx):
                found[str(mtx)] = mtx
    return sorted(found.values(), key=str)


def _read_mtx_header(mtx):
    """(rows, cols, nnz, symmetric) from a MatrixMarket header — reads only the
    banner + first data line, so pre-filtering a whole collection is cheap.
    Raises (caught by the caller) on array/dense or malformed headers."""
    with open(mtx) as f:
        first = f.readline()
        low = first.lower()
        if "complex" in low:
            raise ValueError("complex matrix")   # can't cast to float32 faithfully
        sym = ("symmetric" in low) or ("hermitian" in low)
        line = f.readline()
        while line and line.startswith("%"):
            line = f.readline()
        p = line.split()
        return int(p[0]), int(p[1]), int(p[2]), sym


def _band(x, edges):
    """Index of the band ``x`` falls into (0 = below the first edge)."""
    b = 0
    for e in edges:
        if x >= e:
            b += 1
        else:
            break
    return b


def _spread(items, k):
    """``k`` evenly-spaced elements of ``items`` (deterministic; k==1 -> middle)."""
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if k == 1:
        return [items[n // 2]]
    if k >= n:
        return list(items)
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def _bucket_cap(nnz_band):
    """How many matrices to keep from one (nnz, nnz/row) bucket. Weight the
    small/mid nnz bands (bands 1-3) where the thread cap actually binds; keep the
    tiny (dispatch-bound) and the large (always all-cores) ends thin."""
    return {0: 1, 1: 3, 2: 3, 3: 3, 4: 2, 5: 1}.get(nnz_band, 2)


def _stratify(cands, target):
    """Pick ~``target`` matrices spread across (nnz-band x nnz/row-band) buckets.
    Guarantees >= 1 per non-empty bucket (so the ultra-sparse tail and the large
    anchors are always represented), weights small/mid, and trims the fattest
    buckets down to ``target`` — never removing a bucket's last representative."""
    buckets = defaultdict(list)
    for c in cands:
        buckets[(_band(c.nnz, _NNZ_BAND_EDGES), _band(c.zpr, _ZPR_BAND_EDGES))].append(c)
    picks = {}
    for key, items in buckets.items():
        items = sorted(items, key=lambda c: (c.nnz, c.name))
        picks[key] = _spread(items, _bucket_cap(key[0]))

    def total():
        return sum(len(v) for v in picks.values())

    while total() > target:
        key = max((k for k, v in picks.items() if len(v) > 1),
                  key=lambda k: len(picks[k]), default=None)
        if key is None:
            break
        picks[key].pop()                        # drop one end of the spread
    flat = [c for v in picks.values() for c in v]
    flat.sort(key=lambda c: (c.nnz, c.name))
    return flat


def _real_panel(matrices_dir, panel_size):
    """Discover, header-pre-filter, and stratify a real-matrix panel of Cands."""
    cands = []
    for mtx in _discover_mtx(matrices_dir):
        try:
            r, c, nnz, sym = _read_mtx_header(mtx)
        except Exception:
            continue                            # array/dense/complex/malformed
        eff = nnz * 2 if sym else nnz
        dim = min(r, c)                         # A@A^T truncates to the square
        if dim < 2 or max(r, c) >= MAX_DIM or eff >= MAX_NNZ:
            continue
        cands.append(Cand(mtx.parent.name, mtx, dim, eff, eff / max(dim, 1), sym))
    return _stratify(cands, panel_size)


def _summarize_panel(panel):
    nnz_b = Counter(_band(c.nnz, _NNZ_BAND_EDGES) for c in panel)
    zpr_b = Counter(_band(c.zpr, _ZPR_BAND_EDGES) for c in panel)
    print(f"  nnz-bands {dict(sorted(nnz_b.items()))}  "
          f"nnz/row-bands {dict(sorted(zpr_b.items()))}", flush=True)
    ultra = [c.name for c in panel if c.zpr < 4.0]
    print(f"  ultra-sparse (<4 nnz/row): {len(ultra)}"
          + (f" e.g. {ultra[:5]}" if ultra else " — NONE (grain under-constrained!)"),
          flush=True)


# ------------------------------------------------- scipy/torch/scorch loaders
def _scipy_to_torch_coo(csr):
    """scipy CSR -> torch sparse COO (for the PyTorch reference op)."""
    import numpy as np
    import torch
    coo = csr.tocoo()
    idx = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    val = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, tuple(coo.shape)).coalesce()


def _to_scorch_csr(csr, name):
    """scipy CSR -> scorch STensor in "ds" (DENSE, COMPRESSED = CSR) format,
    built directly from the sparse arrays (no dense n x n intermediate — real
    matrices can be 100k-dim). Mirrors bench/_utils.to_scorch_csr."""
    import numpy as np
    import torch
    from scorch import STensor
    indptr = torch.from_numpy(csr.indptr.astype(np.int32))
    indices = torch.from_numpy(csr.indices.astype(np.int32))
    values = torch.from_numpy(csr.data.astype(np.float32))
    tcsr = torch.sparse_csr_tensor(indptr, indices, values, size=csr.shape)
    return STensor.from_csr(tcsr, name)


# ----------------------------------------------------------- optional download
def _mtx_present_names(dest):
    return {p.stem for p in _discover_mtx(dest)}


# Cap the nnz of DOWNLOADED matrices so a fresh install fetches a small/fast sample
# rather than multi-MB giants. The download sample is for tuning where the thread cap
# BINDS (small/mid + the ultra-sparse tail); the huge "always all cores" end carries
# no tuning signal, and fetching 10MB+ tarballs just makes install slow. A host that
# has the full collection (via --matrices) still gets the large anchors.
_DOWNLOAD_MAX_NNZ = 2_000_000


def download_matrices(dest, n):
    """Fetch a stratified ~``n``-matrix SuiteSparse sample into ``dest`` if not
    already cached. Opt-in and idempotent (skips matrices already on disk)."""
    dest = Path(dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    have = len(_discover_mtx(dest))
    if have >= n:
        print(f"[download] {have} matrices already cached in {dest}; nothing to fetch.")
        return
    print(f"[download] fetching a stratified ~{n}-matrix SuiteSparse sample "
          f"(incl. ultra-sparse) into {dest} ...", flush=True)
    ok = _download_via_ssgetpy(dest, n) or _download_via_mirror(dest, n)
    final = len(_discover_mtx(dest))
    if not ok and final == 0:
        print("[download] WARNING: fetched nothing (no ssgetpy, no network?). "
              "The tool will fall back to the synthetic panel.")
    print(f"[download] {final} matrices now present in {dest}.", flush=True)


def _download_via_ssgetpy(dest, n):
    try:
        import ssgetpy
    except ImportError:
        print("[download] ssgetpy not installed; trying `pip install ssgetpy` ...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                            "ssgetpy"], check=True)
            import ssgetpy
        except Exception as e:                  # noqa: BLE001
            print(f"[download] ssgetpy unavailable ({e}); falling back to TAMU mirror.")
            return False
    try:
        # Pull the in-envelope population's METADATA (nnz filtered locally so we
        # don't depend on ssgetpy's bounds-kwarg naming), then reuse _stratify.
        results = ssgetpy.search(rowbounds=(2, MAX_DIM - 1), limit=100000)
    except Exception as e:                      # noqa: BLE001
        print(f"[download] ssgetpy.search failed ({e}); falling back to TAMU mirror.")
        return False
    cands = []
    for r in results:
        try:
            rows, cols, nnz = int(r.rows), int(r.cols), int(r.nnz)
        except Exception:
            continue
        dim = min(rows, cols)
        if dim < 2 or max(rows, cols) >= MAX_DIM or nnz >= _DOWNLOAD_MAX_NNZ:
            continue
        cands.append(Cand(r.name, r, dim, nnz, nnz / max(dim, 1), False))
    if not cands:
        return False
    picks = _stratify(cands, n)
    # _stratify floors at the number of non-empty buckets (it keeps >= 1 each for
    # coverage), which can exceed a small n. Thin to ~n by an nnz spread (retains the
    # ultra-sparse low-nnz end and a few larger ones) so --download N fetches ~N.
    if len(picks) > n:
        picks = _spread(sorted(picks, key=lambda c: (c.nnz, c.name)), n)
    print(f"[download] selected {len(picks)} matrices; "
          f"ultra-sparse (<4 nnz/row): "
          f"{sum(1 for c in picks if c.zpr < 4.0)}", flush=True)
    present = _mtx_present_names(dest)
    got = 0
    for c in picks:
        if c.name in present:
            got += 1
            continue
        try:
            c.ref.download(format="MM", destpath=str(dest), extract=True)
            got += 1
            print(f"  [get] {c.name:24s} nnz~{c.nnz:<9d} {c.zpr:5.1f} nnz/row", flush=True)
        except Exception as e:                  # noqa: BLE001
            print(f"  [fail] {c.name}: {type(e).__name__}: {e}", flush=True)
    return got > 0


# Small curated fallback spanning nnz bands AND the ultra-sparse tail, used only
# when ssgetpy is unavailable. Groups verified from the SuiteSparse collection.
_MIRROR_MANIFEST = [
    ("HB", "bcspwr03"),      # 118,   476 nnz  ~4/row
    ("HB", "west0132"),      # 132,   414 nnz  ~3/row  ultra
    ("HB", "gre_115"),       # 115,   421 nnz
    ("HB", "1138_bus"),      # 1138,  4054 nnz ~3.6/row  ultra
    ("HB", "orsirr_1"),      # 1030,  6858 nnz
    ("HB", "sherman3"),      # 5005, 20033 nnz ~4/row  ultra
    ("HB", "saylr4"),        # 3564, 22316 nnz
    ("HB", "bcsstk08"),      # 1074, 12960 nnz ~12/row
    ("HB", "bcsstk13"),      # 2003, 83883 nnz ~42/row  dense-few-row
    ("HB", "can_229"),       # 229,  1777 nnz
    ("LPnetlib", "lpi_klein3"),   # ~1082, ~13k nnz  (ultra-sparse LP)
    ("LPnetlib", "lp_israel"),    # dense-ish LP
    ("Bai", "qc324"),        # 324, 26730 nnz  dense-few-row
    ("FIDAP", "ex5"),        # 27, 279 nnz small dense
    ("Norris", "fv1"),       # 9604, 85264 nnz
]


def _download_via_mirror(dest, n):
    import io
    import ssl
    import tarfile
    import urllib.request
    dest = Path(dest).expanduser()
    present = _mtx_present_names(dest)
    ctx = ssl.create_default_context()
    got = 0
    for group, name in _MIRROR_MANIFEST:
        if got >= n:
            break
        if name in present:
            got += 1
            continue
        url = f"https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"
        try:
            with urllib.request.urlopen(url, timeout=90, context=ctx) as resp:
                data = resp.read()
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(dest)            # -> <name>/<name>.mtx (trusted source)
            got += 1
            print(f"  [get] {group}/{name}", flush=True)
        except Exception as e:                  # noqa: BLE001
            print(f"  [fail] {group}/{name}: {type(e).__name__}: {e}", flush=True)
    return got > 0


# ------------------------------------------------------------------ policy model
def predict_nthreads(work, rows, grain, rpt, hw):
    """Python mirror of scorch_nthreads() in csrc/scorch_policy.h."""
    n = rows // rpt
    if work >= 0:
        by_work = work // grain
        if by_work < n:
            n = by_work
    if n < 1:
        n = 1
    if n > hw:
        n = hw
    return int(n)


def predict_chunk(rows, nt, cpt, cmin, cmax):
    """Python mirror of scorch_chunk() (given the already-computed nt)."""
    c = rows // (nt * cpt)
    if c < cmin:
        c = cmin
    if c > cmax:
        c = cmax
    return int(c)


def snap(value, grid):
    """Nearest value in the measured grid (ties -> smaller)."""
    return min(grid, key=lambda g: (abs(g - value), g))


# ------------------------------------------------------------------- sweep worker
def _synth_csr(n, nnz_per_row, seed):
    """Random n x n float32 CSR-ish dense tensor with ~nnz_per_row nonzeros/row."""
    import torch

    g = torch.Generator().manual_seed(seed)
    dens = min(1.0, nnz_per_row / n)
    mask = torch.rand(n, n, generator=g) < dens
    return (mask.float() * torch.randn(n, n, generator=g))


def _panel(kind):
    """(label, rows, nnz_per_row) points spanning the regime where the thread cap
    binds. Large/dense products already use all cores, so tuning there is moot;
    we weight small/mid where the machine's fork-join cliff actually shows."""
    rows_list = [128, 256, 512, 1024, 2048, 4096]
    # nnz/row MUST include the ultra-sparse tail (1) — the whole reason the work
    # grain exists is to stop 1-nnz/row products from over-threading (SuiteSparse
    # `lpi_klein3`: dropping the grain sent it to 32 threads -> 0.5x). A panel that
    # starts at 2 makes a low grain look free and the fit drops it, which would then
    # regress the real sparse tail. Keep 1 in so the grain is honestly constrained.
    nnz_list = [1, 2, 8, 32, 128]
    pts = []
    for r in rows_list:
        for z in nnz_list:
            if z >= r:
                continue
            pts.append((f"{kind}_n{r}_z{z}", r, z))
    return pts


def _grids(hw):
    nt = sorted({n for n in [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, hw] if 1 <= n <= hw})
    chunk = [4, 8, 16, 32, 64]
    return nt, chunk


def _median(xs):
    xs = sorted(xs)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def _time_call(fn, r):
    """Median wall-clock (ms) over ``r`` back-to-back calls."""
    best = []
    for _ in range(r):
        t0 = time.perf_counter()
        fn()
        best.append((time.perf_counter() - t0) * 1e3)
    return _median(best)


# A real A@A^T product can be far heavier (or explode) vs the bounded synthetic
# panel, and each matrix sweeps a full ~ncells x reps x rounds grid. Cap the
# per-matrix wall-clock so one big matrix can't dominate (or hang) the sweep:
# skip anything slower than _SLOW_SKIP_MS per call, and shed reps/rounds to keep
# each matrix under _MATRIX_BUDGET_MS. Tiny synthetic points are far below the cap
# so their reps/rounds are untouched (the synthetic path stays as-validated).
_SLOW_SKIP_MS = 150.0
_MATRIX_BUDGET_MS = 4000.0
# Cap the worker's virtual address space (Linux only) so a pathological A@A^T whose
# product explodes raises a catchable MemoryError instead of OOM-killing the whole
# sweep — the same guard the validated redwood driver (bench_spmspm_all_after.py)
# uses. The stratified panel's legitimate products are tiny (the large anchors run
# in ~1-2ms), so a modest cap fast-fails an explosion without swapping a shared box,
# while leaving huge headroom for every real panel matrix. Best-effort; skipped on
# macOS where RLIMIT_AS is unreliable.
_ADDR_SPACE_CAP_GB = 48

# Upper bound on the TRUE A@A^T flop (sum_k colcount(k)^2) for a spmspm panel matrix.
# The stored `work` uses AVERAGE degree (what the kernel's own thread estimate uses),
# which badly under-counts a skewed matrix — a few very high-degree columns make the
# product explode even when avg degree is tiny (circuit_*/scagr7-2r). Such a product
# can OOM-KILL or segfault the worker at kernel-call time, which try/except and even
# RLIMIT cannot always catch. So gate on the CHEAP true flop and skip explosive
# matrices BEFORE ever calling the kernel. Products this large also always want all
# cores, so they carry no thread-policy signal — losing them costs the fit nothing.
_MAX_SPMSPM_PRODUCT_FLOP = 50_000_000


def _cap_address_space():
    if not sys.platform.startswith("linux"):
        return
    try:
        import resource
        lim = _ADDR_SPACE_CAP_GB * 1024 ** 3
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except Exception:                           # noqa: BLE001
        pass


def _adapt_budget(per_call_ms, reps, rounds, ncells):
    r, rd = reps, rounds
    if per_call_ms <= 0:
        return r, rd
    while r * rd * ncells * per_call_ms > _MATRIX_BUDGET_MS and r > 1:
        r -= 1
    while r * rd * ncells * per_call_ms > _MATRIX_BUDGET_MS and rd > 1:
        rd -= 1
    return r, rd


def _measure_grid(label, n, work, call, ref_call, nt_grid, chunk_grid, reps, rounds):
    """Measure one matrix over the threads x chunk grid, back-to-back, forcing each
    cell via the SCORCH_TUNE_* env hooks. Returns the JSON entry, or None if the
    matrix is too slow to sweep. torch_ms cancels in default-vs-tuned ratios but is
    recorded so the reported speedups are interpretable."""
    os.environ.pop("SCORCH_TUNE_THREADS", None)
    os.environ.pop("SCORCH_TUNE_CHUNK", None)
    # Warm up (build any conversion/JIT kernels + caches) then estimate per-call cost.
    call()
    ref_call()
    t0 = time.perf_counter()
    call()
    per_call_ms = (time.perf_counter() - t0) * 1e3
    ncells = len(nt_grid) * len(chunk_grid)
    if per_call_ms > _SLOW_SKIP_MS:
        print(f"  [slow-skip] {label} ~{per_call_ms:.0f}ms/call "
              f"(> {_SLOW_SKIP_MS:.0f}ms)", flush=True)
        return None
    reps_eff, rounds_eff = _adapt_budget(per_call_ms, reps, rounds, ncells)

    cells = {f"{nt},{ch}": [] for nt in nt_grid for ch in chunk_grid}
    torch_ms = []
    for _rd in range(rounds_eff):
        torch_ms.append(_time_call(ref_call, reps_eff))
        for nt in nt_grid:
            os.environ["SCORCH_TUNE_THREADS"] = str(nt)
            for ch in chunk_grid:
                os.environ["SCORCH_TUNE_CHUNK"] = str(ch)
                cells[f"{nt},{ch}"].append(_time_call(call, reps_eff))
    os.environ.pop("SCORCH_TUNE_THREADS", None)
    os.environ.pop("SCORCH_TUNE_CHUNK", None)
    return {
        "rows": n,
        "work": work,
        "torch_ms": _median(torch_ms),
        "cells": {c: _median(v) for c, v in cells.items()},
    }


def _build_synth_matrix(kind, label, n, z, k_spmm):
    """Synthetic uniform-random point -> (label, rows, work, call, ref_call)."""
    import zlib

    import torch
    import scorch
    from scorch import STensor

    A = _synth_csr(n, z, seed=zlib.crc32(label.encode()) & 0xFFFF)  # deterministic
    sA = STensor.from_torch(A, "A").to_sparse("ds")
    a_nnz = int((A != 0).sum().item())
    if kind == "spmspm":
        At = A.t().contiguous()
        sB = STensor.from_torch(At, "B").to_sparse("ds")
        work = a_nnz * (a_nnz // n + 1)             # A_nnz * avg_B_row (flop)
        call = lambda: scorch.matmul(sA, sB)
        ref_call = lambda: (A @ At)
    else:                                           # spmm: A_sparse @ B_dense
        B = torch.randn(n, k_spmm)
        work = a_nnz * k_spmm
        call = lambda: scorch.matmul(sA, B)
        ref_call = lambda: (A @ B)
    return label, n, work, call, ref_call


def _build_real_matrix(kind, cand, k_spmm):
    """Real SuiteSparse matrix -> (label, rows, work, call, ref_call), or None if
    it fails to load. Builds the scorch input as "ds" CSR directly from the sparse
    arrays (no dense intermediate) and the SAME work estimate as the synthetic and
    prebuilt paths, so fit() consumes it unchanged."""
    import numpy as np
    import scipy.io
    import scipy.sparse
    import torch
    import scorch

    try:
        mat = scipy.io.mmread(str(cand.ref))
        csr = scipy.sparse.csr_matrix(mat, dtype=np.float32)
        d = min(csr.shape)                          # square-truncate (A@A^T, stable nnz)
        if csr.shape != (d, d):
            csr = csr[:d, :d].tocsr()
        n = csr.shape[0]
        a_nnz = int(csr.nnz)
        if n < 2 or a_nnz == 0:
            return None
        if kind == "spmspm":
            # Skip skewed-explosive products BEFORE building tensors or calling the
            # kernel (see _MAX_SPMSPM_PRODUCT_FLOP). colcount(k) = per-column nnz.
            colcnt = np.bincount(csr.indices, minlength=n).astype(np.int64)
            true_flop = int((colcnt * colcnt).sum())
            if true_flop > _MAX_SPMSPM_PRODUCT_FLOP:
                print(f"  [skip-huge] {cand.name}: est A@A^T flop {true_flop:.1e} "
                      f"> cap {_MAX_SPMSPM_PRODUCT_FLOP:.0e} (skewed product)",
                      flush=True)
                return None
            a_csr = _to_scorch_csr(csr, "A")
            b_csr = _to_scorch_csr(csr.T.tocsr(), "B")
            coo = _scipy_to_torch_coo(csr)
            coo_t = coo.t().coalesce()
            work = a_nnz * (a_nnz // n + 1)         # same flop estimate as synthetic
            call = lambda: scorch.matmul(a_csr, b_csr)
            ref_call = lambda: torch.matmul(coo, coo_t)
        else:                                       # spmm: A_sparse @ B_dense
            # spmm's product is dense m x k (always bounded) — no explosion gate.
            a_csr = _to_scorch_csr(csr, "A")
            B = torch.randn(n, k_spmm)
            coo = _scipy_to_torch_coo(csr)
            work = a_nnz * k_spmm
            call = lambda: scorch.matmul(a_csr, B)
            ref_call = lambda: torch.sparse.mm(coo, B)
        return f"{kind}_{cand.name}", n, work, call, ref_call
    except Exception as e:                          # noqa: BLE001
        print(f"  [skip] {cand.name}: {type(e).__name__}: {e}", flush=True)
        return None


def sweep_worker(out_path, kernels, reps, rounds, k_spmm,
                 matrices_dir=None, panel_size=DEFAULT_PANEL_SIZE):
    """Runs inside the instrumented build. Measures the threads x chunk grid for
    each panel matrix, back-to-back, and dumps JSON. The panel is a stratified set
    of REAL SuiteSparse matrices when ``matrices_dir`` is given (faithful for the
    work grains), else the synthetic uniform-random panel (chunk/topology only)."""
    import torch

    _cap_address_space()                        # survive one exploding product
    torch.manual_seed(0)
    hw = os.cpu_count() or 1
    nt_grid, chunk_grid = _grids(hw)
    result = {"hw": hw, "nt_grid": nt_grid, "chunk_grid": chunk_grid, "kernels": {}}

    if matrices_dir:
        panel = _real_panel(matrices_dir, panel_size)
        result["panel"] = f"real:{matrices_dir}"
        result["n_matrices"] = len(panel)
        if not panel:
            raise SystemExit(
                f"no usable matrices under {matrices_dir} after header pre-filter "
                f"(max-dim<{MAX_DIM}, nnz<{MAX_NNZ})")
        print(f"[panel] REAL: {len(panel)} SuiteSparse matrices stratified from "
              f"{matrices_dir}", flush=True)
        _summarize_panel(panel)
    else:
        panel = None
        result["panel"] = "synthetic"
        result["n_matrices"] = len(_panel(kernels[0])) if kernels else 0
        print("[panel] SYNTHETIC uniform-random (chunk/topology faithful; work "
              "grains NOT faithfully tunable — see fidelity note)", flush=True)

    for kind in kernels:
        entries = {}
        if panel is not None:
            for cand in panel:
                # Guard every matrix: a single exploding product (caught as
                # MemoryError via the RLIMIT cap) or a torch/scorch error must skip
                # that matrix, never abort the sweep.
                try:
                    built = _build_real_matrix(kind, cand, k_spmm)
                    if built is None:
                        continue
                    label, n, work, call, ref_call = built
                    entry = _measure_grid(label, n, work, call, ref_call,
                                          nt_grid, chunk_grid, reps, rounds)
                except Exception as e:          # noqa: BLE001
                    os.environ.pop("SCORCH_TUNE_THREADS", None)
                    os.environ.pop("SCORCH_TUNE_CHUNK", None)
                    print(f"  [skip] {cand.name}: {type(e).__name__}: {e}", flush=True)
                    continue
                if entry is None:
                    continue
                entries[label] = entry
                print(f"  [{kind}] {label:30s} work={work:>11d} "
                      f"torch={entry['torch_ms']:.3f}ms "
                      f"best={min(entry['cells'].values()):.3f}ms", flush=True)
        else:
            for label, n, z in _panel(kind):
                label, n, work, call, ref_call = _build_synth_matrix(
                    kind, label, n, z, k_spmm)
                entry = _measure_grid(label, n, work, call, ref_call,
                                      nt_grid, chunk_grid, reps, rounds)
                if entry is None:
                    continue
                entries[label] = entry
                print(f"  [{kind}] {label:18s} work={work:>10d} "
                      f"torch={entry['torch_ms']:.3f}ms "
                      f"best={min(entry['cells'].values()):.3f}ms", flush=True)
        result["kernels"][kind] = entries

    Path(out_path).write_text(json.dumps(result))
    print(f"sweep -> {out_path}", flush=True)


# -------------------------------------------------------------------------- fit
def _score_theta(entries, grain, rpt, cpt, cmin, cmax, nt_grid, chunk_grid, hw):
    """Per-matrix geomean speedup for a candidate constant-set over one kernel's
    measured grid. Predicted (nt,chunk) is snapped to the nearest measured cell."""
    logs = []
    for e in entries.values():
        nt = predict_nthreads(e["work"], e["rows"], grain, rpt, hw)
        ch = predict_chunk(e["rows"], nt, cpt, cmin, cmax)
        cell = f"{snap(nt, nt_grid)},{snap(ch, chunk_grid)}"
        ms = e["cells"][cell]
        if ms <= 0:
            continue
        logs.append(math.log(e["torch_ms"] / ms))
    return math.exp(sum(logs) / len(logs)) if logs else 0.0


# Candidate space for the offline fit (cheap — pure lookups over the measured grid).
GRAIN_CANDIDATES = {
    "spmspm": [500, 1000, 1500, 3000, 6000, 12000, 30000],
    "spmm": [30000, 75000, 150000, 300000, 600000],
}
RPT_CANDIDATES = [8, 12, 16, 24, 32, 48]
CPT_CANDIDATES = [4, 7, 10, 14]
CMIN_CANDIDATES = [2, 4, 8]
CMAX_CANDIDATES = [32, 64, 128]


def fit(measurements, kernels, freeze_grains=False):
    """Joint grid-search: shared (rpt,cpt,cmin,cmax) across kernels + a per-kernel
    grain. Objective = geomean of per-kernel geomean-speedups. Returns (best,
    default) constant-sets and their scores on the identical measurements.

    freeze_grains restricts each grain to its shipped default, so only the
    machine-topology knobs (rows/thr, chunks/thr, chunk bounds) are tuned. Use it
    when you don't want a thin synthetic panel re-deriving the SuiteSparse-validated
    work grains (which encode the matrix-shape distribution, not just the machine)."""
    hw = measurements["hw"]
    nt_grid = measurements["nt_grid"]
    chunk_grid = measurements["chunk_grid"]
    kdata = {k: measurements["kernels"][k] for k in kernels}
    grain_choices = {
        k: ([DEFAULTS[f"SCORCH_GRAIN_{k.upper()}"]] if freeze_grains
            else GRAIN_CANDIDATES[k])
        for k in kernels
    }

    def combined(rpt, cpt, cmin, cmax):
        per_kernel = {}
        for k in kernels:
            best_g, best_s = None, -1.0
            for g in grain_choices[k]:
                s = _score_theta(kdata[k], g, rpt, cpt, cmin, cmax,
                                 nt_grid, chunk_grid, hw)
                if s > best_s:
                    best_g, best_s = g, s
            per_kernel[k] = (best_g, best_s)
        score = math.exp(sum(math.log(max(s, 1e-9)) for _, s in per_kernel.values())
                         / len(per_kernel))
        return score, per_kernel

    best = None
    for rpt in RPT_CANDIDATES:
        for cpt in CPT_CANDIDATES:
            for cmin in CMIN_CANDIDATES:
                for cmax in CMAX_CANDIDATES:
                    if cmin >= cmax:
                        continue
                    score, per_kernel = combined(rpt, cpt, cmin, cmax)
                    cand = (score, rpt, cpt, cmin, cmax, per_kernel)
                    if best is None or score > best[0]:
                        best = cand

    # Default constant-set scored on the SAME measurements (honest baseline).
    d = DEFAULTS
    def_per_kernel = {}
    for k in kernels:
        s = _score_theta(kdata[k], d[f"SCORCH_GRAIN_{k.upper()}"],
                         d["SCORCH_ROWS_PER_THREAD"], d["SCORCH_CHUNKS_PER_THREAD"],
                         d["SCORCH_CHUNK_MIN"], d["SCORCH_CHUNK_MAX"],
                         nt_grid, chunk_grid, hw)
        def_per_kernel[k] = (d[f"SCORCH_GRAIN_{k.upper()}"], s)
    def_score = math.exp(sum(math.log(max(s, 1e-9)) for _, s in def_per_kernel.values())
                         / len(def_per_kernel))

    score, rpt, cpt, cmin, cmax, per_kernel = best
    tuned = {
        "SCORCH_ROWS_PER_THREAD": rpt,
        "SCORCH_CHUNKS_PER_THREAD": cpt,
        "SCORCH_CHUNK_MIN": cmin,
        "SCORCH_CHUNK_MAX": cmax,
    }
    for k in kernels:
        tuned[f"SCORCH_GRAIN_{k.upper()}"] = per_kernel[k][0]
    return {
        "tuned": tuned, "tuned_score": score, "tuned_per_kernel": per_kernel,
        "default_score": def_score, "default_per_kernel": def_per_kernel,
    }


# ------------------------------------------------------------------ header write
def write_tuned_header(tuned, meta):
    lines = [
        "// GENERATED by tools/autotune_policy.py — DO NOT EDIT, DO NOT COMMIT.",
        "// Per-host OpenMP policy constants measured on this build machine.",
        f"// host: {meta['host']}  logical_cores: {meta['hw']}",
        f"// tuned_geomean_speedup: {meta['tuned_score']:.3f}  "
        f"vs default: {meta['default_score']:.3f}  (+{meta['gain_pct']:.1f}%)",
        "// Delete this file (or run pip install without the autotune) to revert",
        "// to the shipped redwood defaults in scorch_policy.h.",
        "#pragma once",
    ]
    for key in ("SCORCH_GRAIN_SPMSPM", "SCORCH_GRAIN_SPMM", "SCORCH_ROWS_PER_THREAD",
                "SCORCH_CHUNKS_PER_THREAD", "SCORCH_CHUNK_MIN", "SCORCH_CHUNK_MAX"):
        if key in tuned:
            lines.append(f"#define {key} {tuned[key]}L")
    TUNED_H.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------- builds
def build_scorch_ops(tune_hooks):
    env = dict(os.environ)
    if tune_hooks:
        env["SCORCH_BUILD_TUNE_HOOKS"] = "1"
    else:
        env.pop("SCORCH_BUILD_TUNE_HOOKS", None)
    (REPO / "csrc" / "ops.cpp").touch()  # header edits don't retrigger ops.cpp compile
    label = "instrumented (-DSCORCH_TUNE_HOOKS)" if tune_hooks else "clean"
    print(f"[build] scorch_ops {label} ...", flush=True)
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation"],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:])
        raise SystemExit(f"build failed (rc={r.returncode})")


# ---------------------------------------------------------------- panel resolve
def _resolve_panel(args):
    """Resolve the real-matrix panel directory (and optionally download into it).
    Resolution order: --matrices arg -> $SCORCH_SUITESPARSE -> /scratch/suitesparse
    (if present) -> ~/.cache/scorch/suitesparse. Returns the dir to sweep, or None
    to fall back (loudly) to the synthetic panel."""
    if args.matrices:
        path = Path(args.matrices).expanduser()
    else:
        env = os.environ.get("SCORCH_SUITESPARSE")
        if env:
            path = Path(env).expanduser()
        elif Path("/scratch/suitesparse").is_dir():
            path = Path("/scratch/suitesparse")
        else:
            path = Path(DEFAULT_MATRIX_CACHE).expanduser()

    if args.download is not None:
        download_matrices(path, args.download)

    if _discover_mtx(path):
        print(f"[panel] using REAL SuiteSparse matrices from {path}")
        return str(path)

    # Nothing usable. An explicit --matrices to an empty dir (and no --download) is
    # a user error; otherwise fall back to synthetic with a loud fidelity warning.
    if args.matrices and args.download is None:
        raise SystemExit(
            f"--matrices {path} contains no SuiteSparse .mtx files "
            "(looked for <name>/<name>.mtx). Pass --download N to fetch a sample, "
            "or drop --matrices to use the synthetic panel.")
    bar = "!" * 72
    print(f"\n[panel] {bar}")
    print("[panel] No SuiteSparse matrices found and --download not given.")
    print(f"[panel]   (looked in {path})")
    print("[panel] Falling back to the SYNTHETIC uniform-random panel.")
    print("[panel] Synthetic faithfully tunes the CHUNK / topology knobs but NOT the")
    print("[panel] work GRAINS (uniform data lacks SuiteSparse's skewed ultra-sparse")
    print("[panel] tail — see the fidelity finding in this file's docstring). Pass")
    print("[panel] --matrices PATH or --download N for a real-matrix panel that can")
    print("[panel] tune the grains trustworthily.")
    print(f"[panel] {bar}\n")
    return None


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-worker", metavar="OUT.json",
                    help="internal: run the measurement sweep and dump JSON")
    ap.add_argument("--kernels", default="spmspm,spmm",
                    help="comma list of kernels to tune (spmspm,spmm)")
    ap.add_argument("--reps", type=int, default=5, help="timed repeats per cell (median)")
    ap.add_argument("--rounds", type=int, default=2, help="grid passes (median across)")
    ap.add_argument("--k-spmm", type=int, default=128, help="dense width k for spmm panel")
    ap.add_argument("--matrices", metavar="PATH",
                    help="directory of SuiteSparse .mtx files (canonical layout "
                    "<name>/<name>.mtx). Use a REAL-matrix panel — faithful for the "
                    "work grains — instead of the synthetic one. Also honors env "
                    "SCORCH_SUITESPARSE; auto-detects /scratch/suitesparse.")
    ap.add_argument("--download", nargs="?", type=int, const=DEFAULT_DOWNLOAD_N,
                    default=None, metavar="N",
                    help="fetch a stratified ~N-matrix SuiteSparse sample (incl. the "
                    "ultra-sparse tail) into the matrices dir if absent (default "
                    f"N={DEFAULT_DOWNLOAD_N}). Uses ssgetpy, falls back to the TAMU mirror.")
    ap.add_argument("--panel-size", type=int, default=DEFAULT_PANEL_SIZE,
                    help="target size of the stratified real-matrix panel (~30-60)")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="min fractional gain over defaults to adopt the tune")
    ap.add_argument("--freeze-grains", action="store_true",
                    help="tune only machine-topology knobs (rows/thr, chunks/thr, "
                    "chunk bounds); keep the SuiteSparse-validated work grains")
    ap.add_argument("--dry-run", action="store_true",
                    help="sweep + fit + report, but write no header and don't rebuild")
    ap.add_argument("--keep-json", metavar="PATH", help="save the raw sweep JSON here")
    args = ap.parse_args()

    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]

    # torch's JIT (a to_sparse conversion kernel is compiled while building panel
    # inputs) defaults its build dir under $HOME, which can be unwritable (e.g.
    # redwood's AFS home). Pin it to a writable repo-local dir unless the caller set
    # one. Must happen before torch is imported (here, and inherited by the worker).
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(REPO / "build" / "torch_ext"))
    Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)

    if args.sweep_worker:
        sweep_worker(args.sweep_worker, kernels, args.reps, args.rounds, args.k_spmm,
                     matrices_dir=args.matrices, panel_size=args.panel_size)
        return

    # Resolve the matrix panel (and optionally download) in the PARENT, then pass
    # the resolved dir to the worker. Falls back to synthetic (loudly) when no
    # dataset is present and --download wasn't given.
    matrices_dir = _resolve_panel(args)

    import platform
    host = platform.node()
    json_path = args.keep_json or str(
        REPO / "build" / "autotune_sweep.json")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. instrumented build so the sweep can force any (nt, chunk) in-process
    build_scorch_ops(tune_hooks=True)

    # Everything after the instrumented build MUST end on a clean build — even on
    # failure — so a crashed autotune never leaves the machine instrumented.
    clean_restored = False
    try:
        # 2. measure this host (subprocess = fresh import of the instrumented build)
        print("[sweep] measuring this host (back-to-back, never cross-time) ...",
              flush=True)
        worker_argv = [
            sys.executable, __file__, "--sweep-worker", json_path,
            "--kernels", ",".join(kernels), "--reps", str(args.reps),
            "--rounds", str(args.rounds), "--k-spmm", str(args.k_spmm),
            "--panel-size", str(args.panel_size)]
        if matrices_dir:
            worker_argv += ["--matrices", matrices_dir]
        worker = subprocess.run(worker_argv, cwd=REPO, env=os.environ)
        if worker.returncode != 0:
            raise SystemExit("sweep worker failed")
        measurements = json.loads(Path(json_path).read_text())
        panel_desc = measurements.get("panel", "synthetic")
        real_panel = panel_desc.startswith("real")
        n_matrices = measurements.get("n_matrices", 0)

        # 3. offline fit against the measured grid
        res = fit(measurements, kernels, freeze_grains=args.freeze_grains)
        gain = res["tuned_score"] / res["default_score"] - 1.0
        print("\n==== autotune result "
              f"({host}, {measurements['hw']} logical cores) ====")
        print(f"  panel: {'REAL SuiteSparse' if real_panel else 'SYNTHETIC uniform-random'}"
              f" ({n_matrices} matrices)"
              + ("" if real_panel else "  — grains NOT faithfully tunable"))
        for k in kernels:
            tg, ts = res["tuned_per_kernel"][k]
            dg, ds = res["default_per_kernel"][k]
            print(f"  {k:7s}  default grain={dg:<7d} geomean={ds:.3f}   ->   "
                  f"tuned grain={tg:<7d} geomean={ts:.3f}")
        print(f"  shared: rows/thr={res['tuned']['SCORCH_ROWS_PER_THREAD']} "
              f"chunks/thr={res['tuned']['SCORCH_CHUNKS_PER_THREAD']} "
              f"chunk=[{res['tuned']['SCORCH_CHUNK_MIN']},{res['tuned']['SCORCH_CHUNK_MAX']}]  "
              f"(defaults 16/7/[4,64])")
        print(f"  combined geomean speedup  default={res['default_score']:.3f}  "
              f"tuned={res['tuned_score']:.3f}  gain={gain*100:+.1f}%")

        if args.dry_run:
            print("\n[dry-run] no header written. Restoring a clean build.")
        elif gain < args.margin:
            # 4a. keep the shipped defaults — the tune didn't clear the margin
            print(f"\n[keep-defaults] gain {gain*100:+.1f}% < margin "
                  f"{args.margin*100:.1f}% — the defaults already fit this host; "
                  "not writing a tuned header.")
            if TUNED_H.exists():
                TUNED_H.unlink()
                print("  removed a stale scorch_policy_tuned.h")
        else:
            # 4b. adopt — the tune wins by the margin
            grain_changes = [
                (k, DEFAULTS[f"SCORCH_GRAIN_{k.upper()}"],
                 res["tuned"][f"SCORCH_GRAIN_{k.upper()}"])
                for k in kernels
                if res["tuned"][f"SCORCH_GRAIN_{k.upper()}"]
                != DEFAULTS[f"SCORCH_GRAIN_{k.upper()}"]
            ]
            if grain_changes and real_panel:
                # A REAL panel sees the skewed tail (stronger evidence than synthetic),
                # BUT a modest panel's unweighted geomean can be dominated by a few
                # low-work/many-row outliers and pick a smaller grain that over-threads
                # the MID-BAND into a hybrid CPU's E-core cliff (observed on redwood: a
                # 40-matrix real fit wanted spmspm 3000->500 for +8.8% geomean, yet
                # per-matrix it regressed the mid-band up to 2.4x; ~all the win was the
                # topology knobs, and grain 3000 was best at the default topology). So a
                # real-panel grain change is better-founded than synthetic but still not
                # a blank check.
                print("\n[note] the fit changed a work grain (REAL panel, "
                      f"{n_matrices} matrices): "
                      + ", ".join(f"{k} {d}->{t}" for k, d, t in grain_changes) + ".")
                print("  Stronger evidence than a synthetic panel, but a modest panel's"
                      " geomean can be")
                print("  outlier-dominated and a smaller grain risks OVER-THREADING the"
                      " mid-band on hybrid")
                print("  CPUs. Inspect the per-matrix breakdown / use a larger,"
                      " distribution-faithful panel")
                print("  (bigger --panel-size), or --freeze-grains, before trusting a"
                      " grain drop.")
            elif grain_changes:
                # The work grains encode the SuiteSparse matrix-shape distribution, not
                # just this machine. A uniform-random synthetic panel can't see the
                # skewed ultra-sparse tail the default grain protects, so it tends to
                # pick a smaller grain (more threads) that may REGRESS real workloads.
                print("\n[WARNING] the fit changed a work grain "
                      + ", ".join(f"{k} {d}->{t}" for k, d, t in grain_changes) + ".")
                print("  Grains are SuiteSparse-distribution-sensitive; a synthetic panel"
                      " can over-thread the")
                print("  real skewed tail. VALIDATE against a real-matrix sweep (rerun with"
                      " --matrices PATH")
                print("  or --download N) before trusting this, or re-run with"
                      " --freeze-grains to tune")
                print("  only the machine-topology knobs (chunk sizing).")
            meta = {"host": host, "hw": measurements["hw"],
                    "tuned_score": res["tuned_score"],
                    "default_score": res["default_score"], "gain_pct": gain * 100}
            write_tuned_header(res["tuned"], meta)
            print(f"\n[write] {TUNED_H.relative_to(REPO)}")

        # 5. clean rebuild: picks up the tuned constants (if written); JIT cache
        #    busts automatically (policy-header text is in the kernel-name hash).
        build_scorch_ops(tune_hooks=False)
        clean_restored = True
        if not args.dry_run and TUNED_H.exists():
            print("[done] tuned + clean-built. Prebuilt kernels use the new "
                  "constants; JIT kernels re-tune on next compile.")
    finally:
        if not clean_restored:
            print("[cleanup] restoring a clean (un-instrumented) build after failure ...")
            try:
                build_scorch_ops(tune_hooks=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[cleanup] clean rebuild FAILED ({exc}); run "
                      "`pip install -e . --no-build-isolation` to restore.")


if __name__ == "__main__":
    main()
