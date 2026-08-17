# Autotuning

Scorch exposes a compiler-style `-O` optimization ladder over its sparse
matrix–dense matrix (SpMM) dispatch. You dial it with a single knob — the
**autotune level** — trading a little dispatch-time overhead for faster
execution on the workloads that can actually benefit. The default is chosen so
that autotuning is *provably neutral* on everything it touches: it never makes a
call slower than the baseline kernel.

This page covers what autotuning controls, the level ladder, the Python and
environment API, and the tiling strategies underneath — including the
no-regression design that lets the feature ship on by default.

## Scope

Autotuning is deliberately narrow. It controls **only** how the drop-in
**CSR-sparse × dense** SpMM path of {func}`~scorch.matmul` is dispatched — that
is, `C = A @ B` where `A` is a CSR ({class}`~scorch.STensor` in `"ds"` format)
and `B` is a dense `torch.Tensor`.

It does **not** control:

- `einsum`, `spmv`, SDDMM, SpGEMM, or any other operation;
- the general JIT compiler pipeline (CIN → LLIR → codegen);
- anything about dense–dense matmul (that path delegates straight to
  `torch.matmul`).

Under the hood, the autotune level selects among a small set of hand-written
**prebuilt** SpMM kernels in the native `scorch_ops` extension, for a single
well-gated regime (high-degree matrices whose dense operand thrashes the
last-level cache). Every other shape falls through to the byte-identical default
kernel, `spmm_csr_float_v2`.

:::{note}
If you are not calling {func}`~scorch.matmul` with a CSR operand and a dense
right-hand side, the autotune level has no effect on your program.
:::

## The level ladder

The five levels form an `-O` ladder trading dispatch overhead for execution
speed. The **eligibility gate is identical at every non-`off` level** — only the
decision made for an *already-eligible* shape changes.

| Level | Behavior |
|-------|----------|
| `off` | No tiling. Short-circuits to the pure `v2` baseline — the cheapest possible path. |
| `analytic` | **Default.** A cost-model pick (tile-j at base width, or tile-ijk at wide N), then **one confirming measurement against `v2`** before the pick is memoized — 6 kernel invocations, against `balanced`'s 18. No width search. |
| `balanced` | First-call **micro-probe** over `{v2, tile-j@{base,/2,/4,/8}, tile-ijk}`, memoized in-process. `v2` is always a candidate, so it is never slower than `v2`. |
| `max` | `balanced` probe **plus a persistent on-disk cache** (per machine), so the search is paid once ever, amortized across processes and runs. |
| `learned` | **Experimental.** An offline-trained gradient-boosted-tree cost model predicts each candidate's runtime in O(1). **Falls back to `analytic`** when no per-machine model file is installed. |

A few distinctions worth internalizing:

`analytic` vs `learned`
: Both pick a candidate from a cost model rather than searching, then confirm that
  one pick against `v2`. `analytic` uses a hand-written DRAM-byte model; `learned`
  uses a trained model over the same candidate set. `learned` silently degrades to
  `analytic` unless a model has been trained and installed for your machine.

  Neither used to confirm, and that was a real defect rather than a design
  trade-off: over a 236-cell grid on redwood, `analytic` shipped six tiled-route
  regressions — worst 0.373x of untiled on `audikw_1` at N=128, against a 1.9%
  noise floor — and `learned` four, while `balanced` (same gate, but it times
  `v2`) had none. The gate cannot be tightened out of it with the features
  available: the span proxy reads 0.823 on a matrix that loses (`crankseg_1`) and
  0.823 on one that wins (`mouse_gene`).

`balanced` vs `max`
: Same first-call probe. `max` adds a persistent JSON cache so the probe cost is
  amortized across process restarts, not just within one process.

The `v2` floor
: In `balanced`/`max`, `v2` is a *timed* candidate. In `analytic`/`learned` the
  cost model must first predict that a tiled kernel wins, and then that single
  pick is *timed* against `v2` once per shape and discarded if it loses. Either
  way the route that gets memoized has been measured to be no slower than not
  tiling, so the whole ladder is no-regression by construction rather than by the
  cheap pre-filter happening to be right.

:::{warning}
`learned` is Phase 2 and only partially landed. Treat it as experimental: without
a trained model file for the current machine it behaves exactly like `analytic`.
:::

## The API

Four public functions, all re-exported at the top level of `scorch`.

{func}`~scorch.set_autotune`
: Set the **process-global** default level. Validates the argument (an unknown
  level raises `ValueError`; a non-string raises `TypeError`).

{func}`~scorch.get_autotune`
: Return the **effective** level — a thread-local override if one is active,
  otherwise the process-global level.

{class}`~scorch.autotune`
: A thread-local scope that works as **both a context manager and a decorator**,
  mirroring `torch.no_grad`. As a context manager it overrides the level for the
  duration of the block and restores the previous value on exit; as a decorator
  each call runs inside a fresh scope. Being thread-local, it nests correctly and
  never affects other threads.

{func}`~scorch.clear_autotune_cache`
: Wipe the **persistent on-disk** probe cache written by the `max` level. It does
  *not* clear the in-process decision memo (which lives for the process lifetime)
  nor the compiled `.so` kernel cache. Best-effort — a missing cache file is fine.

### A runnable example

```python
import torch
import scorch

# The effective level. Default is "analytic" unless SCORCH_AUTOTUNE or a legacy
# env var is set.
print(scorch.get_autotune())          # -> "analytic"

# --- 1. Process-global level ---------------------------------------------------
scorch.set_autotune("off")            # pure v2 baseline, no tiling
scorch.set_autotune("balanced")       # first-call micro-probe, memoized in-process
print(scorch.get_autotune())          # -> "balanced"

# --- 2. Scoped override (context manager, mirrors torch.no_grad) ---------------
A = scorch.from_torch(
    torch.tensor([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]]), "A"
)                                      # CSR-sparse operand
B = torch.randn(3, 128)               # dense right operand

with scorch.autotune("max"):          # balanced probe + persistent on-disk cache
    C = scorch.matmul(A, B)           # tiling only engages for LLC-thrashing shapes;
                                      # this small matrix routes to v2 regardless.
print(scorch.get_autotune())          # back to "balanced" outside the block

# The result is byte-identical to the dense reference at every level.
ref = torch.tensor([[1., 0., 2.], [0., 3., 0.], [4., 0., 5.]]) @ B
assert torch.allclose(C.to_torch(), ref, atol=1e-3, rtol=1e-3)

# --- 3. Decorator form ---------------------------------------------------------
@scorch.autotune("analytic")
def run(A, B):
    return scorch.matmul(A, B)

C = run(A, B)

# --- 4. Wipe the persistent "max" cache (per-machine JSON) ---------------------
scorch.clear_autotune_cache()

# Invalid levels raise (ValueError for unknown, TypeError for non-str):
# scorch.set_autotune("aggressive")   # ValueError: unknown autotune level
```

:::{note}
On a small matrix like the one above, the cache pre-filter (`J·4·N > C`) fails,
so the result is byte-identical to `v2` at every level — autotuning is a no-op
there **by design**. A visible effect requires a high-degree operand large enough
to thrash the last-level cache (reddit/products-class graphs) at moderate-to-wide
free dimension `N`.
:::

## Environment overrides

The Python API is primary; environment variables exist for override and CI. The
initial global level resolves as: `SCORCH_AUTOTUNE` wins if set and valid; else
the legacy `SCORCH_TILING*` vars map on; else the built-in default `analytic`.

| Environment variable | Effect |
|----------------------|--------|
| `SCORCH_AUTOTUNE` | Initial global level (`off` / `analytic` / `balanced` / `max` / `learned`). |
| `SCORCH_AUTOTUNE_CACHE=<path>` | Persistent-cache location; `=0` disables read and write. |
| `SCORCH_TILING=0` | Legacy: maps to `off`. |
| `SCORCH_TILING_PROBE=0` / `=1` | Legacy: maps to `analytic` / `balanced`. |
| `SCORCH_LLC_BYTES=<n>` | Override the queried last-level-cache size (gate knob). |
| `SCORCH_TILING_DEG_FLOOR` / `_NIJK_MIN` / `_LOC_MIN` | Gate knobs (defaults 64 / 512 / 0.3). |
| `SCORCH_AUTOTUNE_MODEL=<path>` / `=0` | Learned model path override / disable. |
| `SCORCH_AUTOTUNE_WIDEN=0` | Learned: revert to the analytic gate (default is widened). |
| `SCORCH_AUTOTUNE_CONFIRM=0` | Skip the one-shot `v2`-confirm that `analytic` and `learned` now always perform. For A/B measurement only — it restores a configuration that is known to regress up to 2.68x. |
| `SCORCH_AUTOTUNE_MARGIN` | Learned `v2`-floor margin (default `0.03`). |

Set a level for a whole run without touching code:

```bash
SCORCH_AUTOTUNE=max python train.py
```

## The tiling strategies

The selector routes among three prebuilt kernels for the computation
$C_{ik} = \sum_j A_{ij} B_{jk}$, where `A` is CSR sparse, `B` is dense, `i` is
the row axis (M), `j` the contraction axis (J), and `k` the free dimension (N).

`v2` — the baseline (`spmm_csr_float_v2`)
: The default drop-in kernel. It streams each sparse row against the *full* dense
  operand `B`. This is optimal when `B` (of size `J·4·N` bytes) fits the
  last-level cache — which covers essentially every GCN and autoencoder shape
  (hidden dims 16–256, batch 256). Everything else falls back here, byte-for-byte.

`tile-j` — contraction-axis cache blocking (`spmm_csr_float_tilej`)
: Splits `B` into column panels of width `Jc` so each panel's rows
  (`Jc·4·N` bytes) fit the LLC, recovering cross-row reuse of `B`. It helps when
  `B` **thrashes** the cache *and* the matrix has **high degree** — enough
  per-column reuse to pay back the extra output re-traffic. The catch: it
  re-streams the output `C` once per panel, so its cost grows with `N` and it
  erodes at very wide free dimensions.

`tile-ijk` — B width-panel relayout (`spmm_csr_float_tileijk`)
: For the very-wide-`B` tail, it relays `B` into contiguous `Nc`-wide free-dim
  strips so the output traffic stays linear in `N` instead of quadratic, at the
  cost of an O(J·N) relayout plus re-scanning `A`. The relayout is done *inside*
  the kernel per call, so a probe honestly times the full cost. It joins the
  candidate set **only** once `N ≥ 512` — above every current Scorch workload — so
  it is provably inert on GCN/AE/attention and only ever a win on the
  general-library wide-`B` case.

### When each helps

| Regime | Winner |
|--------|--------|
| `B` fits the LLC (`J·4·N ≤ C`): all GCN-small, autoencoder, attention, FEM, arxiv | `v2` |
| `B` thrashes the LLC **and** high degree **and** scattered, moderate `N` | `tile-j` |
| Same, but **wide `N`** (≥ 512) where tile-j's re-traffic erodes | `tile-ijk` |
| Well-ordered / banded high-degree (FEM) | `v2` (the locality gate excludes tiling) |

## The no-regression design

The selector is no-regression **by construction** — this is the whole point of
shipping it on by default. Three mechanisms combine:

1. **`v2` is always a candidate.** In every measured decision (`balanced`/`max`)
   the baseline kernel is timed alongside the tiled ones, and the fastest wins. A
   tiled kernel is chosen only when it actually beats `v2`.

2. **An O(1) gate routes the common case straight to `v2`.** Before any dispatch
   closure is built, a cheap integer-comparison pre-filter (the *thrash-and-tile*
   rule: the operand must exceed the cache size *and* the average degree must
   clear a floor) sends the vast majority of shapes — all GCN-small, autoencoder,
   FEM, and arxiv workloads — to the byte-identical `v2` path at integer cost. A
   sampled locality check additionally keeps well-ordered banded matrices (which
   `v2` already streams from cache) off the tile path.

3. **The `tile-ijk` gate is provably inert where it could hurt.** Its `N ≥ 512`
   threshold sits above every free dimension Scorch's own workloads use, so it can
   never fire on a shape it would slow down.

This is a textbook instance of Scorch's performance convention: an optimization
that only helps a sub-regime is gated behind a runtime condition that *provably
cannot fire* on the shapes it would hurt. The result generalizes — neutral or
better across narrow-`N` and wide-`N` SpMM, small and large row counts, sparse
and near-dense — with no regressions on the GCN, autoencoder, and attention
families. Across the CGO 2026 paper's suite, Scorch achieved **1.05–5.80× over
PyTorch Sparse** on sparse-matrix and graph-neural-network workloads.

## See also

- {doc}`Performance tuning guide </performance/tuning_guide>` — the broader
  playbook for getting the most out of Scorch, including JIT warm-up and the
  `.so` cache.
- {doc}`Benchmarks </performance/benchmarks>` — the workloads behind the numbers.
- {doc}`Operations </user_guide/operations>` — the {func}`~scorch.matmul` path
  autotuning dispatches.
