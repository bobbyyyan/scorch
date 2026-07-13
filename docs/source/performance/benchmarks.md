# Benchmarks

Scorch ships a suite of reproducible benchmark harnesses under `bench/` that
measure both the low-level sparse kernels (SpMV, SpMM, SDDMM, SpMSpM) and the
end-to-end application workloads Scorch is designed for (graph neural networks,
weight-sparse autoencoders, sparse attention, and graph analytics). This page
gives the headline result, describes the workload families, and provides an
inventory of every harness so you can rerun the numbers on your own machine.

## Headline result

In the benchmarks reported in the [Scorch CGO 2026
paper](https://ieeexplore.ieee.org/abstract/document/11394842), Scorch achieved
**1.05–5.80× speedups over PyTorch Sparse** across sparse-matrix and graph
neural network workloads.

That is the only speedup figure this page quotes. Absolute numbers depend on
the CPU, the last-level-cache size, the thread pool, and the matrix, so the
harnesses below are the source of truth for your own hardware — run them rather
than porting a headline number between machines.

:::{note}
Scorch is a CPU sparse-tensor library. Its wins come from JIT-specialized,
OpenMP-parallel C++ kernels generated per operation and format combination, not
from a GPU backend. The first call to an operation compiles a kernel (~7 s);
every subsequent call — including across process restarts — loads the cached
shared library from disk and runs at full speed. Benchmark harnesses warm up
before timing so compilation cost is never counted in a measured result.
:::

## What the design guarantees (and does not)

Scorch's performance convention is that **optimizations must generalize** — a
change ships only if it is neutral-or-better across the whole workload space
(narrow- and wide-free-dim SpMM, small and large row counts, sparse and
near-dense, and the GCN / autoencoder / attention families) with no regressions
on any of them. Sub-regime optimizations are gated behind runtime conditions
that provably cannot fire on the shapes they would hurt. The autotune tiling
selector is the textbook instance: it is inert by construction on everything
outside the high-degree, cache-thrashing tail. See
{doc}`/user_guide/autotuning` for the user-facing knob and
{doc}`/performance/tuning_guide` for how to reason about a given shape.

## Workload families

The suite is organized around five families, each with a dedicated harness.

Sparse matrices
: Core-kernel microbenchmarks over curated and SuiteSparse matrices —
  `A_sparse @ B_dense` (SpMM), `A_sparse @ v_dense` (SpMV),
  `A_sparse @ A_sparseᵀ` (SpMSpM / SpGEMM), and
  `S ⊙ (A_dense @ B_dense)` (SDDMM). These isolate the compiled kernels from
  any framework overhead.

Graph neural networks (GCN)
: A standard Kipf & Welling 2-layer GCN with symmetric normalization, compared
  against PyTorch, PyG, and DGL. The dominant cost is
  sparse-adjacency × dense-features SpMM. See {doc}`/tutorials/gcn`.

Sparse autoencoder
: Weight-sparse autoencoder inference. A dense autoencoder is trained, its
  weights are magnitude-pruned to a target sparsity, and inference runs as
  sparse-weight × dense-activation SpMM across frameworks. See
  {doc}`/tutorials/sparse_autoencoder`.

Sparse attention
: A Longformer-style attention pattern (sliding window + global tokens) on
  IMDB, contrasting the sparse-attention $O(n)$ scaling against dense $O(n^2)$
  attention as sequence length grows. See {doc}`/tutorials/sparse_transformer`.

Graph analytics
: Genuinely sparse × sparse SpMSpM workloads — triangle counting and multi-hop
  link prediction — that nothing else in the suite exercises.

## The harnesses

Every script under `bench/` is a standalone `#!/usr/bin/env python3` program
with a module docstring describing what it measures. Activate the `scorch`
conda environment first (the harnesses import the compiled extension):

```bash
conda activate scorch
```

### End-to-end application workloads

| Script | What it measures | How to run |
|--------|------------------|------------|
| `bench_gcn.py` | GCN inference (2-layer, symmetric-normalized): Scorch vs PyTorch vs PyG vs DGL. | `python bench/bench_gcn.py train --dataset cora` then `python bench/bench_gcn.py bench --dataset cora` |
| `bench_sparse_autoencoder.py` | Weight-sparse autoencoder inference across frameworks; magnitude-pruned dense→sparse weights. | `python bench/bench_sparse_autoencoder.py` |
| `bench_sparse_transformer.py` | Longformer-style sparse attention on IMDB; sparse vs dense $O(n^2)$ scaling. | `python bench/bench_sparse_transformer.py train` then `python bench/bench_sparse_transformer.py bench` |
| `bench_graph_analytics.py` | SpMSpM graph analytics — triangle counting + multi-hop link prediction. | `python bench/bench_graph_analytics.py` (`--plot-only` re-plots from CSV) |

Each end-to-end harness has a companion `README_*.md` in `bench/`
(`README_GCN.md`, `README_sparse_autoencoder.md`,
`README_sparse_transformer.md`, `README_graph_analytics.md`) documenting its
datasets, framework dependencies, and any per-framework setup gotchas.

### Core sparse kernels

| Script | What it measures | How to run |
|--------|------------------|------------|
| `bench_spmm.py` | SpMM `A_sparse @ B_dense` over a curated 21-matrix "quick" set or full SuiteSparse. | `python bench/bench_spmm.py` (`--continue` skips already-recorded rows) |
| `bench_spmv.py` | SpMV `A_sparse @ v_dense`. | `python bench/bench_spmv.py` |
| `bench_spmspm.py` | SpMSpM `A_sparse @ A_sparseᵀ` (CSR, OpenMP-parallel). | `python bench/bench_spmspm.py` |
| `bench_sddmm.py` | SDDMM `S ⊙ (A_dense @ B_dense)`. | `python bench/bench_sddmm.py` |
| `bench_spmm_variants.py` | All SpMM kernel variants (tiled / untiled / direct) + PyTorch MKL across SuiteSparse; each matrix in a subprocess for crash isolation. | `python bench/bench_spmm_variants.py` |

### Tiling & autotune studies

These are the R&D harnesses behind the tiling selector documented in
{doc}`/user_guide/autotuning`. They are not needed to *use* Scorch — they exist
to justify (and re-verify) that the selector is no-regression.

| Script | What it measures | How to run |
|--------|------------------|------------|
| `bench_spmm_tiling.py` | Loop-tiling study: none / tile-i / tile-k / tile-ik for CSR×dense SpMM. | `python bench/bench_spmm_tiling.py` |
| `bench_tiling_autotuner.py` | The adaptive selector over the full space (none / tile-i / tile-j / tile-ik / tile-ijk); defines the byte cost model the analytic level mirrors. | `python bench/bench_tiling_autotuner.py` |
| `bench_tiling_noregress.py` | No-regression grid: times `scorch.matmul` with the selector ON vs OFF in-process (so thermal drift cancels), reports ratios. | `python bench/bench_tiling_noregress.py` |
| `bench_tilej_workloads.py` | Whether tile-j (contraction-axis cache blocking) helps the *real* GCN/AE workloads vs a locality-destroyed permuted matrix. | `python bench/bench_tilej_workloads.py` |
| `bench_tilej_vs_v2.py`, `bench_tilej_vs_ik.py`, `bench_tileijk_vs_tilej.py` | Head-to-head comparisons for the selector's candidate kernel pairs. | `python bench/bench_tilej_vs_v2.py` |
| `bench_tilewidth_sweep.py`, `bench_tilej_prodkernel.py` | Panel-width (`Jc`) sweeps and product-kernel width studies. | `python bench/bench_tilewidth_sweep.py` |
| `bench_ijk_relayout.py`, `bench_widetile.py` | tile-ijk B width-panel relayout at wide free dimension. | `python bench/bench_ijk_relayout.py` |
| `bench_spmm_colblock.py`, `bench_spmm_permute.py` | Column-blocking and row-permutation locality studies. | `python bench/bench_spmm_colblock.py` |
| `tiling_selector.py`, `tiling_selector_validate.py`, `tilej_wavefront.py` | Offline selector derivation and probe-vs-oracle validation. | `python bench/tiling_selector_validate.py` |

### Learned autotune level pipeline

The `learned` autotune level (experimental; falls back to `analytic` unless a
per-machine model is installed) is trained by this offline pipeline:

| Script | Role | How to run |
|--------|------|------------|
| `collect_autotune_data.py` | Emits a training CSV — one row per `(matrix, N, candidate)` with cheap features → measured median time / GFLOP/s. | `python bench/collect_autotune_data.py` |
| `train_autotune_model.py` | Trains the gradient-boosted-tree cost model into a dependency-free tree walker; imports the feature builder from `scorch.tiling` so train and serve are byte-identical. | `python bench/train_autotune_model.py` |
| `verify_learned.py` | Verifies the learned model against the analytic and oracle picks. | `python bench/verify_learned.py` |

### Infrastructure

| Script | Role |
|--------|------|
| `bench_omp_vs_tbb.py`, `bench_scheduling.py` | Threading-backend and scheduler studies. |
| `plot_spmm_speedup.py` | Plotting from result CSVs. |
| `_utils.py`, `_run_sddmm_big.py` | Shared harness helpers (not run directly). |

## Result files are run artifacts

Benchmark output CSVs land in `bench/bench_results/`. **They are run artifacts
and are never committed** — the pattern `*.csv` is gitignored. Rerun the
relevant harness to regenerate them for your machine; do not treat a CSV
checked out from someone else's run as authoritative.

:::{warning}
Numbers are hardware-specific. A result measured on one CPU should not be
quoted for another — cache size, core topology (e.g. hybrid P/E cores), and
thread count all move the crossover between the sparse and dense paths. Always
measure across a shape/format/density grid on the target machine rather than a
single shape, as the harnesses do.
:::

## See also

- {doc}`/user_guide/autotuning` — the `off` / `analytic` / `balanced` / `max` /
  `learned` autotune ladder and when tiling engages.
- {doc}`/performance/tuning_guide` — how to reason about whether a given SpMM
  shape benefits from tiling, and the caching model.
- {doc}`/development/building` — building the native extension and the JIT
  kernel cache that the benchmarks warm up.
