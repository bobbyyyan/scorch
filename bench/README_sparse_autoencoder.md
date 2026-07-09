# Weight-Sparse Autoencoder Inference Benchmark

Benchmarks inference through a magnitude-pruned autoencoder across **Scorch**, **PyTorch Dense**, **PyTorch Sparse**, and **PyG (torch_sparse)**. Demonstrates Scorch's SpMM performance when weight matrices are sparse (sparse W × dense activation).

## Motivation

The GCN benchmark covers **input/data sparsity** (sparse adjacency matrices). This benchmark covers **weight sparsity** — train a dense autoencoder, apply magnitude pruning to create sparse weight matrices, then measure how each framework exploits that sparsity during inference. Each layer is `act(W @ x + b)` with `W` sparse, i.e. an SpMM (sparse weight × dense activation batch).

## Setup

Core deps (torch, scorch) are in the `scorch` conda env. Also required:

```bash
conda run -n scorch pip install "torchvision==0.20.1" --no-deps   # pinned to torch 2.5.1
# PyG framework uses torch_sparse (already present in the GCN bench env)
```

Datasets download automatically on first use.

## Usage

Run from the repo root.

### Train

Weights must exist before benchmarking (few epochs — pruned weights only need realistic magnitudes, not converged reconstruction):

```bash
conda run -n scorch python bench/bench_sparse_autoencoder.py train --model all      # all configs
conda run -n scorch python bench/bench_sparse_autoencoder.py train --model mnist     # one config
```

Weights are saved to `weights/autoencoder_{model}.pt`.

### Benchmark

```bash
conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model all
conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model svhn --sparsity 0.9 0.95 0.99
conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model mnist --frameworks scorch pytorch-dense
# include the fastest Scorch path (fused SpMM+bias+act, feature-major chain):
conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model all --frameworks pytorch-dense pytorch-sparse scorch-fused pyg
conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model all --plot-only   # replot from CSV
```

## Model Configs

4-layer symmetric autoencoder (2 encoder + 2 decoder), ReLU hidden + sigmoid output. The dataset sets the input dim, which drives the first/last layer weight shapes:

| Config     | Dataset       | Input dim | Hidden dims  | Largest weight |
|------------|---------------|----------:|-------------:|---------------:|
| mnist      | MNIST         | 784       | [1024, 512]  | 1024×784       |
| mnist_big  | MNIST         | 784       | [4096, 2048] | 4096×2048      |
| fashion    | Fashion-MNIST | 784       | [2048, 1024] | 2048×1024      |
| kmnist     | KMNIST        | 784       | [2048, 1024] | 2048×1024      |
| svhn       | SVHN          | 3072      | [2048, 1024] | 3072×2048      |
| svhn_big   | SVHN          | 3072      | [4096, 2048] | 4096×3072      |
| stl10      | STL-10        | 27648     | [4096, 2048] | 27648×4096     |

Input dims span 784 / 3072 / 27648 for shape diversity; STL-10 stresses the large first/last layers.

> **CIFAR-10/100 note:** originally planned for the 3072-dim tier, but the Toronto
> mirror (`cs.toronto.edu`) is globally throttled to ~25 kB/s (verified from both
> redwood and an off-campus link — CIFAR-10 alone would take ~1 hr). SVHN is the
> same 3072-dim tier and serves at ~117 MB/s (Stanford-hosted), so `svhn` +
> `svhn_big` recover the exact [2048,1024] and [4096,2048] shapes CIFAR would have
> provided. Re-add CIFAR by putting `cifar-{10,100}-python.tar.gz` in `data/` (any
> fast mirror with matching md5) and adding the configs back.

## Sparsity Levels

Default: `[0.5, 0.7, 0.8, 0.9, 0.95, 0.99]`. Global unstructured magnitude pruning (Han et al. 2015): threshold at the target percentile across all weight magnitudes, zero out below.

## Frameworks Compared

| Framework | Method | Exploits sparsity? |
|-----------|--------|-------------------|
| **PyTorch Dense** | `F.linear(x, W_dense, b)` | No — baseline |
| **PyTorch Sparse** | `torch.sparse.mm(W_csr, x.T)` | Yes — PyTorch native sparse |
| **Scorch** | `scorch.matmul(W_stensor, x.T)` | Yes — prebuilt SpMM fast-path |
| **PyG** | `torch_sparse.matmul(SparseTensor(W), x.T)` | Yes — PyG/torch_sparse SpMM |

Weight conversion (dense → CSR / STensor / SparseTensor) happens outside the timing loop.

### Frameworks evaluated but not in the default set

- **DGL** — dropped. `dgl.sparse`'s C++ backend is unavailable in this env (`libdgl_sparse_pytorch_*.so` missing), and DGL `GraphConv` is graph message-passing over an adjacency, not a weight-sparse linear layer — there is no honest mapping for `W @ x`.
- **Scorch (fused)** — the **fastest Scorch path**; opt in with `--frameworks ... scorch-fused`. Kept out of the default set only so the default table compares like-for-like unfused SpMM kernels. Fuses each layer's SpMM + **per-output-channel** bias + activation into ONE prebuilt parallel region (`spmm_csr_linear_fused_float`) and runs the forward **feature-major**: transpose the input once via the fast cache-blocked `scorch.fast_transpose`, keep every intermediate in `[feat, batch]` through `scorch.sparse_linear_fm`, and take a single lazy transpose out — so there is no per-layer transpose and no separate torch bias/act epilogue.

  This overturns an earlier finding that "fusion does not help." That attempt reused the GCN `spmm_csr_bias_relu`, whose bias is on the free/column dim (wrong for a `Linear`'s per-output-channel bias), and the homogeneous-coord workaround (`relu(SpMM([W|b], [x|1]))`) routed through the generic JIT-codegen SpMM at ~2.5–5× slower. A **dedicated per-output-channel fused kernel** plus the fast input transpose resolve both, and the two `x.T.contiguous()` transposes MKL's `torch.sparse.mm(W, x.T).T` also pays are cut by the cache-blocked transpose. Result on redwood x86: fastest of all four frameworks at every sparsity ≥ 0.7, and at 0.99 it beats PyTorch Sparse (MKL) on all 7 models (e.g. SVHN 0.41 vs 0.74 ms, STL-10 5.0 vs 32.2 ms) — fastest in 40/42 (model × sparsity) cells overall (PyTorch Dense wins only the two 50%-sparsity cases where dense GEMM still beats SpMM).

## Output

Per sparsity level, a table like:

```
    ==============================================================
    svhn @ 95% sparsity
    ==============================================================
    Framework           | Median (ms) |  Min (ms) |  Std (ms)
    ----------------------------------------------------------
    PyTorch Dense       |         ... |       ... |       ...
    PyTorch Sparse      |         ... |       ... |       ...
    Scorch              |         ... |       ... |       ...
    PyG                 |         ... |       ... |       ...
```

Correctness: every framework's output is checked against PyTorch Dense with `allclose(atol=1e-2, rtol=1e-2)`.

CSV → `bench_results/sparse_autoencoder_results.csv` (columns: Model, Dataset, InputDim, Hidden, Sparsity, Framework, Median_ms, Min_ms, Std_ms). Plot → `bench_results/sparse_autoencoder.png` (grid, one subplot per config). A Scorch-vs-others speedup summary prints at the end.

## Expected Behavior

- **PyTorch Dense**: roughly constant across sparsity (does not exploit it).
- **Scorch / PyTorch Sparse / PyG**: get faster at higher sparsity. Scorch's advantage widens with matrix size and sparsity; on tiny matrices (e.g. `mnist`) the frameworks are close and noisy.
