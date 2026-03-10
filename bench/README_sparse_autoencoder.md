# Weight-Sparse Autoencoder Inference Benchmark

Benchmarks inference through a pruned autoencoder across Scorch, PyTorch Sparse, and PyTorch Dense. Demonstrates Scorch's SpMM performance advantage when weight matrices are sparse (sparse W × dense activation).

## Motivation

The GCN benchmark covers **input/data sparsity** (sparse adjacency matrices). This benchmark covers **weight sparsity** — train a dense autoencoder, apply magnitude pruning to create sparse weight matrices, then measure how each framework exploits that sparsity during inference.

## Setup

Core dependencies (torch, scorch) are already in the `scorch` conda env. The benchmark also requires torchvision:

```bash
conda run -n scorch pip install torchvision
```

MNIST and CIFAR-10 are downloaded automatically on first use.

## Usage

All commands are run from the repo root.

### Train

Train a dense autoencoder and save weights (required before benchmarking):

```bash
# Single model config
conda run -n scorch python bench/bench_sparse_autoencoder.py train --model small

# All model configs
conda run -n scorch python bench/bench_sparse_autoencoder.py train --model all
```

Weights are saved to `weights/autoencoder_{model}.pt`.

### Benchmark

```bash
# Single model, all sparsity levels (defaults: warmup=5, repeats=20)
conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model small

# All models
conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model all

# Specific sparsity levels
conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model small --sparsity 0.9 0.95 0.99

# Subset of frameworks
conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model small --frameworks scorch pytorch-dense

# Custom timing parameters
conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model small --warmup 10 --repeats 50

# Regenerate plot from existing CSV
conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model small --plot-only
```

## Model Configs

4-layer symmetric autoencoder (2 encoder + 2 decoder) with ReLU activations and sigmoid output:

| Config | Hidden dims | Largest weight matrix | Dataset | Epochs |
|--------|------------:|----------------------:|---------|-------:|
| small  | [1024, 512] | 1024×784              | MNIST   | 20     |
| medium | [2048, 1024]| 2048×1024             | MNIST   | 20     |
| large  | [4096, 2048]| 4096×2048             | MNIST   | 20     |
| xlarge | [4096, 2048]| 4096×3072             | CIFAR-10| 30     |

The hidden dims are chosen to produce large weight matrices where SpMM performance matters. MNIST/CIFAR-10 keep training fast.

## Sparsity Levels

Default: `[0.5, 0.7, 0.8, 0.9, 0.95, 0.99]`

Pruning uses global unstructured magnitude pruning (Han et al. 2015): compute a threshold at the target percentile across all weight magnitudes, zero out everything below.

## Frameworks Compared

| Framework | Method | Exploits sparsity? |
|-----------|--------|-------------------|
| **PyTorch Dense** | `F.linear(x, W_dense, bias)` | No — baseline |
| **PyTorch Sparse** | `torch.sparse.mm(W_csr, x.T)` | Yes — PyTorch native sparse ops |
| **Scorch** | `scorch.matmul(W_stensor, x.T, format="dd")` | Yes — Scorch optimized SpMM kernel |

Weight conversion (dense → CSR/STensor) happens outside the timing loop.

## Output

Results are printed per sparsity level:

```
    ============================================================
    small @ 95% sparsity
    ============================================================
    Framework           | Median (ms) |  Min (ms) |  Std (ms)
    --------------------------------------------------------
    PyTorch Dense       |       2.546 |     2.313 |     0.174
    PyTorch Sparse      |      13.194 |    12.138 |     0.546
    Scorch              |       2.137 |     1.923 |     0.204
```

Correctness is verified by comparing all framework outputs against PyTorch Dense with `torch.allclose(atol=1e-2, rtol=1e-2)`.

CSV results are written to `bench_results/sparse_autoencoder_results.csv` and a line plot to `bench_results/sparse_autoencoder.png`.

## Expected Behavior

- **PyTorch Dense**: Runtime roughly constant across sparsity levels (does not exploit sparsity).
- **Scorch**: Gets faster at higher sparsity. Beats PyTorch Dense at high sparsity (~95%+).
- **PyTorch Sparse**: Also scales with sparsity but has higher overhead than Scorch, especially at moderate sparsity.
