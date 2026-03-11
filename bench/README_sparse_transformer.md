# Sparse Attention Transformer Inference Benchmark

Benchmarks Longformer-style sparse attention transformer inference across Scorch, Sparse PyTorch, and Dense PyTorch. Demonstrates how sparse attention (sliding window + global tokens) scales better than dense O(n^2) attention at increasing sequence lengths.

## Motivation

The GCN benchmark covers **input/data sparsity** and the autoencoder benchmark covers **weight sparsity**. This benchmark covers **architecture/activation sparsity** — where the model structure itself defines a sparse computation pattern. Longformer-style sparse attention (Beltagy et al. 2020) restricts which token pairs interact via a sliding window and global tokens, creating an attention mask that is ~97% sparse at seq_len=4096.

## Setup

Core dependencies (torch, scorch) are already in the `scorch` conda env. The benchmark also requires torchtext:

```bash
conda run -n scorch pip install torchtext
```

IMDB is downloaded automatically on first use.

## Usage

All commands are run from the repo root.

### Train

Train the sparse transformer on IMDB sentiment classification (required before benchmarking):

```bash
conda run -n scorch python bench/bench_sparse_transformer.py train
```

Weights are saved to `weights/sparse_transformer_imdb.pt`.

### Benchmark

```bash
# All sequence lengths, all frameworks (defaults: warmup=5, repeats=20)
conda run -n scorch python bench/bench_sparse_transformer.py bench

# Specific sequence lengths
conda run -n scorch python bench/bench_sparse_transformer.py bench --seq-lengths 512 1024 2048 4096

# Subset of frameworks
conda run -n scorch python bench/bench_sparse_transformer.py bench --frameworks scorch sparse-pytorch

# Custom timing parameters
conda run -n scorch python bench/bench_sparse_transformer.py bench --warmup 10 --repeats 50

# Regenerate plot from existing CSV
conda run -n scorch python bench/bench_sparse_transformer.py bench --plot-only
```

## Model Architecture

Longformer-style sparse attention transformer for binary classification (IMDB):

| Parameter | Value |
|-----------|-------|
| Vocab size | 30,000 |
| Embedding dim | 256 |
| Attention heads | 8 (head_dim=32) |
| Layers | 4 |
| FFN dim | 1,024 |
| Window size | 64 (each side) |
| Global tokens | 1 ([CLS]) |
| Positional encoding | Sinusoidal (up to 8,192) |
| Total params | ~10.8M |

Architecture: pre-norm transformer blocks (LayerNorm before attention/FFN), GELU activation in FFN, residual connections. Classification via [CLS] token pooling.

Training uses dense attention with additive -inf masking (standard approach for backprop). Inference uses framework-specific sparse paths.

## Sparsity Scaling

The Longformer mask combines a sliding window (64 tokens each side = 129 entries per row) with global [CLS] token attention:

| Sequence Length | NNZ | Sparsity |
|----------------:|--------:|---------:|
| 512 | 62,782 | 76.1% |
| 1,024 | 127,038 | 87.9% |
| 2,048 | 257,598 | 93.9% |
| 4,096 | 532,286 | 96.8% |

As sequence length grows, sparsity increases — this is where sparse attention shows its scaling advantage over dense O(n^2) attention.

## Frameworks Compared

All three paths share identical model weights. They differ **only** in how the two sparse attention operations (SDDMM and SpMM) are computed per head. Everything else (embedding, projections, LayerNorm, FFN, softmax) uses identical PyTorch code.

| Framework | SDDMM (scores) | SpMM (attn × V) |
|-----------|----------------|-----------------|
| **Dense PyTorch** | `Q @ K.T` — full (S, S) dense matmul | `attn @ V` — dense matmul |
| **Sparse PyTorch** | Gather: `(Q[rows] * K[cols]).sum(-1)` | `torch.sparse.mm(attn_csr, V)` |
| **Scorch** | `scorch.einsum("ij,ik,jk->ij", mask, Q, K)` | `scorch.matmul(attn_st, V, format="dd")` |

Sparse softmax uses `torch.sparse.softmax` for both Sparse PyTorch and Scorch paths.

## Output

Results are printed per sequence length:

```
  ============================================================
  seq_len=512  sparsity=76.05%  nnz=62,782
  ============================================================
  Framework           | Median (ms) |  Min (ms) |  Std (ms)
  --------------------------------------------------------
  Dense PyTorch       |      12.345 |    11.234 |     0.567
  Sparse PyTorch      |      15.678 |    14.567 |     0.890
  Scorch              |      13.456 |    12.345 |     0.678
```

Correctness is verified by comparing all framework outputs against Dense PyTorch with `torch.allclose(atol=1e-2, rtol=1e-2)`.

CSV results are written to `bench_results/sparse_transformer_results.csv` and a line plot to `bench_results/sparse_transformer.png`.

## Expected Behavior

- **Dense PyTorch**: Runtime grows quadratically with sequence length (O(n^2) attention).
- **Sparse PyTorch / Scorch**: Runtime grows linearly with sequence length (O(nnz) operations). Increasingly faster than Dense PyTorch at longer sequences.
- At short sequences (512), overhead from sparse indexing may make sparse paths slower than dense. The advantage emerges at longer sequences where sparsity is high.
