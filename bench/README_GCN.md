# GCN Inference Benchmark

Benchmarks 2-layer GCN inference across Scorch, PyTorch, PyG, and DGL using standard Kipf & Welling architecture with symmetric normalization.

## Setup

Core dependencies (torch, scorch) are already in the `scorch` conda env. Install graph ML extras:

```bash
conda run -n scorch pip install -r bench/requirements-gcn.txt
```

PyG and DGL are optional — the benchmark skips unavailable frameworks with a warning.

> **Note:** DGL 2.2 does not yet support PyTorch 2.8. DGL benchmarks will be skipped until a compatible release is available.

## Usage

All commands are run from the repo root.

### Train

Train a GCN and save weights (required before benchmarking):

```bash
# Single dataset
conda run -n scorch python bench/bench_gcn.py train --dataset cora

# All datasets
conda run -n scorch python bench/bench_gcn.py train --dataset all
```

Weights are saved to `weights/gcn_{dataset}.pt`.

### Benchmark

```bash
# Single dataset (defaults: warmup=5, repeats=20)
conda run -n scorch python bench/bench_gcn.py bench --dataset cora

# All datasets
conda run -n scorch python bench/bench_gcn.py bench --dataset all

# Custom timing parameters
conda run -n scorch python bench/bench_gcn.py bench --dataset cora --warmup 10 --repeats 50

# Subset of frameworks
conda run -n scorch python bench/bench_gcn.py bench --dataset cora --frameworks pytorch scorch
```

## Datasets

| Dataset | Nodes | Edges | Features | Classes | Hidden | Source |
|---------|------:|------:|---------:|--------:|-------:|--------|
| cora | 2.7K | 10K | 1433 | 7 | 16 | Planetoid |
| citeseer | 3.3K | 9K | 3703 | 6 | 16 | Planetoid |
| pubmed | 19K | 88K | 500 | 3 | 16 | Planetoid |
| ogbn-arxiv | 169K | 2.3M | 128 | 40 | 256 | OGB |
| reddit | 233K | 114M | 602 | 41 | 256 | PyG |
| ogbn-products | 2.4M | 62M | 100 | 47 | 256 | OGB |

Reddit and ogbn-products are large and may OOM — they skip gracefully if memory is insufficient.

## Output

Results are printed as a table:

```
========================================================================
Dataset: cora (2,708 nodes, 10,556 edges, 1433 features, 7 classes)
========================================================================
Framework     | Median (ms) |  Min (ms) |  Std (ms) | Accuracy
--------------------------------------------------------------
PyTorch       |       5.335 |     2.931 |     2.400 |   0.8140
Scorch        |       5.236 |     2.342 |     4.298 |   0.8140
PyG           |       5.279 |     3.330 |     5.188 |   0.8140
```

Correctness is verified by comparing logits against PyTorch with `torch.allclose(atol=1e-4)`.
