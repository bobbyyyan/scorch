#!/usr/bin/env python3
"""
Benchmark SpMM variants on real SuiteSparse matrices.

Downloads a curated sample of matrices spanning different sizes and densities,
runs SpMM (A_sparse @ B_dense) for each variant, checks correctness, and
produces a bar chart + summary table.

Usage:
  conda run -n scorch python tools/benchmark_suitesparse.py
  conda run -n scorch python tools/benchmark_suitesparse.py --k 512 --repeats 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.sparse
import ssgetpy
import torch

import scorch
import scorch_ops as ops
from scorch import STensor

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

CACHE_DIR = Path.home() / ".cache" / "scorch_suitesparse"

# Curated matrix selection: (ssid, name, description)
MATRIX_SELECTION = [
    (361, "msc10848",  "FE (n=10.8K, nnz=1.2M)"),
    (351, "crystk02",  "crystal (n=14.0K, nnz=969K)"),
    (352, "crystk03",  "crystal (n=24.7K, nnz=1.8M)"),
    (52,  "bcsstk30",  "structural (n=28.9K, nnz=2.0M)"),
    (356, "ct20stif",  "stiffness (n=52.3K, nnz=2.6M)"),
    (537, "gupta2",    "LP (n=62.1K, nnz=4.2M)"),
]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    fn: Callable
    kwargs: Dict[str, int]


VARIANTS = [
    VariantSpec("compiler_generated", None, {}),  # special-cased
    VariantSpec("direct", ops.spmm_csr_float_direct, {}),
    VariantSpec("neon", ops.spmm_csr_float_neon, {}),
    VariantSpec("neon2", ops.spmm_csr_float_neon2, {}),
    VariantSpec("neon4", ops.spmm_csr_float_neon4, {}),
    VariantSpec("baseline (tiled)", ops.spmm_csr_float, {}),
    VariantSpec("torch.sparse.mm", None, {}),  # special-cased
]


def _download_matrix(ssid: int, name: str) -> scipy.sparse.csr_matrix:
    """Download a SuiteSparse matrix, cache locally, return as scipy CSR."""
    cache_path = CACHE_DIR / f"{name}.mtx"
    if cache_path.exists():
        mat = scipy.io.mmread(str(cache_path))
        return scipy.sparse.csr_matrix(mat, dtype=np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = ssgetpy.search(name_or_id=ssid)
    if not results:
        raise RuntimeError(f"SuiteSparse matrix id={ssid} ({name}) not found")
    results[0].download(destpath=str(CACHE_DIR), extract=True)
    # ssgetpy extracts to CACHE_DIR/<name>/<name>.mtx (or .tar.gz)
    extracted = CACHE_DIR / name / f"{name}.mtx"
    if extracted.exists():
        mat = scipy.io.mmread(str(extracted))
        # Copy to cache root for simpler future access
        import shutil
        shutil.copy2(str(extracted), str(cache_path))
    elif cache_path.exists():
        mat = scipy.io.mmread(str(cache_path))
    else:
        raise RuntimeError(f"Could not find downloaded matrix for {name}")
    return scipy.sparse.csr_matrix(mat, dtype=np.float32)


def _to_scorch_args(
    csr: scipy.sparse.csr_matrix, b_dense: torch.Tensor,
) -> Tuple[STensor, STensor, List]:
    """Convert scipy CSR + dense B to scorch kernel args."""
    n_rows, n_cols = csr.shape
    k = b_dense.shape[1]

    # Build scorch STensor via PyTorch CSR tensor
    indptr = torch.from_numpy(csr.indptr.astype(np.int32))
    indices = torch.from_numpy(csr.indices.astype(np.int32))
    values = torch.from_numpy(csr.data.astype(np.float32))
    a_torch_csr = torch.sparse_csr_tensor(indptr, indices, values,
                                           size=(n_rows, n_cols))

    a_scorch = STensor.from_csr(a_torch_csr, "A")
    b_scorch = STensor.from_torch(b_dense, "B")

    result_shape = [n_rows, k]
    args = [result_shape]
    for tensor in (a_scorch, b_scorch):
        args.append(list(tensor.shape))
        args.append(tensor.index.mode_indices)
        args.append(tensor.values)
    return a_scorch, b_scorch, args


def _run_prebuilt(
    variant: VariantSpec, args: List, warmup: int, repeats: int,
) -> Tuple[torch.Tensor, List[float]]:
    for _ in range(warmup):
        variant.fn(*args, **variant.kwargs)
    timings = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = variant.fn(*args, **variant.kwargs)
        timings.append(time.perf_counter() - t0)
    return result.storage.value, timings


def _run_compiler(
    a_scorch: STensor, b_scorch: STensor, n_rows: int, k: int,
    warmup: int, repeats: int,
) -> Tuple[torch.Tensor, List[float]]:
    for _ in range(warmup):
        scorch.matmul(a_scorch, b_scorch, format="dd", use_cache=False,
                      output_mode_order=[0, 1])
    timings = []
    result = None
    for _ in range(repeats):
        td: Dict[str, float] = {}
        result = scorch.matmul(
            a_scorch, b_scorch, format="dd", use_cache=False,
            output_mode_order=[0, 1], time_dict=td,
        )
        timings.append(td["eval_time"])
    assert isinstance(result, torch.Tensor)
    return result, timings


def _run_torch(
    csr: scipy.sparse.csr_matrix, b_dense: torch.Tensor,
    warmup: int, repeats: int,
) -> Tuple[torch.Tensor, List[float]]:
    indptr = torch.from_numpy(csr.indptr.astype(np.int32))
    indices = torch.from_numpy(csr.indices.astype(np.int32))
    values = torch.from_numpy(csr.data.astype(np.float32))
    a_torch = torch.sparse_csr_tensor(indptr, indices, values,
                                       size=(csr.shape[0], csr.shape[1]))
    for _ in range(warmup):
        torch.sparse.mm(a_torch, b_dense)
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = torch.sparse.mm(a_torch, b_dense)
        timings.append(time.perf_counter() - t0)
    return result, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="SuiteSparse SpMM benchmark")
    parser.add_argument("--k", type=int, default=8192,
                        help="Number of columns in dense matrix B")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=str, default="suitesparse_spmm.png",
                        help="Output plot filename")
    args = parser.parse_args()

    print(f"SuiteSparse SpMM Benchmark  (B columns k={args.k}, "
          f"warmup={args.warmup}, repeats={args.repeats})")
    print("=" * 100)

    # Collect results for plotting
    matrix_names: List[str] = []
    all_results: Dict[str, List[float]] = {v.name: [] for v in VARIANTS}

    for ssid, name, desc in MATRIX_SELECTION:
        print(f"\nDownloading {name} (id={ssid})... ", end="", flush=True)
        csr = _download_matrix(ssid, name)
        n_rows, n_cols = csr.shape
        nnz = csr.nnz
        k = args.k
        gflops = 2.0 * nnz * k / 1e9
        print(f"OK  ({n_rows}x{n_cols}, nnz={nnz:,}, k={k})")
        print(f"  {desc}  |  GFLOP={gflops:.3f}")

        # Build dense B
        gen = torch.Generator().manual_seed(42)
        b_dense = torch.rand((n_cols, k), generator=gen, dtype=torch.float32)

        # Reference
        a_scorch, b_scorch, kernel_args = _to_scorch_args(csr, b_dense)

        # Run torch reference first for correctness check
        torch_ref, torch_times = _run_torch(csr, b_dense, args.warmup, args.repeats)

        label = f"{name}\n({n_rows}, nnz={nnz//1000}K)"
        matrix_names.append(label)

        variant_medians: Dict[str, float] = {}

        for variant in VARIANTS:
            if variant.name == "compiler_generated":
                output, timings = _run_compiler(
                    a_scorch, b_scorch, n_rows, k, args.warmup, args.repeats
                )
            elif variant.name == "torch.sparse.mm":
                output, timings = torch_ref, torch_times
            else:
                output_raw, timings = _run_prebuilt(
                    variant, kernel_args, args.warmup, args.repeats
                )
                output = output_raw.view(n_rows, k)

            # Correctness check
            if not torch.allclose(output, torch_ref, atol=1e-2, rtol=1e-2):
                max_diff = (output - torch_ref).abs().max().item()
                print(f"  WARNING: {variant.name} max_diff={max_diff:.4f}")

            med = statistics.median(timings) * 1e3
            gf = gflops / (statistics.median(timings)) if statistics.median(timings) > 0 else 0
            variant_medians[variant.name] = med
            all_results[variant.name].append(med)

        # Print table for this matrix
        best_med = min(variant_medians.values())
        print(f"  {'variant':<25s} {'median_ms':>10s} {'GFLOP/s':>10s} {'speedup':>8s}")
        print(f"  {'-'*58}")
        for variant in VARIANTS:
            med = variant_medians[variant.name]
            gf = gflops / (med / 1e3) if med > 0 else 0
            su = best_med / med if med > 0 else 0
            marker = " <-- best" if abs(med - best_med) < 0.01 else ""
            print(f"  {variant.name:<25s} {med:10.3f} {gf:10.2f} {su:7.2f}x{marker}")

    # ── Plot ────────────────────────────────────────────────────────────
    n_matrices = len(matrix_names)
    n_variants = len(VARIANTS)
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(n_matrices)
    width = 0.8 / n_variants
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#607D8B"]

    for idx, variant in enumerate(VARIANTS):
        # Normalize to baseline (tiled) for each matrix
        baseline_times = all_results["baseline (tiled)"]
        speedups = [
            baseline_times[m] / all_results[variant.name][m]
            if all_results[variant.name][m] > 0 else 0
            for m in range(n_matrices)
        ]
        offset = (idx - n_variants / 2 + 0.5) * width
        bars = ax.bar(x + offset, speedups, width, label=variant.name,
                      color=colors[idx % len(colors)], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Speedup vs baseline (tiled)", fontsize=12)
    ax.set_title("SpMM Performance on SuiteSparse Matrices\n"
                 f"(A_sparse × B_dense, k={args.k})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_names, fontsize=9)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    output_path = Path(args.output)
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
