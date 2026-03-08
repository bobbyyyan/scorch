#!/usr/bin/env python3
"""
Benchmark handwritten SpMM variants against compiler-generated kernels.

Usage:
  conda run -n scorch python tools/benchmark_spmm_variants.py
  conda run -n scorch python tools/benchmark_spmm_variants.py --n 4096 --sparsity 0.95 --repeats 10
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

import scorch
import scorch_ops as ops
from scorch import STensor

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    fn: Callable
    kwargs: Dict[str, int]


def _build_inputs(n: int, sparsity: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    a_dense = torch.rand((n, n), generator=generator, dtype=torch.float32)
    b_dense = torch.rand((n, n), generator=generator, dtype=torch.float32)
    mask = torch.rand((n, n), generator=generator) > sparsity
    a_dense = a_dense * mask
    return a_dense, b_dense


def _to_scorch_inputs(a_dense: torch.Tensor, b_dense: torch.Tensor) -> Tuple[STensor, STensor, List]:
    a_scorch = STensor.from_torch(a_dense, "A").to_sparse("ds")
    b_scorch = STensor.from_torch(b_dense, "B")
    result_shape = [a_dense.shape[0], b_dense.shape[1]]
    args = [result_shape]
    for tensor in (a_scorch, b_scorch):
        args.append(list(tensor.shape))
        args.append(tensor.index.mode_indices)
        args.append(tensor.values)
    return a_scorch, b_scorch, args


def _run_variant(
    variant: VariantSpec,
    args: List,
    warmup: int,
    repeats: int,
) -> Tuple[torch.Tensor, List[float]]:
    for _ in range(warmup):
        _ = variant.fn(*args, **variant.kwargs)
    timings: List[float] = []
    last_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = variant.fn(*args, **variant.kwargs)
        timings.append(time.perf_counter() - start)
    assert last_result is not None
    return last_result.storage.value, timings


def _run_compiler_path(
    a_scorch: STensor,
    b_scorch: STensor,
    warmup: int,
    repeats: int,
) -> Tuple[torch.Tensor, List[float]]:
    timings: List[float] = []
    result = None
    for _ in range(warmup):
        result = scorch.matmul(
            a_scorch, b_scorch, format="dd", use_cache=False, output_mode_order=[0, 1]
        )
    for _ in range(repeats):
        time_dict: Dict[str, float] = {}
        result = scorch.matmul(
            a_scorch,
            b_scorch,
            format="dd",
            use_cache=False,
            output_mode_order=[0, 1],
            time_dict=time_dict,
        )
        timings.append(time_dict["eval_time"])
    assert isinstance(result, torch.Tensor)
    return result, timings


def _summarize(times: List[float], gflops_per_op: float) -> Dict[str, float]:
    min_s = min(times)
    median_s = statistics.median(times)
    mean_s = statistics.mean(times)
    return {
        "min_ms": min_s * 1e3,
        "median_ms": median_s * 1e3,
        "mean_ms": mean_s * 1e3,
        "gflops_peak": gflops_per_op / min_s if min_s > 0 else 0.0,
        "gflops_median": gflops_per_op / median_s if median_s > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SpMM kernel variants")
    parser.add_argument("--n", type=int, default=4096, help="Square matrix dimension")
    parser.add_argument("--sparsity", type=float, default=0.95, help="Input A sparsity")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=10, help="Measured iterations")
    args = parser.parse_args()

    a_dense, b_dense = _build_inputs(args.n, args.sparsity, args.seed)
    torch_ref = torch.sparse.mm(a_dense.to_sparse_csr(), b_dense)
    a_scorch, b_scorch, kernel_args = _to_scorch_inputs(a_dense, b_dense)

    # Compute GFLOP count: 2 * nnz(A) * n_cols(B) (multiply + add per nnz)
    nnz_a = int(a_dense.to_sparse_csr().values().numel())
    gflops = 2.0 * nnz_a * args.n / 1e9

    variants = [
        # Existing variants
        VariantSpec("spmm_csr_float_untiled", ops.spmm_csr_float_untiled, {}),
        VariantSpec("spmm_csr_float", ops.spmm_csr_float, {}),
        VariantSpec("spmm_csr_float_optimized", ops.spmm_csr_float_optimized, {}),
        VariantSpec("spmm_csr_float_turbo", ops.spmm_csr_float_turbo, {}),
        VariantSpec("spmm_csr_float_ultra", ops.spmm_csr_float_ultra, {}),
        VariantSpec("spmm_csr_float_apex", ops.spmm_csr_float_apex, {}),
        VariantSpec("spmm_csr_float_tiled_i_k", ops.spmm_csr_float_tiled_i_k, {}),
        # Novel variants
        VariantSpec("spmm_csr_float_direct", ops.spmm_csr_float_direct, {}),
        VariantSpec("spmm_csr_float_neon", ops.spmm_csr_float_neon, {}),
        VariantSpec("spmm_csr_float_row_panel", ops.spmm_csr_float_row_panel, {}),
        VariantSpec("spmm_csr_float_k_parallel", ops.spmm_csr_float_k_parallel, {}),
        VariantSpec("spmm_csr_float_sorted_rows", ops.spmm_csr_float_sorted_rows, {}),
    ]

    results: List[Tuple[str, Dict[str, float]]] = []

    for variant in variants:
        output, timings = _run_variant(variant, kernel_args, args.warmup, args.repeats)
        output_2d = output.view(args.n, args.n)
        if not torch.allclose(output_2d, torch_ref, atol=1e-3, rtol=1e-3):
            max_diff = (output_2d - torch_ref).abs().max().item()
            print(f"WARNING: {variant.name} max diff = {max_diff:.6f}")
            raise AssertionError(f"{variant.name} produced incorrect output")
        results.append((variant.name, _summarize(timings, gflops)))

    compiler_output, compiler_timings = _run_compiler_path(
        a_scorch, b_scorch, args.warmup, args.repeats
    )
    if not torch.allclose(compiler_output, torch_ref, atol=1e-3, rtol=1e-3):
        raise AssertionError("compiler_generated produced incorrect output")
    results.append(("compiler_generated", _summarize(compiler_timings, gflops)))

    torch_times = []
    for _ in range(args.warmup):
        _ = torch.sparse.mm(a_dense.to_sparse_csr(), b_dense)
    for _ in range(args.repeats):
        start = time.perf_counter()
        _ = torch.sparse.mm(a_dense.to_sparse_csr(), b_dense)
        torch_times.append(time.perf_counter() - start)
    results.append(("torch_sparse_mm", _summarize(torch_times, gflops)))

    results.sort(key=lambda item: item[1]["median_ms"])

    # Find best existing variant for speedup calculation
    existing_names = {
        "spmm_csr_float_untiled", "spmm_csr_float", "spmm_csr_float_optimized",
        "spmm_csr_float_turbo", "spmm_csr_float_ultra", "spmm_csr_float_apex",
        "spmm_csr_float_tiled_i_k",
    }
    best_existing_median = min(
        (s["median_ms"] for name, s in results if name in existing_names),
        default=float("inf"),
    )

    print(
        f"\nSpMM benchmark n={args.n} sparsity={args.sparsity} nnz(A)={nnz_a:,} "
        f"GFLOP={gflops:.3f}"
    )
    print(f"warmup={args.warmup}, repeats={args.repeats}")
    print("-" * 110)
    print(
        f"{'variant':35s} {'min_ms':>9s} {'median_ms':>9s} {'mean_ms':>9s} "
        f"{'GFLOP/s':>9s} {'vs best':>9s}"
    )
    print("-" * 110)
    for name, summary in results:
        speedup = best_existing_median / summary["median_ms"] if summary["median_ms"] > 0 else 0.0
        marker = " *" if name not in existing_names and name not in ("compiler_generated", "torch_sparse_mm") else ""
        print(
            f"{name + marker:35s} {summary['min_ms']:9.3f} {summary['median_ms']:9.3f} "
            f"{summary['mean_ms']:9.3f} {summary['gflops_median']:9.2f} "
            f"{speedup:8.2f}x"
        )


if __name__ == "__main__":
    main()
