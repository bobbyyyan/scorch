#!/usr/bin/env python3
"""
Benchmark handwritten SpMM variants against compiler-generated kernels.

Usage:
  conda run -n scorch python tools/benchmark_spmm_variants.py
  conda run -n scorch python tools/benchmark_spmm_variants.py --n 3072 --sparsity 0.97
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


def _summarize(times: List[float]) -> Dict[str, float]:
    return {
        "min_ms": min(times) * 1e3,
        "median_ms": statistics.median(times) * 1e3,
        "mean_ms": statistics.mean(times) * 1e3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SpMM kernel variants")
    parser.add_argument("--n", type=int, default=2048, help="Square matrix dimension")
    parser.add_argument("--sparsity", type=float, default=0.95, help="Input A sparsity")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=5, help="Measured iterations")
    args = parser.parse_args()

    a_dense, b_dense = _build_inputs(args.n, args.sparsity, args.seed)
    torch_ref = torch.sparse.mm(a_dense.to_sparse_csr(), b_dense)
    a_scorch, b_scorch, kernel_args = _to_scorch_inputs(a_dense, b_dense)

    variants = [
        VariantSpec("spmm_csr_float_untiled", ops.spmm_csr_float_untiled, {}),
        VariantSpec("spmm_csr_float", ops.spmm_csr_float, {}),
        VariantSpec("spmm_csr_float_optimized", ops.spmm_csr_float_optimized, {}),
        VariantSpec("spmm_csr_float_turbo", ops.spmm_csr_float_turbo, {}),
        VariantSpec("spmm_csr_float_ultra", ops.spmm_csr_float_ultra, {}),
        VariantSpec("spmm_csr_float_apex", ops.spmm_csr_float_apex, {}),
        VariantSpec("spmm_csr_float_tiled_i_k", ops.spmm_csr_float_tiled_i_k, {}),
    ]

    results: List[Tuple[str, Dict[str, float]]] = []

    for variant in variants:
        output, timings = _run_variant(variant, kernel_args, args.warmup, args.repeats)
        output_2d = output.view(args.n, args.n)
        if not torch.allclose(output_2d, torch_ref, atol=1e-3, rtol=1e-3):
            raise AssertionError(f"{variant.name} produced incorrect output")
        results.append((variant.name, _summarize(timings)))

    compiler_output, compiler_timings = _run_compiler_path(
        a_scorch, b_scorch, args.warmup, args.repeats
    )
    if not torch.allclose(compiler_output, torch_ref, atol=1e-3, rtol=1e-3):
        raise AssertionError("compiler_generated produced incorrect output")
    results.append(("compiler_generated", _summarize(compiler_timings)))

    torch_times = []
    for _ in range(args.warmup):
        _ = torch.sparse.mm(a_dense.to_sparse_csr(), b_dense)
    for _ in range(args.repeats):
        start = time.perf_counter()
        _ = torch.sparse.mm(a_dense.to_sparse_csr(), b_dense)
        torch_times.append(time.perf_counter() - start)
    results.append(("torch_sparse_mm", _summarize(torch_times)))

    results.sort(key=lambda item: item[1]["median_ms"])

    print(
        f"SpMM benchmark n={args.n} sparsity={args.sparsity} "
        f"(warmup={args.warmup}, repeats={args.repeats})"
    )
    print("-" * 90)
    for name, summary in results:
        print(
            f"{name:30s} min={summary['min_ms']:9.3f} ms "
            f"median={summary['median_ms']:9.3f} ms mean={summary['mean_ms']:9.3f} ms"
        )


if __name__ == "__main__":
    main()
