#!/usr/bin/env python3
"""
Benchmark the compiler-generated code to measure the impact of the
sparse re-traversal tiling optimization.

Tests:
  1. SpMM  (target of the optimization — should improve)
  2. Elementwise sparse*dense  (should be unaffected — verify no regression)

Usage:
  conda run -n scorch python tools/benchmark_compiler_optimization.py
"""

from __future__ import annotations

import statistics
import time
from typing import Dict, List, Tuple

import torch

import scorch
import scorch_ops as ops
from scorch import STensor
from scorch.ops import einsum


def _build_sparse_dense(n: int, sparsity: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    a = torch.rand((n, n), generator=gen, dtype=torch.float32)
    b = torch.rand((n, n), generator=gen, dtype=torch.float32)
    mask = torch.rand((n, n), generator=gen) > sparsity
    a = a * mask
    return a, b


def _time_fn(fn, warmup: int, repeats: int) -> List[float]:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    return timings


def _time_compiler_spmm(
    a_dense: torch.Tensor, b_dense: torch.Tensor, warmup: int, repeats: int,
) -> Tuple[torch.Tensor, List[float]]:
    a_s = STensor.from_torch(a_dense, "A").to_sparse("ds")
    b_s = STensor.from_torch(b_dense, "B")

    for _ in range(warmup):
        scorch.matmul(a_s, b_s, format="dd", use_cache=False, output_mode_order=[0, 1])

    timings: List[float] = []
    result = None
    for _ in range(repeats):
        td: Dict[str, float] = {}
        result = scorch.matmul(
            a_s, b_s, format="dd", use_cache=False,
            output_mode_order=[0, 1], time_dict=td,
        )
        timings.append(td["eval_time"])
    assert isinstance(result, torch.Tensor)
    return result, timings


def _time_prebuilt_spmm(
    a_dense: torch.Tensor, b_dense: torch.Tensor, warmup: int, repeats: int,
) -> Tuple[torch.Tensor, List[float]]:
    a_s = STensor.from_torch(a_dense, "A").to_sparse("ds")
    b_s = STensor.from_torch(b_dense, "B")
    n = a_dense.shape[0]
    kernel_args = [
        [n, n], list(a_s.shape), a_s.index.mode_indices, a_s.values,
        list(b_s.shape), b_s.index.mode_indices, b_s.values,
    ]
    for _ in range(warmup):
        ops.spmm_csr_float_direct(*kernel_args)
    timings = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = ops.spmm_csr_float_direct(*kernel_args)
        timings.append(time.perf_counter() - t0)
    return result.storage.value.view(n, n), timings


def _time_compiler_elementwise(
    a_dense: torch.Tensor, b_dense: torch.Tensor, warmup: int, repeats: int,
) -> Tuple[torch.Tensor, List[float]]:
    a_s = STensor.from_torch(a_dense, "A").to_sparse("ds")
    b_s = STensor.from_torch(b_dense, "B")
    n = a_dense.shape[0]

    for _ in range(warmup):
        einsum("ij,ij->ij", a_s, b_s, format="dd", use_cache=False)

    timings: List[float] = []
    result = None
    for _ in range(repeats):
        td: Dict[str, float] = {}
        result = einsum(
            "ij,ij->ij", a_s, b_s, format="dd", use_cache=False, time_dict=td,
        )
        timings.append(td["eval_time"])
    if isinstance(result, STensor):
        return result.values.view(n, n), timings
    return result, timings


def _fmt(times: List[float], gflops: float) -> str:
    mn = min(times) * 1e3
    med = statistics.median(times) * 1e3
    gf = gflops / (statistics.median(times)) if statistics.median(times) > 0 else 0.0
    return f"{mn:9.3f} {med:9.3f} {gf:9.2f}"


def main() -> None:
    n = 4096
    sparsity = 0.95
    seed = 0
    warmup = 3
    repeats = 10

    a_dense, b_dense = _build_sparse_dense(n, sparsity, seed)
    nnz = int(a_dense.to_sparse_csr().values().numel())
    spmm_gflops = 2.0 * nnz * n / 1e9
    ew_gflops = nnz / 1e9  # 1 mul per nnz

    torch_ref = torch.sparse.mm(a_dense.to_sparse_csr(), b_dense)
    ew_ref = a_dense * b_dense

    # ── SpMM ────────────────────────────────────────────────────────────
    compiler_mm, compiler_mm_t = _time_compiler_spmm(
        a_dense, b_dense, warmup, repeats
    )
    assert torch.allclose(compiler_mm, torch_ref, atol=1e-3, rtol=1e-3)

    direct_mm, direct_mm_t = _time_prebuilt_spmm(
        a_dense, b_dense, warmup, repeats
    )
    assert torch.allclose(direct_mm, torch_ref, atol=1e-3, rtol=1e-3)

    torch_mm_t = _time_fn(
        lambda: torch.sparse.mm(a_dense.to_sparse_csr(), b_dense), warmup, repeats
    )

    # ── Elementwise ─────────────────────────────────────────────────────
    compiler_ew, compiler_ew_t = _time_compiler_elementwise(
        a_dense, b_dense, warmup, repeats
    )
    assert torch.allclose(compiler_ew, ew_ref, atol=1e-3, rtol=1e-3)

    torch_ew_t = _time_fn(lambda: a_dense * b_dense, warmup, repeats)

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\nCompiler optimization benchmark  n={n}  sparsity={sparsity}  nnz={nnz:,}")
    print(f"warmup={warmup}  repeats={repeats}")
    print("=" * 80)

    print(f"\nSpMM  (GFLOP = {spmm_gflops:.3f})")
    print(f"  {'variant':<30s} {'min_ms':>9s} {'median_ms':>9s} {'GFLOP/s':>9s}")
    print(f"  {'-'*60}")
    print(f"  {'compiler-generated':<30s} {_fmt(compiler_mm_t, spmm_gflops)}")
    print(f"  {'handwritten direct (best)':<30s} {_fmt(direct_mm_t, spmm_gflops)}")
    print(f"  {'torch.sparse.mm':<30s} {_fmt(torch_mm_t, spmm_gflops)}")

    cm = statistics.median(compiler_mm_t) * 1e3
    dm = statistics.median(direct_mm_t) * 1e3
    tm = statistics.median(torch_mm_t) * 1e3
    print(f"\n  compiler vs torch:    {tm/cm:.2f}x")
    print(f"  compiler vs direct:   {dm/cm:.2f}x  (1.00 = compiler matches best handwritten)")

    print(f"\nElementwise sparse*dense  (GFLOP = {ew_gflops:.6f})")
    print(f"  {'variant':<30s} {'min_ms':>9s} {'median_ms':>9s} {'GFLOP/s':>9s}")
    print(f"  {'-'*60}")
    print(f"  {'compiler-generated':<30s} {_fmt(compiler_ew_t, ew_gflops)}")
    print(f"  {'torch (dense a*b)':<30s} {_fmt(torch_ew_t, ew_gflops)}")


if __name__ == "__main__":
    main()
