#!/usr/bin/env python3
"""Benchmark SpMSpM: A_sparse @ A_sparse^T (COO format)."""

from __future__ import annotations

import argparse
from pathlib import Path

import scipy.sparse
import torch

import scorch
from _utils import (
    MATRIX_SETS,
    add_common_args,
    benchmark_fn,
    check_correctness,
    download_matrix,
    plot_scatter_loglog,
    ResultsCollector,
    scipy_to_torch,
    suppress_torch_warnings,
    to_scorch_coo,
)


def _truncate_to_square(csr: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """Truncate a non-square matrix to min(rows,cols) x min(rows,cols)."""
    r, c = csr.shape
    dim = min(r, c)
    if r == dim and c == dim:
        return csr
    return csr[:dim, :dim]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpMSpM benchmark: A_sparse @ A_sparse^T")
    add_common_args(parser)
    args = parser.parse_args()

    suppress_torch_warnings()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / (args.csv or "spmspm_results.csv")
    plot_path = out_dir / f"spmspm.{args.format}"

    if args.plot_only:
        df = ResultsCollector.load(csv_path)
        plot_scatter_loglog(df, "SpMSpM Performance (A @ A^T)", plot_path)
        return

    collector = ResultsCollector(csv_path)
    matrices = MATRIX_SETS["spmspm"]

    print(f"SpMSpM Benchmark  (warmup={args.warmup}, repeats={args.repeats})")
    print("=" * 80)

    for ssid, group, name, desc in matrices:
        print(f"\n{name} (id={ssid}): {desc}")
        try:
            csr = download_matrix(ssid, group, name)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # Filter: bounded size
        r, c = csr.shape
        nnz = csr.nnz
        if max(r, c) >= 100_000 or nnz >= 10_000_000:
            print(f"  SKIP: too large ({r}x{c}, nnz={nnz:,})")
            continue

        csr = _truncate_to_square(csr)
        n = csr.shape[0]
        nnz = csr.nnz
        print(f"  shape=({n}, {n}), nnz={nnz:,}")

        # COO tensors
        torch_coo = scipy_to_torch(csr, fmt="coo")
        torch_coo_t = torch_coo.t().coalesce()

        # --- PyTorch ---
        ref, pt_timing = benchmark_fn(
            lambda: torch.matmul(torch_coo, torch_coo_t),
            warmup=args.warmup, repeats=args.repeats,
        )
        for t in pt_timing.times:
            collector.append("PyTorch", name, ssid, n, n, nnz, t)
        print(f"  PyTorch  median={pt_timing.median_ms:.3f} ms")

        # --- Scorch ---
        a_st = to_scorch_coo(torch_coo, "A")
        at_st = to_scorch_coo(torch_coo_t, "B")
        sc_result, sc_timing = benchmark_fn(
            lambda: scorch.matmul(a_st, at_st),
            warmup=args.warmup, repeats=args.repeats,
        )
        for t in sc_timing.times:
            collector.append("Scorch", name, ssid, n, n, nnz, t)
        print(f"  Scorch   median={sc_timing.median_ms:.3f} ms")

        # Correctness — convert both sparse outputs to dense
        ref_dense = ref.to_dense() if ref.is_sparse or ref.is_sparse_csr else ref
        if isinstance(sc_result, torch.Tensor):
            sc_dense = sc_result.to_dense() if sc_result.is_sparse or sc_result.is_sparse_csr else sc_result
        else:
            sc_dense = sc_result.to_torch()
            if sc_dense.is_sparse or sc_dense.is_sparse_csr:
                sc_dense = sc_dense.to_dense()
        check_correctness(sc_dense, ref_dense, name)

    df = collector.to_dataframe()
    plot_scatter_loglog(df, "SpMSpM Performance (A @ A^T)", plot_path)
    print(f"\nCSV saved to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
