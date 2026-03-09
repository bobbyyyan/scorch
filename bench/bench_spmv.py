#!/usr/bin/env python3
"""Benchmark SpMV: A_sparse @ v_dense."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    to_scorch_csr,
    to_scorch_dense,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SpMV benchmark: A_sparse @ v_dense")
    add_common_args(parser)
    args = parser.parse_args()

    suppress_torch_warnings()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / (args.csv or "spmv_results.csv")
    plot_path = out_dir / f"spmv.{args.format}"

    if args.plot_only:
        df = ResultsCollector.load(csv_path)
        plot_scatter_loglog(df, "SpMV Performance", plot_path)
        return

    collector = ResultsCollector(csv_path)
    matrices = MATRIX_SETS["spmv"]

    print(f"SpMV Benchmark  (warmup={args.warmup}, repeats={args.repeats})")
    print("=" * 80)

    for ssid, group, name, desc in matrices:
        print(f"\n{name} (id={ssid}): {desc}")
        try:
            csr = download_matrix(ssid, group, name)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        n_rows, n_cols = csr.shape
        nnz = csr.nnz
        print(f"  shape=({n_rows}, {n_cols}), nnz={nnz:,}")

        v_dense = torch.rand(n_cols, dtype=torch.float32)

        # --- PyTorch ---
        torch_csr = scipy_to_torch(csr, fmt="csr")
        ref, pt_timing = benchmark_fn(
            lambda: torch.matmul(torch_csr, v_dense),
            warmup=args.warmup, repeats=args.repeats,
        )
        for t in pt_timing.times:
            collector.append("PyTorch", name, ssid, n_rows, n_cols, nnz, t)
        print(f"  PyTorch  median={pt_timing.median_ms:.3f} ms")

        # --- Scorch ---
        a_st = to_scorch_csr(csr, "A")
        v_st = to_scorch_dense(v_dense, "v")
        sc_result, sc_timing = benchmark_fn(
            lambda: scorch.matmul(a_st, v_st),
            warmup=args.warmup, repeats=args.repeats,
        )
        for t in sc_timing.times:
            collector.append("Scorch", name, ssid, n_rows, n_cols, nnz, t)
        print(f"  Scorch   median={sc_timing.median_ms:.3f} ms")

        # Correctness
        sc_dense = sc_result.to_torch() if hasattr(sc_result, "to_torch") else sc_result
        check_correctness(sc_dense.flatten(), ref.flatten(), name)

    df = collector.to_dataframe()
    plot_scatter_loglog(df, "SpMV Performance", plot_path)
    print(f"\nCSV saved to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
