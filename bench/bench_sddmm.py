#!/usr/bin/env python3
"""Benchmark SDDMM: S ⊙ (A_dense @ B_dense)."""

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
    to_scorch_coo,
    to_scorch_dense,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SDDMM benchmark: S ⊙ (A_dense @ B_dense)")
    add_common_args(parser)
    parser.add_argument("--k", type=int, default=128,
                        help="Inner dimension for dense matmul (default: 128)")
    args = parser.parse_args()

    suppress_torch_warnings()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / (args.csv or "sddmm_results.csv")
    plot_path = out_dir / f"sddmm.{args.format}"

    if args.plot_only:
        df = ResultsCollector.load(csv_path)
        plot_scatter_loglog(df, f"SDDMM Performance (k={args.k})", plot_path)
        return

    collector = ResultsCollector(csv_path)
    matrices = MATRIX_SETS["sddmm"]

    print(f"SDDMM Benchmark  (k={args.k}, warmup={args.warmup}, repeats={args.repeats})")
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
        k = args.k

        # Filter: bounded size
        if max(n_rows, n_cols) >= 800_000:
            print(f"  SKIP: too large ({n_rows}x{n_cols})")
            continue

        print(f"  shape=({n_rows}, {n_cols}), nnz={nnz:,}, k={k}")

        dense_A = torch.rand(n_rows, k, dtype=torch.float32)
        dense_B = torch.rand(k, n_cols, dtype=torch.float32)

        # COO sparse mask
        torch_coo = scipy_to_torch(csr, fmt="coo")

        # --- PyTorch: S ⊙ (A @ B) ---
        ref, pt_timing = benchmark_fn(
            lambda: torch.mul(torch_coo, torch.matmul(dense_A, dense_B)),
            warmup=args.warmup, repeats=args.repeats,
        )
        ref_dense = ref.to_dense() if ref.is_sparse else ref
        for t in pt_timing.times:
            collector.append("PyTorch", name, ssid, n_rows, n_cols, nnz, t, k=k)
        print(f"  PyTorch  median={pt_timing.median_ms:.3f} ms")

        # --- Scorch: einsum("ij,ik,kj->ij", S, A, B) ---
        s_st = to_scorch_coo(torch_coo, "S")
        a_st = to_scorch_dense(dense_A, "A")
        b_st = to_scorch_dense(dense_B, "B")
        sc_result, sc_timing = benchmark_fn(
            lambda: scorch.einsum("ij,ik,kj->ij", s_st, a_st, b_st),
            warmup=args.warmup, repeats=args.repeats,
        )
        for t in sc_timing.times:
            collector.append("Scorch", name, ssid, n_rows, n_cols, nnz, t, k=k)
        print(f"  Scorch   median={sc_timing.median_ms:.3f} ms")

        # Correctness
        if isinstance(sc_result, torch.Tensor):
            sc_dense = sc_result.to_dense() if sc_result.is_sparse else sc_result
        else:
            sc_dense = sc_result.to_torch()
            if sc_dense.is_sparse or sc_dense.is_sparse_csr:
                sc_dense = sc_dense.to_dense()
        check_correctness(sc_dense, ref_dense, name)

    df = collector.to_dataframe()
    plot_scatter_loglog(df, f"SDDMM Performance (k={args.k})", plot_path)
    print(f"\nCSV saved to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
