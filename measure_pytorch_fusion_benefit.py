"""
Measure the fusion benefit for SDDMM in PyTorch alone.

This script compares:
1. PyTorch fused SDDMM using torch.sparse.sampled_addmm (for CSR format)
2. PyTorch unfused SDDMM (separate matmul + elementwise multiply)

Uses SuiteSparse matrices to benchmark on real-world sparse patterns.
"""

import ssgetpy
from scipy.io import mmread
from pathlib import Path
import torch
import numpy as np
import time
import pandas as pd
import traceback
import warnings

# Suppress specific PyTorch UserWarning about Sparse CSR tensor support
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Sparse CSR tensor support is in beta state.*",
)


def scipy_sparse_to_torch_csr(matrix):
    """Convert scipy sparse matrix to PyTorch sparse CSR tensor."""
    matrix = matrix.tocsr()
    crow_indices = torch.LongTensor(matrix.indptr)
    col_indices = torch.LongTensor(matrix.indices)
    values = torch.FloatTensor(matrix.data)
    shape = matrix.shape
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, torch.Size(shape))


def benchmark_pytorch_fused_sddmm(sparse_mask, dense_A, dense_B, num_trials=10, warmup=3):
    """
    Benchmark PyTorch FUSED implementation of SDDMM.

    Uses torch.sparse.sampled_addmm which computes:
    sparse_mask * (dense_A @ dense_B) in a fused manner.

    This is equivalent to: sampled_addmm(input, mat1, mat2, beta=0, alpha=1)
    which computes: beta * input + alpha * (mat1 @ mat2), sampled at sparse_mask locations
    """
    # Warmup runs
    for _ in range(warmup):
        # beta=0 means we ignore the input (sparse_mask values), alpha=1 means normal matmul
        # The result will have the sparsity pattern of sparse_mask
        result = torch.sparse.sampled_addmm(sparse_mask, dense_A, dense_B, beta=0.0, alpha=1.0)

    # Benchmark runs
    times = []
    for _ in range(num_trials):
        start_time = time.perf_counter()
        result = torch.sparse.sampled_addmm(sparse_mask, dense_A, dense_B, beta=0.0, alpha=1.0)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return np.mean(times), np.std(times)


def benchmark_pytorch_unfused_sddmm(sparse_mask, dense_A, dense_B, num_trials=10, warmup=3):
    """
    Benchmark PyTorch UNFUSED implementation of SDDMM.

    Computes in two steps:
    1. Dense matmul: temp = dense_A @ dense_B (fully materialized)
    2. Elementwise multiply with sparse mask: result = sparse_mask * temp
    """
    # Warmup runs
    for _ in range(warmup):
        temp = torch.matmul(dense_A, dense_B)
        result = torch.mul(sparse_mask, temp)

    # Benchmark runs
    times = []
    for _ in range(num_trials):
        start_time = time.perf_counter()
        temp = torch.matmul(dense_A, dense_B)  # Fully materialize the dense result
        result = torch.mul(sparse_mask, temp)  # Sample at sparse locations
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return np.mean(times), np.std(times)


def download_matrices_parallel(matrices, num_matrices=20):
    """Download matrices in parallel using wget."""
    import subprocess
    import os

    selected_matrices = matrices[:num_matrices]
    download_dir = Path("~/.ssgetpy/MM").expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(selected_matrices)} matrices in parallel...")

    # Create wget commands
    wget_commands = []
    for matrix in selected_matrices:
        matrix_dir = download_dir / matrix.group / matrix.name
        matrix_dir.mkdir(parents=True, exist_ok=True)

        matrix_file = matrix_dir / f"{matrix.name}.mtx"

        # Only download if not already present
        if not matrix_file.exists():
            # Get download URL
            url = f"https://suitesparse-collection-website.herokuapp.com/MM/{matrix.group}/{matrix.name}.tar.gz"
            wget_cmd = f"wget -q -O - {url} | tar -xz -C {matrix_dir} --strip-components=1"
            wget_commands.append((wget_cmd, matrix.name))

    if wget_commands:
        print(f"  Downloading {len(wget_commands)} new matrices (others are cached)...")

        # Run wget commands in parallel (limit to 8 concurrent downloads)
        from concurrent.futures import ThreadPoolExecutor

        def run_wget(cmd_tuple):
            cmd, name = cmd_tuple
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
                return name, True
            except subprocess.CalledProcessError as e:
                return name, False

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(run_wget, wget_commands))

        successful = sum(1 for _, success in results if success)
        print(f"  Downloaded {successful}/{len(wget_commands)} matrices successfully")
    else:
        print("  All matrices already cached")

    return selected_matrices


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Measure PyTorch fusion benefit for SDDMM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Compares:
  1. PyTorch FUSED:   torch.sparse.sampled_addmm (CSR format)
  2. PyTorch UNFUSED: torch.matmul + torch.mul (separate operations)
        """)
    parser.add_argument('--output', type=str, default='pytorch_fusion_benefit_results.csv',
                        help='Output CSV file for results (default: pytorch_fusion_benefit_results.csv)')
    parser.add_argument('--continue', dest='continue_run', action='store_true',
                        help='Continue from previous run: read existing CSV and skip already-processed matrices')
    parser.add_argument('--num-matrices', type=int, default=None,
                        help='Limit number of matrices to process (default: all)')
    parser.add_argument('--matrix-dir', type=str, default='/scratch/suitesparse',
                        help='Base directory for matrices (default: /scratch/suitesparse)')
    parser.add_argument('--summary-only', action='store_true',
                        help='Only print summary statistics from existing CSV, do not run benchmarks')

    args = parser.parse_args()

    # Handle --summary-only flag
    if args.summary_only:
        if not Path(args.output).exists():
            print(f"Error: Output file '{args.output}' not found!")
            return

        print("=" * 80)
        print(f"SUMMARY STATISTICS from {args.output}")
        print("=" * 80)

        results_df = pd.read_csv(args.output)

        if len(results_df) > 0:
            avg_fusion_benefit = results_df['Fusion_Benefit'].mean()
            median_fusion_benefit = results_df['Fusion_Benefit'].median()
            min_fusion_benefit = results_df['Fusion_Benefit'].min()
            max_fusion_benefit = results_df['Fusion_Benefit'].max()

            print(f"\nFusion Benefit Statistics:")
            print(f"  Average:  {avg_fusion_benefit:.2f}×")
            print(f"  Median:   {median_fusion_benefit:.2f}×")
            print(f"  Min:      {min_fusion_benefit:.2f}×")
            print(f"  Max:      {max_fusion_benefit:.2f}×")

            print("\n\nTable Format:")
            print("| Implementation | Time (relative) | Description |")
            print("|----------------|-----------------|-------------|")
            print(f"| PyTorch UNFUSED | 1.0× | matmul + elementwise multiply (separate) |")
            print(f"| PyTorch FUSED | {1.0/avg_fusion_benefit:.2f}× | torch.sparse.sampled_addmm (fused) |")
            print(f"\n  → Average fusion benefit: {avg_fusion_benefit:.2f}×")

            # Show top 5 matrices by fusion benefit
            print("\n\nTop 5 matrices by fusion benefit:")
            top_5 = results_df.nlargest(5, 'Fusion_Benefit')
            for i, row in top_5.iterrows():
                print(f"  {row['Matrix_Name']:30s}  {row['Fusion_Benefit']:.2f}×")

            # Show bottom 5 matrices by fusion benefit
            print("\n\nBottom 5 matrices by fusion benefit:")
            bottom_5 = results_df.nsmallest(5, 'Fusion_Benefit')
            for i, row in bottom_5.iterrows():
                print(f"  {row['Matrix_Name']:30s}  {row['Fusion_Benefit']:.2f}×")

            print(f"\n\nTotal matrices: {len(results_df)}")
        else:
            print("No results found in CSV.")

        return

    print("=" * 80)
    print("Measuring PyTorch Fusion Benefit for SDDMM")
    print("=" * 80)
    print("\nComparing:")
    print("  1. PyTorch FUSED:   torch.sparse.sampled_addmm (CSR format)")
    print("  2. PyTorch UNFUSED: torch.matmul + torch.mul (separate operations)")
    print("=" * 80)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Fetch all SuiteSparse matrices
    print("\nFetching SuiteSparse matrices metadata...")
    matrices = ssgetpy.search(limit=10000)
    # Sort by size (ascending) to process smaller ones first
    matrices = sorted(matrices, key=lambda m: m.nnz)
    print(f"Found {len(matrices)} matrices")

    # Handle --continue flag: load existing results and skip processed matrices
    results = []
    processed_matrices = set()

    if args.continue_run and Path(args.output).exists():
        print(f"\nContinuing from existing results in {args.output}")
        try:
            existing_df = pd.read_csv(args.output)
            results = existing_df.to_dict('records')
            processed_matrices = set(existing_df['Matrix_Name'].values)
            print(f"  Loaded {len(results)} existing results")
            print(f"  Will skip {len(processed_matrices)} already-processed matrices")
        except Exception as e:
            print(f"  Warning: Could not load existing CSV: {e}")
            print(f"  Starting fresh benchmark")
            results = []
            processed_matrices = set()
    elif args.continue_run:
        print(f"\n--continue specified but {args.output} not found, starting fresh")

    # Select matrices to process
    num_matrices = args.num_matrices if args.num_matrices else len(matrices)
    selected_matrices = matrices[:num_matrices]

    matrices_to_process = len([m for m in selected_matrices if m.name not in processed_matrices])
    print(f"\nBenchmarking {len(selected_matrices)} matrices ({matrices_to_process} to process, {len(processed_matrices)} already done)")
    print(f"Matrix directory: {args.matrix_dir}\n")

    for idx, matrix in enumerate(selected_matrices):
        # Skip if already processed (when using --continue)
        if matrix.name in processed_matrices:
            print(f"[{idx+1}/{len(selected_matrices)}] Skipping {matrix.name} (already processed)")
            continue
        try:
            print(f"\n[{idx+1}/{len(selected_matrices)}] Processing {matrix.name} ({matrix.group})")
            print(f"  Size: {matrix.rows}×{matrix.cols}, NNZ: {matrix.nnz}")

            # Try to find matrix in matrix_dir first
            matrix_path = Path(args.matrix_dir) / matrix.name / f"{matrix.name}.mtx"

            if not matrix_path.exists():
                # Fall back to ssgetpy cache directory
                matrix_path = Path(f"~/.ssgetpy/MM/{matrix.group}/{matrix.name}/{matrix.name}.mtx").expanduser()

            if not matrix_path.exists():
                print(f"  Matrix not found in {args.matrix_dir} or ~/.ssgetpy/MM/")
                print(f"  Attempting to download from SuiteSparse collection...")

                # Download to ssgetpy directory
                download_dir = Path("~/.ssgetpy/MM").expanduser()
                matrix_dir = download_dir / matrix.group / matrix.name
                matrix_dir.mkdir(parents=True, exist_ok=True)
                matrix_file = matrix_dir / f"{matrix.name}.mtx"

                # Download
                import subprocess
                url = f"https://suitesparse-collection-website.herokuapp.com/MM/{matrix.group}/{matrix.name}.tar.gz"
                wget_cmd = f"wget -q -O - {url} | tar -xz -C {matrix_dir} --strip-components=1"
                try:
                    subprocess.run(wget_cmd, shell=True, check=True, capture_output=True, timeout=300)
                    matrix_path = matrix_file
                    print(f"  Downloaded to {matrix_path}")
                except Exception as e:
                    print(f"  Download failed: {e}")
                    print(f"  Skipping matrix")
                    continue

            if not matrix_path.exists():
                print(f"  Skipping: matrix file not found after download attempt")
                continue

            sparse_matrix = mmread(matrix_path.resolve())

            # Convert to PyTorch CSR format (required for sampled_addmm)
            torch_sparse_csr = scipy_sparse_to_torch_csr(sparse_matrix)

            # Create dense matrices for SDDMM: result[i,j] = mask[i,j] * (A[i,k] @ B[k,j])
            # Using larger feature dimension to see fusion benefits
            feature_dim = 128
            dense_matrix_A = torch.rand((torch_sparse_csr.shape[0], feature_dim), dtype=torch.float32)
            dense_matrix_B = torch.rand((feature_dim, torch_sparse_csr.shape[1]), dtype=torch.float32)

            nnz = sparse_matrix.nnz

            # Benchmark both implementations
            print("  Benchmarking PyTorch FUSED (sampled_addmm)...")
            fused_time, fused_std = benchmark_pytorch_fused_sddmm(
                torch_sparse_csr, dense_matrix_A, dense_matrix_B
            )

            print("  Benchmarking PyTorch UNFUSED (matmul + mul)...")
            unfused_time, unfused_std = benchmark_pytorch_unfused_sddmm(
                torch_sparse_csr, dense_matrix_A, dense_matrix_B
            )

            # Calculate fusion benefit
            fusion_benefit = unfused_time / fused_time

            print(f"  Results:")
            print(f"    PyTorch UNFUSED:  {unfused_time*1000:.2f} ± {unfused_std*1000:.2f} ms (1.0×)")
            print(f"    PyTorch FUSED:    {fused_time*1000:.2f} ± {fused_std*1000:.2f} ms ({fusion_benefit:.2f}×)")
            print(f"    Fusion Benefit:   {fusion_benefit:.2f}×")

            # Store results
            results.append({
                'Matrix_Name': matrix.name,
                'Matrix_Group': matrix.group,
                'Rows': torch_sparse_csr.shape[0],
                'Columns': torch_sparse_csr.shape[1],
                'NNZ': nnz,
                'Feature_Dim': feature_dim,
                'Unfused_Time_Mean': unfused_time,
                'Unfused_Time_Std': unfused_std,
                'Fused_Time_Mean': fused_time,
                'Fused_Time_Std': fused_std,
                'Fusion_Benefit': fusion_benefit
            })

            # Save incrementally
            results_df = pd.DataFrame(results)
            results_df.to_csv(args.output, index=False)
            print(f"  Progress saved to {args.output} ({len(results)}/{len(selected_matrices)} matrices)")

        except Exception as e:
            print(f"  Error: {e}")
            traceback.print_exc()
            continue

    # Final results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output, index=False)
    else:
        print("\nNo results to save!")
        return

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if len(results_df) > 0:
        avg_fusion_benefit = results_df['Fusion_Benefit'].mean()
        median_fusion_benefit = results_df['Fusion_Benefit'].median()
        min_fusion_benefit = results_df['Fusion_Benefit'].min()
        max_fusion_benefit = results_df['Fusion_Benefit'].max()

        print(f"\nFusion Benefit Statistics:")
        print(f"  Average:  {avg_fusion_benefit:.2f}×")
        print(f"  Median:   {median_fusion_benefit:.2f}×")
        print(f"  Min:      {min_fusion_benefit:.2f}×")
        print(f"  Max:      {max_fusion_benefit:.2f}×")

        print("\n\nTable Format:")
        print("| Implementation | Time (relative) | Description |")
        print("|----------------|-----------------|-------------|")
        print(f"| PyTorch UNFUSED | 1.0× | matmul + elementwise multiply (separate) |")
        print(f"| PyTorch FUSED | {1.0/avg_fusion_benefit:.2f}× | torch.sparse.sampled_addmm (fused) |")
        print(f"\n  → Average fusion benefit: {avg_fusion_benefit:.2f}×")

        print(f"\n\nDetailed results saved to '{args.output}'")

        # Show top 5 matrices by fusion benefit
        print("\n\nTop 5 matrices by fusion benefit:")
        top_5 = results_df.nlargest(5, 'Fusion_Benefit')
        for i, row in top_5.iterrows():
            print(f"  {row['Matrix_Name']:30s}  {row['Fusion_Benefit']:.2f}×")
    else:
        print("No successful benchmarks completed.")


if __name__ == "__main__":
    main()
