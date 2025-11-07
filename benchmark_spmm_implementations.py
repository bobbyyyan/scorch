#!/usr/bin/env python3
"""
Benchmark script to compare different SPMM implementation speeds on SuiteSparse matrices.

This script tests all the SPMM implementations exposed in csrc/ops.cpp and measures
their relative performance on a curated set of SuiteSparse matrices.
"""

import ssgetpy
from scipy.io import mmread
from pathlib import Path
import torch
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import scorch._C as scorch_cpp

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Sparse CSR tensor support is in beta state.*")


def scipy_sparse_to_torch_sparse_csr(matrix):
    """Convert scipy sparse matrix to PyTorch sparse CSR tensor."""
    matrix = matrix.tocsr()
    crow_indices = torch.LongTensor(matrix.indptr)
    col_indices = torch.LongTensor(matrix.indices)
    values = torch.FloatTensor(matrix.data)
    shape = matrix.shape
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, torch.Size(shape))


def scipy_sparse_to_torch_sparse_coo(matrix):
    """Convert scipy sparse matrix to PyTorch sparse COO tensor."""
    matrix = matrix.tocoo()
    indices = np.vstack((matrix.row, matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(matrix.data)
    shape = matrix.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))


def extract_csr_components(torch_sparse_csr):
    """Extract CSR components for C++ interface."""
    crow_indices = torch_sparse_csr.crow_indices().int()
    col_indices = torch_sparse_csr.col_indices().int()
    values = torch_sparse_csr.values()
    shape = list(torch_sparse_csr.shape)

    # Format mode_indices as expected by C++ interface
    mode_indices = [
        [],  # Dense first dimension
        [crow_indices, col_indices]  # Compressed second dimension
    ]

    return shape, mode_indices, values


def extract_coo_components(torch_sparse_coo):
    """Extract COO components for C++ interface."""
    # Coalesce first to ensure indices are accessible
    torch_sparse_coo = torch_sparse_coo.coalesce()
    indices = torch_sparse_coo.indices()
    values = torch_sparse_coo.values()
    shape = list(torch_sparse_coo.shape)

    # Format mode_indices as expected by C++ interface
    mode_indices = [
        [indices[0].int()],  # Row coordinates
        [indices[1].int()]   # Column coordinates
    ]

    return shape, mode_indices, values


def benchmark_implementation(impl_name, impl_func, A_args, B_args, result_shape, warmup=3, iterations=10):
    """Benchmark a single SPMM implementation."""
    # Warmup
    for _ in range(warmup):
        try:
            _ = impl_func(*A_args, *B_args)
        except Exception as e:
            return None, str(e)

    # Timing runs
    times = []
    for _ in range(iterations):
        try:
            start = time.perf_counter()
            result = impl_func(*A_args, *B_args)
            end = time.perf_counter()
            times.append(end - start)
        except Exception as e:
            return None, str(e)

    return times, None


def run_benchmark(matrix_names=None, k_values=[1, 16, 32, 64, 128, 256, 512], num_trials=10, small_only=False):
    """
    Run benchmark comparing all SPMM implementations.

    Args:
        matrix_names: List of specific matrix names to benchmark. If None, uses a default set.
        k_values: List of k values (columns in dense matrix) to test
        num_trials: Number of timing iterations per test
        small_only: If True, only use small matrices (< 1000 rows)
    """
    # Default set of diverse matrices if none specified
    if matrix_names is None:
        # Diverse set of matrices ranging from small to large
        # Using matrix IDs which are more reliable
        # Format: ID, Name, Rows, NNZ (approximate)
        matrix_ids = [
            # Tiny (< 500 rows) - fast baseline tests
            166,    # west0067 - 67x67, 294 NNZ
            8,      # ash292 - 292x292, 2.2K NNZ
            1138,   # oscil_dcop_27 - 430x430, 1.5K NNZ
            2,      # 494_bus - 494x494, 1.7K NNZ

            # Small (500-2K rows)
            4,      # 685_bus - 685x685, 3.2K NNZ
            947,    # nos4 - 100 x 100, 594 NNZ (dense pattern)
            948,    # nos5 - 468 x 468, 5.2K NNZ
            950,    # nos7 - 729 x 729, 4.6K NNZ

            # Medium (2K-5K rows)
            935,    # bcsstk15 - 3.9K x 3.9K, 117K NNZ
            679,    # bcsstk16 - 4.9K x 4.9K, 290K NNZ

            # Large (5K-15K rows) - good for showing performance differences
            938,    # bcsstk18 - 11.9K x 11.9K, 149K NNZ
            939,    # bcsstk23 - 3.1K x 3.1K, 21K NNZ
            940,    # bcsstk24 - 3.6K x 3.6K, 159K NNZ

            # Very sparse vs dense patterns
            662,    # lp_qap8 - 912x1632, 7.3K NNZ (rectangular, sparse)
        ]

        # If small_only flag is set, only use matrices < 1000 rows
        if small_only:
            matrix_ids = [166, 8, 1138, 2, 4, 947, 948, 950, 662]
            print("Small-only mode: Using only matrices with < 1000 rows")
    else:
        matrix_ids = None

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"spmm_implementation_benchmark_{timestamp}.csv"

    results = []

    # Determine node-specific matrix storage location
    node_name = os.uname().nodename
    if "sapling" in node_name:
        base_path = Path("/scratch2/suitesparse")
    elif "redwood" in node_name:
        base_path = Path("/scratch/suitesparse")
    else:
        base_path = Path("~/.ssgetpy/MM").expanduser()

    # Get matrix metadata
    print("Fetching SuiteSparse matrix metadata...")
    all_matrices = ssgetpy.search(limit=10000)

    # Build lookup by name and ID
    matrix_by_name = {m.name: m for m in all_matrices}
    matrix_by_id = {m.id: m for m in all_matrices}

    # Resolve matrix names or IDs to matrix objects
    matrices_to_test = []
    if matrix_ids is not None:
        # Using IDs
        for mid in matrix_ids:
            if mid in matrix_by_id:
                matrices_to_test.append(matrix_by_id[mid])
            else:
                print(f"Warning: Matrix ID {mid} not found")
    else:
        # Using names
        for name in matrix_names:
            if name in matrix_by_name:
                matrices_to_test.append(matrix_by_name[name])
            else:
                print(f"Warning: Matrix '{name}' not found")

    if not matrices_to_test:
        print("Error: No valid matrices found to benchmark!")
        return pd.DataFrame(), csv_filename

    for matrix_info in tqdm(matrices_to_test, desc="Processing matrices"):
        try:
            matrix_name = matrix_info.name
            print(f"\n{'='*80}")
            print(f"Processing matrix: {matrix_name} (ID: {matrix_info.id}, Group: {matrix_info.group})")
            print(f"{'='*80}")

            # Find and load matrix based on node type
            if "sapling" in node_name:
                matrix_path = base_path / f"{matrix_name}.mtx"
            elif "redwood" in node_name:
                matrix_path = base_path / matrix_info.group / matrix_name / f"{matrix_name}.mtx"
            else:
                # Standard ssgetpy location: ~/.ssgetpy/MM/{group}/{name}/{name}.mtx
                matrix_path = base_path / matrix_info.group / matrix_name / f"{matrix_name}.mtx"

            # If file doesn't exist, try to download it
            if not matrix_path.exists():
                print(f"Matrix file not found at {matrix_path}, attempting download...")
                try:
                    matrix_info.download(format='MM', extract=True)
                    # Re-check the path after download
                    if not matrix_path.exists():
                        print(f"Download completed but path still incorrect, skipping...")
                        continue
                except Exception as e:
                    print(f"Failed to download matrix: {e}, skipping...")
                    continue

            sparse_matrix = mmread(matrix_path)
            print(f"Matrix shape: {sparse_matrix.shape}, NNZ: {sparse_matrix.nnz}")

            # Convert to CSR format (used by most implementations)
            torch_sparse_csr = scipy_sparse_to_torch_sparse_csr(sparse_matrix)

            # Extract CSR components for C++ interface
            A_shape_csr, A_mode_indices_csr, A_values_csr = extract_csr_components(torch_sparse_csr)

            # COO will be created on-demand for the one COO implementation
            torch_sparse_coo = None
            A_shape_coo, A_mode_indices_coo, A_values_coo = None, None, None

            for k in k_values:
                print(f"\nTesting with k={k}")

                # Create dense matrix B
                B_dense = torch.rand((sparse_matrix.shape[1], k), dtype=torch.float32)
                B_shape = list(B_dense.shape)
                B_mode_indices = [[], []]  # Dense matrix
                B_values = B_dense

                result_shape = [sparse_matrix.shape[0], k]

                # Define all implementations to test
                implementations = [
                    # PyTorch baseline for comparison
                    ("pytorch", None),  # Special case, handled separately

                    ("csr_untiled", lambda rs, As, Am, Av, Bs, Bm, Bv:
                     scorch_cpp.spmm_csr_float_untiled(rs, As, Am, Av, Bs, Bm, Bv)),

                    ("csr_tiled_32", lambda rs, As, Am, Av, Bs, Bm, Bv:
                     scorch_cpp.spmm_csr_float(rs, As, Am, Av, Bs, Bm, Bv, 32)),

                    ("csr_tiled_64", lambda rs, As, Am, Av, Bs, Bm, Bv:
                     scorch_cpp.spmm_csr_float(rs, As, Am, Av, Bs, Bm, Bv, 64)),

                    ("csr_tiled_128", lambda rs, As, Am, Av, Bs, Bm, Bv:
                     scorch_cpp.spmm_csr_float(rs, As, Am, Av, Bs, Bm, Bv, 128)),

                    ("csr_optimized_128", lambda rs, As, Am, Av, Bs, Bm, Bv:
                     scorch_cpp.spmm_csr_float_optimized(rs, As, Am, Av, Bs, Bm, Bv, 128)),

                    ("csr_turbo_128", lambda rs, As, Am, Av, Bs, Bm, Bv:
                     scorch_cpp.spmm_csr_float_turbo(rs, As, Am, Av, Bs, Bm, Bv, 128)),

                    ("csr_ultra_256", lambda rs, As, Am, Av, Bs, Bm, Bv:
                     scorch_cpp.spmm_csr_float_ultra(rs, As, Am, Av, Bs, Bm, Bv, 256)),

                    ("csr_apex_256", lambda rs, As, Am, Av, Bs, Bm, Bv:
                     scorch_cpp.spmm_csr_float_apex(rs, As, Am, Av, Bs, Bm, Bv, 256)),

                    ("coo", lambda rs, As, Am, Av, Bs, Bm, Bv:
                     scorch_cpp.spmm_coo_float(rs, As, Am, Av, Bs, Bm, Bv)),
                ]

                for impl_name, impl_func in implementations:
                    print(f"  Testing {impl_name}...", end=" ", flush=True)

                    # Special handling for PyTorch baseline
                    if impl_name == "pytorch":
                        times = []
                        for _ in range(3):  # Warmup
                            _ = torch.sparse.mm(torch_sparse_csr, B_dense)
                        for _ in range(num_trials):
                            start = time.perf_counter()
                            _ = torch.sparse.mm(torch_sparse_csr, B_dense)
                            end = time.perf_counter()
                            times.append(end - start)

                        mean_time = np.mean(times)
                        std_time = np.std(times)
                        min_time = np.min(times)
                        print(f"Mean: {mean_time*1000:.3f}ms, Min: {min_time*1000:.3f}ms")

                        results.append({
                            'matrix_name': matrix_name,
                            'rows': sparse_matrix.shape[0],
                            'cols': sparse_matrix.shape[1],
                            'nnz': sparse_matrix.nnz,
                            'k': k,
                            'implementation': impl_name,
                            'mean_time': mean_time,
                            'std_time': std_time,
                            'min_time': min_time,
                            'median_time': np.median(times),
                        })
                        continue

                    # Choose appropriate format
                    if impl_name.startswith("coo"):
                        # Lazy initialization of COO format only when needed
                        if torch_sparse_coo is None:
                            torch_sparse_coo = scipy_sparse_to_torch_sparse_coo(sparse_matrix)
                            A_shape_coo, A_mode_indices_coo, A_values_coo = extract_coo_components(torch_sparse_coo)
                        A_shape = A_shape_coo
                        A_mode_indices = A_mode_indices_coo
                        A_values = A_values_coo
                    else:
                        # All CSR implementations
                        A_shape = A_shape_csr
                        A_mode_indices = A_mode_indices_csr
                        A_values = A_values_csr

                    times, error = benchmark_implementation(
                        impl_name, impl_func,
                        (result_shape, A_shape, A_mode_indices, A_values),
                        (B_shape, B_mode_indices, B_values),
                        result_shape,
                        warmup=3,
                        iterations=num_trials
                    )

                    if error:
                        print(f"ERROR: {error}")
                        continue

                    mean_time = np.mean(times)
                    std_time = np.std(times)
                    min_time = np.min(times)

                    print(f"Mean: {mean_time*1000:.3f}ms, Min: {min_time*1000:.3f}ms")

                    results.append({
                        'matrix_name': matrix_name,
                        'rows': sparse_matrix.shape[0],
                        'cols': sparse_matrix.shape[1],
                        'nnz': sparse_matrix.nnz,
                        'k': k,
                        'implementation': impl_name,
                        'mean_time': mean_time,
                        'std_time': std_time,
                        'min_time': min_time,
                        'median_time': np.median(times),
                    })

                # Save intermediate results
                df = pd.DataFrame(results)
                df.to_csv(csv_filename, index=False)

        except Exception as e:
            print(f"Error processing matrix {matrix_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final save and summary
    df = pd.DataFrame(results)
    df.to_csv(csv_filename, index=False)
    print(f"\n{'='*80}")
    print(f"Benchmarking complete! Results saved to: {csv_filename}")
    print(f"{'='*80}")

    return df, csv_filename


def plot_results(df, output_filename=None):
    """Generate visualizations of benchmark results."""
    # Handle empty dataframe
    if df.empty:
        print("Warning: No data to plot (empty dataframe)")
        return

    if output_filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"spmm_implementation_benchmark_{timestamp}.pdf"

    sns.set_style("whitegrid")

    # Create figure with subplots
    num_matrices = df['matrix_name'].nunique()
    num_k_values = df['k'].nunique()

    fig, axes = plt.subplots(num_k_values, 1, figsize=(14, 6 * num_k_values))
    if num_k_values == 1:
        axes = [axes]

    for idx, k in enumerate(sorted(df['k'].unique())):
        ax = axes[idx]
        df_k = df[df['k'] == k]

        # Pivot for heatmap: implementations vs matrices
        pivot_data = df_k.pivot(index='implementation', columns='matrix_name', values='mean_time')

        # Create bar plot
        pivot_data_t = pivot_data.T
        pivot_data_t.plot(kind='bar', ax=ax, width=0.8)

        ax.set_title(f'SPMM Implementation Performance (k={k})', fontsize=16, fontweight='bold')
        ax.set_xlabel('Matrix', fontsize=14)
        ax.set_ylabel('Mean Time (seconds)', fontsize=14)
        ax.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.set_yscale('log')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {output_filename}")
    plt.close()

    # Create speedup comparison (relative to untiled)
    speedup_filename = output_filename.replace('.pdf', '_speedup.pdf')
    fig, axes = plt.subplots(num_k_values, 1, figsize=(14, 6 * num_k_values))
    if num_k_values == 1:
        axes = [axes]

    for idx, k in enumerate(sorted(df['k'].unique())):
        ax = axes[idx]
        df_k = df[df['k'] == k]

        # Calculate speedup relative to untiled
        baseline_times = df_k[df_k['implementation'] == 'csr_untiled'].set_index('matrix_name')['mean_time']

        speedup_data = []
        for impl in df_k['implementation'].unique():
            if impl == 'csr_untiled':
                continue
            impl_times = df_k[df_k['implementation'] == impl].set_index('matrix_name')['mean_time']
            speedups = baseline_times / impl_times
            for matrix_name, speedup in speedups.items():
                speedup_data.append({
                    'matrix_name': matrix_name,
                    'implementation': impl,
                    'speedup': speedup
                })

        speedup_df = pd.DataFrame(speedup_data)
        speedup_pivot = speedup_df.pivot(index='implementation', columns='matrix_name', values='speedup')

        speedup_pivot.T.plot(kind='bar', ax=ax, width=0.8)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (untiled)')

        ax.set_title(f'Speedup vs Untiled Implementation (k={k})', fontsize=16, fontweight='bold')
        ax.set_xlabel('Matrix', fontsize=14)
        ax.set_ylabel('Speedup', fontsize=14)
        ax.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(speedup_filename, bbox_inches='tight', dpi=300)
    print(f"Speedup plot saved to: {speedup_filename}")
    plt.close()


def print_summary_table(df):
    """Print a summary table showing the best implementation for each matrix."""
    # Handle empty dataframe
    if df.empty:
        print("\nNo results to summarize (empty dataframe)")
        return

    print("\n" + "="*100)
    print("SUMMARY: Best Implementation for Each Matrix")
    print("="*100)

    for k in sorted(df['k'].unique()):
        print(f"\nk = {k}:")
        print("-" * 100)
        df_k = df[df['k'] == k]

        for matrix_name in sorted(df_k['matrix_name'].unique()):
            df_matrix = df_k[df_k['matrix_name'] == matrix_name]
            best_impl = df_matrix.loc[df_matrix['mean_time'].idxmin()]

            # Use PyTorch as baseline if available, otherwise csr_untiled
            pytorch_baseline = df_matrix[df_matrix['implementation'] == 'pytorch']['mean_time'].values
            csr_baseline = df_matrix[df_matrix['implementation'] == 'csr_untiled']['mean_time'].values

            if len(pytorch_baseline) > 0:
                baseline = pytorch_baseline[0]
                baseline_name = "PyTorch"
            elif len(csr_baseline) > 0:
                baseline = csr_baseline[0]
                baseline_name = "untiled"
            else:
                baseline = best_impl['mean_time']
                baseline_name = "N/A"

            speedup = baseline / best_impl['mean_time']

            print(f"{matrix_name:20s} | Best: {best_impl['implementation']:20s} | "
                  f"Time: {best_impl['mean_time']*1000:8.3f}ms | Speedup vs {baseline_name}: {speedup:5.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark different SPMM implementation speeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark on default matrices with default k values
  python benchmark_spmm_implementations.py

  # Run benchmark on specific matrices
  python benchmark_spmm_implementations.py --matrices cage4 bcsstk14 consph

  # Test with specific k values
  python benchmark_spmm_implementations.py --k-values 64 128 256

  # Only generate plots from existing CSV
  python benchmark_spmm_implementations.py --plot-only --csv-file benchmark_results.csv
        """
    )

    parser.add_argument('--matrices', nargs='+', default=None,
                        help='Specific matrix names to benchmark')
    parser.add_argument('--small-only', action='store_true',
                        help='Only benchmark small matrices (< 1000 rows) for quick testing')
    parser.add_argument('--k-values', nargs='+', type=int, default=[1, 16, 32, 64, 128, 256, 512],
                        help='K values (dense matrix columns) to test')
    parser.add_argument('--num-trials', type=int, default=10,
                        help='Number of timing iterations per test')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip benchmark and only generate plots from existing CSV')
    parser.add_argument('--csv-file', type=str, default=None,
                        help='CSV file with benchmark results (for --plot-only mode)')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='Output filename for plots')

    args = parser.parse_args()

    if args.plot_only:
        if args.csv_file is None:
            print("Error: --csv-file required for --plot-only mode")
            return

        print(f"Loading results from: {args.csv_file}")
        df = pd.read_csv(args.csv_file)
        plot_results(df, args.output_plot)
        print_summary_table(df)
    else:
        df, csv_filename = run_benchmark(
            matrix_names=args.matrices,
            k_values=args.k_values,
            num_trials=args.num_trials,
            small_only=args.small_only
        )

        plot_results(df, args.output_plot)
        print_summary_table(df)


if __name__ == "__main__":
    main()
