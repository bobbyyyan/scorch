import ssgetpy
from scipy.io import mmread
from pathlib import Path
import torch
import numpy as np
import time
import pandas as pd
import scorch
from tqdm import tqdm
import scorch._C.ops as ops
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import datetime
import glob

import warnings

# Suppress specific PyTorch UserWarning about Sparse CSR tensor support
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Sparse CSR tensor support is in beta state.*",
)

def get_latest_csv():
    """Find the most recent benchmark CSV file based on the timestamp in the filename."""
    csv_files = glob.glob("spmm_tiling_comparison_*.csv")
    if not csv_files:
        return None

    # Sort files by creation time (newest first)
    latest_file = max(csv_files, key=os.path.getctime)
    return latest_file

def scipy_sparse_to_torch_sparse(matrix, format='csr'):
    if format == 'coo':
        matrix = matrix.tocoo()
        indices = np.vstack((matrix.row, matrix.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(matrix.data)
        shape = matrix.shape
        return torch.sparse_coo_tensor(i, v, torch.Size(shape))
    elif format == 'csr':
        matrix = matrix.tocsr()
        crow_indices = torch.LongTensor(matrix.indptr)
        col_indices = torch.LongTensor(matrix.indices)
        values = torch.FloatTensor(matrix.data)
        shape = matrix.shape
        return torch.sparse_csr_tensor(crow_indices, col_indices, values, torch.Size(shape))
    else:
        raise ValueError("Unsupported format: only 'coo' and 'csr' are supported")

def run_k_tiled_spmm_ultra(sparse_tensor, dense_tensor, tile_size=256):
    """Run the Ultra version of k-tiled SpMM with specified tile size"""
    a = scorch.STensor.from_torch(sparse_tensor)
    b = scorch.STensor.from_torch(dense_tensor)

    result_shape = (a.shape[0], b.shape[1])
    args = [result_shape]

    for tensor in [a, b]:
        args.append(tensor.shape)
        args.append(tensor.index.mode_indices)
        args.append(tensor.values)

    args.append(tile_size)

    result_cpp = ops.spmm_csr_float_ultra(*args)

    result = scorch.STensor(
        shape=result_shape,
        index=scorch.TensorIndex(
            mode_indices=result_cpp.storage.index.mode_indices,
            tensor_format="dd",
        ),
        value=result_cpp.storage.value,
    )

    return result.to_torch()

def run_i_k_tiled_spmm(sparse_tensor, dense_tensor, i_tile_size=16, k_tile_size=32):
    """Run the version of SpMM with both i and k tiling"""
    a = scorch.STensor.from_torch(sparse_tensor)
    b = scorch.STensor.from_torch(dense_tensor)

    result_shape = (a.shape[0], b.shape[1])
    args = [result_shape]

    for tensor in [a, b]:
        args.append(tensor.shape)
        args.append(tensor.index.mode_indices)
        args.append(tensor.values)

    # Add the tile size parameters
    args.append(i_tile_size)
    args.append(k_tile_size)

    result_cpp = ops.spmm_csr_float_tiled_i_k(*args)

    result = scorch.STensor(
        shape=result_shape,
        index=scorch.TensorIndex(
            mode_indices=result_cpp.storage.index.mode_indices,
            tensor_format="dd",
        ),
        value=result_cpp.storage.value,
    )

    return result.to_torch()

def run_untiled_spmm(sparse_tensor, dense_tensor):
    """Run the untiled version of SpMM by calling the C++ function directly"""
    a = scorch.STensor.from_torch(sparse_tensor)
    b = scorch.STensor.from_torch(dense_tensor)

    result_shape = (a.shape[0], b.shape[1])
    args = [result_shape]

    for tensor in [a, b]:
        args.append(tensor.shape)
        args.append(tensor.index.mode_indices)
        args.append(tensor.values)

    result_cpp = ops.spmm_csr_float_untiled(*args)

    result = scorch.STensor(
        shape=result_shape,
        index=scorch.TensorIndex(
            mode_indices=result_cpp.storage.index.mode_indices,
            tensor_format="dd",
        ),
        value=result_cpp.storage.value,
    )

    return result.to_torch()

def categorize_matrix(matrix, nnz):
    """Categorize matrix based on its characteristics"""
    density = nnz / (matrix.shape[0] * matrix.shape[1])

    if matrix.shape[0] < 1000:
        size_category = "Small"
    elif matrix.shape[0] < 10000:
        size_category = "Medium"
    else:
        size_category = "Large"

    if density < 0.0001:
        density_category = "Very Sparse"
    elif density < 0.001:
        density_category = "Sparse"
    elif density < 0.01:
        density_category = "Moderately Sparse"
    else:
        density_category = "Dense"

    return f"{size_category} {density_category}"

def generate_plots(results_df, figures_dir="benchmark_figures_tiling"):
    """Generate all benchmark plots from the results DataFrame"""
    # Create a dedicated folder for benchmark figures
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        print(f"Created directory '{figures_dir}' for benchmark figures")

    # Plot settings
    sns.set(style="white", context="talk")

    # Figure 1: Basic scatter plot
    plt.figure(figsize=(12, 6))

    plt.rcParams.update({
        'grid.linestyle': ' ',
        'font.size': 17,
        'axes.labelsize': 22,
        'axes.titlesize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 18,
        'legend.fontsize': 22,
        'legend.title_fontsize': 24,
        'legend.markerscale': 5,
    })

    # Define colors for implementations
    colors = {
        'Untiled': 'red',
        'K-Tiled': 'green',
        'I-K-Tiled': 'blue'
    }

    # Plot all implementations
    for implementation, color in colors.items():
        sub_df = results_df[results_df['Implementation'] == implementation]
        mean_runtime = sub_df.groupby('NNZ')['Runtime'].mean()
        std_runtime = sub_df.groupby('NNZ')['Runtime'].std()

        # Using scatter plot for data points with explicit color
        plt.scatter(mean_runtime.index, mean_runtime, label=implementation, s=2, alpha=0.7, color=color)

    plt.xlabel('Number of Non-Zeros (NNZ)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('SpMM Performance: Tiling Comparison')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"{figures_dir}/spmm_tiling_comparison_plot.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_tiling_comparison_plot.svg", bbox_inches='tight')
    plt.close()

    # Figure 2: Bar plot for overall performance comparison
    plt.figure(figsize=(10, 6))
    summary_df = results_df.groupby('Implementation')['Runtime'].mean().reset_index()
    bar_plot = sns.barplot(x='Implementation', y='Runtime', data=summary_df)

    # Add value labels on bars
    for i, bar in enumerate(bar_plot.patches):
        height = bar.get_height()
        if not np.isnan(height):
            bar_plot.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                        f'{height:.5f}',
                        ha='center', va='bottom', rotation=0, fontsize=12)

    plt.title('Overall SpMM Performance Comparison')
    plt.xlabel('Implementation')
    plt.ylabel('Average Runtime (seconds)')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/spmm_tiling_barplot.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_tiling_barplot.svg", bbox_inches='tight')
    plt.close()

    # Figure 3: Calculate and plot speedup compared to untiled
    baseline_runtime = summary_df[summary_df['Implementation'] == 'Untiled']['Runtime'].values[0]

    speedup_data = []
    for impl in ['K-Tiled', 'I-K-Tiled']:
        impl_runtime = summary_df[summary_df['Implementation'] == impl]['Runtime'].values[0]
        speedup = baseline_runtime / impl_runtime
        speedup_data.append({
            'Implementation': impl,
            'Speedup': speedup
        })

    speedup_df = pd.DataFrame(speedup_data)

    plt.figure(figsize=(8, 6))
    speedup_plot = sns.barplot(x='Implementation', y='Speedup', data=speedup_df)
    for i, bar in enumerate(speedup_plot.patches):
        height = bar.get_height()
        speedup_plot.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}x',
                    ha='center', va='bottom', rotation=0, fontsize=14)

    plt.title('Speedup over Untiled Implementation')
    plt.xlabel('Implementation')
    plt.ylabel('Speedup Factor')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/spmm_tiling_speedup.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_tiling_speedup.svg", bbox_inches='tight')
    plt.close()

    # Figure 4: Performance by matrix category
    plt.figure(figsize=(14, 8))
    category_df = results_df.groupby(['Category', 'Implementation'])['Runtime'].mean().reset_index()
    category_pivot = category_df.pivot(index='Category', columns='Implementation', values='Runtime')

    # Calculate speedups
    category_pivot['K-Tiled Speedup'] = category_pivot['Untiled'] / category_pivot['K-Tiled']
    category_pivot['I-K-Tiled Speedup'] = category_pivot['Untiled'] / category_pivot['I-K-Tiled']

    # Sort by I-K-Tiled speedup
    category_pivot = category_pivot.sort_values('I-K-Tiled Speedup', ascending=False)

    ax = sns.heatmap(category_pivot[['Untiled', 'K-Tiled', 'I-K-Tiled']], annot=True, fmt=".5f", cmap="YlGnBu",
                cbar_kws={'label': 'Runtime (seconds)'})
    plt.title('Performance by Matrix Category')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/spmm_category_heatmap.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_category_heatmap.svg", bbox_inches='tight')
    plt.close()

    # Figure 5: Speedup by matrix category for both tiled implementations
    plt.figure(figsize=(14, 8))

    # Prepare data for grouped bar chart
    k_tiled_speedup = category_pivot.reset_index()[['Category', 'K-Tiled Speedup']]
    k_tiled_speedup['Implementation'] = 'K-Tiled'
    k_tiled_speedup = k_tiled_speedup.rename(columns={'K-Tiled Speedup': 'Speedup'})

    ik_tiled_speedup = category_pivot.reset_index()[['Category', 'I-K-Tiled Speedup']]
    ik_tiled_speedup['Implementation'] = 'I-K-Tiled'
    ik_tiled_speedup = ik_tiled_speedup.rename(columns={'I-K-Tiled Speedup': 'Speedup'})

    speedup_by_category = pd.concat([k_tiled_speedup, ik_tiled_speedup])

    # Create grouped barplot
    sns.barplot(x='Category', y='Speedup', hue='Implementation', data=speedup_by_category)

    plt.title('Speedup by Matrix Category', fontsize=24)
    plt.xlabel('Matrix Category', fontsize=20)
    plt.ylabel('Speedup Factor over Untiled', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Implementation')

    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/spmm_category_tiling_speedup.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_category_tiling_speedup.svg", bbox_inches='tight')
    plt.close()

    # Figure 6: Runtime vs Matrix Size scatter plot
    plt.figure(figsize=(12, 8))
    for implementation in ['Untiled', 'K-Tiled', 'I-K-Tiled']:
        sub_df = results_df[results_df['Implementation'] == implementation]
        plt.scatter(sub_df['Rows'], sub_df['Runtime'], label=implementation, alpha=0.7, s=10)

    plt.xlabel('Matrix Rows')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Matrix Size')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/spmm_size_vs_runtime_tiling.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_size_vs_runtime_tiling.svg", bbox_inches='tight')
    plt.close()

    # Figure 7: Direct comparison of K-Tiled vs I-K-Tiled
    plt.figure(figsize=(10, 6))
    ik_vs_k_speedup = summary_df[summary_df['Implementation'] == 'K-Tiled']['Runtime'].values[0] / \
                   summary_df[summary_df['Implementation'] == 'I-K-Tiled']['Runtime'].values[0]

    comparison_data = pd.DataFrame({
        'Comparison': ['I-K-Tiled vs K-Tiled'],
        'Speedup': [ik_vs_k_speedup]
    })

    comparison_plot = sns.barplot(x='Comparison', y='Speedup', data=comparison_data)
    for i, bar in enumerate(comparison_plot.patches):
        height = bar.get_height()
        comparison_plot.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}x',
                    ha='center', va='bottom', rotation=0, fontsize=14)

    plt.title('Speedup of I-K-Tiled over K-Tiled Implementation')
    plt.xlabel('Comparison')
    plt.ylabel('Speedup Factor')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/spmm_ik_vs_k_speedup.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_ik_vs_k_speedup.svg", bbox_inches='tight')
    plt.close()

    # Print summary statistics
    untiled_runtime = summary_df[summary_df['Implementation'] == 'Untiled']['Runtime'].values[0]
    k_tiled_runtime = summary_df[summary_df['Implementation'] == 'K-Tiled']['Runtime'].values[0]
    ik_tiled_runtime = summary_df[summary_df['Implementation'] == 'I-K-Tiled']['Runtime'].values[0]

    print("\nSummary Statistics:")
    print(f"Average Runtime - Untiled: {untiled_runtime:.6f} seconds")
    print(f"Average Runtime - K-Tiled: {k_tiled_runtime:.6f} seconds")
    print(f"Average Runtime - I-K-Tiled: {ik_tiled_runtime:.6f} seconds")
    print(f"K-Tiled Speedup over Untiled: {untiled_runtime / k_tiled_runtime:.2f}x")
    print(f"I-K-Tiled Speedup over Untiled: {untiled_runtime / ik_tiled_runtime:.2f}x")
    print(f"I-K-Tiled Speedup over K-Tiled: {k_tiled_runtime / ik_tiled_runtime:.2f}x")

    print("\nTop 5 Matrix Categories with Highest I-K-Tiled Speedup:")
    print(category_pivot.reset_index()[['Category', 'I-K-Tiled Speedup']].sort_values('I-K-Tiled Speedup', ascending=False).head(5))

    print(f"\nAnalysis complete. All plots have been saved to the '{figures_dir}' directory in both PDF and SVG formats.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Benchmark SpMM tiling implementations and generate plots')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate plots from existing CSV data without running benchmarks')
    parser.add_argument('--csv-file', type=str, default=None,
                        help='Path to the CSV file with benchmark results (default: use latest benchmark CSV)')
    parser.add_argument('--output-dir', type=str, default='benchmark_figures_tiling',
                        help='Directory to save the generated plots (default: benchmark_figures_tiling)')

    args = parser.parse_args()

    if args.plot_only:
        # If no specific CSV file is provided, find the latest one
        if args.csv_file is None:
            csv_file = get_latest_csv()
            if csv_file is None:
                print("Error: No benchmark CSV files found. Run benchmark first.")
                return
            print(f"Plot-only mode: Using latest CSV file: {csv_file}")
        else:
            csv_file = args.csv_file
            print(f"Plot-only mode: Using specified CSV file: {csv_file}")

        try:
            results_df = pd.read_csv(csv_file)
            print(f"Loaded data with {len(results_df)} records")
            generate_plots(results_df, figures_dir=args.output_dir)
        except FileNotFoundError:
            print(f"Error: CSV file '{csv_file}' not found. Please provide a valid file path.")
            return
    else:
        # Set random seed for reproducibility
        np.random.seed(15)
        torch.manual_seed(15)

        # Generate timestamp for CSV filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_filename = f"spmm_tiling_comparison_{timestamp}.csv"

        # Try to get a diverse set of matrices
        matrices = ssgetpy.search(limit=5000)
        matrices = [matrix for matrix in matrices if max(matrix.rows, matrix.cols) < 7000000]

        # Remove specific problematic indices
        idxs_to_remove = [961, 962, 2775, 2743]
        matrices = [m for m in matrices if m.id not in idxs_to_remove]

        # Categorize all matrices instead of selecting subsets
        test_matrices = matrices
        matrix_categories = {}

        for matrix in test_matrices:
            # Calculate density
            density = matrix.nnz / (matrix.rows * matrix.cols)

            # Categorize by size
            if matrix.rows < 1000:
                size_category = "Small"
            elif matrix.rows < 10000:
                size_category = "Medium"
            else:
                size_category = "Large"

            # Categorize by density
            if density < 0.0001:
                density_category = "Very Sparse"
            elif density < 0.001:
                density_category = "Sparse"
            elif density < 0.01:
                density_category = "Moderately Sparse"
            else:
                density_category = "Dense"

            # Store the category for this matrix
            matrix_categories[matrix.id] = f"{size_category} {density_category}"

        print(f"Testing with {len(test_matrices)} matrices")

        results = []
        save_interval = 10  # Save after every 10 matrices

        for i, matrix in enumerate(tqdm(test_matrices, desc="Benchmarking Matrices")):
            try:
                print(f"Processing matrix {matrix.id} {matrix.name} in group {matrix.group} with {matrix.nnz} NNZ...")

                matrix_path = Path(f"~/.ssgetpy/MM/{matrix.group}/{matrix.name}/{matrix.name}.mtx").expanduser()
                sparse_matrix = mmread(matrix_path.resolve())
                print(f"Matrix shape: {sparse_matrix.shape}")

                matrix_format = 'csr'
                torch_sparse_matrix = scipy_sparse_to_torch_sparse(sparse_matrix, format=matrix_format)
                dense_matrix = torch.rand((torch_sparse_matrix.shape[1], 100), dtype=torch.float32)
                nnz = sparse_matrix.nnz

                # Use the pre-computed category for this matrix
                matrix_category = matrix_categories[matrix.id]

                for implementation in ["Untiled", "K-Tiled", "I-K-Tiled"]:
                    print(f"Benchmarking {implementation}...")

                    # Warm-up run (not timed)
                    if implementation == "Untiled":
                        _ = run_untiled_spmm(torch_sparse_matrix, dense_matrix)
                    elif implementation == "K-Tiled":
                        _ = run_k_tiled_spmm_ultra(torch_sparse_matrix, dense_matrix, tile_size=256)
                    elif implementation == "I-K-Tiled":
                        _ = run_i_k_tiled_spmm(torch_sparse_matrix, dense_matrix, i_tile_size=16, k_tile_size=32)

                    # Timed runs
                    for _ in range(10):
                        if implementation == "Untiled":
                            start_time = time.time()
                            result = run_untiled_spmm(torch_sparse_matrix, dense_matrix)
                        elif implementation == "K-Tiled":
                            start_time = time.time()
                            result = run_k_tiled_spmm_ultra(torch_sparse_matrix, dense_matrix, tile_size=256)
                        elif implementation == "I-K-Tiled":
                            start_time = time.time()
                            result = run_i_k_tiled_spmm(torch_sparse_matrix, dense_matrix, i_tile_size=16, k_tile_size=32)

                        end_time = time.time()

                        results.append({
                            'Implementation': implementation,
                            'Group': matrix.group,
                            'Name': matrix.name,
                            'Category': matrix_category,
                            'Matrix ID': matrix.id,
                            'Rows': torch_sparse_matrix.shape[0],
                            'Columns': torch_sparse_matrix.shape[1],
                            'NNZ': nnz,
                            'Format': matrix_format,
                            'Runtime': end_time - start_time
                        })

                # Periodically save results to avoid data loss
                if (i + 1) % save_interval == 0 or i == len(test_matrices) - 1:
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(csv_filename, index=False)
                    print(f"Saving interim results to '{csv_filename}' after processing {i+1}/{len(test_matrices)} matrices.")

            except Exception as e:
                print(f"Error processing matrix {matrix.name} in group {matrix.group}: {e}")

                # Save on exception to preserve data collected so far
                if results:
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(csv_filename, index=False)
                    print(f"Saved partial results to '{csv_filename}' after error.")

        results_df = pd.DataFrame(results)
        results_df.to_csv(csv_filename, index=False)
        print(f"Benchmarking complete. Results saved to '{csv_filename}'.")

        # Generate plots from the benchmark results
        generate_plots(results_df, figures_dir=args.output_dir)

if __name__ == "__main__":
    main()
