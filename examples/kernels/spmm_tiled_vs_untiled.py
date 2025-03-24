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

import warnings

# Suppress specific PyTorch UserWarning about Sparse CSR tensor support
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Sparse CSR tensor support is in beta state.*",
)

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

def run_tiled_spmm(sparse_tensor, dense_tensor):
    """Run the tiled version of SpMM using the standard matmul function"""
    return scorch.matmul(sparse_tensor, dense_tensor)

def run_tiled_spmm_turbo(sparse_tensor, dense_tensor, tile_size=256):
    """Run the Turbo version of tiled SpMM with specified tile size"""
    a = scorch.STensor.from_torch(sparse_tensor)
    b = scorch.STensor.from_torch(dense_tensor)

    result_shape = (a.shape[0], b.shape[1])
    args = [result_shape]

    for tensor in [a, b]:
        args.append(tensor.shape)
        args.append(tensor.index.mode_indices)
        args.append(tensor.values)

    # Add the tile_size parameter
    args.append(tile_size)

    result_cpp = ops.spmm_csr_float(*args)

    result = scorch.STensor(
        shape=result_shape,
        index=scorch.TensorIndex(
            mode_indices=result_cpp.storage.index.mode_indices,
            tensor_format="dd",
        ),
        value=result_cpp.storage.value,
    )

    return result.to_torch()

def run_tiled_spmm_ultra(sparse_tensor, dense_tensor, tile_size=256):
    """Run the Ultra version of tiled SpMM with specified tile size"""
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

def generate_plots(results_df, figures_dir="benchmark_figures"):
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

    # Define colors to swap between implementations
    tiled_color = 'blue'    # Was previously blue (default first color)
    untiled_color = 'orange'    # Was previously orange (default second color)

    # Plot with swapped colors
    for implementation, color in zip(['Scorch Tiled', 'Scorch Untiled'], [tiled_color, untiled_color]):
        sub_df = results_df[results_df['Implementation'] == implementation]
        mean_runtime = sub_df.groupby('NNZ')['Runtime'].mean()
        std_runtime = sub_df.groupby('NNZ')['Runtime'].std()

        # Using scatter plot for data points with explicit color
        plt.scatter(mean_runtime.index, mean_runtime, label=implementation, s=2, alpha=0.7, color=color)

    plt.xlabel('Number of Non-Zeros (NNZ)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('SpMM Performance: Tiled vs Untiled')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"{figures_dir}/spmm_tiled_vs_untiled_benchmark_plot.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_tiled_vs_untiled_benchmark_plot.svg", bbox_inches='tight')
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
    plt.savefig(f"{figures_dir}/spmm_tiled_vs_untiled_barplot.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_tiled_vs_untiled_barplot.svg", bbox_inches='tight')
    plt.close()

    # Figure 3: Calculate and plot speedup
    baseline_runtime = summary_df[summary_df['Implementation'] == 'Scorch Untiled']['Runtime'].values[0]
    tiled_runtime = summary_df[summary_df['Implementation'] == 'Scorch Tiled']['Runtime'].values[0]
    speedup = baseline_runtime / tiled_runtime

    plt.figure(figsize=(8, 6))
    speedup_data = pd.DataFrame({
        'Implementation': ['Scorch Tiled'],
        'Speedup': [speedup]
    })
    speedup_plot = sns.barplot(x='Implementation', y='Speedup', data=speedup_data)
    for i, bar in enumerate(speedup_plot.patches):
        height = bar.get_height()
        speedup_plot.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}x',
                    ha='center', va='bottom', rotation=0, fontsize=14)

    plt.title('Speedup of Tiled over Untiled Implementation')
    plt.xlabel('Implementation')
    plt.ylabel('Speedup Factor')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/spmm_tiled_speedup.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_tiled_speedup.svg", bbox_inches='tight')
    plt.close()

    # Figure 4: Performance by matrix category
    plt.figure(figsize=(14, 8))
    category_df = results_df.groupby(['Category', 'Implementation'])['Runtime'].mean().reset_index()
    category_pivot = category_df.pivot(index='Category', columns='Implementation', values='Runtime')
    category_pivot['Speedup'] = category_pivot['Scorch Untiled'] / category_pivot['Scorch Tiled']
    category_pivot = category_pivot.sort_values('Speedup', ascending=False)

    ax = sns.heatmap(category_pivot[['Scorch Tiled', 'Scorch Untiled']], annot=True, fmt=".5f", cmap="YlGnBu",
                cbar_kws={'label': 'Runtime (seconds)'})
    plt.title('Performance by Matrix Category')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/spmm_category_heatmap.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_category_heatmap.svg", bbox_inches='tight')
    plt.close()

    # Figure 5: Speedup by matrix category
    plt.figure(figsize=(12, 8))
    speedup_by_category = category_pivot.reset_index()[['Category', 'Speedup']]

    # Create a colormap based on speedup values
    cmap = plt.cm.viridis
    norm = plt.Normalize(speedup_by_category['Speedup'].min(), speedup_by_category['Speedup'].max())
    colors = [cmap(norm(value)) for value in speedup_by_category['Speedup']]

    # Create custom barplot with colors
    speedup_plot = sns.barplot(x='Category', y='Speedup', data=speedup_by_category, palette=colors)

    # Add value annotations with larger font
    for i, bar in enumerate(speedup_plot.patches):
        height = bar.get_height()
        if not np.isnan(height):
            speedup_plot.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.2f}x',
                        ha='center', va='bottom', rotation=0, fontsize=16,
                        fontweight='bold', color='black')

    # Add some extra space at the top of the plot for labels
    plt.ylim(0, speedup_by_category['Speedup'].max() * 1.15)

    plt.title('Speedup by Matrix Category', fontsize=24)
    plt.xlabel('Matrix Category', fontsize=20)
    plt.ylabel('Speedup Factor (Tiled vs Untiled)', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)

    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/spmm_category_speedup.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_category_speedup.svg", bbox_inches='tight')
    plt.close()

    # Figure 6: Runtime vs Matrix Size scatter plot
    plt.figure(figsize=(12, 8))
    for implementation in ['Scorch Tiled', 'Scorch Untiled']:
        sub_df = results_df[results_df['Implementation'] == implementation]
        plt.scatter(sub_df['Rows'], sub_df['Runtime'], label=implementation, alpha=0.7, s=10)

    plt.xlabel('Matrix Rows')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Matrix Size')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/spmm_size_vs_runtime.pdf", bbox_inches='tight')
    plt.savefig(f"{figures_dir}/spmm_size_vs_runtime.svg", bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average Runtime - Tiled: {tiled_runtime:.6f} seconds")
    print(f"Average Runtime - Untiled: {baseline_runtime:.6f} seconds")
    print(f"Overall Speedup: {speedup:.2f}x")

    print("\nTop 5 Matrix Categories with Highest Speedup:")
    print(speedup_by_category.sort_values('Speedup', ascending=False).head(5))

    print(f"\nAnalysis complete. All plots have been saved to the '{figures_dir}' directory in both PDF and SVG formats.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Benchmark SpMM implementations and generate plots')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate plots from existing CSV data without running benchmarks')
    parser.add_argument('--csv-file', type=str, default='spmm_tiled_vs_untiled_benchmark_results.csv',
                        help='Path to the CSV file with benchmark results (default: spmm_tiled_vs_untiled_benchmark_results.csv)')
    parser.add_argument('--output-dir', type=str, default='benchmark_figures',
                        help='Directory to save the generated plots (default: benchmark_figures)')

    args = parser.parse_args()

    if args.plot_only:
        print(f"Plot-only mode: Loading data from {args.csv_file}")
        try:
            results_df = pd.read_csv(args.csv_file)
            print(f"Loaded data with {len(results_df)} records")
            generate_plots(results_df, figures_dir=args.output_dir)
        except FileNotFoundError:
            print(f"Error: CSV file '{args.csv_file}' not found. Please provide a valid file path.")
            return
    else:
        # Set random seed for reproducibility
        np.random.seed(15)
        torch.manual_seed(15)

        # Try to get a diverse set of matrices
        matrices = ssgetpy.search(limit=5000)
        matrices = [matrix for matrix in matrices if max(matrix.rows, matrix.cols) < 700000]

        # Remove specific problematic indices
        idxs_to_remove = [961, 962]
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

        for matrix in tqdm(test_matrices, desc="Benchmarking Matrices"):
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

                for implementation in ["Tiled", "Untiled"]:
                    print(f"Benchmarking Scorch {implementation}...")

                    # Warm-up run (not timed)
                    if implementation == "Tiled":
                        _ = run_tiled_spmm_ultra(torch_sparse_matrix, dense_matrix, tile_size=256)
                    elif implementation == "Untiled":
                        _ = run_untiled_spmm(torch_sparse_matrix, dense_matrix)

                    # Timed runs
                    for _ in range(10):
                        if implementation == "Tiled":
                            start_time = time.time()
                            result = run_tiled_spmm_ultra(torch_sparse_matrix, dense_matrix, tile_size=256)
                        elif implementation == "Untiled":
                            start_time = time.time()
                            result = run_untiled_spmm(torch_sparse_matrix, dense_matrix)

                        end_time = time.time()

                        results.append({
                            'Implementation': f"Scorch {implementation}",
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

            except Exception as e:
                print(f"Error processing matrix {matrix.name} in group {matrix.group}: {e}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(args.csv_file, index=False)
        print(f"Benchmarking complete. Results saved to '{args.csv_file}'.")

        # Generate plots from the benchmark results
        generate_plots(results_df, figures_dir=args.output_dir)

if __name__ == "__main__":
    main()
