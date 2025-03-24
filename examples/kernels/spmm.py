import ssgetpy
from scipy.io import mmread
from pathlib import Path
import torch
import numpy as np
import time
import pandas as pd
import scorch
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import datetime

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

def get_latest_csv():
    """Find the most recent benchmark CSV file based on the timestamp in the filename."""
    csv_files = glob.glob("spmm_benchmark_*.csv")
    if not csv_files:
        return None

    # Sort files by creation time (newest first)
    latest_file = max(csv_files, key=os.path.getctime)
    return latest_file

def run_benchmark():
    # Set random seed for reproducibility
    np.random.seed(15)
    torch.manual_seed(15)

    # Generate timestamp once at the beginning
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"spmm_benchmark_{timestamp}.csv"

    matrices = ssgetpy.search(limit=5000)

    # Remove specific problematic indices
    # idxs_to_remove = [1905, 2459]
    idxs_to_remove = [2511]
    matrices = [m for m in matrices if m.id not in idxs_to_remove]

    results = []
    save_interval = 10  # Save after every 10 matrices

    for i, matrix in enumerate(tqdm(matrices, desc="Benchmarking Matrices")):
        try:
            print(f"Processing matrix {matrix.id} {matrix.name} in group {matrix.group} with {matrix.nnz} NNZ...")
            # sparse_matrix_mm = matrix.download(format='MM', extract=True)

            matrix_path = Path(f"~/.ssgetpy/MM/{matrix.group}/{matrix.name}/{matrix.name}.mtx").expanduser()
            sparse_matrix = mmread(matrix_path.resolve())
            print(f"Matrix shape: {sparse_matrix.shape}")

            matrix_format = 'csr'
            torch_sparse_matrix = scipy_sparse_to_torch_sparse(sparse_matrix, format=matrix_format)
            dense_matrix = torch.rand((torch_sparse_matrix.shape[1], 100), dtype=torch.float32)
            nnz = sparse_matrix.nnz

            for framework in ["PyTorch", "Scorch"]:
                print(f"Benchmarking {framework}...")
                for _ in range(10):
                    if framework == "PyTorch":
                        start_time = time.time()
                        result = torch.sparse.mm(torch_sparse_matrix, dense_matrix)
                    elif framework == "Scorch":
                        start_time = time.time()
                        result = scorch.matmul(torch_sparse_matrix, dense_matrix)
                    end_time = time.time()

                    results.append({
                        'Framework': framework,
                        'Group': matrix.group,
                        'Name': matrix.name,
                        'Matrix ID': matrix.id,
                        'Rows': torch_sparse_matrix.shape[0],
                        'Columns': torch_sparse_matrix.shape[1],
                        'NNZ': nnz,
                        'Format': matrix_format,
                        'Runtime': end_time - start_time
                    })

            # Periodically save results to avoid data loss
            if (i + 1) % save_interval == 0 or i == len(matrices) - 1:
                results_df = pd.DataFrame(results)
                results_df.to_csv(csv_filename, index=False)
                print(f"Saving interim results to '{csv_filename}' after processing {i+1}/{len(matrices)} matrices.")

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
    return results_df, csv_filename

def plot_results(results_df, output_filename="spmm_benchmark_plot.pdf"):
    # Plot settings
    sns.set(style="white", context="talk")

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.rcParams.update({
        'grid.linestyle': ' ',
        'font.size': 17,
        'axes.labelsize': 22,
        'axes.titlesize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 18,
        # Legend settings
        'legend.fontsize': 22,
        'legend.title_fontsize': 24,
        # Legend dot size
        'legend.markerscale': 5,
    })

    for framework in ['PyTorch', 'Scorch']:
        sub_df = results_df[results_df['Framework'] == framework]
        mean_runtime = sub_df.groupby('NNZ')['Runtime'].mean()
        std_runtime = sub_df.groupby('NNZ')['Runtime'].std()

        # Using scatter plot for data points, adjust size using 's' and transparency using 'alpha'
        plt.scatter(mean_runtime.index, mean_runtime, label=f'{framework}', s=2, alpha=0.7)  # smaller size
        # Optional: Adjust error bars
        # plt.errorbar(mean_runtime.index, mean_runtime, yerr=std_runtime, fmt='o', alpha=0.3, capsize=3, markersize=2)

    plt.xlabel('Number of Non-Zeros (NNZ)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Sparse Matrix Multiplication (SpMM) Performance')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Sparse Matrix Multiplication Benchmark')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip benchmark and only generate plots from existing CSV')
    parser.add_argument('--csv-file', type=str, default=None,
                        help='CSV file with benchmark results (default: use latest benchmark CSV)')
    parser.add_argument('--output-plot', type=str, default=None,
                        help='Output filename for the plot (default: derived from CSV filename)')

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

        # Determine output plot filename if not specified
        if args.output_plot is None:
            # Create plot filename based on CSV filename
            base_name = os.path.splitext(csv_file)[0]
            output_plot = f"{base_name}.pdf"
        else:
            output_plot = args.output_plot

        try:
            results_df = pd.read_csv(csv_file)
            plot_results(results_df, output_plot)
        except FileNotFoundError:
            print(f"Error: CSV file '{csv_file}' not found.")
            return
    else:
        # Run benchmark and generate plots
        results_df, csv_filename = run_benchmark()

        # Determine output plot filename if not specified
        if args.output_plot is None:
            # Create plot filename based on CSV filename
            base_name = os.path.splitext(csv_filename)[0]
            output_plot = f"{base_name}.pdf"
        else:
            output_plot = args.output_plot

        plot_results(results_df, output_plot)

if __name__ == "__main__":
    main()
