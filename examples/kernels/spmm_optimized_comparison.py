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

def run_original_spmm(sparse_tensor, dense_tensor, tile_size):
    """Run the original tiled SpMM implementation"""
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

def run_optimized_spmm(sparse_tensor, dense_tensor, tile_size):
    """Run the optimized SpMM implementation"""
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

    result_cpp = ops.spmm_csr_float_optimized(*args)

    result = scorch.STensor(
        shape=result_shape,
        index=scorch.TensorIndex(
            mode_indices=result_cpp.storage.index.mode_indices,
            tensor_format="dd",
        ),
        value=result_cpp.storage.value,
    )

    return result.to_torch()

def run_turbo_spmm(sparse_tensor, dense_tensor, tile_size):
    """Run the turbo-optimized SpMM implementation"""
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

    result_cpp = ops.spmm_csr_float_turbo(*args)

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
    """Run the untiled SpMM implementation as a baseline"""
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

# Set random seed for reproducibility
np.random.seed(15)
torch.manual_seed(15)

# Define the tile sizes to benchmark
tile_sizes = [32, 128, 512]

# Limit the number of matrices to test for quicker benchmarking
matrices = ssgetpy.search(limit=500)
matrices = [matrix for matrix in matrices if max(matrix.rows, matrix.cols) < 700000]

# Remove specific problematic indices if they exist in our set
idxs_to_remove = [961, 962]
matrices = [matrix for idx, matrix in enumerate(matrices) if idx not in idxs_to_remove]

results = []

for matrix in tqdm(matrices, desc="Benchmarking Matrices"):
    try:
        print(f"Processing matrix {matrix.id} {matrix.name} in group {matrix.group} with {matrix.nnz} NNZ...")

        matrix_path = Path(f"~/.ssgetpy/MM/{matrix.group}/{matrix.name}/{matrix.name}.mtx").expanduser()
        sparse_matrix = mmread(matrix_path.resolve())
        print(f"Matrix shape: {sparse_matrix.shape}")

        matrix_format = 'csr'
        torch_sparse_matrix = scipy_sparse_to_torch_sparse(sparse_matrix, format=matrix_format)
        dense_matrix = torch.rand((torch_sparse_matrix.shape[1], 100), dtype=torch.float32)
        nnz = sparse_matrix.nnz

        # Test untiled implementation first
        print("Benchmarking untiled implementation...")
        for _ in range(3):  # Fewer runs for quicker benchmarking
            start_time = time.time()
            result = run_untiled_spmm(torch_sparse_matrix, dense_matrix)
            end_time = time.time()

            results.append({
                'Implementation': 'Untiled',
                'Tile Size': 'N/A',
                'Group': matrix.group,
                'Name': matrix.name,
                'Matrix ID': matrix.id,
                'Rows': torch_sparse_matrix.shape[0],
                'Columns': torch_sparse_matrix.shape[1],
                'NNZ': nnz,
                'Format': matrix_format,
                'Runtime': end_time - start_time
            })

        # Test original and optimized implementations with different tile sizes
        for tile_size in tile_sizes:
            for implementation in ["Original", "Optimized", "Turbo"]:
                print(f"Benchmarking {implementation} implementation with tile size {tile_size}...")
                for _ in range(3):  # Fewer runs for quicker benchmarking
                    if implementation == "Original":
                        start_time = time.time()
                        result = run_original_spmm(torch_sparse_matrix, dense_matrix, tile_size)
                        end_time = time.time()
                    elif implementation == "Optimized":
                        start_time = time.time()
                        result = run_optimized_spmm(torch_sparse_matrix, dense_matrix, tile_size)
                        end_time = time.time()
                    else:
                        start_time = time.time()
                        result = run_turbo_spmm(torch_sparse_matrix, dense_matrix, tile_size)
                        end_time = time.time()

                    results.append({
                        'Implementation': implementation,
                        'Tile Size': tile_size,
                        'Group': matrix.group,
                        'Name': matrix.name,
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
results_df.to_csv("spmm_optimized_comparison_results.csv", index=False)
print("Benchmarking complete. Results saved to 'spmm_optimized_comparison_results.csv'.")

# Plot settings
sns.set(style="whitegrid", context="talk")

# Compute average runtimes per implementation and tile size
summary_df = results_df.groupby(['Implementation', 'Tile Size'])['Runtime'].mean().reset_index()

# Create bar plot
plt.figure(figsize=(14, 8))
bar_plot = sns.barplot(x='Implementation', y='Runtime', hue='Tile Size', data=summary_df)

# Add value labels on bars
for i, bar in enumerate(bar_plot.patches):
    height = bar.get_height()
    bar_plot.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                f'{height:.5f}',
                ha='center', va='bottom', rotation=0, fontsize=9)

plt.title('SpMM Performance: Original vs. Optimized Implementation')
plt.xlabel('Implementation')
plt.ylabel('Average Runtime (seconds)')
plt.legend(title='Tile Size')
plt.tight_layout()
plt.savefig("spmm_optimized_comparison_barplot.pdf", bbox_inches='tight')
plt.show()

# Create a line plot showing performance by NNZ
plt.figure(figsize=(14, 8))
for impl in summary_df['Implementation'].unique():
    for tile in [str(t) for t in tile_sizes]:
        subset = results_df[(results_df['Implementation'] == impl) &
                           (results_df['Tile Size'].astype(str) == tile)]
        if not subset.empty:
            grouped = subset.groupby('NNZ')['Runtime'].mean().reset_index()
            plt.scatter(grouped['NNZ'], grouped['Runtime'],
                      label=f'{impl} (Tile Size={tile})', s=50, alpha=0.7)
            plt.plot(grouped['NNZ'], grouped['Runtime'], alpha=0.5)

# Add untiled implementation
untiled_subset = results_df[results_df['Implementation'] == 'Untiled']
if not untiled_subset.empty:
    grouped = untiled_subset.groupby('NNZ')['Runtime'].mean().reset_index()
    plt.scatter(grouped['NNZ'], grouped['Runtime'],
              label='Untiled', s=50, alpha=0.7)
    plt.plot(grouped['NNZ'], grouped['Runtime'], alpha=0.5)

plt.xscale('log')
plt.yscale('log')
plt.title('SpMM Performance vs Matrix Size')
plt.xlabel('Number of Non-Zeros (NNZ)')
plt.ylabel('Average Runtime (seconds)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("spmm_optimized_comparison_by_nnz.pdf", bbox_inches='tight')
plt.show()

# Calculate speedup over original implementation
baseline_df = summary_df[summary_df['Implementation'] == 'Original'].copy()
baseline_df = baseline_df.rename(columns={'Runtime': 'Baseline_Runtime'})
baseline_df = baseline_df.drop('Implementation', axis=1)

compare_df = summary_df.merge(baseline_df, on='Tile Size')
compare_df['Speedup'] = compare_df['Baseline_Runtime'] / compare_df['Runtime']

# Create a speedup comparison plot
plt.figure(figsize=(12, 6))
speedup_df = compare_df[compare_df['Implementation'] != 'Original']
sns.barplot(x='Tile Size', y='Speedup', data=speedup_df)
plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (Original)')
plt.title('Speedup of Optimized Implementation over Original')
plt.xlabel('Tile Size')
plt.ylabel('Speedup Factor')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("spmm_optimized_speedup.pdf", bbox_inches='tight')
plt.show()
