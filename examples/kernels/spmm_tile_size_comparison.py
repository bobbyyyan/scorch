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

def run_tiled_spmm_with_size(sparse_tensor, dense_tensor, tile_size):
    """Run the tiled version of SpMM with a specific tile size by calling the C++ function directly"""
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

# Set random seed for reproducibility
np.random.seed(15)
torch.manual_seed(15)

# Define the tile sizes to benchmark
tile_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]

# Limit the number of matrices to test for quicker benchmarking
matrices = ssgetpy.search(limit=5000)
matrices = [matrix for matrix in matrices if max(matrix.rows, matrix.cols) < 700000]
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

        for tile_size in tile_sizes:
            print(f"Benchmarking tile size {tile_size}...")
            for _ in range(3):  # Reduce number of runs for quicker benchmarking
                start_time = time.time()
                result = run_tiled_spmm_with_size(torch_sparse_matrix, dense_matrix, tile_size)
                end_time = time.time()

                results.append({
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
results_df.to_csv("spmm_tile_size_benchmark_results.csv", index=False)
print("Benchmarking complete. Results saved to 'spmm_tile_size_benchmark_results.csv'.")

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

# Group by tile size and NNZ
grouped_df = results_df.groupby(['Tile Size', 'NNZ'])['Runtime'].mean().reset_index()

# Create a line plot for each tile size
for tile_size in tile_sizes:
    tile_df = grouped_df[grouped_df['Tile Size'] == tile_size]
    plt.scatter(tile_df['NNZ'], tile_df['Runtime'], label=f'Tile Size = {tile_size}', s=2, alpha=0.7)

plt.xlabel('Number of Non-Zeros (NNZ)')
plt.ylabel('Average Runtime (seconds)')
plt.title('SpMM Performance for Different Tile Sizes')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig("spmm_tile_size_benchmark_plot.pdf", bbox_inches='tight')
plt.show()

# Create a separate plot showing average performance by tile size
plt.figure(figsize=(12, 6))
avg_by_tile = results_df.groupby('Tile Size')['Runtime'].mean().reset_index()
plt.plot(avg_by_tile['Tile Size'], avg_by_tile['Runtime'], 'o-', linewidth=2)
plt.xlabel('Tile Size')
plt.ylabel('Average Runtime (seconds)')
plt.title('SpMM Performance by Tile Size')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("spmm_tile_size_average_performance.pdf", bbox_inches='tight')
plt.show()
