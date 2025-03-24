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

def run_turbo_spmm(sparse_tensor, dense_tensor, tile_size):
    """Run the turbo SpMM implementation"""
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

def run_ultra_spmm(sparse_tensor, dense_tensor, tile_size):
    """Run the ultra SpMM implementation"""
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

# Set random seed for reproducibility
np.random.seed(15)
torch.manual_seed(15)

# Define the tile sizes to benchmark
tile_sizes = [128, 256, 512]

# Try to get a diverse set of matrices
matrices = ssgetpy.search(limit=5000)
matrices = [matrix for matrix in matrices if max(matrix.rows, matrix.cols) < 700000]

# Remove specific problematic indices if they exist in our set
idxs_to_remove = [961, 962]
matrices = [matrix for idx, matrix in enumerate(matrices) if idx not in idxs_to_remove]

# Limit to 25 matrices for quicker benchmarking
matrices = matrices[:25]

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

        # Categorize the matrix
        matrix_category = categorize_matrix(sparse_matrix, nnz)

        # Test untiled implementation first as a baseline
        print("Benchmarking untiled implementation...")
        for _ in range(5):  # More runs for better statistical significance
            # Warm-up run (not timed)
            _ = run_untiled_spmm(torch_sparse_matrix, dense_matrix)

            # Timed run
            start_time = time.time()
            result = run_untiled_spmm(torch_sparse_matrix, dense_matrix)
            end_time = time.time()

            results.append({
                'Implementation': 'Untiled',
                'Tile Size': 'N/A',
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

        # Test Turbo and Ultra implementations with different tile sizes
        for tile_size in tile_sizes:
            for implementation in ["Turbo", "Ultra"]:
                print(f"Benchmarking {implementation} implementation with tile size {tile_size}...")
                for _ in range(5):  # More runs for better statistical significance
                    # Warm-up run (not timed)
                    if implementation == "Turbo":
                        _ = run_turbo_spmm(torch_sparse_matrix, dense_matrix, tile_size)
                    else:
                        _ = run_ultra_spmm(torch_sparse_matrix, dense_matrix, tile_size)

                    # Timed run
                    if implementation == "Turbo":
                        start_time = time.time()
                        result = run_turbo_spmm(torch_sparse_matrix, dense_matrix, tile_size)
                        end_time = time.time()
                    else:
                        start_time = time.time()
                        result = run_ultra_spmm(torch_sparse_matrix, dense_matrix, tile_size)
                        end_time = time.time()

                    results.append({
                        'Implementation': implementation,
                        'Tile Size': tile_size,
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
results_df.to_csv("spmm_turbo_vs_ultra_results.csv", index=False)
print("Benchmarking complete. Results saved to 'spmm_turbo_vs_ultra_results.csv'.")

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
    if not np.isnan(height):
        bar_plot.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                    f'{height:.5f}',
                    ha='center', va='bottom', rotation=0, fontsize=9)

plt.title('SpMM Performance: Turbo vs. Ultra Implementation')
plt.xlabel('Implementation')
plt.ylabel('Average Runtime (seconds)')
plt.legend(title='Tile Size')
plt.tight_layout()
plt.savefig("spmm_turbo_vs_ultra_barplot.pdf", bbox_inches='tight')
plt.show()

# Calculate speedup of Ultra over Turbo
speedup_data = []
for tile_size in tile_sizes:
    turbo_runtime = summary_df[(summary_df['Implementation'] == 'Turbo') &
                            (summary_df['Tile Size'] == tile_size)]['Runtime'].values[0]
    ultra_runtime = summary_df[(summary_df['Implementation'] == 'Ultra') &
                             (summary_df['Tile Size'] == tile_size)]['Runtime'].values[0]
    speedup = turbo_runtime / ultra_runtime
    speedup_data.append({
        'Tile Size': tile_size,
        'Speedup': speedup
    })

speedup_df = pd.DataFrame(speedup_data)

# Create speedup plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Tile Size', y='Speedup', data=speedup_df)
plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (Turbo)')
plt.title('Speedup of Ultra Implementation over Turbo Implementation')
plt.xlabel('Tile Size')
plt.ylabel('Speedup Factor')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("spmm_ultra_vs_turbo_speedup.pdf", bbox_inches='tight')
plt.show()

# Create a plot showing performance by matrix category
plt.figure(figsize=(16, 10))
category_df = results_df[results_df['Implementation'].isin(['Turbo', 'Ultra'])].copy()
category_df = category_df.groupby(['Implementation', 'Tile Size', 'Category'])['Runtime'].mean().reset_index()

# Plot by category
sns.barplot(x='Category', y='Runtime', hue='Implementation', data=category_df)
plt.title('Performance by Matrix Category')
plt.xlabel('Matrix Category')
plt.ylabel('Average Runtime (seconds)')
plt.xticks(rotation=45)
plt.legend(title='Implementation')
plt.tight_layout()
plt.savefig("spmm_ultra_vs_turbo_by_category.pdf", bbox_inches='tight')
plt.show()

# Display statistics about the speedup for different matrix categories
category_speedup = []
for category in category_df['Category'].unique():
    for tile_size in tile_sizes:
        turbo_data = category_df[(category_df['Implementation'] == 'Turbo') &
                             (category_df['Category'] == category) &
                             (category_df['Tile Size'] == tile_size)]
        ultra_data = category_df[(category_df['Implementation'] == 'Ultra') &
                              (category_df['Category'] == category) &
                              (category_df['Tile Size'] == tile_size)]

        if not turbo_data.empty and not ultra_data.empty:
            turbo_runtime = turbo_data['Runtime'].values[0]
            ultra_runtime = ultra_data['Runtime'].values[0]
            speedup = turbo_runtime / ultra_runtime
            category_speedup.append({
                'Category': category,
                'Tile Size': tile_size,
                'Speedup': speedup
            })

category_speedup_df = pd.DataFrame(category_speedup)

if not category_speedup_df.empty:
    plt.figure(figsize=(16, 10))
    sns.barplot(x='Category', y='Speedup', hue='Tile Size', data=category_speedup_df)
    plt.axhline(y=1.0, color='r', linestyle='--', label='No Speedup')
    plt.title('Ultra Implementation Speedup by Matrix Category')
    plt.xlabel('Matrix Category')
    plt.ylabel('Speedup over Turbo Implementation')
    plt.xticks(rotation=45)
    plt.legend(title='Tile Size')
    plt.tight_layout()
    plt.savefig("spmm_ultra_speedup_by_category.pdf", bbox_inches='tight')
    plt.show()

print("Analysis complete. All plots have been saved.")
