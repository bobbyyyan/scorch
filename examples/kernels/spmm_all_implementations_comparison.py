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

def run_apex_spmm(sparse_tensor, dense_tensor, tile_size):
    """Run the apex SpMM implementation"""
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

    result_cpp = ops.spmm_csr_float_apex(*args)

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
np.random.seed(42)
torch.manual_seed(42)

# Define the tile sizes to benchmark
tile_sizes = [128, 256, 512]

# Try to get a diverse set of matrices
matrices = ssgetpy.search(limit=2000)
matrices = [matrix for matrix in matrices if max(matrix.rows, matrix.cols) < 700000]

# Get a good mix of matrices by selecting from different groups
small_dense_matrices = [m for m in matrices if m.rows < 1000 and m.nnz / (m.rows * m.cols) > 0.01]
small_mod_sparse_matrices = [m for m in matrices if m.rows < 1000 and 0.001 < m.nnz / (m.rows * m.cols) <= 0.01]
medium_sparse_matrices = [m for m in matrices if 1000 <= m.rows < 10000 and 0.0001 < m.nnz / (m.rows * m.cols) <= 0.001]
medium_mod_sparse_matrices = [m for m in matrices if 1000 <= m.rows < 10000 and 0.001 < m.nnz / (m.rows * m.cols) <= 0.01]

# Select up to 500 matrices from each category for a balanced test set
test_matrices = []
for category in [small_dense_matrices, small_mod_sparse_matrices, medium_sparse_matrices, medium_mod_sparse_matrices]:
    if category:
        test_matrices.extend(category[:500])

# Remove specific problematic indices if they exist in our set
test_matrices = [m for m in test_matrices if m.id not in [961, 962]]

# # Make sure we have a reasonable number of matrices
# if len(test_matrices) > 20:
#     test_matrices = test_matrices[:20]
# elif len(test_matrices) < 5:
#     # Fallback if categories didn't yield enough matrices
#     test_matrices = matrices[:20]

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

        # Categorize the matrix
        matrix_category = categorize_matrix(sparse_matrix, nnz)

        # Test untiled implementation first as a baseline
        print("Benchmarking untiled implementation...")
        for _ in range(3):  # 3 runs for better statistical significance
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

        # Define all the implementations to test
        implementations = [
            {"name": "Optimized", "function": run_optimized_spmm},
            {"name": "Turbo", "function": run_turbo_spmm},
            {"name": "Ultra", "function": run_ultra_spmm},
            {"name": "Apex", "function": run_apex_spmm},
        ]

        # Test all implementations with different tile sizes
        for tile_size in tile_sizes:
            for impl in implementations:
                print(f"Benchmarking {impl['name']} implementation with tile size {tile_size}...")
                for _ in range(3):  # 3 runs for better statistical significance
                    # Warm-up run (not timed)
                    _ = impl["function"](torch_sparse_matrix, dense_matrix, tile_size)

                    # Timed run
                    start_time = time.time()
                    result = impl["function"](torch_sparse_matrix, dense_matrix, tile_size)
                    end_time = time.time()

                    results.append({
                        'Implementation': impl["name"],
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
results_df.to_csv("spmm_all_implementations_results.csv", index=False)
print("Benchmarking complete. Results saved to 'spmm_all_implementations_results.csv'.")

# Plot settings
sns.set(style="whitegrid", context="talk")

# Compute average runtimes per implementation and tile size
summary_df = results_df.groupby(['Implementation', 'Tile Size'])['Runtime'].mean().reset_index()

# Create bar plot for overall performance comparison
plt.figure(figsize=(14, 8))
bar_plot = sns.barplot(x='Implementation', y='Runtime', hue='Tile Size', data=summary_df)

# Add value labels on bars
for i, bar in enumerate(bar_plot.patches):
    height = bar.get_height()
    if not np.isnan(height):
        bar_plot.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                    f'{height:.5f}',
                    ha='center', va='bottom', rotation=0, fontsize=9)

plt.title('SpMM Performance: All Implementations')
plt.xlabel('Implementation')
plt.ylabel('Average Runtime (seconds)')
plt.legend(title='Tile Size')
plt.tight_layout()
plt.savefig("spmm_all_implementations_barplot.pdf", bbox_inches='tight')
plt.show()

# Calculate speedup over baseline (untiled)
baseline_runtime = summary_df[summary_df['Implementation'] == 'Untiled']['Runtime'].values[0]
summary_df['Speedup_vs_Untiled'] = baseline_runtime / summary_df['Runtime']

# Plot speedup over untiled
plt.figure(figsize=(14, 8))
speedup_plot = sns.barplot(x='Implementation', y='Speedup_vs_Untiled', hue='Tile Size',
                           data=summary_df[summary_df['Implementation'] != 'Untiled'])

for i, bar in enumerate(speedup_plot.patches):
    height = bar.get_height()
    if not np.isnan(height):
        speedup_plot.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}x',
                    ha='center', va='bottom', rotation=0, fontsize=9)

plt.title('Speedup over Untiled Implementation')
plt.xlabel('Implementation')
plt.ylabel('Speedup Factor')
plt.legend(title='Tile Size')
plt.tight_layout()
plt.savefig("spmm_speedup_vs_untiled.pdf", bbox_inches='tight')
plt.show()

# Group by matrix category and implementation
category_df = results_df.groupby(['Category', 'Implementation', 'Tile Size'])['Runtime'].mean().reset_index()

# Compare implementations by matrix category
for category in category_df['Category'].unique():
    plt.figure(figsize=(14, 8))
    category_data = category_df[category_df['Category'] == category]
    ax = sns.barplot(x='Implementation', y='Runtime', hue='Tile Size', data=category_data)

    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                        f'{height:.5f}',
                        ha='center', va='bottom', rotation=0, fontsize=9)

    plt.title(f'Performance Comparison for {category} Matrices')
    plt.xlabel('Implementation')
    plt.ylabel('Average Runtime (seconds)')
    plt.legend(title='Tile Size')
    plt.tight_layout()
    plt.savefig(f"spmm_{category.replace(' ', '_')}_comparison.pdf", bbox_inches='tight')
    plt.show()

# Calculate the best implementation for each matrix category
best_implementations = []
for category in category_df['Category'].unique():
    category_data = category_df[category_df['Category'] == category]
    fastest_config = category_data.loc[category_data['Runtime'].idxmin()]
    best_implementations.append({
        'Category': category,
        'Best Implementation': fastest_config['Implementation'],
        'Best Tile Size': fastest_config['Tile Size'],
        'Runtime': fastest_config['Runtime']
    })

best_df = pd.DataFrame(best_implementations)
print("Best implementation per matrix category:")
print(best_df)

# Create a heatmap showing the performance of Apex implementation across categories
apex_data = category_df[category_df['Implementation'] == 'Apex']
apex_pivot = apex_data.pivot_table(values='Runtime', index='Category', columns='Tile Size')

plt.figure(figsize=(12, 8))
sns.heatmap(apex_pivot, annot=True, fmt=".5f", cmap="YlGnBu", cbar_kws={'label': 'Runtime (seconds)'})
plt.title('Apex Implementation Performance by Matrix Category and Tile Size')
plt.tight_layout()
plt.savefig("spmm_apex_performance_heatmap.pdf", bbox_inches='tight')
plt.show()

# Rank implementations for each category
ranks = []
for category in category_df['Category'].unique():
    for tile_size in tile_sizes:
        category_tile_data = category_df[(category_df['Category'] == category) &
                                      (category_df['Tile Size'] == tile_size)]

        # Skip if we don't have this combination
        if len(category_tile_data) == 0:
            continue

        # Rank implementations by runtime (1 = fastest)
        ranked_data = category_tile_data.sort_values('Runtime')
        for i, row in enumerate(ranked_data.itertuples()):
            ranks.append({
                'Category': category,
                'Tile Size': tile_size,
                'Implementation': row.Implementation,
                'Rank': i + 1,
                'Runtime': row.Runtime
            })

ranks_df = pd.DataFrame(ranks)

# Show ranking statistics for Apex implementation
apex_ranks = ranks_df[ranks_df['Implementation'] == 'Apex']
print("\nApex implementation ranking statistics:")
print(apex_ranks.groupby('Rank').size())

# Calculate consistency score (average rank across all categories)
avg_ranks = ranks_df.groupby(['Implementation', 'Tile Size'])['Rank'].mean().reset_index()
avg_ranks = avg_ranks.sort_values('Rank')
print("\nImplementations by average rank (lower is better):")
print(avg_ranks)

# Create a plot showing consistency (average rank)
plt.figure(figsize=(12, 6))
consistency_plot = sns.barplot(x='Implementation', y='Rank', hue='Tile Size', data=avg_ranks)
plt.title('Implementation Consistency (Average Rank Across Categories)')
plt.xlabel('Implementation')
plt.ylabel('Average Rank (lower is better)')
plt.legend(title='Tile Size')
plt.tight_layout()
plt.savefig("spmm_implementation_consistency.pdf", bbox_inches='tight')
plt.show()

print("Analysis complete. All plots have been saved.")
