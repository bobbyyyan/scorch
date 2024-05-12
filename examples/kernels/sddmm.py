import ssgetpy
from scipy.io import mmread
from pathlib import Path
import torch
import numpy as np
import time
import pandas as pd
import scorch
from tqdm import tqdm

import traceback
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

# Set random seed for reproducibility
np.random.seed(15)
torch.manual_seed(15)

matrices = ssgetpy.search(limit=5000)
matrices = [matrix for matrix in matrices if max(matrix.rows, matrix.cols) < 120000]

results = []

for matrix in tqdm(matrices, desc="Benchmarking SDDMM"):
    try:
        print(f"Processing matrix {matrix.id} {matrix.name} in group {matrix.group} with {matrix.nnz} NNZ...")
        # sparse_matrix_mm = matrix.download(format='MM', extract=True)

        matrix_path = Path(f"~/.ssgetpy/MM/{matrix.group}/{matrix.name}/{matrix.name}.mtx").expanduser()
        sparse_matrix = mmread(matrix_path.resolve())
        print(f"Matrix shape: {sparse_matrix.shape}")

        matrix_format = 'csr'
        torch_sparse_matrix = scipy_sparse_to_torch_sparse(sparse_matrix, format=matrix_format)
        dense_matrix_A = torch.rand((torch_sparse_matrix.shape[0], 1), dtype=torch.float32)
        dense_matrix_B = torch.rand((1, torch_sparse_matrix.shape[1]), dtype=torch.float32)
        nnz = sparse_matrix.nnz

        for framework in ["PyTorch", "Scorch"]:
            print(f"Benchmarking {framework}...")
            for _ in range(10):
                if framework == "PyTorch":
                    start_time = time.time()
                    result = torch.matmul(dense_matrix_A, dense_matrix_B)
                    result = torch.mul(torch_sparse_matrix, result)
                elif framework == "Scorch":
                    start_time = time.time()
                    result = scorch.einsum("ij,ik,kj->ij", torch_sparse_matrix, dense_matrix_A, dense_matrix_B)
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

    except Exception as e:
        traceback.print_exc()
        print(f"Error processing matrix {matrix.name} in group {matrix.group}: {e}")

results_df = pd.DataFrame(results)
results_df.to_csv("sddmm_benchmark_results.csv", index=False)
print("Benchmarking complete. Results saved to 'sddmm_benchmark_results.csv'.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.title('SDDMM (Sampled Dense-Dense Matrix Multiplication) Benchmark')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig("sddmm_benchmark_plot.pdf", bbox_inches='tight')
plt.show()
