import ssgetpy
from scipy.io import mmread
from pathlib import Path
import torch
import numpy as np
import time
import pandas as pd
import scorch
from tqdm import tqdm

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

matrices = ssgetpy.search(limit=50)

results = []

for matrix in tqdm(matrices, desc="Benchmarking Matrices"):
    try:
        sparse_matrix_mm = matrix.download(format='MM', extract=True)
        matrix_path = Path(f"~/.ssgetpy/MM/{matrix.group}/{matrix.name}/{matrix.name}.mtx").expanduser()
        sparse_matrix = mmread(matrix_path.resolve())

        torch_sparse_matrix = scipy_sparse_to_torch_sparse(sparse_matrix, format='csr')
        dense_matrix = torch.rand((torch_sparse_matrix.shape[1], 100), dtype=torch.float32)
        nnz = sparse_matrix.nnz

        for framework in ["PyTorch", "Scorch"]:
            for _ in range(20):
                start_time = time.time()
                if framework == "PyTorch":
                    result = torch.sparse.mm(torch_sparse_matrix, dense_matrix)
                else:  # Scorch
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
                    'Runtime': end_time - start_time
                })

    except Exception as e:
        print(f"Error processing matrix {matrix.name} in group {matrix.group}: {e}")

results_df = pd.DataFrame(results)
results_df.to_csv("spmm_benchmark_results.csv", index=False)
print("Benchmarking complete. Results saved to 'spmm_benchmark_results.csv'.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results from the CSV
results_df = pd.read_csv("spmm_benchmark_results.csv", dtype={"Runtime": float, "NNZ": int})

# Plot settings
sns.set(style="whitegrid")

# Plotting
plt.figure(figsize=(10, 6))
for framework in ['PyTorch', 'Scorch']:
    sub_df = results_df[results_df['Framework'] == framework]
    mean_runtime = sub_df.groupby('NNZ')['Runtime'].mean()
    std_runtime = sub_df.groupby('NNZ')['Runtime'].std()

    plt.plot(mean_runtime.index, mean_runtime, label=f'{framework}')
    plt.fill_between(mean_runtime.index, mean_runtime-std_runtime, mean_runtime+std_runtime, alpha=0.3)

plt.xlabel('Number of Non-Zeros (NNZ)')
plt.ylabel('Average Runtime (seconds)')
plt.title('Sparse Matrix Multiplication Performance')
plt.legend()
plt.xscale('log')
# plt.yscale('log')
plt.savefig("spmm_benchmark_plot.pdf")
plt.show()
