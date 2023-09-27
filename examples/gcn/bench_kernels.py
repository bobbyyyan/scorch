from datetime import datetime

import torch
import time
import scorch
from statistics import mean, stdev

from matplotlib import pyplot as plt
import numpy as np


def scorch_spmm(A, B):
    """Sparse matrix multiplication with Scorch"""
    return scorch.matmul(A, B)


def torch_spmm(A, B):
    """Sparse matrix multiplication with PyTorch"""
    return torch.sparse.mm(A, B)


def scorch_spmv(A, B):
    # return scorch.einsum("ij,j->i", A, B)
    return scorch.matmul(A, B)


def torch_spmv(A, B):
    return torch.matmul(A, B)


def gen_rand_sparse_coo(dim_m, dim_n, sparsity):
    """Generate sparse matrix with PyTorch
    sparsity is the percentage of zero elements
    """
    nnz = int((1 - sparsity) * dim_m * dim_n)
    all_indices = torch.cartesian_prod(torch.arange(dim_m), torch.arange(dim_n))
    perm = torch.randperm(all_indices.shape[0])
    indices = all_indices[perm][:nnz].T

    # Indices are already unique, so no need to check for uniqueness again.

    # Create a tensor that concatenates the indices along the second dimension.
    # Then, sort this tensor along the second dimension.
    # This will effectively sort the indices by row first and then by column.
    indices_flattened = indices[0] * dim_n + indices[1]
    sorted_indices = torch.argsort(indices_flattened)

    # Use the sorted indices to reorder the original indices and values.
    indices = indices[:, sorted_indices]
    # Add a small constant to ensure all values are non-zero
    values = torch.rand(nnz)
    values = values[sorted_indices]

    A = scorch.from_coo(indices, values, (dim_m, dim_n))
    A_torch = torch.sparse_coo_tensor(indices, values, (dim_m, dim_n)).coalesce()

    return A, A_torch


def gen_rand_sparse_csr(dim_m, dim_n, sparsity):
    A, A_torch = gen_rand_sparse_coo(dim_m, dim_n, sparsity)
    A_torch = A_torch.to_sparse_csr()
    A = A.to_sparse("ds")
    return A, A_torch


def spmm(dim_m=10000, dim_n=10000, dim_k=10000):
    scorch_times = []
    torch_times = []

    for _ in range(3):
        # A, A_torch = gen_rand_sparse_coo(dim_m, dim_n, 0.99)
        A, A_torch = gen_rand_sparse_csr(dim_m, dim_n, 0.99)

        assert torch.allclose(A.values, A_torch.values().flatten())

        B_torch = torch.rand(dim_n, dim_k)
        B = scorch.from_torch(B_torch)

        # Warm up
        for _ in range(1):
            scorch_spmm(A, B)

        start = time.perf_counter()
        C = scorch_spmm(A, B)
        end = time.perf_counter()
        scorch_times.append(end - start)

        # Warm up
        for _ in range(1):
            torch_spmm(A_torch, B_torch)

        start = time.perf_counter()
        C_torch = torch_spmm(A_torch, B_torch)
        end = time.perf_counter()
        torch_times.append(end - start)

        assert torch.allclose(C.to_torch(), C_torch)

    print("dim_m: ", dim_m)
    print("A's nnz count: ", A._nnz())
    print(f"A's sparsity level: {100 - A._nnz() / (dim_m * dim_n) * 100:.2f}%")

    print(f"Scorch: {mean(scorch_times)} ± {stdev(scorch_times)}")
    print(f"PyTorch: {mean(torch_times)} ± {stdev(torch_times)}")
    print("std/mean Scorch: ", stdev(scorch_times) / mean(scorch_times))
    print("std/mean PyTorch: ", stdev(torch_times) / mean(torch_times))
    print(
        "PyTorch/Scorch: \n",
        [torch_times[i] / scorch_times[i] for i in range(len(scorch_times))],
    )
    print("Average PyTorch/Scorch: ", sum(torch_times) / sum(scorch_times))


def bench_spmv(dimensions, num_runs=5):
    torch_times = []
    scorch_times = []
    speedups_means = []
    speedups_stds = []

    for dim_mn in dimensions:
        torch_time_run = []
        scorch_time_run = []
        speedup_run = []

        for _ in range(num_runs):
            A, A_torch = gen_rand_sparse_coo(dim_mn, dim_mn, 0.99)

            assert torch.allclose(A.values, A_torch.values().flatten())

            B_torch = torch.rand(dim_mn)
            B = scorch.from_torch(B_torch)

            # Warm up
            for _ in range(1):
                scorch_spmv(A, B)

            start = time.perf_counter()
            scorch_spmv(A, B)
            end = time.perf_counter()
            scorch_time = end - start
            scorch_time_run.append(scorch_time)

            # Warm up
            for _ in range(1):
                torch_spmv(A_torch, B_torch)

            start = time.perf_counter()
            torch_spmv(A_torch, B_torch)
            end = time.perf_counter()
            torch_time = end - start
            torch_time_run.append(torch_time)

            speedup_run.append(torch_time / scorch_time)

        torch_times.append(np.mean(torch_time_run))
        scorch_times.append(np.mean(scorch_time_run))
        speedups_means.append(np.mean(speedup_run))
        speedups_stds.append(np.std(speedup_run))

    return torch_times, scorch_times, speedups_means, speedups_stds


def plot_bench_spmv(dimensions, speedup_means, speedup_stds):
    plt.figure(figsize=(10, 6))

    # Calculate upper and lower bounds for speedup
    speedup_upper = np.array(speedup_means) + np.array(speedup_stds)
    speedup_lower = np.array(speedup_means) - np.array(speedup_stds)

    # Plot mean speedup
    plt.plot(dimensions, speedup_means, "-o", label="Mean speedup (Scorch/PyTorch)")

    # Add shaded area for standard deviation
    plt.fill_between(dimensions, speedup_lower, speedup_upper, color="blue", alpha=0.2)

    plt.title("Speedup of Scorch over PyTorch")
    plt.xlabel("Matrix dimensions")
    plt.ylabel("Speedup factor")
    plt.legend()
    plt.grid(True)

    file_name_with_timestamp = (
        "spmv_speedup-" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    )
    plt.savefig(file_name_with_timestamp, dpi=300)
    # plt.show()


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")  # Set start method to 'spawn'
    # p = multiprocessing.Process(target=main)  # Create a Process object
    # p.start()  # Start the process
    # p.join()  # Wait for the process to complete
    # spmm()
    # spmv()

    dimensions = np.array(
        [
            100,
            500,
            1000,
            5000,
            10000,
            15000,
            20000,
            25000,
            30000,
            # 35000,
            # 40000,
            # 45000,
            # 50000,
        ]
    )
    torch_times, scorch_times, speedups_means, speedups_stds = bench_spmv(dimensions)
    plot_bench_spmv(dimensions, speedups_means, speedups_stds)
