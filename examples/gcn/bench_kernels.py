from datetime import datetime

import torch
from torch.utils.cpp_extension import load, load_inline
import time
import scorch
from statistics import mean, stdev

from matplotlib import pyplot as plt
import numpy as np

from scorch import Tensor
from scorch.compiler.cin import TensorVar
from scorch.storage import TensorIndex
from scorch.utils import parse_format

with open("kernels/header.h", "r") as f:
    header_cpp_code = f.read()

with open("kernels/scorch-spmm-csr.h", "r") as f:
    cpp_code = f.read()

spmm = load_inline(
    name="kernel",
    cpp_sources=[header_cpp_code, cpp_code],
    functions=["evaluate"],
    extra_cflags=["-O3"],
)


def scorch_spmm_custom(A: Tensor, B: Tensor, **kwargs):
    # a = TensorVar("A", fmt=A.format)
    # b = TensorVar("B", fmt=B.format)

    result_shape = (A.shape[0], B.shape[1])
    args = [result_shape]

    for tensor in [A, B]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor.index.mode_indices)  # type: ignore
        args.append(tensor.values)  # type: ignore

    start_time = time.time()
    result_cpp = spmm.evaluate(*args)
    end_time = time.time()
    eval_time = end_time - start_time
    if "time_dict" in kwargs:
        kwargs["time_dict"]["eval_time"] = eval_time

    result = Tensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format="dd",
        ),
        value=result_cpp._storage._value,
    )

    return result


def scorch_spmm(A, B, **kwargs):
    """Sparse matrix multiplication with Scorch"""
    return scorch.matmul(A, B, **kwargs)


def torch_spmm(A, B):
    """Sparse matrix multiplication with PyTorch"""
    # return torch.sparse.mm(A, B)
    return torch.matmul(A, B)


def scorch_spmv(A, B, **kwargs):
    # return scorch.einsum("ij,j->i", A, B)
    return scorch.matmul(A, B, **kwargs)


def torch_spmv(A, B):
    return torch.matmul(A, B)


def scorch_sddmm(B, C, D):
    # B is sparse, e.g. CSR or COO
    # C and D are dense
    return scorch.einsum("ij,ik,kj->ij", B, C, D)


def torch_sddmm(B, C, D):
    return torch.matmul(B, torch.matmul(C, D))


def scorch_sddmm_dense(B, C, D):
    # B is sparse, e.g. CSR or COO
    # C and D are dense
    return scorch.einsum("ij,ik,kj->ij", B, C, D, format="dd")


def torch_sddmm_dense(B, C, D):
    # B is sparse, e.g. CSR or COO
    # C and D are dense
    B_dense = B.to_dense()
    return torch.einsum("ij,ik,kj->ij", B_dense, C, D)
    # return torch.matmul(B, torch.matmul(C, D))
    # return torch.einsum("ij,ik,kj->ij", B, C, D)


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
    A_torch = A_torch.to("cpu")
    A = A.to_sparse("ds")
    return A, A_torch


def bench_sddmm(
    dimensions,
    num_runs=5,
    sparsity=0.99,
    sparse_format="csr",
    output_format=None,
):
    torch_times = []
    scorch_times = []
    speedups_means = []
    speedups_stds = []

    torch_sddmm_func = torch_sddmm
    scorch_sddmm_func = scorch_sddmm

    if output_format == "dense":
        torch_sddmm_func = torch_sddmm_dense
        scorch_sddmm_func = scorch_sddmm_dense

    for dim in dimensions:
        torch_time_run = []
        scorch_time_run = []
        speedup_run = []

        for _ in range(num_runs):
            if sparse_format == "coo":
                B, B_torch = gen_rand_sparse_coo(dim, dim, sparsity)
            else:
                B, B_torch = gen_rand_sparse_csr(dim, dim, sparsity)

            C_torch = torch.rand(dim, dim)
            C = scorch.from_torch(C_torch)

            D_torch = torch.rand(dim, dim)
            D = scorch.from_torch(D_torch)

            # Warm up
            for _ in range(1):
                scorch_sddmm_func(B, C, D)

            start = time.perf_counter()
            scorch_sddmm_func(B, C, D)
            end = time.perf_counter()
            scorch_time = end - start
            scorch_time_run.append(scorch_time)

            # Warm up
            for _ in range(1):
                torch_sddmm_func(B_torch, C_torch, D_torch)

            start = time.perf_counter()
            torch_sddmm_func(B_torch, C_torch, D_torch)
            end = time.perf_counter()
            torch_time = end - start
            torch_time_run.append(torch_time)

            speedup_run.append(torch_time / scorch_time)

        torch_times.append(np.mean(torch_time_run))
        scorch_times.append(np.mean(scorch_time_run))
        speedups_means.append(np.mean(speedup_run))
        speedups_stds.append(np.std(speedup_run))

    return torch_times, scorch_times, speedups_means, speedups_stds


def bench_spmspm(dimensions, num_runs=5, sparsity=0.99, sparse_format="csr"):
    torch_times = []
    scorch_times = []
    speedups_means = []
    speedups_stds = []

    for dim in dimensions:
        torch_time_run = []
        scorch_time_run = []
        speedup_run = []

        for _ in range(num_runs):
            if sparse_format == "coo":
                A, A_torch = gen_rand_sparse_coo(dim, dim, sparsity)
                B, B_torch = gen_rand_sparse_coo(dim, dim, sparsity)
            else:
                A, A_torch = gen_rand_sparse_csr(dim, dim, sparsity)
                B, B_torch = gen_rand_sparse_csr(dim, dim, sparsity)

            # Warm up
            for _ in range(1):
                scorch_spmm(A, B)

            start = time.perf_counter()
            scorch_spmm(A, B)
            end = time.perf_counter()
            scorch_time = end - start
            scorch_time_run.append(scorch_time)

            # Warm up
            for _ in range(1):
                torch_spmm(A_torch, B_torch)

            start = time.perf_counter()
            torch_spmm(A_torch, B_torch)
            end = time.perf_counter()
            torch_time = end - start
            torch_time_run.append(torch_time)

            speedup_run.append(torch_time / scorch_time)

        torch_times.append(np.mean(torch_time_run))
        scorch_times.append(np.mean(scorch_time_run))
        speedups_means.append(np.mean(speedup_run))
        speedups_stds.append(np.std(speedup_run))

    return torch_times, scorch_times, speedups_means, speedups_stds


def bench_spmm(
    dimensions,
    num_warmup_runs=3,
    num_runs=5,
    sparsity=0.99,
    sparse_format="csr",
    custom_func=None,
):
    torch_times = []
    scorch_times = []
    speedups_means = []
    speedups_stds = []

    scorch_spmm_func = scorch_spmm
    if custom_func is not None:
        scorch_spmm_func = custom_func

    gen_rand_sparse_func = gen_rand_sparse_coo if sparse_format == "coo" else gen_rand_sparse_csr

    for dim in dimensions:
        torch_time_run = []
        scorch_time_run = []
        speedup_run = []

        for _ in range(num_runs):
            A, A_torch = gen_rand_sparse_func(dim, dim, sparsity)
            B_torch = torch.rand(dim, dim).to("cpu")
            B = scorch.from_torch(B_torch)

            # Warm up for scorch benchmark.
            for _ in range(num_warmup_runs):
                scorch_spmm_func(A, B)

            time_dict = {}
            start = time.perf_counter()
            scorch_result = scorch_spmm_func(A, B, time_dict=time_dict)
            end = time.perf_counter()
            # scorch_time = end - start
            scorch_time = time_dict["eval_time"]
            scorch_time_run.append(scorch_time)

            # Warm up for torch benchmark.
            for _ in range(num_warmup_runs):
                torch_spmm(A_torch, B_torch)

            start = time.perf_counter()
            torch_result = torch_spmm(A_torch, B_torch)
            end = time.perf_counter()
            torch_time = end - start
            torch_time_run.append(torch_time)

            # assert torch.allclose(torch_result, scorch_result.values)

            speedup_run.append(torch_time / scorch_time)

        torch_times.append(np.mean(torch_time_run))
        scorch_times.append(np.mean(scorch_time_run))
        speedups_means.append(np.mean(speedup_run))
        speedups_stds.append(np.std(speedup_run))

    return torch_times, scorch_times, speedups_means, speedups_stds


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

            time_dict = {}
            start = time.perf_counter()
            scorch_spmv(A, B, time_dict=time_dict)
            end = time.perf_counter()
            # scorch_time = end - start
            scorch_time = time_dict["eval_time"]
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


def plot_benchmark(dimensions, speedup_means, speedup_stds, benchmark="SpMV"):
    plt.figure(figsize=(10, 6))

    # Calculate upper and lower bounds for speedup
    speedup_upper = np.array(speedup_means) + np.array(speedup_stds)
    speedup_lower = np.array(speedup_means) - np.array(speedup_stds)

    # Plot mean speedup
    plt.plot(dimensions, speedup_means, "-o", label="Mean speedup (Scorch/PyTorch)")

    # Add shaded area for standard deviation
    plt.fill_between(dimensions, speedup_lower, speedup_upper, color="blue", alpha=0.2)

    plt.title(f"{benchmark} Speedup of Scorch over PyTorch")
    plt.xlabel("Matrix dimensions")
    plt.ylabel("Speedup factor")
    plt.legend()
    plt.grid(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name_with_timestamp = f"{benchmark.lower()}-{timestamp}.png"

    plt.savefig(file_name_with_timestamp, dpi=300)

    # Also plot Mean speedup (PyTorch/Scorch)
    plt.figure(figsize=(10, 6))

    speedup_means = 1 / np.array(speedup_means)
    speedup_upper = 1 / speedup_upper
    speedup_lower = 1 / speedup_lower

    # Plot mean speedup
    plt.plot(dimensions, speedup_means, "-o", label="Mean speedup (PyTorch/Scorch)")

    # Add shaded area for standard deviation
    plt.fill_between(dimensions, speedup_lower, speedup_upper, color="blue", alpha=0.2)

    plt.title(f"{benchmark} Speedup of PyTorch over Scorch")
    plt.xlabel("Matrix dimensions")
    plt.ylabel("Speedup factor")
    plt.legend()
    plt.grid(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name_with_timestamp = f"{benchmark.lower()}-inverse-{timestamp}.png"

    plt.savefig(file_name_with_timestamp, dpi=300)


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
            1500,
            2000,
            3000,
            5000,
            # 10000,
            # 15000,
            # 20000,
            # 25000,
            # 30000,
            # 35000,
            # 40000,
            # 45000,
            # 50000,
        ]
    )
    # torch_times, scorch_times, speedups_means, speedups_stds = bench_spmv(dimensions)
    # plot_benchmark(dimensions, speedups_means, speedups_stds, benchmark="SpMV")

    torch_times, scorch_times, speedups_means, speedups_stds = bench_spmm(
        dimensions,
        num_runs=10,
        sparsity=0.99,
        sparse_format="csr",
        custom_func=scorch_spmm_custom,
    )
    plot_benchmark(
        dimensions, speedups_means, speedups_stds, benchmark="SpMM 99 (CSR) (After)"
    )

    # torch_times, scorch_times, speedups_means, speedups_stds = bench_spmm(
    #     dimensions,
    #     num_runs=10,
    #     sparsity=0.99,
    #     sparse_format="csr",
    # )
    # plot_benchmark(
    #     dimensions, speedups_means, speedups_stds, benchmark="SpMM 99 (CSR) (Before)",
    # )

    # torch_times, scorch_times, speedups_means, speedups_stds = bench_spmspm(
    #     dimensions, sparsity=0.99, sparse_format="coo"
    # )
    # plot_benchmark(dimensions, speedups_means, speedups_stds, benchmark="SpMSpM (COO)")

    # torch_times, scorch_times, speedups_means, speedups_stds = bench_spmspm(
    #     dimensions, sparsity=0.99, sparse_format="csr"
    # )
    # plot_benchmark(dimensions, speedups_means, speedups_stds, benchmark="SpMSpM (CSR)")

    # torch_times, scorch_times, speedups_means, speedups_stds = bench_sddmm(
    #     dimensions,
    #     sparsity=0.99,
    #     sparse_format="csr",
    #     output_format="dense",
    # )
    # plot_benchmark(
    #     dimensions, speedups_means, speedups_stds, benchmark="SDDMM (CSR, Dense output)"
    # )

    # torch_times, scorch_times, speedups_means, speedups_stds = bench_sddmm(
    #     dimensions,
    #     sparsity=0.99,
    #     sparse_format="csr",
    # )
    # plot_benchmark(dimensions, speedups_means, speedups_stds, benchmark="SDDMM (CSR)")
