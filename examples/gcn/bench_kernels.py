"""Benchmark kernels"""

from datetime import datetime
import time
import torch
from torch.utils.cpp_extension import load_inline
import numpy as np
from matplotlib import pyplot as plt
import os

import scorch
from scorch import Tensor
from scorch.storage import TensorIndex

with open("kernels/header.h", "r", encoding="utf-8") as f:
    header_cpp_code = f.read()

with open("kernels/scorch-spmm-csr.h", "r", encoding="utf-8") as f:
    spmm_cpp_code = f.read()

with open("kernels/scorch-spmspm-csr-original.h", "r", encoding="utf-8") as f:
    spmspm_cpp_code = f.read()


spmm = load_inline(
    name="kernel",
    cpp_sources=[header_cpp_code, spmm_cpp_code],
    functions=["evaluate"],
    # extra_cflags=[
    #     "-O3",
    #     "-march=native",
    #     "-ffast-math",
    #     "-fno-signed-zeros",
    #     "-Werror",
    # ],
    extra_cflags=[
        "-O3",
        "-ffast-math",
        "-fno-signed-zeros",
        "-Werror",
    ],
)

spmspm = load_inline(
    name="kernel",
    cpp_sources=[spmspm_cpp_code],
    extra_include_paths=["kernels/"],
    functions=["evaluate"],
    # extra_cflags=[
    #     "-O3",
    #     "-march=native",
    #     "-ffast-math",
    #     "-fno-signed-zeros",
    #     "-Werror",
    # ],
    extra_cflags=[
        "-O3",
        "-ffast-math",
        "-fno-signed-zeros",
        "-Werror",
    ],
)


def scorch_spmspm_custom(a: Tensor, b: Tensor, **kwargs):
    result_shape = (a.shape[0], b.shape[1])
    args = [result_shape]

    for tensor in [a, b]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor.index.mode_indices)  # type: ignore
        args.append(tensor.values)  # type: ignore

    start_time = time.time()
    result_cpp = spmspm.evaluate(*args)
    end_time = time.time()
    eval_time = end_time - start_time
    if "time_dict" in kwargs:
        kwargs["time_dict"]["eval_time"] = eval_time

    result = Tensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp.storage.index.mode_indices,
            tensor_format="ds",
        ),
        value=result_cpp.storage.value,
    )

    return result


def scorch_spmm_custom(a: Tensor, b: Tensor, **kwargs):
    """
    Sparse matrix multiplication with Scorch
    """
    result_shape = (a.shape[0], b.shape[1])
    args = [result_shape]

    for tensor in [a, b]:
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
            mode_indices=result_cpp.storage.index.mode_indices,
            tensor_format="dd",
        ),
        value=result_cpp.storage.value,
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
    """Sparse matrix-vector multiplication with Scorch"""
    # return scorch.einsum("ij,j->i", A, B)
    return scorch.matmul(A, B, **kwargs)


def torch_spmv(a, b):
    """Sparse matrix-vector multiplication with PyTorch"""
    return torch.matmul(a, b)


def scorch_sddmm(b, c, d):
    """
    B is sparse, e.g. CSR or COO
    C and D are dense
    """
    return scorch.einsum("ij,ik,kj->ij", b, c, d)


def torch_sddmm(b, c, d):
    """SDDMM with PyTorch"""
    raise NotImplementedError("PyTorch SDDMM not implemented.")


def scorch_sddmm_dense(b, c, d):
    """
    B is sparse, e.g. CSR or COO
    C and D are dense
    """
    return scorch.einsum("ij,ik,kj->ij", b, c, d, format="dd")


def torch_sddmm_dense(b, c, d):
    """
    B is sparse, e.g. CSR or COO
    C and D are dense
    """
    b_dense = b.to_dense()
    return torch.einsum("ij,ik,kj->ij", b_dense, c, d)
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
    gen_rand_sparse_func = (
        gen_rand_sparse_coo if sparse_format == "coo" else gen_rand_sparse_csr
    )

    if output_format == "dense":
        torch_sddmm_func = torch_sddmm_dense
        scorch_sddmm_func = scorch_sddmm_dense

    for dim in dimensions:
        torch_time_run = []
        scorch_time_run = []
        speedup_run = []

        for _ in range(num_runs):
            B, B_torch = gen_rand_sparse_func(dim, dim, sparsity)

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


def bench_spmspm(
    dimensions,
    num_runs=5,
    sparsity=0.99,
    sparse_format="csr",
    custom_func=None,
):
    torch_times = []
    scorch_times = []
    speedups_means = []
    speedups_stds = []

    scorch_spmspm_func = scorch_spmm
    if custom_func is not None:
        scorch_spmspm_func = custom_func

    gen_rand_sparse_func = (
        gen_rand_sparse_coo if sparse_format == "coo" else gen_rand_sparse_csr
    )

    for dim in dimensions:
        torch_time_run = []
        scorch_time_run = []
        speedup_run = []

        for _ in range(num_runs):
            A, A_torch = gen_rand_sparse_func(dim, dim, sparsity)
            B, B_torch = gen_rand_sparse_func(dim, dim, sparsity)

            # Warm up
            for _ in range(1):
                scorch_spmspm_func(A, B)

            start = time.perf_counter()
            scorch_result = scorch_spmspm_func(A, B)
            end = time.perf_counter()
            scorch_time = end - start
            scorch_time_run.append(scorch_time)

            # Warm up
            for _ in range(1):
                torch_spmm(A_torch, B_torch)

            start = time.perf_counter()
            torch_result = torch_spmm(A_torch, B_torch)
            end = time.perf_counter()
            torch_time = end - start
            torch_time_run.append(torch_time)

            # if not torch.allclose(torch_result.values().flatten(), scorch_result.values.flatten()):
            #     print("\n\n\n")
            #     print("torch_result.to_dense():")
            #     print(torch_result.to_dense())
            #     print("scorch_result.values.flatten():")
            #     print(scorch_result.values.flatten())
            #     print("A_torch.to_dense():")
            #     print(A_torch.to_dense())
            #     print("B_torch.to_dense():")
            #     print(B_torch.to_dense())
            #     import pdb

            #     pdb.set_trace()

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

    gen_rand_sparse_func = (
        gen_rand_sparse_coo if sparse_format == "coo" else gen_rand_sparse_csr
    )

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

            assert torch.allclose(
                torch_result.flatten(), scorch_result.values.flatten()
            )

            speedup_run.append(torch_time / scorch_time)

        torch_times.append(np.mean(torch_time_run))
        scorch_times.append(np.mean(scorch_time_run))
        speedups_means.append(np.mean(speedup_run))
        speedups_stds.append(np.std(speedup_run))

    return torch_times, scorch_times, speedups_means, speedups_stds


def bench_spmv(dimensions, num_runs=5, sparsity=0.99):
    torch_times = []
    scorch_times = []
    speedups_means = []
    speedups_stds = []

    for dim_mn in dimensions:
        torch_time_run = []
        scorch_time_run = []
        speedup_run = []

        for _ in range(num_runs):
            A, A_torch = gen_rand_sparse_coo(dim_mn, dim_mn, sparsity)

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

    # Plot mean speedup of Scorch over PyTorch
    plt.plot(
        dimensions,
        speedup_means,
        "-o",
        label="Mean speedup (pytorch time / scorch time)",
    )

    # Add shaded area for standard deviation
    plt.fill_between(dimensions, speedup_lower, speedup_upper, color="blue", alpha=0.2)

    plt.title(f"{benchmark} Speedup of Scorch over PyTorch")
    plt.xlabel("Matrix dimensions")
    plt.ylabel("Speedup factor")
    plt.legend()
    plt.grid(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name_with_timestamp = f"figures/{benchmark.lower()}-{timestamp}.png"

    # Mkdir if not exists
    os.makedirs("figures", exist_ok=True)

    plt.savefig(file_name_with_timestamp, dpi=300)

    # Also plot Mean speedup (PyTorch over Scorch)
    plt.figure(figsize=(10, 6))

    speedup_means = 1 / np.array(speedup_means)
    speedup_upper = 1 / speedup_upper
    speedup_lower = 1 / speedup_lower

    # Plot mean speedup
    plt.plot(
        dimensions,
        speedup_means,
        "-o",
        label="Mean speedup (scorch time / pytorch time)",
    )

    # Add shaded area for standard deviation
    plt.fill_between(dimensions, speedup_lower, speedup_upper, color="blue", alpha=0.2)

    plt.title(f"{benchmark} Speedup of PyTorch over Scorch")
    plt.xlabel("Matrix dimensions")
    plt.ylabel("Speedup factor")
    print(f"\nBenchmark: {benchmark}")
    print("speedup_means:", speedup_means)
    plt.legend()
    plt.grid(True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name_with_timestamp = f"figures/{benchmark.lower()}-inverse-{timestamp}.png"

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
            # 128,
            # 512,
            1024,
            # 1536,
            2048,
            # 2560,
            # 3072,
            # 4096,
            # 6144,
            # 5000,
            # 10240,
            # 20480,
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
    # torch_times, scorch_times, speedups_means, speedups_stds = bench_spmv(
    #     dimensions,
    #     sparsity=0.80,
    # )
    # plot_benchmark(dimensions, speedups_means, speedups_stds, benchmark="SpMV 80% (COO)")

    # torch_times, scorch_times, speedups_means, speedups_stds = bench_spmm(
    #     dimensions,
    #     num_runs=10,
    #     sparsity=0.50,
    #     sparse_format="csr",
    #     custom_func=scorch_spmm_custom,
    # )
    # plot_benchmark(
    #     dimensions, speedups_means, speedups_stds, benchmark="SpMM 50% (CSR) (tilesize=512, restrict, unroll, likely, torchtensor)"
    #     # dimensions, speedups_means, speedups_stds, benchmark="SpMM 65% (CSR)"
    # )

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

    torch_times, scorch_times, speedups_means, speedups_stds = bench_spmspm(
        dimensions,
        sparsity=0.5,
        sparse_format="csr",
        custom_func=scorch_spmspm_custom,
    )
    plot_benchmark(
        dimensions,
        speedups_means,
        speedups_stds,
        # benchmark="SpMSpM 50% (CSR) TACO (c1_pos_cvector, c1_crd_cvector, c_val_cvector)",
        benchmark="SpMSpM 50% (CSR) Scorch custom (w scorch v2 workspace)",
    )

    # torch_times, scorch_times, speedups_means, speedups_stds = bench_sddmm(
    #     dimensions,
    #     sparsity=0.80,
    #     sparse_format="csr",
    #     output_format="dense",
    # )
    # plot_benchmark(
    #     dimensions, speedups_means, speedups_stds, benchmark="SDDMM 80% (CSR, Dense output)"
    # )

    # torch_times, scorch_times, speedups_means, speedups_stds = bench_sddmm(
    #     dimensions,
    #     sparsity=0.99,
    #     sparse_format="csr",
    # )
    # plot_benchmark(dimensions, speedups_means, speedups_stds, benchmark="SDDMM (CSR)")
