#!/usr/bin/env python3
"""
Measure Scorch autoscheduling times for different sparse kernels.

This script measures the time spent in:
1. Autoscheduling (loop ordering, tiling, format inference)
2. CIN lowering
3. LLIR lowering (code generation)
4. C++ compilation

The "autoscheduling time" reported is the sum of steps 1-3 (excluding C++ compilation).
"""

import torch
import numpy as np
import scorch
from scipy.sparse import random as sparse_random
import sys

# Clear the kernel cache to ensure we measure first-time compilation
from scorch.ops import _kernel_cache
_kernel_cache.clear()


def scipy_sparse_to_torch_sparse(matrix, format='csr'):
    """Convert scipy sparse matrix to torch sparse tensor."""
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


def measure_spmm():
    """Measure SpMM (Sparse Matrix-Dense Matrix Multiplication)."""
    print("\\nMeasuring SpMM...")

    # Create a small sparse matrix and dense matrix using COO format to avoid cached kernels
    M, N, K = 100, 100, 32
    density = 0.01

    sparse_matrix = sparse_random(M, N, density=density, format='coo', dtype=np.float32)
    torch_sparse_matrix = scipy_sparse_to_torch_sparse(sparse_matrix, format='coo')
    dense_matrix = torch.rand((N, K), dtype=torch.float32)

    # Clear cache before measurement
    _kernel_cache.clear()

    # Use einsum directly to ensure we go through the autoscheduling path
    time_dict = {}
    result = scorch.einsum("ik,kj->ij", torch_sparse_matrix, dense_matrix, time_dict=time_dict)

    return time_dict


def measure_spmv():
    """Measure SpMV (Sparse Matrix-Vector Multiplication)."""
    print("\\nMeasuring SpMV...")

    # Create a small sparse matrix and dense vector
    # Note: SpMV uses a specialized spmv function that doesn't go through einsum
    # So timing will be measured differently
    M, N = 100, 100
    density = 0.01

    sparse_matrix = sparse_random(M, N, density=density, format='csr', dtype=np.float32)
    torch_sparse_matrix = scipy_sparse_to_torch_sparse(sparse_matrix, format='csr')
    dense_vector = torch.rand((N,), dtype=torch.float32)

    # Clear cache before measurement
    _kernel_cache.clear()

    # Use spmv function directly - but note it doesn't use einsum so won't have timing
    # Instead we'll manually time the key phases
    import time as time_module
    from scorch.compiler.cin_lowerer import CINLowerer
    from scorch.compiler.codegen import LLIRLowerer
    from scorch.stensor import STensor

    # Convert to STensor
    a = STensor.from_torch(torch_sparse_matrix)
    b = STensor.from_torch(dense_vector)

    # Manually create the CIN for SpMV and measure phases
    from scorch.compiler.cin import IndexVar, TensorVar, ForAll, Workspace, Where, TensorAssign
    from scorch.compiler.scheduler import Scheduler
    from scorch.utils import parse_format

    y = TensorVar("y", fmt=parse_format("d"))
    A = TensorVar("A", fmt=a.format)
    x = TensorVar("x", fmt=b.format)

    i = IndexVar("i")
    j = IndexVar("j")

    workspace = Workspace(name="wksp", dim=0)

    from scorch.compiler.cin import Operation
    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                j,
                TensorAssign(
                    workspace.get_default_access(), A[i, j] * x[j], op=Operation.ADD
                ),
            ),
            consumer=TensorAssign(y[i], workspace.get_default_access()),
        ),
    )

    time_dict = {}

    # Time auto-scheduling
    start = time_module.time()
    cin_stmt = Scheduler.auto_schedule(cin_stmt)
    time_dict["autoschedule_time"] = time_module.time() - start

    # Time CIN lowering
    start = time_module.time()
    lowerer = CINLowerer()
    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
    time_dict["cin_lowering_time"] = time_module.time() - start

    # Time LLIR lowering
    start = time_module.time()
    llir_lowerer = LLIRLowerer()
    cpp_code = llir_lowerer.lower_llir(lowered_llir)
    time_dict["llir_lowering_time"] = time_module.time() - start

    # Time C++ compilation
    from pathlib import Path
    import os
    PROJECT_ROOT_DIR = Path(__file__).parent
    # Find the csrc directory
    if not (PROJECT_ROOT_DIR / "csrc/header.cpp").exists():
        PROJECT_ROOT_DIR = PROJECT_ROOT_DIR / "src" / "scorch"
        if not (PROJECT_ROOT_DIR.parent.parent / "csrc/header.cpp").exists():
            # Try going up to find it
            curr = PROJECT_ROOT_DIR
            while not (curr / "csrc/header.cpp").exists() and curr != curr.parent:
                curr = curr.parent
            PROJECT_ROOT_DIR = curr

    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    start = time_module.time()
    module = torch.utils.cpp_extension.load_inline(
        name="kernel_spmv",
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=["-O3"],
    )
    time_dict["cpp_compilation_time"] = time_module.time() - start
    time_dict["cache_hit"] = False

    return time_dict


def measure_sddmm():
    """Measure SDDMM (Sampled Dense-Dense Matrix Multiplication)."""
    print("\\nMeasuring SDDMM...")

    # Create a sparse mask and two dense matrices
    M, N = 100, 100
    density = 0.01

    sparse_matrix = sparse_random(M, N, density=density, format='coo', dtype=np.float32)
    torch_sparse_matrix = scipy_sparse_to_torch_sparse(sparse_matrix, format='coo')
    dense_matrix_A = torch.rand((M, 1), dtype=torch.float32)
    dense_matrix_B = torch.rand((1, N), dtype=torch.float32)

    # Clear cache before measurement
    _kernel_cache.clear()

    # Call einsum for SDDMM with time_dict to capture timing
    time_dict = {}
    result = scorch.einsum("ij,ik,kj->ij", torch_sparse_matrix, dense_matrix_A, dense_matrix_B, time_dict=time_dict)

    return time_dict


def measure_spmspm():
    """Measure SpMSpM (Sparse Matrix-Sparse Matrix Multiplication, also known as SpGEMM)."""
    print("\\nMeasuring SpMSpM (SpGEMM)...")

    # Create two small sparse matrices
    M, N = 100, 100
    density = 0.01

    sparse_matrix1 = sparse_random(M, N, density=density, format='coo', dtype=np.float32)
    torch_sparse_matrix1 = scipy_sparse_to_torch_sparse(sparse_matrix1, format='coo')

    sparse_matrix2 = sparse_random(N, M, density=density, format='coo', dtype=np.float32)
    torch_sparse_matrix2 = scipy_sparse_to_torch_sparse(sparse_matrix2, format='coo')

    # Clear cache before measurement
    _kernel_cache.clear()

    # Call matmul with time_dict to capture timing
    time_dict = {}
    result = scorch.matmul(torch_sparse_matrix1, torch_sparse_matrix2, time_dict=time_dict, use_cache=False)

    return time_dict


def print_results_table(results):
    """Print results in a nicely formatted table."""
    print("\\n" + "="*100)
    print("AUTOSCHEDULING TIME MEASUREMENTS")
    print("="*100)
    print()

    # Calculate totals
    for kernel_name, time_dict in results.items():
        if "cache_hit" in time_dict and time_dict["cache_hit"]:
            print(f"Warning: {kernel_name} used cached kernel. Results may not be accurate.")
            continue

        # Autoscheduling time = auto_schedule + CIN lowering + LLIR lowering (codegen)
        autoschedule = time_dict.get("autoschedule_time", 0) * 1000  # Convert to ms
        cin_lowering = time_dict.get("cin_lowering_time", 0) * 1000
        llir_lowering = time_dict.get("llir_lowering_time", 0) * 1000
        cpp_compilation = time_dict.get("cpp_compilation_time", 0) * 1000

        autoscheduling_total = autoschedule + cin_lowering + llir_lowering
        total = autoscheduling_total + cpp_compilation

        time_dict["autoscheduling_total_ms"] = autoscheduling_total
        time_dict["cpp_compilation_ms"] = cpp_compilation
        time_dict["total_ms"] = total

    # Print table
    print(f"{'Kernel':<15} {'Autoschedule':>15} {'CIN Lower':>15} {'LLIR Lower':>15} {'Autosch. Total':>17} {'C++ Compile':>15} {'Total':>15}")
    print(f"{'':15} {'(ms)':>15} {'(ms)':>15} {'(ms)':>15} {'(ms)':>17} {'(ms)':>15} {'(ms)':>15}")
    print("-" * 122)

    for kernel_name, time_dict in results.items():
        if "cache_hit" in time_dict and time_dict["cache_hit"]:
            continue

        autoschedule = time_dict.get("autoschedule_time", 0) * 1000
        cin_lowering = time_dict.get("cin_lowering_time", 0) * 1000
        llir_lowering = time_dict.get("llir_lowering_time", 0) * 1000
        autoscheduling_total = time_dict["autoscheduling_total_ms"]
        cpp_compilation = time_dict["cpp_compilation_ms"]
        total = time_dict["total_ms"]

        print(f"{kernel_name:<15} {autoschedule:>15.2f} {cin_lowering:>15.2f} {llir_lowering:>15.2f} {autoscheduling_total:>17.2f} {cpp_compilation:>15.2f} {total:>15.2f}")

    print()
    print("Note: 'Autosch. Total' = Autoschedule + CIN Lower + LLIR Lower")
    print("      This is the autoscheduling time (excluding C++ compilation)")
    print()


def main():
    print("Measuring autoscheduling times for Scorch sparse kernels...")
    print("This will compile each kernel once and measure the time breakdown.")

    results = {}

    # Measure each kernel
    try:
        results["SpMM"] = measure_spmm()
    except Exception as e:
        print(f"Error measuring SpMM: {e}")
        import traceback
        traceback.print_exc()

    try:
        results["SpMV"] = measure_spmv()
    except Exception as e:
        print(f"Error measuring SpMV: {e}")
        import traceback
        traceback.print_exc()

    try:
        results["SDDMM"] = measure_sddmm()
    except Exception as e:
        print(f"Error measuring SDDMM: {e}")
        import traceback
        traceback.print_exc()

    try:
        results["SpGEMM"] = measure_spmspm()
    except Exception as e:
        print(f"Error measuring SpGEMM: {e}")
        import traceback
        traceback.print_exc()

    # Print results
    print_results_table(results)

    print("\\nFor the rebuttal, use the 'Autosch. Total' column values.")


if __name__ == "__main__":
    main()
