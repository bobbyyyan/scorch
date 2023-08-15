# Benchmark runtime of PyTorch sparse

import torch
import time


def sparse_mm(A, B):
    """Sparse matrix multiplication with PyTorch"""
    return torch.sparse.mm(A, B)


def gen_rand_sparse_coo(dim_m, dim_n, sparsity):
    """Generate sparse matrix with PyTorch
    sparsity is the percentage of zero elements
    """
    density = 1 - sparsity
    A = torch.rand(dim_m, dim_n)
    A[A > density] = 0
    A = A.to_sparse_coo()
    return A


# Create sparse matrix
dim_m = 10000
dim_n = dim_m
dim_k = dim_m

if __name__ == "__main__":
    # Create random sparse COO matrix A with 99% sparsity, COO format
    A = gen_rand_sparse_coo(dim_m, dim_n, 0.99)

    print("A's nnz count: ", A._nnz())
    # to 2 decimal places
    print(f"A's sparsity level: {100 - A._nnz() / (dim_m * dim_n) * 100:.2f}%")

    # Create random dense matrix B
    B = torch.rand(dim_n, dim_k)

    # Time matmul
    start = time.time()

    C = torch.sparse.mm(A, B)
    # C = torch.matmul(A, B)

    end = time.time()
    print("Time: ", end - start)
