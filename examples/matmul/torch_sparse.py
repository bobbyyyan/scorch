# Benchmark runtime of PyTorch sparse

import torch
import time


def sparse_mm(A, B):
    """Sparse matrix multiplication with PyTorch"""
    return torch.sparse.mm(A, B)


def gen_rand_sparse_coo_v1(dim_m, dim_n, sparsity):
    """Generate sparse matrix with PyTorch
    sparsity is the percentage of zero elements
    """
    density = 1 - sparsity
    A = torch.rand(dim_m, dim_n)
    A[A > density] = 0
    A = A.to_sparse_coo()
    return A


def gen_rand_sparse_coo(dim_m, dim_n, sparsity):
    """Generate sparse matrix with PyTorch
    sparsity is the percentage of zero elements
    """
    nnz = int((1 - sparsity) * dim_m * dim_n)
    indices = torch.randint(0, dim_m, size=(2, nnz))
    values = torch.rand(nnz)

    _, inverse_indices = torch.unique(indices, dim=1, return_inverse=True)
    while len(inverse_indices) != len(indices.transpose(0, 1)):
        indices = torch.randint(0, dim_m, size=(2, nnz))
        _, inverse_indices = torch.unique(indices, dim=1, return_inverse=True)

    A = torch.sparse_coo_tensor(indices, values, (dim_m, dim_n))
    return A


# Create sparse matrix
dim_m = 15000
dim_n = dim_m
dim_k = dim_m

if __name__ == "__main__":
    A = gen_rand_sparse_coo(dim_m, dim_n, 0.99)
    print("A's nnz count: ", A._nnz())
    print(f"A's sparsity level: {100 - A._nnz() / (dim_m * dim_n) * 100:.2f}%")

    B = torch.rand(dim_n, dim_k)

    runtimes = []

    # Warm up
    for _ in range(5):
        torch.sparse.mm(A, B)

    for _ in range(5):
        start = time.time()
        C = torch.sparse.mm(A, B)
        end = time.time()
        runtimes.append(end - start)

    print("Times: \n", runtimes)
