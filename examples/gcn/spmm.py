# Benchmark runtime of Scorch

import torch
import time
import scorch


def scorch_spmm(A, B):
    """Sparse matrix multiplication with Scorch"""
    return scorch.matmul(A, B)


def torch_spmm(A, B):
    """Sparse matrix multiplication with PyTorch"""
    return torch.sparse.mm(A, B)


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

    A = scorch.from_coo(indices, values, (dim_m, dim_n))

    return A


def gen_rand_sparse_csr(dim_m, dim_n, sparsity):
    A = gen_rand_sparse_coo(dim_m, dim_n, sparsity)
    A = A.to_sparse("ds")
    return A


# Create sparse matrix
dim_m = 5000
dim_n = dim_m
dim_k = dim_m

if __name__ == "__main__":
    scorch_times = []
    torch_times = []

    # Warm up
    # for _ in range(5):
    #     scorch_spmm(A, B)
    #     torch_spmm(A_torch, B)

    for _ in range(5):
        # A = gen_rand_sparse_coo(dim_m, dim_n, 0.99)
        # A_torch = A.to_torch(in_place=False).to_sparse_coo()

        A = gen_rand_sparse_csr(dim_m, dim_n, 0.99)
        A_torch = A.to_torch(in_place=False)
        A_torch = A_torch.to_sparse_csr()

        B = torch.rand(dim_n, dim_k)

        start = time.perf_counter()
        C = scorch_spmm(A, B)
        end = time.perf_counter()
        scorch_times.append(end - start)

        start = time.perf_counter()
        C = torch_spmm(A_torch, B)
        end = time.perf_counter()
        torch_times.append(end - start)

        # assert torch.allclose(C.to_torch(), torch.matmul(A_torch, B))

    print("dim_m: ", dim_m)
    print("A's nnz count: ", A._nnz())
    print(f"A's sparsity level: {100 - A._nnz() / (dim_m * dim_n) * 100:.2f}%")

    print("Scorch times: \n", scorch_times)
    print("PyTorch times: \n", torch_times)
    print(
        "PyTorch/Scorch: \n",
        [torch_times[i] / scorch_times[i] for i in range(len(scorch_times))],
    )
