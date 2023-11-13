import time

import torch

from scorch import Tensor, matmul, einsum
from scorch.ops import spmv


def test_spmm_dd_ds_dd_time():
    n = 10000
    sparsity = 0.95
    random_tensor_a = torch.rand(n, n).float()
    random_tensor_b = torch.rand(n, n).float()

    # Randomly sparsify each tensor
    random_tensor_a = random_tensor_a * (torch.rand(n, n) > sparsity)
    random_tensor_b = random_tensor_b * (torch.rand(n, n) > sparsity)

    random_tensor_a_csr = random_tensor_a.to_sparse_csr()
    # Convert random_tensor_a to a sparse CSR pytorch tensor
    start_time = time.time()
    # torch_result = torch.matmul(random_tensor_a, random_tensor_b).to_sparse_coo()
    torch_result = torch.sparse.mm(random_tensor_a_csr, random_tensor_b)
    torch_time = time.time() - start_time

    tensor_a_scorch = Tensor.from_torch(random_tensor_a, "A").to_sparse("ds")
    tensor_b_scorch = Tensor.from_torch(random_tensor_b, "B")

    time_dict = {}
    start_time = time.time()
    scorch_result = matmul(
        tensor_a_scorch, tensor_b_scorch, format="dd", time_dict=time_dict
    )
    scorch_total_time = time.time() - start_time
    scorch_eval_time = time_dict["eval_time"]

    # Assert that the results are the same
    assert torch.allclose(torch_result, scorch_result.to_torch())

    print(f"torch time: {torch_time}")
    print(f"scorch total time: {scorch_total_time}")
    print(f"scorch eval time: {scorch_eval_time}")
    print(f"scorch eval time / torch time: {scorch_eval_time / torch_time}")


def test_spmv_d_oo_d_time():
    """
    Compare speed of torch and scorch Sparse matrix * Dense vector
    Use random tensors
    """
    # y[i] = sum_j A[i, j] * x[j]
    # Randomly generate sparse matrix A, which is m by n
    m = 10000
    n = 10000
    sparsity = 0.9
    random_tensor_a = torch.rand(m, n)
    random_tensor_x = torch.rand(n)
    # Sparsify A
    random_tensor_a = random_tensor_a * (torch.rand(m, n) > sparsity).float()
    random_tensor_a_sparse = random_tensor_a.to_sparse_coo()

    start_time = time.time()
    torch_result = torch.matmul(random_tensor_a_sparse, random_tensor_x)
    torch_time = time.time() - start_time

    tensor_a_scorch = Tensor.from_torch(random_tensor_a, "A").to_sparse("oo")
    tensor_x_scorch = Tensor.from_torch(random_tensor_x, "x")

    time_dict = {}
    start_time = time.time()
    # scorch_result = einsum(
    #     "ij,j->i", tensor_a_scorch, tensor_x_scorch, time_dict=time_dict
    # )
    scorch_result = matmul(tensor_a_scorch, tensor_x_scorch, time_dict=time_dict)
    scorch_total_time = time.time() - start_time
    scorch_eval_time = time_dict["eval_time"]

    assert torch.allclose(torch_result, scorch_result.values)

    print(f"torch time: {torch_time}")
    print(f"scorch total time: {scorch_total_time}")
    print(f"scorch eval time: {scorch_eval_time}")
    print(f"scorch eval time / torch time: {scorch_eval_time / torch_time}")


def test_sddmm_dd_ds_dd_dd_time():
    """
    A[i, j] = B[i, j] * C[i, k] * D[k, j]
    A: Dense
    B: CSR
    C: Dense
    D: Dense
    """
    N = 1000
    sparsity = 0.9
    random_tensor_b = torch.rand(N, N)
    random_tensor_c = torch.rand(N, N)
    random_tensor_d = torch.rand(N, N)
    # Sparsify B
    random_tensor_b = random_tensor_b * (torch.rand(N, N) > sparsity).float()
    random_tensor_b_sparse = random_tensor_b.to_sparse_csr()

    start_time = time.time()
    torch_result = torch.einsum(
        "ij,ik,kj->ij", random_tensor_b, random_tensor_c, random_tensor_d
    )
    # torch_result = torch.sparse.sampled_addmm(
    #     random_tensor_b_sparse, random_tensor_c, random_tensor_d, beta=0
    # )
    torch_time = time.time() - start_time

    tensor_b_scorch = Tensor.from_torch(random_tensor_b, "B").to_sparse("ds")
    tensor_c_scorch = Tensor.from_torch(random_tensor_c, "C")
    tensor_d_scorch = Tensor.from_torch(random_tensor_d, "D")

    time_dict = {}
    start_time = time.time()
    scorch_result = einsum(
        "ij,ik,kj->ij",
        tensor_b_scorch,
        tensor_c_scorch,
        tensor_d_scorch,
        time_dict=time_dict,
        format="dd",
    )
    scorch_total_time = time.time() - start_time
    scorch_eval_time = time_dict["eval_time"]

    assert torch.allclose(torch_result, scorch_result.to_torch())

    print(f"torch time: {torch_time}")
    print(f"scorch total time: {scorch_total_time}")
    print(f"scorch eval time: {scorch_eval_time}")
    print(f"scorch eval time / torch time: {scorch_eval_time / torch_time}")
