import time

import torch

from scorch import Tensor, matmul
from scorch.compiler.cin import IndexVar, ForAll, TensorAssign, TensorVar, Operation
from scorch.compiler.scheduler import Scheduler


def test_insert_dense_workspace():
    C = TensorVar("C", fmt="dd")
    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="dd")

    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    cin_stmt = ForAll(
        i,
        ForAll(
            j, ForAll(k, TensorAssign(C[i, k], A[i, j] * B[j, k], op=Operation.ADD))
        ),
    )

    scheduler = Scheduler()

    new_cin = scheduler.insert_workspace(cin_stmt)

    print(new_cin)


def test_add_tile():
    C = TensorVar("C", fmt="dd")
    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="dd")

    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    cin_stmt = ForAll(
        i,
        ForAll(
            j, ForAll(k, TensorAssign(C[i, k], A[i, j] * B[j, k], op=Operation.ADD))
        ),
    )

    scheduler = Scheduler()

    new_cin = scheduler.add_tile(cin_stmt, k, 1024)

    print(new_cin)


def test_spmm_dd_oo_dd_smaller_than_tilesize():
    """
    Compare speed of torch and scorch matmul
    Use random tensors
    """
    n = 1000
    sparsity = 0.99
    random_tensor_a = torch.rand(n, n).float()
    random_tensor_b = torch.rand(n, n).float()

    # Randomly sparsify each tensor
    random_tensor_a = random_tensor_a * (torch.rand(n, n) > sparsity)
    random_tensor_b = random_tensor_b * (torch.rand(n, n) > sparsity)

    # Convert random_tensor_a to a sparse COO pytorch tensor
    random_tensor_a_coo = random_tensor_a.to_sparse()
    start_time = time.time()
    torch_result = torch.matmul(random_tensor_a, random_tensor_b)
    # torch_result = torch.sparse.mm(random_tensor_a_coo, random_tensor_b)
    torch_time = time.time() - start_time

    tensor_a_scorch = Tensor.from_torch(random_tensor_a, "A").to_sparse("oo")
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
