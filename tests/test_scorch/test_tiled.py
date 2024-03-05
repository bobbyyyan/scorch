import time

import torch

from scorch import Tensor, matmul, einsum
from scorch.compiler.cin import (
    TensorAssign,
    ForAll,
    Operation,
    Where,
    Workspace,
    TensorVar,
    IndexVar,
    TileSizeVar,
)
from scorch.ops import spmv, lower_and_exec_cin

# copied from tests/test_scorch/test_perf.py
def test_spmm_dd_ds_dd_tiled_time():
    n = 4096
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

    i = IndexVar("i")
    j = IndexVar("j")
    k_out = IndexVar("k_out")
    k_in = IndexVar("k_in")
    k = IndexVar("k", k_out + k_in)

    k_tile_size = 1024
    k_tile_var = TileSizeVar(
        outer_index_var=k_out, inner_index_var=k_in, size=k_tile_size
    )

    C = TensorVar("C", fmt="dd")
    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="dd")

    # accum_c = TensorVar("accum_c", fmt="d")
    accum_c = Workspace("accum_c", dim=1, dense=True, tile_size_var=k_tile_var)

    cin_stmt = ForAll(
        i,
        ForAll(
            k_out,
            Where(
                producer=ForAll(
                    j,
                    ForAll(
                        k_in,
                        TensorAssign(
                            accum_c[k_in],
                            A[i, j] * B[j, k],
                            op=Operation.ADD,
                        ),
                    ),
                ),
                consumer=ForAll(
                    k_in,
                    TensorAssign(
                        C[i, k],
                        accum_c[k_in],
                    ),
                ),
            ),
        ),
    )

    result_shape = (n, n)

    time_dict = {}
    start_time = time.time()
    scorch_result = lower_and_exec_cin(
        cin_stmt, result_shape, tensor_a_scorch, tensor_b_scorch, time_dict=time_dict
    )
    scorch_total_time = time.time() - start_time
    scorch_eval_time = time_dict["eval_time"]

    # Assert that the results are the same
    assert torch.allclose(torch_result, scorch_result.to_torch())

    print(f"torch time: {torch_time}")
    print(f"scorch total time: {scorch_total_time}")
    print(f"scorch eval time: {scorch_eval_time}")
    print(f"scorch eval time / torch time: {scorch_eval_time / torch_time}")

def test_spmm_dd_ds_dd_time():
    n = 64
    sparsity = 0.8
    random_tensor_a = torch.rand(n, n).float()
    random_tensor_b = torch.rand(n, n).float()

    # Randomly sparsify each tensor
    random_tensor_a = random_tensor_a * (torch.rand(n, n) > sparsity)
    random_tensor_b = random_tensor_b * (torch.rand(n, n) > sparsity)

    random_tensor_a_csr = random_tensor_a.to_sparse_csr()

    start_time = time.time()
    torch_result = torch.matmul(random_tensor_a_csr, random_tensor_b)
    # torch_result = torch.sparse.mm(random_tensor_a_csr, random_tensor_b)
    torch_time = time.time() - start_time

    tensor_a_scorch = Tensor.from_torch(random_tensor_a, "A").to_sparse([["d", "3"], ["s", "2"]])
    tensor_b_scorch = Tensor.from_torch(random_tensor_b, "B")

    time_dict = {}
    start_time = time.time()
    scorch_result = matmul(
        tensor_a_scorch, tensor_b_scorch, format="dd", time_dict=time_dict
    )
    scorch_total_time = time.time() - start_time
    scorch_eval_time = time_dict["eval_time"]
    scorch_result_torch = scorch_result.to_torch()

    print(f"torch time: {torch_time}")
    print(f"scorch total time: {scorch_total_time}")
    print(f"scorch eval time: {scorch_eval_time}")
    print(f"scorch eval time / torch time: {scorch_eval_time / torch_time}")

    assert torch.allclose(torch_result, scorch_result_torch)

test_spmm_dd_ds_dd_time()