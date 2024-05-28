# pretty print
import pprint
import time
from itertools import product
from typing import List, Dict, Any, Iterable, Union, Optional


import torch
from torch.utils.cpp_extension import load_inline

from scorch import STensor, einsum, utils, TensorFormat
from scorch.compiler.cin import (
    ForAll,
    Where,
    TensorAssign,
    Operation,
    IndexVar,
    TensorVar,
    Workspace,
    TileSizeVar,
)
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.format import LevelFormat, LevelType
from scorch.ops import matmul, matmul_wksp, lower_and_exec_cin
from scorch.storage import TensorIndex, TensorStorage
from scorch.utils import PROJECT_ROOT_DIR, parse_format

# indent 2 pretty print
pp = pprint.PrettyPrinter(indent=2)
import pdb


def generate_2d_tensors(
    a_fmt: str,
    b_fmt: str,
    result_fmt: str,
    a_mode_order: Optional[List[int]] = None,
    b_mode_order: Optional[List[int]] = None,
    result_mode_order: Optional[List[int]] = None
):
    """
    Generates tensors A, B, and result given their format
        - values for tensors are hard-coded for debugging purposes
    Args:
        formats can be "csc", "csr", etc. for now, only csc & csr supported
    """
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    tensor_b_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 3, 1],
        ]
    )

    tensor_result_add_torch = torch.Tensor(
        [
            [2, 2, 3, 4, 5],
            [2, 4, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 1, 0],
            [5, 0, 0, 3, 6],
        ]
    )

    tensor_result_elemwise_mul_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 4, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
        ]
    )

    tensor_result_matmul_torch = torch.Tensor(
        [
            [1, 4, 0, 19, 5],
            [2, 4, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 15, 5],
        ]
    )

    a = STensor.from_torch(tensor_a_torch, "a", a_mode_order).to_sparse(a_fmt)
    b = STensor.from_torch(tensor_b_torch, "b", b_mode_order).to_sparse(b_fmt)
    result_add = STensor.from_torch(tensor_result_add_torch, "result_add", result_mode_order).to_sparse(result_fmt)
    result_elemwise_mul = STensor.from_torch(tensor_result_elemwise_mul_torch, "result_elemwise_mul", result_mode_order).to_sparse(result_fmt)
    result_matmul = STensor.from_torch(tensor_result_matmul_torch, "result_matmul", result_mode_order).to_sparse(result_fmt)

    return a, b, result_add, result_elemwise_mul, result_matmul


def test_generate_tensor_3d():
    tensor_result_torch = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 2],
                [3, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 4, 0],
                [5, 6, 0],
                [0, 0, 0],
            ]
        ]
    )

    # result_3d_normal_ddd = Tensor.from_torch(tensor_result_torch, "3d_normal")
    result_3d_normal_dss = STensor.from_torch(tensor_result_torch, "3d_normal").to_sparse("dss")
    assert(result_3d_normal_dss._storage._value.tolist() == [1., 1., 1., 1., 1., 1.])
    # result_3d_reverse_dss = Tensor.from_torch(tensor_result_torch, "3d_reverse", [2, 1, 0]).to_sparse("dss")
    pdb.set_trace()

def test_generate_tensor_3d_kernel():
    tensor_result_torch = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 0],
            ]
        ]
    )

    a = STensor.from_torch(tensor_result_torch, "3d_normal")
    # shape doesn't change, can pass in input shape
    result_cpp = test_custom_kernel([a], a.shape, "tensor_assign_3d.cpp")
    assert result_cpp._storage._value.tolist() == [1., 1., 1., 1., 1., 1.]
    assert result_cpp._storage._index.mode_indices[1][0].tolist() == [0, 3, 3, 5]
    assert result_cpp._storage._index.mode_indices[1][1].tolist() == [0, 1, 2, 0, 1]
    assert result_cpp._storage._index.mode_indices[2][0].tolist() == [0, 1, 2, 3, 4, 6]
    assert result_cpp._storage._index.mode_indices[2][1].tolist() == [0, 2, 0, 1, 0, 1]

def test_spmm_csr_csr_csr():
    a, b, _, _, result = generate_2d_tensors("ds", "ds", "ds")
    result_cpp = test_custom_kernel([a, b], result.shape, "spmm_csr_wksp.cpp")

    assert result_cpp._storage._value.tolist() == result._storage._value.tolist(), "Values are different"
    assert result_cpp._storage._index.mode_indices[0] == result._storage._index.mode_indices[0]
    assert result_cpp._storage._index.mode_indices[1][0].tolist() == result._storage._index.mode_indices[1][0].tolist()
    assert result_cpp._storage._index.mode_indices[1][1].tolist() == result._storage._index.mode_indices[1][1].tolist()


def test_spmm_csc_csc_csc():
    a, b, _, _, result = generate_2d_tensors("ds", "ds", "ds", [1, 0], [1, 0], [1, 0])
    result_cpp = test_custom_kernel([a, b], result.shape, "spmm_csc_wksp.cpp")

    assert result_cpp._storage._value.tolist() == result._storage._value.tolist(), "Values are different"
    assert result_cpp._storage._index.mode_indices[0] == result._storage._index.mode_indices[0]
    assert result_cpp._storage._index.mode_indices[1][0].tolist() == result._storage._index.mode_indices[1][0].tolist()
    assert result_cpp._storage._index.mode_indices[1][1].tolist() == result._storage._index.mode_indices[1][1].tolist()


def test_sparse_to_dense_csc():
    a, _, _, _, _ = generate_2d_tensors("ds", "ds", "ds", [1, 0], [1, 0], [1, 0])
    a_dense = a.to_dense()
    torch_a = a_dense.to_torch()
    pdb.set_trace()
    print(a_dense.shape)
    print(torch_a)


def test_elemwise_2d_add_csc_csc_csc():
    a, b, result, _, _ = generate_2d_tensors("ds", "ds", "ds", [1, 0], [1, 0], [1, 0])
    pdb.set_trace()
    result_cpp = a + b

    assert result_cpp._storage._value.tolist() == result._storage._value.tolist(), "Values are different"
    assert result_cpp._storage._index.mode_indices[0] == result._storage._index.mode_indices[0]
    assert result_cpp._storage._index.mode_indices[1][0].tolist() == result._storage._index.mode_indices[1][0].tolist()
    assert result_cpp._storage._index.mode_indices[1][1].tolist() == result._storage._index.mode_indices[1][1].tolist()
    # pdb.set_trace()
    print(result)

def test_elemwise_2d_add_csr_csr_coo():
    a, b, result, _, _ = generate_2d_tensors("ds", "oo", "ds")
    # pdb.set_trace()
    result_cpp = a + b

    assert result_cpp._storage._value.tolist() == result._storage._value.tolist(), "Values are different"
    assert result_cpp._storage._index.mode_indices[0] == result._storage._index.mode_indices[0]
    assert result_cpp._storage._index.mode_indices[1][0].tolist() == result._storage._index.mode_indices[1][0].tolist()
    assert result_cpp._storage._index.mode_indices[1][1].tolist() == result._storage._index.mode_indices[1][1].tolist()
    # pdb.set_trace()
    print(result)


def test_elemwise_2d_add_csr_csc():
    # CSR = CSR + CSC
    a, b, result, _, _ = generate_2d_tensors("ds", "ds", "ds", [0, 1], [1, 0], [0, 1])

    result_cpp = a + b

    assert result_cpp._storage._value.tolist() == result._storage._value.tolist(), "Values are different"
    assert result_cpp._storage._index.mode_indices[0] == result._storage._index.mode_indices[0]
    assert result_cpp._storage._index.mode_indices[1][0].tolist() == result._storage._index.mode_indices[1][0].tolist()
    assert result_cpp._storage._index.mode_indices[1][1].tolist() == result._storage._index.mode_indices[1][1].tolist()

    # CSC = CSC + CSR
    # a, b, result, _, _ = generate_2d_tensors("ds", "ds", "ds", [1, 0], [0, 1], [1, 0])
    #
    # result_cpp = a + b
    #
    # assert result_cpp._storage._value.tolist() == result._storage._value.tolist(), "Values are different"
    # assert result_cpp._storage._index.mode_indices[0] == result._storage._index.mode_indices[0]
    # assert result_cpp._storage._index.mode_indices[1][0].tolist() == result._storage._index.mode_indices[1][0].tolist()
    # assert result_cpp._storage._index.mode_indices[1][1].tolist() == result._storage._index.mode_indices[1][1].tolist()

    # pdb.set_trace()
    print(result)

def test_elemwise_2d_add_coo():
    # COO [0,1] = COO [0,1] + COO [1, 0]
    a, b, result, _, _ = generate_2d_tensors("oo", "oo", "oo", [0, 1], [1, 0], [0, 1])

    result_cpp = a + b

    assert result_cpp._storage._value.tolist() == result._storage._value.tolist(), "Values are different"
    assert result_cpp._storage._index.mode_indices[0] == result._storage._index.mode_indices[0]
    assert result_cpp._storage._index.mode_indices[1][0].tolist() == result._storage._index.mode_indices[1][0].tolist()
    assert result_cpp._storage._index.mode_indices[1][1].tolist() == result._storage._index.mode_indices[1][1].tolist()

    # COO [1, 0] = COO [1, 0] + COO [0, 1]
    a, b, result, _, _ = generate_2d_tensors("oo", "oo", "oo", [0, 1], [1, 0], [0, 1])

    result_cpp = a + b

    assert result_cpp._storage._value.tolist() == result._storage._value.tolist(), "Values are different"
    assert result_cpp._storage._index.mode_indices[0] == result._storage._index.mode_indices[0]
    assert result_cpp._storage._index.mode_indices[1][0].tolist() == result._storage._index.mode_indices[1][0].tolist()
    assert result_cpp._storage._index.mode_indices[1][1].tolist() == result._storage._index.mode_indices[1][1].tolist()

def test_custom_kernel(lhs_tensors, result_shape, kernel_code_filename):
    # Read header_cpp_code from csrc/header.cpp
    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    with open(PROJECT_ROOT_DIR / f"csrc/{kernel_code_filename}", "r") as f:
        cpp_code = f.read()

    # Load special kernels
    module = load_inline(
        name="kernel",
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        # extra_cflags=["-O3", "-mcpu=apple-m1", "-ffast-math", "-fno-signed-zeros"],
        extra_cflags=["-O3", "-ffast-math", "-fno-signed-zeros"],
    )

    args = [result_shape]
    for tensor in lhs_tensors:
        args.append(tensor.shape)
        args.append(tensor.index.mode_indices)
        args.append(tensor.values)

    result_cpp = module.evaluate(*args)
    return result_cpp


def test_change_mode_order_2d_ds():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    a_csr = (STensor.from_torch(tensor_a_torch, "A")).to_sparse("ds")
    a_csc_test = a_csr.change_mode_order([1, 0])
    a_csc_true = (STensor.from_torch(tensor_a_torch, "A", [1, 0])).to_sparse("ds")

    assert a_csc_test.storage.value.tolist() == a_csc_true.storage.value.tolist(), "Values are different"
    assert a_csc_test.storage.index.mode_indices[0] == a_csc_true.storage.index.mode_indices[0]
    assert a_csc_test.storage.index.mode_indices[1][0].tolist() == a_csc_true.storage.index.mode_indices[1][0].tolist()
    assert a_csc_test.storage.index.mode_indices[1][1].tolist() == a_csc_true.storage.index.mode_indices[1][1].tolist()

    print(a_csc_test.storage.value.tolist())
    print(a_csc_test.storage.index.mode_indices[1][0].tolist())
    print(a_csc_test.storage.index.mode_indices[1][1].tolist())


def test_change_mode_order_2d_coo():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    a_coo = (STensor.from_torch(tensor_a_torch, "A")).to_sparse("oo")
    a_coo_reverse = a_coo.change_mode_order([1, 0])
    a_coo_true = (STensor.from_torch(tensor_a_torch, "A", [1, 0])).to_sparse("oo")

    assert a_coo_reverse.storage.value.tolist() == a_coo_true.storage.value.tolist(), "Values are different"
    assert a_coo_reverse.storage.index.mode_indices[0][0].tolist() == a_coo_true.storage.index.mode_indices[0][0].tolist()
    assert a_coo_reverse.storage.index.mode_indices[1][0].tolist() == a_coo_true.storage.index.mode_indices[1][0].tolist()

    print(a_coo_reverse.storage.value.tolist())
    print(a_coo_reverse.storage.index.mode_indices[0][0].tolist())
    print(a_coo_reverse.storage.index.mode_indices[1][0].tolist())


def test_change_mode_order_3d_dss():
    tensor_a_torch = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 2],
                [3, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 4, 0],
                [5, 6, 0],
                [0, 0, 0],
            ],
        ]
    )
    a = STensor.from_torch(tensor_a_torch, "A").to_sparse("dss")
    pdb.set_trace()
    a.change_mode_order([2, 1, 0])
    pdb.set_trace()
    print(a)

def test_change_mode_order_produce_duplicate():
    tensor_a_torch = torch.Tensor(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
            ],
            [
                [19, 20, 21],
                [22, 23, 24],
                [25, 26, 27],
            ],
        ]
    )

    a_csr = STensor.from_torch(tensor_a_torch, "A").to_sparse("dss")
    a_csc = a_csr.change_mode_order([0, 1, 2])
    pdb.set_trace()
    print(a_csc)

def test_change_mode_order_3d_fixed():
    # tensor_a_torch = torch.Tensor(
    #     [
    #         [
    #             [1, 2, 3],
    #             [4, 5, 6],
    #             [7, 8, 9],
    #         ],
    #         [
    #             [10, 11, 12],
    #             [13, 14, 15],
    #             [16, 17, 18],
    #         ],
    #         [
    #             [19, 20, 21],
    #             [22, 23, 24],
    #             [25, 26, 27],
    #         ],
    #     ]
    # )
    tensor_a_torch = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 2],
                [3, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 4, 0],
                [5, 6, 0],
                [0, 0, 0],
            ],
        ]
    )
    a = STensor.from_torch(tensor_a_torch, "A").to_sparse("dss")

    result_cpp = test_custom_kernel([a], a.shape, "change_mode_order_3d_broken.cpp")
    pdb.set_trace()
    print(result_cpp)


def test_change_mode_order_fixed():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    a_csr = STensor.from_torch(tensor_a_torch, "A").to_sparse("ds")

    result_cpp = test_custom_kernel([a_csr], a_csr.shape, "change_mode_order_fixed.cpp")
    pdb.set_trace()
    print(result_cpp)


def test_coo_to_csr():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_coo = STensor.from_torch(tensor_a_torch, "A").to_sparse("oo")
    pdb.set_trace()
    a_csr = a_coo.to_sparse("ds")
    pdb.set_trace()
    print(a_csr)


def test_different_shapes():
    # 5 x 4 tensor
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4],
            [2, 2, 0, 0],
            [3, 0, 3, 0],
            [0, 0, 0, 0],
            [5, 0, 0, 0],
        ]
    )
    a_row = STensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    a_col = STensor.from_torch(tensor_a_torch, "A", [1, 0])
    a_row.change_mode_order([1, 0])
    pdb.set_trace()
    print(a_col)


def test_einsum_2d_concord_csr_csc_csr():
    a, b, _, _, result_matmul = generate_2d_tensors("ds", "ds", "ds", [0, 1], [1, 0], [0, 1])
    result_cpp = einsum("ik,jk->ij", a, b, format="ds")
    assert result_cpp.storage.value.tolist() == result_matmul.storage.value.tolist(), "Values are different"
    assert result_cpp.storage.index.mode_indices[1][0].tolist() == result_matmul.storage.index.mode_indices[1][0].tolist()
    assert result_cpp.storage.index.mode_indices[1][1].tolist() == result_matmul.storage.index.mode_indices[1][1].tolist()
    print(result_cpp)


# TODO: fix Scheduler.insert_workspace() for test where A & result are CSC and B is CSR
def test_einsum_2d_concord_csc_csr_csc():
    a, b, _, _, result_matmul = generate_2d_tensors("ds", "ds", "ds", [1, 0], [0, 1], [1, 0])
    result_cpp = einsum("ik,kj->ij", a, b, format="ds")
    assert result_cpp.storage.value.tolist() == result_matmul.storage.value.tolist(), "Values are different"
    assert result_cpp.storage.index.mode_indices[1][0].tolist() == result_matmul.storage.index.mode_indices[1][0].tolist()
    assert result_cpp.storage.index.mode_indices[1][1].tolist() == result_matmul.storage.index.mode_indices[1][1].tolist()
    pdb.set_trace()
    print(result_cpp)


def test_einsum_2d_concord_oo_oo_oo():
    a, b, _, _, result_matmul = generate_2d_tensors("oo", "oo", "oo", [0, 1], [1, 0], [1, 0])
    result_cpp = einsum("ik,jk->ji", a, b, format="oo")
    assert result_cpp.storage.value.tolist() == result_matmul.storage.value.tolist(), "Values are different"
    assert result_cpp.storage.index.mode_indices[0][0].tolist() == result_matmul.storage.index.mode_indices[0][0].tolist()
    assert result_cpp.storage.index.mode_indices[1][0].tolist() == result_matmul.storage.index.mode_indices[1][0].tolist()
    print(result_cpp)


def test_einsum_2d_concord_dd_dd_dd():
    a, b, _, _, result_matmul = generate_2d_tensors("dd", "dd", "dd", [0, 1], [1, 0], [0, 1])
    result_cpp = einsum("ik,jk->ij", a, b, format="dd")
    pdb.set_trace()
    print(result_cpp)


def test_change_einsum_kernel():
    a, b, _, _, result_matmul = generate_2d_tensors("dd", "dd", "dd", [0, 1], [1, 0], [0, 1])
    result_cpp = test_custom_kernel([a, b], a.shape, "spmspm_mode_order_broken.cpp")
    pdb.set_trace()
    print(result_cpp)


def test_einsum_2d_cycle():
    a, b, _, _, result_matmul = generate_2d_tensors("ds", "ds", "ds", [0, 1], [0, 1], [1, 0])
    result_cpp = einsum("ik,kj->ji", a, b, format="ds")
    pdb.set_trace()
    print(result_cpp)


def test_einsum_2d_cycle_tricky():
    a, b, _, _, _ = generate_2d_tensors("ds", "ds", "ds", [0, 1], [1, 0], [1, 0])
    result_cpp = einsum("ij,ji->ji", a, b, format="ds")
    pdb.set_trace()
    print(result_cpp)


def test_tensor_addition_2d():
    a, b, _, _, _ = generate_2d_tensors("ds", "ds", "ds")
    result = a + b
    print(result)


def test_to_sparse_sss():
    tensor_result_torch = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 2],
                [3, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 4, 0],
                [5, 6, 0],
                [0, 0, 0],
            ]
        ]
    )

    result = STensor.from_torch(tensor_result_torch, "result").to_sparse("sss")
    pdb.set_trace()
    print(result)

def test_3d_einsum_wksp():
    tensor_a_torch = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 2],
                [3, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 4, 0],
                [5, 6, 0],
                [0, 0, 0],
            ],
        ]
    )

    a = STensor.from_torch(tensor_a_torch, "a").to_sparse("dss")
    b = a
    c = einsum("ijk,ikl->ijl", a, b, format="dss")
    c.to_sparse("ooo")
    pdb.set_trace()
    print(c)

def test_to_sparse_kernel_sss():
    tensor_result_torch = torch.Tensor(
        [
            [
                [1, 0, 0],
                [0, 0, 2],
                [3, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 4, 0],
                [5, 6, 0],
                [0, 0, 0],
            ]
        ]
    )
    result = STensor.from_torch(tensor_result_torch, "result")
    result_cpp = test_custom_kernel([result], result.shape, "to_sparse_broken.cpp")
    pdb.set_trace()
    print(result_cpp)

def test_ss_tensor_addition_2d():
    a, b, result_add, _, _ = generate_2d_tensors("ss", "ss", "ss")
    pdb.set_trace()
def test_my_sandbox():
    tensor_a_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 3, 1],
        ]
    )


    a_sparse = STensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_sparse = STensor.from_torch(tensor_b_torch, "B").to_sparse("ds")

    result = einsum("ik,kj->ij", a_sparse, b_sparse, format="ds")

    pdb.set_trace()
    print("printing values now:")
    # print(result.index.mode_indices[0][0].tolist())
    # print(result.index.mode_indices[0][1].tolist())
    print(result.index.mode_indices[1][0].tolist())
    print(result.index.mode_indices[1][1].tolist())
    print(result.values.tolist())
