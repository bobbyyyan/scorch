# pretty print
import pprint
import time
from itertools import product
from typing import List, Dict, Any, Iterable, Union, Optional


import torch
from torch.utils.cpp_extension import load_inline

from scorch import Tensor, einsum, utils, TensorFormat
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

def generate_tensors(a_format, b_format, result_format):
    """
    Generates tensors A, B, and result given their format
        - values for tensors are hard-coded for debugging purposes
    Args:
        formats can be "csc", "csr", etc. for now, only csc & csr supported
    """

    # For now, assume that all 3 tensors will be csc
    assert(a_format == "csc" or a_format == "csr")
    assert(b_format == "csc" or b_format == "csr")
    assert(result_format == "csc" or result_format == "csr")

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

    tensor_result_torch = torch.Tensor(
        [
            [1, 4, 0, 19, 5],
            [2, 4, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 15, 5],
        ]
    )

    a_csr = Tensor.from_torch(tensor_a_torch, "a").to_sparse("ds")
    b_csr = Tensor.from_torch(tensor_b_torch, "b").to_sparse("ds")
    result_csr = Tensor.from_torch(tensor_result_torch, "result").to_sparse("ds")

    # Hard coding tensors with the above initialization for A, B, result
    a_csc = Tensor(
        name="A",
        shape=(5, 5),
        storage=TensorStorage(
            index=TensorIndex(
                tensor_format=TensorFormat(
                    level_formats=[
                        LevelFormat(mode=LevelType.DENSE),
                        LevelFormat(mode=LevelType.COMPRESSED)
                    ]
                ),
                mode_indices=[[], [torch.tensor([0, 4, 6, 8, 9, 11], dtype=torch.int32), torch.tensor([0, 1, 2, 4, 0, 1, 0, 2, 0, 0, 4], dtype=torch.int32)]],
            ),
            value=torch.tensor([1., 2., 3., 5., 2., 2., 3., 3., 4., 5., 5.], dtype=torch.float32),
        ),
    )

    b_csc = Tensor(
        name="B",
        shape=(5, 5),
        storage=TensorStorage(
            index=TensorIndex(
                tensor_format=TensorFormat(
                    level_formats=[
                        LevelFormat(mode=LevelType.DENSE),
                        LevelFormat(mode=LevelType.COMPRESSED)
                    ]
                ),
                mode_indices=[[], [torch.tensor([0, 1, 2, 2, 4, 5], dtype=torch.int32), torch.tensor([0, 1, 3, 4, 4], dtype=torch.int32)]],
            ),
            value=torch.tensor([1., 2., 1., 3., 1.], dtype=torch.float32),
        ),
    )

    result_csc = Tensor(
        name="result",
        shape=(5, 5),
        storage=TensorStorage(
            index=TensorIndex(
                tensor_format=TensorFormat(
                    level_formats=[
                        LevelFormat(mode=LevelType.DENSE),
                        LevelFormat(mode=LevelType.COMPRESSED)
                    ]
                ),
                mode_indices=[[], [torch.tensor([0, 4, 6, 6, 8, 10], dtype=torch.int32), torch.tensor([0, 1, 2, 4, 0, 1, 0, 4, 0, 4], dtype=torch.int32)]],
            ),
            value=torch.tensor([1., 2., 3., 5., 4., 4., 19., 15., 5., 5.], dtype=torch.float32),
        ),
    )

    format_tensor = {
        "a_csr": a_csr, "b_csr": b_csr, "result_csr": result_csr,
        "a_csc": a_csc, "b_csc": b_csc, "result_csc": result_csc
    }

    return format_tensor[f"a_{a_format}"], format_tensor[f"b_{b_format}"], format_tensor[f"result_{result_format}"]

def test_spmm_csr_csr_csr():
    a, b, result = generate_tensors("csr", "csr", "csr")
    result_cpp = test_custom_kernel(a, b, result, "spmm_csr_wksp.cpp")
    pdb.set_trace()
    result_kernel = Tensor(
        shape=(5, 5),
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format=TensorFormat(
                level_formats=[
                    LevelFormat(mode=LevelType.DENSE),
                    LevelFormat(mode=LevelType.COMPRESSED)
                ]
            ),
        ),
        value=result_cpp._storage._value,
    )

    print(result_kernel.values)
    print(result_kernel.index.mode_indices)
def test_spmm_csc_csc_csc():
    a, b, result = generate_tensors("csc", "csc", "csc")
    result_cpp = test_custom_kernel(a, b, result, "spmm_csc_wksp.cpp")
    pdb.set_trace()
    result_kernel = Tensor(
        shape=(5, 5),
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format=TensorFormat(
                level_formats=[
                    LevelFormat(mode=LevelType.DENSE),
                    LevelFormat(mode=LevelType.COMPRESSED)
                ]
            ),
        ),
        value=result_cpp._storage._value,
    )

    print(result_kernel.values)
    print(result_kernel.index.mode_indices)


def test_custom_kernel(a, b, result, kernel_code_filename):
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

    args = [result.shape]
    for tensor in [a, b]:
        args.append(tensor.shape)
        args.append(tensor.index.mode_indices)
        args.append(tensor.values)

    result_cpp = module.evaluate(*args)
    return result_cpp


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


    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ds")

    result = einsum("ik,kj->ij", a_sparse, b_sparse, format="ds")

    pdb.set_trace()
    print("printing values now:")
    # print(result.index.mode_indices[0][0].tolist())
    # print(result.index.mode_indices[0][1].tolist())
    print(result.index.mode_indices[1][0].tolist())
    print(result.index.mode_indices[1][1].tolist())
    print(result.values.tolist())

def test_dense_copy():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )

    tensor_a = Tensor.from_torch(tensor_a_torch, "A")

    tensor_a_dense = tensor_a.to_dense()

    print(tensor_a_dense)


def test_sparse_to_dense():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("oo")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("oo")

    a_sparse_to_torch = a_sparse.to_torch()
    b_sparse_to_torch = b_sparse.to_torch()

    assert a_sparse_to_torch.tolist() == tensor_a_torch.tolist()
    assert b_sparse_to_torch.tolist() == tensor_b_torch.tolist()


def test_to_torch():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5])

    sp_vector_a = Tensor.from_torch(tensor_a_torch).to_sparse("s")
    sp_vector_b = Tensor.from_torch(tensor_b_torch).to_sparse("s")

    sp_vector_a_to_torch = sp_vector_a.to_torch()
    sp_vector_b_to_torch = sp_vector_b.to_torch()

    assert sp_vector_a_to_torch.tolist() == [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4]
    assert sp_vector_b_to_torch.tolist() == [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5]


def test_elemwise_vector_mul_sss():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5])

    sp_vector_a = Tensor.from_torch(tensor_a_torch).to_sparse("s")
    sp_vector_b = Tensor.from_torch(tensor_b_torch).to_sparse("s")

    result = einsum("i,i->i", sp_vector_a, sp_vector_b, format="s")

    assert result.shape == (12,)
    assert len(result.index.mode_indices) == 1

    mode_index = result.index.mode_indices[0]
    assert mode_index[0].tolist() == [0, 5]
    assert mode_index[1].tolist() == [0, 2, 4, 10, 11]

    assert result.values.tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]


def test_elemwise_vector_add_sss():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5])

    sp_vector_a = Tensor.from_torch(tensor_a_torch).to_sparse("s")
    sp_vector_b = Tensor.from_torch(tensor_b_torch).to_sparse("s")

    result = sp_vector_a + sp_vector_b

    assert result.shape == (12,)
    assert len(result.index.mode_indices) == 1

    mode_index = result.index.mode_indices[0]
    assert mode_index[0].tolist() == [0, 9]
    assert mode_index[1].tolist() == [0, 1, 2, 3, 4, 6, 8, 10, 11]

    assert result.values.tolist() == [3.0, 2.0, 4.0, 2.0, 5.0, 4.0, 5.0, 9.0, 6.5]


def test_elemwise_vector_add_sss_2():
    tensor_a_torch = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    tensor_b_torch = torch.Tensor([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    tensor_c_torch = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12])

    sp_vector_a = Tensor.from_torch(tensor_a_torch).to_sparse("s")
    sp_vector_b = Tensor.from_torch(tensor_b_torch).to_sparse("s")
    sp_vector_c = Tensor.from_torch(tensor_c_torch).to_sparse("s")

    result = sp_vector_a + sp_vector_b + sp_vector_c

    assert result.shape == (12,)
    assert len(result.index.mode_indices) == 1

    mode_index = result.index.mode_indices[0]
    assert mode_index[0].tolist() == [0, 3]
    assert mode_index[1].tolist() == [0, 1, 11]

    assert result.values.tolist() == [1.0, 2.0, 12.0]


def test_elemwise_vector_mul_dss():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 8, 4, 5, 0])
    tensor_b_torch = torch.Tensor([2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2.5, 0, 14])

    sp_vector_a = Tensor.from_torch(tensor_a_torch).to_sparse("s")
    sp_vector_b = Tensor.from_torch(tensor_b_torch).to_sparse("s")

    result = einsum("i,i->i", sp_vector_a, sp_vector_b, format="d")

    assert result.shape == (14,)
    assert len(result.index.mode_indices) == 1

    assert result.values.tolist() == [
        2.0,
        0.0,
        4.0,
        0.0,
        6.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.0,
        10.0,
        0.0,
        0.0,
    ]


def test_elemwise_matrix_mul_oo_oo_oo():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("oo")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("oo")

    result = einsum("ij->ij", a_sparse, b_sparse, format="oo")

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[0][0].tolist() == [0, 1, 2, 3, 4]

    assert result.index.mode_indices[1][0].tolist() == [0, 1, 2, 3, 4]

    assert result.values.tolist() == [1.0, 4.0, 9.0, 16.0, 25.0]


def test_elemwise_matrix_mul_oo_ss_oo():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ss")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("oo")

    result = einsum("ij,ij->ij", a_sparse, b_sparse, format="oo")

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[0][0].tolist() == [0, 1, 2, 3, 4]

    assert result.index.mode_indices[1][0].tolist() == [0, 1, 2, 3, 4]

    assert result.values.tolist() == [1.0, 4.0, 9.0, 16.0, 25.0]


def test_elemwise_matrix_mul_oo_oo_ss():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("oo")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ss")

    result = einsum("ij,ij->ij", a_sparse, b_sparse, format="oo")

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[0][0].tolist() == [0, 1, 2, 3, 4]

    assert result.index.mode_indices[1][0].tolist() == [0, 1, 2, 3, 4]

    assert result.values.tolist() == [1.0, 4.0, 9.0, 16.0, 25.0]


def test_elemwise_matrix_mul_oo_ss_ss():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ss")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ss")

    result = einsum("ij,ij->ij", a_sparse, b_sparse, format="oo")

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[0][0].tolist() == [0, 1, 2, 3, 4]

    assert result.index.mode_indices[1][0].tolist() == [0, 1, 2, 3, 4]

    assert result.values.tolist() == [1.0, 4.0, 9.0, 16.0, 25.0]


def test_elemwise_matrix_mul_oo_ds_ds():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ds")

    result = einsum("ij,ij->ij", a_sparse, b_sparse, format="oo")

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[0][0].tolist() == [0, 1, 2, 3, 4]

    assert result.index.mode_indices[1][0].tolist() == [0, 1, 2, 3, 4]

    assert result.values.tolist() == [1.0, 4.0, 9.0, 16.0, 25.0]


def test_elemwise_matrix_mul_ss_ss_ss():
    # # Generate a random sparse 10x10 torch tensor
    # tensor_a_torch = torch.rand(10, 10)
    # tensor_a_torch[torch.rand(10, 10) > 0.5] = 0
    # tensor_b_torch = torch.rand(10, 10)
    # tensor_b_torch[torch.rand(10, 10) > 0.5] = 0
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ss")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ss")

    result = einsum("ij,ij->ij", a_sparse, b_sparse, format="ss")

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[0][0].tolist() == [0, 5]
    assert result.index.mode_indices[0][1].tolist() == [0, 1, 2, 3, 4]

    assert result.index.mode_indices[1][0].tolist() == [0, 1, 2, 3, 4, 5]
    assert result.index.mode_indices[1][1].tolist() == [0, 1, 2, 3, 4]

    assert result.values.tolist() == [1.0, 4.0, 9.0, 16.0, 25.0]


def test_elemwise_matrix_mul_ss_oo_ss():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("oo")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ss")

    result = einsum("ij,ij->ij", a_sparse, b_sparse, format="ss")

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[0][0].tolist() == [0, 5]
    assert result.index.mode_indices[0][1].tolist() == [0, 1, 2, 3, 4]

    assert result.index.mode_indices[1][0].tolist() == [0, 1, 2, 3, 4, 5]
    assert result.index.mode_indices[1][1].tolist() == [0, 1, 2, 3, 4]

    assert result.values.tolist() == [1.0, 4.0, 9.0, 16.0, 25.0]


def test_ij_i_j_ss_s_s():
    tensor_a_torch = torch.Tensor([1, 0, 2, 0, 3])
    tensor_b_torch = torch.Tensor([7, 8, 0, 9])

    sp_vector_a = Tensor.from_torch(tensor_a_torch).to_sparse()
    sp_vector_b = Tensor.from_torch(tensor_b_torch).to_sparse()

    result = einsum("i,j->ij", sp_vector_a, sp_vector_b, format="ss")

    assert result.shape == (5, 4)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[0][0].tolist() == [0, 3]
    assert result.index.mode_indices[0][1].tolist() == [0, 2, 4]

    assert result.index.mode_indices[1][0].tolist() == [0, 3, 6, 9]
    assert result.index.mode_indices[1][1].tolist() == [0, 1, 3, 0, 1, 3, 0, 1, 3]

    assert result.values.tolist() == [7.0, 8.0, 9.0, 14.0, 16.0, 18.0, 21.0, 24.0, 27.0]


def test_spmm_ds_ds_ds_ikj_gustavson():
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    workspace = Workspace(
        name="wksp",
        dim=1,
    )

    # A[i, j] = ForAll(i,
    #   Where(
    #     producer=ForAll(k, ForAll(j, workspace[j] += B[i, k] * C[k, j])),
    #     consumer=ForAll(j, A[i, j] = workspace[j])
    #   )
    # )

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[j],
                        B[i, k] * C[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    workspace[j],
                ),
            ),
        ),
    )

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    cpp_code = llir_lowerer.lower_llir(lowered_llir)

    print(cpp_code)

    # Read header_cpp_code from csrc/header.cpp
    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    module = torch.utils.cpp_extension.load_inline(
        name="kernel",
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=["-O3"],
    )

    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ds")

    output_format = parse_format("ds")

    result_shape = (5, 5)
    args = [result_shape]

    for tensor in [a_sparse, b_sparse]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor._storage._index.mode_indices)  # type: ignore
        args.append(tensor._storage.value)  # type: ignore

    result_cpp = module.evaluate(*args)
    result = Tensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format=output_format,
        ),
        value=result_cpp._storage._value,
    )

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[1][0].tolist() == [0, 5, 7, 9, 11, 13]
    assert result.index.mode_indices[1][1].tolist() == [
        0,
        1,
        2,
        3,
        4,
        0,
        1,
        0,
        2,
        0,
        3,
        0,
        4,
    ]

    assert result.values.tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        4.0,
        4.0,
        9.0,
        9.0,
        16.0,
        16.0,
        25.0,
        25.0,
    ]


# def test_spmm_ds_ds_ds_kij_outer():
#     """
#     TODO: need out-of-order/discordant iteration to pass
#     """
#     i = IndexVar("i")
#     j = IndexVar("j")
#     k = IndexVar("k")
#
#     A = TensorVar("A", fmt=["dense", "sparse"])
#     B = TensorVar("B", fmt=["dense", "sparse"])
#     C = TensorVar("C", fmt=["dense", "sparse"])
#
#     workspace = Workspace(
#         name="wksp",
#         dim=2,
#     )
#
#     """
#         A[i, j] = Where(
#           producer=ForAll(k, ForAll(i, ForAll(j, workspace[i, j] += B[i, k] * C[k, j]))),
#           consumer=ForAll(i, ForAll(j, A[i, j] = workspace[i, j])),
#         )
#         """
#
#     cin_stmt = Where(
#         producer=ForAll(
#             k,
#             ForAll(
#                 i,
#                 ForAll(
#                     j,
#                     TensorAssign(
#                         workspace[i, j],
#                         B[i, k] * C[k, j],
#                         op=Operation.ADD,
#                     ),
#                 ),
#             ),
#         ),
#         consumer=ForAll(
#             i,
#             ForAll(
#                 j,
#                 TensorAssign(
#                     A[i, j],
#                     workspace[i, j],
#                     op=Operation.ADD,
#                 ),
#             ),
#         ),
#     )
#
#     lowerer = CINLowerer()
#
#     lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
#
#     llir_lowerer = LLIRLowerer()
#
#     cpp_code = llir_lowerer.lower_llir(lowered_llir)
#
#     print(cpp_code)
#
#     # Read header_cpp_code from csrc/header.cpp
#     with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
#         header_cpp_code = f.read()
#
#     module = torch.utils.cpp_extension.load_inline(
#         name="kernel",
#         cpp_sources=[header_cpp_code, cpp_code],
#         functions=["evaluate"],
#     )
#
#     tensor_a_torch = torch.Tensor(
#         [
#             [1, 0, 0, 0, 0],
#             [0, 2, 0, 0, 0],
#             [0, 0, 3, 0, 0],
#             [0, 0, 0, 4, 0],
#             [0, 0, 0, 0, 5],
#         ]
#     )
#     tensor_b_torch = torch.Tensor(
#         [
#             [1, 2, 3, 4, 5],
#             [2, 2, 0, 0, 0],
#             [3, 0, 3, 0, 0],
#             [4, 0, 0, 4, 0],
#             [5, 0, 0, 0, 5],
#         ]
#     )
#
#     a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
#     b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ds")
#
#     output_format = parse_format("ds")
#
#     result_shape = (5, 5)
#     args = [result_shape]
#
#     for tensor in [a_sparse, b_sparse]:
#         args.append(tensor.shape)  # type: ignore
#         args.append(tensor._storage._index.mode_indices)  # type: ignore
#         args.append(tensor._storage.value)  # type: ignore
#
#     result_cpp = module.evaluate(*args)
#     result = Tensor(
#         shape=result_shape,
#         index=TensorIndex(
#             mode_indices=result_cpp._storage._index.mode_indices,
#             tensor_format=output_format,
#         ),
#         value=result_cpp._storage._value,
#     )
#
#     assert result.shape == (5, 5)
#     assert len(result.index.mode_indices) == 2
#
#     assert result.index.mode_indices[1][0].tolist() == [0, 5, 7, 9, 11, 13]
#     assert result.index.mode_indices[1][1].tolist() == [
#         0,
#         1,
#         2,
#         3,
#         4,
#         0,
#         1,
#         0,
#         2,
#         0,
#         3,
#         0,
#         4,
#     ]
#
#     assert result.values.tolist() == [
#         1.0,
#         2.0,
#         3.0,
#         4.0,
#         5.0,
#         4.0,
#         4.0,
#         9.0,
#         9.0,
#         16.0,
#         16.0,
#         25.0,
#         25.0,
#     ]


def test_matmul_wksp():
    print("Testing matmul_wksp")
    # Random matrix
    dim_n = 50
    tensor_a_torch = torch.rand(dim_n, dim_n)
    tensor_b_torch = torch.rand(dim_n, dim_n)

    # Sparsify to 80% sparsity
    sparsity_level = 0.8
    tensor_a_torch = torch.where(
        torch.rand_like(tensor_a_torch) > sparsity_level,
        tensor_a_torch,
        torch.zeros_like(tensor_a_torch),
    )

    start_time = time.time()
    torch_result = torch.matmul(tensor_a_torch, tensor_b_torch)
    end_time = time.time()
    torch_total_time = end_time - start_time

    time_dict = {}

    start_time = time.time()
    scorch_result = matmul_wksp(tensor_a_torch, tensor_b_torch, time_dict=time_dict)
    end_time = time.time()

    scorch_eval_time = time_dict["eval_time"]
    scorch_total_time = end_time - start_time

    # Assert that the results are the same
    assert torch.allclose(torch_result, scorch_result.to_torch())

    print(f"Torch total time taken: {torch_total_time}s")
    print(f"Scorch eval time taken: {scorch_eval_time}s")
    print(f"Scorch total time taken: {scorch_total_time}s")
    print(f"Scorch eval time / Torch total time: {scorch_eval_time / torch_total_time}")

    assert torch.allclose(torch_result, scorch_result.to_torch())


def test_dense_matmul():
    tensor_a_torch = torch.rand(100, 200)
    tensor_b_torch = torch.rand(200, 300)
    torch_result = torch.matmul(tensor_a_torch, tensor_b_torch)

    scorch_result = matmul(tensor_a_torch, tensor_b_torch)

    assert torch_result.tolist() == scorch_result.to_torch().tolist()


def test_matmul_ds_dd_dd():
    n = 1024
    tensor_b_torch = torch.randint(0, 1000, (n, n))
    tensor_a_torch = torch.randint(0, 1000, (n, n))

    a_scorch = Tensor.from_torch(tensor_a_torch, "A").to_dense()
    b_scorch = Tensor.from_torch(tensor_b_torch, "B").to_dense()

    torch_result = torch.matmul(tensor_a_torch, tensor_b_torch)

    scorch_result = einsum("ik,kj->ij", a_scorch, b_scorch, format="ds")
    scorch_result_torch = scorch_result.to_torch()

    assert torch_result.allclose(scorch_result_torch)


def todo_test_matmul_ds_dd_dd_large():
    n = 2048
    tensor_b_torch = torch.randint(0, 1000, (n, n))
    tensor_a_torch = torch.randint(0, 1000, (n, n))

    a_scorch = Tensor.from_torch(tensor_a_torch, "A").to_dense()
    b_scorch = Tensor.from_torch(tensor_b_torch, "B").to_dense()

    torch_result = torch.matmul(tensor_a_torch, tensor_b_torch)

    scorch_result = einsum("ik,kj->ij", a_scorch, b_scorch, format="ds")
    scorch_result_torch = scorch_result.to_torch()

    assert torch_result.allclose(scorch_result_torch)


def todo_test_matmul_wksp_dd_oo_dd_time():
    n = 100
    sparsity = 0.9
    random_tensor_a = torch.rand(n, n)
    random_tensor_b = torch.rand(n, n)

    # Randomly sparsify each tensor
    random_tensor_a = random_tensor_a * (torch.rand(n, n) > sparsity)
    random_tensor_b = random_tensor_b * (torch.rand(n, n) > sparsity)

    start_time = time.time()
    torch_result = torch.matmul(random_tensor_a, random_tensor_b)
    torch_time = time.time() - start_time

    tensor_a_scorch = Tensor.from_torch(random_tensor_a, "A").to_sparse("oo")
    tensor_b_scorch = Tensor.from_torch(random_tensor_b, "B")

    start_time = time.time()
    scorch_result = matmul_wksp(tensor_a_scorch, tensor_b_scorch, output_format="dd")
    scorch_time = time.time() - start_time

    print(f"torch time: {torch_time}")
    print(f"[matmul_wksp] scorch time: {scorch_time}")
    print(f"[matmul_wksp] scorch time / torch time: {scorch_time / torch_time}")


def test_matmul_wksp_ds_time():
    n = 100
    sparsity = 0.9
    random_tensor_a = torch.rand(n, n)
    random_tensor_b = torch.rand(n, n)

    # Randomly sparsify each tensor
    random_tensor_a = random_tensor_a * (torch.rand(n, n) > sparsity).float()
    random_tensor_b = random_tensor_b * (torch.rand(n, n) > sparsity).float()

    start_time = time.time()
    torch_result = torch.matmul(random_tensor_a, random_tensor_b)
    torch_time = time.time() - start_time

    tensor_a_scorch = Tensor.from_torch(random_tensor_a, "A").to_sparse()
    tensor_b_scorch = Tensor.from_torch(random_tensor_b, "B").to_sparse()

    start_time = time.time()
    scorch_result = matmul_wksp(tensor_a_scorch, tensor_b_scorch, output_format="ds")
    scorch_time = time.time() - start_time

    print(f"torch time: {torch_time}")
    print(f"[matmul_wksp] scorch time: {scorch_time}")
    print(f"[matmul_wksp] scorch time / torch time: {scorch_time / torch_time}")
    assert torch.allclose(torch_result, scorch_result.to_torch())


def test_spmm_ds_ds_dd_time():
    """
    Compare speed of torch and scorch matmul
    Use random tensors
    """
    n = 100
    sparsity = 0.9
    random_tensor_a = torch.rand(n, n)
    random_tensor_b = torch.rand(n, n)

    # Randomly sparsify each tensor
    random_tensor_a = random_tensor_a * (torch.rand(n, n) > sparsity).float()
    random_tensor_b = random_tensor_b * (torch.rand(n, n) > sparsity).float()

    start_time = time.time()
    torch_result = torch.matmul(random_tensor_a, random_tensor_b)
    torch_time = time.time() - start_time

    tensor_a_scorch = Tensor.from_torch(random_tensor_a, "A").to_sparse("ds")
    tensor_b_scorch = Tensor.from_torch(random_tensor_b, "B")

    time_dict = {}
    start_time = time.time()
    scorch_result = matmul_wksp(
        tensor_a_scorch, tensor_b_scorch, format="ds", time_dict=time_dict
    )
    scorch_total_time = time.time() - start_time
    scorch_eval_time = time_dict["eval_time"]

    # Assert that the results are the same
    assert torch.allclose(torch_result, scorch_result.to_torch())

    print(f"torch time: {torch_time}")
    print(f"scorch total time: {scorch_total_time}")
    print(f"scorch eval time: {scorch_eval_time}")
    print(f"scorch eval time / torch time: {scorch_eval_time / torch_time}")


def todo_test_spmm_dd_oo_dd_wksp_time():
    n = 100
    sparsity = 0.99
    random_tensor_a = torch.rand(n, n).float()
    random_tensor_b = torch.rand(n, n).float()

    # Randomly sparsify each tensor
    random_tensor_a = random_tensor_a * (torch.rand(n, n) > sparsity)
    random_tensor_b = random_tensor_b * (torch.rand(n, n) > sparsity)

    random_tensor_a_csr = random_tensor_a.to_sparse_coo()
    # Convert random_tensor_a to a sparse COO pytorch tensor
    start_time = time.time()
    # torch_result = torch.matmul(random_tensor_a, random_tensor_b).to_sparse_coo()
    torch_result = torch.sparse.mm(random_tensor_a_csr, random_tensor_b)
    torch_time = time.time() - start_time

    tensor_a_scorch = Tensor.from_torch(random_tensor_a, "A").to_sparse("oo")
    tensor_b_scorch = Tensor.from_torch(random_tensor_b, "B")

    time_dict = {}
    start_time = time.time()
    scorch_result = matmul_wksp(
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


def test_spmm_dd_ds_dd_wksp_time():
    n = 200
    sparsity = 0.99
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
    scorch_result = matmul_wksp(
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


def test_spmm_dd_oo_dd_time():
    """
    Compare speed of torch and scorch matmul
    Use random tensors
    """
    n = 64
    sparsity = 0.99
    random_tensor_a = torch.rand(n, n).float()
    random_tensor_b = torch.rand(n, n).float()

    # Randomly sparsify each tensor
    random_tensor_a = random_tensor_a * (torch.rand(n, n) > sparsity)
    random_tensor_b = random_tensor_b * (torch.rand(n, n) > sparsity)

    # Convert random_tensor_a to a sparse COO pytorch tensor
    random_tensor_a_coo = random_tensor_a.to_sparse()
    start_time = time.time()
    torch_result = torch.matmul(random_tensor_a_coo, random_tensor_b)
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


def test_matmul_time():
    """
    Compare speed of torch and scorch matmul
    Use random tensors
    """
    n = 100
    sparsity = 0.9
    random_tensor_a = torch.rand(n, n)
    random_tensor_b = torch.rand(n, n)

    # Randomly sparsify each tensor
    random_tensor_a = random_tensor_a * (torch.rand(n, n) > sparsity).float()
    random_tensor_b = random_tensor_b * (torch.rand(n, n) > sparsity).float()

    start_time = time.time()
    torch_result = torch.matmul(random_tensor_a, random_tensor_b)
    torch_time = time.time() - start_time

    tensor_a_scorch = Tensor.from_torch(random_tensor_a, "A").to_sparse()
    tensor_b_scorch = Tensor.from_torch(random_tensor_b, "B").to_sparse()

    start_time = time.time()
    scorch_result = matmul(tensor_a_scorch, tensor_b_scorch, format="ds")
    scorch_time = time.time() - start_time

    print(f"torch time: {torch_time}")
    print(f"scorch time: {scorch_time}")
    print(f"scorch time / torch time: {scorch_time / torch_time}")


def test_matmul_dd_dd_dd():
    n = 64
    tensor_a_torch = torch.rand(n, n)
    tensor_b_torch = torch.rand(n, n)

    a_scorch = Tensor.from_torch(tensor_a_torch, "A").to_dense()
    b_scorch = Tensor.from_torch(tensor_b_torch, "B").to_dense()

    scorch_result = einsum("ik,kj->ij", a_scorch, b_scorch, format="dd")

    result_torch = torch.matmul(tensor_a_torch, tensor_b_torch)
    scorch_result_torch = scorch_result.to_torch()

    assert result_torch.allclose(scorch_result_torch)


def test_matmul_ds_ds_ds():
    n = 64
    tensor_a_torch = torch.rand(n, n)
    tensor_b_torch = torch.rand(n, n)

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ds")

    result = matmul(a_sparse, b_sparse, output_format="ds")
    result_torch = torch.matmul(tensor_a_torch, tensor_b_torch)

    assert torch.allclose(result.to_torch(), result_torch)


def test_spmm_ds_ds_ds_random():
    dim_n = 10
    sparsity = 0.9
    tensor_a_torch = torch.rand(dim_n, dim_n)
    tensor_b_torch = torch.rand(dim_n, dim_n)

    # Randomly sparsify each tensor
    tensor_a_torch = tensor_a_torch * (torch.rand(dim_n, dim_n) > sparsity).float()
    tensor_b_torch = tensor_b_torch * (torch.rand(dim_n, dim_n) > sparsity).float()

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ds")

    result = matmul_wksp(a_sparse, b_sparse, output_format="ds")
    result_torch = torch.matmul(tensor_a_torch, tensor_b_torch)

    result_to_torch = result.to_torch()

    assert torch.allclose(result_to_torch, result_torch)

    if not torch.allclose(result_to_torch, result_torch):
        print("\n\nresult_torch:")
        print(result_torch.tolist())
        print("\n\nresult:")
        print(result.index.mode_indices[1][0].tolist())
        print(result.index.mode_indices[1][1].tolist())
        print(result.values.tolist())
        print("\n\nresult_to_torch:")
        print(result_to_torch.tolist())
        print("\n\nresult_to_torch - result_torch:")
        print((result_to_torch - result_torch).tolist())


def test_spmm_ss_dd_dd_ikj_gustavson():
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "dense"])

    workspace = Workspace(
        name="wksp",
        dim=1,
    )

    """
    A[i, j] = ForAll(i,
      Where(
        producer=ForAll(k, ForAll(j, 1[j] += B[i, k] * C[k, j])),
        consumer=ForAll(j, A[i, j] = workspace[j])
      )
    )
    """

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[j],
                        B[i, k] * C[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    workspace[j],
                ),
            ),
        ),
    )

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    cpp_code = llir_lowerer.lower_llir(lowered_llir)

    print(cpp_code)

    # Read header_cpp_code from csrc/header.cpp
    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    module = torch.utils.cpp_extension.load_inline(
        name="kernel",
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=["-O3"],
    )

    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_dense()
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_dense()

    output_format = parse_format("ss")

    result_shape = (5, 5)
    args = [result_shape]

    for tensor in [a_sparse, b_sparse]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor._storage._index.mode_indices)  # type: ignore
        args.append(tensor._storage.value)  # type: ignore

    result_cpp = module.evaluate(*args)
    result = Tensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format=output_format,
        ),
        value=result_cpp._storage._value,
    )

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    result_torch = torch.matmul(tensor_a_torch, tensor_b_torch)
    # print(result.index.mode_indices)
    assert result.to_torch().tolist() == result_torch.tolist()


def test_spmm_ds_dd_dd_ikj_gustavson():
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "dense"])

    workspace = Workspace(
        name="wksp",
        dim=1,
    )

    """
    A[i, j] = ForAll(i,
      Where(
        producer=ForAll(k, ForAll(j, 1[j] += B[i, k] * C[k, j])),
        consumer=ForAll(j, A[i, j] = workspace[j])
      )
    )
    """

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[j],
                        B[i, k] * C[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    workspace[j],
                ),
            ),
        ),
    )

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    cpp_code = llir_lowerer.lower_llir(lowered_llir)

    print(cpp_code)

    # Read header_cpp_code from csrc/header.cpp
    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    module = torch.utils.cpp_extension.load_inline(
        name="kernel",
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=["-O3"],
    )

    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_dense()
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_dense()

    output_format = parse_format("ds")

    result_shape = (5, 5)
    args = [result_shape]

    for tensor in [a_sparse, b_sparse]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor._storage._index.mode_indices)  # type: ignore
        args.append(tensor._storage.value)  # type: ignore

    result_cpp = module.evaluate(*args)
    result = Tensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format=output_format,
        ),
        value=result_cpp._storage._value,
    )

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    result_torch = torch.matmul(tensor_a_torch, tensor_b_torch)
    assert result.to_torch().tolist() == result_torch.tolist()


def test_spmm_ds_ds_ds_ikj_gustavson_random():
    dim_n = 50

    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    workspace = Workspace(
        name="wksp",
        dim=1,
    )

    """
    A[i, j] = ForAll(i,
      Where(
        producer=ForAll(k, ForAll(j, 1[j] += B[i, k] * C[k, j])),
        consumer=ForAll(j, A[i, j] = workspace[j])
      )
    )
    """

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[j],
                        B[i, k] * C[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    workspace[j],
                ),
            ),
        ),
    )

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    cpp_code = llir_lowerer.lower_llir(lowered_llir)

    print(cpp_code)

    # Read header_cpp_code from csrc/header.cpp
    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    module = torch.utils.cpp_extension.load_inline(
        name="kernel",
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=["-O3"],
    )

    tensor_a_torch = torch.rand(dim_n, dim_n)
    tensor_b_torch = torch.rand(dim_n, dim_n)

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ds")

    output_format = parse_format("ds")

    result_shape = (dim_n, dim_n)
    args = [result_shape]

    for tensor in [a_sparse, b_sparse]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor._storage._index.mode_indices)  # type: ignore
        args.append(tensor._storage.value)  # type: ignore

    result_cpp = module.evaluate(*args)
    result = Tensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format=output_format,
        ),
        value=result_cpp._storage._value,
    )

    assert result.shape == (dim_n, dim_n)
    assert len(result.index.mode_indices) == 2

    result_torch = torch.matmul(tensor_a_torch, tensor_b_torch)
    assert result_torch.flatten().allclose(result.to_torch().flatten())


def test_spmm_ss_ds_ds_ikj_gustavson():
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    workspace = Workspace(
        name="wksp",
        dim=1,
    )

    """
    A[i, j] = ForAll(i,
      Where(
        producer=ForAll(k, ForAll(j, 1[j] += B[i, k] * C[k, j])),
        consumer=ForAll(j, A[i, j] = workspace[j])
      )
    )
    """

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[j],
                        B[i, k] * C[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    workspace[j],
                ),
            ),
        ),
    )

    lowerer = CINLowerer()

    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()

    cpp_code = llir_lowerer.lower_llir(lowered_llir)

    print(cpp_code)

    # Read header_cpp_code from csrc/header.cpp
    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    module = torch.utils.cpp_extension.load_inline(
        name="kernel",
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=["-O3"],
    )

    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ds")

    output_format = parse_format("ss")

    result_shape = (5, 5)
    args = [result_shape]

    for tensor in [a_sparse, b_sparse]:
        args.append(tensor.shape)  # type: ignore
        args.append(tensor._storage._index.mode_indices)  # type: ignore
        args.append(tensor._storage.value)  # type: ignore

    result_cpp = module.evaluate(*args)
    result = Tensor(
        shape=result_shape,
        index=TensorIndex(
            mode_indices=result_cpp._storage._index.mode_indices,
            tensor_format=output_format,
        ),
        value=result_cpp._storage._value,
    )

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[0][0].tolist() == [0, 5]
    assert result.index.mode_indices[0][1].tolist() == [0, 1, 2, 3, 4]

    assert result.index.mode_indices[1][0].tolist() == [0, 5, 7, 9, 11, 13]
    assert result.index.mode_indices[1][1].tolist() == [
        0,
        1,
        2,
        3,
        4,
        0,
        1,
        0,
        2,
        0,
        3,
        0,
        4,
    ]
    assert result.values.tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        4.0,
        4.0,
        9.0,
        9.0,
        16.0,
        16.0,
        25.0,
        25.0,
    ]


def test_spmm_ss_ds_ds():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ds")

    # result = einsum("ik,kj->ij", a_sparse, b_sparse, format="ss")
    result = matmul_wksp(a_sparse, b_sparse, output_format="ss")

    assert result.shape == (5, 5)
    assert len(result.index.mode_indices) == 2

    assert result.index.mode_indices[0][0].tolist() == [0, 5]
    assert result.index.mode_indices[0][1].tolist() == [0, 1, 2, 3, 4]

    assert result.index.mode_indices[1][0].tolist() == [0, 5, 7, 9, 11, 13]
    assert result.index.mode_indices[1][1].tolist() == [
        0,
        1,
        2,
        3,
        4,
        0,
        1,
        0,
        2,
        0,
        3,
        0,
        4,
    ]
    assert result.values.tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        4.0,
        4.0,
        9.0,
        9.0,
        16.0,
        16.0,
        25.0,
        25.0,
    ]


def test_matmul_dd_ds_dd():
    n = 64
    tensor_a_torch = torch.rand(n, n)
    tensor_b_torch = torch.rand(n, n)

    a_scorch = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_scorch = Tensor.from_torch(tensor_b_torch, "B")

    result = matmul(a_scorch, b_scorch, format="dd")

    result_torch = torch.matmul(tensor_a_torch, tensor_b_torch)

    assert torch.allclose(result.to_torch(), result_torch)


def test_spmm_dd_ds_ds():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    a_sparse = Tensor.from_torch(tensor_a_torch, "A").to_sparse("ds")
    b_sparse = Tensor.from_torch(tensor_b_torch, "B").to_sparse("ds")

    result = einsum("ik,kj->ij", a_sparse, b_sparse, format="dd")

    assert result.shape == (5, 5)

    assert result.values.tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        4.0,
        4.0,
        0.0,
        0.0,
        0.0,
        9.0,
        0.0,
        9.0,
        0.0,
        0.0,
        16.0,
        0.0,
        0.0,
        16.0,
        0.0,
        25.0,
        0.0,
        0.0,
        0.0,
        25.0,
    ]


def test_spmm_dd_multi_multi():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    tensor_a = Tensor.from_torch(tensor_a_torch, "A")
    tensor_b = Tensor.from_torch(tensor_b_torch, "B")

    input_formats = ["ds", "ss", "oo"]

    input_format_pairs = list(product(input_formats, input_formats))

    for format_a, format_b in input_format_pairs:
        a_sparse = tensor_a.to_sparse(format_a)
        b_sparse = tensor_b.to_sparse(format_b)

        result = einsum("ik,kj->ij", a_sparse, b_sparse, format="dd")

        print("Input formats: ", format_a, format_b)
        print("Output format: ", result.format)

        assert result.shape == (5, 5)

        assert result.values.tolist() == [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            4.0,
            4.0,
            0.0,
            0.0,
            0.0,
            9.0,
            0.0,
            9.0,
            0.0,
            0.0,
            16.0,
            0.0,
            0.0,
            16.0,
            0.0,
            25.0,
            0.0,
            0.0,
            0.0,
            25.0,
        ]


def test_spmm_ds_multi_multi():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    tensor_a = Tensor.from_torch(tensor_a_torch, "A")
    tensor_b = Tensor.from_torch(tensor_b_torch, "B")

    input_formats = ["ds", "ss", "oo"]

    input_format_pairs = list(product(input_formats, input_formats))

    for format_a, format_b in input_format_pairs:
        a_sparse = tensor_a.to_sparse(format_a)
        b_sparse = tensor_b.to_sparse(format_b)

        # result = einsum("ik,kj->ij", a_sparse, b_sparse, format="ds")
        result = matmul_wksp(a_sparse, b_sparse, output_format="ds")

        print("Input formats: ", format_a, format_b)
        print("Output format: ", result.format)

        assert result.shape == (5, 5)
        assert len(result.index.mode_indices) == 2

        assert result.index.mode_indices[1][0].tolist() == [0, 5, 7, 9, 11, 13]
        assert result.index.mode_indices[1][1].tolist() == [
            0,
            1,
            2,
            3,
            4,
            0,
            1,
            0,
            2,
            0,
            3,
            0,
            4,
        ]

        assert result.values.tolist() == [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            4.0,
            4.0,
            9.0,
            9.0,
            16.0,
            16.0,
            25.0,
            25.0,
        ]


def test_spmm_ss_multi_multi():
    tensor_a_torch = torch.Tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ]
    )
    tensor_b_torch = torch.Tensor(
        [
            [1, 2, 3, 4, 5],
            [2, 2, 0, 0, 0],
            [3, 0, 3, 0, 0],
            [4, 0, 0, 4, 0],
            [5, 0, 0, 0, 5],
        ]
    )

    tensor_a = Tensor.from_torch(tensor_a_torch, "A")
    tensor_b = Tensor.from_torch(tensor_b_torch, "B")

    input_formats = ["ds", "ss", "oo"]

    input_format_pairs = list(product(input_formats, input_formats))

    for format_a, format_b in input_format_pairs:
        a_sparse = tensor_a.to_sparse(format_a)
        b_sparse = tensor_b.to_sparse(format_b)

        # result = einsum("ik,kj->ij", a_sparse, b_sparse, format="ss")
        result = matmul_wksp(a_sparse, b_sparse, output_format="ss")

        print("Input formats: ", format_a, format_b)
        print("Output format: ", result.format)

        assert result.shape == (5, 5)
        assert len(result.index.mode_indices) == 2

        assert result.index.mode_indices[0][0].tolist() == [0, 5]
        assert result.index.mode_indices[0][1].tolist() == [0, 1, 2, 3, 4]

        assert result.index.mode_indices[1][0].tolist() == [0, 5, 7, 9, 11, 13]
        assert result.index.mode_indices[1][1].tolist() == [
            0,
            1,
            2,
            3,
            4,
            0,
            1,
            0,
            2,
            0,
            3,
            0,
            4,
        ]
        assert result.values.tolist() == [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            4.0,
            4.0,
            9.0,
            9.0,
            16.0,
            16.0,
            25.0,
            25.0,
        ]


def test_sddmm_ds_ds_dd_dd():
    """
    A[i, j] = B[i, j] * C[i, k] * D[k, j]
    A: CSR
    B: CSR
    C: Dense
    D: Dense
    """
    n = 64
    sparsity = 0.9
    random_tensor_b = torch.rand(n, n)
    random_tensor_c = torch.rand(n, n)
    random_tensor_d = torch.rand(n, n)
    # Sparsify B
    random_tensor_b = random_tensor_b * (torch.rand(n, n) > sparsity)

    start_time = time.time()
    torch_result = torch.einsum(
        "ij,ik,kj->ij", random_tensor_b, random_tensor_c, random_tensor_d
    )

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
        format="ds",
    )
    scorch_total_time = time.time() - start_time
    scorch_eval_time = time_dict["eval_time"]

    assert torch.allclose(torch_result, scorch_result.to_torch())

    print(f"torch time: {torch_time}")
    print(f"scorch total time: {scorch_total_time}")
    print(f"scorch eval time: {scorch_eval_time}")
    print(f"scorch eval time / torch time: {scorch_eval_time / torch_time}")


def test_spmv_d_oo_d():
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
    scorch_result = einsum(
        "ij,j->i",
        tensor_a_scorch,
        tensor_x_scorch,
        time_dict=time_dict,
        format="d",
    )
    # scorch_result = matmul(tensor_a_scorch, tensor_x_scorch, time_dict=time_dict)
    scorch_total_time = time.time() - start_time
    scorch_eval_time = time_dict["eval_time"]

    assert torch.allclose(torch_result, scorch_result.values)

    print(f"torch time: {torch_time}")
    print(f"scorch total time: {scorch_total_time}")
    print(f"scorch eval time: {scorch_eval_time}")
    print(f"scorch eval time / torch time: {scorch_eval_time / torch_time}")
