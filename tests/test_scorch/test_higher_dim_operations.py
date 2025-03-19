"""
Tests for higher dimensional tensor operations (3D and 4D) in the CIN compiler.
"""

from scorch.compiler.cin import (
    IndexVar,
    TensorVar,
    ForAll,
    TensorAssign,
    Operation,
)

from tests.test_scorch.test_helpers import (
    lower_and_print,
    create_index_vars,
    create_tensor_vars,
    create_elementwise_operation,
)


# 3D Tensor Operations

def test_elemwise_mul_3d_tensor_dss_sss_sss():
    """
    Elementwise 3D tensor multiplication
    A[i, j, k] = B[i, j, k] * C[i, j, k]
    taco "A(i,j,k) = B(i,j,k) * C(i,j,k)" -f=A:dss -f=B:sss -f=C:sss -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse"])

    A[i, j, k] = B[i, j, k] * C[i, j, k]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, A._assignment)))

    lower_and_print(cin_stmt)


def test_ttm_ddd_dds_dd_ijkm():
    """
    Tensor-times-matrix multiplication (TTM)
    C[i, j, m] = A[i, j, k] * B[k, m]
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    m = IndexVar("m")

    C = TensorVar("C", fmt="ddd")
    A = TensorVar("A", fmt="dds")
    B = TensorVar("B", fmt="dd")

    C[i, j, m] = A[i, j, k] * B[k, m]

    C._assignment.op = Operation.ADD

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(m, C._assignment))))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_mttkrp_dd_sss_dd_dd_ijkm():
    """
    Matricized tensor times Khatri-Rao product (MTTKRP)
    D[i, m] = A[i, j, k] * B[j, m] * C[k, m]
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    m = IndexVar("m")

    D = TensorVar("D", fmt="dd")
    A = TensorVar("A", fmt="sss")
    B = TensorVar("B", fmt="dd")
    C = TensorVar("C", fmt="dd")

    D[i, m] = A[i, j, k] * B[j, m] * C[k, m]

    D._assignment.op = Operation.ADD

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(m, D._assignment))))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


# 4D Tensor Operations

def test_convert_4d_ssss_oooo():
    """
    Convert from COO format to CSF format for 4D tensor
    taco "A(i,j,k,l) = B(i,j,k,l)" -f=A:ssss:0,1,2,3 -f=B:uccq:0,1,2,3 -print-evaluate
    """
    # Create index variables
    i, j, k, l = create_index_vars("i", "j", "k", "l")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["sparse", "sparse", "sparse", "sparse"],
        "B": ["coord", "coord", "coord", "coord"]
    })

    # Create the assignment
    tensors["A"][i, j, k, l] = tensors["B"][i, j, k, l]

    # Create the CIN statement
    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, tensors["A"]._assignment))))

    # Lower and print with filter_zeros=True
    lower_and_print(cin_stmt, filter_zeros=True)


def test_elemwise_mul_4d_oooo_ssss_ssss():
    """
    Elementwise 4D tensor multiplication with COO and CSF formats
    taco "A(i,j,k,l)=B(i,j,k,l)*C(i,j,k,l)" -f=A:uccq:0,1,2,3 -f=B:ssss:0,1,2,3 -f=C:ssss:0,1,2,3 -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["coord", "coord", "coord", "coord"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lower_and_print(cin_stmt)


def test_elemwise_mul_4d_ssss_ssss_ssss():
    """
    Elementwise 4D tensor multiplication with CSF format
    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ssss -f=B:ssss -f=C:ssss -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["sparse", "sparse", "sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lower_and_print(cin_stmt)


def test_elemwise_mul_4d_ssss_oooo_oooo():
    """
    Elementwise 4D tensor multiplication with mixed formats
    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ssss -f=B:oooo -f=C:oooo -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["sparse", "sparse", "sparse", "sparse"])
    B = TensorVar("B", fmt=["coord", "coord", "coord", "coord"])
    C = TensorVar("C", fmt=["coord", "coord", "coord", "coord"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lower_and_print(cin_stmt)


def test_elemwise_mul_4d_ssdd_ssss_ssss():
    """
    Elementwise 4D tensor multiplication with mixed formats
    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ssdd -f=B:ssss -f=C:ssss -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["sparse", "sparse", "dense", "dense"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lower_and_print(cin_stmt)


def test_elemwise_mul_4d_dddd_ssss_ssss():
    """
    Elementwise 4D tensor multiplication with mixed formats
    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:dddd -f=B:ssss -f=C:ssss -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["dense", "dense", "dense", "dense"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lower_and_print(cin_stmt)


def test_elemwise_mul_4d_ddss_ssss_ssss():
    """
    Elementwise 4D tensor multiplication with mixed formats
    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ddss -f=B:ssss -f=C:ssss -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["dense", "dense", "sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lower_and_print(cin_stmt)


def test_elemwise_mul_4d_ddss_dddd_ssss():
    """
    Elementwise 4D tensor multiplication with mixed formats
    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ddss -f=B:dddd -f=C:ssss -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    l = IndexVar("l")

    A = TensorVar("A", fmt=["dense", "dense", "sparse", "sparse"])
    B = TensorVar("B", fmt=["dense", "dense", "dense", "dense"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse", "sparse"])

    A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, ForAll(l, A._assignment))))

    lower_and_print(cin_stmt)
