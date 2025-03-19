"""
Tests for 2D tensor operations in the CIN compiler.
"""

from scorch.compiler.cin import (
    IndexVar,
    TensorVar,
    ForAll,
)

from tests.test_scorch.test_helpers import (
    lower_and_print,
    create_index_vars,
    create_tensor_vars,
    create_elementwise_operation,
    create_matrix_vector_operation,
)


def test_convert_dd_ds():
    """
    Test converting a CSR matrix to a dense matrix.
    A[i, j] = B[i, j]
    """
    # Create index variables
    i, j = create_index_vars("i", "j")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": "dd",
        "B": "ds"
    })

    # Create the assignment
    tensors["A"][i, j] = tensors["B"][i, j]

    # Create the CIN statement
    cin_stmt = ForAll(i, ForAll(j, tensors["A"]._assignment))

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_elemwise_mul_2d_ss_ss_ss():
    """
    Elementwise matrix multiplication code generation
    A[i, j] = B[i, j] * C[i, j]
    taco "A(i,j) = B(i,j) * C(i,j)" -f=A:ss -f=B:ss -f=C:ss -print-evaluate
    """
    # Create index variables
    i, j = create_index_vars("i", "j")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["sparse", "sparse"],
        "B": ["sparse", "sparse"],
        "C": ["sparse", "sparse"]
    })

    # Create the CIN statement for elementwise multiplication
    cin_stmt = create_elementwise_operation(tensors, (i, j), operation_type="mul")

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_elemwise_mul_2d_oo_oo_oo():
    """
    Elementwise matrix multiplication with coordinate formats
    taco "A(i,j)=B(i,j)*C(i,j)" -f=A:uq:0,1 -f=B:uq:0,1 -f=C:uq:0,1 -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["coord", "coord"])
    B = TensorVar("B", fmt=["coord", "coord"])
    C = TensorVar("C", fmt=["coord", "coord"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lower_and_print(cin_stmt)


def test_elemwise_mul_2d_ds_dd_ds():
    """
    Elementwise matrix multiplication with mixed formats
    taco "A(i,j) = B(i,j) * C(i,j)" -f=A:ds -f=B:dd -f=C:ds -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lower_and_print(cin_stmt)


def test_elemwise_mul_2d_dd_dd_ds():
    """
    Elementwise matrix multiplication with mixed formats
    taco "A(i,j) = B(i,j) * C(i,j)" -f=A:dd -f=B:dd -f=C:ds -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["dense", "dense"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lower_and_print(cin_stmt)


def test_elemwise_mul_2d_ds_ss_ss():
    """
    Elementwise matrix multiplication with mixed formats
    taco "A(i,j) = B(i,j) * C(i,j)" -f=A:ds -f=B:ss -f=C:ss -print-evaluate
    """
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lower_and_print(cin_stmt)


def test_elemwise_mul_2d_ss_dd_ds():
    """
    Elementwise matrix multiplication with mixed formats
    A[i, j] = B[i, j] * C[i, j]
    A: sparse, sparse
    B: dense, dense
    C: dense, sparse
    """
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lower_and_print(cin_stmt)


def test_elemwise_mul_2d_codegen_2():
    """
    Elementwise matrix multiplication code generation
    A[i, j] = B[i, j] * C[i, j]
    """
    # Create index variables
    i, j = create_index_vars("i", "j")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["sparse", "sparse"],
        "B": ["sparse", "dense"],
        "C": ["sparse", "dense"]
    })

    # Create the CIN statement for elementwise multiplication
    cin_stmt = create_elementwise_operation(tensors, (i, j), operation_type="mul")

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_elemwise_mul_2d_diff_order_codegen():
    """
    Elementwise matrix multiplication with different index ordering
    http://tensor-compiler.org/codegen.html?expr=A(i,j)=B(i,j)*C(j,i)&format=A:ss:0,1;B:dd:0,1;C:dd:0,1
    A[i, j] = B[i, j] * C[j, i]
    A: sparse, sparse
    B: dense, dense
    C: dense, dense
    """
    # Create index variables
    i, j = create_index_vars("i", "j")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["sparse", "sparse"],
        "B": ["dense", "dense"],
        "C": ["dense", "dense"]
    })

    # Define index mapping for C tensor (j,i instead of i,j)
    index_maps = {
        "C": [j, i]  # Reversed order of indices
    }

    # Create the CIN statement for elementwise multiplication with custom index ordering
    cin_stmt = create_elementwise_operation(tensors, (i, j), operation_type="mul", index_maps=index_maps)

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_elemwise_matrix_add_ss_ss_ss():
    """
    Elementwise matrix addition
    A[i, j] = B[i, j] + C[i, j]
    taco "A(i,j) = B(i,j) + C(i,j)" -f=A:ss -f=B:ss -f=C:ss -print-evaluate
    """
    # Create index variables
    i, j = create_index_vars("i", "j")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["sparse", "sparse"],
        "B": ["sparse", "sparse"],
        "C": ["sparse", "sparse"]
    })

    # Create the CIN statement for elementwise addition
    cin_stmt = create_elementwise_operation(tensors, (i, j), operation_type="add")

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_elemwise_matrix_add_ds_ss_ss():
    """
    Elementwise matrix addition
    A[i, j] = B[i, j] + C[i, j]
    taco "A(i,j) = B(i,j) + C(i,j)" -f=A:ds -f=B:ss -f=C:ss -print-evaluate
    """
    # Create index variables
    i, j = create_index_vars("i", "j")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["dense", "sparse"],
        "B": ["sparse", "sparse"],
        "C": ["sparse", "sparse"]
    })

    # Create the CIN statement for elementwise addition
    cin_stmt = create_elementwise_operation(tensors, (i, j), operation_type="add")

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_elemwise_matrix_add_sd_ss_ss():
    """
    Elementwise matrix addition
    A[i, j] = B[i, j] + C[i, j]
    taco "A(i,j) = B(i,j) + C(i,j)" -f=A:sd -f=B:ss -f=C:ss -print-evaluate
    """
    # Create index variables
    i, j = create_index_vars("i", "j")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["sparse", "dense"],
        "B": ["sparse", "sparse"],
        "C": ["sparse", "sparse"]
    })

    # Create the CIN statement for elementwise addition
    cin_stmt = create_elementwise_operation(tensors, (i, j), operation_type="add")

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_elemwise_add_2d_ds_ds():
    """
    Elementwise matrix addition with dense-sparse formats
    """
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="ds")
    C = TensorVar("C", fmt="ds")

    C[i, j] = A[i, j] + B[i, j]

    cin_stmt = ForAll(i, ForAll(j, C._assignment))

    lower_and_print(cin_stmt)


def test_elemwise_matrix_add_mul_codegen():
    """
    Complex elementwise matrix expression
    A[i, j] = (B[i, j] + C[i, j]) * (D[i, j] + E[i, j])
    TACO: http://tensor-compiler.org/codegen.html?expr=A(i,j)=(B(i,j)+C(i,j))*(D(i,j)+E(i,j))&format=A:ss:0,1;B:ds:0,1;C:ds:0,1;D:ss:0,1;E:ss:0,1
    """
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])
    D = TensorVar("D", fmt=["sparse", "sparse"])
    E = TensorVar("E", fmt=["sparse", "sparse"])

    A[i, j] = (B[i, j] + C[i, j]) * (D[i, j] + E[i, j])

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_matrix_vector_mul_codegen():
    """
    Matrix vector multiplication code generation
    A[i, j] = B[i, j] * C[j]
    Reference: http://tensor-compiler.org/codegen.html?expr=A(i,j)=B(i,j)*c(j)&format=A:ss:0,1;B:ss:0,1;c:s:0
    """
    # Create index variables
    i, j = create_index_vars("i", "j")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["sparse", "sparse"],
        "B": ["sparse", "sparse"],
        "C": ["sparse"]
    })

    # Create the CIN statement for matrix-vector multiplication
    # The vector uses dimension 1 (j index)
    cin_stmt = create_matrix_vector_operation(tensors, (i, j), vector_dim_index=1, operation_type="mul")

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_ij_i_j_ss_s_s():
    """
    Outer product of two vectors to form a matrix
    taco "A(i, j) = B(i) * C(j)" -f=A:ss -f=B:s -f=C:s -print-evaluate
    """
    # Create index variables
    i, j = create_index_vars("i", "j")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["sparse", "sparse"],
        "B": ["sparse"],
        "C": ["sparse"]
    })

    # Create the assignment
    A = tensors["A"]
    B = tensors["B"]
    C = tensors["C"]

    A[i, j] = B[i] * C[j]

    # Create the CIN statement
    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_coo_to_csr():
    """
    Convert COO format to CSR format
    A[i, j] = B[i, j]
    """
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="oo")

    A[i, j] = B[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)
