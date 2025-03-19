"""
Helper functions:
- lower_and_print: Handles lowering and code generation for CIN statements
- create_index_vars: Creates IndexVar objects for tensor operations
- create_tensor_vars: Creates TensorVar objects with the specified formats
- create_elementwise_operation: Creates elementwise operations like multiplication and addition
- create_matrix_vector_operation: Creates matrix-vector operations
- create_matrix_multiplication: Creates matrix multiplication operations
- create_workspace_operation: Creates operations that use workspaces for intermediate results

Each test function follows a similar pattern:
1. Create index variables
2. Create tensor variables
3. Define the operation using the helper functions
4. Lower the CIN statement and generate code
"""

from scorch.compiler.cin import (
    IndexVar,
    TensorVar,
    ForAll,
    TensorAssign,
    TensorAccess,
    Operation,
    Workspace,
    Where,
    TileSizeVar,
)
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.scheduler import Scheduler


def lower_and_print(cin_stmt, filter_zeros=False, no_comments=False):
    """
    Helper function to lower a CIN statement and print the generated C++ code.

    Args:
        cin_stmt: The CIN statement to lower
        filter_zeros: Whether to filter zeros in the CINLowerer
        no_comments: Whether to include comments in the output code

    Returns:
        The generated C++ code
    """
    print("\nCIN statement:")
    print(cin_stmt)

    lowerer = CINLowerer(filter_zeros=filter_zeros)
    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)

    llir_lowerer = LLIRLowerer()
    cpp_code = llir_lowerer.lower_llir(lowered_llir, no_comments=no_comments)

    print("\nC++ torch extension code:")
    print(cpp_code)

    return cpp_code


def create_index_vars(*names):
    """
    Create index variables with the given names.

    Args:
        *names: Variable number of strings representing index variable names

    Returns:
        A tuple of IndexVar objects
    """
    return tuple(IndexVar(name) for name in names)


def create_tensor_vars(tensor_specs):
    """
    Create tensor variables with the given specifications.

    Args:
        tensor_specs: A dict mapping tensor names to format specifications
            Format can be a string or a list of strings

    Returns:
        A dict mapping tensor names to TensorVar objects
    """
    return {name: TensorVar(name, fmt=fmt) for name, fmt in tensor_specs.items()}


def create_elementwise_operation(tensors, index_vars, operation_type="mul", index_maps=None):
    """
    Create a common elementwise operation (multiplication or addition) on tensors.

    Args:
        tensors: Dict mapping tensor names to TensorVar objects
        index_vars: Tuple of IndexVar objects
        operation_type: String, either "mul" or "add"
        index_maps: Optional dict mapping tensor names to lists of index expressions
                    to allow for different index ordering per tensor

    Returns:
        A CIN statement representing the operation
    """
    # Get the tensor names (assume first one is result, rest are operands)
    tensor_names = list(tensors.keys())
    result_name = tensor_names[0]
    operand_names = tensor_names[1:]

    # Create the assignment
    result = tensors[result_name]
    indices = list(index_vars)

    # Create the right-hand side of the assignment
    first_operand = operand_names[0]
    if index_maps and first_operand in index_maps:
        first_indices = index_maps[first_operand]
    else:
        first_indices = indices

    rhs = tensors[first_operand][first_indices]

    for operand_name in operand_names[1:]:
        if index_maps and operand_name in index_maps:
            operand_indices = index_maps[operand_name]
        else:
            operand_indices = indices

        operand = tensors[operand_name][operand_indices]
        if operation_type == "mul":
            rhs = rhs * operand
        else:  # add
            rhs = rhs + operand

    # Create the assignment
    result[indices] = rhs

    # Nest the ForAll statements
    stmt = result._assignment
    for idx in reversed(indices):
        stmt = ForAll(idx, stmt)

    return stmt


def create_matrix_vector_operation(tensors, index_vars, vector_dim_index, operation_type="mul", preserve_vector_dim=True):
    """
    Create a matrix-vector operation on tensors.

    Args:
        tensors: Dict mapping tensor names to TensorVar objects.
            Expected to have a result tensor, a matrix, and a vector.
        index_vars: Tuple of IndexVar objects.
        vector_dim_index: The index of the dimension that the vector uses.
        operation_type: String, either "mul" or "add"
        preserve_vector_dim: Whether to keep the vector dimension in the result tensor.
                            Set to True for operations like A[i,j] = B[i,j] * C[j]
                            Set to False for operations like A[i] = B[i,j] * C[j]

    Returns:
        A CIN statement representing the operation
    """
    # Get the tensor names (assume first one is result, rest are operands)
    tensor_names = list(tensors.keys())
    result_name = tensor_names[0]
    matrix_name = tensor_names[1]
    vector_name = tensor_names[2]

    # Create the assignment
    result = tensors[result_name]
    matrix = tensors[matrix_name]
    vector = tensors[vector_name]

    # Create indices for the result, matrix, and vector
    matrix_indices = list(index_vars)
    vector_indices = [index_vars[vector_dim_index]]

    # For preserving all dimensions in the result
    if preserve_vector_dim:
        result_indices = list(index_vars)
    else:
        # For dropping the vector dimension from the result
        result_indices = [idx for i, idx in enumerate(index_vars) if i != vector_dim_index]

    # Create the right-hand side of the assignment
    if operation_type == "mul":
        rhs = matrix[matrix_indices] * vector[vector_indices]
    else:  # add
        rhs = matrix[matrix_indices] + vector[vector_indices]

    # Create the assignment
    result[result_indices] = rhs

    # Nest the ForAll statements
    stmt = result._assignment
    for idx in reversed(index_vars):
        stmt = ForAll(idx, stmt)

    return stmt


def create_matrix_multiplication(tensors, index_vars, contract_idx_pos, op=Operation.ADD):
    """
    Create a matrix multiplication operation.

    Args:
        tensors: Dict mapping tensor names to TensorVar objects.
            Expected to have a result matrix and two operand matrices.
        index_vars: Tuple of IndexVar objects (i, j, k) where k is the contraction index.
        contract_idx_pos: The position of the contraction index in the index_vars tuple.
        op: The operation to use for the assignment (e.g., Operation.ADD).

    Returns:
        A CIN statement representing the matrix multiplication
    """
    # Get the tensor names (assume first one is result, rest are operands)
    tensor_names = list(tensors.keys())
    result_name = tensor_names[0]
    left_matrix_name = tensor_names[1]
    right_matrix_name = tensor_names[2]

    # Get the tensors
    result = tensors[result_name]
    left = tensors[left_matrix_name]
    right = tensors[right_matrix_name]

    # Get the indices
    i, j, k = index_vars

    # Create the assignment
    result[i, j] = left[i, k] * right[k, j]
    result._assignment.op = op

    # Create the nested ForAll statement with the proper loop order (i, k, j)
    # This is a common loop order for SpMM operations
    cin_stmt = ForAll(i, ForAll(k, ForAll(j, result._assignment)))

    return cin_stmt


def create_workspace_operation(tensors, workspace, index_vars, operation_type="mul"):
    """
    Create an operation that uses a workspace for intermediate results.
    Common pattern for sparse matrix operations like SpMM with the Gustavson algorithm.

    Args:
        tensors: Dict mapping tensor names to TensorVar objects.
            Expected to have a result tensor and two operand tensors.
        workspace: The workspace object to use.
        index_vars: Tuple of IndexVar objects (i, j, k).
        operation_type: String, either "mul" or "add"

    Returns:
        A CIN statement using a workspace with Gustavson pattern
    """
    # Get the tensor names
    tensor_names = list(tensors.keys())
    result_name = tensor_names[0]
    left_name = tensor_names[1]
    right_name = tensor_names[2]

    # Get the tensors
    result = tensors[result_name]
    left = tensors[left_name]
    right = tensors[right_name]

    # Get the indices
    i, j, k = index_vars

    # Create the producer statement: accumulates B[i,k] * C[k,j] into workspace[j]
    if operation_type == "mul":
        rhs = left[i, k] * right[k, j]
    else:  # add
        rhs = left[i, k] + right[k, j]

    producer = ForAll(
        k,
        ForAll(
            j,
            TensorAssign(
                workspace[j],
                rhs,
                op=Operation.ADD,
            ),
        ),
    )

    # Create the consumer statement: assigns workspace[j] to A[i,j]
    consumer = ForAll(
        j,
        TensorAssign(
            result[i, j],
            workspace[j],
        ),
    )

    # Combine with Where and wrap in the outer ForAll(i)
    cin_stmt = ForAll(i, Where(producer=producer, consumer=consumer))

    return cin_stmt


def test_convert_dd_ds():
    # Test converting a CSR matrix to a dense matrix
    # A[i, j] = B[i, j]

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


def test_elemwise_mul_1d_sss():
    # elementwise vector multiplication code generation
    # a[i] = b[i] * c[i]
    # taco "a(i) = b(i)*c(i)" -f=a:s:0 -f=b:s:0 -f=c:s:0 -print-evaluate
    # Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)*c(i)&format=a:s:0;b:s:0;c:s:0

    # Create index variables
    i = create_index_vars("i")[0]

    # Create tensor variables
    tensors = create_tensor_vars({
        "a": "s",
        "b": "s",
        "c": "s"
    })

    # Create the CIN statement for elementwise multiplication
    cin_stmt = create_elementwise_operation(tensors, (i,), operation_type="mul")

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_elemwise_mul_1d_dss():
    # elementwise vector multiplication code generation
    # a[i] = b[i] * c[i]
    # taco "a(i) = b(i)*c(i)" -f=a:d -f=b:s -f=c:s -print-evaluate

    i = IndexVar("i")

    a = TensorVar("a", fmt="d")
    b = TensorVar("b", fmt="s")
    c = TensorVar("c", fmt="s")

    a[i] = b[i] * c[i]

    cin_stmt = ForAll(i, a._assignment)

    lower_and_print(cin_stmt)


def test_elemwise_mul_1d_sds():
    # elementwise vector multiplication code generation
    # a[i] = b[i] * c[i]
    # Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)*c(i)&format=a:s:0;b:d:0;c:s:0

    i = IndexVar("i")

    a = TensorVar("a", fmt="s")
    b = TensorVar("b", fmt="d")
    c = TensorVar("c", fmt="s")

    a[i] = b[i] * c[i]

    cin_stmt = ForAll(i, a._assignment)

    lower_and_print(cin_stmt)


def test_elemwise_mul_add_1d_sssd():
    # elementwise vector multiplication and addition code generation
    # Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)*c(i)+d(i)&format=a:s:0;b:s:0;c:s:0;d:d:0
    # a[i] = b[i] * c[i] + d[i]
    i = IndexVar("i")

    a = TensorVar("a", fmt="s")
    b = TensorVar("b", fmt="s")
    c = TensorVar("c", fmt="s")
    d = TensorVar("d", fmt="d")

    a[i] = b[i] * c[i] + d[i]

    cin_stmt = ForAll(i, a._assignment)

    lower_and_print(cin_stmt)


def test_elemwise_add_1d_sss():
    # elementwise vector addition code generation
    # a[i] = b[i] + c[i]

    # Create index variables
    i = create_index_vars("i")[0]

    # Create tensor variables
    tensors = create_tensor_vars({
        "a": "s",
        "b": "s",
        "c": "s"
    })

    # Create the CIN statement for elementwise addition
    cin_stmt = create_elementwise_operation(tensors, (i,), operation_type="add")

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_elemwise_add_1d_dss():
    # elementwise vector addition code generation
    # a[i] = b[i] + c[i]
    # taco "a(i) = b(i)+c(i)" -f=a:d -f=b:s -f=c:s -print-evaluate
    i = IndexVar("i")

    a = TensorVar("a", fmt="d")
    b = TensorVar("b", fmt="s")
    c = TensorVar("c", fmt="s")

    a[i] = b[i] + c[i]

    cin_stmt = ForAll(i, a._assignment)

    lower_and_print(cin_stmt)


def test_elemwise_add_2d_ds_ds():
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="ds")
    C = TensorVar("C", fmt="ds")

    C[i, j] = A[i, j] + B[i, j]

    cin_stmt = ForAll(i, ForAll(j, C._assignment))

    lower_and_print(cin_stmt)


def test_elemwise_add_1d_sds():
    # elementwise vector addition code generation
    # a[i] = b[i] + c[i]
    # Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)+c(i)&format=a:s:0;b:d:0;c:s:0
    i = IndexVar("i")

    a = TensorVar("a", fmt="s")
    b = TensorVar("b", fmt="d")
    c = TensorVar("c", fmt="s")

    a[i] = b[i] + c[i]

    cin_stmt = ForAll(i, a._assignment)

    lower_and_print(cin_stmt)


def test_elemwise_mul_3d_tensor_dss_sss_sss():
    # A[i, j, k] = B[i, j, k] * C[i, j, k]
    # taco "A(i,j,k) = B(i,j,k) * C(i,j,k)" -f=A:dss -f=B:sss -f=C:sss -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "sparse", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse", "sparse"])

    A[i, j, k] = B[i, j, k] * C[i, j, k]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, A._assignment)))

    lower_and_print(cin_stmt)


def test_convert_4d_ssss_oooo():
    # taco "A(i,j,k,l) = B(i,j,k,l)" -f=A:ssss:0,1,2,3 -f=B:uccq:0,1,2,3 -print-evaluate

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
    # taco "A(i,j,k,l)=B(i,j,k,l)*C(i,j,k,l)" -f=A:uccq:0,1,2,3 -f=B:ssss:0,1,2,3 -f=C:ssss:0,1,2,3 -print-evaluate
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
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ssss -f=B:ssss -f=C:ssss -print-evaluate
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
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ssss -f=B:oooo -f=C:oooo -print-evaluate
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
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ssdd -f=B:ssss -f=C:ssss -print-evaluate
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
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:dddd -f=B:ssss -f=C:ssss -print-evaluate
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
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ddss -f=B:ssss -f=C:ssss -print-evaluate
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


def test_elemwise_mul_2d_ss_dd_ds():
    # A[i, j] = B[i, j] * C[i, j]
    # A: sparse, sparse
    # B: dense, dense
    # C: dense, sparse
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["sparse", "sparse"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lower_and_print(cin_stmt)


def test_matrix_vector_mul_codegen():
    # matrix vector multiplication code generation
    # A[i, j] = B[i, j] * C[j]
    # Reference: http://tensor-compiler.org/codegen.html?expr=A(i,j)=B(i,j)*c(j)&format=A:ss:0,1;B:ss:0,1;c:s:0

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


def test_elemwise_mul_2d_ss_ss_ss():
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[i, j]
    # taco "A(i,j) = B(i,j) * C(i,j)" -f=A:ss -f=B:ss -f=C:ss -print-evaluate

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
    # taco "A(i,j)=B(i,j)*C(i,j)" -f=A:uq:0,1 -f=B:uq:0,1 -f=C:uq:0,1 -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["coord", "coord"])
    B = TensorVar("B", fmt=["coord", "coord"])
    C = TensorVar("C", fmt=["coord", "coord"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lower_and_print(cin_stmt)


def test_elemwise_mul_2d_ds_dd_ds():
    # taco "A(i,j) = B(i,j) * C(i,j)" -f=A:ds -f=B:dd -f=C:ds -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lower_and_print(cin_stmt)


def test_elemwise_mul_2d_dd_dd_ds():
    # taco "A(i,j) = B(i,j) * C(i,j)" -f=A:dd -f=B:dd -f=C:ds -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["dense", "dense"])
    B = TensorVar("B", fmt=["dense", "dense"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lower_and_print(cin_stmt)


def test_elemwise_mul_2d_ds_ss_ss():
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[i, j]
    # taco "A(i,j) = B(i,j) * C(i,j)" -f=A:ds -f=B:ss -f=C:ss -print-evaluate

    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["sparse", "sparse"])
    C = TensorVar("C", fmt=["sparse", "sparse"])

    A[i, j] = B[i, j] * C[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    lower_and_print(cin_stmt)


def test_elemwise_matrix_add_ss_ss_ss():
    # elementwise matrix addition
    # A[i, j] = B[i, j] + C[i, j]
    # taco "A(i,j) = B(i,j) + C(i,j)" -f=A:ss -f=B:ss -f=C:ss -print-evaluate

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
    # elementwise matrix addition
    # A[i, j] = B[i, j] + C[i, j]
    # taco "A(i,j) = B(i,j) + C(i,j)" -f=A:ds -f=B:ss -f=C:ss -print-evaluate

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
    # elementwise matrix addition
    # A[i, j] = B[i, j] + C[i, j]
    # taco "A(i,j) = B(i,j) + C(i,j)" -f=A:sd -f=B:ss -f=C:ss -print-evaluate

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


def test_elemwise_mul_2d_codegen_2():
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[i, j]

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
    # http://tensor-compiler.org/codegen.html?expr=A(i,j)=B(i,j)*C(j,i)&format=A:ss:0,1;B:dd:0,1;C:dd:0,1
    # elementwise matrix multiplication code generation
    # A[i, j] = B[i, j] * C[j, i]
    # A: sparse, sparse
    # B: dense, dense
    # C: dense, dense

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


def test_elemwise_matrix_add_mul_codegen():
    # elementwise matrix multiplication code generation
    # A[i, j] = (B[i, j] + C[i, j]) * (D[i, j] + E[i, j])
    # TACO reference: http://tensor-compiler.org/codegen.html?expr=A(i,j)=(B(i,j)+C(i,j))*(D(i,j)+E(i,j))&format=A:ss:0,1;B:ds:0,1;C:ds:0,1;D:ss:0,1;E:ss:0,1

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


def test_sddmm_codegen():
    # taco "A(i, j) = B(i, j) * C(i, k) * D(k, j)" -f=A:ds -f=B:ds -f=C:dd -f=D:dd -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="ds")
    C = TensorVar("C", fmt="dd")
    D = TensorVar("D", fmt="dd")

    A[i, j] = B[i, j] * C[i, k] * D[k, j]

    cin_stmt = ForAll(i, ForAll(k, ForAll(j, A._assignment)))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmv_codegen():
    i = IndexVar("i")
    j = IndexVar("j")

    y = TensorVar("y", fmt=["dense"])
    A = TensorVar("A", fmt=["dense", "sparse"])
    x = TensorVar("x", fmt=["dense"])

    y[i] = A[i, j] * x[j]

    cin_stmt = ForAll(i, ForAll(j, y._assignment))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmv_wksp_codegen():
    # taco "y(i) = A(i, j) * x(j)" -f=y:d -f=A:ds -f=x:d -print-evaluate
    i, j = create_index_vars("i", "j")

    # Create tensor variables
    tensors = create_tensor_vars({
        "y": ["dense"],
        "A": ["dense", "sparse"],
        "x": ["dense"]
    })

    # Create a workspace
    workspace = Workspace(name="wksp", dim=0)

    # Create the CIN statement for SpMV with workspace
    # For SpMV: y[i] = A[i,j] * x[j]
    # Producer: ForAll(j, workspace += A[i,j] * x[j])
    # Consumer: y[i] = workspace

    producer = ForAll(
        j,
        TensorAssign(
            workspace.get_default_access(),
            tensors["A"][i, j] * tensors["x"][j],
            op=Operation.ADD,
        ),
    )

    consumer = TensorAssign(
        tensors["y"][i],
        workspace.get_default_access(),
    )

    cin_stmt = ForAll(
        i,
        Where(
            producer=producer,
            consumer=consumer,
        ),
    )

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_spmm_codegen():
    # taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate
    i, j, k = create_index_vars("i", "j", "k")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["dense", "sparse"],
        "B": ["dense", "sparse"],
        "C": ["dense", "sparse"]
    })

    # Create the CIN statement for sparse matrix multiplication
    cin_stmt = create_matrix_multiplication(tensors, (i, j, k), contract_idx_pos=2)

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_ttm_ddd_dds_dd_ijkm():
    # C[i, j, m] = A[i, j, k] * B[k, m]
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
    # D[i, m] = A[i, j, k] * B[j, m] * C[k, m]
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


def test_spmm_dd_dd_ds_ijk_gustavson():
    # taco "C(i, k) = A(i, j) * B(j, k)" -f=A:dd -f=B:dd -f=C:ds -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    C = TensorVar("C", fmt="dd")

    A = TensorVar("A", fmt="dd")
    B = TensorVar("B", fmt="ds")

    cin_stmt = ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(
                    C[i, k],
                    A[i, j] * B[j, k],
                    op=Operation.ADD,
                ),
            ),
        ),
    )

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_ds_ds_ds_kij_outer_workspace():
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "sparse"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    workspace = Workspace(
        name="wksp",
        dim=2,
    )

    """
    A[i, j] = Where(
      producer=ForAll(k, ForAll(i, ForAll(j, workspace[i, j] += B[i, k] * C[k, j]))),
      consumer=ForAll(i, ForAll(j, A[i, j] = workspace[i, j])),
    )
    """

    cin_stmt = Where(
        producer=ForAll(
            k,
            ForAll(
                i,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[i, j],
                        B[i, k] * C[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
        ),
        consumer=ForAll(
            i,
            ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    workspace[i, j],
                    op=Operation.ADD,
                ),
            ),
        ),
    )

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_ds_ds_ds_ikj_gustavson_workspace():
    # taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate
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

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_dd_ds_dd_ijk():
    # taco "C(i, k) = A(i, j) * B(j, k)" -f=C:dd -f=A:ds -f=B:dd -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    C = TensorVar("C", fmt="dd")
    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="dd")

    C[i, k] = A[i, j] * B[j, k]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, C._assignment)))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_dd_ds_dd_ijk_auto_tile():
    # taco "C(i, k) = A(i, j) * B(j, k)" -f=C:dd -f=A:ds -f=B:dd -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    C = TensorVar("C", fmt="dd")
    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="dd")

    C[i, k] = A[i, j] * B[j, k]

    cin_stmt = ForAll(i, ForAll(j, ForAll(k, C._assignment)))
    tiled_cin_stmt = Scheduler.add_tile(cin_stmt, k, 1024)

    print("\nTiled CIN statement:")
    print(tiled_cin_stmt)

    lower_and_print(tiled_cin_stmt)


def test_spmm_dd_ds_dd_tiled():
    """
    C[i, k] = A[i, j] * B[j, k]
    k gets tiled
    loop order: i, k_out, j, k_in
    accumulator: accum_c[k_in]
    ForAll i
      ForAll k_out
        Where(
            producer=ForAll j, ForAll k_in
                k = k_out + k_in
                (accum_c[k_in] += A[i, j] * B[j, k]),
            consumer=ForAll k_in
                k = k_out + k_in
                C[i, k] = accum_c[k_in]
        )
    """
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
    accum_c = Workspace("accum_c", dim=1, dense=True)

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

    print("\nCIN statement:")

    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_ds_ds_ds_ikj_gustavson_workspace():
    # taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="ds")
    C = TensorVar("C", fmt="ds")

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

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_dd_oo_dd_ikj_gustavson_workspace():
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt="dd")
    B = TensorVar("B", fmt="oo")
    C = TensorVar("C", fmt="dd")

    workspace = Workspace(
        name="wksp",
        dim=1,
    )

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

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_dd_ds_ds():
    # taco "A(i, j) = B(i, k) * C(k, j)" -f=A:dd -f=B:ds -f=C:ds -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["dense", "dense"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    cin_stmt = ForAll(
        i,
        ForAll(
            k,
            ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    B[i, k] * C[k, j],
                    op=Operation.ADD,
                ),
            ),
        ),
    )

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_spmm_sd_ds_ds():
    # taco "A(i, j) = B(i, k) * C(k, j)" -f=A:sd -f=B:ds -f=C:ds -print-evaluate
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    A = TensorVar("A", fmt=["sparse", "dense"])
    B = TensorVar("B", fmt=["dense", "sparse"])
    C = TensorVar("C", fmt=["dense", "sparse"])

    A[i, j] = B[i, k] * C[k, j]

    cin_stmt = ForAll(i, ForAll(k, ForAll(j, A._assignment)))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_ij_i_j_ss_s_s():
    # taco "A(i, j) = B(i) * C(j)" -f=A:ss -f=B:s -f=C:s -print-evaluate

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
    # A[i, j] = B[i, j]
    i = IndexVar("i")
    j = IndexVar("j")

    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="oo")

    A[i, j] = B[i, j]

    cin_stmt = ForAll(i, ForAll(j, A._assignment))

    print("\nCIN statement:")
    print(cin_stmt)

    lower_and_print(cin_stmt)


def test_elemwise_mul_4d_ddss_dddd_ssss():
    # A[i, j, k, l] = B[i, j, k, l] * C[i, j, k, l]
    # taco "A(i,j,k,l) = B(i,j,k,l) * C(i,j,k,l)" -f=A:ddss -f=B:dddd -f=C:ssss -print-evaluate

    # Create index variables
    i, j, k, l = create_index_vars("i", "j", "k", "l")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["dense", "dense", "sparse", "sparse"],
        "B": ["dense", "dense", "dense", "dense"],
        "C": ["sparse", "sparse", "sparse", "sparse"]
    })

    # Create the CIN statement for elementwise multiplication
    cin_stmt = create_elementwise_operation(tensors, (i, j, k, l), operation_type="mul")

    # Lower and print the generated code
    lower_and_print(cin_stmt)


def test_spmm_ikj_gustavson_workspace():
    # taco "A(i, j) = B(i, k) * C(k, j)" -f=A:ds -f=B:ds -f=C:ds -print-evaluate
    i, j, k = create_index_vars("i", "j", "k")

    # Create tensor variables
    tensors = create_tensor_vars({
        "A": ["dense", "sparse"],
        "B": ["dense", "sparse"],
        "C": ["dense", "sparse"]
    })

    # Create a workspace
    workspace = Workspace(name="wksp", dim=1)

    # Create the CIN statement for SpMM with Gustavson workspace pattern
    cin_stmt = create_workspace_operation(tensors, workspace, (i, j, k), operation_type="mul")

    # Lower and print the generated code
    lower_and_print(cin_stmt)
