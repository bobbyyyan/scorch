"""
Helper functions for testing CIN compiler functionality.
These functions help with creating and testing CIN statements.
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
