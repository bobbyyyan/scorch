"""
Tests for 1D tensor operations in the CIN compiler.
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
)

def test_elemwise_mul_1d_sss():
    """
    Elementwise vector multiplication code generation
    a[i] = b[i] * c[i]
    taco "a(i) = b(i)*c(i)" -f=a:s:0 -f=b:s:0 -f=c:s:0 -print-evaluate
    Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)*c(i)&format=a:s:0;b:s:0;c:s:0
    """
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
    """
    Elementwise vector multiplication code generation
    a[i] = b[i] * c[i]
    taco "a(i) = b(i)*c(i)" -f=a:d -f=b:s -f=c:s -print-evaluate
    """
    i = IndexVar("i")

    a = TensorVar("a", fmt="d")
    b = TensorVar("b", fmt="s")
    c = TensorVar("c", fmt="s")

    a[i] = b[i] * c[i]

    cin_stmt = ForAll(i, a._assignment)

    lower_and_print(cin_stmt)


def test_elemwise_mul_1d_sds():
    """
    Elementwise vector multiplication code generation
    a[i] = b[i] * c[i]
    Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)*c(i)&format=a:s:0;b:d:0;c:s:0
    """
    i = IndexVar("i")

    a = TensorVar("a", fmt="s")
    b = TensorVar("b", fmt="d")
    c = TensorVar("c", fmt="s")

    a[i] = b[i] * c[i]

    cin_stmt = ForAll(i, a._assignment)

    lower_and_print(cin_stmt)


def test_elemwise_mul_add_1d_sssd():
    """
    Elementwise vector multiplication and addition code generation
    Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)*c(i)+d(i)&format=a:s:0;b:s:0;c:s:0;d:d:0
    a[i] = b[i] * c[i] + d[i]
    """
    i = IndexVar("i")

    a = TensorVar("a", fmt="s")
    b = TensorVar("b", fmt="s")
    c = TensorVar("c", fmt="s")
    d = TensorVar("d", fmt="d")

    a[i] = b[i] * c[i] + d[i]

    cin_stmt = ForAll(i, a._assignment)

    lower_and_print(cin_stmt)


def test_elemwise_add_1d_sss():
    """
    Elementwise vector addition code generation
    a[i] = b[i] + c[i]
    """
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
    """
    Elementwise vector addition code generation
    a[i] = b[i] + c[i]
    taco "a(i) = b(i)+c(i)" -f=a:d -f=b:s -f=c:s -print-evaluate
    """
    i = IndexVar("i")

    a = TensorVar("a", fmt="d")
    b = TensorVar("b", fmt="s")
    c = TensorVar("c", fmt="s")

    a[i] = b[i] + c[i]

    cin_stmt = ForAll(i, a._assignment)

    lower_and_print(cin_stmt)


def test_elemwise_add_1d_sds():
    """
    Elementwise vector addition code generation
    a[i] = b[i] + c[i]
    Reference: http://tensor-compiler.org/codegen.html?expr=a(i)=b(i)+c(i)&format=a:s:0;b:d:0;c:s:0
    """
    i = IndexVar("i")

    a = TensorVar("a", fmt="s")
    b = TensorVar("b", fmt="d")
    c = TensorVar("c", fmt="s")

    a[i] = b[i] + c[i]

    cin_stmt = ForAll(i, a._assignment)

    lower_and_print(cin_stmt)
