import torch
import scorch.tensor as tensor
import scorch.compiler.shapes.jit.compile as jit
from scorch.compiler.shapes.jit.ir import *
import tests.utility as util


def graph(func):
    """Returns the IR graph post-compilation, for testing purposes."""

    def wrapper(*args, **kwargs) -> str:
        name: str = func.__name__
        module: ScorchModule = ScorchModule("module")
        region: ScorchRegion = ScorchRegion(name, module)
        _ = jit._compile(func, region, args, kwargs)
        return region

    return wrapper


def test_zero_simplifications():
    @graph
    def Bar(A, B):
        C = A + B  # [1, 2, 3]
        D = C * B  # [0, 0, 0]
        E = D + A  # [1, 2, 3]
        F = E + (E * B)  # [1, 2, 3]
        return F

    a = tensor.Tensor.from_torch(torch.Tensor([1, 2, 3]), "A").to_sparse("s")
    b = tensor.Tensor.from_torch(torch.Tensor([0, 0, 0]), "B").to_sparse("s")
    # Verify multiplications and additions by zero are simplified.
    util.assert_equal(
        Bar(a, b),
        """
        $Bar:
          %0 = A[3:s]
        """,
    )


def test_simplify_concat_spmv():
    @graph
    def Foo(A1, A2, b):
        return matmul(concat(A1, A2, dim=1), b)

    _A1 = torch.Tensor([[0, 2], [3, 4]])
    _A2 = torch.Tensor([[5, 6], [0, 8]])
    _b = torch.Tensor([1, 2, 3, 4])
    A1 = tensor.Tensor.from_torch(_A1, "A1").to_sparse("ds")
    A2 = tensor.Tensor.from_torch(_A2, "A2").to_sparse("sd")
    b = tensor.Tensor.from_torch(_b, "b")
    # Verify the concatenation is simplified to avoid poor asymptotic complexity.
    util.assert_equal(
        Foo(A1, A2, b),
        """
        $Foo:
          %0 = A1[2:d,2:s]
          %1 = A2[2:s,2:d]
          %2 = b[4:d]
          %5 = slice.0 %2[0:2:1]
          %6 = matmul %0, %5
          %7 = slice.0 %2[2:4:1]
          %8 = matmul %1, %7
          %9 = add %6, %8""",
    )


def test_fusion_1d():
    @graph
    def Foo(A, B):
        C = A * B
        D = C + A
        E = D + D
        F = E + A
        return F

    a = torch.Tensor([0, 2, 0, 0, 0])
    b = torch.Tensor([1, 2, 1, 1, 1])
    util.assert_equal(
        Foo(a, b),
        """
        $Foo:
           %0 = _T0[5:d]
           %1 = _T1[5:d]
           %8 = add (add (add (mul %0, %1), %0), (add (mul %0, %1), %0)), %0
        """,
    )


def test_fusion_with_slice():
    @graph
    def Foo(A, B, C):
        D = A * B
        E = D + C[0:5:1]
        return E

    a = torch.Tensor([0, 2, 0, 0, 0])
    b = torch.Tensor([1, 2, 1, 1, 1])
    c = torch.Tensor([1, 2, 3, 4, 5, 6, 7])
    util.assert_equal(
        Foo(a, b, c),
        """
        $Foo:
           %0 = _T0[5:d]
           %1 = _T1[5:d]
           %2 = _T2[7:d]
           %6 = add (mul %0, %1), slice %2[0:5:1]
        """,
    )
