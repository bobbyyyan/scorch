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


def test_naming():
    @graph
    def Foo(A, B):
        C = A * B
        D = copy(A)
        E = C + D
        return E

    a = torch.Tensor([1, 2, 3, 4, 5])
    b = torch.Tensor([1, 2, 3, 4, 5])
    # Verify the IR gives each input tensor a unique name and SSA ordinal.
    util.assert_equal(
        Foo(a, b),
        """
    $Foo:
      %0 = _T0[5:d]
      %1 = _T1[5:d]
      %2 = mul %0, %1
      %3 = copy %0
      %4 = add %2, %3
    """,
    )


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


def test_simplify_concat():
    @graph
    def Foo(A1, A2, b):
        return matmul(concat(A1, A2, dim=1), b)

    _A1 = torch.Tensor([[0, 2], [3, 4]])
    _A2 = torch.Tensor([[5, 6], [0, 8]])
    _b = torch.Tensor([1, 2, 3, 4])
    A1 = tensor.Tensor.from_torch(_A1, "A1").to_sparse("ds")
    A2 = tensor.Tensor.from_torch(_A2, "A2").to_sparse("sd")
    b = tensor.Tensor.from_torch(_b, "b")
    # Verify the concatenation is simplified, since it
    # would require asymptotic behavior for traversal.
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
