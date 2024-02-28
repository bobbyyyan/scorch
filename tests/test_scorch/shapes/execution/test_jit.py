import torch
import scorch.tensor as tensor
from scorch.compiler.shapes.jit.compile import compile
from scorch.compiler.shapes.jit.ir import *


def test_binary_operations():
    @compile
    def Foo(A, B, C):
        D = A * B
        E = copy(A)
        F = D + C
        return F

    a = torch.Tensor([1, 2, 3, 4, 5])
    b = torch.Tensor([1, 2, 3, 4, 5])
    c = torch.Tensor([0, 0, 0, 0, 0])
    assert torch.allclose(Foo(a, b, c), (a * b) + c)


def test_zero_simplifications():
    @compile
    def Foo(A, B):
        C = A + B  # [1, 2, 3]
        D = C * B  # [0, 0, 0]
        E = D + A  # [1, 2, 3]
        F = E + (E * B)  # [1, 2, 3]
        return F

    a = tensor.Tensor.from_torch(torch.Tensor([1, 2, 3]), "A").to_sparse("s")
    b = tensor.Tensor.from_torch(torch.Tensor([0, 0, 0]), "B").to_sparse("s")
    assert torch.allclose(Foo(a, b), torch.Tensor([1, 2, 3]))


def test_simplify_concat_spmv():
    # TODO(cgyurgyik): This fails because CSC is not supported yet.
    # It will return `[[0, 0], [0, 0]]` for _A2.
    @compile
    def Foo(A1, A2, b):
        return matmul(concat(A1, A2, dim=1), b)

    _A1 = torch.Tensor([[0, 2], [3, 4]])
    _A2 = torch.Tensor([[5, 6], [0, 8]])
    _b = torch.Tensor([1, 2, 3, 4])
    A1 = tensor.Tensor.from_torch(_A1, "A1").to_sparse("ds")
    A2 = tensor.Tensor.from_torch(_A2, "A2").to_sparse("sd")
    b = tensor.Tensor.from_torch(_b, "b")
    assert torch.allclose(
        Foo(A1, A2, b), torch.matmul(torch.concat([_A1, _A2], dim=1), _b)
    )


def test_fusion_1d():
    @compile
    def Scorch(A, B, C):
        D = A * B
        E = C + D
        return E

    @torch.compile
    def Torch(A, B, C):
        D = A * B
        E = C + D
        return E

    a = torch.Tensor([0, 2, 0, 0, 0])
    b = torch.Tensor([1, 2, 1, 1, 1])
    c = torch.Tensor([1, 2, 3, 1, 2])
    assert torch.allclose(Scorch(a, b, c), Torch(a, b, c))


def test_fusion_hygienic_naming():
    @compile
    def Foo(A, B):
        C = A * B
        D = C + A
        return D

    a = torch.Tensor([0, 2, 0, 0, 0])
    b = torch.Tensor([1, 2, 1, 1, 1])
    assert torch.allclose(Foo(a, b), a * b + a)
