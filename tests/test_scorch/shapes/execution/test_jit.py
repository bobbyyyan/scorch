import torch
import scorch.tensor as tensor
from scorch.compiler.shapes.jit.compile import compile
from scorch.compiler.shapes.jit.ir import *


def test_binary_operations():
    @compile
    def Foo(A, B):
        C = A * B
        D = copy(A)
        E = C + D
        return E

    a = torch.Tensor([1, 2, 3, 4, 5])
    b = torch.Tensor([1, 2, 3, 4, 5])
    assert torch.allclose(Foo(a, b), (a * b) + a)


def test_add_zero():
    @compile
    def Foo(A, B):
        C = A + B        # [1, 2, 3]
        D = C * B        # [0, 0, 0]
        E = D + A        # [1, 2, 3]
        F = E + (E * B)  # [1, 2, 3]
        return F

    a = tensor.Tensor.from_torch(torch.Tensor([1, 2, 3]), "A").to_sparse("s")
    b = tensor.Tensor.from_torch(torch.Tensor([0, 0, 0]), "B").to_sparse("s")
    assert torch.allclose(Foo(a, b), torch.Tensor([1, 2, 3]))
