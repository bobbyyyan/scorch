import torch

from scorch import tensor
from scorch.compiler.shapes import ops
from scorch.compiler.shapes.opcode import Opcode

SPARSITY: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]  # Sparsity of the vectors.
N: int = 10  # Length of the vectors.


def dvector() -> torch.Tensor:
    """Returns a dense vector with random values."""
    return torch.rand((N,)).float()


def spvector(sparsity: int) -> torch.Tensor:
    """Returns a sparse vector with random values and the provided sparsity."""
    return dvector() * (torch.rand((N,)) > sparsity)


def test0():
    for sparsity in SPARSITY:
        a: torch.Tensor = spvector(sparsity)
        b: torch.Tensor = spvector(sparsity)
        lhs = tensor.Tensor.from_torch(a, "A").to_sparse("s")
        rhs = tensor.Tensor.from_torch(b, "B").to_sparse("s")
        assert torch.allclose(
            ops.mul(lhs, rhs, format="d").to_torch(),
            a * b,
        )


def test1():
    for sparsity in SPARSITY:
        b: torch.Tensor = spvector(sparsity)
        c: torch.Tensor = spvector(sparsity)
        d: torch.Tensor = spvector(sparsity)
        B = tensor.Tensor.from_torch(b, "B").to_sparse("s")
        C = tensor.Tensor.from_torch(c, "C").to_sparse("s")
        D = tensor.Tensor.from_torch(d, "D").to_sparse("s")
        assert torch.allclose(
            ops.generic_vector(
                [Opcode.ADD, B, Opcode.MUL, C, D], format="d"
            ).to_torch(),
            b + (c * d),
        )


def test2():
    a: torch.Tensor = dvector()
    b: torch.Tensor = dvector()
    lhs = tensor.Tensor.from_torch(a, "A")
    rhs = tensor.Tensor.from_torch(b, "B")
    assert torch.allclose(
        ops.mul(lhs, rhs, format="d").to_torch(),
        a * b,
    )


def test3():
    a: torch.Tensor = dvector()
    b: torch.Tensor = dvector()
    lhs = tensor.Tensor.from_torch(a, "A")
    rhs = tensor.Tensor.from_torch(b, "B")
    assert torch.allclose(
        ops.add(lhs, rhs, format="d").to_torch(),
        a + b,
    )


def test4():
    for sparsity in SPARSITY:
        a: torch.Tensor = spvector(sparsity)
        b: torch.Tensor = dvector()
        lhs = tensor.Tensor.from_torch(a, "A").to_sparse("s")
        rhs = tensor.Tensor.from_torch(b, "B")
        assert torch.allclose(
            ops.mul(lhs, rhs, format="d").to_torch(),
            a * b,
        )


def test5():
    for sparsity in SPARSITY:
        b: torch.Tensor = spvector(sparsity)
        c: torch.Tensor = dvector()
        d: torch.Tensor = spvector(sparsity)
        B = tensor.Tensor.from_torch(b, "B").to_sparse("s")
        C = tensor.Tensor.from_torch(c, "C")
        D = tensor.Tensor.from_torch(d, "D").to_sparse("s")
        assert torch.allclose(
            ops.generic_vector(
                [Opcode.MUL, Opcode.ADD, B, C, D], format="d"
            ).to_torch(),
            (b + c) * d,
        )
