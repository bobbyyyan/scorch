import torch

from scorch import tensor
from scorch.compiler.shapes import ops

# TODO(cgyurgyik): Remove this file after A2 is complete.

# Sparsity of the vectors.
SPARSITY: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
# Length of the vectors.
N: int = 10


def dvector() -> torch.Tensor:
    """Returns a dense vector with random values and the provided sparsity."""
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


def test1():  # Test 1
    for sparsity in SPARSITY:
        b: torch.Tensor = spvector(sparsity)
        c: torch.Tensor = spvector(sparsity)
        d: torch.Tensor = spvector(sparsity)
        B = tensor.Tensor.from_torch(b, "B").to_sparse("s")
        C = tensor.Tensor.from_torch(c, "C").to_sparse("s")
        D = tensor.Tensor.from_torch(d, "D").to_sparse("s")
        assert torch.allclose(
            ops.generic_vector([ops.Op.ADD, B, ops.Op.MUL, C, D], format="d").to_torch(),
            b + (c * d),
        )


def test2():  # Test 2
    for sparsity in SPARSITY:
        a: torch.Tensor = dvector()
        b: torch.Tensor = dvector()
        lhs = tensor.Tensor.from_torch(a, "A")
        rhs = tensor.Tensor.from_torch(b, "B")
        assert torch.allclose(
            ops.mul(lhs, rhs, format="d").to_torch(),
            a * b,
        )


def test3():  # Test 3
    for sparsity in SPARSITY:
        a: torch.Tensor = dvector()
        b: torch.Tensor = dvector()
        lhs = tensor.Tensor.from_torch(a, "A")
        rhs = tensor.Tensor.from_torch(b, "B")
        assert torch.allclose(
            ops.add(lhs, rhs, format="d").to_torch(),
            a + b,
        )


def test4():  # Test 4
    for sparsity in SPARSITY:
        a: torch.Tensor = spvector(sparsity)
        b: torch.Tensor = dvector()
        lhs = tensor.Tensor.from_torch(a, "A").to_sparse("s")
        rhs = tensor.Tensor.from_torch(b, "B")
        assert torch.allclose(
            ops.mul(lhs, rhs, format="d").to_torch(),
            a * b,
        )


def test5():  # Test 5
    for sparsity in SPARSITY:
        b: torch.Tensor = spvector(sparsity)
        c: torch.Tensor = dvector()
        d: torch.Tensor = spvector(sparsity)
        B = tensor.Tensor.from_torch(b, "B").to_sparse("s")
        C = tensor.Tensor.from_torch(c, "C")
        D = tensor.Tensor.from_torch(d, "D").to_sparse("s")
        assert torch.allclose(
            ops.generic_vector([ops.Op.MUL, ops.Op.ADD, B, C, D], format="d").to_torch(),
            (b + c) * d,
        )
