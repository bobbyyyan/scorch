import torch

from scorch import tensor
from scorch.compiler.shapes import ops
from scorch.compiler.shapes.opcode import Opcode

# Test Reshape Operations against their PyTorch equivalent.


def test_slice_1d_s():
    t: torch.Tensor = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7])
    input = tensor.Tensor.from_torch(t, "A").to_sparse("s")
    assert torch.equal(
        ops.slice(input, dim=0, start=2, end=8, stride=2, format="d").to_torch(),
        t[2:8:2],
    )


def test_slice_1d_d():
    t: torch.Tensor = torch.Tensor([0, 1, 2, 3])
    input = tensor.Tensor.from_torch(t, "A")
    assert torch.equal(
        ops.slice(input, dim=0, start=2, end=4, stride=1, format="d").to_torch(),
        t[2:4:1],
    )


def test_slice_2d_dimension0():
    t: torch.Tensor = torch.Tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
        ]
    )

    input = tensor.Tensor.from_torch(t, "A").to_sparse("ds")
    assert torch.equal(
        ops.slice(input, dim=0, start=1, end=3, stride=1, format="dd").to_torch(),
        t[1:3:1],
    )


def test_slice_2d_dimension1():
    t: torch.Tensor = torch.Tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
        ]
    )

    input = tensor.Tensor.from_torch(t, "A").to_sparse("ds")
    assert torch.equal(
        ops.slice(input, dim=1, start=0, end=2, stride=1, format="dd").to_torch(),
        t[:, 0:2:1],
    )


def test_flatten_2d():
    t: torch.Tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    input = tensor.Tensor.from_torch(t, "A").to_sparse("ds")
    assert torch.equal(ops.flatten(input, dim=0, format="d").to_torch(), t.flatten())


def test_unflatten_1d():
    t: torch.Tensor = torch.Tensor([1, 2, 3, 4, 5, 6])
    input = tensor.Tensor.from_torch(t, "A").to_sparse("d")
    assert torch.equal(
        ops.unflatten(input, dim=0, sizes=(2, 3), format="dd").to_torch(),
        t.unflatten(dim=0, sizes=(2, 3)),
    )


def test_add_2d():
    a: torch.Tensor = torch.Tensor([[1, 2], [3, 4]])
    b: torch.Tensor = torch.Tensor([[5, 6], [7, 8]])
    lhs = tensor.Tensor.from_torch(a, "A")
    rhs = tensor.Tensor.from_torch(b, "B")
    assert torch.equal(
        ops.add(lhs, rhs, format="dd").to_torch(),
        a + b,
    )


def test_mul_1d_ss():
    a: torch.Tensor = torch.Tensor([1, 0, 3, 4, 5, 0])
    b: torch.Tensor = torch.Tensor([3, 4, 5, 0, 7, 8])
    lhs = tensor.Tensor.from_torch(a, "A").to_sparse("s")
    rhs = tensor.Tensor.from_torch(b, "B").to_sparse("s")
    assert torch.equal(
        ops.mul(lhs, rhs, format="d").to_torch(),
        a * b,
    )


def test_mul_1d_dd():
    a: torch.Tensor = torch.Tensor([0, 2, 3, 0, 5, 6])
    b: torch.Tensor = torch.Tensor([3, 4, 0, 6, 7, 0])
    lhs = tensor.Tensor.from_torch(a, "A")
    rhs = tensor.Tensor.from_torch(b, "B")
    assert torch.equal(
        ops.mul(lhs, rhs, format="d").to_torch(),
        a * b,
    )


def test_add_1d_dd():
    a: torch.Tensor = torch.Tensor([1, 0, 0, 4, 5, 6])
    b: torch.Tensor = torch.Tensor([3, 4, 0, 6, 7, 8])
    lhs = tensor.Tensor.from_torch(a, "A")
    rhs = tensor.Tensor.from_torch(b, "B")
    assert torch.equal(
        ops.add(lhs, rhs, format="d").to_torch(),
        a + b,
    )


def test_mul_1d_sd():
    a: torch.Tensor = torch.Tensor([0, 2, 3, 4, 5, 0])
    b: torch.Tensor = torch.Tensor([0, 0, 2, 3, 0, 8])
    lhs = tensor.Tensor.from_torch(a, "A").to_sparse("s")
    rhs = tensor.Tensor.from_torch(b, "B")
    assert torch.equal(
        ops.mul(lhs, rhs, format="d").to_torch(),
        a * b,
    )
