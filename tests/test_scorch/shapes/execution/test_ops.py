import torch
import sys

from scorch import tensor
from scorch.compiler.shapes import compile, ops

# Test Reshape Operations against their PyTorch equivalent.


def test_slice_1d():
    t: torch.Tensor = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7])
    input = tensor.Tensor.from_torch(t, "IN").to_sparse("s")
    assert torch.equal(
        ops.slice(input, dim=0, start=0, end=8, stride=2, format="d").to_torch(),
        t[0:8:2],
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

    input = tensor.Tensor.from_torch(t, "IN").to_sparse("ds")
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

    input = tensor.Tensor.from_torch(t, "IN").to_sparse("ds")
    assert torch.equal(
        ops.slice(input, dim=1, start=0, end=2, stride=1, format="dd").to_torch(),
        t[:, 0:2:1],
    )


def test_flatten_2d():
    t: torch.Tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    input = tensor.Tensor.from_torch(t, "IN").to_sparse("ds")
    assert torch.equal(ops.flatten(input, dim=0, format="d").to_torch(), t.flatten())


def test_unflatten_1d():
    t: torch.Tensor = torch.Tensor([1, 2, 3, 4, 5, 6])
    input = tensor.Tensor.from_torch(t, "IN").to_sparse("d")
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


def test_mul_1d_ss():  # Test 0
    a: torch.Tensor = torch.Tensor([1, 0, 3, 4, 5, 0])
    b: torch.Tensor = torch.Tensor([3, 4, 5, 0, 7, 8])
    lhs = tensor.Tensor.from_torch(a, "A").to_sparse("s")
    rhs = tensor.Tensor.from_torch(b, "B").to_sparse("s")
    assert torch.equal(
        ops.mul(lhs, rhs, format="d").to_torch(),
        a * b,
    )


def test_muladd_1d():  # Test 1
    b: torch.Tensor = torch.Tensor([0, 2, 3, 4, 0, 6])
    c: torch.Tensor = torch.Tensor([3, 0, 5, 6, 0, 8])
    d: torch.Tensor = torch.Tensor([10, 11, 0, 0, 14, 0])
    B = tensor.Tensor.from_torch(b, "B").to_sparse("s")
    C = tensor.Tensor.from_torch(c, "C").to_sparse("s")
    D = tensor.Tensor.from_torch(d, "D").to_sparse("s")
    assert torch.equal(
        ops.generic_vector([ops.Op.ADD, B, ops.Op.MUL, C, D], format="d").to_torch(),
        b + (c * d),
    )


def test_mul_1d_dd():  # Test 2
    a: torch.Tensor = torch.Tensor([0, 2, 3, 0, 5, 6])
    b: torch.Tensor = torch.Tensor([3, 4, 0, 6, 7, 0])
    lhs = tensor.Tensor.from_torch(a, "A")
    rhs = tensor.Tensor.from_torch(b, "B")
    assert torch.equal(
        ops.mul(lhs, rhs, format="d").to_torch(),
        a * b,
    )


def test_add_1d_dd():  # Test 3
    a: torch.Tensor = torch.Tensor([1, 0, 0, 4, 5, 6])
    b: torch.Tensor = torch.Tensor([3, 4, 0, 6, 7, 8])
    lhs = tensor.Tensor.from_torch(a, "A")
    rhs = tensor.Tensor.from_torch(b, "B")
    assert torch.equal(
        ops.add(lhs, rhs, format="d").to_torch(),
        a + b,
    )


def test_mul_1d_sd():  # Test 4
    a: torch.Tensor = torch.Tensor([0, 2, 3, 4, 5, 0])
    b: torch.Tensor = torch.Tensor([0, 0, 2, 3, 0, 8])
    lhs = tensor.Tensor.from_torch(a, "A").to_sparse("s")
    rhs = tensor.Tensor.from_torch(b, "B")
    assert torch.equal(
        ops.mul(lhs, rhs, format="d").to_torch(),
        a * b,
    )


def test_addmul_1d():  # Test 5
    b: torch.Tensor = torch.Tensor([1, 0, 3, 4, 0, 6])
    c: torch.Tensor = torch.Tensor([3, 4, 0, 6, 0, 8])
    d: torch.Tensor = torch.Tensor([10, 11, 0, 0, 14, 15])
    B = tensor.Tensor.from_torch(b, "B").to_sparse("s")
    C = tensor.Tensor.from_torch(c, "C")
    D = tensor.Tensor.from_torch(d, "D").to_sparse("s")
    assert torch.equal(
        ops.generic_vector([ops.Op.MUL, ops.Op.ADD, B, C, D], format="d").to_torch(),
        (b + c) * d,
    )
