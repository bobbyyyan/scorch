import torch
import sys

from scorch import tensor
from scorch.compiler import cin
from scorch.compiler.shapes import cfir, codegen, cpp, compile, ops
from scorch.format import LevelType
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence

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
