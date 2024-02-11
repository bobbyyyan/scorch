import torch

from scorch import tensor
from scorch.compiler import cin
from scorch.compiler.shapes import cfir, codegen, cpp, compile
from scorch.format import LevelType
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import tests.utility as util

# Compiles CIN -> CFIR -> CPP and then executes the function, verifying
# the resulting tensor has the correct values.


def test_assign_2d_dd_dd():
    A = cin.TensorVar("A", fmt=["d", "d"], shape=[2, 2])
    B = cin.TensorVar("B", fmt=["d", "d"], shape=[2, 2])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    c = cin.ForAll(
        i,
        cin.ForAll(
            j,
            A._assignment,
            cin.IndexSeq(j, B, size=2, index=1, format=LevelType.DENSE),
        ),
        cin.IndexSeq(i, B, size=2, index=0, format=LevelType.DENSE),
    )

    stmt: cpp.Cpp = compile.Compile(c)
    a = tensor.Tensor.from_torch(torch.zeros(size=(2, 2)), name="A")
    b = tensor.Tensor.from_torch(torch.Tensor([[1, 2], [3, 4]]), name="B")

    assert torch.equal(
        compile.CompileAndExecuteFunction(
            stmt=stmt, arguments=[b], result=a
        ).to_torch(),
        torch.Tensor([[1, 2], [3, 4]]),
    )


def test_assign_2d_dd_ds():
    A = cin.TensorVar("A", fmt=["d", "d"], shape=[2, 2])
    B = cin.TensorVar("B", fmt=["d", "s"], shape=[2, 2])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i, j] = B[i, j]

    c = cin.ForAll(
        i,
        cin.ForAll(
            j,
            A._assignment,
            cin.IndexSeq(j, B, size=2, index=1, format=LevelType.COMPRESSED),
        ),
        cin.IndexSeq(i, B, size=2, index=0, format=LevelType.DENSE),
    )

    stmt: cpp.Cpp = compile.Compile(c)
    a = tensor.Tensor.from_torch(torch.zeros(size=(2, 2)), name="A")
    b = tensor.Tensor.from_torch(torch.Tensor([[1, 2], [0, 4]]), name="B").to_sparse(
        "ds"
    )

    assert torch.equal(
        compile.CompileAndExecuteFunction(
            stmt=stmt, arguments=[b], result=a
        ).to_torch(),
        torch.Tensor([[1, 2], [0, 4]]),
    )


def test_slice():
    A = cin.TensorVar("A", fmt=["d"], shape=[8])
    B = cin.TensorVar("B", fmt=["s"], shape=[8])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    A[i] = B[j]

    c: cin.CIN = cin.ForAll(
        i,
        A._assignment,
        cin.SliceSeq(
            cin.IndexSeq(j, B, size=8, index=0, format=LevelType.COMPRESSED),
            start=0,
            end=8,
            stride=2,
        ),
    )

    stmt: cpp.Cpp = compile.Compile(c)
    a = tensor.Tensor.from_torch(torch.zeros(size=(8,)), name="A")
    b = tensor.Tensor.from_torch(
        torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7]), name="B"
    ).to_sparse("s")
    assert torch.equal(
        compile.CompileAndExecuteFunction(
            stmt=stmt, arguments=[b], result=a
        ).to_torch(),
        torch.Tensor([0, 2, 4, 6, 0, 0, 0, 0]),
    )


def test_collapse():
    A = cin.TensorVar("A", fmt=["d"], shape=[6])
    B = cin.TensorVar("B", fmt=["d", "s"], shape=[2, 3])
    i = cin.IndexVar("i")
    j = cin.IndexVar("j")
    k = cin.IndexVar("k")
    A[k] = B[i, j]

    Bi = cin.IndexSeq(i, B, size=2, index=0, format=LevelType.DENSE)
    Bj = cin.IndexSeq(j, B, size=3, index=1, format=LevelType.COMPRESSED)

    c: cin.CIN = cin.ForAll(k, A._assignment, cin.ProductSeq(Bi, Bj))
    stmt: cpp.Cpp = compile.Compile(c)
    a = tensor.Tensor.from_torch(torch.zeros((6,)), name="A")
    b = tensor.Tensor.from_torch(
        torch.Tensor([[1, 2, 3], [4, 5, 6]]), name="B"
    ).to_sparse("ds")
    assert torch.equal(
        compile.CompileAndExecuteFunction(
            stmt=stmt, arguments=[b], result=a
        ).to_torch(),
        torch.Tensor([1, 2, 3, 4, 5, 6]),
    )
