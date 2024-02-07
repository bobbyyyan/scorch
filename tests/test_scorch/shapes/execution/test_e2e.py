import torch

from scorch import tensor
from scorch.compiler import cin
from scorch.compiler.shapes import cfir, codegen, cpp, compile
from scorch.format import LevelType
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import tests.utility as util

# Compiles CIN -> CFIR -> CPP and then executes the function, verifying
# the resulting tensor has the correct values.


def test_assign_1d_d():
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
