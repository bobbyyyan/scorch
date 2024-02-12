import torch

from scorch.compiler import cin
from scorch import tensor
from scorch.utils import parse_format
from scorch.compiler.shapes import cfir, codegen, cpp, compile
from scorch.format import LevelType, TensorFormat
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence


# Used to hygienically name code generated index variables.
TENSOR_INDEX_NAME = "ijklmnopqrstuvwxyz"


def slice(
    input: torch.Tensor | tensor.Tensor,
    dim: int,
    start: int,
    end: int,
    stride: Optional[int] = 1,
    format: Optional[Union[TensorFormat, str, List[str]]] = None,
) -> tensor.Tensor:
    """Extracts a uniform (possibly strided) subsequence from a dimension.

    input:  The tensor input to be sliced.
    dim:    The dimension in `input` to be sliced.
    start:  The beginning of the slice dimension.
    end:    The end of the slice dimension.
    stride: The (optional) stride of the slice.
    format: The format of the output tensor.

    For example,
      slice(input, dim=0, start=0, end=8, stride=2) == input[0:8:2]
    """
    inname: str = input.name if isinstance(input, tensor.Tensor) else "IN"
    if isinstance(input, torch.Tensor):
        input: tensor.Tensor = tensor.Tensor.from_torch(input)
    outname: str = "OUT"
    assert inname != outname

    B = cin.TensorVar(inname, shape=input.shape, fmt=input.format)
    if format is None:
        format = parse_format("d" * B.levels)
    elif not isinstance(format, TensorFormat):
        format = parse_format(format)
    # The result has the same shape as input save for the sliced dimension.
    A_shape = list(B.shape)
    A_shape[dim] = (end - start) // stride
    A = cin.TensorVar(outname, shape=tuple(A_shape), fmt=format)

    def ConstructSlice(A: cin.TensorVar, B: cin.TensorVar) -> cin.CIN:
        cins: List[Tuple[cin.IndexVar, cin.Seq]] = []
        input_indices: List[cin.IndexVar] = []
        for i, (format, size) in enumerate(zip(B.format.get_level_types(), B.shape)):
            s: str = TENSOR_INDEX_NAME[i]
            output_index = cin.IndexVar(s)
            input_index = cin.IndexVar(f"s{s}") if i == dim else output_index

            sequence = (
                cin.SliceSeq(
                    cin.IndexSeq(
                        input_index,
                        B,
                        size,
                        i,
                        format,
                    ),
                    start=start,
                    end=end,
                    stride=stride,
                )
                if i == dim
                else cin.IndexSeq(input_index, B, size, i, format)
            )
            input_indices.append(input_index)
            cins.append((output_index, sequence))
        output_indices: list[cin.IndexVar] = [idx for (idx, _) in cins]
        A[*output_indices] = B[*input_indices]

        (idx, seq) = cins.pop()
        stmt: cin.CIN = cin.ForAll(idx, A._assignment, seq)
        for idx, seq in reversed(cins):
            stmt = cin.ForAll(idx, stmt, seq)
        return stmt

    return compile.CompileAndExecuteFunction(
        stmt=compile.Compile(ConstructSlice(A, B)),
        arguments=[input],
        result=tensor.Tensor.from_torch(torch.zeros(A_shape), outname),
    )


def flatten(
    input: torch.Tensor | tensor.Tensor,
    dim: int,
    format: Optional[Union[TensorFormat, str, List[str]]] = None,
) -> tensor.Tensor:
    """Flattens the tensor.
    input: The input tensor to be flattene.d
    dim: The dimension in `input` to be flattened with `dim + 1`.
    format: The format of the result tensor.

    For example,
      flatten(input, dim=0) should be equivalent to torch.flatten(input, XXX).
    """
    inname: str = input.name if isinstance(input, tensor.Tensor) else "IN"
    if isinstance(input, torch.Tensor):
        input: tensor.Tensor = tensor.Tensor.from_torch(input)
    outname: str = "OUT"
    assert inname != outname

    B = cin.TensorVar(inname, shape=input.shape, fmt=input.format)
    if format is None:
        format = parse_format("d" * B.levels - 1)
    elif not isinstance(format, TensorFormat):
        format = parse_format(format)

    # The result has the same shape as input save for the collapsed dimensions.
    A_shape: int = []
    for i, size in enumerate(B.shape):
        if i == dim + 1:
            A_shape[-1] *= size
        else:
            A_shape.append(size)

    A = cin.TensorVar(outname, shape=tuple(A_shape), fmt=format)
    raise NotImplementedError("TODO(cgyurgyik)")
    # A = cin.TensorVar("A", fmt=["d"], shape=[6])
    # B = cin.TensorVar("B", fmt=["d", "s"], shape=[2, 3])
    # Bi = cin.IndexSeq(i, B, size=2, index=0, format=LevelType.DENSE)
    # Bj = cin.IndexSeq(j, B, size=3, index=1, format=LevelType.COMPRESSED)
    # c: cin.CIN = cin.ForAll(k, A._assignment, cin.ProductSeq(Bi, Bj))
