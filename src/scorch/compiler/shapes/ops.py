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
      input = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10])
      slice(input, dim=0, start=2, end=8, stride=2) # [2, 4, 6]
    """
    assert dim < input.dim()
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
    input: The input tensor to be flattened
    dim: The dimension in `input` to be flattened with `dim + 1`.
    format: The format of the result tensor.

    For example,
      input = torch.Tensor([[1,2,3], [4,5,6]])
      flatten(input, dim=0) # [1,2,3,4,5,6]
    """
    assert dim + 1 < input.dim()
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

    def ConstructFlatten(A: cin.TensorVar, B: cin.TensorVar) -> cin.CIN:
        cins: List[Tuple[cin.IndexVar, cin.Seq]] = []
        input_indices: List[cin.IndexVar] = []

        level_types: list[LevelType] = B.format.get_level_types()
        shape: list[int] = list(B.shape)
        for i, (format, size) in enumerate(zip(level_types, shape)):
            if i == dim:
                # Fuse these dimensions on the `dim + 1` case.
                continue
            output_index = cin.IndexVar(TENSOR_INDEX_NAME[i])
            input_index = output_index

            if i == dim + 1:
                ai: cin.IndexVar = cin.IndexVar(f"p{i - 1}")
                bi: cin.IndexVar = cin.IndexVar(f"p{i}")
                a: cin.IndexSeq = cin.IndexSeq(
                    ai, B, shape[i - 1], i - 1, level_types[i - 1]
                )
                b: cin.IndexSeq = cin.IndexSeq(bi, B, size, i, format)
                input_indices.extend([ai, bi])
                cins.append((output_index, cin.ProductSeq(a, b)))
                continue
            input_indices.append(input_index)
            cins.append((output_index, cin.IndexSeq(input_index, B, size, i, format)))

        output_indices: list[cin.IndexVar] = [idx for (idx, _) in cins]
        A[*output_indices] = B[*input_indices]

        (idx, seq) = cins.pop()
        stmt: cin.CIN = cin.ForAll(idx, A._assignment, seq)
        for idx, seq in reversed(cins):
            stmt = cin.ForAll(idx, stmt, seq)
        return stmt

    return compile.CompileAndExecuteFunction(
        stmt=compile.Compile(ConstructFlatten(A, B)),
        arguments=[input],
        result=tensor.Tensor.from_torch(torch.zeros(A_shape), outname),
    )
