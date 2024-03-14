from enum import StrEnum
import torch

from scorch.compiler.shapes.lower.opcode import Opcode
from scorch.compiler import cin
from scorch import tensor
from scorch.utils import parse_format
from scorch.compiler.shapes.lower import compile
from scorch.format import LevelType, TensorFormat
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence

# Necessary for ops that aren't supported in Burrito variant of the compiler.
import scorch.ops as experimental

# Concrete operations on Scorch tensors. If the input tensor is PyTorch,
# then it is converted to the Scorch equivalent.

# Used to hygienically name code generated index variables.
TENSOR_INDEX_NAME = "ijklmnopqrstuvwxyz"


# Used to hygienically name results. I know, this is bad.
SUFFIX = 0


def ResultName():
    """Hygienic name (or an attempt) for results."""

    def Suffix() -> int:
        global SUFFIX
        s: int = SUFFIX
        SUFFIX += 1
        return s

    return f"_R{Suffix()}"


def __format(
    format: Optional[Union[TensorFormat, str, List[str]]], rank: int
) -> TensorFormat:
    """Translates the provided format variant into a TensorFormat."""
    if format is None:
        return parse_format("d" * rank)
    if isinstance(format, TensorFormat):
        return format
    return parse_format(format)


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
    format: The format of the result tensor. This is defaulted to dense.

    This is similar to pytorch's `__getitem__` Op. For example,
        input = torch.Tensor([0,1,2,3,4,5,6,7,8,9,10])
        slice(input, dim=0, start=2, end=8, stride=2) # (Scorch)  [2, 4, 6]
        input[2:8:2]                                  # (PyTorch) [2, 4, 6]
    """
    assert dim < input.dim()
    outname: str = ResultName()
    if isinstance(input, torch.Tensor):
        return tensor.Tensor.from_torch(input[start:end:stride], name=outname)

    B = cin.TensorVar(input.name, shape=input.shape, fmt=input.format)
    # The result has the same shape as input save for the sliced dimension.
    A_shape = list(B.shape)
    A_shape[dim] = (end - start) // stride
    A = cin.TensorVar(
        outname, shape=tuple(A_shape), fmt=__format(format, rank=len(A_shape))
    )

    def ConstructSlice(A: cin.TensorVar, B: cin.TensorVar) -> cin.CIN:
        cins: List[Tuple[cin.IndexVar, cin.Seq]] = []
        input_indices: List[cin.IndexVar] = []
        for i, (format, size) in enumerate(zip(B.format.get_level_types(), B.shape)):
            s: str = TENSOR_INDEX_NAME[i]
            output_index = cin.IndexVar(s)
            input_index = cin.IndexVar(f"{B.name}{i}") if i == dim else output_index

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
    format: The format of the result tensor. This is defaulted to dense.

    This is similar to pytorch's `flatten` Op. For example,
      input = torch.Tensor([[1,2], [3,4]])
      flatten(input, dim=0)                        # (Scorch)  [1,2,3,4]
      torch.flatten(input, start_dim=0, end_dim=1) # (PyTorch) [1,2,3,4]
    """
    assert dim + 1 < input.dim()
    outname: str = ResultName()
    if isinstance(input, torch.Tensor):
        return tensor.Tensor.from_torch(
            input.flatten(start_dim=dim, end_dim=dim + 1), name=outname
        )

    B = cin.TensorVar(input.name, shape=input.shape, fmt=input.format)

    # The result has the same shape as input save for the collapsed dimensions.
    A_shape: int = []
    for i, size in enumerate(B.shape):
        if i == dim + 1:
            A_shape[-1] *= size
        else:
            A_shape.append(size)
    A = cin.TensorVar(
        outname, shape=tuple(A_shape), fmt=__format(format, rank=len(A_shape))
    )

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


def unflatten(
    input: torch.Tensor | tensor.Tensor,
    dim: int,
    sizes: Tuple[int],
    format: Optional[Union[TensorFormat, str, List[str]]] = None,
) -> tensor.Tensor:
    """Splits the tensor into chunks.

    input:  The tensor input to be split.
    dim:    The dimension in `input` to split.
    sizes:  The new dimensions for the split dimension.
    format: The format of the result tensor. This is defaulted to dense.

    This is similar to pytorch's `unflatten` Op. For example,
      input = torch.Tensor([1,2,3,4])
      unflatten(input, dim=0, sizes=(2,2))        # (Scorch)  [[1,2],[3,4]]
      torch.unflatten(input, dim=0, sizes=(2, 2)) # (PyTorch) [[1,2],[3,4]]
    """
    assert dim < input.dim()
    (I, J) = sizes
    outname: str = ResultName()
    if isinstance(input, torch.Tensor):
        return tensor.Tensor.from_torch(input.unflatten(dim=dim, sizes=(I, J)), outname)

    B = cin.TensorVar(input.name, shape=input.shape, fmt=input.format)
    # The result has the same shape as input save for the split dimensions.
    A_shape: int = []
    for i, size in enumerate(B.shape):
        A_shape.extend([I, J] if i == dim else [size])
    A = cin.TensorVar(outname, shape=tuple(A_shape), fmt=__format(format, B.levels + 1))

    def ConstructUnflatten(A: cin.TensorVar, B: cin.TensorVar) -> cin.CIN:
        cins: List[Tuple[cin.IndexVar, cin.Seq]] = []
        input_indices: List[cin.IndexVar] = []

        level_types: list[LevelType] = B.format.get_level_types()
        shape: list[int] = list(B.shape)
        for i, (format, size) in enumerate(zip(level_types, shape)):
            output_index = cin.IndexVar(TENSOR_INDEX_NAME[i])
            input_index = output_index
            input_indices.append(input_index)
            if i == dim:
                assert I * J == shape[i]
                a: cin.IndexSeq = cin.IndexSeq(
                    input_index, B, shape[i], i, level_types[i]
                )
                proj0 = cin.ProjectSeq(a, k=0, I=I, J=J)
                proj1 = cin.ProjectSeq(a, k=1, I=I, J=J)
                cins.append((cin.IndexVar(f"p{TENSOR_INDEX_NAME[i]}0"), proj0))
                cins.append((cin.IndexVar(f"p{TENSOR_INDEX_NAME[i]}1"), proj1))
                continue
            cins.append((output_index, cin.IndexSeq(input_index, B, size, i, format)))

        output_indices: list[cin.IndexVar] = [idx for (idx, _) in cins]
        A[*output_indices] = B[*input_indices]

        (idx, seq) = cins.pop()
        stmt: cin.CIN = cin.ForAll(idx, A._assignment, seq)
        for idx, seq in reversed(cins):
            stmt = cin.ForAll(idx, stmt, seq)
        return stmt

    return compile.CompileAndExecuteFunction(
        stmt=compile.Compile(ConstructUnflatten(A, B)),
        arguments=[input],
        result=tensor.Tensor.from_torch(torch.zeros(A_shape), outname),
    )


def __elementwise(
    lhs: torch.Tensor | tensor.Tensor,
    rhs: torch.Tensor | tensor.Tensor,
    op: Opcode,
    format: Optional[Union[TensorFormat, str, List[str]]] = None,
) -> tensor.Tensor:
    """Computes an elementwise operation `op` across the dimensions of `lhs` and `rhs`."""
    assert lhs.dim() == rhs.dim()
    assert type(lhs) == type(rhs)  # ...unnecessarily restrictive.
    outname: str = ResultName()
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        match op:
            case Opcode.ADD:
                return tensor.Tensor.from_torch(lhs + rhs, outname)
            case Opcode.MUL:
                return tensor.Tensor.from_torch(lhs * rhs, outname)
            case _:
                raise NotImplementedError(op)

    if lhs == rhs:
        lhs = lhs.copy()  # Avoid duplicate naming during compilation.
        lhs.name = f"L_{lhs.name}"
        rhs.name = f"R_{rhs.name}"
    A = cin.TensorVar(lhs.name, shape=lhs.shape, fmt=lhs.format)
    B = cin.TensorVar(rhs.name, shape=rhs.shape, fmt=rhs.format)

    def ConstructElementwise(
        R: cin.TensorVar, A: cin.TensorVar, B: cin.TensorVar
    ) -> cin.CIN:
        cins: List[Tuple[cin.IndexVar, cin.Seq]] = []
        input_indices: List[cin.IndexVar] = []

        B_levels: list[LevelType] = B.format.get_level_types()
        B_shape: list[int] = list(B.shape)
        A_levels: list[LevelType] = A.format.get_level_types()
        A_shape: list[int] = list(A.shape)
        for i, (fA, sA, fB, sB) in enumerate(zip(A_levels, A_shape, B_levels, B_shape)):
            output_index = cin.IndexVar(TENSOR_INDEX_NAME[i])
            input_index = output_index
            input_indices.append(input_index)
            seqA = cin.IndexSeq(input_index, A, sA, i, fA)
            seqB = cin.IndexSeq(input_index, B, sB, i, fB)
            match op:
                case Opcode.ADD:
                    cins.append((output_index, cin.UnionSeq(seqA, seqB)))
                case Opcode.MUL:
                    cins.append((output_index, cin.IntersectionSeq(seqA, seqB)))

        output_indices: list[cin.IndexVar] = [idx for (idx, _) in cins]
        match op:
            case Opcode.ADD:
                R[*output_indices] = A[*input_indices] + B[*input_indices]
            case Opcode.MUL:
                R[*output_indices] = A[*input_indices] * B[*input_indices]
            case _:
                raise NotImplementedError(op)

        (idx, seq) = cins.pop()
        stmt: cin.CIN = cin.ForAll(idx, R._assignment, seq)
        for idx, seq in reversed(cins):
            stmt = cin.ForAll(idx, stmt, seq)
        return stmt

    assert A.shape == B.shape
    R = cin.TensorVar(outname, shape=A.shape, fmt=__format(format, rank=A.levels))
    return compile.CompileAndExecuteFunction(
        stmt=compile.Compile(ConstructElementwise(R, A, B)),
        arguments=[lhs, rhs],
        result=tensor.Tensor.from_torch(torch.zeros(R.shape), outname),
    )


def add(
    lhs: torch.Tensor | tensor.Tensor,
    rhs: torch.Tensor | tensor.Tensor,
    format: Optional[Union[TensorFormat, str, List[str]]] = None,
):
    return __elementwise(lhs, rhs, Opcode.ADD, format)


def mul(
    lhs: torch.Tensor | tensor.Tensor,
    rhs: torch.Tensor | tensor.Tensor,
    format: Optional[Union[TensorFormat, str, List[str]]] = None,
):
    return __elementwise(lhs, rhs, Opcode.MUL, format)


def matmul(
    lhs: torch.Tensor | tensor.Tensor,
    rhs: torch.Tensor | tensor.Tensor,
    format: Optional[Union[TensorFormat, str, List[str]]] = None,
):
    # TODO: Implement.
    output = experimental.matmul(lhs, rhs)
    output.name = ResultName()
    return output


def concat(
    lhs: torch.Tensor | tensor.Tensor,
    rhs: torch.Tensor | tensor.Tensor,
    dim: int,
    format: Optional[Union[TensorFormat, str, List[str]]] = None,
):
    # TODO: Implement.
    assert type(lhs) == type(rhs)
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        return torch.concat([lhs, rhs], dim)

    return tensor.Tensor.from_torch(
        torch.concat([lhs.to_torch(), rhs.to_torch()], dim)
    ).to_sparse(format)


def generic_vector(
    instructions: list[Opcode | tensor.Tensor],
    shape: Tuple[int] = None,
    format: Optional[Union[TensorFormat, str, List[str]]] = None,
):
    """Computes the vector instructions provided in polish notation, e.g.,
    generic_vector([Opcode.MUL, Opcode.ADD, B, C, D] == (B + C) * D
    """

    def hygienic(old: Sequence[tensor.Tensor | Opcode]) -> list[tensor.Tensor | Opcode]:
        """
        Each tensor must have a unique name during compilation. Therefore, if
        the same tensor is used twice in the same equation, we must rename it.
        """
        new = []
        names = set()
        for instruction in old:
            match instruction:
                case tensor.Tensor():
                    if instruction.name in names:
                        # Make a copy, since we need two *different* tensors.
                        instruction = instruction.copy()
                        i = 0
                        while instruction.name in names:
                            instruction.name = f"{instruction.name}_{i}"
                            i += 1
                    names.add(instruction.name)
                    new.append(instruction)
                case Opcode() | _:
                    new.append(instruction)
        return new

    instructions: list[Opcode | tensor.Tensor] = hygienic(instructions)

    def __GenSeq1D(
        instructions: list[Opcode | cin.TensorVar], i: cin.IndexVar
    ) -> cin.Seq:
        """Generate CIN sequences for vector operations."""

        def __GenSeq(
            instructions: list[Opcode | cin.TensorVar], v: Optional[cin.IndexVar] = None
        ) -> cin.Seq:
            assert len(instructions) > 0
            next: Opcode | cin.TensorVar = instructions.pop()
            if isinstance(next, cin.TensorVar):
                (level,) = next.format.get_level_types()
                (size,) = next.shape
                return cin.IndexSeq(
                    idx=v if v is not None else i,
                    tensor=next,
                    size=size,
                    index=0,
                    format=level,
                )

            match next:
                case Opcode.ADD:
                    lhs: cin.Seq = __GenSeq(instructions)
                    rhs: cin.Seq = __GenSeq(instructions)
                    return cin.UnionSeq(lhs, rhs)
                case Opcode.MUL:
                    lhs: cin.Seq = __GenSeq(instructions)
                    rhs: cin.Seq = __GenSeq(instructions)
                    return cin.IntersectionSeq(lhs, rhs)
                case (Opcode.SLICE, start, end, stride):
                    input: cin.Seq = __GenSeq(instructions)
                    return cin.SliceSeq(input, start=start, end=end, stride=stride)
                case _:
                    raise NotImplementedError(next)

        return __GenSeq(instructions[::-1])

    def __GenAssign1D(instructions: list[Opcode | cin.TensorVar], i: cin.IndexVar):
        """Generate CIN RHS assignment."""

        def __GenAssign(instructions: list[Opcode | cin.TensorVar]):
            assert len(instructions) > 0
            next: Opcode | cin.TensorVar = instructions.pop()
            if isinstance(next, cin.TensorVar):
                return next[i]

            match next:
                case Opcode.ADD:
                    lhs: cin.Seq = __GenAssign(instructions)
                    rhs: cin.Seq = __GenAssign(instructions)
                    return lhs + rhs
                case Opcode.MUL:
                    lhs: cin.Seq = __GenAssign(instructions)
                    rhs: cin.Seq = __GenAssign(instructions)
                    return lhs * rhs
                case (Opcode.SLICE, _, _, _):
                    return __GenAssign(instructions)
                case _:
                    raise NotImplementedError(next)

        return __GenAssign(instructions[::-1])

    tensors: list[tensor.Tensor] = []
    cins: list[Opcode | cin.TensorVar] = []
    for instruction in instructions:
        if isinstance(instruction, tensor.Tensor):
            cins.append(
                cin.TensorVar(
                    instruction.name,
                    shape=instruction.shape,
                    fmt=instruction.format,
                )
            )
            tensors.append(instruction)
            continue

        cins.append(instruction)

    if shape is None:
        assert len(set(t.shape for t in tensors)) == 1
        shape = tensors[0].shape
    i = cin.IndexVar("i")
    outname: str = ResultName()
    R = cin.TensorVar(outname, shape=shape, fmt=__format(format, rank=1))
    R[i] = __GenAssign1D(cins, i)
    stmt: cin.CIN = cin.ForAll(i, R._assignment, __GenSeq1D(cins, i))
    return compile.CompileAndExecuteFunction(
        stmt=compile.Compile(stmt),
        arguments=tensors,
        result=tensor.Tensor.from_torch(torch.zeros(R.shape), outname),
    )
