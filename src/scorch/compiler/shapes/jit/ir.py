from __future__ import annotations
from dataclasses import dataclass, field
from scorch.compiler.shapes.opcode import Opcode
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import scorch.tensor as tensor
from scorch.format import TensorFormat
from scorch.compiler.shapes import ops
import torch
import functools
import operator


@dataclass
class IR:
    """A scorch intermediate representation over abstract values."""

    region: ScorchRegion
    opcode: Opcode
    ordinal: Optional[int] = None
    _users: set[IR] = field(default_factory=set)

    def __post_init__(self):
        self.region.append(self)
        for op in self.operands():
            op.append_user(self)

    def __mul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)

    def __hash__(self):
        return hash((self.ordinal, self.opcode, self.region.name))

    def operands(self) -> List[IR]:
        pass

    def shape(self) -> Tuple[int]:
        pass

    def format(self) -> Tuple[TensorFormat] | TensorFormat:
        pass

    def users(self) -> List[IR]:
        return list(self._users)

    def append_user(self, user: IR) -> None:
        self._users.add(user)

    def remove_user(self, user: IR) -> IR:
        self._users.remove(user)

    def module(self) -> ScorchModule:
        return self.region.module


@dataclass(kw_only=True)
class Value(IR):
    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class AbstractTensor(Value):
    """Tensor used when tracing the JIT compilation."""

    # TODO(cgyurgyik): We just want to carry the necessary
    # description, rather than the entire data structure itself.
    _tensor: tensor.Tensor

    __match_args__ = ("_tensor",)

    def __init__(self, tensor: tensor.Tensor, region: ScorchRegion):
        super().__init__(region=region, opcode=Opcode.TENSOR)
        self._tensor = tensor

    def tensor(self) -> tensor.Tensor:
        return self._tensor

    def operands(self) -> List[IR]:
        return []

    def id(self) -> str:
        return self._tensor.name

    def rank(self) -> int:
        return self._tensor.dim()

    def shape(self) -> Tuple[int]:
        return self._tensor.shape

    def format(self) -> TensorFormat:
        return self._tensor.format

    def nnz(self) -> int:
        # TODO(cgyurgyik): Symbolic representation...?
        if self.format().is_dense():
            # Conservatively assume all values are non-zero.
            return functools.reduce(operator.mul, self.shape(), 1)
        return self._tensor._nnz()

    def __str__(self):
        df: str = ",".join(
            f"{d}:{f}"
            for (d, f) in zip(self.shape(), self.format().get_level_formats())
        )
        return f"{self.id()}[{df}]"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class Instruction(IR):
    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class FusedOp(IR):
    __match_args__ = ("instructions",)
    instructions: List[IR | Opcode]

    def __init__(
        self, instructions: Sequence[IR | Opcode], region: Optional[ScorchRegion] = None
    ):
        self.instructions = instructions
        super().__init__(region=region, opcode=Opcode.FUSE)

    def format(self) -> TensorFormat:
        return self.instructions.format()

    def __str__(self):
        instructions = self.instructions.copy()[::-1]

        def pp(input):
            s = ""
            match next := input.pop():
                case IR():
                    s += f"%{next.ordinal}"
                case Opcode():
                    x, y = pp(input), pp(input)
                    s += f"({x} {next} {y})"
            return s

        return pp(instructions)

    def operands(self) -> List[IR]:
        return list(filter(lambda x: isinstance(x, IR), self.instructions))

    def shape(self) -> Tuple[int]:
        operands = self.operands()
        s = set([op.shape() for op in operands])
        assert len(s) == 1
        return s.pop()

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class copy(Instruction):
    input: IR
    __match_args__ = ("input",)

    def __init__(self, input: IR, region: Optional[ScorchRegion] = None):
        if region is None:
            region = input.region
        self.input = input
        super().__init__(region=region, opcode=Opcode.COPY)

    def operands(self) -> List[IR]:
        return [self.input]

    def shape(self) -> Tuple[int]:
        return self.input.shape()

    def format(self) -> TensorFormat:
        return self.input.format()

    def __str__(self):
        return f"{self.opcode} %{self.input.ordinal}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class slice(Instruction):
    input: IR
    dim: int
    start: int
    end: int
    __match_args__ = (
        "input",
        "dim",
        "start",
        "end",
        "step",
    )

    def __init__(
        self,
        input: IR,
        dim: int,
        start: int,
        end: int,
        step: int = 1,
        region: Optional[ScorchRegion] = None,
    ):
        if region is None:
            region = input.region
        self.input = input
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step
        super().__init__(region=region, opcode=Opcode.SLICE)

    def operands(self) -> List[IR]:
        return [self.input]

    def shape(self) -> Tuple[int]:
        s = list(self.input.shape())
        s[self.dim] = (self.end - self.start) // self.step
        return tuple(s)

    def format(self) -> TensorFormat:
        return self.input.format()

    def __str__(self):
        return f"{self.opcode}.{self.dim} %{self.input.ordinal}[{self.start}:{self.end}:{self.step}]"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class concat(Instruction):
    A: IR
    B: IR
    dim: int

    __match_args__ = (
        "A",
        "B",
        "dim",
    )

    def __init__(self, A: IR, B: IR, dim: int, region: Optional[ScorchRegion] = None):
        (x, y) = A.region, B.region
        if region is None:
            region = x
        self.A = A
        self.B = B
        self.dim = dim
        super().__init__(region=region, opcode=Opcode.CONCAT)

    def operands(self) -> List[IR]:
        return [self.A, self.B]

    def __str__(self):
        return f"{self.opcode}.{self.dim} %{self.A.ordinal}, %{self.B.ordinal}"

    def shape(self) -> Tuple[int]:
        (x, y) = list(self.A.shape()), list(self.B.shape())
        assert all(i == self.dim or a == b for i, (a, b) in enumerate(zip(x, y)))
        x[self.dim] += y[self.dim]
        return tuple(x)

    def format(self) -> Tuple[TensorFormat]:
        return (self.A.format(), self.B.format())

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class matmul(Instruction):
    A: IR
    B: IR

    __match_args__ = (
        "A",
        "B",
    )

    def __init__(self, A: IR, B: IR, region: Optional[ScorchRegion] = None):
        (x, y) = (A.region, B.region)
        if region is None:
            region = x
        self.A = A
        self.B = B
        super().__init__(region=region, opcode=Opcode.MATMUL)

    def operands(self) -> List[IR]:
        return [self.A, self.B]

    def shape(self) -> Tuple[int]:
        (x, y) = self.A.shape(), self.B.shape()
        # Treat matrix-vector as a special case of matrix-multiply.
        if len(x) == 1:
            x = (x[0], 1)
        if len(y) == 1:
            y = (y[0], 1)
        (x0, x1) = x
        (y0, y1) = y
        assert x1 == y0, f"mismatch in inner dimension: [{x0}, {x1}], [{y0}, {y1}]"
        return (x0, y1)

    def format(self) -> Tuple[TensorFormat]:
        return (self.A.format(), self.B.format())

    def __str__(self):
        return f"{self.opcode} %{self.A.ordinal}, %{self.B.ordinal}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class BinaryOp(Instruction):
    lhs: IR
    rhs: IR

    __match_args__ = (
        "lhs",
        "rhs",
    )

    def __init__(
        self, lhs: IR, rhs: IR, opcode: Opcode, region: Optional[ScorchRegion] = None
    ):
        (x, y) = lhs.region, rhs.region
        if region is None:
            region = x
        self.lhs = lhs
        self.rhs = rhs
        super().__init__(region=region, opcode=opcode)

    def operands(self) -> List[IR]:
        return [self.lhs, self.rhs]

    def shape(self) -> Tuple[int]:
        (x, y) = self.lhs.shape(), self.rhs.shape()
        assert x == y
        return x

    def format(self) -> Tuple[TensorFormat]:
        return (self.lhs.format(), self.rhs.format())

    def __str__(self):
        return f"{self.opcode} %{self.lhs.ordinal}, %{self.rhs.ordinal}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class add(BinaryOp):
    __match_args__ = (
        "lhs",
        "rhs",
    )

    def __init__(self, lhs: IR, rhs: IR, region: Optional[ScorchRegion] = None):
        super().__init__(lhs=lhs, rhs=rhs, opcode=Opcode.ADD, region=region)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class mul(BinaryOp):
    __match_args__ = (
        "lhs",
        "rhs",
    )

    def __init__(self, lhs: IR, rhs: IR, region: Optional[ScorchRegion] = None):
        super().__init__(lhs=lhs, rhs=rhs, opcode=Opcode.MUL, region=region)

    def __hash__(self):
        return super().__hash__()


@dataclass
class IRBuilder:
    """IR Builder for a Scorch IR region."""

    _region: ScorchRegion

    def graph(self):
        return self._region.graph

    def region(self):
        return self._region

    def tensor(self, id: str, input: torch.Tensor | list) -> IR:
        if isinstance(input, list):
            input = torch.Tensor(input)
        return AbstractTensor(tensor.Tensor.from_torch(input, id), self.region())

    def tensor(self, input: tensor.Tensor) -> IR:
        return AbstractTensor(input, self.region())

    def copy(self, input: IR) -> IR:
        return copy(input, self.region())

    def add(self, lhs: IR, rhs: IR) -> IR:
        return add(lhs, rhs, self.region())

    def mul(self, lhs: IR, rhs: IR) -> IR:
        return mul(lhs, rhs, self.region())

    def matmul(self, A: IR, B: IR) -> IR:
        return matmul(A, B, self.region())

    def concat(self, A: IR, B: IR, dim: int) -> IR:
        return concat(A, B, dim, self.region())

    def slice(self, input: IR, dim: int, start: int, end: int, step: int = 1) -> IR:
        return slice(input, dim, start, end, step, self.region())


@dataclass
class ScorchModule:
    name: str
    global_ordinal: int = 0

    def __init__(self, name: str):
        self.name = name
        self.global_ordinal = 0

    def ordinal(self):
        i: int = self.global_ordinal
        self.global_ordinal += 1
        return i


def can_fuse(A: IR, B: IR):
    if A.shape() != B.shape():
        # Conservatively ignores broadcasting.
        return False
    if len(A.users()) > 1 or len(B.users()) > 1:
        # Conversatively only fuse operations with 1 user.
        return False
    if B not in A.operands() or isinstance(B, AbstractTensor):
        # A = B + C
        # B = D + E
        return False
    # TODO(cgyurgyik): For prototype purposes, only supports 1-D.
    if len(A.shape()) != 1:
        return False
    return True


@dataclass()
class ScorchRegion:
    """Intermediate representation graph to enable JIT optimizations."""

    name: str
    # A sequential graph containing the SSA values.
    graph: list[IR]
    module: ScorchModule
    result: Optional[IR] = None

    def __init__(self, name: str, module: ScorchModule):
        self.name = name
        self.graph = []
        self.module = module

    def update_result(self, ir: IR) -> None:
        self.result = ir

    def ordinal(self) -> int:
        """A tracker to guarantee a unique identifier for each SSA value."""
        return self.module.ordinal()

    def append(self, value: IR) -> IR:
        value.ordinal = self.ordinal()
        self.graph.append(value)
        return value

    def evaluate(self, V: IR) -> tensor.Tensor:
        """
        Evaluates the result and returns the resulting Scorch Tensor.
        This is similar to the lazy `eval` approach of MLX.
        """
        match V:
            case AbstractTensor(input):
                return input
            case FusedOp(instructions):

                def convert(instructions):
                    new = []
                    for instruction in instructions:
                        if isinstance(instruction, Opcode):
                            new.append(instruction)
                            continue
                        assert isinstance(instruction, IR)
                        match instruction:
                            case AbstractTensor(input):
                                new.append(input)
                            case add(a, b) | mul(a, b):
                                new.extend(convert([a, b]))
                            case _:
                                raise NotImplementedError(type(instruction))
                    return new

                if len(V.shape()) == 1:
                    return ops.generic_vector(convert(instructions))
                raise NotImplementedError(
                    type(V)
                )  # TODO(cgyurgyik): Support different fusion.
            case add(lhs, rhs):
                return ops.add(self.evaluate(lhs), self.evaluate(rhs))
            case mul(lhs, rhs):
                return ops.mul(
                    self.evaluate(lhs),
                    self.evaluate(rhs),
                )
            case concat(A, B, dim):
                return ops.concat(self.evaluate(A), self.evaluate(B), dim)
            case matmul(A, B):
                return ops.matmul(self.evaluate(A), self.evaluate(B))
            case slice(input, dim, start, end, step):
                return ops.slice(self.evaluate(input), dim, start, end, step)
            case copy(input):
                return tensor.Tensor.copy(self.evaluate(input))
            case _:
                raise NotImplementedError(type(V))

    def fuse_operations(self) -> None:
        def fuse(A: IR, B: IR) -> None:
            # A = B + C
            # B = D * E
            #  =>
            # A = (D * E) + C
            assert B in A.operands()
            op: list[Opcode | IR] = [A.opcode]
            for operand in A.operands():
                if operand == B:
                    op.extend([B.opcode, *B.operands()])
                else:
                    match operand:
                        case AbstractTensor():
                            op.append(operand)
                        case copy(input):
                            op.append(input)
                        case mul(a, b) | add(a, b):
                            op.extend([operand.opcode, a, b])
                        case _:
                            raise NotImplementedError(type(operand))
            op = FusedOp(op, self)
            for old in (A, B):
                self.replace(old, op)

        def _fuse() -> bool:
            """Performs fusion and updates the underlying graph."""
            for instruction in self.graph:
                for operand in instruction.operands():
                    if can_fuse(instruction, operand):
                        fuse(instruction, operand)
                        return False
            return True

        while not _fuse():
            pass

    def replace(self, old: IR, new: IR | ScorchRegion) -> IR:
        """Replaces all cases of `old` with `new`. This does *not* reorder the IR."""
        assert self.result is not None
        for i, instruction in enumerate(graph := self.graph):
            if instruction == old:
                if isinstance(new, ScorchRegion):
                    (*head, new) = new.graph
                    graph[i:i] = list(head)
                else:
                    assert isinstance(new, IR)
                    graph[i] = new
                    if self.result == old:
                        self.update_result(new)
            if old in instruction.operands():
                match instruction:
                    case AbstractTensor(_):
                        pass
                    case copy(input) | slice(input):
                        if old == input:
                            instruction.input = new
                    case add(lhs, rhs) | mul(lhs, rhs):
                        if old == lhs:
                            instruction.lhs = new
                        if old == rhs:
                            instruction.rhs = new
                    case matmul(A, B) | concat(A, B):
                        if old == A:
                            instruction.A = new
                        if old == B:
                            instruction.B = new
                    case _:
                        raise NotImplementedError(type(instruction))
        self.graph = list(dict.fromkeys(graph))

    def simplify(self) -> None:
        def _simplify(instruction: IR) -> Optional[IR | ScorchRegion]:
            match instruction:
                case add(lhs, rhs):
                    if isinstance(lhs, AbstractTensor) and lhs.nnz() == 0:
                        return rhs  # 0 + A = A
                    if isinstance(rhs, AbstractTensor) and rhs.nnz() == 0:
                        return lhs  # A + 0 = A
                case mul(lhs, rhs):
                    if isinstance(lhs, AbstractTensor) and lhs.nnz() == 0:
                        return lhs  # 0 * A = 0
                    if isinstance(rhs, AbstractTensor) and rhs.nnz() == 0:
                        return rhs  # A * 0 = 0
                case concat(A, B, dim):
                    pass
                case matmul(A, B):
                    if isinstance(A, concat) and A.dim == 1 and len(B.shape()) == 1:
                        (a1f, a2f) = A.format()
                        (a1f0, a1f1) = a1f.get_level_types()
                        (a2f0, a2f1) = a2f.get_level_types()
                        if (
                            not a1f.is_dense()
                            and not a2f.is_dense()
                            and (a1f0, a1f1) != (a2f0, a2f1)
                        ):
                            # This results in conflicting iteration graphs. We
                            # can avoid this by iterating over them separately.
                            (b,) = B.shape()
                            child = ScorchRegion("temporary", instruction.module())
                            builder = IRBuilder(child)
                            builder.add(
                                builder.matmul(
                                    A.A, builder.slice(B, dim=0, start=0, end=b // 2)
                                ),
                                builder.matmul(
                                    A.B, builder.slice(B, dim=0, start=b // 2, end=b)
                                ),
                            )
                            return child
                case _:
                    pass
            return None

        while 1:
            converged: bool = True
            for old in (_ := self.graph):
                new: Optional[IR | ScorchRegion] = _simplify(old)
                if new is not None:
                    converged = False
                    self.replace(old, new)
            if converged:
                break

    # TODO(cgyurgyik): This can probably be simplified by adding users/usees as fields.
    def dce(self) -> None:
        def usees(ir: IR):
            return set(op.ordinal for op in ir.operands())

        found: bool = False
        seen = set((self.result.ordinal,))
        dce_graph = []
        for instruction in reversed(self.graph):
            if instruction == self.result:
                found = True
            if not found:  # Result not reached yet.
                continue
            if instruction.ordinal not in seen:  # No users found.
                continue
            seen |= usees(instruction)
            dce_graph.append(instruction)
        self.graph = dce_graph[::-1]

    def torch_evaluate(self) -> torch.Tensor:
        return self.evaluate(self.result).to_torch()

    def __str__(self) -> str:
        s: str = f"${self.name}:\n"
        indent: str = " " * 2

        def stringify(ir: IR):
            return f"{indent} %{ir.ordinal} = {str(ir)}"

        return s + "\n".join(stringify(ir) for ir in self.graph)

    def __repr__(self) -> str:
        return str(self)
