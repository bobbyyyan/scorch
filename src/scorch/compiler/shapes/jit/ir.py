from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import scorch.tensor as tensor
from scorch.format import TensorFormat
from scorch.compiler.shapes import ops
import torch
import functools
import operator
import collections


class Opcode(StrEnum):
    TENSOR = "tensor"
    ADD = "add"
    MUL = "mul"
    COPY = "copy"


@dataclass
class IR():
    """A scorch intermediate representation over abstract values."""
    region: ScorchRegion
    opcode: Opcode
    ordinal: Optional[int] = None

    def __post_init__(self):
        self.region.append(self)

    def __mul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)

    def __hash__(self):
        return hash((self.ordinal, self.opcode, self.region.name))

    def operands(self) -> List[IR]:
        pass


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
        df: str = ','.join(
            f"{d}:{f}" for (d, f) in zip(
                self.shape(), self.format().get_level_formats()
            )
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
class copy(Instruction):
    input: IR
    __match_args__ = ("input",)

    def __init__(self, input: IR):
        super().__init__(region=input.region, opcode=Opcode.COPY)
        self.input = input

    def operands(self) -> List[IR]:
        return [self.input]

    def __str__(self):
        return f"{self.opcode} {self.input}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class BinaryOp(Instruction):
    lhs: IR
    rhs: IR

    __match_args__ = ("lhs", "rhs",)

    def __init__(self, lhs: IR, rhs: IR, opcode: Opcode):
        (x, y) = lhs.region, rhs.region
        assert x == y
        super().__init__(region=x, opcode=opcode)
        self.lhs = lhs
        self.rhs = rhs

    def operands(self) -> List[IR]:
        return [self.lhs, self.rhs]

    def __str__(self):
        return f"{self.opcode} {self.lhs}, {self.rhs}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class add(BinaryOp):
    __match_args__ = ("lhs", "rhs",)

    def __init__(self, lhs: IR, rhs: IR):
        super().__init__(lhs=lhs, rhs=rhs, opcode=Opcode.ADD)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class mul(BinaryOp):
    __match_args__ = ("lhs", "rhs",)

    def __init__(self, lhs: IR, rhs: IR):
        super().__init__(lhs=lhs, rhs=rhs, opcode=Opcode.MUL)

    def __hash__(self):
        return super().__hash__()


@dataclass(kw_only=True)
class ScorchRegion():
    """Intermediate representation graph to enable JIT optimizations."""
    name: str
    # A sequential graph containing the SSA values.
    graph: collections.OrderedDict[IR] = field(default_factory=list)
    # Global identifier used to uniquely name each SSA value in the graph.
    global_ordinal: int = 0

    def __init__(self, name: str):
        self.name = name
        self.graph = []
        self.global_ordinal = 0

    def ordinal(self) -> int:
        """A tracker to guarantee a unique identifier for each SSA value."""
        i: int = self.global_ordinal
        self.global_ordinal += 1
        return i

    def append(self, value: IR) -> IR:
        value.ordinal = self.ordinal()
        self.graph.append(value)
        return value

    def evaluate(self, V: IR) -> tensor.Tensor:
        """
        Evaluates `V` and returns the resulting Scorch Tensor.
        This is similar to the lazy `eval` approach of MLX."""
        match V:
            case AbstractTensor(input):
                return input
            case add(lhs, rhs):
                return ops.add(
                    self.evaluate(lhs),
                    self.evaluate(rhs)
                )
            case mul(lhs, rhs):
                return ops.mul(
                    self.evaluate(lhs),
                    self.evaluate(rhs),
                )
            case copy(input):
                return tensor.Tensor.copy(self.evaluate(input))
            case _:
                raise NotImplementedError(type(V))

    def torch_evaluate(self, V: IR) -> torch.Tensor:
        """Equivalent to `evaluate`, but converts the result to a torch.Tensor."""
        return self.evaluate(V).to_torch()

    def __str__(self) -> str:
        def GetOrdinals(IR: IR) -> List[int]:
            def o(S: IR) -> int:
                return S.ordinal
            match IR:
                case AbstractTensor(_):
                    return []
                case copy(input):
                    return [o(input)]
                case add(lhs, rhs) | mul(lhs, rhs):
                    return [o(lhs), o(rhs)]
                case _:
                    raise NotImplementedError(type(IR))
        s: str = f"${self.name}:\n"
        indent: str = " " * 2
        for node in self.graph:
            s += f"{indent}"
            s += f"%{node.ordinal} = {node.opcode} "
            s += f"{', '.join(f'%{o}' for o in GetOrdinals(node))}"
            # Additional metadata can be appended here.
            if isinstance(node, Value):
                s += str(node)
            s += "\n"
        return s

    def __repr__(self) -> str:
        return str(self)


def simplify(region: ScorchRegion) -> None:
    def _simplify(instruction: IR) -> Optional[IR]:
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
                    return rhs  # A * 0 = A
            case _:
                pass
        return None

    def update(old: IR, new: IR) -> None:
        for (i, instruction) in enumerate(graph := region.graph):
            if instruction == old:
                graph[i] = new
            if old in (instruction.operands()):
                match(instruction):
                    case copy(input):
                        if old == input:
                            instruction.input = new
                    case add(lhs, rhs) | mul(lhs, rhs):
                        if old == lhs:
                            instruction.lhs = new
                        if old == rhs:
                            instruction.rhs = new
                    case _:
                        pass
        # Remove duplicates.
        region.graph = list(dict((k, []) for k in graph).keys())

    while 1:
        converged: bool = True
        for old in (graph := region.graph):
            new: Optional[IR] = _simplify(old)
            if new is not None:
                converged = False
                update(old, new)
        if converged:
            break
