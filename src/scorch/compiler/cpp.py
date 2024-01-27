from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
from . import cin
from scorch import format
from enum import StrEnum


@dataclass
class CppType:
    pass


@dataclass
class Int32(CppType):
    def __str__(self):
        return "int32_t"

    def __repr__(self):
        return str(self)


@dataclass
class IndexType(CppType):
    def __str__(self):
        return "size_t"

    def __repr__(self):
        return str(self)


# ----------------------------------------


@dataclass
class Cpp:
    pass


@dataclass
class Constant(Cpp):
    value: int | float

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self)


@dataclass
class Variable(Cpp):
    name: str

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


@dataclass
class Access(Cpp):
    # TODO(cgyurgyik): This is a hack.
    array: cin.TensorVar | Cpp
    idx: cin.IndexExpr | Cpp

    def __str__(self):
        array = self.array.name if isinstance(self.array, cin.TensorVar) else self.array
        return f"{array}[{self.idx}]"

    def __repr__(self):
        return str(self)


@dataclass
class Assign(Cpp):
    lhs: Cpp
    rhs: Cpp

    def __str__(self):
        return f"{self.lhs} = {self.rhs};"

    def __repr__(self):
        return str(self)


@dataclass
class IncAssign(Cpp):
    lhs: Cpp
    rhs: Cpp

    def __str__(self):
        return f"{self.lhs} += {self.rhs};"

    def __repr__(self):
        return str(self)


@dataclass
class Block(Cpp):
    stmts: List[Cpp]

    def __str__(self):
        return "\n".join(str(s) for s in self.stmts)

    def __repr__(self):
        return str(self)


@dataclass
class While(Cpp):
    cond: Cpp
    body: Block

    def __str__(self):
        return f"while ({self.cond}) {{\n{self.body}\n}}"

    def __repr__(self):
        return str(self)


@dataclass
class Define(Cpp):
    type: CppType
    lhs: Cpp
    rhs: Cpp

    def __str__(self):
        return f"{self.type} {self.lhs} = {self.rhs};"

    def __repr__(self):
        return str(self)


class Op(StrEnum):
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "%"
    EQ = "=="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    NOT = "!"
    LOGICAL_AND = "&&"
    LOGICAL_OR = "||"


@dataclass
class UnaryOp(Cpp):
    input: Cpp
    op: Op

    def __init__(self, input: Cpp, op: Op):
        self.input = input
        self.op = op

    def __str__(self):
        return f"({self.op} {self.input})"

    def __repr__(self):
        return str(self)


@dataclass
class Not(UnaryOp):
    def __init__(self, input: Cpp):
        super().__init__(input=input, op=Op.NOT)


@dataclass
class BinaryOp(Cpp):
    lhs: Cpp
    rhs: Cpp
    op: Op

    def __init__(self, lhs: Cpp, rhs: Cpp, op: Op):
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __str__(self):
        return f"({self.lhs} {self.op} {self.rhs})"

    def __repr__(self):
        return str(self)


@dataclass
class Lt(BinaryOp):
    def __init__(self, lhs: Cpp, rhs: Cpp):
        super().__init__(lhs=lhs, rhs=rhs, op=Op.LT)


@dataclass
class Eq(BinaryOp):
    def __init__(self, lhs: Cpp, rhs: Cpp):
        super().__init__(lhs=lhs, rhs=rhs, op=Op.EQ)


@dataclass
class Add(BinaryOp):
    def __init__(self, lhs: Cpp, rhs: Cpp):
        super().__init__(lhs=lhs, rhs=rhs, op=Op.ADD)


@dataclass
class Sub(BinaryOp):
    def __init__(self, lhs: Cpp, rhs: Cpp):
        super().__init__(lhs=lhs, rhs=rhs, op=Op.SUBTRACT)


@dataclass
class Mul(BinaryOp):
    def __init__(self, lhs: Cpp, rhs: Cpp):
        super().__init__(lhs=lhs, rhs=rhs, op=Op.MULTIPLY)


@dataclass
class And(BinaryOp):
    def __init__(self, lhs: Cpp, rhs: Cpp):
        super().__init__(lhs=lhs, rhs=rhs, op=Op.LOGICAL_AND)


@dataclass
class Or(BinaryOp):
    def __init__(self, lhs: Cpp, rhs: Cpp):
        super().__init__(lhs=lhs, rhs=rhs, op=Op.LOGICAL_OR)


@dataclass
class Mod(BinaryOp):
    def __init__(self, lhs: Cpp, rhs: Cpp):
        super().__init__(lhs=lhs, rhs=rhs, op=Op.MODULO)


@dataclass
class Div(BinaryOp):
    def __init__(self, lhs: Cpp, rhs: Cpp):
        super().__init__(lhs=lhs, rhs=rhs, op=Op.DIVIDE)


# ----------------------------------------


# TODO(cgyurgyik): ... I don't want to include a dependency for multiple dispatch for now,
# but that will likely change soon.
def ArrayIndexVariable2(
    idx: cin.IndexVar, tensor: cin.TensorVar, fmt: format.LevelType
):
    match fmt:
        case format.LevelType.DENSE:
            return Variable(f"{idx}_{tensor.name}")
        case format.LevelType.COMPRESSED:
            return Variable(f"{idx}p_{tensor.name}")
        case _:
            raise NotImplementedError(fmt)


def ArrayIndexVariable(seq: cin.IndexSeq):
    return ArrayIndexVariable2(seq.idx, seq.tensor, seq.format)


def ArrayLowerBound(seq: cin.IndexSeq):
    match fmt := seq.format:
        case format.LevelType.DENSE:
            return Constant(0)
        case format.LevelType.COMPRESSED:
            i = seq.index
            return Access(
                Access(Variable(f"{seq.tensor.name}.pos"), i),
                Constant(0)
                if i == 0
                else ArrayIndexVariable2(seq.parent, seq.tensor, seq.format),
            )
        case _:
            raise NotImplementedError(fmt)


def ArrayUpperBound(seq: cin.IndexSeq):
    match fmt := seq.format:
        case format.LevelType.DENSE:
            return Constant(seq.size)
        case format.LevelType.COMPRESSED:
            i = seq.index
            return Access(
                Access(Variable(f"{seq.tensor.name}.pos"), Constant(i)),
                Constant(1)
                if i == 0
                else Add(
                    ArrayIndexVariable2(seq.parent, seq.tensor, seq.format), Constant(1)
                ),
            )
        case _:
            raise NotImplementedError(fmt)


def ArrayAccessCrd(seq: cin.IndexSeq):
    match fmt := seq.format:
        case format.LevelType.DENSE:
            return ArrayIndexVariable(seq)
        case format.LevelType.COMPRESSED:
            return Access(
                Access(Variable(f"{seq.tensor.name}.crd"), Constant(seq.index)),
                ArrayIndexVariable(seq),
            )


def ArrayAccessCrd2(tensor: cin.TensorVar, index: cin.IndexVar, fmt: format.LevelType):
    match fmt:
        case format.LevelType.DENSE:
            return ArrayIndexVariable2(index, tensor, fmt)
        case format.LevelType.COMPRESSED:
            return Access(
                Access(Variable(f"{tensor.name}.crd"), Constant(index)),
                ArrayIndexVariable2(index, tensor, fmt),
            )


def UpdateCompressedIterators(ta: cin.TensorAccess) -> Optional[Cpp]:
    assert isinstance(ta, cin.TensorAccess), type(ta)
    types = ta.level_types()
    if types[-1] == format.LevelType.DENSE:
        return None
    indices = ta.get_index_vars()
    return IncAssign(
        ArrayIndexVariable2(indices[-1], ta.tensor, types[-1]), Constant(1)
    )
