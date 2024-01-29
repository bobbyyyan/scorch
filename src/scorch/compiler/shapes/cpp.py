from dataclasses import dataclass
from enum import StrEnum
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
from scorch.compiler import cin


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
    # TODO(cgyurgyik): This is a hack, we really shouldn't have dependencies across translation phases
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
class IfBlock(Cpp):
    pairs: List[Cpp | Tuple[Cpp, Cpp]]

    def __str__(self):
        return " else ".join(
            [
                f"if ({p[0]}) {{ {p[1]} }}" if isinstance(p, Tuple) else f"{p}"
                for p in self.pairs
            ]
        )


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
    MINIMUM = "min"
    MAXIMUM = "max"


@dataclass
class UnaryOp(Cpp):
    input: Cpp
    op: Op

    def __init__(self, input: Cpp, op: Op):
        self.input = input
        self.op = op

    def __str__(self):
        return f"({self.op}{self.input})"

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
        if self.op in (Op.MINIMUM, Op.MAXIMUM):
            return f"{self.op}({self.lhs}, {self.rhs})"
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


@dataclass
class Min(BinaryOp):
    def __init__(self, lhs: Cpp, rhs: Cpp):
        super().__init__(lhs=lhs, rhs=rhs, op=Op.MINIMUM)


@dataclass
class Max(BinaryOp):
    def __init__(self, lhs: Cpp, rhs: Cpp):
        super().__init__(lhs=lhs, rhs=rhs, op=Op.MAXIMUM)
