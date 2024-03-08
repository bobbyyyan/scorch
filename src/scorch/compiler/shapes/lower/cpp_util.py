from scorch import format
from multipledispatch import dispatch
from scorch.compiler import cin
from scorch.compiler.shapes.ast import cpp, ir
from scorch.compiler.shapes.lower import seq_util
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence

# Utility functions used in the CFIR -> CPP lowering phase.


@dispatch(int, cin.TensorVar, format.LevelType)
def ArrayIndexVariable(index: int, tensor: cin.TensorVar, fmt: format.LevelType):
    match fmt:
        case format.LevelType.DENSE:
            return cpp.Variable(f"{tensor.name}{index}")
        case format.LevelType.COMPRESSED:
            return cpp.Variable(f"p{tensor.name}{index}")
        case _:
            raise NotImplementedError(fmt)


@dispatch(cin.IndexSeq)
def ArrayIndexVariable(seq: cin.IndexSeq):
    return ArrayIndexVariable(seq.index, seq.tensor, seq.format)


def UniverseIndexVariable(idx: cin.IndexExpr):
    return cpp.Variable(f"U_{idx}")


def ProjectionVariable(k: int):
    return cpp.Variable(f"proj_{k}")


def ArrayLowerBound(seq: cin.IndexSeq):
    match fmt := seq.format:
        case format.LevelType.DENSE:
            return cpp.Constant(0)
        case format.LevelType.COMPRESSED:
            i: int = seq.index
            return cpp.Access(
                cpp.Variable(f"{seq.tensor.name}{i}_pos"),
                cpp.Constant(0)
                if i == 0
                else ArrayIndexVariable(seq_util.GetParent(seq)),
            )
        case _:
            raise NotImplementedError(fmt)


def ArrayUpperBound(seq: cin.IndexSeq):
    match fmt := seq.format:
        case format.LevelType.DENSE:
            return cpp.Constant(seq.size)
        case format.LevelType.COMPRESSED:
            i = seq.index
            return cpp.Access(
                cpp.Variable(f"{seq.tensor.name}{seq.index}_pos"),
                cpp.Constant(1)
                if i == 0
                else cpp.Add(
                    ArrayIndexVariable(seq_util.GetParent(seq)),
                    cpp.Constant(1),
                ),
            )
        case _:
            raise NotImplementedError(fmt)


@dispatch(cin.IndexSeq)
def ArrayAccessCrd(seq: cin.IndexSeq):
    match fmt := seq.format:
        case format.LevelType.DENSE:
            return ArrayIndexVariable(seq)
        case format.LevelType.COMPRESSED:
            return cpp.Access(
                cpp.Variable(f"{seq.tensor.name}{seq.index}_crd"),
                ArrayIndexVariable(seq),
            )
        case _:
            raise NotImplementedError(fmt)


@dispatch(cin.TensorVar, int, format.LevelType)
def ArrayAccessCrd(
    tensor: cin.TensorVar, index: int, fmt: format.LevelType
):
    match fmt:
        case format.LevelType.DENSE:
            return ArrayIndexVariable(index, tensor, fmt)
        case format.LevelType.COMPRESSED:
            return cpp.Access(
                cpp.Variable(f"{tensor.name}{index}_crd"),
                ArrayIndexVariable(index, tensor, fmt),
            )


def UpdateCompressedIterators(access: cin.TensorAccess) -> Optional[cpp.Cpp]:
    assert isinstance(access, cin.TensorAccess), type(access)
    types: List[format.LevelType] = access.level_types()
    if len(types) == 0 or types[-1] == format.LevelType.DENSE:
        return None
    indices: List[cin.IndexVar] = access.get_index_vars()
    return cpp.IncAssign(
        ArrayIndexVariable(len(indices) - 1, access.tensor, types[-1]), cpp.Constant(1)
    )


def Simplify(expr: cpp.Cpp) -> cpp.Cpp:
    """Simple algebraic reductions on Cpp constructs."""
    match expr:
        case cpp.While(cond, body):
            cond = Simplify(cond)
            if cond == cpp.Constant(0):
                return cpp.Nop()
            return cpp.While(cond, Simplify(body))
        case cpp.Block(stmts):
            return cpp.Block([Simplify(stmt) for stmt in stmts])
        case cpp.IfBlock(pairs):
            return cpp.IfBlock([(Simplify(a), Simplify(b)) for (a, b) in pairs])
        case cpp.Assign(lhs, rhs, op):
            return cpp.Assign(Simplify(lhs), Simplify(rhs), op)
        case cpp.IncAssign(lhs, rhs):
            return cpp.IncAssign(Simplify(lhs), Simplify(rhs))
        case cpp.Define(type, lhs, rhs):
            return cpp.Define(type, Simplify(lhs), Simplify(rhs))
        case cpp.Access(array, idx):
            if isinstance(array, cpp.Cpp):
                array = Simplify(array)
            if isinstance(idx, cpp.Cpp):
                idx = Simplify(idx)
            return cpp.Access(array, idx)
        case cpp.Eq(a, b):
            a, b = Simplify(a), Simplify(b)
            if a == b and not isinstance(a, cpp.FunctionCall):
                return cpp.Constant(1)  # a == a, assuming a is pure.
            return cpp.Eq(a, b)
        case cpp.Expression(e):
            return cpp.Expression(Simplify(e))
        case cpp.Not(x):
            x = Simplify(x)
            if x == cpp.Constant(0):
                return cpp.Constant(1)
            if x == cpp.Constant(1):
                return cpp.Constant(0)
            match x:
                case cpp.Or(a, b):  # !(a || b) -> !a && !b
                    return Simplify(cpp.And(cpp.Not(a), cpp.Not(b)))
                case cpp.And(a, b):  # !(a && b) -> !a || !b
                    return Simplify(cpp.Or(cpp.Not(a), cpp.Not(b)))
                case cpp.Lt(a, b):  # !(a < b) == (a >= b)
                    return Simplify(cpp.Ge(a, b))
                case cpp.Le(a, b):  # !(a <= b) == (a > b)
                    return Simplify(cpp.Gt(a, b))
                case cpp.Ge(a, b):  # !(a >= b) == (a < b)
                    return Simplify(cpp.Lt(a, b))
                case cpp.Gt(a, b):  # !(a > b) == (a <= b)
                    return Simplify(cpp.Le(a, b))
                case cpp.Eq(a, b):
                    return Simplify(cpp.Ne(a, b))
            return cpp.Not(x)
        case cpp.And(a, b):
            a, b = Simplify(a), Simplify(b)
            if cpp.Constant(0) in (a, b):
                return cpp.Constant(0)
            if a == b and not isinstance(a, cpp.FunctionCall):
                return a  # a && a == a, assuming a is pure.
            if isinstance(a, cpp.Constant):
                return b  # n && b == b, when n != 0
            if isinstance(b, cpp.Constant):
                return a  # a && n == a, when n != 0
            return cpp.And(a, b)
        case cpp.Or(a, b):
            a, b = Simplify(a), Simplify(b)
            if a == cpp.Constant(0) and b == cpp.Constant(0):
                return cpp.Constant(0)
            if a == cpp.Constant(0):
                return b  # 0 || b == b
            if b == cpp.Constant(0):
                return a  # a || 0 == a
            return cpp.Or(a, b)
        case cpp.Add(a, b):
            a, b = Simplify(a), Simplify(b)
            if a == cpp.Constant(0):
                return b
            if b == cpp.Constant(0):
                return a
            return cpp.Add(a, b)
        case cpp.Sub(a, b):
            a, b = Simplify(a), Simplify(b)
            if b == cpp.Constant(0):
                return a
            return cpp.Sub(a, b)
        case cpp.Mul(a, b):
            a, b = Simplify(a), Simplify(b)
            if cpp.Constant(0) in (a, b):
                return cpp.Constant(0)
            return cpp.Mul(a, b)
        case cpp.Div(a, b):
            a, b = Simplify(a), Simplify(b)
            if b == cpp.Constant(0):
                raise ZeroDivisionError(expr)
            if b == cpp.Constant(1):
                return a
            if a == cpp.Constant(0):
                return a
            return cpp.Div(a, b)
        case cpp.Mod(a, b):
            a, b = Simplify(a), Simplify(b)
            if b == cpp.Constant(0):
                raise ZeroDivisionError(expr)
            if b == cpp.Constant(1):
                return cpp.Constant(0)
            return cpp.Mod(a, b)
        case cpp.Min(a, b):
            return cpp.Min(Simplify(a), Simplify(b))
        case cpp.Max(a, b):
            return cpp.Max(Simplify(a), Simplify(b))
        case _:
            return expr
