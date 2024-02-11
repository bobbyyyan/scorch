from scorch import format
from multipledispatch import dispatch
from scorch.compiler import cin
from scorch.compiler.shapes import cpp, lattice as il
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence

# Utility functions used in the CFIR -> CIN lowering phase.


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
                cpp.Constant(0) if i == 0 else ArrayIndexVariable(il.GetParent(seq)),
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
                    ArrayIndexVariable(il.GetParent(seq)),
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


@dispatch(cin.TensorVar, cin.IndexVar, int, format.LevelType)
def ArrayAccessCrd(
    tensor: cin.TensorVar, idx: cin.IndexVar, index: int, fmt: format.LevelType
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
        case cpp.Add(a, b):
            a, b = Simplify(a), Simplify(b)
            if a == cpp.Constant(0):
                return b
            if b == cpp.Constant(0):
                return a
            return cpp.Add(a, b)
        case cpp.Sub(a, b):
            a, b = Simplify(a), Simplify(b)
            if b == 0:
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
        case _:
            return expr
