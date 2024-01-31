from scorch import format
from multipledispatch import dispatch
from scorch.compiler import cin
from scorch.compiler.shapes import cpp
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence

# Utility functions used in the CFIR -> CIN lowering phase.


@dispatch(cin.IndexVar, cin.TensorVar, format.LevelType)
def ArrayIndexVariable(idx: cin.IndexVar, tensor: cin.TensorVar, fmt: format.LevelType):
    match fmt:
        case format.LevelType.DENSE:
            return cpp.Variable(f"{idx}_{tensor.name}")
        case format.LevelType.COMPRESSED:
            return cpp.Variable(f"{idx}p_{tensor.name}")
        case _:
            raise NotImplementedError(fmt)


@dispatch(cin.IndexSeq)
def ArrayIndexVariable(seq: cin.IndexSeq):
    return ArrayIndexVariable(seq.idx, seq.tensor, seq.format)


def ArrayLowerBound(seq: cin.IndexSeq):
    match fmt := seq.format:
        case format.LevelType.DENSE:
            return cpp.Constant(0)
        case format.LevelType.COMPRESSED:
            i: int = seq.index
            return cpp.Access(
                cpp.Access(cpp.Variable(f"{seq.tensor.name}.pos"), i),
                cpp.Constant(0)
                if i == 0
                else ArrayIndexVariable(seq.parent.idx, seq.tensor, seq.format),
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
                cpp.Access(cpp.Variable(f"{seq.tensor.name}.pos"), cpp.Constant(i)),
                cpp.Constant(1)
                if i == 0
                else cpp.Add(
                    ArrayIndexVariable(seq.parent.idx, seq.tensor, seq.format),
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
                cpp.Access(
                    cpp.Variable(f"{seq.tensor.name}.crd"), cpp.Constant(seq.index)
                ),
                ArrayIndexVariable(seq),
            )


@dispatch(cin.TensorVar, cin.IndexVar, int, format.LevelType)
def ArrayAccessCrd(
    tensor: cin.TensorVar, idx: cin.IndexVar, index: int, fmt: format.LevelType
):
    match fmt:
        case format.LevelType.DENSE:
            return ArrayIndexVariable(idx, tensor, fmt)
        case format.LevelType.COMPRESSED:
            return cpp.Access(
                cpp.Access(cpp.Variable(f"{tensor.name}.crd"), cpp.Constant(index)),
                ArrayIndexVariable(idx, tensor, fmt),
            )


def UpdateCompressedIterators(ta: cin.TensorAccess) -> Optional[cpp.Cpp]:
    assert isinstance(ta, cin.TensorAccess), type(ta)
    types: List[format.LevelType] = ta.level_types()
    if types[-1] == format.LevelType.DENSE:
        return None
    indices: List[cin.IndexVar] = ta.get_index_vars()
    return cpp.IncAssign(
        ArrayIndexVariable(indices[-1], ta.tensor, types[-1]), cpp.Constant(1)
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
