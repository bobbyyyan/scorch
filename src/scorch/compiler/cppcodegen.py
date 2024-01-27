from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
from scorch.compiler import cfir, cin, cpp
import scorch.compiler.siterator as it
import scorch.format as format


def LowerLoop(idx: cin.IndexVar, sexpr: cin.Seq, body: cfir.CFIR, first: bool):
    loop = cpp.While(
        it.Valid(sexpr),
        cpp.Block(
            stmts=[
                cpp.Define(
                    type=cpp.IndexType(), lhs=cpp.Variable(idx.name), rhs=it.Eval(sexpr)
                ),
                Lower(body),
                it.Next(cpp.Variable(idx.name), sexpr)
                if isinstance(body, cfir.Switch)
                else it.UnconditionalNext(sexpr),
            ]
        ),
    )
    return [it.Init(sexpr), loop] if first else loop


def LowerIndexExpr(expr: cin.IndexExpr):
    match expr:
        case cin.TensorAccess():
            tensor = expr.tensor
            return cpp.Access(
                cpp.Variable(f"{tensor.name}.data"), LowerIndexExprRec(expr)
            )
        case _:
            raise NotImplementedError(type(expr))


def LowerIndexExprRec(expr: cin.IndexExpr, i: int = 0):
    match expr:
        case cin.TensorAccess():
            if i >= expr.num_levels:
                return cpp.Constant(0)
            if expr.num_levels == 0:
                return cpp.Variable(expr.tensor.name)
            idx = expr.get_index_vars()[i]
            fmt = expr.level_types()[i]
            match fmt:
                case format.LevelType.COMPRESSED:
                    return cpp.Variable(f"{idx.name}p_{expr.tensor.name}")
                case format.LevelType.DENSE:
                    return cpp.Add(
                        cpp.Mul(
                            LowerIndexExprRec(expr, i + 1), expr.get_tensor().shape[i]
                        ),
                        cpp.Variable(idx.name),
                    )
                case _:
                    raise NotImplementedError(fmt)
        case _:
            raise NotImplementedError(expr)


def LowerAssign(lhs: cin.TensorAccess, rhs: cin.IndexExpr):
    iterators = cpp.UpdateCompressedIterators(lhs)
    return cpp.Block(
        stmts=[
            # Write to compressed dimensions.
            *IndexWriteList(lhs),
            # Perform assignment/compute.
            cpp.Assign(LowerIndexExpr(lhs), LowerIndexExpr(rhs)),
            # Update compressed iterators.
            *([] if iterators is None else [iterators]),
        ]
    )


def IndexWriteList(ta: cin.TensorAccess):
    types: List[format.LevelType] = ta.level_types()
    indices: List[cin.IndexVar] = ta.get_index_vars()

    def is_compressed(pair: Tuple[format.LevelType, cin.IndexVar]):
        return pair[0] == format.LevelType.COMPRESSED

    compressed_levels = list(filter(is_compressed, zip(types, indices)))
    return [IndexWrite(ta, idx) for (_, idx) in compressed_levels]


def IndexWrite(ta: cin.TensorAccess, idx: cin.IndexVar):
    index: int = ta.level_of_index_var(idx)
    format: format.LevelType = ta.level_types()[index]
    return cpp.Assign(
        cpp.ArrayAccessCrd2(ta.tensor, idx, format), cpp.Variable(idx.name)
    )


def Lower(stmt: cfir.CFIR, first=False):
    match stmt:
        case cfir.Loop(idx, sexpr, body):
            return LowerLoop(idx, sexpr, body, first)
        case cfir.Assign(lhs, rhs):
            return LowerAssign(lhs, rhs)
        case list(l):
            return [Lower(c) for c in l]
        case _:
            raise NotImplementedError(type(stmt))
