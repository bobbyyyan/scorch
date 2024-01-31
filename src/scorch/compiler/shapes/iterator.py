from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence

from scorch.compiler import cin as cin
from scorch.compiler.shapes import cpp, cpputil, lattice as il

# An iterator model that follows the work presented in "Compilation of
# Shape Operators on Sparse Arrays" by Root, et. al.


def Init(sexpr: cin.Seq):
    match (sexpr):
        case cin.IndexSeq():
            return cpp.Define(
                type=cpp.IndexType(),
                lhs=cpputil.ArrayIndexVariable(sexpr),
                rhs=cpputil.ArrayLowerBound(sexpr),
            )
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.Block(stmts=[Init(s1), Init(s2)])
        case cin.Concatenate(s1, s2):
            raise NotImplementedError(sexpr)
        case cin.Product(s1, s2):
            return cpp.Block(
                stmts=[
                    Init(s1),
                    Init(s2),
                    cpp.While(
                        cond=cpp.And(Valid(s1), cpp.Not(Valid(s2))),
                        body=cpp.Block(
                            stmts=[
                                UnconditionalNext(s1),
                                Reset(s2),
                            ]
                        ),
                    ),
                ]
            )
        case cin.SliceSeq(a, s, _, r):
            return cpp.Block(
                [
                    Init(a),
                    cpp.While(
                        cond=cpp.And(
                            Valid(a),
                            cpp.Not(
                                cpp.Eq(
                                    cpp.Mod(
                                        cpp.Sub(Eval(a), cpp.Constant(s)),
                                        cpp.Constant(r),
                                    ),
                                    cpp.Constant(0),
                                )
                                # if s is 0, a < 0 is always false.
                                if s == 0
                                else cpp.Or(
                                    cpp.Lt(Eval(a), cpp.Constant(s)),
                                    cpp.Not(
                                        cpp.Eq(
                                            cpp.Mod(
                                                cpp.Sub(Eval(a), cpp.Constant(s)),
                                                cpp.Constant(r),
                                            ),
                                            cpp.Constant(0),
                                        )
                                    ),
                                )
                            ),
                        ),
                        body=UnconditionalNext(a),
                    ),
                ]
            )
        case _:
            raise NotImplementedError(type(sexpr))


def Reset(sexpr: cin.Seq):
    match (sexpr):
        case cin.IndexSeq():
            return cpp.Assign(
                lhs=cpputil.ArrayIndexVariable(sexpr),
                rhs=cpputil.ArrayLowerBound(sexpr),
            )
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.Block(stmts=[Reset(s1), Reset(s2)])
        case cin.Product(s1, s2):
            return cpp.Block(
                stmts=[
                    Reset(s1),
                    Reset(s2),
                    cpp.While(
                        cond=cpp.And(Valid(s1), cpp.Not(Valid(s2))),
                        body=cpp.Block(
                            stmts=[
                                UnconditionalNext(s1),
                                Reset(s2),
                            ]
                        ),
                    ),
                ]
            )
        case cin.Concatenate(s1, s2):
            raise NotImplementedError(type(sexpr))
        case cin.SliceSeq(a, s, e, r):
            raise NotImplementedError(type(sexpr))
        case _:
            raise NotImplementedError(type(sexpr))


def Valid(sexpr: cin.Seq):
    match (sexpr):
        case cin.IndexSeq(_, _):
            return cpp.Lt(
                lhs=cpputil.ArrayIndexVariable(sexpr),
                rhs=cpputil.ArrayUpperBound(sexpr),
            )
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.And(Valid(s1), Valid(s2))
        case cin.Concatenate(s1, s2) | cin.Product(s1, s2):
            return cpp.And(Valid(s1), Valid(s2))
        case cin.SliceSeq(a, _, e, r):
            return cpp.And(Valid(a), cpp.Lt(Eval(a), cpp.Constant(e)))
        case _:
            raise NotImplementedError(type(sexpr))


def Eval(sexpr: cin.Seq):
    match (sexpr):
        case cin.IndexSeq():
            return cpputil.ArrayAccessCrd(sexpr)
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.Min(Eval(s1), Eval(s2))
        case cin.Concatenate(s1, s2):
            raise NotImplementedError(type(sexpr))
        case cin.Product(s1, s2):
            # a * |b| + b
            return cpp.Add(cpp.Mul(Eval(s1), il.Size(s2)), Eval(s2))
        case cin.SliceSeq(a, s, _, r):
            return cpp.Div(cpp.Sub(Eval(a), cpp.Constant(s)), cpp.Constant(r))
        case _:
            raise NotImplementedError(type(sexpr))


def Next(value: cpp.Cpp, sexpr: cin.Seq):
    match (sexpr):
        case cin.IndexSeq(_, _):
            return cpp.IncAssign(
                cpputil.ArrayIndexVariable(sexpr), cpp.Eq(value, Eval(sexpr))
            )
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.Block(stmts=[Next(value, s1), Next(value, s2)])
        case cin.Product(s1, s2):
            return cpp.IfBlock(
                pairs=[
                    (
                        cpp.Eq(value, Eval(sexpr)),
                        cpp.Block(
                            stmts=[
                                UnconditionalNext(s2),
                                cpp.While(
                                    cond=cpp.And(Valid(s1), cpp.Not(Valid(s2))),
                                    body=cpp.Block(
                                        stmts=[
                                            UnconditionalNext(s1),
                                            Reset(s2),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    )
                ]
            )
        case cin.Concatenate(s1, s2):
            raise NotImplementedError(type(sexpr))
        case cin.SliceSeq(a, s, e, r):
            raise NotImplementedError(type(sexpr))
        case _:
            raise NotImplementedError(type(sexpr))


def UnconditionalNext(sexpr: cin.Seq):
    match (sexpr):
        case cin.IndexSeq(_, _):
            return cpp.IncAssign(cpputil.ArrayIndexVariable(sexpr), cpp.Constant(1))
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.Block(stmts=[UnconditionalNext(s1), UnconditionalNext(s2)])
        case cin.SliceSeq(a, s, _, r):
            return cpp.Block(
                [
                    UnconditionalNext(a),
                    cpp.While(
                        cond=cpp.And(
                            Valid(a),
                            cpp.Not(
                                cpp.Eq(
                                    cpp.Mod(
                                        cpp.Sub(Eval(a), cpp.Constant(s)),
                                        cpp.Constant(r),
                                    ),
                                    cpp.Constant(0),
                                )
                            ),
                        ),
                        body=UnconditionalNext(a),
                    ),
                ]
            )
        case cin.Product(s1, s2):
            return cpp.Block(
                stmts=[
                    UnconditionalNext(s2),
                    cpp.While(
                        cond=cpp.And(Valid(s1), cpp.Not(Valid(s2))),
                        body=cpp.Block(
                            [
                                UnconditionalNext(s1),
                                Reset(s2),
                            ]
                        ),
                    ),
                ]
            )
        case cin.Concatenate(s1, s2):
            raise NotImplementedError(type(sexpr))
        case _:
            raise NotImplementedError(type(sexpr))


def Equals(value: cpp.Cpp, sexpr: cin.Seq):
    match (sexpr):
        case cin.IndexSeq():
            return cpp.Eq(value, cpputil.ArrayIndexVariable(sexpr))
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.And(Equals(value, s1), Equals(value, s2))
        case cin.Product(s1, s2):
            return cpp.And(
                Equals(cpp.Div(value, il.Size(s2)), s1),
                Equals(cpp.Mod(value, il.Size(s2)), s2),
            )
        case cin.Concatenate(s1, s2):
            raise NotImplementedError(type(sexpr))
        case cin.SliceSeq(a, s, e, r):
            raise NotImplementedError(type(sexpr))
        case _:
            raise NotImplementedError(type(sexpr))
