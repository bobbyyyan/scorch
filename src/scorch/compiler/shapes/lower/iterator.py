from typing import List, Optional, Any, Tuple, Callable, Union, Sequence

from scorch.compiler import cin as cin
from scorch.compiler.shapes.ast import cpp
from scorch.compiler.shapes.lower import cpputil, sequtil

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
        case cin.Universe(idx):
            return cpp.Define(
                type=cpp.IndexType(),
                lhs=cpputil.UniverseIndexVariable(idx),
                rhs=cpp.Constant(0),
            )
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.Block(stmts=[Init(s1), Init(s2)])
        case cin.ConcatenateSeq(s1, s2):
            raise NotImplementedError(sexpr)
        case cin.ProductSeq(s1, s2):
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
        case cin.ProjectSeq(a, k, I, J):
            return (
                Init(a)
                if k == 0
                else cpp.Define(
                    cpp.IndexType(),
                    cpputil.ProjectionVariable(k),
                    Eval(cin.ProjectSeq(a, 0, I, J)),
                )
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
                            )
                            # TODO(cgyurgyik): we want these fields to be cpp.Constant/cpp.Variable as well.
                            if s == 0  # if s is 0, a < 0 is always false.
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
        case cin.Universe(idx):
            return cpp.Define(
                type=cpp.IndexType(),
                lhs=cpputil.UniverseIndexVariable(idx),
                rhs=cpp.Constant(0),
            )
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.Block(stmts=[Reset(s1), Reset(s2)])
        case cin.ProductSeq(s1, s2):
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
        case cin.ProjectSeq(a, k, I, J):
            return (
                Reset(a)
                if k == 0
                else cpp.Assign(
                    cpputil.ProjectionVariable(k), cin.ProjectSeq(a, 0, I, J)
                )
            )
        case cin.ConcatenateSeq(s1, s2):
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
        case cin.Universe(idx, size):
            return cpp.Lt(cpputil.UniverseIndexVariable(idx), cpp.Constant(size))
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.And(Valid(s1), Valid(s2))
        case cin.ConcatenateSeq(s1, s2) | cin.ProductSeq(s1, s2):
            return cpp.And(Valid(s1), Valid(s2))
        case cin.SliceSeq(a, _, e, r):
            return cpp.And(Valid(a), cpp.Lt(Eval(a), cpp.Constant(e)))
        case cin.ProjectSeq(a, k, I, J):
            return (
                Valid(a)
                if k == 0
                else cpp.Eq(
                    cpputil.ProjectionVariable(k), Eval(cin.ProjectSeq(a, 0, I, J))
                )
            )
        case _:
            raise NotImplementedError(type(sexpr))


def Eval(sexpr: cin.Seq):
    match (sexpr):
        case cin.IndexSeq():
            return cpputil.ArrayAccessCrd(sexpr)
        case cin.Universe(idx):
            return cpputil.UniverseIndexVariable(idx)
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.Min(Eval(s1), Eval(s2))
        case cin.ConcatenateSeq(s1, s2):
            raise NotImplementedError(type(sexpr))
        case cin.ProductSeq(s1, s2):
            # a * |b| + b
            return cpp.Add(cpp.Mul(Eval(s1), sequtil.Size(s2)), Eval(s2))
        case cin.ProjectSeq(a, k, I, J):
            a = Eval(a)
            return cpp.Div(a, J) if k == 0 else cpp.Mod(a, J)
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
        case cin.Universe(idx):
            return cpp.IncAssign(
                cpputil.UniverseIndexVariable(idx), cpp.Eq(value, Eval(sexpr))
            )
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.Block(stmts=[Next(value, s1), Next(value, s2)])
        case cin.ProductSeq(s1, s2):
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
        case cin.ProjectSeq(a, k, I, J):
            raise NotImplementedError(type(sexpr))
        case cin.ConcatenateSeq(s1, s2):
            raise NotImplementedError(type(sexpr))
        case cin.SliceSeq(a, s, e, r):
            raise NotImplementedError(type(sexpr))
        case _:
            raise NotImplementedError(type(sexpr))


def UnconditionalNext(sexpr: cin.Seq):
    match (sexpr):
        case cin.IndexSeq(_, _):
            return cpp.IncAssign(cpputil.ArrayIndexVariable(sexpr), cpp.Constant(1))
        case cin.Universe(idx):
            return cpp.IncAssign(cpputil.UniverseIndexVariable(idx), cpp.Constant(1))
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
        case cin.ProductSeq(s1, s2):
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
        case cin.ProjectSeq(a, k, I, J):
            return (
                cpp.While(
                    cond=cpp.And(
                        Valid(a), cpp.Eq(cpputil.ProjectionVariable(1), Eval(sexpr))
                    ),
                    body=UnconditionalNext(a),
                )
                if k == 0
                else UnconditionalNext(a)
            )
        case cin.ConcatenateSeq(s1, s2):
            raise NotImplementedError(type(sexpr))
        case _:
            raise NotImplementedError(type(sexpr))


def Equals(value: cpp.Cpp, sexpr: cin.Seq):
    match (sexpr):
        case cin.IndexSeq():
            return cpp.Eq(value, cpputil.ArrayAccessCrd(sexpr))
        case cin.Universe(idx):
            return cpp.Eq(value, cpputil.UniverseIndexVariable(idx))
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return cpp.And(Equals(value, s1), Equals(value, s2))
        case cin.ProductSeq(s1, s2):
            return cpp.And(
                Equals(cpp.Div(value, sequtil.Size(s2)), s1),
                Equals(cpp.Mod(value, sequtil.Size(s2)), s2),
            )
        case cin.ProjectSeq(a, k, I, J):
            raise NotImplementedError(type(sexpr))
        case cin.ConcatenateSeq(s1, s2):
            raise NotImplementedError(type(sexpr))
        case cin.SliceSeq(a, s, e, r):
            raise NotImplementedError(type(sexpr))
        case _:
            raise NotImplementedError(type(sexpr))
