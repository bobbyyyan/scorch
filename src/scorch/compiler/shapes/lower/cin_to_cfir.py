from __future__ import annotations
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
from functools import reduce

from scorch.compiler import cin
from scorch.compiler.shapes.ast import cfir, ir
from scorch.compiler.shapes.lower import seq_util, lattice as il


def SimplifyExpr(e: cin.IndexExpr, defs: set[cin.Seq]) -> Optional[cin.IndexExpr]:
    """Simplifies the expression `e` to another (optional) expression."""
    match e:
        case cin.Workspace():
            return e
        case cin.TensorAccess():
            return e if seq_util.IndicesDefined(e, defs) else None
        case cin.BinaryOp():
            x: Optional[cin.IndexExpr] = SimplifyExpr(e.left, defs)
            y: Optional[cin.IndexExpr] = SimplifyExpr(e.right, defs)
            match op := e.op:
                case cin.Operation.ADD:
                    if x is None:
                        return y
                    if y is None:
                        return x
                    return cin.BinaryOp(cin.Operation.ADD, x, y)
                case cin.Operation.MUL:
                    return (
                        None
                        if None in (x, y)
                        else cin.BinaryOp(cin.Operation.MUL, x, y)
                    )
                case _:
                    raise NotImplementedError(op)

        case _:
            raise NotImplementedError(type(e))


def SimplifyStmt(c: cin.IndexStmt, defs: set[cin.Seq]) -> cin.CIN:
    match c:
        case cin.ForAll():
            sexpr: cin.Seq = seq_util.SimplifySeq(c.seq, defs)
            assert not isinstance(sexpr, cin.EmptySeq | cin.FullSeq), f"{c}\n{defs}"
            return cin.ForAll(
                index_var=c.index_var,
                stmt=SimplifyStmt(c.stmt, defs | seq_util.Iters(sexpr)),
                seq=sexpr,
            )
        case cin.TensorAssign():
            rhs = SimplifyExpr(c.rhs, defs)
            assert rhs is not None, f"{c}, {defs}"
            return cin.TensorAssign(lhs=c.lhs, rhs=rhs, op=c.op)
        case cin.Where():
            return cin.Where(
                producer=SimplifyStmt(c.producer, defs),
                consumer=SimplifyStmt(c.consumer, defs),
                workspace=c.workspace,
            )
        case _:
            raise NotImplementedError(type(c))


def BuildLoop(
    point: cin.Seq,
    lattice: il.IterationLattice,
    construct: cin.ForAll,
    defs: set[cin.Seq],
    locs: Tuple[cin.Seq, list[Tuple[cin.Seq, cin.Seq]]],
) -> cfir.Loop:
    locs = FilterLocators(locs, point)
    locdefs = map(lambda p: seq_util.Iters(p[1]), locs)
    locdefs = reduce(set.union, locdefs, set())
    idx: ir.IndexVar = ir.IndexVar.from_cin(construct.index_var)
    body: cin.IndexStmt = construct.stmt

    def Build(child: cin.Seq):
        newdefs: set[cin.Seq] = defs | seq_util.Iters(child) | locdefs
        return cfir.SwitchCase(
            sexpr=child, stmt=Lower(SimplifyStmt(body, newdefs), newdefs)
        )

    bodies = list(map(Build, sorted(il.Subpoints(lattice, point))))

    # Skip creating a switch if there is only a single case.
    body = (
        cfir.Switch(idx, bodies)
        if len(bodies) > 1 or seq_util.ContainsIntersection(point)
        else bodies.pop().stmt
    )
    return cfir.Loop(idx, point, body, locs)


def RemoveDense(sexpr: cin.Seq) -> Tuple[cin.Seq, list[cin.Seq]]:
    if seq_util.IsDense(sexpr):
        return [None, [sexpr]]
    match sexpr:
        case cin.UnionSeq(s1, s2):
            x, xlocs = RemoveDense(s1)
            y, ylocs = RemoveDense(s2)
            if x is None:
                return [y, xlocs + ylocs]
            if y is None:
                return [x, xlocs + ylocs]
            return [cin.UnionSeq(x, y), xlocs + ylocs]
        case cin.IntersectionSeq(s1, s2):
            x, xlocs = RemoveDense(s1)
            y, ylocs = RemoveDense(s2)
            if x is None:
                return [y, xlocs + ylocs]
            if y is None:
                return [x, xlocs + ylocs]
            return [cin.IntersectionSeq(x, y), xlocs + ylocs]
        case _:
            return [sexpr, []]


def FilterLocators(
    locs: Tuple[cin.Seq, list[Tuple[cin.Seq, cin.Seq]]], point: cin.Seq
) -> Tuple[cin.Seq, list[Tuple[cin.Seq, cin.Seq]]]:
    return list(filter(lambda p: seq_util.Contains(point, p[0]), locs))


def FindLocators(
    sexpr: cin.Seq, dense_locate=False
) -> Tuple[cin.Seq, list[Tuple[cin.Seq, ...]]]:
    """
    Note: The `dense_locate` flag is a quick-fix.
    Reference: https://github.com/rootjalex/burrito/issues/1"""
    match sexpr:
        case cin.IndexSeq() | cin.Universe():
            return [sexpr, []]
        case cin.UnionSeq(s1, s2):
            aexpr, alocs = FindLocators(s1)
            bexpr, blocs = FindLocators(s2)
            if seq_util.IsDense(aexpr) and seq_util.IsDense(bexpr):
                if isinstance(s2, cin.IntersectionSeq | cin.UnionSeq):
                    bexpr, blocs = FindLocators(s2, dense_locate=True)
                if dense_locate:
                    return [cin.UnionSeq(aexpr, bexpr), alocs + blocs]
                return [aexpr, alocs + blocs + [(aexpr, bexpr)]]
            return [cin.UnionSeq(aexpr, bexpr), alocs + blocs]
        case cin.IntersectionSeq(s1, s2):
            aexpr, alocs = FindLocators(s1)
            bexpr, blocs = FindLocators(s2)
            if not seq_util.IsDense(s1):
                # Use s1 to locate into s2.
                y, ydense = RemoveDense(bexpr)
                r = aexpr if y is None else cin.IntersectionSeq(aexpr, y)
                ylocs = [(r, y) for y in ydense]
                return [r, alocs + blocs + ylocs]
            if not seq_util.IsDense(s2):
                # Use s2 to locate into s1.
                x, xdense = RemoveDense(aexpr)
                r = bexpr if x is None else cin.IntersectionSeq(x, bexpr)
                xlocs = [(r, x) for x in xdense]
                return [r, alocs + blocs + xlocs]
            # Both are dense.
            if dense_locate:
                return [
                    cin.IntersectionSeq(aexpr, bexpr),
                    alocs + blocs + [(aexpr, bexpr)],
                ]
            return [aexpr, alocs + blocs + [(aexpr, bexpr)]]
        case cin.SliceSeq(a, s, e, r):
            aexpr, alocs = FindLocators(a)
            return [cin.SliceSeq(aexpr, s, e, r), alocs]
        case cin.ConcatenateSeq(s1, s2):
            aexpr, alocs = FindLocators(s1)
            bexpr, blocs = FindLocators(s2)
            return [cin.ConcatenateSeq(aexpr, bexpr), alocs + blocs]
        case cin.ProductSeq(s1, s2):
            aexpr, alocs = FindLocators(s1)
            bexpr, blocs = FindLocators(s2)
            return [cin.ProductSeq(aexpr, bexpr), alocs + blocs]
        case cin.ProjectSeq(a, k, I, J):
            aexpr, alocs = FindLocators(a)
            return [cin.ProjectSeq(aexpr, k, I, J), alocs]
        case _:
            raise NotImplementedError(str(sexpr))


def CompileForAll(fa: cin.ForAll, defs: set[cin.Seq]) -> cfir.CFIR | list[cfir.CFIR]:
    """Lowers CIN `For All` construct to CFIR."""
    sexpr: cin.Seq = fa.seq
    sexpr, locs = FindLocators(sexpr)
    lattice: il.IterationLattice = il.ConstructIterationLattice(sexpr, defs)
    loops: list[cfir.Loop] = [
        BuildLoop(p, lattice, fa, defs, locs) for p in il.TopologicalSort(lattice)
    ]
    return loops.pop() if len(loops) == 1 else cfir.Block(stmts=sorted(loops))


def CompileTensorAssign(c: cin.TensorAssign, defs: set[cin.Seq]) -> cfir.CFIR:
    """Lowers CIN `Tensor Assign` construct to CFIR."""
    return cfir.Assign(c.lhs, c.rhs, c.op)


def CompileWhere(c: cin.Where, defs: set[cin.Seq]) -> cfir.CFIR:
    """Lowers CIN `where` construct to CFIR."""
    return cfir.Allocate(
        t=c.workspace,
        producer=Lower(c.producer, defs),
        consumer=Lower(c.consumer, defs),
    )


def Lower(c: cin.CIN, defs: set[cin.Seq] = set()) -> cfir.CFIR:
    """Lowers Concrete Index Notation (CIN) to CFIR."""
    match c:
        case cin.ForAll():
            return CompileForAll(c, defs)
        case cin.TensorAssign():
            return CompileTensorAssign(c, defs)
        case cin.Where():
            return CompileWhere(c, defs)
        case _:
            raise NotImplementedError(type(c))
