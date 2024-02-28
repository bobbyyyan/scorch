from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
from functools import reduce

from scorch.compiler import cin
from scorch.compiler.shapes import lattice as il

# The control flow intermediate representation (CFIR) as presented in
# "Compilation of Shape Operators on Sparse Arrays" by Root, et. al.


@dataclass
class CFIR:
    pass


@dataclass
class Allocate:
    t: cin.Workspace
    producer: CFIR
    consumer: CFIR

    def __str__(self):
        return f"{self.t} {{ {self.producer} in {self.consumer} }}"

    def __repr__(self):
        return str(self)


@dataclass
class SwitchCase:
    sexpr: cin.Seq
    stmt: CFIR

    def __str__(self):
        return f"case {self.sexpr}: {self.stmt}"

    def __repr__(self):
        return str(self)


@dataclass
class Switch(CFIR):
    idx: cin.IndexVar
    cases: List[SwitchCase]

    def __str__(self):
        newline = "\n"
        return f"switch {self.idx}:{newline}{newline.join(str(c) for c in self.cases)}"

    def __repr__(self):
        return str(self)


@dataclass
class Loop(CFIR):
    idx: cin.IndexVar
    sexpr: cin.Seq
    body: cin.IndexStmt
    locs: list[Tuple[cin.Seq, cin.Seq]]

    def __lt__(self, other):
        return self.sexpr < other.sexpr

    def __str__(self):
        newline = "\n"
        s = f"while {self.idx} <-- {self.sexpr}{newline}{self.body}"
        if len(self.locs) > 0:
            ll = " ".join([f"{p[0]}={p[1]}" for p in self.locs])
            s += f"with {ll}"
        return s

    def __repr__(self):
        return str(self)


@dataclass
class Block(CFIR):
    stmts: list[CFIR]

    def __str__(self):
        return "\n".join(str(stmt) for stmt in self.stmts)


@dataclass
class Assign(CFIR):
    lhs: cin.TensorAccess
    rhs: cin.IndexExpr
    op: Optional[cin.Operation] = None

    def __str__(self):
        return f"{self.lhs} {self.op or ''}= {self.rhs}"

    def __repr__(self):
        return str(self)


def BuildLoop(
    point: cin.Seq,
    lattice: il.Lattice,
    fa: cin.ForAll,
    defs: set[cin.Seq],
    locs: Tuple[cin.Seq, list[Tuple[cin.Seq, cin.Seq]]],
) -> Loop:
    locs = FilterLocators(locs, point)
    locdefs = map(lambda p: il.Iters(p[1]), locs)
    locdefs = reduce(set.union, locdefs, set())
    idx: cin.IndexVar = fa.index_var
    body: cin.IndexStmt = fa.stmt

    def Build(child: cin.Seq):
        newdefs: set[cin.Seq] = defs | il.Iters(child) | locdefs
        return SwitchCase(
            sexpr=child, stmt=Lower(il.SimplifyStmt(body, newdefs), newdefs)
        )

    bodies = list(map(Build, sorted(il.Subpoints(lattice, point))))

    # Skip creating a switch if there is only a single case.
    body = (
        Switch(idx, bodies)
        if len(bodies) > 1 or il.ContainsIntersection(point)
        else bodies.pop().stmt
    )
    return Loop(idx, point, body, locs)


def RemoveDense(sexpr: cin.Seq) -> Tuple[cin.Seq, list[cin.Seq]]:
    if il.IsDense(sexpr):
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


def FilterLocators(locs: Tuple[cin.Seq, list[Tuple[cin.Seq, cin.Seq]]], point: cin.Seq):
    return list(filter(lambda p: il.Contains(point, p[0]), locs))


def FindLocators(
    sexpr: cin.Seq, dense_locate=False
) -> Tuple[cin.Seq, list[Tuple[cin.Seq, ...]]]:
    match sexpr:
        case cin.IndexSeq() | cin.Universe():
            return [sexpr, []]
        case cin.UnionSeq(s1, s2):
            aexpr, alocs = FindLocators(s1)
            bexpr, blocs = FindLocators(s2)
            if il.IsDense(aexpr) and il.IsDense(bexpr):
                if isinstance(s2, cin.IntersectionSeq | cin.UnionSeq):
                    bexpr, blocs = FindLocators(s2, dense_locate=True)
                if dense_locate:
                    return [cin.UnionSeq(aexpr, bexpr), alocs + blocs]
                return [aexpr, alocs + blocs + [(aexpr, bexpr)]]
            return [cin.UnionSeq(aexpr, bexpr), alocs + blocs]
        case cin.IntersectionSeq(s1, s2):
            aexpr, alocs = FindLocators(s1)
            bexpr, blocs = FindLocators(s2)
            if not il.IsDense(s1):
                # Use s1 to locate into s2.
                y, ydense = RemoveDense(bexpr)
                r = aexpr if y is None else cin.IntersectionSeq(aexpr, y)
                ylocs = [(r, y) for y in ydense]
                return [r, alocs + blocs + ylocs]
            if not il.IsDense(s2):
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


def CompileForAll(fa: cin.ForAll, defs: set[cin.Seq]) -> CFIR | list[CFIR]:
    sexpr: cin.Seq = fa.seq
    sexpr, locs = FindLocators(sexpr)
    lattice: il.IterationLattice = il.ConstructIterationLattice(sexpr, defs)
    loops: list[Loop] = [
        BuildLoop(p, lattice, fa, defs, locs) for p in il.TopologicalSort(lattice)
    ]
    return loops.pop() if len(loops) == 1 else Block(stmts=sorted(loops))


def CompileTensorAssign(c: cin.TensorAssign, defs: set[cin.Seq]) -> CFIR:
    return Assign(c.lhs, c.rhs, c.op)


def CompileWhere(c: cin.Where, defs: set[cin.Seq]) -> CFIR:
    return Allocate(
        t=c.workspace,
        producer=Lower(c.producer, defs),
        consumer=Lower(c.consumer, defs),
    )


def Lower(c: cin.CIN, defs: set[cin.Seq] = set()) -> CFIR:
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


#######################################
############# Pretty Print #############
########################################


def PrettyPrint(stmt: CFIR, indent_level: int = 0) -> str:
    """
    Pretty print for the CF intermediate representation.
    This will handle indentation.
    """

    def indent(i: int = indent_level) -> str:
        return i * " "

    def PpExpr(e: CFIR) -> str:
        if not isinstance(e, SwitchCase):
            return str(e)
        pp: str = ""
        pp += "\n"
        pp += indent(indent_level + 2)
        pp += f"case: {case.sexpr}"
        pp += "\n"
        pp += indent(indent_level + 4)
        pp += f"{case.stmt}"
        return pp

    pp: str = ""
    match stmt:
        case Block(stmts):
            return "\n".join(PrettyPrint(stmt, indent_level) for stmt in stmts)
        case Allocate(t, producer, consumer):
            pp += indent()
            pp += "let"
            pp += "\n"
            pp += PrettyPrint(producer, indent_level + 2)
            pp += "\n"
            pp += indent()
            pp += "in"
            pp += "\n"
            pp += PrettyPrint(consumer, indent_level + 4)
        case Switch(idx, cases):
            # switch `idx`:
            #   `s0` = `c0`
            #   `s1` = `c1`
            #   ...
            pp += indent()
            pp += f"switch {PpExpr(idx)}"
            for case in cases:
                pp += PpExpr(case)
        case Loop(idx, sexpr, body, locs):
            # while `idx` <-- `sexpr`
            #   stmt0
            #   stmt1
            #   ...
            pp += indent()
            pp += f"while {PpExpr(idx)} <-- {PpExpr(sexpr)} "
            if len(locs) > 0:
                ll = " ".join([f"{p[0]}={p[1]}" for p in locs])
                pp += f"with {ll}"
            pp += "\n"
            pp += PrettyPrint(body, indent_level + 2)
            pp += "\n"
        case _:
            pp += indent()
            pp += PpExpr(stmt)
    return pp
