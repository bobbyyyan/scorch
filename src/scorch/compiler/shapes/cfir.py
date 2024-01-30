from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence

from scorch.compiler import cin
from scorch.compiler.shapes import lattice as il

# The control flow intermediate representation (CFIR) as presented in
# "Compilation of Shape Operators on Sparse Arrays" by Root, et. al.


@dataclass
class CFIR:
    pass


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

    def __lt__(self, other):
        return self.sexpr < other.sexpr

    def __str__(self):
        newline = "\n"
        return f"while {self.idx} <-- {self.sexpr}{newline}{self.body}"

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

    def __str__(self):
        return f"{self.lhs} = {self.rhs}"

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
    locdefs = set(map(lambda p: il.Iters(p[1]), locs))
    idx: cin.IndexVar = fa.index_var
    body: cin.IndexStmt = fa.stmt

    def Build(child: cin.Seq):
        newdefs: set[cin.Seq] = defs | il.Iters(child) | locdefs
        return SwitchCase(sexpr=child, stmt=Lower(il.Simplify(body, newdefs), newdefs))

    bodies = list(map(Build, sorted(il.Subpoints(lattice, point))))

    # TODO(cgyurgyik): Check if this contains an intersection sequence.
    # Skip creating a switch if there is only a single case.
    body = bodies.pop().stmt if len(bodies) == 1 else Switch(idx, bodies)
    return Loop(idx, point, body)


def FilterLocators(locs: Tuple[cin.Seq, list[Tuple[cin.Seq, cin.Seq]]], point: cin.Seq):
    return list(filter(lambda p: il.Contains(point, p[0]), locs))


def FindLocators(sexpr: cin.Seq) -> Tuple[cin.Seq, list[Tuple[cin.Seq, cin.Seq]]]:
    match sexpr:
        case cin.IndexSeq():
            return [sexpr, []]
        case cin.UnionSeq(s1, s2):
            aexpr, alocs = FindLocators(s1)
            bexpr, blocs = FindLocators(s2)
            return (
                [aexpr, [(a, b) for a, b in zip(alocs, blocs) + [(aexpr, bexpr)]]]
                if il.IsDense(aexpr) and il.IsDense(bexpr)
                else [
                    cin.UnionSeq(aexpr, bexpr),
                    [(a, b) for a, b in zip(alocs, blocs)],
                ]
            )
        case cin.SliceSeq(a, s, e, r):
            aexpr, alocs = FindLocators(a)
            return [cin.SliceSeq(aexpr, s, e, r), alocs]
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
    if c.op is None:
        return Assign(c.lhs, c.rhs)
    raise NotImplementedError(c.op)


def Lower(c: cin.CIN, defs: set[cin.Seq] = set()) -> CFIR:
    """Lowers Concrete Index Notation (CIN) to CFIR."""
    match c:
        case cin.ForAll():
            return CompileForAll(c, defs)
        case cin.TensorAssign():
            return CompileTensorAssign(c, defs)
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
        case Switch(idx, cases):
            # switch `idx`:
            #   `s0` = `c0`
            #   `s1` = `c1`
            #   ...
            pp += indent()
            pp += f"switch {PpExpr(idx)}"
            for case in cases:
                pp += PpExpr(case)
        case Loop(idx, sexpr, body):
            # while `idx` <-- `sexpr`
            #   stmt0
            #   stmt1
            #   ...
            pp += indent()
            pp += f"while {PpExpr(idx)} <-- {PpExpr(sexpr)} "
            pp += "\n"
            pp += PrettyPrint(body, indent_level + 2)
            pp += "\n"
        case _:
            pp += indent()
            pp += PpExpr(stmt)
    return pp
