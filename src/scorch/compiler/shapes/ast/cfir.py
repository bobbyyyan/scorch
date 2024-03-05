from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence

from scorch.compiler import cin

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
