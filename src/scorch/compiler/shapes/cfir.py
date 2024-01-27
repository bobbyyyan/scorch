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
class Switch(CFIR):
    idx: cin.IndexVar
    cases: List[Tuple[cin.Seq, cin.CIN]]

    def __str__(self):
        newline = "\n"
        return f"switch {self.idx}{newline.join(f'case {i[0]}: {i[1]}' for i in self.cases)}"

    def __repr__(self):
        return str(self)


@dataclass
class Loop(CFIR):
    idx: cin.IndexVar
    sexpr: cin.Seq
    # TODO(cgyurgyik): Add `locs`.
    body: cin.IndexStmt

    def __str__(self):
        newline = "\n"
        return f"while {self.idx} <-- {self.sexpr}{newline}{self.body}"

    def __repr__(self):
        return str(self)


@dataclass
class Assign(CFIR):
    lhs: cin.TensorAccess
    rhs: cin.IndexExpr

    def __str__(self):
        return f"{self.lhs} = {self.rhs}"

    def __repr__(self):
        return str(self)


def BuildLoop(point: cin.Seq, lattice: il.Lattice, fa: cin.ForAll) -> CFIR | list[CFIR]:
    """5.3"""
    # TODO(cgyurgyik): Add simplify function for `CIN`.
    # TODO(cgyurgyik): `locs` and `defs` manipulation here.
    idx: cin.IndexVar = fa.index_var
    body: cin.IndexStmt = fa.stmt

    def build(c: cin.CIN):
        return [c, Lower(body)]

    bodies = [build(sub) for sub in il.Subpoints(lattice)]

    # TODO(cgyurgyik): Check if this contains an intersection sequence.
    body = bodies[0][1] if len(bodies) == 1 else Switch(idx, bodies)
    return Loop(idx, point, body)


def CompileForAll(fa: cin.ForAll, defs: set[Any]) -> CFIR | list[CFIR]:
    """5.3"""
    lattice: il.Lattice = il.ConstructLattice(fa.seq, defs)
    loops: list[Loop] = [BuildLoop(p, lattice, fa) for p in il.TopoSort(lattice)]
    return loops.pop() if len(loops) == 1 else loops


def CompileTensorAssign(c: cin.TensorAssign, defs: set[Any]):
    if c.op is None:
        return Assign(c.lhs, c.rhs)
    raise NotImplementedError(c.op)


def Lower(c: cin.CIN, defs: set[Any] = set()) -> CFIR | list[CFIR]:
    """Lowers Concrete Index Notation (CIN) to CFIR."""
    match c:
        case cin.ForAll():
            return CompileForAll(c, defs)
        case cin.TensorAssign():
            return CompileTensorAssign(c, defs)
        case _:
            raise NotImplementedError(type(c))
