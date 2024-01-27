from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence

import scorch.compiler.cin as cin
import scorch.format as format

# The control flow intermediate representation (CFIR) as presented in
# "Compilation of Shape Operators on Sparse Arrays" by Root, et. al.


@dataclass
class CFIR:
    pass


@dataclass
class Switch(CFIR):
    idx: cin.IndexVar
    cases: List[Pair[cin.Seq, cin.CIN]]

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


@dataclass
class Lattice(CFIR):
    sexpr: cin.Seq
    children: dict[cin.Seq, Lattice] = field(default_factory=dict)

    def add_child(self, sub: cin.Seq, point: Lattice):
        self.children[sub] = point

    def seq(self):
        return sexpr

    def subpoints(self):
        return children.values()


def 𝜒(sexpr: cin.Seq):
    match sexpr:
        case cin.FullSeq(_) | cin.EmptySeq(_):
            return {}
        case cin.IndexSeq(_, _, _):
            return {sexpr}
        case cin.SliceSeq(a, _, _, _):
            return 𝜒(a) | {sexpr}
        case _:
            raise NotImplementedError(type(sexpr))


def TopoSortHelper(
    node: cin.Seq, graph: dict[cin.Seq, Lattice], visited: set[cin.Seq] = set()
):
    if node in visited or isinstance(node, cin.FullSeq | cin.EmptySeq):
        return []
    nodes = []
    visited.add(node)
    for c, cs in graph.items():
        if len(cs) == 0:
            continue
        nodes.extend(TopoSortHelper(c, cs, visited))
    return nodes + [node]


def TopoSort(il: IterationLattice):
    t = TopoSortHelper(il.sexpr, il.children)
    t = list(dict.fromkeys(t))  # De-duplicate
    t = t[::-1]  # Reverse
    return t


def Size(sexpr: cin.Seq):
    match sexpr:
        case cin.IndexSeq(idx, tensor, size):
            return size
        case cin.FullSeq(sz) | cin.EmptySeq(sz):
            return sz
        case cin.SliceSeq(a, s, e, r):
            return (e - s) + (r - 1) // r
        case _:
            return NotImplementedError(type(sexpr))


def Remove(sexpr: cin.Seq, sub: cin.Seq):
    if sexpr == sub:
        assert hasattr(sexpr, "format"), sexpr
        size = Size(sexpr)
        return (
            cin.FullSeq(size)
            if sexpr.format == format.LevelType.DENSE
            else cin.EmptySeq(size)
        )
    match sexpr:
        case cin.IndexSeq(_, _, _):
            return sexpr
        case cin.SliceSeq(a, s, e, r):
            return cin.SliceSeq(Remove(a, sub), s, e, r)
        case _:
            raise NotImplementedError(type(sexpr))


def Simplify(sexpr: cin.Seq, defs: set[T]):
    match sexpr:
        case cin.IndexSeq(_, _, size, _, _, format):
            # TODO(cgyurgyik): Verify the index is defined already.
            return (
                cin.FullSeq(size)
                if format == format.LevelType.DENSE
                else cin.EmptySeq(size)
            )
        case cin.FullSeq(_) | cin.EmptySeq(_):
            return sexpr
        case cin.SliceSeq(a, s, e, r):
            match x := Simplify(a, defs):
                case cin.FullSeq(_):
                    return cin.FullSeq(Size(sexpr))
                case cin.EmptySeq(_):
                    return cin.EmptySeq(Size(sexpr))
                case _:
                    return cin.SliceSeq(x, s, e, r)
        case _:
            raise NotImplementedError(type(sexpr))


def ConstructGraph(sexpr: cin.Seq, defs: set[T], visited: set[cin.Seq] = set()):
    if sexpr in visited:
        return {}
    set.update({sexpr})
    graph = {}
    edges = 𝜒(sexpr)
    for sub in edges:
        r = Remove(sexpr, sub)
        s = Simplify(r, defs)
        l = ConstructGraph(s, defs, visited)
        graph[sub] = l
    return graph


def ConstructLattice(top: cin.Seq, defs: set[T]):
    return Lattice(top, ConstructGraph(top, defs))


def Iters(sexpr: cin.Seq):
    match sexpr:
        case cin.SliceSeq(a, _, _, _):
            return Iters(a)
        case cin.IndexSeq(_, _, _):
            return {sexpr}
        case _:
            raise NotImplementedError(type(sexpr))


def Subpoints(il: IterationLattice):
    subs = TopoSortHelper(il.sexpr, il.children)
    iters = Iters(il.sexpr)
    # def f(x): return Iters(x) <= iters
    # return list(filter(f, subs))
    return iters


def BuildLoop(point: cin.Seq, lattice: IterationLattice, fa: cin.ForAll):
    # TODO(cgyurgyik): Add simplify function for `CIN`.
    # TODO(cgyurgyik): `locs` and `defs` manipulation here.
    idx: cin.IndexVar = fa.index_var
    body: cin.IndexStmt = fa.stmt

    def build(c: cin.CIN):
        return [c, Lower(body)]

    subpoints = Subpoints(lattice)
    bodies = [build(sub) for sub in subpoints]

    # TODO(cgyurgyik): Check if this contains an intersection sequence.
    body = bodies[0][1] if len(bodies) == 1 else Switch(idx, bodies)
    return Loop(idx, point, body)


def CompileForAll(fa: cin.ForAll, defs: set[T]):
    lattice = ConstructLattice(fa.seq, defs)
    return [BuildLoop(p, lattice, fa) for p in TopoSort(lattice)]


def CompileTensorAssign(c: cin.TensorAssign, defs: set[T]):
    if c.op is None:
        return Assign(c.lhs, c.rhs)
    raise NotImplementedError(c.op)


def Lower(c: cin.CIN, defs: set[T] = set()):
    """Lowers Concrete Index Notation (CIN) to CFIR."""
    match c:
        case cin.ForAll():
            return CompileForAll(c, defs)
        case cin.TensorAssign():
            return CompileTensorAssign(c, defs)
        case _:
            raise NotImplementedError(type(c))
