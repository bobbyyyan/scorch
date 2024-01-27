from __future__ import annotations
from dataclasses import dataclass, field
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


@dataclass
class Lattice:
    sexpr: cin.Seq
    children: dict[cin.Seq, Lattice] = field(default_factory=dict)


def 𝜒(sexpr: cin.Seq) -> set[cin.Seq]:
    match sexpr:
        case cin.FullSeq() | cin.EmptySeq():
            return {}
        case cin.IndexSeq():
            return {sexpr}
        case cin.SliceSeq(a):
            return 𝜒(a) | {sexpr}
        case _:
            raise NotImplementedError(type(sexpr))


def TopoSortRec(
    node: cin.Seq, graph: dict[cin.Seq, Lattice], visited: set[cin.Seq] = set()
):
    if node in visited or isinstance(node, cin.FullSeq | cin.EmptySeq):
        return []
    nodes = []
    visited.add(node)
    for c, cs in graph.items():
        if len(cs) == 0:
            continue
        nodes.extend(TopoSortRec(c, cs, visited))
    return nodes + [node]


def TopoSort(lattice: Lattice):
    t = TopoSortRec(lattice.sexpr, lattice.children)
    t = list(dict.fromkeys(t))  # De-duplicate
    t = t[::-1]  # Reverse
    return t


def Size(sexpr: cin.Seq) -> int:
    match sexpr:
        case cin.IndexSeq(_, _, size):
            return size
        case cin.FullSeq(sz) | cin.EmptySeq(sz):
            return sz
        case cin.SliceSeq(_, s, e, r):
            return (e - s) + (r - 1) // r
        case _:
            return NotImplementedError(type(sexpr))


def IsDense(sexpr: cin.Seq) -> bool:
    match sexpr:
        case cin.IndexSeq(_, _, _, _, fmt):
            return fmt == format.LevelType.DENSE
        case cin.SliceSeq(a):
            return IsDense(a)
        case _:
            raise NotImplementedError(type(sexpr))


def Remove(sexpr: cin.Seq, sub: cin.Seq) -> cin.Seq:
    if sexpr == sub:
        sz: int = Size(sexpr)
        return cin.FullSeq(sz) if IsDense(sexpr) else cin.EmptySeq(sz)
    match sexpr:
        case cin.IndexSeq(_, _, _):
            return sexpr
        case cin.SliceSeq(a, s, e, r):
            return cin.SliceSeq(Remove(a, sub), s, e, r)
        case _:
            raise NotImplementedError(type(sexpr))


def Simplify(sexpr: cin.Seq, defs: set[Any]) -> cin.Seq:
    match sexpr:
        case cin.IndexSeq(_, _, sz, _, _, format):
            # TODO(cgyurgyik): Verify the index is defined already.
            return (
                cin.FullSeq(sz)
                if format == format.LevelType.DENSE
                else cin.EmptySeq(sz)
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


def ConstructGraph(sexpr: cin.Seq, defs: set[Any], visited: set[cin.Seq] = set()):
    if sexpr in visited:
        return {}
    set.update({sexpr})
    graph = {}
    edges: set[cin.Seq] = 𝜒(sexpr)
    # TODO(cgyurgyik): This is nondeterministic.
    for sub in edges:
        r: cin.Seq = Remove(sexpr, sub)
        s: cin.Seq = Simplify(r, defs)
        l = ConstructGraph(s, defs, visited)
        graph[sub] = l
    return graph


def ConstructLattice(top: cin.Seq, defs: set[Any]):
    return Lattice(top, ConstructGraph(top, defs))


def Iters(sexpr: cin.Seq):
    match sexpr:
        case cin.SliceSeq(a):
            return Iters(a)
        case cin.IndexSeq():
            return {sexpr}
        case _:
            raise NotImplementedError(type(sexpr))


# TODO(cgyurgyik): This needs to be fixed.
def Subpoints(lattice: Lattice):
    subs = TopoSort(lattice)
    iters = Iters(lattice.sexpr)
    # def f(x): return Iters(x) <= iters
    # return list(filter(f, subs))
    return iters


def BuildLoop(point: cin.Seq, lattice: Lattice, fa: cin.ForAll):
    # TODO(cgyurgyik): Add simplify function for `CIN`.
    # TODO(cgyurgyik): `locs` and `defs` manipulation here.
    idx: cin.IndexVar = fa.index_var
    body: cin.IndexStmt = fa.stmt

    def build(c: cin.CIN):
        return [c, Lower(body)]

    bodies = [build(sub) for sub in Subpoints(lattice)]

    # TODO(cgyurgyik): Check if this contains an intersection sequence.
    body = bodies[0][1] if len(bodies) == 1 else Switch(idx, bodies)
    return Loop(idx, point, body)


def CompileForAll(fa: cin.ForAll, defs: set[Any]):
    lattice: Lattice = ConstructLattice(fa.seq, defs)
    loops: list[Loop] = [BuildLoop(p, lattice, fa) for p in TopoSort(lattice)]
    return loops.pop() if len(loops) == 1 else loops


def CompileTensorAssign(c: cin.TensorAssign, defs: set[Any]):
    if c.op is None:
        return Assign(c.lhs, c.rhs)
    raise NotImplementedError(c.op)


def Lower(c: cin.CIN, defs: set[Any] = set()):
    """Lowers Concrete Index Notation (CIN) to CFIR."""
    match c:
        case cin.ForAll():
            return CompileForAll(c, defs)
        case cin.TensorAssign():
            return CompileTensorAssign(c, defs)
        case _:
            raise NotImplementedError(type(c))
