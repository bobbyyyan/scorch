from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
from scorch.compiler import cin
from scorch.compiler.shapes.lower.seq_util import *

# The iteration lattice, as presented in "Compilation of Shape Operators on
# Sparse Arrays" by Root, et. al.


@dataclass
class IterationLattice:
    """A data structure to represent an iteration lattice.
    top: The entry node.
    graph: mapping from sequence expression to (edge, simplified).
    """

    top: cin.Seq
    graph: dict[cin.Seq, Tuple[cin.Seq, cin.Seq]] = field(default_factory=dict)

    def __str__(self):
        def pp(
            node: cin.Seq,
            graph: dict[cin.Seq, Tuple[cin.Seq, cin.Seq]],
            visited: set = set(),
        ):
            s = ""
            if isinstance(node, cin.EmptySeq | cin.FullSeq) or node in visited:
                return s
            visited.add(node)
            children: Tuple[cin.Seq, cin.Seq] = graph[node]
            s += f"{node}"
            s += "{"
            for k, v in children:
                s += "\n"
                s += f"  {k} -- {v}"
            s += "\n"
            s += "}"
            s += "\n"
            for _, v in children:
                s += pp(v, graph)
            return s

        return pp(self.top, self.graph)

    def __repr__(self):
        return self.__str__()


def 𝜒(sexpr: cin.Seq) -> set[cin.Seq]:
    match sexpr:
        case cin.IndexSeq() | cin.Universe():
            return {sexpr}
        case cin.FullSeq() | cin.EmptySeq():
            return {}
        case cin.SliceSeq(a) | cin.ProjectSeq(a):
            return 𝜒(a) | {sexpr}
        case cin.ConcatenateSeq(s1, _) | cin.ProductSeq(s1, _):
            return 𝜒(s1)
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return 𝜒(s1) | 𝜒(s2)
        case _:
            raise NotImplementedError(type(sexpr))


def TopologicalSortRec(
    node: cin.Seq,
    graph: dict[cin.Seq, Tuple[cin.Seq, cin.Seq]],
    visited: set[cin.Seq] = set(),
):
    if node in visited or isinstance(node, cin.EmptySeq | cin.FullSeq):
        return []
    visited.add(node)
    return [
        *sum(map(lambda p: TopologicalSortRec(p[1], graph, visited), graph[node]), []),
        node,
    ]


def TopologicalSort(lattice: IterationLattice):
    return TopologicalSortRec(lattice.top, lattice.graph)[::-1]


def ConstructGraph(sexpr: cin.Seq, defs: set[cin.Seq], visited: set[cin.Seq] = set()):
    if sexpr in visited or isinstance(sexpr, cin.EmptySeq | cin.FullSeq):
        return {}
    visited.update({sexpr})
    edges: set[cin.Seq] = 𝜒(sexpr)
    graph = {}
    pairs: list[Tuple[cin.Seq, cin.Seq]] = []
    for e in edges:
        r: cin.Seq = Remove(sexpr, e)
        s: cin.Seq = SimplifySeq(r, defs)
        graph |= ConstructGraph(s, defs, visited)
        pairs.append((e, s))
    graph[sexpr] = {(e, s) for e, s in pairs}
    return graph


def ConstructIterationLattice(top: cin.Seq, defs: set[cin.Seq]):
    return IterationLattice(top, ConstructGraph(top, defs))


def Subpoints(lattice: IterationLattice, point: cin.Seq):
    """Returns sub-points for `point` in the lattice."""
    subs = TopologicalSortRec(point, lattice.graph, visited=set())
    iters = Iters(point)
    n = list(filter(lambda x: Iters(x) <= iters, subs))
    return n[::-1]
