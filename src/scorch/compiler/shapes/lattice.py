from __future__ import annotations
from dataclasses import dataclass, field
from multipledispatch import dispatch
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
from scorch.compiler import cin
import scorch.format as format

# The iteration lattice, as presented in "Compilation of Shape Operators on
# Sparse Arrays" by Root, et. al.


@dataclass
class IterationLattice:
    """A data structure to represent an iteration lattice.
    sexpr: The entry node.
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
        case cin.SliceSeq(a):
            return 𝜒(a) | {sexpr}
        case cin.Concatenate(s1, _) | cin.Product(s1, _):
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


def Size(sexpr: cin.Seq) -> int:
    match sexpr:
        case cin.IndexSeq(_, _, size):
            # accesses: List[cin.TensorAccess] = var.tensor_accesses()
            # These should all share the same size.
            # (type, ) = set(var.shape[a.level_of_index_var(idx)] for a in accesses)
            return size
        case cin.FullSeq(sz) | cin.EmptySeq(sz):
            return sz
        case cin.SliceSeq(_, s, e, r):
            return (e - s) + (r - 1) // r
        case cin.Product(s1, s2):
            return Size(s1) * Size(s2)
        case cin.Concatenate(s1, s2):
            return Size(s1) + Size(s2)
        case _:
            return NotImplementedError(type(sexpr))


def IsDense(sexpr: cin.Seq) -> bool:
    match sexpr:
        case cin.IndexSeq(_, _, _, _, fmt):
            return fmt == format.LevelType.DENSE
            # accesses: List[cin.TensorAccess] = var.tensor_accesses()
            # # These should all share the same format.
            # (type,) = set(a.level_type_of_index_var(idx) for a in accesses)
            # return type == format.LevelType.DENSE
        case cin.Universe():
            return True
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return IsDense(s1) and IsDense(s2)
        case cin.Product(s1, s2) | cin.Concatenate(s1, s2):
            return IsDense(s1) and IsDense(s2)
        case cin.SliceSeq(a):
            return IsDense(a)
        case _:
            raise NotImplementedError(type(sexpr))


def Remove(sexpr: cin.Seq, sub: cin.Seq) -> cin.Seq:
    if sexpr == sub:
        sz: int = Size(sexpr)
        return cin.FullSeq(sz) if IsDense(sexpr) else cin.EmptySeq(sz)
    match sexpr:
        case cin.IndexSeq() | cin.Universe():
            return sexpr
        case cin.SliceSeq(a, s, e, r):
            return cin.SliceSeq(Remove(a, sub), s, e, r)
        case cin.Product(s1, s2):
            return cin.Product(Remove(s1, sub), Remove(s2, sub))
        case cin.Concatenate(s1, s2):
            return cin.Concatenate(Remove(s1, sub), Remove(s2, sub))
        case cin.UnionSeq(s1, s2):
            return cin.UnionSeq(Remove(s1, sub), Remove(s2, sub))
        case cin.IntersectionSeq(s1, s2):
            return cin.IntersectionSeq(Remove(s1, sub), Remove(s2, sub))
        case _:
            raise NotImplementedError(type(sexpr))


def ConvertToIndexSequences(e: cin.TensorAccess) -> list[cin.IndexSeq]:
    if isinstance(e, cin.WorkspaceAccess):
        return []
    assert isinstance(e, cin.TensorAccess)
    indices: List[cin.IndexVar] = e.get_index_vars()
    levels: List[format.LevelType] = e.level_types()
    tensor: cin.TensorVar = e.get_tensor()

    sequences = []
    for index, (idx, level) in enumerate(zip(indices, levels)):
        sequences.append(
            cin.IndexSeq(
                idx=idx,
                tensor=tensor,
                size=tensor.shape[index],
                index=index,
                format=level,
            )
        )
    return sequences


def GetParent(sexpr: cin.IndexSeq) -> Optional[cin.IndexSeq]:
    idx: cin.IndexVar = sexpr.idx
    var: cin.TensorVar = sexpr.tensor

    accesses: List[cin.TensorAccess] = var.tensor_accesses()
    parent: Optional[cin.IndexVar] = None
    for access in accesses:
        p: Optional[cin.IndexVar] = access.get_parent_index_var(idx)
        if p is None:
            continue
        parent = p
        break

    if parent is None:
        return None
    sexpr: List[cin.IndexSeq] = ConvertToIndexSequences(var[parent])
    return sexpr.pop()


def IndexDefined(sexpr: cin.IndexSeq, defs: set[cin.Seq]):
    assert isinstance(sexpr, cin.IndexSeq), sexpr
    parent: Optional[cin.IndexSeq] = GetParent(sexpr)
    return any(
        [
            sexpr.index == 0,
            parent is not None and parent in defs,
            all(
                [
                    type == format.LevelType.DENSE
                    for type in sexpr.tensor.get_level_types()
                ]
            ),
        ]
    )


def IndicesDefined(expr: cin.TensorAccess, defs: set[cin.Seq]) -> bool:
    assert isinstance(expr, cin.TensorAccess)
    result = all(map(lambda x: x in defs, ConvertToIndexSequences(expr)))
    return result


@dispatch(cin.IndexExpr, set)
def Simplify(e: cin.IndexExpr, defs: set[cin.Seq]) -> Optional[cin.IndexExpr]:
    match e:
        case cin.Workspace():
            return e
        case cin.TensorAccess():
            return e if IndicesDefined(e, defs) else None
        case cin.BinaryOp():
            x: Optional[cin.IndexExpr] = Simplify(e.left, defs)
            y: Optional[cin.IndexExpr] = Simplify(e.right, defs)
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


@dispatch(cin.IndexStmt, set)
def Simplify(c: cin.IndexStmt, defs: set[cin.Seq]) -> cin.CIN:
    match c:
        case cin.ForAll():
            sexpr: cin.Seq = Simplify(c.seq, defs)
            assert not isinstance(sexpr, cin.EmptySeq | cin.FullSeq), f"{c}\n{defs}"
            return cin.ForAll(
                index_var=c.index_var,
                stmt=Simplify(c.stmt, defs | Iters(sexpr)),
                seq=sexpr,
            )
        case cin.TensorAssign():
            rhs = Simplify(c.rhs, defs)
            assert rhs is not None, f"{c}, {defs}"
            return cin.TensorAssign(lhs=c.lhs, rhs=rhs, op=c.op)
        case cin.Where():
            return cin.Where(
                producer=Simplify(c.producer, defs),
                consumer=Simplify(c.consumer, defs),
                workspace=c.workspace
            )
        case _:
            raise NotImplementedError(type(c))


@dispatch(cin.Seq, set)
def Simplify(sexpr: cin.Seq, defs: set[cin.Seq]) -> cin.Seq:
    match sexpr:
        case cin.IndexSeq(_, _, sz, _, fmt):
            return (
                sexpr
                if IndexDefined(sexpr, defs)
                else (
                    cin.FullSeq(sz)
                    if fmt == format.LevelType.DENSE
                    else cin.EmptySeq(sz)
                )
            )
        case cin.Universe(_):
            return sexpr
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
        case cin.Concatenate(s1, s2):
            x, y = Simplify(s1, defs), Simplify(s2, defs)
            if isinstance(x, cin.FullSeq | cin.EmptySeq):
                raise NotImplementedError(sexpr)  # Offset
            if isinstance(y, cin.FullSeq | cin.EmptySeq):
                raise NotImplementedError(sexpr)  # Pad
            return cin.Concatenate(x, y)
        case cin.Product(s1, s2):
            x = Simplify(s1, defs)
            if isinstance(x, cin.FullSeq) and IsDense(s2):
                return cin.FullSeq(Size(s1) * Size(s2))
            if isinstance(x, cin.FullSeq | cin.EmptySeq):
                return cin.EmptySeq(Size(s1) * Size(s2))
            newdefs = defs | Iters(x)
            y = Simplify(s2, newdefs)
            if isinstance(y, cin.FullSeq | cin.EmptySeq):
                raise ValueError(f"unexpected simplification: {sexpr} -> {x} × {y}")
            return cin.Product(x, y)
        case cin.UnionSeq(s1, s2):
            x: cin.Seq = Simplify(s1, defs)
            y: cin.Seq = Simplify(s2, defs)
            match (x, y):
                case (cin.FullSeq(), _):
                    return x
                case (_, cin.FullSeq()):
                    return y
                case (cin.EmptySeq(), cin.EmptySeq()):
                    return x
                case (cin.EmptySeq(), _):
                    return y
                case (_, cin.EmptySeq()):
                    return x
                case _:
                    return cin.UnionSeq(x, y)
        case cin.IntersectionSeq(s1, s2):
            x: cin.Seq = Simplify(s1, defs)
            y: cin.Seq = Simplify(s2, defs)
            match (x, y):
                case (cin.FullSeq(), _):
                    return x
                case (_, cin.FullSeq()):
                    return y
                case (cin.EmptySeq(), cin.EmptySeq()):
                    return x
                case (cin.EmptySeq(), _):
                    return cin.FullSeq(0) if IsDense(y) else x
                case (_, cin.EmptySeq()):
                    return cin.FullSeq(0) if IsDense(x) else y
                case _:
                    return cin.IntersectionSeq(x, y)
        case _:
            raise NotImplementedError(type(sexpr))


def Contains(sexpr: cin.Seq, subpoint: cin.Seq):
    if sexpr == subpoint:
        return True
    match sexpr:
        case cin.IndexSeq() | cin.Universe():
            return False
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return Contains(s1, subpoint) or Contains(s2, subpoint)
        case cin.Concatenate(s1, s2) | cin.Product(s1, s2):
            return Contains(s1, subpoint) or Contains(s2, subpoint)
        case cin.SliceSeq(a, _, _, _):
            return Contains(a, subpoint)
        case _:
            raise NotImplementedError(type(sexpr))


def ContainsIntersection(sexpr: cin.Seq):
    match sexpr:
        case cin.IndexSeq() | cin.Universe():
            return False
        case cin.UnionSeq(s1, s2) | cin.Concatenate(s1, s2) | cin.Product(s1, s2):
            return ContainsIntersection(s1) or ContainsIntersection(s2)
        case cin.IntersectionSeq(s1, s2):
            return True
        case cin.SliceSeq(a, _, _, _):
            return ContainsIntersection(a)
        case _:
            raise NotImplementedError(type(sexpr))


def ConstructGraph(sexpr: cin.Seq, defs: set[cin.Seq], visited: set[cin.Seq] = set()):
    if sexpr in visited or isinstance(sexpr, cin.EmptySeq | cin.FullSeq):
        return {}
    visited.update({sexpr})
    edges: set[cin.Seq] = 𝜒(sexpr)
    graph = {}
    pairs: list[Tuple[cin.Seq, cin.Seq]] = []
    for e in edges:
        r: cin.Seq = Remove(sexpr, e)
        s: cin.Seq = Simplify(r, defs)
        graph |= ConstructGraph(s, defs, visited)
        pairs.append((e, s))
    graph[sexpr] = {(e, s) for e, s in pairs}
    return graph


def ConstructIterationLattice(top: cin.Seq, defs: set[cin.Seq]):
    return IterationLattice(top, ConstructGraph(top, defs))


def Iters(sexpr: cin.Seq):
    match sexpr:
        case cin.IndexSeq():
            return {sexpr}
        case cin.Universe():
            return set()
        case cin.SliceSeq(a):
            return Iters(a)
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return Iters(s1) | Iters(s2)
        case cin.Concatenate(s1, s2) | cin.Product(s1, s2):
            return Iters(s1) | Iters(s2)
        case _:
            raise NotImplementedError(type(sexpr))


def Subpoints(lattice: IterationLattice, point: cin.Seq):
    """Returns sub-points for `point` in the lattice."""
    subs = TopologicalSortRec(point, lattice.graph, visited=set())
    iters = Iters(point)
    n = list(filter(lambda x: Iters(x) <= iters, subs))
    return n[::-1]
