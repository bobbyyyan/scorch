from scorch.compiler import cin
from typing import List, Optional, Any, Tuple, Callable, Union, Sequence
import scorch.format as format

# Utility functions for sequence expressions (Seq).


def Size(sexpr: cin.Seq) -> int:
    """Returns the size of this sequence expression."""
    match sexpr:
        case cin.IndexSeq(_, _, size):
            return size
        case cin.FullSeq(sz) | cin.EmptySeq(sz):
            return sz
        case cin.SliceSeq(_, s, e, r):
            return (e - s) + (r - 1) // r
        case cin.ProductSeq(s1, s2):
            return Size(s1) * Size(s2)
        case cin.ConcatenateSeq(s1, s2):
            return Size(s1) + Size(s2)
        case cin.ProjectSeq(_, k, I, J):
            assert k in (0, 1)
            return I if k == 0 else J
        case _:
            return NotImplementedError(type(sexpr))


def IsDense(sexpr: cin.Seq) -> bool:
    """Returns whether this sequence expression (or any of its children) is dense."""
    match sexpr:
        case cin.IndexSeq(_, _, _, _, fmt):
            return fmt == format.LevelType.DENSE
        case cin.Universe():
            return True
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return IsDense(s1) and IsDense(s2)
        case cin.ProductSeq(s1, s2) | cin.ConcatenateSeq(s1, s2):
            return IsDense(s1) and IsDense(s2)
        case cin.SliceSeq(a) | cin.ProjectSeq(a):
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
        case cin.ProductSeq(s1, s2):
            return cin.ProductSeq(Remove(s1, sub), Remove(s2, sub))
        case cin.ConcatenateSeq(s1, s2):
            return cin.ConcatenateSeq(Remove(s1, sub), Remove(s2, sub))
        case cin.UnionSeq(s1, s2):
            return cin.UnionSeq(Remove(s1, sub), Remove(s2, sub))
        case cin.IntersectionSeq(s1, s2):
            return cin.IntersectionSeq(Remove(s1, sub), Remove(s2, sub))
        case cin.ProjectSeq(a, k, I, J):
            return cin.ProjectSeq(Remove(a, sub), k, I, J)
        case _:
            raise NotImplementedError(type(sexpr))


def IndicesDefined(expr: cin.TensorAccess, defs: set[cin.Seq]) -> bool:
    """Returns whether all the indices in `expr` are defined in `defs`."""
    assert isinstance(expr, cin.TensorAccess)
    return all(map(lambda x: x in defs, ConvertToIndexSequences(expr)))


def ConvertToIndexSequences(e: cin.TensorAccess) -> list[cin.IndexSeq]:
    """Converts the indices of a `TensorAccess` to a sequence of `IndexSeq`."""
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
    """Returns the parent of `sexpr` if it exists, and None otherwise"""
    idx: cin.IndexVar = sexpr.idx
    tensor: cin.TensorVar = sexpr.tensor

    accesses: List[cin.TensorAccess] = tensor.tensor_accesses()
    parent: Optional[cin.IndexVar] = None
    for access in accesses:
        if access.tensor != tensor:
            continue
        p: Optional[cin.IndexVar] = access.get_parent_index_var(idx)
        if p is None:
            continue
        parent = p
        break

    if parent is None:
        return None
    sexpr: List[cin.IndexSeq] = ConvertToIndexSequences(tensor[parent])
    return sexpr.pop()


def GetChild(sexpr: cin.IndexSeq) -> Optional[cin.IndexSeq]:
    """Returns the child of `sexpr` if it exists, and None otherwise."""
    idx: cin.IndexVar = sexpr.idx
    tensor: cin.TensorVar = sexpr.tensor

    accesses: List[cin.TensorAccess] = tensor.tensor_accesses()
    child: Optional[cin.IndexVar] = None
    for access in accesses:
        if access.tensor != tensor:
            continue
        c: Optional[cin.IndexVar] = access.get_child_index_var(idx)
        if c is None:
            continue

        indices: List[cin.IndexVar] = access.get_index_vars()
        levels: List[format.LevelType] = access.level_types()
        for i, (idx, level) in enumerate(zip(indices, levels)):
            if idx != c:
                continue
            return cin.IndexSeq(
                idx=c,
                tensor=access.tensor,
                size=access.tensor.shape[i],
                index=i,
                format=level,
            )

    return None


def IndexDefined(sexpr: cin.IndexSeq, defs: set[cin.Seq]):
    """Returns whether one of the following conditions hold:
    (1) this is the first index,
    (2) the parent of this sequence expression is in `defs`, or
    (3) all the levels are dense.
    """
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


def SimplifySeq(sexpr: cin.Seq, defs: set[cin.Seq]) -> cin.Seq:
    """Simplies `sexpr` and returns the simplified sequence expression."""
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
            match x := SimplifySeq(a, defs):
                case cin.FullSeq(_):
                    return cin.FullSeq(Size(sexpr))
                case cin.EmptySeq(_):
                    return cin.EmptySeq(Size(sexpr))
                case _:
                    return cin.SliceSeq(x, s, e, r)
        case cin.ConcatenateSeq(s1, s2):
            x, y = SimplifySeq(s1, defs), SimplifySeq(s2, defs)
            if isinstance(x, cin.FullSeq | cin.EmptySeq):
                raise NotImplementedError(sexpr)  # Offset
            if isinstance(y, cin.FullSeq | cin.EmptySeq):
                raise NotImplementedError(sexpr)  # Pad
            return cin.ConcatenateSeq(x, y)
        case cin.ProductSeq(s1, s2):
            x = SimplifySeq(s1, defs)
            if isinstance(x, cin.FullSeq) and IsDense(s2):
                return cin.FullSeq(Size(s1) * Size(s2))
            if isinstance(x, cin.FullSeq | cin.EmptySeq):
                return cin.EmptySeq(Size(s1) * Size(s2))
            newdefs = defs | Iters(x)
            y = SimplifySeq(s2, newdefs)
            if isinstance(y, cin.FullSeq | cin.EmptySeq):
                raise ValueError(f"unexpected simplification: {sexpr} -> {x} × {y}")
            return cin.ProductSeq(x, y)
        case cin.ProjectSeq(a, k, I, J):
            x = SimplifySeq(a, defs)
            match x:
                case cin.FullSeq(_):
                    return cin.FullSeq(Size(sexpr))
                case cin.EmptySeq(_):
                    return cin.EmptySeq(Size(sexpr))
                case _:
                    if k == 0 or (k - 1, x) in defs:
                        return cin.ProjectSeq(x, k, I, J)
                    return (
                        cin.FullSeq(Size(sexpr))
                        if IsDense(sexpr)
                        else cin.EmptySeq(Size(sexpr))
                    )
        case cin.UnionSeq(s1, s2):
            x: cin.Seq = SimplifySeq(s1, defs)
            y: cin.Seq = SimplifySeq(s2, defs)
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
            x: cin.Seq = SimplifySeq(s1, defs)
            y: cin.Seq = SimplifySeq(s2, defs)
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


def Contains(sexpr: cin.Seq, subpoint: cin.Seq) -> bool:
    """Returns whether `sexpr` is or contains `subpoint`."""
    if sexpr == subpoint:
        return True
    match sexpr:
        case cin.IndexSeq() | cin.Universe():
            return False
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return Contains(s1, subpoint) or Contains(s2, subpoint)
        case cin.ConcatenateSeq(s1, s2) | cin.ProductSeq(s1, s2):
            return Contains(s1, subpoint) or Contains(s2, subpoint)
        case cin.SliceSeq(a) | cin.ProjectSeq(a):
            return Contains(a, subpoint)
        case _:
            raise NotImplementedError(type(sexpr))


def ContainsIntersection(sexpr: cin.Seq) -> bool:
    """Returns whether sexpr is or contains an intersection sequence expression."""
    match sexpr:
        case cin.IndexSeq() | cin.Universe():
            return False
        case cin.UnionSeq(s1, s2) | cin.ConcatenateSeq(s1, s2) | cin.ProductSeq(s1, s2):
            return ContainsIntersection(s1) or ContainsIntersection(s2)
        case cin.IntersectionSeq(s1, s2):
            return True
        case cin.SliceSeq(a) | cin.ProjectSeq(a):
            return ContainsIntersection(a)
        case _:
            raise NotImplementedError(type(sexpr))


def Iters(sexpr: cin.Seq) -> set[cin.Seq]:
    """Returns the iterators of `sexpr`."""
    match sexpr:
        case cin.IndexSeq():
            return {sexpr}
        case cin.Universe():
            return set()
        case cin.SliceSeq(a):
            return Iters(a)
        case cin.UnionSeq(s1, s2) | cin.IntersectionSeq(s1, s2):
            return Iters(s1) | Iters(s2)
        case cin.ConcatenateSeq(s1, s2) | cin.ProductSeq(s1, s2):
            return Iters(s1) | Iters(s2)
        case cin.ProjectSeq(a, k):
            return {(k, a)} if k == 0 else Iters(a) | {(k, a)}
        case _:
            raise NotImplementedError(type(sexpr))
