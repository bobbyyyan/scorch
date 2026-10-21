"""Pure iteration-domain / merge-lattice analysis over normalized CIN.

This module owns the Phase-5 separation of iteration-domain analysis from
lowering: it consumes one normalized, family-shaped CIN program plus its
verified :class:`~scorch.compiler.loop_plan.LoopPlan` and returns an
immutable per-loop-variable domain table that fully determines the LoopIR
iteration structure (dense loops, single sparse cursors, and structured
UNION/INTERSECTION merges).  ``lower_cin`` materializes LoopIR mechanically
from this table.

Discipline (the properties the legacy ``IterationLattice`` lacked):

- the analysis is a pure function of the CIN expression tree and the plan;
  it never calls back into ``CINLowerer``, never mutates CIN or any phase
  state, and never inspects rendered names or generated C++;
- results are immutable, ID-keyed dataclasses;
- the merge lattice is stated as two total combination rules over operand
  domains (``union`` for ADD/SUB, ``intersection`` for MUL), and every
  combination outside the migrated families fails closed with a stable
  :class:`IterationDomainDefect` code instead of degrading.

Combination rules, per loop variable:

- an access contributes DENSE if the level storing the variable is dense,
  or a single sparse cursor for a compressed level; expressions invariant
  in the variable contribute nothing;
- ``intersection``: an absent or dense operand defers to the other side
  (dense operands are coordinate-loadable inside any sparse domain);
  sparse operands intersect into a merge;
- ``union``: dense-with-dense stays dense; sparse-with-sparse unions into
  a merge; a union of a sparse domain with a dense operand, or with an
  invariant operand that stores nothing sparse, visits every coordinate of
  the dimension and follows the sparse streams in order (DENSE_UNION); a
  union with an invariant operand that stores sparse structure fails closed
  (``unsupported_union_operand``), because the coordinates it covers depend
  on its alignment in an enclosing loop;
- subtraction across any sparse operand fails closed
  (``unsupported_sparse_subtraction``): its one-sided cases need negation,
  which the migrated families do not represent (the legacy lattice cannot
  compile SUB at all — recorded errata);
- merges nested inside further unions, and unions nested inside
  intersections of further sparse operands, fail closed
  (``unsupported_nested_merge``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import List, NoReturn, Optional, Tuple

from ...format import LevelType
from ..cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexStmt,
    Operation,
    TensorAccess,
    TensorAssign,
    _is_index_stmt_instance,
)
from ..cin_analysis import verify_cin_structure
from ..diagnostics import VerificationError
from ..identity import IndexId, SymbolId
from ..loop_plan import LoopPlan


@dataclass(frozen=True)
class IterationDomainDefect:
    """One immutable analysis failure: stable code and message."""

    code: str
    message: str


class IterationDomainError(Exception):
    """The CIN program is outside the analyzable iteration-domain families."""

    def __init__(self, defect: IterationDomainDefect) -> None:
        super().__init__(f"{defect.code}: {defect.message}")
        self.defect = defect


def _fail(code: str, message: str) -> NoReturn:
    raise IterationDomainError(IterationDomainDefect(code, message))


@unique
class DomainKind(Enum):
    """The iteration-domain classification of one loop variable."""

    DENSE = "dense"
    SPARSE = "sparse"
    UNION = "union"
    INTERSECTION = "intersection"
    # Every coordinate of the dimension, with the cursors' stored streams
    # followed in order: a sum one of whose terms is present at every
    # coordinate (a dense operand, or an invariant operand that stores
    # nothing sparse) beside sparse terms.
    DENSE_UNION = "dense_union"


@dataclass(frozen=True)
class SparseLevelRef:
    """One compressed or coordinate level participating in a loop's domain."""

    tensor: SymbolId
    level: int


@dataclass(frozen=True)
class LoopIterationDomain:
    """The resolved iteration domain of one loop variable."""

    index: IndexId
    kind: DomainKind
    cursors: Tuple[SparseLevelRef, ...]


@dataclass(frozen=True)
class IterationDomainTable:
    """Immutable per-loop-variable domains, in verified plan order."""

    domains: Tuple[LoopIterationDomain, ...]

    def domain(self, index: IndexId) -> LoopIterationDomain:
        for domain in self.domains:
            if domain.index == index:
                return domain
        raise KeyError(index)


@dataclass(frozen=True)
class _OperandDomain:
    """One operand's contribution: absent, dense, or sparse-structured."""

    kind: Optional[DomainKind]
    cursors: Tuple[SparseLevelRef, ...]


_ABSENT = _OperandDomain(None, ())
_DENSE = _OperandDomain(DomainKind.DENSE, ())


def _merge_cursors(
    left: Tuple[SparseLevelRef, ...], right: Tuple[SparseLevelRef, ...]
) -> Tuple[SparseLevelRef, ...]:
    merged: List[SparseLevelRef] = list(left)
    for cursor in right:
        if cursor not in merged:
            merged.append(cursor)
    return tuple(merged)


def _merged_domain(
    kind: DomainKind,
    left: Tuple[SparseLevelRef, ...],
    right: Tuple[SparseLevelRef, ...],
) -> _OperandDomain:
    """Canonicalize one binary sparse-support combination.

    Repeated uses of one sparse operand still have one support domain.  Keep
    this pure analysis independent of the current kernel ABI's repeated-input
    restriction and never publish an unmaterializable one-cursor merge.
    """

    cursors = _merge_cursors(left, right)
    if len(cursors) == 1:
        return _OperandDomain(DomainKind.SPARSE, cursors)
    return _OperandDomain(kind, cursors)


def _has_cursors(domain: _OperandDomain) -> bool:
    return domain.kind in (
        DomainKind.SPARSE,
        DomainKind.UNION,
        DomainKind.INTERSECTION,
        DomainKind.DENSE_UNION,
    )


def _dense_union(cursors: Tuple[SparseLevelRef, ...]) -> _OperandDomain:
    return _OperandDomain(DomainKind.DENSE_UNION, cursors)


def _unite(
    left: _OperandDomain,
    right: _OperandDomain,
    *,
    absent_side_is_total: bool = False,
) -> _OperandDomain:
    """The domain of ``left + right``.

    ``absent_side_is_total`` says, when one side is invariant in the loop
    variable, whether that side is present at every coordinate: it stores
    nothing sparse, so its value exists whatever the enclosing loops bind.
    Such a side makes every coordinate a candidate, exactly like a dense
    operand.  An invariant side that stores sparse structure is present only
    where its own cursors aligned in an enclosing loop, so the candidate set
    would change from one enclosing coordinate to the next; that stays
    refused.
    """

    if left.kind is None and right.kind is None:
        return _ABSENT
    if left.kind is None or right.kind is None:
        present = right if left.kind is None else left
        if present.kind is DomainKind.DENSE:
            return _DENSE
        if absent_side_is_total and present.kind in (
            DomainKind.SPARSE,
            DomainKind.UNION,
            DomainKind.DENSE_UNION,
        ):
            return _dense_union(present.cursors)
        _fail(
            "unsupported_union_operand",
            "a union of a sparse domain with an operand invariant in the "
            "loop variable that stores sparse structure: the coordinates the "
            "invariant operand covers depend on whether its own cursors "
            "aligned in an enclosing loop",
        )
    if left.kind is DomainKind.INTERSECTION or right.kind is DomainKind.INTERSECTION:
        _fail(
            "unsupported_nested_merge",
            "an intersection nested inside a union is outside the migrated " "families",
        )
    if left.kind is DomainKind.DENSE and right.kind is DomainKind.DENSE:
        return _DENSE
    if DomainKind.DENSE in (left.kind, right.kind) or DomainKind.DENSE_UNION in (
        left.kind,
        right.kind,
    ):
        # A dense operand stores every coordinate, so the union visits all
        # of them and follows the sparse operands' streams in order.
        return _dense_union(_merge_cursors(left.cursors, right.cursors))
    return _merged_domain(DomainKind.UNION, left.cursors, right.cursors)


def _intersect(left: _OperandDomain, right: _OperandDomain) -> _OperandDomain:
    if left.kind is None:
        return right
    if right.kind is None:
        return left
    if left.kind is DomainKind.DENSE:
        return right
    if right.kind is DomainKind.DENSE:
        return left
    if left.kind is DomainKind.DENSE_UNION and right.kind is DomainKind.DENSE_UNION:
        # Both factors are present at every coordinate, so their product is
        # too; the loop follows both factors' streams.
        return _dense_union(_merge_cursors(left.cursors, right.cursors))
    if DomainKind.UNION in (left.kind, right.kind) or DomainKind.DENSE_UNION in (
        left.kind,
        right.kind,
    ):
        _fail(
            "unsupported_nested_merge",
            "a union nested inside an intersection of further sparse "
            "operands is outside the migrated families",
        )
    return _merged_domain(DomainKind.INTERSECTION, left.cursors, right.cursors)


def _stores_nothing_sparse(expr: object) -> bool:
    """Whether every tensor the expression reads is stored all-dense."""

    if type(expr) is TensorAccess:
        tensor_format = expr.tensor.format
        return tensor_format is not None and all(
            level_type is LevelType.DENSE
            for level_type in tensor_format.get_level_types()
        )
    if type(expr) is CINBinaryOp:
        return _stores_nothing_sparse(expr.left) and _stores_nothing_sparse(expr.right)
    return False


def _access_domain(access: TensorAccess, index: IndexId) -> _OperandDomain:
    index_ids = tuple(access.index_ids)
    occurrences = [
        position for position, bound in enumerate(index_ids) if bound == index
    ]
    if not occurrences:
        return _ABSENT
    if len(occurrences) > 1:
        _fail(
            "unsupported_repeated_access_index",
            f"tensor {access.tensor.name!r} repeats a loop variable within "
            "one access",
        )
    tensor = access.tensor
    tensor_format = tensor.format
    if tensor_format is None:
        _fail(
            "unsupported_format",
            f"tensor {tensor.name!r} declares no format",
        )
    level_types = tuple(tensor_format.get_level_types())
    rank = len(level_types)
    mode_order = tensor.mode_order
    if mode_order is None:
        storage_modes = tuple(range(rank))
    else:
        entries = list(mode_order)
        if (
            len(entries) != rank
            or any(type(entry) is not int for entry in entries)
            or sorted(entries) != list(range(rank))
        ):
            _fail(
                "unsupported_mode_order",
                f"tensor {tensor.name!r} declares a mode order that is not "
                f"a permutation of its rank-{rank} logical modes",
            )
        storage_modes = tuple(entries)
    if storage_modes != tuple(range(rank)) and any(
        level_type is not LevelType.DENSE for level_type in level_types
    ):
        _fail(
            "unsupported_mode_order",
            f"tensor {tensor.name!r} permutes compressed structure, which "
            "stays outside the migrated families",
        )
    # The occurrence is a logical axis; the level that stores it is the
    # physical position of that axis under the tensor's mode order.
    level = storage_modes.index(occurrences[0])
    level_type = level_types[level]
    if level_type is LevelType.DENSE:
        return _DENSE
    if level_type is LevelType.COMPRESSED:
        return _OperandDomain(
            DomainKind.SPARSE,
            (SparseLevelRef(tensor.symbol_id, level),),
        )
    if level_type is LevelType.COORDINATE:
        if any(other is not LevelType.COORDINATE for other in level_types):
            # The first coordinate level of a mixed hierarchy has no parent
            # segment: a coordinate level stores one coordinate per entry
            # and no position array.  The legacy lowering refuses the same
            # tensors under the same code.
            _fail(
                "unsupported_mixed_coordinate_hierarchy",
                f"tensor {tensor.name!r} mixes a coordinate level with a "
                "level of another kind",
            )
        # A COO tensor is iterated level by level like compressed
        # structure: the stored coordinates of one level inside the run its
        # parent selects.
        return _OperandDomain(
            DomainKind.SPARSE,
            (SparseLevelRef(tensor.symbol_id, level),),
        )
    _fail(
        "unsupported_level_type",
        f"tensor {tensor.name!r} level {level} is {level_type}, which is "
        "outside the migrated DENSE/COMPRESSED families",
    )
    raise AssertionError("unreachable")


def _expr_domain(expr: object, index: IndexId) -> _OperandDomain:
    if type(expr) is TensorAccess:
        return _access_domain(expr, index)
    if type(expr) is CINBinaryOp:
        left = _expr_domain(expr.left, index)
        right = _expr_domain(expr.right, index)
        if expr.op is Operation.MUL:
            return _intersect(left, right)
        if expr.op in (Operation.ADD, Operation.SUB):
            if expr.op is Operation.SUB and (_has_cursors(left) or _has_cursors(right)):
                _fail(
                    "unsupported_sparse_subtraction",
                    "subtraction over sparse operands needs negated "
                    "one-sided cases, which the migrated families do not "
                    "represent",
                )
            invariant_side = (
                expr.left
                if left.kind is None
                else expr.right if right.kind is None else None
            )
            return _unite(
                left,
                right,
                absent_side_is_total=invariant_side is not None
                and _stores_nothing_sparse(invariant_side),
            )
        _fail(
            "unsupported_operation",
            f"binary operation {expr.op!r} is outside the migrated families",
        )
    _fail(
        "unsupported_expression",
        f"expression {type(expr).__name__} is outside the migrated families",
    )
    raise AssertionError("unreachable")


def analyze_iteration_domains(cin: IndexStmt, plan: LoopPlan) -> IterationDomainTable:
    """Classify every plan loop variable's iteration domain, purely.

    ``cin`` must be a normalized ``ForAll`` nest over one ``TensorAssign``
    and ``plan`` its verified loop plan; loop variables absent from the
    right-hand side (broadcast variables) resolve to DENSE domains driven
    by the result.
    """

    if not _is_index_stmt_instance(cin):
        raise TypeError("analyze_iteration_domains expects an IndexStmt")
    if type(plan) is not LoopPlan:
        raise TypeError("analyze_iteration_domains expects a LoopPlan")
    try:
        # This public analysis entry is callable independently of the normal
        # CIN-to-LoopIR pipeline.  Bound its recursive expression walk with
        # the same exact-type, cycle, alias, and depth preflight as every
        # other normalized-CIN consumer.
        verify_cin_structure(cin)
    except VerificationError as error:
        raise IterationDomainError(
            IterationDomainDefect(
                "malformed_cin",
                "iteration-domain analysis requires structurally valid "
                "normalized CIN",
            )
        ) from error
    loop_ids: List[IndexId] = []
    seen_nest_frames: set[int] = set()
    current: IndexStmt = cin
    while type(current) is ForAll:
        if id(current) in seen_nest_frames:
            _fail(
                "unsupported_statement",
                "the ForAll nest is cyclic",
            )
        seen_nest_frames.add(id(current))
        loop_ids.append(current.index_var.index_id)
        current = current.stmt
    if type(current) is not TensorAssign:
        _fail(
            "unsupported_statement",
            f"expected a TensorAssign at the nest leaf, got "
            f"{type(current).__name__}",
        )
    if tuple(plan.loop_order) != tuple(loop_ids):
        _fail(
            "plan_mismatch",
            "the verified plan loop order does not match the CIN nest",
        )
    domains: List[LoopIterationDomain] = []
    for index in loop_ids:
        operand = _expr_domain(current.rhs, index)
        if operand.kind is None or operand.kind is DomainKind.DENSE:
            domains.append(LoopIterationDomain(index, DomainKind.DENSE, ()))
            continue
        domains.append(LoopIterationDomain(index, operand.kind, operand.cursors))
    return IterationDomainTable(tuple(domains))
