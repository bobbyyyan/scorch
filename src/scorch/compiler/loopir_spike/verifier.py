"""Fail-closed structural verifier for the Phase-3.5 LoopIR spike.

``verify_program`` is the single authority over spike-program validity.  It
raises :class:`LoopIRVerificationError` carrying a stable defect code and the
lexical path of the offending node on the first defect found.  Constructors
perform no validation, so every boundary here is checked from stored state
with exact types: unknown node subclasses, non-tuple children, forged enum
lookalikes, aliased or cyclic structure, and excessive nesting all fail
closed rather than being coerced or skipped.

Beyond structure and lexical scope the verifier states three families of
invariants locally:

- **Sparse parent/child dominance.**  Every sparse cursor and dense-position
  expression names its dominating parent position explicitly; the checker
  requires that parent to be position-typed with the (tensor, level - 1)
  linkage of the level being entered (the root position for level 0), so a
  compressed child cannot be reached without a dominating parent position in
  scope.
- **Coordinate domains.**  Every bound coordinate carries the logical
  dimension it indexes; loads, stores, appends, dense positions, and merges
  reject coordinates from a different domain, and a merge additionally
  requires all cursor levels to store one shared dimension.
- **Value ownership.**  Only a cursor over the value-bearing leaf level of
  its tensor may expose a scalar ``CursorValue``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Tuple

from ..identity import IndexId, SymbolId
from .nodes import (
    REDUCE_IDENTITIES,
    Accumulate,
    AccumValue,
    AppendEntry,
    BinaryExpr,
    BinaryOp,
    Block,
    CursorId,
    CursorValue,
    DeclAccum,
    DenseFor,
    DensePosition,
    DimensionDecl,
    DimensionId,
    Expr,
    FloatConst,
    IndexValue,
    IntConst,
    LevelDecl,
    LevelKind,
    Load,
    LoopNodeId,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    PositionId,
    PositionValue,
    ReduceOp,
    RootPosition,
    SparseCursorDecl,
    SparseFor,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
)

MAX_NESTING_DEPTH = 64
_MISSING = object()

_EXECUTABLE_LEVEL_KINDS = (LevelKind.DENSE, LevelKind.COMPRESSED)


@dataclass(frozen=True)
class LoopIRDefect:
    """One immutable verification failure: stable code, path, and message."""

    code: str
    path: str
    message: str


class LoopIRVerificationError(Exception):
    """A spike program violated a structural invariant."""

    def __init__(self, defect: LoopIRDefect) -> None:
        super().__init__(f"{defect.code} at {defect.path}: {defect.message}")
        self.defect = defect


class _ExprType:
    """Base of the verifier's expression types."""


@dataclass(frozen=True)
class _CoordType(_ExprType):
    """Coordinate-typed; ``dimension`` is None for domain-free literals."""

    dimension: Optional[DimensionId]


@dataclass(frozen=True)
class _ScalarType(_ExprType):
    """Value-typed (a stored or computed scalar)."""


@dataclass(frozen=True)
class _PositionType(_ExprType):
    """Position-typed; ``tensor`` is None only for the root position."""

    tensor: Optional[SymbolId]
    level: int


_VALUE = _ScalarType()
_ROOT_POSITION = _PositionType(None, -1)


def _fail(code: str, path: str, message: str) -> NoReturn:
    raise LoopIRVerificationError(LoopIRDefect(code, path, message))


class _Context:
    """Mutable walk state: registries, scopes, and traversal guards."""

    def __init__(self) -> None:
        self.dimensions: Dict[DimensionId, DimensionDecl] = {}
        self.tensors: Dict[SymbolId, TensorDecl] = {}
        self.inputs: Set[SymbolId] = set()
        self.outputs: Set[SymbolId] = set()
        self.written_outputs: Set[SymbolId] = set()
        self.bound_indices: Dict[IndexId, DimensionId] = {}
        self.ever_bound_indices: Set[IndexId] = set()
        self.bound_positions: Dict[PositionId, Tuple[SymbolId, int]] = {}
        self.ever_bound_positions: Set[PositionId] = set()
        self.cursors: Dict[CursorId, Tuple[SparseCursorDecl, Optional[MergeMode]]] = {}
        self.ever_cursor_ids: Set[CursorId] = set()
        self.accums: Dict[SymbolId, ReduceOp] = {}
        self.ever_accums: Set[SymbolId] = set()
        self.seen_node_ids: Set[LoopNodeId] = set()
        self.visited_objects: Set[int] = set()
        self.path_objects: Set[int] = set()
        self.in_cursor_default = False

    def level_dimension(self, tensor: SymbolId, level: int) -> DimensionId:
        """The logical dimension stored by one validated tensor level."""

        decl = self.tensors[tensor]
        return decl.dimensions[decl.levels[level].mode]

    def dimension_name(self, dimension: DimensionId) -> str:
        decl = self.dimensions.get(dimension)
        return decl.name if decl is not None else f"<dimension {dimension.value}>"


def _check_node_id(node_id: object, path: str) -> LoopNodeId:
    if (
        type(node_id) is not LoopNodeId
        or type(getattr(node_id, "value", _MISSING)) is not int
    ):
        _fail("invalid_node_id", path, "node_id must be an int-valued LoopNodeId")
    return node_id


def _check_symbol_id(value: object, path: str, what: str) -> SymbolId:
    if (
        type(value) is not SymbolId
        or type(getattr(value, "value", _MISSING)) is not int
    ):
        _fail("invalid_symbol_id", path, f"{what} must be an int-valued SymbolId")
    return value


def _check_index_id(value: object, path: str, what: str) -> IndexId:
    if type(value) is not IndexId or type(getattr(value, "value", _MISSING)) is not int:
        _fail("invalid_index_id", path, f"{what} must be an int-valued IndexId")
    return value


def _check_cursor_id(value: object, path: str) -> CursorId:
    if (
        type(value) is not CursorId
        or type(getattr(value, "value", _MISSING)) is not int
    ):
        _fail("invalid_cursor_id", path, "cursor must be an int-valued CursorId")
    return value


def _check_dimension_id(value: object, path: str, what: str) -> DimensionId:
    if (
        type(value) is not DimensionId
        or type(getattr(value, "value", _MISSING)) is not int
    ):
        _fail(
            "invalid_dimension_id",
            path,
            f"{what} must be an int-valued DimensionId",
        )
    return value


def _check_position_id(value: object, path: str, what: str) -> PositionId:
    if (
        type(value) is not PositionId
        or type(getattr(value, "value", _MISSING)) is not int
    ):
        _fail(
            "invalid_position_id",
            path,
            f"{what} must be an int-valued PositionId",
        )
    return value


def _check_stored_fields(node: object, path: str) -> None:
    """Reject a forged dataclass before any checker reads a missing field."""

    state = getattr(node, "__dict__", None)
    if type(state) is not dict:
        _fail("malformed_state", path, "node must own dataclass field state")
    for field in fields(type(node)):  # type: ignore[arg-type]
        if field.name not in state:
            _fail(
                "malformed_state",
                f"{path}.{field.name}",
                f"stored field {field.name!r} is missing",
            )


def _enter(ctx: _Context, node: object, path: str, depth: int) -> None:
    """Aliasing, cycle, uniqueness, and depth guards for one node object."""

    _check_stored_fields(node, path)
    marker = id(node)
    if marker in ctx.path_objects:
        _fail("cyclic_structure", path, "node is its own ancestor")
    if marker in ctx.visited_objects:
        _fail("shared_node", path, "node object appears more than once")
    if depth > MAX_NESTING_DEPTH:
        _fail(
            "excessive_depth",
            path,
            f"nesting exceeds the {MAX_NESTING_DEPTH}-level verifier bound",
        )
    ctx.path_objects.add(marker)
    ctx.visited_objects.add(marker)
    node_id = _check_node_id(getattr(node, "node_id", None), path)
    if node_id in ctx.seen_node_ids:
        _fail("duplicate_node_id", path, f"node_id {node_id.value} reused")
    ctx.seen_node_ids.add(node_id)


def _leave(ctx: _Context, node: object) -> None:
    ctx.path_objects.discard(id(node))


def _require_value(kind: _ExprType, path: str, what: str) -> None:
    if type(kind) is not _ScalarType:
        _fail("type_mismatch", path, f"{what} must be value-typed")


def _require_coord(
    ctx: _Context,
    kind: _ExprType,
    path: str,
    what: str,
    expected: DimensionId,
) -> None:
    if type(kind) is not _CoordType:
        _fail("type_mismatch", path, f"{what} must be coordinate-typed")
    if kind.dimension is not None and kind.dimension != expected:
        _fail(
            "domain_mismatch",
            path,
            f"{what} is a coordinate of dimension "
            f"{ctx.dimension_name(kind.dimension)!r} but dimension "
            f"{ctx.dimension_name(expected)!r} is required",
        )


def _check_expr(ctx: _Context, expr: object, path: str, depth: int) -> _ExprType:
    if not isinstance(expr, Expr):
        _fail("unknown_expr", path, f"expected an Expr node, got {type(expr).__name__}")
    kind = type(expr)
    checker = _EXPR_CHECKERS.get(kind)
    if checker is None:
        _fail("unknown_expr", path, f"unregistered Expr subclass {kind.__name__}")
    _enter(ctx, expr, path, depth)
    try:
        return checker(ctx, expr, path, depth)
    finally:
        _leave(ctx, expr)


def _check_parent_position(
    ctx: _Context,
    parent: object,
    path: str,
    depth: int,
    tensor: SymbolId,
    level: int,
) -> None:
    """The dominance rule: a level's parent is the position one level up."""

    kind = _check_expr(ctx, parent, path, depth)
    if type(kind) is not _PositionType:
        _fail(
            "parent_position_mismatch",
            path,
            "the parent must be a physical position expression",
        )
    if level == 0:
        if kind.tensor is not None:
            _fail(
                "parent_position_mismatch",
                path,
                "a level-0 parent must be the root position",
            )
    elif kind.tensor != tensor or kind.level != level - 1:
        _fail(
            "parent_position_mismatch",
            path,
            f"level {level} needs the dominating position of level "
            f"{level - 1} of the same tensor",
        )


def _check_int_const(ctx: _Context, expr: IntConst, path: str, depth: int) -> _ExprType:
    if type(expr.value) is not int:
        _fail("malformed_state", path, "IntConst.value must be an exact int")
    return _CoordType(None)


def _check_float_const(
    ctx: _Context, expr: FloatConst, path: str, depth: int
) -> _ExprType:
    if type(expr.value) is not float:
        _fail("malformed_state", path, "FloatConst.value must be an exact float")
    return _VALUE


def _check_index_value(
    ctx: _Context, expr: IndexValue, path: str, depth: int
) -> _ExprType:
    index = _check_index_id(expr.index, path, "IndexValue.index")
    if index not in ctx.bound_indices:
        _fail("unbound_index", path, f"index {index.value} is not bound in scope")
    return _CoordType(ctx.bound_indices[index])


def _check_root_position(
    ctx: _Context, expr: RootPosition, path: str, depth: int
) -> _ExprType:
    return _ROOT_POSITION


def _check_dense_position(
    ctx: _Context, expr: DensePosition, path: str, depth: int
) -> _ExprType:
    tensor = _check_symbol_id(expr.tensor, path, "DensePosition.tensor")
    if tensor not in ctx.tensors:
        _fail(
            "undefined_tensor",
            path,
            "DensePosition references an undeclared tensor",
        )
    if tensor not in ctx.inputs:
        _fail("output_read", path, "positions are only formed on declared inputs")
    decl = ctx.tensors[tensor]
    if type(expr.level) is not int:
        _fail("malformed_state", path, "DensePosition.level must be an exact int")
    if not 0 <= expr.level < len(decl.levels):
        _fail(
            "rank_mismatch",
            path,
            f"level {expr.level} outside rank-{len(decl.levels)} tensor",
        )
    if decl.levels[expr.level].kind is not LevelKind.DENSE:
        _fail(
            "layout_mismatch",
            path,
            "dense positions are only defined on DENSE levels",
        )
    _check_parent_position(
        ctx, expr.parent, f"{path}.parent", depth + 1, tensor, expr.level
    )
    coord_type = _check_expr(ctx, expr.coord, f"{path}.coord", depth + 1)
    _require_coord(
        ctx,
        coord_type,
        f"{path}.coord",
        "the dense-level coordinate",
        ctx.level_dimension(tensor, expr.level),
    )
    return _PositionType(tensor, expr.level)


def _check_position_value(
    ctx: _Context, expr: PositionValue, path: str, depth: int
) -> _ExprType:
    position = _check_position_id(expr.position, path, "PositionValue.position")
    if position not in ctx.bound_positions:
        _fail(
            "unbound_position",
            path,
            f"position {position.value} is not bound in scope",
        )
    tensor, level = ctx.bound_positions[position]
    return _PositionType(tensor, level)


def _check_cursor_value(
    ctx: _Context, expr: CursorValue, path: str, depth: int
) -> _ExprType:
    if ctx.in_cursor_default:
        _fail(
            "default_contains_cursor",
            path,
            "a CursorValue default must not read another cursor",
        )
    cursor = _check_cursor_id(expr.cursor, path)
    if cursor not in ctx.cursors:
        _fail("unbound_cursor", path, f"cursor {cursor.value} is not in scope")
    decl, mode = ctx.cursors[cursor]
    if decl.level != len(ctx.tensors[decl.tensor].levels) - 1:
        _fail(
            "non_leaf_value",
            path,
            "only the value-bearing leaf level owns scalar values; "
            f"level {decl.level} is structural",
        )
    if mode is MergeMode.UNION:
        if expr.default is None:
            _fail(
                "missing_union_default",
                path,
                "a UNION-merged cursor read requires a default value",
            )
        ctx.in_cursor_default = True
        try:
            default_type = _check_expr(ctx, expr.default, f"{path}.default", depth + 1)
        finally:
            ctx.in_cursor_default = False
        if type(default_type) is not _ScalarType:
            _fail("type_mismatch", f"{path}.default", "default must be value-typed")
    else:
        if expr.default is not None:
            _fail(
                "dead_default",
                path,
                "a default is unobservable outside a UNION merge",
            )
    return _VALUE


def _check_accum_value(
    ctx: _Context, expr: AccumValue, path: str, depth: int
) -> _ExprType:
    accumulator = _check_symbol_id(expr.accumulator, path, "AccumValue.accumulator")
    if accumulator not in ctx.accums:
        _fail(
            "undefined_accumulator",
            path,
            "accumulator is not declared and live in an enclosing scope",
        )
    return _VALUE


def _check_load(ctx: _Context, expr: Load, path: str, depth: int) -> _ExprType:
    tensor = _check_symbol_id(expr.tensor, path, "Load.tensor")
    if tensor not in ctx.tensors:
        _fail("undefined_tensor", path, "Load references an undeclared tensor")
    if tensor not in ctx.inputs:
        _fail("output_read", path, "Load may only read declared inputs")
    decl = ctx.tensors[tensor]
    if any(level.kind is not LevelKind.DENSE for level in decl.levels):
        _fail(
            "layout_mismatch",
            path,
            "coordinate loads are only defined on all-dense tensors",
        )
    if type(expr.indices) is not tuple:
        _fail("malformed_state", path, "Load.indices must be an owned tuple")
    if len(expr.indices) != len(decl.levels):
        _fail(
            "rank_mismatch",
            path,
            f"{len(expr.indices)} indices for rank-{len(decl.levels)} tensor",
        )
    for position, index in enumerate(expr.indices):
        index_type = _check_expr(ctx, index, f"{path}.indices[{position}]", depth + 1)
        _require_coord(
            ctx,
            index_type,
            f"{path}.indices[{position}]",
            "a load index",
            decl.dimensions[position],
        )
    return _VALUE


def _check_binary_expr(
    ctx: _Context, expr: BinaryExpr, path: str, depth: int
) -> _ExprType:
    if type(expr.op) is not BinaryOp:
        _fail("malformed_state", path, "BinaryExpr.op must be a BinaryOp member")
    for name, operand in (("lhs", expr.lhs), ("rhs", expr.rhs)):
        operand_type = _check_expr(ctx, operand, f"{path}.{name}", depth + 1)
        _require_value(operand_type, f"{path}.{name}", "a binary operand")
    return _VALUE


_EXPR_CHECKERS: Dict[type, Callable[[_Context, Any, str, int], _ExprType]] = {
    IntConst: _check_int_const,
    FloatConst: _check_float_const,
    IndexValue: _check_index_value,
    RootPosition: _check_root_position,
    DensePosition: _check_dense_position,
    PositionValue: _check_position_value,
    CursorValue: _check_cursor_value,
    AccumValue: _check_accum_value,
    Load: _check_load,
    BinaryExpr: _check_binary_expr,
}


def _bind_index(
    ctx: _Context,
    index: object,
    path: str,
    what: str,
    dimension: DimensionId,
) -> IndexId:
    bound = _check_index_id(index, path, what)
    if bound in ctx.ever_bound_indices:
        _fail(
            "duplicate_index_binding",
            path,
            f"index {bound.value} is bound more than once in the program",
        )
    ctx.ever_bound_indices.add(bound)
    ctx.bound_indices[bound] = dimension
    return bound


def _bind_position(
    ctx: _Context,
    position: object,
    path: str,
    what: str,
    tensor: SymbolId,
    level: int,
) -> PositionId:
    bound = _check_position_id(position, path, what)
    if bound in ctx.ever_bound_positions:
        _fail(
            "duplicate_position_binding",
            path,
            f"position {bound.value} is bound more than once in the program",
        )
    ctx.ever_bound_positions.add(bound)
    ctx.bound_positions[bound] = (tensor, level)
    return bound


def _check_cursor_decl(
    ctx: _Context, decl: object, path: str, depth: int
) -> SparseCursorDecl:
    if type(decl) is not SparseCursorDecl:
        _fail(
            "malformed_state",
            path,
            f"expected a SparseCursorDecl, got {type(decl).__name__}",
        )
    _enter(ctx, decl, path, depth)
    try:
        cursor = _check_cursor_id(decl.cursor, path)
        if cursor in ctx.ever_cursor_ids:
            _fail("duplicate_cursor_id", path, f"cursor id {cursor.value} reused")
        ctx.ever_cursor_ids.add(cursor)
        tensor = _check_symbol_id(decl.tensor, path, "SparseCursorDecl.tensor")
        if tensor not in ctx.tensors:
            _fail("undefined_tensor", path, "cursor references an undeclared tensor")
        if tensor not in ctx.inputs:
            _fail("output_read", path, "cursors may only walk declared inputs")
        levels = ctx.tensors[tensor].levels
        if type(decl.level) is not int:
            _fail("malformed_state", path, "SparseCursorDecl.level must be an int")
        if not 0 <= decl.level < len(levels):
            _fail(
                "rank_mismatch",
                path,
                f"level {decl.level} outside rank-{len(levels)} tensor",
            )
        if levels[decl.level].kind is not LevelKind.COMPRESSED:
            _fail(
                "layout_mismatch",
                path,
                "sparse cursors are only defined on COMPRESSED levels",
            )
        _check_parent_position(
            ctx, decl.parent, f"{path}.parent", depth + 1, tensor, decl.level
        )
        return decl
    finally:
        _leave(ctx, decl)


def _check_stmt(ctx: _Context, stmt: object, path: str, depth: int) -> None:
    if not isinstance(stmt, Stmt):
        _fail("unknown_stmt", path, f"expected a Stmt node, got {type(stmt).__name__}")
    kind = type(stmt)
    checker = _STMT_CHECKERS.get(kind)
    if checker is None:
        _fail("unknown_stmt", path, f"unregistered Stmt subclass {kind.__name__}")
    _enter(ctx, stmt, path, depth)
    try:
        checker(ctx, stmt, path, depth)
    finally:
        _leave(ctx, stmt)


def _check_block(ctx: _Context, block: Block, path: str, depth: int) -> None:
    if type(block.statements) is not tuple:
        _fail("malformed_state", path, "Block.statements must be an owned tuple")
    declared_here: List[SymbolId] = []
    for position, stmt in enumerate(block.statements):
        _check_stmt(ctx, stmt, f"{path}.statements[{position}]", depth + 1)
        if isinstance(stmt, DeclAccum):
            declared_here.append(stmt.accumulator)
    for accumulator in declared_here:
        del ctx.accums[accumulator]


def _check_body(ctx: _Context, body: object, path: str, depth: int) -> None:
    """Route a loop body through the guarded statement dispatch as a Block."""

    if type(body) is not Block:
        _fail(
            "malformed_state",
            path,
            f"body must be a Block, got {type(body).__name__}",
        )
    _check_stmt(ctx, body, path, depth)


def _check_dense_for(ctx: _Context, stmt: DenseFor, path: str, depth: int) -> None:
    dimension = _check_dimension_id(
        stmt.dimension, f"{path}.dimension", "DenseFor.dimension"
    )
    if dimension not in ctx.dimensions:
        _fail(
            "undefined_dimension",
            f"{path}.dimension",
            "DenseFor iterates an undeclared dimension",
        )
    index = _bind_index(ctx, stmt.index, path, "DenseFor.index", dimension)
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
    finally:
        del ctx.bound_indices[index]


def _check_sparse_for(ctx: _Context, stmt: SparseFor, path: str, depth: int) -> None:
    decl = _check_cursor_decl(ctx, stmt.cursor, f"{path}.cursor", depth + 1)
    position = _bind_position(
        ctx, stmt.position, path, "SparseFor.position", decl.tensor, decl.level
    )
    index = _bind_index(
        ctx,
        stmt.coord_index,
        path,
        "SparseFor.coord_index",
        ctx.level_dimension(decl.tensor, decl.level),
    )
    ctx.cursors[decl.cursor] = (decl, None)
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
    finally:
        del ctx.bound_indices[index]
        del ctx.bound_positions[position]
        del ctx.cursors[decl.cursor]


def _check_merged_sparse_for(
    ctx: _Context, stmt: MergedSparseFor, path: str, depth: int
) -> None:
    if type(stmt.mode) is not MergeMode:
        _fail("malformed_state", path, "mode must be a MergeMode member")
    if type(stmt.cursors) is not tuple:
        _fail("malformed_state", path, "cursors must be an owned tuple")
    if len(stmt.cursors) < 2:
        _fail(
            "degenerate_merge",
            path,
            "a merged loop needs at least two sparse cursors",
        )
    decls: List[SparseCursorDecl] = []
    for position, cursor in enumerate(stmt.cursors):
        cursor_path = f"{path}.cursors[{position}]"
        decl = _check_cursor_decl(ctx, cursor, cursor_path, depth + 1)
        if decl.level != len(ctx.tensors[decl.tensor].levels) - 1:
            _fail(
                "unsupported_sparse_hierarchy",
                cursor_path,
                "merged cursors must target the value-bearing leaf level; "
                "hierarchical merge descent is not represented by this spike",
            )
        decls.append(decl)
    merge_dimension = ctx.level_dimension(decls[0].tensor, decls[0].level)
    for position, decl in enumerate(decls[1:], start=1):
        cursor_dimension = ctx.level_dimension(decl.tensor, decl.level)
        if cursor_dimension != merge_dimension:
            _fail(
                "merge_domain_mismatch",
                f"{path}.cursors[{position}]",
                "merged cursors must iterate one shared logical dimension; "
                f"got {ctx.dimension_name(cursor_dimension)!r} beside "
                f"{ctx.dimension_name(merge_dimension)!r}",
            )
    index = _bind_index(
        ctx, stmt.coord_index, path, "MergedSparseFor.coord_index", merge_dimension
    )
    for decl in decls:
        ctx.cursors[decl.cursor] = (decl, stmt.mode)
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
    finally:
        del ctx.bound_indices[index]
        for decl in decls:
            del ctx.cursors[decl.cursor]


def _check_decl_accum(ctx: _Context, stmt: DeclAccum, path: str, depth: int) -> None:
    accumulator = _check_symbol_id(stmt.accumulator, path, "DeclAccum.accumulator")
    if accumulator in ctx.tensors:
        _fail(
            "duplicate_symbol",
            path,
            "accumulator symbol collides with a declared tensor",
        )
    if accumulator in ctx.ever_accums:
        _fail(
            "duplicate_symbol",
            path,
            f"accumulator symbol {accumulator.value} declared more than once",
        )
    if type(stmt.op) is not ReduceOp:
        _fail("malformed_state", path, "DeclAccum.op must be a ReduceOp member")
    init = stmt.init
    identity = REDUCE_IDENTITIES[stmt.op]
    init_type = _check_expr(ctx, init, f"{path}.init", depth + 1)
    if type(init_type) is not _ScalarType or type(init) is not FloatConst:
        _fail(
            "invalid_reduction_identity",
            f"{path}.init",
            "init must be the literal identity of the reduction operator",
        )
    if init.value != identity or math.copysign(1.0, init.value) != math.copysign(
        1.0, identity
    ):
        _fail(
            "invalid_reduction_identity",
            f"{path}.init",
            f"init {init.value!r} is not the {stmt.op.value} identity {identity!r}",
        )
    ctx.ever_accums.add(accumulator)
    ctx.accums[accumulator] = stmt.op


def _check_accumulate(ctx: _Context, stmt: Accumulate, path: str, depth: int) -> None:
    accumulator = _check_symbol_id(stmt.accumulator, path, "Accumulate.accumulator")
    if accumulator not in ctx.accums:
        _fail(
            "undefined_accumulator",
            path,
            "accumulator is not declared and live in an enclosing scope",
        )
    value_type = _check_expr(ctx, stmt.value, f"{path}.value", depth + 1)
    _require_value(value_type, f"{path}.value", "an accumulated value")


def _check_dense_output_indices(
    ctx: _Context, stmt: object, tensor: SymbolId, path: str, depth: int
) -> None:
    """Shared index checks for coordinate-addressed dense-output writes."""

    decl = ctx.tensors[tensor]
    if any(level.kind is not LevelKind.DENSE for level in decl.levels):
        _fail(
            "layout_mismatch",
            path,
            "coordinate stores are only defined on all-dense outputs",
        )
    indices = stmt.indices  # type: ignore[attr-defined]
    if type(indices) is not tuple:
        _fail("malformed_state", path, "indices must be an owned tuple")
    if len(indices) != len(decl.levels):
        _fail(
            "rank_mismatch",
            path,
            f"{len(indices)} indices for rank-{len(decl.levels)} output",
        )
    for position, index in enumerate(indices):
        index_type = _check_expr(ctx, index, f"{path}.indices[{position}]", depth + 1)
        _require_coord(
            ctx,
            index_type,
            f"{path}.indices[{position}]",
            "a store index",
            decl.dimensions[position],
        )


def _check_store(ctx: _Context, stmt: Store, path: str, depth: int) -> None:
    tensor = _check_symbol_id(stmt.tensor, path, "Store.tensor")
    if tensor not in ctx.tensors:
        _fail("undefined_tensor", path, "Store references an undeclared tensor")
    if tensor not in ctx.outputs:
        _fail("output_scope", path, "Store may only write declared outputs")
    _check_dense_output_indices(ctx, stmt, tensor, path, depth)
    value_type = _check_expr(ctx, stmt.value, f"{path}.value", depth + 1)
    _require_value(value_type, f"{path}.value", "a stored value")
    ctx.written_outputs.add(tensor)


def _check_store_reduce(
    ctx: _Context, stmt: StoreReduce, path: str, depth: int
) -> None:
    tensor = _check_symbol_id(stmt.tensor, path, "StoreReduce.tensor")
    if tensor not in ctx.tensors:
        _fail("undefined_tensor", path, "StoreReduce references an undeclared tensor")
    if tensor not in ctx.outputs:
        _fail("output_scope", path, "StoreReduce may only write declared outputs")
    if type(stmt.op) is not ReduceOp:
        _fail("malformed_state", path, "StoreReduce.op must be a ReduceOp member")
    _check_dense_output_indices(ctx, stmt, tensor, path, depth)
    value_type = _check_expr(ctx, stmt.value, f"{path}.value", depth + 1)
    _require_value(value_type, f"{path}.value", "a combined value")
    ctx.written_outputs.add(tensor)


def _check_append_entry(
    ctx: _Context, stmt: AppendEntry, path: str, depth: int
) -> None:
    tensor = _check_symbol_id(stmt.tensor, path, "AppendEntry.tensor")
    if tensor not in ctx.tensors:
        _fail("undefined_tensor", path, "AppendEntry references an undeclared tensor")
    if tensor not in ctx.outputs:
        _fail("output_scope", path, "AppendEntry may only assemble declared outputs")
    decl = ctx.tensors[tensor]
    if all(level.kind is not LevelKind.COMPRESSED for level in decl.levels):
        _fail(
            "layout_mismatch",
            path,
            "appended assembly needs an output with a COMPRESSED level",
        )
    if type(stmt.coords) is not tuple:
        _fail("malformed_state", path, "AppendEntry.coords must be an owned tuple")
    if len(stmt.coords) != len(decl.levels):
        _fail(
            "rank_mismatch",
            path,
            f"{len(stmt.coords)} coordinates for rank-{len(decl.levels)} output",
        )
    for position, coord in enumerate(stmt.coords):
        coord_type = _check_expr(ctx, coord, f"{path}.coords[{position}]", depth + 1)
        _require_coord(
            ctx,
            coord_type,
            f"{path}.coords[{position}]",
            "an appended coordinate",
            decl.dimensions[position],
        )
    value_type = _check_expr(ctx, stmt.value, f"{path}.value", depth + 1)
    _require_value(value_type, f"{path}.value", "an appended value")
    ctx.written_outputs.add(tensor)


_STMT_CHECKERS: Dict[type, Callable[[_Context, Any, str, int], None]] = {
    Block: _check_block,
    DenseFor: _check_dense_for,
    SparseFor: _check_sparse_for,
    MergedSparseFor: _check_merged_sparse_for,
    DeclAccum: _check_decl_accum,
    Accumulate: _check_accumulate,
    Store: _check_store,
    StoreReduce: _check_store_reduce,
    AppendEntry: _check_append_entry,
}


def _check_dimension_decl(ctx: _Context, decl: object, path: str, depth: int) -> None:
    if type(decl) is not DimensionDecl:
        _fail(
            "malformed_state",
            path,
            f"expected a DimensionDecl, got {type(decl).__name__}",
        )
    _enter(ctx, decl, path, depth)
    try:
        dimension = _check_dimension_id(decl.dimension, path, "DimensionDecl.dimension")
        if dimension in ctx.dimensions:
            _fail(
                "duplicate_dimension",
                path,
                f"dimension {dimension.value} declared more than once",
            )
        if type(decl.name) is not str or not decl.name:
            _fail("malformed_state", path, "DimensionDecl.name must be a nonempty str")
        ctx.dimensions[dimension] = decl
    finally:
        _leave(ctx, decl)


def _check_level_decl(
    ctx: _Context, decl: object, path: str, depth: int, rank: int
) -> int:
    if type(decl) is not LevelDecl:
        _fail(
            "malformed_state",
            path,
            f"expected a LevelDecl, got {type(decl).__name__}",
        )
    _enter(ctx, decl, path, depth)
    try:
        if type(decl.kind) is not LevelKind:
            _fail("malformed_state", path, "LevelDecl.kind must be a LevelKind member")
        if decl.kind not in _EXECUTABLE_LEVEL_KINDS:
            _fail(
                "unsupported_level_kind",
                path,
                f"{decl.kind.value} levels are declared production surface; "
                "the spike fails closed on them until a later milestone "
                "represents their iteration",
            )
        if type(decl.mode) is not int:
            _fail("malformed_state", path, "LevelDecl.mode must be an exact int")
        if not 0 <= decl.mode < rank:
            _fail(
                "invalid_mode_order",
                path,
                f"mode {decl.mode} outside the rank-{rank} logical modes",
            )
        return decl.mode
    finally:
        _leave(ctx, decl)


def _check_tensor_decl(ctx: _Context, decl: object, path: str, depth: int) -> None:
    if type(decl) is not TensorDecl:
        _fail(
            "malformed_state",
            path,
            f"expected a TensorDecl, got {type(decl).__name__}",
        )
    _enter(ctx, decl, path, depth)
    try:
        symbol = _check_symbol_id(decl.symbol, path, "TensorDecl.symbol")
        if symbol in ctx.tensors:
            _fail("duplicate_symbol", path, f"tensor symbol {symbol.value} redeclared")
        if type(decl.name) is not str or not decl.name:
            _fail("malformed_state", path, "TensorDecl.name must be a nonempty str")
        if type(decl.dimensions) is not tuple or not decl.dimensions:
            _fail(
                "malformed_state",
                path,
                "TensorDecl.dimensions must be a nonempty owned tuple",
            )
        for position, dimension in enumerate(decl.dimensions):
            dimension_path = f"{path}.dimensions[{position}]"
            checked = _check_dimension_id(
                dimension, dimension_path, "a tensor dimension"
            )
            if checked not in ctx.dimensions:
                _fail(
                    "undefined_dimension",
                    dimension_path,
                    "tensor references an undeclared dimension",
                )
        if type(decl.levels) is not tuple or not decl.levels:
            _fail(
                "malformed_state",
                path,
                "TensorDecl.levels must be a nonempty owned tuple",
            )
        rank = len(decl.dimensions)
        if len(decl.levels) != rank:
            _fail(
                "rank_mismatch",
                path,
                f"{len(decl.levels)} levels for {rank} logical modes",
            )
        modes = [
            _check_level_decl(ctx, level, f"{path}.levels[{position}]", depth + 1, rank)
            for position, level in enumerate(decl.levels)
        ]
        if sorted(modes) != list(range(rank)):
            _fail(
                "invalid_mode_order",
                path,
                "level modes must be a permutation of the logical modes; "
                f"got {tuple(modes)}",
            )
        ctx.tensors[symbol] = decl
    finally:
        _leave(ctx, decl)


def verify_program(program: object) -> None:
    """Fail closed unless ``program`` is a structurally valid spike program."""

    if type(program) is not LoopProgram:
        _fail(
            "malformed_state",
            "program",
            f"expected a LoopProgram, got {type(program).__name__}",
        )
    ctx = _Context()
    _enter(ctx, program, "program", 0)
    try:
        if type(program.dimensions) is not tuple:
            _fail(
                "malformed_state",
                "program.dimensions",
                "dimensions must be an owned tuple",
            )
        for position, dimension_decl in enumerate(program.dimensions):
            _check_dimension_decl(
                ctx, dimension_decl, f"program.dimensions[{position}]", 1
            )
        if type(program.tensors) is not tuple or not program.tensors:
            _fail(
                "malformed_state",
                "program.tensors",
                "tensors must be a nonempty owned tuple",
            )
        for position, tensor_decl in enumerate(program.tensors):
            _check_tensor_decl(ctx, tensor_decl, f"program.tensors[{position}]", 1)
        for role, symbols in (("inputs", program.inputs), ("outputs", program.outputs)):
            if type(symbols) is not tuple:
                _fail(
                    "malformed_state",
                    f"program.{role}",
                    f"{role} must be an owned tuple",
                )
            for position, symbol in enumerate(symbols):
                checked = _check_symbol_id(
                    symbol, f"program.{role}[{position}]", f"{role} entry"
                )
                if checked not in ctx.tensors:
                    _fail(
                        "undefined_tensor",
                        f"program.{role}[{position}]",
                        f"{role} entry is not a declared tensor",
                    )
                if checked in ctx.inputs or checked in ctx.outputs:
                    _fail(
                        "duplicate_symbol",
                        f"program.{role}[{position}]",
                        "tensor listed twice across inputs/outputs",
                    )
                (ctx.inputs if role == "inputs" else ctx.outputs).add(checked)
        if not ctx.outputs:
            _fail("output_scope", "program.outputs", "a program needs an output")
        unassigned = set(ctx.tensors) - ctx.inputs - ctx.outputs
        if unassigned:
            _fail(
                "output_scope",
                "program.tensors",
                "every declared tensor must be an input or an output",
            )
        _check_body(ctx, program.body, "program.body", 1)
        unwritten = ctx.outputs - ctx.written_outputs
        if unwritten:
            _fail(
                "unwritten_output",
                "program.outputs",
                "an output is never stored or appended to",
            )
    finally:
        _leave(ctx, program)
