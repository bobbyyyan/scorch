"""Fail-closed structural verifier for the Phase-3.5 LoopIR spike.

``verify_program`` is the single authority over spike-program validity.  It
raises :class:`LoopIRVerificationError` carrying a stable defect code and the
lexical path of the offending node on the first defect found.  Constructors
perform no validation, so every boundary here is checked from stored state
with exact types: unknown node subclasses, non-tuple children, forged enum
lookalikes, aliased or cyclic structure, and excessive nesting all fail
closed rather than being coerced or skipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, unique
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
    DimSize,
    Expr,
    FloatConst,
    IndexValue,
    IntConst,
    LevelKind,
    Load,
    LoopNodeId,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    ReduceOp,
    SparseCursorDecl,
    SparseFor,
    Stmt,
    Store,
    TensorDecl,
)

MAX_NESTING_DEPTH = 64


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


@unique
class _ValueType(Enum):
    COORD = "coord"
    VALUE = "value"


def _fail(code: str, path: str, message: str) -> NoReturn:
    raise LoopIRVerificationError(LoopIRDefect(code, path, message))


class _Context:
    """Mutable walk state: registries, scopes, and traversal guards."""

    def __init__(self) -> None:
        self.tensors: Dict[SymbolId, TensorDecl] = {}
        self.inputs: Set[SymbolId] = set()
        self.outputs: Set[SymbolId] = set()
        self.written_outputs: Set[SymbolId] = set()
        self.bound_indices: Set[IndexId] = set()
        self.ever_bound_indices: Set[IndexId] = set()
        self.cursors: Dict[CursorId, Tuple[SparseCursorDecl, Optional[MergeMode]]] = {}
        self.ever_cursor_ids: Set[CursorId] = set()
        self.accums: Dict[SymbolId, ReduceOp] = {}
        self.ever_accums: Set[SymbolId] = set()
        self.seen_node_ids: Set[LoopNodeId] = set()
        self.visited_objects: Set[int] = set()
        self.path_objects: Set[int] = set()
        self.in_cursor_default = False


def _check_node_id(node_id: object, path: str) -> LoopNodeId:
    if type(node_id) is not LoopNodeId or type(node_id.value) is not int:
        _fail("invalid_node_id", path, "node_id must be an int-valued LoopNodeId")
    return node_id


def _check_symbol_id(value: object, path: str, what: str) -> SymbolId:
    if type(value) is not SymbolId or type(value.value) is not int:
        _fail("invalid_symbol_id", path, f"{what} must be an int-valued SymbolId")
    return value


def _check_index_id(value: object, path: str, what: str) -> IndexId:
    if type(value) is not IndexId or type(value.value) is not int:
        _fail("invalid_index_id", path, f"{what} must be an int-valued IndexId")
    return value


def _check_cursor_id(value: object, path: str) -> CursorId:
    if type(value) is not CursorId or type(value.value) is not int:
        _fail("invalid_cursor_id", path, "cursor must be an int-valued CursorId")
    return value


def _enter(ctx: _Context, node: object, path: str, depth: int) -> None:
    """Aliasing, cycle, uniqueness, and depth guards for one node object."""

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


def _check_expr(ctx: _Context, expr: object, path: str, depth: int) -> _ValueType:
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


def _check_int_const(
    ctx: _Context, expr: IntConst, path: str, depth: int
) -> _ValueType:
    if type(expr.value) is not int:
        _fail("malformed_state", path, "IntConst.value must be an exact int")
    return _ValueType.COORD


def _check_float_const(
    ctx: _Context, expr: FloatConst, path: str, depth: int
) -> _ValueType:
    if type(expr.value) is not float:
        _fail("malformed_state", path, "FloatConst.value must be an exact float")
    return _ValueType.VALUE


def _check_dim_size(ctx: _Context, expr: DimSize, path: str, depth: int) -> _ValueType:
    tensor = _check_symbol_id(expr.tensor, path, "DimSize.tensor")
    if tensor not in ctx.tensors:
        _fail("undefined_tensor", path, "DimSize references an undeclared tensor")
    if type(expr.dim) is not int:
        _fail("malformed_state", path, "DimSize.dim must be an exact int")
    rank = len(ctx.tensors[tensor].levels)
    if not 0 <= expr.dim < rank:
        _fail("rank_mismatch", path, f"dim {expr.dim} outside rank-{rank} tensor")
    return _ValueType.COORD


def _check_index_value(
    ctx: _Context, expr: IndexValue, path: str, depth: int
) -> _ValueType:
    index = _check_index_id(expr.index, path, "IndexValue.index")
    if index not in ctx.bound_indices:
        _fail("unbound_index", path, f"index {index.value} is not bound in scope")
    return _ValueType.COORD


def _check_cursor_value(
    ctx: _Context, expr: CursorValue, path: str, depth: int
) -> _ValueType:
    if ctx.in_cursor_default:
        _fail(
            "default_contains_cursor",
            path,
            "a CursorValue default must not read another cursor",
        )
    cursor = _check_cursor_id(expr.cursor, path)
    if cursor not in ctx.cursors:
        _fail("unbound_cursor", path, f"cursor {cursor.value} is not in scope")
    mode = ctx.cursors[cursor][1]
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
        if default_type is not _ValueType.VALUE:
            _fail("type_mismatch", f"{path}.default", "default must be value-typed")
    else:
        if expr.default is not None:
            _fail(
                "dead_default",
                path,
                "a default is unobservable outside a UNION merge",
            )
    return _ValueType.VALUE


def _check_accum_value(
    ctx: _Context, expr: AccumValue, path: str, depth: int
) -> _ValueType:
    accumulator = _check_symbol_id(expr.accumulator, path, "AccumValue.accumulator")
    if accumulator not in ctx.accums:
        _fail(
            "undefined_accumulator",
            path,
            "accumulator is not declared and live in an enclosing scope",
        )
    return _ValueType.VALUE


def _check_load(ctx: _Context, expr: Load, path: str, depth: int) -> _ValueType:
    tensor = _check_symbol_id(expr.tensor, path, "Load.tensor")
    if tensor not in ctx.tensors:
        _fail("undefined_tensor", path, "Load references an undeclared tensor")
    if tensor not in ctx.inputs:
        _fail("output_read", path, "Load may only read declared inputs")
    decl = ctx.tensors[tensor]
    if any(level is not LevelKind.DENSE for level in decl.levels):
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
        if index_type is not _ValueType.COORD:
            _fail(
                "type_mismatch",
                f"{path}.indices[{position}]",
                "load indices must be coordinate-typed",
            )
    return _ValueType.VALUE


def _check_binary_expr(
    ctx: _Context, expr: BinaryExpr, path: str, depth: int
) -> _ValueType:
    if type(expr.op) is not BinaryOp:
        _fail("malformed_state", path, "BinaryExpr.op must be a BinaryOp member")
    for name, operand in (("lhs", expr.lhs), ("rhs", expr.rhs)):
        operand_type = _check_expr(ctx, operand, f"{path}.{name}", depth + 1)
        if operand_type is not _ValueType.VALUE:
            _fail(
                "type_mismatch",
                f"{path}.{name}",
                "binary operands must be value-typed",
            )
    return _ValueType.VALUE


_EXPR_CHECKERS: Dict[type, Callable[[_Context, Any, str, int], _ValueType]] = {
    IntConst: _check_int_const,
    FloatConst: _check_float_const,
    DimSize: _check_dim_size,
    IndexValue: _check_index_value,
    CursorValue: _check_cursor_value,
    AccumValue: _check_accum_value,
    Load: _check_load,
    BinaryExpr: _check_binary_expr,
}


def _bind_index(ctx: _Context, index: object, path: str, what: str) -> IndexId:
    bound = _check_index_id(index, path, what)
    if bound in ctx.ever_bound_indices:
        _fail(
            "duplicate_index_binding",
            path,
            f"index {bound.value} is bound more than once in the program",
        )
    ctx.ever_bound_indices.add(bound)
    ctx.bound_indices.add(bound)
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
        if levels[decl.level] is not LevelKind.COMPRESSED:
            _fail(
                "layout_mismatch",
                path,
                "sparse cursors are only defined on compressed levels",
            )
        if type(decl.outer_indices) is not tuple:
            _fail("malformed_state", path, "outer_indices must be an owned tuple")
        if len(decl.outer_indices) != decl.level:
            _fail(
                "rank_mismatch",
                path,
                f"{len(decl.outer_indices)} outer indices for level {decl.level}",
            )
        for position, outer in enumerate(decl.outer_indices):
            outer_type = _check_expr(
                ctx, outer, f"{path}.outer_indices[{position}]", depth + 1
            )
            if outer_type is not _ValueType.COORD:
                _fail(
                    "type_mismatch",
                    f"{path}.outer_indices[{position}]",
                    "outer indices must be coordinate-typed",
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
    extent_type = _check_expr(ctx, stmt.extent, f"{path}.extent", depth + 1)
    if extent_type is not _ValueType.COORD:
        _fail("type_mismatch", f"{path}.extent", "extent must be coordinate-typed")
    index = _bind_index(ctx, stmt.index, path, "DenseFor.index")
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
    finally:
        ctx.bound_indices.discard(index)


def _check_sparse_for(ctx: _Context, stmt: SparseFor, path: str, depth: int) -> None:
    decl = _check_cursor_decl(ctx, stmt.cursor, f"{path}.cursor", depth + 1)
    index = _bind_index(ctx, stmt.coord_index, path, "SparseFor.coord_index")
    ctx.cursors[decl.cursor] = (decl, None)
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
    finally:
        ctx.bound_indices.discard(index)
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
        decls.append(
            _check_cursor_decl(ctx, cursor, f"{path}.cursors[{position}]", depth + 1)
        )
    index = _bind_index(ctx, stmt.coord_index, path, "MergedSparseFor.coord_index")
    for decl in decls:
        ctx.cursors[decl.cursor] = (decl, stmt.mode)
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
    finally:
        ctx.bound_indices.discard(index)
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
    if init_type is not _ValueType.VALUE or type(init) is not FloatConst:
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
    if value_type is not _ValueType.VALUE:
        _fail("type_mismatch", f"{path}.value", "accumulated value must be value-typed")


def _check_store(ctx: _Context, stmt: Store, path: str, depth: int) -> None:
    tensor = _check_symbol_id(stmt.tensor, path, "Store.tensor")
    if tensor not in ctx.tensors:
        _fail("undefined_tensor", path, "Store references an undeclared tensor")
    if tensor not in ctx.outputs:
        _fail("output_scope", path, "Store may only write declared outputs")
    decl = ctx.tensors[tensor]
    if any(level is not LevelKind.DENSE for level in decl.levels):
        _fail(
            "layout_mismatch",
            path,
            "coordinate stores are only defined on all-dense outputs",
        )
    if type(stmt.indices) is not tuple:
        _fail("malformed_state", path, "Store.indices must be an owned tuple")
    if len(stmt.indices) != len(decl.levels):
        _fail(
            "rank_mismatch",
            path,
            f"{len(stmt.indices)} indices for rank-{len(decl.levels)} output",
        )
    for position, index in enumerate(stmt.indices):
        index_type = _check_expr(ctx, index, f"{path}.indices[{position}]", depth + 1)
        if index_type is not _ValueType.COORD:
            _fail(
                "type_mismatch",
                f"{path}.indices[{position}]",
                "store indices must be coordinate-typed",
            )
    value_type = _check_expr(ctx, stmt.value, f"{path}.value", depth + 1)
    if value_type is not _ValueType.VALUE:
        _fail("type_mismatch", f"{path}.value", "stored value must be value-typed")
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
    if all(level is not LevelKind.COMPRESSED for level in decl.levels):
        _fail(
            "layout_mismatch",
            path,
            "appended assembly needs an output with a compressed level",
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
        if coord_type is not _ValueType.COORD:
            _fail(
                "type_mismatch",
                f"{path}.coords[{position}]",
                "appended coordinates must be coordinate-typed",
            )
    value_type = _check_expr(ctx, stmt.value, f"{path}.value", depth + 1)
    if value_type is not _ValueType.VALUE:
        _fail("type_mismatch", f"{path}.value", "appended value must be value-typed")
    ctx.written_outputs.add(tensor)


_STMT_CHECKERS: Dict[type, Callable[[_Context, Any, str, int], None]] = {
    Block: _check_block,
    DenseFor: _check_dense_for,
    SparseFor: _check_sparse_for,
    MergedSparseFor: _check_merged_sparse_for,
    DeclAccum: _check_decl_accum,
    Accumulate: _check_accumulate,
    Store: _check_store,
    AppendEntry: _check_append_entry,
}


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
        if type(decl.levels) is not tuple or not decl.levels:
            _fail(
                "malformed_state",
                path,
                "TensorDecl.levels must be a nonempty owned tuple",
            )
        for position, level in enumerate(decl.levels):
            if type(level) is not LevelKind:
                _fail(
                    "malformed_state",
                    f"{path}.levels[{position}]",
                    "levels must be LevelKind members",
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
        if type(program.tensors) is not tuple or not program.tensors:
            _fail(
                "malformed_state",
                "program.tensors",
                "tensors must be a nonempty owned tuple",
            )
        for position, decl in enumerate(program.tensors):
            _check_tensor_decl(ctx, decl, f"program.tensors[{position}]", 1)
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
