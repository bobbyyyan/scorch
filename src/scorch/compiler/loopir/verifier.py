"""Fail-closed structural verifier for the production LoopIR dense subset.

``verify_program`` is the single authority over LoopIR program validity.  It
raises :class:`LoopIRVerificationError` carrying a stable defect code and the
lexical path of the offending node on the first defect found.  Constructors
perform no validation, so every boundary here is checked from stored state
with exact types: unknown node subclasses, non-tuple children, forged enum
lookalikes, aliased or cyclic structure, and excessive nesting all fail
closed rather than being coerced or skipped.

The invariant families stated locally for this subset:

- **Coordinate domains.**  Every bound coordinate carries the logical
  dimension it indexes; loads and stores reject coordinates from a different
  domain (``domain_mismatch``).
- **Scalar typing.**  Every tensor declares an exact :class:`ScalarType`,
  and this slice requires one uniform scalar type across the whole program
  (``mixed_dtype``); binary operands and stored values must be value-typed.
- **Output semantics.**  Inputs are never written, outputs are never read,
  every output is written, and ``StoreReduce`` admits only ADD — the one
  operator whose identity matches the explicit dense-output
  zero-initialization contract (``ReduceOp`` declares no other member, so
  the exact-member check is the whole contract).
- **Extent resolution.**  Every ``DenseFor`` dimension must be mapped by at
  least one declared tensor so its runtime extent has a source
  (``unresolved_dimension``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, NoReturn, Optional, Set

from ..identity import IndexId, SymbolId
from .nodes import (
    BinaryExpr,
    BinaryOp,
    Block,
    DenseFor,
    DimensionDecl,
    DimensionId,
    Expr,
    IndexValue,
    LevelDecl,
    LevelKind,
    Load,
    LoopIRNodeId,
    LoopProgram,
    ReduceOp,
    ScalarType,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
)

MAX_NESTING_DEPTH = 64
_MISSING = object()

_EXECUTABLE_LEVEL_KINDS = (LevelKind.DENSE,)


@dataclass(frozen=True)
class LoopIRDefect:
    """One immutable verification failure: stable code, path, and message."""

    code: str
    path: str
    message: str


class LoopIRVerificationError(Exception):
    """A LoopIR program violated a structural invariant."""

    def __init__(self, defect: LoopIRDefect) -> None:
        super().__init__(f"{defect.code} at {defect.path}: {defect.message}")
        self.defect = defect


class _ExprType:
    """Base of the verifier's expression types."""


@dataclass(frozen=True)
class _CoordType(_ExprType):
    """Coordinate-typed within one logical dimension's domain."""

    dimension: DimensionId


@dataclass(frozen=True)
class _ScalarValueType(_ExprType):
    """Value-typed (a stored or computed scalar)."""


_VALUE = _ScalarValueType()


def _fail(code: str, path: str, message: str) -> NoReturn:
    raise LoopIRVerificationError(LoopIRDefect(code, path, message))


class _Context:
    """Mutable walk state: registries, scopes, and traversal guards."""

    def __init__(self) -> None:
        self.dimensions: Dict[DimensionId, DimensionDecl] = {}
        self.mapped_dimensions: Set[DimensionId] = set()
        self.tensors: Dict[SymbolId, TensorDecl] = {}
        self.inputs: Set[SymbolId] = set()
        self.outputs: Set[SymbolId] = set()
        self.written_outputs: Set[SymbolId] = set()
        self.bound_indices: Dict[IndexId, DimensionId] = {}
        self.ever_bound_indices: Set[IndexId] = set()
        self.program_dtype: Optional[ScalarType] = None
        self.seen_node_ids: Set[LoopIRNodeId] = set()
        self.visited_objects: Set[int] = set()
        self.path_objects: Set[int] = set()

    def dimension_name(self, dimension: DimensionId) -> str:
        decl = self.dimensions.get(dimension)
        return decl.name if decl is not None else f"<dimension {dimension.value}>"


def _check_node_id(node_id: object, path: str) -> LoopIRNodeId:
    if (
        type(node_id) is not LoopIRNodeId
        or type(getattr(node_id, "value", _MISSING)) is not int
    ):
        _fail("invalid_node_id", path, "node_id must be an int-valued LoopIRNodeId")
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
    if type(kind) is not _ScalarValueType:
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
    if kind.dimension != expected:
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


def _check_index_value(
    ctx: _Context, expr: IndexValue, path: str, depth: int
) -> _ExprType:
    index = _check_index_id(expr.index, path, "IndexValue.index")
    if index not in ctx.bound_indices:
        _fail("unbound_index", path, f"index {index.value} is not bound in scope")
    return _CoordType(ctx.bound_indices[index])


def _check_load(ctx: _Context, expr: Load, path: str, depth: int) -> _ExprType:
    tensor = _check_symbol_id(expr.tensor, path, "Load.tensor")
    if tensor not in ctx.tensors:
        _fail("undefined_tensor", path, "Load references an undeclared tensor")
    if tensor not in ctx.inputs:
        _fail("output_read", path, "Load may only read declared inputs")
    decl = ctx.tensors[tensor]
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
    IndexValue: _check_index_value,
    Load: _check_load,
    BinaryExpr: _check_binary_expr,
}


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
    for position, stmt in enumerate(block.statements):
        _check_stmt(ctx, stmt, f"{path}.statements[{position}]", depth + 1)


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
    if dimension not in ctx.mapped_dimensions:
        _fail(
            "unresolved_dimension",
            f"{path}.dimension",
            "DenseFor dimension has no tensor-mapped runtime extent source",
        )
    index = _check_index_id(stmt.index, path, "DenseFor.index")
    if index in ctx.ever_bound_indices:
        _fail(
            "duplicate_index_binding",
            path,
            f"index {index.value} is bound more than once in the program",
        )
    ctx.ever_bound_indices.add(index)
    ctx.bound_indices[index] = dimension
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
    finally:
        del ctx.bound_indices[index]


def _check_output_write_indices(
    ctx: _Context, stmt: object, tensor: SymbolId, path: str, depth: int
) -> None:
    """Shared index checks for coordinate-addressed dense-output writes."""

    decl = ctx.tensors[tensor]
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
    _check_output_write_indices(ctx, stmt, tensor, path, depth)
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
        # ADD is the only declared ReduceOp member, so an exact member check
        # is the whole reduction-operator contract; adding a member requires
        # adding its explicit output-initialization identity check here.
        _fail("malformed_state", path, "StoreReduce.op must be a ReduceOp member")
    _check_output_write_indices(ctx, stmt, tensor, path, depth)
    value_type = _check_expr(ctx, stmt.value, f"{path}.value", depth + 1)
    _require_value(value_type, f"{path}.value", "a combined value")
    ctx.written_outputs.add(tensor)


_STMT_CHECKERS: Dict[type, Callable[[_Context, Any, str, int], None]] = {
    Block: _check_block,
    DenseFor: _check_dense_for,
    Store: _check_store,
    StoreReduce: _check_store_reduce,
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
                "the Phase-4 dense subset fails closed on them until a later "
                "phase represents their iteration",
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
        if type(decl.dtype) is not ScalarType:
            _fail(
                "invalid_scalar_type",
                path,
                "TensorDecl.dtype must be a ScalarType member",
            )
        if ctx.program_dtype is None:
            ctx.program_dtype = decl.dtype
        elif decl.dtype is not ctx.program_dtype:
            _fail(
                "mixed_dtype",
                path,
                "this subset requires one uniform scalar type per program; "
                f"got {decl.dtype.value} beside {ctx.program_dtype.value}",
            )
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
            ctx.mapped_dimensions.add(checked)
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
    """Fail closed unless ``program`` is a structurally valid LoopIR program."""

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
                "an output is never stored to",
            )
    finally:
        _leave(ctx, program)
