"""Fail-closed structural verifier for the production LoopIR subset.

``verify_program`` is the single authority over LoopIR program validity.  It
raises :class:`LoopIRVerificationError` carrying a stable defect code and the
lexical path of the offending node on the first defect found.  Constructors
perform no validation, so every boundary here is checked from stored state
with exact types: unknown node subclasses, non-tuple children, forged enum
lookalikes, aliased or cyclic structure, and excessive nesting all fail
closed rather than being coerced or skipped.

The invariant families stated locally for this subset:

- **Coordinate domains.**  Every bound coordinate carries the logical
  dimension it indexes; loads, stores, appends, dense positions, and merged
  cursors reject coordinates or levels from a different domain
  (``domain_mismatch`` / ``merge_domain_mismatch``).
- **Physical position dominance.**  A sparse cursor and a dense position
  each name their dominating parent explicitly, and the parent must be a
  position of the immediately enclosing level of the same tensor, grounded
  at the root (``parent_position_mismatch``).  Positions are bound only by
  ``SparseFor`` (``duplicate_position_binding`` / ``unbound_position``).
- **Leaf value ownership.**  Only a cursor over the value-bearing leaf
  level of its tensor owns scalars (``non_leaf_value``); UNION-merged reads
  require an explicit default and always-aligned reads must not carry one
  (``missing_union_default`` / ``dead_default`` /
  ``default_contains_cursor``).
- **Merge structure.**  Merged loops need at least two cursors
  (``degenerate_merge``), leaf-level targets
  (``unsupported_sparse_hierarchy``), and one shared merged dimension;
  aligned advancement, exhaustion, and guaranteed progress are intrinsic
  node semantics stated on :class:`MergedSparseFor`.
- **Scalar typing.**  Every tensor declares an exact :class:`ScalarType`,
  and this slice requires one uniform scalar type across the whole program
  (``mixed_dtype``); binary operands and stored values must be value-typed.
- **Output semantics.**  Inputs are never written, outputs are never read,
  every output is written, ``StoreReduce`` admits only ADD (the operator
  whose identity matches the explicit dense-output zero-initialization
  contract), coordinate stores require all-dense outputs and appended
  assembly requires a compressed output (``layout_mismatch``), and sparse
  outputs use identity-ordered DENSE/COMPRESSED levels only
  (``unsupported_sparse_output``).
- **Extent resolution.**  Every ``DenseFor`` dimension must be mapped by at
  least one declared tensor so its runtime extent has a source
  (``unresolved_dimension``); tile loops share the same rule.
- **Affine splits.**  A strip-mined loop is one ``TileOuterFor`` /
  ``TileInnerFor`` pair owning a unique ``TileId``
  (``invalid_tile_id`` / ``duplicate_tile_id``); the point loop must run
  inside its origin loop's scope (``unbound_tile``), every origin must
  contain its point loop (``missing_tile_inner``), the pair must agree on
  index, dimension, and width (``tile_binding_mismatch``), and widths are
  positive exact ints within the target-neutral canonical-print boundary
  (``invalid_tile_width``).  A split owns its logical loop: the split index
  may be neither bound nor split again in an enclosing scope
  (``tile_index_conflict``), and the point loop's
  coordinate binding participates in the ``duplicate_index_binding``
  discipline — with one deliberately moved boundary from the workspace
  family: one split may bind its point coordinate through *several*
  ``TileInnerFor`` loops of the same ``TileId`` in disjoint sibling scopes
  (a workspace region's producer and consumer each iterate the clamped
  window once); nested rebinding and every other binder kind keep the
  global once-only rule.  Ragged-tail coverage is intrinsic
  ``TileInnerFor`` semantics, stated on the node.
- **Sparse coordinate panels.**  A panel is one ``PanelOuterFor`` /
  ``SparseWindowFor`` pair owning a unique ``TileId`` drawn from the same
  identity space as affine splits (``duplicate_tile_id``); the window must
  run inside its origin loop's scope (``unbound_panel``), every panel
  origin must contain its window (``missing_panel_window``), and the pair
  must agree on the bound coordinate and its domain — the window's
  coordinate index is the panel's logical index and the window's cursor
  level stores the panel's dimension (``panel_binding_mismatch``).  The
  panel's declared bound must be a DENSE level whose logical mode stores
  the panel dimension (``panel_bound_mismatch``).  A panel owns its
  logical loop exactly like an affine split (``tile_index_conflict``),
  its window binds the coordinate under the global once-only discipline,
  and window widths are positive exact ints within the target-neutral
  canonical-print boundary (``invalid_tile_width``).  Clamped-window
  coverage is intrinsic ``SparseWindowFor`` semantics, stated on the node.
- **Workspace regions.**  A stack workspace is declared by exactly one
  region (``invalid_workspace_id`` / ``duplicate_workspace_id``) and spans
  the point domain of one affine split: the region must open between that
  split's origin loop and its point loops (``workspace_scope_mismatch``).
  Allocation and zero-reset are intrinsic region-entry semantics, the
  producer owns writes and the consumer owns reads
  (``workspace_write_scope`` / ``workspace_read_scope`` /
  ``unbound_workspace``), a producer must not write declared outputs
  (``workspace_output_write``), cells are addressed only by the owning
  split's point coordinate bound inside the region
  (``workspace_coord_mismatch``), and a region must actually accumulate
  and copy out (``workspace_dead_region``).
- **Staged-operand relayout regions.**  A relayout is declared by exactly
  one region (``invalid_relayout_id`` / ``duplicate_relayout_id``) staging
  one rank-2 all-dense input whose first level stores the panel's
  dimension and whose last level stores the pack split's dimension
  (``relayout_operand_mismatch``).  The region must open inside its pack
  split's origin loop, a PANEL-scoped region inside its panel's origin
  loop, and a PACK_AXIS-scoped region outside it
  (``relayout_scope_mismatch``).  Staging and teardown are intrinsic
  region semantics; ``StagedRead`` is legal only inside the region
  (``unbound_relayout``), its row index must be the panel's window
  coordinate and its column index the pack split's point coordinate
  (``relayout_read_mismatch``), and a region must actually be read
  (``relayout_dead_region``).
- **Heap result-tile regions.**  A compact result tile is declared by
  exactly one region (``invalid_result_tile_id`` /
  ``duplicate_result_tile_id``) accumulating one all-dense declared output
  of rank at least two whose last storage level stores the pack split's
  dimension (``result_tile_result_mismatch``).  The region must open
  inside its pack split's origin loop, outside the split's point loops,
  and never nested inside another region of the same result
  (``result_tile_scope_mismatch``).  Fresh zero entry, accumulation, and
  exactly-once copy-out at exit are intrinsic region semantics;
  ``TiledReduce`` is legal only inside the region
  (``unbound_result_tile``), its indices must be the result's access
  indices with the trailing index the pack split's point coordinate
  (``result_tile_write_mismatch``), the declared result must not be
  written directly while its region is open
  (``result_tile_residual_write``), and a region must actually accumulate
  (``result_tile_dead_region``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Tuple, cast

from ..identity import IndexId, SymbolId
from .nodes import (
    AppendEntry,
    BinaryExpr,
    BinaryOp,
    Block,
    CursorId,
    CursorValue,
    DenseFor,
    DensePosition,
    DimensionDecl,
    DimensionId,
    Expr,
    FloatConst,
    IndexValue,
    LevelDecl,
    LevelKind,
    Load,
    LoopIRNode,
    LoopIRNodeId,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    PanelOuterFor,
    PositionId,
    ParallelDiscipline,
    ParallelIntent,
    ParallelPart,
    ParallelSelection,
    ParallelWork,
    PositionLoad,
    PositionValue,
    ReduceOp,
    RelayoutDecl,
    RelayoutId,
    RelayoutScope,
    RelayoutStage,
    ResultTileDecl,
    ResultTileId,
    ResultTileRegion,
    RootPosition,
    ScalarType,
    SparseCursorDecl,
    SparseFor,
    SparseWindowFor,
    SparseWorkspaceDecl,
    SparseWorkspaceDrainFor,
    SparseWorkspaceInsert,
    SparseWorkspaceRegion,
    SparseWorkspaceValue,
    SparseWorkSource,
    StagedRead,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
    TiledReduce,
    TileId,
    TileInnerFor,
    TileOuterFor,
    WorkspaceDecl,
    WorkspaceId,
    WorkspaceRead,
    WorkspaceReduce,
    WorkspaceRegion,
)

MAX_NESTING_DEPTH = 64
MAX_LOOPIR_TILE_WIDTH_BITS = 2048
MAX_LOOPIR_TILE_WIDTH = (1 << MAX_LOOPIR_TILE_WIDTH_BITS) - 1
"""Largest target-neutral tile width with a total canonical text form.

CPython permits the decimal integer-conversion limit to be configured as low
as 640 digits.  A positive 2048-bit integer has at most 617 decimal digits, so
every verifier-approved width remains exactly printable and JSON-serializable
under every supported setting.  Target lowerings may impose a much smaller
machine-specific limit.
"""

_MISSING = object()

_EXECUTABLE_LEVEL_KINDS = (LevelKind.DENSE, LevelKind.COMPRESSED)

_CANONICAL_CSR_KINDS = (LevelKind.DENSE, LevelKind.COMPRESSED)
_CANONICAL_CSR_MODES = (0, 1)


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


@dataclass(frozen=True)
class _PositionType(_ExprType):
    """Position-typed within one physical level of one tensor.

    The root position carries ``tensor=None, level=-1``; it dominates every
    level-0 level of every tensor.
    """

    tensor: Optional[SymbolId]
    level: int


_VALUE = _ScalarValueType()
_ROOT_POSITION = _PositionType(None, -1)


def _fail(code: str, path: str, message: str) -> NoReturn:
    raise LoopIRVerificationError(LoopIRDefect(code, path, message))


def _diagnostic_int(value: int) -> str:
    """Render an exact integer without leaking Python's digit-limit error."""

    if value.bit_length() > MAX_LOOPIR_TILE_WIDTH_BITS:
        return "<integer too large to render>"
    try:
        return str(value)
    except ValueError:
        return "<integer too large to render>"


class _WorkspaceState:
    """One open workspace region's walk state."""

    __slots__ = ("decl", "role", "produced", "consumed")

    def __init__(self, decl: WorkspaceDecl) -> None:
        self.decl = decl
        self.role: Optional[str] = None
        self.produced = False
        self.consumed = False


class _SparseWorkspaceState:
    """One open sparse-workspace region's walk state."""

    __slots__ = (
        "decl",
        "role",
        "inserted",
        "drained",
        "drain_depth",
        "consumed",
    )

    def __init__(self, decl: SparseWorkspaceDecl) -> None:
        self.decl = decl
        self.role: Optional[str] = None
        self.inserted = False
        self.drained = False
        self.drain_depth = 0
        self.consumed = False


class _RelayoutState:
    """One open staged-relayout region's walk state."""

    __slots__ = ("decl", "read")

    def __init__(self, decl: RelayoutDecl) -> None:
        self.decl = decl
        self.read = False


class _ResultTileState:
    """One open heap result-tile region's walk state."""

    __slots__ = ("decl", "reduced")

    def __init__(self, decl: ResultTileDecl) -> None:
        self.decl = decl
        self.reduced = False


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
        self.tile_point_bindings: Dict[IndexId, TileId] = {}
        self.ever_tile_point_bindings: Dict[IndexId, TileId] = {}
        self.cursors: Dict[CursorId, Tuple[SparseCursorDecl, Optional[MergeMode]]] = {}
        self.ever_cursor_ids: Set[CursorId] = set()
        self.bound_positions: Dict[PositionId, Tuple[SymbolId, int]] = {}
        self.ever_bound_positions: Set[PositionId] = set()
        self.open_tiles: Dict[TileId, TileOuterFor] = {}
        self.matched_tile_inners: Set[TileId] = set()
        self.ever_tile_ids: Set[TileId] = set()
        self.open_panels: Dict[TileId, PanelOuterFor] = {}
        self.matched_panel_windows: Set[TileId] = set()
        self.open_workspaces: Dict[WorkspaceId, _WorkspaceState] = {}
        self.open_sparse_workspaces: Dict[WorkspaceId, _SparseWorkspaceState] = {}
        self.ever_workspace_ids: Set[WorkspaceId] = set()
        self.open_relayouts: Dict[RelayoutId, _RelayoutState] = {}
        self.ever_relayout_ids: Set[RelayoutId] = set()
        self.open_result_tiles: Dict[ResultTileId, _ResultTileState] = {}
        self.open_result_tile_results: Set[SymbolId] = set()
        self.ever_result_tile_ids: Set[ResultTileId] = set()
        self.producer_depth = 0
        self.in_cursor_default = False
        self.program_dtype: Optional[ScalarType] = None
        self.seen_node_ids: Set[LoopIRNodeId] = set()
        self.visited_objects: Set[int] = set()
        self.path_objects: Set[int] = set()
        # Exact statement ancestry, including Block nodes.  Region lifetime
        # checks use object identity here rather than reconstructing ownership
        # from lexical path strings.
        self.statement_stack: List[Stmt] = []

    def dimension_name(self, dimension: DimensionId) -> str:
        decl = self.dimensions.get(dimension)
        return (
            decl.name
            if decl is not None
            else f"<dimension {_diagnostic_int(dimension.value)}>"
        )

    def level_dimension(self, tensor: SymbolId, level: int) -> DimensionId:
        """The logical dimension stored by one validated tensor level."""

        decl = self.tensors[tensor]
        return decl.dimensions[decl.levels[level].mode]


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


def _check_cursor_id(value: object, path: str) -> CursorId:
    if (
        type(value) is not CursorId
        or type(getattr(value, "value", _MISSING)) is not int
    ):
        _fail("invalid_cursor_id", path, "cursor must be an int-valued CursorId")
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
        _fail(
            "duplicate_node_id",
            path,
            f"node_id {_diagnostic_int(node_id.value)} reused",
        )
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
        _fail(
            "unbound_index",
            path,
            f"index {_diagnostic_int(index.value)} is not bound in scope",
        )
    return _CoordType(ctx.bound_indices[index])


def _check_float_const(
    ctx: _Context, expr: FloatConst, path: str, depth: int
) -> _ExprType:
    if type(expr.value) is not float:
        _fail("malformed_state", path, "FloatConst.value must be an exact float")
    if not math.isfinite(expr.value):
        _fail("malformed_state", path, "FloatConst.value must be finite")
    return _VALUE


def _check_root_position(
    ctx: _Context, expr: RootPosition, path: str, depth: int
) -> _ExprType:
    return _ROOT_POSITION


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
            f"level {_diagnostic_int(expr.level)} outside "
            f"rank-{len(decl.levels)} tensor",
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
            f"position {_diagnostic_int(position.value)} is not bound in scope",
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
        _fail(
            "unbound_cursor",
            path,
            f"cursor {_diagnostic_int(cursor.value)} is not in scope",
        )
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
        _require_value(default_type, f"{path}.default", "a cursor default")
    else:
        if expr.default is not None:
            _fail(
                "dead_default",
                path,
                "a default is unobservable outside a UNION merge",
            )
    return _VALUE


def _check_position_load(
    ctx: _Context, expr: PositionLoad, path: str, depth: int
) -> _ExprType:
    tensor = _check_symbol_id(expr.tensor, path, "PositionLoad.tensor")
    if tensor not in ctx.tensors:
        _fail("undefined_tensor", path, "PositionLoad references an undeclared tensor")
    if tensor not in ctx.inputs:
        _fail("output_read", path, "PositionLoad may only read declared inputs")
    position_type = _check_expr(ctx, expr.position, f"{path}.position", depth + 1)
    if type(position_type) is not _PositionType:
        _fail(
            "type_mismatch",
            f"{path}.position",
            "PositionLoad.position must be position-typed",
        )
    if position_type.tensor != tensor:
        _fail(
            "position_load_mismatch",
            f"{path}.position",
            "the leaf position must belong to the loaded tensor",
        )
    leaf_level = len(ctx.tensors[tensor].levels) - 1
    if position_type.level != leaf_level:
        _fail(
            "non_leaf_value",
            f"{path}.position",
            "only the value-bearing leaf position owns a scalar; "
            f"level {position_type.level} is structural",
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


def _check_workspace_id(value: object, path: str) -> WorkspaceId:
    if (
        type(value) is not WorkspaceId
        or type(getattr(value, "value", _MISSING)) is not int
    ):
        _fail(
            "invalid_workspace_id",
            path,
            "workspace must be an int-valued WorkspaceId",
        )
    return value


def _check_workspace_coord(
    ctx: _Context,
    state: _WorkspaceState,
    coord: object,
    path: str,
    depth: int,
) -> None:
    """Cells are addressed only by the owning split's in-region coordinate."""

    _check_expr(ctx, coord, path, depth)
    if (
        type(coord) is not IndexValue
        or ctx.tile_point_bindings.get(coord.index) != state.decl.tile
    ):
        _fail(
            "workspace_coord_mismatch",
            path,
            "a workspace cell is addressed by the owning split's point "
            "coordinate, bound by a TileInnerFor of that tile inside the "
            "region",
        )


def _check_workspace_read(
    ctx: _Context, expr: WorkspaceRead, path: str, depth: int
) -> _ExprType:
    workspace = _check_workspace_id(expr.workspace, path)
    state = ctx.open_workspaces.get(workspace)
    if state is None:
        _fail(
            "unbound_workspace",
            path,
            f"workspace {_diagnostic_int(workspace.value)} has no enclosing "
            "region in scope",
        )
    if state.role != "consumer":
        _fail(
            "workspace_read_scope",
            path,
            "a workspace is readable only inside its region's consumer",
        )
    _check_workspace_coord(ctx, state, expr.coord, f"{path}.coord", depth + 1)
    state.consumed = True
    return _VALUE


def _check_sparse_workspace_value(
    ctx: _Context, expr: SparseWorkspaceValue, path: str, depth: int
) -> _ExprType:
    workspace = _check_workspace_id(expr.workspace, path)
    state = ctx.open_sparse_workspaces.get(workspace)
    if state is None:
        _fail(
            "unbound_workspace",
            path,
            f"sparse workspace {_diagnostic_int(workspace.value)} has no "
            "enclosing region in scope",
        )
    if state.role != "consumer" or state.drain_depth == 0:
        _fail(
            "workspace_read_scope",
            path,
            "a drained value is readable only inside the owning "
            "workspace's drain loop",
        )
    state.consumed = True
    return _VALUE


def _check_relayout_id(value: object, path: str) -> RelayoutId:
    if (
        type(value) is not RelayoutId
        or type(getattr(value, "value", _MISSING)) is not int
    ):
        _fail(
            "invalid_relayout_id",
            path,
            "relayout must be an int-valued RelayoutId",
        )
    return value


def _check_result_tile_id(value: object, path: str) -> ResultTileId:
    if (
        type(value) is not ResultTileId
        or type(getattr(value, "value", _MISSING)) is not int
    ):
        _fail(
            "invalid_result_tile_id",
            path,
            "result_tile must be an int-valued ResultTileId",
        )
    return value


def _check_staged_read(
    ctx: _Context, expr: StagedRead, path: str, depth: int
) -> _ExprType:
    relayout = _check_relayout_id(expr.relayout, path)
    state = ctx.open_relayouts.get(relayout)
    if state is None:
        _fail(
            "unbound_relayout",
            path,
            f"relayout {_diagnostic_int(relayout.value)} has no enclosing "
            "staging region in scope",
        )
    decl = state.decl
    operand_decl = ctx.tensors[decl.operand]
    if type(expr.indices) is not tuple:
        _fail("malformed_state", path, "StagedRead.indices must be an owned tuple")
    if len(expr.indices) != len(operand_decl.levels):
        _fail(
            "rank_mismatch",
            path,
            f"{len(expr.indices)} indices for rank-{len(operand_decl.levels)} "
            "operand",
        )
    panel = ctx.open_panels.get(decl.panel)
    if panel is None:
        _fail(
            "relayout_read_mismatch",
            path,
            "a staged read needs its panel's origin loop in scope: the "
            "staged row is the panel's window coordinate",
        )
    if ctx.level_dimension(decl.operand, 0) != panel.dimension:
        _fail(
            "relayout_operand_mismatch",
            path,
            "the staged operand's first storage level must store the "
            "panel's dimension",
        )
    for position, index in enumerate(expr.indices):
        index_type = _check_expr(ctx, index, f"{path}.indices[{position}]", depth + 1)
        _require_coord(
            ctx,
            index_type,
            f"{path}.indices[{position}]",
            "a staged read index",
            operand_decl.dimensions[position],
        )
    row_index = expr.indices[operand_decl.levels[0].mode]
    pack_index = expr.indices[operand_decl.levels[1].mode]
    if type(row_index) is not IndexValue or row_index.index != panel.index:
        _fail(
            "relayout_read_mismatch",
            path,
            "the staged row index must be the panel's window coordinate",
        )
    if (
        type(pack_index) is not IndexValue
        or ctx.tile_point_bindings.get(pack_index.index) != decl.pack
    ):
        _fail(
            "relayout_read_mismatch",
            path,
            "the staged column index must be the pack split's point "
            "coordinate, bound by its TileInnerFor",
        )
    state.read = True
    return _VALUE


_EXPR_CHECKERS: Dict[type, Callable[[_Context, Any, str, int], _ExprType]] = {
    IndexValue: _check_index_value,
    FloatConst: _check_float_const,
    RootPosition: _check_root_position,
    DensePosition: _check_dense_position,
    PositionValue: _check_position_value,
    CursorValue: _check_cursor_value,
    PositionLoad: _check_position_load,
    Load: _check_load,
    BinaryExpr: _check_binary_expr,
    WorkspaceRead: _check_workspace_read,
    StagedRead: _check_staged_read,
    SparseWorkspaceValue: _check_sparse_workspace_value,
}


def _bind_index(
    ctx: _Context,
    index: object,
    path: str,
    what: str,
    dimension: DimensionId,
    point_tile: Optional[TileId] = None,
) -> IndexId:
    """Bind one loop coordinate under the once-only discipline.

    ``point_tile`` marks a ``TileInnerFor`` binding: a split's point
    coordinate may be rebound by sibling point loops of the *same* tile
    (each iterates the same clamped window exactly once — the workspace
    producer/consumer shape).  Nested rebinding and every other repeated
    binding remain ``duplicate_index_binding``.
    """

    bound = _check_index_id(index, path, what)
    if bound in ctx.bound_indices:
        _fail(
            "duplicate_index_binding",
            path,
            f"index {_diagnostic_int(bound.value)} is already bound in an "
            "enclosing scope",
        )
    if bound in ctx.ever_bound_indices and (
        point_tile is None or ctx.ever_tile_point_bindings.get(bound) != point_tile
    ):
        _fail(
            "duplicate_index_binding",
            path,
            f"index {_diagnostic_int(bound.value)} is bound more than once in "
            "the program",
        )
    ctx.ever_bound_indices.add(bound)
    if point_tile is not None:
        ctx.ever_tile_point_bindings[bound] = point_tile
        ctx.tile_point_bindings[bound] = point_tile
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
            f"position {_diagnostic_int(bound.value)} is bound more than once "
            "in the program",
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
            _fail(
                "duplicate_cursor_id",
                path,
                f"cursor id {_diagnostic_int(cursor.value)} reused",
            )
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
                f"level {_diagnostic_int(decl.level)} outside "
                f"rank-{len(levels)} tensor",
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
    ctx.statement_stack.append(stmt)
    try:
        checker(ctx, stmt, path, depth)
    finally:
        popped = ctx.statement_stack.pop()
        assert popped is stmt
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


def _check_tile_id(value: object, path: str) -> TileId:
    if type(value) is not TileId or type(getattr(value, "value", _MISSING)) is not int:
        _fail("invalid_tile_id", path, "tile must be an int-valued TileId")
    return value


def _check_tile_loop_dimension(
    ctx: _Context, dimension: object, path: str, what: str
) -> DimensionId:
    checked = _check_dimension_id(dimension, f"{path}.dimension", what)
    if checked not in ctx.dimensions:
        _fail(
            "undefined_dimension",
            f"{path}.dimension",
            f"{what} iterates an undeclared dimension",
        )
    if checked not in ctx.mapped_dimensions:
        _fail(
            "unresolved_dimension",
            f"{path}.dimension",
            f"{what} dimension has no tensor-mapped runtime extent source",
        )
    return checked


def _check_tile_width(width: object, path: str, what: str) -> int:
    if type(width) is not int:
        _fail("invalid_tile_width", path, f"{what} must be an exact int")
    if width < 1:
        _fail("invalid_tile_width", path, f"{what} must be at least 1")
    if width > MAX_LOOPIR_TILE_WIDTH:
        _fail(
            "invalid_tile_width",
            path,
            f"{what} exceeds the target-neutral canonical integer boundary",
        )
    return width


def _check_tile_outer_for(
    ctx: _Context, stmt: TileOuterFor, path: str, depth: int
) -> None:
    tile = _check_tile_id(stmt.tile, f"{path}.tile")
    if tile in ctx.ever_tile_ids:
        _fail(
            "duplicate_tile_id",
            path,
            f"tile id {_diagnostic_int(tile.value)} reused",
        )
    ctx.ever_tile_ids.add(tile)
    _check_tile_loop_dimension(ctx, stmt.dimension, path, "TileOuterFor")
    index = _check_index_id(stmt.index, path, "TileOuterFor.index")
    if index in ctx.bound_indices:
        _fail(
            "tile_index_conflict",
            path,
            f"index {_diagnostic_int(index.value)} is already bound in an "
            "enclosing scope; a split must own its logical loop",
        )
    if any(open_tile.index == index for open_tile in ctx.open_tiles.values()) or any(
        open_panel.index == index for open_panel in ctx.open_panels.values()
    ):
        _fail(
            "tile_index_conflict",
            path,
            f"index {_diagnostic_int(index.value)} is already split by an "
            "enclosing tile",
        )
    _check_tile_width(stmt.width, path, "TileOuterFor.width")
    ctx.open_tiles[tile] = stmt
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
        if tile not in ctx.matched_tile_inners:
            _fail(
                "missing_tile_inner",
                path,
                f"TileOuterFor for tile id {_diagnostic_int(tile.value)} has "
                "no matching TileInnerFor in its body",
            )
    finally:
        del ctx.open_tiles[tile]
        ctx.matched_tile_inners.discard(tile)


def _check_tile_inner_for(
    ctx: _Context, stmt: TileInnerFor, path: str, depth: int
) -> None:
    tile = _check_tile_id(stmt.tile, f"{path}.tile")
    outer = ctx.open_tiles.get(tile)
    if outer is None:
        _fail(
            "unbound_tile",
            path,
            f"tile id {_diagnostic_int(tile.value)} has no dominating "
            "TileOuterFor in scope",
        )
    dimension = _check_tile_loop_dimension(ctx, stmt.dimension, path, "TileInnerFor")
    index = _check_index_id(stmt.index, path, "TileInnerFor.index")
    _check_tile_width(stmt.width, path, "TileInnerFor.width")
    if (
        index != outer.index
        or dimension != outer.dimension
        or stmt.width != outer.width
    ):
        _fail(
            "tile_binding_mismatch",
            path,
            "TileInnerFor must agree with its TileOuterFor on index, "
            "dimension, and width",
        )
    if type(stmt.unroll) is not bool:
        _fail("malformed_state", path, "TileInnerFor.unroll must be a bool")
    bound = _bind_index(
        ctx, stmt.index, path, "TileInnerFor.index", dimension, point_tile=tile
    )
    ctx.matched_tile_inners.add(tile)
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
    finally:
        del ctx.bound_indices[bound]
        del ctx.tile_point_bindings[bound]


def _check_panel_outer_for(
    ctx: _Context, stmt: PanelOuterFor, path: str, depth: int
) -> None:
    tile = _check_tile_id(stmt.tile, f"{path}.tile")
    if tile in ctx.ever_tile_ids:
        _fail(
            "duplicate_tile_id",
            path,
            f"tile id {_diagnostic_int(tile.value)} reused",
        )
    ctx.ever_tile_ids.add(tile)
    dimension = _check_tile_loop_dimension(ctx, stmt.dimension, path, "PanelOuterFor")
    index = _check_index_id(stmt.index, path, "PanelOuterFor.index")
    if index in ctx.bound_indices:
        _fail(
            "tile_index_conflict",
            path,
            f"index {_diagnostic_int(index.value)} is already bound in an "
            "enclosing scope; a panel must own its logical loop",
        )
    if any(open_tile.index == index for open_tile in ctx.open_tiles.values()) or any(
        open_panel.index == index for open_panel in ctx.open_panels.values()
    ):
        _fail(
            "tile_index_conflict",
            path,
            f"index {_diagnostic_int(index.value)} is already split by an "
            "enclosing tile",
        )
    _check_tile_width(stmt.width, path, "PanelOuterFor.width")
    bound_tensor = _check_symbol_id(
        stmt.bound_tensor, f"{path}.bound_tensor", "PanelOuterFor.bound_tensor"
    )
    if bound_tensor not in ctx.tensors:
        _fail(
            "undefined_tensor",
            f"{path}.bound_tensor",
            "panel bound references an undeclared tensor",
        )
    if type(stmt.bound_level) is not int:
        _fail(
            "malformed_state",
            f"{path}.bound_level",
            "PanelOuterFor.bound_level must be an exact int",
        )
    bound_levels = ctx.tensors[bound_tensor].levels
    if not 0 <= stmt.bound_level < len(bound_levels):
        _fail(
            "rank_mismatch",
            f"{path}.bound_level",
            f"level {_diagnostic_int(stmt.bound_level)} outside "
            f"rank-{len(bound_levels)} tensor",
        )
    if bound_levels[stmt.bound_level].kind is not LevelKind.DENSE:
        _fail(
            "panel_bound_mismatch",
            f"{path}.bound_level",
            "the panel bound must name a DENSE storage level; the clamp "
            "extent is a declared dense dimension bound",
        )
    if ctx.level_dimension(bound_tensor, stmt.bound_level) != dimension:
        _fail(
            "panel_bound_mismatch",
            f"{path}.bound_level",
            "the panel bound level must store the panel's own dimension",
        )
    ctx.open_panels[tile] = stmt
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
        if tile not in ctx.matched_panel_windows:
            _fail(
                "missing_panel_window",
                path,
                f"PanelOuterFor for tile id {_diagnostic_int(tile.value)} has "
                "no matching SparseWindowFor in its body",
            )
    finally:
        del ctx.open_panels[tile]
        ctx.matched_panel_windows.discard(tile)


def _check_sparse_window_for(
    ctx: _Context, stmt: SparseWindowFor, path: str, depth: int
) -> None:
    tile = _check_tile_id(stmt.tile, f"{path}.tile")
    panel = ctx.open_panels.get(tile)
    if panel is None:
        _fail(
            "unbound_panel",
            path,
            f"tile id {_diagnostic_int(tile.value)} has no dominating "
            "PanelOuterFor in scope",
        )
    decl = _check_cursor_decl(ctx, stmt.cursor, f"{path}.cursor", depth + 1)
    coord = _check_index_id(stmt.coord_index, path, "SparseWindowFor.coord_index")
    coord_dimension = ctx.level_dimension(decl.tensor, decl.level)
    if coord != panel.index or coord_dimension != panel.dimension:
        _fail(
            "panel_binding_mismatch",
            path,
            "SparseWindowFor must bind its panel's logical index over a "
            "cursor level storing the panel's dimension",
        )
    position = _bind_position(
        ctx, stmt.position, path, "SparseWindowFor.position", decl.tensor, decl.level
    )
    index = _bind_index(
        ctx, stmt.coord_index, path, "SparseWindowFor.coord_index", coord_dimension
    )
    ctx.cursors[decl.cursor] = (decl, None)
    ctx.matched_panel_windows.add(tile)
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
    finally:
        del ctx.bound_indices[index]
        del ctx.bound_positions[position]
        del ctx.cursors[decl.cursor]


def _check_workspace_decl(
    ctx: _Context, decl: object, path: str, depth: int
) -> WorkspaceDecl:
    if type(decl) is not WorkspaceDecl:
        _fail(
            "malformed_state",
            path,
            f"expected a WorkspaceDecl, got {type(decl).__name__}",
        )
    _enter(ctx, decl, path, depth)
    try:
        workspace = _check_workspace_id(decl.workspace, path)
        if workspace in ctx.ever_workspace_ids:
            _fail(
                "duplicate_workspace_id",
                path,
                f"workspace id {_diagnostic_int(workspace.value)} reused",
            )
        ctx.ever_workspace_ids.add(workspace)
        if type(decl.name) is not str or not decl.name:
            _fail("malformed_state", path, "WorkspaceDecl.name must be a nonempty str")
        if type(decl.dtype) is not ScalarType:
            _fail(
                "invalid_scalar_type",
                path,
                "WorkspaceDecl.dtype must be a ScalarType member",
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
        _check_tile_id(decl.tile, f"{path}.tile")
        return decl
    finally:
        _leave(ctx, decl)


def _check_workspace_region(
    ctx: _Context, stmt: WorkspaceRegion, path: str, depth: int
) -> None:
    decl = _check_workspace_decl(ctx, stmt.workspace, f"{path}.workspace", depth + 1)
    outer = ctx.open_tiles.get(decl.tile)
    if outer is None:
        _fail(
            "workspace_scope_mismatch",
            path,
            f"workspace tile id {_diagnostic_int(decl.tile.value)} has no "
            "dominating TileOuterFor in scope; a region needs a current "
            "tile origin",
        )
    if outer.index in ctx.bound_indices:
        _fail(
            "workspace_scope_mismatch",
            path,
            "a workspace region must open outside its own split's point "
            "loops; a per-point workspace could never accumulate",
        )
    state = _WorkspaceState(decl)
    ctx.open_workspaces[decl.workspace] = state
    try:
        state.role = "producer"
        ctx.producer_depth += 1
        try:
            _check_body(ctx, stmt.producer, f"{path}.producer", depth + 1)
        finally:
            ctx.producer_depth -= 1
        state.role = "consumer"
        _check_body(ctx, stmt.consumer, f"{path}.consumer", depth + 1)
        if not state.produced:
            _fail(
                "workspace_dead_region",
                path,
                "a region's producer must reduce into its workspace",
            )
        if not state.consumed:
            _fail(
                "workspace_dead_region",
                path,
                "a region's consumer must read its workspace",
            )
    finally:
        del ctx.open_workspaces[decl.workspace]


def _check_sparse_workspace_decl(
    ctx: _Context, decl: object, path: str, depth: int
) -> SparseWorkspaceDecl:
    if type(decl) is not SparseWorkspaceDecl:
        _fail(
            "malformed_state",
            path,
            f"expected a SparseWorkspaceDecl, got {type(decl).__name__}",
        )
    _enter(ctx, decl, path, depth)
    try:
        workspace = _check_workspace_id(decl.workspace, path)
        if workspace in ctx.ever_workspace_ids:
            _fail(
                "duplicate_workspace_id",
                path,
                f"workspace id {_diagnostic_int(workspace.value)} reused",
            )
        ctx.ever_workspace_ids.add(workspace)
        if type(decl.name) is not str or not decl.name:
            _fail(
                "malformed_state",
                path,
                "SparseWorkspaceDecl.name must be a nonempty str",
            )
        if type(decl.dtype) is not ScalarType:
            _fail(
                "invalid_scalar_type",
                path,
                "SparseWorkspaceDecl.dtype must be a ScalarType member",
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
        dimension = _check_dimension_id(
            decl.drain_dimension,
            f"{path}.drain_dimension",
            "SparseWorkspaceDecl.drain_dimension",
        )
        if dimension not in ctx.dimensions:
            _fail(
                "undefined_dimension",
                f"{path}.drain_dimension",
                "the drain dimension must be a declared dimension",
            )
        return decl
    finally:
        _leave(ctx, decl)


def _check_sparse_workspace_region(
    ctx: _Context, stmt: SparseWorkspaceRegion, path: str, depth: int
) -> None:
    decl = _check_sparse_workspace_decl(
        ctx, stmt.workspace, f"{path}.workspace", depth + 1
    )
    state = _SparseWorkspaceState(decl)
    ctx.open_sparse_workspaces[decl.workspace] = state
    try:
        state.role = "producer"
        ctx.producer_depth += 1
        try:
            _check_body(ctx, stmt.producer, f"{path}.producer", depth + 1)
        finally:
            ctx.producer_depth -= 1
        state.role = "consumer"
        region_state = object.__getattribute__(stmt, "__dict__")
        consumer = (
            region_state.get("consumer", _MISSING)
            if type(region_state) is dict
            else _MISSING
        )
        consumer_state = (
            object.__getattribute__(consumer, "__dict__")
            if type(consumer) is Block
            else None
        )
        consumer_statements = (
            consumer_state.get("statements", _MISSING)
            if type(consumer_state) is dict
            else _MISSING
        )
        if (
            type(consumer) is not Block
            or type(consumer_statements) is not tuple
            or len(consumer_statements) != 1
            or type(consumer_statements[0]) is not SparseWorkspaceDrainFor
        ):
            _fail(
                "workspace_read_scope",
                f"{path}.consumer",
                "a sparse workspace consumer must contain its one ordered "
                "drain directly, outside any repeating control flow",
            )
        _check_body(ctx, consumer, f"{path}.consumer", depth + 1)
        if not state.inserted:
            _fail(
                "workspace_dead_region",
                path,
                "a sparse region's producer must insert into its workspace",
            )
        if not state.drained:
            _fail(
                "workspace_dead_region",
                path,
                "a sparse region's consumer must drain its workspace",
            )
        if not state.consumed:
            _fail(
                "workspace_dead_region",
                path,
                "a sparse region's drain body must consume the current "
                "merged workspace value",
            )
    finally:
        del ctx.open_sparse_workspaces[decl.workspace]


def _check_relayout_decl(
    ctx: _Context, decl: object, path: str, depth: int
) -> RelayoutDecl:
    if type(decl) is not RelayoutDecl:
        _fail(
            "malformed_state",
            path,
            f"expected a RelayoutDecl, got {type(decl).__name__}",
        )
    _enter(ctx, decl, path, depth)
    try:
        relayout = _check_relayout_id(decl.relayout, path)
        if relayout in ctx.ever_relayout_ids:
            _fail(
                "duplicate_relayout_id",
                path,
                f"relayout id {_diagnostic_int(relayout.value)} reused",
            )
        ctx.ever_relayout_ids.add(relayout)
        operand = _check_symbol_id(decl.operand, path, "RelayoutDecl.operand")
        if operand not in ctx.tensors:
            _fail(
                "undefined_tensor",
                path,
                "RelayoutDecl references an undeclared tensor",
            )
        if operand not in ctx.inputs:
            _fail("output_read", path, "staged operands must be declared inputs")
        if type(decl.scope) is not RelayoutScope:
            _fail(
                "malformed_state",
                path,
                "RelayoutDecl.scope must be a RelayoutScope member",
            )
        _check_tile_id(decl.panel, f"{path}.panel")
        _check_tile_id(decl.pack, f"{path}.pack")
        return decl
    finally:
        _leave(ctx, decl)


def _check_relayout_stage(
    ctx: _Context, stmt: RelayoutStage, path: str, depth: int
) -> None:
    decl = _check_relayout_decl(ctx, stmt.decl, f"{path}.decl", depth + 1)
    pack = ctx.open_tiles.get(decl.pack)
    if pack is None:
        _fail(
            "relayout_scope_mismatch",
            path,
            f"relayout pack tile id {_diagnostic_int(decl.pack.value)} has "
            "no dominating TileOuterFor in scope; the staged columns are "
            "the split's current point window",
        )
    if decl.scope is RelayoutScope.PANEL and decl.panel not in ctx.open_panels:
        _fail(
            "relayout_scope_mismatch",
            path,
            "a PANEL-scoped relayout region must open inside its panel's "
            "origin loop; the staged rows are the panel's current window",
        )
    if decl.scope is RelayoutScope.PACK_AXIS and decl.panel in ctx.open_panels:
        _fail(
            "relayout_scope_mismatch",
            path,
            "a PACK_AXIS-scoped relayout region stages every panel row "
            "once per pack origin and must open outside its panel's "
            "origin loop",
        )
    operand_decl = ctx.tensors[decl.operand]
    if len(operand_decl.levels) != 2 or any(
        level.kind is not LevelKind.DENSE for level in operand_decl.levels
    ):
        _fail(
            "relayout_operand_mismatch",
            path,
            "a staged operand must be a rank-2 all-dense tensor",
        )
    if ctx.level_dimension(decl.operand, 1) != pack.dimension:
        _fail(
            "relayout_operand_mismatch",
            path,
            "the staged operand's last storage level must store the pack "
            "split's dimension",
        )
    if decl.scope is RelayoutScope.PANEL:
        panel = ctx.open_panels[decl.panel]
        if ctx.level_dimension(decl.operand, 0) != panel.dimension:
            _fail(
                "relayout_operand_mismatch",
                path,
                "the staged operand's first storage level must store the "
                "panel's dimension",
            )
    state = _RelayoutState(decl)
    ctx.open_relayouts[decl.relayout] = state
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
        if not state.read:
            _fail(
                "relayout_dead_region",
                path,
                "a relayout region's body must read its staged operand "
                "through StagedRead",
            )
    finally:
        del ctx.open_relayouts[decl.relayout]


def _check_result_tile_decl(
    ctx: _Context, decl: object, path: str, depth: int
) -> ResultTileDecl:
    if type(decl) is not ResultTileDecl:
        _fail(
            "malformed_state",
            path,
            f"expected a ResultTileDecl, got {type(decl).__name__}",
        )
    _enter(ctx, decl, path, depth)
    try:
        result_tile = _check_result_tile_id(decl.result_tile, path)
        if result_tile in ctx.ever_result_tile_ids:
            _fail(
                "duplicate_result_tile_id",
                path,
                f"result tile id {_diagnostic_int(result_tile.value)} reused",
            )
        ctx.ever_result_tile_ids.add(result_tile)
        result = _check_symbol_id(decl.result, path, "ResultTileDecl.result")
        if result not in ctx.tensors:
            _fail(
                "undefined_tensor",
                path,
                "ResultTileDecl references an undeclared tensor",
            )
        if result not in ctx.outputs:
            _fail(
                "result_tile_result_mismatch",
                path,
                "a compact result tile accumulates a declared output",
            )
        _check_tile_id(decl.pack, f"{path}.pack")
        return decl
    finally:
        _leave(ctx, decl)


def _check_result_tile_region(
    ctx: _Context, stmt: ResultTileRegion, path: str, depth: int
) -> None:
    decl = _check_result_tile_decl(ctx, stmt.decl, f"{path}.decl", depth + 1)
    pack = ctx.open_tiles.get(decl.pack)
    if pack is None:
        _fail(
            "result_tile_scope_mismatch",
            path,
            f"result tile pack id {_diagnostic_int(decl.pack.value)} has "
            "no dominating TileOuterFor in scope; the compact columns are "
            "the split's current point window",
        )
    if pack.index in ctx.bound_indices:
        _fail(
            "result_tile_scope_mismatch",
            path,
            "a result-tile region must open outside its own split's point "
            "loops; a per-point compact tile could never accumulate",
        )
    # Entry reset and whole-prefix copy-out happen once per pack origin.
    # Therefore the pack origin must be the root computation statement and
    # this region must be a direct child of its body.  An intervening or
    # enclosing repeating loop would reset/copy the complete prefix space
    # several times and erase contributions from earlier iterations.
    ancestry = ctx.statement_stack
    if (
        len(ancestry) != 4
        or type(ancestry[0]) is not Block
        or ancestry[-3] is not pack
        or ancestry[-2] is not pack.body
        or ancestry[-1] is not stmt
    ):
        _fail(
            "result_tile_scope_mismatch",
            path,
            "a result-tile region must execute exactly once per outermost "
            "pack origin as a direct statement of that origin's body",
        )
    if decl.result in ctx.open_result_tile_results:
        _fail(
            "result_tile_scope_mismatch",
            path,
            "a result already accumulates through an open result-tile "
            "region; nested regions of one result conflict at copy-out",
        )
    result_decl = ctx.tensors[decl.result]
    if len(result_decl.levels) < 2 or any(
        level.kind is not LevelKind.DENSE for level in result_decl.levels
    ):
        _fail(
            "result_tile_result_mismatch",
            path,
            "a compact result tile requires an all-dense result of rank "
            "at least two",
        )
    if ctx.level_dimension(decl.result, len(result_decl.levels) - 1) != pack.dimension:
        _fail(
            "result_tile_result_mismatch",
            path,
            "the compact result's last storage level must store the pack "
            "split's dimension",
        )
    state = _ResultTileState(decl)
    ctx.open_result_tiles[decl.result_tile] = state
    ctx.open_result_tile_results.add(decl.result)
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
        if not state.reduced:
            _fail(
                "result_tile_dead_region",
                path,
                "a result-tile region's body must accumulate through " "TiledReduce",
            )
    finally:
        del ctx.open_result_tiles[decl.result_tile]
        ctx.open_result_tile_results.discard(decl.result)


def _check_tiled_reduce(
    ctx: _Context, stmt: TiledReduce, path: str, depth: int
) -> None:
    result_tile = _check_result_tile_id(stmt.result_tile, path)
    state = ctx.open_result_tiles.get(result_tile)
    if state is None:
        _fail(
            "unbound_result_tile",
            path,
            f"result tile {_diagnostic_int(result_tile.value)} has no "
            "enclosing region in scope",
        )
    decl = state.decl
    _require_outside_producer(ctx, path)
    if type(stmt.op) is not ReduceOp:
        # ADD is the only declared ReduceOp member, and its identity is
        # exactly the zero the owning region's entry reset established —
        # the reduction-legality contract.  Adding a member requires adding
        # its explicit reset-identity contract here.
        _fail("malformed_state", path, "TiledReduce.op must be a ReduceOp member")
    result_decl = ctx.tensors[decl.result]
    if type(stmt.indices) is not tuple:
        _fail("malformed_state", path, "TiledReduce.indices must be an owned tuple")
    if len(stmt.indices) != len(result_decl.levels):
        _fail(
            "rank_mismatch",
            path,
            f"{len(stmt.indices)} indices for rank-{len(result_decl.levels)} " "result",
        )
    for position, index in enumerate(stmt.indices):
        index_type = _check_expr(ctx, index, f"{path}.indices[{position}]", depth + 1)
        _require_coord(
            ctx,
            index_type,
            f"{path}.indices[{position}]",
            "a tiled reduce index",
            result_decl.dimensions[position],
        )
    point_index = stmt.indices[result_decl.levels[-1].mode]
    if (
        type(point_index) is not IndexValue
        or ctx.tile_point_bindings.get(point_index.index) != decl.pack
    ):
        _fail(
            "result_tile_write_mismatch",
            path,
            "the tiled reduce's trailing index must be the pack split's "
            "point coordinate, bound by its TileInnerFor",
        )
    value_type = _check_expr(ctx, stmt.value, f"{path}.value", depth + 1)
    _require_value(value_type, f"{path}.value", "a combined value")
    state.reduced = True
    ctx.written_outputs.add(decl.result)


def _check_workspace_reduce(
    ctx: _Context, stmt: WorkspaceReduce, path: str, depth: int
) -> None:
    workspace = _check_workspace_id(stmt.workspace, path)
    state = ctx.open_workspaces.get(workspace)
    if state is None:
        _fail(
            "unbound_workspace",
            path,
            f"workspace {_diagnostic_int(workspace.value)} has no enclosing "
            "region in scope",
        )
    if state.role != "producer":
        _fail(
            "workspace_write_scope",
            path,
            "a workspace is writable only inside its region's producer",
        )
    if type(stmt.op) is not ReduceOp:
        # ADD is the only declared ReduceOp member; its identity is exactly
        # the zero the region-entry reset established.  Adding a member
        # requires adding its explicit reset-identity contract here.
        _fail("malformed_state", path, "WorkspaceReduce.op must be a ReduceOp member")
    _check_workspace_coord(ctx, state, stmt.coord, f"{path}.coord", depth + 1)
    value_type = _check_expr(ctx, stmt.value, f"{path}.value", depth + 1)
    _require_value(value_type, f"{path}.value", "a combined value")
    state.produced = True


def _check_sparse_workspace_insert(
    ctx: _Context, stmt: SparseWorkspaceInsert, path: str, depth: int
) -> None:
    workspace = _check_workspace_id(stmt.workspace, path)
    state = ctx.open_sparse_workspaces.get(workspace)
    if state is None:
        _fail(
            "unbound_workspace",
            path,
            f"sparse workspace {_diagnostic_int(workspace.value)} has no "
            "enclosing region in scope",
        )
    if state.role != "producer":
        _fail(
            "workspace_write_scope",
            path,
            "a sparse workspace accepts insertions only inside its "
            "region's producer",
        )
    if type(stmt.op) is not ReduceOp:
        # ADD is the only declared ReduceOp member; its identity is exactly
        # the absent entry the region's empty-entry contract established.
        _fail(
            "malformed_state",
            path,
            "SparseWorkspaceInsert.op must be a ReduceOp member",
        )
    coord_type = _check_expr(ctx, stmt.coord, f"{path}.coord", depth + 1)
    _require_coord(
        ctx,
        coord_type,
        f"{path}.coord",
        "a sparse-workspace insertion coordinate",
        state.decl.drain_dimension,
    )
    value_type = _check_expr(ctx, stmt.value, f"{path}.value", depth + 1)
    _require_value(value_type, f"{path}.value", "an inserted value")
    state.inserted = True


def _check_sparse_workspace_drain_for(
    ctx: _Context, stmt: SparseWorkspaceDrainFor, path: str, depth: int
) -> None:
    workspace = _check_workspace_id(stmt.workspace, path)
    state = ctx.open_sparse_workspaces.get(workspace)
    if state is None:
        _fail(
            "unbound_workspace",
            path,
            f"sparse workspace {_diagnostic_int(workspace.value)} has no "
            "enclosing region in scope",
        )
    if state.role != "consumer":
        _fail(
            "workspace_read_scope",
            path,
            "a sparse workspace drains only inside its region's consumer",
        )
    if state.drained or state.drain_depth != 0:
        _fail(
            "workspace_read_scope",
            path,
            "a sparse workspace drains at most once per region",
        )
    index = _bind_index(
        ctx,
        stmt.index,
        path,
        "SparseWorkspaceDrainFor.index",
        state.decl.drain_dimension,
    )
    state.drain_depth += 1
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
    finally:
        state.drain_depth -= 1
        del ctx.bound_indices[index]
    state.drained = True


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
    if type(stmt.positions) is not tuple:
        _fail("malformed_state", path, "positions must be an owned tuple")
    if stmt.positions:
        if len(stmt.positions) != len(stmt.cursors):
            _fail(
                "malformed_state",
                path,
                "a position-binding merge names one entry per cursor",
            )
        if all(bound is None for bound in stmt.positions):
            _fail(
                "malformed_state",
                path,
                "a nonempty merge position tuple must bind at least one "
                "position; use the canonical empty tuple otherwise",
            )
        if stmt.mode is MergeMode.UNION:
            if any(bound is None for bound in stmt.positions):
                _fail(
                    "unsupported_sparse_hierarchy",
                    path,
                    "a position-binding UNION merge must bind every "
                    "cursor's position: one-sided descent needs each "
                    "operand's own anchor",
                )
        elif stmt.mode is not MergeMode.INTERSECTION:
            _fail(
                "unsupported_sparse_hierarchy",
                path,
                "merge descent binds aligned positions only under "
                "INTERSECTION or UNION",
            )
    decls = []
    for position, cursor in enumerate(stmt.cursors):
        cursor_path = f"{path}.cursors[{position}]"
        decl = _check_cursor_decl(ctx, cursor, cursor_path, depth + 1)
        binds_position = bool(stmt.positions) and stmt.positions[position] is not None
        if (
            decl.level != len(ctx.tensors[decl.tensor].levels) - 1
            and not binds_position
        ):
            _fail(
                "unsupported_sparse_hierarchy",
                cursor_path,
                "merged cursors must target the value-bearing leaf level "
                "unless the merge binds their position as a descent anchor "
                "(aligned under INTERSECTION; per-cursor under UNION, "
                "where an unaligned parent yields the empty child stream)",
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
    bound_merge_positions = []
    for cursor_position, position_id in enumerate(stmt.positions):
        if position_id is None:
            continue
        decl = decls[cursor_position]
        bound_merge_positions.append(
            _bind_position(
                ctx,
                position_id,
                f"{path}.positions[{cursor_position}]",
                "MergedSparseFor.positions entry",
                decl.tensor,
                decl.level,
            )
        )
    for decl in decls:
        ctx.cursors[decl.cursor] = (decl, stmt.mode)
    try:
        _check_body(ctx, stmt.body, f"{path}.body", depth + 1)
    finally:
        del ctx.bound_indices[index]
        for bound in bound_merge_positions:
            del ctx.bound_positions[bound]
        for decl in decls:
            del ctx.cursors[decl.cursor]


def _require_outside_producer(ctx: _Context, path: str) -> None:
    """A region's producer owns its workspace; output writes are consumer work."""

    if ctx.producer_depth > 0:
        _fail(
            "workspace_output_write",
            path,
            "declared outputs must not be written inside a workspace "
            "region's producer",
        )


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


def _require_dense_store_target(ctx: _Context, tensor: SymbolId, path: str) -> None:
    decl = ctx.tensors[tensor]
    if any(level.kind is not LevelKind.DENSE for level in decl.levels):
        _fail(
            "layout_mismatch",
            path,
            "coordinate stores are only defined on all-dense outputs",
        )


def _require_no_open_result_tile(ctx: _Context, tensor: SymbolId, path: str) -> None:
    """Direct writes conflict with an open region's exactly-once copy-out."""

    if tensor in ctx.open_result_tile_results:
        _fail(
            "result_tile_residual_write",
            path,
            "a result accumulating through an open result-tile region must "
            "not also be written directly; copy-out would overwrite the "
            "direct write",
        )


def _check_store(ctx: _Context, stmt: Store, path: str, depth: int) -> None:
    tensor = _check_symbol_id(stmt.tensor, path, "Store.tensor")
    if tensor not in ctx.tensors:
        _fail("undefined_tensor", path, "Store references an undeclared tensor")
    if tensor not in ctx.outputs:
        _fail("output_scope", path, "Store may only write declared outputs")
    _require_outside_producer(ctx, path)
    _require_no_open_result_tile(ctx, tensor, path)
    _require_dense_store_target(ctx, tensor, path)
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
    _require_outside_producer(ctx, path)
    _require_no_open_result_tile(ctx, tensor, path)
    if type(stmt.op) is not ReduceOp:
        # ADD is the only declared ReduceOp member, so an exact member check
        # is the whole reduction-operator contract; adding a member requires
        # adding its explicit output-initialization identity check here.
        _fail("malformed_state", path, "StoreReduce.op must be a ReduceOp member")
    decl = ctx.tensors[tensor]
    if any(
        level.kind is not LevelKind.DENSE and level.kind is not LevelKind.COMPRESSED
        for level in decl.levels
    ):
        _fail(
            "layout_mismatch",
            path,
            "reductions are only defined on DENSE/COMPRESSED-level outputs",
        )
    # An all-dense target is the executable dense reduction.  A target with
    # compressed levels is the SEMANTIC accumulation form of the sparse
    # result family: the oracle merges it by coordinate, while target
    # lowering refuses it unless a sparse-workspace schedule rewrites it
    # into ordered assembly.
    _check_output_write_indices(ctx, stmt, tensor, path, depth)
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
    _require_outside_producer(ctx, path)
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
    TileOuterFor: _check_tile_outer_for,
    TileInnerFor: _check_tile_inner_for,
    PanelOuterFor: _check_panel_outer_for,
    SparseWindowFor: _check_sparse_window_for,
    SparseFor: _check_sparse_for,
    MergedSparseFor: _check_merged_sparse_for,
    WorkspaceRegion: _check_workspace_region,
    WorkspaceReduce: _check_workspace_reduce,
    SparseWorkspaceRegion: _check_sparse_workspace_region,
    SparseWorkspaceInsert: _check_sparse_workspace_insert,
    SparseWorkspaceDrainFor: _check_sparse_workspace_drain_for,
    RelayoutStage: _check_relayout_stage,
    ResultTileRegion: _check_result_tile_region,
    TiledReduce: _check_tiled_reduce,
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
                f"dimension {_diagnostic_int(dimension.value)} declared more "
                "than once",
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
                "this subset fails closed on them until a later phase "
                "represents their iteration",
            )
        if type(decl.mode) is not int:
            _fail("malformed_state", path, "LevelDecl.mode must be an exact int")
        if not 0 <= decl.mode < rank:
            _fail(
                "invalid_mode_order",
                path,
                f"mode {_diagnostic_int(decl.mode)} outside the rank-{rank} "
                "logical modes",
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
            _fail(
                "duplicate_symbol",
                path,
                f"tensor symbol {_diagnostic_int(symbol.value)} redeclared",
            )
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


def _walk_declared_schema(root: object) -> List[object]:
    """Every schema node reachable through declared fields, root included.

    The walked structure was already admitted by the ordinary traversal, so
    aliasing, cycles, and depth are bounded; this is the shared read-only
    walk the parallel-selection legality analysis performs over verified
    statements.
    """

    collected: List[object] = []
    seen: Set[int] = set()
    pending: List[object] = [root]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        collected.append(node)
        children: List[LoopIRNode] = []
        for field in fields(type(node)):  # type: ignore[arg-type]
            value = getattr(node, field.name, None)
            if isinstance(value, LoopIRNode):
                children.append(value)
            elif type(value) is tuple:
                children.extend(
                    child for child in value if isinstance(child, LoopIRNode)
                )
        # The stack is LIFO: reverse once so fields and tuple members are
        # visited in their declared/source order.  Parallel work ownership
        # intentionally follows the first sparse initializer the LLIR target
        # emits and ``find_sparse_pos_array`` discovers.
        pending.extend(reversed(children))
    return collected


def _parallel_dense_bound_driver(
    program: LoopProgram,
    nodes: List[object],
    index: IndexId,
) -> Optional[Tuple[SymbolId, int]]:
    """Reproduce the exact first dense input bound the target will spell."""

    decls = {decl.symbol: decl for decl in program.tensors}
    for symbol in program.inputs:
        decl = decls[symbol]
        for level, level_decl in enumerate(decl.levels):
            if level_decl.kind is not LevelKind.DENSE:
                continue
            logical_mode = level_decl.mode
            for node in nodes:
                if type(node) is Load and node.tensor == symbol:
                    coordinate = node.indices[logical_mode]
                    if type(coordinate) is IndexValue and coordinate.index == index:
                        return symbol, level
                if (
                    type(node) is DensePosition
                    and node.tensor == symbol
                    and node.level == level
                    and type(node.coord) is IndexValue
                    and node.coord.index == index
                ):
                    return symbol, level
    return None


def _canonical_parallel_work_source(
    program: LoopProgram,
    target: Stmt,
    index: IndexId,
) -> Optional[Tuple[SymbolId, int]]:
    """Derive the one canonical sparse work source for a selected loop.

    The legacy-compatible work policy can use a compressed position total
    only when the first visible sparse cursor is immediately parented by the
    selected coordinate *and* that dense parent is the exact input
    tensor/level whose extent the target uses to spell the selected loop
    bound.  Equal logical extents are insufficient for compatibility:
    legacy sparse-position discovery is tied to that physical driver and
    falls back to its row-count-only policy otherwise.  A merged loop
    likewise forces the row-count-only legacy policy because established
    merged-nest lowering hides its iterator initialization from discovery.
    Returning one value (or ``None``) makes ``ParallelWork`` re-derived
    program state rather than an unchecked policy choice.
    """

    subtree = _walk_declared_schema(target)
    if any(type(node) is MergedSparseFor for node in subtree):
        return None
    cursor: Optional[SparseCursorDecl] = None
    for node in subtree:
        if type(node) in (SparseFor, SparseWindowFor):
            cursor = cast(SparseFor, node).cursor
            break
    if cursor is None or cursor.level < 1:
        return None
    parent = cursor.parent
    if (
        type(parent) is not DensePosition
        or parent.tensor != cursor.tensor
        or parent.level != cursor.level - 1
        or type(parent.coord) is not IndexValue
        or parent.coord.index != index
    ):
        return None
    decls = {decl.symbol: decl for decl in program.tensors}
    decl = decls[cursor.tensor]
    if decl.levels[parent.level].kind is not LevelKind.DENSE:
        return None
    all_nodes = _walk_declared_schema(program.body)
    if _parallel_dense_bound_driver(program, all_nodes, index) != (
        cursor.tensor,
        parent.level,
    ):
        return None
    return cursor.tensor, cursor.level


def _parallel_target_matches(node: object, index: IndexId, part: ParallelPart) -> bool:
    """Whether one statement binds the selected ``(index, part)`` identity."""

    if part is ParallelPart.LOGICAL:
        if type(node) is DenseFor:
            return node.index == index
        if type(node) is SparseFor:
            return node.coord_index == index
        if type(node) is MergedSparseFor:
            return node.coord_index == index
        return False
    if type(node) is TileOuterFor:
        return node.index == index
    if type(node) is PanelOuterFor:
        return node.index == index
    return False


def _selected_coordinate_addresses(indices: object, index: IndexId) -> bool:
    """Whether one write's index tuple carries the selected coordinate."""

    if type(indices) is not tuple:
        return False
    return any(type(entry) is IndexValue and entry.index == index for entry in indices)


def _check_parallel_selection(ctx: _Context, program: LoopProgram) -> None:
    """Validate the program's abstract parallel selection fail-closed.

    The selection is program semantics: its stored fields are exact, its
    loop must resolve uniquely by ``(index, part)`` identity, its work
    estimate must restate the resolved loop's structure, and its declared
    race discipline must be re-proved from the program — never trusted.
    """

    selection = program.parallel
    path = "program.parallel"
    if type(selection) is not ParallelSelection:
        _fail(
            "invalid_parallel_selection",
            path,
            f"expected a ParallelSelection, got {type(selection).__name__}",
        )
    _enter(ctx, selection, path, 1)
    try:
        index = _check_index_id(selection.index, path, "ParallelSelection.index")
        if type(selection.part) is not ParallelPart or not any(
            selection.part is member for member in ParallelPart
        ):
            _fail(
                "invalid_parallel_selection",
                f"{path}.part",
                "part must be a canonical ParallelPart member",
            )
        if type(selection.discipline) is not ParallelDiscipline or not any(
            selection.discipline is member for member in ParallelDiscipline
        ):
            _fail(
                "invalid_parallel_selection",
                f"{path}.discipline",
                "discipline must be a canonical ParallelDiscipline member",
            )
        if type(selection.intent) is not ParallelIntent or not any(
            selection.intent is member for member in ParallelIntent
        ):
            _fail(
                "invalid_parallel_selection",
                f"{path}.intent",
                "intent must be a canonical ParallelIntent member",
            )
        work = selection.work
        if type(work) is not ParallelWork:
            _fail(
                "invalid_parallel_selection",
                f"{path}.work",
                f"expected a ParallelWork, got {type(work).__name__}",
            )
        _enter(ctx, work, f"{path}.work", 2)
        try:
            rows = _check_dimension_id(
                work.rows, f"{path}.work.rows", "the work rows source"
            )
            if rows not in ctx.dimensions:
                _fail(
                    "invalid_parallel_selection",
                    f"{path}.work.rows",
                    "work rows must name a declared dimension",
                )
            nnz = work.nnz
            if nnz is not None:
                if type(nnz) is not SparseWorkSource:
                    _fail(
                        "invalid_parallel_selection",
                        f"{path}.work.nnz",
                        f"expected a SparseWorkSource, got {type(nnz).__name__}",
                    )
                _enter(ctx, nnz, f"{path}.work.nnz", 3)
                try:
                    tensor = _check_symbol_id(
                        nnz.tensor, f"{path}.work.nnz", "the work tensor"
                    )
                    decl = ctx.tensors.get(tensor)
                    if decl is None:
                        _fail(
                            "invalid_parallel_selection",
                            f"{path}.work.nnz",
                            "work source must name a declared tensor",
                        )
                    if (
                        type(nnz.level) is not int
                        or isinstance(nnz.level, bool)
                        or not 0 <= nnz.level < len(decl.levels)
                        or decl.levels[nnz.level].kind is not LevelKind.COMPRESSED
                    ):
                        _fail(
                            "invalid_parallel_selection",
                            f"{path}.work.nnz",
                            "work source must name a compressed level of its " "tensor",
                        )
                finally:
                    _leave(ctx, nnz)
        finally:
            _leave(ctx, work)

        matches = [
            node
            for node in _walk_declared_schema(program.body)
            if isinstance(node, Stmt)
            and _parallel_target_matches(node, index, selection.part)
        ]
        if not matches:
            _fail(
                "parallel_target_missing",
                path,
                "no loop in the program binds the selected (index, part) " "identity",
            )
        # The ordinary binding discipline makes the identity unique on a
        # verified program: ``duplicate_index_binding`` rejects any second
        # LOGICAL binding of one index anywhere in the program, and the
        # split-ownership rules give every OUTER origin a unique owner.
        target = matches[0]

        if type(target) is DenseFor or type(target) is TileOuterFor:
            loop_dimension = target.dimension
        elif type(target) is PanelOuterFor:
            loop_dimension = target.dimension
        elif type(target) is SparseFor:
            loop_dimension = ctx.level_dimension(
                target.cursor.tensor, target.cursor.level
            )
        else:
            first_cursor = cast(MergedSparseFor, target).cursors[0]
            loop_dimension = ctx.level_dimension(
                first_cursor.tensor, first_cursor.level
            )
        if type(target) in (SparseFor, MergedSparseFor):
            _fail(
                "parallel_work_mismatch",
                path,
                "sparse position loops do not have the declared dimension "
                "as their trip count; this work model selects dense logical "
                "loops or affine origin loops only",
            )
        if work.rows != loop_dimension:
            _fail(
                "parallel_work_mismatch",
                f"{path}.work.rows",
                "the work rows source must be the selected loop's declared "
                "dimension",
            )

        subtree = _walk_declared_schema(target)
        expected_source = _canonical_parallel_work_source(program, target, index)
        actual_source = None if work.nnz is None else (work.nnz.tensor, work.nnz.level)
        if actual_source != expected_source:
            _fail(
                "parallel_work_mismatch",
                f"{path}.work.nnz",
                "the sparse work source must be the selected loop's one "
                "canonical, directly parented structural work source",
            )

        regions = [
            node
            for node in _walk_declared_schema(program.body)
            if type(node) is ResultTileRegion
        ]
        if selection.discipline is ParallelDiscipline.RESULT_PARTITION:
            if regions:
                _fail(
                    "parallel_race",
                    path,
                    "a heap result-tile program partitions compact cells; "
                    "the result-partition discipline does not apply",
                )
        else:
            if len(regions) != 1:
                _fail(
                    "parallel_race",
                    path,
                    "the compact-partition discipline requires exactly one "
                    "heap result-tile region",
                )
            region = regions[0]
            region_nodes = _walk_declared_schema(region.body)
            if not any(node is target for node in region_nodes):
                _fail(
                    "parallel_race",
                    path,
                    "the selected loop must execute inside the heap region; "
                    "loops outside it share the reusable compact storage",
                )
            reduces = [
                node
                for node in region_nodes
                if type(node) is TiledReduce
                and node.result_tile == region.decl.result_tile
            ]
            if not reduces or not all(
                _selected_coordinate_addresses(node.indices, index) for node in reduces
            ):
                _fail(
                    "parallel_race",
                    path,
                    "the selected coordinate must address every compact "
                    "accumulation of the heap region",
                )

        workspaces_inside = {
            node.workspace.workspace
            for node in subtree
            if type(node) is WorkspaceRegion
        }
        sparse_workspaces_inside = {
            node.workspace.workspace
            for node in subtree
            if type(node) is SparseWorkspaceRegion
        }
        for node in subtree:
            if type(node) is AppendEntry:
                _fail(
                    "parallel_race",
                    path,
                    "ordered sparse assembly cannot be partitioned by a "
                    "parallel selection",
                )
            if type(node) in (Store, StoreReduce):
                if not _selected_coordinate_addresses(cast(Store, node).indices, index):
                    _fail(
                        "parallel_race",
                        path,
                        "a write inside the selected loop is not addressed "
                        "by the selected coordinate",
                    )
            if type(node) in (WorkspaceRead, WorkspaceReduce):
                if cast(WorkspaceRead, node).workspace not in workspaces_inside:
                    _fail(
                        "parallel_race",
                        path,
                        "a workspace shared across selected iterations is "
                        "not race free",
                    )
            if type(node) in (
                SparseWorkspaceInsert,
                SparseWorkspaceDrainFor,
                SparseWorkspaceValue,
            ):
                if (
                    cast(SparseWorkspaceInsert, node).workspace
                    not in sparse_workspaces_inside
                ):
                    _fail(
                        "parallel_race",
                        path,
                        "a sparse workspace shared across selected "
                        "iterations is not race free",
                    )
    finally:
        _leave(ctx, selection)


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
        for position, symbol in enumerate(program.outputs):
            decl = ctx.tensors[symbol]
            kinds = tuple(level.kind for level in decl.levels)
            if all(kind is LevelKind.DENSE for kind in kinds):
                continue
            modes = tuple(level.mode for level in decl.levels)
            if any(
                kind is not LevelKind.DENSE and kind is not LevelKind.COMPRESSED
                for kind in kinds
            ) or modes != tuple(range(len(kinds))):
                # The generalized ordered-assembly stream admits identity-
                # ordered DENSE/COMPRESSED output levels; every other sparse
                # output layout stays fail-closed.
                _fail(
                    "unsupported_sparse_output",
                    f"program.outputs[{position}]",
                    f"output {decl.name!r} declares a sparse layout outside "
                    "the identity-ordered DENSE/COMPRESSED families; this "
                    "subset fails closed on it",
                )
        _check_body(ctx, program.body, "program.body", 1)
        unwritten = ctx.outputs - ctx.written_outputs
        if unwritten:
            _fail(
                "unwritten_output",
                "program.outputs",
                "an output is never stored to",
            )
        if program.parallel is not None:
            _check_parallel_selection(ctx, program)
    finally:
        _leave(ctx, program)
