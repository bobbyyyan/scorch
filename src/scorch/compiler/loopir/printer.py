"""Deterministic printing and canonical serialization for production LoopIR.

Both surfaces operate only on verified programs (they call the fail-closed
verifier first) and never depend on allocation-order identity values, memory
addresses, or dictionary ordering:

- :func:`print_program` renders a human-readable stage dump.  Dimensions,
  tensors, and loop indices are shown with traversal-canonical local labels
  (``d0``/``t0``/``x0``) plus their display names, so two independently
  constructed equivalent programs — including ones whose global
  ``SymbolId``/``IndexId`` allocation histories differ — print identically.
- :func:`canonical_program_dump` serializes the complete semantic content as
  compact JSON with traversal-renumbered identities and an explicit schema
  version.  Display names are deliberately omitted: they are presentation,
  not semantics, and must not enter cache or fingerprint identity.  There is
  currently no deserializer, so no round-trip contract is declared.
"""

from __future__ import annotations

import json
from typing import Dict, List

from ..identity import IndexId, SymbolId
from .nodes import (
    AppendEntry,
    BinaryExpr,
    Block,
    CursorId,
    CursorValue,
    DenseFor,
    DensePosition,
    DimensionId,
    Expr,
    FloatConst,
    IndexValue,
    Load,
    LoopProgram,
    MergedSparseFor,
    PanelOuterFor,
    ParallelSelection,
    PositionId,
    PositionValue,
    RelayoutId,
    RelayoutStage,
    ResultTileId,
    ResultTileRegion,
    RootPosition,
    SparseCursorDecl,
    SparseFor,
    SparseWindowFor,
    SparseWorkspaceDrainFor,
    SparseWorkspaceInsert,
    SparseWorkspaceRegion,
    SparseWorkspaceValue,
    StagedRead,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
    TiledReduce,
    TileId,
    TileInnerFor,
    TileOuterFor,
    WorkspaceId,
    WorkspaceRead,
    WorkspaceReduce,
    WorkspaceRegion,
)
from .verifier import verify_program

CANONICAL_SCHEMA = "scorch.loopir.canonical.v9"


class _CanonicalIds:
    """First-appearance renumbering for every identity family."""

    def __init__(self) -> None:
        self._dimensions: Dict[DimensionId, int] = {}
        self._symbols: Dict[SymbolId, int] = {}
        self._indices: Dict[IndexId, int] = {}
        self._cursors: Dict[CursorId, int] = {}
        self._positions: Dict[PositionId, int] = {}
        self._tiles: Dict[TileId, int] = {}
        self._workspaces: Dict[WorkspaceId, int] = {}
        self._relayouts: Dict[RelayoutId, int] = {}
        self._result_tiles: Dict[ResultTileId, int] = {}

    def dimension(self, dimension: DimensionId) -> int:
        return self._dimensions.setdefault(dimension, len(self._dimensions))

    def symbol(self, symbol: SymbolId) -> int:
        return self._symbols.setdefault(symbol, len(self._symbols))

    def index(self, index: IndexId) -> int:
        return self._indices.setdefault(index, len(self._indices))

    def cursor(self, cursor: CursorId) -> int:
        return self._cursors.setdefault(cursor, len(self._cursors))

    def position(self, position: PositionId) -> int:
        return self._positions.setdefault(position, len(self._positions))

    def tile(self, tile: TileId) -> int:
        return self._tiles.setdefault(tile, len(self._tiles))

    def workspace(self, workspace: WorkspaceId) -> int:
        return self._workspaces.setdefault(workspace, len(self._workspaces))

    def relayout(self, relayout: RelayoutId) -> int:
        return self._relayouts.setdefault(relayout, len(self._relayouts))

    def result_tile(self, result_tile: ResultTileId) -> int:
        return self._result_tiles.setdefault(result_tile, len(self._result_tiles))


def _seed_expr_ids(expr: Expr, ids: _CanonicalIds) -> None:
    if type(expr) is IndexValue:
        ids.index(expr.index)
        return
    if type(expr) is FloatConst:
        return
    if type(expr) is RootPosition:
        return
    if type(expr) is DensePosition:
        ids.symbol(expr.tensor)
        _seed_expr_ids(expr.parent, ids)
        _seed_expr_ids(expr.coord, ids)
        return
    if type(expr) is PositionValue:
        ids.position(expr.position)
        return
    if type(expr) is CursorValue:
        ids.cursor(expr.cursor)
        if expr.default is not None:
            _seed_expr_ids(expr.default, ids)
        return
    if type(expr) is Load:
        ids.symbol(expr.tensor)
        for index in expr.indices:
            _seed_expr_ids(index, ids)
        return
    if type(expr) is WorkspaceRead:
        ids.workspace(expr.workspace)
        _seed_expr_ids(expr.coord, ids)
        return
    if type(expr) is StagedRead:
        ids.relayout(expr.relayout)
        for index in expr.indices:
            _seed_expr_ids(index, ids)
        return
    if type(expr) is SparseWorkspaceValue:
        ids.workspace(expr.workspace)
        return
    if type(expr) is BinaryExpr:
        _seed_expr_ids(expr.lhs, ids)
        _seed_expr_ids(expr.rhs, ids)
        return
    raise TypeError(f"unsupported LoopIR expression {type(expr).__name__}")


def _seed_cursor_ids(cursor: SparseCursorDecl, ids: _CanonicalIds) -> None:
    ids.cursor(cursor.cursor)
    ids.symbol(cursor.tensor)
    _seed_expr_ids(cursor.parent, ids)


def _seed_stmt_ids(stmt: Stmt, ids: _CanonicalIds) -> None:
    if type(stmt) is Block:
        for child in stmt.statements:
            _seed_stmt_ids(child, ids)
        return
    if type(stmt) is DenseFor:
        ids.index(stmt.index)
        ids.dimension(stmt.dimension)
        _seed_stmt_ids(stmt.body, ids)
        return
    if type(stmt) is TileOuterFor or type(stmt) is TileInnerFor:
        ids.tile(stmt.tile)
        ids.index(stmt.index)
        ids.dimension(stmt.dimension)
        _seed_stmt_ids(stmt.body, ids)
        return
    if type(stmt) is PanelOuterFor:
        ids.tile(stmt.tile)
        ids.index(stmt.index)
        ids.dimension(stmt.dimension)
        ids.symbol(stmt.bound_tensor)
        _seed_stmt_ids(stmt.body, ids)
        return
    if type(stmt) is SparseWindowFor:
        ids.tile(stmt.tile)
        _seed_cursor_ids(stmt.cursor, ids)
        ids.position(stmt.position)
        ids.index(stmt.coord_index)
        _seed_stmt_ids(stmt.body, ids)
        return
    if type(stmt) is SparseFor:
        _seed_cursor_ids(stmt.cursor, ids)
        ids.position(stmt.position)
        ids.index(stmt.coord_index)
        _seed_stmt_ids(stmt.body, ids)
        return
    if type(stmt) is MergedSparseFor:
        for cursor in stmt.cursors:
            _seed_cursor_ids(cursor, ids)
        ids.index(stmt.coord_index)
        _seed_stmt_ids(stmt.body, ids)
        return
    if type(stmt) is WorkspaceRegion:
        ids.workspace(stmt.workspace.workspace)
        ids.tile(stmt.workspace.tile)
        _seed_stmt_ids(stmt.producer, ids)
        _seed_stmt_ids(stmt.consumer, ids)
        return
    if type(stmt) is WorkspaceReduce:
        ids.workspace(stmt.workspace)
        _seed_expr_ids(stmt.coord, ids)
        _seed_expr_ids(stmt.value, ids)
        return
    if type(stmt) is SparseWorkspaceRegion:
        ids.workspace(stmt.workspace.workspace)
        ids.dimension(stmt.workspace.drain_dimension)
        _seed_stmt_ids(stmt.producer, ids)
        _seed_stmt_ids(stmt.consumer, ids)
        return
    if type(stmt) is SparseWorkspaceInsert:
        ids.workspace(stmt.workspace)
        _seed_expr_ids(stmt.coord, ids)
        _seed_expr_ids(stmt.value, ids)
        return
    if type(stmt) is SparseWorkspaceDrainFor:
        ids.workspace(stmt.workspace)
        ids.index(stmt.index)
        _seed_stmt_ids(stmt.body, ids)
        return
    if type(stmt) is RelayoutStage:
        ids.relayout(stmt.decl.relayout)
        ids.symbol(stmt.decl.operand)
        ids.tile(stmt.decl.panel)
        ids.tile(stmt.decl.pack)
        _seed_stmt_ids(stmt.body, ids)
        return
    if type(stmt) is ResultTileRegion:
        ids.result_tile(stmt.decl.result_tile)
        ids.symbol(stmt.decl.result)
        ids.tile(stmt.decl.pack)
        _seed_stmt_ids(stmt.body, ids)
        return
    if type(stmt) is TiledReduce:
        ids.result_tile(stmt.result_tile)
        for index in stmt.indices:
            _seed_expr_ids(index, ids)
        _seed_expr_ids(stmt.value, ids)
        return
    if type(stmt) is Store:
        ids.symbol(stmt.tensor)
        for index in stmt.indices:
            _seed_expr_ids(index, ids)
        _seed_expr_ids(stmt.value, ids)
        return
    if type(stmt) is StoreReduce:
        ids.symbol(stmt.tensor)
        for index in stmt.indices:
            _seed_expr_ids(index, ids)
        _seed_expr_ids(stmt.value, ids)
        return
    if type(stmt) is AppendEntry:
        ids.symbol(stmt.tensor)
        for coord in stmt.coords:
            _seed_expr_ids(coord, ids)
        _seed_expr_ids(stmt.value, ids)
        return
    raise TypeError(f"unsupported LoopIR statement {type(stmt).__name__}")


def _canonical_ids(program: LoopProgram) -> _CanonicalIds:
    """Assign labels from semantic roles and body traversal, not registries."""

    ids = _CanonicalIds()
    for symbol in (*program.inputs, *program.outputs):
        ids.symbol(symbol)
    _seed_stmt_ids(program.body, ids)
    for tensor_decl in sorted(
        program.tensors, key=lambda item: ids.symbol(item.symbol)
    ):
        for dimension in tensor_decl.dimensions:
            ids.dimension(dimension)
    for dimension_decl in sorted(program.dimensions, key=lambda item: item.name):
        ids.dimension(dimension_decl.dimension)
    return ids


def _serialize_expr(expr: Expr, ids: _CanonicalIds) -> Dict[str, object]:
    if type(expr) is IndexValue:
        return {"kind": "index", "index": ids.index(expr.index)}
    if type(expr) is FloatConst:
        return {"kind": "float_const", "value": expr.value}
    if type(expr) is RootPosition:
        return {"kind": "root_position"}
    if type(expr) is DensePosition:
        return {
            "kind": "dense_position",
            "tensor": ids.symbol(expr.tensor),
            "level": expr.level,
            "parent": _serialize_expr(expr.parent, ids),
            "coord": _serialize_expr(expr.coord, ids),
        }
    if type(expr) is PositionValue:
        return {"kind": "position_value", "position": ids.position(expr.position)}
    if type(expr) is CursorValue:
        return {
            "kind": "cursor_value",
            "cursor": ids.cursor(expr.cursor),
            "default": (
                None if expr.default is None else _serialize_expr(expr.default, ids)
            ),
        }
    if type(expr) is Load:
        return {
            "kind": "load",
            "tensor": ids.symbol(expr.tensor),
            "indices": [_serialize_expr(index, ids) for index in expr.indices],
        }
    if type(expr) is WorkspaceRead:
        return {
            "kind": "workspace_read",
            "workspace": ids.workspace(expr.workspace),
            "coord": _serialize_expr(expr.coord, ids),
        }
    if type(expr) is SparseWorkspaceValue:
        return {
            "kind": "sparse_workspace_value",
            "workspace": ids.workspace(expr.workspace),
        }
    if type(expr) is StagedRead:
        return {
            "kind": "staged_read",
            "relayout": ids.relayout(expr.relayout),
            "indices": [_serialize_expr(index, ids) for index in expr.indices],
        }
    if type(expr) is BinaryExpr:
        return {
            "kind": "binary",
            "op": expr.op.value,
            "lhs": _serialize_expr(expr.lhs, ids),
            "rhs": _serialize_expr(expr.rhs, ids),
        }
    raise TypeError(f"unsupported LoopIR expression {type(expr).__name__}")


def _serialize_cursor(
    cursor: SparseCursorDecl, ids: _CanonicalIds
) -> Dict[str, object]:
    return {
        "cursor": ids.cursor(cursor.cursor),
        "tensor": ids.symbol(cursor.tensor),
        "level": cursor.level,
        "parent": _serialize_expr(cursor.parent, ids),
    }


def _serialize_stmt(stmt: Stmt, ids: _CanonicalIds) -> Dict[str, object]:
    if type(stmt) is Block:
        return {
            "kind": "block",
            "statements": [_serialize_stmt(child, ids) for child in stmt.statements],
        }
    if type(stmt) is DenseFor:
        return {
            "kind": "dense_for",
            "index": ids.index(stmt.index),
            "dimension": ids.dimension(stmt.dimension),
            "body": _serialize_stmt(stmt.body, ids),
        }
    if type(stmt) is TileOuterFor:
        return {
            "kind": "tile_outer_for",
            "tile": ids.tile(stmt.tile),
            "index": ids.index(stmt.index),
            "dimension": ids.dimension(stmt.dimension),
            "width": stmt.width,
            "body": _serialize_stmt(stmt.body, ids),
        }
    if type(stmt) is TileInnerFor:
        return {
            "kind": "tile_inner_for",
            "tile": ids.tile(stmt.tile),
            "index": ids.index(stmt.index),
            "dimension": ids.dimension(stmt.dimension),
            "width": stmt.width,
            "unroll": stmt.unroll,
            "body": _serialize_stmt(stmt.body, ids),
        }
    if type(stmt) is PanelOuterFor:
        return {
            "kind": "panel_outer_for",
            "tile": ids.tile(stmt.tile),
            "index": ids.index(stmt.index),
            "dimension": ids.dimension(stmt.dimension),
            "width": stmt.width,
            "bound_tensor": ids.symbol(stmt.bound_tensor),
            "bound_level": stmt.bound_level,
            "body": _serialize_stmt(stmt.body, ids),
        }
    if type(stmt) is SparseWindowFor:
        return {
            "kind": "sparse_window_for",
            "tile": ids.tile(stmt.tile),
            "cursor": _serialize_cursor(stmt.cursor, ids),
            "position": ids.position(stmt.position),
            "coord_index": ids.index(stmt.coord_index),
            "body": _serialize_stmt(stmt.body, ids),
        }
    if type(stmt) is SparseFor:
        return {
            "kind": "sparse_for",
            "cursor": _serialize_cursor(stmt.cursor, ids),
            "position": ids.position(stmt.position),
            "coord_index": ids.index(stmt.coord_index),
            "body": _serialize_stmt(stmt.body, ids),
        }
    if type(stmt) is MergedSparseFor:
        return {
            "kind": "merged_sparse_for",
            "mode": stmt.mode.value,
            "cursors": [_serialize_cursor(cursor, ids) for cursor in stmt.cursors],
            "coord_index": ids.index(stmt.coord_index),
            "body": _serialize_stmt(stmt.body, ids),
        }
    if type(stmt) is WorkspaceRegion:
        return {
            "kind": "workspace_region",
            "workspace": {
                "workspace": ids.workspace(stmt.workspace.workspace),
                "dtype": stmt.workspace.dtype.value,
                "tile": ids.tile(stmt.workspace.tile),
            },
            "producer": _serialize_stmt(stmt.producer, ids),
            "consumer": _serialize_stmt(stmt.consumer, ids),
        }
    if type(stmt) is WorkspaceReduce:
        return {
            "kind": "workspace_reduce",
            "workspace": ids.workspace(stmt.workspace),
            "op": stmt.op.value,
            "coord": _serialize_expr(stmt.coord, ids),
            "value": _serialize_expr(stmt.value, ids),
        }
    if type(stmt) is SparseWorkspaceRegion:
        return {
            "kind": "sparse_workspace_region",
            "workspace": {
                "workspace": ids.workspace(stmt.workspace.workspace),
                "dtype": stmt.workspace.dtype.value,
                "drain_dimension": ids.dimension(stmt.workspace.drain_dimension),
            },
            "producer": _serialize_stmt(stmt.producer, ids),
            "consumer": _serialize_stmt(stmt.consumer, ids),
        }
    if type(stmt) is SparseWorkspaceInsert:
        return {
            "kind": "sparse_workspace_insert",
            "workspace": ids.workspace(stmt.workspace),
            "op": stmt.op.value,
            "coord": _serialize_expr(stmt.coord, ids),
            "value": _serialize_expr(stmt.value, ids),
        }
    if type(stmt) is SparseWorkspaceDrainFor:
        return {
            "kind": "sparse_workspace_drain_for",
            "workspace": ids.workspace(stmt.workspace),
            "index": ids.index(stmt.index),
            "body": _serialize_stmt(stmt.body, ids),
        }
    if type(stmt) is RelayoutStage:
        return {
            "kind": "relayout_stage",
            "relayout": {
                "relayout": ids.relayout(stmt.decl.relayout),
                "operand": ids.symbol(stmt.decl.operand),
                "panel": ids.tile(stmt.decl.panel),
                "pack": ids.tile(stmt.decl.pack),
                "scope": stmt.decl.scope.value,
            },
            "body": _serialize_stmt(stmt.body, ids),
        }
    if type(stmt) is ResultTileRegion:
        return {
            "kind": "result_tile_region",
            "result_tile": {
                "result_tile": ids.result_tile(stmt.decl.result_tile),
                "result": ids.symbol(stmt.decl.result),
                "pack": ids.tile(stmt.decl.pack),
            },
            "body": _serialize_stmt(stmt.body, ids),
        }
    if type(stmt) is TiledReduce:
        return {
            "kind": "tiled_reduce",
            "result_tile": ids.result_tile(stmt.result_tile),
            "op": stmt.op.value,
            "indices": [_serialize_expr(index, ids) for index in stmt.indices],
            "value": _serialize_expr(stmt.value, ids),
        }
    if type(stmt) is AppendEntry:
        return {
            "kind": "append_entry",
            "tensor": ids.symbol(stmt.tensor),
            "coords": [_serialize_expr(coord, ids) for coord in stmt.coords],
            "value": _serialize_expr(stmt.value, ids),
        }
    if type(stmt) is Store:
        return {
            "kind": "store",
            "tensor": ids.symbol(stmt.tensor),
            "indices": [_serialize_expr(index, ids) for index in stmt.indices],
            "value": _serialize_expr(stmt.value, ids),
        }
    if type(stmt) is StoreReduce:
        return {
            "kind": "store_reduce",
            "tensor": ids.symbol(stmt.tensor),
            "op": stmt.op.value,
            "indices": [_serialize_expr(index, ids) for index in stmt.indices],
            "value": _serialize_expr(stmt.value, ids),
        }
    raise TypeError(f"unsupported LoopIR statement {type(stmt).__name__}")


def _serialize_tensor(decl: TensorDecl, ids: _CanonicalIds) -> Dict[str, object]:
    return {
        "symbol": ids.symbol(decl.symbol),
        "dtype": decl.dtype.value,
        "dimensions": [ids.dimension(dimension) for dimension in decl.dimensions],
        "levels": [
            {"kind": level.kind.value, "mode": level.mode} for level in decl.levels
        ],
    }


def _serialize_parallel(
    selection: ParallelSelection, ids: _CanonicalIds
) -> Dict[str, object]:
    nnz = selection.work.nnz
    return {
        "index": ids.index(selection.index),
        "part": selection.part.value,
        "discipline": selection.discipline.value,
        "work": {
            "rows": ids.dimension(selection.work.rows),
            "nnz": (
                None
                if nnz is None
                else {"tensor": ids.symbol(nnz.tensor), "level": nnz.level}
            ),
        },
        "intent": selection.intent.value,
    }


def canonical_program_dump(program: LoopProgram) -> str:
    """Serialize one verified program canonically, independent of ID history."""

    verify_program(program)
    ids = _canonical_ids(program)
    dimension_decls = sorted(
        program.dimensions, key=lambda decl: ids.dimension(decl.dimension)
    )
    tensor_decls = sorted(program.tensors, key=lambda decl: ids.symbol(decl.symbol))
    payload = {
        "schema": CANONICAL_SCHEMA,
        "dimensions": [
            {"dimension": ids.dimension(decl.dimension)} for decl in dimension_decls
        ],
        "tensors": [_serialize_tensor(decl, ids) for decl in tensor_decls],
        "inputs": [ids.symbol(symbol) for symbol in program.inputs],
        "outputs": [ids.symbol(symbol) for symbol in program.outputs],
        "body": _serialize_stmt(program.body, ids),
        "parallel": (
            None
            if program.parallel is None
            else _serialize_parallel(program.parallel, ids)
        ),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _render_expr(expr: Expr, ids: _CanonicalIds, names: Dict[int, str]) -> str:
    if type(expr) is IndexValue:
        return f"x{ids.index(expr.index)}"
    if type(expr) is FloatConst:
        return repr(expr.value)
    if type(expr) is RootPosition:
        return "root"
    if type(expr) is DensePosition:
        return (
            f"dense_pos(t{ids.symbol(expr.tensor)}, level {expr.level}, "
            f"parent {_render_expr(expr.parent, ids, names)}, "
            f"coord {_render_expr(expr.coord, ids, names)})"
        )
    if type(expr) is PositionValue:
        return f"p{ids.position(expr.position)}"
    if type(expr) is CursorValue:
        if expr.default is None:
            return f"value(c{ids.cursor(expr.cursor)})"
        return (
            f"value(c{ids.cursor(expr.cursor)}, "
            f"default {_render_expr(expr.default, ids, names)})"
        )
    if type(expr) is Load:
        rendered = ", ".join(_render_expr(index, ids, names) for index in expr.indices)
        return f"load t{ids.symbol(expr.tensor)}[{rendered}]"
    if type(expr) is WorkspaceRead:
        return (
            f"w{ids.workspace(expr.workspace)}"
            f"[{_render_expr(expr.coord, ids, names)}]"
        )
    if type(expr) is SparseWorkspaceValue:
        return f"drained(w{ids.workspace(expr.workspace)})"
    if type(expr) is StagedRead:
        rendered = ", ".join(_render_expr(index, ids, names) for index in expr.indices)
        return f"staged r{ids.relayout(expr.relayout)}[{rendered}]"
    if type(expr) is BinaryExpr:
        return (
            f"{expr.op.value}({_render_expr(expr.lhs, ids, names)}, "
            f"{_render_expr(expr.rhs, ids, names)})"
        )
    raise TypeError(f"unsupported LoopIR expression {type(expr).__name__}")


def _render_cursor(
    cursor: SparseCursorDecl, ids: _CanonicalIds, names: Dict[int, str]
) -> str:
    return (
        f"c{ids.cursor(cursor.cursor)} over t{ids.symbol(cursor.tensor)} "
        f"level {cursor.level} parent {_render_expr(cursor.parent, ids, names)}"
    )


def _render_stmt(
    stmt: Stmt,
    ids: _CanonicalIds,
    names: Dict[int, str],
    indent: int,
    lines: List[str],
) -> None:
    pad = "  " * indent
    if type(stmt) is Block:
        for child in stmt.statements:
            _render_stmt(child, ids, names, indent, lines)
        return
    if type(stmt) is DenseFor:
        label = f"x{ids.index(stmt.index)}"
        lines.append(f"{pad}for {label} in d{ids.dimension(stmt.dimension)} {{")
        _render_stmt(stmt.body, ids, names, indent + 1, lines)
        lines.append(f"{pad}}}")
        return
    if type(stmt) is TileOuterFor:
        lines.append(
            f"{pad}tile_outer_for s{ids.tile(stmt.tile)} "
            f"x{ids.index(stmt.index)} in d{ids.dimension(stmt.dimension)} "
            f"width {stmt.width} {{"
        )
        _render_stmt(stmt.body, ids, names, indent + 1, lines)
        lines.append(f"{pad}}}")
        return
    if type(stmt) is TileInnerFor:
        unroll = " unroll" if stmt.unroll else ""
        lines.append(
            f"{pad}tile_inner_for s{ids.tile(stmt.tile)} "
            f"x{ids.index(stmt.index)} in d{ids.dimension(stmt.dimension)} "
            f"width {stmt.width}{unroll} {{"
        )
        _render_stmt(stmt.body, ids, names, indent + 1, lines)
        lines.append(f"{pad}}}")
        return
    if type(stmt) is PanelOuterFor:
        lines.append(
            f"{pad}panel_outer_for s{ids.tile(stmt.tile)} "
            f"x{ids.index(stmt.index)} in d{ids.dimension(stmt.dimension)} "
            f"width {stmt.width} "
            f"bound t{ids.symbol(stmt.bound_tensor)}@{stmt.bound_level} {{"
        )
        _render_stmt(stmt.body, ids, names, indent + 1, lines)
        lines.append(f"{pad}}}")
        return
    if type(stmt) is SparseWindowFor:
        cursor = _render_cursor(stmt.cursor, ids, names)
        lines.append(
            f"{pad}sparse_window_for s{ids.tile(stmt.tile)} "
            f"(p{ids.position(stmt.position)}, "
            f"x{ids.index(stmt.coord_index)}) in {cursor} {{"
        )
        _render_stmt(stmt.body, ids, names, indent + 1, lines)
        lines.append(f"{pad}}}")
        return
    if type(stmt) is SparseFor:
        cursor = _render_cursor(stmt.cursor, ids, names)
        lines.append(
            f"{pad}sparse_for (p{ids.position(stmt.position)}, "
            f"x{ids.index(stmt.coord_index)}) in {cursor} {{"
        )
        _render_stmt(stmt.body, ids, names, indent + 1, lines)
        lines.append(f"{pad}}}")
        return
    if type(stmt) is MergedSparseFor:
        cursors = "; ".join(
            _render_cursor(cursor, ids, names) for cursor in stmt.cursors
        )
        lines.append(
            f"{pad}merged_{stmt.mode.value}_for x{ids.index(stmt.coord_index)} "
            f"in ({cursors}) {{"
        )
        _render_stmt(stmt.body, ids, names, indent + 1, lines)
        lines.append(f"{pad}}}")
        return
    if type(stmt) is WorkspaceRegion:
        decl = stmt.workspace
        lines.append(
            f"{pad}workspace_region w{ids.workspace(decl.workspace)} "
            f"{decl.name!r} {decl.dtype.value} over s{ids.tile(decl.tile)} {{"
        )
        lines.append(f"{pad}  producer {{")
        _render_stmt(stmt.producer, ids, names, indent + 2, lines)
        lines.append(f"{pad}  }}")
        lines.append(f"{pad}  consumer {{")
        _render_stmt(stmt.consumer, ids, names, indent + 2, lines)
        lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
        return
    if type(stmt) is WorkspaceReduce:
        lines.append(
            f"{pad}workspace_reduce({stmt.op.value}) "
            f"w{ids.workspace(stmt.workspace)}"
            f"[{_render_expr(stmt.coord, ids, names)}] = "
            f"{_render_expr(stmt.value, ids, names)}"
        )
        return
    if type(stmt) is SparseWorkspaceRegion:
        decl = stmt.workspace
        lines.append(
            f"{pad}sparse_workspace_region w{ids.workspace(decl.workspace)} "
            f"{decl.name!r} {decl.dtype.value} "
            f"drain d{ids.dimension(decl.drain_dimension)} {{"
        )
        lines.append(f"{pad}  producer {{")
        _render_stmt(stmt.producer, ids, names, indent + 2, lines)
        lines.append(f"{pad}  }}")
        lines.append(f"{pad}  consumer {{")
        _render_stmt(stmt.consumer, ids, names, indent + 2, lines)
        lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
        return
    if type(stmt) is SparseWorkspaceInsert:
        lines.append(
            f"{pad}sparse_workspace_insert({stmt.op.value}) "
            f"w{ids.workspace(stmt.workspace)}"
            f"[{_render_expr(stmt.coord, ids, names)}] = "
            f"{_render_expr(stmt.value, ids, names)}"
        )
        return
    if type(stmt) is SparseWorkspaceDrainFor:
        lines.append(
            f"{pad}sparse_workspace_drain_for x{ids.index(stmt.index)} "
            f"in w{ids.workspace(stmt.workspace)} {{"
        )
        _render_stmt(stmt.body, ids, names, indent + 1, lines)
        lines.append(f"{pad}}}")
        return
    if type(stmt) is RelayoutStage:
        relayout_decl = stmt.decl
        lines.append(
            f"{pad}relayout_stage r{ids.relayout(relayout_decl.relayout)} "
            f"t{ids.symbol(relayout_decl.operand)} "
            f"panel s{ids.tile(relayout_decl.panel)} "
            f"pack s{ids.tile(relayout_decl.pack)} "
            f"scope {relayout_decl.scope.value} {{"
        )
        _render_stmt(stmt.body, ids, names, indent + 1, lines)
        lines.append(f"{pad}}}")
        return
    if type(stmt) is ResultTileRegion:
        result_tile_decl = stmt.decl
        lines.append(
            f"{pad}result_tile_region "
            f"h{ids.result_tile(result_tile_decl.result_tile)} "
            f"t{ids.symbol(result_tile_decl.result)} "
            f"pack s{ids.tile(result_tile_decl.pack)} {{"
        )
        _render_stmt(stmt.body, ids, names, indent + 1, lines)
        lines.append(f"{pad}}}")
        return
    if type(stmt) is TiledReduce:
        rendered = ", ".join(_render_expr(index, ids, names) for index in stmt.indices)
        lines.append(
            f"{pad}tiled_reduce({stmt.op.value}) "
            f"h{ids.result_tile(stmt.result_tile)}[{rendered}] = "
            f"{_render_expr(stmt.value, ids, names)}"
        )
        return
    if type(stmt) is AppendEntry:
        rendered = ", ".join(_render_expr(coord, ids, names) for coord in stmt.coords)
        lines.append(
            f"{pad}append t{ids.symbol(stmt.tensor)}[{rendered}] = "
            f"{_render_expr(stmt.value, ids, names)}"
        )
        return
    if type(stmt) is Store:
        rendered = ", ".join(_render_expr(index, ids, names) for index in stmt.indices)
        lines.append(
            f"{pad}store t{ids.symbol(stmt.tensor)}[{rendered}] = "
            f"{_render_expr(stmt.value, ids, names)}"
        )
        return
    if type(stmt) is StoreReduce:
        rendered = ", ".join(_render_expr(index, ids, names) for index in stmt.indices)
        lines.append(
            f"{pad}store_reduce({stmt.op.value}) "
            f"t{ids.symbol(stmt.tensor)}[{rendered}] = "
            f"{_render_expr(stmt.value, ids, names)}"
        )
        return
    raise TypeError(f"unsupported LoopIR statement {type(stmt).__name__}")


def print_program(program: LoopProgram) -> str:
    """Render one verified program as a deterministic human-readable dump."""

    verify_program(program)
    ids = _canonical_ids(program)
    lines: List[str] = ["loopir.program {"]
    for dimension_decl in sorted(
        program.dimensions, key=lambda decl: ids.dimension(decl.dimension)
    ):
        lines.append(
            f"  dimension d{ids.dimension(dimension_decl.dimension)} "
            f"{dimension_decl.name!r}"
        )
    for decl in sorted(program.tensors, key=lambda item: ids.symbol(item.symbol)):
        levels = ", ".join(f"{level.kind.value}@{level.mode}" for level in decl.levels)
        dimensions = ", ".join(
            f"d{ids.dimension(dimension)}" for dimension in decl.dimensions
        )
        lines.append(
            f"  tensor t{ids.symbol(decl.symbol)} {decl.name!r} "
            f"{decl.dtype.value} dims({dimensions}) levels({levels})"
        )
    inputs = ", ".join(f"t{ids.symbol(symbol)}" for symbol in program.inputs)
    outputs = ", ".join(f"t{ids.symbol(symbol)}" for symbol in program.outputs)
    lines.append(f"  inputs({inputs})")
    lines.append(f"  outputs({outputs})")
    if program.parallel is not None:
        selection = program.parallel
        nnz = selection.work.nnz
        nnz_text = (
            "row_only" if nnz is None else f"nnz(t{ids.symbol(nnz.tensor)}@{nnz.level})"
        )
        lines.append(
            f"  parallel x{ids.index(selection.index)}"
            f" part={selection.part.value}"
            f" discipline={selection.discipline.value}"
            f" work(d{ids.dimension(selection.work.rows)}, {nnz_text})"
            f" intent={selection.intent.value}"
        )
    lines.append("  body {")
    _render_stmt(program.body, ids, {}, 2, lines)
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"
