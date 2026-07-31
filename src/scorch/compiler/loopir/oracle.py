"""Production-owned test/debug semantic oracle for the LoopIR subset.

This module promotes the Phase-3.5 spike interpreter's semantics into the
production tree for the migrated dense and sparse families.  It executes
verified LoopIR programs over plain Python containers with no Torch, native
code, or compiler-pipeline involvement, and is loaded only by dedicated
oracle and differential tests — never by ``import scorch``, default
compilation, legacy correctness paths, or release JIT.

Semantics (matching the node contracts exactly):

- shape compatibility is the logical dimension model: every bound tensor
  contributes the extent of each dimension its modes store, all extents must
  agree across inputs and outputs before anything executes, and ``DenseFor``
  resolves its trip count from the dimension extent;
- dense outputs are zero-initialized before the body runs — the explicit
  contract ``StoreReduce`` (ADD) depends on;
- affine splits execute their intrinsic node semantics: ``TileOuterFor``
  iterates the tile origins ``0, width, 2*width, ...`` below the dimension
  extent, and ``TileInnerFor`` binds the clamped point range
  ``origin .. min(origin + width, extent) - 1``, so every coordinate is
  visited exactly once across the pair;
- sparse coordinate panels execute the same way: ``PanelOuterFor``
  iterates the window origins of its dimension, and ``SparseWindowFor``
  visits, in storage order, exactly the stored entries of its cursor's
  segment whose coordinate falls inside the current clamped window
  ``[origin, min(origin + width, extent))`` — panel widths are semantic
  integers, never allocation requests, and a window executed outside its
  panel's origin loop fails closed at runtime;
- staged relayout regions execute their intrinsic strip semantics:
  ``RelayoutStage`` records the operand strip staged for the current
  scope iteration, ``StagedRead`` observes exactly the staged cells
  (staging copies, so values are served from the operand under
  fail-closed window-domain checks with nothing eagerly allocated), and
  a staged read outside its region, its pack origin, or the staged
  row/column domain fails closed at runtime;
- heap result-tile regions execute their intrinsic accumulation
  semantics: ``ResultTileRegion`` observes a fresh all-zero compact tile
  at entry (only written cells are kept, so nothing is eagerly allocated
  from a verifier-approved width), ``TiledReduce`` combines into the
  addressed cell under fail-closed window-domain checks, and region exit
  copies every clamped-window cell of every dense prefix position to the
  declared result exactly once — cells that received no accumulation
  copy the entry zero, which is what discharges the result's
  zero-initialization contract on the heap route; a tiled reduce outside
  its region, its pack origin, or the compact domain fails closed at
  runtime;
- sparse iteration executes over the format-neutral level interface of
  :mod:`~scorch.compiler.loopir.levels` (``segment`` / ``coordinate_at`` /
  ``leaf_value``): all-dense inputs bind nested sequences, inputs with a
  COMPRESSED level bind a :class:`LevelTensorStorage` (canonical CSR
  declarations also accept the :class:`CsrMatrix` adapter), and canonical
  CSR outputs are assembled through an order-checked append builder that
  returns a :class:`CsrMatrix`;
- all arithmetic is Python-float arithmetic in program order; the oracle is
  a semantic reference, not a bit-accuracy model of any particular scalar
  width, so numeric comparisons against compiled kernels use tolerances
  owned by the caller.

Everything unexpected fails closed with :class:`LoopIROracleError`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from ..identity import IndexId, SymbolId
from .levels import (
    CsrMatrix,
    CsrOutputBuilder,
    LevelOutputBuilder,
    LevelStorageError,
    LevelTensorStorage,
    MAX_LEVEL_STORAGE_RANK,
    _diagnostic_int,
    from_csr,
)
from .nodes import (
    AppendEntry,
    BinaryExpr,
    BinaryOp,
    Block,
    CursorId,
    CursorValue,
    DenseFor,
    DensePosition,
    DimensionId,
    Expr,
    FloatConst,
    IndexValue,
    LevelKind,
    Load,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    PanelOuterFor,
    PositionId,
    PositionLoad,
    PositionValue,
    RelayoutDecl,
    RelayoutId,
    RelayoutScope,
    RelayoutStage,
    ResultTileDecl,
    ResultTileId,
    ResultTileRegion,
    RootPosition,
    SparseCursorDecl,
    SparseFor,
    SparseWindowFor,
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
    SparseWorkspaceDecl,
    SparseWorkspaceDrainFor,
    SparseWorkspaceInsert,
    SparseWorkspaceRegion,
    SparseWorkspaceValue,
    WorkspaceRead,
    WorkspaceReduce,
    WorkspaceRegion,
)
from .verifier import verify_program

TensorValue = Any
MAX_ORACLE_RANK = MAX_LEVEL_STORAGE_RANK

_CSR_KINDS = (LevelKind.DENSE, LevelKind.COMPRESSED)
_CSR_MODES = (0, 1)


class LoopIROracleError(Exception):
    """Program execution hit a state the LoopIR oracle rejects."""


def _is_canonical_csr(decl: TensorDecl) -> bool:
    kinds = tuple(level.kind for level in decl.levels)
    modes = tuple(level.mode for level in decl.levels)
    return kinds == _CSR_KINDS and modes == _CSR_MODES


class _CursorState:
    """Runtime state of one sparse cursor inside its owning loop."""

    __slots__ = ("storage", "level", "position", "end", "aligned")

    def __init__(
        self,
        storage: LevelTensorStorage,
        level: int,
        start: int,
        end: int,
        aligned: bool,
    ) -> None:
        self.storage = storage
        self.level = level
        self.position = start
        self.end = end
        self.aligned = aligned

    @property
    def exhausted(self) -> bool:
        return self.position >= self.end

    @property
    def coordinate(self) -> int:
        return self.storage.coordinate_at(self.level, self.position)

    @property
    def value(self) -> float:
        return self.storage.leaf_value(self.position)


def _snapshot_dense(value: object, remaining_rank: int) -> object:
    """Own every sequence container the rank-bounded dense copy may inspect."""

    if remaining_rank == 0 or type(value) not in (list, tuple):
        return value
    owned = cast(Sequence[object], value)
    return [_snapshot_dense(entry, remaining_rank - 1) for entry in owned]


def _dense_shape(rank: int, value: object, trail: str) -> Tuple[Optional[int], ...]:
    """Validate a dense binding and infer every visible prefix extent.

    An empty sequence fixes its own extent at zero but carries no information
    about deeper modes.  Those suffix extents are resolved later through the
    program's shared :class:`DimensionId` bindings instead of rejecting a
    semantically valid zero-extent tensor prematurely.
    """

    shape: List[int] = []
    layer: object = value
    for _level in range(rank):
        if type(layer) not in (list, tuple):
            raise LoopIROracleError(f"{trail} must nest sequences to rank {rank}")
        owned_layer = cast(Sequence[object], layer)
        shape.append(len(owned_layer))
        if not owned_layer:
            return tuple([*shape, *([None] * (rank - len(shape)))])
        layer = owned_layer[0]
    return tuple(shape)


def _dense_copy(value: object, shape: Tuple[int, ...], trail: str) -> Any:
    if type(value) not in (list, tuple):
        raise LoopIROracleError(f"{trail} is ragged or mis-shaped")
    owned_value = cast(Sequence[object], value)
    if len(owned_value) != shape[0]:
        raise LoopIROracleError(f"{trail} is ragged or mis-shaped")
    if len(shape) == 1:
        row: List[float] = []
        for entry in owned_value:
            if type(entry) is not float and type(entry) is not int:
                raise LoopIROracleError(f"{trail} holds a non-numeric entry")
            try:
                row.append(float(entry))
            except (OverflowError, TypeError, ValueError) as error:
                raise LoopIROracleError(
                    f"{trail} holds an unrepresentable numeric entry"
                ) from error
        return row
    return [
        _dense_copy(entry, shape[1:], f"{trail}[{position}]")
        for position, entry in enumerate(owned_value)
    ]


def _zeros(shape: Tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [_zeros(shape[1:]) for _ in range(shape[0])]


class _Oracle:
    def __init__(
        self,
        program: LoopProgram,
        inputs: Mapping[SymbolId, object],
        output_shapes: Mapping[SymbolId, Tuple[int, ...]],
    ) -> None:
        self.program = program
        self.decls: Dict[SymbolId, TensorDecl] = {
            decl.symbol: decl for decl in program.tensors
        }
        self.dimension_names: Dict[DimensionId, str] = {
            decl.dimension: decl.name for decl in program.dimensions
        }
        try:
            input_key_snapshot = tuple(inputs)
        except Exception as error:
            raise LoopIROracleError(
                "input binding keys could not be snapshotted"
            ) from error
        if any(
            type(symbol) is not SymbolId
            or type(getattr(symbol, "value", None)) is not int
            for symbol in input_key_snapshot
        ):
            raise LoopIROracleError(
                "input binding keys must be exact int-valued SymbolId values"
            )
        if set(input_key_snapshot) != set(program.inputs):
            raise LoopIROracleError(
                "input bindings must cover exactly the declared inputs"
            )
        for decl in program.tensors:
            rank = len(decl.levels)
            if rank > MAX_ORACLE_RANK:
                raise LoopIROracleError(
                    f"tensor {decl.name!r} rank {rank} exceeds the oracle "
                    f"limit {MAX_ORACLE_RANK}"
                )

        self.values: Dict[SymbolId, Any] = {}
        self.storages: Dict[SymbolId, LevelTensorStorage] = {}
        self.builders: Dict[SymbolId, "CsrOutputBuilder | LevelOutputBuilder"] = {}
        self.shapes: Dict[SymbolId, Tuple[int, ...]] = {}
        input_values: Dict[SymbolId, object] = {}
        partial_input_shapes: Dict[SymbolId, Tuple[Optional[int], ...]] = {}
        for symbol in program.inputs:
            decl = self.decls[symbol]
            try:
                bound = inputs[symbol]
            except Exception as error:
                raise LoopIROracleError(
                    "input bindings could not be snapshotted"
                ) from error
            if any(level.kind is not LevelKind.DENSE for level in decl.levels):
                self.storages[symbol] = self._bind_sparse_input(decl, bound)
                self.shapes[symbol] = self.storages[symbol].shape
                continue
            owned_bound = _snapshot_dense(bound, len(decl.levels))
            input_values[symbol] = owned_bound
            partial_input_shapes[symbol] = _dense_shape(
                len(decl.levels), owned_bound, f"input {decl.name}"
            )
        try:
            output_key_snapshot = tuple(output_shapes)
        except Exception as error:
            raise LoopIROracleError(
                "output shape keys could not be snapshotted"
            ) from error
        if any(
            type(symbol) is not SymbolId
            or type(getattr(symbol, "value", None)) is not int
            for symbol in output_key_snapshot
        ):
            raise LoopIROracleError(
                "output shape keys must be exact int-valued SymbolId values"
            )
        if set(output_key_snapshot) != set(program.outputs):
            raise LoopIROracleError(
                "output shapes must cover exactly the declared outputs"
            )
        for symbol in program.outputs:
            decl = self.decls[symbol]
            try:
                shape_binding = output_shapes[symbol]
            except Exception as error:
                raise LoopIROracleError(
                    "output shapes could not be snapshotted"
                ) from error
            if (
                type(shape_binding) is not tuple
                or len(shape_binding) != len(decl.levels)
                or any(
                    type(extent) is not int or extent < 0 for extent in shape_binding
                )
            ):
                raise LoopIROracleError(
                    f"output {decl.name} needs a rank-{len(decl.levels)} shape "
                    "of nonnegative ints"
                )
            self.shapes[symbol] = shape_binding

        self.dim_extents: Dict[DimensionId, Tuple[int, str, int]] = {}
        for symbol in program.inputs:
            decl = self.decls[symbol]
            if symbol in self.storages:
                for mode, extent in enumerate(self.shapes[symbol]):
                    self._bind_dimension_extent(
                        decl.dimensions[mode], extent, decl.name, mode
                    )
                continue
            for mode, partial_extent in enumerate(partial_input_shapes[symbol]):
                if partial_extent is not None:
                    self._bind_dimension_extent(
                        decl.dimensions[mode], partial_extent, decl.name, mode
                    )
        for symbol in program.outputs:
            decl = self.decls[symbol]
            for mode, extent in enumerate(self.shapes[symbol]):
                self._bind_dimension_extent(
                    decl.dimensions[mode], extent, decl.name, mode
                )

        for symbol in program.inputs:
            decl = self.decls[symbol]
            if symbol in self.storages:
                continue
            resolved_shape: List[int] = []
            for mode, partial_extent in enumerate(partial_input_shapes[symbol]):
                resolved_extent = partial_extent
                if resolved_extent is None:
                    known = self.dim_extents.get(decl.dimensions[mode])
                    if known is None:
                        raise LoopIROracleError(
                            f"input {decl.name} shape cannot be inferred at "
                            f"mode {mode}; bind another tensor sharing that "
                            "dimension or avoid an empty outer mode"
                        )
                    resolved_extent = known[0]
                resolved_shape.append(resolved_extent)
            shape = tuple(resolved_shape)
            self.shapes[symbol] = shape
            self.values[symbol] = _dense_copy(
                input_values[symbol], shape, f"input {decl.name}"
            )
        # Materialize outputs only after all shared dimensions reconcile and
        # dense input shapes resolve.  Incompatible output extents must fail
        # cheaply instead of allocating from an untrusted shape first.
        for symbol in program.outputs:
            decl = self.decls[symbol]
            shape = self.shapes[symbol]
            level_kinds = tuple(level.kind for level in decl.levels)
            if level_kinds == (LevelKind.DENSE, LevelKind.COMPRESSED):
                self.builders[symbol] = CsrOutputBuilder(decl.name, shape)
            elif any(kind is not LevelKind.DENSE for kind in level_kinds):
                self.builders[symbol] = LevelOutputBuilder(
                    decl.name, shape, level_kinds
                )
            else:
                self.values[symbol] = _zeros(shape) if shape else []
        self.indices: Dict[IndexId, int] = {}
        self.sparse_workspaces: Dict[
            WorkspaceId, Tuple[SparseWorkspaceDecl, Dict[int, float]]
        ] = {}
        self.draining_values: Dict[WorkspaceId, float] = {}
        self.sparse_reductions: Dict[SymbolId, Dict[Tuple[int, ...], float]] = {}
        self.positions: Dict[PositionId, int] = {}
        self.cursors: Dict[CursorId, _CursorState] = {}
        self.tile_origins: Dict[TileId, int] = {}
        # Workspace extents are semantic integers, not allocation requests
        # to the Python oracle.  Keep only written cells and treat every
        # absent entry as ADD's zero identity.  This preserves fresh/reset
        # semantics without eagerly allocating an attacker-sized list for a
        # verifier-approved (target-neutral) tile width.
        self.workspaces: Dict[WorkspaceId, Tuple[TileId, int, Dict[int, float]]] = {}
        self._tile_widths: Dict[TileId, int] = {}
        self.panel_origins: Dict[TileId, int] = {}
        self._panel_widths: Dict[TileId, int] = {}
        # A staged relayout strip is served lazily from the operand it
        # copies: staging preserves values exactly, so the region records
        # only which strip is live and every read is domain-checked against
        # the current window arithmetic.  Nothing is eagerly allocated from
        # a verifier-approved (target-neutral) width.
        self.staged_relayouts: Dict[RelayoutId, RelayoutDecl] = {}
        # A heap result tile keeps only written cells (absent cells are
        # ADD's zero identity), preserving fresh/reset semantics without
        # eagerly allocating from a verifier-approved (target-neutral)
        # width; copy-out at region exit enumerates the caller-allocated
        # result, never the width.
        self.result_tiles: Dict[
            ResultTileId, Tuple[ResultTileDecl, Dict[Tuple[int, ...], float]]
        ] = {}

    def _workspace_cell(
        self, workspace: WorkspaceId, coord: Expr
    ) -> Tuple[Dict[int, float], int]:
        """Resolve one workspace access to its live cells and cell index."""

        state = self.workspaces.get(workspace)
        if state is None:
            raise LoopIROracleError("workspace accessed outside its region's execution")
        tile, width, cells = state
        origin = self.tile_origins.get(tile)
        if origin is None:
            raise LoopIROracleError("workspace accessed outside its tile's origin loop")
        coordinate = self._eval_coord(coord)
        cell = coordinate - origin
        if not 0 <= cell < width:
            raise LoopIROracleError(
                f"workspace cell {cell} outside [0, {width}) for "
                f"coordinate {coordinate} at origin {origin}"
            )
        return cells, cell

    def _bind_sparse_input(self, decl: TensorDecl, bound: object) -> LevelTensorStorage:
        """Snapshot one compressed-layout input behind the level interface."""

        if type(bound) is CsrMatrix:
            if not _is_canonical_csr(decl):
                raise LoopIROracleError(
                    f"input {decl.name} is not a canonical CSR declaration; "
                    "bind a LevelTensorStorage instead"
                )
            try:
                return from_csr(bound)
            except LevelStorageError as error:
                raise LoopIROracleError(
                    f"input {decl.name} has invalid CSR storage: {error}"
                ) from error
        if type(bound) is not LevelTensorStorage:
            raise LoopIROracleError(
                f"input {decl.name} must be bound to a LevelTensorStorage"
                + (" or CsrMatrix" if _is_canonical_csr(decl) else "")
            )
        try:
            storage = bound.snapshot()
        except LevelStorageError as error:
            raise LoopIROracleError(
                f"input {decl.name} has invalid level storage: {error}"
            ) from error
        declared_kinds = tuple(level.kind for level in decl.levels)
        declared_modes = tuple(level.mode for level in decl.levels)
        if storage.kinds != declared_kinds or storage.modes != declared_modes:
            raise LoopIROracleError(
                f"input {decl.name} storage layout "
                f"({tuple(kind.value for kind in storage.kinds)}, "
                f"modes {storage.modes}) does not match its declaration "
                f"({tuple(kind.value for kind in declared_kinds)}, "
                f"modes {declared_modes})"
            )
        return storage

    def _bind_dimension_extent(
        self, dimension: DimensionId, extent: int, tensor_name: str, mode: int
    ) -> None:
        known = self.dim_extents.get(dimension)
        if known is None:
            self.dim_extents[dimension] = (extent, tensor_name, mode)
        elif known[0] != extent:
            name = self.dimension_names.get(dimension, f"<dimension {dimension.value}>")
            raise LoopIROracleError(
                f"dimension extent mismatch for {name!r}: "
                f"{known[1]}[{known[2]}] is {_diagnostic_int(known[0])} but "
                f"{tensor_name}[{mode}] is {_diagnostic_int(extent)}"
            )

    def _dimension_extent(self, dimension: DimensionId) -> int:
        known = self.dim_extents.get(dimension)
        if known is None:
            name = self.dimension_names.get(dimension, f"<dimension {dimension.value}>")
            raise LoopIROracleError(
                f"unresolved dimension extent for {name!r}: no bound tensor "
                "stores this dimension"
            )
        return known[0]

    def run(self) -> Dict[SymbolId, TensorValue]:
        self._exec_stmt(self.program.body)
        results: Dict[SymbolId, TensorValue] = {}
        for symbol, entries in self.sparse_reductions.items():
            builder = self.builders[symbol]
            if getattr(builder, "entries", None) or getattr(builder, "rows", None):
                raise LoopIROracleError(
                    "an output cannot mix ordered appends with the semantic "
                    "sparse accumulation form"
                )
            for coords in sorted(entries):
                builder.append(coords, entries[coords])
        for symbol in self.program.outputs:
            if symbol in self.builders:
                try:
                    results[symbol] = self.builders[symbol].finish()
                except LevelStorageError as error:
                    raise LoopIROracleError(
                        f"output {self.decls[symbol].name} assembly failed: {error}"
                    ) from error
            else:
                results[symbol] = self.values[symbol]
        return results

    def _eval_coord(self, expr: Expr) -> int:
        result = self._eval(expr)
        if type(result) is not int:
            raise LoopIROracleError(
                f"coordinate expression produced {type(result).__name__}"
            )
        return result

    def _eval_value(self, expr: Expr) -> float:
        result = self._eval(expr)
        if type(result) is not float:
            raise LoopIROracleError(
                f"value expression produced {type(result).__name__}"
            )
        return result

    def _eval_position(self, expr: Expr) -> int:
        result = self._eval(expr)
        if type(result) is not int:
            raise LoopIROracleError(
                f"position expression produced {type(result).__name__}"
            )
        return result

    def _eval(self, expr: Expr) -> object:
        if type(expr) is IndexValue:
            return self.indices[expr.index]
        if type(expr) is FloatConst:
            return expr.value
        if type(expr) is RootPosition:
            return 0
        if type(expr) is PositionValue:
            return self.positions[expr.position]
        if type(expr) is DensePosition:
            parent = self._eval_position(expr.parent)
            decl = self.decls[expr.tensor]
            dimension = decl.dimensions[decl.levels[expr.level].mode]
            extent = self._dimension_extent(dimension)
            coord = self._eval_coord(expr.coord)
            if not 0 <= coord < extent:
                raise LoopIROracleError(
                    f"dense-level coordinate {coord} outside [0, {extent}) on "
                    f"{decl.name}"
                )
            return parent * extent + coord
        if type(expr) is CursorValue:
            state = self.cursors[expr.cursor]
            if state.aligned:
                return state.value
            if expr.default is None:
                raise LoopIROracleError("unaligned cursor read without a default")
            return self._eval_value(expr.default)
        if type(expr) is WorkspaceRead:
            cells, cell = self._workspace_cell(expr.workspace, expr.coord)
            return cells.get(cell, 0.0)
        if type(expr) is SparseWorkspaceValue:
            if expr.workspace not in self.draining_values:
                raise LoopIROracleError(
                    "drained value read outside its workspace's drain loop"
                )
            return self.draining_values[expr.workspace]
        if type(expr) is PositionLoad:
            storage = self.storages.get(expr.tensor)
            if storage is None:
                decl = self.decls[expr.tensor]
                dense = self.values.get(expr.tensor)
                if dense is None:
                    raise LoopIROracleError(
                        f"position-loaded tensor {decl.name} has no level storage"
                    )
                try:
                    storage = LevelTensorStorage.from_dense(
                        dense,
                        self.shapes[expr.tensor],
                        tuple(level.mode for level in decl.levels),
                        tuple(level.kind for level in decl.levels),
                    )
                except LevelStorageError as error:
                    raise LoopIROracleError(
                        f"input {decl.name} could not be position-materialized: "
                        f"{error}"
                    ) from error
                self.storages[expr.tensor] = storage
            position = self._eval_position(expr.position)
            try:
                return storage.leaf_value(position)
            except LevelStorageError as error:
                raise LoopIROracleError(
                    f"position load outside {self.decls[expr.tensor].name} "
                    f"leaf storage: {error}"
                ) from error
        if type(expr) is Load:
            current: Any = self.values[expr.tensor]
            for position, index_expr in enumerate(expr.indices):
                index = self._eval_coord(index_expr)
                if not isinstance(current, list) or not 0 <= index < len(current):
                    raise LoopIROracleError(
                        f"load index {index} out of bounds at mode {position}"
                    )
                current = current[index]
            if type(current) is not float:
                raise LoopIROracleError("load did not resolve to a scalar")
            return current
        if type(expr) is StagedRead:
            return self._eval_staged_read(expr)
        if type(expr) is BinaryExpr:
            lhs = self._eval_value(expr.lhs)
            rhs = self._eval_value(expr.rhs)
            if expr.op is BinaryOp.ADD:
                return lhs + rhs
            if expr.op is BinaryOp.SUB:
                return lhs - rhs
            return lhs * rhs
        raise LoopIROracleError(f"unknown expression {type(expr).__name__}")

    def _eval_staged_read(self, expr: StagedRead) -> float:
        """Serve one staged read under the intrinsic strip-domain semantics."""

        decl = self.staged_relayouts.get(expr.relayout)
        if decl is None:
            raise LoopIROracleError(
                "staged read outside its relayout region's execution"
            )
        operand_decl = self.decls[decl.operand]
        coords = [self._eval_coord(index) for index in expr.indices]
        pack_origin = self.tile_origins.get(decl.pack)
        pack_width = self._tile_widths.get(decl.pack)
        if pack_origin is None or pack_width is None:
            raise LoopIROracleError("staged read outside its pack split's origin loop")
        row_mode = operand_decl.levels[0].mode
        pack_mode = operand_decl.levels[1].mode
        row = coords[row_mode]
        column = coords[pack_mode]
        pack_extent = self._dimension_extent(operand_decl.dimensions[pack_mode])
        if not pack_origin <= column < min(pack_origin + pack_width, pack_extent):
            raise LoopIROracleError(
                f"staged read column {column} outside the current pack "
                f"window [{pack_origin}, "
                f"{min(pack_origin + pack_width, pack_extent)})"
            )
        row_extent = self._dimension_extent(operand_decl.dimensions[row_mode])
        if decl.scope is RelayoutScope.PANEL:
            panel_origin = self.panel_origins.get(decl.panel)
            panel_width = self._panel_widths.get(decl.panel)
            if panel_origin is None or panel_width is None:
                raise LoopIROracleError(
                    "panel-scoped staged read outside its panel's origin loop"
                )
            window_end = min(panel_origin + panel_width, row_extent)
            if not panel_origin <= row < window_end:
                raise LoopIROracleError(
                    f"staged read row {row} outside the current panel "
                    f"window [{panel_origin}, {window_end})"
                )
        elif not 0 <= row < row_extent:
            raise LoopIROracleError(
                f"staged read row {row} outside the staged axis " f"[0, {row_extent})"
            )
        current: Any = self.values[decl.operand]
        for position, index in enumerate(coords):
            if not isinstance(current, list) or not 0 <= index < len(current):
                raise LoopIROracleError(
                    f"staged read index {index} out of bounds at mode {position}"
                )
            current = current[index]
        if type(current) is not float:
            raise LoopIROracleError("staged read did not resolve to a scalar")
        return current

    def _locate_store(
        self, tensor: SymbolId, indices: Tuple[Expr, ...]
    ) -> Tuple[List[Any], int]:
        coords = [self._eval_coord(index) for index in indices]
        target: Any = self.values[tensor]
        for index in coords[:-1]:
            if not isinstance(target, list) or not 0 <= index < len(target):
                raise LoopIROracleError(f"store index {index} out of bounds")
            target = target[index]
        last = coords[-1]
        if not isinstance(target, list) or not 0 <= last < len(target):
            raise LoopIROracleError(f"store index {last} out of bounds")
        return target, last

    def _segment(self, decl: SparseCursorDecl) -> Tuple[LevelTensorStorage, int, int]:
        storage = self.storages.get(decl.tensor)
        if storage is None:
            raise LoopIROracleError(
                f"cursor tensor {self.decls[decl.tensor].name} has no level storage"
            )
        parent = self._eval_position(decl.parent)
        try:
            start, end = storage.segment(decl.level, parent)
        except LevelStorageError as error:
            raise LoopIROracleError(
                f"cursor segment selection failed on "
                f"{self.decls[decl.tensor].name}: {error}"
            ) from error
        return storage, start, end

    def _exec_relayout_stage(self, stmt: RelayoutStage) -> None:
        decl = stmt.decl
        if self.tile_origins.get(decl.pack) is None:
            raise LoopIROracleError(
                "relayout region executed outside its pack split's origin loop"
            )
        if (
            decl.scope is RelayoutScope.PANEL
            and self.panel_origins.get(decl.panel) is None
        ):
            raise LoopIROracleError(
                "panel-scoped relayout region executed outside its "
                "panel's origin loop"
            )
        if decl.relayout in self.staged_relayouts:
            raise LoopIROracleError(
                "relayout region re-entered while already executing"
            )
        # Intrinsic region-entry semantics: the operand's current strip is
        # staged and stays valid throughout the body; teardown at exit
        # means the next scope iteration observes a fresh strip.
        self.staged_relayouts[decl.relayout] = decl
        try:
            self._exec_stmt(stmt.body)
        finally:
            del self.staged_relayouts[decl.relayout]

    def _exec_workspace_region(self, stmt: WorkspaceRegion) -> None:
        decl = stmt.workspace
        region_origin = self.tile_origins.get(decl.tile)
        if region_origin is None:
            raise LoopIROracleError(
                "workspace region executed outside its tile's origin loop"
            )
        width = self._tile_widths.get(decl.tile)
        if width is None:
            raise LoopIROracleError(
                "workspace region's tile has no executing origin loop"
            )
        if decl.workspace in self.workspaces:
            raise LoopIROracleError(
                "workspace region re-entered while already executing"
            )
        # Intrinsic region-entry semantics: a fresh buffer of one cell
        # per tile point, every cell zero (ADD's identity).
        self.workspaces[decl.workspace] = (decl.tile, width, {})
        try:
            self._exec_stmt(stmt.producer)
            self._exec_stmt(stmt.consumer)
        finally:
            del self.workspaces[decl.workspace]

    def _exec_sparse_workspace_region(self, stmt: SparseWorkspaceRegion) -> None:
        decl = stmt.workspace
        if decl.workspace in self.sparse_workspaces:
            raise LoopIROracleError(
                "sparse workspace region re-entered while already executing"
            )
        # Intrinsic region-entry semantics: an empty workspace (ADD's
        # identity is the absent entry).
        self.sparse_workspaces[decl.workspace] = (decl, {})
        try:
            self._exec_stmt(stmt.producer)
            self._exec_stmt(stmt.consumer)
        finally:
            del self.sparse_workspaces[decl.workspace]

    def _exec_sparse_workspace_insert(self, stmt: SparseWorkspaceInsert) -> None:
        open_workspace = self.sparse_workspaces.get(stmt.workspace)
        if open_workspace is None:
            raise LoopIROracleError(
                "sparse workspace insert outside an executing region"
            )
        decl, entries = open_workspace
        extent = self._dimension_extent(decl.drain_dimension)
        coordinate = self._eval_coord(stmt.coord)
        if not 0 <= coordinate < extent:
            raise LoopIROracleError(
                f"sparse workspace coordinate {coordinate} outside [0, {extent})"
            )
        contribution = self._eval_value(stmt.value)
        entries[coordinate] = entries.get(coordinate, 0.0) + contribution

    def _exec_sparse_workspace_drain(self, stmt: SparseWorkspaceDrainFor) -> None:
        open_workspace = self.sparse_workspaces.get(stmt.workspace)
        if open_workspace is None:
            raise LoopIROracleError(
                "sparse workspace drain outside an executing region"
            )
        if stmt.workspace in self.draining_values:
            raise LoopIROracleError(
                "sparse workspace drain re-entered while already draining"
            )
        _, entries = open_workspace
        try:
            for coordinate in sorted(entries):
                self.indices[stmt.index] = coordinate
                self.draining_values[stmt.workspace] = entries[coordinate]
                self._exec_stmt(stmt.body)
        finally:
            self.indices.pop(stmt.index, None)
            self.draining_values.pop(stmt.workspace, None)

    _SPARSE_WORKSPACE_EXEC: Dict[type, Any] = {}

    def _exec_result_tile_region(self, stmt: ResultTileRegion) -> None:
        decl = stmt.decl
        origin = self.tile_origins.get(decl.pack)
        width = self._tile_widths.get(decl.pack)
        if origin is None or width is None:
            raise LoopIROracleError(
                "result-tile region executed outside its pack split's " "origin loop"
            )
        if decl.result_tile in self.result_tiles:
            raise LoopIROracleError(
                "result-tile region re-entered while already executing"
            )
        # Intrinsic region-entry semantics: a fresh compact tile with every
        # cell zero (ADD's identity); only written cells are kept.
        cells: Dict[Tuple[int, ...], float] = {}
        self.result_tiles[decl.result_tile] = (decl, cells)
        try:
            self._exec_stmt(stmt.body)
        finally:
            del self.result_tiles[decl.result_tile]
        # Intrinsic region-exit semantics: every clamped-window cell of
        # every dense prefix position is copied to the result exactly once;
        # unwritten cells copy the entry zero.
        result_decl = self.decls[decl.result]
        trailing_mode = result_decl.levels[-1].mode
        extent = self._dimension_extent(result_decl.dimensions[trailing_mode])
        window_end = min(origin + width, extent)

        copied_cells: set[Tuple[int, ...]] = set()

        def copy_out(prefix: Tuple[int, ...], value: Any, mode: int) -> None:
            if not isinstance(value, list):
                raise LoopIROracleError(
                    "result-tile copy-out target is not a dense axis"
                )
            positions = (
                range(origin, window_end)
                if mode == trailing_mode
                else range(len(value))
            )
            for position in positions:
                if not 0 <= position < len(value):
                    raise LoopIROracleError(
                        f"result-tile copy-out coordinate {position} out of bounds "
                        f"at mode {mode}"
                    )
                key_coord = position - origin if mode == trailing_mode else position
                key = prefix + (key_coord,)
                if mode == len(result_decl.levels) - 1:
                    if key in copied_cells:
                        raise LoopIROracleError(
                            "result-tile copied one compact cell more than once"
                        )
                    copied_cells.add(key)
                    value[position] = cells.get(key, 0.0)
                else:
                    copy_out(key, value[position], mode + 1)

        copy_out((), self.values[decl.result], 0)

    def _exec_tiled_reduce(self, stmt: TiledReduce) -> None:
        state = self.result_tiles.get(stmt.result_tile)
        if state is None:
            raise LoopIROracleError(
                "tiled reduce outside its result-tile region's execution"
            )
        decl, cells = state
        origin = self.tile_origins.get(decl.pack)
        width = self._tile_widths.get(decl.pack)
        if origin is None or width is None:
            raise LoopIROracleError("tiled reduce outside its pack split's origin loop")
        result_decl = self.decls[decl.result]
        coords = [self._eval_coord(index) for index in stmt.indices]
        trailing_mode = result_decl.levels[-1].mode
        column = coords[trailing_mode]
        extent = self._dimension_extent(result_decl.dimensions[trailing_mode])
        window_end = min(origin + width, extent)
        if not origin <= column < window_end:
            raise LoopIROracleError(
                f"tiled reduce column {column} outside the current pack "
                f"window [{origin}, {window_end})"
            )
        for position, coordinate in enumerate(coords):
            if position == trailing_mode:
                continue
            prefix_extent = self._dimension_extent(result_decl.dimensions[position])
            if not 0 <= coordinate < prefix_extent:
                raise LoopIROracleError(
                    f"tiled reduce index {coordinate} out of bounds at "
                    f"mode {position}"
                )
        key = tuple(
            column - origin if position == trailing_mode else coordinate
            for position, coordinate in enumerate(coords)
        )
        contribution = self._eval_value(stmt.value)
        cells[key] = cells.get(key, 0.0) + contribution

    def _exec_merge(self, stmt: MergedSparseFor) -> None:
        states: List[_CursorState] = []
        try:
            for decl in stmt.cursors:
                storage, start, end = self._segment(decl)
                state = _CursorState(storage, decl.level, start, end, aligned=False)
                self.cursors[decl.cursor] = state
                states.append(state)
            while True:
                if stmt.mode is MergeMode.UNION:
                    active = [state for state in states if not state.exhausted]
                    if not active:
                        break
                else:
                    if any(state.exhausted for state in states):
                        break
                    active = states
                candidate = min(state.coordinate for state in active)
                aligned = [state for state in active if state.coordinate == candidate]
                for state in states:
                    state.aligned = (
                        not state.exhausted and state.coordinate == candidate
                    )
                if stmt.mode is MergeMode.UNION or len(aligned) == len(states):
                    self.indices[stmt.coord_index] = candidate
                    for cursor_position, bound in enumerate(stmt.positions):
                        if bound is not None:
                            # INTERSECTION: every cursor is aligned at the
                            # body, so its position is the descent anchor.
                            self.positions[bound] = states[cursor_position].position
                    self._exec_stmt(stmt.body)
                for state in aligned:
                    state.position += 1
        finally:
            self.indices.pop(stmt.coord_index, None)
            for bound in stmt.positions:
                if bound is not None:
                    self.positions.pop(bound, None)
            for decl in stmt.cursors:
                self.cursors.pop(decl.cursor, None)

    def _exec_stmt(self, stmt: Stmt) -> None:
        if type(stmt) is Block:
            for child in stmt.statements:
                self._exec_stmt(child)
            return
        if type(stmt) is DenseFor:
            extent = self._dimension_extent(stmt.dimension)
            try:
                for coordinate in range(extent):
                    self.indices[stmt.index] = coordinate
                    self._exec_stmt(stmt.body)
            finally:
                self.indices.pop(stmt.index, None)
            return
        if type(stmt) is TileOuterFor:
            extent = self._dimension_extent(stmt.dimension)
            self._tile_widths[stmt.tile] = stmt.width
            try:
                for origin in range(0, extent, stmt.width):
                    self.tile_origins[stmt.tile] = origin
                    self._exec_stmt(stmt.body)
            finally:
                self.tile_origins.pop(stmt.tile, None)
                self._tile_widths.pop(stmt.tile, None)
            return
        if type(stmt) is TileInnerFor:
            bound_origin = self.tile_origins.get(stmt.tile)
            if bound_origin is None:
                raise LoopIROracleError(
                    "tile point loop executed outside its origin loop"
                )
            origin = bound_origin
            extent = self._dimension_extent(stmt.dimension)
            try:
                for coordinate in range(origin, min(origin + stmt.width, extent)):
                    self.indices[stmt.index] = coordinate
                    self._exec_stmt(stmt.body)
            finally:
                self.indices.pop(stmt.index, None)
            return
        if type(stmt) is WorkspaceRegion:
            self._exec_workspace_region(stmt)
            return
        if type(stmt) is WorkspaceReduce:
            cells, cell = self._workspace_cell(stmt.workspace, stmt.coord)
            contribution = self._eval_value(stmt.value)
            cells[cell] = cells.get(cell, 0.0) + contribution
            return
        sparse_workspace_exec = self._SPARSE_WORKSPACE_EXEC.get(type(stmt))
        if sparse_workspace_exec is not None:
            sparse_workspace_exec(self, stmt)
            return
        if type(stmt) is RelayoutStage:
            self._exec_relayout_stage(stmt)
            return
        if type(stmt) is ResultTileRegion:
            self._exec_result_tile_region(stmt)
            return
        if type(stmt) is TiledReduce:
            self._exec_tiled_reduce(stmt)
            return
        if type(stmt) is PanelOuterFor:
            extent = self._dimension_extent(stmt.dimension)
            self._panel_widths[stmt.tile] = stmt.width
            try:
                for origin in range(0, extent, stmt.width):
                    self.panel_origins[stmt.tile] = origin
                    self._exec_stmt(stmt.body)
            finally:
                self.panel_origins.pop(stmt.tile, None)
                self._panel_widths.pop(stmt.tile, None)
            return
        if type(stmt) is SparseWindowFor:
            bound_origin = self.panel_origins.get(stmt.tile)
            width = self._panel_widths.get(stmt.tile)
            if bound_origin is None or width is None:
                raise LoopIROracleError(
                    "sparse window executed outside its panel's origin loop"
                )
            origin = bound_origin
            cursor_decl = self.decls[stmt.cursor.tensor]
            dimension = cursor_decl.dimensions[
                cursor_decl.levels[stmt.cursor.level].mode
            ]
            extent = self._dimension_extent(dimension)
            window_end = min(origin + width, extent)
            storage, start, end = self._segment(stmt.cursor)
            state = _CursorState(storage, stmt.cursor.level, start, end, aligned=True)
            self.cursors[stmt.cursor.cursor] = state
            try:
                while not state.exhausted:
                    coordinate = state.coordinate
                    if origin <= coordinate < window_end:
                        self.positions[stmt.position] = state.position
                        self.indices[stmt.coord_index] = coordinate
                        self._exec_stmt(stmt.body)
                    state.position += 1
            finally:
                self.indices.pop(stmt.coord_index, None)
                self.positions.pop(stmt.position, None)
                del self.cursors[stmt.cursor.cursor]
            return
        if type(stmt) is SparseFor:
            storage, start, end = self._segment(stmt.cursor)
            state = _CursorState(storage, stmt.cursor.level, start, end, aligned=True)
            self.cursors[stmt.cursor.cursor] = state
            try:
                while not state.exhausted:
                    self.positions[stmt.position] = state.position
                    self.indices[stmt.coord_index] = state.coordinate
                    self._exec_stmt(stmt.body)
                    state.position += 1
            finally:
                self.indices.pop(stmt.coord_index, None)
                self.positions.pop(stmt.position, None)
                del self.cursors[stmt.cursor.cursor]
            return
        if type(stmt) is MergedSparseFor:
            self._exec_merge(stmt)
            return
        if type(stmt) is AppendEntry:
            coords_tuple = tuple(self._eval_coord(coord) for coord in stmt.coords)
            value = self._eval_value(stmt.value)
            builder = self.builders.get(stmt.tensor)
            if builder is None:
                raise LoopIROracleError(
                    f"append target {self.decls[stmt.tensor].name} has no "
                    "assembly builder"
                )
            try:
                builder.append(coords_tuple, value)
            except LevelStorageError as error:
                raise LoopIROracleError(str(error)) from error
            return
        if type(stmt) is Store:
            target, last = self._locate_store(stmt.tensor, stmt.indices)
            target[last] = self._eval_value(stmt.value)
            return
        if type(stmt) is StoreReduce:
            if stmt.tensor in self.builders:
                # The semantic sparse accumulation form: merge by coordinate
                # now, assemble in order once at program exit.
                coords = tuple(self._eval_coord(index) for index in stmt.indices)
                entries = self.sparse_reductions.setdefault(stmt.tensor, {})
                contribution = self._eval_value(stmt.value)
                entries[coords] = entries.get(coords, 0.0) + contribution
                return
            target, last = self._locate_store(stmt.tensor, stmt.indices)
            contribution = self._eval_value(stmt.value)
            target[last] = target[last] + contribution
            return
        raise LoopIROracleError(f"unknown statement {type(stmt).__name__}")


def run_program(
    program: LoopProgram,
    inputs: Mapping[SymbolId, object],
    output_shapes: Mapping[SymbolId, Tuple[int, ...]],
) -> Dict[SymbolId, TensorValue]:
    """Verify then execute one LoopIR program over logical Python values.

    Dense input nesting and ``output_shapes`` are both expressed in logical
    mode order.  Physical level order belongs to :class:`LevelDecl` and to
    target/runtime bindings; callers crossing that boundary must map each
    physical extent through ``level.mode`` before invoking the semantic
    oracle.  Results remain logical nested containers regardless of layout.
    """

    verify_program(program)
    return _Oracle(program, inputs, output_shapes).run()


_Oracle._SPARSE_WORKSPACE_EXEC = {
    SparseWorkspaceRegion: _Oracle._exec_sparse_workspace_region,
    SparseWorkspaceInsert: _Oracle._exec_sparse_workspace_insert,
    SparseWorkspaceDrainFor: _Oracle._exec_sparse_workspace_drain,
}
