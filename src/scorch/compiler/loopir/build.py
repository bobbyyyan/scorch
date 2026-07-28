"""Construction API for production LoopIR programs.

:class:`LoopIRBuilder` is the supported way to construct well-formed LoopIR:
it owns the artifact-local :class:`~scorch.compiler.loopir.nodes.LoopIRNodeId`
and :class:`~scorch.compiler.loopir.nodes.DimensionId` allocation for one
program, so identical construction sequences always allocate identical
identities regardless of process history.  Tensor symbols and loop indices
are production :class:`~scorch.compiler.identity.SymbolId` /
:class:`~scorch.compiler.identity.IndexId` values: callers lowering from CIN
pass the CIN identities through unchanged (provenance), while hand-built
programs may ask the builder for fresh ones.

The builder performs no semantic validation — it only allocates identities
and owns tuple conversion.  ``verifier.verify_program`` remains the single
fail-closed authority; ``program`` deliberately does not call it so
adversarial tests can build malformed programs through the same API surface.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Optional, Sequence, Tuple

from ..identity import IndexId, SymbolId, new_index_id, new_symbol_id
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
    ParallelDiscipline,
    ParallelIntent,
    ParallelPart,
    ParallelSelection,
    ParallelWork,
    PositionId,
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
    SparseWorkSource,
    SparseWorkspaceDecl,
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
    WorkspaceDecl,
    WorkspaceId,
    WorkspaceRead,
    WorkspaceReduce,
    WorkspaceRegion,
)


def _max_identity_values(
    program: LoopProgram,
) -> Tuple[int, int, int, int, int, int, int, int]:
    """Scan one program for the next free value of every builder identity."""

    next_node = 0
    next_dimension = 0
    next_cursor = 0
    next_position = 0
    next_tile = 0
    next_workspace = 0
    next_relayout = 0
    next_result_tile = 0
    seen: set = set()
    pending: list = [program]

    def record_identity(value: object) -> bool:
        nonlocal next_dimension
        nonlocal next_cursor
        nonlocal next_position
        nonlocal next_tile
        nonlocal next_workspace
        nonlocal next_relayout
        nonlocal next_result_tile

        if type(value) is DimensionId and type(value.value) is int:
            next_dimension = max(next_dimension, value.value + 1)
        elif type(value) is CursorId and type(value.value) is int:
            next_cursor = max(next_cursor, value.value + 1)
        elif type(value) is PositionId and type(value.value) is int:
            next_position = max(next_position, value.value + 1)
        elif type(value) is TileId and type(value.value) is int:
            next_tile = max(next_tile, value.value + 1)
        elif type(value) is WorkspaceId and type(value.value) is int:
            next_workspace = max(next_workspace, value.value + 1)
        elif type(value) is RelayoutId and type(value.value) is int:
            next_relayout = max(next_relayout, value.value + 1)
        elif type(value) is ResultTileId and type(value.value) is int:
            next_result_tile = max(next_result_tile, value.value + 1)
        else:
            return False
        return True

    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        node_id = getattr(node, "node_id", None)
        if type(node_id) is LoopIRNodeId and type(node_id.value) is int:
            next_node = max(next_node, node_id.value + 1)
        # Only declared schema fields participate in artifact identity.
        # A forged extra ``__dict__`` entry is neither verifier-visible
        # semantics nor canonical serialization and therefore must not move
        # a continuation allocator.
        for field in fields(type(node)):
            value = getattr(node, field.name, None)
            if record_identity(value):
                continue
            if isinstance(value, LoopIRNode):
                pending.append(value)
            elif type(value) is tuple:
                for child in value:
                    if not record_identity(child) and isinstance(child, LoopIRNode):
                        pending.append(child)
    return (
        next_node,
        next_dimension,
        next_cursor,
        next_position,
        next_tile,
        next_workspace,
        next_relayout,
        next_result_tile,
    )


class LoopIRBuilder:
    """Allocate artifact-local identities and assemble one LoopIR program."""

    def __init__(self) -> None:
        self._next_node_id = 0
        self._next_dimension_id = 0
        self._next_cursor_id = 0
        self._next_position_id = 0
        self._next_tile_id = 0
        self._next_workspace_id = 0
        self._next_relayout_id = 0
        self._next_result_tile_id = 0

    @classmethod
    def resuming(cls, program: LoopProgram) -> "LoopIRBuilder":
        """A builder whose allocators continue past ``program``'s identities.

        Pure typed passes rebuild changed paths of an existing artifact; the
        nodes they create must not collide with retained identities, and the
        continuation must be deterministic — it depends only on the stored
        identity values, never on allocation history or object addresses.
        """

        builder = cls()
        (
            builder._next_node_id,
            builder._next_dimension_id,
            builder._next_cursor_id,
            builder._next_position_id,
            builder._next_tile_id,
            builder._next_workspace_id,
            builder._next_relayout_id,
            builder._next_result_tile_id,
        ) = _max_identity_values(program)
        return builder

    def _node_id(self) -> LoopIRNodeId:
        node_id = LoopIRNodeId(self._next_node_id)
        self._next_node_id += 1
        return node_id

    def new_dimension_id(self) -> DimensionId:
        """Allocate the next artifact-local dimension identity."""

        dimension = DimensionId(self._next_dimension_id)
        self._next_dimension_id += 1
        return dimension

    def new_cursor_id(self) -> CursorId:
        """Allocate the next artifact-local sparse-cursor identity."""

        cursor = CursorId(self._next_cursor_id)
        self._next_cursor_id += 1
        return cursor

    def new_position_id(self) -> PositionId:
        """Allocate the next artifact-local bound-position identity."""

        position = PositionId(self._next_position_id)
        self._next_position_id += 1
        return position

    def new_tile_id(self) -> TileId:
        """Allocate the next artifact-local split or panel identity."""

        tile = TileId(self._next_tile_id)
        self._next_tile_id += 1
        return tile

    def new_workspace_id(self) -> WorkspaceId:
        """Allocate the next artifact-local workspace identity."""

        workspace = WorkspaceId(self._next_workspace_id)
        self._next_workspace_id += 1
        return workspace

    def new_relayout_id(self) -> RelayoutId:
        """Allocate the next artifact-local staged-relayout identity."""

        relayout = RelayoutId(self._next_relayout_id)
        self._next_relayout_id += 1
        return relayout

    def new_result_tile_id(self) -> ResultTileId:
        """Allocate the next artifact-local heap result-tile identity."""

        result_tile = ResultTileId(self._next_result_tile_id)
        self._next_result_tile_id += 1
        return result_tile

    @staticmethod
    def new_symbol_id() -> SymbolId:
        """Allocate a fresh production tensor symbol for a hand-built program."""

        return new_symbol_id()

    @staticmethod
    def new_index_id() -> IndexId:
        """Allocate a fresh production loop index for a hand-built program."""

        return new_index_id()

    def dimension(self, name: str) -> DimensionDecl:
        """Declare a dimension with an identity owned by this builder.

        Callers that need malformed or externally identified declarations can
        construct :class:`DimensionDecl` directly. Keeping that adversarial
        surface out of the supported builder prevents an explicit identity
        from being reissued later by the automatic allocator.
        """

        return DimensionDecl(self._node_id(), self.new_dimension_id(), name)

    def level(self, kind: LevelKind, mode: int) -> LevelDecl:
        return LevelDecl(self._node_id(), kind, mode)

    def dense_levels(self, rank: int) -> Tuple[LevelDecl, ...]:
        """Identity-mode-order all-dense levels for a rank-``rank`` tensor."""

        return tuple(self.level(LevelKind.DENSE, mode) for mode in range(rank))

    def tensor(
        self,
        symbol: SymbolId,
        name: str,
        dtype: ScalarType,
        dimensions: Sequence[DimensionId],
        levels: Sequence[LevelDecl],
    ) -> TensorDecl:
        return TensorDecl(
            self._node_id(),
            symbol,
            name,
            dtype,
            tuple(dimensions),
            tuple(levels),
        )

    def index_value(self, index: IndexId) -> IndexValue:
        return IndexValue(self._node_id(), index)

    def float_const(self, value: float) -> FloatConst:
        return FloatConst(self._node_id(), value)

    def root_position(self) -> RootPosition:
        return RootPosition(self._node_id())

    def dense_position(
        self, tensor: SymbolId, level: int, parent: Expr, coord: Expr
    ) -> DensePosition:
        return DensePosition(self._node_id(), tensor, level, parent, coord)

    def position_value(self, position: PositionId) -> PositionValue:
        return PositionValue(self._node_id(), position)

    def cursor_value(
        self, cursor: CursorId, default: "Expr | None" = None
    ) -> CursorValue:
        return CursorValue(self._node_id(), cursor, default)

    def sparse_cursor(
        self, cursor: CursorId, tensor: SymbolId, level: int, parent: Expr
    ) -> SparseCursorDecl:
        return SparseCursorDecl(self._node_id(), cursor, tensor, level, parent)

    def sparse_for(
        self,
        cursor: SparseCursorDecl,
        position: PositionId,
        coord_index: IndexId,
        body: Block,
    ) -> SparseFor:
        return SparseFor(self._node_id(), cursor, position, coord_index, body)

    def merged_sparse_for(
        self,
        mode: MergeMode,
        cursors: Sequence[SparseCursorDecl],
        coord_index: IndexId,
        body: Block,
        positions: Sequence["PositionId | None"] = (),
    ) -> MergedSparseFor:
        return MergedSparseFor(
            self._node_id(),
            mode,
            tuple(cursors),
            coord_index,
            body,
            tuple(positions),
        )

    def append_entry(
        self, tensor: SymbolId, coords: Sequence[Expr], value: Expr
    ) -> AppendEntry:
        return AppendEntry(self._node_id(), tensor, tuple(coords), value)

    def load(self, tensor: SymbolId, indices: Sequence[Expr]) -> Load:
        return Load(self._node_id(), tensor, tuple(indices))

    def binary(self, op: BinaryOp, lhs: Expr, rhs: Expr) -> BinaryExpr:
        return BinaryExpr(self._node_id(), op, lhs, rhs)

    def block(self, statements: Sequence[Stmt]) -> Block:
        return Block(self._node_id(), tuple(statements))

    def dense_for(
        self, index: IndexId, dimension: DimensionId, body: Block
    ) -> DenseFor:
        return DenseFor(self._node_id(), index, dimension, body)

    def tile_outer_for(
        self,
        tile: TileId,
        index: IndexId,
        dimension: DimensionId,
        width: int,
        body: Block,
    ) -> TileOuterFor:
        return TileOuterFor(self._node_id(), tile, index, dimension, width, body)

    def tile_inner_for(
        self,
        tile: TileId,
        index: IndexId,
        dimension: DimensionId,
        width: int,
        unroll: bool,
        body: Block,
    ) -> TileInnerFor:
        return TileInnerFor(
            self._node_id(), tile, index, dimension, width, unroll, body
        )

    def panel_outer_for(
        self,
        tile: TileId,
        index: IndexId,
        dimension: DimensionId,
        width: int,
        bound_tensor: SymbolId,
        bound_level: int,
        body: Block,
    ) -> PanelOuterFor:
        return PanelOuterFor(
            self._node_id(),
            tile,
            index,
            dimension,
            width,
            bound_tensor,
            bound_level,
            body,
        )

    def sparse_window_for(
        self,
        tile: TileId,
        cursor: SparseCursorDecl,
        position: PositionId,
        coord_index: IndexId,
        body: Block,
    ) -> SparseWindowFor:
        return SparseWindowFor(
            self._node_id(), tile, cursor, position, coord_index, body
        )

    def workspace_decl(
        self,
        workspace: WorkspaceId,
        name: str,
        dtype: ScalarType,
        tile: TileId,
    ) -> WorkspaceDecl:
        return WorkspaceDecl(self._node_id(), workspace, name, dtype, tile)

    def workspace_region(
        self,
        workspace: WorkspaceDecl,
        producer: Block,
        consumer: Block,
    ) -> WorkspaceRegion:
        return WorkspaceRegion(self._node_id(), workspace, producer, consumer)

    def workspace_read(self, workspace: WorkspaceId, coord: Expr) -> WorkspaceRead:
        return WorkspaceRead(self._node_id(), workspace, coord)

    def workspace_reduce(
        self,
        workspace: WorkspaceId,
        coord: Expr,
        op: ReduceOp,
        value: Expr,
    ) -> WorkspaceReduce:
        return WorkspaceReduce(self._node_id(), workspace, coord, op, value)

    def sparse_workspace_decl(
        self,
        workspace: WorkspaceId,
        name: str,
        dtype: ScalarType,
        drain_dimension: DimensionId,
    ) -> SparseWorkspaceDecl:
        return SparseWorkspaceDecl(
            self._node_id(), workspace, name, dtype, drain_dimension
        )

    def sparse_workspace_region(
        self,
        workspace: SparseWorkspaceDecl,
        producer: Block,
        consumer: Block,
    ) -> SparseWorkspaceRegion:
        return SparseWorkspaceRegion(self._node_id(), workspace, producer, consumer)

    def sparse_workspace_insert(
        self,
        workspace: WorkspaceId,
        coord: Expr,
        op: ReduceOp,
        value: Expr,
    ) -> SparseWorkspaceInsert:
        return SparseWorkspaceInsert(self._node_id(), workspace, coord, op, value)

    def sparse_workspace_drain_for(
        self,
        workspace: WorkspaceId,
        index: IndexId,
        body: Block,
    ) -> SparseWorkspaceDrainFor:
        return SparseWorkspaceDrainFor(self._node_id(), workspace, index, body)

    def sparse_workspace_value(self, workspace: WorkspaceId) -> SparseWorkspaceValue:
        return SparseWorkspaceValue(self._node_id(), workspace)

    def relayout_decl(
        self,
        relayout: RelayoutId,
        operand: SymbolId,
        panel: TileId,
        pack: TileId,
        scope: RelayoutScope,
    ) -> RelayoutDecl:
        return RelayoutDecl(self._node_id(), relayout, operand, panel, pack, scope)

    def relayout_stage(self, decl: RelayoutDecl, body: Block) -> RelayoutStage:
        return RelayoutStage(self._node_id(), decl, body)

    def staged_read(self, relayout: RelayoutId, indices: Sequence[Expr]) -> StagedRead:
        return StagedRead(self._node_id(), relayout, tuple(indices))

    def result_tile_decl(
        self,
        result_tile: ResultTileId,
        result: SymbolId,
        pack: TileId,
    ) -> ResultTileDecl:
        return ResultTileDecl(self._node_id(), result_tile, result, pack)

    def result_tile_region(self, decl: ResultTileDecl, body: Block) -> ResultTileRegion:
        return ResultTileRegion(self._node_id(), decl, body)

    def tiled_reduce(
        self,
        result_tile: ResultTileId,
        indices: Sequence[Expr],
        op: ReduceOp,
        value: Expr,
    ) -> TiledReduce:
        return TiledReduce(self._node_id(), result_tile, tuple(indices), op, value)

    def store(self, tensor: SymbolId, indices: Sequence[Expr], value: Expr) -> Store:
        return Store(self._node_id(), tensor, tuple(indices), value)

    def store_reduce(
        self,
        tensor: SymbolId,
        indices: Sequence[Expr],
        op: ReduceOp,
        value: Expr,
    ) -> StoreReduce:
        return StoreReduce(self._node_id(), tensor, tuple(indices), op, value)

    def sparse_work_source(self, tensor: SymbolId, level: int) -> SparseWorkSource:
        return SparseWorkSource(self._node_id(), tensor, level)

    def parallel_work(
        self,
        rows: DimensionId,
        nnz: Optional[SparseWorkSource] = None,
    ) -> ParallelWork:
        return ParallelWork(self._node_id(), rows, nnz)

    def parallel_selection(
        self,
        index: IndexId,
        part: ParallelPart,
        discipline: ParallelDiscipline,
        work: ParallelWork,
        intent: ParallelIntent,
    ) -> ParallelSelection:
        return ParallelSelection(
            self._node_id(),
            index,
            part,
            discipline,
            work,
            intent,
        )

    def program(
        self,
        dimensions: Sequence[DimensionDecl],
        tensors: Sequence[TensorDecl],
        inputs: Sequence[SymbolId],
        outputs: Sequence[SymbolId],
        body: Block,
        parallel: Optional[ParallelSelection] = None,
    ) -> LoopProgram:
        return LoopProgram(
            self._node_id(),
            tuple(dimensions),
            tuple(tensors),
            tuple(inputs),
            tuple(outputs),
            body,
            parallel,
        )
