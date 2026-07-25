"""Pure typed scheduling passes over verified production LoopIR.

Phase 6 applies a verified :class:`~scorch.compiler.loop_plan.LoopPlan` to a
verified unscheduled LoopIR program through pure typed transformations,
replacing the legacy scheduler's private CIN tree surgery for the migrated
schedule families:

- :func:`reorder_loops` permutes a single-chain loop nest into the plan's
  logical loop order;
- :func:`apply_affine_tile` strip-mines one dense logical loop into a
  :class:`~scorch.compiler.loopir.nodes.TileOuterFor` /
  :class:`~scorch.compiler.loopir.nodes.TileInnerFor` pair with an
  artifact-local :class:`~scorch.compiler.loopir.nodes.TileId`, mirroring
  the legacy placement semantics (``outermost`` / ``child_of`` /
  ``at_depth`` resolved against the current chain, applied in plan tile
  order);
- :func:`apply_stack_tile` strip-mines the trailing dense free loop of a
  reduction chain into the same origin/point pair *and* materializes the
  legacy stack workspace as a structured
  :class:`~scorch.compiler.loopir.nodes.WorkspaceRegion`: the reduction
  loops move into the region's producer (accumulating into the workspace
  through the point loop), and the consumer copies the tile out with a
  second point loop of the same split — exactly the legacy
  ``wksp[kTile]`` producer/consumer shape ``insert_workspace`` +
  ``add_tile`` produce.  The workspace's extent is intrinsic to the
  split's width; allocation and zero-reset are intrinsic region-entry
  semantics.  Placement resolves against the loops that remain above the
  region, mirroring the legacy prefix-of-``Where`` rule;
- :func:`apply_panel_tile` windows one single-cursor compressed
  coordinate loop into a :class:`~scorch.compiler.loopir.nodes.PanelOuterFor`
  / :class:`~scorch.compiler.loopir.nodes.SparseWindowFor` pair,
  mirroring the legacy sparse-panel family: the CSR dense-parent row loop
  is the plan-mandated parallel loop, the panel origin is inserted
  strictly above it (``outermost`` or ``child_of`` an outermost affine
  origin loop), and the plan's ``PanelBound`` is materialized into the
  panel's structural (tensor, level) extent source;
- :func:`apply_relayout` stages the packed tile-ijk contraction's dense
  operand behind a typed
  :class:`~scorch.compiler.loopir.nodes.RelayoutStage` region at the
  plan-selected scope (the panel's window rows or the whole panel axis)
  and structurally replaces the operand's verifier-proven **unique**
  ``Load`` occurrence with a
  :class:`~scorch.compiler.loopir.nodes.StagedRead` carrying the fresh
  region identity — the recorded access-identity decision: no occurrence
  identity is added to ``Load`` because the audited family admits exactly
  one occurrence and the pass proves it before redirecting.

:func:`apply_schedule_plan` drives both and returns a
:class:`ScheduledLoopIR` artifact that retains the unscheduled base
program, the exact plan, the scheduled program, and per-loop provenance
``(tile, logical index, part)`` back to the base loops.

Pass discipline (the binding design decisions):

- passes are pure: they never mutate their input artifact, never call the
  legacy lowerers, never parse rendered C++, and never depend on mutable
  phase-order state — everything they need is in the verified program and
  the verified plan;
- inputs and outputs are verified: each pass runs the fail-closed LoopIR
  verifier on its input and on its rebuilt output, and its own legality
  checks fail closed with stable ``SchedulePassError`` codes before any
  rebuilding starts;
- identity is deterministic: rebuilt nodes take fresh
  ``LoopIRNodeId``/``TileId`` values from a builder resumed past the input
  artifact's identities, so identical inputs produce identical outputs;
  retained subtrees (leaves, cursor declarations, access expressions) are
  structurally shared, which is safe because every node is frozen;
- passes are canonically idempotent: reapplying :func:`reorder_loops` with
  the same order is a documented no-op, and re-running
  :func:`apply_schedule_plan` on the same inputs produces a program with
  the same canonical dump.

Legality model of this subset:

- loop *binding* dependencies are pass-verified: a sparse loop's cursor
  parent chain may only reference loops that remain outer after the
  permutation (``reorder_sparse_dependency``), which is exactly the sparse
  parent-dominance direction the verifier enforces structurally;
- ordered CSR assembly pins its nest: programs whose leaf is an
  ``AppendEntry`` reject any non-identity permutation
  (``reorder_ordered_assembly``), because appends must stay
  lexicographically ordered;
- ADD reductions may be reassociated by any legal permutation or split —
  that is the migrated family's explicit reduction contract (``StoreReduce``
  ADD into a zero-initialized output), the same freedom the legacy
  scheduler exercises;
- per-tensor storage-order/nest-order compatibility is deliberately *not*
  a scheduling-legality rule: it is an emission constraint owned by the
  target lowering (``unsupported_loop_order``), where the legacy pipeline's
  equivalent boundary lives.  A semantically legal reorder the target
  cannot emit fails closed there, never silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NoReturn, Optional, Sequence, Set, Tuple, Union

from ..diagnostics import VerificationError
from ..identity import IndexId
from ..loop_plan import (
    MAX_AFFINE_TILE_WIDTH,
    LoopPart,
    LoopPlan,
    LoopPlacement,
    LoopRef,
    LoopTile,
    OperandRelayout,
    PanelBound,
    PlacementKind,
    _validate_loop_plan_structure,
)
from .build import LoopIRBuilder
from .nodes import (
    AppendEntry,
    BinaryExpr,
    Block,
    CursorValue,
    DenseFor,
    DensePosition,
    Expr,
    IndexValue,
    LevelKind,
    Load,
    LoopProgram,
    MergedSparseFor,
    PanelOuterFor,
    PositionId,
    PositionValue,
    ReduceOp,
    RelayoutScope,
    RelayoutStage,
    SparseCursorDecl,
    SparseFor,
    SparseWindowFor,
    StagedRead,
    Stmt,
    Store,
    StoreReduce,
    TileId,
    TileInnerFor,
    TileOuterFor,
    WorkspaceRead,
    WorkspaceReduce,
    WorkspaceRegion,
)
from .verifier import verify_program

_LoopNode = Union[
    DenseFor,
    SparseFor,
    MergedSparseFor,
    TileOuterFor,
    TileInnerFor,
    PanelOuterFor,
    SparseWindowFor,
]
_LeafNode = Union[Store, StoreReduce, AppendEntry]
_ChainEnd = Union[Store, StoreReduce, AppendEntry, WorkspaceRegion]

_LOOP_TYPES = (
    DenseFor,
    SparseFor,
    MergedSparseFor,
    TileOuterFor,
    TileInnerFor,
    PanelOuterFor,
    SparseWindowFor,
)
_LEAF_TYPES = (Store, StoreReduce, AppendEntry)
_PANEL_TYPES = (PanelOuterFor, SparseWindowFor)


@dataclass(frozen=True)
class SchedulePassDefect:
    """One immutable scheduling-pass failure: stable code and message."""

    code: str
    message: str


class SchedulePassError(Exception):
    """A verified program/plan pair is outside the migrated schedule families."""

    def __init__(self, defect: SchedulePassDefect) -> None:
        super().__init__(f"{defect.code}: {defect.message}")
        self.defect = defect


def _fail(code: str, message: str) -> NoReturn:
    raise SchedulePassError(SchedulePassDefect(code, message))


@dataclass(frozen=True)
class ScheduledLoopProvenance:
    """One scheduled-chain loop's provenance back to the base program.

    ``part`` is ``LOGICAL`` for an unsplit loop (``tile`` is ``None``) and
    ``OUTER``/``INNER`` for the two loops of one affine split (``tile`` is
    that split's artifact-local identity); ``index`` is always the logical
    loop the base program bound.
    """

    tile: Optional[TileId]
    index: IndexId
    part: LoopPart


@dataclass(frozen=True)
class ScheduledLoopIR:
    """One applied schedule: base program, exact plan, result, provenance.

    ``program`` is ordinary verified LoopIR — scheduled form is a state of
    the same node model, not a second schema — and ``loops`` lists the
    scheduled chain outermost-first.
    """

    base_program: LoopProgram
    plan: LoopPlan
    program: LoopProgram
    loops: Tuple[ScheduledLoopProvenance, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.loops, (tuple, list)):
            raise TypeError(
                "ScheduledLoopIR.loops must be a tuple or list of provenance values"
            )
        object.__setattr__(self, "loops", tuple(self.loops))


def _loop_key(node: _LoopNode) -> Tuple[IndexId, LoopPart]:
    if type(node) is DenseFor:
        return node.index, LoopPart.LOGICAL
    if type(node) is TileOuterFor:
        return node.index, LoopPart.OUTER
    if type(node) is TileInnerFor:
        return node.index, LoopPart.INNER
    if type(node) is PanelOuterFor:
        return node.index, LoopPart.OUTER
    if type(node) is SparseWindowFor:
        return node.coord_index, LoopPart.INNER
    if type(node) is SparseFor:
        return node.coord_index, LoopPart.LOGICAL
    assert type(node) is MergedSparseFor
    return node.coord_index, LoopPart.LOGICAL


def _decompose_body(
    body: Stmt,
    *,
    leaf_types: Tuple[type, ...],
    allow_empty: bool,
    relayout_sink: Optional[List[RelayoutStage]] = None,
) -> Tuple[List[_LoopNode], Stmt]:
    """Walk one single-statement chain of loops down to its terminator.

    A :class:`RelayoutStage` wrapper is a chain element only for callers
    that pass a ``relayout_sink`` (provenance and erasure, which must read
    relayout-scheduled programs); every scheduling pass keeps the default
    and therefore refuses already-relayouted chains with a stable code.
    """

    loops: List[_LoopNode] = []
    while True:
        if type(body) is not Block or len(body.statements) != 1:
            _fail(
                "unsupported_schedule_shape",
                "scheduling passes expect a single-statement loop chain "
                "over one leaf statement",
            )
        only = body.statements[0]
        if type(only) in _LOOP_TYPES:
            loops.append(only)  # type: ignore[arg-type]
            body = only.body  # type: ignore[attr-defined]
            continue
        if type(only) is RelayoutStage and relayout_sink is not None:
            relayout_sink.append(only)
            body = only.body
            continue
        if type(only) in leaf_types:
            if not loops and not allow_empty:
                _fail(
                    "unsupported_schedule_shape",
                    "scheduling passes require at least one loop",
                )
            return loops, only
        _fail(
            "unsupported_schedule_shape",
            f"unsupported chain statement {type(only).__name__}",
        )


def _decompose_chain(
    program: LoopProgram,
    *,
    relayout_sink: Optional[List[RelayoutStage]] = None,
) -> Tuple[List[_LoopNode], _ChainEnd]:
    """Split one migrated-family program into its loop chain and terminator.

    The terminator is an ordinary store/append leaf, or — for programs a
    stack tile already transformed — one :class:`WorkspaceRegion`.
    """

    loops, end = _decompose_body(
        program.body,
        leaf_types=(*_LEAF_TYPES, WorkspaceRegion),
        allow_empty=False,
        relayout_sink=relayout_sink,
    )
    return loops, end  # type: ignore[return-value]


def _decompose_region(
    region: WorkspaceRegion,
) -> Tuple[List[_LoopNode], WorkspaceReduce, List[_LoopNode], StoreReduce]:
    """Split one workspace region into its producer and consumer chains."""

    producer_loops, producer_leaf = _decompose_body(
        region.producer, leaf_types=(WorkspaceReduce,), allow_empty=False
    )
    consumer_loops, consumer_leaf = _decompose_body(
        region.consumer, leaf_types=(StoreReduce,), allow_empty=False
    )
    return (
        producer_loops,
        producer_leaf,  # type: ignore[return-value]
        consumer_loops,
        consumer_leaf,  # type: ignore[return-value]
    )


def _wrap_loops(
    builder: LoopIRBuilder,
    loops: Sequence[_LoopNode],
    body: Block,
) -> Block:
    """Wrap fresh copies of ``loops`` (innermost last) around one body."""

    for node in reversed(tuple(loops)):
        loop: Stmt
        if type(node) is DenseFor:
            loop = builder.dense_for(node.index, node.dimension, body)
        elif type(node) is TileOuterFor:
            loop = builder.tile_outer_for(
                node.tile, node.index, node.dimension, node.width, body
            )
        elif type(node) is TileInnerFor:
            loop = builder.tile_inner_for(
                node.tile,
                node.index,
                node.dimension,
                node.width,
                node.unroll,
                body,
            )
        elif type(node) is PanelOuterFor:
            loop = builder.panel_outer_for(
                node.tile,
                node.index,
                node.dimension,
                node.width,
                node.bound_tensor,
                node.bound_level,
                body,
            )
        elif type(node) is SparseWindowFor:
            loop = builder.sparse_window_for(
                node.tile, node.cursor, node.position, node.coord_index, body
            )
        elif type(node) is SparseFor:
            loop = builder.sparse_for(
                node.cursor, node.position, node.coord_index, body
            )
        else:
            assert type(node) is MergedSparseFor
            loop = builder.merged_sparse_for(
                node.mode, node.cursors, node.coord_index, body
            )
        body = builder.block((loop,))
    return body


def _rebuild_program(
    program: LoopProgram,
    builder: LoopIRBuilder,
    loops: Sequence[_LoopNode],
    leaf: _ChainEnd,
) -> LoopProgram:
    """Reassemble the chain with fresh loop/block nodes and shared subtrees."""

    body = _wrap_loops(builder, loops, builder.block((leaf,)))
    rebuilt = builder.program(
        program.dimensions,
        program.tensors,
        program.inputs,
        program.outputs,
        body,
    )
    verify_program(rebuilt)
    return rebuilt


def _parent_chain_references(
    parent: Expr,
    indices: Set[IndexId],
    positions: Set[PositionId],
) -> None:
    """Collect the loop bindings one cursor parent chain depends on."""

    pending: List[Expr] = [parent]
    while pending:
        expr = pending.pop()
        if type(expr) is DensePosition:
            pending.append(expr.parent)
            pending.append(expr.coord)
        elif type(expr) is IndexValue:
            indices.add(expr.index)
        elif type(expr) is PositionValue:
            positions.add(expr.position)


def _loop_dependencies(
    loops: Sequence[_LoopNode],
) -> List[Set[int]]:
    """Chain positions each loop's cursor parents require to stay outer."""

    position_owner: Dict[PositionId, int] = {}
    index_owner: Dict[IndexId, int] = {}
    for chain_position, node in enumerate(loops):
        if type(node) is SparseFor:
            position_owner[node.position] = chain_position
        index_owner[_loop_key(node)[0]] = chain_position

    dependencies: List[Set[int]] = []
    for node in loops:
        cursors: Tuple[SparseCursorDecl, ...]
        if type(node) is SparseFor:
            cursors = (node.cursor,)
        elif type(node) is MergedSparseFor:
            cursors = node.cursors
        else:
            dependencies.append(set())
            continue
        indices: Set[IndexId] = set()
        positions: Set[PositionId] = set()
        for cursor in cursors:
            _parent_chain_references(cursor.parent, indices, positions)
        required: Set[int] = set()
        for index in indices:
            owner = index_owner.get(index)
            if owner is not None:
                required.add(owner)
        for position in positions:
            owner = position_owner.get(position)
            if owner is not None:
                required.add(owner)
        dependencies.append(required)
    return dependencies


def reorder_loops(program: LoopProgram, loop_order: Sequence[IndexId]) -> LoopProgram:
    """Permute a verified single-chain nest into ``loop_order``.

    The order names logical loops only; programs that already carry an
    affine split fail closed (split placement is the tiling pass's job).
    Requesting the current order is a legal no-op returning the input
    program unchanged.
    """

    verify_program(program)
    loops, leaf = _decompose_chain(program)
    if type(leaf) is WorkspaceRegion:
        _fail(
            "reorder_split_chain",
            "loop reorder operates on unsplit logical chains; the chain "
            "already carries a workspace region",
        )
    for node in loops:
        if type(node) in (TileOuterFor, TileInnerFor, *_PANEL_TYPES):
            _fail(
                "reorder_split_chain",
                "loop reorder operates on unsplit logical chains; apply it "
                "before affine or panel tiling",
            )
    try:
        requested = tuple(loop_order)
    except (TypeError, RecursionError):
        _fail(
            "reorder_invalid_order",
            "requested order must be a finite sequence of IndexId values",
        )
    if any(
        type(index) is not IndexId or type(getattr(index, "value", None)) is not int
        for index in requested
    ):
        _fail(
            "reorder_invalid_order",
            "requested order must contain exact int-valued IndexId values",
        )
    chain_keys = [_loop_key(node)[0] for node in loops]
    if len(set(requested)) != len(requested):
        _fail("reorder_incomplete_order", "requested order repeats a loop")
    if set(requested) != set(chain_keys) or len(requested) != len(chain_keys):
        _fail(
            "reorder_incomplete_order",
            "requested order must name every chain loop exactly once",
        )
    if list(requested) == chain_keys:
        return program

    if type(leaf) is AppendEntry:
        _fail(
            "reorder_ordered_assembly",
            "ordered sparse assembly pins its nest order; a non-identity "
            "permutation would break append ordering",
        )

    old_position_of: Dict[IndexId, int] = {
        index: position for position, index in enumerate(chain_keys)
    }
    new_positions = [old_position_of[index] for index in requested]
    dependencies = _loop_dependencies(loops)
    placed: Set[int] = set()
    for old_position in new_positions:
        missing = dependencies[old_position] - placed
        if missing:
            _fail(
                "reorder_sparse_dependency",
                "a sparse loop's cursor parent chain must stay dominated by "
                "the loops it references",
            )
        placed.add(old_position)

    builder = LoopIRBuilder.resuming(program)
    reordered = [loops[old_position] for old_position in new_positions]
    return _rebuild_program(program, builder, reordered, leaf)


def _placement_depth(
    loops: Sequence[_LoopNode],
    placement: LoopPlacement,
    target_position: int,
) -> int:
    """Mirror the legacy placement semantics against the current chain."""

    if placement.kind is PlacementKind.OUTERMOST:
        depth = 0
    elif placement.kind is PlacementKind.CHILD_OF:
        parent = placement.parent
        assert parent is not None
        parent_positions = [
            position
            for position, node in enumerate(loops)
            if _loop_key(node) == (parent.index_id, parent.part)
        ]
        if not parent_positions:
            _fail(
                "tile_invalid_placement",
                "child_of placement must name a loop of the current chain",
            )
        depth = parent_positions[0] + 1
    else:
        assert placement.kind is PlacementKind.AT_DEPTH
        assert placement.depth is not None
        depth = placement.depth
        if depth < 0 or depth > len(loops):
            _fail(
                "tile_invalid_placement",
                f"at_depth placement is outside the chain range 0..{len(loops)}",
            )
    if depth > target_position:
        _fail(
            "tile_invalid_placement",
            "the origin loop must dominate its point loop",
        )
    return depth


def apply_affine_tile(program: LoopProgram, tile: LoopTile) -> LoopProgram:
    """Strip-mine one dense logical loop of a verified single-chain nest."""

    verify_program(program)
    if type(tile) is not LoopTile:
        raise TypeError("apply_affine_tile expects a LoopTile")
    try:
        _validate_loop_plan_structure(LoopPlan(loop_order=(), tiles=(tile,)))
    except (VerificationError, AttributeError, TypeError, ValueError) as error:
        _fail("invalid_schedule_tile", str(error))
    if type(tile.loop) is not LoopRef or tile.loop.part is not LoopPart.LOGICAL:
        _fail(
            "tile_target_not_logical",
            "affine tiles must target one unsplit logical loop",
        )
    if tile.kind != "affine":
        _fail(
            "unsupported_schedule_panel",
            "sparse coordinate panel tiling is not a migrated schedule " "family",
        )
    if tile.accumulation != "direct":
        _fail(
            "unsupported_schedule_accumulation",
            f"{tile.accumulation!r} accumulation needs the workspace "
            "families; only direct accumulation is migrated",
        )
    if tile.parallel:
        _fail(
            "unsupported_schedule_parallel",
            "explicit parallel tile selection is not a migrated schedule " "family",
        )

    loops, leaf = _decompose_chain(program)
    if any(type(node) in _PANEL_TYPES for node in loops):
        _fail(
            "unsupported_schedule_shape",
            "affine tiling over a panel-scheduled chain is not migrated; "
            "the panel tile applies after every affine tile",
        )
    target_index = tile.loop.index_id
    for node in loops:
        key = _loop_key(node)
        if key[0] == target_index and key[1] is not LoopPart.LOGICAL:
            _fail(
                "tile_target_already_split",
                "the requested loop already carries an affine split",
            )
    target_positions = [
        position
        for position, node in enumerate(loops)
        if _loop_key(node) == (target_index, LoopPart.LOGICAL)
    ]
    if not target_positions:
        _fail(
            "tile_target_missing",
            "the requested tile loop is not part of the loop chain",
        )
    target_position = target_positions[0]
    target = loops[target_position]
    if type(target) is not DenseFor:
        _fail(
            "tile_target_not_dense",
            "affine tiling cannot split a sparse coordinate loop; windowed "
            "compressed iteration is the panel family's job",
        )
    if (
        type(tile.width) is not int
        or tile.width < 1
        or tile.width > MAX_AFFINE_TILE_WIDTH
    ):
        _fail(
            "tile_invalid_width",
            "affine tile widths must be positive ints representable by the "
            "C++ constexpr int target",
        )
    if type(tile.placement) is not LoopPlacement:
        _fail(
            "tile_invalid_placement",
            "affine tile placement must be an exact LoopPlacement",
        )
    _check_placement_shape(tile.placement)

    depth = _placement_depth(loops, tile.placement, target_position)

    builder = LoopIRBuilder.resuming(program)
    tile_id = builder.new_tile_id()
    inner = builder.tile_inner_for(
        tile_id,
        target.index,
        target.dimension,
        tile.width,
        tile.unroll,
        target.body,
    )
    outer = builder.tile_outer_for(
        tile_id,
        target.index,
        target.dimension,
        tile.width,
        # The outer body is reassembled by _rebuild_program; this placeholder
        # block is replaced there and never enters the rebuilt tree.
        target.body,
    )
    new_chain: List[_LoopNode] = list(loops)
    new_chain[target_position] = inner
    new_chain.insert(depth, outer)
    return _rebuild_program(program, builder, new_chain, leaf)


def apply_stack_tile(program: LoopProgram, tile: LoopTile) -> LoopProgram:
    """Strip-mine the trailing free loop with stack (workspace) accumulation.

    Mirrors the legacy ``insert_workspace`` + ``add_tile`` composition in
    one fused rebuild: the reduction chain from the *last* reduction loop
    down becomes the region's producer accumulating into a width-cell
    workspace through the point loop, the consumer copies the tile out to
    the result with a second point loop, and the origin loop is inserted at
    the requested placement resolved against the loops that remain above
    the region (the legacy prefix-of-``Where`` rule).  The legacy legality
    boundary is mirrored exactly: the target must be the single trailing
    dense free loop after the last reduction of a dense-output ADD
    reduction (``stack_tile_target_invalid``), and the region may not
    replace the chain root (``stack_tile_root_scope``).
    """

    verify_program(program)
    if type(tile) is not LoopTile:
        raise TypeError("apply_stack_tile expects a LoopTile")
    try:
        _validate_loop_plan_structure(LoopPlan(loop_order=(), tiles=(tile,)))
    except (VerificationError, AttributeError, TypeError, ValueError) as error:
        _fail("invalid_schedule_tile", str(error))
    if type(tile.loop) is not LoopRef or tile.loop.part is not LoopPart.LOGICAL:
        _fail(
            "tile_target_not_logical",
            "affine tiles must target one unsplit logical loop",
        )
    if tile.kind != "affine":
        _fail(
            "unsupported_schedule_panel",
            "sparse coordinate panel tiling is not a migrated schedule " "family",
        )
    if tile.accumulation != "stack":
        _fail(
            "unsupported_schedule_accumulation",
            f"apply_stack_tile applies 'stack' accumulation only, got "
            f"{tile.accumulation!r}",
        )
    if tile.parallel:
        _fail(
            "unsupported_schedule_parallel",
            "explicit parallel tile selection is not a migrated schedule " "family",
        )
    if (
        type(tile.width) is not int
        or tile.width < 1
        or tile.width > MAX_AFFINE_TILE_WIDTH
    ):
        _fail(
            "tile_invalid_width",
            "affine tile widths must be positive ints representable by the "
            "C++ constexpr int target",
        )
    if type(tile.placement) is not LoopPlacement:
        _fail(
            "tile_invalid_placement",
            "affine tile placement must be an exact LoopPlacement",
        )
    _check_placement_shape(tile.placement)

    loops, leaf = _decompose_chain(program)
    if any(type(node) in _PANEL_TYPES for node in loops):
        _fail(
            "unsupported_schedule_shape",
            "stack tiling over a panel-scheduled chain is not migrated; "
            "the panel tile applies after every affine tile",
        )
    if type(leaf) is WorkspaceRegion:
        _fail(
            "stack_tile_target_invalid",
            "the chain already carries a workspace region; at most one "
            "stack accumulation exists per program",
        )
    if type(leaf) is not StoreReduce:
        _fail(
            "stack_tile_target_invalid",
            "stack accumulation requires a dense-output ADD reduction leaf",
        )
    target_index = tile.loop.index_id
    for node in loops:
        key = _loop_key(node)
        if key[0] == target_index and key[1] is not LoopPart.LOGICAL:
            _fail(
                "tile_target_already_split",
                "the requested loop already carries an affine split",
            )
    target_positions = [
        position
        for position, node in enumerate(loops)
        if _loop_key(node) == (target_index, LoopPart.LOGICAL)
    ]
    if not target_positions:
        _fail(
            "tile_target_missing",
            "the requested tile loop is not part of the loop chain",
        )
    target_position = target_positions[0]
    target = loops[target_position]

    leaf_index_ids: Set[IndexId] = set()
    for index in leaf.indices:
        if type(index) is not IndexValue:
            _fail(
                "stack_tile_target_invalid",
                "stack accumulation requires directly bound result " "coordinates",
            )
        leaf_index_ids.add(index.index)
    reduction_positions = [
        position
        for position, node in enumerate(loops)
        if _loop_key(node)[0] not in leaf_index_ids
    ]
    if not reduction_positions:
        _fail(
            "stack_tile_target_invalid",
            "stack accumulation is only supported for a trailing dense free "
            "dimension after a reduction",
        )
    last_reduction = max(reduction_positions)
    if target_position != last_reduction + 1 or target_position != len(loops) - 1:
        # Mirrors the legacy trailing-free-accumulator boundary: exactly one
        # free loop follows the last reduction and it is the stack target.
        _fail(
            "stack_tile_target_invalid",
            "stack accumulation is only supported for a trailing dense free "
            "dimension after a reduction",
        )
    if last_reduction == 0:
        _fail(
            "stack_tile_root_scope",
            "stack tiling cannot wrap a workspace inserted at the root scope",
        )
    if type(target) is not DenseFor:
        _fail(
            "tile_target_not_dense",
            "affine tiling cannot split a sparse coordinate loop; windowed "
            "compressed iteration is not represented",
        )

    prefix = loops[:last_reduction]
    depth = _placement_depth(prefix, tile.placement, last_reduction)

    result_symbol = program.outputs[0]
    result_decl = next(decl for decl in program.tensors if decl.symbol == result_symbol)

    builder = LoopIRBuilder.resuming(program)
    tile_id = builder.new_tile_id()
    workspace_id = builder.new_workspace_id()
    workspace_decl = builder.workspace_decl(
        workspace_id, "wksp", result_decl.dtype, tile_id
    )

    reduce_leaf = builder.workspace_reduce(
        workspace_id,
        builder.index_value(target.index),
        ReduceOp.ADD,
        leaf.value,
    )
    producer_body: Block = builder.block((reduce_leaf,))
    producer_point = builder.tile_inner_for(
        tile_id,
        target.index,
        target.dimension,
        tile.width,
        tile.unroll,
        producer_body,
    )
    producer_stmt: Stmt = producer_point
    for node in reversed(loops[last_reduction:target_position]):
        if type(node) is DenseFor:
            producer_stmt = builder.dense_for(
                node.index, node.dimension, builder.block((producer_stmt,))
            )
        elif type(node) is SparseFor:
            producer_stmt = builder.sparse_for(
                node.cursor,
                node.position,
                node.coord_index,
                builder.block((producer_stmt,)),
            )
        else:
            # Merged reductions are analysis-rejected upstream, and a split
            # loop between the last reduction and the target is impossible
            # (the target immediately follows the last reduction).
            _fail(
                "stack_tile_target_invalid",
                "stack accumulation supports dense and single-cursor sparse "
                "reduction loops only",
            )

    copy_out = builder.store_reduce(
        leaf.tensor,
        leaf.indices,
        leaf.op,
        builder.workspace_read(workspace_id, builder.index_value(target.index)),
    )
    consumer_point = builder.tile_inner_for(
        tile_id,
        target.index,
        target.dimension,
        tile.width,
        tile.unroll,
        builder.block((copy_out,)),
    )

    region = builder.workspace_region(
        workspace_decl,
        builder.block((producer_stmt,)),
        builder.block((consumer_point,)),
    )
    outer = builder.tile_outer_for(
        tile_id,
        target.index,
        target.dimension,
        tile.width,
        # The outer body is reassembled by _rebuild_program; this placeholder
        # block is replaced there and never enters the rebuilt tree.
        target.body,
    )
    new_chain: List[_LoopNode] = list(prefix)
    new_chain.insert(depth, outer)
    return _rebuild_program(program, builder, new_chain, region)


def apply_panel_tile(
    program: LoopProgram,
    tile: LoopTile,
    bound: PanelBound,
    parallel_loop: Optional[LoopRef],
) -> LoopProgram:
    """Window one compressed coordinate loop of a verified single-chain nest.

    Mirrors the legacy sparse-panel family exactly: the target must be a
    single-cursor compressed coordinate loop whose dominating parent is a
    dense position (the CSR form — a compressed-parent window is the
    ``panel_nested_compressed`` boundary), the plan's ``parallel_loop``
    must name that dense parent row loop (``panel_parallel_scope`` — the
    fact has no degrees of freedom, so validating it consumes it), and the
    panel origin is inserted strictly above the row loop: ``outermost``
    wraps the chain root, ``child_of`` wraps the loop below the named
    affine origin loop, and ``at_depth`` is not a supported panel
    placement (``panel_placement_invalid``).  The ``PanelBound`` fact is
    consumed by materializing it into the ``PanelOuterFor``'s structural
    (tensor, level) extent source after verifying it names a DENSE level
    of the window coordinate's own dimension (``panel_bound_mismatch``).
    """

    verify_program(program)
    if type(tile) is not LoopTile:
        raise TypeError("apply_panel_tile expects a LoopTile")
    try:
        _validate_loop_plan_structure(
            LoopPlan(
                loop_order=(),
                tiles=(tile,),
                panel_bounds=(bound,),
                parallel_loop=parallel_loop,
            )
        )
    except (VerificationError, AttributeError, TypeError, ValueError) as error:
        _fail("invalid_schedule_tile", str(error))
    if type(tile.loop) is not LoopRef or tile.loop.part is not LoopPart.LOGICAL:
        _fail(
            "tile_target_not_logical",
            "panel tiles must target one unsplit logical loop",
        )
    if tile.kind != "panel":
        _fail(
            "invalid_schedule_panel",
            "apply_panel_tile applies sparse panel tiles only",
        )
    if tile.accumulation != "direct":
        _fail(
            "invalid_schedule_panel",
            "sparse panel tiles require accumulation='direct'",
        )
    if tile.parallel:
        _fail(
            "invalid_schedule_panel",
            "sparse panel origin loops are serial; select the row loop "
            "through the plan's parallel_loop",
        )
    if (
        type(tile.width) is not int
        or tile.width < 1
        or tile.width > MAX_AFFINE_TILE_WIDTH
    ):
        _fail(
            "tile_invalid_width",
            "panel widths must be positive ints representable by the "
            "C++ constexpr int target",
        )
    if type(tile.placement) is not LoopPlacement:
        _fail(
            "tile_invalid_placement",
            "panel placement must be an exact LoopPlacement",
        )
    _check_placement_shape(tile.placement)
    if tile.placement.kind is PlacementKind.AT_DEPTH:
        _fail(
            "panel_placement_invalid",
            "sparse panel tiles support outermost or child_of placement only",
        )
    if bound.loop != tile.loop:
        _fail(
            "invalid_schedule_panel",
            "the PanelBound must reference exactly the panel tile's loop",
        )
    if parallel_loop is None:
        _fail(
            "panel_parallel_scope",
            "sparse panel tiling requires its CSR dense-parent row loop as "
            "the plan's parallel_loop",
        )
    if type(parallel_loop) is not LoopRef or parallel_loop.part is not LoopPart.LOGICAL:
        _fail(
            "panel_parallel_scope",
            "the panel's parallel row loop must be one logical loop",
        )

    loops, leaf = _decompose_chain(program)
    if any(type(node) in _PANEL_TYPES for node in loops):
        _fail(
            "invalid_schedule_panel",
            "the chain already carries a panel; at most one sparse panel "
            "exists per program",
        )
    if type(leaf) is WorkspaceRegion:
        _fail(
            "panel_target_invalid",
            "panel tiling over a workspace region is not a migrated " "composition",
        )
    if type(leaf) is AppendEntry:
        _fail(
            "panel_target_invalid",
            "panel tiling requires a dense result; ordered sparse assembly "
            "pins its nest",
        )
    target_index = tile.loop.index_id
    for node in loops:
        key = _loop_key(node)
        if key[0] == target_index and key[1] is not LoopPart.LOGICAL:
            _fail(
                "tile_target_already_split",
                "the requested loop already carries an affine split",
            )
    target_positions = [
        position
        for position, node in enumerate(loops)
        if _loop_key(node) == (target_index, LoopPart.LOGICAL)
    ]
    if not target_positions:
        _fail(
            "tile_target_missing",
            "the requested panel loop is not part of the loop chain",
        )
    target_position = target_positions[0]
    target = loops[target_position]
    if type(target) is not SparseFor:
        _fail(
            "panel_target_invalid",
            "panel tiling windows a single-cursor compressed coordinate "
            "loop; dense and merged loops are outside the family",
        )
    cursor = target.cursor
    parent = cursor.parent
    if type(parent) is PositionValue:
        _fail(
            "panel_nested_compressed",
            "panel tiling requires a CSR-style compressed level with a "
            "dense parent; compressed-parent windows are not migrated",
        )
    if type(parent) is not DensePosition or type(parent.coord) is not IndexValue:
        _fail(
            "panel_target_invalid",
            "panel tiling requires the window cursor's dominating parent "
            "to be a dense position over a directly bound row coordinate",
        )
    row_index = parent.coord.index
    if parallel_loop.index_id != row_index:
        _fail(
            "panel_parallel_scope",
            "the plan's parallel_loop must be the window cursor's dense "
            "parent row loop",
        )
    row_positions = [
        position
        for position, node in enumerate(loops)
        if _loop_key(node) == (row_index, LoopPart.LOGICAL) and type(node) is DenseFor
    ]
    if not row_positions:
        _fail(
            "panel_parallel_scope",
            "the panel's parallel row loop must be one plain dense chain "
            "loop; a split row loop is not migrated",
        )
    row_position = row_positions[0]

    if tile.placement.kind is PlacementKind.OUTERMOST:
        depth = 0
    else:
        assert tile.placement.kind is PlacementKind.CHILD_OF
        placement_parent = tile.placement.parent
        assert placement_parent is not None
        if placement_parent.part is not LoopPart.OUTER:
            _fail(
                "panel_placement_invalid",
                "a child_of panel placement must name an affine origin loop",
            )
        parent_positions = [
            position
            for position, node in enumerate(loops)
            if type(node) is TileOuterFor
            and _loop_key(node) == (placement_parent.index_id, LoopPart.OUTER)
        ]
        if not parent_positions:
            _fail(
                "panel_placement_invalid",
                "a child_of panel placement must name an affine origin loop "
                "of the current chain",
            )
        parent_position = parent_positions[0]
        if parent_position >= row_position:
            _fail(
                "panel_parallel_scope",
                "a sparse panel loop must surround the selected parallel " "row loop",
            )
        depth = parent_position + 1

    decls = {decl.symbol: decl for decl in program.tensors}
    cursor_decl_tensor = decls[cursor.tensor]
    dimension = cursor_decl_tensor.dimensions[
        cursor_decl_tensor.levels[cursor.level].mode
    ]
    bound_decl = decls.get(bound.tensor_id)
    if bound_decl is None:
        _fail(
            "panel_bound_mismatch",
            "the PanelBound tensor is not declared by the program",
        )
    if not 0 <= bound.level < len(bound_decl.levels):
        _fail(
            "panel_bound_mismatch",
            "the PanelBound level is outside its tensor's rank",
        )
    if bound_decl.levels[bound.level].kind is not LevelKind.DENSE:
        _fail(
            "panel_bound_mismatch",
            "the PanelBound must name a DENSE storage level",
        )
    if bound_decl.dimensions[bound_decl.levels[bound.level].mode] != dimension:
        _fail(
            "panel_bound_mismatch",
            "the PanelBound level must store the window coordinate's own " "dimension",
        )

    builder = LoopIRBuilder.resuming(program)
    tile_id = builder.new_tile_id()
    window = builder.sparse_window_for(
        tile_id,
        target.cursor,
        target.position,
        target.coord_index,
        target.body,
    )
    panel = builder.panel_outer_for(
        tile_id,
        target.coord_index,
        dimension,
        tile.width,
        bound.tensor_id,
        bound.level,
        # The panel body is reassembled by _rebuild_program; this
        # placeholder block is replaced there and never enters the tree.
        target.body,
    )
    new_chain: List[_LoopNode] = list(loops)
    new_chain[target_position] = window
    new_chain.insert(depth, panel)
    return _rebuild_program(program, builder, new_chain, leaf)


def _collect_operand_loads(root: Stmt, operand: object) -> List[Load]:
    """Every ``Load`` of one tensor anywhere in a verified subtree."""

    found: List[Load] = []
    pending: List[object] = [root]
    while pending:
        node = pending.pop()
        if type(node) is Load and node.tensor == operand:
            found.append(node)
        for attribute in vars(node).values():
            if isinstance(attribute, (Expr, Stmt, SparseCursorDecl)):
                pending.append(attribute)
            elif type(attribute) is tuple:
                pending.extend(
                    child
                    for child in attribute
                    if isinstance(child, (Expr, Stmt, SparseCursorDecl))
                )
    return found


def _replace_expr(
    expr: Expr,
    target: Expr,
    replacement: Expr,
    builder: LoopIRBuilder,
) -> Expr:
    """Rebuild the path to one child expression, sharing unchanged subtrees."""

    if expr is target:
        return replacement
    if type(expr) is BinaryExpr:
        lhs = _replace_expr(expr.lhs, target, replacement, builder)
        rhs = _replace_expr(expr.rhs, target, replacement, builder)
        if lhs is expr.lhs and rhs is expr.rhs:
            return expr
        return builder.binary(expr.op, lhs, rhs)
    if type(expr) is Load:
        indices = tuple(
            _replace_expr(index, target, replacement, builder) for index in expr.indices
        )
        if all(new is old for new, old in zip(indices, expr.indices)):
            return expr
        return builder.load(expr.tensor, indices)
    if type(expr) is StagedRead:
        indices = tuple(
            _replace_expr(index, target, replacement, builder) for index in expr.indices
        )
        if all(new is old for new, old in zip(indices, expr.indices)):
            return expr
        return builder.staged_read(expr.relayout, indices)
    if type(expr) is CursorValue and expr.default is not None:
        default = _replace_expr(expr.default, target, replacement, builder)
        if default is expr.default:
            return expr
        return builder.cursor_value(expr.cursor, default)
    if type(expr) is DensePosition:
        parent = _replace_expr(expr.parent, target, replacement, builder)
        coord = _replace_expr(expr.coord, target, replacement, builder)
        if parent is expr.parent and coord is expr.coord:
            return expr
        return builder.dense_position(expr.tensor, expr.level, parent, coord)
    if type(expr) is WorkspaceRead:
        coord = _replace_expr(expr.coord, target, replacement, builder)
        if coord is expr.coord:
            return expr
        return builder.workspace_read(expr.workspace, coord)
    return expr


def apply_relayout(program: LoopProgram, relayout: OperandRelayout) -> LoopProgram:
    """Stage one dense operand's pack strip behind a typed relayout region.

    Mirrors the audited legacy packed tile-ijk family exactly, on
    identities only.  The scheduled chain must be the five-loop shape the
    plan gate admits — the outermost affine pack origin, the panel origin
    directly below it, the parallel dense row loop, the panel's window,
    and the pack point loop — with a direct-accumulation dense-result
    leaf.  Every relayout fact is consumed exactly once: ``pack_loop`` /
    ``panel_loop`` select the two schedule pairs, ``scope_loop`` selects
    the region scope, ``row_loop`` is validated against the window's
    dense-parent row coordinate (the fact has no other legal value),
    ``strip_width`` against the pack split's width, the two operand
    levels against the operand declaration's dimension structure, and
    ``access_indices`` select the redirected read.

    The redirected read is the operand's **unique** ``Load`` occurrence —
    the pass proves uniqueness (``relayout_target_missing`` /
    ``relayout_ambiguous_access``) and replaces it structurally with a
    :class:`StagedRead` carrying the fresh region identity, so no
    occurrence identity, rendered name, or dynamic tag is ever needed;
    residual direct reads are re-checked after the rebuild.
    """

    verify_program(program)
    if type(relayout) is not OperandRelayout:
        raise TypeError("apply_relayout expects an OperandRelayout")
    try:
        _validate_loop_plan_structure(LoopPlan(loop_order=(), relayout=relayout))
    except (VerificationError, AttributeError, TypeError, ValueError) as error:
        _fail("invalid_schedule_relayout", str(error))
    for what, reference in (
        ("pack_loop", relayout.pack_loop),
        ("panel_loop", relayout.panel_loop),
        ("scope_loop", relayout.scope_loop),
        ("row_loop", relayout.row_loop),
    ):
        if type(reference) is not LoopRef or reference.part is not LoopPart.LOGICAL:
            _fail(
                "invalid_schedule_relayout",
                f"the relayout's {what} must name one logical loop",
            )
    if relayout.scope_loop not in (relayout.panel_loop, relayout.pack_loop):
        _fail(
            "invalid_schedule_relayout",
            "the relayout scope must be the panel loop or the pack loop",
        )

    loops, leaf = _decompose_chain(program)
    if type(leaf) is not StoreReduce:
        _fail(
            "invalid_schedule_relayout",
            "packed relayout requires a direct dense-result reduction leaf",
        )

    def _single_position(description: str, positions: List[int]) -> int:
        if len(positions) != 1:
            _fail(
                "invalid_schedule_relayout",
                f"packed relayout requires exactly one {description} in the "
                "scheduled chain",
            )
        return positions[0]

    pack_position = _single_position(
        "pack origin loop",
        [
            position
            for position, node in enumerate(loops)
            if type(node) is TileOuterFor and node.index == relayout.pack_loop.index_id
        ],
    )
    panel_position = _single_position(
        "panel origin loop",
        [
            position
            for position, node in enumerate(loops)
            if type(node) is PanelOuterFor
            and node.index == relayout.panel_loop.index_id
        ],
    )
    row_position = _single_position(
        "parallel row loop",
        [
            position
            for position, node in enumerate(loops)
            if type(node) is DenseFor and node.index == relayout.row_loop.index_id
        ],
    )
    window_position = _single_position(
        "panel window loop",
        [
            position
            for position, node in enumerate(loops)
            if type(node) is SparseWindowFor
            and node.coord_index == relayout.panel_loop.index_id
        ],
    )
    point_position = _single_position(
        "pack point loop",
        [
            position
            for position, node in enumerate(loops)
            if type(node) is TileInnerFor and node.index == relayout.pack_loop.index_id
        ],
    )
    if (
        len(loops) != 5
        or (pack_position, panel_position, row_position) != (0, 1, 2)
        or (window_position, point_position) != (3, 4)
    ):
        _fail(
            "invalid_schedule_relayout",
            "packed relayout requires exactly the audited chain: pack "
            "origin, panel origin, parallel row, panel window, pack point",
        )
    pack_origin = loops[0]
    panel_origin = loops[1]
    window = loops[3]
    pack_point = loops[4]
    assert type(pack_origin) is TileOuterFor
    assert type(panel_origin) is PanelOuterFor
    assert type(window) is SparseWindowFor
    assert type(pack_point) is TileInnerFor
    if pack_origin.tile != pack_point.tile or panel_origin.tile != window.tile:
        _fail(
            "invalid_schedule_relayout",
            "the relayout's pack and panel loops must be schedule pairs of "
            "the current chain",
        )
    window_parent = window.cursor.parent
    if (
        type(window_parent) is not DensePosition
        or type(window_parent.coord) is not IndexValue
        or window_parent.coord.index != relayout.row_loop.index_id
    ):
        _fail(
            "invalid_schedule_relayout",
            "the relayout's row loop must be the panel window's dense-"
            "parent row loop",
        )
    if pack_origin.width != relayout.strip_width:
        _fail(
            "invalid_schedule_relayout",
            "the relayout strip width must match the pack split's width",
        )

    decls = {decl.symbol: decl for decl in program.tensors}
    operand_decl = decls.get(relayout.operand_id)
    if (
        operand_decl is None
        or relayout.operand_id not in program.inputs
        or len(operand_decl.levels) != 2
        or any(level.kind is not LevelKind.DENSE for level in operand_decl.levels)
    ):
        _fail(
            "invalid_schedule_relayout",
            "the staged operand must be a declared rank-2 all-dense input",
        )

    def _level_storing(dimension: object, description: str) -> int:
        positions = [
            level_position
            for level_position, level in enumerate(operand_decl.levels)
            if operand_decl.dimensions[level.mode] == dimension
        ]
        if len(positions) != 1:
            _fail(
                "invalid_schedule_relayout",
                f"the staged operand must store the {description} dimension "
                "on exactly one level",
            )
        return positions[0]

    if _level_storing(panel_origin.dimension, "panel") != 0 or (
        relayout.operand_panel_level != 0
    ):
        _fail(
            "invalid_schedule_relayout",
            "the staged operand's first storage level must store the "
            "panel dimension",
        )
    if _level_storing(pack_origin.dimension, "pack") != 1 or (
        relayout.operand_pack_level != 1
    ):
        _fail(
            "invalid_schedule_relayout",
            "the staged operand's contiguous last storage level must store "
            "the pack dimension",
        )

    def _load_index_ids(load: Load) -> Optional[Tuple[IndexId, ...]]:
        if type(load.indices) is not tuple or any(
            type(index) is not IndexValue for index in load.indices
        ):
            return None
        return tuple(index.index for index in load.indices if type(index) is IndexValue)

    loads = _collect_operand_loads(program.body, relayout.operand_id)
    matching = [
        load for load in loads if _load_index_ids(load) == relayout.access_indices
    ]
    if not matching:
        _fail(
            "relayout_target_missing",
            "the relayout's access indices select no direct read of the "
            "staged operand",
        )
    if len(matching) > 1 or len(loads) > 1:
        _fail(
            "relayout_ambiguous_access",
            "the staged operand must have exactly one read occurrence; "
            "redirection would be ambiguous",
        )
    target_load = matching[0]

    scope = (
        RelayoutScope.PANEL
        if relayout.scope_loop == relayout.panel_loop
        else RelayoutScope.PACK_AXIS
    )
    builder = LoopIRBuilder.resuming(program)
    relayout_id = builder.new_relayout_id()
    staged = builder.staged_read(relayout_id, target_load.indices)
    new_value = _replace_expr(leaf.value, target_load, staged, builder)
    new_leaf = builder.store_reduce(leaf.tensor, leaf.indices, leaf.op, new_value)
    region_decl = builder.relayout_decl(
        relayout_id,
        relayout.operand_id,
        panel_origin.tile,
        pack_origin.tile,
        scope,
    )
    stage_depth = 2 if scope is RelayoutScope.PANEL else 1
    inner = _wrap_loops(builder, loops[stage_depth:], builder.block((new_leaf,)))
    stage = builder.relayout_stage(region_decl, inner)
    body = _wrap_loops(builder, loops[:stage_depth], builder.block((stage,)))
    rebuilt = builder.program(
        program.dimensions,
        program.tensors,
        program.inputs,
        program.outputs,
        body,
    )
    if _collect_operand_loads(rebuilt.body, relayout.operand_id):
        _fail(
            "relayout_ambiguous_access",
            "redirection left a residual direct read of the staged operand",
        )
    verify_program(rebuilt)
    return rebuilt


def _check_plan_families(plan: LoopPlan) -> None:
    """Fail closed on every plan fact outside the migrated schedule families.

    The sparse-panel family admits exactly the legacy plan shape: at most
    one panel tile, listed after every affine tile, with direct serial
    accumulation, exactly one corresponding ``PanelBound``, and the
    mandatory parallel row loop.  ``parallel_loop`` remains an unmigrated
    family on every plan without a panel tile — the panel form is the one
    shape whose row selection has no degrees of freedom.
    """

    if plan.provenance != "explicit":
        _fail(
            "unsupported_schedule_provenance",
            f"{plan.provenance!r} scheduling stays on the legacy path; only "
            "explicit schedules are migrated",
        )
    if plan.result_tile is not None:
        _fail(
            "unsupported_schedule_result_tile",
            "heap result tiling is not a migrated schedule family",
        )
    panel_tiles = [tile for tile in plan.tiles if tile.kind == "panel"]
    if plan.relayout is not None:
        relayout = plan.relayout
        affine_tiles = [tile for tile in plan.tiles if tile.kind == "affine"]
        if len(plan.tiles) != 2 or len(affine_tiles) != 1 or len(panel_tiles) != 1:
            _fail(
                "invalid_schedule_relayout",
                "packed relayout requires exactly one affine pack tile and "
                "one sparse panel tile",
            )
        pack_tile = affine_tiles[0]
        panel_tile = panel_tiles[0]
        for what, reference in (
            ("pack_loop", relayout.pack_loop),
            ("panel_loop", relayout.panel_loop),
            ("scope_loop", relayout.scope_loop),
            ("row_loop", relayout.row_loop),
        ):
            if reference.part is not LoopPart.LOGICAL:
                _fail(
                    "invalid_schedule_relayout",
                    f"the relayout's {what} must name one logical loop",
                )
        if (
            pack_tile.loop != relayout.pack_loop
            or pack_tile.width != relayout.strip_width
            or pack_tile.placement.kind is not PlacementKind.OUTERMOST
        ):
            _fail(
                "invalid_schedule_relayout",
                "the relayout's pack tile must be the outermost affine "
                "split of its pack loop at the strip width",
            )
        if panel_tile.loop != relayout.panel_loop:
            _fail(
                "invalid_schedule_relayout",
                "the relayout's panel tile must window its panel loop",
            )
        expected_parent = LoopRef(relayout.pack_loop.index_id, LoopPart.OUTER)
        if (
            panel_tile.placement.kind is not PlacementKind.CHILD_OF
            or panel_tile.placement.parent != expected_parent
        ):
            _fail(
                "invalid_schedule_relayout",
                "the relayout's panel must be placed directly below the "
                "pack origin loop",
            )
        if plan.parallel_loop != relayout.row_loop:
            _fail(
                "invalid_schedule_relayout",
                "the relayout's row loop must be the plan's parallel loop",
            )
        if relayout.scope_loop not in (relayout.panel_loop, relayout.pack_loop):
            _fail(
                "invalid_schedule_relayout",
                "the relayout scope must be the panel loop or the pack loop",
            )
    if len(panel_tiles) > 1:
        _fail(
            "invalid_schedule_panel",
            "at most one sparse panel tile is supported per plan",
        )
    if panel_tiles and plan.tiles[-1].kind != "panel":
        _fail(
            "invalid_schedule_panel",
            "a sparse panel tile must follow every affine tile in the plan",
        )
    if plan.panel_bounds and not panel_tiles:
        _fail(
            "invalid_schedule_panel",
            "panel bounds require a panel tile to consume them",
        )
    if panel_tiles:
        panel_tile = panel_tiles[0]
        if panel_tile.accumulation != "direct":
            _fail(
                "invalid_schedule_panel",
                "sparse panel tiles require accumulation='direct'",
            )
        if panel_tile.parallel:
            _fail(
                "invalid_schedule_panel",
                "sparse panel origin loops are serial; select the row loop "
                "through the plan's parallel_loop",
            )
        if len(plan.panel_bounds) != 1 or plan.panel_bounds[0].loop != panel_tile.loop:
            _fail(
                "invalid_schedule_panel",
                "a panel plan carries exactly one PanelBound referencing "
                "the panel tile's loop",
            )
        if plan.parallel_loop is None:
            _fail(
                "panel_parallel_scope",
                "sparse panel tiling requires its CSR dense-parent row loop "
                "as the plan's parallel_loop",
            )
        if plan.parallel_loop.part is not LoopPart.LOGICAL:
            _fail(
                "panel_parallel_scope",
                "the panel's parallel row loop must be one logical loop",
            )
        if panel_tile.placement.kind is PlacementKind.AT_DEPTH:
            _fail(
                "panel_placement_invalid",
                "sparse panel tiles support outermost or child_of placement " "only",
            )
        if panel_tile.placement.kind is PlacementKind.CHILD_OF:
            placement_parent = panel_tile.placement.parent
            assert placement_parent is not None
            outermost_affine = any(
                tile.kind == "affine"
                and tile.loop.index_id == placement_parent.index_id
                and tile.placement.kind is PlacementKind.OUTERMOST
                for tile in plan.tiles
            )
            if placement_parent.part is not LoopPart.OUTER or not outermost_affine:
                _fail(
                    "panel_placement_invalid",
                    "a child_of panel placement must name the origin loop of "
                    "an outermost-placed affine tile of the same plan",
                )
    elif plan.parallel_loop is not None:
        _fail(
            "unsupported_schedule_parallel",
            "explicit parallel-loop selection is not a migrated schedule " "family",
        )
    for tile in plan.tiles:
        if tile.kind == "panel":
            continue
        if tile.accumulation not in ("direct", "stack"):
            _fail(
                "unsupported_schedule_accumulation",
                f"{tile.accumulation!r} accumulation needs the heap result-"
                "tile family; only direct and stack accumulation are "
                "migrated",
            )
        if tile.parallel:
            _fail(
                "unsupported_schedule_parallel",
                "explicit parallel tile selection is not a migrated " "schedule family",
            )


def _check_placement_shape(placement: LoopPlacement) -> None:
    """Reject contradictory placement fields before assert-based dispatch."""

    if placement.kind is PlacementKind.OUTERMOST:
        if placement.parent is not None or placement.depth is not None:
            _fail(
                "tile_invalid_placement",
                "outermost placement cannot also name a parent or depth",
            )
        return
    if placement.kind is PlacementKind.CHILD_OF:
        if placement.parent is None or placement.depth is not None:
            _fail(
                "tile_invalid_placement",
                "child_of placement requires exactly one parent loop",
            )
        return
    if placement.kind is PlacementKind.AT_DEPTH:
        if (
            placement.parent is not None
            or type(placement.depth) is not int
            or placement.depth < 0
        ):
            _fail(
                "tile_invalid_placement",
                "at_depth placement requires exactly one nonnegative depth",
            )
        return
    _fail("tile_invalid_placement", "placement kind is not a PlacementKind member")


def _validate_plan_for_pass(plan: object) -> LoopPlan:
    """Establish the LoopPlan structure this pass boundary relies on."""

    if type(plan) is not LoopPlan:
        raise TypeError("apply_schedule_plan expects a LoopPlan")
    try:
        checked = _validate_loop_plan_structure(plan)
    except (VerificationError, AttributeError, TypeError, ValueError) as error:
        _fail("invalid_schedule_plan", str(error))
    _check_plan_families(checked)

    seen_targets: Set[IndexId] = set()
    for tile in checked.tiles:
        if tile.loop.part is not LoopPart.LOGICAL:
            _fail(
                "tile_target_not_logical",
                "LoopPlan affine tiles must target unsplit logical loops",
            )
        if tile.loop.index_id in seen_targets:
            _fail(
                "tile_target_already_split",
                "a LoopPlan cannot split the same logical loop twice",
            )
        seen_targets.add(tile.loop.index_id)
        if tile.width < 1 or tile.width > MAX_AFFINE_TILE_WIDTH:
            _fail(
                "tile_invalid_width",
                "affine tile widths must be positive ints representable by "
                "the C++ constexpr int target",
            )
        _check_placement_shape(tile.placement)
    return checked


def _loop_provenance(node: _LoopNode) -> ScheduledLoopProvenance:
    index, part = _loop_key(node)
    tile = (
        node.tile  # type: ignore[union-attr]
        if type(node) in (TileOuterFor, TileInnerFor, *_PANEL_TYPES)
        else None
    )
    return ScheduledLoopProvenance(tile, index, part)


def _chain_provenance(
    program: LoopProgram,
) -> Tuple[ScheduledLoopProvenance, ...]:
    """The scheduled chain, outermost first.

    For a workspace-region program the documented order is: the prefix
    loops above the region, then the producer chain, then the consumer
    chain — the region's execution order.  A relayout staging region is
    transparent here: it binds no loop, so provenance lists exactly the
    chain loops and the region's placement is covered by the carrier's
    deterministic replay equality.
    """

    relayout_stages: List[RelayoutStage] = []
    loops, leaf = _decompose_chain(program, relayout_sink=relayout_stages)
    ordered: List[_LoopNode] = list(loops)
    if type(leaf) is WorkspaceRegion:
        producer_loops, _, consumer_loops, _ = _decompose_region(leaf)
        ordered.extend(producer_loops)
        ordered.extend(consumer_loops)
    return tuple(_loop_provenance(node) for node in ordered)


def _apply_schedule_program(program: LoopProgram, plan: LoopPlan) -> LoopProgram:
    """Apply the supported plan decisions without constructing the carrier.

    Plan tiles apply in plan order; the family gate has already required
    any panel tile to be last, so the panel windows the fully affine-tiled
    chain exactly as the legacy lowering sequences it.  Each plan fact is
    consumed exactly once: the order by the reorder pass, each tile by its
    pass, the ``PanelBound`` by materialization into the panel node, the
    panel-mandated ``parallel_loop`` by exact validation against the
    window's dense-parent row loop (the fact has no other legal value),
    and the relayout fact — last, on the fully scheduled chain, exactly
    where the legacy lowering completes it — by :func:`apply_relayout`.
    """

    checked = _validate_plan_for_pass(plan)
    scheduled = reorder_loops(program, checked.loop_order)
    for tile in checked.tiles:
        if tile.kind == "panel":
            scheduled = apply_panel_tile(
                scheduled,
                tile,
                checked.panel_bounds[0],
                checked.parallel_loop,
            )
        elif tile.accumulation == "stack":
            scheduled = apply_stack_tile(scheduled, tile)
        else:
            scheduled = apply_affine_tile(scheduled, tile)
    if checked.relayout is not None:
        scheduled = apply_relayout(scheduled, checked.relayout)
    return scheduled


def _check_exact_carrier_state(value: object, expected: Set[str]) -> bool:
    state = object.__getattribute__(value, "__dict__")
    if type(state) is not dict:
        return False
    keys = tuple(state.keys())
    return all(type(key) is str for key in keys) and tuple(sorted(keys)) == tuple(
        sorted(expected)
    )


def _is_exact_int_identity(value: object, identity_type: type) -> bool:
    if type(value) is not identity_type or not _check_exact_carrier_state(
        value, {"value"}
    ):
        return False
    state = object.__getattribute__(value, "__dict__")
    return type(state["value"]) is int


def _verify_scheduled_loopir(
    artifact: object,
    *,
    expected_program: Optional[LoopProgram],
) -> None:
    """Verify carrier ownership and exact plan consumption."""

    if type(artifact) is not ScheduledLoopIR or not _check_exact_carrier_state(
        artifact, {"base_program", "plan", "program", "loops"}
    ):
        _fail(
            "invalid_scheduled_artifact",
            "scheduled artifact must be an exact, fully stored ScheduledLoopIR",
        )
    base_program = artifact.base_program
    program = artifact.program
    if type(base_program) is not LoopProgram or type(program) is not LoopProgram:
        _fail(
            "invalid_scheduled_artifact",
            "base_program and program must be exact LoopProgram values",
        )
    if type(artifact.loops) is not tuple:
        _fail(
            "invalid_scheduled_artifact",
            "ScheduledLoopIR.loops must be an owned tuple",
        )
    for position, provenance in enumerate(artifact.loops):
        if type(provenance) is not ScheduledLoopProvenance or not (
            _check_exact_carrier_state(provenance, {"tile", "index", "part"})
        ):
            _fail(
                "invalid_scheduled_artifact",
                f"loops[{position}] must be an exact provenance value",
            )
        if provenance.tile is not None and not _is_exact_int_identity(
            provenance.tile, TileId
        ):
            _fail(
                "invalid_scheduled_artifact",
                f"loops[{position}].tile must be None or an int-valued TileId",
            )
        if not _is_exact_int_identity(provenance.index, IndexId) or not any(
            provenance.part is expected
            for expected in (LoopPart.LOGICAL, LoopPart.OUTER, LoopPart.INNER)
        ):
            _fail(
                "invalid_scheduled_artifact",
                f"loops[{position}] has malformed index or loop-part provenance",
            )

    if type(artifact.plan) is not LoopPlan:
        _fail(
            "invalid_scheduled_artifact",
            "ScheduledLoopIR.plan must be an exact LoopPlan",
        )
    checked_plan = _validate_plan_for_pass(artifact.plan)
    verify_program(base_program)
    verify_program(program)
    base_relayouts: List[RelayoutStage] = []
    base_loops, base_leaf = _decompose_chain(base_program, relayout_sink=base_relayouts)
    if (
        base_relayouts
        or type(base_leaf) is WorkspaceRegion
        or any(
            type(loop) in (TileOuterFor, TileInnerFor, *_PANEL_TYPES)
            for loop in base_loops
        )
    ):
        _fail(
            "scheduled_base_not_unscheduled",
            "ScheduledLoopIR.base_program must not already contain split "
            "loops, panels, workspace regions, or staging regions",
        )

    replayed = (
        _apply_schedule_program(base_program, checked_plan)
        if expected_program is None
        else expected_program
    )
    if program != replayed:
        _fail(
            "scheduled_program_mismatch",
            "scheduled program is not the deterministic result of applying "
            "the retained plan to the retained base program",
        )
    if artifact.loops != _chain_provenance(program):
        _fail(
            "scheduled_provenance_mismatch",
            "stored loop provenance does not exactly describe the scheduled "
            "program chain",
        )


def verify_scheduled_loopir(artifact: object) -> None:
    """Fail closed unless all ScheduledLoopIR fields agree exactly."""

    _verify_scheduled_loopir(artifact, expected_program=None)


def apply_schedule_plan(program: LoopProgram, plan: LoopPlan) -> ScheduledLoopIR:
    """Apply one verified explicit LoopPlan to one verified base program.

    The plan's decisions are consumed exactly once: the logical order by the
    reorder pass, then each affine tile in plan order.  Every unsupported
    plan family fails closed with a stable code before any transformation
    runs; nothing is silently ignored.
    """

    scheduled = _apply_schedule_program(program, plan)
    artifact = ScheduledLoopIR(
        base_program=program,
        plan=plan,
        program=scheduled,
        loops=_chain_provenance(scheduled),
    )
    _verify_scheduled_loopir(artifact, expected_program=scheduled)
    return artifact


def erase_schedule(program: LoopProgram) -> LoopProgram:
    """Erase every affine split and panel back to its unscheduled loop.

    ``TileInnerFor`` becomes a plain ``DenseFor`` at the point loop's chain
    position and the paired ``TileOuterFor`` is dropped, which restores
    exactly the pre-tiling chain (splitting never moves the point loop).
    A ``SparseWindowFor`` likewise becomes the plain ``SparseFor`` it
    windowed (same cursor, position, and coordinate bindings) and its
    ``PanelOuterFor`` is dropped: across the origin loop's iterations the
    window visits exactly the stored entries the unwindowed loop visits,
    so under the family's ADD-reassociation contract the erasure preserves
    semantics.

    A workspace region erases to its direct-accumulation equivalent: the
    producer chain returns to the main chain with the original result
    reduction as its leaf (the consumer's target combined with the
    producer's value), and the consumer's copy-out loop disappears.  This
    is semantics-preserving under the family's explicit ADD-reassociation
    contract — accumulating into zero-initialized workspace cells and then
    ADD-copying them out is a reassociation of the direct ADD reduction.
    The erasure is defined for the exact copy-out form
    :func:`apply_stack_tile` produces; other verified region consumers fail
    closed (``unsupported_schedule_shape``).

    A relayout staging region erases to the direct operand read: the
    region wrapper is dropped and every :class:`StagedRead` of it becomes
    the plain :class:`Load` it redirected (same operand, same index
    expressions) — staging copies values exactly, so the erasure is an
    identity on the computed result.

    This is the semantics-preserving erasure the oracle differentials use
    to prove scheduled programs visit every iteration point exactly once.
    A program with no splits is returned unchanged.
    """

    verify_program(program)
    relayout_stages: List[RelayoutStage] = []
    loops, leaf = _decompose_chain(program, relayout_sink=relayout_stages)
    if (
        not relayout_stages
        and type(leaf) is not WorkspaceRegion
        and not any(
            type(node) in (TileOuterFor, TileInnerFor, *_PANEL_TYPES) for node in loops
        )
    ):
        return program
    builder = LoopIRBuilder.resuming(program)
    region_loops: List[_LoopNode] = []
    erased_leaf: _ChainEnd = leaf
    staged_operands = {
        stage.decl.relayout: stage.decl.operand for stage in relayout_stages
    }
    if relayout_stages and type(leaf) in _LEAF_TYPES:

        def _erase_staged(expr: Expr) -> Expr:
            if type(expr) is StagedRead and expr.relayout in staged_operands:
                return builder.load(staged_operands[expr.relayout], expr.indices)
            if type(expr) is BinaryExpr:
                lhs = _erase_staged(expr.lhs)
                rhs = _erase_staged(expr.rhs)
                if lhs is expr.lhs and rhs is expr.rhs:
                    return expr
                return builder.binary(expr.op, lhs, rhs)
            if type(expr) is CursorValue and expr.default is not None:
                default = _erase_staged(expr.default)
                if default is expr.default:
                    return expr
                return builder.cursor_value(expr.cursor, default)
            return expr

        erased_value = _erase_staged(leaf.value)  # type: ignore[union-attr]
        if erased_value is not leaf.value:  # type: ignore[union-attr]
            if type(leaf) is StoreReduce:
                leaf = builder.store_reduce(
                    leaf.tensor, leaf.indices, leaf.op, erased_value
                )
            elif type(leaf) is Store:
                leaf = builder.store(leaf.tensor, leaf.indices, erased_value)
            else:
                _fail(
                    "unsupported_schedule_shape",
                    "staged-read erasure is defined for dense store leaves",
                )
            erased_leaf = leaf
    if type(leaf) is WorkspaceRegion:
        producer_loops, producer_leaf, consumer_loops, consumer_leaf = (
            _decompose_region(leaf)
        )
        workspace_id = leaf.workspace.workspace
        copy_out_value = consumer_leaf.value
        if (
            len(consumer_loops) != 1
            or type(consumer_loops[0]) is not TileInnerFor
            or type(copy_out_value) is not WorkspaceRead
            or copy_out_value.workspace != workspace_id
            or producer_leaf.workspace != workspace_id
        ):
            _fail(
                "unsupported_schedule_shape",
                "workspace-region erasure is defined for the exact "
                "stack-tile copy-out form only",
            )
        region_loops = producer_loops
        erased_leaf = builder.store_reduce(
            consumer_leaf.tensor,
            consumer_leaf.indices,
            consumer_leaf.op,
            producer_leaf.value,
        )
    erased: List[_LoopNode] = []
    for node in (*loops, *region_loops):
        if type(node) in (TileOuterFor, PanelOuterFor):
            continue
        if type(node) is TileInnerFor:
            erased.append(builder.dense_for(node.index, node.dimension, node.body))
        elif type(node) is SparseWindowFor:
            erased.append(
                builder.sparse_for(
                    node.cursor, node.position, node.coord_index, node.body
                )
            )
        else:
            erased.append(node)
    return _rebuild_program(program, builder, erased, leaf=erased_leaf)
