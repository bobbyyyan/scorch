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
  one occurrence and the pass proves it before redirecting;
- :func:`apply_result_tile` compacts the heap-accumulation result behind a
  typed :class:`~scorch.compiler.loopir.nodes.ResultTileRegion` wrapping
  the pack origin's entire body (fresh zeroed tile per strip, exactly-once
  copy-out at exit) and structurally replaces the result's verifier-proven
  **unique** ``StoreReduce`` occurrence with a
  :class:`~scorch.compiler.loopir.nodes.TiledReduce` carrying the fresh
  region identity — the same no-occurrence-identity decision, applied
  after :func:`apply_relayout` so the region wraps the fully staged chain.

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

from dataclasses import dataclass, fields, replace
from typing import (
    Any,
    Dict,
    List,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

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
    ResultTile,
    WorkspaceInsertion,
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
    LoopIRNode,
    LoopProgram,
    MergedSparseFor,
    PanelOuterFor,
    ParallelDiscipline,
    ParallelIntent,
    ParallelPart,
    PositionId,
    PositionValue,
    ReduceOp,
    RelayoutScope,
    RelayoutStage,
    ResultTileRegion,
    SparseCursorDecl,
    SparseFor,
    SparseWindowFor,
    StagedRead,
    Stmt,
    Store,
    StoreReduce,
    TiledReduce,
    TileId,
    TileInnerFor,
    TileOuterFor,
    WorkspaceRead,
    WorkspaceReduce,
    WorkspaceRegion,
)
from .verifier import _canonical_parallel_work_source, verify_program

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
_ChainEnd = Union[Store, StoreReduce, AppendEntry, TiledReduce, WorkspaceRegion]

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
    result_tile_sink: Optional[List[ResultTileRegion]] = None,
) -> Tuple[List[_LoopNode], Stmt]:
    """Walk one single-statement chain of loops down to its terminator.

    A :class:`RelayoutStage` or :class:`ResultTileRegion` wrapper is a
    chain element only for callers that pass the matching sink (provenance
    and erasure, which must read already-scheduled programs, and the heap
    pass, which composes after relayout); every other scheduling pass keeps
    the defaults and therefore refuses already-staged chains with a stable
    code.
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
        if type(only) is ResultTileRegion and result_tile_sink is not None:
            result_tile_sink.append(only)
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
    result_tile_sink: Optional[List[ResultTileRegion]] = None,
) -> Tuple[List[_LoopNode], _ChainEnd]:
    """Split one migrated-family program into its loop chain and terminator.

    The terminator is an ordinary store/append leaf, one
    :class:`WorkspaceRegion` for programs a stack tile already transformed,
    or one :class:`TiledReduce` for programs the heap pass already
    transformed (the verifier guarantees a reduce implies its region, so a
    sink-less caller fails on the region wrapper before reaching it).
    """

    loops, end = _decompose_body(
        program.body,
        leaf_types=(*_LEAF_TYPES, WorkspaceRegion, TiledReduce),
        allow_empty=False,
        relayout_sink=relayout_sink,
        result_tile_sink=result_tile_sink,
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
    *,
    keep_parallel: bool = True,
) -> LoopProgram:
    """Reassemble the chain with fresh loop/block nodes and shared subtrees.

    ``keep_parallel`` carries the program's abstract parallel selection
    through structural rebuilds; erasure passes ``False`` because the
    selection is schedule state and the erased program must be the
    unscheduled base.
    """

    body = _wrap_loops(builder, loops, builder.block((leaf,)))
    rebuilt = builder.program(
        program.dimensions,
        program.tensors,
        program.inputs,
        program.outputs,
        body,
        program.parallel if keep_parallel else None,
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
    if tile.accumulation not in ("direct", "heap"):
        _fail(
            "unsupported_schedule_accumulation",
            f"{tile.accumulation!r} accumulation needs the workspace "
            "families; only direct and heap accumulation are migrated",
        )
    # The split itself is accumulation-neutral: a "heap" tile's origin/point
    # geometry is exactly the direct split, and the heap accumulation fact
    # is consumed afterwards by apply_result_tile (the plan gate requires
    # the pairing, so no heap fact is silently dropped on the plan route).
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


def _validate_reduce_out_tile(tile: LoopTile) -> None:
    """Validate one recorded reduce-out candidate tile fact."""

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
            "the reduce-out family records direct serial candidate "
            f"tiles only, got {tile.accumulation!r}",
        )
    if tile.parallel:
        _fail(
            "unsupported_schedule_parallel",
            "explicit parallel tile selection is not a migrated " "schedule family",
        )
    if (
        type(tile.width) is not int
        or tile.width < 1
        or tile.width > MAX_AFFINE_TILE_WIDTH
    ):
        _fail(
            "tile_invalid_width",
            "affine tile widths must be positive ints representable by "
            "the C++ constexpr int target",
        )
    if type(tile.placement) is not LoopPlacement:
        _fail(
            "tile_invalid_placement",
            "affine tile placement must be an exact LoopPlacement",
        )
    _check_placement_shape(tile.placement)


def apply_reduce_out_tiles(
    program: LoopProgram,
    tiles: Tuple[LoopTile, ...],
    workspace: WorkspaceInsertion,
) -> LoopProgram:
    """Strip-mine the dense reduce-out automatic family in one fused rebuild.

    Mirrors the legacy automatic composition exactly: ``insert_workspace``
    materializes a one-cell-per-point dense workspace over the single
    trailing free axis at the last reduction loop, then every recorded
    candidate tile is applied in plan order with ``add_tile`` semantics —
    each point loop splits in place and each origin loop inserts at the
    arm placement, so later origins stack LIFO above earlier ones.  The
    region's producer is the strip-mined reduction chain accumulating into
    the workspace through the axis point loop; the consumer ADD-copies the
    tile out to the result.  Exactly one tile must split the workspace
    axis, exactly one must split the last reduction loop, and every other
    tile splits a distinct dense prefix loop.
    """

    verify_program(program)
    if type(tiles) is not tuple or not all(type(tile) is LoopTile for tile in tiles):
        raise TypeError("apply_reduce_out_tiles expects a tuple of LoopTile")
    if type(workspace) is not WorkspaceInsertion:
        raise TypeError("apply_reduce_out_tiles expects a WorkspaceInsertion")
    try:
        _validate_loop_plan_structure(
            LoopPlan(loop_order=(), tiles=tiles, workspace=workspace)
        )
    except (VerificationError, AttributeError, TypeError, ValueError) as error:
        _fail("invalid_schedule_tile", str(error))
    if len(tiles) < 2:
        _fail(
            "reduce_out_shape_invalid",
            "the reduce-out family strip-mines at least the reduction loop "
            "and the workspace axis",
        )
    if not workspace.dense or len(workspace.axis_loops) != 1:
        _fail(
            "reduce_out_shape_invalid",
            "the reduce-out family materializes one dense workspace over a "
            "single trailing axis",
        )
    for tile in tiles:
        _validate_reduce_out_tile(tile)
    targets = [tile.loop.index_id for tile in tiles]
    if len(set(targets)) != len(targets):
        _fail(
            "tile_target_already_split",
            "a reduce-out plan cannot split the same logical loop twice",
        )

    loops, leaf = _decompose_chain(program)
    if any(type(node) in _PANEL_TYPES for node in loops):
        _fail(
            "unsupported_schedule_shape",
            "reduce-out tiling over a panel-scheduled chain is not " "migrated",
        )
    if type(leaf) is WorkspaceRegion:
        _fail(
            "reduce_out_shape_invalid",
            "the chain already carries a workspace region; at most one "
            "workspace accumulation exists per program",
        )
    if type(leaf) is not StoreReduce:
        _fail(
            "reduce_out_shape_invalid",
            "reduce-out accumulation requires a dense-output ADD reduction " "leaf",
        )
    for node in loops:
        if _loop_key(node)[1] is not LoopPart.LOGICAL:
            _fail(
                "tile_target_already_split",
                "reduce-out tiling operates on the unsplit logical chain",
            )

    leaf_index_ids: Set[IndexId] = set()
    for index in leaf.indices:
        if type(index) is not IndexValue:
            _fail(
                "reduce_out_shape_invalid",
                "reduce-out accumulation requires directly bound result " "coordinates",
            )
        leaf_index_ids.add(index.index)
    reduction_positions = [
        position
        for position, node in enumerate(loops)
        if _loop_key(node)[0] not in leaf_index_ids
    ]
    if not reduction_positions:
        _fail(
            "reduce_out_shape_invalid",
            "reduce-out accumulation requires a reduction loop above the "
            "trailing free axis",
        )
    last_reduction = max(reduction_positions)
    axis_position = last_reduction + 1
    if axis_position != len(loops) - 1:
        _fail(
            "reduce_out_shape_invalid",
            "reduce-out accumulation is only supported for a single "
            "trailing dense free dimension after the last reduction",
        )
    if last_reduction == 0:
        _fail(
            "reduce_out_root_scope",
            "reduce-out tiling cannot wrap a workspace inserted at the " "root scope",
        )
    reduction_node = loops[last_reduction]
    axis_node = loops[axis_position]
    if type(reduction_node) is not DenseFor:
        _fail(
            "tile_target_not_dense",
            "the reduce-out family strip-mines dense loops only; sparse "
            "reductions belong to the stack family",
        )
    if type(axis_node) is not DenseFor:
        _fail(
            "tile_target_not_dense",
            "the reduce-out family strip-mines dense loops only; sparse "
            "reductions belong to the stack family",
        )
    reduction_index = _loop_key(reduction_node)[0]
    axis_index = _loop_key(axis_node)[0]
    if workspace.reduction_loop != LoopRef(reduction_index):
        _fail(
            "reduce_out_shape_invalid",
            "the workspace fact must name the chain's last reduction loop",
        )
    if workspace.axis_loops != (LoopRef(axis_index),):
        _fail(
            "reduce_out_shape_invalid",
            "the workspace fact must name the chain's trailing free axis",
        )
    reduction_tiles = [tile for tile in tiles if tile.loop.index_id == reduction_index]
    axis_tiles = [tile for tile in tiles if tile.loop.index_id == axis_index]
    if len(reduction_tiles) != 1 or len(axis_tiles) != 1:
        _fail(
            "reduce_out_shape_invalid",
            "the reduce-out family strip-mines the last reduction loop and "
            "the workspace axis exactly once each",
        )
    reduction_tile = reduction_tiles[0]
    axis_tile = axis_tiles[0]
    prefix_targets: Dict[IndexId, DenseFor] = {}
    prefix_positions: Dict[IndexId, int] = {}
    for position, node in enumerate(loops[:last_reduction]):
        if type(node) is DenseFor:
            prefix_targets[_loop_key(node)[0]] = node
            prefix_positions[_loop_key(node)[0]] = position
    for tile in tiles:
        if tile is reduction_tile or tile is axis_tile:
            continue
        if tile.loop.index_id not in prefix_targets:
            if any(
                _loop_key(node)[0] == tile.loop.index_id
                for node in loops[:last_reduction]
            ):
                _fail(
                    "tile_target_not_dense",
                    "affine tiling cannot split a sparse coordinate loop",
                )
            _fail(
                "tile_target_missing",
                "every reduce-out candidate tile must split a prefix loop, "
                "the last reduction loop, or the workspace axis",
            )

    result_symbol = program.outputs[0]
    result_decl = next(decl for decl in program.tensors if decl.symbol == result_symbol)

    builder = LoopIRBuilder.resuming(program)
    tile_ids = {id(tile): builder.new_tile_id() for tile in tiles}
    workspace_id = builder.new_workspace_id()
    workspace_decl = builder.workspace_decl(
        workspace_id, "wksp", result_decl.dtype, tile_ids[id(axis_tile)]
    )

    reduce_leaf = builder.workspace_reduce(
        workspace_id,
        builder.index_value(axis_index),
        ReduceOp.ADD,
        leaf.value,
    )
    producer_axis_point = builder.tile_inner_for(
        tile_ids[id(axis_tile)],
        axis_index,
        axis_node.dimension,
        axis_tile.width,
        axis_tile.unroll,
        builder.block((reduce_leaf,)),
    )
    producer_reduction_point = builder.tile_inner_for(
        tile_ids[id(reduction_tile)],
        reduction_index,
        reduction_node.dimension,
        reduction_tile.width,
        reduction_tile.unroll,
        builder.block((producer_axis_point,)),
    )
    copy_out = builder.store_reduce(
        leaf.tensor,
        leaf.indices,
        leaf.op,
        builder.workspace_read(workspace_id, builder.index_value(axis_index)),
    )
    consumer_point = builder.tile_inner_for(
        tile_ids[id(axis_tile)],
        axis_index,
        axis_node.dimension,
        axis_tile.width,
        axis_tile.unroll,
        builder.block((copy_out,)),
    )
    region = builder.workspace_region(
        workspace_decl,
        builder.block((producer_reduction_point,)),
        builder.block((consumer_point,)),
    )

    new_chain: List[_LoopNode] = list(loops[:last_reduction])
    for tile in tiles:
        if tile is reduction_tile or tile is axis_tile:
            continue
        prefix_node = prefix_targets[tile.loop.index_id]
        new_chain[prefix_positions[tile.loop.index_id]] = builder.tile_inner_for(
            tile_ids[id(tile)],
            prefix_node.index,
            prefix_node.dimension,
            tile.width,
            tile.unroll,
            prefix_node.body,
        )
    # ``add_tile`` semantics: origins insert in plan order, each at its
    # placement depth against the current chain, so later origins stack
    # LIFO above earlier ones at the same anchor.
    inner_of = {
        id(tile): tile_ids[id(tile)]
        for tile in tiles
        if tile is not reduction_tile and tile is not axis_tile
    }
    for tile in tiles:
        if tile is reduction_tile:
            origin_dimension = reduction_node.dimension
            origin_index = reduction_index
            domination = len(new_chain)
        elif tile is axis_tile:
            origin_dimension = axis_node.dimension
            origin_index = axis_index
            domination = len(new_chain)
        else:
            origin_dimension = prefix_targets[tile.loop.index_id].dimension
            origin_index = tile.loop.index_id
            domination = next(
                position
                for position, node in enumerate(new_chain)
                if type(node) is TileInnerFor and node.tile == inner_of[id(tile)]
            )
        depth = _placement_depth(new_chain, tile.placement, domination)
        outer = builder.tile_outer_for(
            tile_ids[id(tile)],
            origin_index,
            origin_dimension,
            tile.width,
            # The outer body is reassembled by _rebuild_program; this
            # placeholder block is replaced there and never enters the
            # rebuilt tree.
            axis_node.body,
        )
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


def _walk_schema_nodes(root: LoopIRNode) -> List[LoopIRNode]:
    """Return the declared LoopIR nodes in ``root`` exactly once.

    Verified nodes may carry verifier-invisible extra ``__dict__`` state.
    Scheduling decisions, canonical identity, and continuation allocation
    deliberately ignore that state, so pass-side discovery must do the same.
    Walking dataclass fields also keeps a forged extra alias/cycle from
    hanging discovery or counting one semantic occurrence twice.
    """

    found: List[LoopIRNode] = []
    pending: List[LoopIRNode] = [root]
    seen: Set[int] = set()
    while pending:
        node = pending.pop()
        marker = id(node)
        if marker in seen:
            continue
        seen.add(marker)
        found.append(node)
        for field in fields(cast(Any, node)):
            child = object.__getattribute__(node, field.name)
            if isinstance(child, LoopIRNode):
                pending.append(child)
            elif type(child) is tuple:
                pending.extend(item for item in child if isinstance(item, LoopIRNode))
    return found


def _collect_operand_loads(root: Stmt, operand: object) -> List[Load]:
    """Every schema-owned ``Load`` of one tensor in a verified subtree."""

    found = []
    for node in _walk_schema_nodes(root):
        if type(node) is Load and node.tensor == operand:
            found.append(node)
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
        program.parallel,
    )
    if _collect_operand_loads(rebuilt.body, relayout.operand_id):
        _fail(
            "relayout_ambiguous_access",
            "redirection left a residual direct read of the staged operand",
        )
    verify_program(rebuilt)
    return rebuilt


def _collect_result_writes(root: Stmt, result: object) -> List[Stmt]:
    """Every schema-owned direct write of ``result`` in one verified tree."""

    writes: List[Stmt] = []
    for value in _walk_schema_nodes(root):
        if type(value) in (Store, StoreReduce) and value.tensor == result:  # type: ignore[attr-defined]
            writes.append(value)  # type: ignore[arg-type]
    return writes


def _result_tile_prefix_rank(result_tile: ResultTile) -> int:
    """Admit one rank>=2 heap result-tile fact and return its prefix rank.

    ``result_prefix`` is expressed in physical storage order while
    ``access_indices`` is expressed in logical mode order.  Their exact
    correspondence therefore needs the result declaration's level-to-mode
    mapping and is checked in :func:`apply_result_tile` (and by the
    normalized-CIN plan verifier).  This helper owns only the declaration-
    independent rank relationship.

    Everything downstream — chain length, prefix loop positions, the compact
    linearization, and the copy-out extent — is derived from this one number,
    so the family admits rank-2 and the audited multi-prefix ranks by the same
    rule rather than by enumerating shapes.
    """

    prefix_rank = (
        len(result_tile.result_prefix)
        if type(result_tile.result_prefix) is tuple
        else -1
    )
    if (
        prefix_rank < 1
        or type(result_tile.access_indices) is not tuple
        or len(result_tile.access_indices) != prefix_rank + 1
        or result_tile.result_level != prefix_rank
    ):
        _fail(
            "invalid_schedule_result_tile",
            "the migrated heap family compacts an all-dense result of rank "
            "at least two whose trailing storage level is the tiled free "
            "axis and whose remaining physical levels are the tile's dense "
            "prefix",
        )
    return prefix_rank


def apply_result_tile(program: LoopProgram, result_tile: ResultTile) -> LoopProgram:
    """Accumulate one dense result's trailing strip behind a typed region.

    Mirrors the audited legacy heap result-tile family exactly, on
    identities only, for the migrated rank>=2 trailing-axis shape.  The
    scheduled chain must be one of the admitted forms — the outermost
    affine pack origin over the result's dense prefix loops, one reduction
    loop, and the pack point loop (heap alone), or that rank-derived prefix
    chain composed with one sparse panel.  The audited relayout composition
    remains the packed tile-ijk five-loop/single-prefix subfamily and is
    already staged when this pass runs.  Every form has a direct
    dense-result reduction leaf.  A multi-prefix result contributes one
    dense loop per prefix axis at a derived chain position; no spelling of
    any particular kernel is special-cased.

    Every result-tile fact is consumed exactly once: ``tile_loop``
    selects the pack schedule pair, ``access_indices`` select the
    redirected write, and ``result_prefix``/``result_level`` are
    validation-is-consumption against the result declaration (the
    ``PanelBound`` precedent).

    The redirected write is the result's **unique** ``StoreReduce``
    occurrence — the pass proves uniqueness (``result_tile_target_missing``
    / ``result_tile_ambiguous_write``) and replaces it structurally with a
    :class:`TiledReduce` carrying the fresh region identity, then wraps the
    pack origin's entire body in the :class:`ResultTileRegion`, so no
    occurrence identity, rendered name, or dynamic tag is ever needed;
    residual direct writes are re-checked after the rebuild.  Unlike
    operand reads, result writes are statements the verifier's coordinate
    model fully pins within this family, so the exact fact admission above
    subsumes the scan; it is retained as checked-property defense in depth
    (the assumption stays a checked invariant if the coordinate model ever
    widens).
    """

    if type(result_tile) is not ResultTile:
        raise TypeError("apply_result_tile expects a ResultTile")
    verify_program(program)
    if program.parallel is not None:
        _fail(
            "invalid_schedule_result_tile",
            "parallel selection must run after result-tile construction; "
            "introducing compact storage cannot silently discard or "
            "reinterpret an existing result-partition discipline",
        )
    try:
        _validate_loop_plan_structure(LoopPlan(loop_order=(), result_tile=result_tile))
    except (VerificationError, AttributeError, TypeError, ValueError) as error:
        _fail("invalid_schedule_result_tile", str(error))
    if (
        type(result_tile.tile_loop) is not LoopRef
        or result_tile.tile_loop.part is not LoopPart.LOGICAL
    ):
        _fail(
            "invalid_schedule_result_tile",
            "the result tile's tile_loop must name one logical loop",
        )
    prefix_rank = _result_tile_prefix_rank(result_tile)

    relayout_stages: List[RelayoutStage] = []
    loops, leaf = _decompose_chain(program, relayout_sink=relayout_stages)
    if type(leaf) is not StoreReduce:
        _fail(
            "invalid_schedule_result_tile",
            "heap accumulation requires a direct dense-result reduction leaf",
        )
    kinds = tuple(type(node) for node in loops)
    # One pack origin, the result's dense prefix loops in physical storage
    # order, one
    # reduction loop, and the pack point loop.  ``prefix_rank`` is fixed by
    # the already-admitted result-tile fact, so the chain length and every
    # loop position below are derived, never searched.
    heap_alone = (
        len(kinds) == prefix_rank + 3
        and kinds[0] is TileOuterFor
        and all(kind is DenseFor for kind in kinds[1 : prefix_rank + 1])
        and kinds[prefix_rank + 1] in (DenseFor, SparseFor)
        and kinds[-1] is TileInnerFor
    )
    heap_panel = (
        prefix_rank >= 1
        and len(kinds) == prefix_rank + 4
        and kinds[0] is TileOuterFor
        and kinds[1] is PanelOuterFor
        and all(kind is DenseFor for kind in kinds[2 : prefix_rank + 2])
        and kinds[prefix_rank + 2] is SparseWindowFor
        and kinds[-1] is TileInnerFor
    )
    if not heap_alone and not heap_panel:
        _fail(
            "invalid_schedule_result_tile",
            "heap accumulation requires the audited chain: the pack origin "
            "over the dense result-prefix loops, one reduction loop, and the "
            "pack point loop, optionally windowed by one sparse panel on the "
            "reduction axis",
        )
    if len(relayout_stages) > 1 or (relayout_stages and not heap_panel):
        _fail(
            "invalid_schedule_result_tile",
            "heap accumulation composes with at most one relayout stage on "
            "the packed tile-ijk chain",
        )
    pack_origin = loops[0]
    pack_point = loops[-1]
    prefix_start = 2 if heap_panel else 1
    prefix_loops = loops[prefix_start : prefix_start + prefix_rank]
    assert type(pack_origin) is TileOuterFor
    assert type(pack_point) is TileInnerFor
    assert all(type(loop) is DenseFor for loop in prefix_loops)
    if (
        pack_origin.index != result_tile.tile_loop.index_id
        or pack_point.index != result_tile.tile_loop.index_id
        or pack_origin.tile != pack_point.tile
    ):
        _fail(
            "invalid_schedule_result_tile",
            "the result tile's pack loop must be the chain's outermost "
            "origin/point schedule pair",
        )
    if heap_panel:
        panel_origin = loops[1]
        window = loops[prefix_rank + 2]
        assert type(panel_origin) is PanelOuterFor
        assert type(window) is SparseWindowFor
        if panel_origin.tile != window.tile:
            _fail(
                "invalid_schedule_result_tile",
                "the composed panel origin and window must be one schedule " "pair",
            )
    decls = {decl.symbol: decl for decl in program.tensors}
    result_decl = decls.get(result_tile.result_id)
    if (
        result_decl is None
        or result_tile.result_id not in program.outputs
        or len(result_decl.levels) != prefix_rank + 1
        or any(level.kind is not LevelKind.DENSE for level in result_decl.levels)
    ):
        _fail(
            "invalid_schedule_result_tile",
            "the compacted result must be a declared all-dense output whose "
            "rank is one more than the tile's dense prefix",
        )
    assert result_decl is not None
    physical_prefix = tuple(
        result_tile.access_indices[level.mode]
        for level in result_decl.levels[:prefix_rank]
    )
    packed_index = result_tile.access_indices[result_decl.levels[prefix_rank].mode]
    if (
        physical_prefix != result_tile.result_prefix
        or packed_index != result_tile.tile_loop.index_id
    ):
        _fail(
            "invalid_schedule_result_tile",
            "the result tile's physical prefix and trailing packed loop must "
            "match its logical access through the result's level-to-mode map",
        )
    if (
        tuple(cast(DenseFor, loop).index for loop in prefix_loops)
        != result_tile.result_prefix
    ):
        _fail(
            "invalid_schedule_result_tile",
            "the result tile's dense prefix loops must be the chain's "
            "prefix loops in physical storage order",
        )

    def _write_index_ids(write: Stmt) -> Optional[Tuple[IndexId, ...]]:
        indices = write.indices  # type: ignore[attr-defined]
        if type(indices) is not tuple or any(
            type(index) is not IndexValue for index in indices
        ):
            return None
        return tuple(index.index for index in indices if type(index) is IndexValue)

    writes = _collect_result_writes(program.body, result_tile.result_id)
    matching = [
        write
        for write in writes
        if _write_index_ids(write) == result_tile.access_indices
    ]
    if not matching:
        _fail(
            "result_tile_target_missing",
            "the result tile's access indices select no direct reduction "
            "of the compacted result",
        )
    if len(matching) > 1 or len(writes) > 1 or matching[0] is not leaf:
        _fail(
            "result_tile_ambiguous_write",
            "the compacted result must have exactly one write occurrence; "
            "redirection would be ambiguous",
        )

    builder = LoopIRBuilder.resuming(program)
    result_tile_id = builder.new_result_tile_id()
    new_leaf = builder.tiled_reduce(result_tile_id, leaf.indices, leaf.op, leaf.value)

    def _rebuild_stmt(stmt: Stmt) -> Stmt:
        if stmt is leaf:
            return new_leaf
        if type(stmt) is RelayoutStage:
            return builder.relayout_stage(stmt.decl, _rebuild_block(stmt.body))
        if type(stmt) in _LOOP_TYPES:
            wrapped = _wrap_loops(
                builder,
                (stmt,),  # type: ignore[arg-type]
                _rebuild_block(stmt.body),  # type: ignore[attr-defined]
            )
            return wrapped.statements[0]
        _fail(
            "invalid_schedule_result_tile",
            f"unsupported chain statement {type(stmt).__name__}",
        )

    def _rebuild_block(block: Block) -> Block:
        return builder.block(
            tuple(_rebuild_stmt(statement) for statement in block.statements)
        )

    region_decl = builder.result_tile_decl(
        result_tile_id, result_tile.result_id, pack_origin.tile
    )
    region = builder.result_tile_region(region_decl, _rebuild_block(pack_origin.body))
    new_origin = builder.tile_outer_for(
        pack_origin.tile,
        pack_origin.index,
        pack_origin.dimension,
        pack_origin.width,
        builder.block((region,)),
    )
    rebuilt = builder.program(
        program.dimensions,
        program.tensors,
        program.inputs,
        program.outputs,
        builder.block((new_origin,)),
    )
    if _collect_result_writes(rebuilt.body, result_tile.result_id):
        _fail(
            "result_tile_ambiguous_write",
            "redirection left a residual direct write of the compacted " "result",
        )
    verify_program(rebuilt)
    return rebuilt


def _ordered_schema_nodes(root: LoopIRNode) -> List[LoopIRNode]:
    """Every schema node under ``root`` in document order, root included.

    Declared dataclass fields are visited in declaration order and tuple
    children in sequence, so "first" below means first in the program's
    lexical statement order — the same order the target's own structural
    policy derivation encounters emitted statements.
    """

    collected: List[LoopIRNode] = []
    seen: Set[int] = set()
    stack: List[LoopIRNode] = [root]
    while stack:
        node = stack.pop()
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
        stack.extend(reversed(children))
    return collected


def _selection_target_matches(
    node: LoopIRNode, index: IndexId, part: ParallelPart
) -> bool:
    if part is ParallelPart.LOGICAL:
        if type(node) is DenseFor:
            return node.index == index
        if type(node) is SparseFor or type(node) is MergedSparseFor:
            return node.coord_index == index
        return False
    if type(node) is TileOuterFor or type(node) is PanelOuterFor:
        return node.index == index
    return False


def select_parallel_loop(program: LoopProgram, plan: LoopPlan) -> LoopProgram:
    """Consume the plan's explicit ``parallel_loop`` fact exactly once.

    Runs last in :func:`apply_schedule_plan`, on the fully scheduled
    program, and materializes the fact as the program's abstract
    :class:`ParallelSelection` — the typed twin of the legacy explicit
    route (``CINLowerer._apply_explicit_parallel_schedule``), which marks
    the named loop on the assembled function after every structural
    schedule transformation.

    Legacy anchor semantics are reproduced on identities: a LOGICAL
    anchor naming an affine-split variable selects the split's origin
    loop (legacy's ``{var}_out`` redirect), a point loop is never
    selectable (the ragged-tail clamp), and explicit selection requires
    an all-dense result.  The race, work, and reduction legality proofs
    are the verifier's ``parallel_race`` / ``parallel_work_mismatch``
    obligations, re-proved on the stamped program before it is returned —
    the pass derives facts, the verifier re-derives them.
    """

    verify_program(program)
    checked = _validate_plan_for_pass(plan)
    fact = checked.parallel_loop
    if fact is None:
        if program.parallel is not None:
            _fail(
                "invalid_schedule_parallel",
                "a program already carrying a parallel selection cannot be "
                "paired with a plan that has no parallel-loop fact",
            )
        return program
    if program.parallel is not None:
        _fail(
            "invalid_schedule_parallel",
            "the scheduled program already carries a parallel selection; "
            "the plan fact would not be consumed exactly once",
        )
    if fact.part is LoopPart.INNER:
        _fail(
            "invalid_schedule_parallel",
            "a split's point loop contains the ragged-tail clamp and "
            "cannot be selected for parallel execution",
        )
    index = fact.index_id
    part = fact.part
    if part is LoopPart.LOGICAL and any(
        tile.kind == "affine" and tile.loop.index_id == index for tile in checked.tiles
    ):
        # Legacy redirects a logical anchor naming an affine-split
        # variable to the split's origin loop.
        part = LoopPart.OUTER
    parallel_part = (
        ParallelPart.LOGICAL if part is LoopPart.LOGICAL else ParallelPart.OUTER
    )
    decls = {decl.symbol: decl for decl in program.tensors}
    if any(
        level.kind is not LevelKind.DENSE
        for output in program.outputs
        for level in decls[output].levels
    ):
        _fail(
            "invalid_schedule_parallel",
            "explicit parallel-loop selection requires a dense result " "tensor",
        )
    ordered = _ordered_schema_nodes(program.body)
    matches = [
        node
        for node in ordered
        if _selection_target_matches(node, index, parallel_part)
    ]
    if len(matches) != 1:
        _fail(
            "invalid_schedule_parallel",
            "the plan's parallel loop must resolve to exactly one " "scheduled loop",
        )
    target = matches[0]
    if type(target) not in (DenseFor, TileOuterFor):
        # Legacy explicit selection finds tagged for-loops only; compressed,
        # merged, and panel-origin anchors have no measured legacy comparand
        # and stay outside the migrated family.
        _fail(
            "invalid_schedule_parallel",
            "explicit parallel selection is migrated for dense logical "
            "loops and affine origin loops only",
        )
    typed_target = cast(Stmt, target)
    rows = cast(DenseFor, typed_target).dimension
    nnz_source = _canonical_parallel_work_source(program, typed_target, index)
    discipline = (
        ParallelDiscipline.COMPACT_PARTITION
        if checked.result_tile is not None
        else ParallelDiscipline.RESULT_PARTITION
    )
    builder = LoopIRBuilder.resuming(program)
    work = builder.parallel_work(
        rows,
        (
            None
            if nnz_source is None
            else builder.sparse_work_source(nnz_source[0], nnz_source[1])
        ),
    )
    selection = builder.parallel_selection(
        index,
        parallel_part,
        discipline,
        work,
        ParallelIntent.EXPLICIT,
    )
    stamped = builder.program(
        program.dimensions,
        program.tensors,
        program.inputs,
        program.outputs,
        program.body,
        selection,
    )
    verify_program(stamped)
    return stamped


def _check_heap_plan_family(plan: LoopPlan) -> None:
    """Exact pre-replay admission of the heap result-tile plan family.

    The heap family admits exactly the audited legacy shape: one serial
    outermost heap-accumulation affine tile targeting the plan's innermost
    logical loop, paired one-to-one with the plan's ``result_tile`` fact
    compacting a rank>=2 dense result on its trailing storage level, at most
    one composed sparse panel tile, and one mandatory parallel dense
    result-prefix loop.  Any prefix loop partitions compact cells; the
    panel composition further fixes the selection to its dense-parent row.
    """

    heap_tiles = [
        tile
        for tile in plan.tiles
        if tile.kind == "affine" and tile.accumulation == "heap"
    ]
    if plan.result_tile is None:
        if heap_tiles:
            _fail(
                "invalid_schedule_result_tile",
                "a heap-accumulation tile requires the plan's result-tile " "fact",
            )
        return
    result_tile = plan.result_tile
    if len(heap_tiles) != 1:
        _fail(
            "invalid_schedule_result_tile",
            "a heap result tile requires exactly one heap-accumulation " "affine tile",
        )
    heap_tile = heap_tiles[0]
    if (
        result_tile.tile_loop.part is not LoopPart.LOGICAL
        or heap_tile.loop != result_tile.tile_loop
    ):
        _fail(
            "invalid_schedule_result_tile",
            "the result tile must compact the heap tile's logical loop",
        )
    if heap_tile.placement.kind is not PlacementKind.OUTERMOST:
        _fail(
            "invalid_schedule_result_tile",
            "the heap tile must be outermost so the compact result "
            "spans every enclosed reduction",
        )
    if heap_tile.parallel:
        _fail(
            "invalid_schedule_result_tile",
            "a heap-backed result tile uses shared reusable storage; "
            "its origin loop is serial",
        )
    other_tiles = [tile for tile in plan.tiles if tile is not heap_tile]
    if len(other_tiles) > 1 or any(tile.kind != "panel" for tile in other_tiles):
        _fail(
            "invalid_schedule_result_tile",
            "a heap plan composes with at most one sparse panel tile",
        )
    prefix_rank = _result_tile_prefix_rank(result_tile)
    if not plan.loop_order or plan.loop_order[-1] != result_tile.tile_loop.index_id:
        _fail(
            "invalid_schedule_result_tile",
            "the heap tile must target the plan's innermost logical loop",
        )
    if other_tiles:
        panel_tile = other_tiles[0]
        expected_order = (
            *result_tile.result_prefix,
            panel_tile.loop.index_id,
            result_tile.tile_loop.index_id,
        )
        expected_parent = LoopRef(
            result_tile.tile_loop.index_id,
            LoopPart.OUTER,
        )
        if (
            plan.loop_order != expected_order
            or panel_tile.placement.kind is not PlacementKind.CHILD_OF
            or panel_tile.placement.parent != expected_parent
        ):
            _fail(
                "invalid_schedule_result_tile",
                "a heap-panel plan requires the exact prefix, panel, pack "
                "logical order and places the panel directly below the "
                "heap pack origin",
            )
    elif (
        len(plan.loop_order) != prefix_rank + 2
        or plan.loop_order[:prefix_rank] != result_tile.result_prefix
    ):
        _fail(
            "invalid_schedule_result_tile",
            "a heap-only plan requires the exact dense-prefix, reduction, "
            "pack logical order",
        )
    # Race legality: every compact cell is addressed by the linearized dense
    # prefix position and the pack point, so distinct iterations of any
    # prefix loop write disjoint cells and the reduction loop is enclosed by
    # all of them.  The abstract selection carries the chosen anchor into
    # LoopIR, so any dense prefix loop is admissible — exactly the legacy
    # envelope — while the shared origin loop stays serial.
    if plan.parallel_loop is None or (
        plan.parallel_loop.part is not LoopPart.LOGICAL
        or plan.parallel_loop.index_id not in result_tile.result_prefix
    ):
        _fail(
            "invalid_schedule_result_tile",
            "heap accumulation requires one dense result-prefix loop as "
            "the explicit parallel loop so the shared origin loop stays "
            "serial",
        )


def _check_auto_plan_family(plan: LoopPlan) -> None:
    """Admit exactly the measured automatic replay contracts."""

    if plan.auto_policy is None:
        _fail(
            "auto_origin_policy",
            "automatic plans must carry the versioned origin policy "
            "verified at the LoopPlan boundary",
        )
    if (
        plan.panel_bounds
        or plan.relayout is not None
        or plan.result_tile is not None
        or plan.parallel_loop is not None
    ):
        _fail(
            "unsupported_schedule_auto_family",
            "automatic plans are migrated for the tile-free and "
            "regblock stack-form replay contracts only",
        )
    if not plan.tiles and plan.workspace is None:
        return

    def _auto_tile_shape(tile: LoopTile) -> bool:
        if plan.auto_policy is None or not plan.loop_order:
            return False
        if plan.auto_policy.regblock_enabled:
            placement_ok = (
                tile.placement.kind is PlacementKind.CHILD_OF
                and tile.placement.parent is not None
                and tile.placement.parent == LoopRef(plan.loop_order[0])
            )
        else:
            placement_ok = tile.placement.kind is PlacementKind.OUTERMOST
        return (
            tile.kind == "affine"
            and tile.accumulation == "direct"
            and not tile.parallel
            and tile.unroll
            and tile.width == plan.auto_policy.tile_width
            and placement_ok
        )

    stack_form = (
        plan.auto_policy is not None
        and plan.auto_policy.regblock_enabled
        and len(plan.tiles) == 1
        and plan.workspace is not None
        and plan.workspace.dense
        and plan.workspace.axis_loops == (plan.tiles[0].loop,)
        and _auto_tile_shape(plan.tiles[0])
    )
    reduce_out_form = (
        plan.auto_policy is not None
        and plan.workspace is not None
        and plan.workspace.dense
        and len(plan.workspace.axis_loops) == 1
        and len(plan.tiles) >= 2
        and all(_auto_tile_shape(tile) for tile in plan.tiles)
        and len({tile.loop for tile in plan.tiles}) == len(plan.tiles)
        and sum(tile.loop == plan.workspace.axis_loops[0] for tile in plan.tiles) == 1
        and sum(tile.loop == plan.workspace.reduction_loop for tile in plan.tiles) == 1
    )
    if not stack_form and not reduce_out_form:
        _fail(
            "unsupported_schedule_auto_family",
            "automatic plans are migrated for the tile-free, regblock "
            "stack-form, and dense reduce-out replay contracts only; "
            "the sparse-workspace family stays on the legacy path",
        )
    return


def _check_plan_families(plan: LoopPlan) -> None:
    """Fail closed on every plan fact outside the migrated schedule families.

    The sparse-panel family admits exactly the legacy plan shape: at most
    one panel tile, listed after every affine tile, with direct serial
    accumulation, exactly one corresponding ``PanelBound``, and the
    mandatory parallel row loop.  Outside panel and heap plans, an explicit
    ``parallel_loop`` is admitted only when the final typed selection pass
    can prove a supported dense logical or affine-origin anchor and its race
    and work contracts.

    Automatic plans are admitted for three measured auto-replay contracts:
    the tile-free family (one already verified logical order and no other
    decision), the regblock-arm stack form (one dense workspace over a
    single trailing free axis plus exactly one direct serial ``CHILD_OF``
    tile of that axis under the row loop), and the dense reduce-out form
    (the same dense workspace plus arm-uniform candidate tiles that split
    the last reduction loop and the workspace axis exactly once each).
    ``provenance="auto"`` identifies those replay contracts; the verified
    ``auto_policy`` origin fact ties every recorded tile and workspace
    decision to its re-derivation, but the cost-model order itself remains
    attested rather than re-proved.  The dense root-scope reduction family
    records the elided decision and flows through the tile-free contract
    (its abandoned materialized form never compiled).  The sparse-workspace
    family has no typed emission twin yet and stays fail-closed.  Every
    other provenance fails closed.
    """

    if plan.provenance == "auto":
        _check_auto_plan_family(plan)
        return
    if plan.auto_policy is not None:
        _fail(
            "auto_policy_provenance",
            "the automatic origin policy is valid only for automatic plans",
        )
    if plan.provenance != "explicit":
        _fail(
            "unsupported_schedule_provenance",
            f"{plan.provenance!r} scheduling stays on the legacy path; only "
            "explicit and tile-free automatic schedules are migrated",
        )
    _check_heap_plan_family(plan)
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
        expected_order = (
            relayout.row_loop.index_id,
            relayout.panel_loop.index_id,
            relayout.pack_loop.index_id,
        )
        if plan.loop_order != expected_order:
            _fail(
                "invalid_schedule_relayout",
                "packed relayout requires the exact logical row, panel, "
                "pack loop order",
            )
        if (
            relayout.access_indices
            != (
                relayout.panel_loop.index_id,
                relayout.pack_loop.index_id,
            )
            or relayout.operand_panel_level != 0
            or relayout.operand_pack_level != 1
        ):
            _fail(
                "invalid_schedule_relayout",
                "packed relayout requires the rank-2 panel/pack access and "
                "physical levels 0/1",
            )
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
        if pack_tile.accumulation == "heap":
            if (
                plan.result_tile is None
                or plan.result_tile.tile_loop != relayout.pack_loop
            ):
                _fail(
                    "invalid_schedule_result_tile",
                    "a heap-accumulation pack tile requires the plan's "
                    "result tile to compact the same pack loop",
                )
        elif pack_tile.accumulation != "direct":
            _fail(
                "unsupported_schedule_accumulation",
                "operand relayout composes with direct or heap "
                "accumulation; stack result tiles remain a separate "
                "schedule family",
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
    for tile in plan.tiles:
        if tile.kind == "panel":
            continue
        if tile.accumulation not in ("direct", "stack", "heap"):
            _fail(
                "unsupported_schedule_accumulation",
                f"{tile.accumulation!r} accumulation is not a migrated "
                "schedule family",
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
    chain — the region's execution order.  A relayout staging region or a
    heap result-tile region is transparent here: each binds no loop, so
    provenance lists exactly the chain loops and every region's placement
    is covered by the carrier's deterministic replay equality.
    """

    relayout_stages: List[RelayoutStage] = []
    result_tile_regions: List[ResultTileRegion] = []
    loops, leaf = _decompose_chain(
        program,
        relayout_sink=relayout_stages,
        result_tile_sink=result_tile_regions,
    )
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
    relayout fact on the fully scheduled chain by :func:`apply_relayout`,
    the result-tile fact by :func:`apply_result_tile`, and finally the
    explicit ``parallel_loop`` by :func:`select_parallel_loop`.  The latter
    validates panel/heap anchor restrictions and materializes one verified
    program-level selection after every structural transform.
    """

    checked = _validate_plan_for_pass(plan)
    scheduled = reorder_loops(program, checked.loop_order)
    if (
        checked.provenance == "auto"
        and checked.workspace is not None
        and any(tile.loop == checked.workspace.reduction_loop for tile in checked.tiles)
    ):
        # The reduce-out automatic family strip-mines the last reduction
        # loop as well as the workspace axis (plus any recorded prefix
        # candidates); the fused pass consumes the workspace fact and every
        # tile at once with the legacy add_tile LIFO placement semantics.
        return select_parallel_loop(
            apply_reduce_out_tiles(scheduled, checked.tiles, checked.workspace),
            checked,
        )
    for tile in checked.tiles:
        # The admitted automatic tile family records the legacy surgery's
        # decisions verbatim: a standalone workspace fact plus one direct
        # serial tile of the workspace axis.  That composition is measured
        # byte-identical to the explicit stack tile, so it lowers through
        # the same verified pass; the plan facts themselves stay untouched.
        stack_equivalent = (
            checked.provenance == "auto"
            and checked.workspace is not None
            and tile.kind == "affine"
        )
        if tile.kind == "panel":
            scheduled = apply_panel_tile(
                scheduled,
                tile,
                checked.panel_bounds[0],
                checked.parallel_loop,
            )
        elif tile.accumulation == "stack" or stack_equivalent:
            scheduled = apply_stack_tile(
                scheduled,
                (
                    tile
                    if tile.accumulation == "stack"
                    else replace(tile, accumulation="stack")
                ),
            )
        else:
            scheduled = apply_affine_tile(scheduled, tile)
    if checked.relayout is not None:
        scheduled = apply_relayout(scheduled, checked.relayout)
    if checked.result_tile is not None:
        scheduled = apply_result_tile(scheduled, checked.result_tile)
    scheduled = select_parallel_loop(scheduled, checked)
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
    base_result_tiles: List[ResultTileRegion] = []
    base_loops, base_leaf = _decompose_chain(
        base_program,
        relayout_sink=base_relayouts,
        result_tile_sink=base_result_tiles,
    )
    if (
        base_relayouts
        or base_result_tiles
        or base_program.parallel is not None
        or type(base_leaf) is WorkspaceRegion
        or type(base_leaf) is TiledReduce
        or any(
            type(loop) in (TileOuterFor, TileInnerFor, *_PANEL_TYPES)
            for loop in base_loops
        )
    ):
        _fail(
            "scheduled_base_not_unscheduled",
            "ScheduledLoopIR.base_program must not already contain split "
            "loops, panels, workspace regions, staging regions, "
            "result-tile regions, or a parallel selection",
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

    A heap result-tile region erases to its direct-accumulation
    equivalent: the region wrapper is dropped and its :class:`TiledReduce`
    leaf becomes the plain result :class:`StoreReduce` it redirected —
    accumulating into a fresh zeroed compact tile per strip and copying
    every clamped cell out exactly once is a reassociation of the direct
    ADD reduction into the zero-initialized result, the same explicit
    contract the workspace erasure relies on.

    This is the semantics-preserving erasure the oracle differentials use
    to prove scheduled programs visit every iteration point exactly once.
    A program with no splits is returned unchanged.
    """

    verify_program(program)
    relayout_stages: List[RelayoutStage] = []
    result_tile_regions: List[ResultTileRegion] = []
    loops, leaf = _decompose_chain(
        program,
        relayout_sink=relayout_stages,
        result_tile_sink=result_tile_regions,
    )
    if (
        not relayout_stages
        and not result_tile_regions
        and type(leaf) is not WorkspaceRegion
        and not any(
            type(node) in (TileOuterFor, TileInnerFor, *_PANEL_TYPES) for node in loops
        )
    ):
        if program.parallel is None:
            return program
        # An abstract parallel selection is schedule state: erasing a
        # program that carries only the selection still returns the
        # unscheduled base.
        stripped_builder = LoopIRBuilder.resuming(program)
        return _rebuild_program(
            program,
            stripped_builder,
            loops,
            leaf=leaf,
            keep_parallel=False,
        )
    builder = LoopIRBuilder.resuming(program)
    region_loops: List[_LoopNode] = []
    erased_leaf: _ChainEnd = leaf
    tiled_results = {
        region.decl.result_tile: region.decl.result for region in result_tile_regions
    }
    if type(leaf) is TiledReduce:
        if leaf.result_tile not in tiled_results:
            _fail(
                "unsupported_schedule_shape",
                "tiled-reduce erasure requires the leaf's enclosing "
                "result-tile region on the chain",
            )
        leaf = builder.store_reduce(
            tiled_results[leaf.result_tile], leaf.indices, leaf.op, leaf.value
        )
        erased_leaf = leaf
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
    return _rebuild_program(
        program,
        builder,
        erased,
        leaf=erased_leaf,
        keep_parallel=False,
    )
