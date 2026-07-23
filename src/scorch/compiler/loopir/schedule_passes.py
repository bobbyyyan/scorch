"""Pure typed scheduling passes over verified production LoopIR.

Phase 6 applies a verified :class:`~scorch.compiler.loop_plan.LoopPlan` to a
verified unscheduled LoopIR program through exactly two typed
transformations, replacing the legacy scheduler's private CIN tree surgery
for the migrated schedule families:

- :func:`reorder_loops` permutes a single-chain loop nest into the plan's
  logical loop order;
- :func:`apply_affine_tile` strip-mines one dense logical loop into a
  :class:`~scorch.compiler.loopir.nodes.TileOuterFor` /
  :class:`~scorch.compiler.loopir.nodes.TileInnerFor` pair with an
  artifact-local :class:`~scorch.compiler.loopir.nodes.TileId`, mirroring
  the legacy placement semantics (``outermost`` / ``child_of`` /
  ``at_depth`` resolved against the current chain, applied in plan tile
  order).

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
    PlacementKind,
    _validate_loop_plan_structure,
)
from .build import LoopIRBuilder
from .nodes import (
    AppendEntry,
    Block,
    DenseFor,
    DensePosition,
    Expr,
    IndexValue,
    LoopProgram,
    MergedSparseFor,
    PositionId,
    PositionValue,
    SparseCursorDecl,
    SparseFor,
    Stmt,
    Store,
    StoreReduce,
    TileId,
    TileInnerFor,
    TileOuterFor,
)
from .verifier import verify_program

_LoopNode = Union[DenseFor, SparseFor, MergedSparseFor, TileOuterFor, TileInnerFor]
_LeafNode = Union[Store, StoreReduce, AppendEntry]

_LOOP_TYPES = (DenseFor, SparseFor, MergedSparseFor, TileOuterFor, TileInnerFor)
_LEAF_TYPES = (Store, StoreReduce, AppendEntry)


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
    if type(node) is SparseFor:
        return node.coord_index, LoopPart.LOGICAL
    assert type(node) is MergedSparseFor
    return node.coord_index, LoopPart.LOGICAL


def _decompose_chain(program: LoopProgram) -> Tuple[List[_LoopNode], _LeafNode]:
    """Split one migrated-family program into its loop chain and leaf."""

    loops: List[_LoopNode] = []
    body: Stmt = program.body
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
        if type(only) in _LEAF_TYPES:
            if not loops:
                _fail(
                    "unsupported_schedule_shape",
                    "scheduling passes require at least one loop",
                )
            return loops, only  # type: ignore[return-value]
        _fail(
            "unsupported_schedule_shape",
            f"unsupported chain statement {type(only).__name__}",
        )


def _rebuild_program(
    program: LoopProgram,
    builder: LoopIRBuilder,
    loops: Sequence[_LoopNode],
    leaf: _LeafNode,
) -> LoopProgram:
    """Reassemble the chain with fresh loop/block nodes and shared subtrees."""

    body = builder.block((leaf,))
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
    for node in loops:
        if type(node) in (TileOuterFor, TileInnerFor):
            _fail(
                "reorder_split_chain",
                "loop reorder operates on unsplit logical chains; apply it "
                "before affine tiling",
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
                f"at_depth placement {depth} is outside the chain range "
                f"0..{len(loops)}",
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
            "compressed iteration is not represented",
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


def _check_plan_families(plan: LoopPlan) -> None:
    """Fail closed on every plan fact outside the migrated schedule families."""

    if plan.provenance != "explicit":
        _fail(
            "unsupported_schedule_provenance",
            f"{plan.provenance!r} scheduling stays on the legacy path; only "
            "explicit schedules are migrated",
        )
    if plan.panel_bounds:
        _fail(
            "unsupported_schedule_panel",
            "sparse coordinate panel tiling is not a migrated schedule " "family",
        )
    if plan.relayout is not None:
        _fail(
            "unsupported_schedule_relayout",
            "operand relayout/staging is not a migrated schedule family",
        )
    if plan.result_tile is not None:
        _fail(
            "unsupported_schedule_result_tile",
            "heap result tiling is not a migrated schedule family",
        )
    if plan.parallel_loop is not None:
        _fail(
            "unsupported_schedule_parallel",
            "explicit parallel-loop selection is not a migrated schedule " "family",
        )
    for tile in plan.tiles:
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


def _chain_provenance(
    program: LoopProgram,
) -> Tuple[ScheduledLoopProvenance, ...]:
    loops, _ = _decompose_chain(program)
    provenance: List[ScheduledLoopProvenance] = []
    for node in loops:
        index, part = _loop_key(node)
        tile = (
            node.tile  # type: ignore[union-attr]
            if type(node) in (TileOuterFor, TileInnerFor)
            else None
        )
        provenance.append(ScheduledLoopProvenance(tile, index, part))
    return tuple(provenance)


def _apply_schedule_program(program: LoopProgram, plan: LoopPlan) -> LoopProgram:
    """Apply the supported plan decisions without constructing the carrier."""

    checked = _validate_plan_for_pass(plan)
    scheduled = reorder_loops(program, checked.loop_order)
    for tile in checked.tiles:
        scheduled = apply_affine_tile(scheduled, tile)
    return scheduled


def _check_exact_carrier_state(value: object, expected: Set[str]) -> bool:
    state = getattr(value, "__dict__", None)
    return type(state) is dict and set(state) == expected


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
        if provenance.tile is not None and (
            type(provenance.tile) is not TileId
            or type(provenance.tile.value) is not int
        ):
            _fail(
                "invalid_scheduled_artifact",
                f"loops[{position}].tile must be None or an int-valued TileId",
            )
        if (
            type(provenance.index) is not IndexId
            or type(provenance.index.value) is not int
            or type(provenance.part) is not LoopPart
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
    base_loops, _ = _decompose_chain(base_program)
    if any(type(loop) in (TileOuterFor, TileInnerFor) for loop in base_loops):
        _fail(
            "scheduled_base_not_unscheduled",
            "ScheduledLoopIR.base_program must not already contain split loops",
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
    """Erase every affine split back to its unscheduled point loop.

    ``TileInnerFor`` becomes a plain ``DenseFor`` at the point loop's chain
    position and the paired ``TileOuterFor`` is dropped, which restores
    exactly the pre-tiling chain (splitting never moves the point loop).
    This is the semantics-preserving erasure the oracle differentials use
    to prove scheduled programs visit every iteration point exactly once.
    A program with no splits is returned unchanged.
    """

    verify_program(program)
    loops, leaf = _decompose_chain(program)
    if not any(type(node) in (TileOuterFor, TileInnerFor) for node in loops):
        return program
    builder = LoopIRBuilder.resuming(program)
    erased: List[_LoopNode] = []
    for node in loops:
        if type(node) is TileOuterFor:
            continue
        if type(node) is TileInnerFor:
            erased.append(builder.dense_for(node.index, node.dimension, node.body))
        else:
            erased.append(node)
    return _rebuild_program(program, builder, erased, leaf)
