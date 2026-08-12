"""Semantic legality checks for immutable :class:`LoopPlan` artifacts.

The public scheduler still contains legacy validation while it builds a plan.
This module is the independent trust-boundary proof: it consumes only stable IDs,
one immutable ``CINAnalysis`` result, and the frozen plan.  Derived placement and
lifetime facts are local values; none are attached to CIN or enter cache identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .cin import Operation
from .cin_analysis import (
    AccessKind,
    AccessLayoutInfo,
    AssignmentInfo,
    CINAnalysis,
    FrozenMap,
)
from .diagnostics import InvalidSchedule, UnsupportedFeature, VerificationError
from .identity import AccessId, IndexId, NodeId, SymbolId
from .loop_plan import (
    AutoOriginPolicy,
    LoopPart,
    LoopPlan,
    LoopPlacement,
    LoopRef,
    LoopTile,
    OperandRelayout,
    PlacementKind,
    WorkspaceInsertion,
)
from ..format import LevelType


@dataclass(frozen=True)
class LoopPlanDiagnostic:
    """One immutable semantic failure at the LoopPlan trust boundary."""

    code: str
    message: str
    path: Tuple[str, ...]
    index_id: Optional[IndexId] = None
    symbol_id: Optional[SymbolId] = None
    access_id: Optional[AccessId] = None
    stage: str = "loop_plan"
    pass_name: str = "verify_loop_plan"


@dataclass(frozen=True)
class LoopPlanLegalityFacts:
    """Immutable CIN-derived facts needed by semantic plan verification."""

    analysis: CINAnalysis
    bound_index_ids: Tuple[IndexId, ...]
    free_index_ids: Tuple[IndexId, ...]
    reduction_index_ids: Tuple[IndexId, ...]
    loop_positions: FrozenMap[IndexId, int]
    result_accesses: Tuple[AccessLayoutInfo, ...]
    read_accesses: Tuple[AccessLayoutInfo, ...]
    assignments: Tuple[AssignmentInfo, ...]


@dataclass(frozen=True)
class _PlacementState:
    """Verifier-local concrete common-prefix model, not a compiler IR."""

    prefix: Tuple[LoopRef, ...]
    affine_index_ids: Tuple[IndexId, ...] = ()


def _diagnostic(
    code: str,
    message: str,
    path: Tuple[str, ...],
    *,
    index_id: Optional[IndexId] = None,
    symbol_id: Optional[SymbolId] = None,
    access_id: Optional[AccessId] = None,
) -> LoopPlanDiagnostic:
    return LoopPlanDiagnostic(
        code=code,
        message=message,
        path=path,
        index_id=index_id,
        symbol_id=symbol_id,
        access_id=access_id,
    )


def _invalid(
    code: str,
    message: str,
    path: Tuple[str, ...],
    *,
    index_id: Optional[IndexId] = None,
    symbol_id: Optional[SymbolId] = None,
    access_id: Optional[AccessId] = None,
) -> None:
    diagnostic = _diagnostic(
        code,
        message,
        path,
        index_id=index_id,
        symbol_id=symbol_id,
        access_id=access_id,
    )
    raise InvalidSchedule(
        f"stage=LoopPlan: {code} at {'/'.join(path)}: {message}",
        diagnostics=(diagnostic,),
    )


def _unsupported(
    code: str,
    message: str,
    path: Tuple[str, ...],
    *,
    index_id: Optional[IndexId] = None,
    symbol_id: Optional[SymbolId] = None,
    access_id: Optional[AccessId] = None,
) -> None:
    diagnostic = _diagnostic(
        code,
        message,
        path,
        index_id=index_id,
        symbol_id=symbol_id,
        access_id=access_id,
    )
    raise UnsupportedFeature(
        f"stage=LoopPlan: {code} at {'/'.join(path)}: {message}",
        diagnostics=(diagnostic,),
    )


def _verification_error(code: str, message: str, path: Tuple[str, ...]) -> None:
    diagnostic = _diagnostic(code, message, path)
    raise VerificationError(
        f"stage=LoopPlan: {code} at {'/'.join(path)}: {message}",
        diagnostics=(diagnostic,),
    )


def _build_facts(analysis: CINAnalysis, plan: LoopPlan) -> LoopPlanLegalityFacts:
    bound_index_ids = tuple(
        index_id
        for index_id, definition in analysis.index_definitions.items()
        if definition.bindings
    )
    free_ids = analysis.free_index_ids
    reduction_ids = analysis.reduction_index_ids
    free_set = set(free_ids)
    reduction_set = set(reduction_ids)
    bound_set = set(bound_index_ids)
    if free_set & reduction_set:
        _verification_error(
            "ambiguous_index_classification",
            "normalized CIN classifies an IndexId as both free and reduction",
            ("analysis", "indices"),
        )
    if free_set | reduction_set != bound_set:
        _verification_error(
            "incomplete_index_classification",
            "bound IndexIds must equal the disjoint free/reduction union",
            ("analysis", "indices"),
        )

    positions = FrozenMap.from_items(
        (index_id, position) for position, index_id in enumerate(plan.loop_order)
    )
    result_accesses = tuple(
        layout
        for layout in analysis.access_layouts.values()
        if not layout.is_workspace
        and layout.kind in (AccessKind.WRITE, AccessKind.REDUCTION_WRITE)
    )
    read_accesses = tuple(
        layout
        for layout in analysis.access_layouts.values()
        if layout.kind == AccessKind.READ
    )
    assignments = tuple(
        analysis.assignments[assignment_id]
        for assignment_id in analysis.assignment_order
    )
    return LoopPlanLegalityFacts(
        analysis=analysis,
        bound_index_ids=bound_index_ids,
        free_index_ids=free_ids,
        reduction_index_ids=reduction_ids,
        loop_positions=positions,
        result_accesses=result_accesses,
        read_accesses=read_accesses,
        assignments=assignments,
    )


def _verify_schedulable_scope(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
) -> None:
    """Require the single loop chain that the transitional adapter can rebuild."""

    if not facts.bound_index_ids:
        if plan.provenance != "auto":
            _unsupported(
                "scalar_plan_provenance",
                "loop-free CIN supports only no-decision auto replay",
                ("provenance",),
            )
        return
    binding_scopes = {}
    for index_id in facts.bound_index_ids:
        bindings = facts.analysis.index_definitions[index_id].bindings
        if len(bindings) != 1:
            _unsupported(
                "ambiguous_scheduling_scope",
                "one LoopPlan cannot address multiple bindings of one IndexId",
                ("analysis", "index_bindings"),
                index_id=index_id,
            )
        binding_scopes[bindings[0].scope_id] = index_id

    current = facts.analysis.root_id
    visited = set()
    while current in binding_scopes:
        visited.add(current)
        children = tuple(
            scope_id
            for scope_id in binding_scopes
            if facts.analysis.scope_parents[scope_id] == current
        )
        if not children:
            break
        if len(children) != 1:
            _unsupported(
                "branched_scheduling_scope",
                "LoopPlan replay requires one consecutive outer ForAll chain",
                ("analysis", "index_bindings"),
            )
        current = children[0]
    if set(binding_scopes) != visited:
        _unsupported(
            "non_prefix_scheduling_scope",
            "all scheduled binders must form the root's consecutive ForAll prefix",
            ("analysis", "index_bindings"),
        )


def _require_relative_order(
    facts: LoopPlanLegalityFacts,
    index_ids: Tuple[IndexId, ...],
    code: str,
    message: str,
    path: Tuple[str, ...],
    access_id: AccessId,
) -> None:
    for parent, child in zip(index_ids, index_ids[1:]):
        if facts.loop_positions[parent] >= facts.loop_positions[child]:
            _invalid(
                code,
                message,
                path,
                index_id=child,
                access_id=access_id,
            )


def _verify_storage_order(facts: LoopPlanLegalityFacts) -> None:
    for layout in facts.result_accesses:
        _require_relative_order(
            facts,
            layout.storage_index_ids,
            "result_storage_order",
            "non-workspace result levels must follow physical storage order",
            ("loop_order", "result"),
            layout.access_id,
        )

    for layout in facts.analysis.access_layouts.values():
        if any(level != LevelType.DENSE for level in layout.level_types):
            _require_relative_order(
                facts,
                layout.storage_index_ids,
                "sparse_parent_dominance",
                "every physical parent must precede its sparse-dependent child",
                ("loop_order", "sparse_access"),
                layout.access_id,
            )

    singleton = next(
        (
            layout
            for layout in facts.analysis.access_layouts.values()
            if LevelType.SINGLETON in layout.level_types
        ),
        None,
    )
    if singleton is not None:
        _unsupported(
            "singleton_level",
            "singleton storage levels are not supported by the current lowerer",
            ("analysis", "access_layouts"),
            access_id=singleton.access_id,
        )


def _all_dense_results(facts: LoopPlanLegalityFacts) -> bool:
    return bool(facts.result_accesses) and all(
        all(level == LevelType.DENSE for level in layout.level_types)
        for layout in facts.result_accesses
    )


def _workspace_domain(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
) -> Tuple[IndexId, ...]:
    reduction_positions = tuple(
        facts.loop_positions[index_id] for index_id in facts.reduction_index_ids
    )
    if not reduction_positions or not facts.free_index_ids:
        return ()
    last_reduction = max(reduction_positions)
    free_after = tuple(
        index_id
        for index_id in plan.loop_order[last_reduction + 1 :]
        if index_id in facts.free_index_ids
    )
    if not free_after:
        return ()
    if _all_dense_results(facts):
        if len(free_after) != 1:
            return ()
        trailing_id = free_after[0]
        if not any(
            trailing_id in layout.storage_index_ids
            and layout.level_types[layout.storage_index_ids.index(trailing_id)]
            == LevelType.DENSE
            for layout in facts.result_accesses
        ):
            return ()
    return free_after


def _workspace_uses_dense_storage(
    facts: LoopPlanLegalityFacts,
    workspace_domain: Tuple[IndexId, ...],
) -> bool:
    """Mirror the legacy workspace representation decision exactly.

    Workspace storage depends on the levels addressed by the workspace, not
    on whether every level of the result tensor is dense.  The current
    lowering supports a dense workspace only for one trailing result level;
    every other domain uses the sparse fallback representation.
    """

    if len(workspace_domain) != 1 or not facts.result_accesses:
        return False
    result = facts.result_accesses[0]
    workspace_index = workspace_domain[0]
    try:
        storage_position = result.storage_index_ids.index(workspace_index)
    except ValueError:
        return False
    return result.level_types[storage_position] == LevelType.DENSE


def _assignment_is_additive(assignment: AssignmentInfo) -> bool:
    return assignment.update_op in (None, Operation.ADD)


def _derive_auto_decisions(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
    policy: AutoOriginPolicy,
) -> Tuple[Tuple[LoopTile, ...], Optional[WorkspaceInsertion]]:
    """Re-derive the heuristic tile and workspace facts from the policy.

    This mirrors the legacy automatic surgery exactly — workspace insertion
    at the last in-order reduction, then per-arm tiling of the dense free
    candidates on the post-insertion nest — expressed over the immutable
    analysis instead of mutable CIN.  Candidate order follows the
    post-insertion access traversal (producer reads, then the consumer
    result); a dense-output root-scope insertion is elided by both the
    plan-free production surgery and the empty-Schedule replay (the
    abandoned materialized form never compiled), so the derived facts for
    that family are no tiles and no workspace.
    """

    workspace_domain = _workspace_domain(facts, plan)
    dense_output = _all_dense_results(facts)

    all_bound = set(facts.bound_index_ids)
    sparse_ids = set()
    ordered_layouts = facts.read_accesses + facts.result_accesses
    for layout in ordered_layouts:
        for index_id, level in zip(layout.storage_index_ids, layout.level_types):
            if level != LevelType.DENSE:
                sparse_ids.add(index_id)
    candidates: List[IndexId] = []
    for layout in ordered_layouts:
        if set(layout.logical_index_ids) == all_bound:
            continue
        # Candidate enumeration follows the legacy access-index traversal,
        # which is the access's logical index order.  A permuted physical
        # ``mode_order`` reorders ``storage_index_ids`` but does not change
        # the order in which legacy ``add_tile`` consumes candidates.
        for index_id in layout.logical_index_ids:
            if index_id not in sparse_ids and index_id not in candidates:
                candidates.append(index_id)
    if plan.loop_order:
        first_loop = plan.loop_order[0]
        candidates = [index_id for index_id in candidates if index_id != first_loop]

    def filter_sparse_retraversal(
        candidate_ids: List[IndexId],
        *,
        surviving_prefix_end: Optional[int] = None,
    ) -> List[IndexId]:
        if policy.regblock_enabled:
            return candidate_ids

        def causes_sparse_retraversal(index_id: IndexId) -> bool:
            position = facts.loop_positions.get(index_id)
            if position is None:
                return False
            if surviving_prefix_end is not None and position >= surviving_prefix_end:
                # After workspace insertion the replaced reduction subtree is
                # represented by a Where.  Legacy ``cin.loop_order`` retains
                # only the common prefix as direct IndexVar entries; loops in
                # or below the region therefore do not trigger the
                # re-traversal filter.
                return False
            return any(
                plan.loop_order[inner] in sparse_ids for inner in range(1, position)
            )

        return [
            index_id
            for index_id in candidate_ids
            if not causes_sparse_retraversal(index_id)
        ]

    # The first selection runs before workspace insertion and decides whether
    # a dense-result workspace is worth materializing.  At that point the loop
    # order is still the original flat plan order.
    pre_insertion_candidates = filter_sparse_retraversal(list(candidates))
    should_insert = bool(workspace_domain)
    will_tile = bool(pre_insertion_candidates)
    materialize = should_insert and (not dense_output or will_tile)
    if not materialize:
        return (), None
    last_reduction_position = max(
        facts.loop_positions[index_id] for index_id in facts.reduction_index_ids
    )
    last_reduction_id = plan.loop_order[last_reduction_position]
    workspace_dense = _workspace_uses_dense_storage(facts, workspace_domain)

    # The second selection runs on the post-insertion tree.  Sparse workspace
    # insertion adds the producer's reduction loop to ``no_tile_list``.
    # Moreover, the Where branches are nested list entries in legacy
    # ``cin.loop_order``, so sparse re-traversal is checked only for candidate
    # loops in the surviving common prefix.
    post_insertion_candidates = list(candidates)
    if not workspace_dense:
        post_insertion_candidates = [
            index_id
            for index_id in post_insertion_candidates
            if index_id != last_reduction_id
        ]
    post_insertion_candidates = filter_sparse_retraversal(
        post_insertion_candidates,
        surviving_prefix_end=last_reduction_position,
    )
    root_scope = last_reduction_position == 0
    # An affine tile over a non-dense receiver is illegal: lifting the tile loop
    # (OUTERMOST, or CHILD_OF the root on the regblock arm) stops the nest
    # visiting a compressed result level's parent prefix in lexicographic order,
    # so the assembled position array is malformed.  ``Scheduler.apply_schedule``
    # already refuses exactly this for an explicitly requested schedule; that
    # check reads ``schedule.tiles`` and so is vacuous on the automatic origin,
    # which chooses its tiles here and in
    # ``Scheduler._apply_tiling_heuristics`` instead.
    #
    # This condition and the one in ``_apply_tiling_heuristics`` are the SAME
    # rule stated in the two layers that independently decide it -- the legacy
    # mutation and this typed re-derivation.  They must move together: if only
    # the mutation is changed, the recorded plan stops matching the derived
    # decisions and every affected cell fails ``auto_tile_decision`` instead of
    # reaching its real disposition.
    if root_scope or not dense_output:
        derived_tiles: Tuple[LoopTile, ...] = ()
    else:
        placement = (
            LoopPlacement(
                PlacementKind.CHILD_OF,
                parent=LoopRef(plan.loop_order[0]),
            )
            if policy.regblock_enabled
            else LoopPlacement(PlacementKind.OUTERMOST)
        )
        derived_tiles = tuple(
            LoopTile(
                loop=LoopRef(index_id),
                width=policy.tile_width,
                placement=placement,
                parallel=False,
                kind="affine",
                accumulation="direct",
                unroll=True,
            )
            for index_id in post_insertion_candidates
        )
    record_workspace = not dense_output or bool(derived_tiles)
    if not record_workspace:
        return derived_tiles, None
    derived_workspace = WorkspaceInsertion(
        reduction_loop=LoopRef(plan.loop_order[last_reduction_position]),
        axis_loops=tuple(LoopRef(index_id) for index_id in workspace_domain),
        dense=workspace_dense,
    )
    return derived_tiles, derived_workspace


def _verify_auto_workspace_decision(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
    derived: Optional[WorkspaceInsertion],
) -> None:
    """Require the stored automatic workspace fact to equal the derived one.

    The workspace insertion an automatic plan replays is program-derived
    state, never an unchecked recorded choice: the stored fact (including
    ``None``) must equal the decision this trust boundary re-derives from the
    analyzed CIN, the plan order, and the recorded tiles, exactly.
    """

    del facts
    if plan.workspace != derived:
        _invalid(
            "auto_workspace_decision",
            "the recorded automatic workspace insertion must equal the "
            "derived replay decision exactly",
            ("workspace",),
        )


def _verify_tiling_capabilities(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
) -> bool:
    """Validate accumulation policy and return whether replay inserts workspace."""

    affine_tiles = tuple(tile for tile in plan.tiles if tile.kind == "affine")
    panel_tiles = tuple(tile for tile in plan.tiles if tile.kind == "panel")
    workspace_domain = _workspace_domain(facts, plan)
    source_has_workspace = any(
        layout.is_workspace for layout in facts.analysis.access_layouts.values()
    )
    if source_has_workspace and plan.provenance != "auto":
        _unsupported(
            "workspace_plan_provenance",
            "CIN with an existing workspace supports only no-decision auto replay",
            ("provenance",),
        )
    has_parallel_decision = plan.parallel_loop is not None or any(
        tile.parallel for tile in plan.tiles
    )
    if len(facts.assignments) != 1 and (
        plan.tiles or has_parallel_decision or workspace_domain
    ):
        _unsupported(
            "multi_assignment_schedule",
            "derived workspaces, tiling, and parallel decisions require one assignment",
            ("analysis", "assignments"),
        )
    sparse_workspace = bool(workspace_domain) and not _all_dense_results(facts)
    sparse_workspace = sparse_workspace and not source_has_workspace
    for tile in affine_tiles:
        layouts = tuple(
            layout
            for layout in facts.analysis.access_layouts.values()
            if tile.loop.index_id in layout.storage_index_ids
        )
        if any(
            layout.level_types[layout.storage_index_ids.index(tile.loop.index_id)]
            != LevelType.DENSE
            for layout in layouts
        ):
            _unsupported(
                "sparse_affine_tile",
                "affine tiling cannot split a sparse storage coordinate",
                ("tiles", str(plan.tiles.index(tile))),
                index_id=tile.loop.index_id,
            )
    if plan.provenance == "auto":
        if (
            panel_tiles
            or plan.panel_bounds
            or plan.relayout is not None
            or plan.result_tile is not None
        ):
            _invalid(
                "auto_explicit_decision",
                "auto LoopPlans cannot contain explicit-only scheduling decisions",
                ("provenance",),
            )
        if plan.auto_policy is None:
            _invalid(
                "auto_origin_policy",
                "recorded automatic plans must carry the versioned origin "
                "policy fact",
                ("auto_policy",),
            )
        origin_policy = plan.auto_policy
        assert origin_policy is not None
        for position, tile in enumerate(affine_tiles):
            if tile.accumulation != "direct" or tile.parallel:
                _invalid(
                    "auto_tile_policy",
                    "recorded auto tiles must use serial direct accumulation",
                    ("tiles", str(position)),
                    index_id=tile.loop.index_id,
                )
        if not affine_tiles and not sparse_workspace:
            derived_tiles: Tuple[LoopTile, ...] = ()
            derived_workspace: Optional[WorkspaceInsertion] = None
            if not source_has_workspace:
                derived_tiles, derived_workspace = _derive_auto_decisions(
                    facts, plan, origin_policy
                )
            if plan.tiles != derived_tiles:
                _invalid(
                    "auto_tile_decision",
                    "the recorded automatic tiles must equal the "
                    "policy-derived heuristic decisions exactly",
                    ("tiles",),
                )
            _verify_auto_workspace_decision(facts, plan, derived_workspace)
            return derived_workspace is not None
        if not workspace_domain:
            _invalid(
                "auto_accumulator_lifetime",
                "recorded auto tiles require a legal derived workspace lifetime",
                ("tiles",),
            )
        reduction_assignments = tuple(
            assignment
            for assignment in facts.assignments
            if assignment.reduction_index_ids
        )
        if not reduction_assignments or any(
            not _assignment_is_additive(assignment)
            for assignment in reduction_assignments
        ):
            _unsupported(
                "auto_reduction_operator",
                "auto workspace tiling supports additive reductions only",
                ("analysis", "assignments"),
            )
        derived_tiles = ()
        derived_workspace = None
        if not source_has_workspace:
            derived_tiles, derived_workspace = _derive_auto_decisions(
                facts, plan, origin_policy
            )
        if plan.tiles != derived_tiles:
            _invalid(
                "auto_tile_decision",
                "the recorded automatic tiles must equal the policy-derived "
                "heuristic decisions exactly",
                ("tiles",),
            )
        _verify_auto_workspace_decision(facts, plan, derived_workspace)
        return derived_workspace is not None

    if plan.workspace is not None:
        _invalid(
            "workspace_provenance",
            "workspace insertion is an automatic-provenance decision; "
            "explicit schedules express accumulator lifetime through tile "
            "accumulation",
            ("workspace",),
        )
    if plan.auto_policy is not None:
        _invalid(
            "auto_policy_provenance",
            "the automatic origin policy is recorded only on automatic "
            "plans; explicit schedules carry no scheduler-policy claim",
            ("auto_policy",),
        )

    if affine_tiles and not _all_dense_results(facts):
        _unsupported(
            "tiled_sparse_output",
            "explicit affine tiling currently requires dense result storage",
            ("tiles",),
        )
    reduction_set = set(facts.reduction_index_ids)
    stack_tiles = tuple(tile for tile in affine_tiles if tile.accumulation == "stack")
    if len(stack_tiles) > 1:
        _unsupported(
            "multiple_stack_tiles",
            "the current accumulator supports one trailing stack tile",
            ("tiles",),
        )
    for position, tile in enumerate(affine_tiles):
        if tile.loop.index_id in reduction_set:
            _unsupported(
                "affine_reduction_tile",
                "explicit affine reduction tiling lacks a spanning accumulator",
                ("tiles", str(position)),
                index_id=tile.loop.index_id,
            )
        if tile.accumulation == "stack" and (workspace_domain != (tile.loop.index_id,)):
            _unsupported(
                "stack_accumulator_lifetime",
                "stack accumulation requires the sole trailing dense free axis",
                ("tiles", str(position)),
                index_id=tile.loop.index_id,
            )
    if stack_tiles:
        reduction_assignments = tuple(
            assignment
            for assignment in facts.assignments
            if assignment.reduction_index_ids
        )
        if not reduction_assignments or any(
            not _assignment_is_additive(assignment)
            for assignment in reduction_assignments
        ):
            _unsupported(
                "stack_reduction_operator",
                "stack accumulation supports additive reductions only",
                ("tiles",),
            )
    if sparse_workspace:
        reduction_assignments = tuple(
            assignment
            for assignment in facts.assignments
            if assignment.reduction_index_ids
        )
        if not reduction_assignments or any(
            not _assignment_is_additive(assignment)
            for assignment in reduction_assignments
        ):
            _unsupported(
                "workspace_reduction_operator",
                "derived sparse workspaces support additive reductions only",
                ("analysis", "assignments"),
            )
    return bool(stack_tiles) or sparse_workspace


def _initial_placement_state(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
    workspace_inserted: bool,
) -> _PlacementState:
    prefix_ids = plan.loop_order
    if workspace_inserted:
        last_reduction = max(
            facts.loop_positions[index_id] for index_id in facts.reduction_index_ids
        )
        prefix_ids = plan.loop_order[:last_reduction]
    return _PlacementState(tuple(LoopRef(index_id) for index_id in prefix_ids))


def _verify_workspace_replay_shape(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
    workspace_inserted: bool,
) -> None:
    if not workspace_inserted or not plan.tiles:
        return
    last_reduction = max(
        facts.loop_positions[index_id] for index_id in facts.reduction_index_ids
    )
    if last_reduction == 0:
        _unsupported(
            "root_workspace_tiling",
            "tiling cannot wrap a workspace inserted at the root scope",
            ("tiles",),
        )


def _placement_depth(
    state: _PlacementState,
    placement: LoopPlacement,
    target: LoopRef,
    path: Tuple[str, ...],
) -> int:
    if placement.kind == PlacementKind.OUTERMOST:
        depth = 0
    elif placement.kind == PlacementKind.CHILD_OF:
        assert placement.parent is not None
        if placement.parent not in state.prefix:
            _invalid(
                "placement_parent_scope",
                "child_of must name an existing common-prefix loop",
                path,
                index_id=placement.parent.index_id,
            )
        depth = state.prefix.index(placement.parent) + 1
    else:
        assert placement.depth is not None
        depth = placement.depth
        if depth > len(state.prefix):
            _invalid(
                "placement_depth_scope",
                "at_depth lies outside the current common loop prefix",
                path,
                index_id=target.index_id,
            )
    if target in state.prefix and depth > state.prefix.index(target):
        _invalid(
            "tile_outer_dominance",
            "an affine outer tile must dominate its inner loop",
            path,
            index_id=target.index_id,
        )
    return depth


def _apply_affine_placements(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
    workspace_inserted: bool,
) -> _PlacementState:
    state = _initial_placement_state(facts, plan, workspace_inserted)
    for position, tile in enumerate(plan.tiles):
        if tile.kind != "affine":
            continue
        target = tile.loop
        depth = _placement_depth(
            state,
            tile.placement,
            target,
            ("tiles", str(position), "placement"),
        )
        rewritten = tuple(
            LoopRef(target.index_id, LoopPart.INNER) if loop == target else loop
            for loop in state.prefix
        )
        outer = LoopRef(target.index_id, LoopPart.OUTER)
        state = _PlacementState(
            rewritten[:depth] + (outer,) + rewritten[depth:],
            state.affine_index_ids + (target.index_id,),
        )
    return state


def _effective_parallel_loop(plan: LoopPlan) -> Optional[LoopRef]:
    if plan.parallel_loop is not None:
        return plan.parallel_loop
    parallel_tile = next((tile for tile in plan.tiles if tile.parallel), None)
    if parallel_tile is None:
        return None
    if parallel_tile.kind != "affine":
        _invalid(
            "parallel_panel_loop",
            "a sparse panel window cannot be selected as an affine parallel loop",
            ("tiles", "parallel"),
            index_id=parallel_tile.loop.index_id,
        )
    return LoopRef(parallel_tile.loop.index_id, LoopPart.OUTER)


def _scope_contains(analysis: CINAnalysis, owner: NodeId, scope: NodeId) -> bool:
    current: Optional[NodeId] = scope
    visited: set[NodeId] = set()
    while current is not None and current not in visited:
        if current == owner:
            return True
        visited.add(current)
        current = analysis.scope_parents.get(current)
    return False


def _verify_parallel_legality(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
    workspace_inserted: bool,
) -> Optional[LoopRef]:
    parallel = _effective_parallel_loop(plan)
    if parallel is None:
        return None
    base = parallel.index_id
    if parallel.part == LoopPart.INNER:
        _invalid(
            "parallel_ragged_inner",
            "ragged affine inner loops cannot be parallelized",
            ("parallel_loop",),
            index_id=base,
        )
    if base in facts.reduction_index_ids:
        _invalid(
            "parallel_reduction",
            "reduction loops cannot own independent result writes",
            ("parallel_loop",),
            index_id=base,
        )
    if base not in facts.free_index_ids:
        _invalid(
            "parallel_unclassified_loop",
            "parallel selection must identify a free result loop",
            ("parallel_loop",),
            index_id=base,
        )
    if not _all_dense_results(facts):
        _unsupported(
            "parallel_sparse_result",
            "explicit parallel selection currently requires dense results",
            ("parallel_loop",),
            index_id=base,
        )
    for layout in facts.result_accesses:
        if base not in layout.logical_index_ids:
            _invalid(
                "parallel_result_race",
                "the selected loop must partition every result write",
                ("parallel_loop",),
                index_id=base,
                access_id=layout.access_id,
            )

    affine_ids = {tile.loop.index_id for tile in plan.tiles if tile.kind == "affine"}
    if workspace_inserted and base not in affine_ids:
        last_reduction = max(
            facts.loop_positions[index_id] for index_id in facts.reduction_index_ids
        )
        if facts.loop_positions[base] > last_reduction:
            _invalid(
                "parallel_workspace_scope",
                "an untiled post-reduction loop has duplicated workspace binders",
                ("parallel_loop",),
                index_id=base,
            )
    if plan.provenance == "auto":
        _invalid(
            "auto_explicit_decision",
            "auto LoopPlans cannot contain an explicit parallel selection",
            ("provenance",),
            index_id=base,
        )

    definition = facts.analysis.index_definitions[base]
    binder_scope = definition.bindings[0].scope_id
    workspace_layouts = tuple(
        layout
        for layout in facts.analysis.access_layouts.values()
        if layout.is_workspace
        and layout.kind in (AccessKind.WRITE, AccessKind.REDUCTION_WRITE)
    )
    for layout in workspace_layouts:
        symbol = facts.analysis.symbol_definitions[layout.tensor_id]
        private_owner = _scope_contains(facts.analysis, binder_scope, symbol.scope_id)
        if not private_owner and base not in layout.logical_index_ids:
            _invalid(
                "parallel_workspace_race",
                "workspace writes are neither private nor partitioned",
                ("parallel_loop", "workspace"),
                index_id=base,
                access_id=layout.access_id,
            )
    return parallel


def _verify_panel(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
    state: _PlacementState,
    parallel: Optional[LoopRef],
) -> None:
    panel_tiles = tuple(tile for tile in plan.tiles if tile.kind == "panel")
    if len(panel_tiles) > 1:
        _unsupported(
            "multiple_panel_tiles",
            "the current sparse lowering supports one panel tile",
            ("tiles",),
        )
    if not panel_tiles:
        return
    panel = panel_tiles[0]
    panel_position = plan.tiles.index(panel)
    if panel_position != len(plan.tiles) - 1:
        _invalid(
            "panel_tile_order",
            "the panel tile must follow all affine decisions",
            ("tiles", str(panel_position)),
            index_id=panel.loop.index_id,
        )
    if panel.loop.part != LoopPart.LOGICAL:
        _invalid(
            "panel_logical_loop",
            "panel tiling must target a logical loop",
            ("tiles", str(panel_position)),
            index_id=panel.loop.index_id,
        )
    if panel.parallel or panel.accumulation != "direct":
        _unsupported(
            "panel_policy",
            "panel tiles require the canonical serial/direct panel policy",
            ("tiles", str(panel_position)),
            index_id=panel.loop.index_id,
        )
    if not _all_dense_results(facts):
        _unsupported(
            "panel_sparse_result",
            "panel tiling currently requires dense result storage",
            ("tiles", str(panel_position)),
        )
    if panel.placement.kind == PlacementKind.AT_DEPTH:
        _unsupported(
            "panel_depth_placement",
            "panel tiles do not support at_depth placement",
            ("tiles", str(panel_position), "placement"),
        )
    if panel.placement.kind == PlacementKind.CHILD_OF:
        parent = panel.placement.parent
        assert parent is not None
        parent_tile = next(
            (
                tile
                for tile in plan.tiles[:panel_position]
                if tile.kind == "affine" and tile.loop.index_id == parent.index_id
            ),
            None,
        )
        if (
            parent.part != LoopPart.OUTER
            or parent not in state.prefix
            or parent_tile is None
            or parent_tile.placement.kind != PlacementKind.OUTERMOST
        ):
            _invalid(
                "panel_parent_placement",
                "child panels require an earlier outermost affine outer loop",
                ("tiles", str(panel_position), "placement"),
                index_id=parent.index_id,
            )

    target = panel.loop.index_id
    compressed = tuple(
        layout
        for layout in facts.analysis.access_layouts.values()
        if not layout.is_workspace
        and target in layout.storage_index_ids
        and layout.level_types[layout.storage_index_ids.index(target)]
        == LevelType.COMPRESSED
    )
    if len(compressed) != 1:
        _unsupported(
            "panel_compressed_access",
            "panel target must have exactly one compressed tensor access",
            ("tiles", str(panel_position)),
            index_id=target,
        )
    sparse_layout = compressed[0]
    sparse_level = sparse_layout.storage_index_ids.index(target)
    if (
        sparse_level == 0
        or sparse_layout.level_types[sparse_level - 1] != LevelType.DENSE
    ):
        _unsupported(
            "panel_dense_parent",
            "panel lowering requires a compressed level with a dense parent",
            ("tiles", str(panel_position)),
            index_id=target,
            access_id=sparse_layout.access_id,
        )
    parent_id = sparse_layout.storage_index_ids[sparse_level - 1]
    if panel.placement.kind == PlacementKind.CHILD_OF:
        placement_parent = panel.placement.parent
        assert placement_parent is not None
        if placement_parent.index_id == parent_id:
            _invalid(
                "panel_parallel_scope",
                "a sparse panel must be placed outside its parallel dense-parent row",
                ("tiles", str(panel_position), "placement"),
                index_id=parent_id,
            )
    if parallel != LoopRef(parent_id):
        _invalid(
            "panel_parallel_parent",
            "the dense parent row must be the explicit parallel loop",
            ("parallel_loop",),
            index_id=parent_id,
        )

    bound = plan.panel_bounds[0]
    if bound.loop != panel.loop:
        _invalid(
            "panel_bound_loop",
            "panel bound must identify the exact logical panel loop",
            ("panel_bounds", "0"),
            index_id=target,
        )
    compatible_bounds = tuple(
        layout
        for layout in facts.read_accesses
        if not layout.is_workspace
        and layout.tensor_id == bound.tensor_id
        and bound.level < len(layout.storage_index_ids)
        and layout.storage_index_ids[bound.level] == target
        and layout.level_types[bound.level] == LevelType.DENSE
    )
    if not compatible_bounds:
        _invalid(
            "panel_bound_access",
            "panel bound must name a dense read level for the panel IndexId",
            ("panel_bounds", "0"),
            index_id=target,
            symbol_id=bound.tensor_id,
        )


def _unique_result(
    facts: LoopPlanLegalityFacts, path: Tuple[str, ...]
) -> AccessLayoutInfo:
    if len(facts.result_accesses) != 1:
        _unsupported(
            "unique_result_required",
            "this schedule requires exactly one non-workspace result write",
            path,
        )
    return facts.result_accesses[0]


def _assignment_for_lhs(
    facts: LoopPlanLegalityFacts,
    access_id: AccessId,
    path: Tuple[str, ...],
) -> AssignmentInfo:
    matches = tuple(
        assignment
        for assignment in facts.assignments
        if assignment.lhs_access_id == access_id
    )
    if len(matches) != 1:
        _unsupported(
            "unique_assignment_required",
            "schedule metadata must identify one result assignment",
            path,
            access_id=access_id,
        )
    return matches[0]


def _verify_result_tile(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
    state: _PlacementState,
    parallel: Optional[LoopRef],
) -> None:
    heap_tiles = tuple(
        tile
        for tile in plan.tiles
        if tile.kind == "affine" and tile.accumulation == "heap"
    )
    if len(heap_tiles) > 1:
        _unsupported(
            "multiple_heap_tiles",
            "heap accumulation supports exactly one result tile",
            ("tiles",),
        )
    if bool(heap_tiles) != (plan.result_tile is not None):
        _invalid(
            "heap_result_tile_pairing",
            "heap accumulation and ResultTile metadata must correspond exactly",
            ("result_tile",),
        )
    if not heap_tiles:
        return
    tile = heap_tiles[0]
    result_tile = plan.result_tile
    assert result_tile is not None
    if tile.placement.kind != PlacementKind.OUTERMOST or tile.parallel:
        _invalid(
            "heap_tile_lifetime",
            "the heap tile must be an outermost serial affine loop",
            ("tiles", str(plan.tiles.index(tile))),
            index_id=tile.loop.index_id,
        )
    tile_position = plan.tiles.index(tile)
    if any(
        later.kind == "panel" and later.placement.kind == PlacementKind.OUTERMOST
        for later in plan.tiles[tile_position + 1 :]
    ):
        _invalid(
            "heap_tile_outermost",
            "a later outermost panel cannot wrap the heap result lifetime",
            ("tiles", str(tile_position)),
            index_id=tile.loop.index_id,
        )
    heap_outer = LoopRef(tile.loop.index_id, LoopPart.OUTER)
    if not state.prefix or state.prefix[0] != heap_outer:
        _invalid(
            "heap_tile_outermost",
            "later root-level tiles cannot wrap the heap result lifetime",
            ("tiles", str(plan.tiles.index(tile))),
            index_id=tile.loop.index_id,
        )
    if parallel is None:
        _invalid(
            "heap_parallel_prefix",
            "heap accumulation requires an explicit parallel result prefix",
            ("parallel_loop",),
        )
    assert parallel is not None
    if parallel.index_id == tile.loop.index_id:
        _invalid(
            "heap_parallel_tile",
            "the shared heap tile loop must remain serial",
            ("parallel_loop",),
            index_id=parallel.index_id,
        )

    result = _unique_result(facts, ("result_tile",))
    definition = facts.analysis.symbol_definitions[result.tensor_id]
    if len(result.storage_index_ids) < 2 or any(
        level != LevelType.DENSE for level in result.level_types
    ):
        _unsupported(
            "heap_result_layout",
            "heap accumulation requires a rank-two-or-higher dense result",
            ("result_tile",),
            access_id=result.access_id,
        )
    if definition.dtype not in ("torch.float32", "torch.float64"):
        _unsupported(
            "heap_result_dtype",
            "heap accumulation supports float32 or float64 results",
            ("result_tile",),
            symbol_id=result.tensor_id,
        )
    expected_level = len(result.storage_index_ids) - 1
    expected_prefix = result.storage_index_ids[:-1]
    if (
        result_tile.result_id != result.tensor_id
        or result_tile.tile_loop != tile.loop
        or result_tile.result_level != expected_level
        or result_tile.result_prefix != expected_prefix
        or result_tile.access_indices != result.logical_index_ids
        or result.storage_index_ids[-1] != tile.loop.index_id
    ):
        _invalid(
            "result_tile_access",
            "ResultTile must exactly describe the trailing physical result tile",
            ("result_tile",),
            access_id=result.access_id,
        )
    assignment = _assignment_for_lhs(facts, result.access_id, ("result_tile",))
    if not assignment.reduction_index_ids or not _assignment_is_additive(assignment):
        _unsupported(
            "heap_additive_reduction",
            "heap accumulation requires one enclosed additive reduction",
            ("result_tile",),
            access_id=result.access_id,
        )
    if parallel.index_id not in expected_prefix:
        _invalid(
            "heap_parallel_ownership",
            "parallelism may select only a complete dense result-prefix axis",
            ("parallel_loop",),
            index_id=parallel.index_id,
        )


def _verify_relayout(
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
    parallel: Optional[LoopRef],
) -> None:
    relayout = plan.relayout
    if relayout is None:
        return
    _verify_relayout_shape(facts, plan, relayout, parallel)


def _verify_relayout_shape(  # noqa: C901
    facts: LoopPlanLegalityFacts,
    plan: LoopPlan,
    relayout: OperandRelayout,
    parallel: Optional[LoopRef],
) -> None:
    path = ("relayout",)
    for name, loop in (
        ("pack_loop", relayout.pack_loop),
        ("panel_loop", relayout.panel_loop),
        ("scope_loop", relayout.scope_loop),
        ("row_loop", relayout.row_loop),
    ):
        if loop.part != LoopPart.LOGICAL:
            _invalid(
                "relayout_logical_anchor",
                f"{name} must identify a logical loop",
                path + (name,),
                index_id=loop.index_id,
            )

    staged = tuple(
        layout
        for layout in facts.read_accesses
        if not layout.is_workspace and layout.tensor_id == relayout.operand_id
    )
    if len(staged) != 1:
        _invalid(
            "relayout_operand_access",
            "relayout operand must identify exactly one RHS access",
            path + ("operand_id",),
            symbol_id=relayout.operand_id,
        )
    staged_layout = staged[0]
    if staged_layout.logical_index_ids != relayout.access_indices:
        _invalid(
            "relayout_access_indices",
            "relayout access indices must exactly match the staged read",
            path + ("access_indices",),
            access_id=staged_layout.access_id,
        )
    if len(staged_layout.storage_index_ids) != 2 or any(
        level != LevelType.DENSE for level in staged_layout.level_types
    ):
        _unsupported(
            "relayout_operand_layout",
            "packed relayout requires one rank-two dense operand",
            path,
            access_id=staged_layout.access_id,
        )

    panel_id = relayout.panel_loop.index_id
    pack_id = relayout.pack_loop.index_id
    row_id = relayout.row_loop.index_id
    if staged_layout.storage_index_ids != (panel_id, pack_id):
        _invalid(
            "relayout_storage_axes",
            "staged storage must be ordered as panel then contiguous pack axis",
            path,
            access_id=staged_layout.access_id,
        )
    if relayout.operand_panel_level != 0 or relayout.operand_pack_level != 1:
        _invalid(
            "relayout_operand_levels",
            "relayout physical levels must identify the exact panel and pack axes",
            path,
            access_id=staged_layout.access_id,
        )
    if relayout.scope_loop.index_id not in (panel_id, pack_id):
        _invalid(
            "relayout_scope",
            "staging scope must be the panel or pack loop",
            path + ("scope_loop",),
            index_id=relayout.scope_loop.index_id,
        )

    affine_tiles = tuple(tile for tile in plan.tiles if tile.kind == "affine")
    panel_tiles = tuple(tile for tile in plan.tiles if tile.kind == "panel")
    pack_tiles = tuple(tile for tile in affine_tiles if tile.loop.index_id == pack_id)
    if len(plan.tiles) != 2 or len(pack_tiles) != 1 or len(panel_tiles) != 1:
        _unsupported(
            "relayout_tile_shape",
            "relayout supports one affine pack tile and one sparse panel tile",
            path,
        )
    pack_tile = pack_tiles[0]
    panel_tile = panel_tiles[0]
    if (
        panel_tile.loop.index_id != panel_id
        or pack_tile.width != relayout.strip_width
        or pack_tile.accumulation not in ("direct", "heap")
    ):
        _invalid(
            "relayout_tile_compatibility",
            "relayout loops, width, and result accumulation must match its tiles",
            path,
        )
    if (
        pack_tile.placement.kind != PlacementKind.OUTERMOST
        or panel_tile.placement.kind != PlacementKind.CHILD_OF
        or panel_tile.placement.parent != LoopRef(pack_id, LoopPart.OUTER)
    ):
        _invalid(
            "relayout_tile_placement",
            "relayout requires an outermost pack tile with a child panel",
            path,
        )

    containing = tuple(
        assignment
        for assignment in facts.assignments
        if staged_layout.access_id in assignment.rhs_access_ids
    )
    if len(facts.assignments) != 1 or len(containing) != 1:
        _unsupported(
            "relayout_assignment",
            "relayout requires one contraction assignment",
            path,
        )
    assignment = containing[0]
    if (
        not _assignment_is_additive(assignment)
        or assignment.multiplicative_access_ids is None
        or assignment.multiplicative_access_ids != assignment.rhs_access_ids
        or len(assignment.rhs_access_ids) != 2
    ):
        _unsupported(
            "relayout_expression",
            "both RHS accesses must form one additive multiplicative contraction",
            path,
        )
    counterpart_ids = tuple(
        access_id
        for access_id in assignment.rhs_access_ids
        if access_id != staged_layout.access_id
    )
    counterpart = facts.analysis.access_layouts[counterpart_ids[0]]
    if counterpart.level_types != (LevelType.DENSE, LevelType.COMPRESSED):
        _unsupported(
            "relayout_sparse_operand",
            "relayout requires one rank-two CSR operand with a dense parent",
            path,
            access_id=counterpart.access_id,
        )
    if counterpart.storage_index_ids != (row_id, panel_id):
        _invalid(
            "relayout_sparse_axes",
            "CSR storage must be ordered as row then panel coordinate",
            path,
            access_id=counterpart.access_id,
        )

    result = _unique_result(facts, path)
    if assignment.lhs_access_id != result.access_id or any(
        level != LevelType.DENSE for level in result.level_types
    ):
        _unsupported(
            "relayout_result",
            "relayout requires one dense contraction result",
            path,
            access_id=result.access_id,
        )
    if result.storage_index_ids != (row_id, pack_id):
        _invalid(
            "relayout_result_axes",
            "dense result storage must be ordered as row then pack axis",
            path,
            access_id=result.access_id,
        )
    if panel_id not in facts.reduction_index_ids or pack_id not in facts.free_index_ids:
        _unsupported(
            "relayout_index_roles",
            "panel must be a reduction and pack must be a free result axis",
            path,
        )
    if plan.loop_order != (row_id, panel_id, pack_id):
        _invalid(
            "relayout_loop_order",
            "relayout requires exact structural row, panel, pack loop order",
            ("loop_order",),
        )
    if parallel != LoopRef(row_id) or plan.parallel_loop != LoopRef(row_id):
        _invalid(
            "relayout_parallel_row",
            "relayout requires the logical CSR row as explicit parallel loop",
            ("parallel_loop",),
            index_id=row_id,
        )

    definitions = tuple(
        facts.analysis.symbol_definitions[layout.tensor_id]
        for layout in (staged_layout, counterpart, result)
    )
    dtypes = {definition.dtype for definition in definitions}
    if len(dtypes) != 1 or definitions[0].dtype not in (
        "torch.float32",
        "torch.float64",
    ):
        _unsupported(
            "relayout_dtype",
            "relayout supports matching float32 or float64 tensors",
            path,
        )


def verify_loop_plan_semantics(analysis: CINAnalysis, plan: LoopPlan) -> None:
    """Prove one structurally valid plan legal for its analyzed normalized CIN."""

    facts = _build_facts(analysis, plan)
    _verify_schedulable_scope(facts, plan)
    _verify_storage_order(facts)
    workspace_inserted = _verify_tiling_capabilities(facts, plan)
    _verify_workspace_replay_shape(facts, plan, workspace_inserted)
    state = _apply_affine_placements(facts, plan, workspace_inserted)
    parallel = _verify_parallel_legality(facts, plan, workspace_inserted)
    _verify_panel(facts, plan, state, parallel)
    _verify_result_tile(facts, plan, state, parallel)
    _verify_relayout(facts, plan, parallel)
