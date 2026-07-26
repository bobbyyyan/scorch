"""Immutable scheduling artifacts at the normalized-CIN boundary.

The legacy scheduler still applies its transforms to a private CIN copy.  These
types make the decision crossing that seam explicit and identity based; they do
not introduce a new IR or own analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Tuple, cast

from .cin_analysis import CINAnalysis, analyze_cin
from .diagnostics import VerificationError
from .identity import IndexId, SymbolId

if TYPE_CHECKING:
    from .cin import IndexStmt


MAX_AFFINE_TILE_WIDTH = 2**31 - 1
"""Largest affine-tile width representable by the current C++ target.

The legacy and LoopIR CPU lowerings both materialize tile widths as
``constexpr int`` values.  Keeping this limit beside the shared LoopPlan
contract lets every schedule entry reject an unrepresentable width before
either lowering can silently wrap it.
"""


def _verify_cpp_int_width(width: int, field: str) -> None:
    """Reject a planning width the current C++ target cannot represent."""

    if width > MAX_AFFINE_TILE_WIDTH:
        raise VerificationError(
            f"LoopPlan {field} must fit the C++ constexpr int target"
        )


def _tuple_snapshot(value: object, field_name: str) -> Tuple[object, ...]:
    """Detach one tuple-valued plan field from caller-owned containers."""

    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{field_name} must be a tuple or list of typed plan values")
    return tuple(value)


class LoopPart(Enum):
    LOGICAL = "logical"
    OUTER = "outer"
    INNER = "inner"


@dataclass(frozen=True)
class LoopRef:
    index_id: IndexId
    part: LoopPart = LoopPart.LOGICAL


class PlacementKind(Enum):
    OUTERMOST = "outermost"
    CHILD_OF = "child_of"
    AT_DEPTH = "at_depth"


@dataclass(frozen=True)
class LoopPlacement:
    kind: PlacementKind
    parent: Optional[LoopRef] = None
    depth: Optional[int] = None


@dataclass(frozen=True)
class LoopTile:
    loop: LoopRef
    width: int
    placement: LoopPlacement
    parallel: bool
    kind: str
    accumulation: str
    unroll: bool


@dataclass(frozen=True)
class PanelBound:
    loop: LoopRef
    tensor_id: SymbolId
    level: int


@dataclass(frozen=True)
class OperandRelayout:
    operand_id: SymbolId
    pack_loop: LoopRef
    panel_loop: LoopRef
    scope_loop: LoopRef
    row_loop: LoopRef
    strip_width: int
    access_indices: Tuple[IndexId, ...]
    operand_panel_level: int
    operand_pack_level: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "access_indices",
            cast(
                Tuple[IndexId, ...],
                _tuple_snapshot(self.access_indices, "OperandRelayout.access_indices"),
            ),
        )


@dataclass(frozen=True)
class WorkspaceInsertion:
    """The automatic scheduler's standalone workspace decision, made explicit.

    The legacy auto scheduler inserts an accumulation workspace at the last
    reduction loop before applying its tiling heuristics; replay previously
    re-derived that hidden decision from scheduler policy instead of reading
    it from the plan.  This fact records the decision itself: the reduction
    loop the region wraps, the trailing free axis loops whose cells the
    workspace stores (in loop order), and whether the materialized workspace
    is the 1-D dense accumulator or the sparse fallback representation.  It
    records a workspace that replay must materialize; ``None`` in the legacy
    empty-Schedule replay contract may still represent its established
    root-workspace elision.  The plan-free origin API fails closed on that
    divergent root case instead of pretending the production insertion did
    not happen.  This is an automatic-provenance decision only; explicit
    schedules express workspace lifetime through tile accumulation instead.
    """

    reduction_loop: LoopRef
    axis_loops: Tuple[LoopRef, ...]
    dense: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "axis_loops",
            cast(
                Tuple[LoopRef, ...],
                _tuple_snapshot(self.axis_loops, "WorkspaceInsertion.axis_loops"),
            ),
        )


@dataclass(frozen=True)
class ResultTile:
    result_id: SymbolId
    tile_loop: LoopRef
    result_level: int
    result_prefix: Tuple[IndexId, ...]
    access_indices: Tuple[IndexId, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_prefix",
            cast(
                Tuple[IndexId, ...],
                _tuple_snapshot(self.result_prefix, "ResultTile.result_prefix"),
            ),
        )
        object.__setattr__(
            self,
            "access_indices",
            cast(
                Tuple[IndexId, ...],
                _tuple_snapshot(self.access_indices, "ResultTile.access_indices"),
            ),
        )


@dataclass(frozen=True)
class LoopPlan:
    """A verified scheduling decision expressed only with stable identities."""

    loop_order: Tuple[IndexId, ...]
    tiles: Tuple[LoopTile, ...] = ()
    panel_bounds: Tuple[PanelBound, ...] = ()
    relayout: Optional[OperandRelayout] = None
    result_tile: Optional[ResultTile] = None
    parallel_loop: Optional[LoopRef] = None
    workspace: Optional[WorkspaceInsertion] = None
    provenance: str = "explicit"
    tag: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "loop_order",
            cast(
                Tuple[IndexId, ...],
                _tuple_snapshot(self.loop_order, "LoopPlan.loop_order"),
            ),
        )
        object.__setattr__(
            self,
            "tiles",
            cast(Tuple[LoopTile, ...], _tuple_snapshot(self.tiles, "LoopPlan.tiles")),
        )
        object.__setattr__(
            self,
            "panel_bounds",
            cast(
                Tuple[PanelBound, ...],
                _tuple_snapshot(self.panel_bounds, "LoopPlan.panel_bounds"),
            ),
        )


@dataclass(frozen=True)
class ScheduledCIN:
    """Transitional pair of detached normalized CIN and its verified LoopPlan.

    The carrier and plan are frozen. The contained legacy CIN classes are not;
    consumers must treat them as read-only and use the compatibility adapter
    before any legacy mutation.
    """

    normalized_cin: IndexStmt
    verified_loop_plan: LoopPlan

    def __post_init__(self) -> None:
        from .cin import IndexStmt

        if not isinstance(self.normalized_cin, IndexStmt):
            raise TypeError("ScheduledCIN.normalized_cin must be an IndexStmt")
        if not isinstance(self.verified_loop_plan, LoopPlan):
            raise TypeError("ScheduledCIN.verified_loop_plan must be a LoopPlan")

    def __str__(self) -> str:
        return str(self.normalized_cin)


def _collect_entities(cin: object) -> Tuple[Dict[IndexId, str], Dict[SymbolId, str]]:
    from .cin import IndexStmt

    if not isinstance(cin, IndexStmt):
        raise VerificationError("normalized CIN must be an IndexStmt")
    analysis = analyze_cin(cin)
    reference_errors = tuple(
        diagnostic
        for diagnostic in analysis.diagnostics
        if diagnostic.code.startswith("duplicate_")
        or diagnostic.code.startswith("dangling_")
        or diagnostic.code.endswith("_out_of_scope")
        or diagnostic.code.endswith("_reference_mismatch")
        or diagnostic.code in ("free_index_not_bound", "missing_node_id")
    )
    if reference_errors:
        first = reference_errors[0]
        raise VerificationError(
            f"stage=normalized CIN: {first.code} at "
            f"{'/'.join(first.path)}: {first.message}",
            diagnostics=tuple(reference_errors),
        )
    indices = {
        index_id: definition.display_name
        for index_id, definition in analysis.index_definitions.items()
        if definition.bindings
    }
    symbols = {
        symbol_id: definition.display_name
        for symbol_id, definition in analysis.symbol_definitions.items()
    }
    return indices, symbols


def _analyze_loop_plan_cin(cin: object) -> CINAnalysis:
    """Return trustworthy canonical facts for semantic plan verification."""

    from .cin import IndexStmt

    if not isinstance(cin, IndexStmt):
        raise VerificationError("normalized CIN must be an IndexStmt")
    analysis = analyze_cin(cin)
    if analysis.diagnostics:
        first = analysis.diagnostics[0]
        raise VerificationError(
            f"stage=normalized CIN: {first.code} at "
            f"{'/'.join(first.path)}: {first.message}",
            diagnostics=cast(Tuple[object, ...], analysis.diagnostics),
        )
    return analysis


def entity_display_names(
    cin: object,
) -> Tuple[Dict[IndexId, str], Dict[SymbolId, str]]:
    """Return presentation-only name tables for the legacy lowering seam."""

    return _collect_entities(cin)


def _is_non_bool_int(value: object) -> bool:
    """Whether ``value`` is an exact stored integer plan primitive.

    Plan verification performs comparisons and identity lookups on these
    values.  Accepting an ``int`` subclass would let adversarial comparison
    or hashing methods escape the fail-closed boundary before the plan has
    been established as trustworthy.
    """

    return type(value) is int


def _validate_stored_fields(
    value: object, path: str, expected_fields: Tuple[str, ...]
) -> None:
    """Require one frozen plan carrier to own exactly its declared state.

    These dataclasses are not slotted, and fields with defaults also exist as
    class attributes.  A forged instance with a deleted field would therefore
    otherwise fall back to that class-level default and silently change the
    schedule being verified.
    """

    state = object.__getattribute__(value, "__dict__")
    if type(state) is not dict:
        raise VerificationError(f"LoopPlan {path} has invalid stored fields")
    keys = tuple(state.keys())
    if any(type(key) is not str for key in keys):
        raise VerificationError(f"LoopPlan {path} has invalid stored fields")
    if tuple(sorted(keys)) != tuple(sorted(expected_fields)):
        raise VerificationError(f"LoopPlan {path} has invalid stored fields")


def _diagnostic_int(value: int) -> str:
    """Render an exact int with deterministic, bounded diagnostic size."""

    if value.bit_length() > 4096:
        return "<integer too large to render>"
    try:
        return str(value)
    except ValueError:
        return "<integer too large to render>"


def _validate_index_id(index_id: object, path: str) -> IndexId:
    if type(index_id) is not IndexId:
        raise VerificationError(f"LoopPlan {path} must be a well-formed IndexId")
    _validate_stored_fields(index_id, path, ("value",))
    if not _is_non_bool_int(index_id.value):
        raise VerificationError(f"LoopPlan {path} must be a well-formed IndexId")
    return index_id


def _validate_symbol_id(symbol_id: object, path: str) -> SymbolId:
    if type(symbol_id) is not SymbolId:
        raise VerificationError(f"LoopPlan {path} must be a well-formed SymbolId")
    _validate_stored_fields(symbol_id, path, ("value",))
    if not _is_non_bool_int(symbol_id.value):
        raise VerificationError(f"LoopPlan {path} must be a well-formed SymbolId")
    return symbol_id


def _validate_loop_ref(loop: object, path: str) -> LoopRef:
    if type(loop) is not LoopRef:
        raise VerificationError(f"LoopPlan {path} must be a LoopRef")
    _validate_stored_fields(loop, path, ("index_id", "part"))
    typed_loop = cast(LoopRef, loop)
    _validate_index_id(typed_loop.index_id, f"{path}.index_id")
    if not any(
        typed_loop.part is expected
        for expected in (LoopPart.LOGICAL, LoopPart.OUTER, LoopPart.INNER)
    ):
        raise VerificationError(f"LoopPlan {path}.part must be a LoopPart")
    return typed_loop


def _validate_loop_placement(placement: object, path: str) -> LoopPlacement:
    if type(placement) is not LoopPlacement:
        raise VerificationError(f"LoopPlan {path} must be a LoopPlacement")
    _validate_stored_fields(placement, path, ("kind", "parent", "depth"))
    typed_placement = cast(LoopPlacement, placement)
    if not any(
        typed_placement.kind is expected
        for expected in (
            PlacementKind.OUTERMOST,
            PlacementKind.CHILD_OF,
            PlacementKind.AT_DEPTH,
        )
    ):
        raise VerificationError(f"LoopPlan {path}.kind must be a PlacementKind")
    if typed_placement.parent is not None:
        _validate_loop_ref(typed_placement.parent, f"{path}.parent")
    if typed_placement.depth is not None and not _is_non_bool_int(
        typed_placement.depth
    ):
        raise VerificationError(f"LoopPlan {path}.depth must be an integer or None")
    return typed_placement


def _validate_loop_plan_structure(plan: object) -> LoopPlan:
    """Reject malformed typed fields before semantic plan verification."""

    if type(plan) is not LoopPlan:
        raise VerificationError("LoopPlan verifier received a non-LoopPlan value")
    _validate_stored_fields(
        plan,
        "plan",
        (
            "loop_order",
            "tiles",
            "panel_bounds",
            "relayout",
            "result_tile",
            "parallel_loop",
            "workspace",
            "provenance",
            "tag",
        ),
    )
    typed_plan = cast(LoopPlan, plan)
    if type(typed_plan.loop_order) is not tuple:
        raise VerificationError("LoopPlan.loop_order must be a tuple")
    for position, index_id in enumerate(typed_plan.loop_order):
        _validate_index_id(index_id, f"loop_order[{position}]")
    if type(typed_plan.tiles) is not tuple:
        raise VerificationError("LoopPlan.tiles must be a tuple")
    for position, tile in enumerate(typed_plan.tiles):
        if type(tile) is not LoopTile:
            raise VerificationError(f"LoopPlan tiles[{position}] must be a LoopTile")
        _validate_stored_fields(
            tile,
            f"tiles[{position}]",
            (
                "loop",
                "width",
                "placement",
                "parallel",
                "kind",
                "accumulation",
                "unroll",
            ),
        )
        _validate_loop_ref(tile.loop, f"tiles[{position}].loop")
        _validate_loop_placement(tile.placement, f"tiles[{position}].placement")
        if not _is_non_bool_int(tile.width):
            raise VerificationError("LoopPlan tile widths must be integers")
        if type(tile.parallel) is not bool or type(tile.unroll) is not bool:
            raise VerificationError("LoopPlan tile policy flags must be bool values")
        if type(tile.kind) is not str or tile.kind not in ("affine", "panel"):
            raise VerificationError("LoopPlan tile kind must be 'affine' or 'panel'")
        if type(tile.accumulation) is not str or tile.accumulation not in (
            "stack",
            "direct",
            "heap",
        ):
            raise VerificationError(
                "LoopPlan tile accumulation must be 'stack', 'direct', or 'heap'"
            )
    if type(typed_plan.panel_bounds) is not tuple:
        raise VerificationError("LoopPlan.panel_bounds must be a tuple")
    for position, bound in enumerate(typed_plan.panel_bounds):
        if type(bound) is not PanelBound:
            raise VerificationError(
                f"LoopPlan panel_bounds[{position}] must be a PanelBound"
            )
        _validate_stored_fields(
            bound,
            f"panel_bounds[{position}]",
            ("loop", "tensor_id", "level"),
        )
        _validate_loop_ref(bound.loop, f"panel_bounds[{position}].loop")
        _validate_symbol_id(bound.tensor_id, f"panel_bounds[{position}].tensor_id")
        if not _is_non_bool_int(bound.level):
            raise VerificationError("LoopPlan panel-bound levels must be integers")
    if typed_plan.relayout is not None:
        if type(typed_plan.relayout) is not OperandRelayout:
            raise VerificationError(
                "LoopPlan.relayout must be an OperandRelayout or None"
            )
        relayout = typed_plan.relayout
        _validate_stored_fields(
            relayout,
            "relayout",
            (
                "operand_id",
                "pack_loop",
                "panel_loop",
                "scope_loop",
                "row_loop",
                "strip_width",
                "access_indices",
                "operand_panel_level",
                "operand_pack_level",
            ),
        )
        _validate_symbol_id(relayout.operand_id, "relayout.operand_id")
        for field, loop in (
            ("pack_loop", relayout.pack_loop),
            ("panel_loop", relayout.panel_loop),
            ("scope_loop", relayout.scope_loop),
            ("row_loop", relayout.row_loop),
        ):
            _validate_loop_ref(loop, f"relayout.{field}")
        if not _is_non_bool_int(relayout.strip_width):
            raise VerificationError("LoopPlan relayout strip width must be an integer")
        if type(relayout.access_indices) is not tuple:
            raise VerificationError("LoopPlan relayout access_indices must be a tuple")
        for position, index_id in enumerate(relayout.access_indices):
            _validate_index_id(index_id, f"relayout.access_indices[{position}]")
        if not _is_non_bool_int(relayout.operand_panel_level) or not _is_non_bool_int(
            relayout.operand_pack_level
        ):
            raise VerificationError("LoopPlan relayout levels must be integers")
    if typed_plan.result_tile is not None:
        if type(typed_plan.result_tile) is not ResultTile:
            raise VerificationError("LoopPlan.result_tile must be a ResultTile or None")
        result_tile = typed_plan.result_tile
        _validate_stored_fields(
            result_tile,
            "result_tile",
            (
                "result_id",
                "tile_loop",
                "result_level",
                "result_prefix",
                "access_indices",
            ),
        )
        _validate_symbol_id(result_tile.result_id, "result_tile.result_id")
        _validate_loop_ref(result_tile.tile_loop, "result_tile.tile_loop")
        if not _is_non_bool_int(result_tile.result_level):
            raise VerificationError("LoopPlan result-tile level must be an integer")
        for field, index_ids in (
            ("result_prefix", result_tile.result_prefix),
            ("access_indices", result_tile.access_indices),
        ):
            if type(index_ids) is not tuple:
                raise VerificationError(f"LoopPlan result {field} must be a tuple")
            for position, index_id in enumerate(index_ids):
                _validate_index_id(index_id, f"result_tile.{field}[{position}]")
    if typed_plan.parallel_loop is not None:
        _validate_loop_ref(typed_plan.parallel_loop, "parallel_loop")
    if typed_plan.workspace is not None:
        if type(typed_plan.workspace) is not WorkspaceInsertion:
            raise VerificationError(
                "LoopPlan.workspace must be a WorkspaceInsertion or None"
            )
        workspace = typed_plan.workspace
        _validate_stored_fields(
            workspace,
            "workspace",
            ("reduction_loop", "axis_loops", "dense"),
        )
        _validate_loop_ref(workspace.reduction_loop, "workspace.reduction_loop")
        if type(workspace.axis_loops) is not tuple or not workspace.axis_loops:
            raise VerificationError(
                "LoopPlan workspace.axis_loops must be a non-empty tuple"
            )
        for position, axis in enumerate(workspace.axis_loops):
            _validate_loop_ref(axis, f"workspace.axis_loops[{position}]")
        if type(workspace.dense) is not bool:
            raise VerificationError("LoopPlan workspace.dense must be a bool")
        if workspace.dense and len(workspace.axis_loops) != 1:
            raise VerificationError(
                "LoopPlan dense workspace insertion stores exactly one axis"
            )
    if type(typed_plan.provenance) is not str or not typed_plan.provenance:
        raise VerificationError("LoopPlan.provenance must be a non-empty string")
    if type(typed_plan.tag) is not str:
        raise VerificationError("LoopPlan.tag must be a string")
    return typed_plan


def _verify_empty_loop_plan(plan: LoopPlan) -> None:
    if (
        plan.loop_order
        or plan.tiles
        or plan.panel_bounds
        or plan.relayout is not None
        or plan.result_tile is not None
        or plan.parallel_loop is not None
        or plan.workspace is not None
    ):
        raise VerificationError(
            "LoopPlan references loops but normalized CIN defines none"
        )


def _verify_complete_loop_order(plan: LoopPlan, indices: Dict[IndexId, str]) -> None:
    if len(plan.loop_order) != len(set(plan.loop_order)):
        raise VerificationError("LoopPlan.loop_order contains duplicate IndexId values")
    for position, index_id in enumerate(plan.loop_order):
        if index_id not in indices:
            raise VerificationError(
                "LoopPlan loop_order"
                f"[{position}] references unknown IndexId "
                f"{_diagnostic_int(index_id.value)}"
            )
    missing_loop_ids = tuple(
        index_id for index_id in indices if index_id not in plan.loop_order
    )
    if missing_loop_ids:
        missing_values = ", ".join(
            _diagnostic_int(index_id.value) for index_id in missing_loop_ids
        )
        raise VerificationError(
            "LoopPlan.loop_order must contain every bound IndexId exactly once; "
            f"missing {missing_values}"
        )


def verify_loop_plan(cin: object, plan: LoopPlan) -> LoopPlan:
    """Verify structural and semantic legality at the scheduling boundary."""

    plan = _validate_loop_plan_structure(plan)
    analysis = _analyze_loop_plan_cin(cin)
    indices = {
        index_id: definition.display_name
        for index_id, definition in analysis.index_definitions.items()
        if definition.bindings
    }
    symbols = {
        symbol_id: definition.display_name
        for symbol_id, definition in analysis.symbol_definitions.items()
    }
    if not indices:
        _verify_empty_loop_plan(plan)
        from .loop_plan_legality import verify_loop_plan_semantics

        verify_loop_plan_semantics(analysis, plan)
        return plan

    def check_loop(loop: LoopRef, path: str) -> None:
        if loop.index_id not in indices:
            raise VerificationError(
                f"LoopPlan {path} references unknown IndexId "
                f"{_diagnostic_int(loop.index_id.value)}"
            )

    def check_symbol(symbol_id: SymbolId, path: str) -> None:
        if symbol_id not in symbols:
            raise VerificationError(
                f"LoopPlan {path} references unknown SymbolId "
                f"{_diagnostic_int(symbol_id.value)}"
            )

    _verify_complete_loop_order(plan, indices)

    tiled_ids = set()
    affine_tiled_ids = set()
    for position, tile in enumerate(plan.tiles):
        check_loop(tile.loop, f"tiles[{position}].loop")
        if tile.loop.part != LoopPart.LOGICAL:
            raise VerificationError("LoopPlan tiles must target logical loops")
        if tile.loop.index_id in tiled_ids:
            raise VerificationError("LoopPlan tiles the same logical loop twice")
        tiled_ids.add(tile.loop.index_id)
        if tile.kind == "affine":
            affine_tiled_ids.add(tile.loop.index_id)
        if tile.width <= 0:
            raise VerificationError("LoopPlan tile widths must be positive")
        _verify_cpp_int_width(tile.width, "tile widths")
        if tile.placement.kind == PlacementKind.CHILD_OF:
            if tile.placement.parent is None or tile.placement.depth is not None:
                raise VerificationError(
                    "LoopPlan child placement requires only a parent loop"
                )
        elif tile.placement.kind == PlacementKind.AT_DEPTH:
            if tile.placement.depth is None or tile.placement.depth < 0:
                raise VerificationError(
                    "LoopPlan depth placement requires a non-negative depth"
                )
            if tile.placement.parent is not None:
                raise VerificationError(
                    "LoopPlan depth placement cannot also name a parent"
                )
        elif tile.placement.parent is not None or tile.placement.depth is not None:
            raise VerificationError(
                "LoopPlan outermost placement cannot name a parent or depth"
            )
        if tile.placement.parent is not None:
            check_loop(tile.placement.parent, f"tiles[{position}].placement.parent")

    def check_derived_loop(loop: LoopRef, path: str) -> None:
        if loop.part != LoopPart.LOGICAL and loop.index_id not in affine_tiled_ids:
            raise VerificationError(
                f"LoopPlan {path} references a split part of a non-affine loop"
            )

    for position, tile in enumerate(plan.tiles):
        if tile.placement.parent is not None:
            check_derived_loop(
                tile.placement.parent, f"tiles[{position}].placement.parent"
            )

    panel_tile_ids = {tile.loop.index_id for tile in plan.tiles if tile.kind == "panel"}
    bound_ids = {bound.loop.index_id for bound in plan.panel_bounds}
    if len(bound_ids) != len(plan.panel_bounds):
        raise VerificationError("LoopPlan panel bounds must reference unique loops")
    if panel_tile_ids != bound_ids:
        raise VerificationError(
            "LoopPlan panel tiles and dense panel bounds must correspond exactly"
        )
    for position, bound in enumerate(plan.panel_bounds):
        check_loop(bound.loop, f"panel_bounds[{position}].loop")
        check_symbol(bound.tensor_id, f"panel_bounds[{position}].tensor_id")
        if bound.level < 0:
            raise VerificationError("LoopPlan panel-bound levels must be non-negative")

    if plan.parallel_loop is not None:
        check_loop(plan.parallel_loop, "parallel_loop")
        check_derived_loop(plan.parallel_loop, "parallel_loop")
    if plan.parallel_loop is not None and any(tile.parallel for tile in plan.tiles):
        raise VerificationError(
            "LoopPlan may select parallelism either on the plan or a tile, not both"
        )
    if sum(tile.parallel for tile in plan.tiles) > 1:
        raise VerificationError("LoopPlan may mark at most one tile parallel")

    if plan.relayout is not None:
        relayout = plan.relayout
        check_symbol(relayout.operand_id, "relayout.operand_id")
        for field, loop in (
            ("pack_loop", relayout.pack_loop),
            ("panel_loop", relayout.panel_loop),
            ("scope_loop", relayout.scope_loop),
            ("row_loop", relayout.row_loop),
        ):
            check_loop(loop, f"relayout.{field}")
            check_derived_loop(loop, f"relayout.{field}")
        if relayout.strip_width <= 0:
            raise VerificationError("LoopPlan relayout strip width must be positive")
        _verify_cpp_int_width(relayout.strip_width, "relayout strip width")
        if relayout.operand_panel_level < 0 or relayout.operand_pack_level < 0:
            raise VerificationError("LoopPlan relayout levels must be non-negative")
        for position, index_id in enumerate(relayout.access_indices):
            check_loop(LoopRef(index_id), f"relayout.access_indices[{position}]")

    if plan.result_tile is not None:
        result_tile = plan.result_tile
        check_symbol(result_tile.result_id, "result_tile.result_id")
        check_loop(result_tile.tile_loop, "result_tile.tile_loop")
        check_derived_loop(result_tile.tile_loop, "result_tile.tile_loop")
        if result_tile.result_level < 0:
            raise VerificationError("LoopPlan result-tile level must be non-negative")
        for field, index_ids in (
            ("result_prefix", result_tile.result_prefix),
            ("access_indices", result_tile.access_indices),
        ):
            for position, index_id in enumerate(index_ids):
                check_loop(LoopRef(index_id), f"result_tile.{field}[{position}]")

    from .loop_plan_legality import verify_loop_plan_semantics

    verify_loop_plan_semantics(analysis, plan)
    return plan


def verify_scheduled_cin(scheduled: ScheduledCIN) -> ScheduledCIN:
    """Verify the exact transitional scheduling carrier."""

    if type(scheduled) is not ScheduledCIN:
        raise VerificationError("ScheduledCIN verifier received the wrong artifact")
    _validate_stored_fields(
        scheduled,
        "scheduled carrier",
        ("normalized_cin", "verified_loop_plan"),
    )
    verify_loop_plan(scheduled.normalized_cin, scheduled.verified_loop_plan)
    return scheduled
