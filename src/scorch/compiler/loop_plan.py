"""Immutable scheduling artifacts at the normalized-CIN boundary.

The legacy scheduler still applies its transforms to a private CIN copy.  These
types make the decision crossing that seam explicit and identity based; they do
not introduce a new IR or own analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from .diagnostics import VerificationError
from .identity import IndexId, SymbolId

if TYPE_CHECKING:
    from .cin import IndexStmt


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


@dataclass(frozen=True)
class ResultTile:
    result_id: SymbolId
    tile_loop: LoopRef
    result_level: int
    result_prefix: Tuple[IndexId, ...]
    access_indices: Tuple[IndexId, ...]


@dataclass(frozen=True)
class LoopPlan:
    """A verified scheduling decision expressed only with stable identities."""

    loop_order: Tuple[IndexId, ...]
    tiles: Tuple[LoopTile, ...] = ()
    panel_bounds: Tuple[PanelBound, ...] = ()
    relayout: Optional[OperandRelayout] = None
    result_tile: Optional[ResultTile] = None
    parallel_loop: Optional[LoopRef] = None
    provenance: str = "explicit"
    tag: str = ""


@dataclass(frozen=True)
class ScheduledCIN:
    """Transitional pair of legacy normalized CIN and its verified LoopPlan."""

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
    # Import lazily so CIN does not depend on its transitional carrier.
    from .cin import BinaryOp, ForAll, TensorAccess, TensorAssign, UnaryOp, Where

    indices: Dict[IndexId, str] = {}
    symbols: Dict[SymbolId, str] = {}

    def visit(node: object) -> None:
        if isinstance(node, ForAll):
            indices[node.index_var.index_id] = node.index_var.name
            visit(node.stmt)
        elif isinstance(node, Where):
            visit(node.producer)
            visit(node.consumer)
        elif isinstance(node, TensorAssign):
            visit(node.lhs)
            visit(node.rhs)
        elif isinstance(node, TensorAccess):
            symbols[node.tensor.symbol_id] = node.tensor.name
            for index_var in node.indices or ():
                indices[index_var.index_id] = index_var.name
        elif isinstance(node, BinaryOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, UnaryOp):
            visit(node.expr)

    visit(cin)
    return indices, symbols


def entity_display_names(
    cin: object,
) -> Tuple[Dict[IndexId, str], Dict[SymbolId, str]]:
    """Return presentation-only name tables for the legacy lowering seam."""

    return _collect_entities(cin)


def verify_loop_plan(cin: object, plan: LoopPlan) -> LoopPlan:
    """Verify stable symbol and loop references at the scheduling boundary."""

    if not isinstance(plan, LoopPlan):
        raise VerificationError("LoopPlan verifier received a non-LoopPlan value")

    indices, symbols = _collect_entities(cin)
    if not indices:
        if plan.loop_order or plan.tiles or plan.parallel_loop is not None:
            raise VerificationError(
                "LoopPlan references loops but normalized CIN defines none"
            )
        return plan

    def check_loop(loop: LoopRef, path: str) -> None:
        if loop.index_id not in indices:
            raise VerificationError(
                f"LoopPlan {path} references unknown IndexId {loop.index_id.value}"
            )

    def check_symbol(symbol_id: SymbolId, path: str) -> None:
        if symbol_id not in symbols:
            raise VerificationError(
                f"LoopPlan {path} references unknown SymbolId {symbol_id.value}"
            )

    if len(plan.loop_order) != len(set(plan.loop_order)):
        raise VerificationError("LoopPlan.loop_order contains duplicate IndexId values")
    for position, index_id in enumerate(plan.loop_order):
        check_loop(LoopRef(index_id), f"loop_order[{position}]")

    tiled_ids = set()
    for position, tile in enumerate(plan.tiles):
        check_loop(tile.loop, f"tiles[{position}].loop")
        if tile.loop.part != LoopPart.LOGICAL:
            raise VerificationError("LoopPlan tiles must target logical loops")
        if tile.loop.index_id in tiled_ids:
            raise VerificationError("LoopPlan tiles the same logical loop twice")
        tiled_ids.add(tile.loop.index_id)
        if tile.width <= 0:
            raise VerificationError("LoopPlan tile widths must be positive")
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
        if loop.part != LoopPart.LOGICAL and loop.index_id not in tiled_ids:
            raise VerificationError(
                f"LoopPlan {path} references a split part of an untiled loop"
            )

    for position, tile in enumerate(plan.tiles):
        if tile.placement.parent is not None:
            check_derived_loop(
                tile.placement.parent, f"tiles[{position}].placement.parent"
            )

    panel_tile_ids = {tile.loop.index_id for tile in plan.tiles if tile.kind == "panel"}
    bound_ids = {bound.loop.index_id for bound in plan.panel_bounds}
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

    return plan


def verify_scheduled_cin(scheduled: ScheduledCIN) -> ScheduledCIN:
    """Verify the exact transitional scheduling carrier."""

    if not isinstance(scheduled, ScheduledCIN):
        raise VerificationError("ScheduledCIN verifier received the wrong artifact")
    verify_loop_plan(scheduled.normalized_cin, scheduled.verified_loop_plan)
    return scheduled
