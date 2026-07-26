"""Canonical LoopPlan serialization and the strangler request identity.

Schedule identity is semantic content, never spelling: the canonical form of
a verified :class:`LoopPlan` is derived only from the decisions the plan
records — order, tiles and placements, accumulation, unroll, panel bounds,
relayout, result tile, parallel selection, workspace insertion, and
provenance.  Global allocation-order identities (``IndexId``/``SymbolId``
integers from the process-wide counters), Python ``hash()``, display names,
rendered C++, mutable scheduler state, and insertion history never enter the
serialization.  Plan-referenced identities are rewritten through an
artifact-local canonical numbering derived from the normalized CIN the plan
was verified against: loop indices by outer-to-inner nest binding order, and
tensor symbols by first appearance in a deterministic pre-order walk of the
assignments.  Two equivalent plans built by fresh builders therefore
serialize byte-identically, and every semantic decision changes the bytes.

``plan.tag`` is a presentation-only annotation (the public ``Schedule.tag``
label): it selects no gate, no pass, and no emission, so it is deliberately
outside the canonical form and outside the request identity.

Layering:

- :func:`canonical_plan_dump` — the versioned canonical serialization of one
  plan (provenance included: it selects the strangler gate and the replay
  contract).
- :func:`plan_schedule_digest` — a separate provenance-free digest of the
  schedule decisions alone, for callers comparing schedule content across
  provenances.  It is deliberately a distinct layer, never a substitute for
  request identity.
- :func:`loopir_request_identity` — the strangler-only canonical request
  identity at the compile/shadow request boundary: the canonical normalized
  CIN, the canonical plan (or the explicit unscheduled marker), the result
  shape, and the runtime input bindings.  Cross-provenance rule: provenance
  is part of the plan payload, so requests whose plans differ only in
  provenance have different identities — a different provenance selects a
  different gate and replay contract.  The release source-derived cache is
  untouched; no artifact cache consumes this identity yet.

Collision handling: the canonical dump strings are the authoritative
comparands and are retained on the compiled artifact; the SHA-256 digest is
a content-addressed compact key over those exact bytes.  Digest equality is
therefore checkable against the retained dumps, and no truncated or salted
form is ever used.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

from .cin import (
    BinaryOp,
    ForAll,
    IndexExpr,
    IndexStmt,
    TensorAccess,
    TensorAssign,
    UnaryOp,
    Where,
)
from .diagnostics import VerificationError
from .identity import IndexId, SymbolId
from .loop_plan import (
    LoopPlacement,
    LoopPlan,
    LoopRef,
    verify_loop_plan,
)

CANONICAL_PLAN_SCHEMA = "scorch.loopplan.canonical.v1"
CANONICAL_REQUEST_SCHEMA = "scorch.loopir.request.v1"


def _entity_maps(
    cin: IndexStmt,
) -> Tuple[Dict[IndexId, int], Dict[SymbolId, int]]:
    """Artifact-local canonical numbering for plan-referenced identities.

    Indices are numbered by outer-to-inner nest binding order — the
    plan-independent source order, so a reordering plan stays visible in its
    canonical form.  Symbols are numbered by first appearance in a pre-order
    walk of each statement (left-hand side first, then the right-hand side
    left to right).  The numbering depends only on the normalized CIN's
    structure, never on process-wide allocation order.
    """

    index_map: Dict[IndexId, int] = {}
    symbol_map: Dict[SymbolId, int] = {}

    def note_symbol(symbol: SymbolId) -> None:
        if symbol not in symbol_map:
            symbol_map[symbol] = len(symbol_map)

    def walk_expr(expr: IndexExpr) -> None:
        if isinstance(expr, TensorAccess):
            note_symbol(expr.tensor_id)
            return
        if isinstance(expr, BinaryOp):
            walk_expr(expr.left)
            walk_expr(expr.right)
            return
        if isinstance(expr, UnaryOp):
            walk_expr(expr.expr)
            return
        raise VerificationError(
            f"canonical plan identity cannot walk {type(expr).__name__}"
        )

    def walk_stmt(stmt: IndexStmt) -> None:
        if isinstance(stmt, ForAll):
            if stmt.index_var.index_id not in index_map:
                index_map[stmt.index_var.index_id] = len(index_map)
            walk_stmt(stmt.stmt)
            return
        if isinstance(stmt, Where):
            walk_stmt(stmt.producer)
            walk_stmt(stmt.consumer)
            return
        if isinstance(stmt, TensorAssign):
            walk_expr(stmt.lhs)
            walk_expr(stmt.rhs)
            return
        raise VerificationError(
            f"canonical plan identity cannot walk {type(stmt).__name__}"
        )

    walk_stmt(cin)
    return index_map, symbol_map


def _loop_ref_payload(ref: LoopRef, index_map: Dict[IndexId, int]) -> object:
    if ref.index_id not in index_map:
        raise VerificationError(
            "canonical plan identity received a loop reference outside the "
            "normalized CIN"
        )
    return {"index": index_map[ref.index_id], "part": ref.part.value}


def _placement_payload(
    placement: LoopPlacement, index_map: Dict[IndexId, int]
) -> object:
    return {
        "kind": placement.kind.value,
        "parent": (
            None
            if placement.parent is None
            else _loop_ref_payload(placement.parent, index_map)
        ),
        "depth": placement.depth,
    }


def _symbol_payload(symbol: SymbolId, symbol_map: Dict[SymbolId, int]) -> int:
    if symbol not in symbol_map:
        raise VerificationError(
            "canonical plan identity received a tensor reference outside the "
            "normalized CIN"
        )
    return symbol_map[symbol]


def _plan_payload(
    cin: IndexStmt,
    plan: LoopPlan,
    *,
    provenance_free: bool,
) -> object:
    verified = verify_loop_plan(cin, plan)
    index_map, symbol_map = _entity_maps(cin)

    def index_payload(index_id: IndexId) -> int:
        if index_id not in index_map:
            raise VerificationError(
                "canonical plan identity received an index outside the "
                "normalized CIN"
            )
        return index_map[index_id]

    payload: Dict[str, object] = {
        "schema": CANONICAL_PLAN_SCHEMA,
        "loop_order": [index_payload(index_id) for index_id in verified.loop_order],
        "tiles": [
            {
                "loop": _loop_ref_payload(tile.loop, index_map),
                "width": tile.width,
                "placement": _placement_payload(tile.placement, index_map),
                "parallel": tile.parallel,
                "kind": tile.kind,
                "accumulation": tile.accumulation,
                "unroll": tile.unroll,
            }
            for tile in verified.tiles
        ],
        "panel_bounds": [
            {
                "loop": _loop_ref_payload(bound.loop, index_map),
                "tensor": _symbol_payload(bound.tensor_id, symbol_map),
                "level": bound.level,
            }
            for bound in verified.panel_bounds
        ],
        "relayout": (
            None
            if verified.relayout is None
            else {
                "operand": _symbol_payload(verified.relayout.operand_id, symbol_map),
                "pack_loop": _loop_ref_payload(verified.relayout.pack_loop, index_map),
                "panel_loop": _loop_ref_payload(
                    verified.relayout.panel_loop, index_map
                ),
                "scope_loop": _loop_ref_payload(
                    verified.relayout.scope_loop, index_map
                ),
                "row_loop": _loop_ref_payload(verified.relayout.row_loop, index_map),
                "strip_width": verified.relayout.strip_width,
                "access_indices": [
                    index_payload(index_id)
                    for index_id in verified.relayout.access_indices
                ],
                "operand_panel_level": verified.relayout.operand_panel_level,
                "operand_pack_level": verified.relayout.operand_pack_level,
            }
        ),
        "result_tile": (
            None
            if verified.result_tile is None
            else {
                "result": _symbol_payload(verified.result_tile.result_id, symbol_map),
                "tile_loop": _loop_ref_payload(
                    verified.result_tile.tile_loop, index_map
                ),
                "result_level": verified.result_tile.result_level,
                "result_prefix": [
                    index_payload(index_id)
                    for index_id in verified.result_tile.result_prefix
                ],
                "access_indices": [
                    index_payload(index_id)
                    for index_id in verified.result_tile.access_indices
                ],
            }
        ),
        "parallel_loop": (
            None
            if verified.parallel_loop is None
            else _loop_ref_payload(verified.parallel_loop, index_map)
        ),
        "workspace": (
            None
            if verified.workspace is None
            else {
                "reduction_loop": _loop_ref_payload(
                    verified.workspace.reduction_loop, index_map
                ),
                "axis_loops": [
                    _loop_ref_payload(axis, index_map)
                    for axis in verified.workspace.axis_loops
                ],
                "dense": verified.workspace.dense,
            }
        ),
    }
    if not provenance_free:
        payload["provenance"] = verified.provenance
    return payload


def _dump(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def canonical_plan_dump(cin: IndexStmt, plan: LoopPlan) -> str:
    """The versioned canonical serialization of one verified plan."""

    return _dump(_plan_payload(cin, plan, provenance_free=False))


def plan_schedule_digest(cin: IndexStmt, plan: LoopPlan) -> str:
    """A provenance-free SHA-256 digest of the schedule decisions alone.

    A distinct comparison layer for schedule content across provenances —
    never a substitute for the request identity, which includes provenance
    because provenance selects the gate and replay contract.
    """

    payload = _dump(_plan_payload(cin, plan, provenance_free=True))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def loopir_request_dump(
    cin: IndexStmt,
    plan: Optional[LoopPlan],
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Tuple[int, ...], object]],
) -> str:
    """The canonical strangler request serialization at the compile boundary.

    ``plan=None`` is the explicit unscheduled request marker, distinct from
    every scheduled request.  The normalized CIN enters through its own
    canonical dump (traversal-canonical identities, no allocation-order
    data), so byte equality of two request dumps proves semantic equality of
    the requests.
    """

    from .cin_analysis import canonical_cin_dump

    if not isinstance(cin, IndexStmt):
        raise VerificationError("loopir_request_dump expects an IndexStmt")
    shape_payload: List[int] = []
    for extent in result_shape:
        if type(extent) is not int or isinstance(extent, bool):
            raise VerificationError(
                "loopir_request_dump result shape must contain exact ints"
            )
        shape_payload.append(extent)
    bindings_payload: List[object] = []
    for shape, dtype in input_bindings:
        entry_shape: List[int] = []
        for extent in shape:
            if type(extent) is not int or isinstance(extent, bool):
                raise VerificationError(
                    "loopir_request_dump input shapes must contain exact ints"
                )
            entry_shape.append(extent)
        bindings_payload.append({"shape": entry_shape, "dtype": str(dtype)})
    payload = {
        "schema": CANONICAL_REQUEST_SCHEMA,
        "cin": canonical_cin_dump(cin),
        "plan": (
            None if plan is None else _plan_payload(cin, plan, provenance_free=False)
        ),
        "result_shape": shape_payload,
        "inputs": bindings_payload,
    }
    return _dump(payload)


def loopir_request_identity(
    cin: IndexStmt,
    plan: Optional[LoopPlan],
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Tuple[int, ...], object]],
) -> str:
    """The SHA-256 content key over the canonical request serialization."""

    dump = loopir_request_dump(cin, plan, result_shape, input_bindings)
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()
