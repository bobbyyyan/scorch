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
  shape, the runtime input bindings, and the exact canonical CompileOptions
  state after replacing the public Schedule spelling with that verified plan.
  Cross-provenance rule: provenance is part of the plan payload, so requests
  whose plans differ only in provenance have different identities — a
  different provenance selects a different gate and replay contract.  The
  release source-derived cache is untouched; no artifact cache consumes this
  identity yet.

Collision handling: the canonical dump strings are the authoritative
comparands and are retained on the compiled artifact; the SHA-256 digest is
a content-addressed compact key over those exact bytes.  Digest equality is
therefore checkable against the retained dumps, and no truncated or salted
form is ever used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch

from ..cin import (
    BinaryOp,
    ForAll,
    IndexExpr,
    IndexStmt,
    IndexVar,
    IndexVarAdd,
    TensorAccess,
    TensorAssign,
    UnaryOp,
    Where,
)
from ..cin_analysis import canonical_cin_dump, verify_cin
from ..compile_options import (
    CompileOptions,
    DarwinToolchainOptions,
    KernelBuildOptions,
    SchedulerCostModel,
    SchedulerPolicy,
    VerificationPolicy,
)
from ..diagnostics import VerificationError
from ..identity import IndexId, SymbolId
from ..llir_pass_manager import LLIRPassOptions
from ..loop_plan import (
    LoopPlacement,
    LoopPlan,
    LoopRef,
    verify_loop_plan,
)

CANONICAL_PLAN_SCHEMA = "scorch.loopplan.canonical.v1"
CANONICAL_REQUEST_SCHEMA = "scorch.loopir.request.v2"
_MAX_RUNTIME_EXTENT = 2**63 - 1
_MAX_CANONICAL_CIN_DEPTH = 512
_DTYPE_TOKENS = {
    torch.float32: "float32",
    torch.float64: "float64",
}


def _verify_identity_cin(cin: object) -> IndexStmt:
    """Fail closed before recursively serializing one normalized CIN graph.

    The common CIN verifier diagnoses shared-node and semantic defects, but its
    compatibility preflight historically assumes the statement/expression
    graph is acyclic.  Request identity is a trust boundary even for an
    unscheduled request, so reject cycles and hostile depth first, then run the
    full verifier.  Convert malformed stored-state exceptions into the one
    compiler-owned boundary error rather than leaking ``RecursionError``,
    ``AttributeError``, or a caller-defined exception from serialization.
    """

    if not isinstance(cin, IndexStmt):
        raise VerificationError("loopir request identity expects an IndexStmt")

    active: set[int] = set()
    visited: set[int] = set()

    def visit(node: object, path: str, depth: int) -> None:
        if depth > _MAX_CANONICAL_CIN_DEPTH:
            raise VerificationError(
                "loopir request identity CIN exceeds the maximum structural depth"
            )
        node_key = id(node)
        if node_key in active:
            raise VerificationError(
                f"loopir request identity CIN contains a cycle at {path}"
            )
        if node_key in visited:
            return
        visited.add(node_key)
        active.add(node_key)
        try:
            if isinstance(node, ForAll):
                visit(node.index_var, f"{path}.index_var", depth + 1)
                visit(node.stmt, f"{path}.stmt", depth + 1)
            elif isinstance(node, Where):
                visit(node.producer, f"{path}.producer", depth + 1)
                visit(node.consumer, f"{path}.consumer", depth + 1)
            elif isinstance(node, TensorAssign):
                visit(node.lhs, f"{path}.lhs", depth + 1)
                visit(node.rhs, f"{path}.rhs", depth + 1)
            elif isinstance(node, BinaryOp):
                visit(node.left, f"{path}.left", depth + 1)
                visit(node.right, f"{path}.right", depth + 1)
            elif isinstance(node, UnaryOp):
                visit(node.expr, f"{path}.expr", depth + 1)
            elif isinstance(node, TensorAccess):
                indices = node.indices
                if not isinstance(indices, (tuple, list)):
                    raise VerificationError(
                        f"loopir request identity received malformed indices at {path}"
                    )
                for axis, index_var in enumerate(indices):
                    visit(index_var, f"{path}.indices[{axis}]", depth + 1)
            elif isinstance(node, IndexVar):
                index_expr = node._expr
                if index_expr is not None:
                    if type(index_expr) is not IndexVarAdd:
                        raise VerificationError(
                            "loopir request identity received an unsupported "
                            f"index expression at {path}.expr"
                        )
                    visit(index_expr, f"{path}.expr", depth + 1)
            elif isinstance(node, IndexVarAdd):
                visit(node.lhs, f"{path}.lhs", depth + 1)
                visit(node.rhs, f"{path}.rhs", depth + 1)
        finally:
            active.remove(node_key)

    try:
        visit(cin, "root", 0)
        verify_cin(cin)
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(
            "loopir request identity received malformed normalized CIN state"
        ) from exc
    return cin


def _rhs_binding_count(cin: IndexStmt) -> int:
    """Count ABI input occurrences without the legacy recursive visitor.

    The established ABI visits only the producer of a ``Where`` so its
    consumer-side workspace read does not become a runtime input.  Mirror that
    occurrence contract structurally while handling every verifier-supported
    expression kind, including ``UnaryOp``.
    """

    def count_expr(expr: IndexExpr) -> int:
        if isinstance(expr, TensorAccess):
            return 1
        if isinstance(expr, BinaryOp):
            return count_expr(expr.left) + count_expr(expr.right)
        if isinstance(expr, UnaryOp):
            return count_expr(expr.expr)
        raise VerificationError(
            "loopir request identity received an unsupported RHS expression"
        )

    def count_stmt(stmt: IndexStmt) -> int:
        if isinstance(stmt, ForAll):
            return count_stmt(stmt.stmt)
        if isinstance(stmt, Where):
            return count_stmt(stmt.producer)
        if isinstance(stmt, TensorAssign):
            return count_expr(stmt.rhs)
        raise VerificationError(
            "loopir request identity received an unsupported CIN statement"
        )

    return count_stmt(cin)


def _shape_payload(value: object, owner: str) -> List[int]:
    if type(value) not in (tuple, list):
        raise VerificationError(f"{owner} must be a tuple or list of exact ints")
    result: List[int] = []
    for extent in cast(Sequence[object], value):
        if (
            type(extent) is not int
            or isinstance(extent, bool)
            or extent < 0
            or extent > _MAX_RUNTIME_EXTENT
        ):
            raise VerificationError(f"{owner} must contain nonnegative int64 extents")
        result.append(extent)
    return result


def _dtype_token(value: object) -> str:
    if type(value) is not torch.dtype or value not in _DTYPE_TOKENS:
        raise VerificationError(
            "loopir request input dtypes must be exact supported torch.dtype values"
        )
    return _DTYPE_TOKENS[value]


def _compile_options_payload(options: object) -> object:
    """Canonical options state excluding the separately encoded plan request.

    ``CompileOptions.cache_key`` is already a typed tuple of exact canonical
    scalars.  Clear ``requested_schedule`` because its public spelling and tag
    are replaced at this boundary by the verified canonical LoopPlan; this
    also avoids the legacy Schedule ``repr`` cache key entering the new
    identity.  Every remaining semantic, verification, target, ABI, toolchain,
    and build option stays authoritative in the retained request dump.
    """

    def require_exact_carrier(
        value: object,
        expected_type: Any,
        owner: str,
    ) -> None:
        if type(value) is not expected_type:
            raise VerificationError(
                f"loopir request identity requires an exact {owner} snapshot"
            )
        expected_fields = {field.name for field in fields(cast(Any, expected_type))}
        if set(vars(value)) != expected_fields:
            raise VerificationError(
                f"loopir request identity {owner} has unexpected stored fields"
            )

    if type(options) is not CompileOptions:
        raise VerificationError(
            "loopir request identity requires an exact CompileOptions snapshot"
        )
    typed_options = cast(CompileOptions, options)
    require_exact_carrier(typed_options, CompileOptions, "CompileOptions")
    require_exact_carrier(
        typed_options.scheduler,
        SchedulerPolicy,
        "SchedulerPolicy",
    )
    require_exact_carrier(
        typed_options.scheduler.cost_model,
        SchedulerCostModel,
        "SchedulerCostModel",
    )
    require_exact_carrier(
        typed_options.verification,
        VerificationPolicy,
        "VerificationPolicy",
    )
    require_exact_carrier(
        typed_options.verification.llir_pass_options,
        LLIRPassOptions,
        "LLIRPassOptions",
    )
    require_exact_carrier(
        typed_options.build,
        KernelBuildOptions,
        "KernelBuildOptions",
    )
    if typed_options.build.darwin_toolchain is not None:
        require_exact_carrier(
            typed_options.build.darwin_toolchain,
            DarwinToolchainOptions,
            "DarwinToolchainOptions",
        )

    for owner, value in (
        ("CompileOptions.enabled_llir_passes", typed_options.enabled_llir_passes),
        ("KernelBuildOptions.extra_cflags", typed_options.build.extra_cflags),
        (
            "KernelBuildOptions.direct_extension_cflags",
            typed_options.build.direct_extension_cflags,
        ),
        (
            "KernelBuildOptions.special_kernel_cflags",
            typed_options.build.special_kernel_cflags,
        ),
        ("KernelBuildOptions.extra_ldflags", typed_options.build.extra_ldflags),
        (
            "KernelBuildOptions.torch_include_paths",
            typed_options.build.torch_include_paths,
        ),
    ):
        if type(value) is not tuple:
            raise VerificationError(
                f"loopir request identity {owner} must be stored as an exact tuple"
            )

    # Frozen carriers can still be forged through ``object.__setattr__``.
    # Re-run the pure stored-state validators without reconstructing their
    # constructor inputs: reconstruction would normalize/consume hostile
    # iterables and Darwin construction would improperly probe the host again.
    typed_options.scheduler.cost_model.__post_init__()
    typed_options.scheduler.__post_init__()
    typed_options.verification.__post_init__()
    if typed_options.build.darwin_toolchain is not None:
        typed_options.build.darwin_toolchain._validate_stored_policy()
    typed_options.build.__post_init__()
    typed_options.__post_init__()
    return replace(typed_options, requested_schedule=None).cache_key


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
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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
    *,
    compile_options: CompileOptions,
) -> str:
    """The canonical strangler request serialization at the compile boundary.

    ``plan=None`` is the explicit unscheduled request marker, distinct from
    every scheduled request.  The normalized CIN enters through its own
    canonical dump (traversal-canonical identities, no allocation-order
    data), and CompileOptions enter through their canonical typed cache tuple
    with ``requested_schedule`` cleared because the verified plan owns that
    decision here.  Byte equality therefore proves equality of the full
    strangler compilation requests.
    """

    verified_cin = _verify_identity_cin(cin)
    shape_payload = _shape_payload(result_shape, "loopir request result shape")
    if type(input_bindings) not in (tuple, list):
        raise VerificationError("loopir request input bindings must be a tuple or list")
    bindings_payload: List[object] = []
    for position, binding in enumerate(input_bindings):
        if type(binding) not in (tuple, list) or len(binding) != 2:
            raise VerificationError(
                f"loopir request input binding {position} must be a shape/dtype pair"
            )
        shape, dtype = binding
        bindings_payload.append(
            {
                "shape": _shape_payload(
                    shape, f"loopir request input binding {position} shape"
                ),
                "dtype": _dtype_token(dtype),
            }
        )
    if len(bindings_payload) != _rhs_binding_count(verified_cin):
        raise VerificationError(
            "loopir request input bindings must cover every declared input exactly"
        )
    try:
        cin_dump = canonical_cin_dump(verified_cin)
        plan_payload = (
            None
            if plan is None
            else _plan_payload(verified_cin, plan, provenance_free=False)
        )
        options_payload = _compile_options_payload(compile_options)
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(
            "loopir request identity could not serialize verified request state"
        ) from exc
    payload = {
        "schema": CANONICAL_REQUEST_SCHEMA,
        "cin": cin_dump,
        "plan": plan_payload,
        "result_shape": shape_payload,
        "inputs": bindings_payload,
        "compile_options": options_payload,
    }
    try:
        return _dump(payload)
    except Exception as exc:
        raise VerificationError(
            "loopir request identity could not encode the canonical request"
        ) from exc


def loopir_request_identity(
    cin: IndexStmt,
    plan: Optional[LoopPlan],
    result_shape: Sequence[int],
    input_bindings: Sequence[Tuple[Tuple[int, ...], object]],
    *,
    compile_options: CompileOptions,
) -> str:
    """The SHA-256 content key over the canonical request serialization."""

    dump = loopir_request_dump(
        cin,
        plan,
        result_shape,
        input_bindings,
        compile_options=compile_options,
    )
    return hashlib.sha256(dump.encode("utf-8")).hexdigest()
