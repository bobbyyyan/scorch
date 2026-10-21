"""Immutable scheduling artifacts at the normalized-CIN boundary.

The legacy scheduler still applies its transforms to a private CIN copy.  These
types make the decision crossing that seam explicit and identity based; they do
not introduce a new IR or own analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, NoReturn, Optional, Set, Tuple, cast

from ..format import LevelType, parse_format
from .cin_analysis import CINAnalysis, analyze_cin
from .diagnostics import VerificationError
from .identity import IndexId, SymbolId
from .sparse_accumulator import is_accumulator_structure
from .sparse_assembly import is_assembly_strategy
from ..exceptions import ScorchTypeError

if TYPE_CHECKING:
    from .cin import IndexStmt, TensorVar


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
            f"LoopPlan {field} must fit the C++ constexpr int target",
            code="loop_plan_reference_invalid",
        )


def _tuple_snapshot(value: object, field_name: str) -> Tuple[object, ...]:
    """Detach one tuple-valued plan field from caller-owned containers."""

    if not isinstance(value, (tuple, list)):
        raise ScorchTypeError(
            f"{field_name} must be a tuple or list of typed plan values",
            code="loop_plan_artifact_invalid",
        )
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
class OperandConversion:
    """Convert one operand to another level order before the main kernel runs.

    ``mode_order`` is the converted tensor's mode order (storage level ->
    logical mode) and ``format`` its layout, one ``"d"`` (dense) or ``"s"``
    (compressed) per storage level.  The conversion is a copy of every entry
    the operand stores, whatever its value, and the converted tensor stores
    exactly those entries, so the kernel that reads it produces the stored
    coordinates the operand's own entries imply.  A dense level below a
    compressed one stores every cell under each stored parent, which after a
    permutation would be cells the operand does not hold; so ``format`` has
    its dense levels, if any, above every compressed one.  A CSR operand
    (``"ds"``) converted to mode order ``(1, 0)`` is CSC (``"ds"``); an
    operand stored ``"sd"`` converts to ``"ss"``.

    The loop order determines which operands a plan converts: every operand
    with a compressed level whose storage order the loops do not follow, into
    the loop order restricted to its own indices.
    """

    operand_id: SymbolId
    mode_order: Tuple[int, ...]
    format: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mode_order",
            cast(
                Tuple[int, ...],
                _tuple_snapshot(self.mode_order, "OperandConversion.mode_order"),
            ),
        )


@dataclass(frozen=True)
class WorkspaceInsertion:
    """The automatic scheduler's standalone workspace decision, made explicit.

    The legacy auto scheduler inserts an accumulation workspace before
    applying its tiling heuristics; replay previously re-derived that hidden
    decision from scheduler policy instead of reading it from the plan.  This
    fact records the decision itself: the innermost reduction loop the region
    contains, the free axis loops inside the region whose cells the workspace
    stores (in loop order), and whether the materialized workspace is the 1-D
    dense accumulator or the sparse fallback representation.  Where the region
    begins is derived, not recorded: for a dense result it is that reduction
    loop; for a sparse result it is the first loop after the longest loop-order
    prefix that walks the result's levels in storage order, so every reduction
    loop is inside the region and each result coordinate is drained once
    (``Scheduler._sparse_result_workspace_loop``).  With one reduction loop the
    two coincide.  It
    records a workspace that replay must materialize; ``None`` records the
    elision contract, including the dense root-scope reduction family where
    both the plan-free production surgery and the empty-Schedule replay now
    elide the insertion (the abandoned materialized form emitted both
    workspace APIs over one undeclared symbol and never compiled).  This is
    an automatic-provenance decision only; explicit schedules express
    workspace lifetime through tile accumulation instead.
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


AUTO_ORIGIN_POLICY_SCHEMA = "scorch.autopolicy.v1"


@dataclass(frozen=True)
class AutoOriginPolicy:
    """The typed automatic-scheduling policy inputs one recorded plan claims.

    ``provenance="auto"`` alone never attests cost-model origin: an arbitrary
    plan can carry the string.  This versioned fact records the exact policy
    inputs the originating scheduler consumed — which regblock arm ran and
    the tile width that arm's heuristic uses — so the legality boundary can
    re-derive every heuristic tile and workspace decision from the analyzed
    CIN, the plan order, and this policy, and require the stored facts to
    equal that derivation exactly.  The cost-model loop order itself remains
    an attested decision (verified complete and legal, not re-derived); the
    admitted automatic families are still replay contracts, not proofs that
    the current cost model chose the order.

    The policy is verification state, not schedule content: two verified
    plans with identical decisions produce identical canonical serializations
    and request identities regardless of the recorded policy, exactly as the
    presentation-only ``tag`` stays outside the identity.
    """

    schema: str
    regblock_enabled: bool
    tile_width: int


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
    """A verified scheduling decision expressed only with stable identities.

    ``assembly`` records which sparse-output assembly strategy the result is
    built with -- one traversal or two, distributed across workers or not.  The
    right choice depends on the operands' formats and density, so it is a
    scheduling decision rather than a property of the lowering class that hosts
    the program.  ``None`` records *no* strategy decision and defers to the
    target's own per-receiver choice; it is not a synonym for
    ``single_pass_serial``.  The token names assembly structure only: thread
    counts, chunk widths, pragma spelling and the runtime gate are target
    decisions and never appear in a plan.

    ``accumulator`` records which structure a sparse reduction accumulates
    through -- a list of the keys inserted, or a chain threaded through an array
    indexed by the key.  Which one is faster depends on the density and on the
    receiver's compressed extent, so it is a scheduling decision rather than
    something an assembly transform may substitute on the way past.  ``None``
    records *no* structure decision and leaves every existing layer choosing what
    it chooses today; it is not a synonym for ``coordinate_list``.  This field is
    distinct from ``workspace``, which records WHERE the automatic scheduler
    inserted an accumulation workspace rather than WHICH structure holds it.

    ``conversions`` records the operands converted to another level order
    before the main kernel runs (:class:`OperandConversion`); every other
    decision of the plan is verified against the program with those operands
    converted.  It is empty on every plan the legacy scheduler originates.

    ``vector_loops`` records which loops keep a vector mark
    (``#pragma omp simd``).  ``None`` records no decision: each compiler places
    the marks it placed before this field existed.  Both apply the shared rule
    ``iter_lattice.mark_simd_on_reduction_loops`` in one place, the loops below
    a sparse result entry whose sum is kept in a register (a loop is safe when
    its body holds no loop over a compressed level's positions, it is not
    parallel or unrolled, and it has no early exit); with no decision the
    shipped compiler marks the safe loops there and the new compiler marks
    none.  A tuple, the empty one included, keeps the mark on exactly the named
    loops, in both compilers; naming a loop the rule does not find safe is
    refused when the kernel is lowered (``unsupported_vector_loop``), since
    only there is the loop's body known.  Like ``assembly`` it is independent
    of the loop order, so an automatic plan may carry it.

    ``entries_of`` records the input whose stored entries a COO result stores:
    one result entry per stored entry of that input, in its storage order, an
    entry whose inner sum ran zero times included.  Plan legality admits it
    only where the loops over the result's indices walk that input's stored
    entries with a reduction below them (``sparse_assembly.entry_source``);
    each entry's position is then that input's, known before the loops run,
    and the parallel single pass of a COO result needs it.  ``None`` records no
    decision: the shipped compiler stores every entry of the input its scalar
    accumulation walks, and the new compiler appends only the entries some
    term reached.  An automatic plan may carry it, like ``assembly``.
    """

    loop_order: Tuple[IndexId, ...]
    tiles: Tuple[LoopTile, ...] = ()
    panel_bounds: Tuple[PanelBound, ...] = ()
    relayout: Optional[OperandRelayout] = None
    result_tile: Optional[ResultTile] = None
    parallel_loop: Optional[LoopRef] = None
    workspace: Optional[WorkspaceInsertion] = None
    assembly: Optional[str] = None
    accumulator: Optional[str] = None
    auto_policy: Optional[AutoOriginPolicy] = None
    provenance: str = "explicit"
    tag: str = ""
    conversions: Tuple[OperandConversion, ...] = ()
    vector_loops: Optional[Tuple[LoopRef, ...]] = None
    entries_of: Optional[SymbolId] = None

    def __post_init__(self) -> None:
        if self.vector_loops is not None:
            object.__setattr__(
                self,
                "vector_loops",
                cast(
                    Tuple[LoopRef, ...],
                    _tuple_snapshot(self.vector_loops, "LoopPlan.vector_loops"),
                ),
            )
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
            "conversions",
            cast(
                Tuple[OperandConversion, ...],
                _tuple_snapshot(self.conversions, "LoopPlan.conversions"),
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
        from .cin import _is_exact_index_stmt

        if not _is_exact_index_stmt(self.normalized_cin):
            raise ScorchTypeError(
                "ScheduledCIN.normalized_cin must be an IndexStmt",
                code="loop_plan_artifact_invalid",
            )
        if not isinstance(self.verified_loop_plan, LoopPlan):
            raise ScorchTypeError(
                "ScheduledCIN.verified_loop_plan must be a LoopPlan",
                code="loop_plan_artifact_invalid",
            )

    def __str__(self) -> str:
        return str(self.normalized_cin)


def _collect_entities(cin: object) -> Tuple[Dict[IndexId, str], Dict[SymbolId, str]]:
    from .cin import _is_index_stmt_instance

    if not _is_index_stmt_instance(cin):
        raise VerificationError(
            "normalized CIN must be an IndexStmt", code="loop_plan_artifact_invalid"
        )
    analysis = analyze_cin(cast("IndexStmt", cin))
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

    from .cin import _is_index_stmt_instance

    if not _is_index_stmt_instance(cin):
        raise VerificationError(
            "normalized CIN must be an IndexStmt", code="loop_plan_artifact_invalid"
        )
    analysis = analyze_cin(cast("IndexStmt", cin))
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
        raise VerificationError(
            f"LoopPlan {path} has invalid stored fields",
            code="loop_plan_structure_invalid",
        )
    keys = tuple(state.keys())
    if any(type(key) is not str for key in keys):
        raise VerificationError(
            f"LoopPlan {path} has invalid stored fields",
            code="loop_plan_structure_invalid",
        )
    if tuple(sorted(keys)) != tuple(sorted(expected_fields)):
        raise VerificationError(
            f"LoopPlan {path} has invalid stored fields",
            code="loop_plan_structure_invalid",
        )


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
        raise VerificationError(
            f"LoopPlan {path} must be a well-formed IndexId",
            code="loop_plan_structure_invalid",
        )
    _validate_stored_fields(index_id, path, ("value",))
    if not _is_non_bool_int(index_id.value):
        raise VerificationError(
            f"LoopPlan {path} must be a well-formed IndexId",
            code="loop_plan_structure_invalid",
        )
    return index_id


def _validate_symbol_id(symbol_id: object, path: str) -> SymbolId:
    if type(symbol_id) is not SymbolId:
        raise VerificationError(
            f"LoopPlan {path} must be a well-formed SymbolId",
            code="loop_plan_structure_invalid",
        )
    _validate_stored_fields(symbol_id, path, ("value",))
    if not _is_non_bool_int(symbol_id.value):
        raise VerificationError(
            f"LoopPlan {path} must be a well-formed SymbolId",
            code="loop_plan_structure_invalid",
        )
    return symbol_id


def _validate_loop_ref(loop: object, path: str) -> LoopRef:
    if type(loop) is not LoopRef:
        raise VerificationError(
            f"LoopPlan {path} must be a LoopRef", code="loop_plan_structure_invalid"
        )
    _validate_stored_fields(loop, path, ("index_id", "part"))
    typed_loop = cast(LoopRef, loop)
    _validate_index_id(typed_loop.index_id, f"{path}.index_id")
    if not any(
        typed_loop.part is expected
        for expected in (LoopPart.LOGICAL, LoopPart.OUTER, LoopPart.INNER)
    ):
        raise VerificationError(
            f"LoopPlan {path}.part must be a LoopPart",
            code="loop_plan_structure_invalid",
        )
    return typed_loop


def _validate_loop_placement(placement: object, path: str) -> LoopPlacement:
    if type(placement) is not LoopPlacement:
        raise VerificationError(
            f"LoopPlan {path} must be a LoopPlacement",
            code="loop_plan_structure_invalid",
        )
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
        raise VerificationError(
            f"LoopPlan {path}.kind must be a PlacementKind",
            code="loop_plan_structure_invalid",
        )
    if typed_placement.parent is not None:
        _validate_loop_ref(typed_placement.parent, f"{path}.parent")
    if typed_placement.depth is not None and not _is_non_bool_int(
        typed_placement.depth
    ):
        raise VerificationError(
            f"LoopPlan {path}.depth must be an integer or None",
            code="loop_plan_structure_invalid",
        )
    return typed_placement


def _validate_output_decision_tokens(plan: LoopPlan) -> None:
    """Reject a forged token in either sparse-output decision.

    Both fields are plain strings on the plan, so a value that is not one of the
    recorded tokens would otherwise reach a layer that compares it against enum
    members and silently matches nothing.  Each token set has exactly one
    definition, imported here rather than restated.
    """

    if plan.assembly is not None and not is_assembly_strategy(plan.assembly):
        raise VerificationError(
            "LoopPlan.assembly must be a recorded sparse assembly strategy or None",
            code="loop_plan_structure_invalid",
        )
    if plan.accumulator is not None and not is_accumulator_structure(plan.accumulator):
        raise VerificationError(
            "LoopPlan.accumulator must be a recorded sparse accumulation "
            "structure or None",
            code="loop_plan_structure_invalid",
        )


def _validate_loop_plan_structure(plan: object) -> LoopPlan:
    """Reject malformed typed fields before semantic plan verification."""

    if type(plan) is not LoopPlan:
        raise VerificationError(
            "LoopPlan verifier received a non-LoopPlan value",
            code="loop_plan_structure_invalid",
        )
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
            "assembly",
            "accumulator",
            "auto_policy",
            "provenance",
            "tag",
            "conversions",
            "vector_loops",
            "entries_of",
        ),
    )
    typed_plan = cast(LoopPlan, plan)
    if type(typed_plan.loop_order) is not tuple:
        raise VerificationError(
            "LoopPlan.loop_order must be a tuple", code="loop_plan_structure_invalid"
        )
    for position, index_id in enumerate(typed_plan.loop_order):
        _validate_index_id(index_id, f"loop_order[{position}]")
    if type(typed_plan.tiles) is not tuple:
        raise VerificationError(
            "LoopPlan.tiles must be a tuple", code="loop_plan_structure_invalid"
        )
    for position, tile in enumerate(typed_plan.tiles):
        if type(tile) is not LoopTile:
            raise VerificationError(
                f"LoopPlan tiles[{position}] must be a LoopTile",
                code="loop_plan_structure_invalid",
            )
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
            raise VerificationError(
                "LoopPlan tile widths must be integers",
                code="loop_plan_structure_invalid",
            )
        if type(tile.parallel) is not bool or type(tile.unroll) is not bool:
            raise VerificationError(
                "LoopPlan tile policy flags must be bool values",
                code="loop_plan_structure_invalid",
            )
        if type(tile.kind) is not str or tile.kind not in ("affine", "panel"):
            raise VerificationError(
                "LoopPlan tile kind must be 'affine' or 'panel'",
                code="loop_plan_structure_invalid",
            )
        if type(tile.accumulation) is not str or tile.accumulation not in (
            "stack",
            "direct",
            "heap",
        ):
            raise VerificationError(
                "LoopPlan tile accumulation must be 'stack', 'direct', or 'heap'",
                code="loop_plan_structure_invalid",
            )
    if type(typed_plan.panel_bounds) is not tuple:
        raise VerificationError(
            "LoopPlan.panel_bounds must be a tuple", code="loop_plan_structure_invalid"
        )
    for position, bound in enumerate(typed_plan.panel_bounds):
        if type(bound) is not PanelBound:
            raise VerificationError(
                f"LoopPlan panel_bounds[{position}] must be a PanelBound",
                code="loop_plan_structure_invalid",
            )
        _validate_stored_fields(
            bound,
            f"panel_bounds[{position}]",
            ("loop", "tensor_id", "level"),
        )
        _validate_loop_ref(bound.loop, f"panel_bounds[{position}].loop")
        _validate_symbol_id(bound.tensor_id, f"panel_bounds[{position}].tensor_id")
        if not _is_non_bool_int(bound.level):
            raise VerificationError(
                "LoopPlan panel-bound levels must be integers",
                code="loop_plan_structure_invalid",
            )
    if typed_plan.relayout is not None:
        if type(typed_plan.relayout) is not OperandRelayout:
            raise VerificationError(
                "LoopPlan.relayout must be an OperandRelayout or None",
                code="loop_plan_structure_invalid",
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
            raise VerificationError(
                "LoopPlan relayout strip width must be an integer",
                code="loop_plan_structure_invalid",
            )
        if type(relayout.access_indices) is not tuple:
            raise VerificationError(
                "LoopPlan relayout access_indices must be a tuple",
                code="loop_plan_structure_invalid",
            )
        for position, index_id in enumerate(relayout.access_indices):
            _validate_index_id(index_id, f"relayout.access_indices[{position}]")
        if not _is_non_bool_int(relayout.operand_panel_level) or not _is_non_bool_int(
            relayout.operand_pack_level
        ):
            raise VerificationError(
                "LoopPlan relayout levels must be integers",
                code="loop_plan_structure_invalid",
            )
    if typed_plan.result_tile is not None:
        if type(typed_plan.result_tile) is not ResultTile:
            raise VerificationError(
                "LoopPlan.result_tile must be a ResultTile or None",
                code="loop_plan_structure_invalid",
            )
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
            raise VerificationError(
                "LoopPlan result-tile level must be an integer",
                code="loop_plan_structure_invalid",
            )
        for field, index_ids in (
            ("result_prefix", result_tile.result_prefix),
            ("access_indices", result_tile.access_indices),
        ):
            if type(index_ids) is not tuple:
                raise VerificationError(
                    f"LoopPlan result {field} must be a tuple",
                    code="loop_plan_structure_invalid",
                )
            for position, index_id in enumerate(index_ids):
                _validate_index_id(index_id, f"result_tile.{field}[{position}]")
    if typed_plan.parallel_loop is not None:
        _validate_loop_ref(typed_plan.parallel_loop, "parallel_loop")
    if typed_plan.workspace is not None:
        if type(typed_plan.workspace) is not WorkspaceInsertion:
            raise VerificationError(
                "LoopPlan.workspace must be a WorkspaceInsertion or None",
                code="loop_plan_structure_invalid",
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
                "LoopPlan workspace.axis_loops must be a non-empty tuple",
                code="loop_plan_structure_invalid",
            )
        for position, axis in enumerate(workspace.axis_loops):
            _validate_loop_ref(axis, f"workspace.axis_loops[{position}]")
        if type(workspace.dense) is not bool:
            raise VerificationError(
                "LoopPlan workspace.dense must be a bool",
                code="loop_plan_structure_invalid",
            )
        if workspace.dense and len(workspace.axis_loops) != 1:
            raise VerificationError(
                "LoopPlan dense workspace insertion stores exactly one axis",
                code="loop_plan_structure_invalid",
            )
    _validate_output_decision_tokens(typed_plan)
    if typed_plan.auto_policy is not None:
        _validate_auto_policy_structure(typed_plan.auto_policy)
    if type(typed_plan.provenance) is not str or not typed_plan.provenance:
        raise VerificationError(
            "LoopPlan.provenance must be a non-empty string",
            code="loop_plan_structure_invalid",
        )
    if type(typed_plan.tag) is not str:
        raise VerificationError(
            "LoopPlan.tag must be a string", code="loop_plan_structure_invalid"
        )
    _validate_conversions_structure(typed_plan.conversions)
    if typed_plan.vector_loops is not None:
        if type(typed_plan.vector_loops) is not tuple:
            raise VerificationError(
                "LoopPlan.vector_loops must be a tuple or None",
                code="loop_plan_structure_invalid",
            )
        for position, loop in enumerate(typed_plan.vector_loops):
            _validate_loop_ref(loop, f"vector_loops[{position}]")
    if typed_plan.entries_of is not None:
        _validate_symbol_id(typed_plan.entries_of, "entries_of")
    return typed_plan


def _validate_conversions_structure(conversions: object) -> None:
    """Reject malformed stored operand conversions exactly."""

    if type(conversions) is not tuple:
        raise VerificationError(
            "LoopPlan.conversions must be a tuple", code="loop_plan_conversion_invalid"
        )
    for position, conversion in enumerate(conversions):
        if type(conversion) is not OperandConversion:
            raise VerificationError(
                f"LoopPlan conversions[{position}] must be an OperandConversion",
                code="loop_plan_conversion_invalid",
            )
        _validate_stored_fields(
            conversion,
            f"conversions[{position}]",
            ("operand_id", "mode_order", "format"),
        )
        if type(conversion.format) is not str or set(conversion.format) - {"d", "s"}:
            raise VerificationError(
                f"LoopPlan conversions[{position}].format must be a string of "
                "'d' (dense) and 's' (compressed) levels",
                code="loop_plan_conversion_invalid",
            )
        _validate_symbol_id(
            conversion.operand_id, f"conversions[{position}].operand_id"
        )
        if type(conversion.mode_order) is not tuple or not all(
            _is_non_bool_int(mode) for mode in conversion.mode_order
        ):
            raise VerificationError(
                "LoopPlan conversion mode orders must be tuples of integers",
                code="loop_plan_conversion_invalid",
            )


def _validate_auto_policy_structure(auto_policy: object) -> None:
    """Reject a malformed stored automatic origin policy exactly."""

    if type(auto_policy) is not AutoOriginPolicy:
        raise VerificationError(
            "LoopPlan.auto_policy must be an AutoOriginPolicy or None",
            code="loop_plan_auto_policy_invalid",
        )
    _validate_stored_fields(
        auto_policy,
        "auto_policy",
        ("schema", "regblock_enabled", "tile_width"),
    )
    if (
        type(auto_policy.schema) is not str
        or auto_policy.schema != AUTO_ORIGIN_POLICY_SCHEMA
    ):
        raise VerificationError(
            "LoopPlan auto_policy.schema must be the exact "
            f"{AUTO_ORIGIN_POLICY_SCHEMA!r} version token",
            code="loop_plan_auto_policy_invalid",
        )
    if type(auto_policy.regblock_enabled) is not bool:
        raise VerificationError(
            "LoopPlan auto_policy.regblock_enabled must be a bool",
            code="loop_plan_auto_policy_invalid",
        )
    if not _is_non_bool_int(auto_policy.tile_width):
        raise VerificationError(
            "LoopPlan auto_policy.tile_width must be an integer",
            code="loop_plan_auto_policy_invalid",
        )
    if auto_policy.tile_width <= 0:
        raise VerificationError(
            "LoopPlan auto_policy.tile_width must be positive",
            code="loop_plan_auto_policy_invalid",
        )
    _verify_cpp_int_width(auto_policy.tile_width, "auto_policy.tile_width")


def _verify_empty_loop_plan(plan: LoopPlan) -> None:
    if (
        plan.loop_order
        or plan.tiles
        or plan.panel_bounds
        or plan.relayout is not None
        or plan.result_tile is not None
        or plan.parallel_loop is not None
        or plan.workspace is not None
        or plan.vector_loops
    ):
        raise VerificationError(
            "LoopPlan references loops but normalized CIN defines none",
            code="loop_plan_reference_invalid",
        )


def _verify_complete_loop_order(plan: LoopPlan, indices: Dict[IndexId, str]) -> None:
    if len(plan.loop_order) != len(set(plan.loop_order)):
        raise VerificationError(
            "LoopPlan.loop_order contains duplicate IndexId values",
            code="loop_plan_loop_order_invalid",
        )
    for position, index_id in enumerate(plan.loop_order):
        if index_id not in indices:
            raise VerificationError(
                "LoopPlan loop_order"
                f"[{position}] references unknown IndexId "
                f"{_diagnostic_int(index_id.value)}",
                code="loop_plan_loop_order_invalid",
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
            f"missing {missing_values}",
            code="loop_plan_loop_order_invalid",
        )


def converted_cin(
    cin: "IndexStmt", conversions: Tuple[OperandConversion, ...]
) -> "IndexStmt":
    """An owned copy of ``cin`` with each converted operand in its new order.

    A tensor stored in mode order ``m`` and accessed ``A[x_0, ..., x_n]`` holds
    the same arrays as the tensor stored in identity order, of shape
    ``(extent of mode m_0, ..., extent of mode m_n)``, accessed
    ``A[x_{m_0}, ..., x_{m_n}]``: a CSC matrix is the CSR form of its
    transpose.  Each converted operand is written in that second form, so
    every compressed tensor of the converted program keeps identity mode
    order, and the arrays a conversion produces at run time bind to it
    unchanged (``TensorVar.shape`` and a runtime tensor's shape are both in
    storage order).  The copy (``normalize_cin``) keeps every stable
    identity, so the plan's other decisions name the same loops and tensors
    in it.  An operand already stored in its target order is rewritten the
    same way.  The converted operand takes the conversion's layout.
    """

    from .cin_analysis import normalize_cin

    converted = normalize_cin(cin)
    if not conversions:
        return converted
    by_operand = {conversion.operand_id: conversion for conversion in conversions}
    rewritten = set()
    for access in converted.get_rhs_tensor_accesses():
        tensor = access.tensor
        conversion = by_operand.get(tensor.symbol_id)
        if conversion is None:
            continue
        target = conversion.mode_order
        rank = len(access.indices)
        if sorted(target) != list(range(rank)):
            raise VerificationError(
                "LoopPlan conversion mode order must permute the operand's modes",
                code="loop_plan_conversion_invalid",
            )
        if len(conversion.format) != rank:
            raise VerificationError(
                "LoopPlan conversion format must name one level per operand mode",
                code="loop_plan_conversion_invalid",
            )
        if id(tensor) not in rewritten:
            tensor.format = parse_format(conversion.format)
            current = list(tensor.mode_order or range(rank))
            if tensor.shape is not None:
                logical = [0] * rank
                for level, mode in enumerate(current):
                    logical[mode] = tensor.shape[level]
                tensor.shape = tuple(logical[mode] for mode in target)
            tensor.mode_order = list(range(rank))
            rewritten.add(id(tensor))
        access.indices = [access.indices[mode] for mode in target]
        access.index_ids = tuple(index.index_id for index in access.indices)
    return converted


def out_of_order_modes(access: object) -> Optional[Tuple[int, ...]]:
    """The mode order of a tensor :func:`storage_order_form` rewrites, or ``None``.

    That is a stored tensor (not a workspace) with a non-dense level whose
    mode order is a permutation of its modes other than the identity.
    """

    from .cin import TensorAccess, TensorVar

    if type(access) is not TensorAccess or type(access.tensor) is not TensorVar:
        return None
    tensor = access.tensor
    if tensor.mode_order is None or tensor.format is None:
        return None
    modes = tuple(tensor.mode_order)
    rank = len(access.indices)
    if modes == tuple(range(rank)):
        return None
    if any(type(mode) is not int for mode in modes) or sorted(modes) != list(
        range(rank)
    ):
        return None
    levels = tensor.get_level_types()
    if len(levels) != rank or all(level is LevelType.DENSE for level in levels):
        return None
    return modes


def storage_order_form(cin: "IndexStmt") -> "IndexStmt":
    """``cin`` with every tensor that has a non-dense level in identity mode order.

    The identity :func:`converted_cin` uses for a converted operand holds for
    any stored tensor: one stored in mode order ``m`` and accessed
    ``A[x_0, ..., x_n]`` holds the same arrays as the tensor stored in
    identity order, of the same storage-order shape, accessed
    ``A[x_{m_0}, ..., x_{m_n}]``.  The lowering descends a tensor's
    compressed levels in storage order, so it is given that second form: a
    caller's CSC matrix is lowered as the CSR form of its transpose, and a
    result declared in a permuted order is assembled level by level in that
    order.  The arrays, the shapes (``TensorVar.shape`` is in storage order)
    and the kernel's arguments are unchanged.  The kernel's result arrays are
    in the declared storage order, so whoever wraps them needs the declared
    mode order (:func:`out_of_order_modes` of the result's access in ``cin``).

    Every access to a rewritten tensor is rewritten from the same mode order.
    An all-dense tensor keeps its mode order (the lowering addresses it by
    coordinate), and so does a tensor whose mode order is not a permutation
    of its modes (the lowering refuses it).  Returns ``cin`` itself when
    nothing is rewritten, and otherwise an owned copy (``normalize_cin``,
    which keeps every stable identity, so a plan made on ``cin`` names the
    same loops and tensors in it).
    """

    from .cin_analysis import normalize_cin

    def accesses_of(statement: "IndexStmt") -> list:
        return [
            *statement.get_rhs_tensor_accesses(),
            *statement.get_result_tensor_accesses(),
        ]

    if all(out_of_order_modes(access) is None for access in accesses_of(cin)):
        return cin
    rewritten = normalize_cin(cin)
    # The mode order is read from the tensor before any access resets it; one
    # tensor object may be reached through several accesses.
    modes_of: Dict[int, Tuple["TensorVar", Tuple[int, ...]]] = {}
    for access in accesses_of(rewritten):
        modes = out_of_order_modes(access)
        if modes is not None:
            modes_of.setdefault(id(access.tensor), (access.tensor, modes))
    seen = set()
    for access in accesses_of(rewritten):
        entry = modes_of.get(id(access.tensor))
        if entry is None or id(access) in seen:
            continue
        seen.add(id(access))
        _, modes = entry
        if len(access.indices) != len(modes):
            raise VerificationError(
                "every access to a tensor must name one index per mode"
            )
        access.indices = [access.indices[mode] for mode in modes]
        access.index_ids = tuple(index.index_id for index in access.indices)
    for tensor, modes in modes_of.values():
        tensor.mode_order = list(range(len(modes)))
    return rewritten


def _verify_conversions(cin: object, plan: LoopPlan) -> None:
    """Each conversion names a compressed operand, a new permutation and a
    layout that stores exactly the operand's entries."""

    rhs = {
        tensor.symbol_id: tensor
        for tensor in cast("IndexStmt", cin).get_rhs_tensor_vars()
    }
    seen = set()
    for position, conversion in enumerate(plan.conversions):
        path = f"conversions[{position}]"
        tensor = rhs.get(conversion.operand_id)
        if tensor is None:
            raise VerificationError(
                f"LoopPlan {path} must name a right-hand-side tensor",
                code="loop_plan_conversion_invalid",
            )
        if conversion.operand_id in seen:
            raise VerificationError(
                "LoopPlan converts one operand twice",
                code="loop_plan_conversion_invalid",
            )
        seen.add(conversion.operand_id)
        levels = tensor.get_level_types()
        if sorted(conversion.mode_order) != list(range(len(levels))):
            raise VerificationError(
                f"LoopPlan {path} mode order must permute the operand's modes",
                code="loop_plan_conversion_invalid",
            )
        current = tuple(tensor.mode_order or range(len(levels)))
        if conversion.mode_order == current:
            raise VerificationError(
                f"LoopPlan {path} must change the operand's mode order",
                code="loop_plan_conversion_invalid",
            )
        if all(level is LevelType.DENSE for level in levels):
            # An all-dense operand is addressed by its coordinates in any
            # loop order, so converting one is never needed.
            raise VerificationError(
                f"LoopPlan {path} must name an operand with a compressed level",
                code="loop_plan_conversion_invalid",
            )
        if any(
            level not in (LevelType.DENSE, LevelType.COMPRESSED) for level in levels
        ):
            raise VerificationError(
                f"LoopPlan {path} must name an operand of dense and compressed levels",
                code="loop_plan_conversion_invalid",
            )
        if len(conversion.format) != len(levels):
            raise VerificationError(
                f"LoopPlan {path} format must name one level per operand mode",
                code="loop_plan_conversion_invalid",
            )
        if "sd" in conversion.format:
            # A dense level below a compressed one stores every cell under
            # each stored parent: cells the operand does not hold.
            raise VerificationError(
                f"LoopPlan {path} format must store only the operand's entries: "
                "no dense level below a compressed one",
                code="loop_plan_conversion_invalid",
            )
        if "s" not in conversion.format:
            raise VerificationError(
                f"LoopPlan {path} format must keep a compressed level",
                code="loop_plan_conversion_invalid",
            )


def _verify_vector_loop_refs(
    plan: LoopPlan,
    check_loop: Callable[[LoopRef, str], None],
    check_derived_loop: Callable[[LoopRef, str], None],
    affine_tiled_ids: Set[IndexId],
    panel_tile_ids: Set[IndexId],
) -> None:
    """Each loop ``vector_loops`` names is a loop of the plan's nest, named once.

    ``check_loop`` and ``check_derived_loop`` are :func:`verify_loop_plan`'s
    checks of a reference against the program's indices and the plan's affine
    tiles.  Whether the rule would mark the loop is decided only when the
    kernel is lowered, where the loop's body is known.

    Each refusal carries a code (``invalid_vector_loops`` for a reference to
    no loop of the nest, ``unsupported_vector_loop`` for a loop a panel tile
    splits): a request naming a loop the automatic plan tiled reaches these
    from the public schedule.
    """

    if plan.vector_loops is None:
        return
    if len(set(plan.vector_loops)) != len(plan.vector_loops):
        _vector_loop_refusal(
            "invalid_vector_loops", None, "LoopPlan.vector_loops names a loop twice"
        )
    for position, loop in enumerate(plan.vector_loops):
        for check in (check_loop, check_derived_loop):
            try:
                check(loop, f"vector_loops[{position}]")
            except VerificationError as error:
                _vector_loop_refusal("invalid_vector_loops", position, str(error))
        if loop.part == LoopPart.LOGICAL and loop.index_id in affine_tiled_ids:
            _vector_loop_refusal(
                "invalid_vector_loops",
                position,
                f"LoopPlan vector_loops[{position}] names an unsplit loop "
                "that an affine tile replaces with its two parts",
            )
        if loop.index_id in panel_tile_ids:
            # A panel tile becomes a sparse coordinate window only after
            # lowering, so which of its loops a mark would stay on is not
            # defined by the plan; it is refused rather than guessed.
            _vector_loop_refusal(
                "unsupported_vector_loop",
                position,
                f"LoopPlan vector_loops[{position}] names a loop that a panel "
                "tile splits into a sparse coordinate window after lowering",
            )


def _vector_loop_refusal(code: str, position: Optional[int], message: str) -> NoReturn:
    """A coded refusal of one ``vector_loops`` reference, as plan legality spells
    its refusals (``stage=LoopPlan: <code> at <path>: <message>``)."""

    from .loop_plan_legality import LoopPlanDiagnostic

    path: Tuple[str, ...] = ("plan", "vector_loops")
    if position is not None:
        path = (*path, str(position))
    raise VerificationError(
        f"stage=LoopPlan: {code} at {'/'.join(path)}: {message}",
        diagnostics=(LoopPlanDiagnostic(code=code, message=message, path=path),),
    )


def verify_loop_plan(
    cin: object, plan: LoopPlan, *, dense_results_any_order: bool = False
) -> LoopPlan:
    """Verify structural and semantic legality at the scheduling boundary.

    ``dense_results_any_order``: see
    :func:`loop_plan_legality.verify_loop_plan_semantics`.  A plan with
    ``conversions`` is verified against ``cin`` with those operands converted
    (:func:`converted_cin`).
    """

    plan = _validate_loop_plan_structure(plan)
    analysis = _analyze_loop_plan_cin(cin)
    if plan.conversions:
        _verify_conversions(cin, plan)
        cin = converted_cin(cast("IndexStmt", cin), plan.conversions)
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

        verify_loop_plan_semantics(
            analysis, plan, dense_results_any_order=dense_results_any_order
        )
        return plan

    def check_loop(loop: LoopRef, path: str) -> None:
        if loop.index_id not in indices:
            raise VerificationError(
                f"LoopPlan {path} references unknown IndexId "
                f"{_diagnostic_int(loop.index_id.value)}",
                code="loop_plan_reference_invalid",
            )

    def check_symbol(symbol_id: SymbolId, path: str) -> None:
        if symbol_id not in symbols:
            raise VerificationError(
                f"LoopPlan {path} references unknown SymbolId "
                f"{_diagnostic_int(symbol_id.value)}",
                code="loop_plan_reference_invalid",
            )

    _verify_complete_loop_order(plan, indices)

    tiled_ids = set()
    affine_tiled_ids = set()
    for position, tile in enumerate(plan.tiles):
        check_loop(tile.loop, f"tiles[{position}].loop")
        if tile.loop.part != LoopPart.LOGICAL:
            raise VerificationError(
                "LoopPlan tiles must target logical loops",
                code="loop_plan_reference_invalid",
            )
        if tile.loop.index_id in tiled_ids:
            raise VerificationError(
                "LoopPlan tiles the same logical loop twice",
                code="loop_plan_reference_invalid",
            )
        tiled_ids.add(tile.loop.index_id)
        if tile.kind == "affine":
            affine_tiled_ids.add(tile.loop.index_id)
        if tile.width <= 0:
            raise VerificationError(
                "LoopPlan tile widths must be positive",
                code="loop_plan_reference_invalid",
            )
        _verify_cpp_int_width(tile.width, "tile widths")
        if tile.placement.kind == PlacementKind.CHILD_OF:
            if tile.placement.parent is None or tile.placement.depth is not None:
                raise VerificationError(
                    "LoopPlan child placement requires only a parent loop",
                    code="loop_plan_reference_invalid",
                )
        elif tile.placement.kind == PlacementKind.AT_DEPTH:
            if tile.placement.depth is None or tile.placement.depth < 0:
                raise VerificationError(
                    "LoopPlan depth placement requires a non-negative depth",
                    code="loop_plan_reference_invalid",
                )
            if tile.placement.parent is not None:
                raise VerificationError(
                    "LoopPlan depth placement cannot also name a parent",
                    code="loop_plan_reference_invalid",
                )
        elif tile.placement.parent is not None or tile.placement.depth is not None:
            raise VerificationError(
                "LoopPlan outermost placement cannot name a parent or depth",
                code="loop_plan_reference_invalid",
            )
        if tile.placement.parent is not None:
            check_loop(tile.placement.parent, f"tiles[{position}].placement.parent")

    def check_derived_loop(loop: LoopRef, path: str) -> None:
        if loop.part != LoopPart.LOGICAL and loop.index_id not in affine_tiled_ids:
            raise VerificationError(
                f"LoopPlan {path} references a split part of a non-affine loop",
                code="loop_plan_reference_invalid",
            )

    for position, tile in enumerate(plan.tiles):
        if tile.placement.parent is not None:
            check_derived_loop(
                tile.placement.parent, f"tiles[{position}].placement.parent"
            )

    panel_tile_ids = {tile.loop.index_id for tile in plan.tiles if tile.kind == "panel"}
    bound_ids = {bound.loop.index_id for bound in plan.panel_bounds}
    if len(bound_ids) != len(plan.panel_bounds):
        raise VerificationError(
            "LoopPlan panel bounds must reference unique loops",
            code="loop_plan_reference_invalid",
        )
    if panel_tile_ids != bound_ids:
        raise VerificationError(
            "LoopPlan panel tiles and dense panel bounds must correspond exactly",
            code="loop_plan_reference_invalid",
        )
    for position, bound in enumerate(plan.panel_bounds):
        check_loop(bound.loop, f"panel_bounds[{position}].loop")
        check_symbol(bound.tensor_id, f"panel_bounds[{position}].tensor_id")
        if bound.level < 0:
            raise VerificationError(
                "LoopPlan panel-bound levels must be non-negative",
                code="loop_plan_reference_invalid",
            )

    if plan.parallel_loop is not None:
        check_loop(plan.parallel_loop, "parallel_loop")
        check_derived_loop(plan.parallel_loop, "parallel_loop")
    if plan.parallel_loop is not None and any(tile.parallel for tile in plan.tiles):
        raise VerificationError(
            "LoopPlan may select parallelism either on the plan or a tile, not both",
            code="loop_plan_reference_invalid",
        )
    if sum(tile.parallel for tile in plan.tiles) > 1:
        raise VerificationError(
            "LoopPlan may mark at most one tile parallel",
            code="loop_plan_reference_invalid",
        )

    _verify_vector_loop_refs(
        plan, check_loop, check_derived_loop, affine_tiled_ids, panel_tile_ids
    )
    if plan.entries_of is not None:
        check_symbol(plan.entries_of, "entries_of")

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
            raise VerificationError(
                "LoopPlan relayout strip width must be positive",
                code="loop_plan_reference_invalid",
            )
        _verify_cpp_int_width(relayout.strip_width, "relayout strip width")
        if relayout.operand_panel_level < 0 or relayout.operand_pack_level < 0:
            raise VerificationError(
                "LoopPlan relayout levels must be non-negative",
                code="loop_plan_reference_invalid",
            )
        for position, index_id in enumerate(relayout.access_indices):
            check_loop(LoopRef(index_id), f"relayout.access_indices[{position}]")

    if plan.result_tile is not None:
        result_tile = plan.result_tile
        check_symbol(result_tile.result_id, "result_tile.result_id")
        check_loop(result_tile.tile_loop, "result_tile.tile_loop")
        check_derived_loop(result_tile.tile_loop, "result_tile.tile_loop")
        if result_tile.result_level < 0:
            raise VerificationError(
                "LoopPlan result-tile level must be non-negative",
                code="loop_plan_reference_invalid",
            )
        for field, index_ids in (
            ("result_prefix", result_tile.result_prefix),
            ("access_indices", result_tile.access_indices),
        ):
            for position, index_id in enumerate(index_ids):
                check_loop(LoopRef(index_id), f"result_tile.{field}[{position}]")

    from .loop_plan_legality import verify_loop_plan_semantics

    verify_loop_plan_semantics(
        analysis, plan, dense_results_any_order=dense_results_any_order
    )
    return plan


def verify_scheduled_cin(scheduled: ScheduledCIN) -> ScheduledCIN:
    """Verify the exact transitional scheduling carrier."""

    if type(scheduled) is not ScheduledCIN:
        raise VerificationError(
            "ScheduledCIN verifier received the wrong artifact",
            code="loop_plan_artifact_invalid",
        )
    _validate_stored_fields(
        scheduled,
        "scheduled carrier",
        ("normalized_cin", "verified_loop_plan"),
    )
    verify_loop_plan(scheduled.normalized_cin, scheduled.verified_loop_plan)
    return scheduled
