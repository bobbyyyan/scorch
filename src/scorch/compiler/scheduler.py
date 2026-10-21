import copy
import math
import operator
from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields as dataclass_fields, replace
from itertools import repeat
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    cast,
)

import torch

from scorch.compiler.cin import (
    BinaryOp,
    CIN,
    CINIndexVariablesGetter,
    CINVisitorAccept,
    ForAll,
    IndexStmt,
    IndexVar,
    Operation,
    TensorAccess,
    TensorAssign,
    TensorVar,
    TileSizeVar,
    Where,
    Workspace,
    WorkspaceAccess,
    _is_index_stmt_instance,
    coordinate_result_driver,
)
from .cin_analysis import (
    _compile_options_at_cin_boundary,
    _trusted_normalized_cin,
    normalize_cin,
)
from .compilation_context import CompilerStageId, CompilationContext
from .compile_options import CompileOptions, SchedulerCostModel, SchedulerPolicy
from .identity import IndexId, SymbolId
from .diagnostics import (
    CompileOptionsDiagnostic,
    CompileOptionsError,
    CompilerInvariantError,
    InvalidSchedule,
    UnsupportedFeature,
    VerificationError,
)
from .legacy_cin_adapter import validate_legacy_cin_display_names
from .loop_plan import (
    AUTO_ORIGIN_POLICY_SCHEMA,
    MAX_AFFINE_TILE_WIDTH,
    AutoOriginPolicy,
    LoopPart,
    LoopPlacement,
    LoopPlan,
    LoopRef,
    LoopTile,
    OperandRelayout,
    PanelBound,
    PlacementKind,
    ResultTile,
    ScheduledCIN,
    WorkspaceInsertion,
    entity_display_names,
    verify_loop_plan,
)
from . import sparse_accumulator as _sparse_accumulator_module
from . import sparse_assembly as _sparse_assembly_module
from .sparse_accumulator import (
    SPARSE_ACCUMULATOR_STRUCTURES,
    is_accumulator_structure,
)
from .sparse_assembly import (
    SPARSE_ASSEMBLY_STRATEGIES,
    entry_positioned_receiver,
    is_assembly_strategy,
    partitionable_receiver_levels,
    receiver_requirement,
)
from scorch.format import LevelType
from ..exceptions import (
    ScorchAssertionError,
    ScorchNotImplementedError,
    ScorchTypeError,
    ScorchValueError,
    defect_code,
)

# Compatibility override read only while constructing a snapshot at a direct
# scheduler boundary. Compiler stages receive the resulting immutable options
# explicitly and never consult mutable process or context state.
_REGBLOCK_FORCE: ContextVar[Optional[bool]] = ContextVar(
    "scorch_regblock_force", default=None
)

# The loop-plan legality codes that are properties of the recorded loop ORDER
# alone -- the two rules ``_verify_storage_order`` owns.  Named here so the
# automatic plan origin can recognize a refusal a different legal order could
# answer without restating either rule.
_ORDER_LEGALITY_CODES = frozenset({"result_storage_order", "sparse_parent_dominance"})


def get_forced_regblock() -> Optional[bool]:
    """Return the register-block compatibility override for this context."""

    value = _REGBLOCK_FORCE.get()
    if value is not None and type(value) is not bool:
        raise ScorchTypeError(
            "register-block override must be a bool or None",
            code="regblock_override_type_invalid",
        )
    return value


def _compile_options_at_scheduler_boundary(
    compile_options: Optional[CompileOptions] = None,
) -> CompileOptions:
    """Resolve one immutable snapshot at a direct scheduler boundary."""

    if compile_options is not None:
        if type(compile_options) is not CompileOptions:
            raise ScorchTypeError(
                "compile_options must be a CompileOptions snapshot",
                code="internal_argument_type_invalid",
            )
        return compile_options

    return _compile_options_at_cin_boundary()


def _regblock_enabled(
    compile_options: Optional[CompileOptions] = None,
) -> bool:
    """Return the snapshotted register-block scheduling decision."""

    return _compile_options_at_scheduler_boundary(
        compile_options
    ).scheduler.regblock_enabled


@contextmanager
def regblock_force(value: Optional[bool]):
    """Temporarily override register blocking at direct scheduler boundaries."""

    if value is not None and type(value) is not bool:
        raise TypeError("regblock_force expects a bool or None")
    token = _REGBLOCK_FORCE.set(value)
    try:
        yield
    finally:
        _REGBLOCK_FORCE.reset(token)


def _regblock_max_n(
    compile_options: Optional[CompileOptions] = None,
) -> int:
    """Return the snapshotted runtime free-dimension cutoff."""

    return _compile_options_at_scheduler_boundary(
        compile_options
    ).scheduler.regblock_max_n


def _regblock_tile_width(
    compile_options: Optional[CompileOptions] = None,
) -> int:
    """Return the snapshotted register-block tile width."""

    return _compile_options_at_scheduler_boundary(
        compile_options
    ).scheduler.regblock_tile_width


def _scheduler_costs_at_boundary(
    costs: Optional[SchedulerCostModel],
    compile_options: Optional[CompileOptions],
) -> Tuple[CompileOptions, SchedulerCostModel]:
    """Resolve a snapshot and its effective immutable cost model once."""

    options = _compile_options_at_scheduler_boundary(compile_options)
    if costs is None:
        costs = options.scheduler.cost_model
    elif type(costs) is not SchedulerCostModel:
        raise CompileOptionsError(
            (
                CompileOptionsDiagnostic(
                    code="invalid_type",
                    field="scheduler.cost_model",
                    message="expected an exact SchedulerCostModel",
                ),
            )
        )
    elif compile_options is not None and costs != options.scheduler.cost_model:
        raise CompileOptionsError(
            (
                CompileOptionsDiagnostic(
                    code="conflicting_scheduler_cost_model",
                    field="scheduler.cost_model",
                    message=("the explicit cost model disagrees with CompileOptions"),
                ),
            )
        )
    elif costs != options.scheduler.cost_model:
        options = replace(
            options,
            scheduler=replace(options.scheduler, cost_model=costs),
        )
    return options, costs


def _validate_requested_schedule(
    options: CompileOptions,
    schedule: Optional["Schedule"],
    *,
    exact: bool = False,
) -> None:
    """Keep legacy schedule arguments consistent with the owned snapshot."""

    requested = options.requested_schedule
    if schedule is not None:
        _validate_schedule_structure(schedule)
    if requested is not None and requested is not schedule:
        _validate_schedule_structure(requested)
    if (exact and requested != schedule) or (
        not exact and requested is not None and requested != schedule
    ):
        raise CompileOptionsError(
            (
                CompileOptionsDiagnostic(
                    code="conflicting_schedule",
                    field="requested_schedule",
                    message=("the scheduler argument disagrees with CompileOptions"),
                ),
            )
        )


# Preserve the existing private spelling for downstream tests and extensions
# while making the immutable policy type part of CompileOptions.
_CostModelConstants = SchedulerCostModel


@dataclass(frozen=True)
class TileSpec:
    """A tuner-provided strip-mining decision for one logical index variable.

    ``placement`` controls where the outer tile loop is inserted. The inner loop
    remains at the logical variable's original position, so tiling ``k`` in
    ``i,j,k`` with ``placement="child_of:i"`` produces ``i,k_out,j,k_in``.

    Affine tiles are represented in CIN. ``panel`` requests a sparse coordinate
    window such as SpMM tile-j; it is completed after concrete compressed
    iterators have been lowered to LLIR. ``accum`` selects result lifetime:
    ``direct`` updates final storage, ``stack`` uses the existing row-local tile
    workspace, and ``heap`` uses a compact dense-prefix-by-tile buffer initialized
    at the affine outer-tile entry and copied out at its exit.
    """

    index_var: str
    width: int
    placement: str = "outermost"
    parallel: bool = False
    kind: str = "affine"
    accum: str = "stack"
    unroll: bool = True

    def __post_init__(self) -> None:
        if type(self.index_var) is not str or not self.index_var:
            raise ScorchValueError(
                "TileSpec.index_var must be a non-empty string",
                code="tile_spec_field_value_invalid",
            )
        if type(self.width) is not int:
            raise ScorchTypeError(
                "TileSpec.width must be an integer", code="tile_spec_field_type_invalid"
            )
        if self.width <= 0:
            raise ScorchValueError(
                "TileSpec.width must be greater than zero",
                code="tile_spec_field_value_invalid",
            )
        if self.width > MAX_AFFINE_TILE_WIDTH:
            raise ScorchValueError(
                "TileSpec.width must fit the C++ constexpr int target",
                code="tile_spec_field_value_invalid",
            )
        if type(self.placement) is not str:
            raise ScorchTypeError(
                "TileSpec.placement must be a string",
                code="tile_spec_field_type_invalid",
            )
        if type(self.parallel) is not bool:
            raise ScorchTypeError(
                "TileSpec.parallel must be a bool", code="tile_spec_field_type_invalid"
            )
        if type(self.kind) is not str:
            raise ScorchTypeError(
                "TileSpec.kind must be a string", code="tile_spec_field_type_invalid"
            )
        if self.kind not in ("affine", "panel"):
            raise ScorchValueError(
                "TileSpec.kind must be 'affine' or 'panel'",
                code="tile_spec_field_value_invalid",
            )
        if type(self.accum) is not str:
            raise ScorchTypeError(
                "TileSpec.accum must be a string", code="tile_spec_field_type_invalid"
            )
        if self.accum not in ("stack", "direct", "heap"):
            raise ScorchValueError(
                "TileSpec.accum must be 'stack', 'direct', or 'heap'",
                code="tile_spec_field_value_invalid",
            )
        if type(self.unroll) is not bool:
            raise ScorchTypeError(
                "TileSpec.unroll must be a bool", code="tile_spec_field_type_invalid"
            )
        if not (
            self.placement == "outermost"
            or self.placement.startswith("child_of:")
            or self.placement.startswith("at_depth:")
        ):
            raise ScorchValueError(
                "TileSpec.placement must be 'outermost', 'child_of:<var>', "
                "or 'at_depth:<n>'",
                code="tile_spec_field_value_invalid",
            )
        if self.placement.startswith("child_of:"):
            if not self.placement.split(":", 1)[1]:
                raise ScorchValueError(
                    "child_of placement requires an index variable",
                    code="tile_spec_field_value_invalid",
                )
        elif self.placement.startswith("at_depth:"):
            depth = self.placement.split(":", 1)[1]
            try:
                parsed_depth = int(depth)
            except ValueError as exc:
                raise ScorchValueError(
                    "at_depth placement requires a non-negative integer",
                    code="tile_spec_field_value_invalid",
                ) from exc
            if parsed_depth < 0:
                raise ScorchValueError(
                    "at_depth placement requires a non-negative integer",
                    code="tile_spec_field_value_invalid",
                )


@dataclass(frozen=True)
class RelayoutSpec:
    """Stage one dense input across a tiled logical variable.

    ``operand`` names the input tensor to stage, ``pack_var`` names its
    contiguous tiled dimension, and ``strip_width`` is the staged row stride.
    ``scope_var`` is the logical tiled loop whose outer iteration refreshes the
    staged contents.  For example, selecting a reduction-panel variable stages
    one panel, while selecting ``pack_var`` stages the full remaining access at
    the enclosing free-axis tile.  It is a logical loop anchor, not a kernel or
    tensor-role enum.

    ``scope_var=None`` is retained as a compatibility spelling for the unique
    panel variable.  :class:`Schedule` canonicalizes it to that explicit logical
    name before computing cache identity.

    The current lowering recognizes structurally compatible rank-2 dense reads
    in a CSR-by-dense contraction, but the schedule primitive itself only names
    an access, a tiled axis, and a refresh scope.
    """

    operand: str
    pack_var: str
    strip_width: int
    scope_var: Optional[str] = None

    def __post_init__(self) -> None:
        if type(self.operand) is not str or type(self.pack_var) is not str:
            raise ScorchTypeError(
                "RelayoutSpec operand and pack_var must be strings",
                code="relayout_spec_field_type_invalid",
            )
        if not self.operand or not self.pack_var:
            raise ScorchValueError(
                "RelayoutSpec operand and pack_var must be non-empty",
                code="relayout_spec_field_value_invalid",
            )
        if type(self.strip_width) is not int:
            raise ScorchTypeError(
                "RelayoutSpec.strip_width must be an integer",
                code="relayout_spec_field_type_invalid",
            )
        if self.strip_width <= 0:
            raise ScorchValueError(
                "RelayoutSpec.strip_width must be greater than zero",
                code="relayout_spec_field_value_invalid",
            )
        if self.strip_width > MAX_AFFINE_TILE_WIDTH:
            raise ScorchValueError(
                "RelayoutSpec.strip_width must fit the C++ constexpr int target",
                code="relayout_spec_field_value_invalid",
            )
        if self.scope_var is not None:
            if type(self.scope_var) is not str:
                raise ScorchTypeError(
                    "RelayoutSpec.scope_var must be a string or None",
                    code="relayout_spec_field_type_invalid",
                )
            if not self.scope_var:
                raise ScorchValueError(
                    "RelayoutSpec.scope_var must be non-empty",
                    code="relayout_spec_field_value_invalid",
                )


@dataclass(frozen=True)
class _RelayoutPlan:
    """CIN-derived metadata consumed by the post-lowering staging pass."""

    operand: str
    operand_id: SymbolId
    pack_var: str
    panel_var: str
    scope_var: str
    row_var: str
    access_index_vars: Tuple[str, ...]
    access_index_ids: Tuple[IndexId, ...]
    operand_panel_level: int
    operand_pack_level: int


@dataclass(frozen=True)
class _ResultTilePlan:
    """CIN-derived metadata for a heap-backed dense result tile.

    The compact buffer covers every dense result prefix position and one tile of
    the trailing free axis.  It is initialized at the affine outer-tile entry,
    receives redirected result updates for all enclosed reductions/panels, and
    is copied to the final result at outer-tile exit.
    """

    result: str
    result_id: SymbolId
    tile_var: str
    result_level: int
    result_prefix_vars: Tuple[str, ...]
    access_index_vars: Tuple[str, ...]
    access_index_ids: Tuple[IndexId, ...]


@dataclass(frozen=True)
class Schedule:
    """A complete, immutable scheduling decision suitable for cache keys.

    ``loop_order`` names the unsplit logical variables. ``tiles`` are applied in
    tuple order, making placement deterministic for multi-axis tiling.
    ``parallel_loop`` may name a logical loop or a generated ``*_out``/``*_in``
    loop. The full representation, rather than the human tag, is the cache key.

    ``assembly`` requests one sparse-output assembly strategy by name -- one
    traversal or two, distributed across workers or not
    (``SPARSE_ASSEMBLY_STRATEGIES`` lists them).  Which one is fastest depends
    on the operands' formats and density rather than on the receiver alone, so
    it is a schedule decision like the others.  ``None`` requests nothing and
    leaves the compiler its own per-receiver choice; it is not a synonym for
    ``"single_pass_serial"``.  A strategy the receiver cannot support is
    refused rather than silently downgraded.

    ``accumulator`` requests one sparse accumulation structure by name -- a list
    of the keys inserted, or a chain indexed by the key
    (``SPARSE_ACCUMULATOR_STRUCTURES`` lists them).  Which one is faster depends
    on the density and on the receiver's compressed extent, so it is a schedule
    decision like the others.  ``None`` requests nothing and leaves every layer
    its own choice -- which, where the two-phase assembly transform fires,
    remains the chain it substitutes today; it is not a synonym for
    ``"coordinate_list"``.  A structure this program cannot use is refused rather
    than silently ignored.

    ``vector_loops`` names the loops that keep a vector mark
    (``#pragma omp simd``), each spelled like ``parallel_loop``: a logical loop,
    or a generated ``*_out``/``*_in`` loop.  ``None`` requests nothing: each
    compiler places the marks it placed before this field existed.  Both
    compilers apply one rule, in one place: the loops below a sparse result
    entry whose sum is kept in a register, each marked when it is safe (no loop
    over a compressed level's positions in its body, not parallel or unrolled,
    no early exit).  With no request this compiler marks those loops and the
    new compiler marks none.  A tuple, the empty one included, keeps the mark
    on exactly the named loops; naming a loop the rule does not find safe is
    refused (``unsupported_vector_loop``) rather than ignored.  The new
    compiler reads the same decision, so one tuple marks the same loops in
    both; a comparison of the two compilers names the loops rather than
    relying on ``None``.

    ``entries_of`` names the input whose stored entries a COO result stores:
    one result entry for each stored entry of that input, in its storage order,
    an entry whose inner sum ran zero times included (its value is zero).  It
    applies where the loops over the result's indices walk that input's
    stored entries with a reduction below them
    (``sparse_assembly.entry_source``); each entry's position is then that
    input's, known before the loops run, which the parallel single pass of a
    COO result (``single_pass_chunk_parallel``) needs.  ``None`` requests
    nothing: each compiler stores what it stored before this field existed.
    This compiler stores every entry of the input its scalar accumulation
    walks; the new compiler appends only the entries some term reached.  A
    comparison of the two compilers names the input rather than relying on
    ``None``.  An input whose entries the loops do not walk that way, or a
    request a compiler cannot build, is refused (``unsupported_entries_of``).
    """

    loop_order: Optional[Tuple[str, ...]] = None
    tiles: Tuple[TileSpec, ...] = ()
    relayout: Optional[RelayoutSpec] = None
    tag: str = ""
    parallel_loop: Optional[str] = None
    assembly: Optional[str] = None
    accumulator: Optional[str] = None
    vector_loops: Optional[Tuple[str, ...]] = None
    entries_of: Optional[str] = None

    def __post_init__(self) -> None:
        if self.entries_of is not None:
            if type(self.entries_of) is not str:
                raise ScorchTypeError(
                    "Schedule.entries_of must be an input name or None",
                    code="schedule_field_type_invalid",
                )
            if not self.entries_of:
                raise ScorchValueError(
                    "Schedule.entries_of must be a non-empty name",
                    code="schedule_field_value_invalid",
                )
        if self.vector_loops is not None:
            if type(self.vector_loops) not in (list, tuple):
                raise ScorchTypeError(
                    "Schedule.vector_loops must be an owned list or tuple of names",
                    code="schedule_field_type_invalid",
                )
            if type(self.vector_loops) is list:
                object.__setattr__(self, "vector_loops", tuple(self.vector_loops))
            if any(type(name) is not str or not name for name in self.vector_loops):
                raise ScorchValueError(
                    "Schedule.vector_loops must contain non-empty strings",
                    code="schedule_field_value_invalid",
                )
            if len(self.vector_loops) != len(set(self.vector_loops)):
                raise ScorchValueError(
                    "Schedule.vector_loops names a loop twice",
                    code="schedule_loop_name_duplicate",
                )
        if self.loop_order is not None:
            if type(self.loop_order) is str:
                raise ScorchTypeError(
                    "Schedule.loop_order must be a sequence of names",
                    code="schedule_field_type_invalid",
                )
            if type(self.loop_order) not in (list, tuple):
                raise ScorchTypeError(
                    "Schedule.loop_order must be an owned list or tuple of names",
                    code="schedule_field_type_invalid",
                )
            if type(self.loop_order) is list:
                object.__setattr__(self, "loop_order", tuple(self.loop_order))
            if any(type(name) is not str or not name for name in self.loop_order):
                raise ScorchValueError(
                    "Schedule.loop_order must contain non-empty strings",
                    code="schedule_field_value_invalid",
                )
            if len(self.loop_order) != len(set(self.loop_order)):
                raise ScorchValueError(
                    "Schedule.loop_order contains duplicate variables",
                    code="schedule_loop_name_duplicate",
                )
        if type(self.tiles) not in (list, tuple):
            raise ScorchTypeError(
                "Schedule.tiles must be an owned list or tuple",
                code="schedule_field_type_invalid",
            )
        if type(self.tiles) is list:
            object.__setattr__(self, "tiles", tuple(self.tiles))
        if any(type(tile) is not TileSpec for tile in self.tiles):
            # Exact-type admission: a TileSpec subclass is caller code that
            # would otherwise execute inside compiler-trusted scopes.
            raise ScorchTypeError(
                "Schedule.tiles must contain exact TileSpec instances",
                code="schedule_field_type_invalid",
            )
        if self.relayout is not None and type(self.relayout) is not RelayoutSpec:
            raise ScorchTypeError(
                "Schedule.relayout must be an exact RelayoutSpec or None",
                code="schedule_field_type_invalid",
            )
        if self.relayout is not None and self.relayout.scope_var is None:
            panel_tiles = [tile for tile in self.tiles if tile.kind == "panel"]
            if len(panel_tiles) == 1:
                object.__setattr__(
                    self,
                    "relayout",
                    replace(self.relayout, scope_var=panel_tiles[0].index_var),
                )
        tile_names = [tile.index_var for tile in self.tiles]
        if len(tile_names) != len(set(tile_names)):
            raise ScorchValueError(
                "Schedule cannot tile the same index variable twice",
                code="schedule_tiles_duplicate_index",
            )
        if sum(tile.parallel for tile in self.tiles) > 1:
            raise ScorchValueError(
                "Schedule may select at most one parallel tile loop",
                code="schedule_parallel_selection_conflict",
            )
        if self.parallel_loop is not None and any(tile.parallel for tile in self.tiles):
            raise ScorchValueError(
                "Use either Schedule.parallel_loop or TileSpec.parallel, not both",
                code="schedule_parallel_selection_conflict",
            )
        if self.parallel_loop is not None:
            if type(self.parallel_loop) is not str:
                raise ScorchTypeError(
                    "Schedule.parallel_loop must be a string or None",
                    code="schedule_field_type_invalid",
                )
            if not self.parallel_loop:
                raise ScorchValueError(
                    "Schedule.parallel_loop must be a non-empty string",
                    code="schedule_parallel_selection_conflict",
                )
        if type(self.tag) is not str:
            raise ScorchTypeError(
                "Schedule.tag must be a string", code="schedule_field_type_invalid"
            )
        if self.assembly is not None and not is_assembly_strategy(self.assembly):
            raise ScorchValueError(
                "Schedule.assembly must name a sparse assembly strategy "
                f"({', '.join(SPARSE_ASSEMBLY_STRATEGIES)}) or be None",
                code="schedule_assembly_unknown",
            )
        if self.accumulator is not None and not is_accumulator_structure(
            self.accumulator
        ):
            raise ScorchValueError(
                "Schedule.accumulator must name a sparse accumulation structure "
                f"({', '.join(SPARSE_ACCUMULATOR_STRUCTURES)}) or be None",
                code="schedule_accumulator_unknown",
            )

    @property
    def cache_key(self) -> str:
        """Canonical cache discriminator containing every schedule field.

        The dataclass ``repr`` of the validated schedule, except that a field
        added after the key was first spelled (:data:`_SCHEDULE_KEY_WHEN_SET`)
        enters it only when set: a schedule that sets none of them keeps the
        key it had before they existed, and so does every options digest that
        carries it.

        Every read validates the schedule's current state, but the full validation
        (``_canonical_schedule``) and the formatting run once per distinct state.  A
        read first rebuilds its state's memo key (``_schedule_memo_key``), which makes
        the reads and refusals the full validation starts with, and returns the
        string memoized under it.  A state changed through ``object.__setattr__``
        has another memo key, so its next read validates it in full and refuses it
        if it is invalid.  A state the memo does not key is validated in full on
        every read, as before.  Rebuilding two Schedule objects on each read cost
        a warm ``einsum`` with a schedule about 87,000 instructions per read, twice
        per call (sapling 85220 and 85225).
        """

        context = _schedule_validation_context()
        memo_key = _schedule_memo_key(self, context)
        if memo_key is not None:
            try:
                known = _SCHEDULE_KEY_MEMO.get(memo_key)
            except TypeError:  # module state replaced by an unhashable value
                memo_key = known = None
            if known is not None:
                return known
        canonical = _canonical_schedule(self)
        key = _format_schedule_key(canonical)
        if memo_key is not None:
            # Under the memo key of the state validated and formatted: the canonical
            # copy, which no caller holds, rather than the key read above, which
            # another thread could have changed since.
            stored_key = _schedule_memo_key(canonical, context)
            if stored_key is not None:
                if len(_SCHEDULE_KEY_MEMO) >= _SCHEDULE_KEY_MEMO_LIMIT:
                    _SCHEDULE_KEY_MEMO.clear()
                _SCHEDULE_KEY_MEMO[stored_key] = key
        return key


#: Schedule fields whose unset value (``None``) leaves the cache key unchanged.
_SCHEDULE_KEY_WHEN_SET = frozenset({"vector_loops", "entries_of"})

#: The stored fields of each schedule carrier, in declaration order: the names
#: ``_stored_schedule_fields`` requires, and the order of a state's memo key.
_SCHEDULE_FIELDS = (
    "loop_order",
    "tiles",
    "relayout",
    "tag",
    "parallel_loop",
    "assembly",
    "accumulator",
    "vector_loops",
    "entries_of",
)
_TILE_SPEC_FIELDS = (
    "index_var",
    "width",
    "placement",
    "parallel",
    "kind",
    "accum",
    "unroll",
)
_RELAYOUT_SPEC_FIELDS = ("operand", "pack_var", "strip_width", "scope_var")

#: The cache key of each schedule state validated so far, under the state's memo
#: key (``_schedule_memo_key``).  Bounded like the parse memo: emptied when full.
_SCHEDULE_KEY_MEMO: Dict[tuple, str] = {}
_SCHEDULE_KEY_MEMO_LIMIT = 4096

#: The leaf values a memo key holds: exact immutable built-in types whose equal
#: values have equal ``repr``, so that equal memo keys format equal cache keys.
#: Not ``float`` (``0.0 == -0.0``, and their reprs differ).
_MEMO_LEAF_TYPES = frozenset({str, int, bool, type(None)})
_EXACT_STR = frozenset({str})


class _UnkeyedState(Exception):
    """A schedule state the memo does not key: its read takes the full validation."""


def _schedule_validation_context() -> tuple:
    """The module state the validation and the key format read besides the schedule.

    The assembly and accumulator name registries (``is_assembly_strategy`` and
    ``is_accumulator_structure`` read their modules' tuples), the tile width bound
    and the fields a key leaves out when unset.  It enters every memo key, so a
    change to any of them is a new state.
    """

    return (
        _sparse_assembly_module.SPARSE_ASSEMBLY_STRATEGIES,
        _sparse_accumulator_module.SPARSE_ACCUMULATOR_STRUCTURES,
        MAX_AFFINE_TILE_WIDTH,
        _SCHEDULE_KEY_WHEN_SET,
    )


def _schedule_memo_key(schedule: object, context: tuple) -> Optional[tuple]:
    """The memo key of ``schedule``'s current state, or None if the memo does not
    key it."""

    try:
        return (context, _carrier_memo_key(schedule, Schedule, _SCHEDULE_FIELDS))
    except _UnkeyedState:
        return None


def _carrier_memo_key(
    value: object, expected_type: type, fields: Tuple[str, ...]
) -> tuple:
    """The typed key of one schedule carrier's stored state.

    Read the way ``_stored_schedule_fields`` reads it, and ``_UnkeyedState`` wherever
    that function refuses: a value not exactly ``expected_type``, a ``__dict__`` that
    is not a dict of exactly ``fields`` under exact ``str`` names, or a field whose
    ``object.__getattribute__`` read is not the stored object.  A TileSpec or
    RelayoutSpec held by the carrier is keyed by the same rule.  No method of a
    stored name or value runs: every one is known to be of an exact built-in type
    before anything hashes or compares it.
    """

    if type(value) is not expected_type:
        raise _UnkeyedState
    try:
        state = object.__getattribute__(value, "__dict__")
    except Exception:
        raise _UnkeyedState from None
    if (
        type(state) is not dict
        or len(state) != len(fields)
        or not _EXACT_STR.issuperset(map(type, state))
    ):
        raise _UnkeyedState
    try:
        stored = tuple(map(state.__getitem__, fields))
        observed = tuple(map(object.__getattribute__, repeat(value), fields))
    except Exception:
        raise _UnkeyedState from None
    if not all(map(operator.is_, observed, stored)):
        raise _UnkeyedState
    return _memo_tuple(stored)


def _memo_tuple(values: tuple) -> tuple:
    """A tuple of stored values as it enters a memo key: each value's exact type
    beside the values, a nested tuple or carrier replaced by its own typed key."""

    types = tuple(map(type, values))
    if not _MEMO_LEAF_TYPES.issuperset(types):
        values = tuple(map(_memo_value, values))
    return (types, values)


def _memo_value(value: object) -> object:
    kind = type(value)
    if kind in _MEMO_LEAF_TYPES:
        return value
    if kind is tuple:
        return _memo_tuple(cast(tuple, value))
    if kind is TileSpec:
        return _carrier_memo_key(value, TileSpec, _TILE_SPEC_FIELDS)
    if kind is RelayoutSpec:
        return _carrier_memo_key(value, RelayoutSpec, _RELAYOUT_SPEC_FIELDS)
    raise _UnkeyedState


def _format_schedule_key(schedule: Schedule) -> str:
    """The cache key of one validated schedule (see ``Schedule.cache_key``)."""

    keyed = [
        f"{field.name}={getattr(schedule, field.name)!r}"
        for field in dataclass_fields(schedule)
        if field.repr
        and (
            field.name not in _SCHEDULE_KEY_WHEN_SET
            or getattr(schedule, field.name) is not None
        )
    ]
    return f"{type(schedule).__qualname__}({', '.join(keyed)})"


def _stored_schedule_fields(
    value: object,
    expected_type: type,
    expected_fields: Tuple[str, ...],
    what: str,
) -> Dict[str, Any]:
    """Snapshot one exact frozen schedule carrier without invoking caller code."""

    if type(value) is not expected_type:
        raise InvalidSchedule(
            f"{what} must be an exact {expected_type.__name__}",
            code="schedule_state_invalid",
        )
    try:
        state = object.__getattribute__(value, "__dict__")
    except Exception as error:
        raise InvalidSchedule(
            f"{what} has inaccessible stored state", code="schedule_state_invalid"
        ) from error
    if (
        type(state) is not dict
        or any(type(key) is not str for key in state)
        or set(state) != set(expected_fields)
    ):
        raise InvalidSchedule(
            f"{what} has invalid stored fields", code="schedule_state_invalid"
        )
    for field_name in expected_fields:
        try:
            observed = object.__getattribute__(value, field_name)
        except Exception as error:
            raise InvalidSchedule(
                f"{what}.{field_name} is not safely readable",
                code="schedule_state_invalid",
            ) from error
        if observed is not state[field_name]:
            raise InvalidSchedule(
                f"{what}.{field_name} diverges from its stored value",
                code="schedule_state_invalid",
            )
    return state


def _assembly_diagnostic(code: str, strategy: str, reason: str) -> Tuple[str, object]:
    """One structured assembly refusal, spelled the way the plan boundary spells it.

    The explicit surface refuses earlier than the plan legality boundary does, and
    a refusal without a structured code is not a refusal this branch's
    fail-closed gate can count.  Building the diagnostic here rather than
    importing the boundary's private helpers keeps the module dependency one-way.
    """

    from .loop_plan_legality import LoopPlanDiagnostic

    message = f"stage=Schedule: {code} at schedule/assembly: {strategy!r} {reason}"
    return message, LoopPlanDiagnostic(
        code=code,
        message=f"{strategy!r} {reason}",
        path=("schedule", "assembly"),
        stage="schedule",
        pass_name="apply_schedule",
    )


def _schedule_field_diagnostic(
    code: str, path: Tuple[str, ...], reason: str
) -> Tuple[str, object]:
    """One structured refusal of a schedule field, spelled like the others."""

    from .loop_plan_legality import LoopPlanDiagnostic

    full_path = ("schedule", *path)
    message = f"stage=Schedule: {code} at {'/'.join(full_path)}: {reason}"
    return message, LoopPlanDiagnostic(
        code=code,
        message=reason,
        path=full_path,
        stage="schedule",
        pass_name="apply_schedule",
    )


def _unsupported_assembly(strategy: str, reason: str) -> NoReturn:
    message, diagnostic = _assembly_diagnostic(
        "unsupported_schedule_assembly", strategy, reason
    )
    raise UnsupportedFeature(message, diagnostics=(diagnostic,))


def _invalid_assembly(strategy: str, reason: str) -> NoReturn:
    message, diagnostic = _assembly_diagnostic(
        "invalid_schedule_assembly", strategy, reason
    )
    raise InvalidSchedule(message, diagnostics=(diagnostic,))


def _accumulator_diagnostic(
    code: str, structure: str, reason: str
) -> Tuple[str, object]:
    """One structured accumulation-structure refusal, spelled like the others."""

    from .loop_plan_legality import LoopPlanDiagnostic

    message = f"stage=Schedule: {code} at schedule/accumulator: {structure!r} {reason}"
    return message, LoopPlanDiagnostic(
        code=code,
        message=f"{structure!r} {reason}",
        path=("schedule", "accumulator"),
        stage="schedule",
        pass_name="apply_schedule",
    )


def _unsupported_accumulator(structure: str, reason: str) -> NoReturn:
    message, diagnostic = _accumulator_diagnostic(
        "unsupported_schedule_accumulator", structure, reason
    )
    raise UnsupportedFeature(message, diagnostics=(diagnostic,))


def _validate_schedule_structure(schedule: object) -> Schedule:
    """Revalidate a frozen public schedule at every compiler use boundary."""

    _canonical_schedule(schedule)
    return cast(Schedule, schedule)


def _canonical_schedule(schedule: object) -> Schedule:
    """Validate ``schedule``'s stored state and return its canonical copy.

    The copy is rebuilt from the state read here, compares equal to ``schedule``, and
    is held by no caller.
    """

    state = _stored_schedule_fields(schedule, Schedule, _SCHEDULE_FIELDS, "Schedule")
    loop_order = state["loop_order"]
    tiles = state["tiles"]
    relayout = state["relayout"]
    tag = state["tag"]
    parallel_loop = state["parallel_loop"]
    assembly = state["assembly"]
    accumulator = state["accumulator"]
    vector_loops = state["vector_loops"]
    entries_of = state["entries_of"]
    if loop_order is not None and type(loop_order) is not tuple:
        raise InvalidSchedule(
            "Schedule.loop_order must be an owned tuple or None",
            code="schedule_state_invalid",
        )
    if type(tiles) is not tuple:
        raise InvalidSchedule(
            "Schedule.tiles must be an owned tuple", code="schedule_state_invalid"
        )
    if relayout is not None and type(relayout) is not RelayoutSpec:
        raise InvalidSchedule(
            "Schedule.relayout must be an exact RelayoutSpec or None",
            code="schedule_state_invalid",
        )
    if type(tag) is not str:
        raise InvalidSchedule(
            "Schedule.tag must be an exact str", code="schedule_state_invalid"
        )
    if parallel_loop is not None and type(parallel_loop) is not str:
        raise InvalidSchedule(
            "Schedule.parallel_loop must be an exact str or None",
            code="schedule_state_invalid",
        )
    if assembly is not None and not is_assembly_strategy(assembly):
        raise InvalidSchedule(
            "Schedule.assembly must name a sparse assembly strategy or be None",
            code="schedule_state_invalid",
        )
    if accumulator is not None and not is_accumulator_structure(accumulator):
        raise InvalidSchedule(
            "Schedule.accumulator must name a sparse accumulation structure "
            "or be None",
            code="schedule_state_invalid",
        )
    if vector_loops is not None and (
        type(vector_loops) is not tuple
        or any(type(name) is not str for name in vector_loops)
    ):
        raise InvalidSchedule(
            "Schedule.vector_loops must be an owned tuple of exact str or None",
            code="schedule_state_invalid",
        )
    if entries_of is not None and type(entries_of) is not str:
        raise InvalidSchedule(
            "Schedule.entries_of must be an exact str or None",
            code="schedule_state_invalid",
        )

    checked_tiles = []
    for position, tile in enumerate(tiles):
        tile_state = _stored_schedule_fields(
            tile, TileSpec, _TILE_SPEC_FIELDS, f"Schedule.tiles[{position}]"
        )
        try:
            checked_tiles.append(TileSpec(**tile_state))
        except (TypeError, ValueError) as error:
            raise InvalidSchedule(
                f"Schedule.tiles[{position}] has invalid stored values",
                code="schedule_state_invalid",
            ) from error

    checked_relayout = None
    if relayout is not None:
        relayout_state = _stored_schedule_fields(
            relayout, RelayoutSpec, _RELAYOUT_SPEC_FIELDS, "Schedule.relayout"
        )
        try:
            checked_relayout = RelayoutSpec(**relayout_state)
        except (TypeError, ValueError) as error:
            raise InvalidSchedule(
                "Schedule.relayout has invalid stored values",
                code="schedule_state_invalid",
            ) from error

    try:
        canonical = Schedule(
            loop_order=loop_order,
            tiles=tuple(checked_tiles),
            relayout=checked_relayout,
            tag=tag,
            parallel_loop=parallel_loop,
            assembly=assembly,
            accumulator=accumulator,
            vector_loops=vector_loops,
            entries_of=entries_of,
        )
    except (TypeError, ValueError) as error:
        raise InvalidSchedule(
            "Schedule has invalid stored values", code="schedule_state_invalid"
        ) from error
    if canonical != schedule:
        raise InvalidSchedule(
            "Schedule state is not in canonical owned form",
            code="schedule_state_invalid",
        )
    return canonical


def _entity_ids_by_name(
    cin: CIN,
) -> Tuple[Dict[str, IndexVar], Dict[str, TensorVar]]:
    """Resolve public display names once at the Schedule-to-LoopPlan boundary."""

    index_names, symbol_names = entity_display_names(cin)
    index_vars = {
        index_var.index_id: index_var
        for index_var in Scheduler.get_index_variables(cin)
    }
    for access in cin.tensor_accesses:
        for index_var in access.indices or ():
            index_vars.setdefault(index_var.index_id, index_var)

    indices_by_name: Dict[str, IndexVar] = {}
    for index_id, name in index_names.items():
        index_var = index_vars[index_id]
        previous = indices_by_name.get(name)
        if previous is not None and previous.index_id != index_id:
            raise ScorchValueError(
                f"Schedule name {name!r} is ambiguous between distinct IndexId values",
                code="schedule_name_ambiguous",
            )
        indices_by_name[name] = index_var

    symbols_by_name: Dict[str, TensorVar] = {}
    symbol_objects = {
        access.tensor.symbol_id: access.tensor for access in cin.tensor_accesses
    }
    for symbol_id, name in symbol_names.items():
        symbol = symbol_objects[symbol_id]
        previous = symbols_by_name.get(name)
        if previous is not None and previous.symbol_id != symbol_id:
            raise ScorchValueError(
                f"Schedule name {name!r} is ambiguous between distinct SymbolId values",
                code="schedule_name_ambiguous",
            )
        symbols_by_name[name] = symbol
    return indices_by_name, symbols_by_name


def _loop_ref_from_name(
    name: str,
    indices_by_name: Dict[str, IndexVar],
) -> LoopRef:
    index_var = indices_by_name.get(name)
    if index_var is not None:
        return LoopRef(index_var.index_id)
    for suffix, part in (("_out", LoopPart.OUTER), ("_in", LoopPart.INNER)):
        if name.endswith(suffix):
            index_var = indices_by_name.get(name[: -len(suffix)])
            if index_var is not None:
                return LoopRef(index_var.index_id, part)
    raise ScorchValueError(
        f"Schedule references unknown logical loop {name!r}",
        code="schedule_loop_unknown",
    )


def _loop_placement_from_public(
    placement: str,
    indices_by_name: Dict[str, IndexVar],
) -> LoopPlacement:
    if placement == "outermost":
        return LoopPlacement(PlacementKind.OUTERMOST)
    if placement.startswith("child_of:"):
        return LoopPlacement(
            PlacementKind.CHILD_OF,
            parent=_loop_ref_from_name(placement.split(":", 1)[1], indices_by_name),
        )
    if placement.startswith("at_depth:"):
        return LoopPlacement(
            PlacementKind.AT_DEPTH,
            depth=int(placement.split(":", 1)[1]),
        )
    raise ScorchValueError(
        f"Unsupported loop placement {placement!r}", code="tile_placement_invalid"
    )


def _build_loop_plan(
    cin: CIN,
    schedule: Schedule,
    logical_order: Sequence[IndexVar],
    panel_bounds: Sequence[PanelBound],
    relayout_plan: Optional[_RelayoutPlan],
    result_tile_plan: Optional[_ResultTilePlan],
    provenance: str,
) -> LoopPlan:
    indices_by_name, symbols_by_name = _entity_ids_by_name(cin)

    tiles = tuple(
        LoopTile(
            loop=_loop_ref_from_name(tile.index_var, indices_by_name),
            width=tile.width,
            placement=_loop_placement_from_public(tile.placement, indices_by_name),
            parallel=tile.parallel,
            kind=tile.kind,
            accumulation=tile.accum,
            unroll=tile.unroll,
        )
        for tile in schedule.tiles
    )

    relayout = None
    if relayout_plan is not None:
        if schedule.relayout is None:
            raise CompilerInvariantError(
                "validated relayout metadata has no public relayout decision",
                code="relayout_decision_missing",
            )
        relayout = OperandRelayout(
            operand_id=relayout_plan.operand_id,
            pack_loop=_loop_ref_from_name(relayout_plan.pack_var, indices_by_name),
            panel_loop=_loop_ref_from_name(relayout_plan.panel_var, indices_by_name),
            scope_loop=_loop_ref_from_name(relayout_plan.scope_var, indices_by_name),
            row_loop=_loop_ref_from_name(relayout_plan.row_var, indices_by_name),
            strip_width=schedule.relayout.strip_width,
            access_indices=relayout_plan.access_index_ids,
            operand_panel_level=relayout_plan.operand_panel_level,
            operand_pack_level=relayout_plan.operand_pack_level,
        )

    result_tile = None
    if result_tile_plan is not None:
        result_tile = ResultTile(
            result_id=result_tile_plan.result_id,
            tile_loop=_loop_ref_from_name(result_tile_plan.tile_var, indices_by_name),
            result_level=result_tile_plan.result_level,
            result_prefix=tuple(
                indices_by_name[name].index_id
                for name in result_tile_plan.result_prefix_vars
            ),
            access_indices=result_tile_plan.access_index_ids,
        )

    parallel_loop = (
        _loop_ref_from_name(schedule.parallel_loop, indices_by_name)
        if schedule.parallel_loop is not None
        else None
    )
    plan = LoopPlan(
        loop_order=tuple(index_var.index_id for index_var in logical_order),
        tiles=tiles,
        panel_bounds=tuple(panel_bounds),
        relayout=relayout,
        result_tile=result_tile,
        parallel_loop=parallel_loop,
        assembly=schedule.assembly,
        accumulator=schedule.accumulator,
        provenance=provenance,
        tag=schedule.tag,
        vector_loops=_vector_loop_refs(cin, schedule.vector_loops),
        entries_of=_entries_of_symbol(cin, schedule.entries_of),
    )
    return verify_loop_plan(cin, plan)


def _vector_loop_refs(
    cin: CIN, names: Optional[Tuple[str, ...]]
) -> Optional[Tuple[LoopRef, ...]]:
    """Resolve a public ``vector_loops`` request to stable loop references."""

    if names is None:
        return None
    indices_by_name, _ = _entity_ids_by_name(cin)
    refs: List[LoopRef] = []
    for position, name in enumerate(names):
        try:
            refs.append(_loop_ref_from_name(name, indices_by_name))
        except ValueError as error:
            message, diagnostic = _schedule_field_diagnostic(
                "invalid_vector_loops", ("vector_loops", str(position)), str(error)
            )
            raise InvalidSchedule(message, diagnostics=(diagnostic,)) from error
    return tuple(refs)


def _refuse_loop_requests_without_loops(schedule: Schedule) -> None:
    """A statement with no loops has no loop to mark and no loops to walk an
    input's entries: refuse either request with its code."""

    if schedule.vector_loops:
        message, diagnostic = _schedule_field_diagnostic(
            "invalid_vector_loops",
            ("vector_loops",),
            "names a loop of a statement that has no loops",
        )
        raise InvalidSchedule(message, diagnostics=(diagnostic,))
    if schedule.entries_of is not None:
        message, diagnostic = _schedule_field_diagnostic(
            "unsupported_entries_of",
            ("entries_of",),
            "names an input whose entries a statement with no loops cannot walk",
        )
        raise UnsupportedFeature(message, diagnostics=(diagnostic,))


def _with_vector_loop_request(
    cin: CIN, plan: LoopPlan, schedule: Schedule, dense_results_any_order: bool
) -> LoopPlan:
    """The automatic plan with the schedule's ``vector_loops``.

    Marks are independent of the order and tiles the origin chose, so they are
    attached to the verified plan rather than threaded through origination, and
    the plan boundary verifies them again.  ``entries_of`` is not attached
    here: the assembly's legality reads it, so origination records it with the
    assembly.  A schedule that names no vector loops returns the plan unchanged.
    """

    if schedule.vector_loops is None:
        return plan
    return verify_loop_plan(
        cin,
        replace(plan, vector_loops=_vector_loop_refs(cin, schedule.vector_loops)),
        dense_results_any_order=dense_results_any_order,
    )


def _entries_of_symbol(cin: CIN, name: Optional[str]) -> Optional[SymbolId]:
    """Resolve a public ``entries_of`` request to the named tensor's identity.

    Whether that tensor is an input whose stored entries the result's loops
    walk is plan legality's rule (``unsupported_entries_of``); here the name
    only has to name a tensor of the statement.
    """

    if name is None:
        return None
    _, symbols_by_name = _entity_ids_by_name(cin)
    symbol = symbols_by_name.get(name)
    if symbol is None:
        message, diagnostic = _schedule_field_diagnostic(
            "invalid_entries_of",
            ("entries_of",),
            f"{name!r} names no tensor of this statement",
        )
        raise InvalidSchedule(message, diagnostics=(diagnostic,))
    return symbol.symbol_id


def _render_loop_ref(loop: LoopRef, index_names: Dict[IndexId, str]) -> str:
    name = index_names[loop.index_id]
    if loop.part == LoopPart.OUTER:
        return f"{name}_out"
    if loop.part == LoopPart.INNER:
        return f"{name}_in"
    return name


def _loop_ref_from_legacy_index_var(index_var: IndexVar) -> LoopRef:
    """Recover the logical loop reference represented by a legacy tile part."""

    if index_var.has_parent:
        if index_var.is_outer:
            return LoopRef(index_var.parent.index_id, LoopPart.OUTER)
        if index_var.is_inner:
            return LoopRef(index_var.parent.index_id, LoopPart.INNER)
    return LoopRef(index_var.index_id)


def _render_placement(
    placement: LoopPlacement,
    index_names: Dict[IndexId, str],
) -> str:
    if placement.kind == PlacementKind.OUTERMOST:
        return "outermost"
    if placement.kind == PlacementKind.CHILD_OF:
        if placement.parent is None:
            raise VerificationError(
                "child placement has no parent loop", code="loop_placement_unrenderable"
            )
        return f"child_of:{_render_loop_ref(placement.parent, index_names)}"
    if placement.depth is None:
        raise VerificationError(
            "depth placement has no depth", code="loop_placement_unrenderable"
        )
    return f"at_depth:{placement.depth}"


def materialize_legacy_schedule(
    cin: CIN,
    plan: LoopPlan,
) -> Tuple[
    Schedule,
    Dict[str, str],
    Optional[_RelayoutPlan],
    Optional[_ResultTilePlan],
]:
    """Translate a verified LoopPlan only at the legacy LLIR compatibility seam."""

    verify_loop_plan(cin, plan)
    index_names, symbol_names = entity_display_names(cin)
    tiles = tuple(
        TileSpec(
            index_var=_render_loop_ref(tile.loop, index_names),
            width=tile.width,
            placement=_render_placement(tile.placement, index_names),
            parallel=tile.parallel,
            kind=tile.kind,
            accum=tile.accumulation,
            unroll=tile.unroll,
        )
        for tile in plan.tiles
    )

    relayout_spec = None
    relayout_plan = None
    if plan.relayout is not None:
        relayout = plan.relayout
        relayout_spec = RelayoutSpec(
            operand=symbol_names[relayout.operand_id],
            pack_var=_render_loop_ref(relayout.pack_loop, index_names),
            strip_width=relayout.strip_width,
            scope_var=_render_loop_ref(relayout.scope_loop, index_names),
        )
        relayout_plan = _RelayoutPlan(
            operand=symbol_names[relayout.operand_id],
            operand_id=relayout.operand_id,
            pack_var=_render_loop_ref(relayout.pack_loop, index_names),
            panel_var=_render_loop_ref(relayout.panel_loop, index_names),
            scope_var=_render_loop_ref(relayout.scope_loop, index_names),
            row_var=_render_loop_ref(relayout.row_loop, index_names),
            access_index_vars=tuple(index_names[i] for i in relayout.access_indices),
            access_index_ids=tuple(relayout.access_indices),
            operand_panel_level=relayout.operand_panel_level,
            operand_pack_level=relayout.operand_pack_level,
        )

    result_tile_plan = None
    if plan.result_tile is not None:
        result_tile = plan.result_tile
        result_tile_plan = _ResultTilePlan(
            result=symbol_names[result_tile.result_id],
            result_id=result_tile.result_id,
            tile_var=_render_loop_ref(result_tile.tile_loop, index_names),
            result_level=result_tile.result_level,
            result_prefix_vars=tuple(index_names[i] for i in result_tile.result_prefix),
            access_index_vars=tuple(index_names[i] for i in result_tile.access_indices),
            access_index_ids=tuple(result_tile.access_indices),
        )

    schedule = Schedule(
        loop_order=tuple(index_names[index_id] for index_id in plan.loop_order),
        tiles=tiles,
        relayout=relayout_spec,
        tag=plan.tag,
        parallel_loop=(
            _render_loop_ref(plan.parallel_loop, index_names)
            if plan.parallel_loop is not None
            else None
        ),
        assembly=plan.assembly,
        accumulator=plan.accumulator,
        vector_loops=(
            None
            if plan.vector_loops is None
            else tuple(
                _render_loop_ref(loop, index_names) for loop in plan.vector_loops
            )
        ),
        entries_of=(None if plan.entries_of is None else symbol_names[plan.entries_of]),
    )
    rendered_panel_bounds = {
        _render_loop_ref(bound.loop, index_names): (
            f"{symbol_names[bound.tensor_id]}{bound.level}_size"
        )
        for bound in plan.panel_bounds
    }
    return schedule, rendered_panel_bounds, relayout_plan, result_tile_plan


_SCHEDULE_FORCE: ContextVar[Optional[Schedule]] = ContextVar(
    "scorch_schedule_force", default=None
)

# Whether schedule_force has run in this process with a schedule: set by its first such
# entry, before the context variable, and never cleared. The readers that would otherwise
# read the variable on every call -- ops.matmul's plan check, autograd.py's routes,
# call_without_forced_schedule, and the C entry point through its own flag
# (planned_matmul.h, schedule_force_seen) -- read it only once this is set, so a process
# that never forces a schedule pays one list read per call. Set before the variable, so a
# context that holds a schedule was made after it was set. A count of open blocks would
# not do: a context copied inside one (an asyncio task, copy_context().run) keeps the
# schedule after the block exits. A one-element list, as plan.ENABLED_CELL, so that a
# module importing it reads the current value.
FORCE_SEEN_CELL = [False]
# Called by the first schedule_force, before FORCE_SEEN_CELL is set: ops.py adds the C
# entry point's setter here.
_ON_FIRST_SCHEDULE_FORCE: List[Callable[[], None]] = []


def _note_schedule_force() -> None:
    for hook in _ON_FIRST_SCHEDULE_FORCE:
        hook()
    FORCE_SEEN_CELL[0] = True


def get_forced_schedule() -> Optional[Schedule]:
    """Return the schedule active in this execution context, if any."""
    return _SCHEDULE_FORCE.get()


@contextmanager
def schedule_force(value: Optional[Schedule]):
    """Temporarily force a schedule for ``matmul``/``einsum`` in this context.

    Every product the caller writes in the context is computed by the generated kernel
    under ``value``, or refused with a coded error where that kernel cannot take it (a
    matrix times a vector: ScheduleNotThreadedThroughSpMVError). No product is served by a
    prebuilt kernel or a plan meanwhile, since neither takes a schedule. The schedule names
    the forward expression only: the products a derivative computes (a backward's
    ``A^T @ G``, an SDDMM, a forward-mode tangent) are other expressions and run under the
    default schedule, even when ``backward()`` is called inside the context
    (``call_without_forced_schedule``).
    """
    if value is not None and not isinstance(value, Schedule):
        raise TypeError("schedule_force expects a Schedule or None")
    if value is not None and not FORCE_SEEN_CELL[0]:
        _note_schedule_force()
    token = _SCHEDULE_FORCE.set(value)
    try:
        yield
    finally:
        _SCHEDULE_FORCE.reset(token)


_Result = TypeVar("_Result")


def call_without_forced_schedule(
    fn: Callable[..., _Result], *args: Any, **kwargs: Any
) -> _Result:
    """``fn(*args, **kwargs)`` with no schedule forced, whatever the calling context forces.

    A forced schedule names the forward expression its caller wrote. The products a
    derivative computes -- a backward's ``A^T @ G``, an SDDMM, a forward-mode tangent --
    are other expressions, so they run under the default schedule even when
    ``backward()`` is called inside ``schedule_force``. Every such product goes through
    this function: autograd._matmul's, and einsum_grad._einsum's for the einsum's
    derivative. One list read in a process that has never
    forced a schedule (FORCE_SEEN_CELL), one context-variable read in one that has, and a
    set and a reset where one is forced.
    """
    if not FORCE_SEEN_CELL[0] or _SCHEDULE_FORCE.get() is None:
        return fn(*args, **kwargs)
    token = _SCHEDULE_FORCE.set(None)
    try:
        return fn(*args, **kwargs)
    finally:
        _SCHEDULE_FORCE.reset(token)


class Scheduler:
    """
    Auto-schedules CIN statements.
    """

    _DEFAULT_COSTS = _CostModelConstants()

    def __init__(self):
        pass

    @staticmethod
    def _is_sparse_level(level_type: LevelType) -> bool:
        return level_type in (LevelType.COMPRESSED, LevelType.COORDINATE)

    @staticmethod
    def _unique_index_vars(index_vars: List[IndexVar]) -> List[IndexVar]:
        seen: Set[IndexId] = set()
        unique: List[IndexVar] = []
        for index_var in index_vars:
            if index_var.index_id in seen:
                continue
            seen.add(index_var.index_id)
            unique.append(index_var)
        return unique

    @staticmethod
    def _has_dense_output(cin: CIN) -> bool:
        body = cin
        while isinstance(body, ForAll):
            body = body.stmt
        if isinstance(body, TensorAssign):
            return body.lhs.get_tensor().is_dense()
        return False

    @staticmethod
    def _has_proven_non_dense_receiver(cin: CIN) -> bool:
        """Is the FINAL result receiver provably NOT dense?

        ``_has_dense_output`` cannot answer this once a workspace has been
        inserted: it walks the ``ForAll`` chain looking for a ``TensorAssign``,
        and insertion puts a ``Where`` there, so past that point it answers
        ``False`` for *every* receiver -- dense ones included.  Anything using it
        as a post-insertion legality guard therefore silently disables itself
        everywhere rather than firing where it means to.

        The receiver the tiling-legality question is about is the single
        non-``Workspace`` result tensor.  The polarity here is deliberate: this
        returns ``True`` only when there is exactly one such receiver *and* its
        format is known and non-dense.  Anything unrecognized answers ``False``
        and leaves the existing behaviour alone, so the guard can only ever
        subtract tiling from a shape it has positively identified.
        """

        receivers = [
            var
            for var in cin.get_result_tensor_vars()
            if not isinstance(var, Workspace)
        ]
        if len(receivers) != 1:
            return False
        result_format = receivers[0].format
        if result_format is None:
            return False
        return not result_format.is_dense()

    @staticmethod
    def _require_legal_assembly_request(cin: CIN, schedule: Schedule) -> None:
        """State the explicit path's assembly-strategy legality rules.

        The explicit path states its own legality rules and the automatic origin
        is then held to them -- the ordering §57.5's blocker-1 fix established,
        after fixing only one of two duplicated layers landed twelve cells on the
        wrong code.  There are two rules, and both are structural:

        1. every strategy except the always-legal serial one needs the shared
           receiver contract -- a dense level zero with every level below it
           compressed -- because that is the correspondence that lets the
           assembly be split by outer cell at all; the parallel single pass
           also admits a COO result that stores one input's entries
           (``Schedule.entries_of``), whose entries' positions are known
           before the loops run;
        2. a requested strategy owns the result's assembly, so it cannot compose
           with a panel tile, a relayout, or an explicit parallel loop, each of
           which owns part of the same assembly.

        A request that cannot be honoured is REFUSED rather than silently
        downgraded to the default: a user who asks for a strategy and gets
        another one has no way to know.  That is the difference between this and
        the target's own choice, which declines silently because declining there
        means "keep emitting exactly what you emit today".

        The refusals carry the same structured codes the plan legality boundary
        raises (``unsupported_schedule_assembly`` /
        ``invalid_schedule_assembly``), and the receiver rule itself is the one
        shared predicate rather than a restatement of it -- the two boundaries
        differ in WHEN they refuse, never in WHAT they refuse.

        Extents, densities and thread counts are deliberately absent.  Whether a
        legal strategy pays is cost, and cost is the selector's question.
        """

        strategy = schedule.assembly
        if strategy is None or strategy == "single_pass_serial":
            return
        receivers = [
            var
            for var in cin.get_result_tensor_vars()
            if not isinstance(var, Workspace)
        ]
        if len(receivers) != 1 or receivers[0].format is None:
            _unsupported_assembly(
                strategy,
                "requires exactly one result tensor with a known format",
            )
        levels = tuple(receivers[0].format.get_level_types())
        if strategy == "single_pass_chunk_parallel" and entry_positioned_receiver(
            levels, schedule.entries_of is not None
        ):
            # A COO result that stores one input's entries.  Whether the final
            # loop order walks that input is the rest of the rule
            # (``sparse_assembly.entry_source``), which plan legality checks once
            # the order is known.
            levels_admitted = True
        else:
            levels_admitted = partitionable_receiver_levels(levels)
        if not levels_admitted:
            _unsupported_assembly(
                strategy,
                f"requires {receiver_requirement(strategy)}; "
                f"{receivers[0].get_name()!r} is not one",
            )
        if (
            any(tile.kind == "panel" for tile in schedule.tiles)
            or schedule.relayout is not None
            or schedule.parallel_loop is not None
            or any(tile.parallel for tile in schedule.tiles)
        ):
            _invalid_assembly(
                strategy,
                "owns the result's assembly and its own parallel policy; it "
                "cannot compose with a sparse panel tile, an operand relayout, "
                "or an explicit parallel loop selection",
            )

    @staticmethod
    def _require_legal_accumulator_request(cin: CIN, schedule: Schedule) -> None:
        """State the explicit path's accumulation-structure legality rule.

        One rule, and it is the only one provable without the emitted body: an
        accumulation structure is a property of a SPARSE reduction's workspace,
        and a dense result has no sparse accumulation workspace to hold.  So a
        recorded structure requires exactly one result tensor with a known
        format that is not dense.

        Everything else this decision can be refused for needs a fact this
        boundary does not have -- the workspace's key rank, and whether the
        emitted body puts the declaration where the two-phase transform can
        reach it -- so those refusals live in target lowering and in the pass,
        each where its input lives.  The two boundaries differ in WHEN they
        refuse, never in WHAT.

        A request that cannot be honoured is REFUSED rather than ignored, for
        the reason the assembly request is: a caller who asks for one structure
        and gets another has no way to find out.

        Extents and densities are deliberately absent.  Whether a legal
        structure pays is cost, and cost is the selector's question.
        """

        structure = schedule.accumulator
        if structure is None:
            return
        receivers = [
            var
            for var in cin.get_result_tensor_vars()
            if not isinstance(var, Workspace)
        ]
        if len(receivers) != 1 or receivers[0].format is None:
            _unsupported_accumulator(
                structure,
                "requires exactly one result tensor with a known format",
            )
        if receivers[0].format.is_dense():
            _unsupported_accumulator(
                structure,
                "names the structure a sparse reduction accumulates through; "
                f"{receivers[0].get_name()!r} is dense and has no sparse "
                "accumulation workspace",
            )

    @staticmethod
    def _extract_loop_chain(cin: CIN) -> Tuple[List[IndexVar], CIN]:
        loop_order: List[IndexVar] = []
        body: CIN = cin
        while isinstance(body, ForAll):
            loop_order.append(body.index_var)
            body = body.stmt
        return loop_order, body

    @staticmethod
    def _rebuild_loop_nest(cin: CIN, loop_order: List[IndexVar]) -> CIN:
        if not isinstance(cin, ForAll):
            return cin

        _, body = Scheduler._extract_loop_chain(cin)
        rebuilt: CIN = body
        for index_var in reversed(loop_order):
            rebuilt = ForAll(index_var=index_var, stmt=rebuilt)

        rebuilt.inserted_workspace = cin.inserted_workspace
        rebuilt.no_tile_list = list(cin.no_tile_list)
        return rebuilt

    @staticmethod
    def get_index_variables(cin: CIN) -> List[IndexVar]:
        loop_order, _ = Scheduler._extract_loop_chain(cin)
        if loop_order:
            return Scheduler._unique_index_vars(loop_order)
        return sorted(
            Scheduler._unique_index_vars(cin.index_vars),
            key=lambda index_var: index_var.name,
        )

    @staticmethod
    def _get_rhs_tensor_accesses(cin: CIN) -> List[TensorAccess]:
        if hasattr(cin, "get_rhs_tensor_accesses"):
            return cin.get_rhs_tensor_accesses()
        return cin.tensor_accesses

    @staticmethod
    def _estimate_index_selectivity(
        index_var: IndexVar,
        tensor_accesses: List[TensorAccess],
        costs: _CostModelConstants,
    ) -> float:
        selectivity = 1.0
        for tensor_access in tensor_accesses:
            if not tensor_access.has_index_var(index_var):
                continue
            level_type = tensor_access.level_type_of_index_var(index_var)
            if Scheduler._is_sparse_level(level_type):
                selectivity *= costs.rho
        return max(selectivity, 1e-12)

    @staticmethod
    def _is_sparse_filter(
        index_var: IndexVar,
        tensor_accesses: List[TensorAccess],
    ) -> bool:
        has_sparse_level = False
        has_dense_level = False
        missing_from_some_tensor = False

        for tensor_access in tensor_accesses:
            if not tensor_access.has_index_var(index_var):
                missing_from_some_tensor = True
                continue

            level_type = tensor_access.level_type_of_index_var(index_var)
            if Scheduler._is_sparse_level(level_type):
                has_sparse_level = True
            else:
                has_dense_level = True

        return has_sparse_level and (has_dense_level or missing_from_some_tensor)

    @staticmethod
    def _sparse_filter_score(
        index_var: IndexVar,
        rhs_tensor_accesses: List[TensorAccess],
        costs: _CostModelConstants,
    ) -> float:
        if not Scheduler._is_sparse_filter(index_var, rhs_tensor_accesses):
            return 0.0
        selectivity = Scheduler._estimate_index_selectivity(
            index_var=index_var,
            tensor_accesses=rhs_tensor_accesses,
            costs=costs,
        )
        return 1.0 - selectivity

    @staticmethod
    def _mode_position_score(
        index_var: IndexVar,
        tensor_accesses: List[TensorAccess],
    ) -> float:
        levels = [
            tensor_access.level_of_index_var(index_var)
            for tensor_access in tensor_accesses
            if tensor_access.has_index_var(index_var)
        ]
        if not levels:
            return float("inf")
        return sum(levels) / len(levels)

    @staticmethod
    def sort_by_sparsity_descending(
        index_vars: List[IndexVar],
        cin: CIN,
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> List[IndexVar]:
        rhs_tensor_accesses = Scheduler._get_rhs_tensor_accesses(cin)
        all_tensor_accesses = cin.tensor_accesses
        base_order = {
            index_var: idx
            for idx, index_var in enumerate(Scheduler._unique_index_vars(index_vars))
        }

        sparse_scores = {
            index_var: Scheduler._sparse_filter_score(
                index_var=index_var,
                rhs_tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            for index_var in index_vars
        }

        return sorted(
            index_vars,
            key=lambda index_var: (
                -sparse_scores[index_var],
                Scheduler._mode_position_score(index_var, all_tensor_accesses),
                base_order[index_var],
                index_var.name,
            ),
        )

    @staticmethod
    def init_loop_order(
        cin: CIN,
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> List[IndexVar]:
        index_vars = Scheduler.get_index_variables(cin)
        return Scheduler.sort_by_sparsity_descending(
            index_vars=index_vars,
            cin=cin,
            costs=costs,
        )

    @staticmethod
    def move_to_position(
        loop_order: List[IndexVar],
        index_var: IndexVar,
        pos: int,
    ) -> List[IndexVar]:
        if index_var not in loop_order:
            return loop_order[:]

        new_loop_order = list(loop_order)
        current_pos = new_loop_order.index(index_var)
        elem = new_loop_order.pop(current_pos)
        new_pos = max(0, min(pos, len(new_loop_order)))
        new_loop_order.insert(new_pos, elem)
        return new_loop_order

    @staticmethod
    def _estimate_index_extent(
        index_var: IndexVar,
        tensor_accesses: List[TensorAccess],
        costs: _CostModelConstants,
    ) -> float:
        sizes: List[float] = []
        for tensor_access in tensor_accesses:
            if not tensor_access.has_index_var(index_var):
                continue
            tensor_shape = tensor_access.get_tensor().shape
            if tensor_shape is None:
                continue

            logical_pos = tensor_access.get_index_vars().index(index_var)
            mode_order = tensor_access.get_tensor().mode_order
            physical_pos = logical_pos
            if mode_order is not None:
                try:
                    physical_pos = mode_order.index(logical_pos)
                except ValueError:
                    # The normalized-CIN verifier owns the malformed
                    # permutation diagnostic.  A cost heuristic must not
                    # leak a raw lookup error before that boundary.
                    continue
            if physical_pos < len(tensor_shape):
                sizes.append(float(tensor_shape[physical_pos]))

        if sizes:
            return max(max(sizes), 1.0)
        return float(costs.default_dim_size)

    @staticmethod
    def _estimate_tensor_nnz(tensor, costs: _CostModelConstants) -> float:
        level_types = tensor.get_level_types()
        if tensor.shape:
            shape = list(tensor.shape)
        else:
            shape = []

        if len(shape) < len(level_types):
            shape = shape + [costs.default_dim_size] * (len(level_types) - len(shape))

        mode_order = (
            tensor.mode_order if tensor.mode_order else list(range(len(level_types)))
        )

        nnz = 1.0
        for level, level_type in enumerate(level_types):
            logical_dim = mode_order[level] if level < len(mode_order) else level
            dim_size = (
                float(shape[logical_dim])
                if logical_dim < len(shape)
                else float(costs.default_dim_size)
            )
            density = costs.rho if Scheduler._is_sparse_level(level_type) else 1.0
            nnz *= max(dim_size, 1.0) * density
        return max(nnz, 1.0)

    @staticmethod
    def _compute_comp_cost(
        cin: CIN,
        loop_order: List[IndexVar],
        costs: _CostModelConstants,
    ) -> float:
        rhs_tensor_accesses = Scheduler._get_rhs_tensor_accesses(cin)
        all_tensor_accesses = cin.tensor_accesses

        index_extents = {
            index_var: Scheduler._estimate_index_extent(
                index_var=index_var,
                tensor_accesses=all_tensor_accesses,
                costs=costs,
            )
            for index_var in loop_order
        }
        index_selectivities = {
            index_var: Scheduler._estimate_index_selectivity(
                index_var=index_var,
                tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            for index_var in loop_order
        }

        sparse_filters = {
            index_var
            for index_var in loop_order
            if Scheduler._sparse_filter_score(
                index_var=index_var,
                rhs_tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            > 0
        }

        comp_cost = 1.0
        for idx, index_var in enumerate(loop_order):
            if index_var in sparse_filters:
                applicable_filter = index_selectivities[index_var]
            else:
                applicable_filter = 1.0
                for sf in loop_order[:idx]:
                    if sf not in sparse_filters:
                        continue
                    for ta in rhs_tensor_accesses:
                        if (
                            ta.has_index_var(sf)
                            and ta.has_index_var(index_var)
                            and ta.get_parent_index_var(index_var) == sf
                            and Scheduler._is_sparse_level(
                                ta.level_type_of_index_var(index_var)
                            )
                        ):
                            applicable_filter *= index_selectivities[sf]
                            break
            effective_iters = index_extents[index_var] * applicable_filter
            comp_cost *= max(effective_iters, 1e-12)
        return comp_cost

    @staticmethod
    def _compute_workspace_cost(
        cin: CIN,
        loop_order: List[IndexVar],
        costs: _CostModelConstants,
    ) -> float:
        if Scheduler._has_dense_output(cin):
            return 0.0

        cin_ivar_getter = CINIndexVariablesGetter()
        cin_ivar_getter.visit(cin)

        reduction_vars = Scheduler._unique_index_vars(
            cin_ivar_getter.get_reduction_vars()
        )
        free_vars = Scheduler._unique_index_vars(cin_ivar_getter.get_free_vars())
        if not reduction_vars:
            return 0.0

        loop_pos = {index_var: pos for pos, index_var in enumerate(loop_order)}
        reduction_vars_in_loop = [
            index_var for index_var in reduction_vars if index_var in loop_pos
        ]
        if not reduction_vars_in_loop:
            return 0.0

        last_reduction_pos = max(
            loop_pos[index_var] for index_var in reduction_vars_in_loop
        )
        free_after_last_reduction = [
            index_var
            for index_var in free_vars
            if index_var in loop_pos and loop_pos[index_var] > last_reduction_pos
        ]
        dim_workspace = len(free_after_last_reduction)
        if dim_workspace == 0:
            return 0.0

        rhs_tensor_accesses = Scheduler._get_rhs_tensor_accesses(cin)
        all_tensor_accesses = cin.tensor_accesses
        index_extents = {
            index_var: Scheduler._estimate_index_extent(
                index_var=index_var,
                tensor_accesses=all_tensor_accesses,
                costs=costs,
            )
            for index_var in loop_order
        }
        sparse_filters = {
            index_var
            for index_var in loop_order
            if Scheduler._sparse_filter_score(
                index_var=index_var,
                rhs_tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            > 0
        }
        index_selectivities = {
            index_var: Scheduler._estimate_index_selectivity(
                index_var=index_var,
                tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            for index_var in loop_order
        }

        n_insert = 1.0
        for pos, index_var in enumerate(loop_order):
            if index_var in sparse_filters:
                applicable_filter = index_selectivities[index_var]
            else:
                applicable_filter = 1.0
                for sf in loop_order[:pos]:
                    if sf not in sparse_filters:
                        continue
                    for ta in rhs_tensor_accesses:
                        if (
                            ta.has_index_var(sf)
                            and ta.has_index_var(index_var)
                            and ta.get_parent_index_var(index_var) == sf
                            and Scheduler._is_sparse_level(
                                ta.level_type_of_index_var(index_var)
                            )
                        ):
                            applicable_filter *= index_selectivities[sf]
                            break
            n_insert *= max(index_extents[index_var] * applicable_filter, 1.0)
            if pos >= last_reduction_pos:
                break

        n_entries = 1.0
        for index_var in free_after_last_reduction:
            n_entries *= max(index_extents[index_var], 1.0)

        insert_term = costs.c_insert * n_insert * dim_workspace
        sort_term = (
            costs.c_sort * n_entries * dim_workspace * math.log(max(n_entries, 2.0), 2)
        )
        return insert_term + sort_term

    @staticmethod
    def _compute_transposition_cost(
        cin: CIN,
        loop_order: List[IndexVar],
        costs: _CostModelConstants,
    ) -> float:
        loop_pos = {index_var: pos for pos, index_var in enumerate(loop_order)}
        needs_transpose: Dict[str, bool] = {}
        tensor_nnz: Dict[str, float] = {}

        for tensor_access in cin.tensor_accesses:
            if tensor_access.is_workspace():
                continue

            sorted_index_vars = [
                index_var
                for index_var in tensor_access.get_sorted_index_vars()
                if index_var in loop_pos
            ]
            if len(sorted_index_vars) < 2:
                continue

            # Dense tensors can be accessed in any order via index arithmetic
            # (e.g., A[i*N + j]). No physical data restructuring is needed.
            if all(
                not Scheduler._is_sparse_level(
                    tensor_access.level_type_of_index_var(iv)
                )
                for iv in sorted_index_vars
            ):
                continue

            violates = any(
                loop_pos[sorted_index_vars[i]] > loop_pos[sorted_index_vars[i + 1]]
                for i in range(len(sorted_index_vars) - 1)
            )
            tensor = tensor_access.get_tensor()
            tensor_name = tensor.name
            needs_transpose[tensor_name] = (
                needs_transpose.get(tensor_name, False) or violates
            )
            if tensor_name not in tensor_nnz:
                tensor_nnz[tensor_name] = Scheduler._estimate_tensor_nnz(tensor, costs)

        return sum(
            costs.c_trans * tensor_nnz[tensor_name]
            for tensor_name, transpose in needs_transpose.items()
            if transpose
        )

    @staticmethod
    def loop_order_cost(
        cin: CIN,
        loop_order: Sequence[IndexId],
        *,
        transposed: Optional[CIN] = None,
        compile_options: Optional[CompileOptions] = None,
    ) -> float:
        """The cost model's objective for visiting ``cin`` in ``loop_order``.

        ``alpha * computation + beta * workspace + gamma * transposition``, the
        quantity :meth:`cost_to_push` compares between two orders.  The
        computation and workspace terms are those of ``cin``; the
        transposition term counts the tensors of ``transposed`` (``cin`` when
        omitted) stored against ``loop_order``.  A plan that converts operands
        runs its kernel on the converted program and pays for converting the
        original program's operands, so it passes the converted program as
        ``cin`` and the original as ``transposed``.  Neither program is
        modified.
        """

        _, costs = _scheduler_costs_at_boundary(None, compile_options)

        def in_order(program: CIN) -> Tuple[CIN, List[IndexVar]]:
            working = copy.deepcopy(program)
            variables: Dict[IndexId, IndexVar] = {}
            stmt = working
            while isinstance(stmt, ForAll):
                variables[stmt.index_var.index_id] = stmt.index_var
                stmt = stmt.stmt
            return working, [variables[index_id] for index_id in loop_order]

        main, main_order = in_order(cin)
        source, source_order = in_order(cin if transposed is None else transposed)
        computation = Scheduler._compute_comp_cost(
            cin=main, loop_order=main_order, costs=costs
        )
        workspace = Scheduler._compute_workspace_cost(
            cin=main, loop_order=main_order, costs=costs
        )
        transposition = Scheduler._compute_transposition_cost(
            cin=source, loop_order=source_order, costs=costs
        )
        return (
            costs.alpha * computation
            + costs.beta * workspace
            + costs.gamma * transposition
        )

    @staticmethod
    def cost_to_push(
        cin: CIN,
        loop_order: List[IndexVar],
        index_var: IndexVar,
        pos: int,
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> float:
        if index_var not in loop_order:
            return 0.0

        new_loop_order = Scheduler.move_to_position(loop_order, index_var, pos)
        if new_loop_order == loop_order:
            return 0.0

        comp_cost_before = Scheduler._compute_comp_cost(
            cin=cin, loop_order=loop_order, costs=costs
        )
        comp_cost_after = Scheduler._compute_comp_cost(
            cin=cin, loop_order=new_loop_order, costs=costs
        )
        ws_cost_before = Scheduler._compute_workspace_cost(
            cin=cin, loop_order=loop_order, costs=costs
        )
        ws_cost_after = Scheduler._compute_workspace_cost(
            cin=cin, loop_order=new_loop_order, costs=costs
        )
        trans_cost_before = Scheduler._compute_transposition_cost(
            cin=cin, loop_order=loop_order, costs=costs
        )
        trans_cost_after = Scheduler._compute_transposition_cost(
            cin=cin, loop_order=new_loop_order, costs=costs
        )

        delta_comp = comp_cost_after - comp_cost_before
        delta_ws = ws_cost_after - ws_cost_before
        delta_trans = trans_cost_after - trans_cost_before
        return (
            costs.alpha * delta_comp + costs.beta * delta_ws + costs.gamma * delta_trans
        )

    @staticmethod
    def optimize_loop_order(
        cin: CIN,
        loop_order: List[IndexVar],
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> List[IndexVar]:
        rhs_tensor_accesses = Scheduler._get_rhs_tensor_accesses(cin)
        sparse_filters_ordered = [
            index_var
            for index_var in Scheduler.sort_by_sparsity_descending(
                index_vars=loop_order,
                cin=cin,
                costs=costs,
            )
            if Scheduler._sparse_filter_score(
                index_var=index_var,
                rhs_tensor_accesses=rhs_tensor_accesses,
                costs=costs,
            )
            > 0
        ]

        optimized_order = list(loop_order)
        for index_var in sparse_filters_ordered:
            if index_var not in optimized_order:
                continue

            current_pos = optimized_order.index(index_var)
            for pos in range(current_pos + 1, len(optimized_order) + 1):
                push_cost = Scheduler.cost_to_push(
                    cin=cin,
                    loop_order=optimized_order,
                    index_var=index_var,
                    pos=pos,
                    costs=costs,
                )
                if push_cost < 0:
                    optimized_order = Scheduler.move_to_position(
                        loop_order=optimized_order,
                        index_var=index_var,
                        pos=pos,
                    )
                    current_pos = optimized_order.index(index_var)

        return optimized_order

    @staticmethod
    def _build_mode_order_graph(
        index_vars: List[IndexVar],
        tensor_accesses: List[TensorAccess],
        costs: _CostModelConstants,
    ) -> Tuple[
        Dict[IndexVar, Set[IndexVar]],
        Dict[Tuple[IndexVar, IndexVar], Set[str]],
        Dict[str, float],
    ]:
        adjacency: Dict[IndexVar, Set[IndexVar]] = {
            index_var: set() for index_var in index_vars
        }
        edge_to_tensor_names: Dict[Tuple[IndexVar, IndexVar], Set[str]] = defaultdict(
            set
        )
        tensor_nnz: Dict[str, float] = {}
        allowed_index_vars = set(index_vars)

        for tensor_access in tensor_accesses:
            if tensor_access.is_workspace():
                continue

            sorted_index_vars = [
                index_var
                for index_var in tensor_access.get_sorted_index_vars()
                if index_var in allowed_index_vars
            ]
            if len(sorted_index_vars) < 2:
                continue

            # Dense tensors support any access order via index arithmetic;
            # mode-order constraints are only hard requirements for sparse levels.
            if all(
                not Scheduler._is_sparse_level(
                    tensor_access.level_type_of_index_var(iv)
                )
                for iv in sorted_index_vars
            ):
                continue

            tensor = tensor_access.get_tensor()
            tensor_name = tensor.name
            if tensor_name not in tensor_nnz:
                tensor_nnz[tensor_name] = Scheduler._estimate_tensor_nnz(tensor, costs)

            for i in range(len(sorted_index_vars) - 1):
                src = sorted_index_vars[i]
                dst = sorted_index_vars[i + 1]
                if src == dst:
                    continue
                adjacency[src].add(dst)
                edge_to_tensor_names[(src, dst)].add(tensor_name)

        return adjacency, edge_to_tensor_names, tensor_nnz

    @staticmethod
    def _contains_cycles(
        adjacency: Dict[IndexVar, Set[IndexVar]],
        index_vars: List[IndexVar],
    ) -> bool:
        indegree: Dict[IndexVar, int] = {index_var: 0 for index_var in index_vars}
        for src, dsts in adjacency.items():
            for dst in dsts:
                if dst in indegree:
                    indegree[dst] += 1

        queue = deque(
            [index_var for index_var in index_vars if indegree[index_var] == 0]
        )
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for dst in sorted(adjacency.get(node, set()), key=lambda var: var.name):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)

        return visited != len(index_vars)

    @staticmethod
    def _find_cycle_edges(
        adjacency: Dict[IndexVar, Set[IndexVar]],
        index_vars: List[IndexVar],
    ) -> List[Tuple[IndexVar, IndexVar]]:
        visited: Set[IndexVar] = set()
        in_stack: Set[IndexVar] = set()
        stack: List[IndexVar] = []

        def dfs(node: IndexVar) -> List[Tuple[IndexVar, IndexVar]]:
            visited.add(node)
            in_stack.add(node)
            stack.append(node)

            for neighbor in sorted(
                adjacency.get(node, set()), key=lambda var: var.name
            ):
                if neighbor not in visited:
                    cycle_edges = dfs(neighbor)
                    if cycle_edges:
                        return cycle_edges
                elif neighbor in in_stack:
                    cycle_start = stack.index(neighbor)
                    cycle_nodes = stack[cycle_start:] + [neighbor]
                    return [
                        (cycle_nodes[i], cycle_nodes[i + 1])
                        for i in range(len(cycle_nodes) - 1)
                    ]

            stack.pop()
            in_stack.remove(node)
            return []

        for index_var in sorted(index_vars, key=lambda var: var.name):
            if index_var in visited:
                continue
            cycle_edges = dfs(index_var)
            if cycle_edges:
                return cycle_edges
        return []

    @staticmethod
    def _remove_cheapest_cycle_edge(
        adjacency: Dict[IndexVar, Set[IndexVar]],
        cycle_edges: List[Tuple[IndexVar, IndexVar]],
        edge_to_tensor_names: Dict[Tuple[IndexVar, IndexVar], Set[str]],
        tensor_nnz: Dict[str, float],
    ) -> Tuple[IndexVar, IndexVar]:
        def edge_cost(edge: Tuple[IndexVar, IndexVar]) -> float:
            tensor_names = edge_to_tensor_names.get(edge, set())
            if not tensor_names:
                return float("inf")
            return sum(tensor_nnz.get(tensor_name, 1.0) for tensor_name in tensor_names)

        edge_to_remove = min(
            cycle_edges,
            key=lambda edge: (edge_cost(edge), edge[0].name, edge[1].name),
        )
        src, dst = edge_to_remove
        adjacency[src].remove(dst)
        return edge_to_remove

    @staticmethod
    def _topological_sort_with_priority(
        adjacency: Dict[IndexVar, Set[IndexVar]],
        index_vars: List[IndexVar],
        priority: Dict[IndexVar, int],
    ) -> List[IndexVar]:
        indegree: Dict[IndexVar, int] = {index_var: 0 for index_var in index_vars}
        for src, dsts in adjacency.items():
            for dst in dsts:
                if dst in indegree:
                    indegree[dst] += 1

        zero_indegree = [
            index_var for index_var in index_vars if indegree[index_var] == 0
        ]
        order: List[IndexVar] = []
        while zero_indegree:
            zero_indegree.sort(
                key=lambda index_var: (
                    priority.get(index_var, len(priority)),
                    index_var.name,
                )
            )
            node = zero_indegree.pop(0)
            order.append(node)
            for dst in sorted(adjacency.get(node, set()), key=lambda var: var.name):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    zero_indegree.append(dst)

        if len(order) < len(index_vars):
            remaining = [
                index_var for index_var in index_vars if index_var not in order
            ]
            remaining.sort(
                key=lambda index_var: (
                    priority.get(index_var, len(priority)),
                    index_var.name,
                )
            )
            order.extend(remaining)

        return order

    @staticmethod
    def apply_mode_order_constraints(
        cin: CIN,
        loop_order: List[IndexVar],
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> List[IndexVar]:
        if not loop_order:
            return loop_order

        unique_loop_order = Scheduler._unique_index_vars(loop_order)
        adjacency, edge_to_tensor_names, tensor_nnz = Scheduler._build_mode_order_graph(
            index_vars=unique_loop_order,
            tensor_accesses=cin.tensor_accesses,
            costs=costs,
        )

        while Scheduler._contains_cycles(adjacency, unique_loop_order):
            cycle_edges = Scheduler._find_cycle_edges(adjacency, unique_loop_order)
            if not cycle_edges:
                break
            Scheduler._remove_cheapest_cycle_edge(
                adjacency=adjacency,
                cycle_edges=cycle_edges,
                edge_to_tensor_names=edge_to_tensor_names,
                tensor_nnz=tensor_nnz,
            )

        priority = {index_var: pos for pos, index_var in enumerate(unique_loop_order)}
        return Scheduler._topological_sort_with_priority(
            adjacency=adjacency,
            index_vars=unique_loop_order,
            priority=priority,
        )

    @staticmethod
    def _sparse_result_workspace_loop(
        cin: CIN, loop_order: Sequence[IndexVar], reduction_vars: Sequence[IndexVar]
    ) -> Optional[IndexVar]:
        """The reduction loop a sparse result's workspace ``Where`` must replace.

        A compressed result is appended to in iteration order, so each of its
        coordinates must be written once, after everything that contributes to it
        has been summed. That holds when the workspace's ``Where`` replaces the first
        loop after the longest prefix of ``loop_order`` that walks the result's levels
        in storage order: every loop outside it then fixes one result prefix, and
        every reduction loop is inside it. Wrapping only the last reduction loop --
        the rule for a dense result, which is written by random access -- leaves any
        earlier reduction loop outside the ``Where``, which then drains the same
        coordinates once per iteration of that loop (``einsum('ijk->k')`` over a
        compressed tensor stored each ``k`` once per ``i``). With a single reduction
        loop the two rules pick the same loop.

        None for a dense result, and when the first loop after the prefix is not a
        reduction loop (the loop order visits the result's own variables out of
        storage order, which this rule does not decide); the caller then keeps the
        last reduction loop.
        """

        result_access = next(
            (
                access
                for access in cin.get_result_tensor_accesses()
                if not isinstance(access.tensor, Workspace)
            ),
            None,
        )
        if result_access is None or result_access.get_tensor().is_dense():
            return None
        levels = result_access.get_sorted_index_vars()
        depth = 0
        while (
            depth < len(loop_order)
            and depth < len(levels)
            and loop_order[depth] == levels[depth]
        ):
            depth += 1
        if depth == len(loop_order):
            return None
        enclosed = loop_order[depth]
        return enclosed if enclosed in reduction_vars else None

    @staticmethod
    def should_insert_workspace(
        cin: CIN,
        loop_order: List[IndexVar],
    ) -> bool:
        if cin.inserted_workspace:
            return False
        if not isinstance(cin, ForAll):
            return False

        cin_ivar_getter = CINIndexVariablesGetter()
        cin_ivar_getter.visit(cin)

        reduction_vars = Scheduler._unique_index_vars(
            cin_ivar_getter.get_reduction_vars()
        )
        if not reduction_vars:
            return False

        free_vars = Scheduler._unique_index_vars(cin_ivar_getter.get_free_vars())
        if not free_vars:
            return False

        loop_pos = {index_var: pos for pos, index_var in enumerate(loop_order)}
        reductions_in_loop = [
            index_var for index_var in reduction_vars if index_var in loop_pos
        ]
        if not reductions_in_loop:
            return False

        last_reduction_pos = max(
            loop_pos[index_var] for index_var in reductions_in_loop
        )
        enclosed = Scheduler._sparse_result_workspace_loop(
            cin, loop_order, reductions_in_loop
        )
        if enclosed is not None:
            last_reduction_pos = loop_pos[enclosed]
        free_after_last_reduction = [
            index_var
            for index_var in free_vars
            if index_var in loop_pos and loop_pos[index_var] > last_reduction_pos
        ]
        if not free_after_last_reduction:
            return False

        if Scheduler._has_dense_output(cin):
            # For dense outputs, keep workspace insertion conservative:
            # only enable a 1D dense accumulator over a trailing dense axis.
            if len(free_after_last_reduction) != 1:
                return False
            result_tensor_accesses = cin.get_result_tensor_accesses()
            if not result_tensor_accesses:
                return False
            result_tensor_access = result_tensor_accesses[0]
            free_index_var = free_after_last_reduction[0]
            if not result_tensor_access.has_index_var(free_index_var):
                return False
            if (
                result_tensor_access.level_type_of_index_var(free_index_var)
                != LevelType.DENSE
            ):
                return False

        return any(
            index_var in loop_pos and loop_pos[index_var] > last_reduction_pos
            for index_var in free_vars
        )

    @staticmethod
    def _workspace_insertion_record(cin: CIN) -> Optional[WorkspaceInsertion]:
        """Derive the workspace decision :meth:`insert_workspace` will apply.

        This is the recording twin of the derivation inside
        :meth:`insert_workspace` — the last in-order reduction variable, the
        free variables inside the region (after the loop the region replaces,
        see :meth:`_sparse_result_workspace_loop`), and the dense-versus-sparse
        representation (dense only for a single all-dense trailing axis).  It
        is computed on the pre-insertion nest so the automatic scheduler can
        record its standalone workspace decision as an explicit, replayable
        ``LoopPlan`` fact instead of hidden policy state.  Returns ``None``
        when insertion would not materialize a workspace.
        """

        if not isinstance(cin, ForAll):
            return None
        cin_ivar_getter = CINIndexVariablesGetter()
        cin_ivar_getter.visit(cin)
        result_tensor_accesses = cin.get_result_tensor_accesses()
        if not result_tensor_accesses:
            return None
        result_tensor_access = result_tensor_accesses[0]
        reduction_vars = cin_ivar_getter.get_reduction_vars()
        free_vars = cin_ivar_getter.get_free_vars()
        index_vars_ordered = cin.loop_order
        reduction_vars_todo = [
            var for var in reduction_vars if var in index_vars_ordered
        ]
        if not reduction_vars_todo:
            return None
        next_reduction_var = reduction_vars_todo[-1]
        # The region may begin at an earlier reduction loop than the one the
        # fact records (a sparse result); its key is the free loops inside it.
        region_var = Scheduler._sparse_result_workspace_loop(
            cin, index_vars_ordered, reduction_vars_todo
        )
        if region_var is None:
            region_var = next_reduction_var
        region_index = index_vars_ordered.index(region_var)
        vars_after_last_reduction = index_vars_ordered[region_index + 1 :]
        free_vars_after_last_reduction = [
            var for var in vars_after_last_reduction if var in free_vars
        ]
        if not free_vars_after_last_reduction:
            return None
        level_types = [
            result_tensor_access.level_type_of_index_var(var)
            for var in free_vars_after_last_reduction
        ]
        are_all_dense_levels = all(
            level_type == LevelType.DENSE for level_type in level_types
        )
        if are_all_dense_levels and len(free_vars_after_last_reduction) > 1:
            are_all_dense_levels = False
        return WorkspaceInsertion(
            reduction_loop=LoopRef(next_reduction_var.index_id),
            axis_loops=tuple(
                LoopRef(var.index_id) for var in free_vars_after_last_reduction
            ),
            dense=are_all_dense_levels,
        )

    @staticmethod
    def _find_index_var_by_name(cin: CIN, name: str) -> IndexVar:
        matches = [index_var for index_var in cin.index_vars if index_var.name == name]
        if not matches:
            raise ScorchValueError(
                f"Unknown index variable {name!r}", code="schedule_loop_unknown"
            )
        return matches[0]

    @staticmethod
    def _find_index_var_by_id(cin: CIN, index_id: IndexId) -> IndexVar:
        matches = [
            index_var for index_var in cin.index_vars if index_var.index_id == index_id
        ]
        if not matches:
            raise ScorchValueError(
                f"Unknown IndexId {index_id.value}", code="index_id_unknown"
            )
        return matches[0]

    @staticmethod
    def _outer_loop_prefix(cin: CIN) -> List[ForAll]:
        prefix: List[ForAll] = []
        current: CIN = cin
        while isinstance(current, ForAll):
            prefix.append(current)
            current = current.stmt
        return prefix

    @staticmethod
    def _placement_depth(cin: CIN, placement: str, target_name: str) -> int:
        prefix = Scheduler._outer_loop_prefix(cin)

        if placement == "outermost":
            depth = 0
        elif placement.startswith("child_of:"):
            parent_name = placement.split(":", 1)[1]
            if not parent_name:
                raise ScorchValueError(
                    "child_of placement requires an index variable",
                    code="tile_placement_invalid",
                )
            exact = [
                pos
                for pos, loop in enumerate(prefix)
                if loop.index_var.name == parent_name
            ]
            if not exact:
                raise ScorchValueError(
                    f"Cannot place tile child of {parent_name!r}; it is not in "
                    "the common outer loop prefix",
                    code="tile_placement_invalid",
                )
            depth = exact[0] + 1
        elif placement.startswith("at_depth:"):
            value = placement.split(":", 1)[1]
            try:
                depth = int(value)
            except ValueError as exc:
                raise ScorchValueError(
                    "at_depth placement requires an integer",
                    code="tile_placement_invalid",
                ) from exc
            if depth < 0 or depth > len(prefix):
                raise ScorchValueError(
                    f"at_depth:{depth} is outside the common loop-prefix range "
                    f"0..{len(prefix)}",
                    code="tile_placement_invalid",
                )
        else:
            raise ScorchValueError(
                f"Unsupported tile placement {placement!r}",
                code="tile_placement_invalid",
            )

        target_in_prefix = next(
            (
                pos
                for pos, loop in enumerate(prefix)
                if loop.index_var.name == target_name
            ),
            None,
        )
        if target_in_prefix is not None and depth > target_in_prefix:
            raise ScorchValueError(
                f"Tile outer loop for {target_name!r} must dominate its inner loop",
                code="tile_placement_invalid",
            )
        return depth

    @staticmethod
    def _insert_loop_at_depth(cin: CIN, loop: ForAll, depth: int) -> CIN:
        if depth == 0:
            loop.stmt = cin
            loop.inserted_workspace = cin.inserted_workspace
            loop.no_tile_list = list(cin.no_tile_list)
            return loop

        parent: CIN = cin
        for _ in range(depth - 1):
            if not isinstance(parent, ForAll):
                raise ScorchValueError(
                    "Tile placement does not name a common loop prefix",
                    code="tile_placement_invalid",
                )
            parent = parent.stmt
        if not isinstance(parent, ForAll):
            raise ScorchValueError(
                "Tile placement does not name a common loop prefix",
                code="tile_placement_invalid",
            )

        child = parent.stmt
        loop.stmt = child
        parent.stmt = loop
        return cin

    @staticmethod
    def add_tile(
        cin: CIN,
        index_var: IndexVar,
        tile_size: int,
        placement: Optional[str] = None,
        parallel: bool = False,
        unroll: bool = True,
        use_workspace: bool = True,
        compile_options: Optional[CompileOptions] = None,
    ) -> CIN:
        """
        Tile the index_var of a CIN statement.
        Specifically,
            1) it stripmines the index_var (i.e. splits it into inner and outer loops)
            2) reorders the outer loop past all other inner loops
        Returns a new CIN statement.

        For example, for SpMM C[i, k] += A[i, j] * B[j, k],

        i = IndexVar("i")
        j = IndexVar("j")
        k = IndexVar("k")

        cin_stmt = ForAll(
            i,
            ForAll(
                j,
                ForAll(
                    k,
                    TensorAssign(
                        C[i, k],
                        A[i, j] * B[j, k],
                        op=Operation.ADD
                    )
                )
            )
        )

        add_tile(cin_stmt, k, 32) will return a new cin that is equivalent to being constructed as follows:

        1) The first step is to insert any necessary dense workspaces:

        cin_stmt = ForAll(
            i,
            Where(
                producer=ForAll(
                    j,
                    ForAll(
                        k,
                        TensorAssign(
                            accum_c[k],
                            A[i, j] * B[j, k],
                            op=Operation.ADD
                        )
                    )
                ),
                consumer=ForAll(
                    k,
                    TensorAssign(
                        C[i, k],
                        accum_c[k],
                    )
                )
            )
        )


        2) The second step is to stripmine the index_var:

        i = IndexVar("i")
        j = IndexVar("j")
        k_out = IndexVar("k_out")
        k_in = IndexVar("k_in")
        k = IndexVar("k", k_out + k_in)

        k_tile_size = 32
        k_tile_var = TileSizeVar(
            outer_index_var=k_out,
            inner_index_var=k_in,
            size=k_tile_size
        )

        accum_c = Workspace(name="accum_c", dim=1, dtype=C.dtype, dense=True)

        cin_stmt = ForAll(
            i,
            ForAll(
                k_out,
                Where(
                    producer=ForAll(
                        j,
                        ForAll(
                            k_in,
                            TensorAssign(
                                accum_c[k_in],
                                A[i, j] * B[j, k],
                                op=Operation.ADD,
                            ),
                        ),
                    ),
                    consumer=ForAll(
                        k_in,
                        TensorAssign(
                            C[i, k],
                            accum_c[k_in],
                        )
                    )
                )
            )
        )


        """

        if isinstance(tile_size, bool) or not isinstance(tile_size, int):
            raise ScorchTypeError(
                "tile_size must be an integer", code="tile_size_invalid"
            )
        if tile_size <= 0:
            raise ScorchValueError(
                "tile_size must be greater than zero", code="tile_request_invalid"
            )
        if not isinstance(cin, ForAll):
            raise ScorchValueError(
                "Expected input CIN to be a ForAll statement",
                code="tile_request_invalid",
            )
        if compile_options is not None:
            _compile_options_at_scheduler_boundary(compile_options)

        target_name = index_var.name
        target_id = index_var.index_id
        if (
            use_workspace
            and not cin.inserted_workspace
            and Scheduler._tile_target_needs_workspace(cin, target_name)
        ):
            loop_order, _ = Scheduler._extract_loop_chain(cin)
            if Scheduler.should_insert_workspace(cin, loop_order):
                cin = Scheduler.insert_workspace(cin, allow_dense=True)

        # Workspace insertion deep-copies the graph. Resolve the target by name
        # in the transformed graph instead of retaining the pre-copy identity.
        index_var = Scheduler._find_index_var_by_id(cin, target_id)
        if index_var.is_tiled or getattr(index_var, "_expr", None) is not None:
            raise ScorchValueError(
                f"Index variable {target_name!r} is already tiled",
                code="tile_request_invalid",
            )

        sparse_accesses = [
            tensor_access
            for tensor_access in cin.tensor_accesses
            if tensor_access.has_index_var(index_var)
            and tensor_access.level_type_of_index_var(index_var) != LevelType.DENSE
        ]
        if sparse_accesses:
            raise ScorchNotImplementedError(
                f"Affine tiling cannot split sparse index variable {target_name!r}; "
                "windowed compressed iterators are not supported",
                code="sparse_index_affine_tiling_unsupported",
            )

        if placement is None:
            options = _compile_options_at_scheduler_boundary(compile_options)
            if _regblock_enabled(options):
                placement = f"child_of:{cin.index_var.name}"
            else:
                placement = "outermost"
        insertion_depth = Scheduler._placement_depth(cin, placement, target_name)

        inner_index_var = IndexVar(f"{target_name}_in")
        outer_index_var = IndexVar(f"{target_name}_out")
        index_var.expr = outer_index_var + inner_index_var
        TileSizeVar(
            outer_index_var=outer_index_var,
            inner_index_var=inner_index_var,
            size=tile_size,
            unroll=unroll,
        )

        class ReplaceIndexVarVisitor(CINVisitorAccept):
            """
            This visitor replaces the index_var in the Where statement
                - indexing into the workspace with the inner index var
                - in the ForAll statement with the inner index var

            """

            def __init__(self, old_index_var: IndexVar, new_index_var: IndexVar):
                self.old_index_var = old_index_var
                self.new_index_var = new_index_var
                self.replacements = 0

            def visit_ForAll(self, forall: ForAll):
                if forall.index_var == self.old_index_var:
                    forall.index_var = self.new_index_var
                    self.replacements += 1
                self.visit(forall.stmt)

            def visit_Where(self, where: Where):
                self.visit(where.producer)
                self.visit(where.consumer)

            def visit_TensorAssign(self, tensor_assign: TensorAssign):
                self.visit(tensor_assign.lhs)
                self.visit(tensor_assign.rhs)

            def visit_WorkspaceAccess(self, workspace_access: WorkspaceAccess):
                if not workspace_access.indices:
                    return
                indices = [
                    self.new_index_var if index == self.old_index_var else index
                    for index in workspace_access.indices
                ]
                workspace_access.update_indices(indices)

        replace_index_var_visitor = ReplaceIndexVarVisitor(
            old_index_var=index_var,
            new_index_var=inner_index_var,
        )
        replace_index_var_visitor.visit(cin)
        if replace_index_var_visitor.replacements == 0:
            raise ScorchValueError(
                f"Cannot tile {target_name!r}: no matching ForAll binder was found",
                code="tile_request_invalid",
            )

        outer_forall = ForAll(
            index_var=outer_index_var,
            stmt=cin,
            parallel=True if parallel else None,
        )
        return Scheduler._insert_loop_at_depth(cin, outer_forall, insertion_depth)

    @staticmethod
    def insert_workspace(cin: CIN, allow_dense=False) -> CIN:
        """
        Args:
            cin: CIN statement to insert a workspace into
            allow_dense: If True, then allow dense workspaces to be inserted.

        Returns:
            A new CIN statement with a workspace inserted.

        Insert a workspace into a CIN statement, if necessary.
        Only works on the last reduction variable in the loop order.

        This function should be idempotent.
        """

        # Collect all the reduction variables
        cin_ivar_getter = CINIndexVariablesGetter()
        cin_ivar_getter.visit(cin)

        if not isinstance(cin, ForAll):
            raise ScorchAssertionError(
                "Expected input CIN to be a ForAll statement.",
                code="workspace_insertion_target_not_forall",
            )

        result_tensor_accesses = cin.get_result_tensor_accesses()
        result_tensor_access: TensorAccess = result_tensor_accesses[0]

        reduction_vars = cin_ivar_getter.get_reduction_vars()
        free_vars = cin_ivar_getter.get_free_vars()

        # loop_order_getter = LoopOrderGetter(cin)
        # index_vars_ordered = loop_order_getter.index_vars_ordered
        index_vars_ordered = cin.loop_order

        if len(reduction_vars) == 0:
            return cin

        reduction_vars_todo = [
            var for var in reduction_vars if var in index_vars_ordered
        ]

        if len(reduction_vars_todo) == 0:
            return cin

        next_reduction_var = Scheduler._sparse_result_workspace_loop(
            cin, index_vars_ordered, reduction_vars_todo
        )
        if next_reduction_var is None:
            next_reduction_var = reduction_vars_todo[-1]

        last_reduction_var_index = index_vars_ordered.index(next_reduction_var)

        # List of variables that come after the last reduction variable
        # in the loop order
        vars_after_last_reduction = index_vars_ordered[last_reduction_var_index + 1 :]
        # List of free variables that come after the last reduction variable
        # in the loop order
        free_vars_after_last_reduction = [
            var for var in vars_after_last_reduction if var in free_vars
        ]

        dim_workspace = len(free_vars_after_last_reduction)
        if dim_workspace == 0:
            return cin

        free_vars_after_last_reduction_level_types = [
            result_tensor_access.level_type_of_index_var(var)
            for var in free_vars_after_last_reduction
        ]

        are_all_dense_levels = all(
            level_type == LevelType.DENSE
            for level_type in free_vars_after_last_reduction_level_types
        )
        # Current lowering supports dense workspaces only for 1D accesses.
        # Fall back to sparse workspace representation for higher dimensions.
        if are_all_dense_levels and dim_workspace > 1:
            are_all_dense_levels = False

        if not allow_dense and are_all_dense_levels:
            return cin

        new_cin = copy.deepcopy(cin)
        copied_indices = {
            index_var.index_id: index_var for index_var in new_cin.index_vars
        }
        workspace_indices = [
            copied_indices[index_var.index_id]
            for index_var in free_vars_after_last_reduction
        ]

        workspace = Workspace(
            name="wksp",
            dim=dim_workspace,
            dtype=result_tensor_access.tensor.dtype,
            dense=are_all_dense_levels,
        )

        workspace_access = WorkspaceAccess(
            wksp=workspace,
            indices=workspace_indices,
        )

        # Note: parent_forall not necessarily ForAll statement at the end
        parent_forall = new_cin
        while (
            isinstance(parent_forall.stmt, ForAll)
            and parent_forall.stmt.index_var != next_reduction_var
            and parent_forall.index_var != next_reduction_var
        ):
            parent_forall = parent_forall.stmt

        if not (isinstance(parent_forall, ForAll)):
            raise ScorchAssertionError(
                "Expected parent_forall to be a ForAll statement.",
                code="workspace_insertion_target_not_forall",
            )

        """
        For example, if parent_forall's stmt is:

        ForAll(
            k,
            ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    B[i, k] * C[k, j],
                ),
            ),
        )

        Then it needs to be transformed into:

        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[j],
                        B[i, k] * C[k, j],
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    A[i, j],
                    workspace[j],
                ),
            ),
        )

        """

        reduction_forall = (
            parent_forall
            if parent_forall.index_var == next_reduction_var
            else parent_forall.stmt
        )

        # If we have already inserted a workspace, then we should not insert another one.
        if isinstance(reduction_forall, Where):
            return cin

        # Create the producer forall
        producer_forall = copy.deepcopy(reduction_forall)

        producer_forall_tensor_access_parent = producer_forall

        # Iterate until the TensorAssign statement
        while not isinstance(producer_forall_tensor_access_parent.stmt, TensorAssign):
            producer_forall_tensor_access_parent = (
                producer_forall_tensor_access_parent.stmt
            )

        # Replace the TensorAssign's lhs with the workspace
        producer_forall_tensor_access_parent.stmt.lhs = workspace_access

        # Create the consumer forall
        consumer_forall = copy.deepcopy(reduction_forall)

        consumer_forall_tensor_access_parent = consumer_forall
        # Iterate until the TensorAssign statement
        while not isinstance(consumer_forall_tensor_access_parent.stmt, TensorAssign):
            consumer_forall_tensor_access_parent = (
                consumer_forall_tensor_access_parent.stmt
            )

        # Replace the TensorAssign's rhs with the workspace
        consumer_forall_tensor_access_parent.stmt.rhs = workspace_access

        # Create the Where statement
        where_stmt = Where(
            producer=producer_forall,
            consumer=consumer_forall,
        )

        if not are_all_dense_levels:
            if not isinstance(producer_forall, ForAll):
                raise ScorchAssertionError(
                    "isinstance(producer_forall, ForAll)",
                    code="workspace_insertion_target_not_forall",
                )
            new_cin.no_tile_list.append(producer_forall.index_var)

        # Replace the reduction forall with the Where statement
        if (
            isinstance(parent_forall.stmt, ForAll)
            and parent_forall.stmt.index_var == next_reduction_var
        ):
            parent_forall.stmt = where_stmt
        else:
            new_cin = where_stmt

        new_cin.inserted_workspace = True

        return new_cin

    @staticmethod
    def select_loop_order(
        cin: CIN,
        costs: Optional[_CostModelConstants] = None,
        compile_options: Optional[CompileOptions] = None,
        *,
        pre_forced_order: Optional[List[IndexVar]] = None,
    ) -> List[IndexVar]:
        """Select the automatic logical loop order.

        ``pre_forced_order``, when given an empty list, receives the composed
        order as it stands immediately before the sparse-output forced reorder
        below -- the one block of this composition that no legality rule
        constrains.  It is an out-parameter rather than a second return value so
        that every existing caller, which passes nothing, executes exactly the
        statements it executed before.
        """
        if costs is None:
            _, costs = _scheduler_costs_at_boundary(costs, compile_options)
        elif compile_options is not None:
            _, costs = _scheduler_costs_at_boundary(costs, compile_options)
        elif type(costs) is not SchedulerCostModel:
            raise CompileOptionsError(
                (
                    CompileOptionsDiagnostic(
                        code="invalid_type",
                        field="scheduler.cost_model",
                        message="expected an exact SchedulerCostModel",
                    ),
                )
            )
        loop_order = Scheduler.init_loop_order(cin, costs=costs)
        loop_order = Scheduler.optimize_loop_order(cin, loop_order, costs=costs)
        loop_order = Scheduler.apply_mode_order_constraints(
            cin, loop_order, costs=costs
        )
        if pre_forced_order is not None:
            if type(pre_forced_order) is not list or pre_forced_order:
                raise ScorchTypeError(
                    "pre_forced_order must be an empty exact list",
                    code="internal_argument_type_invalid",
                )
            pre_forced_order.extend(loop_order)

        # For sparse output with reduction variables, ensure at least one free
        # variable appears after the last reduction variable.  The lowerer
        # requires the innermost loop to correspond to a result-tensor level,
        # and workspace insertion needs free variables after the reduction to
        # define the workspace dimensions.
        #
        # Only skip forced reordering for all-coordinate (COO) output when
        # an input tensor shares the exact same index variables as the output
        # (e.g., SDDMM's S[i,j] determines output sparsity).  For SpMM-like
        # ops where output sparsity differs from any single input, we still
        # need the workspace path.
        #
        # NOTE: For SDDMM-like patterns (S[i,j] = M[i,j]*Q[i,k]*K[j,k])
        # where an input tensor mirrors the output sparsity, the optimal
        # loop order is i→j→k (reduction innermost).  The cost model
        # correctly identifies this as cheapest, but the lowerer cannot
        # handle reduction-innermost for non-COO sparse outputs.  For COO
        # output, the scalar-accum path in iter_lattice.py handles this
        # correctly — einsum() auto-infers COO format for SDDMM patterns
        # to leverage this (see ops.py format inference).  For CSR output
        # (explicit format="ds"), the forced reorder still applies.
        # TODO: Extend _should_use_scalar_accum in iter_lattice.py to
        # support CSR output, then remove the _all_coo gate below.
        if not Scheduler._has_dense_output(cin):
            _needs_forced_reorder = True
            _result_accesses = cin.get_result_tensor_accesses()
            if _result_accesses:
                _non_ws = [
                    a for a in _result_accesses if not isinstance(a.tensor, Workspace)
                ]
                if _non_ws:
                    _all_coo = all(
                        lt == LevelType.COORDINATE
                        for lt in _non_ws[0].get_tensor().get_level_types()
                    )
                    # Check if an input tensor mirrors the output sparsity.
                    # Sharing the result's index variables is not enough: the
                    # loops over them must walk that input's stored entries and
                    # nothing else (``coordinate_result_driver``).  With a
                    # dense ``x[a]`` and a sparse ``B[a,b]``, ``x`` has the
                    # indices of ``C[a]`` but the loop over ``a`` walks ``B``;
                    # left unordered, that program stored ``a`` once per stored
                    # entry of ``B``.  It takes the workspace path like every
                    # other reduction into a sparse result.
                    if _all_coo:
                        _assignment: object = cin
                        while isinstance(_assignment, ForAll):
                            _assignment = _assignment.stmt
                        if (
                            isinstance(_assignment, TensorAssign)
                            and coordinate_result_driver(_non_ws[0], _assignment.rhs)
                            is not None
                        ):
                            _needs_forced_reorder = False

            if _needs_forced_reorder:
                cin_ivar_getter = CINIndexVariablesGetter()
                cin_ivar_getter.visit(cin)
                reduction_vars = set(
                    Scheduler._unique_index_vars(cin_ivar_getter.get_reduction_vars())
                )
                if reduction_vars:
                    reductions_in_loop = [v for v in loop_order if v in reduction_vars]
                    if reductions_in_loop:
                        last_red_pos = max(
                            loop_order.index(v) for v in reductions_in_loop
                        )
                        free_after = [
                            v
                            for v in loop_order[last_red_pos + 1 :]
                            if v not in reduction_vars
                        ]
                        if not free_after:
                            free_vars = [
                                v for v in loop_order if v not in reduction_vars
                            ]
                            if free_vars:
                                last_free = free_vars[-1]
                                idx = loop_order.index(last_free)
                                loop_order = (
                                    loop_order[:idx]
                                    + loop_order[idx + 1 :]
                                    + [last_free]
                                )

        return loop_order

    @staticmethod
    def resolve_loop_order(
        cin: CIN,
        loop_order: Sequence[str],
    ) -> List[IndexVar]:
        """Resolve and validate a tuner-provided logical loop permutation."""
        requested = list(loop_order)
        if any(not isinstance(name, str) or not name for name in requested):
            raise ScorchValueError(
                "Schedule.loop_order must contain non-empty names",
                code="schedule_field_value_invalid",
            )
        if len(requested) != len(set(requested)):
            raise ScorchValueError(
                "Schedule.loop_order contains duplicate variables",
                code="schedule_loop_name_duplicate",
            )

        available_vars = Scheduler.get_index_variables(cin)
        by_name = {index_var.name: index_var for index_var in available_vars}
        unknown = [name for name in requested if name not in by_name]
        missing = [name for name in by_name if name not in requested]
        if unknown or missing:
            details = []
            if unknown:
                details.append(f"unknown={unknown}")
            if missing:
                details.append(f"missing={missing}")
            raise ScorchValueError(
                "Schedule.loop_order must be a complete permutation ("
                + ", ".join(details)
                + ")",
                code="schedule_loop_order_incomplete",
            )

        positions = {name: index for index, name in enumerate(requested)}
        for result_access in cin.get_result_tensor_accesses():
            if isinstance(result_access.tensor, Workspace):
                continue
            result_order = [
                index_var.name for index_var in result_access.get_sorted_index_vars()
            ]
            for parent, child in zip(result_order, result_order[1:]):
                if positions[parent] > positions[child]:
                    raise ScorchValueError(
                        "Schedule.loop_order violates result storage order: "
                        f"{parent!r} must precede {child!r}",
                        code="loop_order_result_storage_discordant_unsupported",
                    )
        return [by_name[name] for name in requested]

    @staticmethod
    def _tile_target_needs_workspace(cin: CIN, target_name: str) -> bool:
        """Whether a stack tile targets the trailing free accumulator domain."""
        loop_order, _ = Scheduler._extract_loop_chain(cin)
        if not loop_order or not Scheduler.should_insert_workspace(cin, loop_order):
            return False

        getter = CINIndexVariablesGetter()
        getter.visit(cin)
        reduction_names = {var.name for var in getter.get_reduction_vars()}
        free_names = {var.name for var in getter.get_free_vars()}
        positions = {var.name: pos for pos, var in enumerate(loop_order)}
        reductions = [positions[name] for name in reduction_names if name in positions]
        return bool(
            target_name in free_names
            and target_name in positions
            and reductions
            and positions[target_name] > max(reductions)
        )

    @staticmethod
    def _set_explicit_parallel_loop(cin: CIN, loop_name: str) -> None:
        matches: List[ForAll] = []

        class LoopFinder(CINVisitorAccept):
            def visit_ForAll(self, forall: ForAll):
                if forall.index_var.name == loop_name:
                    matches.append(forall)
                self.visit(forall.stmt)

            def visit_Where(self, where: Where):
                self.visit(where.producer)
                self.visit(where.consumer)

            def visit_TensorAssign(self, tensor_assign: TensorAssign):
                return

        LoopFinder().visit(cin)
        if len(matches) != 1:
            raise ScorchValueError(
                f"Schedule.parallel_loop {loop_name!r} must identify exactly one "
                f"ForAll loop; found {len(matches)}",
                code="schedule_parallel_loop_unknown",
            )
        matches[0].parallel = True

    @staticmethod
    def _validate_heap_result_tile(
        cin: CIN,
        schedule: Schedule,
        reduction_names: Set[str],
    ) -> Optional[_ResultTilePlan]:
        """Validate a compact heap-backed result tile before transforming CIN.

        Eligibility comes from the dense result access, physical level order,
        free/reduction roles, and schedule loop anchors.  The compact buffer
        covers every dense result prefix position and one trailing-axis tile.
        """
        heap_tiles = [tile for tile in schedule.tiles if tile.accum == "heap"]
        if not heap_tiles:
            return None
        if len(heap_tiles) != 1:
            raise ScorchNotImplementedError(
                "Heap accumulation currently supports exactly one result tile",
                code="heap_accumulation_tile_unsupported",
            )

        tile = heap_tiles[0]
        if tile.kind != "affine":
            raise ScorchValueError(
                "Heap accumulation requires an affine result tile",
                code="heap_tile_schedule_invalid",
            )
        if tile.placement != "outermost":
            raise ScorchValueError(
                "Heap accumulation requires its affine tile to be outermost so "
                "the compact result spans every enclosed reduction",
                code="heap_tile_schedule_invalid",
            )
        tile_position = schedule.tiles.index(tile)
        outer_wrappers = []
        for candidate in schedule.tiles[tile_position + 1 :]:
            affine_at_root = candidate.kind == "affine" and (
                candidate.placement == "outermost"
                or (
                    candidate.placement.startswith("at_depth:")
                    and int(candidate.placement.split(":", 1)[1]) == 0
                )
            )
            panel_at_root = (
                candidate.kind == "panel" and candidate.placement == "outermost"
            )
            if affine_at_root or panel_at_root:
                outer_wrappers.append(candidate.index_var)
        if outer_wrappers:
            raise ScorchValueError(
                "Heap accumulation requires its affine result tile to remain "
                "outermost after all scheduled tiles are applied; later root "
                f"tiles would wrap it: {outer_wrappers}",
                code="heap_tile_schedule_invalid",
            )
        if tile.parallel:
            raise ScorchValueError(
                "A heap-backed result tile uses shared reusable storage; its "
                "outer tile loop must be serial",
                code="heap_tile_schedule_invalid",
            )
        if schedule.parallel_loop in (
            tile.index_var,
            f"{tile.index_var}_out",
            f"{tile.index_var}_in",
        ):
            raise ScorchValueError(
                "A heap-backed result tile cannot select its shared tile loop "
                "for parallel execution",
                code="heap_tile_schedule_invalid",
            )

        assignment: CIN = cin
        while isinstance(assignment, ForAll):
            assignment = assignment.stmt
        if not isinstance(assignment, TensorAssign) or assignment.op not in (
            None,
            Operation.ADD,
        ):
            raise ScorchNotImplementedError(
                "Heap accumulation requires one additive result assignment",
                code="heap_accumulation_result_unsupported",
            )
        if not reduction_names:
            raise ScorchNotImplementedError(
                "Heap accumulation requires an enclosed reduction to accumulate",
                code="heap_accumulation_without_reduction_unsupported",
            )
        if tile.index_var in reduction_names:
            raise ScorchValueError(
                "Heap accumulation must tile a free result axis, not a reduction",
                code="heap_tile_schedule_invalid",
            )

        result_accesses = [
            access
            for access in cin.get_result_tensor_accesses()
            if not isinstance(access.tensor, Workspace)
        ]
        if len(result_accesses) != 1 or not result_accesses[0].is_dense():
            raise ScorchNotImplementedError(
                "Heap accumulation requires exactly one dense result tensor",
                code="heap_accumulation_result_unsupported",
            )
        result_access = result_accesses[0]
        result_names = tuple(
            index_var.name for index_var in result_access.get_sorted_index_vars()
        )
        if len(result_names) < 2 or result_names[-1] != tile.index_var:
            raise ScorchNotImplementedError(
                "Heap accumulation requires the tiled free axis to be the dense "
                "result's trailing storage level",
                code="heap_accumulation_tile_unsupported",
            )
        parallel_anchors = [
            candidate.index_var for candidate in schedule.tiles if candidate.parallel
        ]
        if schedule.parallel_loop is not None:
            parallel_anchors.append(schedule.parallel_loop)
        for anchor in parallel_anchors:
            logical_anchor = anchor
            for candidate in schedule.tiles:
                if anchor in (
                    f"{candidate.index_var}_out",
                    f"{candidate.index_var}_in",
                ):
                    logical_anchor = candidate.index_var
                    break
            if logical_anchor not in result_names[:-1]:
                raise ScorchValueError(
                    "Heap accumulation may parallelize only a dense result-prefix "
                    f"loop; {anchor!r} does not partition compact result rows",
                    code="heap_tile_schedule_invalid",
                )
        if any(level != LevelType.DENSE for level in result_access.level_types()):
            raise ScorchNotImplementedError(
                "Heap accumulation currently supports all-dense result storage",
                code="heap_accumulation_result_unsupported",
            )
        tile_index_var = Scheduler._find_index_var_by_name(cin, tile.index_var)
        result_level = result_access.level_of_index_var(tile_index_var)
        if result_level != result_access.num_levels - 1:
            raise ScorchNotImplementedError(
                "Heap accumulation requires the tiled free axis at the final "
                "result storage level",
                code="heap_accumulation_tile_unsupported",
            )
        if result_access.tensor.dtype not in (torch.float32, torch.float64):
            raise ScorchNotImplementedError(
                "Heap accumulation supports float32 or float64 dense results",
                code="heap_accumulation_result_unsupported",
            )
        if not parallel_anchors:
            raise ScorchValueError(
                "Heap accumulation requires an explicit parallel dense "
                "result-prefix loop so the shared outer result-tile loop remains "
                "serial",
                code="heap_tile_schedule_invalid",
            )

        return _ResultTilePlan(
            result=result_access.tensor.name,
            result_id=result_access.tensor_id,
            tile_var=tile.index_var,
            result_level=result_level,
            result_prefix_vars=result_names[:-1],
            access_index_vars=tuple(
                index_var.name for index_var in result_access.indices
            ),
            access_index_ids=tuple(result_access.index_ids),
        )

    @staticmethod
    def _multiplicative_rhs_accesses(
        expr: CIN,
    ) -> Optional[List[TensorAccess]]:
        """Collect tensor leaves when an RHS is purely multiplicative."""
        if isinstance(expr, TensorAccess):
            return [expr]
        if not isinstance(expr, BinaryOp) or expr.op != Operation.MUL:
            return None
        left = Scheduler._multiplicative_rhs_accesses(expr.left)
        right = Scheduler._multiplicative_rhs_accesses(expr.right)
        if left is None or right is None:
            return None
        return left + right

    @staticmethod
    def _validate_relayout(
        cin: CIN,
        schedule: Schedule,
        panel_tiles: List[TileSpec],
        reduction_names: Set[str],
    ) -> Optional[_RelayoutPlan]:
        """Recognize the supported packed tile-ijk contraction by structure.

        This intentionally validates before any CIN transform.  Names enter only
        through the user's ``RelayoutSpec``; eligibility comes from tensor
        accesses, level formats, free/reduction roles, and tile placement.
        """
        relayout = schedule.relayout
        if relayout is None:
            return None

        assignment: CIN = cin
        while isinstance(assignment, ForAll):
            assignment = assignment.stmt
        if not isinstance(assignment, TensorAssign) or assignment.op not in (
            None,
            Operation.ADD,
        ):
            raise ScorchNotImplementedError(
                "Packed relayout requires one additive contraction assignment",
                code="relayout_operand_shape_unsupported",
            )

        rhs_accesses = cin.get_rhs_tensor_accesses()
        packed_accesses = [
            access for access in rhs_accesses if access.tensor.name == relayout.operand
        ]
        if len(packed_accesses) != 1:
            raise ScorchValueError(
                "RelayoutSpec.operand must name exactly one input tensor access; "
                f"found {len(packed_accesses)} for {relayout.operand!r}",
                code="relayout_schedule_invalid",
            )
        packed_access = packed_accesses[0]
        packed_names = [
            index_var.name for index_var in packed_access.get_sorted_index_vars()
        ]
        if relayout.pack_var not in packed_names:
            raise ScorchValueError(
                f"Relayout pack_var {relayout.pack_var!r} does not index operand "
                f"{relayout.operand!r}",
                code="relayout_schedule_invalid",
            )
        if not packed_access.is_dense() or packed_access.num_levels != 2:
            raise ScorchNotImplementedError(
                "Packed relayout requires a rank-2 dense input operand",
                code="relayout_operand_shape_unsupported",
            )
        if packed_names[-1] != relayout.pack_var:
            raise ScorchNotImplementedError(
                "Packed relayout requires pack_var to be the operand's contiguous "
                "last storage level",
                code="relayout_operand_shape_unsupported",
            )
        multiplicative_accesses = Scheduler._multiplicative_rhs_accesses(assignment.rhs)
        if multiplicative_accesses is None or sorted(
            id(access) for access in multiplicative_accesses
        ) != sorted(id(access) for access in rhs_accesses):
            raise ScorchNotImplementedError(
                "Packed relayout requires the staged dense and compressed "
                "accesses to participate in one multiplicative contraction "
                "expression",
                code="relayout_operand_shape_unsupported",
            )
        panel_var = packed_names[0]
        scope_var = relayout.scope_var
        if scope_var not in (panel_var, relayout.pack_var):
            raise ScorchValueError(
                "RelayoutSpec.scope_var must name either the staged access's "
                f"panel axis {panel_var!r} or tiled axis {relayout.pack_var!r}; "
                f"got {scope_var!r}",
                code="relayout_schedule_invalid",
            )
        if scope_var is None:
            raise ScorchAssertionError(
                "scope_var is not None", code="relayout_scope_unresolved"
            )

        if len(panel_tiles) != 1 or panel_tiles[0].index_var != panel_var:
            raise ScorchValueError(
                "Packed relayout requires exactly one sparse panel tile on the "
                f"other operand index {panel_var!r}",
                code="relayout_schedule_invalid",
            )
        panel_tile = panel_tiles[0]
        affine_tiles = [tile for tile in schedule.tiles if tile.kind == "affine"]
        pack_tiles = [
            tile for tile in affine_tiles if tile.index_var == relayout.pack_var
        ]
        if len(pack_tiles) != 1:
            raise ScorchValueError(
                "Packed relayout requires exactly one affine tile for pack_var "
                f"{relayout.pack_var!r}",
                code="relayout_schedule_invalid",
            )
        pack_tile = pack_tiles[0]
        if len(schedule.tiles) != 2 or len(affine_tiles) != 1:
            raise ScorchNotImplementedError(
                "Packed relayout currently supports only one affine pack tile "
                "and one sparse panel tile",
                code="relayout_operand_shape_unsupported",
            )
        if pack_tile.width != relayout.strip_width:
            raise ScorchValueError(
                "Relayout strip_width must match the affine pack tile width "
                f"({relayout.strip_width} != {pack_tile.width})",
                code="relayout_schedule_invalid",
            )
        if pack_tile.accum not in ("direct", "heap"):
            raise ScorchNotImplementedError(
                "Packed relayout supports direct or heap-backed output accumulation",
                code="relayout_accumulation_unsupported",
            )
        expected_panel_placement = f"child_of:{relayout.pack_var}_out"
        if (
            pack_tile.placement != "outermost"
            or panel_tile.placement != expected_panel_placement
        ):
            raise ScorchValueError(
                "Packed relayout requires an outermost pack tile followed by a "
                f"panel placed at {expected_panel_placement!r}",
                code="relayout_schedule_invalid",
            )

        sparse_rhs = [access for access in rhs_accesses if not access.is_dense()]
        compressed_rhs = [
            access
            for access in sparse_rhs
            if LevelType.COMPRESSED in access.level_types()
        ]
        if len(rhs_accesses) != 2 or len(sparse_rhs) != 1 or len(compressed_rhs) != 1:
            raise ScorchNotImplementedError(
                "Packed relayout requires exactly one CSR input and one dense "
                "input tensor access",
                code="relayout_operand_shape_unsupported",
            )
        compressed_access = compressed_rhs[0]
        if (
            compressed_access.level_types()
            != [
                LevelType.DENSE,
                LevelType.COMPRESSED,
            ]
            or compressed_access.num_levels != 2
        ):
            raise ScorchNotImplementedError(
                "Packed relayout requires a rank-2 CSR input with a dense parent",
                code="relayout_operand_shape_unsupported",
            )
        compressed_names = [
            index_var.name for index_var in compressed_access.get_sorted_index_vars()
        ]
        row_var, compressed_panel_var = compressed_names
        if compressed_panel_var != panel_var:
            raise ScorchNotImplementedError(
                "The CSR compressed coordinate must be the packed operand's "
                f"panel index {panel_var!r}",
                code="relayout_operand_shape_unsupported",
            )

        result_accesses = cin.get_result_tensor_accesses()
        if len(result_accesses) != 1 or not result_accesses[0].is_dense():
            raise ScorchNotImplementedError(
                "Packed relayout requires exactly one dense result tensor",
                code="relayout_operand_shape_unsupported",
            )
        result_access = result_accesses[0]
        result_names = [
            index_var.name for index_var in result_access.get_sorted_index_vars()
        ]
        if result_names != [row_var, relayout.pack_var]:
            raise ScorchNotImplementedError(
                "Packed relayout requires the dense result to be indexed by the "
                "CSR row followed by pack_var",
                code="relayout_operand_shape_unsupported",
            )
        if panel_var not in reduction_names or relayout.pack_var in reduction_names:
            raise ScorchNotImplementedError(
                "Packed relayout requires panel_var to be a reduction and "
                "pack_var to be a free result axis",
                code="relayout_operand_shape_unsupported",
            )

        expected_order = (row_var, panel_var, relayout.pack_var)
        if schedule.loop_order != expected_order:
            raise ScorchValueError(
                "Packed relayout requires loop_order to match the structural "
                f"(row, panel, pack) order {expected_order!r}",
                code="relayout_schedule_invalid",
            )
        if schedule.parallel_loop != row_var:
            raise ScorchValueError(
                "Packed relayout requires the CSR row loop to be selected for "
                "parallel execution",
                code="relayout_schedule_invalid",
            )

        tensors = [access.tensor for access in rhs_accesses] + [result_access.tensor]
        dtypes = {tensor.dtype for tensor in tensors}
        if len(dtypes) != 1 or packed_access.tensor.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ScorchNotImplementedError(
                "Packed relayout supports matching float32 or float64 operand "
                "and result dtypes",
                code="relayout_dtype_unsupported",
            )

        return _RelayoutPlan(
            operand=relayout.operand,
            operand_id=packed_access.tensor_id,
            pack_var=relayout.pack_var,
            panel_var=panel_var,
            scope_var=scope_var,
            row_var=row_var,
            access_index_vars=tuple(
                index_var.name for index_var in packed_access.indices
            ),
            access_index_ids=tuple(packed_access.index_ids),
            operand_panel_level=packed_access.level_of_index_var(
                Scheduler._find_index_var_by_name(cin, panel_var)
            ),
            operand_pack_level=packed_access.level_of_index_var(
                Scheduler._find_index_var_by_name(cin, relayout.pack_var)
            ),
        )

    @staticmethod
    def _validate_legacy_replay_scope(
        cin: ForAll,
        schedule: Schedule,
        costs: _CostModelConstants,
    ) -> bool:
        """Validate public decisions that transitional replay can represent."""

        is_identity = (
            schedule.loop_order is None
            and not schedule.tiles
            and schedule.relayout is None
            and schedule.parallel_loop is None
        )
        # ``assembly`` is deliberately NOT part of this test, and that was
        # measured rather than assumed.  Treating a schedule that carries one as
        # explicit makes three of the four strategies unreachable: they apply
        # only to sparse-output programs, every such program needs an
        # accumulation workspace, and ``WorkspaceInsertion`` is an
        # automatic-provenance decision that the explicit path has no way to
        # record -- so the explicit route produced a CIN with a workspace and a
        # plan without the fact, and all 24 legal cells failed
        # ``unsupported_program_shape`` / ``sparse_parent_dominance``.  A
        # requested strategy therefore rides on the automatic plan as an
        # ATTESTED decision, exactly as the cost-model loop order does: recorded,
        # legality-verified, and not re-derived.  The origin still chooses
        # nothing, so an automatic request without one records ``None`` and emits
        # today's kernel byte for byte.
        #
        # ``accumulator`` is out of the test for exactly the same reason: the
        # structure is a property of the accumulation workspace, so a schedule
        # carrying only one must reach the route that inserts the workspace.
        if not is_identity and cin.get_workspace_accesses():
            raise ScorchNotImplementedError(
                "CIN with an existing workspace supports only an empty auto Schedule",
                code="schedule_scope_unsupported",
            )
        if len(cin.get_result_tensor_accesses()) == 1:
            return is_identity
        if (
            schedule.tiles
            or schedule.relayout is not None
            or schedule.parallel_loop is not None
        ):
            raise ScorchNotImplementedError(
                "Tiling, relayout, and explicit parallel scheduling require "
                "one assignment",
                code="schedule_scope_unsupported",
            )
        prospective_order = (
            Scheduler.select_loop_order(cin, costs=costs)
            if schedule.loop_order is None
            else Scheduler.resolve_loop_order(cin, schedule.loop_order)
        )
        if Scheduler.should_insert_workspace(cin, prospective_order):
            raise ScorchNotImplementedError(
                "Derived workspace scheduling requires one assignment",
                code="schedule_scope_unsupported",
            )
        return is_identity

    @staticmethod
    def _validate_stack_workspace_scope(
        loop_order: Sequence[IndexVar],
        reduction_names: Set[str],
        stack_targets: Set[str],
    ) -> None:
        """Reject a stack workspace that would replace the replay root."""

        if not stack_targets:
            return
        reduction_positions = [
            position
            for position, index_var in enumerate(loop_order)
            if index_var.name in reduction_names
        ]
        if reduction_positions and max(reduction_positions) == 0:
            raise ScorchNotImplementedError(
                "Stack tiling cannot wrap a workspace inserted at the root scope",
                code="stack_tiling_root_workspace_unsupported",
            )

    @staticmethod
    def _apply_schedule_legacy(
        cin: IndexStmt,
        schedule: Schedule,
        costs: Optional[_CostModelConstants] = None,
        compile_options: Optional[CompileOptions] = None,
        *,
        composed_order_candidate: bool = False,
        dense_results_any_order: bool = False,
    ) -> ScheduledCIN:
        """Apply an explicit tuner schedule to a CIN loop nest.

        An empty schedule delegates to :meth:`auto_schedule`, preserving the
        existing scheduler. A non-empty schedule owns loop order and tiling: no
        implicit tile heuristic is added on top of it.
        """
        if not isinstance(schedule, Schedule):
            raise ScorchTypeError(
                "apply_schedule expects a Schedule",
                code="internal_argument_type_invalid",
            )
        options, costs = _scheduler_costs_at_boundary(costs, compile_options)
        _validate_requested_schedule(options, schedule)
        source_cin = cin
        # Legacy scheduling still uses local tree surgery.  Keep that mutation
        # behind this boundary by applying it only to a private working copy.
        cin = copy.deepcopy(cin)
        if not isinstance(cin, ForAll):
            _refuse_loop_requests_without_loops(schedule)
            if (
                schedule.loop_order is not None
                or schedule.tiles
                or schedule.relayout is not None
                or schedule.parallel_loop is not None
            ):
                raise ScorchNotImplementedError(
                    "Non-empty schedules require a ForAll CIN statement",
                    code="schedule_without_loops_unsupported",
                )
            plan = verify_loop_plan(
                source_cin,
                LoopPlan(
                    loop_order=(),
                    auto_policy=Scheduler._auto_origin_policy(options.scheduler),
                    provenance="auto",
                    tag=schedule.tag,
                ),
            )
            return ScheduledCIN(cin, plan)
        is_identity = Scheduler._validate_legacy_replay_scope(cin, schedule, costs)
        panel_tiles = [tile for tile in schedule.tiles if tile.kind == "panel"]
        if len(panel_tiles) > 1:
            raise ScorchNotImplementedError(
                "Only one sparse panel tile is supported",
                code="multiple_panel_tiles_unsupported",
            )
        if panel_tiles and schedule.tiles[-1].kind != "panel":
            raise ScorchValueError(
                "A sparse panel tile must follow all affine tiles in Schedule.tiles",
                code="panel_tile_order_invalid",
            )
        if panel_tiles and not Scheduler._has_dense_output(cin):
            raise ScorchNotImplementedError(
                "Sparse panel tiling currently requires a dense result tensor",
                code="panel_tile_sparse_result_unsupported",
            )
        if panel_tiles and panel_tiles[0].parallel:
            raise ScorchValueError(
                "Sparse panel outer loops must be serial; select the row loop "
                "with Schedule.parallel_loop",
                code="panel_parallel_loop_invalid",
            )
        for tile in schedule.tiles:
            if tile.kind == "panel" and tile.accum != "direct":
                raise ScorchNotImplementedError(
                    "Sparse panel tiles currently require accum='direct'",
                    code="panel_tile_accumulation_unsupported",
                )

        logical_names = {index_var.name for index_var in cin.index_vars}
        unknown_tiles = [
            tile.index_var
            for tile in schedule.tiles
            if tile.index_var not in logical_names
        ]
        if unknown_tiles:
            raise ScorchValueError(
                f"Schedule tiles refer to unknown index variables {unknown_tiles}",
                code="schedule_loop_unknown",
            )

        index_var_getter = CINIndexVariablesGetter()
        index_var_getter.visit(cin)
        reduction_names = {
            index_var.name for index_var in index_var_getter.get_reduction_vars()
        }
        affine_reductions = [
            tile.index_var
            for tile in schedule.tiles
            if tile.kind == "affine" and tile.index_var in reduction_names
        ]
        if affine_reductions:
            raise ScorchNotImplementedError(
                "Affine reduction tiling requires an accumulator spanning outer "
                f"tiles; unsupported reduction variables: {affine_reductions}",
                code="reduction_tiling_accumulator_unsupported",
            )
        if any(tile.kind == "affine" for tile in schedule.tiles) and not (
            Scheduler._has_dense_output(cin)
        ):
            raise ScorchNotImplementedError(
                "Explicit affine tiling currently requires a dense result tensor; "
                "tiled sparse-output assembly is unsupported",
                code="tiling_sparse_result_unsupported",
            )
        Scheduler._require_legal_assembly_request(cin, schedule)
        Scheduler._require_legal_accumulator_request(cin, schedule)

        parallel_names = [tile.index_var for tile in schedule.tiles if tile.parallel]
        if schedule.parallel_loop is not None:
            parallel_names.append(schedule.parallel_loop)
        if parallel_names and not Scheduler._has_dense_output(cin):
            raise ScorchNotImplementedError(
                "Explicit parallel-loop selection currently requires a dense "
                "result tensor",
                code="explicit_parallel_sparse_result_unsupported",
            )
        tiled_inner_loops = {
            f"{tile.index_var}_in" for tile in schedule.tiles if tile.kind == "affine"
        }
        parallel_inner_loops = [
            name for name in parallel_names if name in tiled_inner_loops
        ]
        if parallel_inner_loops:
            raise ScorchValueError(
                "Tiled inner loops contain a ragged-tail break and cannot be "
                f"parallelized: {parallel_inner_loops}",
                code="parallel_ragged_tile_invalid",
            )
        parallel_reductions = []
        for parallel_name in parallel_names:
            logical_name = parallel_name
            for tile in schedule.tiles:
                if tile.kind != "affine":
                    continue
                if parallel_name in (
                    tile.index_var,
                    f"{tile.index_var}_out",
                    f"{tile.index_var}_in",
                ):
                    logical_name = tile.index_var
                    break
            if logical_name in reduction_names:
                parallel_reductions.append(parallel_name)
        if parallel_reductions:
            raise ScorchValueError(
                "Reduction loops cannot be selected for parallel execution: "
                f"{parallel_reductions}",
                code="parallel_reduction_loop_invalid",
            )

        result_tile_plan = Scheduler._validate_heap_result_tile(
            cin,
            schedule,
            reduction_names,
        )

        for tile in panel_tiles:
            target = Scheduler._find_index_var_by_name(cin, tile.index_var)
            compressed_accesses = [
                access
                for access in cin.tensor_accesses
                if access.has_index_var(target)
                and access.level_type_of_index_var(target) == LevelType.COMPRESSED
            ]
            if len(compressed_accesses) != 1:
                raise ScorchNotImplementedError(
                    f"Panel index {tile.index_var!r} must have exactly one "
                    "compressed tensor access",
                    code="panel_compressed_access_unsupported",
                )
            compressed_access = compressed_accesses[0]
            parent = compressed_access.get_parent_index_var(target)
            if (
                parent is None
                or compressed_access.level_type_of_index_var(parent) != LevelType.DENSE
            ):
                raise ScorchNotImplementedError(
                    "Sparse panel tiling currently requires a CSR-style "
                    "compressed level with a dense parent",
                    code="panel_tile_format_unsupported",
                )
            if schedule.parallel_loop != parent.name:
                raise ScorchValueError(
                    "Sparse panel tiling requires its CSR dense-parent row loop "
                    f"{parent.name!r} as Schedule.parallel_loop",
                    code="panel_parallel_loop_invalid",
                )
            if schedule.loop_order is None:
                panel_logical_order = [
                    index_var.name
                    for index_var in Scheduler.select_loop_order(
                        cin,
                        costs=costs,
                    )
                ]
            else:
                panel_logical_order = list(schedule.loop_order)
            if (
                parent.name in panel_logical_order
                and target.name in panel_logical_order
                and panel_logical_order.index(parent.name)
                > panel_logical_order.index(target.name)
            ):
                raise ScorchValueError(
                    "Sparse panel tiling requires the CSR row loop to precede "
                    "the compressed panel coordinate in logical loop_order",
                    code="panel_loop_order_invalid",
                )
            if tile.placement.startswith("at_depth:"):
                raise ScorchNotImplementedError(
                    "Sparse panel tiles do not support at_depth placement",
                    code="panel_tile_depth_placement_unsupported",
                )
            if tile.placement.startswith("child_of:"):
                placement_parent = tile.placement.split(":", 1)[1]
                if placement_parent in (
                    parent.name,
                    f"{parent.name}_out",
                    f"{parent.name}_in",
                ):
                    raise ScorchValueError(
                        "A sparse panel loop must be placed outside its parallel "
                        "CSR row loop",
                        code="panel_placement_invalid",
                    )
                affine_parent = next(
                    (
                        affine_tile
                        for affine_tile in schedule.tiles
                        if affine_tile.kind == "affine"
                        and f"{affine_tile.index_var}_out" == placement_parent
                    ),
                    None,
                )
                if affine_parent is None or affine_parent.placement != "outermost":
                    raise ScorchValueError(
                        "A child_of sparse panel placement must name an "
                        "outermost affine tile loop",
                        code="panel_placement_invalid",
                    )

        relayout_plan = Scheduler._validate_relayout(
            cin,
            schedule,
            panel_tiles,
            reduction_names,
        )

        if is_identity:
            scheduled_cin, plan = Scheduler._originate_auto_plan(
                cin,
                source_cin,
                options,
                costs,
                tag=schedule.tag,
                assembly=schedule.assembly,
                accumulator=schedule.accumulator,
                entries_of=_entries_of_symbol(source_cin, schedule.entries_of),
                composed_order_candidate=composed_order_candidate,
                dense_results_any_order=dense_results_any_order,
            )
            plan = _with_vector_loop_request(
                source_cin, plan, schedule, dense_results_any_order
            )
            return ScheduledCIN(scheduled_cin, plan)

        if schedule.loop_order is None:
            logical_order = Scheduler.select_loop_order(
                cin,
                costs=costs,
            )
        else:
            logical_order = Scheduler.resolve_loop_order(cin, schedule.loop_order)
        cin = Scheduler._rebuild_loop_nest(cin, logical_order)

        loop_order, _ = Scheduler._extract_loop_chain(cin)
        stack_tiles = [
            tile
            for tile in schedule.tiles
            if tile.kind == "affine" and tile.accum == "stack"
        ]
        unsupported_stack_tiles = [
            tile.index_var
            for tile in stack_tiles
            if not Scheduler._tile_target_needs_workspace(cin, tile.index_var)
        ]
        if unsupported_stack_tiles:
            raise ScorchNotImplementedError(
                "Stack accumulation is only supported for a trailing dense free "
                f"dimension after a reduction: {unsupported_stack_tiles}",
                code="stack_accumulation_layout_unsupported",
            )
        stack_targets = {tile.index_var for tile in stack_tiles}
        Scheduler._validate_stack_workspace_scope(
            loop_order,
            reduction_names,
            stack_targets,
        )
        if Scheduler.should_insert_workspace(cin, loop_order) and (
            not Scheduler._has_dense_output(cin) or stack_targets
        ):
            cin = Scheduler.insert_workspace(cin, allow_dense=True)

        generated_outer_names: Dict[str, str] = {}
        for tile in schedule.tiles:
            if tile.kind == "panel":
                continue
            target = Scheduler._find_index_var_by_name(cin, tile.index_var)
            cin = Scheduler.add_tile(
                cin=cin,
                index_var=target,
                tile_size=tile.width,
                placement=tile.placement,
                parallel=tile.parallel,
                unroll=tile.unroll,
                use_workspace=False,
                compile_options=options,
            )
            generated_outer_names[tile.index_var] = f"{tile.index_var}_out"

        if schedule.parallel_loop is not None:
            parallel_name = generated_outer_names.get(
                schedule.parallel_loop, schedule.parallel_loop
            )
            Scheduler._set_explicit_parallel_loop(cin, parallel_name)

        panel_bounds: List[PanelBound] = []
        for tile in panel_tiles:
            target = Scheduler._find_index_var_by_name(cin, tile.index_var)
            dense_accesses = [
                access
                for access in cin.tensor_accesses
                if access.has_index_var(target)
                and access.level_type_of_index_var(target) == LevelType.DENSE
            ]
            if not dense_accesses:
                raise ScorchNotImplementedError(
                    f"Panel index {tile.index_var!r} has no dense dimension bound",
                    code="panel_without_dense_bound_unsupported",
                )
            access = dense_accesses[0]
            level = access.level_of_index_var(target)
            panel_bounds.append(
                PanelBound(
                    loop=LoopRef(target.index_id),
                    tensor_id=access.tensor.symbol_id,
                    level=level,
                )
            )

        plan = _build_loop_plan(
            source_cin,
            schedule,
            logical_order,
            panel_bounds=panel_bounds,
            relayout_plan=relayout_plan,
            result_tile_plan=result_tile_plan,
            provenance="explicit",
        )
        return ScheduledCIN(cin, plan)

    @staticmethod
    def apply_schedule(
        cin: IndexStmt,
        schedule: Schedule,
        costs: Optional[_CostModelConstants] = None,
        compile_options: Optional[CompileOptions] = None,
        compilation_context: Optional[CompilationContext] = None,
        *,
        composed_order_candidate: bool = False,
        dense_results_any_order: bool = False,
    ) -> ScheduledCIN:
        """Pair detached semantic CIN with a verified stable-ID schedule.

        The legacy validator still performs private tree surgery. Its mutable
        result is discarded here and replayed only inside the lowering adapter.

        ``composed_order_candidate`` is for a caller whose lowering accepts a
        reduction innermost into a sparse result and requires every tensor in
        storage order -- the typed LoopIR pipeline.  It lets an automatic plan
        (an empty ``Schedule``) take the order composed before the forced reorder
        when that order keeps every tensor in storage order and the forced one
        does not; see :meth:`_originate_auto_plan`.  Legacy lowering cannot lower
        those orders, so every other caller leaves it off and its plans are
        unchanged.

        ``dense_results_any_order`` is for a caller whose lowering writes an
        all-dense result by its coordinates (the LoopIR pipeline): an automatic
        plan may then visit such a result's levels out of storage order.  Legacy
        lowering's dense-workspace consumers write at the workspace level as if
        it were the result's last, so every other caller leaves it off.
        """

        if type(composed_order_candidate) is not bool:
            raise ScorchTypeError(
                "composed_order_candidate must be an exact bool",
                code="internal_argument_type_invalid",
            )
        if type(dense_results_any_order) is not bool:
            raise ScorchTypeError(
                "dense_results_any_order must be an exact bool",
                code="internal_argument_type_invalid",
            )
        _validate_schedule_structure(schedule)
        options, costs = _scheduler_costs_at_boundary(costs, compile_options)
        if compile_options is None and options.requested_schedule is None:
            options = replace(options, requested_schedule=schedule)
        _validate_requested_schedule(options, schedule, exact=True)
        # Normalization owns the structural preflight, so it must run before
        # the display-name walk: the legacy-name validator traverses stored
        # forward edges recursively and would otherwise leak raw attribute
        # or recursion failures on a forged graph.  The clone preserves every
        # display name and stable ID, so validating the normalized graph is
        # semantically identical.
        normalized_cin = normalize_cin(
            cin,
            compile_options=options,
            compilation_context=compilation_context,
        )
        validate_legacy_cin_display_names(normalized_cin)
        stage_token = (
            compilation_context.begin_stage(
                CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION,
                compile_options=options,
            )
            if compilation_context is not None
            else None
        )
        try:
            with _trusted_normalized_cin(normalized_cin):
                try:
                    legacy_scheduled = Scheduler._apply_schedule_legacy(
                        normalized_cin,
                        schedule,
                        costs=costs,
                        compile_options=options,
                        composed_order_candidate=composed_order_candidate,
                        dense_results_any_order=dense_results_any_order,
                    )
                except (InvalidSchedule, UnsupportedFeature, VerificationError):
                    raise
                except NotImplementedError as exc:
                    raise UnsupportedFeature(
                        str(exc),
                        code=defect_code(exc) or "schedule_rewrite_unsupported",
                    ) from exc
                except ValueError as exc:
                    raise InvalidSchedule(
                        str(exc), code=defect_code(exc) or "schedule_rewrite_invalid"
                    ) from exc
                scheduled = ScheduledCIN(
                    normalized_cin,
                    legacy_scheduled.verified_loop_plan,
                )
        except Exception:
            if stage_token is not None and compilation_context is not None:
                compilation_context.fail_stage(stage_token)
            raise
        if stage_token is not None and compilation_context is not None:
            compilation_context.complete_stage(stage_token)
        return scheduled

    @staticmethod
    def _select_index_vars_to_tile(
        cin: CIN,
        scheduler_policy: SchedulerPolicy,
    ) -> List[IndexVar]:
        all_index_vars = cin.index_vars
        tensor_accesses = cin.tensor_accesses

        index_vars_to_tile: List[IndexVar] = []
        index_vars_sparse: Set[IndexVar] = set()

        # First, populate the list of index variables to tile by iterating
        # through each of the tensor access; if the tensor access does not
        # use all the index variables, then we add the index variables in
        # that tensor access corresponding to dense levels to the to tile
        # list.
        for tensor_access in tensor_accesses:
            tensor_access_index_vars = tensor_access.index_vars
            for index_var in tensor_access_index_vars:
                if tensor_access.level_type_of_index_var(index_var) != LevelType.DENSE:
                    index_vars_sparse.add(index_var)

            if set(tensor_access_index_vars) != set(all_index_vars):
                for index_var in tensor_access.index_vars:
                    if (
                        index_var not in index_vars_sparse
                        and index_var not in index_vars_to_tile
                    ):
                        index_vars_to_tile.append(index_var)

        if not cin.loop_order:
            return []

        first_loop_index_var = cin.loop_order[0]

        # We should remove the first loop index var from the list of index vars to tile
        # because tiling that doesn't help.
        if first_loop_index_var in index_vars_to_tile:
            index_vars_to_tile.remove(first_loop_index_var)

        # Remove the index variables that are sparse
        index_vars_to_tile = [
            index_var
            for index_var in index_vars_to_tile
            if index_var not in index_vars_sparse
        ]

        # Remove the index variables that are already in the no_tile_list
        # TODO: check that the condition for adding to no_tile_list is correct
        index_vars_to_tile = [
            index_var
            for index_var in index_vars_to_tile
            if index_var not in cin.no_tile_list
        ]

        # Remove index variables whose tiling would cause sparse re-traversal.
        # When tiling lifts the outer tile loop above a sparse loop, that sparse
        # loop is re-executed once per tile instead of once total.  For typical
        # problem sizes the re-traversal cost far outweighs the cache benefit
        # of a smaller tile working set.
        # The register-block path (SCORCH_REGBLOCK) DELIBERATELY tiles the free dim
        # even though it re-traverses the sparse contraction once per tile: the cost
        # model's "re-traversal loses" assumption is inverted when the output tile is
        # register-resident (the sparse indices are L1-hot across the ceil(N/T) passes,
        # and holding the accumulator in registers removes the per-nonzero output
        # round-trip). So skip this guard when regblock is on. Byte-identical when off.
        loop_order = cin.loop_order
        if loop_order and not scheduler_policy.regblock_enabled:

            def _causes_sparse_retraversal(iv: IndexVar) -> bool:
                if iv not in loop_order:
                    return False
                iv_pos = loop_order.index(iv)
                # Tiling lifts iv_out to position 1 (after the outermost loop).
                # Any sparse loop between position 1 and iv_pos would be nested
                # inside the tile loop, causing re-traversal.
                for pos in range(1, iv_pos):
                    if loop_order[pos] in index_vars_sparse:
                        return True
                return False

            index_vars_to_tile = [
                iv for iv in index_vars_to_tile if not _causes_sparse_retraversal(iv)
            ]

        return index_vars_to_tile

    @staticmethod
    def _apply_tiling_heuristics(
        cin: CIN,
        compile_options: CompileOptions,
        scheduler_policy: SchedulerPolicy,
        plan_tiles: Optional[List[LoopTile]] = None,
    ) -> CIN:
        if not isinstance(cin, ForAll):
            return cin
        # Preserve the legacy auto-scheduler's scope: before ``add_tile`` grew
        # into a general strip-mining transform it only changed loop nests that
        # had a workspace/Where region.  Direct affine tiling is intentionally
        # available through ``Schedule``, but must not silently change default
        # schedules such as reduction-innermost SDDMM.
        if not cin.inserted_workspace:
            return cin
        # An affine tile over a non-dense result is illegal, and the scheduler
        # already says so -- ``apply_schedule`` refuses exactly this composition
        # for an explicitly requested schedule ("Explicit affine tiling currently
        # requires a dense result tensor; tiled sparse-output assembly is
        # unsupported").  That check reads ``schedule.tiles``, which is empty on
        # the automatic origin, so it is vacuous precisely where the automatic
        # heuristic then goes on to choose tiles itself.  The automatic path was
        # constructing the artifact the explicit path declares unsupported, and
        # nothing downstream re-checked it.
        #
        # The illegality is not theoretical.  Strip-mining a dense prefix loop and
        # lifting the tile loop (OUTERMOST, or CHILD_OF the root on the regblock
        # arm) stops the nest visiting a compressed result level's parent prefix
        # in lexicographic order, so the assembled position array is malformed.
        #
        # Byte-neutral wherever tiling is legal, by construction: the guard cannot
        # fire on a dense receiver, and it answers "not proven" -- leaving
        # behaviour unchanged -- for any receiver shape it does not recognize.
        if Scheduler._has_proven_non_dense_receiver(cin):
            return cin
        regblock_enabled = scheduler_policy.regblock_enabled
        tile_width = (
            scheduler_policy.regblock_tile_width
            if regblock_enabled
            else scheduler_policy.auto_tile_width
        )
        for index_var in Scheduler._select_index_vars_to_tile(cin, scheduler_policy):
            if regblock_enabled:
                parent_ref = _loop_ref_from_legacy_index_var(cin.index_var)
                placement = f"child_of:{cin.index_var.name}"
                loop_placement = LoopPlacement(
                    PlacementKind.CHILD_OF,
                    parent=parent_ref,
                )
            else:
                placement = "outermost"
                loop_placement = LoopPlacement(PlacementKind.OUTERMOST)
            cin = Scheduler.add_tile(
                cin,
                index_var,
                tile_width,
                placement=placement,
                compile_options=compile_options,
            )
            if plan_tiles is not None:
                plan_tiles.append(
                    LoopTile(
                        loop=LoopRef(index_var.index_id),
                        width=tile_width,
                        placement=loop_placement,
                        parallel=False,
                        kind="affine",
                        accumulation="direct",
                        unroll=True,
                    )
                )
        return cin

    @staticmethod
    def _auto_origin_policy(scheduler_policy: SchedulerPolicy) -> AutoOriginPolicy:
        """The typed origin fact for one automatic-scheduling arm.

        Records exactly the policy inputs the heuristics consume so plan
        verification can re-derive every tile and workspace decision:
        which regblock arm ran and the tile width that arm applies.
        """

        return AutoOriginPolicy(
            schema=AUTO_ORIGIN_POLICY_SCHEMA,
            regblock_enabled=scheduler_policy.regblock_enabled,
            tile_width=(
                scheduler_policy.regblock_tile_width
                if scheduler_policy.regblock_enabled
                else scheduler_policy.auto_tile_width
            ),
        )

    @staticmethod
    def _refused_for_order_legality(error: InvalidSchedule) -> bool:
        """Whether one plan refusal names a rule about the loop ORDER alone.

        ``loop_plan_legality._verify_storage_order`` owns exactly two such
        rules -- a non-workspace result's own physical storage order, and every
        sparse access's physical-parent dominance -- and both are properties of
        the recorded order rather than of the program.  A refusal naming one of
        them is therefore a statement that the order the automatic origin chose
        is illegal, which is the only refusal a different legal order can
        answer.  Every other code stays untouched, including the tile and
        workspace decisions the same layer pins to a re-derived heuristic.
        """

        diagnostics = getattr(error, "diagnostics", ())
        if type(diagnostics) is not tuple:
            return False
        return any(
            getattr(diagnostic, "code", None) in _ORDER_LEGALITY_CODES
            for diagnostic in diagnostics
        )

    @staticmethod
    def _verified_auto_plan(
        working: CIN,
        plan_source: CIN,
        logical_order: List[IndexVar],
        options: CompileOptions,
        scheduler_policy: SchedulerPolicy,
        *,
        tag: str,
        require_complete_plan: bool,
        assembly: Optional[str] = None,
        accumulator: Optional[str] = None,
        entries_of: Optional[SymbolId] = None,
        dense_results_any_order: bool = False,
    ) -> Tuple[CIN, LoopPlan]:
        """Apply one automatic order and verify the plan that records it.

        ``assembly`` is the sparse-output assembly strategy the public schedule
        REQUESTED, carried onto the automatic plan as an attested decision -- the
        same standing the cost-model loop order has: recorded, legality-verified
        at the plan boundary, and not re-derived.  The automatic origin itself
        chooses no strategy, so an ordinary automatic compilation records ``None``
        here and emits exactly the kernel it emitted before the field existed.

        ``accumulator`` rides along on exactly the same terms and for exactly the
        same reason: the structure applies only to sparse-output programs, whose
        workspace insertion is a decision only the automatic path can record.

        ``entries_of`` (the input whose entries a COO result stores) is recorded
        with them because the assembly's legality reads it: a parallel single
        pass is legal only for a result that stores a named input's entries, so
        the plan must carry the name when the boundary verifies the assembly.
        """

        auto_tiles: List[LoopTile] = []
        auto_workspace: List[WorkspaceInsertion] = []
        scheduled_cin = Scheduler._apply_auto_order_owned(
            working,
            logical_order,
            options,
            scheduler_policy=scheduler_policy,
            plan_tiles=auto_tiles,
            plan_workspace=auto_workspace,
            require_complete_plan=require_complete_plan,
        )
        plan = verify_loop_plan(
            plan_source,
            LoopPlan(
                loop_order=tuple(index_var.index_id for index_var in logical_order),
                tiles=tuple(auto_tiles),
                workspace=auto_workspace[0] if auto_workspace else None,
                assembly=assembly,
                accumulator=accumulator,
                entries_of=entries_of,
                auto_policy=Scheduler._auto_origin_policy(scheduler_policy),
                provenance="auto",
                tag=tag,
            ),
            dense_results_any_order=dense_results_any_order,
        )
        return scheduled_cin, plan

    @staticmethod
    def _originate_auto_plan(
        working: CIN,
        plan_source: CIN,
        options: CompileOptions,
        costs: _CostModelConstants,
        *,
        scheduler_policy: Optional[SchedulerPolicy] = None,
        tag: str = "",
        require_complete_plan: bool = False,
        assembly: Optional[str] = None,
        accumulator: Optional[str] = None,
        entries_of: Optional[SymbolId] = None,
        composed_order_candidate: bool = False,
        dense_results_any_order: bool = False,
    ) -> Tuple[CIN, LoopPlan]:
        """Originate one verified automatic plan at a plan-producing boundary.

        :meth:`select_loop_order` ends with an unchecked forced reorder that
        moves the last free variable inward so that a free variable follows the
        last reduction -- the nest shape legacy lowering and workspace insertion
        require.  Nothing constrains that block, and for a sparse result whose
        every coordinate is already bound above the outermost reduction it
        produces an order the plan's own legality rules refuse: moving one
        result coordinate inward puts a compressed physical parent below its
        child.  Such a program needs no workspace at all, so the reorder buys
        nothing and costs legality.

        When exactly that happens -- the recorded plan refused for an
        order-legality rule, and the composition's own pre-forced order
        different from the forced one -- the plan is re-originated from the
        pre-forced order and offered to the same trust boundary, which accepts
        or refuses it on its own terms.  Two properties make this narrow:

        * It defers to the legality layer instead of restating either the
          reorder or the rules it breaks, so no policy is duplicated here.
        * It runs only after a refusal, so a program that produces generated
          code today cannot reach it; and if the pre-forced order is refused
          too, the original refusal is the one reported.

        ``composed_order_candidate`` adds the pre-forced order as a second
        candidate for a lowering that can use it (the typed LoopIR pipeline).
        The plan's legality rules check the parent-before-child order only of
        tensors with a compressed level, while that lowering requires every
        tensor's storage order, so a forced order that reads an all-dense
        operand against its storage order verifies here and is refused there
        (``unsupported_loop_order``) -- SDDMM into CSR or DCSR is one such
        program.  The verified forced plan is kept unless all of these hold:
        the forced reorder changed the order, the forced order visits some
        tensor's levels out of storage order, the pre-forced order visits every
        tensor's levels in storage order, and the plan re-originated from the
        pre-forced order verifies.  Then that plan is returned instead.  The
        forced order stays the first candidate and is returned whenever any
        condition fails, so this adds an order beside it and never removes one.

        The cost model can also compose an order that reads a dense operand
        against its storage order on its own, without the forced reorder; which
        order it composes then depends on the order of the caller's loop nest.
        When neither the forced nor the pre-forced order visits every tensor in
        storage order, the flag adds a third candidate: the pre-forced order
        re-sorted, as little as possible, so that it does
        (:meth:`_storage_order_sort`).  It is taken when it exists and its plan
        verifies.  LoopIR lowering refuses every order that breaks a tensor's
        storage order, so this candidate can only replace an order that
        pipeline would refuse.  Off, the origin is exactly as described above.
        """

        policy = options.scheduler if scheduler_policy is None else scheduler_policy
        pre_forced: List[IndexVar] = []
        logical_order = Scheduler.select_loop_order(
            working, costs=costs, pre_forced_order=pre_forced
        )
        reordered = [index_var.index_id for index_var in pre_forced] != [
            index_var.index_id for index_var in logical_order
        ]

        def reoriginate(in_storage_order: bool) -> Optional[Tuple[CIN, LoopPlan]]:
            return Scheduler._from_pre_forced_order(
                plan_source,
                options,
                costs,
                policy,
                tag=tag,
                require_complete_plan=require_complete_plan,
                assembly=assembly,
                accumulator=accumulator,
                entries_of=entries_of,
                in_storage_order=in_storage_order,
                dense_results_any_order=dense_results_any_order,
            )

        forced: Optional[Tuple[CIN, LoopPlan]] = None
        refusal: Optional[InvalidSchedule] = None
        try:
            forced = Scheduler._verified_auto_plan(
                working,
                plan_source,
                logical_order,
                options,
                policy,
                tag=tag,
                require_complete_plan=require_complete_plan,
                assembly=assembly,
                accumulator=accumulator,
                entries_of=entries_of,
                dense_results_any_order=dense_results_any_order,
            )
        except InvalidSchedule as error:
            # Only a refusal for an order rule can be answered by another order.
            if not Scheduler._refused_for_order_legality(error) or (
                not reordered and not composed_order_candidate
            ):
                raise
            refusal = error
        if composed_order_candidate:
            if forced is not None and Scheduler._visits_every_tensor_in_storage_order(
                plan_source, logical_order
            ):
                return forced
            composed_in_storage_order = Scheduler._visits_every_tensor_in_storage_order(
                plan_source, pre_forced
            )
            if reordered and composed_in_storage_order:
                composed = reoriginate(False)
                if composed is not None:
                    return composed
            if not composed_in_storage_order:
                sorted_plan = reoriginate(True)
                if sorted_plan is not None:
                    return sorted_plan
        if forced is not None:
            return forced
        if refusal is None:
            raise ScorchAssertionError(
                "refusal is not None", code="auto_plan_refusal_missing"
            )
        if not reordered:
            raise refusal
        repaired = reoriginate(False)
        if repaired is None:
            raise refusal
        return repaired

    @staticmethod
    def _from_pre_forced_order(
        plan_source: CIN,
        options: CompileOptions,
        costs: _CostModelConstants,
        policy: SchedulerPolicy,
        *,
        tag: str,
        require_complete_plan: bool,
        assembly: Optional[str],
        accumulator: Optional[str],
        entries_of: Optional[SymbolId] = None,
        in_storage_order: bool = False,
        dense_results_any_order: bool = False,
    ) -> Optional[Tuple[CIN, LoopPlan]]:
        """The verified plan of the pre-forced order on a fresh copy, or None.

        With ``in_storage_order`` the order is first re-sorted by
        :meth:`_storage_order_sort`; None also when no such order exists.
        """

        repaired = copy.deepcopy(plan_source)
        repaired_order: List[IndexVar] = []
        Scheduler.select_loop_order(
            repaired, costs=costs, pre_forced_order=repaired_order
        )
        if in_storage_order:
            resorted = Scheduler._storage_order_sort(repaired, repaired_order)
            if resorted is None:
                return None
            repaired_order = resorted
        try:
            return Scheduler._verified_auto_plan(
                repaired,
                plan_source,
                repaired_order,
                options,
                policy,
                tag=tag,
                require_complete_plan=require_complete_plan,
                assembly=assembly,
                accumulator=accumulator,
                entries_of=entries_of,
                dense_results_any_order=dense_results_any_order,
            )
        except (InvalidSchedule, UnsupportedFeature, VerificationError):
            return None

    @staticmethod
    def _storage_order_sort(
        cin: CIN, order: Sequence[IndexVar]
    ) -> Optional[List[IndexVar]]:
        """``order`` re-sorted so that it binds every tensor's levels in storage order.

        A stable topological sort: each tensor's storage order (its access's
        index variables permuted by its mode order) requires each level's
        variable before the next level's, and among the variables whose
        predecessors are all placed, the one earliest in ``order`` goes next.
        So an order that already satisfies every tensor comes back unchanged,
        and otherwise each variable moves only as far as a storage order forces
        it.  None when the tensors' storage orders contradict one another.
        """

        before: Dict[Any, Set[Any]] = {index_var.index_id: set() for index_var in order}
        for access in cin.tensor_accesses:
            levels = [
                index_var.index_id
                for index_var in access.get_sorted_index_vars()
                if index_var.index_id in before
            ]
            for parent, child in zip(levels, levels[1:]):
                before[child].add(parent)
        placed: Set[Any] = set()
        remaining = list(order)
        resorted: List[IndexVar] = []
        while remaining:
            ready = next((v for v in remaining if before[v.index_id] <= placed), None)
            if ready is None:
                return None
            remaining.remove(ready)
            resorted.append(ready)
            placed.add(ready.index_id)
        return resorted

    @staticmethod
    def _visits_every_tensor_in_storage_order(
        cin: CIN, order: Sequence[IndexVar]
    ) -> bool:
        """Whether ``order`` binds each tensor's level coordinates outermost first.

        A tensor's storage order is its access's index variables permuted by its
        mode order.  An order that binds a deeper level's variable above a
        shallower one's reads that tensor against its storage order.  Variables
        the order does not bind are ignored.
        """

        position = {index_var.index_id: n for n, index_var in enumerate(order)}
        for access in cin.tensor_accesses:
            visited = [
                position[index_var.index_id]
                for index_var in access.get_sorted_index_vars()
                if index_var.index_id in position
            ]
            if visited != sorted(visited):
                return False
        return True

    @staticmethod
    def _apply_auto_order_owned(
        cin: CIN,
        loop_order: List[IndexVar],
        compile_options: CompileOptions,
        scheduler_policy: Optional[SchedulerPolicy] = None,
        plan_tiles: Optional[List[LoopTile]] = None,
        plan_workspace: Optional[List[WorkspaceInsertion]] = None,
        require_complete_plan: bool = False,
    ) -> CIN:
        """Apply selected auto decisions on a CIN owned by the scheduler."""

        if type(require_complete_plan) is not bool:
            raise ScorchTypeError(
                "require_complete_plan must be an exact bool",
                code="internal_argument_type_invalid",
            )
        if require_complete_plan and (
            type(plan_tiles) is not list
            or type(plan_workspace) is not list
            or plan_tiles is plan_workspace
            or plan_tiles
            or plan_workspace
        ):
            raise ScorchTypeError(
                "require_complete_plan needs empty exact plan tile and "
                "workspace sinks",
                code="internal_argument_type_invalid",
            )
        if not isinstance(cin, ForAll):
            return cin
        if scheduler_policy is None:
            scheduler_policy = compile_options.scheduler

        cin = Scheduler._rebuild_loop_nest(cin, loop_order)

        recorded: Optional[WorkspaceInsertion] = None
        dense_output = Scheduler._has_dense_output(cin)
        if Scheduler.should_insert_workspace(cin, loop_order):
            # For dense outputs the workspace only exists to support tiling.
            # If nothing will be tiled, the workspace is pure overhead
            # (extra memset + copy-back per iteration).
            will_tile = (
                len(Scheduler._select_index_vars_to_tile(cin, scheduler_policy)) > 0
            )
            insert = not dense_output or will_tile
            if insert and dense_output:
                # A dense-output insertion at the nest root demotes the root
                # to a Where, so the candidate tiles that justified the
                # workspace can never materialize — and the legacy emission of
                # that composition mixes the dense and sparse workspace APIs
                # over one undeclared symbol, so it has never compiled
                # (retained einsum('jk->k') clang evidence).  Elide exactly as
                # the established empty-Schedule replay contract does, keeping
                # the plan-free production route and replay in agreement.
                root_getter = CINIndexVariablesGetter()
                root_getter.visit(cin)
                reductions_in_order = [
                    var
                    for var in root_getter.get_reduction_vars()
                    if var in cin.loop_order
                ]
                if reductions_in_order and cin.index_var == reductions_in_order[-1]:
                    insert = False
            if insert:
                # Recording is a plan-construction concern.  Ordinary release
                # auto-scheduling owns only the legacy mutation and must not
                # depend on this helper when no recording sink was requested.
                if plan_workspace is not None:
                    recorded = Scheduler._workspace_insertion_record(cin)
                cin = Scheduler.insert_workspace(cin, allow_dense=True)
                if plan_workspace is not None:
                    materialized = {
                        (access.tensor.symbol_id, access.tensor.dense)
                        for access in cin.get_workspace_accesses()
                    }
                    if (
                        recorded is None
                        or not cin.inserted_workspace
                        or len(materialized) != 1
                        or next(iter(materialized))[1] is not recorded.dense
                    ):
                        raise VerificationError(
                            "stage=auto plan recording: the workspace "
                            "insertion decision did not materialize as derived",
                            code="auto_plan_replay_mismatch",
                        )

        if isinstance(cin, ForAll):
            cin = Scheduler._apply_tiling_heuristics(
                cin,
                compile_options,
                scheduler_policy,
                plan_tiles=plan_tiles,
            )
        if plan_workspace is not None and recorded is not None:
            # The root-scope dense elision above makes production surgery and
            # the ``ScheduledCIN`` replay contract agree by construction: any
            # materialized dense-output workspace now has candidate tiles that
            # actually materialize.  The guard below is defense in depth — if
            # a future heuristic change reintroduces a materialized dense
            # workspace whose tiles bailed, recording ``None`` would falsely
            # claim a complete recording, so complete-plan mode fails closed
            # instead.
            if not dense_output or (plan_tiles is not None and plan_tiles):
                plan_workspace.append(recorded)
            elif require_complete_plan:
                raise UnsupportedFeature(
                    "stage=auto plan recording: a materialized dense "
                    "workspace without derived tiles is not represented by "
                    "the automatic LoopPlan replay contract",
                    code="auto_plan_dense_workspace_unrecorded",
                )
        return cin

    @staticmethod
    def _auto_schedule_owned(
        cin: CIN,
        compile_options: CompileOptions,
        costs: Optional[_CostModelConstants] = None,
        scheduler_policy: Optional[SchedulerPolicy] = None,
        plan_tiles: Optional[List[LoopTile]] = None,
        plan_workspace: Optional[List[WorkspaceInsertion]] = None,
    ) -> CIN:
        """Select policy once, then mutate only scheduler-owned CIN."""

        if not isinstance(cin, ForAll):
            return cin
        if scheduler_policy is None:
            scheduler_policy = compile_options.scheduler
        if costs is None:
            costs = scheduler_policy.cost_model
        loop_order = Scheduler.select_loop_order(
            cin,
            costs=costs,
        )
        return Scheduler._apply_auto_order_owned(
            cin,
            loop_order,
            compile_options,
            scheduler_policy=scheduler_policy,
            plan_tiles=plan_tiles,
            plan_workspace=plan_workspace,
        )

    @staticmethod
    def _replay_auto_plan_owned(
        cin: IndexStmt,
        plan: LoopPlan,
        compile_options: Optional[CompileOptions] = None,
    ) -> IndexStmt:
        """Replay recorded auto decisions without rerunning scheduling policy."""

        options = _compile_options_at_scheduler_boundary(compile_options)
        verify_loop_plan(cin, plan)
        if not isinstance(cin, ForAll):
            if plan.loop_order or plan.tiles:
                raise VerificationError(
                    "stage=legacy auto replay: non-loop CIN has loop decisions",
                    code="auto_plan_replay_mismatch",
                )
            return cin
        if (
            plan.panel_bounds
            or plan.relayout is not None
            or plan.result_tile is not None
            or plan.parallel_loop is not None
        ):
            raise VerificationError(
                "stage=legacy auto replay: auto LoopPlan contains explicit-only "
                "decisions",
                code="auto_plan_replay_mismatch",
            )

        index_names, _ = entity_display_names(cin)
        logical_order = [
            Scheduler._find_index_var_by_id(cin, index_id)
            for index_id in plan.loop_order
        ]
        cin = Scheduler._rebuild_loop_nest(cin, logical_order)

        # Replay consumes the recorded workspace decision instead of
        # re-deriving hidden scheduler-policy state; the plan verifier has
        # already proved the stored fact equals the derived decision, and the
        # replayed nest is cross-checked against it before materialization.
        if plan.workspace is not None:
            derived = Scheduler._workspace_insertion_record(cin)
            if derived != plan.workspace:
                raise VerificationError(
                    "stage=legacy auto replay: the recorded workspace "
                    "insertion does not match the replayed nest",
                    code="auto_plan_replay_mismatch",
                )
            cin = Scheduler.insert_workspace(cin, allow_dense=True)
            if not cin.inserted_workspace:
                raise VerificationError(
                    "stage=legacy auto replay: the recorded workspace "
                    "insertion did not materialize",
                    code="auto_plan_replay_mismatch",
                )

        for tile in plan.tiles:
            if tile.loop.part != LoopPart.LOGICAL or tile.kind != "affine":
                raise VerificationError(
                    "stage=legacy auto replay: unsupported recorded auto tile",
                    code="auto_plan_replay_mismatch",
                )
            target = Scheduler._find_index_var_by_id(cin, tile.loop.index_id)
            cin = Scheduler.add_tile(
                cin=cin,
                index_var=target,
                tile_size=tile.width,
                placement=_render_placement(tile.placement, index_names),
                parallel=tile.parallel,
                unroll=tile.unroll,
                use_workspace=False,
                compile_options=options,
            )
        return cin

    @staticmethod
    def auto_schedule(
        cin: CIN,
        costs: Optional[_CostModelConstants] = None,
        compile_options: Optional[CompileOptions] = None,
        compilation_context: Optional[CompilationContext] = None,
    ) -> CIN:
        """Auto-schedule without mutating caller-owned CIN state."""

        options, costs = _scheduler_costs_at_boundary(costs, compile_options)
        _validate_requested_schedule(options, None)
        return Scheduler._auto_schedule_boundary_owned(
            cin,
            options,
            costs,
            options.scheduler,
            compilation_context=compilation_context,
        )

    @staticmethod
    def auto_schedule_plan(
        cin: CIN,
        costs: Optional[_CostModelConstants] = None,
        compile_options: Optional[CompileOptions] = None,
        compilation_context: Optional[CompilationContext] = None,
        *,
        regblock_enabled: Optional[bool] = None,
    ) -> ScheduledCIN:
        """Originate a verified automatic LoopPlan at the auto boundary.

        The plan-producing twin of :meth:`auto_schedule` and one regblock
        arm of the internal dual-path algorithm: the same policy selects the
        loop order, workspace insertion, and heuristic tiles, but every
        decision is recorded as an explicit fact of one verified
        ``provenance="auto"`` LoopPlan instead of surviving only as private
        tree surgery.  The mutable surgery result is discarded exactly as
        :meth:`apply_schedule` discards it; release dispatch does not consume
        this entry yet.  The dense root-scope reduction family records the
        elided decision (``workspace=None``): plan-free production surgery
        now elides that insertion exactly as the empty-Schedule replay
        contract always has, because the abandoned materialized form never
        produced compilable C++ (retained einsum('jk->k') clang evidence).
        """

        options, costs = _scheduler_costs_at_boundary(costs, compile_options)
        _validate_requested_schedule(options, None)
        scheduler_policy = options.scheduler
        if regblock_enabled is not None:
            if type(regblock_enabled) is not bool:
                raise TypeError("regblock_enabled must be an exact bool")
            scheduler_policy = replace(
                scheduler_policy,
                regblock_enabled=regblock_enabled,
            )
            costs = scheduler_policy.cost_model
        if not _is_index_stmt_instance(cin):
            raise TypeError("auto_schedule_plan expects an IndexStmt")
        # Normalization owns the structural preflight; validate display names
        # on the normalized clone so a forged graph fails closed first (the
        # clone preserves every display name and stable ID).
        normalized = normalize_cin(
            cin,
            compile_options=options,
            compilation_context=compilation_context,
        )
        validate_legacy_cin_display_names(normalized)
        stage_token = (
            compilation_context.begin_stage(
                CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION,
                compile_options=options,
            )
            if compilation_context is not None
            else None
        )
        try:
            with _trusted_normalized_cin(normalized):
                working = copy.deepcopy(normalized)
                if not isinstance(working, ForAll):
                    plan = verify_loop_plan(
                        normalized,
                        LoopPlan(
                            loop_order=(),
                            auto_policy=Scheduler._auto_origin_policy(scheduler_policy),
                            provenance="auto",
                        ),
                    )
                else:
                    _, plan = Scheduler._originate_auto_plan(
                        working,
                        normalized,
                        options,
                        costs,
                        scheduler_policy=scheduler_policy,
                        require_complete_plan=True,
                    )
                scheduled = ScheduledCIN(normalized, plan)
        except Exception:
            if stage_token is not None and compilation_context is not None:
                compilation_context.fail_stage(stage_token)
            raise
        if stage_token is not None and compilation_context is not None:
            compilation_context.complete_stage(stage_token)
        return scheduled

    @staticmethod
    def _auto_schedule_regblock_arm(
        cin: CIN,
        *,
        enabled: bool,
        compile_options: CompileOptions,
        compilation_context: Optional[CompilationContext] = None,
    ) -> CIN:
        """Run one deterministic arm of the internal dual-path algorithm."""

        if type(enabled) is not bool:
            raise ScorchTypeError(
                "enabled must be an exact bool", code="internal_argument_type_invalid"
            )
        if type(compile_options) is not CompileOptions:
            raise ScorchTypeError(
                "compile_options must be an exact CompileOptions",
                code="internal_argument_type_invalid",
            )
        _validate_requested_schedule(compile_options, None)
        scheduler_policy = replace(
            compile_options.scheduler,
            regblock_enabled=enabled,
        )
        return Scheduler._auto_schedule_boundary_owned(
            cin,
            compile_options,
            scheduler_policy.cost_model,
            scheduler_policy,
            compilation_context=compilation_context,
        )

    @staticmethod
    def _auto_schedule_boundary_owned(
        cin: CIN,
        options: CompileOptions,
        costs: _CostModelConstants,
        scheduler_policy: SchedulerPolicy,
        compilation_context: Optional[CompilationContext] = None,
    ) -> CIN:
        """Shared owned implementation after the public boundary is resolved."""

        if not _is_index_stmt_instance(cin):
            raise ScorchTypeError(
                "auto_schedule expects an IndexStmt",
                code="internal_argument_type_invalid",
            )
        # Normalization owns the structural preflight; validate display names
        # on the normalized clone so a forged graph fails closed first (the
        # clone preserves every display name and stable ID).
        working = normalize_cin(
            cin,
            compile_options=options,
            compilation_context=compilation_context,
        )
        validate_legacy_cin_display_names(working)
        stage_token = (
            compilation_context.begin_stage(
                CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION,
                compile_options=options,
            )
            if compilation_context is not None
            else None
        )
        try:
            with _trusted_normalized_cin(working):
                scheduled = Scheduler._auto_schedule_owned(
                    working,
                    options,
                    costs=costs,
                    scheduler_policy=scheduler_policy,
                )
                if isinstance(scheduled, IndexStmt):
                    validate_legacy_cin_display_names(scheduled)
        except Exception:
            if stage_token is not None and compilation_context is not None:
                compilation_context.fail_stage(stage_token)
            raise
        if stage_token is not None and compilation_context is not None:
            compilation_context.complete_stage(stage_token)
        return scheduled
