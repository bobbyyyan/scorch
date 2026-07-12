import copy
import math
import os
from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from scorch.compiler.cin import (
    BinaryOp,
    CIN,
    CINIndexVariablesGetter,
    CINVisitorAccept,
    ForAll,
    IndexVar,
    Operation,
    TensorAccess,
    TensorAssign,
    TileSizeVar,
    Where,
    Workspace,
    WorkspaceAccess,
)
from scorch.format import LevelType

# Per-call override for the register-block decision. When not None it takes
# precedence over the SCORCH_REGBLOCK env var. Phase 2b lowers the SAME CIN twice
# (regblock forced OFF -> baseline nest, forced ON -> tiled nest) and stitches a
# runtime free-dim branch (see ops._build_regblock_dual_path); the two lowerings
# must be able to select their path independently of the process-wide env.
_REGBLOCK_FORCE: Optional[bool] = None


def _regblock_enabled() -> bool:
    """SCORCH_REGBLOCK=1 opts into the free-dim register-block tiling (Phase 2 of
    the codegen-parity work): tile the dense free/output dim of a sparse contraction
    so the output tile accumulates in a stack-local buffer across the reduction
    (register-eligible) instead of round-tripping memory per nonzero.

    A per-call `regblock_force(...)` override wins over the env var when set.

    Default OFF -> codegen is byte-identical to before. See codegen-parity notes.
    """
    if _REGBLOCK_FORCE is not None:
        return _REGBLOCK_FORCE
    return os.environ.get("SCORCH_REGBLOCK", "0") == "1"


@contextmanager
def regblock_force(value: Optional[bool]):
    """Temporarily force `_regblock_enabled()` to `value` (or restore env-driven
    behavior with None). Used by the Phase 2b dual-path builder to lower the baseline
    and register-block nests from one CIN. Nestable and exception-safe."""
    global _REGBLOCK_FORCE
    prev = _REGBLOCK_FORCE
    _REGBLOCK_FORCE = value
    try:
        yield
    finally:
        _REGBLOCK_FORCE = prev


def _regblock_max_n() -> int:
    """Runtime free-dim cutoff for the dual-path branch (SCORCH_REGBLOCK_MAX_N,
    default 8): the register-block arm runs only when the free/output dim size is
    <= this. It is ALSO used as the tile width so the regblock arm is always a
    single tile (no sparse re-traversal, no ragged tail). Data (doc 04): N<=3 wins,
    N>=16 regresses; 8 places the threshold between them (tune with N in {4,8})."""
    try:
        n = int(os.environ.get("SCORCH_REGBLOCK_MAX_N", "8"))
    except ValueError:
        return 8
    return n if n > 0 else 8


def _regblock_tile_width() -> int:
    """Free-dim tile width for the register-block path. Defaults to REGBLOCK_MAX_N so
    the regblock arm is single-tile; SCORCH_REGBLOCK_T overrides for experiments."""
    env = os.environ.get("SCORCH_REGBLOCK_T")
    if env is not None:
        try:
            t = int(env)
            if t > 0:
                return t
        except ValueError:
            pass
    return _regblock_max_n()


@dataclass(frozen=True)
class _CostModelConstants:
    alpha: float = 2.975
    beta: float = 0.1005
    gamma: float = 43.55
    c_insert: float = 85.34
    c_sort: float = 1.741
    c_trans: float = 40.61
    rho: float = 0.0014
    default_dim_size: int = 1024


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
        if not isinstance(self.index_var, str) or not self.index_var:
            raise ValueError("TileSpec.index_var must be a non-empty string")
        if isinstance(self.width, bool) or not isinstance(self.width, int):
            raise TypeError("TileSpec.width must be an integer")
        if self.width <= 0:
            raise ValueError("TileSpec.width must be greater than zero")
        if not isinstance(self.placement, str):
            raise TypeError("TileSpec.placement must be a string")
        if not isinstance(self.parallel, bool):
            raise TypeError("TileSpec.parallel must be a bool")
        if not isinstance(self.kind, str):
            raise TypeError("TileSpec.kind must be a string")
        if self.kind not in ("affine", "panel"):
            raise ValueError("TileSpec.kind must be 'affine' or 'panel'")
        if not isinstance(self.accum, str):
            raise TypeError("TileSpec.accum must be a string")
        if self.accum not in ("stack", "direct", "heap"):
            raise ValueError("TileSpec.accum must be 'stack', 'direct', or 'heap'")
        if not isinstance(self.unroll, bool):
            raise TypeError("TileSpec.unroll must be a bool")
        if not (
            self.placement == "outermost"
            or self.placement.startswith("child_of:")
            or self.placement.startswith("at_depth:")
        ):
            raise ValueError(
                "TileSpec.placement must be 'outermost', 'child_of:<var>', "
                "or 'at_depth:<n>'"
            )
        if self.placement.startswith("child_of:"):
            if not self.placement.split(":", 1)[1]:
                raise ValueError("child_of placement requires an index variable")
        elif self.placement.startswith("at_depth:"):
            depth = self.placement.split(":", 1)[1]
            try:
                parsed_depth = int(depth)
            except ValueError as exc:
                raise ValueError(
                    "at_depth placement requires a non-negative integer"
                ) from exc
            if parsed_depth < 0:
                raise ValueError("at_depth placement requires a non-negative integer")


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
        if not isinstance(self.operand, str) or not isinstance(self.pack_var, str):
            raise TypeError("RelayoutSpec operand and pack_var must be strings")
        if not self.operand or not self.pack_var:
            raise ValueError("RelayoutSpec operand and pack_var must be non-empty")
        if isinstance(self.strip_width, bool) or not isinstance(self.strip_width, int):
            raise TypeError("RelayoutSpec.strip_width must be an integer")
        if self.strip_width <= 0:
            raise ValueError("RelayoutSpec.strip_width must be greater than zero")
        if self.scope_var is not None:
            if not isinstance(self.scope_var, str):
                raise TypeError("RelayoutSpec.scope_var must be a string or None")
            if not self.scope_var:
                raise ValueError("RelayoutSpec.scope_var must be non-empty")


@dataclass(frozen=True)
class _RelayoutPlan:
    """CIN-derived metadata consumed by the post-lowering staging pass."""

    operand: str
    pack_var: str
    panel_var: str
    scope_var: str
    row_var: str
    access_index_vars: Tuple[str, ...]
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
    tile_var: str
    result_level: int
    result_prefix_vars: Tuple[str, ...]
    access_index_vars: Tuple[str, ...]


@dataclass(frozen=True)
class Schedule:
    """A complete, immutable scheduling decision suitable for cache keys.

    ``loop_order`` names the unsplit logical variables. ``tiles`` are applied in
    tuple order, making placement deterministic for multi-axis tiling.
    ``parallel_loop`` may name a logical loop or a generated ``*_out``/``*_in``
    loop. The full representation, rather than the human tag, is the cache key.
    """

    loop_order: Optional[Tuple[str, ...]] = None
    tiles: Tuple[TileSpec, ...] = ()
    relayout: Optional[RelayoutSpec] = None
    tag: str = ""
    parallel_loop: Optional[str] = None

    def __post_init__(self) -> None:
        if self.loop_order is not None:
            if isinstance(self.loop_order, str):
                raise TypeError("Schedule.loop_order must be a sequence of names")
            if not isinstance(self.loop_order, tuple):
                object.__setattr__(self, "loop_order", tuple(self.loop_order))
            if any(not isinstance(name, str) or not name for name in self.loop_order):
                raise ValueError("Schedule.loop_order must contain non-empty strings")
            if len(self.loop_order) != len(set(self.loop_order)):
                raise ValueError("Schedule.loop_order contains duplicate variables")
        if not isinstance(self.tiles, tuple):
            try:
                object.__setattr__(self, "tiles", tuple(self.tiles))
            except TypeError as exc:
                raise TypeError("Schedule.tiles must be a sequence") from exc
        if any(not isinstance(tile, TileSpec) for tile in self.tiles):
            raise TypeError("Schedule.tiles must contain TileSpec instances")
        if self.relayout is not None and not isinstance(self.relayout, RelayoutSpec):
            raise TypeError("Schedule.relayout must be a RelayoutSpec or None")
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
            raise ValueError("Schedule cannot tile the same index variable twice")
        if sum(tile.parallel for tile in self.tiles) > 1:
            raise ValueError("Schedule may select at most one parallel tile loop")
        if self.parallel_loop is not None and any(tile.parallel for tile in self.tiles):
            raise ValueError(
                "Use either Schedule.parallel_loop or TileSpec.parallel, not both"
            )
        if self.parallel_loop is not None:
            if not isinstance(self.parallel_loop, str):
                raise TypeError("Schedule.parallel_loop must be a string or None")
            if not self.parallel_loop:
                raise ValueError("Schedule.parallel_loop must be a non-empty string")
        if not isinstance(self.tag, str):
            raise TypeError("Schedule.tag must be a string")

    @property
    def cache_key(self) -> str:
        """Canonical cache discriminator containing every schedule field."""
        return repr(self)


_SCHEDULE_FORCE: ContextVar[Optional[Schedule]] = ContextVar(
    "scorch_schedule_force", default=None
)


def get_forced_schedule() -> Optional[Schedule]:
    """Return the schedule active in this execution context, if any."""
    return _SCHEDULE_FORCE.get()


@contextmanager
def schedule_force(value: Optional[Schedule]):
    """Temporarily force a schedule for ``matmul``/``einsum`` in this context."""
    if value is not None and not isinstance(value, Schedule):
        raise TypeError("schedule_force expects a Schedule or None")
    token = _SCHEDULE_FORCE.set(value)
    try:
        yield
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
        seen: Set[str] = set()
        unique: List[IndexVar] = []
        for index_var in index_vars:
            if index_var.name in seen:
                continue
            seen.add(index_var.name)
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
            if logical_pos < len(tensor_shape):
                sizes.append(float(tensor_shape[logical_pos]))

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
    def _find_index_var_by_name(cin: CIN, name: str) -> IndexVar:
        matches = [index_var for index_var in cin.index_vars if index_var.name == name]
        if not matches:
            raise ValueError(f"Unknown index variable {name!r}")
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
                raise ValueError("child_of placement requires an index variable")
            exact = [
                pos
                for pos, loop in enumerate(prefix)
                if loop.index_var.name == parent_name
            ]
            if not exact:
                raise ValueError(
                    f"Cannot place tile child of {parent_name!r}; it is not in "
                    "the common outer loop prefix"
                )
            depth = exact[0] + 1
        elif placement.startswith("at_depth:"):
            value = placement.split(":", 1)[1]
            try:
                depth = int(value)
            except ValueError as exc:
                raise ValueError("at_depth placement requires an integer") from exc
            if depth < 0 or depth > len(prefix):
                raise ValueError(
                    f"at_depth:{depth} is outside the common loop-prefix range "
                    f"0..{len(prefix)}"
                )
        else:
            raise ValueError(f"Unsupported tile placement {placement!r}")

        target_in_prefix = next(
            (
                pos
                for pos, loop in enumerate(prefix)
                if loop.index_var.name == target_name
            ),
            None,
        )
        if target_in_prefix is not None and depth > target_in_prefix:
            raise ValueError(
                f"Tile outer loop for {target_name!r} must dominate its inner loop"
            )
        return depth

    @staticmethod
    def _insert_loop_at_depth(cin: CIN, loop: ForAll, depth: int) -> CIN:
        if depth == 0:
            loop.stmt = cin
            cin.set_parent(loop)
            loop.inserted_workspace = cin.inserted_workspace
            loop.no_tile_list = list(cin.no_tile_list)
            return loop

        parent: CIN = cin
        for _ in range(depth - 1):
            if not isinstance(parent, ForAll):
                raise ValueError("Tile placement does not name a common loop prefix")
            parent = parent.stmt
        if not isinstance(parent, ForAll):
            raise ValueError("Tile placement does not name a common loop prefix")

        child = parent.stmt
        loop.stmt = child
        child.set_parent(loop)
        parent.stmt = loop
        loop.set_parent(parent)
        # ``loop`` starts with the root as a temporary constructor body. Restore
        # the actual root relationship after inserting it deeper.
        cin.parent = None
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

        accum_c = Workspace(name="accum_c", dim=1, dense=True)

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
            raise TypeError("tile_size must be an integer")
        if tile_size <= 0:
            raise ValueError("tile_size must be greater than zero")
        if not isinstance(cin, ForAll):
            raise ValueError("Expected input CIN to be a ForAll statement")

        target_name = index_var.name
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
        index_var = Scheduler._find_index_var_by_name(cin, target_name)
        if index_var.is_tiled or getattr(index_var, "_expr", None) is not None:
            raise ValueError(f"Index variable {target_name!r} is already tiled")

        sparse_accesses = [
            tensor_access
            for tensor_access in cin.tensor_accesses
            if tensor_access.has_index_var(index_var)
            and tensor_access.level_type_of_index_var(index_var) != LevelType.DENSE
        ]
        if sparse_accesses:
            raise NotImplementedError(
                f"Affine tiling cannot split sparse index variable {target_name!r}; "
                "windowed compressed iterators are not supported"
            )

        if placement is None:
            if _regblock_enabled():
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
                workspace_access.indices = indices
                workspace_access.update_indices(indices)

        replace_index_var_visitor = ReplaceIndexVarVisitor(
            old_index_var=index_var,
            new_index_var=inner_index_var,
        )
        replace_index_var_visitor.visit(cin)
        if replace_index_var_visitor.replacements == 0:
            raise ValueError(
                f"Cannot tile {target_name!r}: no matching ForAll binder was found"
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

        assert isinstance(cin, ForAll), "Expected input CIN to be a ForAll statement."

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

        workspace = Workspace(
            name="wksp", dim=dim_workspace, dense=are_all_dense_levels
        )

        workspace_access = WorkspaceAccess(
            wksp=workspace,
            indices=free_vars_after_last_reduction,
        )

        # Note: parent_forall not necessarily ForAll statement at the end
        parent_forall = new_cin
        while (
            isinstance(parent_forall.stmt, ForAll)
            and parent_forall.stmt.index_var != next_reduction_var
            and parent_forall.index_var != next_reduction_var
        ):
            parent_forall = parent_forall.stmt

        assert isinstance(
            parent_forall, ForAll
        ), "Expected parent_forall to be a ForAll statement."

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
            assert isinstance(producer_forall, ForAll)
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
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> List[IndexVar]:
        loop_order = Scheduler.init_loop_order(cin, costs=costs)
        loop_order = Scheduler.optimize_loop_order(cin, loop_order, costs=costs)
        loop_order = Scheduler.apply_mode_order_constraints(
            cin, loop_order, costs=costs
        )

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
                    _result_ivs = set(iv.name for iv in _non_ws[0].get_index_vars())
                    _all_coo = all(
                        lt == LevelType.COORDINATE
                        for lt in _non_ws[0].get_tensor().get_level_types()
                    )
                    # Check if an input tensor mirrors the output sparsity
                    if _all_coo:
                        _rhs_accesses = [
                            a
                            for a in cin.tensor_accesses
                            if a not in _result_accesses and not a.is_workspace()
                        ]
                        for _rhs in _rhs_accesses:
                            _rhs_ivs = set(iv.name for iv in _rhs.get_index_vars())
                            if _rhs_ivs == _result_ivs:
                                _needs_forced_reorder = False
                                break

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
            raise ValueError("Schedule.loop_order must contain non-empty names")
        if len(requested) != len(set(requested)):
            raise ValueError("Schedule.loop_order contains duplicate variables")

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
            raise ValueError(
                "Schedule.loop_order must be a complete permutation ("
                + ", ".join(details)
                + ")"
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
                    raise ValueError(
                        "Schedule.loop_order violates result storage order: "
                        f"{parent!r} must precede {child!r}"
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
            raise ValueError(
                f"Schedule.parallel_loop {loop_name!r} must identify exactly one "
                f"ForAll loop; found {len(matches)}"
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
            raise NotImplementedError(
                "Heap accumulation currently supports exactly one result tile"
            )

        tile = heap_tiles[0]
        if tile.kind != "affine":
            raise ValueError("Heap accumulation requires an affine result tile")
        if tile.placement != "outermost":
            raise ValueError(
                "Heap accumulation requires its affine tile to be outermost so "
                "the compact result spans every enclosed reduction"
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
            raise ValueError(
                "Heap accumulation requires its affine result tile to remain "
                "outermost after all scheduled tiles are applied; later root "
                f"tiles would wrap it: {outer_wrappers}"
            )
        if tile.parallel:
            raise ValueError(
                "A heap-backed result tile uses shared reusable storage; its "
                "outer tile loop must be serial"
            )
        if schedule.parallel_loop in (
            tile.index_var,
            f"{tile.index_var}_out",
            f"{tile.index_var}_in",
        ):
            raise ValueError(
                "A heap-backed result tile cannot select its shared tile loop "
                "for parallel execution"
            )

        assignment: CIN = cin
        while isinstance(assignment, ForAll):
            assignment = assignment.stmt
        if not isinstance(assignment, TensorAssign) or assignment.op not in (
            None,
            Operation.ADD,
        ):
            raise NotImplementedError(
                "Heap accumulation requires one additive result assignment"
            )
        if not reduction_names:
            raise NotImplementedError(
                "Heap accumulation requires an enclosed reduction to accumulate"
            )
        if tile.index_var in reduction_names:
            raise ValueError(
                "Heap accumulation must tile a free result axis, not a reduction"
            )

        result_accesses = [
            access
            for access in cin.get_result_tensor_accesses()
            if not isinstance(access.tensor, Workspace)
        ]
        if len(result_accesses) != 1 or not result_accesses[0].is_dense():
            raise NotImplementedError(
                "Heap accumulation requires exactly one dense result tensor"
            )
        result_access = result_accesses[0]
        result_names = tuple(
            index_var.name for index_var in result_access.get_sorted_index_vars()
        )
        if len(result_names) < 2 or result_names[-1] != tile.index_var:
            raise NotImplementedError(
                "Heap accumulation requires the tiled free axis to be the dense "
                "result's trailing storage level"
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
                raise ValueError(
                    "Heap accumulation may parallelize only a dense result-prefix "
                    f"loop; {anchor!r} does not partition compact result rows"
                )
        if any(level != LevelType.DENSE for level in result_access.level_types()):
            raise NotImplementedError(
                "Heap accumulation currently supports all-dense result storage"
            )
        tile_index_var = Scheduler._find_index_var_by_name(cin, tile.index_var)
        result_level = result_access.level_of_index_var(tile_index_var)
        if result_level != result_access.num_levels - 1:
            raise NotImplementedError(
                "Heap accumulation requires the tiled free axis at the final "
                "result storage level"
            )
        if result_access.tensor.dtype not in (torch.float32, torch.float64):
            raise NotImplementedError(
                "Heap accumulation supports float32 or float64 dense results"
            )
        if not parallel_anchors:
            raise ValueError(
                "Heap accumulation requires an explicit parallel dense "
                "result-prefix loop so the shared outer result-tile loop remains "
                "serial"
            )

        return _ResultTilePlan(
            result=result_access.tensor.name,
            tile_var=tile.index_var,
            result_level=result_level,
            result_prefix_vars=result_names[:-1],
            access_index_vars=tuple(
                index_var.name for index_var in result_access.indices
            ),
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
            raise NotImplementedError(
                "Packed relayout requires one additive contraction assignment"
            )

        rhs_accesses = cin.get_rhs_tensor_accesses()
        packed_accesses = [
            access for access in rhs_accesses if access.tensor.name == relayout.operand
        ]
        if len(packed_accesses) != 1:
            raise ValueError(
                "RelayoutSpec.operand must name exactly one input tensor access; "
                f"found {len(packed_accesses)} for {relayout.operand!r}"
            )
        packed_access = packed_accesses[0]
        packed_names = [
            index_var.name for index_var in packed_access.get_sorted_index_vars()
        ]
        if relayout.pack_var not in packed_names:
            raise ValueError(
                f"Relayout pack_var {relayout.pack_var!r} does not index operand "
                f"{relayout.operand!r}"
            )
        if not packed_access.is_dense() or packed_access.num_levels != 2:
            raise NotImplementedError(
                "Packed relayout requires a rank-2 dense input operand"
            )
        if packed_names[-1] != relayout.pack_var:
            raise NotImplementedError(
                "Packed relayout requires pack_var to be the operand's contiguous "
                "last storage level"
            )
        multiplicative_accesses = Scheduler._multiplicative_rhs_accesses(assignment.rhs)
        if multiplicative_accesses is None or sorted(
            id(access) for access in multiplicative_accesses
        ) != sorted(id(access) for access in rhs_accesses):
            raise NotImplementedError(
                "Packed relayout requires the staged dense and compressed "
                "accesses to participate in one multiplicative contraction "
                "expression"
            )
        panel_var = packed_names[0]
        scope_var = relayout.scope_var
        if scope_var not in (panel_var, relayout.pack_var):
            raise ValueError(
                "RelayoutSpec.scope_var must name either the staged access's "
                f"panel axis {panel_var!r} or tiled axis {relayout.pack_var!r}; "
                f"got {scope_var!r}"
            )
        assert scope_var is not None

        if len(panel_tiles) != 1 or panel_tiles[0].index_var != panel_var:
            raise ValueError(
                "Packed relayout requires exactly one sparse panel tile on the "
                f"other operand index {panel_var!r}"
            )
        panel_tile = panel_tiles[0]
        affine_tiles = [tile for tile in schedule.tiles if tile.kind == "affine"]
        pack_tiles = [
            tile for tile in affine_tiles if tile.index_var == relayout.pack_var
        ]
        if len(pack_tiles) != 1:
            raise ValueError(
                "Packed relayout requires exactly one affine tile for pack_var "
                f"{relayout.pack_var!r}"
            )
        pack_tile = pack_tiles[0]
        if len(schedule.tiles) != 2 or len(affine_tiles) != 1:
            raise NotImplementedError(
                "Packed relayout currently supports only one affine pack tile "
                "and one sparse panel tile"
            )
        if pack_tile.width != relayout.strip_width:
            raise ValueError(
                "Relayout strip_width must match the affine pack tile width "
                f"({relayout.strip_width} != {pack_tile.width})"
            )
        if pack_tile.accum not in ("direct", "heap"):
            raise NotImplementedError(
                "Packed relayout supports direct or heap-backed output accumulation"
            )
        expected_panel_placement = f"child_of:{relayout.pack_var}_out"
        if (
            pack_tile.placement != "outermost"
            or panel_tile.placement != expected_panel_placement
        ):
            raise ValueError(
                "Packed relayout requires an outermost pack tile followed by a "
                f"panel placed at {expected_panel_placement!r}"
            )

        sparse_rhs = [access for access in rhs_accesses if not access.is_dense()]
        compressed_rhs = [
            access
            for access in sparse_rhs
            if LevelType.COMPRESSED in access.level_types()
        ]
        if len(rhs_accesses) != 2 or len(sparse_rhs) != 1 or len(compressed_rhs) != 1:
            raise NotImplementedError(
                "Packed relayout requires exactly one CSR input and one dense "
                "input tensor access"
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
            raise NotImplementedError(
                "Packed relayout requires a rank-2 CSR input with a dense parent"
            )
        compressed_names = [
            index_var.name for index_var in compressed_access.get_sorted_index_vars()
        ]
        row_var, compressed_panel_var = compressed_names
        if compressed_panel_var != panel_var:
            raise NotImplementedError(
                "The CSR compressed coordinate must be the packed operand's "
                f"panel index {panel_var!r}"
            )

        result_accesses = cin.get_result_tensor_accesses()
        if len(result_accesses) != 1 or not result_accesses[0].is_dense():
            raise NotImplementedError(
                "Packed relayout requires exactly one dense result tensor"
            )
        result_access = result_accesses[0]
        result_names = [
            index_var.name for index_var in result_access.get_sorted_index_vars()
        ]
        if result_names != [row_var, relayout.pack_var]:
            raise NotImplementedError(
                "Packed relayout requires the dense result to be indexed by the "
                "CSR row followed by pack_var"
            )
        if panel_var not in reduction_names or relayout.pack_var in reduction_names:
            raise NotImplementedError(
                "Packed relayout requires panel_var to be a reduction and "
                "pack_var to be a free result axis"
            )

        expected_order = (row_var, panel_var, relayout.pack_var)
        if schedule.loop_order != expected_order:
            raise ValueError(
                "Packed relayout requires loop_order to match the structural "
                f"(row, panel, pack) order {expected_order!r}"
            )
        if schedule.parallel_loop != row_var:
            raise ValueError(
                "Packed relayout requires the CSR row loop to be selected for "
                "parallel execution"
            )

        tensors = [access.tensor for access in rhs_accesses] + [result_access.tensor]
        dtypes = {tensor.dtype for tensor in tensors}
        if len(dtypes) != 1 or packed_access.tensor.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise NotImplementedError(
                "Packed relayout supports matching float32 or float64 operand "
                "and result dtypes"
            )

        return _RelayoutPlan(
            operand=relayout.operand,
            pack_var=relayout.pack_var,
            panel_var=panel_var,
            scope_var=scope_var,
            row_var=row_var,
            access_index_vars=tuple(
                index_var.name for index_var in packed_access.indices
            ),
            operand_panel_level=packed_access.level_of_index_var(
                Scheduler._find_index_var_by_name(cin, panel_var)
            ),
            operand_pack_level=packed_access.level_of_index_var(
                Scheduler._find_index_var_by_name(cin, relayout.pack_var)
            ),
        )

    @staticmethod
    def apply_schedule(
        cin: CIN,
        schedule: Schedule,
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> CIN:
        """Apply an explicit tuner schedule to a CIN loop nest.

        An empty schedule delegates to :meth:`auto_schedule`, preserving the
        existing scheduler. A non-empty schedule owns loop order and tiling: no
        implicit tile heuristic is added on top of it.
        """
        if not isinstance(schedule, Schedule):
            raise TypeError("apply_schedule expects a Schedule")
        if not isinstance(cin, ForAll):
            if (
                schedule.loop_order is not None
                or schedule.tiles
                or schedule.relayout is not None
                or schedule.parallel_loop is not None
            ):
                raise NotImplementedError(
                    "Non-empty schedules require a ForAll CIN statement"
                )
            return cin
        panel_tiles = [tile for tile in schedule.tiles if tile.kind == "panel"]
        if len(panel_tiles) > 1:
            raise NotImplementedError("Only one sparse panel tile is supported")
        if panel_tiles and schedule.tiles[-1].kind != "panel":
            raise ValueError(
                "A sparse panel tile must follow all affine tiles in Schedule.tiles"
            )
        if panel_tiles and not Scheduler._has_dense_output(cin):
            raise NotImplementedError(
                "Sparse panel tiling currently requires a dense result tensor"
            )
        if panel_tiles and panel_tiles[0].parallel:
            raise ValueError(
                "Sparse panel outer loops must be serial; select the row loop "
                "with Schedule.parallel_loop"
            )
        for tile in schedule.tiles:
            if tile.kind == "panel" and tile.accum != "direct":
                raise NotImplementedError(
                    "Sparse panel tiles currently require accum='direct'"
                )

        logical_names = {index_var.name for index_var in cin.index_vars}
        unknown_tiles = [
            tile.index_var
            for tile in schedule.tiles
            if tile.index_var not in logical_names
        ]
        if unknown_tiles:
            raise ValueError(
                f"Schedule tiles refer to unknown index variables {unknown_tiles}"
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
            raise NotImplementedError(
                "Affine reduction tiling requires an accumulator spanning outer "
                f"tiles; unsupported reduction variables: {affine_reductions}"
            )
        if any(tile.kind == "affine" for tile in schedule.tiles) and not (
            Scheduler._has_dense_output(cin)
        ):
            raise NotImplementedError(
                "Explicit affine tiling currently requires a dense result tensor; "
                "tiled sparse-output assembly is unsupported"
            )

        parallel_names = [tile.index_var for tile in schedule.tiles if tile.parallel]
        if schedule.parallel_loop is not None:
            parallel_names.append(schedule.parallel_loop)
        if parallel_names and not Scheduler._has_dense_output(cin):
            raise NotImplementedError(
                "Explicit parallel-loop selection currently requires a dense "
                "result tensor"
            )
        tiled_inner_loops = {
            f"{tile.index_var}_in" for tile in schedule.tiles if tile.kind == "affine"
        }
        parallel_inner_loops = [
            name for name in parallel_names if name in tiled_inner_loops
        ]
        if parallel_inner_loops:
            raise ValueError(
                "Tiled inner loops contain a ragged-tail break and cannot be "
                f"parallelized: {parallel_inner_loops}"
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
            raise ValueError(
                "Reduction loops cannot be selected for parallel execution: "
                f"{parallel_reductions}"
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
                raise NotImplementedError(
                    f"Panel index {tile.index_var!r} must have exactly one "
                    "compressed tensor access"
                )
            compressed_access = compressed_accesses[0]
            parent = compressed_access.get_parent_index_var(target)
            if (
                parent is None
                or compressed_access.level_type_of_index_var(parent) != LevelType.DENSE
            ):
                raise NotImplementedError(
                    "Sparse panel tiling currently requires a CSR-style "
                    "compressed level with a dense parent"
                )
            if schedule.parallel_loop != parent.name:
                raise ValueError(
                    "Sparse panel tiling requires its CSR dense-parent row loop "
                    f"{parent.name!r} as Schedule.parallel_loop"
                )
            if schedule.loop_order is None:
                panel_logical_order = [
                    index_var.name
                    for index_var in Scheduler.select_loop_order(cin, costs=costs)
                ]
            else:
                panel_logical_order = list(schedule.loop_order)
            if (
                parent.name in panel_logical_order
                and target.name in panel_logical_order
                and panel_logical_order.index(parent.name)
                > panel_logical_order.index(target.name)
            ):
                raise ValueError(
                    "Sparse panel tiling requires the CSR row loop to precede "
                    "the compressed panel coordinate in logical loop_order"
                )
            if tile.placement.startswith("at_depth:"):
                raise NotImplementedError(
                    "Sparse panel tiles do not support at_depth placement"
                )
            if tile.placement.startswith("child_of:"):
                placement_parent = tile.placement.split(":", 1)[1]
                if placement_parent in (
                    parent.name,
                    f"{parent.name}_out",
                    f"{parent.name}_in",
                ):
                    raise ValueError(
                        "A sparse panel loop must be placed outside its parallel "
                        "CSR row loop"
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
                    raise ValueError(
                        "A child_of sparse panel placement must name an "
                        "outermost affine tile loop"
                    )

        relayout_plan = Scheduler._validate_relayout(
            cin,
            schedule,
            panel_tiles,
            reduction_names,
        )

        is_identity = (
            schedule.loop_order is None
            and not schedule.tiles
            and schedule.relayout is None
            and schedule.parallel_loop is None
        )
        if is_identity:
            return Scheduler.auto_schedule(cin, costs=costs)

        if schedule.loop_order is None:
            logical_order = Scheduler.select_loop_order(cin, costs=costs)
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
            raise NotImplementedError(
                "Stack accumulation is only supported for a trailing dense free "
                f"dimension after a reduction: {unsupported_stack_tiles}"
            )
        stack_targets = {tile.index_var for tile in stack_tiles}
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
            )
            generated_outer_names[tile.index_var] = f"{tile.index_var}_out"

        if schedule.parallel_loop is not None:
            parallel_name = generated_outer_names.get(
                schedule.parallel_loop, schedule.parallel_loop
            )
            Scheduler._set_explicit_parallel_loop(cin, parallel_name)

        panel_bounds: Dict[str, str] = {}
        for tile in panel_tiles:
            target = Scheduler._find_index_var_by_name(cin, tile.index_var)
            dense_accesses = [
                access
                for access in cin.tensor_accesses
                if access.has_index_var(target)
                and access.level_type_of_index_var(target) == LevelType.DENSE
            ]
            if not dense_accesses:
                raise NotImplementedError(
                    f"Panel index {tile.index_var!r} has no dense dimension bound"
                )
            access = dense_accesses[0]
            level = access.level_of_index_var(target)
            panel_bounds[tile.index_var] = f"{access.tensor.name}{level}_size"

        # The CIN lowerer consumes panel tiles after concrete sparse iterators are
        # available in LLIR. Affine tiles above remain ordinary CIN transforms.
        cin.explicit_schedule = schedule
        cin.panel_bounds = panel_bounds
        cin.relayout_plan = relayout_plan
        cin.result_tile_plan = result_tile_plan
        return cin

    @staticmethod
    def _select_index_vars_to_tile(cin: CIN) -> List[IndexVar]:
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
        if loop_order and not _regblock_enabled():

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
    def _apply_tiling_heuristics(cin: CIN) -> CIN:
        if not isinstance(cin, ForAll):
            return cin
        # Preserve the legacy auto-scheduler's scope: before ``add_tile`` grew
        # into a general strip-mining transform it only changed loop nests that
        # had a workspace/Where region.  Direct affine tiling is intentionally
        # available through ``Schedule``, but must not silently change default
        # schedules such as reduction-innermost SDDMM.
        if not cin.inserted_workspace:
            return cin
        tile_width = _regblock_tile_width() if _regblock_enabled() else 32
        for index_var in Scheduler._select_index_vars_to_tile(cin):
            cin = Scheduler.add_tile(cin, index_var, tile_width)
        return cin

    @staticmethod
    def auto_schedule(
        cin: CIN,
        costs: _CostModelConstants = _DEFAULT_COSTS,
    ) -> CIN:
        if not isinstance(cin, ForAll):
            return cin

        loop_order = Scheduler.select_loop_order(cin, costs=costs)
        cin = Scheduler._rebuild_loop_nest(cin, loop_order)

        if Scheduler.should_insert_workspace(cin, loop_order):
            # For dense outputs the workspace only exists to support tiling.
            # If nothing will be tiled, the workspace is pure overhead
            # (extra memset + copy-back per iteration).
            will_tile = len(Scheduler._select_index_vars_to_tile(cin)) > 0
            if not Scheduler._has_dense_output(cin) or will_tile:
                cin = Scheduler.insert_workspace(cin, allow_dense=True)

        if not isinstance(cin, ForAll):
            return cin

        cin = Scheduler._apply_tiling_heuristics(cin)
        return cin
