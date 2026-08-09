"""LoopIR-to-structured-LLIR lowering for the migrated families.

The existing structured LLIR remains the target-specific CxxIR boundary; no
new target IR is introduced.  This module lowers one verified LoopIR
program into a complete LLIR ``evaluate`` function by reusing the exact
production target components the legacy path uses:

- :class:`~scorch.compiler.torch_cpp_abi.TorchCppKernelABI` /
  :class:`~scorch.compiler.torch_cpp_abi.KernelTensorABI` own the public
  signature, validation, and input prologue (dense extents plus compressed
  ``pos``/``crd`` pointer bindings);
- :class:`~scorch.compiler.torch_cpp_abi.ResultTensorAssembler` owns result
  storage initialization and final assembly for dense and canonical-CSR
  outputs;
- the managed production LLIR pass pipeline
  (:class:`~scorch.compiler.llir_pass_manager.LLIRPassManager`) applies the
  same typed optimization passes (sparse prefetch, dense pointer hoisting,
  single-iteration elimination, invariant hoisting, dynamic-vector
  rewriting, ...);
- :func:`~scorch.compiler.parallel_marking_pass.mark_first_for_loop_parallel`
  applies the same outer-loop parallel policy, under the same gate the
  legacy lowering uses (dense result written by the outer loop variable).

Because the raw loop-nest emission mirrors the legacy lowering
statement-for-statement — dense position chains, sparse position loops,
two-cursor coordinate merges with UNION tail loops, ordered CSR assembly
counters, the affine-split origin/point loops (width constants, the
stepping origin loop, the reconstructed logical coordinate, and the
ragged-tail overshoot break), and the stack workspace region (the
``float wksp[kTile] = {};`` declaration-with-reset, the input-bounded
producer point loop reducing into the workspace, and the synthesized
result-bounded ``// Lower consumer CIN`` copy-out loop) — the generated
C++ for the migrated families is byte-identical to the legacy pipeline's
output; the differential suites lock that equality.  LoopIR itself
contains none of these target details — this module is where target
spelling begins.

Shapes are runtime bindings, not LoopIR content: callers pass concrete input
shapes and the result shape, and this boundary re-resolves every logical
dimension's extent across all of them, failing closed on any disagreement.

Fail-closed surface: this target lowering accepts the migrated program
shapes only.  Unsupported neighbors—permuted compressed structure, merges
of more than two cursors, unbound hierarchical descent, heterogeneous or
n-ary united assemblies, unmanaged sparse reductions, unsupported
schedule/assembly compositions, and arbitrary statement shapes—fail with
:class:`LoopIRTargetError` and a stable code rather than being
approximated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
from types import FunctionType, MappingProxyType
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    NoReturn,
    Optional,
    Set,
    Tuple,
    cast,
)
from weakref import ReferenceType, ref as weakref_ref

import torch

from ...format import LevelType
from ..identity import AccessId, IndexId, SymbolId, new_access_id
from .. import llir
from ..compile_options import CompileOptions
from ..compilation_context import CompilationContext, CompilerStageId
from ..dense_pointer_hoist_pass import DensePointerHoistContext
from ..llir_pass_manager import (
    CompressedWhereOpenMPPassSpec,
    DensePointerHoistPassSpec,
    LLIRPassPartialFailure,
    LLIRPassManager,
    LLIRRewriteArtifact,
    LLIRStatementListArtifact,
)
from ..llir_traversal import (
    LLIRRewriter,
    LLIRTraversalError,
    LLIRTraversalContext,
    LLIRWalker,
    SUPPORTED_LLIR_STATEMENT_NODE_TYPES,
)
from ..loop_plan import MAX_AFFINE_TILE_WIDTH
from ..parallel_marking_pass import (
    _CPP_KEYWORDS,
    EMPTY_PARALLEL_WORKSPACE_CLUSTER,
    apply_parallel_policy,
    extract_loop_bound,
    find_sparse_pos_array,
    has_sparse_inner_loop,
    mark_first_for_loop_parallel,
)
from ..schedule_lowerer import (
    _contains_tensor_access,
    _declared_names,
    _heap_compact_access,
    _heap_result_copy_group,
    _heap_result_init_group,
    _heap_result_storage_statements,
    _heap_result_tile_names,
    _panel_bound_expression,
    _panel_range_guard,
    _redirect_sparse_prefetch,
    _relayout_pack_loop,
    _relayout_storage_statements,
    _remove_dense_result_zero,
    _rewrite_stmt_access_sequence,
    _unique_name,
)
from ..torch_cpp_abi import (
    KernelTensorABI,
    ResultTensorAssembler,
    TorchCppKernelABI,
)
from ...utils import dtype_to_c_datatype
from .nodes import (
    AppendEntry,
    BinaryExpr,
    BinaryOp,
    Block,
    CursorId,
    CursorValue,
    DenseFor,
    DensePosition,
    DimensionDecl,
    DimensionId,
    Expr,
    FloatConst,
    IndexValue,
    LevelDecl,
    LevelKind,
    Load,
    LoopIRNodeId,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    PanelOuterFor,
    ParallelDiscipline,
    ParallelIntent,
    ParallelPart,
    ParallelSelection,
    ParallelWork,
    PositionId,
    PositionLoad,
    PositionValue,
    ReduceOp,
    RelayoutDecl,
    RelayoutId,
    RelayoutScope,
    RelayoutStage,
    ResultTileDecl,
    ResultTileId,
    ResultTileRegion,
    RootPosition,
    ScalarType,
    SparseCursorDecl,
    SparseFor,
    SparseWorkSource,
    SparseWindowFor,
    SparseWorkspaceDecl,
    SparseWorkspaceDrainFor,
    SparseWorkspaceInsert,
    SparseWorkspaceRegion,
    SparseWorkspaceValue,
    StagedRead,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
    TiledReduce,
    TileId,
    TileInnerFor,
    TileOuterFor,
    WorkspaceId,
    WorkspaceDecl,
    WorkspaceRead,
    WorkspaceReduce,
    WorkspaceRegion,
)
from .verifier import LoopIRVerificationError, verify_program

_LOOPIR_GRAPH_NODE_TYPES = (
    AppendEntry,
    BinaryExpr,
    Block,
    CursorValue,
    DenseFor,
    DensePosition,
    DimensionDecl,
    FloatConst,
    IndexValue,
    LevelDecl,
    Load,
    LoopProgram,
    MergedSparseFor,
    PanelOuterFor,
    ParallelSelection,
    ParallelWork,
    PositionLoad,
    PositionValue,
    RelayoutDecl,
    RelayoutStage,
    ResultTileDecl,
    ResultTileRegion,
    RootPosition,
    SparseCursorDecl,
    SparseFor,
    SparseWorkSource,
    SparseWindowFor,
    SparseWorkspaceDecl,
    SparseWorkspaceDrainFor,
    SparseWorkspaceInsert,
    SparseWorkspaceRegion,
    SparseWorkspaceValue,
    StagedRead,
    Store,
    StoreReduce,
    TensorDecl,
    TiledReduce,
    TileInnerFor,
    TileOuterFor,
    WorkspaceDecl,
    WorkspaceRead,
    WorkspaceReduce,
    WorkspaceRegion,
)
_LOOPIR_GRAPH_ENUM_TYPES = (
    BinaryOp,
    LevelKind,
    MergeMode,
    ParallelDiscipline,
    ParallelIntent,
    ParallelPart,
    ReduceOp,
    RelayoutScope,
    ScalarType,
)
_LOOPIR_IDENTITY_TYPES = (
    SymbolId,
    IndexId,
    LoopIRNodeId,
    DimensionId,
    CursorId,
    PositionId,
    TileId,
    WorkspaceId,
    RelayoutId,
    ResultTileId,
)
_LOOPIR_NODE_TYPE_BY_ID = {
    id(node_type): node_type for node_type in _LOOPIR_GRAPH_NODE_TYPES
}
_LOOPIR_ENUM_TYPE_BY_ID = {
    id(enum_type): enum_type for enum_type in _LOOPIR_GRAPH_ENUM_TYPES
}
_LOOPIR_IDENTITY_TYPE_BY_ID = {
    id(identity_type): identity_type for identity_type in _LOOPIR_IDENTITY_TYPES
}
_LOOPIR_NODE_FIELDS_BY_ID = {
    id(node_type): tuple(field.name for field in dataclass_fields(node_type))
    for node_type in _LOOPIR_GRAPH_NODE_TYPES
}
_MAPPING_WITNESS_FIELDS = frozenset(
    {"_bound_position_snapshot", "_position_load_signatures"}
)
_TUPLE_WITNESS_FIELDS = frozenset(
    {"_target_owner_snapshot", "_value_expression_snapshot"}
)
_IMMUTABLE_WITNESS_FIELDS = _MAPPING_WITNESS_FIELDS | _TUPLE_WITNESS_FIELDS
_OPAQUE_MAPPING_CACHE_FIELDS = frozenset(
    {"cursor_loops", "decls", "dimension_names", "loop_positions", "shapes"}
)
_OPAQUE_TUPLE_CACHE_FIELDS = frozenset(
    {
        "_input_symbols",
        "_program_input_values",
        "_program_inputs_container",
        "cursor_values",
        "loads",
        "position_loads",
    }
)

_SCALAR_TO_TORCH: Dict[ScalarType, torch.dtype] = {
    ScalarType.FLOAT32: torch.float32,
    ScalarType.FLOAT64: torch.float64,
}

_BINARY_TO_CXX: Dict[BinaryOp, str] = {
    BinaryOp.ADD: "+",
    BinaryOp.SUB: "-",
    BinaryOp.MUL: "*",
}

_LEVEL_KIND_TO_LEVEL_TYPE: Dict[LevelKind, LevelType] = {
    LevelKind.DENSE: LevelType.DENSE,
    LevelKind.COMPRESSED: LevelType.COMPRESSED,
}

_CPP_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TARGET_RESERVED_NAMES = frozenset(
    {
        "Tensor",
        "__restrict__",
        "evaluate",
        "int32_t",
        "int64_t",
        "result_shape",
        "scorch_chunk",
        "scorch_native",
        "scorch_nthreads",
        "scorch_tensor_from_vector",
        "scorch_vector_set",
        "scorch_zero_dense",
        "size_t",
        "std",
        "torch",
    }
)


def _safe_cpp_display_identifier(name: object) -> bool:
    """Whether a user-owned name stays outside C++ reserved namespaces.

    Target lowering derives more identifiers by adding underscore-prefixed
    suffixes and, for pointer views, a leading underscore.  Rejecting an
    underscore at either edge, as well as any embedded double underscore,
    prevents an otherwise innocent display name from manufacturing an
    implementation-reserved identifier.
    """

    return (
        type(name) is str
        and _CPP_IDENTIFIER.fullmatch(name) is not None
        and name not in _CPP_KEYWORDS
        and "__" not in name
        and not name.startswith("_")
        and not name.endswith("_")
    )


_DENSE = "dense"
_SPARSE = "sparse"
_MERGED = "merged"
_TILE_OUTER = "tile_outer"
_TILE_INNER = "tile_inner"
_PANEL_OUTER = "panel_outer"
_SPARSE_WINDOW = "sparse_window"


@dataclass(frozen=True)
class LoopIRTargetDefect:
    """One immutable target-lowering failure: stable code and message."""

    code: str
    message: str


class LoopIRTargetError(Exception):
    """A verified LoopIR program is outside this target lowering's surface."""

    def __init__(self, defect: LoopIRTargetDefect) -> None:
        super().__init__(f"{defect.code}: {defect.message}")
        self.defect = defect


def _fail(code: str, message: str) -> NoReturn:
    raise LoopIRTargetError(LoopIRTargetDefect(code, message))


@dataclass(frozen=True)
class _TargetSealAuthority:
    """Externally retained construction authority for one live target."""

    target_ref: ReferenceType[Any]
    target_type: Optional[type] = None
    program: Optional[LoopProgram] = None
    input_container: Optional[Tuple[SymbolId, ...]] = None
    input_values: Optional[Tuple[int, ...]] = None
    seal_token: Optional[object] = None
    graph_snapshot: Optional[Tuple[object, ...]] = None
    graph_owners: Optional[Tuple[object, ...]] = None
    target_snapshot: Optional[Tuple[object, ...]] = None
    target_owners: Optional[Tuple[object, ...]] = None


_TARGET_SEAL_AUTHORITIES: Dict[int, _TargetSealAuthority] = {}


def _begin_target_input_authority(
    target: object,
    program: LoopProgram,
    inputs: Tuple[SymbolId, ...],
    values: Tuple[int, ...],
) -> object:
    """Bind immutable input authority before construction reads the graph."""

    target_id = id(target)
    existing = _TARGET_SEAL_AUTHORITIES.get(target_id)
    if existing is not None:
        if existing.target_ref() is target:
            _fail(
                "unsupported_program_shape",
                "the target's retained input authority is already bound and "
                "cannot be rebound",
            )
        _fail(
            "unsupported_program_shape",
            "the target seal registry contains a conflicting live identity",
        )

    def release_target(reference: ReferenceType[Any]) -> None:
        current = _TARGET_SEAL_AUTHORITIES.get(target_id)
        if current is not None and current.target_ref is reference:
            _TARGET_SEAL_AUTHORITIES.pop(target_id, None)

    try:
        target_ref = weakref_ref(target, release_target)
    except TypeError:
        _fail(
            "unsupported_program_shape",
            "the target's retained program caches cannot be weakly owned",
        )
    seal_token = object()
    _TARGET_SEAL_AUTHORITIES[target_id] = _TargetSealAuthority(
        target_ref=target_ref,
        target_type=type(target),
        program=program,
        input_container=inputs,
        input_values=values,
        seal_token=seal_token,
    )
    return seal_token


def _target_input_authority(target: object) -> _TargetSealAuthority:
    """Return the construction-time input authority for one live target."""

    authority = _TARGET_SEAL_AUTHORITIES.get(id(target))
    if (
        type(authority) is not _TargetSealAuthority
        or authority.target_ref() is not target
        or authority.target_type is None
        or type(authority.program) is not LoopProgram
        or authority.input_container is None
        or authority.input_values is None
        or authority.seal_token is None
        or type(authority.input_container) is not tuple
        or type(authority.input_values) is not tuple
        or any(type(value) is not int for value in authority.input_values)
    ):
        _fail(
            "unsupported_program_shape",
            "the target's retained input authority is missing or incomplete",
        )
    return authority


def _target_seal_authority(target: object) -> _TargetSealAuthority:
    """Return the complete external authority for one exact live target."""

    authority = _target_input_authority(target)
    if (
        authority.graph_snapshot is None
        or authority.graph_owners is None
        or authority.target_snapshot is None
        or authority.target_owners is None
    ):
        _fail(
            "unsupported_program_shape",
            "the target's retained program authority is missing or incomplete",
        )
    return authority


@dataclass(frozen=True)
class _Loop:
    """One loop of the family nest, outermost first."""

    kind: str
    index: object
    dimension: DimensionId
    node: Any
    cursors: Tuple[SparseCursorDecl, ...]


class _TargetLowering:
    def __init__(
        self,
        program: LoopProgram,
        input_shapes: Mapping[SymbolId, Tuple[int, ...]],
        result_shape: Tuple[int, ...],
    ) -> None:
        self.program = program
        self._input_symbols, seal_token = self._snapshot_program_inputs(program)
        self.decls: Dict[SymbolId, TensorDecl] = {
            decl.symbol: decl for decl in program.tensors
        }
        self.result_symbol = program.outputs[0]
        self.result_decl = self.decls[self.result_symbol]
        self.result_is_dense = all(
            level.kind is LevelKind.DENSE for level in self.result_decl.levels
        )
        self.sparse_program = any(
            level.kind is not LevelKind.DENSE
            for decl in program.tensors
            for level in decl.levels
        )
        self.dimension_names: Dict[DimensionId, str] = {}
        self._access_ids: Dict[SymbolId, AccessId] = {}
        # Workspace-region state; populated by _collect_loop_nest when the
        # chain terminates at a region instead of a store leaf.
        self.region: Optional[WorkspaceRegion] = None
        self.region_start = 0
        self.producer_leaf: Optional[WorkspaceReduce] = None
        self.consumer_point: Optional[TileInnerFor] = None
        self._region_leaf: Optional[StoreReduce] = None
        # Sparse-panel state; populated by _validate_panel_shape when the
        # chain carries a coordinate-window pair.  The panel is completed
        # on the assembled function (marking, windowing, wrap) at exactly
        # the pipeline position the legacy schedule lowering uses, driven
        # by the emitted-object records below instead of name discovery.
        self.panel: Optional[PanelOuterFor] = None
        self.panel_position = -1
        self.window_position = -1
        self.panel_row_position = -1
        self._emitted_loop_headers: Dict[int, llir.ForLoop] = {}
        self._window_end_snapshot: Optional[llir.VarInit] = None
        # Staged-relayout state; populated by _collect_loop_nest and
        # _validate_relayout_shape when the chain carries a staging region.
        # The relayout is completed on the assembled function immediately
        # after the panel completion — exactly the legacy pipeline order —
        # consuming the panel completion's retained objects.
        self.relayout: Optional[RelayoutDecl] = None
        self.relayout_depth = -1
        self._staged_views: Dict[int, Load] = {}
        self._staged_access_snapshot: Optional[llir.ArrayAccess] = None
        # Heap result-tile state; populated by _collect_loop_nest and
        # _validate_result_tile_shape when the chain carries an accumulation
        # region.  The compact redirection, init/copy groups, storage, and
        # zero-fill removal are completed on the assembled function between
        # the panel and relayout completions — exactly the legacy
        # apply_schedule_to_llir order (panel, heap, relayout).
        self.result_tile: Optional[ResultTileDecl] = None
        self.result_tile_depth = -1
        self.result_tile_row_position = -1
        self._tiled_leaf: Optional[TiledReduce] = None
        self._tiled_view: Optional[StoreReduce] = None
        self._tiled_write_snapshot: Optional[llir.ArrayAccess] = None
        self._window_coord_snapshot: Optional[
            Tuple[llir.Comment, llir.VarInit, llir.BlankLine]
        ] = None
        self._panel_completion: Optional[
            Tuple[llir.ForLoop, llir.ForLoop, llir.ForLoop, llir.ForLoop]
        ] = None
        self._panel_parallel_snapshot: Optional[llir.ForLoop] = None
        # Abstract parallel-selection state; populated by
        # _validate_parallel_selection when the program carries the fact.
        # Direct and stack routes suppress the emission-time auto gate and
        # mark the selected loop on the assembled function — exactly the
        # legacy explicit-parallel order.  Panel (including relayout) and heap
        # completions own the corresponding mark; every route independently
        # checks the legacy marker state and then realizes the selection's
        # revalidated work fact.
        self.parallel = program.parallel
        self.parallel_position = -1
        self._parallel_signature: Optional[Tuple[object, ...]] = None
        self._validate_display_names()
        self._validate_layouts()
        self.shapes = self._validate_shapes(input_shapes, result_shape)
        self.loops = self._collect_loop_nest()
        self._validate_loop_variable_names()
        self.loop_positions: Dict[object, int] = {
            loop.index: position for position, loop in enumerate(self.loops)
        }
        self.leaf = self._collect_leaf()
        self._validate_panel_shape()
        self._validate_relayout_shape()
        self._validate_result_tile_shape()
        self._validate_parallel_selection()
        self.cursor_loops: Dict[CursorId, int] = {}
        for position, loop in enumerate(self.loops):
            for cursor in loop.cursors:
                self.cursor_loops[cursor.cursor] = position
        self._bound_position_snapshot = self._validated_bound_position_bindings()
        self._position_load_signatures: Dict[
            int, Tuple[int, Tuple[Tuple[int, int], ...], int]
        ] = {}
        self.loads, self.cursor_values = self._collect_accesses()
        self._value_expression_snapshot = self._validated_value_expression_signature(
            self._access_value_expression()
        )
        self._target_owner_snapshot = self._validated_target_owner_signature()
        self.level_drivers = self._compute_level_drivers()
        self._validate_access_orders()
        self._reserve_merge_names()
        self._reserve_tile_names()
        self._reserve_workspace_names()
        self._reserve_panel_names()
        self._seal_target_state(seal_token)

    # -- boundary validation -------------------------------------------------

    def _snapshot_program_inputs(
        self, program: LoopProgram
    ) -> Tuple[Tuple[SymbolId, ...], object]:
        """Own the exact declared-input sequence used throughout lowering.

        A verified ``LoopProgram`` is frozen only by convention: adversarial
        callers can still replace a field through ``object.__setattr__`` and
        a tuple subclass can execute arbitrary code from membership or
        iteration.  Snapshot exact identities once, then ensure the program
        still exposes that same tuple at every post-construction expression
        boundary.  Target code never consults a caller-controlled container.
        """

        self._target_type_snapshot = type(self)
        self._program_container = program
        if type(program) is not LoopProgram:
            _fail(
                "unsupported_program_shape",
                "the target must retain an exact LoopProgram",
            )
        state = object.__getattribute__(program, "__dict__")
        fields = _LOOPIR_NODE_FIELDS_BY_ID[id(LoopProgram)]
        state_keys = tuple(state) if type(state) is dict else ()
        if (
            type(state) is not dict
            or len(state_keys) != len(fields)
            or any(type(key) is not str for key in state_keys)
            or any(field not in state for field in fields)
        ):
            _fail(
                "unsupported_program_shape",
                "the program must retain exact stored input declarations",
            )
        inputs = state["inputs"]
        if type(inputs) is not tuple:
            _fail(
                "unsupported_program_shape",
                "the program input declarations must be an owned tuple",
            )
        values = tuple(_stored_identity_value(symbol, SymbolId) for symbol in inputs)
        if any(value is None for value in values) or len(set(values)) != len(values):
            _fail(
                "unsupported_program_shape",
                "the program inputs must be unique exact symbol identities",
            )
        self._program_inputs_container = inputs
        self._program_input_values = cast(Tuple[int, ...], values)
        seal_token = _begin_target_input_authority(
            self,
            program,
            inputs,
            self._program_input_values,
        )
        return inputs, seal_token

    def _validated_program_graph_signature(
        self,
        program: object,
        *,
        owners: Optional[List[object]] = None,
    ) -> Tuple[object, ...]:
        """Own the complete verified LoopIR artifact by identity and state.

        The target retains loop, region, declaration, and leaf objects while it
        builds legacy-shaped LLIR. Frozen dataclasses are only a conventional
        boundary in Python: ``object.__setattr__`` can replace an ancestor body
        with a fresh, verifier-valid subtree and leave every retained object
        unchanged. This stored-state walk starts at the current program root,
        validates every exact node/container edge, and records one shallow
        immutable authority record for every mutable node, identity, and enum.
        Exact tuples need no record of their own: their owning node record binds
        the tuple by identity, and their elements cannot subsequently change.
        The resulting authority supports a non-recursive pre-emission check
        while retaining the same cycle and foreign-value boundary.
        """

        active: Set[int] = set()
        seen: Set[int] = set()
        records: List[Tuple[object, ...]] = []

        def fail(message: str) -> NoReturn:
            _fail("unsupported_program_shape", message)

        def visit(value: object, depth: int) -> None:
            if depth > 256:
                fail("the target program graph exceeds the ownership depth bound")
            if value is None:
                return
            if type(value) is bool:
                return
            if type(value) is int:
                return
            if type(value) is float:
                return
            if type(value) is str:
                return
            value_type = type(value)
            object_id = id(value)
            if object_id in active:
                fail("the target program graph contains a cycle")
            if object_id in seen:
                return
            identity_type = _LOOPIR_IDENTITY_TYPE_BY_ID.get(id(value_type))
            if identity_type is value_type:
                try:
                    state = object.__getattribute__(value, "__dict__")
                except Exception:
                    fail("the target program graph contains a malformed identity")
                keys = tuple(state) if type(state) is dict else ()
                if (
                    type(state) is not dict
                    or len(keys) != 1
                    or type(keys[0]) is not str
                    or keys[0] != "value"
                    or type(state["value"]) is not int
                ):
                    fail("the target program graph contains a malformed identity")
                if owners is not None:
                    owners.append(value)
                seen.add(object_id)
                records.append(
                    (
                        "identity",
                        value,
                        value_type,
                        state["value"],
                    )
                )
                return
            enum_type = _LOOPIR_ENUM_TYPE_BY_ID.get(id(value_type))
            if enum_type is value_type:
                try:
                    name = object.__getattribute__(value, "_name_")
                except Exception:
                    fail("the target program graph contains a malformed enum value")
                if type(name) is not str:
                    fail("the target program graph contains a malformed enum value")
                if owners is not None:
                    owners.append(value)
                seen.add(object_id)
                records.append(("enum", value, enum_type, name))
                return

            if type(value) is tuple:
                if owners is not None:
                    owners.append(value)
                active.add(object_id)
                seen.add(object_id)
                try:
                    for item in value:
                        visit(item, depth + 1)
                    return
                finally:
                    active.discard(object_id)

            node_type = _LOOPIR_NODE_TYPE_BY_ID.get(id(value_type))
            if node_type is not value_type:
                fail("the target program graph contains an unsupported stored value")
            try:
                state = object.__getattribute__(value, "__dict__")
            except Exception:
                fail("the target program graph contains malformed node state")
            fields = _LOOPIR_NODE_FIELDS_BY_ID[id(value_type)]
            state_keys = tuple(state) if type(state) is dict else ()
            if (
                type(state) is not dict
                or len(state_keys) != len(fields)
                or any(type(key) is not str for key in state_keys)
                or any(field not in state for field in fields)
            ):
                fail("the target program graph contains malformed node state")

            if owners is not None:
                owners.append(value)
            active.add(object_id)
            seen.add(object_id)
            try:
                records.append(
                    (
                        "state",
                        value,
                        node_type,
                        fields,
                        tuple(state[key] for key in fields),
                    )
                )
                for key in fields:
                    visit(state[key], depth + 1)
                return
            finally:
                active.discard(object_id)

        if type(program) is not LoopProgram:
            fail("the target program graph must remain an exact LoopProgram")
        visit(program, 0)
        if (
            not records
            or records[0][0] != "state"
            or records[0][1] is not program
            or records[0][2] is not LoopProgram
        ):
            fail("the target program graph must remain an exact LoopProgram")
        return tuple(records)

    def _require_program_graph_snapshot_unchanged(
        self,
        program: object,
        snapshot: object,
    ) -> None:
        """Check constructor-owned graph records without rebuilding the graph."""

        def fail(message: str) -> NoReturn:
            _fail("unsupported_program_shape", message)

        def changed() -> NoReturn:
            # Preserve the recursive authority's specific cycle, depth,
            # foreign-value, and malformed-state diagnostics on the cold
            # path. A structurally valid but different graph receives the
            # established generic retained-graph diagnostic.
            _TargetLowering._validated_program_graph_signature(self, program)
            fail(
                "the program graph, including a target owning statement, "
                "changed after target construction"
            )

        if type(program) is not LoopProgram or type(snapshot) is not tuple:
            fail("the target program graph must remain an exact LoopProgram")
        if not snapshot:
            fail("the target program graph must remain an exact LoopProgram")
        first = snapshot[0]
        if (
            type(first) is not tuple
            or len(first) != 5
            or type(first[0]) is not str
            or first[0] != "state"
            or first[1] is not program
            or first[2] is not LoopProgram
        ):
            fail("the target program graph must remain an exact LoopProgram")

        for record in snapshot:
            if type(record) is not tuple or not record or type(record[0]) is not str:
                fail("the target program graph contains malformed authority state")
            kind = record[0]
            if kind == "identity":
                if len(record) != 4:
                    fail("the target program graph contains malformed authority state")
                value, expected_type, expected_value = record[1:]
                identity_type = _LOOPIR_IDENTITY_TYPE_BY_ID.get(id(expected_type))
                if (
                    identity_type is not expected_type
                    or type(expected_value) is not int
                ):
                    fail("the target program graph contains malformed authority state")
                if type(value) is not expected_type:
                    changed()
                try:
                    state = object.__getattribute__(value, "__dict__")
                except Exception:
                    changed()
                keys = tuple(state) if type(state) is dict else ()
                if (
                    type(state) is not dict
                    or any(type(key) is not str for key in keys)
                    or keys != ("value",)
                    or type(state["value"]) is not int
                    or state["value"] != expected_value
                ):
                    changed()
                continue
            if kind == "enum":
                if len(record) != 4:
                    fail("the target program graph contains malformed authority state")
                value, expected_type, expected_name = record[1:]
                enum_type = _LOOPIR_ENUM_TYPE_BY_ID.get(id(expected_type))
                if enum_type is not expected_type or type(expected_name) is not str:
                    fail("the target program graph contains malformed authority state")
                if type(value) is not expected_type:
                    changed()
                try:
                    current_name = object.__getattribute__(value, "_name_")
                except Exception:
                    changed()
                if type(current_name) is not str or current_name != expected_name:
                    changed()
                continue
            if kind != "state" or len(record) != 5:
                fail("the target program graph contains malformed authority state")
            value, expected_type, fields, expected_values = record[1:]
            node_type = _LOOPIR_NODE_TYPE_BY_ID.get(id(expected_type))
            if (
                node_type is not expected_type
                or type(fields) is not tuple
                or any(type(field) is not str for field in fields)
                or fields != _LOOPIR_NODE_FIELDS_BY_ID[id(expected_type)]
                or type(expected_values) is not tuple
                or len(expected_values) != len(fields)
            ):
                fail("the target program graph contains malformed authority state")
            if type(value) is not expected_type:
                changed()
            try:
                state = object.__getattribute__(value, "__dict__")
            except Exception:
                changed()
            state_keys = tuple(state) if type(state) is dict else ()
            if (
                type(state) is not dict
                or len(state_keys) != len(fields)
                or any(type(key) is not str for key in state_keys)
                or any(field not in state for field in fields)
            ):
                changed()
            for field, expected in zip(fields, expected_values):
                current = state[field]
                expected_type_of_value = type(expected)
                if expected is None:
                    matches = current is None
                elif expected_type_of_value is bool:
                    matches = type(current) is bool and current == expected
                elif expected_type_of_value is int:
                    matches = type(current) is int and current == expected
                elif expected_type_of_value is float:
                    matches = (
                        type(current) is float
                        and cast(float, current).hex() == cast(float, expected).hex()
                    )
                elif expected_type_of_value is str:
                    matches = type(current) is str and current == expected
                else:
                    matches = current is expected
                if not matches:
                    changed()

    def _validated_target_state_signature(  # noqa: C901
        self,
        *,
        owners: Optional[List[object]] = None,
    ) -> Tuple[object, ...]:
        """Bind every pre-emission cache that interprets the program graph.

        The program graph can remain pristine while a caller rewrites a
        retained target cache such as ``loops`` or ``result_decl``. Because
        lowering consumes those caches, bind the complete constructor-final
        target state before emission. Program nodes are state-checked by the
        graph signature; this signature binds the target's exact references,
        containers, derived maps, and synthetic views.
        """

        def fail() -> NoReturn:
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches contain malformed state",
            )

        state = object.__getattribute__(self, "__dict__")
        if type(state) is not dict or any(type(key) is not str for key in state):
            fail()
        graph_owners = state.get("_program_graph_owners")
        graph_owned_ids = state.get("_program_graph_owner_ids")
        if type(graph_owners) is not tuple or type(graph_owned_ids) is not frozenset:
            fail()
        if any(type(value) is not int for value in graph_owned_ids):
            fail()
        active: Set[int] = set()
        seen: Set[int] = set()
        signature: List[object] = []

        def retain(value: object) -> None:
            if owners is not None:
                owners.append(value)

        def visit(value: object, depth: int) -> None:
            if depth > 512:
                fail()
            value_type = type(value)
            if value is None:
                signature.append("none")
                return
            if value_type is bool:
                signature.extend(("bool", value))
                return
            if value_type is int:
                signature.extend(("int", value))
                return
            if value_type is float:
                signature.extend(("float", cast(float, value).hex()))
                return
            if value_type is str:
                signature.extend(("str", value))
                return
            object_id = id(value)
            if object_id in active:
                fail()
            if object_id in seen:
                signature.extend(("reference", object_id))
                return
            if value_type is torch.dtype:
                retain(value)
                seen.add(object_id)
                signature.extend(("torch_dtype", id(value), str(value)))
                return
            identity_type = _LOOPIR_IDENTITY_TYPE_BY_ID.get(id(value_type))
            if identity_type is value_type:
                primitive = _stored_identity_value(value, value_type)
                if primitive is None:
                    fail()
                retain(value)
                seen.add(object_id)
                signature.extend(
                    ("identity", value_type.__name__, id(value), primitive)
                )
                return
            enum_type = _LOOPIR_ENUM_TYPE_BY_ID.get(id(value_type))
            if enum_type is value_type:
                retain(value)
                name = object.__getattribute__(value, "_name_")
                if type(name) is not str:
                    fail()
                seen.add(object_id)
                signature.extend(("enum", value_type.__name__, id(value), name))
                return
            node_type = _LOOPIR_NODE_TYPE_BY_ID.get(id(value_type))
            if node_type is value_type and id(value) in graph_owned_ids:
                retain(value)
                seen.add(object_id)
                signature.extend(("program_node", value_type.__name__, id(value)))
                return
            if node_type is value_type:
                node_state = object.__getattribute__(value, "__dict__")
                fields = _LOOPIR_NODE_FIELDS_BY_ID[id(value_type)]
                node_state_keys = tuple(node_state) if type(node_state) is dict else ()
                if (
                    type(node_state) is not dict
                    or len(node_state_keys) != len(fields)
                    or any(type(key) is not str for key in node_state_keys)
                    or any(field not in node_state for field in fields)
                ):
                    fail()
                retain(value)
                active.add(object_id)
                seen.add(object_id)
                try:
                    signature.extend(
                        (
                            "synthetic_node",
                            value_type.__name__,
                            object_id,
                            len(fields),
                        )
                    )
                    for key in fields:
                        signature.append(key)
                        visit(node_state[key], depth + 1)
                    return
                finally:
                    active.discard(object_id)
            if value_type is _Loop:
                state = object.__getattribute__(value, "__dict__")
                if type(state) is not dict or tuple(state) != (
                    "kind",
                    "index",
                    "dimension",
                    "node",
                    "cursors",
                ):
                    fail()
                retain(value)
                active.add(object_id)
                seen.add(object_id)
                try:
                    signature.extend(("loop_cache", object_id, len(state)))
                    for field in state:
                        signature.append(field)
                        visit(state[field], depth + 1)
                    return
                finally:
                    active.discard(object_id)
            if value_type is tuple or value_type is list:
                sequence = cast(Any, value)
                retain(value)
                active.add(object_id)
                seen.add(object_id)
                try:
                    signature.extend(
                        (
                            "tuple" if value_type is tuple else "list",
                            object_id,
                            len(sequence),
                        )
                    )
                    for item in sequence:
                        visit(item, depth + 1)
                    return
                finally:
                    active.discard(object_id)
            if value_type is dict:
                mapping = cast(Dict[object, object], value)
                retain(value)
                active.add(object_id)
                seen.add(object_id)
                try:
                    signature.extend(("dict", object_id, len(mapping)))
                    for mapping_key, item in mapping.items():
                        visit(mapping_key, depth + 1)
                        visit(item, depth + 1)
                    return
                finally:
                    active.discard(object_id)
            fail()

        target_type = type(self)
        retain(target_type)
        signature.extend(("target_type", id(target_type)))
        excluded = {
            "_target_state_owners",
            "_target_state_snapshot",
            "_target_type_snapshot",
        }
        for key in sorted(state):
            if key in excluded:
                continue
            signature.extend(("field", key))
            value = state[key]
            if key in _MAPPING_WITNESS_FIELDS:
                if type(value) is not MappingProxyType:
                    fail()
                retain(value)
                signature.extend(("opaque_mapping_witness", id(value)))
            elif key in _TUPLE_WITNESS_FIELDS:
                if type(value) is not tuple:
                    fail()
                retain(value)
                signature.extend(("opaque_tuple_witness", id(value)))
            elif key in _OPAQUE_MAPPING_CACHE_FIELDS:
                if type(value) is not MappingProxyType:
                    fail()
                retain(value)
                signature.extend(("opaque_mapping_cache", id(value)))
            elif key in _OPAQUE_TUPLE_CACHE_FIELDS:
                if type(value) is not tuple:
                    fail()
                retain(value)
                signature.extend(("opaque_tuple_cache", id(value)))
            elif key in {"_program_graph_owners", "_program_graph_snapshot"}:
                if type(value) is not tuple:
                    fail()
                retain(value)
                signature.extend(("opaque_tuple", id(value), len(value)))
            elif key == "_program_graph_owner_ids":
                if type(value) is not frozenset:
                    fail()
                retain(value)
                signature.extend(("opaque_frozenset", id(value), len(value)))
            else:
                visit(value, 0)
        return tuple(signature)

    def _seal_target_state(self, seal_token: object = None) -> None:
        """Own constructor-final target caches until raw emission begins."""

        target_id = id(self)
        existing = _target_input_authority(self)
        if (
            existing.graph_snapshot is not None
            or existing.graph_owners is not None
            or existing.target_snapshot is not None
            or existing.target_owners is not None
        ):
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches are already sealed and "
                "cannot be rebound",
            )
        if seal_token is not existing.seal_token:
            _fail(
                "unsupported_program_shape",
                "the target lacks its construction-time sealing authority",
            )
        if (
            existing.target_type is not type(self)
            or existing.program is not self._program_container
            or existing.input_container is not self._program_inputs_container
            or existing.input_values is not self._program_input_values
        ):
            _fail(
                "unsupported_program_shape",
                "the target's retained input authority changed during target "
                "construction",
            )
        _TargetLowering._require_program_inputs_unchanged(self)
        target_ref = existing.target_ref

        state = object.__getattribute__(self, "__dict__")
        if type(state) is not dict or any(type(key) is not str for key in state):
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches contain malformed state",
            )
        authority_fields = {
            "_program_graph_snapshot",
            "_program_graph_owners",
            "_program_graph_owner_ids",
            "_target_state_snapshot",
            "_target_state_owners",
        }
        if any(field in state for field in authority_fields):
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches are already sealed and "
                "cannot be rebound",
            )
        if type(state.get("_reserve_generated_name")) is not FunctionType:
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches contain malformed state",
            )
        object.__delattr__(self, "_reserve_generated_name")
        self._target_type_snapshot = type(self)
        # The narrow integrity witnesses duplicate facts in the complete graph
        # signature.  Freeze their mutable mappings once, then bind every
        # witness opaquely by exact identity in the target-state signature.
        # Successful emission can therefore consult ``get`` / mapping equality
        # without exposing a caller-controlled object, while the target guard
        # avoids recursively walking the same value/position trees again.
        present_witness_fields = _IMMUTABLE_WITNESS_FIELDS.intersection(state)
        if (
            present_witness_fields
            and present_witness_fields != _IMMUTABLE_WITNESS_FIELDS
        ):
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches contain malformed state",
            )
        for field in _MAPPING_WITNESS_FIELDS:
            if field not in present_witness_fields:
                continue
            witness = state.get(field)
            if type(witness) is dict:
                object.__setattr__(self, field, MappingProxyType(dict(witness)))
            elif type(witness) is not MappingProxyType:
                _fail(
                    "unsupported_program_shape",
                    "the target's retained program caches contain malformed state",
                )
        if any(
            type(state.get(field)) is not tuple
            for field in _TUPLE_WITNESS_FIELDS
            if field in present_witness_fields
        ):
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches contain malformed state",
            )
        for field in _OPAQUE_MAPPING_CACHE_FIELDS:
            if field not in state:
                continue
            cache = state[field]
            if type(cache) is not dict:
                _fail(
                    "unsupported_program_shape",
                    "the target's retained program caches contain malformed state",
                )
            object.__setattr__(self, field, MappingProxyType(dict(cache)))
        for field in _OPAQUE_TUPLE_CACHE_FIELDS:
            if field not in state:
                continue
            cache = state[field]
            if type(cache) is list:
                object.__setattr__(self, field, tuple(cache))
            elif type(cache) is not tuple:
                _fail(
                    "unsupported_program_shape",
                    "the target's retained program caches contain malformed state",
                )
        graph_owners: List[object] = []
        self._program_graph_snapshot = self._validated_program_graph_signature(
            self._program_container,
            owners=graph_owners,
        )
        # ``id`` values are meaningful only while their objects remain alive.
        # Keep the full graph strongly owned so a replacement cannot recycle
        # an address into either the graph or target-cache signature.
        self._program_graph_owners = tuple(graph_owners)
        self._program_graph_owner_ids = frozenset(id(value) for value in graph_owners)
        owners: List[object] = []
        self._target_state_snapshot = self._validated_target_state_signature(
            owners=owners
        )
        self._target_state_owners = tuple(owners)
        current_authority = _TARGET_SEAL_AUTHORITIES.get(target_id)
        if current_authority is not existing:
            _fail(
                "unsupported_program_shape",
                "the target seal registry changed during target construction",
            )
        _TARGET_SEAL_AUTHORITIES[target_id] = _TargetSealAuthority(
            target_ref=target_ref,
            target_type=type(self),
            program=self._program_container,
            input_container=self._program_inputs_container,
            input_values=self._program_input_values,
            seal_token=existing.seal_token,
            graph_snapshot=self._program_graph_snapshot,
            graph_owners=self._program_graph_owners,
            target_snapshot=self._target_state_snapshot,
            target_owners=self._target_state_owners,
        )

    def _require_exact_target_type(self) -> Dict[str, object]:
        """Return exact stored target state without invoking subclass hooks."""

        state = object.__getattribute__(self, "__dict__")
        authority = _target_seal_authority(self)
        if type(state) is not dict or any(type(key) is not str for key in state):
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches changed after target "
                "construction",
            )
        if (
            type(self) is not authority.target_type
            or state.get("_target_type_snapshot") is not authority.target_type
        ):
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches changed after target "
                "construction",
            )
        return cast(Dict[str, object], state)

    def _require_program_inputs_unchanged(self) -> None:
        """Fail closed if the retained input sequence changed after binding."""

        target_state = object.__getattribute__(self, "__dict__")
        authority = _target_input_authority(self)
        if type(target_state) is not dict or any(
            type(key) is not str for key in target_state
        ):
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches changed after target "
                "construction",
            )
        if (
            type(self) is not authority.target_type
            or target_state.get("_target_type_snapshot") is not authority.target_type
        ):
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches changed after target "
                "construction",
            )
        bound_program = target_state.get("_program_container")
        if (
            type(target_state) is not dict
            or type(bound_program) is not LoopProgram
            or bound_program is not authority.program
            or target_state.get("program") is not authority.program
        ):
            _fail(
                "unsupported_program_shape",
                "the target program reference changed after target construction",
            )
        state = object.__getattribute__(bound_program, "__dict__")
        if type(state) is not dict or any(type(key) is not str for key in state):
            _fail(
                "unsupported_program_shape",
                "the program input declarations changed after target construction",
            )
        current = state.get("inputs")
        if (
            type(current) is not tuple
            or current is not authority.input_container
            or target_state.get("_program_inputs_container")
            is not authority.input_container
        ):
            _fail(
                "unsupported_program_shape",
                "the program input declarations changed after target construction",
            )
        values = tuple(_stored_identity_value(symbol, SymbolId) for symbol in current)
        if (
            values != authority.input_values
            or target_state.get("_program_input_values") is not authority.input_values
        ):
            _fail(
                "unsupported_program_shape",
                "a program input identity changed after target construction",
            )

    def _require_target_state_unchanged(self) -> None:
        """Fail closed before any target-private cache is interpreted."""

        _TargetLowering._require_exact_target_type(self)
        authority = _target_seal_authority(self)
        current_target_state = _TargetLowering._validated_target_state_signature(self)
        if current_target_state != authority.target_snapshot:
            _fail(
                "unsupported_program_shape",
                "the target's retained program caches changed after target "
                "construction",
            )

    def _require_program_graph_unchanged(
        self, *, target_state_checked: bool = False
    ) -> None:
        """Fail closed if any retained program state changed after binding."""

        _TargetLowering._require_program_inputs_unchanged(self)
        target_state = _TargetLowering._require_exact_target_type(self)
        authority = _target_seal_authority(self)
        bound_program = authority.program
        if (
            type(bound_program) is not LoopProgram
            or target_state.get("_program_container") is not bound_program
        ):
            _fail(
                "unsupported_program_shape",
                "the target program reference changed after target construction",
            )
        _TargetLowering._require_program_graph_snapshot_unchanged(
            self,
            bound_program,
            authority.graph_snapshot,
        )
        if not target_state_checked:
            _TargetLowering._require_target_state_unchanged(self)

    def _validate_display_names(self) -> None:
        display_names: Dict[str, str] = {}
        for dimension_decl in self.program.dimensions:
            if not _safe_cpp_display_identifier(dimension_decl.name):
                _fail(
                    "invalid_display_name",
                    f"dimension name {dimension_decl.name!r} is not a safe "
                    "ASCII C++ identifier",
                )
            if dimension_decl.name in display_names:
                _fail(
                    "duplicate_display_name",
                    f"display name {dimension_decl.name!r} is used more " "than once",
                )
            display_names[dimension_decl.name] = "dimension"
            self.dimension_names[dimension_decl.dimension] = dimension_decl.name
        for decl in self.program.tensors:
            if not _safe_cpp_display_identifier(decl.name):
                _fail(
                    "invalid_display_name",
                    f"tensor name {decl.name!r} is not a safe ASCII C++ identifier",
                )
            if decl.name in display_names:
                _fail(
                    "duplicate_display_name",
                    f"display name {decl.name!r} is used more " "than once",
                )
            display_names[decl.name] = "tensor"

        generated: Dict[str, str] = {
            name: "the target runtime" for name in _TARGET_RESERVED_NAMES
        }

        def reserve(name: str, owner: str) -> None:
            known = generated.get(name)
            if known is not None:
                _fail(
                    "generated_name_collision",
                    f"generated C++ identifier {name!r} for {owner} conflicts "
                    f"with {known}",
                )
            generated[name] = owner

        self._reserve_generated_name = reserve

        for dimension_decl in self.program.dimensions:
            reserve(
                dimension_decl.name,
                f"dimension {dimension_decl.name!r}",
            )
        for symbol in self._input_symbols:
            decl = self.decls[symbol]
            owner = f"input tensor {decl.name!r}"
            for name in (
                decl.name,
                f"{decl.name}_shape",
                f"{decl.name}_mode_indices",
                f"{decl.name}_values",
                f"{decl.name}_val",
                f"_{decl.name}_val_ptr",
            ):
                reserve(name, owner)
            for level, level_decl in enumerate(decl.levels):
                reserve(f"{decl.name}{level}_size", owner)
                reserve(f"p{decl.name}{level}", owner)
                if level_decl.kind is LevelKind.COMPRESSED:
                    reserve(f"{decl.name}{level}_pos", owner)
                    reserve(f"{decl.name}{level}_crd", owner)
                    reserve(f"p{decl.name}{level}_end", owner)

        output = self.result_decl
        output_owner = f"output tensor {output.name!r}"
        for name in (
            output.name,
            f"{output.name}_capacity",
            f"{output.name}_values",
            f"{output.name}_values_torch",
        ):
            reserve(name, output_owner)
        for level, level_decl in enumerate(output.levels):
            reserve(f"{output.name}{level}_size", output_owner)
            reserve(f"p{output.name}{level}", output_owner)
            if level_decl.kind is LevelKind.COMPRESSED:
                reserve(f"{output.name}{level}_pos", output_owner)
                reserve(f"{output.name}{level}_crd", output_owner)
                reserve(f"{output.name}{level}_pos_index", output_owner)
                reserve(f"{output.name}{level}_pos_torch", output_owner)
                reserve(f"{output.name}{level}_crd_torch", output_owner)

    def _reserve_merge_names(self) -> None:
        """Reserve the per-cursor coordinate temporaries merges generate."""

        for loop in self.loops:
            if loop.kind is not _MERGED:
                continue
            dimension_name = self.dimension_names[loop.dimension]
            for cursor in loop.cursors:
                tensor_name = self.decls[cursor.tensor].name
                self._reserve_generated_name(
                    f"{dimension_name}_{tensor_name}",
                    f"merged coordinate of input tensor {tensor_name!r}",
                )

    def _reserve_tile_names(self) -> None:
        """Reserve the derived loop and width names affine splits generate."""

        for loop in self.loops:
            if loop.kind is not _TILE_OUTER:
                continue
            name = self.dimension_names[loop.dimension]
            owner = f"affine split of dimension {name!r}"
            for generated in (f"{name}_out", f"{name}_in", f"kTile_{name}"):
                self._reserve_generated_name(generated, owner)

    def _reserve_workspace_names(self) -> None:
        """Reserve the workspace's emitted C++ identifier."""

        if self.region is None:
            return
        name = self.region.workspace.name
        if not _safe_cpp_display_identifier(name):
            _fail(
                "invalid_display_name",
                f"workspace name {name!r} is not a safe ASCII C++ identifier",
            )
        self._reserve_generated_name(name, f"workspace {name!r}")

    def _validate_layouts(self) -> None:
        if len(self.program.outputs) != 1:
            _fail(
                "unsupported_program_shape",
                "this target lowering supports exactly one output tensor",
            )
        for decl in self.program.tensors:
            # verify_program already rejected COORDINATE/SINGLETON level
            # kinds, non-CSR sparse outputs, and non-permutation level
            # modes, so only the admitted storage-permutation scope needs a
            # target-boundary check: all-dense tensors may permute, while
            # compressed structure may not.
            modes = tuple(level.mode for level in decl.levels)
            if modes == tuple(range(len(decl.levels))):
                continue
            if any(level.kind is not LevelKind.DENSE for level in decl.levels):
                _fail(
                    "unsupported_mode_order",
                    f"tensor {decl.name!r} permutes compressed structure, "
                    "which the migrated families do not cover",
                )
        for symbol in self._input_symbols:
            decl = self.decls[symbol]
            compressed = [
                level
                for level, level_decl in enumerate(decl.levels)
                if level_decl.kind is LevelKind.COMPRESSED
            ]
            if len(compressed) > 1:
                # One compressed level descends by a single bound cursor:
                # at the leaf it owns the value read, above the leaf the
                # dense sub-tree loads through its physical position.
                _fail(
                    "unsupported_program_shape",
                    f"input {decl.name!r} declares hierarchical compressed "
                    "structure; hierarchical compressed descent is outside "
                    "the migrated families",
                )

    def _validate_shapes(
        self,
        input_shapes: Mapping[SymbolId, Tuple[int, ...]],
        result_shape: Tuple[int, ...],
    ) -> Dict[SymbolId, Tuple[int, ...]]:
        try:
            input_keys = set(input_shapes)
        except Exception as error:
            raise LoopIRTargetError(
                LoopIRTargetDefect(
                    "invalid_shape_binding",
                    "input shapes could not be snapshotted",
                )
            ) from error
        if input_keys != set(self._input_symbols):
            _fail(
                "invalid_shape_binding",
                "input shapes must cover exactly the declared inputs",
            )
        shapes: Dict[SymbolId, Tuple[int, ...]] = {}
        for symbol in self._input_symbols:
            decl = self.decls[symbol]
            try:
                shape = input_shapes[symbol]
            except Exception as error:
                raise LoopIRTargetError(
                    LoopIRTargetDefect(
                        "invalid_shape_binding",
                        "input shapes could not be snapshotted",
                    )
                ) from error
            if (
                type(shape) is not tuple
                or len(shape) != len(decl.levels)
                or any(type(extent) is not int or extent < 0 for extent in shape)
            ):
                _fail(
                    "invalid_shape_binding",
                    f"input {decl.name!r} needs a rank-{len(decl.levels)} "
                    "shape of nonnegative ints",
                )
            shapes[symbol] = shape
        if (
            type(result_shape) is not tuple
            or len(result_shape) != len(self.result_decl.levels)
            or any(type(extent) is not int or extent < 0 for extent in result_shape)
        ):
            _fail(
                "invalid_shape_binding",
                f"result {self.result_decl.name!r} needs a rank-"
                f"{len(self.result_decl.levels)} shape of nonnegative ints",
            )
        shapes[self.result_symbol] = result_shape

        extents: Dict[DimensionId, Tuple[int, str, int]] = {}
        for decl in self.program.tensors:
            shape = shapes[decl.symbol]
            # Runtime shapes are physical level extents, so each level binds
            # the dimension of the logical mode it stores.  Identity layouts
            # keep the exact historical binding and messages.
            for level, level_decl in enumerate(decl.levels):
                dimension = decl.dimensions[level_decl.mode]
                known = extents.get(dimension)
                if known is None:
                    extents[dimension] = (shape[level], decl.name, level)
                elif known[0] != shape[level]:
                    _fail(
                        "dimension_extent_mismatch",
                        f"dimension "
                        f"{self.dimension_names[dimension]!r}: "
                        f"{known[1]}[{known[2]}] is {known[0]} but "
                        f"{decl.name}[{level}] is {shape[level]}",
                    )
        return shapes

    def _level_dimension(self, symbol: SymbolId, level: int) -> DimensionId:
        decl = self.decls[symbol]
        return decl.dimensions[decl.levels[level].mode]

    def _collect_loop_nest(self) -> List[_Loop]:
        loops: List[_Loop] = []
        body: Stmt = self.program.body
        while True:
            if type(body) is not Block or len(body.statements) != 1:
                _fail(
                    "unsupported_program_shape",
                    "this target lowering expects a single-statement loop "
                    "nest over one store leaf",
                )
            only = body.statements[0]
            if type(only) is DenseFor:
                loops.append(_Loop(_DENSE, only.index, only.dimension, only, ()))
                body = only.body
                continue
            if type(only) is TileOuterFor:
                # The origin loop binds no readable coordinate, so its
                # position key is a sentinel that can never collide with a
                # logical IndexId; the logical index stays reachable through
                # the node for bound resolution and the parallel gate.
                loops.append(
                    _Loop(
                        _TILE_OUTER,
                        ("tile_outer", only.tile),
                        only.dimension,
                        only,
                        (),
                    )
                )
                body = only.body
                continue
            if type(only) is TileInnerFor:
                loops.append(_Loop(_TILE_INNER, only.index, only.dimension, only, ()))
                body = only.body
                continue
            if type(only) is PanelOuterFor:
                # Like the affine origin, the panel origin binds no readable
                # coordinate; its position key is a sentinel and the node
                # keeps the logical index and bound reachable.
                loops.append(
                    _Loop(
                        _PANEL_OUTER,
                        ("panel_outer", only.tile),
                        only.dimension,
                        only,
                        (),
                    )
                )
                body = only.body
                continue
            if type(only) is SparseWindowFor:
                cursor = only.cursor
                loops.append(
                    _Loop(
                        _SPARSE_WINDOW,
                        only.coord_index,
                        self._level_dimension(cursor.tensor, cursor.level),
                        only,
                        (cursor,),
                    )
                )
                body = only.body
                continue
            if type(only) is SparseFor:
                cursor = only.cursor
                loops.append(
                    _Loop(
                        _SPARSE,
                        only.coord_index,
                        self._level_dimension(cursor.tensor, cursor.level),
                        only,
                        (cursor,),
                    )
                )
                body = only.body
                continue
            if type(only) is MergedSparseFor:
                first = only.cursors[0]
                loops.append(
                    _Loop(
                        _MERGED,
                        only.coord_index,
                        self._level_dimension(first.tensor, first.level),
                        only,
                        tuple(only.cursors),
                    )
                )
                body = only.body
                continue
            if type(only) is RelayoutStage:
                if self.relayout is not None:
                    _fail(
                        "unsupported_program_shape",
                        "this target lowering supports exactly one staged "
                        "relayout region",
                    )
                self.relayout = only.decl
                self.relayout_depth = len(loops)
                body = only.body
                continue
            if type(only) is ResultTileRegion:
                if self.result_tile is not None:
                    _fail(
                        "unsupported_program_shape",
                        "this target lowering supports exactly one result-"
                        "tile region",
                    )
                self.result_tile = only.decl
                self.result_tile_depth = len(loops)
                body = only.body
                continue
            if type(only) is TiledReduce:
                if not loops:
                    _fail(
                        "unsupported_program_shape",
                        "this target lowering requires at least one loop",
                    )
                if self.result_tile is None or only.result_tile != (
                    self.result_tile.result_tile
                ):
                    _fail(
                        "unsupported_program_shape",
                        "a TiledReduce outside the validated result-tile "
                        "region is not lowerable",
                    )
                # The compute leaf is recorded as a synthetic direct
                # StoreReduce view so raw emission and every managed pass
                # see byte-for-byte the tree the legacy pipeline
                # transforms; the compact redirection happens in
                # complete_result_tile on the assembled function.
                self._tiled_leaf = only
                self._tiled_view = StoreReduce(
                    LoopIRNodeId(-1),
                    self.result_tile.result,
                    only.indices,
                    only.op,
                    only.value,
                )
                self._validate_loop_kinds(loops, self._tiled_view)
                return loops
            if type(only) is WorkspaceRegion:
                if not loops:
                    _fail(
                        "unsupported_program_shape",
                        "a workspace region requires at least one outer loop",
                    )
                self.region = only
                self.region_start = len(loops)
                loops.extend(self._collect_region_chains(only, loops))
                assert self._region_leaf is not None
                self._validate_loop_kinds(loops, self._region_leaf)
                return loops
            if type(only) in (Store, StoreReduce, AppendEntry):
                if not loops:
                    _fail(
                        "unsupported_program_shape",
                        "this target lowering requires at least one loop",
                    )
                self._validate_loop_kinds(loops, only)
                return loops
            _fail(
                "unsupported_program_shape",
                f"unsupported nest statement {type(only).__name__}",
            )

    def _validate_loop_kinds(self, loops: List[_Loop], leaf: Stmt) -> None:
        if loops[0].kind not in (_DENSE, _TILE_OUTER, _PANEL_OUTER, _SPARSE):
            # A single-cursor sparse outermost loop is the mixed dense-leaf
            # operand chain's root-parent row loop; merged outermost
            # iteration stays outside the migrated families.
            _fail(
                "unsupported_program_shape",
                "the migrated families require a dense, sparse, tile-origin, "
                "or panel-origin outermost loop",
            )
        if type(leaf) is StoreReduce and any(
            level.kind is not LevelKind.DENSE for level in self.result_decl.levels
        ):
            # The semantic coordinate-merged sparse reduction has no generic
            # serial twin: the legacy generic route writes an unsized result
            # vector, so only a verified sparse-workspace schedule may
            # rewrite this leaf into ordered assembly.  Anything reaching
            # the general lowering unscheduled fails closed.
            _fail(
                "unsupported_program_shape",
                "reducing into sparse result storage requires the verified "
                "sparse-workspace schedule; the generic route is not a "
                "migrated family",
            )
        if any(
            loop.kind in (_TILE_OUTER, _TILE_INNER, _PANEL_OUTER)
            and loop.node.width > MAX_AFFINE_TILE_WIDTH
            for loop in loops
        ):
            _fail(
                "unsupported_tile_width",
                "the C++ target emits affine tile widths as constexpr int "
                f"and therefore requires widths <= {MAX_AFFINE_TILE_WIDTH}",
            )
        tile_positions = [
            position
            for position, loop in enumerate(loops)
            if loop.kind in (_TILE_OUTER, _TILE_INNER)
        ]
        if tile_positions:
            if any(loop.kind is _MERGED for loop in loops):
                _fail(
                    "unsupported_program_shape",
                    "affine splits over merged iteration are outside the "
                    "migrated schedule families",
                )
            if type(leaf) is AppendEntry:
                _fail(
                    "unsupported_program_shape",
                    "affine splits over ordered sparse assembly are outside "
                    "the migrated schedule families",
                )
        merged_positions = [
            position for position, loop in enumerate(loops) if loop.kind is _MERGED
        ]
        if merged_positions:
            if merged_positions != [len(loops) - 1]:
                _fail(
                    "unsupported_program_shape",
                    "a merged loop is supported only as the innermost loop",
                )
            if len(loops[-1].cursors) != 2:
                _fail(
                    "unsupported_program_shape",
                    "this target lowering merges exactly two sparse cursors",
                )
            if type(leaf) is StoreReduce:
                _fail(
                    "unsupported_program_shape",
                    "merged reductions are outside the migrated families",
                )
        elif type(leaf) is AppendEntry:
            _fail(
                "unsupported_program_shape",
                "ordered sparse assembly requires a merged innermost loop "
                "in the migrated families",
            )
        for loop in loops:
            for cursor in loop.cursors:
                if cursor.level < 1 and loop.kind is not _SPARSE:
                    # A single-cursor loop descends a root-parent level-0
                    # segment exactly like a dense-parent one (the mixed
                    # dense-leaf operand chain's row loop); merged or
                    # windowed level-0 iteration stays fail-closed.
                    _fail(
                        "unsupported_program_shape",
                        "level-0 (root-parent) cursors are outside the "
                        "migrated merged and windowed families",
                    )

    def _collect_region_chains(
        self, region: WorkspaceRegion, prefix: List[_Loop]
    ) -> List[_Loop]:
        """Validate the migrated workspace shape; return the producer loops.

        The supported region form is exactly the legacy ``wksp[kTile]``
        producer/consumer shape: the producer is a dense/single-cursor
        reduction chain whose innermost loop is the owning split's point
        loop over one ``WorkspaceReduce``, and the consumer is one point
        loop of the same split over one result ``StoreReduce`` whose value
        is the plain workspace read.  Anything else fails closed.
        """

        decl = region.workspace
        origin_positions = [
            position
            for position, loop in enumerate(prefix)
            if loop.kind is _TILE_OUTER and loop.node.tile == decl.tile
        ]
        if not origin_positions:
            _fail(
                "unsupported_program_shape",
                "a workspace region's origin loop must be part of the " "outer chain",
            )

        producer_loops: List[_Loop] = []
        body: Stmt = region.producer
        while True:
            if type(body) is not Block or len(body.statements) != 1:
                _fail(
                    "unsupported_program_shape",
                    "a workspace producer must be a single-statement loop "
                    "chain over one workspace reduction",
                )
            only = body.statements[0]
            if type(only) is DenseFor:
                producer_loops.append(
                    _Loop(_DENSE, only.index, only.dimension, only, ())
                )
                body = only.body
                continue
            if type(only) is SparseFor:
                cursor = only.cursor
                producer_loops.append(
                    _Loop(
                        _SPARSE,
                        only.coord_index,
                        self._level_dimension(cursor.tensor, cursor.level),
                        only,
                        (cursor,),
                    )
                )
                body = only.body
                continue
            if type(only) is TileInnerFor:
                if only.tile != decl.tile:
                    # The reduce-out family strip-mines reduction loops
                    # above the owning point loop; each such split's origin
                    # must already be part of the outer chain.
                    if not any(
                        loop.kind is _TILE_OUTER and loop.node.tile == only.tile
                        for loop in prefix
                    ):
                        _fail(
                            "unsupported_program_shape",
                            "a strip-mined workspace reduction loop needs "
                            "its origin loop on the outer chain",
                        )
                    producer_loops.append(
                        _Loop(_TILE_INNER, only.index, only.dimension, only, ())
                    )
                    body = only.body
                    continue
                producer_loops.append(
                    _Loop(_TILE_INNER, only.index, only.dimension, only, ())
                )
                point_body = only.body
                if (
                    type(point_body) is not Block
                    or len(point_body.statements) != 1
                    or type(point_body.statements[0]) is not WorkspaceReduce
                ):
                    _fail(
                        "unsupported_program_shape",
                        "the workspace producer's point loop must reduce "
                        "into the workspace and nothing else",
                    )
                reduce_leaf = point_body.statements[0]
                if reduce_leaf.workspace != decl.workspace:
                    _fail(
                        "unsupported_program_shape",
                        "the workspace producer must reduce into the "
                        "region's own workspace",
                    )
                self.producer_leaf = reduce_leaf
                break
            _fail(
                "unsupported_program_shape",
                "workspace producers support dense loops, single-cursor "
                "sparse loops, strip-mined reduction point loops, and the "
                "owning point loop only",
            )
        if len(producer_loops) < 2:
            _fail(
                "unsupported_program_shape",
                "a workspace producer needs at least one reduction loop "
                "above its point loop",
            )

        consumer = region.consumer
        if (
            type(consumer) is not Block
            or len(consumer.statements) != 1
            or type(consumer.statements[0]) is not TileInnerFor
        ):
            _fail(
                "unsupported_program_shape",
                "a workspace consumer must be exactly one point loop of the "
                "owning split",
            )
        consumer_point = consumer.statements[0]
        if consumer_point.tile != decl.tile:
            _fail(
                "unsupported_program_shape",
                "a workspace consumer must be exactly one point loop of the "
                "owning split",
            )
        consumer_body = consumer_point.body
        if (
            type(consumer_body) is not Block
            or len(consumer_body.statements) != 1
            or type(consumer_body.statements[0]) is not StoreReduce
        ):
            _fail(
                "unsupported_program_shape",
                "a workspace consumer must copy the tile out with one "
                "result reduction",
            )
        copy_out = consumer_body.statements[0]
        value = copy_out.value
        if (
            type(value) is not WorkspaceRead
            or value.workspace != decl.workspace
            or type(value.coord) is not IndexValue
            or value.coord.index != consumer_point.index
        ):
            _fail(
                "unsupported_program_shape",
                "a workspace copy-out value must be the plain workspace "
                "read at the point coordinate",
            )
        self.consumer_point = consumer_point
        self._region_leaf = copy_out
        return producer_loops

    def _collect_leaf(self) -> Stmt:
        if self.region is not None:
            assert self._region_leaf is not None
            return self._region_leaf
        if self._tiled_view is not None:
            return self._tiled_view
        innermost = self.loops[-1].node.body
        return innermost.statements[0]

    def _validate_panel_shape(self) -> None:
        """Establish the supported panel form and record its chain anatomy.

        The migrated shape is exactly the legacy tile-j family: one panel
        origin, one window over a dense-parented (CSR) cursor, the dense
        row loop strictly between them, a dense result store leaf, and no
        merged, append-assembly, or workspace-region coexistence.
        """

        panel_positions = [
            position
            for position, loop in enumerate(self.loops)
            if loop.kind is _PANEL_OUTER
        ]
        window_positions = [
            position
            for position, loop in enumerate(self.loops)
            if loop.kind is _SPARSE_WINDOW
        ]
        if not panel_positions and not window_positions:
            return
        if len(panel_positions) != 1 or len(window_positions) != 1:
            _fail(
                "unsupported_program_shape",
                "this target lowering supports exactly one sparse panel "
                "(one origin loop and one window)",
            )
        panel_node = self.loops[panel_positions[0]].node
        window_node = self.loops[window_positions[0]].node
        if window_node.tile != panel_node.tile:
            _fail(
                "unsupported_program_shape",
                "the panel origin and window must share one tile identity",
            )
        if self.region is not None:
            _fail(
                "unsupported_program_shape",
                "sparse panels do not compose with workspace regions in the "
                "migrated families",
            )
        if any(loop.kind is _MERGED for loop in self.loops):
            _fail(
                "unsupported_program_shape",
                "sparse panels over merged iteration are outside the "
                "migrated families",
            )
        if not self.result_is_dense or type(self.leaf) is AppendEntry:
            _fail(
                "unsupported_program_shape",
                "sparse panels require a dense result store leaf",
            )
        cursor = window_node.cursor
        parent = cursor.parent
        if type(parent) is not DensePosition or type(parent.coord) is not IndexValue:
            _fail(
                "unsupported_program_shape",
                "the panel window's cursor must be dominated by a dense "
                "position over a directly bound row coordinate (the CSR "
                "form)",
            )
        row_index = parent.coord.index
        row_positions = [
            position
            for position, loop in enumerate(self.loops)
            if loop.kind is _DENSE and loop.index == row_index
        ]
        if not row_positions:
            _fail(
                "unsupported_program_shape",
                "the panel window's row coordinate must be bound by a plain "
                "dense chain loop",
            )
        if not panel_positions[0] < row_positions[0] < window_positions[0]:
            _fail(
                "unsupported_program_shape",
                "the panel origin must sit strictly above the parallel row "
                "loop and the window strictly below it",
            )
        result_indices = {
            self._index_of(index, "the panel result access")
            for index in self._leaf_indices()
        }
        if row_index not in result_indices:
            _fail(
                "unsupported_program_shape",
                "the panel's selected parallel row loop must partition the "
                "dense result access",
            )
        self.panel = panel_node
        self.panel_position = panel_positions[0]
        self.window_position = window_positions[0]
        self.panel_row_position = row_positions[0]

    def _validate_relayout_shape(self) -> None:
        """Establish the supported relayout form and its staged read view.

        The migrated shape is exactly the audited legacy packed tile-ijk
        family: the outermost affine pack origin, the panel origin directly
        below it, the parallel dense row loop, the panel window, and the
        pack point loop, with the staging region opened at the plan scope
        and the leaf reading the operand through exactly one StagedRead.
        The staged read is recorded as a synthetic direct-Load view so raw
        emission and every managed pass see byte-for-byte the tree the
        legacy pipeline transforms; the redirection to packed storage
        happens in :meth:`complete_relayout` on the assembled function,
        exactly where the legacy schedule lowering performs it.
        """

        if self.relayout is None:
            return
        decl = self.relayout
        if self.panel is None:
            _fail(
                "unsupported_program_shape",
                "a staged relayout region requires the sparse panel family",
            )
        kinds = [loop.kind for loop in self.loops]
        if kinds != [_TILE_OUTER, _PANEL_OUTER, _DENSE, _SPARSE_WINDOW, _TILE_INNER]:
            _fail(
                "unsupported_program_shape",
                "staged relayout supports exactly the packed tile-ijk "
                "chain: pack origin, panel origin, parallel row, panel "
                "window, pack point",
            )
        pack_node = self.loops[0].node
        panel_node = self.loops[1].node
        point_node = self.loops[4].node
        if (
            pack_node.tile != point_node.tile
            or decl.pack != pack_node.tile
            or decl.panel != panel_node.tile
        ):
            _fail(
                "unsupported_program_shape",
                "the staging region must name the chain's pack split and " "panel pair",
            )
        expected_depth = 2 if decl.scope is RelayoutScope.PANEL else 1
        if self.relayout_depth != expected_depth:
            _fail(
                "unsupported_program_shape",
                "the staging region must open directly below its scope "
                "loop: the panel origin for PANEL scope, the pack origin "
                "for PACK_AXIS scope",
            )
        operand_decl = self.decls.get(decl.operand)
        if (
            operand_decl is None
            or decl.operand not in self._input_symbols
            or len(operand_decl.levels) != 2
            or any(level.kind is not LevelKind.DENSE for level in operand_decl.levels)
            or operand_decl.dimensions[operand_decl.levels[0].mode]
            != panel_node.dimension
            or operand_decl.dimensions[operand_decl.levels[1].mode]
            != pack_node.dimension
        ):
            _fail(
                "unsupported_program_shape",
                "the staged operand must be the declared rank-2 dense "
                "input storing the panel then pack dimensions",
            )

        staged_reads: List[StagedRead] = []

        def walk(expr: Expr) -> None:
            if type(expr) is StagedRead:
                staged_reads.append(expr)
                return
            if type(expr) is BinaryExpr:
                walk(expr.lhs)
                walk(expr.rhs)

        walk(self.leaf.value)  # type: ignore[attr-defined]
        if len(staged_reads) != 1 or staged_reads[0].relayout != decl.relayout:
            _fail(
                "unsupported_program_shape",
                "staged relayout requires exactly one StagedRead of its "
                "region in the compute leaf",
            )
        staged = staged_reads[0]
        panel_mode = operand_decl.levels[0].mode
        pack_mode = operand_decl.levels[1].mode
        if (
            len(staged.indices) != 2
            or self._index_of(
                staged.indices[panel_mode],
                "the staged panel index",
            )
            != panel_node.index
            or self._index_of(
                staged.indices[pack_mode],
                "the staged pack index",
            )
            != point_node.index
        ):
            _fail(
                "unsupported_program_shape",
                "the staged read's physical levels must be driven by the "
                "panel window coordinate and the pack point coordinate",
            )
        self._staged_views[id(staged)] = Load(
            LoopIRNodeId(-1), decl.operand, staged.indices
        )

    def _validate_result_tile_shape(self) -> None:
        """Establish the supported heap result-tile form and its anatomy.

        The migrated shape is exactly the audited legacy family on the
        rank>=2 trailing-axis result: the outermost pack origin whose whole
        body the region wraps, the result's dense prefix loops in physical
        storage order (the outermost of which carries the parallel policy),
        one reduction loop (or the panel window pair), and the pack point
        loop, with the compute leaf accumulating through exactly one
        TiledReduce of the region.  A multi-prefix result contributes one
        dense loop per prefix axis at a derived chain position; a sparse
        panel may window the reduction below that derived prefix.  The
        relayout composition remains its separately audited
        single-prefix/rank-2 operand-staging subfamily.  The leaf was
        recorded as a synthetic direct StoreReduce view by the
        nest walk; the compact redirection happens in
        :meth:`complete_result_tile` on the assembled function, exactly
        where the legacy schedule lowering performs it.
        """

        if self.result_tile is None:
            return
        decl = self.result_tile
        if self._tiled_leaf is None:
            _fail(
                "unsupported_program_shape",
                "a result-tile region requires the tiled-reduce compute "
                "leaf of its own region",
            )
        result_decl = self.result_decl
        rank = len(result_decl.levels)
        prefix_rank = rank - 1
        kinds = [loop.kind for loop in self.loops]
        heap_alone = (
            rank >= 2
            and len(kinds) == rank + 2
            and kinds[0] == _TILE_OUTER
            and all(kind == _DENSE for kind in kinds[1:rank])
            and kinds[rank] in (_DENSE, _SPARSE)
            and kinds[-1] == _TILE_INNER
        )
        heap_panel = (
            rank >= 2
            and len(kinds) == rank + 3
            and kinds[0] == _TILE_OUTER
            and kinds[1] == _PANEL_OUTER
            and all(kind == _DENSE for kind in kinds[2 : rank + 1])
            and kinds[rank + 1] == _SPARSE_WINDOW
            and kinds[-1] == _TILE_INNER
        )
        if not heap_alone and not heap_panel:
            _fail(
                "unsupported_program_shape",
                "heap accumulation supports exactly the audited chains: "
                "the pack origin over the result's dense prefix loops, one "
                "reduction loop or the panel window pair, and the pack "
                "point loop",
            )
        if self.result_tile_depth != 1:
            _fail(
                "unsupported_program_shape",
                "the result-tile region must wrap the pack origin's " "entire body",
            )
        pack_node = self.loops[0].node
        point_node = self.loops[-1].node
        if pack_node.tile != point_node.tile or decl.pack != pack_node.tile:
            _fail(
                "unsupported_program_shape",
                "the result-tile region must name the chain's pack split " "pair",
            )
        prefix_start = 2 if heap_panel else 1
        # The composed default anchor is the window's dense parent — the
        # innermost prefix loop and the only anchor legacy admits for the
        # composition; the bare chain defaults to the outermost prefix.
        # An abstract parallel selection may adopt any admissible prefix
        # anchor afterwards (_validate_parallel_selection).
        row_position = prefix_start + prefix_rank - 1 if heap_panel else 1
        prefix_nodes = [
            self.loops[prefix_start + offset].node for offset in range(prefix_rank)
        ]
        prefix_levels = result_decl.levels[:prefix_rank]
        pack_level = result_decl.levels[prefix_rank]
        if (
            decl.result != self.result_symbol
            or any(level.kind is not LevelKind.DENSE for level in result_decl.levels)
            or any(
                result_decl.dimensions[level.mode] != prefix_nodes[position].dimension
                for position, level in enumerate(prefix_levels)
            )
            or result_decl.dimensions[pack_level.mode] != pack_node.dimension
        ):
            _fail(
                "unsupported_program_shape",
                "the compacted result must be the declared all-dense output "
                "storing its prefix dimensions then the pack dimension",
            )
        tiled = self._tiled_leaf
        if (
            len(tiled.indices) != rank
            or any(
                self._index_of(
                    tiled.indices[level.mode],
                    "the compacted prefix index",
                )
                != prefix_nodes[position].index
                for position, level in enumerate(prefix_levels)
            )
            or self._index_of(
                tiled.indices[pack_level.mode],
                "the compacted column index",
            )
            != point_node.index
        ):
            _fail(
                "unsupported_program_shape",
                "the tiled reduce must be indexed by the prefix coordinates "
                "and the pack point coordinate",
            )
        self.result_tile_row_position = row_position

    def _validate_parallel_selection(self) -> None:
        """Resolve the program's abstract parallel selection on the chain.

        A missing selection preserves every existing derivation: the
        generic auto gate, the panel row, and the heap's outermost-prefix
        default all stay byte-identical — bare verified programs keep
        direct structural activation.  A present selection must resolve to
        exactly one chain loop, restate that loop's trip-count dimension,
        and agree with the route that owns the marking: the heap
        completion adopts the selected prefix loop as its parallel row,
        the panel (including relayout) and heap completions require the
        selection to name the row they mark, and the direct/stack routes
        mark the selected loop on the assembled function through
        :meth:`complete_parallel`.  Completion revalidates the immutable
        fact before realizing its exact work policy.
        """

        selection = self.parallel
        if selection is None:
            return
        position = -1
        for candidate, loop in enumerate(self.loops):
            node = loop.node
            if selection.part is ParallelPart.LOGICAL:
                matched = (
                    type(node) is DenseFor
                    and node.index == selection.index
                    or type(node) in (SparseFor, MergedSparseFor)
                    and node.coord_index == selection.index
                )
            else:
                matched = (
                    type(node) in (TileOuterFor, PanelOuterFor)
                    and node.index == selection.index
                )
            if matched:
                position = candidate
                break
        if position < 0:
            _fail(
                "unsupported_parallel_selection",
                "the program's parallel selection does not name a chain "
                "loop this target emits",
            )
        self.parallel_position = position
        selected = self.loops[position]
        if type(selected.node) not in (DenseFor, TileOuterFor):
            # The legacy explicit route marks tagged for-loops only;
            # compressed, merged, and panel-origin anchors have no measured
            # comparand, so this target refuses them rather than inventing
            # a policy.
            _fail(
                "unsupported_parallel_selection",
                "this target realizes parallel selections on dense logical "
                "loops and affine origin loops only",
            )
        loop_dimension = selected.node.dimension
        if selection.work.rows != loop_dimension:
            _fail(
                "unsupported_parallel_selection",
                "the parallel selection's work estimate does not restate "
                "the selected loop's dimension",
            )
        if self.result_tile is not None:
            prefix_rank = len(self.result_decl.levels) - 1
            prefix_start = 2 if self.panel is not None else 1
            if (
                selection.discipline is not ParallelDiscipline.COMPACT_PARTITION
                or not prefix_start <= position < prefix_start + prefix_rank
            ):
                _fail(
                    "unsupported_parallel_selection",
                    "a heap program's parallel selection must partition the "
                    "compact tile through one dense prefix loop",
                )
            if self.panel is not None and position != self.panel_row_position:
                _fail(
                    "unsupported_parallel_selection",
                    "a composed heap-panel selection must name the window's "
                    "dense-parent row loop the panel completion marks",
                )
            self.result_tile_row_position = position
        elif self.panel is not None:
            if (
                selection.discipline is not ParallelDiscipline.RESULT_PARTITION
                or position != self.panel_row_position
            ):
                _fail(
                    "unsupported_parallel_selection",
                    "a panel program's parallel selection must name the "
                    "window's dense-parent row loop the completion marks",
                )
        elif selection.discipline is not ParallelDiscipline.RESULT_PARTITION:
            _fail(
                "unsupported_parallel_selection",
                "a direct-accumulation program partitions its dense result; "
                "the compact discipline does not apply",
            )
        self._parallel_signature = self._parallel_selection_signature(selection)

    @staticmethod
    def _parallel_selection_signature(selection: object) -> Tuple[object, ...]:
        """Snapshot the exact primitive state target realization depends on."""

        if type(selection) is not ParallelSelection:
            raise TypeError("parallel selection must remain exact")
        work = selection.work
        if type(work) is not ParallelWork:
            raise TypeError("parallel work must remain exact")

        def identity_value(value: object, expected: type, owner: str) -> int:
            if type(value) is not expected:
                raise TypeError(f"{owner} must remain an exact {expected.__name__}")
            stored = object.__getattribute__(value, "__dict__")
            if (
                type(stored) is not dict
                or tuple(stored) != ("value",)
                or type(stored["value"]) is not int
            ):
                raise TypeError(f"{owner} must retain one exact integer value")
            return stored["value"]

        source = work.nnz
        source_signature: Optional[Tuple[int, int, int]]
        if source is None:
            source_signature = None
        else:
            if type(source) is not SparseWorkSource:
                raise TypeError("parallel sparse work source must remain exact")
            source_signature = (
                identity_value(
                    source.node_id,
                    LoopIRNodeId,
                    "SparseWorkSource.node_id",
                ),
                identity_value(source.tensor, SymbolId, "SparseWorkSource.tensor"),
                source.level,
            )
            if type(source.level) is not int:
                raise TypeError("SparseWorkSource.level must remain an exact int")
        return (
            identity_value(
                selection.node_id,
                LoopIRNodeId,
                "ParallelSelection.node_id",
            ),
            identity_value(selection.index, IndexId, "ParallelSelection.index"),
            selection.part,
            selection.discipline,
            identity_value(work.node_id, LoopIRNodeId, "ParallelWork.node_id"),
            identity_value(work.rows, DimensionId, "ParallelWork.rows"),
            source_signature,
            selection.intent,
        )

    def _parallel_work_policy_spec(
        self,
        error_code: str,
    ) -> Optional[str]:
        """Revalidate and resolve the selected work fact to owned C++ names."""

        try:
            verify_program(self.program)
            if self.program.parallel is not self.parallel:
                raise TypeError("the program's parallel selection was replaced")
            signature = self._parallel_selection_signature(self.parallel)
            if signature != self._parallel_signature:
                raise TypeError(
                    "the program's parallel selection changed after lowering"
                )
            assert self.parallel is not None
            source = self.parallel.work.nnz
            if source is None:
                return None
            decl = self.decls.get(source.tensor)
            if (
                decl is None
                or source.level < 1
                or source.level >= len(decl.levels)
                or decl.levels[source.level].kind is not LevelKind.COMPRESSED
                or decl.levels[source.level - 1].kind is not LevelKind.DENSE
            ):
                raise TypeError(
                    "the parallel sparse work source is not a dense-parented "
                    "compressed level"
                )
            return f"{decl.name}{source.level}_pos"
        except (
            AssertionError,
            AttributeError,
            KeyError,
            LoopIRVerificationError,
            RecursionError,
            TypeError,
            ValueError,
        ) as error:
            _fail(error_code, str(error))

    def _apply_selected_parallel_policy(
        self,
        loop: llir.ForLoop,
        policy_spec: Optional[str],
        error_code: str,
        *,
        invoke_marker: bool = True,
    ) -> None:
        """Apply one exact selected policy, honoring row-only work explicitly."""

        try:
            if policy_spec is not None:
                actual_position_array = find_sparse_pos_array(loop.body)
                if actual_position_array != policy_spec:
                    raise ValueError(
                        "the selected loop's structural work source disagrees "
                        "with the canonical parallel work fact"
                    )
                loop_bound = extract_loop_bound(loop)
                if loop_bound is None:
                    raise ValueError(
                        "the selected loop no longer has a canonical target bound"
                    )
            if invoke_marker:
                mark_first_for_loop_parallel(
                    [loop],
                    EMPTY_PARALLEL_WORKSPACE_CLUSTER,
                )
            else:
                # Construct the independent expected snapshot without
                # trusting the marker whose output it checks.
                loop.omp_parallel_for = True
                if has_sparse_inner_loop(loop.body):
                    loop.omp_schedule = "dynamic, 64"
            if policy_spec is None:
                # ``None`` is the canonical row-only estimate, including the
                # established merged-nest behavior.  Override any discovery
                # the target's more explicit LLIR happens to expose.
                apply_parallel_policy(loop, body=())
            else:
                apply_parallel_policy(
                    loop,
                    work_expr=f"{policy_spec}[{loop_bound}]",
                )
        except (
            AttributeError,
            LLIRTraversalError,
            RecursionError,
            TypeError,
            ValueError,
        ) as error:
            _fail(error_code, str(error))

    @staticmethod
    def _apply_expected_parallel_marker(loop: llir.ForLoop) -> None:
        """Construct the EMPTY-cluster marker's expected state independently."""

        loop.omp_parallel_for = True
        if has_sparse_inner_loop(loop.body):
            loop.omp_schedule = "dynamic, 64"
        apply_parallel_policy(loop)

    def _require_retained_panel_parallel_policy(self, error_code: str) -> None:
        """Revalidate the policy handed from panel to a later completion."""

        state = self._panel_completion
        expected = self._panel_parallel_snapshot
        if (
            state is None
            or expected is None
            or not self._panel_loop_header_matches(state[2], expected)
        ):
            _fail(
                error_code,
                "the panel-owned row policy changed before the next "
                "completion boundary",
            )
        if self.parallel is not None:
            self._parallel_work_policy_spec(error_code)

    def _reserve_panel_names(self) -> None:
        """Reserve the derived loop, bound, and search names panels generate."""

        if self.panel is None:
            return
        name = self.dimension_names[self.panel.dimension]
        owner = f"sparse panel of dimension {name!r}"
        for generated in (f"{name}_out", f"{name}_out_end", f"kTile_{name}"):
            self._reserve_generated_name(generated, owner)
        cursor = self.loops[self.window_position].cursors[0]
        position_name = self._cursor_position_name(cursor)
        for generated in (
            f"{position_name}_row_end",
            f"{position_name}_panel_begin",
        ):
            self._reserve_generated_name(generated, owner)

    def _index_of(self, expr: Expr, path: str) -> object:
        if type(expr) is not IndexValue:
            _fail(
                "unsupported_program_shape",
                f"{path} must be a directly bound loop coordinate",
            )
        return expr.index

    def _access_value_expression(self) -> Expr:
        """Return the one value-expression root this target accepted."""

        leaf: object
        if self.region is not None:
            leaf = self.producer_leaf
        else:
            leaf = self.leaf
        if type(leaf) not in (Store, StoreReduce, AppendEntry, WorkspaceReduce):
            _fail(
                "unsupported_program_shape",
                "the target leaf no longer owns one supported value expression",
            )
        state = object.__getattribute__(leaf, "__dict__")
        if type(state) is not dict or "value" not in state:
            _fail(
                "unsupported_program_shape",
                "the target leaf has malformed stored value state",
            )
        return cast(Expr, state["value"])

    def _validated_value_expression_signature(
        self, expression: Expr
    ) -> Tuple[object, ...]:
        """Build an ordered, occurrence-sensitive primitive value signature.

        The verifier establishes semantic validity before target construction,
        while this target additionally owns exactly one physical read per
        input.  Frozen nodes remain forgeable, however: replacing one read by
        another node kind, or reusing an already-admitted read object twice,
        used to bypass the per-kind access census.  Preserve the entire value
        tree -- node identity, operator, ordered children, and every access
        address -- so any post-construction rewrite fails before emission.

        Shared DAG children are legal when they were present in the verified
        tree; ``active`` detects only a back-edge, and the repeated occurrence
        is represented a second time in the returned signature.
        """

        active: Set[int] = set()

        def state_of(
            value: object, expected: type, fields: Set[str]
        ) -> Tuple[Dict[str, Any], int]:
            if type(value) is not expected:
                _fail(
                    "unsupported_program_shape",
                    "the target value expression contains a malformed node",
                )
            state = object.__getattribute__(value, "__dict__")
            if (
                type(state) is not dict
                or any(type(key) is not str for key in state)
                or set(state) != fields
            ):
                _fail(
                    "unsupported_program_shape",
                    "the target value expression contains malformed stored state",
                )
            node_value = _stored_identity_value(state["node_id"], LoopIRNodeId)
            if node_value is None:
                _fail(
                    "unsupported_program_shape",
                    "the target value expression contains a malformed node identity",
                )
            return state, node_value

        def identity(value: object, expected: type, label: str) -> int:
            primitive = _stored_identity_value(value, expected)
            if primitive is None:
                _fail(
                    "unsupported_program_shape",
                    f"the target value expression has a malformed {label}",
                )
            return primitive

        def visit(value: object, depth: int) -> Tuple[object, ...]:
            if depth > 256 or id(value) in active:
                _fail(
                    "unsupported_program_shape",
                    "the target value expression must be finite and acyclic",
                )
            active.add(id(value))
            try:
                if type(value) is BinaryExpr:
                    state, node = state_of(
                        value,
                        BinaryExpr,
                        {"node_id", "op", "lhs", "rhs"},
                    )
                    op = state["op"]
                    if (
                        op is not BinaryOp.ADD
                        and op is not BinaryOp.SUB
                        and op is not BinaryOp.MUL
                    ):
                        _fail(
                            "unsupported_program_shape",
                            "the target value expression has a malformed binary operator",
                        )
                    op_tag = (
                        "add"
                        if op is BinaryOp.ADD
                        else "sub" if op is BinaryOp.SUB else "mul"
                    )
                    return (
                        "binary",
                        id(value),
                        node,
                        op_tag,
                        visit(state["lhs"], depth + 1),
                        visit(state["rhs"], depth + 1),
                    )
                if type(value) is Load:
                    state, node = state_of(
                        value,
                        Load,
                        {"node_id", "tensor", "indices"},
                    )
                    indices = state["indices"]
                    if type(indices) is not tuple:
                        _fail(
                            "unsupported_program_shape",
                            "a coordinate load must retain its owned index tuple",
                        )
                    return (
                        "load",
                        id(value),
                        node,
                        identity(state["tensor"], SymbolId, "load tensor identity"),
                        tuple(visit(index, depth + 1) for index in indices),
                    )
                if type(value) is PositionLoad:
                    state, node = state_of(
                        value,
                        PositionLoad,
                        {"node_id", "tensor", "position"},
                    )
                    return (
                        "position_load",
                        id(value),
                        node,
                        identity(
                            state["tensor"],
                            SymbolId,
                            "position-load tensor identity",
                        ),
                        visit(state["position"], depth + 1),
                    )
                if type(value) is StagedRead:
                    state, node = state_of(
                        value,
                        StagedRead,
                        {"node_id", "relayout", "indices"},
                    )
                    indices = state["indices"]
                    if type(indices) is not tuple:
                        _fail(
                            "unsupported_program_shape",
                            "a staged read must retain its owned index tuple",
                        )
                    return (
                        "staged_read",
                        id(value),
                        node,
                        identity(state["relayout"], RelayoutId, "relayout identity"),
                        tuple(visit(index, depth + 1) for index in indices),
                    )
                if type(value) is CursorValue:
                    state, node = state_of(
                        value,
                        CursorValue,
                        {"node_id", "cursor", "default"},
                    )
                    default = state["default"]
                    return (
                        "cursor_value",
                        id(value),
                        node,
                        identity(state["cursor"], CursorId, "cursor identity"),
                        None if default is None else visit(default, depth + 1),
                    )
                if type(value) is IndexValue:
                    state, node = state_of(value, IndexValue, {"node_id", "index"})
                    return (
                        "index",
                        id(value),
                        node,
                        identity(state["index"], IndexId, "index identity"),
                    )
                if type(value) is DensePosition:
                    state, node = state_of(
                        value,
                        DensePosition,
                        {"node_id", "tensor", "level", "parent", "coord"},
                    )
                    level = state["level"]
                    if type(level) is not int or level < 0:
                        _fail(
                            "unsupported_program_shape",
                            "a dense position has a malformed level",
                        )
                    return (
                        "dense_position",
                        id(value),
                        node,
                        identity(
                            state["tensor"], SymbolId, "dense-position tensor identity"
                        ),
                        level,
                        visit(state["parent"], depth + 1),
                        visit(state["coord"], depth + 1),
                    )
                if type(value) is PositionValue:
                    state, node = state_of(
                        value, PositionValue, {"node_id", "position"}
                    )
                    return (
                        "position",
                        id(value),
                        node,
                        identity(state["position"], PositionId, "position identity"),
                    )
                if type(value) is RootPosition:
                    _, node = state_of(value, RootPosition, {"node_id"})
                    return ("root_position", id(value), node)
                if type(value) is FloatConst:
                    state, node = state_of(value, FloatConst, {"node_id", "value"})
                    literal = state["value"]
                    if type(literal) is not float:
                        _fail(
                            "unsupported_program_shape",
                            "a cursor default must retain an exact float value",
                        )
                    return ("float", id(value), node, literal.hex())
                if type(value) is WorkspaceRead:
                    state, node = state_of(
                        value,
                        WorkspaceRead,
                        {"node_id", "workspace", "coord"},
                    )
                    return (
                        "workspace_read",
                        id(value),
                        node,
                        identity(state["workspace"], WorkspaceId, "workspace identity"),
                        visit(state["coord"], depth + 1),
                    )
                if type(value) is SparseWorkspaceValue:
                    state, node = state_of(
                        value,
                        SparseWorkspaceValue,
                        {"node_id", "workspace"},
                    )
                    return (
                        "sparse_workspace_value",
                        id(value),
                        node,
                        identity(state["workspace"], WorkspaceId, "workspace identity"),
                    )
                _fail(
                    "unsupported_program_shape",
                    f"unsupported target value node {type(value).__name__}",
                )
            finally:
                active.discard(id(value))

        return visit(expression, 0)

    def _validated_target_owner_signature(self) -> Tuple[object, ...]:
        """Bind value trees and access coordinates to their actual owners.

        Heap and workspace lowering deliberately expose synthetic direct-write
        views to the legacy-shaped LLIR emitter.  The semantic LoopIR owners
        must nevertheless remain the exact statements from which those views
        were derived: otherwise replacing a ``TiledReduce`` field or the
        workspace consumer could be silently ignored, while mutating a shared
        index child could redirect compact writes.  Snapshot both the actual
        owner and any synthetic view, in order, with exact stored fields.
        """

        def state_of(
            statement: object, expected: type, fields: Set[str]
        ) -> Tuple[Dict[str, Any], int]:
            if type(statement) is not expected:
                _fail(
                    "unsupported_program_shape",
                    "the target value owner contains a malformed statement",
                )
            state = object.__getattribute__(statement, "__dict__")
            if (
                type(state) is not dict
                or any(type(key) is not str for key in state)
                or set(state) != fields
            ):
                _fail(
                    "unsupported_program_shape",
                    "the target value owner contains malformed stored state",
                )
            node = _stored_identity_value(state["node_id"], LoopIRNodeId)
            if node is None:
                _fail(
                    "unsupported_program_shape",
                    "the target value owner has a malformed node identity",
                )
            return state, node

        def identity(value: object, expected: type, label: str) -> int:
            primitive = _stored_identity_value(value, expected)
            if primitive is None:
                _fail(
                    "unsupported_program_shape",
                    f"the target value owner has a malformed {label}",
                )
            return primitive

        def indices(value: object, label: str) -> Tuple[Tuple[object, ...], ...]:
            if type(value) is not tuple:
                _fail(
                    "unsupported_program_shape",
                    f"the target {label} must remain an owned tuple",
                )
            return tuple(
                self._validated_value_expression_signature(index) for index in value
            )

        def reduction_op(value: object) -> str:
            if value is not ReduceOp.ADD:
                _fail(
                    "unsupported_program_shape",
                    "the target value owner has a malformed reduction operator",
                )
            return "add"

        def statement(statement: object) -> Tuple[object, ...]:
            if type(statement) is Store:
                state, node = state_of(
                    statement,
                    Store,
                    {"node_id", "tensor", "indices", "value"},
                )
                return (
                    "store",
                    id(statement),
                    node,
                    identity(state["tensor"], SymbolId, "store tensor identity"),
                    indices(state["indices"], "store indices"),
                    self._validated_value_expression_signature(state["value"]),
                )
            if type(statement) is StoreReduce:
                state, node = state_of(
                    statement,
                    StoreReduce,
                    {"node_id", "tensor", "indices", "op", "value"},
                )
                return (
                    "store_reduce",
                    id(statement),
                    node,
                    identity(state["tensor"], SymbolId, "store-reduce tensor identity"),
                    indices(state["indices"], "store-reduce indices"),
                    reduction_op(state["op"]),
                    self._validated_value_expression_signature(state["value"]),
                )
            if type(statement) is AppendEntry:
                state, node = state_of(
                    statement,
                    AppendEntry,
                    {"node_id", "tensor", "coords", "value"},
                )
                return (
                    "append_entry",
                    id(statement),
                    node,
                    identity(state["tensor"], SymbolId, "append tensor identity"),
                    indices(state["coords"], "append coordinates"),
                    self._validated_value_expression_signature(state["value"]),
                )
            if type(statement) is WorkspaceReduce:
                state, node = state_of(
                    statement,
                    WorkspaceReduce,
                    {"node_id", "workspace", "coord", "op", "value"},
                )
                return (
                    "workspace_reduce",
                    id(statement),
                    node,
                    identity(state["workspace"], WorkspaceId, "workspace identity"),
                    self._validated_value_expression_signature(state["coord"]),
                    reduction_op(state["op"]),
                    self._validated_value_expression_signature(state["value"]),
                )
            if type(statement) is TiledReduce:
                state, node = state_of(
                    statement,
                    TiledReduce,
                    {"node_id", "result_tile", "indices", "op", "value"},
                )
                return (
                    "tiled_reduce",
                    id(statement),
                    node,
                    identity(
                        state["result_tile"],
                        ResultTileId,
                        "result-tile identity",
                    ),
                    indices(state["indices"], "tiled-reduce indices"),
                    reduction_op(state["op"]),
                    self._validated_value_expression_signature(state["value"]),
                )
            _fail(
                "unsupported_program_shape",
                "the target leaf no longer has a supported value owner",
            )

        if self.region is not None:
            if self.producer_leaf is None or self._region_leaf is None:
                _fail(
                    "unsupported_program_shape",
                    "the workspace region lost a producer or consumer owner",
                )
            return (
                "workspace_region",
                statement(self.producer_leaf),
                statement(self._region_leaf),
            )
        if self._tiled_leaf is not None:
            if self._tiled_view is None:
                _fail(
                    "unsupported_program_shape",
                    "the result tile lost its synthetic direct-write view",
                )
            return (
                "result_tile",
                statement(self._tiled_leaf),
                statement(self._tiled_view),
            )
        return ("leaf", statement(self.leaf))

    def _require_value_expression_unchanged(self) -> None:
        """Revalidate the exact value tree immediately before target use."""

        # Preserve the narrowest diagnostic at this boundary: malformed
        # position spines and replaced access/owner nodes are characterized by
        # the value validators before the complete graph guard reports a more
        # general retained-graph change.  On the overwhelmingly common
        # unchanged path, the complete graph signature already subsumes those
        # narrower signatures, so return before rebuilding them.  If the graph
        # scan itself fails, defer that error until the narrow validators have
        # had the opportunity to retain their established diagnostics.
        _TargetLowering._require_program_inputs_unchanged(self)
        _TargetLowering._require_target_state_unchanged(self)
        target_state = _TargetLowering._require_exact_target_type(self)
        authority = _target_seal_authority(self)
        bound_program = authority.program
        if (
            type(bound_program) is not LoopProgram
            or target_state.get("_program_container") is not bound_program
        ):
            _fail(
                "unsupported_program_shape",
                "the target program reference changed after target construction",
            )
        graph_error: Optional[LoopIRTargetError] = None
        try:
            _TargetLowering._require_program_graph_snapshot_unchanged(
                self,
                bound_program,
                authority.graph_snapshot,
            )
        except LoopIRTargetError as error:
            graph_error = error
        else:
            return

        current_bindings = self._validated_bound_position_bindings()
        if current_bindings != self._bound_position_snapshot:
            _fail(
                "unsupported_program_shape",
                "a position-binding loop changed after target construction",
            )
        current_value = self._validated_value_expression_signature(
            self._access_value_expression()
        )
        current_owner = self._validated_target_owner_signature()
        if (
            current_value != self._value_expression_snapshot
            or current_owner != self._target_owner_snapshot
        ):
            _fail(
                "unsupported_program_shape",
                "the target value expression changed after target construction, "
                "or its owning statement was replaced",
            )
        if graph_error is not None:
            raise graph_error
        _fail(
            "unsupported_program_shape",
            "the program graph, including a target owning statement, "
            "changed after target construction",
        )

    def _collect_accesses(self) -> Tuple[List[Load], List[CursorValue]]:
        loads: List[Load] = []
        cursor_values: List[CursorValue] = []
        position_loads: List[PositionLoad] = []
        self.position_loads = position_loads

        def walk(expr: Expr) -> None:
            if type(expr) is Load:
                loads.append(expr)
                return
            if type(expr) is PositionLoad:
                position_loads.append(expr)
                return
            if type(expr) is StagedRead:
                # The staged read's direct-Load view participates in the
                # access machinery exactly like the read it redirected, so
                # drivers, bounds, and raw emission match the unstaged
                # program byte-for-byte; complete_relayout redirects the
                # emitted value expression afterwards.
                view = self._staged_views.get(id(expr))
                if view is None:
                    _fail(
                        "unsupported_program_shape",
                        "a StagedRead outside the validated staging region "
                        "is not lowerable",
                    )
                loads.append(view)
                return
            if type(expr) is CursorValue:
                cursor_values.append(expr)
                return
            if type(expr) is BinaryExpr:
                walk(expr.lhs)
                walk(expr.rhs)
                return
            if type(expr) is IndexValue:
                _fail(
                    "unsupported_program_shape",
                    "coordinate values are not value expressions in the "
                    "migrated families",
                )
            _fail(
                "unsupported_program_shape",
                f"unsupported value expression {type(expr).__name__}",
            )

        if self.region is not None:
            # The producer's reduction value owns every input access; the
            # consumer's value is the workspace read, not a tensor access.
            assert self.producer_leaf is not None
            walk(self.producer_leaf.value)
        else:
            walk(self.leaf.value)  # type: ignore[attr-defined]
        seen: Set[SymbolId] = set()
        for load in loads:
            if load.tensor in seen:
                _fail(
                    "unsupported_repeated_operand",
                    f"input tensor {self.decls[load.tensor].name!r} is loaded "
                    "more than once; this target owns one physical "
                    "position chain per input",
                )
            seen.add(load.tensor)
        for position_load in position_loads:
            tensor, _, signature = self._validated_position_load_spine(
                position_load, require_unconditional=False
            )
            self._position_load_signatures[id(position_load)] = signature
            if tensor in seen:
                _fail(
                    "unsupported_repeated_operand",
                    f"input tensor "
                    f"{self.decls[tensor].name!r} is loaded "
                    "more than once; this target owns one physical "
                    "position chain per input",
                )
            seen.add(tensor)
        for cursor_value in cursor_values:
            loop_position = self.cursor_loops.get(cursor_value.cursor)
            if loop_position is None:
                _fail(
                    "unsupported_program_shape",
                    "a cursor value must read a nest loop cursor",
                )
            cursor = self._cursor_decl(cursor_value.cursor)
            if cursor.tensor in seen:
                _fail(
                    "unsupported_repeated_operand",
                    f"input tensor {self.decls[cursor.tensor].name!r} is read "
                    "more than once; this target owns one physical "
                    "position chain per input",
                )
            seen.add(cursor.tensor)
        return loads, cursor_values

    def _cursor_decl(self, cursor: CursorId) -> SparseCursorDecl:
        position = self.cursor_loops[cursor]
        for decl in self.loops[position].cursors:
            if decl.cursor == cursor:
                return decl
        raise AssertionError("unreachable")

    def _bound_position_owner(
        self, position: PositionId
    ) -> Optional[Tuple[SymbolId, int]]:
        """The (tensor, level) whose nest loop binds one position."""

        position_value = _stored_identity_value(position, PositionId)
        if position_value is None:
            return None
        bindings = self._validated_bound_position_bindings()
        if bindings != self._bound_position_snapshot:
            _fail(
                "unsupported_program_shape",
                "a position-binding loop changed after target construction",
            )
        binding = bindings.get(position_value)
        if binding is None:
            return None
        tensor_value, level, _, _ = binding
        for symbol in self.decls:
            if _stored_identity_value(symbol, SymbolId) == tensor_value:
                return symbol, level
        return None

    def _validated_bound_position_bindings(
        self,
    ) -> Dict[int, Tuple[int, int, str, Optional[MergeMode]]]:
        """Read every position binding without invoking forged equality.

        Position loads depend on the tensor/level owner and, for merged loops,
        whether a binding can be absent.  Snapshot those primitive facts when
        the target is built and re-read exact stored state at emission, so a
        forged position container or hostile ``__eq__`` cannot escape the
        target boundary or silently retarget a load.
        """

        bindings: Dict[int, Tuple[int, int, str, Optional[MergeMode]]] = {}

        def state_of(value: object, expected: type, fields: Set[str]) -> Dict[str, Any]:
            if type(value) is not expected:
                _fail(
                    "unsupported_program_shape",
                    "a position-binding loop contains a malformed node",
                )
            state = object.__getattribute__(value, "__dict__")
            if (
                type(state) is not dict
                or any(type(key) is not str for key in state)
                or set(state) != fields
            ):
                _fail(
                    "unsupported_program_shape",
                    "a position-binding loop contains malformed stored state",
                )
            if _stored_identity_value(state["node_id"], LoopIRNodeId) is None:
                _fail(
                    "unsupported_program_shape",
                    "a position-binding loop contains a malformed node identity",
                )
            return state

        def cursor_owner(cursor: object) -> Tuple[int, int]:
            state = state_of(
                cursor,
                SparseCursorDecl,
                {"node_id", "cursor", "tensor", "level", "parent"},
            )
            if _stored_identity_value(state["cursor"], CursorId) is None:
                _fail(
                    "unsupported_program_shape",
                    "a position-binding loop has a malformed cursor identity",
                )
            tensor_value = _stored_identity_value(state["tensor"], SymbolId)
            level = state["level"]
            if tensor_value is None or type(level) is not int or level < 0:
                _fail(
                    "unsupported_program_shape",
                    "a position-binding loop has a malformed cursor owner",
                )
            return tensor_value, level

        def record(
            bound: object,
            cursor: object,
            kind: str,
            mode: Optional[MergeMode],
        ) -> None:
            position_value = _stored_identity_value(bound, PositionId)
            if position_value is None:
                _fail(
                    "unsupported_program_shape",
                    "a position-binding loop must bind exact position identities",
                )
            if position_value in bindings:
                _fail(
                    "unsupported_program_shape",
                    "a position identity may be bound by only one loop cursor",
                )
            tensor_value, level = cursor_owner(cursor)
            bindings[position_value] = (tensor_value, level, kind, mode)

        for loop in self.loops:
            if loop.kind is _SPARSE:
                state = state_of(
                    loop.node,
                    SparseFor,
                    {"node_id", "cursor", "position", "coord_index", "body"},
                )
                if len(loop.cursors) != 1 or state["cursor"] is not loop.cursors[0]:
                    _fail(
                        "unsupported_program_shape",
                        "a sparse loop's cursor changed after nest collection",
                    )
                record(state["position"], loop.cursors[0], _SPARSE, None)
                continue
            if loop.kind is _SPARSE_WINDOW:
                state = state_of(
                    loop.node,
                    SparseWindowFor,
                    {
                        "node_id",
                        "tile",
                        "cursor",
                        "position",
                        "coord_index",
                        "body",
                    },
                )
                if len(loop.cursors) != 1 or state["cursor"] is not loop.cursors[0]:
                    _fail(
                        "unsupported_program_shape",
                        "a sparse-window loop's cursor changed after nest collection",
                    )
                record(state["position"], loop.cursors[0], _SPARSE_WINDOW, None)
                continue
            if loop.kind is not _MERGED:
                continue
            state = state_of(
                loop.node,
                MergedSparseFor,
                {"node_id", "mode", "cursors", "coord_index", "body", "positions"},
            )
            mode = state["mode"]
            if mode is not MergeMode.UNION and mode is not MergeMode.INTERSECTION:
                _fail(
                    "unsupported_program_shape",
                    "a merged position-binding loop has a malformed merge mode",
                )
            cursors = state["cursors"]
            positions = state["positions"]
            if (
                type(cursors) is not tuple
                or len(cursors) != len(loop.cursors)
                or any(
                    actual is not retained
                    for actual, retained in zip(cursors, loop.cursors)
                )
                or type(positions) is not tuple
                or (positions and len(positions) != len(loop.cursors))
            ):
                _fail(
                    "unsupported_program_shape",
                    "a merged loop's position bindings changed after nest collection",
                )
            for cursor, bound in zip(loop.cursors, positions):
                if bound is not None:
                    record(bound, cursor, _MERGED, mode)
        return bindings

    def _validated_position_load_spine(
        self,
        load: PositionLoad,
        *,
        require_unconditional: bool,
    ) -> Tuple[
        SymbolId,
        Tuple[Tuple[int, IndexId], ...],
        Tuple[int, Tuple[Tuple[int, int], ...], int],
    ]:
        """Revalidate one physical load spine at the target boundary.

        Verification happens before target construction, but frozen LoopIR
        nodes remain forgeable through ``object.__setattr__``.  Read exact
        stored fields, bound the walk by the tensor rank, and re-check the
        tensor/level/coordinate linkage so a post-verification cycle or
        cross-owner substitution fails closed instead of hanging or emitting
        an access through another tensor's position.
        """

        _TargetLowering._require_program_inputs_unchanged(self)

        def state_of(value: object, expected: type, fields: Set[str]) -> Dict[str, Any]:
            if type(value) is not expected:
                _fail(
                    "unsupported_program_shape",
                    "a position-load spine contains a malformed node",
                )
            stored = object.__getattribute__(value, "__dict__")
            if (
                type(stored) is not dict
                or any(type(key) is not str for key in stored)
                or set(stored) != fields
            ):
                _fail(
                    "unsupported_program_shape",
                    "a position-load spine contains malformed stored state",
                )
            if _stored_identity_value(stored["node_id"], LoopIRNodeId) is None:
                _fail(
                    "unsupported_program_shape",
                    "a position-load spine contains a malformed node identity",
                )
            return stored

        load_state = state_of(load, PositionLoad, {"node_id", "tensor", "position"})
        tensor = load_state["tensor"]
        tensor_value = _stored_identity_value(tensor, SymbolId)
        if (
            tensor_value is None
            or tensor not in self.decls
            or tensor_value not in self._program_input_values
        ):
            _fail(
                "unsupported_program_shape",
                "a position load must name one declared input tensor",
            )
        decl = self.decls[tensor]
        level = len(decl.levels) - 1
        expr = load_state["position"]
        seen: Set[int] = set()
        drivers: List[Tuple[int, IndexId]] = []
        driver_signature: List[Tuple[int, int]] = []

        while type(expr) is DensePosition:
            if id(expr) in seen or len(seen) >= len(decl.levels):
                _fail(
                    "unsupported_program_shape",
                    "a position-load spine must be finite and acyclic",
                )
            seen.add(id(expr))
            dense = state_of(
                expr,
                DensePosition,
                {"node_id", "tensor", "level", "parent", "coord"},
            )
            if (
                _stored_identity_value(dense["tensor"], SymbolId) != tensor_value
                or type(dense["level"]) is not int
                or dense["level"] != level
                or level < 0
                or decl.levels[level].kind is not LevelKind.DENSE
            ):
                _fail(
                    "unsupported_program_shape",
                    f"the position load of {decl.name!r} must descend its own "
                    "dense levels in storage order",
                )
            coordinate = state_of(dense["coord"], IndexValue, {"node_id", "index"})[
                "index"
            ]
            coordinate_value = _stored_identity_value(coordinate, IndexId)
            loop_dimensions = {
                dimension_value
                for loop in self.loops
                if _stored_identity_value(loop.index, IndexId) == coordinate_value
                for dimension_value in (
                    _stored_identity_value(loop.dimension, DimensionId),
                )
                if dimension_value is not None
            }
            if coordinate_value is None or len(loop_dimensions) != 1:
                _fail(
                    "unsupported_program_shape",
                    f"the level-{level} position coordinate of {decl.name!r} "
                    "must be a directly bound loop coordinate",
                )
            levels = decl.levels
            dimensions = decl.dimensions
            mode = levels[level].mode if 0 <= level < len(levels) else None
            expected_dimension = (
                _stored_identity_value(dimensions[mode], DimensionId)
                if type(levels) is tuple
                and type(dimensions) is tuple
                and type(mode) is int
                and 0 <= mode < len(dimensions)
                else None
            )
            if expected_dimension is None or loop_dimensions != {expected_dimension}:
                _fail(
                    "unsupported_program_shape",
                    f"the level-{level} position coordinate of {decl.name!r} "
                    "must belong to that level's logical dimension",
                )
            drivers.append((level, coordinate))
            driver_signature.append((level, coordinate_value))
            expr = dense["parent"]
            level -= 1

        position = state_of(expr, PositionValue, {"node_id", "position"})["position"]
        position_value = _stored_identity_value(position, PositionId)
        if position_value is None:
            _fail(
                "unsupported_program_shape",
                "a position load must ground at an exact bound position",
            )
        owner = self._bound_position_owner(position)
        if owner != (tensor, level):
            _fail(
                "unsupported_program_shape",
                f"the position load of {decl.name!r} must ground at the "
                f"bound position of its own level-{level} cursor",
            )
        if require_unconditional:
            binding = self._validated_bound_position_bindings().get(position_value)
            if binding is None or (
                binding[2] == _MERGED and binding[3] is MergeMode.UNION
            ):
                _fail(
                    "unsupported_program_shape",
                    "a merged-case position load must ground at an "
                    "unconditionally bound position; a UNION-bound position "
                    "carries no value at a one-sided coordinate",
                )
        return (
            tensor,
            tuple(drivers),
            (tensor_value, tuple(driver_signature), position_value),
        )

    def _compute_level_drivers(self) -> Dict[SymbolId, Dict[int, object]]:
        """Map every tensor's physical level to the loop index driving it."""

        drivers: Dict[SymbolId, Dict[int, object]] = {}

        def record(symbol: SymbolId, level: int, index: object, what: str) -> None:
            per_tensor = drivers.setdefault(symbol, {})
            known = per_tensor.get(level)
            if known is not None and known != index:
                _fail(
                    "unsupported_program_shape",
                    f"{what} drives level {level} of "
                    f"{self.decls[symbol].name!r} with conflicting loop "
                    "coordinates",
                )
            if index not in self.loop_positions:
                _fail(
                    "unsupported_program_shape",
                    "access coordinates must be nest loop variables",
                )
            per_tensor[level] = index

        def record_parent_chain(cursor: SparseCursorDecl) -> None:
            parent: Expr = cursor.parent
            level = cursor.level - 1
            while True:
                if type(parent) is RootPosition:
                    if level != -1:
                        _fail(
                            "unsupported_program_shape",
                            "a cursor parent chain must ground at the root "
                            "below level 0",
                        )
                    return
                if type(parent) is PositionValue:
                    # A compressed parent level: the chain grounds at the
                    # bound position of this tensor's own parent-level
                    # cursor (single or merged), whose loop already records
                    # that level's driver.
                    owner = self._bound_position_owner(
                        cast(PositionValue, parent).position
                    )
                    if owner != (cursor.tensor, level):
                        _fail(
                            "unsupported_program_shape",
                            "cursor parent chains must ground at the bound "
                            "position of the same tensor's parent level",
                        )
                    return
                if type(parent) is DensePosition:
                    if parent.tensor != cursor.tensor or parent.level != level:
                        _fail(
                            "unsupported_program_shape",
                            "cursor parent chains must walk this tensor's "
                            "levels in order",
                        )
                    coord = self._index_of(
                        parent.coord,
                        f"the level-{level} parent coordinate of "
                        f"{self.decls[cursor.tensor].name!r}",
                    )
                    record(
                        cursor.tensor,
                        level,
                        coord,
                        "a cursor parent chain",
                    )
                    parent = parent.parent
                    level -= 1
                    continue
                _fail(
                    "unsupported_program_shape",
                    "cursor parents must be dense-position chains over the "
                    "root in the migrated families",
                )

        def record_position_load(load: PositionLoad) -> None:
            """One dense spine descending the loaded tensor's own levels.

            The spine walks from the value-bearing leaf upward through
            contiguous DENSE levels and must ground at the bound position of
            this tensor's own single-cursor sparse loop; every spine level's
            coordinate drives that physical level exactly like a coordinate
            load's logical index does.
            """

            tensor, drivers, signature = self._validated_position_load_spine(
                load, require_unconditional=False
            )
            expected_signature = self._position_load_signatures.get(id(load))
            if expected_signature is None or signature != expected_signature:
                _fail(
                    "unsupported_program_shape",
                    "a position load changed after target construction",
                )
            for level, coord in drivers:
                record(tensor, level, coord, "a position-load spine")

        for load in self.loads:
            # Load indices are logical mode order by contract; level
            # ``level`` is driven by the logical coordinate it stores.
            decl = self.decls[load.tensor]
            for level, level_decl in enumerate(decl.levels):
                index = load.indices[level_decl.mode]
                bound = self._index_of(index, f"access of {decl.name!r}")
                record(load.tensor, level, bound, "a coordinate load")
        for position_load in self.position_loads:
            record_position_load(position_load)
        for position, loop in enumerate(self.loops):
            for cursor in loop.cursors:
                record(cursor.tensor, cursor.level, loop.index, "a sparse loop")
                record_parent_chain(cursor)
        for level, index in enumerate(self._result_storage_indices()):
            bound = self._index_of(index, f"access of {self.result_decl.name!r}")
            record(self.result_symbol, level, bound, "the result access")
        return drivers

    def _leaf_indices(self) -> Tuple[Expr, ...]:
        if type(self.leaf) is AppendEntry:
            return self.leaf.coords
        return self.leaf.indices  # type: ignore[attr-defined]

    def _result_storage_indices(self) -> Tuple[Expr, ...]:
        """Return the result access coordinates in physical level order."""

        logical = self._leaf_indices()
        return tuple(logical[level.mode] for level in self.result_decl.levels)

    def _validate_access_orders(self) -> None:
        for symbol, per_tensor in self.level_drivers.items():
            decl = self.decls[symbol]
            if sorted(per_tensor) != list(range(len(decl.levels))):
                _fail(
                    "unsupported_program_shape",
                    f"tensor {decl.name!r} has levels with no driving loop "
                    "coordinate",
                )
            positions = [
                self.loop_positions[per_tensor[level]]
                for level in range(len(decl.levels))
            ]
            if positions != sorted(positions) or len(set(positions)) != len(positions):
                _fail(
                    "unsupported_loop_order",
                    f"tensor {decl.name!r} storage order "
                    "conflicts with the loop nest order",
                )
        leaf_indices = self._leaf_indices()
        leaf_index_ids = [
            self._index_of(index, "the result access") for index in leaf_indices
        ]
        if type(self.leaf) is AppendEntry:
            nest_ids = [loop.index for loop in self.loops]
            if leaf_index_ids != nest_ids:
                _fail(
                    "unsupported_program_shape",
                    "ordered sparse assembly requires the nest loops to be "
                    "exactly the appended coordinates, in order",
                )
        elif type(self.leaf) is Store and any(
            loop.kind is _MERGED for loop in self.loops
        ):
            if set(leaf_index_ids) != {loop.index for loop in self.loops}:
                _fail(
                    "unsupported_program_shape",
                    "a merged store must write every nest coordinate",
                )
        # Inputs not read by the leaf value cannot anchor a position chain.
        read_symbols = {load.tensor for load in self.loads} | {
            self._cursor_decl(cursor_value.cursor).tensor
            for cursor_value in self.cursor_values
        }
        for symbol in self._input_symbols:
            if symbol not in read_symbols and self.decls[symbol].levels:
                if symbol not in self.level_drivers:
                    _fail(
                        "unsupported_program_shape",
                        f"input {self.decls[symbol].name!r} is never read",
                    )

    def _validate_loop_variable_names(self) -> None:
        """Reject distinct coordinates that the target would spell alike.

        LoopIR dimensions identify coordinate domains, not loop binders: two
        distinct logical indices may legally iterate the same dimension.
        This C++ target currently derives a loop variable's spelling from the
        dimension display name, however, so lowering both indices would
        silently shadow one coordinate with the other.  The outer/inner nodes
        of one affine split remain legal because they share the same logical
        ``IndexId`` after normalization through ``_loop_logical_index``.
        """

        index_by_dimension: Dict[DimensionId, object] = {}
        for loop in self.loops:
            logical_index = self._loop_logical_index(loop)
            known = index_by_dimension.get(loop.dimension)
            if known is None:
                index_by_dimension[loop.dimension] = logical_index
                continue
            if known != logical_index:
                name = self.dimension_names[loop.dimension]
                _fail(
                    "generated_name_collision",
                    f"distinct logical loop coordinates share dimension "
                    f"{name!r}; this target would spell both C++ variables "
                    f"as {name!r}",
                )

    # -- emission ------------------------------------------------------------

    def _loop_var_name(self, loop: _Loop) -> str:
        return self.dimension_names[loop.dimension]

    def _access_id(self, symbol: SymbolId) -> AccessId:
        known = self._access_ids.get(symbol)
        if known is None:
            known = new_access_id()
            self._access_ids[symbol] = known
        return known

    def _input_metadata(self, symbol: SymbolId) -> llir.TensorAccessMetadata:
        decl = self.decls[symbol]
        per_tensor = self.level_drivers[symbol]
        logical_indices: List[Optional[IndexId]] = [None] * len(decl.levels)
        for level, level_decl in enumerate(decl.levels):
            logical_indices[level_decl.mode] = cast(IndexId, per_tensor[level])
        if any(index_id is None for index_id in logical_indices):
            _fail(
                "unsupported_program_shape",
                f"tensor {decl.name!r} has incomplete logical access metadata",
            )
        return _detach_tensor_access_metadata(
            llir.TensorAccessMetadata(
                access_id=self._access_id(symbol),
                tensor_id=symbol,
                index_ids=cast(Tuple[IndexId, ...], tuple(logical_indices)),
                role=llir.TensorAccessRole.INPUT_READ,
            )
        )

    def _result_metadata(self) -> llir.TensorAccessMetadata:
        return _detach_tensor_access_metadata(
            llir.TensorAccessMetadata(
                access_id=self._access_id(self.result_symbol),
                tensor_id=self.result_symbol,
                index_ids=tuple(
                    self._index_of(index, "store index")  # type: ignore[misc]
                    for index in self._leaf_indices()
                ),
                role=llir.TensorAccessRole.RESULT_WRITE,
            ),
        )

    def _loop_logical_index(self, loop: _Loop) -> object:
        if loop.kind in (_TILE_OUTER, _TILE_INNER, _PANEL_OUTER):
            return loop.node.index
        return loop.index

    def _input_bound_var(self, loop: _Loop) -> Optional[llir.Var]:
        lookup = self._loop_logical_index(loop)
        for symbol in self._input_symbols:
            decl = self.decls[symbol]
            per_tensor = self.level_drivers.get(symbol, {})
            for level in range(len(decl.levels)):
                if (
                    per_tensor.get(level) == lookup
                    and decl.levels[level].kind is LevelKind.DENSE
                ):
                    return llir.Var(
                        name=f"{decl.name}{level}_size", type=llir.DataType.INT64
                    )
        return None

    def _loop_bound_var(self, loop: _Loop) -> llir.Var:
        bound = self._input_bound_var(loop)
        if bound is not None:
            return bound
        lookup = self._loop_logical_index(loop)
        for level, index in enumerate(self._result_storage_indices()):
            if self._index_of(index, "store index") == lookup:
                return llir.Var(
                    name=f"{self.result_decl.name}{level}_size",
                    type=llir.DataType.INT64,
                )
        _fail("unsupported_program_shape", "a loop variable is never used")
        raise AssertionError("unreachable")

    def _tile_bound_var(self, loop: _Loop) -> llir.Var:
        """The dimension-size spelling that bounds one affine split.

        The legacy lattice resolves this bound from the first dense access
        containing the split variable — inputs first, and the result access
        for a broadcast coordinate the result alone drives.  That is exactly
        the shared loop-bound policy, so the origin loop's bound, the point
        loop's overshoot guard, and the derived parallel trip count all use
        one spelling.
        """

        return self._loop_bound_var(loop)

    def _position_init(
        self,
        tensor_name: str,
        level: int,
        loop: _Loop,
        result_chain: bool = False,
    ) -> llir.VarInit:
        loop_var = llir.Var(name=self._loop_var_name(loop), type=llir.DataType.INT64)
        if level == 0:
            value: llir.Expr = loop_var
        elif result_chain and self.sparse_program:
            # The legacy sparse lowering builds result position chains with
            # plain binary nodes; input chains keep the Add/Mul forms.  The
            # rendered C++ is identical either way — this mirrors the legacy
            # raw statements exactly so every managed pass sees the same
            # tree shapes it sees on the legacy path.
            value = llir.BinOp(
                op="+",
                left=llir.BinOp(
                    op="*",
                    left=llir.Var(
                        name=f"p{tensor_name}{level - 1}",
                        type=llir.DataType.INT,
                    ),
                    right=llir.Var(
                        name=f"{tensor_name}{level}_size",
                        type=llir.DataType.INT,
                    ),
                ),
                right=llir.Var(name=self._loop_var_name(loop), type=llir.DataType.INT),
            )
        else:
            value = llir.Add(
                left=llir.Mul(
                    left=llir.Var(
                        name=f"p{tensor_name}{level - 1}",
                        type=llir.DataType.INT64,
                    ),
                    right=llir.Var(
                        name=f"{tensor_name}{level}_size",
                        type=llir.DataType.INT64,
                    ),
                ),
                right=loop_var,
            )
        return llir.VarInit(
            var=llir.Var(name=f"p{tensor_name}{level}", type=llir.DataType.INT),
            value=value,
        )

    def _input_resolves_at(self, loop: _Loop) -> List[llir.Stmt]:
        stmts: List[llir.Stmt] = []
        for symbol in self._input_symbols:
            decl = self.decls[symbol]
            per_tensor = self.level_drivers.get(symbol, {})
            for level in range(len(decl.levels)):
                if per_tensor.get(level) != loop.index:
                    continue
                if decl.levels[level].kind is not LevelKind.DENSE:
                    continue
                stmts.append(self._position_init(decl.name, level, loop))
        return stmts

    def _result_resolves_at(self, loop: _Loop) -> List[llir.Stmt]:
        stmts: List[llir.Stmt] = []
        if (
            self.region is not None
            and self.loop_positions.get(loop.index, -1) >= self.region_start
        ):
            # Result positions driven by the split coordinate belong to the
            # synthesized consumer loop, never to the producer chain —
            # exactly where the legacy consumer lowering resolves them.
            return stmts
        for level, index in enumerate(self._result_storage_indices()):
            if self._index_of(index, "store index") != loop.index:
                continue
            if self.result_decl.levels[level].kind is not LevelKind.DENSE:
                continue
            stmts.append(
                self._position_init(
                    self.result_decl.name, level, loop, result_chain=True
                )
            )
        return stmts

    def _lower_value(self, expr: Expr) -> llir.Expr:
        if type(expr) is StagedRead:
            view = self._staged_views.get(id(expr))
            if view is None:
                _fail(
                    "unsupported_program_shape",
                    "a StagedRead outside the validated staging region is "
                    "not lowerable",
                )
            lowered = self._lower_value(view)
            if type(lowered) is not llir.ArrayAccess:
                _fail(
                    "unsupported_program_shape",
                    "the staged operand must lower to one direct array access",
                )
            if self._staged_access_snapshot is not None:
                _fail(
                    "unsupported_program_shape",
                    "the staged operand was lowered more than once",
                )
            snapshot = LLIRRewriter(
                LLIRTraversalContext(
                    stage="LoopIR target lowering",
                    pass_name="snapshot_relayout_operand_access",
                )
            ).rewrite(lowered)
            if type(snapshot) is not llir.ArrayAccess:
                _fail(
                    "unsupported_program_shape",
                    "the staged operand snapshot did not remain an array access",
                )
            metadata = snapshot.tensor_access
            if type(metadata) is not llir.TensorAccessMetadata:
                _fail(
                    "unsupported_program_shape",
                    "the staged operand snapshot did not retain typed access "
                    "metadata",
                )
            # LLIRRewriter detaches nodes, but intentionally preserves the
            # frozen provenance value object.  Rebuild every identity as a
            # separate value so an in-place forged metadata mutation after
            # the managed passes cannot also mutate this review boundary's
            # supposedly detached snapshot.
            detached_metadata = _detach_tensor_access_metadata(metadata)
            self._staged_access_snapshot = llir.ArrayAccess(
                array=snapshot.array,
                index=snapshot.index,
                tensor_access=detached_metadata,
            )
            return lowered
        if type(expr) is Load:
            decl = self.decls[expr.tensor]
            if len(expr.indices) == 1:
                bound = self._index_of(expr.indices[0], "load index")
                position = self.loops[self.loop_positions[bound]]
                physical_index = self._loop_var_name(position)
            else:
                physical_index = f"p{decl.name}{len(expr.indices) - 1}"
            torch_dtype = _SCALAR_TO_TORCH[decl.dtype]
            return llir.ArrayAccess(
                array=llir.Var(
                    name=f"{decl.name}_val",
                    type=llir.DataType.ptr_type(torch_dtype),
                ),
                index=llir.Var(name=physical_index, type=llir.DataType.INT),
                tensor_access=self._input_metadata(expr.tensor),
            )
        if type(expr) is PositionLoad:
            # The value-bearing leaf position variable is the resolved end
            # of the load's validated dense spine (``p{name}{leaf}``), or
            # the cursor position itself when the spine is empty.
            decl = self.decls[expr.tensor]
            torch_dtype = _SCALAR_TO_TORCH[decl.dtype]
            return llir.ArrayAccess(
                array=llir.Var(
                    name=f"{decl.name}_val",
                    type=llir.DataType.ptr_type(torch_dtype),
                ),
                index=llir.Var(
                    name=f"p{decl.name}{len(decl.levels) - 1}",
                    type=llir.DataType.INT,
                ),
                tensor_access=self._input_metadata(expr.tensor),
            )
        if type(expr) is CursorValue:
            cursor = self._cursor_decl(expr.cursor)
            loop = self.loops[self.cursor_loops[expr.cursor]]
            if loop.kind not in (_SPARSE, _SPARSE_WINDOW):
                _fail(
                    "unsupported_program_shape",
                    "merged cursor values are lowered per alignment case, "
                    "not as direct value expressions",
                )
            decl = self.decls[cursor.tensor]
            torch_dtype = _SCALAR_TO_TORCH[decl.dtype]
            return llir.ArrayAccess(
                array=llir.Var(
                    name=f"{decl.name}_val",
                    type=llir.DataType.ptr_type(torch_dtype),
                ),
                index=llir.Var(
                    name=f"p{decl.name}{cursor.level}", type=llir.DataType.INT
                ),
                tensor_access=self._input_metadata(cursor.tensor),
            )
        if type(expr) is BinaryExpr:
            return llir.BinOp(
                op=_BINARY_TO_CXX[expr.op],
                left=self._lower_value(expr.lhs),
                right=self._lower_value(expr.rhs),
            )
        _fail(
            "unsupported_program_shape",
            f"unsupported value expression {type(expr).__name__}",
        )
        raise AssertionError("unreachable")

    def _lower_leaf(self) -> List[llir.Stmt]:
        if self.region is not None:
            # The producer chain's leaf: ``wksp[k_in] += <value>;`` exactly
            # as the legacy dense-workspace TensorAssign lowering emits it
            # (untyped array symbol, no access metadata).
            assert self.producer_leaf is not None
            assert self.consumer_point is not None
            name = self.dimension_names[self.consumer_point.dimension]
            return [
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=self.region.workspace.name,
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=llir.Var(
                            name=f"{name}_in",
                            type=llir.DataType.INT64,
                        ),
                    ),
                    value=self._lower_value(self.producer_leaf.value),
                    op=llir.AssignOp.ADD_ASSIGN,
                )
            ]
        leaf = self.leaf
        leaf_indices = leaf.indices  # type: ignore[attr-defined]
        rhs = self._lower_value(leaf.value)  # type: ignore[attr-defined]
        target = llir.ArrayAccess(
            array=llir.Var(
                name=f"{self.result_decl.name}_values",
                type=llir.DataType.NO_TYPE,
            ),
            index=llir.Var(
                name=f"p{self.result_decl.name}{len(leaf_indices) - 1}",
                type=llir.DataType.INT64,
            ),
            tensor_access=self._result_metadata(),
        )
        if self._tiled_leaf is not None:
            if self._tiled_write_snapshot is not None:
                _fail(
                    "unsupported_program_shape",
                    "the compacted result write was lowered more than once",
                )
            snapshot = LLIRRewriter(
                LLIRTraversalContext(
                    stage="LoopIR target lowering",
                    pass_name="snapshot_result_tile_write",
                )
            ).rewrite(target)
            if type(snapshot) is not llir.ArrayAccess:
                _fail(
                    "unsupported_program_shape",
                    "the compacted result snapshot did not remain an array " "access",
                )
            metadata = snapshot.tensor_access
            if type(metadata) is not llir.TensorAccessMetadata:
                _fail(
                    "unsupported_program_shape",
                    "the compacted result snapshot did not retain typed "
                    "access metadata",
                )
            # LLIRRewriter detaches nodes but intentionally preserves the
            # frozen provenance value object; rebuild every identity as a
            # separate value so an in-place forged metadata mutation after
            # the managed passes cannot also mutate this detached snapshot
            # (the reviewed relayout boundary's discipline).
            detached_metadata = _detach_tensor_access_metadata(metadata)
            self._tiled_write_snapshot = llir.ArrayAccess(
                array=snapshot.array,
                index=snapshot.index,
                tensor_access=detached_metadata,
            )
        if type(leaf) is StoreReduce:
            return [llir.Assign(var=target, value=rhs, op=llir.AssignOp.ADD_ASSIGN)]
        return [llir.Assign(var=target, value=rhs)]

    # -- merged-loop case machinery -------------------------------------------

    def _require_unconditional_position_load(self, load: PositionLoad) -> None:
        """Refuse a merged-case position load over an optional position.

        :class:`PositionLoad` deliberately carries no merge-alignment or
        default semantics -- the node contract reserves those for
        :class:`CursorValue` -- so partially evaluating one for a
        cursor-alignment case is sound only when the position it grounds at
        is bound unconditionally.  ``SparseFor``/``SparseWindowFor`` bindings
        and INTERSECTION merges always bind; a UNION merge's binding is
        optional at a one-sided coordinate, where the loaded tensor owns no
        value-bearing position at all.  The verifier's position typing
        already refuses to type a UNION-bound position for a position-load
        spine; this is the owning target's independent check.
        """

        tensor, drivers, signature = self._validated_position_load_spine(
            load, require_unconditional=True
        )
        if self._position_load_signatures.get(id(load)) != signature:
            _fail(
                "unsupported_program_shape",
                "a position load changed after target construction",
            )
        expected = self.level_drivers.get(tensor, {})
        if any(expected.get(level) != coordinate for level, coordinate in drivers):
            _fail(
                "unsupported_program_shape",
                "a position-load spine changed after target construction",
            )

    def _merged_case_value(
        self, expr: Expr, aligned: Set[CursorId]
    ) -> Optional[llir.Expr]:
        """Partially evaluate the leaf value for one cursor-alignment case.

        Returns ``None`` when the case value folds to the additive identity
        (nothing is emitted for that case, exactly as the legacy iteration
        lattice drops it).

        A :class:`PositionLoad` reads through the loaded tensor's own
        validated dense spine rather than through a merge cursor, so it is
        case-invariant -- but only when its grounding position is bound
        unconditionally, which
        :meth:`_require_unconditional_position_load` enforces.
        """

        if type(expr) is CursorValue:
            cursor = self._cursor_decl(expr.cursor)
            if expr.cursor in aligned:
                decl = self.decls[cursor.tensor]
                torch_dtype = _SCALAR_TO_TORCH[decl.dtype]
                return llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{decl.name}_val",
                        type=llir.DataType.ptr_type(torch_dtype),
                    ),
                    index=llir.Var(
                        name=f"p{decl.name}{cursor.level}",
                        type=llir.DataType.INT,
                    ),
                    tensor_access=self._input_metadata(cursor.tensor),
                )
            default = expr.default
            if type(default) is not FloatConst or default.value != 0.0:
                _fail(
                    "unsupported_union_default",
                    "the migrated UNION families require the additive "
                    "identity 0.0 as the unaligned-cursor default",
                )
            return None
        if type(expr) is Load:
            return self._lower_value(expr)
        if type(expr) is PositionLoad:
            self._require_unconditional_position_load(cast(PositionLoad, expr))
            return self._lower_value(expr)
        if type(expr) is BinaryExpr:
            left = self._merged_case_value(expr.lhs, aligned)
            right = self._merged_case_value(expr.rhs, aligned)
            if expr.op is BinaryOp.ADD:
                if left is None:
                    return right
                if right is None:
                    return left
                return llir.BinOp(op="+", left=left, right=right)
            if expr.op is BinaryOp.MUL:
                if left is None or right is None:
                    return None
                return llir.BinOp(op="*", left=left, right=right)
            _fail(
                "unsupported_program_shape",
                "subtraction over merged sparse operands is outside the "
                "migrated families",
            )
        _fail(
            "unsupported_program_shape",
            f"unsupported merged value expression {type(expr).__name__}",
        )
        raise AssertionError("unreachable")

    def _merged_case_stmts(
        self, loop: _Loop, aligned: Set[CursorId]
    ) -> Optional[List[llir.Stmt]]:
        value = self._merged_case_value(
            self.leaf.value, aligned  # type: ignore[attr-defined]
        )
        if value is None:
            return None
        dimension_name = self._loop_var_name(loop)
        result_name = self.result_decl.name
        leaf_level = len(self.result_decl.levels) - 1
        target = llir.ArrayAccess(
            array=llir.Var(
                name=f"{result_name}_values",
                type=llir.DataType.NO_TYPE,
            ),
            index=llir.Var(
                name=f"p{result_name}{leaf_level}",
                type=llir.DataType.INT64,
            ),
            tensor_access=self._result_metadata(),
        )
        stmts: List[llir.Stmt] = [llir.Assign(var=target, value=value)]
        if type(self.leaf) is AppendEntry:
            stmts.append(llir.Comment("Set coordinates"))
            stmts.append(
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_name}{leaf_level}_crd",
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=llir.Var(
                            name=f"p{result_name}{leaf_level}",
                            type=llir.DataType.INT64,
                        ),
                    ),
                    value=llir.Var(name=dimension_name, type=llir.DataType.NO_TYPE),
                )
            )
            stmts.append(
                llir.Increment(
                    var=llir.Var(
                        name=f"p{result_name}{leaf_level}",
                        type=llir.DataType.INT64,
                    )
                )
            )
        return stmts

    def _merged_coordinate_name(self, loop: _Loop, cursor: SparseCursorDecl) -> str:
        return f"{self._loop_var_name(loop)}_{self.decls[cursor.tensor].name}"

    def _cursor_position_name(self, cursor: SparseCursorDecl) -> str:
        return f"p{self.decls[cursor.tensor].name}{cursor.level}"

    def _cursor_pos_array(self, cursor: SparseCursorDecl) -> llir.Var:
        name = self.decls[cursor.tensor].name
        return llir.Var(name=f"{name}{cursor.level}_pos", type=llir.DataType.PTR_INT)

    def _cursor_crd_array(self, cursor: SparseCursorDecl) -> llir.Var:
        name = self.decls[cursor.tensor].name
        return llir.Var(name=f"{name}{cursor.level}_crd", type=llir.DataType.PTR_INT)

    def _cursor_parent_var(self, cursor: SparseCursorDecl) -> llir.Var:
        name = self.decls[cursor.tensor].name
        return llir.Var(name=f"p{name}{cursor.level - 1}", type=llir.DataType.INT)

    def _cursor_parent_index(self, cursor: SparseCursorDecl, offset: int) -> llir.Expr:
        """The pos-array subscript at one cursor's parent position.

        A level-0 cursor is dominated by the physical root position, which
        the legacy lowering folds to the exact integer subscript (``pos[0]``
        and ``pos[1]``); deeper levels keep the parent position variable.
        """

        if cursor.level == 0:
            return llir.Literal(offset, llir.DataType.INT)
        parent = self._cursor_parent_var(cursor)
        if offset == 0:
            return parent
        return llir.Add(left=parent, right=llir.Literal(offset, llir.DataType.INT))

    def _iterator_inits(self, loop: _Loop) -> List[llir.Stmt]:
        """The legacy ``Initialize iterators`` group for one sparse loop."""

        stmts: List[llir.Stmt] = []
        if loop.kind in (_SPARSE, _SPARSE_WINDOW):
            cursor = loop.cursors[0]
            end_init = llir.VarInit(
                var=llir.Var(
                    name=f"{self._cursor_position_name(cursor)}_end",
                    type=llir.DataType.INT,
                ),
                value=llir.ArrayAccess(
                    array=self._cursor_pos_array(cursor),
                    index=self._cursor_parent_index(cursor, 1),
                ),
            )
            if loop.kind is _SPARSE_WINDOW:
                # The panel completion rewires exactly this declaration on
                # the assembled function.  Keep a detached pre-pass snapshot
                # so an in-place or detaching pass cannot redefine the
                # iterator range before structural completion.
                self._window_end_snapshot = cast(
                    llir.VarInit,
                    LLIRRewriter(
                        LLIRTraversalContext(
                            stage="LoopIR target lowering",
                            pass_name="snapshot_panel_window_end",
                        )
                    ).rewrite(end_init),
                )
            stmts.append(end_init)
            return stmts
        for cursor in loop.cursors:
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=self._cursor_position_name(cursor),
                        type=llir.DataType.INT,
                    ),
                    value=llir.ArrayAccess(
                        array=self._cursor_pos_array(cursor),
                        index=self._cursor_parent_index(cursor, 0),
                    ),
                )
            )
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"{self._cursor_position_name(cursor)}_end",
                        type=llir.DataType.INT,
                    ),
                    value=llir.ArrayAccess(
                        array=self._cursor_pos_array(cursor),
                        index=self._cursor_parent_index(cursor, 1),
                    ),
                )
            )
        return stmts

    def _result_pos_set(self, level: Optional[int] = None) -> llir.Assign:
        """``C1_pos[C1_pos_index + 1] = C1_crd.size()`` (legacy spelling)."""

        result_name = self.result_decl.name
        if level is None:
            level = len(self.result_decl.levels) - 1
        return llir.Assign(
            var=llir.ArrayAccess(
                array=llir.Var(
                    name=f"{result_name}{level}_pos",
                    type=llir.DataType.STD_VECTOR_C_INT,
                ),
                index=llir.Add(
                    llir.Var(
                        name=f"{result_name}{level}_pos_index",
                        type=llir.DataType.INT,
                    ),
                    llir.Literal(1, llir.DataType.INT32),
                ),
            ),
            value=llir.FunctionCall(
                name=f"{result_name}{level}_crd.size",
                args=[],
            ),
        )

    def _assembly_result_pos_set(self, level: Optional[int] = None) -> llir.Stmt:
        """One result-position close in the form this target owns.

        The generic target deliberately emits the indexed assignment the
        production dynamic-vector pass consumes.  Dedicated assembly targets
        may override this narrow builder when they own the exact checked form
        before that generic pass.
        """

        return self._result_pos_set(level)

    def _dense_assembly_close_level(self, position: int) -> int:
        """The result level a dense assembly loop's catch-up closes."""

        return len(self.result_decl.levels) - 1

    def _assembly_catch_up(
        self, loop: _Loop, level: Optional[int] = None
    ) -> llir.ForLoop:
        """The legacy per-row ``Assemble COMPRESSED level`` catch-up loop."""

        result_name = self.result_decl.name
        if level is None:
            level = len(self.result_decl.levels) - 1
        pos_index = llir.Var(
            name=f"{result_name}{level}_pos_index",
            type=llir.DataType.INT,
        )
        return llir.ForLoop(
            init=None,
            cond=llir.BinOp(
                op="<",
                left=pos_index,
                right=llir.Var(name=self._loop_var_name(loop), type=llir.DataType.INT),
            ),
            update=llir.Increment(
                var=llir.Var(
                    name=f"{result_name}{level}_pos_index",
                    type=llir.DataType.INT,
                )
            ),
            body=[self._assembly_result_pos_set(level)],
        )

    def _lower_merged(self, position: int) -> List[llir.Stmt]:
        loop = self.loops[position]
        node = loop.node
        dimension_name = self._loop_var_name(loop)
        cursors = loop.cursors
        result_resolves = self._result_resolves_at(loop)

        def coordinate_loads() -> List[llir.Stmt]:
            stmts: List[llir.Stmt] = [llir.Comment("Load coordinates")]
            for cursor in cursors:
                stmts.append(
                    llir.VarInit(
                        var=llir.Var(
                            name=self._merged_coordinate_name(loop, cursor),
                            type=llir.DataType.INT,
                        ),
                        value=llir.ArrayAccess(
                            array=self._cursor_crd_array(cursor),
                            index=llir.Var(
                                name=self._cursor_position_name(cursor),
                                type=llir.DataType.INT,
                            ),
                        ),
                    )
                )
            return stmts

        def coordinate_var(cursor: SparseCursorDecl) -> llir.Var:
            return llir.Var(
                name=self._merged_coordinate_name(loop, cursor),
                type=llir.DataType.INT,
            )

        def dimension_var() -> llir.Var:
            return llir.Var(name=dimension_name, type=llir.DataType.INT)

        def aligned_guard(cursor: SparseCursorDecl) -> llir.Expr:
            return llir.BinOp(
                op="==", left=coordinate_var(cursor), right=dimension_var()
            )

        # Alignment cases in the legacy lattice order: all cursors aligned
        # first, then each single-cursor case for UNION merges.
        cases: List[Tuple[Set[CursorId], llir.Expr]] = []
        both_guard = llir.BinOp(
            op="&&",
            left=aligned_guard(cursors[0]),
            right=aligned_guard(cursors[1]),
        )
        cases.append(({cursor.cursor for cursor in cursors}, both_guard))
        if node.mode is MergeMode.UNION:
            for cursor in cursors:
                cases.append(({cursor.cursor}, aligned_guard(cursor)))

        cond_list: List[llir.Expr] = []
        then_body_list: List[List[llir.Stmt]] = []
        for aligned, guard in cases:
            case_stmts = self._merged_case_stmts(loop, aligned)
            if case_stmts is None:
                continue
            cond_list.append(guard)
            then_body_list.append(case_stmts)
        if not cond_list:
            _fail(
                "unsupported_program_shape",
                "a merged loop must emit at least one alignment case",
            )

        while_body: List[llir.Stmt] = [
            *coordinate_loads(),
            llir.BlankLine(),
            llir.Comment("Resolve coordinates"),
            llir.VarInit(
                var=llir.Var(name=dimension_name, type=llir.DataType.INT),
                value=llir.FunctionCall(
                    name="std::min",
                    args=[
                        llir.Array(
                            values=tuple(coordinate_var(cursor) for cursor in cursors),
                            data_type=llir.DataType.INT,
                        )
                    ],
                ),
            ),
            llir.BlankLine(),
        ]
        input_resolves = self._input_resolves_at(loop)
        if input_resolves:
            while_body.append(llir.Comment("Resolve dense coordinates"))
            while_body.extend(input_resolves)
        if result_resolves:
            while_body.append(
                llir.Comment("Resolve index into dense level of values array")
            )
            while_body.extend(result_resolves)
        while_body.append(llir.Comment("Inner loops over child regions"))
        while_body.append(
            llir.IfThenElse(
                cond_list=cond_list,
                then_body_list=then_body_list,
                make_last_case_else=False,
            )
        )
        while_body.append(llir.BlankLine())
        while_body.append(llir.Comment("Advance iterators"))
        for cursor in cursors:
            while_body.append(
                llir.Assign(
                    var=llir.Var(
                        name=self._cursor_position_name(cursor),
                        type=llir.DataType.INT,
                    ),
                    value=llir.BinOp(
                        op="==",
                        left=coordinate_var(cursor),
                        right=dimension_var(),
                    ),
                    op=llir.AssignOp.ADD_ASSIGN,
                    cast=True,
                )
            )

        def position_cond(cursor: SparseCursorDecl) -> llir.Expr:
            return llir.BinOp(
                op="<",
                left=llir.Var(
                    name=self._cursor_position_name(cursor),
                    type=llir.DataType.INT,
                ),
                right=llir.Var(
                    name=f"{self._cursor_position_name(cursor)}_end",
                    type=llir.DataType.INT,
                ),
            )

        merge_loop = llir.WhileLoop(
            cond=llir.BinOp(
                op="&&",
                left=position_cond(cursors[0]),
                right=position_cond(cursors[1]),
            ),
            body=while_body,
        )
        merge_loop.scorch_index_var = dimension_name

        stmts: List[llir.Stmt] = [
            llir.Comment("Initialize iterators"),
            *self._iterator_inits(loop),
            llir.BlankLine(),
            merge_loop,
        ]

        if node.mode is MergeMode.UNION:
            for cursor in cursors:
                tail_case = self._merged_case_stmts(loop, {cursor.cursor})
                if tail_case is None:
                    continue
                tail_body: List[llir.Stmt] = [
                    llir.Comment("Resolve coordinates"),
                    llir.VarInit(
                        var=llir.Var(name=dimension_name, type=llir.DataType.INT),
                        value=llir.ArrayAccess(
                            array=self._cursor_crd_array(cursor),
                            index=llir.Var(
                                name=self._cursor_position_name(cursor),
                                type=llir.DataType.INT,
                            ),
                        ),
                    ),
                    llir.BlankLine(),
                ]
                tail_input_resolves = self._input_resolves_at(loop)
                if tail_input_resolves:
                    tail_body.append(llir.Comment("Resolve dense coordinates"))
                    tail_body.extend(tail_input_resolves)
                tail_resolves = self._result_resolves_at(loop)
                if tail_resolves:
                    tail_body.append(
                        llir.Comment("Resolve index into dense level of values array")
                    )
                    tail_body.extend(tail_resolves)
                tail_body.extend(tail_case)
                tail_body.append(llir.Comment("Advance iterator"))
                tail_body.append(
                    llir.Increment(
                        var=llir.Var(
                            name=self._cursor_position_name(cursor),
                            type=llir.DataType.INT,
                        )
                    )
                )
                stmts.append(llir.WhileLoop(cond=position_cond(cursor), body=tail_body))
        return stmts

    def _lower_sparse(self, position: int) -> List[llir.Stmt]:
        loop = self.loops[position]
        cursor = loop.cursors[0]
        dimension_name = self._loop_var_name(loop)
        position_name = self._cursor_position_name(cursor)

        body: List[llir.Stmt] = [
            llir.Comment("Resolve coordinates"),
            llir.VarInit(
                var=llir.Var(name=dimension_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_crd_array(cursor),
                    index=llir.Var(name=position_name, type=llir.DataType.INT),
                ),
            ),
            llir.BlankLine(),
        ]
        if loop.kind is _SPARSE_WINDOW and self.relayout is not None:
            # The relayout completion inserts the compatibility range guard
            # directly after exactly this declaration on the assembled
            # function.  Snapshot its complete local skeleton so an
            # unchanged declaration moved after a dependent statement is
            # rejected instead of being accepted by content alone.
            context_snapshot = LLIRRewriter(
                LLIRTraversalContext(
                    stage="LoopIR target lowering",
                    pass_name="snapshot_relayout_window_coord",
                )
            ).rewrite(tuple(body))
            if (
                type(context_snapshot) is not tuple
                or len(context_snapshot) != 3
                or type(context_snapshot[0]) is not llir.Comment
                or type(context_snapshot[1]) is not llir.VarInit
                or type(context_snapshot[2]) is not llir.BlankLine
            ):
                _fail(
                    "unsupported_program_shape",
                    "the staged relayout coordinate context could not be "
                    "snapshotted",
                )
            self._window_coord_snapshot = cast(
                Tuple[llir.Comment, llir.VarInit, llir.BlankLine],
                context_snapshot,
            )
        input_resolves = self._input_resolves_at(loop)
        result_resolves = self._result_resolves_at(loop)
        if input_resolves:
            body.append(llir.Comment("Resolve dense coordinates"))
            body.extend(input_resolves)
        if result_resolves:
            body.append(llir.Comment("Resolve index into dense level of values array"))
            body.extend(result_resolves)
        body.extend(self._loop_children(position))

        position_var = llir.Var(name=position_name, type=llir.DataType.INT)
        for_loop = llir.ForLoop(
            init=llir.VarInit(
                var=llir.Var(name=position_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_pos_array(cursor),
                    index=self._cursor_parent_index(cursor, 0),
                ),
            ),
            cond=llir.BinOp(
                op="<",
                left=position_var,
                right=llir.Var(name=f"{position_name}_end", type=llir.DataType.INT),
            ),
            update=llir.Increment(
                var=llir.Var(name=position_name, type=llir.DataType.INT)
            ),
            body=body,
        )
        for_loop.scorch_index_var = dimension_name
        self._record_emitted_loop(position, for_loop)
        return [
            llir.Comment("Initialize iterators"),
            *self._iterator_inits(loop),
            llir.BlankLine(),
            for_loop,
        ]

    def _loop_children(self, position: int) -> List[llir.Stmt]:
        """Child-loop or leaf statements appended inside one loop's body."""

        if self.region is not None and position + 1 == self.region_start:
            return self._lower_workspace_region(position)
        if position + 1 < len(self.loops):
            child = self.loops[position + 1]
            if child.kind is _PANEL_OUTER:
                # The panel origin is applied to the assembled function by
                # complete_panel, exactly where the legacy schedule
                # lowering wraps it; the raw statements stay unpaneled so
                # every managed pass sees the same trees it sees on the
                # legacy path.
                return self._loop_children(position + 1)
            if child.kind is _DENSE:
                return [llir.BlankLine(), self._lower_dense(position + 1)]
            if child.kind is _TILE_OUTER:
                return [llir.BlankLine(), self._lower_tile_outer(position + 1)]
            if child.kind is _TILE_INNER:
                return [llir.BlankLine(), self._lower_tile_inner(position + 1)]
            if child.kind in (_SPARSE, _SPARSE_WINDOW):
                return self._lower_sparse(position + 1)
            return self._lower_merged(position + 1)
        return self._lower_leaf()

    def _lower_workspace_region(self, position: int) -> List[llir.Stmt]:
        """The legacy ``lower_Where`` stack shape at the region's position.

        ``// Initialize workspaces`` + the tile-width stack array declaration
        (allocation and zero-reset in one statement — the region's intrinsic
        entry semantics), then the producer chain with the workspace
        reduction as its leaf, then the synthesized consumer copy-out loop.
        """

        assert self.region is not None
        assert self.consumer_point is not None
        decl = self.region.workspace
        name = self.dimension_names[self.consumer_point.dimension]
        torch_dtype = _SCALAR_TO_TORCH[decl.dtype]
        element_type = dtype_to_c_datatype(torch_dtype)
        stmts: List[llir.Stmt] = [
            llir.Comment("Initialize workspaces"),
            llir.FixedStackArrayDecl(
                name=decl.name,
                element_type=element_type,
                extent=llir.Var(
                    name=f"kTile_{name}",
                    type=llir.DataType.CONSTEXPR_INT,
                ),
                initializer=llir.Array(values=[], data_type=element_type),
            ),
        ]
        producer = self.loops[position + 1]
        if producer.kind is _DENSE:
            stmts.extend([llir.BlankLine(), self._lower_dense(position + 1)])
        elif producer.kind is _SPARSE:
            stmts.extend(self._lower_sparse(position + 1))
        elif producer.kind is _TILE_INNER:
            # The reduce-out family opens its producer with the strip-mined
            # reduction point loop; its origin loop is on the outer chain.
            stmts.extend([llir.BlankLine(), self._lower_tile_inner(position + 1)])
        else:
            _fail(
                "unsupported_program_shape",
                "a workspace producer must open with a dense, single-cursor "
                "sparse, or strip-mined reduction loop",
            )
        stmts.extend(self._lower_workspace_consumer())
        return stmts

    def _lower_workspace_consumer(self) -> List[llir.Stmt]:
        """The synthesized consumer loop, exactly as legacy emits it.

        ``for (int64_t k_in = 0; k_in < kTile_k; k_in++)`` resolving the
        logical coordinate, breaking past the *result* bound, resolving the
        result position with the consumer's own int64 spelling, and
        ADD-assigning the workspace cell into the result values.
        """

        assert self.region is not None
        assert self.consumer_point is not None
        region = self.region
        point = self.consumer_point
        name = self.dimension_names[point.dimension]
        result_name = self.result_decl.name
        levels = [
            level
            for level, index in enumerate(self._result_storage_indices())
            if self._index_of(index, "store index") == point.index
        ]
        if len(levels) != 1 or levels[0] == 0:
            _fail(
                "unsupported_program_shape",
                "the workspace copy-out coordinate must drive exactly one "
                "trailing result level",
            )
        level = levels[0]
        torch_dtype = _SCALAR_TO_TORCH[region.workspace.dtype]
        loop_var = llir.Var(name=f"{name}_in", type=llir.DataType.INT64)
        body: List[llir.Stmt] = [
            llir.VarInit(
                var=llir.Var(name=name, type=llir.DataType.INT64),
                value=llir.Add(
                    left=llir.Var(name=f"{name}_out", type=llir.DataType.INT64),
                    right=llir.Var(name=f"{name}_in", type=llir.DataType.INT64),
                ),
            ),
            llir.IfThenElse(
                cond=llir.BinOp(
                    op=">=",
                    left=llir.Var(name=name, type=llir.DataType.INT),
                    right=llir.Var(
                        name=f"{result_name}{level}_size",
                        type=llir.DataType.INT,
                    ),
                ),
                then_body=[llir.Break()],
            ),
            llir.VarInit(
                var=llir.Var(
                    name=f"p{result_name}{level}",
                    type=llir.DataType.INT64,
                ),
                value=llir.Add(
                    left=llir.Mul(
                        left=llir.Var(
                            name=f"p{result_name}{level - 1}",
                            type=llir.DataType.INT64,
                        ),
                        right=llir.Var(
                            name=f"{result_name}{level}_size",
                            type=llir.DataType.INT64,
                        ),
                    ),
                    right=llir.Var(name=name, type=llir.DataType.INT64),
                ),
            ),
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{result_name}_values",
                        type=llir.DataType.NO_TYPE,
                    ),
                    index=llir.Var(
                        name=f"p{result_name}{level}",
                        type=llir.DataType.INT64,
                    ),
                    tensor_access=self._result_metadata(),
                ),
                value=llir.ArrayAccess(
                    array=llir.Var(
                        name=region.workspace.name,
                        type=llir.DataType.ptr_type(torch_dtype),
                    ),
                    index=llir.Var(
                        name=loop_var.name,
                        type=loop_var.type,
                    ),
                ),
                op=llir.AssignOp.ADD_ASSIGN,
            ),
        ]
        for_loop = llir.ForLoop(
            init=llir.VarInit(var=loop_var, value=llir.Literal(0)),
            cond=llir.BinOp(
                op="<",
                left=loop_var,
                right=llir.Var(
                    name=f"kTile_{name}",
                    type=llir.DataType.INT64,
                ),
            ),
            update=llir.Increment(var=loop_var),
            body=body,
            unroll=point.unroll,
        )
        return [
            llir.BlankLine(),
            llir.Comment("Lower consumer CIN"),
            for_loop,
        ]

    def _lower_dense(self, position: int) -> llir.ForLoop:
        loop = self.loops[position]
        name = self._loop_var_name(loop)
        result_is_csr_row = (
            not self.result_is_dense
            and type(self.leaf) is AppendEntry
            and self._index_of(self.leaf.coords[0], "the appended row coordinate")
            == loop.index
        )
        input_resolves = self._input_resolves_at(loop)
        result_resolves = self._result_resolves_at(loop)
        loop_drives_an_input = any(
            per_tensor.get(level) == loop.index
            for symbol, per_tensor in self.level_drivers.items()
            if symbol != self.result_symbol
            for level in per_tensor
        )
        body: List[llir.Stmt] = []
        if result_is_csr_row:
            body.append(llir.Comment("Assemble COMPRESSED level"))
            body.append(
                self._assembly_catch_up(
                    loop, self._dense_assembly_close_level(position)
                )
            )
        body.append(llir.Comment("Resolve dense coordinates"))
        body.extend(input_resolves)
        if not loop_drives_an_input:
            # A broadcast loop iterates only the result; its position chain
            # is the loop's driving dense iterator, exactly as the legacy
            # dense lattice emits it.
            body.extend(result_resolves)
        elif result_resolves:
            body.append(llir.Comment("Resolve index into dense level of values array"))
            body.extend(result_resolves)
        body.extend(self._loop_children(position))
        if result_is_csr_row:
            body.append(llir.BlankLine())
            body.append(llir.Comment("Assembly compressed _level indices"))
            body.append(
                self._assembly_result_pos_set(
                    self._dense_assembly_close_level(position)
                )
            )
        loop_var = llir.Var(name=name, type=llir.DataType.INT64)
        for_loop = llir.ForLoop(
            init=llir.VarInit(var=loop_var, value=llir.Literal(0)),
            cond=llir.BinOp(op="<", left=loop_var, right=self._loop_bound_var(loop)),
            update=llir.Increment(var=loop_var),
            body=body,
        )
        for_loop.scorch_index_var = name
        self._record_emitted_loop(position, for_loop)
        return for_loop

    def _lower_tile_outer(self, position: int) -> llir.ForLoop:
        """The origin loop of one affine split, exactly as legacy emits it.

        ``for (int64_t k_out = 0; k_out < <bound>; k_out += kTile_k)`` —
        the origin steps by the width constant, the body carries no
        coordinate resolves (the origin is unreadable), and the width
        constant is declared once in the function preamble.
        """

        loop = self.loops[position]
        name = self.dimension_names[loop.dimension]
        bound = self._tile_bound_var(loop)
        loop_var = llir.Var(name=f"{name}_out", type=llir.DataType.INT64)
        for_loop = llir.ForLoop(
            init=llir.VarInit(var=loop_var, value=llir.Literal(0)),
            cond=llir.BinOp(op="<", left=loop_var, right=bound),
            update=llir.Assign(
                var=loop_var,
                value=llir.Var(name=f"kTile_{name}", type=llir.DataType.INT),
                op=llir.AssignOp.ADD_ASSIGN,
            ),
            body=self._loop_children(position),
        )
        for_loop.scorch_index_var = f"{name}_out"
        self._record_emitted_loop(position, for_loop)
        return for_loop

    def _lower_tile_inner(self, position: int) -> llir.ForLoop:
        """The point loop of one affine split, exactly as legacy emits it.

        The body first reconstructs the logical coordinate
        (``int64_t k = k_out + k_in;``), breaks past the ragged tail
        (``if (k >= <bound>) break;``), then continues with the ordinary
        dense-body emission for the logical loop.
        """

        loop = self.loops[position]
        name = self.dimension_names[loop.dimension]
        bound = self._tile_bound_var(loop)
        input_resolves = self._input_resolves_at(loop)
        result_resolves = self._result_resolves_at(loop)
        loop_drives_an_input = any(
            per_tensor.get(level) == loop.index
            for symbol, per_tensor in self.level_drivers.items()
            if symbol != self.result_symbol
            for level in per_tensor
        )
        body: List[llir.Stmt] = [
            llir.Comment("Resolve tiled index var"),
            llir.VarInit(
                var=llir.Var(name=name, type=llir.DataType.INT64),
                value=llir.Add(
                    left=llir.Var(name=f"{name}_out", type=llir.DataType.INT64),
                    right=llir.Var(name=f"{name}_in", type=llir.DataType.INT64),
                ),
            ),
            llir.IfThenElse(
                cond=llir.BinOp(
                    op=">=",
                    left=llir.Var(name=name, type=llir.DataType.INT),
                    right=llir.Var(name=bound.name, type=llir.DataType.INT),
                ),
                then_body=[llir.Break()],
            ),
            llir.Comment("Resolve dense coordinates"),
        ]
        body.extend(input_resolves)
        if not loop_drives_an_input:
            body.extend(result_resolves)
        elif result_resolves:
            body.append(llir.Comment("Resolve index into dense level of values array"))
            body.extend(result_resolves)
        body.extend(self._loop_children(position))
        loop_var = llir.Var(name=f"{name}_in", type=llir.DataType.INT64)
        for_loop = llir.ForLoop(
            init=llir.VarInit(var=loop_var, value=llir.Literal(0)),
            cond=llir.BinOp(
                op="<",
                left=loop_var,
                right=llir.Var(name=f"kTile_{name}", type=llir.DataType.INT),
            ),
            update=llir.Increment(var=loop_var),
            body=body,
            unroll=loop.node.unroll,
        )
        for_loop.scorch_index_var = f"{name}_in"
        self._record_emitted_loop(position, for_loop)
        return for_loop

    def tile_size_inits(self) -> List[llir.Stmt]:
        """Width-constant declarations, one per split, in nest order."""

        tile_loops = [loop for loop in self.loops if loop.kind is _TILE_OUTER]
        if not tile_loops:
            return []
        stmts: List[llir.Stmt] = [
            llir.BlankLine(),
            llir.Comment("Initialize tile sizes"),
        ]
        for loop in tile_loops:
            name = self.dimension_names[loop.dimension]
            stmts.append(
                llir.VarInit(
                    var=llir.Var(
                        name=f"kTile_{name}",
                        type=llir.DataType.CONSTEXPR_INT,
                    ),
                    value=llir.Literal(loop.node.width),
                )
            )
        return stmts

    def raw_loop_statements(self) -> List[llir.Stmt]:
        """The raw pre-pass loop-nest statements, legacy shape included.

        Panel and heap programs emit the *unpaneled, uncompacted* nest with
        no parallel marking: the legacy explicit-parallel route suppresses
        the emission-time auto gate and marks the selected row loop on the
        assembled function, and the panel wrap/windowing and heap
        compaction follow it there — so all of it happens in
        :meth:`complete_panel` / :meth:`complete_result_tile`, after the
        managed passes.
        """

        _TargetLowering._require_value_expression_unchanged(self)

        first_position = 0
        while self.loops[first_position].kind is _PANEL_OUTER:
            first_position += 1
        first = self.loops[first_position]
        stmts: List[llir.Stmt]
        if first.kind is _SPARSE:
            # The root-parent row loop of the mixed dense-leaf operand
            # chain: iterator initialization replaces the pre-loop blank
            # line at function scope, exactly the legacy outer-sparse shape.
            stmts = list(self._lower_sparse(first_position))
        else:
            outer_loop = (
                self._lower_tile_outer(first_position)
                if first.kind is _TILE_OUTER
                else self._lower_dense(first_position)
            )
            stmts = [llir.BlankLine(), outer_loop]
        if self.panel is not None or self.result_tile is not None:
            return stmts
        if self.parallel is not None:
            # The legacy explicit-parallel route suppresses the
            # emission-time auto gate and marks the selected loop on the
            # assembled function; complete_parallel owns that marking.
            return stmts
        leaf_indices = self._leaf_indices()
        outer_index = self._loop_logical_index(first)
        outer_in_result = any(
            self._index_of(index, "store index") == outer_index
            for index in leaf_indices
        )
        if self.result_is_dense and outer_in_result:
            # The legacy gate: parallelize the outer loop only when the
            # result is dense and the outer coordinate addresses it.
            if any(loop.kind is _MERGED for loop in self.loops):
                # Merged nests iterate their cursors through while loops, and
                # the legacy marker runs before its nested statement lists
                # are flattened, so it cannot see the merge's position-array
                # initializers: the applied policy is the row-count-only
                # ``scorch_nthreads(-1, rows)`` form.  Byte parity requires
                # the same policy, so the position-array search is
                # deliberately given no statements to inspect.
                outer = stmts[1]
                assert type(outer) is llir.ForLoop
                outer.omp_parallel_for = True
                apply_parallel_policy(outer, body=())
            else:
                mark_first_for_loop_parallel(stmts, EMPTY_PARALLEL_WORKSPACE_CLUSTER)
        return stmts

    def kernel_abi(self) -> TorchCppKernelABI:
        return TorchCppKernelABI(
            result_shape=self.shapes[self.result_symbol],
            result_rank=len(self.result_decl.levels),
            input_tensors=tuple(
                KernelTensorABI(
                    name=self.decls[symbol].name,
                    level_types=tuple(
                        _LEVEL_KIND_TO_LEVEL_TYPE[level.kind]
                        for level in self.decls[symbol].levels
                    ),
                    mode_order=tuple(level.mode for level in self.decls[symbol].levels),
                    shape=self.shapes[symbol],
                    dtype=_SCALAR_TO_TORCH[self.decls[symbol].dtype],
                )
                for symbol in self._input_symbols
            ),
        )

    def result_assembler(self) -> ResultTensorAssembler:
        return ResultTensorAssembler(
            name=self.result_decl.name,
            level_types=tuple(
                _LEVEL_KIND_TO_LEVEL_TYPE[level.kind]
                for level in self.result_decl.levels
            ),
            dtype=_SCALAR_TO_TORCH[self.result_decl.dtype],
        )

    def result_size_inits(self) -> List[llir.Stmt]:
        stmts: List[llir.Stmt] = []
        for level, level_decl in enumerate(self.result_decl.levels):
            if level_decl.kind is not LevelKind.DENSE:
                continue
            stmts.append(
                llir.VarInit(
                    llir.Var(
                        name=f"{self.result_decl.name}{level}_size",
                        type=llir.DataType.INT64,
                    ),
                    value=llir.ArrayAccess(
                        array=llir.Var(
                            name="result_shape",
                            type=llir.DataType.STD_VECTOR_INT,
                        ),
                        index=llir.Literal(
                            value=level,
                            data_type=llir.DataType.INT64,
                        ),
                    ),
                )
            )
        return stmts

    def value_array_ctypes(self) -> Tuple[Tuple[str, str], ...]:
        return tuple(
            (
                f"{self.decls[symbol].name}_val",
                dtype_to_c_datatype(_SCALAR_TO_TORCH[self.decls[symbol].dtype]).value,
            )
            for symbol in self._input_symbols
        )

    def complete_sparse_workspace(self, function: llir.Function) -> llir.Function:
        """Validate sparse-workspace post-pass effects when this family owns them."""

        return function

    def prepare_result_level_indices(
        self, statements: List[llir.Stmt]
    ) -> List[llir.Stmt]:
        """Prepare result-index initialization owned by this target."""

        return statements

    def owns_two_phase_output(self) -> bool:
        """Whether the shared compressed-Where pass owns this result's assembly."""

        return False

    def compressed_where_pass_spec(
        self, compile_options: CompileOptions
    ) -> Optional[CompressedWhereOpenMPPassSpec]:
        """The two-phase pass configuration for families that own one."""

        return None

    # -- panel completion ----------------------------------------------------

    def _record_emitted_loop(self, position: int, loop: llir.ForLoop) -> None:
        """Snapshot one chain-loop header before managed passes can touch it."""

        if self.panel is None and self.result_tile is None and self.parallel is None:
            return
        header = llir.ForLoop(
            init=loop.init,
            cond=loop.cond,
            update=loop.update,
            body=[],
            omp_parallel_for=loop.omp_parallel_for,
            omp_schedule=loop.omp_schedule,
            unroll=loop.unroll,
            simd=loop.simd,
            pre_parallel_body=loop.pre_parallel_body,
            post_parallel_body=loop.post_parallel_body,
            omp_num_threads=loop.omp_num_threads,
            omp_chunk_expr=loop.omp_chunk_expr,
            before_parallel_body=loop.before_parallel_body,
        )
        header.scorch_index_var = loop.scorch_index_var
        snapshot = LLIRRewriter(
            LLIRTraversalContext(
                stage="LoopIR target lowering",
                pass_name="snapshot_panel_loop_header",
            )
        ).rewrite(header)
        if type(snapshot) is not llir.ForLoop:
            _fail(
                "panel_completion_lost",
                "the panel loop-header snapshot did not remain a ForLoop",
            )
        self._emitted_loop_headers[position] = snapshot

    @staticmethod
    def _nested_statement_lists(stmt: llir.Stmt) -> List[List[llir.Stmt]]:
        lists: List[List[llir.Stmt]] = []
        if isinstance(stmt, (llir.ForLoop, llir.WhileLoop, llir.ForLoopAuto)):
            state = object.__getattribute__(stmt, "__dict__")
            if type(state) is not dict or type(state.get("body")) is not list:
                _fail(
                    "panel_completion_lost",
                    "a control statement in the completed panel has malformed "
                    "stored body state",
                )
            lists.append(state["body"])
        elif isinstance(stmt, llir.IfThenElse):
            state = object.__getattribute__(stmt, "__dict__")
            required = ("then_body", "else_body", "then_body_list")
            if type(state) is not dict or any(name not in state for name in required):
                _fail(
                    "panel_completion_lost",
                    "a conditional in the completed panel has malformed "
                    "stored branch state",
                )
            for name in ("then_body", "else_body"):
                body = state[name]
                if body is not None:
                    if type(body) is not list:
                        _fail(
                            "panel_completion_lost",
                            "a conditional panel branch is not an owned list",
                        )
                    lists.append(body)
            branches = state["then_body_list"]
            if branches is not None:
                if type(branches) is not list or any(
                    type(branch) is not list for branch in branches
                ):
                    _fail(
                        "panel_completion_lost",
                        "conditional panel branches are not owned lists",
                    )
                lists.extend(branches)
        return lists

    def _locate_statement(
        self, stmts: List[llir.Stmt], target: llir.Stmt
    ) -> Optional[Tuple[List[llir.Stmt], int]]:
        """Find one statement object by identity within nested lists."""

        for index, stmt in enumerate(stmts):
            if stmt is target:
                return stmts, index
            for nested in self._nested_statement_lists(stmt):
                located = self._locate_statement(nested, target)
                if located is not None:
                    return located
        return None

    def _for_loops_with_owners(
        self, stmts: List[llir.Stmt]
    ) -> List[Tuple[llir.ForLoop, Optional[llir.Stmt]]]:
        """Collect loops with the control statement owning their body list."""

        found: List[Tuple[llir.ForLoop, Optional[llir.Stmt]]] = []
        active: Set[int] = set()
        visited: Set[int] = set()

        def enter(value: object) -> None:
            marker = id(value)
            if marker in active:
                _fail(
                    "panel_completion_lost",
                    "the assembled panel loop structure is cyclic",
                )
            if marker in visited:
                _fail(
                    "panel_completion_lost",
                    "the assembled panel loop structure shares statement " "ownership",
                )
            active.add(marker)
            visited.add(marker)

        def walk(
            statements: List[llir.Stmt],
            owner: Optional[llir.Stmt],
            depth: int,
        ) -> None:
            if depth > 256:
                _fail(
                    "panel_completion_lost",
                    "the assembled panel loop structure is excessively deep",
                )
            if type(statements) is not list:
                _fail(
                    "panel_completion_lost",
                    "the assembled panel loop structure is not list-owned",
                )
            enter(statements)
            try:
                for stmt in statements:
                    if type(stmt) not in SUPPORTED_LLIR_STATEMENT_NODE_TYPES:
                        _fail(
                            "panel_completion_lost",
                            "the assembled panel contains an unknown or "
                            "non-statement member",
                        )
                    enter(stmt)
                    try:
                        if isinstance(stmt, llir.ForLoop):
                            found.append((stmt, owner))
                        for nested in self._nested_statement_lists(stmt):
                            walk(nested, stmt, depth + 1)
                    finally:
                        active.discard(id(stmt))
            finally:
                active.discard(id(statements))

        walk(stmts, None, 0)
        return found

    @classmethod
    def _panel_loop_header_matches(
        cls, actual: llir.ForLoop, expected: llir.ForLoop
    ) -> bool:
        """Whether a post-pass loop retains its complete pre-pass header."""

        if type(actual) is not llir.ForLoop or type(expected) is not llir.ForLoop:
            return False
        field_names = (
            "init",
            "cond",
            "update",
            "omp_parallel_for",
            "omp_schedule",
            "unroll",
            "simd",
            "omp_num_threads",
            "omp_chunk_expr",
            "before_parallel_body",
            "pre_parallel_body",
            "post_parallel_body",
            "scorch_index_var",
        )
        compatibility_fields = (
            "_use_atomic_scheduling",
            "_atomic_chunk_var",
            "_atomic_counter_var",
            "_loop_bound",
        )
        actual_state = object.__getattribute__(actual, "__dict__")
        expected_state = object.__getattribute__(expected, "__dict__")
        actual_names = cls._exact_state_field_names(actual_state)
        expected_names = cls._exact_state_field_names(expected_state)
        if actual_names is None or expected_names is None:
            return False
        if any(
            field_name not in actual_names or field_name not in expected_names
            for field_name in (*field_names, "body")
        ):
            return False
        if type(actual_state["body"]) is not list:
            return False
        for field_name in field_names:
            actual_value = actual_state[field_name]
            expected_value = expected_state[field_name]
            if not cls._exact_panel_state_matches(actual_value, expected_value):
                return False
        for field_name in compatibility_fields:
            if (field_name in actual_names) != (field_name in expected_names):
                return False
            if field_name in actual_names:
                actual_value = actual_state[field_name]
                expected_value = expected_state[field_name]
                if not cls._exact_panel_state_matches(actual_value, expected_value):
                    return False
        return True

    @staticmethod
    def _exact_state_field_names(value: object) -> Optional[Tuple[str, ...]]:
        """Return sorted exact-string keys without invoking forged equality."""

        if type(value) is not dict:
            return None
        names = tuple(value)
        if any(type(name) is not str for name in names):
            return None
        return tuple(sorted(cast(Tuple[str, ...], names)))

    @classmethod
    def _panel_var_state_is_valid(cls, value: object) -> bool:
        """Whether an exact Var owns complete, type-correct stored state."""

        if type(value) is not llir.Var:
            return False
        state = object.__getattribute__(value, "__dict__")
        required = ("name", "type", "is_ptr", "is_restrict", "tensor_access")
        state_names = cls._exact_state_field_names(state)
        if state_names is None or any(name not in state_names for name in required):
            return False
        if (
            type(state["name"]) is not str
            or not state["name"].isidentifier()
            or type(state["type"]) is not llir.DataType
            or type(state["is_ptr"]) is not bool
            or type(state["is_restrict"]) is not bool
        ):
            return False
        metadata = state["tensor_access"]
        if metadata is None:
            return True
        if type(metadata) is not llir.TensorAccessMetadata:
            return False
        metadata_state = object.__getattribute__(metadata, "__dict__")
        if type(metadata_state) is not dict:
            return False
        index_ids = metadata_state.get("index_ids")
        return (
            type(metadata_state.get("access_id")) is AccessId
            and type(metadata_state.get("tensor_id")) is SymbolId
            and type(index_ids) is tuple
            and all(type(index_id) is IndexId for index_id in index_ids)
            and type(metadata_state.get("role")) is llir.TensorAccessRole
        )

    @classmethod
    def _exact_panel_state_matches(
        cls,
        actual: object,
        expected: object,
        active: Optional[Set[Tuple[int, int]]] = None,
        depth: int = 0,
    ) -> bool:
        """Compare a captured LLIR subtree without invoking forged ``__eq__``."""

        if type(actual) is not type(expected) or depth > 256:
            return False
        if actual is None:
            return True
        if type(actual) in (bool, int, float, str, llir.DataType):
            return actual == expected
        if isinstance(actual, Enum):
            return actual is expected
        if type(actual) in (AccessId, SymbolId, IndexId):
            actual_state = object.__getattribute__(actual, "__dict__")
            expected_state = object.__getattribute__(expected, "__dict__")
            actual_names = cls._exact_state_field_names(actual_state)
            expected_names = cls._exact_state_field_names(expected_state)
            return bool(
                actual_names == ("value",)
                and expected_names == ("value",)
                and type(actual_state["value"]) is int
                and type(expected_state["value"]) is int
                and actual_state["value"] == expected_state["value"]
            )
        if type(actual) is llir.TensorAccessMetadata:
            actual_state = object.__getattribute__(actual, "__dict__")
            expected_state = object.__getattribute__(expected, "__dict__")
            field_names = ("access_id", "tensor_id", "index_ids", "role")
            expected_field_names = tuple(sorted(field_names))
            if (
                cls._exact_state_field_names(actual_state) != expected_field_names
                or cls._exact_state_field_names(expected_state) != expected_field_names
            ):
                return False
            if active is None:
                active = set()
            pair = (id(actual), id(expected))
            if pair in active:
                return False
            active.add(pair)
            try:
                return all(
                    cls._exact_panel_state_matches(
                        actual_state[field_name],
                        expected_state[field_name],
                        active,
                        depth + 1,
                    )
                    for field_name in field_names
                )
            finally:
                active.remove(pair)
        if isinstance(actual, llir.Node):
            actual_state = object.__getattribute__(actual, "__dict__")
            expected_state = object.__getattribute__(expected, "__dict__")
            actual_names = cls._exact_state_field_names(actual_state)
            expected_names = cls._exact_state_field_names(expected_state)
            if (
                actual_names is None
                or expected_names is None
                or actual_names != expected_names
            ):
                return False
            if active is None:
                active = set()
            pair = (id(actual), id(expected))
            if pair in active:
                return False
            active.add(pair)
            try:
                return all(
                    cls._exact_panel_state_matches(
                        actual_state[name],
                        expected_state[name],
                        active,
                        depth + 1,
                    )
                    for name in actual_names
                )
            finally:
                active.remove(pair)
        if type(actual) in (list, tuple):
            actual_sequence = cast(Any, actual)
            expected_sequence = cast(Any, expected)
            if len(actual_sequence) != len(expected_sequence):
                return False
            if active is None:
                active = set()
            pair = (id(actual), id(expected))
            if pair in active:
                return False
            active.add(pair)
            try:
                return all(
                    cls._exact_panel_state_matches(
                        actual_item,
                        expected_item,
                        active,
                        depth + 1,
                    )
                    for actual_item, expected_item in zip(
                        actual_sequence, expected_sequence
                    )
                )
            finally:
                active.remove(pair)
        return False

    def _completed_panel_chain(
        self, function: llir.Function
    ) -> Dict[int, llir.ForLoop]:
        """Re-identify the exact direct chain represented before the passes."""

        expected_positions = tuple(
            position
            for position, loop in enumerate(self.loops)
            if loop.kind is not _PANEL_OUTER
        )
        found = self._for_loops_with_owners(function.body)
        if len(found) != len(expected_positions):
            _fail(
                "panel_completion_lost",
                "the assembled function does not carry exactly the emitted "
                "chain loops; the structural completion refuses to guess",
            )
        completed: Dict[int, llir.ForLoop] = {}
        prior: Optional[llir.ForLoop] = None
        for position, (actual, owner) in zip(expected_positions, found):
            snapshot = self._emitted_loop_headers.get(position)
            if (
                snapshot is None
                or owner is not prior
                or not self._panel_loop_header_matches(actual, snapshot)
            ):
                _fail(
                    "panel_completion_lost",
                    "the assembled function's direct loop chain disagrees "
                    "with the detached pre-pass headers",
                )
            completed[position] = actual
            prior = actual
        if set(self._emitted_loop_headers) != set(expected_positions):
            _fail(
                "panel_completion_lost",
                "the panel's emitted loop-header snapshots are incomplete",
            )
        return completed

    def complete_panel(self, function: llir.Function) -> llir.Function:
        """Mark, window, and wrap the panel on the assembled function.

        This runs at exactly the legacy pipeline position: the managed
        passes transformed the unpaneled, unmarked statements, the
        function is assembled, and only then does the legacy route mark
        the explicit parallel row loop
        (``CINLowerer._apply_explicit_parallel_schedule``), rewrite the
        compressed loop's bounds to the coordinate window, wrap the panel
        origin loop, and prepend the width constant
        (``schedule_lowerer._apply_panel_tile``).  Rewrite targets are
        located by chain position over the preserved loop skeleton and
        cross-checked against the retained emission records; any
        disagreement fails closed instead of guessing.
        """

        if self.panel is None:
            return function
        end_snapshot = self._window_end_snapshot
        if end_snapshot is None:
            _fail(
                "panel_completion_lost",
                "the panel's emitted statements were never recorded",
            )
        completed = self._completed_panel_chain(function)
        window_loop = completed[self.window_position]
        row_loop = completed[self.panel_row_position]
        wrapped_loop = completed[self.panel_position + 1]
        located_window = self._locate_statement(function.body, window_loop)
        located_wrap = self._locate_statement(function.body, wrapped_loop)
        if located_window is None or located_wrap is None:
            _fail(
                "panel_completion_lost",
                "the verified panel chain cannot be located in the function",
            )
        window_container, window_index = located_window
        end_candidates: List[Tuple[int, llir.VarInit]] = []
        for index, stmt in enumerate(window_container):
            if type(stmt) is not llir.VarInit:
                continue
            state = object.__getattribute__(stmt, "__dict__")
            required = ("var", "value", "op", "cast")
            if (
                type(state) is not dict
                or any(field_name not in state for field_name in required)
                or not self._panel_var_state_is_valid(state.get("var"))
                or not isinstance(state.get("value"), llir.Expr)
                or type(state.get("op")) is not str
                or type(state.get("cast")) is not bool
            ):
                _fail(
                    "panel_completion_lost",
                    "a variable initialization beside the panel window has "
                    "malformed stored state",
                )
            if self._exact_panel_state_matches(state["var"], end_snapshot.var):
                end_candidates.append((index, stmt))
        if (
            len(end_candidates) != 1
            or not self._exact_panel_state_matches(end_candidates[0][1], end_snapshot)
            or end_candidates[0][0] >= window_index
            or any(
                type(statement) is not llir.BlankLine
                for statement in window_container[
                    end_candidates[0][0] + 1 : window_index
                ]
            )
        ):
            _fail(
                "panel_completion_lost",
                "the window's iterator end declaration does not exactly "
                "precede the loop with its detached pre-pass state",
            )
        end_index, located_end_init = end_candidates[0]
        end_container = window_container
        located_end = (end_container, end_index)
        end_init = located_end_init

        # 1. Explicit parallel marking of the row loop, exactly as the
        # legacy explicit-parallel schedule applies it post-assembly (the
        # panel family carries no CIN workspace, so the cluster is empty).
        expected_marked = LLIRRewriter(
            LLIRTraversalContext(
                stage="LoopIR target lowering",
                pass_name="snapshot_panel_parallel_policy",
            )
        ).rewrite(row_loop)
        if type(expected_marked) is not llir.ForLoop:
            _fail(
                "panel_completion_lost",
                "the panel row-loop policy snapshot did not remain a ForLoop",
            )
        if self.parallel is None:
            expected_marked.omp_parallel_for = True
            expected_marked.omp_schedule = "dynamic, 64"
            apply_parallel_policy(expected_marked)
            mark_first_for_loop_parallel(
                [row_loop],
                EMPTY_PARALLEL_WORKSPACE_CLUSTER,
            )
        else:
            policy_spec = self._parallel_work_policy_spec(
                "panel_completion_lost",
            )
            self._apply_expected_parallel_marker(expected_marked)
            mark_first_for_loop_parallel(
                [row_loop],
                EMPTY_PARALLEL_WORKSPACE_CLUSTER,
            )
            if not self._panel_loop_header_matches(row_loop, expected_marked):
                _fail(
                    "panel_completion_lost",
                    "the panel's selected row loop did not acquire the "
                    "unmodified legacy parallel marker state",
                )
            self._apply_selected_parallel_policy(
                expected_marked,
                policy_spec,
                "panel_completion_lost",
                invoke_marker=False,
            )
            self._apply_selected_parallel_policy(
                row_loop,
                policy_spec,
                "panel_completion_lost",
                invoke_marker=False,
            )
        if not self._panel_loop_header_matches(row_loop, expected_marked):
            _fail(
                "panel_completion_lost",
                "the panel's selected row loop did not acquire the required "
                "sparse parallel policy",
            )
        self._panel_parallel_snapshot = expected_marked

        # 2. Window the compressed loop: capture the row end, derive the
        # search-based panel begin/end, and start the loop at the window.
        cursor = self.loops[self.window_position].cursors[0]
        position_name = self._cursor_position_name(cursor)
        dimension_name = self.dimension_names[self.panel.dimension]
        panel_var = f"{dimension_name}_out"
        panel_end_var = f"{panel_var}_end"
        row_end = f"{position_name}_row_end"
        panel_begin = f"{position_name}_panel_begin"
        row_end_value = llir.Var(row_end, end_init.var.type)
        panel_begin_value = llir.Var(panel_begin, end_init.var.type)
        row_begin = llir.ArrayAccess(
            array=self._cursor_pos_array(cursor),
            index=self._cursor_parent_var(cursor),
        )
        lower_value = _panel_bound_expression(
            self._cursor_crd_array(cursor),
            row_begin,
            row_end_value,
            llir.Var(panel_var, llir.DataType.INT64),
        )
        upper_value = _panel_bound_expression(
            self._cursor_crd_array(cursor),
            panel_begin_value,
            row_end_value,
            llir.Var(panel_end_var, llir.DataType.INT64),
        )
        end_container, end_index = located_end
        end_container[end_index : end_index + 1] = [
            llir.VarInit(
                var=llir.Var(row_end, end_init.var.type),
                value=end_init.value,
            ),
            llir.VarInit(
                var=llir.Var(panel_begin, end_init.var.type),
                value=lower_value,
            ),
            llir.VarInit(var=end_init.var, value=upper_value),
        ]
        if type(window_loop.init) is not llir.VarInit:
            _fail(
                "panel_completion_lost",
                "the verified panel window no longer has its iterator " "initializer",
            )
        window_loop.init.value = llir.Var(panel_begin, end_init.var.type)

        # 3. Wrap the loop below the panel's chain position in the origin
        # loop, with the clamped window end declared at its head.
        bound_name = (
            f"{self.decls[self.panel.bound_tensor].name}"
            f"{self.panel.bound_level}_size"
        )
        tile_var = f"kTile_{dimension_name}"
        panel_loop = llir.ForLoop(
            init=llir.VarInit(
                var=llir.Var(panel_var, llir.DataType.INT64),
                value=llir.Literal(0),
            ),
            cond=llir.BinOp(
                op="<",
                left=llir.Var(panel_var, llir.DataType.INT64),
                right=llir.Var(bound_name, llir.DataType.INT64),
            ),
            update=llir.Assign(
                var=llir.Var(panel_var, llir.DataType.INT64),
                value=llir.Var(tile_var, llir.DataType.INT64),
                op=llir.AssignOp.ADD_ASSIGN,
            ),
            body=[
                llir.VarInit(
                    var=llir.Var(panel_end_var, llir.DataType.INT64),
                    value=llir.FunctionCall(
                        name="std::min",
                        args=[
                            llir.Add(
                                llir.Var(panel_var, llir.DataType.INT64),
                                llir.Var(tile_var, llir.DataType.INT64),
                            ),
                            llir.Var(bound_name, llir.DataType.INT64),
                        ],
                    ),
                ),
                wrapped_loop,
            ],
        )
        panel_loop.scorch_index_var = panel_var
        wrap_container, wrap_index = located_wrap
        wrap_container[wrap_index] = panel_loop

        # 4. The width constant at the top of the function body, exactly
        # where the legacy pass prepends it.
        function.body[0:0] = [
            llir.Comment(f"Initialize {dimension_name} panel tile size"),
            llir.VarInit(
                var=llir.Var(tile_var, llir.DataType.CONSTEXPR_INT),
                value=llir.Literal(self.panel.width),
            ),
            llir.BlankLine(),
        ]
        if self.relayout is not None or self.result_tile is not None:
            # The heap and relayout completions run next on this same
            # assembled function and consume the loops this completion has
            # already re-identified and created — retained object identity,
            # never a second discovery pass.  The validated heap/relayout
            # chains place the pack origin at chain position 0.
            self._panel_completion = (
                completed[0],
                panel_loop,
                row_loop,
                window_loop,
            )
        return function

    def complete_relayout(self, function: llir.Function) -> llir.Function:
        """Stage the packed operand on the assembled function.

        A no-op without a validated staging region; otherwise the full
        legacy relayout completion, driven by retained completion objects
        and typed emission spellings (see ``_complete_relayout_impl``).
        """

        if self.relayout is None:
            return function
        return _complete_relayout_impl(self, function)

    def complete_result_tile(self, function: llir.Function) -> llir.Function:
        """Compact the heap result tile on the assembled function.

        A no-op without a validated accumulation region; otherwise the full
        legacy heap completion — parallel row marking for the bare chain,
        the exact-one compact redirection, zero-fill removal, per-strip
        init and copy-out groups, and the reusable storage block — between
        the panel and relayout completions, exactly the legacy
        ``apply_schedule_to_llir`` order (see
        ``_complete_result_tile_impl``).
        """

        if self.result_tile is None:
            return function
        return _complete_result_tile_impl(self, function)

    def complete_parallel(self, function: llir.Function) -> llir.Function:
        """Mark the abstract parallel selection on the assembled function.

        Runs first among the completions — exactly where the legacy route
        applies ``CINLowerer._apply_explicit_parallel_schedule`` before its
        schedule lowering — for direct and stack routes.  Panel (including
        relayout) and heap completions own their corresponding marking after
        assembly.  The complete selected-loop ancestor chain is
        re-identified against detached pre-pass headers, the program and
        primitive selection snapshot are revalidated, and the shared legacy
        marker is checked independently before the exact owned work policy is
        realized.  Exactly one final marking may survive; any disagreement is
        the stage-owned ``parallel_completion_lost``.
        """

        if (
            self.parallel is None
            or self.panel is not None
            or self.result_tile is not None
        ):
            return function
        policy_spec = self._parallel_work_policy_spec(_PARALLEL_LOST)
        try:
            found = self._for_loops_with_owners(function.body)
        except LoopIRTargetError as error:
            _fail(_PARALLEL_LOST, error.defect.message)
        if any(
            loop.omp_parallel_for or getattr(loop, "_use_atomic_scheduling", False)
            for loop, _owner in found
        ):
            _fail(
                _PARALLEL_LOST,
                "an assembled loop acquired a parallel marking before the "
                "explicit selection completion ran",
            )

        # Re-identify the complete ancestor prefix, not merely a matching
        # header anywhere in the function.  Direct chains own every prefix
        # loop; stack chains branch only below the selected race-free prefix.
        # Each header must be unique and each loop must be owned by its
        # immediate predecessor, so relocating the selected loop cannot
        # preserve a superficially matching header.
        prior: Optional[llir.ForLoop] = None
        selected: Optional[llir.ForLoop] = None
        for position in range(self.parallel_position + 1):
            snapshot = self._emitted_loop_headers.get(position)
            if snapshot is None:
                _fail(
                    _PARALLEL_LOST,
                    "a selected-loop ancestor header was never recorded",
                )
            matches = [
                (loop, owner)
                for loop, owner in found
                if self._panel_loop_header_matches(loop, snapshot)
            ]
            if len(matches) != 1 or matches[0][1] is not prior:
                _fail(
                    _PARALLEL_LOST,
                    "the assembled function does not retain the selected "
                    "loop's exact detached ancestor chain",
                )
            selected = matches[0][0]
            prior = selected
        if selected is None:
            _fail(
                _PARALLEL_LOST,
                "the selected loop could not be re-identified",
            )
        try:
            expected_marked = LLIRRewriter(
                LLIRTraversalContext(
                    stage="LoopIR target lowering",
                    pass_name="snapshot_parallel_selection_policy",
                )
            ).rewrite(selected)
            if type(expected_marked) is not llir.ForLoop:
                _fail(
                    _PARALLEL_LOST,
                    "the selection policy snapshot did not remain a ForLoop",
                )
            self._apply_expected_parallel_marker(expected_marked)
            mark_first_for_loop_parallel(
                [selected],
                EMPTY_PARALLEL_WORKSPACE_CLUSTER,
            )
            if not self._panel_loop_header_matches(selected, expected_marked):
                _fail(
                    _PARALLEL_LOST,
                    "the selected loop did not acquire the unmodified "
                    "legacy parallel marker state",
                )
            self._apply_selected_parallel_policy(
                expected_marked,
                policy_spec,
                _PARALLEL_LOST,
                invoke_marker=False,
            )
            self._apply_selected_parallel_policy(
                selected,
                policy_spec,
                _PARALLEL_LOST,
                invoke_marker=False,
            )
        except (
            LLIRTraversalError,
            AttributeError,
            RecursionError,
            TypeError,
            ValueError,
        ) as error:
            _fail(_PARALLEL_LOST, str(error))
        if not self._panel_loop_header_matches(selected, expected_marked):
            _fail(
                _PARALLEL_LOST,
                "the selected loop did not acquire the required parallel " "policy",
            )
        marked = [
            loop
            for loop, _owner in found
            if loop.omp_parallel_for or getattr(loop, "_use_atomic_scheduling", False)
        ]
        if marked != [selected]:
            _fail(
                _PARALLEL_LOST,
                "the selected loop must own the assembled function's only "
                "parallel marking",
            )
        return function


def _sparse_workspace_chain(program: LoopProgram) -> bool:
    """Whether the program is the serial sparse-workspace family chain.

    Routing is purely structural over the verified typed tree: one outer
    single-cursor sparse loop whose whole body is one
    :class:`SparseWorkspaceRegion`.  Everything else — including any other
    placement of a sparse workspace region — stays on the general target
    lowering and keeps its existing fail-closed boundary.
    """

    body = program.body
    if type(body) is not Block or len(body.statements) != 1:
        return False
    outer = body.statements[0]
    if type(outer) is not SparseFor:
        return False
    inner = outer.body
    return (
        type(inner) is Block
        and len(inner.statements) == 1
        and type(inner.statements[0]) is SparseWorkspaceRegion
    )


def _parallel_sparse_workspace_chain(program: LoopProgram) -> bool:
    """Whether the program is the dense-row parallel sparse-workspace chain.

    Routing is purely structural over the verified typed tree: one outer
    dense row loop whose whole body is one :class:`SparseWorkspaceRegion`.
    Everything else — including any other placement of a sparse workspace
    region — stays on the general target lowering and keeps its existing
    fail-closed boundary.
    """

    body = program.body
    if type(body) is not Block or len(body.statements) != 1:
        return False
    outer = body.statements[0]
    if type(outer) is not DenseFor:
        return False
    inner = outer.body
    return (
        type(inner) is Block
        and len(inner.statements) == 1
        and type(inner.statements[0]) is SparseWorkspaceRegion
    )


def _llir_assignment_root_name(target: llir.Expr) -> Optional[str]:
    """Return the structured root variable of one validated assignment target."""

    current = target
    while type(current) in (llir.ArrayAccess, llir.MemberAccess):
        current = (
            cast(llir.ArrayAccess, current).array
            if type(current) is llir.ArrayAccess
            else cast(llir.MemberAccess, current).base
        )
    return current.name if type(current) is llir.Var else None


def _stored_identity_value(value: object, identity_type: type) -> Optional[int]:
    """Read one exact frozen identity without invoking forged descriptors."""

    if type(value) is not identity_type:
        return None
    state = object.__getattribute__(value, "__dict__")
    if (
        type(state) is not dict
        or len(state) != 1
        or any(type(key) is not str for key in state)
        or "value" not in state
        or type(state["value"]) is not int
    ):
        return None
    return state["value"]


def _detach_tensor_access_metadata(
    metadata: llir.TensorAccessMetadata,
) -> llir.TensorAccessMetadata:
    """Copy every provenance identity so managed passes own no program state."""

    if type(metadata) is not llir.TensorAccessMetadata:
        raise TypeError("tensor access metadata must be exact")
    access_id = _stored_identity_value(metadata.access_id, AccessId)
    tensor_id = _stored_identity_value(metadata.tensor_id, SymbolId)
    if (
        access_id is None
        or tensor_id is None
        or type(metadata.index_ids) is not tuple
        or type(metadata.role) is not llir.TensorAccessRole
    ):
        raise TypeError("tensor access metadata contains malformed identity state")
    index_values = tuple(
        _stored_identity_value(index_id, IndexId) for index_id in metadata.index_ids
    )
    if any(value is None for value in index_values):
        raise TypeError(
            "tensor access metadata contains malformed index identity state"
        )
    return llir.TensorAccessMetadata(
        access_id=AccessId(access_id),
        tensor_id=SymbolId(tensor_id),
        index_ids=tuple(IndexId(cast(int, value)) for value in index_values),
        role=metadata.role,
    )


def _capture_sparse_completion_enum_states() -> (
    Dict[Tuple[type, int], Tuple[Tuple[str, object], ...]]
):
    """Snapshot LLIR enum member state before any managed pass can mutate it."""

    snapshots: Dict[Tuple[type, int], Tuple[Tuple[str, object], ...]] = {}
    for enum_type in (llir.AssignOp, llir.DataType, llir.TensorAccessRole):
        for member in enum_type.__members__.values():
            state = object.__getattribute__(member, "__dict__")
            snapshots[(enum_type, id(member))] = tuple(
                (key, state[key]) for key in state
            )
    return snapshots


_SPARSE_COMPLETION_ENUM_STATES = _capture_sparse_completion_enum_states()


def _sparse_completion_enum_state_matches(value: Enum) -> bool:
    """Require one LLIR enum singleton to retain its import-time stored state."""

    expected = _SPARSE_COMPLETION_ENUM_STATES.get((type(value), id(value)))
    if expected is None:
        return False
    state = object.__getattribute__(value, "__dict__")
    if (
        type(state) is not dict
        or len(state) != len(expected)
        or any(type(key) is not str for key in state)
        or tuple(state) != tuple(key for key, _ in expected)
    ):
        return False
    for key, expected_value in expected:
        actual_value = state[key]
        if key == "__objclass__":
            if actual_value is not expected_value:
                return False
        elif (
            type(actual_value) is not type(expected_value)
            or actual_value != expected_value
        ):
            return False
    return True


def _sparse_completion_enum_state_matches_once(
    value: Enum,
    validated_enum_states: Set[Tuple[type, int]],
) -> bool:
    """Validate one enum once during a synchronous completion comparison."""

    key = (type(value), id(value))
    if key in validated_enum_states:
        return True
    if not _sparse_completion_enum_state_matches(value):
        return False
    validated_enum_states.add(key)
    return True


def _metadata_state_matches(
    actual: object,
    expected: object,
    validated_enum_states: Set[Tuple[type, int]],
) -> bool:
    """Compare two frozen access-provenance values field by exact field."""

    actual_state = object.__getattribute__(actual, "__dict__")
    expected_state = object.__getattribute__(expected, "__dict__")
    expected_fields = ("access_id", "tensor_id", "index_ids", "role")
    if (
        type(actual_state) is not dict
        or type(expected_state) is not dict
        or len(actual_state) != len(expected_fields)
        or len(expected_state) != len(expected_fields)
        or any(type(key) is not str for key in actual_state)
        or any(type(key) is not str for key in expected_state)
        or any(field not in actual_state for field in expected_fields)
        or any(field not in expected_state for field in expected_fields)
    ):
        return False

    for side in (actual_state, expected_state):
        if (
            _stored_identity_value(side["access_id"], AccessId) is None
            or _stored_identity_value(side["tensor_id"], SymbolId) is None
            or type(side["index_ids"]) is not tuple
            or any(
                _stored_identity_value(index_id, IndexId) is None
                for index_id in side["index_ids"]
            )
            or type(side["role"]) is not llir.TensorAccessRole
            or not _sparse_completion_enum_state_matches_once(
                side["role"], validated_enum_states
            )
        ):
            return False
    return (
        _stored_identity_value(actual_state["access_id"], AccessId)
        == _stored_identity_value(expected_state["access_id"], AccessId)
        and _stored_identity_value(actual_state["tensor_id"], SymbolId)
        == _stored_identity_value(expected_state["tensor_id"], SymbolId)
        and len(actual_state["index_ids"]) == len(expected_state["index_ids"])
        and all(
            _stored_identity_value(actual_index, IndexId)
            == _stored_identity_value(expected_index, IndexId)
            for actual_index, expected_index in zip(
                actual_state["index_ids"], expected_state["index_ids"]
            )
        )
        and actual_state["role"] is expected_state["role"]
    )


def _exact_sparse_completion_matches(actual: object, expected: object) -> bool:
    """Compare completed sparse-assembly state with fresh ownership.

    The generic panel comparator is intentionally defensive and recursive
    because it serves several heterogeneous completion boundaries. The sparse
    assembly targets check one large, known LLIR tree on every activating
    compile; this equivalent iterative form avoids recursion and repeated
    field-name sorting on that hot path. The managed LLIR pipeline promises a
    detached tree, so repeated node/container ownership is invalid here; one
    global actual-object census rejects shared aggregates and cycles in the
    same constant-time check.
    """

    pending: List[Tuple[object, object, int]] = [(actual, expected, 0)]
    seen_actual: Set[int] = set()
    # Managed passes have returned before this callback-free, exact-type
    # traversal starts.  Validate each global enum singleton on its first
    # occurrence in this comparison; never retain the result across compiles,
    # where a later hostile mutation must be observed.
    validated_enum_states: Set[Tuple[type, int]] = set()
    while pending:
        actual_value, expected_value, depth = pending.pop()
        if type(actual_value) is not type(expected_value) or depth > 256:
            return False
        if actual_value is None:
            continue
        if type(actual_value) in (bool, int, float, str):
            if actual_value != expected_value:
                return False
            continue
        if isinstance(actual_value, Enum):
            if (
                actual_value is not expected_value
                or not _sparse_completion_enum_state_matches_once(
                    actual_value, validated_enum_states
                )
            ):
                return False
            continue
        if type(actual_value) in (AccessId, SymbolId, IndexId):
            actual_state = object.__getattribute__(actual_value, "__dict__")
            expected_state = object.__getattribute__(expected_value, "__dict__")
            if (
                type(actual_state) is not dict
                or type(expected_state) is not dict
                or len(actual_state) != 1
                or len(expected_state) != 1
            ):
                return False
            actual_field = next(iter(actual_state))
            expected_field = next(iter(expected_state))
            if (
                type(actual_field) is not str
                or actual_field != "value"
                or type(expected_field) is not str
                or expected_field != "value"
                or type(actual_state[actual_field]) is not int
                or type(expected_state[expected_field]) is not int
                or actual_state[actual_field] != expected_state[expected_field]
            ):
                return False
            continue

        if type(actual_value) is llir.TensorAccessMetadata:
            # Frozen immutable access provenance is value state, not owned
            # tree structure: the production two-phase rewrite legitimately
            # duplicates a work body whose detached statements retain the
            # same metadata values, so metadata compares exactly by stored
            # state and stays outside the fresh-ownership census (like the
            # interned empty tuple, it cannot be mutated and cannot close a
            # cycle through its immutable leaf fields).
            if not _metadata_state_matches(
                actual_value, expected_value, validated_enum_states
            ):
                return False
            continue

        if isinstance(actual_value, llir.Node):
            actual_state = object.__getattribute__(actual_value, "__dict__")
            expected_state = object.__getattribute__(expected_value, "__dict__")
            if type(actual_state) is not dict or type(expected_state) is not dict:
                return False
            if len(actual_state) != len(expected_state):
                return False
            field_names = tuple(actual_state)
            if any(
                type(field_name) is not str or field_name not in expected_state
                for field_name in field_names
            ):
                return False
            actual_id = id(actual_value)
            if actual_id in seen_actual:
                return False
            seen_actual.add(actual_id)
            for field_name in reversed(field_names):
                pending.append(
                    (
                        actual_state[field_name],
                        expected_state[field_name],
                        depth + 1,
                    )
                )
            continue
        elif type(actual_value) in (list, tuple):
            actual_sequence = cast(Any, actual_value)
            expected_sequence = cast(Any, expected_value)
            if len(actual_sequence) != len(expected_sequence):
                return False
            # CPython interns the empty tuple used by default call/template
            # arguments. It carries no child ownership, cannot form a cycle,
            # and is immutable, so its intentional sharing is outside the
            # detachment invariant enforced for nodes and every other
            # container.  An empty *list* stays censused: it is mutable, so
            # sharing one between two owners is exactly the aliased state
            # this boundary exists to reject.
            if type(actual_value) is tuple and not actual_sequence:
                continue
            actual_id = id(actual_value)
            if actual_id in seen_actual:
                return False
            seen_actual.add(actual_id)
            for index in range(len(actual_sequence) - 1, -1, -1):
                pending.append(
                    (
                        actual_sequence[index],
                        expected_sequence[index],
                        depth + 1,
                    )
                )
            continue
        else:
            return False
    return True


class _SparseWorkspaceLowering(_TargetLowering):
    """Dedicated target lowering for the serial B1 sparse-workspace family.

    Admits exactly the verified shape ``apply_sparse_workspace`` produces —
    one outer level-0 sparse row loop over a doubly-compressed input, a
    :class:`SparseWorkspaceRegion` whose producer is a two-cursor
    INTERSECTION merge descending through a position-bound child sparse
    loop into a merging ADD insertion, and whose consumer is the one
    ordered drain appending each merged entry to the identity-ordered
    rank-2 doubly-compressed result.  Anything else fails closed with
    ``unsupported_program_shape``.

    The raw emission mirrors the retained serial ``coo_workspace_1d``
    legacy lowering statement-for-statement — workspace allocation inside
    the row loop, the while-merge with bound-position descent, ``sort()``
    plus the range-for drain writing indexed appends the shared
    dynamic-vector pass rewrites to ``emplace_back``, the per-row
    compressed-parent assembly, and the final root position close — so the
    generated C++ is byte-identical to the legacy pipeline's output for
    both automatic policy arms.  The general hierarchical-compressed
    restrictions of :class:`_TargetLowering` are untouched: this class is
    reachable only through the exact structural chain above.
    """

    def __init__(
        self,
        program: LoopProgram,
        input_shapes: Mapping[SymbolId, Tuple[int, ...]],
        result_shape: Tuple[int, ...],
    ) -> None:
        self.program = program
        self._input_symbols, seal_token = self._snapshot_program_inputs(program)
        self.decls = {decl.symbol: decl for decl in program.tensors}
        if len(program.outputs) != 1:
            _fail(
                "unsupported_program_shape",
                "this target lowering supports exactly one output tensor",
            )
        self.result_symbol = program.outputs[0]
        self.result_decl = self.decls[self.result_symbol]
        self.result_is_dense = False
        self.sparse_program = True
        self.dimension_names = {}
        self._access_ids = {}
        # Fields the shared driver surface and inherited helpers read: the
        # family carries no dense-workspace region, panel, staging region,
        # heap tile, split loops, or parallel selection.
        self.loops: List[_Loop] = []
        self.region = None
        self.panel = None
        self.relayout = None
        self.result_tile = None
        self.result_tile_depth = -1
        self._tiled_leaf = None
        self._tiled_view = None
        self.relayout_depth = -1
        if program.parallel is not None:
            _fail(
                "unsupported_program_shape",
                "the serial sparse-workspace family owns no parallel " "selection",
            )
        self.parallel = None
        self._validate_display_names()
        self._reserve_generated_name(
            "coo_workspace_1d",
            "the serial sparse-workspace runtime type",
        )
        self.shapes = self._validate_shapes(input_shapes, result_shape)
        self._validate_family_shape()
        self._reserve_family_names()
        self._seal_target_state(seal_token)

    # -- family-shape validation ---------------------------------------------

    def _admits_result_layout(self) -> bool:
        return (
            len(self.result_decl.levels) == 2
            and tuple(level.kind for level in self.result_decl.levels)
            == (LevelKind.COMPRESSED, LevelKind.COMPRESSED)
            and tuple(level.mode for level in self.result_decl.levels) == (0, 1)
        )

    def _result_layout_requirement(self) -> str:
        return "an identity-ordered doubly-compressed result"

    def _validate_family_shape(self) -> None:
        def require(condition: bool, what: str) -> None:
            if not condition:
                _fail(
                    "unsupported_program_shape",
                    f"the serial sparse-workspace target requires {what}",
                )

        def doubly_compressed(decl: TensorDecl) -> bool:
            return (
                len(decl.levels) == 2
                and tuple(level.kind for level in decl.levels)
                == (LevelKind.COMPRESSED, LevelKind.COMPRESSED)
                and tuple(level.mode for level in decl.levels) == (0, 1)
            )

        program = self.program
        body = program.body
        require(
            type(body) is Block and len(body.statements) == 1,
            "a single-statement program body",
        )
        outer = body.statements[0]
        require(type(outer) is SparseFor, "an outer sparse row loop")
        outer = cast(SparseFor, outer)
        outer_cursor = outer.cursor
        require(
            outer_cursor.level == 0 and type(outer_cursor.parent) is RootPosition,
            "a root-parented level-0 outer row cursor",
        )
        inner = outer.body
        require(
            type(inner) is Block
            and len(inner.statements) == 1
            and type(inner.statements[0]) is SparseWorkspaceRegion,
            "the sparse workspace region as the row loop's whole body",
        )
        region = cast(SparseWorkspaceRegion, inner.statements[0])
        workspace_decl = region.workspace
        require(
            type(workspace_decl) is SparseWorkspaceDecl,
            "an exact sparse workspace declaration",
        )
        producer = region.producer
        require(
            type(producer) is Block
            and len(producer.statements) == 1
            and type(producer.statements[0]) is MergedSparseFor,
            "a producer opening with the merged reduction loop",
        )
        merge = cast(MergedSparseFor, producer.statements[0])
        require(
            merge.mode is MergeMode.INTERSECTION
            and len(merge.cursors) == 2
            and len(merge.positions) == 2
            and all(type(bound) is PositionId for bound in merge.positions),
            "a two-cursor position-binding INTERSECTION merge",
        )
        descended = [
            (position, cursor)
            for position, cursor in enumerate(merge.cursors)
            if cursor.tensor == outer_cursor.tensor
            and cursor.level == 1
            and type(cursor.parent) is PositionValue
            and cast(PositionValue, cursor.parent).position == outer.position
        ]
        require(
            len(descended) == 1,
            "one merge cursor descending from the outer row position",
        )
        descended_position, descended_cursor = descended[0]
        rooted = [
            (position, cursor)
            for position, cursor in enumerate(merge.cursors)
            if position != descended_position
            and cursor.tensor != descended_cursor.tensor
            and cursor.level == 0
            and type(cursor.parent) is RootPosition
        ]
        require(
            len(rooted) == 1,
            "one root-parented level-0 merge cursor over the second input",
        )
        rooted_position, rooted_cursor = rooted[0]
        merge_body = merge.body
        require(
            type(merge_body) is Block
            and len(merge_body.statements) == 1
            and type(merge_body.statements[0]) is SparseFor,
            "a child sparse loop as the merge body",
        )
        child = cast(SparseFor, merge_body.statements[0])
        child_cursor = child.cursor
        require(
            child_cursor.tensor == rooted_cursor.tensor
            and child_cursor.level == 1
            and type(child_cursor.parent) is PositionValue
            and cast(PositionValue, child_cursor.parent).position
            == merge.positions[rooted_position],
            "a child cursor descending from the root cursor's merge-bound position",
        )
        child_body = child.body
        require(
            type(child_body) is Block
            and len(child_body.statements) == 1
            and type(child_body.statements[0]) is SparseWorkspaceInsert,
            "the merging insertion as the child loop's whole body",
        )
        insert = cast(SparseWorkspaceInsert, child_body.statements[0])
        require(
            insert.workspace == workspace_decl.workspace
            and insert.op is ReduceOp.ADD
            and type(insert.coord) is IndexValue
            and cast(IndexValue, insert.coord).index == child.coord_index,
            "an ADD insertion at the child loop coordinate",
        )
        consumer = region.consumer
        require(
            type(consumer) is Block
            and len(consumer.statements) == 1
            and type(consumer.statements[0]) is SparseWorkspaceDrainFor,
            "the one ordered drain as the whole consumer",
        )
        drain = cast(SparseWorkspaceDrainFor, consumer.statements[0])
        require(
            drain.workspace == workspace_decl.workspace,
            "a drain of the region's own workspace",
        )
        drain_body = drain.body
        require(
            type(drain_body) is Block
            and len(drain_body.statements) == 1
            and type(drain_body.statements[0]) is AppendEntry,
            "the ordered append as the drain loop's whole body",
        )
        append = cast(AppendEntry, drain_body.statements[0])
        require(
            append.tensor == self.result_symbol
            and len(append.coords) == 2
            and all(type(coord) is IndexValue for coord in append.coords)
            and cast(IndexValue, append.coords[0]).index == outer.coord_index
            and cast(IndexValue, append.coords[1]).index == drain.index
            and type(append.value) is SparseWorkspaceValue
            and cast(SparseWorkspaceValue, append.value).workspace
            == workspace_decl.workspace,
            "an append of the drained value at the row and drain coordinates",
        )

        descended_decl = self.decls[descended_cursor.tensor]
        rooted_decl = self.decls[rooted_cursor.tensor]
        require(
            len(program.inputs) == 2
            and set(program.inputs) == {descended_cursor.tensor, rooted_cursor.tensor}
            and doubly_compressed(descended_decl)
            and doubly_compressed(rooted_decl),
            "exactly two identity-ordered doubly-compressed inputs",
        )
        require(
            self._admits_result_layout(),
            self._result_layout_requirement(),
        )
        require(
            descended_decl.dtype is rooted_decl.dtype
            and descended_decl.dtype is self.result_decl.dtype
            and workspace_decl.dtype is self.result_decl.dtype
            and self.result_decl.dtype in _SCALAR_TO_TORCH,
            "one shared supported scalar type",
        )
        require(
            self._level_dimension(descended_cursor.tensor, 0)
            == self.result_decl.dimensions[0]
            and self._level_dimension(descended_cursor.tensor, 1)
            == self._level_dimension(rooted_cursor.tensor, 0)
            and self._level_dimension(rooted_cursor.tensor, 1)
            == self.result_decl.dimensions[1]
            and workspace_decl.drain_dimension == self.result_decl.dimensions[1],
            "row, reduction, and drain dimensions in matmul agreement",
        )
        row_dimension = self._level_dimension(descended_cursor.tensor, 0)
        merge_dimension = self._level_dimension(descended_cursor.tensor, 1)
        drain_dimension = workspace_decl.drain_dimension
        require(
            len({row_dimension, merge_dimension, drain_dimension}) == 3,
            "three distinct loop dimensions",
        )

        self.outer_loop = outer
        self.sparse_region = region
        self.workspace_decl = workspace_decl
        self.merge = merge
        self.child_loop = child
        self.sparse_insert = insert
        self.sparse_drain = drain
        self.sparse_append = append
        self.row_name = self.dimension_names[row_dimension]
        self.merge_name = self.dimension_names[merge_dimension]
        self.drain_name = self.dimension_names[drain_dimension]
        # The value-typed cursors admissible in the insertion value, and
        # the level drivers backing the shared input access metadata.
        self._value_cursors = {
            descended_cursor.cursor: descended_cursor,
            child_cursor.cursor: child_cursor,
        }
        self.level_drivers = {
            descended_cursor.tensor: {
                0: outer.coord_index,
                1: merge.coord_index,
            },
            rooted_cursor.tensor: {
                0: merge.coord_index,
                1: child.coord_index,
            },
        }
        self._validate_insert_value(insert.value)

    def _validate_insert_value(self, expr: Expr) -> None:
        if type(expr) is CursorValue:
            cursor_value = cast(CursorValue, expr)
            if cursor_value.cursor not in self._value_cursors:
                _fail(
                    "unsupported_program_shape",
                    "the insertion value may read only the merge's descended "
                    "cursor and the child cursor",
                )
            return
        if type(expr) is BinaryExpr:
            binary = cast(BinaryExpr, expr)
            if binary.op not in (BinaryOp.ADD, BinaryOp.MUL):
                _fail(
                    "unsupported_program_shape",
                    "the insertion value supports ADD and MUL only",
                )
            self._validate_insert_value(binary.lhs)
            self._validate_insert_value(binary.rhs)
            return
        _fail(
            "unsupported_program_shape",
            f"unsupported insertion value expression {type(expr).__name__}",
        )

    def _reserve_family_names(self) -> None:
        workspace_name = self.workspace_decl.name
        if not _safe_cpp_display_identifier(workspace_name):
            _fail(
                "invalid_display_name",
                f"workspace name {workspace_name!r} is not a safe ASCII C++ "
                "identifier",
            )
        owner = f"workspace {workspace_name!r}"
        self._reserve_generated_name(workspace_name, owner)
        self._reserve_generated_name(f"{workspace_name}_value", owner)
        self._reserve_generated_name("it", "the workspace drain iterator")
        for cursor in self.merge.cursors:
            tensor_name = self.decls[cursor.tensor].name
            self._reserve_generated_name(
                f"{self.merge_name}_{tensor_name}",
                f"merged coordinate of input tensor {tensor_name!r}",
            )

    # -- emission -------------------------------------------------------------

    def _sparse_value(self, expr: Expr) -> llir.Expr:
        """Lower the insertion value: every admitted cursor is aligned."""

        if type(expr) is CursorValue:
            cursor = self._value_cursors[cast(CursorValue, expr).cursor]
            decl = self.decls[cursor.tensor]
            torch_dtype = _SCALAR_TO_TORCH[decl.dtype]
            return llir.ArrayAccess(
                array=llir.Var(
                    name=f"{decl.name}_val",
                    type=llir.DataType.ptr_type(torch_dtype),
                ),
                index=llir.Var(
                    name=self._cursor_position_name(cursor),
                    type=llir.DataType.INT,
                ),
                tensor_access=self._input_metadata(cursor.tensor),
            )
        binary = cast(BinaryExpr, expr)
        return llir.BinOp(
            op=_BINARY_TO_CXX[binary.op],
            left=self._sparse_value(binary.lhs),
            right=self._sparse_value(binary.rhs),
        )

    def _append_metadata(self) -> llir.TensorAccessMetadata:
        return _detach_tensor_access_metadata(
            llir.TensorAccessMetadata(
                access_id=self._access_id(self.result_symbol),
                tensor_id=self.result_symbol,
                index_ids=(
                    self.outer_loop.coord_index,
                    self.sparse_drain.index,
                ),
                role=llir.TensorAccessRole.RESULT_WRITE,
            ),
        )

    def _segment_start(self, cursor: SparseCursorDecl) -> llir.Expr:
        """The pos-array index opening one cursor's segment (level-0 aware)."""

        if cursor.level == 0:
            return llir.Literal(0, llir.DataType.INT)
        return self._cursor_parent_var(cursor)

    def _segment_end(self, cursor: SparseCursorDecl) -> llir.Expr:
        if cursor.level == 0:
            return llir.Literal(1, llir.DataType.INT)
        return llir.Add(
            left=self._cursor_parent_var(cursor),
            right=llir.Literal(1, llir.DataType.INT),
        )

    def _child_loop_statements(self) -> List[llir.Stmt]:
        cursor = self.child_loop.cursor
        position_name = self._cursor_position_name(cursor)
        body: List[llir.Stmt] = [
            llir.Comment("Resolve coordinates"),
            llir.VarInit(
                var=llir.Var(name=self.drain_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_crd_array(cursor),
                    index=llir.Var(name=position_name, type=llir.DataType.INT),
                ),
            ),
            llir.BlankLine(),
            llir.FunctionCallStmt(
                name=f"{self.workspace_decl.name}.insert",
                args=[
                    llir.Array(
                        values=[
                            llir.Var(
                                name=self.drain_name,
                                type=llir.DataType.INT64,
                            )
                        ],
                        data_type=llir.DataType.INT64,
                    ),
                    self._sparse_value(self.sparse_insert.value),
                ],
            ),
        ]
        position_var = llir.Var(name=position_name, type=llir.DataType.INT)
        for_loop = llir.ForLoop(
            init=llir.VarInit(
                var=llir.Var(name=position_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_pos_array(cursor),
                    index=self._segment_start(cursor),
                ),
            ),
            cond=llir.BinOp(
                op="<",
                left=position_var,
                right=llir.Var(name=f"{position_name}_end", type=llir.DataType.INT),
            ),
            update=llir.Increment(
                var=llir.Var(name=position_name, type=llir.DataType.INT)
            ),
            body=body,
        )
        for_loop.scorch_index_var = self.drain_name
        return [
            llir.Comment("Initialize iterators"),
            llir.VarInit(
                var=llir.Var(name=f"{position_name}_end", type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_pos_array(cursor),
                    index=self._segment_end(cursor),
                ),
            ),
            llir.BlankLine(),
            for_loop,
        ]

    def _merge_statements(self) -> List[llir.Stmt]:
        cursors = self.merge.cursors
        dimension_name = self.merge_name

        def coordinate_var(cursor: SparseCursorDecl) -> llir.Var:
            return llir.Var(
                name=f"{dimension_name}_{self.decls[cursor.tensor].name}",
                type=llir.DataType.INT,
            )

        def dimension_var() -> llir.Var:
            return llir.Var(name=dimension_name, type=llir.DataType.INT)

        iterator_inits: List[llir.Stmt] = []
        for cursor in cursors:
            position_name = self._cursor_position_name(cursor)
            iterator_inits.append(
                llir.VarInit(
                    var=llir.Var(name=position_name, type=llir.DataType.INT),
                    value=llir.ArrayAccess(
                        array=self._cursor_pos_array(cursor),
                        index=self._segment_start(cursor),
                    ),
                )
            )
            iterator_inits.append(
                llir.VarInit(
                    var=llir.Var(name=f"{position_name}_end", type=llir.DataType.INT),
                    value=llir.ArrayAccess(
                        array=self._cursor_pos_array(cursor),
                        index=self._segment_end(cursor),
                    ),
                )
            )

        while_body: List[llir.Stmt] = [llir.Comment("Load coordinates")]
        for cursor in cursors:
            while_body.append(
                llir.VarInit(
                    var=coordinate_var(cursor),
                    value=llir.ArrayAccess(
                        array=self._cursor_crd_array(cursor),
                        index=llir.Var(
                            name=self._cursor_position_name(cursor),
                            type=llir.DataType.INT,
                        ),
                    ),
                )
            )
        while_body.extend(
            [
                llir.BlankLine(),
                llir.Comment("Resolve coordinates"),
                llir.VarInit(
                    var=llir.Var(name=dimension_name, type=llir.DataType.INT),
                    value=llir.FunctionCall(
                        name="std::min",
                        args=[
                            llir.Array(
                                values=tuple(
                                    coordinate_var(cursor) for cursor in cursors
                                ),
                                data_type=llir.DataType.INT,
                            )
                        ],
                    ),
                ),
                llir.BlankLine(),
                llir.Comment("Inner loops over child regions"),
                llir.IfThenElse(
                    cond_list=[
                        llir.BinOp(
                            op="&&",
                            left=llir.BinOp(
                                op="==",
                                left=coordinate_var(cursors[0]),
                                right=dimension_var(),
                            ),
                            right=llir.BinOp(
                                op="==",
                                left=coordinate_var(cursors[1]),
                                right=dimension_var(),
                            ),
                        )
                    ],
                    then_body_list=[self._child_loop_statements()],
                    make_last_case_else=False,
                ),
                llir.BlankLine(),
                llir.Comment("Advance iterators"),
            ]
        )
        for cursor in cursors:
            while_body.append(
                llir.Assign(
                    var=llir.Var(
                        name=self._cursor_position_name(cursor),
                        type=llir.DataType.INT,
                    ),
                    value=llir.BinOp(
                        op="==",
                        left=coordinate_var(cursor),
                        right=dimension_var(),
                    ),
                    op=llir.AssignOp.ADD_ASSIGN,
                    cast=True,
                )
            )

        def position_cond(cursor: SparseCursorDecl) -> llir.Expr:
            return llir.BinOp(
                op="<",
                left=llir.Var(
                    name=self._cursor_position_name(cursor),
                    type=llir.DataType.INT,
                ),
                right=llir.Var(
                    name=f"{self._cursor_position_name(cursor)}_end",
                    type=llir.DataType.INT,
                ),
            )

        merge_loop = llir.WhileLoop(
            cond=llir.BinOp(
                op="&&",
                left=position_cond(cursors[0]),
                right=position_cond(cursors[1]),
            ),
            body=while_body,
        )
        merge_loop.scorch_index_var = dimension_name
        return [
            llir.Comment("Initialize iterators"),
            *iterator_inits,
            llir.BlankLine(),
            merge_loop,
        ]

    def _drain_statements(self) -> List[llir.Stmt]:
        workspace_name = self.workspace_decl.name
        result_name = self.result_decl.name
        torch_dtype = _SCALAR_TO_TORCH[self.workspace_decl.dtype]
        iterator_var = llir.Var(name="it", type=llir.DataType.CONST_AUTO_REF)
        counter_var = llir.Var(name=f"p{result_name}1", type=llir.DataType.INT64)
        drain_body: List[llir.Stmt] = [
            llir.VarInit(
                var=llir.Var(name=self.drain_name, type=llir.DataType.INT64),
                value=llir.MemberAccess(base=iterator_var, member="first"),
            ),
            llir.VarInit(
                var=llir.Var(
                    name=f"{workspace_name}_value",
                    type=dtype_to_c_datatype(torch_dtype),
                ),
                value=llir.MemberAccess(base=iterator_var, member="second"),
            ),
            llir.BlankLine(),
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{result_name}_values",
                        type=llir.DataType.NO_TYPE,
                    ),
                    index=counter_var,
                    tensor_access=self._append_metadata(),
                ),
                value=llir.Var(
                    name=f"{workspace_name}_value",
                    type=llir.DataType.NO_TYPE,
                ),
            ),
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{result_name}1_crd",
                        type=llir.DataType.NO_TYPE,
                    ),
                    index=counter_var,
                ),
                value=llir.Var(name=self.drain_name, type=llir.DataType.NO_TYPE),
            ),
            llir.Increment(var=counter_var),
        ]
        return [
            llir.BlankLine(),
            llir.Comment("Lower consumer CIN"),
            llir.FunctionCallStmt(name=f"{workspace_name}.sort", args=[]),
            llir.ForLoopAuto(
                var=iterator_var,
                array=llir.Var(name=workspace_name, type=llir.DataType.AUTO),
                body=drain_body,
            ),
        ]

    def _row_assembly_statements(self) -> List[llir.Stmt]:
        result_name = self.result_decl.name
        return [
            llir.BlankLine(),
            llir.BlankLine(),
            llir.Comment("Assembly compressed _level indices"),
            llir.IfThenElse(
                cond=llir.BinOp(
                    op="<",
                    left=llir.FunctionCall(name=f"{result_name}1_pos.back", args=[]),
                    right=llir.Var(name=f"p{result_name}1", type=llir.DataType.INT64),
                ),
                then_body=[
                    llir.FunctionCallStmt(
                        name=f"{result_name}0_crd.push_back",
                        args=[llir.Var(name=self.row_name, type=llir.DataType.INT64)],
                    ),
                ],
            ),
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{result_name}1_pos",
                        type=llir.DataType.STD_VECTOR_C_INT,
                    ),
                    index=llir.FunctionCall(name=f"{result_name}0_crd.size", args=[]),
                ),
                value=llir.FunctionCall(name=f"{result_name}1_crd.size", args=[]),
            ),
        ]

    def _workspace_init_statement(self) -> llir.VarInit:
        workspace = self.workspace_decl
        torch_dtype = _SCALAR_TO_TORCH[workspace.dtype]
        element_type = dtype_to_c_datatype(torch_dtype)
        return llir.VarInit(
            var=llir.Var(name=workspace.name, type=llir.DataType.AUTO),
            value=llir.FunctionCall(
                name=f"coo_workspace_1d<{element_type.value}, 1>",
                args=[llir.Literal(value=1024, data_type=llir.DataType.INT)],
            ),
        )

    def _region_statements(self) -> List[llir.Stmt]:
        return [
            llir.Comment("Initialize workspaces"),
            self._workspace_init_statement(),
            *self._merge_statements(),
            *self._drain_statements(),
            *self._row_assembly_statements(),
        ]

    def _row_prologue_statements(self) -> List[llir.Stmt]:
        """Resolve the row coordinate at the start of the outer sparse loop."""

        cursor = self.outer_loop.cursor
        position_name = self._cursor_position_name(cursor)
        return [
            llir.Comment("Resolve coordinates"),
            llir.VarInit(
                var=llir.Var(name=self.row_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_crd_array(cursor),
                    index=llir.Var(name=position_name, type=llir.DataType.INT),
                ),
            ),
            llir.BlankLine(),
        ]

    def _outer_loop_prefix_statements(self) -> List[llir.Stmt]:
        """Initialize the outer sparse cursor immediately before its loop."""

        cursor = self.outer_loop.cursor
        position_name = self._cursor_position_name(cursor)
        return [
            llir.Comment("Initialize iterators"),
            llir.VarInit(
                var=llir.Var(name=f"{position_name}_end", type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_pos_array(cursor),
                    index=self._segment_end(cursor),
                ),
            ),
            llir.BlankLine(),
        ]

    def _completed_drain_loop(self) -> llir.ForLoopAuto:
        """The exact drain shape the dynamic-vector pass must leave behind."""

        workspace_name = self.workspace_decl.name
        result_name = self.result_decl.name
        iterator_var = llir.Var(name="it", type=llir.DataType.CONST_AUTO_REF)
        workspace_value_name = f"{workspace_name}_value"
        return llir.ForLoopAuto(
            var=iterator_var,
            array=llir.Var(name=workspace_name, type=llir.DataType.AUTO),
            body=[
                llir.VarInit(
                    var=llir.Var(name=self.drain_name, type=llir.DataType.INT64),
                    value=llir.MemberAccess(base=iterator_var, member="first"),
                ),
                llir.VarInit(
                    var=llir.Var(
                        name=workspace_value_name,
                        type=dtype_to_c_datatype(
                            _SCALAR_TO_TORCH[self.workspace_decl.dtype]
                        ),
                    ),
                    value=llir.MemberAccess(base=iterator_var, member="second"),
                ),
                llir.BlankLine(),
                llir.FunctionCallStmt(
                    name=f"{result_name}_values.emplace_back",
                    args=[
                        llir.Var(
                            name=workspace_value_name,
                            type=llir.DataType.NO_TYPE,
                        )
                    ],
                ),
                llir.FunctionCallStmt(
                    name=f"{result_name}1_crd.emplace_back",
                    args=[
                        llir.Var(
                            name=self.drain_name,
                            type=llir.DataType.NO_TYPE,
                        )
                    ],
                ),
                llir.Increment(
                    var=llir.Var(
                        name=f"p{result_name}1",
                        type=llir.DataType.INT64,
                    )
                ),
            ],
        )

    def _completed_root_position_statement(self) -> llir.FunctionCallStmt:
        """The exact checked root-position close left by the vector pass."""

        result_name = self.result_decl.name
        return llir.FunctionCallStmt(
            name="scorch_vector_set",
            args=[
                llir.Var(
                    name=f"{result_name}0_pos",
                    type=llir.DataType.NO_TYPE,
                ),
                llir.Add(
                    llir.Var(
                        name=f"{result_name}0_pos_index",
                        type=llir.DataType.INT64,
                    ),
                    llir.Literal(value=1, data_type=llir.DataType.INT32),
                ),
                llir.FunctionCall(name=f"{result_name}0_crd.size", args=[]),
            ],
        )

    def _completed_position_init_statement(
        self,
        level: int,
    ) -> llir.FunctionCallStmt:
        """One checked position sentinel left by the dynamic-vector pass."""

        return llir.FunctionCallStmt(
            name="scorch_vector_set",
            args=[
                llir.Var(
                    name=f"{self.result_decl.name}{level}_pos",
                    type=llir.DataType.NO_TYPE,
                ),
                llir.Literal(value=0, data_type=llir.DataType.INT32),
                llir.Literal(value=0, data_type=llir.DataType.INT32),
            ],
        )

    def _outer_row_loop(self, body: List[llir.Stmt]) -> llir.ForLoop:
        """Construct the outer sparse row loop around one row body.

        Both the emission and the completion reference call this builder, so
        the completion comparand never shares mutable nodes with the tree the
        managed passes see.  A hostile or defective pass that rewrites the
        emitted header in place therefore diverges from this reconstruction
        and fails the exact comparison instead of escaping through a shared
        snapshot.
        """

        cursor = self.outer_loop.cursor
        position_name = self._cursor_position_name(cursor)
        position_var = llir.Var(name=position_name, type=llir.DataType.INT)
        row_loop = llir.ForLoop(
            init=llir.VarInit(
                var=llir.Var(name=position_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_pos_array(cursor),
                    index=self._segment_start(cursor),
                ),
            ),
            cond=llir.BinOp(
                op="<",
                left=position_var,
                right=llir.Var(name=f"{position_name}_end", type=llir.DataType.INT),
            ),
            update=llir.Increment(
                var=llir.Var(name=position_name, type=llir.DataType.INT)
            ),
            body=body,
        )
        row_loop.scorch_index_var = self.row_name
        return row_loop

    def complete_sparse_workspace(self, function: llir.Function) -> llir.Function:
        """Require the exact dynamic-vector rewrite owned by the B1 target.

        B1 emits indexed assignments deliberately so it can share the
        production dynamic-vector pass with the legacy route.  An omitted,
        duplicated, or partial rewrite would otherwise leave unchecked writes
        into empty result vectors.  Validate the assembled tree after all
        managed passes and translate every malformed/cyclic state into the
        target-owned completion diagnostic.
        """

        result_name = self.result_decl.name
        try:
            expected_assembler = self.result_assembler()
            expected_root_close = self._completed_root_position_statement()
            expected_final_assembly = expected_assembler.emit_final_assembly()
            expected_level_indices = expected_assembler.emit_level_indices_init()
            expected_position_levels = {
                f"{result_name}{level}_pos": level for level in (0, 1)
            }
            completed_position_inits: Set[str] = set()
            for index, statement in enumerate(expected_level_indices):
                if type(statement) is not llir.Assign:
                    continue
                root_name = _llir_assignment_root_name(statement.var)
                level = expected_position_levels.get(cast(str, root_name))
                if level is None:
                    continue
                expected_level_indices[index] = self._completed_position_init_statement(
                    level
                )
                completed_position_inits.add(cast(str, root_name))
            if completed_position_inits != set(expected_position_levels):
                _fail(
                    _SPARSE_WORKSPACE_LOST,
                    "the result assembler must own both position sentinels",
                )
            expected_level_index_group: List[llir.Stmt] = [
                llir.Comment("Init result level indices"),
                *expected_level_indices,
            ]
            expected_size_stmts = self.result_size_inits()
            if expected_size_stmts:
                expected_size_stmts = [
                    llir.Comment("Init result tensor level sizes"),
                    *expected_size_stmts,
                ]
            expected_kernel_abi = self.kernel_abi()
            expected_row_tail = self._row_assembly_statements()
            expected_row_tail[-1] = llir.FunctionCallStmt(
                name="scorch_vector_set",
                args=[
                    llir.Var(
                        name=f"{result_name}1_pos",
                        type=llir.DataType.NO_TYPE,
                    ),
                    llir.FunctionCall(name=f"{result_name}0_crd.size", args=[]),
                    llir.FunctionCall(name=f"{result_name}1_crd.size", args=[]),
                ],
            )
            expected_row_body: List[llir.Stmt] = [
                *self._row_prologue_statements(),
                llir.Comment("Initialize workspaces"),
                self._workspace_init_statement(),
                *self._merge_statements(),
                llir.BlankLine(),
                llir.Comment("Lower consumer CIN"),
                llir.FunctionCallStmt(
                    name=f"{self.workspace_decl.name}.sort",
                    args=[],
                ),
                self._completed_drain_loop(),
                *expected_row_tail,
            ]
            expected_body: List[llir.Stmt] = [
                *expected_kernel_abi.emit_validation(),
                *expected_size_stmts,
                *expected_kernel_abi.emit_input_prologue(),
                llir.BlankLine(),
                *expected_level_index_group,
                llir.Comment("Initialize result value array"),
                *expected_assembler.emit_value_array_init(),
                *self.tile_size_inits(),
                llir.BlankLine(),
                *self._outer_loop_prefix_statements(),
                self._outer_row_loop(expected_row_body),
                llir.BlankLine(),
                llir.Comment("Assembly compressed _level indices"),
                expected_root_close,
                *expected_final_assembly,
            ]
            expected_function = expected_kernel_abi.assemble_function(expected_body)
        except (
            LLIRTraversalError,
            AttributeError,
            RecursionError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            _fail(_SPARSE_WORKSPACE_LOST, str(error))
        try:
            completion_matches = _exact_sparse_completion_matches(
                function, expected_function
            )
        except LoopIRTargetError:
            raise
        except Exception as error:
            _fail(_SPARSE_WORKSPACE_LOST, str(error))
        if not completion_matches:
            _fail(
                _SPARSE_WORKSPACE_LOST,
                "the assembled function must exactly match the completed "
                "sparse-workspace target, including both checked position "
                "sentinels, the canonical producer/drain row, and final assembly",
            )
        return function

    def raw_loop_statements(self) -> List[llir.Stmt]:
        _TargetLowering._require_program_graph_unchanged(self)
        result_name = self.result_decl.name
        row_body: List[llir.Stmt] = [
            *self._row_prologue_statements(),
            *self._region_statements(),
        ]
        row_loop = self._outer_row_loop(row_body)
        return [
            *self._outer_loop_prefix_statements(),
            row_loop,
            llir.BlankLine(),
            llir.Comment("Assembly compressed _level indices"),
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{result_name}0_pos",
                        type=llir.DataType.STD_VECTOR_C_INT,
                    ),
                    index=llir.Add(
                        llir.Var(
                            name=f"{result_name}0_pos_index",
                            type=llir.DataType.INT64,
                        ),
                        llir.Literal(1),
                    ),
                ),
                value=llir.FunctionCall(name=f"{result_name}0_crd.size", args=[]),
            ),
        ]


class _RowScopeSparseWorkspaceLowering(_SparseWorkspaceLowering):
    """The sound dense-row CSR twin of the serial sparse-workspace family.

    Admits exactly the B1 producer/drain chain with an identity-ordered
    dense-row CSR (``ds``) result.  The defective legacy comparand for
    this family sizes ``C1_pos`` by the first operand's stored-row count,
    silently associating later rows' values with earlier rows whenever a
    logical row is empty; this typed route instead sizes and closes
    ``C1_pos`` from the logical result row extent: a per-row positional
    catch-up closes every skipped empty row before the stored row's
    drain, the stored row closes its own entry, and a final catch-up
    after the loop closes through ``C0_size``.  By construction the
    generated source therefore never byte-matches the legacy kernel; the
    family is proven against the production LoopIR oracle and the
    PyTorch dense reference under the established no-parity discipline.
    """

    def _admits_result_layout(self) -> bool:
        return (
            len(self.result_decl.levels) == 2
            and tuple(level.kind for level in self.result_decl.levels)
            == (LevelKind.DENSE, LevelKind.COMPRESSED)
            and tuple(level.mode for level in self.result_decl.levels) == (0, 1)
        )

    def _result_layout_requirement(self) -> str:
        return "an identity-ordered dense-row CSR result"

    def _row_close_statement(self) -> llir.Assign:
        """``C1_pos[C1_pos_index + 1] = C1_crd.size()`` (raw indexed form)."""

        result_name = self.result_decl.name
        return llir.Assign(
            var=llir.ArrayAccess(
                array=llir.Var(
                    name=f"{result_name}1_pos",
                    type=llir.DataType.STD_VECTOR_C_INT,
                ),
                index=llir.Add(
                    llir.Var(
                        name=f"{result_name}1_pos_index",
                        type=llir.DataType.INT64,
                    ),
                    llir.Literal(1),
                ),
            ),
            value=llir.FunctionCall(name=f"{result_name}1_crd.size", args=[]),
        )

    def _completed_row_close_statement(self) -> llir.FunctionCallStmt:
        """The exact checked row close left by the dynamic-vector pass."""

        result_name = self.result_decl.name
        return llir.FunctionCallStmt(
            name="scorch_vector_set",
            args=[
                llir.Var(
                    name=f"{result_name}1_pos",
                    type=llir.DataType.NO_TYPE,
                ),
                llir.Add(
                    llir.Var(
                        name=f"{result_name}1_pos_index",
                        type=llir.DataType.INT64,
                    ),
                    llir.Literal(value=1, data_type=llir.DataType.INT32),
                ),
                llir.FunctionCall(name=f"{result_name}1_crd.size", args=[]),
            ],
        )

    def _row_catch_up_loop(self, bound: llir.Expr, *, completed: bool) -> llir.ForLoop:
        """Close every skipped logical row through ``bound`` exclusively."""

        result_name = self.result_decl.name
        pos_index = llir.Var(
            name=f"{result_name}1_pos_index",
            type=llir.DataType.INT64,
        )
        return llir.ForLoop(
            init=None,
            cond=llir.BinOp(op="<", left=pos_index, right=bound),
            update=llir.Increment(
                var=llir.Var(
                    name=f"{result_name}1_pos_index",
                    type=llir.DataType.INT64,
                )
            ),
            body=[
                (
                    self._completed_row_close_statement()
                    if completed
                    else self._row_close_statement()
                )
            ],
        )

    def _row_bound_var(self) -> llir.Var:
        return llir.Var(name=self.row_name, type=llir.DataType.INT64)

    def _extent_bound_var(self) -> llir.Var:
        return llir.Var(
            name=f"{self.result_decl.name}0_size",
            type=llir.DataType.INT64,
        )

    def _row_assembly_statements(self) -> List[llir.Stmt]:
        return [
            llir.BlankLine(),
            llir.BlankLine(),
            llir.Comment("Assembly compressed _level indices"),
            self._row_close_statement(),
        ]

    def raw_loop_statements(self) -> List[llir.Stmt]:
        _TargetLowering._require_program_graph_unchanged(self)
        row_body: List[llir.Stmt] = [
            *self._row_prologue_statements(),
            llir.Comment("Assemble COMPRESSED level"),
            self._row_catch_up_loop(self._row_bound_var(), completed=False),
            *self._region_statements(),
        ]
        row_loop = self._outer_row_loop(row_body)
        return [
            *self._outer_loop_prefix_statements(),
            row_loop,
            llir.BlankLine(),
            llir.Comment("Assembly compressed _level indices"),
            self._row_catch_up_loop(self._extent_bound_var(), completed=False),
        ]

    def complete_sparse_workspace(self, function: llir.Function) -> llir.Function:
        """Require the exact dynamic-vector rewrite owned by this target.

        The reconstruction mirrors the B1 discipline — every expected node
        is locally owned, the managed pass is never consulted — with the
        dense-row CSR differences: one checked position sentinel, the
        per-row and final positional catch-ups, and no compressed-parent
        coordinate assembly.
        """

        result_name = self.result_decl.name
        try:
            expected_assembler = self.result_assembler()
            expected_final_assembly = expected_assembler.emit_final_assembly()
            expected_level_indices = expected_assembler.emit_level_indices_init()
            sentinel_name = f"{result_name}1_pos"
            completed_position_inits: Set[str] = set()
            for index, statement in enumerate(expected_level_indices):
                if type(statement) is not llir.Assign:
                    continue
                root_name = _llir_assignment_root_name(statement.var)
                if root_name != sentinel_name:
                    continue
                expected_level_indices[index] = self._completed_position_init_statement(
                    1
                )
                completed_position_inits.add(cast(str, root_name))
            if completed_position_inits != {sentinel_name}:
                _fail(
                    _SPARSE_WORKSPACE_LOST,
                    "the result assembler must own the row position sentinel",
                )
            expected_level_index_group: List[llir.Stmt] = [
                llir.Comment("Init result level indices"),
                *expected_level_indices,
            ]
            expected_size_stmts = self.result_size_inits()
            if expected_size_stmts:
                expected_size_stmts = [
                    llir.Comment("Init result tensor level sizes"),
                    *expected_size_stmts,
                ]
            expected_kernel_abi = self.kernel_abi()
            expected_row_tail: List[llir.Stmt] = [
                llir.BlankLine(),
                llir.BlankLine(),
                llir.Comment("Assembly compressed _level indices"),
                self._completed_row_close_statement(),
            ]
            expected_row_body: List[llir.Stmt] = [
                *self._row_prologue_statements(),
                llir.Comment("Assemble COMPRESSED level"),
                self._row_catch_up_loop(self._row_bound_var(), completed=True),
                llir.Comment("Initialize workspaces"),
                self._workspace_init_statement(),
                *self._merge_statements(),
                llir.BlankLine(),
                llir.Comment("Lower consumer CIN"),
                llir.FunctionCallStmt(
                    name=f"{self.workspace_decl.name}.sort",
                    args=[],
                ),
                self._completed_drain_loop(),
                *expected_row_tail,
            ]
            expected_body: List[llir.Stmt] = [
                *expected_kernel_abi.emit_validation(),
                *expected_size_stmts,
                *expected_kernel_abi.emit_input_prologue(),
                llir.BlankLine(),
                *expected_level_index_group,
                llir.Comment("Initialize result value array"),
                *expected_assembler.emit_value_array_init(),
                *self.tile_size_inits(),
                llir.BlankLine(),
                *self._outer_loop_prefix_statements(),
                self._outer_row_loop(expected_row_body),
                llir.BlankLine(),
                llir.Comment("Assembly compressed _level indices"),
                self._row_catch_up_loop(self._extent_bound_var(), completed=True),
                *expected_final_assembly,
            ]
            expected_function = expected_kernel_abi.assemble_function(expected_body)
        except (
            LLIRTraversalError,
            AttributeError,
            RecursionError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            _fail(_SPARSE_WORKSPACE_LOST, str(error))
        try:
            completion_matches = _exact_sparse_completion_matches(
                function, expected_function
            )
        except LoopIRTargetError:
            raise
        except Exception as error:
            _fail(_SPARSE_WORKSPACE_LOST, str(error))
        if not completion_matches:
            _fail(
                _SPARSE_WORKSPACE_LOST,
                "the assembled function must exactly match the completed "
                "row-scope sparse-workspace target, including the checked "
                "position sentinel, both positional catch-ups, and the "
                "canonical producer/drain row",
            )
        return function


class _ParallelSparseWorkspaceLowering(_TargetLowering):
    """Dedicated target lowering for the dense-row parallel workspace family.

    Admits exactly the verified shape ``apply_sparse_workspace`` produces
    for the dense-row CSR SpGEMM family — one outer dense row loop over a
    :class:`SparseWorkspaceRegion` whose producer descends through one
    dense-parented sparse operand loop into a second dense-parented sparse
    operand loop ending in a merging ADD insertion, and whose consumer is
    the one ordered drain appending each merged entry to the identity-
    ordered dense-row CSR result.  Anything else fails closed with
    ``unsupported_program_shape``.

    The raw emission is the exact serial per-row assembly the legacy
    pipeline hands to the shared production compressed-``Where``/OpenMP
    pass, and this lowering supplies that same pass configuration to the
    managed pipeline.  The pass owns the target's runtime composition —
    the per-thread ``linked_list_workspace_1d`` pool sized by the derived
    thread count, the borrowed per-worker ``make_view()`` lifetimes, the
    two-phase count/fill parallel regions with the derived SpGEMM
    work-estimate/chunk policy, exact Torch-owned output allocation, and
    honest final assembly — so the generated C++ is byte-identical to the
    legacy pipeline's output for both automatic policy arms.  All runtime
    spellings live in the target layer; the semantic program carries only
    the format-neutral region.
    """

    def __init__(
        self,
        program: LoopProgram,
        input_shapes: Mapping[SymbolId, Tuple[int, ...]],
        result_shape: Tuple[int, ...],
    ) -> None:
        self.program = program
        self._input_symbols, seal_token = self._snapshot_program_inputs(program)
        self.decls = {decl.symbol: decl for decl in program.tensors}
        if len(program.outputs) != 1:
            _fail(
                "unsupported_program_shape",
                "this target lowering supports exactly one output tensor",
            )
        self.result_symbol = program.outputs[0]
        self.result_decl = self.decls[self.result_symbol]
        self.result_is_dense = False
        self.sparse_program = True
        self.dimension_names = {}
        self._access_ids = {}
        # Fields the shared driver surface and inherited helpers read: the
        # family carries no dense-workspace region, panel, staging region,
        # heap tile, split loops, or parallel selection.
        self.loops: List[_Loop] = []
        self.region = None
        self.panel = None
        self.relayout = None
        self.result_tile = None
        self.result_tile_depth = -1
        self._tiled_leaf = None
        self._tiled_view = None
        self.relayout_depth = -1
        if program.parallel is not None:
            _fail(
                "unsupported_program_shape",
                "the parallel sparse-workspace family derives its policy in "
                "target lowering and owns no program-level selection",
            )
        self.parallel = None
        self._validate_display_names()
        for name, purpose in (
            ("coo_workspace_1d", "the serial sparse-workspace runtime type"),
            ("linked_list_workspace_1d", "the pooled sparse-workspace runtime type"),
            ("omp_get_thread_num", "the OpenMP worker identity call"),
            ("SCORCH_GRAIN_CODEGEN_SPGEMM", "the SpGEMM work-grain policy token"),
        ):
            # scorch_nthreads/scorch_chunk are globally target-reserved
            # already; the names above are family-specific runtime spellings.
            self._reserve_generated_name(name, purpose)
        self.shapes = self._validate_shapes(input_shapes, result_shape)
        self._validate_family_shape()
        self._reserve_family_names()
        self._seal_target_state(seal_token)

    # -- family-shape validation ----------------------------------------------

    def _validate_family_shape(self) -> None:
        def require(condition: bool, what: str) -> None:
            if not condition:
                _fail(
                    "unsupported_program_shape",
                    f"the parallel sparse-workspace target requires {what}",
                )

        def dense_row_csr(decl: TensorDecl) -> bool:
            return (
                len(decl.levels) == 2
                and tuple(level.kind for level in decl.levels)
                == (LevelKind.DENSE, LevelKind.COMPRESSED)
                and tuple(level.mode for level in decl.levels) == (0, 1)
            )

        program = self.program
        body = program.body
        require(
            type(body) is Block and len(body.statements) == 1,
            "a single-statement program body",
        )
        outer = body.statements[0]
        require(type(outer) is DenseFor, "an outer dense row loop")
        outer = cast(DenseFor, outer)
        inner = outer.body
        require(
            type(inner) is Block
            and len(inner.statements) == 1
            and type(inner.statements[0]) is SparseWorkspaceRegion,
            "the sparse workspace region as the row loop's whole body",
        )
        region = cast(SparseWorkspaceRegion, inner.statements[0])
        workspace_decl = region.workspace
        require(
            type(workspace_decl) is SparseWorkspaceDecl,
            "an exact sparse workspace declaration",
        )
        producer = region.producer
        require(
            type(producer) is Block
            and len(producer.statements) == 1
            and type(producer.statements[0]) is SparseFor,
            "a producer opening with one dense-parented sparse operand loop",
        )
        reduction = cast(SparseFor, producer.statements[0])
        reduction_cursor = reduction.cursor

        def dense_parented(cursor: SparseCursorDecl, index: IndexId) -> bool:
            parent = cursor.parent
            return (
                cursor.level == 1
                and type(parent) is DensePosition
                and cast(DensePosition, parent).tensor == cursor.tensor
                and cast(DensePosition, parent).level == 0
                and type(cast(DensePosition, parent).parent) is RootPosition
                and type(cast(DensePosition, parent).coord) is IndexValue
                and cast(IndexValue, cast(DensePosition, parent).coord).index == index
            )

        require(
            dense_parented(reduction_cursor, outer.index),
            "a reduction cursor descending the first operand's dense row " "position",
        )
        reduction_body = reduction.body
        require(
            type(reduction_body) is Block
            and len(reduction_body.statements) == 1
            and type(reduction_body.statements[0]) is SparseFor,
            "a child sparse loop as the reduction loop's whole body",
        )
        child = cast(SparseFor, reduction_body.statements[0])
        child_cursor = child.cursor
        require(
            child_cursor.tensor != reduction_cursor.tensor
            and dense_parented(child_cursor, reduction.coord_index),
            "a child cursor descending the second operand's dense reduction "
            "position",
        )
        child_body = child.body
        require(
            type(child_body) is Block
            and len(child_body.statements) == 1
            and type(child_body.statements[0]) is SparseWorkspaceInsert,
            "the merging insertion as the child loop's whole body",
        )
        insert = cast(SparseWorkspaceInsert, child_body.statements[0])
        require(
            insert.workspace == workspace_decl.workspace
            and insert.op is ReduceOp.ADD
            and type(insert.coord) is IndexValue
            and cast(IndexValue, insert.coord).index == child.coord_index,
            "an ADD insertion at the child loop coordinate",
        )
        consumer = region.consumer
        require(
            type(consumer) is Block
            and len(consumer.statements) == 1
            and type(consumer.statements[0]) is SparseWorkspaceDrainFor,
            "the one ordered drain as the whole consumer",
        )
        drain = cast(SparseWorkspaceDrainFor, consumer.statements[0])
        require(
            drain.workspace == workspace_decl.workspace,
            "a drain of the region's own workspace",
        )
        drain_body = drain.body
        require(
            type(drain_body) is Block
            and len(drain_body.statements) == 1
            and type(drain_body.statements[0]) is AppendEntry,
            "the ordered append as the drain loop's whole body",
        )
        append = cast(AppendEntry, drain_body.statements[0])
        require(
            append.tensor == self.result_symbol
            and len(append.coords) == 2
            and all(type(coord) is IndexValue for coord in append.coords)
            and cast(IndexValue, append.coords[0]).index == outer.index
            and cast(IndexValue, append.coords[1]).index == drain.index
            and type(append.value) is SparseWorkspaceValue
            and cast(SparseWorkspaceValue, append.value).workspace
            == workspace_decl.workspace,
            "an append of the drained value at the row and drain coordinates",
        )

        reduction_decl = self.decls[reduction_cursor.tensor]
        child_decl = self.decls[child_cursor.tensor]
        require(
            len(program.inputs) == 2
            and set(program.inputs) == {reduction_cursor.tensor, child_cursor.tensor}
            and dense_row_csr(reduction_decl)
            and dense_row_csr(child_decl)
            and dense_row_csr(self.result_decl),
            "exactly two identity-ordered dense-row CSR inputs and result",
        )
        require(
            reduction_decl.dtype is child_decl.dtype
            and reduction_decl.dtype is self.result_decl.dtype
            and workspace_decl.dtype is self.result_decl.dtype
            and self.result_decl.dtype in _SCALAR_TO_TORCH,
            "one shared supported scalar type",
        )
        require(
            self._level_dimension(reduction_cursor.tensor, 0)
            == self.result_decl.dimensions[0]
            and self._level_dimension(reduction_cursor.tensor, 1)
            == self._level_dimension(child_cursor.tensor, 0)
            and self._level_dimension(child_cursor.tensor, 1)
            == self.result_decl.dimensions[1]
            and outer.dimension == self.result_decl.dimensions[0]
            and workspace_decl.drain_dimension == self.result_decl.dimensions[1],
            "row, reduction, and drain dimensions in matmul agreement",
        )
        row_dimension = self.result_decl.dimensions[0]
        reduction_dimension = self._level_dimension(reduction_cursor.tensor, 1)
        drain_dimension = workspace_decl.drain_dimension
        require(
            len({row_dimension, reduction_dimension, drain_dimension}) == 3,
            "three distinct loop dimensions",
        )

        self.outer_loop = outer
        self.sparse_region = region
        self.workspace_decl = workspace_decl
        self.reduction_loop = reduction
        self.child_loop = child
        self.sparse_insert = insert
        self.sparse_drain = drain
        self.sparse_append = append
        self.row_name = self.dimension_names[row_dimension]
        self.reduction_name = self.dimension_names[reduction_dimension]
        self.drain_name = self.dimension_names[drain_dimension]
        # The value-typed cursors admissible in the insertion value, and
        # the level drivers backing the shared input access metadata.
        self._value_cursors = {
            reduction_cursor.cursor: reduction_cursor,
            child_cursor.cursor: child_cursor,
        }
        self.level_drivers = {
            reduction_cursor.tensor: {
                0: outer.index,
                1: reduction.coord_index,
            },
            child_cursor.tensor: {
                0: reduction.coord_index,
                1: child.coord_index,
            },
        }
        self._validate_insert_value(insert.value)

    def _validate_insert_value(self, expr: Expr) -> None:
        if type(expr) is CursorValue:
            cursor_value = cast(CursorValue, expr)
            if cursor_value.cursor not in self._value_cursors:
                _fail(
                    "unsupported_program_shape",
                    "the insertion value may read only the reduction cursor "
                    "and the child cursor",
                )
            return
        if type(expr) is BinaryExpr:
            binary = cast(BinaryExpr, expr)
            if binary.op not in (BinaryOp.ADD, BinaryOp.MUL):
                _fail(
                    "unsupported_program_shape",
                    "the insertion value supports ADD and MUL only",
                )
            self._validate_insert_value(binary.lhs)
            self._validate_insert_value(binary.rhs)
            return
        _fail(
            "unsupported_program_shape",
            f"unsupported insertion value expression {type(expr).__name__}",
        )

    def _reserve_family_names(self) -> None:
        workspace_name = self.workspace_decl.name
        if not _safe_cpp_display_identifier(workspace_name):
            _fail(
                "invalid_display_name",
                f"workspace name {workspace_name!r} is not a safe ASCII C++ "
                "identifier",
            )
        owner = f"workspace {workspace_name!r}"
        result_name = self.result_decl.name
        for name in (
            workspace_name,
            f"{workspace_name}_value",
            f"{workspace_name}_pool",
            f"{workspace_name}_thread_count",
        ):
            self._reserve_generated_name(name, owner)
        self._reserve_generated_name("it", "the workspace drain iterator")
        two_phase_owner = "the two-phase count/fill assembly"
        for name in (
            "_worker",
            "_cnt1",
            "_count1",
            "_offset1",
            "_total1",
            "_i",
            "_base1",
            "_pos1",
        ):
            self._reserve_generated_name(name, two_phase_owner)
        for name in (
            f"{result_name}1_pos_data",
            f"{result_name}1_crd_data",
            f"{result_name}_values_data",
        ):
            self._reserve_generated_name(name, two_phase_owner)

    # -- emission ---------------------------------------------------------------

    def owns_two_phase_output(self) -> bool:
        return True

    def compressed_where_pass_spec(
        self, compile_options: CompileOptions
    ) -> CompressedWhereOpenMPPassSpec:
        """The shared production two-phase configuration for this family."""

        from ..compressed_where_openmp_pass import CompressedWhereOpenMPContext

        torch_dtype = _SCALAR_TO_TORCH[self.workspace_decl.dtype]
        return CompressedWhereOpenMPPassSpec(
            CompressedWhereOpenMPContext(
                result_name=self.result_decl.name,
                # The managed pass must not be able to mutate the SymbolId
                # that keys this lowering's verified program state.
                result_id=SymbolId(self.result_symbol.value),
                compressed_levels=(1,),
                result_assembler=self.result_assembler(),
                workspace_name=self.workspace_decl.name,
                workspace_ctype=dtype_to_c_datatype(torch_dtype).value,
                compile_options=compile_options,
            )
        )

    def _sparse_value(self, expr: Expr) -> llir.Expr:
        """Lower the insertion value: every admitted cursor is aligned."""

        if type(expr) is CursorValue:
            cursor = self._value_cursors[cast(CursorValue, expr).cursor]
            decl = self.decls[cursor.tensor]
            torch_dtype = _SCALAR_TO_TORCH[decl.dtype]
            return llir.ArrayAccess(
                array=llir.Var(
                    name=f"{decl.name}_val",
                    type=llir.DataType.ptr_type(torch_dtype),
                ),
                index=llir.Var(
                    name=self._cursor_position_name(cursor),
                    type=llir.DataType.INT,
                ),
                tensor_access=self._input_metadata(cursor.tensor),
            )
        binary = cast(BinaryExpr, expr)
        return llir.BinOp(
            op=_BINARY_TO_CXX[binary.op],
            left=self._sparse_value(binary.lhs),
            right=self._sparse_value(binary.rhs),
        )

    def _append_metadata(self) -> llir.TensorAccessMetadata:
        return _detach_tensor_access_metadata(
            llir.TensorAccessMetadata(
                access_id=self._access_id(self.result_symbol),
                tensor_id=self.result_symbol,
                index_ids=(
                    self.outer_loop.index,
                    self.sparse_drain.index,
                ),
                role=llir.TensorAccessRole.RESULT_WRITE,
            ),
        )

    def _dense_position_resolve(
        self,
        cursor: SparseCursorDecl,
        coordinate_name: str,
    ) -> List[llir.Stmt]:
        """``pX0 = <coordinate>;`` for one dense-parented operand cursor."""

        parent_name = f"p{self.decls[cursor.tensor].name}0"
        return [
            llir.Comment("Resolve dense coordinates"),
            llir.VarInit(
                var=llir.Var(name=parent_name, type=llir.DataType.INT),
                value=llir.Var(name=coordinate_name, type=llir.DataType.INT),
            ),
        ]

    def _sparse_segment_open(self, cursor: SparseCursorDecl) -> List[llir.Stmt]:
        """``Initialize iterators`` + segment-end bound for one child level."""

        position_name = self._cursor_position_name(cursor)
        parent_var = llir.Var(
            name=f"p{self.decls[cursor.tensor].name}0",
            type=llir.DataType.INT,
        )
        return [
            llir.Comment("Initialize iterators"),
            llir.VarInit(
                var=llir.Var(name=f"{position_name}_end", type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_pos_array(cursor),
                    index=llir.Add(
                        left=parent_var,
                        right=llir.Literal(1, llir.DataType.INT),
                    ),
                ),
            ),
            llir.BlankLine(),
        ]

    def _sparse_operand_loop(
        self,
        cursor: SparseCursorDecl,
        coordinate_name: str,
        body_tail: List[llir.Stmt],
    ) -> llir.ForLoop:
        """One dense-parented sparse operand loop resolving its coordinate."""

        position_name = self._cursor_position_name(cursor)
        position_var = llir.Var(name=position_name, type=llir.DataType.INT)
        body: List[llir.Stmt] = [
            llir.Comment("Resolve coordinates"),
            llir.VarInit(
                var=llir.Var(name=coordinate_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_crd_array(cursor),
                    index=llir.Var(name=position_name, type=llir.DataType.INT),
                ),
            ),
            llir.BlankLine(),
            *body_tail,
        ]
        for_loop = llir.ForLoop(
            init=llir.VarInit(
                var=llir.Var(name=position_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_pos_array(cursor),
                    index=llir.Var(
                        name=f"p{self.decls[cursor.tensor].name}0",
                        type=llir.DataType.INT,
                    ),
                ),
            ),
            cond=llir.BinOp(
                op="<",
                left=position_var,
                right=llir.Var(name=f"{position_name}_end", type=llir.DataType.INT),
            ),
            update=llir.Increment(
                var=llir.Var(name=position_name, type=llir.DataType.INT)
            ),
            body=body,
        )
        for_loop.scorch_index_var = coordinate_name
        return for_loop

    def _producer_statements(self, *, unchecked: bool = False) -> List[llir.Stmt]:
        """The workspace producer: reduction descent into the insertion.

        The serial emission inserts through the checked spelling; the
        completed two-phase reference reconstructs the pass-owned
        ``insert_unchecked`` rewrite of the same statement.
        """

        insert_name = (
            f"{self.workspace_decl.name}.insert_unchecked"
            if unchecked
            else f"{self.workspace_decl.name}.insert"
        )
        insert_statement = llir.FunctionCallStmt(
            name=insert_name,
            args=[
                llir.Array(
                    values=[
                        llir.Var(
                            name=self.drain_name,
                            type=llir.DataType.INT64,
                        )
                    ],
                    data_type=llir.DataType.INT64,
                ),
                self._sparse_value(self.sparse_insert.value),
            ],
        )
        child_cursor = self.child_loop.cursor
        child_tail: List[llir.Stmt] = [
            *self._dense_position_resolve(child_cursor, self.reduction_name),
            *self._sparse_segment_open(child_cursor),
            self._sparse_operand_loop(
                child_cursor,
                self.drain_name,
                [insert_statement],
            ),
        ]
        reduction_cursor = self.reduction_loop.cursor
        return [
            *self._sparse_segment_open(reduction_cursor),
            self._sparse_operand_loop(
                reduction_cursor,
                self.reduction_name,
                child_tail,
            ),
        ]

    def _workspace_init_statement(self) -> llir.VarInit:
        workspace = self.workspace_decl
        torch_dtype = _SCALAR_TO_TORCH[workspace.dtype]
        element_type = dtype_to_c_datatype(torch_dtype)
        return llir.VarInit(
            var=llir.Var(name=workspace.name, type=llir.DataType.AUTO),
            value=llir.FunctionCall(
                name=f"coo_workspace_1d<{element_type.value}, 1>",
                args=[llir.Literal(value=1024, data_type=llir.DataType.INT)],
            ),
        )

    def _drain_statements(self) -> List[llir.Stmt]:
        """``sort()`` plus the range-for drain writing indexed appends."""

        workspace_name = self.workspace_decl.name
        result_name = self.result_decl.name
        iterator_var = llir.Var(name="it", type=llir.DataType.CONST_AUTO_REF)
        workspace_value_name = f"{workspace_name}_value"
        counter_name = f"p{result_name}1"
        drain_loop = llir.ForLoopAuto(
            var=iterator_var,
            array=llir.Var(name=workspace_name, type=llir.DataType.AUTO),
            body=[
                llir.VarInit(
                    var=llir.Var(name=self.drain_name, type=llir.DataType.INT64),
                    value=llir.MemberAccess(base=iterator_var, member="first"),
                ),
                llir.VarInit(
                    var=llir.Var(
                        name=workspace_value_name,
                        type=dtype_to_c_datatype(
                            _SCALAR_TO_TORCH[self.workspace_decl.dtype]
                        ),
                    ),
                    value=llir.MemberAccess(base=iterator_var, member="second"),
                ),
                llir.BlankLine(),
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_name}_values",
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=llir.Var(name=counter_name, type=llir.DataType.INT64),
                        tensor_access=self._append_metadata(),
                    ),
                    value=llir.Var(
                        name=workspace_value_name,
                        type=llir.DataType.NO_TYPE,
                    ),
                ),
                llir.Assign(
                    var=llir.ArrayAccess(
                        array=llir.Var(
                            name=f"{result_name}1_crd",
                            type=llir.DataType.NO_TYPE,
                        ),
                        index=llir.Var(name=counter_name, type=llir.DataType.INT64),
                    ),
                    value=llir.Var(name=self.drain_name, type=llir.DataType.NO_TYPE),
                ),
                llir.Increment(
                    var=llir.Var(name=counter_name, type=llir.DataType.NO_TYPE)
                ),
            ],
        )
        return [
            llir.BlankLine(),
            llir.Comment("Lower consumer CIN"),
            llir.FunctionCallStmt(name=f"{workspace_name}.sort", args=[]),
            drain_loop,
        ]

    def raw_loop_statements(self) -> List[llir.Stmt]:
        _TargetLowering._require_program_graph_unchanged(self)
        result_name = self.result_decl.name
        row_name = self.row_name
        pos_index = llir.Var(
            name=f"{result_name}1_pos_index",
            type=llir.DataType.INT,
        )
        catch_up = llir.ForLoop(
            init=None,
            cond=llir.BinOp(
                op="<",
                left=pos_index,
                right=llir.Var(name=row_name, type=llir.DataType.INT),
            ),
            update=llir.Increment(
                var=llir.Var(
                    name=f"{result_name}1_pos_index",
                    type=llir.DataType.INT,
                )
            ),
            body=[self._row_close_statement()],
        )
        row_body: List[llir.Stmt] = [
            llir.Comment("Assemble COMPRESSED level"),
            catch_up,
            *self._dense_position_resolve(self.reduction_loop.cursor, row_name),
            llir.Comment("Resolve index into dense level of values array"),
            llir.VarInit(
                var=llir.Var(name=f"p{result_name}0", type=llir.DataType.INT),
                value=llir.Var(name=row_name, type=llir.DataType.INT),
            ),
            llir.Comment("Initialize workspaces"),
            self._workspace_init_statement(),
            *self._producer_statements(),
            *self._drain_statements(),
            llir.BlankLine(),
            llir.BlankLine(),
            llir.Comment("Assembly compressed _level indices"),
            self._row_close_statement(),
        ]
        row_var = llir.Var(name=row_name, type=llir.DataType.INT64)
        row_loop = llir.ForLoop(
            init=llir.VarInit(var=row_var, value=llir.Literal(0)),
            cond=llir.BinOp(
                op="<",
                left=row_var,
                right=llir.Var(
                    name=f"{self.decls[self.reduction_loop.cursor.tensor].name}0_size",
                    type=llir.DataType.INT,
                ),
            ),
            update=llir.Increment(var=row_var),
            body=row_body,
        )
        row_loop.scorch_index_var = row_name
        return [llir.BlankLine(), row_loop]

    def _row_close_statement(self) -> llir.Assign:
        """``C1_pos[C1_pos_index + 1] = C1_crd.size()`` (legacy spelling)."""

        result_name = self.result_decl.name
        return llir.Assign(
            var=llir.ArrayAccess(
                array=llir.Var(
                    name=f"{result_name}1_pos",
                    type=llir.DataType.STD_VECTOR_C_INT,
                ),
                index=llir.Add(
                    llir.Var(
                        name=f"{result_name}1_pos_index",
                        type=llir.DataType.INT,
                    ),
                    llir.Literal(1, llir.DataType.INT32),
                ),
            ),
            value=llir.FunctionCall(
                name=f"{result_name}1_crd.size",
                args=[],
            ),
        )

    # -- exact post-pass completion ---------------------------------------------

    def _policy_work_and_rows(self) -> Tuple[str, str, llir.Expr, llir.Var]:
        """The derived SpGEMM policy operands, in string and typed form.

        Both forms are constructed together from the same verified
        structural facts — the reduction operand's stored leaf count and
        the child operand's mean row fanout — so neither is recovered by
        parsing the other.
        """

        reduction_name = self.decls[self.reduction_loop.cursor.tensor].name
        child_name = self.decls[self.child_loop.cursor.tensor].name
        rows_text = f"{reduction_name}0_size"
        work_text = (
            f"(long){reduction_name}1_pos[{rows_text}] * "
            f"({child_name}0_size > 0 ? "
            f"{child_name}1_pos[{child_name}0_size] / {child_name}0_size + 1 : 1)"
        )
        rows_var = llir.Var(name=rows_text, type=llir.DataType.INT)

        def child_rows() -> llir.Var:
            return llir.Var(name=f"{child_name}0_size", type=llir.DataType.INT64)

        work_expr: llir.Expr = llir.Mul(
            llir.Cast(
                expr=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{reduction_name}1_pos",
                        type=llir.DataType.NO_TYPE,
                    ),
                    index=llir.Var(name=rows_text, type=llir.DataType.INT),
                ),
                data_type=llir.DataType.LONG,
            ),
            llir.Select(
                cond=llir.BinOp(
                    op=">",
                    left=child_rows(),
                    right=llir.Literal(0, llir.DataType.INT),
                ),
                when_true=llir.Add(
                    llir.BinOp(
                        op="/",
                        left=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{child_name}1_pos",
                                type=llir.DataType.NO_TYPE,
                            ),
                            index=child_rows(),
                        ),
                        right=child_rows(),
                    ),
                    llir.Literal(1, llir.DataType.INT),
                ),
                when_false=llir.Literal(1, llir.DataType.INT),
            ),
        )
        return work_text, rows_text, work_expr, rows_var

    def _completed_pool_statements(self) -> List[llir.Stmt]:
        """Pool sizing, reservation, and per-worker construction."""

        workspace_name = self.workspace_decl.name
        torch_dtype = _SCALAR_TO_TORCH[self.workspace_decl.dtype]
        pool_type = llir.DataType.linked_list_workspace_pool_type(
            dtype_to_c_datatype(torch_dtype).value
        )
        _, _, work_expr, rows_var = self._policy_work_and_rows()
        thread_count = llir.Var(
            name=f"{workspace_name}_thread_count",
            type=llir.DataType.INT,
        )
        worker = llir.Var(name="_worker", type=llir.DataType.INT)
        return [
            llir.VarInit(
                var=thread_count,
                value=llir.FunctionCall(
                    name="scorch_nthreads",
                    args=[
                        work_expr,
                        rows_var,
                        llir.Var(
                            name="SCORCH_GRAIN_CODEGEN_SPGEMM",
                            type=llir.DataType.NO_TYPE,
                        ),
                    ],
                ),
            ),
            llir.VarDecl(var=llir.Var(name=f"{workspace_name}_pool", type=pool_type)),
            llir.MemberCallStmt(
                base=llir.Var(name=f"{workspace_name}_pool", type=pool_type),
                member="reserve",
                args=(
                    llir.Cast(
                        expr=llir.Var(
                            name=f"{workspace_name}_thread_count",
                            type=llir.DataType.INT,
                        ),
                        data_type=llir.DataType.SIZE_T,
                    ),
                ),
            ),
            llir.ForLoop(
                init=llir.VarInit(
                    var=llir.Var(name="_worker", type=llir.DataType.INT),
                    value=llir.Literal(0, llir.DataType.INT),
                ),
                cond=llir.BinOp(
                    op="<",
                    left=worker,
                    right=llir.Var(
                        name=f"{workspace_name}_thread_count",
                        type=llir.DataType.INT,
                    ),
                ),
                update=llir.Increment(
                    var=llir.Var(name="_worker", type=llir.DataType.INT)
                ),
                body=[
                    llir.MemberCallStmt(
                        base=llir.Var(name=f"{workspace_name}_pool", type=pool_type),
                        member="emplace_back",
                        args=(
                            llir.ArrayAccess(
                                array=llir.Var(
                                    name="result_shape",
                                    type=llir.DataType.STD_VECTOR_INT,
                                ),
                                index=llir.Literal(1, llir.DataType.INT64),
                            ),
                            llir.Literal(True, llir.DataType.BOOL),
                        ),
                    )
                ],
            ),
        ]

    def _completed_view_statement(self) -> llir.VarInit:
        """The borrowed per-worker workspace view opening each phase region."""

        workspace_name = self.workspace_decl.name
        torch_dtype = _SCALAR_TO_TORCH[self.workspace_decl.dtype]
        pool_type = llir.DataType.linked_list_workspace_pool_type(
            dtype_to_c_datatype(torch_dtype).value
        )
        return llir.VarInit(
            var=llir.Var(name=workspace_name, type=llir.DataType.AUTO),
            value=llir.MemberCall(
                base=llir.ArrayAccess(
                    array=llir.Var(name=f"{workspace_name}_pool", type=pool_type),
                    index=llir.Cast(
                        expr=llir.FunctionCall(name="omp_get_thread_num", args=()),
                        data_type=llir.DataType.SIZE_T,
                    ),
                ),
                member="make_view",
            ),
        )

    def _completed_phase_loop(self, body: List[llir.Stmt]) -> llir.ForLoop:
        """One completed parallel phase loop around one phase body."""

        work_text, rows_text, _, _ = self._policy_work_and_rows()
        grain = "SCORCH_GRAIN_CODEGEN_SPGEMM"
        row_var = llir.Var(name=self.row_name, type=llir.DataType.INT64)
        loop = llir.ForLoop(
            init=llir.VarInit(var=row_var, value=llir.Literal(0)),
            cond=llir.BinOp(
                op="<",
                left=llir.Var(name=self.row_name, type=llir.DataType.INT64),
                right=llir.Var(name=rows_text, type=llir.DataType.INT),
            ),
            update=llir.Increment(
                var=llir.Var(name=self.row_name, type=llir.DataType.INT64)
            ),
            body=body,
            omp_parallel_for=True,
            omp_schedule="dynamic, 64",
            pre_parallel_body=[self._completed_view_statement()],
            omp_num_threads=f"scorch_nthreads({work_text}, {rows_text}, {grain})",
            omp_chunk_expr=f"scorch_chunk({rows_text}, {work_text}, {grain})",
        )
        return loop

    def _completed_phase_shared_statements(self) -> List[llir.Stmt]:
        """The row statements both completed phases share."""

        result_name = self.result_decl.name
        return [
            llir.Comment("Assemble COMPRESSED level"),
            *self._dense_position_resolve(self.reduction_loop.cursor, self.row_name),
            llir.Comment("Resolve index into dense level of values array"),
            llir.VarInit(
                var=llir.Var(name=f"p{result_name}0", type=llir.DataType.INT),
                value=llir.Var(name=self.row_name, type=llir.DataType.INT),
            ),
            llir.Comment("Initialize workspaces"),
            *self._producer_statements(unchecked=True),
            llir.BlankLine(),
            llir.Comment("Lower consumer CIN"),
        ]

    def _completed_drain_bindings(self) -> List[llir.Stmt]:
        iterator_var = llir.Var(name="it", type=llir.DataType.CONST_AUTO_REF)
        return [
            llir.VarInit(
                var=llir.Var(name=self.drain_name, type=llir.DataType.INT64),
                value=llir.MemberAccess(base=iterator_var, member="first"),
            ),
            llir.VarInit(
                var=llir.Var(
                    name=f"{self.workspace_decl.name}_value",
                    type=dtype_to_c_datatype(
                        _SCALAR_TO_TORCH[self.workspace_decl.dtype]
                    ),
                ),
                value=llir.MemberAccess(base=iterator_var, member="second"),
            ),
            llir.BlankLine(),
        ]

    def _completed_drain_loop(self, body_tail: List[llir.Stmt]) -> llir.ForLoopAuto:
        return llir.ForLoopAuto(
            var=llir.Var(name="it", type=llir.DataType.CONST_AUTO_REF),
            array=llir.Var(name=self.workspace_decl.name, type=llir.DataType.AUTO),
            body=[*self._completed_drain_bindings(), *body_tail],
        )

    def _completed_clear_statement(self) -> llir.MemberCallStmt:
        return llir.MemberCallStmt(
            base=llir.Var(name=self.workspace_decl.name, type=llir.DataType.NO_TYPE),
            member="clear",
        )

    def _completed_count_loop(self) -> llir.ForLoop:
        counter = llir.Var(name="_cnt1", type=llir.DataType.INT)
        body: List[llir.Stmt] = [
            llir.VarInit(var=counter, value=llir.Literal(0, llir.DataType.INT)),
            *self._completed_phase_shared_statements(),
            self._completed_drain_loop(
                [llir.Increment(var=llir.Var(name="_cnt1", type=llir.DataType.INT))]
            ),
            llir.BlankLine(),
            llir.BlankLine(),
            llir.Comment("Assembly compressed _level indices"),
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name="_count1",
                        type=llir.DataType.STD_VECTOR_C_INT,
                    ),
                    index=llir.Var(name=self.row_name, type=llir.DataType.INT64),
                ),
                value=llir.Var(name="_cnt1", type=llir.DataType.INT),
            ),
            self._completed_clear_statement(),
        ]
        return self._completed_phase_loop(body)

    def _completed_fill_loop(self) -> llir.ForLoop:
        result_name = self.result_decl.name

        def offset_index() -> llir.Add:
            return llir.Add(
                llir.Var(name="_base1", type=llir.DataType.INT64),
                llir.Var(name="_pos1", type=llir.DataType.INT),
            )

        body: List[llir.Stmt] = [
            llir.VarInit(
                var=llir.Var(name="_base1", type=llir.DataType.INT64),
                value=llir.ArrayAccess(
                    array=llir.Var(
                        name="_offset1",
                        type=llir.DataType.STD_VECTOR_INT,
                    ),
                    index=llir.Var(name=self.row_name, type=llir.DataType.INT64),
                ),
            ),
            llir.VarInit(
                var=llir.Var(name="_pos1", type=llir.DataType.INT),
                value=llir.Literal(0, llir.DataType.INT),
            ),
            *self._completed_phase_shared_statements(),
            llir.FunctionCallStmt(name=f"{self.workspace_decl.name}.sort", args=[]),
            self._completed_drain_loop(
                [
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{result_name}_values_data",
                                type=llir.DataType.ptr_type(
                                    _SCALAR_TO_TORCH[self.result_decl.dtype]
                                ),
                            ),
                            index=offset_index(),
                        ),
                        value=llir.Var(
                            name=f"{self.workspace_decl.name}_value",
                            type=llir.DataType.NO_TYPE,
                        ),
                    ),
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name=f"{result_name}1_crd_data",
                                type=llir.DataType.PTR_INT,
                            ),
                            index=offset_index(),
                        ),
                        value=llir.Var(
                            name=self.drain_name,
                            type=llir.DataType.NO_TYPE,
                        ),
                    ),
                    llir.Increment(var=llir.Var(name="_pos1", type=llir.DataType.INT)),
                ]
            ),
            llir.BlankLine(),
            llir.BlankLine(),
            llir.Comment("Assembly compressed _level indices"),
            self._completed_clear_statement(),
        ]
        return self._completed_phase_loop(body)

    def _completed_interlude_statements(self) -> List[llir.Stmt]:
        """The serial prefix sum and exact Torch-owned output allocation."""

        _, rows_text, _, _ = self._policy_work_and_rows()
        rows_int = llir.Var(name=rows_text, type=llir.DataType.INT)
        index = llir.Var(name="_i", type=llir.DataType.INT)
        assembler = self.result_assembler()
        offsets = llir.Var(name="_offset1", type=llir.DataType.STD_VECTOR_INT)
        total = llir.Var(name="_total1", type=llir.DataType.INT64)
        return [
            llir.DirectInit(
                var=offsets,
                args=[
                    llir.Add(
                        llir.Cast(expr=rows_int, data_type=llir.DataType.SIZE_T),
                        llir.Literal(1, llir.DataType.INT),
                    )
                ],
            ),
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(name="_offset1", type=llir.DataType.STD_VECTOR_INT),
                    index=llir.Literal(0, llir.DataType.INT),
                ),
                value=llir.Literal(0, llir.DataType.INT),
            ),
            llir.ForLoop(
                init=llir.VarInit(
                    var=llir.Var(name="_i", type=llir.DataType.INT),
                    value=llir.Literal(0, llir.DataType.INT),
                ),
                cond=llir.BinOp(op="<", left=index, right=rows_int),
                update=llir.Increment(var=llir.Var(name="_i", type=llir.DataType.INT)),
                body=[
                    llir.Assign(
                        var=llir.ArrayAccess(
                            array=llir.Var(
                                name="_offset1",
                                type=llir.DataType.STD_VECTOR_INT,
                            ),
                            index=llir.Add(index, llir.Literal(1, llir.DataType.INT)),
                        ),
                        value=llir.Add(
                            llir.ArrayAccess(
                                array=llir.Var(
                                    name="_offset1",
                                    type=llir.DataType.STD_VECTOR_INT,
                                ),
                                index=index,
                            ),
                            llir.ArrayAccess(
                                array=llir.Var(
                                    name="_count1",
                                    type=llir.DataType.STD_VECTOR_C_INT,
                                ),
                                index=index,
                            ),
                        ),
                    )
                ],
            ),
            llir.VarInit(
                var=total,
                value=llir.ArrayAccess(
                    array=llir.Var(name="_offset1", type=llir.DataType.STD_VECTOR_INT),
                    index=rows_int,
                ),
            ),
            *assembler.emit_first_compressed_position_allocation(
                llir.Var(name=rows_text, type=llir.DataType.INT),
                llir.Var(name="_offset1", type=llir.DataType.STD_VECTOR_INT),
            ),
            *assembler.emit_compressed_coordinate_allocations(
                (llir.Var(name="_total1", type=llir.DataType.INT64),)
            ),
            *assembler.emit_compressed_value_allocation(
                llir.Var(name="_total1", type=llir.DataType.INT64)
            ),
        ]

    def complete_sparse_workspace(self, function: llir.Function) -> llir.Function:
        """Require the exact two-phase composition owned by the shared pass.

        The completed function is reconstructed from the verified program
        facts using only locally owned constructions and the frozen result
        ABI snapshot — never the managed pass being validated — so a
        missing, duplicated, moved, wrapped, aliased, malformed, or
        cyclically shared state anywhere in the assembled function fails
        closed with the family's stable completion code.
        """

        try:
            assembler = self.result_assembler()
            expected_kernel_abi = self.kernel_abi()
            expected_size_stmts = self.result_size_inits()
            if expected_size_stmts:
                expected_size_stmts = [
                    llir.Comment("Init result tensor level sizes"),
                    *expected_size_stmts,
                ]
            counter_var = llir.Var(name="_count1", type=llir.DataType.STD_VECTOR_C_INT)
            _, rows_text, _, _ = self._policy_work_and_rows()
            expected_body: List[llir.Stmt] = [
                *expected_kernel_abi.emit_validation(),
                *expected_size_stmts,
                *expected_kernel_abi.emit_input_prologue(),
                llir.BlankLine(),
                *self.tile_size_inits(),
                llir.BlankLine(),
                llir.BlankLine(),
                *self._completed_pool_statements(),
                llir.DirectInit(
                    var=counter_var,
                    args=[
                        llir.Cast(
                            expr=llir.Var(name=rows_text, type=llir.DataType.INT),
                            data_type=llir.DataType.SIZE_T,
                        ),
                        llir.Literal(0, llir.DataType.INT),
                    ],
                ),
                self._completed_count_loop(),
                *self._completed_interlude_statements(),
                self._completed_fill_loop(),
                assembler.emit_result_declaration(),
                *assembler.emit_storage_epilogue(),
            ]
            expected_function = expected_kernel_abi.assemble_function(expected_body)
        except (
            LLIRTraversalError,
            AttributeError,
            RecursionError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            _fail(_SPARSE_WORKSPACE_LOST, str(error))
        try:
            completion_matches = _exact_sparse_completion_matches(
                function, expected_function
            )
        except LoopIRTargetError:
            raise
        except Exception as error:
            _fail(_SPARSE_WORKSPACE_LOST, str(error))
        if not completion_matches:
            _fail(
                _SPARSE_WORKSPACE_LOST,
                "the assembled function must exactly match the completed "
                "two-phase parallel sparse-workspace target, including the "
                "derived pool policy, both phase regions, the exact "
                "allocation interlude, and honest final assembly",
            )
        return function


def _dense_domain_mixed_chain(program: LoopProgram) -> bool:
    """Whether the program is the dense-domain mixed dense-leaf chain.

    Routing is purely structural: a nest of single-statement dense loops
    over one :class:`AppendEntry` leaf assembling a compressed-parent/
    dense-suffix result.  Everything else stays on the general target
    lowering and keeps its existing fail-closed boundaries.
    """

    if len(program.outputs) != 1:
        return False
    result_decl = next(
        (decl for decl in program.tensors if decl.symbol == program.outputs[0]),
        None,
    )
    if (
        result_decl is None
        or len(result_decl.levels) < 2
        or result_decl.levels[0].kind is not LevelKind.COMPRESSED
        or any(level.kind is not LevelKind.DENSE for level in result_decl.levels[1:])
    ):
        return False
    body: Stmt = program.body
    depth = 0
    while type(body) is Block and len(body.statements) == 1:
        only = body.statements[0]
        if type(only) is DenseFor:
            body = only.body
            depth += 1
            continue
        return type(only) is AppendEntry and depth > 0
    return False


class _DenseDomainMixedLowering(_TargetLowering):
    """Dedicated target lowering for the B2 mixed dense-leaf family.

    Admits exactly the dense-domain assembly form ``lower_normalized_cin``
    produces for compressed-parent/dense-suffix results: an all-dense
    single-statement nest, one loop per result coordinate in identity
    storage order, appending one value per innermost iteration.  The
    parent coordinate materializes exactly when the dense suffix has
    nonzero extent, matching the canonical level-storage contract for
    zero-extent trailing dense levels; assembly appends the parent
    coordinate once per materialized row and closes the root position
    after the nest.

    The retained legacy comparand for this family produces malformed
    storage: it appends values but never appends their compressed-parent
    coordinates.  There is therefore deliberately no byte-parity gate:
    the family is proven against the production LoopIR oracle and the
    PyTorch dense reference, and the emission is correct-by-construction
    in the established target style (``push_back``/``emplace_back`` call
    statements, ``scorch_vector_set`` position closes).  The separately
    retained sparse-reduction comparand owns the memory-safety failure; it
    is not evidence for this dense-domain family.
    """

    def __init__(
        self,
        program: LoopProgram,
        input_shapes: Mapping[SymbolId, Tuple[int, ...]],
        result_shape: Tuple[int, ...],
    ) -> None:
        self.program = program
        self._input_symbols, seal_token = self._snapshot_program_inputs(program)
        self.decls = {decl.symbol: decl for decl in program.tensors}
        if len(program.outputs) != 1:
            _fail(
                "unsupported_program_shape",
                "this target lowering supports exactly one output tensor",
            )
        self.result_symbol = program.outputs[0]
        self.result_decl = self.decls[self.result_symbol]
        self.result_is_dense = False
        self.sparse_program = True
        self.dimension_names = {}
        self._access_ids = {}
        self.region = None
        self.panel = None
        self.relayout = None
        self.result_tile = None
        self.result_tile_depth = -1
        self._tiled_leaf = None
        self._tiled_view = None
        self.relayout_depth = -1
        if program.parallel is not None:
            _fail(
                "unsupported_program_shape",
                "the mixed dense-leaf family owns no parallel selection",
            )
        self.parallel = None
        self._validate_display_names()
        self.shapes = self._validate_shapes(input_shapes, result_shape)
        self.loops = self._collect_mixed_chain()
        self._validate_loop_variable_names()
        self.loop_positions = {
            loop.index: position for position, loop in enumerate(self.loops)
        }
        self.cursor_loops: Dict[CursorId, int] = {}
        self._bound_position_snapshot = self._validated_bound_position_bindings()
        self._position_load_signatures = {}
        self.loads, self.cursor_values = self._collect_accesses()
        self._value_expression_snapshot = self._validated_value_expression_signature(
            self._access_value_expression()
        )
        self._target_owner_snapshot = self._validated_target_owner_signature()
        self.level_drivers = self._compute_level_drivers()
        self._validate_access_orders()
        self._seal_target_state(seal_token)

    def _collect_mixed_chain(self) -> List[_Loop]:
        def require(condition: bool, what: str) -> None:
            if not condition:
                _fail(
                    "unsupported_program_shape",
                    f"the mixed dense-leaf target requires {what}",
                )

        result_levels = self.result_decl.levels
        require(
            len(result_levels) >= 2
            and result_levels[0].kind is LevelKind.COMPRESSED
            and all(level.kind is LevelKind.DENSE for level in result_levels[1:])
            and tuple(level.mode for level in result_levels)
            == tuple(range(len(result_levels))),
            "an identity-ordered compressed-parent/dense-suffix result",
        )
        for symbol in self._input_symbols:
            require(
                all(
                    level.kind is LevelKind.DENSE for level in self.decls[symbol].levels
                ),
                "all-dense inputs (mixed dense-leaf operands are a distinct " "gap)",
            )
        loops: List[_Loop] = []
        body: Stmt = self.program.body
        while True:
            require(
                type(body) is Block and len(cast(Block, body).statements) == 1,
                "a single-statement dense loop nest",
            )
            only = cast(Block, body).statements[0]
            if type(only) is DenseFor:
                loops.append(_Loop(_DENSE, only.index, only.dimension, only, ()))
                body = only.body
                continue
            require(type(only) is AppendEntry, "an ordered append leaf")
            append = cast(AppendEntry, only)
            break
        require(
            len(loops) == len(result_levels),
            "one dense loop per result coordinate",
        )
        require(
            append.tensor == self.result_symbol
            and len(append.coords) == len(result_levels)
            and all(type(coord) is IndexValue for coord in append.coords)
            and tuple(cast(IndexValue, coord).index for coord in append.coords)
            == tuple(loop.index for loop in loops),
            "appends of exactly the nest coordinates in order",
        )
        self.leaf = append
        return loops

    def result_size_inits(self) -> List[llir.Stmt]:
        """Initialize every result extent the mixed nest may use as a bound.

        The generic target needs size variables only for dense result levels:
        sparse loops obtain their bounds from position arrays.  B2 is
        different—the compressed parent is intentionally traversed as a dense
        domain before its coordinates are assembled.  A broadcast expression
        may have no input level driving that parent dimension, so omitting its
        result size would leave (for example) ``C0_size`` undeclared in the
        emitted loop header.
        """

        return [
            llir.VarInit(
                llir.Var(
                    name=f"{self.result_decl.name}{level}_size",
                    type=llir.DataType.INT64,
                ),
                value=llir.ArrayAccess(
                    array=llir.Var(
                        name="result_shape",
                        type=llir.DataType.STD_VECTOR_INT,
                    ),
                    index=llir.Literal(
                        value=level,
                        data_type=llir.DataType.INT64,
                    ),
                ),
            )
            for level in range(len(self.result_decl.levels))
        ]

    def _suffix_guard(self) -> llir.Expr:
        """``C1_size > 0 [&& ...]`` over every trailing dense level."""

        result_name = self.result_decl.name
        guards: List[llir.Expr] = [
            llir.BinOp(
                op=">",
                left=llir.Var(
                    name=f"{result_name}{level}_size",
                    type=llir.DataType.INT64,
                ),
                right=llir.Literal(0, llir.DataType.INT),
            )
            for level in range(1, len(self.result_decl.levels))
        ]
        guard = guards[0]
        for clause in guards[1:]:
            guard = llir.BinOp(op="&&", left=guard, right=clause)
        return guard

    def _lower_mixed_dense(self, position: int) -> llir.ForLoop:
        loop = self.loops[position]
        name = self._loop_var_name(loop)
        body: List[llir.Stmt] = []
        input_resolves = self._input_resolves_at(loop)
        if input_resolves:
            body.append(llir.Comment("Resolve dense coordinates"))
            body.extend(input_resolves)
        if position == 0:
            body.append(llir.Comment("Assembly compressed _level indices"))
            body.append(
                llir.IfThenElse(
                    cond=self._suffix_guard(),
                    then_body=[
                        llir.FunctionCallStmt(
                            name=f"{self.result_decl.name}0_crd.push_back",
                            args=[llir.Var(name=name, type=llir.DataType.INT64)],
                        ),
                    ],
                )
            )
        if position + 1 < len(self.loops):
            body.append(llir.BlankLine())
            body.append(self._lower_mixed_dense(position + 1))
        else:
            body.append(
                llir.FunctionCallStmt(
                    name=f"{self.result_decl.name}_values.emplace_back",
                    args=[self._lower_value(cast(AppendEntry, self.leaf).value)],
                )
            )
        loop_var = llir.Var(name=name, type=llir.DataType.INT64)
        for_loop = llir.ForLoop(
            init=llir.VarInit(var=loop_var, value=llir.Literal(0)),
            cond=llir.BinOp(op="<", left=loop_var, right=self._loop_bound_var(loop)),
            update=llir.Increment(var=loop_var),
            body=body,
        )
        for_loop.scorch_index_var = name
        return for_loop

    def raw_loop_statements(self) -> List[llir.Stmt]:
        _TargetLowering._require_value_expression_unchanged(self)
        result_name = self.result_decl.name
        return [
            llir.BlankLine(),
            self._lower_mixed_dense(0),
            llir.BlankLine(),
            llir.Comment("Assembly compressed _level indices"),
            llir.Assign(
                var=llir.ArrayAccess(
                    array=llir.Var(
                        name=f"{result_name}0_pos",
                        type=llir.DataType.STD_VECTOR_C_INT,
                    ),
                    index=llir.Add(
                        llir.Var(
                            name=f"{result_name}0_pos_index",
                            type=llir.DataType.INT64,
                        ),
                        llir.Literal(1),
                    ),
                ),
                value=llir.FunctionCall(name=f"{result_name}0_crd.size", args=[]),
            ),
        ]


def _multi_compressed_assembly_chain(program: LoopProgram) -> bool:
    """Whether the program is a multi-compressed assembly chain.

    Routing is purely structural: a single-statement nest of dense loops
    over nested stream loops (single-cursor sparse or two-cursor merged)
    appending into a dense-prefix/multi-compressed-suffix result, where the
    suffix is two-or-more compressed levels or the degenerate rank-1
    all-compressed result (one stream, no dense prefix, no parent level).
    Everything else stays on its existing route and keeps its fail-closed
    boundaries.
    """

    if len(program.outputs) != 1:
        return False
    result_decl = next(
        (decl for decl in program.tensors if decl.symbol == program.outputs[0]),
        None,
    )
    if result_decl is None:
        return False
    kinds = tuple(level.kind for level in result_decl.levels)
    compressed_suffix = 0
    while (
        compressed_suffix < len(kinds)
        and kinds[-1 - compressed_suffix] is LevelKind.COMPRESSED
    ):
        compressed_suffix += 1
    # Two or more compressed suffix levels, or -- degenerately -- a rank-1
    # all-compressed result: the same ordered stream assembly with no dense
    # prefix and no parent level to close.
    ordered_compressed_suffix = (
        compressed_suffix >= 2 or len(kinds) == compressed_suffix == 1
    )
    if not ordered_compressed_suffix or any(
        kind is not LevelKind.DENSE for kind in kinds[: len(kinds) - compressed_suffix]
    ):
        return False
    body: Stmt = program.body
    streams = 0
    while type(body) is Block and len(body.statements) == 1:
        only = body.statements[0]
        if type(only) is DenseFor and streams == 0:
            body = only.body
            continue
        if type(only) is MergedSparseFor:
            streams += 1
            body = only.body
            continue
        if type(only) is SparseFor:
            streams += 1
            body = only.body
            continue
        return type(only) is AppendEntry and (
            streams >= 2 or streams == len(kinds) == 1
        )
    return False


class _MultiCompressedAssemblyLowering(_TargetLowering):
    """Dedicated target lowering for the multi-compressed assembly families.

    Admits exactly the assembly forms ``lower_normalized_cin`` produces for
    dense-prefix/multi-compressed-suffix results — including the degenerate
    rank-1 all-compressed result, which is one stream loop with no dense
    prefix and no parent level to close: at most one dense prefix
    loop, then one stream loop per compressed result level — a single-cursor
    :class:`SparseFor` over one stored stream (dense co-operands are read at
    their resolved coordinates), or a two-cursor :class:`MergedSparseFor`.
    INTERSECTION binds both aligned cursor positions; UNION additionally
    lowers the aligned, one-sided, and post-exhaustion cases while treating an
    unaligned parent as an empty child stream.  Both forms descend to one
    ordered :class:`AppendEntry` leaf.  Anything else fails closed with
    ``unsupported_program_shape``.

    The raw emission mirrors the legacy generic lowering statement-for-
    statement — per-level ``Initialize iterators`` groups (root-parent
    subscripts folded to exact integers), the while-merge with
    ``std::min`` coordinate resolution, checked leaf appends and position
    writes in the exact form the shared dynamic-vector pass would produce,
    one conditional compressed-parent append plus child position close per
    structural level, the dense-prefix catch-up, and the root position
    close — so the generated C++ is byte-identical to the legacy pipeline's
    output for both automatic policy arms.  Building those mutations safely
    at the owning target boundary makes correctness independent of that
    generic rewrite while preserving its pipeline stage.  The legacy
    comparand is honest for this family (empty child intersections suppress
    their parent coordinates and cascade), so byte parity is the gate,
    exactly like B1.
    """

    def __init__(
        self,
        program: LoopProgram,
        input_shapes: Mapping[SymbolId, Tuple[int, ...]],
        result_shape: Tuple[int, ...],
    ) -> None:
        self.program = program
        self._input_symbols, seal_token = self._snapshot_program_inputs(program)
        self.decls = {decl.symbol: decl for decl in program.tensors}
        if len(program.outputs) != 1:
            _fail(
                "unsupported_program_shape",
                "this target lowering supports exactly one output tensor",
            )
        self.result_symbol = program.outputs[0]
        self.result_decl = self.decls[self.result_symbol]
        self.result_is_dense = False
        self.sparse_program = True
        self.dimension_names = {}
        self._access_ids = {}
        self.region = None
        self.panel = None
        self.relayout = None
        self.result_tile = None
        self.result_tile_depth = -1
        self._tiled_leaf = None
        self._tiled_view = None
        self.relayout_depth = -1
        if program.parallel is not None:
            _fail(
                "unsupported_program_shape",
                "the multi-compressed assembly family owns no parallel " "selection",
            )
        self.parallel = None
        self._validate_display_names()
        self._validate_assembly_layouts()
        self.shapes = self._validate_shapes(input_shapes, result_shape)
        self.loops = self._collect_assembly_chain()
        self._validate_loop_variable_names()
        self.loop_positions = {
            loop.index: position for position, loop in enumerate(self.loops)
        }
        self.cursor_loops: Dict[CursorId, int] = {}
        for position, loop in enumerate(self.loops):
            for cursor in loop.cursors:
                self.cursor_loops[cursor.cursor] = position
        self._bound_position_snapshot = self._validated_bound_position_bindings()
        self._position_load_signatures = {}
        self.loads, self.cursor_values = self._collect_accesses()
        self._value_expression_snapshot = self._validated_value_expression_signature(
            self._access_value_expression()
        )
        self._target_owner_snapshot = self._validated_target_owner_signature()
        self.level_drivers = self._compute_level_drivers()
        self._validate_access_orders()
        self._reserve_merge_names()
        self._seal_target_state(seal_token)

    def _validate_assembly_layouts(self) -> None:
        for decl in self.program.tensors:
            levels = decl.levels
            if type(levels) is not tuple:
                _fail(
                    "unsupported_mode_order",
                    f"tensor {decl.name!r} has malformed level declarations",
                )
            modes: List[object] = []
            kinds: List[object] = []
            for level in levels:
                if type(level) is not LevelDecl:
                    _fail(
                        "unsupported_mode_order",
                        f"tensor {decl.name!r} has malformed level declarations",
                    )
                state = object.__getattribute__(level, "__dict__")
                if (
                    type(state) is not dict
                    or any(type(key) is not str for key in state)
                    or set(state) != {"node_id", "kind", "mode"}
                    or _stored_identity_value(state["node_id"], LoopIRNodeId) is None
                    or (
                        state["kind"] is not LevelKind.DENSE
                        and state["kind"] is not LevelKind.COMPRESSED
                    )
                ):
                    _fail(
                        "unsupported_mode_order",
                        f"tensor {decl.name!r} has malformed level declarations",
                    )
                modes.append(state["mode"])
                kinds.append(state["kind"])
            if any(type(mode) is not int for mode in modes) or sorted(
                cast(List[int], modes)
            ) != list(range(len(levels))):
                _fail(
                    "unsupported_mode_order",
                    f"tensor {decl.name!r} must use one exact permutation of "
                    "its logical modes",
                )
            if tuple(modes) == tuple(range(len(levels))):
                continue
            if any(kind is not LevelKind.DENSE for kind in kinds):
                _fail(
                    "unsupported_mode_order",
                    f"tensor {decl.name!r} permutes compressed structure, "
                    "which the migrated families do not cover",
                )

    def _collect_assembly_chain(self) -> List[_Loop]:
        def require(condition: bool, what: str) -> None:
            if not condition:
                _fail(
                    "unsupported_program_shape",
                    f"the multi-compressed assembly target requires {what}",
                )

        result_levels = self.result_decl.levels
        compressed_suffix = 0
        while (
            compressed_suffix < len(result_levels)
            and result_levels[-1 - compressed_suffix].kind is LevelKind.COMPRESSED
        ):
            compressed_suffix += 1
        prefix = len(result_levels) - compressed_suffix
        # The degenerate rank-1 all-compressed result is one stored stream
        # with no dense prefix and no parent level to close.
        ordered_compressed_suffix = (
            compressed_suffix >= 2 or len(result_levels) == compressed_suffix == 1
        )
        require(
            ordered_compressed_suffix
            and all(level.kind is LevelKind.DENSE for level in result_levels[:prefix]),
            "a dense-prefix/multi-compressed-suffix result",
        )
        require(
            prefix <= 1,
            "at most one dense prefix loop (deeper dense parents are a "
            "distinct gap)",
        )
        loops: List[_Loop] = []
        body: Stmt = self.program.body
        while True:
            require(
                type(body) is Block and len(cast(Block, body).statements) == 1,
                "a single-statement assembly loop nest",
            )
            only = cast(Block, body).statements[0]
            if type(only) is DenseFor:
                require(
                    not loops,
                    "the dense prefix loop to precede every stream loop",
                )
                require(prefix == 1, "no dense loop over an all-compressed result")
                loops.append(_Loop(_DENSE, only.index, only.dimension, only, ()))
                body = only.body
                continue
            if type(only) is SparseFor:
                require(
                    len(loops) >= prefix,
                    "stream loops to follow the dense prefix",
                )
                cursor = only.cursor
                loops.append(
                    _Loop(
                        _SPARSE,
                        only.coord_index,
                        self._level_dimension(cursor.tensor, cursor.level),
                        only,
                        (cursor,),
                    )
                )
                body = only.body
                continue
            if type(only) is MergedSparseFor:
                require(
                    len(loops) >= prefix,
                    "stream loops to follow the dense prefix",
                )
                require(
                    only.mode in (MergeMode.INTERSECTION, MergeMode.UNION)
                    and len(only.cursors) == 2
                    and len(only.positions) == 2
                    and all(bound is not None for bound in only.positions),
                    "two-cursor merges binding both aligned positions",
                )
                first = only.cursors[0]
                loops.append(
                    _Loop(
                        _MERGED,
                        only.coord_index,
                        self._level_dimension(first.tensor, first.level),
                        only,
                        tuple(only.cursors),
                    )
                )
                body = only.body
                continue
            require(
                type(only) is AppendEntry,
                "an ordered append leaf",
            )
            require(
                len(loops) == len(result_levels)
                and sum(1 for loop in loops if loop.kind in (_MERGED, _SPARSE))
                == compressed_suffix,
                "one loop per result level with one stream loop per "
                "compressed level",
            )
            union_pairs = {
                tuple(cursor.tensor for cursor in loop.cursors)
                for loop in loops
                if loop.kind is _MERGED
                and cast(MergedSparseFor, loop.node).mode is MergeMode.UNION
            }
            if union_pairs:
                # One-sided union descent drains a single operand's whole
                # subtree, so every stream level must unite the same two
                # operands in the same order; mixing united levels with
                # single-cursor or intersected levels has no sound
                # one-sided child iterator.
                require(
                    len(union_pairs) == 1
                    and all(
                        loop.kind is _MERGED
                        and cast(MergedSparseFor, loop.node).mode is MergeMode.UNION
                        for loop in loops
                        if loop.kind is not _DENSE
                    ),
                    "united assemblies to unite the same two operands at "
                    "every compressed level",
                )
                # The union family's proven envelope is the elementwise sum
                # of exactly the two united operands; wider expressions
                # (further factors or nested operations) stay fail-closed
                # rather than approximated.
                value = cast(AppendEntry, only).value
                require(
                    type(value) is BinaryExpr
                    and value.op is BinaryOp.ADD
                    and type(value.lhs) is CursorValue
                    and type(value.rhs) is CursorValue,
                    "a united leaf to append exactly the sum of its two "
                    "united operand reads",
                )
            self.leaf = only
            return loops

    def _parent_append_statements(self, level: int) -> List[llir.Stmt]:
        """One conditional parent append plus child position close.

        ``level`` is the parent result level whose coordinate materializes
        exactly when the child level appended entries; the child position
        array then closes at the parent coordinate count — the exact
        legacy conditional-assembly spelling.
        """

        result_name = self.result_decl.name
        coord_name = self._loop_var_name(self.loops[level])
        child_position_index = llir.FunctionCall(
            name=f"{result_name}{level}_crd.size", args=[]
        )
        child_coordinate_count = llir.FunctionCall(
            name=f"{result_name}{level + 1}_crd.size", args=[]
        )
        child_position_close = llir.FunctionCallStmt(
            name="scorch_vector_set",
            args=[
                llir.Var(
                    name=f"{result_name}{level + 1}_pos",
                    type=llir.DataType.NO_TYPE,
                ),
                child_position_index,
                child_coordinate_count,
            ],
        )
        return [
            llir.IfThenElse(
                cond=llir.BinOp(
                    op="<",
                    left=llir.FunctionCall(
                        name=f"{result_name}{level + 1}_pos.back", args=[]
                    ),
                    right=llir.Var(
                        name=f"p{result_name}{level + 1}",
                        type=llir.DataType.INT64,
                    ),
                ),
                then_body=[
                    llir.FunctionCallStmt(
                        name=f"{result_name}{level}_crd.push_back",
                        args=[llir.Var(name=coord_name, type=llir.DataType.INT64)],
                    ),
                    llir.Increment(
                        var=llir.Var(
                            name=f"p{result_name}{level}",
                            type=llir.DataType.INT64,
                        )
                    ),
                ],
            ),
            child_position_close,
        ]

    def _merged_case_stmts(
        self, loop: _Loop, aligned: Set[CursorId]
    ) -> Optional[List[llir.Stmt]]:
        position = self.loop_positions[loop.index]
        if position == len(self.loops) - 1:
            leaf_stmts = super()._merged_case_stmts(loop, aligned)
            if leaf_stmts is None:
                return leaf_stmts
            result_name = self.result_decl.name
            leaf_level = len(self.result_decl.levels) - 1
            if (
                len(leaf_stmts) != 4
                or type(leaf_stmts[0]) is not llir.Assign
                or type(leaf_stmts[1]) is not llir.Comment
                or type(leaf_stmts[2]) is not llir.Assign
                or type(leaf_stmts[3]) is not llir.Increment
            ):
                _fail(
                    "unsupported_program_shape",
                    "the multi-compressed leaf no longer has the canonical "
                    "value/coordinate append shape",
                )
            value_store = cast(llir.Assign, leaf_stmts[0])
            coordinate_store = cast(llir.Assign, leaf_stmts[2])
            return [
                llir.FunctionCallStmt(
                    name=f"{result_name}_values.emplace_back",
                    args=[value_store.value],
                ),
                leaf_stmts[1],
                llir.FunctionCallStmt(
                    name=f"{result_name}{leaf_level}_crd.emplace_back",
                    args=[coordinate_store.value],
                ),
                leaf_stmts[3],
            ]
        aligned_tensors = {
            cursor.tensor for cursor in loop.cursors if cursor.cursor in aligned
        }
        if len(aligned_tensors) == len({c.tensor for c in loop.cursors}):
            child_stmts = self._child_stream_statements(position)
        else:
            # A one-sided UNION case drains the single aligned operand's
            # whole subtree in stored order.
            child_stmts = self._one_sided_stream_statements(
                position + 1, next(iter(aligned_tensors))
            )
        return [
            *child_stmts,
            llir.BlankLine(),
            llir.Comment("Assembly compressed _level indices"),
            *self._parent_append_statements(position),
        ]

    def _child_stream_statements(self, position: int) -> List[llir.Stmt]:
        """The child stream loop below one assembly level, by driver kind."""

        child = self.loops[position + 1]
        if child.kind is _SPARSE:
            return self._lower_sparse(position + 1)
        if child.kind is _MERGED:
            return self._lower_merged(position + 1)
        _fail(
            "unsupported_program_shape",
            "a multi-compressed assembly level must nest a stream loop",
        )
        raise AssertionError("unreachable")

    def _one_sided_cursor(self, position: int, tensor: SymbolId) -> SparseCursorDecl:
        for cursor in self.loops[position].cursors:
            if cursor.tensor == tensor:
                return cursor
        _fail(
            "unsupported_program_shape",
            "a one-sided union drain requires the aligned operand's cursor "
            "at every child level",
        )
        raise AssertionError("unreachable")

    def _one_sided_stream_statements(
        self, position: int, tensor: SymbolId
    ) -> List[llir.Stmt]:
        """One operand's ordered subtree drain below a one-sided union case.

        Emits the exact single-cursor stream loop the legacy union lattice
        produces for the aligned operand — the ``_end`` iterator init, the
        stored-order ``for`` over the operand's own child segment, the
        folded leaf appends (the absent operand's default folds away), and
        the same conditional parent append and child close every assembly
        level owns.
        """

        loop = self.loops[position]
        cursor = self._one_sided_cursor(position, tensor)
        dimension_name = self._loop_var_name(loop)
        position_name = self._cursor_position_name(cursor)
        if position == len(self.loops) - 1:
            leaf_stmts = self._merged_case_stmts(loop, {cursor.cursor})
            if leaf_stmts is None:
                _fail(
                    "unsupported_program_shape",
                    "a one-sided union leaf must append its stored entries",
                )
            child_body: List[llir.Stmt] = leaf_stmts
        else:
            child_body = [
                *self._one_sided_stream_statements(position + 1, tensor),
                llir.BlankLine(),
                llir.Comment("Assembly compressed _level indices"),
                *self._parent_append_statements(position),
            ]
        body: List[llir.Stmt] = [
            llir.Comment("Resolve coordinates"),
            llir.VarInit(
                var=llir.Var(name=dimension_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_crd_array(cursor),
                    index=llir.Var(name=position_name, type=llir.DataType.INT),
                ),
            ),
            llir.BlankLine(),
            *child_body,
        ]
        position_var = llir.Var(name=position_name, type=llir.DataType.INT)
        for_loop = llir.ForLoop(
            init=llir.VarInit(
                var=llir.Var(name=position_name, type=llir.DataType.INT),
                value=llir.ArrayAccess(
                    array=self._cursor_pos_array(cursor),
                    index=self._cursor_parent_index(cursor, 0),
                ),
            ),
            cond=llir.BinOp(
                op="<",
                left=position_var,
                right=llir.Var(name=f"{position_name}_end", type=llir.DataType.INT),
            ),
            update=llir.Increment(
                var=llir.Var(name=position_name, type=llir.DataType.INT)
            ),
            body=body,
        )
        for_loop.scorch_index_var = dimension_name
        return [
            llir.Comment("Initialize iterators"),
            llir.VarInit(
                var=llir.Var(
                    name=f"{position_name}_end",
                    type=llir.DataType.INT,
                ),
                value=llir.ArrayAccess(
                    array=self._cursor_pos_array(cursor),
                    index=self._cursor_parent_index(cursor, 1),
                ),
            ),
            llir.BlankLine(),
            for_loop,
        ]

    def _loop_children(self, position: int) -> List[llir.Stmt]:
        """Single-cursor levels append the leaf or the child stream.

        A single-cursor stream loop appends its level's entries in stored
        order: the leaf level emits the checked value/coordinate appends,
        and every other level nests its child stream followed by the same
        conditional parent append and child close the merged levels own.
        """

        loop = self.loops[position]
        if position == len(self.loops) - 1:
            leaf_stmts = self._merged_case_stmts(
                loop, {cursor.cursor for cursor in loop.cursors}
            )
            if leaf_stmts is None:
                _fail(
                    "unsupported_program_shape",
                    "a single-cursor assembly leaf must append its stored " "entries",
                )
            return leaf_stmts
        if loop.kind is _DENSE:
            # The dense prefix closes its child level through the assembly
            # catch-up in ``_lower_dense``, never a conditional append.
            return self._child_stream_statements(position)
        return [
            *self._child_stream_statements(position),
            llir.BlankLine(),
            llir.Comment("Assembly compressed _level indices"),
            *self._parent_append_statements(position),
        ]

    def _dense_assembly_close_level(self, position: int) -> int:
        return position + 1

    def _exact_dense_parent_positions(self) -> bool:
        """Whether the legacy assembler pre-sizes dense-parent position arrays.

        The legacy lowering sizes a compressed level's position vector from
        its dense parent's extent for single-operand assemblies whose input
        carries compressed structure (and for very large dense conversions,
        which this family's dense-domain seam keeps out).  The choice is a
        structural property of the verified program — operand count, the
        single operand's declared layout, and the statically bound result
        extents — never a runtime-format probe.
        """

        if len(self._input_symbols) != 1:
            return False
        input_decl = self.decls[self._input_symbols[0]]
        if any(level.kind is not LevelKind.DENSE for level in input_decl.levels):
            return True
        result_cells = 1
        for extent in self.shapes[self.result_symbol]:
            result_cells *= extent
        return result_cells >= 1024 * 1024

    def _fixed_position_count(self, level: int) -> bool:
        """Whether one result position vector is pre-sized by a dense parent."""

        return (
            self._exact_dense_parent_positions()
            and level > 0
            and self.result_decl.levels[level - 1].kind is LevelKind.DENSE
        )

    def result_assembler(self) -> ResultTensorAssembler:
        return ResultTensorAssembler(
            name=self.result_decl.name,
            level_types=tuple(
                _LEVEL_KIND_TO_LEVEL_TYPE[level.kind]
                for level in self.result_decl.levels
            ),
            dtype=_SCALAR_TO_TORCH[self.result_decl.dtype],
            exact_dense_parent_positions=self._exact_dense_parent_positions(),
        )

    def _assembly_result_pos_set(self, level: Optional[int] = None) -> llir.Stmt:
        raw = super()._result_pos_set(level)
        if type(raw.var) is not llir.ArrayAccess or type(raw.var.array) is not llir.Var:
            _fail(
                "unsupported_program_shape",
                "the multi-compressed position close lost its vector target",
            )
        if self._fixed_position_count(
            len(self.result_decl.levels) - 1 if level is None else level
        ):
            # A pre-sized position vector is written in bounds by
            # construction (its dense parent extent is ABI-validated), so
            # the raw indexed close is the safe legacy spelling.
            return raw
        return llir.FunctionCallStmt(
            name="scorch_vector_set",
            args=[
                llir.Var(name=raw.var.array.name, type=llir.DataType.NO_TYPE),
                raw.var.index,
                raw.value,
            ],
        )

    def _checked_position_init_statement(self, level: int) -> llir.FunctionCallStmt:
        """One checked position sentinel owned directly by this target."""

        return llir.FunctionCallStmt(
            name="scorch_vector_set",
            args=[
                llir.Var(
                    name=f"{self.result_decl.name}{level}_pos",
                    type=llir.DataType.NO_TYPE,
                ),
                llir.Literal(value=0, data_type=llir.DataType.INT32),
                llir.Literal(value=0, data_type=llir.DataType.INT32),
            ],
        )

    def prepare_result_level_indices(
        self, statements: List[llir.Stmt]
    ) -> List[llir.Stmt]:
        """Build every append-owned position sentinel in its checked form.

        Pre-sized dense-parent position vectors are declared at their exact
        extent and carry no zero sentinel, so only the dynamically grown
        levels are converted and required here.
        """

        expected_position_levels = {
            f"{self.result_decl.name}{level}_pos": level
            for level, decl in enumerate(self.result_decl.levels)
            if decl.kind is LevelKind.COMPRESSED
            and not self._fixed_position_count(level)
        }
        prepared: List[llir.Stmt] = []
        converted: Set[str] = set()
        for statement in statements:
            root_name = (
                _llir_assignment_root_name(statement.var)
                if type(statement) is llir.Assign
                else None
            )
            level = expected_position_levels.get(cast(str, root_name))
            if level is None:
                prepared.append(statement)
                continue
            assignment = cast(llir.Assign, statement)
            target = assignment.var
            if (
                type(target) is not llir.ArrayAccess
                or type(target.array) is not llir.Var
                or type(target.index) is not llir.Literal
                or type(target.index.value) is not int
                or target.index.value != 0
                or type(assignment.value) is not llir.Literal
                or type(assignment.value.value) is not int
                or assignment.value.value != 0
                or cast(str, root_name) in converted
            ):
                _fail(
                    "unsupported_program_shape",
                    "the result assembler must initialize each append-owned "
                    "compressed position vector exactly once at zero",
                )
            prepared.append(self._checked_position_init_statement(level))
            converted.add(cast(str, root_name))
        if converted != set(expected_position_levels):
            _fail(
                "unsupported_program_shape",
                "the result assembler must own every compressed position sentinel",
            )
        return prepared

    def raw_loop_statements(self) -> List[llir.Stmt]:
        _TargetLowering._require_value_expression_unchanged(self)
        if self.loops[0].kind in (_MERGED, _SPARSE):
            stmts: List[llir.Stmt] = (
                list(self._lower_merged(0))
                if self.loops[0].kind is _MERGED
                else list(self._lower_sparse(0))
            )
            stmts.append(llir.BlankLine())
            stmts.append(llir.Comment("Assembly compressed _level indices"))
            stmts.append(self._assembly_result_pos_set(0))
            return stmts
        return [llir.BlankLine(), self._lower_dense(0)]


_RELAYOUT_LOST = "relayout_completion_lost"
_RESULT_TILE_LOST = "result_tile_completion_lost"
_PARALLEL_LOST = "parallel_completion_lost"
_SPARSE_WORKSPACE_LOST = "sparse_workspace_completion_lost"


def _relayout_access_candidates(
    statements: List[llir.Stmt],
    metadata: llir.TensorAccessMetadata,
) -> List[llir.Expr]:
    """Collect the exact logical access before relayout redirection."""

    class _Collector(LLIRWalker):
        def __init__(self) -> None:
            super().__init__(
                LLIRTraversalContext(
                    stage="LoopIR target lowering",
                    pass_name="locate_relayout_operand_access",
                )
            )
            self.matches: List[llir.Expr] = []

        def leave_node(self, node: llir.Node, path: Tuple[str, ...]) -> None:
            # Match only after the common walker has validated the complete
            # node.  Forged exact TensorAccessMetadata instances can own
            # hostile or missing fields; consulting those fields from
            # enter_node would run before visit_var/visit_array_access can
            # reject them.
            expression: llir.Var | llir.ArrayAccess
            if type(node) is llir.Var:
                expression = cast(llir.Var, node)
            elif type(node) is llir.ArrayAccess:
                expression = cast(llir.ArrayAccess, node)
            else:
                return
            node_metadata = expression.tensor_access
            if (
                node_metadata is not None
                and not _TargetLowering._exact_panel_state_matches(
                    node_metadata, node_metadata
                )
            ):
                raise ValueError(
                    "tensor access metadata does not retain exact stored "
                    "identity state"
                )
            if _TargetLowering._exact_panel_state_matches(node_metadata, metadata):
                self.matches.append(expression)

    collector = _Collector()
    collector.walk(cast(Any, statements))
    return collector.matches


def _result_tile_assign_owners(
    statements: List[llir.Stmt],
    target: llir.Expr,
) -> List[llir.Assign]:
    """Find exact assignments whose lvalue is ``target`` by identity."""

    class _Collector(LLIRWalker):
        def __init__(self) -> None:
            super().__init__(
                LLIRTraversalContext(
                    stage="LoopIR target lowering",
                    pass_name="locate_result_tile_write_owner",
                )
            )
            self.matches: List[llir.Assign] = []

        def leave_node(self, node: llir.Node, path: Tuple[str, ...]) -> None:
            if type(node) is llir.Assign and node.var is target:
                self.matches.append(cast(llir.Assign, node))

    collector = _Collector()
    collector.walk(cast(Any, statements))
    return collector.matches


def _named_var_candidates(
    statements: List[llir.Stmt],
    name: str,
) -> List[llir.Var]:
    """Collect every exact physical occurrence of one validated C++ Var."""

    class _Collector(LLIRWalker):
        def __init__(self) -> None:
            super().__init__(
                LLIRTraversalContext(
                    stage="LoopIR target lowering",
                    pass_name="locate_result_tile_physical_accesses",
                )
            )
            self.matches: List[llir.Var] = []

        def leave_node(self, node: llir.Node, path: Tuple[str, ...]) -> None:
            if type(node) is llir.Var and node.name == name:
                self.matches.append(cast(llir.Var, node))

    collector = _Collector()
    collector.walk(cast(Any, statements))
    return collector.matches


def _named_call_candidates(
    function: llir.Function, call_name: str
) -> List[llir.FunctionCallStmt]:
    """Collect every validated call statement of one exact name recursively."""

    class _Collector(LLIRWalker):
        def __init__(self) -> None:
            super().__init__(
                LLIRTraversalContext(
                    stage="LoopIR target lowering",
                    pass_name="locate_result_tile_named_call",
                )
            )
            self.matches: List[llir.FunctionCallStmt] = []

        def leave_node(self, node: llir.Node, path: Tuple[str, ...]) -> None:
            if type(node) is llir.FunctionCallStmt and node.name == call_name:
                self.matches.append(cast(llir.FunctionCallStmt, node))

    collector = _Collector()
    collector.walk(function)
    return collector.matches


def _dense_zero_candidates(function: llir.Function) -> List[llir.FunctionCallStmt]:
    """Collect every validated dense-zero call recursively."""

    return _named_call_candidates(function, "scorch_zero_dense")


def _require_canonical_result_shape_validation(
    lowering: "_TargetLowering", function: llir.Function
) -> None:
    """Pin the complete ABI validation block to its canonical prologue.

    Heap completion must not merely census the result-shape call by name:
    moving the exact call after allocation or compute would let invalid
    runtime shape state drive writes before validation.  Function assembly
    owns the complete validation block at offset zero; panel completion is
    the sole predecessor that prepends statements, and owns exactly its
    comment/width/blank prefix at offset three.  The whole freshly
    reconstructed block is compared structurally before any heap mutation.
    """

    expected_validations = lowering.kernel_abi().emit_validation()
    offset = 3 if lowering.panel is not None else 0
    if (
        type(function.body) is not list
        or len(function.body) < offset + len(expected_validations)
        or any(
            not lowering._exact_panel_state_matches(actual, expected)
            for actual, expected in zip(
                function.body[offset : offset + len(expected_validations)],
                expected_validations,
            )
        )
    ):
        _fail(
            _RESULT_TILE_LOST,
            "heap accumulation requires the complete canonical ABI "
            "validation block before allocation and compute",
        )
    validation_candidates = _named_call_candidates(
        function,
        "scorch_native::validate_jit_result_shape",
    )
    if (
        not expected_validations
        or len(validation_candidates) != 1
        or validation_candidates[0] is not function.body[offset]
    ):
        _fail(
            _RESULT_TILE_LOST,
            "heap accumulation requires exactly one canonical ABI "
            "result-shape validation",
        )


_RESULT_TILE_POLICY_TOKEN = re.compile(
    r"""
    (?P<space>\s+)
    |(?P<identifier>[A-Za-z_][A-Za-z0-9_]*)
    |(?P<integer>[0-9]+)
    |(?P<operator>&&|\|\||==|!=|<=|>=|::|[()[\],+\-*/%?:<>])
    """,
    re.ASCII | re.VERBOSE,
)
_RESULT_TILE_POLICY_MACROS = {"SCORCH_GRAIN_CODEGEN_SPGEMM"}
_RESULT_TILE_NUMERIC_LITERAL = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)" r"(?:[eE][+-]?[0-9]+)?)(?:[fFlLuU]*)\Z",
    re.ASCII,
)
_RESULT_TILE_UNARY_OPERATORS = {"+", "-", "!", "~", "*", "&"}
# ``BinOp.op`` is interpolated into the emitted C++ verbatim, exactly like the
# unary operator above.  Only these non-mutating spellings may separate two
# operands; assignment, compound assignment, comma, and statement punctuation
# would splice a fresh effect between the proven result accesses.
_RESULT_TILE_BINARY_OPERATORS = {
    "||",
    "&&",
    "|",
    "^",
    "&",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "<<",
    ">>",
    "+",
    "-",
    "*",
    "/",
    "%",
}


def _safe_cpp_qualified_name(name: object) -> bool:
    """Whether ``name`` is one or more exact ASCII C++ identifiers."""

    if type(name) is not str:
        return False
    parts = name.split("::")
    return bool(parts) and all(
        _CPP_IDENTIFIER.fullmatch(part) is not None and part not in _CPP_KEYWORDS
        for part in parts
    )


def _result_tile_policy_tokens(value: str) -> List[Tuple[str, str]]:
    """Tokenize one legacy OpenMP policy expression without accepting C++ text."""

    tokens: List[Tuple[str, str]] = []
    position = 0
    while position < len(value):
        match = _RESULT_TILE_POLICY_TOKEN.match(value, position)
        if match is None:
            raise ValueError("heap result-tile policy contains unsupported text")
        position = match.end()
        if match.lastgroup != "space":
            assert match.lastgroup is not None
            tokens.append((match.lastgroup, match.group()))
    return tokens


def _validate_result_tile_policy(
    value: object,
    *,
    helper: str,
    known_names: Set[str],
    protected_names: Set[str],
) -> None:
    """Validate one compiler-owned legacy OpenMP helper expression.

    These fields predate typed pragma expressions and codegen emits them
    verbatim.  Heap completion may therefore reason about result ownership only
    after proving that the retained text is a single helper call over
    non-mutating arithmetic, subscripts, and identifiers already present in the
    structured function.  Nested calls, statement punctuation, member access,
    assignment, increment/decrement, and result-owned identifiers are excluded.
    """

    if type(value) is not str or not value:
        raise ValueError("heap result-tile policy must be a non-empty string")
    if (
        "\n" in value
        or "\r" in value
        or "++" in value
        or "--" in value
        or "//" in value
        or "/*" in value
        or "*/" in value
    ):
        raise ValueError("heap result-tile policy contains effectful text")
    tokens = _result_tile_policy_tokens(value)
    if len(tokens) < 3 or tokens[:2] != [
        ("identifier", helper),
        ("operator", "("),
    ]:
        raise ValueError(f"heap result-tile policy must call {helper}")

    delimiters: List[str] = []
    for index, (kind, token) in enumerate(tokens):
        if token in ("(", "["):
            delimiters.append(token)
        elif token in (")", "]"):
            expected = "(" if token == ")" else "["
            if not delimiters or delimiters.pop() != expected:
                raise ValueError("heap result-tile policy delimiters are unbalanced")
            if not delimiters and index != len(tokens) - 1:
                raise ValueError(
                    "heap result-tile policy contains text after its helper call"
                )
        if kind != "identifier":
            continue
        if token in protected_names:
            raise ValueError("heap result-tile policy references result-owned storage")
        if token in _CPP_KEYWORDS and token != "long":
            raise ValueError("heap result-tile policy contains a C++ keyword")
        if (
            index + 1 < len(tokens)
            and tokens[index + 1] == ("operator", "(")
            and index != 0
        ):
            raise ValueError("heap result-tile policy contains a nested call")
        if (
            token != helper
            and token != "long"
            and token not in known_names
            and token not in _RESULT_TILE_POLICY_MACROS
        ):
            raise ValueError(
                "heap result-tile policy references an undeclared identifier"
            )
    if delimiters:
        raise ValueError("heap result-tile policy delimiters are unbalanced")


def _result_tile_expression_root_name(expr: object) -> Optional[str]:
    """Return the exact C++ root identifier of one structured access."""

    current = expr
    visited: Set[int] = set()
    while type(current) in (llir.ArrayAccess, llir.MemberAccess, llir.AddressOf):
        if id(current) in visited:
            raise ValueError("heap result-tile access root must be acyclic")
        visited.add(id(current))
        if type(current) is llir.ArrayAccess:
            current = cast(llir.ArrayAccess, current).array
        elif type(current) is llir.MemberAccess:
            current = cast(llir.MemberAccess, current).base
        else:
            current = cast(llir.AddressOf, current).operand
    if type(current) is llir.Var:
        name = cast(llir.Var, current).name
        if type(name) is str:
            return name
    return None


def _result_tile_protected_uses(
    value: object,
    protected_names: Set[str],
) -> Set[str]:
    """Collect protected exact-Var uses from one validated LLIR value."""

    class _Collector(LLIRWalker):
        def __init__(self) -> None:
            super().__init__(
                LLIRTraversalContext(
                    stage="LoopIR target lowering",
                    pass_name="locate_result_tile_protected_uses",
                )
            )
            self.names: Set[str] = set()

        def leave_node(self, node: llir.Node, path: Tuple[str, ...]) -> None:
            if type(node) is llir.Var and node.name in protected_names:
                self.names.add(node.name)

    collector = _Collector()
    if type(value) in (list, tuple):
        for item in cast(Any, value):
            collector.walk(item)
    else:
        collector.walk(cast(Any, value))
    return collector.names


def _protected_torch_empty_candidates(
    function: llir.Function,
    protected_names: Set[str],
) -> List[llir.FunctionCall]:
    """Collect every ``torch::empty`` expression fed protected result state."""

    class _Collector(LLIRWalker):
        def __init__(self) -> None:
            super().__init__(
                LLIRTraversalContext(
                    stage="LoopIR target lowering",
                    pass_name="locate_result_tile_torch_allocations",
                )
            )
            self.matches: List[llir.FunctionCall] = []

        def leave_node(self, node: llir.Node, path: Tuple[str, ...]) -> None:
            if (
                type(node) is llir.FunctionCall
                and node.name == "torch::empty"
                and _result_tile_protected_uses(node.args, protected_names)
            ):
                self.matches.append(cast(llir.FunctionCall, node))

    collector = _Collector()
    collector.walk(function)
    return collector.matches


class _ResultTileTextValidator(LLIRWalker):
    """Reject every verbatim-text route the C++ target emits directly.

    Lifted to module scope so each per-node checker is an independently
    measurable unit; the walked function's result-owned storage names are
    supplied once at construction.
    """

    def __init__(self, protected_names: Set[str]) -> None:
        super().__init__(
            LLIRTraversalContext(
                stage="LoopIR target lowering",
                pass_name="validate_result_tile_rendered_text",
            )
        )
        self.protected_names = protected_names
        self.declared_names: Dict[str, str] = {}
        self.policies: List[Tuple[object, str]] = []

    @staticmethod
    def _identifier(value: object, owner: str) -> str:
        if (
            type(value) is not str
            or _CPP_IDENTIFIER.fullmatch(value) is None
            or value in _CPP_KEYWORDS
        ):
            raise ValueError(f"{owner} must be a safe ASCII C++ identifier")
        return value

    def _declare_name(self, value: object, owner: str) -> str:
        name = self._identifier(value, owner)
        previous = self.declared_names.get(name)
        if previous is not None:
            raise ValueError(
                f"{owner} duplicates C++ declaration {name!r} already owned "
                f"by {previous}"
            )
        self.declared_names[name] = owner
        return name

    def _declare_var(self, value: object, owner: str) -> str:
        if type(value) is not llir.Var:
            raise ValueError(f"{owner} must declare an exact Var")
        return self._declare_name(getattr(value, "name", None), owner)

    def leave_node(self, node: llir.Node, path: Tuple[str, ...]) -> None:
        # Exact-type dispatch, never isinstance: a node type with no
        # verbatim-text field of its own is fully pinned by the common
        # walker this validator extends.
        checker = self._CHECKERS.get(type(node))
        if checker is not None:
            checker(self, node)

    def _check_var(self, node: llir.Node) -> None:
        self._identifier(getattr(node, "name", None), "Var.name")
        if type(getattr(node, "type", None)) is not llir.DataType:
            raise ValueError("Var.type must be a DataType")
        if type(getattr(node, "is_ptr", None)) is not bool:
            raise ValueError("Var.is_ptr must be a bool")
        if type(getattr(node, "is_restrict", None)) is not bool:
            raise ValueError("Var.is_restrict must be a bool")

    def _check_unary_op(self, node: llir.Node) -> None:
        operator = getattr(node, "op", None)
        if type(operator) is not str or operator not in _RESULT_TILE_UNARY_OPERATORS:
            raise ValueError("UnaryOp.op must be a non-mutating unary operator")

    def _check_binary_op(self, node: llir.Node) -> None:
        operator = getattr(node, "op", None)
        if type(operator) is not str or operator not in _RESULT_TILE_BINARY_OPERATORS:
            raise ValueError(
                "binary expression operators must be non-mutating "
                "C++ binary operators"
            )

    def _check_call_name(self, node: llir.Node) -> None:
        name = getattr(node, "name", None)
        if not _safe_cpp_qualified_name(name):
            raise ValueError(f"{type(node).__name__}.name must be a qualified C++ name")
        if (
            type(node) is llir.FunctionCall
            and _result_tile_protected_uses(
                getattr(node, "args", None),
                self.protected_names,
            )
            and name != "torch::empty"
        ):
            raise ValueError(
                "FunctionCall exposes protected result state to an unowned call"
            )

    def _check_qualified_name(self, node: llir.Node) -> None:
        self._identifier(
            getattr(node, "namespace", None),
            "QualifiedName.namespace",
        )
        self._identifier(getattr(node, "name", None), "QualifiedName.name")

    def _check_member_name(self, node: llir.Node) -> None:
        member = self._identifier(
            getattr(node, "member", None),
            f"{type(node).__name__}.member",
        )
        if type(node) is llir.MemberAccess:
            return
        # A member call is an unknown callee exactly like a free function:
        # protected result state may not escape through its arguments even
        # when the receiver itself is unprotected (a non-const reference
        # parameter could mutate the forwarded state).  The canonical owned
        # ``data_ptr`` acquisition carries no arguments, so no production
        # spelling is affected.
        if _result_tile_protected_uses(
            getattr(node, "args", None),
            self.protected_names,
        ):
            raise ValueError(
                f"{type(node).__name__} exposes protected result state to "
                "an unowned call"
            )
        root_name = _result_tile_expression_root_name(getattr(node, "base", None))
        if root_name not in self.protected_names:
            return
        data_pointer_roots = {
            name for name in self.protected_names if name.endswith("_values_torch")
        }
        if (
            type(node) is llir.MemberCall
            and member == "data_ptr"
            and root_name in data_pointer_roots
        ):
            return
        raise ValueError(
            f"{type(node).__name__} mutates or calls protected result state"
        )

    def _check_address_of(self, node: llir.Node) -> None:
        root_name = _result_tile_expression_root_name(getattr(node, "operand", None))
        if root_name in self.protected_names:
            raise ValueError("AddressOf exposes protected result state")

    def _check_literal(self, node: llir.Node) -> None:
        value = getattr(node, "value", None)
        data_type = getattr(node, "data_type", None)
        if (
            type(value) is str
            and data_type is not llir.DataType.STRING
            and _RESULT_TILE_NUMERIC_LITERAL.fullmatch(value) is None
        ):
            raise ValueError("non-STRING Literal text must be a numeric C++ literal")

    def _check_var_init(self, node: llir.Node) -> None:
        operator = getattr(node, "op", None)
        if type(operator) is not str or operator != "=":
            raise ValueError("VarInit.op must remain '='")
        if type(getattr(node, "cast", None)) is not bool:
            raise ValueError("VarInit.cast must be a bool")
        self._declare_var(getattr(node, "var", None), "VarInit.var")

    def _check_var_decl(self, node: llir.Node) -> None:
        self._declare_var(getattr(node, "var", None), "VarDecl.var")

    def _check_direct_init(self, node: llir.Node) -> None:
        self._declare_var(getattr(node, "var", None), "DirectInit.var")

    def _check_fixed_stack_array_decl(self, node: llir.Node) -> None:
        self._declare_name(
            getattr(node, "name", None),
            "FixedStackArrayDecl.name",
        )

    def _check_for_loop_auto(self, node: llir.Node) -> None:
        self._declare_var(getattr(node, "var", None), "ForLoopAuto.var")

    def _check_comment(self, node: llir.Node) -> None:
        value = getattr(node, "value", None)
        if (
            type(value) is not str
            or "\n" in value
            or "\r" in value
            or "\\" in value
            or "??/" in value
        ):
            raise ValueError("Comment.value must be a single-line string")

    def _check_raw_stmt(self, node: llir.Node) -> None:
        raise ValueError("heap result-tile completion requires fully structured LLIR")

    def _check_if_then_else(self, node: llir.Node) -> None:
        if type(getattr(node, "make_last_case_else", None)) is not bool:
            raise ValueError("IfThenElse.make_last_case_else must be a bool")
        cond = getattr(node, "cond", None)
        then_body = getattr(node, "then_body", None)
        cond_list = getattr(node, "cond_list", None)
        then_body_list = getattr(node, "then_body_list", None)
        if cond_list:
            if not then_body_list:
                raise ValueError("IfThenElse with cond_list requires then_body_list")
            if len(cond_list) != len(then_body_list):
                raise ValueError("IfThenElse condition and body counts must match")
        elif cond is None:
            raise ValueError("IfThenElse requires a condition")
        elif not then_body:
            raise ValueError("IfThenElse requires a then body")

    def _check_function(self, node: llir.Node) -> None:
        self._identifier(getattr(node, "name", None), "Function.name")
        if type(getattr(node, "return_type", None)) is not llir.DataType:
            raise ValueError("Function.return_type must be a DataType")
        args = getattr(node, "args", None)
        if type(args) not in (list, tuple):
            raise ValueError("Function.args must contain exact Var nodes")
        if any(type(arg) is not llir.Var for arg in cast(Any, args)):
            raise ValueError("Function.args must contain exact Var nodes")
        for arg in cast(Any, args):
            self._declare_var(arg, "Function.args")

    def _check_for_loop(self, node: llir.Node) -> None:
        schedule = getattr(node, "omp_schedule", None)
        if schedule is not None and (
            type(schedule) is not str
            or schedule not in ("static", "dynamic", "dynamic, 16", "dynamic, 64")
        ):
            raise ValueError("ForLoop.omp_schedule is not a recognized compiler policy")
        for field in ("omp_parallel_for", "unroll", "simd"):
            if type(getattr(node, field, None)) is not bool:
                raise ValueError(f"ForLoop.{field} must be a bool")
        for field, helper in (
            ("omp_num_threads", "scorch_nthreads"),
            ("omp_chunk_expr", "scorch_chunk"),
        ):
            value = getattr(node, field, None)
            if value is not None:
                self.policies.append((value, helper))
        atomic = getattr(node, "_use_atomic_scheduling", False)
        if type(atomic) is not bool:
            raise ValueError("ForLoop._use_atomic_scheduling must be a bool")
        if not atomic:
            return
        counter_name = self._declare_name(
            getattr(node, "_atomic_counter_var", None),
            "ForLoop._atomic_counter_var",
        )
        if counter_name in self.protected_names:
            raise ValueError(
                "ForLoop._atomic_counter_var references result-owned storage"
            )
        self._declare_name("_start", "atomic codegen _start")
        self._declare_name("_end", "atomic codegen _end")
        if getattr(node, "init", None) is None:
            self._declare_name("i", "atomic codegen fallback loop variable")
        for field in (
            "_atomic_chunk_var",
            "_loop_bound",
        ):
            atomic_name = self._identifier(
                getattr(node, field, None),
                f"ForLoop.{field}",
            )
            if atomic_name in self.protected_names:
                raise ValueError(f"ForLoop.{field} references result-owned storage")

    _CHECKERS = {
        llir.Var: _check_var,
        llir.UnaryOp: _check_unary_op,
        llir.BinOp: _check_binary_op,
        llir.Add: _check_binary_op,
        llir.Mul: _check_binary_op,
        llir.FunctionCall: _check_call_name,
        llir.FunctionCallStmt: _check_call_name,
        llir.QualifiedName: _check_qualified_name,
        llir.MemberAccess: _check_member_name,
        llir.MemberCall: _check_member_name,
        llir.MemberCallStmt: _check_member_name,
        llir.AddressOf: _check_address_of,
        llir.Literal: _check_literal,
        llir.VarInit: _check_var_init,
        llir.VarDecl: _check_var_decl,
        llir.DirectInit: _check_direct_init,
        llir.FixedStackArrayDecl: _check_fixed_stack_array_decl,
        llir.Comment: _check_comment,
        llir.RawStmt: _check_raw_stmt,
        llir.IfThenElse: _check_if_then_else,
        llir.Function: _check_function,
        llir.ForLoop: _check_for_loop,
        llir.ForLoopAuto: _check_for_loop_auto,
    }

    def validate_policies(self) -> None:
        for value, helper in self.policies:
            _validate_result_tile_policy(
                value,
                helper=helper,
                known_names=set(self.declared_names),
                protected_names=self.protected_names,
            )


class _ResultTileVarUseValidator(LLIRWalker):
    """Resolve expression variables against one exact lexical environment."""

    def __init__(self, visible_names: Set[str]) -> None:
        super().__init__(
            LLIRTraversalContext(
                stage="LoopIR target lowering",
                pass_name="validate_result_tile_variable_use",
            )
        )
        self.visible_names = visible_names

    def leave_node(self, node: llir.Node, path: Tuple[str, ...]) -> None:
        if type(node) is not llir.Var:
            return
        name = getattr(node, "name", None)
        if type(name) is not str or name not in self.visible_names:
            raise ValueError(f"Var {name!r} is used before a visible declaration")


class _ResultTileBindingValidator:
    """Validate declaration order and lexical scope as emitted by codegen.

    Heap completion relies on one function-wide declaration owner per name,
    which :class:`_ResultTileTextValidator` enforces conservatively.  This
    second pass checks the complementary property: each exact ``Var`` use is
    visible at its C++ emission point.  It intentionally models the legacy
    OpenMP/atomic lowering because those branches synthesize scopes and a few
    declarations that are not explicit statement nodes.
    """

    def __init__(
        self,
        protected_names: Set[str],
        *,
        result_name: Optional[str],
    ) -> None:
        self.protected_names = protected_names
        self.result_name = result_name

    @staticmethod
    def _name(var: object, owner: str) -> str:
        if type(var) is not llir.Var:
            raise ValueError(f"{owner} must be an exact Var")
        name = getattr(var, "name", None)
        return _ResultTileTextValidator._identifier(name, owner)

    @staticmethod
    def _bind_expression(expr: object, visible_names: Set[str]) -> None:
        if not isinstance(expr, llir.Expr):
            raise ValueError("variable binding requires an exact LLIR expression")
        _ResultTileVarUseValidator(visible_names).walk(expr)

    @classmethod
    def _bind_expressions(cls, expressions: object, visible_names: Set[str]) -> None:
        if type(expressions) not in (list, tuple):
            raise ValueError("expression arguments must be a list or tuple")
        for expression in cast(Any, expressions):
            cls._bind_expression(expression, visible_names)

    @staticmethod
    def _declare_name(name: str, visible_names: Set[str]) -> None:
        # Global uniqueness was already proved by the text validator.
        visible_names.add(name)

    @classmethod
    def _declare_var(cls, var: object, owner: str, visible_names: Set[str]) -> None:
        cls._declare_name(cls._name(var, owner), visible_names)

    def _bind_policy(
        self,
        value: object,
        *,
        helper: str,
        visible_names: Set[str],
    ) -> None:
        _validate_result_tile_policy(
            value,
            helper=helper,
            known_names=visible_names,
            protected_names=self.protected_names,
        )

    def _bind_sequence(
        self,
        statements: object,
        visible_names: Set[str],
        *,
        loop_depth: int,
    ) -> None:
        if type(statements) not in (list, tuple):
            raise ValueError("statement scope must be a list or tuple")
        for statement in cast(Any, statements):
            if type(statement) in (list, tuple):
                # Nested containers emit no braces and therefore share scope.
                self._bind_sequence(
                    statement,
                    visible_names,
                    loop_depth=loop_depth,
                )
            elif isinstance(statement, llir.Stmt):
                self._bind_statement(
                    statement,
                    visible_names,
                    loop_depth=loop_depth,
                )
            else:
                raise ValueError("statement scope contains a non-statement value")

    def _bind_declaration(self, statement: llir.Stmt, visible_names: Set[str]) -> None:
        if type(statement) is llir.VarInit:
            self._bind_expression(statement.value, visible_names)
            self._declare_var(statement.var, "VarInit.var", visible_names)
        elif type(statement) is llir.VarDecl:
            self._declare_var(statement.var, "VarDecl.var", visible_names)
        elif type(statement) is llir.DirectInit:
            self._bind_expressions(statement.args, visible_names)
            self._declare_var(statement.var, "DirectInit.var", visible_names)
        elif type(statement) is llir.FixedStackArrayDecl:
            self._bind_expression(statement.extent, visible_names)
            self._bind_expression(statement.initializer, visible_names)
            self._declare_name(statement.name, visible_names)
        else:
            raise ValueError("expected one exact LLIR declaration")

    def _bind_call_statement(
        self,
        statement: llir.Stmt,
        visible_names: Set[str],
    ) -> None:
        if type(statement) is llir.FunctionCallStmt:
            protected_uses = _result_tile_protected_uses(
                statement.args,
                self.protected_names,
            )
            allowed_calls = {
                "scorch_native::validate_jit_result_shape",
                "scorch_zero_dense",
            }
            if protected_uses and statement.name not in allowed_calls:
                raise ValueError(
                    "FunctionCallStmt exposes protected result state to an "
                    "unowned call"
                )
            self._bind_expressions(statement.args, visible_names)
            return
        if type(statement) is llir.MemberCallStmt:
            # The text validator rejects protected receivers before binding.
            self._bind_expression(statement.base, visible_names)
            self._bind_expressions(statement.args, visible_names)
            return
        if type(statement) is llir.GuardedCallStmt:
            if _result_tile_protected_uses(
                statement.call.args,
                self.protected_names,
            ):
                raise ValueError("GuardedCallStmt exposes protected result state")
            self._bind_expression(statement.cond, visible_names)
            self._bind_expressions(statement.call.args, visible_names)
            return
        raise ValueError("expected one exact LLIR call statement")

    def _bind_for_loop(
        self,
        loop: llir.ForLoop,
        visible_names: Set[str],
        *,
        loop_depth: int,
    ) -> None:
        before = loop.before_parallel_body
        if before:
            self._bind_sequence(
                before,
                visible_names,
                loop_depth=loop_depth,
            )

        atomic = getattr(loop, "_use_atomic_scheduling", False)
        has_split_region = bool(
            loop.omp_parallel_for
            and (loop.pre_parallel_body or loop.post_parallel_body)
        )
        has_regular_pragma = bool(loop.omp_parallel_for or loop.unroll or loop.simd)
        if before and not atomic and not has_regular_pragma:
            raise ValueError(
                "ForLoop.before_parallel_body would not be emitted by codegen"
            )

        if atomic:
            counter_name = _ResultTileTextValidator._identifier(
                getattr(loop, "_atomic_counter_var", None),
                "ForLoop._atomic_counter_var",
            )
            # Codegen emits the atomic counter after before_parallel_body and
            # before the parallel pragma, so the policy can refer to it.
            self._declare_name(counter_name, visible_names)

        if loop.omp_num_threads is not None:
            if not atomic and not loop.omp_parallel_for:
                raise ValueError(
                    "ForLoop.omp_num_threads would not be emitted by codegen"
                )
            self._bind_policy(
                loop.omp_num_threads,
                helper="scorch_nthreads",
                visible_names=visible_names,
            )

        hidden_pre_or_post = bool(
            (loop.pre_parallel_body or loop.post_parallel_body)
            and not atomic
            and not has_split_region
        )
        if hidden_pre_or_post:
            raise ValueError(
                "ForLoop pre/post parallel statements would not be emitted by codegen"
            )
        if getattr(loop, "_hoisted_ptr_decls", None):
            raise ValueError(
                "ForLoop._hoisted_ptr_decls survived without an emission owner"
            )

        if atomic:
            if loop.omp_chunk_expr is not None:
                raise ValueError(
                    "atomic ForLoop.omp_chunk_expr would not be emitted by codegen"
                )
            parallel_names = set(visible_names)
            if loop.pre_parallel_body:
                self._bind_sequence(
                    loop.pre_parallel_body,
                    parallel_names,
                    loop_depth=loop_depth,
                )
            chunk_name = _ResultTileTextValidator._identifier(
                getattr(loop, "_atomic_chunk_var", None),
                "ForLoop._atomic_chunk_var",
            )
            if chunk_name not in parallel_names:
                raise ValueError(
                    "ForLoop._atomic_chunk_var references "
                    f"{chunk_name!r} before a visible declaration"
                )
            after_start_names = set(parallel_names)
            after_start_names.add("_start")
            bound_name = _ResultTileTextValidator._identifier(
                getattr(loop, "_loop_bound", None),
                "ForLoop._loop_bound",
            )
            if bound_name not in after_start_names:
                raise ValueError(
                    f"ForLoop._loop_bound references {bound_name!r} "
                    "before a visible declaration"
                )
            inner_names = set(after_start_names)
            inner_names.add("_end")
            if loop.init is None:
                loop_name = "i"
            else:
                loop_name = self._name(loop.init.var, "ForLoop.init.var")
            inner_names.add(loop_name)
            self._bind_sequence(
                loop.body,
                inner_names,
                loop_depth=loop_depth + 1,
            )
            if loop.post_parallel_body:
                self._bind_sequence(
                    loop.post_parallel_body,
                    parallel_names,
                    loop_depth=loop_depth,
                )
            return

        if loop.omp_chunk_expr is not None and not loop.omp_parallel_for:
            raise ValueError("ForLoop.omp_chunk_expr would not be emitted by codegen")

        if has_split_region:
            parallel_names = set(visible_names)
            if loop.pre_parallel_body:
                self._bind_sequence(
                    loop.pre_parallel_body,
                    parallel_names,
                    loop_depth=loop_depth,
                )
            if loop.omp_chunk_expr is not None:
                self._bind_policy(
                    loop.omp_chunk_expr,
                    helper="scorch_chunk",
                    visible_names=parallel_names,
                )
            loop_names = set(parallel_names)
            if loop.init is not None:
                self._bind_declaration(loop.init, loop_names)
            self._bind_expression(loop.cond, loop_names)
            if type(loop.update) is llir.FunctionCall:
                self._bind_expression(loop.update, loop_names)
            else:
                self._bind_statement(
                    cast(llir.Stmt, loop.update),
                    loop_names,
                    loop_depth=loop_depth,
                    loop_update_name=(
                        self._name(loop.init.var, "ForLoop.init.var")
                        if loop.init is not None
                        else None
                    ),
                )
            self._bind_sequence(
                loop.body,
                set(loop_names),
                loop_depth=loop_depth + 1,
            )
            if loop.post_parallel_body:
                self._bind_sequence(
                    loop.post_parallel_body,
                    parallel_names,
                    loop_depth=loop_depth,
                )
            return

        if loop.omp_chunk_expr is not None:
            self._bind_policy(
                loop.omp_chunk_expr,
                helper="scorch_chunk",
                visible_names=visible_names,
            )
        loop_names = set(visible_names)
        if loop.init is not None:
            self._bind_declaration(loop.init, loop_names)
        self._bind_expression(loop.cond, loop_names)
        if type(loop.update) is llir.FunctionCall:
            self._bind_expression(loop.update, loop_names)
        else:
            self._bind_statement(
                cast(llir.Stmt, loop.update),
                loop_names,
                loop_depth=loop_depth,
                loop_update_name=(
                    self._name(loop.init.var, "ForLoop.init.var")
                    if loop.init is not None
                    else None
                ),
            )
        self._bind_sequence(
            loop.body,
            set(loop_names),
            loop_depth=loop_depth + 1,
        )

    def _bind_if_then_else(
        self,
        conditional: llir.IfThenElse,
        visible_names: Set[str],
        *,
        loop_depth: int,
    ) -> None:
        if conditional.cond is not None:
            self._bind_expression(conditional.cond, visible_names)
        if conditional.then_body is not None:
            self._bind_sequence(
                conditional.then_body,
                set(visible_names),
                loop_depth=loop_depth,
            )
        if conditional.cond_list is not None:
            self._bind_expressions(conditional.cond_list, visible_names)
        if conditional.then_body_list is not None:
            for branch in conditional.then_body_list:
                self._bind_sequence(
                    branch,
                    set(visible_names),
                    loop_depth=loop_depth,
                )
        if conditional.else_body is not None:
            self._bind_sequence(
                conditional.else_body,
                set(visible_names),
                loop_depth=loop_depth,
            )

    def _bind_statement(
        self,
        statement: llir.Stmt,
        visible_names: Set[str],
        *,
        loop_depth: int,
        loop_update_name: Optional[str] = None,
    ) -> None:
        statement_type = type(statement)
        if statement_type in (
            llir.VarInit,
            llir.VarDecl,
            llir.DirectInit,
            llir.FixedStackArrayDecl,
        ):
            self._bind_declaration(statement, visible_names)
        elif statement_type is llir.Assign:
            assignment = cast(llir.Assign, statement)
            target_root = _result_tile_expression_root_name(assignment.var)
            if target_root in self.protected_names:
                metadata = getattr(assignment.var, "tensor_access", None)
                owns_result_write = bool(
                    type(assignment.var) is llir.ArrayAccess
                    and type(metadata) is llir.TensorAccessMetadata
                    and metadata.role is llir.TensorAccessRole.RESULT_WRITE
                )
                owns_terminal_assembly = bool(
                    type(assignment.var) is llir.MemberAccess
                    and target_root == self.result_name
                )
                owns_position_update = bool(
                    target_root == loop_update_name
                    and self.result_name is not None
                    and target_root is not None
                    and target_root.startswith(f"p{self.result_name}")
                )
                if not (
                    owns_result_write or owns_terminal_assembly or owns_position_update
                ):
                    raise ValueError(
                        f"Assign mutates protected result state {target_root!r}"
                    )
            self._bind_expression(assignment.var, visible_names)
            self._bind_expression(assignment.value, visible_names)
        elif statement_type is llir.Increment:
            increment = cast(llir.Increment, statement)
            target_root = _result_tile_expression_root_name(increment.var)
            owns_position_update = bool(
                target_root == loop_update_name
                and self.result_name is not None
                and target_root is not None
                and target_root.startswith(f"p{self.result_name}")
            )
            if target_root in self.protected_names and not owns_position_update:
                raise ValueError(
                    f"Increment mutates protected result state {target_root!r}"
                )
            self._bind_expression(increment.var, visible_names)
        elif statement_type is llir.Return:
            return_statement = cast(llir.Return, statement)
            self._bind_expression(return_statement.value, visible_names)
        elif statement_type in (
            llir.FunctionCallStmt,
            llir.MemberCallStmt,
            llir.GuardedCallStmt,
        ):
            self._bind_call_statement(statement, visible_names)
        elif statement_type is llir.ForLoop:
            self._bind_for_loop(
                cast(llir.ForLoop, statement),
                visible_names,
                loop_depth=loop_depth,
            )
        elif statement_type is llir.ForLoopAuto:
            loop = cast(llir.ForLoopAuto, statement)
            self._bind_expression(loop.array, visible_names)
            loop_names = set(visible_names)
            self._declare_var(loop.var, "ForLoopAuto.var", loop_names)
            self._bind_sequence(
                loop.body,
                loop_names,
                loop_depth=loop_depth + 1,
            )
        elif statement_type is llir.WhileLoop:
            while_loop = cast(llir.WhileLoop, statement)
            self._bind_expression(while_loop.cond, visible_names)
            self._bind_sequence(
                while_loop.body,
                set(visible_names),
                loop_depth=loop_depth + 1,
            )
        elif statement_type is llir.IfThenElse:
            self._bind_if_then_else(
                cast(llir.IfThenElse, statement),
                visible_names,
                loop_depth=loop_depth,
            )
        elif statement_type is llir.Function:
            raise ValueError("nested LLIR Function definitions are not supported")
        elif statement_type in (
            llir.Comment,
            llir.BlankLine,
        ):
            return
        elif statement_type in (llir.Continue, llir.Break):
            if loop_depth < 1:
                raise ValueError(
                    f"{statement_type.__name__} requires an enclosing loop"
                )
        elif statement_type is llir.RawStmt:
            # The text validator owns the more specific structured-only error.
            raise ValueError("heap result-tile completion requires structured LLIR")
        else:
            raise ValueError(
                f"heap result-tile binding does not own {statement_type.__name__}"
            )

    def validate(self, function: llir.Function) -> None:
        visible_names: Set[str] = set()
        for argument in function.args:
            self._declare_var(argument, "Function.args", visible_names)
        self._bind_sequence(function.body, visible_names, loop_depth=0)


def _validate_result_tile_rendered_text(
    function: llir.Function,
    *,
    protected_names: Set[str],
    result_name: Optional[str] = None,
) -> Set[str]:
    """Close every verbatim-text route before proving heap result ownership."""

    validator = _ResultTileTextValidator(protected_names)
    validator.walk(function)
    _ResultTileBindingValidator(
        protected_names,
        result_name=result_name,
    ).validate(function)
    validator.validate_policies()
    return set(validator.declared_names)


def _direct_prefetch_targets_array(stmt: llir.Stmt, array_name: str) -> bool:
    """Whether one validated guard directly prefetches ``array_name[...]``."""

    if type(stmt) is not llir.GuardedCallStmt:
        return False
    call = stmt.call
    if (
        type(call) is not llir.FunctionCallStmt
        or type(call.name) is not str
        or call.name != "__builtin_prefetch"
        or type(call.args) is not tuple
        or not call.args
    ):
        return False
    address = call.args[0]
    if type(address) is not llir.AddressOf:
        return False
    operand = address.operand
    if type(operand) is not llir.ArrayAccess:
        return False
    array = operand.array
    return (
        type(array) is llir.Var and type(array.name) is str and array.name == array_name
    )


def _count_direct_prefetches(statements: List[llir.Stmt], array_name: str) -> int:
    """Count direct array prefetches recursively after common validation."""

    class _Collector(LLIRWalker):
        def __init__(self) -> None:
            super().__init__(
                LLIRTraversalContext(
                    stage="LoopIR target lowering",
                    pass_name="locate_residual_relayout_prefetches",
                )
            )
            self.count = 0

        def leave_node(self, node: llir.Node, path: Tuple[str, ...]) -> None:
            if isinstance(node, llir.Stmt) and _direct_prefetch_targets_array(
                node, array_name
            ):
                self.count += 1

    collector = _Collector()
    collector.walk(cast(Any, statements))
    return collector.count


def _complete_result_tile_chain(
    lowering: "_TargetLowering",
    function: llir.Function,
) -> llir.ForLoop:
    """Re-identify the heap chain and own its completion-time row policy."""

    if lowering.panel is not None:
        state = lowering._panel_completion
        if state is None:
            _fail(
                _RESULT_TILE_LOST,
                "the panel completion did not retain the heap chain's loops",
            )
        # Panel completion already realized the policy on the shared row,
        # but heap completion is a distinct mutation boundary and must not
        # trust either the header or the owned selection in between.
        lowering._require_retained_panel_parallel_policy(_RESULT_TILE_LOST)
        return state[0]
    try:
        completed = lowering._completed_panel_chain(function)
    except LoopIRTargetError as error:
        _fail(_RESULT_TILE_LOST, error.defect.message)
    tile_loop = completed[0]
    row_loop = completed[lowering.result_tile_row_position]
    try:
        expected_marked = LLIRRewriter(
            LLIRTraversalContext(
                stage="LoopIR target lowering",
                pass_name="snapshot_result_tile_parallel_policy",
            )
        ).rewrite(row_loop)
        if type(expected_marked) is not llir.ForLoop:
            _fail(
                _RESULT_TILE_LOST,
                "the heap row-loop policy snapshot did not remain a ForLoop",
            )
        if lowering.parallel is None:
            mark_first_for_loop_parallel(
                [expected_marked],
                EMPTY_PARALLEL_WORKSPACE_CLUSTER,
            )
            mark_first_for_loop_parallel(
                [row_loop],
                EMPTY_PARALLEL_WORKSPACE_CLUSTER,
            )
        else:
            policy_spec = lowering._parallel_work_policy_spec(
                _RESULT_TILE_LOST,
            )
            lowering._apply_expected_parallel_marker(expected_marked)
            mark_first_for_loop_parallel(
                [row_loop],
                EMPTY_PARALLEL_WORKSPACE_CLUSTER,
            )
            if not lowering._panel_loop_header_matches(
                row_loop,
                expected_marked,
            ):
                _fail(
                    _RESULT_TILE_LOST,
                    "the heap row loop did not acquire the unmodified "
                    "legacy parallel marker state",
                )
            lowering._apply_selected_parallel_policy(
                expected_marked,
                policy_spec,
                _RESULT_TILE_LOST,
                invoke_marker=False,
            )
            lowering._apply_selected_parallel_policy(
                row_loop,
                policy_spec,
                _RESULT_TILE_LOST,
                invoke_marker=False,
            )
    except (
        LLIRTraversalError,
        AttributeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        _fail(_RESULT_TILE_LOST, str(error))
    if not lowering._panel_loop_header_matches(row_loop, expected_marked):
        _fail(
            _RESULT_TILE_LOST,
            "the heap chain's selected row loop did not acquire the "
            "required parallel policy",
        )
    return tile_loop


def _complete_result_tile_impl(
    lowering: "_TargetLowering", function: llir.Function
) -> llir.Function:
    """Complete one heap result tile on the assembled function.

    Mirrors the legacy ``_apply_heap_result_tile`` exactly, driven by the
    retained completion objects and the target's own emission-record
    spellings — no name, tag, or ordinal discovery.  Every disagreement
    with the recorded pre-pass state is the stage-owned
    ``result_tile_completion_lost`` diagnostic.
    """

    decl = lowering.result_tile
    assert decl is not None
    write_snapshot = lowering._tiled_write_snapshot
    if write_snapshot is None:
        _fail(
            _RESULT_TILE_LOST,
            "the result tile's emitted statements were never recorded",
        )
    try:
        # Completion runs after the managed pass pipeline and panel
        # completion.  Validate the complete assembled function before any
        # name discovery or mutation so forged top-level state cannot escape
        # the stage-owned diagnostic.
        LLIRWalker(
            LLIRTraversalContext(
                stage="LoopIR target lowering",
                pass_name="validate_result_tile_completion_input",
            )
        ).walk(function)
        result_name = lowering.result_decl.name
        protected_names = {
            result_name,
            "result_shape",
            f"{result_name}_values",
            f"{result_name}_values_torch",
            f"{result_name}_capacity",
            f"{result_name}_shape",
            f"{result_name}_mode_indices",
        }
        for level in range(len(lowering.result_decl.levels)):
            protected_names.update(
                {
                    f"{result_name}{level}_size",
                    f"{result_name}{level}_pos",
                    f"{result_name}{level}_crd",
                    f"p{result_name}{level}",
                }
            )
        _validate_result_tile_rendered_text(
            function,
            protected_names=protected_names,
            result_name=result_name,
        )
    except (
        LLIRTraversalError,
        AttributeError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _fail(_RESULT_TILE_LOST, str(error))
    _require_canonical_result_shape_validation(lowering, function)

    # 1. Re-identify the chain.  A panel chain hands over the loops its
    # completion already re-identified (retained object identity); the bare
    # heap chain re-identifies its direct loops against the detached
    # pre-pass headers and marks the parallel row exactly as the legacy
    # explicit-parallel schedule does before its heap transformation.
    tile_loop = _complete_result_tile_chain(lowering, function)

    # 2. The target's own spellings for the compact family.
    result_name = lowering.result_decl.name
    last_level = len(lowering.result_decl.levels) - 1
    pack_dimension_name = lowering.dimension_names[lowering.loops[0].node.dimension]
    tile_outer_name = f"{pack_dimension_name}_out"
    tile_inner_name = f"{pack_dimension_name}_in"
    tile_size_name = f"kTile_{pack_dimension_name}"
    torch_dtype = _SCALAR_TO_TORCH[lowering.result_decl.dtype]
    pointer_type = llir.DataType.ptr_type(torch_dtype)
    if lowering.result_decl.dtype is ScalarType.FLOAT32:
        scalar_type = llir.DataType.FLOAT32
        zero_value = "0.0f"
    else:
        scalar_type = llir.DataType.FLOAT64
        zero_value = "0.0"
    try:
        (
            compact_name,
            storage_name,
            init_prefix,
            init_inner,
            init_logical,
            copy_prefix,
            copy_inner,
            copy_logical,
        ) = _heap_result_tile_names(
            function,
            result_name,
            pack_dimension_name,
            reserved_names=_validate_result_tile_rendered_text(
                function,
                protected_names=protected_names,
                result_name=result_name,
            ),
        )
    except (
        LLIRTraversalError,
        AttributeError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _fail(_RESULT_TILE_LOST, str(error))
    trailing_bound = f"{result_name}{last_level}_size"
    prefix_dimension_names = tuple(
        f"{result_name}{level}_size" for level in range(last_level)
    )
    prefix_extent = " * ".join(prefix_dimension_names)

    # 3. Redirect the emitted result write to compact storage: exactly one
    # metadata candidate whose entire physical subtree matches the detached
    # pre-pass snapshot, then the exact-one rewrite with a residual
    # re-check — the reviewed relayout boundary's discipline.
    metadata = lowering._result_metadata()
    try:
        write_candidates = _relayout_access_candidates(tile_loop.body, metadata)
        result_vars = _named_var_candidates(
            tile_loop.body,
            f"{result_name}_values",
        )
    except (
        LLIRTraversalError,
        AttributeError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _fail(_RESULT_TILE_LOST, str(error))
    if (
        len(write_candidates) != 1
        or type(write_candidates[0]) is not llir.ArrayAccess
        or not lowering._exact_panel_state_matches(write_candidates[0], write_snapshot)
    ):
        _fail(
            _RESULT_TILE_LOST,
            "the compacted result's physical emitted write does not retain "
            "exactly its detached pre-pass state",
        )
    write_candidate = write_candidates[0]
    try:
        owners = _result_tile_assign_owners(tile_loop.body, write_candidate)
    except (
        LLIRTraversalError,
        AttributeError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _fail(_RESULT_TILE_LOST, str(error))
    candidate_array = write_candidate.array
    if (
        len(owners) != 1
        or owners[0].op is not llir.AssignOp.ADD_ASSIGN
        or type(candidate_array) is not llir.Var
        or len(result_vars) != 1
        or result_vars[0] is not candidate_array
    ):
        _fail(
            _RESULT_TILE_LOST,
            "the compacted result access must remain the sole physical "
            "result occurrence and the lvalue of one additive assignment",
        )
    compact_access = _heap_compact_access(
        compact_name=compact_name,
        pointer_type=pointer_type,
        prefix_position_name=f"p{result_name}{last_level - 1}",
        tile_size_name=tile_size_name,
        tile_inner_name=tile_inner_name,
    )
    try:
        rewritten_body, rewritten = _rewrite_stmt_access_sequence(
            cast(Any, tile_loop.body),
            lowering.result_symbol,
            metadata.index_ids,
            llir.TensorAccessRole.RESULT_WRITE,
            compact_access,
        )
        tile_loop.body = cast(List[llir.Stmt], rewritten_body)
        residual = _contains_tensor_access(
            tile_loop.body,
            lowering.result_symbol,
            metadata.index_ids,
            llir.TensorAccessRole.RESULT_WRITE,
        )
        physical_residual = _named_var_candidates(
            tile_loop.body,
            f"{result_name}_values",
        )
    except (
        LLIRTraversalError,
        AttributeError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _fail(_RESULT_TILE_LOST, str(error))
    if rewritten != 1 or residual or physical_residual:
        _fail(
            _RESULT_TILE_LOST,
            "the compacted result's emitted write was not redirected "
            "exactly once or an unowned physical result access survived",
        )

    # 4. The exactly-once copy-out coverage replaces the whole-result zero
    # fill.  Require the one exact ABI-generated top-level call and reject
    # every nested or altered physical zero effect before invoking the
    # unchanged legacy compatibility remover.
    expected_zero = llir.FunctionCallStmt(
        name="scorch_zero_dense",
        args=(
            llir.Var(f"{result_name}_values", pointer_type),
            llir.Var(f"{result_name}_capacity", llir.DataType.INT64),
        ),
    )
    expected_result_pointer_init = llir.VarInit(
        var=llir.Var(
            f"{result_name}_values",
            pointer_type,
            is_restrict=True,
        ),
        value=llir.MemberCall(
            base=llir.Var(
                f"{result_name}_values_torch",
                llir.DataType.TORCH_TENSOR,
            ),
            member="data_ptr",
            template_args=(scalar_type,),
        ),
    )
    try:
        expected_value_init = lowering.result_assembler().emit_value_array_init()
        expected_result_tensor_inits = [
            stmt
            for stmt in expected_value_init
            if type(stmt) is llir.VarInit
            and type(stmt.var) is llir.Var
            and stmt.var.name == f"{result_name}_values_torch"
        ]
        expected_final_assembly = lowering.result_assembler().emit_final_assembly()
        expected_storage_positions = [
            index
            for index, stmt in enumerate(expected_final_assembly)
            if type(stmt) is llir.Assign
            and type(stmt.value) is llir.Var
            and stmt.value.name == f"{result_name}_values_torch"
        ]
        if (
            len(expected_result_tensor_inits) != 1
            or len(expected_storage_positions) != 1
            or len(function.body) < len(expected_final_assembly)
        ):
            _fail(
                _RESULT_TILE_LOST,
                "the dense result assembler no longer has one canonical "
                "Torch allocation and final storage owner",
            )
        actual_final_assembly = function.body[-len(expected_final_assembly) :]
        if not lowering._exact_panel_state_matches(
            actual_final_assembly,
            expected_final_assembly,
        ):
            _fail(
                _RESULT_TILE_LOST,
                "the dense result's final assembly no longer retains its "
                "canonical terminal state",
            )
        actual_storage_assign = actual_final_assembly[expected_storage_positions[0]]
        assert type(actual_storage_assign) is llir.Assign
        assert type(actual_storage_assign.value) is llir.Var
        result_tensor_inits = [
            stmt
            for stmt in function.body
            if type(stmt) is llir.VarInit
            and type(stmt.var) is llir.Var
            and stmt.var.name == f"{result_name}_values_torch"
        ]
        if len(result_tensor_inits) != 1 or not lowering._exact_panel_state_matches(
            result_tensor_inits[0],
            expected_result_tensor_inits[0],
        ):
            _fail(
                _RESULT_TILE_LOST,
                "heap accumulation requires exactly one canonical generated "
                "dense-result Torch allocation",
            )
        canonical_allocation = result_tensor_inits[0].value
        protected_allocations = _protected_torch_empty_candidates(
            function,
            protected_names,
        )
        if (
            type(canonical_allocation) is not llir.FunctionCall
            or len(protected_allocations) != 1
            or protected_allocations[0] is not canonical_allocation
        ):
            _fail(
                _RESULT_TILE_LOST,
                "protected result shape state may reach only the one "
                "canonical dense-result Torch allocation",
            )
        result_pointer_inits = [
            stmt
            for stmt in function.body
            if type(stmt) is llir.VarInit
            and type(stmt.var) is llir.Var
            and stmt.var.name == f"{result_name}_values"
        ]
        if len(result_pointer_inits) != 1 or not lowering._exact_panel_state_matches(
            result_pointer_inits[0],
            expected_result_pointer_init,
        ):
            _fail(
                _RESULT_TILE_LOST,
                "heap accumulation requires exactly one canonical generated "
                "dense-result pointer declaration",
            )
        pointer_value = result_pointer_inits[0].value
        if (
            type(pointer_value) is not llir.MemberCall
            or type(pointer_value.base) is not llir.Var
        ):
            _fail(
                _RESULT_TILE_LOST,
                "the canonical dense-result pointer lost its Torch storage " "owner",
            )
        result_tensor_vars = _named_var_candidates(
            function.body,
            f"{result_name}_values_torch",
        )
        owned_result_tensor_vars = (
            result_tensor_inits[0].var,
            pointer_value.base,
            actual_storage_assign.value,
        )
        if (
            len(result_tensor_vars) != 3
            or len({id(value) for value in owned_result_tensor_vars}) != 3
            or {id(value) for value in result_tensor_vars}
            != {id(value) for value in owned_result_tensor_vars}
        ):
            _fail(
                _RESULT_TILE_LOST,
                "an unowned use of the dense result's Torch storage "
                "survived heap completion",
            )
        zero_candidates = _dense_zero_candidates(function)
        if len(zero_candidates) != 1 or not lowering._exact_panel_state_matches(
            zero_candidates[0], expected_zero
        ):
            _fail(
                _RESULT_TILE_LOST,
                "heap accumulation requires exactly one canonical generated "
                "dense-result zero at function scope",
            )
        physical_before_zero_removal = _named_var_candidates(
            function.body,
            f"{result_name}_values",
        )
        if len(physical_before_zero_removal) != 2 or {
            id(value) for value in physical_before_zero_removal
        } != {
            id(result_pointer_inits[0].var),
            id(zero_candidates[0].args[0]),
        }:
            _fail(
                _RESULT_TILE_LOST,
                "an unowned physical result access survived compact-write "
                "redirection",
            )
        located_zero = lowering._locate_statement(function.body, zero_candidates[0])
        if located_zero is None or located_zero[0] is not function.body:
            _fail(
                _RESULT_TILE_LOST,
                "the canonical generated dense-result zero moved out of "
                "function scope",
            )
        _remove_dense_result_zero(function, result_name)
        remaining_physical = _named_var_candidates(
            function.body,
            f"{result_name}_values",
        )
        if (
            _dense_zero_candidates(function)
            or len(remaining_physical) != 1
            or remaining_physical[0] is not result_pointer_inits[0].var
        ):
            _fail(
                _RESULT_TILE_LOST,
                "a dense-result zero or unowned physical result effect "
                "survived heap completion",
            )
    except (
        LLIRTraversalError,
        NotImplementedError,
        AttributeError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _fail(_RESULT_TILE_LOST, str(error))

    # 5. The per-strip init and copy-out groups inside the pack origin, and
    # the reusable storage block directly above it at function scope.
    tile_loop.body[0:0] = _heap_result_init_group(
        result=result_name,
        compact_name=compact_name,
        pointer_type=pointer_type,
        scalar_type=scalar_type,
        zero_value=zero_value,
        tile_outer_name=tile_outer_name,
        tile_size_name=tile_size_name,
        trailing_bound=trailing_bound,
        prefix_extent=prefix_extent,
        init_prefix=init_prefix,
        init_inner=init_inner,
        init_logical=init_logical,
    )
    tile_loop.body.extend(
        _heap_result_copy_group(
            result=result_name,
            result_values=f"{result_name}_values",
            compact_name=compact_name,
            pointer_type=pointer_type,
            tile_outer_name=tile_outer_name,
            tile_size_name=tile_size_name,
            trailing_bound=trailing_bound,
            prefix_extent=prefix_extent,
            copy_prefix=copy_prefix,
            copy_inner=copy_inner,
            copy_logical=copy_logical,
        )
    )
    located = lowering._locate_statement(function.body, tile_loop)
    if located is None or located[0] is not function.body:
        _fail(
            _RESULT_TILE_LOST,
            "the pack origin loop cannot be located at function scope",
        )
    tile_container, tile_index = located
    tile_container[tile_index:tile_index] = _heap_result_storage_statements(
        result=result_name,
        storage_name=storage_name,
        compact_name=compact_name,
        pointer_type=pointer_type,
        scalar_type=scalar_type,
        prefix_dimension_names=prefix_dimension_names,
        tile_size_name=tile_size_name,
    )
    return function


def _complete_relayout_impl(
    lowering: "_TargetLowering", function: llir.Function
) -> llir.Function:
    """Stage the packed operand on the assembled, panel-completed function.

    Runs immediately after :meth:`_TargetLowering.complete_panel` — the
    legacy ``apply_schedule_to_llir`` order — consuming the panel
    completion's retained, already re-identified loop objects.  Every
    spelling comes from this lowering's own typed emission records or the
    schedule lowerer's shared constructors (one source per spelling); no
    rendered-name discovery, regexes, dynamic tags, or bare ordinals are
    consulted, and every re-identified auxiliary statement is cross-checked
    against its detached pre-pass snapshot, failing closed as
    ``relayout_completion_lost``.
    """

    decl = lowering.relayout
    assert decl is not None
    state = lowering._panel_completion
    coord_snapshot = lowering._window_coord_snapshot
    access_snapshot = lowering._staged_access_snapshot
    if (
        state is None
        or coord_snapshot is None
        or access_snapshot is None
        or lowering.panel is None
    ):
        _fail(
            _RELAYOUT_LOST,
            "the relayout's emitted statements were never recorded",
        )
    pack_loop, panel_loop, row_loop, window_loop = state
    lowering._require_retained_panel_parallel_policy(_RELAYOUT_LOST)
    operand_decl = lowering.decls[decl.operand]
    operand_name = operand_decl.name
    panel_dim_name = lowering.dimension_names[lowering.panel.dimension]
    pack_dim_name = lowering.dimension_names[lowering.loops[0].dimension]
    panel_outer_name = f"{panel_dim_name}_out"
    panel_end = f"{panel_outer_name}_end"
    pack_outer_name = f"{pack_dim_name}_out"
    pack_inner_name = f"{pack_dim_name}_in"
    pack_tile_var = f"kTile_{pack_dim_name}"
    panel_tile_var = f"kTile_{panel_dim_name}"
    operand_value_array = f"{operand_name}_val"
    # The validated relayout family stores the panel dimension on level 0
    # and the pack dimension on the contiguous level 1 of the operand.
    panel_axis_bound = f"{operand_name}0_size"
    pack_axis_bound = f"{operand_name}1_size"
    torch_dtype = _SCALAR_TO_TORCH[operand_decl.dtype]
    pointer_type = llir.DataType.ptr_type(torch_dtype)
    scalar_type = (
        llir.DataType.FLOAT32
        if operand_decl.dtype is ScalarType.FLOAT32
        else llir.DataType.FLOAT64
    )
    panel_scoped = decl.scope is RelayoutScope.PANEL
    used_names = _declared_names(function)
    packed_name = _unique_name(f"packed_{operand_name}", used_names)
    storage_name = _unique_name(f"{packed_name}_storage", used_names)
    stage_row_origin = panel_outer_name if panel_scoped else None

    # 1. Redirect the emitted operand read to packed storage — the same
    # typed (tensor, index identities, role) metadata triple the legacy
    # rewriter consumes, made unambiguous by the pass-level single-
    # occurrence proof.
    staged_read_row: llir.Expr = (
        llir.BinOp(
            op="-",
            left=llir.Var(panel_dim_name, llir.DataType.INT64),
            right=llir.Var(panel_outer_name, llir.DataType.INT64),
        )
        if panel_scoped
        else llir.Var(panel_dim_name, llir.DataType.INT64)
    )
    packed_read = llir.ArrayAccess(
        array=llir.Var(packed_name, pointer_type),
        index=llir.Add(
            llir.Mul(
                staged_read_row,
                llir.Var(pack_tile_var, llir.DataType.INT64),
            ),
            llir.Var(pack_inner_name, llir.DataType.INT64),
        ),
    )
    metadata = lowering._input_metadata(decl.operand)
    try:
        access_candidates = _relayout_access_candidates(row_loop.body, metadata)
    except (
        LLIRTraversalError,
        AttributeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        _fail(_RELAYOUT_LOST, str(error))
    if len(access_candidates) != 1 or not lowering._exact_panel_state_matches(
        access_candidates[0], access_snapshot
    ):
        _fail(
            _RELAYOUT_LOST,
            "the staged operand's physical emitted access does not retain "
            "exactly its detached pre-pass state",
        )
    try:
        rewritten_body, rewritten = _rewrite_stmt_access_sequence(
            cast(Any, row_loop.body),
            decl.operand,
            metadata.index_ids,
            llir.TensorAccessRole.INPUT_READ,
            packed_read,
        )
        row_loop.body = cast(List[llir.Stmt], rewritten_body)
        residual = _contains_tensor_access(
            row_loop.body,
            decl.operand,
            metadata.index_ids,
            llir.TensorAccessRole.INPUT_READ,
        )
    except (
        LLIRTraversalError,
        AttributeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        _fail(_RELAYOUT_LOST, str(error))
    if rewritten != 1 or residual:
        _fail(
            _RELAYOUT_LOST,
            "the staged operand's emitted read was not redirected exactly once",
        )

    # 2. Adapt the sparse prefetch to the packed storage, passing this
    # lowering's own coordinate-array spelling so the legacy name scan
    # never runs.
    cursor = lowering.loops[lowering.window_position].cursors[0]
    try:
        redirected_prefetches = _redirect_sparse_prefetch(
            window_loop,
            operand_value_array,
            packed_name,
            panel_outer_name,
            panel_end,
            pack_tile_var,
            stage_row_origin,
            coordinate_array_name=lowering._cursor_crd_array(cursor).name,
        )
    except NotImplementedError as error:
        _fail(_RELAYOUT_LOST, str(error))
    if redirected_prefetches != 1:
        _fail(
            _RELAYOUT_LOST,
            "the staged operand requires exactly one canonical sparse "
            "prefetch guard",
        )
    try:
        residual_operand_prefetches = _count_direct_prefetches(
            window_loop.body, operand_value_array
        )
    except (
        LLIRTraversalError,
        AttributeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        _fail(_RELAYOUT_LOST, str(error))
    if residual_operand_prefetches:
        _fail(
            _RELAYOUT_LOST,
            "the staged operand retains a noncanonical sparse prefetch guard",
        )

    # 3. Insert the compatibility range guard directly after the window's
    # complete resolved-coordinate skeleton, re-identified against its
    # detached pre-pass snapshot at the canonical lexical position directly
    # after the rewritten prefetch.  Content equality alone is insufficient:
    # moving either the declaration or its whole context below a use must fail
    # closed.
    coord_candidates = [
        index + 1
        for index in range(len(window_loop.body) - 2)
        if all(
            lowering._exact_panel_state_matches(actual, expected)
            for actual, expected in zip(
                window_loop.body[index : index + 3],
                coord_snapshot,
            )
        )
    ]
    if coord_candidates != [2]:
        _fail(
            _RELAYOUT_LOST,
            "the window's resolved coordinate does not retain its exact "
            "contiguous pre-pass context and lexical position",
        )
    coord_index = coord_candidates[0]
    window_loop.body.insert(
        coord_index + 1,
        _panel_range_guard(panel_dim_name, panel_outer_name, panel_end),
    )

    # 4. Build and place the staging (pack) loop at the scope position.
    pack_row = _unique_name(f"{panel_dim_name}_pack", used_names)
    pack_col = _unique_name(f"{pack_dim_name}_pack", used_names)
    logical_pack_col = _unique_name(f"{pack_dim_name}_packed", used_names)
    pack_outer = _relayout_pack_loop(
        panel_scoped=panel_scoped,
        pack_row=pack_row,
        pack_col=pack_col,
        logical_pack_col=logical_pack_col,
        packed_name=packed_name,
        pointer_type=pointer_type,
        operand_value_array=operand_value_array,
        pack_outer_name=pack_outer_name,
        pack_tile_var=pack_tile_var,
        pack_axis_bound=pack_axis_bound,
        panel_axis_bound=panel_axis_bound,
        panel_outer_name=panel_outer_name,
        panel_end=panel_end,
    )
    pack_outer.scorch_index_var = f"pack:{operand_name}"
    scope_description = (
        f"{panel_dim_name} panel" if panel_scoped else f"full {panel_dim_name} axis"
    )
    stage_statements: List[llir.Stmt] = [
        llir.Comment(
            f"Pack {operand_name} {scope_description} into contiguous "
            f"{panel_dim_name}-major storage"
        ),
        pack_outer,
        llir.BlankLine(),
    ]
    if panel_scoped:
        stage_container = panel_loop.body
        stage_positions = [
            index for index, stmt in enumerate(stage_container) if stmt is row_loop
        ]
        if len(stage_positions) != 1:
            _fail(
                _RELAYOUT_LOST,
                "the completed panel no longer owns its row loop; the "
                "staging region cannot be placed",
            )
        stage_index = stage_positions[0]
    else:
        stage_container = pack_loop.body
        stage_index = 0
    stage_container[stage_index:stage_index] = stage_statements

    # 5. Allocate the reusable packed storage directly above the pack
    # origin loop, located by retained object identity.
    located_pack = lowering._locate_statement(function.body, pack_loop)
    if located_pack is None:
        _fail(
            _RELAYOUT_LOST,
            "the retained pack origin loop cannot be located in the "
            "assembled function",
        )
    pack_container, pack_index = located_pack
    stage_rows = panel_tile_var if panel_scoped else panel_axis_bound
    stage_rows_type = (
        llir.DataType.CONSTEXPR_INT if panel_scoped else llir.DataType.INT64
    )
    pack_container[pack_index:pack_index] = _relayout_storage_statements(
        operand=operand_name,
        storage_name=storage_name,
        packed_name=packed_name,
        pointer_type=pointer_type,
        scalar_type=scalar_type,
        stage_rows=stage_rows,
        stage_rows_type=stage_rows_type,
        pack_tile_var=pack_tile_var,
    )
    return function


def _lower_loopir_to_llir_owned(
    program: LoopProgram,
    *,
    input_shapes: Mapping[SymbolId, Tuple[int, ...]],
    result_shape: Tuple[int, ...],
    compile_options: CompileOptions,
    compilation_context: Optional[CompilationContext],
) -> llir.Function:
    """Execute target lowering under an already-owned timing boundary."""

    # Preserve the target's cheap malformed-program boundary before invoking
    # caller-controlled Mapping code. A second verification below closes the
    # mutation window after every callback from a custom mapping has returned;
    # an exact dict over exact SymbolId keys executes no caller code here.
    verify_program(program)
    mapping_callbacks_possible = type(input_shapes) is not dict
    try:
        shape_keys = tuple(input_shapes)
    except Exception as error:
        raise LoopIRTargetError(
            LoopIRTargetDefect(
                "invalid_shape_binding",
                "input shapes could not be snapshotted",
            )
        ) from error
    shape_key_values: List[int] = []
    for key in shape_keys:
        if type(key) is not SymbolId or type(getattr(key, "value", None)) is not int:
            _fail(
                "invalid_shape_binding",
                "input shape keys must be exact int-valued SymbolId values",
            )
        shape_key_values.append(key.value)
    if len(set(shape_key_values)) != len(shape_key_values):
        _fail(
            "invalid_shape_binding",
            "input shape keys must be unique SymbolId values",
        )
    owned_input_shapes: Dict[SymbolId, Tuple[int, ...]] = {}
    for key, key_value in zip(shape_keys, shape_key_values):
        try:
            shape = input_shapes[key]
        except Exception as error:
            raise LoopIRTargetError(
                LoopIRTargetDefect(
                    "invalid_shape_binding",
                    "input shapes could not be snapshotted",
                )
            ) from error
        if (
            type(key) is not SymbolId
            or type(getattr(key, "value", None)) is not int
            or key.value != key_value
        ):
            _fail(
                "invalid_shape_binding",
                "input shape keys changed while values were being snapshotted",
            )
        owned_input_shapes[SymbolId(key_value)] = shape

    # Mapping iteration and lookup can execute caller-controlled code. Own
    # every binding and reverify after custom callbacks so they cannot leave
    # a frozen program in untrusted state.
    if mapping_callbacks_possible:
        verify_program(program)

    lowering: _TargetLowering
    if _sparse_workspace_chain(program):
        result_decl = next(
            (
                decl
                for decl in program.tensors
                if program.outputs and decl.symbol == program.outputs[0]
            ),
            None,
        )
        row_scope = result_decl is not None and tuple(
            level.kind for level in result_decl.levels
        ) == (LevelKind.DENSE, LevelKind.COMPRESSED)
        if row_scope:
            lowering = _RowScopeSparseWorkspaceLowering(
                program, owned_input_shapes, result_shape
            )
        else:
            lowering = _SparseWorkspaceLowering(
                program, owned_input_shapes, result_shape
            )
    elif _parallel_sparse_workspace_chain(program):
        lowering = _ParallelSparseWorkspaceLowering(
            program, owned_input_shapes, result_shape
        )
    elif _dense_domain_mixed_chain(program):
        lowering = _DenseDomainMixedLowering(program, owned_input_shapes, result_shape)
    elif _multi_compressed_assembly_chain(program):
        lowering = _MultiCompressedAssemblyLowering(
            program, owned_input_shapes, result_shape
        )
    else:
        lowering = _TargetLowering(program, owned_input_shapes, result_shape)
    raw_statements = lowering.raw_loop_statements()
    kernel_abi = lowering.kernel_abi()
    assembler = lowering.result_assembler()

    validation_stmts = kernel_abi.emit_validation()
    size_stmts = lowering.result_size_inits()
    if size_stmts:
        # An all-compressed result has no dense level sizes; the legacy
        # pipeline emits neither the initializers nor their comment.
        size_stmts = [
            llir.Comment("Init result tensor level sizes"),
            *size_stmts,
        ]
    prologue_stmts = kernel_abi.emit_input_prologue()
    value_init_stmts = assembler.emit_value_array_init()
    tile_size_stmts = lowering.tile_size_inits()
    level_indices_stmts = lowering.prepare_result_level_indices(
        assembler.emit_level_indices_init()
    )
    if level_indices_stmts:
        level_indices_stmts = [
            llir.Comment("Init result level indices"),
            *level_indices_stmts,
        ]
    final_assembly_stmts = assembler.emit_final_assembly()

    def assemble_body(
        transformed_body: LLIRStatementListArtifact,
        compressed_output_parallel: bool,
    ) -> LLIRRewriteArtifact:
        if compressed_output_parallel != lowering.owns_two_phase_output():
            # Families outside the parallel workspace target never produce
            # two-phase output; the owning family conversely requires the
            # shared pass to have taken ownership, so a silently detached
            # no-op cannot degrade it to an unallocated serial assembly.
            raise LoopIRTargetError(
                LoopIRTargetDefect(
                    (
                        "unsupported_program_shape"
                        if compressed_output_parallel
                        else _SPARSE_WORKSPACE_LOST
                    ),
                    (
                        "LoopIR lowering never produces compressed "
                        "two-phase output assembly"
                        if compressed_output_parallel
                        else "the shared compressed-Where pass did not take "
                        "ownership of the two-phase parallel assembly"
                    ),
                )
            )
        if compressed_output_parallel:
            # The pass owns output allocation, both phase loops, final
            # assembly, and the return; mirror the legacy applied-branch
            # composition exactly.
            return LLIRRewriteArtifact(
                [
                    *validation_stmts,
                    *size_stmts,
                    *prologue_stmts,
                    llir.BlankLine(),
                    *tile_size_stmts,
                    llir.BlankLine(),
                    *transformed_body.statements,
                ]
            )
        body_stmts: List[llir.Stmt] = [
            *validation_stmts,
            *size_stmts,
            *prologue_stmts,
            llir.BlankLine(),
            *level_indices_stmts,
            llir.Comment("Initialize result value array"),
            *value_init_stmts,
            *tile_size_stmts,
            llir.BlankLine(),
            *transformed_body.statements,
            *final_assembly_stmts,
        ]
        return LLIRRewriteArtifact(body_stmts)

    manager = LLIRPassManager.from_compile_options(compile_options)
    try:
        pipeline_result = manager.run_production_pipeline(
            LLIRStatementListArtifact(raw_statements),
            compressed_where_pass_spec=lowering.compressed_where_pass_spec(
                compile_options
            ),
            dense_pointer_pass_spec=DensePointerHoistPassSpec(
                DensePointerHoistContext(
                    value_array_ctypes=lowering.value_array_ctypes()
                )
            ),
            body_assembler=assemble_body,
        )
    except LLIRPassPartialFailure as failure:
        if compilation_context is not None:
            compilation_context.record_llir_pass_runs(
                failure.completed_run_records,
                compile_options=compile_options,
                stage_id=CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
            )
        raise failure.failure from None
    if compilation_context is not None:
        compilation_context.record_llir_pass_runs(
            pipeline_result.run_records,
            compile_options=compile_options,
            stage_id=CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
        )
    assembled = kernel_abi.assemble_function(pipeline_result.artifact.value)
    completed_sparse_workspace = lowering.complete_sparse_workspace(assembled)
    return lowering.complete_relayout(
        lowering.complete_result_tile(
            lowering.complete_panel(
                lowering.complete_parallel(completed_sparse_workspace)
            )
        )
    )


def lower_loopir_to_llir(
    program: LoopProgram,
    *,
    input_shapes: Mapping[SymbolId, Tuple[int, ...]],
    result_shape: Tuple[int, ...],
    compile_options: Optional[CompileOptions] = None,
    compilation_context: Optional[CompilationContext] = None,
) -> llir.Function:
    """Lower one verified LoopIR program to a complete LLIR function.

    Runtime shapes are bound and cross-checked here; the returned function is
    the same structured-LLIR artifact the legacy path produces, ready for the
    exhaustive C++ emitter.  When a compilation context is supplied, this
    boundary owns the complete ``LOOPIR_TO_LLIR_LOWERING`` stage and its
    managed-pass records; callers do not pre-open that stage.
    """

    if (
        compilation_context is not None
        and type(compilation_context) is not CompilationContext
    ):
        raise TypeError("compilation_context must be a CompilationContext")
    if compile_options is None:
        compile_options = (
            compilation_context.compile_options
            if compilation_context is not None
            else CompileOptions.from_environment()
        )
    elif type(compile_options) is not CompileOptions:
        raise TypeError("compile_options must be a CompileOptions snapshot")
    if compilation_context is None:
        return _lower_loopir_to_llir_owned(
            program,
            input_shapes=input_shapes,
            result_shape=result_shape,
            compile_options=compile_options,
            compilation_context=None,
        )

    compilation_context.require_compile_options(
        compile_options,
        stage_id=CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
    )
    token = compilation_context.begin_stage(
        CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
        compile_options=compile_options,
    )
    try:
        lowered = _lower_loopir_to_llir_owned(
            program,
            input_shapes=input_shapes,
            result_shape=result_shape,
            compile_options=compile_options,
            compilation_context=compilation_context,
        )
    except Exception:
        compilation_context.fail_stage(token)
        raise
    compilation_context.complete_stage(token)
    return lowered
