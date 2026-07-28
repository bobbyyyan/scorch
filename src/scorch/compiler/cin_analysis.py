"""Pure ownership analyses and canonical serialization for semantic CIN.

The current CIN node classes remain mutable legacy objects.  This module treats
only their forward structural fields as authoritative and returns deeply
immutable, typed-ID side tables.  Results are recomputed on demand and are never
stored on CIN nodes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union, cast

import torch

from ..format import LevelFormat, LevelType, TensorFormat
from .cin import (
    BinaryOp,
    ForAll,
    IndexExpr,
    IndexStmt,
    IndexVar,
    IndexVarAdd,
    Operation,
    TensorAccess,
    TensorAssign,
    TensorVar,
    TileSizeVar,
    UnaryOp,
    Where,
    Workspace,
    WorkspaceAccess,
    _is_exact_index_stmt,
    _is_index_stmt_instance,
)
from .diagnostics import VerificationError
from .identity import AccessId, IndexId, NodeId, SymbolId
from .compile_options import CompileOptions
from .compilation_context import CompilerStageId, CompilationContext

K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True)
class FrozenMap(Mapping[K, V], Generic[K, V]):
    """A small tuple-backed immutable mapping.

    CIN programs are small enough that linear lookup is preferable to retaining
    a mutable dictionary behind an otherwise frozen analysis result.
    """

    _items: Tuple[Tuple[K, V], ...] = ()

    @classmethod
    def from_items(cls, items: Iterable[Tuple[K, V]]) -> "FrozenMap[K, V]":
        return cls(tuple(items))

    def __getitem__(self, key: K) -> V:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[K]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


class AccessKind(Enum):
    READ = "read"
    WRITE = "write"
    REDUCTION_WRITE = "reduction_write"


@dataclass(frozen=True)
class ParentRelation:
    child_id: NodeId
    parent_id: NodeId
    edge: str


@dataclass(frozen=True)
class IndexBinding:
    binder_id: NodeId
    scope_id: NodeId


@dataclass(frozen=True)
class IndexDefinition:
    index_id: IndexId
    definition_node_id: NodeId
    display_name: str
    bindings: Tuple[IndexBinding, ...]


@dataclass(frozen=True)
class IndexUse:
    index_id: IndexId
    access_id: AccessId
    node_id: NodeId
    scope_id: NodeId
    axis: int


@dataclass(frozen=True)
class SymbolDefinition:
    symbol_id: SymbolId
    definition_node_id: NodeId
    scope_id: NodeId
    display_name: str
    is_workspace: bool
    rank: int
    shape: Optional[Tuple[int, ...]]
    mode_order: Tuple[int, ...]
    level_types: Tuple[str, ...]
    level_bit_widths: Tuple[Optional[int], ...]
    dtype: str


@dataclass(frozen=True)
class SymbolUse:
    symbol_id: SymbolId
    access_id: AccessId
    node_id: NodeId
    scope_id: NodeId
    kind: AccessKind


@dataclass(frozen=True)
class AccessInfo:
    access_id: AccessId
    node_id: NodeId
    tensor_id: SymbolId
    index_ids: Tuple[IndexId, ...]
    scope_id: NodeId
    kind: AccessKind
    order: int


@dataclass(frozen=True)
class AccessLayoutInfo:
    access_id: AccessId
    tensor_id: SymbolId
    logical_index_ids: Tuple[IndexId, ...]
    storage_index_ids: Tuple[IndexId, ...]
    level_types: Tuple[LevelType, ...]
    physical_extents: Tuple[Optional[int], ...]
    scope_id: NodeId
    kind: AccessKind
    is_workspace: bool


@dataclass(frozen=True)
class AssignmentInfo:
    assignment_id: NodeId
    lhs_access_id: AccessId
    rhs_access_ids: Tuple[AccessId, ...]
    update_op: Optional[Operation]
    lhs_index_ids: Tuple[IndexId, ...]
    reduction_index_ids: Tuple[IndexId, ...]
    multiplicative_access_ids: Optional[Tuple[AccessId, ...]]


@dataclass(frozen=True)
class AccessOccurrence:
    access_id: AccessId
    node_id: NodeId
    scope_id: NodeId
    kind: AccessKind
    order: int
    path: Tuple[str, ...]


EntityId = Union[NodeId, AccessId, IndexId, SymbolId]


@dataclass(frozen=True)
class CINDiagnostic:
    code: str
    message: str
    path: Tuple[str, ...]
    entity_id: Optional[EntityId] = None
    stage: str = "normalized_cin"
    pass_name: str = "verify_cin"


_MAX_CIN_STRUCTURE_DEPTH = 256
_MAX_CIN_TENSOR_RANK = 64
_MAX_RUNTIME_EXTENT = (1 << 63) - 1
_MISSING_CIN_FIELD = object()


def _is_supported_cin_node_type(node_type: type) -> bool:
    """Use identity comparisons without hashing/equality on a hostile type."""

    return (
        node_type is ForAll
        or node_type is Where
        or node_type is TensorAssign
        or node_type is BinaryOp
        or node_type is UnaryOp
        or node_type is WorkspaceAccess
        or node_type is TensorAccess
        or node_type is IndexVarAdd
        or node_type is IndexVar
        or node_type is Workspace
        or node_type is TensorVar
    )


def _is_exact_identity(value: object, identity_type: type) -> bool:
    """Recognize one safe, hashable stable-ID wrapper without invoking hooks."""

    if type(value) is not identity_type:
        return False
    state = object.__getattribute__(value, "__dict__")
    if type(state) is not dict or len(state) != 1:
        return False
    key = next(iter(state), None)
    return (
        type(key) is str
        and key == "value"
        and type(state["value"]) is int
        and 0 <= state["value"] <= _MAX_RUNTIME_EXTENT
    )


def _safe_exact_dict_value(value: object, field_name: str) -> object:
    """Read one exact-string key without hashing forged stored-state keys."""

    state = object.__getattribute__(value, "__dict__")
    if type(state) is not dict:
        return None
    for key, stored in state.items():
        if type(key) is str and key == field_name:
            return stored
    return None


def _preflight_cin_structure_impl(  # noqa: C901
    cin: IndexStmt,
    *,
    allow_legacy_schedule_aliases: bool = False,
) -> Tuple[Tuple[CINDiagnostic, ...], bool]:
    """Validate stored forward structure without recursive Python calls.

    CIN remains a mutable legacy object graph, so callers can forge missing
    fields, cycles, and arbitrarily deep trees after construction.  The full
    ownership analysis intentionally remains recursive while the CIN-to-LoopIR
    strangler is in progress; this bounded iterative preflight makes that
    recursion safe and gives every malformed structural boundary the same
    structured diagnostic contract.

    Completed objects are not rejected here when another parent references
    them.  The ownership analysis retains authority for
    ``duplicate_node_reference`` and related identity diagnostics, while an
    object reached again before its exit frame is unambiguously a cycle.

    Only the exact built-in CIN node classes are admitted.  This is important
    because downstream legacy consumers use properties and helper methods in
    addition to the stored forward fields: accepting an arbitrary subclass
    would let a stateful descriptor show the preflight one graph and a later
    consumer another.  Exact-class admission makes the stored fields the sole
    authoritative view without executing attacker-controlled descriptors.

    The private ``allow_legacy_schedule_aliases`` mode exists only for the
    direct legacy-lowering adapter.  It additionally validates mutable
    scheduler compatibility fields because that adapter's forward copier
    consumes them; normalization deliberately discards those fields and
    therefore does not let their contents affect semantic admission.
    Workspace insertion historically clones producer and consumer syntax while
    retaining same-kind ``NodeId`` values and clones logical ``IndexVar``
    objects while retaining their paired ``NodeId``, ``IndexId``, and display
    name.  The adapter does not consume node identity and canonicalizes the
    index aliases before lowering. Normalization, analysis, LoopIR, and request
    identity never enable this compatibility mode.
    """

    if type(allow_legacy_schedule_aliases) is not bool:
        raise TypeError("legacy schedule alias policy must be an exact bool")

    diagnostics: List[CINDiagnostic] = []
    legacy_alias_candidates: List[
        Tuple[
            type,
            object,
            object,
            Tuple[str, ...],
            Tuple[str, ...],
            str,
        ]
    ] = []
    workspace_branch_prefixes: set[Tuple[str, ...]] = set()
    where_paths: set[Tuple[str, ...]] = set()
    root_has_workspace_marker = False
    active: set[int] = set()
    complete: set[int] = set()
    forward_index_objects: Dict[int, Tuple[IndexVar, Tuple[str, ...]]] = {}
    forward_access_objects: Dict[
        int,
        Tuple[TensorAccess, Tuple[str, ...]],
    ] = {}
    pending_index_references: List[Tuple[IndexVar, Tuple[str, ...], str]] = []
    pending_access_references: List[Tuple[TensorAccess, Tuple[str, ...], str]] = []
    pending_tile_size_vars: List[Tuple[TileSizeVar, Tuple[str, ...]]] = []
    workspace_access_occurrences: Dict[
        int,
        List[Tuple[str, ...]],
    ] = {}
    depth_reported = False
    identity_owners: Dict[type, Dict[int, Tuple[object, Tuple[str, ...]]]] = {
        NodeId: {},
        IndexId: {},
        SymbolId: {},
        AccessId: {},
    }
    identity_cache: Dict[
        Tuple[int, type],
        Tuple[object, Optional[int]],
    ] = {}
    tensor_ranks: Dict[int, int] = {}
    pending_accesses: List[Tuple[TensorAccess, Tuple[str, ...]]] = []
    stack: List[Tuple[bool, object, Tuple[str, ...], int]] = [
        (False, cin, ("root",), 0)
    ]

    def diagnose(
        code: str,
        message: str,
        path: Tuple[str, ...],
    ) -> None:
        diagnostics.append(CINDiagnostic(code, message, path))

    def exact_identity_value(
        value: object,
        identity_type: type,
    ) -> Optional[int]:
        cache_key = (id(value), identity_type)
        cached = identity_cache.get(cache_key)
        if cached is not None and cached[0] is value:
            return cached[1]
        identity_value = (
            cast(int, object.__getattribute__(value, "__dict__")["value"])
            if _is_exact_identity(value, identity_type)
            else None
        )
        identity_cache[cache_key] = (value, identity_value)
        return identity_value

    def record_identity(
        value: object,
        identity_type: type,
        owner: object,
        path: Tuple[str, ...],
        duplicate_code: str,
    ) -> None:
        identity_value = exact_identity_value(value, identity_type)
        if identity_value is None:
            return
        previous = identity_owners[identity_type].get(identity_value)
        if previous is not None and previous[0] is not owner:
            if (
                allow_legacy_schedule_aliases
                and identity_type is NodeId
                and type(previous[0]) is type(owner)
            ):
                legacy_alias_candidates.append(
                    (
                        identity_type,
                        previous[0],
                        owner,
                        previous[1],
                        path,
                        duplicate_code,
                    )
                )
                return
            if allow_legacy_schedule_aliases and identity_type is IndexId:
                previous_node_id = _safe_exact_dict_value(previous[0], "node_id")
                current_node_id = _safe_exact_dict_value(owner, "node_id")
                previous_node_value = exact_identity_value(previous_node_id, NodeId)
                current_node_value = exact_identity_value(current_node_id, NodeId)
                previous_name = _safe_exact_dict_value(previous[0], "_name")
                current_name = _safe_exact_dict_value(owner, "_name")
                if (
                    type(previous[0]) is IndexVar
                    and type(owner) is IndexVar
                    and previous_node_value is not None
                    and previous_node_value == current_node_value
                    and type(previous_name) is str
                    and type(current_name) is str
                    and previous_name == current_name
                ):
                    legacy_alias_candidates.append(
                        (
                            identity_type,
                            previous[0],
                            owner,
                            previous[1],
                            path,
                            duplicate_code,
                        )
                    )
                    return
            diagnose(
                duplicate_code,
                f"{identity_type.__name__} belongs to distinct CIN entities",
                path,
            )
            return
        identity_owners[identity_type][identity_value] = (owner, path)

    def stored_field(
        node: object,
        field_name: str,
        path: Tuple[str, ...],
    ) -> object:
        state = object.__getattribute__(node, "__dict__")
        if type(state) is not dict:
            diagnose(
                "invalid_cin_field",
                "CIN node stored state must be an exact dict",
                path,
            )
            return _MISSING_CIN_FIELD
        if field_name not in state:
            diagnose(
                "missing_cin_field",
                f"{type(node).__name__}.{field_name} is missing",
                path + (field_name,),
            )
            return _MISSING_CIN_FIELD
        return state[field_name]

    def typed_child(
        node: object,
        field_name: str,
        expected_type: type,
        path: Tuple[str, ...],
        children: List[Tuple[object, Tuple[str, ...]]],
    ) -> None:
        value = stored_field(node, field_name, path)
        if value is _MISSING_CIN_FIELD:
            return
        child_path = path + (field_name,)
        value_type = type(value)
        if expected_type is IndexStmt:
            valid_type = (
                value_type is ForAll
                or value_type is Where
                or value_type is TensorAssign
            )
        elif expected_type is TensorAccess:
            valid_type = value_type is TensorAccess or value_type is WorkspaceAccess
        else:
            valid_type = value_type is expected_type
        if not valid_type:
            diagnose(
                "invalid_cin_field",
                f"{type(node).__name__}.{field_name} must be a "
                f"{expected_type.__name__}",
                child_path,
            )
            return
        children.append((value, child_path))

    def expression_child(
        node: object,
        field_name: str,
        path: Tuple[str, ...],
        children: List[Tuple[object, Tuple[str, ...]]],
    ) -> None:
        """Admit only expressions the normalizer and analyses can execute."""

        value = stored_field(node, field_name, path)
        if value is _MISSING_CIN_FIELD:
            return
        child_path = path + (field_name,)
        value_type = type(value)
        if (
            value_type is not TensorAccess
            and value_type is not WorkspaceAccess
            and value_type is not BinaryOp
            and value_type is not UnaryOp
        ):
            code = (
                "unsupported_expression"
                if value_type is TensorVar
                or value_type is IndexVar
                or value_type is IndexVarAdd
                else "invalid_cin_field"
            )
            diagnose(
                code,
                f"{type(node).__name__}.{field_name} must be an executable "
                "CIN expression",
                child_path,
            )
            return
        children.append((value, child_path))

    validated_tile_size_vars: set[int] = set()

    def validate_legacy_index_reference(
        value: object,
        path: Tuple[str, ...],
        owner: str,
    ) -> None:
        if type(value) is not IndexVar:
            diagnose(
                "invalid_cin_field",
                f"{owner} must reference an exact IndexVar",
                path,
            )
            return
        node_id = _safe_exact_dict_value(value, "node_id")
        index_id = _safe_exact_dict_value(value, "index_id")
        name = _safe_exact_dict_value(value, "_name")
        if (
            exact_identity_value(node_id, NodeId) is None
            or exact_identity_value(index_id, IndexId) is None
            or type(name) is not str
        ):
            diagnose(
                "invalid_cin_field",
                f"{owner} must reference a well-formed IndexVar",
                path,
            )
            return
        pending_index_references.append((cast(IndexVar, value), path, owner))

    def validate_tile_size_var(
        value: object,
        path: Tuple[str, ...],
    ) -> None:
        if type(value) is not TileSizeVar:
            diagnose(
                "invalid_cin_field",
                "tile metadata must be an exact TileSizeVar",
                path,
            )
            return
        object_id = id(value)
        if object_id in validated_tile_size_vars:
            return
        validated_tile_size_vars.add(object_id)
        pending_tile_size_vars.append((cast(TileSizeVar, value), path))
        state = object.__getattribute__(value, "__dict__")
        if type(state) is not dict or any(type(key) is not str for key in state):
            diagnose(
                "invalid_cin_field",
                "TileSizeVar stored state must be an exact string-keyed dict",
                path,
            )
            return
        if "_index_var" not in state:
            diagnose(
                "missing_cin_field",
                "TileSizeVar._index_var is missing",
                path + ("_index_var",),
            )
        tile_node_id = _safe_exact_dict_value(value, "node_id")
        if exact_identity_value(tile_node_id, NodeId) is None:
            diagnose(
                "invalid_node_id",
                "TileSizeVar.node_id must be an exact int-valued NodeId",
                path + ("node_id",),
            )
        for field_name in ("outer_index_var", "inner_index_var"):
            field = _safe_exact_dict_value(value, field_name)
            validate_legacy_index_reference(
                field,
                path + (field_name,),
                f"TileSizeVar.{field_name}",
            )
        index_var = _safe_exact_dict_value(value, "_index_var")
        if index_var is not None:
            validate_legacy_index_reference(
                index_var,
                path + ("_index_var",),
                "TileSizeVar._index_var",
            )
        size = _safe_exact_dict_value(value, "size")
        if type(size) is not int or size <= 0 or size > _MAX_RUNTIME_EXTENT:
            diagnose(
                "invalid_cin_field",
                "TileSizeVar.size must be a positive signed-int64 exact int",
                path + ("size",),
            )
        name = _safe_exact_dict_value(value, "_name")
        if type(name) is not str:
            diagnose(
                "invalid_cin_field",
                "TileSizeVar._name must be an exact str",
                path + ("_name",),
            )
        unroll = _safe_exact_dict_value(value, "unroll")
        if type(unroll) is not bool:
            diagnose(
                "invalid_cin_field",
                "TileSizeVar.unroll must be an exact bool",
                path + ("unroll",),
            )
        inserted_workspace = _safe_exact_dict_value(value, "inserted_workspace")
        no_tile_list = _safe_exact_dict_value(value, "no_tile_list")
        if type(inserted_workspace) is not bool or type(no_tile_list) is not list:
            diagnose(
                "invalid_cin_field",
                "TileSizeVar compatibility state is malformed",
                path,
            )
        elif inserted_workspace:
            diagnose(
                "invalid_cin_field",
                "TileSizeVar.inserted_workspace must remain false",
                path + ("inserted_workspace",),
            )
        else:
            for position, index_var in enumerate(no_tile_list):
                entry_path = path + ("no_tile_list", f"[{position}]")
                validate_legacy_index_reference(
                    index_var,
                    entry_path,
                    "TileSizeVar.no_tile_list entry",
                )

    while stack:
        exiting, node, path, depth = stack.pop()
        object_id = id(node)
        if exiting:
            active.discard(object_id)
            complete.add(object_id)
            continue
        if depth > _MAX_CIN_STRUCTURE_DEPTH:
            if not depth_reported:
                diagnose(
                    "cin_structure_depth_exceeded",
                    "CIN forward structure exceeds the supported depth "
                    f"{_MAX_CIN_STRUCTURE_DEPTH}",
                    path,
                )
                depth_reported = True
            continue
        if type(node) is WorkspaceAccess:
            workspace_access_occurrences.setdefault(object_id, []).append(path)
        if object_id in active:
            diagnose(
                "cyclic_cin_structure",
                "a CIN node is reachable from itself",
                path,
            )
            continue
        if object_id in complete:
            continue

        node_type = type(node)
        if not _is_supported_cin_node_type(node_type):
            diagnose(
                "invalid_cin_field",
                "CIN node must have an exact supported built-in node type",
                path,
            )
            continue
        if node_type is IndexVar:
            forward_index_objects[object_id] = (cast(IndexVar, node), path)
        elif node_type is TensorAccess or node_type is WorkspaceAccess:
            forward_access_objects[object_id] = (cast(TensorAccess, node), path)
        elif node_type is Where:
            where_paths.add(path)
        node_state = object.__getattribute__(node, "__dict__")
        if type(node_state) is not dict:
            diagnose(
                "invalid_cin_field",
                "CIN node stored state must be an exact dict",
                path,
            )
            continue
        if any(type(key) is not str for key in node_state):
            diagnose(
                "invalid_cin_field",
                "CIN node stored-state keys must be exact strings",
                path,
            )
            continue

        active.add(object_id)
        stack.append((True, node, path, depth))
        children: List[Tuple[object, Tuple[str, ...]]] = []

        if allow_legacy_schedule_aliases:
            # Normalization deliberately discards these mutable scheduler
            # compatibility fields.  Validate them only at the raw legacy
            # lowering boundary, whose forward copier consumes them.  Treating
            # them as semantic CIN would make ignored caller metadata affect
            # normalization and source comparison.
            inserted_workspace = node_state.get(
                "inserted_workspace",
                _MISSING_CIN_FIELD,
            )
            if inserted_workspace is _MISSING_CIN_FIELD:
                # Frozen operation dataclasses retain their exact class default
                # without materializing this false compatibility flag in
                # ``__dict__``.
                if node_type is not BinaryOp and node_type is not UnaryOp:
                    diagnose(
                        "missing_cin_field",
                        f"{node_type.__name__}.inserted_workspace is missing",
                        path + ("inserted_workspace",),
                    )
            elif type(inserted_workspace) is not bool:
                diagnose(
                    "invalid_cin_field",
                    f"{node_type.__name__}.inserted_workspace must be an exact bool",
                    path + ("inserted_workspace",),
                )
            if path == ("root",):
                root_has_workspace_marker = inserted_workspace is True
            no_tile_list = stored_field(node, "no_tile_list", path)
            if no_tile_list is not _MISSING_CIN_FIELD:
                if type(no_tile_list) is not list:
                    diagnose(
                        "invalid_cin_field",
                        f"{node_type.__name__}.no_tile_list must be an exact list",
                        path + ("no_tile_list",),
                    )
                elif any(type(index_var) is not IndexVar for index_var in no_tile_list):
                    diagnose(
                        "invalid_cin_field",
                        f"{node_type.__name__}.no_tile_list must contain IndexVar "
                        "objects",
                        path + ("no_tile_list",),
                    )
                else:
                    for position, index_var in enumerate(no_tile_list):
                        validate_legacy_index_reference(
                            index_var,
                            path + ("no_tile_list", f"[{position}]"),
                            f"{node_type.__name__}.no_tile_list entry",
                        )

        # Stable-reference typing remains the full ownership verifier's
        # authority.  Presence and non-shadowing are structural requirements
        # because normalization reads this field even in release mode.
        node_id = stored_field(node, "node_id", path)
        if (
            node_id is not _MISSING_CIN_FIELD
            and exact_identity_value(node_id, NodeId) is None
        ):
            diagnose(
                "invalid_node_id",
                f"{node_type.__name__}.node_id must be an exact int-valued NodeId",
                path + ("node_id",),
            )
        else:
            record_identity(
                node_id,
                NodeId,
                node,
                path + ("node_id",),
                "duplicate_node_id",
            )

        if type(node) is ForAll:
            typed_child(node, "index_var", IndexVar, path, children)
            typed_child(node, "stmt", IndexStmt, path, children)
            parallel = stored_field(node, "parallel", path)
            if parallel is not _MISSING_CIN_FIELD and parallel is not None:
                if type(parallel) is not bool:
                    diagnose(
                        "invalid_cin_field",
                        "ForAll.parallel must be an exact bool or None",
                        path + ("parallel",),
                    )
        elif type(node) is Where:
            typed_child(node, "producer", IndexStmt, path, children)
            typed_child(node, "consumer", IndexStmt, path, children)
        elif type(node) is TensorAssign:
            typed_child(node, "lhs", TensorAccess, path, children)
            expression_child(node, "rhs", path, children)
            op = stored_field(node, "op", path)
            if op is not _MISSING_CIN_FIELD and op is not None:
                if type(op) is not Operation:
                    diagnose(
                        "invalid_cin_field",
                        "TensorAssign.op must be an exact Operation or None",
                        path + ("op",),
                    )
        elif type(node) is BinaryOp:
            expression_child(node, "left", path, children)
            expression_child(node, "right", path, children)
            op = stored_field(node, "op", path)
            if op is not _MISSING_CIN_FIELD and type(op) is not Operation:
                diagnose(
                    "invalid_cin_field",
                    "BinaryOp.op must be an exact Operation",
                    path + ("op",),
                )
        elif type(node) is UnaryOp:
            expression_child(node, "expr", path, children)
            op = stored_field(node, "op", path)
            if op is not _MISSING_CIN_FIELD and type(op) is not Operation:
                diagnose(
                    "invalid_cin_field",
                    "UnaryOp.op must be an exact Operation",
                    path + ("op",),
                )
        elif type(node) is TensorAccess or type(node) is WorkspaceAccess:
            if type(node) is WorkspaceAccess:
                for edge_position in range(len(path) - 1, -1, -1):
                    if path[edge_position] in ("producer", "consumer"):
                        candidate_prefix = path[:edge_position]
                        if candidate_prefix in where_paths:
                            workspace_branch_prefixes.add(candidate_prefix)
                        break
            tensor = stored_field(node, "tensor", path)
            if tensor is not _MISSING_CIN_FIELD:
                expected_tensor_type = (
                    Workspace if type(node) is WorkspaceAccess else TensorVar
                )
                if type(tensor) is not expected_tensor_type:
                    diagnose(
                        "invalid_cin_field",
                        f"{type(node).__name__}.tensor must be an exact "
                        f"{expected_tensor_type.__name__}",
                        path + ("tensor",),
                    )
                else:
                    children.append((tensor, path + ("tensor",)))
            indices = stored_field(node, "indices", path)
            if indices is not _MISSING_CIN_FIELD:
                indices_type = type(indices)
                if indices_type is not list and indices_type is not tuple:
                    diagnose(
                        "invalid_cin_field",
                        "TensorAccess.indices must be an exact list or tuple",
                        path + ("indices",),
                    )
                else:
                    typed_indices = cast(
                        Union[List[object], Tuple[object, ...]],
                        indices,
                    )
                    if len(typed_indices) > _MAX_CIN_TENSOR_RANK:
                        diagnose(
                            "invalid_cin_field",
                            "TensorAccess.indices exceeds the supported rank",
                            path + ("indices",),
                        )
                    else:
                        for axis, index_var in enumerate(typed_indices):
                            index_path = path + (f"indices[{axis}]",)
                            if type(index_var) is not IndexVar:
                                diagnose(
                                    "invalid_cin_field",
                                    "TensorAccess.indices entries must be exact "
                                    "IndexVar objects",
                                    index_path,
                                )
                            else:
                                children.append((index_var, index_path))
            access_id = stored_field(node, "access_id", path)
            if (
                access_id is not _MISSING_CIN_FIELD
                and exact_identity_value(access_id, AccessId) is None
            ):
                diagnose(
                    "invalid_access_id",
                    "TensorAccess.access_id must be an exact int-valued AccessId",
                    path + ("access_id",),
                )
            else:
                record_identity(
                    access_id,
                    AccessId,
                    node,
                    path + ("access_id",),
                    "duplicate_access_id",
                )
            tensor_id = stored_field(node, "tensor_id", path)
            if (
                tensor_id is not _MISSING_CIN_FIELD
                and exact_identity_value(tensor_id, SymbolId) is None
            ):
                diagnose(
                    "invalid_symbol_reference",
                    "TensorAccess.tensor_id must be an exact int-valued SymbolId",
                    path + ("tensor_id",),
                )
            index_ids = stored_field(node, "index_ids", path)
            if index_ids is not _MISSING_CIN_FIELD:
                if type(index_ids) is not tuple:
                    diagnose(
                        "invalid_index_reference",
                        "TensorAccess.index_ids must be an exact tuple",
                        path + ("index_ids",),
                    )
                elif len(index_ids) > _MAX_CIN_TENSOR_RANK:
                    diagnose(
                        "invalid_index_reference",
                        "TensorAccess.index_ids exceeds the supported rank",
                        path + ("index_ids",),
                    )
                else:
                    for axis, index_id in enumerate(index_ids):
                        if exact_identity_value(index_id, IndexId) is None:
                            diagnose(
                                "invalid_index_reference",
                                "TensorAccess.index_ids entries must be exact "
                                "int-valued IndexId values",
                                path + ("index_ids", f"[{axis}]"),
                            )
            if type(node) is WorkspaceAccess:
                workspace = stored_field(node, "wksp", path)
                if workspace is not _MISSING_CIN_FIELD:
                    if type(workspace) is not Workspace:
                        diagnose(
                            "invalid_cin_field",
                            "WorkspaceAccess.wksp must be an exact Workspace",
                            path + ("wksp",),
                        )
                    elif tensor is not _MISSING_CIN_FIELD and workspace is not tensor:
                        diagnose(
                            "invalid_cin_field",
                            "WorkspaceAccess.wksp must be its tensor",
                            path + ("wksp",),
                        )
            pending_accesses.append((cast(TensorAccess, node), path))
        elif type(node) is IndexVarAdd:
            typed_child(node, "lhs", IndexVar, path, children)
            typed_child(node, "rhs", IndexVar, path, children)
        elif type(node) is IndexVar:
            expression = stored_field(node, "_expr", path)
            if expression is not _MISSING_CIN_FIELD and expression is not None:
                if type(expression) is not IndexVarAdd:
                    diagnose(
                        "invalid_cin_field",
                        "IndexVar._expr must be an exact IndexVarAdd or None",
                        path + ("_expr",),
                    )
                else:
                    children.append((expression, path + ("_expr",)))
            index_id = stored_field(node, "index_id", path)
            if (
                index_id is not _MISSING_CIN_FIELD
                and exact_identity_value(index_id, IndexId) is None
            ):
                diagnose(
                    "invalid_index_id",
                    "IndexVar.index_id must be an exact int-valued IndexId",
                    path + ("index_id",),
                )
            else:
                record_identity(
                    index_id,
                    IndexId,
                    node,
                    path + ("index_id",),
                    "duplicate_index_id",
                )
            name = stored_field(node, "_name", path)
            if name is not _MISSING_CIN_FIELD and type(name) is not str:
                diagnose(
                    "invalid_cin_field",
                    "IndexVar._name must be an exact str",
                    path + ("_name",),
                )
            if allow_legacy_schedule_aliases:
                parent = stored_field(node, "_parent", path)
                if parent is not _MISSING_CIN_FIELD and parent is not None:
                    validate_legacy_index_reference(
                        parent,
                        path + ("_parent",),
                        "IndexVar._parent",
                    )
                for flag_name in ("is_tiled", "is_outer", "is_inner"):
                    flag = stored_field(node, flag_name, path)
                    if flag is not _MISSING_CIN_FIELD and type(flag) is not bool:
                        diagnose(
                            "invalid_cin_field",
                            f"IndexVar.{flag_name} must be an exact bool",
                            path + (flag_name,),
                        )
                tile_size_var = stored_field(node, "tile_size_var", path)
                if (
                    tile_size_var is not _MISSING_CIN_FIELD
                    and tile_size_var is not None
                ):
                    validate_tile_size_var(
                        tile_size_var,
                        path + ("tile_size_var",),
                    )
                legacy_accesses = stored_field(
                    node,
                    "_legacy_tensor_accesses",
                    path,
                )
                if legacy_accesses is not _MISSING_CIN_FIELD:
                    if type(legacy_accesses) is not list:
                        diagnose(
                            "invalid_cin_field",
                            "IndexVar._legacy_tensor_accesses must be an exact list",
                            path + ("_legacy_tensor_accesses",),
                        )
                    else:
                        for position, access in enumerate(legacy_accesses):
                            access_path = path + (
                                "_legacy_tensor_accesses",
                                f"[{position}]",
                            )
                            if (
                                type(access) is not TensorAccess
                                and type(access) is not WorkspaceAccess
                            ):
                                diagnose(
                                    "invalid_cin_field",
                                    "IndexVar._legacy_tensor_accesses entries must "
                                    "be exact TensorAccess objects",
                                    access_path,
                                )
                            else:
                                pending_access_references.append(
                                    (
                                        cast(TensorAccess, access),
                                        access_path,
                                        "IndexVar._legacy_tensor_accesses entry",
                                    )
                                )
        elif type(node) is TensorVar or type(node) is Workspace:
            symbol_id = stored_field(node, "symbol_id", path)
            if (
                symbol_id is not _MISSING_CIN_FIELD
                and exact_identity_value(symbol_id, SymbolId) is None
            ):
                diagnose(
                    "invalid_symbol_id",
                    "TensorVar.symbol_id must be an exact int-valued SymbolId",
                    path + ("symbol_id",),
                )
            else:
                record_identity(
                    symbol_id,
                    SymbolId,
                    node,
                    path + ("symbol_id",),
                    "duplicate_symbol_id",
                )
            name = stored_field(node, "_name", path)
            if name is not _MISSING_CIN_FIELD and type(name) is not str:
                diagnose(
                    "invalid_cin_field",
                    "TensorVar._name must be an exact str",
                    path + ("_name",),
                )
            tensor_format = stored_field(node, "_format", path)
            format_rank: Optional[int] = None
            if tensor_format is not _MISSING_CIN_FIELD:
                if (
                    tensor_format is not None
                    and type(tensor_format) is not TensorFormat
                ):
                    diagnose(
                        "invalid_cin_field",
                        "TensorVar._format must be an exact TensorFormat or None",
                        path + ("_format",),
                    )
                elif type(tensor_format) is TensorFormat:
                    format_state = object.__getattribute__(
                        tensor_format,
                        "__dict__",
                    )
                    if type(format_state) is not dict:
                        diagnose(
                            "invalid_cin_field",
                            "TensorVar._format stored state must be an exact dict",
                            path + ("_format",),
                        )
                    else:
                        format_keys = tuple(format_state)
                    if type(format_state) is dict and (
                        len(format_keys) != 1
                        or type(format_keys[0]) is not str
                        or format_keys[0] != "_level_formats"
                    ):
                        diagnose(
                            "invalid_cin_field",
                            "TensorVar._format has malformed stored state",
                            path + ("_format",),
                        )
                    elif type(format_state) is dict:
                        level_formats = format_state["_level_formats"]
                        if type(level_formats) is not tuple:
                            diagnose(
                                "invalid_cin_field",
                                "TensorVar._format levels must be an exact tuple",
                                path + ("_format", "_level_formats"),
                            )
                        else:
                            format_rank = len(level_formats)
                            if format_rank > _MAX_CIN_TENSOR_RANK:
                                diagnose(
                                    "invalid_cin_field",
                                    "TensorVar._format exceeds the supported rank",
                                    path + ("_format",),
                                )
                            for level, level_format in enumerate(
                                level_formats[:_MAX_CIN_TENSOR_RANK],
                            ):
                                level_path = path + (
                                    "_format",
                                    f"_level_formats[{level}]",
                                )
                                if type(level_format) is not LevelFormat:
                                    diagnose(
                                        "invalid_cin_field",
                                        "TensorFormat levels must be exact "
                                        "LevelFormat values",
                                        level_path,
                                    )
                                    continue
                                level_state = object.__getattribute__(
                                    level_format,
                                    "__dict__",
                                )
                                if type(level_state) is not dict:
                                    diagnose(
                                        "invalid_cin_field",
                                        "LevelFormat stored state must be an exact dict",
                                        level_path,
                                    )
                                    continue
                                level_keys = tuple(level_state)
                                if (
                                    len(level_keys) != 2
                                    or any(type(key) is not str for key in level_keys)
                                    or set(level_keys) != {"_mode", "_bit_width"}
                                ):
                                    diagnose(
                                        "invalid_cin_field",
                                        "LevelFormat has malformed stored state",
                                        level_path,
                                    )
                                    continue
                                mode = level_state["_mode"]
                                bit_width = level_state["_bit_width"]
                                if type(mode) is not LevelType:
                                    diagnose(
                                        "invalid_cin_field",
                                        "LevelFormat._mode must be an exact LevelType",
                                        level_path + ("_mode",),
                                    )
                                if bit_width is not None and (
                                    type(bit_width) is not int
                                    or bit_width <= 0
                                    or bit_width > _MAX_RUNTIME_EXTENT
                                ):
                                    diagnose(
                                        "invalid_cin_field",
                                        "LevelFormat._bit_width must be a positive "
                                        "signed-int64 exact int or None",
                                        level_path + ("_bit_width",),
                                    )
            shape = stored_field(node, "shape", path)
            shape_rank: Optional[int] = None
            if shape is not _MISSING_CIN_FIELD and shape is not None:
                invalid_shape = type(shape) is not tuple
                if type(shape) is tuple:
                    invalid_shape = len(shape) > _MAX_CIN_TENSOR_RANK
                    product = 1
                    for extent in shape[:_MAX_CIN_TENSOR_RANK]:
                        if (
                            type(extent) is not int
                            or extent < 0
                            or extent > _MAX_RUNTIME_EXTENT
                            or (extent != 0 and product > _MAX_RUNTIME_EXTENT // extent)
                        ):
                            invalid_shape = True
                            break
                        product *= extent
                if invalid_shape:
                    diagnose(
                        "invalid_cin_field",
                        "TensorVar.shape must be an exact rank-bounded tuple whose "
                        "extents are nonnegative int64 values and whose element "
                        "count fits signed int64, or None",
                        path + ("shape",),
                    )
                else:
                    shape_rank = len(cast(Tuple[object, ...], shape))
            dtype = stored_field(node, "dtype", path)
            if dtype is not _MISSING_CIN_FIELD and type(dtype) is not torch.dtype:
                diagnose(
                    "invalid_cin_field",
                    "TensorVar.dtype must be an exact torch.dtype",
                    path + ("dtype",),
                )
            mode_order = stored_field(node, "mode_order", path)
            mode_rank: Optional[int] = None
            if mode_order is not _MISSING_CIN_FIELD and mode_order is not None:
                mode_order_type = type(mode_order)
                if mode_order_type is not list and mode_order_type is not tuple:
                    diagnose(
                        "invalid_cin_field",
                        "TensorVar.mode_order must be an exact list or tuple of ints "
                        "or None",
                        path + ("mode_order",),
                    )
                else:
                    typed_mode_order = cast(
                        Union[List[object], Tuple[object, ...]],
                        mode_order,
                    )
                    if len(typed_mode_order) > _MAX_CIN_TENSOR_RANK or any(
                        type(mode) is not int for mode in typed_mode_order
                    ):
                        diagnose(
                            "invalid_cin_field",
                            "TensorVar.mode_order must be an exact list or tuple "
                            "of ints or None",
                            path + ("mode_order",),
                        )
                    else:
                        mode_rank = len(typed_mode_order)
                        expected_rank = (
                            len(shape)
                            if type(shape) is tuple
                            else (
                                format_rank
                                if format_rank is not None
                                else len(typed_mode_order)
                            )
                        )
                        if len(typed_mode_order) != expected_rank or set(
                            typed_mode_order
                        ) != set(range(expected_rank)):
                            diagnose(
                                "invalid_cin_field",
                                "TensorVar.mode_order must be an exact permutation "
                                f"of range({expected_rank})",
                                path + ("mode_order",),
                            )
            if type(node) is Workspace:
                dim = stored_field(node, "dim", path)
                valid_dim = dim is not _MISSING_CIN_FIELD and not (
                    type(dim) is not int or dim < 0 or dim > _MAX_CIN_TENSOR_RANK
                )
                if dim is not _MISSING_CIN_FIELD and not valid_dim:
                    diagnose(
                        "invalid_cin_field",
                        "Workspace.dim must be a supported nonnegative exact rank",
                        path + ("dim",),
                    )
                dense = stored_field(node, "dense", path)
                if dense is not _MISSING_CIN_FIELD and type(dense) is not bool:
                    diagnose(
                        "invalid_cin_field",
                        "Workspace.dense must be an exact bool",
                        path + ("dense",),
                    )
                if allow_legacy_schedule_aliases:
                    tile_size_var = stored_field(node, "_tile_size_var", path)
                    if (
                        tile_size_var is not _MISSING_CIN_FIELD
                        and tile_size_var is not None
                    ):
                        validate_tile_size_var(
                            tile_size_var,
                            path + ("_tile_size_var",),
                        )
                    workspace_accesses = stored_field(
                        node,
                        "workspace_accesses",
                        path,
                    )
                    if workspace_accesses is not _MISSING_CIN_FIELD:
                        if type(workspace_accesses) is not list:
                            diagnose(
                                "invalid_cin_field",
                                "Workspace.workspace_accesses must be an exact list",
                                path + ("workspace_accesses",),
                            )
                        else:
                            for position, access in enumerate(workspace_accesses):
                                access_path = path + (
                                    "workspace_accesses",
                                    f"[{position}]",
                                )
                                if type(access) is not WorkspaceAccess:
                                    diagnose(
                                        "invalid_cin_field",
                                        "Workspace.workspace_accesses entries must be "
                                        "exact WorkspaceAccess objects",
                                        access_path,
                                    )
                                else:
                                    pending_access_references.append(
                                        (
                                            cast(WorkspaceAccess, access),
                                            access_path,
                                            "Workspace.workspace_accesses entry",
                                        )
                                    )
                rank_candidates = [
                    rank
                    for rank in (
                        cast(int, dim) if valid_dim else None,
                        mode_rank,
                    )
                    if rank is not None
                ]
            else:
                rank_candidates = [
                    rank
                    for rank in (format_rank, shape_rank, mode_rank)
                    if rank is not None
                ]
            if rank_candidates:
                declared_rank = rank_candidates[0]
                if any(rank != declared_rank for rank in rank_candidates[1:]):
                    diagnose(
                        "invalid_cin_field",
                        "TensorVar format, shape, mode order, and workspace rank "
                        "must agree",
                        path,
                    )
                else:
                    tensor_ranks[id(node)] = declared_rank
                    if declared_rank > 0 and mode_order is None:
                        diagnose(
                            "invalid_cin_field",
                            "rankful TensorVar.mode_order must declare its "
                            "physical-to-logical permutation",
                            path + ("mode_order",),
                        )

        for child, child_path in reversed(children):
            stack.append((False, child, child_path, depth + 1))

    for index_var, path, owner in pending_index_references:
        if id(index_var) not in forward_index_objects and owner not in (
            "IndexVar._parent",
            "TileSizeVar._index_var",
        ):
            diagnose(
                "invalid_cin_field",
                f"{owner} references an IndexVar outside the forward CIN graph",
                path,
            )
    for access, path, owner in pending_access_references:
        if id(access) not in forward_access_objects:
            diagnose(
                "invalid_cin_field",
                f"{owner} references a TensorAccess outside the forward CIN graph",
                path,
            )

    def valid_detached_tile_parent(
        parent: IndexVar,
        outer: IndexVar,
        inner: IndexVar,
    ) -> bool:
        """Recognize one scheduler-created logical split parent.

        Two successive legacy tiles can replace every forward occurrence of an
        earlier logical index while the later outer/inner components retain
        that logical object as their parent. It is schedule-authoritative, but
        intentionally not a forward expression node. Admit only the exact,
        bounded state produced by ``Scheduler.add_tile``; arbitrary foreign
        parent graphs remain invalid and the forward copier follows no other
        compatibility state.
        """

        parent_state = object.__getattribute__(parent, "__dict__")
        if type(parent_state) is not dict or any(
            type(key) is not str for key in parent_state
        ):
            return False
        if (
            exact_identity_value(parent_state.get("node_id"), NodeId) is None
            or exact_identity_value(parent_state.get("index_id"), IndexId) is None
            or type(parent_state.get("_name")) is not str
            or parent_state.get("inserted_workspace") is not False
            or "_parent" not in parent_state
            or parent_state["_parent"] is not None
            or parent_state.get("is_tiled") is not True
            or parent_state.get("is_outer") is not False
            or parent_state.get("is_inner") is not False
            or "tile_size_var" not in parent_state
            or parent_state["tile_size_var"] is not None
        ):
            return False
        parent_node_value = exact_identity_value(
            parent_state.get("node_id"),
            NodeId,
        )
        parent_index_value = exact_identity_value(
            parent_state.get("index_id"),
            IndexId,
        )
        parent_name = parent_state.get("_name")
        matching_forward_alias = any(
            exact_identity_value(
                _safe_exact_dict_value(candidate, "node_id"),
                NodeId,
            )
            == parent_node_value
            and exact_identity_value(
                _safe_exact_dict_value(candidate, "index_id"),
                IndexId,
            )
            == parent_index_value
            and _safe_exact_dict_value(candidate, "_name") == parent_name
            for candidate, _ in forward_index_objects.values()
        )
        if not matching_forward_alias:
            return False
        no_tile_list = parent_state.get("no_tile_list")
        legacy_accesses = parent_state.get("_legacy_tensor_accesses")
        if (
            type(no_tile_list) is not list
            or any(
                type(index_var) is not IndexVar
                or id(index_var) not in forward_index_objects
                for index_var in no_tile_list
            )
            or type(legacy_accesses) is not list
            or any(
                (
                    type(access) is not TensorAccess
                    and type(access) is not WorkspaceAccess
                )
                or id(access) not in forward_access_objects
                for access in legacy_accesses
            )
        ):
            return False
        expression = parent_state.get("_expr")
        if type(expression) is not IndexVarAdd:
            return False
        expression_state = object.__getattribute__(expression, "__dict__")
        if type(expression_state) is not dict or any(
            type(key) is not str for key in expression_state
        ):
            return False
        expression_no_tile = expression_state.get("no_tile_list", [])
        return (
            exact_identity_value(expression_state.get("node_id"), NodeId) is not None
            and expression_state.get("inserted_workspace", False) is False
            and type(expression_no_tile) is list
            and all(
                type(index_var) is IndexVar and id(index_var) in forward_index_objects
                for index_var in expression_no_tile
            )
            and expression_state.get("lhs") is outer
            and expression_state.get("rhs") is inner
        )

    for index_var, path in forward_index_objects.values():
        if not allow_legacy_schedule_aliases:
            # Normalization keeps the semantic IndexVar expression but resets
            # every mutable scheduling backlink/role below.
            continue
        expression = _safe_exact_dict_value(index_var, "_expr")
        parent = _safe_exact_dict_value(index_var, "_parent")
        is_tiled = _safe_exact_dict_value(index_var, "is_tiled")
        is_outer = _safe_exact_dict_value(index_var, "is_outer")
        is_inner = _safe_exact_dict_value(index_var, "is_inner")
        tile_size_var = _safe_exact_dict_value(index_var, "tile_size_var")
        has_split_role = is_outer is True or is_inner is True

        if is_outer is True and is_inner is True:
            diagnose(
                "invalid_cin_field",
                "IndexVar cannot be both a tile outer and tile inner index",
                path,
            )
        if type(expression) is IndexVarAdd and is_tiled is True:
            if (
                parent is not None
                or is_outer is not False
                or is_inner is not False
                or tile_size_var is not None
            ):
                diagnose(
                    "invalid_cin_field",
                    "a split logical IndexVar must own only its IndexVarAdd",
                    path,
                )
        elif is_tiled is True:
            diagnose(
                "invalid_cin_field",
                "a tiled logical IndexVar must own an IndexVarAdd",
                path + ("_expr",),
            )

        if has_split_role:
            if (
                is_tiled is not False
                or type(parent) is not IndexVar
                or type(tile_size_var) is not TileSizeVar
            ):
                diagnose(
                    "invalid_cin_field",
                    "a tile component must reference its logical parent and "
                    "TileSizeVar",
                    path,
                )
        elif parent is not None or tile_size_var is not None:
            diagnose(
                "invalid_cin_field",
                "only tile component IndexVar objects may retain tile backlinks",
                path,
            )

    for tile_size_var, path in pending_tile_size_vars:
        outer = _safe_exact_dict_value(tile_size_var, "outer_index_var")
        inner = _safe_exact_dict_value(tile_size_var, "inner_index_var")
        base = _safe_exact_dict_value(tile_size_var, "_index_var")
        if type(outer) is not IndexVar or type(inner) is not IndexVar:
            continue
        outer_parent = _safe_exact_dict_value(outer, "_parent")
        inner_parent = _safe_exact_dict_value(inner, "_parent")
        logical_parent = outer_parent if type(outer_parent) is IndexVar else None
        logical_expression = (
            _safe_exact_dict_value(logical_parent, "_expr")
            if logical_parent is not None
            else None
        )
        lhs = (
            _safe_exact_dict_value(logical_expression, "lhs")
            if type(logical_expression) is IndexVarAdd
            else None
        )
        rhs = (
            _safe_exact_dict_value(logical_expression, "rhs")
            if type(logical_expression) is IndexVarAdd
            else None
        )
        if (
            outer is inner
            or id(outer) not in forward_index_objects
            or id(inner) not in forward_index_objects
            or logical_parent is None
            or (
                id(logical_parent) not in forward_index_objects
                and not valid_detached_tile_parent(logical_parent, outer, inner)
            )
            or inner_parent is not logical_parent
            or type(logical_expression) is not IndexVarAdd
            or lhs is not outer
            or rhs is not inner
            or _safe_exact_dict_value(logical_parent, "is_tiled") is not True
            or _safe_exact_dict_value(outer, "is_tiled") is not False
            or _safe_exact_dict_value(outer, "is_outer") is not True
            or _safe_exact_dict_value(outer, "is_inner") is not False
            or _safe_exact_dict_value(outer, "tile_size_var") is not tile_size_var
            or _safe_exact_dict_value(inner, "is_tiled") is not False
            or _safe_exact_dict_value(inner, "is_outer") is not False
            or _safe_exact_dict_value(inner, "is_inner") is not True
            or _safe_exact_dict_value(inner, "tile_size_var") is not tile_size_var
            or (base is not None and base is not logical_parent)
        ):
            diagnose(
                "invalid_cin_field",
                "TileSizeVar links must describe one exact outer/inner split",
                path,
            )

    for access, path in pending_accesses:
        tensor = _safe_exact_dict_value(access, "tensor")
        indices = _safe_exact_dict_value(access, "indices")
        index_ids = _safe_exact_dict_value(access, "index_ids")
        tensor_id = _safe_exact_dict_value(access, "tensor_id")
        if type(tensor) is not TensorVar and type(tensor) is not Workspace:
            continue
        tensor_format = _safe_exact_dict_value(tensor, "_format")
        if (
            type(tensor) is TensorVar
            and tensor_format is None
            and (type(indices) is list or type(indices) is tuple)
            and len(indices) > 0
        ):
            diagnose(
                "invalid_cin_field",
                "a rankful accessed TensorVar must declare a TensorFormat",
                path + ("tensor", "_format"),
            )
        if type(indices) is not list and type(indices) is not tuple:
            continue
        if type(index_ids) is not tuple:
            continue
        stored_rank = tensor_ranks.get(id(tensor))
        if stored_rank is not None and len(indices) != stored_rank:
            diagnose(
                "tensor_access_rank_mismatch",
                "TensorAccess index rank must match its tensor rank",
                path + ("indices",),
            )
        if len(index_ids) != len(indices):
            diagnose(
                "index_reference_mismatch",
                "TensorAccess.index_ids must mirror its indices",
                path + ("index_ids",),
            )
        else:
            for axis, (index, index_id) in enumerate(zip(indices, index_ids)):
                stored_index_id = (
                    _safe_exact_dict_value(index, "index_id")
                    if type(index) is IndexVar
                    else None
                )
                reference_value = exact_identity_value(index_id, IndexId)
                stored_value = exact_identity_value(stored_index_id, IndexId)
                if (
                    reference_value is not None
                    and stored_value is not None
                    and reference_value != stored_value
                ):
                    diagnose(
                        "index_reference_mismatch",
                        "TensorAccess.index_ids must mirror its indices",
                        path + ("index_ids", f"[{axis}]"),
                    )
                if (
                    reference_value is not None
                    and reference_value not in identity_owners[IndexId]
                ):
                    diagnose(
                        "dangling_index_reference",
                        "TensorAccess.index_ids references an undefined IndexId",
                        path + ("index_ids", f"[{axis}]"),
                    )
        tensor_symbol_id = _safe_exact_dict_value(tensor, "symbol_id")
        reference_symbol = exact_identity_value(tensor_id, SymbolId)
        stored_symbol = exact_identity_value(tensor_symbol_id, SymbolId)
        if (
            reference_symbol is not None
            and stored_symbol is not None
            and reference_symbol != stored_symbol
        ):
            diagnose(
                "symbol_reference_mismatch",
                "TensorAccess.tensor_id must mirror its tensor",
                path + ("tensor_id",),
            )
        if (
            reference_symbol is not None
            and reference_symbol not in identity_owners[SymbolId]
        ):
            diagnose(
                "dangling_symbol_reference",
                "TensorAccess.tensor_id references an undefined SymbolId",
                path + ("tensor_id",),
            )

    def workspace_branch(
        path: Tuple[str, ...],
        prefix: Tuple[str, ...],
    ) -> Optional[str]:
        if len(path) <= len(prefix) or path[: len(prefix)] != prefix:
            return None
        edge = path[len(prefix)]
        return edge if edge == "producer" or edge == "consumer" else None

    def workspace_suffix(
        path: Tuple[str, ...],
        prefix: Tuple[str, ...],
    ) -> Optional[Tuple[str, ...]]:
        branch = workspace_branch(path, prefix)
        if branch is None:
            return None
        return path[len(prefix) + 1 :]

    def is_access_index_path(path: Tuple[str, ...]) -> bool:
        return any(
            component.startswith("indices[") and component.endswith("]")
            for component in path
        )

    def same_index_identity(first: object, second: object) -> bool:
        if type(first) is not IndexVar or type(second) is not IndexVar:
            return False
        first_node = exact_identity_value(
            _safe_exact_dict_value(first, "node_id"),
            NodeId,
        )
        second_node = exact_identity_value(
            _safe_exact_dict_value(second, "node_id"),
            NodeId,
        )
        first_index = exact_identity_value(
            _safe_exact_dict_value(first, "index_id"),
            IndexId,
        )
        second_index = exact_identity_value(
            _safe_exact_dict_value(second, "index_id"),
            IndexId,
        )
        first_name = _safe_exact_dict_value(first, "_name")
        second_name = _safe_exact_dict_value(second, "_name")
        return (
            first_node is not None
            and first_node == second_node
            and first_index is not None
            and first_index == second_index
            and type(first_name) is str
            and first_name == second_name
        )

    def paired_statement_clone(
        first: object,
        second: object,
    ) -> bool:
        if type(first) is ForAll and type(second) is ForAll:
            first_index = _safe_exact_dict_value(first, "index_var")
            second_index = _safe_exact_dict_value(second, "index_var")
            return same_index_identity(
                first_index, second_index
            ) and _safe_exact_dict_value(first, "parallel") == _safe_exact_dict_value(
                second, "parallel"
            )
        if type(first) is TensorAssign and type(second) is TensorAssign:
            return _safe_exact_dict_value(first, "op") is _safe_exact_dict_value(
                second,
                "op",
            )
        return False

    legacy_aliases_used = False
    candidate_workspace_prefix = (
        next(iter(workspace_branch_prefixes))
        if root_has_workspace_marker and len(workspace_branch_prefixes) == 1
        else None
    )
    workspace_prefix = None
    if (
        candidate_workspace_prefix is not None
        and len(workspace_access_occurrences) == 1
    ):
        access_paths = next(iter(workspace_access_occurrences.values()))
        access_roles = {
            (
                workspace_branch(path, candidate_workspace_prefix),
                path[-1] if path else None,
            )
            for path in access_paths
        }
        if len(access_paths) == 2 and access_roles == {
            ("producer", "lhs"),
            ("consumer", "rhs"),
        }:
            workspace_prefix = candidate_workspace_prefix
    for (
        identity_type,
        previous_owner,
        current_owner,
        previous_path,
        path,
        duplicate_code,
    ) in legacy_alias_candidates:
        allowed = False
        if workspace_prefix is not None:
            previous_branch = workspace_branch(previous_path, workspace_prefix)
            branch = workspace_branch(path, workspace_prefix)
            previous_suffix = workspace_suffix(previous_path, workspace_prefix)
            suffix = workspace_suffix(path, workspace_prefix)
            if identity_type is IndexId or (
                identity_type is NodeId
                and type(previous_owner) is IndexVar
                and type(current_owner) is IndexVar
            ):
                allowed = same_index_identity(previous_owner, current_owner) and (
                    is_access_index_path(previous_path)
                    or is_access_index_path(path)
                    or (
                        {previous_branch, branch} == {"producer", "consumer"}
                        and previous_suffix == suffix
                    )
                )
            elif identity_type is NodeId:
                allowed = (
                    {previous_branch, branch} == {"producer", "consumer"}
                    and previous_suffix == suffix
                    and paired_statement_clone(previous_owner, current_owner)
                )
        if allowed:
            legacy_aliases_used = True
        else:
            diagnose(
                duplicate_code,
                f"{identity_type.__name__} belongs to distinct CIN entities",
                path,
            )

    return tuple(diagnostics), legacy_aliases_used


def _preflight_cin_structure(cin: IndexStmt) -> Tuple[CINDiagnostic, ...]:
    """Run the strict normalized-CIN structural preflight."""

    diagnostics, aliases_used = _preflight_cin_structure_impl(cin)
    assert not aliases_used
    return diagnostics


def _raise_cin_verification(diagnostics: Tuple[CINDiagnostic, ...]) -> None:
    first = diagnostics[0]
    raise VerificationError(
        f"stage=normalized CIN: {first.code} at "
        f"{'/'.join(first.path)}: {first.message}",
        diagnostics=cast(Tuple[object, ...], diagnostics),
    )


def verify_cin_structure(cin: IndexStmt) -> None:
    """Fail closed on malformed CIN forward structure before recursive work."""

    if not _is_index_stmt_instance(cin):
        raise TypeError("verify_cin_structure expects an IndexStmt")
    if id(cin) in _TRUSTED_CIN_ROOTS.get():
        return
    diagnostics = _preflight_cin_structure(cin)
    if diagnostics:
        _raise_cin_verification(diagnostics)


def _verify_legacy_cin_lowering_structure(cin: IndexStmt) -> bool:
    """Validate raw legacy-lowering input without rejecting scheduler aliases.

    A plan-free ``Scheduler.auto_schedule`` result is a private compatibility
    graph, not normalized CIN: workspace insertion duplicates same-kind
    statement nodes and paired-node, same-named logical index objects while
    preserving their stable IDs. ``legacy_cin_working_copy`` canonicalizes
    those index aliases before any recursive legacy consumer runs. Admit
    exactly that historical shape here while retaining every field, type,
    cycle, depth, reference, symbol, and access-identity check from the shared
    preflight.
    """

    if not _is_index_stmt_instance(cin):
        raise TypeError("legacy CIN lowering expects an IndexStmt")
    diagnostics, aliases_used = _preflight_cin_structure_impl(
        cin,
        allow_legacy_schedule_aliases=True,
    )
    if diagnostics:
        _raise_cin_verification(diagnostics)
    return aliases_used


@dataclass(frozen=True)
class CINAnalysis:
    root_id: NodeId
    parents: FrozenMap[NodeId, ParentRelation]
    node_scopes: FrozenMap[NodeId, NodeId]
    scope_parents: FrozenMap[NodeId, Optional[NodeId]]
    symbol_definitions: FrozenMap[SymbolId, SymbolDefinition]
    symbol_uses: FrozenMap[SymbolId, Tuple[SymbolUse, ...]]
    index_definitions: FrozenMap[IndexId, IndexDefinition]
    index_uses: FrozenMap[IndexId, Tuple[IndexUse, ...]]
    accesses: FrozenMap[AccessId, AccessInfo]
    access_layouts: FrozenMap[AccessId, AccessLayoutInfo]
    assignments: FrozenMap[NodeId, AssignmentInfo]
    access_occurrences: Tuple[AccessOccurrence, ...]
    tensor_accesses: FrozenMap[SymbolId, Tuple[AccessId, ...]]
    access_order: Tuple[AccessId, ...]
    assignment_order: Tuple[NodeId, ...]
    free_index_ids: Tuple[IndexId, ...]
    reduction_index_ids: Tuple[IndexId, ...]
    diagnostics: Tuple[CINDiagnostic, ...]


class _IdPreflight:
    """Validate typed IDs and global NodeId uniqueness before full analysis."""

    def __init__(self) -> None:
        self.node_objects: Dict[NodeId, object] = {}
        self.node_paths: Dict[NodeId, Tuple[str, ...]] = {}
        self.visited: set[int] = set()
        self.diagnostics: List[CINDiagnostic] = []

    def diagnose(
        self,
        code: str,
        message: str,
        path: Tuple[str, ...],
        entity_id: Optional[EntityId] = None,
    ) -> None:
        self.diagnostics.append(CINDiagnostic(code, message, path, entity_id))

    def record_node(self, node: object, path: Tuple[str, ...]) -> None:
        node_id = object.__getattribute__(node, "__dict__").get("node_id")
        if not _is_exact_identity(node_id, NodeId):
            self.diagnose(
                "invalid_node_id",
                f"{type(node).__name__}.node_id must be an exact int-valued NodeId",
                path,
            )
            return
        assert type(node_id) is NodeId
        previous = self.node_objects.get(node_id)
        if previous is not None and previous is not node:
            self.diagnose(
                "duplicate_node_id",
                f"NodeId {node_id.value} belongs to distinct CIN entities at "
                f"{self.node_paths[node_id]} and {path}",
                path,
                node_id,
            )
            return
        self.node_objects[node_id] = node
        self.node_paths[node_id] = path

    def visit_index(self, index_var: object, path: Tuple[str, ...]) -> None:
        if not isinstance(index_var, IndexVar):
            self.diagnose(
                "invalid_index_reference",
                "access or binder does not reference an IndexVar",
                path,
            )
            return
        self.record_node(index_var, path)
        index_id = object.__getattribute__(index_var, "__dict__").get("index_id")
        if not _is_exact_identity(index_id, IndexId):
            self.diagnose(
                "invalid_index_id",
                "IndexVar.index_id must be an exact int-valued IndexId",
                path,
            )
        object_id = id(index_var)
        if object_id in self.visited:
            return
        self.visited.add(object_id)
        if isinstance(index_var._expr, IndexVarAdd):
            self.record_node(index_var._expr, path + ("expr",))
            self.visit_index(index_var._expr.lhs, path + ("expr", "lhs"))
            self.visit_index(index_var._expr.rhs, path + ("expr", "rhs"))

    def visit_tensor(self, tensor: object, path: Tuple[str, ...]) -> None:
        if not isinstance(tensor, TensorVar):
            self.diagnose(
                "invalid_symbol_reference",
                "access does not reference a TensorVar",
                path,
            )
            return
        self.record_node(tensor, path)
        symbol_id = object.__getattribute__(tensor, "__dict__").get("symbol_id")
        if not _is_exact_identity(symbol_id, SymbolId):
            self.diagnose(
                "invalid_symbol_id",
                "TensorVar.symbol_id must be an exact int-valued SymbolId",
                path,
            )

    def visit_access(self, access: TensorAccess, path: Tuple[str, ...]) -> None:
        self.record_node(access, path)
        state = object.__getattribute__(access, "__dict__")
        if not _is_exact_identity(state.get("access_id"), AccessId):
            self.diagnose(
                "invalid_access_id",
                "TensorAccess.access_id must be an exact int-valued AccessId",
                path,
            )
        if not _is_exact_identity(state.get("tensor_id"), SymbolId):
            self.diagnose(
                "invalid_symbol_reference",
                "TensorAccess.tensor_id must be an exact int-valued SymbolId",
                path + ("tensor_ref",),
            )
        self.visit_tensor(state.get("tensor"), path + ("tensor",))

        indices = state.get("indices")
        if not isinstance(indices, (list, tuple)):
            self.diagnose(
                "invalid_index_reference",
                "TensorAccess.indices must be a sequence of IndexVar references",
                path + ("indices",),
            )
            indices = ()
        for axis, index_var in enumerate(indices):
            self.visit_index(index_var, path + (f"indices[{axis}]",))

        index_ids = state.get("index_ids")
        if not isinstance(index_ids, tuple):
            self.diagnose(
                "invalid_index_reference",
                "TensorAccess.index_ids must be an immutable tuple of IndexId",
                path + ("index_refs",),
            )
            return
        for axis, index_id in enumerate(index_ids):
            if not _is_exact_identity(index_id, IndexId):
                self.diagnose(
                    "invalid_index_reference",
                    "TensorAccess.index_ids entries must be exact int-valued "
                    "IndexId values",
                    path + (f"index_refs[{axis}]",),
                )

    def visit_expr(self, expr: object, path: Tuple[str, ...]) -> None:
        if isinstance(expr, TensorAccess):
            self.visit_access(expr, path)
            return
        if isinstance(expr, BinaryOp):
            self.record_node(expr, path)
            self.visit_expr(expr.left, path + ("left",))
            self.visit_expr(expr.right, path + ("right",))
            return
        if isinstance(expr, UnaryOp):
            self.record_node(expr, path)
            self.visit_expr(expr.expr, path + ("expr",))

    def visit_stmt(self, stmt: object, path: Tuple[str, ...]) -> None:
        if not isinstance(stmt, IndexStmt):
            return
        self.record_node(stmt, path)
        if isinstance(stmt, ForAll):
            self.visit_index(stmt.index_var, path + ("index_var",))
            self.visit_stmt(stmt.stmt, path + ("stmt",))
        elif isinstance(stmt, Where):
            self.visit_stmt(stmt.producer, path + ("producer",))
            self.visit_stmt(stmt.consumer, path + ("consumer",))
        elif isinstance(stmt, TensorAssign):
            self.visit_expr(stmt.lhs, path + ("lhs",))
            self.visit_expr(stmt.rhs, path + ("rhs",))


def _empty_analysis(
    root_id: NodeId,
    diagnostics: Tuple[CINDiagnostic, ...],
) -> CINAnalysis:
    return CINAnalysis(
        root_id=root_id,
        parents=FrozenMap(),
        node_scopes=FrozenMap(),
        scope_parents=FrozenMap.from_items(((root_id, None),)),
        symbol_definitions=FrozenMap(),
        symbol_uses=FrozenMap(),
        index_definitions=FrozenMap(),
        index_uses=FrozenMap(),
        accesses=FrozenMap(),
        access_layouts=FrozenMap(),
        assignments=FrozenMap(),
        access_occurrences=(),
        tensor_accesses=FrozenMap(),
        access_order=(),
        assignment_order=(),
        free_index_ids=(),
        reduction_index_ids=(),
        diagnostics=diagnostics,
    )


def _tensor_metadata(
    tensor: TensorVar,
) -> Tuple[
    int,
    Optional[Tuple[int, ...]],
    Tuple[int, ...],
    Tuple[str, ...],
    Tuple[Optional[int], ...],
    str,
]:
    shape = tuple(tensor.shape) if tensor.shape is not None else None
    mode_order = tuple(tensor.mode_order or ())
    tensor_format = tensor.format
    level_types = (
        tuple(level.value for level in tensor_format.get_level_types())
        if tensor_format is not None
        else ()
    )
    level_bit_widths = (
        tuple(level.bit_width for level in tensor_format.get_level_formats())
        if tensor_format is not None
        else ()
    )
    if isinstance(tensor, Workspace):
        rank = tensor.dim
    elif shape is not None:
        rank = len(shape)
    elif level_types:
        rank = len(level_types)
    else:
        rank = len(mode_order)
    return (
        rank,
        shape,
        mode_order,
        level_types,
        level_bit_widths,
        str(tensor.dtype),
    )


def _compute_cin_analysis(cin: IndexStmt) -> CINAnalysis:  # noqa: C901
    """Compute immutable ownership/use/access facts without mutating ``cin``."""

    # This exhaustive traversal stays local until the next planned seam extracts
    # the shared typed CIN walker/rewriter. Keeping it explicit here avoids
    # prematurely coupling Phase 1 ownership analysis to pass infrastructure.

    if not _is_index_stmt_instance(cin):
        raise TypeError("analyze_cin expects an IndexStmt")

    structural_diagnostics = (
        () if id(cin) in _TRUSTED_CIN_ROOTS.get() else _preflight_cin_structure(cin)
    )
    if structural_diagnostics:
        raw_root_id = _safe_exact_dict_value(cin, "node_id")
        root_id = (
            cast(NodeId, raw_root_id)
            if _is_exact_identity(raw_root_id, NodeId)
            else NodeId(-1)
        )
        return _empty_analysis(root_id, structural_diagnostics)

    preflight = _IdPreflight()
    preflight.visit_stmt(cin, ("root",))
    raw_root_id = _safe_exact_dict_value(cin, "node_id")
    root_id = (
        cast(NodeId, raw_root_id)
        if _is_exact_identity(raw_root_id, NodeId)
        else NodeId(-1)
    )
    if preflight.diagnostics:
        return _empty_analysis(root_id, tuple(preflight.diagnostics))
    parent_relations: Dict[NodeId, ParentRelation] = {}
    node_scopes: Dict[NodeId, NodeId] = {}
    scope_parents: Dict[NodeId, Optional[NodeId]] = {root_id: None}
    node_objects: Dict[NodeId, object] = {}
    node_paths: Dict[NodeId, Tuple[str, ...]] = {}

    index_objects: Dict[IndexId, IndexVar] = {}
    index_node_ids: Dict[IndexId, NodeId] = {}
    index_names: Dict[IndexId, str] = {}
    index_bindings: Dict[IndexId, List[IndexBinding]] = {}
    index_order: List[IndexId] = []
    index_uses: Dict[IndexId, List[IndexUse]] = {}

    symbol_objects: Dict[SymbolId, TensorVar] = {}
    symbol_definitions: Dict[SymbolId, SymbolDefinition] = {}
    symbol_order: List[SymbolId] = []
    symbol_uses: Dict[SymbolId, List[SymbolUse]] = {}

    access_objects: Dict[AccessId, TensorAccess] = {}
    accesses: Dict[AccessId, AccessInfo] = {}
    access_layouts: Dict[AccessId, AccessLayoutInfo] = {}
    assignments: Dict[NodeId, AssignmentInfo] = {}
    access_occurrences: List[AccessOccurrence] = []
    tensor_accesses: Dict[SymbolId, List[AccessId]] = {}
    access_order: List[AccessId] = []
    assignment_order: List[NodeId] = []
    free_index_ids: List[IndexId] = []
    diagnostics: List[CINDiagnostic] = []

    def diagnose(
        code: str,
        message: str,
        path: Tuple[str, ...],
        entity_id: Optional[EntityId] = None,
    ) -> None:
        diagnostics.append(CINDiagnostic(code, message, path, entity_id))

    def record_node(
        node: object,
        parent_id: Optional[NodeId],
        edge: str,
        scope_id: NodeId,
        path: Tuple[str, ...],
    ) -> bool:
        node_id = getattr(node, "node_id", None)
        if not isinstance(node_id, NodeId):
            diagnose(
                "missing_node_id",
                f"{type(node).__name__} has no typed NodeId",
                path,
            )
            return False
        previous = node_objects.get(node_id)
        if previous is not None:
            code = (
                "duplicate_node_reference" if previous is node else "duplicate_node_id"
            )
            diagnose(
                code,
                f"NodeId {node_id.value} occurs at both "
                f"{node_paths[node_id]} and {path}",
                path,
                node_id,
            )
            return False
        node_objects[node_id] = node
        node_paths[node_id] = path
        node_scopes[node_id] = scope_id
        if parent_id is not None:
            parent_relations[node_id] = ParentRelation(node_id, parent_id, edge)
        return True

    def register_index(index_var: IndexVar, path: Tuple[str, ...]) -> None:
        index_id = index_var.index_id
        previous = index_objects.get(index_id)
        if previous is None:
            index_objects[index_id] = index_var
            index_node_ids[index_id] = index_var.node_id
            index_names[index_id] = index_var.name
            index_order.append(index_id)
        elif previous is not index_var:
            diagnose(
                "duplicate_index_id",
                f"IndexId {index_id.value} belongs to distinct index objects",
                path,
                index_id,
            )

    def register_symbol(
        tensor: TensorVar,
        owner_scope: NodeId,
        path: Tuple[str, ...],
    ) -> None:
        symbol_id = tensor.symbol_id
        previous = symbol_objects.get(symbol_id)
        if previous is not None:
            if previous is not tensor:
                diagnose(
                    "duplicate_symbol_id",
                    f"SymbolId {symbol_id.value} belongs to distinct symbols",
                    path,
                    symbol_id,
                )
            return

        symbol_objects[symbol_id] = tensor
        symbol_order.append(symbol_id)
        (
            rank,
            shape,
            mode_order,
            level_types,
            level_bit_widths,
            dtype,
        ) = _tensor_metadata(tensor)
        symbol_definitions[symbol_id] = SymbolDefinition(
            symbol_id=symbol_id,
            definition_node_id=tensor.node_id,
            scope_id=owner_scope,
            display_name=tensor.name,
            is_workspace=isinstance(tensor, Workspace),
            rank=rank,
            shape=shape,
            mode_order=mode_order,
            level_types=level_types,
            level_bit_widths=level_bit_widths,
            dtype=dtype,
        )

        if shape is not None and len(shape) != rank:
            diagnose(
                "tensor_shape_rank_mismatch",
                f"tensor {tensor.name!r} shape rank {len(shape)} != {rank}",
                path,
                symbol_id,
            )
        if level_types and len(level_types) != rank:
            diagnose(
                "tensor_format_rank_mismatch",
                f"tensor {tensor.name!r} format rank {len(level_types)} != {rank}",
                path,
                symbol_id,
            )
        if mode_order and (
            len(mode_order) != rank or sorted(mode_order) != list(range(rank))
        ):
            diagnose(
                "tensor_mode_order_mismatch",
                f"tensor {tensor.name!r} mode_order is not a rank-{rank} permutation",
                path,
                symbol_id,
            )

    def visit_access(
        access: TensorAccess,
        parent_id: NodeId,
        edge: str,
        scope_id: NodeId,
        workspace_scope: Optional[NodeId],
        kind: AccessKind,
        path: Tuple[str, ...],
    ) -> None:
        recorded = record_node(access, parent_id, edge, scope_id, path)
        actual_symbol_id = access.tensor.symbol_id
        owner_scope = (
            workspace_scope
            if isinstance(access.tensor, Workspace) and workspace_scope is not None
            else root_id
        )
        register_symbol(access.tensor, owner_scope, path + ("tensor",))

        tensor_id = access.tensor_id
        if tensor_id != actual_symbol_id:
            diagnose(
                "symbol_reference_mismatch",
                f"access SymbolId {tensor_id.value} does not match embedded "
                f"symbol {actual_symbol_id.value}",
                path,
                tensor_id,
            )

        stable_index_ids = tuple(access.index_ids)
        actual_index_ids = tuple(index_var.index_id for index_var in access.indices)
        for axis, index_var in enumerate(access.indices):
            register_index(index_var, path + (f"indices[{axis}]",))
        if stable_index_ids != actual_index_ids:
            diagnose(
                "index_reference_mismatch",
                "access stable index references do not match embedded indices",
                path,
                access.access_id,
            )

        access_id = access.access_id
        order = len(access_occurrences)
        occurrence = AccessOccurrence(
            access_id,
            access.node_id,
            scope_id,
            kind,
            order,
            path,
        )
        access_occurrences.append(occurrence)
        access_order.append(access_id)
        tensor_accesses.setdefault(tensor_id, []).append(access_id)

        previous_access = access_objects.get(access_id)
        if previous_access is not None:
            code = (
                "duplicate_access_reference"
                if previous_access is access
                else "duplicate_access_id"
            )
            diagnose(
                code,
                f"AccessId {access_id.value} occurs more than once",
                path,
                access_id,
            )
        else:
            access_objects[access_id] = access
            accesses[access_id] = AccessInfo(
                access_id,
                access.node_id,
                tensor_id,
                stable_index_ids,
                scope_id,
                kind,
                order,
            )
            mode_order = tuple(access.tensor.mode_order or ())
            if len(mode_order) == len(stable_index_ids) and sorted(mode_order) == list(
                range(len(stable_index_ids))
            ):
                storage_index_ids = tuple(
                    stable_index_ids[logical_axis] for logical_axis in mode_order
                )
            else:
                storage_index_ids = stable_index_ids
            level_types = (
                tuple(access.tensor.format.get_level_types())
                if access.tensor.format is not None
                else ()
            )
            shape = tuple(access.tensor.shape or ())
            physical_extents = tuple(
                shape[level] if level < len(shape) else None
                for level in range(len(storage_index_ids))
            )
            access_layouts[access_id] = AccessLayoutInfo(
                access_id=access_id,
                tensor_id=tensor_id,
                logical_index_ids=stable_index_ids,
                storage_index_ids=storage_index_ids,
                level_types=level_types,
                physical_extents=physical_extents,
                scope_id=scope_id,
                kind=kind,
                is_workspace=isinstance(access.tensor, Workspace),
            )

        symbol_uses.setdefault(tensor_id, []).append(
            SymbolUse(tensor_id, access_id, access.node_id, scope_id, kind)
        )
        for axis, index_id in enumerate(stable_index_ids):
            index_uses.setdefault(index_id, []).append(
                IndexUse(index_id, access_id, access.node_id, scope_id, axis)
            )

        definition = symbol_definitions.get(actual_symbol_id)
        if definition is not None and len(stable_index_ids) != definition.rank:
            diagnose(
                "tensor_access_rank_mismatch",
                f"access rank {len(stable_index_ids)} does not match tensor "
                f"rank {definition.rank}",
                path,
                access_id,
            )

        if kind in (AccessKind.WRITE, AccessKind.REDUCTION_WRITE) and not isinstance(
            access.tensor, Workspace
        ):
            for index_id in stable_index_ids:
                if index_id not in free_index_ids:
                    free_index_ids.append(index_id)

        if not recorded:
            return

    def expression_access_ids(expr: IndexExpr) -> Tuple[AccessId, ...]:
        if isinstance(expr, TensorAccess):
            return (expr.access_id,)
        if isinstance(expr, BinaryOp):
            return expression_access_ids(expr.left) + expression_access_ids(expr.right)
        if isinstance(expr, UnaryOp):
            return expression_access_ids(expr.expr)
        return ()

    def multiplicative_access_ids(
        expr: IndexExpr,
    ) -> Optional[Tuple[AccessId, ...]]:
        if isinstance(expr, TensorAccess):
            return (expr.access_id,)
        if isinstance(expr, BinaryOp) and expr.op == Operation.MUL:
            left_ids = multiplicative_access_ids(expr.left)
            right_ids = multiplicative_access_ids(expr.right)
            if left_ids is not None and right_ids is not None:
                return left_ids + right_ids
        return None

    def visit_expr(
        expr: IndexExpr,
        parent_id: NodeId,
        edge: str,
        scope_id: NodeId,
        workspace_scope: Optional[NodeId],
        kind: AccessKind,
        path: Tuple[str, ...],
    ) -> None:
        if isinstance(expr, TensorAccess):
            visit_access(
                expr,
                parent_id,
                edge,
                scope_id,
                workspace_scope,
                kind,
                path,
            )
            return
        if isinstance(expr, BinaryOp):
            if not record_node(expr, parent_id, edge, scope_id, path):
                return
            visit_expr(
                expr.left,
                expr.node_id,
                "left",
                scope_id,
                workspace_scope,
                kind,
                path + ("left",),
            )
            visit_expr(
                expr.right,
                expr.node_id,
                "right",
                scope_id,
                workspace_scope,
                kind,
                path + ("right",),
            )
            return
        if isinstance(expr, UnaryOp):
            if not record_node(expr, parent_id, edge, scope_id, path):
                return
            visit_expr(
                expr.expr,
                expr.node_id,
                "expr",
                scope_id,
                workspace_scope,
                kind,
                path + ("expr",),
            )
            return
        diagnose(
            "unsupported_expression",
            f"unsupported CIN expression {type(expr).__name__}",
            path,
            getattr(expr, "node_id", None),
        )

    def visit_stmt(
        stmt: IndexStmt,
        parent_id: Optional[NodeId],
        edge: str,
        parent_scope: Optional[NodeId],
        workspace_scope: Optional[NodeId],
        active_indices: Tuple[IndexId, ...],
        path: Tuple[str, ...],
    ) -> None:
        scope_id = stmt.node_id
        scope_parents.setdefault(scope_id, parent_scope)
        if not record_node(stmt, parent_id, edge, scope_id, path):
            return

        if isinstance(stmt, ForAll):
            index_var = stmt.index_var
            register_index(index_var, path + ("index_var",))
            binding = IndexBinding(stmt.node_id, scope_id)
            if index_var.index_id in active_indices:
                diagnose(
                    "duplicate_index_binding",
                    f"IndexId {index_var.index_id.value} is rebound in an "
                    "overlapping scope",
                    path + ("index_var",),
                    index_var.index_id,
                )
            index_bindings.setdefault(index_var.index_id, []).append(binding)
            visit_stmt(
                stmt.stmt,
                stmt.node_id,
                "stmt",
                scope_id,
                workspace_scope,
                active_indices + (index_var.index_id,),
                path + ("stmt",),
            )
            return
        if isinstance(stmt, Where):
            visit_stmt(
                stmt.producer,
                stmt.node_id,
                "producer",
                scope_id,
                scope_id,
                active_indices,
                path + ("producer",),
            )
            visit_stmt(
                stmt.consumer,
                stmt.node_id,
                "consumer",
                scope_id,
                scope_id,
                active_indices,
                path + ("consumer",),
            )
            return
        if isinstance(stmt, TensorAssign):
            lhs_kind = (
                AccessKind.REDUCTION_WRITE if stmt.op is not None else AccessKind.WRITE
            )
            visit_expr(
                stmt.lhs,
                stmt.node_id,
                "lhs",
                scope_id,
                workspace_scope,
                lhs_kind,
                path + ("lhs",),
            )
            visit_expr(
                stmt.rhs,
                stmt.node_id,
                "rhs",
                scope_id,
                workspace_scope,
                AccessKind.READ,
                path + ("rhs",),
            )
            rhs_access_ids = expression_access_ids(stmt.rhs)
            lhs_index_ids = tuple(stmt.lhs.index_ids)
            reduction_index_ids: List[IndexId] = []
            for access_id in rhs_access_ids:
                access_info = accesses.get(access_id)
                if access_info is None:
                    continue
                for index_id in access_info.index_ids:
                    if (
                        index_id not in lhs_index_ids
                        and index_id not in reduction_index_ids
                    ):
                        reduction_index_ids.append(index_id)
            assignments[stmt.node_id] = AssignmentInfo(
                assignment_id=stmt.node_id,
                lhs_access_id=stmt.lhs.access_id,
                rhs_access_ids=rhs_access_ids,
                update_op=stmt.op,
                lhs_index_ids=lhs_index_ids,
                reduction_index_ids=tuple(reduction_index_ids),
                multiplicative_access_ids=multiplicative_access_ids(stmt.rhs),
            )
            assignment_order.append(stmt.node_id)
            return
        diagnose(
            "unsupported_statement",
            f"unsupported CIN statement {type(stmt).__name__}",
            path,
            stmt.node_id,
        )

    visit_stmt(cin, None, "root", None, None, (), ("root",))

    def scope_contains(owner: NodeId, scope: NodeId) -> bool:
        current: Optional[NodeId] = scope
        visited: set[NodeId] = set()
        while current is not None and current not in visited:
            if current == owner:
                return True
            visited.add(current)
            current = scope_parents.get(current)
        return False

    for symbol_id, symbol_use_records in symbol_uses.items():
        definition = symbol_definitions.get(symbol_id)
        for symbol_use in symbol_use_records:
            occurrence = next(
                (
                    item
                    for item in access_occurrences
                    if item.access_id == symbol_use.access_id
                ),
                None,
            )
            occurrence_path = occurrence.path if occurrence is not None else ("root",)
            if definition is None:
                diagnose(
                    "dangling_symbol_reference",
                    f"access references undefined SymbolId {symbol_id.value}",
                    occurrence_path,
                    symbol_id,
                )
            elif not scope_contains(definition.scope_id, symbol_use.scope_id):
                diagnose(
                    "symbol_reference_out_of_scope",
                    f"SymbolId {symbol_id.value} is not visible in this scope",
                    occurrence_path,
                    symbol_id,
                )

    for index_id, index_use_records in index_uses.items():
        bindings = index_bindings.get(index_id, [])
        for index_use in index_use_records:
            occurrence = next(
                (
                    item
                    for item in access_occurrences
                    if item.access_id == index_use.access_id
                ),
                None,
            )
            path = occurrence.path if occurrence is not None else ("root",)
            if not bindings:
                diagnose(
                    "dangling_index_reference",
                    f"access references unbound IndexId {index_id.value}",
                    path + (f"indices[{index_use.axis}]",),
                    index_id,
                )
            elif not any(
                scope_contains(binding.scope_id, index_use.scope_id)
                for binding in bindings
            ):
                diagnose(
                    "index_reference_out_of_scope",
                    f"IndexId {index_id.value} has no binding visible in this scope",
                    path + (f"indices[{index_use.axis}]",),
                    index_id,
                )

    for index_id in free_index_ids:
        if index_id not in index_bindings:
            diagnose(
                "free_index_not_bound",
                f"free IndexId {index_id.value} has no binding",
                ("root",),
                index_id,
            )

    occurrence_paths = {
        occurrence.access_id: occurrence.path for occurrence in access_occurrences
    }
    known_extents: Dict[IndexId, Tuple[int, SymbolId]] = {}
    for access_id, access_info in accesses.items():
        definition = symbol_definitions.get(access_info.tensor_id)
        if definition is None or definition.shape is None:
            continue
        for logical_axis, index_id in enumerate(access_info.index_ids):
            if definition.mode_order:
                if logical_axis not in definition.mode_order:
                    continue
                physical_axis = definition.mode_order.index(logical_axis)
            else:
                physical_axis = logical_axis
            if physical_axis >= len(definition.shape):
                continue
            extent = definition.shape[physical_axis]
            previous = known_extents.get(index_id)
            if previous is not None and previous[0] != extent:
                diagnose(
                    "index_extent_mismatch",
                    f"IndexId {index_id.value} has incompatible extents "
                    f"{previous[0]} and {extent}",
                    occurrence_paths.get(access_id, ("root",))
                    + (f"indices[{logical_axis}]",),
                    index_id,
                )
            else:
                known_extents[index_id] = (extent, access_info.tensor_id)

    reduction_evidence: set[IndexId] = set()
    for access_info in accesses.values():
        if access_info.kind != AccessKind.READ:
            continue
        assignment = node_objects.get(access_info.scope_id)
        if not isinstance(assignment, TensorAssign):
            continue
        lhs_ids = tuple(assignment.lhs.index_ids)
        reduction_evidence.update(
            index_id for index_id in access_info.index_ids if index_id not in lhs_ids
        )

    used_index_ids = {
        index_id for index_id, use_records in index_uses.items() if use_records
    }
    for index_id, bindings in index_bindings.items():
        if bindings and index_id not in used_index_ids:
            diagnose(
                "unused_index_binding",
                f"bound IndexId {index_id.value} has no tensor-access use",
                ("root",),
                index_id,
            )
    inconsistent_reductions = [
        index_id
        for index_id in index_order
        if index_id in index_bindings
        and index_id in used_index_ids
        and index_id not in free_index_ids
        and index_id not in reduction_evidence
    ]
    for index_id in inconsistent_reductions:
        diagnose(
            "index_classification_inconsistent",
            f"non-free IndexId {index_id.value} has no reduction assignment",
            ("root",),
            index_id,
        )
    for index_id in free_index_ids:
        if index_id in reduction_evidence:
            diagnose(
                "free_reduction_conflict",
                f"IndexId {index_id.value} is both free and reduced",
                ("root",),
                index_id,
            )

    index_definitions = {
        index_id: IndexDefinition(
            index_id,
            index_node_ids[index_id],
            index_names[index_id],
            tuple(index_bindings.get(index_id, ())),
        )
        for index_id in index_order
    }
    reduction_index_ids = tuple(
        index_id
        for index_id in index_order
        if index_id in index_bindings
        and index_id in reduction_evidence
        and index_id not in free_index_ids
    )
    symbol_use_order = symbol_order + [
        symbol_id for symbol_id in symbol_uses if symbol_id not in symbol_order
    ]
    index_use_order = index_order + [
        index_id for index_id in index_uses if index_id not in index_order
    ]
    tensor_access_order = symbol_order + [
        symbol_id for symbol_id in tensor_accesses if symbol_id not in symbol_order
    ]

    return CINAnalysis(
        root_id=root_id,
        parents=FrozenMap.from_items(parent_relations.items()),
        node_scopes=FrozenMap.from_items(node_scopes.items()),
        scope_parents=FrozenMap.from_items(scope_parents.items()),
        symbol_definitions=FrozenMap.from_items(
            (symbol_id, symbol_definitions[symbol_id]) for symbol_id in symbol_order
        ),
        symbol_uses=FrozenMap.from_items(
            (symbol_id, tuple(symbol_uses.get(symbol_id, ())))
            for symbol_id in symbol_use_order
        ),
        index_definitions=FrozenMap.from_items(
            (index_id, index_definitions[index_id]) for index_id in index_order
        ),
        index_uses=FrozenMap.from_items(
            (index_id, tuple(index_uses.get(index_id, ())))
            for index_id in index_use_order
        ),
        accesses=FrozenMap.from_items(accesses.items()),
        access_layouts=FrozenMap.from_items(access_layouts.items()),
        assignments=FrozenMap.from_items(
            (assignment_id, assignments[assignment_id])
            for assignment_id in assignment_order
        ),
        access_occurrences=tuple(access_occurrences),
        tensor_accesses=FrozenMap.from_items(
            (symbol_id, tuple(tensor_accesses.get(symbol_id, ())))
            for symbol_id in tensor_access_order
        ),
        access_order=tuple(access_order),
        assignment_order=tuple(assignment_order),
        free_index_ids=tuple(free_index_ids),
        reduction_index_ids=reduction_index_ids,
        diagnostics=tuple(diagnostics),
    )


def analyze_cin(cin: IndexStmt) -> CINAnalysis:
    """Compatibility entry routed through the canonical pure analysis runner."""

    from .analysis_runner import COMMON_ANALYSIS_RUNNER

    return COMMON_ANALYSIS_RUNNER.analyze_cin(cin)


def verify_cin(cin: IndexStmt) -> CINAnalysis:
    """Run the full normalized-CIN verifier and return its immutable analysis."""

    analysis = analyze_cin(cin)
    if analysis.diagnostics:
        _raise_cin_verification(analysis.diagnostics)
    return analysis


_VERIFY_CIN_CONTEXT: ContextVar[bool] = ContextVar(
    "scorch_verify_normalized_cin",
    default=False,
)
_TRUSTED_CIN_ROOTS: ContextVar[frozenset[int]] = ContextVar(
    "scorch_trusted_normalized_cin_roots",
    default=frozenset(),
)


@contextmanager
def _trusted_normalized_cin(cin: IndexStmt) -> Iterator[None]:
    """Avoid re-preflighting one already-verified root within a compiler call.

    The caller must either have obtained ``cin`` from :func:`normalize_cin` or
    have called :func:`verify_cin_structure` immediately before entering this
    synchronous context.  No caller-controlled work may occur between those
    operations.  The trust is held out-of-band rather than on the mutable
    compatibility object, so forged caller metadata cannot opt into it.
    """

    if not _is_exact_index_stmt(cin):
        raise TypeError("trusted normalized CIN must be an exact IndexStmt")
    roots = _TRUSTED_CIN_ROOTS.get()
    token = _TRUSTED_CIN_ROOTS.set(roots | {id(cin)})
    try:
        yield
    finally:
        _TRUSTED_CIN_ROOTS.reset(token)


def get_full_cin_verification() -> bool:
    """Return the debug verification override at the public boundary."""

    enabled = _VERIFY_CIN_CONTEXT.get()
    if type(enabled) is not bool:
        raise TypeError("full CIN verification override must be a bool")
    return enabled


@contextmanager
def full_cin_verification(enabled: bool = True) -> Iterator[None]:
    """Enable full CIN verification at compiler boundaries for tests/debug."""

    if type(enabled) is not bool:
        raise TypeError("full_cin_verification expects a bool")
    token = _VERIFY_CIN_CONTEXT.set(enabled)
    try:
        yield
    finally:
        _VERIFY_CIN_CONTEXT.reset(token)


def _compile_options_at_cin_boundary(
    compile_options: Optional[CompileOptions] = None,
) -> CompileOptions:
    """Snapshot process/debug policy once at a direct CIN boundary.

    An explicit snapshot is already owned by an outer compilation and therefore
    must not be combined with mutable context state again.
    """

    if compile_options is not None:
        if type(compile_options) is not CompileOptions:
            raise TypeError("compile_options must be a CompileOptions snapshot")
        return compile_options

    return CompileOptions.from_environment()


def verify_cin_if_enabled(
    cin: IndexStmt, enabled: Optional[bool] = None
) -> Optional[CINAnalysis]:
    """Verify only in explicit test/debug mode; production JIT stays cheap."""

    if enabled is None:
        enabled = _compile_options_at_cin_boundary().verification.verify_cin
    if type(enabled) is not bool:
        raise TypeError("CIN verification policy must be a bool")
    if enabled:
        return verify_cin(cin)
    return None


def _initialize_cin_clone(clone: object, node_id: NodeId) -> None:
    object.__setattr__(clone, "node_id", node_id)
    object.__setattr__(clone, "inserted_workspace", False)
    object.__setattr__(clone, "no_tile_list", [])


def normalize_cin(
    cin: IndexStmt,
    compile_options: Optional[CompileOptions] = None,
    compilation_context: Optional[CompilationContext] = None,
) -> IndexStmt:
    """Detach authoritative CIN structure while preserving all stable IDs.

    The returned graph is still a mutable legacy CIN graph; only its ownership is
    normalized.  Parent pointers, reverse access lists, result ``_assignment``,
    and schedule metadata are deliberately absent.
    """

    if not _is_index_stmt_instance(cin):
        raise TypeError("normalize_cin expects an IndexStmt")
    options = _compile_options_at_cin_boundary(compile_options)
    if compilation_context is None:
        return _normalize_cin_owned(cin, options)
    stage_token = compilation_context.begin_stage(
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION,
        compile_options=options,
    )
    try:
        normalized = _normalize_cin_owned(cin, options)
    except Exception:
        compilation_context.fail_stage(stage_token)
        raise
    compilation_context.complete_stage(stage_token)
    return normalized


def _normalize_cin_owned(cin: IndexStmt, options: CompileOptions) -> IndexStmt:
    """Normalize after the public boundary resolved one exact options snapshot."""

    # Normalization owns the structural boundary for every caller, including
    # direct release-mode ``normalize_cin``: the clone walk below recurses over
    # stored forward edges, so forged fields, cycles, and unbounded depth must
    # fail closed with stable diagnostics before any recursive work.  The
    # bounded iterative preflight also caps the depth the recursive clone can
    # observe.
    verify_cin_structure(cin)
    verify_cin_if_enabled(cin, options.verification.verify_cin)

    index_memo: Dict[int, IndexVar] = {}
    tensor_memo: Dict[int, TensorVar] = {}
    node_memo: Dict[int, object] = {}

    def clone_index(index_var: IndexVar) -> IndexVar:
        key = id(index_var)
        cached = index_memo.get(key)
        if cached is not None:
            return cached
        clone = object.__new__(IndexVar)
        index_memo[key] = clone
        _initialize_cin_clone(clone, index_var.node_id)
        clone.index_id = index_var.index_id
        clone._name = index_var.name
        clone._expr = None
        clone._parent = None
        clone.is_tiled = False
        clone.is_outer = False
        clone.is_inner = False
        clone.tile_size_var = None
        clone.tensor_accesses = []
        if isinstance(index_var._expr, IndexVarAdd):
            clone._expr = clone_index_expr(index_var._expr)
        elif index_var._expr is not None:
            raise TypeError(
                f"unsupported CIN index expression {type(index_var._expr).__name__}"
            )
        return clone

    def clone_index_expr(expr: IndexVarAdd) -> IndexVarAdd:
        clone = object.__new__(IndexVarAdd)
        _initialize_cin_clone(clone, expr.node_id)
        clone.lhs = clone_index(expr.lhs)
        clone.rhs = clone_index(expr.rhs)
        return clone

    def clone_tensor(tensor: TensorVar) -> TensorVar:
        key = id(tensor)
        cached = tensor_memo.get(key)
        if cached is not None:
            return cached
        clone = object.__new__(type(tensor))
        tensor_memo[key] = clone
        _initialize_cin_clone(clone, tensor.node_id)
        clone.symbol_id = tensor.symbol_id
        clone._name = tensor.name
        clone._format = tensor._format
        clone._assignment = None
        clone.shape = tuple(tensor.shape) if tensor.shape is not None else None
        clone.dtype = tensor.dtype
        clone.mode_order = (
            list(tensor.mode_order) if tensor.mode_order is not None else None
        )
        if isinstance(tensor, Workspace):
            workspace = cast(Workspace, clone)
            workspace.dim = tensor.dim
            workspace.dense = tensor.dense
            workspace._tile_size_var = None
            workspace.workspace_accesses = []
        return clone

    def clone_expr(expr: IndexExpr) -> IndexExpr:
        key = id(expr)
        cached = node_memo.get(key)
        if cached is not None:
            return cast(IndexExpr, cached)
        if isinstance(expr, WorkspaceAccess):
            workspace_clone = object.__new__(WorkspaceAccess)
            node_memo[key] = workspace_clone
            _initialize_cin_clone(workspace_clone, expr.node_id)
            workspace_clone.access_id = expr.access_id
            workspace_clone.wksp = cast(Workspace, clone_tensor(expr.wksp))
            workspace_clone.tensor = workspace_clone.wksp
            workspace_clone.tensor_id = expr.tensor_id
            workspace_clone.indices = [clone_index(index) for index in expr.indices]
            workspace_clone.index_ids = tuple(expr.index_ids)
            return workspace_clone
        if isinstance(expr, TensorAccess):
            access_clone = object.__new__(TensorAccess)
            node_memo[key] = access_clone
            _initialize_cin_clone(access_clone, expr.node_id)
            access_clone.access_id = expr.access_id
            access_clone.tensor = clone_tensor(expr.tensor)
            access_clone.tensor_id = expr.tensor_id
            access_clone.indices = [clone_index(index) for index in expr.indices]
            access_clone.index_ids = tuple(expr.index_ids)
            return access_clone
        if isinstance(expr, BinaryOp):
            binary_clone = object.__new__(BinaryOp)
            node_memo[key] = binary_clone
            _initialize_cin_clone(binary_clone, expr.node_id)
            object.__setattr__(binary_clone, "op", expr.op)
            object.__setattr__(binary_clone, "left", clone_expr(expr.left))
            object.__setattr__(binary_clone, "right", clone_expr(expr.right))
            return binary_clone
        if isinstance(expr, UnaryOp):
            unary_clone = object.__new__(UnaryOp)
            node_memo[key] = unary_clone
            _initialize_cin_clone(unary_clone, expr.node_id)
            object.__setattr__(unary_clone, "op", expr.op)
            object.__setattr__(unary_clone, "expr", clone_expr(expr.expr))
            return unary_clone
        raise TypeError(f"unsupported CIN expression {type(expr).__name__}")

    def clone_stmt(stmt: IndexStmt) -> IndexStmt:
        key = id(stmt)
        cached = node_memo.get(key)
        if cached is not None:
            return cast(IndexStmt, cached)
        if isinstance(stmt, ForAll):
            forall_clone = object.__new__(ForAll)
            node_memo[key] = forall_clone
            _initialize_cin_clone(forall_clone, stmt.node_id)
            forall_clone.lhs = None
            forall_clone.rhs = None
            forall_clone.parent = None
            forall_clone.index_var = clone_index(stmt.index_var)
            forall_clone.stmt = clone_stmt(stmt.stmt)
            forall_clone.parallel = stmt.parallel
            return forall_clone
        if isinstance(stmt, Where):
            where_clone = object.__new__(Where)
            node_memo[key] = where_clone
            _initialize_cin_clone(where_clone, stmt.node_id)
            where_clone.lhs = None
            where_clone.rhs = None
            where_clone.parent = None
            where_clone.producer = clone_stmt(stmt.producer)
            where_clone.consumer = clone_stmt(stmt.consumer)
            return where_clone
        if isinstance(stmt, TensorAssign):
            assign_clone = object.__new__(TensorAssign)
            node_memo[key] = assign_clone
            _initialize_cin_clone(assign_clone, stmt.node_id)
            assign_clone.parent = None
            assign_clone.lhs = cast(TensorAccess, clone_expr(stmt.lhs))
            assign_clone.rhs = clone_expr(stmt.rhs)
            assign_clone.op = stmt.op
            return assign_clone
        raise TypeError(f"unsupported CIN statement {type(stmt).__name__}")

    return clone_stmt(cin)


def canonical_cin_dump(cin: IndexStmt) -> str:
    """Serialize CIN with traversal-canonical IDs and no allocation-order data."""

    if not _is_index_stmt_instance(cin):
        raise TypeError("canonical_cin_dump expects an IndexStmt")
    try:
        verify_cin_structure(cin)
    except VerificationError as strict_error:
        diagnostics = cast(
            Tuple[CINDiagnostic, ...],
            strict_error.diagnostics,
        )
        if not diagnostics or any(
            diagnostic.code not in {"duplicate_node_id", "duplicate_index_id"}
            for diagnostic in diagnostics
        ):
            raise
        # Legacy automatic workspace insertion clones producer/consumer
        # syntax with one exact, validated stable-ID alias receipt.  The dump
        # has historically been usable to compare those detached schedule
        # artifacts.  Admit only the same narrow shape as raw legacy lowering;
        # any unrelated duplicate or malformed compatibility field still
        # fails closed.
        _verify_legacy_cin_lowering_structure(cin)

    node_ids: Dict[NodeId, int] = {}
    access_ids: Dict[AccessId, int] = {}
    index_ids: Dict[IndexId, int] = {}
    symbol_ids: Dict[SymbolId, int] = {}
    emitted_symbols: set[SymbolId] = set()

    def canonical_id(mapping: Dict[object, int], value: object) -> int:
        if value not in mapping:
            mapping[value] = len(mapping)
        return mapping[value]

    def node_id(value: NodeId) -> int:
        return canonical_id(cast(Dict[object, int], node_ids), value)

    def access_id(value: AccessId) -> int:
        return canonical_id(cast(Dict[object, int], access_ids), value)

    def index_id(value: IndexId) -> int:
        return canonical_id(cast(Dict[object, int], index_ids), value)

    def symbol_id(value: SymbolId) -> int:
        return canonical_id(cast(Dict[object, int], symbol_ids), value)

    def serialize_tensor(tensor: TensorVar) -> Dict[str, object]:
        sid = symbol_id(tensor.symbol_id)
        result: Dict[str, object] = {"id": sid}
        if tensor.symbol_id in emitted_symbols:
            return result
        emitted_symbols.add(tensor.symbol_id)
        (
            rank,
            shape,
            mode_order,
            levels,
            level_bit_widths,
            dtype,
        ) = _tensor_metadata(tensor)
        result.update(
            {
                "kind": "workspace" if isinstance(tensor, Workspace) else "tensor",
                "rank": rank,
                "shape": shape,
                "mode_order": mode_order,
                "levels": tuple(zip(levels, level_bit_widths)),
                "dtype": dtype,
            }
        )
        return result

    def serialize_expr(expr: IndexExpr) -> Dict[str, object]:
        if isinstance(expr, TensorAccess):
            return {
                "node": node_id(expr.node_id),
                "kind": (
                    "workspace_access"
                    if isinstance(expr, WorkspaceAccess)
                    else "tensor_access"
                ),
                "access": access_id(expr.access_id),
                "tensor_ref": symbol_id(expr.tensor_id),
                "tensor": serialize_tensor(expr.tensor),
                "indices": tuple(index_id(value) for value in expr.index_ids),
            }
        if isinstance(expr, BinaryOp):
            return {
                "node": node_id(expr.node_id),
                "kind": "binary",
                "op": expr.op.value,
                "left": serialize_expr(expr.left),
                "right": serialize_expr(expr.right),
            }
        if isinstance(expr, UnaryOp):
            return {
                "node": node_id(expr.node_id),
                "kind": "unary",
                "op": expr.op.value,
                "expr": serialize_expr(expr.expr),
            }
        raise TypeError(f"unsupported CIN expression {type(expr).__name__}")

    def serialize_index(index_var: IndexVar) -> Dict[str, object]:
        result: Dict[str, object] = {"id": index_id(index_var.index_id)}
        if isinstance(index_var._expr, IndexVarAdd):
            result["expr"] = {
                "kind": "add",
                "lhs": index_id(index_var._expr.lhs.index_id),
                "rhs": index_id(index_var._expr.rhs.index_id),
            }
        return result

    def serialize_stmt(stmt: IndexStmt) -> Dict[str, object]:
        if isinstance(stmt, ForAll):
            return {
                "node": node_id(stmt.node_id),
                "kind": "forall",
                "index": serialize_index(stmt.index_var),
                "parallel": stmt.parallel,
                "body": serialize_stmt(stmt.stmt),
            }
        if isinstance(stmt, Where):
            return {
                "node": node_id(stmt.node_id),
                "kind": "where",
                "producer": serialize_stmt(stmt.producer),
                "consumer": serialize_stmt(stmt.consumer),
            }
        if isinstance(stmt, TensorAssign):
            return {
                "node": node_id(stmt.node_id),
                "kind": "assign",
                "op": stmt.op.value if isinstance(stmt.op, Operation) else None,
                "lhs": serialize_expr(stmt.lhs),
                "rhs": serialize_expr(stmt.rhs),
            }
        raise TypeError(f"unsupported CIN statement {type(stmt).__name__}")

    payload = {
        "schema": "scorch.cin.canonical.v1",
        "root": serialize_stmt(cin),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
