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

from ..format import LevelType
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
    UnaryOp,
    Where,
    Workspace,
    WorkspaceAccess,
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
        node_id = getattr(node, "node_id", None)
        if not isinstance(node_id, NodeId):
            self.diagnose(
                "invalid_node_id",
                f"{type(node).__name__}.node_id must be a NodeId",
                path,
            )
            return
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
        if not isinstance(getattr(index_var, "index_id", None), IndexId):
            self.diagnose(
                "invalid_index_id",
                "IndexVar.index_id must be an IndexId",
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
        if not isinstance(getattr(tensor, "symbol_id", None), SymbolId):
            self.diagnose(
                "invalid_symbol_id",
                "TensorVar.symbol_id must be a SymbolId",
                path,
            )

    def visit_access(self, access: TensorAccess, path: Tuple[str, ...]) -> None:
        self.record_node(access, path)
        if not isinstance(getattr(access, "access_id", None), AccessId):
            self.diagnose(
                "invalid_access_id",
                "TensorAccess.access_id must be an AccessId",
                path,
            )
        if not isinstance(getattr(access, "tensor_id", None), SymbolId):
            self.diagnose(
                "invalid_symbol_reference",
                "TensorAccess.tensor_id must be a SymbolId",
                path + ("tensor_ref",),
            )
        self.visit_tensor(getattr(access, "tensor", None), path + ("tensor",))

        indices = getattr(access, "indices", None)
        if not isinstance(indices, (list, tuple)):
            self.diagnose(
                "invalid_index_reference",
                "TensorAccess.indices must be a sequence of IndexVar references",
                path + ("indices",),
            )
            indices = ()
        for axis, index_var in enumerate(indices):
            self.visit_index(index_var, path + (f"indices[{axis}]",))

        index_ids = getattr(access, "index_ids", None)
        if not isinstance(index_ids, tuple):
            self.diagnose(
                "invalid_index_reference",
                "TensorAccess.index_ids must be an immutable tuple of IndexId",
                path + ("index_refs",),
            )
            return
        for axis, index_id in enumerate(index_ids):
            if not isinstance(index_id, IndexId):
                self.diagnose(
                    "invalid_index_reference",
                    "TensorAccess.index_ids entries must be IndexId values",
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

    if not isinstance(cin, IndexStmt):
        raise TypeError("analyze_cin expects an IndexStmt")

    preflight = _IdPreflight()
    preflight.visit_stmt(cin, ("root",))
    raw_root_id = getattr(cin, "node_id", None)
    root_id = raw_root_id if isinstance(raw_root_id, NodeId) else NodeId(-1)
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
        first = analysis.diagnostics[0]
        raise VerificationError(
            f"stage=normalized CIN: {first.code} at "
            f"{'/'.join(first.path)}: {first.message}",
            diagnostics=cast(Tuple[object, ...], analysis.diagnostics),
        )
    return analysis


_VERIFY_CIN_CONTEXT: ContextVar[bool] = ContextVar(
    "scorch_verify_normalized_cin",
    default=False,
)


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

    if not isinstance(cin, IndexStmt):
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

    if not isinstance(cin, IndexStmt):
        raise TypeError("canonical_cin_dump expects an IndexStmt")

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
