"""Single ownership adapter for the mutable legacy CIN implementation.

Semantic CIN and its analyses do not rely on node-owned parent/access lists.  The
legacy scheduler and lowerer still read a few of those lists, so this module is
the only place that recreates them.  Recreation always happens after a deep copy;
the returned tree is a private working artifact and must not cross back into the
normalized-CIN boundary.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Dict, List, Optional, Set, cast

from .cin import (
    BinaryOp,
    ForAll,
    IndexStmt,
    IndexVar,
    IndexVarAdd,
    TensorAccess,
    TensorAssign,
    TensorVar,
    UnaryOp,
    Where,
    Workspace,
    WorkspaceAccess,
)
from .diagnostics import VerificationError
from .identity import AccessId, IndexId, SymbolId

if TYPE_CHECKING:
    from .compile_options import CompileOptions
    from .loop_plan import LoopPlan


def legacy_cin_working_copy(
    cin: IndexStmt,
    plan: Optional[LoopPlan] = None,
    compile_options: Optional[CompileOptions] = None,
) -> IndexStmt:
    """Return a private scheduled tree with compatibility backlinks restored."""

    return _prepare_legacy_cin(
        cin,
        plan,
        copy_input=True,
        compile_options=compile_options,
    )


def claim_legacy_cin_working_tree(
    cin: IndexStmt,
    plan: Optional[LoopPlan] = None,
    compile_options: Optional[CompileOptions] = None,
) -> IndexStmt:
    """Claim an already detached tree and restore compatibility backlinks.

    This is the ownership-transfer form of :func:`legacy_cin_working_copy`.  It
    exists solely for compiler frontends that have just received a detached tree
    from ``Scheduler`` or ``normalize_cin`` and retain no externally visible
    alias.  Public lowering continues to use the copying form above.
    """

    return _prepare_legacy_cin(
        cin,
        plan,
        copy_input=False,
        compile_options=compile_options,
    )


def _prepare_legacy_cin(
    cin: IndexStmt,
    plan: Optional[LoopPlan],
    *,
    copy_input: bool,
    compile_options: Optional[CompileOptions],
) -> IndexStmt:
    validate_legacy_cin_display_names(cin)
    working = copy.deepcopy(cin) if copy_input else cin
    if plan is not None:
        # Import lazily to keep CIN ownership independent of scheduling policy.
        from .compile_options import CompileOptions
        from .scheduler import Scheduler, materialize_legacy_schedule

        if compile_options is None:
            compile_options = CompileOptions.from_environment()
        elif type(compile_options) is not CompileOptions:
            raise TypeError("compile_options must be a CompileOptions instance")

        if plan.provenance == "auto":
            working = Scheduler._replay_auto_plan_owned(
                working,
                plan,
                compile_options=compile_options,
            )
        else:
            schedule, _, _, _ = materialize_legacy_schedule(cin, plan)
            replayed = Scheduler._apply_schedule_legacy(
                working,
                schedule,
                compile_options=compile_options,
            )
            working = replayed.normalized_cin
        # Schedule replay can introduce derived display names, so validate the
        # resulting private tree as well as its semantic source.
        validate_legacy_cin_display_names(working)
    _canonicalize_legacy_entities(working)
    _materialize_legacy_backreferences(working)
    return working


def validate_legacy_cin_display_names(cin: IndexStmt) -> None:
    """Fail before the name-rendered legacy implementation can conflate IDs."""

    index_names: Dict[str, IndexId] = {}
    for index_var in cin.index_vars:
        previous_index = index_names.get(index_var.name)
        if previous_index is not None and previous_index != index_var.index_id:
            raise VerificationError(
                "stage=legacy CIN adapter: display name "
                f"{index_var.name!r} refers to distinct IndexId values"
            )
        index_names[index_var.name] = index_var.index_id

    symbol_names: Dict[str, SymbolId] = {}
    all_accesses = list(cin.tensor_accesses) + list(cin.get_workspace_accesses())
    for access in all_accesses:
        name = access.tensor.name
        previous_symbol = symbol_names.get(name)
        if previous_symbol is not None and previous_symbol != access.tensor.symbol_id:
            raise VerificationError(
                "stage=legacy CIN adapter: display name "
                f"{name!r} refers to distinct SymbolId values"
            )
        symbol_names[name] = access.tensor.symbol_id


def _canonicalize_legacy_entities(cin: IndexStmt) -> None:
    """Unify copied legacy entity objects by their stable semantic IDs."""

    index_vars: Dict[IndexId, IndexVar] = {}
    tensor_vars: Dict[SymbolId, TensorVar] = {}
    visited: Set[int] = set()

    def collect(node: object) -> None:
        object_id = id(node)
        if object_id in visited:
            return
        visited.add(object_id)
        if isinstance(node, ForAll):
            index_vars.setdefault(node.index_var.index_id, node.index_var)
            collect(node.stmt)
        elif isinstance(node, Where):
            collect(node.producer)
            collect(node.consumer)
        elif isinstance(node, TensorAssign):
            collect(node.lhs)
            collect(node.rhs)
        elif isinstance(node, TensorAccess):
            tensor_vars.setdefault(node.tensor.symbol_id, node.tensor)
            for index_var in node.indices:
                index_vars.setdefault(index_var.index_id, index_var)
        elif isinstance(node, BinaryOp):
            collect(node.left)
            collect(node.right)
        elif isinstance(node, UnaryOp):
            collect(node.expr)

    collect(cin)
    visited.clear()

    def rewrite(node: object) -> None:
        object_id = id(node)
        if object_id in visited:
            return
        visited.add(object_id)
        if isinstance(node, ForAll):
            node.index_var = index_vars[node.index_var.index_id]
            rewrite(node.stmt)
        elif isinstance(node, Where):
            rewrite(node.producer)
            rewrite(node.consumer)
        elif isinstance(node, TensorAssign):
            rewrite(node.lhs)
            rewrite(node.rhs)
        elif isinstance(node, TensorAccess):
            node.tensor = tensor_vars[node.tensor.symbol_id]
            node.tensor_id = node.tensor.symbol_id
            if isinstance(node, WorkspaceAccess):
                node.wksp = cast(Workspace, node.tensor)
            node.indices = [index_vars[index.index_id] for index in node.indices]
            node.index_ids = tuple(index.index_id for index in node.indices)
        elif isinstance(node, BinaryOp):
            rewrite(node.left)
            rewrite(node.right)
        elif isinstance(node, UnaryOp):
            rewrite(node.expr)

    rewrite(cin)

    for index_var in index_vars.values():
        if isinstance(index_var._expr, IndexVarAdd):
            index_var._expr.lhs = index_vars[index_var._expr.lhs.index_id]
            index_var._expr.rhs = index_vars[index_var._expr.rhs.index_id]
            index_var._expr.lhs._parent = index_var
            index_var._expr.rhs._parent = index_var
        tile_size_var = index_var.tile_size_var
        if tile_size_var is not None:
            tile_size_var.outer_index_var = index_vars[
                tile_size_var.outer_index_var.index_id
            ]
            tile_size_var.inner_index_var = index_vars[
                tile_size_var.inner_index_var.index_id
            ]
            if tile_size_var._index_var is not None:
                tile_size_var._index_var = index_vars[tile_size_var._index_var.index_id]


def _materialize_legacy_backreferences(cin: IndexStmt) -> None:
    """Rebuild legacy-only backlinks on an already private CIN tree.

    Access lists retain historical access-construction order by sorting stable
    ``AccessId`` values.  Some legacy bound selection depends on that order, so a
    structural traversal order would change generated C++ for supported kernels.
    """

    accesses: Dict[AccessId, TensorAccess] = {}
    index_vars: Dict[IndexId, IndexVar] = {}
    tensor_vars: Dict[SymbolId, TensorVar] = {}
    visited_objects: Set[int] = set()

    def remember_index(index_var: IndexVar) -> None:
        index_vars.setdefault(index_var.index_id, index_var)

    def remember_tensor(tensor_var: TensorVar) -> None:
        tensor_vars.setdefault(tensor_var.symbol_id, tensor_var)

    def visit(node: object, parent: IndexStmt | None = None) -> None:
        object_id = id(node)
        if object_id in visited_objects:
            return
        visited_objects.add(object_id)

        if isinstance(node, IndexStmt):
            node.parent = parent

        if isinstance(node, ForAll):
            remember_index(node.index_var)
            visit(node.stmt, node)
        elif isinstance(node, Where):
            visit(node.producer, node)
            visit(node.consumer, node)
        elif isinstance(node, TensorAssign):
            visit(node.lhs, node)
            visit(node.rhs, node)
        elif isinstance(node, TensorAccess):
            accesses.setdefault(node.access_id, node)
            remember_tensor(node.tensor)
            for index_var in node.indices:
                remember_index(index_var)
        elif isinstance(node, BinaryOp):
            visit(node.left, parent)
            visit(node.right, parent)
        elif isinstance(node, UnaryOp):
            visit(node.expr, parent)

    visit(cin)

    for index_var in index_vars.values():
        index_var.tensor_accesses = []
    for tensor_var in tensor_vars.values():
        if isinstance(tensor_var, Workspace):
            tensor_var.workspace_accesses = []

    ordered_accesses: List[TensorAccess] = sorted(
        accesses.values(), key=lambda access: access.access_id.value
    )
    for access in ordered_accesses:
        for index_var in access.indices:
            index_var.add_tensor_access(access)
        if isinstance(access, WorkspaceAccess):
            access.wksp.add_workspace_access(access)
            for index_var in access.indices:
                if index_var.is_inner and index_var.tile_size_var is not None:
                    access.wksp.tile_size_var = index_var.tile_size_var
