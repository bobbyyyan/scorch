"""Single ownership adapter for the mutable legacy CIN implementation.

Semantic CIN and its analyses do not rely on node-owned parent/access lists.  The
legacy scheduler and lowerer still read a few of those lists, so this module is
the only place that recreates them. Recreation always happens after an explicit
forward-field copy; the returned tree is a private working artifact and must not
cross back into the normalized-CIN boundary.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Optional, Set, cast

import torch

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
)
from .diagnostics import VerificationError
from .identity import AccessId, IndexId, SymbolId
from ..format import TensorFormat

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
    working = _copy_legacy_cin_forward(cin) if copy_input else cin
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
            # The verified plan canonicalizes omitted public spellings such
            # as ``Schedule.loop_order=None`` into explicit identity facts.
            # Private replay must validate against that canonical schedule,
            # not reject it merely because the original options retained the
            # shorter equivalent spelling.
            replay_options = compile_options
            requested = compile_options.requested_schedule
            if (
                requested is not None
                and requested != schedule
                and requested.loop_order is None
                and replace(requested, loop_order=schedule.loop_order) == schedule
            ):
                # ``loop_order=None`` asks the scheduler to choose an order.
                # Once that choice is frozen in LoopPlan, materialization
                # necessarily spells it explicitly.  Canonicalize only that
                # one proven-equivalent difference; every other disagreement
                # remains visible to _apply_schedule_legacy's conflict check.
                replay_options = replace(
                    compile_options,
                    requested_schedule=schedule,
                )
            replayed = Scheduler._apply_schedule_legacy(
                working,
                schedule,
                compile_options=replay_options,
            )
            working = replayed.normalized_cin
        # Schedule replay can introduce derived display names, so validate the
        # resulting private tree as well as its semantic source.
        validate_legacy_cin_display_names(working)
    _canonicalize_legacy_entities(working)
    _materialize_legacy_backreferences(working)
    return working


def _copy_legacy_cin_forward(cin: IndexStmt) -> IndexStmt:
    """Copy only validated forward and schedule-authoritative CIN state.

    ``copy.deepcopy`` follows every compatibility backlink and every arbitrary
    instance attribute. A caller can therefore hide an unbounded object graph
    in state that the lowerer immediately discards, turning the public lowering
    boundary into a raw ``RecursionError``. The structural preflight validates
    the fields consumed below; this copier deliberately drops statement
    parents, result-assignment links, and reverse access lists because
    :func:`_materialize_legacy_backreferences` rebuilds them from the copied
    forward graph.
    """

    memo: Dict[int, object] = {}

    def state(value: object) -> Dict[str, object]:
        return cast(Dict[str, object], object.__getattribute__(value, "__dict__"))

    def initialize(source: object, clone: object) -> None:
        source_state = state(source)
        object.__setattr__(clone, "node_id", source_state["node_id"])
        object.__setattr__(
            clone,
            "inserted_workspace",
            source_state.get("inserted_workspace", False),
        )
        object.__setattr__(clone, "no_tile_list", [])

    def finish_no_tile_list(source: object, clone: object) -> None:
        source_state = state(source)
        object.__setattr__(
            clone,
            "no_tile_list",
            [
                clone_index(cast(IndexVar, index_var))
                for index_var in cast(
                    List[object], source_state.get("no_tile_list", [])
                )
            ],
        )

    def clone_tile_size_var(tile: TileSizeVar) -> TileSizeVar:
        key = id(tile)
        cached = memo.get(key)
        if cached is not None:
            return cast(TileSizeVar, cached)
        clone = object.__new__(TileSizeVar)
        memo[key] = clone
        initialize(tile, clone)
        tile_state = state(tile)
        clone.outer_index_var = clone_index(
            cast(IndexVar, tile_state["outer_index_var"])
        )
        clone.inner_index_var = clone_index(
            cast(IndexVar, tile_state["inner_index_var"])
        )
        clone.size = cast(int, tile_state["size"])
        clone._name = cast(str, tile_state["_name"])
        base = tile_state["_index_var"]
        clone._index_var = (
            clone_index(cast(IndexVar, base)) if base is not None else None
        )
        clone.unroll = cast(bool, tile_state["unroll"])
        finish_no_tile_list(tile, clone)
        return clone

    def clone_index(index_var: IndexVar) -> IndexVar:
        key = id(index_var)
        cached = memo.get(key)
        if cached is not None:
            return cast(IndexVar, cached)
        clone = object.__new__(IndexVar)
        memo[key] = clone
        initialize(index_var, clone)
        index_state = state(index_var)
        clone.index_id = cast(IndexId, index_state["index_id"])
        clone._name = cast(str, index_state["_name"])
        clone._expr = None
        clone._parent = None
        clone.is_tiled = cast(bool, index_state["is_tiled"])
        clone.is_outer = cast(bool, index_state["is_outer"])
        clone.is_inner = cast(bool, index_state["is_inner"])
        clone.tile_size_var = None
        clone._legacy_tensor_accesses = []
        expression = index_state["_expr"]
        if expression is not None:
            clone._expr = clone_index_expr(cast(IndexVarAdd, expression))
        parent = index_state["_parent"]
        if parent is not None:
            clone._parent = clone_index(cast(IndexVar, parent))
        tile_size_var = index_state["tile_size_var"]
        if tile_size_var is not None:
            clone.tile_size_var = clone_tile_size_var(cast(TileSizeVar, tile_size_var))
        finish_no_tile_list(index_var, clone)
        return clone

    def clone_index_expr(expression: IndexVarAdd) -> IndexVarAdd:
        key = id(expression)
        cached = memo.get(key)
        if cached is not None:
            return cast(IndexVarAdd, cached)
        clone = object.__new__(IndexVarAdd)
        memo[key] = clone
        initialize(expression, clone)
        expression_state = state(expression)
        clone.lhs = clone_index(cast(IndexVar, expression_state["lhs"]))
        clone.rhs = clone_index(cast(IndexVar, expression_state["rhs"]))
        finish_no_tile_list(expression, clone)
        return clone

    def clone_tensor(tensor: TensorVar) -> TensorVar:
        key = id(tensor)
        cached = memo.get(key)
        if cached is not None:
            return cast(TensorVar, cached)
        clone = object.__new__(type(tensor))
        memo[key] = clone
        initialize(tensor, clone)
        tensor_state = state(tensor)
        clone.symbol_id = cast(SymbolId, tensor_state["symbol_id"])
        clone._name = cast(str, tensor_state["_name"])
        clone._format = cast(Optional[TensorFormat], tensor_state["_format"])
        clone._assignment = None
        shape = tensor_state["shape"]
        clone.shape = cast(Optional[tuple[int, ...]], shape)
        clone.dtype = cast(torch.dtype, tensor_state["dtype"])
        mode_order = tensor_state["mode_order"]
        clone.mode_order = (
            list(cast(List[int], mode_order)) if mode_order is not None else None
        )
        if type(tensor) is Workspace:
            workspace = cast(Workspace, clone)
            workspace.dim = cast(int, tensor_state["dim"])
            workspace.dense = cast(bool, tensor_state["dense"])
            tile_size_var = tensor_state["_tile_size_var"]
            workspace._tile_size_var = (
                clone_tile_size_var(cast(TileSizeVar, tile_size_var))
                if tile_size_var is not None
                else None
            )
            workspace.workspace_accesses = []
        finish_no_tile_list(tensor, clone)
        return clone

    def clone_expr(expression: object) -> object:
        key = id(expression)
        cached = memo.get(key)
        if cached is not None:
            return cached
        expression_type = type(expression)
        expression_state = state(expression)
        if expression_type is WorkspaceAccess:
            workspace_clone = object.__new__(WorkspaceAccess)
            memo[key] = workspace_clone
            initialize(expression, workspace_clone)
            workspace = cast(
                Workspace,
                clone_tensor(cast(Workspace, expression_state["wksp"])),
            )
            workspace_clone.access_id = cast(
                AccessId,
                expression_state["access_id"],
            )
            workspace_clone.wksp = workspace
            workspace_clone.tensor = workspace
            workspace_clone.tensor_id = cast(
                SymbolId,
                expression_state["tensor_id"],
            )
            workspace_clone.indices = [
                clone_index(cast(IndexVar, index_var))
                for index_var in cast(List[IndexVar], expression_state["indices"])
            ]
            workspace_clone.index_ids = cast(
                tuple[IndexId, ...],
                expression_state["index_ids"],
            )
            finish_no_tile_list(expression, workspace_clone)
            return workspace_clone
        if expression_type is TensorAccess:
            access_clone = object.__new__(TensorAccess)
            memo[key] = access_clone
            initialize(expression, access_clone)
            access_clone.access_id = cast(AccessId, expression_state["access_id"])
            access_clone.tensor = clone_tensor(
                cast(TensorVar, expression_state["tensor"])
            )
            access_clone.tensor_id = cast(
                SymbolId,
                expression_state["tensor_id"],
            )
            access_clone.indices = [
                clone_index(cast(IndexVar, index_var))
                for index_var in cast(List[IndexVar], expression_state["indices"])
            ]
            access_clone.index_ids = cast(
                tuple[IndexId, ...],
                expression_state["index_ids"],
            )
            finish_no_tile_list(expression, access_clone)
            return access_clone
        if expression_type is BinaryOp:
            binary_clone = object.__new__(BinaryOp)
            memo[key] = binary_clone
            initialize(expression, binary_clone)
            object.__setattr__(binary_clone, "op", expression_state["op"])
            object.__setattr__(
                binary_clone,
                "left",
                clone_expr(expression_state["left"]),
            )
            object.__setattr__(
                binary_clone,
                "right",
                clone_expr(expression_state["right"]),
            )
            finish_no_tile_list(expression, binary_clone)
            return binary_clone
        if expression_type is UnaryOp:
            unary_clone = object.__new__(UnaryOp)
            memo[key] = unary_clone
            initialize(expression, unary_clone)
            object.__setattr__(unary_clone, "op", expression_state["op"])
            object.__setattr__(
                unary_clone,
                "expr",
                clone_expr(expression_state["expr"]),
            )
            finish_no_tile_list(expression, unary_clone)
            return unary_clone
        raise VerificationError(
            "stage=legacy CIN adapter: unsupported expression in forward copy"
        )

    def clone_stmt(statement: IndexStmt) -> IndexStmt:
        key = id(statement)
        cached = memo.get(key)
        if cached is not None:
            return cast(IndexStmt, cached)
        statement_type = type(statement)
        statement_state = state(statement)
        if statement_type is ForAll:
            forall_clone = object.__new__(ForAll)
            memo[key] = forall_clone
            initialize(statement, forall_clone)
            forall_clone.lhs = None
            forall_clone.rhs = None
            forall_clone.parent = None
            forall_clone.index_var = clone_index(
                cast(IndexVar, statement_state["index_var"])
            )
            forall_clone.stmt = clone_stmt(cast(IndexStmt, statement_state["stmt"]))
            forall_clone.parallel = cast(
                Optional[bool],
                statement_state["parallel"],
            )
            finish_no_tile_list(statement, forall_clone)
            return forall_clone
        if statement_type is Where:
            where_clone = object.__new__(Where)
            memo[key] = where_clone
            initialize(statement, where_clone)
            where_clone.lhs = None
            where_clone.rhs = None
            where_clone.parent = None
            where_clone.producer = clone_stmt(
                cast(IndexStmt, statement_state["producer"])
            )
            where_clone.consumer = clone_stmt(
                cast(IndexStmt, statement_state["consumer"])
            )
            finish_no_tile_list(statement, where_clone)
            return where_clone
        if statement_type is TensorAssign:
            assignment_clone = object.__new__(TensorAssign)
            memo[key] = assignment_clone
            initialize(statement, assignment_clone)
            assignment_clone.parent = None
            assignment_clone.lhs = cast(
                TensorAccess,
                clone_expr(statement_state["lhs"]),
            )
            assignment_clone.rhs = cast(
                IndexExpr,
                clone_expr(statement_state["rhs"]),
            )
            assignment_clone.op = cast(
                Optional[Operation],
                statement_state["op"],
            )
            finish_no_tile_list(statement, assignment_clone)
            return assignment_clone
        raise VerificationError(
            "stage=legacy CIN adapter: unsupported statement in forward copy"
        )

    return clone_stmt(cin)


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

    def canonical_index(index_var: IndexVar, relation: str) -> IndexVar:
        canonical = index_vars.get(index_var.index_id)
        if canonical is None:
            raise VerificationError(
                "stage=legacy CIN adapter: "
                f"{relation} references an index outside the working tree"
            )
        return canonical

    for index_var in index_vars.values():
        if isinstance(index_var._expr, IndexVarAdd):
            index_var._expr.lhs = canonical_index(
                index_var._expr.lhs,
                "IndexVar expression lhs",
            )
            index_var._expr.rhs = canonical_index(
                index_var._expr.rhs,
                "IndexVar expression rhs",
            )
            index_var._expr.lhs._parent = index_var
            index_var._expr.rhs._parent = index_var
        tile_size_var = index_var.tile_size_var
        if tile_size_var is not None:
            tile_size_var.outer_index_var = canonical_index(
                tile_size_var.outer_index_var,
                "TileSizeVar outer index",
            )
            tile_size_var.inner_index_var = canonical_index(
                tile_size_var.inner_index_var,
                "TileSizeVar inner index",
            )
            if tile_size_var._index_var is not None:
                tile_size_var._index_var = canonical_index(
                    tile_size_var._index_var,
                    "TileSizeVar base index",
                )


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
