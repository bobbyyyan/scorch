"""Post-CIN lowering for schedule operations that need concrete iterators.

Affine strip-mining is represented directly in CIN. Sparse coordinate panels are
different: their inner bounds depend on both the containing sparse row and the
outer coordinate panel. This pass lowers that explicit schedule after the CSR
position/coordinate iterators exist in LLIR.
"""

import re
from typing import Dict, List, Optional, Tuple

from . import llir
from .scheduler import Schedule, TileSpec, _RelayoutPlan, _ResultTilePlan

LoopLocation = Tuple[List[llir.Stmt], int, llir.ForLoop]


def _nested_bodies(stmt: llir.Stmt) -> List[List[llir.Stmt]]:
    bodies: List[List[llir.Stmt]] = []
    if isinstance(stmt, (llir.ForLoop, llir.WhileLoop, llir.ForLoopAuto)):
        bodies.append(stmt.body)
    elif isinstance(stmt, llir.IfThenElse):
        if stmt.then_body:
            bodies.append(stmt.then_body)
        if stmt.else_body:
            bodies.append(stmt.else_body)
        if stmt.then_body_list:
            bodies.extend(stmt.then_body_list)
    return bodies


def _find_tagged_loop(
    stmts: List[llir.Stmt],
    name: str,
    ancestors: Optional[List[LoopLocation]] = None,
) -> Tuple[LoopLocation, List[LoopLocation]]:
    ancestors = ancestors or []
    for index, stmt in enumerate(stmts):
        if isinstance(stmt, llir.ForLoop):
            location = (stmts, index, stmt)
            if getattr(stmt, "scorch_index_var", None) == name:
                return location, ancestors
            try:
                return _find_tagged_loop(stmt.body, name, ancestors + [location])
            except LookupError:
                pass
        else:
            for body in _nested_bodies(stmt):
                try:
                    return _find_tagged_loop(body, name, ancestors)
                except LookupError:
                    pass
    raise LookupError(f"No generated loop is tagged for index variable {name!r}")


def _find_coordinate_array(stmts: List[llir.Stmt], position: str) -> str:
    pattern = re.compile(rf"^([A-Za-z_]\w*_crd)\[{re.escape(position)}\]$")
    for stmt in stmts:
        if isinstance(stmt, llir.VarInit) and isinstance(stmt.value, llir.Var):
            match = pattern.match(stmt.value.name)
            if match:
                return match.group(1)
        for body in _nested_bodies(stmt):
            try:
                return _find_coordinate_array(body, position)
            except LookupError:
                pass
    raise LookupError(
        f"Cannot find a coordinate array indexed by sparse position {position!r}"
    )


def _find_end_init(
    container: List[llir.Stmt],
    loop_index: int,
    end_name: str,
) -> Tuple[int, llir.VarInit]:
    for index in range(loop_index - 1, -1, -1):
        stmt = container[index]
        if isinstance(stmt, llir.VarInit) and stmt.var.name == end_name:
            return index, stmt
    raise LookupError(f"Cannot find sparse iterator end declaration {end_name!r}")


def _window_sparse_loop(
    location: LoopLocation,
    panel_var: str,
    panel_end_var: str,
) -> None:
    container, loop_index, sparse_loop = location
    if not isinstance(sparse_loop.init, llir.VarInit):
        raise NotImplementedError("Panel tiling requires a canonical sparse for-loop")
    if not isinstance(sparse_loop.init.value, llir.Var):
        raise NotImplementedError("Panel tiling requires a named CSR row begin")
    if not isinstance(sparse_loop.cond, llir.BinOp):
        raise NotImplementedError("Panel tiling requires a canonical sparse bound")
    if not isinstance(sparse_loop.cond.right, llir.Var):
        raise NotImplementedError("Panel tiling requires a named CSR row end")

    position = sparse_loop.init.var.name
    end_name = sparse_loop.cond.right.name
    end_index, end_init = _find_end_init(container, loop_index, end_name)
    if not isinstance(end_init.value, llir.Var):
        raise NotImplementedError("Panel tiling requires a named CSR row end value")
    coordinate_array = _find_coordinate_array(sparse_loop.body, position)

    row_end = f"{position}_row_end"
    panel_begin = f"{position}_panel_begin"
    row_begin_expr = sparse_loop.init.value.name
    lower_expr = (
        f"(int) (std::lower_bound({coordinate_array} + {row_begin_expr}, "
        f"{coordinate_array} + {row_end}, {panel_var}) - {coordinate_array})"
    )
    upper_expr = (
        f"(int) (std::lower_bound({coordinate_array} + {panel_begin}, "
        f"{coordinate_array} + {row_end}, {panel_end_var}) - {coordinate_array})"
    )

    replacements: List[llir.Stmt] = [
        llir.VarInit(
            var=llir.Var(row_end, end_init.var.type),
            value=end_init.value,
        ),
        llir.VarInit(
            var=llir.Var(panel_begin, end_init.var.type),
            value=llir.Var(lower_expr, end_init.var.type),
        ),
        llir.VarInit(
            var=end_init.var,
            value=llir.Var(upper_expr, end_init.var.type),
        ),
    ]
    container[end_index : end_index + 1] = replacements
    sparse_loop.init.value = llir.Var(panel_begin, end_init.var.type)


def _parallel_ancestor(ancestors: List[LoopLocation]) -> LoopLocation:
    for location in reversed(ancestors):
        loop = location[2]
        if loop.omp_parallel_for or getattr(loop, "_use_atomic_scheduling", False):
            return location
    raise NotImplementedError(
        "Sparse panel tiling requires a parallel row loop surrounding the "
        "compressed iterator"
    )


def _apply_panel_tile(
    function: llir.Function,
    tile: TileSpec,
    panel_bound: str,
) -> None:
    target_location, ancestors = _find_tagged_loop(function.body, tile.index_var)
    row_location = _parallel_ancestor(ancestors)
    row_loop = row_location[2]
    row_position = next(
        index for index, location in enumerate(ancestors) if location[2] is row_loop
    )

    if tile.placement.startswith("child_of:"):
        requested_parent = tile.placement.split(":", 1)[1]
        parent_position = next(
            (
                index
                for index, location in enumerate(ancestors)
                if getattr(location[2], "scorch_index_var", None) == requested_parent
            ),
            None,
        )
        if parent_position is None:
            raise ValueError(
                f"Panel placement parent {requested_parent!r} is not a generated "
                "ancestor loop"
            )
        if parent_position >= row_position:
            raise ValueError(
                "A sparse panel loop must surround the selected parallel loop; "
                f"it cannot be placed inside {requested_parent!r}"
            )
        insertion_location = ancestors[parent_position + 1]
    elif tile.placement == "outermost":
        insertion_location = ancestors[0]
    elif tile.placement != "outermost":
        raise NotImplementedError(
            "Sparse panel tiles support outermost or child_of:<generated-loop> "
            "placement"
        )

    panel_var = f"{tile.index_var}_out"
    panel_end = f"{panel_var}_end"
    tile_var = f"kTile_{tile.index_var}"
    _window_sparse_loop(target_location, panel_var, panel_end)

    insertion_container, insertion_index, wrapped_loop = insertion_location

    panel_loop = llir.ForLoop(
        init=llir.VarInit(
            var=llir.Var(panel_var, llir.DataType.INT64),
            value=llir.Literal(0),
        ),
        cond=llir.BinOp(
            op="<",
            left=llir.Var(panel_var, llir.DataType.INT64),
            right=llir.Var(panel_bound, llir.DataType.INT64),
        ),
        update=llir.Assign(
            var=llir.Var(panel_var, llir.DataType.INT64),
            value=llir.Var(tile_var, llir.DataType.INT64),
            op=llir.AssignOp.ADD_ASSIGN,
        ),
        body=[
            llir.VarInit(
                var=llir.Var(panel_end, llir.DataType.INT64),
                value=llir.FunctionCall(
                    name="std::min",
                    args=[
                        llir.Add(
                            llir.Var(panel_var, llir.DataType.INT64),
                            llir.Var(tile_var, llir.DataType.INT64),
                        ),
                        llir.Var(panel_bound, llir.DataType.INT64),
                    ],
                ),
            ),
            wrapped_loop,
        ],
    )
    panel_loop.scorch_index_var = panel_var
    insertion_container[insertion_index] = panel_loop

    function.body[0:0] = [
        llir.Comment(f"Initialize {tile.index_var} panel tile size"),
        llir.VarInit(
            var=llir.Var(tile_var, llir.DataType.CONSTEXPR_INT),
            value=llir.Literal(tile.width),
        ),
        llir.BlankLine(),
    ]


def _find_var_init(stmts: List[llir.Stmt], name: str) -> llir.VarInit:
    for stmt in stmts:
        if isinstance(stmt, llir.VarInit) and stmt.var.name == name:
            return stmt
        for body in _nested_bodies(stmt):
            try:
                return _find_var_init(body, name)
            except LookupError:
                pass
    raise LookupError(f"Cannot find generated declaration for {name!r}")


def _declared_names(function: llir.Function) -> set[str]:
    names = {arg.name for arg in function.args if isinstance(arg, llir.Var)}

    def collect(stmts: List[llir.Stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (llir.VarInit, llir.VarDecl)):
                names.add(stmt.var.name)
            elif isinstance(stmt, llir.ForLoop) and stmt.init is not None:
                names.add(stmt.init.var.name)
            elif isinstance(stmt, llir.ForLoopAuto):
                names.add(stmt.var.name)
            for body in _nested_bodies(stmt):
                collect(body)

    collect(function.body)
    return names


def _unique_name(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _matches_tensor_access(
    expr: llir.Expr,
    tensor_name: str,
    index_vars: Tuple[str, ...],
    role: llir.TensorAccessRole,
) -> bool:
    if not isinstance(expr, llir.Var) or expr.tensor_access is None:
        return False
    metadata = expr.tensor_access
    return (
        metadata.tensor_name == tensor_name
        and metadata.index_vars == index_vars
        and metadata.role == role
    )


def _rewrite_expr_access(
    expr: llir.Expr,
    tensor_name: str,
    index_vars: Tuple[str, ...],
    role: llir.TensorAccessRole,
    replacement: llir.Var,
) -> Tuple[llir.Expr, int]:
    """Redirect one CIN tensor access in a structured LLIR expression tree."""
    if isinstance(expr, llir.Var):
        if _matches_tensor_access(expr, tensor_name, index_vars, role):
            return replacement, 1
        return expr, 0
    if isinstance(expr, llir.BinOp):
        expr.left, left_count = _rewrite_expr_access(
            expr.left, tensor_name, index_vars, role, replacement
        )
        expr.right, right_count = _rewrite_expr_access(
            expr.right, tensor_name, index_vars, role, replacement
        )
        return expr, left_count + right_count
    if isinstance(expr, llir.UnaryOp):
        expr.operand, count = _rewrite_expr_access(
            expr.operand, tensor_name, index_vars, role, replacement
        )
        return expr, count
    if isinstance(expr, llir.FunctionCall):
        count = 0
        rewritten_args = []
        for arg in expr.args:
            new_arg, arg_count = _rewrite_expr_access(
                arg, tensor_name, index_vars, role, replacement
            )
            rewritten_args.append(new_arg)
            count += arg_count
        expr.args = rewritten_args
        return expr, count
    if isinstance(expr, llir.ArrayAccess):
        expr.array, array_count = _rewrite_expr_access(
            expr.array, tensor_name, index_vars, role, replacement
        )
        expr.index, index_count = _rewrite_expr_access(
            expr.index, tensor_name, index_vars, role, replacement
        )
        return expr, array_count + index_count
    if isinstance(expr, llir.Cast):
        rewritten_expr, count = _rewrite_expr_access(
            expr.expr, tensor_name, index_vars, role, replacement
        )
        return llir.Cast(rewritten_expr, expr.data_type), count
    return expr, 0


def _rewrite_stmt_accesses(
    stmts: List[llir.Stmt],
    tensor_name: str,
    index_vars: Tuple[str, ...],
    role: llir.TensorAccessRole,
    replacement: llir.Var,
) -> int:
    """Rewrite matching tensor accesses without parsing rendered C++ names."""
    count = 0
    for stmt in stmts:
        if isinstance(stmt, llir.VarInit):
            stmt.value, rewritten = _rewrite_expr_access(
                stmt.value, tensor_name, index_vars, role, replacement
            )
            count += rewritten
        elif isinstance(stmt, llir.Assign):
            stmt.var, lhs_count = _rewrite_expr_access(
                stmt.var, tensor_name, index_vars, role, replacement
            )
            stmt.value, rhs_count = _rewrite_expr_access(
                stmt.value, tensor_name, index_vars, role, replacement
            )
            count += lhs_count + rhs_count
        elif isinstance(stmt, llir.ForLoop):
            if stmt.init is not None:
                count += _rewrite_stmt_accesses(
                    [stmt.init], tensor_name, index_vars, role, replacement
                )
            stmt.cond, cond_count = _rewrite_expr_access(
                stmt.cond, tensor_name, index_vars, role, replacement
            )
            count += cond_count
            if isinstance(stmt.update, llir.Assign):
                count += _rewrite_stmt_accesses(
                    [stmt.update], tensor_name, index_vars, role, replacement
                )
            count += _rewrite_stmt_accesses(
                stmt.body, tensor_name, index_vars, role, replacement
            )
        elif isinstance(stmt, llir.WhileLoop):
            stmt.cond, cond_count = _rewrite_expr_access(
                stmt.cond, tensor_name, index_vars, role, replacement
            )
            count += cond_count
            count += _rewrite_stmt_accesses(
                stmt.body, tensor_name, index_vars, role, replacement
            )
        elif isinstance(stmt, llir.IfThenElse):
            if stmt.cond is not None:
                stmt.cond, cond_count = _rewrite_expr_access(
                    stmt.cond, tensor_name, index_vars, role, replacement
                )
                count += cond_count
            if stmt.cond_list:
                rewritten_conditions = []
                for condition in stmt.cond_list:
                    condition, cond_count = _rewrite_expr_access(
                        condition, tensor_name, index_vars, role, replacement
                    )
                    rewritten_conditions.append(condition)
                    count += cond_count
                stmt.cond_list = rewritten_conditions
            for body in _nested_bodies(stmt):
                count += _rewrite_stmt_accesses(
                    body, tensor_name, index_vars, role, replacement
                )
        elif isinstance(stmt, llir.FunctionCallStmt):
            rewritten_args = []
            for arg in stmt.args:
                arg, arg_count = _rewrite_expr_access(
                    arg, tensor_name, index_vars, role, replacement
                )
                rewritten_args.append(arg)
                count += arg_count
            stmt.args = rewritten_args
    return count


def _contains_tensor_access(
    stmts: List[llir.Stmt],
    tensor_name: str,
    index_vars: Tuple[str, ...],
    role: llir.TensorAccessRole,
) -> bool:
    """Whether structured LLIR still contains a selected logical access."""
    for stmt in stmts:
        expressions: List[llir.Expr] = []
        if isinstance(stmt, llir.VarInit):
            expressions.append(stmt.value)
        elif isinstance(stmt, llir.Assign):
            expressions.extend([stmt.var, stmt.value])
        elif isinstance(stmt, (llir.ForLoop, llir.WhileLoop)):
            expressions.append(stmt.cond)
        elif isinstance(stmt, llir.IfThenElse):
            if stmt.cond is not None:
                expressions.append(stmt.cond)
            expressions.extend(stmt.cond_list or [])
        elif isinstance(stmt, llir.FunctionCallStmt):
            expressions.extend(stmt.args)

        pending = list(expressions)
        while pending:
            expr = pending.pop()
            if _matches_tensor_access(expr, tensor_name, index_vars, role):
                return True
            if isinstance(expr, llir.BinOp):
                pending.extend([expr.left, expr.right])
            elif isinstance(expr, llir.UnaryOp):
                pending.append(expr.operand)
            elif isinstance(expr, llir.FunctionCall):
                pending.extend(expr.args)
            elif isinstance(expr, llir.ArrayAccess):
                pending.extend([expr.array, expr.index])
            elif isinstance(expr, llir.Cast):
                pending.append(expr.expr)

        for body in _nested_bodies(stmt):
            if _contains_tensor_access(body, tensor_name, index_vars, role):
                return True
    return False


def _redirect_sparse_prefetch(
    sparse_loop: llir.ForLoop,
    operand_value_array: str,
    packed_name: str,
    panel_var: str,
    panel_end: str,
    panel_tile_var: str,
    stage_row_origin: Optional[str],
) -> None:
    if not isinstance(sparse_loop.init, llir.VarInit):
        raise NotImplementedError(
            "Packed relayout requires a canonical compressed iterator"
        )
    if not isinstance(sparse_loop.cond, llir.BinOp) or not isinstance(
        sparse_loop.cond.right, llir.Var
    ):
        raise NotImplementedError(
            "Packed relayout requires a named compressed iterator end"
        )
    position = sparse_loop.init.var.name
    coordinate_array = _find_coordinate_array(sparse_loop.body, position)
    end_name = sparse_loop.cond.right.name
    marker = f"__builtin_prefetch(&{operand_value_array}["
    removed = False
    retained: List[llir.Stmt] = []
    for stmt in sparse_loop.body:
        if isinstance(stmt, llir.RawStmt) and marker in stmt.code:
            removed = True
            continue
        retained.append(stmt)
    sparse_loop.body = retained
    if removed:
        coordinate = f"{coordinate_array}[{position} + 1]"
        staged_row = (
            coordinate
            if stage_row_origin is None
            else f"({coordinate} - {stage_row_origin})"
        )
        sparse_loop.body.insert(
            0,
            llir.RawStmt(
                code=(
                    f"if ({position} + 1 < {end_name} && "
                    f"{coordinate_array}[{position} + 1] >= {panel_var} && "
                    f"{coordinate_array}[{position} + 1] < {panel_end}) "
                    f"__builtin_prefetch(&{packed_name}[{staged_row} * "
                    f"{panel_tile_var}], 0, 1)"
                )
            ),
        )


def _remove_dense_result_zero(function: llir.Function, result: str) -> None:
    """Remove the default whole-result zero after copy-out coverage is proven."""
    value_name = f"{result}_values"
    matches = [
        index
        for index, stmt in enumerate(function.body)
        if isinstance(stmt, llir.FunctionCallStmt)
        and stmt.name == "scorch_zero_dense"
        and stmt.args
        and isinstance(stmt.args[0], llir.Var)
        and stmt.args[0].name == value_name
    ]
    if len(matches) != 1:
        raise NotImplementedError(
            "Heap accumulation requires exactly one generated dense-result zero"
        )
    del function.body[matches[0]]


def _apply_heap_result_tile(
    function: llir.Function,
    schedule: Schedule,
    plan: _ResultTilePlan,
) -> None:
    """Redirect a dense result tile to compact storage and copy it out once."""
    tile = next(
        (
            candidate
            for candidate in schedule.tiles
            if candidate.index_var == plan.tile_var
            and candidate.kind == "affine"
            and candidate.accum == "heap"
        ),
        None,
    )
    if tile is None:
        raise ValueError("Missing heap TileSpec for a derived result-tile plan")

    tile_outer_name = f"{plan.tile_var}_out"
    tile_inner_name = f"{plan.tile_var}_in"
    tile_size_name = f"kTile_{plan.tile_var}"
    tile_location, tile_ancestors = _find_tagged_loop(function.body, tile_outer_name)
    if tile_ancestors or tile_location[0] is not function.body:
        raise NotImplementedError(
            "Heap accumulation requires the affine result tile to be outermost"
        )
    tile_loop = tile_location[2]

    result_values = f"{plan.result}_values"
    result_value_init = _find_var_init(function.body, result_values)
    pointer_type = result_value_init.var.type
    if pointer_type.value == "float*":
        scalar_type = llir.DataType.FLOAT32
        zero_value = "0.0f"
    elif pointer_type.value == "double*":
        scalar_type = llir.DataType.FLOAT64
        zero_value = "0.0"
    else:
        raise NotImplementedError(
            "Heap accumulation supports generated float or double result pointers"
        )

    used_names = _declared_names(function)
    compact_name = _unique_name(f"tiled_{plan.result}", used_names)
    storage_name = _unique_name(f"{compact_name}_storage", used_names)
    init_prefix = _unique_name(f"{plan.result}_tile_init", used_names)
    init_inner = _unique_name(f"{plan.tile_var}_tile_init", used_names)
    init_logical = _unique_name(f"{plan.tile_var}_tile_logical", used_names)
    copy_prefix = _unique_name(f"{plan.result}_tile_copy", used_names)
    copy_inner = _unique_name(f"{plan.tile_var}_tile_copy", used_names)
    copy_logical = _unique_name(f"{plan.tile_var}_copy_logical", used_names)

    trailing_bound = f"{plan.result}{plan.result_level}_size"
    prefix_extent = " * ".join(
        f"{plan.result}{level}_size" for level in range(plan.result_level)
    )
    if not prefix_extent:
        raise NotImplementedError(
            "Heap accumulation requires at least one dense result prefix level"
        )

    compact_access = llir.Var(
        name=(
            f"{compact_name}[p{plan.result}{plan.result_level - 1} * "
            f"{tile_size_name} + {tile_inner_name}]"
        ),
        type=scalar_type,
    )
    rewritten = _rewrite_stmt_accesses(
        tile_loop.body,
        plan.result,
        plan.access_index_vars,
        llir.TensorAccessRole.RESULT_WRITE,
        compact_access,
    )
    if rewritten == 0:
        raise NotImplementedError(
            "Cannot redirect the selected logical result access to compact storage"
        )
    if _contains_tensor_access(
        tile_loop.body,
        plan.result,
        plan.access_index_vars,
        llir.TensorAccessRole.RESULT_WRITE,
    ):
        raise NotImplementedError(
            "Heap accumulation left a direct result write in the compute region"
        )

    init_inner_loop = llir.ForLoop(
        init=llir.VarInit(
            var=llir.Var(init_inner, llir.DataType.INT64),
            value=llir.Literal(0),
        ),
        cond=llir.BinOp(
            op="<",
            left=llir.Var(init_inner, llir.DataType.INT64),
            right=llir.Var(tile_size_name, llir.DataType.INT64),
        ),
        update=llir.Increment(llir.Var(init_inner, llir.DataType.INT64)),
        body=[
            llir.VarInit(
                var=llir.Var(init_logical, llir.DataType.INT64),
                value=llir.Add(
                    llir.Var(tile_outer_name, llir.DataType.INT64),
                    llir.Var(init_inner, llir.DataType.INT64),
                ),
            ),
            llir.IfThenElse(
                cond=llir.BinOp(
                    op=">=",
                    left=llir.Var(init_logical, llir.DataType.INT64),
                    right=llir.Var(trailing_bound, llir.DataType.INT64),
                ),
                then_body=[llir.Break()],
            ),
            llir.Assign(
                var=llir.Var(
                    f"{compact_name}[{init_prefix} * {tile_size_name} + "
                    f"{init_inner}]",
                    scalar_type,
                ),
                value=llir.Var(zero_value, scalar_type),
            ),
        ],
    )
    init_loop = llir.ForLoop(
        init=llir.VarInit(
            var=llir.Var(init_prefix, llir.DataType.INT64),
            value=llir.Literal(0),
        ),
        cond=llir.BinOp(
            op="<",
            left=llir.Var(init_prefix, llir.DataType.INT64),
            right=llir.Var(prefix_extent, llir.DataType.INT64),
        ),
        update=llir.Increment(llir.Var(init_prefix, llir.DataType.INT64)),
        body=[init_inner_loop],
        omp_parallel_for=True,
        omp_schedule="static",
        omp_num_threads=(
            f"scorch_nthreads(({prefix_extent}) * {tile_size_name}, "
            f"({prefix_extent}))"
        ),
    )
    init_loop.scorch_index_var = f"init:{plan.result}"

    copy_inner_loop = llir.ForLoop(
        init=llir.VarInit(
            var=llir.Var(copy_inner, llir.DataType.INT64),
            value=llir.Literal(0),
        ),
        cond=llir.BinOp(
            op="<",
            left=llir.Var(copy_inner, llir.DataType.INT64),
            right=llir.Var(tile_size_name, llir.DataType.INT64),
        ),
        update=llir.Increment(llir.Var(copy_inner, llir.DataType.INT64)),
        body=[
            llir.VarInit(
                var=llir.Var(copy_logical, llir.DataType.INT64),
                value=llir.Add(
                    llir.Var(tile_outer_name, llir.DataType.INT64),
                    llir.Var(copy_inner, llir.DataType.INT64),
                ),
            ),
            llir.IfThenElse(
                cond=llir.BinOp(
                    op=">=",
                    left=llir.Var(copy_logical, llir.DataType.INT64),
                    right=llir.Var(trailing_bound, llir.DataType.INT64),
                ),
                then_body=[llir.Break()],
            ),
            llir.Assign(
                var=llir.Var(
                    f"{result_values}[{copy_prefix} * {trailing_bound} + "
                    f"{copy_logical}]",
                    scalar_type,
                ),
                value=llir.Var(
                    f"{compact_name}[{copy_prefix} * {tile_size_name} + "
                    f"{copy_inner}]",
                    scalar_type,
                ),
            ),
        ],
    )
    copy_loop = llir.ForLoop(
        init=llir.VarInit(
            var=llir.Var(copy_prefix, llir.DataType.INT64),
            value=llir.Literal(0),
        ),
        cond=llir.BinOp(
            op="<",
            left=llir.Var(copy_prefix, llir.DataType.INT64),
            right=llir.Var(prefix_extent, llir.DataType.INT64),
        ),
        update=llir.Increment(llir.Var(copy_prefix, llir.DataType.INT64)),
        body=[copy_inner_loop],
        omp_parallel_for=True,
        omp_schedule="static",
        omp_num_threads=(
            f"scorch_nthreads(({prefix_extent}) * {tile_size_name}, "
            f"({prefix_extent}))"
        ),
    )
    copy_loop.scorch_index_var = f"copy:{plan.result}"

    _remove_dense_result_zero(function, plan.result)
    tile_loop.body[0:0] = [
        llir.Comment(f"Initialize compact result tile for {plan.result}"),
        init_loop,
        llir.BlankLine(),
    ]
    tile_loop.body.extend(
        [
            llir.BlankLine(),
            llir.Comment(f"Copy compact result tile to {plan.result}"),
            copy_loop,
        ]
    )

    # Removing the default zero shifts top-level positions, so resolve the tagged
    # tile location again before inserting its reusable allocation.
    (tile_container, tile_index, _), _ = _find_tagged_loop(
        function.body, tile_outer_name
    )
    tile_container[tile_index:tile_index] = [
        llir.Comment(f"Allocate reusable compact result tile for {plan.result}"),
        llir.RawStmt(
            code=(
                f"std::vector<{scalar_type.value}> {storage_name}("
                f"(size_t) ({prefix_extent}) * (size_t) {tile_size_name})"
            )
        ),
        llir.VarInit(
            var=llir.Var(compact_name, pointer_type, is_restrict=True),
            value=llir.Var(f"{storage_name}.data()", pointer_type),
        ),
        llir.BlankLine(),
    ]


def _apply_relayout(
    function: llir.Function,
    schedule: Schedule,
    plan: _RelayoutPlan,
) -> None:
    relayout = schedule.relayout
    if relayout is None:
        raise ValueError("Missing RelayoutSpec for a derived relayout plan")

    pack_outer_name = f"{plan.pack_var}_out"
    pack_inner_name = f"{plan.pack_var}_in"
    panel_outer_name = f"{plan.panel_var}_out"
    pack_location, pack_ancestors = _find_tagged_loop(function.body, pack_outer_name)
    if pack_ancestors or pack_location[0] is not function.body:
        raise NotImplementedError(
            "Packed relayout requires the affine pack tile to be outermost"
        )
    panel_location, panel_ancestors = _find_tagged_loop(function.body, panel_outer_name)
    if not panel_ancestors or panel_ancestors[-1][2] is not pack_location[2]:
        raise NotImplementedError(
            "Packed relayout requires the panel loop directly inside the pack tile"
        )
    row_location, row_ancestors = _find_tagged_loop(function.body, plan.row_var)
    if not row_ancestors or row_ancestors[-1][2] is not panel_location[2]:
        raise NotImplementedError(
            "Packed relayout requires the parallel CSR row loop directly inside "
            "the panel loop"
        )
    row_loop = row_location[2]
    if not row_loop.omp_parallel_for and not getattr(
        row_loop, "_use_atomic_scheduling", False
    ):
        raise NotImplementedError(
            "Packed relayout requires a generated parallel CSR row loop"
        )
    sparse_location, _ = _find_tagged_loop(row_loop.body, plan.panel_var)
    sparse_loop = sparse_location[2]
    _find_tagged_loop(sparse_loop.body, pack_inner_name)

    operand_value_array = f"{plan.operand}_val"
    operand_value_init = _find_var_init(function.body, operand_value_array)
    pointer_type = operand_value_init.var.type
    if pointer_type.value == "float*":
        scalar_type = llir.DataType.FLOAT32
    elif pointer_type.value == "double*":
        scalar_type = llir.DataType.FLOAT64
    else:
        raise NotImplementedError(
            "Packed relayout supports generated float or double operand pointers"
        )

    used_names = _declared_names(function)
    packed_name = _unique_name(f"packed_{plan.operand}", used_names)
    storage_name = _unique_name(f"{packed_name}_storage", used_names)
    pack_tile_var = f"kTile_{plan.pack_var}"
    panel_tile_var = f"kTile_{plan.panel_var}"
    panel_end = f"{panel_outer_name}_end"
    pack_axis_bound = f"{plan.operand}{plan.operand_pack_level}_size"
    panel_axis_bound = f"{plan.operand}{plan.operand_panel_level}_size"
    panel_scoped = plan.scope_var == plan.panel_var
    stage_row_origin = panel_outer_name if panel_scoped else None
    staged_read_row = (
        f"({plan.panel_var} - {panel_outer_name})" if panel_scoped else plan.panel_var
    )
    packed_read = llir.Var(
        name=(
            f"{packed_name}[{staged_read_row} * "
            f"{pack_tile_var} + {pack_inner_name}]"
        ),
        type=scalar_type,
    )
    rewritten = _rewrite_stmt_accesses(
        row_loop.body,
        plan.operand,
        plan.access_index_vars,
        llir.TensorAccessRole.INPUT_READ,
        packed_read,
    )
    if rewritten == 0:
        raise NotImplementedError(
            "Cannot redirect the selected logical operand access to packed storage"
        )
    _redirect_sparse_prefetch(
        sparse_loop,
        operand_value_array,
        packed_name,
        panel_outer_name,
        panel_end,
        pack_tile_var,
        stage_row_origin,
    )
    panel_coordinate_index = next(
        (
            index
            for index, stmt in enumerate(sparse_loop.body)
            if isinstance(stmt, llir.VarInit) and stmt.var.name == plan.panel_var
        ),
        None,
    )
    if panel_coordinate_index is None:
        raise LookupError(
            "Cannot find the resolved sparse panel coordinate for packed access"
        )
    sparse_loop.body.insert(
        panel_coordinate_index + 1,
        llir.IfThenElse(
            cond=llir.BinOp(
                op="||",
                left=llir.BinOp(
                    op="<",
                    left=llir.Var(plan.panel_var, llir.DataType.INT64),
                    right=llir.Var(panel_outer_name, llir.DataType.INT64),
                ),
                right=llir.BinOp(
                    op=">=",
                    left=llir.Var(plan.panel_var, llir.DataType.INT64),
                    right=llir.Var(panel_end, llir.DataType.INT64),
                ),
            ),
            then_body=[llir.Continue()],
        ),
    )
    if _contains_tensor_access(
        row_loop.body,
        plan.operand,
        plan.access_index_vars,
        llir.TensorAccessRole.INPUT_READ,
    ):
        raise NotImplementedError(
            "Packed relayout left an unpacked operand read in the compute region"
        )

    pack_row = _unique_name(f"{plan.panel_var}_pack", used_names)
    pack_col = _unique_name(f"{plan.pack_var}_pack", used_names)
    logical_pack_col = _unique_name(f"{plan.pack_var}_packed", used_names)
    source_index = f"{pack_row} * {pack_axis_bound} + {logical_pack_col}"
    destination_row = f"({pack_row} - {panel_outer_name})" if panel_scoped else pack_row
    destination_index = f"{destination_row} * {pack_tile_var} + {pack_col}"
    pack_inner = llir.ForLoop(
        init=llir.VarInit(
            var=llir.Var(pack_col, llir.DataType.INT64),
            value=llir.Literal(0),
        ),
        cond=llir.BinOp(
            op="<",
            left=llir.Var(pack_col, llir.DataType.INT64),
            right=llir.Var(pack_tile_var, llir.DataType.INT64),
        ),
        update=llir.Increment(llir.Var(pack_col, llir.DataType.INT64)),
        body=[
            llir.VarInit(
                var=llir.Var(logical_pack_col, llir.DataType.INT64),
                value=llir.Add(
                    llir.Var(pack_outer_name, llir.DataType.INT64),
                    llir.Var(pack_col, llir.DataType.INT64),
                ),
            ),
            llir.IfThenElse(
                cond=llir.BinOp(
                    op=">=",
                    left=llir.Var(logical_pack_col, llir.DataType.INT64),
                    right=llir.Var(pack_axis_bound, llir.DataType.INT64),
                ),
                then_body=[llir.Break()],
            ),
            llir.Assign(
                var=llir.Var(f"{packed_name}[{destination_index}]", scalar_type),
                value=llir.Var(f"{operand_value_array}[{source_index}]", scalar_type),
            ),
        ],
    )
    pack_outer = llir.ForLoop(
        init=llir.VarInit(
            var=llir.Var(pack_row, llir.DataType.INT64),
            value=(
                llir.Var(panel_outer_name, llir.DataType.INT64)
                if panel_scoped
                else llir.Literal(0)
            ),
        ),
        cond=llir.BinOp(
            op="<",
            left=llir.Var(pack_row, llir.DataType.INT64),
            right=llir.Var(
                panel_end if panel_scoped else panel_axis_bound,
                llir.DataType.INT64,
            ),
        ),
        update=llir.Increment(llir.Var(pack_row, llir.DataType.INT64)),
        body=[pack_inner],
        omp_parallel_for=True,
        omp_schedule="static",
        omp_num_threads=(
            "scorch_nthreads("
            + (
                f"({panel_end} - {panel_outer_name}) * {pack_tile_var}, "
                f"({panel_end} - {panel_outer_name})"
                if panel_scoped
                else f"{panel_axis_bound} * {pack_tile_var}, {panel_axis_bound}"
            )
            + ")"
        ),
    )
    pack_outer.scorch_index_var = f"pack:{plan.operand}"

    if panel_scoped:
        stage_container = panel_location[2].body
        stage_index = next(
            (index for index, stmt in enumerate(stage_container) if stmt is row_loop),
            None,
        )
        scope_description = f"{plan.panel_var} panel"
    else:
        stage_container = pack_location[2].body
        # A full-access stage depends only on the enclosing affine tile.  Place
        # it at tile entry so it is realized once before result initialization
        # and before every enclosed reduction panel.
        stage_index = 0
        scope_description = f"full {plan.panel_var} axis"
    if stage_index is None:
        raise LookupError("Cannot place packed operand at its requested loop scope")
    stage_container[stage_index:stage_index] = [
        llir.Comment(
            f"Pack {plan.operand} {scope_description} into contiguous "
            f"{plan.panel_var}-major storage"
        ),
        pack_outer,
        llir.BlankLine(),
    ]

    pack_container, pack_index, _ = pack_location
    stage_rows = panel_tile_var if panel_scoped else panel_axis_bound
    pack_container[pack_index:pack_index] = [
        llir.Comment(f"Allocate reusable packed storage for {plan.operand}"),
        llir.RawStmt(
            code=(
                f"std::vector<{scalar_type.value}> {storage_name}("
                f"(size_t) {stage_rows} * (size_t) {pack_tile_var})"
            )
        ),
        llir.VarInit(
            var=llir.Var(packed_name, pointer_type, is_restrict=True),
            value=llir.Var(f"{storage_name}.data()", pointer_type),
        ),
        llir.BlankLine(),
    ]


def apply_schedule_to_llir(
    function: llir.Function,
    schedule: Schedule,
    panel_bounds: Dict[str, str],
    relayout_plan: Optional[_RelayoutPlan] = None,
    result_tile_plan: Optional[_ResultTilePlan] = None,
) -> llir.Function:
    """Apply schedule operations that require concrete LLIR iterator bounds."""
    panel_tiles = [tile for tile in schedule.tiles if tile.kind == "panel"]
    for tile in panel_tiles:
        bound = panel_bounds.get(tile.index_var)
        if bound is None:
            raise ValueError(f"Missing dense bound for panel {tile.index_var!r}")
        _apply_panel_tile(function, tile, bound)
    if any(tile.accum == "heap" for tile in schedule.tiles):
        if result_tile_plan is None:
            raise ValueError("Missing CIN-derived metadata for heap result tile")
        _apply_heap_result_tile(function, schedule, result_tile_plan)
    if schedule.relayout is not None:
        if relayout_plan is None:
            raise ValueError("Missing CIN-derived metadata for packed relayout")
        _apply_relayout(function, schedule, relayout_plan)
    return function
