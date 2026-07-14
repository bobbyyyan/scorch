"""Typed two-phase OpenMP lowering for legacy compressed ``Where`` output.

The production contract is deliberately narrow: the input is a top-level
statement list containing the serial outer loop for a ``d,s[,s...]`` result.
The first compatible *top-level* :class:`llir.ForLoop` is selected.  A missing
loop, or a selected loop whose bound is not ``loop_var < bound_var``, is a
legal detached no-op.  Successful application replaces the serial assembly
with independent count and fill loops separated by exact allocation and prefix
sum statements.  The returned ``applied`` bit tells the ABI builder that this
pass now owns output allocation, final assembly, and return emission.

This pass intentionally preserves the remaining legacy spelling contracts.
It filters result/workspace statements by generated variable names, rewrites a
hoisted workspace's ``.insert`` spelling, and renders phase bodies to C++ when
recovering the existing SpGEMM flop estimate.  Those dependencies are contained
here until structured access and work-estimate metadata replace them; this seam
does not broaden or redesign generated names.

The selected serial loop is replaced rather than rebuilt in place.  Matching
legacy behavior, only its init/condition/update survive; its tag, optional
parallel regions, hoisted declarations, unroll/SIMD flags, and previous OpenMP
properties do not transfer to the two newly configured phase loops.  Retained
prefix statements and nested work statements preserve all supported LLIR
fields through the common detached rewriter.  The complete transformation is a
single-use production operation and is not idempotent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, NoReturn, Optional, Sequence, Tuple, cast

from . import llir
from .codegen import LLIRLowerer
from .llir_traversal import (
    LLIRPath,
    LLIRRewriter,
    LLIRStatementSequence,
    LLIRStatementValue,
    LLIRTraversalContext,
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
)
from .llir_pass_manager import (
    PRODUCTION_LLIR_PASS_OPTIONS,
    LLIRPassManager,
    LLIRPassPartialFailure,
    LLIRPassRunRecord,
    LLIRRewriteArtifact,
    ResultWritePassSpec,
)
from .result_write_pass import ResultWriteContext

if TYPE_CHECKING:
    from .compile_options import CompileOptions

COMPRESSED_WHERE_OPENMP_TRAVERSAL_CONTEXT = LLIRTraversalContext(
    stage="LLIR transformation",
    pass_name="transform_compressed_where_for_openmp",
)


@dataclass(frozen=True)
class CompressedWhereOpenMPPolicy:
    """Explicit target-policy spellings used by the two phase loops."""

    omp_schedule: str = "dynamic, 64"
    flop_grain: str = "SCORCH_GRAIN_CODEGEN_SPGEMM"


COMPRESSED_WHERE_OPENMP_POLICY = CompressedWhereOpenMPPolicy()


@dataclass(frozen=True)
class CompressedWhereOpenMPContext:
    """All lowerer-independent inputs required by this transformation.

    ``compressed_levels`` structurally encodes the supported physical output
    pattern: it must be exactly ``(1,)``, ``(1, 2)``, and so on.  Level zero is
    therefore dense and every remaining level is compressed.

    ``workspace_ctype`` is also the legacy result-value C type used for exact
    output allocation.  Production obtains both workspace fields together from
    the one-dimensional sparse ``Where`` workspace. ``compile_options`` carries
    the exact outer snapshot through the two nested result-write passes and
    their spelling-only renderers; standalone pass callers may omit it.
    """

    result_name: str
    compressed_levels: Tuple[int, ...]
    workspace_name: str
    workspace_ctype: str
    policy: CompressedWhereOpenMPPolicy = COMPRESSED_WHERE_OPENMP_POLICY
    traversal: LLIRTraversalContext = COMPRESSED_WHERE_OPENMP_TRAVERSAL_CONTEXT
    compile_options: Optional["CompileOptions"] = None


@dataclass(frozen=True)
class CompressedWhereOpenMPResult:
    """Detached LLIR and whether this pass took ownership of final assembly."""

    statements: List[llir.Stmt]
    applied: bool


@dataclass(frozen=True)
class _ManagedCompressedWhereOpenMPExecution:
    """Exact result plus the two independent result-write run records."""

    result: CompressedWhereOpenMPResult
    nested_run_records: Tuple[LLIRPassRunRecord, ...]


@dataclass(frozen=True)
class _ParallelPolicyDecision:
    num_threads: Optional[str]
    chunk_expr: Optional[str]


def _diagnostic_context(context: object) -> LLIRTraversalContext:
    if type(context) is CompressedWhereOpenMPContext:
        traversal = cast(CompressedWhereOpenMPContext, context).traversal
        if type(traversal) is LLIRTraversalContext:
            return traversal
    return COMPRESSED_WHERE_OPENMP_TRAVERSAL_CONTEXT


def _raise_compressed_where_error(
    context: object,
    *,
    code: str,
    message: str,
    path: LLIRPath,
    value: object,
) -> NoReturn:
    traversal = _diagnostic_context(context)
    raise LLIRTraversalError(
        LLIRTraversalDiagnostic(
            code=code,
            message=message,
            path=path,
            node_type=type(value).__name__,
            stage=traversal.stage,
            pass_name=traversal.pass_name,
        )
    )


def _validate_non_empty_string(
    context: CompressedWhereOpenMPContext,
    value: object,
    *,
    code: str,
    field: str,
) -> None:
    if type(value) is not str or not value:
        _raise_compressed_where_error(
            context,
            code=code,
            message=f"{field} must be a non-empty string",
            path=("context", field),
            value=value,
        )


def _validate_context(context: object) -> CompressedWhereOpenMPContext:
    if type(context) is not CompressedWhereOpenMPContext:
        _raise_compressed_where_error(
            context,
            code="invalid_compressed_where_context",
            message="expected an immutable CompressedWhereOpenMPContext",
            path=("context",),
            value=context,
        )

    typed_context = cast(CompressedWhereOpenMPContext, context)
    traversal = typed_context.traversal
    if (
        type(traversal) is not LLIRTraversalContext
        or type(traversal.stage) is not str
        or not traversal.stage
        or type(traversal.pass_name) is not str
        or not traversal.pass_name
    ):
        _raise_compressed_where_error(
            COMPRESSED_WHERE_OPENMP_TRAVERSAL_CONTEXT,
            code="invalid_compressed_where_traversal_context",
            message="traversal stage and pass name must be non-empty strings",
            path=("context", "traversal"),
            value=traversal,
        )

    if typed_context.compile_options is not None:
        from .compile_options import CompileOptions

        if type(typed_context.compile_options) is not CompileOptions:
            _raise_compressed_where_error(
                typed_context,
                code="invalid_compressed_where_compile_options",
                message="compile_options must be an exact CompileOptions snapshot",
                path=("context", "compile_options"),
                value=typed_context.compile_options,
            )

    _validate_non_empty_string(
        typed_context,
        typed_context.result_name,
        code="invalid_compressed_where_result_name",
        field="result_name",
    )
    _validate_non_empty_string(
        typed_context,
        typed_context.workspace_name,
        code="invalid_compressed_where_workspace_name",
        field="workspace_name",
    )
    _validate_non_empty_string(
        typed_context,
        typed_context.workspace_ctype,
        code="invalid_compressed_where_workspace_ctype",
        field="workspace_ctype",
    )

    levels = typed_context.compressed_levels
    if type(levels) is not tuple or not levels:
        _raise_compressed_where_error(
            typed_context,
            code="invalid_compressed_where_levels",
            message="compressed_levels must be a non-empty immutable tuple",
            path=("context", "compressed_levels"),
            value=levels,
        )
    if any(type(level) is not int for level in levels):
        _raise_compressed_where_error(
            typed_context,
            code="invalid_compressed_where_levels",
            message="compressed levels must be exact integers",
            path=("context", "compressed_levels"),
            value=levels,
        )
    expected_levels = tuple(range(1, len(levels) + 1))
    if levels != expected_levels:
        _raise_compressed_where_error(
            typed_context,
            code="unsupported_compressed_where_layout",
            message=(
                "compressed levels must be contiguous after dense level zero; "
                f"expected {expected_levels}"
            ),
            path=("context", "compressed_levels"),
            value=levels,
        )

    policy = typed_context.policy
    if type(policy) is not CompressedWhereOpenMPPolicy:
        _raise_compressed_where_error(
            typed_context,
            code="invalid_compressed_where_policy",
            message="policy must be an immutable CompressedWhereOpenMPPolicy",
            path=("context", "policy"),
            value=policy,
        )
    _validate_non_empty_string(
        typed_context,
        policy.omp_schedule,
        code="invalid_compressed_where_schedule",
        field="policy.omp_schedule",
    )
    _validate_non_empty_string(
        typed_context,
        policy.flop_grain,
        code="invalid_compressed_where_flop_grain",
        field="policy.flop_grain",
    )
    return typed_context


def _validate_root(
    statements: object, context: CompressedWhereOpenMPContext
) -> List[llir.Stmt]:
    if type(statements) is not list:
        _raise_compressed_where_error(
            context,
            code="unsupported_compressed_where_root",
            message="the compressed Where transform requires a statement-list root",
            path=("root",),
            value=statements,
        )
    typed_statements = cast(List[llir.Stmt], statements)
    for index, statement in enumerate(typed_statements):
        if not isinstance(statement, llir.Stmt):
            _raise_compressed_where_error(
                context,
                code="invalid_compressed_where_root_member",
                message="the top-level list may contain only LLIR statements",
                path=("root", f"[{index}]"),
                value=statement,
            )
    LLIRWalker(context.traversal).walk(cast(LLIRValue, typed_statements))
    return typed_statements


class _WorkspaceInsertRewriter(LLIRRewriter):
    """Detach LLIR while matching the legacy workspace-rewrite recursion."""

    def __init__(self, context: CompressedWhereOpenMPContext) -> None:
        super().__init__(context.traversal)
        self._old = f"{context.workspace_name}.insert"
        self._new = f"{context.workspace_name}.insert_unchecked"
        self._identity = LLIRRewriter(context.traversal)

    def _rewrite_legacy_expr(self, expression: llir.Expr) -> llir.Expr:
        rewritten = self._identity.rewrite(expression)
        if isinstance(rewritten, llir.Var):
            if self._old in rewritten.name:
                rewritten.name = rewritten.name.replace(self._old, self._new)
            return rewritten
        if isinstance(rewritten, llir.BinOp):
            rewritten.left = self._rewrite_legacy_expr(rewritten.left)
            rewritten.right = self._rewrite_legacy_expr(rewritten.right)
            return rewritten
        if isinstance(rewritten, llir.ArrayAccess):
            rewritten.array = self._rewrite_legacy_expr(rewritten.array)
            rewritten.index = self._rewrite_legacy_expr(rewritten.index)
        return rewritten

    def _rewrite_for_loop(self, node: llir.ForLoop) -> llir.ForLoop:
        rewritten = cast(llir.ForLoop, self._identity.rewrite(node))
        rewritten.body = cast(
            List[llir.Stmt],
            self.rewrite_statement_sequence(
                cast(LLIRStatementSequence, node.body), ("workspace", "body")
            ),
        )
        return rewritten

    def _rewrite_if_then_else(self, node: llir.IfThenElse) -> llir.IfThenElse:
        rewritten = cast(llir.IfThenElse, self._identity.rewrite(node))
        if node.then_body is not None:
            rewritten.then_body = cast(
                List[llir.Stmt],
                self.rewrite_statement_sequence(
                    cast(LLIRStatementSequence, node.then_body),
                    ("workspace", "then_body"),
                ),
            )
        if node.else_body is not None:
            rewritten.else_body = cast(
                List[llir.Stmt],
                self.rewrite_statement_sequence(
                    cast(LLIRStatementSequence, node.else_body),
                    ("workspace", "else_body"),
                ),
            )
        if node.then_body_list is not None:
            rewritten.then_body_list = [
                cast(
                    List[llir.Stmt],
                    self.rewrite_statement_sequence(
                        cast(LLIRStatementSequence, body),
                        ("workspace", f"then_body_list[{index}]"),
                    ),
                )
                for index, body in enumerate(node.then_body_list)
            ]
        return rewritten

    def _rewrite_legacy_statement(self, node: llir.Stmt) -> llir.Stmt:
        if type(node) is llir.Assign:
            assign = cast(llir.Assign, node)
            rewritten_assign = cast(llir.Assign, self._identity.rewrite(assign))
            rewritten_assign.var = self._rewrite_legacy_expr(assign.var)
            rewritten_assign.value = self._rewrite_legacy_expr(assign.value)
            return rewritten_assign
        if type(node) is llir.VarInit:
            var_init = cast(llir.VarInit, node)
            rewritten_init = cast(llir.VarInit, self._identity.rewrite(var_init))
            rewritten_init.value = self._rewrite_legacy_expr(var_init.value)
            return rewritten_init
        if type(node) is llir.FunctionCallStmt:
            call = cast(llir.FunctionCallStmt, node)
            rewritten_call = cast(llir.FunctionCallStmt, self._identity.rewrite(call))
            rewritten_call.name = call.name.replace(self._old, self._new)
            rewritten_call.args = [self._rewrite_legacy_expr(arg) for arg in call.args]
            return rewritten_call
        if type(node) is llir.ForLoop:
            return self._rewrite_for_loop(cast(llir.ForLoop, node))
        if type(node) is llir.IfThenElse:
            return self._rewrite_if_then_else(cast(llir.IfThenElse, node))
        if type(node) is llir.RawStmt:
            raw = cast(llir.RawStmt, node)
            rewritten_raw = cast(llir.RawStmt, self._identity.rewrite(raw))
            rewritten_raw.code = raw.code.replace(self._old, self._new)
            return rewritten_raw
        return cast(llir.Stmt, self._identity.rewrite(node))

    def rewrite_statement_sequence(
        self, statements: LLIRStatementSequence, path: LLIRPath
    ) -> LLIRStatementSequence:
        rewritten: List[LLIRStatementValue] = []
        for statement in statements:
            if isinstance(statement, llir.Stmt):
                rewritten.append(self._rewrite_legacy_statement(statement))
            else:
                # The legacy helper did not descend through arbitrary nested
                # containers.  Detach them without broadening rewrite scope.
                if type(statement) is list:
                    rewritten.append(
                        self._identity.rewrite(
                            cast(List[LLIRStatementValue], statement)
                        )
                    )
                else:
                    rewritten.append(
                        self._identity.rewrite(
                            cast(Tuple[LLIRStatementValue, ...], statement)
                        )
                    )
        if type(statements) is tuple:
            return tuple(rewritten)
        return rewritten


def _is_openmp_compatible_for_loop(for_loop: llir.ForLoop) -> bool:
    if type(for_loop.init) is not llir.VarInit:
        return False
    init = cast(llir.VarInit, for_loop.init)
    if type(init.var) is not llir.Var:
        return False
    loop_var = init.var

    if type(for_loop.update) is llir.Increment:
        increment = cast(llir.Increment, for_loop.update)
        if increment.var.name != loop_var.name:
            return False
    elif type(for_loop.update) is llir.Assign:
        update = cast(llir.Assign, for_loop.update)
        if type(update.var) is not llir.Var:
            return False
        if update.var.name != loop_var.name:
            return False
        if update.op not in (
            llir.AssignOp.ADD_ASSIGN,
            llir.AssignOp.SUB_ASSIGN,
        ):
            return False
    else:
        return False

    if type(for_loop.cond) is not llir.BinOp:
        return False
    condition = cast(llir.BinOp, for_loop.cond)
    if condition.op not in ("<", "<=", ">", ">="):
        return False
    if type(condition.left) is not llir.Var:
        return False
    return condition.left.name == loop_var.name


def _find_outer_loop(
    statements: List[llir.Stmt],
) -> Tuple[Optional[int], Optional[llir.ForLoop]]:
    for index, statement in enumerate(statements):
        if type(statement) is llir.ForLoop and _is_openmp_compatible_for_loop(
            statement
        ):
            return index, statement
    return None, None


def _extract_loop_bound(for_loop: llir.ForLoop) -> Optional[str]:
    if type(for_loop.cond) is llir.BinOp and for_loop.cond.op == "<":
        right = for_loop.cond.right
        if type(right) is llir.Var:
            return right.name
    return None


def _should_drop_work_statement(
    statement: llir.Stmt,
    *,
    result_name: str,
    first_compressed_level: int,
    workspace_name: str,
) -> Tuple[bool, bool]:
    if type(statement) is llir.ForLoop:
        condition = statement.cond
        pos_index_name = f"{result_name}{first_compressed_level}_pos_index"
        if (
            type(condition) is llir.BinOp
            and type(condition.left) is llir.Var
            and pos_index_name in condition.left.name
        ):
            return True, False
        init = statement.init
        if type(init) is llir.VarInit and init.var.name.startswith(f"p{result_name}"):
            return True, False
    elif type(statement) is llir.VarInit:
        if statement.var.name == workspace_name:
            return True, True
    elif type(statement) is llir.Assign:
        target_name = getattr(statement.var, "name", "")
        if f"{result_name}{first_compressed_level}_pos[" in target_name:
            return True, False
    return False, False


def _extract_work_body(
    for_loop: llir.ForLoop, context: CompressedWhereOpenMPContext
) -> Tuple[List[llir.Stmt], bool]:
    work_body: List[llir.Stmt] = []
    workspace_hoisted = False
    first_level = context.compressed_levels[0]
    for statement in for_loop.body:
        drop, found_workspace = _should_drop_work_statement(
            statement,
            result_name=context.result_name,
            first_compressed_level=first_level,
            workspace_name=context.workspace_name,
        )
        workspace_hoisted = workspace_hoisted or found_workspace
        if not drop:
            work_body.append(statement)

    if workspace_hoisted:
        rewritten = _WorkspaceInsertRewriter(context).rewrite(work_body)
        return cast(List[llir.Stmt], rewritten), True
    return work_body, False


def _workspace_view_statement(context: CompressedWhereOpenMPContext) -> llir.RawStmt:
    workspace_name = context.workspace_name
    return llir.RawStmt(
        code=(
            f"auto {workspace_name} = {workspace_name}_pool["
            "(size_t)omp_get_thread_num()].make_view()"
        )
    )


def _phase_header_copy(
    source: llir.ForLoop, context: CompressedWhereOpenMPContext
) -> llir.ForLoop:
    return cast(llir.ForLoop, LLIRRewriter(context.traversal).rewrite(source))


def _find_sparse_pos_array(body: Sequence[llir.Stmt]) -> Optional[str]:
    for statement in body:
        if type(statement) is llir.VarInit:
            code = statement.var.name + " " + str(getattr(statement.value, "name", ""))
            match = re.search(r"(\w+_pos)\[", code)
            if match:
                return match.group(1)
        if type(statement) in (llir.ForLoop, llir.WhileLoop):
            loop = cast(Sequence[llir.Stmt], getattr(statement, "body"))
            result = _find_sparse_pos_array(loop)
            if result:
                return result
        if type(statement) is llir.RawStmt:
            match = re.search(r"(\w+_pos)\[", statement.code)
            if match:
                return match.group(1)
    return None


def _sparse_pos_work_expr(
    sparse_pos: Optional[str], loop_bound: Optional[str]
) -> Optional[str]:
    if sparse_pos is None or loop_bound is None:
        return None
    match = re.match(r"([A-Za-z_]\w*?)(\d+)_pos$", sparse_pos)
    if match is None:
        return None
    operand, level_text = match.groups()
    level = int(level_text)
    if level == 0 or loop_bound != f"{operand}{level - 1}_size":
        return None
    return f"{sparse_pos}[{loop_bound}]"


def _expr_to_str(expression: llir.Expr) -> str:
    if type(expression) is llir.Var:
        return expression.name
    if type(expression) is llir.Literal:
        return str(expression.value)
    if isinstance(expression, llir.BinOp):
        return (
            f"({_expr_to_str(expression.left)} {expression.op} "
            f"{_expr_to_str(expression.right)})"
        )
    return str(expression)


def _parallel_rows_expr(for_loop: llir.ForLoop, bound: str) -> str:
    update = for_loop.update
    if type(update) is llir.Assign and update.op == llir.AssignOp.ADD_ASSIGN:
        step = _expr_to_str(update.value)
        return f"(({bound} + {step} - 1) / {step})"
    return bound


def _find_all_sparse_pos_arrays(
    body: Sequence[llir.Stmt],
    compile_options: Optional["CompileOptions"],
) -> List[str]:
    """Recover legacy sparse-array spellings from rendered phase C++."""

    try:
        text = LLIRLowerer(compile_options=compile_options).lower_llir(list(body))
    except Exception:
        # This is the characterized legacy policy fallback.  The input LLIR has
        # already passed exact-type traversal validation; rendering failure only
        # disables the richer work estimate.
        return []
    found: List[str] = []
    for match in re.finditer(r"(\w+_pos)\[", text):
        if match.group(1) not in found:
            found.append(match.group(1))
    return found


def _parse_pos(pos_name: str) -> Optional[Tuple[str, int]]:
    match = re.match(r"([A-Za-z_]\w*?)(\d+)_pos$", pos_name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _spgemm_flop_work_expr(
    body: Sequence[llir.Stmt],
    bound: str,
    compile_options: Optional["CompileOptions"],
) -> Optional[str]:
    levels: Dict[str, set[int]] = {}
    for position_name in _find_all_sparse_pos_arrays(body, compile_options):
        parsed = _parse_pos(position_name)
        if parsed is not None:
            levels.setdefault(parsed[0], set()).add(parsed[1])

    if not bound.endswith("0_size"):
        return None
    left_prefix = bound[: -len("0_size")]
    if left_prefix not in levels or 0 in levels[left_prefix]:
        return None
    right_prefix = next(
        (
            prefix
            for prefix, prefix_levels in levels.items()
            if prefix != left_prefix and 0 not in prefix_levels
        ),
        None,
    )
    if right_prefix is None:
        return None

    left_nnz = f"(long){left_prefix}{max(levels[left_prefix])}_pos[{bound}]"
    right_outer = f"{right_prefix}0_size"
    right_nnz = f"{right_prefix}{max(levels[right_prefix])}_pos[{right_outer}]"
    return (
        f"{left_nnz} * ({right_outer} > 0 ? " f"({right_nnz} / {right_outer}) + 1 : 1)"
    )


def _parallel_policy_decision(
    loop: llir.ForLoop,
    body: Sequence[llir.Stmt],
    *,
    work_expr: Optional[str],
    grain: Optional[str],
) -> _ParallelPolicyDecision:
    bound = _extract_loop_bound(loop)
    if bound is None:
        return _ParallelPolicyDecision(None, None)
    rows = _parallel_rows_expr(loop, bound)
    if work_expr is None:
        sparse_pos = _find_sparse_pos_array(body)
        work = _sparse_pos_work_expr(sparse_pos, bound) or "-1"
    else:
        work = work_expr
    grain_suffix = f", {grain}" if grain is not None else ""
    return _ParallelPolicyDecision(
        num_threads=f"scorch_nthreads({work}, {rows}{grain_suffix})",
        chunk_expr=f"scorch_chunk({rows}, {work}{grain_suffix})",
    )


def _build_phase_loop(
    source: llir.ForLoop,
    body: List[llir.Stmt],
    context: CompressedWhereOpenMPContext,
    *,
    workspace_hoisted: bool,
    loop_bound: str,
) -> llir.ForLoop:
    header = _phase_header_copy(source, context)
    flop_work = _spgemm_flop_work_expr(
        body,
        loop_bound,
        context.compile_options,
    )
    decision = _parallel_policy_decision(
        header,
        body,
        work_expr=flop_work,
        grain=context.policy.flop_grain if flop_work else None,
    )
    pre_parallel_body: Optional[List[llir.Stmt]] = (
        [_workspace_view_statement(context)] if workspace_hoisted else None
    )
    return llir.ForLoop(
        init=header.init,
        cond=header.cond,
        update=header.update,
        body=body,
        omp_parallel_for=True,
        omp_schedule=context.policy.omp_schedule,
        pre_parallel_body=pre_parallel_body,
        omp_num_threads=decision.num_threads,
        omp_chunk_expr=decision.chunk_expr,
    )


def _build_count_body(
    work_body: List[llir.Stmt],
    context: CompressedWhereOpenMPContext,
    *,
    loop_var: str,
    workspace_hoisted: bool,
    manager: LLIRPassManager,
) -> Tuple[List[llir.Stmt], Tuple[LLIRPassRunRecord, ...]]:
    levels = context.compressed_levels
    body: List[llir.Stmt] = [
        llir.RawStmt(code=f"int _cnt{level} = 0") for level in levels
    ]
    body.extend(llir.RawStmt(code=f"int _prev{level} = 0") for level in levels[1:])
    result_write = manager.run_result_write(
        LLIRRewriteArtifact(work_body),
        ResultWritePassSpec(
            ResultWriteContext(
                result_name=context.result_name,
                compressed_levels=levels,
                mode="count",
                traversal=context.traversal,
                compile_options=context.compile_options,
            )
        ),
    )
    body.extend(cast(List[llir.Stmt], result_write.artifact.value))
    body.extend(
        llir.RawStmt(code=f"_count{level}[{loop_var}] = _cnt{level}")
        for level in levels
    )
    if workspace_hoisted:
        body.append(llir.RawStmt(code=f"{context.workspace_name}.clear()"))
    return body, result_write.run_records


def _build_fill_body(
    work_body: List[llir.Stmt],
    context: CompressedWhereOpenMPContext,
    *,
    loop_var: str,
    workspace_hoisted: bool,
    manager: LLIRPassManager,
) -> Tuple[List[llir.Stmt], Tuple[LLIRPassRunRecord, ...]]:
    levels = context.compressed_levels
    body: List[llir.Stmt] = []
    for level in levels:
        body.extend(
            [
                llir.RawStmt(code=f"int64_t _base{level} = _offset{level}[{loop_var}]"),
                llir.RawStmt(code=f"int _pos{level} = 0"),
            ]
        )
    body.extend(llir.RawStmt(code=f"int _prev{level} = 0") for level in levels[1:])
    result_write = manager.run_result_write(
        LLIRRewriteArtifact(work_body),
        ResultWritePassSpec(
            ResultWriteContext(
                result_name=context.result_name,
                compressed_levels=levels,
                mode="fill",
                traversal=context.traversal,
                compile_options=context.compile_options,
            )
        ),
    )
    body.extend(cast(List[llir.Stmt], result_write.artifact.value))
    if workspace_hoisted:
        body.append(llir.RawStmt(code=f"{context.workspace_name}.clear()"))
    return body, result_write.run_records


def _should_drop_prefix_statement(
    statement: llir.Stmt, context: CompressedWhereOpenMPContext
) -> bool:
    result_name = context.result_name
    levels = context.compressed_levels
    if type(statement) is llir.VarDecl:
        name = statement.var.name
        if name.startswith(f"{result_name}_values"):
            return True
        return any(
            name.startswith(f"{result_name}{level}_pos")
            or name.startswith(f"{result_name}{level}_crd")
            for level in levels
        )
    if type(statement) is llir.VarInit:
        name = statement.var.name
        return any(
            name in (f"p{result_name}{level}", f"{result_name}{level}_pos_index")
            for level in levels
        )
    if type(statement) is llir.Assign:
        name = getattr(statement.var, "name", "")
        return any(f"{result_name}{level}_pos[" in name for level in levels)
    if type(statement) is llir.ForLoop:
        init = statement.init
        return bool(
            type(init) is llir.VarInit and init.var.name.startswith(f"p{result_name}")
        )
    return False


def _filtered_prefix(
    statements: List[llir.Stmt],
    stop: int,
    context: CompressedWhereOpenMPContext,
) -> List[llir.Stmt]:
    return [
        statement
        for statement in statements[:stop]
        if not _should_drop_prefix_statement(statement, context)
    ]


def _workspace_pool_statement(
    context: CompressedWhereOpenMPContext,
    count_loop: llir.ForLoop,
    fill_loop: llir.ForLoop,
) -> llir.RawStmt:
    count_threads = count_loop.omp_num_threads or "omp_get_max_threads()"
    fill_threads = fill_loop.omp_num_threads or "omp_get_max_threads()"
    thread_count_expr = (
        count_threads
        if count_threads == fill_threads
        else f"std::max({count_threads}, {fill_threads})"
    )
    workspace = context.workspace_name
    ctype = context.workspace_ctype
    first_level = context.compressed_levels[0]
    return llir.RawStmt(
        code=(
            f"int {workspace}_thread_count = {thread_count_expr};\n"
            f"std::vector<linked_list_workspace_1d<{ctype}>> {workspace}_pool;\n"
            f"{workspace}_pool.reserve((size_t){workspace}_thread_count);\n"
            f"for (int _worker = 0; _worker < {workspace}_thread_count; "
            "_worker++) {\n"
            f"  {workspace}_pool.emplace_back(result_shape[{first_level}], true);\n"
            "}"
        ),
        add_semicolon=False,
    )


def _count_and_offset_statements(
    context: CompressedWhereOpenMPContext, loop_bound: str
) -> List[llir.Stmt]:
    statements: List[llir.Stmt] = []
    for level in context.compressed_levels:
        statements.append(
            llir.RawStmt(
                code=f"std::vector<int> _count{level}((size_t){loop_bound}, 0)"
            )
        )
    return statements


def _prefix_sum_statements(
    context: CompressedWhereOpenMPContext, loop_bound: str
) -> List[llir.Stmt]:
    statements: List[llir.Stmt] = []
    for level in context.compressed_levels:
        statements.extend(
            [
                llir.RawStmt(
                    code=(
                        f"std::vector<int64_t> _offset{level}("
                        f"(size_t){loop_bound} + 1)"
                    )
                ),
                llir.RawStmt(code=f"_offset{level}[0] = 0"),
                llir.RawStmt(
                    code=(
                        f"for (int _i = 0; _i < {loop_bound}; _i++) "
                        f"_offset{level}[_i + 1] = _offset{level}[_i] + "
                        f"_count{level}[_i];"
                    ),
                    add_semicolon=False,
                ),
            ]
        )
    statements.extend(
        llir.RawStmt(code=f"int64_t _total{level} = _offset{level}[{loop_bound}]")
        for level in context.compressed_levels
    )
    return statements


def _position_and_coordinate_allocations(
    context: CompressedWhereOpenMPContext, loop_bound: str
) -> List[llir.Stmt]:
    result_name = context.result_name
    levels = context.compressed_levels
    first_level = levels[0]
    statements: List[llir.Stmt] = [
        llir.RawStmt(
            code=(
                f"torch::Tensor {result_name}{first_level}_pos_torch = "
                f"torch::empty({{(long long)({loop_bound} + 1)}}, torch::kInt);\n"
                f"  int* {result_name}{first_level}_pos_data = "
                f"{result_name}{first_level}_pos_torch.data_ptr<int>();"
            ),
            add_semicolon=False,
        ),
        llir.RawStmt(
            code=(
                f"for (int _i = 0; _i <= {loop_bound}; _i++) "
                f"{result_name}{first_level}_pos_data[_i] = "
                f"(int)_offset{first_level}[_i];"
            ),
            add_semicolon=False,
        ),
    ]
    for level in levels:
        statements.append(
            llir.RawStmt(
                code=(
                    f"torch::Tensor {result_name}{level}_crd_torch = "
                    f"torch::empty({{(long long)_total{level}}}, torch::kInt);\n"
                    f"  int* {result_name}{level}_crd_data = "
                    f"{result_name}{level}_crd_torch.data_ptr<int>();"
                ),
                add_semicolon=False,
            )
        )
    for parent_level, level in zip(levels, levels[1:]):
        statements.extend(
            [
                llir.RawStmt(
                    code=(
                        f"torch::Tensor {result_name}{level}_pos_torch = "
                        f"torch::empty({{(long long)(_total{parent_level} + 1)}}, "
                        "torch::kInt);\n"
                        f"  int* {result_name}{level}_pos_data = "
                        f"{result_name}{level}_pos_torch.data_ptr<int>();"
                    ),
                    add_semicolon=False,
                ),
                llir.RawStmt(code=f"{result_name}{level}_pos_data[0] = 0"),
            ]
        )
    return statements


_CTYPE_TO_TORCH: Dict[str, str] = {
    "float": "torch::kFloat32",
    "double": "torch::kFloat64",
    "int": "torch::kInt32",
    "int32_t": "torch::kInt32",
    "long long": "torch::kInt64",
    "int64_t": "torch::kInt64",
}


def _value_allocation(context: CompressedWhereOpenMPContext) -> llir.RawStmt:
    result_name = context.result_name
    leaf = context.compressed_levels[-1]
    ctype = context.workspace_ctype
    torch_dtype = _CTYPE_TO_TORCH.get(ctype, "torch::kFloat32")
    return llir.RawStmt(
        code=(
            f"torch::Tensor {result_name}_values_torch = "
            f"torch::empty({{(long long)_total{leaf}}}, {torch_dtype});\n"
            f"  {ctype}* {result_name}_values_data = "
            f"{result_name}_values_torch.data_ptr<{ctype}>();"
        ),
        add_semicolon=False,
    )


def _final_assembly(context: CompressedWhereOpenMPContext) -> llir.RawStmt:
    result_name = context.result_name
    rank = context.compressed_levels[-1] + 1
    mode_indices = ["{}"]
    mode_indices.extend(
        f"{{{result_name}{level}_pos_torch, {result_name}{level}_crd_torch}}"
        for level in range(1, rank)
    )
    return llir.RawStmt(
        code=(
            f"Tensor {result_name};\n"
            f"  {result_name}.storage.index.mode_indices = "
            f"{{{', '.join(mode_indices)}}};\n"
            f"  {result_name}.storage.value = {result_name}_values_torch;\n"
            f"  return {result_name};"
        ),
        add_semicolon=False,
    )


def _build_transformed_statements(
    statements: List[llir.Stmt],
    loop_index: int,
    for_loop: llir.ForLoop,
    loop_bound: str,
    context: CompressedWhereOpenMPContext,
    manager: LLIRPassManager,
) -> Tuple[List[llir.Stmt], Tuple[LLIRPassRunRecord, ...]]:
    loop_var = cast(llir.VarInit, for_loop.init).var.name
    work_body, workspace_hoisted = _extract_work_body(for_loop, context)
    count_body, count_records = _build_count_body(
        work_body,
        context,
        loop_var=loop_var,
        workspace_hoisted=workspace_hoisted,
        manager=manager,
    )
    try:
        fill_body, fill_records = _build_fill_body(
            work_body,
            context,
            loop_var=loop_var,
            workspace_hoisted=workspace_hoisted,
            manager=manager,
        )
    except Exception as failure:
        raise LLIRPassPartialFailure(failure, count_records) from failure

    nested_run_records = (*count_records, *fill_records)
    try:
        count_loop = _build_phase_loop(
            for_loop,
            count_body,
            context,
            workspace_hoisted=workspace_hoisted,
            loop_bound=loop_bound,
        )
        fill_loop = _build_phase_loop(
            for_loop,
            fill_body,
            context,
            workspace_hoisted=workspace_hoisted,
            loop_bound=loop_bound,
        )

        result = _filtered_prefix(statements, loop_index, context)
        if workspace_hoisted:
            result.append(_workspace_pool_statement(context, count_loop, fill_loop))
        result.extend(_count_and_offset_statements(context, loop_bound))
        result.append(count_loop)
        result.extend(_prefix_sum_statements(context, loop_bound))
        result.extend(_position_and_coordinate_allocations(context, loop_bound))
        result.append(_value_allocation(context))
        result.append(fill_loop)
        result.append(_final_assembly(context))
    except Exception as failure:
        raise LLIRPassPartialFailure(failure, nested_run_records) from failure
    return result, nested_run_records


def _transform_compressed_where_for_openmp_managed(
    statements: List[llir.Stmt],
    context: CompressedWhereOpenMPContext,
    manager: LLIRPassManager,
) -> _ManagedCompressedWhereOpenMPExecution:
    """Run the transform and retain managed internal composition records.

    Legal no-ops are an input with no compatible top-level plain ``ForLoop`` or
    one whose first compatible loop has no extractable ``< bound_var`` bound.
    Both return structurally identical, fully detached LLIR and ``applied=False``.

    The root must be an exact statement list and the context must describe a
    ``d,s[,s...]`` result.  Unknown LLIR nodes, malformed typed children, other
    roots, and invalid/unsupported contexts fail through this pass's structured
    :class:`LLIRTraversalError` diagnostic.

    Reapplying the complete pass is outside the production contract.  Matching
    the characterized legacy behavior, it is not detected as an idempotent
    no-op: the first generated count loop is itself a compatible loop.
    """

    checked_context = _validate_context(context)
    checked_statements = _validate_root(statements, checked_context)
    detached = cast(
        List[llir.Stmt],
        LLIRRewriter(checked_context.traversal).rewrite(checked_statements),
    )
    loop_index, for_loop = _find_outer_loop(detached)
    if loop_index is None or for_loop is None:
        return _ManagedCompressedWhereOpenMPExecution(
            CompressedWhereOpenMPResult(detached, False),
            (),
        )
    loop_bound = _extract_loop_bound(for_loop)
    if loop_bound is None:
        return _ManagedCompressedWhereOpenMPExecution(
            CompressedWhereOpenMPResult(detached, False),
            (),
        )

    transformed, nested_run_records = _build_transformed_statements(
        detached,
        loop_index,
        for_loop,
        loop_bound,
        checked_context,
        manager,
    )
    try:
        LLIRWalker(checked_context.traversal).walk(cast(LLIRValue, transformed))
    except Exception as failure:
        raise LLIRPassPartialFailure(failure, nested_run_records) from failure
    return _ManagedCompressedWhereOpenMPExecution(
        CompressedWhereOpenMPResult(transformed, True),
        nested_run_records,
    )


def transform_compressed_where_for_openmp(
    statements: List[llir.Stmt],
    context: CompressedWhereOpenMPContext,
) -> CompressedWhereOpenMPResult:
    """Return the exact detached compressed-Where semantic result.

    This public typed pass keeps its proven return type.  Its internal count and
    fill result-write calls route through the production runner configuration;
    manager-only run information is retained only by the managed entry point.
    """

    try:
        return _transform_compressed_where_for_openmp_managed(
            statements,
            context,
            LLIRPassManager(PRODUCTION_LLIR_PASS_OPTIONS),
        ).result
    except LLIRPassPartialFailure as failure:
        raise failure.failure from None
