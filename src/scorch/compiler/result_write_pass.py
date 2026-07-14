"""Typed rewriting of legacy compressed-result writes.

This pass preserves the current compressed-output/OpenMP transformation while
moving its LLIR rewrite behind the common detached ownership boundary.  The
legacy LLIR still encodes result accesses and vector operations in ``Var.name``
and ``FunctionCallStmt.name`` strings, and fill mode must render expressions
back to C++ before placing them in ``RawStmt`` nodes.  Those spelling
dependencies are intentionally contained here until structured access nodes
replace them; this extraction does not broaden that migration seam.

Count and fill are independent transformations.  Production applies each mode
once to the same original work body.  Applying one mode to the output of the
other is unsupported because the generated ``RawStmt`` nodes are opaque.
Special position-boundary conditionals are a generated-shape contract for
compressed levels that have a preceding compressed parent; the surrounding
OpenMP transform does not declare ``_prev`` for the first compressed level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    List,
    Literal,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from . import llir
from .codegen import LLIRLowerer
from .diagnostics import CodegenError
from .llir_traversal import (
    LLIRPath,
    LLIRRewriteValueT,
    LLIRRewriter,
    LLIRStatementSequence,
    LLIRTraversalContext,
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
)

if TYPE_CHECKING:
    from .compile_options import CompileOptions

ResultWriteMode = Literal["count", "fill"]

RESULT_WRITE_TRAVERSAL_CONTEXT = LLIRTraversalContext(
    stage="LLIR rewrite",
    pass_name="rewrite_result_writes",
)


@dataclass(frozen=True)
class ResultWriteContext:
    """All explicit state required to rewrite one result's writes.

    Production carries the outer compilation snapshot so its spelling-only
    renderer cannot fall back to an independent emission policy. Direct pass
    callers may omit it because they are their own compatibility boundary.
    """

    result_name: str
    compressed_levels: Tuple[int, ...]
    mode: ResultWriteMode
    traversal: LLIRTraversalContext = RESULT_WRITE_TRAVERSAL_CONTEXT
    compile_options: Optional["CompileOptions"] = None


def _diagnostic_context(context: object) -> LLIRTraversalContext:
    if type(context) is ResultWriteContext:
        traversal = cast(ResultWriteContext, context).traversal
        if type(traversal) is LLIRTraversalContext:
            return traversal
    return RESULT_WRITE_TRAVERSAL_CONTEXT


def _raise_result_write_error(
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


def _validate_context(context: object) -> ResultWriteContext:
    if type(context) is not ResultWriteContext:
        _raise_result_write_error(
            context,
            code="invalid_result_write_context",
            message="expected an immutable ResultWriteContext",
            path=("context",),
            value=context,
        )

    typed_context = cast(ResultWriteContext, context)
    traversal = typed_context.traversal
    if (
        type(traversal) is not LLIRTraversalContext
        or type(traversal.stage) is not str
        or not traversal.stage
        or type(traversal.pass_name) is not str
        or not traversal.pass_name
    ):
        _raise_result_write_error(
            RESULT_WRITE_TRAVERSAL_CONTEXT,
            code="invalid_result_write_traversal_context",
            message="traversal stage and pass name must be non-empty strings",
            path=("context", "traversal"),
            value=traversal,
        )

    if typed_context.compile_options is not None:
        from .compile_options import CompileOptions

        if type(typed_context.compile_options) is not CompileOptions:
            _raise_result_write_error(
                typed_context,
                code="invalid_result_write_compile_options",
                message="compile_options must be an exact CompileOptions snapshot",
                path=("context", "compile_options"),
                value=typed_context.compile_options,
            )

    if type(typed_context.result_name) is not str or not typed_context.result_name:
        _raise_result_write_error(
            typed_context,
            code="invalid_result_write_name",
            message="result_name must be a non-empty string",
            path=("context", "result_name"),
            value=typed_context.result_name,
        )

    levels = typed_context.compressed_levels
    if type(levels) is not tuple or not levels:
        _raise_result_write_error(
            typed_context,
            code="invalid_compressed_levels",
            message="compressed_levels must be a non-empty immutable tuple",
            path=("context", "compressed_levels"),
            value=levels,
        )
    if any(type(level) is not int or level < 0 for level in levels):
        _raise_result_write_error(
            typed_context,
            code="invalid_compressed_levels",
            message="compressed levels must be non-negative exact integers",
            path=("context", "compressed_levels"),
            value=levels,
        )
    if any(current >= following for current, following in zip(levels, levels[1:])):
        _raise_result_write_error(
            typed_context,
            code="invalid_compressed_levels",
            message="compressed levels must be strictly increasing and unique",
            path=("context", "compressed_levels"),
            value=levels,
        )

    if type(typed_context.mode) is not str or typed_context.mode not in (
        "count",
        "fill",
    ):
        _raise_result_write_error(
            typed_context,
            code="invalid_result_write_mode",
            message="mode must be exactly 'count' or 'fill'",
            path=("context", "mode"),
            value=typed_context.mode,
        )
    return typed_context


class _ResultWriteRewriter(LLIRRewriter):
    """Rewrite result assembly statements while preserving legacy regions."""

    _IDENTITY_ONLY_REGIONS = frozenset(
        {
            "before_parallel_body",
            "_hoisted_ptr_decls",
        }
    )

    def __init__(self, context: ResultWriteContext) -> None:
        super().__init__(context.traversal)
        self._pass_context = context
        self._result_name = context.result_name
        self._compressed_levels = context.compressed_levels
        self._leaf = context.compressed_levels[-1]
        self._mode = context.mode
        self._codegen = LLIRLowerer(compile_options=context.compile_options)
        self._identity = LLIRRewriter(context.traversal)

    def rewrite_statement_sequence(
        self, statements: LLIRStatementSequence, path: LLIRPath
    ) -> LLIRStatementSequence:
        # The legacy transform never descended into these loop-owned regions.
        # They still pass through an identity rewrite so the output is detached.
        if path and path[-1] in self._IDENTITY_ONLY_REGIONS:
            return self._identity.rewrite_statement_sequence(statements, path)
        return super().rewrite_statement_sequence(statements, path)

    def rewrite_statement_sequence_member(
        self, node: llir.Stmt, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        node_type = type(node)
        if node_type is llir.Assign:
            return self._rewrite_assign_statement(cast(llir.Assign, node), path)
        if node_type is llir.Increment:
            return self._rewrite_increment_statement(cast(llir.Increment, node), path)
        if node_type is llir.FunctionCallStmt:
            return self._rewrite_call_statement(cast(llir.FunctionCallStmt, node), path)
        if node_type is llir.VarInit:
            return self._rewrite_var_init_statement(cast(llir.VarInit, node), path)
        if node_type is llir.IfThenElse:
            return self._rewrite_if_statement(cast(llir.IfThenElse, node), path)
        return super().rewrite_statement_sequence_member(node, path)

    def _render_expression(self, expression: llir.Expr, path: LLIRPath) -> str:
        try:
            return self._codegen.lower_llir(expression)
        except CodegenError as error:
            _raise_result_write_error(
                self._pass_context,
                code="result_write_expression_render_failed",
                message=str(error),
                path=path,
                value=expression,
            )

    def _rewrite_assign_statement(
        self, node: llir.Assign, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        target_name = getattr(node.var, "name", "")

        if f"{self._result_name}_values[" in target_name:
            if self._mode == "count":
                return ()
            value = self._render_expression(node.value, path + ("value",))
            return (
                llir.RawStmt(
                    code=(
                        f"{self._result_name}_values_data"
                        f"[_base{self._leaf} + _pos{self._leaf}] = {value}"
                    )
                ),
            )

        for level in self._compressed_levels:
            if f"{self._result_name}{level}_crd[" in target_name:
                if self._mode == "count":
                    return (llir.RawStmt(code=f"_cnt{level}++"),)
                value = self._render_expression(node.value, path + ("value",))
                return (
                    llir.RawStmt(
                        code=(
                            f"{self._result_name}{level}_crd_data"
                            f"[_base{level} + _pos{level}] = {value}"
                        )
                    ),
                )

        if any(
            f"{self._result_name}{level}_pos[" in target_name
            for level in self._compressed_levels
        ):
            return ()
        return (node,)

    def _rewrite_increment_statement(
        self, node: llir.Increment, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        for level in self._compressed_levels:
            if node.var.name == f"p{self._result_name}{level}":
                if self._mode == "fill":
                    return (llir.RawStmt(code=f"_pos{level}++"),)
                return ()
        return (node,)

    def _rewrite_call_statement(
        self, node: llir.FunctionCallStmt, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        for index, level in enumerate(self._compressed_levels):
            if node.name != f"{self._result_name}{level}_crd.push_back":
                continue
            if self._mode == "count":
                return (llir.RawStmt(code=f"_cnt{level}++"),)

            coordinate = (
                self._render_expression(node.args[0], path + ("args", "[0]"))
                if node.args
                else "0"
            )
            replacements = [
                llir.RawStmt(
                    code=(
                        f"{self._result_name}{level}_crd_data"
                        f"[_base{level} + _pos{level}] = {coordinate}"
                    )
                ),
                llir.RawStmt(code=f"_pos{level}++"),
            ]
            if index + 1 < len(self._compressed_levels):
                next_level = self._compressed_levels[index + 1]
                replacements.append(
                    llir.RawStmt(
                        code=(
                            f"{self._result_name}{next_level}_pos_data"
                            f"[_base{level} + _pos{level}] = "
                            f"_base{next_level} + _pos{next_level}"
                        )
                    )
                )
            return replacements

        if node.name == f"{self._result_name}_values.push_back":
            if self._mode == "count":
                return ()
            value = (
                self._render_expression(node.args[0], path + ("args", "[0]"))
                if node.args
                else "0"
            )
            return (
                llir.RawStmt(
                    code=(
                        f"{self._result_name}_values_data"
                        f"[_base{self._leaf} + _pos{self._leaf}] = {value}"
                    )
                ),
            )

        if ".sort" in node.name:
            if self._mode == "fill":
                return (node,)
            return ()
        return (node,)

    def _rewrite_var_init_statement(
        self, node: llir.VarInit, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        if any(
            node.var.name == f"p{self._result_name}{level}"
            for level in self._compressed_levels
        ):
            return ()
        return (node,)

    def _special_if_level(self, node: llir.IfThenElse) -> Optional[int]:
        condition = node.cond
        if (
            type(condition) is not llir.BinOp
            or condition.op != "<"
            or type(condition.left) is not llir.FunctionCall
        ):
            return None
        for level in self._compressed_levels:
            if condition.left.name == f"{self._result_name}{level}_pos.back":
                return level
        return None

    def _rewrite_if_statement(
        self, node: llir.IfThenElse, path: LLIRPath
    ) -> Sequence[llir.Stmt]:
        level = self._special_if_level(node)
        if level is None:
            return (node,)

        level_index = self._compressed_levels.index(level)
        parent_level = (
            self._compressed_levels[level_index - 1] if level_index > 0 else None
        )
        if self._mode == "count":
            count_body: List[llir.Stmt] = []
            if parent_level is not None:
                count_body.append(llir.RawStmt(code=f"_cnt{parent_level}++"))
            count_body.append(llir.RawStmt(code=f"_prev{level} = _cnt{level}"))
            return (
                llir.IfThenElse(
                    cond=self._progress_condition("_cnt", level),
                    then_body=count_body,
                ),
            )

        coordinate = self._find_serial_coordinate(node, path)
        fill_body: List[llir.Stmt] = []
        if parent_level is not None and coordinate is not None:
            fill_body.extend(
                [
                    llir.RawStmt(
                        code=(
                            f"{self._result_name}{parent_level}_crd_data"
                            f"[_base{parent_level} + _pos{parent_level}] = "
                            f"{coordinate}"
                        )
                    ),
                    llir.RawStmt(code=f"_pos{parent_level}++"),
                    llir.RawStmt(
                        code=(
                            f"{self._result_name}{level}_pos_data"
                            f"[_base{parent_level} + _pos{parent_level}] = "
                            f"_base{level} + _pos{level}"
                        )
                    ),
                ]
            )
        fill_body.append(llir.RawStmt(code=f"_prev{level} = _pos{level}"))
        return (
            llir.IfThenElse(
                cond=self._progress_condition("_pos", level),
                then_body=fill_body,
            ),
        )

    @staticmethod
    def _progress_condition(prefix: str, level: int) -> llir.BinOp:
        return llir.BinOp(
            op=">",
            left=llir.Var(name=f"{prefix}{level}", type=llir.DataType.INT64),
            right=llir.Var(name=f"_prev{level}", type=llir.DataType.INT64),
        )

    def _find_serial_coordinate(
        self, node: llir.IfThenElse, path: LLIRPath
    ) -> Optional[str]:
        if not node.then_body:
            return None
        for index, statement in enumerate(node.then_body):
            if (
                type(statement) is llir.FunctionCallStmt
                and ".push_back" in statement.name
                and statement.args
            ):
                return self._render_expression(
                    statement.args[0],
                    path + ("then_body", f"[{index}]", "args", "[0]"),
                )
        return None

    def rewrite_function(self, node: llir.Function, path: LLIRPath) -> llir.Function:
        return self._identity.rewrite_function(node, path)

    def rewrite_case(self, node: llir.Case, path: LLIRPath) -> llir.Case:
        return self._identity.rewrite_case(node, path)

    def rewrite_switch(self, node: llir.Switch, path: LLIRPath) -> llir.Switch:
        return self._identity.rewrite_switch(node, path)


def rewrite_result_writes(
    value: LLIRRewriteValueT,
    context: ResultWriteContext,
) -> LLIRRewriteValueT:
    """Return a detached LLIR value with compressed-result writes rewritten.

    Valid contexts use a non-empty result name, a strictly increasing tuple of
    compressed physical levels, and exactly ``"count"`` or ``"fill"`` mode.
    A valid input with no recognized legacy result-write spelling is a detached
    no-op.  Unknown nodes and malformed typed children fail through the shared
    traversal diagnostic.

    A scalar statement root is supported only when its replacement cardinality
    is one.  Deletion and expansion require a statement-list/tuple root so this
    function can preserve the caller's root category.  Count/fill composition
    and a first-level special position-boundary conditional are outside the
    supported production contract.
    """

    checked_context = _validate_context(context)
    LLIRWalker(checked_context.traversal).walk(cast(LLIRValue, value))
    rewriter = _ResultWriteRewriter(checked_context)

    if isinstance(value, llir.Stmt):
        scalar_root: List[llir.Stmt] = [value]
        rewritten = rewriter.rewrite(scalar_root)
        if len(rewritten) != 1:
            _raise_result_write_error(
                checked_context,
                code="unsupported_scalar_result_write_root",
                message=(
                    "a scalar statement root cannot preserve its root category "
                    f"when the rewrite produces {len(rewritten)} statements"
                ),
                path=("root",),
                value=value,
            )
        return cast(LLIRRewriteValueT, rewritten[0])
    return rewriter.rewrite(value)
