"""Typed single-iteration loop elimination over detached current LLIR.

Version 1 deliberately preserves the narrow generated-string behavior formerly
owned by ``CINLowerer``.  Analysis processes direct ``ForLoop.body`` and
``IfThenElse.then_body``/``else_body`` children before their containing
statement sequence.  It does not analyze auxiliary parallel bodies,
``then_body_list``, other control-flow containers, or raw nested list/tuple
members.  Reference rewriting has a separate, broader legacy scope documented
in ``_rewrite_statement_references`` below.

The common LLIR rewriter first validates and detaches the complete input tree,
including containers that the semantic transform intentionally omits.  Unknown
subclasses and malformed typed children therefore fail closed, while every
legal structural miss returns a fully detached no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, NoReturn, Sequence, Tuple, cast

from . import llir
from .llir_traversal import (
    LLIRPath,
    LLIRRewriter,
    LLIRStatementSequence,
    LLIRStatementValue,
    LLIRTraversalContext,
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
)

SINGLE_ITERATION_LOOP_ELIMINATION_TRAVERSAL_CONTEXT = LLIRTraversalContext(
    stage="LLIR transformation",
    pass_name="eliminate_single_iteration_loops",
)


@dataclass(frozen=True)
class SingleIterationLoopEliminationContext:
    """Immutable diagnostic identity for the fixed version-1 transformation."""

    traversal: LLIRTraversalContext = (
        SINGLE_ITERATION_LOOP_ELIMINATION_TRAVERSAL_CONTEXT
    )


SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT = SingleIterationLoopEliminationContext()


@dataclass(frozen=True)
class _LoopMatch:
    loop_variable: str
    end_variable: str
    base: str


@dataclass(frozen=True)
class _ReferenceReplacements:
    generated_strings: Tuple[Tuple[str, str], ...]
    structured_indices: Tuple[Tuple[str, str], ...]


_SINGLE_STEP_BOUND = re.compile(r"^(\w+) \+ 1$")
_BINOP_FAMILY = (llir.BinOp, llir.Add, llir.Mul)


def _diagnostic_context(context: object) -> LLIRTraversalContext:
    if type(context) is SingleIterationLoopEliminationContext:
        traversal = cast(SingleIterationLoopEliminationContext, context).traversal
        if type(traversal) is LLIRTraversalContext:
            return traversal
    return SINGLE_ITERATION_LOOP_ELIMINATION_TRAVERSAL_CONTEXT


def _raise_single_iteration_error(
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


def _validate_context(context: object) -> SingleIterationLoopEliminationContext:
    if type(context) is not SingleIterationLoopEliminationContext:
        _raise_single_iteration_error(
            context,
            code="invalid_single_iteration_loop_elimination_context",
            message="expected an immutable SingleIterationLoopEliminationContext",
            path=("context",),
            value=context,
        )

    typed_context = cast(SingleIterationLoopEliminationContext, context)
    traversal = typed_context.traversal
    if (
        type(traversal) is not LLIRTraversalContext
        or type(traversal.stage) is not str
        or not traversal.stage
        or type(traversal.pass_name) is not str
        or not traversal.pass_name
    ):
        _raise_single_iteration_error(
            SINGLE_ITERATION_LOOP_ELIMINATION_CONTEXT,
            code="invalid_single_iteration_loop_elimination_traversal_context",
            message="traversal stage and pass name must be non-empty strings",
            path=("context", "traversal"),
            value=traversal,
        )
    return typed_context


def _validate_root(
    statements: object,
    context: SingleIterationLoopEliminationContext,
) -> List[LLIRStatementValue]:
    if type(statements) is not list:
        _raise_single_iteration_error(
            context,
            code="unsupported_single_iteration_loop_elimination_root",
            message="single-iteration elimination requires a statement-list root",
            path=("root",),
            value=statements,
        )

    typed_statements = cast(List[LLIRStatementValue], statements)
    for index, statement in enumerate(typed_statements):
        if (
            not isinstance(statement, llir.Stmt)
            and type(statement) is not list
            and type(statement) is not tuple
        ):
            _raise_single_iteration_error(
                context,
                code="invalid_single_iteration_loop_elimination_root_member",
                message=(
                    "the top-level list may contain only LLIR statements or "
                    "nested statement lists/tuples"
                ),
                path=("root", f"[{index}]"),
                value=statement,
            )
    return typed_statements


def _checked_var_name(
    variable: llir.Var,
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
) -> str:
    if type(variable.name) is not str:
        _raise_single_iteration_error(
            context,
            code="invalid_single_iteration_loop_var_name",
            message="a consumed single-iteration Var name must be a string",
            path=path + ("name",),
            value=variable.name,
        )
    return variable.name


def _checked_binary_operator(
    binary: llir.BinOp,
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
) -> str:
    if type(binary.op) is not str:
        _raise_single_iteration_error(
            context,
            code="invalid_single_iteration_loop_binary_operator",
            message="a candidate single-iteration BinOp operator must be a string",
            path=path + ("op",),
            value=binary.op,
        )
    return binary.op


def _checked_function_name(
    statement: llir.FunctionCallStmt,
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
) -> str:
    if type(statement.name) is not str:
        _raise_single_iteration_error(
            context,
            code="invalid_single_iteration_loop_function_name",
            message="a rewritten FunctionCallStmt name must be a string",
            path=path + ("name",),
            value=statement.name,
        )
    return statement.name


def _checked_raw_code(
    statement: llir.RawStmt,
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
) -> str:
    if type(statement.code) is not str:
        _raise_single_iteration_error(
            context,
            code="invalid_single_iteration_loop_raw_statement",
            message="a rewritten RawStmt code field must be a string",
            path=path + ("code",),
            value=statement.code,
        )
    return statement.code


def _collect_single_step_bounds(
    statements: Sequence[LLIRStatementValue],
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
) -> Dict[str, str]:
    """Recompute the direct declaration map in source order."""

    bounds: Dict[str, str] = {}
    for index, statement in enumerate(statements):
        if type(statement) is not llir.VarInit:
            continue
        initializer = cast(llir.VarInit, statement)
        statement_path = path + (f"[{index}]",)
        base: str | None = None
        if type(initializer.value) is llir.Var:
            value_name = _checked_var_name(
                cast(llir.Var, initializer.value),
                context,
                statement_path + ("value",),
            )
            match = _SINGLE_STEP_BOUND.match(value_name)
            if match is not None:
                base = match.group(1)
        elif type(initializer.value) is llir.Add:
            value = cast(llir.Add, initializer.value)
            if (
                type(value.left) is llir.Var
                and type(value.right) is llir.Literal
                and value.left.type is llir.DataType.INT64
                and not value.left.is_ptr
                and not value.left.is_restrict
                and value.left.tensor_access is None
                and type(value.right.value) is int
                and value.right.value == 1
                and value.right.data_type is llir.DataType.INT64
            ):
                candidate = _checked_var_name(
                    cast(llir.Var, value.left),
                    context,
                    statement_path + ("value", "left"),
                )
                if re.fullmatch(r"\w+", candidate) is not None:
                    base = candidate
        if base is None:
            continue
        end_variable = _checked_var_name(
            initializer.var,
            context,
            statement_path + ("var",),
        )
        bounds[end_variable] = base
    return bounds


def _match_loop(
    statement: LLIRStatementValue,
    bounds: Dict[str, str],
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
) -> _LoopMatch | None:
    if type(statement) is not llir.ForLoop:
        return None
    loop = cast(llir.ForLoop, statement)
    if type(loop.init) is not llir.VarInit or type(loop.cond) not in _BINOP_FAMILY:
        return None

    condition = cast(llir.BinOp, loop.cond)
    if _checked_binary_operator(condition, context, path + ("cond",)) != "<":
        return None
    if type(condition.right) is not llir.Var:
        return None

    loop_variable = _checked_var_name(
        cast(llir.VarInit, loop.init).var,
        context,
        path + ("init", "var"),
    )
    end_variable = _checked_var_name(
        cast(llir.Var, condition.right),
        context,
        path + ("cond", "right"),
    )

    initial_value = cast(llir.VarInit, loop.init).value
    if type(initial_value) is llir.Var:
        initial_name = _checked_var_name(
            cast(llir.Var, initial_value),
            context,
            path + ("init", "value"),
        )
    elif type(initial_value) is llir.Literal:
        initial_name = str(cast(llir.Literal, initial_value).value)
    else:
        return None

    base = bounds.get(end_variable)
    if base is None or initial_name != base or loop_variable == base:
        return None
    return _LoopMatch(loop_variable, end_variable, base)


def _rewrite_expression_references(
    expression: llir.Expr,
    replacements: _ReferenceReplacements,
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
    *,
    in_array_index: bool = False,
) -> llir.Expr:
    if type(expression) is llir.Var:
        variable = cast(llir.Var, expression)
        name = _checked_var_name(variable, context, path)
        changed = False
        if in_array_index:
            for old, new in replacements.structured_indices:
                if name == old:
                    name = new
                    changed = True
        for old, new in replacements.generated_strings:
            if name == old or old in name:
                name = name.replace(old, new)
                changed = True
        if not changed:
            return variable
        return llir.Var(
            name=name,
            type=variable.type,
            is_ptr=variable.is_ptr,
            is_restrict=variable.is_restrict,
            tensor_access=variable.tensor_access,
        )

    if type(expression) in _BINOP_FAMILY:
        binary = cast(llir.BinOp, expression)
        left = _rewrite_expression_references(
            binary.left,
            replacements,
            context,
            path + ("left",),
            in_array_index=in_array_index,
        )
        right = _rewrite_expression_references(
            binary.right,
            replacements,
            context,
            path + ("right",),
            in_array_index=in_array_index,
        )
        return llir.rebuild_binary_expression(binary, left, right)
    if type(expression) is llir.ArrayAccess:
        access = cast(llir.ArrayAccess, expression)
        return llir.ArrayAccess(
            array=_rewrite_expression_references(
                access.array,
                replacements,
                context,
                path + ("array",),
            ),
            index=_rewrite_expression_references(
                access.index,
                replacements,
                context,
                path + ("index",),
                in_array_index=True,
            ),
            tensor_access=access.tensor_access,
        )
    return expression


def _rewrite_expression_sequence(
    expressions: Sequence[llir.Expr],
    replacements: _ReferenceReplacements,
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
) -> Sequence[llir.Expr]:
    rewritten = [
        _rewrite_expression_references(
            expression,
            replacements,
            context,
            path + (f"[{index}]",),
        )
        for index, expression in enumerate(expressions)
    ]
    if type(expressions) is tuple:
        return tuple(rewritten)
    return rewritten


def _rewrite_statement_references(
    statements: Sequence[LLIRStatementValue],
    replacements: _ReferenceReplacements,
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
) -> None:
    """Apply exactly the broader legacy generated-string rewrite scope.

    Loop headers and auxiliary parallel bodies, ``WhileLoop``, ``ForLoopAuto``,
    ``Function``, raw nested statement containers, and unsupported expression
    containers remain intentionally untouched. The common detachment performed
    before this function still validates them.
    """

    for index, statement in enumerate(statements):
        statement_path = path + (f"[{index}]",)
        if type(statement) is llir.Assign:
            assignment = cast(llir.Assign, statement)
            rewritten_target = _rewrite_expression_references(
                assignment.var,
                replacements,
                context,
                statement_path + ("var",),
            )
            llir._validate_assignment_target(rewritten_target)
            assignment.var = cast(llir.AssignmentTarget, rewritten_target)
            assignment.value = _rewrite_expression_references(
                assignment.value,
                replacements,
                context,
                statement_path + ("value",),
            )
        elif type(statement) is llir.VarInit:
            initializer = cast(llir.VarInit, statement)
            initializer.value = _rewrite_expression_references(
                initializer.value,
                replacements,
                context,
                statement_path + ("value",),
            )
        elif type(statement) is llir.FunctionCallStmt:
            call = cast(llir.FunctionCallStmt, statement)
            name = _checked_function_name(call, context, statement_path)
            for old, new in replacements.generated_strings:
                name = name.replace(old, new)
            call.name = name
            call.args = cast(
                List[llir.Expr],
                _rewrite_expression_sequence(
                    call.args,
                    replacements,
                    context,
                    statement_path + ("args",),
                ),
            )
        elif type(statement) is llir.ForLoop:
            loop = cast(llir.ForLoop, statement)
            _rewrite_statement_references(
                cast(Sequence[LLIRStatementValue], loop.body),
                replacements,
                context,
                statement_path + ("body",),
            )
        elif type(statement) is llir.IfThenElse:
            conditional = cast(llir.IfThenElse, statement)
            if conditional.then_body is not None:
                _rewrite_statement_references(
                    cast(Sequence[LLIRStatementValue], conditional.then_body),
                    replacements,
                    context,
                    statement_path + ("then_body",),
                )
            if conditional.else_body is not None:
                _rewrite_statement_references(
                    cast(Sequence[LLIRStatementValue], conditional.else_body),
                    replacements,
                    context,
                    statement_path + ("else_body",),
                )
            if conditional.then_body_list is not None:
                for branch_index, branch in enumerate(conditional.then_body_list):
                    _rewrite_statement_references(
                        cast(Sequence[LLIRStatementValue], branch),
                        replacements,
                        context,
                        statement_path + ("then_body_list", f"[{branch_index}]"),
                    )
        elif type(statement) is llir.RawStmt:
            raw_statement = cast(llir.RawStmt, statement)
            code = _checked_raw_code(raw_statement, context, statement_path)
            for old, new in replacements.generated_strings:
                code = code.replace(old, new)
            raw_statement.code = code


def _is_bound_declaration(
    statement: LLIRStatementValue,
    end_variable: str,
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
) -> bool:
    if type(statement) is not llir.VarInit:
        return False
    initializer = cast(llir.VarInit, statement)
    return _checked_var_name(initializer.var, context, path + ("var",)) == end_variable


def _process_child_sequences_first(
    statements: Sequence[LLIRStatementValue],
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
) -> None:
    """Apply the legacy analysis recursion before the containing sequence."""

    for index, statement in enumerate(statements):
        statement_path = path + (f"[{index}]",)
        if type(statement) is llir.ForLoop:
            loop = cast(llir.ForLoop, statement)
            loop.body = cast(
                List[llir.Stmt],
                _eliminate_in_sequence(
                    cast(LLIRStatementSequence, loop.body),
                    context,
                    statement_path + ("body",),
                ),
            )
        elif type(statement) is llir.IfThenElse:
            conditional = cast(llir.IfThenElse, statement)
            if conditional.then_body is not None:
                conditional.then_body = cast(
                    List[llir.Stmt],
                    _eliminate_in_sequence(
                        cast(LLIRStatementSequence, conditional.then_body),
                        context,
                        statement_path + ("then_body",),
                    ),
                )
            if conditional.else_body is not None:
                conditional.else_body = cast(
                    List[llir.Stmt],
                    _eliminate_in_sequence(
                        cast(LLIRStatementSequence, conditional.else_body),
                        context,
                        statement_path + ("else_body",),
                    ),
                )


def _eliminate_in_sequence(
    statements: LLIRStatementSequence,
    context: SingleIterationLoopEliminationContext,
    path: LLIRPath,
) -> LLIRStatementSequence:
    _process_child_sequences_first(statements, context, path)
    bounds = _collect_single_step_bounds(statements, context, path)

    working: List[LLIRStatementValue] = list(statements)
    index = 0
    while index < len(working):
        statement = working[index]
        statement_path = path + (f"[{index}]",)
        match = _match_loop(statement, bounds, context, statement_path)
        if match is None:
            index += 1
            continue

        loop = cast(llir.ForLoop, statement)
        inlined: List[LLIRStatementValue] = list(
            cast(Sequence[LLIRStatementValue], loop.body)
        )
        replacements = _ReferenceReplacements(
            generated_strings=(
                (f"{match.loop_variable}]", f"{match.base}]"),
                (f"[{match.loop_variable}]", f"[{match.base}]"),
                (f"{match.loop_variable} ", f"{match.base} "),
            ),
            structured_indices=((match.loop_variable, match.base),),
        )
        _rewrite_statement_references(
            inlined,
            replacements,
            context,
            statement_path + ("body",),
        )

        retained: List[LLIRStatementValue] = []
        for retained_index, candidate in enumerate(working):
            if not _is_bound_declaration(
                candidate,
                match.end_variable,
                context,
                path + (f"[{retained_index}]",),
            ):
                retained.append(candidate)
        working = retained

        loop_index = next(
            (
                candidate_index
                for candidate_index, candidate in enumerate(working)
                if candidate is statement
            ),
            None,
        )
        if loop_index is None:
            break
        working[loop_index : loop_index + 1] = inlined
        index = loop_index

    if type(statements) is tuple:
        return tuple(working)
    return working


def eliminate_single_iteration_loops(
    statements: List[llir.Stmt],
    context: SingleIterationLoopEliminationContext,
) -> List[llir.Stmt]:
    """Return detached LLIR with the legacy single-iteration loops inlined.

    The root must be an exact statement list.  Unknown subclasses, malformed
    typed children, and invalid contexts fail through structured
    :class:`LLIRTraversalError` diagnostics.  Normal repeated application is
    structurally idempotent and still returns a newly detached result.
    """

    checked_context = _validate_context(context)
    checked_statements = _validate_root(statements, checked_context)
    detached = cast(
        List[LLIRStatementValue],
        LLIRRewriter(checked_context.traversal).rewrite(checked_statements),
    )
    transformed = _eliminate_in_sequence(detached, checked_context, ("root",))
    return cast(List[llir.Stmt], transformed)
