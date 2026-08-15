"""Typed loop-invariant factor hoisting over detached current LLIR.

Version 1 preserves the deliberately narrow generated-string optimization
formerly owned by ``CINLowerer``.  Transform recursion processes direct
``ForLoop.body`` and truthy ``IfThenElse.then_body``/``else_body`` children
before their containing sequence.  Defined-variable analysis has a distinct
scope: it additionally enters ``WhileLoop.body`` and nested loop bodies, but
still omits auxiliary parallel bodies, ``then_body_list``, ``ForLoopAuto``,
``Function``, and raw nested list/tuple members.

The common LLIR rewriter first validates and detaches the complete input tree,
including containers that the semantic transform intentionally omits.  Unknown
subclasses and malformed typed children therefore fail closed, while every
legal structural miss returns a fully detached no-op.  Matching, factor
classification, raw-substring dependence, and generated names remain
intentionally non-symbolic compatibility behavior.  Successful output
materialization stays structured for the canonical LLIR emitter.  Exact
generated declaration/post pairs remain invisible to the legacy definition
name scan, matching their former opaque-statement behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NoReturn, Sequence, Tuple, cast

from . import llir
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

LOOP_INVARIANT_FACTOR_HOIST_TRAVERSAL_CONTEXT = LLIRTraversalContext(
    stage="LLIR transformation",
    pass_name="hoist_loop_invariant_factors",
)


@dataclass(frozen=True)
class LoopInvariantFactorHoistContext:
    """Immutable diagnostic identity for the fixed version-1 transformation."""

    traversal: LLIRTraversalContext = LOOP_INVARIANT_FACTOR_HOIST_TRAVERSAL_CONTEXT


LOOP_INVARIANT_FACTOR_HOIST_CONTEXT = LoopInvariantFactorHoistContext()

_BINOP_FAMILY = (llir.BinOp, llir.Add, llir.Mul)
_Factor = Tuple[llir.Expr, LLIRPath]


def _diagnostic_context(context: object) -> LLIRTraversalContext:
    if type(context) is LoopInvariantFactorHoistContext:
        traversal = cast(LoopInvariantFactorHoistContext, context).traversal
        if type(traversal) is LLIRTraversalContext:
            return traversal
    return LOOP_INVARIANT_FACTOR_HOIST_TRAVERSAL_CONTEXT


def _raise_factor_hoist_error(
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


def _validate_context(context: object) -> LoopInvariantFactorHoistContext:
    if type(context) is not LoopInvariantFactorHoistContext:
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_hoist_context",
            message="expected an immutable LoopInvariantFactorHoistContext",
            path=("context",),
            value=context,
        )

    typed_context = cast(LoopInvariantFactorHoistContext, context)
    traversal = typed_context.traversal
    if (
        type(traversal) is not LLIRTraversalContext
        or type(traversal.stage) is not str
        or not traversal.stage
        or type(traversal.pass_name) is not str
        or not traversal.pass_name
    ):
        _raise_factor_hoist_error(
            LOOP_INVARIANT_FACTOR_HOIST_CONTEXT,
            code="invalid_loop_invariant_factor_hoist_traversal_context",
            message="traversal stage and pass name must be non-empty strings",
            path=("context", "traversal"),
            value=traversal,
        )
    return typed_context


def _validate_root(
    statements: object,
    context: LoopInvariantFactorHoistContext,
) -> List[LLIRStatementValue]:
    if type(statements) is not list:
        _raise_factor_hoist_error(
            context,
            code="unsupported_loop_invariant_factor_hoist_root",
            message="loop-invariant factor hoisting requires a statement-list root",
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
            _raise_factor_hoist_error(
                context,
                code="invalid_loop_invariant_factor_hoist_root_member",
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
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> str:
    if type(variable.name) is not str:
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_var_name",
            message="a consumed loop-invariant-factor Var name must be a string",
            path=path + ("name",),
            value=variable.name,
        )
    return variable.name


def _checked_binary_operator(
    binary: llir.BinOp,
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> str:
    if type(binary.op) is not str:
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_binary_operator",
            message="a consumed loop-invariant-factor operator must be a string",
            path=path + ("op",),
            value=binary.op,
        )
    return binary.op


def _checked_assign_operator(
    assignment: llir.Assign,
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> llir.AssignOp:
    operator = assignment.op
    if type(operator) is not llir.AssignOp or type(operator.value) is not str:
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_assign_operator",
            message="a candidate assignment requires an exact AssignOp value",
            path=path + ("op",),
            value=operator,
        )
    return operator


def _checked_target(
    target: llir.Expr,
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> llir.Var:
    if type(target) is not llir.Var:
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_target",
            message="a qualifying accumulation target must be an exact LLIR Var",
            path=path,
            value=target,
        )
    return cast(llir.Var, target)


def _collect_defined_vars(
    statements: Sequence[LLIRStatementValue],
    defined: set[str],
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> None:
    """Recompute exactly the legacy recursive definition-name set."""

    for index, statement in enumerate(statements):
        statement_path = path + (f"[{index}]",)
        if type(statement) is llir.FixedStackArrayDecl:
            declaration = cast(llir.FixedStackArrayDecl, statement)
            defined.add(declaration.name)
        elif type(statement) is llir.VarInit:
            initializer = cast(llir.VarInit, statement)
            if not _is_materialized_factor_declaration(statements, index):
                defined.add(
                    _checked_var_name(
                        initializer.var,
                        context,
                        statement_path + ("var",),
                    )
                )
        elif type(statement) is llir.ForLoop:
            loop = cast(llir.ForLoop, statement)
            if type(loop.init) is llir.VarInit:
                defined.add(
                    _checked_var_name(
                        loop.init.var,
                        context,
                        statement_path + ("init", "var"),
                    )
                )
            _collect_defined_vars(
                cast(Sequence[LLIRStatementValue], loop.body),
                defined,
                context,
                statement_path + ("body",),
            )
        elif type(statement) is llir.WhileLoop:
            while_loop = cast(llir.WhileLoop, statement)
            _collect_defined_vars(
                cast(Sequence[LLIRStatementValue], while_loop.body),
                defined,
                context,
                statement_path + ("body",),
            )
        elif type(statement) is llir.IfThenElse:
            conditional = cast(llir.IfThenElse, statement)
            if conditional.then_body:
                _collect_defined_vars(
                    cast(Sequence[LLIRStatementValue], conditional.then_body),
                    defined,
                    context,
                    statement_path + ("then_body",),
                )
            if conditional.else_body:
                _collect_defined_vars(
                    cast(Sequence[LLIRStatementValue], conditional.else_body),
                    defined,
                    context,
                    statement_path + ("else_body",),
                )


def _matches_materialization_var(
    variable: object,
    *,
    expected_name: str | None,
    expected_type: llir.DataType | None,
    expected_is_ptr: bool | None,
    expected_is_restrict: bool | None,
) -> bool:
    return (
        type(variable) is llir.Var
        and type(variable.name) is str
        and bool(variable.name)
        and (expected_name is None or variable.name == expected_name)
        and type(variable.type) is llir.DataType
        and (expected_type is None or variable.type is expected_type)
        and type(variable.is_ptr) is bool
        and (expected_is_ptr is None or variable.is_ptr is expected_is_ptr)
        and type(variable.is_restrict) is bool
        and (
            expected_is_restrict is None or variable.is_restrict is expected_is_restrict
        )
        and variable.tensor_access is None
    )


def _matches_generated_factor_declaration(
    statement: object,
    sequence_index: int,
) -> bool:
    if type(statement) is not llir.VarInit:
        return False
    initializer = cast(llir.VarInit, statement)
    return (
        _matches_materialization_var(
            initializer.var,
            expected_name=f"_inv_{sequence_index}",
            expected_type=llir.DataType.FLOAT32,
            expected_is_ptr=False,
            expected_is_restrict=False,
        )
        and isinstance(initializer.value, llir.Expr)
        and initializer.op == "="
        and initializer.cast is False
    )


def _same_materialization_target(left: object, right: object) -> bool:
    if not _matches_materialization_var(
        left,
        expected_name=None,
        expected_type=None,
        expected_is_ptr=None,
        expected_is_restrict=None,
    ) or not _matches_materialization_var(
        right,
        expected_name=None,
        expected_type=None,
        expected_is_ptr=None,
        expected_is_restrict=None,
    ):
        return False
    typed_left = cast(llir.Var, left)
    typed_right = cast(llir.Var, right)
    return (
        typed_left.name == typed_right.name
        and typed_left.type is typed_right.type
        and typed_left.is_ptr is typed_right.is_ptr
        and typed_left.is_restrict is typed_right.is_restrict
    )


def _loop_contains_materialized_target(loop: llir.ForLoop, target: object) -> bool:
    return any(
        type(statement) is llir.Assign
        and cast(llir.Assign, statement).op is llir.AssignOp.ADD_ASSIGN
        and _same_materialization_target(cast(llir.Assign, statement).var, target)
        for statement in loop.body
    )


def _is_materialized_factor_declaration(
    statements: Sequence[LLIRStatementValue],
    declaration_index: int,
) -> bool:
    """Recognize an exact generated declaration/loop/post wrapper block."""

    if not _matches_generated_factor_declaration(
        statements[declaration_index],
        declaration_index,
    ):
        return False

    loop_index = declaration_index + 1
    while loop_index < len(statements) and _matches_generated_factor_declaration(
        statements[loop_index],
        loop_index,
    ):
        loop_index += 1
    if (
        loop_index >= len(statements)
        or type(statements[loop_index]) is not llir.ForLoop
    ):
        return False
    loop = cast(llir.ForLoop, statements[loop_index])

    declaration_indices = range(declaration_index, loop_index)
    for post_offset, generated_index in enumerate(
        reversed(declaration_indices),
        start=1,
    ):
        post_index = loop_index + post_offset
        if (
            post_index >= len(statements)
            or type(statements[post_index]) is not llir.Assign
        ):
            return False
        assignment = cast(llir.Assign, statements[post_index])
        if (
            assignment.op is not llir.AssignOp.MUL_ASSIGN
            or assignment.cast is not False
            or not _matches_materialization_var(
                assignment.value,
                expected_name=f"_inv_{generated_index}",
                expected_type=llir.DataType.FLOAT32,
                expected_is_ptr=False,
                expected_is_restrict=False,
            )
            or not _loop_contains_materialized_target(loop, assignment.var)
        ):
            return False
    return True


def _collect_mul_factors(
    expression: llir.Expr,
    factors: List[_Factor],
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> None:
    """Flatten exact multiply-family trees in deterministic left-to-right order."""

    if type(expression) in _BINOP_FAMILY:
        binary = cast(llir.BinOp, expression)
        if _checked_binary_operator(binary, context, path) == "*":
            _collect_mul_factors(
                binary.left,
                factors,
                context,
                path + ("left",),
            )
            _collect_mul_factors(
                binary.right,
                factors,
                context,
                path + ("right",),
            )
            return
    factors.append((expression, path))


def _partition_factors(
    factors: Sequence[_Factor],
    body_defined_vars: set[str],
    context: LoopInvariantFactorHoistContext,
) -> Tuple[List[_Factor], List[_Factor]]:
    invariant: List[_Factor] = []
    variant: List[_Factor] = []
    for factor, path in factors:
        if _factor_is_variant(factor, body_defined_vars, context, path):
            variant.append((factor, path))
        else:
            invariant.append((factor, path))
    return invariant, variant


def _factor_is_variant(
    expression: llir.Expr,
    body_defined_vars: set[str],
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> bool:
    if type(expression) is llir.Var:
        name = _checked_var_name(cast(llir.Var, expression), context, path)
        return "_ptr[" in name or any(
            defined_name in name for defined_name in body_defined_vars
        )
    if type(expression) is llir.ArrayAccess:
        access = cast(llir.ArrayAccess, expression)
        if type(access.array) is llir.Var:
            array_name = _checked_var_name(
                cast(llir.Var, access.array),
                context,
                path + ("array",),
            )
            if array_name.endswith("_ptr"):
                return True
        return _factor_is_variant(
            access.array,
            body_defined_vars,
            context,
            path + ("array",),
        ) or _factor_is_variant(
            access.index,
            body_defined_vars,
            context,
            path + ("index",),
        )
    return False


def _rebuild_product(factors: Sequence[_Factor]) -> llir.Expr:
    product = factors[0][0]
    for factor, _ in factors[1:]:
        product = llir.BinOp(left=product, op="*", right=factor)
    return product


def _replace_body_assignment(
    loop: llir.ForLoop,
    body_index: int,
    assignment: llir.Assign,
    variant_expression: llir.Expr,
) -> None:
    # The same assignment with a hoisted-out invariant factor, so it carries
    # the original's result-storage marker: same target, same direction, same
    # arrays named by the target.  Dropping it would un-mark a result write.
    replacement = llir.Assign(
        var=assignment.var,
        value=variant_expression,
        op=assignment.op,
        cast=False,
        result_storage=assignment.result_storage,
    )
    replacement.cast = assignment.cast
    rewritten_body: List[LLIRStatementValue] = list(
        cast(Sequence[LLIRStatementValue], loop.body)
    )
    rewritten_body[body_index] = replacement
    if type(loop.body) is tuple:
        loop.body = cast(List[llir.Stmt], tuple(rewritten_body))
    else:
        loop.body = cast(List[llir.Stmt], rewritten_body)


def _validate_materialization_var(
    variable: object,
    *,
    expected_name: str | None,
    expected_type: llir.DataType | None,
    expected_is_ptr: bool | None,
    expected_is_restrict: bool | None,
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> llir.Var:
    if not _matches_materialization_var(
        variable,
        expected_name=expected_name,
        expected_type=expected_type,
        expected_is_ptr=expected_is_ptr,
        expected_is_restrict=expected_is_restrict,
    ):
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_materialization_var",
            message="generated factor materialization requires an exact scalar Var",
            path=path,
            value=variable,
        )
    return cast(llir.Var, variable)


def _validate_materialization(
    declaration: object,
    post: object,
    loop: llir.ForLoop,
    body_index: object,
    invariant_expression: object,
    sequence_index: int,
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> Tuple[llir.VarInit, llir.Assign]:
    """Validate the complete generated pair at its owning pass boundary."""

    if type(declaration) is not llir.VarInit:
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_materialization_declaration",
            message="generated factor declaration must be an exact VarInit",
            path=path,
            value=declaration,
        )
    typed_declaration = cast(llir.VarInit, declaration)
    invariant_name = f"_inv_{sequence_index}"
    _validate_materialization_var(
        typed_declaration.var,
        expected_name=invariant_name,
        expected_type=llir.DataType.FLOAT32,
        expected_is_ptr=False,
        expected_is_restrict=False,
        context=context,
        path=path + ("var",),
    )
    if typed_declaration.op != "=" or typed_declaration.cast is not False:
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_materialization_declaration_fields",
            message="generated factor declaration requires default initialization fields",
            path=path,
            value=typed_declaration,
        )
    if typed_declaration.value is not invariant_expression:
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_materialization_declaration_value",
            message="generated factor declaration must own the selected invariant",
            path=path + ("value",),
            value=typed_declaration.value,
        )

    if (
        type(body_index) is not int
        or body_index < 0
        or body_index >= len(loop.body)
        or type(loop.body[body_index]) is not llir.Assign
    ):
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_materialization_source",
            message="generated factor materialization requires its source assignment",
            path=path,
            value=body_index,
        )
    source_assignment = cast(llir.Assign, loop.body[body_index])
    expected_target = _validate_materialization_var(
        source_assignment.var,
        expected_name=None,
        expected_type=None,
        expected_is_ptr=None,
        expected_is_restrict=None,
        context=context,
        path=path + (f"source_body[{body_index}]", "var"),
    )

    post_path = path[:-1] + (f"[{sequence_index + 2}]",)
    if type(post) is not llir.Assign:
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_materialization_assignment",
            message="generated factor post statement must be an exact Assign",
            path=post_path,
            value=post,
        )
    typed_post = cast(llir.Assign, post)
    _validate_materialization_var(
        typed_post.var,
        expected_name=expected_target.name,
        expected_type=expected_target.type,
        expected_is_ptr=expected_target.is_ptr,
        expected_is_restrict=expected_target.is_restrict,
        context=context,
        path=post_path + ("var",),
    )
    _validate_materialization_var(
        typed_post.value,
        expected_name=invariant_name,
        expected_type=llir.DataType.FLOAT32,
        expected_is_ptr=False,
        expected_is_restrict=False,
        context=context,
        path=post_path + ("value",),
    )
    if typed_post.op is not llir.AssignOp.MUL_ASSIGN or typed_post.cast is not False:
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_materialization_assignment_fields",
            message="generated factor post statement requires an uncast multiply-assign",
            path=post_path,
            value=typed_post,
        )

    LLIRWalker(context.traversal).walk(cast(LLIRValue, [typed_declaration, typed_post]))
    return typed_declaration, typed_post


def _try_hoist_from_loop(
    loop: llir.ForLoop,
    sequence_index: int,
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> Tuple[llir.VarInit, llir.Assign, int, llir.Expr] | None:
    if type(loop.update) is not llir.Increment:
        return None

    loop_variable = _checked_var_name(
        loop.update.var,
        context,
        path + ("update", "var"),
    )
    body_defined_vars = {loop_variable}
    _collect_defined_vars(
        cast(Sequence[LLIRStatementValue], loop.body),
        body_defined_vars,
        context,
        path + ("body",),
    )

    for body_index, body_statement in enumerate(loop.body):
        if type(body_statement) is not llir.Assign:
            continue
        assignment = cast(llir.Assign, body_statement)
        assignment_path = path + ("body", f"[{body_index}]")
        assign_operator = _checked_assign_operator(
            assignment,
            context,
            assignment_path,
        )
        if assign_operator.value != "+=":
            continue
        if type(assignment.value) not in _BINOP_FAMILY:
            continue
        root = cast(llir.BinOp, assignment.value)
        if (
            _checked_binary_operator(
                root,
                context,
                assignment_path + ("value",),
            )
            != "*"
        ):
            continue

        if type(assignment.var) is llir.ArrayAccess:
            # Indexed accumulations are legal stores, but this scalar factor
            # hoist does not own their storage or aliasing semantics.
            continue

        target = _checked_target(
            assignment.var,
            context,
            assignment_path + ("var",),
        )
        accumulator_name = _checked_var_name(
            target,
            context,
            assignment_path + ("var",),
        )
        factors: List[_Factor] = []
        _collect_mul_factors(
            assignment.value,
            factors,
            context,
            assignment_path + ("value",),
        )
        if len(factors) < 2:
            continue
        invariant, variant = _partition_factors(
            factors,
            body_defined_vars,
            context,
        )
        if not invariant or not variant:
            continue

        invariant_expression = _rebuild_product(invariant)
        variant_expression = _rebuild_product(variant)
        _replace_body_assignment(
            loop,
            body_index,
            assignment,
            variant_expression,
        )

        invariant_name = f"_inv_{sequence_index}"
        return (
            llir.VarInit(
                var=llir.Var(
                    name=invariant_name,
                    type=llir.DataType.FLOAT32,
                ),
                value=invariant_expression,
            ),
            llir.Assign(
                var=llir.Var(
                    name=accumulator_name,
                    type=target.type,
                    is_ptr=target.is_ptr,
                    is_restrict=target.is_restrict,
                    tensor_access=target.tensor_access,
                ),
                value=llir.Var(
                    name=invariant_name,
                    type=llir.DataType.FLOAT32,
                ),
                op=llir.AssignOp.MUL_ASSIGN,
            ),
            body_index,
            invariant_expression,
        )
    return None


def _process_child_sequences_first(
    statements: Sequence[LLIRStatementValue],
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> None:
    """Apply exactly the legacy post-order transform recursion."""

    for index, statement in enumerate(statements):
        statement_path = path + (f"[{index}]",)
        if type(statement) is llir.ForLoop:
            loop = cast(llir.ForLoop, statement)
            loop.body = cast(
                List[llir.Stmt],
                _hoist_in_sequence(
                    cast(LLIRStatementSequence, loop.body),
                    context,
                    statement_path + ("body",),
                ),
            )
        elif type(statement) is llir.IfThenElse:
            conditional = cast(llir.IfThenElse, statement)
            if conditional.then_body:
                conditional.then_body = cast(
                    List[llir.Stmt],
                    _hoist_in_sequence(
                        cast(LLIRStatementSequence, conditional.then_body),
                        context,
                        statement_path + ("then_body",),
                    ),
                )
            if conditional.else_body:
                conditional.else_body = cast(
                    List[llir.Stmt],
                    _hoist_in_sequence(
                        cast(LLIRStatementSequence, conditional.else_body),
                        context,
                        statement_path + ("else_body",),
                    ),
                )


def _hoist_in_sequence(
    statements: LLIRStatementSequence,
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> LLIRStatementSequence:
    _process_child_sequences_first(statements, context, path)
    working: List[LLIRStatementValue] = list(statements)
    index = 0
    while index < len(working):
        statement = working[index]
        if type(statement) is not llir.ForLoop:
            index += 1
            continue
        loop = cast(llir.ForLoop, statement)
        emitted = _try_hoist_from_loop(
            loop,
            index,
            context,
            path + (f"[{index}]",),
        )
        if emitted is None:
            index += 1
            continue
        declaration, post, body_index, invariant_expression = emitted
        before, after = _validate_materialization(
            declaration,
            post,
            loop,
            body_index,
            invariant_expression,
            index,
            context,
            path + (f"[{index}]",),
        )
        working[index : index + 1] = [before, loop, after]
        index += 2

    if type(statements) is tuple:
        return tuple(working)
    return working


def hoist_loop_invariant_factors(
    statements: List[llir.Stmt],
    context: LoopInvariantFactorHoistContext,
) -> List[llir.Stmt]:
    """Return detached LLIR with legacy invariant factors hoisted.

    The root must be an exact statement list.  Unknown subclasses, malformed
    typed children, consumed malformed scalar fields, and invalid contexts fail
    through structured :class:`LLIRTraversalError` diagnostics.  The pass stops
    on the first failure and performs no retry, skip, or failure-to-no-op
    conversion.
    """

    checked_context = _validate_context(context)
    checked_statements = _validate_root(statements, checked_context)
    detached = cast(
        List[LLIRStatementValue],
        LLIRRewriter(checked_context.traversal).rewrite(checked_statements),
    )
    transformed = _hoist_in_sequence(detached, checked_context, ("root",))
    return cast(List[llir.Stmt], transformed)
