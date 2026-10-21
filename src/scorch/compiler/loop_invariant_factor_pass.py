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

The rewrite replaces ``acc += s * x`` inside a loop by ``acc += x`` and a
multiplication ``acc *= s`` after the loop.  That is the same sum only when
the accumulator is zero on entry to the loop and the loop uses the
accumulator nowhere else.  The pass hoists only when it can show both, by a
sufficient condition:

* a ``VarInit`` of the accumulator with a zero literal precedes the loop in
  the loop's own statement sequence, and no statement between that
  declaration and the loop spells the accumulator's name; and
* the matched accumulation is the only statement of the loop that spells the
  accumulator's name, and its right-hand side does not spell it.

Nested statement lists are emitted inline and count as part of the sequence
that holds them.  A name is searched for as a whole C++ identifier in every
string of a statement: typed fields, composite spellings and raw code alike.
The condition assumes control enters that stretch of the sequence only from
its top: LLIR has no jump statement, and raw code is assumed to hold no label.
An accumulator that is declared in an enclosing sequence, declared without an
initializer or with a nonzero one, declared as raw code, read or written
between its declaration and the loop, or not declared at all is not shown to
be zero, and its loop is left unchanged.  That is a legal miss, not an error.

The hoisted temporary takes the accumulator's scalar type, read from the
zero-initializing declaration; the pass never chooses a scalar type of its
own.  It declares a factor only with an element type a kernel's result
values can have (``scorch.utils.PYTORCH_DTYPE_TO_C_DATATYPE``: float,
double, int8_t, uint8_t, int32_t, int64_t), so a loop whose accumulator is
declared with any other type, or with no type, is left unchanged: one more
legal miss.  A typed accumulator reference that carries a different type
from its typed declaration is malformed IR and raises a structured error.
The validation boundary judges a generated pair that already exists, so
there both cases raise.

For an integer accumulator the rewrite computes the same value: integer
addition and multiplication are associative and distributive, and int8_t
and uint8_t wrap modulo 256 in both forms alike.  For a signed type of int
width or wider, a partial sum of the rewritten loop is the original's
partial sum divided by the factor, so it overflows only where the
original's does, with one exception: when the factor is zero, every term
of the original is zero and only the rewritten sum can overflow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, NoReturn, Optional, Pattern, Sequence, Tuple, cast

from . import llir
from ..utils import PYTORCH_DTYPE_TO_C_DATATYPE
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
# The scalar types a hoisted factor may be declared with: the element types a
# kernel's result values can have.
_FACTOR_SCALAR_TYPES = tuple(PYTORCH_DTYPE_TO_C_DATATYPE.values())
# A literal that initializes a scalar to zero, as C++ spells it.
_ZERO_LITERAL = re.compile(r"[+-]?(?:0+\.?0*|\.0+)(?:[eE][+-]?[0-9]+)?[fFlL]?")


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


def _is_factor_scalar_type(data_type: object) -> bool:
    return any(data_type is scalar_type for scalar_type in _FACTOR_SCALAR_TYPES)


def _generated_factor_declaration_type(
    statement: object,
    sequence_index: int,
) -> llir.DataType | None:
    """The scalar type of an exact generated declaration, else ``None``."""

    if type(statement) is not llir.VarInit:
        return None
    initializer = cast(llir.VarInit, statement)
    if (
        _matches_materialization_var(
            initializer.var,
            expected_name=f"_inv_{sequence_index}",
            expected_type=None,
            expected_is_ptr=False,
            expected_is_restrict=False,
        )
        and _is_factor_scalar_type(initializer.var.type)
        and isinstance(initializer.value, llir.Expr)
        and initializer.op == "="
        and initializer.cast is False
    ):
        return initializer.var.type
    return None


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

    if (
        _generated_factor_declaration_type(
            statements[declaration_index],
            declaration_index,
        )
        is None
    ):
        return False

    loop_index = declaration_index + 1
    while (
        loop_index < len(statements)
        and _generated_factor_declaration_type(statements[loop_index], loop_index)
        is not None
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
        # The post multiplies in the factor its own declaration declared, so
        # the pair carries one scalar type.
        factor_type = _generated_factor_declaration_type(
            statements[generated_index],
            generated_index,
        )
        if (
            factor_type is None
            or assignment.op is not llir.AssignOp.MUL_ASSIGN
            or assignment.cast is not False
            or not _matches_materialization_var(
                assignment.value,
                expected_name=f"_inv_{generated_index}",
                expected_type=factor_type,
                expected_is_ptr=False,
                expected_is_restrict=False,
            )
            or not _loop_contains_materialized_target(loop, assignment.var)
        ):
            return False
    return True


def _identifier_pattern(name: str) -> Pattern[str]:
    """Match ``name`` where it stands as a whole C++ identifier."""

    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")


def _spelling_count(value: object, identifier: Pattern[str]) -> int:
    """How often the strings reachable from ``value`` spell the identifier.

    Variable names, composite spellings such as ``values[position]`` and raw
    code are all strings of the tree, so one search covers a read, a write and
    a declaration, typed or not.  Only strings, statement sequences and LLIR
    nodes are entered; any other field value, a statement's result-storage
    marker among them, spells nothing.
    """

    if isinstance(value, str):
        return len(identifier.findall(value))
    if isinstance(value, (list, tuple)):
        return sum(_spelling_count(member, identifier) for member in value)
    if isinstance(value, llir.Node):
        return sum(_spelling_count(child, identifier) for child in vars(value).values())
    return 0


def _is_zero_literal(value: object) -> bool:
    if type(value) is not llir.Literal:
        return False
    literal = cast(llir.Literal, value)
    if type(literal.value) is str:
        # A string literal without a numeric type is emitted in quotes.
        return (
            literal.data_type is not llir.DataType.STRING
            and _ZERO_LITERAL.fullmatch(literal.value) is not None
        )
    return type(literal.value) in (int, float) and literal.value == 0


def _straight_line_statements(
    statements: Sequence[LLIRStatementValue],
) -> List[LLIRStatementValue]:
    """``statements`` with nested sequences, which are emitted inline, expanded."""

    flat: List[LLIRStatementValue] = []
    for statement in statements:
        if type(statement) is list or type(statement) is tuple:
            flat.extend(
                _straight_line_statements(cast(Sequence[LLIRStatementValue], statement))
            )
        else:
            flat.append(statement)
    return flat


def _zero_initialization(
    preceding: Sequence[LLIRStatementValue],
    name: str,
) -> llir.VarInit | None:
    """The declaration that shows ``name`` is zero where ``preceding`` ends.

    ``preceding`` holds the statements of one sequence that come before a
    loop.  The result is the ``VarInit`` of ``name`` with a zero literal that
    no later statement of ``preceding`` follows with a spelling of ``name``,
    or ``None`` when there is no such declaration.  ``None`` means only that
    zero was not shown.
    """

    identifier = _identifier_pattern(name)
    for statement in reversed(_straight_line_statements(preceding)):
        if type(statement) is llir.VarInit:
            initializer = cast(llir.VarInit, statement)
            if (
                _matches_materialization_var(
                    initializer.var,
                    expected_name=name,
                    expected_type=None,
                    expected_is_ptr=False,
                    expected_is_restrict=False,
                )
                and initializer.op == "="
                and initializer.cast is False
                and _is_zero_literal(initializer.value)
            ):
                return initializer
        if _spelling_count(statement, identifier):
            return None
    return None


def _accumulation_is_the_only_use(
    loop: llir.ForLoop,
    body_index: int,
    name: str,
) -> bool:
    """Whether ``loop`` spells ``name`` only as the target of one statement.

    A second write, a read, or the accumulator on its own right-hand side
    would each see the partial sum before the hoisted factor is applied.  The
    whole loop is searched, its header and every auxiliary field included, so
    the spellings in the loop must be exactly those in the matched target.
    """

    identifier = _identifier_pattern(name)
    target = cast(llir.Assign, loop.body[body_index]).var
    return _spelling_count(loop, identifier) == _spelling_count(target, identifier)


def _accumulator_scalar_type(
    target: llir.Var,
    declaration: llir.VarInit,
    context: LoopInvariantFactorHoistContext,
    path: LLIRPath,
) -> Optional[llir.DataType]:
    """The scalar type a factor hoisted out of ``target`` is declared with.

    ``declaration`` is the accumulator's zero-initializing declaration and the
    source of the type.  ``None`` means the declaration is neither FLOAT32 nor
    FLOAT64, so no factor can be declared for this accumulator.  A typed
    reference that carries a different type from a typed declaration is
    malformed IR and raises.
    """

    declared_type = declaration.var.type
    referenced_type = target.type
    if (
        referenced_type is not llir.DataType.NO_TYPE
        and declared_type is not llir.DataType.NO_TYPE
        and declared_type is not referenced_type
    ):
        _raise_factor_hoist_error(
            context,
            code="inconsistent_loop_invariant_factor_accumulator_type",
            message=(
                "the accumulator reference and its zero-initializing declaration "
                "carry different scalar types"
            ),
            path=path,
            value=target,
        )
    if not _is_factor_scalar_type(declared_type):
        return None
    return declared_type


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
    preceding: Sequence[LLIRStatementValue],
) -> Tuple[llir.VarInit, llir.Assign]:
    """Validate the complete generated pair at its owning pass boundary.

    The expected target, the proof that it is zero on entry and the expected
    scalar type are all derived here from the loop's own source assignment
    and ``preceding``, the statements of the loop's sequence that come before
    it, never taken from the pair under validation.
    """

    if type(declaration) is not llir.VarInit:
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_materialization_declaration",
            message="generated factor declaration must be an exact VarInit",
            path=path,
            value=declaration,
        )
    typed_declaration = cast(llir.VarInit, declaration)

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
    source_target_path = path + (f"source_body[{body_index}]", "var")
    expected_target = _validate_materialization_var(
        source_assignment.var,
        expected_name=None,
        expected_type=None,
        expected_is_ptr=None,
        expected_is_restrict=None,
        context=context,
        path=source_target_path,
    )
    zero_initialization = _zero_initialization(preceding, expected_target.name)
    if zero_initialization is None or not _accumulation_is_the_only_use(
        loop,
        cast(int, body_index),
        expected_target.name,
    ):
        _raise_factor_hoist_error(
            context,
            code="invalid_loop_invariant_factor_materialization_source",
            message=(
                "generated factor materialization requires an accumulator that "
                "is zero on entry to its loop and used nowhere else in it"
            ),
            path=source_target_path,
            value=expected_target,
        )
    accumulator_type = _accumulator_scalar_type(
        expected_target,
        cast(llir.VarInit, zero_initialization),
        context,
        source_target_path,
    )
    if accumulator_type is None:
        _raise_factor_hoist_error(
            context,
            code="unsupported_loop_invariant_factor_accumulator_type",
            message=(
                "a hoisted loop-invariant factor requires an accumulator "
                "declared with a result element type"
            ),
            path=source_target_path,
            value=zero_initialization,
        )

    invariant_name = f"_inv_{sequence_index}"
    _validate_materialization_var(
        typed_declaration.var,
        expected_name=invariant_name,
        expected_type=accumulator_type,
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
        expected_type=accumulator_type,
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
    preceding: Sequence[LLIRStatementValue],
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

        # ``acc += s * x`` equals ``acc += x`` followed by ``acc *= s`` only
        # for an accumulator that is zero on entry and that the loop uses
        # nowhere else.  A loop for which that is not shown is left as it is.
        zero_initialization = _zero_initialization(preceding, accumulator_name)
        if zero_initialization is None or not _accumulation_is_the_only_use(
            loop,
            body_index,
            accumulator_name,
        ):
            continue

        # Resolved before the loop is rewritten: the factor multiplies into
        # this accumulator only, so it is declared with the accumulator's
        # scalar type and never with a type the pass picks.  An accumulator
        # of any other type keeps its loop: unchanged is always correct.
        factor_type = _accumulator_scalar_type(
            target,
            zero_initialization,
            context,
            assignment_path + ("var",),
        )
        if factor_type is None:
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
                    type=factor_type,
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
                    type=factor_type,
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
        # The statements before the loop: read before ``working`` gains the pair.
        preceding = working[:index]
        emitted = _try_hoist_from_loop(
            loop,
            index,
            context,
            path + (f"[{index}]",),
            preceding,
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
            preceding,
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
    through structured :class:`LLIRTraversalError` diagnostics, and so does a
    zero-initialized accumulator whose typed reference and typed declaration
    carry different scalar types.  A loop whose accumulator is not shown to
    be zero on entry, is used elsewhere in the loop, or is declared with a
    type that is not a result element type is a legal miss and is left
    unchanged.
    The pass stops on the first failure and performs no retry, skip, or
    failure-to-no-op conversion.
    """

    checked_context = _validate_context(context)
    checked_statements = _validate_root(statements, checked_context)
    detached = cast(
        List[LLIRStatementValue],
        LLIRRewriter(checked_context.traversal).rewrite(checked_statements),
    )
    transformed = _hoist_in_sequence(detached, checked_context, ("root",))
    return cast(List[llir.Stmt], transformed)
