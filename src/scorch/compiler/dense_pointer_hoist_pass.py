"""Typed dense-pointer hoisting over a detached LLIR statement list.

Version 1 preserves the characterized optimization while structurally owning
typed tensor-value reads.  A
candidate is discovered only in a direct ``ForLoop.body`` and only when its
position initializer has the exact affine shape used by ``CINLowerer``.  Value
arrays are associated with positions through
``ArrayAccess(Var("B_val"), Var("pB1"))`` expressions found in direct
assignments.  The preexisting flat ``<name>_val[<position>]`` compatibility
form remains accepted while legacy producers are retired.

The common LLIR rewriter first validates and detaches the complete input tree.
Semantic analysis is intentionally narrower: it follows only direct
``ForLoop.body`` children and processes children before parents.  Reference
rewriting has the distinct, broader legacy scope documented in
``_rewrite_statement_references`` below.  Unsupported containers are still
validated and detached by the common boundary even when this pass does not
inspect their semantics.  Legal misses therefore return detached no-ops.
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

DENSE_POINTER_HOIST_TRAVERSAL_CONTEXT = LLIRTraversalContext(
    stage="LLIR transformation",
    pass_name="hoist_dense_pointers",
)


@dataclass(frozen=True)
class DensePointerHoistContext:
    """Immutable value-array type snapshot and diagnostic identity."""

    value_array_ctypes: Tuple[Tuple[str, str], ...]
    traversal: LLIRTraversalContext = DENSE_POINTER_HOIST_TRAVERSAL_CONTEXT


@dataclass(frozen=True)
class _PositionCandidate:
    position: str
    base: str
    stride: str
    body_index: int


@dataclass(frozen=True)
class _LoopAnalysis:
    loop_variable: str
    candidates: Tuple[_PositionCandidate, ...]
    position_to_value_array: Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class _StructuredAccessReplacement:
    value_array: str
    position: str
    pointer: str
    loop_variable: str


@dataclass(frozen=True)
class _ReferenceReplacements:
    generated_strings: Tuple[Tuple[str, str], ...]
    structured_accesses: Tuple[_StructuredAccessReplacement, ...]


_STRING_VALUE_ACCESS = re.compile(r"^(\w+_val)\[(\w+)\]$")
_STRUCTURED_VALUE_ARRAY = re.compile(r"^\w+_val$")


class _DensePointerDetacher(LLIRRewriter):
    """Detach the one legacy attribute consumed on any direct statement.

    ``LLIRRewriter`` already owns this compatibility field on ``ForLoop``.
    The legacy insertion sweep, however, consumes it from any statement in the
    sequence.  Preserve that characterized behavior without copying any other
    open-ended dynamic attributes.
    """

    def _rewrite_stmt(self, node: llir.Stmt, path: LLIRPath) -> llir.Stmt:
        rewritten = super()._rewrite_stmt(node, path)
        if type(node) is not llir.ForLoop and hasattr(node, "_hoisted_ptr_decls"):
            declarations = getattr(node, "_hoisted_ptr_decls")
            setattr(
                rewritten,
                "_hoisted_ptr_decls",
                self._rewrite_statements(
                    declarations,
                    path + ("_hoisted_ptr_decls",),
                ),
            )
        return rewritten


def _diagnostic_context(context: object) -> LLIRTraversalContext:
    if type(context) is DensePointerHoistContext:
        traversal = cast(DensePointerHoistContext, context).traversal
        if type(traversal) is LLIRTraversalContext:
            return traversal
    return DENSE_POINTER_HOIST_TRAVERSAL_CONTEXT


def _raise_dense_pointer_error(
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


def _validate_context(
    context: object,
) -> Tuple[DensePointerHoistContext, Dict[str, str]]:
    if type(context) is not DensePointerHoistContext:
        _raise_dense_pointer_error(
            context,
            code="invalid_dense_pointer_hoist_context",
            message="expected an immutable DensePointerHoistContext",
            path=("context",),
            value=context,
        )

    typed_context = cast(DensePointerHoistContext, context)
    traversal = typed_context.traversal
    if (
        type(traversal) is not LLIRTraversalContext
        or type(traversal.stage) is not str
        or not traversal.stage
        or type(traversal.pass_name) is not str
        or not traversal.pass_name
    ):
        _raise_dense_pointer_error(
            DENSE_POINTER_HOIST_TRAVERSAL_CONTEXT,
            code="invalid_dense_pointer_hoist_traversal_context",
            message="traversal stage and pass name must be non-empty strings",
            path=("context", "traversal"),
            value=traversal,
        )

    entries = typed_context.value_array_ctypes
    if type(entries) is not tuple:
        _raise_dense_pointer_error(
            typed_context,
            code="invalid_dense_pointer_hoist_type_map",
            message="value-array C types must be an immutable tuple of pairs",
            path=("context", "value_array_ctypes"),
            value=entries,
        )

    type_map: Dict[str, str] = {}
    for index, entry in enumerate(entries):
        entry_path = ("context", "value_array_ctypes", f"[{index}]")
        if type(entry) is not tuple or len(entry) != 2:
            _raise_dense_pointer_error(
                typed_context,
                code="invalid_dense_pointer_hoist_type_map_entry",
                message="each value-array C-type entry must be an exact pair",
                path=entry_path,
                value=entry,
            )
        value_array, c_type = entry
        if (
            type(value_array) is not str
            or not value_array
            or type(c_type) is not str
            or not c_type
        ):
            _raise_dense_pointer_error(
                typed_context,
                code="invalid_dense_pointer_hoist_type_map_entry",
                message="value-array names and C types must be non-empty strings",
                path=entry_path,
                value=entry,
            )
        if value_array in type_map:
            _raise_dense_pointer_error(
                typed_context,
                code="duplicate_dense_pointer_hoist_value_array",
                message="value-array names must be unique in the C-type snapshot",
                path=entry_path + ("[0]",),
                value=value_array,
            )
        type_map[value_array] = c_type
    return typed_context, type_map


def _validate_root(
    statements: object,
    context: DensePointerHoistContext,
) -> List[LLIRStatementValue]:
    if type(statements) is not list:
        _raise_dense_pointer_error(
            context,
            code="unsupported_dense_pointer_hoist_root",
            message="dense-pointer hoisting requires a statement-list root",
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
            _raise_dense_pointer_error(
                context,
                code="invalid_dense_pointer_hoist_root_member",
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
    context: DensePointerHoistContext,
    path: LLIRPath,
) -> str:
    if type(variable.name) is not str:
        _raise_dense_pointer_error(
            context,
            code="invalid_dense_pointer_hoist_var_name",
            message="a dense-pointer Var name must be a string",
            path=path + ("name",),
            value=variable.name,
        )
    return variable.name


def _checked_binary_operator(
    binary: llir.BinOp,
    context: DensePointerHoistContext,
    path: LLIRPath,
) -> str:
    if type(binary.op) is not str:
        _raise_dense_pointer_error(
            context,
            code="invalid_dense_pointer_hoist_binary_operator",
            message="a matched dense-pointer BinOp operator must be a string",
            path=path + ("op",),
            value=binary.op,
        )
    return binary.op


def _collect_value_array_references(
    expression: llir.Expr,
    position_to_value_array: Dict[str, str],
    context: DensePointerHoistContext,
    path: LLIRPath,
) -> None:
    """Collect exact structured and legacy value-array accesses."""

    if type(expression) is llir.Var:
        variable = cast(llir.Var, expression)
        match = _STRING_VALUE_ACCESS.match(_checked_var_name(variable, context, path))
        if match:
            position_to_value_array[match.group(2)] = match.group(1)
    if type(expression) in (llir.BinOp, llir.Add, llir.Mul):
        binary = cast(llir.BinOp, expression)
        _collect_value_array_references(
            binary.left,
            position_to_value_array,
            context,
            path + ("left",),
        )
        _collect_value_array_references(
            binary.right,
            position_to_value_array,
            context,
            path + ("right",),
        )
    if type(expression) is llir.ArrayAccess:
        access = cast(llir.ArrayAccess, expression)
        if type(access.array) is llir.Var and type(access.index) is llir.Var:
            array_name = _checked_var_name(
                cast(llir.Var, access.array),
                context,
                path + ("array",),
            )
            position = _checked_var_name(
                cast(llir.Var, access.index),
                context,
                path + ("index",),
            )
            if _STRUCTURED_VALUE_ARRAY.match(array_name):
                position_to_value_array[position] = array_name
                return
        _collect_value_array_references(
            access.array,
            position_to_value_array,
            context,
            path + ("array",),
        )
        _collect_value_array_references(
            access.index,
            position_to_value_array,
            context,
            path + ("index",),
        )


def _analyze_loop(
    loop: llir.ForLoop,
    context: DensePointerHoistContext,
    path: LLIRPath,
) -> _LoopAnalysis | None:
    """Purely recompute the direct-body facts needed for one loop."""

    if type(loop.update) is not llir.Increment:
        return None
    loop_variable = _checked_var_name(
        loop.update.var,
        context,
        path + ("update", "var"),
    )

    candidates: List[_PositionCandidate] = []
    for index, statement in enumerate(loop.body):
        if type(statement) is not llir.VarInit:
            continue
        initializer = cast(llir.VarInit, statement)
        value = initializer.value
        if type(value) is not llir.Add:
            continue
        add = cast(llir.Add, value)
        if type(add.left) not in (llir.BinOp, llir.Add, llir.Mul):
            continue
        multiply = cast(llir.BinOp, add.left)
        if (
            _checked_binary_operator(
                multiply,
                context,
                path + ("body", f"[{index}]", "value", "left"),
            )
            != "*"
            or type(add.right) is not llir.Var
        ):
            continue
        right = cast(llir.Var, add.right)
        if (
            _checked_var_name(
                right,
                context,
                path + ("body", f"[{index}]", "value", "right"),
            )
            != loop_variable
            or type(multiply.left) is not llir.Var
            or type(multiply.right) is not llir.Var
        ):
            continue
        base = cast(llir.Var, multiply.left)
        stride = cast(llir.Var, multiply.right)
        candidates.append(
            _PositionCandidate(
                position=_checked_var_name(
                    initializer.var,
                    context,
                    path + ("body", f"[{index}]", "var"),
                ),
                base=_checked_var_name(
                    base,
                    context,
                    path + ("body", f"[{index}]", "value", "left", "left"),
                ),
                stride=_checked_var_name(
                    stride,
                    context,
                    path + ("body", f"[{index}]", "value", "left", "right"),
                ),
                body_index=index,
            )
        )

    if not candidates:
        return None

    position_to_value_array: Dict[str, str] = {}
    for index, statement in enumerate(loop.body):
        if type(statement) is not llir.Assign:
            continue
        assignment = cast(llir.Assign, statement)
        assignment_path = path + ("body", f"[{index}]")
        _collect_value_array_references(
            assignment.value,
            position_to_value_array,
            context,
            assignment_path + ("value",),
        )
        if type(assignment.var) in (llir.Var, llir.ArrayAccess):
            _collect_value_array_references(
                assignment.var,
                position_to_value_array,
                context,
                assignment_path + ("var",),
            )

    return _LoopAnalysis(
        loop_variable=loop_variable,
        candidates=tuple(candidates),
        position_to_value_array=tuple(position_to_value_array.items()),
    )


def _rewrite_expression_references(
    expression: llir.Expr,
    replacements: _ReferenceReplacements,
    context: DensePointerHoistContext,
    path: LLIRPath,
) -> llir.Expr:
    if type(expression) is llir.Var:
        variable = cast(llir.Var, expression)
        name = _checked_var_name(variable, context, path)
        rewritten: llir.Expr = variable
        for old, new in replacements.generated_strings:
            if name == old or old in name:
                name = name.replace(old, new)
                rewritten = llir.Var(
                    name=name,
                    type=variable.type,
                    is_ptr=variable.is_ptr,
                    is_restrict=variable.is_restrict,
                    tensor_access=variable.tensor_access,
                )
        return rewritten
    if type(expression) in (llir.BinOp, llir.Add, llir.Mul):
        binary = cast(llir.BinOp, expression)
        left = _rewrite_expression_references(
            binary.left,
            replacements,
            context,
            path + ("left",),
        )
        right = _rewrite_expression_references(
            binary.right,
            replacements,
            context,
            path + ("right",),
        )
        return llir.rebuild_binary_expression(binary, left, right)
    if type(expression) is llir.ArrayAccess:
        access = cast(llir.ArrayAccess, expression)
        if type(access.array) is llir.Var and type(access.index) is llir.Var:
            array = cast(llir.Var, access.array)
            index = cast(llir.Var, access.index)
            array_name = _checked_var_name(array, context, path + ("array",))
            index_name = _checked_var_name(index, context, path + ("index",))
            for replacement in replacements.structured_accesses:
                if (
                    array_name == replacement.value_array
                    and index_name == replacement.position
                ):
                    return llir.ArrayAccess(
                        array=llir.Var(
                            name=replacement.pointer,
                            type=array.type,
                            is_ptr=True,
                            is_restrict=True,
                        ),
                        index=llir.Var(
                            name=replacement.loop_variable,
                            type=index.type,
                        ),
                        tensor_access=access.tensor_access,
                    )
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
            ),
            tensor_access=access.tensor_access,
        )
    if type(expression) is llir.AddressOf:
        address = cast(llir.AddressOf, expression)
        rewritten_operand = _rewrite_expression_references(
            address.operand,
            replacements,
            context,
            path + ("operand",),
        )
        return llir.AddressOf(
            operand=cast(llir.AssignmentTarget, rewritten_operand),
        )
    return expression


def _rewrite_expression_sequence(
    expressions: Sequence[llir.Expr],
    replacements: _ReferenceReplacements,
    context: DensePointerHoistContext,
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


def _checked_function_name(
    statement: llir.FunctionCallStmt,
    context: DensePointerHoistContext,
    path: LLIRPath,
) -> str:
    if type(statement.name) is not str:
        _raise_dense_pointer_error(
            context,
            code="invalid_dense_pointer_hoist_function_name",
            message="a rewritten FunctionCallStmt name must be a string",
            path=path + ("name",),
            value=statement.name,
        )
    return statement.name


def _checked_raw_code(
    statement: llir.RawStmt,
    context: DensePointerHoistContext,
    path: LLIRPath,
) -> str:
    if type(statement.code) is not str:
        _raise_dense_pointer_error(
            context,
            code="invalid_dense_pointer_hoist_raw_statement",
            message="a rewritten RawStmt code field must be a string",
            path=path + ("code",),
            value=statement.code,
        )
    return statement.code


def _rewrite_statement_references(
    statements: LLIRStatementSequence,
    replacements: _ReferenceReplacements,
    context: DensePointerHoistContext,
    path: LLIRPath,
) -> LLIRStatementSequence:
    """Apply exactly the broader legacy rewrite scope.

    Loop headers, auxiliary parallel bodies, WhileLoop, ForLoopAuto, Function,
    and unsupported expression containers are deliberately omitted. The common
    rewrite performed before this function has still validated and detached all
    of them.
    """

    rewritten: List[LLIRStatementValue] = []
    for index, statement in enumerate(statements):
        statement_path = path + (f"[{index}]",)
        rewritten_statement = statement
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
            rewritten_statement = llir.FunctionCallStmt(
                name=name,
                args=_rewrite_expression_sequence(
                    call.args,
                    replacements,
                    context,
                    statement_path + ("args",),
                ),
            )
        elif type(statement) is llir.MemberCallStmt:
            member_call = cast(llir.MemberCallStmt, statement)
            rewritten_statement = llir.MemberCallStmt(
                base=_rewrite_expression_references(
                    member_call.base,
                    replacements,
                    context,
                    statement_path + ("base",),
                ),
                member=member_call.member,
                template_args=member_call.template_args,
                args=_rewrite_expression_sequence(
                    member_call.args,
                    replacements,
                    context,
                    statement_path + ("args",),
                ),
            )
        elif type(statement) is llir.ForLoop:
            loop = cast(llir.ForLoop, statement)
            loop.body = cast(
                List[llir.Stmt],
                _rewrite_statement_references(
                    cast(LLIRStatementSequence, loop.body),
                    replacements,
                    context,
                    statement_path + ("body",),
                ),
            )
        elif type(statement) is llir.IfThenElse:
            conditional = cast(llir.IfThenElse, statement)
            if conditional.then_body:
                conditional.then_body = cast(
                    List[llir.Stmt],
                    _rewrite_statement_references(
                        cast(LLIRStatementSequence, conditional.then_body),
                        replacements,
                        context,
                        statement_path + ("then_body",),
                    ),
                )
            if conditional.else_body:
                conditional.else_body = cast(
                    List[llir.Stmt],
                    _rewrite_statement_references(
                        cast(LLIRStatementSequence, conditional.else_body),
                        replacements,
                        context,
                        statement_path + ("else_body",),
                    ),
                )
            if conditional.then_body_list:
                rewritten_branches: List[LLIRStatementSequence] = []
                for branch_index, branch in enumerate(conditional.then_body_list):
                    rewritten_branches.append(
                        _rewrite_statement_references(
                            cast(LLIRStatementSequence, branch),
                            replacements,
                            context,
                            statement_path + ("then_body_list", f"[{branch_index}]"),
                        )
                    )
                if type(conditional.then_body_list) is tuple:
                    conditional.then_body_list = cast(
                        List[List[llir.Stmt]], tuple(rewritten_branches)
                    )
                else:
                    conditional.then_body_list = cast(
                        List[List[llir.Stmt]], rewritten_branches
                    )
        elif type(statement) is llir.RawStmt:
            raw_statement = cast(llir.RawStmt, statement)
            code = _checked_raw_code(raw_statement, context, statement_path)
            for old, new in replacements.generated_strings:
                code = code.replace(old, new)
            raw_statement.code = code
        rewritten.append(rewritten_statement)

    if type(statements) is tuple:
        return tuple(rewritten)
    return rewritten


def _apply_loop_analysis(
    loop: llir.ForLoop,
    analysis: _LoopAnalysis,
    type_map: Dict[str, str],
    context: DensePointerHoistContext,
    path: LLIRPath,
) -> None:
    position_to_value_array = dict(analysis.position_to_value_array)
    declarations: List[llir.Stmt] = []
    indices_to_remove: set[int] = set()
    replacements: Dict[str, str] = {}
    structured_replacements: List[_StructuredAccessReplacement] = []

    for candidate in analysis.candidates:
        value_array = position_to_value_array.get(candidate.position)
        if value_array is None:
            continue
        scalar_type = type_map.get(value_array)
        if scalar_type is None:
            continue
        pointer_name = f"_{value_array}_ptr"
        declarations.append(
            llir.RawStmt(
                code=(
                    f"const {scalar_type}* __restrict__ {pointer_name} = "
                    f"&{value_array}[{candidate.base} * {candidate.stride}]"
                )
            )
        )
        replacements[f"{value_array}[{candidate.position}]"] = (
            f"{pointer_name}[{analysis.loop_variable}]"
        )
        structured_replacements.append(
            _StructuredAccessReplacement(
                value_array=value_array,
                position=candidate.position,
                pointer=pointer_name,
                loop_variable=analysis.loop_variable,
            )
        )
        indices_to_remove.add(candidate.body_index)

    if not declarations:
        return

    # The compatibility attribute is deliberate.  It reproduces the legacy
    # deferred insertion protocol, including overwriting a preexisting value.
    setattr(loop, "_hoisted_ptr_decls", declarations)
    retained = [
        statement
        for index, statement in enumerate(loop.body)
        if index not in indices_to_remove
    ]
    if type(loop.body) is tuple:
        loop.body = cast(List[llir.Stmt], tuple(retained))
    else:
        loop.body = cast(List[llir.Stmt], retained)
    loop.body = cast(
        List[llir.Stmt],
        _rewrite_statement_references(
            cast(LLIRStatementSequence, loop.body),
            _ReferenceReplacements(
                generated_strings=tuple(replacements.items()),
                structured_accesses=tuple(structured_replacements),
            ),
            context,
            path + ("body",),
        ),
    )


def _transform_direct_for_loop_bodies(
    statements: LLIRStatementSequence,
    type_map: Dict[str, str],
    context: DensePointerHoistContext,
    path: LLIRPath,
) -> LLIRStatementSequence:
    transformed: List[LLIRStatementValue] = []
    for index, candidate in enumerate(statements):
        candidate_path = path + (f"[{index}]",)
        if type(candidate) is llir.ForLoop:
            loop = cast(llir.ForLoop, candidate)
            loop.body = cast(
                List[llir.Stmt],
                _transform_direct_for_loop_bodies(
                    cast(LLIRStatementSequence, loop.body),
                    type_map,
                    context,
                    candidate_path + ("body",),
                ),
            )
            analysis = _analyze_loop(loop, context, candidate_path)
            if analysis is not None:
                _apply_loop_analysis(
                    loop,
                    analysis,
                    type_map,
                    context,
                    candidate_path,
                )
        transformed.append(candidate)

    # Match the legacy second pass.  Truthy declarations are consumed and
    # inserted immediately before their owner in reverse candidate order.  An
    # empty preexisting attribute remains attached.
    expanded: List[LLIRStatementValue] = []
    for candidate in transformed:
        if isinstance(candidate, llir.Stmt):
            declarations = getattr(candidate, "_hoisted_ptr_decls", None)
            if declarations:
                expanded.extend(reversed(declarations))
                delattr(candidate, "_hoisted_ptr_decls")
        expanded.append(candidate)

    if type(statements) is tuple:
        return tuple(expanded)
    return expanded


def hoist_dense_pointers(
    statements: List[llir.Stmt],
    context: DensePointerHoistContext,
) -> List[llir.Stmt]:
    """Return a detached statement-list root with legacy pointer hoisting.

    Unknown LLIR subclasses, malformed typed children, wrong root categories,
    and invalid immutable context snapshots fail through structured
    ``LLIRTraversalError`` diagnostics.  The pass stops on the first failure.
    """

    checked_context, type_map = _validate_context(context)
    checked_statements = _validate_root(statements, checked_context)
    detached = cast(
        List[LLIRStatementValue],
        _DensePointerDetacher(checked_context.traversal).rewrite(checked_statements),
    )
    transformed = _transform_direct_for_loop_bodies(
        detached,
        type_map,
        checked_context,
        ("root",),
    )
    return cast(List[llir.Stmt], transformed)
