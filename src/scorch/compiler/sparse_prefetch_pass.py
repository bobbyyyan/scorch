"""Typed sparse-prefetch insertion over a detached LLIR statement list.

Version 1 intentionally preserves the generated-name contracts of the legacy
``CINLowerer`` optimization.  Sparse position loops are recognized through
``_pos[`` text, coordinates through structured ``ArrayAccess`` nodes, dense
value accesses through ``_val`` spellings, and already-hoisted value pointers
through the exact legacy ``RawStmt`` declaration.  The inserted target spelling
and next-position distance are fixed as ``__builtin_prefetch(..., 0, 1)``.
These are compatibility dependencies, not configurable policy.

The common LLIR boundary validates and detaches the complete input tree.  The
semantic scan remains deliberately narrower: it follows only chains of direct
``ForLoop.body`` children, processes nested loops before their containing loop,
and does not enter conditionals, switches, functions, or auxiliary parallel
regions.  The root must be an exact list; nested list/tuple members admitted by
the current LLIR statement-sequence contract are validated and detached but are
also semantically omitted.  A legal structural miss is a detached no-op.
Reapplying the complete pass inserts another identical group of prefetch
statements, matching the characterized non-idempotent legacy behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, NoReturn, Sequence, Tuple, cast

from . import llir
from .llir_traversal import (
    LLIRPath,
    LLIRRewriter,
    LLIRStatementValue,
    LLIRTraversalContext,
    LLIRTraversalDiagnostic,
    LLIRTraversalError,
)

SPARSE_PREFETCH_TRAVERSAL_CONTEXT = LLIRTraversalContext(
    stage="LLIR transformation",
    pass_name="insert_sparse_prefetch",
)


@dataclass(frozen=True)
class SparsePrefetchContext:
    """Immutable diagnostic identity for the fixed version-1 transformation.

    The pass deliberately exposes no policy fields: generated-name matching,
    prefetch distance, access mode, and locality are fixed legacy contracts.
    """

    traversal: LLIRTraversalContext = SPARSE_PREFETCH_TRAVERSAL_CONTEXT


SPARSE_PREFETCH_CONTEXT = SparsePrefetchContext()

_DenseArrayStride = Tuple[str, str]
_STRING_VALUE_ACCESS = re.compile(r"^(\w+_val)\[(\w+)\]$")
_HOISTED_VALUE_POINTER = re.compile(
    r"const (?:float|double)\* __restrict__ "
    r"_(\w+_val)_ptr = &(\w+_val)\[(\w+) \* (\w+)\]"
)


def _diagnostic_context(context: object) -> LLIRTraversalContext:
    if type(context) is SparsePrefetchContext:
        traversal = cast(SparsePrefetchContext, context).traversal
        if type(traversal) is LLIRTraversalContext:
            return traversal
    return SPARSE_PREFETCH_TRAVERSAL_CONTEXT


def _raise_sparse_prefetch_error(
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


def _validate_context(context: object) -> SparsePrefetchContext:
    if type(context) is not SparsePrefetchContext:
        _raise_sparse_prefetch_error(
            context,
            code="invalid_sparse_prefetch_context",
            message="expected an immutable SparsePrefetchContext",
            path=("context",),
            value=context,
        )

    typed_context = cast(SparsePrefetchContext, context)
    traversal = typed_context.traversal
    if (
        type(traversal) is not LLIRTraversalContext
        or type(traversal.stage) is not str
        or not traversal.stage
        or type(traversal.pass_name) is not str
        or not traversal.pass_name
    ):
        _raise_sparse_prefetch_error(
            SPARSE_PREFETCH_TRAVERSAL_CONTEXT,
            code="invalid_sparse_prefetch_traversal_context",
            message="traversal stage and pass name must be non-empty strings",
            path=("context", "traversal"),
            value=traversal,
        )
    return typed_context


def _validate_root(
    statements: object, context: SparsePrefetchContext
) -> List[LLIRStatementValue]:
    if type(statements) is not list:
        _raise_sparse_prefetch_error(
            context,
            code="unsupported_sparse_prefetch_root",
            message="sparse-prefetch insertion requires a statement-list root",
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
            _raise_sparse_prefetch_error(
                context,
                code="invalid_sparse_prefetch_root_member",
                message=(
                    "the top-level list may contain only LLIR statements or "
                    "nested statement lists/tuples"
                ),
                path=("root", f"[{index}]"),
                value=statement,
            )
    return typed_statements


def _checked_name(
    node: llir.Var,
    context: SparsePrefetchContext,
    path: LLIRPath,
) -> str:
    if type(node.name) is not str:
        _raise_sparse_prefetch_error(
            context,
            code="invalid_sparse_prefetch_var_name",
            message="a matched LLIR Var name must be a string",
            path=path + ("name",),
            value=node.name,
        )
    return node.name


def _append_dense_array(
    results: List[_DenseArrayStride],
    pair: _DenseArrayStride,
) -> None:
    if pair not in results:
        results.append(pair)


def _find_all_val_array_accesses(
    expression: llir.Expr,
    position_to_stride: Dict[str, str],
    results: List[_DenseArrayStride],
    context: SparsePrefetchContext,
    path: LLIRPath,
) -> None:
    """Collect legacy value-array matches in BinOp left-to-right order."""

    if type(expression) is llir.Var:
        variable = cast(llir.Var, expression)
        match = _STRING_VALUE_ACCESS.match(_checked_name(variable, context, path))
        if match:
            array_name, position = match.group(1), match.group(2)
            if position in position_to_stride:
                _append_dense_array(
                    results,
                    (array_name, position_to_stride[position]),
                )
    if type(expression) in (llir.BinOp, llir.Add, llir.Mul):
        binary = cast(llir.BinOp, expression)
        _find_all_val_array_accesses(
            binary.left,
            position_to_stride,
            results,
            context,
            path + ("left",),
        )
        _find_all_val_array_accesses(
            binary.right,
            position_to_stride,
            results,
            context,
            path + ("right",),
        )
    if type(expression) is llir.ArrayAccess:
        access = cast(llir.ArrayAccess, expression)
        if (
            type(access.array) is llir.Var
            and "_val"
            in _checked_name(cast(llir.Var, access.array), context, path + ("array",))
            and type(access.index) is llir.Var
        ):
            array = cast(llir.Var, access.array)
            index = cast(llir.Var, access.index)
            position = _checked_name(index, context, path + ("index",))
            if position in position_to_stride:
                _append_dense_array(
                    results,
                    (
                        _checked_name(array, context, path + ("array",)),
                        position_to_stride[position],
                    ),
                )


def _coordinate_array(
    loop: llir.ForLoop,
    iterator: str,
    context: SparsePrefetchContext,
    path: LLIRPath,
) -> str | None:
    for index, statement in enumerate(loop.body):
        if type(statement) is not llir.VarInit:
            continue
        value = statement.value
        if type(value) is not llir.ArrayAccess:
            continue
        access = cast(llir.ArrayAccess, value)
        if type(access.array) is not llir.Var or type(access.index) is not llir.Var:
            continue
        array = cast(llir.Var, access.array)
        access_index = cast(llir.Var, access.index)
        array_name = _checked_name(
            array,
            context,
            path + ("body", f"[{index}]", "value", "array"),
        )
        index_name = _checked_name(
            access_index,
            context,
            path + ("body", f"[{index}]", "value", "index"),
        )
        if (
            access.tensor_access is None
            and array.type is llir.DataType.PTR_INT
            and array.tensor_access is None
            and array_name.isidentifier()
            and array_name.endswith("_crd")
            and access_index.type is llir.DataType.INT
            and access_index.tensor_access is None
            and index_name == iterator
        ):
            return array_name

    return None


def _assignment_dense_arrays(
    loop: llir.ForLoop,
    context: SparsePrefetchContext,
    path: LLIRPath,
) -> List[_DenseArrayStride]:
    dense_arrays: List[_DenseArrayStride] = []
    for body_index, body_statement in enumerate(loop.body):
        if type(body_statement) is not llir.ForLoop:
            continue
        inner_loop = cast(llir.ForLoop, body_statement)

        inner_path = path + ("body", f"[{body_index}]")
        position_to_stride: Dict[str, str] = {}
        for inner_index, inner_statement in enumerate(inner_loop.body):
            if (
                type(inner_statement) is llir.VarInit
                and type(inner_statement.value) is llir.Add
            ):
                add = inner_statement.value
                if type(add.left) not in (llir.BinOp, llir.Add, llir.Mul):
                    continue
                multiply = cast(llir.BinOp, add.left)
                if multiply.op == "*" and type(multiply.right) is llir.Var:
                    position = _checked_name(
                        inner_statement.var,
                        context,
                        inner_path + ("body", f"[{inner_index}]", "var"),
                    )
                    stride = _checked_name(
                        cast(llir.Var, multiply.right),
                        context,
                        inner_path
                        + ("body", f"[{inner_index}]", "value", "left", "right"),
                    )
                    position_to_stride[position] = stride

        for inner_index, inner_statement in enumerate(inner_loop.body):
            if type(inner_statement) is not llir.Assign:
                continue
            _find_all_val_array_accesses(
                inner_statement.value,
                position_to_stride,
                dense_arrays,
                context,
                inner_path + ("body", f"[{inner_index}]", "value"),
            )
    return dense_arrays


def _augment_from_hoisted_pointers(
    loop: llir.ForLoop,
    dense_arrays: List[_DenseArrayStride],
    context: SparsePrefetchContext,
    path: LLIRPath,
) -> None:
    # Legacy behavior checks raw pointers only after an assignment-based match.
    if not dense_arrays:
        return

    for index, statement in enumerate(loop.body):
        if type(statement) is not llir.RawStmt:
            continue
        statement = cast(llir.RawStmt, statement)
        code = statement.code
        if type(code) is not str:
            _raise_sparse_prefetch_error(
                context,
                code="invalid_sparse_prefetch_raw_statement",
                message="a matched LLIR RawStmt code field must be a string",
                path=path + ("body", f"[{index}]", "code"),
                value=code,
            )
        if "_ptr" not in code:
            continue
        match = _HOISTED_VALUE_POINTER.match(code)
        if match:
            _append_dense_array(dense_arrays, (match.group(2), match.group(4)))


def _prepend_prefetches(
    loop: llir.ForLoop,
    iterator: str,
    end: str,
    coordinate_array: str,
    dense_arrays: Sequence[_DenseArrayStride],
) -> None:
    prefetches: List[llir.Stmt] = []
    seen: set[_DenseArrayStride] = set()
    for value_array, stride in dense_arrays:
        key = (value_array, stride)
        if key in seen:
            continue
        seen.add(key)
        prefetches.append(
            llir.RawStmt(
                code=(
                    f"if ({iterator} + 1 < {end}) "
                    f"__builtin_prefetch(&{value_array}["
                    f"{coordinate_array}[{iterator} + 1] * {stride}], 0, 1)"
                ),
                add_semicolon=True,
            )
        )

    if type(loop.body) is list:
        for statement in reversed(prefetches):
            loop.body.insert(0, statement)
        return

    # The common traversal contract also admits tuple statement children.  Keep
    # that child-container category while preserving the same insertion order.
    loop.body = cast(List[llir.Stmt], tuple(prefetches) + tuple(loop.body))


def _insert_in_for_loop_bodies(
    statements: Sequence[object],
    context: SparsePrefetchContext,
    path: LLIRPath,
) -> None:
    for index, candidate in enumerate(statements):
        if type(candidate) is not llir.ForLoop:
            continue
        loop = cast(llir.ForLoop, candidate)
        loop_path = path + (f"[{index}]",)

        # Post-order recursion is intentionally limited to the main body.
        _insert_in_for_loop_bodies(loop.body, context, loop_path + ("body",))

        if not (
            type(loop.init) is llir.VarInit
            and type(loop.init.value) is llir.Var
            and "_pos["
            in _checked_name(
                cast(llir.Var, loop.init.value),
                context,
                loop_path + ("init", "value"),
            )
        ):
            continue

        iterator = _checked_name(
            loop.init.var,
            context,
            loop_path + ("init", "var"),
        )
        if type(loop.cond) not in (llir.BinOp, llir.Add, llir.Mul):
            continue
        condition = cast(llir.BinOp, loop.cond)
        if type(condition.right) is not llir.Var:
            continue
        end = _checked_name(
            cast(llir.Var, condition.right),
            context,
            loop_path + ("cond", "right"),
        )

        coordinate_array = _coordinate_array(loop, iterator, context, loop_path)
        if coordinate_array is None:
            continue

        dense_arrays = _assignment_dense_arrays(loop, context, loop_path)
        if not dense_arrays:
            continue
        _augment_from_hoisted_pointers(loop, dense_arrays, context, loop_path)
        _prepend_prefetches(
            loop,
            iterator,
            end,
            coordinate_array,
            dense_arrays,
        )


def insert_sparse_prefetch(
    statements: List[llir.Stmt],
    context: SparsePrefetchContext,
) -> List[llir.Stmt]:
    """Return a detached statement-list root with legacy sparse prefetches.

    Unknown LLIR subclasses, malformed typed children, other root categories,
    and invalid contexts fail through structured ``LLIRTraversalError``
    diagnostics.  A valid unmatched input returns an equally detached no-op.
    """

    checked_context = _validate_context(context)
    checked_statements = _validate_root(statements, checked_context)
    detached = cast(
        List[LLIRStatementValue],
        LLIRRewriter(checked_context.traversal).rewrite(checked_statements),
    )
    _insert_in_for_loop_bodies(detached, checked_context, ("root",))
    return cast(List[llir.Stmt], detached)
