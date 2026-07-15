"""Checked dynamic-vector access rewriting over the typed LLIR boundary."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Sequence, Tuple, cast

from . import llir
from .llir_traversal import (
    LLIRPath,
    LLIRRewriteValueT,
    LLIRRewriter,
    LLIRStatementSequence,
    LLIRStatementValue,
    LLIRTraversalContext,
    LLIRValue,
    LLIRWalker,
)


@dataclass(frozen=True)
class DynamicVectorAccessConfig:
    """Immutable policy for generated dynamic-vector access rewriting."""

    vector_type_prefix: str
    append_suffixes: Tuple[str, ...]
    deduplicate_suffixes: Tuple[str, ...]
    append_method: str
    checked_set_function: str


@dataclass(frozen=True)
class DynamicVectorAccessContext:
    """All explicit input other than the LLIR value transformed by the pass."""

    traversal: LLIRTraversalContext
    config: DynamicVectorAccessConfig


DYNAMIC_VECTOR_ACCESS_CONTEXT = DynamicVectorAccessContext(
    traversal=LLIRTraversalContext(
        stage="LLIR rewrite",
        pass_name="rewrite_dynamic_vector_accesses",
    ),
    config=DynamicVectorAccessConfig(
        vector_type_prefix="std::vector<",
        append_suffixes=("_crd", "_values"),
        deduplicate_suffixes=("_crd",),
        append_method="emplace_back",
        checked_set_function="scorch_vector_set",
    ),
)


class _DynamicVectorDeclarationCollector(LLIRWalker):
    def __init__(self, context: DynamicVectorAccessContext) -> None:
        super().__init__(context.traversal)
        self._config = context.config
        self._names: List[str] = []
        self._seen: set[str] = set()

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self._names)

    def visit_var_decl(self, node: llir.VarDecl, path: LLIRPath) -> None:
        # Validate and walk the typed child before using declaration metadata so
        # malformed nodes fail through the shared structured diagnostic path.
        super().visit_var_decl(node, path)
        if (
            node.var.type.value.startswith(self._config.vector_type_prefix)
            and node.var.name not in self._seen
        ):
            self._seen.add(node.var.name)
            self._names.append(node.var.name)


class _DynamicVectorAccessRewriter(LLIRRewriter):
    def __init__(
        self,
        context: DynamicVectorAccessContext,
        vector_names: Tuple[str, ...],
    ) -> None:
        super().__init__(context.traversal)
        self._config = context.config
        self._vector_names = frozenset(vector_names)
        self._read_patterns = tuple(
            (
                vector_name,
                re.compile(rf"\b{re.escape(vector_name)}\[([^\[\]]+)\]"),
                rf"{vector_name}.at(\1)",
            )
            for vector_name in vector_names
        )

    def _match_store(self, expression: llir.Expr) -> Optional[Tuple[str, llir.Expr]]:
        if type(expression) is not llir.ArrayAccess:
            return None
        access = cast(llir.ArrayAccess, expression)
        if type(access.array) is not llir.Var:
            return None
        vector_name = cast(llir.Var, access.array).name
        if vector_name not in self._vector_names:
            return None
        return vector_name, access.index

    def _rewrite_name(self, name: str) -> str:
        if "[" not in name:
            return name
        for vector_name, pattern, replacement in self._read_patterns:
            if vector_name in name:
                name = pattern.sub(replacement, name)
        return name

    def prepare_statement_sequence(
        self, statements: LLIRStatementSequence, path: LLIRPath
    ) -> Sequence[LLIRStatementValue]:
        deduplicated: List[LLIRStatementValue] = []
        for candidate in statements:
            matched = (
                self._match_store(candidate.var)
                if type(candidate) is llir.Assign
                else None
            )
            if (
                matched is not None
                and cast(llir.Assign, candidate).op == llir.AssignOp.ASSIGN
                and matched[0].endswith(self._config.deduplicate_suffixes)
                and deduplicated
                and type(deduplicated[-1]) is llir.Assign
                and candidate == deduplicated[-1]
            ):
                continue
            deduplicated.append(candidate)
        return deduplicated

    def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
        rewritten = super().rewrite_var(node, path)
        rewritten.name = self._rewrite_name(rewritten.name)
        return rewritten

    def rewrite_assign(self, node: llir.Assign, path: LLIRPath) -> llir.Stmt:
        if path and path[-1] == "update":
            # A ForLoop update is a scalar header expression.  The legacy pass
            # rewrote its operands but never replaced the Assign with a call
            # statement, which would not be a legal update node.
            return super().rewrite_assign(node, path)

        matched = self._match_store(node.var)
        value = self._rewrite_expr(node.value, path + ("value",))
        if matched is not None and node.op == llir.AssignOp.ASSIGN:
            vector_name, position = matched
            if vector_name.endswith(self._config.append_suffixes):
                return llir.FunctionCallStmt(
                    name=(f"{vector_name}.{self._config.append_method}"),
                    args=[value],
                )

            return llir.FunctionCallStmt(
                name=self._config.checked_set_function,
                args=[
                    llir.Var(name=vector_name, type=llir.DataType.NO_TYPE),
                    self._rewrite_expr(position, path + ("var", "index")),
                    value,
                ],
            )

        rewritten = llir.Assign(
            var=self._rewrite_assignment_target(node.var, path + ("var",)),
            value=value,
            op=node.op,
            cast=False,
        )
        rewritten.cast = node.cast
        return rewritten


def rewrite_dynamic_vector_accesses(
    value: LLIRRewriteValueT, context: DynamicVectorAccessContext
) -> LLIRRewriteValueT:
    """Return a detached LLIR value with checked dynamic-vector accesses.

    Only vectors declared by ``VarDecl`` are dynamic.  Pre-sized vectors use
    ``VarInit`` and intentionally stay on the indexed schedule-storage path.
    """

    collector = _DynamicVectorDeclarationCollector(context)
    llir_value = cast(LLIRValue, value)
    collector.walk(llir_value)
    vector_names = collector.names
    if not vector_names:
        return LLIRRewriter(context.traversal).rewrite(value)
    return _DynamicVectorAccessRewriter(context, vector_names).rewrite(value)
