"""Variables a lowered kernel reads outside the block that declares them.

A generated kernel that names a variable where no enclosing block declares it
does not compile.  The target lowering runs this check on the function it
returns and refuses such a kernel with a defect code
(``variable_read_out_of_scope``), so the failure is the compiler's own, named
exit and never an error from the C++ compiler at JIT time.

What counts as a declaration: ``VarInit``, ``VarDecl``, ``DirectInit``,
``FixedStackArrayDecl``, a ``for`` loop's init, a range-``for``'s variable, a
lambda's name and parameters, and a function's arguments.  Only names some node
of the function declares are checked: a name the function never declares (a
global, a macro, text inside a ``RawStmt``) is the C++ compiler's to resolve.

Block structure, as the C++ emitter renders it: a function body; a loop, whose
init is visible in its header and body; each branch of an ``if``; a lambda's
body.  A nested statement list is part of the block that holds it.  A ``for``
loop's ``before_parallel_body``, ``pre_parallel_body``, ``post_parallel_body``
and hoisted pointer declarations are counted in the block that holds the loop.
For the last three that is wider than the emitted text (they sit inside the
parallel region around the loop), so the check never reports a read the C++
compiler accepts; it can miss a read after the loop of a name those three
declare.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

from .. import llir
from ..llir_traversal import LLIRPath, LLIRTraversalContext, LLIRWalker

ScopeDefect = Tuple[str, LLIRPath]


def _argument_names(function: llir.Function) -> Set[str]:
    return {arg.name for arg in function.args if type(arg) is llir.Var}


class _DeclaredNames(LLIRWalker):
    """Every name some node of the walked function declares."""

    def __init__(self, context: LLIRTraversalContext) -> None:
        super().__init__(context)
        self.names: Set[str] = set()

    def visit_var_decl(self, node: llir.VarDecl, path: LLIRPath) -> None:
        self.names.add(node.var.name)
        super().visit_var_decl(node, path)

    def visit_var_init(self, node: llir.VarInit, path: LLIRPath) -> None:
        self.names.add(node.var.name)
        super().visit_var_init(node, path)

    def visit_direct_init(self, node: llir.DirectInit, path: LLIRPath) -> None:
        self.names.add(node.var.name)
        super().visit_direct_init(node, path)

    def visit_fixed_stack_array_decl(
        self, node: llir.FixedStackArrayDecl, path: LLIRPath
    ) -> None:
        self.names.add(node.name)
        super().visit_fixed_stack_array_decl(node, path)

    def visit_for_loop_auto(self, node: llir.ForLoopAuto, path: LLIRPath) -> None:
        self.names.add(node.var.name)
        super().visit_for_loop_auto(node, path)

    def visit_lambda_def(self, node: llir.LambdaDef, path: LLIRPath) -> None:
        self.names.add(node.var.name)
        self.names.update(param.name for param in node.params)
        super().visit_lambda_def(node, path)

    def visit_function(self, node: llir.Function, path: LLIRPath) -> None:
        self.names.update(_argument_names(node))
        super().visit_function(node, path)


class _ScopedReads(LLIRWalker):
    """Walk a function block by block and record each read out of scope."""

    def __init__(self, context: LLIRTraversalContext, declared: Set[str]) -> None:
        super().__init__(context)
        self.declared = declared
        self.blocks: List[Set[str]] = [set()]
        self.found: List[ScopeDefect] = []

    def _block(self, statements: Optional[Sequence[llir.Stmt]], path: LLIRPath) -> None:
        if statements is None:
            return
        self.blocks.append(set())
        try:
            self._walk_statements(statements, path)
        finally:
            self.blocks.pop()

    def visit_var(self, node: llir.Var, path: LLIRPath) -> None:
        super().visit_var(node, path)
        if node.name in self.declared and not any(
            node.name in block for block in self.blocks
        ):
            self.found.append((node.name, path))

    def visit_var_decl(self, node: llir.VarDecl, path: LLIRPath) -> None:
        self.blocks[-1].add(node.var.name)
        super().visit_var_decl(node, path)

    def visit_var_init(self, node: llir.VarInit, path: LLIRPath) -> None:
        self.blocks[-1].add(node.var.name)
        super().visit_var_init(node, path)

    def visit_direct_init(self, node: llir.DirectInit, path: LLIRPath) -> None:
        self.blocks[-1].add(node.var.name)
        super().visit_direct_init(node, path)

    def visit_fixed_stack_array_decl(
        self, node: llir.FixedStackArrayDecl, path: LLIRPath
    ) -> None:
        self.blocks[-1].add(node.name)
        super().visit_fixed_stack_array_decl(node, path)

    def visit_function(self, node: llir.Function, path: LLIRPath) -> None:
        # The arguments and the body are one block.
        self.blocks.append(_argument_names(node))
        try:
            super().visit_function(node, path)
        finally:
            self.blocks.pop()

    def visit_for_loop(self, node: llir.ForLoop, path: LLIRPath) -> None:
        self._walk_optional_statements(
            node.before_parallel_body, path + ("before_parallel_body",)
        )
        self._walk_optional_statements(
            node.pre_parallel_body, path + ("pre_parallel_body",)
        )
        if hasattr(node, "_hoisted_ptr_decls"):
            self._walk_statements(
                getattr(node, "_hoisted_ptr_decls"),
                path + ("_hoisted_ptr_decls",),
            )
        self.blocks.append(set())
        try:
            if node.init is not None:
                self._walk_for_loop_init(node.init, path + ("init",))
            self._walk_expr(node.cond, path + ("cond",))
            self._walk_for_loop_update(node.update, path + ("update",))
            self._block(node.body, path + ("body",))
        finally:
            self.blocks.pop()
        self._walk_optional_statements(
            node.post_parallel_body, path + ("post_parallel_body",)
        )

    def visit_for_loop_auto(self, node: llir.ForLoopAuto, path: LLIRPath) -> None:
        self._walk_expr(node.array, path + ("array",))
        self.blocks.append({node.var.name})
        try:
            self._walk_var_child(node.var, path + ("var",))
            self._block(node.body, path + ("body",))
        finally:
            self.blocks.pop()

    def visit_lambda_def(self, node: llir.LambdaDef, path: LLIRPath) -> None:
        self.blocks[-1].add(node.var.name)
        self._walk_var_child(node.var, path + ("var",))
        self.blocks.append({param.name for param in node.params})
        try:
            for index, param in enumerate(node.params):
                self._walk_var_child(param, path + ("params", str(index)))
            self._block(node.body, path + ("body",))
        finally:
            self.blocks.pop()

    def visit_while_loop(self, node: llir.WhileLoop, path: LLIRPath) -> None:
        self._walk_expr(node.cond, path + ("cond",))
        self._block(node.body, path + ("body",))

    def visit_if_then_else(self, node: llir.IfThenElse, path: LLIRPath) -> None:
        if node.cond is not None:
            self._walk_expr(node.cond, path + ("cond",))
        self._block(node.then_body, path + ("then_body",))
        if node.cond_list is not None:
            self._walk_expr_sequence(node.cond_list, path + ("cond_list",))
        if node.then_body_list is not None:
            for index, branch in enumerate(node.then_body_list):
                self._block(branch, path + ("then_body_list", f"[{index}]"))
        self._block(node.else_body, path + ("else_body",))


def reads_out_of_scope(function: llir.Function) -> Tuple[ScopeDefect, ...]:
    """``(name, path)`` for every read of a declared name outside its block."""

    context = LLIRTraversalContext(
        stage="LoopIR target lowering", pass_name="reads_out_of_scope"
    )
    declared = _DeclaredNames(context)
    declared.walk(function)
    reads = _ScopedReads(context, declared.names)
    reads.walk(function)
    return tuple(reads.found)
