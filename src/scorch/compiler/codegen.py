from typing import TYPE_CHECKING, List, Optional, TypeVar, Union, cast

from . import llir
from .diagnostics import CodegenError

if TYPE_CHECKING:
    from .compile_options import CompileOptions

LLIR_NODE = TypeVar("LLIR_NODE", bound=llir.Node)


class LLIRLowerer:
    """
    This is a class to lower LLIR to C++ code (string).
    """

    indent_str = "  "
    indent_level = 0
    no_comments = False

    # Higher values bind more tightly.  The table intentionally contains only
    # operators represented by the current LLIR; accepting arbitrary spelling
    # here would let malformed IR bypass the codegen boundary.
    _BINARY_PRECEDENCE = {
        "||": 1,
        "&&": 2,
        "|": 3,
        "^": 4,
        "&": 5,
        "==": 6,
        "!=": 6,
        "<": 7,
        "<=": 7,
        ">": 7,
        ">=": 7,
        "<<": 8,
        ">>": 8,
        "+": 9,
        "-": 9,
        "*": 10,
        "/": 10,
        "%": 10,
    }
    _UNARY_PRECEDENCE = 11
    _POSTFIX_PRECEDENCE = 12
    _PRIMARY_PRECEDENCE = 13
    _UNARY_OPERATORS = {"+", "-", "!", "~", "*", "&", "++", "--"}

    def __init__(self, compile_options: Optional["CompileOptions"] = None) -> None:
        """Bind code emission to the compilation's immutable options snapshot.

        Standalone LLIR rendering remains available without constructing compiler
        options. Production compilation paths pass the snapshot explicitly.
        """

        if compile_options is not None:
            from .compile_options import CompileOptions

            if type(compile_options) is not CompileOptions:
                raise TypeError("compile_options must be a CompileOptions instance")
        self.compile_options = compile_options
        self.no_comments = (
            not compile_options.emit_comments if compile_options is not None else False
        )

    @staticmethod
    def _lower_typed_var(var: llir.Var) -> str:
        qualifier = "__restrict__ " if var.is_restrict else ""
        return f"{var.type.value} {qualifier}{var.name}"

    def lower_llir(
        self,
        ir: Union[LLIR_NODE, List[LLIR_NODE], str, List[str]],
        indent_level: int = 0,
        no_semicolon: bool = False,
        no_comments: bool = False,
    ) -> str:
        if no_comments:
            if self.compile_options is not None and self.compile_options.emit_comments:
                from .diagnostics import (
                    CompileOptionsDiagnostic,
                    CompileOptionsError,
                )

                raise CompileOptionsError(
                    (
                        CompileOptionsDiagnostic(
                            code="conflicting_emission_policy",
                            field="no_comments",
                            message=(
                                "an explicit CompileOptions snapshot requires "
                                "comment emission"
                            ),
                        ),
                    )
                )
            self.no_comments = True

        if isinstance(ir, str):
            return indent_level * self.indent_str + ir

        elif isinstance(ir, list):
            lines = [self.lower_llir(node, indent_level) for node in ir]
            lines = [line for line in lines if line != ""]
            return "\n".join(lines)

        elif isinstance(ir, llir.Comment):
            if self.no_comments:
                return ""
            return self.lower_llir(f"// {ir.value}", indent_level)

        elif isinstance(ir, llir.BlankLine):
            return self.lower_llir(" ", indent_level)

        elif isinstance(ir, llir.VarInit):
            return self.lower_llir(
                f"{self._lower_typed_var(ir.var)} {ir.op} {self._render_expression(ir.value)};",
                indent_level,
            )

        elif isinstance(ir, llir.Assign):
            if not isinstance(ir.var, llir.Var):
                raise CodegenError(
                    "LLIR codegen requires Assign.var to be Var, got "
                    f"{type(ir.var).__name__}"
                )
            if no_semicolon:
                return self.lower_llir(
                    f"{ir.var.name} {ir.op.value} {self._render_expression(ir.value)}",
                    indent_level,
                )
            return self.lower_llir(
                f"{ir.var.name} {ir.op.value} {self._render_expression(ir.value)};",
                indent_level,
            )

        elif isinstance(ir, llir.Expr):
            return self.lower_expression(ir, indent_level)

        elif isinstance(ir, llir.FunctionCallStmt):
            return self.lower_llir(
                f"{ir.name}({', '.join(self._render_expression(arg) for arg in ir.args)});",
                indent_level,
            )

        elif isinstance(ir, (llir.WhileLoop, llir.ForLoop, llir.ForLoopAuto)):
            return self.lower_loop_construct(ir, indent_level)

        elif isinstance(ir, llir.IfThenElse):
            return self.lower_conditional(ir, indent_level)

        elif isinstance(ir, llir.VarDecl):
            return self.lower_llir(f"{self._lower_typed_var(ir.var)};", indent_level)

        elif isinstance(ir, llir.RawStmt):
            suffix = ";" if ir.add_semicolon else ""
            return self.lower_llir(f"{ir.code}{suffix}", indent_level)

        elif isinstance(ir, llir.Break):
            return self.lower_llir("break;", indent_level)

        elif isinstance(ir, llir.Continue):
            return self.lower_llir("continue;", indent_level)

        elif isinstance(ir, llir.Increment):
            if no_semicolon:
                return self.lower_llir(f"{ir.var.name}++", indent_level)
            return self.lower_llir(f"{ir.var.name}++;", indent_level)

        elif isinstance(ir, llir.Function):
            return self.lower_function_definition(ir, indent_level)

        elif isinstance(ir, llir.Return):
            return self.lower_llir(
                f"return {self._render_expression(ir.value)};", indent_level
            )

        raise CodegenError(
            f"No C++ codegen implemented for LLIR node type: {type(ir).__name__}"
        )

    def lower_expression(
        self,
        ir: llir.Expr,
        indent_level: int = 0,
    ) -> str:
        return self.lower_llir(self._render_expression(ir), indent_level)

    def _render_expression(self, ir: llir.Expr) -> str:
        """Render an expression while preserving the LLIR expression tree.

        Parentheses are emitted only when C++ would otherwise parse a different
        tree.  In particular, a right child at the same precedence must remain
        parenthesized because all currently supported binary operators are
        left-associative and floating-point reassociation is not semantics
        preserving.
        """
        if isinstance(ir, llir.Literal):
            return str(ir.value)

        if isinstance(ir, llir.Var):
            return ir.name

        if isinstance(ir, llir.Cast):
            operand = self._render_operand(
                ir.expr,
                parent_precedence=self._UNARY_PRECEDENCE,
                is_right_child=True,
            )
            return f"({ir.data_type.value}) {operand}"

        if isinstance(ir, llir.Sizeof):
            return f"sizeof({ir.data_type.value})"

        if isinstance(ir, llir.BinOp):
            precedence = self._binary_precedence(ir.op)
            left = self._render_operand(
                ir.left,
                parent_precedence=precedence,
                is_right_child=False,
            )
            right = self._render_operand(
                ir.right,
                parent_precedence=precedence,
                is_right_child=True,
            )
            return f"{left} {ir.op} {right}"

        if isinstance(ir, llir.UnaryOp):
            if ir.op not in self._UNARY_OPERATORS:
                raise CodegenError(f"Unsupported LLIR unary operator: {ir.op!r}")
            operand = self._render_operand(
                ir.operand,
                parent_precedence=self._UNARY_PRECEDENCE,
                is_right_child=True,
            )
            return f"{ir.op} {operand}"

        if isinstance(ir, llir.FunctionCall):
            return (
                f"{ir.name}"
                f"({', '.join(self._render_expression(arg) for arg in ir.args)})"
            )

        if isinstance(ir, llir.Array):
            return "{" + ", ".join(self._render_expression(v) for v in ir.values) + "}"

        if type(ir) is llir.ArrayAccess:
            array = self._render_operand(
                ir.array,
                parent_precedence=self._POSTFIX_PRECEDENCE,
                is_right_child=False,
            )
            return f"{array}[{self._render_expression(ir.index)}]"

        raise CodegenError(
            f"No C++ codegen implemented for LLIR expression type: {type(ir).__name__}"
        )

    def _render_operand(
        self,
        ir: llir.Expr,
        *,
        parent_precedence: int,
        is_right_child: bool,
    ) -> str:
        rendered = self._render_expression(ir)
        child_precedence = self._expression_precedence(ir)
        needs_parentheses = child_precedence < parent_precedence

        if isinstance(ir, llir.BinOp) and child_precedence == parent_precedence:
            # The left child naturally parses first for left-associative binary
            # operators.  The right child does not, regardless of whether its
            # operator spelling matches the parent.
            needs_parentheses = is_right_child

        if needs_parentheses:
            return f"({rendered})"
        return rendered

    def _expression_precedence(self, ir: llir.Expr) -> int:
        if isinstance(ir, llir.BinOp):
            return self._binary_precedence(ir.op)
        if isinstance(ir, (llir.Cast, llir.UnaryOp, llir.Sizeof)):
            return self._UNARY_PRECEDENCE
        if isinstance(ir, llir.FunctionCall) or type(ir) is llir.ArrayAccess:
            return self._POSTFIX_PRECEDENCE
        if isinstance(ir, (llir.Literal, llir.Var, llir.Array)):
            return self._PRIMARY_PRECEDENCE
        raise CodegenError(
            f"No C++ precedence defined for LLIR expression type: {type(ir).__name__}"
        )

    def _binary_precedence(self, op: str) -> int:
        try:
            return self._BINARY_PRECEDENCE[op]
        except KeyError as exc:
            raise CodegenError(f"Unsupported LLIR binary operator: {op!r}") from exc

    @staticmethod
    def _omp_num_threads_clause(ir: "llir.ForLoop") -> str:
        """`` num_threads(<expr>)`` when a work-aware thread cap is set, else ``""``."""
        expr = getattr(ir, "omp_num_threads", None)
        return f" num_threads({expr})" if expr else ""

    @staticmethod
    def _omp_schedule_clause(ir: "llir.ForLoop") -> str:
        """`` schedule(...)`` clause. An adaptive chunk expr (omp_chunk_expr)
        overrides the static chunk baked into omp_schedule."""
        chunk_expr = getattr(ir, "omp_chunk_expr", None)
        if chunk_expr:
            return f" schedule(dynamic, {chunk_expr})"
        if ir.omp_schedule:
            return f" schedule({ir.omp_schedule})"
        return ""

    def lower_loop_construct(
        self,
        ir: Union[llir.WhileLoop, llir.ForLoop, llir.ForLoopAuto],
        indent_level: int = 0,
    ) -> str:
        pragma_lines: List[str] = []
        if isinstance(ir, llir.WhileLoop):
            header = f"while ({self.lower_llir(ir.cond)}) {{"
        elif isinstance(ir, llir.ForLoop):
            # Atomic work-stealing: replace for loop with while + atomic counter
            if getattr(ir, "_use_atomic_scheduling", False):
                chunk_var = ir._atomic_chunk_var
                counter_var = ir._atomic_counter_var
                loop_bound = ir._loop_bound
                loop_var = ir.init.var.name if ir.init else "i"
                parts = []
                if ir.before_parallel_body:
                    parts.append(self.lower_llir(ir.before_parallel_body, indent_level))
                parts.extend(
                    [
                        self.lower_llir(
                            f"std::atomic<int> {counter_var}{{0}};", indent_level
                        ),
                        self.lower_llir(
                            "#pragma omp parallel" + self._omp_num_threads_clause(ir),
                            indent_level,
                        ),
                        self.lower_llir("{", indent_level),
                    ]
                )
                if ir.pre_parallel_body:
                    parts.append(
                        self.lower_llir(ir.pre_parallel_body, indent_level + 1)
                    )
                # Atomic while loop
                parts.append(self.lower_llir(f"while (true) {{", indent_level + 1))
                parts.append(
                    self.lower_llir(
                        f"const int _start = {counter_var}.fetch_add({chunk_var}, std::memory_order_relaxed);",
                        indent_level + 2,
                    )
                )
                parts.append(
                    self.lower_llir(
                        f"if (_start >= {loop_bound}) break;", indent_level + 2
                    )
                )
                parts.append(
                    self.lower_llir(
                        f"const int _end = std::min(_start + {chunk_var}, {loop_bound});",
                        indent_level + 2,
                    )
                )
                parts.append(
                    self.lower_llir(
                        f"for (int {loop_var} = _start; {loop_var} < _end; {loop_var}++) {{",
                        indent_level + 2,
                    )
                )
                parts.append(self.lower_llir(ir.body, indent_level + 3))
                parts.append(self.lower_llir("}", indent_level + 2))  # close for
                parts.append(self.lower_llir("}", indent_level + 1))  # close while
                if ir.post_parallel_body:
                    parts.append(
                        self.lower_llir(ir.post_parallel_body, indent_level + 1)
                    )
                parts.append(self.lower_llir("}", indent_level))  # close parallel
                return "\n".join(parts)

            # When pre/post_parallel_body is set, split into:
            #   #pragma omp parallel { pre; #pragma omp for ...; post }
            if ir.omp_parallel_for and (ir.pre_parallel_body or ir.post_parallel_body):
                omp_for = "#pragma omp for" + self._omp_schedule_clause(ir)
                init_lowered = self.lower_llir(ir.init) if ir.init is not None else ";"
                for_header = (
                    f"for ({init_lowered} {self.lower_llir(ir.cond)};"
                    f" {self.lower_llir(ir.update, no_semicolon=True)}) {{"
                )
                parts = []
                if ir.before_parallel_body:
                    parts.append(self.lower_llir(ir.before_parallel_body, indent_level))
                parts.extend(
                    [
                        self.lower_llir(
                            "#pragma omp parallel" + self._omp_num_threads_clause(ir),
                            indent_level,
                        ),
                        self.lower_llir("{", indent_level),
                    ]
                )
                if ir.pre_parallel_body:
                    parts.append(
                        self.lower_llir(ir.pre_parallel_body, indent_level + 1)
                    )
                parts.append(self.lower_llir(omp_for, indent_level + 1))
                parts.append(self.lower_llir(for_header, indent_level + 1))
                parts.append(self.lower_llir(ir.body, indent_level + 2))
                parts.append(self.lower_llir("}", indent_level + 1))
                if ir.post_parallel_body:
                    parts.append(
                        self.lower_llir(ir.post_parallel_body, indent_level + 1)
                    )
                parts.append(self.lower_llir("}", indent_level))
                return "\n".join(parts)

            if ir.omp_parallel_for:
                omp_pragma = (
                    "#pragma omp parallel for"
                    + self._omp_num_threads_clause(ir)
                    + self._omp_schedule_clause(ir)
                )
                pragma_lines.append(omp_pragma)
            if ir.unroll:
                pragma_lines.append("#pragma unroll")
            if ir.simd:
                pragma_lines.append("#pragma omp simd")
            init_lowered = self.lower_llir(ir.init) if ir.init is not None else ";"
            header = (
                f"for ({init_lowered} {self.lower_llir(ir.cond)};"
                f" {self.lower_llir(ir.update, no_semicolon=True)}) {{"
            )
        elif isinstance(ir, llir.ForLoopAuto):
            header = (
                f"for ({ir.var.type.value} {self.lower_llir(ir.var)}"
                f" : {self.lower_llir(ir.array)}) {{"
            )
        else:
            raise ValueError(f"Unknown loop type: {type(ir)}")

        loop_text = (
            self.lower_llir(header, indent_level)
            + "\n"
            + self.lower_llir(ir.body, indent_level + 1)
            + "\n"
            + self.lower_llir("}", indent_level)
        )
        if not pragma_lines:
            return loop_text

        pragma_text = "\n".join(
            self.lower_llir(pragma_line, indent_level) for pragma_line in pragma_lines
        )
        before_parallel = getattr(ir, "before_parallel_body", None)
        if before_parallel:
            return (
                self.lower_llir(before_parallel, indent_level)
                + "\n"
                + pragma_text
                + "\n"
                + loop_text
            )
        return pragma_text + "\n" + loop_text

    def lower_conditional(self, ir: llir.IfThenElse, indent_level: int = 0) -> str:
        result = ""
        if ir.cond_list:
            if not ir.then_body_list:
                raise CodegenError(
                    "LLIR IfThenElse with cond_list requires then_body_list"
                )
            if len(ir.cond_list) != len(ir.then_body_list):
                raise CodegenError(
                    "LLIR IfThenElse condition and body counts must match"
                )

            total_num_conds = len(ir.cond_list) + (1 if ir.else_body else 0)

            for i, cond in enumerate(ir.cond_list):
                if i == 0:
                    result += (
                        self.lower_llir(
                            f"if ({self.lower_llir(cond)}) {{", indent_level
                        )
                        + "\n"
                        + self.lower_llir(ir.then_body_list[i], indent_level + 1)
                        + "\n"
                    )
                elif ir.make_last_case_else and i == total_num_conds - 1:
                    result += (
                        self.lower_llir("} else {", indent_level)
                        + "\n"
                        + self.lower_llir(ir.then_body_list[i], indent_level + 1)
                        + "\n"
                    )
                else:
                    result += (
                        self.lower_llir(
                            f"}} else if ({self.lower_llir(cond)}) {{", indent_level
                        )
                        + "\n"
                        + self.lower_llir(ir.then_body_list[i], indent_level + 1)
                        + "\n"
                    )
        else:
            if ir.cond is None:
                raise CodegenError("LLIR IfThenElse requires a condition")
            if not ir.then_body:
                raise CodegenError("LLIR IfThenElse requires a then body")
            result += (
                self.lower_llir(f"if ({self.lower_llir(ir.cond)}) {{", indent_level)
                + "\n"
                + self.lower_llir(ir.then_body, indent_level + 1)
                + "\n"
            )

        if ir.else_body:
            result += (
                self.lower_llir("} else {", indent_level)
                + "\n"
                + self.lower_llir(ir.else_body, indent_level + 1)
                + "\n"
            )

        result += self.lower_llir("}", indent_level)
        return result

    def lower_function_definition(
        self, ir: llir.Function, indent_level: int = 0
    ) -> str:
        if not all(isinstance(arg, llir.Var) for arg in ir.args):
            invalid_types = [
                type(arg).__name__ for arg in ir.args if not isinstance(arg, llir.Var)
            ]
            raise CodegenError(
                "LLIR Function arguments must be Var nodes; got "
                + ", ".join(invalid_types)
            )
        args = cast(List[llir.Var], ir.args)
        if not all(arg.type for arg in args):
            raise CodegenError("All LLIR Function arguments must have types")
        header = (
            f"{ir.return_type.value} {ir.name}"
            + f"({', '.join([self._lower_typed_var(arg) for arg in args])}) {{"
        )
        return (
            self.lower_llir(header, indent_level)
            + "\n"
            + self.lower_llir(ir.body, indent_level + 1)
            + "\n"
            + self.lower_llir("}", indent_level)
        )
