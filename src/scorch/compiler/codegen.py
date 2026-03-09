from typing import Union, List, TypeVar, cast

from . import llir

LLIR_NODE = TypeVar("LLIR_NODE", bound=llir.Node)


class LLIRLowerer:
    """
    This is a class to lower LLIR to C++ code (string).
    """

    indent_str = "  "
    indent_level = 0
    no_comments = False

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
                f"{self._lower_typed_var(ir.var)} {ir.op} {self.lower_llir(ir.value)};",
                indent_level,
            )

        elif isinstance(ir, llir.Assign):
            assert isinstance(ir.var, llir.Var), f"Invalid var: {ir.var}"
            if no_semicolon:
                return self.lower_llir(
                    f"{ir.var.name} {ir.op.value} {self.lower_llir(ir.value)}",
                    indent_level,
                )
            return self.lower_llir(
                f"{ir.var.name} {ir.op.value} {self.lower_llir(ir.value)};",
                indent_level,
            )

        elif isinstance(ir, (llir.Literal, llir.Cast, llir.Sizeof, llir.BinOp,
                             llir.UnaryOp, llir.FunctionCall, llir.Array,
                             llir.ArrayAccess)):
            return self.lower_expression(ir, indent_level)

        elif isinstance(ir, llir.FunctionCallStmt):
            return self.lower_llir(
                f"{ir.name}({', '.join([self.lower_llir(arg) for arg in ir.args])});",
                indent_level,
            )

        elif isinstance(ir, (llir.WhileLoop, llir.ForLoop, llir.ForLoopAuto)):
            return self.lower_loop_construct(ir, indent_level)

        elif isinstance(ir, llir.IfThenElse):
            return self.lower_conditional(ir, indent_level)

        elif isinstance(ir, llir.Var):
            return ir.name

        elif isinstance(ir, llir.VarDecl):
            return self.lower_llir(f"{self._lower_typed_var(ir.var)};", indent_level)

        elif isinstance(ir, llir.RawStmt):
            suffix = ";" if ir.add_semicolon else ""
            return self.lower_llir(f"{ir.code}{suffix}", indent_level)

        elif isinstance(ir, llir.Break):
            return self.lower_llir("break;", indent_level)

        elif isinstance(ir, llir.Increment):
            if no_semicolon:
                return self.lower_llir(f"{ir.var.name}++", indent_level)
            return self.lower_llir(f"{ir.var.name}++;", indent_level)

        elif isinstance(ir, llir.Function):
            return self.lower_function_definition(ir, indent_level)

        elif isinstance(ir, llir.Return):
            return self.lower_llir(f"return {self.lower_llir(ir.value)};", indent_level)

        return self.lower_llir(
            f"No code gen implemented for node type: {ir.__class__.__name__}",
            indent_level,
        )

    def lower_expression(
        self,
        ir: Union[llir.Literal, llir.Cast, llir.Sizeof, llir.BinOp,
                  llir.UnaryOp, llir.FunctionCall, llir.Array, llir.ArrayAccess],
        indent_level: int = 0,
    ) -> str:
        if isinstance(ir, llir.Literal):
            return self.lower_llir(str(ir.value), indent_level)

        elif isinstance(ir, llir.Cast):
            return self.lower_llir(
                f"({ir.data_type.value}) {self.lower_llir(ir.expr)}", indent_level
            )

        elif isinstance(ir, llir.Sizeof):
            return self.lower_llir(f"sizeof({ir.data_type.value})", indent_level)

        elif isinstance(ir, llir.BinOp):
            return self.lower_llir(
                f"{self.lower_llir(ir.left)} {ir.op} {self.lower_llir(ir.right)}",
                indent_level,
            )

        elif isinstance(ir, llir.UnaryOp):
            return self.lower_llir(
                f"{ir.op} {self.lower_llir(ir.operand)}", indent_level
            )

        elif isinstance(ir, llir.FunctionCall):
            return self.lower_llir(
                f"{ir.name}({', '.join([self.lower_llir(arg) for arg in ir.args])})",
                indent_level,
            )

        elif isinstance(ir, llir.Array):
            return self.lower_llir(
                f"{{{', '.join([self.lower_llir(v) for v in ir.values])}}}",
                indent_level,
            )

        elif isinstance(ir, llir.ArrayAccess):
            return self.lower_llir(
                f"{self.lower_llir(ir.array)}[{self.lower_llir(ir.index)}]",
                indent_level,
            )

        raise ValueError(f"Unknown expression type: {type(ir)}")

    def lower_loop_construct(
        self,
        ir: Union[llir.WhileLoop, llir.ForLoop, llir.ForLoopAuto],
        indent_level: int = 0,
    ) -> str:
        pragma_lines: List[str] = []
        if isinstance(ir, llir.WhileLoop):
            header = f"while ({self.lower_llir(ir.cond)}) {{"
        elif isinstance(ir, llir.ForLoop):
            if ir.omp_parallel_for:
                omp_pragma = "#pragma omp parallel for"
                if ir.omp_schedule:
                    omp_pragma += f" schedule({ir.omp_schedule})"
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
        return pragma_text + "\n" + loop_text

    def lower_conditional(
        self, ir: llir.IfThenElse, indent_level: int = 0
    ) -> str:
        result = ""
        if ir.cond_list:
            assert ir.then_body_list, "then_body_list must be provided"
            assert len(ir.cond_list) == len(
                ir.then_body_list
            ), "Number of conditions and then_body's must be the same"

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
            assert ir.cond, "If condition must be provided"
            assert ir.then_body, "If then body must be provided"
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
        if ir.args:
            assert [
                isinstance(arg, llir.Var) for arg in ir.args
            ], "Args must be llir.Var's"
        args = cast(List[llir.Var], ir.args)
        assert all([arg.type for arg in args]), "All args must have types"
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
