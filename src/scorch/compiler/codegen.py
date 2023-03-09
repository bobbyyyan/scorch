from typing import Union, List, TypeVar, cast

from src.scorch.compiler import llir

LLIR_NODE = TypeVar("LLIR_NODE", bound=llir.Node)


class LLIRLowerer:
    """
    This is a class to lower LLIR to C++ code (string).
    """

    indent_str = "  "
    indent_level = 0

    def lower_llir(
        self,
        ir: Union[LLIR_NODE, List[LLIR_NODE], str, List[str]],
        indent_level: int = 0,
        no_semicolon: bool = False,
    ) -> str:
        if isinstance(ir, str):
            return indent_level * self.indent_str + ir

        elif isinstance(ir, list):
            return "\n".join([self.lower_llir(node, indent_level) for node in ir])

        elif isinstance(ir, llir.Comment):
            return self.lower_llir(f"// {ir.value}", indent_level)

        elif isinstance(ir, llir.BlankLine):
            return self.lower_llir("", indent_level)

        elif isinstance(ir, llir.Literal):
            return self.lower_llir(str(ir.value), indent_level)

        elif isinstance(ir, llir.VarInit):
            return self.lower_llir(
                f"{ir.var.type.value} {ir.var.name} {ir.op} {self.lower_llir(ir.value)};",
                indent_level,
            )

        elif isinstance(ir, llir.Assign):
            return self.lower_llir(
                f"{ir.var.name} {ir.op.value} {self.lower_llir(ir.value)};",
                indent_level,
            )

        elif isinstance(ir, llir.Cast):
            return self.lower_llir(
                f"({ir.data_type.value}) {self.lower_llir(ir.expr)}", indent_level
            )

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
        elif isinstance(ir, llir.FunctionCallStmt):
            return self.lower_llir(
                f"{ir.name}({', '.join([self.lower_llir(arg) for arg in ir.args])});",
                indent_level,
            )
        elif isinstance(ir, llir.Array):
            return self.lower_llir(
                f"{{{', '.join([self.lower_llir(v) for v in ir.values])}}}",
                indent_level,
            )
        elif isinstance(ir, llir.WhileLoop):
            return (
                self.lower_llir(f"while ({self.lower_llir(ir.cond)}) {{", indent_level)
                + "\n"
                + self.lower_llir(ir.body, indent_level + 1)
                + "\n"
                + self.lower_llir("}", indent_level)
            )

        elif isinstance(ir, llir.ForLoop):
            return (
                self.lower_llir(
                    f"for ({self.lower_llir(ir.init)} {self.lower_llir(ir.cond)}; {self.lower_llir(ir.update, no_semicolon=True)}) {{",
                    indent_level,
                )
                + "\n"
                + self.lower_llir(ir.body, indent_level + 1)
                + "\n"
                + self.lower_llir("}", indent_level)
            )
        elif isinstance(ir, llir.IfThenElse):
            # Handle the case where cond is a list of conditions and then_body is a list of list
            # of statements
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

        elif isinstance(ir, llir.Var):
            return ir.name

        elif isinstance(ir, llir.VarDecl):
            return self.lower_llir(f"{ir.var.type.value} {ir.var.name};", indent_level)

        elif isinstance(ir, llir.Increment):
            if no_semicolon:
                return self.lower_llir(f"{ir.var.name}++", indent_level)
            return self.lower_llir(f"{ir.var.name}++;", indent_level)

        elif isinstance(ir, llir.Function):
            # args must be llir.Var's
            if ir.args:
                assert [
                    isinstance(arg, llir.Var) for arg in ir.args
                ], "Args must be llir.Var's"
            args = cast(List[llir.Var], ir.args)
            # Assert all the args have types so that type checker is happy
            assert all([arg.type for arg in args]), "All args must have types"
            return (
                self.lower_llir(
                    f"{ir.return_type.value} {ir.name}"
                    + f"({', '.join([f'{arg.type.value} {arg.name}' for arg in args])}) {{",
                    indent_level,
                )
                + "\n"
                + self.lower_llir(ir.body, indent_level + 1)
                + "\n"
                + self.lower_llir("}", indent_level)
            )

        elif isinstance(ir, llir.Return):
            return self.lower_llir(f"return {self.lower_llir(ir.value)};", indent_level)

        return self.lower_llir(
            f"No code gen implemented for node type: {ir.__class__.__name__}",
            indent_level,
        )
