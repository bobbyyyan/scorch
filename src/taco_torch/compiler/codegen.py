from typing import Union, List

from src.taco_torch.compiler import llir


class LLIRLowerer:
    """
    This is a class to lower LLIR to C++ code (string).
    """

    indent_str = "  "
    indent_level = 0

    def lower_llir(
        self,
        ir: Union[llir.Node, List[llir.Node], str, List[str]],
        indent_level: int = 0,
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

        elif isinstance(ir, llir.VarAssign):
            if ir.var.type:
                return self.lower_llir(
                    f"{ir.var.type.value} {ir.var.name} {ir.op} {self.lower_llir(ir.value)};",
                    indent_level,
                )
            else:
                return self.lower_llir(
                    f"{ir.var.name} {ir.op} {self.lower_llir(ir.value)};", indent_level
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
        elif isinstance(ir, llir.WhileLoop):
            return (
                self.lower_llir(f"while ({self.lower_llir(ir.cond)}) {{", indent_level)
                + "\n"
                + self.lower_llir(ir.body, indent_level + 1)
                + "\n"
                + self.lower_llir("}", indent_level)
            )
        elif isinstance(ir, llir.IfThenElse):
            # Handle the case where cond is a list of conditions and then_body is a list of list
            # of statements
            result = ""
            if isinstance(ir.cond, list):
                assert isinstance(ir.then_body, list)
                assert len(ir.cond) == len(
                    ir.then_body
                ), "Number of conditions and then_body's must be the same"

                total_num_conds = len(ir.cond) + (1 if ir.else_body else 0)

                for i, cond in enumerate(ir.cond):
                    if i == 0:
                        result += (
                            self.lower_llir(
                                f"if ({self.lower_llir(cond)}) {{", indent_level
                            )
                            + "\n"
                            + self.lower_llir(ir.then_body[i], indent_level + 1)
                            + "\n"
                        )
                    elif ir.make_last_case_else and i == total_num_conds - 1:
                        result += (
                            self.lower_llir("} else {", indent_level)
                            + "\n"
                            + self.lower_llir(ir.then_body[i], indent_level + 1)
                            + "\n"
                        )
                    else:
                        result += (
                            self.lower_llir(
                                f"}} else if ({self.lower_llir(cond)}) {{", indent_level
                            )
                            + "\n"
                            + self.lower_llir(ir.then_body[i], indent_level + 1)
                            + "\n"
                        )
            else:
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

        elif isinstance(ir, llir.Increment):
            return self.lower_llir(f"{ir.var.name}++;", indent_level)

        elif isinstance(ir, llir.Function):
            # args must be llir.Var's
            assert [isinstance(arg, llir.Var) for arg in ir.args]
            return (
                self.lower_llir(
                    f"{ir.return_type.value} {ir.name}"
                    + f"({', '.join([f'{arg.type.value} {arg.name}' for arg in ir.args])}) {{",
                    indent_level,
                )
                + "\n"
                + self.lower_llir(ir.body, indent_level + 1)
                + "\n"
                + self.lower_llir("}", indent_level)
            )
        return self.lower_llir(
            f"No code gen implemented for node type: {ir.__class__.__name__}",
            indent_level,
        )
