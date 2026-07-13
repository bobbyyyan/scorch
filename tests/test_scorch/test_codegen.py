import pytest

from scorch.compiler import llir
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.diagnostics import CodegenError


def _var(name: str) -> llir.Var:
    return llir.Var(name=name, type=llir.DataType.INT)


def test_codegen_rejects_unknown_statement_node() -> None:
    class UnknownStmt(llir.Stmt):
        pass

    with pytest.raises(CodegenError, match="UnknownStmt"):
        LLIRLowerer().lower_llir(UnknownStmt())


def test_codegen_rejects_known_but_unsupported_expression_node() -> None:
    expression = llir.GetTensorProperty(
        tensor=_var("tensor"), tensor_property=llir.TensorProperty.VALUES
    )

    with pytest.raises(CodegenError, match="GetTensorProperty"):
        LLIRLowerer().lower_llir(expression)


def test_codegen_rejects_unknown_binary_operator() -> None:
    expression = llir.BinOp(op="<=>", left=_var("a"), right=_var("b"))

    with pytest.raises(CodegenError, match="binary operator.*<=>"):
        LLIRLowerer().lower_llir(expression)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (
            llir.BinOp(
                op="*",
                left=llir.BinOp(op="+", left=_var("a"), right=_var("b")),
                right=_var("c"),
            ),
            "(a + b) * c",
        ),
        (
            llir.BinOp(
                op="+",
                left=_var("a"),
                right=llir.BinOp(op="*", left=_var("b"), right=_var("c")),
            ),
            "a + b * c",
        ),
        (
            llir.BinOp(
                op="-",
                left=_var("a"),
                right=llir.BinOp(op="+", left=_var("b"), right=_var("c")),
            ),
            "a - (b + c)",
        ),
        (
            llir.BinOp(
                op="+",
                left=llir.BinOp(op="+", left=_var("a"), right=_var("b")),
                right=_var("c"),
            ),
            "a + b + c",
        ),
        (
            llir.BinOp(
                op="+",
                left=_var("a"),
                right=llir.BinOp(op="+", left=_var("b"), right=_var("c")),
            ),
            "a + (b + c)",
        ),
        (
            llir.BinOp(
                op="&&",
                left=llir.BinOp(op="||", left=_var("a"), right=_var("b")),
                right=_var("c"),
            ),
            "(a || b) && c",
        ),
    ],
)
def test_binary_expression_codegen_preserves_ast_precedence(
    expression: llir.Expr, expected: str
) -> None:
    assert LLIRLowerer().lower_llir(expression) == expected


def test_cast_codegen_parenthesizes_lower_precedence_operand() -> None:
    expression = llir.Cast(
        expr=llir.BinOp(op="+", left=_var("a"), right=_var("b")),
        data_type=llir.DataType.FLOAT32,
    )

    assert LLIRLowerer().lower_llir(expression) == "(float) (a + b)"


def test_array_access_codegen_parenthesizes_expression_array() -> None:
    expression = llir.ArrayAccess(
        array=llir.BinOp(op="+", left=_var("a"), right=_var("offset")),
        index=_var("i"),
    )

    assert LLIRLowerer().lower_llir(expression) == "(a + offset)[i]"


def test_function_codegen_rejects_non_var_argument() -> None:
    function = llir.Function(
        return_type=llir.DataType.VOID,
        name="kernel",
        args=[llir.Literal(1)],
        body=[],
    )

    with pytest.raises(CodegenError, match="arguments must be Var"):
        LLIRLowerer().lower_llir(function)


def test_conditional_codegen_rejects_mismatched_condition_bodies() -> None:
    conditional = llir.IfThenElse(
        cond_list=[_var("a"), _var("b")], then_body_list=[[llir.Break()]]
    )

    with pytest.raises(CodegenError, match="condition and body counts"):
        LLIRLowerer().lower_llir(conditional)
