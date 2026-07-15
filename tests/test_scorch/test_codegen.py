from dataclasses import FrozenInstanceError
from typing import Optional, Tuple, get_type_hints

import pytest

from scorch.compiler import llir
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.diagnostics import CodegenError
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]


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


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        (
            llir.ArrayAccess(
                array=llir.BinOp(op="+", left=_var("a"), right=_var("offset")),
                index=_var("i"),
            ),
            "(a + offset)[i]",
        ),
        (
            llir.ArrayAccess(
                array=_var("values"),
                index=llir.Add(_var("i"), _var("stride")),
            ),
            "values[i + stride]",
        ),
        (
            llir.ArrayAccess(
                array=llir.ArrayAccess(_var("values"), _var("i")),
                index=_var("j"),
            ),
            "values[i][j]",
        ),
    ),
)
def test_array_access_codegen_is_byte_exact_and_precedence_correct(
    expression: llir.ArrayAccess,
    expected: str,
) -> None:
    assert LLIRLowerer().lower_llir(expression) == expected


def test_array_access_is_frozen_typed_validated_and_structurally_equal() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(1),
        tensor_id=SymbolId(2),
        index_ids=(IndexId(3),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    other_metadata = llir.TensorAccessMetadata(
        access_id=AccessId(4),
        tensor_id=SymbolId(5),
        index_ids=(IndexId(6),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    access = llir.ArrayAccess(_var("values"), _var("i"), metadata)

    assert access.array == _var("values")
    assert access.index == _var("i")
    assert access.tensor_access is metadata
    # ArrayAccess equality is emission-structural; semantic consumers compare
    # the stable IDs in the deliberately non-comparing provenance field.
    assert access == llir.ArrayAccess(_var("values"), _var("i"), other_metadata)
    assert metadata != other_metadata
    assert access != llir.ArrayAccess(_var("values"), _var("j"), metadata)
    hints = get_type_hints(llir.ArrayAccess)
    assert hints["array"] is llir.Expr
    assert hints["index"] is llir.Expr
    assert hints["tensor_access"] == Optional[llir.TensorAccessMetadata]

    metadata_hints = get_type_hints(llir.TensorAccessMetadata)
    assert metadata_hints == {
        "access_id": AccessId,
        "tensor_id": SymbolId,
        "index_ids": Tuple[IndexId, ...],
        "role": llir.TensorAccessRole,
    }

    with pytest.raises(FrozenInstanceError):
        access.index = _var("j")
    with pytest.raises(FrozenInstanceError):
        metadata.tensor_id = SymbolId(99)

    with pytest.raises(TypeError, match="ArrayAccess.array"):
        llir.ArrayAccess("values", _var("i"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ArrayAccess.index"):
        llir.ArrayAccess(_var("values"), 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ArrayAccess.tensor_access"):
        llir.ArrayAccess(
            _var("values"),
            _var("i"),
            tensor_access="metadata",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {
            "access_id": 1,
            "tensor_id": SymbolId(2),
            "index_ids": (IndexId(3),),
            "role": llir.TensorAccessRole.INPUT_READ,
        },
        {
            "access_id": AccessId(1),
            "tensor_id": 2,
            "index_ids": (IndexId(3),),
            "role": llir.TensorAccessRole.INPUT_READ,
        },
        {
            "access_id": AccessId(1),
            "tensor_id": SymbolId(2),
            "index_ids": [IndexId(3)],
            "role": llir.TensorAccessRole.INPUT_READ,
        },
        {
            "access_id": AccessId(1),
            "tensor_id": SymbolId(2),
            "index_ids": (3,),
            "role": llir.TensorAccessRole.INPUT_READ,
        },
        {
            "access_id": AccessId(1),
            "tensor_id": SymbolId(2),
            "index_ids": (IndexId(3),),
            "role": "input_read",
        },
    ),
)
def test_tensor_access_metadata_rejects_malformed_typed_identity(kwargs) -> None:
    with pytest.raises(TypeError):
        llir.TensorAccessMetadata(**kwargs)


def test_codegen_rejects_array_access_subclasses_and_unknown_children() -> None:
    class UnknownArrayAccess(llir.ArrayAccess):
        pass

    class UnknownExpr(llir.Expr):
        pass

    with pytest.raises(CodegenError, match="UnknownArrayAccess"):
        LLIRLowerer().lower_llir(
            UnknownArrayAccess(array=_var("values"), index=_var("i"))
        )
    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(llir.ArrayAccess(array=UnknownExpr(), index=_var("i")))


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
