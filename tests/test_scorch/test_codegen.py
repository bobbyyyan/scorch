from dataclasses import FrozenInstanceError
from typing import Optional, Tuple, Union, cast, get_type_hints

import pytest

from scorch.compiler import llir
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.diagnostics import CodegenError
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]


def _var(name: str) -> llir.Var:
    return llir.Var(name=name, type=llir.DataType.INT)


def _result_metadata(access_id: int = 1) -> llir.TensorAccessMetadata:
    return llir.TensorAccessMetadata(
        access_id=AccessId(access_id),
        tensor_id=SymbolId(2),
        index_ids=(IndexId(3),),
        role=llir.TensorAccessRole.RESULT_WRITE,
    )


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
        (llir.MemberAccess(_var("it"), "second"), "it.second"),
        (
            llir.ArrayAccess(
                llir.MemberAccess(_var("it"), "first"),
                llir.Literal(0, llir.DataType.INT64),
            ),
            "it.first[0]",
        ),
        (
            llir.MemberAccess(
                llir.ArrayAccess(_var("entries"), _var("i")),
                "second",
            ),
            "entries[i].second",
        ),
        (
            llir.MemberAccess(
                llir.BinOp(op="+", left=_var("entry"), right=_var("offset")),
                "second",
            ),
            "(entry + offset).second",
        ),
        (
            llir.MemberAccess(
                llir.MemberAccess(_var("entry"), "first"),
                "value",
            ),
            "entry.first.value",
        ),
    ),
)
def test_member_access_codegen_is_byte_exact_and_precedence_correct(
    expression: llir.Expr,
    expected: str,
) -> None:
    assert LLIRLowerer().lower_llir(expression) == expected


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


def test_indexed_assign_codegen_is_byte_exact_and_precedence_correct() -> None:
    assignment = llir.Assign(
        var=llir.ArrayAccess(
            array=_var("values"),
            index=llir.Add(
                _var("i"),
                llir.Mul(_var("tile"), llir.Add(_var("j"), llir.Literal(1))),
            ),
            tensor_access=_result_metadata(),
        ),
        value=llir.ArrayAccess(
            array=_var("source"),
            index=llir.Add(_var("p"), _var("offset")),
        ),
        op=llir.AssignOp.ADD_ASSIGN,
    )

    assert (
        LLIRLowerer().lower_llir(assignment)
        == "values[i + tile * (j + 1)] += source[p + offset];"
    )
    assert (
        LLIRLowerer().lower_llir(assignment, no_semicolon=True)
        == "values[i + tile * (j + 1)] += source[p + offset]"
    )


def test_assign_target_is_narrowly_typed_frozen_and_structurally_equal() -> None:
    target = llir.ArrayAccess(
        array=_var("values"),
        index=_var("i"),
        tensor_access=_result_metadata(),
    )
    assignment = llir.Assign(
        var=target,
        value=llir.Literal(2),
        op=llir.AssignOp.MUL_ASSIGN,
    )
    equal_assignment = llir.Assign(
        var=llir.ArrayAccess(
            array=_var("values"),
            index=_var("i"),
            tensor_access=_result_metadata(access_id=99),
        ),
        value=llir.Literal(2),
        op=llir.AssignOp.MUL_ASSIGN,
    )

    assert assignment.var is target
    assert assignment.value == llir.Literal(2)
    assert assignment.op is llir.AssignOp.MUL_ASSIGN
    assert assignment.cast is False
    assert assignment == equal_assignment
    assert assignment != llir.Assign(target, llir.Literal(3), llir.AssignOp.MUL_ASSIGN)
    assert get_type_hints(llir.Assign) == {
        "var": Union[llir.Var, llir.ArrayAccess],
        "value": llir.Expr,
        "op": llir.AssignOp,
        "cast": bool,
    }

    with pytest.raises(FrozenInstanceError):
        target.array = _var("other")
    with pytest.raises(FrozenInstanceError):
        target.index = _var("j")


@pytest.mark.parametrize(
    ("target", "message"),
    (
        (llir.Literal(1), "exact Var or ArrayAccess"),
        (llir.FunctionCall("target"), "exact Var or ArrayAccess"),
        (llir.MemberAccess(_var("target"), "member"), "exact Var or ArrayAccess"),
        (_var("values[i]"), "identifier or member path"),
        (_var("call()"), "identifier or member path"),
        (_var("left + right"), "identifier or member path"),
        (_var("42"), "identifier or member path"),
        (
            llir.ArrayAccess(
                llir.BinOp("+", _var("values"), _var("offset")), _var("i")
            ),
            "ArrayAccess.array must be an exact Var",
        ),
        (
            llir.ArrayAccess(_var("values[i]"), _var("j")),
            "ArrayAccess.array name must be an identifier or member path",
        ),
        (
            llir.ArrayAccess(_var("values"), _var("indices[j]")),
            "assignment index Var names",
        ),
        (
            llir.ArrayAccess(_var("values"), llir.Literal("i + 1")),
            "Literal.value must be an int",
        ),
        (
            llir.ArrayAccess(_var("values"), llir.FunctionCall("i + 1")),
            "FunctionCall.name must be an identifier or member path",
        ),
        (
            llir.ArrayAccess(
                _var("values"),
                llir.BinOp("] = 0", _var("i"), llir.Literal(1)),
            ),
            "supported arithmetic operator",
        ),
        (
            llir.ArrayAccess(
                _var("values"),
                llir.Cast(_var("i"), cast(llir.DataType, "int")),
            ),
            "Cast.data_type must be a DataType",
        ),
        (
            llir.ArrayAccess(
                _var("values"),
                llir.Sizeof(cast(llir.DataType, "int")),
            ),
            "Sizeof.data_type must be a DataType",
        ),
        (
            llir.ArrayAccess(llir.ArrayAccess(_var("values"), _var("i")), _var("j")),
            "ArrayAccess.array must be an exact Var",
        ),
        (
            llir.ArrayAccess(
                _var("values"),
                _var("i"),
                llir.TensorAccessMetadata(
                    access_id=AccessId(1),
                    tensor_id=SymbolId(2),
                    index_ids=(IndexId(3),),
                    role=llir.TensorAccessRole.INPUT_READ,
                ),
            ),
            "RESULT_WRITE role",
        ),
    ),
)
def test_assign_rejects_arbitrary_rvalues_and_malformed_lvalues(
    target: llir.Expr,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        llir.Assign(
            var=cast(llir.AssignmentTarget, target),
            value=llir.Literal(1),
        )


def test_assign_rejects_target_subclasses_and_invalid_fields() -> None:
    class UnknownVar(llir.Var):
        pass

    class UnknownArrayAccess(llir.ArrayAccess):
        pass

    with pytest.raises(TypeError, match="exact Var or ArrayAccess"):
        llir.Assign(UnknownVar("value", llir.DataType.INT), llir.Literal(1))
    with pytest.raises(TypeError, match="exact Var or ArrayAccess"):
        llir.Assign(
            UnknownArrayAccess(_var("values"), _var("i")),
            llir.Literal(1),
        )
    with pytest.raises(TypeError, match="unsupported LLIR expression"):
        llir.Assign(
            llir.ArrayAccess(_var("values"), UnknownVar("i", llir.DataType.INT)),
            llir.Literal(1),
        )
    with pytest.raises(TypeError, match="Assign.value"):
        llir.Assign(_var("value"), "not an expression")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Assign.op"):
        llir.Assign(
            _var("value"),
            llir.Literal(1),
            op="=",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="Assign.cast"):
        llir.Assign(
            _var("value"),
            llir.Literal(1),
            cast=1,  # type: ignore[arg-type]
        )


def test_assign_cast_is_scalar_only_and_emits_the_existing_spelling() -> None:
    scalar = llir.Assign(
        _var("value"),
        llir.Add(llir.Literal(1), llir.Literal(2)),
        cast=True,
    )
    assert LLIRLowerer().lower_llir(scalar) == "value = (int) (1 + 2);"

    with pytest.raises(TypeError, match="cast requires an exact Var target"):
        llir.Assign(
            llir.ArrayAccess(_var("values"), _var("i")),
            llir.Literal(1),
            cast=True,
        )


def test_indexed_assign_accepts_a_structured_member_call_index() -> None:
    assignment = llir.Assign(
        llir.ArrayAccess(
            _var("values"),
            llir.FunctionCall("indices.size"),
        ),
        llir.Literal(1),
    )

    assert LLIRLowerer().lower_llir(assignment) == "values[indices.size()] = 1;"


@pytest.mark.parametrize(
    "malformation",
    [
        "rvalue",
        "subclass",
        "unknown_child",
        "unknown_var",
        "metadata",
        "cast_type",
    ],
)
def test_codegen_rejects_forged_malformed_assignment_targets(
    malformation: str,
) -> None:
    class UnknownArrayAccess(llir.ArrayAccess):
        pass

    class UnknownExpr(llir.Expr):
        pass

    class UnknownVar(llir.Var):
        pass

    assignment = llir.Assign(_var("value"), llir.Literal(1))
    if malformation == "rvalue":
        assignment.var = cast(llir.AssignmentTarget, llir.Literal(0))
        expected = "Invalid LLIR assignment target"
    elif malformation == "subclass":
        assignment.var = cast(
            llir.AssignmentTarget,
            UnknownArrayAccess(_var("values"), _var("i")),
        )
        expected = "Invalid LLIR assignment target"
    elif malformation == "unknown_child":
        target = object.__new__(llir.ArrayAccess)
        object.__setattr__(target, "array", _var("values"))
        object.__setattr__(target, "index", UnknownExpr())
        object.__setattr__(target, "tensor_access", None)
        assignment.var = target
        expected = "UnknownExpr"
    elif malformation == "unknown_var":
        target = object.__new__(llir.ArrayAccess)
        object.__setattr__(target, "array", _var("values"))
        object.__setattr__(target, "index", UnknownVar("i", llir.DataType.INT))
        object.__setattr__(target, "tensor_access", None)
        assignment.var = target
        expected = "UnknownVar"
    elif malformation == "metadata":
        metadata = object.__new__(llir.TensorAccessMetadata)
        object.__setattr__(metadata, "access_id", 1)
        object.__setattr__(metadata, "tensor_id", SymbolId(2))
        object.__setattr__(metadata, "index_ids", (IndexId(3),))
        object.__setattr__(metadata, "role", llir.TensorAccessRole.RESULT_WRITE)
        target = object.__new__(llir.ArrayAccess)
        object.__setattr__(target, "array", _var("values"))
        object.__setattr__(target, "index", _var("i"))
        object.__setattr__(target, "tensor_access", metadata)
        assignment.var = target
        expected = "AccessId"
    else:
        target = object.__new__(llir.ArrayAccess)
        object.__setattr__(target, "array", _var("values"))
        object.__setattr__(
            target,
            "index",
            llir.Cast(_var("i"), cast(llir.DataType, "int")),
        )
        object.__setattr__(target, "tensor_access", None)
        assignment.var = target
        expected = "Cast.data_type must be a DataType"

    with pytest.raises(CodegenError, match=expected):
        LLIRLowerer().lower_llir(assignment)


def test_codegen_rejects_forged_structured_cast_target() -> None:
    assignment = llir.Assign(
        llir.ArrayAccess(_var("values"), _var("i")),
        llir.Literal(1),
    )
    assignment.cast = True

    with pytest.raises(CodegenError, match="cast requires an exact Var target"):
        LLIRLowerer().lower_llir(assignment)


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


def test_member_access_is_frozen_typed_validated_and_structurally_equal() -> None:
    base = _var("it")
    access = llir.MemberAccess(base=base, member="second")

    assert access.base is base
    assert access.member == "second"
    assert access == llir.MemberAccess(_var("it"), "second")
    assert access != llir.MemberAccess(_var("it"), "first")
    assert access != llir.MemberAccess(_var("other"), "second")
    assert get_type_hints(llir.MemberAccess) == {
        "base": llir.Expr,
        "member": str,
    }

    with pytest.raises(FrozenInstanceError):
        access.base = _var("other")
    with pytest.raises(FrozenInstanceError):
        access.member = "first"

    with pytest.raises(TypeError, match="MemberAccess.base"):
        llir.MemberAccess("it", "second")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MemberAccess.member"):
        llir.MemberAccess(_var("it"), "")
    with pytest.raises(TypeError, match="MemberAccess.member"):
        llir.MemberAccess(_var("it"), "first.second")
    with pytest.raises(TypeError, match="MemberAccess.member"):
        llir.MemberAccess(_var("it"), 1)  # type: ignore[arg-type]


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


def test_codegen_rejects_member_access_subclasses_and_unknown_children() -> None:
    class UnknownMemberAccess(llir.MemberAccess):
        pass

    class UnknownExpr(llir.Expr):
        pass

    with pytest.raises(CodegenError, match="UnknownMemberAccess"):
        LLIRLowerer().lower_llir(UnknownMemberAccess(_var("it"), "second"))
    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(llir.MemberAccess(UnknownExpr(), "second"))


@pytest.mark.parametrize("malformation", ("base", "member"))
def test_codegen_rejects_forged_member_access_fields(malformation: str) -> None:
    access = object.__new__(llir.MemberAccess)
    object.__setattr__(access, "base", _var("it"))
    object.__setattr__(access, "member", "second")
    if malformation == "base":
        object.__setattr__(access, "base", "it")
        expected = "MemberAccess.base"
    else:
        object.__setattr__(access, "member", "first.second")
        expected = "MemberAccess.member"

    with pytest.raises(CodegenError, match=expected):
        LLIRLowerer().lower_llir(access)


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
