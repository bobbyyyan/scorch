from dataclasses import FrozenInstanceError
from typing import Any, Optional, Tuple, Union, cast, get_type_hints

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


def test_increment_is_frozen_typed_structural_and_byte_exact() -> None:
    variable = _var("counter")
    increment = llir.Increment(variable)
    equal = llir.Increment(_var("counter"))

    assert increment.var is variable
    assert increment == equal
    assert hash(increment) == hash(equal)
    assert increment != llir.Increment(_var("other"))
    assert get_type_hints(llir.Increment) == {"var": llir.Var}
    assert LLIRLowerer().lower_llir(increment) == "counter++;"
    assert LLIRLowerer().lower_llir(increment, no_semicolon=True) == "counter++"

    with pytest.raises(FrozenInstanceError):
        increment.var = _var("other")


def test_increment_rejects_nonexact_vars_and_codegen_fails_closed() -> None:
    class UnknownVar(llir.Var):
        pass

    class UnknownIncrement(llir.Increment):
        pass

    with pytest.raises(TypeError, match="Increment.var must be an exact LLIR Var"):
        llir.Increment(cast(llir.Var, llir.Literal(1)))
    with pytest.raises(TypeError, match="Increment.var must be an exact LLIR Var"):
        llir.Increment(UnknownVar("counter", llir.DataType.INT))
    with pytest.raises(CodegenError, match="UnknownIncrement"):
        LLIRLowerer().lower_llir(UnknownIncrement(_var("counter")))

    forged = object.__new__(llir.Increment)
    object.__setattr__(forged, "var", llir.Literal(1))
    with pytest.raises(CodegenError, match="Increment.var must be an exact LLIR Var"):
        LLIRLowerer().lower_llir(forged)


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


def test_cast_is_frozen_typed_and_structurally_equal() -> None:
    operand = _var("value")
    expression = llir.Cast(operand, llir.DataType.INT)
    equal = llir.Cast(_var("value"), llir.DataType.INT)

    assert expression.expr is operand
    assert expression.data_type is llir.DataType.INT
    assert expression == equal
    assert hash(expression) == hash(equal)
    assert expression != llir.Cast(_var("other"), llir.DataType.INT)
    assert expression != llir.Cast(_var("value"), llir.DataType.INT64)
    assert get_type_hints(llir.Cast) == {
        "expr": llir.Expr,
        "data_type": llir.DataType,
    }

    with pytest.raises(FrozenInstanceError):
        expression.expr = _var("other")
    with pytest.raises(FrozenInstanceError):
        expression.data_type = llir.DataType.INT64


@pytest.mark.parametrize(
    ("expression", "data_type", "message"),
    (
        ("value", llir.DataType.INT, "Cast.expr"),
        (_var("value"), "int", "Cast.data_type"),
    ),
)
def test_cast_rejects_malformed_constructor_fields(
    expression: object,
    data_type: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        llir.Cast(
            cast(llir.Expr, expression),
            cast(llir.DataType, data_type),
        )


@pytest.mark.parametrize("malformation", ("expression", "data_type"))
def test_codegen_rejects_forged_cast_fields(malformation: str) -> None:
    expression = llir.Cast(_var("value"), llir.DataType.INT)
    if malformation == "expression":
        object.__setattr__(expression, "expr", "value")
        message = "Cast.expr"
    else:
        object.__setattr__(expression, "data_type", "int")
        message = "Cast.data_type"

    with pytest.raises(CodegenError, match=message):
        LLIRLowerer().lower_llir(expression)


def test_codegen_rejects_unknown_cast_subclass() -> None:
    class UnknownCast(llir.Cast):
        pass

    with pytest.raises(CodegenError, match="UnknownCast"):
        LLIRLowerer().lower_llir(UnknownCast(_var("value"), llir.DataType.INT))


def test_codegen_rejects_unknown_cast_child() -> None:
    class UnknownExpr(llir.Expr):
        pass

    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(llir.Cast(UnknownExpr(), llir.DataType.INT))


def test_binary_and_literal_nodes_are_frozen_typed_structural_values() -> None:
    left = _var("i")
    one = llir.Literal(1, llir.DataType.INT64)
    add = llir.Add(left, one)
    equal_add = llir.Add(_var("i"), llir.Literal(1, llir.DataType.INT64))
    generic_add = llir.BinOp("+", _var("i"), llir.Literal(1, llir.DataType.INT64))
    multiply = llir.Mul(_var("i"), llir.Literal(1, llir.DataType.INT64))

    assert type(add) is llir.Add
    assert add.op == "+"
    assert add.left is left
    assert add.right is one
    assert type(multiply) is llir.Mul
    assert multiply.op == "*"
    assert llir.Literal(1).data_type is llir.DataType.INT32
    assert llir.Literal(True).data_type is llir.DataType.BOOL
    assert llir.Literal(1.0).data_type is llir.DataType.FLOAT32
    assert llir.Literal("value").data_type is llir.DataType.STRING
    binary_hints = {
        "op": str,
        "left": llir.Expr,
        "right": llir.Expr,
    }
    assert get_type_hints(llir.BinOp) == binary_hints
    assert get_type_hints(llir.Add) == binary_hints
    assert get_type_hints(llir.Mul) == binary_hints
    assert get_type_hints(llir.Literal) == {
        "value": Any,
        "data_type": Optional[llir.DataType],
    }

    assert add == equal_add
    assert hash(add) == hash(equal_add)
    assert add != generic_add
    assert add != multiply
    assert len({add, equal_add, generic_add, multiply}) == 3
    assert llir.Literal(1) == llir.Literal(1, llir.DataType.INT32)
    assert hash(llir.Literal(1)) == hash(llir.Literal(1, llir.DataType.INT32))
    assert llir.Literal(1) != llir.Literal(1, llir.DataType.INT64)
    assert llir.Literal(True) != llir.Literal(1)

    with pytest.raises(FrozenInstanceError):
        add.op = "-"
    with pytest.raises(FrozenInstanceError):
        add.left = _var("j")
    with pytest.raises(FrozenInstanceError):
        one.value = 2
    with pytest.raises(FrozenInstanceError):
        one.data_type = llir.DataType.INT32


def test_binary_and_literal_construction_rejects_malformed_fields() -> None:
    with pytest.raises(TypeError, match="BinOp.op"):
        llir.BinOp("", _var("left"), _var("right"))
    with pytest.raises(TypeError, match="BinOp.op"):
        llir.BinOp(1, _var("left"), _var("right"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BinOp.left"):
        llir.BinOp("+", "left", _var("right"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BinOp.right"):
        llir.Add(_var("left"), "right")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Literal.value"):
        llir.Literal(object())
    with pytest.raises(TypeError, match="Literal.data_type"):
        llir.Literal(1, cast(llir.DataType, "int64_t"))


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        (llir.Add(_var("p"), llir.Literal(1)), "p + 1"),
        (
            llir.Mul(llir.Add(_var("a"), _var("b")), _var("c")),
            "(a + b) * c",
        ),
        (
            llir.Add(llir.Mul(_var("a"), _var("b")), _var("c")),
            "a * b + c",
        ),
        (
            llir.Add(_var("a"), llir.Add(_var("b"), _var("c"))),
            "a + (b + c)",
        ),
        (
            llir.Mul(_var("a"), llir.Mul(_var("b"), _var("c"))),
            "a * (b * c)",
        ),
    ),
)
def test_exact_add_mul_codegen_is_byte_exact_and_precedence_safe(
    expression: llir.BinOp,
    expected: str,
) -> None:
    assert LLIRLowerer().lower_llir(expression) == expected


@pytest.mark.parametrize(
    "node_type",
    (llir.BinOp, llir.Add, llir.Mul, llir.Literal),
)
def test_codegen_rejects_unknown_arithmetic_subclasses(
    node_type: type[llir.Expr],
) -> None:
    if node_type is llir.Literal:

        class UnknownArithmetic(node_type):  # type: ignore[misc, valid-type]
            pass

        expression = UnknownArithmetic(1)
    else:

        class UnknownArithmetic(node_type):  # type: ignore[misc, valid-type, no-redef]
            pass

        if node_type is llir.BinOp:
            expression = UnknownArithmetic("+", _var("left"), _var("right"))
        else:
            expression = UnknownArithmetic(_var("left"), _var("right"))

    with pytest.raises(CodegenError, match="UnknownArithmetic"):
        LLIRLowerer().lower_llir(expression)


@pytest.mark.parametrize(
    ("node_type", "field", "invalid", "message"),
    (
        (llir.BinOp, "op", "", "non-empty string"),
        (llir.Add, "op", "-", "Add.op"),
        (llir.Mul, "op", "+", "Mul.op"),
        (llir.BinOp, "left", "left", "BinOp.left"),
        (llir.Add, "right", "right", "BinOp.right"),
    ),
)
def test_codegen_rejects_forged_binary_fields(
    node_type: type[llir.BinOp],
    field: str,
    invalid: object,
    message: str,
) -> None:
    expression = object.__new__(node_type)
    object.__setattr__(expression, "op", "*" if node_type is llir.Mul else "+")
    object.__setattr__(expression, "left", _var("left"))
    object.__setattr__(expression, "right", _var("right"))
    object.__setattr__(expression, field, invalid)

    with pytest.raises(CodegenError, match=message):
        LLIRLowerer().lower_llir(expression)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("value", object(), "Literal.value"),
        ("data_type", None, "Literal.data_type"),
    ),
)
def test_codegen_rejects_forged_literal_fields(
    field: str,
    invalid: object,
    message: str,
) -> None:
    literal = object.__new__(llir.Literal)
    object.__setattr__(literal, "value", 1)
    object.__setattr__(literal, "data_type", llir.DataType.INT32)
    object.__setattr__(literal, field, invalid)

    with pytest.raises(CodegenError, match=message):
        LLIRLowerer().lower_llir(literal)


def test_codegen_rejects_unknown_binary_children() -> None:
    class UnknownExpr(llir.Expr):
        pass

    expression = llir.Add(UnknownExpr(), llir.Literal(1))

    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(expression)


def test_qualified_name_is_frozen_typed_and_structurally_equal() -> None:
    expression = llir.QualifiedName(
        namespace="torch",
        name="kInt",
        data_type=llir.DataType.TORCH_SCALAR_TYPE,
    )
    equal = llir.QualifiedName(
        namespace="torch",
        name="kInt",
        data_type=llir.DataType.TORCH_SCALAR_TYPE,
    )

    assert expression.namespace == "torch"
    assert expression.name == "kInt"
    assert expression.data_type is llir.DataType.TORCH_SCALAR_TYPE
    assert llir.DataType.TORCH_SCALAR_TYPE.value == "torch::ScalarType"
    assert expression == equal
    assert hash(expression) == hash(equal)
    assert expression != llir.QualifiedName(
        "at", "kInt", llir.DataType.TORCH_SCALAR_TYPE
    )
    assert expression != llir.QualifiedName(
        "torch", "kFloat32", llir.DataType.TORCH_SCALAR_TYPE
    )
    assert expression != llir.QualifiedName("torch", "kInt", llir.DataType.INT)
    assert get_type_hints(llir.QualifiedName) == {
        "namespace": str,
        "name": str,
        "data_type": llir.DataType,
    }

    with pytest.raises(FrozenInstanceError):
        expression.namespace = "at"
    with pytest.raises(FrozenInstanceError):
        expression.name = "kFloat32"
    with pytest.raises(FrozenInstanceError):
        expression.data_type = llir.DataType.INT


@pytest.mark.parametrize(
    ("namespace", "name", "data_type", "message"),
    (
        (1, "kInt", llir.DataType.TORCH_SCALAR_TYPE, "QualifiedName.namespace"),
        ("", "kInt", llir.DataType.TORCH_SCALAR_TYPE, "QualifiedName.namespace"),
        (
            "torch::detail",
            "kInt",
            llir.DataType.TORCH_SCALAR_TYPE,
            "QualifiedName.namespace",
        ),
        ("torch", 1, llir.DataType.TORCH_SCALAR_TYPE, "QualifiedName.name"),
        ("torch", "", llir.DataType.TORCH_SCALAR_TYPE, "QualifiedName.name"),
        (
            "torch",
            "k-Int",
            llir.DataType.TORCH_SCALAR_TYPE,
            "QualifiedName.name",
        ),
        ("torch", "kInt", "torch::ScalarType", "QualifiedName.data_type"),
    ),
)
def test_qualified_name_rejects_malformed_constructor_fields(
    namespace: object,
    name: object,
    data_type: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        llir.QualifiedName(
            namespace=cast(str, namespace),
            name=cast(str, name),
            data_type=cast(llir.DataType, data_type),
        )


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        (
            llir.QualifiedName("torch", "kInt", llir.DataType.TORCH_SCALAR_TYPE),
            "torch::kInt",
        ),
        (
            llir.FunctionCall(
                "consume",
                (
                    llir.QualifiedName(
                        "torch", "kFloat32", llir.DataType.TORCH_SCALAR_TYPE
                    ),
                ),
            ),
            "consume(torch::kFloat32)",
        ),
        (
            llir.BinOp(
                "==",
                llir.QualifiedName("torch", "kInt", llir.DataType.TORCH_SCALAR_TYPE),
                llir.QualifiedName(
                    "torch", "kFloat32", llir.DataType.TORCH_SCALAR_TYPE
                ),
            ),
            "torch::kInt == torch::kFloat32",
        ),
        (
            llir.Cast(
                llir.QualifiedName("torch", "kInt", llir.DataType.TORCH_SCALAR_TYPE),
                llir.DataType.INT,
            ),
            "(int) torch::kInt",
        ),
        (
            llir.MemberAccess(
                llir.QualifiedName("torch", "kInt", llir.DataType.TORCH_SCALAR_TYPE),
                "value",
            ),
            "torch::kInt.value",
        ),
        (
            llir.ArrayAccess(
                llir.QualifiedName("torch", "kInt", llir.DataType.TORCH_SCALAR_TYPE),
                llir.Literal(0),
            ),
            "torch::kInt[0]",
        ),
    ),
)
def test_qualified_name_codegen_is_byte_exact_and_precedence_safe(
    expression: llir.Expr,
    expected: str,
) -> None:
    assert LLIRLowerer().lower_llir(expression) == expected


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("namespace", "torch::detail", "QualifiedName.namespace"),
        ("name", "k-Int", "QualifiedName.name"),
        ("data_type", "torch::ScalarType", "QualifiedName.data_type"),
    ),
)
def test_codegen_rejects_forged_qualified_name_fields(
    field: str,
    invalid: object,
    message: str,
) -> None:
    expression = object.__new__(llir.QualifiedName)
    object.__setattr__(expression, "namespace", "torch")
    object.__setattr__(expression, "name", "kInt")
    object.__setattr__(
        expression,
        "data_type",
        llir.DataType.TORCH_SCALAR_TYPE,
    )
    object.__setattr__(expression, field, invalid)

    with pytest.raises(CodegenError, match=message):
        LLIRLowerer().lower_llir(expression)


def test_codegen_rejects_unknown_qualified_name_subclass() -> None:
    class UnknownQualifiedName(llir.QualifiedName):
        pass

    expression = llir.FunctionCall(
        "consume",
        (UnknownQualifiedName("torch", "kInt", llir.DataType.TORCH_SCALAR_TYPE),),
    )

    with pytest.raises(CodegenError, match="UnknownQualifiedName"):
        LLIRLowerer().lower_llir(expression)


def test_function_call_is_frozen_typed_owned_and_structurally_equal() -> None:
    argument = _var("values")
    caller_args = [argument]
    call = llir.FunctionCall("std::move", caller_args)
    tuple_call = llir.FunctionCall("std::move", (_var("values"),))

    caller_args.append(_var("later"))

    assert call.name == "std::move"
    assert type(call.args) is tuple
    assert call.args == (argument,)
    assert call == tuple_call
    assert call != llir.FunctionCall("std::move", (_var("other"),))
    assert call != llir.FunctionCall("other", (_var("values"),))
    assert get_type_hints(llir.FunctionCall) == {
        "name": str,
        "args": Tuple[llir.Expr, ...],
    }

    with pytest.raises(FrozenInstanceError):
        call.name = "other"
    with pytest.raises(FrozenInstanceError):
        call.args = ()


@pytest.mark.parametrize("name", ("", "   ", 1, None))
def test_function_call_rejects_malformed_names(name: object) -> None:
    with pytest.raises(TypeError, match="FunctionCall.name"):
        llir.FunctionCall(cast(str, name))


@pytest.mark.parametrize("arguments", ("argument", {"argument"}, {_var("value")}))
def test_function_call_rejects_malformed_argument_containers(
    arguments: object,
) -> None:
    with pytest.raises(TypeError, match="FunctionCall.args must be a list or tuple"):
        llir.FunctionCall("call", cast(list[llir.Expr], arguments))


def test_function_call_rejects_non_expression_arguments() -> None:
    with pytest.raises(TypeError, match="contain only LLIR expressions"):
        llir.FunctionCall("call", [cast(llir.Expr, "argument")])


def test_member_call_is_frozen_typed_owned_and_structurally_equal() -> None:
    base = llir.Var("tensor", llir.DataType.TORCH_TENSOR)
    argument = llir.Var("offset", llir.DataType.INT64)
    caller_template_args = [llir.DataType.FLOAT32]
    caller_args = [argument]
    call = llir.MemberCall(
        base=base,
        member="data_ptr",
        template_args=caller_template_args,
        args=caller_args,
    )
    equal = llir.MemberCall(
        base=llir.Var("tensor", llir.DataType.TORCH_TENSOR),
        member="data_ptr",
        template_args=(llir.DataType.FLOAT32,),
        args=(llir.Var("offset", llir.DataType.INT64),),
    )

    caller_template_args.append(llir.DataType.INT)
    caller_args.append(_var("later"))

    assert call.base is base
    assert call.member == "data_ptr"
    assert type(call.template_args) is tuple
    assert call.template_args == (llir.DataType.FLOAT32,)
    assert type(call.args) is tuple
    assert call.args == (argument,)
    assert call == equal
    assert hash(call) == hash(equal)
    assert call != llir.MemberCall(
        llir.Var("other", llir.DataType.TORCH_TENSOR),
        "data_ptr",
        (llir.DataType.FLOAT32,),
        (llir.Var("offset", llir.DataType.INT64),),
    )
    assert call != llir.MemberCall(
        llir.Var("tensor", llir.DataType.TORCH_TENSOR),
        "data_ptr",
        (llir.DataType.FLOAT64,),
        (llir.Var("offset", llir.DataType.INT64),),
    )
    assert get_type_hints(llir.MemberCall) == {
        "base": llir.Expr,
        "member": str,
        "template_args": Tuple[llir.DataType, ...],
        "args": Tuple[llir.Expr, ...],
    }

    with pytest.raises(FrozenInstanceError):
        call.base = _var("other")
    with pytest.raises(FrozenInstanceError):
        call.member = "data"
    with pytest.raises(FrozenInstanceError):
        call.template_args = ()
    with pytest.raises(FrozenInstanceError):
        call.args = ()


def test_member_call_defaults_are_independent_empty_immutable_tuples() -> None:
    first = llir.MemberCall(_var("first"), "data")
    second = llir.MemberCall(_var("second"), "data")

    assert first.template_args == second.template_args == ()
    assert first.args == second.args == ()
    assert type(first.template_args) is tuple
    assert type(first.args) is tuple


@pytest.mark.parametrize("member", ("", "first.second", 1, None))
def test_member_call_rejects_malformed_base_and_member(member: object) -> None:
    with pytest.raises(TypeError, match="MemberCall.member"):
        llir.MemberCall(_var("tensor"), cast(str, member))

    with pytest.raises(TypeError, match="MemberCall.base"):
        llir.MemberCall(cast(llir.Expr, "tensor"), "data")


@pytest.mark.parametrize(
    "template_args",
    ("float", {llir.DataType.FLOAT32}, llir.DataType.FLOAT32),
)
def test_member_call_rejects_malformed_template_argument_containers(
    template_args: object,
) -> None:
    with pytest.raises(TypeError, match="template_args must be a list or tuple"):
        llir.MemberCall(
            _var("tensor"),
            "data_ptr",
            template_args=cast(Any, template_args),
        )


def test_member_call_rejects_malformed_template_arguments_and_call_arguments() -> None:
    with pytest.raises(TypeError, match="contain only DataType values"):
        llir.MemberCall(
            _var("tensor"),
            "data_ptr",
            template_args=[cast(llir.DataType, "float")],
        )
    with pytest.raises(TypeError, match="args must be a list or tuple"):
        llir.MemberCall(_var("tensor"), "at", args=cast(Any, "index"))
    with pytest.raises(TypeError, match="contain only LLIR expressions"):
        llir.MemberCall(
            _var("tensor"),
            "at",
            args=[cast(llir.Expr, "index")],
        )


def test_array_is_frozen_typed_owned_and_structurally_equal() -> None:
    value = _var("extent")
    caller_values = [value]
    array = llir.Array(caller_values, llir.DataType.INT64)
    tuple_array = llir.Array((_var("extent"),), llir.DataType.INT64)

    caller_values.append(_var("later"))

    assert type(array.values) is tuple
    assert array.values == (value,)
    assert array.data_type is llir.DataType.INT64
    assert array == tuple_array
    assert hash(array) == hash(tuple_array)
    assert array != llir.Array((_var("other"),), llir.DataType.INT64)
    assert array != llir.Array((_var("extent"),), llir.DataType.INT)
    assert get_type_hints(llir.Array) == {
        "values": Tuple[llir.Expr, ...],
        "data_type": llir.DataType,
    }

    with pytest.raises(FrozenInstanceError):
        array.values = ()
    with pytest.raises(FrozenInstanceError):
        array.data_type = llir.DataType.INT


@pytest.mark.parametrize("values", ("value", {_var("value")}, None))
def test_array_rejects_malformed_value_containers(values: object) -> None:
    with pytest.raises(TypeError, match="Array.values must be a list or tuple"):
        llir.Array(cast(Any, values), llir.DataType.INT64)


def test_array_rejects_malformed_children_and_data_type() -> None:
    with pytest.raises(TypeError, match="contain only LLIR expressions"):
        llir.Array([cast(llir.Expr, "extent")], llir.DataType.INT64)
    with pytest.raises(TypeError, match="Array.data_type"):
        llir.Array([_var("extent")], cast(llir.DataType, "int64_t"))


def test_array_codegen_is_byte_exact_for_empty_nested_and_expression_values() -> None:
    empty = llir.Array((), llir.DataType.INT64)
    nested = llir.FunctionCall(
        "consume",
        [
            llir.Array(
                (
                    llir.Array((_var("row"),), llir.DataType.INT64),
                    llir.Add(_var("column"), llir.Literal(1)),
                ),
                llir.DataType.INT64,
            )
        ],
    )

    assert LLIRLowerer().lower_llir(empty) == "{}"
    assert LLIRLowerer().lower_llir(nested) == "consume({{row}, column + 1})"


@pytest.mark.parametrize(
    "expression",
    (
        llir.BinOp(
            "+",
            llir.Array((_var("value"),), llir.DataType.INT),
            _var("offset"),
        ),
        llir.MemberAccess(
            llir.Array((_var("value"),), llir.DataType.INT),
            "member",
        ),
        llir.ArrayAccess(
            llir.Array((_var("value"),), llir.DataType.INT),
            _var("index"),
        ),
    ),
)
def test_array_codegen_rejects_operand_contexts_without_a_cpp_precedence(
    expression: llir.Expr,
) -> None:
    with pytest.raises(CodegenError, match="precedence.*Array"):
        LLIRLowerer().lower_llir(expression)


@pytest.mark.parametrize("malformation", ("values", "value", "data_type"))
def test_codegen_rejects_forged_array_fields(malformation: str) -> None:
    array = object.__new__(llir.Array)
    object.__setattr__(array, "values", (_var("extent"),))
    object.__setattr__(array, "data_type", llir.DataType.INT64)
    if malformation == "values":
        object.__setattr__(array, "values", [_var("extent")])
        expected = "Array.values must be a tuple"
    elif malformation == "value":
        object.__setattr__(array, "values", ("extent",))
        expected = "contain only LLIR expressions"
    else:
        object.__setattr__(array, "data_type", "int64_t")
        expected = "Array.data_type"

    with pytest.raises(CodegenError, match=expected):
        LLIRLowerer().lower_llir(array)


def test_codegen_rejects_array_subclasses_and_unknown_children() -> None:
    class UnknownArray(llir.Array):
        pass

    class UnknownExpr(llir.Expr):
        pass

    with pytest.raises(CodegenError, match="UnknownArray"):
        LLIRLowerer().lower_llir(UnknownArray((_var("extent"),), llir.DataType.INT64))
    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(llir.Array((UnknownExpr(),), llir.DataType.INT64))


def test_nested_function_call_codegen_is_byte_exact_and_postfix_correct() -> None:
    conversion = llir.FunctionCall(
        "scorch_tensor_from_vector",
        [
            llir.FunctionCall(
                "std::move",
                [llir.Var("Result_values", llir.DataType.STD_VECTOR_FLOAT32)],
            ),
            llir.Var("torch::kFloat32", llir.DataType.NO_TYPE),
        ],
    )
    indexed_call = llir.ArrayAccess(
        llir.FunctionCall("factory", [llir.Add(_var("a"), _var("b"))]),
        llir.Literal(0),
    )

    assert LLIRLowerer().lower_llir(conversion) == (
        "scorch_tensor_from_vector(std::move(Result_values), torch::kFloat32)"
    )
    assert LLIRLowerer().lower_llir(indexed_call) == "factory(a + b)[0]"


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        (
            llir.MemberCall(
                llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
                "data",
            ),
            "storage.data()",
        ),
        (
            llir.MemberCall(
                llir.Var("tensor", llir.DataType.TORCH_TENSOR),
                "data_ptr",
                template_args=(llir.DataType.FLOAT32,),
            ),
            "tensor.data_ptr<float>()",
        ),
        (
            llir.MemberCall(
                llir.ArrayAccess(_var("tensors"), _var("i")),
                "data_ptr",
                template_args=(llir.DataType.INT,),
            ),
            "tensors[i].data_ptr<int>()",
        ),
        (
            llir.MemberCall(
                llir.MemberAccess(_var("owner"), "storage"),
                "at",
                template_args=(llir.DataType.INT, llir.DataType.INT64),
                args=(llir.Add(_var("i"), llir.Literal(1)),),
            ),
            "owner.storage.at<int, int64_t>(i + 1)",
        ),
        (
            llir.MemberCall(
                llir.BinOp("+", _var("owner"), _var("offset")),
                "data",
            ),
            "(owner + offset).data()",
        ),
        (
            llir.MemberCall(llir.FunctionCall("factory"), "data"),
            "factory().data()",
        ),
    ),
)
def test_member_call_codegen_is_byte_exact_typed_and_precedence_correct(
    expression: llir.MemberCall,
    expected: str,
) -> None:
    assert LLIRLowerer().lower_llir(expression) == expected


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
        "var": Union[llir.Var, llir.MemberAccess, llir.ArrayAccess],
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
        (llir.Literal(1), "exact Var, MemberAccess, or ArrayAccess"),
        (
            llir.FunctionCall("target"),
            "exact Var, MemberAccess, or ArrayAccess",
        ),
        (
            llir.QualifiedName("torch", "kInt", llir.DataType.TORCH_SCALAR_TYPE),
            "exact Var, MemberAccess, or ArrayAccess",
        ),
        (_var("values[i]"), "assignment Var.name must be an identifier or member path"),
        (_var("call()"), "assignment Var.name must be an identifier or member path"),
        (
            _var("left + right"),
            "assignment Var.name must be an identifier or member path",
        ),
        (_var("42"), "assignment Var.name must be an identifier or member path"),
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


def test_assign_rejects_forged_cast_index_data_type() -> None:
    malformed_cast = llir.Cast(_var("i"), llir.DataType.INT)
    object.__setattr__(malformed_cast, "data_type", "int")

    with pytest.raises(TypeError, match="Cast.data_type must be a DataType"):
        llir.Assign(
            var=llir.ArrayAccess(_var("values"), malformed_cast),
            value=llir.Literal(1),
        )


def test_assign_rejects_target_subclasses_and_invalid_fields() -> None:
    class UnknownVar(llir.Var):
        pass

    class UnknownArrayAccess(llir.ArrayAccess):
        pass

    class UnknownMemberAccess(llir.MemberAccess):
        pass

    with pytest.raises(TypeError, match="exact Var, MemberAccess, or ArrayAccess"):
        llir.Assign(UnknownVar("value", llir.DataType.INT), llir.Literal(1))
    with pytest.raises(TypeError, match="exact Var, MemberAccess, or ArrayAccess"):
        llir.Assign(
            UnknownArrayAccess(_var("values"), _var("i")),
            llir.Literal(1),
        )
    with pytest.raises(TypeError, match="exact Var, MemberAccess, or ArrayAccess"):
        llir.Assign(
            UnknownMemberAccess(_var("value"), "member"),
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


def test_nested_member_assignment_target_is_structured_and_byte_exact() -> None:
    target = llir.MemberAccess(
        llir.MemberAccess(
            llir.MemberAccess(
                llir.Var("Result", llir.DataType.TACO_TENSOR),
                "storage",
            ),
            "index",
        ),
        "mode_indices",
    )
    value = llir.Array(
        (
            llir.Array((), llir.DataType.STD_VECTOR_TORCH_TENSOR),
            llir.Array(
                (
                    llir.Var("Result1_pos_torch", llir.DataType.TORCH_TENSOR),
                    llir.Var("Result1_crd_torch", llir.DataType.TORCH_TENSOR),
                ),
                llir.DataType.STD_VECTOR_TORCH_TENSOR,
            ),
        ),
        llir.DataType.STD_VECTOR_2D_TORCH_TENSOR,
    )
    assignment = llir.Assign(target, value)

    assert assignment.var is target
    assert LLIRLowerer().lower_llir(assignment) == (
        "Result.storage.index.mode_indices = "
        "{{}, {Result1_pos_torch, Result1_crd_torch}};"
    )

    # The transitional grammar still accepts an existing dotted Var lvalue;
    # the production string budget separately prevents migrated producers from
    # reintroducing one.
    compatibility_assignment = llir.Assign(
        _var("Result.storage.value"),
        _var("Result_values_torch"),
    )
    assert LLIRLowerer().lower_llir(compatibility_assignment) == (
        "Result.storage.value = Result_values_torch;"
    )


def test_nested_member_assignment_rejects_invalid_roots_and_bases() -> None:
    class UnknownVar(llir.Var):
        pass

    class UnknownMemberAccess(llir.MemberAccess):
        pass

    invalid_targets = (
        llir.MemberAccess(
            llir.BinOp("+", _var("Result"), _var("offset")),
            "storage",
        ),
        llir.MemberAccess(
            UnknownVar("Result", llir.DataType.TACO_TENSOR),
            "storage",
        ),
        llir.MemberAccess(
            UnknownMemberAccess(
                llir.Var("Result", llir.DataType.TACO_TENSOR),
                "storage",
            ),
            "value",
        ),
    )
    for target in invalid_targets:
        with pytest.raises(TypeError, match="exact Var root through exact"):
            llir.Assign(target, llir.Literal(1))


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
        malformed_cast = llir.Cast(_var("i"), llir.DataType.INT)
        object.__setattr__(malformed_cast, "data_type", "int")
        target = object.__new__(llir.ArrayAccess)
        object.__setattr__(target, "array", _var("values"))
        object.__setattr__(target, "index", malformed_cast)
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


def test_codegen_rejects_member_call_subclasses_and_unknown_children() -> None:
    class UnknownMemberCall(llir.MemberCall):
        pass

    class UnknownExpr(llir.Expr):
        pass

    with pytest.raises(CodegenError, match="UnknownMemberCall"):
        LLIRLowerer().lower_llir(UnknownMemberCall(_var("tensor"), "data"))
    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(llir.MemberCall(UnknownExpr(), "data"))
    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(
            llir.MemberCall(_var("tensor"), "at", args=(UnknownExpr(),))
        )


def test_codegen_rejects_function_call_subclasses_and_unknown_children() -> None:
    class UnknownFunctionCall(llir.FunctionCall):
        pass

    class UnknownExpr(llir.Expr):
        pass

    with pytest.raises(CodegenError, match="UnknownFunctionCall"):
        LLIRLowerer().lower_llir(UnknownFunctionCall("call"))
    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(llir.FunctionCall("call", [UnknownExpr()]))


@pytest.mark.parametrize("malformation", ("name", "args", "argument"))
def test_codegen_rejects_forged_function_call_fields(malformation: str) -> None:
    call = object.__new__(llir.FunctionCall)
    object.__setattr__(call, "name", "call")
    object.__setattr__(call, "args", (_var("argument"),))
    if malformation == "name":
        object.__setattr__(call, "name", " ")
        expected = "FunctionCall.name"
    elif malformation == "args":
        object.__setattr__(call, "args", [_var("argument")])
        expected = "FunctionCall.args must be a tuple"
    else:
        object.__setattr__(call, "args", ("argument",))
        expected = "contain only LLIR expressions"

    with pytest.raises(CodegenError, match=expected):
        LLIRLowerer().lower_llir(call)


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


@pytest.mark.parametrize(
    "malformation",
    ("base", "member", "template_args", "template_arg", "args", "argument"),
)
def test_codegen_rejects_forged_member_call_fields(malformation: str) -> None:
    call = object.__new__(llir.MemberCall)
    object.__setattr__(call, "base", _var("tensor"))
    object.__setattr__(call, "member", "data_ptr")
    object.__setattr__(call, "template_args", (llir.DataType.FLOAT32,))
    object.__setattr__(call, "args", (_var("argument"),))
    if malformation == "base":
        object.__setattr__(call, "base", "tensor")
        expected = "MemberCall.base"
    elif malformation == "member":
        object.__setattr__(call, "member", "tensor.data_ptr")
        expected = "MemberCall.member"
    elif malformation == "template_args":
        object.__setattr__(call, "template_args", [llir.DataType.FLOAT32])
        expected = "MemberCall.template_args"
    elif malformation == "template_arg":
        object.__setattr__(call, "template_args", ("float",))
        expected = "MemberCall.template_args"
    elif malformation == "args":
        object.__setattr__(call, "args", [_var("argument")])
        expected = "MemberCall.args"
    else:
        object.__setattr__(call, "args", ("argument",))
        expected = "MemberCall.args"

    with pytest.raises(CodegenError, match=expected):
        LLIRLowerer().lower_llir(call)


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
