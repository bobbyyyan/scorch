from dataclasses import FrozenInstanceError
from typing import Any, Optional, Tuple, Union, cast, get_type_hints

import pytest

from scorch.compiler import llir
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.diagnostics import CodegenError
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]


def _var(name: str) -> llir.Var:
    return llir.Var(name=name, type=llir.DataType.INT)


def _forged_sizeof(data_type: object) -> llir.Sizeof:
    expression = object.__new__(llir.Sizeof)
    object.__setattr__(expression, "data_type", data_type)
    return expression


def _result_metadata(access_id: int = 1) -> llir.TensorAccessMetadata:
    return llir.TensorAccessMetadata(
        access_id=AccessId(access_id),
        tensor_id=SymbolId(2),
        index_ids=(IndexId(3),),
        role=llir.TensorAccessRole.RESULT_WRITE,
    )


def test_llir_does_not_expose_legacy_duplicate_codegen() -> None:
    assert not hasattr(llir, "NodeVisitor")
    assert not hasattr(llir, "CppCodeGenerator")


def test_codegen_rejects_unknown_statement_node() -> None:
    class UnknownStmt(llir.Stmt):
        pass

    with pytest.raises(CodegenError, match="UnknownStmt"):
        LLIRLowerer().lower_llir(UnknownStmt())


def test_codegen_rejects_cyclic_statement_containers() -> None:
    statements: list[object] = []
    statements.append(statements)

    with pytest.raises(CodegenError, match=r"root\[0\] must be acyclic"):
        LLIRLowerer().lower_llir(cast(list[llir.Stmt], statements))


def test_codegen_exact_tree_scan_allows_shared_expression_dags() -> None:
    shared = _var("position")

    assert LLIRLowerer().lower_llir(llir.Add(shared, shared)) == ("position + position")


@pytest.mark.parametrize(
    "name",
    (
        "TensorProperty",
        "GetTensorProperty",
        "Allocate",
        "Free",
        "Print",
        "Case",
        "Switch",
    ),
)
def test_llir_does_not_expose_non_emittable_schema(name: str) -> None:
    assert not hasattr(llir, name)


def test_codegen_rejects_unknown_expression_node() -> None:
    class UnknownExpr(llir.Expr):
        pass

    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(UnknownExpr())


def test_codegen_accepts_exact_list_and_tuple_statement_sequences() -> None:
    statements = [
        llir.Break(),
        (llir.Continue(), [llir.Return(llir.Literal(1))]),
    ]
    expected = "break;\ncontinue;\nreturn 1;"

    assert LLIRLowerer().lower_llir(statements) == expected
    assert LLIRLowerer().lower_llir(tuple(statements)) == expected


@pytest.mark.parametrize(
    "value",
    (
        llir.Literal(1),
        "raw",
    ),
)
def test_codegen_statement_sequences_reject_non_statement_members(
    value: object,
) -> None:
    with pytest.raises(CodegenError, match="must contain exact LLIR statements"):
        LLIRLowerer().lower_llir(cast(Any, [value]))


def test_codegen_rejects_string_and_sequence_subclasses() -> None:
    class UnknownString(str):
        pass

    class UnknownList(list):
        pass

    class UnknownTuple(tuple):
        pass

    lowerer = LLIRLowerer()
    with pytest.raises(CodegenError, match="UnknownString"):
        lowerer.lower_llir(cast(Any, UnknownString("raw")))
    with pytest.raises(CodegenError, match="UnknownList"):
        lowerer.lower_llir(cast(Any, UnknownList([llir.Break()])))
    with pytest.raises(CodegenError, match="UnknownTuple"):
        lowerer.lower_llir(cast(Any, UnknownTuple((llir.Break(),))))


def test_direct_codegen_helpers_reject_node_subclasses() -> None:
    class UnknownVar(llir.Var):
        pass

    class UnknownForLoop(llir.ForLoop):
        pass

    class UnknownIfThenElse(llir.IfThenElse):
        pass

    class UnknownFunction(llir.Function):
        pass

    lowerer = LLIRLowerer()
    with pytest.raises(CodegenError, match="UnknownVar"):
        lowerer.lower_expression(UnknownVar("value", llir.DataType.INT))
    with pytest.raises(CodegenError, match="UnknownForLoop"):
        lowerer.lower_loop_construct(
            UnknownForLoop(
                None,
                _var("condition"),
                llir.Increment(_var("index")),
                [],
            )
        )
    with pytest.raises(CodegenError, match="UnknownIfThenElse"):
        lowerer.lower_conditional(
            UnknownIfThenElse(cond=_var("condition"), then_body=[llir.Break()])
        )
    with pytest.raises(CodegenError, match="UnknownFunction"):
        lowerer.lower_function_definition(
            UnknownFunction(llir.DataType.VOID, "function", [], [])
        )


def test_codegen_rejects_unknown_nested_node_subclasses() -> None:
    class UnknownVar(llir.Var):
        pass

    class UnknownLiteral(llir.Literal):
        pass

    class UnknownBreak(llir.Break):
        pass

    unknown_var = UnknownVar("value", llir.DataType.INT)
    unknown_literal = UnknownLiteral(1)
    statements = (
        llir.VarDecl(unknown_var),
        llir.VarInit(_var("value"), unknown_literal),
        llir.Assign(_var("value"), unknown_literal),
        llir.Function(
            llir.DataType.VOID,
            "bad_argument",
            [unknown_var],
            [],
        ),
        llir.Function(
            llir.DataType.VOID,
            "bad_body",
            [],
            [UnknownBreak()],
        ),
    )

    lowerer = LLIRLowerer()
    for statement in statements:
        with pytest.raises(CodegenError):
            lowerer.lower_llir(statement)


def test_codegen_rejects_nested_sequence_subclasses() -> None:
    class UnknownList(list):
        pass

    function_args = llir.Function(
        llir.DataType.VOID,
        "function_args",
        cast(Any, UnknownList()),
        [],
    )
    call_args = llir.FunctionCallStmt("call", [])
    object.__setattr__(call_args, "args", UnknownList([llir.Literal(1)]))
    conditions = llir.IfThenElse(
        cond_list=cast(Any, UnknownList([_var("condition")])),
        then_body_list=[[llir.Break()]],
    )
    optional_region = llir.ForLoop(
        None,
        _var("condition"),
        llir.Increment(_var("index")),
        [],
        before_parallel_body=cast(Any, UnknownList()),
    )

    lowerer = LLIRLowerer()
    for node in (function_args, call_args, conditions, optional_region):
        with pytest.raises(CodegenError, match="list/tuple subclass"):
            lowerer.lower_llir(node)


@pytest.mark.parametrize(
    "node",
    (
        llir.WhileLoop(cast(Any, llir.Break()), []),
        llir.IfThenElse(cond=cast(Any, llir.Break()), then_body=[llir.Break()]),
        llir.WhileLoop(_var("condition"), cast(Any, llir.Break())),
        llir.Function(
            llir.DataType.VOID,
            "body",
            [],
            cast(Any, llir.Break()),
        ),
        llir.IfThenElse(
            cond=_var("condition"),
            then_body=cast(Any, llir.Break()),
        ),
        llir.IfThenElse(
            cond_list=[_var("condition")],
            then_body_list=cast(Any, [llir.Break()]),
        ),
        llir.ForLoop(
            None,
            _var("condition"),
            llir.VarInit(_var("update"), llir.Literal(1)),
            [],
        ),
    ),
)
def test_codegen_rejects_invalid_nested_node_categories(node: llir.Node) -> None:
    with pytest.raises(CodegenError):
        LLIRLowerer().lower_llir(node)


def test_codegen_rejects_unknown_descendants_in_inactive_fields() -> None:
    class UnknownExpr(llir.Expr):
        pass

    hidden_conditional_fields = llir.IfThenElse(
        cond=llir.UnaryOp("-", UnknownExpr()),
        then_body=[llir.Return(UnknownExpr())],
        cond_list=[_var("active")],
        then_body_list=[[llir.Break()]],
    )
    hidden_else_condition = llir.IfThenElse(
        cond_list=[_var("active"), llir.UnaryOp("-", UnknownExpr())],
        then_body_list=[[llir.Break()], [llir.Continue()]],
        make_last_case_else=True,
    )
    hidden_before_parallel = llir.ForLoop(
        None,
        _var("condition"),
        llir.Increment(_var("index")),
        [],
        before_parallel_body=[llir.Return(UnknownExpr())],
    )
    hidden_hoisted_declaration = llir.ForLoop(
        None,
        _var("condition"),
        llir.Increment(_var("index")),
        [],
    )
    hidden_hoisted_declaration._hoisted_ptr_decls = [llir.Return(UnknownExpr())]

    lowerer = LLIRLowerer()
    for node in (
        hidden_conditional_fields,
        hidden_else_condition,
        hidden_before_parallel,
        hidden_hoisted_declaration,
    ):
        with pytest.raises(CodegenError, match="UnknownExpr"):
            lowerer.lower_llir(node)


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


def test_direct_init_is_frozen_typed_structural_and_byte_exact() -> None:
    target = llir.Var("packed_B_storage", llir.DataType.STD_VECTOR_FLOAT32)
    extent = llir.Mul(
        llir.Cast(
            llir.Var("kTile_j", llir.DataType.CONSTEXPR_INT),
            llir.DataType.SIZE_T,
        ),
        llir.Cast(
            llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
            llir.DataType.SIZE_T,
        ),
    )
    caller_args = [extent]
    declaration = llir.DirectInit(target, caller_args)
    equal = llir.DirectInit(
        llir.Var("packed_B_storage", llir.DataType.STD_VECTOR_FLOAT32),
        (
            llir.Mul(
                llir.Cast(
                    llir.Var("kTile_j", llir.DataType.CONSTEXPR_INT),
                    llir.DataType.SIZE_T,
                ),
                llir.Cast(
                    llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
                    llir.DataType.SIZE_T,
                ),
            ),
        ),
    )

    caller_args.append(llir.Literal(0))

    assert declaration.var is target
    assert declaration.args == (extent,)
    assert type(declaration.args) is tuple
    assert declaration == equal
    assert hash(declaration) == hash(equal)
    assert get_type_hints(llir.DirectInit) == {
        "var": llir.Var,
        "args": Tuple[llir.Expr, ...],
    }
    assert (
        LLIRLowerer().lower_llir(declaration) == "std::vector<float> packed_B_storage("
        "(size_t)kTile_j * (size_t)kTile_k);"
    )
    assert (
        LLIRLowerer().lower_llir(declaration, indent_level=2)
        == "    std::vector<float> packed_B_storage("
        "(size_t)kTile_j * (size_t)kTile_k);"
    )

    with pytest.raises(FrozenInstanceError):
        declaration.var = _var("other")
    with pytest.raises(FrozenInstanceError):
        declaration.args = (llir.Literal(1),)


def test_direct_init_emits_multiple_arguments_in_order() -> None:
    declaration = llir.DirectInit(
        llir.Var("positions", llir.DataType.STD_VECTOR_C_INT),
        (llir.Literal(4), llir.Literal(0)),
    )

    assert LLIRLowerer().lower_llir(declaration) == (
        "std::vector<int> positions(4, 0);"
    )


def test_direct_init_rejects_malformed_construction() -> None:
    class UnknownVar(llir.Var):
        pass

    metadata = _result_metadata()
    invalid_targets = (
        cast(llir.Var, llir.Literal(1)),
        UnknownVar("storage", llir.DataType.STD_VECTOR_FLOAT32),
        llir.Var("not a name", llir.DataType.STD_VECTOR_FLOAT32),
        llir.Var("storage", llir.DataType.NO_TYPE),
        llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32, is_ptr=True),
        llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32, is_restrict=True),
        llir.Var(
            "storage",
            llir.DataType.STD_VECTOR_FLOAT32,
            tensor_access=metadata,
        ),
    )
    for target in invalid_targets:
        with pytest.raises((TypeError, ValueError), match="DirectInit.var"):
            llir.DirectInit(target, (llir.Literal(1),))

    with pytest.raises(TypeError, match="DirectInit.args must be a list or tuple"):
        llir.DirectInit(
            llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
            cast(Any, {llir.Literal(1)}),
        )
    with pytest.raises(TypeError, match="DirectInit.args must be non-empty"):
        llir.DirectInit(
            llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
            (),
        )
    with pytest.raises(TypeError, match="only LLIR expressions"):
        llir.DirectInit(
            llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
            cast(Any, ("extent",)),
        )


@pytest.mark.parametrize("malformation", ("subclass", "target", "args", "child"))
def test_direct_init_codegen_rejects_unknown_and_forged_nodes(
    malformation: str,
) -> None:
    class UnknownDirectInit(llir.DirectInit):
        pass

    class UnknownVar(llir.Var):
        pass

    class UnknownExpr(llir.Expr):
        pass

    declaration: llir.DirectInit = llir.DirectInit(
        llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
        (llir.Literal(4),),
    )
    if malformation == "subclass":
        declaration = UnknownDirectInit(
            llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
            (llir.Literal(4),),
        )
        expected = "UnknownDirectInit"
    elif malformation == "target":
        object.__setattr__(
            declaration,
            "var",
            UnknownVar("storage", llir.DataType.STD_VECTOR_FLOAT32),
        )
        expected = "UnknownVar at root.var"
    elif malformation == "args":
        object.__setattr__(declaration, "args", [llir.Literal(4)])
        expected = "DirectInit.args must be a tuple"
    else:
        object.__setattr__(declaration, "args", (UnknownExpr(),))
        expected = "UnknownExpr"

    with pytest.raises(CodegenError, match=expected):
        LLIRLowerer().lower_llir(declaration)


@pytest.mark.parametrize("missing_field", ("var", "args"))
def test_direct_init_codegen_rejects_missing_forged_fields(
    missing_field: str,
) -> None:
    declaration = object.__new__(llir.DirectInit)
    if missing_field != "var":
        object.__setattr__(
            declaration,
            "var",
            llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
        )
    if missing_field != "args":
        object.__setattr__(declaration, "args", (llir.Literal(4),))

    with pytest.raises(CodegenError, match=f"DirectInit.{missing_field}"):
        LLIRLowerer().lower_llir(declaration)


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

    assert LLIRLowerer().lower_llir(expression) == "(float)(a + b)"


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


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (
            llir.Select(_var("a"), _var("b"), _var("c")),
            "a ? b : c",
        ),
        (
            llir.Select(
                llir.BinOp("||", _var("a"), _var("b")),
                _var("c"),
                _var("d"),
            ),
            "a || b ? c : d",
        ),
        (
            llir.Select(
                llir.Select(_var("a"), _var("b"), _var("c")),
                _var("d"),
                _var("e"),
            ),
            "(a ? b : c) ? d : e",
        ),
        (
            llir.Select(
                _var("a"),
                llir.Select(_var("b"), _var("c"), _var("d")),
                _var("e"),
            ),
            "a ? b ? c : d : e",
        ),
        (
            llir.Select(
                _var("a"),
                _var("b"),
                llir.Select(_var("c"), _var("d"), _var("e")),
            ),
            "a ? b : c ? d : e",
        ),
        (
            llir.Mul(
                llir.Cast(_var("work"), llir.DataType.LONG),
                llir.Select(
                    llir.BinOp(">", _var("rows"), llir.Literal(0)),
                    llir.Add(
                        llir.BinOp("/", _var("nnz"), _var("rows")),
                        llir.Literal(1),
                    ),
                    llir.Literal(1),
                ),
            ),
            "(long)work * (rows > 0 ? nnz / rows + 1 : 1)",
        ),
        (
            llir.Cast(
                llir.Select(_var("a"), _var("b"), _var("c")),
                llir.DataType.SIZE_T,
            ),
            "(size_t)(a ? b : c)",
        ),
        (
            llir.FunctionCall(
                "scorch_nthreads",
                [
                    llir.Select(_var("a"), _var("b"), _var("c")),
                    _var("rows"),
                ],
            ),
            "scorch_nthreads(a ? b : c, rows)",
        ),
    ],
)
def test_select_codegen_preserves_ast_precedence(
    expression: llir.Expr, expected: str
) -> None:
    assert LLIRLowerer().lower_llir(expression) == expected


def test_select_is_frozen_typed_and_structurally_equal() -> None:
    condition = _var("cond")
    expression = llir.Select(condition, _var("a"), _var("b"))
    equal = llir.Select(_var("cond"), _var("a"), _var("b"))

    assert expression.cond is condition
    assert expression == equal
    assert hash(expression) == hash(equal)
    assert expression != llir.Select(_var("other"), _var("a"), _var("b"))
    assert expression != llir.Select(_var("cond"), _var("a"), _var("other"))
    assert get_type_hints(llir.Select) == {
        "cond": llir.Expr,
        "when_true": llir.Expr,
        "when_false": llir.Expr,
    }

    with pytest.raises(FrozenInstanceError):
        expression.cond = _var("other")
    with pytest.raises(FrozenInstanceError):
        expression.when_true = _var("other")
    with pytest.raises(FrozenInstanceError):
        expression.when_false = _var("other")


@pytest.mark.parametrize(
    ("cond", "when_true", "when_false", "message"),
    (
        ("cond", _var("a"), _var("b"), "Select.cond"),
        (_var("cond"), "a", _var("b"), "Select.when_true"),
        (_var("cond"), _var("a"), "b", "Select.when_false"),
    ),
)
def test_select_rejects_malformed_constructor_fields(
    cond: object,
    when_true: object,
    when_false: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        llir.Select(
            cast(llir.Expr, cond),
            cast(llir.Expr, when_true),
            cast(llir.Expr, when_false),
        )


@pytest.mark.parametrize("malformation", ("cond", "when_true", "when_false"))
def test_codegen_rejects_forged_select_fields(malformation: str) -> None:
    expression = llir.Select(_var("cond"), _var("a"), _var("b"))
    object.__setattr__(expression, malformation, "forged")

    with pytest.raises(CodegenError, match=f"Select.{malformation}"):
        LLIRLowerer().lower_llir(expression)


@pytest.mark.parametrize("missing_field", ("cond", "when_true", "when_false"))
def test_codegen_rejects_forged_select_missing_fields(missing_field: str) -> None:
    expression = llir.Select(_var("cond"), _var("a"), _var("b"))
    object.__delattr__(expression, missing_field)

    with pytest.raises(CodegenError, match=f"Select.{missing_field}"):
        LLIRLowerer().lower_llir(expression)


def test_codegen_rejects_unknown_select_subclass() -> None:
    class UnknownSelect(llir.Select):
        pass

    with pytest.raises(CodegenError, match="UnknownSelect"):
        LLIRLowerer().lower_llir(UnknownSelect(_var("cond"), _var("a"), _var("b")))


def test_codegen_rejects_unknown_select_child() -> None:
    class UnknownExpr(llir.Expr):
        pass

    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(llir.Select(_var("cond"), UnknownExpr(), _var("b")))


def test_sizeof_is_frozen_typed_and_structurally_equal() -> None:
    expression = llir.Sizeof(llir.DataType.FLOAT32)
    equal = llir.Sizeof(llir.DataType.FLOAT32)

    assert expression.data_type is llir.DataType.FLOAT32
    assert expression == equal
    assert hash(expression) == hash(equal)
    assert expression != llir.Sizeof(llir.DataType.FLOAT64)
    assert get_type_hints(llir.Sizeof) == {"data_type": llir.DataType}

    with pytest.raises(FrozenInstanceError):
        expression.data_type = llir.DataType.FLOAT64


def test_sizeof_rejects_malformed_constructor_data_type() -> None:
    with pytest.raises(TypeError, match="Sizeof.data_type must be a DataType"):
        llir.Sizeof(cast(llir.DataType, "float"))


@pytest.mark.parametrize("malformation", ("invalid", "missing"))
@pytest.mark.parametrize("nested_in_zero_fill", (False, True))
def test_codegen_rejects_forged_sizeof_data_type(
    malformation: str,
    nested_in_zero_fill: bool,
) -> None:
    expression = (
        _forged_sizeof("float")
        if malformation == "invalid"
        else object.__new__(llir.Sizeof)
    )
    ir: llir.Expr | llir.Stmt = expression
    if nested_in_zero_fill:
        ir = llir.FunctionCallStmt(
            "memset",
            (
                _var("workspace"),
                llir.Literal(0),
                llir.Mul(_var("size"), expression),
            ),
        )

    with pytest.raises(CodegenError, match="Sizeof.data_type must be a DataType"):
        LLIRLowerer().lower_llir(ir)


def _forged_address_of(operand: object) -> llir.AddressOf:
    expression = object.__new__(llir.AddressOf)
    object.__setattr__(expression, "operand", operand)
    return expression


def _without_instance_field(value: object, field: str) -> object:
    forged = object.__new__(type(value))
    for name, item in vars(value).items():
        if name != field:
            object.__setattr__(forged, name, item)
    return forged


def _malformed_address_operand(malformation: str) -> object:
    if malformation == "member_cycle":
        cyclic_member = object.__new__(llir.MemberAccess)
        object.__setattr__(cyclic_member, "base", cyclic_member)
        object.__setattr__(cyclic_member, "member", "field")
        return cyclic_member
    if malformation == "index_array_cycle":
        cyclic_access = object.__new__(llir.ArrayAccess)
        object.__setattr__(cyclic_access, "array", _var("values"))
        object.__setattr__(cyclic_access, "index", cyclic_access)
        object.__setattr__(cyclic_access, "tensor_access", None)
        return cyclic_access
    if malformation == "index_binop_cycle":
        cyclic_binary = object.__new__(llir.BinOp)
        object.__setattr__(cyclic_binary, "op", "+")
        object.__setattr__(cyclic_binary, "left", cyclic_binary)
        object.__setattr__(cyclic_binary, "right", llir.Literal(1))
        return llir.ArrayAccess(_var("values"), cyclic_binary)
    if malformation == "index_unary_op_subclass":

        class StringSubclass(str):
            pass

        unary = llir.UnaryOp("+", _var("position"))
        unary.op = StringSubclass("+")
        return llir.ArrayAccess(_var("values"), unary)
    if malformation.startswith("index_var_"):
        field = malformation.removeprefix("index_var_")
        index = _var("position")
        if field.startswith("invalid_"):
            setattr(index, field.removeprefix("invalid_"), "invalid")
        else:
            index = cast(llir.Var, _without_instance_field(index, field))
        return llir.ArrayAccess(_var("values"), index)
    if malformation == "index_array_tensor_access":
        nested = _without_instance_field(
            llir.ArrayAccess(_var("indices"), _var("position")),
            "tensor_access",
        )
        return llir.ArrayAccess(
            _var("values"),
            cast(llir.ArrayAccess, nested),
        )
    if malformation.startswith("var_"):
        return _without_instance_field(
            _var("value"),
            malformation.removeprefix("var_"),
        )
    if malformation.startswith("member_"):
        return _without_instance_field(
            llir.MemberAccess(_var("value"), "field"),
            malformation.removeprefix("member_"),
        )
    if malformation == "metadata_role":
        metadata = _without_instance_field(_result_metadata(), "role")
        return llir.ArrayAccess(
            _var("values"),
            _var("position"),
            tensor_access=cast(llir.TensorAccessMetadata, metadata),
        )
    return _without_instance_field(
        llir.ArrayAccess(_var("values"), _var("position")),
        malformation.removeprefix("array_"),
    )


def _write_back_destination() -> llir.AddressOf:
    return llir.AddressOf(
        operand=llir.ArrayAccess(
            array=llir.Var(name="C_values", type=llir.DataType.PTR_FLOAT32),
            index=llir.Mul(
                llir.Var(name="pC0", type=llir.DataType.INT64),
                llir.Var(name="C1_size", type=llir.DataType.INT64),
            ),
        ),
    )


def test_address_of_is_frozen_typed_and_structurally_equal() -> None:
    operand = llir.ArrayAccess(_var("values"), _var("position"))
    expression = llir.AddressOf(operand)
    equal = llir.AddressOf(llir.ArrayAccess(_var("values"), _var("position")))

    assert expression.operand is operand
    assert expression == equal
    assert hash(expression) == hash(equal)
    assert expression != llir.AddressOf(_var("values"))
    assert get_type_hints(llir.AddressOf) == {"operand": llir.AssignmentTarget}

    with pytest.raises(FrozenInstanceError):
        expression.operand = _var("other")


@pytest.mark.parametrize(
    "operand",
    (
        "C_values",
        llir.Literal(0),
        llir.Add(_var("base"), _var("offset")),
        llir.Sizeof(llir.DataType.FLOAT32),
        llir.FunctionCall("get_value"),
        llir.AddressOf(_var("value")),
    ),
)
def test_address_of_rejects_non_lvalue_constructor_operands(
    operand: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="AddressOf.operand must be an exact Var, MemberAccess, or ArrayAccess",
    ):
        llir.AddressOf(cast(llir.AssignmentTarget, operand))


@pytest.mark.parametrize(
    "malformation",
    (
        "var_name",
        "var_type",
        "var_is_ptr",
        "var_is_restrict",
        "var_tensor_access",
        "member_base",
        "member_member",
        "array_array",
        "array_index",
        "array_tensor_access",
        "metadata_role",
        "index_var_is_ptr",
        "index_var_is_restrict",
        "index_var_tensor_access",
        "index_var_invalid_is_ptr",
        "index_var_invalid_is_restrict",
        "index_array_tensor_access",
        "member_cycle",
        "index_array_cycle",
        "index_binop_cycle",
        "index_unary_op_subclass",
    ),
)
def test_address_of_rejects_malformed_structured_lvalues(
    malformation: str,
) -> None:
    with pytest.raises(TypeError, match="AddressOf.operand"):
        llir.AddressOf(
            cast(llir.AssignmentTarget, _malformed_address_operand(malformation))
        )


def test_address_of_accepts_structured_lvalues_with_either_metadata_role() -> None:
    for role in (
        llir.TensorAccessRole.INPUT_READ,
        llir.TensorAccessRole.RESULT_WRITE,
    ):
        metadata = llir.TensorAccessMetadata(
            access_id=AccessId(1),
            tensor_id=SymbolId(2),
            index_ids=(IndexId(3),),
            role=role,
        )
        operand = llir.ArrayAccess(
            _var("values"),
            _var("position"),
            tensor_access=metadata,
        )

        assert llir.AddressOf(operand).operand is operand


def test_codegen_renders_addressed_expressions_byte_exact() -> None:
    lowerer = LLIRLowerer()

    assert lowerer.lower_llir(_write_back_destination()) == ("&C_values[pC0 * C1_size]")
    assert lowerer.lower_llir(llir.AddressOf(_var("wksp"))) == "&wksp"
    assert (
        lowerer.lower_llir(
            llir.AddressOf(
                llir.MemberAccess(
                    llir.MemberAccess(_var("entry"), "first"),
                    "value",
                )
            )
        )
        == "&entry.first.value"
    )
    assert (
        lowerer.lower_llir(llir.Add(llir.AddressOf(_var("value")), _var("offset")))
        == "&value + offset"
    )
    assert (
        lowerer.lower_llir(
            llir.ArrayAccess(llir.AddressOf(_var("value")), _var("index"))
        )
        == "(&value)[index]"
    )

    copy = llir.FunctionCallStmt(
        "memcpy",
        (
            _write_back_destination(),
            llir.Var(name="wksp", type=llir.DataType.PTR_FLOAT32),
            llir.Mul(
                llir.Var(name="wksp0_size", type=llir.DataType.INT64),
                llir.Sizeof(llir.DataType.FLOAT32),
            ),
        ),
    )
    assert lowerer.lower_llir(copy) == (
        "memcpy(&C_values[pC0 * C1_size], wksp, wksp0_size * sizeof(float));"
    )


@pytest.mark.parametrize(
    "operand",
    (
        "C_values",
        llir.Literal(0),
        llir.Add(_var("base"), _var("offset")),
        llir.Sizeof(llir.DataType.FLOAT32),
        llir.FunctionCall("get_value"),
        llir.AddressOf(_var("value")),
    ),
)
@pytest.mark.parametrize("nested_in_write_back", (False, True))
def test_codegen_rejects_forged_non_lvalue_address_of_operand(
    operand: object,
    nested_in_write_back: bool,
) -> None:
    expression = _forged_address_of(operand)
    ir: llir.Expr | llir.Stmt = expression
    if nested_in_write_back:
        ir = llir.FunctionCallStmt(
            "memcpy",
            (
                expression,
                _var("workspace"),
                llir.Mul(_var("size"), llir.Sizeof(llir.DataType.FLOAT32)),
            ),
        )

    with pytest.raises(
        CodegenError,
        match="AddressOf.operand must be an exact Var, MemberAccess, or ArrayAccess",
    ):
        LLIRLowerer().lower_llir(ir)


@pytest.mark.parametrize("nested_in_write_back", (False, True))
def test_codegen_rejects_forged_missing_address_of_operand(
    nested_in_write_back: bool,
) -> None:
    expression = object.__new__(llir.AddressOf)
    ir: llir.Expr | llir.Stmt = expression
    if nested_in_write_back:
        ir = llir.FunctionCallStmt(
            "memcpy",
            (
                expression,
                _var("workspace"),
                llir.Mul(_var("size"), llir.Sizeof(llir.DataType.FLOAT32)),
            ),
        )

    with pytest.raises(
        CodegenError,
        match="AddressOf.operand must be an exact Var, MemberAccess, or ArrayAccess",
    ):
        LLIRLowerer().lower_llir(ir)


@pytest.mark.parametrize(
    "malformation",
    (
        "var_name",
        "var_type",
        "var_is_ptr",
        "var_is_restrict",
        "var_tensor_access",
        "member_base",
        "member_member",
        "array_array",
        "array_index",
        "array_tensor_access",
        "metadata_role",
        "index_var_is_ptr",
        "index_var_is_restrict",
        "index_var_tensor_access",
        "index_var_invalid_is_ptr",
        "index_var_invalid_is_restrict",
        "index_array_tensor_access",
        "member_cycle",
        "index_array_cycle",
        "index_binop_cycle",
        "index_unary_op_subclass",
    ),
)
@pytest.mark.parametrize("nested_in_write_back", (False, True))
def test_codegen_rejects_forged_malformed_address_of_lvalue(
    malformation: str,
    nested_in_write_back: bool,
) -> None:
    expression = _forged_address_of(_malformed_address_operand(malformation))
    ir: llir.Expr | llir.Stmt = expression
    if nested_in_write_back:
        ir = llir.FunctionCallStmt(
            "memcpy",
            (
                expression,
                _var("workspace"),
                llir.Mul(_var("size"), llir.Sizeof(llir.DataType.FLOAT32)),
            ),
        )

    with pytest.raises(
        CodegenError,
        match="AddressOf.operand|must be acyclic|string subclass",
    ):
        LLIRLowerer().lower_llir(ir)


def test_assign_rejects_address_of_subscript_index() -> None:
    with pytest.raises(
        TypeError,
        match="assignment index contains an unsupported LLIR expression",
    ):
        llir.Assign(
            var=llir.ArrayAccess(
                _var("values"),
                llir.AddressOf(_var("position")),
            ),
            value=llir.Literal(1),
        )


def test_assign_rejects_forged_cyclic_member_chain_without_hanging() -> None:
    root = llir.MemberAccess(_var("entry"), "first")
    outer = llir.MemberAccess(root, "second")
    object.__setattr__(root, "base", outer)

    with pytest.raises(TypeError, match="assignment MemberAccess chain must be"):
        llir._validate_assignment_target(outer)


def test_assign_rejects_forged_cyclic_index_with_a_boundary_error() -> None:
    index = llir.Add(_var("i"), llir.Literal(1))
    object.__setattr__(index, "left", index)

    with pytest.raises(TypeError, match="assignment index must be acyclic"):
        llir._validate_assignment_index(index)


def test_assign_rejects_forged_missing_fields_without_attribute_fallback() -> None:
    target = _var("values")
    del target.__dict__["name"]
    with pytest.raises(TypeError, match="assignment Var.name must be an identifier"):
        llir._validate_assignment_target(target)

    access_target = llir.ArrayAccess(_var("values"), _var("i"))
    del access_target.__dict__["tensor_access"]
    with pytest.raises(
        TypeError,
        match="assignment ArrayAccess metadata must be TensorAccessMetadata",
    ):
        llir._validate_assignment_target(access_target)


@pytest.mark.parametrize("target_shape", ("var", "array", "member"))
@pytest.mark.parametrize("field", ("is_ptr", "is_restrict"))
def test_assign_and_codegen_reject_missing_root_flag_fields(
    target_shape: str,
    field: str,
) -> None:
    root = _var("values")
    if target_shape == "var":
        target: llir.AssignmentTarget = root
    elif target_shape == "array":
        target = llir.ArrayAccess(root, _var("index"))
    else:
        target = llir.MemberAccess(root, "field")
    assignment = llir.Assign(target, llir.Literal(1))
    del root.__dict__[field]

    with pytest.raises(TypeError, match=field):
        llir._validate_assignment_target(target)
    with pytest.raises(CodegenError, match=field):
        LLIRLowerer().lower_llir(assignment)


def test_deep_assignment_index_fails_without_recursion_error() -> None:
    index: llir.Expr = _var("index")
    for _ in range(1_100):
        index = llir.Add(index, llir.Literal(1))
    target = llir.ArrayAccess(_var("values"), index)

    with pytest.raises(TypeError, match="maximum supported nesting depth"):
        llir._validate_assignment_index(index)
    with pytest.raises(TypeError, match="maximum supported nesting depth"):
        llir.Assign(target, llir.Literal(1))

    forged = llir.Assign(_var("placeholder"), llir.Literal(1))
    forged.var = target
    with pytest.raises(CodegenError, match="maximum supported nesting depth"):
        LLIRLowerer().lower_llir(forged)


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


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        (llir.Literal("name", llir.DataType.STRING), '"name"'),
        (llir.Literal("", llir.DataType.STRING), '""'),
        (llir.Literal("kernel_5ec9", llir.DataType.STRING), '"kernel_5ec9"'),
        (llir.Literal('quote " inside', llir.DataType.STRING), '"quote \\" inside"'),
        (llir.Literal("back\\slash", llir.DataType.STRING), '"back\\\\slash"'),
        (llir.Literal("line\nbreak", llir.DataType.STRING), '"line\\nbreak"'),
        (llir.Literal("tab\tstop", llir.DataType.STRING), '"tab\\tstop"'),
        (llir.Literal("carriage\rreturn", llir.DataType.STRING), '"carriage\\rreturn"'),
        (llir.Literal("café", llir.DataType.STRING), '"caf\\303\\251"'),
        (llir.Literal("π", llir.DataType.STRING), '"\\317\\200"'),
        (llir.Literal("π7", llir.DataType.STRING), '"\\317\\2007"'),
        (llir.Literal("𐐀", llir.DataType.STRING), '"\\360\\220\\220\\200"'),
        (
            llir.Literal("spaces, punctuation; (x <= ~y)!", llir.DataType.STRING),
            '"spaces, punctuation; (x <= ~y)!"',
        ),
        (
            llir.FunctionCall(
                "validate",
                (llir.Literal("result", llir.DataType.STRING),),
            ),
            'validate("result")',
        ),
    ),
)
def test_semantic_string_literal_codegen_is_quoted_and_byte_exact(
    expression: llir.Expr,
    expected: str,
) -> None:
    assert LLIRLowerer().lower_llir(expression) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        (llir.Literal(True), "true"),
        (llir.Literal(False), "false"),
        (llir.Literal(True, llir.DataType.BOOL), "true"),
        (llir.Literal(False, llir.DataType.BOOL), "false"),
        (
            llir.BinOp("&&", llir.Literal(True), llir.Literal(False)),
            "true && false",
        ),
    ),
)
def test_semantic_bool_literal_codegen_is_byte_exact(
    expression: llir.Expr,
    expected: str,
) -> None:
    assert LLIRLowerer().lower_llir(expression) == expected


@pytest.mark.parametrize(
    "value",
    (
        "null\x00byte",
        "escape\x1bcode",
        "delete\x7fcharacter",
        "bell\acharacter",
        "surrogate\ud800character",
    ),
)
def test_semantic_string_literal_rejects_unsupported_characters(
    value: str,
) -> None:
    with pytest.raises(CodegenError, match="unsupported character"):
        LLIRLowerer().lower_llir(llir.Literal(value, llir.DataType.STRING))


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        # The characterized raw spellings: a str value is quoted only when the
        # data type is exactly STRING, so explicit non-STRING raw spellings
        # keep their legacy unquoted emission.
        (llir.Literal("0.0f", llir.DataType.FLOAT32), "0.0f"),
        (llir.Literal("0", llir.DataType.INT), "0"),
        (llir.Literal("1024", llir.DataType.INT), "1024"),
        # A bool value is spelled true/false only when the data type is
        # exactly BOOL; a STRING data type quotes only str values.
        (llir.Literal(True, llir.DataType.INT), "True"),
        (llir.Literal(False, llir.DataType.INT32), "False"),
        (llir.Literal(1, llir.DataType.STRING), "1"),
        (llir.Literal(1, llir.DataType.BOOL), "1"),
        (llir.Literal(0), "0"),
        (llir.Literal(-3), "-3"),
        (llir.Literal(1.5), "1.5"),
    ),
)
def test_nonsemantic_literal_combinations_keep_legacy_spelling(
    expression: llir.Expr,
    expected: str,
) -> None:
    assert LLIRLowerer().lower_llir(expression) == expected


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
            "(int)torch::kInt",
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
    assert type(call.template_args) is tuple
    assert call.template_args == ()
    assert type(call.args) is tuple
    assert call.args == (argument,)
    assert call == tuple_call
    assert call != llir.FunctionCall("std::move", (_var("other"),))
    assert call != llir.FunctionCall("other", (_var("values"),))
    assert call != llir.FunctionCall(
        "std::move",
        (_var("values"),),
        template_args=(llir.DataType.INT64,),
    )
    template_list = [llir.DataType.INT64]
    templated = llir.FunctionCall("std::move", (_var("values"),), template_list)
    template_list.append(llir.DataType.INT)
    assert templated.template_args == (llir.DataType.INT64,)
    assert get_type_hints(llir.FunctionCall) == {
        "name": str,
        "template_args": Tuple[llir.DataType, ...],
        "args": Tuple[llir.Expr, ...],
    }

    with pytest.raises(FrozenInstanceError):
        call.name = "other"
    with pytest.raises(FrozenInstanceError):
        call.template_args = ()
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


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (
            llir.FunctionCall(
                "scorch_make_aligned_buffer",
                [_var("bytes")],
                template_args=(llir.DataType.FLOAT32,),
            ),
            "scorch_make_aligned_buffer<float>(bytes)",
        ),
        (
            llir.FunctionCall(
                "std::min",
                [_var("a"), _var("b")],
                template_args=(llir.DataType.INT64,),
            ),
            "std::min<int64_t>(a, b)",
        ),
        (
            llir.FunctionCall(
                "identity",
                (),
                template_args=(llir.DataType.INT, llir.DataType.INT64),
            ),
            "identity<int, int64_t>()",
        ),
        (
            llir.FunctionCall("plain", [_var("a")], template_args=()),
            "plain(a)",
        ),
    ],
)
def test_function_call_template_arguments_render_typed_spellings(
    node: llir.FunctionCall, expected: str
) -> None:
    assert LLIRLowerer().lower_llir(node) == expected


def test_function_call_stmt_template_arguments_render_typed_spellings() -> None:
    statement = llir.FunctionCallStmt(
        "scorch_zero_dense",
        [_var("target")],
        template_args=(llir.DataType.FLOAT64,),
    )

    assert LLIRLowerer().lower_llir(statement) == "scorch_zero_dense<double>(target);"
    assert (
        LLIRLowerer().lower_llir(llir.FunctionCallStmt("plain", [_var("a")]))
        == "plain(a);"
    )


@pytest.mark.parametrize("owner", (llir.FunctionCall, llir.FunctionCallStmt))
@pytest.mark.parametrize(
    "template_args",
    ("float", {llir.DataType.FLOAT32}, (llir.DataType.FLOAT32, "float"), ("float",)),
)
def test_call_nodes_reject_malformed_template_arguments(
    owner: type,
    template_args: object,
) -> None:
    with pytest.raises(TypeError, match=f"{owner.__name__}.template_args"):
        owner(
            "call",
            (),
            cast(Tuple[llir.DataType, ...], template_args),
        )


def test_function_call_stmt_is_frozen_typed_owned_and_structurally_equal() -> None:
    argument = _var("values")
    caller_args = [argument]
    call = llir.FunctionCallStmt("wksp.insert", caller_args)
    tuple_call = llir.FunctionCallStmt("wksp.insert", (_var("values"),))

    caller_args.append(_var("later"))

    assert call.name == "wksp.insert"
    assert type(call.template_args) is tuple
    assert call.template_args == ()
    assert type(call.args) is tuple
    assert call.args == (argument,)
    assert call == tuple_call
    assert call != llir.FunctionCallStmt("wksp.insert", (_var("other"),))
    assert call != llir.FunctionCallStmt("other", (_var("values"),))
    assert call != llir.FunctionCallStmt(
        "wksp.insert",
        (_var("values"),),
        template_args=(llir.DataType.INT64,),
    )
    assert get_type_hints(llir.FunctionCallStmt) == {
        "name": str,
        "template_args": Tuple[llir.DataType, ...],
        "args": Tuple[llir.Expr, ...],
        # The statement-level result-storage marker.  It is one of the node's
        # declared fields and so belongs in this lock; it is ``compare=False``,
        # which is why the equality assertions above are unaffected by it.
        "result_storage": Optional[llir.ResultStorageMetadata],
    }

    with pytest.raises(FrozenInstanceError):
        call.name = "other"
    with pytest.raises(FrozenInstanceError):
        call.template_args = ()
    with pytest.raises(FrozenInstanceError):
        call.args = ()


@pytest.mark.parametrize("name", ("", "   ", 1, None))
def test_function_call_stmt_rejects_malformed_names(name: object) -> None:
    with pytest.raises(TypeError, match="FunctionCallStmt.name"):
        llir.FunctionCallStmt(cast(str, name))


@pytest.mark.parametrize("arguments", ("argument", {"argument"}, {_var("value")}))
def test_function_call_stmt_rejects_malformed_argument_containers(
    arguments: object,
) -> None:
    with pytest.raises(
        TypeError, match="FunctionCallStmt.args must be a list or tuple"
    ):
        llir.FunctionCallStmt("call", cast(list[llir.Expr], arguments))


def test_function_call_stmt_rejects_non_expression_arguments() -> None:
    with pytest.raises(TypeError, match="contain only LLIR expressions"):
        llir.FunctionCallStmt("call", [cast(llir.Expr, "argument")])


def test_member_call_stmt_is_frozen_typed_owned_and_structurally_equal() -> None:
    base = llir.Var("wksp", llir.DataType.NO_TYPE)
    argument = llir.Var("offset", llir.DataType.INT64)
    caller_template_args = [llir.DataType.FLOAT32]
    caller_args = [argument]
    call = llir.MemberCallStmt(
        base=base,
        member="reserve",
        template_args=caller_template_args,
        args=caller_args,
    )
    equal = llir.MemberCallStmt(
        base=llir.Var("wksp", llir.DataType.NO_TYPE),
        member="reserve",
        template_args=(llir.DataType.FLOAT32,),
        args=(llir.Var("offset", llir.DataType.INT64),),
    )

    caller_template_args.append(llir.DataType.INT)
    caller_args.append(_var("later"))

    assert call.base is base
    assert call.member == "reserve"
    assert type(call.template_args) is tuple
    assert call.template_args == (llir.DataType.FLOAT32,)
    assert type(call.args) is tuple
    assert call.args == (argument,)
    assert call == equal
    assert hash(call) == hash(equal)
    assert call != llir.MemberCallStmt(
        llir.Var("other", llir.DataType.NO_TYPE),
        "reserve",
        (llir.DataType.FLOAT32,),
        (llir.Var("offset", llir.DataType.INT64),),
    )
    assert call != llir.MemberCallStmt(
        llir.Var("wksp", llir.DataType.NO_TYPE),
        "reserve",
        (llir.DataType.FLOAT64,),
        (llir.Var("offset", llir.DataType.INT64),),
    )
    assert get_type_hints(llir.MemberCallStmt) == {
        "base": llir.Expr,
        "member": str,
        "template_args": Tuple[llir.DataType, ...],
        "args": Tuple[llir.Expr, ...],
        "result_storage": Optional[llir.ResultStorageMetadata],
    }

    with pytest.raises(FrozenInstanceError):
        call.base = _var("other")
    with pytest.raises(FrozenInstanceError):
        call.member = "clear"
    with pytest.raises(FrozenInstanceError):
        call.template_args = ()
    with pytest.raises(FrozenInstanceError):
        call.args = ()


@pytest.mark.parametrize("member", ("", "first.second", 1, None))
def test_member_call_stmt_rejects_malformed_base_and_member(member: object) -> None:
    with pytest.raises(TypeError, match="MemberCallStmt.member"):
        llir.MemberCallStmt(_var("workspace"), cast(str, member))

    with pytest.raises(TypeError, match="MemberCallStmt.base"):
        llir.MemberCallStmt(cast(llir.Expr, "workspace"), "clear")


@pytest.mark.parametrize(
    "template_args",
    ("float", {llir.DataType.FLOAT32}, llir.DataType.FLOAT32),
)
def test_member_call_stmt_rejects_malformed_template_argument_containers(
    template_args: object,
) -> None:
    with pytest.raises(TypeError, match="template_args must be a list or tuple"):
        llir.MemberCallStmt(
            _var("workspace"),
            "clear",
            template_args=cast(Any, template_args),
        )


def test_member_call_stmt_rejects_malformed_template_and_call_arguments() -> None:
    with pytest.raises(TypeError, match="contain only DataType values"):
        llir.MemberCallStmt(
            _var("workspace"),
            "clear",
            template_args=[cast(llir.DataType, "float")],
        )
    with pytest.raises(TypeError, match="args must be a list or tuple"):
        llir.MemberCallStmt(_var("workspace"), "reserve", args=cast(Any, "extent"))
    with pytest.raises(TypeError, match="contain only LLIR expressions"):
        llir.MemberCallStmt(
            _var("workspace"),
            "reserve",
            args=[cast(llir.Expr, "extent")],
        )


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


def test_fixed_stack_array_decl_is_frozen_typed_owned_and_structural() -> None:
    extent = llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT)
    caller_values: list[llir.Expr] = []
    initializer = llir.Array(caller_values, llir.DataType.FLOAT32)
    declaration = llir.FixedStackArrayDecl(
        name="wksp",
        element_type=llir.DataType.FLOAT32,
        extent=extent,
        initializer=initializer,
    )
    equal = llir.FixedStackArrayDecl(
        name="wksp",
        element_type=llir.DataType.FLOAT32,
        extent=llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
        initializer=llir.Array((), llir.DataType.FLOAT32),
    )
    caller_values.append(llir.Literal(0.0))

    assert declaration.name == "wksp"
    assert declaration.element_type is llir.DataType.FLOAT32
    assert declaration.extent is extent
    assert declaration.initializer is initializer
    assert declaration.initializer.values == ()
    assert type(declaration.extent) is llir.Var
    assert type(declaration.initializer) is llir.Array
    assert declaration == equal
    assert hash(declaration) == hash(equal)
    assert declaration != llir.FixedStackArrayDecl(
        "other",
        llir.DataType.FLOAT32,
        llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
        llir.Array((), llir.DataType.FLOAT32),
    )
    assert declaration != llir.FixedStackArrayDecl(
        "wksp",
        llir.DataType.FLOAT64,
        llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
        llir.Array((), llir.DataType.FLOAT64),
    )
    assert get_type_hints(llir.FixedStackArrayDecl) == {
        "name": str,
        "element_type": llir.DataType,
        "extent": llir.Expr,
        "initializer": llir.Array,
    }

    with pytest.raises(FrozenInstanceError):
        declaration.name = "other"
    with pytest.raises(FrozenInstanceError):
        declaration.element_type = llir.DataType.FLOAT64
    with pytest.raises(FrozenInstanceError):
        declaration.extent = llir.Literal(8)
    with pytest.raises(FrozenInstanceError):
        declaration.initializer = llir.Array((), llir.DataType.FLOAT32)


@pytest.mark.parametrize(
    ("element_type", "expected"),
    (
        (llir.DataType.FLOAT32, "float wksp[kTile_k] = {};"),
        (llir.DataType.FLOAT64, "double wksp[kTile_k] = {};"),
        (llir.DataType.INT32, "int32_t wksp[kTile_k] = {};"),
        (llir.DataType.INT64, "int64_t wksp[kTile_k] = {};"),
        (llir.DataType.INT8, "int8_t wksp[kTile_k] = {};"),
        (llir.DataType.UINT8, "uint8_t wksp[kTile_k] = {};"),
    ),
)
def test_fixed_stack_array_decl_codegen_is_byte_exact_for_supported_scalars(
    element_type: llir.DataType,
    expected: str,
) -> None:
    declaration = llir.FixedStackArrayDecl(
        "wksp",
        element_type,
        llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
        llir.Array((), element_type),
    )

    assert LLIRLowerer().lower_llir(declaration) == expected
    assert LLIRLowerer().lower_llir(declaration, indent_level=2) == f"    {expected}"


@pytest.mark.parametrize(
    "extent_type",
    (
        llir.DataType.INT,
        llir.DataType.INT32,
        llir.DataType.INT64,
        llir.DataType.UINT32,
        llir.DataType.UINT64,
    ),
)
def test_fixed_stack_array_literal_extent_codegen_is_structured_and_byte_exact(
    extent_type: llir.DataType,
) -> None:
    declaration = llir.FixedStackArrayDecl(
        "wksp",
        llir.DataType.FLOAT32,
        llir.Literal(8, extent_type),
        llir.Array((), llir.DataType.FLOAT32),
    )

    assert type(declaration.extent) is llir.Literal
    assert LLIRLowerer().lower_llir(declaration) == "float wksp[8] = {};"


@pytest.mark.parametrize("name", ("", "workspace[8]", "two words", 1, None))
def test_fixed_stack_array_decl_rejects_malformed_names(name: object) -> None:
    with pytest.raises(TypeError, match="FixedStackArrayDecl.name"):
        llir.FixedStackArrayDecl(
            cast(str, name),
            llir.DataType.FLOAT32,
            llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
            llir.Array((), llir.DataType.FLOAT32),
        )


@pytest.mark.parametrize(
    "element_type",
    (llir.DataType.AUTO, llir.DataType.UINT32, llir.DataType.PTR_FLOAT32, "float"),
)
def test_fixed_stack_array_decl_rejects_unsupported_element_types(
    element_type: object,
) -> None:
    with pytest.raises(TypeError, match="supported scalar DataType"):
        llir.FixedStackArrayDecl(
            "wksp",
            cast(llir.DataType, element_type),
            llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
            llir.Array((), llir.DataType.FLOAT32),
        )


@pytest.mark.parametrize(
    "extent",
    (
        llir.Var("runtime", llir.DataType.INT),
        llir.Var("kTile + 1", llir.DataType.CONSTEXPR_INT),
        llir.Var("pointer", llir.DataType.CONSTEXPR_INT, is_ptr=True),
        llir.Var("restricted", llir.DataType.CONSTEXPR_INT, is_restrict=True),
        llir.Var(
            "annotated",
            llir.DataType.CONSTEXPR_INT,
            tensor_access=_result_metadata(),
        ),
        llir.Literal(0),
        llir.Literal(-1),
        llir.Literal(True),
        llir.Literal(1.5),
        llir.Literal(8, llir.DataType.UINT16),
        llir.Add(llir.Literal(4), llir.Literal(4)),
    ),
)
def test_fixed_stack_array_decl_rejects_nonfixed_extents(extent: llir.Expr) -> None:
    with pytest.raises(TypeError, match="exact metadata-free constexpr Var"):
        llir.FixedStackArrayDecl(
            "wksp",
            llir.DataType.FLOAT32,
            extent,
            llir.Array((), llir.DataType.FLOAT32),
        )


def test_fixed_stack_array_decl_rejects_nonexact_extent_subclasses() -> None:
    class UnknownVar(llir.Var):
        pass

    class UnknownLiteral(llir.Literal):
        pass

    for extent in (
        UnknownVar("kTile_k", llir.DataType.CONSTEXPR_INT),
        UnknownLiteral(8),
    ):
        with pytest.raises(TypeError, match="exact metadata-free constexpr Var"):
            llir.FixedStackArrayDecl(
                "wksp",
                llir.DataType.FLOAT32,
                extent,
                llir.Array((), llir.DataType.FLOAT32),
            )


def test_fixed_stack_array_decl_requires_exact_empty_matching_initializer() -> None:
    class UnknownArray(llir.Array):
        pass

    extent = llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT)
    with pytest.raises(TypeError, match="initializer must be an exact Array"):
        llir.FixedStackArrayDecl(
            "wksp",
            llir.DataType.FLOAT32,
            extent,
            cast(llir.Array, "{}"),
        )
    with pytest.raises(TypeError, match="initializer must be an exact Array"):
        llir.FixedStackArrayDecl(
            "wksp",
            llir.DataType.FLOAT32,
            extent,
            UnknownArray((), llir.DataType.FLOAT32),
        )
    with pytest.raises(TypeError, match="initializer must be empty"):
        llir.FixedStackArrayDecl(
            "wksp",
            llir.DataType.FLOAT32,
            extent,
            llir.Array((llir.Literal(0.0),), llir.DataType.FLOAT32),
        )
    with pytest.raises(TypeError, match="type must match element_type"):
        llir.FixedStackArrayDecl(
            "wksp",
            llir.DataType.FLOAT32,
            extent,
            llir.Array((), llir.DataType.FLOAT64),
        )


@pytest.mark.parametrize(
    ("malformation", "expected"),
    (
        ("name", "FixedStackArrayDecl.name"),
        ("element_type", "supported scalar DataType"),
        ("extent", "FixedStackArrayDecl.extent"),
        ("initializer", "initializer must be an exact Array"),
        ("initializer_values_container", "initializer values must be a tuple"),
        ("initializer_values", "initializer must be empty"),
        ("initializer_type", "initializer type must match element_type"),
    ),
)
def test_codegen_rejects_forged_fixed_stack_array_fields(
    malformation: str,
    expected: str,
) -> None:
    declaration = llir.FixedStackArrayDecl(
        "wksp",
        llir.DataType.FLOAT32,
        llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
        llir.Array((), llir.DataType.FLOAT32),
    )
    if malformation == "name":
        object.__setattr__(declaration, "name", "wksp[8]")
    elif malformation == "element_type":
        object.__setattr__(declaration, "element_type", llir.DataType.AUTO)
    elif malformation == "extent":
        object.__setattr__(declaration, "extent", _var("runtime"))
    elif malformation == "initializer":
        object.__setattr__(declaration, "initializer", "{}")
    elif malformation == "initializer_values_container":
        initializer = object.__new__(llir.Array)
        object.__setattr__(initializer, "values", [])
        object.__setattr__(initializer, "data_type", llir.DataType.FLOAT32)
        object.__setattr__(declaration, "initializer", initializer)
    elif malformation == "initializer_values":
        object.__setattr__(
            declaration,
            "initializer",
            llir.Array((llir.Literal(0.0),), llir.DataType.FLOAT32),
        )
    else:
        object.__setattr__(
            declaration,
            "initializer",
            llir.Array((), llir.DataType.FLOAT64),
        )

    with pytest.raises(CodegenError, match=expected):
        LLIRLowerer().lower_llir(declaration)


def test_codegen_rejects_fixed_stack_array_subclasses_and_unknown_children() -> None:
    class UnknownFixedStackArrayDecl(llir.FixedStackArrayDecl):
        pass

    class UnknownExpr(llir.Expr):
        pass

    extent = llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT)
    initializer = llir.Array((), llir.DataType.FLOAT32)
    with pytest.raises(CodegenError, match="UnknownFixedStackArrayDecl"):
        LLIRLowerer().lower_llir(
            UnknownFixedStackArrayDecl(
                "wksp",
                llir.DataType.FLOAT32,
                extent,
                initializer,
            )
        )

    forged = llir.FixedStackArrayDecl(
        "wksp",
        llir.DataType.FLOAT32,
        extent,
        initializer,
    )
    object.__setattr__(forged, "extent", UnknownExpr())
    with pytest.raises(CodegenError, match="UnknownExpr at root.extent"):
        LLIRLowerer().lower_llir(forged)


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
    ("statement", "expected"),
    (
        (
            llir.MemberCallStmt(
                llir.Var("wksp", llir.DataType.NO_TYPE),
                "clear",
            ),
            "wksp.clear();",
        ),
        (
            llir.MemberCallStmt(
                llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
                "reserve",
                args=(_var("extent"),),
            ),
            "storage.reserve(extent);",
        ),
        (
            llir.MemberCallStmt(
                llir.ArrayAccess(_var("pool"), _var("i")),
                "emplace_back",
                args=(_var("rows"), llir.Literal(True, llir.DataType.BOOL)),
            ),
            "pool[i].emplace_back(rows, true);",
        ),
        (
            llir.MemberCallStmt(
                llir.MemberAccess(_var("owner"), "storage"),
                "resize",
                template_args=(llir.DataType.INT, llir.DataType.INT64),
                args=(llir.Add(_var("i"), llir.Literal(1)),),
            ),
            "owner.storage.resize<int, int64_t>(i + 1);",
        ),
        (
            llir.MemberCallStmt(
                llir.BinOp("+", _var("owner"), _var("offset")),
                "clear",
            ),
            "(owner + offset).clear();",
        ),
    ),
)
def test_member_call_stmt_codegen_is_byte_exact_typed_and_precedence_correct(
    statement: llir.MemberCallStmt,
    expected: str,
) -> None:
    assert LLIRLowerer().lower_llir(statement) == expected
    assert LLIRLowerer().lower_llir(statement, 3) == "      " + expected


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
        "result_storage": Optional[llir.ResultStorageMetadata],
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
            "assignment index Var name",
        ),
        (
            llir.ArrayAccess(_var("values"), llir.Literal("i + 1")),
            "Literal value must be an int",
        ),
        (
            llir.ArrayAccess(_var("values"), llir.FunctionCall("i + 1")),
            "FunctionCall name must be an identifier or member path",
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
                _forged_sizeof("int"),
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


def test_assign_rejects_forged_sizeof_missing_data_type() -> None:
    with pytest.raises(TypeError, match="Sizeof.data_type must be a DataType"):
        llir.Assign(
            var=llir.ArrayAccess(
                _var("values"),
                object.__new__(llir.Sizeof),
            ),
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
    assert LLIRLowerer().lower_llir(scalar) == "value = (int)(1 + 2);"

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


def test_indexed_assign_rejects_function_call_missing_template_args() -> None:
    call = object.__new__(llir.FunctionCall)
    object.__setattr__(call, "name", "indices.size")
    object.__setattr__(call, "args", ())

    with pytest.raises(TypeError, match="FunctionCall.template_args"):
        llir.Assign(
            llir.ArrayAccess(_var("values"), call),
            llir.Literal(1),
        )


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
        expected = "UnknownArrayAccess at root.var"
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


@pytest.mark.parametrize(
    "malformation",
    ("name", "missing_template_args", "template_arg", "args", "argument"),
)
def test_codegen_rejects_forged_function_call_fields(malformation: str) -> None:
    call = object.__new__(llir.FunctionCall)
    object.__setattr__(call, "name", "call")
    if malformation != "missing_template_args":
        object.__setattr__(call, "template_args", ())
    object.__setattr__(call, "args", (_var("argument"),))
    if malformation == "name":
        object.__setattr__(call, "name", " ")
        expected = "FunctionCall.name"
    elif malformation == "missing_template_args":
        expected = "FunctionCall.template_args must be a tuple of DataType values"
    elif malformation == "template_arg":
        object.__setattr__(call, "template_args", ("float",))
        expected = "FunctionCall.template_args must be a tuple of DataType values"
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


def test_codegen_rejects_member_call_stmt_subclasses_and_unknown_children() -> None:
    class UnknownMemberCallStmt(llir.MemberCallStmt):
        pass

    class UnknownExpr(llir.Expr):
        pass

    with pytest.raises(CodegenError, match="UnknownMemberCallStmt"):
        LLIRLowerer().lower_llir(UnknownMemberCallStmt(_var("workspace"), "clear"))
    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(llir.MemberCallStmt(UnknownExpr(), "clear"))
    with pytest.raises(CodegenError, match="UnknownExpr"):
        LLIRLowerer().lower_llir(
            llir.MemberCallStmt(_var("workspace"), "reserve", args=(UnknownExpr(),))
        )


@pytest.mark.parametrize(
    "malformation",
    ("base", "member", "template_args", "template_arg", "args", "argument"),
)
def test_codegen_rejects_forged_member_call_stmt_fields(malformation: str) -> None:
    call = object.__new__(llir.MemberCallStmt)
    object.__setattr__(call, "base", _var("workspace"))
    object.__setattr__(call, "member", "clear")
    object.__setattr__(call, "template_args", (llir.DataType.FLOAT32,))
    object.__setattr__(call, "args", (_var("argument"),))
    if malformation == "base":
        object.__setattr__(call, "base", "workspace")
        expected = "MemberCallStmt.base"
    elif malformation == "member":
        object.__setattr__(call, "member", "workspace.clear")
        expected = "MemberCallStmt.member"
    elif malformation == "template_args":
        object.__setattr__(call, "template_args", [llir.DataType.FLOAT32])
        expected = "MemberCallStmt.template_args"
    elif malformation == "template_arg":
        object.__setattr__(call, "template_args", ("float",))
        expected = "MemberCallStmt.template_args"
    elif malformation == "args":
        object.__setattr__(call, "args", [_var("argument")])
        expected = "MemberCallStmt.args"
    else:
        object.__setattr__(call, "args", ("argument",))
        expected = "MemberCallStmt.args"

    with pytest.raises(CodegenError, match=expected):
        LLIRLowerer().lower_llir(call)


@pytest.mark.parametrize(
    ("missing_field", "expected"),
    (
        ("base", "MemberCallStmt.base"),
        ("member", "MemberCallStmt.member"),
        ("template_args", "MemberCallStmt.template_args"),
        ("args", "MemberCallStmt.args"),
    ),
)
def test_codegen_rejects_forged_member_call_stmt_missing_fields(
    missing_field: str,
    expected: str,
) -> None:
    call = object.__new__(llir.MemberCallStmt)
    fields = {
        "base": _var("workspace"),
        "member": "clear",
        "template_args": (),
        "args": (),
    }
    for field, value in fields.items():
        if field != missing_field:
            object.__setattr__(call, field, value)

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
