from typing import Dict, List, Sequence, Set, Tuple, Type, cast

import pytest

from scorch.compiler import llir
from scorch.compiler.codegen import (  # type: ignore[import-untyped]
    EMITTED_LLIR_NODE_TYPES,
    LLIRLowerer,
)
from scorch.compiler.diagnostics import CodegenError  # type: ignore[import-untyped]
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]
from scorch.compiler.llir_traversal import (
    LLIRPath,
    LLIRRewriter,
    LLIRStatementValue,
    LLIRTraversalContext,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
    SUPPORTED_LLIR_NODE_TYPES,
)

_CONTEXT = LLIRTraversalContext(stage="LLIR test", pass_name="identity")


def _var(name: str, data_type: llir.DataType = llir.DataType.INT) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _result_metadata(access_id: int = 1) -> llir.TensorAccessMetadata:
    return llir.TensorAccessMetadata(
        access_id=AccessId(access_id),
        tensor_id=SymbolId(2),
        index_ids=(IndexId(3),),
        role=llir.TensorAccessRole.RESULT_WRITE,
    )


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


class _RecordingWalker(LLIRWalker):
    def __init__(self) -> None:
        super().__init__(_CONTEXT)
        self.events: List[str] = []

    def enter_node(self, node: llir.Node, path: LLIRPath) -> None:
        label = type(node).__name__
        if type(node) is llir.Var:
            label += f":{cast(llir.Var, node).name}"
        elif type(node) is llir.Literal:
            label += f":{cast(llir.Literal, node).value}"
        elif type(node) is llir.QualifiedName:
            qualified = cast(llir.QualifiedName, node)
            label += f":{qualified.namespace}::{qualified.name}"
        self.events.append(label)


def _record(value: object) -> List[str]:
    walker = _RecordingWalker()
    walker.walk(cast(LLIRValue, value))
    return walker.events


def _structural_snapshot(value: object) -> object:
    if isinstance(value, llir.Node):
        return (
            type(value).__name__,
            tuple(
                (name, _structural_snapshot(child))
                for name, child in sorted(vars(value).items())
            ),
        )
    if isinstance(value, list):
        return ("list", tuple(_structural_snapshot(child) for child in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_structural_snapshot(child) for child in value))
    return value


def _mutable_ir_ids(value: object) -> Set[int]:
    mutable_ids: Set[int] = set()
    if isinstance(value, llir.Node):
        mutable_ids.add(id(value))
        for child in vars(value).values():
            mutable_ids.update(_mutable_ir_ids(child))
    elif isinstance(value, list):
        mutable_ids.add(id(value))
        for child in value:
            mutable_ids.update(_mutable_ir_ids(child))
    elif isinstance(value, tuple):
        for child in value:
            mutable_ids.update(_mutable_ir_ids(child))
    return mutable_ids


def _declared_node_types() -> Set[Type[llir.Node]]:
    declared: Set[Type[llir.Node]] = set()
    pending: List[Type[llir.Node]] = [llir.Node]
    while pending:
        parent = pending.pop()
        for child in parent.__subclasses__():
            pending.append(child)
            if child.__module__ == llir.__name__ and child not in (
                llir.Expr,
                llir.Stmt,
            ):
                declared.add(child)
    return declared


def _node_samples() -> Dict[Type[llir.Node], llir.Node]:
    value = _var("value")
    index = _var("index")
    literal = llir.Literal(1)
    return {
        llir.Var: value,
        llir.UnaryOp: llir.UnaryOp("-", value),
        llir.BinOp: llir.BinOp("+", value, literal),
        llir.Add: llir.Add(value, literal),
        llir.Mul: llir.Mul(value, literal),
        llir.Literal: literal,
        llir.QualifiedName: llir.QualifiedName(
            "torch", "kInt", llir.DataType.TORCH_SCALAR_TYPE
        ),
        llir.Increment: llir.Increment(index),
        llir.Return: llir.Return(value),
        llir.VarDecl: llir.VarDecl(value),
        llir.VarInit: llir.VarInit(value, literal),
        llir.DirectInit: llir.DirectInit(
            llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
            (literal,),
        ),
        llir.FixedStackArrayDecl: llir.FixedStackArrayDecl(
            "workspace",
            llir.DataType.FLOAT32,
            llir.Var("kTile", llir.DataType.CONSTEXPR_INT),
            llir.Array((), llir.DataType.FLOAT32),
        ),
        llir.Assign: llir.Assign(value, literal),
        llir.Comment: llir.Comment("comment"),
        llir.BlankLine: llir.BlankLine(),
        llir.RawStmt: llir.RawStmt("raw"),
        llir.Continue: llir.Continue(),
        llir.Break: llir.Break(),
        llir.Function: llir.Function(
            return_type=llir.DataType.INT,
            name="function",
            args=[value],
            body=[llir.Return(value)],
        ),
        llir.FunctionCall: llir.FunctionCall("call", [value]),
        llir.FunctionCallStmt: llir.FunctionCallStmt("call", [value]),
        llir.MemberCallStmt: llir.MemberCallStmt(
            value,
            "member",
            (llir.DataType.INT,),
            (index,),
        ),
        llir.Array: llir.Array([value, literal], llir.DataType.INT),
        llir.MemberAccess: llir.MemberAccess(value, "member"),
        llir.MemberCall: llir.MemberCall(
            value,
            "member",
            (llir.DataType.INT,),
            (index,),
        ),
        llir.ArrayAccess: llir.ArrayAccess(value, index),
        llir.ForLoop: llir.ForLoop(
            init=None,
            cond=value,
            update=llir.Increment(index),
            body=[llir.Break()],
        ),
        llir.ForLoopAuto: llir.ForLoopAuto(
            var=index,
            array=value,
            body=[llir.Break()],
        ),
        llir.WhileLoop: llir.WhileLoop(value, [llir.Break()]),
        llir.IfThenElse: llir.IfThenElse(
            cond=value,
            then_body=[llir.Break()],
            else_body=[llir.Continue()],
        ),
        llir.Cast: llir.Cast(value, llir.DataType.INT64),
        llir.Sizeof: llir.Sizeof(llir.DataType.INT64),
        llir.AddressOf: llir.AddressOf(llir.ArrayAccess(value, index)),
    }


def _node_emissions() -> Dict[Type[llir.Node], str]:
    return {
        llir.Var: "value",
        llir.UnaryOp: "- value",
        llir.BinOp: "value + 1",
        llir.Add: "value + 1",
        llir.Mul: "value * 1",
        llir.Literal: "1",
        llir.QualifiedName: "torch::kInt",
        llir.Increment: "index++;",
        llir.Return: "return value;",
        llir.VarDecl: "int value;",
        llir.VarInit: "int value = 1;",
        llir.DirectInit: "std::vector<float> storage(1);",
        llir.FixedStackArrayDecl: "float workspace[kTile] = {};",
        llir.Assign: "value = 1;",
        llir.Comment: "// comment",
        llir.BlankLine: " ",
        llir.RawStmt: "raw;",
        llir.Continue: "continue;",
        llir.Break: "break;",
        llir.Function: "int function(int value) {\n  return value;\n}",
        llir.FunctionCall: "call(value)",
        llir.FunctionCallStmt: "call(value);",
        llir.MemberCallStmt: "value.member<int>(index);",
        llir.Array: "{value, 1}",
        llir.MemberAccess: "value.member",
        llir.MemberCall: "value.member<int>(index)",
        llir.ArrayAccess: "value[index]",
        llir.ForLoop: "for (; value; index++) {\n  break;\n}",
        llir.ForLoopAuto: "for (int index : value) {\n  break;\n}",
        llir.WhileLoop: "while (value) {\n  break;\n}",
        llir.IfThenElse: "if (value) {\n  break;\n} else {\n  continue;\n}",
        llir.Cast: "(int64_t)value",
        llir.Sizeof: "sizeof(int64_t)",
        llir.AddressOf: "&value[index]",
    }


def test_walker_has_deterministic_preorder() -> None:
    index = _var("i")
    output = llir.ArrayAccess(
        _var("out"),
        llir.Add(_var("p"), llir.Literal(1)),
        tensor_access=_result_metadata(),
    )
    function = llir.Function(
        return_type=llir.DataType.INT,
        name="ordered",
        args=[_var("arg0"), _var("arg1")],
        body=[
            llir.ForLoop(
                init=llir.VarInit(index, llir.Literal(0)),
                cond=llir.BinOp("<", index, _var("n")),
                update=llir.Increment(index),
                body=[
                    llir.IfThenElse(
                        cond_list=[_var("cond0"), _var("cond1")],
                        then_body_list=[
                            [llir.Assign(output, _var("value"))],
                            [llir.Break()],
                        ],
                        else_body=[llir.Continue()],
                    )
                ],
                before_parallel_body=[llir.Comment("before")],
                pre_parallel_body=[llir.RawStmt("pre")],
                post_parallel_body=[llir.Return(output)],
            )
        ],
    )

    expected = [
        "Function",
        "Var:arg0",
        "Var:arg1",
        "ForLoop",
        "Comment",
        "VarInit",
        "Var:i",
        "Literal:0",
        "BinOp",
        "Var:i",
        "Var:n",
        "Increment",
        "Var:i",
        "RawStmt",
        "IfThenElse",
        "Var:cond0",
        "Var:cond1",
        "Assign",
        "ArrayAccess",
        "Var:out",
        "Add",
        "Var:p",
        "Literal:1",
        "Var:value",
        "Break",
        "Continue",
        "Return",
        "ArrayAccess",
        "Var:out",
        "Add",
        "Var:p",
        "Literal:1",
    ]
    assert _record(function) == expected
    assert _record(function) == expected


def test_arithmetic_walker_has_deterministic_preorder() -> None:
    expression = llir.BinOp(
        "<",
        llir.Add(
            llir.Mul(_var("tile"), llir.Literal(4, llir.DataType.INT64)),
            _var("offset"),
        ),
        llir.Literal(32, llir.DataType.INT64),
    )

    expected = [
        "BinOp",
        "Add",
        "Mul",
        "Var:tile",
        "Literal:4",
        "Var:offset",
        "Literal:32",
    ]
    assert _record(expression) == expected
    assert _record(expression) == expected


def test_member_access_walker_has_deterministic_preorder() -> None:
    expression = llir.ArrayAccess(
        array=llir.MemberAccess(
            base=llir.MemberAccess(_var("it"), "first"),
            member="coordinates",
        ),
        index=llir.Literal(0, llir.DataType.INT64),
    )

    assert _record(expression) == [
        "ArrayAccess",
        "MemberAccess",
        "MemberAccess",
        "Var:it",
        "Literal:0",
    ]


def test_function_call_walker_has_deterministic_preorder() -> None:
    expression = llir.FunctionCall(
        "scorch_tensor_from_vector",
        [
            llir.FunctionCall("std::move", [_var("values")]),
            llir.QualifiedName("torch", "kFloat32", llir.DataType.TORCH_SCALAR_TYPE),
        ],
    )

    assert _record(expression) == [
        "FunctionCall",
        "FunctionCall",
        "Var:values",
        "QualifiedName:torch::kFloat32",
    ]


def test_member_call_walker_visits_base_then_arguments_in_deterministic_order() -> None:
    expression = llir.MemberCall(
        base=llir.MemberAccess(_var("tensor"), "storage"),
        member="select",
        template_args=(llir.DataType.FLOAT32,),
        args=(
            llir.Add(_var("index"), llir.Literal(1)),
            llir.QualifiedName(
                "torch",
                "kFloat32",
                llir.DataType.TORCH_SCALAR_TYPE,
            ),
        ),
    )

    expected = [
        "MemberCall",
        "MemberAccess",
        "Var:tensor",
        "Add",
        "Var:index",
        "Literal:1",
        "QualifiedName:torch::kFloat32",
    ]
    assert _record(expression) == expected
    assert _record(expression) == expected


def test_increment_walker_has_deterministic_preorder() -> None:
    increment = llir.Increment(_var("counter"))

    assert _record(increment) == ["Increment", "Var:counter"]
    assert _record(increment) == ["Increment", "Var:counter"]


def test_array_walker_has_deterministic_nested_preorder() -> None:
    expression = llir.FunctionCall(
        "consume",
        (
            llir.Array(
                (
                    _var("extent", llir.DataType.INT64),
                    llir.Array(
                        (llir.Add(_var("offset"), llir.Literal(1)),),
                        llir.DataType.INT64,
                    ),
                ),
                llir.DataType.INT64,
            ),
        ),
    )

    expected = [
        "FunctionCall",
        "Array",
        "Var:extent",
        "Array",
        "Add",
        "Var:offset",
        "Literal:1",
    ]
    assert _record(expression) == expected
    assert _record(expression) == expected


def test_fixed_stack_array_walker_has_deterministic_preorder() -> None:
    declaration = llir.FixedStackArrayDecl(
        "wksp",
        llir.DataType.FLOAT32,
        llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
        llir.Array((), llir.DataType.FLOAT32),
    )

    expected = ["FixedStackArrayDecl", "Var:kTile_k", "Array"]
    assert _record(declaration) == expected
    assert _record(declaration) == expected


def test_fixed_stack_array_rewriter_visits_extent_before_initializer() -> None:
    events: List[str] = []

    class RecordingRewriter(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            events.append("extent")
            return super().rewrite_var(node, path)

        def rewrite_array(self, node: llir.Array, path: LLIRPath) -> llir.Array:
            events.append("initializer")
            return super().rewrite_array(node, path)

    declaration = llir.FixedStackArrayDecl(
        "wksp",
        llir.DataType.FLOAT32,
        llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
        llir.Array((), llir.DataType.FLOAT32),
    )

    RecordingRewriter(_CONTEXT).rewrite(declaration)
    assert events == ["extent", "initializer"]


def test_panel_bound_cast_walker_has_deterministic_preorder() -> None:
    coordinate = _var("coordinates", llir.DataType.PTR_INT)
    expression = llir.Cast(
        llir.BinOp(
            "-",
            llir.FunctionCall(
                "std::lower_bound",
                [
                    llir.Add(
                        coordinate,
                        llir.ArrayAccess(
                            _var("positions", llir.DataType.PTR_INT),
                            _var("parent"),
                        ),
                    ),
                    llir.Add(
                        _var("coordinates", llir.DataType.PTR_INT),
                        _var("row_end"),
                    ),
                    _var("panel", llir.DataType.INT64),
                ],
            ),
            _var("coordinates", llir.DataType.PTR_INT),
        ),
        llir.DataType.INT,
    )

    expected = [
        "Cast",
        "BinOp",
        "FunctionCall",
        "Add",
        "Var:coordinates",
        "ArrayAccess",
        "Var:positions",
        "Var:parent",
        "Add",
        "Var:coordinates",
        "Var:row_end",
        "Var:panel",
        "Var:coordinates",
    ]
    assert _record(expression) == expected
    assert _record(expression) == expected


def test_direct_init_walker_and_rewriter_have_deterministic_preorder() -> None:
    declaration = llir.DirectInit(
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
    expected = [
        "DirectInit",
        "Var:packed_B_storage",
        "Mul",
        "Cast",
        "Var:kTile_j",
        "Cast",
        "Var:kTile_k",
    ]

    assert _record(declaration) == expected

    rewrite_events: List[str] = []

    class RecordingRewriter(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewrite_events.append(node.name)
            return super().rewrite_var(node, path)

    rewritten = RecordingRewriter(_CONTEXT).rewrite(declaration)
    assert type(rewritten) is llir.DirectInit
    assert rewrite_events == ["packed_B_storage", "kTile_j", "kTile_k"]
    assert _record(rewritten) == expected


def test_walker_and_rewriter_cover_every_declared_node() -> None:
    samples = _node_samples()
    emissions = _node_emissions()
    assert set(SUPPORTED_LLIR_NODE_TYPES) == _declared_node_types()
    assert set(samples) == set(SUPPORTED_LLIR_NODE_TYPES)
    assert set(emissions) == set(SUPPORTED_LLIR_NODE_TYPES)
    assert EMITTED_LLIR_NODE_TYPES == SUPPORTED_LLIR_NODE_TYPES

    walker = LLIRWalker(_CONTEXT)
    rewriter = LLIRRewriter(_CONTEXT)
    lowerer = LLIRLowerer()
    for node_type in SUPPORTED_LLIR_NODE_TYPES:
        sample = samples[node_type]
        assert type(sample) is node_type
        walker.walk(sample)
        assert lowerer.lower_llir(sample) == emissions[node_type]
        rewritten = rewriter.rewrite(sample)
        assert type(rewritten) is node_type
        assert _structural_snapshot(rewritten) == _structural_snapshot(sample)
        assert _mutable_ir_ids(rewritten).isdisjoint(_mutable_ir_ids(sample))


def test_every_declared_node_subclass_fails_closed_in_traversal_and_codegen() -> None:
    walker = LLIRWalker(_CONTEXT)
    lowerer = LLIRLowerer()
    for node_type, sample in _node_samples().items():
        unknown_type = type(f"Unknown{node_type.__name__}", (node_type,), {})
        unknown: llir.Node = object.__new__(unknown_type)
        vars(unknown).update(vars(sample))

        with pytest.raises(LLIRTraversalError) as traversal_error:
            walker.walk(unknown)
        assert traversal_error.value.diagnostic.node_type == unknown_type.__name__
        with pytest.raises(CodegenError, match=unknown_type.__name__):
            lowerer.lower_llir(unknown)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_statement_fails_with_structured_stage_diagnostic(
    operation: str,
) -> None:
    class UnknownStmt(llir.Stmt):
        pass

    unknown = UnknownStmt()
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.stage == "LLIR test"
    assert diagnostic.pass_name == "identity"
    assert diagnostic.node_type == "UnknownStmt"
    assert diagnostic.path == ("root",)
    assert "stage=LLIR test pass=identity" in str(raised.value)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_subclass_of_supported_expression_fails_closed(
    operation: str,
) -> None:
    class UnknownBinOp(llir.BinOp):
        pass

    unknown = UnknownBinOp("+", _var("left"), _var("right"))
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownBinOp"
    assert raised.value.diagnostic.code == "unknown_llir_node"


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_arithmetic_subclasses_fail_closed(operation: str) -> None:
    class UnknownAdd(llir.Add):
        pass

    class UnknownLiteral(llir.Literal):
        pass

    for unknown in (
        UnknownAdd(_var("left"), _var("right")),
        UnknownLiteral(1),
    ):
        with pytest.raises(LLIRTraversalError) as raised:
            if operation == "walk":
                LLIRWalker(_CONTEXT).walk(unknown)
            else:
                LLIRRewriter(_CONTEXT).rewrite(unknown)

        assert raised.value.diagnostic.node_type == type(unknown).__name__
        assert raised.value.diagnostic.code == "unknown_llir_node"
        assert raised.value.diagnostic.path == ("root",)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_qualified_name_subclass_fails_closed(operation: str) -> None:
    class UnknownQualifiedName(llir.QualifiedName):
        pass

    unknown = UnknownQualifiedName("torch", "kInt", llir.DataType.TORCH_SCALAR_TYPE)
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownQualifiedName"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root",)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("node_type", "field", "invalid", "diagnostic_code", "expected_path"),
    (
        (
            llir.BinOp,
            "op",
            "",
            "invalid_binary_operator",
            ("root", "op"),
        ),
        (
            llir.Add,
            "op",
            "-",
            "invalid_add_operator",
            ("root", "op"),
        ),
        (
            llir.Mul,
            "op",
            "+",
            "invalid_mul_operator",
            ("root", "op"),
        ),
        (
            llir.BinOp,
            "left",
            "left",
            "invalid_binary_child",
            ("root", "left"),
        ),
        (
            llir.Add,
            "right",
            "right",
            "invalid_binary_child",
            ("root", "right"),
        ),
    ),
)
def test_forged_binary_fields_fail_at_traversal_boundary(
    operation: str,
    node_type: Type[llir.BinOp],
    field: str,
    invalid: object,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    expression = object.__new__(node_type)
    object.__setattr__(expression, "op", "*" if node_type is llir.Mul else "+")
    object.__setattr__(expression, "left", _var("left"))
    object.__setattr__(expression, "right", _var("right"))
    object.__setattr__(expression, field, invalid)

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(expression)
        else:
            LLIRRewriter(_CONTEXT).rewrite(expression)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("field", "invalid", "diagnostic_code", "expected_path"),
    (
        (
            "value",
            object(),
            "invalid_literal_value",
            ("root", "value"),
        ),
        (
            "data_type",
            None,
            "invalid_literal_data_type",
            ("root", "data_type"),
        ),
    ),
)
def test_forged_literal_fields_fail_at_traversal_boundary(
    operation: str,
    field: str,
    invalid: object,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    literal = object.__new__(llir.Literal)
    object.__setattr__(literal, "value", 1)
    object.__setattr__(literal, "data_type", llir.DataType.INT32)
    object.__setattr__(literal, field, invalid)

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(literal)
        else:
            LLIRRewriter(_CONTEXT).rewrite(literal)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("field", "invalid", "diagnostic_code", "expected_path"),
    (
        (
            "namespace",
            "torch::detail",
            "invalid_qualified_name_namespace",
            ("root", "namespace"),
        ),
        (
            "name",
            "k-Int",
            "invalid_qualified_name_name",
            ("root", "name"),
        ),
        (
            "data_type",
            "torch::ScalarType",
            "invalid_qualified_name_data_type",
            ("root", "data_type"),
        ),
    ),
)
def test_forged_qualified_name_fields_fail_at_traversal_boundary(
    operation: str,
    field: str,
    invalid: object,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
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

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(expression)
        else:
            LLIRRewriter(_CONTEXT).rewrite(expression)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_binary_unknown_child_reports_exact_path(operation: str) -> None:
    class UnknownExpr(llir.Expr):
        pass

    expression = llir.Add(UnknownExpr(), llir.Literal(1))
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(expression)
        else:
            LLIRRewriter(_CONTEXT).rewrite(expression)

    assert raised.value.diagnostic.node_type == "UnknownExpr"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "left")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_array_access_subclass_fails_closed(operation: str) -> None:
    class UnknownArrayAccess(llir.ArrayAccess):
        pass

    unknown = UnknownArrayAccess(_var("array"), _var("index"))
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownArrayAccess"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root",)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_array_subclass_fails_closed(operation: str) -> None:
    class UnknownArray(llir.Array):
        pass

    unknown = UnknownArray((_var("extent"),), llir.DataType.INT64)
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownArray"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root",)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_fixed_stack_array_decl_subclass_fails_closed(
    operation: str,
) -> None:
    class UnknownFixedStackArrayDecl(llir.FixedStackArrayDecl):
        pass

    unknown = UnknownFixedStackArrayDecl(
        "wksp",
        llir.DataType.FLOAT32,
        llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
        llir.Array((), llir.DataType.FLOAT32),
    )
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownFixedStackArrayDecl"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root",)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("malformation", "diagnostic_code", "expected_path"),
    (
        (
            "name",
            "invalid_fixed_stack_array_name",
            ("root", "name"),
        ),
        (
            "element_type",
            "invalid_fixed_stack_array_element_type",
            ("root", "element_type"),
        ),
        (
            "extent",
            "invalid_fixed_stack_array_extent",
            ("root", "extent"),
        ),
        (
            "initializer",
            "invalid_fixed_stack_array_initializer",
            ("root", "initializer"),
        ),
        (
            "initializer_values_container",
            "invalid_fixed_stack_array_initializer",
            ("root", "initializer", "values"),
        ),
        (
            "initializer_values",
            "invalid_fixed_stack_array_initializer",
            ("root", "initializer", "values"),
        ),
        (
            "initializer_type",
            "invalid_fixed_stack_array_initializer_type",
            ("root", "initializer", "data_type"),
        ),
    ),
)
def test_forged_fixed_stack_array_fields_fail_at_traversal_boundary(
    operation: str,
    malformation: str,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
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

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(declaration)
        else:
            LLIRRewriter(_CONTEXT).rewrite(declaration)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize("child_kind", ["extent", "initializer"])
def test_fixed_stack_array_unknown_children_fail_at_exact_path(
    operation: str,
    child_kind: str,
) -> None:
    class UnknownExpr(llir.Expr):
        pass

    class UnknownArray(llir.Array):
        pass

    declaration = llir.FixedStackArrayDecl(
        "wksp",
        llir.DataType.FLOAT32,
        llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
        llir.Array((), llir.DataType.FLOAT32),
    )
    if child_kind == "extent":
        object.__setattr__(declaration, "extent", UnknownExpr())
        diagnostic_code = "invalid_fixed_stack_array_extent"
        expected_path = ("root", "extent")
    else:
        object.__setattr__(
            declaration,
            "initializer",
            UnknownArray((), llir.DataType.FLOAT32),
        )
        diagnostic_code = "invalid_fixed_stack_array_initializer"
        expected_path = ("root", "initializer")

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(declaration)
        else:
            LLIRRewriter(_CONTEXT).rewrite(declaration)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_member_access_subclass_fails_closed(operation: str) -> None:
    class UnknownMemberAccess(llir.MemberAccess):
        pass

    unknown = UnknownMemberAccess(_var("it"), "second")
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownMemberAccess"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root",)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_member_call_subclass_fails_closed(operation: str) -> None:
    class UnknownMemberCall(llir.MemberCall):
        pass

    unknown = UnknownMemberCall(_var("tensor"), "data")
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownMemberCall"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root",)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_function_call_subclass_fails_closed(operation: str) -> None:
    class UnknownFunctionCall(llir.FunctionCall):
        pass

    unknown = UnknownFunctionCall("call")
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownFunctionCall"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root",)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_increment_subclass_fails_closed(operation: str) -> None:
    class UnknownIncrement(llir.Increment):
        pass

    unknown = UnknownIncrement(_var("counter"))
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownIncrement"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root",)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_forged_increment_child_fails_at_exact_path(operation: str) -> None:
    increment = object.__new__(llir.Increment)
    object.__setattr__(increment, "var", llir.Literal(1))

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(increment)
        else:
            LLIRRewriter(_CONTEXT).rewrite(increment)

    assert raised.value.diagnostic.code == "invalid_var_child"
    assert raised.value.diagnostic.path == ("root", "var")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_cast_subclass_fails_closed(operation: str) -> None:
    class UnknownCast(llir.Cast):
        pass

    unknown = UnknownCast(_var("value"), llir.DataType.INT)
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownCast"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root",)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("malformation", "diagnostic_code", "expected_path"),
    (
        ("expression", "invalid_cast_expression", ("root", "expr")),
        ("data_type", "invalid_cast_data_type", ("root", "data_type")),
    ),
)
def test_forged_cast_fields_fail_at_traversal_boundary(
    operation: str,
    malformation: str,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    expression = llir.Cast(_var("value"), llir.DataType.INT)
    if malformation == "expression":
        object.__setattr__(expression, "expr", "value")
    else:
        object.__setattr__(expression, "data_type", "int")

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(expression)
        else:
            LLIRRewriter(_CONTEXT).rewrite(expression)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_cast_unknown_child_reports_exact_path(operation: str) -> None:
    class UnknownExpr(llir.Expr):
        pass

    expression = llir.Cast(UnknownExpr(), llir.DataType.INT)
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(expression)
        else:
            LLIRRewriter(_CONTEXT).rewrite(expression)

    assert raised.value.diagnostic.node_type == "UnknownExpr"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "expr")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize("malformation", ("invalid", "missing"))
def test_forged_sizeof_data_type_fails_at_traversal_boundary(
    operation: str,
    malformation: str,
) -> None:
    expression = object.__new__(llir.Sizeof)
    if malformation == "invalid":
        object.__setattr__(expression, "data_type", "float")

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(expression)
        else:
            LLIRRewriter(_CONTEXT).rewrite(expression)

    assert raised.value.diagnostic.code == "invalid_sizeof_data_type"
    assert raised.value.diagnostic.path == ("root", "data_type")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_nested_zero_fill_sizeof_reports_exact_traversal_path(
    operation: str,
) -> None:
    expression = object.__new__(llir.Sizeof)
    object.__setattr__(expression, "data_type", "float")
    zero_fill = llir.FunctionCallStmt(
        "memset",
        (
            _var("workspace"),
            llir.Literal(0),
            llir.Mul(_var("size"), expression),
        ),
    )

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(zero_fill)
        else:
            LLIRRewriter(_CONTEXT).rewrite(zero_fill)

    assert raised.value.diagnostic.code == "invalid_sizeof_data_type"
    assert raised.value.diagnostic.path == (
        "root",
        "args",
        "[2]",
        "right",
        "data_type",
    )


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
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
def test_forged_non_lvalue_address_of_operand_fails_at_traversal_boundary(
    operation: str,
    operand: object,
) -> None:
    expression = object.__new__(llir.AddressOf)
    object.__setattr__(expression, "operand", operand)

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(expression)
        else:
            LLIRRewriter(_CONTEXT).rewrite(expression)

    assert raised.value.diagnostic.code == "invalid_address_of_operand"
    assert raised.value.diagnostic.path == ("root", "operand")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_forged_missing_address_of_operand_fails_at_traversal_boundary(
    operation: str,
) -> None:
    expression = object.__new__(llir.AddressOf)

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(expression)
        else:
            LLIRRewriter(_CONTEXT).rewrite(expression)

    assert raised.value.diagnostic.code == "invalid_address_of_operand"
    assert raised.value.diagnostic.path == ("root", "operand")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("malformation", "relative_path"),
    (
        ("var_name", ("name",)),
        ("var_type", ("type",)),
        ("var_is_ptr", ("is_ptr",)),
        ("var_is_restrict", ("is_restrict",)),
        ("var_tensor_access", ("tensor_access",)),
        ("member_base", ("base",)),
        ("member_member", ("member",)),
        ("array_array", ("array",)),
        ("array_index", ("index",)),
        ("array_tensor_access", ("tensor_access",)),
        ("metadata_role", ("tensor_access", "role")),
        ("index_var_is_ptr", ("index", "is_ptr")),
        ("index_var_is_restrict", ("index", "is_restrict")),
        ("index_var_tensor_access", ("index", "tensor_access")),
        ("index_var_invalid_is_ptr", ("index", "is_ptr")),
        ("index_var_invalid_is_restrict", ("index", "is_restrict")),
        ("index_array_tensor_access", ("index", "tensor_access")),
        ("member_cycle", ("base",)),
        ("index_array_cycle", ("index", "index")),
        ("index_binop_cycle", ("index", "left")),
        ("index_unary_op_subclass", ("index", "op")),
    ),
)
def test_malformed_address_lvalue_reports_exact_traversal_path(
    operation: str,
    malformation: str,
    relative_path: Tuple[str, ...],
) -> None:
    expression = object.__new__(llir.AddressOf)
    object.__setattr__(
        expression,
        "operand",
        _malformed_address_operand(malformation),
    )

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(expression)
        else:
            LLIRRewriter(_CONTEXT).rewrite(expression)

    assert raised.value.diagnostic.code == "invalid_address_of_operand"
    assert raised.value.diagnostic.path == ("root", "operand") + relative_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_nested_write_back_address_of_reports_exact_traversal_path(
    operation: str,
) -> None:
    expression = object.__new__(llir.AddressOf)
    object.__setattr__(
        expression,
        "operand",
        llir.Add(_var("base"), _var("offset")),
    )
    write_back = llir.FunctionCallStmt(
        "memcpy",
        (
            expression,
            _var("workspace"),
            llir.Mul(_var("size"), llir.Sizeof(llir.DataType.FLOAT32)),
        ),
    )

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(write_back)
        else:
            LLIRRewriter(_CONTEXT).rewrite(write_back)

    assert raised.value.diagnostic.code == "invalid_address_of_operand"
    assert raised.value.diagnostic.path == (
        "root",
        "args",
        "[0]",
        "operand",
    )


def test_address_of_rewrite_rejects_a_non_lvalue_child_replacement() -> None:
    class ReplaceOperandWithLiteral(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            return cast(llir.Var, llir.Literal(0))

    with pytest.raises(LLIRTraversalError) as raised:
        ReplaceOperandWithLiteral(_CONTEXT).rewrite(llir.AddressOf(_var("value")))

    assert raised.value.diagnostic.code == "invalid_address_of_operand"
    assert raised.value.diagnostic.path == ("root", "operand")


def test_addressed_copy_walker_has_deterministic_preorder() -> None:
    write_back = llir.FunctionCallStmt(
        "memcpy",
        (
            llir.AddressOf(
                operand=llir.ArrayAccess(
                    array=_var("C_values", llir.DataType.PTR_FLOAT32),
                    index=llir.Mul(
                        _var("pC0", llir.DataType.INT64),
                        _var("C1_size", llir.DataType.INT64),
                    ),
                ),
            ),
            _var("wksp", llir.DataType.PTR_FLOAT32),
            llir.Mul(
                _var("wksp0_size", llir.DataType.INT64),
                llir.Sizeof(llir.DataType.FLOAT32),
            ),
        ),
    )

    expected = [
        "FunctionCallStmt",
        "AddressOf",
        "ArrayAccess",
        "Var:C_values",
        "Mul",
        "Var:pC0",
        "Var:C1_size",
        "Var:wksp",
        "Mul",
        "Var:wksp0_size",
        "Sizeof",
    ]
    assert _record(write_back) == expected
    assert _record(write_back) == expected


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_direct_init_subclass_fails_closed(operation: str) -> None:
    class UnknownDirectInit(llir.DirectInit):
        pass

    unknown = UnknownDirectInit(
        llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
        (llir.Literal(4),),
    )
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownDirectInit"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root",)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("malformation", "diagnostic_code", "expected_path"),
    (
        ("target", "invalid_direct_init_var", ("root", "var")),
        ("name", "invalid_direct_init_var_name", ("root", "var", "name")),
        ("type", "invalid_direct_init_var_type", ("root", "var", "type")),
        (
            "pointer",
            "invalid_direct_init_var_is_ptr",
            ("root", "var", "is_ptr"),
        ),
        (
            "restrict",
            "invalid_direct_init_var_is_restrict",
            ("root", "var", "is_restrict"),
        ),
        (
            "metadata",
            "invalid_direct_init_var_metadata",
            ("root", "var", "tensor_access"),
        ),
        ("args", "invalid_direct_init_args", ("root", "args")),
        ("empty", "empty_direct_init_args", ("root", "args")),
        (
            "argument",
            "invalid_direct_init_argument",
            ("root", "args", "[0]"),
        ),
    ),
)
def test_forged_direct_init_fields_fail_at_exact_traversal_path(
    operation: str,
    malformation: str,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    declaration = llir.DirectInit(
        llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
        (llir.Literal(4),),
    )
    if malformation == "target":
        object.__setattr__(declaration, "var", llir.Literal(1))
    elif malformation == "name":
        declaration.var.name = "not a name"
    elif malformation == "type":
        declaration.var.type = llir.DataType.VOID
    elif malformation == "pointer":
        declaration.var.is_ptr = True
    elif malformation == "restrict":
        declaration.var.is_restrict = True
    elif malformation == "metadata":
        declaration.var.tensor_access = _result_metadata()
    elif malformation == "args":
        object.__setattr__(declaration, "args", [llir.Literal(4)])
    elif malformation == "empty":
        object.__setattr__(declaration, "args", ())
    else:
        object.__setattr__(declaration, "args", ("extent",))

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(declaration)
        else:
            LLIRRewriter(_CONTEXT).rewrite(declaration)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_direct_init_unknown_child_reports_exact_path(operation: str) -> None:
    class UnknownExpr(llir.Expr):
        pass

    declaration = llir.DirectInit(
        llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
        (UnknownExpr(),),
    )
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(declaration)
        else:
            LLIRRewriter(_CONTEXT).rewrite(declaration)

    assert raised.value.diagnostic.node_type == "UnknownExpr"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "args", "[0]")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("missing_field", "diagnostic_code", "expected_path"),
    (
        ("var", "invalid_direct_init_var", ("root", "var")),
        ("args", "invalid_direct_init_args", ("root", "args")),
        ("var_name", "invalid_direct_init_var_name", ("root", "var", "name")),
    ),
)
def test_direct_init_missing_forged_fields_fail_closed(
    operation: str,
    missing_field: str,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    declaration = object.__new__(llir.DirectInit)
    variable = (
        object.__new__(llir.Var)
        if missing_field == "var_name"
        else llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32)
    )
    if missing_field != "var":
        object.__setattr__(declaration, "var", variable)
    if missing_field != "args":
        object.__setattr__(declaration, "args", (llir.Literal(4),))

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(declaration)
        else:
            LLIRRewriter(_CONTEXT).rewrite(declaration)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


def test_direct_init_rewrite_rejects_invalid_target_replacement() -> None:
    declaration = llir.DirectInit(
        llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
        (llir.Literal(4),),
    )

    class InvalidTargetReplacement(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            return llir.Var(node.name, llir.DataType.VOID)

    with pytest.raises(LLIRTraversalError) as raised:
        InvalidTargetReplacement(_CONTEXT).rewrite(declaration)

    assert raised.value.diagnostic.code == "invalid_direct_init_var_type"
    assert raised.value.diagnostic.path == ("root", "var", "type")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize("target_kind", ["array", "member"])
def test_unknown_assignment_target_subclass_fails_closed(
    operation: str,
    target_kind: str,
) -> None:
    class UnknownArrayAccess(llir.ArrayAccess):
        pass

    class UnknownMemberAccess(llir.MemberAccess):
        pass

    assignment = llir.Assign(_var("output"), llir.Literal(1))
    assignment.var = cast(
        llir.AssignmentTarget,
        (
            UnknownArrayAccess(_var("values"), _var("index"))
            if target_kind == "array"
            else UnknownMemberAccess(_var("output"), "value")
        ),
    )

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(assignment)
        else:
            LLIRRewriter(_CONTEXT).rewrite(assignment)

    assert raised.value.diagnostic.node_type == (
        "UnknownArrayAccess" if target_kind == "array" else "UnknownMemberAccess"
    )
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "var")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_array_access_unknown_child_reports_exact_path(operation: str) -> None:
    class UnknownExpr(llir.Expr):
        pass

    access = llir.ArrayAccess(UnknownExpr(), _var("index"))
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(access)
        else:
            LLIRRewriter(_CONTEXT).rewrite(access)

    assert raised.value.diagnostic.node_type == "UnknownExpr"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "array")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_member_access_unknown_child_reports_exact_path(operation: str) -> None:
    class UnknownExpr(llir.Expr):
        pass

    access = llir.MemberAccess(UnknownExpr(), "second")
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(access)
        else:
            LLIRRewriter(_CONTEXT).rewrite(access)

    assert raised.value.diagnostic.node_type == "UnknownExpr"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "base")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize("child", ["base", "argument"])
def test_member_call_unknown_children_report_exact_paths(
    operation: str,
    child: str,
) -> None:
    class UnknownExpr(llir.Expr):
        pass

    call = (
        llir.MemberCall(UnknownExpr(), "data")
        if child == "base"
        else llir.MemberCall(_var("tensor"), "at", args=(UnknownExpr(),))
    )
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(call)
        else:
            LLIRRewriter(_CONTEXT).rewrite(call)

    assert raised.value.diagnostic.node_type == "UnknownExpr"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == (
        ("root", "base") if child == "base" else ("root", "args", "[0]")
    )


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_function_call_unknown_child_reports_exact_path(operation: str) -> None:
    class UnknownExpr(llir.Expr):
        pass

    call = llir.FunctionCall("call", [UnknownExpr()])
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(call)
        else:
            LLIRRewriter(_CONTEXT).rewrite(call)

    assert raised.value.diagnostic.node_type == "UnknownExpr"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "args", "[0]")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_array_unknown_child_reports_exact_path(operation: str) -> None:
    class UnknownExpr(llir.Expr):
        pass

    array = llir.Array((UnknownExpr(),), llir.DataType.INT64)
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(array)
        else:
            LLIRRewriter(_CONTEXT).rewrite(array)

    assert raised.value.diagnostic.node_type == "UnknownExpr"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "values", "[0]")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("malformation", "diagnostic_code", "expected_path"),
    (
        ("values", "invalid_array_values", ("root", "values")),
        ("value", "invalid_array_value", ("root", "values", "[0]")),
        ("data_type", "invalid_array_data_type", ("root", "data_type")),
    ),
)
def test_forged_array_fields_fail_at_traversal_boundary(
    operation: str,
    malformation: str,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    array = object.__new__(llir.Array)
    object.__setattr__(array, "values", (_var("extent"),))
    object.__setattr__(array, "data_type", llir.DataType.INT64)
    if malformation == "values":
        object.__setattr__(array, "values", [_var("extent")])
    elif malformation == "value":
        object.__setattr__(array, "values", ("extent",))
    else:
        object.__setattr__(array, "data_type", "int64_t")

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(array)
        else:
            LLIRRewriter(_CONTEXT).rewrite(array)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("malformation", "diagnostic_code", "expected_path"),
    (
        ("name", "invalid_function_call_name", ("root", "name")),
        ("args", "invalid_function_call_args", ("root", "args")),
        (
            "argument",
            "invalid_expression_sequence_member",
            ("root", "args", "[0]"),
        ),
    ),
)
def test_forged_function_call_fields_fail_at_traversal_boundary(
    operation: str,
    malformation: str,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    call = object.__new__(llir.FunctionCall)
    object.__setattr__(call, "name", "call")
    object.__setattr__(call, "args", (_var("argument"),))
    if malformation == "name":
        object.__setattr__(call, "name", " ")
    elif malformation == "args":
        object.__setattr__(call, "args", [_var("argument")])
    else:
        object.__setattr__(call, "args", ("argument",))

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(call)
        else:
            LLIRRewriter(_CONTEXT).rewrite(call)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("malformation", "diagnostic_code", "expected_path"),
    (
        ("base", "invalid_member_access_base", ("root", "base")),
        ("member", "invalid_member_access_member", ("root", "member")),
    ),
)
def test_forged_member_access_fields_fail_at_traversal_boundary(
    operation: str,
    malformation: str,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    access = object.__new__(llir.MemberAccess)
    object.__setattr__(access, "base", _var("it"))
    object.__setattr__(access, "member", "second")
    if malformation == "base":
        object.__setattr__(access, "base", "it")
    else:
        object.__setattr__(access, "member", "first.second")

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(access)
        else:
            LLIRRewriter(_CONTEXT).rewrite(access)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("malformation", "diagnostic_code", "expected_path"),
    (
        ("base", "invalid_member_call_base", ("root", "base")),
        ("member", "invalid_member_call_member", ("root", "member")),
        (
            "template_args",
            "invalid_member_call_template_args",
            ("root", "template_args"),
        ),
        (
            "template_arg",
            "invalid_member_call_template_arg",
            ("root", "template_args", "[0]"),
        ),
        ("args", "invalid_member_call_args", ("root", "args")),
        (
            "argument",
            "invalid_member_call_argument",
            ("root", "args", "[0]"),
        ),
    ),
)
def test_forged_member_call_fields_fail_at_traversal_boundary(
    operation: str,
    malformation: str,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    call = object.__new__(llir.MemberCall)
    object.__setattr__(call, "base", _var("tensor"))
    object.__setattr__(call, "member", "data_ptr")
    object.__setattr__(call, "template_args", (llir.DataType.FLOAT32,))
    object.__setattr__(call, "args", (_var("argument"),))
    if malformation == "base":
        object.__setattr__(call, "base", "tensor")
    elif malformation == "member":
        object.__setattr__(call, "member", "tensor.data_ptr")
    elif malformation == "template_args":
        object.__setattr__(call, "template_args", [llir.DataType.FLOAT32])
    elif malformation == "template_arg":
        object.__setattr__(call, "template_args", ("float",))
    elif malformation == "args":
        object.__setattr__(call, "args", [_var("argument")])
    else:
        object.__setattr__(call, "args", ("argument",))

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(call)
        else:
            LLIRRewriter(_CONTEXT).rewrite(call)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("malformation", "diagnostic_code", "expected_path"),
    (
        ("base", "invalid_member_call_stmt_base", ("root", "base")),
        ("member", "invalid_member_call_stmt_member", ("root", "member")),
        (
            "template_args",
            "invalid_member_call_stmt_template_args",
            ("root", "template_args"),
        ),
        (
            "template_arg",
            "invalid_member_call_stmt_template_arg",
            ("root", "template_args", "[0]"),
        ),
        ("args", "invalid_member_call_stmt_args", ("root", "args")),
        (
            "argument",
            "invalid_member_call_stmt_argument",
            ("root", "args", "[0]"),
        ),
    ),
)
def test_forged_member_call_stmt_fields_fail_at_traversal_boundary(
    operation: str,
    malformation: str,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    call = object.__new__(llir.MemberCallStmt)
    object.__setattr__(call, "base", _var("workspace"))
    object.__setattr__(call, "member", "clear")
    object.__setattr__(call, "template_args", (llir.DataType.FLOAT32,))
    object.__setattr__(call, "args", (_var("argument"),))
    if malformation == "base":
        object.__setattr__(call, "base", "workspace")
    elif malformation == "member":
        object.__setattr__(call, "member", "workspace.clear")
    elif malformation == "template_args":
        object.__setattr__(call, "template_args", [llir.DataType.FLOAT32])
    elif malformation == "template_arg":
        object.__setattr__(call, "template_args", ("float",))
    elif malformation == "args":
        object.__setattr__(call, "args", [_var("argument")])
    else:
        object.__setattr__(call, "args", ("argument",))

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(call)
        else:
            LLIRRewriter(_CONTEXT).rewrite(call)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("missing_field", "diagnostic_code", "expected_path"),
    (
        ("base", "invalid_member_call_stmt_base", ("root", "base")),
        ("member", "invalid_member_call_stmt_member", ("root", "member")),
        (
            "template_args",
            "invalid_member_call_stmt_template_args",
            ("root", "template_args"),
        ),
        ("args", "invalid_member_call_stmt_args", ("root", "args")),
    ),
)
def test_forged_member_call_stmt_missing_fields_fail_closed(
    operation: str,
    missing_field: str,
    diagnostic_code: str,
    expected_path: Tuple[str, ...],
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

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(call)
        else:
            LLIRRewriter(_CONTEXT).rewrite(call)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    "malformation",
    [
        "rvalue",
        "rvalue_name",
        "opaque_member_expression",
        "member_root",
        "array_base",
        "flat_index",
        "read_role",
    ],
)
def test_forged_malformed_assignment_target_fails_at_traversal_boundary(
    operation: str,
    malformation: str,
) -> None:
    assignment = llir.Assign(_var("output"), llir.Literal(1))
    if malformation == "rvalue":
        assignment.var = cast(llir.AssignmentTarget, llir.Literal(0))
    elif malformation == "rvalue_name":
        assignment.var = _var("left + right")
    elif malformation == "opaque_member_expression":
        assignment.var = _var("Result.storage.value[0]")
    elif malformation == "member_root":
        assignment.var = llir.MemberAccess(
            llir.Add(_var("Result"), _var("offset")),
            "storage",
        )
    elif malformation == "array_base":
        assignment.var = llir.ArrayAccess(
            llir.Add(_var("values"), _var("offset")),
            _var("index"),
        )
    elif malformation == "flat_index":
        assignment.var = llir.ArrayAccess(
            _var("values"),
            _var("indices[index]"),
        )
    else:
        assignment.var = llir.ArrayAccess(
            _var("values"),
            _var("index"),
            llir.TensorAccessMetadata(
                access_id=AccessId(1),
                tensor_id=SymbolId(2),
                index_ids=(IndexId(3),),
                role=llir.TensorAccessRole.INPUT_READ,
            ),
        )

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(assignment)
        else:
            LLIRRewriter(_CONTEXT).rewrite(assignment)

    assert raised.value.diagnostic.code == "invalid_assignment_target"
    assert raised.value.diagnostic.path == ("root", "var")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_nested_member_assignment_target_subclass_fails_closed(
    operation: str,
) -> None:
    class UnknownMemberAccess(llir.MemberAccess):
        pass

    target = llir.MemberAccess(
        UnknownMemberAccess(_var("Result"), "storage"),
        "value",
    )
    assignment = llir.Assign(_var("output"), llir.Literal(1))
    assignment.var = target

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(assignment)
        else:
            LLIRRewriter(_CONTEXT).rewrite(assignment)

    assert raised.value.diagnostic.node_type == "UnknownMemberAccess"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "var", "base")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_assignment_target_unknown_child_reports_exact_path(operation: str) -> None:
    class UnknownExpr(llir.Expr):
        pass

    target = object.__new__(llir.ArrayAccess)
    object.__setattr__(target, "array", _var("values"))
    object.__setattr__(target, "index", UnknownExpr())
    object.__setattr__(target, "tensor_access", None)
    assignment = llir.Assign(_var("output"), llir.Literal(1))
    assignment.var = target

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(assignment)
        else:
            LLIRRewriter(_CONTEXT).rewrite(assignment)

    assert raised.value.diagnostic.node_type == "UnknownExpr"
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "var", "index")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_forged_structured_cast_target_fails_at_traversal_boundary(
    operation: str,
) -> None:
    assignment = llir.Assign(
        llir.ArrayAccess(_var("values"), _var("index")),
        llir.Literal(1),
    )
    assignment.cast = True

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(assignment)
        else:
            LLIRRewriter(_CONTEXT).rewrite(assignment)

    assert raised.value.diagnostic.code == "invalid_assign_cast_target"
    assert raised.value.diagnostic.path == ("root", "var")


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_forged_array_access_metadata_fails_at_traversal_boundary(
    operation: str,
) -> None:
    access = object.__new__(llir.ArrayAccess)
    object.__setattr__(access, "array", _var("array"))
    object.__setattr__(access, "index", _var("index"))
    object.__setattr__(access, "tensor_access", "invalid")

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(access)
        else:
            LLIRRewriter(_CONTEXT).rewrite(access)

    assert raised.value.diagnostic.code == "invalid_tensor_access_metadata"
    assert raised.value.diagnostic.path == ("root", "tensor_access")
    assert raised.value.diagnostic.node_type == "str"


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize("owner", ["var", "array_access"])
@pytest.mark.parametrize(
    ("field", "invalid", "expected_path"),
    (
        ("access_id", 1, ("access_id",)),
        ("tensor_id", "tensor", ("tensor_id",)),
        ("index_ids", [IndexId(3)], ("index_ids",)),
        ("index_ids", (3,), ("index_ids", "[0]")),
        ("role", "input_read", ("role",)),
    ),
)
def test_forged_exact_metadata_fields_fail_at_common_traversal_boundary(
    operation: str,
    owner: str,
    field: str,
    invalid: object,
    expected_path: Tuple[str, ...],
) -> None:
    metadata = object.__new__(llir.TensorAccessMetadata)
    object.__setattr__(metadata, "access_id", AccessId(1))
    object.__setattr__(metadata, "tensor_id", SymbolId(2))
    object.__setattr__(metadata, "index_ids", (IndexId(3),))
    object.__setattr__(metadata, "role", llir.TensorAccessRole.INPUT_READ)
    object.__setattr__(metadata, field, invalid)
    node: llir.Expr
    if owner == "var":
        node = llir.Var("value", llir.DataType.FLOAT32, tensor_access=metadata)
    else:
        node = llir.ArrayAccess(_var("array"), _var("index"), metadata)

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(node)
        else:
            LLIRRewriter(_CONTEXT).rewrite(node)

    assert raised.value.diagnostic.code == "invalid_tensor_access_metadata"
    assert raised.value.diagnostic.path == (
        "root",
        "tensor_access",
        *expected_path,
    )


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_unknown_subclass_of_supported_statement_fails_closed(
    operation: str,
) -> None:
    class UnknownBreak(llir.Break):
        pass

    unknown = UnknownBreak()
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(unknown)
        else:
            LLIRRewriter(_CONTEXT).rewrite(unknown)

    assert raised.value.diagnostic.node_type == "UnknownBreak"
    assert raised.value.diagnostic.code == "unknown_llir_node"


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_nested_unknown_node_reports_the_same_exact_path(operation: str) -> None:
    class UnknownStmt(llir.Stmt):
        pass

    root: List[LLIRStatementValue] = [[UnknownStmt()]]
    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(root)
        else:
            LLIRRewriter(_CONTEXT).rewrite(root)

    assert raised.value.diagnostic.path == ("root", "[0]", "[0]")
    assert raised.value.diagnostic.node_type == "UnknownStmt"


def test_identity_rewriter_is_detached_and_structurally_idempotent() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(11),
        tensor_id=SymbolId(12),
        index_ids=(IndexId(13),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    tagged_var = llir.Var(
        name="Input_values[pInput]",
        type=llir.DataType.FLOAT32,
        tensor_access=metadata,
    )
    access = llir.ArrayAccess(
        array=_var("Input_values", llir.DataType.PTR_FLOAT32),
        index=_var("pInput"),
        tensor_access=metadata,
    )
    target = llir.ArrayAccess(
        array=_var("Output_values", llir.DataType.PTR_FLOAT32),
        index=llir.Add(_var("pOutput"), llir.Literal(1)),
        tensor_access=_result_metadata(access_id=14),
    )
    cast_init = llir.VarInit(_var("converted"), llir.Literal(1), cast=True)
    cast_assign = llir.Assign(_var("assigned"), llir.Literal(2), cast=True)
    loop = llir.ForLoop(
        init=llir.VarInit(_var("i"), llir.Literal(0)),
        cond=llir.BinOp("<", _var("i"), _var("n")),
        update=llir.Increment(_var("i")),
        body=[llir.Assign(target, access)],
        omp_parallel_for=True,
        omp_schedule="dynamic, 8",
        unroll=True,
        simd=True,
        before_parallel_body=[llir.Comment("before")],
        pre_parallel_body=[],
        post_parallel_body=[llir.RawStmt("post")],
        omp_num_threads="threads",
        omp_chunk_expr="chunk_size",
    )
    loop.scorch_index_var = "i"
    setattr(loop, "_use_atomic_scheduling", True)
    setattr(loop, "_atomic_chunk_var", "chunk")
    setattr(loop, "_atomic_counter_var", "counter")
    setattr(loop, "_loop_bound", "n")
    setattr(loop, "_hoisted_ptr_decls", [llir.RawStmt("hoisted")])
    while_loop = llir.WhileLoop(_var("keep_going"), [llir.Break()])
    while_loop.scorch_index_var = "while_index"

    root: List[LLIRStatementValue] = [
        llir.VarDecl(tagged_var),
        (cast_init, [cast_assign, loop]),
        while_loop,
    ]
    rewriter = LLIRRewriter(_CONTEXT)
    first = rewriter.rewrite(root)
    second = rewriter.rewrite(first)

    assert _record(root) == _record(first) == _record(second)
    assert _structural_snapshot(root) == _structural_snapshot(first)
    assert _structural_snapshot(first) == _structural_snapshot(second)
    assert _mutable_ir_ids(root).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    assert first is not root
    assert type(first[1]) is tuple
    first_nested = cast(Tuple[object, ...], first[1])
    second_nested = cast(Tuple[object, ...], second[1])
    first_cast_init = cast(llir.VarInit, first_nested[0])
    second_cast_init = cast(llir.VarInit, second_nested[0])
    assert first_cast_init.cast is True
    assert type(first_cast_init.value) is llir.Cast
    assert type(cast(llir.Cast, first_cast_init.value).expr) is llir.Literal
    assert type(second_cast_init.value) is llir.Cast
    assert type(cast(llir.Cast, second_cast_init.value).expr) is llir.Literal
    first_nested_body = cast(List[LLIRStatementValue], first_nested[1])
    second_nested_body = cast(List[LLIRStatementValue], second_nested[1])
    first_cast_assign = cast(llir.Assign, first_nested_body[0])
    second_cast_assign = cast(llir.Assign, second_nested_body[0])
    assert type(first_cast_assign.value) is llir.Cast
    assert type(cast(llir.Cast, first_cast_assign.value).expr) is llir.Literal
    assert type(second_cast_assign.value) is llir.Cast
    assert type(cast(llir.Cast, second_cast_assign.value).expr) is llir.Literal

    first_decl = cast(llir.VarDecl, first[0])
    assert first_decl.var.tensor_access is metadata
    first_loop = cast(llir.ForLoop, first_nested_body[1])
    first_assignment = cast(llir.Assign, first_loop.body[0])
    first_target = cast(llir.ArrayAccess, first_assignment.var)
    first_access = cast(llir.ArrayAccess, first_assignment.value)
    assert first_target.tensor_access is target.tensor_access
    assert first_target is not target
    assert first_target.array is not target.array
    assert first_target.index is not target.index
    assert first_access.tensor_access is metadata
    assert first_access is not access
    assert first_loop.scorch_index_var == "i"
    assert first_loop.omp_parallel_for is True
    assert first_loop.omp_schedule == "dynamic, 8"
    assert first_loop.unroll is True
    assert first_loop.simd is True
    assert first_loop.omp_num_threads == "threads"
    assert first_loop.omp_chunk_expr == "chunk_size"
    assert first_loop.before_parallel_body is not None
    assert type(first_loop.before_parallel_body[0]) is llir.Comment
    assert getattr(first_loop, "_use_atomic_scheduling") is True
    assert getattr(first_loop, "_atomic_chunk_var") == "chunk"
    assert getattr(first_loop, "_atomic_counter_var") == "counter"
    assert getattr(first_loop, "_loop_bound") == "n"
    assert first_loop.pre_parallel_body == []
    assert type(getattr(first_loop, "_hoisted_ptr_decls")[0]) is llir.RawStmt
    first_while = cast(llir.WhileLoop, first[2])
    assert first_while.scorch_index_var == "while_index"

    first_decl.var.name = "changed"
    first_loop.body.append(llir.Break())
    assert cast(llir.VarDecl, root[0]).var.name == "Input_values[pInput]"
    original_nested = cast(Tuple[object, ...], root[1])
    original_body = cast(List[LLIRStatementValue], original_nested[1])
    original_loop = cast(llir.ForLoop, original_body[1])
    assert len(original_loop.body) == 1


def test_fixed_stack_array_rewrite_is_detached_repeatable_and_replacement_owned() -> (
    None
):
    original_extent = llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT)
    original_initializer = llir.Array((), llir.DataType.FLOAT32)
    original = llir.FixedStackArrayDecl(
        "wksp",
        llir.DataType.FLOAT32,
        original_extent,
        original_initializer,
    )
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(llir.FixedStackArrayDecl, rewriter.rewrite(original))
    second = cast(llir.FixedStackArrayDecl, rewriter.rewrite(first))

    assert _record(original) == _record(first) == _record(second)
    assert original == first == second
    assert hash(original) == hash(first) == hash(second)
    assert _mutable_ir_ids(original).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    assert type(first.extent) is llir.Var
    assert type(second.extent) is llir.Var
    assert type(first.initializer) is llir.Array
    assert type(second.initializer) is llir.Array
    assert first.extent is not original_extent
    assert second.extent is not first.extent
    assert first.initializer is not original_initializer
    assert second.initializer is not first.initializer
    assert first.initializer.values == second.initializer.values == ()

    class ReplaceExtent(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            if node.name == "kTile_k":
                rewritten.name = "replacement_extent"
            return rewritten

    replacement = cast(
        llir.FixedStackArrayDecl,
        ReplaceExtent(_CONTEXT).rewrite(original),
    )
    replacement_extent = cast(llir.Var, replacement.extent)
    first_extent = cast(llir.Var, first.extent)

    assert replacement is not original
    assert replacement_extent is not original_extent
    assert replacement.initializer is not original_initializer
    assert replacement_extent.name == "replacement_extent"
    assert replacement_extent.type is llir.DataType.CONSTEXPR_INT
    assert original_extent.name == "kTile_k"
    replacement_extent.name = "owned_replacement"
    assert original_extent.name == "kTile_k"
    assert first_extent.name == "kTile_k"


def test_arithmetic_rewrite_is_detached_repeatable_and_replacement_owned() -> None:
    original = llir.BinOp(
        "<",
        llir.Add(
            llir.Mul(_var("scale"), llir.Literal(2, llir.DataType.INT64)),
            llir.Literal(1, llir.DataType.INT64),
        ),
        _var("limit"),
    )
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(llir.BinOp, rewriter.rewrite(original))
    second = cast(llir.BinOp, rewriter.rewrite(first))

    assert type(first) is llir.BinOp
    assert type(first.left) is llir.Add
    assert type(cast(llir.Add, first.left).left) is llir.Mul
    assert type(second) is llir.BinOp
    assert type(second.left) is llir.Add
    assert type(cast(llir.Add, second.left).left) is llir.Mul
    assert _record(original) == _record(first) == _record(second)
    assert original == first == second
    assert hash(original) == hash(first) == hash(second)
    assert _mutable_ir_ids(original).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))

    class ReplaceScale(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            if node.name == "scale":
                rewritten.name = "replacement"
            return rewritten

    replacement = cast(llir.BinOp, ReplaceScale(_CONTEXT).rewrite(original))
    replacement_add = cast(llir.Add, replacement.left)
    replacement_mul = cast(llir.Mul, replacement_add.left)
    original_add = cast(llir.Add, original.left)
    original_mul = cast(llir.Mul, original_add.left)
    replacement_var = cast(llir.Var, replacement_mul.left)
    original_var = cast(llir.Var, original_mul.left)

    assert replacement_var.name == "replacement"
    assert original_var.name == "scale"
    assert replacement is not original
    assert replacement.left is not original.left
    assert replacement_mul is not original_mul
    assert replacement_var is not original_var

    replacement_var.name = "owned"
    assert original_var.name == "scale"
    assert cast(
        llir.Var, cast(llir.Mul, cast(llir.Add, second.left).left).left
    ).name == ("scale")


def test_qualified_name_rewrite_is_detached_repeatable_and_replacement_owned() -> None:
    original_qualified = llir.QualifiedName(
        "torch", "kInt", llir.DataType.TORCH_SCALAR_TYPE
    )
    original = llir.FunctionCall("consume", (original_qualified,))
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(llir.FunctionCall, rewriter.rewrite(original))
    second = cast(llir.FunctionCall, rewriter.rewrite(first))
    first_qualified = cast(llir.QualifiedName, first.args[0])
    second_qualified = cast(llir.QualifiedName, second.args[0])

    assert _record(original) == _record(first) == _record(second)
    assert original == first == second
    assert hash(original) == hash(first) == hash(second)
    assert first is not original
    assert second is not first
    assert first_qualified is not original_qualified
    assert second_qualified is not first_qualified

    class ReplaceQualifiedName(LLIRRewriter):
        def rewrite_qualified_name(
            self,
            node: llir.QualifiedName,
            path: LLIRPath,
        ) -> llir.QualifiedName:
            rewritten = super().rewrite_qualified_name(node, path)
            if node.name == "kInt":
                return llir.QualifiedName(
                    rewritten.namespace,
                    "kFloat32",
                    rewritten.data_type,
                )
            return rewritten

    replacement = cast(
        llir.FunctionCall,
        ReplaceQualifiedName(_CONTEXT).rewrite(original),
    )
    replacement_qualified = cast(llir.QualifiedName, replacement.args[0])

    assert replacement is not original
    assert replacement_qualified is not original_qualified
    assert replacement_qualified.name == "kFloat32"
    assert replacement_qualified.namespace == "torch"
    assert replacement_qualified.data_type is llir.DataType.TORCH_SCALAR_TYPE
    assert original_qualified.name == "kInt"
    assert first_qualified.name == "kInt"


def test_member_access_identity_rewrite_is_detached_idempotent_and_owned() -> None:
    original = llir.ArrayAccess(
        array=llir.MemberAccess(_var("it"), "first"),
        index=llir.Literal(0, llir.DataType.INT64),
    )
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(llir.ArrayAccess, rewriter.rewrite(original))
    second = cast(llir.ArrayAccess, rewriter.rewrite(first))

    assert _structural_snapshot(original) == _structural_snapshot(first)
    assert _structural_snapshot(first) == _structural_snapshot(second)
    assert _mutable_ir_ids(original).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    first_member = cast(llir.MemberAccess, first.array)
    second_member = cast(llir.MemberAccess, second.array)
    original_member = cast(llir.MemberAccess, original.array)
    assert first_member is not original_member
    assert second_member is not first_member
    assert first_member.base is not original_member.base
    assert second_member.base is not first_member.base

    cast(llir.Var, first_member.base).name = "owned"
    assert cast(llir.Var, original_member.base).name == "it"
    assert cast(llir.Var, second_member.base).name == "it"


def test_fill_base_load_rewrite_is_deterministic_repeatable_and_owned() -> None:
    original = llir.VarInit(
        llir.Var("_base1", llir.DataType.INT64),
        llir.ArrayAccess(
            llir.Var("_offset1", llir.DataType.STD_VECTOR_INT),
            llir.Var("row", llir.DataType.INT64),
        ),
    )
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(llir.VarInit, rewriter.rewrite(original))
    second = cast(llir.VarInit, rewriter.rewrite(first))

    assert _record(original) == [
        "VarInit",
        "Var:_base1",
        "ArrayAccess",
        "Var:_offset1",
        "Var:row",
    ]
    assert _record(first) == _record(original)
    assert _record(second) == _record(original)
    assert original == first == second
    assert hash(original) == hash(first) == hash(second)
    assert _mutable_ir_ids(original).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))

    original_access = cast(llir.ArrayAccess, original.value)
    first_access = cast(llir.ArrayAccess, first.value)
    second_access = cast(llir.ArrayAccess, second.value)
    assert first.var is not original.var
    assert second.var is not first.var
    assert first_access is not original_access
    assert second_access is not first_access
    assert first_access.array is not original_access.array
    assert first_access.index is not original_access.index
    assert second_access.array is not first_access.array
    assert second_access.index is not first_access.index

    cast(llir.Var, first_access.array).name = "owned_offset"
    cast(llir.Var, first_access.index).name = "owned_index"
    assert cast(llir.Var, original_access.array).name == "_offset1"
    assert cast(llir.Var, original_access.index).name == "row"
    assert cast(llir.Var, second_access.array).name == "_offset1"
    assert cast(llir.Var, second_access.index).name == "row"

    class ReplaceLoadVars(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            replacements = {
                "_base1": "replacement_base",
                "_offset1": "replacement_offset",
                "row": "replacement_row",
            }
            rewritten.name = replacements.get(node.name, node.name)
            return rewritten

    replacement = cast(llir.VarInit, ReplaceLoadVars(_CONTEXT).rewrite(original))
    replacement_access = cast(llir.ArrayAccess, replacement.value)
    assert replacement.var.name == "replacement_base"
    assert cast(llir.Var, replacement_access.array).name == "replacement_offset"
    assert cast(llir.Var, replacement_access.index).name == "replacement_row"
    assert replacement.var is not original.var
    assert replacement_access is not original_access
    assert replacement_access.array is not original_access.array
    assert replacement_access.index is not original_access.index
    assert original.var.name == "_base1"
    assert cast(llir.Var, original_access.array).name == "_offset1"
    assert cast(llir.Var, original_access.index).name == "row"


def test_known_nnz_coordinate_allocation_rewrite_is_deterministic_and_owned() -> None:
    original: List[llir.Stmt] = [
        llir.VarInit(
            var=llir.Var("Result0_crd_torch", llir.DataType.TORCH_TENSOR),
            value=llir.FunctionCall(
                "torch::empty",
                (
                    llir.Array(
                        (llir.Var("_known_nnz", llir.DataType.INT64),),
                        llir.DataType.INT64,
                    ),
                    llir.QualifiedName(
                        "torch",
                        "kInt",
                        llir.DataType.TORCH_SCALAR_TYPE,
                    ),
                ),
            ),
        ),
        llir.VarInit(
            var=llir.Var("Result0_crd", llir.DataType.PTR_INT),
            value=llir.MemberCall(
                base=llir.Var(
                    "Result0_crd_torch",
                    llir.DataType.TORCH_TENSOR,
                ),
                member="data_ptr",
                template_args=(llir.DataType.INT,),
            ),
        ),
    ]
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(List[llir.Stmt], rewriter.rewrite(original))
    second = cast(List[llir.Stmt], rewriter.rewrite(first))

    assert _record(original) == [
        "VarInit",
        "Var:Result0_crd_torch",
        "FunctionCall",
        "Array",
        "Var:_known_nnz",
        "QualifiedName:torch::kInt",
        "VarInit",
        "Var:Result0_crd",
        "MemberCall",
        "Var:Result0_crd_torch",
    ]
    assert _record(first) == _record(original)
    assert _record(second) == _record(original)
    assert _structural_snapshot(original) == _structural_snapshot(first)
    assert _structural_snapshot(first) == _structural_snapshot(second)
    assert _mutable_ir_ids(original).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    for original_statement, first_statement, second_statement in zip(
        original,
        first,
        second,
    ):
        assert original_statement == first_statement == second_statement
        assert (
            hash(original_statement) == hash(first_statement) == hash(second_statement)
        )

    original_owner = cast(llir.VarInit, original[0])
    first_owner = cast(llir.VarInit, first[0])
    second_owner = cast(llir.VarInit, second[0])
    original_extent = cast(
        llir.Array,
        cast(llir.FunctionCall, original_owner.value).args[0],
    )
    first_extent = cast(
        llir.Array,
        cast(llir.FunctionCall, first_owner.value).args[0],
    )
    second_extent = cast(
        llir.Array,
        cast(llir.FunctionCall, second_owner.value).args[0],
    )
    original_pointer = cast(llir.VarInit, original[1])
    first_pointer = cast(llir.VarInit, first[1])
    second_pointer = cast(llir.VarInit, second[1])
    original_receiver = cast(
        llir.Var, cast(llir.MemberCall, original_pointer.value).base
    )
    first_receiver = cast(llir.Var, cast(llir.MemberCall, first_pointer.value).base)
    second_receiver = cast(llir.Var, cast(llir.MemberCall, second_pointer.value).base)

    cast(llir.Var, first_extent.values[0]).name = "owned_extent"
    first_receiver.name = "owned_receiver"
    assert cast(llir.Var, original_extent.values[0]).name == "_known_nnz"
    assert cast(llir.Var, second_extent.values[0]).name == "_known_nnz"
    assert original_receiver.name == "Result0_crd_torch"
    assert second_receiver.name == "Result0_crd_torch"

    class ReplaceCoordinateVars(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            rewritten.name = {
                "Result0_crd_torch": "Replacement0_crd_torch",
                "Result0_crd": "Replacement0_crd",
                "_known_nnz": "replacement_nnz",
            }.get(node.name, node.name)
            return rewritten

    replacement = cast(
        List[llir.Stmt],
        ReplaceCoordinateVars(_CONTEXT).rewrite(original),
    )
    replacement_owner = cast(llir.VarInit, replacement[0])
    replacement_extent = cast(
        llir.Array,
        cast(llir.FunctionCall, replacement_owner.value).args[0],
    )
    replacement_pointer = cast(llir.VarInit, replacement[1])
    replacement_receiver = cast(
        llir.Var,
        cast(llir.MemberCall, replacement_pointer.value).base,
    )
    assert replacement_owner.var.name == "Replacement0_crd_torch"
    assert cast(llir.Var, replacement_extent.values[0]).name == "replacement_nnz"
    assert replacement_pointer.var.name == "Replacement0_crd"
    assert replacement_receiver.name == "Replacement0_crd_torch"
    assert _mutable_ir_ids(original).isdisjoint(_mutable_ir_ids(replacement))


def test_member_call_rewrite_is_detached_repeatable_and_replacement_owned() -> None:
    original = llir.MemberCall(
        base=llir.MemberAccess(
            _var("tensor", llir.DataType.TORCH_TENSOR),
            "storage",
        ),
        member="select",
        template_args=(llir.DataType.FLOAT32,),
        args=(
            llir.Add(
                _var("index", llir.DataType.INT64),
                llir.Literal(1, llir.DataType.INT64),
            ),
        ),
    )
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(llir.MemberCall, rewriter.rewrite(original))
    second = cast(llir.MemberCall, rewriter.rewrite(first))

    assert _record(original) == _record(first) == _record(second)
    assert original == first == second
    assert hash(original) == hash(first) == hash(second)
    assert _mutable_ir_ids(original).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    assert type(first.template_args) is tuple
    assert type(first.args) is tuple
    assert first.template_args == (llir.DataType.FLOAT32,)

    original_base = cast(llir.MemberAccess, original.base)
    first_base = cast(llir.MemberAccess, first.base)
    second_base = cast(llir.MemberAccess, second.base)
    assert first_base is not original_base
    assert second_base is not first_base
    assert first_base.base is not original_base.base
    assert second_base.base is not first_base.base
    assert first.args[0] is not original.args[0]
    assert second.args[0] is not first.args[0]

    cast(llir.Var, first_base.base).name = "owned"
    assert cast(llir.Var, original_base.base).name == "tensor"
    assert cast(llir.Var, second_base.base).name == "tensor"

    class ReplaceMemberCallVars(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            if node.name == "tensor":
                rewritten.name = "replacement_tensor"
            elif node.name == "index":
                rewritten.name = "replacement_index"
            return rewritten

    replacement = cast(
        llir.MemberCall,
        ReplaceMemberCallVars(_CONTEXT).rewrite(original),
    )
    replacement_base = cast(llir.MemberAccess, replacement.base)
    replacement_argument = cast(llir.Add, replacement.args[0])
    assert cast(llir.Var, replacement_base.base).name == "replacement_tensor"
    assert cast(llir.Var, replacement_argument.left).name == "replacement_index"
    assert replacement_base is not original_base
    assert replacement_base.base is not original_base.base
    assert replacement_argument is not original.args[0]
    assert cast(llir.Var, original_base.base).name == "tensor"
    assert cast(llir.Var, cast(llir.Add, original.args[0]).left).name == "index"


def test_nested_member_assignment_target_walk_and_rewrite_are_structural() -> None:
    target = llir.MemberAccess(
        llir.MemberAccess(
            _var("Result", llir.DataType.TACO_TENSOR),
            "storage",
        ),
        "value",
    )
    assignment = llir.Assign(
        target,
        _var("Result_values_torch", llir.DataType.TORCH_TENSOR),
    )

    expected_record = [
        "Assign",
        "MemberAccess",
        "MemberAccess",
        "Var:Result",
        "Var:Result_values_torch",
    ]
    assert _record(assignment) == expected_record
    assert _record(assignment) == expected_record

    rewriter = LLIRRewriter(_CONTEXT)
    first = cast(llir.Assign, rewriter.rewrite(assignment))
    second = cast(llir.Assign, rewriter.rewrite(first))
    assert _record(first) == _record(second) == expected_record
    assert _structural_snapshot(assignment) == _structural_snapshot(first)
    assert _structural_snapshot(first) == _structural_snapshot(second)
    assert _mutable_ir_ids(assignment).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))

    first_target = cast(llir.MemberAccess, first.var)
    first_inner = cast(llir.MemberAccess, first_target.base)
    second_target = cast(llir.MemberAccess, second.var)
    second_inner = cast(llir.MemberAccess, second_target.base)
    original_inner = cast(llir.MemberAccess, target.base)
    assert first_target is not target
    assert first_inner is not original_inner
    assert second_target is not first_target
    assert second_inner is not first_inner
    assert first_inner.base is not original_inner.base
    assert second_inner.base is not first_inner.base

    class ReplaceMemberRoot(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            if node.name == "Result":
                rewritten.name = "Replacement"
            return rewritten

    replacement = cast(llir.Assign, ReplaceMemberRoot(_CONTEXT).rewrite(assignment))
    replacement_target = cast(llir.MemberAccess, replacement.var)
    replacement_inner = cast(llir.MemberAccess, replacement_target.base)
    assert cast(llir.Var, replacement_inner.base).name == "Replacement"
    assert replacement_target is not target
    assert replacement_inner is not original_inner
    assert replacement_inner.base is not original_inner.base
    assert cast(llir.Var, original_inner.base).name == "Result"


def test_increment_rewrite_is_detached_repeatable_and_replacement_owned() -> None:
    original = llir.Increment(_var("counter"))
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(llir.Increment, rewriter.rewrite(original))
    second = cast(llir.Increment, rewriter.rewrite(first))

    assert original == first == second
    assert hash(original) == hash(first) == hash(second)
    assert first is not original
    assert second is not first
    assert first.var is not original.var
    assert second.var is not first.var

    first.var.name = "owned"
    assert original.var.name == "counter"
    assert second.var.name == "counter"

    class ReplaceCounter(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            if node.name == "counter":
                rewritten.name = "replacement"
            return rewritten

    replacement = cast(llir.Increment, ReplaceCounter(_CONTEXT).rewrite(original))
    assert replacement.var.name == "replacement"
    assert replacement.var is not original.var
    assert original.var.name == "counter"


def test_function_call_rewrite_is_detached_repeatable_and_replacement_owned() -> None:
    original = llir.FunctionCall(
        "scorch_tensor_from_vector",
        [llir.FunctionCall("std::move", [_var("values")]), _var("dtype")],
    )
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(llir.FunctionCall, rewriter.rewrite(original))
    second = cast(llir.FunctionCall, rewriter.rewrite(first))

    assert _record(original) == _record(first) == _record(second)
    assert _structural_snapshot(original) == _structural_snapshot(first)
    assert _structural_snapshot(first) == _structural_snapshot(second)
    assert _mutable_ir_ids(original).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    assert type(first.args) is tuple
    assert type(second.args) is tuple

    original_move = cast(llir.FunctionCall, original.args[0])
    first_move = cast(llir.FunctionCall, first.args[0])
    second_move = cast(llir.FunctionCall, second.args[0])
    assert original_move is not first_move
    assert first_move is not second_move
    assert original_move.args[0] is not first_move.args[0]
    assert first_move.args[0] is not second_move.args[0]

    cast(llir.Var, first_move.args[0]).name = "owned"
    assert cast(llir.Var, original_move.args[0]).name == "values"
    assert cast(llir.Var, second_move.args[0]).name == "values"

    class ReplaceValue(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            if node.name == "values":
                rewritten.name = "replacement"
            return rewritten

    replacement = cast(llir.FunctionCall, ReplaceValue(_CONTEXT).rewrite(original))
    replacement_move = cast(llir.FunctionCall, replacement.args[0])
    assert cast(llir.Var, replacement_move.args[0]).name == "replacement"
    assert cast(llir.Var, original_move.args[0]).name == "values"
    assert replacement_move.args[0] is not original_move.args[0]


def test_array_rewrite_is_detached_repeatable_and_replacement_owned() -> None:
    original = llir.Array(
        (
            llir.Add(_var("extent"), llir.Literal(1, llir.DataType.INT64)),
            llir.Array((_var("other"),), llir.DataType.INT64),
        ),
        llir.DataType.INT64,
    )
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(llir.Array, rewriter.rewrite(original))
    second = cast(llir.Array, rewriter.rewrite(first))

    assert _record(original) == _record(first) == _record(second)
    assert original == first == second
    assert hash(original) == hash(first) == hash(second)
    assert _mutable_ir_ids(original).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    assert type(first.values) is tuple
    assert type(second.values) is tuple

    class ReplaceExtent(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            if node.name == "extent":
                rewritten.name = "replacement"
            return rewritten

    replacement = cast(llir.Array, ReplaceExtent(_CONTEXT).rewrite(original))
    replacement_add = cast(llir.Add, replacement.values[0])
    original_add = cast(llir.Add, original.values[0])
    first_add = cast(llir.Add, first.values[0])

    assert cast(llir.Var, replacement_add.left).name == "replacement"
    assert cast(llir.Var, original_add.left).name == "extent"
    assert replacement_add is not original_add
    assert replacement_add.left is not original_add.left
    cast(llir.Var, replacement_add.left).name = "owned"
    assert cast(llir.Var, original_add.left).name == "extent"
    assert cast(llir.Var, first_add.left).name == "extent"


def test_cast_rewrite_is_detached_repeatable_and_replacement_owned() -> None:
    original = llir.Cast(
        llir.Add(_var("offset"), llir.Literal(1, llir.DataType.INT)),
        llir.DataType.INT64,
    )
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(llir.Cast, rewriter.rewrite(original))
    second = cast(llir.Cast, rewriter.rewrite(first))

    assert _record(original) == _record(first) == _record(second)
    assert _structural_snapshot(original) == _structural_snapshot(first)
    assert _structural_snapshot(first) == _structural_snapshot(second)
    assert _mutable_ir_ids(original).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    assert first.data_type is llir.DataType.INT64
    assert second.data_type is llir.DataType.INT64

    class ReplaceOffset(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            if node.name == "offset":
                rewritten.name = "replacement"
            return rewritten

    replacement = cast(llir.Cast, ReplaceOffset(_CONTEXT).rewrite(original))
    replacement_add = cast(llir.Add, replacement.expr)
    original_add = cast(llir.Add, original.expr)
    assert cast(llir.Var, replacement_add.left).name == "replacement"
    assert cast(llir.Var, original_add.left).name == "offset"
    assert replacement_add.left is not original_add.left

    cast(llir.Var, replacement_add.left).name = "owned"
    assert cast(llir.Var, original_add.left).name == "offset"


def test_direct_init_rewrite_is_detached_repeatable_and_replacement_owned() -> None:
    original = llir.DirectInit(
        llir.Var("storage", llir.DataType.STD_VECTOR_FLOAT32),
        (
            llir.Mul(
                llir.Cast(
                    _var("rows", llir.DataType.INT64),
                    llir.DataType.SIZE_T,
                ),
                llir.Cast(
                    _var("columns", llir.DataType.CONSTEXPR_INT),
                    llir.DataType.SIZE_T,
                ),
            ),
        ),
    )
    rewriter = LLIRRewriter(_CONTEXT)

    first = cast(llir.DirectInit, rewriter.rewrite(original))
    second = cast(llir.DirectInit, rewriter.rewrite(first))

    assert _record(original) == _record(first) == _record(second)
    assert original == first == second
    assert hash(original) == hash(first) == hash(second)
    assert _mutable_ir_ids(original).isdisjoint(_mutable_ir_ids(first))
    assert _mutable_ir_ids(first).isdisjoint(_mutable_ir_ids(second))
    assert type(first.args) is tuple
    assert type(second.args) is tuple

    class ReplaceRows(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            if node.name == "rows":
                rewritten.name = "replacement_rows"
            return rewritten

    replacement = cast(llir.DirectInit, ReplaceRows(_CONTEXT).rewrite(original))
    replacement_extent = cast(llir.Mul, replacement.args[0])
    replacement_rows = cast(llir.Var, cast(llir.Cast, replacement_extent.left).expr)
    original_extent = cast(llir.Mul, original.args[0])
    original_rows = cast(llir.Var, cast(llir.Cast, original_extent.left).expr)
    first_extent = cast(llir.Mul, first.args[0])
    first_rows = cast(llir.Var, cast(llir.Cast, first_extent.left).expr)

    assert replacement_rows.name == "replacement_rows"
    assert original_rows.name == "rows"
    assert replacement_rows is not original_rows
    replacement_rows.name = "owned"
    assert original_rows.name == "rows"
    assert first_rows.name == "rows"


def test_assignment_target_rewrite_owns_replacement_and_preserves_original() -> None:
    class ReplaceIndexRewriter(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: LLIRPath) -> llir.Var:
            rewritten = super().rewrite_var(node, path)
            if node.name == "old_index":
                rewritten.name = "new_index"
            return rewritten

    metadata = _result_metadata()
    target = llir.ArrayAccess(
        _var("values", llir.DataType.PTR_FLOAT32),
        llir.Add(_var("old_index"), llir.Literal(1)),
        tensor_access=metadata,
    )
    assignment = llir.Assign(target, _var("source", llir.DataType.FLOAT32))

    rewritten = ReplaceIndexRewriter(_CONTEXT).rewrite(assignment)

    assert type(rewritten) is llir.Assign
    rewritten_target = cast(llir.ArrayAccess, rewritten.var)
    rewritten_index = cast(llir.Add, rewritten_target.index)
    assert cast(llir.Var, rewritten_index.left).name == "new_index"
    assert cast(llir.Var, cast(llir.Add, target.index).left).name == "old_index"
    assert rewritten_target.tensor_access is metadata
    assert rewritten_target is not target
    assert rewritten_target.array is not target.array
    assert rewritten_target.index is not target.index
    assert rewritten.value is not assignment.value


def test_list_tuple_nesting_and_optional_children_are_preserved() -> None:
    conditional = llir.IfThenElse(
        cond=None,
        then_body=[],
        else_body=None,
        cond_list=[_var("first"), _var("second")],
        then_body_list=[[llir.Break()], [llir.Continue()]],
    )
    conditional.then_body_list = cast(
        List[List[llir.Stmt]],
        [(llir.Break(),), []],
    )
    root: List[LLIRStatementValue] = [
        [conditional],
        (llir.RawStmt("tail"),),
    ]

    rewritten = LLIRRewriter(_CONTEXT).rewrite(root)
    assert type(rewritten) is list
    assert type(rewritten[0]) is list
    assert type(rewritten[1]) is tuple
    rewritten_conditional = cast(llir.IfThenElse, cast(List[object], rewritten[0])[0])
    assert rewritten_conditional.cond is None
    assert rewritten_conditional.then_body == []
    assert rewritten_conditional.else_body is None
    assert rewritten_conditional.cond_list is not None
    assert len(rewritten_conditional.cond_list) == 2
    assert rewritten_conditional.then_body_list is not None
    assert type(rewritten_conditional.then_body_list[0]) is tuple
    assert rewritten_conditional.then_body_list[1] == []


def test_statement_sequence_member_hook_can_delete_and_expand_statements() -> None:
    class SplicingRewriter(LLIRRewriter):
        def rewrite_statement_sequence_member(
            self, node: llir.Stmt, path: LLIRPath
        ) -> Sequence[llir.Stmt]:
            if type(node) is llir.Comment and cast(llir.Comment, node).value == "drop":
                return []
            if type(node) is llir.Comment:
                return [llir.RawStmt("first"), llir.Break()]
            if type(node) is llir.RawStmt:
                return (llir.Continue(), llir.RawStmt("second"))
            return super().rewrite_statement_sequence_member(node, path)

    passthrough = llir.VarDecl(_var("value"))
    root: List[LLIRStatementValue] = [
        llir.Comment("expand"),
        (
            llir.Comment("drop"),
            [llir.RawStmt("expand")],
        ),
        passthrough,
    ]

    rewritten = SplicingRewriter(_CONTEXT).rewrite(root)

    assert type(rewritten) is list
    assert [type(statement) for statement in rewritten[:2]] == [
        llir.RawStmt,
        llir.Break,
    ]
    assert type(rewritten[2]) is tuple
    nested_tuple = cast(Tuple[LLIRStatementValue, ...], rewritten[2])
    assert type(nested_tuple[0]) is list
    nested_list = cast(List[LLIRStatementValue], nested_tuple[0])
    assert [type(statement) for statement in nested_list] == [
        llir.Continue,
        llir.RawStmt,
    ]
    assert type(rewritten[3]) is llir.VarDecl
    assert rewritten[3] is not passthrough
    assert cast(llir.VarDecl, rewritten[3]).var is not passthrough.var

    tuple_root = (llir.Comment("expand"), llir.Comment("drop"))
    rewritten_tuple = SplicingRewriter(_CONTEXT).rewrite(tuple_root)
    assert type(rewritten_tuple) is tuple
    assert [type(statement) for statement in rewritten_tuple] == [
        llir.RawStmt,
        llir.Break,
    ]


def test_statement_sequence_member_hook_requires_exact_list_or_tuple() -> None:
    class InvalidContainerRewriter(LLIRRewriter):
        def rewrite_statement_sequence_member(
            self, node: llir.Stmt, path: LLIRPath
        ) -> Sequence[llir.Stmt]:
            return cast(Sequence[llir.Stmt], iter((node,)))

    with pytest.raises(LLIRTraversalError) as raised:
        InvalidContainerRewriter(_CONTEXT).rewrite([llir.Break()])

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_statement_rewrite_sequence"
    assert diagnostic.path == ("root", "[0]")
    assert diagnostic.node_type == "tuple_iterator"


def test_statement_sequence_member_hook_rejects_non_statement_members() -> None:
    class InvalidMemberRewriter(LLIRRewriter):
        def rewrite_statement_sequence_member(
            self, node: llir.Stmt, path: LLIRPath
        ) -> Sequence[llir.Stmt]:
            return cast(Sequence[llir.Stmt], [llir.Literal(1)])

    with pytest.raises(LLIRTraversalError) as raised:
        InvalidMemberRewriter(_CONTEXT).rewrite([llir.Break()])

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_statement_rewrite_member"
    assert diagnostic.path == ("root", "[0]")
    assert diagnostic.node_type == "Literal"


def test_statement_sequence_member_hook_validates_replacement_children() -> None:
    class MalformedReplacementRewriter(LLIRRewriter):
        def rewrite_statement_sequence_member(
            self, node: llir.Stmt, path: LLIRPath
        ) -> Sequence[llir.Stmt]:
            declaration = llir.VarDecl(_var("value"))
            declaration.var = cast(llir.Var, llir.Literal(1))
            return (declaration,)

    with pytest.raises(LLIRTraversalError) as raised:
        MalformedReplacementRewriter(_CONTEXT).rewrite([llir.Break()])

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_var_child"
    assert diagnostic.path == ("root", "[0]", "var")
    assert diagnostic.node_type == "Literal"


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("node", "field"),
    [
        (
            llir.Function(llir.DataType.VOID, "function", [], []),
            "body",
        ),
        (
            llir.ForLoop(None, _var("cond"), llir.Increment(_var("i")), []),
            "body",
        ),
        (llir.ForLoopAuto(_var("i"), _var("array"), []), "body"),
        (llir.WhileLoop(_var("cond"), []), "body"),
    ],
)
def test_required_statement_children_reject_none(
    operation: str,
    node: llir.Stmt,
    field: str,
) -> None:
    setattr(node, field, None)

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(node)
        else:
            LLIRRewriter(_CONTEXT).rewrite(node)

    assert raised.value.diagnostic.code == "invalid_statement_sequence"
    assert raised.value.diagnostic.path == ("root", field)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
@pytest.mark.parametrize(
    ("field", "invalid_value", "diagnostic_code"),
    [
        ("init", llir.Break(), "invalid_for_loop_init"),
        (
            "update",
            llir.VarInit(_var("j"), llir.Literal(1)),
            "invalid_for_loop_update",
        ),
        ("update", [llir.Increment(_var("i"))], "invalid_for_loop_update"),
    ],
)
def test_for_loop_header_children_are_scalar_and_typed(
    operation: str,
    field: str,
    invalid_value: object,
    diagnostic_code: str,
) -> None:
    loop = llir.ForLoop(
        init=None,
        cond=_var("cond"),
        update=llir.Increment(_var("i")),
        body=[],
    )
    setattr(loop, field, invalid_value)

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(loop)
        else:
            LLIRRewriter(_CONTEXT).rewrite(loop)

    assert raised.value.diagnostic.code == diagnostic_code
    assert raised.value.diagnostic.path == ("root", field)


@pytest.mark.parametrize("operation", ["walk", "rewrite"])
def test_typed_var_children_reject_other_supported_expressions(
    operation: str,
) -> None:
    declaration = llir.VarDecl(_var("value"))
    declaration.var = cast(llir.Var, llir.Literal(1))

    with pytest.raises(LLIRTraversalError) as raised:
        if operation == "walk":
            LLIRWalker(_CONTEXT).walk(declaration)
        else:
            LLIRRewriter(_CONTEXT).rewrite(declaration)

    assert raised.value.diagnostic.code == "invalid_var_child"
    assert raised.value.diagnostic.node_type == "Literal"
    assert raised.value.diagnostic.path == ("root", "var")


def test_function_call_default_arguments_are_empty_immutable_tuples() -> None:
    first = llir.FunctionCall("first")
    second = llir.FunctionCall("second")

    assert type(first.args) is tuple
    assert type(second.args) is tuple
    assert first.args == second.args == ()


def test_function_call_stmt_default_arguments_are_empty_immutable_tuples() -> None:
    first = llir.FunctionCallStmt("first")
    second = llir.FunctionCallStmt("second")

    assert type(first.args) is tuple
    assert type(second.args) is tuple
    assert first.args == second.args == ()


def test_member_call_stmt_default_arguments_are_empty_immutable_tuples() -> None:
    first = llir.MemberCallStmt(_var("first"), "clear")
    second = llir.MemberCallStmt(_var("second"), "clear")

    assert type(first.template_args) is tuple
    assert type(second.template_args) is tuple
    assert first.template_args == second.template_args == ()
    assert type(first.args) is tuple
    assert type(second.args) is tuple
    assert first.args == second.args == ()
