from typing import List, cast

import pytest

from scorch.compiler import llir
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.dynamic_vector_access_pass import (
    DYNAMIC_VECTOR_ACCESS_CONTEXT,
    rewrite_dynamic_vector_accesses,
)
from scorch.compiler.llir_traversal import (
    LLIRStatementValue,
    LLIRTraversalError,
)


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _access(
    array: str,
    index: str | llir.Expr,
    data_type: llir.DataType = llir.DataType.NO_TYPE,
) -> llir.ArrayAccess:
    index_expr = _var(index) if type(index) is str else index
    return llir.ArrayAccess(array=_var(array, data_type), index=index_expr)


def _legacy_dynamic_vector_fixture() -> List[llir.Stmt]:
    coordinate_store = llir.Assign(
        var=_access("out_crd", "p", llir.DataType.STD_VECTOR_C_INT),
        value=_var("out_pos[q]"),
    )
    return [
        llir.VarDecl(_var("out_pos", llir.DataType.STD_VECTOR_C_INT)),
        llir.VarDecl(_var("out_crd", llir.DataType.STD_VECTOR_C_INT)),
        llir.VarDecl(_var("out_values", llir.DataType.STD_VECTOR_FLOAT32)),
        llir.VarInit(
            _var("scratch", llir.DataType.STD_VECTOR_FLOAT32),
            _var("std::vector<float>(4)"),
        ),
        coordinate_store,
        llir.Assign(
            var=_access("out_crd", "p", llir.DataType.STD_VECTOR_C_INT),
            value=_var("out_pos[q]"),
        ),
        llir.Assign(
            var=_access("out_values", "p", llir.DataType.STD_VECTOR_FLOAT32),
            value=_var("out_crd[q]"),
        ),
        llir.Assign(
            var=_access(
                "out_pos",
                llir.Add(_var("p"), llir.Literal(1)),
                llir.DataType.STD_VECTOR_C_INT,
            ),
            value=_var("out_values[q]"),
        ),
        llir.VarInit(_var("read", llir.DataType.INT), _var("out_pos[p]")),
        llir.Assign(
            var=_access("scratch", "i", llir.DataType.STD_VECTOR_FLOAT32),
            value=_var("out_values[q]"),
        ),
    ]


def _array_access_parts(access: llir.AssignmentTarget) -> tuple[str, str]:
    assert type(access) is llir.ArrayAccess
    assert type(access.array) is llir.Var
    assert type(access.index) is llir.Var
    return cast(llir.Var, access.array).name, cast(llir.Var, access.index).name


def _cpp(statements: List[llir.Stmt]) -> str:
    return LLIRLowerer().lower_llir(statements)


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


def test_dynamic_vector_pass_matches_legacy_transformation_structurally() -> None:
    source = _legacy_dynamic_vector_fixture()
    source_cpp = _cpp(source)

    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    assert _cpp(source) == source_cpp
    assert rewritten is not source
    assert len(source) == 10
    assert len(rewritten) == 9
    assert [type(statement) for statement in rewritten] == [
        llir.VarDecl,
        llir.VarDecl,
        llir.VarDecl,
        llir.VarInit,
        llir.FunctionCallStmt,
        llir.FunctionCallStmt,
        llir.FunctionCallStmt,
        llir.VarInit,
        llir.Assign,
    ]

    coordinate_append = cast(llir.FunctionCallStmt, rewritten[4])
    assert coordinate_append.name == "out_crd.emplace_back"
    assert cast(llir.Var, coordinate_append.args[0]).name == "out_pos.at(q)"

    value_append = cast(llir.FunctionCallStmt, rewritten[5])
    assert value_append.name == "out_values.emplace_back"
    assert cast(llir.Var, value_append.args[0]).name == "out_crd.at(q)"

    position_store = cast(llir.FunctionCallStmt, rewritten[6])
    assert position_store.name == "scorch_vector_set"
    assert cast(llir.Var, position_store.args[0]).name == "out_pos"
    position = cast(llir.Add, position_store.args[1])
    assert cast(llir.Var, position.left).name == "p"
    assert cast(llir.Literal, position.right).value == 1
    assert cast(llir.Var, position_store.args[2]).name == "out_values.at(q)"

    pre_sized_store = cast(llir.Assign, rewritten[8])
    assert _array_access_parts(pre_sized_store.var) == ("scratch", "i")
    assert cast(llir.Var, pre_sized_store.value).name == "out_values.at(q)"

    expected_cpp = """std::vector<int> out_pos;
std::vector<int> out_crd;
std::vector<float> out_values;
std::vector<float> scratch = std::vector<float>(4);
out_crd.emplace_back(out_pos.at(q));
out_values.emplace_back(out_crd.at(q));
scorch_vector_set(out_pos, p + 1, out_values.at(q));
int read = out_pos.at(p);
scratch[i] = out_values.at(q);"""
    assert _cpp(rewritten) == expected_cpp


def test_dynamic_vector_pass_does_not_mutate_or_alias_caller_input() -> None:
    source = _legacy_dynamic_vector_fixture()
    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    rewritten_decl = cast(llir.VarDecl, rewritten[0])
    rewritten_decl.var.name = "changed"
    rewritten_call = cast(llir.FunctionCallStmt, rewritten[4])
    cast(llir.Var, rewritten_call.args[0]).name = "changed_read"
    rewritten.append(llir.Break())

    assert cast(llir.VarDecl, source[0]).var.name == "out_pos"
    assert _array_access_parts(cast(llir.Assign, source[4]).var) == (
        "out_crd",
        "p",
    )
    assert cast(llir.Assign, source[4]).value.name == "out_pos[q]"
    assert len(source) == 10


def test_dynamic_vector_pass_is_idempotent_for_generated_access_shapes() -> None:
    source = _legacy_dynamic_vector_fixture()
    once = rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
    twice = rewrite_dynamic_vector_accesses(once, DYNAMIC_VECTOR_ACCESS_CONTEXT)

    assert _cpp(once) == _cpp(twice)
    assert [type(statement) for statement in once] == [
        type(statement) for statement in twice
    ]
    assert _structural_snapshot(once) == _structural_snapshot(twice)


def test_dynamic_vector_pass_preserves_compound_store_and_loop_update_shape() -> None:
    loop = llir.ForLoop(
        init=llir.VarInit(_var("i", llir.DataType.INT), llir.Literal(0)),
        cond=llir.BinOp("<", _var("i"), _var("n")),
        update=llir.Assign(
            _access("out_pos", "i", llir.DataType.STD_VECTOR_C_INT),
            _var("out_values[i]"),
            op=llir.AssignOp.ADD_ASSIGN,
        ),
        body=[],
    )
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_pos", llir.DataType.STD_VECTOR_C_INT)),
        llir.VarDecl(_var("out_values", llir.DataType.STD_VECTOR_FLOAT32)),
        loop,
    ]

    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )
    rewritten_loop = cast(llir.ForLoop, rewritten[2])
    update = cast(llir.Assign, rewritten_loop.update)

    assert type(update) is llir.Assign
    assert update.op == llir.AssignOp.ADD_ASSIGN
    assert _array_access_parts(update.var) == ("out_pos", "i")
    assert cast(llir.Var, update.value).name == "out_values.at(i)"


def test_dynamic_vector_pass_no_vector_declaration_is_detached_no_op() -> None:
    source: List[llir.Stmt] = [
        llir.Assign(
            _access("scratch", "i", llir.DataType.PTR_FLOAT32),
            _var("value"),
        )
    ]
    snapshot = _structural_snapshot(source)

    rewritten = rewrite_dynamic_vector_accesses(
        source,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    assert rewritten is not source
    assert _structural_snapshot(rewritten) == snapshot
    rewritten_store = cast(llir.Assign, rewritten[0])
    rewritten_access = cast(llir.ArrayAccess, rewritten_store.var)
    cast(llir.Var, rewritten_access.array).name = "changed"
    assert _array_access_parts(cast(llir.Assign, source[0]).var) == (
        "scratch",
        "i",
    )


def test_dynamic_vector_pass_detaches_and_preserves_shape_extent_reads() -> None:
    source: List[llir.Stmt] = [
        llir.VarDecl(_var("out_values", llir.DataType.STD_VECTOR_FLOAT32)),
        llir.VarInit(
            _var("Input0_size", llir.DataType.INT64),
            llir.ArrayAccess(
                array=_var("Input_shape", llir.DataType.STD_VECTOR_INT),
                index=llir.Literal(0, data_type=llir.DataType.INT64),
            ),
        ),
    ]
    source_snapshot = _structural_snapshot(source)

    once = rewrite_dynamic_vector_accesses(source, DYNAMIC_VECTOR_ACCESS_CONTEXT)
    twice = rewrite_dynamic_vector_accesses(once, DYNAMIC_VECTOR_ACCESS_CONTEXT)

    assert _structural_snapshot(source) == source_snapshot
    assert _structural_snapshot(once) == _structural_snapshot(twice)
    assert _cpp(once) == _cpp(twice)
    assert _cpp(once) == (
        "std::vector<float> out_values;\n"
        "int64_t Input0_size = Input_shape[0];"
    )
    first_access = cast(llir.ArrayAccess, cast(llir.VarInit, once[1]).value)
    repeated_access = cast(llir.ArrayAccess, cast(llir.VarInit, twice[1]).value)
    source_access = cast(llir.ArrayAccess, cast(llir.VarInit, source[1]).value)
    assert first_access is not source_access
    assert repeated_access is not first_access
    assert first_access.array is not source_access.array
    assert first_access.index is not source_access.index
    cast(llir.Var, first_access.array).name = "changed"
    assert cast(llir.Var, source_access.array).name == "Input_shape"
    assert cast(llir.Var, repeated_access.array).name == "Input_shape"


def test_pass_preserves_nested_list_and_tuple_statement_containers() -> None:
    nested: List[LLIRStatementValue] = [
        llir.VarDecl(_var("values", llir.DataType.STD_VECTOR_FLOAT32)),
        ([llir.VarInit(_var("read"), _var("values[i]"))],),
    ]

    rewritten = rewrite_dynamic_vector_accesses(
        nested,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )
    assert type(rewritten) is list
    assert type(rewritten[1]) is tuple
    tuple_body = cast(tuple, rewritten[1])
    assert type(tuple_body[0]) is list
    initializer = cast(llir.VarInit, cast(list, tuple_body[0])[0])
    assert cast(llir.Var, initializer.value).name == "values.at(i)"


def test_pass_accepts_a_scalar_function_root() -> None:
    function = llir.Function(
        return_type=llir.DataType.VOID,
        name="function",
        args=[],
        body=[
            llir.VarDecl(_var("values", llir.DataType.STD_VECTOR_FLOAT32)),
            llir.VarInit(_var("read"), _var("values[i]")),
        ],
    )

    rewritten = rewrite_dynamic_vector_accesses(
        function,
        DYNAMIC_VECTOR_ACCESS_CONTEXT,
    )

    assert type(rewritten) is llir.Function
    assert rewritten is not function
    initializer = cast(llir.VarInit, rewritten.body[1])
    assert cast(llir.Var, initializer.value).name == "values.at(i)"


def test_pass_unknown_node_reports_its_own_stage_and_name() -> None:
    class UnknownBreak(llir.Break):
        pass

    with pytest.raises(LLIRTraversalError) as raised:
        rewrite_dynamic_vector_accesses(
            UnknownBreak(),
            DYNAMIC_VECTOR_ACCESS_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.stage == "LLIR rewrite"
    assert diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert diagnostic.node_type == "UnknownBreak"


def test_malformed_vector_declaration_fails_through_structured_diagnostic() -> None:
    class UnknownExpr(llir.Expr):
        pass

    declaration = llir.VarDecl(_var("values", llir.DataType.STD_VECTOR_FLOAT32))
    declaration.var = cast(llir.Var, UnknownExpr())

    with pytest.raises(LLIRTraversalError) as raised:
        rewrite_dynamic_vector_accesses(
            [declaration],
            DYNAMIC_VECTOR_ACCESS_CONTEXT,
        )

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.stage == "LLIR rewrite"
    assert diagnostic.pass_name == "rewrite_dynamic_vector_accesses"
    assert diagnostic.node_type == "UnknownExpr"
    assert diagnostic.path == ("root", "[0]", "var")
