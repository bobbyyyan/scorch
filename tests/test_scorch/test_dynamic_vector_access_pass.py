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


def _legacy_dynamic_vector_fixture() -> List[llir.Stmt]:
    coordinate_store = llir.Assign(
        var=_var("out_crd[p]"),
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
            var=_var("out_crd[p]"),
            value=_var("out_pos[q]"),
        ),
        llir.Assign(
            var=_var("out_values[p]"),
            value=_var("out_crd[q]"),
        ),
        llir.Assign(
            var=_var("out_pos[p + 1]"),
            value=_var("out_values[q]"),
        ),
        llir.VarInit(_var("read", llir.DataType.INT), _var("out_pos[p]")),
        llir.Assign(
            var=_var("scratch[i]"),
            value=_var("out_values[q]"),
        ),
    ]


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
    assert [cast(llir.Var, arg).name for arg in position_store.args] == [
        "out_pos",
        "p + 1",
        "out_values.at(q)",
    ]

    pre_sized_store = cast(llir.Assign, rewritten[8])
    assert cast(llir.Var, pre_sized_store.var).name == "scratch[i]"
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
    assert cast(llir.Assign, source[4]).var.name == "out_crd[p]"
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
            _var("out_pos[i]"),
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
    assert cast(llir.Var, update.var).name == "out_pos.at(i)"
    assert cast(llir.Var, update.value).name == "out_values.at(i)"


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
