"""Focused boundary tests for the single-line guarded-call statement node."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Set, cast, get_type_hints

import pytest

from scorch.compiler import llir
from scorch.compiler.codegen import CodegenError, LLIRLowerer
from scorch.compiler.llir_traversal import (
    LLIRRewriter,
    LLIRTraversalContext,
    LLIRTraversalError,
    LLIRWalker,
)

_CONTEXT = LLIRTraversalContext(stage="LLIR test", pass_name="guarded_call")


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _int_literal(value: int) -> llir.Literal:
    return llir.Literal(value, llir.DataType.INT)


def _next_coordinate(
    coordinate_array: str = "A1_crd",
    iterator: str = "pA1",
) -> llir.ArrayAccess:
    return llir.ArrayAccess(
        array=_var(coordinate_array),
        index=llir.Add(_var(iterator), _int_literal(1)),
    )


def _prefetch_call(
    value_array: str = "B_val",
    stride: str = "B1_size",
) -> llir.FunctionCallStmt:
    return llir.FunctionCallStmt(
        "__builtin_prefetch",
        [
            llir.AddressOf(
                operand=llir.ArrayAccess(
                    array=_var(value_array),
                    index=llir.Mul(_next_coordinate(), _var(stride)),
                )
            ),
            _int_literal(0),
            _int_literal(1),
        ],
    )


def _single_condition() -> llir.BinOp:
    return llir.BinOp(
        "<",
        llir.Add(_var("pA1"), _int_literal(1)),
        _var("pA1_end"),
    )


def _three_conjunct_condition() -> llir.BinOp:
    return llir.BinOp(
        "&&",
        llir.BinOp(
            "&&",
            _single_condition(),
            llir.BinOp(">=", _next_coordinate(), _var("j_out")),
        ),
        llir.BinOp("<", _next_coordinate(), _var("j_out_end")),
    )


def _guarded_prefetch() -> llir.GuardedCallStmt:
    return llir.GuardedCallStmt(cond=_single_condition(), call=_prefetch_call())


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


def test_guarded_call_is_frozen_typed_and_structurally_equal() -> None:
    statement = _guarded_prefetch()
    equal = _guarded_prefetch()

    assert statement == equal
    assert statement != llir.GuardedCallStmt(
        cond=_single_condition(),
        call=_prefetch_call(value_array="C_val"),
    )
    assert get_type_hints(llir.GuardedCallStmt) == {
        "cond": llir.Expr,
        "call": llir.FunctionCallStmt,
    }
    with pytest.raises(FrozenInstanceError):
        statement.cond = _single_condition()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        statement.call = _prefetch_call()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("cond", "message"),
    (
        (
            _var("flag"),
            "GuardedCallStmt.cond must be an exact comparison or '&&' conjunction",
        ),
        (
            llir.Add(_var("pA1"), _int_literal(1)),
            "GuardedCallStmt.cond must be an exact comparison or '&&' conjunction",
        ),
        (
            llir.BinOp("||", _single_condition(), _single_condition()),
            "GuardedCallStmt.cond comparisons must use one supported",
        ),
        (
            llir.BinOp("+", _var("pA1"), _var("pA1_end")),
            "GuardedCallStmt.cond comparisons must use one supported",
        ),
        (
            llir.BinOp("&&", _var("flag"), _single_condition()),
            "GuardedCallStmt.cond must be an exact comparison or '&&' conjunction",
        ),
        (
            llir.BinOp(
                "<",
                llir.Select(_var("a"), _var("b"), _var("c")),
                _var("pA1_end"),
            ),
            "GuardedCallStmt.cond comparison operand contains an unsupported",
        ),
        (
            llir.BinOp(
                "<",
                _var("weird name"),
                _var("pA1_end"),
            ),
            "GuardedCallStmt.cond comparison operand Var name must be an "
            "identifier or member path",
        ),
    ),
)
def test_construction_rejects_conditions_outside_the_guard_grammar(
    cond: llir.Expr,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        llir.GuardedCallStmt(cond=cond, call=_prefetch_call())


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (
            llir.FunctionCall("__builtin_prefetch"),
            "GuardedCallStmt.call must be an exact FunctionCallStmt",
        ),
        (
            llir.RawStmt("__builtin_prefetch(&B_val[0], 0, 1)"),
            "GuardedCallStmt.call must be an exact FunctionCallStmt",
        ),
        (
            llir.FunctionCallStmt("not an identifier()"),
            "GuardedCallStmt.call name must be an identifier or member path",
        ),
        (
            llir.FunctionCallStmt(
                "__builtin_prefetch",
                [llir.Select(_var("a"), _var("b"), _var("c"))],
            ),
            "GuardedCallStmt.call argument contains an unsupported",
        ),
    ),
)
def test_construction_rejects_callees_outside_the_guard_grammar(
    call: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        llir.GuardedCallStmt(
            cond=_single_condition(),
            call=cast(llir.FunctionCallStmt, call),
        )


def test_construction_accepts_the_packed_relayout_forms() -> None:
    for origin in (None, "j_out"):
        staged: llir.Expr = _next_coordinate()
        if origin is not None:
            staged = llir.BinOp("-", staged, _var(origin))
        statement = llir.GuardedCallStmt(
            cond=_three_conjunct_condition(),
            call=llir.FunctionCallStmt(
                "__builtin_prefetch",
                [
                    llir.AddressOf(
                        operand=llir.ArrayAccess(
                            array=_var("packed_B"),
                            index=llir.Mul(staged, _var("kTile_k")),
                        )
                    ),
                    _int_literal(0),
                    _int_literal(1),
                ],
            ),
        )
        assert type(statement.cond) is llir.BinOp
        assert statement.call.name == "__builtin_prefetch"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("cond", None, "GuardedCallStmt.cond must be an exact comparison"),
        ("cond", "pA1 + 1 < pA1_end", "GuardedCallStmt.cond must be an exact"),
        ("call", None, "GuardedCallStmt.call must be an exact FunctionCallStmt"),
    ),
)
def test_forged_fields_fail_walker_rewriter_and_codegen(
    field: str,
    value: object,
    message: str,
) -> None:
    statement = _guarded_prefetch()
    object.__setattr__(statement, field, value)

    with pytest.raises(LLIRTraversalError) as walk_error:
        LLIRWalker(_CONTEXT).walk(statement)
    assert walk_error.value.diagnostic.code == "invalid_guarded_call_stmt"
    assert message.split(" must")[0] in walk_error.value.diagnostic.message

    with pytest.raises(LLIRTraversalError) as rewrite_error:
        LLIRRewriter(_CONTEXT).rewrite(statement)
    assert rewrite_error.value.diagnostic.code == "invalid_guarded_call_stmt"

    with pytest.raises(CodegenError, match=message.split(" must")[0]):
        LLIRLowerer().lower_llir(statement)


def test_forged_missing_fields_fail_closed() -> None:
    statement = _guarded_prefetch()
    del statement.__dict__["call"]

    with pytest.raises(LLIRTraversalError) as walk_error:
        LLIRWalker(_CONTEXT).walk(statement)
    assert walk_error.value.diagnostic.code == "invalid_guarded_call_stmt"
    with pytest.raises(CodegenError, match="GuardedCallStmt.call"):
        LLIRLowerer().lower_llir(statement)


def test_forged_cyclic_condition_fails_with_a_controlled_boundary_error() -> None:
    statement = _guarded_prefetch()
    conjunction = llir.BinOp("&&", _single_condition(), _single_condition())
    object.__setattr__(conjunction, "right", conjunction)
    object.__setattr__(statement, "cond", conjunction)

    with pytest.raises(LLIRTraversalError) as walk_error:
        LLIRWalker(_CONTEXT).walk(statement)
    assert "acyclic" in walk_error.value.diagnostic.message
    with pytest.raises(CodegenError, match="acyclic"):
        LLIRLowerer().lower_llir(statement)


@pytest.mark.parametrize(
    "operator",
    (
        type("StringSubclass", (str,), {})("&&"),
        type(
            "ExplosiveEquality",
            (),
            {"__eq__": lambda self, other: (_ for _ in ()).throw(RuntimeError())},
        )(),
    ),
)
def test_forged_condition_operator_requires_an_exact_string(operator: object) -> None:
    statement = _guarded_prefetch()
    condition = _single_condition()
    object.__setattr__(condition, "op", operator)
    object.__setattr__(statement, "cond", condition)

    with pytest.raises(TypeError, match="operator must be an exact string"):
        llir._validate_guarded_call_fields(statement)
    with pytest.raises(LLIRTraversalError) as walk_error:
        LLIRWalker(_CONTEXT).walk(statement)
    assert walk_error.value.diagnostic.code == "invalid_guarded_call_stmt"
    assert walk_error.value.diagnostic.path == ("root", "cond", "op")
    with pytest.raises(CodegenError, match="exact string|string subclass"):
        LLIRLowerer().lower_llir(statement)


def test_deep_guard_condition_fails_without_recursion_error() -> None:
    condition = _single_condition()
    for _ in range(300):
        condition = llir.BinOp("&&", condition, _single_condition())
    statement = object.__new__(llir.GuardedCallStmt)
    object.__setattr__(statement, "cond", condition)
    object.__setattr__(statement, "call", _prefetch_call())

    with pytest.raises(TypeError, match="maximum supported nesting depth"):
        llir._validate_guarded_call_fields(statement)
    with pytest.raises(LLIRTraversalError) as walk_error:
        LLIRWalker(_CONTEXT).walk(statement)
    assert walk_error.value.diagnostic.code == "invalid_guarded_call_stmt"
    assert "maximum supported nesting depth" in walk_error.value.diagnostic.message
    with pytest.raises(CodegenError, match="maximum supported nesting depth"):
        LLIRLowerer().lower_llir(statement)


def test_rewriter_reports_a_guard_child_that_becomes_unrepresentable() -> None:
    class _InvalidatingRewriter(LLIRRewriter):
        def rewrite_var(self, node: llir.Var, path: tuple[str, ...]) -> llir.Var:
            if node.name == "pA1" and path[:2] == ("root", "cond"):
                return cast(
                    llir.Var,
                    llir.Select(_var("flag"), _var("yes"), _var("no")),
                )
            return super().rewrite_var(node, path)

    with pytest.raises(LLIRTraversalError) as raised:
        _InvalidatingRewriter(_CONTEXT).rewrite(_guarded_prefetch())

    assert raised.value.diagnostic.code == "invalid_guarded_call_stmt"
    assert raised.value.diagnostic.path == ("root", "cond", "left", "left")
    assert "unsupported LLIR expression" in raised.value.diagnostic.message


def test_rewriter_detaches_every_mutable_child() -> None:
    statement = _guarded_prefetch()
    rewritten = LLIRRewriter(_CONTEXT).rewrite(statement)

    assert type(rewritten) is llir.GuardedCallStmt
    assert rewritten == statement
    assert _mutable_ir_ids(rewritten).isdisjoint(_mutable_ir_ids(statement))


def test_walker_enters_condition_and_call_children_in_order() -> None:
    statement = _guarded_prefetch()
    visited: list[tuple[str, tuple[str, ...]]] = []

    class _Recorder(LLIRWalker):
        def enter_node(self, node: llir.Node, path: tuple[str, ...]) -> None:
            visited.append((type(node).__name__, path))

    _Recorder(_CONTEXT).walk(statement)

    assert visited[0] == ("GuardedCallStmt", ("root",))
    condition_indices = [
        index for index, (_, path) in enumerate(visited) if path[1:2] == ("cond",)
    ]
    call_indices = [
        index for index, (_, path) in enumerate(visited) if path[1:2] == ("call",)
    ]
    assert condition_indices and call_indices
    assert max(condition_indices) < min(call_indices)
    assert ("FunctionCallStmt", ("root", "call")) in visited


def test_unknown_subclass_fails_closed_in_walker_rewriter_and_codegen() -> None:
    statement = _guarded_prefetch()

    class UnknownGuardedCallStmt(llir.GuardedCallStmt):
        pass

    unknown = object.__new__(UnknownGuardedCallStmt)
    vars(unknown).update(vars(statement))

    with pytest.raises(LLIRTraversalError) as walk_error:
        LLIRWalker(_CONTEXT).walk(unknown)
    assert walk_error.value.diagnostic.code == "unknown_llir_node"
    with pytest.raises(LLIRTraversalError):
        LLIRRewriter(_CONTEXT).rewrite(unknown)
    with pytest.raises(CodegenError, match="UnknownGuardedCallStmt"):
        LLIRLowerer().lower_llir(unknown)


def test_codegen_is_byte_exact_for_the_legacy_prefetch_spellings() -> None:
    lowerer = LLIRLowerer()

    assert lowerer.lower_llir(_guarded_prefetch(), 3) == (
        "      if (pA1 + 1 < pA1_end) "
        "__builtin_prefetch(&B_val[A1_crd[pA1 + 1] * B1_size], 0, 1);"
    )

    for origin, staged_text in (
        (None, "A1_crd[pA1 + 1]"),
        ("j_out", "(A1_crd[pA1 + 1] - j_out)"),
    ):
        staged: llir.Expr = _next_coordinate()
        if origin is not None:
            staged = llir.BinOp("-", staged, _var(origin))
        packed = llir.GuardedCallStmt(
            cond=_three_conjunct_condition(),
            call=llir.FunctionCallStmt(
                "__builtin_prefetch",
                [
                    llir.AddressOf(
                        operand=llir.ArrayAccess(
                            array=_var("packed_B"),
                            index=llir.Mul(staged, _var("kTile_k")),
                        )
                    ),
                    _int_literal(0),
                    _int_literal(1),
                ],
            ),
        )
        assert lowerer.lower_llir(packed) == (
            "if (pA1 + 1 < pA1_end && A1_crd[pA1 + 1] >= j_out && "
            "A1_crd[pA1 + 1] < j_out_end) "
            f"__builtin_prefetch(&packed_B[{staged_text} * kTile_k], 0, 1);"
        )


def test_codegen_emits_one_semicolon_terminated_line_inside_loop_bodies() -> None:
    loop = llir.ForLoop(
        init=llir.VarInit(_var("pA1", llir.DataType.INT), _int_literal(0)),
        cond=llir.BinOp("<", _var("pA1"), _var("pA1_end")),
        update=llir.Increment(_var("pA1")),
        body=[_guarded_prefetch()],
    )
    rendered = LLIRLowerer().lower_llir(loop)
    lines = rendered.splitlines()

    guard_lines = [line for line in lines if "__builtin_prefetch" in line]
    assert len(guard_lines) == 1
    assert guard_lines[0].startswith("  if (")
    assert guard_lines[0].endswith(", 0, 1);")
    assert "{" not in guard_lines[0]
    assert "}" not in guard_lines[0]
