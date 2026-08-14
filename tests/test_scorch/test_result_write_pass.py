from dataclasses import FrozenInstanceError
from typing import List, Literal, Set, Tuple, cast

import pytest

from scorch.compiler import llir  # type: ignore[import-untyped]
from scorch.compiler.codegen import LLIRLowerer  # type: ignore[import-untyped]
from scorch.compiler.identity import (  # type: ignore[import-untyped]
    AccessId,
    IndexId,
    SymbolId,
)
from scorch.compiler.llir_traversal import (  # type: ignore[import-untyped]
    LLIRStatementValue,
    LLIRTraversalContext,
    LLIRTraversalError,
)
from scorch.compiler.result_write_pass import (  # type: ignore[import-untyped]
    RESULT_WRITE_TRAVERSAL_CONTEXT,
    ResultWriteContext,
    rewrite_result_writes,
)

_Mode = Literal["count", "fill"]

_RESULT_ID = SymbolId(700)
_RESULT_INDEX_ID = IndexId(701)


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _access(
    array_name: str,
    index: llir.Expr,
    array_type: llir.DataType = llir.DataType.NO_TYPE,
    *,
    tensor_access: llir.TensorAccessMetadata | None = None,
) -> llir.ArrayAccess:
    return llir.ArrayAccess(
        array=_var(array_name, array_type),
        index=index,
        tensor_access=tensor_access,
    )


def _result_value_access(
    index: llir.Expr,
    *,
    tensor_id: SymbolId = _RESULT_ID,
) -> llir.ArrayAccess:
    return _access(
        "Result_values",
        index,
        tensor_access=llir.TensorAccessMetadata(
            access_id=AccessId(702),
            tensor_id=tensor_id,
            index_ids=(_RESULT_INDEX_ID,),
            role=llir.TensorAccessRole.RESULT_WRITE,
        ),
    )


def _phase_index(level: int) -> llir.Add:
    return llir.Add(
        _var(f"_base{level}", llir.DataType.INT64),
        _var(f"_pos{level}", llir.DataType.INT),
    )


def _fill_store(
    array_name: str,
    index: llir.Expr,
    value: llir.Expr,
    array_type: llir.DataType = llir.DataType.NO_TYPE,
) -> llir.Assign:
    return llir.Assign(
        _access(array_name, index, array_type),
        value,
    )


def _context(
    mode: _Mode,
    compressed_levels: Tuple[int, ...] = (1,),
    *,
    value_pointer_type: llir.DataType = llir.DataType.PTR_FLOAT32,
) -> ResultWriteContext:
    return ResultWriteContext(
        result_name="Result",
        result_id=_RESULT_ID,
        compressed_levels=compressed_levels,
        mode=mode,
        value_pointer_type=value_pointer_type,
    )


def test_result_write_context_is_frozen_typed_and_value_equal() -> None:
    context = _context("fill")

    assert context == _context("fill")
    assert context != _context("count")
    assert ResultWriteContext.__annotations__ == {
        "result_name": "str",
        "result_id": "SymbolId",
        "compressed_levels": "Tuple[int, ...]",
        "mode": "ResultWriteMode",
        "value_pointer_type": "llir.DataType",
        "traversal": "LLIRTraversalContext",
        "compile_options": "Optional['CompileOptions']",
    }
    with pytest.raises(FrozenInstanceError):
        context.value_pointer_type = llir.DataType.PTR_FLOAT64  # type: ignore[misc]


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


def _cpp(statements: List[llir.Stmt]) -> str:
    return LLIRLowerer().lower_llir(statements)


def _single_level_serial_writes() -> List[llir.Stmt]:
    return [
        llir.Assign(
            _result_value_access(_var("pResult1")),
            llir.BinOp("+", _var("left"), _var("right")),
        ),
        llir.Assign(
            _access("Result1_crd", _var("pResult1")),
            _var("coordinate"),
        ),
        llir.Assign(
            _access("Result1_pos", llir.Add(_var("row"), llir.Literal(1))),
            _var("pResult1"),
        ),
        llir.Increment(_var("pResult1", llir.DataType.INT64)),
        llir.FunctionCallStmt(
            "Result1_crd.push_back",
            [_var("pushed_coordinate")],
        ),
        llir.FunctionCallStmt("Result_values.push_back", []),
        llir.FunctionCallStmt("workspace.sort", []),
        llir.VarInit(
            _var("pResult1", llir.DataType.INT64),
            llir.Literal(0, data_type=llir.DataType.INT64),
        ),
        llir.Assign(_var("scratch"), _var("keep")),
    ]


def _multiple_level_serial_writes() -> List[llir.Stmt]:
    boundary = llir.IfThenElse(
        cond=llir.BinOp(
            "<",
            llir.FunctionCall("Result2_pos.back", []),
            _var("pResult2", llir.DataType.INT64),
        ),
        then_body=[
            llir.FunctionCallStmt(
                "Result1_crd.push_back",
                [_var("parent_coordinate")],
            ),
            llir.RawStmt("discarded_then"),
        ],
        else_body=[llir.RawStmt("discarded_else")],
        cond_list=[_var("discarded_condition")],
        then_body_list=[[llir.RawStmt("discarded_branch")]],
        make_last_case_else=True,
    )
    return [
        llir.FunctionCallStmt(
            "Result1_crd.push_back",
            [_var("outer_coordinate")],
        ),
        boundary,
        llir.Assign(
            _access(
                "Result2_pos",
                llir.FunctionCall("Result1_crd.size", []),
            ),
            _var("Result2_crd.size()"),
        ),
    ]


def test_count_rewrite_matches_legacy_single_level_structure() -> None:
    source = _single_level_serial_writes()
    source_snapshot = _structural_snapshot(source)
    expected: List[llir.Stmt] = [
        llir.Increment(_var("_cnt1", llir.DataType.INT)),
        llir.Increment(_var("_cnt1", llir.DataType.INT)),
        llir.Assign(_var("scratch"), _var("keep")),
    ]

    rewritten = rewrite_result_writes(source, _context("count"))

    assert _structural_snapshot(rewritten) == _structural_snapshot(expected)
    assert _structural_snapshot(source) == source_snapshot
    assert _cpp(rewritten) == "_cnt1++;\n_cnt1++;\nscratch = keep;"


def test_fill_rewrite_matches_legacy_single_level_structure() -> None:
    source = _single_level_serial_writes()
    expected: List[llir.Stmt] = [
        _fill_store(
            "Result_values_data",
            _phase_index(1),
            llir.BinOp("+", _var("left"), _var("right")),
            llir.DataType.PTR_FLOAT32,
        ),
        _fill_store(
            "Result1_crd_data",
            _phase_index(1),
            _var("coordinate"),
            llir.DataType.PTR_INT,
        ),
        llir.Increment(_var("_pos1", llir.DataType.INT)),
        _fill_store(
            "Result1_crd_data",
            _phase_index(1),
            _var("pushed_coordinate"),
            llir.DataType.PTR_INT,
        ),
        llir.Increment(_var("_pos1", llir.DataType.INT)),
        _fill_store(
            "Result_values_data",
            _phase_index(1),
            llir.Literal(0),
            llir.DataType.PTR_FLOAT32,
        ),
        llir.FunctionCallStmt("workspace.sort", []),
        llir.Assign(_var("scratch"), _var("keep")),
    ]

    rewritten = rewrite_result_writes(source, _context("fill"))

    assert _structural_snapshot(rewritten) == _structural_snapshot(expected)
    assert _cpp(rewritten) == """Result_values_data[_base1 + _pos1] = left + right;
Result1_crd_data[_base1 + _pos1] = coordinate;
_pos1++;
Result1_crd_data[_base1 + _pos1] = pushed_coordinate;
_pos1++;
Result_values_data[_base1 + _pos1] = 0;
workspace.sort();
scratch = keep;"""


@pytest.mark.parametrize(
    ("scalar_type", "pointer_type"),
    (
        pytest.param(
            llir.DataType.FLOAT32,
            llir.DataType.PTR_FLOAT32,
            id="float32",
        ),
        pytest.param(
            llir.DataType.FLOAT64,
            llir.DataType.PTR_FLOAT64,
            id="float64",
        ),
        pytest.param(
            llir.DataType.INT32,
            llir.DataType.PTR_INT_32,
            id="int32",
        ),
        pytest.param(
            llir.DataType.INT64,
            llir.DataType.PTR_INT_64,
            id="int64",
        ),
        pytest.param(
            llir.DataType.INT8,
            llir.DataType.PTR_INT8,
            id="int8",
        ),
        pytest.param(
            llir.DataType.UINT8,
            llir.DataType.PTR_UINT8,
            id="uint8",
        ),
    ),
)
def test_fill_rewrite_preserves_each_canonical_value_pointer_type(
    scalar_type: llir.DataType,
    pointer_type: llir.DataType,
) -> None:
    source: List[llir.Stmt] = [
        llir.Assign(
            _result_value_access(_var("pResult1")),
            _var("assigned_value", scalar_type),
        ),
        llir.FunctionCallStmt(
            "Result_values.push_back",
            [_var("pushed_value", scalar_type)],
        ),
    ]

    rewritten = rewrite_result_writes(
        source,
        _context("fill", value_pointer_type=pointer_type),
    )

    assert llir.DataType.ptr_type(scalar_type) is pointer_type
    assert [type(statement) for statement in rewritten] == [llir.Assign, llir.Assign]
    for statement in rewritten:
        assignment = cast(llir.Assign, statement)
        assert type(assignment.var) is llir.ArrayAccess
        target = cast(llir.ArrayAccess, assignment.var)
        assert type(target.array) is llir.Var
        pointer = cast(llir.Var, target.array)
        assert pointer.name == "Result_values_data"
        assert pointer.type is pointer_type
        assert pointer.is_ptr is False
        assert pointer.is_restrict is False
        assert pointer.tensor_access is None
    assert _cpp(cast(List[llir.Stmt], rewritten)) == (
        "Result_values_data[_base1 + _pos1] = assigned_value;\n"
        "Result_values_data[_base1 + _pos1] = pushed_value;"
    )
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(rewritten))


@pytest.mark.parametrize(
    "scalar_type",
    (llir.DataType.INT8, llir.DataType.UINT8),
    ids=("int8-scalar", "uint8-scalar"),
)
def test_result_write_rejects_new_scalar_types_where_pointers_are_required(
    scalar_type: llir.DataType,
) -> None:
    context = _context("fill", value_pointer_type=scalar_type)

    with pytest.raises(LLIRTraversalError) as raised:
        rewrite_result_writes([llir.Break()], context)

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_result_write_value_pointer_type"
    assert diagnostic.path == ("context", "value_pointer_type")
    assert diagnostic.stage == RESULT_WRITE_TRAVERSAL_CONTEXT.stage
    assert diagnostic.pass_name == RESULT_WRITE_TRAVERSAL_CONTEXT.pass_name


def test_fill_rewrite_preserves_structured_workspace_pair_reads() -> None:
    coordinate = llir.ArrayAccess(
        llir.MemberAccess(
            _var("it", llir.DataType.CONST_AUTO_REF),
            "first",
        ),
        llir.Literal(1, llir.DataType.INT64),
    )
    value = llir.MemberAccess(
        _var("it", llir.DataType.CONST_AUTO_REF),
        "second",
    )
    source = [
        llir.Assign(_result_value_access(_var("pResult1")), value),
        llir.Assign(_access("Result1_crd", _var("pResult1")), coordinate),
    ]
    source_snapshot = _structural_snapshot(source)

    rewritten = rewrite_result_writes(source, _context("fill"))

    assert _structural_snapshot(source) == source_snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(rewritten))
    assert _cpp(rewritten) == """Result_values_data[_base1 + _pos1] = it.second;
Result1_crd_data[_base1 + _pos1] = it.first[1];"""
    value_store = cast(llir.Assign, rewritten[0])
    coordinate_store = cast(llir.Assign, rewritten[1])
    assert type(value_store.value) is llir.MemberAccess
    assert type(coordinate_store.value) is llir.ArrayAccess
    rewritten_coordinate = cast(llir.ArrayAccess, coordinate_store.value)
    assert type(rewritten_coordinate.array) is llir.MemberAccess
    original_member = cast(llir.MemberAccess, coordinate.array)
    assert cast(llir.MemberAccess, rewritten_coordinate.array).base is not (
        original_member.base
    )


def test_count_rewrite_matches_legacy_multiple_level_boundary() -> None:
    source = _multiple_level_serial_writes()
    expected: List[llir.Stmt] = [
        llir.Increment(_var("_cnt1", llir.DataType.INT)),
        llir.IfThenElse(
            cond=llir.BinOp(
                ">",
                _var("_cnt2", llir.DataType.INT),
                _var("_prev2", llir.DataType.INT),
            ),
            then_body=[
                llir.Increment(_var("_cnt1", llir.DataType.INT)),
                llir.Assign(
                    _var("_prev2", llir.DataType.INT),
                    _var("_cnt2", llir.DataType.INT),
                ),
            ],
        ),
    ]

    rewritten = rewrite_result_writes(source, _context("count", (1, 2)))

    assert _structural_snapshot(rewritten) == _structural_snapshot(expected)


def test_fill_rewrite_matches_legacy_multiple_level_boundary() -> None:
    source = _multiple_level_serial_writes()
    expected: List[llir.Stmt] = [
        _fill_store(
            "Result1_crd_data",
            _phase_index(1),
            _var("outer_coordinate"),
            llir.DataType.PTR_INT,
        ),
        llir.Increment(_var("_pos1", llir.DataType.INT)),
        _fill_store(
            "Result2_pos_data",
            _phase_index(1),
            _phase_index(2),
            llir.DataType.PTR_INT,
        ),
        llir.IfThenElse(
            cond=llir.BinOp(
                ">",
                _var("_pos2", llir.DataType.INT),
                _var("_prev2", llir.DataType.INT),
            ),
            then_body=[
                _fill_store(
                    "Result1_crd_data",
                    _phase_index(1),
                    _var("parent_coordinate"),
                    llir.DataType.PTR_INT,
                ),
                llir.Increment(_var("_pos1", llir.DataType.INT)),
                _fill_store(
                    "Result2_pos_data",
                    _phase_index(1),
                    _phase_index(2),
                    llir.DataType.PTR_INT,
                ),
                llir.Assign(
                    _var("_prev2", llir.DataType.INT),
                    _var("_pos2", llir.DataType.INT),
                ),
            ],
        ),
    ]

    rewritten = rewrite_result_writes(source, _context("fill", (1, 2)))

    assert _structural_snapshot(rewritten) == _structural_snapshot(expected)


def test_legacy_control_flow_regions_are_transformed_and_detached() -> None:
    auto_loop = llir.ForLoopAuto(
        var=_var("item"),
        array=_var("items"),
        body=[
            llir.Assign(
                _access("Result1_crd", _var("pResult1")),
                _var("auto"),
            )
        ],
    )
    while_loop = llir.WhileLoop(
        _var("keep_going"),
        [llir.Increment(_var("pResult1", llir.DataType.INT64))],
    )
    conditional = llir.IfThenElse(
        cond=_var("guard"),
        then_body=[auto_loop],
        else_body=[while_loop],
        cond_list=[_var("branch_guard")],
        then_body_list=[
            [
                llir.Assign(
                    _access("Result1_crd", _var("pResult1")),
                    _var("branch"),
                )
            ]
        ],
    )
    loop = llir.ForLoop(
        init=llir.VarInit(
            _var("pResult1", llir.DataType.INT64),
            llir.Literal(0, data_type=llir.DataType.INT64),
        ),
        cond=llir.BinOp("<", _var("pResult1"), _var("limit")),
        update=llir.Increment(_var("pResult1", llir.DataType.INT64)),
        body=[conditional],
        # The two identity-only regions are probed with the workspace drain's
        # sort rather than with a result write.  The sort is dispatched by
        # ``_rewrite_call_statement`` and DROPPED in count mode, so surviving
        # here proves the region really was left alone -- and it is not result
        # storage, so it does not collide with the pass's postcondition.  A
        # result write in either region is refused; see
        # ``test_identity_only_regions_refuse_a_surviving_result_write``.
        before_parallel_body=[llir.FunctionCallStmt("workspace.sort", [])],
        pre_parallel_body=[
            llir.Assign(
                _access("Result1_crd", _var("pResult1")),
                _var("pre"),
            )
        ],
        post_parallel_body=[
            llir.Assign(
                _access("Result1_crd", _var("pResult1")),
                _var("post"),
            )
        ],
    )
    setattr(
        loop,
        "_hoisted_ptr_decls",
        [llir.FunctionCallStmt("workspace.sort", [])],
    )
    source: List[llir.Stmt] = [loop]
    source_snapshot = _structural_snapshot(source)

    rewritten = rewrite_result_writes(source, _context("count"))
    rewritten_loop = cast(llir.ForLoop, rewritten[0])
    rewritten_if = cast(llir.IfThenElse, rewritten_loop.body[0])
    rewritten_auto = cast(
        llir.ForLoopAuto, cast(List[llir.Stmt], rewritten_if.then_body)[0]
    )
    rewritten_while = cast(
        llir.WhileLoop, cast(List[llir.Stmt], rewritten_if.else_body)[0]
    )
    rewritten_branches = cast(List[List[llir.Stmt]], rewritten_if.then_body_list)

    assert type(rewritten_loop.init) is llir.VarInit
    assert cast(llir.VarInit, rewritten_loop.init).var.name == "pResult1"
    assert type(rewritten_loop.update) is llir.Increment
    assert cast(llir.Increment, rewritten_loop.update).var.name == "pResult1"
    assert rewritten_auto.body == [llir.Increment(_var("_cnt1", llir.DataType.INT))]
    assert rewritten_while.body == []
    assert rewritten_branches[0] == [llir.Increment(_var("_cnt1", llir.DataType.INT))]
    assert rewritten_loop.pre_parallel_body is not None
    assert rewritten_loop.pre_parallel_body[0] == llir.Increment(
        _var("_cnt1", llir.DataType.INT)
    )
    assert rewritten_loop.post_parallel_body is not None
    assert rewritten_loop.post_parallel_body[0] == llir.Increment(
        _var("_cnt1", llir.DataType.INT)
    )

    # Both identity-only regions kept a statement count mode drops everywhere
    # else, which is what "the legacy transform never descended into these"
    # means, and both are fresh objects rather than the input's lists.
    assert rewritten_loop.before_parallel_body is not None
    before = cast(llir.FunctionCallStmt, rewritten_loop.before_parallel_body[0])
    assert type(before) is llir.FunctionCallStmt
    assert before.name == "workspace.sort"
    assert rewritten_loop.before_parallel_body is not loop.before_parallel_body
    hoisted = cast(List[llir.Stmt], getattr(rewritten_loop, "_hoisted_ptr_decls"))
    assert type(hoisted[0]) is llir.FunctionCallStmt
    assert cast(llir.FunctionCallStmt, hoisted[0]).name == "workspace.sort"

    assert _structural_snapshot(source) == source_snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(rewritten))


def test_none_and_empty_optional_loop_regions_preserve_their_shape() -> None:
    absent = llir.ForLoop(
        init=None,
        cond=_var("guard"),
        update=llir.Increment(_var("i")),
        body=[],
    )
    empty = llir.ForLoop(
        init=None,
        cond=_var("guard"),
        update=llir.Increment(_var("i")),
        body=[],
        before_parallel_body=[],
        pre_parallel_body=[],
        post_parallel_body=[],
    )

    rewritten = rewrite_result_writes([absent, empty], _context("count"))
    rewritten_absent = cast(llir.ForLoop, rewritten[0])
    rewritten_empty = cast(llir.ForLoop, rewritten[1])

    assert rewritten_absent.before_parallel_body is None
    assert rewritten_absent.pre_parallel_body is None
    assert rewritten_absent.post_parallel_body is None
    assert rewritten_empty.before_parallel_body == []
    assert rewritten_empty.pre_parallel_body == []
    assert rewritten_empty.post_parallel_body == []
    assert rewritten_empty.before_parallel_body is not empty.before_parallel_body
    assert rewritten_empty.pre_parallel_body is not empty.pre_parallel_body
    assert rewritten_empty.post_parallel_body is not empty.post_parallel_body


def test_function_body_is_identity_only_and_detached() -> None:
    # Probed with the drain's sort for the same reason the identity-only loop
    # regions are: count mode drops it everywhere the rewriter descends, so
    # surviving proves ``rewrite_function`` did not descend, and it is not
    # result storage.  A result write inside a nested function body is refused;
    # see ``test_nested_function_body_refuses_a_surviving_result_write``.
    source: List[llir.Stmt] = [
        llir.Function(
            return_type=llir.DataType.VOID,
            name="nested",
            args=[],
            body=[llir.FunctionCallStmt("workspace.sort", [])],
        ),
    ]

    rewritten = rewrite_result_writes(source, _context("count"))

    assert _structural_snapshot(rewritten) == _structural_snapshot(source)
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(rewritten))


def test_nested_list_and_tuple_roots_preserve_container_shapes() -> None:
    source: List[LLIRStatementValue] = [
        (
            [
                llir.Assign(
                    _access("Result1_crd", _var("pResult1")),
                    _var("coordinate"),
                )
            ],
        ),
        [
            (
                llir.FunctionCallStmt(
                    "Result1_crd.push_back",
                    [_var("pushed")],
                ),
            )
        ],
    ]

    rewritten = rewrite_result_writes(source, _context("fill"))

    assert type(rewritten) is list
    assert type(rewritten[0]) is tuple
    first_tuple = cast(Tuple[LLIRStatementValue, ...], rewritten[0])
    assert type(first_tuple[0]) is list
    first_list = cast(List[LLIRStatementValue], first_tuple[0])
    assert type(first_list[0]) is llir.Assign
    assert type(rewritten[1]) is list
    second_list = cast(List[LLIRStatementValue], rewritten[1])
    assert type(second_list[0]) is tuple
    second_tuple = cast(Tuple[LLIRStatementValue, ...], second_list[0])
    assert [type(statement) for statement in second_tuple] == [
        llir.Assign,
        llir.Increment,
    ]


def test_result_write_pass_shares_no_mutable_ir_with_input() -> None:
    source: List[LLIRStatementValue] = [
        cast(List[LLIRStatementValue], _single_level_serial_writes()),
        tuple(cast(List[LLIRStatementValue], _multiple_level_serial_writes())),
    ]
    source_snapshot = _structural_snapshot(source)

    rewritten = rewrite_result_writes(source, _context("fill", (1, 2)))

    assert _structural_snapshot(source) == source_snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(rewritten))


def test_unknown_subclass_fails_with_result_write_stage_diagnostic() -> None:
    class UnknownAssign(llir.Assign):
        pass

    unknown = UnknownAssign(_result_value_access(_var("pResult1")), _var("value"))
    unknown_root: List[llir.Stmt] = [unknown]

    with pytest.raises(LLIRTraversalError) as raised:
        rewrite_result_writes(unknown_root, _context("count"))

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.stage == RESULT_WRITE_TRAVERSAL_CONTEXT.stage
    assert diagnostic.pass_name == RESULT_WRITE_TRAVERSAL_CONTEXT.pass_name
    assert diagnostic.node_type == "UnknownAssign"
    assert diagnostic.path == ("root", "[0]")


@pytest.mark.parametrize(
    ("context", "expected_code", "expected_path"),
    [
        (
            cast(ResultWriteContext, object()),
            "invalid_result_write_context",
            ("context",),
        ),
        (
            ResultWriteContext(
                "", _RESULT_ID, (1,), "count", llir.DataType.PTR_FLOAT32
            ),
            "invalid_result_write_name",
            ("context", "result_name"),
        ),
        (
            ResultWriteContext(
                cast(str, 3),
                _RESULT_ID,
                (1,),
                "count",
                llir.DataType.PTR_FLOAT32,
            ),
            "invalid_result_write_name",
            ("context", "result_name"),
        ),
        (
            ResultWriteContext(
                "Result",
                cast(SymbolId, 3),
                (1,),
                "count",
                llir.DataType.PTR_FLOAT32,
            ),
            "invalid_result_write_id",
            ("context", "result_id"),
        ),
        (
            ResultWriteContext(
                "Result", _RESULT_ID, (), "count", llir.DataType.PTR_FLOAT32
            ),
            "invalid_compressed_levels",
            ("context", "compressed_levels"),
        ),
        (
            ResultWriteContext(
                "Result",
                _RESULT_ID,
                cast(Tuple[int, ...], [1]),
                "count",
                llir.DataType.PTR_FLOAT32,
            ),
            "invalid_compressed_levels",
            ("context", "compressed_levels"),
        ),
        (
            ResultWriteContext(
                "Result", _RESULT_ID, (-1,), "count", llir.DataType.PTR_FLOAT32
            ),
            "invalid_compressed_levels",
            ("context", "compressed_levels"),
        ),
        (
            ResultWriteContext(
                "Result", _RESULT_ID, (1, 1), "count", llir.DataType.PTR_FLOAT32
            ),
            "invalid_compressed_levels",
            ("context", "compressed_levels"),
        ),
        (
            ResultWriteContext(
                "Result", _RESULT_ID, (2, 1), "count", llir.DataType.PTR_FLOAT32
            ),
            "invalid_compressed_levels",
            ("context", "compressed_levels"),
        ),
        (
            ResultWriteContext(
                "Result",
                _RESULT_ID,
                (1,),
                cast(_Mode, "invalid"),
                llir.DataType.PTR_FLOAT32,
            ),
            "invalid_result_write_mode",
            ("context", "mode"),
        ),
        (
            ResultWriteContext(
                "Result",
                _RESULT_ID,
                (1,),
                "count",
                cast(llir.DataType, "float*"),
            ),
            "invalid_result_write_value_pointer_type",
            ("context", "value_pointer_type"),
        ),
        (
            ResultWriteContext(
                "Result",
                _RESULT_ID,
                (1,),
                "count",
                llir.DataType.FLOAT32,
            ),
            "invalid_result_write_value_pointer_type",
            ("context", "value_pointer_type"),
        ),
        (
            ResultWriteContext(
                "Result",
                _RESULT_ID,
                (1,),
                "count",
                llir.DataType.PTR_FLOAT32,
                traversal=LLIRTraversalContext(stage="", pass_name="custom"),
            ),
            "invalid_result_write_traversal_context",
            ("context", "traversal"),
        ),
    ],
)
def test_invalid_contexts_fail_structurally(
    context: ResultWriteContext,
    expected_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    valid_root: List[llir.Stmt] = [llir.Break()]
    with pytest.raises(LLIRTraversalError) as raised:
        rewrite_result_writes(valid_root, context)

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == expected_code
    assert diagnostic.path == expected_path
    assert diagnostic.stage == RESULT_WRITE_TRAVERSAL_CONTEXT.stage
    assert diagnostic.pass_name == RESULT_WRITE_TRAVERSAL_CONTEXT.pass_name


def test_valid_no_op_is_structurally_equal_and_detached() -> None:
    source: List[llir.Stmt] = [
        llir.VarInit(_var("scratch"), llir.Literal(1)),
        llir.IfThenElse(
            cond=_var("guard"),
            then_body=[llir.Assign(_var("output"), _var("input"))],
            else_body=[llir.Break()],
        ),
    ]

    rewritten = rewrite_result_writes(source, _context("count"))

    assert _structural_snapshot(rewritten) == _structural_snapshot(source)
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(rewritten))


def test_expression_root_is_a_detached_legal_no_op() -> None:
    source = llir.BinOp("+", _var("left"), _var("right"))

    rewritten = rewrite_result_writes(source, _context("count"))

    assert type(rewritten) is llir.BinOp
    assert _structural_snapshot(rewritten) == _structural_snapshot(source)
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(rewritten))


@pytest.mark.parametrize("mode", ["count", "fill"])
def test_same_mode_rewrite_is_idempotent_for_generated_shapes(
    mode: _Mode,
) -> None:
    source = [*_single_level_serial_writes(), *_multiple_level_serial_writes()]
    context = _context(mode, (1, 2))

    once = rewrite_result_writes(source, context)
    twice = rewrite_result_writes(once, context)

    assert _structural_snapshot(twice) == _structural_snapshot(once)
    assert _mutable_ir_ids(once).isdisjoint(_mutable_ir_ids(twice))


def test_count_and_fill_are_independent_not_composable() -> None:
    source: List[llir.Stmt] = [
        llir.Assign(_result_value_access(_var("pResult1")), _var("value"))
    ]
    counted = rewrite_result_writes(source, _context("count"))

    fill_after_count = rewrite_result_writes(counted, _context("fill"))
    direct_fill = rewrite_result_writes(source, _context("fill"))

    assert _structural_snapshot(fill_after_count) == _structural_snapshot(counted)
    assert _structural_snapshot(fill_after_count) != _structural_snapshot(direct_fill)
    assert fill_after_count is not counted


@pytest.mark.parametrize(
    ("statement", "context", "replacement_count"),
    [
        (
            llir.Assign(
                _result_value_access(_var("pResult1")),
                _var("value"),
            ),
            _context("count"),
            0,
        ),
        (
            llir.FunctionCallStmt(
                "Result1_crd.push_back",
                [_var("coordinate")],
            ),
            _context("fill"),
            2,
        ),
    ],
)
def test_scalar_deletion_and_expansion_fail_structurally(
    statement: llir.Stmt,
    context: ResultWriteContext,
    replacement_count: int,
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        rewrite_result_writes(statement, context)

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unsupported_scalar_result_write_root"
    assert diagnostic.path == ("root",)
    assert diagnostic.stage == RESULT_WRITE_TRAVERSAL_CONTEXT.stage
    assert diagnostic.pass_name == RESULT_WRITE_TRAVERSAL_CONTEXT.pass_name
    assert f"produces {replacement_count} statements" in diagnostic.message


def test_scalar_one_for_one_replacement_preserves_root_category() -> None:
    source = llir.Assign(
        _result_value_access(_var("pResult1")),
        _var("value"),
    )

    rewritten = rewrite_result_writes(source, _context("fill"))

    assert type(rewritten) is llir.Assign
    assert type(rewritten.var) is llir.ArrayAccess
    assert cast(llir.Var, rewritten.var.array).name == "Result_values_data"
    assert cast(llir.Var, rewritten.var.array).type is llir.DataType.PTR_FLOAT32
    assert _cpp([rewritten]) == "Result_values_data[_base1 + _pos1] = value;"
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(rewritten))


def test_result_value_matching_uses_stable_identity_not_rendered_name() -> None:
    # A RESULT_WRITE marker naming a DIFFERENT tensor is not this result's
    # value store and passes through.  The physical name is another tensor's
    # too, which is what production emits: one generated function spells one
    # result's storage one way, so a foreign tensor cannot also be called
    # ``Result_values``.  The contradictory pairing -- this result's storage
    # name carrying a foreign marker -- is refused instead, because the name
    # is what codegen emits and that declaration is gone; see
    # ``test_foreign_marker_on_this_results_storage_name_is_refused``.
    foreign = llir.ArrayAccess(
        array=_var("Other_values", llir.DataType.PTR_FLOAT32),
        index=_var("pOther1"),
        tensor_access=llir.TensorAccessMetadata(
            access_id=AccessId(702),
            tensor_id=SymbolId(_RESULT_ID.value + 1),
            index_ids=(_RESULT_INDEX_ID,),
            role=llir.TensorAccessRole.RESULT_WRITE,
        ),
    )
    source: List[llir.Stmt] = [llir.Assign(foreign, _var("value"))]

    rewritten = rewrite_result_writes(source, _context("fill"))

    assert _structural_snapshot(rewritten) == _structural_snapshot(source)
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(rewritten))
    assert _cpp(cast(List[llir.Stmt], rewritten)) == ("Other_values[pOther1] = value;")


def test_result_value_matching_ignores_physical_name_when_identity_matches() -> None:
    logical_target = _result_value_access(_var("pResult1"))
    target = llir.ArrayAccess(
        array=_var("renamed_physical_values", llir.DataType.PTR_FLOAT32),
        index=logical_target.index,
        tensor_access=logical_target.tensor_access,
    )
    source: List[llir.Stmt] = [llir.Assign(target, _var("value"))]

    rewritten = rewrite_result_writes(source, _context("fill"))

    assignment = cast(llir.Assign, rewritten[0])
    assert type(assignment.var) is llir.ArrayAccess
    assert cast(llir.Var, assignment.var.array).name == "Result_values_data"
    assert cast(llir.Var, assignment.var.array).type is llir.DataType.PTR_FLOAT32
    assert _cpp(cast(List[llir.Stmt], rewritten)) == (
        "Result_values_data[_base1 + _pos1] = value;"
    )


def test_malformed_rvalue_assignment_target_fails_at_result_write_boundary() -> None:
    malformed = llir.Assign(_var("scratch"), _var("value"))
    malformed.var = cast(
        llir.AssignmentTarget,
        llir.FunctionCall("not_an_lvalue", []),
    )

    with pytest.raises(LLIRTraversalError) as raised:
        rewrite_result_writes([malformed], _context("fill"))

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_assignment_target"
    assert diagnostic.path == ("root", "[0]", "var")
    assert diagnostic.stage == RESULT_WRITE_TRAVERSAL_CONTEXT.stage
    assert diagnostic.pass_name == RESULT_WRITE_TRAVERSAL_CONTEXT.pass_name


# -- The postcondition: no reference to the removed storage may survive -------
#
# These lock the check the pass runs over its own OUTPUT.  It asks a different
# question than ``_touches_result_storage`` does: that guard asks of an input
# statement "is this a result write I recognize", so it can only refuse a shape
# somebody anticipated, while this asks of the output "did a reference to the
# removed storage survive", which is a question about names rather than about
# statement shapes.  The two tests named ``..._f_weak_...`` are the pair that
# separates the two versions of the postcondition described in
# ``COMPILER_IR_RESULT_WRITE_GUARD_OPTIONS.md``: both shapes are invisible to
# an enumeration of known write forms and both are caught here.


def _residue_diagnostic(
    statements: List[llir.Stmt], mode: _Mode = "count"
) -> LLIRTraversalError:
    with pytest.raises(LLIRTraversalError) as raised:
        rewrite_result_writes(statements, _context(mode))
    assert raised.value.diagnostic.code == "residual_result_storage_reference"
    assert raised.value.diagnostic.stage == RESULT_WRITE_TRAVERSAL_CONTEXT.stage
    assert raised.value.diagnostic.pass_name == RESULT_WRITE_TRAVERSAL_CONTEXT.pass_name
    return raised.value


def test_postcondition_accepts_every_rewrite_the_pass_produces() -> None:
    """The fill phase stores INTO the result, and that is not residue.

    This is the check's whole well-formedness question.  ``_store`` emits
    ``{R}{L}_crd_data`` / ``{R}_values_data`` / ``{R}{L}_pos_data``, which share
    the bare names' leading characters, so a prefix rule rather than an
    exact-plus-dot rule would refuse the pass's own output on every fill.
    """

    for mode in cast(Tuple[_Mode, ...], ("count", "fill")):
        rewritten = rewrite_result_writes(_single_level_serial_writes(), _context(mode))
        emitted = _cpp(cast(List[llir.Stmt], rewritten))
        assert "Result_values_data" in emitted or mode == "count"
        assert "Result1_crd[" not in emitted
        assert "Result_values[" not in emitted


def test_free_function_taking_a_result_array_as_an_argument_is_refused() -> None:
    """Gap A: the property is argument-shaped and the input guard reads the callee.

    ``loopir/parallel_chunk_assembly`` already emits four statements of this
    shape.  ``_touches_result_storage`` matches ``FunctionCallStmt.name``
    against the array prefixes, and ``scorch_concat_chunks`` starts with none
    of them, so the input guard passes it through; it survives, and the
    postcondition is what sees it.
    """

    error = _residue_diagnostic(
        [
            llir.FunctionCallStmt(
                "scorch_concat_chunks",
                [_var("Result_values"), _var("chunks")],
            )
        ]
    )
    assert "Result_values" in error.diagnostic.message


def test_argument_shaped_result_write_is_invisible_to_f_weak() -> None:
    """The one cell that separates the two versions of the postcondition.

    An enumeration of known write forms recognizes ``scorch_vector_set`` by its
    first argument and knows nothing of ``scorch_concat_chunks``, so the same
    statement above would survive it as unflagged residue.  Asserted here as
    the property of the enumeration rather than of any code that ships: the
    callee is not an append spelling and not the one registered helper.
    """

    residue = llir.FunctionCallStmt(
        "scorch_concat_chunks",
        [_var("Result_values"), _var("chunks")],
    )
    assert not residue.name.endswith((".push_back", ".emplace_back"))
    assert residue.name != "scorch_vector_set"
    assert not residue.name.startswith("Result_values.")
    # ...and yet the postcondition refuses it, on the argument alone.
    _residue_diagnostic([residue])


def test_member_call_statement_on_a_result_array_is_refused() -> None:
    """Gap B: ``MemberCallStmt`` is never dispatched by the rewriter.

    ``rewrite_statement_sequence_member`` names ``Assign``, ``Increment``,
    ``FunctionCallStmt``, ``VarInit`` and ``IfThenElse``; a ``MemberCallStmt``
    falls to the identity path, so the input guard -- which runs inside
    ``_rewrite_call_statement`` -- is never offered it.  The postcondition runs
    over the whole traversal instead of inside one handler, so it reaches the
    receiver as a child expression.
    """

    error = _residue_diagnostic(
        [
            llir.MemberCallStmt(
                base=_var("Result1_crd"),
                member="push_back",
                args=[_var("coordinate")],
            )
        ]
    )
    assert "Result1_crd" in error.diagnostic.message


def test_identity_only_regions_refuse_a_surviving_result_write() -> None:
    """A region the rewriter declines to descend into is not a safe hiding place.

    ``_IDENTITY_ONLY_REGIONS`` keeps ``before_parallel_body`` and
    ``_hoisted_ptr_decls`` un-rewritten because the legacy transform never
    descended into them.  A result write there survives into the count body
    against a declaration the surrounding transform has dropped, which is the
    same defect an unrecognized spelling causes.  Unreachable on the survey
    matrix -- measured zero -- and refused rather than retained.
    """

    def result_write() -> llir.Assign:
        return llir.Assign(_access("Result1_crd", _var("pResult1")), _var("coordinate"))

    for region in ("before_parallel_body", "_hoisted_ptr_decls"):
        loop = llir.ForLoop(
            init=None,
            cond=_var("guard"),
            update=llir.Increment(_var("row", llir.DataType.INT64)),
            body=[],
        )
        setattr(loop, region, [result_write()])
        error = _residue_diagnostic([loop])
        assert region in error.diagnostic.message


def test_nested_function_body_refuses_a_surviving_result_write() -> None:
    """``rewrite_function`` is identity, so a result write in a body survives.

    Same class as the identity-only loop regions.  ``Function`` is not among
    the statement types the survey matrix presents to this pass, so this is
    unreachable today and refused rather than retained.
    """

    error = _residue_diagnostic(
        [
            llir.Function(
                return_type=llir.DataType.VOID,
                name="nested",
                args=[],
                body=[
                    llir.Assign(
                        _access("Result1_crd", _var("pResult1")),
                        _var("coordinate"),
                    )
                ],
            )
        ]
    )
    assert "Result1_crd" in error.diagnostic.message


def test_foreign_marker_on_this_results_storage_name_is_refused() -> None:
    """A foreign ``RESULT_WRITE`` marker does not license this result's name.

    ``_is_result_value_target`` decides by logical identity, so this statement
    is correctly NOT rewritten as the value store.  But the name is what
    codegen emits, and ``Result_values``'s declaration is gone, so retaining it
    would emit a dangling reference.  The refusal is the postcondition's, not
    the recognizer's.
    """

    contradictory = llir.ArrayAccess(
        array=_var("Result_values", llir.DataType.PTR_FLOAT32),
        index=_var("index"),
        tensor_access=llir.TensorAccessMetadata(
            access_id=AccessId(702),
            tensor_id=SymbolId(_RESULT_ID.value + 1),
            index_ids=(_RESULT_INDEX_ID,),
            role=llir.TensorAccessRole.RESULT_WRITE,
        ),
    )
    error = _residue_diagnostic([llir.Assign(contradictory, _var("value"))], "fill")
    assert "Result_values" in error.diagnostic.message


def test_surviving_result_write_marker_is_refused_in_both_modes() -> None:
    """The typed axis, which is how the drain's value store is recognized.

    A marked reference must survive neither mode: count drops the store and
    fill replaces it with a ``_store`` that carries no metadata.  Reached here
    through a statement type the rewriter does not dispatch, so the marker is
    the only thing left to see it by.
    """

    for mode in cast(Tuple[_Mode, ...], ("count", "fill")):
        error = _residue_diagnostic(
            [
                llir.MemberCallStmt(
                    base=_result_value_access(_var("index")),
                    member="assign",
                    args=[_var("value")],
                )
            ],
            mode,
        )
        assert "result_write" in error.diagnostic.message


def test_a_read_of_removed_storage_is_refused_because_it_dangles() -> None:
    """A surviving READ is a defect too, which is why the rule is not shape-based.

    ``compressed_where_openmp_pass._should_drop_prefix_statement`` drops the
    declarations of ``{R}_values``, ``{R}{L}_pos``, ``{R}{L}_crd`` and
    ``p{R}{L}``, so in this pass's output those names do not exist and a read
    of one is as dangling as a write.  ``{R}{L}_crd.size`` as an index and
    ``{R}{L}_pos.back`` in a condition are both legal in the INPUT -- 178 of
    them per mode over the survey matrix -- and the pass removes every one.
    """

    surviving_read = llir.Assign(
        _var("scratch"),
        llir.FunctionCall("Result1_crd.size", []),
    )
    error = _residue_diagnostic([surviving_read])
    assert "Result1_crd.size" in error.diagnostic.message


def test_the_position_cursor_is_checked_where_the_pass_rewrites_it() -> None:
    """Cursors are narrower than arrays, and the loop header is why.

    An array's declaration is always dropped, so any reference to one dangles.
    A cursor can be bound locally by the header of the loop that walks it, and
    ``ForLoop.init``/``update`` are also positions the rewriter structurally
    cannot reach.  So a cursor is checked as an ``Increment`` or ``VarInit``
    sequence member -- the two forms the pass has a rewrite for -- and not
    wherever its name appears.  This is the F-weak prototype's cursor coverage,
    kept rather than widened.
    """

    # A loop header binding its own cursor is accepted.
    header_bound = llir.ForLoop(
        init=llir.VarInit(
            _var("pResult1", llir.DataType.INT64),
            llir.Literal(0, data_type=llir.DataType.INT64),
        ),
        cond=llir.BinOp("<", _var("pResult1"), _var("limit")),
        update=llir.Increment(_var("pResult1", llir.DataType.INT64)),
        body=[],
    )
    rewrite_result_writes([header_bound], _context("count"))

    # The same bump as a sequence member IS dispatched, and fill mode rewrites
    # it to ``_pos1`` rather than keeping it, so it never survives there.
    dispatched = rewrite_result_writes(
        [llir.Increment(_var("pResult1", llir.DataType.INT64))], _context("fill")
    )
    assert dispatched == [llir.Increment(_var("_pos1", llir.DataType.INT))]

    # Where it is NOT dispatched -- an identity-only region -- it survives, and
    # then the fill body's positions never advance.  That is residue.
    undispatched = llir.ForLoop(
        init=None,
        cond=_var("guard"),
        update=llir.Increment(_var("row", llir.DataType.INT64)),
        body=[],
        before_parallel_body=[llir.Increment(_var("pResult1", llir.DataType.INT64))],
    )
    error = _residue_diagnostic([undispatched], "fill")
    assert "pResult1" in error.diagnostic.message


def test_verbatim_cpp_mentioning_removed_storage_is_refused() -> None:
    """``RawStmt`` holds C++ text, so the check reads it as text.

    The boundary rule is whole-identifier, and both directions matter: the
    pass's own ``Result1_crd_data`` pointer shares the bare name's leading
    characters and must NOT match, or every fill body would refuse itself.
    """

    error = _residue_diagnostic([llir.RawStmt("Result1_crd.push_back(coordinate)")])
    assert "Result1_crd" in error.diagnostic.message

    # A longer identifier that merely contains the name is not a reference to
    # it.  ``Result1_crd_data`` is the pass's own pointer; ``myResult1_crd`` and
    # ``Result1_crds`` are unrelated names.
    for accepted in (
        "Result1_crd_data[_base1 + _pos1] = coordinate",
        "myResult1_crd = 0",
        "Result1_crds.clear()",
        "int Result_valuesize = 0",
    ):
        rewrite_result_writes([llir.RawStmt(accepted)], _context("count"))


def test_result_write_pass_defect_codes_are_the_locked_set() -> None:
    """Lock the pass's structured refusal surface, failing in both directions.

    This is the analogue of
    ``test_loopir_verifier.test_defect_codes_are_the_documented_production_subset``
    for this pass.  That set is the LoopIR VERIFIER's -- it is built from
    ``_fail("...")`` occurrences in ``loopir/verifier.py`` -- so none of this
    pass's codes belong in it, and adding one there would fail the equality it
    asserts.  This is where they are locked instead.
    """

    import re

    import scorch.compiler.result_write_pass as pass_module

    source = open(pass_module.__file__).read()
    found = set(re.findall(r"code=\"([a-z_]+)\"", source))
    assert found == {
        # Context validation.
        "invalid_result_write_context",
        "invalid_result_write_traversal_context",
        "invalid_result_write_compile_options",
        "invalid_result_write_name",
        "invalid_result_write_id",
        "invalid_compressed_levels",
        "invalid_result_write_mode",
        "invalid_result_write_value_pointer_type",
        # Root-category preservation.
        "unsupported_scalar_result_write_root",
        # The input-side guard: a call naming result storage with no rewrite.
        "unsupported_result_write_statement",
        # The postcondition: a reference to removed storage survived.
        "residual_result_storage_reference",
    }
