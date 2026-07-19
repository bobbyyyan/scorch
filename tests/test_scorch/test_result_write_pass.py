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
) -> ResultWriteContext:
    return ResultWriteContext(
        result_name="Result",
        result_id=_RESULT_ID,
        compressed_levels=compressed_levels,
        mode=mode,
        value_pointer_type=llir.DataType.PTR_FLOAT32,
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
        before_parallel_body=[
            llir.Assign(
                _access("Result1_crd", _var("pResult1")),
                _var("before"),
            )
        ],
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
        [
            llir.Assign(
                _access("Result1_crd", _var("pResult1")),
                _var("hoisted"),
            )
        ],
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

    assert rewritten_loop.before_parallel_body is not None
    before = cast(llir.Assign, rewritten_loop.before_parallel_body[0])
    assert type(before.var) is llir.ArrayAccess
    assert cast(llir.Var, cast(llir.ArrayAccess, before.var).array).name == (
        "Result1_crd"
    )
    hoisted = cast(List[llir.Stmt], getattr(rewritten_loop, "_hoisted_ptr_decls"))
    assert type(hoisted[0]) is llir.Assign
    hoisted_target = cast(llir.Assign, hoisted[0]).var
    assert type(hoisted_target) is llir.ArrayAccess
    assert cast(llir.Var, cast(llir.ArrayAccess, hoisted_target).array).name == (
        "Result1_crd"
    )

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
    def result_write(value: str) -> llir.Assign:
        return llir.Assign(
            _access("Result1_crd", _var("pResult1")),
            _var(value),
        )

    source: List[llir.Stmt] = [
        llir.Function(
            return_type=llir.DataType.VOID,
            name="nested",
            args=[],
            body=[result_write("function")],
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
    source: List[llir.Stmt] = [
        llir.Assign(
            _result_value_access(
                _var("pResult1"),
                tensor_id=SymbolId(_RESULT_ID.value + 1),
            ),
            _var("value"),
        )
    ]

    rewritten = rewrite_result_writes(source, _context("fill"))

    assert _structural_snapshot(rewritten) == _structural_snapshot(source)
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(rewritten))
    assert _cpp(cast(List[llir.Stmt], rewritten)) == (
        "Result_values[pResult1] = value;"
    )


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
