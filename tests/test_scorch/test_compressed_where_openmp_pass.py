import hashlib
from collections import Counter
from dataclasses import FrozenInstanceError
import re
from typing import List, Set, Tuple, cast

import pytest

from scorch.compiler import llir
from scorch.compiler import compressed_where_openmp_pass as compressed_where_module
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.compressed_where_openmp_pass import (
    CompressedWhereOpenMPContext,
    CompressedWhereOpenMPPolicy,
    CompressedWhereOpenMPResult,
    transform_compressed_where_for_openmp,
)
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]
from scorch.compiler.llir_traversal import (
    LLIRStatementValue,
    LLIRTraversalContext,
    LLIRTraversalError,
    LLIRValue,
    LLIRWalker,
)
from scorch.compiler.llir_pass_manager import DEBUG_LLIR_PASS_OPTIONS
from scorch.compiler.scheduler import Scheduler
from scorch.compiler.torch_cpp_abi import (  # type: ignore[import-untyped]
    ResultTensorAssembler,
)


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _context(
    compressed_levels: Tuple[int, ...] = (1,),
    *,
    policy: CompressedWhereOpenMPPolicy = CompressedWhereOpenMPPolicy(),
    result_id: SymbolId = SymbolId(1),
) -> CompressedWhereOpenMPContext:
    return CompressedWhereOpenMPContext(
        result_name="Result",
        result_id=result_id,
        compressed_levels=compressed_levels,
        workspace_name="wksp",
        workspace_ctype="float",
        policy=policy,
    )


def _workspace_init() -> llir.VarInit:
    return llir.VarInit(
        _var("wksp", llir.DataType.AUTO),
        llir.FunctionCall("coo_workspace_1d<float, 1>", [llir.Literal(1024)]),
    )


def _compatible_loop(
    body: List[llir.Stmt],
    *,
    bound: str = "A0_size",
    cond_op: str = "<",
    update: llir.Assign | None = None,
) -> llir.ForLoop:
    row = _var("row", llir.DataType.INT)
    return llir.ForLoop(
        init=llir.VarInit(row, llir.Literal(0)),
        cond=llir.BinOp(cond_op, row, _var(bound, llir.DataType.INT64)),
        update=update or llir.Increment(_var("row", llir.DataType.INT)),
        body=body,
    )


def _ds_work_body(
    *, workspace: bool = True, both_operands: bool = True
) -> List[llir.Stmt]:
    body: List[llir.Stmt] = []
    if workspace:
        body.append(_workspace_init())
    body.append(llir.RawStmt("int pA1 = A1_pos[row]"))
    if both_operands:
        body.append(llir.RawStmt("int pB1 = B1_pos[reduction]"))
    body.extend(
        [
            llir.FunctionCallStmt("wksp.insert", [_var("value")]),
            llir.FunctionCallStmt("Result1_crd.push_back", [_var("column")]),
            llir.FunctionCallStmt("Result_values.push_back", [_var("value")]),
        ]
    )
    return body


def _structured_ds_work_body() -> List[llir.Stmt]:
    def end_init(array: str, parent: str, iterator: str) -> llir.VarInit:
        return llir.VarInit(
            _var(f"{iterator}_end", llir.DataType.INT),
            llir.ArrayAccess(
                _var(array, llir.DataType.PTR_INT),
                llir.Add(
                    _var(parent, llir.DataType.INT),
                    llir.Literal(1, llir.DataType.INT),
                ),
            ),
        )

    return [
        _workspace_init(),
        end_init("A1_pos", "pA0", "pA1"),
        end_init("B1_pos", "pB0", "pB1"),
        llir.FunctionCallStmt("wksp.insert", [_var("value")]),
        llir.FunctionCallStmt("Result1_crd.push_back", [_var("column")]),
        llir.FunctionCallStmt("Result_values.push_back", [_var("value")]),
    ]


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


def _raw_codes(value: object) -> List[str]:
    codes: List[str] = []
    if type(value) is llir.RawStmt:
        codes.append(cast(llir.RawStmt, value).code)
    elif isinstance(value, llir.Node):
        for child in vars(value).values():
            codes.extend(_raw_codes(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            codes.extend(_raw_codes(child))
    return codes


def _assignments(value: object) -> List[llir.Assign]:
    assignments: List[llir.Assign] = []
    if type(value) is llir.Assign:
        assignments.append(cast(llir.Assign, value))
    elif isinstance(value, llir.Node):
        for child in vars(value).values():
            assignments.extend(_assignments(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            assignments.extend(_assignments(child))
    return assignments


def _assignment_codes(value: object) -> List[str]:
    return [
        LLIRLowerer().lower_llir([assignment]).removesuffix(";")
        for assignment in _assignments(value)
    ]


def _phase_state_statements(value: object) -> List[llir.Stmt]:
    statements: List[llir.Stmt] = []
    if type(value) is llir.VarInit:
        initialization = cast(llir.VarInit, value)
        if initialization.var.name.startswith(("_cnt", "_pos", "_prev")):
            statements.append(initialization)
    elif type(value) is llir.Increment:
        increment = cast(llir.Increment, value)
        if increment.var.name.startswith(("_cnt", "_pos")):
            statements.append(increment)
    elif type(value) is llir.Assign:
        assignment = cast(llir.Assign, value)
        if type(assignment.var) is llir.Var and assignment.var.name.startswith("_prev"):
            statements.append(assignment)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            statements.extend(_phase_state_statements(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            statements.extend(_phase_state_statements(child))
    return statements


def _phase_state_codes(value: object) -> List[str]:
    return [
        LLIRLowerer().lower_llir(statement).removesuffix(";")
        for statement in _phase_state_statements(value)
    ]


def _phase_state_vars(value: object) -> List[llir.Var]:
    if type(value) is llir.Var:
        variable = cast(llir.Var, value)
        if re.fullmatch(r"_(?:cnt|pos|prev)\d+", variable.name):
            return [variable]
        return []
    variables: List[llir.Var] = []
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            variables.extend(_phase_state_vars(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            variables.extend(_phase_state_vars(child))
    return variables


def _base_offset_loads(value: object) -> List[llir.VarInit]:
    loads: List[llir.VarInit] = []
    if type(value) is llir.VarInit:
        initializer = cast(llir.VarInit, value)
        access = initializer.value
        if (
            re.fullmatch(r"_base\d+", initializer.var.name)
            and type(access) is llir.ArrayAccess
            and type(cast(llir.ArrayAccess, access).array) is llir.Var
            and re.fullmatch(
                r"_offset\d+",
                cast(llir.Var, cast(llir.ArrayAccess, access).array).name,
            )
        ):
            loads.append(initializer)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            loads.extend(_base_offset_loads(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            loads.extend(_base_offset_loads(child))
    return loads


def _call_names(value: object) -> List[str]:
    names: List[str] = []
    if type(value) is llir.FunctionCallStmt:
        names.append(cast(llir.FunctionCallStmt, value).name)
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            names.extend(_call_names(child))
    elif type(value) is list or type(value) is tuple:
        for child in value:
            names.extend(_call_names(child))
    return names


def _phase_loops(result: CompressedWhereOpenMPResult) -> Tuple[llir.ForLoop, ...]:
    loops = tuple(
        cast(llir.ForLoop, statement)
        for statement in result.statements
        if type(statement) is llir.ForLoop and statement.omp_parallel_for
    )
    assert len(loops) == 2
    return loops


def test_policy_context_and_result_are_frozen() -> None:
    policy = CompressedWhereOpenMPPolicy()
    context = _context(policy=policy)
    result = CompressedWhereOpenMPResult(statements=[], applied=False)

    with pytest.raises(FrozenInstanceError):
        policy.omp_schedule = "static"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        context.result_name = "Other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.applied = True  # type: ignore[misc]


def test_ds_transform_builds_exact_count_fill_allocation_and_policy() -> None:
    source: List[llir.Stmt] = [_compatible_loop(_ds_work_body())]

    result = transform_compressed_where_for_openmp(source, _context())

    assert result.applied is True
    count_loop, fill_loop = _phase_loops(result)
    count_codes = _raw_codes(count_loop.body)
    fill_codes = _raw_codes(fill_loop.body)
    count_state_codes = _phase_state_codes(count_loop.body)
    fill_state_codes = _phase_state_codes(fill_loop.body)
    count_assignment_codes = _assignment_codes(count_loop.body)
    fill_assignment_codes = _assignment_codes(fill_loop.body)
    all_assignment_codes = _assignment_codes(result.statements)
    top_level_codes = [
        statement.code
        for statement in result.statements
        if type(statement) is llir.RawStmt
    ]

    assert "int _cnt1 = 0" in count_state_codes
    assert "_cnt1++" in count_state_codes
    assert "_count1[row] = _cnt1" in count_assignment_codes
    assert "_count1[row] = _cnt1" not in count_codes
    assert "wksp.clear()" in count_codes
    base_load_codes = [
        LLIRLowerer().lower_llir(load).removesuffix(";")
        for load in _base_offset_loads(fill_loop.body)
    ]
    assert base_load_codes == ["int64_t _base1 = _offset1[row]"]
    assert "int64_t _base1 = _offset1[row]" not in fill_codes
    assert "int _pos1 = 0" in fill_state_codes
    assert "Result1_crd_data[_base1 + _pos1] = column" in fill_assignment_codes
    assert "Result1_crd_data[_base1 + _pos1] = column" not in fill_codes
    assert "_pos1++" in fill_state_codes
    assert "Result_values_data[_base1 + _pos1] = value" in fill_assignment_codes
    assert "Result_values_data[_base1 + _pos1] = value" not in fill_codes
    assert "wksp.clear()" in fill_codes
    assert "_offset1[0] = 0" in all_assignment_codes
    assert "_offset1[0] = 0" not in _raw_codes(result.statements)
    assert all(
        type(assignment.var) is llir.ArrayAccess
        for assignment in _assignments(count_loop.body) + _assignments(fill_loop.body)
    )
    assert _call_names(count_loop.body).count("wksp.insert_unchecked") == 1
    assert _call_names(fill_loop.body).count("wksp.insert_unchecked") == 1

    assert "std::vector<int> _count1((size_t)A0_size, 0)" in top_level_codes
    assert "std::vector<int64_t> _offset1((size_t)A0_size + 1)" in top_level_codes
    assert "int64_t _total1 = _offset1[A0_size]" in top_level_codes
    assert any("torch::Tensor Result1_pos_torch" in code for code in top_level_codes)
    assert any("torch::Tensor Result1_crd_torch" in code for code in top_level_codes)
    assert any("torch::Tensor Result_values_torch" in code for code in top_level_codes)
    assert any(
        "Tensor Result;" in code and "return Result;" in code
        for code in top_level_codes
    )

    flop = (
        "(long)A1_pos[A0_size] * (B0_size > 0 ? " "(B1_pos[B0_size] / B0_size) + 1 : 1)"
    )
    for loop in (count_loop, fill_loop):
        assert loop.omp_schedule == "dynamic, 64"
        assert loop.omp_num_threads == (
            f"scorch_nthreads({flop}, A0_size, " "SCORCH_GRAIN_CODEGEN_SPGEMM)"
        )
        assert loop.omp_chunk_expr == (
            f"scorch_chunk(A0_size, {flop}, SCORCH_GRAIN_CODEGEN_SPGEMM)"
        )


def test_structured_position_bounds_drive_the_exact_spgemm_policy() -> None:
    source: List[llir.Stmt] = [_compatible_loop(_structured_ds_work_body())]

    result = transform_compressed_where_for_openmp(source, _context())

    flop = (
        "(long)A1_pos[A0_size] * (B0_size > 0 ? " "(B1_pos[B0_size] / B0_size) + 1 : 1)"
    )
    for loop in _phase_loops(result):
        assert loop.omp_num_threads == (
            f"scorch_nthreads({flop}, A0_size, SCORCH_GRAIN_CODEGEN_SPGEMM)"
        )
        assert loop.omp_chunk_expr == (
            f"scorch_chunk(A0_size, {flop}, SCORCH_GRAIN_CODEGEN_SPGEMM)"
        )


def test_dss_transform_builds_each_compressed_boundary_once() -> None:
    boundary = llir.IfThenElse(
        cond=llir.BinOp(
            "<",
            llir.FunctionCall("Result2_pos.back", []),
            _var("pResult2", llir.DataType.INT64),
        ),
        then_body=[
            llir.FunctionCallStmt("Result1_crd.push_back", [_var("parent_coordinate")])
        ],
    )
    body: List[llir.Stmt] = [
        _workspace_init(),
        llir.FunctionCallStmt("Result1_crd.push_back", [_var("row_coordinate")]),
        boundary,
        llir.FunctionCallStmt("Result2_crd.push_back", [_var("leaf_coordinate")]),
        llir.FunctionCallStmt("Result_values.push_back", [_var("value")]),
    ]

    result = transform_compressed_where_for_openmp(
        [_compatible_loop(body)], _context((1, 2))
    )

    count_loop, fill_loop = _phase_loops(result)
    count_state_codes = _phase_state_codes(count_loop.body)
    all_codes = _raw_codes(result.statements)
    count_boundary = cast(
        llir.IfThenElse,
        next(
            statement
            for statement in count_loop.body
            if type(statement) is llir.IfThenElse
        ),
    )
    fill_boundary = cast(
        llir.IfThenElse,
        next(
            statement
            for statement in fill_loop.body
            if type(statement) is llir.IfThenElse
        ),
    )

    assert "int _cnt1 = 0" in count_state_codes
    assert "int _cnt2 = 0" in count_state_codes
    assert "int _prev2 = 0" in count_state_codes
    assert cast(llir.BinOp, count_boundary.cond).op == ">"
    assert _phase_state_codes(count_boundary.then_body) == [
        "_cnt1++",
        "_prev2 = _cnt2",
    ]
    assert cast(llir.BinOp, fill_boundary.cond).op == ">"
    assert _phase_state_codes(fill_boundary.then_body) == [
        "_pos1++",
        "_prev2 = _pos2",
    ]
    assert [
        code
        for code in _assignment_codes(fill_boundary.then_body)
        if not code.startswith("_prev")
    ] == [
        "Result1_crd_data[_base1 + _pos1] = parent_coordinate",
        "Result2_pos_data[_base1 + _pos1] = _base2 + _pos2",
    ]
    assert "Result2_crd_data[_base2 + _pos2] = leaf_coordinate" in _assignment_codes(
        fill_loop.body
    )
    assert "Result2_pos_data[0] = 0" in _assignment_codes(result.statements)
    assert "Result2_pos_data[0] = 0" not in all_codes
    assert _assignment_codes(result.statements).count("Result2_pos_data[0] = 0") == 1
    assert any(
        "{{}, {Result1_pos_torch, Result1_crd_torch}, "
        "{Result2_pos_torch, Result2_crd_torch}}" in code
        for code in all_codes
    )


def test_count_fill_state_is_typed_structural_fresh_and_never_raw() -> None:
    boundary = llir.IfThenElse(
        cond=llir.BinOp(
            "<",
            llir.FunctionCall("Result2_pos.back"),
            _var("pResult2", llir.DataType.INT64),
        ),
        then_body=[
            llir.FunctionCallStmt(
                "Result1_crd.push_back",
                [_var("parent_coordinate")],
            )
        ],
    )
    source: List[llir.Stmt] = [
        _compatible_loop(
            [
                _workspace_init(),
                llir.FunctionCallStmt(
                    "Result1_crd.push_back",
                    [_var("row_coordinate")],
                ),
                boundary,
                llir.FunctionCallStmt(
                    "Result2_crd.push_back",
                    [_var("leaf_coordinate")],
                ),
            ]
        )
    ]
    snapshot = _structural_snapshot(source)

    first = transform_compressed_where_for_openmp(source, _context((1, 2)))
    second = transform_compressed_where_for_openmp(source, _context((1, 2)))
    first_state = _phase_state_statements(first.statements)
    second_state = _phase_state_statements(second.statements)
    first_state_vars = _phase_state_vars(first.statements)
    second_state_vars = _phase_state_vars(second.statements)

    assert _structural_snapshot(source) == snapshot
    assert first_state == second_state
    assert _mutable_ir_ids(first_state).isdisjoint(_mutable_ir_ids(second_state))
    assert first_state_vars == second_state_vars
    assert {id(var) for var in first_state_vars}.isdisjoint(
        {id(var) for var in second_state_vars}
    )
    assert _phase_state_codes(first.statements) == _phase_state_codes(second.statements)
    assert not any(
        re.search(r"\b_(?:cnt|pos|prev)\d+\b", code)
        for code in _raw_codes(first.statements)
    )

    for statement in first_state:
        if type(statement) is llir.VarInit:
            initialization = cast(llir.VarInit, statement)
            assert type(initialization.var) is llir.Var
            assert initialization.var.type is llir.DataType.INT
            assert type(initialization.value) is llir.Literal
            literal = cast(llir.Literal, initialization.value)
            assert literal.value == 0
            assert literal.data_type is llir.DataType.INT
        elif type(statement) is llir.Increment:
            increment = cast(llir.Increment, statement)
            assert type(increment.var) is llir.Var
            assert increment.var.type is llir.DataType.INT
        else:
            assignment = cast(llir.Assign, statement)
            assert type(assignment.var) is llir.Var
            assert type(assignment.value) is llir.Var
            assert assignment.var.type is llir.DataType.INT
            assert assignment.value.type is llir.DataType.INT
            assert assignment.var is not assignment.value

    assert all(type(var) is llir.Var for var in first_state_vars)
    assert all(var.type is llir.DataType.INT for var in first_state_vars)
    assert Counter(var.name for var in first_state_vars) == {
        "_cnt1": 4,
        "_cnt2": 5,
        "_pos1": 7,
        "_pos2": 7,
        "_prev2": 6,
    }
    assert len({id(var) for var in first_state_vars}) == len(first_state_vars)


def test_fill_base_offset_loads_are_typed_owned_structural_and_never_raw() -> None:
    source: List[llir.Stmt] = [_compatible_loop(_ds_work_body(), bound="A0_size")]
    source_header = cast(llir.VarInit, cast(llir.ForLoop, source[0]).init).var
    snapshot = _structural_snapshot(source)

    first = transform_compressed_where_for_openmp(source, _context((1, 2)))
    second = transform_compressed_where_for_openmp(source, _context((1, 2)))
    first_loads = _base_offset_loads(first.statements)
    second_loads = _base_offset_loads(second.statements)

    assert _structural_snapshot(source) == snapshot
    assert [load.var.name for load in first_loads] == ["_base1", "_base2"]
    assert first_loads == second_loads
    assert [hash(load) for load in first_loads] == [hash(load) for load in second_loads]
    assert _mutable_ir_ids(first_loads).isdisjoint(_mutable_ir_ids(second_loads))
    assert [LLIRLowerer().lower_llir(load) for load in first_loads] == [
        "int64_t _base1 = _offset1[row];",
        "int64_t _base2 = _offset2[row];",
    ]
    assert not any(
        re.search(r"int64_t _base\d+ = _offset\d+\[", code)
        for code in _raw_codes(first.statements)
    )

    for level, initializer in enumerate(first_loads, start=1):
        assert type(initializer) is llir.VarInit
        assert type(initializer.var) is llir.Var
        assert initializer.var.name == f"_base{level}"
        assert initializer.var.type is llir.DataType.INT64
        assert initializer.var.is_ptr is False
        assert initializer.var.is_restrict is False
        assert initializer.var.tensor_access is None
        assert initializer.op == "="
        assert initializer.cast is False

        assert type(initializer.value) is llir.ArrayAccess
        access = cast(llir.ArrayAccess, initializer.value)
        assert access.tensor_access is None
        assert type(access.array) is llir.Var
        offset = cast(llir.Var, access.array)
        assert offset.name == f"_offset{level}"
        assert offset.type is llir.DataType.STD_VECTOR_INT
        assert offset.is_ptr is False
        assert offset.is_restrict is False
        assert offset.tensor_access is None
        assert type(access.index) is llir.Var
        index = cast(llir.Var, access.index)
        assert index.name == "row"
        assert index.type is llir.DataType.INT
        assert index.is_ptr is False
        assert index.is_restrict is False
        assert index.tensor_access is None
        assert index is not source_header
        assert len({id(initializer.var), id(offset), id(index)}) == 3

        expected = llir.VarInit(
            llir.Var(f"_base{level}", llir.DataType.INT64),
            llir.ArrayAccess(
                llir.Var(f"_offset{level}", llir.DataType.STD_VECTOR_INT),
                llir.Var("row", llir.DataType.INT),
            ),
        )
        assert initializer == expected
        assert hash(initializer) == hash(expected)

    first_access = cast(llir.ArrayAccess, first_loads[0].value)
    cast(llir.Var, first_access.array).name = "owned_offset"
    cast(llir.Var, first_access.index).name = "owned_index"
    assert cast(llir.Var, cast(llir.ArrayAccess, second_loads[0].value).array).name == (
        "_offset1"
    )
    assert cast(llir.Var, cast(llir.ArrayAccess, second_loads[0].value).index).name == (
        "row"
    )
    assert source_header.name == "row"


def test_first_top_level_compatible_loop_is_selected_and_suffix_is_discarded() -> None:
    auto_prefix = llir.ForLoopAuto(
        _var("item"), _var("items"), [llir.RawStmt("keep_auto_prefix")]
    )
    incompatible = _compatible_loop(
        [llir.RawStmt("keep_incompatible")],
        bound="Ignored0_size",
    )
    cast(llir.VarInit, incompatible.init).var.name = "other"
    first = _compatible_loop([llir.RawStmt("first_work")], bound="First0_size")
    second = _compatible_loop([llir.RawStmt("second_work")], bound="Second0_size")
    source: List[llir.Stmt] = [
        auto_prefix,
        incompatible,
        first,
        second,
        llir.RawStmt("suffix"),
    ]

    result = transform_compressed_where_for_openmp(source, _context())

    count_loop, fill_loop = _phase_loops(result)
    assert cast(llir.Var, cast(llir.BinOp, count_loop.cond).right).name == (
        "First0_size"
    )
    assert cast(llir.Var, cast(llir.BinOp, fill_loop.cond).right).name == (
        "First0_size"
    )
    codes = _raw_codes(result.statements)
    assert "keep_auto_prefix" in codes
    assert "keep_incompatible" in codes
    assert codes.count("first_work") == 2
    assert "second_work" not in codes
    assert "suffix" not in codes


@pytest.mark.parametrize("nested", [False, True])
def test_no_top_level_loop_or_immediate_unextractable_bound_is_detached_no_op(
    nested: bool,
) -> None:
    valid_later = _compatible_loop([llir.RawStmt("later")], bound="Later0_size")
    if nested:
        source: List[llir.Stmt] = [
            llir.IfThenElse(cond=_var("guard"), then_body=[valid_later])
        ]
    else:
        source = [
            _compatible_loop(
                [llir.RawStmt("first")],
                bound="First0_size",
                cond_op="<=",
            ),
            valid_later,
        ]
    snapshot = _structural_snapshot(source)

    result = transform_compressed_where_for_openmp(source, _context())

    assert result.applied is False
    assert _structural_snapshot(result.statements) == snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(result.statements))


def test_legacy_prefix_and_work_body_filters_are_top_level_only() -> None:
    assembly_loop = _compatible_loop(
        [llir.RawStmt("drop_assembly_loop")], bound="limit"
    )
    assembly_loop.cond = llir.BinOp("<", _var("Result1_pos_index"), _var("limit"))
    position_loop = _compatible_loop([llir.RawStmt("drop_position_loop")])
    cast(llir.VarInit, position_loop.init).var.name = "pResult1"
    nested = llir.IfThenElse(
        cond=_var("guard"),
        then_body=[llir.RawStmt("keep_nested")],
    )
    selected = _compatible_loop(
        [
            _workspace_init(),
            assembly_loop,
            position_loop,
            llir.Assign(
                llir.ArrayAccess(_var("Result1_pos"), _var("row")),
                _var("pResult1"),
            ),
            llir.RawStmt("keep_work"),
            nested,
        ]
    )
    prefix_position_loop = _compatible_loop([llir.RawStmt("drop_prefix_loop")])
    cast(llir.VarInit, prefix_position_loop.init).var.name = "pResultPrefix"
    source = [
        llir.Comment("keep-prefix"),
        llir.VarDecl(_var("Result_values")),
        llir.VarDecl(_var("Result1_pos")),
        llir.VarDecl(_var("Result1_crd")),
        llir.VarInit(_var("pResult1"), llir.Literal(0)),
        llir.VarInit(_var("Result1_pos_index"), llir.Literal(0)),
        llir.Assign(
            llir.ArrayAccess(_var("Result1_pos"), llir.Literal(0)),
            llir.Literal(0),
        ),
        prefix_position_loop,
        selected,
        llir.RawStmt("drop_suffix"),
    ]

    result = transform_compressed_where_for_openmp(source, _context())

    codes = _raw_codes(result.statements)
    assert codes.count("keep_work") == 2
    assert codes.count("keep_nested") == 2
    assert "drop_assembly_loop" not in codes
    assert "drop_position_loop" not in codes
    assert "drop_prefix_loop" not in codes
    assert "drop_suffix" not in codes
    assert [
        statement.value
        for statement in result.statements
        if type(statement) is llir.Comment
    ] == ["keep-prefix"]
    assert not any(
        type(statement) is llir.VarDecl
        and statement.var.name in {"Result_values", "Result1_pos", "Result1_crd"}
        for statement in result.statements
    )


def test_nested_control_flow_and_statement_containers_follow_legacy_scopes() -> None:
    def writes(marker: str) -> List[llir.Stmt]:
        return [
            llir.FunctionCallStmt("Result1_crd.push_back", [_var(marker)]),
            llir.FunctionCallStmt("wksp.insert", [_var(marker)]),
        ]

    nested_for = _compatible_loop(writes("for_value"), bound="inner_limit")
    auto_loop = llir.ForLoopAuto(_var("item"), _var("items"), writes("auto_value"))
    while_loop = llir.WhileLoop(_var("keep_going"), writes("while_value"))
    conditional = llir.IfThenElse(
        cond=_var("guard"),
        then_body=writes("then_value"),
        else_body=writes("else_value"),
        cond_list=[_var("other_guard")],
        then_body_list=[writes("branch_value")],
    )
    nested_list: List[LLIRStatementValue] = cast(
        List[LLIRStatementValue], writes("list_value")
    )
    nested_tuple = tuple(writes("tuple_value"))
    body = cast(
        List[llir.Stmt],
        [
            _workspace_init(),
            nested_for,
            auto_loop,
            while_loop,
            conditional,
            nested_list,
            nested_tuple,
        ],
    )

    result = transform_compressed_where_for_openmp([_compatible_loop(body)], _context())

    count_loop, _ = _phase_loops(result)
    count_state_codes = _phase_state_codes(count_loop.body)
    calls = _call_names(count_loop.body)
    assert count_state_codes.count("_cnt1++") == 8
    # The exact call-name rewrite descends into ForLoop bodies and If branches,
    # but not ForLoopAuto, WhileLoop, or bare nested statement containers.
    assert calls.count("wksp.insert_unchecked") == 4
    assert calls.count("wksp.insert") == 4


def test_workspace_insert_rewrite_matches_only_the_exact_call_name() -> None:
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(1),
        tensor_id=SymbolId(2),
        index_ids=(IndexId(3),),
        role=llir.TensorAccessRole.INPUT_READ,
    )
    assignment = llir.Assign(
        llir.ArrayAccess(_var("wksp.insert_targets"), _var("wksp.insert_target_index")),
        llir.Add(
            _var("wksp.insert_value"),
            llir.ArrayAccess(
                _var("wksp.insert_values"),
                _var("wksp.insert_value_index"),
                tensor_access=metadata,
            ),
        ),
    )
    initialization = llir.VarInit(
        _var("initialized"), _var("wksp.insert_initial_value")
    )
    raw = llir.RawStmt("wksp.insert(raw_value)", add_semicolon=False)
    calls = [
        llir.FunctionCallStmt("wksp.insert", [_var("wksp.insert_argument")]),
        llir.FunctionCallStmt("prefix_wksp.insert", [_var("prefix_argument")]),
        llir.FunctionCallStmt("wksp.insert_suffix", [_var("suffix_argument")]),
        llir.FunctionCallStmt("wksp.insert.more", [_var("member_argument")]),
        llir.FunctionCallStmt("wksp.insert_unchecked", [_var("unchecked_argument")]),
    ]
    source: List[llir.Stmt] = [
        _compatible_loop([_workspace_init(), assignment, initialization, raw, *calls])
    ]
    snapshot = _structural_snapshot(source)

    result = transform_compressed_where_for_openmp(source, _context())
    repeated = transform_compressed_where_for_openmp(source, _context())

    count_loop, fill_loop = _phase_loops(result)
    rewritten_assignment = cast(
        llir.Assign,
        next(
            statement for statement in count_loop.body if type(statement) is llir.Assign
        ),
    )
    target = cast(llir.ArrayAccess, rewritten_assignment.var)
    value = cast(llir.Add, rewritten_assignment.value)
    value_access = cast(llir.ArrayAccess, value.right)
    rewritten_initialization = cast(
        llir.VarInit,
        next(
            statement
            for statement in count_loop.body
            if type(statement) is llir.VarInit and statement.var.name == "initialized"
        ),
    )

    assert cast(llir.Var, target.array).name == "wksp.insert_targets"
    assert cast(llir.Var, target.index).name == "wksp.insert_target_index"
    assert cast(llir.Var, value.left).name == "wksp.insert_value"
    assert cast(llir.Var, value_access.array).name == "wksp.insert_values"
    assert cast(llir.Var, value_access.index).name == "wksp.insert_value_index"
    assert value_access.tensor_access is metadata
    fill_assignment = cast(
        llir.Assign,
        next(
            statement for statement in fill_loop.body if type(statement) is llir.Assign
        ),
    )
    fill_value = cast(llir.Add, fill_assignment.value)
    assert type(value) is llir.Add
    assert type(fill_value) is llir.Add
    assert value == fill_value
    assert value is not assignment.value
    assert fill_value is not assignment.value
    assert fill_value is not value
    assert value.left is not cast(llir.Add, assignment.value).left
    assert value.right is not cast(llir.Add, assignment.value).right
    assert fill_value.left is not value.left
    assert fill_value.right is not value.right
    assert cast(llir.ArrayAccess, fill_value.right).tensor_access is metadata
    assert cast(llir.Var, rewritten_initialization.value).name == (
        "wksp.insert_initial_value"
    )

    expected_names = {
        "wksp.insert_argument": "wksp.insert_unchecked",
        "prefix_argument": "prefix_wksp.insert",
        "suffix_argument": "wksp.insert_suffix",
        "member_argument": "wksp.insert.more",
        "unchecked_argument": "wksp.insert_unchecked",
    }
    for phase_loop in (count_loop, fill_loop):
        phase_calls = [
            cast(llir.FunctionCallStmt, statement)
            for statement in phase_loop.body
            if type(statement) is llir.FunctionCallStmt
        ]
        actual_names = {
            cast(llir.Var, call.args[0]).name: call.name for call in phase_calls
        }
        assert actual_names == expected_names
        assert all(type(call.args[0]) is llir.Var for call in phase_calls)
        rewritten_raw = next(
            cast(llir.RawStmt, statement)
            for statement in phase_loop.body
            if type(statement) is llir.RawStmt
            and statement.code == "wksp.insert(raw_value)"
        )
        assert rewritten_raw.add_semicolon is False

    assert _structural_snapshot(source) == snapshot
    assert _structural_snapshot(result.statements) == _structural_snapshot(
        repeated.statements
    )
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(result.statements))
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(repeated.statements))
    assert _mutable_ir_ids(result.statements).isdisjoint(
        _mutable_ir_ids(repeated.statements)
    )


@pytest.mark.parametrize(
    "name",
    [None, "", "   ", type("WorkspaceCallName", (str,), {})("wksp.insert")],
    ids=["none", "empty", "whitespace", "str-subclass"],
)
def test_workspace_insert_rewrite_rejects_malformed_call_names(
    name: object,
) -> None:
    malformed = llir.FunctionCallStmt(cast(str, name), [_var("value")])
    source = [_compatible_loop([_workspace_init(), malformed])]
    snapshot = _structural_snapshot(source)

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, _context())

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_workspace_insert_call_name"
    assert diagnostic.stage == _context().traversal.stage
    assert diagnostic.pass_name == _context().traversal.pass_name
    assert diagnostic.path == ("root", "[0]", "name")
    assert _structural_snapshot(source) == snapshot


@pytest.mark.parametrize(
    ("args", "expected_code", "expected_path"),
    [
        (
            object(),
            "invalid_expression_sequence",
            ("root", "[0]", "body", "[1]", "args"),
        ),
        (
            ["not_an_expression"],
            "invalid_expression_sequence_member",
            ("root", "[0]", "body", "[1]", "args", "[0]"),
        ),
    ],
    ids=["container", "member"],
)
def test_workspace_insert_rewrite_rejects_malformed_call_args(
    args: object,
    expected_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    malformed = llir.FunctionCallStmt("wksp.insert", [])
    malformed.args = cast(List[llir.Expr], args)
    source = [_compatible_loop([_workspace_init(), malformed])]
    snapshot = _structural_snapshot(source)

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, _context())

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == expected_code
    assert diagnostic.stage == _context().traversal.stage
    assert diagnostic.pass_name == _context().traversal.pass_name
    assert diagnostic.path == expected_path
    assert _structural_snapshot(source) == snapshot


def test_retained_metadata_is_preserved_but_selected_loop_policy_is_reset() -> None:
    inner = _compatible_loop([llir.RawStmt("inner")], bound="inner_limit")
    inner.omp_parallel_for = True
    inner.omp_schedule = "static, 2"
    inner.unroll = True
    inner.simd = True
    inner.before_parallel_body = [llir.RawStmt("inner_before")]
    inner.pre_parallel_body = [llir.RawStmt("inner_pre")]
    inner.post_parallel_body = [llir.RawStmt("inner_post")]
    inner.omp_num_threads = "inner_threads"
    inner.omp_chunk_expr = "inner_chunk"
    inner.scorch_index_var = "inner_tag"
    setattr(inner, "_use_atomic_scheduling", True)
    setattr(inner, "_hoisted_ptr_decls", [llir.RawStmt("inner_hoisted")])

    selected = _compatible_loop([inner])
    selected.omp_parallel_for = True
    selected.omp_schedule = "static"
    selected.unroll = True
    selected.simd = True
    selected.before_parallel_body = [llir.RawStmt("selected_before")]
    selected.pre_parallel_body = [llir.RawStmt("selected_pre")]
    selected.post_parallel_body = [llir.RawStmt("selected_post")]
    selected.scorch_index_var = "selected_tag"
    setattr(selected, "_hoisted_ptr_decls", [llir.RawStmt("selected_hoisted")])

    result = transform_compressed_where_for_openmp([selected], _context())

    count_loop, fill_loop = _phase_loops(result)
    for phase in (count_loop, fill_loop):
        rewritten_inner = cast(
            llir.ForLoop,
            next(
                statement for statement in phase.body if type(statement) is llir.ForLoop
            ),
        )
        assert rewritten_inner.omp_parallel_for is True
        assert rewritten_inner.omp_schedule == "static, 2"
        assert rewritten_inner.unroll is True
        assert rewritten_inner.simd is True
        assert rewritten_inner.omp_num_threads == "inner_threads"
        assert rewritten_inner.omp_chunk_expr == "inner_chunk"
        assert rewritten_inner.scorch_index_var == "inner_tag"
        assert _raw_codes(rewritten_inner.before_parallel_body) == ["inner_before"]
        assert _raw_codes(rewritten_inner.pre_parallel_body) == ["inner_pre"]
        assert _raw_codes(rewritten_inner.post_parallel_body) == ["inner_post"]
        assert getattr(rewritten_inner, "_use_atomic_scheduling") is True
        assert _raw_codes(getattr(rewritten_inner, "_hoisted_ptr_decls")) == [
            "inner_hoisted"
        ]

        # The selected serial loop is replaced, not rebuilt in place. Its old
        # region metadata is intentionally not copied to either generated phase.
        assert phase.omp_parallel_for is True
        assert phase.omp_schedule == "dynamic, 64"
        assert phase.unroll is False
        assert phase.simd is False
        assert phase.before_parallel_body is None
        assert phase.pre_parallel_body is None
        assert phase.post_parallel_body is None
        assert phase.scorch_index_var is None
        assert not hasattr(phase, "_hoisted_ptr_decls")


@pytest.mark.parametrize(
    ("body", "update", "expected_work", "expected_rows", "expected_grain"),
    [
        (
            _ds_work_body(workspace=False, both_operands=False),
            None,
            "A1_pos[A0_size]",
            "A0_size",
            False,
        ),
        (
            [llir.RawStmt("work")],
            llir.Assign(
                _var("row", llir.DataType.INT),
                llir.Literal(4),
                op=llir.AssignOp.ADD_ASSIGN,
            ),
            "-1",
            "((A0_size + 4 - 1) / 4)",
            False,
        ),
    ],
)
def test_parallel_policy_fallbacks_preserve_work_and_trip_count_decisions(
    body: List[llir.Stmt],
    update: llir.Assign | None,
    expected_work: str,
    expected_rows: str,
    expected_grain: bool,
) -> None:
    loop = _compatible_loop(body, update=update)

    result = transform_compressed_where_for_openmp([loop], _context())

    for phase in _phase_loops(result):
        assert phase.omp_num_threads == (
            f"scorch_nthreads({expected_work}, {expected_rows})"
        )
        assert phase.omp_chunk_expr == (
            f"scorch_chunk({expected_rows}, {expected_work})"
        )
        assert ("SCORCH_GRAIN" in phase.omp_num_threads) is expected_grain


def test_custom_parallel_policy_is_an_explicit_context_input() -> None:
    policy = CompressedWhereOpenMPPolicy(
        omp_schedule="guided, 7",
        flop_grain="CUSTOM_FLOP_GRAIN",
    )

    result = transform_compressed_where_for_openmp(
        [_compatible_loop(_ds_work_body())],
        _context(policy=policy),
    )

    for phase in _phase_loops(result):
        assert phase.omp_schedule == "guided, 7"
        assert "CUSTOM_FLOP_GRAIN" in cast(str, phase.omp_num_threads)
        assert "CUSTOM_FLOP_GRAIN" in cast(str, phase.omp_chunk_expr)


@pytest.mark.parametrize(
    ("workspace_ctype", "torch_dtype", "pointer_type"),
    [
        ("float", "torch::kFloat32", llir.DataType.PTR_FLOAT32),
        ("double", "torch::kFloat64", llir.DataType.PTR_FLOAT64),
        ("int", "torch::kInt32", llir.DataType.PTR_INT),
        ("int32_t", "torch::kInt32", llir.DataType.PTR_INT_32),
        ("int64_t", "torch::kInt64", llir.DataType.PTR_INT_64),
        ("custom_scalar", "torch::kFloat32", llir.DataType.NO_TYPE),
    ],
)
def test_workspace_ctype_explicitly_controls_value_allocation(
    workspace_ctype: str,
    torch_dtype: str,
    pointer_type: llir.DataType,
) -> None:
    context = CompressedWhereOpenMPContext(
        result_name="Result",
        result_id=SymbolId(1),
        compressed_levels=(1,),
        workspace_name="wksp",
        workspace_ctype=workspace_ctype,
    )

    result = transform_compressed_where_for_openmp(
        [_compatible_loop(_ds_work_body())], context
    )

    allocations = [
        code
        for code in _raw_codes(result.statements)
        if "torch::Tensor Result_values_torch" in code
    ]
    assert allocations == [
        "torch::Tensor Result_values_torch = "
        f"torch::empty({{(long long)_total1}}, {torch_dtype});\n"
        f"  {workspace_ctype}* Result_values_data = "
        f"Result_values_torch.data_ptr<{workspace_ctype}>();"
    ]
    value_targets = [
        cast(llir.ArrayAccess, assignment.var)
        for assignment in _assignments(result.statements)
        if type(assignment.var) is llir.ArrayAccess
        and type(cast(llir.ArrayAccess, assignment.var).array) is llir.Var
        and cast(llir.Var, cast(llir.ArrayAccess, assignment.var).array).name
        == "Result_values_data"
    ]
    assert value_targets
    assert all(
        cast(llir.Var, target.array).type is pointer_type for target in value_targets
    )


def test_transform_does_not_mutate_or_alias_caller_owned_llir() -> None:
    source: List[llir.Stmt] = [_compatible_loop(_ds_work_body())]
    snapshot = _structural_snapshot(source)

    result = transform_compressed_where_for_openmp(source, _context())

    assert _structural_snapshot(source) == snapshot
    assert _mutable_ir_ids(source).isdisjoint(_mutable_ir_ids(result.statements))
    original_loop = cast(llir.ForLoop, source[0])
    original_insert = cast(llir.FunctionCallStmt, original_loop.body[3])
    assert original_insert.name == "wksp.insert"

    result.statements.append(llir.Break())
    assert len(source) == 1


def test_unknown_node_fails_with_pass_specific_structured_diagnostic() -> None:
    class UnknownStmt(llir.Stmt):
        pass

    source = [
        _compatible_loop([llir.RawStmt("work")]),
        UnknownStmt(),
    ]
    context = _context()

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(source, context)

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "unknown_llir_node"
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.node_type == "UnknownStmt"
    assert diagnostic.path == ("root", "[1]")


def test_malformed_typed_child_fails_with_pass_specific_diagnostic() -> None:
    malformed = _compatible_loop([])
    malformed.body = cast(List[llir.Stmt], [_var("not_a_statement")])
    context = _context()

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp([malformed], context)

    diagnostic = raised.value.diagnostic
    assert diagnostic.code == "invalid_statement_sequence_member"
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.path == ("root", "[0]", "body", "[0]")


@pytest.mark.parametrize(
    ("malformation", "expected_code", "expected_suffix"),
    (
        ("unknown_array", "unknown_llir_node", ("value", "array")),
        ("unknown_index", "unknown_llir_node", ("value", "index")),
        ("access_subclass", "unknown_llir_node", ("value",)),
        (
            "forged_metadata",
            "invalid_tensor_access_metadata",
            ("value", "tensor_access"),
        ),
    ),
)
def test_generated_fill_base_loads_fail_closed_at_compressed_owner(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    expected_code: str,
    expected_suffix: Tuple[str, ...],
) -> None:
    class UnknownExpr(llir.Expr):
        pass

    class UnknownArrayAccess(llir.ArrayAccess):
        pass

    injected: List[str] = []

    class InjectingWalker(LLIRWalker):
        def walk(self, value: LLIRValue) -> None:
            loads = _base_offset_loads(value)
            if loads and not injected:
                initializer = loads[0]
                access = cast(llir.ArrayAccess, initializer.value)
                if malformation == "unknown_array":
                    initializer.value = llir.ArrayAccess(
                        UnknownExpr(),
                        access.index,
                    )
                elif malformation == "unknown_index":
                    initializer.value = llir.ArrayAccess(
                        access.array,
                        UnknownExpr(),
                    )
                elif malformation == "access_subclass":
                    unknown = object.__new__(UnknownArrayAccess)
                    object.__setattr__(unknown, "array", access.array)
                    object.__setattr__(unknown, "index", access.index)
                    object.__setattr__(unknown, "tensor_access", None)
                    initializer.value = unknown
                else:
                    forged = object.__new__(llir.ArrayAccess)
                    object.__setattr__(forged, "array", access.array)
                    object.__setattr__(forged, "index", access.index)
                    object.__setattr__(forged, "tensor_access", object())
                    initializer.value = forged
                injected.append(initializer.var.name)
            super().walk(value)

    monkeypatch.setattr(compressed_where_module, "LLIRWalker", InjectingWalker)
    context = _context()

    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(
            [_compatible_loop(_ds_work_body())],
            context,
        )

    diagnostic = raised.value.diagnostic
    assert injected == ["_base1"]
    assert diagnostic.code == expected_code
    assert diagnostic.stage == context.traversal.stage
    assert diagnostic.pass_name == context.traversal.pass_name
    assert diagnostic.path[-len(expected_suffix) :] == expected_suffix


@pytest.mark.parametrize(
    ("context", "expected_code", "expected_path"),
    [
        (
            cast(CompressedWhereOpenMPContext, object()),
            "invalid_compressed_where_context",
            ("context",),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="",
                result_id=SymbolId(1),
                compressed_levels=(1,),
                workspace_name="wksp",
                workspace_ctype="float",
            ),
            "invalid_compressed_where_result_name",
            ("context", "result_name"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=cast(SymbolId, object()),
                compressed_levels=(1,),
                workspace_name="wksp",
                workspace_ctype="float",
            ),
            "invalid_compressed_where_result_id",
            ("context", "result_id"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(),
                workspace_name="wksp",
                workspace_ctype="float",
            ),
            "invalid_compressed_where_levels",
            ("context", "compressed_levels"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(2,),
                workspace_name="wksp",
                workspace_ctype="float",
            ),
            "unsupported_compressed_where_layout",
            ("context", "compressed_levels"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(1,),
                workspace_name="",
                workspace_ctype="float",
            ),
            "invalid_compressed_where_workspace_name",
            ("context", "workspace_name"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(1,),
                workspace_name="wksp",
                workspace_ctype="",
            ),
            "invalid_compressed_where_workspace_ctype",
            ("context", "workspace_ctype"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(1,),
                workspace_name="wksp",
                workspace_ctype="float",
                policy=CompressedWhereOpenMPPolicy(omp_schedule="", flop_grain="grain"),
            ),
            "invalid_compressed_where_schedule",
            ("context", "policy.omp_schedule"),
        ),
        (
            CompressedWhereOpenMPContext(
                result_name="Result",
                result_id=SymbolId(1),
                compressed_levels=(1,),
                workspace_name="wksp",
                workspace_ctype="float",
                traversal=LLIRTraversalContext(stage="", pass_name="pass"),
            ),
            "invalid_compressed_where_traversal_context",
            ("context", "traversal"),
        ),
    ],
)
def test_invalid_contexts_fail_structurally(
    context: CompressedWhereOpenMPContext,
    expected_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp([llir.Break()], context)

    assert raised.value.diagnostic.code == expected_code
    assert raised.value.diagnostic.path == expected_path


@pytest.mark.parametrize(
    ("root", "expected_code", "expected_path"),
    [
        (
            cast(List[llir.Stmt], (llir.Break(),)),
            "unsupported_compressed_where_root",
            ("root",),
        ),
        (
            cast(List[llir.Stmt], [llir.Break(), _var("not_a_statement")]),
            "invalid_compressed_where_root_member",
            ("root", "[1]"),
        ),
    ],
)
def test_invalid_roots_fail_structurally(
    root: List[llir.Stmt],
    expected_code: str,
    expected_path: Tuple[str, ...],
) -> None:
    with pytest.raises(LLIRTraversalError) as raised:
        transform_compressed_where_for_openmp(root, _context())

    assert raised.value.diagnostic.code == expected_code
    assert raised.value.diagnostic.path == expected_path


def test_successful_transform_is_single_use_and_not_idempotent() -> None:
    context = _context()
    first = transform_compressed_where_for_openmp(
        [_compatible_loop(_ds_work_body())], context
    )

    second = transform_compressed_where_for_openmp(first.statements, context)

    assert first.applied is True
    assert second.applied is True
    assert _structural_snapshot(second.statements) != _structural_snapshot(
        first.statements
    )
    assert _mutable_ir_ids(first.statements).isdisjoint(
        _mutable_ir_ids(second.statements)
    )


def test_production_ds_generated_cpp_matches_pre_extraction_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("SparseProduct", fmt="ds")
    left = TensorVar("SparseLeft", fmt="ds")
    right = TensorVar("SparseRight", fmt="ds")
    assignment = TensorAssign(
        result[row, column],
        left[row, reduction] * right[reduction, column],
        op=Operation.ADD,
    )
    cin = cast(
        ForAll,
        Scheduler.auto_schedule(
            ForAll(row, ForAll(reduction, ForAll(column, assignment)))
        ),
    )

    def reject_ordinary_final_assembly(
        assembler: ResultTensorAssembler,
    ) -> List[llir.Stmt]:
        del assembler
        raise AssertionError(
            "compressed-output ABI must not run ordinary final assembly"
        )

    monkeypatch.setattr(
        ResultTensorAssembler,
        "emit_final_assembly",
        reject_ordinary_final_assembly,
    )
    lowerer = CINLowerer()
    lowered = lowerer.lower_IndexStmt(cin)
    assert type(lowered) is llir.Function
    function = cast(llir.Function, lowered)
    assert [cast(llir.Var, argument).name for argument in function.args] == [
        "result_shape",
        "SparseLeft_shape",
        "SparseLeft_mode_indices",
        "SparseLeft_values",
        "SparseRight_shape",
        "SparseRight_mode_indices",
        "SparseRight_values",
    ]
    validation = [cast(llir.RawStmt, statement).code for statement in function.body[:3]]
    assert validation[0] == (
        'scorch_native::validate_jit_result_shape(result_shape, {}, 2, "evaluate")'
    )
    assert '"SparseLeft"' in validation[1]
    assert '"SparseRight"' in validation[2]
    assert all("wksp" not in statement for statement in validation)
    production_base_loads = _base_offset_loads(function)
    assert [load.var.name for load in production_base_loads] == ["_base1"]
    production_access = cast(llir.ArrayAccess, production_base_loads[0].value)
    assert cast(llir.Var, production_access.array).type is (
        llir.DataType.STD_VECTOR_INT
    )
    assert cast(llir.Var, production_access.index).name == "r"
    assert cast(llir.Var, production_access.index).type is llir.DataType.INT64
    cpp = LLIRLowerer().lower_llir(function)

    assert len(cpp) == 7117
    assert hashlib.sha256(cpp.encode()).hexdigest() == (
        "d4443cacbdb721dc88803da9cc21fa9018eb005f49d0f550e5fac3630d2ccd1f"
    )
    assert cpp.count("wksp.insert_unchecked(") == 2
    assert "wksp.insert(" not in cpp
    assert cpp.count("int pSparseLeft1_end = SparseLeft1_pos[pSparseLeft0 + 1];") == 2
    assert cpp.count("pSparseLeft1 < pSparseLeft1_end; pSparseLeft1++") == 2
    assert (
        cpp.count("int pSparseRight1_end = SparseRight1_pos[pSparseRight0 + 1];") == 2
    )
    assert cpp.count("pSparseRight1 < pSparseRight1_end; pSparseRight1++") == 2
    assert [record.pass_name for record in lowerer.llir_pass_run_records] == [
        "transform_compressed_where_for_openmp",
        "rewrite_result_writes",
        "rewrite_result_writes",
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.configuration_name for record in lowerer.llir_pass_run_records] == [
        "compressed_where_openmp",
        "count",
        "fill",
        "sparse_prefetch",
        "dense_pointer_hoist",
        "single_iteration_loop_elimination",
        "loop_invariant_factor_hoist",
        "dynamic_vector_access",
    ]
    assert [record.sequence_index for record in lowerer.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    assert all(
        not record.verified_before and not record.verified_after
        for record in lowerer.llir_pass_run_records
    )
    debug_lowerer = CINLowerer(llir_pass_options=DEBUG_LLIR_PASS_OPTIONS)
    debug_cpp = LLIRLowerer().lower_llir(debug_lowerer.lower_IndexStmt(cin))
    assert debug_cpp == cpp
    assert all(
        record.verified_before and record.verified_after
        for record in debug_lowerer.llir_pass_run_records
    )
    assert not hasattr(lowerer, "_compressed_output_parallel")
    assert [tensor.get_name() for tensor in lowerer.need_compute] == [
        "SparseProduct",
        "wksp",
        "SparseProduct",
        "wksp",
        "SparseProduct",
        "wksp",
        "wksp",
        "wksp",
    ]


def test_production_dss_generated_cpp_matches_pre_extraction_bytes() -> None:
    batch, row, reduction, column = (
        IndexVar("batch"),
        IndexVar("row"),
        IndexVar("reduction"),
        IndexVar("column"),
    )
    result = TensorVar("Result", fmt="dss")
    left = TensorVar("Left", fmt="dss")
    right = TensorVar("Right", fmt="dss")
    result[batch, row, column] = (
        left[batch, row, reduction] * right[batch, reduction, column]
    )
    assignment = result._assignment
    assert assignment is not None
    cin = cast(
        ForAll,
        Scheduler.auto_schedule(
            ForAll(
                batch,
                ForAll(row, ForAll(reduction, ForAll(column, assignment))),
            )
        ),
    )

    lowerer = CINLowerer()
    lowered = lowerer.lower_IndexStmt(cin)
    production_base_loads = _base_offset_loads(lowered)
    assert [load.var.name for load in production_base_loads] == ["_base1", "_base2"]
    for level, load in enumerate(production_base_loads, start=1):
        access = cast(llir.ArrayAccess, load.value)
        assert cast(llir.Var, access.array).name == f"_offset{level}"
        assert cast(llir.Var, access.array).type is llir.DataType.STD_VECTOR_INT
        assert cast(llir.Var, access.index).name == "batch"
        assert cast(llir.Var, access.index).type is llir.DataType.INT64
    cpp = LLIRLowerer().lower_llir(lowered)

    assert len(cpp) == 8660
    assert hashlib.sha256(cpp.encode()).hexdigest() == (
        "1471ec06cf2682e4d80f1b433f03e18f833b1d7d092b7f6ad6701a17caa0c83e"
    )
    assert cpp.count("wksp.insert(") == 2
    assert "wksp.insert_unchecked(" not in cpp
    assert cpp.count("int pLeft1_end = Left1_pos[pLeft0 + 1];") == 2
    assert cpp.count("pLeft1 < pLeft1_end; pLeft1++") == 2
    assert cpp.count("int pLeft2_end = Left2_pos[pLeft1 + 1];") == 2
    assert cpp.count("pLeft2 < pLeft2_end && pRight1 < pRight1_end") == 2
    assert cpp.count("int pRight1_end = Right1_pos[pRight0 + 1];") == 2
    assert cpp.count("int pRight2_end = Right2_pos[pRight1 + 1];") == 2
    assert cpp.count("pRight2 < pRight2_end; pRight2++") == 2
    assert [record.configuration_name for record in lowerer.llir_pass_run_records] == [
        "compressed_where_openmp",
        "count",
        "fill",
        "sparse_prefetch",
        "dense_pointer_hoist",
        "single_iteration_loop_elimination",
        "loop_invariant_factor_hoist",
        "dynamic_vector_access",
    ]
