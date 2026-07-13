import hashlib
from dataclasses import FrozenInstanceError
from typing import List, Set, Tuple, cast

import pytest

from scorch.compiler import llir
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.compressed_where_openmp_pass import (
    CompressedWhereOpenMPContext,
    CompressedWhereOpenMPPolicy,
    CompressedWhereOpenMPResult,
    transform_compressed_where_for_openmp,
)
from scorch.compiler.llir_traversal import (
    LLIRStatementValue,
    LLIRTraversalContext,
    LLIRTraversalError,
)
from scorch.compiler.llir_pass_manager import DEBUG_LLIR_PASS_OPTIONS
from scorch.compiler.scheduler import Scheduler


def _var(name: str, data_type: llir.DataType = llir.DataType.NO_TYPE) -> llir.Var:
    return llir.Var(name=name, type=data_type)


def _context(
    compressed_levels: Tuple[int, ...] = (1,),
    *,
    policy: CompressedWhereOpenMPPolicy = CompressedWhereOpenMPPolicy(),
) -> CompressedWhereOpenMPContext:
    return CompressedWhereOpenMPContext(
        result_name="Result",
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
    top_level_codes = [
        statement.code
        for statement in result.statements
        if type(statement) is llir.RawStmt
    ]

    assert "int _cnt1 = 0" in count_codes
    assert "_cnt1++" in count_codes
    assert "_count1[row] = _cnt1" in count_codes
    assert "wksp.clear()" in count_codes
    assert "int64_t _base1 = _offset1[row]" in fill_codes
    assert "int _pos1 = 0" in fill_codes
    assert "Result1_crd_data[_base1 + _pos1] = column" in fill_codes
    assert "_pos1++" in fill_codes
    assert "Result_values_data[_base1 + _pos1] = value" in fill_codes
    assert "wksp.clear()" in fill_codes
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
    count_codes = _raw_codes(count_loop.body)
    fill_codes = _raw_codes(fill_loop.body)
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

    assert "int _cnt1 = 0" in count_codes
    assert "int _cnt2 = 0" in count_codes
    assert "int _prev2 = 0" in count_codes
    assert cast(llir.BinOp, count_boundary.cond).op == ">"
    assert _raw_codes(count_boundary.then_body) == ["_cnt1++", "_prev2 = _cnt2"]
    assert cast(llir.BinOp, fill_boundary.cond).op == ">"
    assert _raw_codes(fill_boundary.then_body) == [
        "Result1_crd_data[_base1 + _pos1] = parent_coordinate",
        "_pos1++",
        "Result2_pos_data[_base1 + _pos1] = _base2 + _pos2",
        "_prev2 = _pos2",
    ]
    assert "Result2_crd_data[_base2 + _pos2] = leaf_coordinate" in fill_codes
    assert "Result2_pos_data[0] = 0" in all_codes
    assert sum(code == "Result2_pos_data[0] = 0" for code in all_codes) == 1
    assert any(
        "{{}, {Result1_pos_torch, Result1_crd_torch}, "
        "{Result2_pos_torch, Result2_crd_torch}}" in code
        for code in all_codes
    )


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
            llir.Assign(_var("Result1_pos[row]"), _var("pResult1")),
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
        llir.Assign(_var("Result1_pos[0]"), llir.Literal(0)),
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
    count_codes = _raw_codes(count_loop.body)
    calls = _call_names(count_loop.body)
    assert count_codes.count("_cnt1++") == 8
    # The legacy name rewrite descends into ForLoop bodies and If branches, but
    # not ForLoopAuto, WhileLoop, or bare nested statement containers.
    assert calls.count("wksp.insert_unchecked") == 4
    assert calls.count("wksp.insert") == 4


def test_workspace_reference_rewrite_covers_each_legacy_field_form() -> None:
    assignment = llir.Assign(
        llir.ArrayAccess(_var("wksp.insert_targets"), _var("wksp.insert_target_index")),
        llir.BinOp(
            "+",
            _var("wksp.insert_value"),
            llir.ArrayAccess(
                _var("wksp.insert_values"), _var("wksp.insert_value_index")
            ),
        ),
    )
    initialization = llir.VarInit(
        _var("initialized"), _var("wksp.insert_initial_value")
    )
    raw = llir.RawStmt("wksp.insert(raw_value)")
    source: List[llir.Stmt] = [
        _compatible_loop([_workspace_init(), assignment, initialization, raw])
    ]
    snapshot = _structural_snapshot(source)

    result = transform_compressed_where_for_openmp(source, _context())

    count_loop, _ = _phase_loops(result)
    rewritten_assignment = cast(
        llir.Assign,
        next(
            statement for statement in count_loop.body if type(statement) is llir.Assign
        ),
    )
    target = cast(llir.ArrayAccess, rewritten_assignment.var)
    value = cast(llir.BinOp, rewritten_assignment.value)
    value_access = cast(llir.ArrayAccess, value.right)
    rewritten_initialization = cast(
        llir.VarInit,
        next(
            statement
            for statement in count_loop.body
            if type(statement) is llir.VarInit
        ),
    )

    assert cast(llir.Var, target.array).name == "wksp.insert_unchecked_targets"
    assert cast(llir.Var, target.index).name == "wksp.insert_unchecked_target_index"
    assert cast(llir.Var, value.left).name == "wksp.insert_unchecked_value"
    assert cast(llir.Var, value_access.array).name == "wksp.insert_unchecked_values"
    assert cast(llir.Var, value_access.index).name == (
        "wksp.insert_unchecked_value_index"
    )
    assert cast(llir.Var, rewritten_initialization.value).name == (
        "wksp.insert_unchecked_initial_value"
    )
    assert "wksp.insert_unchecked(raw_value)" in _raw_codes(count_loop.body)
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
    ("workspace_ctype", "torch_dtype"),
    [
        ("float", "torch::kFloat32"),
        ("double", "torch::kFloat64"),
        ("int32_t", "torch::kInt32"),
        ("int64_t", "torch::kInt64"),
        ("custom_scalar", "torch::kFloat32"),
    ],
)
def test_workspace_ctype_explicitly_controls_value_allocation(
    workspace_ctype: str, torch_dtype: str
) -> None:
    context = CompressedWhereOpenMPContext(
        result_name="Result",
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
    ("context", "expected_code", "expected_path"),
    [
        (
            cast(CompressedWhereOpenMPContext, object()),
            "invalid_compressed_where_context",
            ("context",),
        ),
        (
            CompressedWhereOpenMPContext("", (1,), "wksp", "float"),
            "invalid_compressed_where_result_name",
            ("context", "result_name"),
        ),
        (
            CompressedWhereOpenMPContext("Result", (), "wksp", "float"),
            "invalid_compressed_where_levels",
            ("context", "compressed_levels"),
        ),
        (
            CompressedWhereOpenMPContext("Result", (2,), "wksp", "float"),
            "unsupported_compressed_where_layout",
            ("context", "compressed_levels"),
        ),
        (
            CompressedWhereOpenMPContext("Result", (1,), "", "float"),
            "invalid_compressed_where_workspace_name",
            ("context", "workspace_name"),
        ),
        (
            CompressedWhereOpenMPContext("Result", (1,), "wksp", ""),
            "invalid_compressed_where_workspace_ctype",
            ("context", "workspace_ctype"),
        ),
        (
            CompressedWhereOpenMPContext(
                "Result",
                (1,),
                "wksp",
                "float",
                policy=CompressedWhereOpenMPPolicy(omp_schedule="", flop_grain="grain"),
            ),
            "invalid_compressed_where_schedule",
            ("context", "policy.omp_schedule"),
        ),
        (
            CompressedWhereOpenMPContext(
                "Result",
                (1,),
                "wksp",
                "float",
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


def test_production_ds_generated_cpp_matches_pre_extraction_bytes() -> None:
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

    lowerer = CINLowerer()
    cpp = LLIRLowerer().lower_llir(lowerer.lower_IndexStmt(cin))

    assert len(cpp) == 7117
    assert hashlib.sha256(cpp.encode()).hexdigest() == (
        "d4443cacbdb721dc88803da9cc21fa9018eb005f49d0f550e5fac3630d2ccd1f"
    )
    assert [record.pass_name for record in lowerer.llir_pass_run_records] == [
        "transform_compressed_where_for_openmp",
        "rewrite_result_writes",
        "rewrite_result_writes",
        "rewrite_dynamic_vector_accesses",
    ]
    assert [record.configuration_name for record in lowerer.llir_pass_run_records] == [
        "compressed_where_openmp",
        "count",
        "fill",
        "dynamic_vector_access",
    ]
    assert [record.sequence_index for record in lowerer.llir_pass_run_records] == [
        0,
        1,
        2,
        3,
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
    cpp = LLIRLowerer().lower_llir(lowerer.lower_IndexStmt(cin))

    assert len(cpp) == 8660
    assert hashlib.sha256(cpp.encode()).hexdigest() == (
        "1471ec06cf2682e4d80f1b433f03e18f833b1d7d092b7f6ad6701a17caa0c83e"
    )
    assert [record.configuration_name for record in lowerer.llir_pass_run_records] == [
        "compressed_where_openmp",
        "count",
        "fill",
        "dynamic_vector_access",
    ]
