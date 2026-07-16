from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from scorch.compiler import llir  # type: ignore[import-untyped]
from scorch.compiler.cin import (
    CIN,
    ForAll,
    IndexStmt,
    IndexVar,
    Operation,
    PostOp,
    PostOps,
    TensorAssign,
    TensorVar,
    Where,
    Workspace,
)
from scorch.compiler.cin_lowerer import CINLowerer, ResultTensorAssembler
from scorch.compiler.codegen import LLIRLowerer  # type: ignore[import-untyped]
from scorch.compiler.diagnostics import CompilerInvariantError, UnsupportedFeature
from scorch.compiler.iterator import ModeIterator  # type: ignore[import-untyped]
from scorch.compiler.llir_traversal import LLIRTraversalContext, LLIRWalker
from scorch.compiler.scheduler import Scheduler  # type: ignore[import-untyped]
from scorch.format import LevelType  # type: ignore[import-untyped]


class UnknownCIN(CIN):
    pass


class UnknownIndexStmt(IndexStmt):
    pass


def _build_all_coo_transform_loop() -> llir.ForLoop:
    return llir.ForLoop(
        init=llir.VarInit(
            llir.Var("pMask0", llir.DataType.INT),
            llir.Literal(0),
        ),
        cond=llir.BinOp(
            "<",
            llir.Var("pMask0", llir.DataType.INT),
            llir.Var("pMask0_end", llir.DataType.INT),
        ),
        update=llir.Assign(
            llir.Var("pMask0", llir.DataType.INT),
            llir.Var("pMask1_end", llir.DataType.INT),
        ),
        body=[
            llir.VarInit(
                llir.Var("r", llir.DataType.INT),
                llir.ArrayAccess(
                    array=llir.Var("Mask0_crd", llir.DataType.PTR_INT),
                    index=llir.Var("pMask0", llir.DataType.INT),
                ),
            )
        ],
    )


def _all_coo_transform_coordinate_read(
    *, scalar_accumulator: bool, source: llir.ForLoop | None = None
) -> tuple[llir.ForLoop, llir.VarInit, llir.ArrayAccess]:
    lowerer = CINLowerer()
    lowerer._used_scalar_accum = scalar_accumulator
    transformed = lowerer._transform_coo_loop_for_openmp(
        [source if source is not None else _build_all_coo_transform_loop()]
    )
    parallel_loop = next(
        cast(llir.ForLoop, statement)
        for statement in transformed
        if type(statement) is llir.ForLoop
        and cast(llir.ForLoop, statement).omp_parallel_for
    )
    initializer = next(
        cast(llir.VarInit, statement)
        for statement in parallel_loop.body
        if type(statement) is llir.VarInit
        and cast(llir.VarInit, statement).var.name == "r"
    )
    assert type(initializer.value) is llir.ArrayAccess
    return parallel_loop, initializer, cast(llir.ArrayAccess, initializer.value)


def _build_outer_workspace_statement(result_format: str) -> tuple[Where, TensorVar]:
    row, reduction, column = IndexVar("r"), IndexVar("q"), IndexVar("c")
    result = TensorVar("Result", fmt=result_format)
    left = TensorVar("Left", fmt="ds", mode_order=[1, 0])
    right = TensorVar("Right", fmt="ds")
    workspace = Workspace("wksp", dim=2)
    return (
        Where(
            producer=ForAll(
                reduction,
                ForAll(
                    row,
                    ForAll(
                        column,
                        TensorAssign(
                            workspace[row, column],
                            left[row, reduction] * right[reduction, column],
                            op=Operation.ADD,
                        ),
                    ),
                ),
            ),
            consumer=ForAll(
                row,
                ForAll(
                    column,
                    TensorAssign(result[row, column], workspace[row, column]),
                ),
            ),
        ),
        result,
    )


def _assert_workspace_pair_read(
    value: llir.Expr,
    expected_member: str,
    expected_index: int | None,
) -> llir.Var:
    if expected_index is None:
        assert type(value) is llir.MemberAccess
        member_access = cast(llir.MemberAccess, value)
    else:
        assert type(value) is llir.ArrayAccess
        array_access = cast(llir.ArrayAccess, value)
        assert array_access.tensor_access is None
        assert type(array_access.index) is llir.Literal
        literal = cast(llir.Literal, array_access.index)
        assert literal.value == expected_index
        assert literal.data_type is llir.DataType.INT64
        assert type(array_access.array) is llir.MemberAccess
        member_access = cast(llir.MemberAccess, array_access.array)

    assert member_access.member == expected_member
    assert type(member_access.base) is llir.Var
    base = cast(llir.Var, member_access.base)
    assert base.name == "it"
    assert base.type is llir.DataType.CONST_AUTO_REF
    assert base.tensor_access is None
    return base


def _assert_vector_move_initialization(
    statement: llir.Stmt,
    *,
    target_name: str,
    vector_name: str,
    vector_type: llir.DataType,
    dtype_name: str,
) -> llir.FunctionCall:
    assert type(statement) is llir.VarInit
    initializer = cast(llir.VarInit, statement)
    assert initializer.var.name == target_name
    assert initializer.var.type is llir.DataType.TORCH_TENSOR
    assert type(initializer.value) is llir.FunctionCall
    conversion = cast(llir.FunctionCall, initializer.value)
    assert conversion.name == "scorch_tensor_from_vector"
    assert type(conversion.args) is tuple
    assert len(conversion.args) == 2
    assert type(conversion.args[0]) is llir.FunctionCall
    move = cast(llir.FunctionCall, conversion.args[0])
    assert move.name == "std::move"
    assert type(move.args) is tuple
    assert len(move.args) == 1
    assert type(move.args[0]) is llir.Var
    vector = cast(llir.Var, move.args[0])
    assert vector.name == vector_name
    assert vector.type is vector_type
    assert vector.tensor_access is None
    assert type(conversion.args[1]) is llir.Var
    assert cast(llir.Var, conversion.args[1]).name == dtype_name
    return move


def _collect_move_calls(value: llir.Node) -> list[llir.FunctionCall]:
    calls: list[llir.FunctionCall] = []

    class MoveCollector(LLIRWalker):
        def visit_function_call(
            self,
            node: llir.FunctionCall,
            path: tuple[str, ...],
        ) -> None:
            if node.name == "std::move":
                calls.append(node)
            super().visit_function_call(node, path)

    MoveCollector(
        LLIRTraversalContext(stage="test", pass_name="collect_move_calls")
    ).walk(value)
    return calls


def test_unknown_cin_node_fails_at_cin_lowering():
    with pytest.raises(
        CompilerInvariantError,
        match=r"stage=CIN lowering: unknown CIN node type 'UnknownCIN'",
    ):
        CINLowerer().lower_CIN(UnknownCIN())


def test_unknown_index_statement_fails_before_lowering_an_empty_kernel():
    with pytest.raises(
        CompilerInvariantError,
        match=r"stage=CIN lowering: unknown IndexStmt node type 'UnknownIndexStmt'",
    ):
        CINLowerer().lower_IndexStmt(UnknownIndexStmt(lhs=None, rhs=None))


def test_unknown_post_op_fails_at_cin_lowering():
    post_ops = PostOps(ops=[PostOp(kind="clip")], extra_tensors=[])

    with pytest.raises(
        UnsupportedFeature,
        match=r"stage=CIN lowering: unsupported post-op kind 'clip'",
    ):
        CINLowerer(post_ops=post_ops)


@pytest.mark.parametrize(
    ("kind", "tensor_name"),
    [
        ("add", "bias"),
        ("mul", "scale"),
        ("relu", None),
        ("gelu", None),
        ("tanh", None),
        ("sigmoid", None),
    ],
)
def test_supported_post_ops_still_lower(kind, tensor_name):
    extra_tensors = [tensor_name] if tensor_name else []
    post_ops = PostOps(
        ops=[PostOp(kind=kind, tensor_name=tensor_name)],
        extra_tensors=extra_tensors,
    )

    statements = CINLowerer(post_ops=post_ops)._emit_post_ops("output", "i")

    assert len(statements) == 1
    assignment = cast(llir.Assign, statements[0])
    assert type(assignment.var) is llir.ArrayAccess
    target = cast(llir.ArrayAccess, assignment.var)
    assert cast(llir.Var, target.array).name == "output"
    assert cast(llir.Var, target.index).name == "i"
    assert LLIRLowerer().lower_llir(assignment).startswith("output[i] ")


def test_result_position_initialization_uses_a_frozen_structured_target() -> None:
    result = TensorVar("Result", fmt="ds")

    statements = ResultTensorAssembler(result).emit_level_indices_init()
    assignment = next(
        cast(llir.Assign, statement)
        for statement in statements
        if type(statement) is llir.Assign
    )

    assert type(assignment.var) is llir.ArrayAccess
    target = cast(llir.ArrayAccess, assignment.var)
    assert cast(llir.Var, target.array).name == "Result1_pos"
    assert cast(llir.Literal, target.index).value == 0
    assert LLIRLowerer().lower_llir(assignment) == "Result1_pos[0] = 0;"


def test_final_result_assembly_uses_structured_typed_move_calls() -> None:
    statements = ResultTensorAssembler(
        TensorVar("Result", fmt="ds")
    ).emit_final_assembly()
    initializers = {
        statement.var.name: statement
        for statement in statements
        if type(statement) is llir.VarInit
    }

    expected = (
        (
            "Result1_pos_torch",
            "Result1_pos",
            llir.DataType.STD_VECTOR_C_INT,
            "torch::kInt",
        ),
        (
            "Result1_crd_torch",
            "Result1_crd",
            llir.DataType.STD_VECTOR_C_INT,
            "torch::kInt",
        ),
        (
            "Result_values_torch",
            "Result_values",
            llir.DataType.STD_VECTOR_FLOAT32,
            "torch::kFloat32",
        ),
    )
    moves = [
        _assert_vector_move_initialization(
            initializers[target_name],
            target_name=target_name,
            vector_name=vector_name,
            vector_type=vector_type,
            dtype_name=dtype_name,
        )
        for target_name, vector_name, vector_type, dtype_name in expected
    ]

    assert [LLIRLowerer().lower_llir(initializers[name]) for name, *_ in expected] == [
        "torch::Tensor Result1_pos_torch = "
        "scorch_tensor_from_vector(std::move(Result1_pos), torch::kInt);",
        "torch::Tensor Result1_crd_torch = "
        "scorch_tensor_from_vector(std::move(Result1_crd), torch::kInt);",
        "torch::Tensor Result_values_torch = "
        "scorch_tensor_from_vector(std::move(Result_values), torch::kFloat32);",
    ]
    assert len({id(cast(llir.Var, move.args[0])) for move in moves}) == len(moves)


def test_all_coo_workspace_assembly_targets_use_storage_types() -> None:
    statement, result = _build_outer_workspace_statement("oo")
    lowerer = CINLowerer()
    lowerer.outermost_stmt = statement
    lowerer.result_tensor_var = result
    lowerer.result_tensor_access = statement.consumer.get_result_tensor_accesses()[0]
    lowered = lowerer.lower_outer_ConsumerIndexStmt(statement.consumer)
    workspace_loop = next(
        cast(llir.ForLoopAuto, node)
        for node in lowered
        if type(node) is llir.ForLoopAuto
    )

    workspace_assignments = [
        cast(llir.Assign, node)
        for node in workspace_loop.body
        if type(node) is llir.Assign
    ]
    assert [LLIRLowerer().lower_llir(node) for node in workspace_assignments] == [
        "Result0_crd[pResult0] = it.first[0];",
        "Result1_crd[pResult1] = it.first[1];",
        "Result_values[pResult0] = it.second;",
    ]
    pair_bases = [
        _assert_workspace_pair_read(node.value, member, index)
        for node, member, index in zip(
            workspace_assignments,
            ("first", "first", "second"),
            (0, 1, None),
        )
    ]
    assert len({id(base) for base in pair_bases}) == len(pair_bases)
    assert all(base is not workspace_loop.var for base in pair_bases)

    storage_types = {
        cast(llir.Var, cast(llir.ArrayAccess, node.var).array)
        .name: cast(llir.Var, cast(llir.ArrayAccess, node.var).array)
        .type
        for node in workspace_assignments
    }
    assert storage_types == {
        "Result0_crd": llir.DataType.STD_VECTOR_C_INT,
        "Result1_crd": llir.DataType.STD_VECTOR_C_INT,
        "Result_values": llir.DataType.STD_VECTOR_FLOAT32,
    }

    lowered_function = CINLowerer().lower_IndexStmt(statement)
    dynamic_vector_assignments: list[llir.Assign] = []

    class DynamicVectorAssignmentCollector(LLIRWalker):
        def visit_assign(self, node: llir.Assign, path: tuple[str, ...]) -> None:
            dynamic_vector_assignments.append(node)
            super().visit_assign(node, path)

    DynamicVectorAssignmentCollector(
        LLIRTraversalContext(
            stage="test",
            pass_name="collect_dynamic_vector_assignments",
        )
    ).walk(lowered_function)
    assert not any(
        type(node.var) is llir.ArrayAccess
        and cast(llir.Var, cast(llir.ArrayAccess, node.var).array).name
        in {"Result0_crd", "Result1_crd", "Result_values"}
        for node in dynamic_vector_assignments
    )


def test_outer_workspace_intermediate_reads_use_structured_pair_members() -> None:
    statement, result = _build_outer_workspace_statement("ds")
    original = str(statement)
    lowerer = CINLowerer()
    lowerer.outermost_stmt = statement
    lowerer.result_tensor_var = result
    lowerer.result_tensor_access = statement.consumer.get_result_tensor_accesses()[0]

    lowered = lowerer.lower_outer_ConsumerIndexStmt(statement.consumer)
    workspace_loop = next(
        cast(llir.ForLoopAuto, node)
        for node in lowered
        if type(node) is llir.ForLoopAuto
    )
    workspace_assignments = [
        cast(llir.Assign, node)
        for node in workspace_loop.body
        if type(node) is llir.Assign
    ]

    assert [LLIRLowerer().lower_llir(node) for node in workspace_assignments] == [
        "T0_crd_vec[pT] = it.first[0];",
        "T1_crd_vec[pT] = it.first[1];",
        "T_val_vec[pT] = it.second;",
    ]
    pair_bases = [
        _assert_workspace_pair_read(node.value, member, index)
        for node, member, index in zip(
            workspace_assignments,
            ("first", "first", "second"),
            (0, 1, None),
        )
    ]
    assert len({id(base) for base in pair_bases}) == len(pair_bases)
    assert all(base is not workspace_loop.var for base in pair_bases)

    assembly_initializers = {
        node.var.name: node
        for node in lowered
        if type(node) is llir.VarInit
        and type(node.value) is llir.FunctionCall
        and cast(llir.FunctionCall, node.value).name == "scorch_tensor_from_vector"
    }
    expected_moves = (
        (
            "T0_crd_tensor",
            "T0_crd_vec",
            llir.DataType.STD_VECTOR_C_INT,
            "torch::kInt",
        ),
        (
            "T1_crd_tensor",
            "T1_crd_vec",
            llir.DataType.STD_VECTOR_C_INT,
            "torch::kInt",
        ),
        (
            "T_val_tensor",
            "T_val_vec",
            llir.DataType.STD_VECTOR_FLOAT32,
            "torch::kFloat32",
        ),
    )
    moves = [
        _assert_vector_move_initialization(
            assembly_initializers[target_name],
            target_name=target_name,
            vector_name=vector_name,
            vector_type=vector_type,
            dtype_name=dtype_name,
        )
        for target_name, vector_name, vector_type, dtype_name in expected_moves
    ]
    assert [
        LLIRLowerer().lower_llir(assembly_initializers[name])
        for name, *_ in expected_moves
    ] == [
        "torch::Tensor T0_crd_tensor = "
        "scorch_tensor_from_vector(std::move(T0_crd_vec), torch::kInt);",
        "torch::Tensor T1_crd_tensor = "
        "scorch_tensor_from_vector(std::move(T1_crd_vec), torch::kInt);",
        "torch::Tensor T_val_tensor = "
        "scorch_tensor_from_vector(std::move(T_val_vec), torch::kFloat32);",
    ]
    assert len({id(cast(llir.Var, move.args[0])) for move in moves}) == len(moves)
    assert str(statement) == original


def test_move_calls_survive_both_production_paths_with_independent_ownership() -> None:
    statement, _ = _build_outer_workspace_statement("ds")
    original = str(statement)

    first = CINLowerer().lower_IndexStmt(statement)
    second = CINLowerer().lower_IndexStmt(statement)

    assert type(first) is llir.Function
    assert type(second) is llir.Function
    first_calls = _collect_move_calls(cast(llir.Function, first))
    second_calls = _collect_move_calls(cast(llir.Function, second))
    expected = [
        ("T0_crd_vec", llir.DataType.STD_VECTOR_C_INT),
        ("T1_crd_vec", llir.DataType.STD_VECTOR_C_INT),
        ("T_val_vec", llir.DataType.STD_VECTOR_FLOAT32),
        ("Result1_pos", llir.DataType.STD_VECTOR_C_INT),
        ("Result1_crd", llir.DataType.STD_VECTOR_C_INT),
        ("Result_values", llir.DataType.STD_VECTOR_FLOAT32),
    ]

    def call_values(
        calls: list[llir.FunctionCall],
    ) -> list[tuple[str, llir.DataType]]:
        return [
            (cast(llir.Var, call.args[0]).name, cast(llir.Var, call.args[0]).type)
            for call in calls
        ]

    assert call_values(first_calls) == call_values(second_calls) == expected
    assert LLIRLowerer().lower_llir(first) == LLIRLowerer().lower_llir(second)
    assert str(statement) == original
    assert {id(cast(llir.Var, call.args[0])) for call in first_calls}.isdisjoint(
        {id(cast(llir.Var, call.args[0])) for call in second_calls}
    )

    cast(llir.Var, first_calls[0].args[0]).name = "owned_by_first"
    assert call_values(second_calls) == expected
    assert str(statement) == original


@pytest.mark.parametrize(
    ("result_format", "index_names", "expected_initializers"),
    [
        pytest.param(
            "s",
            ("i",),
            ("int64_t i = it.first;", "float wksp_value = it.second;"),
            id="rank-one",
        ),
        pytest.param(
            "oo",
            ("i", "j"),
            (
                "int64_t i = it.first[0];",
                "int64_t j = it.first[1];",
                "float wksp_value = it.second;",
            ),
            id="rank-two",
        ),
    ],
)
def test_nested_workspace_reads_use_structured_pair_members(
    result_format: str,
    index_names: tuple[str, ...],
    expected_initializers: tuple[str, ...],
) -> None:
    index_vars = tuple(IndexVar(name) for name in index_names)
    result = TensorVar("Result", fmt=result_format)
    workspace = Workspace("wksp", dim=len(index_vars))
    access_key = index_vars[0] if len(index_vars) == 1 else index_vars
    consumer = TensorAssign(result[access_key], workspace[access_key])
    original = str(consumer)
    lowerer = CINLowerer()
    lowerer.outermost_stmt = ForAll(IndexVar("outer"), consumer)

    lowered = lowerer.lower_ConsumerIndexStmt(consumer)
    workspace_loop = next(
        cast(llir.ForLoopAuto, node)
        for node in lowered
        if type(node) is llir.ForLoopAuto
    )
    initializers = [
        cast(llir.VarInit, node)
        for node in workspace_loop.body
        if type(node) is llir.VarInit
    ][: len(index_vars) + 1]

    assert [LLIRLowerer().lower_llir(node) for node in initializers] == list(
        expected_initializers
    )
    expected_reads: list[tuple[str, int | None]] = (
        [("first", None)]
        if len(index_vars) == 1
        else [("first", index) for index in range(len(index_vars))]
    )
    expected_reads.append(("second", None))
    pair_bases = [
        _assert_workspace_pair_read(node.value, member, index)
        for node, (member, index) in zip(initializers, expected_reads)
    ]
    assert len({id(base) for base in pair_bases}) == len(pair_bases)
    assert all(base is not workspace_loop.var for base in pair_bases)
    assert str(consumer) == original


@pytest.mark.parametrize(
    ("level_type", "fmt"),
    (
        pytest.param(LevelType.COORDINATE, "oo", id="coordinate"),
        pytest.param(LevelType.COMPRESSED, "ds", id="compressed"),
    ),
)
def test_sparse_mode_iterator_coordinate_reads_are_structured_typed_and_owned(
    level_type: LevelType,
    fmt: str,
) -> None:
    tensor = TensorVar("Input", fmt=fmt)
    index = IndexVar("column")
    parent = IndexVar("row")
    tensor_state = (
        tensor.name,
        tensor.symbol_id,
        tensor.format,
        tensor.shape,
        tensor.dtype,
        tuple(tensor.mode_order or ()),
        tensor._assignment,
    )

    def index_state(value: IndexVar) -> tuple[object, ...]:
        return (
            value.name,
            value.index_id,
            value._expr,
            value._parent,
            value.is_tiled,
            value.is_outer,
            value.is_inner,
            value.tile_size_var,
            tuple(value.tensor_accesses),
        )

    index_before = index_state(index)
    parent_before = index_state(parent)

    first_iterator = ModeIterator(
        _tensor_var=tensor,
        index_var=index,
        parent_index_var=parent,
        _level=1,
        level_type=level_type,
    )
    second_iterator = ModeIterator(
        _tensor_var=tensor,
        index_var=index,
        parent_index_var=parent,
        _level=1,
        level_type=level_type,
    )
    first = first_iterator.get_coord_var_value_llir()
    second = second_iterator.get_coord_var_value_llir()

    expected = llir.ArrayAccess(
        array=llir.Var("Input1_crd", llir.DataType.PTR_INT),
        index=llir.Var("pInput1", llir.DataType.INT),
    )
    assert type(first) is llir.ArrayAccess
    assert type(second) is llir.ArrayAccess
    first_access = cast(llir.ArrayAccess, first)
    second_access = cast(llir.ArrayAccess, second)
    assert first_access == expected == second_access
    assert hash(first_access) == hash(expected) == hash(second_access)
    assert first_access.tensor_access is None
    assert second_access.tensor_access is None
    assert type(first_access.array) is llir.Var
    assert type(first_access.index) is llir.Var
    first_array = cast(llir.Var, first_access.array)
    first_index = cast(llir.Var, first_access.index)
    assert first_array.name == "Input1_crd"
    assert first_array.type is llir.DataType.PTR_INT
    assert first_array.tensor_access is None
    assert first_index.name == "pInput1"
    assert first_index.type is llir.DataType.INT
    assert first_index.tensor_access is None

    assert first_access is not second_access
    assert first_access.array is not second_access.array
    assert first_access.index is not second_access.index
    assert first_iterator.iterator_var_llir is not second_iterator.iterator_var_llir
    assert first_access.index is not first_iterator.iterator_var_llir
    assert second_access.index is not second_iterator.iterator_var_llir
    assert LLIRLowerer().lower_llir(first_access) == "Input1_crd[pInput1]"
    with pytest.raises(FrozenInstanceError):
        first_access.index = llir.Var("other", llir.DataType.INT)

    assert first_iterator.tensor_var is tensor
    assert first_iterator.index_var is index
    assert first_iterator.parent_index_var is parent
    assert tensor_state == (
        tensor.name,
        tensor.symbol_id,
        tensor.format,
        tensor.shape,
        tensor.dtype,
        tuple(tensor.mode_order or ()),
        tensor._assignment,
    )
    assert index_state(index) == index_before
    assert index_state(parent) == parent_before


@pytest.mark.parametrize("scalar_accumulator", (False, True))
def test_all_coo_transform_coordinate_initializers_are_structured_typed_and_owned(
    scalar_accumulator: bool,
) -> None:
    source = _build_all_coo_transform_loop()
    source_initializer = cast(llir.VarInit, source.body[0])
    source_access = cast(llir.ArrayAccess, source_initializer.value)
    source_cpp = LLIRLowerer().lower_llir(source)
    assert source_cpp == (
        "for (int pMask0 = 0; pMask0 < pMask0_end; pMask0 = pMask1_end) {\n"
        "  int r = Mask0_crd[pMask0];\n"
        "}"
    )

    first_loop, first_initializer, first = _all_coo_transform_coordinate_read(
        scalar_accumulator=scalar_accumulator,
        source=source,
    )
    second_loop, _, second = _all_coo_transform_coordinate_read(
        scalar_accumulator=scalar_accumulator
    )

    expected = llir.ArrayAccess(
        array=llir.Var("Mask0_crd", llir.DataType.PTR_INT),
        index=llir.Var("pMask0", llir.DataType.INT64),
    )
    assert first == expected == second
    assert hash(first) == hash(expected) == hash(second)
    assert first.tensor_access is None
    assert type(first.array) is llir.Var
    assert cast(llir.Var, first.array).type is llir.DataType.PTR_INT
    assert type(first.index) is llir.Var
    assert cast(llir.Var, first.index).type is llir.DataType.INT64
    assert first is not second
    assert first.array is not second.array
    assert first.index is not second.index
    assert first is not source_access
    assert first != source_access
    assert not any(statement is source_initializer for statement in first_loop.body)
    assert not any(statement == source_initializer for statement in first_loop.body)

    first_candidates = [cast(llir.VarInit, first_loop.init)] + [
        cast(llir.VarInit, statement)
        for statement in first_loop.body
        if type(statement) is llir.VarInit
    ]
    second_candidates = [cast(llir.VarInit, second_loop.init)] + [
        cast(llir.VarInit, statement)
        for statement in second_loop.body
        if type(statement) is llir.VarInit
    ]
    first_iterator = next(
        initializer.var
        for initializer in first_candidates
        if initializer.var.name == "pMask0"
    )
    second_iterator = next(
        initializer.var
        for initializer in second_candidates
        if initializer.var.name == "pMask0"
    )
    assert first.index is not first_iterator
    assert second.index is not second_iterator
    assert LLIRLowerer().lower_llir(first_initializer) == (
        "int64_t r = Mask0_crd[pMask0];"
    )
    first_loop_cpp = LLIRLowerer().lower_llir(first_loop)
    assert first_loop_cpp.count("  int64_t r = Mask0_crd[pMask0];") == 1
    assert "  int r = Mask0_crd[pMask0];" not in first_loop_cpp
    assert LLIRLowerer().lower_llir(source) == source_cpp
    with pytest.raises(FrozenInstanceError):
        first.index = llir.Var("other", llir.DataType.INT64)


def test_mutated_unknown_post_op_cannot_be_silently_skipped():
    post_ops = PostOps(ops=[], extra_tensors=[])
    lowerer = CINLowerer(post_ops=post_ops)
    post_ops.ops.append(PostOp(kind="clip"))

    with pytest.raises(
        UnsupportedFeature,
        match=r"stage=CIN lowering: unsupported post-op kind 'clip'",
    ):
        lowerer._emit_post_ops("output", "i")


@pytest.mark.parametrize(
    ("fmt", "indices", "expected_index"),
    (
        ("d", ("i",), "i"),
        ("ds", ("i", "j"), "pInput1"),
    ),
)
def test_nonworkspace_tensor_reads_lower_to_frozen_structured_accesses(
    fmt: str,
    indices: tuple[str, ...],
    expected_index: str,
) -> None:
    index_vars = tuple(IndexVar(name) for name in indices)
    tensor = TensorVar("Input", fmt=fmt)
    access = tensor[index_vars[0] if len(index_vars) == 1 else index_vars]
    original_indices = tuple(access.indices)

    lowered = CINLowerer().lower_TensorAccess(access)

    assert type(lowered) is llir.ArrayAccess
    structured = cast(llir.ArrayAccess, lowered)
    assert cast(llir.Var, structured.array).name == "Input_val"
    assert cast(llir.Var, structured.array).type is llir.DataType.PTR_FLOAT32
    assert cast(llir.Var, structured.index).name == expected_index
    assert cast(llir.Var, structured.index).type is llir.DataType.INT
    assert structured.tensor_access == llir.TensorAccessMetadata(
        access_id=access.access_id,
        tensor_id=access.tensor_id,
        index_ids=access.index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    assert LLIRLowerer().lower_llir(structured) == f"Input_val[{expected_index}]"
    assert tuple(access.indices) == original_indices
    assert access.tensor is tensor


def test_dense_level_shape_reads_use_frozen_structured_array_accesses() -> None:
    row, column = IndexVar("row"), IndexVar("column")
    result = TensorVar("Result", shape=(3, 5), fmt="dd")
    operand = TensorVar("Input", shape=(3, 5), fmt="dd")
    statement = ForAll(
        row,
        ForAll(
            column,
            TensorAssign(result[row, column], operand[row, column]),
        ),
    )

    lowered = CINLowerer().lower_IndexStmt(statement)

    assert type(lowered) is llir.Function
    function = cast(llir.Function, lowered)
    initializers = {
        node.var.name: cast(llir.VarInit, node)
        for node in function.body
        if type(node) is llir.VarInit
    }
    expected = {
        "Result0_size": ("result_shape", 0),
        "Result1_size": ("result_shape", 1),
        "Input0_size": ("Input_shape", 0),
        "Input1_size": ("Input_shape", 1),
    }
    for initializer_name, (shape_name, level) in expected.items():
        initializer = initializers[initializer_name]
        assert type(initializer.value) is llir.ArrayAccess
        access = cast(llir.ArrayAccess, initializer.value)
        assert access == llir.ArrayAccess(
            array=llir.Var(shape_name, llir.DataType.STD_VECTOR_INT),
            index=llir.Literal(level, data_type=llir.DataType.INT64),
        )
        assert type(access.array) is llir.Var
        assert cast(llir.Var, access.array).type is llir.DataType.STD_VECTOR_INT
        assert type(access.index) is llir.Literal
        assert cast(llir.Literal, access.index).data_type is llir.DataType.INT64
        assert access.tensor_access is None
        assert LLIRLowerer().lower_llir(initializer) == (
            f"int64_t {initializer_name} = {shape_name}[{level}];"
        )

    result_access = cast(llir.ArrayAccess, initializers["Result0_size"].value)
    with pytest.raises(FrozenInstanceError):
        result_access.index = llir.Literal(1)


def test_cin_reference_rewrite_rebuilds_frozen_access_and_preserves_metadata() -> None:
    index = IndexVar("ix")
    tensor = TensorVar("Input", fmt="d")
    logical_access = tensor[index]
    metadata = llir.TensorAccessMetadata(
        access_id=logical_access.access_id,
        tensor_id=logical_access.tensor_id,
        index_ids=logical_access.index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    source = llir.ArrayAccess(
        array=llir.Var("Array[ix]", llir.DataType.PTR_FLOAT32),
        index=llir.Var("ix offset", llir.DataType.INT),
        tensor_access=metadata,
    )

    rewritten = CINLowerer._rewrite_expr_refs(source, {"ix": "root"})
    repeated = CINLowerer._rewrite_expr_refs(rewritten, {"ix": "root"})

    assert type(rewritten) is llir.ArrayAccess
    assert cast(llir.Var, rewritten.array).name == "Array[root]"
    assert cast(llir.Var, rewritten.index).name == "root offset"
    assert rewritten.tensor_access is metadata
    assert repeated == rewritten
    assert repeated is not rewritten
    assert cast(llir.Var, source.array).name == "Array[ix]"
    assert cast(llir.Var, source.index).name == "ix offset"


def test_nondefault_coo_intersection_keeps_live_coordinate_end_bounds():
    row, column = IndexVar("r"), IndexVar("c")
    result = TensorVar("Intersect", fmt="oo", mode_order=[1, 0])
    left = TensorVar("Left", fmt="oo", mode_order=[1, 0])
    right = TensorVar("Right", fmt="oo", mode_order=[1, 0])
    assignment = TensorAssign(
        result[row, column],
        left[row, column] * right[row, column],
    )
    scheduled = cast(
        ForAll,
        Scheduler.auto_schedule(ForAll(row, ForAll(column, assignment))),
    )

    cpp = LLIRLowerer().lower_llir(CINLowerer().lower_IndexStmt(scheduled))

    assert "int pLeft1_end = 0;" in cpp
    assert "pLeft1_end = pLeft0 + 1;" in cpp
    assert "pLeft1_end < pLeft0_end" in cpp
    assert "pLeft1 < pLeft1_end && pRight1 < pRight1_end" in cpp
    assert "pLeft0 = pLeft1_end;" in cpp
    assert "int pRight1_end = 0;" in cpp
    assert "pRight1_end = pRight0 + 1;" in cpp
    assert "pRight1_end < pRight0_end" in cpp
    assert "pRight0 = pRight1_end;" in cpp
