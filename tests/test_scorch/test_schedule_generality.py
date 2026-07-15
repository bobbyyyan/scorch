from typing import cast

import pytest
import torch

from scorch import STensor, einsum
from scorch.compiler import llir
from scorch.compiler.cin import (
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
)
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.identity import AccessId, IndexId, SymbolId  # type: ignore[import-untyped]
from scorch.compiler.legacy_cin_adapter import legacy_cin_working_copy
from scorch.compiler.llir_traversal import LLIRTraversalError  # type: ignore[import-untyped]
from scorch.compiler.schedule_lowerer import (  # type: ignore[import-untyped]
    _contains_tensor_access,
    _matches_tensor_access,
    _rewrite_expr_access,
    _rewrite_stmt_accesses,
)
from scorch.compiler.scheduler import (
    Schedule,
    Scheduler,
    TileSpec,
    regblock_force,
)


def _nest(index_vars, assignment):
    stmt = assignment
    for index_var in reversed(index_vars):
        stmt = ForAll(index_var, stmt)
    assert isinstance(stmt, ForAll)
    return stmt


def _lower_to_cpp(stmt: ForAll) -> str:
    return LLIRLowerer().lower_llir(CINLowerer().lower_IndexStmt(stmt))


def _sparse_stensor(tensor: torch.Tensor, name: str, fmt: str = "ds") -> STensor:
    return STensor.from_torch(tensor, name).to_sparse(fmt)


def _build_elementwise(fmt: str) -> ForAll:
    row, col = IndexVar("r"), IndexVar("c")
    out = TensorVar("ElemOut", fmt=fmt)
    left = TensorVar("ElemLeft", fmt=fmt)
    right = TensorVar("ElemRight", fmt=fmt)
    assignment = TensorAssign(out[row, col], left[row, col] * right[row, col])
    return _nest((row, col), assignment)


def _build_ttm() -> ForAll:
    batch, row, reduction, feature = (
        IndexVar("a"),
        IndexVar("b"),
        IndexVar("c"),
        IndexVar("d"),
    )
    out = TensorVar("Projected", fmt="ddd")
    core = TensorVar("Core", fmt="ddd")
    factor = TensorVar("Factor", fmt="dd")
    assignment = TensorAssign(
        out[batch, row, feature],
        core[batch, row, reduction] * factor[reduction, feature],
        op=Operation.ADD,
    )
    return _nest((batch, row, reduction, feature), assignment)


def _build_sddmm() -> ForAll:
    row, col, reduction = IndexVar("r"), IndexVar("c"), IndexVar("q")
    out = TensorVar("Sampled", fmt="oo")
    mask = TensorVar("Mask", fmt="ds")
    left = TensorVar("Query", fmt="dd")
    right = TensorVar("Key", fmt="dd")
    assignment = TensorAssign(
        out[row, col],
        mask[row, col] * left[row, reduction] * right[col, reduction],
        op=Operation.ADD,
    )
    return _nest((row, col, reduction), assignment)


def _build_spgemm() -> ForAll:
    row, reduction, col = IndexVar("r"), IndexVar("q"), IndexVar("c")
    out = TensorVar("SparseProduct", fmt="ds")
    left = TensorVar("SparseLeft", fmt="ds")
    right = TensorVar("SparseRight", fmt="ds")
    assignment = TensorAssign(
        out[row, col],
        left[row, reduction] * right[reduction, col],
        op=Operation.ADD,
    )
    return _nest((row, reduction, col), assignment)


def _build_spmv() -> ForAll:
    row, reduction = IndexVar("r"), IndexVar("q")
    out = TensorVar("VectorOut", fmt="d")
    matrix = TensorVar("SparseMatrix", fmt="ds")
    vector = TensorVar("DenseVector", fmt="d")
    assignment = TensorAssign(
        out[row],
        matrix[row, reduction] * vector[reduction],
        op=Operation.ADD,
    )
    return _nest((row, reduction), assignment)


def _build_dense_matmul() -> ForAll:
    row, reduction, col = IndexVar("r"), IndexVar("q"), IndexVar("c")
    out = TensorVar("DenseProduct", fmt="dd")
    left = TensorVar("DenseLeft", fmt="dd")
    right = TensorVar("DenseRight", fmt="dd")
    assignment = TensorAssign(
        out[row, col],
        left[row, reduction] * right[reduction, col],
        op=Operation.ADD,
    )
    return _nest((row, reduction, col), assignment)


def test_dense_elementwise_affine_free_axis_tiles_are_generic_and_ragged():
    schedule = Schedule(
        loop_order=("r", "c"),
        tiles=(
            TileSpec("r", 3, accum="direct", unroll=False),
            TileSpec(
                "c",
                4,
                placement="child_of:r_in",
                accum="direct",
                unroll=False,
            ),
        ),
        tag="generic-elementwise-r-c",
    )

    scheduled = Scheduler.apply_schedule(_build_elementwise("dd"), schedule)
    cpp = _lower_to_cpp(scheduled)

    assert "constexpr int kTile_r = 3;" in cpp
    assert "constexpr int kTile_c = 4;" in cpp
    assert cpp.index("r_out = 0") < cpp.index("r_in = 0")
    assert cpp.index("r_in = 0") < cpp.index("c_out = 0")
    assert cpp.index("c_out = 0") < cpp.index("c_in = 0")
    assert "if (r >= ElemLeft0_size)" in cpp
    assert "if (c >= ElemLeft1_size)" in cpp
    assert "packed_" not in cpp

    left = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    right = (torch.arange(35, dtype=torch.float32).reshape(7, 5) + 1) / 7
    result = einsum(
        "rc,rc->rc",
        STensor.from_torch(left, "ElemLeft"),
        STensor.from_torch(right, "ElemRight"),
        format="dd",
        schedule=schedule,
    )

    assert torch.allclose(result.to_torch(), left * right)


def test_dense_elementwise_affine_tile_and_cache_support_float64():
    schedule = Schedule(
        loop_order=("r", "c"),
        tiles=(TileSpec("r", 3, accum="direct", unroll=False),),
        tag="generic-elementwise-float64",
    )
    left = torch.arange(35, dtype=torch.float64).reshape(7, 5)
    right = torch.linspace(0.5, 2.0, 35, dtype=torch.float64).reshape(7, 5)

    float_result = einsum(
        "rc,rc->rc",
        STensor.from_torch(left.float(), "ElemLeft"),
        STensor.from_torch(right.float(), "ElemRight"),
        format="dd",
        schedule=schedule,
    )
    result = einsum(
        "rc,rc->rc",
        STensor.from_torch(left, "ElemLeft"),
        STensor.from_torch(right, "ElemRight"),
        format="dd",
        schedule=schedule,
    )

    assert float_result.dtype == torch.float32
    assert result.dtype == torch.float64
    assert torch.allclose(result.to_torch(), left * right)


def test_llir_continue_has_a_general_cpp_lowering():
    assert LLIRLowerer().lower_llir(llir.Continue()) == "continue;"


def test_schedule_access_matching_uses_typed_identity_not_display_spelling():
    tensor_id = SymbolId(1)
    index_ids = (IndexId(2), IndexId(3))
    access = llir.ArrayAccess(
        array=llir.Var("SameDisplay_val", llir.DataType.PTR_FLOAT32),
        index=llir.Var("same_position", llir.DataType.INT),
        tensor_access=llir.TensorAccessMetadata(
            access_id=AccessId(4),
            tensor_id=tensor_id,
            index_ids=index_ids,
            role=llir.TensorAccessRole.INPUT_READ,
        ),
    )

    assert _matches_tensor_access(
        access,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
    )
    assert not _matches_tensor_access(
        access,
        SymbolId(99),
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
    )
    assert not _matches_tensor_access(
        access,
        tensor_id,
        (IndexId(2), IndexId(99)),
        llir.TensorAccessRole.INPUT_READ,
    )
    assert not _matches_tensor_access(
        access,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.RESULT_WRITE,
    )
    same_logical_access = llir.ArrayAccess(
        array=llir.Var("DifferentDisplay_val", llir.DataType.PTR_FLOAT32),
        index=llir.Var("different_position", llir.DataType.INT),
        tensor_access=llir.TensorAccessMetadata(
            access_id=AccessId(999),
            tensor_id=tensor_id,
            index_ids=index_ids,
            role=llir.TensorAccessRole.INPUT_READ,
        ),
    )
    assert _matches_tensor_access(
        same_logical_access,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
    )


def test_schedule_access_rewrite_is_detached_repeatable_and_fail_closed():
    tensor_id = SymbolId(11)
    other_tensor_id = SymbolId(12)
    index_ids = (IndexId(13),)
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(14),
        tensor_id=tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    other_metadata = llir.TensorAccessMetadata(
        access_id=AccessId(15),
        tensor_id=other_tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    selected = llir.ArrayAccess(
        llir.Var("Input_val", llir.DataType.PTR_FLOAT32),
        llir.Var("pInput", llir.DataType.INT),
        metadata,
    )
    unselected = llir.ArrayAccess(
        llir.Var("Other_val", llir.DataType.PTR_FLOAT32),
        llir.Var("pOther", llir.DataType.INT),
        other_metadata,
    )
    source = llir.BinOp("+", selected, unselected)
    replacement = llir.ArrayAccess(
        llir.Var("packed_Input", llir.DataType.PTR_FLOAT32),
        llir.Var("packed_position", llir.DataType.INT),
    )
    lowerer = LLIRLowerer()
    source_cpp = lowerer.lower_llir(source)

    first, first_count = _rewrite_expr_access(
        source,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )
    second, second_count = _rewrite_expr_access(
        first,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )

    assert first_count == 1
    assert second_count == 0
    assert lowerer.lower_llir(source) == source_cpp
    assert (
        lowerer.lower_llir(first) == "packed_Input[packed_position] + Other_val[pOther]"
    )
    assert lowerer.lower_llir(second) == lowerer.lower_llir(first)
    assert first is not source
    assert second is not first
    assert first.left is not replacement
    assert first.right is not unselected
    assert first.right.tensor_access is other_metadata
    assert not _contains_tensor_access(
        [llir.Assign(llir.Var("out", llir.DataType.FLOAT32), first)],
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
    )

    class UnknownExpr(llir.Expr):
        pass

    malformed = llir.ArrayAccess(
        llir.Var("Input_val", llir.DataType.PTR_FLOAT32),
        UnknownExpr(),
    )
    with pytest.raises(LLIRTraversalError) as raised:
        _rewrite_expr_access(
            malformed,
            tensor_id,
            index_ids,
            llir.TensorAccessRole.INPUT_READ,
            replacement,
        )
    assert raised.value.diagnostic.code == "unknown_llir_node"
    assert raised.value.diagnostic.path == ("root", "index")


def test_schedule_statement_access_rewrite_counts_and_clones_each_replacement():
    tensor_id = SymbolId(21)
    index_ids = (IndexId(22),)
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(23),
        tensor_id=tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    access = llir.ArrayAccess(
        llir.Var("Input_val", llir.DataType.PTR_FLOAT32),
        llir.Var("pInput", llir.DataType.INT),
        metadata,
    )
    statements = [
        llir.Assign(llir.Var("left", llir.DataType.FLOAT32), access),
        llir.Assign(llir.Var("right", llir.DataType.FLOAT32), access),
    ]
    replacement = llir.ArrayAccess(
        llir.Var("packed_Input", llir.DataType.PTR_FLOAT32),
        llir.Var("packed_position", llir.DataType.INT),
    )

    first_count = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )
    first_values = [statement.value for statement in statements]
    second_count = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.INPUT_READ,
        replacement,
    )

    assert first_count == 2
    assert second_count == 0
    assert first_values[0] is not first_values[1]
    assert first_values[0] is not replacement
    assert first_values[1] is not replacement
    assert all(
        LLIRLowerer().lower_llir(statement.value) == "packed_Input[packed_position]"
        for statement in statements
    )


def test_schedule_result_target_rewrite_remains_a_valid_detached_lvalue() -> None:
    tensor_id = SymbolId(31)
    index_ids = (IndexId(32), IndexId(33))
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(34),
        tensor_id=tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.RESULT_WRITE,
    )
    target = llir.ArrayAccess(
        llir.Var("Result_values", llir.DataType.PTR_FLOAT32),
        llir.Var("pResult", llir.DataType.INT64),
        metadata,
    )
    replacement = llir.ArrayAccess(
        llir.Var("compact_Result", llir.DataType.PTR_FLOAT32),
        llir.Add(
            llir.Mul(llir.Var("row", llir.DataType.INT64), llir.Literal(4)),
            llir.Var("lane", llir.DataType.INT64),
        ),
    )
    statements = [llir.Assign(target, llir.Var("value", llir.DataType.FLOAT32))]

    count = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.RESULT_WRITE,
        replacement,
    )
    repeated = _rewrite_stmt_accesses(
        statements,
        tensor_id,
        index_ids,
        llir.TensorAccessRole.RESULT_WRITE,
        replacement,
    )

    rewritten = cast(llir.ArrayAccess, statements[0].var)
    assert count == 1
    assert repeated == 0
    assert rewritten is not replacement
    assert rewritten.array is not replacement.array
    assert rewritten.index is not replacement.index
    assert LLIRLowerer().lower_llir(statements[0]) == (
        "compact_Result[row * 4 + lane] = value;"
    )
    assert LLIRLowerer().lower_llir(target) == "Result_values[pResult]"

    malformed = [llir.Assign(target, llir.Var("value", llir.DataType.FLOAT32))]
    with pytest.raises(LLIRTraversalError) as raised:
        _rewrite_stmt_accesses(
            malformed,
            tensor_id,
            index_ids,
            llir.TensorAccessRole.RESULT_WRITE,
            llir.BinOp("+", llir.Var("left", llir.DataType.INT64), llir.Literal(1)),
        )
    assert raised.value.diagnostic.code == "invalid_assignment_target"
    assert raised.value.diagnostic.path == ("root", "[0]", "var")
    assert malformed[0].var is target
    assert LLIRLowerer().lower_llir(malformed[0]) == ("Result_values[pResult] = value;")


def test_schedule_nested_read_replacement_preflights_without_mutating_lvalue() -> None:
    tensor_id = SymbolId(41)
    index_ids = (IndexId(42),)
    metadata = llir.TensorAccessMetadata(
        access_id=AccessId(43),
        tensor_id=tensor_id,
        index_ids=index_ids,
        role=llir.TensorAccessRole.INPUT_READ,
    )
    indirect_index = llir.ArrayAccess(
        llir.Var("indices", llir.DataType.PTR_INT),
        llir.Var("i", llir.DataType.INT64),
        metadata,
    )
    target = llir.ArrayAccess(
        llir.Var("output", llir.DataType.PTR_FLOAT32),
        indirect_index,
    )
    statements = [llir.Assign(target, llir.Var("value", llir.DataType.FLOAT32))]

    with pytest.raises(LLIRTraversalError) as raised:
        _rewrite_stmt_accesses(
            statements,
            tensor_id,
            index_ids,
            llir.TensorAccessRole.INPUT_READ,
            llir.Array([llir.Literal(0)], llir.DataType.INT),
        )

    assert raised.value.diagnostic.code == "invalid_assignment_target"
    assert raised.value.diagnostic.path == ("root", "[0]", "var")
    assert statements[0].var is target
    assert cast(llir.ArrayAccess, statements[0].var).index is indirect_index
    assert LLIRLowerer().lower_llir(statements[0]) == ("output[indices[i]] = value;")


def test_dense_elementwise_llir_tensor_access_metadata_survives_rewrites():
    source = _build_elementwise("dd")
    lowered = CINLowerer().lower_IndexStmt(source)
    assert isinstance(lowered, llir.Function)

    tagged_expressions = []

    def collect_expr(expr):
        if isinstance(expr, llir.Var):
            if expr.tensor_access is not None:
                tagged_expressions.append(expr)
        elif isinstance(expr, llir.BinOp):
            collect_expr(expr.left)
            collect_expr(expr.right)
        elif isinstance(expr, llir.ArrayAccess):
            if expr.tensor_access is not None:
                tagged_expressions.append(expr)
            collect_expr(expr.array)
            collect_expr(expr.index)

    def collect_stmts(stmts):
        for stmt in stmts:
            if isinstance(stmt, llir.Assign):
                collect_expr(stmt.var)
                collect_expr(stmt.value)
            elif isinstance(stmt, llir.VarInit):
                collect_expr(stmt.value)
            elif isinstance(stmt, (llir.ForLoop, llir.WhileLoop)):
                collect_stmts(stmt.body)
            elif isinstance(stmt, llir.IfThenElse):
                for body in [stmt.then_body, stmt.else_body] + (
                    stmt.then_body_list or []
                ):
                    if body:
                        collect_stmts(body)

    collect_stmts(lowered.body)
    metadata = {expression.tensor_access for expression in tagged_expressions}
    expected_metadata = {
        llir.TensorAccessMetadata(
            access_id=access.access_id,
            tensor_id=access.tensor_id,
            index_ids=access.index_ids,
            role=(
                llir.TensorAccessRole.RESULT_WRITE
                if access.tensor.name == "ElemOut"
                else llir.TensorAccessRole.INPUT_READ
            ),
        )
        for access in source.tensor_accesses
    }
    assert metadata == expected_metadata
    assert all(
        type(expression) is llir.ArrayAccess for expression in tagged_expressions
    )
    result_writes = [
        cast(llir.ArrayAccess, expression)
        for expression in tagged_expressions
        if expression.tensor_access.role is llir.TensorAccessRole.RESULT_WRITE
    ]
    assert len(result_writes) == 1
    assert cast(llir.Var, result_writes[0].array).name == "ElemOut_values"
    # Dense pointer hoisting rewrites the physical access spelling, while the
    # logical tensor/index identity remains available to later schedule passes.
    lowerer = LLIRLowerer()
    assert {lowerer.lower_llir(expression) for expression in tagged_expressions} == {
        "_ElemLeft_val_ptr[c]",
        "_ElemRight_val_ptr[c]",
        "ElemOut_values[pElemOut1]",
    }

    for expression in tagged_expressions:
        if type(expression) is llir.Var:
            without_metadata = llir.Var(
                name=expression.name,
                type=expression.type,
                is_ptr=expression.is_ptr,
                is_restrict=expression.is_restrict,
            )
        else:
            assert type(expression) is llir.ArrayAccess
            without_metadata = llir.ArrayAccess(
                array=expression.array,
                index=expression.index,
            )
        assert lowerer.lower_llir(expression) == lowerer.lower_llir(without_metadata)


def test_ttm_row_and_free_axis_tiles_compose_with_ragged_tails():
    schedule = Schedule(
        loop_order=("a", "b", "c", "d"),
        tiles=(
            TileSpec("a", 2, accum="direct", unroll=False),
            TileSpec("d", 3, placement="child_of:b", accum="stack"),
        ),
        tag="generic-ttm-a-d",
    )

    scheduled = Scheduler.apply_schedule(_build_ttm(), schedule)
    cpp = _lower_to_cpp(scheduled)

    assert "constexpr int kTile_a = 2;" in cpp
    assert "constexpr int kTile_d = 3;" in cpp
    assert cpp.index("a_out = 0") < cpp.index("a_in = 0")
    assert cpp.index("a_in = 0") < cpp.index("d_out = 0")
    assert "float wksp[kTile_d] = {};" in cpp
    assert "if (a >= Core0_size)" in cpp
    assert "if (d >= Factor1_size)" in cpp
    assert "Projected_values" in cpp
    assert "packed_" not in cpp

    torch.manual_seed(101)
    core = torch.randn(5, 3, 4)
    factor = torch.randn(4, 7)
    result = einsum(
        "abc,cd->abd",
        STensor.from_torch(core, "Core"),
        STensor.from_torch(factor, "Factor"),
        format="ddd",
        schedule=schedule,
    )
    reference = torch.einsum("abc,cd->abd", core, factor)

    assert torch.allclose(result.to_torch(), reference, atol=1e-3, rtol=1e-3)


def test_ttm_heap_result_tile_is_generic_compact_accumulation():
    schedule = Schedule(
        loop_order=("a", "b", "c", "d"),
        tiles=(
            TileSpec(
                "d",
                3,
                placement="outermost",
                accum="heap",
                unroll=False,
            ),
        ),
        tag="generic-ttm-heap-d",
        parallel_loop="a",
    )

    scheduled = Scheduler.apply_schedule(_build_ttm(), schedule)
    cpp = _lower_to_cpp(scheduled)

    allocation = (
        "std::vector<float> tiled_Projected_storage("
        "(size_t) (Projected0_size * Projected1_size) * (size_t) kTile_d);"
    )
    init_loop = "Projected_tile_init < Projected0_size * Projected1_size"
    copy_loop = "Projected_tile_copy < Projected0_size * Projected1_size"
    assert allocation in cpp
    assert cpp.index(allocation) < cpp.index("d_out = 0")
    assert cpp.count(init_loop) == 1
    assert cpp.count(copy_loop) == 1
    assert "tiled_Projected[pProjected1 * kTile_d + d_in] +=" in cpp
    assert "Projected_values[pProjected2] +=" not in cpp
    assert "scorch_zero_dense(Projected_values, Projected_capacity)" not in cpp

    torch.manual_seed(105)
    core = torch.randn(5, 3, 4)
    factor = torch.randn(4, 7)
    result = einsum(
        "abc,cd->abd",
        STensor.from_torch(core, "Core"),
        STensor.from_torch(factor, "Factor"),
        format="ddd",
        schedule=schedule,
    )
    reference = torch.einsum("abc,cd->abd", core, factor)

    assert torch.allclose(result.to_torch(), reference, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    ("factory", "schedule", "message"),
    [
        pytest.param(
            lambda: _build_elementwise("dd"),
            Schedule(
                loop_order=("r", "c"),
                tiles=(TileSpec("c", 3, accum="heap", unroll=False),),
            ),
            "requires an enclosed reduction",
            id="elementwise-no-reduction",
        ),
        pytest.param(
            _build_sddmm,
            Schedule(
                loop_order=("r", "c", "q"),
                tiles=(TileSpec("r", 3, accum="heap", unroll=False),),
            ),
            "tiled sparse-output assembly",
            id="sddmm-sparse-result",
        ),
        pytest.param(
            _build_spgemm,
            Schedule(
                loop_order=("r", "q", "c"),
                tiles=(TileSpec("r", 3, accum="heap", unroll=False),),
            ),
            "tiled sparse-output assembly",
            id="spgemm-sparse-result",
        ),
        pytest.param(
            _build_spmv,
            Schedule(
                loop_order=("r", "q"),
                tiles=(TileSpec("r", 3, accum="heap", unroll=False),),
            ),
            "trailing storage level",
            id="spmv-rank-one-result",
        ),
        pytest.param(
            _build_ttm,
            Schedule(
                loop_order=("a", "b", "c", "d"),
                tiles=(
                    TileSpec(
                        "d",
                        3,
                        placement="outermost",
                        accum="heap",
                    ),
                ),
            ),
            "requires an explicit parallel dense result-prefix loop",
            id="ttm-heap-missing-safe-parallel-anchor",
        ),
        pytest.param(
            _build_ttm,
            Schedule(
                loop_order=("a", "b", "c", "d"),
                tiles=(
                    TileSpec(
                        "d",
                        3,
                        placement="outermost",
                        accum="heap",
                    ),
                    TileSpec(
                        "a",
                        2,
                        placement="outermost",
                        accum="direct",
                    ),
                ),
                parallel_loop="a",
            ),
            "requires.*remain outermost",
            id="ttm-later-affine-tile-wraps-heap-tile",
        ),
        pytest.param(
            _build_ttm,
            Schedule(
                loop_order=("a", "b", "c", "d"),
                tiles=(
                    TileSpec(
                        "d",
                        3,
                        placement="child_of:b",
                        accum="heap",
                    ),
                ),
            ),
            "requires.*outermost",
            id="ttm-unsafe-lifetime",
        ),
    ],
)
def test_heap_result_tile_rejects_unsupported_non_spmm_structures_before_lowering(
    factory, schedule, message
):
    stmt = factory()
    original = str(stmt)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        Scheduler.apply_schedule(stmt, schedule)
    assert str(stmt) == original


def test_sparse_elementwise_affine_tile_is_rejected_before_lowering():
    stmt = _build_elementwise("ds")
    original = str(stmt)
    schedule = Schedule(
        loop_order=("r", "c"),
        tiles=(TileSpec("r", 3, accum="direct", unroll=False),),
        tag="unsupported-sparse-elementwise-row-tile",
    )

    with pytest.raises(NotImplementedError, match="tiled sparse-output assembly"):
        Scheduler.apply_schedule(stmt, schedule)

    assert str(stmt) == original


def test_sddmm_default_scalar_accumulator_and_simd_are_unchanged():
    with regblock_force(False):
        scheduled = Scheduler.auto_schedule(_build_sddmm())
        cpp = _lower_to_cpp(scheduled)

    assert not scheduled.inserted_workspace
    assert "float _accum = 0.0f;" in cpp
    assert "#pragma omp simd" in cpp
    assert "wksp" not in cpp
    assert "packed_" not in cpp
    assert "int64_t _known_nnz = Mask_values.size(0);" in cpp
    assert "torch::Tensor Sampled0_crd_torch = torch::empty" in cpp
    assert "std::vector<float> Sampled_values" not in cpp

    torch.manual_seed(102)
    mask = torch.randn(5, 7)
    mask *= torch.rand(5, 7) < 0.35
    query = torch.randn(5, 3)
    key = torch.randn(7, 3)
    result = einsum(
        "rc,rq,cq->rc",
        _sparse_stensor(mask, "Mask"),
        STensor.from_torch(query, "Query"),
        STensor.from_torch(key, "Key"),
        format="oo",
    )
    reference = mask * (query @ key.T)

    assert torch.allclose(result.to_torch(), reference, atol=1e-3, rtol=1e-3)


def test_all_coo_sddmm_has_no_redundant_end_and_remains_correct():
    row, column, reduction = IndexVar("i"), IndexVar("j"), IndexVar("k")
    out = TensorVar("Sampled", fmt="oo")
    mask_var = TensorVar("Mask", fmt="oo")
    query_var = TensorVar("Query", fmt="dd")
    key_var = TensorVar("Key", fmt="dd")
    assignment = TensorAssign(
        out[row, column],
        mask_var[row, column] * query_var[row, reduction] * key_var[column, reduction],
        op=Operation.ADD,
    )

    with regblock_force(False):
        scheduled = Scheduler.auto_schedule(_nest((row, column, reduction), assignment))
        cpp = _lower_to_cpp(scheduled)

    assert "pMask1_end" not in cpp
    assert "Mask1_crd[pMask0]" in cpp
    assert "Mask_val[pMask0]" in cpp

    torch.manual_seed(103)
    mask = torch.randn(5, 7)
    mask *= torch.rand(5, 7) < 0.35
    query = torch.randn(5, 3)
    key = torch.randn(7, 3)
    result = einsum(
        "ij,ik,jk->ij",
        _sparse_stensor(mask, "Mask", "oo"),
        STensor.from_torch(query, "Query"),
        STensor.from_torch(key, "Key"),
        format="oo",
        schedule=Schedule(
            loop_order=("i", "j", "k"),
            tag="all-coo-pmask-end-native",
        ),
    )
    reference = mask * (query @ key.T)

    assert torch.allclose(result.to_torch(), reference, atol=1e-3, rtol=1e-3)


def test_sddmm_affine_reduction_tile_is_rejected_during_validation():
    schedule = Schedule(
        loop_order=("r", "c", "q"),
        tiles=(
            TileSpec(
                "q",
                2,
                placement="child_of:c",
                accum="direct",
            ),
        ),
        tag="unsupported-sddmm-reduction-tile",
    )

    with pytest.raises(NotImplementedError, match="Affine reduction tiling"):
        Scheduler.apply_schedule(_build_sddmm(), schedule)


def test_spgemm_default_workspace_and_sparse_assembly_are_unchanged():
    with regblock_force(False):
        auto_scheduled = Scheduler.auto_schedule(_build_spgemm())
        empty_scheduled = Scheduler.apply_schedule(_build_spgemm(), Schedule())
        auto_cpp = _lower_to_cpp(auto_scheduled)
        empty_cpp = _lower_to_cpp(empty_scheduled)

    _, auto_body = Scheduler._extract_loop_chain(auto_scheduled)
    assert auto_scheduled.inserted_workspace
    assert isinstance(auto_body, Where)
    empty_working = legacy_cin_working_copy(
        empty_scheduled.normalized_cin,
        empty_scheduled.verified_loop_plan,
    )
    assert str(empty_working) == str(auto_scheduled)
    assert empty_cpp == auto_cpp
    assert "std::vector<linked_list_workspace_1d" in auto_cpp
    assert auto_cpp.count("].make_view()") == 2
    assert ", true);" in auto_cpp
    assert "std::current_exception()" not in auto_cpp
    assert "for (int _worker = 0" in auto_cpp
    assert ".insert_unchecked(" in auto_cpp
    assert "torch::Tensor SparseProduct1_crd_torch = torch::empty" in auto_cpp
    assert "SparseProduct1_pos_data" in auto_cpp
    assert "packed_" not in auto_cpp

    torch.manual_seed(103)
    left = torch.randn(5, 6)
    right = torch.randn(6, 4)
    left *= torch.rand(5, 6) < 0.35
    right *= torch.rand(6, 4) < 0.4
    result = einsum(
        "rq,qc->rc",
        _sparse_stensor(left, "SparseLeft"),
        _sparse_stensor(right, "SparseRight"),
        format="ds",
    )

    assert torch.allclose(result.to_torch(), left @ right, atol=1e-3, rtol=1e-3)


def test_spgemm_affine_tile_is_rejected_before_sparse_output_assembly():
    schedule = Schedule(
        loop_order=("r", "q", "c"),
        tiles=(TileSpec("r", 2, accum="direct", unroll=False),),
        tag="unsupported-spgemm-row-tile",
    )

    with pytest.raises(NotImplementedError, match="tiled sparse-output assembly"):
        Scheduler.apply_schedule(_build_spgemm(), schedule)


@pytest.mark.parametrize(
    ("factory", "loop_order"),
    [
        pytest.param(_build_spmv, ("r", "q"), id="spmv"),
        pytest.param(_build_dense_matmul, ("r", "q", "c"), id="dense-matmul"),
    ],
)
def test_spmv_and_dense_matmul_empty_schedule_preserve_default_codegen(
    factory, loop_order
):
    with regblock_force(False):
        auto_scheduled = Scheduler.auto_schedule(factory())
        empty_scheduled = Scheduler.apply_schedule(factory(), Schedule())
        auto_cpp = _lower_to_cpp(auto_scheduled)
        empty_cpp = _lower_to_cpp(empty_scheduled)

    empty_working = legacy_cin_working_copy(
        empty_scheduled.normalized_cin,
        empty_scheduled.verified_loop_plan,
    )
    assert str(empty_working) == str(auto_scheduled)
    assert empty_cpp == auto_cpp
    assert "packed_" not in auto_cpp


@pytest.mark.parametrize(
    ("factory", "loop_order"),
    [
        pytest.param(_build_spmv, ("r", "q"), id="spmv"),
        pytest.param(_build_dense_matmul, ("r", "q", "c"), id="dense-matmul"),
    ],
)
def test_spmv_and_dense_matmul_reduction_tiles_are_rejected(factory, loop_order):
    schedule = Schedule(
        loop_order=loop_order,
        tiles=(TileSpec("q", 2, accum="direct"),),
        tag="unsupported-reduction-tile",
    )

    with pytest.raises(NotImplementedError, match="Affine reduction tiling"):
        Scheduler.apply_schedule(factory(), schedule)


def test_spmv_and_dense_matmul_default_numerics_are_unchanged():
    torch.manual_seed(104)

    sparse_matrix = torch.randn(7, 5)
    sparse_matrix *= torch.rand(7, 5) < 0.35
    vector = torch.randn(5)
    spmv_result = einsum(
        "rq,q->r",
        _sparse_stensor(sparse_matrix, "SparseMatrix"),
        STensor.from_torch(vector, "DenseVector"),
        format="d",
    )
    assert torch.allclose(
        spmv_result.to_torch(), sparse_matrix @ vector, atol=1e-3, rtol=1e-3
    )

    left = torch.randn(5, 3)
    right = torch.randn(3, 6)
    dense_result = einsum(
        "rq,qc->rc",
        STensor.from_torch(left, "DenseLeft"),
        STensor.from_torch(right, "DenseRight"),
        format="dd",
    )
    assert torch.allclose(dense_result.to_torch(), left @ right, atol=1e-3, rtol=1e-3)
