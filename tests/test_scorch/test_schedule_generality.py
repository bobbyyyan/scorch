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


def test_dense_elementwise_llir_tensor_access_metadata_survives_rewrites():
    lowered = CINLowerer().lower_IndexStmt(_build_elementwise("dd"))
    assert isinstance(lowered, llir.Function)

    tagged_vars = []

    def collect_expr(expr):
        if isinstance(expr, llir.Var):
            if expr.tensor_access is not None:
                tagged_vars.append(expr)
        elif isinstance(expr, llir.BinOp):
            collect_expr(expr.left)
            collect_expr(expr.right)
        elif isinstance(expr, llir.ArrayAccess):
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
    metadata = {var.tensor_access for var in tagged_vars}
    assert metadata == {
        llir.TensorAccessMetadata(
            "ElemLeft",
            ("r", "c"),
            llir.TensorAccessRole.INPUT_READ,
        ),
        llir.TensorAccessMetadata(
            "ElemRight",
            ("r", "c"),
            llir.TensorAccessRole.INPUT_READ,
        ),
        llir.TensorAccessMetadata(
            "ElemOut",
            ("r", "c"),
            llir.TensorAccessRole.RESULT_WRITE,
        ),
    }
    # Dense pointer hoisting rewrites the physical access spelling, while the
    # logical tensor/index identity remains available to later schedule passes.
    assert {var.name for var in tagged_vars} == {
        "_ElemLeft_val_ptr[c]",
        "_ElemRight_val_ptr[c]",
        "ElemOut_values[pElemOut1]",
    }

    cpp_with_metadata = LLIRLowerer().lower_llir(lowered)
    for var in tagged_vars:
        var.tensor_access = None
    assert LLIRLowerer().lower_llir(lowered) == cpp_with_metadata


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
    assert str(empty_scheduled) == str(auto_scheduled)
    assert empty_cpp == auto_cpp
    assert "linked_list_workspace_1d" in auto_cpp
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

    assert str(empty_scheduled) == str(auto_scheduled)
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
