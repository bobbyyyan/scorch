import hashlib
from typing import cast

import torch
import pytest

from scorch import STensor
from scorch.compiler import llir  # type: ignore[import-untyped]
from scorch.compiler.cin import ForAll, IndexVar, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.llir_traversal import (  # type: ignore[import-untyped]
    LLIRTraversalContext,
    LLIRWalker,
)
from scorch.compiler.scheduler import (
    RelayoutSpec,
    Schedule,
    ScheduledCIN,
    Scheduler,
    TileSpec,
    schedule_force,
)
from scorch.ops import matmul


def _free_k_schedule(tile_width: int = 4) -> Schedule:
    return Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                index_var="k",
                width=tile_width,
                placement="child_of:i",
                accum="stack",
            ),
        ),
        tag=f"free-k-t{tile_width}",
    )


def _panel_j_schedule(tile_width: int = 4) -> Schedule:
    return Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                index_var="j",
                width=tile_width,
                placement="outermost",
                kind="panel",
                accum="direct",
            ),
        ),
        tag=f"panel-j-t{tile_width}",
        parallel_loop="i",
    )


def _packed_tileijk_schedule(
    nc: int = 4,
    jc: int = 3,
    *,
    scope_var: str = "j",
    accum: str = "direct",
) -> Schedule:
    return Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                nc,
                placement="outermost",
                accum=accum,
                unroll=False,
            ),
            TileSpec(
                "j",
                jc,
                placement="child_of:k_out",
                kind="panel",
                accum="direct",
            ),
        ),
        relayout=RelayoutSpec("B", "k", nc, scope_var=scope_var),
        tag=f"packed-tile-ijk-{scope_var}-{accum}",
        parallel_loop="i",
    )


def _build_spmm_cin():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")
    c[i, k] = a[i, j] * b[j, k]
    return ForAll(i, ForAll(j, ForAll(k, c._assignment)))


def _lower_free_k_workspace_copy_read() -> (
    tuple[llir.ArrayAccess, tuple[llir.Var, ...]]
):
    statement = _build_spmm_cin()
    scheduled = Scheduler.apply_schedule(statement, _free_k_schedule())
    lowered = CINLowerer().lower_IndexStmt(scheduled)
    reads: list[llir.ArrayAccess] = []
    tile_loop_vars: list[llir.Var] = []

    class WorkspaceCopyCollector(LLIRWalker):
        def visit_assign(
            self,
            node: llir.Assign,
            path: tuple[str, ...],
        ) -> None:
            if type(node.value) is llir.ArrayAccess:
                value = cast(llir.ArrayAccess, node.value)
                if (
                    type(value.array) is llir.Var
                    and cast(llir.Var, value.array).name == "wksp"
                ):
                    reads.append(value)
            super().visit_assign(node, path)

        def visit_for_loop(
            self,
            node: llir.ForLoop,
            path: tuple[str, ...],
        ) -> None:
            if (
                type(node.init) is llir.VarInit
                and type(node.init.var) is llir.Var
                and node.init.var.name == "k_in"
            ):
                tile_loop_vars.append(node.init.var)
            super().visit_for_loop(node, path)

    WorkspaceCopyCollector(
        LLIRTraversalContext(
            stage="test",
            pass_name="collect_free_k_workspace_copy_read",
        )
    ).walk(lowered)
    assert len(reads) == 1
    assert len(tile_loop_vars) == 2
    return reads[0], tuple(tile_loop_vars)


def _lower_free_k_stack_workspace(
    scheduled: ScheduledCIN,
) -> tuple[llir.FixedStackArrayDecl, llir.VarInit, str]:
    lowered = CINLowerer().lower_IndexStmt(scheduled)
    declarations: list[llir.FixedStackArrayDecl] = []
    tile_size_initializers: list[llir.VarInit] = []

    class StackWorkspaceCollector(LLIRWalker):
        def visit_fixed_stack_array_decl(
            self,
            node: llir.FixedStackArrayDecl,
            path: tuple[str, ...],
        ) -> None:
            declarations.append(node)
            super().visit_fixed_stack_array_decl(node, path)

        def visit_var_init(
            self,
            node: llir.VarInit,
            path: tuple[str, ...],
        ) -> None:
            if type(node.var) is llir.Var and node.var.name == "kTile_k":
                tile_size_initializers.append(node)
            super().visit_var_init(node, path)

    StackWorkspaceCollector(
        LLIRTraversalContext(
            stage="test",
            pass_name="collect_free_k_stack_workspace",
        )
    ).walk(lowered)
    assert len(declarations) == 1
    assert len(tile_size_initializers) == 1
    return (
        declarations[0],
        tile_size_initializers[0],
        LLIRLowerer().lower_llir(lowered),
    )


def test_tuner_free_k_schedule_emits_row_outer_stack_tile():
    scheduled = Scheduler.apply_schedule(_build_spmm_cin(), _free_k_schedule())
    lowered = CINLowerer().lower_IndexStmt(scheduled)
    cpp = LLIRLowerer().lower_llir(lowered)

    row_loop = "for (int64_t i = 0; i < A0_size; i++)"
    tile_loop = "for (int64_t k_out = 0; k_out < B1_size; " "k_out += kTile_k)"

    assert "constexpr int kTile_k = 4;" in cpp
    assert cpp.count("#pragma omp parallel for") == 1
    assert "scorch_nthreads(A1_pos[A0_size], A0_size)" in cpp
    assert cpp.index(row_loop) < cpp.index(tile_loop)
    assert "float wksp[kTile_k] = {};" in cpp
    assert "for (int pA1 = A1_pos[pA0]; pA1 < pA1_end; pA1++)" in cpp
    assert "int64_t k = k_out + k_in;" in cpp
    assert "if (k >= B1_size)" in cpp
    assert "if (k >= C1_size)" in cpp
    assert "wksp[k_in] += A_val[pA1] * B_val[pB1];" in cpp
    assert "C_values[pC1] += wksp[k_in];" in cpp
    assert "aligned_alloc" not in cpp
    assert "A1_pos[B1_size]" not in cpp


def test_tuner_free_k_stack_workspace_is_structured_owned_and_byte_exact() -> None:
    def ownership_state(cin: ForAll) -> tuple[object, ...]:
        return (
            str(cin),
            tuple(
                (
                    id(index_var),
                    index_var.index_id,
                    index_var.name,
                    index_var.is_tiled,
                    index_var.is_outer,
                    index_var.is_inner,
                    index_var.tile_size_var,
                    index_var._parent,
                    tuple(index_var._legacy_tensor_accesses),
                )
                for index_var in cin.index_vars
            ),
            tuple(
                (
                    id(access),
                    access.access_id,
                    id(access.tensor),
                    tuple(id(index_var) for index_var in access.indices),
                    access.index_ids,
                )
                for access in cin.tensor_accesses
            ),
        )

    statement = _build_spmm_cin()
    original_state = ownership_state(statement)
    scheduled = Scheduler.apply_schedule(statement, _free_k_schedule())
    scheduled_cin = cast(ForAll, scheduled.normalized_cin)
    scheduled_state = ownership_state(scheduled_cin)
    scheduled_plan = scheduled.verified_loop_plan

    first, first_tile_size, first_cpp = _lower_free_k_stack_workspace(scheduled)
    second, second_tile_size, second_cpp = _lower_free_k_stack_workspace(scheduled)

    expected = llir.FixedStackArrayDecl(
        name="wksp",
        element_type=llir.DataType.FLOAT32,
        extent=llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
        initializer=llir.Array(values=[], data_type=llir.DataType.FLOAT32),
    )
    assert type(first) is llir.FixedStackArrayDecl
    assert type(second) is llir.FixedStackArrayDecl
    assert first == second == expected
    assert hash(first) == hash(second) == hash(expected)
    assert first is not second
    assert first.name == "wksp"
    assert type(first.name) is str
    assert first.element_type is llir.DataType.FLOAT32
    assert type(first.extent) is llir.Var
    assert first.extent.name == "kTile_k"
    assert first.extent.type is llir.DataType.CONSTEXPR_INT
    assert first.extent.is_ptr is False
    assert first.extent.is_restrict is False
    assert first.extent.tensor_access is None
    assert type(first.initializer) is llir.Array
    assert first.initializer.values == ()
    assert first.initializer.data_type is llir.DataType.FLOAT32

    assert first.extent is not second.extent
    assert first.initializer is not second.initializer
    assert (
        first_tile_size
        == second_tile_size
        == llir.VarInit(
            var=llir.Var("kTile_k", llir.DataType.CONSTEXPR_INT),
            value=llir.Literal(4),
        )
    )
    assert first.extent == first_tile_size.var
    assert second.extent == second_tile_size.var
    assert first.extent is not first_tile_size.var
    assert second.extent is not second_tile_size.var
    assert first_tile_size is not second_tile_size
    assert first_tile_size.var is not second_tile_size.var
    assert LLIRLowerer().lower_llir(first) == "float wksp[kTile_k] = {};"

    assert first_cpp == second_cpp
    assert first_cpp.count("float wksp[kTile_k] = {};") == 1
    assert len(first_cpp) == 3060
    assert hashlib.sha256(first_cpp.encode()).hexdigest() == (
        "2b9d28654e33225e1093500fc861e76f0f67cb241c7ab28b43c05b774d9f7222"
    )

    assert ownership_state(statement) == original_state
    assert scheduled.normalized_cin is scheduled_cin
    assert ownership_state(scheduled_cin) == scheduled_state
    assert scheduled.verified_loop_plan is scheduled_plan


def test_tuner_free_k_workspace_copy_read_is_structured_typed_and_owned() -> None:
    first, first_loop_vars = _lower_free_k_workspace_copy_read()
    second, second_loop_vars = _lower_free_k_workspace_copy_read()

    expected = llir.ArrayAccess(
        array=llir.Var("wksp", llir.DataType.PTR_FLOAT32),
        index=llir.Var("k_in", llir.DataType.INT64),
    )
    assert first == expected
    assert second == expected
    assert first is not second
    assert first.array is not second.array
    assert first.index is not second.index
    assert all(first.index is not loop_var for loop_var in first_loop_vars)
    assert all(second.index is not loop_var for loop_var in second_loop_vars)
    assert first.tensor_access is None
    assert LLIRLowerer().lower_llir(first) == "wksp[k_in]"


def test_tuner_free_k_schedule_is_correct_for_ragged_tail_and_empty_row():
    torch.manual_seed(0)
    m, j, n = 7, 11, 11
    a = torch.randn(m, j, dtype=torch.float32)
    a = a * (torch.rand(m, j) < 0.35)
    a[1] = 0
    b = torch.randn(j, n, dtype=torch.float32)

    result = matmul(
        a.to_sparse_csr(),
        b,
        schedule=_free_k_schedule(),
    )
    reference = a @ b

    assert torch.allclose(result, reference, atol=1e-3, rtol=1e-3)
    assert torch.count_nonzero(result[1]).item() == 0

    with schedule_force(_free_k_schedule()):
        context_result = matmul(a.to_sparse_csr(), b)
    assert torch.allclose(context_result, reference, atol=1e-3, rtol=1e-3)


def test_tuner_row_schedule_is_correct_for_ragged_tail_and_empty_row():
    torch.manual_seed(3)
    m, j, n = 11, 7, 5
    a = torch.randn(m, j, dtype=torch.float32)
    a = a * (torch.rand(m, j) < 0.3)
    a[4] = 0
    b = torch.randn(j, n, dtype=torch.float32)
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                index_var="i",
                width=4,
                placement="outermost",
                accum="direct",
                unroll=False,
            ),
        ),
        tag="row-i-t4",
    )

    result = matmul(
        a.to_sparse_csr(),
        b,
        use_cache=False,
        schedule=schedule,
    )
    reference = a @ b

    assert torch.allclose(result, reference, atol=1e-3, rtol=1e-3)
    assert torch.count_nonzero(result[4]).item() == 0


def test_tuner_parallel_free_tile_is_correct_when_columns_exceed_rows():
    torch.manual_seed(4)
    m, j, n = 3, 7, 17
    a = torch.randn(m, j, dtype=torch.float32)
    a = a * (torch.rand(m, j) < 0.4)
    b = torch.randn(j, n, dtype=torch.float32)
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                4,
                parallel=True,
                accum="direct",
                unroll=False,
            ),
        ),
        tag="parallel-free-k",
    )

    result = matmul(a.to_sparse_csr(), b, schedule=schedule)

    assert result.shape == (m, n)
    assert torch.allclose(result, a @ b, atol=1e-3, rtol=1e-3)


def test_tuner_panel_j_schedule_is_correct_for_tail_and_empty_row():
    torch.manual_seed(5)
    m, j, n = 8, 11, 7
    a = torch.randn(m, j, dtype=torch.float32)
    a = a * (torch.rand(m, j) < 0.35)
    a[2] = 0
    b = torch.randn(j, n, dtype=torch.float32)

    result = matmul(
        a.to_sparse_csr(),
        b,
        use_cache=False,
        schedule=_panel_j_schedule(),
    )
    reference = a @ b

    assert torch.allclose(result, reference, atol=1e-3, rtol=1e-3)
    assert torch.count_nonzero(result[2]).item() == 0


def test_tuner_panel_j_and_free_k_tiles_compose():
    torch.manual_seed(6)
    m, j, n = 7, 10, 11
    a = torch.randn(m, j, dtype=torch.float32)
    a = a * (torch.rand(m, j) < 0.4)
    a[1] = 0
    b = torch.randn(j, n, dtype=torch.float32)
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                4,
                placement="outermost",
                accum="direct",
                unroll=False,
            ),
            TileSpec(
                "j",
                3,
                placement="child_of:k_out",
                kind="panel",
                accum="direct",
            ),
        ),
        tag="panel-j-free-k",
        parallel_loop="i",
    )

    result = matmul(
        a.to_sparse_csr(),
        b,
        use_cache=False,
        schedule=schedule,
    )
    reference = a @ b

    assert torch.allclose(result, reference, atol=1e-3, rtol=1e-3)
    assert torch.count_nonzero(result[1]).item() == 0


def test_tuner_packed_tileijk_emits_pack_before_parallel_compute():
    scheduled = Scheduler.apply_schedule(
        _build_spmm_cin(),
        _packed_tileijk_schedule(),
    )
    cpp = LLIRLowerer().lower_llir(CINLowerer().lower_IndexStmt(scheduled))

    allocation = (
        "std::vector<float> packed_B_storage((size_t) kTile_j * " "(size_t) kTile_k);"
    )
    pack_loop = "for (int64_t j_pack = j_out; j_pack < j_out_end; j_pack++)"
    row_loop = "for (int64_t i = 0; i < A0_size; i++)"
    packed_read = "packed_B[(j - j_out) * kTile_k + k_in]"

    assert allocation in cpp
    assert "packed_B_storage.data()" in cpp
    assert "packed_B[(j_pack - j_out) * kTile_k + k_pack] = " in cpp
    assert "B_val[j_pack * B1_size + k_packed]" in cpp
    assert packed_read in cpp
    assert "B_val[pB1]" not in cpp
    assert "__builtin_prefetch(&packed_B[" in cpp
    assert cpp.index("k_out = 0") < cpp.index("j_out = 0")
    assert cpp.index(pack_loop) < cpp.index(row_loop)
    assert cpp.count("scorch_zero_dense(C_values, C_capacity)") == 1
    assert "C_values[pC1] += A_val[pA1] * " + packed_read in cpp


@pytest.mark.parametrize(
    ("scope_var", "accum"),
    [("j", "direct"), ("k", "direct"), ("j", "heap"), ("k", "heap")],
    ids=("panel-direct", "full-direct", "panel-compact", "full-compact"),
)
def test_tuner_packed_tileijk_storage_scopes_and_loop_nesting(scope_var, accum):
    schedule = _packed_tileijk_schedule(scope_var=scope_var, accum=accum)
    scheduled = Scheduler.apply_schedule(_build_spmm_cin(), schedule)
    cpp = LLIRLowerer().lower_llir(CINLowerer().lower_IndexStmt(scheduled))

    k_loop = "for (int64_t k_out = 0; k_out < B1_size; k_out += kTile_k)"
    panel_loop = "for (int64_t j_out = 0; j_out < B0_size; j_out += kTile_j)"
    row_loop = "for (int64_t i = 0; i < A0_size; i++)"
    if scope_var == "j":
        allocation = (
            "std::vector<float> packed_B_storage((size_t) kTile_j * "
            "(size_t) kTile_k);"
        )
        pack_loop = "for (int64_t j_pack = j_out; j_pack < j_out_end; j_pack++)"
        packed_read = "packed_B[(j - j_out) * kTile_k + k_in]"
        assert cpp.index(panel_loop) < cpp.index(pack_loop) < cpp.index(row_loop)
        assert "packed_B[(j_pack - j_out) * kTile_k + k_pack] =" in cpp
    else:
        allocation = (
            "std::vector<float> packed_B_storage((size_t) B0_size * "
            "(size_t) kTile_k);"
        )
        pack_loop = "for (int64_t j_pack = 0; j_pack < B0_size; j_pack++)"
        packed_read = "packed_B[j * kTile_k + k_in]"
        assert cpp.index(k_loop) < cpp.index(pack_loop) < cpp.index(panel_loop)
        assert "packed_B[j_pack * kTile_k + k_pack] =" in cpp

    assert cpp.index(allocation) < cpp.index(k_loop)
    assert cpp.index(pack_loop) < cpp.index(row_loop)
    assert "B_val[j_pack * B1_size + k_packed]" in cpp
    assert "B_val[pB1]" not in cpp
    assert packed_read in cpp

    if accum == "direct":
        assert cpp.count("scorch_zero_dense(C_values, C_capacity)") == 1
        assert "tiled_C_storage" not in cpp
        assert f"C_values[pC1] += A_val[pA1] * {packed_read}" in cpp
    else:
        allocation_c = (
            "std::vector<float> tiled_C_storage((size_t) (C0_size) * "
            "(size_t) kTile_k);"
        )
        init_loop = (
            "for (int64_t C_tile_init = 0; C_tile_init < C0_size; " "C_tile_init++)"
        )
        copy_loop = (
            "for (int64_t C_tile_copy = 0; C_tile_copy < C0_size; " "C_tile_copy++)"
        )
        assert cpp.index(allocation_c) < cpp.index(k_loop)
        assert cpp.count(init_loop) == 1
        assert cpp.count(copy_loop) == 1
        assert cpp.index(k_loop) < cpp.index(init_loop) < cpp.index(panel_loop)
        assert cpp.index(panel_loop) < cpp.index(copy_loop)
        assert "scorch_zero_dense(C_values, C_capacity)" not in cpp
        assert f"tiled_C[pC0 * kTile_k + k_in] += A_val[pA1] * {packed_read}" in cpp
        assert "C_values[pC1] +=" not in cpp
        assert (
            "C_values[C_tile_copy * C1_size + k_copy_logical] = "
            "tiled_C[C_tile_copy * kTile_k + k_tile_copy];"
        ) in cpp


@pytest.mark.parametrize(
    ("m", "j", "n", "empty_row"),
    [
        (7, 10, 11, 2),
        (5, 13, 6, 4),
        (9, 2, 17, 0),
    ],
    ids=("ragged-all", "rectangular", "wide-multi-k-panel"),
)
@pytest.mark.parametrize(
    ("scope_var", "accum"),
    [("j", "direct"), ("k", "direct"), ("j", "heap"), ("k", "heap")],
    ids=("panel-direct", "full-direct", "panel-compact", "full-compact"),
)
def test_tuner_packed_tileijk_matches_torch_with_ragged_panels(
    m, j, n, empty_row, scope_var, accum
):
    torch.manual_seed(m * 100 + j * 10 + n)
    a = torch.randn(m, j, dtype=torch.float32)
    a = a * (torch.rand(m, j) < 0.4)
    a[empty_row] = 0
    b = torch.randn(j, n, dtype=torch.float32)

    result = matmul(
        a.to_sparse_csr(),
        b,
        schedule=_packed_tileijk_schedule(scope_var=scope_var, accum=accum),
    )
    reference = a @ b

    assert result.shape == (m, n)
    assert torch.allclose(result, reference, atol=1e-4, rtol=1e-4)
    assert torch.count_nonzero(result[empty_row]).item() == 0


@pytest.mark.parametrize(("m", "j", "n"), [(4, 0, 7), (4, 5, 0), (0, 5, 7)])
@pytest.mark.parametrize(
    ("scope_var", "accum"),
    [("j", "direct"), ("k", "direct"), ("j", "heap"), ("k", "heap")],
    ids=("panel-direct", "full-direct", "panel-compact", "full-compact"),
)
def test_tuner_packed_tileijk_handles_zero_sized_domains(m, j, n, scope_var, accum):
    a = torch.zeros((m, j), dtype=torch.float32)
    b = torch.randn((j, n), dtype=torch.float32)

    result = matmul(
        a.to_sparse_csr(),
        b,
        schedule=_packed_tileijk_schedule(scope_var=scope_var, accum=accum),
    )

    assert result.shape == (m, n)
    assert torch.equal(result, a @ b)


@pytest.mark.parametrize(
    ("scope_var", "accum"),
    [("j", "direct"), ("k", "direct"), ("j", "heap"), ("k", "heap")],
    ids=("panel-direct", "full-direct", "panel-compact", "full-compact"),
)
def test_tuner_packed_tileijk_uses_generated_float64_storage(scope_var, accum):
    torch.manual_seed(21)
    a = torch.randn(4, 7, dtype=torch.float64)
    a = a * (torch.rand(4, 7) < 0.4)
    b = torch.randn(7, 5, dtype=torch.float64)
    stmt = _build_spmm_cin()
    for access in stmt.tensor_accesses:
        access.tensor.dtype = torch.float64

    schedule = _packed_tileijk_schedule(scope_var=scope_var, accum=accum)
    scheduled = Scheduler.apply_schedule(stmt, schedule)
    cpp = LLIRLowerer().lower_llir(CINLowerer().lower_IndexStmt(scheduled))
    result = matmul(a.to_sparse_csr(), b, schedule=schedule)

    assert "std::vector<double> packed_B_storage" in cpp
    assert "double* __restrict__ packed_B" in cpp
    if accum == "heap":
        assert "std::vector<double> tiled_C_storage" in cpp
        assert "double* __restrict__ tiled_C" in cpp
    assert torch.allclose(result, a @ b, atol=1e-10, rtol=1e-10)


def test_tuner_i_j_k_tiles_compose_with_all_ragged_tails():
    torch.manual_seed(11)
    m, j, n = 9, 10, 11
    a = torch.randn(m, j, dtype=torch.float32)
    a = a * (torch.rand(m, j) < 0.35)
    a[3] = 0
    b = torch.randn(j, n, dtype=torch.float32)
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "i",
                3,
                placement="outermost",
                accum="direct",
                unroll=False,
            ),
            TileSpec("k", 4, placement="child_of:i_in", accum="stack"),
            TileSpec("j", 3, kind="panel", accum="direct"),
        ),
        tag="tile-ijk-geometry",
        parallel_loop="i",
    )

    result = matmul(a.to_sparse_csr(), b, schedule=schedule)
    reference = a @ b

    assert torch.allclose(result, reference, atol=1e-3, rtol=1e-3)
    assert torch.count_nonzero(result[3]).item() == 0


@pytest.mark.parametrize("loop_order", [("i", "k", "j"), ("j", "i", "k")])
def test_explicit_loop_order_preserves_logical_operand_shapes(loop_order):
    torch.manual_seed(12)
    m, j, n = 3, 4, 5
    a = torch.randn(m, j, dtype=torch.float32)
    a = a * (torch.rand(m, j) < 0.5)
    b = torch.randn(j, n, dtype=torch.float32)
    schedule = Schedule(loop_order=loop_order, tag="".join(loop_order) + "-shape")

    result = matmul(a.to_sparse_csr(), b, schedule=schedule)
    cached_result = matmul(a.to_sparse_csr(), b, schedule=schedule)

    assert result.shape == (m, n)
    assert torch.allclose(result, a @ b, atol=1e-3, rtol=1e-3)
    assert cached_result.shape == (m, n)
    assert torch.allclose(cached_result, a @ b, atol=1e-3, rtol=1e-3)


def test_scheduled_matmul_honors_output_format_alias():
    torch.manual_seed(13)
    a = torch.randn(5, 7, dtype=torch.float32)
    a = a * (torch.rand(5, 7) < 0.4)
    b = torch.randn(7, 4, dtype=torch.float32)

    result = matmul(
        a.to_sparse_csr(),
        b,
        output_format="ds",
        schedule=Schedule(tag="output-format-alias"),
    )

    assert isinstance(result, STensor)
    assert str(result.format) == "d,s"
    assert torch.allclose(result.to_torch(), a @ b, atol=1e-3, rtol=1e-3)
