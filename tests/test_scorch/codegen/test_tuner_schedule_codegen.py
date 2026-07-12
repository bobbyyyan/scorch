import torch
import pytest

from scorch import STensor
from scorch.compiler.cin import ForAll, IndexVar, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.scheduler import (
    RelayoutSpec,
    Schedule,
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


def _packed_tileijk_schedule(nc: int = 4, jc: int = 3) -> Schedule:
    return Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                nc,
                placement="outermost",
                accum="direct",
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
        relayout=RelayoutSpec("B", "k", nc),
        tag="packed-tile-ijk",
        parallel_loop="i",
    )


def _build_spmm_cin():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")
    c[i, k] = a[i, j] * b[j, k]
    return ForAll(i, ForAll(j, ForAll(k, c._assignment)))


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
    ("m", "j", "n", "empty_row"),
    [
        (7, 10, 11, 2),
        (5, 13, 6, 4),
        (9, 2, 17, 0),
    ],
    ids=("ragged-all", "rectangular", "wide-multi-k-panel"),
)
def test_tuner_packed_tileijk_matches_torch_with_ragged_panels(m, j, n, empty_row):
    torch.manual_seed(m * 100 + j * 10 + n)
    a = torch.randn(m, j, dtype=torch.float32)
    a = a * (torch.rand(m, j) < 0.4)
    a[empty_row] = 0
    b = torch.randn(j, n, dtype=torch.float32)

    result = matmul(
        a.to_sparse_csr(),
        b,
        schedule=_packed_tileijk_schedule(),
    )
    reference = a @ b

    assert result.shape == (m, n)
    assert torch.allclose(result, reference, atol=1e-4, rtol=1e-4)
    assert torch.count_nonzero(result[empty_row]).item() == 0


@pytest.mark.parametrize(("j", "n"), [(0, 7), (5, 0)])
def test_tuner_packed_tileijk_handles_zero_sized_domains(j, n):
    m = 4
    a = torch.zeros((m, j), dtype=torch.float32)
    b = torch.randn((j, n), dtype=torch.float32)

    result = matmul(
        a.to_sparse_csr(),
        b,
        schedule=_packed_tileijk_schedule(),
    )

    assert result.shape == (m, n)
    assert torch.equal(result, a @ b)


def test_tuner_packed_tileijk_uses_generated_float64_storage():
    torch.manual_seed(21)
    a = torch.randn(4, 7, dtype=torch.float64)
    a = a * (torch.rand(4, 7) < 0.4)
    b = torch.randn(7, 5, dtype=torch.float64)
    stmt = _build_spmm_cin()
    for access in stmt.tensor_accesses:
        access.tensor.dtype = torch.float64

    scheduled = Scheduler.apply_schedule(stmt, _packed_tileijk_schedule())
    cpp = LLIRLowerer().lower_llir(CINLowerer().lower_IndexStmt(scheduled))
    result = matmul(a.to_sparse_csr(), b, schedule=_packed_tileijk_schedule())

    assert "std::vector<double> packed_B_storage" in cpp
    assert "double* __restrict__ packed_B" in cpp
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
