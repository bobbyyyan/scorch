import torch

from scorch import STensor, einsum
from scorch.compiler.cin import (
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
    Workspace,
)
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.scheduler import Scheduler


def _lower_to_cpp(cin_stmt) -> str:
    lowered = CINLowerer().lower_IndexStmt(cin_stmt)
    return LLIRLowerer().lower_llir(lowered)


def test_spmm_codegen_emits_parallel_restrict_no_workspace():
    """SpMM with sparse input should NOT tile or use workspace, since tiling
    would force re-traversal of the sparse structure once per tile."""
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")

    c[i, j] = a[i, k] * b[k, j]
    cin_stmt = Scheduler.auto_schedule(ForAll(i, ForAll(k, ForAll(j, c._assignment))))
    cpp_code = _lower_to_cpp(cin_stmt)

    assert "#pragma omp parallel for" in cpp_code
    assert "__restrict__" in cpp_code
    # No workspace or tiling: accumulate directly into output
    assert "wksp" not in cpp_code
    assert "kTile_" not in cpp_code


def test_non_tiled_dense_workspace_is_zero_initialized_and_raii_owned():
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")

    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    wksp = Workspace("wksp", dim=1, dense=True)

    cin_stmt = ForAll(
        i,
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        wksp[j],
                        a[i, k] * b[k, j],
                        op=Operation.ADD,
                    ),
                ),
            ),
            consumer=ForAll(
                j,
                TensorAssign(
                    c[i, j],
                    wksp[j],
                ),
            ),
        ),
    )

    cpp_code = _lower_to_cpp(cin_stmt)
    assert "scorch_make_aligned_buffer<float>" in cpp_code
    assert "wksp_pool_owner.get()" in cpp_code
    assert cpp_code.index("wksp_pool_owner") < cpp_code.index("#pragma omp parallel")
    assert "omp_get_thread_num()" in cpp_code
    assert "memset(wksp, 0," in cpp_code
    assert "free(wksp);" not in cpp_code


def test_dynamic_sparse_output_uses_checked_raii_vector_transfer():
    i = IndexVar("i")
    result = TensorVar("Result", fmt="s")
    left = TensorVar("Left", fmt="s")
    right = TensorVar("Right", fmt="s")
    result[i] = left[i] + right[i]

    cpp_code = _lower_to_cpp(ForAll(i, result._assignment))

    assert "std::vector<int> Result0_pos" in cpp_code
    assert "std::vector<float> Result_values" in cpp_code
    assert "Result_values.emplace_back" in cpp_code
    assert "Result0_crd.emplace_back" in cpp_code
    assert "scorch_vector_set(Result0_pos" in cpp_code
    assert "scorch_tensor_from_vector(std::move(Result_values)" in cpp_code
    assert "Result.storage.index.mode_indices" in cpp_code
    assert "Result.storage.value" in cpp_code
    assert "cvector" not in cpp_code
    assert "from_blob" not in cpp_code
    assert "malloc" not in cpp_code


def test_dense_output_is_torch_owned_from_allocation():
    i = IndexVar("i")
    result = TensorVar("Result", fmt="d")
    left = TensorVar("Left", fmt="s")
    right = TensorVar("Right", fmt="s")
    result[i] = left[i] + right[i]

    cpp_code = _lower_to_cpp(ForAll(i, result._assignment))

    assert (
        "torch::Tensor Result_values_torch = "
        "torch::empty({Result_capacity}, torch::kFloat32);"
    ) in cpp_code
    assert "Result_values_torch.data_ptr<float>()" in cpp_code
    assert "scorch_zero_dense(Result_values" in cpp_code
    assert "from_blob" not in cpp_code
    assert "malloc" not in cpp_code


def test_dynamic_float64_sparse_output_uses_standard_vector():
    i = IndexVar("i")
    result = TensorVar("Result", fmt="s", dtype=torch.float64)
    left = TensorVar("Left", fmt="s", dtype=torch.float64)
    right = TensorVar("Right", fmt="s", dtype=torch.float64)
    result[i] = left[i] * right[i]

    cpp_code = _lower_to_cpp(ForAll(i, result._assignment))

    assert "std::vector<double> Result_values" in cpp_code
    assert "scorch_tensor_from_vector(std::move(Result_values)" in cpp_code


def test_dense_parent_positions_use_exact_size_raii_vector():
    row = IndexVar("row")
    column = IndexVar("column")
    result = TensorVar("Result", fmt="ds")
    source = TensorVar("Source", fmt="oo")
    result[row, column] = source[row, column]

    cpp_code = _lower_to_cpp(ForAll(row, ForAll(column, result._assignment)))

    assert "std::vector<int> Result1_pos((size_t) Result0_size + 1, 0)" in cpp_code
    assert "torch::zeros({Result0_size + 1}" not in cpp_code
    assert "scorch_tensor_from_vector(std::move(Result1_pos)" in cpp_code

    dense_result = TensorVar("DenseResult", shape=(1024, 1024), fmt="ds")
    dense_source = TensorVar("DenseSource", shape=(1024, 1024), fmt="dd")
    dense_result[row, column] = dense_source[row, column]
    dense_cpp = _lower_to_cpp(ForAll(row, ForAll(column, dense_result._assignment)))

    assert (
        "std::vector<int> DenseResult1_pos((size_t) DenseResult0_size + 1, 0)"
        in dense_cpp
    )
    assert "scorch_vector_set(DenseResult1_pos" not in dense_cpp


def test_rank2_dense_conversion_reserves_bounded_leaf_capacity():
    row = IndexVar("row")
    column = IndexVar("column")
    result = TensorVar("Result", fmt="oo")
    source = TensorVar("Source", fmt="dd")
    result[row, column] = source[row, column]

    cpp_code = _lower_to_cpp(ForAll(row, ForAll(column, result._assignment)))

    assert "int64_t _dynamic_reserve = std::min<int64_t>" in cpp_code
    assert 'result_shape, "evaluate", "result_shape", true), 2048)' in cpp_code
    assert "Result0_crd.reserve(_dynamic_reserve)" in cpp_code
    assert "Result1_crd.reserve(_dynamic_reserve)" in cpp_code
    assert "Result_values.reserve(_dynamic_reserve)" in cpp_code


def test_parallel_dss_positions_give_each_boundary_one_writer():
    batch = IndexVar("batch")
    row = IndexVar("row")
    column = IndexVar("column")
    reduction = IndexVar("reduction")

    result = TensorVar("Result", fmt="dss")
    left = TensorVar("Left", fmt="dss")
    right = TensorVar("Right", fmt="dss")
    result[batch, row, column] = (
        left[batch, row, reduction] * right[batch, reduction, column]
    )

    cin_stmt = Scheduler.auto_schedule(
        ForAll(
            batch,
            ForAll(
                row,
                ForAll(
                    reduction,
                    ForAll(column, result._assignment),
                ),
            ),
        )
    )
    cpp_code = _lower_to_cpp(cin_stmt)

    assert cpp_code.count("#pragma omp parallel for") == 2
    assert cpp_code.count("Result2_pos_data[0] = 0;") == 1
    assert "Result2_pos_data[_base1]" not in cpp_code
    assert "Result2_pos_data[_base1 + _pos1] = _base2 + _pos2;" in cpp_code

    first_parallel = cpp_code.index("#pragma omp parallel for")
    fill_parallel = cpp_code.index("#pragma omp parallel for", first_parallel + 1)
    assert cpp_code.index("Result2_pos_data[0] = 0;") < fill_parallel


def test_parallel_dss_runtime_positions_are_complete_and_monotonic():
    torch.manual_seed(20260712)
    batch, rows, reduction, columns = 96, 12, 12, 12
    left = torch.randn(batch, rows, reduction)
    left *= torch.rand(batch, rows, reduction) < 0.35
    right = torch.randn(batch, reduction, columns)
    right *= torch.rand(batch, reduction, columns) < 0.35

    # Empty outer slices exercise repeated offsets between non-empty slices.
    left[::7] = 0
    right[::11] = 0
    left_sparse = STensor.from_torch(left, "left").to_sparse("dss")
    right_sparse = STensor.from_torch(right, "right").to_sparse("dss")
    expected = torch.bmm(left, right)

    for _ in range(3):
        result = einsum("bij,bjk->bik", left_sparse, right_sparse, format="dss")

        for level in (1, 2):
            positions, coordinates = result.storage.index.mode_indices[level]
            assert positions[0].item() == 0
            assert torch.all(positions[1:] >= positions[:-1])
            assert positions[-1].item() == coordinates.numel()
        assert (
            result.storage.index.mode_indices[2][0][-1].item()
            == result.storage.value.numel()
        )
        assert torch.allclose(result.to_torch(), expected, rtol=1e-5, atol=1e-5)
