"""A new int64 CSR pair is checked and narrowed in one pass (native_abi.h,
abi_narrow_csr_first_sight).

The native boundary reads int32 index arrays. An int64 pair it had not seen used to be read four
times: a range pass and a copy per array (checked_index_tensor), then a structural scan of the
copies (checked_csr_view). Now one pass checks the structure and writes the copies, and leaves the
memos as those checks would have left them, so the checks find their answers there. These tests pin
that it is only that: the same products, the same messages for every malformed pair, and the same
memo contents.
"""

import pytest
import torch

import scorch_ops

KERNELS = [("spmm_csr_float_v2", torch.float32), ("spmm_csr_double_v2", torch.float64)]


def parts(rows, cols, density, dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((rows, cols), generator=generator) < density
    dense = (torch.rand((rows, cols), generator=generator) * mask).to(dtype)
    if rows > 2:
        dense[rows // 2] = 0
    a = dense.to_sparse_csr()
    return a.crow_indices(), a.col_indices(), a.values(), dense


def call(kernel, crow, col, values, rows, cols, b):
    k = b.shape[1]
    return getattr(scorch_ops, kernel)(
        [rows, k],
        [rows, cols],
        [[], [crow, col]],
        values,
        [cols, k],
        [[], []],
        b.reshape(-1),
    ).storage.value.reshape(rows, k)


@pytest.mark.parametrize("kernel, dtype", KERNELS)
@pytest.mark.parametrize(
    "rows, cols, density, k",
    [(1, 1, 1.0, 1), (9, 7, 0.4, 3), (64, 80, 0.0, 2), (700, 650, 0.02, 16)],
)
def test_a_new_int64_pair_gives_the_int32_pairs_product(
    kernel, dtype, rows, cols, density, k
):
    crow, col, values, dense = parts(rows, cols, density, dtype)
    b = torch.rand(cols, k, dtype=dtype)
    want = call(kernel, crow.int(), col.int(), values, rows, cols, b)
    got = call(kernel, crow.long().clone(), col.long().clone(), values, rows, cols, b)
    assert torch.equal(got, want)
    torch.testing.assert_close(got, dense @ b)


def test_it_leaves_one_copy_per_array_and_finds_them_again():
    rows, cols = 300, 200
    crow, col, values, _ = parts(rows, cols, 0.05, torch.float32)
    b = torch.rand(cols, 4)
    crow, col = crow.long().clone(), col.long().clone()
    # From empty memos, so that no sweep of earlier tests' dead entries falls due in between.
    scorch_ops.abi_memo_clear()
    entries, held = scorch_ops.abi_narrow_held()
    first = call("spmm_csr_float_v2", crow, col, values, rows, cols, b)
    after_entries, after_held = scorch_ops.abi_narrow_held()
    assert after_entries - entries == 2
    assert after_held - held == 4 * (rows + 1) + 4 * col.numel()
    size = scorch_ops.abi_memo_size()
    again = call("spmm_csr_float_v2", crow, col, values, rows, cols, b)
    assert scorch_ops.abi_memo_size() == size
    assert scorch_ops.abi_narrow_held() == (after_entries, after_held)
    assert torch.equal(first, again)


def test_a_write_to_the_arrays_is_seen_on_the_next_call():
    rows, cols = 50, 40
    crow, col, values, _ = parts(rows, cols, 0.1, torch.float32)
    crow, col = crow.long().clone(), col.long().clone()
    b = torch.rand(cols, 3)
    call("spmm_csr_float_v2", crow, col, values, rows, cols, b)
    starts = crow[:-1].repeat_interleave(crow[1:] - crow[:-1])
    col.sub_(col[starts].clone())
    got = call("spmm_csr_float_v2", crow, col, values, rows, cols, b)
    want = torch.sparse_csr_tensor(crow, col, values, size=(rows, cols)).to_dense() @ b
    torch.testing.assert_close(got, want)


def malformed(case):
    crow = [0, 2, 3, 5]
    col = [0, 2, 1, 0, 3]
    if case == "unsorted":
        col = [2, 0, 1, 0, 3]
    elif case == "column_past_the_end":
        col = [0, 2, 1, 0, 4]
    elif case == "negative_column":
        col = [0, 2, -1, 0, 3]
    elif case == "decreasing_positions":
        crow = [0, 3, 2, 5]
    elif case == "first_position":
        crow = [1, 2, 3, 5]
    elif case == "position_past_nnz":
        crow = [0, 2, 6, 5]
    return crow, col


CASES = [
    "unsorted",
    "column_past_the_end",
    "negative_column",
    "decreasing_positions",
    "first_position",
    "position_past_nnz",
]


@pytest.mark.parametrize("case", CASES)
def test_a_malformed_int64_pair_is_reported_as_the_int32_pair_is(case):
    """Narrowing an in-range array changes no value, so the report is the int32 pair's."""
    crow, col = malformed(case)
    values = torch.rand(5)
    b = torch.rand(4, 2)
    with pytest.raises(RuntimeError) as narrow:
        call(
            "spmm_csr_float_v2",
            torch.tensor(crow, dtype=torch.int32),
            torch.tensor(col, dtype=torch.int32),
            values,
            3,
            4,
            b,
        )
    with pytest.raises(RuntimeError) as wide:
        call(
            "spmm_csr_float_v2",
            torch.tensor(crow, dtype=torch.int64),
            torch.tensor(col, dtype=torch.int64),
            values,
            3,
            4,
            b,
        )
    assert str(wide.value).splitlines()[0] == str(narrow.value).splitlines()[0]


@pytest.mark.parametrize("case", CASES)
def test_a_malformed_pair_is_not_remembered(case):
    crow, col = malformed(case)
    crow, col = torch.tensor(crow), torch.tensor(col)
    scorch_ops.abi_memo_clear()
    entries, _ = scorch_ops.abi_narrow_held()
    size = scorch_ops.abi_memo_size()
    with pytest.raises(RuntimeError):
        call("spmm_csr_float_v2", crow, col, torch.rand(5), 3, 4, torch.rand(4, 2))
    # The old checks keep a copy of an array that narrows cleanly before the structure is
    # found wrong; the one pass keeps nothing for a pair it does not clear. Neither keeps a
    # verdict for it.
    assert scorch_ops.abi_narrow_held()[0] - entries <= 2
    assert scorch_ops.abi_memo_size() - size <= 2


def test_a_value_past_int32_is_reported_by_the_narrowing():
    crow = torch.tensor([0, 1, 2], dtype=torch.int64)
    col = torch.tensor([0, 2**31], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="cannot be represented as int32"):
        call("spmm_csr_float_v2", crow, col, torch.rand(2), 2, 4, torch.rand(4, 2))


@pytest.mark.parametrize("index", [torch.int32, torch.int64])
@pytest.mark.parametrize("extent", [0, -1])
def test_a_limit_that_is_not_positive_leaves_no_coordinate_in_range(index, extent):
    """The screen tests the range with one unsigned comparison, under which a negative limit
    would admit every coordinate; it must answer as the signed test did."""
    crow = torch.tensor([0, 1, 2], dtype=index)
    col = torch.tensor([0, 0], dtype=index)
    assert (
        scorch_ops.abi_screen_compressed_level(crow, col, extent, True, 1 << 16) is True
    )
    assert (
        scorch_ops.abi_screen_compressed_level(crow, col, extent, False, 1 << 16)
        is True
    )


@pytest.fixture
def four_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("defect", [None, "unsorted", "column_past_the_end"])
def test_a_pair_past_the_boundary_grain_is_narrowed_by_split_scans(
    four_threads, defect
):
    """1.2M nonzeros, past the boundary's 1M-element grain, so the pass that checks and narrows
    the pair splits across workers: its positions scan, and its coordinate scan with the
    boundary walk. A defect in the last worker's range is reported as for the int32 pair.
    """
    rows, cols, deg = 150000, 1000, 8
    generator = torch.Generator().manual_seed(0)
    crow = torch.arange(rows + 1, dtype=torch.int64) * deg
    col = torch.randint(0, cols, (rows, deg), generator=generator).sort(dim=1).values
    if defect == "unsorted":
        col[140001] = torch.arange(deg - 1, -1, -1)
    elif defect == "column_past_the_end":
        col[140001, 5] = cols
    col = col.reshape(-1)
    values = torch.rand(rows * deg, generator=generator)
    b = torch.rand(cols, 2, generator=generator)
    scorch_ops.abi_memo_clear()
    if defect is None:
        want = call("spmm_csr_float_v2", crow.int(), col.int(), values, rows, cols, b)
        got = call(
            "spmm_csr_float_v2", crow.clone(), col.clone(), values, rows, cols, b
        )
        assert torch.equal(got, want)
        return
    with pytest.raises(RuntimeError) as narrow:
        call("spmm_csr_float_v2", crow.int(), col.int(), values, rows, cols, b)
    with pytest.raises(RuntimeError) as wide:
        call("spmm_csr_float_v2", crow.clone(), col.clone(), values, rows, cols, b)
    assert str(wide.value).splitlines()[0] == str(narrow.value).splitlines()[0]
