"""Converting a torch sparse tensor shares its index arrays, as torch itself does.

``torch.sparse_csr_tensor(crow, col, values)`` keeps the caller's own ``crow`` and
``col``, and a later write to them shows up in the tensor. ``STensor.from_torch``,
``from_csr`` and ``from_coo`` used to copy them, which cost two O(nnz) copies per
conversion and -- because the copies were new arrays every time -- defeated the
verdicts and narrowed copies the native call boundary keeps per array, so a matmul on a
torch CSR built afresh every call also re-walked and re-narrowed its indices.

Sharing is safe because nothing in Scorch writes into an index array it was given, and
because the native call boundary re-checks index arrays per call, keyed on identity and
version counter: these tests pin both halves, and that the verdict memo never vouches
for an array written since its verdict was reached.
"""

import pytest
import torch

import scorch
import scorch_ops
from scorch import STensor
from scorch.exceptions import TensorIndexError


def csr(rows=6, cols=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    dense = torch.rand((rows, cols), generator=generator)
    dense = dense * (dense < 0.5)
    return dense.to_sparse_csr(), dense


def index_arrays(tensor):
    return [array for arrays in tensor._storage._mode_indices for array in arrays]


@pytest.mark.parametrize("convert", [STensor.from_torch, STensor.from_csr])
def test_csr_conversion_shares_the_tensors_index_arrays(convert):
    a, _ = csr()
    s = convert(a)
    crow, col = index_arrays(s)
    assert crow.data_ptr() == a.crow_indices().data_ptr()
    assert col.data_ptr() == a.col_indices().data_ptr()


@pytest.mark.parametrize(
    "convert", [STensor.from_torch, lambda t: STensor.from_coo(t)]
)
def test_coalesced_coo_conversion_shares_the_tensors_indices(convert):
    _, dense = csr()
    a = dense.to_sparse_coo().coalesce()
    s = convert(a)
    rows, cols = index_arrays(s)
    assert rows.data_ptr() == a.indices()[0].data_ptr()
    assert cols.data_ptr() == a.indices()[1].data_ptr()


def test_a_write_through_torch_is_seen_like_torch_sees_it():
    a, dense = csr()
    s = STensor.from_torch(a)
    b = torch.rand(5, 3)
    before = scorch.matmul(s, b)
    torch.testing.assert_close(before, dense @ b, atol=1e-5, rtol=1e-5)
    # Move row 0's first nonzero to another column, keeping the row sorted: a valid
    # structure, so both torch and scorch must now compute the new product.
    col = a.col_indices()
    first, end = int(a.crow_indices()[0]), int(a.crow_indices()[1])
    assert end - first >= 1, "precondition: row 0 has a nonzero"
    upper = int(col[first + 1]) if end - first > 1 else 5
    if int(col[first]) + 1 < upper:
        col[first] += 1
    elif int(col[first]) > 0:
        col[first] -= 1
    else:
        pytest.skip("row 0 has no free column to move to")
    after = scorch.matmul(s, b)
    torch.testing.assert_close(after, a.to_dense() @ b, atol=1e-5, rtol=1e-5)
    assert not torch.equal(after, before)


@pytest.mark.parametrize("through", ["stensor", "torch_operand"])
def test_an_out_of_range_write_after_conversion_fails_closed(through):
    a, _ = csr()
    s = STensor.from_torch(a)
    b = torch.rand(5, 3)
    operand = s if through == "stensor" else a
    for _ in range(3):  # a plan for the torch operand, and every verdict, in place
        scorch.matmul(operand, b)
    a.col_indices()[0] = 1000
    with pytest.raises((RuntimeError, TensorIndexError)):
        scorch.matmul(operand, b)


def test_the_verdict_memo_never_vouches_for_a_written_array():
    a, _ = csr()
    STensor.from_torch(a)  # leaves a clean verdict for these arrays
    a.col_indices()[0] = 1000
    with pytest.raises(TensorIndexError):
        STensor.from_torch(a)


def test_converting_the_same_arrays_again_reaches_the_same_verdict():
    a, _ = csr()
    scorch_ops.abi_memo_clear()
    STensor.from_torch(a)
    held = scorch_ops.abi_memo_size()
    assert held >= 1, "precondition: the first conversion left a verdict"
    for _ in range(3):
        STensor.from_torch(torch.sparse_csr_tensor(
            a.crow_indices(), a.col_indices(), a.values(), size=a.shape
        ))
    assert scorch_ops.abi_memo_size() == held, "the same arrays were stored again"


def test_coo_bounds_and_order_verdicts_do_not_evict_each_other():
    """A COO operand's level-0 bounds verdict and its order verdict are both about
    level 0's coordinates. Keyed on the address alone, each store replaced the other,
    so both scans ran on every call; keyed on (address, check), both are kept."""
    _, dense = csr(12, 10)
    a = dense.to_sparse_coo().coalesce()
    scorch_ops.abi_memo_clear()
    STensor.from_torch(a)
    assert scorch_ops.abi_memo_size() == 3, "expected two bounds verdicts and one order"
