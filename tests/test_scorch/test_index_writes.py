"""An in-place write to a torch sparse operand's index arrays must reach every later product.

torch keeps the index arrays a sparse tensor was built from as its members -- the caller's own
tensors -- and gives the accessors crow_indices(), col_indices() and indices() aliases that share
the SPARSE TENSOR's version counter instead of the arrays'. So a write through the caller's array
moves one counter and a write through an accessor moves another, and a cache keyed on either
tensor alone misses the other kind. Every cache scorch keeps over index arrays -- the validation
memo's int32 copies, plans, autograd's structures and transposes, x @ A's records -- was keyed on
the accessors' aliases, and products after `col.copy_(...)` came back computed from the old
columns.

Each test here warms a product until whatever caches it uses exist, writes new column indices in
place by one route, and compares the next product with torch's own on the matrix as it is now.
"""

import gc

import pytest
import torch

import scorch
import scorch_ops
from scorch import plan as plan_mod

NEW_COLS = [2, 1, 0]
WARM_CALLS = 3


def _arrays(index_dtype):
    crow = torch.tensor([0, 1, 2, 3], dtype=index_dtype)
    col = torch.tensor([0, 1, 2], dtype=index_dtype)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    return crow, col, values


def _csr(crow, col, values):
    return torch.sparse_csr_tensor(crow, col, values, size=(3, 3))


def _rhs(requires_grad=False):
    b = torch.arange(12, dtype=torch.float64).reshape(3, 4).div(7.0)
    return b.requires_grad_(requires_grad)


def _lhs(requires_grad=False):
    x = torch.arange(12, dtype=torch.float64).reshape(4, 3).div(5.0)
    return x.requires_grad_(requires_grad)


def _check_product(product, a, grad):
    """scorch's product with ``a`` as it is now against torch's on its dense form, and with
    ``grad``, the dense operand's gradient too."""
    dense = a.detach().to_dense()
    if product == "a@b":
        b = _rhs(grad)
        out = scorch.matmul(a, b)
        torch.testing.assert_close(out.detach(), dense @ b.detach(), rtol=0, atol=1e-12)
        if grad:
            out.sum().backward()
            torch.testing.assert_close(
                b.grad,
                dense.t() @ torch.ones(3, 4, dtype=torch.float64),
                rtol=0,
                atol=1e-12,
            )
    else:
        x = _lhs(grad)
        out = scorch.matmul(x, a)
        torch.testing.assert_close(out.detach(), x.detach() @ dense, rtol=0, atol=1e-12)
        if grad:
            out.sum().backward()
            torch.testing.assert_close(
                x.grad,
                torch.ones(4, 3, dtype=torch.float64) @ dense.t(),
                rtol=0,
                atol=1e-12,
            )


def _warm(product, a, grad):
    for _ in range(WARM_CALLS):
        _check_product(product, a, grad)


ROUTES = (
    # a write through the operand's own accessor: moves only the operand's counter
    "operand_accessor",
    # a write through the caller's array: moves only the array's counter
    "caller_array",
    # a write through a view of the caller's array: the same counter as the array's
    "caller_array_view",
    # a new operand over arrays written through the caller's array after another used them
    "fresh_operand_after_caller_write",
    # a new operand over arrays written through ANOTHER operand's accessor, one scorch has read
    "fresh_operand_after_sibling_accessor",
    # the warmed operand itself, after a write through a sibling's accessor that scorch read
    # after the warm-up: the plan's copy of the witness list must take the sibling in
    "sibling_accessor_after_warmup",
)


@pytest.mark.parametrize("plans", [True, False], ids=["plans", "no_plans"])
@pytest.mark.parametrize("grad", [False, True], ids=["nograd", "grad"])
@pytest.mark.parametrize("product", ["a@b", "x@a"])
@pytest.mark.parametrize(
    "index_dtype", [torch.int64, torch.int32], ids=["int64", "int32"]
)
@pytest.mark.parametrize("route", ROUTES)
def test_csr_index_write_reaches_the_next_product(
    route, index_dtype, product, grad, plans
):
    was = plan_mod.enabled()
    plan_mod.set_enabled(plans)
    try:
        crow, col, values = _arrays(index_dtype)
        a = _csr(crow, col, values)
        _warm(product, a, grad)
        new = torch.tensor(NEW_COLS, dtype=index_dtype)
        if route == "operand_accessor":
            a.col_indices().copy_(new)
            now = a
        elif route == "caller_array":
            col.copy_(new)
            now = a
        elif route == "caller_array_view":
            col[:].copy_(new)
            now = a
        elif route == "fresh_operand_after_caller_write":
            col.copy_(new)
            now = _csr(crow, col, values)
        elif route == "fresh_operand_after_sibling_accessor":
            a.col_indices().copy_(new)
            now = _csr(crow, col, values)
        else:
            sibling = _csr(crow, col, values)
            scorch.matmul(sibling, _rhs())
            sibling.col_indices().copy_(new)
            now = a
        assert now.col_indices().tolist() == NEW_COLS
        _check_product(product, now, grad)
    finally:
        plan_mod.set_enabled(was)


@pytest.mark.parametrize(
    "index_dtype", [torch.int64, torch.int32], ids=["int64", "int32"]
)
def test_stensor_from_torch_sees_a_write_through_the_caller_array(index_dtype):
    crow, col, values = _arrays(index_dtype)
    a = _csr(crow, col, values)
    s = scorch.from_torch(a)
    b = _rhs()
    for _ in range(WARM_CALLS):
        scorch.matmul(s, b)
    col.copy_(torch.tensor(NEW_COLS, dtype=index_dtype))
    torch.testing.assert_close(
        scorch.matmul(s, b), a.to_dense() @ b, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("route", ["operand_accessor", "caller_array"])
def test_stensor_from_coo_sees_a_write_to_a_coalesced_operand(route):
    # from_coo adopts a coalesced operand's indices rather than copying them.
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    a = torch.sparse_coo_tensor(indices, values, (3, 3), is_coalesced=True)
    s = scorch.STensor.from_coo(a)
    b = _rhs()
    for _ in range(WARM_CALLS):
        scorch.matmul(s, b)
    new = torch.tensor(NEW_COLS)
    (indices[1] if route == "caller_array" else a.indices()[1]).copy_(new)
    torch.testing.assert_close(
        scorch.matmul(s, b), a.to_dense() @ b, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("route", ["operand_accessor", "caller_array", "fresh_operand"])
def test_coo_index_write_reaches_the_next_product(route):
    # A COO operand built coalesced keeps the caller's indices as its member, as CSR does, and
    # its product runs on a JIT-compiled kernel, whose own validation memo reads the epochs
    # through scorch_ops' capsule.
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    a = torch.sparse_coo_tensor(indices, values, (3, 3), is_coalesced=True)
    b = _rhs()
    for _ in range(WARM_CALLS):
        scorch.matmul(a, b)
    if route == "operand_accessor":
        a.indices()[1].copy_(torch.tensor(NEW_COLS))
        now = a
    elif route == "caller_array":
        indices[1].copy_(torch.tensor(NEW_COLS))
        now = a
    else:
        indices[1].copy_(torch.tensor(NEW_COLS))
        now = torch.sparse_coo_tensor(indices, values, (3, 3), is_coalesced=True)
    assert now.indices()[1].tolist() == NEW_COLS
    # Not bitwise: float64 COO x dense is known to round differently from torch's (~5e-8 here,
    # before this change too); a stale product is off by O(1).
    torch.testing.assert_close(
        scorch.matmul(now, b), now.to_dense() @ b, rtol=1e-6, atol=1e-6
    )


# --------------------------------------------------------------------------- #
# The witnesses themselves (csrc/native_abi.h, Write witnesses)
# --------------------------------------------------------------------------- #


def test_members_are_the_caller_arrays_and_share_their_counter():
    crow, col, values = _arrays(torch.int64)
    a = _csr(crow, col, values)
    m_crow, m_col = scorch_ops.torch_sparse_index_members(a)
    assert m_col.data_ptr() == col.data_ptr() and m_crow.data_ptr() == crow.data_ptr()
    before = m_col._version
    col.add_(0)
    assert m_col._version == before + 1
    # ...and an accessor write moves the operand's counter, not theirs
    a.col_indices().add_(0)
    assert m_col._version == before + 1


def test_an_operand_is_one_witness_however_often_it_is_read():
    crow, col, values = _arrays(torch.int64)
    a = _csr(crow, col, values)
    for _ in range(4):
        scorch_ops.torch_sparse_index_members(a)
    assert scorch_ops.abi_witness_count(col) == 1
    assert scorch_ops.abi_witness_count(crow) == 1


def test_an_accessor_write_moves_the_epoch():
    crow, col, values = _arrays(torch.int64)
    a = _csr(crow, col, values)
    scorch_ops.torch_sparse_index_members(a)
    before = scorch_ops.abi_epoch(col)
    assert scorch_ops.abi_epoch(col) == before  # observing alone moves nothing
    a.col_indices().add_(0)
    assert scorch_ops.abi_epoch(col) == before + 1
    assert scorch_ops.abi_epoch(col) == before + 1  # re-baselined: one write, one step


def test_an_operand_written_before_scorch_saw_it_moves_the_epoch():
    crow, col, values = _arrays(torch.int64)
    a = _csr(crow, col, values)
    scorch_ops.torch_sparse_index_members(_csr(crow, col, values))  # a first witness
    before = scorch_ops.abi_epoch(col)
    a.col_indices().add_(0)  # a has not been read by scorch yet
    scorch_ops.torch_sparse_index_members(a)
    assert scorch_ops.abi_epoch(col) == before + 1


def test_a_dead_operand_stops_being_a_witness_once_no_alias_of_it_survives():
    crow, col, values = _arrays(torch.int64)
    a = _csr(crow, col, values)
    scorch_ops.torch_sparse_index_members(a)
    alias = a.col_indices()
    del a
    gc.collect()
    scorch_ops.abi_epoch(col)
    assert scorch_ops.abi_witness_count(col) == 1, "an alias of it can still write"
    before = scorch_ops.abi_epoch(col)
    alias.add_(0)  # the operand is gone; its alias still writes through its counter
    assert scorch_ops.abi_epoch(col) == before + 1
    del alias
    gc.collect()
    # Only registration prunes (native_abi.h, AbiWatchCopy): the next operand read over
    # these arrays drops the dead one and is then the only witness.
    other = _csr(crow, col, values)
    scorch_ops.torch_sparse_index_members(other)
    assert scorch_ops.abi_witness_count(col) == 1


def test_a_new_operand_per_call_does_not_accumulate_witnesses():
    crow, col, values = _arrays(torch.int64)
    b = _rhs()
    for _ in range(50):
        scorch.matmul(_csr(crow, col, values), b)
    gc.collect()
    scorch_ops.abi_epoch(col)
    # The last operand's witness, and the one before it: the arrays' entry keeps the version
    # counter (not the operand) of the last operand recorded through it, so that a persistent
    # operand is recognised without the lock (csrc/torch_structures.h), and a witness is pruned
    # only once nothing else holds its counter.
    assert scorch_ops.abi_witness_count(col) <= 2


def test_more_witnesses_than_a_plan_copies_inline():
    # A plan copies up to AbiWatchCopy::kInline (4) witnesses and checks under the lock past
    # that; a write through the last of six live operands must still reach the product.
    crow, col, values = _arrays(torch.int64)
    a = _csr(crow, col, values)
    b = _rhs()
    for _ in range(WARM_CALLS):
        scorch.matmul(a, b)
    siblings = [_csr(crow, col, values) for _ in range(5)]
    for sibling in siblings:
        scorch.matmul(sibling, b)
    assert scorch_ops.abi_witness_count(col) == 6
    scorch.matmul(a, b)
    siblings[-1].col_indices().copy_(torch.tensor(NEW_COLS))
    torch.testing.assert_close(
        scorch.matmul(a, b), a.to_dense() @ b, rtol=0, atol=1e-12
    )


@pytest.mark.xfail(
    strict=True,
    reason="documented blind spot (native_abi.h, Write witnesses): torch gives the alias of an "
    "operand scorch never read a counter no tensor scorch holds shares",
)
def test_a_write_through_an_operand_scorch_never_read():
    crow, col, values = _arrays(torch.int64)
    b = _rhs()
    seen = _csr(crow, col, values)
    for _ in range(WARM_CALLS):
        scorch.matmul(seen, b)
    unseen = _csr(crow, col, values)
    unseen.col_indices().copy_(torch.tensor(NEW_COLS))
    fresh = _csr(crow, col, values)
    torch.testing.assert_close(
        scorch.matmul(fresh, b), fresh.to_dense() @ b, rtol=0, atol=1e-12
    )
