"""CSR operands past the int32 offset bound run as row panels (native_abi.h, abi_csr_row_panels).

A CSR operand with more than INT32_MAX nonzeros used to be refused by the native boundary while
torch.sparse.mm computed it; six SuiteSparse matrices are that large. It now runs as row panels
whose own offsets fit. Building such a matrix takes ~26 GB, so these tests lower the bound
(scorch_ops.abi_csr_panel_limit) to drive the panelled path on small matrices, and check each
answer against a float64 dense reference and against the same product run whole.
"""

import pytest
import torch

import scorch
import scorch.tiling as tiling
import scorch_ops
from scorch import STensor


@pytest.fixture
def panel_limit():
    """Sets the bound for one test and restores it after."""
    original = scorch_ops.abi_csr_panel_limit()

    def set_limit(limit):
        scorch_ops.abi_csr_panel_limit(limit)

    yield set_limit
    scorch_ops.abi_csr_panel_limit(original)
    assert scorch_ops.abi_csr_panel_limit() == original


def _csr(rows, cols, seed, index_dtype=torch.int64, dtype=torch.float32, long_row=0):
    """Rows of varied degree, some empty; `long_row` > 0 gives row 1 that many nonzeros."""
    g = torch.Generator().manual_seed(seed)
    degree = torch.randint(0, 12, (rows,), generator=g)
    degree[torch.randint(0, rows, (rows // 8,), generator=g)] = 0
    if long_row:
        degree[1] = long_row
    crow = torch.zeros(rows + 1, dtype=torch.int64)
    crow[1:] = torch.cumsum(degree, 0)
    col = torch.cat(
        [torch.randperm(cols, generator=g)[: int(d)].sort().values for d in degree]
    )
    values = torch.rand(int(crow[-1]), generator=g, dtype=dtype) + 0.5
    return torch.sparse_csr_tensor(
        crow.to(index_dtype), col.to(index_dtype), values, size=(rows, cols)
    )


def _reference(a, b):
    return (a.to_dense().double() @ b.double()).to(b.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("limit", [37, 500, "nnz - 1"])
def test_a_panelled_product_is_the_whole_product(
    panel_limit, dtype, index_dtype, limit
):
    a = _csr(
        400,
        300,
        seed=7 if limit == "nnz - 1" else limit,
        index_dtype=index_dtype,
        dtype=dtype,
    )
    if limit == "nnz - 1":  # two panels, the second holding the last row's nonzeros
        limit = a._nnz() - 1
    assert (
        a._nnz() > limit
    ), "the operand must exceed the bound for this to test anything"
    for k in (1, 2, 3, 4, 8, 17, 64):
        b = torch.rand(300, k, dtype=dtype)
        whole = scorch.matmul(STensor.from_csr(a, "A"), b)
        panel_limit(limit)
        before = scorch_ops.abi_csr_panelled_calls()
        panelled = scorch.matmul(STensor.from_csr(a, "A"), b)
        assert scorch_ops.abi_csr_panelled_calls() == before + 1, "the call ran whole"
        panel_limit(2**31 - 1)
        torch.testing.assert_close(panelled, _reference(a, b), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(panelled, whole, rtol=1e-5, atol=1e-5)


def test_a_torch_operand_past_the_bound_is_computed(panel_limit):
    a = _csr(300, 250, seed=3)
    b = torch.rand(250, 8)
    panel_limit(50)
    before = scorch_ops.abi_csr_panelled_calls()
    got = scorch.matmul(a, b)
    assert scorch_ops.abi_csr_panelled_calls() > before
    torch.testing.assert_close(got, _reference(a, b), rtol=1e-4, atol=1e-4)


def test_a_row_longer_than_the_bound_is_a_panel_of_its_own(panel_limit):
    a = _csr(200, 400, seed=5, long_row=300)
    b = torch.rand(400, 4)
    panel_limit(64)
    before = scorch_ops.abi_csr_panelled_calls()
    got = scorch.matmul(STensor.from_csr(a, "A"), b)
    assert scorch_ops.abi_csr_panelled_calls() == before + 1
    torch.testing.assert_close(got, _reference(a, b), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("operand", ["torch", "stensor"])
def test_an_operand_past_the_bound_is_never_planned(panel_limit, operand):
    """A plan runs the kernel on the whole operand, so one that must run as row panels gets
    none, and every call reaches the boundary and is panelled. (A matrix really past INT32_MAX
    could not be planned anyway: its positions do not narrow. The lowered bound has to be
    declined explicitly, or these calls would be served whole by a plan.)"""
    from scorch import plan as plan_mod

    a = _csr(300, 250, seed=17)
    b = torch.rand(250, 8)
    x = a if operand == "torch" else STensor.from_csr(a, "A")
    enabled = plan_mod.enabled()
    plan_mod.set_enabled(True)
    try:
        panel_limit(50)
        before = scorch_ops.abi_csr_panelled_calls()
        for _ in range(5):
            got = scorch.matmul(x, b)
        assert scorch_ops.abi_csr_panelled_calls() == before + 5
        assert plan_mod.plans_of(x) == {}
        # The second sighting asked for a plan and was refused: the decline ran, not a path
        # that never plans.
        assert len(plan_mod.refused_of(x)) == 1
    finally:
        plan_mod.set_enabled(enabled)
    torch.testing.assert_close(got, _reference(a, b), rtol=1e-4, atol=1e-4)


def _native_args(positions, coordinates, rows=3, cols=4, k=2, values=None, b=None):
    values = torch.ones(coordinates.numel()) if values is None else values
    b = torch.ones(cols * k) if b is None else b.reshape(-1)
    return (
        [rows, k],
        [rows, cols],
        [[], [positions, coordinates]],
        values,
        [cols, k],
        [[], []],
        b,
    )


def _native_product(crow, col, values, shape, b):
    """The drop-in kernel called directly on the caller's arrays: no plan, so every call is an
    ordinary one."""
    rows, cols = shape
    args = _native_args(crow, col, rows, cols, b.shape[1], values, b)
    return scorch_ops.spmm_csr_float_v2(*args).storage.value.reshape(rows, b.shape[1])


def _arrays(a):
    """The operand's index arrays as arrays of the caller's own (copies, each with its own
    version counter), the way a program that keeps its arrays holds them."""
    return a.crow_indices().clone(), a.col_indices().clone(), a.values().clone()


def test_repeated_calls_find_their_panels_validated(panel_limit):
    """A panel's rebased positions are made once, so the validator's verdict and the narrowed
    coordinates are found again rather than redone on every call."""
    a = _csr(500, 300, seed=7)
    crow, col, values = _arrays(a)
    b = torch.rand(300, 8)
    panel_limit(100)
    before = scorch_ops.abi_csr_panelled_calls()
    first = _native_product(crow, col, values, a.shape, b)
    held = scorch_ops.abi_memo_size()
    for _ in range(3):
        again = _native_product(crow, col, values, a.shape, b)
        torch.testing.assert_close(again, first, rtol=0, atol=0)
    assert scorch_ops.abi_memo_size() == held
    assert scorch_ops.abi_csr_panelled_calls() == before + 4
    torch.testing.assert_close(first, _reference(a, b), rtol=1e-4, atol=1e-4)


def test_a_write_to_the_positions_is_seen_by_the_next_call(panel_limit):
    a = _csr(300, 200, seed=11)
    crow, col, values = _arrays(a)
    b = torch.rand(200, 4)
    panel_limit(60)
    _native_product(crow, col, values, a.shape, b)
    crow[1] = 0  # row 0's nonzeros move to row 1: the same arrays, a different matrix
    moved = torch.sparse_csr_tensor(crow, col, values, size=a.shape)
    torch.testing.assert_close(
        _native_product(crow, col, values, a.shape, b),
        _reference(moved, b),
        rtol=1e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "positions, message",
    [
        ([1, 2, 4, 5], "row panels: positions\\[0\\] must be 0"),
        ([0, 3, 2, 5], "row panels: positions must be nondecreasing"),
        ([0, 2, 4, 4], "row panels: terminal position"),
    ],
)
def test_malformed_positions_past_the_bound_fail_closed(
    panel_limit, positions, message
):
    coordinates = torch.tensor([0, 1, 2, 3, 0], dtype=torch.int64)
    panel_limit(1)
    pos = torch.tensor(positions, dtype=torch.int64)
    with pytest.raises(RuntimeError, match=message):
        scorch_ops.spmm_csr_float_v2(*_native_args(pos, coordinates))


def test_the_bound_is_int32_and_cannot_be_raised_past_it(panel_limit):
    assert scorch_ops.abi_csr_panel_limit() == 2**31 - 1
    panel_limit(2**40)
    assert scorch_ops.abi_csr_panel_limit() == 2**31 - 1


def test_at_the_int32_bound_nothing_an_int32_kernel_takes_is_split():
    a = _csr(400, 300, seed=13)
    before = scorch_ops.abi_csr_panelled_calls()
    scorch.matmul(STensor.from_csr(a, "A"), torch.rand(300, 8))
    scorch.matmul(a, torch.rand(300, 8))
    assert scorch_ops.abi_csr_panelled_calls() == before


class _Huge:
    """A stand-in index array reporting n elements, for the selector's test of size."""

    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n


class _HugeOperand:
    shape = (1 << 20, 1 << 20)

    def __init__(self, nnz):
        self.nnz = nnz

    def _native_mode_indices(self):
        return [[], [_Huge(self.shape[0] + 1), _Huge(self.nnz)]]


def test_the_tiling_selector_leaves_an_operand_past_int32_to_v2(monkeypatch):
    """The tiled kernels take int32 offsets whole and have no panelled form."""
    if not tiling._HAS_TILEJ:
        pytest.skip("no tiled kernels in this build")

    class Reached(Exception):
        pass

    def reached():
        raise Reached

    monkeypatch.setattr(tiling, "query_llc", reached)
    b = torch.empty(1, 64).expand(1 << 20, 64)
    args = (b, (1 << 20, 64), None, None)
    assert tiling.maybe_dispatch(_HugeOperand(2**31), *args, level="balanced") is None
    # One nonzero fewer and the selector goes on to weigh it: the bound is what declined.
    with pytest.raises(Reached):
        tiling.maybe_dispatch(_HugeOperand(2**31 - 1), *args, level="balanced")
