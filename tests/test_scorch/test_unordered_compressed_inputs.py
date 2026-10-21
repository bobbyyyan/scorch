"""A caller's sparse storage whose coordinates are out of order or repeated inside a parent.

torch accepts a CSR tensor whose columns are out of order, or repeated, within a row: its
invariant checks are off by default, and every operation on the tensor sums the repeats.
Scorch accepts the same storage. An STensor over it holds a private copy in canonical form
(each parent's coordinates sorted, a coordinate stored once with the sum of its values;
``storage._canonical_index``), and the autograd path keeps its CSR structure in the same
form with a map back to the caller's entries. Everything else wrong with an index still
fails closed, and a kernel's own result is never repaired: there the same defect means
the compiler emitted something wrong.

These tests pin: the canonical form against a dense reference written independently of
it, over every arrangement of dense and compressed levels up to rank 3; each public route
against torch on the same operand (float64 reference, atol=rtol=1e-3); gradients; that
the caller's arrays are never written; and that well-formed input still shares the
caller's arrays, the path whose cost is unchanged.
"""

import itertools

import pytest
import torch

import scorch
from scorch import STensor, TensorLayout
from scorch.exceptions import TensorIndexError, TensorStorageError
from scorch.format import LevelType
from scorch.storage import SparseStorage, _validate_index_storage

ATOL = 1e-3
RTOL = 1e-3

D, C = LevelType.DENSE, LevelType.COMPRESSED
_LETTER = {"d": D, "s": C}


# -- an independent reference ---------------------------------------------------------


def stored_entries(level_types, extents, mode_indices, value_count):
    """Every stored value's full coordinate, in storage order, by walking the levels the
    way their definitions read (no sorting, no assumption that coordinates are distinct)."""
    frontier = [((), 0)]  # (coordinate prefix, position at the level above)
    for level_type, arrays, extent in zip(level_types, mode_indices, extents):
        below = []
        for prefix, parent in frontier:
            if level_type == D:
                for c in range(extent):
                    below.append((prefix + (c,), parent * extent + c))
            else:
                positions, coordinates = arrays
                for p in range(int(positions[parent]), int(positions[parent + 1])):
                    below.append((prefix + (int(coordinates[p]),), p))
        frontier = below
    assert len(frontier) == value_count
    # Positions at the leaf are value positions; report in value order.
    by_position = sorted(frontier, key=lambda item: item[1])
    return [prefix for prefix, _ in by_position]


def dense_reference(level_types, extents, mode_indices, values):
    out = torch.zeros(extents, dtype=torch.float64)
    entries = stored_entries(level_types, extents, mode_indices, values.numel())
    for coordinate, value in zip(entries, values.tolist()):
        out[coordinate] += value
    return out


def assert_canonical(level_types, extents, mode_indices, value_count):
    """Each parent's coordinates strictly increasing; hence no coordinate stored twice."""
    entries = stored_entries(level_types, extents, mode_indices, value_count)
    assert entries == sorted(set(entries)), "canonical form must be sorted and distinct"


# -- storage with every kind of ordering defect ---------------------------------------


def unordered_storage(fmt, extents, index_dtype, value_dtype, seed):
    """Levels in ``fmt`` over ``extents`` whose compressed parents hold coordinates out of
    order and repeated, built top-down; values random and nonzero."""
    g = torch.Generator().manual_seed(seed)
    level_types = [_LETTER[ch] for ch in fmt]
    mode_indices = []
    parent_count = 1
    for level_type, extent in zip(level_types, extents):
        if level_type == D:
            mode_indices.append([])
            parent_count *= extent
            continue
        positions = [0]
        coordinates = []
        for _ in range(parent_count):
            n = int(torch.randint(0, 2 * extent + 1, (1,), generator=g))
            row = torch.randint(0, extent, (n,), generator=g).tolist()
            coordinates.extend(row)
            positions.append(len(coordinates))
        mode_indices.append(
            [
                torch.tensor(positions, dtype=index_dtype),
                torch.tensor(coordinates, dtype=index_dtype),
            ]
        )
        parent_count = len(coordinates)
    if value_dtype.is_floating_point:
        values = (torch.rand(parent_count, generator=g, dtype=torch.float64) + 0.5).to(
            value_dtype
        )
    else:
        values = torch.randint(1, 5, (parent_count,), generator=g).to(value_dtype)
    return level_types, mode_indices, values


FORMATS = [
    "".join(levels)
    for rank in (1, 2, 3)
    for levels in itertools.product("ds", repeat=rank)
    if "s" in levels
]
EXTENTS = {1: (7,), 2: (5, 6), 3: (3, 4, 5)}


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64, torch.int32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("fmt", FORMATS)
def test_a_caller_storage_is_held_in_canonical_form(fmt, index_dtype, value_dtype, seed):
    extents = EXTENTS[len(fmt)]
    level_types, mode_indices, values = unordered_storage(
        fmt, extents, index_dtype, value_dtype, seed
    )
    expected = dense_reference(level_types, extents, mode_indices, values)
    given = [[t.clone() for t in arrays] for arrays in mode_indices]
    given_values = values.clone()

    layout = TensorLayout.from_physical_shape(extents, fmt, index_dtype=index_dtype)
    storage = SparseStorage(layout, values, mode_indices=mode_indices)

    held = [list(arrays) for arrays in storage._mode_indices]
    held_values = storage._value
    assert held_values.dtype == value_dtype
    assert all(t.dtype == index_dtype for arrays in held for t in arrays)
    assert_canonical(level_types, extents, held, held_values.numel())
    # Summed in the values' own dtype, so float32 sums differ from the float64 reference
    # in the last bits.
    assert torch.allclose(
        dense_reference(level_types, extents, held, held_values),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )
    # Well-formed by the strict check, which every later validation runs.
    _validate_index_storage(layout, storage._mode_indices, held_values)
    # The caller's arrays are untouched and shared with nothing.
    for mine, theirs in zip(mode_indices, given):
        assert all(torch.equal(a, b) for a, b in zip(mine, theirs))
    assert torch.equal(values, given_values)
    caller_buffers = {t.data_ptr() for arrays in mode_indices for t in arrays if t.numel()}
    assert not caller_buffers & {t.data_ptr() for arrays in held for t in arrays}


def test_an_explicitly_stored_zero_is_kept_as_torch_keeps_it():
    layout = TensorLayout.from_physical_shape((1, 4), "ds", index_dtype=torch.int64)
    storage = SparseStorage(
        layout,
        torch.tensor([0.0, 2.0, 0.0]),
        mode_indices=[
            [],
            [torch.tensor([0, 3]), torch.tensor([3, 1, 3])],
        ],
    )
    assert storage._mode_indices[1][1].tolist() == [1, 3]
    assert storage._value.tolist() == [2.0, 0.0]


# -- what still fails closed -----------------------------------------------------------


@pytest.mark.parametrize(
    ("positions", "coordinates", "error"),
    [
        ([0, 2, 3], [1, 6, 0], TensorIndexError),  # column past the end, in a row that
        ([0, 2, 3], [-1, 0, 2], TensorIndexError),  # is also unordered; and a negative one
        ([1, 2, 3], [2, 1, 0], TensorIndexError),  # positions not starting at zero
        ([0, 3, 2], [2, 1, 0], TensorIndexError),  # positions decreasing
        ([0, 2, 4], [2, 1, 0], TensorIndexError),  # terminal past the coordinates
    ],
)
def test_every_other_defect_still_raises(positions, coordinates, error):
    layout = TensorLayout.from_physical_shape((2, 6), "ds", index_dtype=torch.int64)
    with pytest.raises(error):
        SparseStorage(
            layout,
            torch.ones(len(coordinates)),
            mode_indices=[[], [torch.tensor(positions), torch.tensor(coordinates)]],
        )


def test_a_value_count_mismatch_still_raises():
    layout = TensorLayout.from_physical_shape((2, 6), "ds", index_dtype=torch.int64)
    with pytest.raises(TensorStorageError):
        SparseStorage(
            layout,
            torch.ones(2),
            mode_indices=[[], [torch.tensor([0, 2, 3]), torch.tensor([3, 1, 0])]],
        )


# -- the public routes, against torch on the same operand ------------------------------


def unordered_csr(
    rows, cols, per_row, dtype, seed, index_dtype=torch.int64, repeats=True
):
    """A torch CSR tensor with columns out of order inside its rows, and (``repeats``)
    repeated inside some of them; without ``repeats`` each row's columns are distinct."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, min(per_row, cols) + 1, (rows,), generator=g)
    crow = torch.zeros(rows + 1, dtype=torch.int64)
    crow[1:] = torch.cumsum(counts, 0)
    nnz = int(crow[-1])
    if repeats:
        col = torch.randint(0, cols, (nnz,), generator=g)
    else:
        col = torch.cat(
            [torch.randperm(cols, generator=g)[: int(n)] for n in counts]
            or [torch.zeros(0, dtype=torch.int64)]
        )
    # Make sure the defects are there: repeat a column and reverse a row.
    for row in range(min(rows, 8)):
        start, stop = int(crow[row]), int(crow[row + 1])
        if repeats and stop - start >= 2:
            col[start + 1] = col[start]
        col[start:stop] = col[start:stop].flip(0)
    values = torch.rand(nnz, generator=g, dtype=torch.float64).to(dtype)
    return torch.sparse_csr_tensor(
        crow.to(index_dtype), col.to(index_dtype), values, size=(rows, cols)
    )


def reference_dense(csr):
    crow, col = csr.crow_indices().tolist(), csr.col_indices().tolist()
    out = torch.zeros(csr.shape, dtype=torch.float64)
    for row in range(csr.shape[0]):
        for p in range(crow[row], crow[row + 1]):
            out[row, col[p]] += float(csr.values()[p])
    return out


SHAPES = [(40, 30, 6), (300, 257, 12)]


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(("rows", "cols", "per_row"), SHAPES)
@pytest.mark.parametrize("k", [1, 16, 64])
def test_spmm_on_an_unordered_torch_csr_equals_torch(
    rows, cols, per_row, k, dtype, index_dtype
):
    a = unordered_csr(rows, cols, per_row, dtype, seed=rows + k, index_dtype=index_dtype)
    b = torch.rand(cols, k, generator=torch.Generator().manual_seed(k), dtype=dtype)
    expected = reference_dense(a) @ b.double()
    snapshot = (a.crow_indices().clone(), a.col_indices().clone(), a.values().clone())
    # Repeated calls: whatever a later call reuses from an earlier one must stay right.
    for _ in range(3):
        out = scorch.matmul(a, b)
        assert out.dtype == dtype
        assert torch.allclose(out.double(), expected, atol=ATOL, rtol=RTOL)
    assert torch.allclose(
        torch.sparse.mm(a, b).double(), expected, atol=ATOL, rtol=RTOL
    ), "the reference itself must be torch's answer"
    assert torch.equal(a.crow_indices(), snapshot[0])
    assert torch.equal(a.col_indices(), snapshot[1])
    assert torch.equal(a.values(), snapshot[2])


def test_a_write_to_the_callers_values_reaches_the_next_product():
    a = unordered_csr(64, 48, 8, torch.float32, seed=11)
    b = torch.rand(48, 8, generator=torch.Generator().manual_seed(3))
    for scale in (1.0, 2.0, -0.5):
        if scale != 1.0:
            a.values().mul_(scale)
        out = scorch.matmul(a, b)
        assert torch.allclose(
            out.double(), reference_dense(a) @ b.double(), atol=ATOL, rtol=RTOL
        )


@pytest.mark.parametrize(("left", "right"), [("ds", "ds"), ("ds", "ss"), ("ss", "ds")])
def test_a_sparse_sparse_product_with_an_unordered_operand_is_right(left, right):
    """Co-iterating two sparse operands is what repeated coordinates used to break."""
    a_csr = unordered_csr(30, 24, 6, torch.float64, seed=5)
    b_csr = unordered_csr(24, 20, 5, torch.float64, seed=6)
    a = STensor.from_csr(a_csr, "A")
    b = STensor.from_csr(b_csr, "B")
    if left != "ds":
        a = scorch.from_torch(a.to_torch(), "A").to_sparse(left)
    if right != "ds":
        b = scorch.from_torch(b.to_torch(), "B").to_sparse(right)
    out = scorch.matmul(a, b)
    dense = out.to_torch() if isinstance(out, STensor) else out
    expected = reference_dense(a_csr) @ reference_dense(b_csr)
    assert torch.allclose(dense.double(), expected, atol=ATOL, rtol=RTOL)


def test_the_stensor_holds_the_sum_and_is_usable_downstream():
    a_csr = unordered_csr(50, 40, 7, torch.float32, seed=9)
    a = STensor.from_csr(a_csr, "A")
    assert torch.allclose(
        a.to_torch().double(), reference_dense(a_csr), atol=ATOL, rtol=RTOL
    )
    x = torch.rand(40, 4, generator=torch.Generator().manual_seed(1))
    y = torch.rand(4, 3, generator=torch.Generator().manual_seed(2))
    out = scorch.matmul(scorch.matmul(a, x), y)
    assert torch.allclose(
        out.double(),
        reference_dense(a_csr) @ x.double() @ y.double(),
        atol=ATOL,
        rtol=RTOL,
    )


def test_an_unordered_operand_is_held_as_a_copy():
    """Documented: the STensor holds the canonical copy, so a later write to the caller's
    values is not seen by it -- as with an uncoalesced COO tensor, which from_torch
    coalesces into a new one. (A torch operand passed to matmul is wrapped afresh on every
    call, so its products do see the write: see the test above.)"""
    a_csr = unordered_csr(20, 16, 5, torch.float64, seed=4)
    a = STensor.from_csr(a_csr, "A")
    before = a.to_torch().clone()
    a_csr.values().mul_(3.0)
    assert torch.equal(a.to_torch(), before)


def test_well_formed_csr_still_shares_the_callers_arrays():
    """The unchanged path: no copy, so the native boundary's per-array verdicts and the
    caller's writes both carry over, as before."""
    dense = (torch.rand(30, 20, generator=torch.Generator().manual_seed(0)) < 0.2) * 1.0
    csr = dense.to_sparse_csr()
    a = STensor.from_csr(csr, "A")
    positions, coordinates = a._storage._mode_indices[1]
    assert coordinates.data_ptr() == csr.col_indices().data_ptr()
    assert positions.data_ptr() == csr.crow_indices().data_ptr()
    assert a._storage._value.data_ptr() == csr.values().data_ptr()


# -- gradients -------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradients_through_an_out_of_order_csr_equal_torch(dtype):
    """Rows out of order but distinct: the gradient reaches each of the caller's stored
    values, in the caller's order, as torch.sparse.mm gives it. scorch.matmul gives the
    operand torch.matmul's dense gradient, and the operand's construction samples it at
    the stored coordinates, so the values get torch.sparse.mm's gradient."""
    a_ref = unordered_csr(36, 28, 6, dtype, seed=21, repeats=False)
    b_ref = torch.rand(28, 5, generator=torch.Generator().manual_seed(7), dtype=dtype)
    g = torch.rand(36, 5, generator=torch.Generator().manual_seed(8), dtype=dtype)

    def run(matmul):
        values = a_ref.values().clone().requires_grad_(True)
        a = torch.sparse_csr_tensor(
            a_ref.crow_indices(), a_ref.col_indices(), values, size=a_ref.shape
        )
        b = b_ref.clone().requires_grad_(True)
        out = matmul(a, b)
        (out * g).sum().backward()
        return out.detach(), values.grad, b.grad

    out_s, gv_s, gb_s = run(scorch.matmul)
    out_t, gv_t, gb_t = run(torch.sparse.mm)
    assert torch.allclose(out_s.double(), out_t.double(), atol=ATOL, rtol=RTOL)
    assert torch.allclose(gb_s.double(), gb_t.double(), atol=ATOL, rtol=RTOL)
    # One gradient per stored value of the caller's operand, in the caller's order.
    assert gv_s.shape == a_ref.values().shape
    assert torch.allclose(gv_s.double(), gv_t.double(), atol=ATOL, rtol=RTOL)


def test_a_repeated_column_gives_torchs_own_values_gradient():
    """A CSR tensor with a repeated column, built from values that require grad: the
    values' gradient comes from torch's backward through the tensor's construction, given
    the operand's gradient. scorch.matmul gives the operand torch.matmul's (dense), so
    whatever torch's construction backward makes of it -- a values gradient, or its own
    error -- is the same for both, value for value or message for message. (torch 2.13's
    backward rejects the gradient torch.sparse.mm gives such an operand, one value per
    distinct coordinate: SparseCompressedTensorBackward0.)"""
    a_ref = unordered_csr(12, 10, 4, torch.float64, seed=23)
    b = torch.rand(10, 3, dtype=torch.float64)

    def run(matmul):
        values = a_ref.values().clone().requires_grad_(True)
        a = torch.sparse_csr_tensor(
            a_ref.crow_indices(), a_ref.col_indices(), values, size=a_ref.shape
        )
        try:
            matmul(a, b).sum().backward()
        except RuntimeError as error:
            return "raises", str(error).splitlines()[0]
        return "gradient", values.grad

    got, want = run(scorch.matmul), run(torch.matmul)
    assert got[0] == want[0]
    if got[0] == "raises":
        assert got[1] == want[1]
    else:
        torch.testing.assert_close(got[1], want[1], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_dense_operands_gradient_through_a_repeated_column_is_right(dtype):
    a = unordered_csr(36, 28, 6, dtype, seed=22)
    dense = reference_dense(a)
    g = torch.Generator().manual_seed(9)
    b = torch.rand(28, 5, generator=g, dtype=dtype).requires_grad_(True)
    upstream = torch.rand(36, 5, generator=g, dtype=dtype)
    (scorch.matmul(a, b) * upstream).sum().backward()
    assert torch.allclose(
        b.grad.double(), dense.t() @ upstream.double(), atol=ATOL, rtol=RTOL
    )
    x = torch.rand(7, 36, generator=g, dtype=dtype).requires_grad_(True)
    upstream = torch.rand(7, 28, generator=g, dtype=dtype)
    (scorch.matmul(x, a) * upstream).sum().backward()
    assert torch.allclose(
        x.grad.double(), upstream.double() @ dense.t(), atol=ATOL, rtol=RTOL
    )


def test_dense_times_an_unordered_csr_equals_torch():
    """``x @ A`` runs on A's transpose, built from the canonical structure."""
    a = unordered_csr(32, 26, 6, torch.float32, seed=31)
    x = torch.rand(9, 32, generator=torch.Generator().manual_seed(2))
    expected = x.double() @ reference_dense(a)
    for _ in range(3):
        out = scorch.matmul(x, a)
        assert torch.allclose(out.double(), expected, atol=ATOL, rtol=RTOL)


def test_a_repaired_structure_is_kept_across_calls():
    """The structure is keyed on the caller's arrays, not on the copy it holds, so a second
    ``x @ A`` over the same operand finds it instead of rebuilding it."""
    from scorch.plan import GRAD_ATTR, torch_object_state

    a = unordered_csr(32, 26, 6, torch.float32, seed=32)
    x = torch.rand(5, 32, generator=torch.Generator().manual_seed(4))
    scorch.matmul(x, a)
    held = torch_object_state(a, create=True)[GRAD_ATTR]
    assert held.destination is not None
    scorch.matmul(x, a)
    assert torch_object_state(a, create=True)[GRAD_ATTR] is held
