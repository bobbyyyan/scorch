"""The CSR x CSR product split into a symbolic phase and a numeric phase.

``spgemm_csr_symbolic`` computes C = A @ B's structure once; ``spgemm_csr_numeric`` then
computes C's values for that structure with no count, linked list or sort -- the candidate a
plan that keeps the structure runs on every call after the first (design note, section 2 item
4). Its claim is that every value is bit-identical to the two-phase kernel's
(``spgemm_csr_torch_result``), because each row is accumulated in the same order, and that the
structure is the two-phase kernel's exactly. Both are tested here over every dtype, both result
index types, and the operands the two-phase kernel is tested on: rows out of order, repeated
coordinates, torch.sparse.mm's own output, empty rows and empty extents.
"""

import warnings

import numpy as np
import pytest
import torch

import scorch_ops

warnings.filterwarnings("ignore", message=".*Sparse.*")

pytestmark = pytest.mark.skipif(
    not hasattr(scorch_ops, "spgemm_csr_numeric"),
    reason="this build has no numeric-only CSR x CSR kernel",
)

DTYPES = [torch.float32, torch.float64, torch.int32, torch.int64]


def random_csr(rows, cols, density, dtype, index=torch.int64, seed=0, min_per_row=0):
    rng = np.random.default_rng(seed)
    mask = rng.random((rows, cols)) < density
    if min_per_row and cols:
        for i in range(rows):
            mask[i, rng.choice(cols, min(min_per_row, cols), replace=False)] = True
    if dtype.is_floating_point:
        values = rng.uniform(-1.0, 1.0, (rows, cols))
    else:
        values = rng.integers(1, 4, (rows, cols)) * rng.choice([-1, 1], (rows, cols))
    a = torch.from_numpy(np.where(mask, values, 0)).to(dtype).to_sparse_csr()
    return torch.sparse_csr_tensor(
        a.crow_indices().to(index), a.col_indices().to(index), a.values(), size=(rows, cols)
    )


def reversed_rows(a):
    crow, col, values = a.crow_indices(), a.col_indices().clone(), a.values().clone()
    for i in range(a.shape[0]):
        lo, hi = int(crow[i]), int(crow[i + 1])
        col[lo:hi] = col[lo:hi].flip(0)
        values[lo:hi] = values[lo:hi].flip(0)
    return torch.sparse_csr_tensor(crow.clone(), col, values, size=a.shape)


def repeated_coordinates(a):
    crow, col, values = a.crow_indices(), a.col_indices(), a.values()
    out_crow, out_col, out_val = [0], [], []
    for i in range(a.shape[0]):
        lo, hi = int(crow[i]), int(crow[i + 1])
        c, v = col[lo:hi].tolist(), values[lo:hi].tolist()
        if len(c) >= 2:
            c, v = (c + [c[0]])[::-1], (v + [v[0]])[::-1]
        out_col += c
        out_val += v
        out_crow.append(len(out_col))
    return torch.sparse_csr_tensor(
        torch.tensor(out_crow, dtype=crow.dtype),
        torch.tensor(out_col, dtype=crow.dtype),
        torch.tensor(out_val, dtype=a.dtype),
        size=a.shape,
    )


def arrays(x):
    return x.crow_indices(), x.col_indices(), x.values()


def two_phase(a, b, int64_indices):
    return scorch_ops.spgemm_csr_torch_result(
        *arrays(a), *arrays(b), a.shape[0], a.shape[1], b.shape[1], int64_indices
    )


def split(a, b, int64_indices):
    positions, coordinates = scorch_ops.spgemm_csr_symbolic(
        *arrays(a), *arrays(b), a.shape[0], a.shape[1], b.shape[1], int64_indices
    )
    values = scorch_ops.spgemm_csr_numeric(
        *arrays(a), *arrays(b), positions, coordinates, a.shape[0], a.shape[1], b.shape[1]
    )
    return positions, coordinates, values


def same(a, b, int64_indices):
    want = two_phase(a, b, int64_indices)
    positions, coordinates, values = split(a, b, int64_indices)
    assert positions.dtype is want.crow_indices().dtype
    assert torch.equal(positions, want.crow_indices())
    assert torch.equal(coordinates, want.col_indices())
    assert values.dtype is want.values().dtype
    assert torch.equal(values, want.values()), "values are not bit-identical"
    return positions, coordinates


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("int64_indices", [False, True])
@pytest.mark.parametrize("operand", ["sorted", "reversed", "repeated", "int32_input"])
def test_split_is_the_two_phase_product_exactly(dtype, int64_indices, operand):
    index = torch.int32 if operand == "int32_input" else torch.int64
    a = random_csr(70, 50, 0.08, dtype, index, seed=1, min_per_row=1)
    b = random_csr(50, 90, 0.08, dtype, index, seed=2, min_per_row=1)
    if operand == "reversed":
        a, b = reversed_rows(a), reversed_rows(b)
    elif operand == "repeated":
        a, b = repeated_coordinates(a), repeated_coordinates(b)
    same(a, b, int64_indices)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_sparse_mm_output(dtype):
    s = random_csr(80, 80, 0.06, dtype, seed=3, min_per_row=2)
    try:
        t = torch.sparse.mm(s, s)
    except RuntimeError as error:
        # A torch built without MKL (the M5's) multiplies CSR x CSR only in COO.
        if "without MKL" not in str(error):
            raise
        pytest.skip("this torch has no CSR x CSR product: " + str(error)[:80])
    same(t, s, True)
    same(s, t, True)
    same(t, t, False)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "rows, inner, cols, density",
    [(1, 1, 1, 1.0), (5, 0, 4, 0.0), (0, 6, 3, 0.5), (9, 7, 0, 0.5), (16, 16, 16, 0.0)],
)
def test_empty_extents(dtype, rows, inner, cols, density):
    same(
        random_csr(rows, inner, density, dtype, seed=4),
        random_csr(inner, cols, density, dtype, seed=5),
        True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_one_structure_many_values(dtype):
    """The structure computed once serves every later set of values over the same arrays."""
    a = random_csr(120, 100, 0.05, dtype, seed=6, min_per_row=1)
    b = random_csr(100, 110, 0.05, dtype, seed=7, min_per_row=1)
    positions, coordinates = same(a, b, True)
    for step in range(4):
        a.values().copy_(torch.randn_like(a.values()))
        b.values().copy_(torch.randn_like(b.values()))
        values = scorch_ops.spgemm_csr_numeric(
            *arrays(a), *arrays(b), positions, coordinates, 120, 100, 110
        )
        assert torch.equal(values, two_phase(a, b, True).values()), step


def test_one_thread_and_many_agree():
    a = random_csr(300, 300, 0.03, torch.float64, seed=8, min_per_row=2)
    b = random_csr(300, 300, 0.03, torch.float64, seed=9, min_per_row=2)
    _, _, many = split(a, b, True)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, _, one = split(a, b, True)
    finally:
        torch.set_num_threads(previous)
    assert torch.equal(one, many)


def test_a_structure_that_does_not_fit_is_refused():
    a = random_csr(10, 8, 0.3, torch.float64, seed=10, min_per_row=1)
    b = random_csr(8, 6, 0.3, torch.float64, seed=11, min_per_row=1)
    positions, coordinates = scorch_ops.spgemm_csr_symbolic(
        *arrays(a), *arrays(b), 10, 8, 6, True
    )
    out_of_range = coordinates.clone()
    out_of_range[0] = 6
    for pos, crd in (
        (positions[:-1], coordinates),  # too few positions
        (positions, out_of_range),  # a coordinate outside [0, cols)
        (positions.to(torch.int32), coordinates),  # two index dtypes
    ):
        with pytest.raises(Exception):
            scorch_ops.spgemm_csr_numeric(*arrays(a), *arrays(b), pos, crd, 10, 8, 6)
