"""float64 CSR x dense on the register-resident kernel, across its own boundaries.

float64 used to resolve `spmm_csr_double`, which is the original reference kernel:
it accumulates into C with a read-modify-write per nonzero, memsets the whole output
serially before the parallel region, and hands rows out on a fixed
`schedule(dynamic, 16)`. It now resolves `spmm_csr_double_v2`, which is the SAME
kernel float32 uses (`spmm_csr_v2_core`) instantiated at double.

The boundaries are NOT the float32 ones, which is the whole reason this file exists
separately. A YMM register holds 4 doubles rather than 8, so with the same register
budget the narrow-k register-blocked path covers k <= 16 instead of k <= 32, the
wide-k tile is 32 elements instead of 64, and the ragged-tail mask has 4 lanes
instead of 8. Every one of those was a fresh chance to write a lane count down wrong,
and none of it is exercised by the float32 suite -- nor by *any* test on Apple
silicon, where the AVX2 block is not compiled at all.

Tolerance here is 1e-12 relative, not the house 1e-3: at double precision the only
thing separating this from the dense reference is summation order, so a loose
tolerance would pass a kernel that had lost whole lanes.
"""

import pytest
import torch

import scorch
from scorch.prebuilt_kernels import resolve_prebuilt_matmul
from scorch.stensor import STensor

pytest.importorskip("scorch_ops")

REL = 1e-12
ROWS, COLS, DEGREE = 71, 53, 9


def operands(k, seed=11, rows=ROWS, cols=COLS):
    generator = torch.Generator().manual_seed(seed)
    dense = torch.zeros(rows, cols, dtype=torch.float64)
    for row in range(rows):
        if row % 13 == 0:
            continue  # an empty row: the kernel skips it and the zero-fill owns it
        columns = torch.randperm(cols, generator=generator)[:DEGREE]
        dense[row, columns] = torch.randn(
            DEGREE, generator=generator, dtype=torch.float64
        )
    b = torch.randn(cols, k, generator=generator, dtype=torch.float64)
    return dense, b


def relative_error(got, want):
    scale = want.abs().max().item()
    return (got - want).abs().max().item() / max(scale, 1e-300)


def test_float64_resolves_the_register_resident_kernel():
    """If this fails, every timing claim about float64 is about a different kernel."""
    dense, b = operands(32)
    resolved = resolve_prebuilt_matmul(
        STensor.from_torch(dense.to_sparse_csr()),
        STensor.from_torch(b),
        output_format="dd",
    )
    assert resolved.symbol_name == "spmm_csr_double_v2"


# 1..40 crosses all four narrow-k instantiations (ceil(k/4) = 1..4 for k <= 16), the
# narrow/wide switch at 16/17, a whole 32-wide tile, and the first ragged tile above
# it. The wider widths carry multi-tile rows.
@pytest.mark.parametrize("k", list(range(1, 41)) + [47, 48, 63, 64, 65, 96, 127, 128])
def test_matches_a_dense_reference(k):
    dense, b = operands(k)
    got = scorch.matmul(
        STensor.from_torch(dense.to_sparse_csr()), STensor.from_torch(b)
    )
    assert got.dtype == torch.float64
    assert torch.isfinite(got).all()
    assert relative_error(got, dense @ b) < REL


@pytest.mark.parametrize("k", [1, 4, 5, 16, 17, 32, 33, 64])
def test_agrees_with_the_reference_kernel_it_replaced(k):
    """The two kernels sum in different orders, so this is a tolerance, not equality --
    but it pins that the replacement computes the same product."""
    scorch_ops = pytest.importorskip("scorch_ops")
    dense, b = operands(k)
    csr = dense.to_sparse_csr()
    pos = csr.crow_indices().to(torch.int32)
    crd = csr.col_indices().to(torch.int32)
    shapes = (
        [ROWS, k],
        [ROWS, COLS],
        [[], [pos, crd]],
        csr.values(),
        [COLS, k],
        [[], []],
        b.reshape(-1),
    )
    reference = scorch_ops.spmm_csr_double(*shapes).storage.value.reshape(ROWS, k)
    new = scorch_ops.spmm_csr_double_v2(*shapes).storage.value.reshape(ROWS, k)
    assert relative_error(new, reference) < REL


def test_an_all_empty_matrix_yields_zeros():
    """Nothing writes any row, so the output is entirely the zero-fill's work."""
    empty = torch.zeros(ROWS, COLS, dtype=torch.float64)
    b = torch.randn(COLS, 24, dtype=torch.float64)
    got = scorch.matmul(
        STensor.from_torch(empty.to_sparse_csr()), STensor.from_torch(b)
    )
    assert torch.equal(got, torch.zeros(ROWS, 24, dtype=torch.float64))


def test_more_output_rows_than_the_matrix_has_is_rejected():
    """The kernel zero-fills output rows past the last sparse row (``C0_size >
    A0_size``). That branch is NOT reachable from Python: the ABI validator requires
    ``result_shape == [A.rows, B.cols]``, so it is defensive cover for a future C++
    caller, and the only behaviour testable from here is the rejection.

    Stated because an earlier version of this test claimed to exercise the branch and
    silently did not -- it passed a result shape equal to the matrix's and then
    compared against a dense reference, which is what the parametrized test above
    already does.
    """
    scorch_ops = pytest.importorskip("scorch_ops")
    dense, b = operands(20)
    csr = dense.to_sparse_csr()
    shapes = [
        [ROWS + 4, 20],  # four rows more than the matrix has
        [ROWS, COLS],
        [[], [csr.crow_indices().to(torch.int32), csr.col_indices().to(torch.int32)]],
        csr.values(),
        [COLS, 20],
        [[], []],
        b.reshape(-1),
    ]
    with pytest.raises(RuntimeError, match=r"result_shape must equal"):
        scorch_ops.spmm_csr_double_v2(*shapes)


def test_empty_rows_inside_the_matrix_are_zeroed():
    """The zero-fill that IS reachable: a row with no nonzeros is never written by the
    kernel, so its output comes entirely from the per-empty-row memset. ``operands``
    leaves every 13th row empty; check those rows are exactly zero rather than merely
    close, since anything the kernel wrote there would be nonzero garbage.
    """
    dense, b = operands(20)
    got = scorch.matmul(
        STensor.from_torch(dense.to_sparse_csr()), STensor.from_torch(b)
    )
    empty = [row for row in range(ROWS) if row % 13 == 0]
    assert empty, "the fixture stopped producing empty rows"
    assert torch.equal(got[empty], torch.zeros(len(empty), 20, dtype=torch.float64))
    assert relative_error(got, dense @ b) < REL


def test_a_row_wider_than_one_tile_is_not_truncated():
    """A 32-wide tile means k=96 is three whole tiles; a lane-count error that dropped
    the last tile would leave zeros a dense comparison catches but a checksum might
    not, so compare column blocks explicitly."""
    dense, b = operands(96)
    got = scorch.matmul(
        STensor.from_torch(dense.to_sparse_csr()), STensor.from_torch(b)
    )
    want = dense @ b
    for start in (0, 32, 64):
        block = slice(start, start + 32)
        assert relative_error(got[:, block], want[:, block]) < REL
