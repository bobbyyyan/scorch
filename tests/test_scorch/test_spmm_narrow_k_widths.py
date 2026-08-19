"""The drop-in CSR SpMM across every narrow free-dimension width, ragged and not.

The narrow-k register kernel holds a whole output row in ceil(k/8) YMM registers,
so k splits its behaviour three ways: whether the final vector is entirely valid
(k % 8 == 0, no mask needed), whether it is ragged (a genuine partial vector, mask
required), and whether k is narrow enough to take that path at all. The kernel is
specialized on ceil(k/8) for 1..4, so k = 8, 16, 24, 32 are also the boundaries
between instantiations.

Dropping the mask where k is a multiple of 8 is semantics-preserving by
construction -- with all eight lanes enabled, a masked load is a load and a masked
store is a store -- but "provably equivalent" is an argument about the code, and
the widths where the argument is made are exactly the widths nothing was checking.
So every width from 1 to 40 is compared against a dense reference here, which
covers both sides of all four instantiation boundaries and the crossing into the
wide (register-tiled) path above 32.

These run on any build; the masked-versus-unmasked timing comparison needs the
instrumented build and lives in bench/bench_spmm_narrowk_mask.py.
"""

import pytest
import torch

from scorch import ops
from scorch.prebuilt_kernels import (
    execute_prebuilt_binary_kernel,
    resolve_prebuilt_matmul,
)
from scorch.stensor import STensor

pytest.importorskip("scorch_ops")

ATOL = RTOL = 1e-3
ROWS, COLS, DEGREE = 97, 83, 11


def csr_operand(seed=5):
    """A small CSR matrix with a couple of empty rows, which the row loop skips."""
    generator = torch.Generator().manual_seed(seed)
    dense = torch.zeros(ROWS, COLS)
    for row in range(ROWS):
        if row % 17 == 0:
            continue  # an empty row: pA_begin == pA_end
        columns = torch.randperm(COLS, generator=generator)[:DEGREE]
        dense[row, columns] = torch.randn(DEGREE, generator=generator)
    return dense


def v2(a_st, b_dense, k):
    """The drop-in SpMM through the same entry `ops.matmul` uses.

    Going through execute_prebuilt_binary_kernel rather than naming the kernel's
    positional arguments here is deliberate: that argument list has a `tile_size`
    parameter ahead of the thread count, and spelling it out by hand is how a
    caller ends up passing the thread count as a tile width.
    """
    b_st = STensor.from_torch(b_dense)
    resolved = resolve_prebuilt_matmul(a_st, b_st, output_format="dd")
    nthreads, atparallel = ops._composition_hints(resolved)
    out, _shape = execute_prebuilt_binary_kernel(
        resolved.fn, a_st, b_st, nthreads=nthreads, atparallel=atparallel
    )
    return out.storage.value.view(ROWS, k)


@pytest.mark.parametrize("k", list(range(1, 41)))
def test_every_narrow_width_matches_a_dense_reference(k):
    dense = csr_operand()
    a_st = STensor.from_torch(dense.to_sparse_csr())
    generator = torch.Generator().manual_seed(100 + k)
    b = torch.randn(COLS, k, generator=generator)
    expected = dense @ b
    actual = v2(a_st, b, k)
    assert torch.allclose(
        actual, expected, atol=ATOL, rtol=RTOL
    ), f"k={k}: max diff {(actual - expected).abs().max().item()}"


@pytest.mark.parametrize("k", [8, 16, 24, 32])
def test_a_full_final_vector_writes_no_further_than_the_row(k):
    """The unmasked store must not write past the row it owns.

    A masked store on the final vector also bounded the write. Where the mask is
    dropped because all eight lanes are valid the bound comes from k itself, so a
    width that is a multiple of 8 is where an off-by-one would show up as one row
    corrupting the next -- and with a contiguous row-major output, silently.
    """
    dense = csr_operand()
    a_st = STensor.from_torch(dense.to_sparse_csr())
    generator = torch.Generator().manual_seed(7)
    b = torch.randn(COLS, k, generator=generator)
    expected = dense @ b
    actual = v2(a_st, b, k)
    # Every row, including the empty ones, which must come back exactly zero
    # rather than holding whatever a neighbouring row's store spilled.
    for row in range(ROWS):
        assert torch.allclose(
            actual[row], expected[row], atol=ATOL, rtol=RTOL
        ), f"k={k} row={row} differs; empty={row % 17 == 0}"


def test_the_widths_either_side_of_every_instantiation_boundary_agree():
    """ceil(k/8) selects the instantiation; k=8/9, 16/17, 24/25, 32/33 straddle them.

    A width where the mask was dropped sits next to one where it was kept, so if
    the two disagree about anything other than the number of columns, the pair
    catches it.
    """
    dense = csr_operand()
    a_st = STensor.from_torch(dense.to_sparse_csr())
    for narrow, wide in ((8, 9), (16, 17), (24, 25), (32, 33)):
        generator = torch.Generator().manual_seed(narrow)
        b_wide = torch.randn(COLS, wide, generator=generator)
        expected = dense @ b_wide
        # The first `narrow` columns of the wider product must equal the product
        # computed at width `narrow` from the same columns of B.
        got_wide = v2(a_st, b_wide, wide)
        got_narrow = v2(a_st, b_wide[:, :narrow].contiguous(), narrow)
        assert torch.allclose(got_wide[:, :narrow], got_narrow, atol=ATOL, rtol=RTOL)
        assert torch.allclose(got_wide, expected, atol=ATOL, rtol=RTOL)
