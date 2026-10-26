"""COO operands with int32 indices reach both prebuilt COO kernels, and multiply correctly.

``STensor.from_components(..., index_dtype=torch.int32)`` builds an ``"o,o"`` operand whose
coordinate arrays are int32.  ``matmul``'s prebuilt resolution reads the operands' formats
and value dtype only, so such an operand reaches ``spmm_coo_float`` (COO x dense) and
``spmspm_coo_float`` (COO x COO).  The native boundary (``checked_coo_view``) takes int32
coordinates as they are and narrows int64 ones into int32 copies, so both widths run the
same kernel on the same coordinates.  Each case here checks that:

1. the operand stores int32 coordinates, and resolution names the COO kernel;
2. ``matmul`` runs that kernel on that operand (``execute_prebuilt_binary_kernel``, spied),
   and no generated kernel;
3. the product equals torch's exactly: the values are small integers, so every sum is exact
   in float32 and the comparison is ``torch.equal`` against a float64 product;
4. the int32 operand's product equals the int64 operand's bit for bit (coordinates and
   values of a COO result);
5. the call leaves the operand's int32 arrays in place, unchanged.

The widths include 4112 for ``spmm_coo_float``: it walks B's columns in tiles of 4096 and
then the remainder, so 4112 runs both loops, and 16 runs the remainder alone.  Each kernel
also runs with an operand that stores no entry.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch

import scorch
import scorch_ops
from scorch import ops
from scorch.prebuilt_kernels import resolve_prebuilt_matmul
from scorch.stensor import STensor

ROWS, INNER, COLS = 37, 29, 23


def _valued(rows: int, cols: int, seed: int, empty: bool = False) -> torch.Tensor:
    """Small integers at about a quarter of the positions, with an empty row and column."""
    g = torch.Generator().manual_seed(seed)
    if empty:
        return torch.zeros(rows, cols, dtype=torch.float32)
    keep = torch.rand(rows, cols, generator=g) < 0.25
    dense = (torch.randint(1, 9, (rows, cols), generator=g) * keep).to(torch.float32)
    dense[rows // 2, :] = 0  # an empty row
    dense[:, cols // 3] = 0  # and an empty column
    dense[0, cols - 1] = 3  # the last column is read
    dense[rows - 1, 0] = 5  # and the last row
    return dense


def _coo(dense: torch.Tensor, index: torch.dtype) -> STensor:
    coo = dense.to_sparse_coo().coalesce()
    rows, cols = coo.indices()
    return STensor.from_components(
        tuple(dense.shape),
        "oo",
        [[rows.to(index)], [cols.to(index)]],
        coo.values().clone(),
        index_dtype=index,
    )


def _arrays(t: STensor) -> List[torch.Tensor]:
    return [level[0] for level in t._native_mode_indices()]


@pytest.fixture
def kernels(monkeypatch):
    """The kernel each prebuilt call ran, with its operands, and the generated calls."""
    prebuilt: List[Tuple[object, STensor, STensor]] = []
    generated: List[object] = []
    real_prebuilt = ops.execute_prebuilt_binary_kernel
    real_generated = ops._generated_matmul

    def prebuilt_spy(fn, a, b, *args, **kwargs):
        prebuilt.append((fn, a, b))
        return real_prebuilt(fn, a, b, *args, **kwargs)

    def generated_spy(*args, **kwargs):
        generated.append(args)
        return real_generated(*args, **kwargs)

    monkeypatch.setattr(ops, "execute_prebuilt_binary_kernel", prebuilt_spy)
    monkeypatch.setattr(ops, "_generated_matmul", generated_spy)
    return prebuilt, generated


def _product(kernels, a: STensor, b, symbol: str):
    """``matmul(a, b)``, checked to have run ``symbol`` on ``a`` and nothing generated."""
    prebuilt, generated = kernels
    del prebuilt[:], generated[:]
    resolved = resolve_prebuilt_matmul(a, b if isinstance(b, STensor) else STensor.from_torch(b))
    assert resolved is not None and resolved.symbol_name == symbol
    out = scorch.matmul(a, b)
    assert len(prebuilt) == 1 and not generated, (len(prebuilt), len(generated))
    fn, got_a, got_b = prebuilt[0]
    assert fn is getattr(scorch_ops, symbol)
    assert got_a is a
    if isinstance(b, STensor):
        assert got_b is b
    return out


@pytest.mark.parametrize("empty", [False, True], ids=["entries", "no_entry"])
@pytest.mark.parametrize("width", [16, 4112])
def test_int32_coo_times_dense_runs_spmm_coo_float(kernels, width, empty):
    a_dense = _valued(ROWS, INNER, 1, empty)
    b = _valued(INNER, width, 2)
    a32, a64 = _coo(a_dense, torch.int32), _coo(a_dense, torch.int64)
    arrays = _arrays(a32)
    assert a32.index_dtype == torch.int32
    assert [x.dtype for x in arrays] == [torch.int32, torch.int32]
    before = [(x.data_ptr(), x.clone()) for x in arrays]

    got32 = _product(kernels, a32, b, "spmm_coo_float")
    got64 = _product(kernels, a64, b, "spmm_coo_float")

    assert isinstance(got32, torch.Tensor) and got32.dtype == torch.float32
    assert torch.equal(got32.double(), a_dense.double() @ b.double())
    assert torch.equal(got32, got64)
    after = _arrays(a32)
    assert [x.data_ptr() for x in after] == [p for p, _ in before]
    assert all(x.dtype == torch.int32 and torch.equal(x, c) for x, (_, c) in zip(after, before))


@pytest.mark.parametrize("empty", [False, True], ids=["entries", "no_entry"])
def test_int32_coo_times_coo_runs_spmspm_coo_float(kernels, empty):
    a_dense = _valued(ROWS, INNER, 3, empty)
    b_dense = _valued(INNER, COLS, 4)
    a32, b32 = _coo(a_dense, torch.int32), _coo(b_dense, torch.int32)
    a64, b64 = _coo(a_dense, torch.int64), _coo(b_dense, torch.int64)
    arrays = _arrays(a32) + _arrays(b32)
    assert a32.index_dtype == b32.index_dtype == torch.int32
    assert all(x.dtype == torch.int32 for x in arrays)
    before = [(x.data_ptr(), x.clone()) for x in arrays]

    got32 = _product(kernels, a32, b32, "spmspm_coo_float")
    got64 = _product(kernels, a64, b64, "spmspm_coo_float")

    assert isinstance(got32, STensor) and str(got32.format) == "o,o"
    expected = (a_dense.double() @ b_dense.double()).to_sparse_coo().coalesce()
    coordinates32 = [x.to(torch.int64) for x in _arrays(got32)]
    assert torch.equal(torch.stack(coordinates32), expected.indices())
    assert torch.equal(got32._raw_values.double(), expected.values())
    # The same kernel on the same (narrowed) coordinates: the same result, bit for bit.
    coordinates64 = [x.to(torch.int64) for x in _arrays(got64)]
    assert all(torch.equal(x, y) for x, y in zip(coordinates32, coordinates64))
    assert torch.equal(got32._raw_values, got64._raw_values)
    after = _arrays(a32) + _arrays(b32)
    assert [x.data_ptr() for x in after] == [p for p, _ in before]
    assert all(x.dtype == torch.int32 and torch.equal(x, c) for x, (_, c) in zip(after, before))
