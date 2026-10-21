"""The prebuilt kernels' refusals name their kernel (patch 0075).

``bind_typed_prebuilt_kernels`` (csrc/ops.cpp) builds the names of its twelve entries,
``prebuilt_{spmm,spmspm,spmv}_csr_{f32,f64,i32,i64}``, as local strings and handed
their ``c_str()`` to the binders, whose function objects kept the pointer.  The strings
were destroyed when registration returned, so every refusal those entries made printed
freed memory as the kernel's name: an empty or garbled name, or a ``UnicodeDecodeError``
in place of the ``RuntimeError`` when the bytes were not UTF-8, which also hid a coded
refusal from ``raise_if_native_coded``.  The binders now keep their own copy.

1. Each of the twelve entries, called repeatedly with a contraction mismatch, raises a
   ``RuntimeError`` whose first line is the native check's, with the entry's own name and
   the code ``raise_if_native_coded`` reads (patch 0074b).
2. Each of the twelve, called with a result past the 32-bit position bound, reaches
   Python as text ``raise_if_native_coded`` turns into ``IndexCapacityError``, whose
   message still names the entry.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest
import torch

import scorch_ops
from scorch.exceptions import IndexCapacityError, defect_code, raise_if_native_coded
from scorch.stensor import STensor

ROWS, COLS, WIDTH = 6, 4, 3
DTYPES = {"f32": torch.float32, "f64": torch.float64, "i32": torch.int32, "i64": torch.int64}
ENTRIES = [
    f"prebuilt_{op}_csr_{suffix}" for op in ("spmm", "spmspm", "spmv") for suffix in DTYPES
]


def _matrix(rows: int, cols: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(rows, cols, generator=g) < 0.5
    values = torch.randint(1, 9, (rows, cols), generator=g)
    dense = (values * mask).to(dtype)
    dense[0, 0] = 1
    return dense


def _abi(tensor: STensor) -> list:
    """An operand as the native entries take it: shape, levels' arrays, values."""
    return [list(tensor.shape), tensor._native_mode_indices(), tensor._raw_values]


def _arguments(entry: str, b_rows: int, result_shape=None) -> Tuple[list, List[int]]:
    """The entry's arguments for an A of COLS columns and a B of ``b_rows`` rows."""
    dtype = DTYPES[entry.rsplit("_", 1)[1]]
    a = STensor.from_torch(_matrix(ROWS, COLS, dtype, 1)).to_sparse("ds")
    if "_spmv_" in entry:
        b = STensor.from_torch(_matrix(b_rows, 1, dtype, 2).reshape(-1))
        shape = [ROWS]
    elif "_spmspm_" in entry:
        b = STensor.from_torch(_matrix(b_rows, WIDTH, dtype, 2)).to_sparse("ds")
        shape = [ROWS, WIDTH]
    else:
        b = STensor.from_torch(_matrix(b_rows, WIDTH, dtype, 2))
        shape = [ROWS, WIDTH]
    shape = shape if result_shape is None else result_shape
    return [shape] + _abi(a) + _abi(b), shape


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_refusal_names_its_kernel(entry):
    args, _ = _arguments(entry, COLS + 1)
    expected = (
        f"{entry}: contraction mismatch: A.shape[1]={COLS} but B.shape[0]={COLS + 1}"
        " [einsum_index_extent_mismatch]"
    )
    for _ in range(20):
        # Strings made and dropped between calls, so freed memory is written again.
        churn = [str(n) * 40 for n in range(200)]
        del churn
        with pytest.raises(RuntimeError) as raised:
            getattr(scorch_ops, entry)(*args)
        assert str(raised.value).splitlines()[0] == expected


def _past_the_bound(entry: str) -> list:
    """A result shape the native checks refuse at the 32-bit position bound before they
    read anything else of the result."""
    if "_spmv_" in entry:
        return [2**31]
    if "_spmspm_" in entry:
        return [2**31, WIDTH]  # 2**31 + 1 row offsets
    return [2**16, 2**16]  # 2**32 dense entries


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_coded_refusal_keeps_its_code(entry):
    args, _ = _arguments(entry, COLS, result_shape=_past_the_bound(entry))
    with pytest.raises(IndexCapacityError) as raised:
        try:
            getattr(scorch_ops, entry)(*args)
        except RuntimeError as error:
            raise_if_native_coded(error)
            raise
    assert defect_code(raised.value) == "index_position_exceeds_int32"
    assert str(raised.value).startswith(entry), str(raised.value)
