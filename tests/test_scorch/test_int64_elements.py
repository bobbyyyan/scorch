"""B and results past INT32_MAX elements (native_abi.h, CsrDenseMatmulInt64Elements).

The v2 SpMM and its tiled forms address B and the result with 64-bit offsets, so the boundary
no longer holds their element counts to int32; the legacy kernels compute offsets in int and
still are. The cheap tests below show which bound each binding applies without allocating
anything large: the bound is checked before the arrays are. The products themselves need
8.6 GB arrays and run only with SCORCH_BIGMEM_TESTS=1 on a machine with 64 GB or more.
"""

import os

import numpy as np
import pytest
import torch

import scorch
import scorch_ops
from scorch import STensor
from scorch import plan as plan_mod

BOUND = "exceeds the current native int32 allocation bound"


def _args(result_shape, b_shape, dtype=torch.float32):
    """A 2 x 3 CSR operand and a 3 x 2 B, with the shapes passed to the kernel overridden."""
    positions = torch.tensor([0, 2, 3], dtype=torch.int32)
    coordinates = torch.tensor([0, 2, 1], dtype=torch.int32)
    a_values = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    b_values = torch.arange(6, dtype=dtype)
    return (
        result_shape,
        [2, 3],
        [[], [positions, coordinates]],
        a_values,
        b_shape,
        [[], []],
        b_values,
    )


HUGE = [2**16, 2**16]  # 4.3e9 elements, each extent within int32


def test_a_legacy_kernel_keeps_the_int32_element_bound():
    with pytest.raises(RuntimeError, match=BOUND):
        scorch_ops.spmm_csr_float(*_args(HUGE, [3, 2]))
    with pytest.raises(RuntimeError, match=BOUND):
        scorch_ops.spmm_csr_float(*_args([2, 2], HUGE))


@pytest.mark.parametrize(
    "kernel, dtype",
    [
        ("spmm_csr_float_v2", torch.float32),
        ("spmm_csr_double_v2", torch.float64),
        ("spmm_csr_float_tilej", torch.float32),
        ("spmm_csr_float_tileijk", torch.float32),
    ],
)
@pytest.mark.parametrize("which", ["result", "B"])
def test_v2_and_tiled_kernels_do_not_bound_element_counts(kernel, dtype, which):
    """These still refuse the call -- the shapes do not match the arrays -- but for that reason,
    which the element bound used to pre-empt."""
    fn = getattr(scorch_ops, kernel, None)
    if fn is None:
        pytest.skip(f"{kernel} is not in this build")
    args = (
        _args(HUGE, [3, 2], dtype) if which == "result" else _args([2, 2], HUGE, dtype)
    )
    with pytest.raises(RuntimeError) as raised:
        fn(*args)
    assert BOUND not in str(raised.value)
    expected = (
        "result_shape must equal" if which == "result" else "B values must contain"
    )
    assert expected in str(raised.value)


def _bigmem():
    if os.environ.get("SCORCH_BIGMEM_TESTS") != "1":
        return False
    psutil = pytest.importorskip("psutil")
    return psutil.virtual_memory().total >= 64 * 2**30


bigmem = pytest.mark.skipif(
    not _bigmem(), reason="needs SCORCH_BIGMEM_TESTS=1 and 64 GB of memory"
)


def _operand(rows, cols, seed=0):
    """Nonzeros in every 997th row and the last 64; the last 16 rows hold 600 each, so the
    long-row kernels run at the highest offsets. Odd rows include the last column."""
    rng = np.random.default_rng(seed)
    nonempty = np.unique(
        np.concatenate([np.arange(0, rows, 997), np.arange(rows - 64, rows)])
    )
    per_row = []
    for r in nonempty:
        c = np.unique(
            rng.integers(0, cols, rng.integers(1, 10) if r < rows - 16 else 600)
        )
        if r % 2 and c[-1] != cols - 1:
            c[-1] = cols - 1
        per_row.append(c)
    counts = np.zeros(rows, dtype=np.int64)
    counts[nonempty] = [c.size for c in per_row]
    indptr = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.concatenate(per_row).astype(np.int64)
    values = rng.random(indices.size) + 0.5
    return nonempty, indptr, indices, values


def _check(result, nonempty, indptr, indices, values, b, rows, k):
    result = result.reshape(rows, k)
    bn = b.double().numpy()
    want = np.stack(
        [
            values[indptr[r] : indptr[r + 1]] @ bn[indices[indptr[r] : indptr[r + 1]]]
            for r in nonempty
        ]
    )
    got = result[torch.from_numpy(nonempty)].double().numpy()
    tol = 1e-5 if b.dtype == torch.float32 else 1e-12
    assert np.abs(got - want).max() <= tol * np.abs(want).max()
    empty = torch.ones(rows, dtype=torch.bool)
    empty[torch.from_numpy(nonempty)] = False
    assert not (result[empty] != 0).any()


@bigmem
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_product_whose_b_and_result_exceed_int32_elements(dtype):
    rows = cols = 2**23 + 1000
    k = 256  # B and the result: 2.15e9 elements each
    nonempty, indptr, indices, values = _operand(rows, cols)
    crow, col = torch.from_numpy(indptr), torch.from_numpy(indices)
    vals = torch.from_numpy(values).to(dtype)
    b = torch.rand(cols, k, dtype=dtype, generator=torch.Generator().manual_seed(1))
    a = torch.sparse_csr_tensor(crow, col, vals, size=(rows, cols))
    facts = (nonempty, indptr, indices, values, b, rows, k)

    _check(scorch.matmul(STensor.from_csr(a, "A"), b), *facts)
    enabled = plan_mod.enabled()
    plan_mod.set_enabled(True)
    try:
        for _ in range(3):
            planned = scorch.matmul(a, b)
        assert len(plan_mod.plans_of(a)) == 1, "the third call was not planned"
    finally:
        plan_mod.set_enabled(enabled)
    _check(planned, *facts)
    del planned
    if dtype == torch.float32 and hasattr(scorch_ops, "spmm_csr_float_tilej"):
        args = (
            [rows, k],
            [rows, cols],
            [[], [crow.to(torch.int32), col.to(torch.int32)]],
            vals,
            [cols, k],
            [[], []],
            b.reshape(-1),
        )
        _check(scorch_ops.spmm_csr_float_tilej(*args, cols // 4).storage.value, *facts)
        _check(
            scorch_ops.spmm_csr_float_tileijk(*args, k // 2, cols // 4).storage.value,
            *facts,
        )
