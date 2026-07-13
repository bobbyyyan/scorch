"""ASan/UBSan regression probes for native output ownership.

These tests are skipped by the ordinary suite. ``run_sanitizers.sh`` rebuilds
both the prebuilt extension and generated JIT kernels through the sanitizer
compiler shim, preloads ASan before Python starts, and enables this module.
"""

from __future__ import annotations

import ctypes
import gc
import os
from itertools import chain

import pytest
import torch

import scorch
import scorch_ops
from scorch import STensor

pytestmark = pytest.mark.skipif(
    os.environ.get("SCORCH_SANITIZER_RUN") != "1",
    reason="run with tests/sanitizers/run_sanitizers.sh",
)

GROWTH_BOUNDARIES = (
    0,
    1,
    2,
    3,
    7,
    8,
    9,
    15,
    16,
    17,
    31,
    32,
    33,
    1023,
    1024,
    1025,
)


def _assert_asan_is_preloaded() -> None:
    process = ctypes.CDLL(None)
    assert any(
        hasattr(process, symbol) for symbol in ("__asan_init", "___asan_init")
    ), "ASan must be loaded before Python imports an instrumented extension"


def _owned_stensor_buffers(tensor: STensor) -> list[torch.Tensor]:
    return [
        tensor.values,
        *chain.from_iterable(tensor.index.mode_indices),
    ]


def _owned_native_buffers(result) -> list[torch.Tensor]:
    return [
        result.storage.value,
        *chain.from_iterable(result.storage.index.mode_indices),
    ]


def _assert_buffers_survive_owner_destruction(
    buffers: list[torch.Tensor], snapshots: list[torch.Tensor]
) -> None:
    gc.collect()

    # Churn both libc and the Torch CPU allocator before reading through the
    # retained tensor handles. ASan turns a stale from_blob handoff into a
    # deterministic use-after-free report rather than a flaky value mismatch.
    libc_churn = [bytearray(4096) for _ in range(64)]
    torch_churn = [torch.empty(4096, dtype=torch.uint8) for _ in range(64)]
    assert libc_churn and torch_churn

    for buffer, snapshot in zip(buffers, snapshots):
        assert torch.equal(buffer, snapshot)


def test_unknown_nnz_jit_outputs_cross_growth_boundaries() -> None:
    """Exercise generated sparse growth and Torch ownership transfer."""

    _assert_asan_is_preloaded()
    extent = max(GROWTH_BOUNDARIES) + 5

    for nnz in GROWTH_BOUNDARIES:
        left_dense = torch.zeros(extent)
        right_dense = torch.zeros(extent)
        split = nnz // 2
        left_dense[:split] = torch.arange(1, split + 1, dtype=torch.float32)
        right_dense[split:nnz] = torch.arange(split + 1, nnz + 1, dtype=torch.float32)
        expected = left_dense + right_dense

        left = STensor.from_torch(left_dense, "SanLeft").to_sparse("s")
        right = STensor.from_torch(right_dense, "SanRight").to_sparse("s")
        result = left + right

        assert result.values.numel() == nnz
        assert torch.equal(result.to_torch(in_place=False), expected)

        retained = _owned_stensor_buffers(result)
        snapshots = [buffer.clone() for buffer in retained]
        del result, left, right
        _assert_buffers_survive_owner_destruction(retained, snapshots)


def test_empty_grouped_coordinate_jit_output() -> None:
    """An empty COO input must not probe coordinate zero during pre-scans."""

    _assert_asan_is_preloaded()
    left = STensor.from_torch(torch.zeros(3, 5), "SanEmptyLeft").to_sparse("oo")
    right = STensor.from_torch(torch.zeros(3, 5), "SanEmptyRight").to_sparse("oo")
    result = left + right

    assert result.values.numel() == 0
    assert torch.equal(result.to_torch(in_place=False), torch.zeros(3, 5))

    retained = _owned_stensor_buffers(result)
    snapshots = [buffer.clone() for buffer in retained]
    del result, left, right
    _assert_buffers_survive_owner_destruction(retained, snapshots)


def test_zero_column_compressed_workspace_jit_output() -> None:
    """A valid zero extent must not throw while constructing an OMP workspace."""

    _assert_asan_is_preloaded()
    left_dense = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]], dtype=torch.float32)
    right_dense = torch.empty((2, 0), dtype=torch.float32)
    left = STensor.from_torch(left_dense, "SanZeroColumnLeft").to_sparse("ds")
    right = STensor.from_torch(right_dense, "SanZeroColumnRight").to_sparse("ds")

    result = scorch.matmul(left, right, output_format="ds", use_cache=False)

    assert tuple(result.shape) == (3, 0)
    assert result.values.numel() == 0
    assert torch.equal(result.to_torch(in_place=False), left_dense @ right_dense)


def _csr_diagonal(nnz: int, extent: int):
    positions = torch.minimum(
        torch.arange(extent + 1, dtype=torch.int32),
        torch.tensor(nnz, dtype=torch.int32),
    )
    coordinates = torch.arange(nnz, dtype=torch.int32)
    values = torch.arange(1, nnz + 1, dtype=torch.float32)
    return positions, coordinates, values


def test_native_csr_spgemm_owns_empty_and_growing_outputs() -> None:
    """Exercise former cvector owners in the prebuilt CSR SpGEMM path."""

    _assert_asan_is_preloaded()
    extent = max(GROWTH_BOUNDARIES) + 5
    identity_positions = torch.arange(extent + 1, dtype=torch.int32)
    identity_coordinates = torch.arange(extent, dtype=torch.int32)
    identity_values = torch.ones(extent)

    for nnz in GROWTH_BOUNDARIES:
        a_positions, a_coordinates, a_values = _csr_diagonal(nnz, extent)
        result = scorch_ops.spmspm_csr_float(
            [extent, extent],
            [extent, extent],
            [[], [a_positions, a_coordinates]],
            a_values,
            [extent, extent],
            [[], [identity_positions, identity_coordinates]],
            identity_values,
        )

        assert result.storage.value.tolist() == a_values.tolist()
        retained = _owned_native_buffers(result)
        snapshots = [buffer.clone() for buffer in retained]
        del result
        _assert_buffers_survive_owner_destruction(retained, snapshots)


def test_native_csr_spgemm_owns_long_row_sort_scratch() -> None:
    """Exercise the move-only SortEntry pool used for output rows over 32."""

    _assert_asan_is_preloaded()
    width = 65
    result = scorch_ops.spmspm_csr_float(
        [1, width],
        [1, 1],
        [
            [],
            [
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
            ],
        ],
        torch.tensor([2.0]),
        [1, width],
        [
            [],
            [
                torch.tensor([0, width], dtype=torch.int32),
                torch.arange(width, dtype=torch.int32),
            ],
        ],
        torch.arange(1, width + 1, dtype=torch.float32),
    )

    positions, coordinates = result.storage.index.mode_indices[1]
    assert positions.tolist() == [0, width]
    assert coordinates.tolist() == list(range(width))
    assert (
        result.storage.value.tolist()
        == (2 * torch.arange(1, width + 1, dtype=torch.float32)).tolist()
    )

    retained = _owned_native_buffers(result)
    snapshots = [buffer.clone() for buffer in retained]
    del result
    _assert_buffers_survive_owner_destruction(retained, snapshots)


def test_native_coo_spgemm_owns_empty_and_growing_outputs() -> None:
    """Exercise former cvector owners in the prebuilt COO SpGEMM path."""

    _assert_asan_is_preloaded()
    extent = max(GROWTH_BOUNDARIES) + 5
    identity_coordinates = torch.arange(extent, dtype=torch.int32)
    identity_values = torch.ones(extent)

    for nnz in GROWTH_BOUNDARIES:
        coordinates = torch.arange(nnz, dtype=torch.int32)
        values = torch.arange(1, nnz + 1, dtype=torch.float32)
        result = scorch_ops.spmspm_coo_float(
            [extent, extent],
            [extent, extent],
            [[coordinates], [coordinates]],
            values,
            [extent, extent],
            [[identity_coordinates], [identity_coordinates]],
            identity_values,
        )

        assert result.storage.value.tolist() == values.tolist()
        retained = _owned_native_buffers(result)
        snapshots = [buffer.clone() for buffer in retained]
        del result
        _assert_buffers_survive_owner_destruction(retained, snapshots)


@pytest.mark.parametrize("width", (1, 2, 1023, 1024, 1025, 4095, 4096, 4097))
def test_native_coo_spmm_owns_residual_and_tile_boundary_outputs(width: int) -> None:
    """Exercise dense Torch ownership around the COO SpMM tile boundary."""

    _assert_asan_is_preloaded()
    a_rows = torch.tensor([0, 1], dtype=torch.int32)
    a_columns = torch.tensor([0, 2], dtype=torch.int32)
    a_values = torch.tensor([2.0, 3.0])
    b_values_2d = torch.arange(3 * width, dtype=torch.float32).reshape(3, width)

    result = scorch_ops.spmm_coo_float(
        [2, width],
        [2, 3],
        [[a_rows], [a_columns]],
        a_values,
        [3, width],
        [[], []],
        b_values_2d.reshape(-1),
    )

    expected = torch.stack((2 * b_values_2d[0], 3 * b_values_2d[2])).reshape(-1)
    assert torch.equal(result.storage.value, expected)
    retained = _owned_native_buffers(result)
    snapshots = [buffer.clone() for buffer in retained]
    del result
    _assert_buffers_survive_owner_destruction(retained, snapshots)
