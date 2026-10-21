"""``scorch.ops.spmv`` computes ``A @ x`` for an ``A`` stored in either mode order.

Its kernel reads ``A`` by rows in storage order and took the stored shape for the
logical one, so an ``A`` stored transposed (``mode_order=[1, 0]``) gave the product with
the storage layout: shape ``(5,)`` for a logical 7 x 5 matrix, and wrong values, on
864745d and on the merge candidate (mlsys, sapling jobs 82871 and 82901).
``scorch.matmul`` was right because it moves such an operand to row order first;
``spmv`` now does the same.  Both automatic arms, both vector layouts, a dense and a
sparse result, compared with the float64 product exactly.
"""

from typing import Optional

import pytest
import torch

from scorch import STensor
from scorch.ops import spmv

F64 = torch.float64
ROWS, COLS = 7, 5


@pytest.fixture(params=["0", "1"], ids=["regblock-off", "regblock-on"])
def arm(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("SCORCH_REGBLOCK", request.param)
    return request.param


def _dense_matrix() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    values = torch.randint(1, 9, (ROWS, COLS), generator=generator).to(F64)
    keep = torch.rand(ROWS, COLS, generator=generator) < 0.5
    keep[3] = False
    keep[3, 1] = True
    return values * keep


def _stored(dense: torch.Tensor, fmt: str, transposed: bool) -> STensor:
    """``dense`` in ``fmt``, stored row first or column first."""

    physical = dense.t().contiguous() if transposed else dense
    rows, cols = physical.shape
    positions = [0]
    coordinates = []
    data = []
    nonzero_rows = []
    for row in range(rows):
        columns = [col for col in range(cols) if physical[row, col] != 0]
        if fmt == "ss" and not columns:
            continue
        nonzero_rows.append(row)
        coordinates.extend(columns)
        data.extend(float(physical[row, col]) for col in columns)
        positions.append(len(coordinates))
    index = torch.int64
    columns_level = [
        torch.tensor(positions, dtype=index),
        torch.tensor(coordinates, dtype=index),
    ]
    if fmt == "ds":
        mode_indices = [[], columns_level]
    else:
        rows_level = [
            torch.tensor([0, len(nonzero_rows)], dtype=index),
            torch.tensor(nonzero_rows, dtype=index),
        ]
        mode_indices = [rows_level, columns_level]
    return STensor.from_components(
        (rows, cols),
        fmt,
        mode_indices,
        torch.tensor(data, dtype=F64),
        name="A",
        mode_order=[1, 0] if transposed else None,
    )


def _vector(fmt: str) -> STensor:
    values = torch.tensor([2.0, 0.0, -1.0, 3.0, 4.0], dtype=F64)
    if fmt == "d":
        return STensor.from_components((COLS,), "d", [[]], values, name="x")
    stored = [0, 2, 3, 4]
    return STensor.from_components(
        (COLS,),
        "s",
        [[torch.tensor([0, len(stored)]), torch.tensor(stored)]],
        values[stored],
        name="x",
    )


@pytest.mark.parametrize("matrix_format", ["ds", "ss"])
@pytest.mark.parametrize("vector_format", ["d", "s"])
@pytest.mark.parametrize("output_format", [None, "s"])
def test_spmv_of_a_transposed_matrix_is_its_product(
    arm: str,
    matrix_format: str,
    vector_format: str,
    output_format: Optional[str],
) -> None:
    dense = _dense_matrix()
    x = _vector(vector_format)
    expected = dense @ x.copy().to_torch()
    for transposed in (False, True):
        a = _stored(dense, matrix_format, transposed)
        order = a.storage.index.mode_order
        before = (None if order is None else list(order), a.values.clone())
        result = spmv(a, x, output_format=output_format)
        assert tuple(result.shape) == (ROWS,), transposed
        torch.testing.assert_close(result.to_torch(), expected, rtol=0, atol=0)
        # The operand is moved on a copy: the caller's tensor is unchanged.
        assert a.storage.index.mode_order == before[0]
        assert torch.equal(a.values, before[1])
