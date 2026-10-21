"""``A + B`` for every pair of rank-3 layouts, executed.

``STensor.__add__`` lowers its own program (it does not go through ``einsum``) and
stores the result in the left operand's format.  The addition tests before this one
executed every pair of rank-1 and rank-2 layouts and a dozen rank-3 pairs; one more
rank-3 pair (``ooo + ddd``) turned out to return level arrays of different lengths, as
did every other sum with a rank-3 coordinate left operand.  This file executes all 81
pairs of the nine rank-3 layouts, at an extent one worker handles and at one that two
workers share (16 rows per worker).

Each result is checked three ways, from its own level arrays read in PyTorch:

* its level arrays are well formed (``logical_dense`` / ``coordinate_entries``);
* it stores exactly the cells the left operand's format stores for the union of the
  cells either operand stores -- no cell missing, none extra;
* every cell equals ``a + b`` to 1e-12 (float64).
"""

import itertools

import pytest
import torch

from scorch.stensor import STensor
from tests.test_scorch.level_array_support import assert_matches, logical_dense, stored
from tests.test_scorch.test_addition_carried_position import pair
from tests.test_scorch.test_sum_coordinate_result_stores_each_level_once import (
    cells_stored,
)

THREE = ["ddd", "dds", "dsd", "dss", "sdd", "sds", "ssd", "sss", "ooo"]
EXTENTS = {"small": (4, 3, 5), "large": (40, 3, 5)}


@pytest.fixture(autouse=True)
def _torch_allows_the_teams_these_shapes_were_built_for(torch_threads):
    # "large" gives 40 // 16 = 2 workers where a loop is parallel.
    # torch's count must allow that, or a process started with MKL_NUM_THREADS=1 runs every kernel
    # here on one worker (tests/conftest.py, torch_threads).
    torch_threads(2, needs_team=False)


def cells_a_result_stores(result):
    """Which cells ``result`` stores, read from its own level arrays."""

    ones = STensor.from_components(
        tuple(result.physical_shape),
        "".join(
            {"DENSE": "d", "COMPRESSED": "s", "COORDINATE": "o"}[level_type.name]
            for level_type in result.format.level_types
        ),
        result.storage.index.mode_indices,
        torch.ones(result.values.reshape(-1).shape, dtype=torch.float64),
        name="M",
    )
    return logical_dense(ones) != 0


def test_the_stored_cell_reading_follows_the_format():
    dense = torch.zeros((2, 3, 2), dtype=torch.float64)
    dense[1, 2, 0] = 4.0
    assert int(cells_stored(dense, "sss").sum()) == 1
    assert int(cells_stored(dense, "ssd").sum()) == 2
    assert int(cells_stored(dense, "sdd").sum()) == 6
    assert int(cells_stored(dense, "ddd").sum()) == 12
    for fmt in THREE:
        assert torch.equal(
            cells_a_result_stores(stored(dense, fmt, "A")), cells_stored(dense, fmt)
        )


@pytest.mark.parametrize("size", ["small", "large"])
@pytest.mark.parametrize(("left", "right"), list(itertools.product(THREE, repeat=2)))
def test_rank_three_addition_matches_and_stores_the_union(left, right, size):
    a, b = pair(left, right, EXTENTS[size], torch.float64, 41)

    result = stored(a, left, "A") + stored(b, right, "B")

    assert_matches(result, a + b, torch.float64)
    union = cells_stored(a, left) | cells_stored(b, right)
    assert torch.equal(
        cells_a_result_stores(result), cells_stored(union.to(torch.float64), left)
    )
