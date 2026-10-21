"""``A + B`` on operands stored in the same non-identity mode order, executed.

``STensor.__add__`` relays out the right operand only when its mode order differs from
the left operand's.  Two operands that already share an order -- two CSC matrices, two
tensors stored ``(k, i, j)`` -- are added by one generated kernel whose loops follow
that order, and the result keeps it.  The addition tests before this one stored every
operand in logical order.

Each result is read from its own level arrays in PyTorch and checked three ways: its
level arrays are well formed, every cell equals ``a + b`` to 1e-12 (float64), and it
stores exactly the cells the left operand's format stores for the union of the cells
either operand stores.  The extent of the outermost stored mode is 40, so loops that
are parallel run on two workers (16 rows per worker).
"""

import itertools

import pytest
import torch

from tests.test_scorch.level_array_support import assert_matches, stored
from tests.test_scorch.test_addition_carried_position import pair
from tests.test_scorch.test_addition_every_rank_three_pair import cells_a_result_stores
from tests.test_scorch.test_sum_coordinate_result_stores_each_level_once import (
    cells_stored,
)

TWO = ["dd", "ds", "sd", "ss", "oo"]
THREE_PAIRS = [
    ("sss", "sss"),
    ("dss", "sss"),
    ("ddd", "sss"),
    ("ooo", "ddd"),
    ("ssd", "dds"),
]
THREE_ORDERS = [(2, 0, 1), (1, 2, 0), (0, 2, 1), (2, 1, 0), (1, 0, 2)]


def shape_for(order, outer):
    """A logical shape whose outermost stored mode has extent ``outer``."""

    extents = [3, 4, 5][: len(order)]
    extents[order[0]] = outer
    return tuple(extents)


def assert_sum_in_shared_order(left, right, order, outer):
    shape = shape_for(order, outer)
    a, b = pair(left, right, shape, torch.float64, 43)
    # ``pair`` empties whole slices of logical mode 0; empty slices of the outermost
    # stored mode as well, so the outer loop meets stored and empty positions.
    generator = torch.Generator().manual_seed(47)
    for operand, fmt in ((a, left), (b, right)):
        if set(fmt) != {"d"}:
            keep = torch.rand(shape[order[0]], generator=generator) < 0.6
            view = [1] * len(shape)
            view[order[0]] = -1
            operand *= keep.reshape(view)

    result = stored(a, left, "A", mode_order=order) + stored(
        b, right, "B", mode_order=order
    )

    assert [int(mode) for mode in result.mode_order] == list(order)
    assert_matches(result, a + b, torch.float64)
    # Stored cells are compared in storage order, the order the formats describe.
    physical_a, physical_b = a.permute(order), b.permute(order)
    union = cells_stored(physical_a, left) | cells_stored(physical_b, right)
    assert torch.equal(
        cells_a_result_stores(result), cells_stored(union.to(torch.float64), left)
    )


def test_the_shape_puts_the_outer_extent_on_the_outermost_stored_mode():
    assert shape_for((1, 0), 40) == (3, 40)
    assert shape_for((2, 0, 1), 40) == (3, 4, 40)
    assert shape_for((0, 2, 1), 40) == (40, 4, 5)


@pytest.mark.parametrize(("left", "right"), list(itertools.product(TWO, repeat=2)))
def test_column_major_matrices_add(left, right):
    assert_sum_in_shared_order(left, right, (1, 0), 40)


@pytest.mark.parametrize("order", THREE_ORDERS)
@pytest.mark.parametrize(("left", "right"), THREE_PAIRS)
def test_rank_three_tensors_in_a_shared_order_add(left, right, order):
    assert_sum_in_shared_order(left, right, order, 40)
