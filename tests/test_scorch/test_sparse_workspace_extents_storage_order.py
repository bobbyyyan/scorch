"""A coordinate workspace is sized by the result's extents in storage order.

``coo_workspace<T, N>`` takes one extent per key component and rejects a coordinate at
or past its component's extent.  The kernel's ``result_shape`` argument holds the
result's extents by storage level.  ``_sparse_workspace_extents`` looked a key
variable's extent up at the variable's position in the result *access*, which is the
logical mode.  The two agree only while the result is stored in its logical order.

Public ``einsum`` stores a result in the order its loops visit the result's indices,
so ``einsum('abc->cb')`` stores ``(b, c)``.  Its workspace is keyed ``(b, c)`` and was
given the extents ``(c, b)``.  With a 3-wide ``b`` and a 2-wide ``c`` the first insert
at ``b == 2`` raised ``RuntimeError: workspace coordinate is out of range``.
"""

import pytest
import torch

import scorch
from tests.test_scorch.level_array_support import (
    assert_matches,
    emitted_einsum_cpp,
    reference,
    stored,
)

# (expression, operand shape, operand format, result format)
REORDERED = [
    ("abc->cb", (2, 3, 2), "sss", "sd"),
    ("abc->cb", (2, 3, 2), "sss", "ss"),
    ("abc->cb", (2, 2, 3), "sss", "sd"),
    ("abcd->cb", (4, 2, 3, 2), "ssss", "sd"),
    ("abcd->dcb", (2, 4, 3, 2), "ssss", "ssd"),
]


def operand(shape, dtype, fill):
    generator = torch.Generator().manual_seed(11)
    if dtype.is_floating_point:
        values = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
    else:
        values = torch.randint(1, 1000, shape, generator=generator)
    keep = torch.rand(shape, generator=generator, dtype=torch.float64) < fill
    return (values * keep).to(dtype)


def test_workspace_extents_follow_the_key_through_the_result_storage_order(
    monkeypatch,
):
    """``abc->cb`` stores ``(b, c)`` and keys its workspace ``(b, c)``."""

    source = emitted_einsum_cpp(
        monkeypatch, "abc->cb", ["sss"], {"a": 2, "b": 3, "c": 2}, format="sd"
    )

    assert "wksp.insert({b, c}," in source
    assert "coo_workspace<double, 2>(1024, result_shape);" in source
    assert "result_shape[1], result_shape[0]" not in source


def test_workspace_keyed_on_part_of_the_result_takes_those_levels(monkeypatch):
    """An identity-order result keyed on its last two levels is unchanged."""

    source = emitted_einsum_cpp(
        monkeypatch,
        "abcd->acd",
        ["ssss"],
        {"a": 2, "b": 3, "c": 4, "d": 5},
        format="sss",
    )

    assert "wksp.insert({c, d}," in source
    assert "(1024, {result_shape[1], result_shape[2]});" in source


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
@pytest.mark.parametrize("fill", [1.0, 0.5])
@pytest.mark.parametrize(("expr", "shape", "fmt", "result_fmt"), REORDERED)
def test_reduction_into_a_reordered_sparse_result_matches(
    expr, shape, fmt, result_fmt, fill, dtype
):
    dense = operand(shape, dtype, fill)

    result = scorch.einsum(expr, stored(dense, fmt, "A"), format=result_fmt)

    assert_matches(result, reference(expr, [dense]), dtype)
