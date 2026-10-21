"""A sparse workspace drained into a result coordinate in every level.

A coordinate ("o") level stores one coordinate per entry, so a result that is coordinate
in every level stores the whole coordinate tuple with each entry.  The drain of a sparse
workspace into such a result set the workspace level's coordinate and its parent's only:
``einsum('abc,cd->abd', format='ooo')`` came back with no coordinates in level 0, and a
rank-4 result with none in levels 0 and 1.  The drain now sets every coordinate level
above the workspace's.

Each program runs under several result formats, float64 to 1e-12 of each cell, with a
coordinate result's entries in range, in order and stored once.
"""

import pytest
import torch

import scorch
from tests.test_scorch.level_array_support import assert_matches, reference, stored


def _operand(shape, seed):
    g = torch.Generator().manual_seed(seed)
    keep = torch.rand(shape, generator=g) < 0.5
    return (0.5 + torch.rand(shape, generator=g, dtype=torch.float64)) * keep


PROGRAMS = [
    # Rank 2: the parent was already set; the control.
    ("ab,bc->ac", ["ss", "ss"], {"a": 4, "b": 5, "c": 3}, ["oo", "ss", "ds"]),
    (
        "abc,cd->abd",
        ["sss", "ss"],
        {"a": 3, "b": 4, "c": 2, "d": 3},
        ["ooo", "sss", "dsd"],
    ),
    (
        "abc,cd->abd",
        ["ddd", "dd"],
        {"a": 3, "b": 4, "c": 2, "d": 3},
        ["ooo", "sss", "dsd"],
    ),
    (
        "abcd,de->abce",
        ["ssss", "ss"],
        {"a": 2, "b": 3, "c": 2, "d": 2, "e": 3},
        ["oooo", "ssss", "dsds"],
    ),
]


@pytest.mark.parametrize(
    "expr, formats, extents, result_format",
    [(e, f, x, out) for e, f, x, outs in PROGRAMS for out in outs],
)
def test_coordinate_result_drain_sets_every_level(
    expr, formats, extents, result_format
):
    terms = expr.split("->")[0].split(",")
    dense = [
        _operand(tuple(extents[name] for name in term), 70 + position)
        for position, term in enumerate(terms)
    ]
    operands = [
        stored(value, fmt, "ABDEFG"[position])
        for position, (value, fmt) in enumerate(zip(dense, formats))
    ]
    result = scorch.einsum(expr, *operands, format=result_format)
    assert_matches(result, reference(expr, dense), torch.float64)
