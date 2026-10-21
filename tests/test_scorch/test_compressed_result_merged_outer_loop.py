"""A compressed result assembled under a merged outermost loop.

When the outermost loop merges the coordinates of two compressed levels it is a
``while`` loop.  ``CINLowerer._should_parallelize_compressed_where`` admitted such a
kernel to the two-pass parallel assembly of a compressed result, which handles a counted
``for`` loop only (``compressed_where_openmp_pass``), and the pass refused the kernel with
``invalid_compressed_where_root_member``: ``einsum('ij,ik->ij')`` with both operands
compressed in both levels did not compile.  The gate now admits only a counted outermost
loop; a merged one is assembled serially.  The layout sweep's
``ijkl,ijkl->il ssss,ssdd -> [ds]`` is the same refusal: both operands are compressed in
``i``.

Each program runs under several result formats, on operands with and without empty first
and last slices, float64 to 1e-12 of each cell.
"""

import pytest
import torch

import scorch
from tests.test_scorch.level_array_support import assert_matches, reference, stored


def _operand(shape, seed, gap):
    g = torch.Generator().manual_seed(seed)
    keep = torch.rand(shape, generator=g) < 0.5
    if gap and shape[0] > 1:
        keep[0] = False
        keep[-1] = False
    return (0.5 + torch.rand(shape, generator=g, dtype=torch.float64)) * keep


PROGRAMS = [
    ("ij,ik->ij", ["ss", "ss"], {"i": 5, "j": 4, "k": 3}, ["ds", "ss", "dd"]),
    (
        "bijk,bjk->bi",
        ["ssds", "ssd"],
        {"b": 3, "i": 4, "j": 2, "k": 4},
        ["ds", "ss", "dd"],
    ),
    (
        "abij,abjk->abik",
        ["sdss", "ssds"],
        {"a": 2, "b": 3, "i": 4, "j": 4, "k": 3},
        ["dsss", "ssss", "dddd"],
    ),
    (
        "ijkl,ijkl->il",
        ["ssss", "ssdd"],
        {"i": 3, "j": 4, "k": 4, "l": 4},
        ["ds", "ss", "dd"],
    ),
    (
        "ijkl,ijkl->il",
        ["ssss", "ssdd"],
        {"i": 9, "j": 5, "k": 6, "l": 11},
        ["ds", "ss"],
    ),
]


@pytest.mark.parametrize("gap", [False, True])
@pytest.mark.parametrize(
    "expr, formats, extents, result_format",
    [(e, f, x, out) for e, f, x, outs in PROGRAMS for out in outs],
)
def test_compressed_result_under_a_merged_outer_loop(
    expr, formats, extents, result_format, gap
):
    terms = expr.split("->")[0].split(",")
    dense = [
        _operand(tuple(extents[name] for name in term), 50 + position, gap)
        for position, term in enumerate(terms)
    ]
    operands = [
        stored(value, fmt, "ABDEFG"[position])
        for position, (value, fmt) in enumerate(zip(dense, formats))
    ]
    result = scorch.einsum(expr, *operands, format=result_format)
    assert_matches(result, reference(expr, dense), torch.float64)
