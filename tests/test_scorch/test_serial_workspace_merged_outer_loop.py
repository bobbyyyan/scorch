"""A dense workspace allocated serially before a merged outermost loop.

A kernel no parallel loop was selected for allocates its dense workspaces once, before
its outermost loop (``CINLowerer._place_unplaced_workspace_clusters``).  The placement
found that loop with ``_find_tagged_for_loop``, which matches a counted ``for`` loop
only, so when the outermost loop merged two compressed levels -- a ``while`` loop --
it found nothing and raised ``CompilerInvariantError``: ``einsum('bij,bjk->bik')`` with
``b`` compressed in both operands and the result ``dsd`` did not compile.  The placement
now finds the tagged loop of either kind.  The layout sweep's
``ijkl,ijkl->il ssss,ssdd -> [sd]`` raised the same error.

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
    (
        "bij,bjk->bik",
        ["sss", "sds"],
        {"b": 3, "i": 4, "j": 4, "k": 3},
        ["dsd", "ssd", "ddd"],
    ),
    (
        "abij,abjk->abik",
        ["sdss", "ssds"],
        {"a": 2, "b": 3, "i": 4, "j": 4, "k": 3},
        ["dssd", "ddsd", "dddd"],
    ),
    (
        "ijkl,ijkl->il",
        ["ssss", "ssdd"],
        {"i": 3, "j": 4, "k": 4, "l": 4},
        ["sd", "dd"],
    ),
    (
        "ijkl,ijkl->il",
        ["ssss", "ssdd"],
        {"i": 9, "j": 5, "k": 6, "l": 11},
        ["sd"],
    ),
]


@pytest.mark.parametrize("gap", [False, True])
@pytest.mark.parametrize(
    "expr, formats, extents, result_format",
    [(e, f, x, out) for e, f, x, outs in PROGRAMS for out in outs],
)
def test_serial_workspace_before_a_merged_outer_loop(
    expr, formats, extents, result_format, gap
):
    terms = expr.split("->")[0].split(",")
    dense = [
        _operand(tuple(extents[name] for name in term), 60 + position, gap)
        for position, term in enumerate(terms)
    ]
    operands = [
        stored(value, fmt, "ABDEFG"[position])
        for position, (value, fmt) in enumerate(zip(dense, formats))
    ]
    result = scorch.einsum(expr, *operands, format=result_format)
    assert_matches(result, reference(expr, dense), torch.float64)
