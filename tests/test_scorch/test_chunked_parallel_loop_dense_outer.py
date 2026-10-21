"""The chunked parallel loop over a dense outermost level.

A parallel outermost loop whose work per iteration follows a compressed level below it
is run in chunks handed out through a shared counter.  The generated loop was written
for a loop over stored positions, which counts in ``int``.  Over a dense outermost level
the loop counts coordinates in ``int64_t`` up to an ``int64_t`` extent, and the chunk's
end, ``std::min(_start + _chunk, A0_size)``, mixed the two types: no overload, and the
kernel did not build.  ``einsum('aij,ajk->aik', format='ddd')`` with ``A`` dense,
compressed, dense and ``B`` dense, dense, compressed is the smallest such program.  The
loop now keeps the type of the loop it replaces.

Each program runs under several result formats, with few and with many outermost
iterations (one chunk is at least 16), float64 to 1e-12 of each cell.
"""

import pytest
import torch

import scorch
from tests.test_scorch.level_array_support import assert_matches, reference, stored


def _operand(shape, fill, seed):
    g = torch.Generator().manual_seed(seed)
    keep = torch.rand(shape, generator=g) < fill
    return (0.5 + torch.rand(shape, generator=g, dtype=torch.float64)) * keep


PROGRAMS = [
    ("aij,ajk->aik", ["dsd", "dds"], {"i": 3, "j": 4, "k": 3}, ["ddd", "dds", "sss"]),
    (
        "abij,abjk->abik",
        ["dsds", "ddds"],
        {"b": 2, "i": 3, "j": 2, "k": 2},
        ["dddd", "ddds", "ssss"],
    ),
]


@pytest.mark.parametrize("outer", [4, 50])
@pytest.mark.parametrize(
    "expr, formats, extents, result_format",
    [(e, f, x, out) for e, f, x, outs in PROGRAMS for out in outs],
)
def test_chunked_parallel_loop_over_a_dense_level(
    expr, formats, extents, result_format, outer
):
    extents = dict(extents, a=outer)
    terms = expr.split("->")[0].split(",")
    dense = [
        _operand(tuple(extents[name] for name in term), 0.6, 90 + position)
        for position, term in enumerate(terms)
    ]
    operands = [
        stored(value, fmt, "ABDEFG"[position])
        for position, (value, fmt) in enumerate(zip(dense, formats))
    ]
    result = scorch.einsum(expr, *operands, format=result_format)
    assert_matches(result, reference(expr, dense), torch.float64)
