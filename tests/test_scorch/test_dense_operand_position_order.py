"""Dense operand positions declared parent level first.

A dense level's position is its parent's position times its extent plus its coordinate.
The declarations wait until the loops have bound their index variables, and were then
emitted in the order they were first met.  When the loops bind a dense operand's levels
inner level first, the inner level's declaration was met first, and the kernel read the
parent's position one line before declaring it: ``einsum('ab,bc->ca', format='ss')``
with a compressed ``A`` and a dense ``B`` emitted
``int pB1 = pB0 * B1_size + b; int pB0 = c;`` and did not build.  The declarations are
now emitted parents first.

Each program runs under several result formats, float64 to 1e-12 of each cell.
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
    ("ab,bc->ca", ["ss", "dd"], {"a": 5, "b": 4, "c": 3}, ["ss", "ds"]),
    ("ab,cb->ca", ["ss", "dd"], {"a": 5, "b": 4, "c": 3}, ["ss", "ds"]),
    ("ab,ac,bc->ab", ["dd", "dd", "ss"], {"a": 5, "b": 4, "c": 3}, ["ss", "ds"]),
    (
        "abc,abd,cd->abc",
        ["ddd", "ddd", "ss"],
        {"a": 3, "b": 4, "c": 2, "d": 3},
        ["sss", "dss"],
    ),
]


@pytest.mark.parametrize(
    "expr, formats, extents, result_format",
    [(e, f, x, out) for e, f, x, outs in PROGRAMS for out in outs],
)
def test_dense_operand_positions_parent_first(expr, formats, extents, result_format):
    terms = expr.split("->")[0].split(",")
    dense = [
        _operand(
            tuple(extents[name] for name in term),
            1.0 if set(fmt) == {"d"} else 0.5,
            80 + position,
        )
        for position, (term, fmt) in enumerate(zip(terms, formats))
    ]
    operands = [
        stored(value, fmt, "ABDEFG"[position])
        for position, (value, fmt) in enumerate(zip(dense, formats))
    ]
    result = scorch.einsum(expr, *operands, format=result_format)
    assert_matches(result, reference(expr, dense), torch.float64)
