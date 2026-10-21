"""``to_sparse``, ``to_dense`` and ``to_torch`` between every pair of layouts, ranks 1-3.

A conversion is a generated kernel like any other, and every public operation that
relays an operand out runs one first; a wrong conversion is wrong input to the kernel
that follows.  The source is built as level arrays in PyTorch and the converted tensor
is read back from its own level arrays, so each test runs the one conversion kernel.

Every source layout (each level dense or compressed, or coordinate in every level) goes
to every other layout, to dense, and to a ``torch.Tensor``.  40 or more rows, some of
them empty, so a conversion whose row loop is parallel runs on several workers (one per
16 rows).  float64 to 1e-12 of each cell; the result must report the layout asked for.

``change_mode_order`` is not here: it is tested with the relayout work.
"""

import itertools

import pytest
import torch

from tests.test_scorch.level_array_support import assert_matches, stored

SHAPES = {1: (64,), 2: (48, 7), 3: (40, 5, 4)}


def layouts(rank):
    return ["".join(kinds) for kinds in itertools.product("ds", repeat=rank)] + [
        "o" * rank
    ]


def _dense(rank, seed):
    generator = torch.Generator().manual_seed(seed)
    shape = SHAPES[rank]
    values = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
    keep = torch.rand(shape, generator=generator, dtype=torch.float64) < 0.4
    rows = torch.rand(shape[0], generator=generator, dtype=torch.float64) < 0.7
    return values * (keep & rows.reshape((-1,) + (1,) * (rank - 1)))


def _kind(letter):
    return {"d": "DENSE", "s": "COMPRESSED", "o": "COORDINATE"}[letter]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (source, target)
        for rank in (1, 2, 3)
        for source in layouts(rank)
        for target in layouts(rank)
        if source != target
    ],
)
def test_conversion_between_layouts_matches(source, target):
    dense = _dense(len(source), 53)
    tensor = stored(dense, source, "A")

    if set(target) == {"d"}:
        converted = tensor.to_dense()
    else:
        converted = tensor.to_sparse(target)

    assert [level.name for level in converted.format.level_types] == [
        _kind(letter) for letter in target
    ]
    assert_matches(converted, dense, torch.float64)


@pytest.mark.parametrize(
    "source", [layout for rank in (1, 2, 3) for layout in layouts(rank)]
)
def test_to_torch_matches(source):
    dense = _dense(len(source), 59)

    result = stored(dense, source, "A").to_torch()

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result.to(torch.float64), dense)
