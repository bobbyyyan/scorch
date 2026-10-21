"""A sparse operand that stores no entry: the product is zero everywhere.

A compressed operand with nothing stored has position arrays of zeros and empty
coordinate arrays; a coordinate operand has empty coordinate arrays.  Every loop over its
entries runs zero times, and the kernel still has to hand back a well-formed result: a
dense result of zeros, a compressed result whose position arrays close every parent, a
coordinate result with no entries.

The executed sample of scanned programs skipped exactly these inputs (its runner stops
when the dense reference is all zero), so they are run here: each program under several
result formats, once with every operand stored and once per sparse operand with that
operand emptied.  float64 to 1e-12 of each cell; a result's level arrays are checked as
they are decoded.
"""

import pytest
import torch

import scorch
from tests.test_scorch.level_array_support import assert_matches, reference, stored

# (expression, operand formats, extents, result formats)
PROGRAMS = [
    ("ab,bc->ac", ["ss", "ss"], {"a": 5, "b": 4, "c": 6}, ["ss", "dd"]),
    ("ab,bc->ac", ["ds", "ds"], {"a": 5, "b": 4, "c": 6}, ["ds", "oo"]),
    ("ab,bc->ac", ["oo", "dd"], {"a": 5, "b": 4, "c": 6}, ["oo", "dd"]),
    ("ab,bc->ac", ["ss", "dd"], {"a": 5, "b": 4, "c": 6}, ["ss", "ds"]),
    ("abc,cd->abd", ["sss", "ss"], {"a": 3, "b": 4, "c": 5, "d": 3}, ["sss", "ddd"]),
    ("abc,cd->abd", ["dss", "dd"], {"a": 3, "b": 4, "c": 5, "d": 3}, ["ooo", "dss"]),
    ("abc,cd->abd", ["ooo", "dd"], {"a": 3, "b": 4, "c": 5, "d": 3}, ["ooo", "ddd"]),
    ("abc->ab", ["sss"], {"a": 4, "b": 5, "c": 3}, ["ss", "dd"]),
    ("abc->ab", ["dss"], {"a": 4, "b": 5, "c": 3}, ["ds", "ss"]),
    ("abc->ab", ["ooo"], {"a": 4, "b": 5, "c": 3}, ["oo", "dd"]),
    ("abc,abc->abc", ["sss", "sss"], {"a": 3, "b": 4, "c": 5}, ["sss"]),
    ("ab,ac,bc->ab", ["oo", "ds", "ds"], {"a": 5, "b": 4, "c": 6}, ["oo", "ss"]),
    (
        "abcd,de->abce",
        ["ssss", "ss"],
        {"a": 2, "b": 3, "c": 4, "d": 3, "e": 2},
        ["ssss", "dddd"],
    ),
    ("abc,c->ab", ["sss", "s"], {"a": 4, "b": 5, "c": 3}, ["ss", "dd"]),
    ("abc,c->ab", ["sss", "o"], {"a": 4, "b": 5, "c": 3}, ["ss", "dd"]),
    # The four programs the executed sample stopped on.
    ("abc,d->abcd", ["sdd", "s"], {"a": 11, "b": 4, "c": 9, "d": 3}, ["sdds"]),
    (
        "abc,cd,d->ab",
        ["dss", "sd", "s"],
        {"a": 11, "b": 4, "c": 9, "d": 3},
        ["ss"],
    ),
    ("a,b,c->abc", ["o", "o", "s"], {"a": 3, "b": 9, "c": 4}, ["dsd", "sss"]),
    (
        "abc,cd,d->ab",
        ["dds", "sd", "o"],
        {"a": 11, "b": 4, "c": 9, "d": 3},
        ["sd", "dd"],
    ),
]


def _dense(expr, formats, extents, seed):
    generator = torch.Generator().manual_seed(seed)
    dense = []
    for term, fmt in zip(expr.split("->")[0].split(","), formats):
        shape = tuple(extents[index] for index in term)
        values = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
        if set(fmt) != {"d"}:
            keep = torch.rand(shape, generator=generator, dtype=torch.float64) < 0.6
            values = values * keep
        dense.append(values)
    return dense


@pytest.mark.parametrize(
    "expr, formats, extents, result_format",
    [(e, f, x, out) for e, f, x, outs in PROGRAMS for out in outs],
)
def test_each_sparse_operand_emptied_in_turn(expr, formats, extents, result_format):
    dense = _dense(expr, formats, extents, 41)
    emptied = [None] + [
        position for position, fmt in enumerate(formats) if set(fmt) != {"d"}
    ]
    for position in emptied:
        values = [
            torch.zeros_like(value) if at == position else value
            for at, value in enumerate(dense)
        ]
        expected = reference(expr, values)
        assert (position is None) == bool((expected != 0).any())
        operands = [
            stored(value, fmt, "ABDEFGH"[at])
            for at, (value, fmt) in enumerate(zip(values, formats))
        ]

        result = scorch.einsum(expr, *operands, format=result_format)

        try:
            assert_matches(result, expected, torch.float64)
        except AssertionError as error:
            raise AssertionError(f"operand {position} emptied: {error}") from error
