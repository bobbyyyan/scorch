"""A workspace consumer inside a reduction loop must add, not store.

The scheduler inserts a workspace at one reduction loop and leaves the loops above it in
place, reductions included (``Scheduler.insert_workspace``).  Public
``einsum('abce,cef->abf')`` over a compressed ``A`` puts the dense workspace over ``f``
inside the loop over ``c``, and ``einsum('bce,cef->bf')`` puts it inside the outer loop of
a tiled ``e``.  The workspace then holds one iteration's share of the sum, and its
consumer runs once per iteration.  The shipped lowering copied it into the result with
``memcpy``, a store, so each iteration overwrote the last and the result kept only the
final iteration's share: a silent wrong answer (sweep key ``6fb83d8cc021be20``,
``abcde,cdef->abf``, relative error 1.0).  The copy-out now adds when a reduction loop
encloses the Where; the result is zeroed before the loops run.

Each program runs under several result formats, float64 to 1e-12 of each cell.
"""

import pytest
import torch

import scorch
from tests.test_scorch.level_array_support import assert_matches, reference, stored


def _operands(terms, extents, fills, seed, holes=()):
    """Operands drawn as the high-rank sweep draws them (``battle_high_rank.py``)."""

    g = torch.Generator().manual_seed(seed)
    dense = []
    for position, term in enumerate(terms):
        shape = tuple(extents[name] for name in term)
        keep = torch.rand(shape, generator=g, dtype=torch.float64) < fills[position]
        value = (0.5 + torch.rand(shape, generator=g, dtype=torch.float64)) * keep
        if position == 0:
            for mode, at in holes:
                index = [slice(None)] * len(term)
                index[mode] = at
                value[tuple(index)] = 0
        dense.append(value)
    return dense


# (expr, operand formats, extents, fills, seed, holes, result formats)
PROGRAMS = [
    # The sweep's program, operands and seed (key 6fb83d8cc021be20, float64).
    (
        "abcde,cdef->abf",
        ["sdsss", "ssds"],
        {"a": 1, "b": 4, "c": 11, "d": 4, "e": 5, "f": 2},
        (0.08, 0.5),
        int("6fb83d8cc021be20", 16) % (2**31),
        [[2, 9]],
        ["ddd", "dds", "sss"],
    ),
    # The same program with denser operands, so most (a, b, f) cells sum many (c, d).
    (
        "abcde,cdef->abf",
        ["sdsss", "ssds"],
        {"a": 2, "b": 4, "c": 6, "d": 4, "e": 5, "f": 3},
        (0.4, 0.6),
        7,
        [],
        ["ddd", "dds", "sss"],
    ),
    # One reduction loop (over c) around the workspace.
    (
        "abce,cef->abf",
        ["sdss", "sds"],
        {"a": 2, "b": 4, "c": 5, "e": 5, "f": 3},
        (0.6, 0.6),
        11,
        [],
        ["ddd", "dds", "sss"],
    ),
    # The layout sweep's program (``reduce: ijkl->jl ddds -> [dd] 2x2x4x2``, relative
    # error 0.49 on a fully stored operand): the loop over i encloses the row
    # workspace placed at the loop over k.  Then with j wider than one tile of 8.
    (
        "ijkl->jl",
        ["ddds"],
        {"i": 2, "j": 2, "k": 4, "l": 2},
        (1.0,),
        23,
        [],
        ["dd", "ds", "ss"],
    ),
    (
        "ijkl->jl",
        ["ddds"],
        {"i": 5, "j": 11, "k": 4, "l": 6},
        (0.5,),
        29,
        [],
        ["dd", "ds", "ss"],
    ),
    # A one-dimensional result: the workspace is copied out at the result's first
    # level, inside the loop over c and the outer loop of the tiled e.
    (
        "ce,cef->f",
        ["sd", "dds"],
        {"c": 5, "e": 11, "f": 3},
        (0.7, 0.6),
        17,
        [],
        ["d", "s"],
    ),
    (
        "ce,cef->f",
        ["dd", "dds"],
        {"c": 5, "e": 11, "f": 3},
        (1.0, 0.6),
        19,
        [],
        ["d", "s"],
    ),
    # Rank 3 by rank 3: the loop over c and the outer loop of the tiled e (11 > the
    # tile width of 8) both enclose the workspace.
    (
        "bce,cef->bf",
        ["dsd", "sds"],
        {"b": 4, "c": 5, "e": 11, "f": 3},
        (0.6, 0.6),
        13,
        [],
        ["dd", "ds", "ss"],
    ),
]


@pytest.mark.parametrize(
    "expr, formats, extents, fills, seed, holes, result_format",
    [(*program[:-1], out) for program in PROGRAMS for out in program[-1]],
)
def test_workspace_inside_reduction(
    expr, formats, extents, fills, seed, holes, result_format
):
    terms = expr.split("->")[0].split(",")
    dense = _operands(terms, extents, fills, seed, holes)
    expected = reference(expr, dense)
    assert int((expected != 0).sum()) > 0
    operands = [
        stored(value, fmt, "ABDEFG"[position])
        for position, (value, fmt) in enumerate(zip(dense, formats))
    ]
    result = scorch.einsum(expr, *operands, format=result_format)
    assert_matches(result, expected, torch.float64)
