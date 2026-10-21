"""Loop order for a reduction into an all-coordinate result that no input drives.

``Scheduler.select_loop_order`` moves the reductions of a program with a sparse result
inside its result loops, so the result is assembled through a workspace.  It skipped
that for a result coordinate ("o") in every level when some input had the result's
index variables -- SDDMM, where the sampled tensor ``S[i,j]`` has them and its stored
entries are the result's.  Having them is not enough: in ``x[a] * B[a,b]`` into
``C[a]`` the dense vector ``x`` has them, but the loop over ``a`` walks the stored
entries of ``B``.  The skip now holds only when that input drives the result
(``cin.coordinate_result_driver``); every other program takes the workspace path.

Before this, the shipped lowering refused these programs with
``unsupported_coordinate_result_reduction`` (and before that, built them wrong).  Each
must now be correct under every result format -- float64 to 1e-12 of each cell, with a
coordinate result's entries in range, in order and stored once.

An input with the result's index variables does not drive the result either when
another input is compressed along one of them: the loop over that index walks the
coordinates both store, so the result can hold fewer entries than the input.  The old
kernel sized the result by the input's stored entries and returned the arrays whole,
uninitialized tail included.  ``ROW_MISSING`` gives the compressed input a row with
nothing stored where the first input stores entries.
"""

import pytest
import torch

import scorch
from tests.test_scorch.level_array_support import assert_matches, reference, stored

ONE = ["o", "s", "d"]
TWO = ["oo", "ss", "ds", "dd"]
THREE = ["ooo", "sss", "dsd"]


def _operand(shape, fill, seed):
    g = torch.Generator().manual_seed(seed)
    keep = torch.rand(shape, generator=g, dtype=torch.float64) < fill
    return (0.5 + torch.rand(shape, generator=g, dtype=torch.float64)) * keep


# ``a,ab->a`` at 6 rows runs on one thread; at 20000 rows the kernel the old schedule
# built ran its loop on every processor the host gave it, racing on the result counter.
PROGRAMS = [
    ("a,ab->a", ["d", "oo"], {"a": 6, "b": 5}, ONE),
    ("a,ab->a", ["d", "oo"], {"a": 20000, "b": 8}, ONE),
    ("a,ab->a", ["s", "oo"], {"a": 7, "b": 5}, ONE),
    ("a,ab->a", ["o", "oo"], {"a": 7, "b": 5}, ONE),
    ("ab,ac->ab", ["dd", "oo"], {"a": 5, "b": 4, "c": 6}, TWO),
    ("ab,ac->ab", ["dd", "ss"], {"a": 5, "b": 4, "c": 6}, TWO),
    ("ab,ac->ab", ["ss", "ss"], {"a": 5, "b": 4, "c": 6}, ["oo", "ss", "dd"]),
    ("ab,ac->ab", ["ds", "oo"], {"a": 5, "b": 4, "c": 6}, TWO),
    ("ab,ac->ab", ["oo", "oo"], {"a": 5, "b": 4, "c": 6}, ["oo", "ss"]),
    ("ab,ac,bc->ab", ["ss", "ss", "dd"], {"a": 5, "b": 4, "c": 3}, ["oo", "ss", "dd"]),
    ("ab,ac,bc->ab", ["ss", "dd", "ss"], {"a": 5, "b": 4, "c": 3}, TWO),
    ("ab,ac,bc->ab", ["dd", "oo", "dd"], {"a": 5, "b": 4, "c": 3}, TWO),
    ("ab,ac,bc->ab", ["dd", "ss", "dd"], {"a": 5, "b": 4, "c": 3}, TWO),
    ("ab,ac,bc->ab", ["oo", "oo", "dd"], {"a": 5, "b": 4, "c": 3}, ["oo", "ss"]),
    ("ab,ac,bc->ab", ["dd", "dd", "oo"], {"a": 5, "b": 4, "c": 3}, ["oo", "ss", "ds"]),
    ("abc,abd->abc", ["ddd", "sss"], {"a": 3, "b": 4, "c": 2, "d": 3}, THREE),
    ("abc,abd->abc", ["ddd", "ooo"], {"a": 3, "b": 4, "c": 2, "d": 3}, THREE),
]

# (expr, operand formats, extents, result formats, (operand, mode, coordinate) to empty)
ROW_MISSING = [
    ("a,ab->a", ["s", "oo"], {"a": 7, "b": 5}, ["o", "s"], (1, 0, 2)),
    ("ab,ac->ab", ["oo", "oo"], {"a": 5, "b": 4, "c": 6}, ["oo", "ss"], (1, 0, 2)),
    (
        "ab,ac->ab",
        ["ss", "ss"],
        {"a": 5, "b": 4, "c": 6},
        ["oo", "ss", "dd"],
        (1, 0, 2),
    ),
    ("ab,ac->ab", ["oo", "ss"], {"a": 5, "b": 4, "c": 6}, ["oo", "ss"], (1, 0, 2)),
    (
        "ab,ac,bc->ab",
        ["ss", "ss", "dd"],
        {"a": 5, "b": 4, "c": 3},
        ["oo", "ss", "dd"],
        (1, 0, 2),
    ),
    (
        "ab,ac,bc->ab",
        ["oo", "ss", "dd"],
        {"a": 5, "b": 4, "c": 3},
        ["oo", "ss"],
        (1, 0, 2),
    ),
    (
        "ab,ac,bc->ab",
        ["ss", "dd", "ss"],
        {"a": 5, "b": 4, "c": 3},
        ["oo", "ss"],
        (2, 0, 1),
    ),
]


@pytest.mark.parametrize(
    "expr, formats, extents, result_format, emptied",
    [(e, f, x, out, None) for e, f, x, outs in PROGRAMS for out in outs]
    + [(e, f, x, out, gap) for e, f, x, outs, gap in ROW_MISSING for out in outs],
)
def test_reduction_without_a_driving_input_is_correct(
    expr, formats, extents, result_format, emptied
):
    terms = expr.split("->")[0].split(",")
    dense = [
        _operand(
            tuple(extents[name] for name in term),
            1.0 if fmt and set(fmt) == {"d"} else 0.5,
            40 + position,
        )
        for position, (term, fmt) in enumerate(zip(terms, formats))
    ]
    if emptied is not None:
        operand, mode, coordinate = emptied
        dense[operand].select(mode, coordinate).zero_()
        # The first input stores an entry the emptied one has no coordinate for.
        at = [0] * len(terms[0])
        at[terms[0].index(terms[operand][mode])] = coordinate
        dense[0][tuple(at)] = 1.25
    operands = [
        stored(value, fmt, "ABDEFG"[position])
        for position, (value, fmt) in enumerate(zip(dense, formats))
    ]
    result = scorch.einsum(expr, *operands, format=result_format)
    assert_matches(result, reference(expr, dense), torch.float64)
