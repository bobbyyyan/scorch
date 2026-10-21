"""The flat parallel loop of a coordinate-driven kernel is for two levels only.

A product whose coordinate ("o") operand has the result's indices, with every other
operand dense along them, writes one result entry per stored entry of that operand.
For a matrix the lowering turns the two nested coordinate loops into one parallel loop
over stored entries (``CINLowerer._transform_coo_loop_for_openmp``).  The transform
rewrites the outermost level and the one below it; with a third level it left that
level's loop reading ``pA1`` and ``pA1_end``, which it had removed, and wrote the third
level's end position from every thread.  ``einsum('abc,abd->abc', A, B, format='ooo')``
with a coordinate ``A`` did not build.  A result of three or more levels now keeps its
nested loops and runs them serially.
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


def _lines(source):
    return [line.strip() for line in source.splitlines() if line.strip()]


def test_three_coordinate_levels_keep_their_nested_loops(monkeypatch):
    source = _lines(
        emitted_einsum_cpp(
            monkeypatch,
            "abc,abd->abc",
            ["ooo", "ddd"],
            {"a": 3, "b": 4, "c": 2, "d": 3},
            format="ooo",
        )
    )

    assert not any(line.startswith("#pragma omp parallel") for line in source)
    assert "for (int pA0 = 0; pA0 < pA0_end; pA0 = pA1_end) {" in source
    assert "for (int pA1 = pA0; pA1 < pA1_end; pA1 = pA2_end) {" in source
    assert "for (int pA2 = pA1; pA2 < pA2_end; pA2++) {" in source


def test_two_coordinate_levels_keep_the_flat_parallel_loop(monkeypatch):
    """SDDMM with a coordinate sampling matrix: the kernel this transform is for."""

    source = _lines(
        emitted_einsum_cpp(
            monkeypatch,
            "ab,ac,bc->ab",
            ["oo", "dd", "dd"],
            {"a": 5, "b": 4, "c": 3},
            format="oo",
        )
    )

    pragma = next(n for n, line in enumerate(source) if line.startswith("#pragma omp"))
    assert source[pragma].startswith("#pragma omp parallel for")
    assert source[pragma + 1] == "for (int64_t pA0 = 0; pA0 < pA0_end; pA0++) {"


def _operand(shape, fill, seed):
    g = torch.Generator().manual_seed(seed)
    keep = torch.rand(shape, generator=g, dtype=torch.float64) < fill
    return (0.5 + torch.rand(shape, generator=g, dtype=torch.float64)) * keep


# (expr, operand formats, extents, result formats)
PROGRAMS = [
    (
        "abc,abd->abc",
        ["ooo", "ddd"],
        {"a": 3, "b": 4, "c": 2, "d": 3},
        ["ooo", "sss", "dsd"],
    ),
    (
        "abc,abd->abc",
        ["ooo", "ddd"],
        {"a": 40, "b": 30, "c": 9, "d": 5},
        ["ooo", "sss"],
    ),
    (
        "abc,ad,bcd->abc",
        ["ooo", "dd", "ddd"],
        {"a": 3, "b": 4, "c": 2, "d": 3},
        ["ooo", "sss"],
    ),
    (
        "abcd,abce->abcd",
        ["oooo", "dddd"],
        {"a": 2, "b": 3, "c": 2, "d": 3, "e": 2},
        ["oooo", "ssss"],
    ),
    # Two levels, the flat parallel loop: the control, small and at 20000 rows.
    ("ab,ac,bc->ab", ["oo", "dd", "dd"], {"a": 5, "b": 4, "c": 3}, ["oo", "ss"]),
    ("ab,ac,bc->ab", ["oo", "dd", "dd"], {"a": 20000, "b": 6, "c": 3}, ["oo"]),
]


@pytest.mark.parametrize(
    "expr, formats, extents, result_format",
    [(e, f, x, out) for e, f, x, outs in PROGRAMS for out in outs],
)
def test_coordinate_driven_product_matches(expr, formats, extents, result_format):
    terms = expr.split("->")[0].split(",")
    dense = [
        _operand(
            tuple(extents[name] for name in term),
            1.0 if set(fmt) == {"d"} else 0.4,
            90 + position,
        )
        for position, (term, fmt) in enumerate(zip(terms, formats))
    ]
    expected = reference(expr, dense)
    assert int((expected != 0).sum()) > 0
    operands = [
        stored(value, fmt, "ABDEFG"[position])
        for position, (value, fmt) in enumerate(zip(dense, formats))
    ]

    result = scorch.einsum(expr, *operands, format=result_format)

    assert_matches(result, expected, torch.float64)
