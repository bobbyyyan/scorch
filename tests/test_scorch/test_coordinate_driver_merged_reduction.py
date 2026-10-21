"""A reduction that merges compressed levels, under a coordinate input's stored entries.

``einsum('ab,ac,bc->ab')`` with a coordinate ``A`` and a coordinate result visits ``A``'s
stored entries in one loop: the loop over ``A``'s second level runs once per entry, and
single-iteration loop elimination removes it, renaming its position ``pA1`` to the
entry's position ``pA0`` in the body.  With ``B`` and ``C`` compressed along ``c`` the
reduction over ``c`` is a ``while`` loop that advances both, and the renaming did not
enter a ``while`` loop: the product kept ``A_val[pA1]`` after the loop that declared
``pA1`` was gone, and the kernel did not build (``'pA1' was not declared in this
scope``).  A dense ``B`` or ``C`` makes the reduction a ``for`` loop, which was renamed.

The renaming now enters ``while`` bodies, and a loop is removed only when no statement
left names its variable (``single_iteration_loop_pass``).

Each program runs under several result formats; float64 to 1e-12 of each cell, float32
to 1e-5, int64 exactly, with a coordinate result's entries in range, in order and
stored once.
"""

import re

import pytest
import torch

import scorch
from tests.test_scorch.level_array_support import (
    assert_matches,
    emitted_einsum_cpp,
    reference,
    stored,
)

SMALL = {"a": 5, "b": 4, "c": 6, "d": 3}
# Enough stored entries of ``A`` that the loop over them runs on several threads.
LARGE = {"a": 2000, "b": 300, "c": 40, "d": 3}

# (expression, operand formats).  In each, ``A`` is coordinate and every other operand
# is dense along ``a`` and ``b``, so the result's entries are ``A``'s; at least two
# operands are compressed along a reduced index.
MERGED = [
    ("ab,ac,bc->ab", ["oo", "ds", "ds"]),
    ("ab,ac,ac->ab", ["oo", "ds", "ds"]),
    ("ab,bc,bc->ab", ["oo", "ds", "ds"]),
    ("ab,abc,bc->ab", ["oo", "dds", "ds"]),
    ("ab,abc,abc->ab", ["oo", "dds", "dds"]),
    ("ab,ac,bc,c->ab", ["oo", "ds", "ds", "d"]),
    ("ab,ac,bc,ac->ab", ["oo", "ds", "ds", "ds"]),
    ("ab,acd,bcd->ab", ["oo", "dss", "dss"]),
    ("ab,acd,bcd->ab", ["oo", "dds", "dds"]),
    ("ab,acd,bcd->ab", ["oo", "dsd", "dsd"]),
]
# The same products with a ``for`` loop over the reduced index: built before.
NOT_MERGED = [
    ("ab,ac,bc->ab", ["oo", "ds", "dd"]),
    ("ab,ac,bc->ab", ["oo", "dd", "dd"]),
]
RESULT_FORMATS = ["oo", "ss", "ds", "dd"]
DTYPES = [torch.float64, torch.float32, torch.int64]

_POSITION = re.compile(r"\bp[A-Z]\d+(?:_end)?\b")
_POSITION_DECLARED = re.compile(r"\b(?:int|int64_t)\s+(p[A-Z]\d+(?:_end)?)\b")


def positions_never_declared(source):
    """Position variables the kernel reads and no statement declares."""

    body = source[source.index("evaluate(") :]
    return sorted(set(_POSITION.findall(body)) - set(_POSITION_DECLARED.findall(body)))


def operands(expr, formats, extents, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    dense = []
    for position, (term, fmt) in enumerate(
        zip(expr.split("->")[0].split(","), formats)
    ):
        shape = tuple(extents[index] for index in term)
        if dtype.is_floating_point:
            values = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
        else:
            values = torch.randint(1, 100, shape, generator=generator)
        if set(fmt) != {"d"}:
            # ``A`` sparse enough to be worth storing by coordinate; the factors half
            # full, so some entries of ``A`` meet no common coordinate and sum to zero.
            fill = 0.04 if position == 0 and extents is LARGE else 0.5
            keep = torch.rand(shape, generator=generator, dtype=torch.float64) < fill
            values = values * keep
        dense.append(values.to(dtype))
    return dense


def test_the_position_scan_finds_a_position_nothing_declares():
    source = "evaluate(int n) {\n  int pA0 = 0;\n  x += A_val[pA1] + B_val[pA0];\n}"
    assert positions_never_declared(source) == ["pA1"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("result_format", RESULT_FORMATS)
@pytest.mark.parametrize(("expr", "formats"), MERGED + NOT_MERGED)
def test_kernel_reads_no_position_nothing_declares(
    monkeypatch, expr, formats, result_format, dtype
):
    source = emitted_einsum_cpp(
        monkeypatch, expr, formats, SMALL, dtype=dtype, format=result_format
    )
    assert positions_never_declared(source) == []


def test_merged_reduction_reads_the_entry_position(monkeypatch):
    """``ab,ac,bc->ab``: one loop over ``A``'s entries, the product inside the merge."""

    source = [
        line.strip()
        for line in emitted_einsum_cpp(
            monkeypatch, "ab,ac,bc->ab", ["oo", "ds", "ds"], SMALL, format="oo"
        ).splitlines()
    ]

    entries = source.index("for (int64_t pA0 = 0; pA0 < pA0_end; pA0++) {")
    merge = source.index("while (pB1 < pB1_end && pC1 < pC1_end) {")
    product = source.index("_accum += A_val[pA0] * B_val[pB1] * C_val[pC1];")
    assert entries < merge < product
    assert "int b = A1_crd[pA0];" in source
    assert "D_values[pA0] = _accum;" in source
    assert not any("pA1" in line for line in source)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("result_format", RESULT_FORMATS)
@pytest.mark.parametrize(("expr", "formats"), MERGED + NOT_MERGED)
def test_merged_reduction_under_coordinate_entries_matches(
    expr, formats, result_format, dtype
):
    dense = operands(expr, formats, SMALL, dtype, 31)
    tensors = [
        stored(values, fmt, "ABDEFGH"[position])
        for position, (values, fmt) in enumerate(zip(dense, formats))
    ]

    result = scorch.einsum(expr, *tensors, format=result_format)

    assert_matches(result, reference(expr, dense), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("result_format", ["oo", "ss"])
@pytest.mark.parametrize(
    ("expr", "formats"),
    [MERGED[0], MERGED[3], MERGED[5], MERGED[7], NOT_MERGED[0]],
)
def test_merged_reduction_over_many_entries_matches(
    expr, formats, result_format, dtype
):
    dense = operands(expr, formats, LARGE, dtype, 37)
    assert int((dense[0] != 0).sum()) > 20000
    tensors = [
        stored(values, fmt, "ABDEFGH"[position])
        for position, (values, fmt) in enumerate(zip(dense, formats))
    ]

    result = scorch.einsum(expr, *tensors, format=result_format)

    assert_matches(result, reference(expr, dense), dtype)
