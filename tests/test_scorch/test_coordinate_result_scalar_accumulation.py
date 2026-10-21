"""Reductions into an all-coordinate result, through scalar accumulation.

The shipped lowering can reduce into a result that is coordinate ("o") in every level
with one scalar accumulator per result entry: the entry is appended when its reduction
ends, and the result's arrays are sized by one input's stored entries.  That is the
result only when the loops over the result's indices walk exactly that input's stored
entries, one result entry each -- an input with exactly the result's indices, every
other input dense along them (SDDMM: ``S[i,j] * Q[i,k] * K[j,k]`` with dense ``Q`` and
``K``).

The lowering checked neither condition.  Public ``einsum("a,ab->a", x, B)`` with a
dense ``x`` and a coordinate ``B`` stored ``a`` once per stored entry of ``B`` and never
summed over ``b``, through a result counter shared by the threads of a parallel loop;
with many rows the increments raced and the result held coordinates nobody wrote.  With
a compressed or dense driving input the arrays were sized by the first input with a
sparse last level, whichever it was, and the kernel wrote past them or returned their
unwritten tail.  The lowering now refuses with
``unsupported_coordinate_result_reduction`` when no input drives the result, and sizes
the result by the input that does.

Each public program runs under several result formats.  A coordinate result may be
refused with the code; any result that is returned must hold the reference -- float64
to 1e-12 of each cell -- with its coordinates in range, in order and stored once.
"""

import re

import pytest
import torch

import scorch
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.diagnostics import UnsupportedFeature
from scorch.compiler.loopir.pipeline import legacy_generated_cpp
from tests.test_scorch.level_array_support import (
    assert_matches,
    emitted_einsum_cpp,
    reference,
    stored,
)

CODE = "unsupported_coordinate_result_reduction"


def _codes(error):
    return [getattr(d, "code", None) for d in getattr(error, "diagnostics", ()) or ()]


def _operand(shape, fill, seed):
    g = torch.Generator().manual_seed(seed)
    keep = torch.rand(shape, generator=g, dtype=torch.float64) < fill
    return (0.5 + torch.rand(shape, generator=g, dtype=torch.float64)) * keep


def _run(expr, formats, result_format, dense):
    operands = [
        stored(value, fmt, "ABDEFG"[position])
        for position, (value, fmt) in enumerate(zip(dense, formats))
    ]
    return scorch.einsum(expr, *operands, format=result_format)


def _raw_reduction(expr, formats, result_format, extents):
    """The CIN ``einsum`` would build, loops in first-appearance order, unscheduled."""

    lhs, out = expr.split("->")
    terms = lhs.split(",")
    names = "".join(dict.fromkeys("".join(terms)))
    index = {name: IndexVar(name) for name in names}
    rhs = None
    for position, (term, fmt) in enumerate(zip(terms, formats)):
        access = TensorVar("ABDEFG"[position], fmt=fmt, dtype=torch.float64)[
            tuple(index[name] for name in term)
        ]
        rhs = access if rhs is None else rhs * access
    result = TensorVar("C", fmt=result_format, dtype=torch.float64)
    stmt = TensorAssign(
        result[tuple(index[name] for name in out)], rhs, op=Operation.ADD
    )
    for name in reversed(names):
        stmt = ForAll(index[name], stmt)
    bindings = tuple(
        (tuple(extents[name] for name in term), torch.float64) for term in terms
    )
    return stmt, tuple(extents[name] for name in out), bindings


# Raw reductions handed straight to the lowering, no scheduler in between: the three
# programs first reported, and the public race's program.  Nothing is compiled.
@pytest.mark.parametrize(
    "expr, formats, result_format",
    [
        ("ij->i", ["ss"], "o"),
        ("ij->i", ["oo"], "o"),
        ("ijk->ij", ["ooo"], "oo"),
        ("a,ab->a", ["d", "oo"], "o"),
        ("a,ab->a", ["d", "ss"], "o"),
    ],
)
def test_lowering_refuses_a_reduction_without_a_driving_input(
    expr, formats, result_format
):
    program = _raw_reduction(
        expr, formats, result_format, {"i": 3, "j": 4, "k": 2, "a": 3, "b": 4}
    )
    with pytest.raises(UnsupportedFeature) as caught:
        legacy_generated_cpp(*program)
    assert CODE in _codes(caught.value)


def _refused_or_correct(expr, formats, result_format, dense):
    try:
        result = _run(expr, formats, result_format, dense)
    except UnsupportedFeature as error:
        assert set(result_format) == {"o"}, f"{result_format} refused: {error}"
        assert CODE in _codes(error)
        return
    assert_matches(result, reference(expr, dense), torch.float64)


# No input drives the result.  ``a,ab->a`` at 6 rows runs its loop on one thread; at
# 20000 rows the shipped kernel's loop runs on every processor the host gives it.
@pytest.mark.parametrize("result_format", ["o", "s", "d"])
@pytest.mark.parametrize("rows, cols", [(6, 5), (20000, 8)])
def test_dense_vector_times_coordinate_matrix(rows, cols, result_format):
    dense = [_operand((rows,), 1.0, 1), _operand((rows, cols), 0.6, 2)]
    _refused_or_correct("a,ab->a", ["d", "oo"], result_format, dense)


TWO = ["oo", "ss", "ds", "dd"]
THREE = ["ooo", "sss", "dsd"]
NO_DRIVER = [
    ("ab,ac->ab", ["dd", "oo"], (5, 4, 6), TWO),
    ("ab,ac->ab", ["dd", "ss"], (5, 4, 6), TWO),
    ("ab,ac->ab", ["ss", "ss"], (5, 4, 6), ["oo", "ss", "dd"]),
    ("ab,ac,bc->ab", ["ss", "ss", "dd"], (5, 4, 3), ["oo", "ss", "dd"]),
    ("ab,ac,bc->ab", ["dd", "oo", "dd"], (5, 4, 3), TWO),
    ("abc,abd->abc", ["ddd", "sss"], (3, 4, 2, 3), THREE),
]


@pytest.mark.parametrize(
    "expr, formats, sizes, result_format",
    [(e, f, n, out) for e, f, n, outs in NO_DRIVER for out in outs],
)
def test_reduction_without_a_driving_input(expr, formats, sizes, result_format):
    extents = dict(zip("abcd", sizes))
    terms = expr.split("->")[0].split(",")
    dense = [
        _operand(tuple(extents[name] for name in term), 0.5, 10 + position)
        for position, term in enumerate(terms)
    ]
    _refused_or_correct(expr, formats, result_format, dense)


# An input drives the result, so scalar accumulation applies; the result is sized by
# that input.  The shipped lowering sized it by the first input whose last level is
# sparse: B (more stored entries than the result has) for the first two, and A -- sparse
# along the REDUCED index c -- for the third.  The extents make the sizing input store
# more entries than the result, so the old kernel returned an unwritten tail.
SIZED = [
    ("a,ab->a", ["d", "ds"], {"a": 4, "b": 6}, (1.0, 0.8)),
    ("ab,ac->ab", ["dd", "ds"], {"a": 3, "b": 2, "c": 5}, (1.0, 0.9)),
    ("ac,ab->ab", ["ds", "ss"], {"a": 4, "b": 5, "c": 6}, (0.9, 0.3)),
]


@pytest.mark.parametrize(
    "expr, formats, extents, fills, result_format",
    [
        (*program, out)
        for program in SIZED
        for out in (["o", "s", "d"] if program[0] == "a,ab->a" else TWO)
    ],
)
def test_result_sized_by_its_driving_input(
    expr, formats, extents, fills, result_format
):
    terms = expr.split("->")[0].split(",")
    dense = [
        _operand(tuple(extents[name] for name in term), fill, 20 + position)
        for position, (term, fill) in enumerate(zip(terms, fills))
    ]
    result = _run(expr, formats, result_format, dense)
    assert_matches(result, reference(expr, dense), torch.float64)


# Negative control: SDDMM, and a rank-3 program of the same shape, keep scalar
# accumulation sized by the sampled tensor, and are correct under every result format.
DRIVEN = [
    ("ab,ac,bc->ab", ["oo", "dd", "dd"], (6, 5, 4), TWO),
    ("ab,ac,bc->ab", ["ss", "dd", "dd"], (6, 5, 4), TWO),
    ("ab,ac,bc->ab", ["ds", "dd", "dd"], (6, 5, 4), TWO),
    ("abc,ad,bcd->abc", ["sss", "dd", "ddd"], (3, 4, 2, 3), THREE),
    ("abc,abd->abc", ["sss", "ddd"], (3, 4, 2, 3), THREE),
]


@pytest.mark.parametrize(
    "expr, formats, sizes, result_format",
    [
        (expr, formats, sizes, out)
        for expr, formats, sizes, outs in DRIVEN
        for out in outs
    ],
)
def test_driven_reduction_keeps_scalar_accumulation(
    monkeypatch, expr, formats, sizes, result_format
):
    extents = dict(zip("abcd", sizes))
    terms = expr.split("->")[0].split(",")
    if set(result_format) == {"o"}:
        source = emitted_einsum_cpp(
            monkeypatch, expr, formats, extents, format=result_format
        )
        assert "_accum" in source
        assert re.search(r"_known_nnz = A_values\.size\(0\)", source)
        monkeypatch.undo()
    dense = [
        _operand(tuple(extents[name] for name in term), 0.5, 30 + position)
        for position, term in enumerate(terms)
    ]
    result = _run(expr, formats, result_format, dense)
    assert_matches(result, reference(expr, dense), torch.float64)
