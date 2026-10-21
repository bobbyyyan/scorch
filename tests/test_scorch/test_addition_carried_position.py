"""``A + B`` with a dense ``A``: the loop over rows carries ``B``'s position.

``STensor.__add__`` stores its result in the left operand's format.  With a dense left
operand and a right operand whose first level is compressed or coordinate, the loop
over the first index runs over the dense extent, and the right operand's position at
that level is declared before the loop and advanced at the end of each iteration::

    int pC0 = C0_pos[0];
    for (int64_t i0 = 0; i0 < B0_size; i0++) {
      int i0_C = pC0 < pC0_end ? C0_crd[pC0] : 0;
      ...
      pC0 += (int)(i0_C == i0);
    }

Iteration ``i0`` is right only after iterations ``0 .. i0 - 1`` have run, in order.
The outermost loop was marked ``omp parallel for`` all the same, so every worker read
and advanced the one shared ``pC0``.  The thread count is one worker per 16 rows
(``scorch_nthreads``): results were right under 32 rows, which is what the existing
addition tests use, and wrong from 32 rows up on any host with two processors.

A loop whose body assigns a scalar declared outside it is now left serial
(``parallel_marking_pass.scalars_carried_between_iterations``).

Operands are built and results read as level arrays in PyTorch, so each test runs one
generated kernel.  float64 to 1e-12 of each cell, float32 to 1e-5, int64 exactly.
"""

import itertools
import re

import pytest
import torch

from scorch import stensor
from tests.test_scorch.level_array_support import assert_matches, stored

ONE = ["d", "s", "o"]
TWO = ["dd", "ds", "sd", "ss", "oo"]
THREE = ["ddd", "dss", "sds", "ssd", "sdd", "sss", "ooo"]
# Dense left operand, right operand compressed or coordinate in its first level.
CARRIED = [
    ("d", "s"),
    ("d", "o"),
    ("dd", "ss"),
    ("dd", "sd"),
    ("dd", "oo"),
    ("ddd", "sss"),
    ("ddd", "ssd"),
    ("ddd", "sds"),
    ("ddd", "sdd"),
    ("ddd", "ooo"),
]
SMALL = {1: (7,), 2: (5, 4), 3: (4, 3, 5)}
# 16 rows per worker: several workers from 32 rows up.
LARGE = {1: (4000,), 2: (600, 9), 3: (200, 5, 4)}

_DECLARED = re.compile(r"\b(?:int|int64_t|double|float|bool|auto)\s+([A-Za-z_]\w*)")
_ASSIGNED = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\+=|-=|=(?!=)|\+\+)", re.MULTILINE)


class _Emitted(Exception):
    """Raised in place of building the kernel."""


def parallel_loops_assigning_outer_scalars(source):
    """``(loop header, names)`` per OpenMP loop whose body assigns a scalar it does
    not declare."""

    lines = source.splitlines()
    found = []
    for at, line in enumerate(lines):
        if "#pragma omp" not in line or " for" not in line:
            continue
        depth, loop = 0, []
        for inner in lines[at + 1 :]:
            depth += inner.count("{") - inner.count("}")
            loop.append(inner)
            if depth == 0:
                break
        body = "\n".join(text.split("//")[0] for text in loop[1:])
        declared = set(_DECLARED.findall(body)) | set(_DECLARED.findall(loop[0]))
        outer = sorted(set(_ASSIGNED.findall(body)) - declared)
        if outer:
            found.append((loop[0].strip(), outer))
    return found


def pair(left, right, shape, dtype, seed):
    """Two operands; the sparse ones store nothing in about half their rows."""

    generator = torch.Generator().manual_seed(seed)
    dense = []
    for fmt in (left, right):
        if dtype.is_floating_point:
            values = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
        else:
            values = torch.randint(1, 100, shape, generator=generator)
        if set(fmt) != {"d"}:
            keep = torch.rand(shape, generator=generator, dtype=torch.float64) < 0.5
            rows = torch.rand(shape[0], generator=generator, dtype=torch.float64) < 0.5
            keep = keep & rows.reshape((-1,) + (1,) * (len(shape) - 1))
            values = values * keep
        dense.append(values.to(dtype))
    return dense


def emitted_addition_cpp(monkeypatch, left, right, shape, dtype=torch.float64):
    """The kernel ``A + B`` emits; nothing is built."""

    captured = []

    def stop_before_native(prepared):
        captured.append(prepared.request.cpp_sources[-1])
        raise _Emitted

    monkeypatch.setattr(stensor, "_load_validated_prepared_kernel", stop_before_native)
    a, b = pair(left, right, shape, dtype, 3)
    with pytest.raises(_Emitted):
        stored(a, left, "A") + stored(b, right, "B")
    return captured[0][captured[0].index("evaluate(") :]


def test_the_scan_finds_a_parallel_loop_that_advances_an_outer_position():
    source = (
        "int pC0 = C0_pos[0];\n"
        "#pragma omp parallel for\n"
        "for (int64_t i = 0; i < n; i++) {\n"
        "  int i_C = C0_crd[pC0];\n"
        "  A_values[i] = B_val[i];\n"
        "  pC0 += (int)(i_C == i);\n"
        "}\n"
    )
    assert parallel_loops_assigning_outer_scalars(source) == [
        ("for (int64_t i = 0; i < n; i++) {", ["pC0"])
    ]
    assert parallel_loops_assigning_outer_scalars(source.replace("#pragma", "//")) == []


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pair_
        for formats in (ONE, TWO, THREE)
        for pair_ in itertools.product(formats, repeat=2)
    ],
)
def test_no_parallel_loop_assigns_a_scalar_declared_outside_it(
    monkeypatch, left, right
):
    source = emitted_addition_cpp(monkeypatch, left, right, SMALL[len(left)])
    assert parallel_loops_assigning_outer_scalars(source) == []


@pytest.mark.parametrize(("left", "right"), CARRIED)
def test_the_loop_that_carries_a_position_is_serial(monkeypatch, left, right):
    lines = [
        line.strip()
        for line in emitted_addition_cpp(
            monkeypatch, left, right, SMALL[len(left)]
        ).splitlines()
    ]

    loop = lines.index("for (int64_t i0 = 0; i0 < B0_size; i0++) {")
    assert not lines[loop - 1].startswith("#pragma omp")
    assert any(line.startswith("int pC0 = ") for line in lines[:loop])
    assert any(line.startswith("pC0 ") for line in lines[loop:])


def test_addition_of_two_dense_operands_stays_parallel(monkeypatch):
    lines = [
        line.strip()
        for line in emitted_addition_cpp(monkeypatch, "dd", "dd", SMALL[2]).splitlines()
    ]

    loop = lines.index("for (int64_t i0 = 0; i0 < B0_size; i0++) {")
    assert lines[loop - 1].startswith("#pragma omp parallel for")


@pytest.mark.parametrize("extents", [SMALL, LARGE], ids=["small", "large"])
@pytest.mark.parametrize(
    ("left", "right"),
    list(itertools.product(ONE, repeat=2))
    + list(itertools.product(TWO, repeat=2))
    + [pair_ for pair_ in CARRIED if len(pair_[0]) == 3]
    + [("dss", "ddd"), ("sss", "ddd"), ("dss", "sss"), ("ooo", "ddd")],
)
def test_addition_matches(left, right, extents):
    a, b = pair(left, right, extents[len(left)], torch.float64, 17)

    result = stored(a, left, "A") + stored(b, right, "B")

    assert_matches(result, a + b, torch.float64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize(
    ("left", "right"), CARRIED + [("ds", "ss"), ("ss", "dd"), ("dd", "ds")]
)
def test_addition_over_many_rows_matches_in_other_value_types(left, right, dtype):
    a, b = pair(left, right, LARGE[len(left)], dtype, 19)

    result = stored(a, left, "A") + stored(b, right, "B")

    assert_matches(result, (a + b).to(torch.float64), dtype)
