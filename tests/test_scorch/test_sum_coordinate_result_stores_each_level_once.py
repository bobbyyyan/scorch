"""A sum into an all-coordinate result of rank three or more stores each level once.

A coordinate result stores, for each value, one coordinate per level.  The lowering
emits them as stores at the value's position, ``A2_crd[pA2] = i2; A1_crd[pA2] = i1;
A0_crd[pA2] = i0;``, and a later pass turns each store into an append.  The levels to
store were read off the lowering's list of loop variables, innermost first.  That list
only grows, and a sum lowers the loops under a merge once per branch of the merge, so
under the second branch of the outermost merge the list read ``i0, i1, i2, i2, i1, i2``
and the stores were ``A2, A1, A2, A1, A0``: two coordinates appended to levels 1 and 2
for one value.  Rank two produces only adjacent repeats (``i0, i1, i1``), which the
append pass already drops, so every matrix sum was right and every sum of rank three
or more into a coordinate result returned level arrays of different lengths
(``TensorStorageError: COO mode 1 has 90 coordinates, but values has 60 elements``).

Public ``A + B`` stores its result in the left operand's format, so this is every
``A + B`` with a coordinate ``A`` of rank three or more.  On the developer entry
(``lower_and_exec_cin``) it is every sum declared with such a result.

Each loop variable is now stored once (``CINLowerer.lower_TensorAssign``).

Operands are built and results read as level arrays in PyTorch.  An executed test
compares the result's stored coordinates exactly with the cells either operand
stores, in lexicographic order, and each stored value with the sum there: float64 to
1e-12, float32 to 1e-5, int64 exactly.
"""

import itertools
import re

import pytest
import torch

from scorch.ops import lower_and_exec_cin
from scorch.stensor import STensor
from tests.test_scorch import sum_cin_support
from tests.test_scorch.level_array_support import (
    assert_matches,
    coordinate_entries,
    level_arrays,
    logical_dense,
    stored,
)
from tests.test_scorch.test_addition_carried_position import (
    emitted_addition_cpp,
    pair,
)

ONE = ["d", "s", "o"]
TWO = ["dd", "ds", "sd", "ss", "oo"]
THREE = ["ddd", "dds", "dsd", "dss", "sdd", "sds", "ssd", "sss", "ooo"]
FOUR = ["dddd", "ssss", "dsds", "sdsd", "oooo"]
SMALL = {1: (7,), 2: (5, 4), 3: (4, 3, 5), 4: (3, 4, 2, 3)}
# 16 rows per worker: several workers wherever a loop is parallel.
LARGE = {3: (200, 5, 4), 4: (70, 3, 4, 3)}


@pytest.fixture(autouse=True)
def _torch_allows_the_teams_these_shapes_were_built_for(torch_threads):
    # LARGE gives up to 200 // 16 = 12 workers, and sum_cin_support.LARGE (300 rows) 18.
    # torch's count must allow that, or a process started with MKL_NUM_THREADS=1 runs every kernel
    # here on one worker (tests/conftest.py, torch_threads).
    torch_threads(18, needs_team=False)


# Sums on the developer entry: (operand formats, indices, coordinate result format).
DEVELOPER_ENTRY = [
    (formats, "ijk", "ooo")
    for formats in sum_cin_support.RANK3
    + [("ooo", "ooo"), ("ooo", "ddd"), ("ddd", "ooo"), ("sss", "ooo")]
] + [
    (formats, "ij", "oo")
    for formats in sum_cin_support.RANK2 + [("oo", "oo"), ("oo", "dd")]
]


def cells_stored(dense, fmt):
    """Which cells ``dense`` held in ``fmt`` stores, read back from its level arrays."""

    mode_indices, values = level_arrays(dense, fmt)
    ones = STensor.from_components(
        tuple(dense.shape),
        fmt,
        mode_indices,
        torch.ones(values.shape, dtype=torch.float64),
        name="M",
    )
    return logical_dense(ones) != 0


def assert_stores_the_union(result, a, b, left, right, dtype):
    """The coordinate ``result`` stores exactly the cells ``a`` in ``left`` or ``b`` in
    ``right`` stores, in lexicographic order, each with the sum there."""

    coordinates, values = coordinate_entries(result)
    expected = (cells_stored(a, left) | cells_stored(b, right)).nonzero()
    assert torch.equal(coordinates, expected)
    total = (a.to(torch.float64) + b.to(torch.float64))[tuple(expected.t())]
    if not dtype.is_floating_point:
        assert torch.equal(values.to(torch.float64), total)
        return
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    assert torch.allclose(values.to(torch.float64), total, rtol=tolerance, atol=0.0)


def unbalanced_append_runs(source, result):
    """Runs of appends to ``result`` that do not store one entry in every vector.

    A run is a maximal sequence of statements that append to the result's coordinate
    or value vectors or advance its position; comments and blank lines do not end one.
    Each run is returned as ``{vector: appends}``.
    """

    append = re.compile(rf"^\s*{result}(\d+_crd|_values)\.emplace_back\(")
    advance = re.compile(rf"^\s*p{result}\d+\+\+;")
    found, run = [], {}
    for line in source.splitlines() + ["}"]:
        matched = append.match(line)
        if matched:
            run[matched.group(1)] = run.get(matched.group(1), 0) + 1
        elif advance.match(line) or not line.strip() or line.strip().startswith("//"):
            continue
        else:
            if run and len(set(run.values())) > 1:
                found.append(run)
            run = {}
    return found


def test_the_scan_finds_a_level_appended_twice_for_one_value():
    source = (
        "A_values.emplace_back(x);\n"
        "// Set coordinates\n"
        "A2_crd.emplace_back(k);\n"
        "A1_crd.emplace_back(j);\n"
        "A2_crd.emplace_back(k);\n"
        "A1_crd.emplace_back(j);\n"
        "A0_crd.emplace_back(i);\n"
        "pA2++;\n"
    )
    assert unbalanced_append_runs(source, "A") == [
        {"_values": 1, "2_crd": 2, "1_crd": 2, "0_crd": 1}
    ]
    once = source.replace("A2_crd.emplace_back(k);\nA1_crd.emplace_back(j);\n", "", 1)
    assert unbalanced_append_runs(once, "A") == []
    assert unbalanced_append_runs(source, "B") == []


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pair_
        for formats in (ONE, TWO, THREE, FOUR)
        for pair_ in itertools.product(formats, repeat=2)
    ],
)
def test_addition_appends_one_coordinate_per_level_for_each_value(
    monkeypatch, left, right
):
    source = emitted_addition_cpp(monkeypatch, left, right, SMALL[len(left)])
    assert unbalanced_append_runs(source, "A") == []
    if set(left) == {"o"}:
        # The scan read something: a coordinate result is assembled by appends.
        assert "A_values.emplace_back(" in source
        assert f"A{len(left) - 1}_crd.emplace_back(" in source


@pytest.mark.parametrize(("formats", "indices", "result_format"), DEVELOPER_ENTRY)
def test_developer_entry_sum_appends_one_coordinate_per_level_for_each_value(
    formats, indices, result_format
):
    source = "\n".join(
        sum_cin_support.emitted(formats, indices, indices, result_format)
    )
    assert unbalanced_append_runs(source, "C") == []
    assert "C_values.emplace_back(" in source


def test_second_branch_of_the_outer_merge_stores_three_coordinates(monkeypatch):
    """``ooo + ddd``: where ``A`` stores nothing in a slice, the dense operand's cells
    are appended with one coordinate each for ``i0``, ``i1`` and ``i2``."""

    lines = [
        line.strip()
        for line in emitted_addition_cpp(
            monkeypatch, "ooo", "ddd", SMALL[3]
        ).splitlines()
        if line.strip() and not line.strip().startswith("//")
    ]

    appended = [at for at, line in enumerate(lines) if line == "pA2++;"]
    assert len(appended) == 4
    for at in appended:
        assert lines[at - 3 : at] == [
            "A2_crd.emplace_back(i2);",
            "A1_crd.emplace_back(i1);",
            "A0_crd.emplace_back(i0);",
        ]
        assert lines[at - 4].startswith("A_values.emplace_back(")


@pytest.mark.parametrize("extents", [SMALL, LARGE], ids=["small", "large"])
@pytest.mark.parametrize("right", THREE)
def test_rank_three_coordinate_addition_matches(right, extents):
    a, b = pair("ooo", right, extents[3], torch.float64, 23)

    result = stored(a, "ooo", "A") + stored(b, right, "B")

    assert_stores_the_union(result, a, b, "ooo", right, torch.float64)
    assert_matches(result, a + b, torch.float64)


@pytest.mark.parametrize("extents", [SMALL, LARGE], ids=["small", "large"])
@pytest.mark.parametrize("right", ["dddd", "ssss", "oooo"])
def test_rank_four_coordinate_addition_matches(right, extents):
    a, b = pair("oooo", right, extents[4], torch.float64, 29)

    result = stored(a, "oooo", "A") + stored(b, right, "B")

    assert_stores_the_union(result, a, b, "oooo", right, torch.float64)
    assert_matches(result, a + b, torch.float64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("right", ["ddd", "sss", "ooo"])
def test_rank_three_coordinate_addition_matches_in_other_value_types(right, dtype):
    a, b = pair("ooo", right, LARGE[3], dtype, 31)

    result = stored(a, "ooo", "A") + stored(b, right, "B")

    assert_stores_the_union(result, a, b, "ooo", right, dtype)
    assert_matches(result, (a + b).to(torch.float64), dtype)


@pytest.mark.parametrize("extents", ["small", "large"])
@pytest.mark.parametrize(
    "formats", [("sss", "sss"), ("ddd", "sss"), ("ooo", "ooo"), ("sss", "ooo")]
)
def test_developer_entry_sum_into_a_rank_three_coordinate_result_matches(
    formats, extents
):
    sizes = sum_cin_support.SMALL if extents == "small" else sum_cin_support.LARGE
    shape = tuple(sizes[name] for name in "ijk")
    a, b = sum_cin_support.operands(formats, shape, torch.float64, 37)

    result = lower_and_exec_cin(
        sum_cin_support.sum_cin(formats, "ijk", "ijk", "ooo", torch.float64),
        shape,
        stored(a, formats[0], "A"),
        stored(b, formats[1], "B"),
    )

    assert_stores_the_union(result, a, b, formats[0], formats[1], torch.float64)
    assert_matches(result, a + b, torch.float64)
