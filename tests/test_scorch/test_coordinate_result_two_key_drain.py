"""A sparse workspace keyed by two loop variables, drained into a coordinate result.

``einsum('abc,bd->adc', format='ooo')`` accumulates each ``a`` slice in a workspace
keyed by ``(d, c)`` and drains it, sorted, into the result.  A coordinate ("o") level
stores one coordinate per entry, so each drained key stores ``c`` in level 2, ``d`` in
level 1 and the enclosing loop's ``a`` in level 0.

The drain was written for a key of one variable.  With two it stored the key's *first*
variable in the last level and again in the level above, and nothing in level 0: the
result's level arrays had different lengths.  Since the drain learned to store every
coordinate level it has refused such a program
(``unsupported_coordinate_drain_unbound_level``), because the level above the last is
not an enclosing loop's variable; it is the key's.  That refused 44 of the 9,061
programs of the generated einsum population, 41 of them layouts of ``abc,bd->adc``.

The drain now stores each key variable in its own level and takes the levels above the
key from the enclosing loops.  A sorted drain stores a coordinate result in order only
when the key's variables are the result's last levels, in level order; any other key is
refused with ``unsupported_coordinate_drain_key_order``.

Operands are built and results read as level arrays in PyTorch.  An executed test
checks float64 to 1e-12 of each cell, with the result's entries in range, in
lexicographic order and stored once.
"""

import itertools
import re

import pytest
import torch

import scorch
from scorch.compiler.cin import IndexVar, TensorAssign, TensorVar, Workspace
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.diagnostics import UnsupportedFeature
from tests.test_scorch.level_array_support import (
    assert_matches,
    emitted_einsum_cpp,
    reference,
    stored,
)
from tests.test_scorch.test_sum_coordinate_result_stores_each_level_once import (
    unbalanced_append_runs,
)

THREE = ["".join(levels) for levels in itertools.product("ds", repeat=3)] + ["ooo"]
TWO = ["".join(levels) for levels in itertools.product("ds", repeat=2)] + ["oo"]
SMALL = {"a": 3, "b": 4, "c": 2, "d": 3}
# 16 slices per worker of the outermost result index, whichever it is.
LARGE = {"a": 40, "b": 5, "c": 4, "d": 40}

# The other two expressions of the population with a two-variable key.
OTHER = [
    ("abc,ad->dbc", ("sdd", "dd")),
    ("abc,ad->dbc", ("sds", "dd")),
    ("abc,cd->abd", ("ddd", "sd")),
]
EVERY_LAYOUT = [("abc,bd->adc", formats) for formats in itertools.product(THREE, TWO)]
LARGE_LAYOUTS = [
    ("abc,bd->adc", formats)
    for formats in [
        ("sss", "ss"),
        ("ddd", "dd"),
        ("ooo", "oo"),
        ("dss", "ds"),
        ("sdd", "sd"),
        ("ssd", "dd"),
        ("dds", "ss"),
        ("ooo", "dd"),
        ("sss", "oo"),
    ]
]


def operand(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    keep = torch.rand(shape, generator=generator) < 0.5
    return (0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)) * keep


def drain_appends(source):
    """``{level: value appended}`` for the result's coordinates in the drain loop."""

    lines = [line.strip() for line in source.splitlines()]
    start = next(
        at for at, line in enumerate(lines) if line.startswith("for (const auto& it")
    )
    appended = {}
    for line in lines[start:]:
        matched = re.fullmatch(r"C(\d)_crd\.emplace_back\((\w+)\);", line)
        if matched:
            assert int(matched.group(1)) not in appended
            appended[int(matched.group(1))] = matched.group(2)
        elif line == "pC2++;":
            break
    return appended


@pytest.mark.parametrize(("expr", "formats"), EVERY_LAYOUT + OTHER)
def test_the_drain_stores_each_result_index_in_its_level(monkeypatch, expr, formats):
    source = emitted_einsum_cpp(monkeypatch, expr, formats, SMALL, format="ooo")

    assert unbalanced_append_runs(source, "C") == []
    # Two layouts (a dense ``abc`` with a compressed or coordinate ``bd``) key the
    # workspace by the whole result coordinate and drain it by another loop.
    if "wksp_value" in source:
        result_indices = expr.split("->")[1]
        assert drain_appends(source) == dict(enumerate(result_indices))
        assert "C_values.emplace_back(wksp_value);" in source


def test_a_two_variable_key_is_drained_by_these_programs(monkeypatch):
    """The scan above reads a drain: the programs it is about have one."""

    drained = [
        (expr, formats)
        for expr, formats in EVERY_LAYOUT + OTHER
        if "coo_workspace<double, 2>"
        in emitted_einsum_cpp(monkeypatch, expr, formats, SMALL, format="ooo")
    ]
    # 42 layouts of ``abc,bd->adc`` and the three others; 44 of the 45 were refused.
    assert len(drained) == 45
    assert set(OTHER) <= set(drained)


def consumer(result_format, result_names, key_names):
    ivars = {name: IndexVar(name) for name in "ijk"}
    result = TensorVar("Result", fmt=result_format, dtype=torch.float32)
    workspace = Workspace("wksp", dim=len(key_names), dtype=torch.float32)
    statement = TensorAssign(
        result[tuple(ivars[name] for name in result_names)],
        workspace[tuple(ivars[name] for name in key_names)],
    )
    lowerer = CINLowerer()
    lowerer.outermost_stmt = statement
    return LLIRLowerer().lower_llir(lowerer.lower_ConsumerIndexStmt(statement))


def test_a_key_in_the_results_level_order_is_drained():
    source = consumer("oo", "ij", "ij")
    assert "Result1_crd[pResult1] = j;" in source
    assert "Result0_crd[pResult1] = i;" in source


@pytest.mark.parametrize(
    ("result_format", "result_names", "key_names"),
    [("oo", "ij", "ji"), ("ooo", "ijk", "kj"), ("ooo", "ijk", "ik")],
)
def test_a_key_in_another_order_is_refused(result_format, result_names, key_names):
    with pytest.raises(UnsupportedFeature) as refusal:
        consumer(result_format, result_names, key_names)

    (diagnostic,) = refusal.value.diagnostics
    assert diagnostic.code == "unsupported_coordinate_drain_key_order"


def assert_einsum_matches(expr, formats, extents):
    terms = expr.split("->")[0].split(",")
    dense = [
        operand(tuple(extents[name] for name in term), 90 + position)
        for position, term in enumerate(terms)
    ]
    operands = [
        stored(value, fmt, "AB"[position])
        for position, (value, fmt) in enumerate(zip(dense, formats))
    ]

    result = scorch.einsum(expr, *operands, format="ooo")

    assert_matches(result, reference(expr, dense), torch.float64)


@pytest.mark.parametrize(("expr", "formats"), EVERY_LAYOUT + OTHER)
def test_two_key_drain_matches(expr, formats):
    assert_einsum_matches(expr, formats, SMALL)


@pytest.mark.parametrize(("expr", "formats"), LARGE_LAYOUTS + OTHER)
def test_two_key_drain_matches_at_a_larger_extent(expr, formats):
    assert_einsum_matches(expr, formats, LARGE)
