"""A dense result's positions are resolved once the variables they need are bound.

A dense level's position is its parent's position times its own extent plus its own
coordinate.  The lowering emitted that statement in the loop over the level's own
index variable, which is right while loops bind the result's levels outermost
first.  A result that is dense in every level is not held to that order: the
scheduler's mode-order constraints leave an all-dense tensor out, and public
``einsum`` rebinds dense operands to the selected loop order but not the result.

``einsum('ab,ca->cb')`` with a compressed ``A`` and a dense ``B`` loops ``a, b, c``
and stores its result as ``(c, b)``.  The loop over ``b`` emitted
``int pC1 = pC0 * C1_size + b;`` before the loop over ``c`` declared ``pC0``, and the
kernel did not compile (``'pC0' was not declared in this scope``).  The same happened
to every product of a sparse tensor with a dense matrix whose dense result puts the
matrix's own index ahead of the tensor's: ``abcd,ea->ebcd``, ``abcde,cf->abfde``,
TTMc.
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

# (expression, extents, operand formats).  Every result is requested dense in every
# level: without ``format=`` einsum infers a compressed result for these operands,
# which is a different program (a workspace and an ordered drain).
OUT_OF_ORDER = [
    ("ab,ca->cb", {"a": 3, "b": 4, "c": 2}, ("ss", "dd")),
    ("ab,ac->cb", {"a": 3, "b": 4, "c": 2}, ("ss", "dd")),
    ("abc,da->dbc", {"a": 3, "b": 2, "c": 4, "d": 2}, ("sss", "dd")),
    ("abcd,ea->ebcd", {"a": 3, "b": 2, "c": 4, "d": 2, "e": 3}, ("ssss", "dd")),
    (
        "abcde,cf->abfde",
        {"a": 2, "b": 3, "c": 2, "d": 3, "e": 2, "f": 4},
        ("sssss", "dd"),
    ),
    (
        "abcd,be,cf,dg->aefg",
        {"a": 3, "b": 2, "c": 3, "d": 2, "e": 3, "f": 2, "g": 4},
        ("ssss", "dd", "dd", "ds"),
    ),
]


def lines_of(source):
    return [line.strip() for line in source.splitlines() if line.strip()]


def loop_opened_before(source, position):
    """The nearest ``for`` header above ``position``."""

    return next(
        line for line in reversed(source[:position]) if line.startswith("for (")
    )


def operands(expr, extents, formats, dtype, fill):
    generator = torch.Generator().manual_seed(7)
    dense = []
    for term, fmt in zip(expr.split("->")[0].split(","), formats):
        shape = tuple(extents[index] for index in term)
        if dtype.is_floating_point:
            values = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
        else:
            values = torch.randint(1, 100, shape, generator=generator)
        if "s" in fmt and fill < 1.0:
            keep = torch.rand(shape, generator=generator, dtype=torch.float64) < fill
            values = values * keep
        dense.append(values.to(dtype))
    return dense


def test_position_waits_for_the_loop_that_binds_its_parent(monkeypatch):
    """``ab,ca->cb``: level 1 needs ``c``, which the innermost loop binds."""

    source = lines_of(
        emitted_einsum_cpp(
            monkeypatch,
            "ab,ca->cb",
            ["ss", "dd"],
            {"a": 3, "b": 4, "c": 2},
            format="dd",
        )
    )

    inner_loop = source.index("for (int64_t c = 0; c < B1_size; c++) {")
    level_0 = source.index("int pC0 = c;")
    level_1 = source.index("int pC1 = pC0 * C1_size + b;")
    assert inner_loop < level_0 < level_1
    assert level_1 + 1 == source.index("C_values[pC1] += A_val[pA1] * _B_val_ptr[c];")
    assert sum(line.startswith("int pC") for line in source) == 2


def test_levels_bound_in_storage_order_resolve_one_per_loop(monkeypatch):
    """A copy into a dense result binds ``a, b, c`` in storage order: each loop
    resolves its own level, as before."""

    source = lines_of(
        emitted_einsum_cpp(
            monkeypatch, "abc->abc", ["sss"], {"a": 2, "b": 3, "c": 2}, format="ddd"
        )
    )

    resolved = [
        "int pB0 = a;",
        "int pB1 = pB0 * B1_size + b;",
        "int pB2 = pB1 * B2_size + c;",
    ]
    assert [line for line in source if line.startswith("int pB")] == resolved
    for level, statement in enumerate(resolved):
        loop = loop_opened_before(source, source.index(statement))
        assert loop.startswith(f"for (int pA{level} = ")


def test_a_deeper_level_waits_with_the_level_above_it(monkeypatch):
    """``abcde,cf->abfde``: ``a`` and ``b`` resolve in their own loops, and the
    loop over ``f`` resolves ``f``, ``d`` and ``e`` in level order."""

    source = lines_of(
        emitted_einsum_cpp(
            monkeypatch,
            "abcde,cf->abfde",
            ["sssss", "dd"],
            {"a": 2, "b": 3, "c": 2, "d": 3, "e": 2, "f": 4},
            format="ddddd",
        )
    )

    inner_loop = source.index("for (int64_t f = 0; f < B1_size; f++) {")
    assert source.index("int pC0 = a;") < source.index("int pC1 = pC0 * C1_size + b;")
    assert source.index("int pC1 = pC0 * C1_size + b;") < inner_loop
    level_2 = source.index("int pC2 = pC1 * C2_size + f;")
    assert inner_loop < level_2
    assert source[level_2 + 1] == "int pC3 = pC2 * C3_size + d;"
    assert source[level_2 + 2] == "int pC4 = pC3 * C4_size + e;"


@pytest.mark.parametrize(
    ("dtype", "fill"),
    [(torch.float64, 1.0), (torch.float64, 0.5), (torch.int64, 1.0)],
)
@pytest.mark.parametrize(("expr", "extents", "formats"), OUT_OF_ORDER)
def test_dense_result_visited_out_of_storage_order_matches(
    expr, extents, formats, dtype, fill
):
    dense = operands(expr, extents, formats, dtype, fill)
    tensors = [
        stored(values, fmt, "ABDEFGH"[position])
        for position, (values, fmt) in enumerate(zip(dense, formats))
    ]

    result = scorch.einsum(expr, *tensors, format="d" * len(expr.split("->")[1]))

    assert_matches(result, reference(expr, dense), dtype)


# A dense workspace copied into a dense result whose levels the loops do not bind
# outermost first.  The workspace variable's level is then not the result's last
# level, or is its first, and the copy-out has to address the cell from that level
# down.  ``abc,dc->adb`` wrote ``C_values[pC1]``, a row's position, where a cell's was
# needed; ``abc->c`` and ``ab,cb->ca`` raised ``TypeError`` in the lowering; the last
# four did not build (``'pC2' was not declared``).  A tile is 8 wide, so ``d`` runs
# both under and over one tile.
THROUGH_A_WORKSPACE = [
    ("abc,dc->adb", {"a": 3, "b": 4, "c": 2, "d": 3}, ("sss", "dd")),
    ("abc,dc->adb", {"a": 3, "b": 4, "c": 2, "d": 11}, ("sss", "dd")),
    ("abc->c", {"a": 3, "b": 4, "c": 5}, ("ssd",)),
    ("abc->c", {"a": 3, "b": 4, "c": 11}, ("ddd",)),
    ("abc->c", {"a": 3, "b": 4, "c": 5}, ("dsd",)),
    ("ab,cb->ca", {"a": 5, "b": 4, "c": 3}, ("ss", "dd")),
    ("abc,cd->dab", {"a": 3, "b": 4, "c": 2, "d": 3}, ("sss", "dd")),
    ("ab,bc->cab", {"a": 5, "b": 4, "c": 3}, ("ss", "dd")),
    ("ab,bc->bca", {"a": 5, "b": 4, "c": 3}, ("ds", "dd")),
    ("abc,bd->dac", {"a": 3, "b": 4, "c": 2, "d": 3}, ("sss", "dd")),
    ("ij,kl->ijkl", {"i": 4, "j": 3, "k": 3, "l": 2}, ("sd", "ss")),
    ("ijkl,ka->ijal", {"i": 3, "j": 4, "k": 3, "l": 4, "a": 3}, ("ddss", "sd")),
    (
        "ijklm,ka->ijalm",
        {"i": 3, "j": 4, "k": 2, "l": 2, "m": 4, "a": 3},
        ("ssssd", "dd"),
    ),
    (
        "ijklm,ia,jb,kc->abclm",
        {"i": 3, "j": 2, "k": 2, "l": 4, "m": 2, "a": 3, "b": 3, "c": 4},
        ("sdssd", "ds", "sd", "ds"),
    ),
]


@pytest.mark.parametrize("level_type", ["d", "s"])
@pytest.mark.parametrize(
    ("dtype", "fill"),
    [(torch.float64, 1.0), (torch.float64, 0.5), (torch.int64, 1.0)],
)
@pytest.mark.parametrize(("expr", "extents", "formats"), THROUGH_A_WORKSPACE)
def test_dense_result_through_a_workspace_matches(
    expr, extents, formats, dtype, fill, level_type
):
    dense = operands(expr, extents, formats, dtype, fill)
    tensors = [
        stored(values, fmt, "ABDEFGH"[position])
        for position, (values, fmt) in enumerate(zip(dense, formats))
    ]

    result = scorch.einsum(expr, *tensors, format=level_type * len(expr.split("->")[1]))

    assert_matches(result, reference(expr, dense), dtype)


# An outer product into a dense result.  The loops follow the operands' stored levels,
# so a compressed level of the second operand can be bound before the first operand's:
# ``ij,klm->ijklm`` with ``B`` compressed in ``k`` loops ``k`` outermost and emitted
# ``int pC2 = pC1 * C2_size + k;`` there, before any loop declared ``pC1`` (the layout
# sweep's three outer products; none built).  No workspace is involved.
OUTER_PRODUCTS = [
    ("ij,kl->ijkl", {"i": 4, "j": 3, "k": 3, "l": 2}, ("sd", "ss")),
    ("ij,klm->ijklm", {"i": 4, "j": 3, "k": 4, "l": 2, "m": 4}, ("ds", "sds")),
    ("ij,klm->ijklm", {"i": 9, "j": 5, "k": 6, "l": 3, "m": 7}, ("ds", "sds")),
    (
        "ijk,lmn->ijklmn",
        {"i": 3, "j": 4, "k": 4, "l": 2, "m": 2, "n": 2},
        ("sdd", "dss"),
    ),
]


@pytest.mark.parametrize("level_type", ["d", "s"])
@pytest.mark.parametrize(
    ("dtype", "fill"),
    [(torch.float64, 1.0), (torch.float64, 0.5), (torch.int64, 1.0)],
)
@pytest.mark.parametrize(("expr", "extents", "formats"), OUTER_PRODUCTS)
def test_outer_product_into_a_dense_result_matches(
    expr, extents, formats, dtype, fill, level_type
):
    dense = operands(expr, extents, formats, dtype, fill)
    tensors = [
        stored(values, fmt, "ABDEFGH"[position])
        for position, (values, fmt) in enumerate(zip(dense, formats))
    ]

    result = scorch.einsum(expr, *tensors, format=level_type * len(expr.split("->")[1]))

    assert_matches(result, reference(expr, dense), dtype)
