"""A coordinate workspace is declared for whatever rank its key has.

``llir.DataType`` listed ``coo_workspace<T, N>`` for ranks 1 to 5 only, and the
lowering looks the type up by its spelling.  A program that keys a coordinate
workspace by six index variables -- a reduction over a rank-7 operand into a rank-6
compressed result, or the reordering of a rank-6 operand's levels that
``einsum('abcdef->a')`` asks for -- raised ``ValueError: 'coo_workspace<double, 6>'
is not a valid DataType`` out of the lowering.  The C++ ``coo_workspace<T, N>`` is a
template over ``N`` and never had the limit.
"""

import copy
import pickle

import pytest
import torch

import scorch
from scorch.compiler import llir
from tests.test_scorch.level_array_support import (
    assert_matches,
    emitted_einsum_cpp,
    reference,
    stored,
)

SCALARS = (
    llir.DataType.INT,
    llir.DataType.FLOAT32,
    llir.DataType.FLOAT64,
    llir.DataType.INT32,
    llir.DataType.INT64,
    llir.DataType.INT8,
    llir.DataType.UINT8,
)


@pytest.mark.parametrize("rank", (6, 7, 12))
@pytest.mark.parametrize("scalar", SCALARS, ids=lambda scalar: scalar.name)
def test_a_rank_above_the_listed_ones_is_one_member_per_spelling(scalar, rank):
    spelling = f"coo_workspace<{scalar.value}, {rank}>"

    member = llir.DataType.coo_workspace_type_with_dim(scalar, rank)

    assert type(member) is llir.DataType
    assert member.value == spelling
    assert member.name == f"COO_WORKSPACE_{scalar.name}_{rank}"
    assert llir.DataType.coo_workspace_type_with_dim(scalar, rank) is member
    assert llir.DataType(spelling) is member
    assert pickle.loads(pickle.dumps(member)) is member
    assert copy.deepcopy(member) is member


def test_the_listed_ranks_are_the_listed_members():
    listed = len(llir.DataType)

    for scalar in SCALARS:
        for rank in range(1, 6):
            assert (
                llir.DataType.coo_workspace_type_with_dim(scalar, rank)
                is llir.DataType[f"COO_WORKSPACE_{scalar.name}_{rank}"]
            )
        llir.DataType.coo_workspace_type_with_dim(scalar, 6)

    # A higher rank is reachable by its spelling and does not grow the table.
    assert len(llir.DataType) == listed
    assert "COO_WORKSPACE_FLOAT64_6" not in llir.DataType.__members__


@pytest.mark.parametrize(
    "spelling",
    (
        "coo_workspace<double, 0>",
        "coo_workspace<double, -6>",
        "coo_workspace<double, 06>",
        "coo_workspace<double, 6.0>",
        "coo_workspace<double,6>",
        "coo_workspace<double, 6> ",
        " coo_workspace<double, 6>",
        "coo_workspace<double, 6",
        "coo_workspace<double, 6>>",
        "coo_workspace<double, 6>; int x",
        "coo_workspace<double, N>",
        "coo_workspace<bool, 6>",
        "coo_workspace<std::string, 6>",
        "coo_workspace<torch::kFloat64, 6>",
        "coo_workspace_1d<double, 6>",
        "std::vector<coo_workspace<double, 6>>",
    ),
)
def test_text_that_is_not_a_workspace_type_is_still_refused(spelling):
    with pytest.raises(ValueError, match="is not a valid DataType"):
        llir.DataType(spelling)


@pytest.mark.parametrize("value", (6, 6.0, None, b"coo_workspace<double, 6>", ("a",)))
def test_a_value_that_is_not_text_is_still_refused(value):
    with pytest.raises(ValueError, match="is not a valid DataType"):
        llir.DataType(value)


@pytest.mark.parametrize("scalar", (llir.DataType.BOOL, llir.DataType.STRING))
def test_a_scalar_with_no_workspace_is_still_refused_at_every_rank(scalar):
    for rank in (1, 5, 6):
        with pytest.raises(ValueError, match="is not a valid DataType"):
            llir.DataType.coo_workspace_type_with_dim(scalar, rank)


def test_six_key_workspace_is_declared(monkeypatch):
    """``gabcdef->abcdef``: the reduced index is the operand's outermost level, so
    the sum accumulates into a workspace keyed by the six result indices."""

    extents = {name: 2 for name in "abcdefg"}

    source = emitted_einsum_cpp(
        monkeypatch, "gabcdef->abcdef", ["sssssss"], extents, format="ssssss"
    )

    assert "coo_workspace<double, 6> wksp = coo_workspace<double, 6>(1024, " in source


# (expression, operand format, result format, operand shape).  The first two key a
# six-coordinate workspace in the kernel that reorders the operand's levels, the
# third in its own kernel.
RANK_SIX = [
    ("abcdef->a", "ssssss", "s", (3, 2, 2, 2, 2, 2)),
    ("abcdef->bce", "ssssss", "ssd", (2, 3, 2, 2, 3, 2)),
    ("gabcdef->abcdef", "sssssss", "ssssss", (2, 2, 3, 2, 2, 2, 2)),
]


@pytest.mark.parametrize(
    ("dtype", "fill"),
    [(torch.float64, 1.0), (torch.float64, 0.5), (torch.int64, 0.5)],
)
@pytest.mark.parametrize(("expr", "operand_format", "result_format", "shape"), RANK_SIX)
def test_program_that_keys_a_workspace_by_six_indices_matches(
    expr, operand_format, result_format, shape, dtype, fill
):
    generator = torch.Generator().manual_seed(11)
    if dtype.is_floating_point:
        values = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
    else:
        values = torch.randint(1, 100, shape, generator=generator)
    if fill < 1.0:
        values = values * (torch.rand(shape, generator=generator) < fill)
    dense = [values.to(dtype)]

    result = scorch.einsum(
        expr, stored(dense[0], operand_format, "A"), format=result_format
    )

    assert_matches(result, reference(expr, dense), dtype)
