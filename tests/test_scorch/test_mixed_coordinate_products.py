"""Products of a coordinate (COO) operand with a compressed one.

Output-format inference copies each output index's level type from the inputs, so a
product mixing a COO operand with a CSR/DCSR one used to infer an output that mixes a
coordinate level with dense or compressed levels -- CSR x COO gave dense rows over
coordinate columns. Code generation cannot lower such a tensor: the merged tree refused
the product (`unsupported_mixed_coordinate_hierarchy` for the result), and the
performance line returned it with the per-row grouping missing, so the result could not
be interpreted and its own `to_dense` failed to compile.

Inference now substitutes compressed levels for coordinate ones when the output would
mix them. These tests pin that the products compute the right values against a torch
reference (atol=rtol=1e-3, the suite's convention), that the result is usable as an
operand downstream, and that an explicitly requested mixed format is still refused.
"""

import pytest
import torch

import scorch
from scorch import STensor, einsum
from scorch.compiler.diagnostics import UnsupportedFeature
from scorch.format import LevelFormat, LevelType
from scorch.ops import _without_mixed_coordinate_levels

ATOL = 1e-3
RTOL = 1e-3

MIXED_PAIRS = [("ds", "oo"), ("ss", "oo"), ("oo", "ds"), ("oo", "ss")]
SHAPES = [(64, 64, 64), (48, 40, 32)]
DTYPES = [torch.float32, torch.float64]


def _sparse(m, n, density, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    values = torch.rand(m, n, generator=g, dtype=dtype)
    return values * (torch.rand(m, n, generator=g) < density).to(dtype)


def _dense(result):
    return result.to_torch() if isinstance(result, STensor) else result


def _levels(*types):
    return [LevelFormat(t) for t in types]


def _types(level_formats):
    return [lf.get_level_type() for lf in level_formats]


D, C, O = LevelType.DENSE, LevelType.COMPRESSED, LevelType.COORDINATE


@pytest.mark.parametrize(
    ("inferred", "expected"),
    [
        ((D, O), (D, C)),
        ((C, O), (C, C)),
        ((O, C), (C, C)),
        ((D, C, O), (D, C, C)),
        ((O, O), (O, O)),  # all-coordinate COO is coherent and kept
        ((D, C), (D, C)),  # coordinate-free is untouched
        ((D, D), (D, D)),
        ((O,), (O,)),
    ],
)
def test_only_a_mixed_coordinate_output_is_rewritten(inferred, expected):
    assert _types(_without_mixed_coordinate_levels(_levels(*inferred))) == list(
        expected
    )


def test_a_coherent_output_is_returned_unchanged():
    levels = _levels(O, O)
    assert _without_mixed_coordinate_levels(levels) is levels


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("m", "k", "n"), SHAPES)
@pytest.mark.parametrize(("a_format", "b_format"), MIXED_PAIRS)
def test_mixed_coordinate_products_compute_the_product(
    a_format, b_format, m, k, n, dtype
):
    a = _sparse(m, k, 0.1, 1, dtype)
    b = _sparse(k, n, 0.1, 2, dtype)
    result = scorch.matmul(
        scorch.from_torch(a, "A").to_sparse(a_format),
        scorch.from_torch(b, "B").to_sparse(b_format),
    )
    assert isinstance(result, STensor)
    level_types = result.format.get_level_types()
    assert LevelType.COORDINATE not in level_types or all(
        t is LevelType.COORDINATE for t in level_types
    ), level_types
    assert torch.allclose(_dense(result), a @ b, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("a_format", "b_format"), MIXED_PAIRS)
def test_the_result_is_usable_as_an_operand(a_format, b_format, dtype):
    a = _sparse(48, 40, 0.1, 3, dtype)
    b = _sparse(40, 32, 0.1, 4, dtype)
    x = torch.rand(32, 8, generator=torch.Generator().manual_seed(5), dtype=dtype)
    product = scorch.matmul(
        scorch.from_torch(a, "A").to_sparse(a_format),
        scorch.from_torch(b, "B").to_sparse(b_format),
    )
    downstream = scorch.matmul(product, x)
    assert torch.allclose(_dense(downstream), (a @ b) @ x, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(("a_format", "b_format"), MIXED_PAIRS)
def test_einsum_takes_the_same_inference(a_format, b_format):
    a = _sparse(30, 20, 0.15, 6, torch.float32)
    b = _sparse(20, 25, 0.15, 7, torch.float32)
    result = einsum(
        "ij,jk->ik",
        scorch.from_torch(a, "A").to_sparse(a_format),
        scorch.from_torch(b, "B").to_sparse(b_format),
    )
    assert torch.allclose(_dense(result), a @ b, atol=ATOL, rtol=RTOL)


def test_an_explicitly_requested_mixed_format_is_still_refused():
    a = _sparse(16, 12, 0.2, 8, torch.float32)
    b = _sparse(12, 10, 0.2, 9, torch.float32)
    with pytest.raises(UnsupportedFeature) as error:
        scorch.matmul(
            scorch.from_torch(a, "A").to_sparse("ds"),
            scorch.from_torch(b, "B").to_sparse("oo"),
            format="do",
        )
    assert error.value.diagnostics[0].code == "unsupported_mixed_coordinate_hierarchy"
