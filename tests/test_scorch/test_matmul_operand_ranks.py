"""``scorch.matmul``'s generated path follows ``torch.matmul``'s rank rules.

The generated path sent every pair of operands to ``einsum("ij,jk->ik")``, so a vector
first operand (``x @ A``) raised a plain ``CompileSpecError`` ("einsum operand 0 has 2
labels but tensor rank 1") with no code, on 864745d and on the merge candidate.  The
call reaches that path whenever it has a sparse ``STensor`` vector or any keyword (a
format, a schedule); a dense ``torch.Tensor`` vector with no keyword is computed by
``autograd.serve``.  ken's run of the coded-schedule-refusal acceptance found 86 such
cases with a schedule (172 arms).

The expression now comes from a table of operand ranks: ``"j,jk->k"`` for vector x
matrix, ``"j,j->"`` for vector x vector (the dot product, a rank-0 result), and
``"ij,jk->ik"`` as before.  A pair with no entry, a batched
product, is refused with ``MatmulOperandRankError`` (code
``matmul_operand_rank_unsupported``) before any kernel is built.  Results are compared
with the float64 product to 1e-12, on both automatic arms.
"""

import itertools

import pytest
import torch

import scorch
from scorch import ops, stensor, utils
from scorch.compiler.scheduler import Schedule
from scorch.exceptions import CompileSpecError, MatmulOperandRankError
from tests.test_scorch.level_array_support import assert_matches, stored

F64 = torch.float64
MATRIX = ["ss", "ds", "sd", "dd", "oo"]
VECTOR = ["d", "s", "o"]
SHAPES = {"small": (4, 5), "large": (40, 300)}


class _Reached(Exception):
    """Raised by a loop-order sort or a build that a refused call reached."""


def _reached(*args, **kwargs):
    raise _Reached


@pytest.fixture(params=["0", "1"], ids=["regblock-off", "regblock-on"])
def arm(request, monkeypatch):
    monkeypatch.setenv("SCORCH_REGBLOCK", request.param)
    return request.param


def _no_build(monkeypatch):
    for module, name in [
        (ops, "topo_sort_characters"),
        (utils, "_prepare_jit_build"),
        (utils, "_load_validated_prepared_kernel"),
        (ops, "_prepare_jit_build"),
        (ops, "_load_validated_prepared_kernel"),
        (stensor, "_prepare_jit_build"),
        (stensor, "_load_validated_prepared_kernel"),
    ]:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _reached)


def _dense(shape, sparse, seed):
    generator = torch.Generator().manual_seed(seed)
    values = 0.5 + torch.rand(shape, generator=generator, dtype=F64)
    if sparse:
        values = values * (torch.rand(shape, generator=generator, dtype=F64) < 0.4)
    return values


@pytest.mark.parametrize("ranks", sorted(ops._MATMUL_EXPRESSIONS))
def test_each_expression_is_the_torch_product_of_its_ranks(ranks):
    shapes = {(2, 2): [(3, 4), (4, 5)], (1, 2): [(4,), (4, 5)]}
    shapes[(2, 1)] = [(3, 4), (4,)]
    shapes[(1, 1)] = [(4,), (4,)]
    left, right = (_dense(shape, False, 3 + n) for n, shape in enumerate(shapes[ranks]))
    expected = torch.matmul(left, right)
    got = torch.einsum(ops._MATMUL_EXPRESSIONS[ranks], left, right)
    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("size", ["small", "large"])
@pytest.mark.parametrize(
    ("vector_format", "matrix_format", "result_format"),
    [
        combination
        for combination in itertools.product(VECTOR, MATRIX, ["d", "s"])
        if combination[:2] != ("d", "dd")
    ],
)
def test_vector_times_matrix_matches(
    arm, vector_format, matrix_format, result_format, size
):
    rows, cols = SHAPES[size]
    x = _dense((rows,), vector_format != "d", 11)
    a = _dense((rows, cols), set(matrix_format) != {"d"}, 12)
    expected = x @ a
    assert int((expected != 0).sum()) > 0
    result = scorch.matmul(
        stored(x, vector_format, "x"),
        stored(a, matrix_format, "A"),
        format=result_format,
    )
    assert_matches(result, expected, F64)


@pytest.mark.parametrize("matrix_format", ["ds", "ss"])
def test_a_dense_torch_vector_times_a_sparse_matrix_with_a_keyword(arm, matrix_format):
    x = _dense((40,), False, 21)
    a = _dense((40, 30), True, 22)
    operand = stored(a, matrix_format, "A")
    assert_matches(scorch.matmul(x, operand, format="d"), x @ a, F64)
    # With no keyword the product is computed by autograd.serve, as before.
    assert_matches(scorch.matmul(x, stored(a, matrix_format, "A")), x @ a, F64)


@pytest.mark.parametrize("vector_format", ["d", "s"])
def test_vector_times_matrix_with_a_schedule(arm, vector_format):
    x = _dense((40,), vector_format != "d", 31)
    a = _dense((40, 30), True, 32)
    result = scorch.matmul(
        stored(x, vector_format, "x"),
        stored(a, "ds", "A"),
        format="d",
        schedule=Schedule(),
    )
    assert_matches(result, x @ a, F64)


@pytest.mark.parametrize("formats", [("s", "s"), ("d", "s"), ("s", "o")])
def test_vector_dot_vector_matches_torch_dot(arm, formats):
    left = _dense((40,), True, 41)
    right = _dense((40,), True, 42)
    expected = torch.dot(left, right)
    assert expected != 0
    x = stored(left, formats[0], "x")
    y = stored(right, formats[1], "y")
    for kwargs in ({}, {"format": ""}, {"schedule": Schedule()}):
        result = scorch.matmul(x, y, **kwargs)
        assert tuple(result.shape) == ()
        assert_matches(result, expected, F64)
    # The result has rank 0, so a rank-1 output format is refused by einsum's check.
    with pytest.raises(CompileSpecError, match="does not match result rank 0"):
        scorch.matmul(x, y, format="d")


BATCHED = [
    ((2, 3, 4), "sss", (4, 5), "ds"),
    ((3, 4), "ds", (2, 4, 5), "sss"),
    ((2, 3, 4), "sss", (2, 4, 5), "sss"),
    ((2, 3, 4), "ooo", (4,), "s"),
    ((4,), "s", (2, 4, 5), "dss"),
]


@pytest.mark.parametrize(
    ("left_shape", "left_format", "right_shape", "right_format"), BATCHED
)
def test_a_batched_product_is_refused_by_code_before_any_build(
    arm, monkeypatch, left_shape, left_format, right_shape, right_format
):
    left = stored(_dense(left_shape, True, 51), left_format, "A")
    right = stored(_dense(right_shape, True, 52), right_format, "B")
    _no_build(monkeypatch)
    for kwargs in ({}, {"format": "d"}, {"schedule": Schedule()}):
        with pytest.raises(MatmulOperandRankError) as raised:
            scorch.matmul(left, right, **kwargs)
        assert raised.value.code == "matmul_operand_rank_unsupported"
        assert isinstance(raised.value, CompileSpecError)
        assert raised.value.ranks == (len(left_shape), len(right_shape))


def test_dense_batched_operands_still_go_to_torch():
    left = _dense((2, 3, 4), False, 61)
    right = _dense((2, 4, 5), False, 62)
    result = scorch.matmul(stored(left, "ddd", "A"), stored(right, "ddd", "B"))
    torch.testing.assert_close(result, torch.matmul(left, right), rtol=1e-12, atol=0.0)
