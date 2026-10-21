"""``matmul(A, x, format=...)`` for every layout of a matrix, a vector and the result.

A 2-D ``A`` times a 1-D ``x`` takes a prebuilt kernel when one matches (canonical CSR
with a dense vector) and otherwise the generic matrix-vector kernel (``ops.spmv``): one
loop over rows with a scalar accumulator, lowered from its own CIN rather than through
``einsum``.  That is 45 generated kernels -- five matrix layouts, three vector layouts,
three result layouts -- and the existing tests run a handful of them at a few rows.

Each runs here at 5 rows and at 300 (one worker per 16 rows, so several workers where
the row loop is parallel), with rows and vector entries that store nothing.  float64 to
1e-12 of each cell; a sparse result's level arrays are checked as they are decoded.
"""

import itertools

import pytest
import torch

import scorch
from tests.test_scorch.level_array_support import assert_matches, stored

MATRIX = ["ss", "ds", "sd", "dd", "oo"]
VECTOR = ["d", "s", "o"]
SHAPES = {"small": (5, 4), "large": (300, 40)}


def _operands(shape, matrix_format, vector_format, seed):
    generator = torch.Generator().manual_seed(seed)
    a = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
    x = 0.5 + torch.rand(shape[1:], generator=generator, dtype=torch.float64)
    if set(matrix_format) != {"d"}:
        keep = torch.rand(shape, generator=generator, dtype=torch.float64) < 0.3
        rows = torch.rand(shape[0], generator=generator, dtype=torch.float64) < 0.6
        a = a * (keep & rows.reshape(-1, 1))
    if vector_format != "d":
        x = x * (torch.rand(shape[1:], generator=generator, dtype=torch.float64) < 0.5)
    return a, x


@pytest.mark.parametrize("size", ["small", "large"])
@pytest.mark.parametrize(
    ("matrix_format", "vector_format", "result_format"),
    list(itertools.product(MATRIX, VECTOR, VECTOR)),
)
def test_matrix_times_vector_matches(matrix_format, vector_format, result_format, size):
    a, x = _operands(SHAPES[size], matrix_format, vector_format, 47)
    expected = a @ x
    assert int((expected != 0).sum()) > 0

    result = scorch.matmul(
        stored(a, matrix_format, "A"),
        stored(x, vector_format, "x"),
        format=result_format,
    )

    assert_matches(result, expected, torch.float64)
