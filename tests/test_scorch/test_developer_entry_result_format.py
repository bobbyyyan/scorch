"""A sparse result from the developer entry comes back in its declared format.

``lower_and_exec_cin`` wrapped a dense result in the format and mode order of its
declaration and any other result as a two-level dense tensor, which no sparse kernel's
level arrays satisfy: every sum into a compressed result raised ``TensorIndexError``
after its kernel had run.  The result is now wrapped as declared, as ``einsum`` and
``STensor.__add__`` wrap theirs.
"""

import pytest
import torch

from tests.test_scorch.sum_cin_support import (
    LARGE,
    NO_REDUCTION,
    SMALL,
    run,
)


@pytest.mark.parametrize("extents", [SMALL, LARGE], ids=["small", "large"])
@pytest.mark.parametrize(
    ("formats", "indices", "result_indices", "result_format"), NO_REDUCTION
)
def test_a_sum_matches_in_every_result_format(
    formats, indices, result_indices, result_format, extents
):
    run(formats, indices, result_indices, result_format, extents, torch.float64, 29)


@pytest.mark.parametrize(
    ("formats", "indices", "result_indices", "result_format"),
    [
        (("ss", "ds"), "ij", "ij", "ss"),
        (("sss", "sss"), "ijk", "ijk", "sss"),
    ],
)
def test_a_sum_over_many_rows_matches_in_float32(
    formats, indices, result_indices, result_format
):
    run(formats, indices, result_indices, result_format, LARGE, torch.float32, 31)
