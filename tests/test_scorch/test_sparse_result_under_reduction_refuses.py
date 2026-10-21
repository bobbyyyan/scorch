"""A sum reduced into a sparse result, unscheduled: refused under any reduction loop.

Appending to a compressed result inside a loop over a reduced index stores one partial
sum per iteration of that loop.  The refusal ``unscheduled_sparse_result_reduction``
covered the innermost loop only.  Under an outer reduction loop the same coordinate is
appended once per iteration of it, out of order, and the kernels ran past their arrays
(``ss + ss`` reduced over ``i`` into a compressed vector: SIGSEGV).  The refusal now
covers any enclosing reduction loop and names each one.
"""

import pytest

from scorch.compiler.diagnostics import UnsupportedFeature
from tests.test_scorch.sum_cin_support import (
    SPARSE_REDUCTIONS,
    emitted,
)


@pytest.mark.parametrize(
    ("formats", "indices", "result_indices", "result_format"), SPARSE_REDUCTIONS
)
def test_a_sum_reduced_into_a_sparse_result_refuses_unscheduled(
    formats, indices, result_indices, result_format
):
    with pytest.raises(UnsupportedFeature) as refusal:
        emitted(formats, indices, result_indices, result_format)

    assert [diagnostic.code for diagnostic in refusal.value.diagnostics] == [
        "unscheduled_sparse_result_reduction"
    ]


def test_the_refusal_names_every_reduction_loop():
    with pytest.raises(UnsupportedFeature) as refusal:
        emitted(("sss", "sss"), "ijk", "k", "s")

    (diagnostic,) = refusal.value.diagnostics
    assert "the loops over i, j sum over variables" in diagnostic.message
