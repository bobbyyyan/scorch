"""A sum reduced over an index, through the developer entry: one loop emission.

The loop over an index the result does not have was emitted by a second copy of the
loop cases in ``IterationLattice``, and for a single iterator that copy always
re-initialized it.  After the merge of ``A``'s and ``B``'s stored entries along ``j``,
the loop over ``A``'s remaining entries started again at the row's first one
(``for (int pA1 = A1_pos[pA0]; ...)``): ``ss + ss`` reduced into a dense vector added
every merged entry twice.  Under a dense extent the same copy put the iterator's
declaration in the header of the loop over the dense variable
(``for (int pA1 = A1_pos[pA0]; j < B1_size; j++)``), which did not build.  There is one
emission now; a loop that resolves no result level differs only in having no result
positions to resolve.
"""

import pytest
import torch

from tests.test_scorch.sum_cin_support import (
    DENSE_REDUCTIONS,
    LARGE,
    SMALL,
    emitted,
    run,
)


def test_the_loop_after_a_merge_continues_from_the_merge():
    """``ss + ss`` reduced over ``j``: the remaining entries, not the row again."""

    source = emitted(("ss", "ss"), "ij", "i", "d")

    merge = source.index("while (pA1 < pA1_end && pB1 < pB1_end) {")
    both_rows_end = source.index("} else if (i_A == i) {")
    after_merge = source[merge:both_rows_end]
    assert "while (pA1 < pA1_end) {" in after_merge
    assert "while (pB1 < pB1_end) {" in after_merge
    assert not any(line.startswith("for (int pA1 = A1_pos") for line in after_merge)
    assert not any(line.startswith("for (int pB1 = B1_pos") for line in after_merge)
    # A row only one operand stores is still one loop from the row's first entry.
    assert "for (int pA1 = A1_pos[pA0]; pA1 < pA1_end; pA1++) {" in source


def test_a_reduced_index_under_a_dense_extent_loops_over_the_extent():
    """``ss + dd`` reduced over ``j``: a loop over ``j`` that carries ``A``'s position."""

    source = emitted(("ss", "dd"), "ij", "i", "d")

    loop = source.index("for (int64_t j = 0; j < B1_size; j++) {")
    assert source[loop - 1] == "int pA1 = A1_pos[pA0];"
    assert "pA1 += (int)(j_A == j);" in source[loop:]
    assert not any(line.startswith("for (int pA1 = ") for line in source)


@pytest.mark.parametrize("extents", [SMALL, LARGE], ids=["small", "large"])
@pytest.mark.parametrize(
    ("formats", "indices", "result_indices", "result_format"), DENSE_REDUCTIONS
)
def test_a_sum_reduced_into_a_dense_result_matches(
    formats, indices, result_indices, result_format, extents
):
    run(formats, indices, result_indices, result_format, extents, torch.float64, 23)


@pytest.mark.parametrize(
    ("formats", "indices", "result_indices", "result_format"),
    [
        (("ss", "ss"), "ij", "i", "d"),
        (("ss", "dd"), "ij", "j", "d"),
        (("sss", "dss"), "ijk", "ik", "dd"),
    ],
)
def test_a_sum_over_many_rows_matches_in_float32(
    formats, indices, result_indices, result_format
):
    run(formats, indices, result_indices, result_format, LARGE, torch.float32, 31)
