"""Products whose rows are short or empty: the work the thread policy is told for them.

``scorch_spmm_work`` is the work the drop-in SpMM states to ``scorch_spmm_nthreads``: a nonzero is
credited its width floored at 16 (its gather touches a whole cache line), an empty row the workers
zero its output width. Rows themselves are not credited (see the function's comment for the
measurement). These pin, through the exported function (never a re-derivation), the formula at
narrow and wide widths, and that a matrix of mostly empty rows and a few dozen nonzeros
(flowmeter5_B: 9669 rows, 24 nonzeros) stays on one worker, where a 5 us call does not pay for
waking the pool.
"""
import pytest
import torch

import scorch
from scorch import STensor

scorch_ops = pytest.importorskip("scorch_ops")
if not hasattr(scorch_ops, "scorch_spmm_work"):
    pytest.skip("scorch_ops predates scorch_spmm_work", allow_module_level=True)

GRAIN = 150000  # SCORCH_GRAIN_SPMM


@pytest.mark.parametrize("width", [1, 2, 4, 15, 16, 32, 256])
def test_formula(width):
    w = max(width, 16)
    assert scorch_ops.scorch_spmm_work(1000, 300, width, width) == 1000 * w + 300 * width


def test_a_nearly_empty_matrix_stays_on_one_worker():
    rows, nnz, empty, pool = 9669, 24, 9645, 24  # flowmeter5_B
    work = scorch_ops.scorch_spmm_work(nnz, empty, 1, 1)
    assert work < GRAIN
    assert scorch_ops.scorch_spmm_nthreads(work, rows, pool, nnz, nnz, True) == 1
