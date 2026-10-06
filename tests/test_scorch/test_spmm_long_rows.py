"""Rows on both sides of the register block's long-row length test, against torch's own product.

A chunk of rows holding a row of at least SCORCH_SPMM_REGBLOCK_LONGROW (64) nonzeros runs the copy of
the row loop whose kernel takes such a row in four nonzero streams and finishes it in the two-stream
loop; a chunk without one runs the plain copy. "mixed" puts rows just under, at and just over the
length, with every remainder the four-stream loop can leave (64-67, 127-129), next to short and empty
rows, so a row that skipped or repeated a nonzero at either loop boundary reads back wrong. "rare"
puts one long row among hundreds of short ones, so most chunks take the plain copy and a few take the
other. The widths cover every vector count the loop serves (float32 k <= 16, float64 k <= 8) with
full and masked last vectors, plus one width past it as a control.
"""
import pytest
import torch

import scorch
from scorch import STensor

LENGTHS = {
    "mixed": [0, 1, 2, 3, 63, 64, 65, 66, 67, 68, 0, 127, 128, 129, 300, 5, 64, 0, 71] * 23,
    "rare": ([3, 0, 5, 1, 7, 2, 63] * 70 + [97]) * 4,
}


def _rows(cols, seed, layout):
    g = torch.Generator().manual_seed(seed)
    lengths = LENGTHS[layout]
    crow, parts = [0], []
    for n in lengths:
        parts.append(torch.randperm(cols, generator=g)[:n].sort().values)
        crow.append(crow[-1] + n)
    col = torch.cat(parts)
    vals = torch.randn(len(col), generator=g, dtype=torch.float64)
    return torch.tensor(crow), col, vals, len(lengths)


@pytest.mark.parametrize("layout", ["mixed", "rare"])
@pytest.mark.parametrize(
    "dtype,width",
    [(torch.float32, k) for k in (4, 5, 7, 8, 9, 12, 16, 24)]
    + [(torch.float64, k) for k in (2, 3, 4, 5, 8, 12)],
)
def test_long_and_short_rows_match_torch(dtype, width, layout):
    cols = 700
    crow, col, vals, rows = _rows(cols, seed=width, layout=layout)
    a = torch.sparse_csr_tensor(crow, col, vals.to(dtype), size=(rows, cols))
    b = torch.randn(cols, width, generator=torch.Generator().manual_seed(11), dtype=dtype)
    got = scorch.matmul(STensor.from_torch(a), b)
    want = torch.sparse.mm(a.to(torch.float64), b.to(torch.float64)).to(dtype)
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(got, want, atol=tol * 30, rtol=tol)
