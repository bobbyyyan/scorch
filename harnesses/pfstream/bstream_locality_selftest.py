"""Planted cases for bstream_locality.stats_for. Each pins a value a formula error moves.

The first version of `pack` was the reciprocal of its own docstring. Case 2 below is the one
that separates them: it reads 0.25 under the correct formula and 4.0 under the inverted one.
Case 1 reads 1.0 under BOTH, so a self-test holding only the packed case would have passed on
the broken formula -- which is why both ends are here and why case 1 alone is not enough.
"""
import sys

import numpy as np

from bstream_locality import stats_for

FAIL = []


def check(label, got, want, tol=1e-12):
    ok = abs(got - want) <= tol
    print("  %-56s %-12.6f %s" % (label, got, "ok" if ok else "FAIL want %.6f" % want))
    if not ok:
        FAIL.append(label)


# k=2 float64 -> 16 bytes per B row -> 4 B rows per 64-byte line.
BPL = 4

# Case 1: one row, columns 0,1,2,3 -- one line, fully consumed.
lpn, pack, bpl = stats_for(np.array([0, 4]), np.array([0, 1, 2, 3]), 2, 8)
check("packed row: brows_per_line", float(bpl), BPL)
check("packed row: lines_per_nnz (1 line / 4 nnz)", lpn, 0.25)
check("packed row: pack (whole line used)", pack, 1.0)

# Case 2: one row, columns 0,4,8,12 -- four lines, one B row used from each.
# This is the case the inverted formula gets wrong: 1.0 * 4 = 4.0 instead of 0.25.
lpn, pack, bpl = stats_for(np.array([0, 4]), np.array([0, 4, 8, 12]), 2, 8)
check("scattered row: lines_per_nnz (4 lines / 4 nnz)", lpn, 1.0)
check("scattered row: pack (1 of 4 B rows per line)", pack, 0.25)

# Case 3: unsorted columns within a row must give the same answer as sorted. np.unique does
# not require sortedness; an implementation that diffed adjacent indices would read 4 lines
# here and 1 line for case 1, so this is the metamorphic form of case 1.
lpn, pack, _ = stats_for(np.array([0, 4]), np.array([3, 0, 2, 1]), 2, 8)
check("unsorted packed row: lines_per_nnz == sorted", lpn, 0.25)
check("unsorted packed row: pack == sorted", pack, 1.0)

# Case 4: at k=16 float64 a B row is 128 bytes, wider than a line, so brows_per_line clamps
# to 1 and every nonzero pays its own line regardless of structure. pack is then pinned at
# 1.0 and the predictor is CONSTANT -- which is the registered reason the mechanism must
# stop separating cells at wide k, stated as a computed value rather than an expectation.
lpn, pack, bpl = stats_for(np.array([0, 4]), np.array([0, 1, 2, 3]), 16, 8)
check("k=16 f64: brows_per_line clamps to 1", float(bpl), 1.0)
check("k=16 f64: lines_per_nnz pinned at 1", lpn, 1.0)
check("k=16 f64: pack pinned at 1 (no structure signal)", pack, 1.0)
lpn2, pack2, _ = stats_for(np.array([0, 4]), np.array([0, 400, 8000, 12]), 16, 8)
check("k=16 f64: scattered reads the SAME as packed", pack2, pack)

# Case 5: float32 at k=2 -> 8 bytes per B row -> 8 per line, so the floor is 1/8 not 1/4.
# Pins that the dtype reaches the arithmetic rather than being carried by the k=2 label.
lpn, pack, bpl = stats_for(np.array([0, 8]), np.arange(8), 2, 4)
check("k=2 f32: brows_per_line", float(bpl), 8.0)
check("k=2 f32: packed 8 columns, one line", lpn, 0.125)
check("k=2 f32: pack", pack, 1.0)

# Case 6: nnz 0 must return None, not a division by zero.
print("  %-56s %s" % ("empty matrix returns None",
                      "ok" if stats_for(np.array([0, 0]), np.array([], dtype=int), 2, 8) is None
                      else "FAIL"))
if stats_for(np.array([0, 0]), np.array([], dtype=int), 2, 8) is not None:
    FAIL.append("empty")

print()
if FAIL:
    print("FAILED %d: %s" % (len(FAIL), FAIL))
    sys.exit(1)
print("all planted cases pass (15 assertions, 6 cases)")
