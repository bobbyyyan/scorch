"""Planted cases for bstream_locality, plus a brute-force reference for the vectorized path.

Two defects this file exists to have caught, both real and both found by review rather than by
running: `pack` was once the reciprocal of its own docstring, and the line model was once
`max(1, 64 // S)` whole rows per line, which reports NO sharing at k=6 float64 where four
consecutive 48-byte B rows plainly occupy three 64-byte lines. Case groups 2 and 4 below are
the ones that separate the models; a self-test holding only perfectly-packed k=2 cases passes
under all three formulas, which is why the scattered and the k=6 cases are here.
"""
import sys

import numpy as np

from bstream_locality import lines_per_row, stats_for

FAIL = []
LINE = 64


def ref_lines(indptr, indices, S):
    """Brute force: walk each row, add every line id its byte ranges touch. No vectorization,
    no key packing, no clipping -- so it shares no arithmetic with the function it checks."""
    total = 0
    for r in range(len(indptr) - 1):
        seen = set()
        for j in indices[int(indptr[r]):int(indptr[r + 1])]:
            a = int(j) * S
            for ln in range(a // LINE, (a + S - 1) // LINE + 1):
                seen.add(ln)
        total += len(seen)
    return total


def chk(label, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print("  %-58s %-11.6f %s" % (label, got, "ok" if ok else "FAIL want %.6f" % want))
    if not ok:
        FAIL.append(label)


def chki(label, got, want):
    ok = got == want
    print("  %-58s %-11s %s" % (label, got, "ok" if ok else "FAIL want %s" % want))
    if not ok:
        FAIL.append(label)


print("GROUP 1 -- k=2 float64 (S=16), the width the deficit concentrates at")
lpn, pack, S, sh = stats_for(np.array([0, 4]), np.array([0, 1, 2, 3]), 2, 8)
chki("S", S, 16); chki("shares_line (16 %% 64 != 0)", sh, 1)
chk("4 consecutive columns: lines_per_nnz (1 line / 4 nnz)", lpn, 0.25)
chk("4 consecutive columns: pack (whole line consumed)", pack, 1.0)
lpn, pack, _, _ = stats_for(np.array([0, 4]), np.array([0, 4, 8, 12]), 2, 8)
chk("4 scattered columns: lines_per_nnz", lpn, 1.0)
chk("4 scattered columns: pack (16 of 64 bytes used)", pack, 0.25)

print("GROUP 2 -- k=6 float64 (S=48), where `64 // S == 1` wrongly reported no sharing")
lpn, pack, S, sh = stats_for(np.array([0, 4]), np.array([0, 1, 2, 3]), 6, 8)
chki("S", S, 48); chki("shares_line (48 %% 64 != 0)", sh, 1)
chk("4 consecutive: lines_per_nnz = 3 lines / 4 nnz", lpn, 0.75)
chk("4 consecutive: pack = 192 useful / 192 fetched", pack, 1.0)
lpn, pack, _, _ = stats_for(np.array([0, 2]), np.array([1, 2]), 6, 8)
chk("2 straddling columns: lines_per_nnz = 3 / 2", lpn, 1.5)
chk("2 straddling columns: pack = 96 / 192", pack, 0.5)

print("GROUP 3 -- k=8 and k=16 float64: S %% 64 == 0, so NO column pattern can matter")
for k, want_lpn in ((8, 1.0), (16, 2.0)):
    lpnA, packA, S, sh = stats_for(np.array([0, 4]), np.array([0, 1, 2, 3]), k, 8)
    lpnB, packB, _, _ = stats_for(np.array([0, 4]), np.array([0, 400, 8000, 12]), k, 8)
    chki("k=%d S" % k, S, 8 * k); chki("k=%d shares_line" % k, sh, 0)
    chk("k=%d consecutive: lines_per_nnz" % k, lpnA, want_lpn)
    chk("k=%d scattered reads IDENTICALLY (no signal)" % k, lpnB, lpnA)
    chk("k=%d pack pinned at 1.0 (every fetched byte used)" % k, packA, 1.0)

print("GROUP 4 -- the 2x understatement the old model had at k=10 and k=12")
for k in (10, 12):
    lpn, pack, S, sh = stats_for(np.array([0, 4]), np.array([0, 1000, 2000, 3000]), k, 8)
    chki("k=%d shares_line (old model said no signal)" % k, sh, 1)
    chk("k=%d scattered: lines_per_nnz = 2.0, old model charged 1.0" % k, lpn, 2.0)
    chk("k=%d scattered: pack = %d / 128" % (k, 8 * k), pack, (8.0 * k) / 128.0)

print("GROUP 5 -- float32: the dtype must reach the arithmetic, not be carried by the k label")
lpn, pack, S, sh = stats_for(np.array([0, 8]), np.arange(8), 2, 4)
chki("k=2 f32 S", S, 8); chki("k=2 f32 shares_line", sh, 1)
chk("k=2 f32: 8 consecutive columns in one line", lpn, 0.125)
chk("k=2 f32: pack", pack, 1.0)
chki("k=16 f32 shares_line (S=64, exclusive)", stats_for(np.array([0, 2]), np.array([0, 9]), 16, 4)[3], 0)

print("GROUP 6 -- multi-row, so the row/line key packing is exercised")
ip = np.array([0, 4, 8]); ix = np.array([0, 1, 2, 3, 0, 4, 8, 12])
lpn, pack, _, _ = stats_for(ip, ix, 2, 8)
chk("row0 1 line + row1 4 lines = 5 over 8 nnz", lpn, 0.625)
chk("pack = 128 useful / 320 fetched", pack, 0.4)

print("GROUP 7 -- the vectorized path against the brute-force reference")
rng = np.random.default_rng(20261002)
mism = 0
for trial in range(60):
    nrows = int(rng.integers(1, 12))
    cnt = rng.integers(0, 7, size=nrows)
    ip = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    ix = rng.integers(0, 40, size=int(ip[-1])).astype(np.int64)
    # Sort within each row, as CSR is. Duplicates are left in deliberately: both the
    # vectorized path and the reference deduplicate via sets, so a row repeating a column is
    # valid input to THIS comparison even though main() refuses it downstream.
    for r in range(nrows):
        a, b = int(ip[r]), int(ip[r + 1])
        if b > a:
            ix[a:b] = np.sort(ix[a:b])
    if int(ip[-1]) == 0:
        continue
    for k in (2, 4, 6, 8, 10, 12, 16):
        S, got = k * 8, lines_per_row(ip, ix, k * 8)
        want = ref_lines(ip, ix, S)
        if got != want:
            mism += 1
            print("    MISMATCH trial %d k=%d: %d vs %d" % (trial, k, got, want))
chki("vectorized == brute force over 60 random matrices x 7 widths", mism, 0)

print("GROUP 8 -- the pack assertion fires on its own failing input")
# A repeated column index inflates the numerator without touching the denominator. Eight
# copies of column 0 at k=2 float64: 8*16 = 128 useful bytes charged against one 64-byte line.
_, pack, _, _ = stats_for(np.array([0, 8]), np.zeros(8, dtype=np.int64), 2, 8)
chk("8 duplicate columns -> pack 2.0, above the asserted bound", pack, 2.0)
chki("that value would be REFUSED by the bound in main()", pack > 1.0, True)
chki("nnz 0 returns None", stats_for(np.array([0, 0]), np.array([], dtype=np.int64), 2, 8), None)

print()
if FAIL:
    print("FAILED %d: %s" % (len(FAIL), FAIL))
    sys.exit(1)
print("all planted cases pass (8 groups)")
