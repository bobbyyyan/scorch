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


NASSERT = 0
NGROUPS = 0


def group(label):
    """Print a group header and count it. NGROUPS was a literal `11` baked in at patch time,
    which is the defect this file's footer was fixed to avoid: adding a group without editing
    that line printed the old count. Counted where the header prints, so the two cannot drift."""
    global NGROUPS
    NGROUPS += 1
    print(label)


def chk(label, got, want, tol=1e-9):
    global NASSERT
    NASSERT += 1
    ok = abs(got - want) <= tol
    print("  %-58s %-11.6f %s" % (label, got, "ok" if ok else "FAIL want %.6f" % want))
    if not ok:
        FAIL.append(label)


def chki(label, got, want):
    global NASSERT
    NASSERT += 1
    ok = got == want
    print("  %-58s %-11s %s" % (label, got, "ok" if ok else "FAIL want %s" % want))
    if not ok:
        FAIL.append(label)


group("GROUP 1 -- k=2 float64 (S=16), the width the deficit concentrates at")
lpn, pack, S, sh = stats_for(np.array([0, 4]), np.array([0, 1, 2, 3]), 2, 8)
chki("S", S, 16); chki("shares_line (16 %% 64 != 0)", sh, 1)
chk("4 consecutive columns: lines_per_nnz (1 line / 4 nnz)", lpn, 0.25)
chk("4 consecutive columns: pack (whole line consumed)", pack, 1.0)
lpn, pack, _, _ = stats_for(np.array([0, 4]), np.array([0, 4, 8, 12]), 2, 8)
chk("4 scattered columns: lines_per_nnz", lpn, 1.0)
chk("4 scattered columns: pack (16 of 64 bytes used)", pack, 0.25)

group("GROUP 2 -- k=6 float64 (S=48), where `64 // S == 1` wrongly reported no sharing")
lpn, pack, S, sh = stats_for(np.array([0, 4]), np.array([0, 1, 2, 3]), 6, 8)
chki("S", S, 48); chki("shares_line (48 %% 64 != 0)", sh, 1)
chk("4 consecutive: lines_per_nnz = 3 lines / 4 nnz", lpn, 0.75)
chk("4 consecutive: pack = 192 useful / 192 fetched", pack, 1.0)
lpn, pack, _, _ = stats_for(np.array([0, 2]), np.array([1, 2]), 6, 8)
chk("2 straddling columns: lines_per_nnz = 3 / 2", lpn, 1.5)
chk("2 straddling columns: pack = 96 / 192", pack, 0.5)

group("GROUP 3 -- k=8 and k=16 float64: S %% 64 == 0, so NO column pattern can matter")
for k, want_lpn in ((8, 1.0), (16, 2.0)):
    lpnA, packA, S, sh = stats_for(np.array([0, 4]), np.array([0, 1, 2, 3]), k, 8)
    lpnB, packB, _, _ = stats_for(np.array([0, 4]), np.array([0, 400, 8000, 12]), k, 8)
    chki("k=%d S" % k, S, 8 * k); chki("k=%d shares_line" % k, sh, 0)
    chk("k=%d consecutive: lines_per_nnz" % k, lpnA, want_lpn)
    chk("k=%d scattered reads IDENTICALLY (no signal)" % k, lpnB, lpnA)
    chk("k=%d pack pinned at 1.0 (every fetched byte used)" % k, packA, 1.0)

group("GROUP 4 -- the 2x understatement the old model had at k=10 and k=12")
for k in (10, 12):
    lpn, pack, S, sh = stats_for(np.array([0, 4]), np.array([0, 1000, 2000, 3000]), k, 8)
    chki("k=%d shares_line (old model said no signal)" % k, sh, 1)
    chk("k=%d scattered: lines_per_nnz = 2.0, old model charged 1.0" % k, lpn, 2.0)
    chk("k=%d scattered: pack = %d / 128" % (k, 8 * k), pack, (8.0 * k) / 128.0)

group("GROUP 5 -- float32: the dtype must reach the arithmetic, not be carried by the k label")
lpn, pack, S, sh = stats_for(np.array([0, 8]), np.arange(8), 2, 4)
chki("k=2 f32 S", S, 8); chki("k=2 f32 shares_line", sh, 1)
chk("k=2 f32: 8 consecutive columns in one line", lpn, 0.125)
chk("k=2 f32: pack", pack, 1.0)
chki("k=16 f32 shares_line (S=64, exclusive)", stats_for(np.array([0, 2]), np.array([0, 9]), 16, 4)[3], 0)

group("GROUP 6 -- multi-row, so the row/line key packing is exercised")
ip = np.array([0, 4, 8]); ix = np.array([0, 1, 2, 3, 0, 4, 8, 12])
lpn, pack, _, _ = stats_for(ip, ix, 2, 8)
chk("row0 1 line + row1 4 lines = 5 over 8 nnz", lpn, 0.625)
chk("pack = 128 useful / 320 fetched", pack, 0.4)

group("GROUP 7 -- the vectorized path against the brute-force reference")
rng = np.random.default_rng(20261002)
mism = ncmp = ntrial = ndup = 0
spans = set()
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
    ntrial += 1
    ndup += 1 if any(len(set(ix[int(ip[r]):int(ip[r + 1])].tolist()))
                     < int(ip[r + 1]) - int(ip[r]) for r in range(nrows)) else 0
    for k in (2, 4, 6, 8, 10, 12, 16):
        S, got = k * 8, lines_per_row(ip, ix, k * 8)
        want = ref_lines(ip, ix, S)
        ncmp += 1
        spans.add((S - 1) // LINE + 2)
        if got != want:
            mism += 1
            print("    MISMATCH trial %d k=%d: %d vs %d" % (trial, k, got, want))
chki("vectorized == brute force, mismatches", mism, 0)
# A printed 0 cannot distinguish "413 comparisons all agreed" from "0 comparisons ran". One of
# the 60 trials draws all-zero row counts and hits the `continue`, so the label "60 matrices x
# 7 widths" overstates by 7. Assert the comparison COUNT, not only the mismatch count.
chki("comparisons actually run (59 non-empty trials x 7 widths)", ncmp, 413)
chki("trials skipped for being entirely empty", 60 - ntrial, 1)
print("    non-vacuity: %d of %d non-empty trials contain a repeated column index in some row,"
      % (ndup, ntrial))
print("    so the duplicate path is exercised by this comparison and not only by GROUP 8")
chki("trials exercising duplicate columns is nonzero", ndup > 0, True)
chki("spans exercised by GROUP 7", sorted(spans), [2, 3])

group("GROUP 7b -- the reference has discriminating power: it REJECTS the superseded model")
# GROUP 7 passing shows the two implementations agree. It does not show the comparison could
# fail. Running the same reference against the model this file replaced -- whole B rows per
# line -- must DISAGREE, otherwise GROUP 7 is a check that cannot go red.
def old_model_lines(ip, ix, S):
    bpl = max(1, LINE // S)
    tot = 0
    for r in range(len(ip) - 1):
        a_, b_ = int(ip[r]), int(ip[r + 1])
        if b_ > a_:
            tot += int(np.unique(ix[a_:b_] // bpl).size)
    return tot
ip_d = np.array([0, 2]); ix_d = np.array([1, 5])
chki("at k=6 the reference and the OLD model disagree (4 vs 2 lines)",
     (ref_lines(ip_d, ix_d, 48), old_model_lines(ip_d, ix_d, 48)), (4, 2))
caught = sum(1 for k in (2, 4, 6, 8, 10, 12, 16)
             if ref_lines(ip_d, ix_d, 8 * k) != old_model_lines(ip_d, ix_d, 8 * k))
chki("widths at which the reference catches the old model", caught, 4)

group("GROUP 7c -- the answer is invariant to the row-blocking size")
# Row blocking is a memory bound, not a model change: the total is a sum over independent
# rows. Shrinking the block to 3 nonzeros must not move any answer.
import bstream_locality as _bl
keep = _bl.CHUNK_NNZ
moved = 0
try:
    for trial in range(12):
        nrows = int(rng.integers(2, 9))
        cnt = rng.integers(0, 6, size=nrows)
        ip = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        if int(ip[-1]) == 0:
            continue
        ix = np.sort(rng.integers(0, 50, size=int(ip[-1]))).astype(np.int64)
        for k in (2, 6, 16):
            _bl.CHUNK_NNZ = 10 ** 9
            unblocked = _bl.lines_per_row(ip, ix, 8 * k)
            _bl.CHUNK_NNZ = 3
            blocked = _bl.lines_per_row(ip, ix, 8 * k)
            if unblocked != blocked:
                moved += 1
finally:
    _bl.CHUNK_NNZ = keep
chki("blocked == unblocked over 12 matrices x 3 widths", moved, 0)

group("GROUP 7d -- the int64 packing predicate, planted at values no corpus can reach")
from bstream_locality import fits_int64
MAXI = np.iinfo(np.int64).max
chki("fits_int64 accepts the collection's extremes (9888048 rows, stride 9380004)",
     fits_int64(9888048, 9380004), True)
chki("fits_int64 REFUSES a product past int64", fits_int64(MAXI // 2, 4), False)
chki("fits_int64 at exactly the boundary", fits_int64(MAXI // 1000, 1000), True)
# The ADJACENT failing value, not two past it. Mutation table over these five plants:
# `<` instead of `<=` is caught by the boundary plant, but a predicate one value too
# PERMISSIVE -- `nrows <= MAXI // stride + 1`, which admits a real wrap of up to `stride` --
# survives every other plant here and is caught only by this one.
chki("fits_int64 at the adjacent failing value (+1)", fits_int64(MAXI // 1000 + 1, 1000), False)
chki("degenerate stride is accepted rather than dividing by zero", fits_int64(10, 0), True)

group("GROUP 8 -- the pack assertion fires on its own failing input")
# A repeated column index inflates the numerator without touching the denominator. Eight
# copies of column 0 at k=2 float64: 8*16 = 128 useful bytes charged against one 64-byte line.
_, pack, _, _ = stats_for(np.array([0, 8]), np.zeros(8, dtype=np.int64), 2, 8)
chk("8 duplicate columns -> pack 2.0, above the asserted bound", pack, 2.0)
chki("that value would be REFUSED by the bound in main()", pack > 1.0, True)
chki("nnz 0 returns None", stats_for(np.array([0, 0]), np.array([], dtype=np.int64), 2, 8), None)

group("GROUP 12 -- the .smtx reader, including the failure its format allows")
import os
import tempfile
from bstream_locality import read_smtx, load_csr

TD = tempfile.mkdtemp()


def write_smtx(name, header, ipt, idx):
    p = os.path.join(TD, name + ".smtx")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        fh.write(header + "\n" + ipt + "\n" + idx + "\n")
    return p


# A well-formed 2x8 matrix with 6 nonzeros: row 0 has columns 0,1,2 and row 1 has 4,5,6.
good = write_smtx("ok/m", "2, 8, 6", "0 3 6", "0 1 2 4 5 6")
ip, ix, r, c = read_smtx(good)
chki("smtx rows", r, 2)
chki("smtx cols", c, 8)
chki("smtx indptr", list(ip), [0, 3, 6])
chki("smtx indices", list(ix), [0, 1, 2, 4, 5, 6])
chki("load_csr dispatches to .smtx when the file exists",
     load_csr(TD, "ok/m", None)[2:], (2, 8))

# The failure this format allows: a truncated indices line. The file parses, every array is
# well-formed, and the result reads as a legitimately sparser matrix. Only the header
# cross-check sees it, which is why the reader does not trust the header.
trunc = write_smtx("bad/trunc", "2, 8, 6", "0 3 6", "0 1 2 4")
try:
    read_smtx(trunc)
    chki("truncated indices REFUSED", False, True)
except ValueError as e:
    chki("truncated indices REFUSED", "len(indices) 4" in str(e), True)

# An indptr of the wrong length is the other half, and a different check.
shortp = write_smtx("bad/shortp", "3, 8, 6", "0 3 6", "0 1 2 4 5 6")
try:
    read_smtx(shortp)
    chki("indptr length mismatch REFUSED", False, True)
except ValueError as e:
    chki("indptr length mismatch REFUSED", "should hold 4 entries, holds 3" in str(e), True)

# And the predictor runs end to end on a parsed .smtx. At k=2 float64 (S=16): row 0 is columns
# 0,1,2 at bytes 0..47, inside line 0; row 1 is columns 4,5,6 at bytes 64..111, ALL inside
# line 1 -- not two lines, which is what I first wrote here. 2 lines over 6 nonzeros.
lpn, pack, S, sh = stats_for(ip, ix, 2, 8)
chki("smtx -> stats_for S", S, 16)
chk("smtx -> lines_per_nnz (1 line per row, 2 over 6 nnz)", lpn, 2.0 / 6.0)
chk("smtx -> pack (6*16 useful / 2*64 fetched)", pack, (6 * 16.0) / (2 * 64.0))
# A column that straddles: column 3 is bytes 48..63 (line 0) and column 4 is 64..79 (line 1),
# so adding column 3 to row 0 keeps it at one line while row 1 stays at one. Column 7 would
# reach bytes 112..127, still line 1. The first column to add a line to row 1 is 8.
ip2 = np.array([0, 3, 7]); ix2 = np.array([0, 1, 2, 4, 5, 6, 8])
chk("adding column 8 to row 1 adds a line (3 over 7 nnz)", stats_for(ip2, ix2, 2, 8)[0], 3.0 / 7.0)

print()
if FAIL:
    print("FAILED %d: %s" % (len(FAIL), FAIL))
    sys.exit(1)
print("all planted cases pass (%d groups, %d assertions)" % (NGROUPS, NASSERT))
