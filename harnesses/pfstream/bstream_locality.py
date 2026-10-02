"""Structural predictor of B-side memory traffic for CSR SpMM, per key and per width.

WHY THIS EXISTS. The float64 screen leaves 192 of 2926 cells behind MKL, concentrated at
narrow k and high mean degree (129 at k=2, none at k=12 or k=16), and NEITHER prefetch arm
moves them: PFD=0 wins 1.3-2.9% across k=4..16 and reads +0.139% on the behind set at k=2.
So the deficit is not the register-block prefetch. The standing hypothesis is that the kernel
is bound by B-row traffic: every nonzero gathers one row of B, and whether two gathered rows
land in the same cache line is a property of A's column index pattern, not of the kernel.

This script measures that and nothing else. It reads only `indptr` and `indices` from the
cached npz -- no values, no timing, no kernel -- so no arm can influence it, and it can be
computed before any per-cell verdict is read.

THE MODEL IS BYTE EXTENTS, NOT WHOLE-ROW PACKING. A B row for column j occupies bytes
[j*S, j*S + S) where S = k * itemsize, and it spans every 64-byte line that range touches.
The first version of this file instead used `brows_per_line = max(1, 64 // S)` -- how many
WHOLE B rows fit in a line -- and that is the wrong predicate for sharing. At k=6 float64 a B
row is 48 bytes, `64 // 48 == 1`, and the old code therefore reported no sharing at all; but
four consecutive 48-byte rows occupy 192 bytes = exactly three lines, so they plainly share.
The correct condition for a B row to own its lines exclusively is `S % 64 == 0`, which for
float64 holds only at k = 8 and k = 16. Measured against the exact extent model, the old
column was right at k in {2, 4, 8}, wrong in BOTH directions at k=6 (1.5 lines per nonzero for
scattered columns, 0.75 for four consecutive, against a charged 1.0), and understated by
exactly 2x at k = 10, 12 and 16. The extent model needs no per-width case and no clamp.

ASSUMPTION, STATED BECAUSE IT IS NOT DERIVABLE FROM THE INPUTS. B's row 0 is taken to start
on a 64-byte boundary, so line ids are `(j*S)//64`. A torch tensor's data pointer is at least
64-byte aligned in practice but the npz does not record it, and a one-row offset changes the
count by at most one line per gathered row. The second assumption, no reuse ACROSS rows of A,
is what makes this an upper bound on traffic rather than a prediction of it; `b_footprint` is
reported so a reader can see where B is small enough that the bound is loose, instead of this
script picking a cache size on their behalf.

COST AND COVERAGE, both stated because neither is derivable from the output. Row-blocked at
2,000,000 nonzeros, so peak working memory is about 24 * span * CHUNK_NNZ bytes regardless of
matrix size, roughly 144 MB at span 3. `span` is `(S-1)//64 + 2`, which is 2 for S <= 64 and 3
for 64 < S <= 128; `span >= 4` needs S >= 193, and no registered width reaches it (float64
tops out at S=128 for k=16, and the float32 DLMC width set at S=128 for k=32). So the blocked
union path is exercised at exactly the spans the registered runs use.

WHAT IT REPORTS, per (key, k):
  deg            nnz / rows, the known separator. Reported so the new columns can be shown to
                 add information OVER it rather than restate it.
  row_bytes      S = k * itemsize, the gathered B row width.
  shares_line    1 if S % 64 != 0, so two distinct B rows can occupy one line; 0 if every B
                 row owns its lines outright and no column pattern can change the traffic.
                 This is the predictor's reach, derived rather than asserted.
  lines_per_nnz  distinct 64-byte B lines touched per row of A, summed and divided by nnz.
  b_bytes_nnz    64 * lines_per_nnz: bytes of B fetched per nonzero under the no-reuse
                 assumption. Comparable to a_bytes_nnz.
  a_bytes_nnz    itemsize + 4, the value and the column index. Fixed by dtype, not structure.
  pack           useful B bytes over fetched B bytes, = nnz*S / (64 * total_lines). In (0, 1];
                 1.0 means every fetched byte is consumed. A value ABOVE 1 is impossible for
                 well-formed CSR and means some row repeats a column index, which inflates the
                 numerator without touching the denominator -- so the assertion at the bottom
                 of this file is a live input check, not decoration.
  b_footprint    cols * S, the whole B panel in bytes.

WHAT IT DOES NOT DO. It does not classify a matrix, apply a threshold, or name a cache level.
Every such constant would be a knob tuned on a corpus whose verdicts I have already seen. The
prediction registered against it is amendment 26 in PREREG_pfstream_screen.md, and it is
directional and conditional on degree: a predictor that only reproduces the degree split has
added nothing.
"""
import csv, os, sys

import numpy as np


ITEMSIZE = {"float32": 4, "float64": 8}
LINE = 64


def fits_int64(nrows, stride):
    """Can (row, line) pairs be packed as `row * stride + line` without wrapping int64?

    Pulled out as a pure predicate so it has a plant. An earlier version inlined this as a
    bare `raise OverflowError` whose message named a "pairwise unique path" that does not
    exist in this file, in a module where every other failure returns a "REFUSE: ..." string
    from main(); and no self-test input reached it. A guard with no reachable input is not a
    guard, which is the comment three lines below the one it sat above.

    A wrap here would alias two different rows onto one key and UNDER-count lines, reading as
    better locality -- the direction that flatters the hypothesis this script exists to test.
    """
    if nrows <= 0 or stride <= 0:
        return True
    return nrows <= (np.iinfo(np.int64).max // stride)


# Nonzeros per row-block. The total is a SUM over independent rows, so blocking changes no
# answer and bounds peak memory: the unblocked version allocated `span` int64 copies of the
# whole index array plus numpy's sort buffer, about 24*span*nnz bytes, which is 2.09 GB at the
# collection's largest matrix (28,967,291 nonzeros) at span 3 -- and 75 of 4584 keys exceed
# 1e7 nonzeros, so that cost lands late in a sweep and on the slowest keys.
CHUNK_NNZ = 2_000_000


def lines_per_row(indptr, indices, S):
    """Total distinct 64-byte B lines summed over rows, blocked over rows.

    A row's gathered B rows span lines `lo = (j*S)//64` through `hi = (j*S+S-1)//64`. A
    length-S byte range spans at most `ceil(S/64) + 1 = (S-1)//64 + 2` lines, so unioning that
    many shifted copies clipped at `hi` enumerates every line touched, with the surplus copies
    collapsing under `unique`. Row ids are made block-local and the stride is taken from the
    block, so the packed key stays small.
    """
    indptr = indptr.astype(np.int64)
    nnz = int(indptr[-1])
    if nnz == 0:
        return 0
    nrows = len(indptr) - 1
    span = (S - 1) // LINE + 2
    total = 0
    r0 = 0
    while r0 < nrows:
        # Advance to the first row whose cumulative nonzero count passes the budget, always
        # taking at least one row so a single row wider than the budget still progresses.
        r1 = int(np.searchsorted(indptr, indptr[r0] + CHUNK_NNZ, side="right")) - 1
        r1 = max(r1, r0 + 1)
        r1 = min(r1, nrows)
        a, b = int(indptr[r0]), int(indptr[r1])
        if b > a:
            off = indices[a:b].astype(np.int64) * S
            lo, hi = off // LINE, (off + S - 1) // LINE
            local = np.repeat(np.arange(r1 - r0, dtype=np.int64),
                              np.diff(indptr[r0:r1 + 1]))
            stride = int(hi.max()) + 1
            if not fits_int64(r1 - r0, stride):
                raise OverflowError("block of %d rows times line stride %d exceeds int64"
                                    % (r1 - r0, stride))
            keys = [local * stride + np.minimum(lo + t, hi) for t in range(span)]
            total += int(np.unique(np.concatenate(keys)).size)
        r0 = r1
    return total


def stats_for(indptr, indices, k, itemsize):
    """Return (lines_per_nnz, pack, S, shares_line) for one matrix at one width, or None."""
    nnz = int(indptr[-1])
    if nnz == 0:
        return None
    S = k * itemsize
    total = lines_per_row(indptr, indices, S)
    lpn = total / float(nnz)
    pack = (nnz * float(S)) / (LINE * total)
    return lpn, pack, S, int(S % LINE != 0)


def main(argv):
    if len(argv) < 5:
        return "usage: bstream_locality.py <cache> <keysfile> <dtype> <k,k,...> [out.csv]"
    cache, keysfile, dtype, kspec = argv[1], argv[2], argv[3], argv[4]
    out = argv[5] if len(argv) > 5 else "-"
    if dtype not in ITEMSIZE:
        return "REFUSE: dtype %r is not one of %s" % (dtype, sorted(ITEMSIZE))
    itemsize = ITEMSIZE[dtype]
    ks = [int(x) for x in kspec.split(",") if x.strip()]
    if len(set(ks)) != len(ks):
        return "REFUSE: duplicate widths in %r; a duplicate silently doubles a key's weight" % kspec

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from csrresolve import resolve_all, describe
    except ImportError:
        return ("REFUSE: csrresolve.py is not importable next to this script. Resolving keys "
                "here with a second os.path.exists is the defect that module exists to prevent.")

    keys = [ln.strip() for ln in open(keysfile) if ln.strip() and not ln.startswith("#")]
    paths, how = resolve_all(cache, keys)
    print(describe(how))
    missing = [k for k in keys if paths.get(k) is None]
    if missing:
        return ("REFUSE: %d of %d keys did not resolve, first %s. A structural predictor "
                "computed on a subset is not comparable to a verdict table computed on the "
                "whole corpus." % (len(missing), len(keys), missing[:3]))

    print("reach, from S %% 64 (1 = two B rows can share a line, so column pattern matters):")
    for k in ks:
        S = k * itemsize
        print("  k=%-3d S=%-4d shares_line=%d" % (k, S, int(S % LINE != 0)))

    rows_out, zero = [], 0
    for n, key in enumerate(keys, 1):
        z = np.load(paths[key])
        indptr, indices, shape = z["indptr"], z["indices"], z["shape"]
        rows, cols = int(shape[0]), int(shape[1])
        nnz = int(indptr[-1])
        for k in ks:
            # nnz == 0 is handled in ONE place, inside stats_for, so this guard has reachable
            # input. An earlier version skipped such keys before the call AND kept this branch,
            # which made the branch dead -- a guard with no reachable input is not a guard.
            try:
                st = stats_for(indptr, indices, k, itemsize)
            except OverflowError as e:
                return ("REFUSE: key %s at k=%d could not be keyed safely: %s. Every other "
                        "failure in this module returns a REFUSE line; this one did too."
                        % (key, k, e))
            if st is None:
                zero += 1
                continue
            lpn, pack, S, shares = st
            rows_out.append(dict(
                key=key, k=k, rows=rows, cols=cols, nnz=nnz,
                deg="%.4f" % (nnz / float(rows)),
                row_bytes=S,
                shares_line=shares,
                lines_per_nnz="%.6f" % lpn,
                b_bytes_nnz="%.4f" % (LINE * lpn),
                a_bytes_nnz=itemsize + 4,
                pack="%.6f" % pack,
                b_footprint=cols * S,
            ))
        if n % 50 == 0:
            print("  %d of %d keys" % (n, len(keys)))

    print("rows %d ; (key, k) pairs skipped for nnz 0: %d" % (len(rows_out), zero))
    if not rows_out:
        return "REFUSE: no keys produced a row; nothing was measured"
    fh = sys.stdout if out == "-" else open(out, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
    if out != "-":
        fh.close()
        print("wrote %d rows to %s" % (len(rows_out), out))

    # pack is useful bytes over fetched bytes. It cannot exceed 1 for well-formed CSR: a row
    # touching FEWER lines than it has nonzeros is the intended case (four consecutive columns
    # at k=2 float64 share one line), and that lowers the denominator and the numerator
    # together. The only way past 1 is a repeated column index within a row, which counts the
    # same fetched bytes twice in the numerator. So this fires on malformed input, and the
    # lower half can never fire, which is why it is stated as a strict positive and not a
    # computed floor.
    bad = [r for r in rows_out if not (0.0 < float(r["pack"]) <= 1.0 + 1e-12)]
    if bad:
        return ("REFUSE: %d rows have pack outside (0, 1], which means a repeated column index "
                "within a row. First: %s" % (len(bad), bad[0]))
    print("pack within (0, 1] on all %d rows (no repeated column indices)" % len(rows_out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
