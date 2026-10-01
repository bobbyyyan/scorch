#!/usr/bin/env python3
"""Per-ROW exposure of a degree-gated kernel over the DLMC collection.

WHY THIS EXISTS. The degree sweep puts V6's break-even at row degree 24-32 on the P-core, so a
four-chain kernel needs a guard that falls back below that. The exposure of such a guard was
quoted from chain210's `deg` column -- 1051 of 3820 keys (27.5%) below average degree 32, holding
3.06% of all nonzeros. The peer's objection is correct and voids that as the exposure statistic:
`deg` is nnz/rows, a PER-MATRIX average, while the guard reads A1_pos[i+1]-A1_pos[i], which is
PER-ROW. A matrix of average degree 32 has rows on both sides of the threshold. chain210's columns
are build, slice, key, k, rows, nnz, deg, arm, ms, batch -- per-matrix aggregates only -- and no
row-length distribution exists anywhere in the spmm-beat-mkl ledger, so the per-row form had no
number in either direction and I had asserted it "should have none" from inference.

It is computable without re-running anything and without reading any values. A DLMC `.smtx` file
is three lines: "rows, cols, nnz", then the CSR row-pointer array, then the column indices. The
row lengths are the successive differences of line 2, so the whole distribution comes from one
line per file.

WHAT IT REPORTS, and the distinction that matters for the convention:
  - share of ROWS below each threshold: how often the fallback fires.
  - share of NONZEROS in those rows: the arithmetic the fallback runs, i.e. what fraction of the
    work does NOT get the faster kernel. This is the per-row exposure.
  - per-matrix worst case: the matrix whose nonzeros are most concentrated in short rows, because
    the performance convention counts per-matrix regressions and a matrix that is ENTIRELY short
    rows gets no speedup at all even though it also cannot regress.
"""
import glob, os, statistics, sys

THR = (4, 8, 12, 16, 24, 32, 48, 64, 128)


def row_lengths(path):
    with open(path) as f:
        hdr = f.readline()
        ptr = f.readline()
    try:
        rows, cols, nnz = [int(x) for x in hdr.replace(",", " ").split()[:3]]
    except ValueError:
        return None
    p = [int(x) for x in ptr.split()]
    if len(p) != rows + 1:
        return None                      # refuse rather than guess at a malformed file
    if p[-1] != nnz:
        return None                      # header and pointer array must agree
    return [p[i + 1] - p[i] for i in range(rows)], nnz


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/dlmc")
    files = sorted(glob.glob(os.path.join(root, "**", "*.smtx"), recursive=True))
    if not files:
        sys.exit("REFUSE: no .smtx under %s" % root)
    tot_rows = tot_nnz = 0
    rows_below = {t: 0 for t in THR}
    nnz_below = {t: 0 for t in THR}
    bad = 0
    permat = []                           # (fraction of this matrix's nnz in rows < 32, path)
    for fp in files:
        r = row_lengths(fp)
        if r is None:
            bad += 1
            continue
        lens, nnz = r
        tot_rows += len(lens)
        tot_nnz += nnz
        short32 = 0
        for L in lens:
            for t in THR:
                if L < t:
                    rows_below[t] += 1
                    nnz_below[t] += L
            if L < 32:
                short32 += L
        permat.append((short32 / nnz if nnz else 0.0, fp))
    print("DLMC per-ROW degree distribution, from the CSR row pointers")
    print("  files read %d   malformed/refused %d   rows %d   nonzeros %d"
          % (len(files) - bad, bad, tot_rows, tot_nnz))
    print("  (the refused count is printed so a silent parse failure cannot read as a clean run)")
    print()
    print("  %6s %14s %10s %16s %10s" % ("thresh", "rows below", "% of rows", "nnz in them",
                                         "% of nnz"))
    print("  " + "-" * 62)
    for t in THR:
        print("  %6d %14d %9.1f%% %16d %9.2f%%"
              % (t, rows_below[t], 100.0 * rows_below[t] / tot_rows, nnz_below[t],
                 100.0 * nnz_below[t] / tot_nnz))
    print()
    print("  THE PER-ROW EXPOSURE is the '%% of nnz' column: the share of the collection's")
    print("  arithmetic that a per-row guard sends to the fallback and that therefore gets no")
    print("  speedup. It is NOT a regression -- those rows run exactly today's kernel.")
    print()
    fr = sorted(x[0] for x in permat)
    print("  PER-MATRIX, fraction of a matrix's nonzeros in rows shorter than 32:")
    print("    median %.4f   p75 %.4f   p90 %.4f   p99 %.4f   max %.4f"
          % (fr[len(fr)//2], fr[int(0.75*(len(fr)-1))], fr[int(0.90*(len(fr)-1))],
             fr[int(0.99*(len(fr)-1))], fr[-1]))
    for cut in (0.5, 0.9, 0.99):
        n = sum(1 for x in fr if x >= cut)
        print("    matrices with >= %.0f%% of their nonzeros in short rows: %d of %d (%.1f%%)"
              % (100*cut, n, len(fr), 100.0*n/len(fr)))
    print()
    worst = max(permat)
    print("  most short-row-concentrated matrix: %.4f of its nonzeros, %s"
          % (worst[0], os.path.relpath(worst[1], root)))


if __name__ == "__main__":
    main()
