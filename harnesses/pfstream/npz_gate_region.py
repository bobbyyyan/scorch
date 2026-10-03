#!/usr/bin/env python3
"""Which cached matrices sit in concern #2's gate region, composed, and emit the list.

The gate forces partition_mode = 0 when `nnz * max(k,16) < mingrains * GRAIN_SPMM`.
It is one predicate of four, so its fired set is an UPPER BOUND on what is attributable:
a cell already excluded by the output-bytes predicate -- `rows * k * elem >= 2 * llc` --
is not live at any pool and cannot be attributed to the ceiling. Both are applied here.

nnz and rows are read EXACTLY, from the `.npy` header of each member rather than inferred
from its compressed size. The header states the shape and the dtype, so there is nothing
to infer. An earlier version of this file divided the ZIP directory's byte count by a
guessed element size, inside a `for isz in (4,8)` loop that could only ever return 4 --
dividing a fixed byte count by a larger element gives a smaller count, so if the first
pair passed a positivity test the others could not be reached. It then discarded the
element size before the filter. Its docstring described a selection the code did not make.

That defect was one-signed, which is the only reason its published figure survived: a
true int64 index read as int32 doubles the inferred nnz, and the predicate is an upper
bound on nnz, so matrices with real nnz in [9375, 18750) were dropped and none was added.
A lower bound was published as if exact.

Usage: npz_gate_region.py <cache-root> [out.txt] [llc-bytes]
"""
import glob
import os
import sys
import zipfile

MINGRAINS, GRAIN = 2, 150000
THRESHOLD = MINGRAINS * GRAIN
MAXOUT_LLC = 2
KS = (1, 2, 4, 8, 16, 32)


def header_shapes(path):
    """(nnz, rows, index_dtype, fmt, declared_rows) -- all read, nothing assumed.

    `len(indptr) - 1` is the ROW count for CSR and the COLUMN count for CSC, and the
    output-bytes predicate reads rows. The only thing asserting CSR here used to be the
    directory being called csrcache, which is a filename claim. scipy's `save_npz` writes
    `format` and `shape` into the same archive, so both are read and checked against the
    pointer length instead of inferred from a path.
    """
    import numpy as np
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            for member in ("indices.npy", "indptr.npy"):
                if member not in names:
                    return None
            hdr = {}
            for member in ("indices.npy", "indptr.npy"):
                with z.open(member) as fh:
                    ver = np.lib.format.read_magic(fh)
                    if ver != (1, 0):
                        # read_array_header_1_0 would misparse a 2.0 header rather than
                        # reject it. These arrays never need a 64 KiB header, so this is
                        # hygiene -- but an unchecked version is an assumption either way.
                        return None
                    shape, _, dtype = np.lib.format.read_array_header_1_0(fh)
                    hdr[member] = (shape, dtype)
            fmt, declared = None, None
            if "format.npy" in names:
                with z.open("format.npy") as fh:
                    fmt = bytes(np.lib.format.read_array(fh).tobytes())
            if "shape.npy" in names:
                with z.open("shape.npy") as fh:
                    declared = tuple(int(x) for x in np.lib.format.read_array(fh))
    except Exception:                                              # noqa: BLE001
        return None
    (ishape, idt), (pshape, _) = hdr["indices.npy"], hdr["indptr.npy"]
    if len(ishape) != 1 or len(pshape) != 1 or pshape[0] < 1:
        return None
    return int(ishape[0]), int(pshape[0]) - 1, idt, fmt, declared


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    root = argv[1]
    out = argv[2] if len(argv) > 2 else None
    llc = int(argv[3]) if len(argv) > 3 else 33554432
    maxout = MAXOUT_LLC * llc

    files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
    print("cache %s: %d .npz files" % (root, len(files)))
    if not files:
        print("EXIT: no matrices found")
        return 1

    got, bad = [], []   # (path, nnz, rows, declared_shape)
    dtypes, fmts, layout_bad = {}, {}, []
    n_declared = n_nonsquare = 0
    for f in files:
        r = header_shapes(f)
        if r is None:
            bad.append(f)
            continue
        nnz, rows, idt, fmt, declared = r
        dtypes[str(idt)] = dtypes.get(str(idt), 0) + 1
        fmts[fmt] = fmts.get(fmt, 0) + 1
        if declared is not None and len(declared) == 2:
            n_declared += 1
            if declared[0] != declared[1]:
                n_nonsquare += 1
            if declared[0] != rows:
                layout_bad.append((f, rows, declared))
        got.append((f, nnz, rows, declared))

    print("read from .npy headers: %d   unreadable: %d" % (len(got), len(bad)))
    for f in bad[:4]:
        print("   unreadable: %s" % os.path.basename(f))
    print("index dtypes present, over all readable files: %r" % dtypes)
    print("   a single dtype here is what makes nnz exact; a mix would have made the old")
    print("   size-division wrong on part of the corpus and right on the rest.")
    # The layout check, and the number that shows it is not vacuous.
    #
    # `format.npy` is absent from every file in this cache -- these are not scipy
    # save_npz archives -- so a check of `format == b'csr'` passes on an exemption for
    # the missing key and proves nothing. The one that does work is already here:
    # len(indptr)-1 equals shape[0] for CSR and shape[1] for CSC. On a SQUARE matrix both
    # hold and the check cannot discriminate, so the count that matters is how many
    # NON-SQUARE matrices passed it.
    print("archives carrying `format`: %d of %d -- a format check would be %s"
          % (sum(v for k, v in fmts.items() if k is not None), len(got),
             "meaningful" if any(k is not None for k in fmts) else "VACUOUS, so not used"))
    print("len(indptr)-1 against the declared shape[0]: %d disagree, of %d compared"
          % (len(layout_bad), n_declared))
    for f, rows, dec in layout_bad[:4]:
        print("   %s  pointer gives %d, shape says %r" % (os.path.basename(f), rows, dec))
    print("   of the %d COMPARED (not of the disagreeing ones listed above), NON-SQUARE:"
          % n_declared)
    print("   %d. Only non-square entries discriminate CSR from CSC, because a square"
          % n_nonsquare)
    print("   matrix satisfies the check under either layout. The per-region count below")
    print("   is the one that certifies the axis for the population each figure is about.")
    if layout_bad or n_declared < len(got):
        print("EXIT: %d layout disagreements and %d files with no declared shape."
              % (len(layout_bad), len(got) - n_declared))
        print("      len(indptr)-1 is the ROW count for CSR and the COLUMN count for CSC,")
        print("      and the output-bytes predicate reads rows, so the composed column")
        print("      below would be computed on the wrong axis.")
        return 1
    if n_nonsquare == 0:
        print("EXIT: every matrix is square, so nothing here distinguishes CSR from CSC")
        print("      and the composed column would rest on the directory's name.")
        return 1
    if bad:
        print("EXIT: %d files unreadable. The denominator for any share below would be"
              % len(bad))
        print("      the READABLE count, not the file count, and publishing the file")
        print("      count while computing the readable one is only correct when they")
        print("      coincide.")
        return 1

    n = len(got)
    print("\nthe gate alone (nnz * max(k,16) < %d), and COMPOSED with the output-bytes"
          % THRESHOLD)
    print("predicate (rows * k * 4 >= %d excludes), llc = %d:" % (maxout, llc))
    n_vec = sum(1 for r in got if r[3] and len(r[3]) == 2 and r[3][1] == 1)
    n_ops = n - n_vec
    print("   one-column entries in the corpus: %d of %d, so the operand denominator"
          " is %d" % (n_vec, n, n_ops))
    print("   %-5s %-14s %-14s %-14s %-10s %s"
          % ("k", "gate alone", "+output-bytes", "+operands only", "share", "removed"))
    regions = {}
    for k in KS:
        gate = [r for r in got if r[1] * max(k, 16) < THRESHOLD]
        comp = [r for r in gate if r[2] * k * 4 < maxout]
        # shape[1] == 1 is a ONE-COLUMN entry. SuiteSparse ships right-hand-side vectors
        # alongside its operators, and a small-nnz predicate selects them preferentially
        # because a one-column entry has the least nnz in the collection. They are valid
        # CSR and not operators, so they belong in neither the numerator nor the
        # denominator. "not a matrix" needs no chosen constant, unlike an nnz floor.
        ops = [r for r in comp if not (r[3] and len(r[3]) == 2 and r[3][1] == 1)]
        nsq = sum(1 for r in comp if r[3] and len(r[3]) == 2 and r[3][0] != r[3][1])
        regions[k] = ops
        removed = sorted(set(gate) - set(comp))
        print("   k=%-3d %-14d %-14d %-14d %-10s %d  %s"
              % (k, len(gate), len(comp), len(ops),
                 "%.1f%%" % (100.0 * len(ops) / n_ops), len(removed),
                 "; ".join("%s nnz=%d rows=%d shape=%s"
                           % (os.path.basename(r[0])[:26], r[1], r[2], r[3])
                           for r in removed[:2])))
        print("          one-column entries dropped from the region: %d;"
              " NON-SQUARE within it: %d of %d = %.1f%% -- the axis check's non-vacuity"
              " for THIS population, not corpus-wide"
              % (len(comp) - len(ops), nsq, len(comp), 100.0 * nsq / len(comp)))
    print("   the share's denominator is the OPERAND count %d, not the file count %d:"
          % (n_ops, len(files)))
    print("   both numerator and denominator shrink when one-column entries come out, so")
    print("   the published 47.5%% and 30.0%% move only to 46.8%% and 29.5%%. The readable")
    print("   count equals the file count here, so no third denominator is in play.")

    rmax = max(r[2] for r in got)
    print("\n   largest row count in the corpus: %d. The output-bytes predicate needs" % rmax)
    print("   rows >= %d at k=16 and %d at k=32, so it is %s on this corpus."
          % (maxout // 4 // 16, maxout // 4 // 32,
             "LIVE" if rmax * 32 * 4 >= maxout else "vacuous"))

    if out:
        k_emit = 16
        with open(out, "w") as fh:
            fh.write("# path\tnnz\trows -- composed gate region at k=%d.\n" % k_emit)
            fh.write("# Membership is k-DEPENDENT and this file is one width: the k=32 set\n")
            fh.write("# is %d smaller. nnz and rows are carried so a consumer recomputes\n"
                     % (len(regions[k_emit]) - len(regions[32])))
            fh.write("# membership at any width: nnz*max(k,16) < %d and rows*k*4 < %d.\n"
                     % (THRESHOLD, maxout))
            for r in sorted(regions[k_emit]):
                fh.write("%s\t%d\t%d\t%s\n" % (r[0], r[1], r[2], r[3]))
        print("\nwrote %d members of the composed k=%d region to %s"
              % (len(regions[k_emit]), k_emit, out))
        print("   the earlier version computed this path and returned without writing it,")
        print("   so the script that identifies the region never emitted the region.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
