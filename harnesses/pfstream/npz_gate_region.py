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
    """(nnz, rows, index_dtype) from the .npy headers. Exact, not inferred."""
    import numpy as np
    out = {}
    try:
        with zipfile.ZipFile(path) as z:
            for member in ("indices.npy", "indptr.npy"):
                if member not in z.namelist():
                    return None
                with z.open(member) as fh:
                    np.lib.format.read_magic(fh)
                    shape, _, dtype = np.lib.format.read_array_header_1_0(fh)
                    out[member] = (shape, dtype)
    except Exception:                                              # noqa: BLE001
        return None
    (ishape, idt), (pshape, _) = out["indices.npy"], out["indptr.npy"]
    if len(ishape) != 1 or len(pshape) != 1 or pshape[0] < 1:
        return None
    return int(ishape[0]), int(pshape[0]) - 1, idt


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

    got, bad = [], []
    dtypes = {}
    for f in files:
        r = header_shapes(f)
        if r is None:
            bad.append(f)
            continue
        nnz, rows, idt = r
        dtypes[str(idt)] = dtypes.get(str(idt), 0) + 1
        got.append((f, nnz, rows))

    print("read from .npy headers: %d   unreadable: %d" % (len(got), len(bad)))
    for f in bad[:4]:
        print("   unreadable: %s" % os.path.basename(f))
    print("index dtypes present, over all readable files: %r" % dtypes)
    print("   a single dtype here is what makes nnz exact; a mix would have made the old")
    print("   size-division wrong on part of the corpus and right on the rest.")
    if bad:
        print("EXIT: %d files unreadable. The denominator for any share below would be the")
        print("      READABLE count, not the file count, and publishing the file count")
        print("      while computing the readable one is only correct when they coincide."
              % len(bad))
        return 1

    n = len(got)
    print("\nthe gate alone (nnz * max(k,16) < %d), and COMPOSED with the output-bytes"
          % THRESHOLD)
    print("predicate (rows * k * 4 >= %d excludes), llc = %d:" % (maxout, llc))
    print("   %-5s %-14s %-14s %-10s %s" % ("k", "gate alone", "composed", "share", "removed"))
    regions = {}
    for k in KS:
        gate = [(f, nz, rw) for f, nz, rw in got if nz * max(k, 16) < THRESHOLD]
        comp = [(f, nz, rw) for f, nz, rw in gate if rw * k * 4 < maxout]
        regions[k] = comp
        print("   k=%-3d %-14d %-14d %-10s %d"
              % (k, len(gate), len(comp), "%.1f%%" % (100.0 * len(comp) / n), len(gate) - len(comp)))
    print("   denominator is the READABLE count, %d, which equals the file count here" % n)

    rmax = max(rw for _, _, rw in got)
    print("\n   largest row count in the corpus: %d. The output-bytes predicate needs" % rmax)
    print("   rows >= %d at k=16 and %d at k=32, so it is %s on this corpus."
          % (maxout // 4 // 16, maxout // 4 // 32,
             "LIVE" if rmax * 32 * 4 >= maxout else "vacuous"))

    if out:
        k_emit = 16
        with open(out, "w") as fh:
            for f, nz, rw in sorted(regions[k_emit]):
                fh.write("%s\t%d\t%d\n" % (f, nz, rw))
        print("\nwrote %d members of the composed k=%d region to %s"
              % (len(regions[k_emit]), k_emit, out))
        print("   the earlier version computed this path and returned without writing it,")
        print("   so the script that identifies the region never emitted the region.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
