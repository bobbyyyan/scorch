#!/usr/bin/env python3
"""Which cached SuiteSparse matrices sit in concern #2's gate region?

The gate forces partition_mode=0 when nnz*max(k,16) < 300000, so at k<=16 it needs
nnz < 18750. nnz is read from each .npz's ZIP DIRECTORY rather than by decompressing:
the uncompressed size of the `indices` member is 4 or 8 bytes per nonzero. That is an
inference, so it is cross-checked against `indptr` on a sample and the check is printed --
a size-derived nnz that disagrees with the array it claims to describe would silently
select the wrong matrices.
"""
import glob
import os
import sys
import zipfile

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/scr/bobbyy/mklsweep/csrcache/ss"
THRESH = 300000


def nnz_from_dir(path):
    """(nnz, rows) inferred from member sizes, or None."""
    try:
        with zipfile.ZipFile(path) as z:
            info = {os.path.splitext(n)[0]: z.getinfo(n).file_size for n in z.namelist()}
    except Exception:                                              # noqa: BLE001
        return None
    idx = info.get("indices")
    ptr = info.get("indptr")
    if idx is None or ptr is None:
        return None
    # numpy .npy members carry a header; it is small and fixed-ish, so divide the payload
    for isz in (4, 8):
        for psz in (4, 8):
            n, r = (idx - 128) // isz, (ptr - 128) // psz - 1
            if n > 0 and r > 0:
                return n, r, isz, psz
    return None


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*", "*", "*.npz")))
    print("cache %s: %d .npz files" % (ROOT, len(files)))
    if not files:
        print("EXIT: no matrices found")
        return 1

    got, bad = [], 0
    for f in files:
        r = nnz_from_dir(f)
        if r is None:
            bad += 1
            continue
        n, rows, isz, psz = r
        got.append((f, n, rows, isz))
    print("readable: %d   unreadable: %d" % (len(got), bad))

    # the cross-check: decompress a sample and compare
    import numpy as np
    sample = got[:: max(1, len(got) // 12)][:12]
    agree = 0
    print("\nCROSS-CHECK of the size-derived nnz against the actual arrays, %d matrices:"
          % len(sample))
    for f, n, rows, isz in sample:
        try:
            with np.load(f) as z:
                real = len(z["indices"])
                realr = len(z["indptr"]) - 1
        except Exception as exc:                                   # noqa: BLE001
            print("   %-44s could not load: %s" % (os.path.basename(f), exc))
            continue
        ok = (n == real)
        agree += ok
        print("   %-40s inferred %9d  actual %9d  rows %7d/%-7d %s"
              % (os.path.basename(f)[:40], n, real, rows, realr, "ok" if ok else "MISMATCH"))
    print("   %d of %d agree" % (agree, len(sample)))
    if agree != len(sample):
        print("EXIT: the size-derived nnz is not reliable, so the selection below would be")
        print("      a selection on the wrong quantity.")
        return 1

    region = [(f, n, r) for f, n, r in [(f, n, r) for f, n, r, _ in got] if n * 16 < THRESH]
    print("\nin the gate region (nnz < %d, so firing at every k <= 16): %d of %d = %.1f%%"
          % (THRESH // 16, len(region), len(got), 100.0 * len(region) / len(got)))
    region.sort(key=lambda x: -x[1])
    print("   largest few: %r" % [(os.path.basename(f)[:28], n, r) for f, n, r in region[:6]])
    print("   smallest few: %r" % [(os.path.basename(f)[:28], n, r) for f, n, r in region[-4:]])
    out = os.path.join(os.path.dirname(ROOT.rstrip("/")), "gate_region_keys.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
