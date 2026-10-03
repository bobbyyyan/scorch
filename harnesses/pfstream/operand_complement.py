"""Is `min(shape) == 1` a COMPLETE operand predicate for the SuiteSparse cache?

Showing that a filename match misses most non-operators does not show that the shape
predicate catches all of them. The complement -- an entry that is not an SpMM operator and
still has min(shape) > 1 -- needs its own test. A multi-input state-space B is n x m and a
multi-output C is m x n, and neither is an operator while both survive the shipped check.

Three tests, and only the first uses a list. A typed list of suffixes to grep for can
only catch what the author thought to list, and this one is also OVER-inclusive: a square
matrix is an operator whatever its suffix, so `_E` descriptor-system mass matrices and
square `_B` input matrices are flagged by it and should not be. The two tests that carry
the conclusion use no list at all.

THE BRACKET VARIES ONE THING. An earlier version swept `min(shape) >= c` over the whole
corpus, which also discards every SQUARE matrix below c x c -- operators this file argues
are legitimate, and small ones, so they sit preferentially inside the small-nnz region
being measured. That span answers "how much do the share and the operator definition move
together", not "how much does the incompleteness move the share". Here the cutoff applies
only to NON-SQUARE entries and the square count the naive sweep would have removed is
printed beside each row, so the two mechanisms are separate and visible.
"""
import collections
import glob
import os
import sys
import zipfile

import numpy as np

GRAIN, MINGRAINS = 150000, 2
THRESH = MINGRAINS * GRAIN
SUF = ("b", "B", "C", "E", "iv", "x", "rhs")


def dims(path):
    """(nnz, rows, shape) from the .npy headers. shape is the VALUE, read not inferred."""
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if "indices.npy" not in names or "indptr.npy" not in names:
                return None
            h = {}
            for m in ("indices.npy", "indptr.npy"):
                with z.open(m) as fh:
                    if np.lib.format.read_magic(fh) != (1, 0):
                        return None
                    h[m] = np.lib.format.read_array_header_1_0(fh)[0]
            if "shape.npy" not in names:
                return None
            with z.open("shape.npy") as fh:
                shape = tuple(int(x) for x in np.lib.format.read_array(fh))
    except Exception:                                                  # noqa: BLE001
        return None
    if len(shape) != 2:
        return None
    return int(h["indices.npy"][0]), int(h["indptr.npy"][0]) - 1, shape


def suffix(name):
    return name.rsplit("_", 1)[-1] if "_" in name else None


def main(root, llc):
    files = sorted(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
    print("cache %s: %d .npz files" % (root, len(files)))
    got, bad = [], []
    for p in files:
        d = dims(p)
        if d is None:
            bad.append(os.path.basename(p))
            continue
        got.append((os.path.basename(p)[:-4], d[0], d[1], d[2]))
    print("read %d, unreadable-or-no-shape %d%s"
          % (len(got), len(bad), ("  e.g. " + ", ".join(bad[:3])) if bad else ""))
    if not got or len(got) < 0.99 * len(files):
        return 1 if print("REFUSE: read %d of %d. A partial scan answers a different "
                          "question." % (len(got), len(files))) is None else 1

    sq = [g for g in got if g[3][0] == g[3][1]]
    nsq = [g for g in got if g[3][0] != g[3][1]]
    deg = [g for g in got if min(g[3]) == 1]
    print("square %d   non-square %d   degenerate in one axis %d"
          % (len(sq), len(nsq), len(deg)))
    assert len(sq) + len(nsq) == len(got)
    # A 1 x 1 entry would be square AND degenerate, so this identity is also the check
    # that no such entry exists; it prints a visible non-identity if one ever appears.
    print("every degenerate entry is non-square: %s"
          % ("yes" if all(g[3][0] != g[3][1] for g in deg) else "NO -- a 1x1 entry exists"))

    # ------------------------------------------------------------- 1, the typed list
    hits = [g for g in got if min(g[3]) > 1 and suffix(g[0]) in SUF]
    hsq = [g for g in hits if g[3][0] == g[3][1]]
    print("\n1. TYPED LIST (and it is both incomplete AND over-inclusive).")
    print("   entries with a suffix in %r and min(shape) > 1: %d" % (SUF, len(hits)))
    print("   of those SQUARE, hence operators and wrongly flagged: %d" % len(hsq))
    for s, n in collections.Counter(suffix(g[0]) for g in hits).most_common():
        fam = [g for g in hits if suffix(g[0]) == s]
        nq = sum(1 for g in fam if g[3][0] != g[3][1])
        print("     _%-4s %4d  (%3d non-square, %3d square)  e.g. %s"
              % (s, n, nq, n - nq,
                 ", ".join("%s %s" % (e[0], e[3]) for e in fam[:2])))
    print("   This is why the list is not the test: it misses non-operators under suffixes")
    print("   not listed, and flags square operators that are listed. Tests 2 and 3 do not")
    print("   use it.")

    # ----------------------------------------------- 2, list-free: thin non-square
    print("\n2. LIST-FREE. min(shape) over the %d NON-SQUARE entries. A non-operator that"
          % len(nsq))
    print("   survives the degenerate test is THIN, so incompleteness shows up as a")
    print("   population just above 1. A jump from 1 to the hundreds would mean nothing")
    print("   survives for a wider predicate to catch.")
    BANDS = [(1, 1), (2, 4), (5, 16), (17, 64), (65, 256), (257, 1 << 60)]
    tot = 0
    for lo, hi in BANDS:
        sel = [g for g in nsq if lo <= min(g[3]) <= hi]
        tot += len(sel)
        lab = "%d" % lo if lo == hi else ("%d-%d" % (lo, hi) if hi < (1 << 60) else ">=%d" % lo)
        print("     min(shape) %-8s %4d   %s"
              % (lab, len(sel),
                 ", ".join("%s %s" % (e[0], e[3])
                           for e in sorted(sel, key=lambda g: min(g[3]))[:3])))
    print("     %s bands sum to %d, the non-square total %d"
          % ("OK," if tot == len(nsq) else "MISMATCH:", tot, len(nsq)))

    # -------------------------------------- 3, list-free: families mixing degenerate
    print("\n3. LIST-FREE. Base names holding BOTH a degenerate and a thin non-square")
    print("   member. That is the shape a sometimes-thin family makes and it needs no")
    print("   suffix list. Square members are not counted: they are operators.")
    fam = collections.defaultdict(list)
    for g in got:
        if "_" in g[0]:
            fam[g[0].rsplit("_", 1)[0]].append(g)
    mixed = [(b, [m for m in ms if min(m[3]) == 1],
              [m for m in ms if min(m[3]) > 1 and m[3][0] != m[3][1]])
             for b, ms in fam.items()]
    mixed = [m for m in mixed if m[1] and m[2]]
    print("   families with >=2 members %d   MIXED %d%s"
          % (sum(1 for v in fam.values() if len(v) > 1), len(mixed),
             "   <- zero would retract tests 1-3" if not mixed else ""))
    for b, d, t in sorted(mixed)[:6]:
        print("     %-20s degenerate %s | thin %s"
              % (b, ", ".join("%s%s" % (x[0], x[3]) for x in d[:1]),
                 ", ".join("%s%s" % (x[0], x[3]) for x in t[:2])))

    # ------------------------------------------- the bracket, varying ONE thing
    maxout = 2 * llc

    def inregion(g, k):
        return g[1] * max(k, 16) < THRESH and g[2] * k * 4 < maxout

    print("\n4. THE SENSITIVITY, APPLIED TO NON-SQUARE ENTRIES ONLY. Cutoff c drops an entry")
    print("   when it is non-square AND min(shape) < c. Square entries are never dropped,")
    print("   because a square matrix is an operator whatever its size or suffix. The last")
    print("   column is the number of SQUARE entries a corpus-wide `min(shape) >= c` sweep")
    print("   would also have removed -- the contamination this restriction takes out.")
    print()
    print("   %-5s %-9s %-21s %-21s %s"
          % ("c", "operands", "k<=16 region / share", "k=32 region / share", "sq. spared"))
    rows = []
    for c in (2, 3, 5, 17, 65, 257):
        ops = [g for g in got if g[3][0] == g[3][1] or min(g[3]) >= c]
        spared = sum(1 for g in sq if min(g[3]) < c)
        r16 = [g for g in ops if inregion(g, 16)]
        r32 = [g for g in ops if inregion(g, 32)]
        s16, s32 = 100.0 * len(r16) / len(ops), 100.0 * len(r32) / len(ops)
        rows.append((c, s16, s32))
        print("   %-5d %-9d %5d / %7.3f%%        %5d / %7.3f%%        %d"
              % (c, len(ops), len(r16), s16, len(r32), s32, spared))
    sp16 = max(r[1] for r in rows) - min(r[1] for r in rows)
    sp32 = max(r[2] for r in rows) - min(r[2] for r in rows)
    print()
    print("   span over a %d-fold range of c: %.2f pp at k<=16 and %.2f pp at k=32."
          % (257 // 2, sp16, sp32))
    print("   The denominator MOVES with c here (an excluded entry leaves both numerator")
    print("   and denominator), which is stated because the two conventions differ by over")
    print("   a point at the widest cutoff and a share alone does not say which was used.")
    print("   Reported at c = 2: it is the only cutoff that needs no chosen constant.")

    # -------------------------------------- the implied property behind the 23
    onerow = [g for g in got if g[3][0] == 1]
    mx = max(g[1] for g in onerow)
    print("\n5. THE PROPERTY THE ONE-ROW COUNT RESTS ON. All %d one-row entries leaving the"
          % len(onerow))
    print("   k=32 region REQUIRES nnz < %d for each (%d*32 = %d). Not free: a dense-row _C"
          % (THRESH // 32, THRESH // 32, THRESH))
    print("   would fail it -- LF10000_C is 1 x 19998, and a dense row gives nnz*32 = %d."
          % (19998 * 32))
    print("   max nnz among the %d: %d   %s"
          % (len(onerow), mx,
             "all below %d, so the count holds" % (THRESH // 32) if mx < THRESH // 32
             else "NOT all below -- the k=32 count is WRONG"))
    for g in sorted(onerow, key=lambda x: -x[1])[:3]:
        print("     %-18s shape %-15s nnz %5d  (dense row would be %d)"
              % (g[0], str(g[3]), g[1], g[3][1]))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: operand_complement.py <cache-root> [llc-bytes]")
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 33554432))
