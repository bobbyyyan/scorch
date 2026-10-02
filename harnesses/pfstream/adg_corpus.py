#!/usr/bin/env python3
"""The ADOPT_GRAIN corpus. The selection rule is stated completely, because a partial statement
of it is not reproducible and because the filter below is load-bearing.

THE RULE, in full. Walk every .smtx under --root. Parse each header and keep the matrix only if
nnz > 0 and the file is self-consistent (indptr has rows+1 entries, indptr[-1] == header nnz,
len(indices) == nnz) -- a truncated file parses cleanly and reads as a sparser matrix, so a
corpus built without that check silently contains different matrices than its keys name. Group
the survivors by their ROW COUNT, which in DLMC takes eight distinct values and is the axis the
lever's row-ceiling clause reads. Within each group sort by nnz ascending and take the keys at
indices round(i*(n-1)/(m-1)) for i = 0..m-1, where m is --per-class, over THAT filtered and
sorted group. Deduplicate while preserving order. No seed, no randomness.

WHY ROW COUNT IS THE STRATUM. The adoption clause caps at rows/16 and the graded cap at
work_true/grain, so the two bind in different places and which one binds is a function of rows
and nnz jointly. Stratifying on rows and spanning nnz inside each stratum puts both regimes in
the corpus by construction rather than by luck -- including the wide-k small-nonzero corner,
where the cap narrows a team at a width we already beat MKL at.
"""
import argparse, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--per-class", type=int, default=9)
ap.add_argument("--out", required=True)
A = ap.parse_args()

rows_groups = {}
scanned = truncated = empty = 0
for dirpath, _, files in os.walk(A.root):
    for f in sorted(files):
        if not f.endswith(".smtx"):
            continue
        p = os.path.join(dirpath, f)
        scanned += 1
        try:
            with open(p) as fh:
                head = fh.readline(); ipt = fh.readline(); idx = fh.readline()
            r, c, n = (int(x) for x in head.replace(",", " ").split())
            indptr = np.array(ipt.split(), dtype=np.int64)
            nind = len(idx.split())
            if len(indptr) != r + 1 or int(indptr[-1]) != n or nind != n:
                truncated += 1
                continue
        except Exception:
            truncated += 1
            continue
        if n <= 0:
            empty += 1
            continue
        key = os.path.relpath(p, A.root)[:-len(".smtx")]
        rows_groups.setdefault(r, []).append((n, key))

print("scanned %d .smtx, %d rejected as truncated/unparseable, %d rejected as nnz<=0"
      % (scanned, truncated, empty))
if not rows_groups:
    sys.exit("REFUSE: no stratum was formed, so the corpus would be empty and every downstream "
             "table could only print zero.")
picked = []
for r in sorted(rows_groups):
    g = sorted(rows_groups[r])
    m = min(A.per_class, len(g))
    idxs = [0] if m == 1 else [round(i * (len(g) - 1) / (m - 1)) for i in range(m)]
    take = [g[i] for i in sorted(set(idxs))]
    print("  rows=%-7d pool %-5d take %-3d nnz %d..%d"
          % (r, len(g), len(take), take[0][0], take[-1][0]))
    picked.extend(k for _, k in take)
seen = set()
final = [k for k in picked if not (k in seen or seen.add(k))]
with open(A.out, "w") as fh:
    for k in final:
        fh.write(k + "\n")
print("wrote %d keys (%d before dedup) to %s" % (len(final), len(picked), A.out))
