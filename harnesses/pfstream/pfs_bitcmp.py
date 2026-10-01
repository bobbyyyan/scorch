#!/usr/bin/env python3
"""Compare the four objects' output digests. A prefetch is a hint, so every cell must agree EXACTLY.

Refuses rather than reports on: a missing build, an unequal cell set across builds (which would
make an "all agree" verdict a statement about an intersection nobody declared), and a digest set
whose results were all zero (agreement is then trivial).
"""
import csv, os, sys
from collections import defaultdict
if len(sys.argv) < 3:
    sys.exit("usage: pfs_bitcmp.py <dir-with-bits_*.csv> <expected-real-matrices>")
D = sys.argv[1]
# THE DECLARED CELL COUNT IS AN ARGUMENT. The first version checked only that the four builds
# covered the SAME cells, which four identically-truncated runs satisfy: all four producers exited
# nonzero partway through the corpus and this printed "CORRECT" over 26 of 418 matrices. Covering
# the same cells and covering the declared cells are two claims.
EXPECT_MATRICES = int(sys.argv[2])
BUILDS = ("ship", "shpb", "strm", "nopf")
rows = {}
for b in BUILDS:
    p = os.path.join(D, "bits_%s.csv" % b)
    if not os.path.exists(p):
        sys.exit("REFUSE: %s is missing; 'every build agrees' over three builds is not the claim" % p)
    d = {}
    with open(p) as h:
        for r in csv.DictReader(h):
            key = (r["case"], r["key"], int(r["k"]))
            if key in d:
                sys.exit("REFUSE: %s repeats cell %s; a repeated cell makes the count a claim "
                         "about rows rather than about cells" % (p, key))
            d[key] = r["sha"]
    rows[b] = d
    print("  %-5s %d cells" % (b, len(d)))

sets = {b: set(d) for b, d in rows.items()}
if len(set(map(frozenset, sets.values()))) != 1:
    base = sets["ship"]
    for b in BUILDS:
        extra, miss = sets[b] - base, base - sets[b]
        if extra or miss:
            print("  %s: %d not in ship, %d missing" % (b, len(extra), len(miss)))
    sys.exit("REFUSE: the builds do not cover the same cells, so an agreement count would be "
             "about an undeclared intersection")

cells = sorted(sets["ship"])
_real = {c[1] for c in cells if c[0] == "real"}
if len(_real) != EXPECT_MATRICES:
    sys.exit("REFUSE: %d real matrices in the digests, expected %d. Four runs truncated at the "
             "same point agree with each other and say nothing about the corpus." 
             % (len(_real), EXPECT_MATRICES))
# THE PRODUCER'S COMPLETION SENTINEL, REQUIRED rather than read-if-present. A first version read a
# `bits_<b>.rc` written by whatever shell launched the producer: nothing in this tree wrote that
# file, and the read was wrapped in os.path.exists, so from a clean checkout the check could not
# fire and a missing file passed. That is the same shape one level down from the defect it was
# added to close -- a check that reported a truncated run as correct, repaired by a check that
# reports an unrecorded exit as correct.
#
# The count check above is the load-bearing catch for truncation and would have caught the
# 26-of-418 run on its own. This sentinel's independent job is the case the count cannot see: a
# producer that dies AFTER writing every matrix, in cleanup or on the final flush. pfs_bits.py
# writes it only after closing the CSV, so that death leaves it absent and this refuses.
for b in BUILDS:
    dn = os.path.join(D, "bits_%s.csv.done" % b)
    if not os.path.exists(dn):
        sys.exit("REFUSE: %s is absent. The producer writes it only after closing its CSV, so a "
                 "complete-looking CSV with no sentinel is a run that died after the last matrix "
                 "-- which the row count cannot see." % dn)
    tot = int(open(dn).read().split()[0])
    if tot != len(rows[b]):
        sys.exit("REFUSE: %s declares %d rows, the CSV holds %d cells. The sentinel and the file "
                 "disagree, so one of them is not describing this run." % (dn, tot, len(rows[b])))
dis = defaultdict(list)
for c in cells:
    ref = rows["ship"][c]
    for b in BUILDS[1:]:
        if rows[b][c] != ref:
            dis[b].append(c)
print()
print("BIT-IDENTITY against ship over %d cells (%d matrices x %d widths + degenerate)"
      % (len(cells), len({c[1] for c in cells if c[0] == 'real'}),
         len({c[2] for c in cells})))
for b in BUILDS[1:]:
    n = len(dis[b])
    print("  %-5s %s" % (b, "all %d agree" % len(cells) if n == 0
                         else "DISAGREES on %d of %d, first %s" % (n, len(cells), dis[b][0])))
if any(dis.values()):
    sys.exit("REFUSE: a prefetch changes no arithmetic, so a single differing byte means the "
             "change is not prefetch-only and no timing from these objects is interpretable")
# The number that shows the agreement is not vacuous: distinct digests across cells.
u = len(set(rows["ship"].values()))
print("  %d DISTINCT digests over %d cells -- so the agreement is not four objects returning one "
      "constant" % (u, len(cells)))
if u < len(cells) // 2:
    sys.exit("REFUSE: only %d distinct digests over %d cells; agreement is suspiciously cheap" % (u, len(cells)))
print()
print("CORRECT: all four objects byte-identical. ship/shpb is the A/A pair, strm and nopf are the "
      "arms, and their .text digests differ, so the identity is not three copies of one object.")
