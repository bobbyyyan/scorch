#!/usr/bin/env python3
"""The column count J per chain210 key, and the covariate structure a chain211 fit depends on.

WHY THIS EXISTS. The chain211 board's header is
`build,slice,key,k,rows,nnz,deg,arm,ms,batch` -- there is no `cols`. So B's footprint, which is
`4 * k * J` bytes for a dense B, cannot be computed from the board at all, and any footprint
hypothesis is untestable until J is joined in from the collection.

WHY J IS NOT OPTIONAL EVEN IF FOOTPRINT IS NOT THE HYPOTHESIS. Two candidate mechanisms for the
deep kernel's regression load the SAME regressor with the SAME sign:

  per-ROW cost (accumulator init, deeper fold)   predicts slope(log nnz) < 0
  footprint / traffic                            predicts the ratio falls as footprint rises,
                                                 and footprint is proportional to J at fixed k,
                                                 and log J correlates +0.41 with log nnz

so in a fit of y on {log nnz, log rows} the footprint signal arrives almost entirely through the
log(nnz) partial -- the same coefficient the per-row family predicts to be negative. The design as
first registered cannot separate them, and whichever hypothesis the analyst holds when reading the
number will look confirmed. Adding log J separates them on a coefficient neither shares: the
per-row family predicts slope(log J) = 0, the footprint family predicts slope(log J) < 0.

That addition is legitimate and this file checks that it is, rather than assuming it. `degree` is
exactly `nnz/rows`, so {log nnz, log rows, log degree} has rank two and must never be fitted
together; J is NOT determined by nnz and rows, and the rank check below demonstrates it.

The figures are reported with their shape, because a slope on a covariate with six effective
levels and two thirds of its mass at one of them is not a slope with n=3820 behind it.
"""
import collections, csv, os, sys
import numpy as np

HOME = os.path.expanduser("~")
keys_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/k211_all.txt"
ref = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    HOME, ".cache/scorch-codex/verifier-dlmc-dims/dlmc_dims.csv")

keys = [l.strip() for l in open(keys_path) if l.strip()]
dims, unresolved = {}, []
for key in keys:
    f = os.path.join(HOME, key + ".smtx")
    if not os.path.exists(f):
        unresolved.append(key)
        continue
    with open(f) as h:
        r, c, n = (int(x) for x in h.readline().split(","))
    dims[key] = (r, c, n)
print("keys %d   resolved from ~/dlmc %d   UNRESOLVED %d" % (len(keys), len(dims), len(unresolved)))
if unresolved:
    sys.exit("REFUSE: %d key(s) have no .smtx; a partial join would silently drop them from "
             "every fit: %s" % (len(unresolved), " ".join(unresolved[:5])))

# ---- two-copy check against an independently obtained copy, if one is present.
# Two copies agreeing on 3 integers x every key is something a substituted or regenerated
# collection cannot do; a consistency check on one copy cannot distinguish those cases.
if os.path.exists(ref):
    other = {}
    with open(ref) as h:
        rdr = csv.reader(l for l in h if not l.startswith("#"))
        ix = {n: i for i, n in enumerate(next(rdr))}
        for row in rdr:
            other[row[ix["key"]]] = (int(row[ix["rows"]]), int(row[ix["cols"]]),
                                     int(row[ix["nnz"]]))
    both = set(dims) & set(other)
    dis = sorted(k for k in both if dims[k] != other[k])
    print("TWO-COPY vs %s" % os.path.basename(ref))
    print("  %d keys x 3 integers = %d compared; disagreements %d; mine-only %d; theirs-only %d"
          % (len(both), 3 * len(both), len(dis), len(set(dims) - set(other)),
             len(set(other) - set(dims))))
    if dis:
        for k in dis[:5]:
            print("    %s mine=%s theirs=%s" % (k, dims[k], other[k]))
        sys.exit("REFUSE: the two copies disagree; one of them is not this collection")
else:
    print("TWO-COPY: no reference copy at %s -- single-copy only, provenance NOT established" % ref)

rows = np.array([v[0] for v in dims.values()], float)
cols = np.array([v[1] for v in dims.values()], float)
nnz = np.array([v[2] for v in dims.values()], float)
lr, lc, ln = np.log(rows), np.log(cols), np.log(nnz)
ld = ln - lr                                      # log degree, exactly, by construction

J = collections.Counter(int(x) for x in cols)
tot = sum(J.values())
print()
print("J (column count) takes %d distinct values over the board:" % len(J))
print("  " + "  ".join("%d:%d" % (j, J[j]) for j in sorted(J)))
big = [j for j in J if J[j] >= 100]
top = J.most_common(1)[0]
print("  J=%d carries %d of %d = %.1f%%;  levels with >=100 keys: %d;  square (rows==cols): %.1f%%"
      % (top[0], top[1], tot, 100 * top[1] / tot, len(big),
         100 * float((rows == cols).sum()) / tot))
print("  SO a J or footprint slope is a %d-point contrast with %.0f%% of the mass at one point,"
      % (len(big), 100 * top[1] / tot))
print("  and the levels track DLMC's layer shapes, so it is partly a contrast across families.")

corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
print()
print("Pearson on logs:  log J vs log rows %+.4f   log nnz %+.4f   log degree %+.4f"
      % (corr(lc, lr), corr(lc, ln), corr(lc, ld)))
print("  log J is near-orthogonal to log rows and correlates with log nnz, which is exactly why")
print("  the footprint signal would arrive through the log(nnz) partial if J were omitted.")

print()
X2 = np.column_stack([ln, lr, np.ones_like(ln)])
_, res, _, _ = np.linalg.lstsq(X2, lc, rcond=None)
r3 = np.linalg.matrix_rank(np.column_stack([ln, lr, lc, np.ones_like(ln)]))
r_bad = np.linalg.matrix_rank(np.column_stack([ln, lr, ld, np.ones_like(ln)]))
print("RANK CHECKS, so the added regressor is demonstrated and not assumed:")
print("  residual SS of log J on {log nnz, log rows, 1} = %.1f  (0 would mean collinear)"
      % (float(res[0]) if len(res) else 0.0))
print("  rank{log nnz, log rows, log J,      1} = %d  <- usable, J adds a dimension" % r3)
print("  rank{log nnz, log rows, log degree, 1} = %d  <- DEFICIENT, degree is nnz/rows exactly"
      % r_bad)

st = [k for k in dims if dims[k][0] == 512]
sj = collections.Counter(dims[k][1] for k in st)
print()
print("STRATIFIED ALTERNATIVE, which needs no partialling: rows==512 holds %d keys = %.1f%% of"
      % (len(st), 100 * len(st) / tot))
print("  the board, with rows constant by construction, so the J slope is read directly.")
print("  its J levels: " + "  ".join("%d:%d" % (j, sj[j]) for j in sorted(sj)))
dom = sorted(sj.items(), key=lambda t: -t[1])[:2]
print("  effectively a %d-versus-%d contrast (%d and %d keys) with small wings -- state that"
      % (dom[0][0], dom[1][0], dom[0][1], dom[1][1]))
print("  beside any slope estimated there.")
