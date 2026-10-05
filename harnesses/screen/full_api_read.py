#!/usr/bin/env python3
"""Read a sliced full-collection board of scorch.matmul against MKL (full_api.py CSVs under
<run_dir>/csv, any file names).

Completeness is checked by IDENTITY against keys_<corpus>.txt: every key must appear at every width
with all four timings (api, mkl, api_cold, mkl_cold), or be recorded as unresolved, or as an answer
mismatch at that width. Anything else is listed, and the board is not read until it is empty.

Then per corpus and width, back to back and cold: the cells behind MKL (MKL time / ours < 1), the
geometric mean of that ratio, its 1st percentile, and the behind cells themselves, worst first.

usage: full_api_read.py <run_dir> [ks] [n_listed]
"""
import csv, glob, math, os, sys
R = sys.argv[1]
KS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4,8,16,32").split(",")]
NL = int(sys.argv[3]) if len(sys.argv) > 3 else 15
ARMS = ("api", "mkl", "api_cold", "mkl_cold")
cells, unresolved, mismatch = {}, set(), set()
for f in glob.glob(os.path.join(R, "csv", "*.csv")):
    for r in csv.DictReader(open(f)):
        key = (r["corpus"], r["key"])
        if r["arm"] == "unresolved": unresolved.add(key); continue
        if r["arm"] == "mismatch": mismatch.add(key + (int(r["k"]),)); continue
        cells.setdefault(key + (int(r["k"]),), {"rows": int(r["rows"]), "nnz": int(r["nnz"])})[r["arm"]] = float(r["us"])
missing = []
for corpus in ("ss", "dlmc"):
    keys = [l.strip() for l in open(os.path.join(R, "keys_%s.txt" % corpus)) if l.strip()]
    for key in keys:
        if (corpus, key) in unresolved: continue
        for k in KS:
            x = (corpus, key, k)
            if x in mismatch: continue
            v = cells.get(x)
            if v is None or any(a not in v for a in ARMS): missing.append(x)
    print("%-4s keys %d | unresolved %d | answer mismatches %d cells" % (
        corpus, len(keys), sum(c == corpus for c, _ in unresolved), sum(m[0] == corpus for m in mismatch)))
print("missing cells: %d%s" % (len(missing), "  e.g. " + str(missing[:3]) if missing else ""))
if mismatch: print("MISMATCHES:", sorted(mismatch)[:20])
if missing: sys.exit("incomplete board: not read")
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
for reg, o, m in (("back to back", "api", "mkl"), ("cold", "api_cold", "mkl_cold")):
    print("==== %s: MKL time / scorch.matmul time (>1 = scorch faster)" % reg)
    for corpus in ("ss", "dlmc"):
        for k in KS:
            xs = [x for x in cells if x[0] == corpus and x[2] == k and all(a in cells[x] for a in ARMS)]
            r = {x: cells[x][m] / cells[x][o] for x in xs}
            v = sorted(r.values())
            beh = sorted((x for x in xs if r[x] < 1), key=lambda x: r[x])
            print("%-4s k=%-2d n=%4d | behind %4d (%.2f%%) | gmean %.3f | p1 %.3f | behind by >5%%: %d" % (
                corpus, k, len(xs), len(beh), 100.0 * len(beh) / max(len(xs), 1), gm(v), v[int(0.01 * (len(v) - 1))] if v else float("nan"),
                sum(r[x] < 1 / 1.05 for x in beh)))
            for x in beh[:NL]:
                c = cells[x]
                print("      %.3f  %-48s rows %9d nnz %10d deg %7.1f | ours %9.1f mkl %9.1f us" % (
                    r[x], x[1][-48:], c["rows"], c["nnz"], c["nnz"] / max(c["rows"], 1), c[o], c[m]))
