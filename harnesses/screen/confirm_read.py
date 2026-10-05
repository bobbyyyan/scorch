#!/usr/bin/env python3
"""Read the board together with its confirmation passes (full_api.py re-run on every key that had
a cell behind MKL or a stalled reading).

Every reading pairs scorch.matmul with torch.sparse.mm in the same process and pass, so the
per-pass ratio MKL time / scorch time is the unit; a cell's verdict is the MEDIAN of its per-pass
ratios (board + confirmation passes), and how many of those passes read below 1 is reported beside
it. A reading where one side took over 1 ms and over five times the other is counted (MKL has genuinely slow cells, so not every one is a stall)
and listed, but it is still a reading: the median is what decides, so a single stall cannot put a
cell behind or take it out.

Cells the confirmation did not re-measure keep their single board reading; they were ahead in it.

usage: confirm_read.py <board_dir> <confirm_dir> [ks] [n_listed]
"""
import csv, glob, os, statistics as st, sys, collections
B, CD = sys.argv[1], sys.argv[2]
KS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3,4,8,16,32").split(",")]
NL = int(sys.argv[4]) if len(sys.argv) > 4 else 10
REG = (("back to back", "api", "mkl"), ("cold", "api_cold", "mkl_cold"))
def load(files):
    t = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            if r["arm"] in ("mismatch", "unresolved"): continue
            t.setdefault((r["corpus"], r["key"], int(r["k"])), {"rows": int(r["rows"]), "nnz": int(r["nnz"])})[r["arm"]] = float(r["us"])
    return t
board = load(glob.glob(os.path.join(B, "csv", "*.csv")))
passes = [load(glob.glob(os.path.join(CD, "*_p%d.csv" % p))) for p in (1, 2, 3)]
confirmed_keys = {x[:2] for P in passes for x in P}
print("board cells %d | re-measured keys %d | cells with all 3 confirmation readings %d" % (
    len(board), len(confirmed_keys), sum(all(x in P for P in passes) for x in board)))
gm = lambda v: __import__("math").exp(sum(map(__import__("math").log, v)) / len(v)) if v else float("nan")
for reg, o, m in REG:
    print("==== %s: MKL time / scorch.matmul time, median over passes (>1 = scorch faster)" % reg)
    for corpus in ("ss", "dlmc"):
        for k in KS:
            xs = [x for x in board if x[0] == corpus and x[2] == k and o in board[x] and m in board[x]]
            single_behind, med, stalls, listing = 0, {}, 0, []
            for x in xs:
                reads = [board[x]] + [P[x] for P in passes if x in P and o in P[x] and m in P[x]]
                rs = [v[m] / v[o] for v in reads]
                stalls += sum((v[o] > 1000 and v[o] > 5 * v[m]) or (v[m] > 1000 and v[m] > 5 * v[o]) for v in reads)
                single_behind += rs[0] < 1
                med[x] = st.median(rs)
                if med[x] < 1:
                    listing.append((med[x], sum(r < 1 for r in rs), len(rs), x, st.median(v[o] for v in reads), st.median(v[m] for v in reads)))
            listing.sort()
            print("%-4s k=%-2d n=%4d | behind: board alone %4d -> median %4d | of those, behind in every pass %4d | gmean %.3f | readings >5x apart, >1 ms: %d" % (
                corpus, k, len(xs), single_behind, len(listing), sum(b == n for _, b, n, *_ in listing), gm(list(med.values())), stalls))
            for r, b, n, x, to, tm in listing[:NL]:
                c = board[x]
                print("      %.3f  %d/%d  %-46s rows %8d nnz %9d deg %7.1f | ours %8.1f mkl %8.1f us" % (
                    r, b, n, x[1][-46:], c["rows"], c["nnz"], c["nnz"] / max(c["rows"], 1), to, tm))
