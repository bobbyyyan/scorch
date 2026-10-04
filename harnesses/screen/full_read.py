#!/usr/bin/env python3
"""Read a sliced full-collection A/B (slice_run.py CSVs named <corpus>_<slice>_<arm>.csv).

Completeness is checked by IDENTITY: every key in keys_<corpus>.txt must appear at every width in
both arms, or be recorded as unresolved by both; anything else is listed. The A/A is free: cells
the policy runs on one worker execute identical code in both builds (the change touches only the
multi-worker row handout), so their ratio distribution is the noise against which the cells where
the change applies are read.

usage: full_read.py <run_dir> <base_arm> <cand_arm> [ks]     (run in an allocation the size of the
measured one: the policy resolver reads the host's processor count)"""
import csv, glob, math, os, sys, collections
import torch, scorch_ops as so
R, base, cand = sys.argv[1:4]; KS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "2,3,4,8,16,32").split(",")]
P = torch.get_num_threads()
data = {base: {}, cand: {}}; unresolved = {base: set(), cand: set()}
for f in glob.glob(os.path.join(R, "csv", "*.csv")):
    arm = os.path.basename(f)[:-4].rsplit("_", 1)[1]
    if arm not in data: continue
    for r in csv.DictReader(open(f)):
        if r["arm"] == "unresolved": unresolved[arm].add((r["corpus"], r["key"])); continue
        data[arm].setdefault((r["corpus"], r["key"], int(r["k"])), {"rows": int(r["rows"]), "nnz": int(r["nnz"])})[r["arm"]] = float(r["us"])
missing = []
for corpus in ("ss", "dlmc"):
    keys = [l.strip() for l in open(os.path.join(R, "keys_%s.txt" % corpus)) if l.strip()]
    for key in keys:
        for arm in (base, cand):
            if (corpus, key) in unresolved[arm]: continue
            for k in KS:
                v = data[arm].get((corpus, key, k))
                if v is None or "ours" not in v or "ours_cold" not in v: missing.append((arm, corpus, key, k))
    print("%-4s keys %d | unresolved %s %d, %s %d" % (corpus, len(keys), base, sum(c == corpus for c, _ in unresolved[base]), cand, sum(c == corpus for c, _ in unresolved[cand])))
print("missing (arm, cell) pairs: %d%s" % (len(missing), "  e.g. " + str(missing[:3]) if missing else ""))
gm = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] if xs else float("nan")
for reg, arm in (("warm", "ours"), ("cold", "ours_cold")):
    print("==== %s: %s time / %s time (>1 = %s faster)" % (reg, base, cand, cand))
    for corpus in ("ss", "dlmc"):
        for k in KS:
            fire, aa = [], []
            for x, v in data[cand].items():
                if x[0] != corpus or x[2] != k or x not in data[base]: continue
                b = data[base][x]
                if arm not in v or arm not in b: continue
                nnz, rows = v["nnz"], v["rows"]
                nt = so.scorch_spmm_nthreads(nnz * max(k, 16), rows, P, nnz * k, nnz)
                applies = nt > 1 and so.scorch_spmm_partition_mode(rows, nnz, k, k, 4, nt, P) == 3
                (fire if applies else aa).append(b[arm] / v[arm])
            print("%-4s k=%-2d applies n=%4d gmean %.3f p1 %.3f slower>5%% %4.1f%% >10%% %4.1f%% | identical code n=%4d gmean %.3f p1 %.3f slower>5%% %4.1f%% >10%% %4.1f%%" % (
                corpus, k, len(fire), gm(fire), q(fire, .01), 100.0 * sum(r < .95 for r in fire) / max(len(fire), 1), 100.0 * sum(r < .9 for r in fire) / max(len(fire), 1),
                len(aa), gm(aa), q(aa, .01), 100.0 * sum(r < .95 for r in aa) / max(len(aa), 1), 100.0 * sum(r < .9 for r in aa) / max(len(aa), 1)))
