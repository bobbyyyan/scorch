#!/usr/bin/env python3
"""Draw the fixed screening sample: N keys per collection, once, before any candidate is tried.

A candidate kernel is first timed against MKL on this sample (minutes), and only a candidate
that wins almost everywhere here earns a full-collection run, which then serves as the held-out
check that the candidate was not fitted to the sample.

Method: systematic sampling from a sorted frame (implicit stratification). Within each
collection the frame is every key in the cache, sorted by (floor(log2 nonzeros-per-row),
nonzeros, key); the sample is the evenly spaced positions floor((i + 0.5) * N_pop / N). No seed,
so the draw is reproducible from the frame alone. Allocation across degree bins is proportional.

Features (rows, nnz) come from a prior full-collection CSV that timed every key; a key without
features is a REFUSAL, not a skip, because a silently smaller frame changes the sample.

usage: screen_draw.py <cache_dir> <features_csv_dir> <out_dir> [n_per_collection]
"""
import csv, glob, math, os, sys

cache, featdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 100
os.makedirs(out, exist_ok=True)

def keys_from_cache(d):
    ks = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".npz"):
                ks.append(os.path.relpath(os.path.join(root, f), d)[:-4])
    return sorted(ks)

def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            rows.extend(csv.DictReader(fh))
    return rows

def ratios(rows, keyset=None):
    """mkl_ms / ours_ms per (key, k); >1 means we are faster."""
    t = {}
    for r in rows:
        if keyset is not None and r["key"] not in keyset:
            continue
        t.setdefault((r["key"], int(r["k"])), {})[r["arm"]] = float(r["ms"])
    return {kk: v["mkl"] / v["ours"] for kk, v in t.items() if "mkl" in v and "ours" in v}

summary = []
for corpus in ("ss", "dlmc"):
    pop = keys_from_cache(os.path.join(cache, corpus))
    paths = sorted(glob.glob(os.path.join(featdir, "%s_float32_ship_s*.csv" % corpus)))
    if not pop:
        sys.exit("REFUSE: %s cache yields 0 keys" % corpus)
    if not paths:
        sys.exit("REFUSE: no feature CSVs for %s in %s" % (corpus, featdir))
    rows = load_rows(paths)
    feat = {}
    for r in rows:
        feat[r["key"]] = (int(r["rows"]), int(r["nnz"]))
    missing = [k for k in pop if k not in feat]
    extra = [k for k in feat if k not in set(pop)]
    print("%s: frame %d keys from cache, features for %d, missing %d, features not in cache %d"
          % (corpus, len(pop), len(feat), len(missing), len(extra)))
    if missing:
        sys.exit("REFUSE: %d %s keys lack features, e.g. %s" % (len(missing), corpus, missing[:3]))

    def dbin(k):
        m, nnz = feat[k]
        return math.floor(math.log2(nnz / m)) if nnz > 0 and m > 0 else -99

    frame = sorted(pop, key=lambda k: (dbin(k), feat[k][1], k))
    P = len(frame)
    idx = [int((i + 0.5) * P / N) for i in range(N)]
    if len(set(idx)) != N:
        sys.exit("REFUSE: duplicate positions; N=%d exceeds what the frame supports" % N)
    sample = [frame[i] for i in idx]
    with open(os.path.join(out, "screen_%s.txt" % corpus), "w") as fh:
        fh.write("\n".join(sample) + "\n")

    # Coverage: degree-bin counts, population vs sample.
    pb, sb = {}, {}
    for k in frame: pb[dbin(k)] = pb.get(dbin(k), 0) + 1
    for k in sample: sb[dbin(k)] = sb.get(dbin(k), 0) + 1
    print("  log2(nnz/row) bin: pop -> sample")
    for b in sorted(pb):
        print("    %4d: %5d -> %3d   (proportional %.1f)" % (b, pb[b], sb.get(b, 0), pb[b] * N / P))

    # Representativeness against the prior full-collection timing: the share of cells behind MKL
    # in the sample must sit near the share in the population, or the sample is not a screen.
    rp, rs = ratios(rows), ratios(rows, set(sample))
    print("  behind MKL (mkl/ours < 1), prior full-collection ship timing: population vs sample")
    for k in sorted({kk[1] for kk in rp}):
        p = [v for (key, kk), v in rp.items() if kk == k]
        s = [v for (key, kk), v in rs.items() if kk == k]
        gp = math.exp(sum(map(math.log, p)) / len(p)); gs = math.exp(sum(map(math.log, s)) / len(s))
        print("    k=%-2d pop n=%4d behind %5.1f%% gmean %.3f | sample n=%3d behind %5.1f%% gmean %.3f"
              % (k, len(p), 100 * sum(v < 1 for v in p) / len(p), gp,
                 len(s), 100 * sum(v < 1 for v in s) / len(s), gs))
print("DRAW DONE")
