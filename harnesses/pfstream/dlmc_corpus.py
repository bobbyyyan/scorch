"""Build a STRATIFIED DLMC key list, and report the strata so the sample is auditable.

WHY STRATIFY RATHER THAN SAMPLE FLAT. DLMC is a factorial design already: model x pruning method
x sparsity x layer. The layer counts are wildly unequal across strata, so a flat random or
first-N sample would load the corpus onto whichever model and sparsity happen to have the most
layers, and a kernel tuned against that sample would be tuned against one corner of the design.
Sparsity in particular is the axis a sparse kernel is most likely to behave non-monotonically in,
so it must be represented evenly rather than in proportion to layer count.

SELECTION WITHIN A STRATUM IS EVENLY SPACED BY NONZERO COUNT after sorting, not random:
reproducible without carrying a seed, and it spans the stratum instead of clustering in it. The
same rule the SuiteSparse band sampler uses.

WHAT THIS PRINTS. Every stratum with its pool size and how many it took, plus the strata it could
not fill. A sample is only auditable if the shortfalls are visible: a stratum that contributed
zero looks identical to one that does not exist unless the enumeration says otherwise.
"""
import collections
import os
import sys


def strata(root):
    """(model, method, sparsity) -> [(key, path)]. Keys are cache-relative, without .npz."""
    out = collections.defaultdict(list)
    skipped = collections.Counter()
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if not f.endswith(".npz"):
                skipped["not npz"] += 1
                continue
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, root)
            parts = rel.split(os.sep)
            if len(parts) != 4:
                # model/method/sparsity/layer.npz is the expected shape; anything else is not
                # placed in the design and is counted rather than silently dropped.
                skipped["unexpected depth %d" % len(parts)] += 1
                continue
            model, method, sparsity, leaf = parts
            key = "%s/%s/%s/%s" % (model, method, sparsity, os.path.splitext(leaf)[0])
            out[(model, method, sparsity)].append((key, full))
    return out, skipped


def nnz_of(path):
    import numpy as np
    try:
        z = np.load(path)
        return int(z["indices"].shape[0])
    except Exception:
        return None


def main(argv):
    if len(argv) < 3:
        return "usage: dlmc_corpus.py <cache-root> <out-keys> [per_stratum]"
    root, out = argv[1], argv[2]
    per = int(argv[3]) if len(argv) > 3 else 8
    if not os.path.isdir(root):
        return "REFUSE: %s is not a directory" % root
    st, skipped = strata(root)
    total = sum(len(v) for v in st.values())
    print("cache %s" % root)
    print("files placed in the design: %d across %d strata; skipped %s"
          % (total, len(st), dict(skipped) or "{}"))
    if not st:
        return "REFUSE: no stratum was formed, so the key list would be empty"

    models = sorted({m for (m, _me, _s) in st})
    methods = sorted({me for (_m, me, _s) in st})
    sparsities = sorted({s for (_m, _me, s) in st}, key=lambda x: float(x))
    print("models %s" % models)
    print("methods %s" % methods)
    print("sparsities %s" % sparsities)
    print("design cells %d x %d x %d = %d; populated %d"
          % (len(models), len(methods), len(sparsities),
             len(models) * len(methods) * len(sparsities), len(st)))

    picked, short, empty, degen = [], [], [], []
    print()
    print("  %-12s %-22s %-6s %6s %5s %6s %8s" % ("model", "method", "sparse", "pool", "took",
                                                   "zero", "min nnz"))
    for key in sorted(st, key=lambda k: (k[0], k[1], float(k[2]))):
        pool = st[key]
        # order by nonzero count so the spacing below spans the stratum's size range
        sized = [(nnz_of(p), k, p) for k, p in pool]
        unreadable = [x for x in sized if x[0] is None]
        # DEGENERATE MATRICES ARE DROPPED AND COUNTED. nnz_of returns None only on an exception,
        # so a matrix with zero nonzeros returns 0 and passes an `is not None` filter. That
        # interacts badly with the selection below: the spacing's i=0 term is round(0) = 0 for
        # every pool size, so the MINIMUM-nnz member of every stratum is picked unconditionally
        # and ties at the minimum sort to the front. A stratum containing zero-nonzero files
        # would therefore contribute them first, every time. This cache has none -- they were
        # excluded upstream, which is why one stratum's pool is 71 where its siblings are 97 --
        # but the filter must not depend on that.
        degenerate = [x for x in sized if x[0] is not None and x[0] <= 0]
        usable = sorted([x for x in sized if x[0] is not None and x[0] > 0])
        if unreadable:
            short.append((key, len(unreadable)))
        if degenerate:
            degen.append((key, len(degenerate)))
        if not usable:
            empty.append(key)
            print("  %-12s %-22s %-6s %6d %5s %6d %8s"
                  % (key[0], key[1], key[2], len(pool), "0 !", len(degenerate), "-"))
            continue
        take = min(per, len(usable))
        idx = ([round(i * (len(usable) - 1) / max(1, take - 1)) for i in range(take)]
               if take > 1 else [0])
        sel = [usable[i] for i in sorted(set(idx))]
        picked += [k for _n, k, _p in sel]
        # The dropped count sits beside the pool size on purpose: a stratum that loses 26 of 97
        # members has a different effective n from its neighbours, and a summary that reports
        # only "took 8" conceals that.
        print("  %-12s %-22s %-6s %6d %5d %6d %8d"
              % (key[0], key[1], key[2], len(usable), len(sel), len(degenerate), sel[0][0]))

    print()
    if short:
        print("  strata with unreadable files: %d (%s)" % (len(short), short[:5]))
    if empty:
        print("  EMPTY strata: %d -- no claim about those design cells is supported: %s"
              % (len(empty), empty[:6]))
    if degen:
        print("  strata containing ZERO-NONZERO files, dropped: %d (%s)" % (len(degen), degen[:5]))
    if not short and not empty and not degen:
        print("  every stratum contributed; no unreadable and no zero-nonzero files")
    print("  ^ 'every stratum contributed' is about STRATA. The zero column above is about")
    print("    MATRICES, and an earlier version could only report the former -- so it printed")
    print("    'no shortfalls' in exactly the case where a degenerate matrix had been selected.")
    # Duplicate basenames among the selected keys would collide in a flat cache. This cache is
    # NOT flat, but the check is cheap and the failure it guards against was real on the other
    # host, where exactly one such collision silently served one key another key's matrix.
    base = collections.Counter(os.path.basename(k) for k in picked)
    dup = {n: c for n, c in base.items() if c > 1}
    print("  selected keys %d; distinct basenames %d; basenames shared by >1 key %d"
          % (len(picked), len(base), len(dup)))
    print("  ^ this cache is model/method/sparsity-qualified, so a shared basename is expected")
    print("    and harmless HERE; it is fatal only in a flat cache, which is why it is counted.")
    with open(out, "w") as fh:
        for k in sorted(picked):
            fh.write(k + "\n")
    print("  wrote %d keys -> %s" % (len(picked), out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
