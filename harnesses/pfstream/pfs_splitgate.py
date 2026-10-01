"""Ask the shipped row-split gate whether it fires -- and refuse to answer if it cannot.

WHY THIS EXISTS, AND WHY IT LEADS WITH A REFUSAL. `scorch_spmm_split_seg` is exported so a
harness asks production's own rule instead of keeping a second copy that drifts. But the gate's
first line is

    if (min_imb <= 0 || rows <= 1 || !A1_pos) return 0;      scorch_policy.h:2502

and `SCORCH_SPMM_SPLIT_MIN_IMBALANCE` ships at 0L (scorch_policy.h:438). The env override that
could raise it sits inside `#ifdef SCORCH_TUNE_HOOKS` (2493-2495). So on a build without those
hooks the function returns 0 BEFORE it reads the indptr, for every matrix ever passed to it.

A zero from such a build does not mean "the split does not fire on these matrices". It means the
gate is compiled off -- the same thing reading the header tells you, and strictly weaker than a
corpus measurement, not stronger. Measured on redwood over three configurations:

    hooks build, env unset (the SHIPPED gate value)   0 of 3 synthetic cases fire
    hooks build, SCORCH_SPMM_SPLIT_MIN_IMBALANCE=1    2 of 3 fire, seg = 1972
    non-hooks build, same env set                     0 of 3 fire (the env is not compiled in)

The first configuration returns 0 on an input whose measured row imbalance is 63.0, which is what
makes the vacuity concrete rather than theoretical. `scorch_spmm_row_imbalance` IS live on a
non-hooks build -- it reported that 63.0 -- so the imbalance is always measurable; it is only the
GATE that is inert. See [[null_result_indistinguishable]] and [[guards_must_fire_on_their_own_input]]:
beside a number that could be vacuous, report the number that shows it was not.

USAGE
    python pfs_splitgate.py <tree-root> [cache-dir keyfile]

The tree root is argv[1] rather than a hardcoded path, and <tree-root>/src goes on sys.path so
the build under test is the one asked. With a cache and keyfile it evaluates the gate over the
corpus -- but only after the synthetic cases prove the probe can fire on this configuration.
"""
import os
import sys

K, ESZ, REFN = 8, 4, 64

CASES = (
    # one row holding ~99% of the nonzeros: maximally imbalanced, clears MIN_NNZ and MIN_DEGREE
    ("imbalanced 1x1e6 + 999x10", [10 ** 6] + [10] * 999),
    # the same nonzeros spread evenly: clears the size screens, tests the imbalance screen
    ("balanced 1000x1010", [1010] * 1000),
    # imbalanced but under MIN_NNZ = 150000, so it must be refused on size whatever else holds
    ("imbalanced but under MIN_NNZ", [10 ** 4] + [10] * 99),
)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().split("USAGE")[-1].strip())
    tree = os.path.abspath(sys.argv[1])
    src = os.path.join(tree, "src")
    if not os.path.isdir(src):
        sys.exit("REFUSE: %s has no src/, so there is no build to ask" % tree)
    sys.path.insert(0, src)
    import numpy as np
    import torch                                                      # noqa: F401
    import scorch_ops as S

    loaded = os.path.dirname(os.path.abspath(S.__file__))
    if os.path.abspath(loaded) != os.path.abspath(src):
        sys.exit("REFUSE: asked for %s but imported %s -- an installed copy won" % (src, loaded))
    for name in ("scorch_spmm_split_seg", "scorch_spmm_row_imbalance"):
        if not hasattr(S, name):
            sys.exit("REFUSE: this build does not export %s" % name)

    hooks = getattr(S, "spmm_tune_hooks", None)
    envset = os.environ.get("SCORCH_SPMM_SPLIT_MIN_IMBALANCE")
    print("tree          %s" % tree)
    print("module        %s" % S.__file__)
    print("tune_hooks    %s" % hooks)
    print("env MIN_IMBALANCE %r" % envset)
    print()

    def indptr(lens):
        return torch.from_numpy(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))

    print("OWN-INPUT CHECK -- can this configuration return nonzero at all?")
    print("%-30s %8s %11s %10s" % ("case", "rows", "imbalance", "split_seg"))
    fired = 0
    for lab, lens in CASES:
        ip, rows = indptr(lens), len(lens)
        imb = S.scorch_spmm_row_imbalance(ip, rows, REFN)
        seg = S.scorch_spmm_split_seg(ip, rows, K, ESZ)
        fired += 1 if seg else 0
        print("%-30s %8d %11.4f %10d" % (lab, rows, imb, seg))
    print()
    if fired == 0:
        print("VERDICT: the probe CANNOT FIRE on this configuration, so every zero it reports")
        print("         reflects the build and not the matrix. Refusing a corpus answer.")
        print("         To obtain a discriminating configuration, use a build whose tune_hooks")
        print("         is True and set SCORCH_SPMM_SPLIT_MIN_IMBALANCE to a positive value;")
        print("         on a non-hooks build that env var is not compiled in and is ignored.")
        return 2
    print("VERDICT: the probe DISCRIMINATES here (%d of %d cases fire), so a zero is a fact"
          % (fired, len(CASES)))
    print("         about the matrix. Note a positive MIN_IMBALANCE of 1 admits even the")
    print("         perfectly balanced case, since the test is `imbalance < min_imb`.")

    if len(sys.argv) < 4:
        print()
        print("No cache and keyfile given, so no corpus answer was attempted.")
        return 0

    cache, keyfile = sys.argv[2], sys.argv[3]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import csrresolve
    keys = csrresolve.read_keys(keyfile)
    paths, how, bad = csrresolve.resolve_all(cache, keys)
    print()
    for ln in csrresolve.describe(how, bad, "corpus keys"):
        print(ln)
    if not paths:
        sys.exit("REFUSE: zero keys resolved, so a corpus count would be vacuous")

    firing = []
    for key in keys:
        p = paths.get(key)
        if p is None:
            continue
        with np.load(p) as z:
            ip = torch.from_numpy(z["indptr"].astype(np.int32))
        rows = int(ip.numel()) - 1
        seg = S.scorch_spmm_split_seg(ip, rows, K, ESZ)
        if seg:
            firing.append((key, rows, int(ip[-1]), seg,
                           S.scorch_spmm_row_imbalance(ip, rows, REFN)))
    print()
    print("CORPUS ANSWER at k=%d, elem_size=%d: the gate fires on %d of %d resolved matrices"
          % (K, ESZ, len(firing), len(paths)))
    for key, rows, nnz, seg, imb in firing[:20]:
        print("    %-46s rows %8d nnz %10d seg %7d imb %8.2f"
              % (key[-46:], rows, nnz, seg, imb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
