"""Is the masked last vector costing anything when k is a multiple of 8?

The narrow-k register kernel holds an output row in NVEC YMM registers and used a
masked AVX2 load and store on the final vector unconditionally. For a ragged k
that is necessary. For k = 8, 16, 24, 32 -- which is most of what narrow-k SpMM
sees -- the last vector is entirely valid and the mask is doing nothing except
costing what a mask costs: vmaskmovps is 2 uops on the load side and cannot fold
into the FMA as a memory operand, and its store form is worse. At k=16 that made
half of every row's B loads masked for no reason.

Both instantiations exist in an instrumented build (SCORCH_TUNE_HOOKS), and the
hook is read once per op, so this flips between them WITHIN one process against
ONE binary. That removes build-to-build and process-to-process variance entirely;
what is left is the row kernel. Arms are interleaved with a rotating start, and an
A/A control -- the unmasked arm entered twice -- says what the floor is.

Ragged k is measured too, and must come out neutral: the mask is still there, so a
difference on a ragged width would mean the change did something other than what
it claims.

Usage:
  python bench/bench_spmm_narrowk_mask.py --mtxdir DIR --matrices a,b --Ks 8,16,24,32
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import scorch_ops as SO  # noqa: E402
from scorch import tiling  # noqa: E402  (only for query_llc in the banner)


def load_bin(path: str):
    with open(path, "rb") as f:
        M, J, nnz = np.frombuffer(f.read(24), dtype=np.int64)
        pos = np.frombuffer(f.read(4 * (int(M) + 1)), dtype=np.int32).copy()
        crd = np.frombuffer(f.read(4 * int(nnz)), dtype=np.int32).copy()
        val = np.frombuffer(f.read(4 * int(nnz)), dtype=np.float32).copy()
    return int(M), int(J), int(nnz), pos, crd, val


def make_arms(M, J, pos, crd, val, N, nthreads):
    """One callable per arm; they differ only in the environment the kernel reads."""
    tp, tc, tv = (torch.from_numpy(pos), torch.from_numpy(crd), torch.from_numpy(val))
    Aidx = [[], [tp, tc]]
    B = torch.randn(J, N, dtype=torch.float32)
    Bv = B.reshape(-1)
    shapes = ([M, N], [M, J], Aidx, tv, [J, N], [[], []], Bv)

    def call(masked: str):
        os.environ["SCORCH_SPMM_MASKED"] = masked
        return SO.spmm_csr_float_v2(*shapes, nthreads, False)

    return call


def interleaved(arms, rounds):
    """Warm every arm, then rotate the start each round; keep each arm's minimum."""
    n = len(arms)
    best = [float("inf")] * n
    for fn in arms:
        fn()
    for r in range(rounds):
        for i in range(n):
            j = (i + r) % n
            t0 = time.perf_counter()
            arms[j]()
            dt = time.perf_counter() - t0
            if dt < best[j]:
                best[j] = dt
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtxdir", required=True)
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--Ks", default="8,16,24,32,17,23,33")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    nthreads = torch.get_num_threads()
    if not hasattr(SO, "spmm_csr_float_v2"):
        print("no spmm_csr_float_v2 in this build")
        return
    if not getattr(SO, "scorch_tune_hooks", lambda: False)():
        print(
            "REFUSING TO RUN: this build has no SCORCH_TUNE_HOOKS, so the hook "
            "this harness flips is inert and both arms would be the SAME code. "
            "The result would be a difference of zero that reads as 'the change "
            "did nothing'. Rebuild with -DSCORCH_TUNE_HOOKS."
        )
        sys.exit(2)
    print(
        f"threads={nthreads}  LLC={tiling.query_llc() >> 20}MiB  "
        f"rounds={args.rounds}"
    )
    print("k%8==0 rows are the ones the change touches; ragged k must be neutral.\n")
    print(
        f"{'matrix':<18}{'k':>4}{'ragged':>8}{'masked us':>11}"
        f"{'unmasked':>11}{'A/A':>7}{'speedup':>9}"
    )
    rows = []
    for name in args.matrices.split(","):
        p = os.path.join(args.mtxdir, name if name.endswith(".bin") else name + ".bin")
        M, J, nnz, pos, crd, val = load_bin(p)
        for N in (int(x) for x in args.Ks.split(",")):
            call = make_arms(M, J, pos, crd, val, N, nthreads)
            # unmasked entered twice, at both ends: the A/A control.
            arms = [lambda: call("0"), lambda: call("1"), lambda: call("0")]
            t_u1, t_m, t_u2 = interleaved(arms, args.rounds)
            aa = max(t_u1, t_u2) / min(t_u1, t_u2)
            t_u = min(t_u1, t_u2)
            ragged = "yes" if N % 8 else "no"
            speedup = t_m / t_u
            rows.append((name, N, ragged, aa, speedup))
            print(
                f"{name:<18}{N:>4}{ragged:>8}{t_m * 1e6:11.1f}"
                f"{t_u * 1e6:11.1f}{aa:7.3f}{speedup:9.3f}"
            )
    os.environ.pop("SCORCH_SPMM_MASKED", None)

    print("\n" + "=" * 68)
    for tag, want in (
        ("k % 8 == 0 (the change applies)", "no"),
        ("ragged k (must be neutral)", "yes"),
    ):
        sel = [r for r in rows if r[2] == want]
        if not sel:
            continue
        sp = sorted(r[4] for r in sel)
        aa = sorted(r[3] for r in sel)
        geo = float(np.exp(np.mean(np.log(sp))))
        print(
            f"{tag:<34} n={len(sel):<3} geomean {geo:.3f}  "
            f"min {sp[0]:.3f}  max {sp[-1]:.3f}   A/A {aa[0]:.3f}-{aa[-1]:.3f}"
        )
    print("\nspeedup = masked / unmasked, so > 1 means dropping the mask helped.")
    print("A cell only counts if its speedup is outside its own A/A control.")


if __name__ == "__main__":
    main()
