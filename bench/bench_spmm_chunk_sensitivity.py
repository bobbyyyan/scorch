"""How much does the SpMM's work-stealing chunk width matter, and is one constant
for the machine term defensible on more than one host?

`scorch_spmm_chunk` picks rows-per-chunk from

    chunk* = rows * sqrt(K * KREF / (nnz * k))

clamped below by the generic chunk and above by a load-balance bound. Everything
matrix-specific is in rows, nnz and k. K is the machine term -- the cost of one
contended atomic against the cost of one nonzero of work -- and it arrived here as
a literal 16, fitted on one host and never measured on another. The performance
convention does not allow shipping that.

This maps the response surface directly instead of sweeping K, because chunk* only
moves as sqrt(K): a 4x error in K is a 2x error in chunk, so a sweep of K would
compress exactly the axis being calibrated. Sweeping the chunk override over a wide
ladder gives the per-cell optimum, and any K can then be evaluated against it
without re-running anything.

What it reports per cell:
  generic   the width the rule replaces -- the status quo
  formula   the width the shipped rule picks (read from the kernel, not restated)
  best      the best width ON THE LADDER, which is powers of two only
  vs_gen    time at the generic width over time at the rule's width, > 1 = the
            rule is faster than what it replaces. This is the decision.
  vs_best   time at the rule's width over time at the best ladder rung. NOT a
            per-cell oracle: the rule picks widths like 104 and 698, which are not
            on a power-of-two ladder, so this can come out below 1 and does.
  K_implied what K the winning ladder rung would have required

The decision this feeds: if the loss at K=16 is small on both hosts, and the
implied-K distributions overlap, one constant is justified by measurement. If they
do not overlap, K has to be derived on the host rather than written down.

Needs an instrumented build (SCORCH_TUNE_HOOKS) for the chunk override.

Usage:
  python bench/bench_spmm_chunk_sensitivity.py --mtxdir DIR --matrices a,b --Ks 8,16,32
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

KREF = 8  # SCORCH_SPMM_CHUNK_KREF: the k at which the machine term was read
GRAIN_SPMM = 150000  # SCORCH_GRAIN_SPMM, the work threshold the generic rule uses


def load_bin(path: str):
    with open(path, "rb") as f:
        M, J, nnz = np.frombuffer(f.read(24), dtype=np.int64)
        pos = np.frombuffer(f.read(4 * (int(M) + 1)), dtype=np.int32).copy()
        crd = np.frombuffer(f.read(4 * int(nnz)), dtype=np.int32).copy()
        val = np.frombuffer(f.read(4 * int(nnz)), dtype=np.float32).copy()
    return int(M), int(J), int(nnz), pos, crd, val


def chunk_ladder(rows: int) -> list:
    """Powers of two up to the row count, which is the widest meaningful chunk."""
    out, c = [], 1
    while c <= rows:
        out.append(c)
        c *= 2
    if rows not in out:
        out.append(rows)
    return out


def interleaved(arms, rounds):
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


def implied_k(chunk: int, rows: int, nnz: int, k: int) -> float:
    """Invert chunk* = rows*sqrt(K*KREF/(nnz*k)) for K."""
    if rows <= 0:
        return float("nan")
    return (chunk / rows) ** 2 * nnz * k / KREF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtxdir", required=True)
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--Ks", default="8,16,32,64")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    nthreads = torch.get_num_threads()
    if not getattr(SO, "scorch_tune_hooks", lambda: False)():
        print(
            "REFUSING TO RUN: no SCORCH_TUNE_HOOKS in this build, so the chunk "
            "override is inert and every arm would be the SAME width. Rebuild with "
            "SCORCH_BUILD_TUNE_HOOKS=1."
        )
        sys.exit(2)
    print(f"threads={nthreads}  rounds={args.rounds}  KREF={KREF}")
    print(
        f"\n{'matrix':<18}{'k':>5}{'rows':>9}{'generic':>8}{'formula':>9}"
        f"{'best':>8}{'vs_gen':>8}{'vs_best':>8}{'A/A':>7}{'K_impl':>9}"
    )

    rows_out = []
    for name in args.matrices.split(","):
        p = os.path.join(args.mtxdir, name if name.endswith(".bin") else name + ".bin")
        M, J, nnz, pos, crd, val = load_bin(p)
        tp, tc, tv = (
            torch.from_numpy(pos),
            torch.from_numpy(crd),
            torch.from_numpy(val),
        )
        Aidx = [[], [tp, tc]]
        for k in (int(x) for x in args.Ks.split(",")):
            B = torch.randn(J, k, dtype=torch.float32)
            shapes = ([M, k], [M, J], Aidx, tv, [J, k], [[], []], B.reshape(-1))

            def call(chunk):
                if chunk is None:
                    os.environ.pop("SCORCH_SPMM_CHUNK", None)
                else:
                    os.environ["SCORCH_SPMM_CHUNK"] = str(chunk)
                return SO.spmm_csr_float_v2(
                    *shapes, nthreads_override=nthreads, atparallel=False
                )

            # Clear the override BEFORE asking the rule what it would pick. The
            # override short-circuits the rule, and the arms below set it, so a
            # leftover from the previous cell is read back as "the rule chose this".
            # With a rotating start the leftover is deterministic, which makes the
            # wrong number look stable across every row of the table.
            os.environ.pop("SCORCH_SPMM_CHUNK", None)
            formula = SO.scorch_spmm_chunk(M, nnz, k, nthreads)
            generic = SO.scorch_chunk_generic(M, nnz * k, GRAIN_SPMM)
            ladder = chunk_ladder(M)
            # The formula's own setting appears twice, at both ends: the A/A control.
            arms = [lambda: call(None)]
            arms += [(lambda c=c: call(c)) for c in ladder]
            arms.append(lambda: call(None))
            times = interleaved(arms, args.rounds)
            t_f1, t_f2 = times[0], times[-1]
            aa = max(t_f1, t_f2) / min(t_f1, t_f2)
            t_formula = min(t_f1, t_f2)
            best_i = min(range(1, len(arms) - 1), key=lambda i: times[i])
            best_chunk = ladder[best_i - 1]
            t_best = times[best_i]
            loss = t_formula / t_best
            # The status quo: the generic width, timed on the same ladder. This is
            # the comparison that decides whether the rule should ship at all.
            t_generic = (
                times[1 + ladder.index(generic)] if generic in ladder else float("nan")
            )
            vs_gen = t_generic / t_formula  # > 1 means the rule beat the status quo
            ki = implied_k(best_chunk, M, nnz, k)
            rows_out.append(
                (name, k, M, formula, best_chunk, loss, aa, ki, generic, vs_gen)
            )
            print(
                f"{name:<18}{k:>5}{M:>9}{generic:>8}{formula:>9}{best_chunk:>8}"
                f"{vs_gen:>8.3f}{loss:>8.3f}{aa:>7.3f}{ki:>9.1f}"
            )
    os.environ.pop("SCORCH_SPMM_CHUNK", None)

    print("\n" + "=" * 76)
    gains = sorted(r[9] for r in rows_out if r[9] == r[9])
    if gains:
        ggeo = float(np.exp(np.mean(np.log(gains))))
        print(
            f"the rule against the GENERIC width it replaces: n={len(gains)} "
            f"geomean {ggeo:.3f}  min {gains[0]:.3f}  max {gains[-1]:.3f}"
        )
        print("  (> 1 means the rule is faster than the status quo)")
    losses = sorted(r[5] for r in rows_out)
    aas = sorted(r[6] for r in rows_out)
    geo = float(np.exp(np.mean(np.log(losses))))
    print(
        f"the rule vs the best POWER-OF-TWO width (not an oracle): n={len(losses)} "
        f"geomean {geo:.3f}  median {losses[len(losses) // 2]:.3f}  max {losses[-1]:.3f}"
    )
    print(f"A/A control on the formula's own width:        {aas[0]:.3f}-{aas[-1]:.3f}")
    # Only cells where the rule departs from the generic clamp say anything about K;
    # where it returns the generic width, K never entered the answer.
    # SCORCH_CHUNK_MAX is 64, so the generic chunk can never exceed it: a pick wider
    # than 64 rows can only have come from the rule's own analytic term, and a pick
    # of exactly 64 or below means the generic clamp decided and K never entered.
    binding = [r for r in rows_out if r[3] > 64]
    print(
        f"\ncells where the rule's own term binds (formula > 64 rows): "
        f"{len(binding)}/{len(rows_out)}"
    )
    if binding:
        ks = sorted(r[7] for r in binding)
        print(
            f"  K implied by the winning width: min {ks[0]:.1f} "
            f"median {ks[len(ks) // 2]:.1f} max {ks[-1]:.1f}"
        )
        print("  (the shipped constant is 16; compare across hosts before trusting it)")
    else:
        print("  the generic clamp bound every cell -- K is inert on this grid")
    print("\nA cell only counts if its loss is outside its own A/A control.")


if __name__ == "__main__":
    main()
