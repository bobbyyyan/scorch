"""Does the SpMM work-stealing chunk rule beat the generic width it replaces?

`scorch_spmm_chunk` picks rows-per-chunk from

    chunk* = rows * sqrt(K * KREF / (nnz * k))

clamped below by the generic chunk, above by a load-balance bound, and finally
snapped back to generic unless it asks for at least SCORCH_SPMM_CHUNK_MINRATIO
times that width. K is the machine term -- one contended atomic over one nonzero
of work -- and it arrived as a literal 16 fitted on one host.

WHAT THE FIRST VERSION OF THIS HARNESS GOT WRONG, because the correction is the
whole reason this file looks the way it does:

  1. It ordered arms by ROTATION, j = (i + r) % n. A rotation moves each arm's
     absolute position but never its predecessor -- the cyclic order is fixed --
     so it cancels the cold-start effect and leaves neighbour effects fully
     intact. With a power-of-two chunk ladder in the arm list, the arms are not
     equal-cost: chunk=1 makes every row a contended atomic. The generic-width
     arm always ran directly after chunk=32, and the two control arms sat at
     indices 0 and n-1, always after the cheapest arm on the ladder. That is a
     bias, not variance, and it survived every round. It reported cells where the
     rule provably changes nothing (rule and override both resolve to 64) as
     2.26x and 3.90x wins, largest exactly where per-call contention dominates
     the runtime and absent on the big memory-bound matrices.
  2. It mutated os.environ INSIDE the timed region, so one arm paid a putenv and
     the other an unsetenv.
  3. It compared min-over-two-arms for the rule against min-over-one for the
     generic width. Min of more samples is smaller; the estimators must match.

So: arms are randomly permuted every round, the override is set before the clock
starts, and every quantity that gets compared is entered twice.

The control that decides whether any of this is measurable at all is `mech`: the
rule's OWN chosen width, requested through the override. Identical width, identical
kernel, so it must read 1.000. Whatever it does read is this instrument's noise
floor, and no vs_gen may be believed inside it.

Columns:
  generic   the width the rule replaces -- the status quo
  formula   the width the shipped rule picks (read from the kernel, not restated)
  vs_gen    time at generic / time at the rule's width. > 1 = the rule is faster.
            THE DECISION.
  mech      time at the rule's width / same width via the override. Must be 1.000.
  A/A       widest same-width pair, i.e. this cell's floor.
  best      best width on the power-of-two ladder (--ladder only)
  K_impl    what K the winning rung implies (--ladder only)

Needs an instrumented build (SCORCH_TUNE_HOOKS) for the chunk override.

Usage:
  python bench/bench_spmm_chunk_sensitivity.py --mtxdir DIR --matrices a,b \
      [--ladder] [--order shuffle|rotate] [--rounds N]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import scorch_ops as SO  # noqa: E402

KREF = 8  # SCORCH_SPMM_CHUNK_KREF: the k at which the machine term was read
NULL_MIN_CELLS = 8  # below this a null group cannot set a floor; see the report below
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


def timed_min(specs, rounds, order, seed=0):
    """Time each arm `rounds` times and keep its own minimum.

    specs is a list of (setup, run) pairs: setup happens OUTSIDE the clock, so the
    cost of installing an override is never charged to the arm that needs one.

    order='shuffle' draws a fresh permutation every round, which is what makes a
    neighbour effect show up as variance -- visible in the A/A control -- instead
    of as a fixed per-arm offset. order='rotate' reproduces the original defect and
    exists only so the difference can be shown rather than asserted.
    """
    n = len(specs)
    best = [float("inf")] * n
    rng = random.Random(seed)
    for setup, run in specs:
        setup()
        run()
    for r in range(rounds):
        if order == "shuffle":
            visit = rng.sample(range(n), n)
        else:
            visit = [(i + r) % n for i in range(n)]
        for j in visit:
            setup, run = specs[j]
            setup()
            t0 = time.perf_counter()
            run()
            dt = time.perf_counter() - t0
            if dt < best[j]:
                best[j] = dt
    return best


def implied_k(chunk: int, rows: int, nnz: int, k: int) -> float:
    """Invert chunk* = rows*sqrt(K*KREF/(nnz*k)) for K."""
    if rows <= 0:
        return float("nan")
    return (chunk / rows) ** 2 * nnz * k / KREF


def spread(a: float, b: float) -> float:
    return max(a, b) / min(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtxdir", required=True)
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--Ks", default="8,16,32,64")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--order", choices=("shuffle", "rotate"), default="shuffle")
    ap.add_argument(
        "--ladder",
        action="store_true",
        help="also sweep the power-of-two chunk ladder. Adds arms whose cost spans "
        "10x, which is what made neighbour bias large enough to see.",
    )
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
    print(
        f"threads={nthreads}  rounds={args.rounds}  KREF={KREF}  "
        f"order={args.order}  ladder={args.ladder}"
    )
    head = (
        f"\n{'matrix':<18}{'k':>5}{'rows':>9}{'generic':>8}{'formula':>9}"
        f"{'vs_gen':>8}{'mech':>7}{'A/A':>7}"
    )
    if args.ladder:
        head += f"{'best':>8}{'K_impl':>9}"
    print(head)

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

            def setup(chunk):
                def go():
                    if chunk is None:
                        os.environ.pop("SCORCH_SPMM_CHUNK", None)
                    else:
                        os.environ["SCORCH_SPMM_CHUNK"] = str(chunk)

                return go

            def run():
                return SO.spmm_csr_float_v2(
                    *shapes, nthreads_override=nthreads, atparallel=False
                )

            # Clear the override BEFORE asking the rule what it would pick: the
            # override short-circuits the rule, so a leftover from the previous
            # cell reads back as "the rule chose this".
            os.environ.pop("SCORCH_SPMM_CHUNK", None)
            # The thread count the KERNEL resolves, not the one this process asked
            # for. They differ -- omp_get_num_procs() reports 32 on a 24-physical-
            # core part -- and using torch's number attributed the kernel's chunk to
            # a thread count it never used.
            kernel_nt = SO.scorch_spmm_nthreads(nnz * k, M, nthreads)
            formula = SO.scorch_spmm_chunk(M, nnz, k, kernel_nt)
            generic = SO.scorch_chunk_generic(M, nnz * k, GRAIN_SPMM)

            # Every compared quantity is entered twice, and `mech` re-requests the
            # rule's own width through the override: same width, same kernel, so it
            # is a null by construction and reads this cell's noise floor.
            tags = ["rule", "rule", "mech", "mech", "gen", "gen"]
            chunks = [None, None, formula, formula, generic, generic]
            ladder = chunk_ladder(M) if args.ladder else []
            for c in ladder:
                tags.append("lad")
                chunks.append(c)
            specs = [(setup(c), run) for c in chunks]
            times = timed_min(specs, args.rounds, args.order)

            t_rule = min(times[0], times[1])
            t_mech = min(times[2], times[3])
            t_gen = min(times[4], times[5])
            vs_gen = t_gen / t_rule
            mech = t_rule / t_mech
            aa = max(
                spread(times[0], times[1]),
                spread(times[2], times[3]),
                spread(times[4], times[5]),
            )
            line = (
                f"{name:<18}{k:>5}{M:>9}{generic:>8}{formula:>9}"
                f"{vs_gen:>8.3f}{mech:>7.3f}{aa:>7.3f}"
            )
            best_chunk, ki = 0, float("nan")
            if ladder:
                lo = 6
                best_i = min(range(lo, len(times)), key=lambda i: times[i])
                best_chunk = ladder[best_i - lo]
                ki = implied_k(best_chunk, M, nnz, k)
                line += f"{best_chunk:>8}{ki:>9.1f}"
            print(line, flush=True)
            rows_out.append(
                dict(
                    name=name,
                    k=k,
                    rows=M,
                    generic=generic,
                    formula=formula,
                    vs_gen=vs_gen,
                    mech=mech,
                    aa=aa,
                    best=best_chunk,
                    ki=ki,
                    noop=(formula == generic),
                )
            )
    os.environ.pop("SCORCH_SPMM_CHUNK", None)

    def geo(xs):
        return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")

    print("\n" + "=" * 78)
    mechs = sorted(r["mech"] for r in rows_out)
    aas = sorted(r["aa"] for r in rows_out)
    print(
        f"MECHANISM NULL, same width through the override: n={len(mechs)} "
        f"geomean {geo(mechs):.3f}  range {mechs[0]:.3f}-{mechs[-1]:.3f}"
    )
    print(
        f"same-width A/A pairs:                            "
        f"range {aas[0]:.3f}-{aas[-1]:.3f}"
    )
    print("  Nothing below is believable inside these two bands.\n")

    # Cells where the rule returns the generic width run identical code, so their
    # vs_gen is a second, independent null -- and it is the one that caught the
    # rotation bias.
    noop = sorted(r["vs_gen"] for r in rows_out if r["noop"])
    fires = sorted(r["vs_gen"] for r in rows_out if not r["noop"])
    if noop:
        print(
            f"NO-OP cells (rule returns generic, identical code): n={len(noop)} "
            f"geomean {geo(noop):.3f}  range {noop[0]:.3f}-{noop[-1]:.3f}"
        )
    if fires:
        print(
            f"cells where the rule FIRES: n={len(fires)} geomean {geo(fires):.3f}  "
            f"range {fires[0]:.3f}-{fires[-1]:.3f}"
        )
        # A floor is only a floor if enough cells drew it. The no-op group exists only
        # where the gate is SHUT, so on a grid where the rule fires almost everywhere it
        # can come out with a handful of cells and a spuriously tight range -- three
        # cells spanning 1.002-1.004 once flagged five neutral cells as "below the
        # floor". Under that threshold, say so and defer to the mechanism null, which is
        # the one control that exists on every cell including the firing ones.
        if len(noop) >= NULL_MIN_CELLS:
            below = [x for x in fires if x < noop[0]]
            print(
                f"  firing cells below the no-op null's floor ({noop[0]:.3f}): "
                f"{len(below)}"
            )
        else:
            print(
                f"  no-op null has only {len(noop)} cells, too few to set a floor -- "
                f"judge each firing cell against its own mechanism null instead"
            )
            for r in rows_out:
                if not r["noop"] and r["mech"] > 0:
                    net = r["vs_gen"] / r["mech"]
                    print(
                        f"    {r['name']:<18}k={r['k']:<4} vs_gen {r['vs_gen']:.3f} / "
                        f"mech {r['mech']:.3f} = {net:.4f}"
                    )
    allg = sorted(r["vs_gen"] for r in rows_out)
    print(
        f"whole grid: n={len(allg)} geomean {geo(allg):.3f}  "
        f"min {allg[0]:.3f}  max {allg[-1]:.3f}"
    )

    binding = [r for r in rows_out if r["formula"] > 64 and r["best"]]
    if binding:
        ks = sorted(r["ki"] for r in binding)
        print(
            f"\nK implied by the winning rung, over the {len(binding)} cells where "
            f"the rule's own term binds: min {ks[0]:.1f} "
            f"median {ks[len(ks) // 2]:.1f} max {ks[-1]:.1f}"
        )
        print("  (the shipped constant is 16; compare across hosts before trusting it)")


if __name__ == "__main__":
    main()
