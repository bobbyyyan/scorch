"""Does composing `scorch.compile`'s fused SpMM+bias+act with the tiling selector pay?

Four questions, one harness:

1. **The headline.** On a fused graph over a high-degree operand that overflows the
   last-level cache, what does letting the tiling selector serve the SpMM buy over
   the fused kernel alone? (`fused_off / fused_on`.)
2. **Is fusion still worth it once both sides tile?** Before this composition a fused
   graph could not reach the tiled kernels, so on reddit-class shapes "fusion" meant
   giving up tiling and `fused / unfused` was a loss. (`fused_on / unfused_on`.)
3. **What does the out-of-line tail cost?** The tiled kernels have no fused epilogue,
   so on the tiled route `bias`+`relu` is a separate pass over the M x N output. Timed
   directly, and cross-checked against the N-scaling: a per-output-element cost must
   scale with N and be independent of nnz.
4. **Does anything regress?** Every cell also runs an A/A control -- the same arm
   twice -- so a difference is only reported against that cell's own noise floor. What
   the composition costs the shapes the gate DECLINES is not measured here: end to end
   it is a fraction of a percent of a 15-500 us call and drowns in the noise six
   interleaved OpenMP teams in one process make (measured A/A floor 0.85-1.06 on those
   cells). `bench_fused_tiling_declined.py` measures it by isolation instead.

Arms are interleaved in a fresh random order every round and estimated by minimum,
not mean: each arm launches an OpenMP team and the arm that runs after another
inherits its spinners (see the openmp_private_team_poisoning note), so ordering
effects are noise in both directions and the minimum is the least contaminated
statistic available.

`--level off` is the "before" arm rather than a second checkout: `tiling.is_candidate`
short-circuits on that level, so `_dispatch_tiled` returns before touching a kernel
and the fused path runs exactly the fused kernel it ran before this composition
existed. Same binary, same process, no build to keep in sync.

Usage:
    python bench/bench_fused_tiling.py --matrix /path/to/reddit.npz --rounds 5
"""

import argparse
import gc
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import scorch  # noqa: E402
from scorch import ops, tiling  # noqa: E402
from scorch.prebuilt_kernels import resolve_prebuilt_matmul  # noqa: E402
from scorch.stensor import STensor  # noqa: E402


def load_csr(path, keep_every=1):
    """Load a CSR matrix from an npz, optionally thinning its nonzeros.

    `keep_every=2` keeps every second nonzero of every row: same M, same N, same row
    structure, half the nnz. That is the discriminator for question 3 -- a cost that
    belongs to the output pass must not move when only nnz moves.
    """
    z = np.load(path)
    indptr, indices, data = z["indptr"], z["indices"], z["data"]
    rows, cols = int(z["shape"][0]), int(z["shape"][1])
    if keep_every > 1:
        # Vectorized: a nonzero's position within its own row is its flat index minus
        # its row's start, so keeping every k-th of each row is one modulo. A Python
        # loop over 232k rows and 115M nonzeros is minutes and several copies.
        degrees = np.diff(indptr.astype(np.int64))
        row_of = np.repeat(np.arange(rows, dtype=np.int64), degrees)
        row_start = indptr[:-1].astype(np.int64)[row_of]
        keep = (np.arange(indices.size, dtype=np.int64) - row_start) % keep_every == 0
        indices, data = indices[keep], data[keep]
        # bincount, not add.reduceat: reduceat on an empty row reads the next row's
        # first element instead of returning zero, and real graphs have empty rows.
        indptr = np.concatenate(
            ([0], np.cumsum(np.bincount(row_of[keep], minlength=rows)))
        ).astype(np.int64)
    csr = torch.sparse_csr_tensor(
        torch.from_numpy(indptr.astype(np.int64)),
        torch.from_numpy(indices.astype(np.int64)),
        torch.from_numpy(data.astype(np.float32)),
        size=(rows, cols),
    )
    return STensor.from_torch(csr), rows, cols, int(indices.size)


@scorch.compile
def fused_graph(a, b, bias):
    return torch.relu(scorch.matmul(a, b) + bias)


def unfused(a, b, bias):
    return torch.relu(scorch.matmul(a, b) + bias)


def timed(fn, repeats):
    """Minimum wall time over `repeats` calls, after one warmup."""
    fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bare_tiled_arm(a, b, level, baseline_tag):
    """The tiled kernel the selector chose, run WITHOUT the tail.

    `tail_only` times the tail on a cold buffer of its own, which is a proxy. This is
    the direct measurement: whatever separates this from `fused_on` is what the
    out-of-line tail costs *in the route*, with the output in whatever cache state the
    tiled kernel just left it. Returns None when the selector declined the shape (no
    tiled kernel ran, so there is nothing to compare against).
    """
    verdict = tiling.decided(a, b.shape[1], level=level, baseline_tag=baseline_tag)
    if verdict is None or verdict[0] == "v2":
        return None
    kind, param = verdict
    result_shape = [a.shape[0], b.shape[1]]
    b_st = STensor.from_torch(b) if isinstance(b, torch.Tensor) else b
    # Through the shared helper, not by re-deriving it: the tiled kernels are handed
    # these hints on the real route, and an arm that configures them differently is
    # measuring a kernel the library never runs. That divergence is the defect this
    # whole composition exists to make impossible, so the harness must not reintroduce
    # it.
    resolved = resolve_prebuilt_matmul(a, b_st, output_format="dd")
    nthreads, _atparallel = ops._composition_hints(resolved)
    if nthreads is None:
        nthreads = -1
    if kind == "tilej":
        args = tiling._tilej_args(a, b_st, result_shape, param, nthreads)
        return lambda: tiling._ops.spmm_csr_float_tilej(*args)
    Nc, Jc = param
    args = tiling._tileijk_args(a, b_st, result_shape, Nc, Jc, nthreads)
    return lambda: tiling._ops.spmm_csr_float_tileijk(*args)


def measure_cell(a, b, bias, level, rounds, repeats):
    """Run every arm `rounds` times in a fresh random order; keep each arm's minimum."""

    def with_level(fn, lvl):
        def run():
            previous = scorch.get_autotune()
            scorch.set_autotune(lvl)
            try:
                return fn()
            finally:
                scorch.set_autotune(previous)

        return run

    out_buffer = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32)

    def tail_only():
        # The out-of-line tail on its own: one read-modify-write pass over M x N.
        out_buffer.add_(bias).relu_()

    arms = {
        "fused_on": with_level(lambda: fused_graph(a, b, bias), level),
        "fused_off": with_level(lambda: fused_graph(a, b, bias), "off"),
        "unfused_on": with_level(lambda: unfused(a, b, bias), level),
        "unfused_off": with_level(lambda: unfused(a, b, bias), "off"),
        "tail_only": tail_only,
        # A/A control: the same arm under a second name. Whatever separates these two
        # is the floor this cell can resolve.
        "fused_on_control": with_level(lambda: fused_graph(a, b, bias), level),
    }

    # Pay every arm's first-call cost outside the timed region: the selector's probe
    # times ~5 candidates x 3 invocations, and charging that to arm one would read as
    # a regression that never happens again.
    for run in arms.values():
        run()
        run()

    # Now that a fused call has landed and the selector has decided, the winning tiled
    # kernel can be timed on its own.
    bare = bare_tiled_arm(a, b, level, "spmm_csr_bias_relu_float")
    if bare is not None:
        arms["tilej_bare"] = bare
        bare()
        bare()

    best = {name: float("inf") for name in arms}
    order = list(arms)
    for _ in range(rounds):
        random.shuffle(order)
        for name in order:
            gc.collect()
            best[name] = min(best[name], timed(arms[name], repeats))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--name", default="reddit")
    parser.add_argument("--free-dims", default="16,64,128,256")
    parser.add_argument("--keep-every", default="1,2")
    parser.add_argument("--level", default="balanced")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"host threads: {torch.get_num_threads()}   level: {args.level}   "
          f"rounds: {args.rounds}   repeats: {args.repeats}", flush=True)
    print(f"LLC as scorch sees it: {tiling.query_llc() / 2**20:.1f} MiB", flush=True)

    header = (
        f"{'matrix':<22} {'nnz':>12} {'N':>5} "
        f"{'fused_on':>10} {'fused_off':>10} {'unfused_on':>11} {'unfused_off':>11} "
        f"{'tail':>8} {'inroute':>8} {'off/on':>7} {'fus/unf':>8} {'A/A':>6} "
        f"{'verdict':>22}"
    )
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)

    for keep in [int(k) for k in args.keep_every.split(",")]:
        a, rows, cols, nnz = load_csr(args.matrix, keep)
        label = f"{args.name}" + ("" if keep == 1 else f"/nnz÷{keep}")
        for n in [int(x) for x in args.free_dims.split(",")]:
            b = torch.randn(cols, n)
            bias = torch.randn(n)
            best = measure_cell(a, b, bias, args.level, args.rounds, args.repeats)

            verdict = tiling.decided(
                a, n, level=args.level,
                baseline_tag="spmm_csr_bias_relu_float",
            )
            us = lambda k: best[k] * 1e6
            in_route = (
                f"{us('fused_on') - us('tilej_bare'):>8.0f}"
                if "tilej_bare" in best
                else f"{'-':>8}"
            )
            print(
                f"{label:<22} {nnz:>12,} {n:>5} "
                f"{us('fused_on'):>10.0f} {us('fused_off'):>10.0f} "
                f"{us('unfused_on'):>11.0f} {us('unfused_off'):>11.0f} "
                f"{us('tail_only'):>8.0f} {in_route} "
                f"{best['fused_off'] / best['fused_on']:>7.3f} "
                f"{best['fused_on'] / best['unfused_on']:>8.3f} "
                f"{best['fused_on_control'] / best['fused_on']:>6.3f} "
                f"{str(verdict):>22}",
                flush=True,
            )
            del b, bias
            gc.collect()
        del a
        gc.collect()


if __name__ == "__main__":
    main()
