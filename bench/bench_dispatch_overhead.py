#!/usr/bin/env python3
"""Per-call Python dispatch overhead of ``scorch.matmul``, and where it goes.

Overhead is defined as ``matmul(A, B) - kernel(*args)``: the same prebuilt kernel, once
through the public entry point and once called directly with an argument list built
before timing starts. The difference is everything ``ops.matmul`` does in Python on the
way to the kernel — operand normalization, kernel resolution, the tiling gate, argument
marshalling — and it is what this measures.

It matters only at the small end. The overhead is nearly flat in problem size, so it is
~10% of a large SpMM and most of a small one; on the smallest cell here scorch's kernel
beats torch's whole call and scorch still loses on dispatch alone.

Method
------
* Arms visited in a freshly shuffled order every round; median over rounds.
* ``aa`` is the ``matmul`` arm entered under a second name, so |aa/matmul - 1| is the
  cell's in-process noise floor. Nothing smaller than the floor counts.
* ``kernel`` is the same native symbol the dispatch would have chosen, with its
  arguments hoisted out of the timed region.
* ``torch`` is a whole ``torch.sparse.mm`` call, its dispatch and its kernel together --
  an end-to-end reference point, not an overhead figure, so it is comparable to the
  ``matmul`` arm and not to ``matmul - kernel``. Unchanged by anything in this tree, so
  it doubles as the cross-process control when comparing two trees.
* Every cell is checked against a float64 reference.

Usage
-----
    python bench/bench_dispatch_overhead.py --csv before.csv --tag before
    python bench/bench_dispatch_overhead.py --profile      # cProfile the small cell
"""
from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time

import numpy as np
import torch


# (rows, degree, N) — spans "dispatch is everything" to "dispatch is noise".
CELLS = [
    # The first two exist to read the overhead almost directly: the kernel is a few
    # microseconds, so `matmul_us` is nearly all dispatch. Overhead on the large cells
    # is a difference of two large medians and is correspondingly noisy — attribute
    # levers on the small cells, and use the large ones only to prove nothing regressed.
    (64, 2, 1),
    (256, 2, 4),
    (500, 4, 8),
    (500, 4, 32),
    (2000, 8, 8),
    (2000, 8, 32),
    (2000, 8, 128),
    (20000, 24, 32),
    (20000, 24, 128),
]


# Compute-dominated cells (--big). Dispatch is noise here; they exist to show that
# splitting the kernel entry into a wrapper plus a pointer-based core, which a plan
# calls directly, did not cost the kernel anything.
BIG_CELLS = [
    (50000, 32, 128),
    (20000, 64, 256),
]


def build(rows, deg, seed=0):
    rng = np.random.default_rng(seed)
    cols = np.concatenate(
        [np.sort(rng.choice(rows, size=deg, replace=False)) for _ in range(rows)]
    ).astype(np.int64)
    indptr = np.arange(rows + 1, dtype=np.int64) * deg
    data = rng.standard_normal(rows * deg).astype(np.float32)
    return indptr, cols, data


def timed(fn, inner):
    """Seconds per call, holding results so the allocator behaves as it would."""
    held = []
    t0 = time.perf_counter()
    for _ in range(inner):
        held.append(fn())
    dt = (time.perf_counter() - t0) / inner
    del held
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--inner", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--tag", default="cand")
    ap.add_argument("--arms", default="all",
                    help="comma-separated subset of matmul,aa,kernel,torch,decline,"
                         "mixed,plan. "
                         "One arm per process removes arm-to-arm interference, which "
                         "matters when comparing two trees whose arm SETS differ: the "
                         "thread team and allocator a cell inherits depend on what ran "
                         "before it, so an extra arm in one tree biases the others.")
    ap.add_argument("--big", action="store_true",
                    help="append two compute-dominated cells (kernel refactor check)")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile the smallest cell instead of timing the grid")
    args = ap.parse_args()

    import scorch
    import scorch_ops
    from scorch.stensor import STensor

    try:
        from scorch.plan import plans_of
    except ImportError:  # a tree from before call plans existed, for A/B
        def plans_of(_tensor):
            return {}

    torch.set_num_threads(args.threads)

    if args.profile:
        import cProfile
        import io
        import pstats

        rows, deg, N = CELLS[0]
        indptr, cols, data = build(rows, deg)
        t = torch.sparse_csr_tensor(torch.from_numpy(indptr), torch.from_numpy(cols),
                                    torch.from_numpy(data), size=(rows, rows))
        A = STensor.from_torch(t)
        B = torch.ones(rows, N)
        for _ in range(50):
            scorch.matmul(A, B)
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(4000):
            scorch.matmul(A, B)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumtime").print_stats(20)
        print(f"# profile of {rows}x{rows} deg={deg} N={N}, 4000 calls")
        print("\n".join(s.getvalue().splitlines()[4:28]))
        return 0

    print(f"# tag={args.tag} threads={args.threads} reps={args.reps} "
          f"inner={args.inner}")
    print(f"{'cell':>18s} {'kernel_us':>10s} {'plan_us':>8s} {'matmul_us':>10s} "
          f"{'over_us':>8s} {'share':>6s} {'floor':>6s} {'declin_us':>9s} "
          f"{'mixed_us':>8s} {'torch_us':>9s} {'relerr':>9s}")

    cells = CELLS + (BIG_CELLS if args.big else [])
    out = []
    for rows, deg, N in cells:
        indptr, cols, data = build(rows, deg)
        t = torch.sparse_csr_tensor(torch.from_numpy(indptr), torch.from_numpy(cols),
                                    torch.from_numpy(data), size=(rows, rows))
        A = STensor.from_torch(t)
        B = torch.ones(rows, N)

        # The argument list the dispatch would have built, hoisted out of the timing.
        kargs = [(rows, N), tuple(A.shape), A._native_mode_indices(), A.values,
                 (rows, N), [[], []], B.reshape(-1)]
        kfn = scorch_ops.spmm_csr_float_v2

        def call_matmul():
            return scorch.matmul(A, B)

        def call_kernel():
            return kfn(*kargs)

        def call_torch():
            return torch.sparse.mm(t, B)

        # The installed plan, invoked directly: matmul_us - plan_us is what Python
        # still costs once a plan is serving (the type tests, the dict lookup and
        # the thread-count read). None when plans are disabled or declined.
        plan = None
        nt = torch.get_num_threads()
        for _ in range(3):
            scorch.matmul(A, B)
        for candidate in plans_of(A).values():
            if candidate.free_dim == N:
                plan = candidate

        def call_plan():
            return plan.run(A._storage._value, B, nt, True)

        # A right-hand operand a plan must decline on every call: same shape and dtype
        # as B, so the lookup hits, but not contiguous, so the plan has to refuse and
        # let the ordinary path materialize it. Two arms, because the cost differs and
        # so does the remedy:
        #
        #   decline -- its own operand, so nothing else ever installs a plan that
        #              serves. This is a call site a plan can never help, and after
        #              plan.MAX_FRUITLESS_DECLINES refusals the plan withdraws itself,
        #              so the steady state should be what it was before plans existed.
        #   mixed   -- shares the operand with the `matmul` arm, which is serving the
        #              contiguous B under the same key thousands of times. The plan is
        #              earning its keep, so it is deliberately NOT withdrawn, and these
        #              calls pay the refusal. Bounded and net-positive for the site,
        #              but it is the one path where plans cost more than they save.
        #
        # A separate STensor over the same CSR arrays gives `decline` its own plan
        # cache: plans live on the operand, so two wrappers of one matrix are
        # independent.
        Bt = B.T.contiguous().T
        assert not Bt.is_contiguous() or N == 1
        A_lonely = STensor.from_torch(t)

        def call_decline():
            return scorch.matmul(A_lonely, Bt)

        def call_mixed():
            return scorch.matmul(A, Bt)

        ref = torch.sparse.mm(
            torch.sparse_csr_tensor(torch.from_numpy(indptr), torch.from_numpy(cols),
                                    torch.from_numpy(data.astype(np.float64)),
                                    size=(rows, rows)),
            B.to(torch.float64),
        ).numpy()
        got = call_matmul()
        g = got.numpy() if hasattr(got, "numpy") else got.to_torch().numpy()
        relerr = float(np.abs(g.reshape(ref.shape) - ref).max()
                       / max(np.abs(ref).max(), 1e-30))

        for _ in range(3):  # warm the caches, the pool, and the memo
            call_matmul(); call_kernel(); call_torch()

        arms = {"matmul": call_matmul, "aa": call_matmul,
                "kernel": call_kernel, "torch": call_torch,
                "decline": call_decline, "mixed": call_mixed}
        if plan is not None and plan.run(A._storage._value, B, nt, True) is not None:
            arms["plan"] = call_plan
        if args.arms != "all":
            wanted = set(args.arms.split(","))
            arms = {k: v for k, v in arms.items() if k in wanted}
            if "matmul" not in arms:  # the floor and overhead columns need a baseline
                arms["matmul"] = call_matmul
        acc = {k: [] for k in arms}
        for _ in range(args.reps):
            order = list(arms)
            random.shuffle(order)
            for k in order:
                acc[k].append(timed(arms[k], args.inner))
        us = {k: statistics.median(v) * 1e6 for k, v in acc.items()}
        for absent in ("plan", "kernel", "torch", "decline", "mixed", "aa"):
            us.setdefault(absent, float("nan"))
        floor = abs(us["aa"] / us["matmul"] - 1.0) * 100
        over = us["matmul"] - us["kernel"]
        label = f"{rows}x{deg}@{N}"
        print(f"{label:>18s} {us['kernel']:10.1f} {us['plan']:8.1f} "
              f"{us['matmul']:10.1f} {over:8.1f} "
              f"{over / us['matmul'] * 100:5.1f}% {floor:5.2f}% {us['decline']:9.1f} "
              f"{us['mixed']:8.1f} {us['torch']:9.1f} {relerr:9.2e}")
        out.append(dict(tag=args.tag, rows=rows, deg=deg, N=N,
                        kernel_us=us["kernel"], matmul_us=us["matmul"],
                        overhead_us=over, floor_pct=floor, torch_us=us["torch"],
                        plan_us=us["plan"], decline_us=us["decline"],
                        mixed_us=us["mixed"], relerr=relerr))

    med = statistics.median([r["overhead_us"] for r in out])
    print(f"\nmedian overhead {med:.1f} us over {len(out)} cells "
          f"(min {min(r['overhead_us'] for r in out):.1f}, "
          f"max {max(r['overhead_us'] for r in out):.1f})")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"wrote {args.csv}")
    bad = [r for r in out if r["relerr"] > 1e-4]
    if bad:
        print(f"CORRECTNESS FAILURES: {len(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
