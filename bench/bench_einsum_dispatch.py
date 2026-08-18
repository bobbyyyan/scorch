#!/usr/bin/env python3
"""What a warm `scorch.einsum` spends before and after the kernel, and what fixing it won.

On a hit in the dispatch cache the whole scheduling pipeline is skipped and the call is
supposed to be: look up the module, bind the operands, run the kernel, wrap the result.
Four things in that sequence were re-derived on every call and are constants of it:

  key       the dispatch key rendered each operand's layout to JSON (`json.dumps`) to get
            a key for an in-process dict; the layout is a frozen value object and hashes
  labels    the expression's labels were validated character by character per call
  sizes     the logical index -> size map was built twice, once to validate and once to
            shape the result
  wrap      a dense result's index, layout and metadata were rebuilt per call, and a
            relayout into the order the result already had was requested and declined

None of these can be swapped at runtime -- three are the absence of code -- so the arms
here are two *trees* sharing one binary and one JIT cache, run as subprocesses in a fresh
random order every round. `base` twice is the A/A control: its spread is what this host
cannot distinguish, and a cell whose cand/base ratio sits inside it did not move.

The estimator is the minimum, not the median: each arm is a fresh process, so the arms
warm differently and a median measures the warming. The minimum of a few thousand
iterations is the same statistic in both.

Usage:
    # one tree, printed
    python bench/bench_einsum_dispatch.py

    # two trees, interleaved, with the A/A control
    python bench/bench_einsum_dispatch.py --arms /path/to/base /path/to/cand --reps 5
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time

import numpy as np
import torch


def csr_operand(rows, cols, degree, seed=0):
    """A CSR STensor with `degree` nonzeros per row, columns sorted within each row."""
    from scorch import STensor

    generator = np.random.default_rng(seed)
    columns = np.concatenate(
        [
            np.sort(generator.choice(cols, size=min(degree, cols), replace=False))
            for _ in range(rows)
        ]
    ).astype(np.int64)
    per_row = min(degree, cols)
    positions = (np.arange(rows + 1) * per_row).astype(np.int64)
    values = generator.standard_normal(rows * per_row).astype(np.float32)
    torch_csr = torch.sparse_csr_tensor(
        torch.from_numpy(positions),
        torch.from_numpy(columns),
        torch.from_numpy(values),
        size=(rows, cols),
    )
    return STensor.from_torch(torch_csr, "A").to_sparse("ds")


# (label, rows, degree, free dim, output format, iterations per sample). The grid spans
# from cells where the fixed cost is most of the call to cells where the kernel is, so a
# claim about the fixed cost cannot hide a regression in the kernel-bound end.
CELLS = [
    ("spmm 64x4 N=4", 64, 4, 4, "dd", 4000),
    ("spmm 64x4 N=64", 64, 4, 64, "dd", 3000),
    ("spmm 256x8 N=8", 256, 8, 8, "dd", 2000),
    ("spmm 256x8 N=64", 256, 8, 64, "dd", 1000),
    ("spmm 2000x8 N=8", 2000, 8, 8, "dd", 500),
    ("spmm 2000x8 N=64", 2000, 8, 64, "dd", 200),
    ("spmm 20000x16 N=32", 20000, 16, 32, "dd", 50),
    ("spmm 64x4 N=4 ->ds", 64, 4, 4, "ds", 3000),
    ("spmm 256x8 N=8 ->ds", 256, 8, 8, "ds", 1500),
    ("spmm 2000x8 N=64 ->ds", 2000, 8, 64, "ds", 100),
]
# Sparse x sparse, which reaches a different generated kernel and a sparse-result wrap.
SPGEMM_CELLS = [
    ("spgemm 64x4", 64, 4, "ds", 1000),
    ("spgemm 512x8", 512, 8, "ds", 300),
]
# A second expression, so a memo keyed on the expression is exercised by more than one.
ELEMENTWISE_CELLS = [
    ("mul 64x4", 64, 4, "ds", 2000),
    ("mul 2000x8", 2000, 8, "ds", 500),
]


def best(call, inner):
    """The minimum per-iteration time over one sample of `inner` iterations."""
    start = time.perf_counter()
    for _ in range(inner):
        call()
    return (time.perf_counter() - start) / inner


def measure(reps, threads):
    import scorch

    torch.set_num_threads(threads)
    calls = []
    for label, rows, degree, free_dim, fmt, inner in CELLS:
        A = csr_operand(rows, rows, degree)
        B = torch.rand(rows, free_dim)
        calls.append(
            (
                label,
                inner,
                lambda A=A, B=B, f=fmt: scorch.einsum("ik,kj->ij", A, B, format=f),
            )
        )
    for label, rows, degree, fmt, inner in SPGEMM_CELLS:
        A = csr_operand(rows, rows, degree)
        C = csr_operand(rows, rows, degree, seed=1)
        calls.append(
            (
                label,
                inner,
                lambda A=A, C=C, f=fmt: scorch.einsum("ik,kj->ij", A, C, format=f),
            )
        )
    for label, rows, degree, fmt, inner in ELEMENTWISE_CELLS:
        A = csr_operand(rows, rows, degree)
        C = csr_operand(rows, rows, degree, seed=1)
        calls.append(
            (
                label,
                inner,
                lambda A=A, C=C, f=fmt: scorch.einsum("ij,ij->ij", A, C, format=f),
            )
        )

    results = {}
    for label, inner, call in calls:
        for _ in range(3):  # JIT compile and first-touch, outside every timed region
            call()
        results[label] = min(best(call, inner) for _ in range(reps)) * 1e6
    return results


def run_arm(tree, reps, threads):
    """Measure in a fresh process whose `scorch` comes from `tree`."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(tree, "src")
    out = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--emit",
            "--reps",
            str(reps),
            "--threads",
            str(threads),
        ],
        env=env,
        capture_output=True,
        text=True,
        cwd=tree,
        check=True,
    )
    return json.loads(out.stdout.splitlines()[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--arms", nargs=2, metavar=("BASE", "CAND"))
    parser.add_argument("--emit", action="store_true", help="internal: print one arm")
    args = parser.parse_args()

    if args.emit:
        print(json.dumps(measure(args.reps, args.threads)))
        return

    if not args.arms:
        results = measure(args.reps, args.threads)
        print(f"{'cell':>24s} {'us':>10s}")
        for label, value in results.items():
            print(f"{label:>24s} {value:10.2f}")
        return

    base, cand = (os.path.abspath(p) for p in args.arms)
    samples = {"base": [], "cand": [], "control": []}
    for _ in range(args.rounds):
        arms = [("base", base), ("cand", cand), ("control", base)]
        random.shuffle(arms)
        for name, tree in arms:
            samples[name].append(run_arm(tree, args.reps, args.threads))

    labels = list(samples["base"][0])
    print(f"threads={args.threads} rounds={args.rounds} reps={args.reps}")
    print(f"base={base}\ncand={cand}\n")
    print(
        f"{'cell':>24s} {'base_us':>10s} {'cand_us':>10s} {'cand/base':>10s} "
        f"{'A/A':>7s} {'verdict':>9s}"
    )
    for label in labels:
        b = min(round[label] for round in samples["base"])
        c = min(round[label] for round in samples["cand"])
        a = min(round[label] for round in samples["control"])
        control = a / b
        ratio = c / b
        floor = abs(1.0 - control)
        verdict = (
            "flat"
            if abs(1.0 - ratio) <= max(floor, 0.01)
            else ("faster" if ratio < 1.0 else "SLOWER")
        )
        print(
            f"{label:>24s} {b:10.2f} {c:10.2f} {ratio:10.3f} {control:7.3f} "
            f"{verdict:>9s}"
        )
    geo = statistics.geometric_mean(
        [
            min(r[label] for r in samples["cand"])
            / min(r[label] for r in samples["base"])
            for label in labels
        ]
    )
    geo_control = statistics.geometric_mean(
        [
            min(r[label] for r in samples["control"])
            / min(r[label] for r in samples["base"])
            for label in labels
        ]
    )
    print(f"\ngeomean cand/base {geo:.3f}   A/A control {geo_control:.3f}")


if __name__ == "__main__":
    main()
