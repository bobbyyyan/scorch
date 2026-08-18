#!/usr/bin/env python3
"""What a transposed dense operand costs, and what each way of not paying it is worth.

`SparseStorage` holds a flat contiguous values array, so `scorch.matmul(A, B)` with a
non-contiguous `B` -- `W.T`, `x.permute(1, 0)`, a strided slice -- has to materialize a
copy. Two independent things reduce that bill and this harness measures both, crossed:

  xpose   materialize with the cache-blocked native transpose instead of torch's
          element-scatter `.contiguous()`; bit-identical, nothing retained
  memo    remember the copy, keyed on the base tensor's identity and version counter, so
          a repeated unmodified operand copies once

They answer different questions, so the grid has two scenarios and both matter:

  stable   the same base tensor every call -- an inference loop, a frozen weight, an
           attention mask. The memo hits every time.
  changing the base is written in place before each use, the way a dataloader refills a
           buffer, so its version counter moves and the memo misses every time. Only the
           faster materialization can help here, and the memo's arm shows what its lookup
           costs when it cannot pay.

A first attempt at the second scenario cycled a pool of operands larger than the memo's
bound, on the theory that this would always miss. It does not: with a bound of 16 and a
pool of 32 the first 16 keys stay resident, so exactly half the calls hit and the arm
reads as a partial win. The in-place write is what actually makes it miss. It costs every
arm the same ~2 us, so it compresses the ratios slightly rather than tilting them.

Every arm is the same binary and the same tree -- the memo is a runtime cell and the native
symbol is rebound outside the timed region -- and arms are visited in a fresh random order
every round. The `plain` arm run twice under two names is the A/A control.

Usage:
    python bench/bench_operand_copy.py --reps 9
"""

import argparse
import random
import statistics
import time

import numpy as np
import torch

import scorch
import scorch.stensor as stensor_module
from scorch import STensor

NATIVE_TRANSPOSE = stensor_module._NATIVE_TRANSPOSE

SCENARIOS = ("stable", "changing")

# (label, rows, free dim, iterations per sample)
CELLS = [
    ("256x64", 256, 64, 400),
    ("2000x64", 2000, 64, 200),
    ("2000x256", 2000, 256, 60),
    ("20000x32", 20000, 32, 60),
]
ARMS = ("plain", "control", "xpose", "memo", "both")
POOL = 32  # more distinct operands than the memo's bound, so it has to choose
WARMUP = 3 * stensor_module._OPERAND_COPY_GIVE_UP + 16


def select(arm):
    """Bind the two levers. `control` is `plain` under a second name."""
    stensor_module._MEMO_OPERAND_COPY[0] = arm in ("memo", "both")
    stensor_module._NATIVE_TRANSPOSE = (
        NATIVE_TRANSPOSE if arm in ("xpose", "both") else None
    )
    stensor_module._OPERAND_COPY_CACHE.clear()
    # The withdrawal counter and the sweep countdown are process state, so an arm that
    # provoked a withdrawal must not hand it to the next one.
    stensor_module._OPERAND_COPY_STATE[:] = [0, stensor_module._OPERAND_COPY_CACHE_MAX]


def csr_operand(rows, degree=16, seed=0):
    generator = np.random.default_rng(seed)
    columns = np.concatenate(
        [
            np.sort(generator.choice(rows, size=degree, replace=False))
            for _ in range(rows)
        ]
    ).astype(np.int64)
    positions = (np.arange(rows + 1) * degree).astype(np.int64)
    values = generator.standard_normal(rows * degree).astype(np.float32)
    return STensor.from_torch(
        torch.sparse_csr_tensor(
            torch.from_numpy(positions),
            torch.from_numpy(columns),
            torch.from_numpy(values),
            size=(rows, rows),
        ),
        "A",
    ).to_sparse("ds")


def timed(call, inner):
    start = time.perf_counter()
    for _ in range(inner):
        call()
    return (time.perf_counter() - start) / inner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    if NATIVE_TRANSPOSE is None:
        print(
            "scorch_transpose_2d_float is unavailable; the xpose arms are meaningless"
        )

    print(f"threads={args.threads} reps={args.reps}")
    header = f"{'scenario':>9s} {'cell':>10s}"
    for arm in ARMS:
        header += f" {arm + '_us':>11s}"
    print(header + f" {'best':>7s}")

    for scenario in SCENARIOS:
        for label, rows, free_dim, inner in CELLS:
            A = csr_operand(rows)
            # The operand is the transpose of a contiguous [free_dim, rows] matrix, which
            # is what `W.T` is: 2-D, float32, column-major, storage offset 0.
            bases = [torch.rand(free_dim, rows) for _ in range(POOL)]
            state = {"next": 0}

            def call(A=A, bases=bases, state=state, scenario=scenario):
                if scenario == "stable":
                    operand = bases[0].T
                else:
                    # A new base object would also miss, but allocating one per call would
                    # dominate the cell. Refilling one in place is what a dataloader does
                    # and is what moves the version counter the memo keys on.
                    state["next"] = (state["next"] + 1) % len(bases)
                    base = bases[state["next"]]
                    base[0, 0] += 0.0
                    operand = base.T
                return scorch.matmul(A, operand)

            # Correctness before speed: every arm must produce the same product. Checked
            # on one fixed operand rather than through `call`, which in the `fresh`
            # scenario hands out a different operand every time it is invoked.
            fixed = bases[0].T
            reference = None
            for arm in ARMS:
                select(arm)
                got = scorch.matmul(A, fixed)
                got = got if isinstance(got, torch.Tensor) else got.to_torch(False)
                if reference is None:
                    reference = got
                else:
                    torch.testing.assert_close(got, reference, atol=0, rtol=0)

            samples = {arm: [] for arm in ARMS}
            for _ in range(args.reps):
                order = list(ARMS)
                random.shuffle(order)
                for arm in order:
                    select(arm)
                    # Long enough to reach steady state, not just to warm a cache. In the
                    # `changing` scenario the memo withdraws after a few dozen calls, so a
                    # two-call warmup would leave every timed rep paying a withdrawal that
                    # a real program pays once in its life.
                    for _ in range(WARMUP):
                        call()
                    samples[arm].append(timed(call, inner))
            select("plain")

            figures = {a: statistics.median(v) * 1e6 for a, v in samples.items()}
            row = f"{scenario:>9s} {label:>10s}"
            for arm in ARMS:
                row += f" {figures[arm]:11.2f}"
            best = min(figures, key=figures.get)
            print(f"{row} {best:>7s}")

    print(
        "\nRead `control` against `plain`: same code, so their gap is this host's floor."
    )


if __name__ == "__main__":
    main()
