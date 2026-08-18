#!/usr/bin/env python3
"""What it costs Scorch to accept a sparse matrix, and what each fix to that was worth.

`_validate_index_storage` (src/scorch/storage.py) runs on every STensor built over a
compressed or coordinate level: `from_torch`, `from_csr`, `from_coo`, `to_sparse`, a
relayout, and the result of every generated kernel. Four things were wrong with it, and
this harness is how each was measured. Every arm is the SAME binary and the same tree --
the implementations are swapped at runtime, outside the timed region -- so nothing here
depends on comparing two builds.

Compressed ladder (`--what csr`):
  loop2   what shipped: a Python loop over parents, each iteration slicing the
          coordinates, launching a comparison and syncing on `.item()`; run twice per
          construction, because SparseStorage.__init__ validated and then
          STensor._set_state validated the same arrays again
  vec2    the sortedness predicate as two whole-array kernels, still run twice
  vec1    ... run once, the second walk skipped while a stamp of what was validated
          still matches
  screen  ... and the whole-array torch checks replaced by one fused native pass
          (csrc/native_abi.h, exposed as abi_screen_compressed_level)

COO arms (`--what coo`):
  python  what shipped: `tolist()` per mode, then two tuples built and compared per
          NONZERO -- per nonzero, where the compressed loop beside it was per row
  screen  one fused native pass (abi_screen_lex_levels)

Arms are visited in a fresh random order every round and the figure is the median.
Every arm's result is compared against the others before any of them is timed, so an
arm that got faster by accepting different storage fails here instead of looking good.

Usage:
    python bench/bench_index_validation.py --what both --reps 9
"""

import argparse
import random
import statistics
import time

import numpy as np
import torch

import scorch.storage as storage_module
from scorch import STensor
from scorch.exceptions import TensorIndexError
from scorch.storage import SparseStorage

VECTORIZED = storage_module._check_sorted_within_parents
SKIPPING = SparseStorage.validate_unless_already_checked
NATIVE = (
    storage_module._NATIVE_SCREEN,
    storage_module._NATIVE_BOUNDS,
    storage_module._NATIVE_LEX,
)
NO_NATIVE = (None, None, None)


def loop_check(positions, coordinates, mode):
    """The implementation the vectorized predicate replaced, kept verbatim."""
    position_values = positions.tolist()
    for parent, (start, end) in enumerate(
        zip(position_values[:-1], position_values[1:])
    ):
        segment = coordinates[start:end]
        if segment.numel() > 1 and bool(torch.any(segment[1:] < segment[:-1]).item()):
            raise TensorIndexError(
                f"compressed mode {mode} coordinates must be sorted "
                f"within parent {parent}"
            )


# name -> (sortedness impl, second-walk impl, native screen handles)
CSR_ARMS = {
    "loop2": (loop_check, SparseStorage.validate, NO_NATIVE),
    "vec2": (VECTORIZED, SparseStorage.validate, NO_NATIVE),
    "vec1": (VECTORIZED, SKIPPING, NO_NATIVE),
    "screen": (VECTORIZED, SKIPPING, NATIVE),
}
COO_ARMS = {
    "python": (VECTORIZED, SKIPPING, NO_NATIVE),
    "screen": (VECTORIZED, SKIPPING, NATIVE),
}


def select(arms, name):
    sortedness, second, native = arms[name]
    storage_module._check_sorted_within_parents = sortedness
    SparseStorage.validate_unless_already_checked = second
    (
        storage_module._NATIVE_SCREEN,
        storage_module._NATIVE_BOUNDS,
        storage_module._NATIVE_LEX,
    ) = native


def csr_torch(rows, degree, seed=0):
    generator = np.random.default_rng(seed)
    cols = np.concatenate(
        [
            np.sort(generator.choice(rows, size=degree, replace=False))
            for _ in range(rows)
        ]
    ).astype(np.int32)
    positions = (np.arange(rows + 1) * degree).astype(np.int32)
    values = generator.standard_normal(rows * degree).astype(np.float32)
    return torch.sparse_csr_tensor(
        torch.from_numpy(positions),
        torch.from_numpy(cols),
        torch.from_numpy(values),
        size=(rows, rows),
    )


def coo_torch(nnz, extent, seed=0):
    generator = np.random.default_rng(seed)
    rows = generator.integers(0, extent, size=nnz)
    cols = generator.integers(0, extent, size=nnz)
    order = np.lexsort((cols, rows))
    indices = torch.from_numpy(np.stack([rows[order], cols[order]]))
    values = torch.from_numpy(generator.standard_normal(nnz).astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, size=(extent, extent)).coalesce()


def timed(fn, inner):
    held = []
    start = time.perf_counter()
    for _ in range(inner):
        held.append(fn())
    elapsed = (time.perf_counter() - start) / inner
    del held
    return elapsed


def run(arms, cases, reps, label_width=22):
    names = list(arms)
    header = f"{'case':>{label_width}s}"
    for name in names:
        header += f" {name + '_us':>12s}"
    header += f" {'best/worst':>11s}"
    print(header)
    try:
        for label, call, inner in cases:
            reference = None
            for name in names:
                select(arms, name)
                produced = call()
                if reference is None:
                    reference = produced
                else:
                    assert torch.equal(
                        reference.values, produced.values
                    ), f"{label}: arm {name} disagrees"
            acc = {name: [] for name in names}
            for _ in range(reps):
                order = list(names)
                random.shuffle(order)
                for name in order:
                    select(arms, name)
                    acc[name].append(timed(call, inner))
            us = {k: statistics.median(v) * 1e6 for k, v in acc.items()}
            row = f"{label:>{label_width}s}"
            for name in names:
                row += f" {us[name]:12.1f}"
            row += f" {max(us.values()) / min(us.values()):10.1f}x"
            print(row)
    finally:
        select(arms, names[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", default="both", choices=["csr", "coo", "both"])
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print(f"# reps={args.reps} threads={args.threads}")

    if args.what in ("csr", "both"):
        print("\n## compressed (CSR), from_torch")
        cases = []
        for rows, degree in ((128, 4), (1000, 8), (20000, 24), (100000, 16)):
            tensor = csr_torch(rows, degree)
            cases.append(
                (
                    f"from_torch {rows}x{degree}",
                    lambda t=tensor: STensor.from_torch(t),
                    max(3, min(50, 200000 // rows)),
                )
            )
        dense = torch.rand(2000, 2000)
        dense[dense < 0.99] = 0.0
        try:
            select(CSR_ARMS, "screen")
            STensor.from_torch(dense).to_sparse("ds")
        except Exception as exc:  # pragma: no cover - host without a working JIT
            print(f"# skipping to_sparse: {type(exc).__name__}: {str(exc)[:60]}")
        else:
            cases.append(
                (
                    "to_sparse ds 2000x2000",
                    lambda: STensor.from_torch(dense).to_sparse("ds"),
                    5,
                )
            )
        run(CSR_ARMS, cases, args.reps)

    if args.what in ("coo", "both"):
        print("\n## coordinate (COO), from_torch")
        cases = []
        for nnz, extent in (
            (1000, 500),
            (10000, 2000),
            (100000, 5000),
            (400000, 10000),
            (1000000, 20000),
        ):
            tensor = coo_torch(nnz, extent)
            cases.append(
                (
                    f"from_torch nnz={tensor._nnz()}",
                    lambda t=tensor: STensor.from_torch(t),
                    max(3, min(100, 300000 // tensor._nnz())),
                )
            )
        run(COO_ARMS, cases, args.reps)


if __name__ == "__main__":
    main()
