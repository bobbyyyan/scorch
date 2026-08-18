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

import scorch
import scorch.storage as storage_module
from scorch import STensor
from scorch.exceptions import TensorIndexError
from scorch.storage import SparseStorage

# What the module ships with, so the sweep restores it rather than a literal.
DEFAULT_GRAIN = storage_module._WRAP_GRAIN

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


def scan_sweep(reps, interpose):
    """The grain, measured on the scan and not on the allocator.

    Sweeping the grain through ``from_torch`` cannot resolve it. At 480k and 1.6M
    nonzeros that call clones 13-25 MB of index arrays, and on redwood the same grain
    read 141.5, 784.0 and 418.7 us across three runs of the identical configuration --
    a 5.5x spread, against a grain effect of a few percent to 2x. The small cells were
    steady to 2% in all three, which places the variance in the large allocations.

    So this times ``_validate_index_storage`` on index arrays that already exist: no
    clone, no allocation, nothing in the loop but the thing the grain controls.

    ``interpose`` runs between validations. With a torch operation there it reproduces
    what the wrap path really does -- alternate between torch's thread team and the
    screen's -- which is where a team reshape would show up. With ``None`` the screen
    runs back to back and keeps its team, which is the intrinsic scan cost.
    """
    grains = [4096, 16384, 65536, 262144, 1048576]
    cases = []
    shapes = ((1000, 8), (2000, 10), (4000, 8), (8000, 8), (20000, 24), (100000, 16))
    for rows, degree in shapes:
        storage = STensor.from_torch(csr_torch(rows, degree))._storage
        cases.append((f"csr {rows}x{degree}", storage))
    for n in (20000, 100000, 1000000):
        storage = STensor.from_torch(coo_torch(n, max(500, n // 50)))._storage
        cases.append((f"coo nnz={n}", storage))

    header = f"{'case':>16s} {'nnz':>9s}"
    for grain in grains:
        header += f" {str(grain) + '_us':>12s}"
    print(header + f" {'best':>9s} {'serial/best':>12s}")
    for label, storage in cases:
        layout = storage._layout
        modes = storage._mode_indices
        value = storage._value
        nnz = value.numel()
        # Enough iterations that a sample is milliseconds, so the clock is not the
        # measurement; capped so the whole sweep stays minutes.
        inner = max(5, min(400, 4_000_000 // max(nnz, 1)))
        spare = torch.empty_like(value) if interpose else None

        def once():
            if interpose:
                spare.copy_(value)
            storage_module._validate_index_storage(layout, modes, value)

        samples = {grain: [] for grain in grains}
        for _ in range(reps):
            order = list(grains)
            random.shuffle(order)
            for grain in order:
                storage_module._WRAP_GRAIN = grain
                samples[grain].append(timed(once, inner))
        storage_module._WRAP_GRAIN = DEFAULT_GRAIN
        timings = {g: statistics.median(v) * 1e6 for g, v in samples.items()}
        row = f"{label:>16s} {nnz:9d}"
        for grain in grains:
            row += f" {timings[grain]:12.2f}"
        best = min(timings, key=timings.get)
        # 1048576 leaves every case here serial (all are under a million nonzeros
        # except the last, which gets one worker), so it is the serial baseline.
        print(f"{row} {best:9d} {timings[1048576] / timings[best]:11.3f}x")


def adopt_ab(reps):
    """What the two changes to a *generated result's* wrap are worth, in one process.

    Wrapping a kernel's sparse result used to do two things it did not need to do: copy the
    index arrays the kernel had just allocated for its own output, and then walk them to
    check a structure our own codegen had just built. Three arms, in the order they were
    removed:

      copy+walk    what shipped before: `TensorIndex` clones, `SparseStorage` validates
      adopt+walk   arrays taken as they are, still validated
      adopt        what ships now: taken as they are, walk left to the test suite

    Both switches are cells, so all three arms are the same binary and differ only in the
    two booleans. The arms are checked for agreement before anything is timed -- and for
    *disagreement* on identity, since "adoption is on" is otherwise indistinguishable from
    "the flag did nothing".
    """
    from scorch import stensor as stensor_module

    cases = []
    for rows, degree in ((200, 4), (2000, 8), (20000, 16)):
        left = STensor.from_torch(csr_torch(rows, degree), "L").to_sparse("ds")
        cases.append((f"ds {rows}x{degree}", left, torch.rand(rows, 8)))

    ARMS = ("copy+walk", "adopt+walk", "adopt")

    def configure(arm):
        stensor_module._ADOPT_CELL[0] = arm != "copy+walk"
        storage_module._VALIDATE_KERNEL_RESULTS[0] = arm != "adopt"

    print(
        f"{'case':>16s} {'nnz':>8s} {'copy+walk':>10s} {'adopt+walk':>11s} "
        f"{'adopt':>8s} {'adopt/base':>11s} {'walk saved':>11s}"
    )
    for label, left, right in cases:
        nnz = left.values.numel()

        def call():
            return scorch.einsum("ik,kj->ij", left, right, format="ds")

        # Every arm must give the same answer; only the adopting ones may share storage
        # with what the kernel returned.
        results = {}
        for arm in ARMS:
            configure(arm)
            results[arm] = call()
        for arm in ARMS[1:]:
            torch.testing.assert_close(
                results["copy+walk"].to_torch().to_dense(),
                results[arm].to_torch().to_dense(),
                atol=1e-3,
                rtol=1e-3,
            )

        samples = {arm: [] for arm in ARMS}
        for _ in range(reps):
            order = list(ARMS)
            random.shuffle(order)
            for arm in order:
                configure(arm)
                samples[arm].append(timed(call, 20))
        configure("adopt")
        us = {a: statistics.median(v) * 1e6 for a, v in samples.items()}
        print(
            f"{label:>16s} {nnz:8d} {us['copy+walk']:10.2f} {us['adopt+walk']:11.2f} "
            f"{us['adopt']:8.2f} {us['copy+walk'] / us['adopt']:10.3f}x "
            f"{us['adopt+walk'] / us['adopt']:10.3f}x"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--what",
        default="both",
        choices=["csr", "coo", "both", "grain", "scan", "adopt"],
    )
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print(f"# reps={args.reps} threads={args.threads}")

    if args.what == "adopt":
        print("\n## adopting a kernel's index arrays vs copying them")
        adopt_ab(args.reps)
        return

    if args.what == "scan":
        print("\n## grain on the scan alone (back-to-back, screen keeps its team)")
        scan_sweep(args.reps, interpose=False)
        print("\n## grain with a torch operation interposed (team reshapes each call)")
        scan_sweep(args.reps, interpose=True)
        return

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

    if args.what == "grain":
        print("\n## nonzeros per validation worker (median of reps)")
        grain_sweep(args.reps)
        return

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


def grain_sweep(reps):
    """How many nonzeros a validation worker should get, on this machine.

    This sweeps the grain through the whole of `from_torch`, which is the cost a caller
    actually pays -- but it CANNOT settle the grain, and the numbers below should not be
    read as if it could. At 480k and 1.6M nonzeros the call clones 13-25 MB of index
    arrays, and on redwood one configuration read 141.5, 784.0 and 418.7 us across three
    runs of it. A 5.5x spread swamps a grain effect of a few percent to 3x. The small
    cells held to 2% in all three runs, which places the variance in the large
    allocations rather than in the machine.

    Use `--what scan` to choose the grain; it times the scan on arrays that already
    exist. Use this to confirm the choice is not visibly worse end to end.
    """
    grains = [4096, 16384, 65536, 262144, 1048576]
    shapes = [(128, 4), (1000, 8), (20000, 24), (100000, 16)]
    coo_sizes = [1000, 100000, 1000000]

    tensors = [
        (f"csr {r}x{d}", csr_torch(r, d), max(3, min(50, 200000 // r)))
        for r, d in shapes
    ]
    tensors += [
        (f"coo nnz={n}", coo_torch(n, max(500, n // 50)), max(3, min(100, 300000 // n)))
        for n in coo_sizes
    ]

    header = f"{'case':>16s}"
    for grain in grains:
        header += f" {str(grain) + '_us':>13s}"
    print(header + f" {'best':>9s} {'vs default':>11s}")
    for label, tensor, inner in tensors:
        select(CSR_ARMS, "screen")
        samples = {grain: [] for grain in grains}
        for _ in range(reps):
            # Fresh random order every rep. Visiting the grains in ascending order
            # once each -- what this did first -- makes any warming or thermal trend
            # read as a grain effect.
            order = list(grains)
            random.shuffle(order)
            for grain in order:
                storage_module._WRAP_GRAIN = grain
                samples[grain].append(
                    timed(lambda t=tensor: STensor.from_torch(t), inner)
                )
        storage_module._WRAP_GRAIN = DEFAULT_GRAIN
        timings = {g: statistics.median(v) * 1e6 for g, v in samples.items()}
        row = f"{label:>16s}"
        for grain in grains:
            row += f" {timings[grain]:13.1f}"
        best = min(timings, key=timings.get)
        print(f"{row} {best:9d} {timings[DEFAULT_GRAIN] / timings[best]:10.3f}x")


if __name__ == "__main__":
    main()
