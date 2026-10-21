"""Time the captured kernels of each grid cell side by side, in one process.

usage: python time_cells.py <out_dir> <process index> <types tree> [<part> <parts>]

Reads every capture.*.jsonl in <out_dir>.  For each cell (pair, size, fill) whose captures are
correct it loads every kernel ``.so`` a variant's route called, and a second copy of each (the
same file copied under ``<out_dir>/aa/<process>/``, which the loader maps as a separate image:
the A/A control), and calls each module's ``evaluate`` with the arguments production passed
it (``capture.py``).  A route that ran several kernels for the operation (a conversion of an
operand or of the result beside the program's own kernel) is timed as all of them, called in
order.  No Scorch tree is imported, so kernels built by different trees run in one process on
the same machine state.  The one exception is the ``scorch_ops`` extension of
<types tree>, loaded by path before any kernel: a generated kernel returns the C++ ``Tensor``
struct of ``csrc/header.h``, and ``scorch_ops`` registers its Python type, as it does in
production.  ``run_host.sh`` requires the two trees to define those structs identically.

Each round visits every arm once, in an order drawn afresh from a seeded generator, and times a
batch of ``calls`` back-to-back calls with ``time.perf_counter_ns``.  ``calls`` is set once per
cell from the slowest arm's warm call so that a batch lasts at least ``BATCH_NS`` and is the same
for every arm; an arm slower than ``BATCH_NS`` is called once per round (a route that converts
a 2000 x 2000 operand takes about half a second, and three calls of it per round would triple
the run for no narrower an interval).  Before timing, each copy's result must equal its
primary's exactly, so a module that loaded another kernel, or a kernel that carries state
between calls, is caught and its cell skipped.

Arms: each captured variant and its copy, and one positive control: the first variant called
twice per counted call, which the analysis must report as a regression (``analyze.py``).

One JSON line per (cell, arm, round): <out_dir>/times.<process>.<part>.jsonl.
"""

from __future__ import annotations

import gc
import glob
import importlib.util
import json
import os
import shutil
import sys
import time
from collections import defaultdict

import torch

OUT = sys.argv[1]
PROCESS = int(sys.argv[2])
TYPES_TREE = sys.argv[3]
PART, PARTS = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
ROUNDS = int(os.environ.get("GRID_ROUNDS", "21"))
BATCH_NS = int(os.environ.get("GRID_BATCH_NS", "2000000"))
MAX_CALLS = 2000

LOADED: dict = {}


def load(path: str, copy_dir: str = ""):
    """The extension module at ``path``, or at a copy of it under ``copy_dir``."""

    key = (path, copy_dir)
    if key in LOADED:
        return LOADED[key]
    target = path
    if copy_dir:
        os.makedirs(copy_dir, exist_ok=True)
        target = os.path.join(copy_dir, os.path.basename(path))
        if not os.path.exists(target):
            shutil.copyfile(path, target)
    name = os.path.basename(path).split(".")[0]
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    LOADED[key] = module
    return module


def register_result_types():
    """Load <types tree>'s ``scorch_ops`` by path: it registers the kernels' ``Tensor`` type."""

    paths = glob.glob(os.path.join(TYPES_TREE, "src", "scorch_ops*.so"))
    assert len(paths) == 1, paths
    spec = importlib.util.spec_from_file_location("scorch_ops", paths[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "Tensor") and "scorch" not in sys.modules
    return paths[0]


def as_dense(value):
    """A kernel's raw result as one float64 tensor, for the cross-arm check."""

    # torch.Tensor first: it has a ``storage`` method of its own.
    if isinstance(value, torch.Tensor):
        return value.to(torch.float64).reshape(-1)
    if hasattr(value, "storage"):  # the C++ Tensor struct: index arrays, then values
        storage = value.storage
        return as_dense([*storage.index.mode_indices, storage.value])
    if isinstance(value, (list, tuple)):
        parts = [as_dense(v) for v in value]
        return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)
    return torch.tensor([float(value)], dtype=torch.float64)


def route(steps):
    """One counted call of an arm: every kernel call of the route, in order."""

    def run():
        result = None
        for call, args in steps:
            result = call(*args)
        return result

    return run


def main():
    types_from = register_result_types()
    captures = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(OUT, "capture.*.jsonl"))):
        for line in open(path):
            record = json.loads(line)
            if "key" not in record or record.get("outcome") != "correct":
                continue
            if not record.get("kernel"):
                continue
            arm = f"{record['variant']}@{record['label']}"
            captures[record["key"]][arm] = record
    keys = sorted(k for k, arms in captures.items() if len(arms) >= 2)
    out = open(os.path.join(OUT, f"times.{PROCESS}.{PART}.jsonl"), "a")
    out.write(json.dumps(dict(result_types_from=types_from, process=PROCESS)) + "\n")
    generator = torch.Generator().manual_seed(1000 * PROCESS + PART)
    copy_dir = os.path.join(OUT, "aa", str(PROCESS))
    gc.disable()
    for index, key in enumerate(keys):
        if index % PARTS != PART:
            continue
        arms_records = captures[key]
        names = sorted(arms_records)
        arms = []
        for name in names:
            record = arms_records[name]
            loaded = [
                (kernel["kernel"], torch.load(kernel["args"], weights_only=False))
                for kernel in record["kernels"]
            ]
            primary = route([(load(path).evaluate, args) for path, args in loaded])
            twin = route(
                [(load(path, copy_dir).evaluate, args) for path, args in loaded]
            )
            arms.append((name, primary, record))
            arms.append((name + "#aa", twin, record))
        first_name, first_call, first_record = arms[0]

        def doubled(_call=first_call):
            _call()
            return _call()

        arms.append((first_name + "#x2", doubled, first_record))
        # Each copy must return exactly what its primary returns: a module that loaded
        # another kernel, or state carried between calls, fails here and the cell is
        # skipped.  Different variants may store different arrays for one result (a
        # sparse result may keep the exact zeros its loops visit), so across variants
        # only the stored-array lengths are recorded, not compared.
        outputs = {name: as_dense(call()) for name, call, _ in arms}
        broken = [
            name
            for name in outputs
            if name.endswith("#aa")
            and not torch.equal(outputs[name], outputs[name[: -len("#aa")]])
        ]
        if broken:
            out.write(
                json.dumps(dict(key=key, skipped="copy differs", arms=broken)) + "\n"
            )
            continue
        mismatched = sorted(
            name for name in outputs if outputs[name].shape != outputs[first_name].shape
        )
        # Calibrate one batch length for every arm from the slowest warm call.
        slowest = 0
        for name, call, _ in arms:
            for _ in range(2):
                call()
            started = time.perf_counter_ns()
            call()
            slowest = max(slowest, time.perf_counter_ns() - started)
        calls = max(1, min(MAX_CALLS, -(-BATCH_NS // max(slowest, 1))))
        for round_index in range(ROUNDS):
            order = torch.randperm(len(arms), generator=generator).tolist()
            for position, arm_index in enumerate(order):
                name, call, record = arms[arm_index]
                started = time.perf_counter_ns()
                for _ in range(calls):
                    call()
                elapsed = time.perf_counter_ns() - started
                out.write(
                    json.dumps(
                        dict(
                            key=key,
                            name=record["name"],
                            dtype=record["dtype"],
                            why=record["why"],
                            size=record["size"],
                            fill=record["fill"],
                            arm=name,
                            process=PROCESS,
                            round=round_index,
                            position=position,
                            calls=calls,
                            kernel_calls=len(record["kernels"]),
                            ns_per_call=elapsed / calls,
                            mismatched=mismatched,
                        )
                    )
                    + "\n"
                )
        out.flush()
        gc.collect()
    out.write(
        json.dumps({"part_done": PART, "parts": PARTS, "process": PROCESS}) + "\n"
    )


if __name__ == "__main__":
    main()
