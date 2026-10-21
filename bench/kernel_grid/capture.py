"""Capture the generated kernels each grid cell's call runs, with their exact arguments.

usage: PYTHONHASHSEED=0 PYTHONPATH=<tree>/src python capture.py <tree> <pairs.json> <out_dir>
           <variants> [<part> <parts>]

<variants> is a comma-separated subset of:
  shipped  the public ``scorch.einsum(expr, *operands, format=result)``, as a user calls it;
  new      the new compiler (``execute_cin_via_loopir``) on the program einsum hands to its
           scheduler, with each operand built in the layout that program declares for it,
           under the automatic plan (``requested_schedule=Schedule()``, as the census that
           listed the pairs called it) and the environment's register-blocking setting;
           run only for pairs listed because the new compiler's C++ is not empty
           (``new_c_nonempty``).  einsum's dispatch cache is cleared first, so the
           scheduler runs even when the shipped variant just compiled the same cell.
           It is followed by ``new_other``, the same call under the other
           register-blocking setting: judged always, and recorded for timing only when its
           kernels differ from ``new``'s.

For every pair (program, dtype) of <pairs.json> and every grid cell (size, fill) this runs the
variant on two input families (``FAMILIES``: the operands as drawn, and the same operands with
empty rows forced, below), each judged; a kernel is correct only when it is correct on both,
and the empty-row family is the one recorded for timing.  Each run goes through a hook on
``utils._load_validated_prepared_kernel``: every generated kernel the call loads is wrapped,
and each ``evaluate`` call records the kernel's ``.so`` path and its arguments.  One call
may run several kernels: the public call converts an operand to another layout before the
program's own kernel, or the result after it, and the new compiler's plan may convert operands
too.  Which operand is converted is a schedule decision, so every kernel call of the route is
recorded, in order, and all of them are timed.  The arguments are saved with ``torch.save``
exactly as production passed them, so the timing step (``time_cells.py``) calls the same
machine code on the same arrays without importing any Scorch tree.  The variant's result is
judged against ``torch.einsum`` in float64 (relative 1e-3 for float32, 1e-12 for float64;
``reference_einsum`` orders the contraction so no intermediate passes 1 GiB); only a correct
route is timed.

One JSON line per (pair, cell, variant) goes to <out_dir>/capture.<tree label>.<part>.jsonl:
the outcome, each kernel's ``.so`` and argument file (``kernels``), the error against the
reference, and the faster of two warm runs of those kernel calls (``warm_ns``, used only to
size the timing step).

Grid cells (``CELLS``): three sizes, each a budget for the largest tensor's element count (the
same extent for every index, ``floor(budget ** (1 / largest rank))``), times two fills (the
fraction of an operand's cells that are nonzero; all-dense operands are full).  The extent is
capped so that extent ** (distinct indices) <= budget ** 1.5, the iteration space of a square
matrix product at that budget (``extents_for``; README.md).
``GRID_SIZES=S`` keeps only the named sizes, for a smoke run.  A pair whose
name fixes a degenerate shape (``unit(1, 1)``, ``zero-both(0, 0)``) runs at that shape only.

Operand values are 0.5 + uniform(0, 1) in float64, then cast; the seed is derived from the
pair's name and the cell, so every tree and every variant sees the same values.  In the
``empty_rows`` family, which is timed, every operand with a compressed level also has its
first and last slices along its first mode empty (from extent 3), and the middle one too from
extent 5, so every cell holds empty rows and no operand is emptied entirely (7ac70f2 sized a
CSR result's rows by a DCSR operand's stored rows; a fill of 0.55 alone almost never leaves a
row empty).  Operands are
built from level arrays made with PyTorch (``sparse_levels``, as in execute_frontier_v9.py), or
by the library for coordinate layouts (``STensor.from_coo``), so no conversion kernel runs
before the call under test unless the call itself chooses one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
import traceback

TREE = os.path.realpath(sys.argv[1])
PAIRS = sys.argv[2]
OUT = sys.argv[3]
VARIANTS = [
    name
    for variant in sys.argv[4].split(",")
    for name in ((variant, "new_other") if variant == "new" else (variant,))
]
PART, PARTS = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)

import torch  # noqa: E402

import scorch  # noqa: E402
import scorch.utils as U  # noqa: E402
from scorch.stensor import STensor  # noqa: E402

assert os.path.realpath(scorch.__file__).startswith(TREE + "/"), scorch.__file__

CELLS = (
    ("S", 4_000, 0.05),
    ("S", 4_000, 0.55),
    ("M", 260_000, 0.05),
    ("M", 260_000, 0.55),
    ("L", 4_000_000, 0.05),
    ("L", 4_000_000, 0.55),
)
TOLERANCE = {"float32": 1e-3, "float64": 1e-12}
LABEL = hashlib.sha256(TREE.encode()).hexdigest()[:8] + "_" + os.path.basename(TREE)


# -- the capture hook ----------------------------------------------------------------------

CALLS: list = []


class _Recording:
    """A loaded kernel whose ``evaluate`` calls are recorded, then forwarded."""

    def __init__(self, module):
        self._module = module

    def evaluate(self, *args):
        CALLS.append((self._module, args))
        return self._module.evaluate(*args)

    def __getattr__(self, name):
        return getattr(self._module, name)


_REAL_LOAD = U._load_validated_prepared_kernel


def _recording_load(prepared):
    return _Recording(_REAL_LOAD(prepared))


U._load_validated_prepared_kernel = _recording_load
for _name in ("scorch.ops", "scorch.stensor"):
    _module = sys.modules.get(_name)
    if _module is not None and hasattr(_module, "_load_validated_prepared_kernel"):
        _module._load_validated_prepared_kernel = _recording_load

# The program einsum hands to its scheduler, deep-copied before the call rebinds operand
# orders (the new compiler's input; as in mlsys's front-door census).
from scorch.compiler.scheduler import Scheduler  # noqa: E402

OPERAND_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SCHEDULED: dict = {"cin": None}
_REAL_SELECT = Scheduler.select_loop_order


def _select(cin, *args, **kwargs):
    if SCHEDULED["cin"] is None:
        SCHEDULED["cin"] = copy.deepcopy(cin)
    return _REAL_SELECT(cin, *args, **kwargs)


Scheduler.select_loop_order = staticmethod(_select)


# -- operands --------------------------------------------------------------------------------


def sparse_levels(dense_values, fmt):
    """The level arrays and stored values that hold ``dense_values`` in ``fmt`` (d and s levels).

    As in execute_frontier_v9.py: built from the nonzero pattern with PyTorch only, so no
    conversion kernel runs before the call under test.
    """

    shape = tuple(dense_values.shape)
    coords = dense_values.nonzero()
    position = torch.zeros(coords.shape[0], dtype=torch.int64)
    count = 1
    mode_indices = []
    for level, kind in enumerate(fmt):
        extent = shape[level]
        key = position * extent + coords[:, level]
        if kind == "d":
            mode_indices.append([])
            position, count = key, count * extent
            continue
        assert kind == "s", fmt
        stored, position = torch.unique(key, sorted=True, return_inverse=True)
        parent = torch.div(stored, extent, rounding_mode="floor") if extent else stored
        pos = torch.zeros(count + 1, dtype=torch.int64)
        pos[1:] = torch.cumsum(torch.bincount(parent, minlength=count), 0)
        mode_indices.append([pos, stored - parent * extent])
        count = stored.numel()
    values = torch.zeros(count, dtype=dense_values.dtype)
    values[position] = dense_values[tuple(coords.t())]
    return mode_indices, values


def build(dense, fmt, name, mode_order=None):
    """``dense`` (logical axes) stored in ``fmt`` with ``mode_order`` (physical -> logical)."""

    order = (
        list(range(dense.dim())) if mode_order is None else [int(m) for m in mode_order]
    )
    physical = dense.permute(order).contiguous()
    if set(fmt) == {"o"}:
        cells = physical.nonzero()
        tensor = STensor.from_coo(
            indices=cells.t().contiguous(),
            values=physical[tuple(cells.t())],
            shape=tuple(physical.shape),
            name=name,
        )
        if order != list(range(dense.dim())):
            raise NotImplementedError("a coordinate operand in another mode order")
        return tensor
    if set(fmt) == {"d"} and mode_order is None:
        return STensor.from_torch(dense.clone(), name)
    mode_indices, values = sparse_levels(physical, fmt)
    return STensor.from_components(
        tuple(physical.shape),
        fmt,
        mode_indices,
        values,
        name=name,
        mode_order=None if mode_order is None else order,
    )


def extents_for(pair, budget):
    expr = pair["expr"]
    terms, output = expr.split("->")[0].split(","), expr.split("->")[1]
    letters = "".join(dict.fromkeys("".join(terms) + output))
    if pair.get("fixed_shape"):
        shape = pair["fixed_shape"]
        return dict(zip(terms[0], shape)) | {
            c: shape[-1] for c in letters if c not in terms[0]
        }
    rank = max(len(t) for t in terms + [output])
    # The largest e with e ** rank <= budget and e ** len(letters) <= budget ** 1.5 (squared,
    # so the comparison is in integers).
    extent = max(
        1,
        min(
            largest_root(budget, max(rank, 1)),
            largest_root(budget**3, 2 * len(letters)),
        ),
    )
    return {c: extent for c in letters}


def largest_root(limit, power):
    """The largest integer e with e ** power <= limit."""

    e = int(round(limit ** (1.0 / power)))
    while e**power > limit:
        e -= 1
    while (e + 1) ** power <= limit:
        e += 1
    return e


# Input families every kernel is judged on; the last is the one timed.
FAMILIES = ("no_empty_rows", "empty_rows")


def dense_operands(pair, cell, empty_rows=True):
    size, budget, fill = cell
    extents = extents_for(pair, budget)
    terms = pair["expr"].split("->")[0].split(",")
    dtype = getattr(torch, pair["dtype"])
    seed = int(
        hashlib.sha256(f"{pair['name']}|{size}|{fill}".encode()).hexdigest()[:12], 16
    )
    denses = []
    for position, (term, fmt) in enumerate(zip(terms, pair["operand_formats"])):
        shape = tuple(extents[c] for c in term)
        generator = torch.Generator().manual_seed(seed + position)
        values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5
        keep = 1.0 if set(fmt) == {"d"} else fill
        dense = values * (torch.rand(shape, generator=generator) < keep)
        if empty_rows and set(fmt) != {"d"} and shape and shape[0] >= 3:
            # Empty the first and last slices along the first mode, and from extent
            # 5 the middle one too: a stored level-0 coordinate missing (an empty
            # DCSR row) or an empty CSR row, before, between and after stored ones.
            # Below extent 5 the middle slices are kept, or nothing would be stored
            # (the six-index programs have extent 3 at the smallest size).  Not
            # below extent 3, so a degenerate pair keeps its fixed entries.
            emptied = {0, shape[0] - 1}
            if shape[0] >= 5:
                emptied.add(shape[0] // 2)
            dense[sorted(emptied)] = 0
        denses.append(dense.to(dtype))
    return extents, denses


# -- the reference ---------------------------------------------------------------------------

#: The largest intermediate the reference may hold, in elements (1 GiB of float64).
REFERENCE_LIMIT = 1 << 27


def reference_einsum(expr, denses):
    """``torch.einsum`` in float64, contracted two operands at a time, smallest result first.

    Without opt_einsum installed ``torch.einsum`` contracts left to right, so
    ``ij,ik,jk->ij`` at extent 2000 first builds ``ijk``: 8e9 float64, 64 GB, in every capture
    process that reaches the cell.  Contracting ``ik,jk->ij`` first needs 4e6.  A program
    whose smallest pairwise intermediate still passes ``REFERENCE_LIMIT`` stops the capture
    with a message instead of allocating it.
    """

    inputs, output = expr.split("->")
    terms = inputs.split(",")
    operands = [dense.to(torch.float64) for dense in denses]
    extent = {c: n for term, op in zip(terms, operands) for c, n in zip(term, op.shape)}

    def elements(letters):
        count = 1
        for c in set(letters):
            count *= extent[c]
        return count

    while len(terms) > 2:
        best = None
        for a in range(len(terms)):
            for b in range(a + 1, len(terms)):
                later = output + "".join(
                    term for n, term in enumerate(terms) if n not in (a, b)
                )
                kept = "".join(
                    c for c in dict.fromkeys(terms[a] + terms[b]) if c in later
                )
                cost = (elements(kept), elements(terms[a] + terms[b]), a, b)
                if best is None or cost < best[0]:
                    best = (cost, kept)
        (size, _, a, b), kept = best
        if size > REFERENCE_LIMIT:
            raise MemoryError(
                f"the reference for {expr} needs an intermediate of {size} elements"
            )
        merged = torch.einsum(
            f"{terms[a]},{terms[b]}->{kept}", operands[a], operands[b]
        )
        terms = [term for n, term in enumerate(terms) if n not in (a, b)] + [kept]
        operands = [op for n, op in enumerate(operands) if n not in (a, b)] + [merged]
    return torch.einsum(",".join(terms) + "->" + output, *operands)


# -- judging ---------------------------------------------------------------------------------


def judge(result, reference, dtype):
    got = result.to_torch(in_place=False) if isinstance(result, STensor) else result
    got = got.to(torch.float64)
    if got.shape != reference.shape:
        return "wrong_shape", None
    scale = max(1.0, float(reference.abs().max())) if reference.numel() else 1.0
    error = float((got - reference).abs().max()) if reference.numel() else 0.0
    return ("correct" if error <= TOLERANCE[dtype] * scale else "wrong_values"), error


MAIN: dict = {"call": None}


def warm_ns():
    """The faster of two more runs of the route's kernel calls; only sizes the timing step.

    Called after the result is judged, so a kernel that writes into an argument cannot change
    the result under judgement.
    """

    times = []
    for _ in range(2):
        started = time.perf_counter_ns()
        for module, args in MAIN["call"]:
            module.evaluate(*args)
        times.append(time.perf_counter_ns() - started)
    return min(times)


def record_call(key, variant, before):
    """Save every kernel call the route made, in order; return the record's fields.

    A route may run more than one generated kernel for one operation: the public call moves
    an operand to another layout with a conversion kernel before the main kernel, or the
    result after it, and the new compiler's plan may convert operands as well.  Which
    operand is moved is a schedule decision, so the route's cost is all of its kernel
    calls, and all of them are timed.
    """

    calls = CALLS[before:]
    if not calls:
        return {"kernel": None}
    kernels = []
    for position, (module, args) in enumerate(calls):
        so_path = module.__file__
        args_file = os.path.join(OUT, "args", f"{key}.{variant}.{LABEL}.{position}.pt")
        torch.save(list(args), args_file)
        kernels.append(
            dict(
                kernel=so_path,
                args=args_file,
                sha256=hashlib.sha256(open(so_path, "rb").read()).hexdigest(),
            )
        )
    MAIN["call"] = list(calls)
    return {
        # The last kernel called; ``kernels`` is what is timed.
        "kernel": kernels[-1]["kernel"],
        "kernels": kernels,
        "kernel_calls": len(kernels),
        "kernel_sha256": hashlib.sha256(
            "".join(k["sha256"] for k in kernels).encode()
        ).hexdigest(),
    }


def run_shipped(pair, cell, key, denses):
    operands = [
        build(dense, fmt, OPERAND_NAMES[n])
        for n, (dense, fmt) in enumerate(zip(denses, pair["operand_formats"]))
    ]
    before = len(CALLS)
    result = scorch.einsum(pair["expr"], *operands, format=pair["result_format"])
    return result, record_call(key, "shipped", before)


def run_new(pair, cell, key, denses, other_setting=False):
    from dataclasses import replace

    from scorch.compiler.cin import ForAll, TensorAccess, TensorAssign
    from scorch.compiler.compile_options import CompileOptions
    from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
    from scorch.compiler.scheduler import Schedule

    options = CompileOptions.from_environment()
    if other_setting:
        options = options.with_regblock_enabled(not options.scheduler.regblock_enabled)
    options = replace(options, requested_schedule=Schedule())

    SCHEDULED["cin"] = None
    getattr(scorch.ops, "_einsum_dispatch_cache", {}).clear()
    operands = [
        build(dense, fmt, OPERAND_NAMES[n])
        for n, (dense, fmt) in enumerate(zip(denses, pair["operand_formats"]))
    ]
    scorch.einsum(pair["expr"], *operands, format=pair["result_format"])
    cin = SCHEDULED["cin"]
    if cin is None:
        return None, {"kernel": None, "note": "einsum built no scheduled program"}
    stmt = cin
    while isinstance(stmt, ForAll):
        stmt = stmt.stmt
    assert isinstance(stmt, TensorAssign)
    accesses = []

    def collect(expr):
        if isinstance(expr, TensorAccess):
            accesses.append(expr)
        else:
            collect(expr.left)
            collect(expr.right)

    collect(stmt.rhs)
    # einsum names its operands A, B, C, ... in call order and the result after them.
    by_name = dict(zip(OPERAND_NAMES, denses))
    assert stmt.lhs.tensor.name == OPERAND_NAMES[len(denses)], stmt.lhs.tensor.name
    built = []
    for access in accesses:
        tensor = access.tensor
        letters = "".join(
            {"DENSE": "d", "COMPRESSED": "s", "COORDINATE": "o"}[level.name]
            for level in tensor.get_format().get_level_types()
        )
        order = tensor.mode_order
        identity = order is None or list(order) == list(range(len(letters)))
        built.append(
            build(
                by_name[tensor.name], letters, tensor.name, None if identity else order
            )
        )
    before = len(CALLS)
    result, _ = execute_cin_via_loopir(
        copy.deepcopy(cin),
        tuple(stmt.lhs.tensor.shape),
        *built,
        compile_options=options,
    )
    return result, record_call(key, "new_other" if other_setting else "new", before)


def run_new_other(pair, cell, key, denses):
    return run_new(pair, cell, key, denses, other_setting=True)


def main():
    pairs = json.load(open(PAIRS))
    os.makedirs(os.path.join(OUT, "args"), exist_ok=True)
    out = open(os.path.join(OUT, f"capture.{LABEL}.{PART}.jsonl"), "a")
    sizes = os.environ.get("GRID_SIZES", "S,M,L").split(",")
    work = [
        (pair, cell)
        for pair in pairs
        for cell in (CELLS[:2] if pair.get("fixed_shape") else CELLS)
        if cell[0] in sizes
    ]
    for index, (pair, cell) in enumerate(work):
        if index % PARTS != PART:
            continue
        key = hashlib.sha256(
            f"{pair['name']}|{pair['dtype']}|{cell[0]}|{cell[2]}".encode()
        ).hexdigest()[:16]
        inputs = {}
        for family in FAMILIES:
            extents, denses = dense_operands(pair, cell, family == "empty_rows")
            reference = reference_einsum(pair["expr"], denses)
            inputs[family] = (denses, reference)
        new_kernel = None
        for variant in VARIANTS:
            if variant != "shipped" and "new_c_nonempty" not in pair["why"]:
                continue
            base = dict(
                key=key,
                tree=TREE,
                label=LABEL,
                variant=variant,
                name=pair["name"],
                dtype=pair["dtype"],
                why=pair["why"],
                size=cell[0],
                fill=cell[2],
                extents=extents,
                operand_nonzeros=[int((d != 0).sum()) for d in inputs["empty_rows"][0]],
            )
            started = time.time()
            runner = {
                "shipped": run_shipped,
                "new": run_new,
                "new_other": run_new_other,
            }[variant]
            families = {}
            fields = {}
            for family in FAMILIES:
                denses, reference = inputs[family]
                # Each run's calls are its own: the list holds the arguments' arrays.
                CALLS.clear()
                MAIN["call"] = None
                try:
                    result, fields = runner(pair, cell, key, denses)
                    outcome, error = (
                        judge(result, reference, pair["dtype"])
                        if result is not None
                        else ("not_run", None)
                    )
                    families[family] = dict(outcome=outcome, max_error=error)
                except Exception as error:  # noqa: BLE001 -- an outcome, recorded
                    fields = {}
                    families[family] = dict(
                        outcome=f"raised:{type(error).__name__}",
                        message=str(error)[:300],
                        where=traceback.format_exc().splitlines()[-3:],
                    )
            # The timed family ran last, so the recorded kernel and arguments are its.
            base.update(fields, families=families)
            failed = [f for f in FAMILIES if families[f]["outcome"] != "correct"]
            if not failed:
                base["outcome"] = "correct"
                if MAIN["call"] is not None:
                    try:
                        base["warm_ns"] = warm_ns()
                    except Exception as error:  # noqa: BLE001 -- an outcome
                        base.update(
                            outcome=f"raised:{type(error).__name__}",
                            message=f"on a repeated call: {str(error)[:260]}",
                        )
            else:
                first = families[failed[0]]
                base["outcome"] = first["outcome"]
                base["failed_families"] = failed
                base["message"] = (
                    f"[{', '.join(failed)}] " + first.get("message", "")
                ).strip()
            base["seconds"] = round(time.time() - started, 3)
            if variant == "new":
                new_kernel = base.get("kernel_sha256")
            elif variant == "new_other" and base.get("kernel_sha256") == new_kernel:
                # Judged above; the same machine code as ``new`` is not timed twice.
                base.update(kernel=None, same_kernel_as="new")
            out.write(json.dumps(base) + "\n")
            out.flush()
            CALLS.clear()
            MAIN["call"] = None
    out.write(json.dumps({"part_done": PART, "parts": PARTS}) + "\n")


if __name__ == "__main__":
    main()
