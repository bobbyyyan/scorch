"""Dense operands looped out of their storage order under a dense workspace.

``scorch.einsum`` picks a loop order, lays the operands out to match it, and
can then pick the loop order again (the reorder a sparse result forces).  An
all-dense operand laid out for the first order can be read in the second with
its outer level looped inside its inner one: ``B[c,d]`` stored ``d``-major
under loops ``c, d``.  A dense level is random access, so this is legal; its
position is the parent's position times its extent plus its coordinate, and
the lowering declares those positions parents first once the loops have bound
their coordinates (``_parents_first`` in ``iter_lattice.py``).

Until 2026-10-12 the lowering refused every such program whose producer fills
a dense workspace, with an ``UnsupportedFeature`` that carried no diagnostic
code (raised in ``CINLowerer._dense_level_extent_accesses``).  The refusal now
covers only what cannot be lowered -- a COMPRESSED level looped outside a level
above it -- and carries a code.

The loop order follows Python's string hashing (the scheduler iterates a set
of index names), so each program runs in its own child process with
``PYTHONHASHSEED=0``, the seed at which they were found.  Each child also
records whether the program reached the case: a dense operand level looped
outside a level above it in a dense workspace's producer.  The programs found
refused are required to reach it, so the test cannot pass by the scheduler
choosing some other order.

One case per program and result layout (145): each builds one kernel, about
20 s, so pytest-xdist spreads the builds.  Until 2026-10-17 a case held a
whole family and result layout, up to 64 builds one after another in one
child (1380 s with the node to itself), which ran past the child's 3600 s
timeout when 15 pytest-xdist workers shared the node.

Every result is judged by ``test_dense_below_compressed_result._judge``:
canonical level arrays, float64 values within 1e-12 times max(1, the largest
reference magnitude) of a float64 ``torch.einsum`` (until 2026-10-17 the bound
was 1e-3, which cannot see a result computed at float32 precision; that file's
controls show the bound fails a NaN and float32 rounding), and stored
coordinates equal to what the operands' stored entries imply.  Until
senior-engineer-dennis's relayout fix (eefc9c04), 22 program-and-result pairs
into ``ss`` and ``ds`` missed the last check, with values right, because
``STensor.change_mode_order`` kept the format letters by storage position and
dropped stored zeros.  Every program now stores the implied structure.

After that fix two of the programs refused before no longer loop a dense
operand level outside a level above it; they are listed in
``NOW_LOOPED_IN_ORDER`` and still judged like every other program.

This file passes on the shipped compiler with senior-engineer-ken's patches
0000-0018 and senior-engineer-dennis's relayout fix (eefc9c04) applied.
Without ken's patches, 16 programs of ``abc,abd,cd->ab`` into ``sd`` stop with
a ``CompilerInvariantError`` in workspace placement (0009) and 5 into ``ds``
are refused as ``invalid_compressed_where_root_member`` (0008).
"""

import itertools
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

# The extents of the scan that found these programs.
EXTENT = dict(a=3, b=9, c=4, d=5, e=2)
KINDS = ("ragged", "ends", "stored_zero", "empty", "hollow")


def _layouts(rank):
    return ["".join(p) for p in itertools.product("ds", repeat=rank)]


# (expression, operand layouts) for every program of the four families the
# scan covered: one operand all dense, every layout of the others.
FAMILIES = {
    "abc,bd,cd->ad": [
        ("ddd", b, c) for b, c in itertools.product(_layouts(2), _layouts(2))
    ],
    "abc,abd,cd->ab": [
        (a, b, "dd") for a, b in itertools.product(_layouts(3), _layouts(3))
    ],
    "abc,cd,d->ab": [(a, "dd", d) for a, d in itertools.product(_layouts(3), "ds")],
    "ab,bc,cd,de->ae": [("ds", "dd", "sd", "ds")],
}

# The programs the shipped compiler refused into an ``sd`` result at
# PYTHONHASHSEED=0 (float64, the extents above).
REFUSED_BEFORE = {
    "abc,bd,cd->ad": {("ddd", "sd", "sd"), ("ddd", "ss", "ss")},
    "abc,abd,cd->ab": {
        ("dds", "ssd", "dd"),
        ("dds", "sss", "dd"),
        ("dsd", "sds", "dd"),
        ("dss", "dds", "dd"),
        ("dss", "dsd", "dd"),
        ("sdd", "dsd", "dd"),
        ("sdd", "ssd", "dd"),
        ("sds", "dsd", "dd"),
        ("sds", "dss", "dd"),
        ("sds", "ssd", "dd"),
        ("sds", "sss", "dd"),
        ("ssd", "dds", "dd"),
        ("ssd", "sds", "dd"),
        ("ssd", "sss", "dd"),
    },
    "abc,cd,d->ab": {
        ("dds", "dd", "s"),
        ("dsd", "dd", "d"),
        ("dsd", "dd", "s"),
        ("sdd", "dd", "s"),
        ("ssd", "dd", "d"),
        ("ssd", "dd", "s"),
        ("sss", "dd", "s"),
    },
    "ab,bc,cd,de->ae": {("ds", "dd", "sd", "ds")},
}


# Programs refused before that no longer reach the case into ``sd`` since
# senior-engineer-dennis's relayout fix (eefc9c04): after the relayout
# ``scorch.einsum`` does first, no dense workspace's producer loops one of their
# dense levels outside a level above it.  Each must still have been relaid.
# Strict: a listed program that reaches the case again fails until it leaves
# this table, so the premise below stays exact.
NOW_LOOPED_IN_ORDER = {
    "abc,abd,cd->ab": {
        ("dss", "dsd", "dd"),
        ("ssd", "sss", "dd"),
    },
}


def _cases():
    cases = []
    for expr, programs in FAMILIES.items():
        # Every program into ``sd``; the ones refused before also into the
        # neighbouring sparse layouts.
        cases.extend((expr, "sd", program) for program in programs)
        refused = sorted(REFUSED_BEFORE[expr])
        for result_fmt in ("ss", "ds"):
            cases.extend((expr, result_fmt, program) for program in refused)
    return cases


# A case builds one kernel: about 20 s (64 took 1380 s with the node to
# itself, 21.6 s each).  600 s is 28 times that, room for a loaded node, and
# still bounds a hung build.
_CHILD_TIMEOUT = 600


def _child(expr, result_fmt, programs):
    """Run in the child: every program on every input, judged; one JSON row each."""

    import torch

    import scorch
    from scorch import ops
    from scorch.compiler.cin import ForAll
    from scorch.compiler.cin_lowerer import CINLowerer
    from scorch.format import LevelType
    from scorch.stensor import STensor
    from tests.test_scorch.test_dense_below_compressed_result import (
        _inputs,
        _judge,
        _operand,
    )

    reached = []
    original = CINLowerer._dense_level_extent_accesses

    def recording(self, stmt, index_var):
        # The premise: some dense operand level is looped outside the loop
        # over a level above it in this dense workspace's producer.
        order = list(self._enclosing_forall_vars)
        body = stmt.producer
        while isinstance(body, ForAll):
            order.append(body.get_index_var())
            body = body.stmt
        logical = [v.parent if v.has_parent else v for v in order]
        for access in stmt.producer.get_rhs_tensor_accesses():
            if access.is_workspace():
                continue
            level_vars = access.get_sorted_index_vars()
            if any(v not in logical for v in level_vars):
                continue
            depths = [logical.index(v) for v in level_vars]
            for level, level_type in enumerate(access.level_types()):
                if level_type == LevelType.DENSE and any(
                    depth > depths[level] for depth in depths[:level]
                ):
                    reached.append(access.tensor.get_name())
        return original(self, stmt, index_var)

    CINLowerer._dense_level_extent_accesses = recording

    # Whether a relayout moved an operand (``relaid``), and whether it moved a
    # format letter to another logical index (``relabelled``, reported only).
    relabelled, moved = [], []
    original_relayout = STensor.change_mode_order

    def recording_relayout(self, mode_order, **kwargs):
        def letters(tensor):
            kinds = str(tensor.format).split(",")
            order = (
                list(range(len(kinds)))
                if tensor.mode_order is None
                else [int(m) for m in tensor.mode_order]
            )
            return {order[level]: kind for level, kind in enumerate(kinds)}

        before = letters(self)
        previous = None if self.mode_order is None else list(self.mode_order)
        relaid = original_relayout(self, mode_order, **kwargs)
        if letters(relaid) != before:
            relabelled.append(self.name)
        if list(mode_order) != (previous or list(range(len(before)))):
            moved.append(self.name)
        return relaid

    STensor.change_mode_order = recording_relayout
    terms = expr.split("->")[0].split(",")
    rows = []
    for operand_fmts in programs:
        # Every program lowers its own kernel.  The in-process kernel cache keys a
        # kernel by the scheduled program and its operands' formats and orders, so
        # a program whose relaid operands match an earlier program's would take
        # that kernel without lowering, and read as not reaching the case (five
        # programs refused before did, in a run of all 64 into ``sd``).
        ops._kernel_cache.clear()
        ops._einsum_dispatch_cache.clear()
        seed = zlib.crc32(",".join(operand_fmts).encode()) % 997
        reached.clear()
        relabelled.clear()
        moved.clear()
        failures = []
        for kind in KINDS:
            try:
                operands, indicators, denses = [], [], []
                pairs = _inputs(kind, terms, seed, EXTENT, torch.float64)
                for (dense, pattern, ghosts), fmt, name in zip(
                    pairs, operand_fmts, "ABCD"
                ):
                    operand, indicator = _operand(dense, pattern, fmt, name, ghosts)
                    operands.append(operand)
                    indicators.append(indicator)
                    denses.append(dense)
                result = scorch.einsum(expr, *operands, format=result_fmt)
                _judge(
                    result,
                    result_fmt,
                    torch.einsum(expr, *denses),
                    torch.einsum(expr, *indicators) > 0,
                    dtype=torch.float64,
                )
            except Exception as error:  # every input is reported, not the first
                failures.append(f"{kind}: {type(error).__name__}: {str(error)[:300]}")
        rows.append(
            {
                "operands": list(operand_fmts),
                "failures": failures,
                "reached": bool(reached),
                "relabelled": bool(relabelled),
                "relaid": bool(moved),
            }
        )
    print("ROWS " + json.dumps(rows))


def test_the_premise_tables_keep_a_program_that_reaches_the_case():
    """Some program refused before still reaches the case in every family, and
    every program listed as now looped in order was refused before."""

    for expr, relaid in NOW_LOOPED_IN_ORDER.items():
        assert relaid <= REFUSED_BEFORE[expr], (expr, relaid - REFUSED_BEFORE[expr])
    for expr, refused in REFUSED_BEFORE.items():
        assert len(refused) > len(NOW_LOOPED_IN_ORDER.get(expr, set())), expr
        assert refused <= set(FAMILIES[expr]), (expr, refused - set(FAMILIES[expr]))


@pytest.mark.parametrize(
    "expr,result_fmt,program",
    [
        pytest.param(*case, id=f"{case[0]} as {case[1]} with {','.join(case[2])}")
        for case in _cases()
    ],
)
def test_dense_operands_out_of_storage_order_match_pytorch(expr, result_fmt, program):
    import scorch

    root = Path(__file__).resolve().parents[2]
    source = Path(scorch.__file__).resolve().parents[1]
    environment = dict(
        os.environ,
        PYTHONHASHSEED="0",
        PYTHONPATH=os.pathsep.join([str(source), str(root)]),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.test_scorch.test_dense_operand_out_of_storage_order",
            json.dumps([expr, result_fmt, [list(program)]]),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("ROWS ")]
    assert completed.returncode == 0 and lines, completed.stderr[-3000:]
    rows = json.loads(lines[-1][len("ROWS ") :])
    assert [tuple(row["operands"]) for row in rows] == [program]
    (row,) = rows
    assert not row["failures"], "; ".join(row["failures"])
    # The programs refused before must reach the case into ``sd``, where they
    # were found, except those the relayout now lays out in the loop's order; a
    # schedule or relayout change that moves any of them fails here.
    if result_fmt == "sd" and program in REFUSED_BEFORE[expr]:
        if program in NOW_LOOPED_IN_ORDER.get(expr, set()):
            assert not row["reached"], (
                "loops a dense operand out of order again; remove it from "
                "NOW_LOOPED_IN_ORDER"
            )
            assert row["relaid"], "listed as relaid but not relaid"
        else:
            assert row["reached"], "no longer loops a dense operand out of order"


if __name__ == "__main__":
    expr_arg, result_arg, programs_arg = json.loads(sys.argv[1])
    _child(expr_arg, result_arg, [tuple(p) for p in programs_arg])
