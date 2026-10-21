"""The LoopIR route's outcome does not depend on the order a program's loops are written in.

The public einsum writes its loop nest in a topological order of every tensor's
storage order, computed by iterating a set (``topo_sort_characters`` in
``scorch/utils.py``), so the written position of indices that no storage order
relates follows ``PYTHONHASHSEED``.  The scheduler then selects a plan order of
its own.  ``lower_normalized_cin_to_loopir`` used to build the base program in
the written order and verify that order against the sparse family's rules
(result storage order, parent-before-child of every compressed level) before the
reorder pass applied the plan, so a program written in a nest that visits a
compressed or result level out of storage order was refused even when the plan
order verifies.  The base program is now nested in the plan order, and the
written order only fixes the order dimensions are declared in.

The plan itself could also follow the spelling: ``Scheduler.select_loop_order``
starts from the written nest and keeps its order wherever its cost model ties.
On the LoopIR route an automatic schedule now starts from the order the public
einsum would write (``pipeline._nest_in_storage_order``).

Two consequences are locked here, on both automatic arms: a program built
directly in every order its loops can be written in gets one kernel per arm (or
one refusal code), and the programs the public einsum builds are accepted or
refused the same way under several hash seeds.
"""

import itertools
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace

import pytest
import torch

from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir
from scorch.compiler.scheduler import Schedule

EXTENT = {"i": 5, "j": 4, "k": 3, "l": 2}

# (name, operands as (symbol, format, indices), result format, result indices).
NEST_PROGRAMS = [
    ("TTM sss x dd -> dss", (("A", "sss", "ijk"), ("B", "dd", "kl")), "dss", "ijl"),
    ("TTM sss x ss -> sss", (("A", "sss", "ijk"), ("B", "ss", "kl")), "sss", "ijl"),
    (
        "MTTKRP sss x dd x dd -> ds",
        (("A", "sss", "ijk"), ("B", "dd", "jl"), ("D", "dd", "kl")),
        "ds",
        "il",
    ),
    (
        "SDDMM ss x dd x dd -> ss",
        (("A", "ss", "ij"), ("B", "dd", "ik"), ("D", "dd", "jk")),
        "ss",
        "ij",
    ),
    ("MM ss x ds -> ss", (("A", "ss", "ik"), ("B", "ds", "kj")), "ss", "ij"),
    ("MM ds x dd -> dd", (("A", "ds", "ik"), ("B", "dd", "kj")), "dd", "ij"),
    # The scheduler's cost model ties i and k here, and its stable sort used to
    # keep whichever was written first: plan ijkl or kijl by spelling.
    ("ij,kl->jl ds x ds -> dd", (("A", "ds", "ij"), ("B", "ds", "kl")), "dd", "jl"),
]


def program_id(cell):
    return cell[0]


def auto_options(regblock_enabled):
    return replace(
        CompileOptions.from_environment(environ={}).with_regblock_enabled(
            regblock_enabled
        ),
        requested_schedule=Schedule(),
    )


def build_cin(cell, nest):
    _, operands, result_fmt, result_indices = cell
    ivars = {name: IndexVar(name) for name in nest}
    rhs = None
    for symbol, fmt, indices in operands:
        access = TensorVar(symbol, fmt=fmt, dtype=torch.float64)[
            tuple(ivars[name] for name in indices)
        ]
        rhs = access if rhs is None else CINBinaryOp(Operation.MUL, rhs, access)
    result = TensorVar("C", fmt=result_fmt, dtype=torch.float64)[
        tuple(ivars[name] for name in result_indices)
    ]
    stmt = TensorAssign(result, rhs)
    for name in reversed(nest):
        stmt = ForAll(ivars[name], stmt)
    return stmt


def outcome(compile_once):
    """``emitted`` and the C++ source, or the structured refusal's code."""

    try:
        kernel = compile_once()
    except Exception as error:  # noqa: BLE001 -- every refusal must carry a code
        defect = getattr(error, "defect", None)
        code = getattr(defect, "code", None)
        assert code, f"unstructured refusal {type(error).__name__}: {error}"
        return f"refused:{code}", None
    return "emitted", kernel.cpp_source


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["arm0", "arm1"])
@pytest.mark.parametrize("cell", NEST_PROGRAMS, ids=program_id)
def test_every_written_nest_has_one_outcome(cell, regblock_enabled):
    _, operands, _, result_indices = cell
    letters = sorted({name for _, _, indices in operands for name in indices})
    shape = tuple(EXTENT[name] for name in result_indices)
    bindings = tuple(
        (tuple(EXTENT[name] for name in indices), torch.float64)
        for _, _, indices in operands
    )
    outcomes = {}
    for nest in itertools.permutations(letters):
        outcomes["".join(nest)] = outcome(
            lambda: compile_cin_via_loopir(
                build_cin(cell, nest),
                shape,
                bindings,
                compile_options=auto_options(regblock_enabled),
            )
        )
    assert len({kind for kind, _ in outcomes.values()}) == 1, {
        nest: kind for nest, (kind, _) in outcomes.items()
    }
    # Not only the same outcome: the same kernel, whatever the spelling.
    assert len({source for _, source in outcomes.values()}) == 1


# -- the programs the public einsum builds, under several hash seeds ----------

# (expression, operand formats), each run into every dense/compressed result
# format of its rank, named explicitly: the front end's format inference may
# change.  In the first seven, two indices start no tensor's storage order after
# another's, so the order the public einsum writes them in follows the hash
# seed, while every tensor's layout does not.  The rest are written in one nest
# under every seed.  Programs such as ``ijk->ikj``, whose result layout itself
# follows the hash seed, are a different program under each seed and are not
# listed.
FRONT_DOOR_EXPRESSIONS = [
    ("ij,k->ij", ("ss", "s")),
    ("ij,k->ij", ("ss", "d")),
    ("ij,kl->jl", ("ss", "ss")),
    ("ij,kl->jl", ("ds", "ds")),
    ("ij,kl->ijl", ("ss", "ds")),
    ("ijk,l->ijk", ("sss", "d")),
    ("ijk,l->ij", ("sss", "d")),
    ("ijk,kl->ijl", ("sss", "dd")),
    ("ijk,jl,kl->il", ("sss", "dd", "dd")),
    ("ij,ik,jk->ij", ("ss", "dd", "dd")),
]
FRONT_DOOR_PROGRAMS = [
    (expression, formats, "".join(levels))
    for expression, formats in FRONT_DOOR_EXPRESSIONS
    for levels in itertools.product("ds", repeat=len(expression.split("->")[1]))
]
SEEDS = (0, 1, 2, 3)

# Runs in a child interpreter, because the hash seed is fixed at startup.  The
# public einsum is called on data-free tensor specs; the first loop-order
# selection receives the program the front door built, which is copied and
# handed to the LoopIR route, and the call is stopped there, so nothing is built.
CHILD = textwrap.dedent("""
    import copy, json, sys
    import torch
    import scorch
    from dataclasses import replace
    from scorch.compiler.cin import ForAll
    from scorch.compiler.compile_options import CompileOptions
    from scorch.compiler.loopir.pipeline import compile_cin_via_loopir
    from scorch.compiler.scheduler import Schedule, Scheduler
    from scorch.layout import TensorSpec

    EXTENT = {"i": 5, "j": 4, "k": 3, "l": 2}

    class Stop(BaseException):
        pass

    captured = []
    real_select = Scheduler.__dict__["select_loop_order"]

    def select(cin, *args, **kwargs):
        captured.append(copy.deepcopy(cin))
        raise Stop

    rows = {}
    for expression, formats, result_format in json.loads(sys.argv[1]):
        terms = expression.split("->")[0].split(",")
        specs = [
            TensorSpec(fmt, tuple(EXTENT[c] for c in term), dtype=torch.float64)
            for term, fmt in zip(terms, formats)
        ]
        captured.clear()
        Scheduler.select_loop_order = staticmethod(select)
        try:
            scorch.einsum(expression, *specs, format=result_format, compile_only=True)
        except Stop:
            pass
        finally:
            Scheduler.select_loop_order = real_select
        key = f"{expression} {','.join(formats)} -> {result_format}"
        if not captured:
            rows[key] = {"captured": False}
            continue
        cin = captured[0]
        nest, stmt = [], cin
        while isinstance(stmt, ForAll):
            nest.append(stmt.index_var.name)
            stmt = stmt.stmt
        row = {
            "captured": True,
            "nest": "".join(nest),
            "layouts": [
                [str(tv.format), list(tv.mode_order or ())]
                for tv in [*cin.get_rhs_tensor_vars(), stmt.lhs.tensor]
            ],
        }
        bindings = tuple(
            (tuple(tv.shape), tv.dtype) for tv in cin.get_rhs_tensor_vars()
        )
        for arm in (0, 1):
            options = replace(
                CompileOptions.from_environment(environ={}).with_regblock_enabled(
                    bool(arm)
                ),
                requested_schedule=Schedule(),
            )
            try:
                compile_cin_via_loopir(
                    copy.deepcopy(cin),
                    tuple(stmt.lhs.tensor.shape),
                    bindings,
                    compile_options=options,
                )
                row[f"arm{arm}"] = "emitted"
            except Exception as error:
                code = getattr(getattr(error, "defect", None), "code", None)
                row[f"arm{arm}"] = (
                    f"refused:{code}"
                    if code
                    else f"unstructured:{type(error).__name__}: {error}"
                )
        rows[key] = row
    print(json.dumps(rows))
    """)


def front_door_outcomes(programs):
    """Each seed's rows for ``programs``, the children run side by side."""

    children = {
        seed: subprocess.Popen(
            [sys.executable, "-c", CHILD, json.dumps(programs)],
            env=dict(os.environ, PYTHONHASHSEED=str(seed)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for seed in SEEDS
    }
    rows = {}
    for seed, child in children.items():
        stdout, stderr = child.communicate(timeout=600)
        assert child.returncode == 0, stderr[-4000:]
        rows[seed] = json.loads(stdout.strip().splitlines()[-1])
    return rows


def test_public_einsum_programs_are_accepted_the_same_under_every_hash_seed():
    rows = front_door_outcomes(FRONT_DOOR_PROGRAMS)

    for seed in SEEDS:
        missing = [key for key, row in rows[seed].items() if not row["captured"]]
        assert not missing, f"seed {seed}: no program reached scheduling: {missing}"
        unstructured = {
            (key, arm): row[arm]
            for key, row in rows[seed].items()
            for arm in ("arm0", "arm1")
            if row[arm].startswith("unstructured:")
        }
        assert not unstructured, f"seed {seed}: {unstructured}"

    # Each listed program must be the same program under every seed, and some must
    # be written in a different nest under some seed, or the comparison below
    # could not fail.
    layouts = {
        key: {json.dumps(rows[seed][key]["layouts"]) for seed in SEEDS}
        for key in rows[0]
    }
    assert all(len(seen) == 1 for seen in layouts.values()), layouts
    nests = {key: {rows[seed][key]["nest"] for seed in SEEDS} for key in rows[0]}
    assert sum(len(written) > 1 for written in nests.values()) >= 20, nests

    for arm in ("arm0", "arm1"):
        accepted = {
            seed: sorted(
                key for key, row in rows[seed].items() if row[arm] == "emitted"
            )
            for seed in SEEDS
        }
        assert all(accepted[seed] == accepted[SEEDS[0]] for seed in SEEDS), (
            arm,
            accepted,
            nests,
        )


def test_the_plan_does_not_follow_the_written_nest():
    """``ij,kl->jl`` is written ``ikjl`` or ``kijl`` by seed under the old front door.

    The scheduler's cost model ties i and k, and its stable sort kept whichever
    the nest wrote first; with register blocking on, the LoopIR schedule passes
    accept the family plan ``kijl`` records and refuse the one plan ``ijkl``
    records, so acceptance followed the spelling.  The LoopIR route now
    re-nests the program in storage order before the scheduler sees it.
    """

    rows = front_door_outcomes([("ij,kl->jl", ("ds", "ds"), "dd")])
    outcomes = [row for by_key in rows.values() for row in by_key.values()]
    if len({row["nest"] for row in outcomes}) < 2:
        pytest.fail(f"every seed wrote one nest, so nothing is compared: {outcomes}")
    assert len({(row["arm0"], row["arm1"]) for row in outcomes}) == 1, outcomes
