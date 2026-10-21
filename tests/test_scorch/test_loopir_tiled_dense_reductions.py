"""Tiled dense-result reductions through the new compiler's stack families.

For a dense-result ADD reduction the automatic schedule records a dense
workspace over the trailing free loop and affine tiles placed per arm: under
the row loop on the regblock arm, outermost on the other.  The plans below
reach the target in four shapes:

* tile-free: no tile, and no workspace materialized;
* the stack form: one tile, on the workspace's axis;
* the row-tiled stack form: tiles on distinct loops above the reduction,
  each a plain affine split, with or without a tile on the axis (without
  one, the producer accumulates directly into the result);
* the reduce-out form: tiles on the axis and on the last reduction loop;
* the reduction-tiled form: a tile on the last reduction loop and none on the
  axis, which is then a sparse loop no affine tile can split (dense times CSR
  into a dense result).  Tiles of loops above the reduction may come with it.
  Every tile is a plain affine split and the producer accumulates directly
  into the result.

Each program is one the new compiler used to refuse.  Every result is
judged from its own arrays against float64 ``torch.einsum`` on the
high-rank test's four inputs, under both arms, on the population's extents,
on extents with a zero, and on extents wider than a tile.  Where the new
compiler's C++ equals the shipped compiler's for the same scheduled program
the test pins the equality; where it differs, the program is listed with
the reason.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import pytest
import torch

from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.compiler.scheduler import Schedule
from tests.test_scorch.test_high_rank_programs import (
    Program,
    _cin,
    _inputs,
    _judge,
    _options,
)

VARIANTS = ("full", "sparse", "gap", "hollow")

# (family, expression, operand formats, result format, population extents,
#  extents with a zero, extents wider than one tile)
_TABLE = (
    (
        "reduce",
        "ijkl->ijl",
        ("ddsd",),
        "ddd",
        (4, 2, 2, 4),
        (4, 0, 2, 4),
        (3, 37, 3, 40),
    ),
    ("reduce", "ijkl->jl", ("ddds",), "dd", (2, 2, 4, 2), (2, 2, 0, 2), (3, 40, 5, 37)),
    (
        "reduce",
        "ijklm->ijkm",
        ("dddsd",),
        "dddd",
        (4, 2, 4, 4, 4),
        (4, 2, 4, 0, 4),
        (2, 3, 33, 3, 35),
    ),
    (
        "ttm",
        "ijkl,la->ijka",
        ("ddds", "sd"),
        "dddd",
        (3, 4, 2, 3, 2),
        (3, 4, 2, 3, 0),
        (2, 3, 34, 3, 36),
    ),
    (
        "ttm",
        "ijklm,ma->ijkla",
        ("sssds", "sd"),
        "ddddd",
        (3, 4, 3, 4, 3, 3),
        (3, 0, 3, 4, 3, 3),
        (2, 2, 2, 33, 3, 35),
    ),
    (
        "batched",
        "abij,abjk->abik",
        ("dsds", "ddds"),
        "dddd",
        (4, 2, 3, 2, 2),
        (4, 2, 3, 0, 2),
        (2, 3, 34, 3, 37),
    ),
    (
        "mttkrp",
        "ijklm,jr,kr,lr,mr->ir",
        ("ddsdd", "dd", "sd", "dd", "dd"),
        "dd",
        (4, 2, 2, 2, 3, 2),
        (4, 2, 2, 2, 0, 2),
        (5, 3, 4, 3, 40, 37),
    ),
    ("reduce", "ijk->k", ("ddd",), "d", (3, 4, 5), (3, 0, 5), (3, 5, 40)),
    ("reduce", "ijk->ik", ("ddd",), "dd", (3, 4, 5), (0, 4, 5), (35, 5, 40)),
    ("reduce", "ijk->ki", ("ddd",), "dd", (3, 4, 5), (3, 4, 0), (35, 5, 40)),
    ("reduce", "ijk->kj", ("ddd",), "dd", (3, 4, 5), (3, 0, 5), (3, 35, 40)),
    # The reduction-tiled form.  The wide extents put more than one tile on
    # every tiled loop, and more rows than one thread's chunk on the outer one.
    ("matmul", "ij,jk->ik", ("dd", "ds"), "dd", (3, 4, 5), (3, 0, 5), (35, 40, 37)),
    (
        "reduce",
        "ij,ik,jk->k",
        ("dd", "dd", "ds"),
        "d",
        (3, 4, 5),
        (0, 4, 5),
        (35, 40, 37),
    ),
    (
        "chain",
        "ij,jk,kl->il",
        ("dd", "dd", "ds"),
        "dd",
        (3, 4, 5, 3),
        (3, 4, 0, 3),
        (35, 40, 37, 34),
    ),
    (
        "ttm",
        "ijk,kl->ijl",
        ("ddd", "ds"),
        "ddd",
        (3, 4, 5, 3),
        (3, 0, 5, 3),
        (3, 35, 40, 37),
    ),
    (
        "ttm",
        "ijk,kl->ijl",
        ("dsd", "ds"),
        "ddd",
        (3, 4, 5, 3),
        (3, 4, 0, 3),
        (3, 35, 40, 37),
    ),
)


def _program(family, expr, operands, result, sizes) -> Program:
    letters = "".join(dict.fromkeys(expr.split("->")[0].replace(",", "")))
    return Program(
        family=family,
        expr=expr,
        operand_formats=operands,
        result_format=result,
        extents=tuple(zip(letters, sizes)),
    )


REGULAR = [_program(*row[:4], row[4]) for row in _TABLE]
PROGRAMS = [
    _program(*row[:4], sizes) for row in _TABLE for sizes in (row[4], row[5], row[6])
]

# The plan shape each regular program takes on each arm (plain, blocked).
FAMILIES: Dict[str, Tuple[str, str]] = {
    REGULAR[0].name: ("row-tiled", "row-tiled"),
    REGULAR[1].name: ("row-tiled", "row-tiled"),
    REGULAR[2].name: ("row-tiled", "row-tiled"),
    REGULAR[3].name: ("row-tiled", "row-tiled"),
    REGULAR[4].name: ("tile-free", "row-tiled"),
    REGULAR[5].name: ("tile-free", "row-tiled"),
    REGULAR[6].name: ("reduce-out", "reduce-out"),
    REGULAR[7].name: ("stack", "stack"),
    REGULAR[8].name: ("stack", "stack"),
    REGULAR[9].name: ("stack", "stack"),
    REGULAR[10].name: ("tile-free", "tile-free"),
    REGULAR[11].name: ("reduction-tiled", "reduction-tiled"),
    REGULAR[12].name: ("reduction-tiled", "reduction-tiled"),
    REGULAR[13].name: ("reduction-tiled", "reduction-tiled"),
    REGULAR[14].name: ("reduction-tiled", "reduction-tiled"),
    REGULAR[15].name: ("tile-free", "reduction-tiled"),
}

# Regular programs whose C++ differs from the shipped compiler's for the same
# scheduled program, per arm, with the reason.  The others are byte-identical
# wherever the shipped compiler produces a kernel for that program.
DIFFERENT_FROM_SHIPPED: Dict[Tuple[str, bool], str] = {
    (REGULAR[1].name, False): "no axis tile: the shipped compiler keeps a "
    "full-row workspace, the new compiler accumulates into the result",
    (REGULAR[1].name, True): "no axis tile: the shipped compiler keeps a "
    "full-row workspace, the new compiler accumulates into the result",
    (REGULAR[5].name, True): "no axis tile: the shipped compiler keeps a "
    "full-row workspace, the new compiler accumulates into the result",
    (REGULAR[6].name, False): "reduce-out over a merged reduction: the shipped "
    "kernel does not apply the recorded tiles",
    (REGULAR[6].name, True): "reduce-out over a merged reduction: the shipped "
    "kernel does not apply the recorded tiles",
}

# Every reduction-tiled plan differs the same way: no axis tile.
DIFFERENT_FROM_SHIPPED.update(
    {
        (name, blocking): "no axis tile: the shipped compiler keeps a "
        "full-row workspace, the new compiler accumulates into the result"
        for name, families in FAMILIES.items()
        for blocking in (False, True)
        if families[blocking] == "reduction-tiled"
    }
)

# Regular programs for which the shipped compiler's scheduled route produces
# no kernel to compare against: it rejects a result stored in an order other
# than the loop order.
NO_SHIPPED_COMPARAND = {REGULAR[9].name, REGULAR[10].name}


def _compile_options(blocking: bool):
    return replace(_options(blocking), requested_schedule=Schedule())


def _bindings(program: Program):
    return tuple((program.shape(tensor), torch.float32) for tensor in program.inputs)


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("program", PROGRAMS, ids=[p.name for p in PROGRAMS])
def test_tiled_reductions_match_pytorch(program, blocking):
    for seed, variant in enumerate(VARIANTS):
        operands, reference, structure = _inputs(program, variant, seed)
        result, _ = execute_cin_via_loopir(
            _cin(program),
            program.shape(program.output),
            *operands,
            compile_options=_compile_options(blocking),
        )
        try:
            _judge(program, result, reference, structure)
        except AssertionError as error:
            raise AssertionError(f"{variant} input: {error}") from error


def _family(plan) -> str:
    if not plan.tiles:
        return "tile-free"
    workspace = plan.workspace
    assert workspace is not None and workspace.dense, plan
    tiled = [tile.loop for tile in plan.tiles]
    if workspace.reduction_loop in tiled:
        if workspace.axis_loops[0] in tiled:
            return "reduce-out"
        return "reduction-tiled"
    if tiled == [workspace.axis_loops[0]]:
        return "stack"
    return "row-tiled"


# The wide extents again with 35 added to each one that exceeds a tile: every
# tiled loop then runs two full 32-wide tiles and a tail (the wide extents run
# one full tile and a tail).
PAST_TWO_TILES = [
    (
        _program(*row[:4], row[6]),
        _program(*row[:4], tuple(e + 35 if e > 32 else e for e in row[6])),
    )
    for row in _TABLE
]


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize(
    "wide,program", PAST_TWO_TILES, ids=[p.name for _, p in PAST_TWO_TILES]
)
def test_tiled_reductions_match_pytorch_past_two_tiles(wide, program, blocking):
    """Each program, in the plan shape it takes at the wide extents, with every
    tiled extent past twice the tile width."""

    shapes = []
    for sized in (wide, program):
        kernel = compile_cin_via_loopir(
            _cin(sized),
            sized.shape(sized.output),
            _bindings(sized),
            compile_options=_compile_options(blocking),
        )
        assert kernel.schedule is not None
        shapes.append(_family(kernel.schedule.plan))
    assert shapes[0] == shapes[1]
    for seed, variant in enumerate(VARIANTS):
        operands, reference, structure = _inputs(program, variant, seed)
        result, _ = execute_cin_via_loopir(
            _cin(program),
            program.shape(program.output),
            *operands,
            compile_options=_compile_options(blocking),
        )
        try:
            _judge(program, result, reference, structure)
        except AssertionError as error:
            raise AssertionError(f"{variant} input: {error}") from error


def test_the_two_tile_extents_run_the_stack_and_reduction_tiled_forms():
    """Both single-tile forms are among the plans the two-tile extents run, on
    both settings."""

    for blocking in (False, True):
        found = set()
        for _, program in PAST_TWO_TILES:
            kernel = compile_cin_via_loopir(
                _cin(program),
                program.shape(program.output),
                _bindings(program),
                compile_options=_compile_options(blocking),
            )
            assert kernel.schedule is not None
            found.add(_family(kernel.schedule.plan))
        assert {"stack", "reduction-tiled"} <= found, (blocking, found)


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
def test_the_programs_take_their_plan_shapes(blocking):
    """Each program reaches the target through the plan shape it names.

    Without this the table could pass with every program taking another
    route: a plan whose tiles moved or vanished still computes the right
    answer.
    """

    found: List[Tuple[str, str]] = []
    for program in REGULAR:
        kernel = compile_cin_via_loopir(
            _cin(program),
            program.shape(program.output),
            _bindings(program),
            compile_options=_compile_options(blocking),
        )
        assert kernel.schedule is not None
        found.append((program.name, _family(kernel.schedule.plan)))
    expected = [(name, FAMILIES[name][blocking]) for name, _ in found]
    assert found == expected


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize(
    "program",
    [p for p in REGULAR if p.name not in NO_SHIPPED_COMPARAND],
    ids=[p.name for p in REGULAR if p.name not in NO_SHIPPED_COMPARAND],
)
def test_the_kernel_is_the_shipped_kernel_unless_listed(program, blocking):
    options = _compile_options(blocking)
    new = compile_cin_via_loopir(
        _cin(program),
        program.shape(program.output),
        _bindings(program),
        compile_options=options,
    ).cpp_source
    shipped = legacy_generated_cpp(
        _cin(program),
        program.shape(program.output),
        _bindings(program),
        compile_options=options,
    )
    reason = DIFFERENT_FROM_SHIPPED.get((program.name, blocking))
    if reason is None:
        assert new == shipped
    else:
        # Listed as different: if the kernels now agree, the entry is stale.
        assert new != shipped, f"now byte-identical; remove the entry ({reason})"
