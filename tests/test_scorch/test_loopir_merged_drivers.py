"""Merged loops wherever a sparse result's assembly or reduction puts one.

The new compiler's sparse-result targets accept a merged loop (a co-iteration
of several operands' stored coordinates) in six places beyond the innermost
level of an elementwise assembly.  Each cell below names one, and runs it into
several result layouts, because which layout einsum infers is not fixed:

* an intersection driving a dense level of the result's prefix, above the
  first compressed level, with no reduction (an elementwise product);
* an intersection driving a level above a sorted sparse workspace (a batched
  matrix product, whose batch coordinates both operands store);
* an intersection binding a result level above a reduction whose result
  coordinates are all bound before it (``bijk,bjk->bi`` over ``b``);
* a merged reduction loop below every result level (``bijk,bjk->bi`` over
  ``j`` or ``k``);
* an intersection of three or more operands (MTTKRP, whose factor matrices
  all store ``r``);
* an intersection binding the column of a result whose row loop iterates a
  stored stream, above the reduced loop (``ijk,jk->ij`` into CSR): each
  stored ``(i, j)`` is one scalar accumulation, not a workspace key.
* an innermost intersection of three operands in an elementwise product
  (``ij,ij,ij->ij``), which the generic target once refused for having more
  than two cursors.

A sixth cell is a nest that opens like one of the two-operand workspace
families (one outer stored row loop, a region keyed by one coordinate) but
continues past it; the router hands it to the general region target.

Two more cells are not about merges but sit on the same routes:

* a merged loop in a dense result's nest, at any depth (``bijk,bjk->bi`` and
  MTTKRP into all-dense results);
* a copy or product into canonical CSR (``ds``) whose drivers are anything
  but a dense row over a merged column: a stored or dense row, a stored or
  dense column.  These run on rank-2 extents that include zero and one.

Every result is judged from its own arrays against float64 ``torch.einsum``:
its values, and stored coordinates equal to what the operands' stored entries
imply.  The inputs are the high-rank test's four, on a regular and on a
zero extent, under both automatic schedule arms.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import replace
from typing import List, Tuple

import pytest

from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from scorch.compiler.scheduler import Schedule
from tests.test_scorch.test_high_rank_programs import (
    Program,
    _cin,
    _inputs,
    _judge,
    _options,
)

VARIANTS = ("full", "sparse", "gap", "hollow")

# (cell name, expression, operand format sets, result formats)
CELLS: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    (
        "intersection-dense-prefix",
        "ijk,ijk->ijk",
        ("sdd,ssd", "sss,sds", "dss,sss"),
        ("dss", "dds", "dsd"),
    ),
    (
        "intersection-dense-prefix",
        "ijkl,ijkl->ijkl",
        ("sddd,ssdd", "sdss,sdsd"),
        ("dsss", "ddss", "ddds", "dsds"),
    ),
    (
        "intersection-above-workspace",
        "abij,abjk->abik",
        ("sdss,ssds", "ssdd,sssd", "ssss,ssss"),
        ("dssd", "ssds", "dsss", "ssss", "sdsd"),
    ),
    (
        "intersection-bound-level",
        "bijk,bjk->bi",
        ("ssds,ssd", "ssss,sss"),
        ("ds", "ss", "sd"),
    ),
    (
        "merged-reduction",
        "bijk,bjk->bi",
        ("ddsd,sss", "ddss,dds", "ddds,dss"),
        ("ss", "sd", "ds"),
    ),
    (
        "narrow-family-fallback",
        "ijkl,ijkl->il",
        ("sssd,dsdd", "ssss,ssss"),
        ("ss", "ds", "sd"),
    ),
    (
        "narrow-family-fallback",
        "ijkl,jklm->im",
        ("ssds,ssdd",),
        ("ds", "ss", "sd"),
    ),
    (
        "intersection-of-three",
        "ijkl,jr,kr,lr->ir",
        ("ssss,ss,ss,ss", "sssd,ss,ds,ss"),
        ("ds", "ss", "sd"),
    ),
    (
        "intersection-of-three",
        "ijkl,ir,kr,lr->jr",
        ("dsdd,ss,ss,ds", "dsss,ss,ss,ss"),
        ("ss", "ds", "sd"),
    ),
    (
        "merged-column-above-reduction",
        "ijk,jk->ij",
        ("ssd,sd", "sss,ss"),
        ("ds", "ss", "sd"),
    ),
    (
        "merged-column-above-reduction",
        "ijk,ijk->ij",
        ("dsd,ssd", "sss,dss"),
        ("ds", "ss", "sd"),
    ),
    (
        "innermost-intersection-of-three",
        "ij,ij,ij->ij",
        ("ds,ds,ds", "ss,ds,ds", "ss,ss,ss"),
        ("ds", "ss"),
    ),
    (
        "innermost-intersection-of-three",
        "ijk,ijk,ijk->ijk",
        ("dss,dss,dss", "sss,dss,sss"),
        ("dss", "sss"),
    ),
    (
        "dense-result-merges",
        "bijk,bjk->bi",
        ("dsss,dss", "ssss,sss"),
        ("dd",),
    ),
    (
        "dense-result-merges",
        "ijkl,jr,kr,lr->ir",
        ("dsdd,sd,ss,sd", "ssss,ss,ss,ss"),
        ("dd",),
    ),
    (
        "dense-result-merges",
        "ijkl,ijkl->il",
        ("ssss,ssdd",),
        ("dd",),
    ),
    ("csr-assembly", "ij->ij", ("dd", "ds", "sd", "ss"), ("ds",)),
    ("csr-assembly", "ij,ij->ij", ("ds,dd", "sd,ss", "dd,sd"), ("ds",)),
)

# Rank-2 extents for the CSR cells: regular, a zero on each mode, all unit.
_RANK2_SIZES = ((3, 4), (0, 3), (2, 0), (1, 1))


def _programs() -> List[Program]:
    programs = []
    for family, expr, operand_sets, results in CELLS:
        letters = "".join(dict.fromkeys(expr.split("->")[0].replace(",", "")))
        for operands, result in itertools.product(operand_sets, results):
            for sizes in (
                _RANK2_SIZES
                if family == "csr-assembly"
                else ((3, 4, 2, 3, 2, 3), (2, 3, 0, 3, 2, 2))
            ):
                programs.append(
                    Program(
                        family=family,
                        expr=expr,
                        operand_formats=tuple(operands.split(",")),
                        result_format=result,
                        extents=tuple(zip(letters, sizes[: len(letters)])),
                    )
                )
    return programs


PROGRAMS = _programs()


def _run(program: Program, operands, blocking: bool):
    options = replace(_options(blocking), requested_schedule=Schedule())
    return execute_cin_via_loopir(
        _cin(program),
        program.shape(program.output),
        *operands,
        compile_options=options,
    )


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("program", PROGRAMS, ids=[p.name for p in PROGRAMS])
def test_merged_drivers_match_pytorch(program, blocking):
    for seed, variant in enumerate(VARIANTS):
        operands, reference, structure = _inputs(program, variant, seed)
        result, _ = _run(program, operands, blocking)
        try:
            _judge(program, result, reference, structure)
        except AssertionError as error:
            raise AssertionError(f"{variant} input: {error}") from error


def _merges(program_text: str) -> List[Tuple[int, int]]:
    """``(depth, cursor count)`` of every merged loop, outermost first."""

    found = []
    for line in program_text.splitlines():
        match = re.match(
            r"(\s*)merged_(?:intersection|union)_for .* in \((.*)\) binds", line
        )
        if match:
            found.append((len(match.group(1)), match.group(2).count(" over ")))
    return found


def _first_program(family: str, result: str) -> Program:
    return next(
        p
        for p in PROGRAMS
        if p.family == family
        and p.result_format == result
        and 0 not in dict(p.extents).values()
    )


def test_the_cells_exercise_their_merges():
    """Each cell's LoopIR carries the merge it names, where it names it.

    Without this the table could pass by every program taking some other
    route: a cell whose merge moved, or vanished, would still compute the
    right answer.
    """

    def text(program: Program) -> str:
        operands, _, _ = _inputs(program, "full", 0)
        _, kernel = _run(program, operands, False)
        return kernel.program_text

    def loop_depths(program_text: str) -> List[int]:
        return [
            len(line) - len(line.lstrip())
            for line in program_text.splitlines()
            if re.match(r"\s*(for|sparse_for|merged_\w+_for) ", line)
        ]

    # A merged loop is the program's outermost loop (it binds the first
    # result level) in the prefix and bound-level cells.
    for family, result in (
        ("intersection-dense-prefix", "dss"),
        ("intersection-above-workspace", "dssd"),
        ("intersection-bound-level", "ds"),
    ):
        program_text = text(_first_program(family, result))
        merges = _merges(program_text)
        assert merges and merges[0][0] == min(loop_depths(program_text)), (
            family,
            merges,
        )
    # Three or more cursors in one merge.
    for result in ("ds", "ss"):
        merges = _merges(text(_first_program("intersection-of-three", result)))
        assert any(cursors >= 3 for _, cursors in merges), merges
    # The innermost merge of the three-operand product holds all three; it
    # binds no positions (the product reads through its cursors).
    program_text = text(_first_program("innermost-intersection-of-three", "ds"))
    assert [
        line.count(" over ")
        for line in program_text.splitlines()
        if "merged_intersection_for" in line
    ] == [3], program_text
    # The merged-reduction cell's merge lies below both result loops.
    program_text = text(_first_program("merged-reduction", "ss"))
    merges = _merges(program_text)
    depths = sorted(loop_depths(program_text))
    assert merges and merges[-1][0] > depths[1], (merges, depths)
