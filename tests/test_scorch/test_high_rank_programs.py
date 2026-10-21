"""Generated programs on four to six indices, run on both compilers against PyTorch.

Each program is an einsum over four to six indices -- a copy or permutation, a
reduction, a contraction with one or more matrices (one mode, several modes at
once, MTTKRP), a tensor-tensor contraction, a batched product, an outer product
or an elementwise product -- with every operand and the result in a random mix
of dense and compressed levels.  Each program runs through the shipped compiler
(``scorch.einsum``) and the new one (``execute_cin_via_loopir``), on both
automatic schedules, in float64, on four inputs:

* ``full``: every entry stored, values in [0.5, 1.5), so every cell of a
  nonempty result has a nonzero reference;
* ``sparse``: each entry stored with probability 0.35;
* ``gap``: as ``sparse``, then the first and the last slice along each
  operand's first index store nothing.
* ``hollow``: as ``gap``, and every compressed level that has a compressed
  level below it also stores coordinates with nothing under them (their child
  segments are empty), the whole first slice among them.  Such a coordinate
  stores no entry, so it implies no result coordinate.

A result passes when

* its arrays pass the library's own storage validation (``STensor.copy``);
* its values match ``torch.einsum`` on dense float64 copies to 1e-12 of the
  reference's scale;
* it stores exactly the coordinates the operands' stored entries imply: a
  compressed coordinate is stored exactly when the operands store something
  under it (structure, never value), and a dense level stores every cell.  The
  expected level arrays are built from the operands' storage indicators in the
  result's own storage order and compared array by array.

A compiler may refuse a program, but only with a named reason, and only if
``high_rank_expected.json`` lists that program under that compiler with that
reason.  The file is the record of what each compiler does not lower: a listed
program that starts compiling fails until its line is removed, and an unlisted
program that is refused fails, so a refusal can neither appear nor disappear
unnoticed.  Any other exception fails the test, and so does a wrong result:
``KNOWN_DEFECTS`` names the programs a compiler currently gets wrong, each with
its cause, as strict expected failures.

``SCORCH_HIGH_RANK_RECORD=<directory>`` writes one JSON line per run with its
outcome instead of judging it against the file; that is how the file is
regenerated after the generator or a compiler changes.

The result format is always passed to ``scorch.einsum`` explicitly, never
inferred, and the population has two parts.  The first draws operand formats,
the result layout and the extents at random, several times per expression, with
at least one compressed result level.  The second is a layout sweep: each
expression keeps one set of operand formats and extents, and its result takes
every dense/compressed layout of its rank in turn, the all-dense one included,
so along one expression's row the result layout is the only thing that varies.

The population is generated deterministically.  ``SCORCH_HIGH_RANK_PROGRAMS``
picks how many programs of each part run (default 48, an even sample across
families); ``all`` runs every generated program, and ``@<file>`` the programs
the file names, one per line.
"""

from __future__ import annotations

import itertools
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

import scorch
from scorch import STensor
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import UnsupportedFeature
from scorch.compiler.loop_plan_legality import InvalidSchedule
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from scorch.compiler.loopir.schedule_passes import SchedulePassError
from scorch.compiler.scheduler import Schedule

DTYPE = torch.float64
TOLERANCE = 1e-12
MAX_CELLS = 4096


@dataclass(frozen=True)
class Program:
    family: str
    expr: str
    operand_formats: Tuple[str, ...]
    result_format: str
    extents: Tuple[Tuple[str, int], ...]

    @property
    def inputs(self) -> List[str]:
        return self.expr.split("->")[0].split(",")

    @property
    def output(self) -> str:
        return self.expr.split("->")[1]

    def extent(self, index: str) -> int:
        return dict(self.extents)[index]

    def shape(self, indices: str) -> Tuple[int, ...]:
        return tuple(self.extent(index) for index in indices)

    @property
    def name(self) -> str:
        operands = ",".join(self.operand_formats)
        sizes = "x".join(str(extent) for _, extent in self.extents)
        return (
            f"{self.family}: {self.expr} {operands} -> [{self.result_format}] "
            f"{sizes}"
        )


# -- the population ---------------------------------------------------------------


def _formats(rng: random.Random, rank: int, *, sparse: bool = False) -> str:
    """A random dense/compressed level string, with a compressed level if asked."""

    while True:
        fmt = "".join(rng.choice("ds") for _ in range(rank))
        if not sparse or "s" in fmt:
            return fmt


def _extents(rng: random.Random, indices: str, degenerate: bool) -> Dict[str, int]:
    while True:
        extents = {index: rng.randint(2, 4) for index in indices}
        if degenerate:
            extents[rng.choice(indices)] = rng.choice((0, 1))
        cells = 1
        for extent in extents.values():
            cells *= extent
        if cells <= MAX_CELLS:
            return extents


def _program(
    rng: random.Random, family: str, expr: str, *, degenerate: bool = False
) -> Program:
    inputs, output = expr.split("->")
    operands = inputs.split(",")
    indices = "".join(dict.fromkeys("".join(operands)))
    extents = _extents(rng, indices, degenerate)
    return Program(
        family=family,
        expr=expr,
        operand_formats=tuple(_formats(rng, len(operand)) for operand in operands),
        result_format=_formats(rng, len(output), sparse=True),
        extents=tuple((index, extents[index]) for index in indices),
    )


def _random_population() -> List[Program]:
    rng = random.Random(20261011)
    programs: List[Program] = []

    def add(family: str, exprs: Sequence[str], copies: int) -> None:
        for expr in exprs:
            for copy in range(copies):
                programs.append(
                    _program(rng, family, expr, degenerate=copy == copies - 1)
                )

    # Copies and permutations of the operand's modes.
    for rank in (4, 5, 6):
        letters = "ijklmn"[:rank]
        permuted = "".join(reversed(letters))
        rotated = letters[1:] + letters[0]
        add(
            "copy",
            [
                f"{letters}->{letters}",
                f"{letters}->{permuted}",
                f"{letters}->{rotated}",
            ],
            6,
        )
    # Reductions over one to five indices, kept indices in order and permuted.
    for rank in (4, 5, 6):
        letters = "ijklmn"[:rank]
        exprs = []
        for kept in (1, 2, rank - 1):
            for subset in itertools.combinations(letters, kept):
                exprs.append(f"{letters}->{''.join(subset)}")
        exprs = rng.sample(exprs, min(len(exprs), 8))
        exprs.append(f"{letters}->{letters[-1]}{letters[0]}")
        add("reduce", exprs, 3)
    # A matrix times each mode of a four- and five-dimensional tensor.
    add(
        "ttm",
        [
            "ijkl,ia->ajkl",
            "ijkl,ja->iakl",
            "ijkl,ka->ijal",
            "ijkl,la->ijka",
            "ijklm,ka->ijalm",
            "ijklm,ma->ijkla",
        ],
        4,
    )
    # Several matrices at once, and MTTKRP.
    add(
        "multi-ttm",
        ["ijkl,ia,jb->abkl", "ijkl,ka,lb->ijab", "ijklm,ia,jb,kc->abclm"],
        4,
    )
    add(
        "mttkrp", ["ijkl,jr,kr,lr->ir", "ijkl,ir,kr,lr->jr", "ijklm,jr,kr,lr,mr->ir"], 4
    )
    # Tensor-tensor contractions and batched products.
    add("contract", ["ijkl,klmn->ijmn", "ijkl,jklm->im", "ijkl,lmkn->ijmn"], 4)
    add("batched", ["bijk,bkl->bijl", "abij,abjk->abik", "bijk,bjk->bi"], 4)
    # Outer and elementwise products.
    add("outer", ["ij,kl->ijkl", "ijk,lmn->ijklmn", "ij,klm->ijklm"], 4)
    add("elementwise", ["ijklm,ijklm->ijklm", "ijkl,ijkl->ijkl", "ijkl,ijkl->il"], 4)
    return programs


def _layout_sweep(drawn: Sequence[Program]) -> List[Program]:
    """Every result layout of every expression, its operands and extents fixed.

    The drawn population gives an expression only the few result layouts its
    copies happened to draw, and never an all-dense one.  Here the result of
    each expression takes all ``2 ** rank`` dense/compressed layouts.  The first
    operand has a compressed level, so every program reads sparse storage.
    """

    rng = random.Random(20261012)
    taken = {program.name for program in drawn}
    swept: List[Program] = []
    for family, expr in dict.fromkeys((p.family, p.expr) for p in drawn):
        inputs, output = expr.split("->")
        operands = inputs.split(",")
        indices = "".join(dict.fromkeys("".join(operands)))
        extents = _extents(rng, indices, False)
        operand_formats = tuple(
            _formats(rng, len(operand), sparse=position == 0)
            for position, operand in enumerate(operands)
        )
        for layout in itertools.product("ds", repeat=len(output)):
            program = Program(
                family=family,
                expr=expr,
                operand_formats=operand_formats,
                result_format="".join(layout),
                extents=tuple((index, extents[index]) for index in indices),
            )
            # A drawn program with the same name is this very program.
            if program.name not in taken:
                taken.add(program.name)
                swept.append(program)
    return swept


def _population() -> List[Program]:
    """The drawn programs, then the layout sweep.

    The drawn programs come first so that their places, which seed their
    inputs, are what they were before the sweep existed.
    """

    drawn = _random_population()
    return [*drawn, *_layout_sweep(drawn)]


def _sample(programs: List[Program]) -> List[Program]:
    wanted = os.environ.get("SCORCH_HIGH_RANK_PROGRAMS", "48")
    if wanted == "all":
        return programs
    if wanted.startswith("@"):
        # A file of program names, one per line: rerun exactly those.
        names = set(Path(wanted[1:]).read_text().splitlines())
        return [program for program in programs if program.name in names]
    drawn = _random_population()
    return [
        *_sample_evenly(drawn, int(wanted)),
        *_sample_evenly(programs[len(drawn) :], int(wanted)),
    ]


def _sample_evenly(programs: List[Program], count: int) -> List[Program]:
    by_family: Dict[str, List[Program]] = defaultdict(list)
    for program in programs:
        by_family[program.family].append(program)
    rng = random.Random(7)
    chosen: List[Program] = []
    families = sorted(by_family)
    while len(chosen) < min(count, len(programs)):
        for family in families:
            pool = [p for p in by_family[family] if p not in chosen]
            if pool and len(chosen) < count:
                chosen.append(rng.choice(pool))
    return chosen


_POPULATION = _population()
PROGRAMS = _sample(_POPULATION)
# Each program's inputs are seeded by its place in the whole population, so a
# program gets the same inputs whatever sample it is run in.
_POPULATION_INDEX = {program.name: index for index, program in enumerate(_POPULATION)}
assert len(_POPULATION_INDEX) == len(_POPULATION), "program names must be unique"


# -- operands, the reference and the expected structure ----------------------------


def _level_arrays(
    indicator: torch.Tensor, fmt: str, ghosts: Optional[torch.Tensor] = None
) -> Tuple[List[Tuple[int, ...]], List[Optional[Tuple[List[int], List[int]]]]]:
    """Leaf coordinates and per-level (pos, crd) of ``indicator`` stored as ``fmt``.

    A dense level stores every coordinate below every stored parent; a
    compressed level stores exactly the coordinates something nonzero lies
    under.  This is the storage rule, written from the definition.

    ``ghosts`` builds an operand that is legal but not minimal: each cell of it
    contributes its coordinate to every compressed level that has a compressed
    level below, and nothing to the lowest compressed level, so the child
    segments under a ghost coordinate are empty and no entry is stored for it.
    """

    shape = tuple(indicator.shape)
    nonzero = [tuple(int(c) for c in coord) for coord in indicator.nonzero()]
    children: Dict[Tuple[int, ...], set] = defaultdict(set)
    for coord in nonzero:
        for level in range(len(shape)):
            children[coord[:level]].add(coord[level])
    lowest_compressed = max(
        (level for level, kind in enumerate(fmt) if kind != "d"), default=-1
    )
    if ghosts is not None:
        for coord in ghosts.nonzero():
            ghost = tuple(int(c) for c in coord)
            for level in range(lowest_compressed):
                children[ghost[:level]].add(ghost[level])
    parents: List[Tuple[int, ...]] = [()]
    levels: List[Optional[Tuple[List[int], List[int]]]] = []
    for level, kind in enumerate(fmt):
        if kind == "d":
            levels.append(None)
            parents = [p + (c,) for p in parents for c in range(shape[level])]
            continue
        pos, crd, below = [0], [], []
        for parent in parents:
            kids = sorted(children.get(parent, ()))
            crd.extend(kids)
            below.extend(parent + (kid,) for kid in kids)
            pos.append(len(crd))
        levels.append((pos, crd))
        parents = below
    return parents, levels


def _operand(
    dense: torch.Tensor,
    keep: torch.Tensor,
    fmt: str,
    name: str,
    ghosts: Optional[torch.Tensor] = None,
):
    """(STensor built from explicit level arrays, its storage indicator)."""

    leaves, levels = _level_arrays(keep, fmt, ghosts)
    mode_indices = [
        (
            []
            if arrays is None
            else [
                torch.tensor(arrays[0], dtype=torch.int64),
                torch.tensor(arrays[1], dtype=torch.int64),
            ]
        )
        for arrays in levels
    ]
    values = torch.tensor([float(dense[leaf]) for leaf in leaves], dtype=DTYPE)
    indicator = torch.zeros(dense.shape, dtype=DTYPE)
    for leaf in leaves:
        indicator[leaf] = 1.0
    tensor = STensor.from_components(
        tuple(dense.shape), fmt, mode_indices, values, name=name
    )
    return tensor, indicator


def _inputs(program: Program, variant: str, seed: int):
    operands, references, indicators = [], [], []
    for position, (indices, fmt) in enumerate(
        zip(program.inputs, program.operand_formats)
    ):
        shape = program.shape(indices)
        generator = torch.Generator().manual_seed(1000 * seed + position)
        values = torch.rand(shape, generator=generator, dtype=DTYPE) + 0.5
        if variant == "full":
            keep = torch.ones(shape, dtype=torch.bool)
        else:
            keep = torch.rand(shape, generator=generator) < 0.35
            if variant in ("gap", "hollow") and len(shape) and shape[0] > 1:
                keep[0] = False
                keep[-1] = False
        ghosts = None
        if variant == "hollow":
            # Drawn after ``keep``, so the other inputs are what they were
            # before this one existed.  The whole first slice is ghost, so a
            # coordinate with nothing under it exists wherever the layout can
            # hold one.
            ghosts = (torch.rand(shape, generator=generator) < 0.35) & ~keep
            if len(shape) and shape[0] > 1:
                ghosts[0] = True
        dense = values * keep
        name = "ABDEFGH"[position]
        operand, indicator = _operand(dense, keep, fmt, name, ghosts)
        if (
            variant == "hollow"
            and fmt.count("s") >= 2
            and shape[0] > 1
            and all(extent > 0 for extent in shape)
        ):
            # The input's premise, checked: the operand stores more
            # coordinates than entries account for.
            minimal = _level_arrays(keep, fmt)[1]
            hollow = _level_arrays(keep, fmt, ghosts)[1]
            assert sum(len(level[1]) for level in hollow if level) > sum(
                len(level[1]) for level in minimal if level
            ), f"{program.name}: the hollow input has no empty segment"
        operands.append(operand)
        references.append(dense)
        indicators.append(indicator)
    reference = torch.einsum(program.expr, *references)
    structure = torch.einsum(program.expr, *indicators) > 0
    return operands, reference, structure


# -- the two compilers -------------------------------------------------------------

_SHIPPED_REFUSALS = (NotImplementedError, UnsupportedFeature, InvalidSchedule)
_NEW_REFUSALS = (
    LoopIRLoweringError,
    LoopIRTargetError,
    InvalidSchedule,
    UnsupportedFeature,
    SchedulePassError,
)


def _options(blocking: bool) -> CompileOptions:
    return CompileOptions.from_environment().with_regblock_enabled(blocking)


def _cin(program: Program):
    letters = "".join(dict.fromkeys("".join(program.inputs)))
    variables = {index: IndexVar(index) for index in letters}
    product = None
    for position, (indices, fmt) in enumerate(
        zip(program.inputs, program.operand_formats)
    ):
        access = TensorVar("ABDEFGH"[position], fmt=fmt, dtype=DTYPE)[
            tuple(variables[index] for index in indices)
        ]
        product = (
            access if product is None else CINBinaryOp(Operation.MUL, product, access)
        )
    result = TensorVar("C", fmt=program.result_format, dtype=DTYPE)[
        tuple(variables[index] for index in program.output)
    ]
    statement = TensorAssign(result, product, op=Operation.ADD)
    for index in reversed(letters):
        statement = ForAll(variables[index], statement)
    return statement


def _run_shipped(program: Program, operands, blocking: bool):
    return scorch.einsum(
        program.expr,
        *operands,
        format=program.result_format,
        _compile_options=_options(blocking),
    )


def _run_new(program: Program, operands, blocking: bool):
    options = replace(_options(blocking), requested_schedule=Schedule())
    result = execute_cin_via_loopir(
        _cin(program),
        program.shape(program.output),
        *operands,
        compile_options=options,
    )
    return result[0] if isinstance(result, tuple) else result


# -- judging -----------------------------------------------------------------------


def _storage_order(result, rank: int) -> Tuple[int, ...]:
    order = getattr(result, "mode_order", None)
    if order is None:
        order = getattr(result.storage.index, "mode_order", None)
    return tuple(range(rank)) if order is None else tuple(int(m) for m in order)


def _judge(program: Program, result, reference, structure) -> None:
    if isinstance(result, torch.Tensor):
        got = result.to(DTYPE)
    else:
        result.copy()  # the library's own storage validation
        got = result.to_torch(in_place=False).to(DTYPE)
    scale = max(1.0, float(reference.abs().max())) if reference.numel() else 1.0
    error = float((got - reference).abs().max()) if reference.numel() else 0.0
    assert error <= TOLERANCE * scale, f"max error {error:.3e} at scale {scale:.3e}"
    if isinstance(result, torch.Tensor):
        return
    rank = len(program.output)
    # The result's levels are stored in its own mode order, and the format
    # letters name those storage levels in order.
    order = _storage_order(result, rank)
    stored_format = program.result_format
    expected_leaves, expected = _level_arrays(
        structure.permute(order) if rank else structure, stored_format
    )
    actual = []
    for level, arrays in enumerate(result.storage.index.mode_indices):
        if stored_format[level] == "d":
            actual.append(None)
        else:
            pos, crd = (t.reshape(-1).tolist() for t in arrays)
            actual.append((pos, crd))
    assert actual == expected, "stored coordinates differ from the operands' structure"
    assert result.storage.value.numel() == len(expected_leaves)


def _refusal_code(error: BaseException) -> str:
    """The structured reason of a refusal: a defect or diagnostic code."""

    defect = getattr(error, "defect", None)
    if defect is not None:
        return str(defect.code)
    for diagnostic in getattr(error, "diagnostics", ()):
        code = getattr(diagnostic, "code", None)
        if code is not None:
            return str(code)
    return type(error).__name__


class WrongResult(AssertionError):
    """A result the compiler returned that fails a check.

    Kept apart from an ``AssertionError`` the compiler itself raises, which is
    an unstructured failure, not a wrong answer.
    """


def _outcome(run, program: Program, blocking: bool, refusals) -> str:
    """``"correct"``, or ``"refused: <code>"``; a wrong result raises."""

    seed = _POPULATION_INDEX[program.name]
    outcomes = []
    for variant in ("full", "sparse", "gap", "hollow"):
        operands, reference, structure = _inputs(program, variant, seed)
        try:
            result = run(program, operands, blocking)
        except refusals as error:
            outcomes.append(f"refused: {_refusal_code(error)}")
            continue
        try:
            _judge(program, result, reference, structure)
        except AssertionError as error:
            raise WrongResult(f"{variant} input: {error}") from error
        outcomes.append("correct")
    # One program is one kernel: its inputs differ in what they store, never
    # in shape or format, so they cannot disagree about whether it compiles.
    assert len(set(outcomes)) == 1, f"the inputs disagree: {outcomes}"
    return outcomes[0]


EXPECTED_PATH = Path(__file__).with_name("high_rank_expected.json")
_EXPECTED_FILE = json.loads(EXPECTED_PATH.read_text()) if EXPECTED_PATH.exists() else {}
# ``{"shipped": {program: "refused: <code>"}, "new": {...}}``
EXPECTED: Dict[str, Dict[str, str]] = _EXPECTED_FILE.get("refused", {})
RECORD = os.environ.get("SCORCH_HIGH_RANK_RECORD")

# Programs a compiler currently gets wrong or fails on without a named reason,
# each with the cause, kept in the same file under ``"defects"``.  Strict: a
# program that starts passing must leave the table.
KNOWN_DEFECTS: Dict[Tuple[str, str], str] = {
    (compiler, name): cause
    for compiler, programs in _EXPECTED_FILE.get("defects", {}).items()
    for name, cause in programs.items()
}


def _record(compiler: str, program: Program, blocking: bool, outcome: str) -> None:
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    line = {
        "compiler": compiler,
        "program": program.name,
        "family": program.family,
        "blocking": blocking,
        "outcome": outcome,
    }
    with open(Path(RECORD) / f"{worker}.jsonl", "a") as handle:
        handle.write(json.dumps(line) + "\n")


def _check(compiler: str, run, program: Program, blocking: bool, refusals) -> None:
    if RECORD:
        try:
            outcome = _outcome(run, program, blocking, refusals)
        except WrongResult as error:
            outcome = f"wrong: {str(error)[:300]}"
        except Exception as error:
            outcome = f"error: {type(error).__name__}: {str(error)[:300]}"
        _record(compiler, program, blocking, outcome)
        return
    outcome = _outcome(run, program, blocking, refusals)
    listed = EXPECTED.get(compiler, {}).get(program.name, "correct")
    assert outcome == listed, (
        f"the {compiler} compiler's outcome is {outcome!r}; "
        f"{EXPECTED_PATH.name} says {listed!r}"
    )


def _parameters(compiler: str):
    parameters = []
    for program in PROGRAMS:
        reason = KNOWN_DEFECTS.get((compiler, program.name))
        marks = (
            [pytest.mark.xfail(strict=True, reason=reason)]
            if reason and not RECORD
            else []
        )
        parameters.append(pytest.param(program, id=program.name, marks=marks))
    return parameters


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("program", _parameters("shipped"))
def test_shipped_compiler_matches_pytorch(program, blocking):
    _check("shipped", _run_shipped, program, blocking, _SHIPPED_REFUSALS)


@pytest.mark.parametrize("blocking", (False, True), ids=("plain", "blocked"))
@pytest.mark.parametrize("program", _parameters("new"))
def test_new_compiler_matches_pytorch(program, blocking):
    _check("new", _run_new, program, blocking, _NEW_REFUSALS)


def test_the_expected_file_names_only_generated_programs():
    names = {program.name for program in _population()}
    for compiler, refused in EXPECTED.items():
        assert compiler in ("shipped", "new")
        assert set(refused) <= names, sorted(set(refused) - names)[:5]
        assert all(reason.startswith("refused: ") for reason in refused.values())
    assert {name for _, name in KNOWN_DEFECTS} <= names


def test_every_expression_meets_every_result_layout():
    layouts: Dict[Tuple[str, str], set] = defaultdict(set)
    for program in _population():
        layouts[(program.family, program.expr)].add(program.result_format)
    for (_, expr), met in layouts.items():
        rank = len(expr.split("->")[1])
        every = {"".join(layout) for layout in itertools.product("ds", repeat=rank)}
        assert met == every, f"{expr}: {sorted(every - met)}"


def test_population_covers_every_family_and_rank():
    programs = _population()
    families = {program.family for program in programs}
    assert families == {
        "copy",
        "reduce",
        "ttm",
        "multi-ttm",
        "mttkrp",
        "contract",
        "batched",
        "outer",
        "elementwise",
    }
    ranks = {len(program.extents) for program in programs}
    assert {4, 5, 6} <= ranks
    assert all(
        len(fmt) == len(indices)
        for program in programs
        for fmt, indices in zip(program.operand_formats, program.inputs)
    )
