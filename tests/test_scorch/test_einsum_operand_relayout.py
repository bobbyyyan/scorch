"""einsum's operand relayout: the format a relaid operand is stored in, and which entries move.

``scorch.einsum`` moves an operand to the storage order its kernel reads with
``STensor.change_mode_order``, which keeps the format letters by storage position and
copies only nonzero values.  ``scorch.ops._einsum_operand_relayout`` decides per operand,
from metadata only, whether the operand is converted instead (one move, every level
compressed, the kernel compiled for those letters) and whether every stored entry is
copied.  Its docstring states the rule and why it holds.  These tests hold it to that:

* class 2 (some operand converts): on a constructed input -- the converted operand stores
  the cells whose modes at or above its deepest compressed level are all 0 and every cell
  below, every value nonzero; every other operand stores every cell -- the legacy relayout
  stores result coordinates the operands do not imply, and the decision stores exactly the
  implied ones;
* an operand relaid without conversion (class 3): exact stored coordinates on the high-rank
  test's sparse, gap and hollow inputs (stored zeros below a compressed level, empty
  slices, coordinates with nothing under them), the only inputs where the argument for
  leaving its letters alone could fail;
* stored zeros: the relaid operand stores its entries with the value 0; the legacy copy
  drops them and the result misses the coordinates they imply; the decision keeps them;
* main-kernel text at PYTHONHASHSEED 0 and 1 (einsum's loop order follows the hash seed):
  identical with and without the decision for every program in which no operand converts,
  different in every program in which one does, and those programs are exactly the pinned
  class-2 lists;
* the extent clause: a dense level of extent 1 below a compressed one adds no cell, and an
  index of extent 0 empties the iteration space, so neither converts; the same programs
  with that extent raised to 2 convert.

"Legacy" is this tree with ``_einsum_operand_relayout`` answering what einsum did before it
existed: the same moves, letters kept, nonzero values copied.  Values are compared to the
float64 reference at a float32-level tolerance (a workspace accumulates in float32 until the
workspace-dtype patch); stored coordinates are compared exactly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import pytest
import torch

import scorch
import scorch.ops as ops
from scorch.format import conversion_format, stores_only_its_entries
from tests.test_scorch.kernel_cache_support import empty_kernel_caches

import test_high_rank_programs as H

TESTS = Path(__file__).parent
RTOL = 1e-6
DECIDE = ops._einsum_operand_relayout

# The programs whose relayout einsum converts, by population, at PYTHONHASHSEED 0 and 1.
# The high-rank population at seed 0; seed 1 adds HIGH_RANK_CLASS_2_SEED_1.
HIGH_RANK_CLASS_2 = (
    "batched: bijk,bjk->bi ddsd,sss -> [ss] 4x3x2x2",
    "reduce: ijkl->i ddsd -> [s] 2x4x2x3",
    "reduce: ijkl->i ddsd -> [s] 3x2x3x2",
    "reduce: ijkl->ij dssd -> [ds] 3x2x3x2",
    "reduce: ijkl->i ssdd -> [s] 4x4x2x3",
    "reduce: ijkl->jk sssd -> [ss] 2x3x2x3",
    "reduce: ijklm->i dsdsd -> [s] 2x3x4x2x4",
    "reduce: ijklm->i dssdd -> [s] 3x4x1x4x4",
    "reduce: ijklm->ijkl sdssd -> [dsss] 4x2x2x2x1",
    "reduce: ijklm->i ssssd -> [s] 2x2x4x3x4",
    "reduce: ijklm->kl sddsd -> [ss] 2x2x3x3x4",
    "reduce: ijklmn->i dsdddd -> [s] 3x2x2x4x3x2",
    "reduce: ijklmn->i dssssd -> [s] 2x2x4x3x2x3",
    "reduce: ijklmn->i sddssd -> [s] 4x4x2x2x4x3",
    "reduce: ijklmn->k ddsssd -> [s] 3x3x3x2x4x3",
)
HIGH_RANK_CLASS_2_SEED_1 = (
    "contract: ijkl,lmkn->ijmn dsdd,dsds -> [ssss] 2x3x2x4x2x3",
)
# The reduction sweep's cells (test_sparse_output_reduction_sweep.py), and scorch-merge's
# shipped census on two and three indices (their extents here: 3, 4, 5).
SWEEP_CLASS_2 = (
    "dsd ijk->i as s",
    "sdd ijk->i as s",
    "ssd ijk->i as s",
    "dsd ijk->j as s",
    "ssd ijk->j as s",
    "dsd ijk->ij as ss",
    "dsd ijk->ij as ds",
    "ssd ijk->ij as ss",
    "ssd ijk->ij as ds",
)
CENSUS_CLASS_2 = (
    "sd ij->i as s",
    "dsd ijk->ji as ds",
    "dsd ijk->ji as ss",
    "ssd ijk->ji as ds",
    "ssd ijk->ji as ss",
)  # and SWEEP_CLASS_2 less "sdd ijk->i as s"
# The extent clause: (program, the same program with that extent raised to 2).
EXTENT_ONLY = (
    (
        "reduce: ijkl->i ssdd -> [s] 1x3x2x3",
        "reduce: ijkl->i ssdd -> [s] 2x3x2x3",
    ),
    (
        "reduce: ijklmn->ni ddddsd -> [ss] 4x2x3x0x3x4",
        "reduce: ijklmn->ni ddddsd -> [ss] 4x2x3x2x3x4",
    ),
)
# Raised to 2, the first converts.  The second does not move at all: with l = 2 einsum's
# loop order keeps the operand in its input order.  The clause is checked on the
# decision function with the orders einsum gives the extent-0 program
# (test_the_extent_clause_alone_keeps_the_extent_zero_program_unconverted).
EXTENT_RAISED_CONVERTS = (EXTENT_ONLY[0][1],)
# The programs that relay an operand whose letters put a dense level below a compressed one
# without converting it, at PYTHONHASHSEED 0 or 1 (75 at seed 0, 74 at seed 1).
HIGH_RANK_CLASS_3 = (
    "batched: bijk,bjk->bi ssds,ssd -> [ds] 3x4x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [ddds] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [ddsd] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [ddss] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [dsdd] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [dsds] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [dssd] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [dsss] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [sddd] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [sdds] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [sdsd] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [sdss] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [ssdd] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [ssds] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [sssd] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [ssss] 3x2x3x3x2x4",
    "contract: ijkl,lmkn->ijmn ddsd,sssd -> [ddsd] 3x2x2x4x2x2",
    "contract: ijkl,lmkn->ijmn dsdd,dsds -> [ssss] 2x3x2x4x2x3",
    "contract: ijkl,lmkn->ijmn ssss,dsds -> [ddsd] 3x3x4x2x2x3",
    "reduce: ijkl->i ssdd -> [s] 1x3x2x3",
    "reduce: ijkl->ij dsds -> [ds] 3x4x2x4",
    "reduce: ijkl->ij sddd -> [ds] 2x4x1x2",
    "reduce: ijkl->ij sdds -> [ds] 3x2x2x4",
    "reduce: ijkl->ij sdds -> [sd] 3x2x2x4",
    "reduce: ijkl->ij sdds -> [ss] 3x2x2x4",
    "reduce: ijkl->ik ddsd -> [sd] 4x4x2x1",
    "reduce: ijkl->ik sddd -> [ds] 3x3x3x2",
    "reduce: ijkl->ik sddd -> [sd] 3x3x3x2",
    "reduce: ijkl->ik sddd -> [ss] 3x3x3x2",
    "reduce: ijkl->jk dsdd -> [ds] 4x2x4x3",
    "reduce: ijkl->jk dsdd -> [sd] 4x2x4x3",
    "reduce: ijkl->jk dsdd -> [ss] 4x2x4x3",
    "reduce: ijkl->jk dssd -> [sd] 3x3x4x0",
    "reduce: ijklm->ijkl ddsds -> [ddds] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [ddsd] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [ddss] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [dsdd] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [dsds] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [dssd] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [dsss] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [sddd] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [sdds] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [sdsd] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [sdss] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [ssdd] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [ssds] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [sssd] 4x4x2x3x2",
    "reduce: ijklm->ijkl ddsds -> [ssss] 4x4x2x3x2",
    "reduce: ijklm->ijkl ssdsd -> [dssd] 4x4x2x4x4",
    "reduce: ijklm->ijkl ssdss -> [ssss] 2x2x2x2x3",
    "reduce: ijklm->ik dddsd -> [sd] 4x2x3x4x0",
    "reduce: ijklm->ik ddsds -> [sd] 2x4x4x2x2",
    "reduce: ijklm->ik ddsds -> [sd] 4x4x2x2x4",
    "reduce: ijklm->ik sdsds -> [ds] 3x4x2x3x4",
    "reduce: ijklm->ik sdsds -> [sd] 3x4x2x3x4",
    "reduce: ijklm->ik sdsds -> [ss] 3x4x2x3x4",
    "reduce: ijklm->kl dsdss -> [ss] 4x2x2x4x3",
    "reduce: ijklm->l ssdss -> [s] 4x4x3x4x4",
    "reduce: ijklm->l sssds -> [s] 3x2x4x2x4",
    "reduce: ijklm->l sssds -> [s] 4x0x3x4x4",
    "reduce: ijklmn->im dsddss -> [ds] 2x2x3x4x1x2",
    "reduce: ijklmn->im dssddd -> [ds] 2x4x4x3x4x4",
    "reduce: ijklmn->im dssddd -> [sd] 2x4x4x3x4x4",
    "reduce: ijklmn->im dssddd -> [ss] 2x4x4x3x4x4",
    "reduce: ijklmn->im sddssd -> [sd] 4x4x3x2x2x2",
    "reduce: ijklmn->jk ddsddd -> [sd] 3x3x3x2x3x4",
    "reduce: ijklmn->jk ssddds -> [ds] 2x3x2x2x4x4",
    "reduce: ijklmn->jk ssddds -> [sd] 2x3x2x2x4x4",
    "reduce: ijklmn->jk ssddds -> [ss] 2x3x2x2x4x4",
    "reduce: ijklmn->jk sssddd -> [sd] 2x1x2x2x4x4",
    "reduce: ijklmn->k dsdsds -> [s] 4x3x2x2x3x4",
    "reduce: ijklmn->l dsdddd -> [s] 3x4x2x4x4x2",
    "reduce: ijklmn->l sddddd -> [s] 2x4x4x2x4x4",
    "reduce: ijklmn->l sssddd -> [s] 2x2x3x2x4x1",
    "reduce: ijklmn->ni ddddsd -> [ss] 4x2x3x0x3x4",
)
# Stored zeros: the relaid operand stores its entries with the value 0.
STORED_ZERO_PROGRAMS = (
    "reduce: ijkl->ik sddd -> [ss] 3x3x3x2",
    "batched: bijk,bjk->bi dsss,dss -> [ss] 3x2x4x4",
    "contract: ijkl,lmkn->ijmn ddsd,dddd -> [ssss] 3x2x3x3x2x4",
    "reduce: ijkl->ik dsss -> [ss] 3x3x2x4",
    "reduce: ijkl->i ssdd -> [s] 4x4x2x3",
)
# A relayout of a six-index operand runs through a six-key workspace, which the shipped
# code generator does not provide (llir.DataType lists coo_workspace<T, N> for N <= 5).
SIX_KEY_WORKSPACE = "coo_workspace"


def _parse(name: str) -> H.Program:
    """A program from its high-rank name (``family: expr formats -> [result] extents``)."""

    family, rest = name.split(": ", 1)
    expr, formats, _, result, sizes = rest.split(" ")
    letters = "".join(dict.fromkeys(expr.split("->")[0].replace(",", "")))
    extents = tuple(zip(letters, (int(size) for size in sizes.split("x"))))
    return H.Program(family, expr, tuple(formats.split(",")), result[1:-1], extents)


# The reduction sweep's SHAPES on three and four indices, and two for the census's
# two-index program.
ONE_OPERAND_SHAPES = {2: (3, 4), 3: (3, 4, 5), 4: (2, 3, 4, 5)}


def _one_operand_reduction(
    expr: str, operand_format: str, result_format: str
) -> H.Program:
    letters = expr.split("->")[0]
    extents = tuple(zip(letters, ONE_OPERAND_SHAPES[len(letters)]))
    return H.Program("reduce", expr, (operand_format,), result_format, extents)


def _sweep_program(cell_id: str) -> H.Program:
    """A one-operand reduction from a reduction-sweep cell id (``fmt expr as result``)."""

    operand_format, expr, _, result_format = cell_id.split(" ")
    return _one_operand_reduction(expr, operand_format, result_format)


CLASS_2_PROGRAMS = (
    [_parse(name) for name in HIGH_RANK_CLASS_2 + HIGH_RANK_CLASS_2_SEED_1]
    + [_sweep_program(cell) for cell in SWEEP_CLASS_2]
    + [_sweep_program(cell) for cell in CENSUS_CLASS_2]
)


def _legacy(
    operand: int,
    operand_indices: Sequence[Sequence[str]],
    operand_letters: Sequence[str],
    input_orders: Sequence[Sequence[int]],
    aligned_orders: Sequence[Sequence[int]],
    *_: object,
) -> ops._OperandRelayout:
    """What einsum did before the decision existed: every aligned move, letters kept,
    nonzero values copied."""

    moves: List[Tuple[int, ...]] = []
    current = tuple(input_orders[operand])
    for aligned_order in aligned_orders:
        if tuple(aligned_order) != current:
            current = tuple(aligned_order)
            moves.append(current)
    return ops._OperandRelayout(tuple(moves), None, False)


def _run(
    program: H.Program, operands, decide: Callable[..., ops._OperandRelayout]
) -> Tuple[object, List[ops._OperandRelayout]]:
    decisions: List[ops._OperandRelayout] = []

    def recording(*args: object) -> ops._OperandRelayout:
        answer = decide(*args)
        decisions.append(answer)
        return answer

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ops, "_einsum_operand_relayout", recording)
        empty_kernel_caches(patch)
        result = scorch.einsum(
            program.expr,
            *operands,
            format=program.result_format,
            _compile_options=H._options(False),
        )
    return result, decisions


def _coordinates(program: H.Program, result) -> Tuple[set, list]:
    """(the result's stored coordinates, its level arrays) in its own storage order."""

    rank = len(program.output)
    order = H._storage_order(result, rank)
    actual = []
    for level, arrays in enumerate(result.storage.index.mode_indices):
        if program.result_format[level] == "d":
            actual.append(None)
        else:
            pos, crd = (t.reshape(-1).tolist() for t in arrays)
            actual.append((pos, crd))
    stored = set()
    parents: List[Tuple[int, ...]] = [()]
    shape = [program.extent(program.output[mode]) for mode in order]
    for level, arrays in enumerate(actual):
        if arrays is None:
            parents = [p + (c,) for p in parents for c in range(shape[level])]
        else:
            pos, crd = arrays
            parents = [
                parent + (crd[q],)
                for p, parent in enumerate(parents)
                for q in range(pos[p], pos[p + 1])
            ]
    for leaf in parents:
        stored.add(tuple(leaf[order.index(mode)] for mode in range(rank)))
    return stored, actual


def _implied(program: H.Program, result, structure) -> Tuple[set, list]:
    rank = len(program.output)
    order = H._storage_order(result, rank)
    leaves, levels = H._level_arrays(
        structure.permute(order) if rank else structure, program.result_format
    )
    implied = {
        tuple(leaf[order.index(mode)] for mode in range(rank)) for leaf in leaves
    }
    return implied, levels


def _values_match(result, reference) -> bool:
    got = result.to_torch(in_place=False).to(H.DTYPE)
    scale = max(1.0, float(reference.abs().max())) if reference.numel() else 1.0
    error = float((got - reference).abs().max()) if reference.numel() else 0.0
    return error <= RTOL * scale


def _exact(program: H.Program, result, reference, structure) -> bool:
    result.copy()  # the library's own storage validation
    stored, actual = _coordinates(program, result)
    implied, expected = _implied(program, result, structure)
    return (
        stored == implied
        and actual == expected
        and result.storage.value.numel() == len(stored)
        and _values_match(result, reference)
    )


def _constructed(
    program: H.Program, relaid: int, zeros: bool = False
) -> Tuple[list, torch.Tensor, torch.Tensor]:
    """Operand ``relaid`` stores the cells whose modes at or above its deepest compressed
    level are all 0 and every cell below; with ``zeros`` every value it stores is 0.
    Every other operand stores every cell.  Values otherwise in [0.5, 1.5)."""

    operands, references, indicators = [], [], []
    for position, (indices, fmt) in enumerate(
        zip(program.inputs, program.operand_formats)
    ):
        shape = program.shape(indices)
        generator = torch.Generator().manual_seed(7000 + position)
        values = torch.rand(shape, generator=generator, dtype=H.DTYPE) + 0.5
        keep = torch.ones(shape, dtype=torch.bool)
        if position == relaid:
            deepest = max(
                (level for level, letter in enumerate(fmt) if letter != "d"),
                default=-1,
            )
            for level in range(deepest + 1):
                index = [slice(None)] * len(shape)
                index[level] = slice(1, None)
                keep[tuple(index)] = False
            if zeros:
                values = torch.zeros(shape, dtype=H.DTYPE)
        operand, indicator = H._operand(values * keep, keep, fmt, "ABDEFGH"[position])
        operands.append(operand)
        references.append(values * keep)
        indicators.append(indicator)
    reference = torch.einsum(program.expr, *references)
    structure = torch.einsum(program.expr, *indicators) > 0
    return operands, reference, structure


def _outcome(program, relaid, decide, zeros=False):
    """("raised", message) | ("ok", exact, decisions, excess, missing)."""

    operands, reference, structure = _constructed(program, relaid, zeros)
    try:
        result, decisions = _run(program, operands, decide)
    except ValueError as error:
        if SIX_KEY_WORKSPACE in str(error):
            return ("raised", str(error))
        raise
    stored, _ = _coordinates(program, result)
    implied, _ = _implied(program, result, structure)
    return (
        "ok",
        _exact(program, result, reference, structure),
        decisions,
        len(stored - implied),
        len(implied - stored),
    )


# -- the two functions moved to scorch.format ---------------------------------------


def test_conversion_format_compresses_every_level_when_a_dense_level_lies_below():
    assert stores_only_its_entries("dds")
    assert not stores_only_its_entries("dsd")
    assert conversion_format("ds") == "ds"
    assert conversion_format("dsd") == "sss"
    assert conversion_format("ssdd") == "ssss"
    assert ops.conversion_format is conversion_format
    assert ops.stores_only_its_entries is stores_only_its_entries


def test_the_loopir_operand_conversion_uses_the_same_two_functions():
    # The new compiler's operand conversion decides the converted layout with the same
    # functions, imported from scorch.format; skipped on a tree without that module.
    operand_conversion = pytest.importorskip(
        "scorch.compiler.loopir.operand_conversion"
    )
    assert operand_conversion.conversion_format is conversion_format
    assert operand_conversion.stores_only_its_entries is stores_only_its_entries


# -- (i) class 2: wrong before, exact after -------------------------------------------


@pytest.mark.parametrize("program", CLASS_2_PROGRAMS, ids=lambda p: p.name)
def test_a_converted_operand_stores_only_its_entries(program):
    after = _outcome(program, 0, DECIDE)
    before = _outcome(program, 0, _legacy)
    if after[0] == "raised":
        # The relayout of a six-index operand runs through a six-key workspace,
        # which the code generator does not provide; the decision does not change it.
        assert len(program.inputs[0]) == 6, after[1]
        return
    assert after[1], f"after: {after[3]} excess, {after[4]} missing"
    converted = any(d.format_letters is not None for d in after[2])
    if converted and before[0] == "ok":
        assert not before[1], "the legacy relayout is exact here"
        assert before[3] > 0, "the legacy relayout stores no extra coordinate"
    elif converted:
        assert len(program.inputs[0]) == 6, before[1]
    else:
        # Under this hash seed einsum's loop order exposes nothing; both are exact.
        assert before[0] == "ok" and before[1]


@pytest.mark.parametrize(
    "program",
    [_sweep_program(cell) for cell in SWEEP_CLASS_2]
    + [_parse(name) for name in HIGH_RANK_CLASS_2[:6]],
    ids=lambda p: p.name,
)
def test_a_call_served_from_the_dispatch_cache_converts_the_same_way(program):
    # The second call takes einsum's cached path, which moves each operand once with
    # the decision stored in the cache entry.
    operands, reference, structure = _constructed(program, 0)
    with pytest.MonkeyPatch.context() as patch:
        empty_kernel_caches(patch)
        results = [
            scorch.einsum(
                program.expr,
                *operands,
                format=program.result_format,
                _compile_options=H._options(False),
            )
            for _ in range(2)
        ]
        assert len(ops._einsum_dispatch_cache) == 1
    for result in results:
        assert _exact(program, result, reference, structure)


# -- (ii) relaid without conversion: exact on stored zeros and empty slices ------------


@pytest.mark.parametrize("variant", ("sparse", "gap", "hollow"))
@pytest.mark.parametrize("name", HIGH_RANK_CLASS_3)
def test_an_operand_relaid_without_conversion_keeps_the_stored_structure(name, variant):
    program = _parse(name)
    operands, reference, structure = H._inputs(
        program, variant, H._POPULATION_INDEX[name]
    )
    try:
        result, decisions = _run(program, operands, DECIDE)
    except ValueError as error:
        assert SIX_KEY_WORKSPACE in str(error)
        assert any(len(indices) == 6 for indices in program.inputs)
        return
    assert _exact(program, result, reference, structure)


@pytest.mark.parametrize("pair", EXTENT_ONLY, ids=lambda pair: pair[0])
def test_an_extent_only_program_is_exact_on_its_constructed_input(pair):
    program = _parse(pair[0])
    after = _outcome(program, 0, DECIDE)
    if after[0] == "raised":
        assert any(len(indices) == 6 for indices in program.inputs)
        return
    assert after[1], f"{after[3]} excess, {after[4]} missing"
    assert not any(d.format_letters is not None for d in after[2])


# -- (iii) stored zeros --------------------------------------------------------------


@pytest.mark.parametrize("name", STORED_ZERO_PROGRAMS)
def test_every_stored_entry_moves_whatever_its_value(name):
    program = _parse(name)
    probe = _outcome(program, 0, DECIDE)
    relaid = next(
        (
            position
            for position, decision in enumerate(probe[2])
            if decision.mode_orders and "s" in program.operand_formats[position]
        ),
        None,
    )
    assert relaid is not None, "no operand with a compressed level moves"
    after = _outcome(program, relaid, DECIDE, zeros=True)
    before = _outcome(program, relaid, _legacy, zeros=True)
    assert after[0] == "ok" and after[1], f"after: {after[4]} missing"
    assert before[0] == "ok" and before[4] > 0, "the legacy copy misses nothing"


# -- (iv) main-kernel text and the extent clause, at hash seeds 0 and 1 ----------------

_TEXT_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import scorch
import scorch.ops as ops
from scorch.layout import TensorSpec
import test_high_rank_programs as H
import test_einsum_operand_relayout as R
from kernel_cache_support import clear_kernel_caches

class Blocked(Exception):
    pass

captured = {}

def capture(header, cpp, *args, **kwargs):
    captured["main"] = cpp
    raise Blocked()

ops._prepare_generated_kernel_build = capture
real = ops._einsum_operand_relayout

def main_text(program, blocking, decide):
    decisions = []

    def recording(*args):
        answer = decide(*args)
        decisions.append(answer)
        return answer

    ops._einsum_operand_relayout = recording
    clear_kernel_caches()
    captured.clear()
    operands = [
        TensorSpec(fmt, program.shape(indices), dtype=H.DTYPE, name="ABDEFGH"[t])
        for t, (indices, fmt) in enumerate(zip(program.inputs, program.operand_formats))
    ]
    try:
        scorch.einsum(program.expr, *operands, compile_only=True,
                      format=program.result_format, _compile_options=H._options(blocking))
        text = "returned"
    except Blocked:
        text = captured["main"]
    except Exception as error:
        text = "raised " + type(error).__name__ + ": " + str(error)[:200]
    return text, decisions

out = {}
for name in json.loads(sys.argv[2]):
    program = R._parse(name) if ": " in name else R._sweep_program(name)
    for blocking in (False, True):
        text, decisions = main_text(program, blocking, real)
        converted = any(d.format_letters is not None for d in decisions)
        default = all(not d.move_kwargs() for d in decisions)
        legacy = text if default else main_text(program, blocking, R._legacy)[0]
        out[name + (" blocked" if blocking else " plain")] = {
            "converted": converted, "changed": text != legacy,
            "raised": text.startswith("raised"),
        }
print(json.dumps(out))
"""


def _text_population() -> List[str]:
    """The high-rank test's selected programs (SCORCH_HIGH_RANK_PROGRAMS), the reduction
    sweep's cells, and every program pinned above."""

    sweep = __import__("test_sparse_output_reduction_sweep")
    names = [program.name for program in H.PROGRAMS]
    names += [sweep._cell_id(cell) for cell in sweep.CELLS]
    names += [name for name in CENSUS_CLASS_2 if name not in names]
    names += [
        name
        for name in HIGH_RANK_CLASS_2 + HIGH_RANK_CLASS_2_SEED_1
        if name not in names
    ]
    names += [name for pair in EXTENT_ONLY for name in pair if name not in names]
    names += [name for name in STORED_ZERO_PROGRAMS if name not in names]
    return names


@pytest.mark.parametrize("seed", ("0", "1"))
def test_main_kernel_text_changes_exactly_where_an_operand_converts(seed):
    names = _text_population()
    environment = dict(os.environ, PYTHONHASHSEED=seed)
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    completed = subprocess.run(
        [sys.executable, "-c", _TEXT_SCRIPT, str(TESTS), json.dumps(names)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    runs = json.loads(completed.stdout.strip().splitlines()[-1])
    assert all(run["changed"] == run["converted"] for run in runs.values()), [
        key for key, run in runs.items() if run["changed"] != run["converted"]
    ]
    converted = {key.rsplit(" ", 1)[0] for key, run in runs.items() if run["converted"]}
    class_2 = set(HIGH_RANK_CLASS_2) | set(SWEEP_CLASS_2) | set(CENSUS_CLASS_2)
    if seed == "1":
        class_2 |= set(HIGH_RANK_CLASS_2_SEED_1)
    assert converted == (class_2 | set(EXTENT_RAISED_CONVERTS)) & set(names)
    for kept, _ in EXTENT_ONLY:
        for arm in ("plain", "blocked"):
            assert not runs[f"{kept} {arm}"]["changed"]
    assert not runs[f"{EXTENT_ONLY[1][1]} plain"]["converted"]


def test_the_extent_clause_alone_keeps_the_extent_zero_program_unconverted():
    # einsum's operand orders for reduce: ijklmn->ni ddddsd -> [ss] 4x2x3x0x3x4: the
    # operand moves from ijklmn to jklmni, which puts the result mode i (extent 4) on a
    # dense level below the compressed m.  Only the extent 0 of l keeps it unconverted.
    extents = {"i": 4, "j": 2, "k": 3, "l": 0, "m": 3, "n": 4}
    arguments = (
        0,
        [list("ijklmn")],
        ["ddddsd"],
        [(0, 1, 2, 3, 4, 5)],
        ((1, 2, 3, 4, 5, 0), (1, 2, 3, 4, 5, 0)),
        list("ni"),
        "ss",
        (0, 1),
    )
    kept = DECIDE(*arguments, extents)
    assert kept.format_letters is None and kept.keep_every_stored_entry
    assert kept.mode_orders == ((1, 2, 3, 4, 5, 0),)
    converted = DECIDE(*arguments, dict(extents, l=2))
    assert converted.format_letters == "ssssss"
    assert converted.mode_orders == ((1, 2, 3, 4, 5, 0),)
    one = DECIDE(*arguments, dict(extents, l=2, i=1))
    assert one.format_letters is None, "a dense level of extent 1 adds no cell"
