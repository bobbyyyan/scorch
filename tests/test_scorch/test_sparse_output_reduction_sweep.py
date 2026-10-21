"""Every order-preserving reduction into a sparse result, through both compilers.

A sparse result is appended to in iteration order, so the shipped compiler must
drain each result coordinate once, after every contribution to it has been
summed, and must assemble every compressed level above the drain.  Two defects
broke that for any nest with more than one summed-over loop or more than two
compressed result levels: the workspace wrapped only the innermost reduction
loop (so an outer reduction loop drained the same coordinates again), and the
nested drain closed only the level directly above it.  The cells below are the
whole family at rank three and four, not the shapes that were found broken.

For each cell:

* the shipped compiler (public ``scorch.einsum``) must return the PyTorch
  result, in storage that passes validation (the suite validates every kernel
  result);
* the refactor's LoopIR route must return the same result, or refuse with a
  structured defect code -- never a wrong answer, never an unclassified error;
* each compiler's level arrays must be the ones the operand's stored entries
  imply: a compressed result coordinate is stored exactly when the operand
  stores an entry under it, whatever its value (structure decides, never
  values).  The reference is computed from the operand's own levels, not from
  either compiler, so a defect in one compiler cannot pass as the standard
  the other is held to;
* where both store the implied structure, their values must agree to float32
  rounding.
"""

from dataclasses import replace
from itertools import combinations, product

import pytest
import torch

import scorch
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from scorch.compiler.scheduler import Schedule
from scorch.stensor import STensor

ATOL = RTOL = 1e-3

SHAPES = {3: (3, 4, 5), 4: (2, 3, 4, 5)}
OPERAND_FORMATS = {
    3: ["".join(levels) for levels in product("ds", repeat=3)],
    4: ["ssss", "dsss", "sdss", "ssds"],
}


def _result_formats(rank):
    formats = ["s" * rank]
    if rank > 1:
        formats.append("d" + "s" * (rank - 1))
    return formats


def _cells():
    cells = []
    for rank, operand_formats in OPERAND_FORMATS.items():
        operand = "ijkl"[:rank]
        for size in range(1, rank):
            for kept in combinations(operand, size):
                result = "".join(kept)
                for operand_fmt in operand_formats:
                    for result_fmt in _result_formats(len(result)):
                        cells.append((operand, result, operand_fmt, result_fmt))
    return cells


CELLS = _cells()


def _options():
    return replace(
        CompileOptions.from_environment().with_regblock_enabled(False),
        requested_schedule=Schedule(),
    )


def _operand(operand, seed):
    generator = torch.Generator().manual_seed(seed)
    shape = SHAPES[len(operand)]
    values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5
    keep = torch.rand(shape, generator=generator) < 0.55
    return (values * keep).to(torch.float32)


def _sparse(dense, fmt):
    """The operand in ``fmt``, built level by level from its own nonzeros.

    Not through ``to_sparse``: a conversion into a layout with a dense level
    below a compressed one (``sdss``, ``dsd``, ...) or a compressed level below
    two dense ones (``dds``) runs a shipped assembly kernel with its own known
    defects, and this sweep is about reductions.  Building the levels directly
    keeps a conversion defect from failing a reduction cell.
    """

    nonzeros = {tuple(int(t) for t in index) for index in dense.nonzero()}
    mode_indices, parents = [], [()]
    for level, kind in enumerate(fmt):
        if kind == "d":
            mode_indices.append([])
            parents = [p + (c,) for p in parents for c in range(dense.shape[level])]
            continue
        positions, coordinates, children = [0], [], []
        for parent in parents:
            stored = sorted({c[level] for c in nonzeros if c[:level] == parent})
            coordinates += stored
            children += [parent + (c,) for c in stored]
            positions.append(len(coordinates))
        mode_indices.append(
            [
                torch.tensor(positions, dtype=torch.int64),
                torch.tensor(coordinates, dtype=torch.int64),
            ]
        )
        parents = children
    values = torch.tensor([float(dense[c]) for c in parents], dtype=dense.dtype)
    return STensor.from_components(
        tuple(dense.shape), fmt, mode_indices, values, name="A"
    )


def _stored_coordinates(dense, fmt):
    """Every coordinate the operand's levels hold: a dense level holds every
    coordinate under each stored parent, a compressed one only the nonzeros'
    coordinates (``_sparse`` builds the levels the same way)."""

    nonzeros = {tuple(int(t) for t in index) for index in dense.nonzero()}
    parents = [()]
    for level, kind in enumerate(fmt):
        if kind == "d":
            parents = [p + (c,) for p in parents for c in range(dense.shape[level])]
        else:
            parents = [
                p + (c,)
                for p in parents
                for c in sorted({c[level] for c in nonzeros if c[:level] == p})
            ]
    return parents


def _implied_levels(operand, result, operand_fmt, result_fmt, dense):
    """The result's level arrays, in ``_storage``'s form, that the operand's
    stored coordinates imply: the stored result coordinates are the projections
    of the operand's stored coordinates onto the result indices."""

    projected = {
        tuple(coordinate[operand.index(c)] for c in result)
        for coordinate in _stored_coordinates(dense, operand_fmt)
    }
    shape = [dense.shape[operand.index(c)] for c in result]
    levels, parents = [], [()]
    for level, kind in enumerate(result_fmt):
        if kind == "d":
            levels.append(())
            parents = [p + (c,) for p in parents for c in range(shape[level])]
            continue
        positions, coordinates, children = [0], [], []
        for parent in parents:
            stored = sorted({c[level] for c in projected if c[:level] == parent})
            coordinates += stored
            children += [parent + (c,) for c in stored]
            positions.append(len(coordinates))
        levels.append((tuple(positions), tuple(coordinates)))
        parents = children
    return tuple(levels)


def _reduction_cin(operand, result, operand_fmt, result_fmt):
    ivars = {name: IndexVar(name) for name in operand}
    source = TensorVar("A", fmt=operand_fmt, dtype=torch.float32)[
        tuple(ivars[name] for name in operand)
    ]
    target = TensorVar("C", fmt=result_fmt, dtype=torch.float32)[
        tuple(ivars[name] for name in result)
    ]
    statement = TensorAssign(target, source, op=Operation.ADD)
    for name in reversed(operand):
        statement = ForAll(ivars[name], statement)
    return statement


def _storage(result):
    levels = tuple(
        tuple(tuple(int(x) for x in part.tolist()) for part in mode)
        for mode in result.storage.index.mode_indices
    )
    return levels, tuple(float(v) for v in result.storage.value.tolist())


def _cell_id(cell):
    return f"{cell[2]} {cell[0]}->{cell[1]} as {cell[3]}"


# Cells whose shipped result stores coordinates the operand does not imply,
# each with what the shipped compiler does instead.  Values are right in every
# one (each is held to the float64 reference); only the stored structure is
# wrong.  Strict: a listed cell whose shipped structure becomes the implied one
# fails until it leaves this table.
SHIPPED_STRUCTURE_DEFECTS: dict = {}


def test_the_defect_table_names_only_cells_of_this_sweep():
    assert set(SHIPPED_STRUCTURE_DEFECTS) <= {_cell_id(cell) for cell in CELLS}


@pytest.mark.parametrize("cell", CELLS, ids=[_cell_id(cell) for cell in CELLS])
def test_sparse_result_reduction_is_correct_on_both_compilers(cell):
    operand, result, operand_fmt, result_fmt = cell
    dense = _operand(operand, seed=len(CELLS) + CELLS.index(cell))
    expected = torch.einsum(f"{operand}->{result}", dense.to(torch.float64))

    shipped = scorch.einsum(
        f"{operand}->{result}",
        _sparse(dense, operand_fmt),
        format=result_fmt,
        _compile_options=_options(),
    )
    assert str(shipped.format) == ",".join(result_fmt)
    torch.testing.assert_close(
        shipped.to_torch(in_place=False).to(torch.float64),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )

    implied = _implied_levels(operand, result, operand_fmt, result_fmt, dense)
    shipped_known_wrong = _cell_id(cell) in SHIPPED_STRUCTURE_DEFECTS
    if shipped_known_wrong:
        assert _storage(shipped)[0] != implied, (
            "the shipped structure is the implied one now; remove the cell from "
            "SHIPPED_STRUCTURE_DEFECTS"
        )
    else:
        assert _storage(shipped)[0] == implied, "the shipped stored structure"

    result_shape = tuple(SHAPES[len(operand)][operand.index(c)] for c in result)
    try:
        loopir, _ = execute_cin_via_loopir(
            _reduction_cin(operand, result, operand_fmt, result_fmt),
            result_shape,
            _sparse(dense, operand_fmt),
            compile_options=_options(),
        )
    except (LoopIRLoweringError, LoopIRTargetError) as refusal:
        assert type(refusal.defect.code) is str and refusal.defect.code
        return
    torch.testing.assert_close(
        loopir.to_torch(in_place=False).to(torch.float64),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )
    loopir_levels, loopir_values = _storage(loopir)
    assert loopir_levels == implied, "the new compiler's stored structure"
    if shipped_known_wrong:
        return
    # The values are sums of the same float32 terms, but the two compilers
    # need not add them in the same order (a scalar accumulator against a
    # workspace), so they agree to float32 rounding rather than bit for bit;
    # each was already held to the float64 reference above.
    torch.testing.assert_close(
        torch.tensor(loopir_values, dtype=torch.float64),
        torch.tensor(_storage(shipped)[1], dtype=torch.float64),
        atol=0.0,
        rtol=1e-5,
    )
