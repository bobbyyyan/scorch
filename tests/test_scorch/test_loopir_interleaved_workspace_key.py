"""Workspace keys bound between two reductions, through the LoopIR route.

``C[j] += A[i, j, k]`` under the loop order ``i, j, k`` binds its result
coordinate between two reductions.  The new compiler used to refuse it: the
ordered-key split defined the key as the result coordinates bound below the
INNERMOST reduction, so a coordinate between two reductions belonged to
neither the prefix nor the key.  The shipped compiler emits these programs.

The key is now every result coordinate bound below the OUTERMOST reduction.
One sparse workspace region starts at that reduction and contains every loop
below it, so every contribution to a key is accumulated before the region's
drain appends it, and the drain sorts the keys itself -- the order in which
the producer visits keys does not matter.  The insertion stays at the
innermost loop and names its key coordinates by what binds them, not by
depth.

Every check executes against a float64 PyTorch reference and against the
operand's own stored structure (structure, never value: a key whose
contributions sum to zero, or are stored zeros, is still stored).
"""

from itertools import product

import pytest
import torch

from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.nodes import (
    Block,
    SparseWorkspaceInsert,
    SparseWorkspaceRegion,
)
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.stensor import STensor
from tests.test_scorch.test_loopir_dense_domain_sparse_result import (
    _assert_no_empty_segments,
    _expected_structure,
    _options,
    _reduction_cin,
    _stored_entries,
)

ATOL = RTOL = 1e-3
SHAPES = {3: (3, 4, 5), 4: (3, 4, 3, 5)}


def _is_interleaved(operand, result):
    """Some result coordinate sits between the outermost and innermost reductions."""

    reductions = [position for position, name in enumerate(operand) if name not in result]
    if not reductions:
        return False
    return any(
        min(reductions) < operand.index(name) < max(reductions) for name in result
    )


def _interleaved_cells():
    cells = []
    for rank in (3, 4):
        operand = "ijkl"[:rank]
        for size in (1, 2):
            for result in ("".join(c) for c in product(operand, repeat=size)):
                if sorted(result) != list(result) or len(set(result)) != size:
                    continue
                if not _is_interleaved(operand, result):
                    continue
                for operand_fmt in ("".join(f) for f in product("ds", repeat=rank)):
                    for result_fmt in ("".join(f) for f in product("ds", repeat=size)):
                        if set(result_fmt) == {"d"}:
                            continue
                        cells.append((operand, result, operand_fmt, result_fmt))
    return cells


INTERLEAVED_CELLS = _interleaved_cells()

# The programs this change admits: the 1138-program frontier's interleaved-key
# programs, all refused by ``_ordered_key_split`` or by the sparse workspace
# pass before 2026-10-11.  They must not regress to a refusal on either arm.
NEWLY_ADMITTED = [
    ("ijk", "j", operand_fmt, "s")
    for operand_fmt in ("dds", "dsd", "dss", "sds", "ssd", "sss")
] + [
    ("ijkl", result, operand_fmt, "s" * len(result))
    for operand_fmt in ("ddss", "dsss", "sdss", "ssss")
    for result in ("j", "k", "ik", "jk", "jl")
]


def _dense(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5
    keep = torch.rand(shape, generator=generator) < 0.4
    dense = values * keep
    # One fully empty leading slice, so some prefix stores nothing.
    dense[0] = 0.0
    return dense.to(torch.float32)


def _sparse(dense, fmt):
    tensor = STensor.from_torch(dense.clone(), "A")
    if set(fmt) != {"d"}:
        tensor.to_sparse(fmt)
    return tensor


def _cell_id(cell):
    operand, result, operand_fmt, result_fmt = cell
    return f"{operand_fmt} {operand}->{result} [{result_fmt}]"


def _check_against_reference(cell, tensor, dense, got):
    operand, result, _, result_fmt = cell
    shape = tuple(dense.shape)
    result_shape = tuple(shape[operand.index(name)] for name in result)
    reference = torch.einsum(f"{operand}->{result}", dense.to(torch.float64))
    torch.testing.assert_close(
        got.to_torch(in_place=False).to(torch.float64),
        reference,
        atol=ATOL,
        rtol=RTOL,
    )
    keep = [operand.index(name) for name in result]
    projected = {
        tuple(coords[axis] for axis in keep) for coords in _stored_entries(tensor)
    }
    assert set(_stored_entries(got)) == _expected_structure(
        projected, result_fmt, result_shape
    )
    _assert_no_empty_segments(got)


@pytest.mark.parametrize("cell", INTERLEAVED_CELLS, ids=_cell_id)
def test_every_interleaved_reduction_is_correct_or_refused_with_a_code(cell):
    operand, result, operand_fmt, result_fmt = cell
    shape = SHAPES[len(operand)]
    dense = _dense(shape, seed=INTERLEAVED_CELLS.index(cell))
    tensor = _sparse(dense, operand_fmt)
    result_shape = tuple(shape[operand.index(name)] for name in result)
    try:
        got, _ = execute_cin_via_loopir(
            _reduction_cin(operand, result, operand_fmt, result_fmt),
            result_shape,
            tensor,
            compile_options=_options(),
        )
    except (LoopIRLoweringError, LoopIRTargetError) as refusal:
        assert type(refusal.defect.code) is str and refusal.defect.code
        return
    _check_against_reference(cell, tensor, dense, got)


@pytest.mark.parametrize("cell", NEWLY_ADMITTED, ids=_cell_id)
def test_the_newly_admitted_programs_execute_correctly(cell):
    operand, result, operand_fmt, result_fmt = cell
    shape = SHAPES[len(operand)]
    dense = _dense(shape, seed=101 + NEWLY_ADMITTED.index(cell))
    tensor = _sparse(dense, operand_fmt)
    result_shape = tuple(shape[operand.index(name)] for name in result)
    got, _ = execute_cin_via_loopir(
        _reduction_cin(operand, result, operand_fmt, result_fmt),
        result_shape,
        tensor,
        compile_options=_options(),
    )
    _check_against_reference(cell, tensor, dense, got)


@pytest.mark.parametrize("cell", NEWLY_ADMITTED, ids=_cell_id)
def test_the_newly_admitted_programs_are_arm_source_identical(cell):
    operand, result, operand_fmt, result_fmt = cell
    shape = SHAPES[len(operand)]
    result_shape = tuple(shape[operand.index(name)] for name in result)
    sources = {
        arm: compile_cin_via_loopir(
            _reduction_cin(operand, result, operand_fmt, result_fmt),
            result_shape,
            ((shape, torch.float32),),
            compile_options=_options(arm),
        ).cpp_source
        for arm in (False, True)
    }
    assert sources[False] == sources[True]


def _region_and_insert(program):
    """The one workspace region, the loops above it, and the insertion."""

    body, above = program.body, []
    while True:
        assert type(body) is Block and len(body.statements) == 1
        only = body.statements[0]
        if type(only) is SparseWorkspaceRegion:
            region = only
            break
        above.append(only)
        body = only.body
    producer, inside = region.producer, []
    while True:
        assert type(producer) is Block and len(producer.statements) == 1
        only = producer.statements[0]
        if type(only) is SparseWorkspaceInsert:
            return above, region, inside, only
        inside.append(only)
        producer = only.body


def _bound_index(loop):
    return loop.coord_index if hasattr(loop, "coord_index") else loop.index


@pytest.mark.parametrize(
    "operand,result,operand_fmt,prefix",
    [
        ("ijk", "j", "sss", 0),
        ("ijk", "j", "dds", 0),
        ("ijkl", "ik", "dsss", 1),
        ("ijkl", "jl", "ssss", 0),
    ],
)
def test_the_region_starts_at_the_outermost_reduction(
    operand, result, operand_fmt, prefix
):
    """Structure of the region, read off the LoopIR program, not the C++ text.

    The loops above the region bind exactly the result prefix; the region's
    first loop is a reduction; the insertion is at the innermost loop and its
    coordinates are the key loops' own indices, at least one of which is bound
    above some reduction loop inside the region.
    """

    shape = SHAPES[len(operand)]
    result_shape = tuple(shape[operand.index(name)] for name in result)
    kernel = compile_cin_via_loopir(
        _reduction_cin(operand, result, operand_fmt, "s" * len(result)),
        result_shape,
        ((shape, torch.float32),),
        compile_options=_options(),
    )
    above, _, inside, insert = _region_and_insert(kernel.schedule.program)
    assert len(above) == prefix
    assert len(above) + len(inside) == len(operand)
    inside_indices = [_bound_index(loop) for loop in inside]
    key_indices = [coord.index for coord in insert.coords]
    assert len(key_indices) == len(result) - prefix
    assert inside_indices[0] not in key_indices
    key_positions = [inside_indices.index(index) for index in key_indices]
    assert key_positions == sorted(key_positions)
    assert key_positions[0] < len(inside) - 1
    assert kernel.cpp_source.count("wksp.sort();") == 1


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["direct", "regblock"])
def test_a_key_is_stored_by_structure_not_by_value(regblock_enabled):
    """``sss ijk->j``: a key whose contributions cancel, or are stored zeros, stays.

    ``j = 0`` gets ``+1`` under ``i = 0`` and ``-1`` under ``i = 1``; ``j = 2`` gets
    one stored zero; ``j = 1`` and ``j = 3`` are never stored.
    """

    shape = (2, 4, 2)
    pos0 = torch.tensor([0, 2], dtype=torch.int32)
    crd0 = torch.tensor([0, 1], dtype=torch.int32)
    pos1 = torch.tensor([0, 2, 3], dtype=torch.int32)
    crd1 = torch.tensor([0, 2, 0], dtype=torch.int32)
    pos2 = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    crd2 = torch.tensor([1, 0, 1], dtype=torch.int32)
    values = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)
    operand = STensor.from_components(
        shape, "sss", [[pos0, crd0], [pos1, crd1], [pos2, crd2]], values, name="A"
    )
    got, kernel = execute_cin_via_loopir(
        _reduction_cin("ijk", "j", "sss", "s"),
        (4,),
        operand,
        compile_options=_options(regblock_enabled),
    )
    assert "wksp.insert({j}" in kernel.cpp_source
    stored = [array.tolist() for array in got.storage.index.mode_indices[0]]
    assert stored == [[0, 2], [0, 2]]
    assert got.storage.value.tolist() == pytest.approx([0.0, 0.0])


def test_the_sweep_reaches_every_newly_admitted_program():
    """The sweep is a superset of the admitted list, not a separate population."""

    assert set(NEWLY_ADMITTED) <= set(INTERLEAVED_CELLS)
    assert len(NEWLY_ADMITTED) == 26
    assert all(_is_interleaved(cell[0], cell[1]) for cell in INTERLEAVED_CELLS)
    assert not _is_interleaved("ijk", "k") and not _is_interleaved("ijkl", "ij")
