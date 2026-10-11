"""Compressed result levels driven by dense domains, through the LoopIR route.

A compressed result level appended to by a loop over a dense range visits
every coordinate of that range, including coordinates the operands stored
nothing under.  The rule, which is Bobby's decision recorded in review §57 and
derived in ``DENSE_DOMAIN_ASSEMBLY_SEMANTICS.md``: the level appends a
coordinate exactly when the operands' stored structure under it is non-empty.
Structure, never value -- a stored zero keeps its key, and contributions that
cancel exactly keep theirs.

The target realizes that rule three ways, one per position of the level:

* a level with a compressed result level below it appends under its child's
  existing position counter (``C{l+1}_pos.back() < pC{l+1}``);
* a leaf level of an assembly appends every coordinate, because a dense
  domain there means every operand is dense at that level and dense storage
  holds every cell;
* a leaf level above a reduction appends when the reduction sub-nest ran at
  least once, recorded by one flag set in the sub-nest's body.

Two neighbouring normalizations ship with it and are locked here: an ADD update
with no reduction loop is plain assignment for a sparse result (it selects no
accumulation family), and keeps its operator for a dense one.

Every check below compares against a float64 PyTorch reference and against the
operand's own stored structure, computed independently from its storage
arrays, so a kernel that is numerically right but stores the wrong cells fails.
"""

from dataclasses import replace
from itertools import product

import pytest
import torch

from scorch.compiler.cin import BinaryOp as CINBinaryOp
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.lower_cin import (
    LoopIRLoweringError,
    lower_normalized_cin_to_loopir,
)
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.nodes import AppendEntry, Block, StoreReduce
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.scheduler import Schedule
from scorch.format import LevelType
from scorch.stensor import STensor

ATOL = RTOL = 1e-3
SHAPES = {2: (4, 5), 3: (3, 4, 5)}


def _options(regblock_enabled=False):
    return replace(
        CompileOptions.from_environment().with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(),
    )


def _formats(rank):
    return ["".join(levels) for levels in product("ds", repeat=rank)]


def _copy_cells():
    cells = []
    for rank in (2, 3):
        for operand_fmt in _formats(rank):
            for result_fmt in _formats(rank):
                if set(result_fmt) == {"d"}:
                    continue
                for op in (None, Operation.ADD):
                    cells.append((operand_fmt, result_fmt, op))
    return cells


COPY_CELLS = _copy_cells()


def _copy_cin(operand_fmt, result_fmt, op, dtype=torch.float32):
    names = "ijk"[: len(operand_fmt)]
    ivars = [IndexVar(name) for name in names]
    source = TensorVar("A", fmt=operand_fmt, dtype=dtype)[tuple(ivars)]
    target = TensorVar("C", fmt=result_fmt, dtype=dtype)[tuple(ivars)]
    stmt = TensorAssign(target, source, op=op)
    for ivar in reversed(ivars):
        stmt = ForAll(ivar, stmt)
    return stmt


def _reduction_cin(operand, result, operand_fmt, result_fmt, dtype=torch.float32):
    ivars = {name: IndexVar(name) for name in operand}
    source = TensorVar("A", fmt=operand_fmt, dtype=dtype)[
        tuple(ivars[name] for name in operand)
    ]
    target = TensorVar("C", fmt=result_fmt, dtype=dtype)[
        tuple(ivars[name] for name in result)
    ]
    stmt = TensorAssign(target, source, op=Operation.ADD)
    for name in reversed(operand):
        stmt = ForAll(ivars[name], stmt)
    return stmt


def _dense(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5
    keep = torch.rand(shape, generator=generator) < 0.45
    dense = values * keep
    # One fully empty leading slice, so a dense-driven level visits a cell
    # under which nothing is stored.
    dense[0] = 0.0
    return dense.to(torch.float32)


def _sparse(dense, fmt):
    tensor = STensor.from_torch(dense.clone(), "A")
    if set(fmt) != {"d"}:
        tensor.to_sparse(fmt)
    return tensor


def _stored_entries(tensor):
    """Every stored leaf entry, ``coords -> value``, read off the storage arrays.

    Independent of every compiler: dense levels materialize every child of a
    stored parent, compressed levels hold exactly their ``pos``/``crd``
    segments.  Explicit zeros are entries.
    """

    levels = tensor.format.level_types
    assert list(tensor.mode_order) == list(range(len(levels)))
    shape = tuple(tensor.shape)
    mode_indices = tensor.storage.index.mode_indices
    frontier = [((), 0)]
    for level, kind in enumerate(levels):
        extent = shape[level]
        grown = []
        if kind is LevelType.DENSE:
            for coords, parent in frontier:
                for coordinate in range(extent):
                    grown.append((coords + (coordinate,), parent * extent + coordinate))
        else:
            assert kind is LevelType.COMPRESSED
            pos, crd = (array.tolist() for array in mode_indices[level])
            assert len(pos) == len(frontier) + 1 or level == 0
            for coords, parent in frontier:
                for position in range(pos[parent], pos[parent + 1]):
                    grown.append((coords + (int(crd[position]),), position))
        frontier = grown
    values = tensor.storage.value.tolist()
    assert len(values) == len(frontier)
    return {coords: float(values[position]) for coords, position in frontier}


def _expected_structure(entries, result_fmt, result_shape):
    """The cells a result of ``result_fmt`` must store, from operand entries.

    ``entries`` are operand entries already projected onto the result's
    coordinates.  A cell is stored exactly when its prefix through the deepest
    compressed result level is a prefix of some entry: a compressed level holds
    a coordinate iff something is stored under it, and a dense level holds every
    child of a stored parent.  With no compressed level every cell is stored.
    """

    compressed = [level for level, char in enumerate(result_fmt) if char == "s"]
    cells = set(product(*(range(extent) for extent in result_shape)))
    if not compressed:
        return cells
    depth = compressed[-1] + 1
    prefixes = {coords[:depth] for coords in entries}
    return {cell for cell in cells if cell[:depth] in prefixes}


def _assert_no_empty_segments(tensor):
    """No compressed non-leaf coordinate is stored with nothing below it."""

    levels = tensor.format.level_types
    mode_indices = tensor.storage.index.mode_indices
    for level in range(len(levels) - 1):
        if levels[level] is not LevelType.COMPRESSED:
            continue
        child = level + 1
        if levels[child] is not LevelType.COMPRESSED:
            continue
        child_pos = mode_indices[child][0].tolist()
        for position in range(len(child_pos) - 1):
            assert child_pos[position + 1] > child_pos[position], (
                f"level {level} stores a coordinate at position {position} "
                "with an empty child segment"
            )


def _refusal_code(error):
    return error.defect.code


@pytest.mark.parametrize(
    "cell",
    COPY_CELLS,
    ids=[f"{a}->{c}{'+=' if op else '='}" for a, c, op in COPY_CELLS],
)
def test_every_copy_is_correct_or_refused_with_a_code(cell):
    operand_fmt, result_fmt, op = cell
    shape = SHAPES[len(operand_fmt)]
    dense = _dense(shape, seed=COPY_CELLS.index(cell))
    operand = _sparse(dense, operand_fmt)
    try:
        result, _ = execute_cin_via_loopir(
            _copy_cin(operand_fmt, result_fmt, op),
            shape,
            operand,
            compile_options=_options(),
        )
    except (LoopIRLoweringError, LoopIRTargetError) as refusal:
        assert type(_refusal_code(refusal)) is str and _refusal_code(refusal)
        return
    torch.testing.assert_close(
        result.to_torch(in_place=False).to(torch.float64),
        dense.to(torch.float64),
        atol=ATOL,
        rtol=RTOL,
    )
    # Structure, never value: the copy stores exactly the cells the operand's
    # own stored structure implies for this result layout.
    assert set(_stored_entries(result)) == _expected_structure(
        _stored_entries(operand), result_fmt, shape
    )
    _assert_no_empty_segments(result)


# The copies this change admits, which must not regress to a refusal.  Each
# drives a compressed result level with a dense domain or a dense prefix level
# with a stored stream; ``+=`` and ``=`` must agree.
NEWLY_ADMITTED_COPIES = [
    ("ds", "ss"),
    ("sd", "ss"),
    ("dd", "ss"),
    ("dss", "sss"),
    ("ssd", "sss"),
    ("ddd", "sss"),
    ("dds", "dss"),
    ("sss", "dss"),
    ("sss", "dds"),
]


@pytest.mark.parametrize("operand_fmt,result_fmt", NEWLY_ADMITTED_COPIES)
@pytest.mark.parametrize("op", [None, Operation.ADD], ids=["assign", "add"])
@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["direct", "regblock"])
def test_dense_driven_copies_are_admitted_on_both_arms(
    operand_fmt, result_fmt, op, regblock_enabled
):
    shape = SHAPES[len(operand_fmt)]
    compile_cin_via_loopir(
        _copy_cin(operand_fmt, result_fmt, op),
        shape,
        ((shape, torch.float32),),
        compile_options=_options(regblock_enabled),
    )


@pytest.mark.parametrize(
    "result_fmt,operand_fmts,shape",
    [
        ("ss", ("dd", "dd"), (4, 5)),
        ("ss", ("dd", "d"), (4, 5)),
        ("dds", ("ddd", "ddd"), (3, 4, 5)),
        ("dds", ("dsd",), (3, 4, 5)),
    ],
    ids=["dd*dd->ss", "dd*d->ss", "ddd*ddd->dds", "dsd->dds"],
)
def test_the_moved_seam_cells_execute_correctly(result_fmt, operand_fmts, shape):
    """The cells three seam locks used to hold, executed rather than refused."""

    names = "ijk"[: len(shape)]
    ivars = [IndexVar(name) for name in names]
    operands, accesses = [], []
    for position, fmt in enumerate(operand_fmts):
        rank = len(fmt)
        operand_shape = shape[len(shape) - rank :]
        dense = _dense(operand_shape, seed=31 + position)
        if rank == len(shape):
            dense[0] = 0.0
        operands.append((dense, _sparse(dense, fmt)))
        tensor = TensorVar("AB"[position], fmt=fmt, dtype=torch.float32)
        accesses.append(tensor[tuple(ivars[len(shape) - rank :])])
    rhs = accesses[0]
    for access in accesses[1:]:
        rhs = CINBinaryOp(Operation.MUL, rhs, access)
    stmt = TensorAssign(
        TensorVar("C", fmt=result_fmt, dtype=torch.float32)[tuple(ivars)], rhs
    )
    for ivar in reversed(ivars):
        stmt = ForAll(ivar, stmt)
    result, _ = execute_cin_via_loopir(
        stmt, shape, *(tensor for _, tensor in operands), compile_options=_options()
    )
    reference = operands[0][0].to(torch.float64)
    for dense, _ in operands[1:]:
        reference = reference * dense.to(torch.float64)
    torch.testing.assert_close(
        result.to_torch(in_place=False).to(torch.float64),
        reference,
        atol=ATOL,
        rtol=RTOL,
    )
    # An intersection with a dense operand is the sparse operand's structure;
    # dense-only products store every cell.
    entries = set(product(*(range(extent) for extent in shape)))
    for _, tensor in operands:
        if len(tensor.shape) == len(shape):
            entries &= set(_stored_entries(tensor))
    assert set(_stored_entries(result)) == _expected_structure(
        entries, result_fmt, shape
    )
    _assert_no_empty_segments(result)


def test_an_all_dense_driven_copy_closes_its_root_segment():
    """The first compressed level's root position is written after the nest.

    A dense loop over a compressed first level owns no catch-up of its own, so
    the root close has to be emitted after the whole nest exactly as it is for
    a stored driver.  Without it ``C0_pos`` ends at zero and the result reads as
    empty -- the defect the first draft of this change shipped and its own
    generation-time checks accepted.
    """

    dense = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    result, kernel = execute_cin_via_loopir(
        _copy_cin("dd", "ss", None),
        (3, 3),
        _sparse(dense, "dd"),
        compile_options=_options(),
    )
    assert "scorch_vector_set(C0_pos, C0_pos_index + 1, C0_crd.size());" in (
        kernel.cpp_source
    )
    level0 = [array.tolist() for array in result.storage.index.mode_indices[0]]
    # Every row of a dense operand is stored structure, zeros included.
    assert level0 == [[0, 3], [0, 1, 2]]
    assert result.storage.value.numel() == 9
    torch.testing.assert_close(result.to_torch(in_place=False), dense)


def _explicit_zero_ds_operand():
    """A ``ds`` operand with an empty row, a stored-zero row, and a full row."""

    # Row 0: nothing stored.  Row 1: one explicit stored zero.  Row 2: two
    # entries that cancel exactly.  Row 3: one ordinary entry.
    pos = torch.tensor([0, 0, 1, 3, 4], dtype=torch.int32)
    crd = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    values = torch.tensor([0.0, 1.5, -1.5, 2.0], dtype=torch.float32)
    return STensor.from_components((4, 4), "ds", [[], [pos, crd]], values, name="A")


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["direct", "regblock"])
def test_a_dense_driven_reduction_keys_rows_by_structure(regblock_enabled):
    operand = _explicit_zero_ds_operand()
    result, kernel = execute_cin_via_loopir(
        _reduction_cin("ij", "i", "ds", "s"),
        (4,),
        operand,
        compile_options=_options(regblock_enabled),
    )
    assert "bool C_nonempty = false;" in kernel.cpp_source
    assert "if (C_nonempty)" in kernel.cpp_source
    stored = [array.tolist() for array in result.storage.index.mode_indices[0]]
    # Row 0 stored nothing and gets no key; rows 1 and 2 keep theirs although
    # their sums are zero; row 3 is ordinary.
    assert stored == [[0, 3], [1, 2, 3]]
    assert result.storage.value.tolist() == pytest.approx([0.0, 0.0, 2.0])


def test_a_zero_extent_dense_level_below_a_stored_one_stores_nothing():
    """``dsd ijk->i`` with ``K == 0``: stored ``j`` entries hold no cell."""

    dense = torch.zeros((3, 4, 0), dtype=torch.float32)
    pos1 = torch.tensor([0, 2, 2, 3], dtype=torch.int32)
    crd1 = torch.tensor([0, 3, 1], dtype=torch.int32)
    operand = STensor.from_components(
        (3, 4, 0), "dsd", [[], [pos1, crd1], []], dense.reshape(-1), name="A"
    )
    result, _ = execute_cin_via_loopir(
        _reduction_cin("ijk", "i", "dsd", "s"),
        (3,),
        operand,
        compile_options=_options(),
    )
    stored = [array.tolist() for array in result.storage.index.mode_indices[0]]
    assert stored == [[0, 0], []]
    assert result.storage.value.numel() == 0


@pytest.mark.parametrize(
    "operand,result,operand_fmt,result_fmt",
    [
        ("ij", "i", "ds", "s"),
        ("ijk", "i", "dss", "s"),
        ("ijk", "i", "dds", "s"),
        ("ijk", "i", "dsd", "s"),
        ("ijk", "ij", "dds", "ss"),
        ("ijk", "ij", "sss", "ss"),
        ("ijk", "ik", "dss", "ss"),
        ("ijk", "ik", "dds", "ss"),
        ("ijk", "ik", "ssd", "ss"),
    ],
)
def test_dense_driven_reductions_store_the_projected_structure(
    operand, result, operand_fmt, result_fmt
):
    shape = SHAPES[len(operand)]
    dense = _dense(shape, seed=11)
    tensor = _sparse(dense, operand_fmt)
    result_shape = tuple(shape[operand.index(name)] for name in result)
    got, _ = execute_cin_via_loopir(
        _reduction_cin(operand, result, operand_fmt, result_fmt),
        result_shape,
        tensor,
        compile_options=_options(),
    )
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


def _leaf(program):
    body = program.body
    while True:
        assert type(body) is Block and len(body.statements) == 1
        only = body.statements[0]
        if type(only) in (AppendEntry, StoreReduce):
            return only
        body = only.body


@pytest.mark.parametrize(
    "result_fmt,leaf_type", [("ss", AppendEntry), ("dd", StoreReduce)]
)
def test_an_add_copy_is_assignment_only_where_the_operator_picks_a_family(
    result_fmt, leaf_type
):
    """``+=`` with no reduction loop: assembly for a sparse result, kept for dense.

    A sparse result's update operator selects the accumulation families, none of
    which owns a copy, so it is normalized away.  A dense result's ``+=`` is a
    direct store into a zeroed array either way and keeps the operator, which is
    the emission legacy parity is measured on.
    """

    program = lower_normalized_cin_to_loopir(
        normalize_cin(_copy_cin("ds", result_fmt, Operation.ADD))
    ).program
    assert type(_leaf(program)) is leaf_type
