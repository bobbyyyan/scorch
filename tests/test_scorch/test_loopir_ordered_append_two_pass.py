"""Two-pass assembly of results the new compiler appends in loop order.

A result stored ``ds`` or ``dss`` whose entries are each one reduction, appended
in loop order with no workspace, is lowered by
``_MultiCompressedAssemblyLowering``.  The shipped compiler assembles the same
programs in two parallel passes around a workspace.  The family now offers the
two-pass strategies as well: the shared two-phase pass rewrites its serial nest
into a count loop and a fill loop, with result segments numbered by the outer
loop's coordinate, so a loop over the stored entries of an operand's compressed
level leaves the coordinates it skips with a count of zero.

What is held here:

* the automatic plan records ``two_pass_parallel`` from ``2 * ROWS_PER_THREAD``
  rows and the serial single pass below, where the kernel text is the one the
  family emitted before;
* each two-pass kernel stores exactly the arrays the serial single pass stores,
  and the values of a dense float64 reference, on operands with empty leading,
  interior and trailing slices and at an extent where several threads run;
* a program the strategy cannot assemble is refused it with a code.
"""

import re
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
from scorch.compiler.loopir import lower_llir as lower_llir_module
from scorch.compiler.loopir.parallel_chunk_assembly import ROWS_PER_THREAD
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.scheduler import Schedule
from scorch.stensor import STensor

_F64 = torch.float64
SERIAL = "single_pass_serial"
TWO_PASS = ("two_pass_serial", "two_pass_parallel")
HOST = "_MultiCompressedAssemblyLowering"

#: Extents by index letter.  The first result index gets the row count under
#: test; the others stay small so a rank-4 dense operand stays small.
EXTENT = {"i": 8, "j": 6, "k": 7, "l": 5}

#: (label, operands as (name, format, indices), result indices, result format,
#: the kind of the outermost loop).
CASES = [
    # One compressed result level; reductions one and two loops deep.
    ("dense outer, ds", (("A", "dd", "ij"), ("B", "ds", "kj")), "ik", "ds", "dense"),
    ("dense outer, ds, own sum", (("A", "dss", "ijk"),), "ij", "ds", "dense"),
    (
        "dense outer, ds, vector",
        (("A", "dds", "ijk"), ("B", "s", "k")),
        "ij",
        "ds",
        "dense",
    ),
    ("stored outer, ds", (("A", "sd", "ij"), ("B", "ds", "kj")), "ik", "ds", "stored"),
    (
        "stored outer, ds, three operands",
        (("A", "ss", "ij"), ("B", "ds", "ik"), ("D", "ds", "jk")),
        "ij",
        "ds",
        "stored",
    ),
    # Two compressed result levels.
    (
        "dense outer, dss",
        (("A", "dddd", "ijkl"), ("B", "s", "l")),
        "ijk",
        "dss",
        "dense",
    ),
    (
        "stored outer, dss",
        (("A", "sddd", "ijkl"), ("B", "s", "l")),
        "ijk",
        "dss",
        "stored",
    ),
    # A workspace region below every result level but the last.
    (
        "dense outer, dss, region",
        (("A", "dsd", "ijk"), ("B", "dss", "ijl")),
        "ijk",
        "dss",
        "dense",
    ),
    (
        "stored outer, dss, region",
        (("A", "ssd", "ijk"), ("B", "dsd", "ijl")),
        "ijk",
        "dss",
        "stored",
    ),
]
CASE_IDS = [case[0] for case in CASES]

#: A workspace region that owns the whole nest: the first top-level loop is a
#: producer loop, not the loop that assembles the result's rows.
ROOT_REGION = ((("A", "sd", "ij"), ("B", "dd", "ik")), "kj", "ds")


def program(operands, result_indices, result_format):
    """``C[result] += product of the operands``."""

    letters = "".join(dict.fromkeys("".join(indices for _, _, indices in operands)))
    variables = {letter: IndexVar(letter) for letter in letters}
    product = None
    for name, fmt, indices in operands:
        access = TensorVar(name, fmt=fmt, dtype=_F64)[
            tuple(variables[letter] for letter in indices)
        ]
        product = (
            access if product is None else CINBinaryOp(Operation.MUL, product, access)
        )
    result = TensorVar("C", fmt=result_format, dtype=_F64)[
        tuple(variables[letter] for letter in result_indices)
    ]
    statement = TensorAssign(result, product, op=Operation.ADD)
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    return statement


def extents(result_indices, rows):
    return {**EXTENT, result_indices[0]: rows}


def shapes(operands, result_indices, rows):
    extent = extents(result_indices, rows)
    result_shape = tuple(extent[letter] for letter in result_indices)
    bindings = tuple(
        (tuple(extent[letter] for letter in indices), _F64)
        for _, _, indices in operands
    )
    return result_shape, bindings


def options(assembly=None, regblock_enabled=False, *, jit=False):
    base = (
        CompileOptions.from_environment()
        if jit
        else CompileOptions.from_environment(environ={})
    )
    return replace(
        base.with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(assembly=assembly),
    )


def compiled(case, rows, assembly=None, regblock_enabled=False):
    _, operands, result_indices, result_format, _ = case
    result_shape, bindings = shapes(operands, result_indices, rows)
    return compile_cin_via_loopir(
        program(operands, result_indices, result_format),
        result_shape,
        bindings,
        compile_options=options(assembly, regblock_enabled),
    )


def pragmas(source):
    body = source[source.rindex(" evaluate(") :]
    return [line.strip() for line in body.splitlines() if "#pragma omp" in line]


# -- what is recorded and emitted ------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_the_family_offers_the_two_pass_pair(case, regblock_enabled):
    kernel = compiled(case, 2 * ROWS_PER_THREAD, None, regblock_enabled)
    decision = kernel.assembly_decision
    assert decision.host == HOST
    assert decision.emittable == (SERIAL, *TWO_PASS)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_the_automatic_plan_is_parallel_from_32_rows(case, regblock_enabled):
    below = compiled(case, 2 * ROWS_PER_THREAD - 1, None, regblock_enabled)
    assert below.assembly_decision.strategy == SERIAL
    assert pragmas(below.cpp_source) == []
    assert (
        below.cpp_source
        == compiled(case, 2 * ROWS_PER_THREAD - 1, SERIAL, regblock_enabled).cpp_source
    )
    at = compiled(case, 2 * ROWS_PER_THREAD, None, regblock_enabled)
    assert at.assembly_decision.strategy == "two_pass_parallel"
    assert (
        at.cpp_source
        == compiled(
            case, 2 * ROWS_PER_THREAD, "two_pass_parallel", regblock_enabled
        ).cpp_source
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_two_passes_are_a_count_loop_and_a_fill_loop(case):
    source = compiled(case, 64, "two_pass_parallel").cpp_source
    found = pragmas(source)
    assert len(found) == 2 and found[0] == found[1]
    assert found[0].startswith("#pragma omp parallel for num_threads(scorch_nthreads(")
    serial = compiled(case, 64, "two_pass_serial").cpp_source
    assert pragmas(serial) == []
    for text in (source, serial):
        # Positions come from the prefix sums, so no pass keeps a growing vector,
        # a running position index or a catch-up loop.
        assert "C1_pos_index" not in text
        assert "std::vector<int> C1_crd" not in text
        assert "_offset1[_i + 1] = _offset1[_i] + _count1[_i];" in text


@pytest.mark.parametrize(
    "case", [case for case in CASES if case[4] == "stored"], ids=lambda case: case[0]
)
def test_a_stored_outer_loop_numbers_rows_by_coordinate(case):
    """The loop variable is a position into the operand's coordinate array.

    A count array indexed by it would put every stored row's entries at the top
    of the result and leave the rows the operand does not store out of place.
    """

    source = compiled(case, 64, "two_pass_parallel").cpp_source
    assert "_count1[A0_crd[pA0]] = _cnt1;" in source
    assert "int64_t _base1 = _offset1[A0_crd[pA0]];" in source
    # Sized by the extent of the result's first level (whichever operand or
    # result variable carries it), never by the loop's own bound.
    sized = re.search(r"std::vector<int> _count1\(\(size_t\)(\w+), 0\);", source)
    assert sized is not None and re.fullmatch(r"[A-Z]\d_size", sized.group(1))
    assert "pA0_end" not in sized.group(1)


@pytest.mark.parametrize("assembly", TWO_PASS)
def test_a_region_that_owns_the_nest_is_refused_the_two_pass(assembly):
    operands, result_indices, result_format = ROOT_REGION
    case = ("root region", operands, result_indices, result_format, "stored")
    kernel = compiled(case, 64)
    assert kernel.assembly_decision.emittable == (SERIAL,)
    assert kernel.assembly_decision.strategy == SERIAL
    with pytest.raises(lower_llir_module.LoopIRTargetError) as refused:
        compiled(case, 64, assembly)
    assert refused.value.defect.code == "unsupported_assembly_strategy"


def test_a_coordinate_result_stays_serial():
    case = ("coo", (("A", "dd", "ij"), ("B", "dd", "ik")), "ij", "oo", "dense")
    kernel = compiled(case, 64)
    assert kernel.assembly_decision.host == "_CoordinateResultAssemblyLowering"
    assert kernel.assembly_decision.emittable == (SERIAL,)


# -- execution --------------------------------------------------------------------

#: Enough rows for ``scorch_nthreads(-1, rows)`` to return four.
ROWS = 4 * ROWS_PER_THREAD + 6


@pytest.fixture(autouse=True)
def _torch_allows_the_teams_these_shapes_were_built_for(torch_threads):
    # ROWS was built for scorch_nthreads(-1, rows) to return ROWS // ROWS_PER_THREAD = 4.
    # torch's count must allow that, or a process started with MKL_NUM_THREADS=1 runs every kernel
    # here on one worker (tests/conftest.py, torch_threads).
    torch_threads(ROWS // ROWS_PER_THREAD, needs_team=False)


#: Slices of the first result index every operand that has it leaves empty: the
#: first two, one alone, a run, and the last two.
EMPTY = (0, 1, 9, 30, 31, 32, 33, ROWS - 2, ROWS - 1)


def dense_operand(shape, seed, indices, first, density):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=_F64) + 0.5
    keep = torch.rand(shape, generator=generator, dtype=_F64) < density
    dense = values * keep
    if first in indices:
        index = [slice(None)] * len(shape)
        index[indices.index(first)] = list(EMPTY)
        dense[tuple(index)] = 0.0
    return dense


def storage(result):
    return (
        [[part.clone() for part in mode] for mode in result.storage.index.mode_indices],
        result.storage.value.clone(),
    )


def same_arrays(left, right):
    left_index, left_values = left
    right_index, right_values = right
    assert len(left_index) == len(right_index)
    for left_mode, right_mode in zip(left_index, right_index):
        assert len(left_mode) == len(right_mode)
        for left_part, right_part in zip(left_mode, right_mode):
            assert left_part.dtype == right_part.dtype
            assert torch.equal(left_part, right_part)
    assert left_values.dtype == right_values.dtype
    assert torch.allclose(left_values, right_values, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("density", [0.15, 0.6])
def test_each_strategy_stores_what_the_serial_single_pass_stores(case, density):
    _, operands, result_indices, result_format, _ = case
    extent = extents(result_indices, ROWS)
    first = result_indices[0]
    dense = [
        dense_operand(
            tuple(extent[letter] for letter in indices),
            20261012 + position,
            indices,
            first,
            1.0 if set(fmt) == {"d"} and first not in indices else density,
        )
        for position, (_, fmt, indices) in enumerate(operands)
    ]
    tensors = [
        STensor.from_torch(values.clone(), name).to_sparse(fmt)
        for values, (name, fmt, _) in zip(dense, operands)
    ]
    spec = ",".join(indices for _, _, indices in operands) + "->" + result_indices
    reference = torch.einsum(spec, *dense)
    result_shape = tuple(extent[letter] for letter in result_indices)

    def run(assembly):
        return execute_cin_via_loopir(
            program(operands, result_indices, result_format),
            result_shape,
            *tensors,
            compile_options=options(assembly, jit=True),
        )[0]

    serial = run(SERIAL)
    assert torch.allclose(
        serial.to_torch(in_place=False), reference, rtol=1e-12, atol=1e-12
    )
    stored_rows = serial.storage.index.mode_indices[1][0]
    assert int(stored_rows[-1]) > 0
    if any(first in indices and set(fmt) != {"d"} for _, fmt, indices in operands):
        # An operand with a compressed level stores nothing in its empty slices,
        # so those rows of the result are empty too.  (A dense operand's zeros
        # are visited, and the result stores the zeros computed from them.)
        for row in EMPTY:
            assert int(stored_rows[row]) == int(stored_rows[row + 1])
    expected = storage(serial)
    for assembly in (*TWO_PASS, None):
        result = run(assembly)
        same_arrays(storage(result), expected)
        assert torch.allclose(
            result.to_torch(in_place=False), reference, rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_operands_with_no_entries_give_the_empty_result(case):
    _, operands, result_indices, result_format, _ = case
    extent = extents(result_indices, ROWS)
    tensors = [
        STensor.from_torch(
            torch.zeros(tuple(extent[letter] for letter in indices), dtype=_F64), name
        ).to_sparse(fmt)
        for name, fmt, indices in operands
    ]
    result_shape = tuple(extent[letter] for letter in result_indices)
    for assembly in (*TWO_PASS, None):
        result = execute_cin_via_loopir(
            program(operands, result_indices, result_format),
            result_shape,
            *tensors,
            compile_options=options(assembly, jit=True),
        )[0]
        if all(set(fmt) == {"d"} for _, fmt, _ in operands):
            continue
        assert int(result.storage.value.numel()) == 0
        assert result.storage.index.mode_indices[1][0].tolist() == [0] * (ROWS + 1)
