"""A COO result that stores one input's stored entries, on request (2026-10-13).

A reduction into a COO result, ``C[i, j] = sum_k A[i, j] * B[i, k]`` with a COO
or compressed ``A`` and a ``B`` dense along ``i`` and ``j``, can store one result
entry per stored entry of ``A``: the loops over the result's indices walk ``A``'s
stored entries and the reduction runs below each.  The shipped compiler builds
such a result under scalar accumulation: at ``A``'s entry count, each entry
written at ``A``'s position (``_known_nnz``), an empty inner sum storing zero.
The new compiler appends to growing vectors and stores an entry only if its
inner sum ran at least once; on ``oo,dd ij,ik->ij`` its kernel took 8.8x the
shipped kernel's time (M5, 2026-10-12, same loop order).

The two stored structures have the same values, and which is faster is a
measurement, so it is a schedule decision: ``Schedule.entries_of`` names the
input whose stored entries the result stores, recorded as
``LoopPlan.entries_of`` and ``LoopProgram.entries_of``.  With no request each
compiler emits what it emitted before the field existed (mlsys-submission's
review of 2026-10-13 04:12: the change of default emission needs the runtime
grid, so it goes behind a request).

Checked, for every program below and both register-blocking settings:

* with no request, the new compiler names no input and keeps the append, the
  "the sum ran" flag and a serial loop, at small and large result sizes, and
  stores exactly the entries some term reaches;
* with the driving input named, the program records it; the kernel sizes the
  result by that input's entry count, writes by position, appends nothing and
  keeps no flag; the result stores exactly the driving input's stored entries,
  in order, entries whose inner sum is empty and entries the input stores as
  zero included, with ``torch.einsum``'s values; the reference interpreter
  stores the same; the shipped compiler emits its own text unchanged, and the
  shipped kernel stores the same entries as the new one;
* naming any other tensor is refused with a code at the boundary that can
  tell (the schedule, plan legality, the front end, the shipped lowering, the
  verifier or the target);
* the parallel single pass of a COO result needs the driving input named and
  a COO input of two levels or more; it stores the same arrays as the serial
  kernel at 1.6e5 entries on 8 threads.
"""

import dataclasses
import math
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import pytest
import torch

import scorch
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import LoopIRTargetError, assembly_choices
from scorch.compiler.loopir.nodes import LevelKind, LoopProgram
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.compiler.loopir.plan_identity import canonical_plan_dump
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.schedule_passes import erase_schedule
from scorch.compiler.loopir.verifier import LoopIRVerificationError, verify_program
from scorch.compiler.diagnostics import InvalidSchedule, UnsupportedFeature
from scorch.compiler.loop_plan import (
    LoopPlacement,
    LoopRef,
    LoopTile,
    PlacementKind,
    _analyze_loop_plan_cin,
    verify_loop_plan,
)
from scorch.compiler.loop_plan_legality import (
    _apply_affine_placements,
    _build_facts,
    _verify_assembly,
    _verify_entries_of,
)
from scorch.compiler.scheduler import Schedule, Scheduler
from tests.test_scorch.test_loopir_coordinate_levels import (
    coordinate_entries,
    label,
    operand,
    options,
    shipped_text,
)
from tests.test_scorch.test_loopir_operand_conversions import EXTENT, stored_cells

Operand = Tuple[str, str, str]
Case = Tuple[List[Operand], str, str]

# The driving input is listed first.  The cause table's COO-result programs
# (``~/.cache/scorch-codex/new-vs-shipped-runtime-2026-10-12/m5e_t``) and the
# shapes around them: COO, compressed, CSR, row-compressed and dense drivers;
# dense and sparse reduction operands, and two of them merged.
DRIVEN: List[Case] = [
    ([("A", "oo", "ij"), ("B", "dd", "ik")], "ij", "oo"),
    ([("A", "oo", "ij"), ("B", "dd", "ik"), ("D", "dd", "jk")], "ij", "oo"),
    ([("A", "oo", "ij"), ("B", "ds", "ik")], "ij", "oo"),
    ([("A", "oo", "ij"), ("B", "ds", "ik"), ("D", "ds", "jk")], "ij", "oo"),
    ([("A", "ss", "ij"), ("B", "dd", "ik"), ("D", "dd", "jk")], "ij", "oo"),
    ([("A", "ds", "ij"), ("B", "dd", "ik"), ("D", "ds", "jk")], "ij", "oo"),
    ([("A", "sd", "ij"), ("B", "dd", "ik"), ("D", "ds", "jk")], "ij", "oo"),
    ([("A", "dd", "ij"), ("B", "dd", "ik"), ("D", "ds", "jk")], "ij", "oo"),
    ([("A", "dds", "ijk"), ("B", "ddd", "ijl")], "ijk", "ooo"),
    ([("A", "sds", "ijk"), ("B", "ddd", "ijl")], "ijk", "ooo"),
    ([("A", "ooo", "ijk"), ("B", "dd", "kl")], "ijk", "ooo"),
    ([("A", "o", "i"), ("B", "ds", "ij")], "i", "o"),
]
DRIVEN_IDS = [label(case) for case in DRIVEN]

# No single input drives these, or nothing is reduced: each keeps the append,
# and naming an input is refused.
APPENDED: List[Case] = [
    ([("A", "ss", "ij"), ("B", "ss", "ik")], "ij", "oo"),
    ([("A", "oo", "ij"), ("B", "dd", "jk")], "ik", "oo"),
    ([("A", "ss", "ij"), ("B", "ss", "ij")], "ij", "oo"),
    ([("A", "oo", "ij")], "ij", "oo"),
]
APPENDED_IDS = [label(case) for case in APPENDED]

KINDS = ("random", "empty inner sums", "stored zeros", "empty driver")

# Every result of ``DRIVEN`` has 32 cells or more at these extents (the
# automatic rule's threshold for a parallel assembly), every reduction 3 terms
# or more.
LARGE_EXTENT = {"i": 40, "j": 9, "k": 3, "l": 3}


def program_of(operands, result_indices, result_format, dtype):
    """``C[result] (+)= product of operands`` in ``dtype``."""

    letters = "".join(dict.fromkeys("".join(indices for _, _, indices in operands)))
    variables = {letter: IndexVar(letter) for letter in letters}
    product = None
    for name, fmt, indices in operands:
        access = TensorVar(name, fmt=fmt, dtype=dtype)[
            tuple(variables[letter] for letter in indices)
        ]
        product = (
            access if product is None else CINBinaryOp(Operation.MUL, product, access)
        )
    result = TensorVar("C", fmt=result_format, dtype=dtype)[
        tuple(variables[letter] for letter in result_indices)
    ]
    reduced = any(letter not in result_indices for letter in letters)
    statement = TensorAssign(result, product, op=Operation.ADD if reduced else None)
    for letter in reversed(letters):
        statement = ForAll(variables[letter], statement)
    return statement


def shapes_of(operands, result_indices, dtype, extent=EXTENT):
    result_shape = tuple(extent[letter] for letter in result_indices)
    bindings = tuple(
        (tuple(extent[letter] for letter in indices), dtype)
        for _, _, indices in operands
    )
    return result_shape, bindings


def expression_of(operands, result_indices):
    return ",".join(indices for _, _, indices in operands) + "->" + result_indices


def cells_stored(keep: torch.Tensor, fmt: str) -> torch.Tensor:
    return keep.clone() if set(fmt) == {"o"} else stored_cells(keep, fmt)


def inputs(operands, kind, dtype, extent=EXTENT, density=0.6):
    """``(tensors, effective dense values, stored cells)``, one per operand.

    Every value is nonzero except where ``stored zeros`` sets half the driving
    input's stored entries to zero.  ``empty inner sums`` stores nothing in the
    odd first-mode slices of every sparse operand but the driving one, so the
    inner sum of every driving entry over such a slice is empty.  The driver always
    stores an entry in its first odd first-mode slice, so such an entry exists
    whenever the driver's first mode is the other operands' first mode; in the
    other cases the random draw supplies one, which the test asserts.
    """

    tensors, effective, cells = [], [], []
    for position, (name, fmt, indices) in enumerate(operands):
        shape = tuple(extent[letter] for letter in indices)
        generator = torch.Generator().manual_seed(200 + position)
        if dtype.is_floating_point:
            values = (torch.rand(shape, generator=generator) + 0.5).to(dtype)
        else:
            values = torch.randint(1, 6, shape, generator=generator).to(dtype)
        if set(fmt) == {"d"}:
            keep = torch.ones(shape, dtype=torch.bool)
        else:
            keep = torch.rand(shape, generator=generator) < density
        if position == 0 and kind == "empty driver":
            keep = torch.zeros(shape, dtype=torch.bool)
        if position == 0 and kind == "empty inner sums" and shape[0] > 1:
            # The premise needs a driving entry in an odd first-mode slice;
            # the random draw alone may store none there.
            keep.reshape(shape[0], -1)[1, 0] = True
        if position > 0 and kind == "empty inner sums" and set(fmt) != {"d"}:
            first = torch.arange(shape[0]).reshape(-1, *([1] * (len(shape) - 1)))
            keep = keep & (first % 2 == 0)
        stored_here = cells_stored(keep, fmt)
        if position == 0 and kind == "stored zeros":
            flat = stored_here.reshape(-1).nonzero().reshape(-1)
            zeroed = torch.zeros(stored_here.numel(), dtype=torch.bool)
            zeroed[flat[::2]] = True
            values = values.masked_fill(zeroed.reshape(shape), 0)
        tensors.append(operand(values, keep, fmt, name))
        effective.append(values * stored_here.to(dtype))
        cells.append(stored_here)
    return tensors, effective, cells


def entry_source(program: LoopProgram) -> Optional[str]:
    """The name of the input ``program`` stores the entries of, if it names one."""

    if program.entries_of is None:
        return None
    return next(
        decl.name for decl in program.tensors if decl.symbol == program.entries_of
    )


def symbol_named(program: LoopProgram, name: str):
    return next(decl.symbol for decl in program.tensors if decl.name == name)


def driver_of(case: Case) -> str:
    return case[0][0][0]


def scheduling(arm: int, **fields) -> CompileOptions:
    """Options requesting ``Schedule(**fields)``, with the given register blocking."""

    return replace(options(arm, False), requested_schedule=Schedule(**fields))


def naming_driver(case: Case, arm: int, **fields) -> CompileOptions:
    """Options requesting that the result store the driving input's entries."""

    return scheduling(arm, entries_of=driver_of(case), **fields)


def compiled(case, compile_options, dtype=torch.float64, extent=EXTENT):
    operands, result_indices, result_format = case
    result_shape, bindings = shapes_of(operands, result_indices, dtype, extent)
    return compile_cin_via_loopir(
        program_of(operands, result_indices, result_format, dtype),
        result_shape,
        bindings,
        compile_options=compile_options,
    )


def executed(case, compile_options, kind, dtype=torch.float64, extent=EXTENT):
    """``(out, kernel, torch.einsum reference in float64, stored cells)``."""

    operands, result_indices, result_format = case
    result_shape, _ = shapes_of(operands, result_indices, dtype, extent)
    tensors, effective, cells = inputs(operands, kind, dtype, extent)
    reference = torch.einsum(
        expression_of(operands, result_indices),
        *[value.to(torch.float64) for value in effective],
    )
    out, kernel = execute_cin_via_loopir(
        program_of(operands, result_indices, result_format, dtype),
        result_shape,
        *tensors,
        compile_options=compile_options,
    )
    return out, kernel, reference, cells


def densified(entries, values, result_shape) -> torch.Tensor:
    got = torch.zeros(result_shape, dtype=torch.float64)
    for entry, value in zip(entries, values.tolist()):
        got[entry] = value
    return got


def tolerance(dtype):
    if dtype == torch.float64:
        return 1e-12
    if dtype == torch.float32:
        return 1e-5
    return 0.0


def refusal_code(error: BaseException) -> str:
    """The structured code of a refusal: a defect's, or the first diagnostic's."""

    defect = getattr(error, "defect", None)
    if defect is not None:
        return str(defect.code)
    diagnostics = getattr(error, "diagnostics", ())
    if diagnostics:
        return str(diagnostics[0].code)
    raise AssertionError(f"an unstructured refusal: {type(error).__name__}: {error}")


def compile_refusal(case, compile_options) -> str:
    with pytest.raises(Exception) as error:
        compiled(case, compile_options)
    return refusal_code(error.value)


# -- with no request -------------------------------------------------------------


@pytest.mark.parametrize("case", DRIVEN, ids=DRIVEN_IDS)
@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("scheduled", [True, False], ids=["auto", "unscheduled"])
@pytest.mark.parametrize("extent", [EXTENT, LARGE_EXTENT], ids=["small", "large"])
def test_with_no_request_a_driven_coo_result_keeps_the_append(
    case, arm, scheduled, extent
):
    kernel = compiled(case, options(arm, scheduled), extent=extent)
    assert entry_source(kernel.lowering.program) is None
    source = kernel.cpp_source
    assert "_known_nnz" not in source
    assert ".emplace_back(" in source
    assert "C_nonempty" in source
    assert "#pragma omp parallel for" not in source
    assert "#pragma omp simd" not in source


@pytest.mark.parametrize("case", DRIVEN, ids=DRIVEN_IDS)
@pytest.mark.parametrize("kind", KINDS)
def test_with_no_request_the_result_stores_the_entries_some_term_reaches(case, kind):
    operands, result_indices, _ = case
    out, kernel, reference, cells = executed(case, options(0, True), kind)
    assert "_known_nnz" not in kernel.cpp_source
    result_shape = tuple(EXTENT[letter] for letter in result_indices)
    entries, values = coordinate_entries(out, result_shape)
    terms = torch.einsum(
        expression_of(operands, result_indices),
        *[stored.to(torch.float64) for stored in cells],
    )
    assert entries == [tuple(cell) for cell in (terms > 0).nonzero().tolist()], kind
    got = densified(entries, values, result_shape)
    assert torch.allclose(got, reference, rtol=1e-12, atol=1e-12), kind


# -- with the driving input named ------------------------------------------------


@pytest.mark.parametrize("case", DRIVEN, ids=DRIVEN_IDS)
@pytest.mark.parametrize("arm", [0, 1])
def test_a_named_driver_sizes_the_result_and_writes_by_position(case, arm):
    operands, _, result_format = case
    kernel = compiled(case, naming_driver(case, arm))
    driver = driver_of(case)
    assert entry_source(kernel.lowering.program) == driver
    assert entry_source(kernel.schedule.program) == driver
    leaf = len(result_format) - 1
    source = kernel.cpp_source
    assert f"int64_t _known_nnz = {driver}_values.size(0);" in source
    assert "torch::empty({_known_nnz}" in source
    assert f"C_values[p{driver}{leaf}] = " in source
    for level in range(len(result_format)):
        assert f"C{level}_crd[p{driver}{leaf}] = " in source
    assert ".emplace_back(" not in source
    assert "_nonempty" not in source
    # The kernel's parameters are std::vectors under the native ABI; its body
    # holds no growable buffer.
    body = source.split("Tensor evaluate(", 1)[1].split(") {", 1)[1]
    assert "std::vector<" not in body


@pytest.mark.parametrize("case", DRIVEN, ids=DRIVEN_IDS)
@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("kind", KINDS)
def test_a_named_driver_stores_every_driving_entry(case, arm, kind):
    operands, result_indices, _ = case
    out, kernel, reference, cells = executed(case, naming_driver(case, arm), kind)
    assert "_known_nnz" in kernel.cpp_source
    result_shape = tuple(EXTENT[letter] for letter in result_indices)
    entries, values = coordinate_entries(out, result_shape)
    driving = [tuple(cell) for cell in cells[0].nonzero().tolist()]
    assert entries == driving, kind
    got = densified(entries, values, result_shape)
    assert torch.allclose(got, reference, rtol=1e-12, atol=1e-12), (
        kind,
        float((got - reference).abs().max()),
    )
    if kind == "empty inner sums" and any(
        set(fmt) != {"d"} for _, fmt, _ in operands[1:]
    ):
        # The premise: some driving entry's inner sum visits nothing, and the
        # entry is stored all the same, holding zero.
        terms = torch.einsum(
            expression_of(operands, result_indices),
            *[c.to(torch.float64) for c in cells],
        )
        empty = [entry for entry in driving if float(terms[entry]) == 0.0]
        assert empty, "no driving entry with an empty inner sum"
        assert all(float(got[entry]) == 0.0 for entry in empty)


@pytest.mark.parametrize("case", DRIVEN, ids=DRIVEN_IDS)
@pytest.mark.parametrize("kind", ["random", "empty inner sums"])
def test_the_reference_interpreter_stores_the_same_entries(case, kind):
    operands, result_indices, _ = case
    dtype = torch.float64
    result_shape, _ = shapes_of(operands, result_indices, dtype)
    _, effective, _ = inputs(operands, kind, dtype)
    out, kernel, _, _ = executed(case, naming_driver(case, 0), kind)
    program = kernel.lowering.program
    by_name: Dict[str, torch.Tensor] = {
        name: dense for (name, _, _), dense in zip(operands, effective)
    }
    bindings = {}
    for decl in program.tensors:
        if decl.symbol not in program.inputs:
            continue
        dense = by_name[decl.name]
        if all(level.kind is LevelKind.DENSE for level in decl.levels):
            # An all-dense input binds nested sequences in logical mode order.
            bindings[decl.symbol] = dense.tolist()
            continue
        # Every stored value is nonzero here, so the interpreter's storage,
        # which keeps nonzero cells, holds exactly what the tensors store.
        bindings[decl.symbol] = LevelTensorStorage.from_dense(
            dense.tolist(),
            tuple(dense.shape),
            tuple(level.mode for level in decl.levels),
            tuple(level.kind for level in decl.levels),
        )
    oracle = run_program(program, bindings, {program.outputs[0]: result_shape})[
        program.outputs[0]
    ]
    entries, values = coordinate_entries(out, result_shape)
    oracle_entries = list(zip(*oracle.coordinates)) if oracle.values else []
    assert entries == [tuple(entry) for entry in oracle_entries]
    assert torch.allclose(
        values, torch.tensor(oracle.values, dtype=torch.float64), rtol=1e-12, atol=0
    )


@pytest.mark.parametrize(
    "case", [DRIVEN[0], DRIVEN[3], DRIVEN[8]], ids=[DRIVEN_IDS[i] for i in (0, 3, 8)]
)
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.int64, torch.int32], ids=["f32", "i64", "i32"]
)
@pytest.mark.parametrize("kind", ["random", "empty inner sums", "stored zeros"])
def test_a_named_driver_in_other_element_types(case, dtype, kind):
    operands, result_indices, _ = case
    out, kernel, reference, cells = executed(case, naming_driver(case, 0), kind, dtype)
    assert entry_source(kernel.lowering.program) == driver_of(case)
    assert out.storage.value.dtype == dtype
    result_shape = tuple(EXTENT[letter] for letter in result_indices)
    entries, values = coordinate_entries(out, result_shape)
    assert entries == [tuple(cell) for cell in cells[0].nonzero().tolist()]
    got = densified(entries, values, result_shape)
    atol = tolerance(dtype)
    assert torch.allclose(got, reference, rtol=atol, atol=atol)


@pytest.mark.parametrize("case", DRIVEN, ids=DRIVEN_IDS)
@pytest.mark.parametrize("arm", [0, 1])
def test_the_shipped_compiler_emits_its_own_text_when_its_driver_is_named(case, arm):
    """The shipped compiler already stores the driving input's entries under
    scalar accumulation, so naming that input changes nothing in its kernel."""

    operands, result_indices, result_format = case
    default = shipped_text(operands, result_indices, result_format, arm)
    if default is None:
        pytest.skip("the shipped compiler builds no kernel for this program")
    result_shape, bindings = shapes_of(operands, result_indices, torch.float64)
    named = legacy_generated_cpp(
        program_of(operands, result_indices, result_format, torch.float64),
        result_shape,
        bindings,
        compile_options=naming_driver(case, arm),
    )
    assert named == default


@pytest.mark.parametrize("case", DRIVEN, ids=DRIVEN_IDS)
@pytest.mark.parametrize("kind", KINDS)
def test_both_compilers_store_the_same_entries_when_the_driver_is_named(case, kind):
    """The shipped kernel (public einsum, its own schedule) against the new kernel
    with the driving input named: the same level arrays, the same values."""

    operands, result_indices, result_format = case
    tensors, _, _ = inputs(operands, kind, torch.float64)
    shipped = scorch.einsum(
        expression_of(operands, result_indices), *tensors, format=result_format
    )
    out, _, _, _ = executed(case, naming_driver(case, 0), kind)
    result_shape = tuple(EXTENT[letter] for letter in result_indices)
    new_entries, new_values = coordinate_entries(out, result_shape)
    shipped_entries, shipped_values = coordinate_entries(shipped, result_shape)
    assert new_entries == shipped_entries, kind
    assert torch.allclose(new_values, shipped_values, rtol=1e-12, atol=1e-12), kind


# -- naming any other tensor -----------------------------------------------------


@pytest.mark.parametrize("case", APPENDED, ids=APPENDED_IDS)
@pytest.mark.parametrize("arm", [0, 1])
def test_a_coo_result_no_single_input_drives_keeps_the_append(case, arm):
    kernel = compiled(case, options(arm, True))
    assert entry_source(kernel.lowering.program) is None
    assert "_known_nnz" not in kernel.cpp_source
    assert ".emplace_back(" in kernel.cpp_source


@pytest.mark.parametrize("case", APPENDED, ids=APPENDED_IDS)
def test_naming_an_input_no_loop_walks_is_refused(case):
    for compile_options in (scheduling(0, entries_of="A"), naming_driver(case, 1)):
        assert compile_refusal(case, compile_options) == "unsupported_entries_of"
    operands, result_indices, result_format = case
    with pytest.raises(UnsupportedFeature) as error:
        Scheduler.apply_schedule(
            program_of(operands, result_indices, result_format, torch.float64),
            Schedule(entries_of="A"),
            dense_results_any_order=True,
        )
    assert refusal_code(error.value) == "unsupported_entries_of"


@pytest.mark.parametrize("case", DRIVEN, ids=DRIVEN_IDS)
def test_naming_another_tensor_is_refused(case):
    operands = case[0]
    others = [name for name, _, _ in operands[1:]]
    for name in others:
        assert compile_refusal(case, scheduling(0, entries_of=name)) == (
            "unsupported_entries_of"
        ), name
    assert compile_refusal(case, scheduling(0, entries_of="C")) == (
        "invalid_entries_of"
    )
    with pytest.raises(InvalidSchedule) as error:
        compiled(case, scheduling(0, entries_of="Z"))
    assert refusal_code(error.value) == "invalid_entries_of"
    assert "'Z'" in str(error.value)


def test_a_dense_result_refuses_entries_of():
    case = ([("A", "oo", "ij"), ("B", "dd", "ik")], "ij", "dd")
    assert compile_refusal(case, naming_driver(case, 0)) == "unsupported_entries_of"


def test_the_front_end_refuses_a_plan_its_program_does_not_bear_out():
    """Plan legality and the front end apply the same rule to the same nest;
    the front end checks it again on the facts it lowers, so a plan that
    reaches it naming another input is refused there with the same code."""

    from scorch.compiler.loopir.lower_cin import lower_normalized_cin_to_loopir

    operands, result_indices, result_format = DRIVEN[0]
    scheduled = Scheduler.apply_schedule(
        program_of(operands, result_indices, result_format, torch.float64),
        Schedule(entries_of="A"),
        dense_results_any_order=True,
    )
    plan = scheduled.verified_loop_plan
    program = compiled(DRIVEN[0], naming_driver(DRIVEN[0], 0)).lowering.program
    with pytest.raises(LoopIRLoweringError) as error:
        lower_normalized_cin_to_loopir(
            scheduled.normalized_cin,
            planned_loop_order=plan.loop_order,
            planned_entries_of=symbol_named(program, "B"),
        )
    assert error.value.defect.code == "unsupported_entries_of"


def test_a_statement_with_no_loops_refuses_entries_of():
    statement = TensorAssign(TensorVar("C", shape=())[[]], TensorVar("A", shape=())[[]])
    with pytest.raises(UnsupportedFeature) as error:
        Scheduler.apply_schedule(statement, Schedule(entries_of="A"))
    assert refusal_code(error.value) == "unsupported_entries_of"


# -- the field ---------------------------------------------------------------------


def input_shapes(program: LoopProgram, operands) -> Dict[object, Tuple[int, ...]]:
    return {
        symbol_named(program, name): tuple(EXTENT[letter] for letter in indices)
        for name, _, indices in operands
    }


def test_the_schedule_field_and_its_cache_key():
    assert Schedule(entries_of="A").entries_of == "A"
    with pytest.raises(TypeError):
        Schedule(entries_of=("A",))
    with pytest.raises(ValueError):
        Schedule(entries_of="")
    # A schedule that sets neither field added on 2026-10-13 keeps the key
    # spelled before them, and so does every options digest carrying it.
    assert Schedule().cache_key == (
        "Schedule(loop_order=None, tiles=(), relayout=None, tag='', "
        "parallel_loop=None, assembly=None, accumulator=None)"
    )
    assert Schedule(assembly="single_pass_serial").cache_key == (
        "Schedule(loop_order=None, tiles=(), relayout=None, tag='', "
        "parallel_loop=None, assembly='single_pass_serial', accumulator=None)"
    )
    assert Schedule(entries_of="A").cache_key.endswith(", entries_of='A')")
    assert len({Schedule().cache_key, Schedule(entries_of="A").cache_key}) == 2


def test_the_plan_records_the_field_only_when_requested():
    operands, result_indices, result_format = DRIVEN[0]
    cin = program_of(operands, result_indices, result_format, torch.float64)
    plain = Scheduler.apply_schedule(cin, Schedule(), dense_results_any_order=True)
    named = Scheduler.apply_schedule(
        cin, Schedule(entries_of="A"), dense_results_any_order=True
    )
    assert plain.verified_loop_plan.entries_of is None
    assert named.verified_loop_plan.entries_of is not None
    plain_dump = canonical_plan_dump(plain.normalized_cin, plain.verified_loop_plan)
    named_dump = canonical_plan_dump(named.normalized_cin, named.verified_loop_plan)
    assert '"entries_of"' not in plain_dump
    assert '"entries_of":' in named_dump
    assert plain_dump != named_dump


def test_the_verifier_refuses_a_malformed_or_inapplicable_entry_source():
    program = compiled(DRIVEN[0], naming_driver(DRIVEN[0], 0)).lowering.program
    verify_program(program)
    refusals = [
        ("A", "malformed_state"),
        (program.outputs[0], "invalid_entries_of"),
        # B[i, k] does not have the result's dimensions.
        (symbol_named(program, "B"), "unsupported_entries_of"),
    ]
    for source, code in refusals:
        with pytest.raises(LoopIRVerificationError) as error:
            verify_program(dataclasses.replace(program, entries_of=source))
        assert error.value.defect.code == code, source
    dense = compiled(
        ([("A", "oo", "ij"), ("B", "dd", "ik")], "ij", "dd"), options(0, False)
    ).lowering.program
    assert dense.entries_of is None
    with pytest.raises(LoopIRVerificationError) as error:
        verify_program(dataclasses.replace(dense, entries_of=symbol_named(dense, "A")))
    assert error.value.defect.code == "unsupported_entries_of"


def test_the_target_refuses_an_entry_source_its_loops_do_not_walk():
    """``ss,ss ij,ik->ij``: the loop over ``i`` merges both inputs, so the result
    loops do not walk ``A``'s stored entries one by one.  The program is legal
    (``A`` has the result's dimensions and multiplies the write), and the target
    refuses it with a code instead of storing fewer entries."""

    operands, result_indices, _ = APPENDED[0]
    program = compiled(APPENDED[0], options(0, False)).lowering.program
    named = dataclasses.replace(program, entries_of=symbol_named(program, "A"))
    verify_program(named)
    with pytest.raises(LoopIRTargetError) as error:
        assembly_choices(
            named,
            input_shapes=input_shapes(named, operands),
            result_shape=tuple(EXTENT[letter] for letter in result_indices),
        )
    assert error.value.defect.code == "unsupported_entries_of"


def test_the_canonical_form_names_the_entry_source_only_when_there_is_one():
    driven = compiled(DRIVEN[0], naming_driver(DRIVEN[0], 0)).lowering.program
    plain = compiled(DRIVEN[0], options(0, False)).lowering.program
    assert '"entries_of":' in canonical_program_dump(driven)
    assert '"entries_of"' not in canonical_program_dump(plain)
    assert canonical_program_dump(plain) == canonical_program_dump(
        dataclasses.replace(driven, entries_of=None)
    )


@pytest.mark.parametrize("case", DRIVEN, ids=DRIVEN_IDS)
def test_scheduling_and_schedule_erasure_keep_the_entry_source(case):
    driver = driver_of(case)
    kernel = compiled(case, naming_driver(case, 0))
    assert entry_source(kernel.lowering.program) == driver
    assert entry_source(kernel.schedule.program) == driver
    assert entry_source(erase_schedule(kernel.schedule.program)) == driver


# -- the parallel loop over a COO driver's entries -------------------------------

COO_DRIVEN = [
    case for case in DRIVEN if set(case[0][0][1]) == {"o"} and len(case[0][0][1]) >= 2
]
COO_DRIVEN_IDS = [label(case) for case in COO_DRIVEN]
PARALLEL = "single_pass_chunk_parallel"


@pytest.mark.parametrize("case", COO_DRIVEN, ids=COO_DRIVEN_IDS)
@pytest.mark.parametrize("arm", [0, 1])
def test_a_coo_driver_is_walked_by_one_parallel_loop_over_its_entries(case, arm):
    """The shipped compiler's loop: one counted loop over the driver's entries,
    every level's coordinate read at the entry, run in parallel; no loop over
    runs of equal coordinates."""

    operands, _, result_format = case
    kernel = compiled(case, naming_driver(case, arm, assembly=PARALLEL))
    driver = driver_of(case)
    entry = f"p{driver}{len(result_format) - 1}"
    source = kernel.cpp_source
    loop = f"for (int64_t {entry} = 0; {entry} < p{driver}0_end; {entry}++)"
    lines = source.splitlines()
    at = next(number for number, line in enumerate(lines) if loop in line)
    assert lines[at - 1].strip().startswith("#pragma omp parallel for")
    assert f"p{driver}1_end" not in source
    assert f"int64_t _known_nnz = {driver}_values.size(0);" in source
    assert ".emplace_back(" not in source


@pytest.mark.parametrize("case", COO_DRIVEN, ids=COO_DRIVEN_IDS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [torch.float64, torch.int64], ids=["f64", "i64"])
def test_the_parallel_loop_stores_every_driving_entry(case, kind, dtype):
    operands, result_indices, _ = case
    out, kernel, reference, cells = executed(
        case, naming_driver(case, 0, assembly=PARALLEL), kind, dtype
    )
    assert "#pragma omp parallel for" in kernel.cpp_source
    result_shape = tuple(EXTENT[letter] for letter in result_indices)
    entries, values = coordinate_entries(out, result_shape)
    assert entries == [tuple(cell) for cell in cells[0].nonzero().tolist()], kind
    got = densified(entries, values, result_shape)
    atol = tolerance(dtype)
    assert torch.allclose(got, reference, rtol=atol, atol=atol), kind


# 1.6e5 result cells; a COO driver at density 0.8 stores about 1.3e5 entries.
MANY_ENTRIES = {"i": 400, "j": 400, "k": 3, "l": 3}


@pytest.mark.parametrize(
    "case", [DRIVEN[0], DRIVEN[3]], ids=[DRIVEN_IDS[0], DRIVEN_IDS[3]]
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.int64], ids=["f64", "i64"])
def test_the_parallel_loop_on_eight_threads_matches_the_serial_kernel(case, dtype):
    """Review finding 6: every position written once, by more than one worker.
    The arrays are allocated uninitialised, so a skipped position would hold
    garbage: the parallel kernel's level arrays and values must equal the
    serial kernel's exactly (integers) or to rounding (float64, the same
    per-entry sums in the same order)."""

    operands, result_indices, result_format = case
    result_shape, _ = shapes_of(operands, result_indices, dtype, MANY_ENTRIES)
    tensors, _, cells = inputs(operands, "random", dtype, MANY_ENTRIES, 0.8)
    assert int(cells[0].sum()) >= 100_000
    cin = program_of(operands, result_indices, result_format, dtype)
    threads = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        arrays = {}
        for strategy in ("single_pass_serial", PARALLEL):
            out, kernel = execute_cin_via_loopir(
                cin,
                result_shape,
                *tensors,
                compile_options=naming_driver(case, 0, assembly=strategy),
            )
            assert ("#pragma omp parallel for" in kernel.cpp_source) == (
                strategy == PARALLEL
            )
            arrays[strategy] = (
                [level[0].clone() for level in out.storage.index.mode_indices],
                out.storage.value.clone(),
            )
    finally:
        torch.set_num_threads(threads)
    serial_levels, serial_values = arrays["single_pass_serial"]
    parallel_levels, parallel_values = arrays[PARALLEL]
    for serial, parallel in zip(serial_levels, parallel_levels):
        assert torch.equal(serial, parallel)
    if dtype.is_floating_point:
        assert torch.allclose(serial_values, parallel_values, rtol=1e-15, atol=0)
    else:
        assert torch.equal(serial_values, parallel_values)


@pytest.mark.parametrize("case", COO_DRIVEN, ids=COO_DRIVEN_IDS)
@pytest.mark.parametrize("extent", [EXTENT, LARGE_EXTENT], ids=["small", "large"])
def test_the_automatic_rule_runs_a_named_coo_driver_in_parallel_from_32_cells(
    case, extent
):
    """With the driving input named and no strategy requested, the rule's
    threshold applies to what the loop splits: the driver's entries, at most
    the result's cells.  Fewer than 32 can reach only one worker
    (``scorch_nthreads``), so the serial loop is kept there."""

    _, result_indices, _ = case
    cells = math.prod(extent[letter] for letter in result_indices)
    kernel = compiled(case, naming_driver(case, 0), extent=extent)
    assert ("#pragma omp parallel for" in kernel.cpp_source) == (cells >= 32)


def test_a_parallel_request_for_a_csr_driver_is_refused_at_plan_legality():
    """Legal to store ``A``'s entries, but a CSR driver's levels are counted
    loops of their own, which no lowering splits by entry: the shipped compiler
    runs them serially and so does this one, until the timing grid says where
    a parallel row loop pays."""

    case = ([("A", "ds", "ij"), ("B", "dd", "ik"), ("D", "ds", "jk")], "ij", "oo")
    code = compile_refusal(case, naming_driver(case, 0, assembly=PARALLEL))
    assert code == "unsupported_schedule_assembly"


@pytest.mark.parametrize("case", COO_DRIVEN, ids=COO_DRIVEN_IDS)
def test_a_parallel_request_without_the_driver_named_is_refused(case):
    code = compile_refusal(case, scheduling(0, assembly=PARALLEL))
    assert code == "unsupported_schedule_assembly"


def test_a_parallel_request_for_an_appended_coo_result_is_refused():
    code = compile_refusal(APPENDED[0], scheduling(0, assembly=PARALLEL))
    assert code == "unsupported_schedule_assembly"


def _chunk_parallel_plan(case):
    operands, result_indices, result_format = case
    scheduled = Scheduler.apply_schedule(
        program_of(operands, result_indices, result_format, torch.float64),
        Schedule(assembly=PARALLEL, entries_of=driver_of(case)),
        dense_results_any_order=True,
    )
    ids = {}
    stmt = scheduled.normalized_cin
    while isinstance(stmt, ForAll):
        ids[stmt.index_var.name] = stmt.index_var.index_id
        stmt = stmt.stmt
    return scheduled, ids


def _affine(ids, letter, placement):
    return LoopTile(
        loop=LoopRef(ids[letter]),
        width=2,
        placement=placement,
        parallel=False,
        kind="affine",
        accumulation="direct",
        unroll=False,
    )


@pytest.mark.parametrize(
    "placed",
    ["outermost", "child_of_first_result_loop", "child_of_last_result_loop"],
)
def test_entries_of_and_the_parallel_single_pass_refuse_a_tile(placed):
    """mlsys-submission's review: the result's loops must stay the outermost ones,
    so a reduction tile placed above or between them must be refused, and so must
    one below them, which no lowering is shown to run.

    Plan legality refuses every such tile before it reaches the entries_of and
    assembly checks: the automatic plan derives no tile for a sparse result
    (``auto_accumulator_lifetime``), and an explicit plan may not tile a sparse
    result (``tiled_sparse_output``).  Both checks refuse it as well, so neither
    rule rests on those earlier refusals."""

    scheduled, ids = _chunk_parallel_plan(DRIVEN[0])  # oo,dd ij,ik->ij
    assert verify_loop_plan(
        scheduled.normalized_cin,
        scheduled.verified_loop_plan,
        dense_results_any_order=True,
    )
    assert scheduled.verified_loop_plan.provenance == "auto"
    placement = {
        "outermost": LoopPlacement(PlacementKind.OUTERMOST),
        "child_of_first_result_loop": LoopPlacement(
            PlacementKind.CHILD_OF, LoopRef(ids["i"])
        ),
        "child_of_last_result_loop": LoopPlacement(
            PlacementKind.CHILD_OF, LoopRef(ids["j"])
        ),
    }[placed]
    plan = replace(scheduled.verified_loop_plan, tiles=(_affine(ids, "k", placement),))
    with pytest.raises(InvalidSchedule) as error:
        verify_loop_plan(scheduled.normalized_cin, plan, dense_results_any_order=True)
    assert error.value.diagnostics[0].code == "auto_accumulator_lifetime"
    explicit = replace(plan, provenance="explicit", auto_policy=None, workspace=None)
    with pytest.raises(UnsupportedFeature) as error:
        verify_loop_plan(
            scheduled.normalized_cin, explicit, dense_results_any_order=True
        )
    assert error.value.diagnostics[0].code == "tiled_sparse_output"
    facts = _build_facts(_analyze_loop_plan_cin(scheduled.normalized_cin), explicit)
    state = _apply_affine_placements(facts, explicit, False)
    with pytest.raises(UnsupportedFeature) as error:
        _verify_entries_of(facts, explicit, state)
    assert error.value.diagnostics[0].code == "unsupported_entries_of"
    with pytest.raises(UnsupportedFeature) as error:
        _verify_assembly(facts, explicit, state)
    assert error.value.diagnostics[0].code == "unsupported_schedule_assembly"


def test_a_tile_on_a_coo_result_index_is_refused_at_plan_legality():
    scheduled, ids = _chunk_parallel_plan(DRIVEN[0])
    plan = replace(
        scheduled.verified_loop_plan,
        tiles=(_affine(ids, "i", LoopPlacement(PlacementKind.OUTERMOST)),),
    )
    with pytest.raises(UnsupportedFeature) as error:
        verify_loop_plan(scheduled.normalized_cin, plan, dense_results_any_order=True)
    assert error.value.diagnostics[0].code == "sparse_affine_tile"


@pytest.mark.parametrize("case", APPENDED[:2], ids=APPENDED_IDS[:2])
def test_the_refusal_names_both_receivers_the_parallel_single_pass_takes(case):
    operands, result_indices, result_format = case
    with pytest.raises(UnsupportedFeature) as error:
        Scheduler.apply_schedule(
            program_of(operands, result_indices, result_format, torch.float64),
            Schedule(assembly=PARALLEL),
            dense_results_any_order=True,
        )
    assert "unsupported_schedule_assembly" in str(error.value)
    assert "a COO result that stores the entries of a COO input" in str(error.value)
    assert "named by entries_of" in str(error.value)
