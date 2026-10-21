"""Which loops carry a vector mark, as a schedule decision both compilers read.

Bobby, 2026-10-12 22:49 (desk thread t-new-vs-shipped): a speed comparison of the
shipped and the new compiler counts only when both ran the same schedule, so
every decision a compiler makes about a generated kernel has to be expressible
in the schedule.  Which loops carry ``#pragma omp simd`` was not: each compiler
applied one rule (``iter_lattice.mark_simd_on_reduction_loops``: a loop of a sum
kept in a register, with no loop over a compressed level's positions in its
body, not parallel or unrolled, and with no early exit) and nothing could ask
for anything else.

``Schedule.vector_loops``, ``LoopPlan.vector_loops`` and
``LoopProgram.vector_loops`` record it.  ``None`` keeps each compiler's own
marks, so every kernel is emitted as before; a tuple keeps the mark on exactly
the named loops among those the rule finds safe.  The two compilers' marks
with no request differ: both apply the rule only to the loops below a sparse
result entry whose sum is kept in a register, and there the shipped compiler
marks the safe loops while the new compiler, which never marked them, marks
none.  A comparison of the two that must run the same kernel therefore names
the loops (or the empty tuple) rather than relying on ``None``.

The programs below reduce into a COO result one input drives, and their
schedules name that input (``Schedule.entries_of``), so both compilers store
every one of its entries.  Without it the new compiler stores only the entries
some term reaches and writes, in every inner iteration, a flag recording that
the sum ran; it marks no such loop, since the mark (with no reduction clause)
is shown safe only for the one sum the shipped compiler's marked loops carry,
and a request naming one is refused.  Checked, in both compilers and both
register-blocking settings:
* naming a loop the rule does not mark is refused (``unsupported_vector_loop``),
  and an unknown loop name is an invalid schedule;
* the kernel with and without marks computes ``torch.einsum``'s values, in
  float64 and int64, with a reduction of 64 terms;
* the field's validation, plan identity, canonical program form, verifier codes
  and schedule erasure.
"""

import dataclasses
import re
from dataclasses import replace
from typing import List, Optional

import pytest
import torch

from scorch.compiler.diagnostics import InvalidSchedule, VerificationError
from scorch.compiler.loop_plan import LoopPart, LoopRef, verify_loop_plan
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.lower_llir import LoopIRTargetError, lower_loopir_to_llir
from scorch.compiler.loopir.nodes import (
    DenseFor,
    MergedSparseFor,
    SparseFor,
    VectorLoopPart,
)
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.compiler.loopir.plan_identity import canonical_plan_dump
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.schedule_passes import erase_schedule
from scorch.compiler.loopir.verifier import (
    LoopIRVerificationError,
    _walk_declared_schema,
    verify_program,
)
from scorch.compiler.scheduler import Schedule, Scheduler
from tests.test_scorch.test_loopir_coordinate_levels import (
    coordinate_entries,
    label,
    options,
)
from tests.test_scorch.test_loopir_operand_conversions import EXTENT
from tests.test_scorch.test_loopir_coordinate_result_known_count import (
    DRIVEN,
    expression_of,
    input_shapes,
    inputs,
    program_of,
    shapes_of,
)

# Programs whose kernel carries exactly one vector mark, on a dense reduction
# loop, through the shipped compiler's scheduled route, and whose new-compiler
# kernel takes the same mark when it is named: the loop is named beside each.
VECTOR_CASES = [
    (DRIVEN[0], "k"),
    (DRIVEN[1], "k"),
    (DRIVEN[4], "k"),
    (DRIVEN[8], "l"),
    (DRIVEN[10], "l"),
]
VECTOR_IDS = [label(case) for case, _ in VECTOR_CASES]

PRAGMA_LINE = re.compile(r"[ \t]*#pragma omp simd\n")


def driving_input(case) -> Optional[str]:
    """The input a COO result's entries are named after; ``None`` otherwise."""

    operands, _, result_format = case
    return operands[0][0] if set(result_format) == {"o"} else None


def requesting(arm, vector_loops, entries_of=None):
    return replace(
        options(arm, False),
        requested_schedule=Schedule(vector_loops=vector_loops, entries_of=entries_of),
    )


def shipped_cpp(case, arm, vector_loops, name_driver=True) -> str:
    operands, result_indices, result_format = case
    result_shape, bindings = shapes_of(operands, result_indices, torch.float64)
    return legacy_generated_cpp(
        program_of(operands, result_indices, result_format, torch.float64),
        result_shape,
        bindings,
        compile_options=requesting(
            arm, vector_loops, driving_input(case) if name_driver else None
        ),
    )


def new_kernel(case, arm, vector_loops, name_driver=True):
    operands, result_indices, result_format = case
    result_shape, bindings = shapes_of(operands, result_indices, torch.float64)
    return compile_cin_via_loopir(
        program_of(operands, result_indices, result_format, torch.float64),
        result_shape,
        bindings,
        compile_options=requesting(
            arm, vector_loops, driving_input(case) if name_driver else None
        ),
    )


def new_cpp(case, arm, vector_loops, name_driver=True) -> str:
    return new_kernel(case, arm, vector_loops, name_driver).cpp_source


COMPILERS = {"shipped": shipped_cpp, "new": new_cpp}


def marked_loops(cpp: str) -> List[str]:
    """The variable of every loop the kernel marks for vector instructions."""

    return re.findall(r"#pragma omp simd\n\s*for \((?:int64_t|int) (\w+) = ", cpp)


def refusal_code(error: BaseException) -> str:
    defect = getattr(error, "defect", None)
    if defect is not None:
        return defect.code
    diagnostics = getattr(error, "diagnostics", ())
    if diagnostics:
        return diagnostics[0].code
    return str(error)


@pytest.mark.parametrize("case,loop", VECTOR_CASES, ids=VECTOR_IDS)
@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("compiler", sorted(COMPILERS))
def test_naming_the_rules_loops_or_none_changes_only_the_marks(
    case, loop, arm, compiler
):
    text = COMPILERS[compiler]
    default = text(case, arm, None)
    # With no request each compiler keeps its own marks (module docstring).
    assert marked_loops(default) == ([loop] if compiler == "shipped" else [])
    named = text(case, arm, (loop,))
    assert marked_loops(named) == [loop]
    assert PRAGMA_LINE.sub("", named) == PRAGMA_LINE.sub("", default)
    if compiler == "shipped":
        assert named == default
    unmarked = text(case, arm, ())
    assert "#pragma omp simd" not in unmarked
    assert unmarked == PRAGMA_LINE.sub("", default)


@pytest.mark.parametrize("case,loop", VECTOR_CASES, ids=VECTOR_IDS)
@pytest.mark.parametrize("arm", [0, 1])
def test_one_request_marks_the_same_loops_in_both_compilers(case, loop, arm):
    for request in ((), (loop,)):
        assert marked_loops(shipped_cpp(case, arm, request)) == marked_loops(
            new_cpp(case, arm, request)
        ), request


# A dense result: the automatic arm runs the outermost loop in parallel, a
# decision made after the plan, so only lowering can refuse a mark on it.
DENSE_PARALLEL_CASE = ([("A", "dd", "ij"), ("B", "dd", "jk")], "ik", "dd")
PARALLEL_LINE = re.compile(
    r"#pragma omp parallel for[^\n]*\n\s*for \((?:int64_t|int) (\w+) = "
)


@pytest.mark.parametrize("compiler", sorted(COMPILERS))
def test_a_mark_on_the_loop_the_automatic_arm_runs_in_parallel_is_refused(compiler):
    default = COMPILERS[compiler](DENSE_PARALLEL_CASE, 1, None)
    parallel = PARALLEL_LINE.findall(default)
    assert parallel, "the premise: the automatic arm runs a loop in parallel"
    with pytest.raises(Exception) as error:
        COMPILERS[compiler](DENSE_PARALLEL_CASE, 1, (parallel[0],))
    assert refusal_code(error.value) == "unsupported_vector_loop", str(error.value)


@pytest.mark.parametrize("case,loop", VECTOR_CASES, ids=VECTOR_IDS)
@pytest.mark.parametrize("compiler", sorted(COMPILERS))
def test_a_loop_the_rule_does_not_mark_is_refused(case, loop, compiler):
    """The outermost loop binds a result coordinate: it is not a loop of the sum
    kept in a register, so the rule never marks it and a request is refused."""

    operands, result_indices, _ = case
    outer = result_indices[0]
    with pytest.raises(Exception) as error:
        COMPILERS[compiler](case, 0, (loop, outer))
    assert refusal_code(error.value) == "unsupported_vector_loop", str(error.value)
    assert repr(outer) in str(error.value)


@pytest.mark.parametrize("compiler", sorted(COMPILERS))
def test_an_unknown_loop_name_is_an_invalid_schedule(compiler):
    case, _ = VECTOR_CASES[0]
    with pytest.raises(InvalidSchedule, match="zz"):
        COMPILERS[compiler](case, 0, ("zz",))


@pytest.mark.parametrize("compiler", sorted(COMPILERS))
def test_a_split_part_of_an_unsplit_loop_is_an_invalid_schedule(compiler):
    case, loop = VECTOR_CASES[0]
    with pytest.raises((InvalidSchedule, VerificationError), match="split part"):
        COMPILERS[compiler](case, 0, (f"{loop}_in",))


@pytest.mark.parametrize("case,loop", VECTOR_CASES, ids=VECTOR_IDS)
@pytest.mark.parametrize("arm", [0, 1])
def test_without_the_driver_named_the_new_compiler_refuses_a_guarded_mark(
    case, loop, arm
):
    """Review finding 5: with no input named the new kernel's sum loop also
    writes the "the sum ran" flag, so it is not marked and a request naming it
    is refused; the shipped kernel, which stores every driving entry and keeps
    no flag, marks it either way."""

    plain = new_cpp(case, arm, None, name_driver=False)
    assert "C_nonempty = true" in plain
    assert "#pragma omp simd" not in plain
    with pytest.raises(Exception) as error:
        new_cpp(case, arm, (loop,), name_driver=False)
    assert refusal_code(error.value) == "unsupported_vector_loop", str(error.value)
    assert marked_loops(shipped_cpp(case, arm, (loop,), name_driver=False)) == [loop]


@pytest.mark.parametrize("case,loop", VECTOR_CASES, ids=VECTOR_IDS)
@pytest.mark.parametrize("dtype", [torch.float64, torch.int64], ids=["f64", "i64"])
def test_the_kernel_computes_the_same_values_marked_and_unmarked(case, loop, dtype):
    """Review finding 5: the marked loop runs 64 iterations, so a vectorised sum
    is exercised; marked and unmarked kernels against each other and against
    ``torch.einsum``.  Both kernels may reorder a float sum (``-ffast-math``),
    so float64 is judged against twice the error bound of a 64-term sum of
    terms below 3.375 (a product of at most three values below 1.5), integers
    exactly."""

    operands, result_indices, result_format = case
    extent = dict(EXTENT, **{loop: 64})
    result_shape, _ = shapes_of(operands, result_indices, dtype, extent)
    tensors, effective, _ = inputs(operands, "random", dtype, extent)
    reference = torch.einsum(
        expression_of(operands, result_indices),
        *[value.to(torch.float64) for value in effective],
    )
    got = {}
    for marked in (False, True):
        out, kernel = execute_cin_via_loopir(
            program_of(operands, result_indices, result_format, dtype),
            result_shape,
            *tensors,
            compile_options=requesting(
                0, (loop,) if marked else (), driving_input(case)
            ),
        )
        assert marked_loops(kernel.cpp_source) == ([loop] if marked else [])
        entries, values = coordinate_entries(out, result_shape)
        dense = torch.zeros(result_shape, dtype=torch.float64)
        for entry, value in zip(entries, values.tolist()):
            dense[entry] = value
        got[marked] = (entries, dense)
    assert got[True][0] == got[False][0]
    atol = 2 * 64 * (64 * 3.375) * 2.0**-53 if dtype.is_floating_point else 0.0
    for marked in (False, True):
        assert torch.allclose(got[marked][1], reference, rtol=0, atol=atol), marked
    assert torch.allclose(got[True][1], got[False][1], rtol=0, atol=atol)


def test_the_schedule_field_is_validated():
    assert Schedule(vector_loops=["k", "j"]).vector_loops == ("k", "j")
    with pytest.raises(TypeError):
        Schedule(vector_loops="k")
    with pytest.raises(ValueError):
        Schedule(vector_loops=("k", "k"))
    with pytest.raises(ValueError):
        Schedule(vector_loops=("",))
    keys = {
        Schedule().cache_key,
        Schedule(vector_loops=()).cache_key,
        Schedule(vector_loops=("k",)).cache_key,
    }
    assert len(keys) == 3


def test_the_plan_records_the_field_only_when_requested():
    (operands, result_indices, result_format), loop = VECTOR_CASES[0]
    cin = program_of(operands, result_indices, result_format, torch.float64)
    dumps = {}
    for request in (None, (), (loop,)):
        scheduled = Scheduler.apply_schedule(cin, Schedule(vector_loops=request))
        plan = scheduled.verified_loop_plan
        if request is None:
            assert plan.vector_loops is None
        else:
            assert len(plan.vector_loops) == len(request)
            assert all(ref.part is LoopPart.LOGICAL for ref in plan.vector_loops)
        dumps[request] = canonical_plan_dump(scheduled.normalized_cin, plan)
    assert '"vector_loops"' not in dumps[None]
    assert '"vector_loops":[]' in dumps[()]
    assert len(set(dumps.values())) == 3
    # The plan without a decision has the identity it had before the field.
    plain = Scheduler.apply_schedule(cin, Schedule())
    assert (
        canonical_plan_dump(plain.normalized_cin, plain.verified_loop_plan)
        == dumps[None]
    )


def test_the_plan_refuses_a_loop_named_twice():
    (operands, result_indices, result_format), _ = VECTOR_CASES[0]
    cin = program_of(operands, result_indices, result_format, torch.float64)
    scheduled = Scheduler.apply_schedule(cin, Schedule())
    plan = scheduled.verified_loop_plan
    ref = LoopRef(plan.loop_order[-1])
    with pytest.raises(VerificationError, match="twice"):
        verify_loop_plan(
            scheduled.normalized_cin, dataclasses.replace(plan, vector_loops=(ref, ref))
        )


def test_the_program_records_erases_and_verifies_the_field():
    case, loop = VECTOR_CASES[0]
    # The scheduled program carries the field; ``lowering.program`` is the
    # program before the plan is applied.
    marked = new_kernel(case, 0, (loop,)).schedule.program
    default = new_kernel(case, 0, None).schedule.program
    assert default.vector_loops is None
    assert '"vector_loops"' not in canonical_program_dump(default)
    assert len(marked.vector_loops) == 1
    assert marked.vector_loops[0].part is VectorLoopPart.LOGICAL
    assert '"vector_loops"' in canonical_program_dump(marked)
    assert erase_schedule(marked).vector_loops is None

    builder = LoopIRBuilder.resuming(marked)
    index = marked.vector_loops[0].index
    refusals = [
        (
            (
                marked.vector_loops[0],
                builder.vector_loop(index, VectorLoopPart.LOGICAL),
            ),
            "invalid_vector_loops",
        ),
        ((builder.vector_loop(index, VectorLoopPart.INNER),), "invalid_vector_loops"),
        (("k",), "malformed_state"),
    ]
    for marks, code in refusals:
        with pytest.raises(LoopIRVerificationError) as error:
            verify_program(dataclasses.replace(marked, vector_loops=marks))
        assert error.value.defect.code == code, marks
    with pytest.raises(LoopIRVerificationError) as error:
        verify_program(dataclasses.replace(marked, vector_loops=[]))
    assert error.value.defect.code == "malformed_state"


def test_the_target_refuses_a_recorded_mark_it_does_not_place():
    """A verified program may name any loop it binds; the target refuses a loop
    its rule does not mark, here the outermost loop, with a code."""

    case, _ = VECTOR_CASES[0]
    operands, result_indices, _ = case
    program = new_kernel(case, 0, None).lowering.program
    loops = [
        node
        for node in _walk_declared_schema(program.body)
        if type(node) in (DenseFor, SparseFor, MergedSparseFor)
    ]
    first = loops[0]
    index = first.index if type(first) is DenseFor else first.coord_index
    builder = LoopIRBuilder.resuming(program)
    named = dataclasses.replace(
        program,
        vector_loops=(builder.vector_loop(index, VectorLoopPart.LOGICAL),),
    )
    verify_program(named)
    result_shape, _ = shapes_of(operands, result_indices, torch.float64)
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            named,
            input_shapes=input_shapes(program, operands),
            result_shape=result_shape,
        )
    assert error.value.defect.code == "unsupported_vector_loop"
