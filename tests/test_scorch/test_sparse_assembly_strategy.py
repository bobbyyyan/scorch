"""Sparse-output assembly strategy as a scheduling decision.

Which way a sparse result is assembled -- one traversal or two, distributed
across workers or not -- was decided by which lowering class hosted the program:
one virtual returning a constant, overridden by exactly one family for exactly
one receiver rank.  The measurements say the right choice varies with the
operands' formats and density
(``~/.cache/scorch-codex/assembly-strategy/DESIGN.md`` collects them), so it is a
scheduling decision and this file locks the properties that make it one.

Five properties, each a thing that would otherwise be a claim.

**The vocabulary and the receiver contract have ONE definition.**  Two copies of
one scheduling rule drifting apart is the defect that cost twelve cells when the
automatic tile heuristic was fixed in only one of its two layers, so the tokens
and the ``PARTITIONABLE`` predicate live in ``sparse_assembly`` and every layer
imports them.

**The automatic origin records no strategy, and the new compiler's assembly
stage completes its plan.**  The scheduler writes nothing into the field: its
plans are shared with the shipped compiler, and the request identity is computed
from them.  The stage that follows schedule application records a strategy for
every sparse result (``loopir/assembly_selection.py``): below
``2 * ROWS_PER_THREAD`` first-level coordinates the serial single pass, whose
kernel is the one these programs emitted before plans recorded a strategy; from
there a parallel strategy where the target can emit one -- the two-pass
parallel assembly, which is the shipped compiler's, before the chunked single
pass.  The rule has one definition (``loopir/assembly_rule.py``).  An explicit
plan that names no strategy and an unscheduled program get the same rule, with
the decision recorded on the kernel and the plan left as written, and so does
a program handed straight to target lowering.  No target family holds a default
of its own.

**Legality and cost are different predicates in different places.**  The
receiver contract is legality: structural, extent-free, refused with a code.  The
``2 * ROWS_PER_THREAD`` extent test is COST -- a 16-row receiver assembles
correctly from one chunk -- so it is part of the automatic rule and not of any
legality predicate; the rule is what a selector replaces.  An explicit request
at a small extent is therefore honoured.

**A request that cannot be honoured is REFUSED, never silently downgraded.**  A
caller who asks for one strategy and gets another has no way to find out.  Every
refusal carries a structured code, and the code distinguishes "this receiver
cannot be assembled this way" from "this host cannot emit it".

**The schemas moved, and both of them.**  A strategy selects which kernel one
program emits, so it is schedule content: the LoopIR canonical schema goes
v11 -> v12 and the plan canonical schema v1 -> v2, and the plan's canonical bytes
change for every plan.
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
from scorch.compiler.compilation_context import CompilationContext, CompilerStageId
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import InvalidSchedule, UnsupportedFeature
from scorch.compiler.loop_plan import LoopPlan, verify_loop_plan
from scorch.compiler.loopir import assembly_selection
from scorch.compiler.loopir import lower_llir as lower_llir_module
from scorch.compiler.loopir import pipeline as pipeline_module
from scorch.compiler.loopir.assembly_selection import (
    AUTOMATIC_PARALLEL_PREFERENCE,
    PARALLEL_FROM_EXTENT,
    AssemblyDecision,
    automatic_assembly,
    complete_automatic_plan,
)
from scorch.compiler.loopir.nodes import AssemblyStrategy
from scorch.compiler.loopir.parallel_chunk_assembly import ROWS_PER_THREAD
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir
from scorch.compiler.loopir.schedule_passes import apply_schedule_plan
from scorch.compiler.loopir.plan_identity import (
    CANONICAL_PLAN_SCHEMA,
    canonical_plan_dump,
)
from scorch.compiler.loopir.printer import CANONICAL_SCHEMA
from scorch.compiler.scheduler import Schedule, Scheduler
from scorch.compiler.sparse_assembly import (
    DEFAULT_SERIAL_STRATEGY,
    PARALLEL_ASSEMBLY_STRATEGIES,
    SINGLE_PASS_STRATEGIES,
    SPARSE_ASSEMBLY_STRATEGIES,
    TWO_PASS_STRATEGIES,
    compressed_levels_of,
    is_assembly_strategy,
    partitionable_receiver_levels,
)
from scorch.format import LevelType

_F32 = torch.float32
_D = LevelType.DENSE
_S = LevelType.COMPRESSED
_O = LevelType.COORDINATE


def auto_options(regblock_enabled=False, *, assembly=None, jit=False):
    base = (
        CompileOptions.from_environment()
        if jit
        else CompileOptions.from_environment(environ={})
    )
    return replace(
        base.with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(assembly=assembly),
    )


def ttm_cin(a_fmt, b_fmt, c_fmt, dtype=_F32):
    """``C[i,j,l] += A[i,j,k] * B[k,l]`` -- the family the strategies were measured on."""

    # ``l`` is the contraction's own index name, so E741 is silenced rather than
    # renamed: every other TTM site in the tree spells it the same way.
    i, j, k, l = (IndexVar(name) for name in ("i", "j", "k", "l"))  # noqa: E741
    a = TensorVar("A", fmt=a_fmt, dtype=dtype)[i, j, k]
    b = TensorVar("B", fmt=b_fmt, dtype=dtype)[k, l]
    c = TensorVar("C", fmt=c_fmt, dtype=dtype)[i, j, l]
    statement = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for variable in (l, k, j, i):
        statement = ForAll(variable, statement)
    return statement


def reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices):
    ivars = {name: IndexVar(name) for name in operand_indices}
    operand = TensorVar("A", fmt=operand_fmt, dtype=_F32)[
        tuple(ivars[name] for name in operand_indices)
    ]
    result = TensorVar("C", fmt=result_fmt, dtype=_F32)[
        tuple(ivars[name] for name in result_indices)
    ]
    statement = TensorAssign(result, operand, op=Operation.ADD)
    for name in reversed(operand_indices):
        statement = ForAll(ivars[name], statement)
    return statement


# -- the vocabulary, defined once ---------------------------------------------


def test_the_four_strategies_partition_into_pass_structure_and_parallelism():
    """Four tokens, and the two axes each cover them exactly once.

    The four are an inventory, not a product of two independent flags: each
    parallel strategy emits its OWN runtime gate whose closed arm is not the
    corresponding serial strategy's kernel (a two-pass pragma at one thread still
    enters the region, measured at 4-10%).  So the axes are recorded as
    memberships over one enumeration rather than as a pair of booleans, which is
    what keeps a fifth nonsense state unrepresentable.
    """

    assert SPARSE_ASSEMBLY_STRATEGIES == (
        "single_pass_serial",
        "single_pass_chunk_parallel",
        "two_pass_serial",
        "two_pass_parallel",
    )
    assert set(SINGLE_PASS_STRATEGIES) | set(TWO_PASS_STRATEGIES) == set(
        SPARSE_ASSEMBLY_STRATEGIES
    )
    assert not set(SINGLE_PASS_STRATEGIES) & set(TWO_PASS_STRATEGIES)
    assert set(PARALLEL_ASSEMBLY_STRATEGIES) < set(SPARSE_ASSEMBLY_STRATEGIES)
    assert DEFAULT_SERIAL_STRATEGY == "single_pass_serial"
    assert all(is_assembly_strategy(token) for token in SPARSE_ASSEMBLY_STRATEGIES)
    assert not is_assembly_strategy("two_pass")
    assert not is_assembly_strategy(None)
    # The typed LoopIR enum carries exactly the same tokens, so the plan, the
    # public schedule and the program cannot disagree about what a strategy is.
    assert {member.value for member in AssemblyStrategy} == set(
        SPARSE_ASSEMBLY_STRATEGIES
    )


@pytest.mark.parametrize(
    "levels, partitionable",
    [
        ((_D, _S), True),
        ((_D, _S, _S), True),
        ((_D, _S, _S, _S), True),
        # A stored prefix: no dense extent to split, permanently illegal rather
        # than pending work.
        ((_S, _S), False),
        ((_S,), False),
        # A dense prefix deeper than one: the first compressed level's position
        # array is indexed by a FLATTENED dense cell, so the pre-size, per-worker
        # start and shift range become products of extents.  Declined here.
        ((_D, _D, _S), False),
        # Rank 1 has nothing to partition.
        ((_D,), False),
        ((_S,), False),
        # A level kind outside the executable two answers False rather than
        # defaulting to one of them.
        ((_D, _O), False),
    ],
)
def test_the_receiver_contract_is_one_predicate(levels, partitionable):
    assert partitionable_receiver_levels(levels) is partitionable
    assert compressed_levels_of(levels) == (
        tuple(range(1, len(levels))) if partitionable else ()
    )


def test_the_contract_matches_the_shared_two_phase_passs_own_requirement():
    """``compressed_levels_of`` produces exactly what the shared pass validates.

    The two-phase pass independently requires ``(1, ..., rank-1)``.  Deriving the
    tuple from the receiver in one place is what makes the two implementations of
    "this receiver can be split by outer cell" the same rule rather than two that
    happen to agree today.
    """

    for rank in (2, 3, 4, 5):
        levels = (_D,) + (_S,) * (rank - 1)
        assert compressed_levels_of(levels) == tuple(range(1, rank))


# -- the automatic origin records no strategy; the assembly stage does ---------


def test_the_automatic_origin_records_no_strategy():
    """The scheduler chooses nothing; the stage after schedule application does.

    The cost model never writes this field, so the plan the scheduler returns
    carries nothing and the request identity is computed from that plan.  The
    plan the kernel keeps is the completed one, and the kernel carries the
    record of the decision.
    """

    options = auto_options()
    scheduled = Scheduler.apply_schedule(
        ttm_cin("dss", "ss", "dss"), Schedule(), compile_options=options
    )
    assert scheduled.verified_loop_plan.assembly is None

    kernel = compile_cin_via_loopir(
        ttm_cin("dss", "ss", "dss"),
        (4, 5, 3),
        (((4, 5, 6), _F32), ((6, 3), _F32)),
        compile_options=options,
    )
    assert kernel.schedule is not None
    assert '"assembly":null' in kernel.request_dump
    assert kernel.schedule.plan.assembly == "single_pass_serial"
    assert kernel.schedule.program.assembly is AssemblyStrategy.SINGLE_PASS_SERIAL
    assert '"assembly":"single_pass_serial"' in kernel.program_dump
    assert kernel.assembly_decision == AssemblyDecision(
        host="_OrderedKeySparseWorkspaceLowering",
        sparse_result=True,
        emittable=SPARSE_ASSEMBLY_STRATEGIES,
        split_extent=4,
        parallel_from=2 * ROWS_PER_THREAD,
        strategy="single_pass_serial",
    )


def test_a_requested_strategy_reaches_the_program_as_a_typed_fact():
    kernel = compile_cin_via_loopir(
        ttm_cin("dss", "ss", "dss"),
        (4, 5, 3),
        (((4, 5, 6), _F32), ((6, 3), _F32)),
        compile_options=auto_options(assembly="single_pass_chunk_parallel"),
    )
    assert kernel.schedule is not None
    assert kernel.schedule.plan.assembly == "single_pass_chunk_parallel"
    assert (
        kernel.schedule.program.assembly is AssemblyStrategy.SINGLE_PASS_CHUNK_PARALLEL
    )
    assert '"assembly":"single_pass_chunk_parallel"' in kernel.program_dump


def test_a_schedule_carrying_only_a_strategy_is_not_silently_treated_as_empty():
    """The request must survive the empty-schedule/automatic delegation.

    ``Schedule()`` is the automatic marker, so a schedule carrying only
    ``assembly`` looks empty to every field-by-field test.  If the request were
    dropped there the caller would get the default kernel with no way to tell,
    which is the one failure mode this whole decision must not have.
    """

    plan_free = compile_cin_via_loopir(
        ttm_cin("dss", "ss", "dss"),
        (4, 5, 3),
        (((4, 5, 6), _F32), ((6, 3), _F32)),
        compile_options=auto_options(),
    )
    requested = compile_cin_via_loopir(
        ttm_cin("dss", "ss", "dss"),
        (4, 5, 3),
        (((4, 5, 6), _F32), ((6, 3), _F32)),
        compile_options=auto_options(assembly="single_pass_chunk_parallel"),
    )
    # Four rows: the stage records the serial single pass for the plan that
    # asks for nothing, and does not run for the plan that asks.
    assert plan_free.schedule.plan.assembly == "single_pass_serial"
    assert plan_free.assembly_decision is not None
    assert requested.schedule.plan.assembly == "single_pass_chunk_parallel"
    assert requested.assembly_decision is None


# -- legality is not cost -----------------------------------------------------


def test_the_extent_test_is_cost_and_no_longer_gates_legality():
    """A receiver too small to reach two threads is still LEGAL.

    ``scorch_nthreads``'s row term alone caps the count at one below
    ``2 * ROWS_PER_THREAD``, so the automatic rule records a serial strategy
    there -- that is a judgement about whether a parallel assembly pays, and it
    keeps those kernels byte-identical.  It is not a claim that the program
    cannot be assembled in chunks, so an explicit request is honoured and emits
    the gate.  Keeping the two answers apart is what lets a selector replace the
    cost half without touching correctness.  At the threshold the rule records
    the two-pass parallel assembly, the shipped compiler's strategy, which this
    program can also be emitted with; the chunked single pass is then reached
    by request only.
    """

    rows = 2 * ROWS_PER_THREAD
    small, large = rows - 1, rows

    def emit(outer, assembly):
        return compile_cin_via_loopir(
            ttm_cin("dss", "ss", "dss"),
            (outer, 5, 3),
            (((outer, 5, 6), _F32), ((6, 3), _F32)),
            compile_options=auto_options(assembly=assembly),
        ).cpp_source

    # Cost: below the threshold the automatic choice is the serial builder.  At
    # the threshold it is the two-pass parallel assembly, not the chunked one.
    assert emit(small, None) == emit(small, "single_pass_serial")
    assert emit(large, None) == emit(large, "two_pass_parallel")
    assert "_assembly_chunks" not in emit(large, None)
    # Legality: the request is honoured on BOTH sides of that threshold.
    assert "_assembly_chunks" in emit(small, "single_pass_chunk_parallel")
    assert "_assembly_chunks" in emit(large, "single_pass_chunk_parallel")
    # And requesting the serial strategy at a large extent gets the serial
    # builder, which is the same kernel the small extent chose by default.
    assert "_assembly_chunks" not in emit(large, "single_pass_serial")


def test_requesting_the_serial_strategy_emits_the_ungated_nest():
    """``single_pass_serial`` at a chunk-capable extent is the base's kernel.

    The chunked strategy's serial arm is that nest verbatim, so the two must
    agree modulo the gate -- and an explicit serial request is how a measurement
    isolates the strategy from the gate without editing emitted text, which is
    how the ``legacy_serial`` column had to be built before this existed.
    """

    rows = 2 * ROWS_PER_THREAD

    def emit(assembly):
        return compile_cin_via_loopir(
            ttm_cin("dss", "ss", "dss"),
            (rows, 5, 3),
            (((rows, 5, 6), _F32), ((6, 3), _F32)),
            compile_options=auto_options(assembly=assembly),
        ).cpp_source

    serial = emit("single_pass_serial")
    assert "#pragma omp" not in serial
    assert "_assembly_threads" not in serial
    gated = emit("single_pass_chunk_parallel")
    assert gated.count("#pragma omp") == 1
    assert "_assembly_threads" in gated
    # The gate's else arm is the serial kernel's nest: every statement of the
    # serial emission's loop body appears in the gated emission too.
    body = re.search(r"for \(int64_t i = 0; i < A0_size; i\+\+\) \{", serial)
    assert body is not None
    assert "for (int64_t i = 0; i < A0_size; i++) {" in gated


# -- refusals are structured, and they name what refused ----------------------


@pytest.mark.parametrize("strategy", SPARSE_ASSEMBLY_STRATEGIES[1:])
def test_a_non_partitionable_receiver_refuses_every_non_serial_strategy(strategy):
    """``unsupported_schedule_assembly``, at the plan boundary, with a code.

    ``ss ij->j [s]`` has a rank-1 compressed receiver: there is no dense level
    zero to partition, so no strategy that splits or distributes the assembly can
    apply.  The refusal is a structured diagnostic rather than a bare exception
    because "zero unclassified over the frontier" is a gate.
    """

    with pytest.raises((UnsupportedFeature, InvalidSchedule)) as raised:
        compile_cin_via_loopir(
            reduction_cin("ss", "s", "ij", "j"),
            (3,),
            (((2, 3), _F32),),
            compile_options=auto_options(assembly=strategy),
        )
    codes = {
        getattr(diagnostic, "code", None)
        for diagnostic in getattr(raised.value, "diagnostics", ())
    }
    assert "unsupported_schedule_assembly" in codes


def test_a_stored_outer_loop_refuses_the_chunked_strategy_at_the_target():
    """``unsupported_assembly_strategy``: legal receiver, wrong program.

    ``sss ijk->ik [ds]`` HAS a partitionable ``ds`` receiver, so the plan
    boundary admits the request.  Its outermost loop is a stored stream over the
    operand's compressed level zero, whose coordinates do not partition the dense
    result extent, so the program cannot express the chunk partition.  The two
    halves of legality are proved in the two layers that own their inputs, and
    they carry different codes so the reason is never guessed.
    """

    with pytest.raises(lower_llir_module.LoopIRTargetError) as raised:
        compile_cin_via_loopir(
            reduction_cin("sss", "ds", "ijk", "ik"),
            (2, 4),
            (((2, 3, 4), _F32),),
            compile_options=auto_options(assembly="single_pass_chunk_parallel"),
        )
    assert raised.value.defect.code == "unsupported_assembly_strategy"


def test_a_host_that_cannot_emit_a_legal_strategy_says_so_by_name():
    """``unsupported_assembly_host``, never a silent downgrade.

    ``MM ds x ds -> ds`` is hosted by the two-phase family, whose own completion
    validator requires the assembled function to carry the two-phase parallel
    shape -- so the single-pass strategies are legal for that receiver and that
    family cannot emit them.  Naming the host is the difference between a refusal
    a caller can act on and an internal completion failure.
    """

    with pytest.raises(lower_llir_module.LoopIRTargetError) as raised:
        compile_cin_via_loopir(
            _matmul_cin("ds", "ds", "ds"),
            (2, 4),
            (((2, 3), _F32), ((3, 4), _F32)),
            compile_options=auto_options(assembly="single_pass_serial"),
        )
    assert raised.value.defect.code == "unsupported_assembly_host"
    assert "_ParallelSparseWorkspaceLowering" in raised.value.defect.message


def _ttm_second_mode_cin():
    """``C[i,l,k] += A[i,j,k] * B[j,l]``: a contraction of the middle mode."""

    i, j, k, l = (IndexVar(name) for name in ("i", "j", "k", "l"))  # noqa: E741
    a = TensorVar("A", fmt="dss", dtype=_F32)[i, j, k]
    b = TensorVar("B", fmt="dd", dtype=_F32)[j, l]
    c = TensorVar("C", fmt="dss", dtype=_F32)[i, l, k]
    statement = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for variable in (k, l, j, i):
        statement = ForAll(variable, statement)
    return statement


def _matmul_cin(a_fmt, b_fmt, c_fmt):
    i, j, k = (IndexVar(name) for name in ("i", "j", "k"))
    a = TensorVar("A", fmt=a_fmt, dtype=_F32)[i, j]
    b = TensorVar("B", fmt=b_fmt, dtype=_F32)[j, k]
    c = TensorVar("C", fmt=c_fmt, dtype=_F32)[i, k]
    statement = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for variable in (k, j, i):
        statement = ForAll(variable, statement)
    return statement


# -- what the target can emit, asked before a strategy is chosen --------------


_REFUSAL_CODES = {
    "unsupported_schedule_assembly",
    "unsupported_assembly_strategy",
    "unsupported_assembly_host",
}

# (program, result shape, operand shapes by name, host, strategies emitted)
_CHOICE_CASES = {
    # A dense outer loop over the result's dense level, and a position array
    # indexed by its bound: every strategy.
    "dense outer loop": (
        lambda: ttm_cin("dss", "ss", "dss"),
        (4, 5, 3),
        {"A": (4, 5, 6), "B": (6, 3)},
        "_OrderedKeySparseWorkspaceLowering",
        SPARSE_ASSEMBLY_STRATEGIES,
    ),
    # A stored outer loop does not partition the dense extent, so no chunks.
    "stored outer loop": (
        lambda: reduction_cin("sss", "ds", "ijk", "ik"),
        (2, 4),
        {"A": (2, 3, 4)},
        "_OrderedKeySparseWorkspaceLowering",
        ("single_pass_serial", "two_pass_serial", "two_pass_parallel"),
    ),
    # A dense outer loop whose body's first position array hangs from TWO dense
    # levels: the chunked strategy is legal by shape and its gate has no
    # estimate of the stored entries to open on.
    "no estimate for the gate": (
        lambda: ttm_cin("dds", "dd", "dss"),
        (4, 5, 3),
        {"A": (4, 5, 6), "B": (6, 3)},
        "_OrderedKeySparseWorkspaceLowering",
        ("single_pass_serial", "two_pass_serial", "two_pass_parallel"),
    ),
    # A drain that assembles two compressed levels: the two-pass strategies
    # decline that shape, so the chunked single pass is the only parallel one.
    "only the chunked pass is parallel": (
        _ttm_second_mode_cin,
        (4, 3, 6),
        {"A": (4, 5, 6), "B": (5, 3)},
        "_OrderedKeySparseWorkspaceLowering",
        ("single_pass_serial", "single_pass_chunk_parallel"),
    ),
    # The two-phase family emits its own strategy only.
    "two-phase family": (
        lambda: _matmul_cin("ds", "ds", "ds"),
        (2, 4),
        {"A": (2, 3), "B": (3, 4)},
        "_ParallelSparseWorkspaceLowering",
        ("two_pass_parallel",),
    ),
    # A family that appends in loop order has no workspace; the shared two-phase
    # pass turns its nest into a count loop and a fill loop.
    "ordered-append family": (
        lambda: reduction_cin("dss", "ds", "ijk", "ij"),
        (2, 3),
        {"A": (2, 3, 4)},
        "_MultiCompressedAssemblyLowering",
        ("single_pass_serial", "two_pass_serial", "two_pass_parallel"),
    ),
    # A coordinate result has no position array to fill from per-row counts.
    "coordinate result": (
        lambda: reduction_cin("dss", "oo", "ijk", "ij"),
        (2, 3),
        {"A": (2, 3, 4)},
        "_CoordinateResultAssemblyLowering",
        ("single_pass_serial",),
    ),
    # No dense level zero to split by.
    "compressed level zero": (
        lambda: reduction_cin("ss", "s", "ij", "j"),
        (3,),
        {"A": (2, 3)},
        "_OrderedKeySparseWorkspaceLowering",
        ("single_pass_serial",),
    ),
}


def _compile_choice_case(case, assembly=None):
    program, result_shape, shapes, _, _ = _CHOICE_CASES[case]
    return compile_cin_via_loopir(
        program(),
        result_shape,
        tuple((shapes[name], _F32) for name in sorted(shapes)),
        compile_options=auto_options(assembly=assembly),
    )


def _choices(case):
    """The case's scheduled program before a strategy is recorded, and the target's
    answer for it."""

    _, result_shape, shapes, _, _ = _CHOICE_CASES[case]
    schedule = _compile_choice_case(case).schedule
    program = apply_schedule_plan(
        schedule.base_program, replace(schedule.plan, assembly=None)
    ).program
    assert program.assembly is None
    return program, lower_llir_module.assembly_choices(
        program,
        input_shapes={
            decl.symbol: shapes[decl.name]
            for decl in program.tensors
            if decl.symbol in program.inputs
        },
        result_shape=result_shape,
    )


@pytest.mark.parametrize("case", sorted(_CHOICE_CASES))
def test_the_target_reports_exactly_the_strategies_a_request_emits(case):
    """``assembly_choices`` is the list a scheduler may choose from.

    A scheduler that records a strategy needs to know, before it records one,
    which strategies this program is emitted with: a recorded strategy outside
    the list is refused, and the automatic route must not turn a program the
    serial builder handles into a refusal.  So the list is checked against what
    a request for each strategy does, on one program per kind of answer.
    """

    _, _, _, host, expected = _CHOICE_CASES[case]
    _, choices = _choices(case)
    assert choices.host == host
    assert choices.emittable == expected
    assert choices.sparse_result
    for strategy in SPARSE_ASSEMBLY_STRATEGIES:
        try:
            _compile_choice_case(case, strategy)
        except (
            lower_llir_module.LoopIRTargetError,
            UnsupportedFeature,
            InvalidSchedule,
        ) as refusal:
            codes = {
                getattr(diagnostic, "code", None)
                for diagnostic in getattr(refusal, "diagnostics", ())
            } | {getattr(getattr(refusal, "defect", None), "code", None)}
            assert codes & _REFUSAL_CODES, (strategy, refusal)
            assert strategy not in choices.emittable, strategy
        else:
            assert strategy in choices.emittable, strategy


def test_a_requested_chunked_strategy_with_no_gate_estimate_is_refused():
    """``unsupported_assembly_strategy``, where the serial kernel used to come back.

    The chunked builder opens its parallel arm on ``scorch_nthreads(work,
    rows)``, with ``work`` the stored entries under the outermost loop: the
    first position array of the loop body, read at the loop's bound.  In
    ``TTM dds x dd -> dss`` that array hangs from two dense levels, so it is not
    indexed by the outer loop's bound and the builder declines.  For a request,
    declining returned the serial kernel under the requested strategy's name;
    it is refused now.  The automatic plan does not record the chunked strategy
    for this program, because the target does not list it: it records the
    two-pass parallel assembly, whose thread count needs no such estimate.
    """

    rows = 2 * ROWS_PER_THREAD

    def emit(assembly):
        return compile_cin_via_loopir(
            ttm_cin("dds", "dd", "dss"),
            (rows, 5, 3),
            (((rows, 5, 6), _F32), ((6, 3), _F32)),
            compile_options=auto_options(assembly=assembly),
        ).cpp_source

    with pytest.raises(lower_llir_module.LoopIRTargetError) as raised:
        emit("single_pass_chunk_parallel")
    assert raised.value.defect.code == "unsupported_assembly_strategy"
    assert "the estimate cannot be derived" in raised.value.defect.message
    assert emit(None) == emit("two_pass_parallel")


def test_the_choices_are_asked_before_a_strategy_is_recorded():
    """A program that already records a strategy is not asked what it could be."""

    program, _ = _choices("dense outer loop")
    _, result_shape, shapes, _, _ = _CHOICE_CASES["dense outer loop"]
    with pytest.raises(lower_llir_module.LoopIRTargetError) as raised:
        lower_llir_module.assembly_choices(
            replace(program, assembly=AssemblyStrategy.SINGLE_PASS_SERIAL),
            input_shapes={
                decl.symbol: shapes[decl.name]
                for decl in program.tensors
                if decl.symbol in program.inputs
            },
            result_shape=result_shape,
        )
    assert raised.value.defect.code == "invalid_schedule_assembly"


def _compile_with_rows(case, rows, assembly=None):
    """The case's program with ``rows`` coordinates along the result's first level."""

    program, result_shape, shapes, _, _ = _CHOICE_CASES[case]
    shapes = dict(shapes, A=(rows,) + shapes["A"][1:])
    return compile_cin_via_loopir(
        program(),
        (rows,) + result_shape[1:],
        tuple((shapes[name], _F32) for name in sorted(shapes)),
        compile_options=auto_options(assembly=assembly),
    )


@pytest.mark.parametrize(
    ("case", "chosen"),
    [
        ("dense outer loop", "two_pass_parallel"),
        ("stored outer loop", "two_pass_parallel"),
        ("no estimate for the gate", "two_pass_parallel"),
        ("only the chunked pass is parallel", "single_pass_chunk_parallel"),
        ("ordered-append family", "two_pass_parallel"),
    ],
)
def test_the_automatic_plan_records_a_parallel_strategy_from_32_rows(case, chosen):
    """The choice is the plan's, made from what the target can emit.

    From ``2 * ROWS_PER_THREAD`` first-level coordinates the automatic plan
    records the two-pass parallel assembly -- the shipped compiler's strategy
    for these results -- wherever the target can emit it, the chunked single
    pass included among the alternatives or not, and the chunked single pass
    only where it is the one parallel strategy the program has.  The kernel is
    the one a request for that strategy emits.  One coordinate below, the plan
    records the serial single pass, and the kernel is the serial builder these
    programs emitted before the plan chose at all.
    """

    rows = 2 * ROWS_PER_THREAD
    assert rows == PARALLEL_FROM_EXTENT
    automatic = _compile_with_rows(case, rows)
    assert automatic.schedule.plan.assembly == chosen
    assert automatic.schedule.program.assembly is AssemblyStrategy(chosen)
    assert automatic.assembly_decision.strategy == chosen
    assert automatic.assembly_decision.split_extent == rows
    assert "#pragma omp" in automatic.cpp_source
    assert automatic.cpp_source == _compile_with_rows(case, rows, chosen).cpp_source

    below = _compile_with_rows(case, rows - 1)
    assert below.schedule.plan.assembly == "single_pass_serial"
    assert below.schedule.program.assembly is AssemblyStrategy.SINGLE_PASS_SERIAL
    assert below.assembly_decision.emittable == automatic.assembly_decision.emittable
    assert "#pragma omp" not in below.cpp_source
    assert (
        below.cpp_source
        == _compile_with_rows(case, rows - 1, "single_pass_serial").cpp_source
    )


def _cpp_through_the_target_entry(program, shapes, result_shape):
    """The C++ of ``program`` handed to target lowering directly, as it is."""

    from scorch.ops import _lower_generated_llir

    options = auto_options()
    context = CompilationContext(options)
    function = lower_llir_module.lower_loopir_to_llir(
        program,
        input_shapes={
            decl.symbol: shapes[decl.name]
            for decl in program.tensors
            if decl.symbol in program.inputs
        },
        result_shape=result_shape,
        compile_options=options,
        compilation_context=context,
    )
    return _lower_generated_llir(function, options, context)


@pytest.mark.parametrize("rows", [PARALLEL_FROM_EXTENT - 1, PARALLEL_FROM_EXTENT])
@pytest.mark.parametrize(
    "case", sorted(case for case in _CHOICE_CASES if case != "compressed level zero")
)
def test_a_program_without_a_strategy_is_lowered_with_the_rules(case, rows):
    """One rule, whichever way a program reaches target lowering.

    The stage records the rule's strategy in an automatic plan.  A caller can
    also hand target lowering a program that records none -- a scheduled
    program whose plan names no strategy.  The target entry then applies the
    same rule to the same two facts, so the kernel is the one the automatic
    plan compiled to, on both sides of the extent the rule turns parallel at.
    Before the rule had one definition each family answered for itself there,
    and the ordered-key family's answer was never a two-pass strategy.
    """

    program, result_shape, shapes, _, _ = _CHOICE_CASES[case]
    shapes = dict(shapes, A=(rows,) + shapes["A"][1:])
    result_shape = (rows,) + result_shape[1:]
    kernel = _compile_with_rows(case, rows)
    schedule = kernel.schedule
    unrecorded = apply_schedule_plan(
        schedule.base_program, replace(schedule.plan, assembly=None)
    ).program
    assert unrecorded.assembly is None
    assert (
        _cpp_through_the_target_entry(unrecorded, shapes, result_shape)
        == kernel.cpp_source
    )
    recorded, choices = lower_llir_module.with_recorded_assembly(
        unrecorded,
        input_shapes={
            decl.symbol: shapes[decl.name]
            for decl in unrecorded.tensors
            if decl.symbol in unrecorded.inputs
        },
        result_shape=result_shape,
    )
    assert unrecorded.assembly is None, "the caller's program is not changed"
    assert recorded.assembly.value == kernel.assembly_decision.strategy
    assert choices.emittable == kernel.assembly_decision.emittable
    # A program that already records a strategy is returned as it is.
    again, unasked = lower_llir_module.with_recorded_assembly(
        recorded,
        input_shapes={
            decl.symbol: shapes[decl.name]
            for decl in recorded.tensors
            if decl.symbol in recorded.inputs
        },
        result_shape=result_shape,
    )
    assert again is recorded and unasked is None


def test_no_target_family_answers_for_itself():
    """The rule is the only place an unrequested strategy is chosen.

    A target family used to carry a default of its own (the ordered-key
    family's read the first extent), which an unscheduled program and an
    explicit plan reached while an automatic plan got the rule.  No family
    defines one now, and what a lowering assumes of a program without a
    strategy is one method on the base class that reads no extent.
    """

    def families(cls):
        for child in cls.__subclasses__():
            yield child
            yield from families(child)

    base = lower_llir_module._TargetLowering
    assert not hasattr(base, "default_assembly")
    for family in families(base):
        assert "default_assembly" not in vars(family), family.__name__
        assert "unrecorded_assembly" not in vars(family), family.__name__
    assert lower_llir_module.automatic_assembly is automatic_assembly
    assert assembly_selection.automatic_assembly is automatic_assembly


@pytest.mark.parametrize("route", ["unscheduled", "explicit plan"])
@pytest.mark.parametrize("rows", [PARALLEL_FROM_EXTENT - 1, 4 * ROWS_PER_THREAD])
def test_the_stage_records_its_decision_on_every_route(route, rows):
    """An unscheduled program and an explicit plan get the rule and a record.

    Neither has an automatic plan to complete: the plan, where there is one, is
    left as its caller wrote it and the program the kernel shows records no
    strategy.  The decision record is where the strategy is written down, and
    the kernel is the one the automatic plan compiles to: the serial single pass
    below ``2 * ROWS_PER_THREAD`` rows and two parallel passes from there.
    (``dss ijk->ij [ds]`` is lowered by the ordered-append family on all three
    routes; the workspace families are reached by the automatic schedule only.)
    """

    case = "ordered-append family"
    program, result_shape, shapes, host, _ = _CHOICE_CASES[case]
    shapes = dict(shapes, A=(rows,) + shapes["A"][1:])
    options = CompileOptions.from_environment(environ={}).with_regblock_enabled(False)
    if route == "explicit plan":
        options = replace(
            options, requested_schedule=Schedule(loop_order=("i", "j", "k"))
        )
    context = CompilationContext(options)
    kernel = compile_cin_via_loopir(
        program(),
        (rows,) + result_shape[1:],
        tuple((shapes[name], _F32) for name in sorted(shapes)),
        compile_options=options,
        compilation_context=context,
    )
    decision = kernel.assembly_decision
    assert decision is not None
    assert decision.host == host
    assert decision.strategy == (
        "two_pass_parallel" if rows >= PARALLEL_FROM_EXTENT else "single_pass_serial"
    )
    assert decision.split_extent == rows
    assert '"assembly":null' in kernel.program_dump
    if route == "explicit plan":
        assert kernel.schedule.plan.provenance == "explicit"
        assert kernel.schedule.plan.assembly is None
        assert kernel.schedule.program.assembly is None
    else:
        assert kernel.schedule is None
    stages = [record.stage_id for record in context.stage_run_records]
    at = stages.index(CompilerStageId.LOOPIR_ASSEMBLY_SELECTION)
    assert stages[at + 1] is CompilerStageId.LOOPIR_TO_LLIR_LOWERING
    assert stages.count(CompilerStageId.LOOPIR_ASSEMBLY_SELECTION) == 1
    assert kernel.cpp_source == _compile_with_rows(case, rows).cpp_source


def test_the_automatic_plan_records_the_serial_pass_where_no_parallel_one_is_emitted():
    """A family that emits the serial single pass only records exactly that.

    ``dss ijk->ij [oo]`` assembles a coordinate result, which has no parallel
    assembly, so at any extent the plan records the serial single pass and the
    kernel is the serial one.
    """

    rows = 4 * ROWS_PER_THREAD
    kernel = _compile_with_rows("coordinate result", rows)
    assert kernel.schedule.plan.assembly == "single_pass_serial"
    assert kernel.assembly_decision.emittable == ("single_pass_serial",)
    assert kernel.assembly_decision.split_extent == rows
    assert "#pragma omp" not in kernel.cpp_source


def test_erasing_the_schedule_erases_the_recorded_strategy():
    """A recorded strategy is schedule state, like a parallel selection.

    ``dss ijk->ij [ds]`` has no tile, panel or workspace region, so its scheduled
    program differs from the base program only in the strategy the assembly
    stage recorded; the erased program is the base program.
    """

    from scorch.compiler.loopir.printer import canonical_program_dump
    from scorch.compiler.loopir.schedule_passes import erase_schedule

    kernel = _compile_choice_case("ordered-append family")
    schedule = kernel.schedule
    assert schedule.program.assembly is AssemblyStrategy.SINGLE_PASS_SERIAL
    assert schedule.base_program.assembly is None
    erased = erase_schedule(schedule.program)
    assert erased.assembly is None
    assert canonical_program_dump(erased) == canonical_program_dump(
        schedule.base_program
    )
    assert erase_schedule(schedule.base_program) is schedule.base_program


def test_a_family_with_one_strategy_records_it_at_every_extent():
    """The two-phase family emits the two-pass parallel assembly only.

    Two rows is far below the extent from which the rule prefers a parallel
    strategy, and the serial single pass is not among the strategies the target
    lists, so the plan records the family's one strategy.  The kernel is the
    one the family emitted when the plan recorded nothing.
    """

    kernel = _compile_choice_case("two-phase family")
    assert kernel.assembly_decision.emittable == ("two_pass_parallel",)
    assert kernel.assembly_decision.split_extent == 2
    assert kernel.schedule.plan.assembly == "two_pass_parallel"
    assert (
        kernel.cpp_source
        == _compile_choice_case("two-phase family", "two_pass_parallel").cpp_source
    )
    schedule = kernel.schedule
    unrecorded = apply_schedule_plan(
        schedule.base_program, replace(schedule.plan, assembly=None)
    ).program
    _, result_shape, shapes, _, _ = _CHOICE_CASES["two-phase family"]
    options = auto_options()
    context = CompilationContext(options)
    function = lower_llir_module.lower_loopir_to_llir(
        unrecorded,
        input_shapes={
            decl.symbol: shapes[decl.name]
            for decl in unrecorded.tensors
            if decl.symbol in unrecorded.inputs
        },
        result_shape=result_shape,
        compile_options=options,
        compilation_context=context,
    )
    from scorch.ops import _lower_generated_llir

    assert _lower_generated_llir(function, options, context) == kernel.cpp_source


@pytest.mark.parametrize(
    ("emittable", "extent", "expected"),
    [
        (SPARSE_ASSEMBLY_STRATEGIES, PARALLEL_FROM_EXTENT, "two_pass_parallel"),
        (
            ("single_pass_serial", "single_pass_chunk_parallel"),
            PARALLEL_FROM_EXTENT,
            "single_pass_chunk_parallel",
        ),
        (SPARSE_ASSEMBLY_STRATEGIES, PARALLEL_FROM_EXTENT - 1, "single_pass_serial"),
        (SPARSE_ASSEMBLY_STRATEGIES, 0, "single_pass_serial"),
        (SPARSE_ASSEMBLY_STRATEGIES, None, "single_pass_serial"),
        (
            ("single_pass_serial", "two_pass_serial", "two_pass_parallel"),
            PARALLEL_FROM_EXTENT,
            "two_pass_parallel",
        ),
        (("single_pass_serial",), 10 * PARALLEL_FROM_EXTENT, "single_pass_serial"),
        (("two_pass_serial", "two_pass_parallel"), 1, "two_pass_serial"),
        (("two_pass_parallel",), 1, "two_pass_parallel"),
        (
            ("single_pass_chunk_parallel", "two_pass_parallel"),
            1,
            "single_pass_chunk_parallel",
        ),
    ],
)
def test_the_rule_answers_for_every_set_of_strategies(emittable, extent, expected):
    """The rule is total: any non-empty set of strategies and any extent."""

    assert automatic_assembly(emittable, extent) == expected


def test_the_rule_answers_for_every_subset_of_the_vocabulary():
    """Every non-empty subset, on both sides of the extent, gets a member of it."""

    count = len(SPARSE_ASSEMBLY_STRATEGIES)
    for mask in range(1, 1 << count):
        emittable = tuple(
            strategy
            for bit, strategy in enumerate(SPARSE_ASSEMBLY_STRATEGIES)
            if mask >> bit & 1
        )
        for extent in (None, 0, PARALLEL_FROM_EXTENT - 1, PARALLEL_FROM_EXTENT, 4096):
            chosen = automatic_assembly(emittable, extent)
            assert chosen in emittable
            parallel = [s for s in AUTOMATIC_PARALLEL_PREFERENCE if s in emittable]
            if extent is not None and extent >= PARALLEL_FROM_EXTENT and parallel:
                assert chosen == parallel[0]
    with pytest.raises(ValueError):
        automatic_assembly((), PARALLEL_FROM_EXTENT)


def test_a_dense_result_records_no_strategy():
    """The strategies describe a sparse assembly; a dense result has none.

    The pipeline does not begin the assembly stage for it, so its kernel has no
    decision record and its stage list is the one it had before the stage
    existed.  Asked directly, the stage's function reports what the target said
    and returns the plan it was given.
    """

    options = auto_options()
    context = CompilationContext(options)
    kernel = compile_cin_via_loopir(
        _matmul_cin("ds", "dd", "dd"),
        (64, 4),
        (((64, 3), _F32), ((3, 4), _F32)),
        compile_options=options,
        compilation_context=context,
    )
    assert kernel.schedule.plan.assembly is None
    assert kernel.schedule.program.assembly is None
    assert kernel.assembly_decision is None
    assert CompilerStageId.LOOPIR_ASSEMBLY_SELECTION not in [
        record.stage_id for record in context.stage_run_records
    ]

    schedule = kernel.schedule
    completed, decision = complete_automatic_plan(
        schedule,
        schedule.plan,
        input_shapes={
            decl.symbol: {"A": (64, 3), "B": (3, 4)}[decl.name]
            for decl in schedule.program.tensors
            if decl.symbol in schedule.program.inputs
        },
        result_shape=(64, 4),
    )
    assert completed is schedule.plan
    assert not decision.sparse_result
    assert decision.strategy is None and decision.split_extent == 64


def test_the_decision_record_is_canonical_text():
    kernel = _compile_with_rows("stored outer loop", PARALLEL_FROM_EXTENT)
    assert kernel.assembly_decision.dump() == (
        '{"emittable":["single_pass_serial","two_pass_serial","two_pass_parallel"],'
        '"host":"_OrderedKeySparseWorkspaceLowering","parallel_from":32,'
        '"sparse_result":true,"split_extent":32,"strategy":"two_pass_parallel"}'
    )


def test_the_assembly_stage_is_recorded_between_schedule_and_target_lowering():
    """Its own stage, begun only for an automatic plan that records no strategy."""

    def stages(assembly):
        options = auto_options(assembly=assembly)
        context = CompilationContext(options)
        compile_cin_via_loopir(
            ttm_cin("dss", "ss", "dss"),
            (4, 5, 3),
            (((4, 5, 6), _F32), ((6, 3), _F32)),
            compile_options=options,
            compilation_context=context,
        )
        return [record.stage_id for record in context.stage_run_records]

    automatic = stages(None)
    at = automatic.index(CompilerStageId.LOOPIR_ASSEMBLY_SELECTION)
    assert automatic[at - 1] is CompilerStageId.LOOPIR_SCHEDULE_APPLICATION
    assert automatic[at + 1] is CompilerStageId.LOOPIR_TO_LLIR_LOWERING
    assert automatic.count(CompilerStageId.LOOPIR_ASSEMBLY_SELECTION) == 1
    requested = stages("single_pass_serial")
    assert CompilerStageId.LOOPIR_ASSEMBLY_SELECTION not in requested
    assert requested == [
        stage
        for stage in automatic
        if stage is not CompilerStageId.LOOPIR_ASSEMBLY_SELECTION
    ]


def test_the_completed_plan_is_verified_before_it_is_applied(monkeypatch):
    """The stage's plan is a new plan and takes the plan verifier's refusals.

    The verifier sees the completed plan, conversions included, before the
    schedule passes apply it; a refusal fails the assembly stage and nothing
    is lowered.
    """

    seen = []
    real_verify = pipeline_module.verify_loop_plan
    real_apply = pipeline_module.apply_schedule_plan

    def verify(cin, plan, **keywords):
        seen.append(("verify", plan.assembly, keywords))
        return real_verify(cin, plan, **keywords)

    def apply(program, plan):
        seen.append(("apply", plan.assembly, {}))
        return real_apply(program, plan)

    monkeypatch.setattr(pipeline_module, "verify_loop_plan", verify)
    monkeypatch.setattr(pipeline_module, "apply_schedule_plan", apply)
    kernel = _compile_with_rows("stored outer loop", PARALLEL_FROM_EXTENT)
    assert kernel.schedule.plan.assembly == "two_pass_parallel"
    assert [entry[:2] for entry in seen if entry[0] == "apply"] == [
        ("apply", None),
        ("apply", "two_pass_parallel"),
    ]
    completed = [n for n, entry in enumerate(seen) if entry[1] == "two_pass_parallel"]
    assert [seen[n][0] for n in completed] == ["verify", "apply"]
    assert seen[completed[0]][2] == {"dense_results_any_order": True}

    def refuse(cin, plan, **keywords):
        if plan.assembly is not None:
            raise RuntimeError("refused by the verifier")
        return real_verify(cin, plan, **keywords)

    monkeypatch.setattr(pipeline_module, "verify_loop_plan", refuse)
    options = auto_options()
    context = CompilationContext(options)
    program, result_shape, shapes, _, _ = _CHOICE_CASES["stored outer loop"]
    shapes = dict(shapes, A=(PARALLEL_FROM_EXTENT,) + shapes["A"][1:])
    with pytest.raises(RuntimeError, match="refused by the verifier"):
        compile_cin_via_loopir(
            program(),
            (PARALLEL_FROM_EXTENT,) + result_shape[1:],
            tuple((shapes[name], _F32) for name in sorted(shapes)),
            compile_options=options,
            compilation_context=context,
        )
    recorded = [record.stage_id for record in context.stage_run_records]
    assert CompilerStageId.LOOPIR_ASSEMBLY_SELECTION not in recorded
    assert CompilerStageId.LOOPIR_TO_LLIR_LOWERING not in recorded
    assert recorded[-1] is CompilerStageId.LOOPIR_SCHEDULE_APPLICATION


def test_the_extent_constant_is_the_runtime_policys():
    """``ROWS_PER_THREAD`` mirrors ``SCORCH_ROWS_PER_THREAD`` in ``scorch_policy.h``.

    The rule's extent and the kernels' runtime thread gate must read the same
    number: with a smaller mirror the plan would record a parallel strategy the
    gate never opens, with a larger one it would keep serial a kernel the gate
    would have opened.
    """

    import pathlib

    import scorch

    header = pathlib.Path(scorch.__file__).parent / "csrc" / "scorch_policy.h"
    defined = re.findall(
        r"^#\s*define\s+SCORCH_ROWS_PER_THREAD\s+(\d+)L?\b", header.read_text(), re.M
    )
    assert defined == [str(ROWS_PER_THREAD)]
    assert assembly_selection.PARALLEL_FROM_EXTENT == 2 * ROWS_PER_THREAD


# -- the two-pass strategies: cells are numbered by coordinate ----------------


def test_a_stored_outer_loop_numbers_two_pass_cells_by_coordinate():
    """The count array is per RECEIVER CELL, and a position is not a cell.

    ``sss ijk->ik [ds]``'s outermost loop walks the operand's stored level zero,
    so its loop variable is a position into ``A0_crd`` and its bound is that
    array's length.  The first compressed level's position array has one entry per
    coordinate of the receiver's dense level zero, over its full extent.  Sizing
    the counts by the loop's trip count and indexing them by its variable -- which
    is what the pass did when both facts came from the loop header -- produced a
    position array of the wrong length, indexed by the wrong thing.
    """

    source = compile_cin_via_loopir(
        reduction_cin("sss", "ds", "ijk", "ik"),
        (2, 4),
        (((2, 3, 4), _F32),),
        compile_options=auto_options(assembly="two_pass_serial"),
    ).cpp_source

    # Sized and swept by the receiver's own extent, indexed by the coordinate the
    # stored loop resolves.
    assert "std::vector<int> _count1((size_t)C0_size, 0);" in source
    assert "_count1[A0_crd[pA0]] = _cnt1;" in source
    assert "std::vector<int64_t> _offset1((size_t)C0_size + 1);" in source
    assert "for (int _i = 0; _i < C0_size; _i++) {" in source
    assert "int64_t _total1 = _offset1[C0_size];" in source
    assert (
        "torch::Tensor C1_pos_torch = torch::empty({(int64_t)(C0_size + 1)}, "
        "torch::kInt);" in source
    )
    assert "int64_t _base1 = _offset1[A0_crd[pA0]];" in source
    # And never by the loop's own position bound, which is the defect's signature.
    assert "_count1((size_t)pA0_end" not in source
    assert "_count1[pA0]" not in source
    assert "_offset1[pA0]" not in source


def test_a_dense_outer_loop_numbers_two_pass_cells_by_its_loop_variable():
    """The dense case is the loop header's own derivation, unchanged.

    A loop over the receiver's dense level zero has a loop variable that IS the
    cell index and a bound that IS the cell count, so the family states exactly
    what the pass would have derived and the emission is the one legacy has always
    produced for this shape.
    """

    source = compile_cin_via_loopir(
        ttm_cin("dss", "dd", "dss"),
        (4, 5, 3),
        (((4, 5, 6), _F32), ((6, 3), _F32)),
        compile_options=auto_options(assembly="two_pass_serial"),
    ).cpp_source

    assert "std::vector<int> _count1((size_t)A0_size, 0);" in source
    assert "std::vector<int> _count2((size_t)A0_size, 0);" in source
    assert "_count1[i] = _cnt1;" in source
    assert "_count2[i] = _cnt2;" in source
    assert "int64_t _base1 = _offset1[i];" in source
    assert "int64_t _base2 = _offset2[i];" in source


def test_the_two_pass_fill_advances_its_cursor_once_per_coordinate():
    """One advance per appended coordinate, not one per spelling of the advance.

    This family appends with ``emplace_back`` -- which grows the vector, so the
    append is the advance -- and also bumps its own mirror cursor ``pC{L}``, which
    the rewrite replaces by ``_pos{L}`` as a value.  Honouring both moved the
    cursor twice per entry, so every coordinate and value landed at twice its
    offset and the write ran past the exactly-sized buffers.  The counting pass
    was unaffected, which is why a compile-only census could not see it.
    """

    rank2 = compile_cin_via_loopir(
        reduction_cin("sss", "ds", "ijk", "ik"),
        (2, 4),
        (((2, 3, 4), _F32),),
        compile_options=auto_options(assembly="two_pass_serial"),
    ).cpp_source
    assert rank2.count("_pos1++;") == 1

    rank3 = compile_cin_via_loopir(
        ttm_cin("dss", "dd", "dss"),
        (4, 5, 3),
        (((4, 5, 6), _F32), ((6, 3), _F32)),
        compile_options=auto_options(assembly="two_pass_serial"),
    ).cpp_source
    # One for the drained leaf coordinate, one for the parent coordinate the
    # position-boundary conditional appends.
    assert rank3.count("_pos2++;") == 1
    assert rank3.count("_pos1++;") == 1


@pytest.mark.parametrize("strategy", TWO_PASS_STRATEGIES)
def test_a_drain_spanning_two_compressed_levels_refuses_the_two_pass(strategy):
    """``unsupported_assembly_strategy``: legal receiver, wrong program, again.

    ``ssss ijkl->ikl [dss]`` has a partitionable receiver and a prefix of one, so
    its workspace drain assembles BOTH compressed levels.  It opens a new segment
    at the upper level by comparing the drained key against
    ``C1_crd.back()`` -- a read of the coordinate vector, which neither phase of a
    two-pass assembly has, since both write into preallocated buffers.  A prefix
    loop instead closes that level with the position-boundary conditional the
    shared pass rewrites in both phases, so the shape that works is one where
    every compressed level above the leaf is closed by a prefix loop.

    Refusing by shape is what keeps this from surfacing as the pass's own
    postcondition failure, which names a dangling reference rather than the
    program shape that caused it.
    """

    with pytest.raises(lower_llir_module.LoopIRTargetError) as raised:
        compile_cin_via_loopir(
            reduction_cin("ssss", "dss", "ijkl", "ikl"),
            (2, 4, 5),
            (((2, 3, 4, 5), _F32),),
            compile_options=auto_options(assembly=strategy),
        )
    assert raised.value.defect.code == "unsupported_assembly_strategy"
    assert "closed by a prefix loop" in raised.value.defect.message


@pytest.mark.parametrize(
    ("operand_fmt", "result_fmt", "operand_indices", "result_indices", "shape"),
    [
        pytest.param("sss", "ds", "ijk", "ik", (5, 4), id="one-stored-prefix-loop"),
        pytest.param(
            "ssss", "dss", "ijkl", "ijl", (3, 4, 5), id="two-stored-prefix-loops"
        ),
    ],
)
def test_the_two_pass_strategies_reproduce_the_single_pass_storage(
    operand_fmt, result_fmt, operand_indices, result_indices, shape
):
    """Interchangeable means the same output tensor, index arrays included.

    The receiver's trailing coordinates are left EMPTY on purpose.  A stored
    prefix loop stops at its last stored coordinate, so the single pass closes the
    cells past it with one final catch-up while the two-pass gets them from the
    prefix sum's tail; an input whose last rows are non-empty cannot tell a
    correct tail from a missing one.  The leading coordinates are empty for the
    same reason at the other end.

    Two shapes, because the second prefix loop is where they differ.  ``ijk->ik``
    has one stored prefix loop and a drain that assembles the only compressed
    level.  ``ijkl->ijl`` has TWO stored prefix loops -- the second one binding a
    compressed result level, which it closes with the position-boundary
    conditional -- so its counting phase carries a count array per compressed
    level and both are indexed by the outermost loop's coordinate.
    """

    from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
    from scorch.stensor import STensor

    rows = 2 * ROWS_PER_THREAD
    generator = torch.Generator().manual_seed(20260816)
    operand_shape = (rows,) + shape
    dense = torch.rand(operand_shape, generator=generator, dtype=torch.float64)
    dense = dense * (
        torch.rand(operand_shape, generator=generator, dtype=torch.float64) < 0.4
    )
    dense[:2] = 0.0
    dense[-3:] = 0.0
    dense = dense.to(_F32)
    operand = STensor.from_torch(dense.clone(), "A").to_sparse(operand_fmt)
    result_shape = (rows,) + tuple(
        operand_shape[operand_indices.index(char)] for char in result_indices[1:]
    )

    def storage(strategy):
        result = execute_cin_via_loopir(
            reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
            result_shape,
            operand,
            compile_options=auto_options(assembly=strategy, jit=True),
        )[0]
        return (
            tuple(
                tuple(tuple(int(x) for x in part.tolist()) for part in mode)
                for mode in result.storage.index.mode_indices
            ),
            tuple(result.storage.value.tolist()),
        )

    serial = storage("single_pass_serial")
    assert storage("two_pass_serial") == serial
    assert storage("two_pass_parallel") == serial
    # And the assembled result is the reduction, not merely self-consistent: the
    # first compressed level's positions open at zero, stay there through the
    # empty leading cells, and close at the entry count through the empty
    # trailing ones.
    positions = serial[0][1][0]
    assert positions[0] == 0
    assert positions[1] == 0 and positions[2] == 0
    assert positions[-1] == positions[-2] == positions[-3] == positions[-4]
    if result_fmt == "ds":
        assert positions[-1] == len(serial[1])


def test_a_strategy_cannot_compose_with_a_decision_that_owns_the_same_assembly():
    """A panel, a relayout, a result tile or an explicit parallel loop each own
    part of the result's assembly, so a strategy cannot be composed with one."""

    with pytest.raises((InvalidSchedule, UnsupportedFeature)) as raised:
        composed = Schedule(assembly="single_pass_chunk_parallel", parallel_loop="i")
        Scheduler.apply_schedule(
            ttm_cin("dss", "ss", "dss"),
            composed,
            compile_options=replace(
                CompileOptions.from_environment(environ={}),
                requested_schedule=composed,
            ),
        )
    codes = {
        getattr(diagnostic, "code", None)
        for diagnostic in getattr(raised.value, "diagnostics", ())
    }
    assert codes & {"invalid_schedule_assembly", "unsupported_schedule_assembly"}


def test_the_public_schedule_rejects_a_token_that_is_not_a_strategy():
    with pytest.raises(ValueError, match="sparse assembly strategy"):
        Schedule(assembly="two_pass")
    with pytest.raises(ValueError, match="sparse assembly strategy"):
        Schedule(assembly="")


def test_the_plan_verifier_rejects_a_token_that_is_not_a_strategy():
    cin = ttm_cin("dss", "ss", "dss")
    order = tuple(
        index_var.index_id for index_var in Scheduler.get_index_variables(cin)
    )
    with pytest.raises(Exception, match="sparse assembly strategy"):
        verify_loop_plan(cin, LoopPlan(loop_order=order, assembly="nope"))


# -- the schemas, and the identity they carry ---------------------------------


def test_both_canonical_schemas_moved_and_the_plan_bytes_carry_the_strategy():
    """A strategy selects which kernel one program emits, so it is schedule
    content: it enters the plan's canonical form and therefore every identity
    derived from it."""

    # Both have moved again since, for the accumulation structure that
    # ``test_sparse_accumulator_structure.py`` owns; what this case locks is that
    # the strategy is IN the canonical form, not which version it arrived at.
    assert CANONICAL_SCHEMA == "scorch.loopir.canonical.v13"
    assert CANONICAL_PLAN_SCHEMA == "scorch.loopplan.canonical.v3"

    cin = ttm_cin("dss", "ss", "dss")
    order = tuple(
        index_var.index_id for index_var in Scheduler.get_index_variables(cin)
    )
    plain = verify_loop_plan(cin, LoopPlan(loop_order=order))
    chunked = verify_loop_plan(
        cin,
        LoopPlan(loop_order=order, assembly="single_pass_chunk_parallel"),
    )
    plain_dump = canonical_plan_dump(cin, plain)
    chunked_dump = canonical_plan_dump(cin, chunked)
    assert '"assembly":null' in plain_dump
    assert '"assembly":"single_pass_chunk_parallel"' in chunked_dump
    # Two plans differing only in strategy are different schedules.
    assert plain_dump != chunked_dump
    assert CANONICAL_PLAN_SCHEMA in plain_dump


def test_the_public_cache_key_discriminates_the_strategy():
    """``Schedule.cache_key`` is the release JIT's schedule discriminator; two
    schedules that emit different kernels must not share one."""

    keys = {
        Schedule(assembly=strategy).cache_key
        for strategy in (None, *SPARSE_ASSEMBLY_STRATEGIES)
    }
    assert len(keys) == len(SPARSE_ASSEMBLY_STRATEGIES) + 1


def test_the_plan_round_trips_through_the_legacy_schedule_seam():
    """A field dropped by ``materialize_legacy_schedule`` would silently downgrade
    an explicit request to the default, which is unobservable to the caller."""

    from scorch.compiler.scheduler import materialize_legacy_schedule

    cin = ttm_cin("dss", "ss", "dss")
    order = tuple(
        index_var.index_id for index_var in Scheduler.get_index_variables(cin)
    )
    for strategy in (None, *SPARSE_ASSEMBLY_STRATEGIES):
        plan = verify_loop_plan(cin, LoopPlan(loop_order=order, assembly=strategy))
        schedule, _bounds, _relayout, _result_tile = materialize_legacy_schedule(
            cin, plan
        )
        assert schedule.assembly == strategy
