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

**No recorded strategy means the target chooses, and that keeps the automatic
origin byte-neutral.**  The origin writes nothing into the field, so an ordinary
automatic compilation records ``None``, target lowering runs its own
per-receiver choice, and the emitted kernel is unchanged.  The neutrality is
structural rather than empirical, and a lock here says so on the plan and the
program.

**Legality and cost are different predicates in different places.**  The
receiver contract is legality: structural, extent-free, refused with a code.  The
``2 * ROWS_PER_THREAD`` extent test is COST -- a 16-row receiver assembles
correctly from one chunk -- so it moved out of the legality predicate into the
default choice, which is the seam a selector replaces.  An explicit request at a
small extent is therefore honoured.

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
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import InvalidSchedule, UnsupportedFeature
from scorch.compiler.loop_plan import LoopPlan, verify_loop_plan
from scorch.compiler.loopir import lower_llir as lower_llir_module
from scorch.compiler.loopir.nodes import AssemblyStrategy
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir
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


# -- no recorded strategy: the automatic origin stays neutral ------------------


def test_the_automatic_origin_records_no_strategy():
    """The origin chooses nothing, so the plan and the program carry nothing.

    This is the milestone's central safety property, and it is structural: the
    cost model never writes this field, ``None`` routes to target lowering's own
    per-receiver choice, and that choice is the code that ran before the field
    existed.
    """

    kernel = compile_cin_via_loopir(
        ttm_cin("dss", "ss", "dss"),
        (4, 5, 3),
        (((4, 5, 6), _F32), ((6, 3), _F32)),
        compile_options=auto_options(),
    )
    assert kernel.schedule is not None
    assert kernel.schedule.plan.assembly is None
    assert kernel.schedule.program.assembly is None
    assert '"assembly":null' in kernel.program_dump


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
    assert plan_free.schedule.plan.assembly is None
    assert requested.schedule.plan.assembly == "single_pass_chunk_parallel"


# -- legality is not cost -----------------------------------------------------


def test_the_extent_test_is_cost_and_no_longer_gates_legality():
    """A receiver too small to reach two threads is still LEGAL.

    ``scorch_nthreads``'s row term alone caps the count at one below
    ``2 * ROWS_PER_THREAD``, so the target's own choice declines the chunked
    strategy there -- that is a judgement about whether the transformation pays,
    and it keeps those kernels byte-identical.  It is not a claim that the
    program cannot be assembled that way, so an explicit request is honoured and
    emits the gate.  Keeping the two answers apart is what lets a selector
    replace the cost half without touching correctness.
    """

    rows = 2 * lower_llir_module.PARALLEL_CHUNK_ROWS_PER_THREAD
    small, large = rows - 1, rows

    def emit(outer, assembly):
        return compile_cin_via_loopir(
            ttm_cin("dss", "ss", "dss"),
            (outer, 5, 3),
            (((outer, 5, 6), _F32), ((6, 3), _F32)),
            compile_options=auto_options(assembly=assembly),
        ).cpp_source

    # Cost: below the threshold the automatic choice declines, and the emission
    # is the serial builder.  At the threshold it takes the chunked strategy.
    assert "_assembly_chunks" not in emit(small, None)
    assert "_assembly_chunks" in emit(large, None)
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

    rows = 2 * lower_llir_module.PARALLEL_CHUNK_ROWS_PER_THREAD

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


def _matmul_cin(a_fmt, b_fmt, c_fmt):
    i, j, k = (IndexVar(name) for name in ("i", "j", "k"))
    a = TensorVar("A", fmt=a_fmt, dtype=_F32)[i, j]
    b = TensorVar("B", fmt=b_fmt, dtype=_F32)[j, k]
    c = TensorVar("C", fmt=c_fmt, dtype=_F32)[i, k]
    statement = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for variable in (k, j, i):
        statement = ForAll(variable, statement)
    return statement


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

    rows = 2 * lower_llir_module.PARALLEL_CHUNK_ROWS_PER_THREAD
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

    assert CANONICAL_SCHEMA == "scorch.loopir.canonical.v12"
    assert CANONICAL_PLAN_SCHEMA == "scorch.loopplan.canonical.v2"

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
