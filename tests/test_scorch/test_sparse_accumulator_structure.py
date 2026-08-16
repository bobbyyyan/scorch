"""Sparse accumulation structure as a scheduling decision.

Which structure a sparse reduction accumulates through was decided by an
assembly transform on its way past: when the workspace declaration sat at the top
of the loop it replaced, the two-phase parallel transform swapped whatever
structure the program had declared for a per-worker pool of the chained one.
Review section 67 measured what that swap costs on the emitted kernel -- 0.872x
to 1.574x, changing SIGN with density, outside its own same-binary A/A floor on
56 of 64 configurations -- so it is a scheduling decision, and this file locks
the properties that make it one.

**The vocabulary has ONE definition.**  Two copies of one scheduling rule
drifting apart is the defect that cost twelve cells when the automatic tile
heuristic was fixed in only one of its two layers, so the tokens and the
single-key contract live in ``sparse_accumulator`` and every layer imports them.

**No recorded structure means every existing layer chooses what it chooses
today, and that keeps the emission byte-neutral.**  This one is awkward and it is
stated rather than hidden: the layer this work calls wrong is still the default.
``None`` leaves the transform substituting exactly where it substitutes today,
which is the only default that lands without moving a shipped byte -- and the
substitution is on the pipeline that ships, not only on the typed one.

**Legality and cost are different predicates in different places.**  The
single-key contract and "a dense result has no sparse accumulation workspace" are
legality: structural, extent-free, refused with a code.  Which structure PAYS
depends on the density, on the receiver's compressed extent and partly on the
worker count, and none of that appears anywhere in this decision's legality.

**A request that cannot be honoured is REFUSED, never dropped.**  Each refusal
names the layer that could not honour it: the schedule boundary for a dense
result, the host for a family that cannot emit the structure, and the transform
itself for a body it has nothing to substitute in.

**The schemas moved, and both of them.**  LoopIR canonical v12 -> v13 and the
plan canonical v2 -> v3, so every plan's canonical bytes and every schedule's
cache key move once.
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
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import InvalidSchedule, UnsupportedFeature
from scorch.compiler.loop_plan import LoopPlan, verify_loop_plan
from scorch.compiler.loopir.nodes import AccumulatorStructure
from scorch.compiler.loopir.plan_identity import (
    CANONICAL_PLAN_SCHEMA,
    canonical_plan_dump,
    loopir_request_identity,
    plan_schedule_digest,
)
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir
from scorch.compiler.loopir.printer import CANONICAL_SCHEMA
from scorch.compiler.scheduler import Schedule, Scheduler
from scorch.compiler.sparse_accumulator import (
    CHAINED_ACCUMULATOR_STRUCTURE,
    DECLARED_ACCUMULATOR_STRUCTURE,
    SPARSE_ACCUMULATOR_STRUCTURES,
    is_accumulator_structure,
    single_coordinate_key,
)
from scorch.stensor import STensor

# -- the vocabulary -----------------------------------------------------------


def test_the_token_set_is_exactly_the_two_structures_that_exist():
    """Two, not four.

    A structure and whether it is pooled per worker look like two independent
    bits, and they are not: of the four cells that product admits, only two have
    a producer.  The chained structure's constructor allocates and fills arrays of
    the receiver's compressed extent, so it must be hoisted out of the assembly
    loop, so under a region it must be one per worker -- pooling is entailed by
    that structure rather than chosen beside it.  The declared coordinate list's
    constructor is independent of the receiver, so nothing pools it.  A pooled
    coordinate list is a candidate fix nobody has built and becomes a THIRD token
    when somebody does; a per-iteration chained accumulator has no producer and no
    reason to want one.
    """

    assert SPARSE_ACCUMULATOR_STRUCTURES == ("coordinate_list", "linked_list")
    assert DECLARED_ACCUMULATOR_STRUCTURE == "coordinate_list"
    assert CHAINED_ACCUMULATOR_STRUCTURE == "linked_list"
    assert set(SPARSE_ACCUMULATOR_STRUCTURES) == {
        DECLARED_ACCUMULATOR_STRUCTURE,
        CHAINED_ACCUMULATOR_STRUCTURE,
    }


def test_the_typed_enum_and_the_token_set_are_the_same_vocabulary():
    """A second copy of the token set is how the two would drift apart."""

    assert {member.value for member in AccumulatorStructure} == set(
        SPARSE_ACCUMULATOR_STRUCTURES
    )


def test_only_the_exact_tokens_are_structures():
    """Exact-string admission: a near miss is not a structure."""

    for structure in SPARSE_ACCUMULATOR_STRUCTURES:
        assert is_accumulator_structure(structure)
    for rejected in (
        None,
        "",
        "Coordinate_List",
        "coordinate list",
        "linked_list ",
        b"linked_list",
        0,
    ):
        assert not is_accumulator_structure(rejected)


def test_the_chained_structure_requires_one_key_component():
    """Its chain lives in an array indexed by the key, so there is one array.

    Structural and extent-free: the predicate asks whether the key HAS a single
    bounded coordinate, never how large that coordinate's extent is.
    """

    assert single_coordinate_key(1)
    assert not single_coordinate_key(0)
    assert not single_coordinate_key(2)
    assert not single_coordinate_key(3)
    for rejected in (None, "1", 1.0, True):
        assert not single_coordinate_key(rejected)


def test_the_vocabulary_module_consults_no_extent_density_or_thread_count():
    """Legality lives here; cost does not.

    Source-level, because the separation is the property and a docstring saying
    so is not the property.  The module may not mention an extent, a density, a
    thread count or a measurement.
    """

    import io
    import tokenize

    import scorch.compiler.sparse_accumulator as module

    source = open(module.__file__).read()
    # Comments and docstrings are allowed to explain what is NOT here; the code
    # is not, so they are tokenized away rather than filtered by eye.
    code = " ".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in (tokenize.COMMENT, tokenize.STRING)
    )
    for forbidden in (
        "nthreads",
        "omp_get_thread_num",
        "density",
        "capacity",
        "extent",
        "measure",
    ):
        assert forbidden not in code, forbidden
    assert not re.search(r"\bimport\s+torch\b", source)


# -- fixtures -----------------------------------------------------------------

_F32 = torch.float32


def spgemm_cin(a_fmt="ss", b_fmt="ds", c_fmt="ds", dtype=_F32):
    """``C[i,j] += A[i,k] * B[k,j]`` -- the cell the confound was measured on."""

    i, k, j = (IndexVar(name) for name in ("i", "k", "j"))
    a = TensorVar("A", fmt=a_fmt, dtype=dtype)[i, k]
    b = TensorVar("B", fmt=b_fmt, dtype=dtype)[k, j]
    c = TensorVar("C", fmt=c_fmt, dtype=dtype)[i, j]
    statement = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for variable in (j, k, i):
        statement = ForAll(variable, statement)
    return statement


def ttm_cin(a_fmt="dss", b_fmt="ss", c_fmt="dss", dtype=_F32):
    """``C[i,j,l] += A[i,j,k] * B[k,l]`` -- the control, whose workspace is one
    loop deeper than the transform can see."""

    # ``l`` is the contraction's own index name, spelled as every other TTM site
    # in the tree spells it.
    i, j, k, l = (IndexVar(name) for name in ("i", "j", "k", "l"))  # noqa: E741
    a = TensorVar("A", fmt=a_fmt, dtype=dtype)[i, j, k]
    b = TensorVar("B", fmt=b_fmt, dtype=dtype)[k, l]
    c = TensorVar("C", fmt=c_fmt, dtype=dtype)[i, j, l]
    statement = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for variable in (l, k, j, i):
        statement = ForAll(variable, statement)
    return statement


def legacy_cpp(cin, schedule):
    """One emission through the LEGACY route -- the pipeline that ships."""

    scheduled = Scheduler.apply_schedule(cin, schedule)
    return LLIRLowerer().lower_llir(CINLowerer().lower_IndexStmt(scheduled))


def plan_order(cin):
    return tuple(index_var.index_id for index_var in Scheduler.get_index_variables(cin))


# -- the representation, and the identity it carries --------------------------


def test_the_plan_the_program_and_the_public_schedule_all_carry_it():
    """One decision, spelled once per layer, with ``None`` meaning no decision."""

    cin = spgemm_cin()
    order = plan_order(cin)
    assert LoopPlan(loop_order=order).accumulator is None
    for structure in SPARSE_ACCUMULATOR_STRUCTURES:
        plan = verify_loop_plan(cin, LoopPlan(loop_order=order, accumulator=structure))
        assert plan.accumulator == structure
        assert Schedule(accumulator=structure).accumulator == structure


def test_a_forged_token_is_refused_at_every_boundary_that_stores_one():
    """A value that is not a structure would reach a layer comparing it against
    enum members and match nothing."""

    with pytest.raises(ValueError, match="sparse accumulation structure"):
        Schedule(accumulator="chained")
    cin = spgemm_cin()
    with pytest.raises(Exception, match="sparse accumulation structure"):
        verify_loop_plan(
            cin, LoopPlan(loop_order=plan_order(cin), accumulator="chained")
        )


def test_both_canonical_schemas_moved_and_every_identity_carries_the_key():
    """A structure selects which kernel one program emits, so it is schedule
    content: it enters the plan's canonical form and every identity derived from
    it."""

    assert CANONICAL_SCHEMA == "scorch.loopir.canonical.v13"
    assert CANONICAL_PLAN_SCHEMA == "scorch.loopplan.canonical.v3"

    cin = spgemm_cin()
    order = plan_order(cin)
    plain = verify_loop_plan(cin, LoopPlan(loop_order=order))
    chained = verify_loop_plan(
        cin, LoopPlan(loop_order=order, accumulator=CHAINED_ACCUMULATOR_STRUCTURE)
    )
    plain_dump = canonical_plan_dump(cin, plain)
    chained_dump = canonical_plan_dump(cin, chained)
    # Present with a null value on a plan that records nothing, so "no decision"
    # stays distinguishable from "coordinate list by decision" and every plan's
    # bytes move exactly once.
    assert '"accumulator":null' in plain_dump
    assert '"accumulator":"linked_list"' in chained_dump
    assert plain_dump != chained_dump
    assert CANONICAL_PLAN_SCHEMA in plain_dump
    assert plan_schedule_digest(cin, plain) != plan_schedule_digest(cin, chained)

    options = CompileOptions.from_environment(environ={})
    bindings = (((8, 8), _F32), ((8, 8), _F32))
    assert loopir_request_identity(
        cin, plain, (8, 8), bindings, compile_options=options
    ) != loopir_request_identity(
        cin, chained, (8, 8), bindings, compile_options=options
    )


def test_the_public_cache_key_discriminates_the_structure():
    """``Schedule.cache_key`` is the release JIT's schedule discriminator; two
    schedules that emit different kernels must not share one."""

    keys = {
        Schedule(accumulator=structure).cache_key
        for structure in (None, *SPARSE_ACCUMULATOR_STRUCTURES)
    }
    assert len(keys) == len(SPARSE_ACCUMULATOR_STRUCTURES) + 1


def test_the_plan_round_trips_through_the_legacy_schedule_seam():
    """A field dropped at the legacy seam would silently downgrade an explicit
    request, which is unobservable to the caller."""

    from scorch.compiler.scheduler import materialize_legacy_schedule

    cin = spgemm_cin()
    order = plan_order(cin)
    for structure in (None, *SPARSE_ACCUMULATOR_STRUCTURES):
        plan = verify_loop_plan(cin, LoopPlan(loop_order=order, accumulator=structure))
        schedule, _bounds, _relayout, _tile = materialize_legacy_schedule(cin, plan)
        assert schedule.accumulator == structure


def test_a_schedule_carrying_only_the_structure_is_still_the_automatic_marker():
    """It must be, and the reason is measured rather than assumed.

    The structure is a property of the accumulation workspace, and the workspace
    insertion is a decision only the automatic route records -- so a schedule
    carrying only this field has to reach that route.  Treating it as explicit
    would produce a CIN with a workspace and a plan without the fact, which is
    the failure section 60.5 measured for the assembly strategy.
    """

    cin = spgemm_cin()
    scheduled = Scheduler.apply_schedule(
        cin, Schedule(accumulator=DECLARED_ACCUMULATOR_STRUCTURE)
    )
    plan = scheduled.verified_loop_plan
    assert plan.provenance == "auto"
    assert plan.workspace is not None
    assert plan.accumulator == DECLARED_ACCUMULATOR_STRUCTURE


# -- legality, which is not cost ----------------------------------------------


def test_a_dense_result_cannot_record_a_structure_and_says_so_with_a_code():
    """A dense result has no sparse accumulation workspace to hold."""

    i, k, j = (IndexVar(name) for name in ("i", "k", "j"))
    a = TensorVar("A", fmt="ds", dtype=_F32)[i, k]
    b = TensorVar("B", fmt="ds", dtype=_F32)[k, j]
    c = TensorVar("C", fmt="dd", dtype=_F32)[i, j]
    statement = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for variable in (j, k, i):
        statement = ForAll(variable, statement)

    with pytest.raises(UnsupportedFeature) as refusal:
        Scheduler.apply_schedule(
            statement, Schedule(accumulator=DECLARED_ACCUMULATOR_STRUCTURE)
        )
    assert [d.code for d in refusal.value.diagnostics] == [
        "unsupported_schedule_accumulator"
    ]

    # And the plan boundary states the same rule for a plan that reaches it
    # without passing the explicit surface.
    with pytest.raises((UnsupportedFeature, InvalidSchedule)) as plan_refusal:
        verify_loop_plan(
            statement,
            LoopPlan(
                loop_order=plan_order(statement),
                accumulator=DECLARED_ACCUMULATOR_STRUCTURE,
            ),
        )
    assert "unsupported_schedule_accumulator" in str(plan_refusal.value)


def test_a_structure_is_never_refused_for_an_extent():
    """Legality is structural, so the same request is honoured on a receiver far
    too small for any parallel arm to fire and on a large one alike.

    The precedent is section 60.3's reclassification: the ``2 * ROWS_PER_THREAD``
    compile-time decline is COST, so an explicit request at that extent is
    honoured rather than declined.  Whether keeping the coordinate list PAYS at
    either extent is a different question in a different place.
    """

    bindings_small = (((4, 4), _F32), ((4, 4), _F32))
    bindings_large = (((4096, 512), _F32), ((512, 4096), _F32))
    for shape, bindings in (((4, 4), bindings_small), ((4096, 4096), bindings_large)):
        emitted = typed_cpp(
            spgemm_cin("ss", "ds", "ds"),
            shape,
            bindings,
            assembly="two_pass_serial",
            accumulator=DECLARED_ACCUMULATOR_STRUCTURE,
        )
        assert "coo_workspace_1d" in emitted
        assert "linked_list_workspace_1d" not in emitted


# -- what the transform does with it ------------------------------------------


def test_no_decision_emits_exactly_what_the_pipeline_emitted_before():
    """The awkward property, and the one the whole design turns on.

    The two-phase transform substitutes the chained structure on the pipeline
    that SHIPS, not only on the typed one, so "no decision" has to keep meaning
    "substitute exactly where you substitute today" or a shipped byte moves.  An
    explicit request for the structure the transform would have chosen anyway
    therefore emits the same kernel, and only the OTHER request changes anything.
    """

    cin = spgemm_cin(c_fmt="ds")
    no_decision = legacy_cpp(cin, Schedule())
    chained = legacy_cpp(cin, Schedule(accumulator=CHAINED_ACCUMULATOR_STRUCTURE))
    assert no_decision == chained
    assert "std::vector<linked_list_workspace_1d" in no_decision
    assert ".insert_unchecked(" in no_decision
    assert "coo_workspace_1d" not in no_decision


def test_the_declared_structure_is_kept_when_it_is_asked_for():
    """The override removed: the pool, the type substitution and the
    ``insert`` -> ``insert_unchecked`` rename are one bit inside the transform,
    so keeping the declaration switches off all three -- and the two-phase
    structure the strategy asked for survives, which is the point."""

    cin = spgemm_cin(c_fmt="ds")
    kept = legacy_cpp(cin, Schedule(accumulator=DECLARED_ACCUMULATOR_STRUCTURE))
    assert "linked_list_workspace_1d" not in kept
    assert "coo_workspace_1d" in kept
    assert ".insert_unchecked(" not in kept
    assert ".insert(" in kept
    assert "make_view()" not in kept
    assert "_pool" not in kept
    # The counting phase is still there: this changes the accumulator, not the
    # assembly.
    assert "_cnt1" in kept
    assert "_count1" in kept


def test_a_chained_request_the_transform_cannot_honour_is_refused_not_dropped():
    """Two ways it cannot be honoured, and both name a layer.

    A doubly-compressed receiver never runs the transform at all, so the legacy
    route refuses at its single pipeline entry; a caller who got the coordinate
    list back instead would have no way to find out.
    """

    cin = spgemm_cin(c_fmt="ss")
    kept = legacy_cpp(cin, Schedule(accumulator=DECLARED_ACCUMULATOR_STRUCTURE))
    assert "coo_workspace_1d" in kept
    assert kept == legacy_cpp(cin, Schedule())

    with pytest.raises(UnsupportedFeature) as refusal:
        legacy_cpp(cin, Schedule(accumulator=CHAINED_ACCUMULATOR_STRUCTURE))
    assert [d.code for d in refusal.value.diagnostics] == [
        "unsupported_accumulator_structure"
    ]


def test_the_transform_itself_refuses_a_body_it_has_nothing_to_substitute_in():
    """The pass is the only layer that knows whether the emitted body declares
    the accumulator where it can be hoisted, so the refusal is the pass's."""

    from scorch.compiler import llir
    from scorch.compiler.compressed_where_openmp_pass import (
        CompressedWhereOpenMPContext,
        transform_compressed_where_for_openmp,
    )
    from scorch.compiler.llir_traversal import LLIRTraversalError
    from scorch.compiler.torch_cpp_abi import ResultTensorAssembler
    from scorch.format import LevelType

    def context(accumulator):
        return CompressedWhereOpenMPContext(
            result_name="Result",
            result_id=__import__(
                "scorch.compiler.identity", fromlist=["SymbolId"]
            ).SymbolId(1),
            compressed_levels=(1,),
            result_assembler=ResultTensorAssembler(
                name="Result",
                level_types=(LevelType.DENSE, LevelType.COMPRESSED),
                dtype=torch.float32,
            ),
            workspace_name="wksp",
            workspace_ctype="float",
            accumulator=accumulator,
        )

    # A loop whose body declares no workspace at all.
    loop = llir.ForLoop(
        init=llir.VarInit(
            var=llir.Var(name="row", type=llir.DataType.INT),
            value=llir.Literal(0, llir.DataType.INT),
        ),
        cond=llir.BinOp(
            "<",
            llir.Var(name="row", type=llir.DataType.INT),
            llir.Var(name="Result0_size", type=llir.DataType.INT),
        ),
        update=llir.Increment(llir.Var(name="row", type=llir.DataType.INT)),
        body=[llir.Comment("no workspace here")],
    )

    with pytest.raises(LLIRTraversalError) as refusal:
        transform_compressed_where_for_openmp(
            [loop], context(CHAINED_ACCUMULATOR_STRUCTURE)
        )
    assert refusal.value.diagnostic.code == "unsupported_accumulator_structure"

    # A forged token is refused before anything is transformed.
    with pytest.raises(LLIRTraversalError) as forged:
        transform_compressed_where_for_openmp([loop], context("chained"))
    assert forged.value.diagnostic.code == "invalid_compressed_where_accumulator"

    # No decision, and the declared structure, are both fine on the same body.
    for accumulator in (None, DECLARED_ACCUMULATOR_STRUCTURE):
        result = transform_compressed_where_for_openmp([loop], context(accumulator))
        assert result.applied


# -- the typed route, and the probe it reproduces -----------------------------


def typed_cpp(cin, result_shape, bindings, *, assembly, accumulator, arm=0):
    options = replace(
        CompileOptions.from_environment(environ={}).with_regblock_enabled(bool(arm)),
        requested_schedule=Schedule(assembly=assembly, accumulator=accumulator),
    )
    return compile_cin_via_loopir(
        cin, result_shape, bindings, compile_options=options
    ).cpp_source


@pytest.mark.parametrize("arm", (0, 1))
@pytest.mark.parametrize("assembly", ("two_pass_serial", "two_pass_parallel"))
def test_the_production_column_emits_what_the_ablation_probe_emitted(arm, assembly):
    """The point of the change, stated as an equality.

    Section 67's third ablation column suppressed the substitution by patching a
    module attribute from a harness -- nothing in the tree implemented it.  An
    explicit request for the declared structure must produce that same program,
    character for character, or the measurement the fix is judged against is
    measuring something else.
    """

    import scorch.compiler.compressed_where_openmp_pass as two_phase
    from scorch.compiler import llir

    cin_factory = lambda: spgemm_cin("ss", "ds", "ds")  # noqa: E731
    shape = (16, 12)
    bindings = (((16, 10), _F32), ((10, 12), _F32))

    original = two_phase._should_drop_work_statement

    def keep(statement, *, result_name, first_compressed_level, workspace_name):
        if type(statement) is llir.VarInit and statement.var.name == workspace_name:
            return False, False
        return original(
            statement,
            result_name=result_name,
            first_compressed_level=first_compressed_level,
            workspace_name=workspace_name,
        )

    two_phase._should_drop_work_statement = keep
    try:
        probed = typed_cpp(
            cin_factory(),
            shape,
            bindings,
            assembly=assembly,
            accumulator=None,
            arm=arm,
        )
    finally:
        two_phase._should_drop_work_statement = original

    requested = typed_cpp(
        cin_factory(),
        shape,
        bindings,
        assembly=assembly,
        accumulator=DECLARED_ACCUMULATOR_STRUCTURE,
        arm=arm,
    )
    assert requested == probed

    # And the substituted column is a different program, so the equality above
    # is not the trivial one.
    substituted = typed_cpp(
        cin_factory(),
        shape,
        bindings,
        assembly=assembly,
        accumulator=None,
        arm=arm,
    )
    assert substituted != requested
    assert substituted == typed_cpp(
        cin_factory(),
        shape,
        bindings,
        assembly=assembly,
        accumulator=CHAINED_ACCUMULATOR_STRUCTURE,
        arm=arm,
    )


@pytest.mark.parametrize("arm", (0, 1))
def test_a_workspace_one_loop_deeper_is_left_alone_and_says_so(arm):
    """The control: the transform never sees the TTM family's declaration, so
    the declared structure is a no-op there and the chained one is refused."""

    shape = (8, 8, 8)
    bindings = (((8, 8, 8), _F32), ((8, 8), _F32))
    for assembly in ("two_pass_serial", "two_pass_parallel"):
        plain = typed_cpp(
            ttm_cin(),
            shape,
            bindings,
            assembly=assembly,
            accumulator=None,
            arm=arm,
        )
        kept = typed_cpp(
            ttm_cin(),
            shape,
            bindings,
            assembly=assembly,
            accumulator=DECLARED_ACCUMULATOR_STRUCTURE,
            arm=arm,
        )
        assert kept == plain
        assert "linked_list_workspace_1d" not in plain
        with pytest.raises(Exception) as refusal:
            typed_cpp(
                ttm_cin(),
                shape,
                bindings,
                assembly=assembly,
                accumulator=CHAINED_ACCUMULATOR_STRUCTURE,
                arm=arm,
            )
        assert "unsupported_accumulator_structure" in str(refusal.value)


def test_the_family_that_cannot_emit_the_declared_structure_names_itself():
    """The parallel sparse-workspace family's completion mirror reconstructs the
    pooled shape and refuses anything else, so listing only what it can emit
    turns an internal completion failure into an honest refusal."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    parallel_family = lower_llir_module._ParallelSparseWorkspaceLowering
    assert parallel_family.supported_accumulators(parallel_family) == (
        CHAINED_ACCUMULATOR_STRUCTURE,
    )
    ordered_key = lower_llir_module._OrderedKeySparseWorkspaceLowering
    assert set(ordered_key.supported_accumulators(ordered_key)) == set(
        SPARSE_ACCUMULATOR_STRUCTURES
    )
    serial_family = lower_llir_module._SparseWorkspaceLowering
    assert serial_family.supported_accumulators(serial_family) == (
        DECLARED_ACCUMULATOR_STRUCTURE,
    )
    base = lower_llir_module._TargetLowering
    assert base.supported_accumulators(base) == ()
    # No family CHOOSES one today; that is the seam a selector replaces.
    for family in (parallel_family, ordered_key, serial_family, base):
        assert family.default_accumulator(family) is None


# -- correctness --------------------------------------------------------------


@pytest.mark.parametrize("assembly", ("two_pass_serial", "two_pass_parallel"))
def test_keeping_the_declared_structure_reproduces_the_single_pass_bit_for_bit(
    assembly,
):
    """Section 67.5's free result, as a lock: with the coordinate list kept, the
    two-pass kernel's whole output storage equals the single pass's exactly --
    which the substituted column does not, on this cell, by one unit in the last
    place."""

    from scorch.compiler.loopir.pipeline import execute_cin_via_loopir

    torch.manual_seed(11)
    left = (torch.rand(24, 20) < 0.25).float() * torch.rand(24, 20)
    right = (torch.rand(20, 18) < 0.25).float() * torch.rand(20, 18)
    operands = (
        STensor.from_torch(left.clone(), "A").to_sparse("ss"),
        STensor.from_torch(right.clone(), "B").to_sparse("ds"),
    )

    def run(assembly_token, accumulator):
        options = replace(
            CompileOptions.from_environment(),
            requested_schedule=Schedule(
                assembly=assembly_token, accumulator=accumulator
            ),
        )
        result, _ = execute_cin_via_loopir(
            spgemm_cin("ss", "ds", "ds"),
            (24, 18),
            *operands,
            compile_options=options,
        )
        return result

    baseline = run("single_pass_serial", None)
    kept = run(assembly, DECLARED_ACCUMULATOR_STRUCTURE)
    reference = (left @ right).to(torch.float64)

    baseline_arrays = _storage_of(baseline)
    kept_arrays = _storage_of(kept)
    assert set(baseline_arrays) == set(kept_arrays)
    for key in baseline_arrays:
        assert baseline_arrays[key] == kept_arrays[key], key
    assert (
        kept.to_torch(in_place=False).to(torch.float64) - reference
    ).abs().max() < 1e-5


def _storage_of(result):
    arrays = {}
    for level, mode in enumerate(result.storage.index.get_mode_indices()):
        for position, tensor in enumerate(mode):
            arrays[f"L{level}A{position}"] = tensor.tolist()
    arrays["values"] = result.storage.value.tolist()
    return arrays
