"""Canonical LoopPlan serialization and the strangler request identity."""

import hashlib
import json
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from scorch.compiler.cin import (
    BinaryOp,
    ForAll,
    IndexVar,
    IndexVarAdd,
    Operation,
    TensorAccess,
    TensorAssign,
    TensorVar,
    UnaryOp,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.compilation_context import CompilationContext, CompilerStageId
from scorch.compiler.diagnostics import VerificationError
from scorch.compiler.loop_plan import LoopRef, WorkspaceInsertion
from scorch.compiler.loopir.plan_identity import (
    CANONICAL_PLAN_SCHEMA,
    CANONICAL_REQUEST_SCHEMA,
    canonical_plan_dump,
    loopir_request_dump,
    loopir_request_identity,
    plan_schedule_digest,
)
from scorch.compiler.scheduler import RelayoutSpec, Schedule, Scheduler, TileSpec
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir

F32 = torch.float32
MATMUL_BINDINGS = (((4, 5), F32), ((5, 6), F32))
ELEMENTWISE_BINDINGS = (((4, 5), F32), ((4, 5), F32))
SPMM_BINDINGS = (((4, 5), F32), ((5, 6), F32))
IDENTITY_OPTIONS = CompileOptions.from_environment()


def build_matmul():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    return ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(
                    c[i, k],
                    BinaryOp(Operation.MUL, a[i, j], b[j, k]),
                    op=Operation.ADD,
                ),
            ),
        ),
    )


def build_elementwise():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    return ForAll(
        i,
        ForAll(j, TensorAssign(c[i, j], BinaryOp(Operation.ADD, a[i, j], b[i, j]))),
    )


def build_spmm():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    return ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(
                    c[i, k],
                    BinaryOp(Operation.MUL, a[i, j], b[j, k]),
                    op=Operation.ADD,
                ),
            ),
        ),
    )


def scheduled(build, schedule):
    return Scheduler.apply_schedule(build(), schedule)


def request_identity(
    artifact,
    result_shape=(4, 6),
    bindings=MATMUL_BINDINGS,
    compile_options=IDENTITY_OPTIONS,
):
    return loopir_request_identity(
        artifact.normalized_cin,
        artifact.verified_loop_plan,
        result_shape,
        bindings,
        compile_options=compile_options,
    )


BASE_SCHEDULE = Schedule(
    loop_order=("i", "j", "k"),
    tiles=(TileSpec("k", 4, accum="direct"),),
)


def test_fresh_builders_produce_identical_canonical_plans() -> None:
    """Equivalent plans built by fresh builders serialize byte-identically.

    Process-global IndexId/SymbolId allocation differs between the two
    builds; only the artifact-local canonical numbering may enter the
    serialization.
    """

    first = scheduled(build_matmul, BASE_SCHEDULE)
    second = scheduled(build_matmul, BASE_SCHEDULE)
    assert (
        first.verified_loop_plan.loop_order != second.verified_loop_plan.loop_order
    ), "the probe needs fresh global identities to be meaningful"
    first_dump = canonical_plan_dump(first.normalized_cin, first.verified_loop_plan)
    second_dump = canonical_plan_dump(second.normalized_cin, second.verified_loop_plan)
    assert first_dump == second_dump
    assert json.loads(first_dump)["schema"] == CANONICAL_PLAN_SCHEMA
    assert request_identity(first) == request_identity(second)


def test_repeated_dumps_are_deterministic() -> None:
    artifact = scheduled(build_matmul, BASE_SCHEDULE)
    dumps = {
        canonical_plan_dump(artifact.normalized_cin, artifact.verified_loop_plan)
        for _ in range(5)
    }
    assert len(dumps) == 1
    identities = {request_identity(artifact) for _ in range(5)}
    assert len(identities) == 1


@pytest.mark.parametrize(
    "name, schedule",
    [
        (
            "order",
            Schedule(
                loop_order=("i", "k", "j"), tiles=(TileSpec("k", 4, accum="direct"),)
            ),
        ),
        (
            "width",
            Schedule(
                loop_order=("i", "j", "k"), tiles=(TileSpec("k", 8, accum="direct"),)
            ),
        ),
        (
            "accumulation",
            Schedule(
                loop_order=("i", "j", "k"), tiles=(TileSpec("k", 4, accum="stack"),)
            ),
        ),
        (
            "unroll",
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(TileSpec("k", 4, accum="direct", unroll=False),),
            ),
        ),
        (
            "placement",
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(TileSpec("k", 4, accum="direct", placement="child_of:i"),),
            ),
        ),
        ("tile-free", Schedule(loop_order=("i", "j", "k"))),
    ],
)
def test_each_semantic_decision_changes_the_identity(name, schedule) -> None:
    base = scheduled(build_matmul, BASE_SCHEDULE)
    variant = scheduled(build_matmul, schedule)
    assert request_identity(variant) != request_identity(base), name
    assert plan_schedule_digest(
        variant.normalized_cin, variant.verified_loop_plan
    ) != plan_schedule_digest(base.normalized_cin, base.verified_loop_plan), name


def test_panel_relayout_result_tile_and_parallel_enter_the_identity() -> None:
    """The sparse scheduling decisions all reach the canonical form."""

    reorder = scheduled(build_spmm, Schedule(loop_order=("i", "j", "k")))
    panel = scheduled(
        build_spmm,
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(TileSpec("j", 3, kind="panel", accum="direct"),),
            parallel_loop="i",
        ),
    )
    anchored = scheduled(
        build_spmm,
        Schedule(loop_order=("i", "j", "k"), parallel_loop="i"),
    )
    relayout = scheduled(
        build_spmm,
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(
                TileSpec(
                    "k",
                    4,
                    placement="outermost",
                    accum="direct",
                    unroll=False,
                ),
                TileSpec(
                    "j",
                    3,
                    placement="child_of:k_out",
                    kind="panel",
                    accum="direct",
                ),
            ),
            relayout=RelayoutSpec("B", "k", 4, scope_var="j"),
            parallel_loop="i",
        ),
    )
    heap = scheduled(
        build_spmm,
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(
                TileSpec(
                    "k",
                    4,
                    placement="outermost",
                    accum="heap",
                    unroll=False,
                ),
            ),
            parallel_loop="i",
        ),
    )
    identities = {
        request_identity(reorder),
        request_identity(panel),
        request_identity(anchored),
        request_identity(relayout),
        request_identity(heap),
    }
    assert len(identities) == 5
    panel_payload = json.loads(
        canonical_plan_dump(panel.normalized_cin, panel.verified_loop_plan)
    )
    assert panel_payload["panel_bounds"], "panel bound must enter the dump"
    assert panel_payload["parallel_loop"] is not None
    relayout_payload = json.loads(
        canonical_plan_dump(relayout.normalized_cin, relayout.verified_loop_plan)
    )
    assert relayout_payload["relayout"] == {
        "access_indices": [1, 2],
        "operand": 2,
        "operand_pack_level": 1,
        "operand_panel_level": 0,
        "pack_loop": {"index": 2, "part": "logical"},
        "panel_loop": {"index": 1, "part": "logical"},
        "row_loop": {"index": 0, "part": "logical"},
        "scope_loop": {"index": 1, "part": "logical"},
        "strip_width": 4,
    }
    heap_payload = json.loads(
        canonical_plan_dump(heap.normalized_cin, heap.verified_loop_plan)
    )
    assert heap_payload["result_tile"] == {
        "access_indices": [0, 2],
        "result": 0,
        "result_level": 1,
        "result_prefix": [0],
        "tile_loop": {"index": 2, "part": "logical"},
    }


def test_workspace_fact_changes_the_identity() -> None:
    auto = scheduled(build_matmul, Schedule())
    assert auto.verified_loop_plan.workspace is not None
    payload = json.loads(
        canonical_plan_dump(auto.normalized_cin, auto.verified_loop_plan)
    )
    assert payload["workspace"] is not None
    assert payload["workspace"]["dense"] is True

    # The two verified regblock arms of one SpMM differ exactly in the
    # recorded workspace and tile facts, so their canonical bytes differ.
    def spmm_arm(regblock):
        options = CompileOptions.from_environment(
            environ={},
            requested_schedule=Schedule(),
            regblock_override=regblock,
        )
        return Scheduler.apply_schedule(
            build_spmm(), Schedule(), compile_options=options
        )

    tile_free_arm = spmm_arm(False)
    stack_arm = spmm_arm(True)
    assert tile_free_arm.verified_loop_plan.workspace is None
    assert stack_arm.verified_loop_plan.workspace is not None
    assert canonical_plan_dump(
        tile_free_arm.normalized_cin, tile_free_arm.verified_loop_plan
    ) != canonical_plan_dump(stack_arm.normalized_cin, stack_arm.verified_loop_plan)


def test_auto_policy_is_outside_the_canonical_identity() -> None:
    """Identical decisions serialize identically regardless of the policy arm.

    Both regblock arms derive the decision-free plan for an elementwise
    program, so only the recorded ``auto_policy`` differs — and it must not
    reach the canonical bytes or the request identity.
    """

    def elementwise_arm(regblock):
        options = CompileOptions.from_environment(
            environ={},
            requested_schedule=Schedule(),
            regblock_override=regblock,
        )
        return Scheduler.apply_schedule(
            build_elementwise(), Schedule(), compile_options=options
        )

    off_arm = elementwise_arm(False)
    on_arm = elementwise_arm(True)
    assert off_arm.verified_loop_plan.auto_policy is not None
    assert on_arm.verified_loop_plan.auto_policy is not None
    assert (
        off_arm.verified_loop_plan.auto_policy != on_arm.verified_loop_plan.auto_policy
    )
    assert off_arm.verified_loop_plan.tiles == on_arm.verified_loop_plan.tiles == ()
    assert canonical_plan_dump(
        off_arm.normalized_cin, off_arm.verified_loop_plan
    ) == canonical_plan_dump(on_arm.normalized_cin, on_arm.verified_loop_plan)
    assert request_identity(off_arm, (4, 5), ELEMENTWISE_BINDINGS) == request_identity(
        on_arm, (4, 5), ELEMENTWISE_BINDINGS
    )


def test_cross_provenance_rule_and_schedule_digest_layer() -> None:
    """Provenance changes the request identity; the digest layer ignores it.

    An automatic elementwise plan and an explicit plan spelling the same
    identity order record the same schedule decisions, so their
    provenance-free schedule digests agree while their request identities
    differ — a different provenance selects a different gate and replay
    contract.
    """

    auto = scheduled(build_elementwise, Schedule())
    explicit = scheduled(build_elementwise, Schedule(loop_order=("i", "j")))
    assert auto.verified_loop_plan.provenance == "auto"
    assert explicit.verified_loop_plan.provenance == "explicit"
    assert plan_schedule_digest(
        auto.normalized_cin, auto.verified_loop_plan
    ) == plan_schedule_digest(explicit.normalized_cin, explicit.verified_loop_plan)
    assert request_identity(auto, (4, 5), ELEMENTWISE_BINDINGS) != request_identity(
        explicit, (4, 5), ELEMENTWISE_BINDINGS
    )


def test_tag_is_not_semantic_identity() -> None:
    tagged = scheduled(build_matmul, replace(BASE_SCHEDULE, tag="tuned-v3"))
    plain = scheduled(build_matmul, BASE_SCHEDULE)
    assert tagged.verified_loop_plan.tag == "tuned-v3"
    assert request_identity(tagged) == request_identity(plain)
    assert canonical_plan_dump(
        tagged.normalized_cin, tagged.verified_loop_plan
    ) == canonical_plan_dump(plain.normalized_cin, plain.verified_loop_plan)


def test_request_identity_covers_shapes_bindings_and_plan_presence() -> None:
    artifact = scheduled(build_matmul, BASE_SCHEDULE)
    base = request_identity(artifact)
    assert request_identity(artifact, (8, 6)) != base
    assert (
        request_identity(
            artifact,
            (4, 6),
            (((8, 5), F32), ((5, 6), F32)),
        )
        != base
    )
    assert (
        request_identity(
            artifact,
            (4, 6),
            (((4, 5), torch.float64), ((5, 6), torch.float64)),
        )
        != base
    )
    unscheduled = loopir_request_identity(
        artifact.normalized_cin,
        None,
        (4, 6),
        MATMUL_BINDINGS,
        compile_options=IDENTITY_OPTIONS,
    )
    assert unscheduled != base


def test_malformed_and_hostile_state_fails_closed() -> None:
    artifact = scheduled(build_matmul, BASE_SCHEDULE)
    cin = artifact.normalized_cin
    plan = artifact.verified_loop_plan

    forged = replace(plan)
    object.__setattr__(forged, "ghost", True)
    with pytest.raises(VerificationError):
        canonical_plan_dump(cin, forged)

    foreign_axis = replace(
        plan,
        workspace=WorkspaceInsertion(
            reduction_loop=LoopRef(plan.loop_order[0]),
            axis_loops=(LoopRef(plan.loop_order[1]),),
            dense=True,
        ),
    )
    with pytest.raises(Exception):
        canonical_plan_dump(cin, foreign_axis)

    with pytest.raises(VerificationError):
        loopir_request_dump(
            cin,
            plan,
            (4, True),
            MATMUL_BINDINGS,
            compile_options=IDENTITY_OPTIONS,
        )
    with pytest.raises(VerificationError):
        loopir_request_dump(
            cin,
            plan,
            (4, 6),
            (((4, "x"), F32), ((5, 6), F32)),
            compile_options=IDENTITY_OPTIONS,
        )
    with pytest.raises(VerificationError):
        loopir_request_dump(
            object(),
            plan,
            (4, 6),
            MATMUL_BINDINGS,
            compile_options=IDENTITY_OPTIONS,
        )


def test_request_dtype_identity_is_exact_and_never_calls_str() -> None:
    artifact = scheduled(build_matmul, BASE_SCHEDULE)

    class AliasDtype:
        calls = 0

        def __str__(self):
            self.calls += 1
            return "torch.float32"

    alias = AliasDtype()
    with pytest.raises(VerificationError, match="exact supported torch.dtype"):
        loopir_request_dump(
            artifact.normalized_cin,
            artifact.verified_loop_plan,
            (4, 6),
            (((4, 5), alias), ((5, 6), F32)),
            compile_options=IDENTITY_OPTIONS,
        )
    assert alias.calls == 0
    with pytest.raises(VerificationError, match="exact supported torch.dtype"):
        loopir_request_dump(
            artifact.normalized_cin,
            artifact.verified_loop_plan,
            (4, 6),
            (((4, 5), torch.int32), ((5, 6), torch.int32)),
            compile_options=IDENTITY_OPTIONS,
        )


@pytest.mark.parametrize(
    "extent",
    [-1, 2**63, 1 << 20000],
    ids=("negative", "above-int64", "hostile-huge"),
)
def test_request_extent_identity_is_bounded_before_json(extent) -> None:
    artifact = scheduled(build_matmul, BASE_SCHEDULE)
    with pytest.raises(VerificationError, match="nonnegative int64"):
        loopir_request_dump(
            artifact.normalized_cin,
            artifact.verified_loop_plan,
            (extent, 6),
            MATMUL_BINDINGS,
            compile_options=IDENTITY_OPTIONS,
        )


def test_request_identity_rejects_cyclic_unscheduled_cin() -> None:
    cyclic = build_elementwise()
    cyclic.stmt = cyclic
    with pytest.raises(VerificationError, match="cyclic_cin_structure"):
        loopir_request_dump(
            cyclic,
            None,
            (4, 5),
            ELEMENTWISE_BINDINGS,
            compile_options=IDENTITY_OPTIONS,
        )


def test_request_identity_rejects_cyclic_index_expression() -> None:
    cyclic = build_elementwise()
    outer = cyclic.index_var
    assert isinstance(cyclic.stmt, ForAll)
    inner = cyclic.stmt.index_var
    outer._expr = IndexVarAdd(outer, inner)
    with pytest.raises(VerificationError, match="cyclic_cin_structure"):
        loopir_request_dump(
            cyclic,
            None,
            (4, 5),
            ELEMENTWISE_BINDINGS,
            compile_options=IDENTITY_OPTIONS,
        )


def test_request_identity_never_executes_diverging_access_descriptor() -> None:
    """The shared structural boundary runs before identity's recursive walk."""

    cin = build_elementwise()
    assert isinstance(cin.stmt, ForAll)
    assignment = cin.stmt.stmt
    assert isinstance(assignment, TensorAssign)
    assert isinstance(assignment.rhs, BinaryOp)
    access = assignment.rhs.left
    assert isinstance(access, TensorAccess)
    reads = 0

    class HostileAccess(TensorAccess):
        @property
        def indices(self):  # type: ignore[override]
            nonlocal reads
            reads += 1
            return object.__getattribute__(self, "__dict__")["indices"]

    access.__class__ = HostileAccess

    with pytest.raises(VerificationError, match="invalid_cin_field"):
        loopir_request_dump(
            cin,
            None,
            (4, 5),
            ELEMENTWISE_BINDINGS,
            compile_options=IDENTITY_OPTIONS,
        )
    assert reads == 0


def test_request_identity_counts_unary_rhs_without_legacy_recursion() -> None:
    unary = build_elementwise()
    assert isinstance(unary.stmt, ForAll)
    assignment = unary.stmt.stmt
    assert isinstance(assignment, TensorAssign)
    assignment.rhs = UnaryOp(Operation.ADD, assignment.rhs)

    dump = loopir_request_dump(
        unary,
        None,
        (4, 5),
        ELEMENTWISE_BINDINGS,
        compile_options=IDENTITY_OPTIONS,
    )
    assert json.loads(dump)["schema"] == CANONICAL_REQUEST_SCHEMA
    with pytest.raises(VerificationError, match="cover every declared input"):
        loopir_request_dump(
            unary,
            None,
            (4, 5),
            ELEMENTWISE_BINDINGS[:1],
            compile_options=IDENTITY_OPTIONS,
        )


@pytest.mark.parametrize("forgery", ("nan", "wrong-type", "ghost"))
def test_request_identity_revalidates_nested_compile_options(forgery) -> None:
    artifact = scheduled(build_matmul, BASE_SCHEDULE)
    options = CompileOptions.from_environment()
    cost_model = options.scheduler.cost_model
    if forgery == "nan":
        object.__setattr__(cost_model, "alpha", float("nan"))
    elif forgery == "wrong-type":
        object.__setattr__(cost_model, "alpha", "2.975")
    else:
        object.__setattr__(cost_model, "ghost", True)

    with pytest.raises(VerificationError):
        request_identity(artifact, compile_options=options)


@pytest.mark.parametrize(
    ("owner", "field_name"),
    (
        ("options", "enabled_llir_passes"),
        ("build", "extra_cflags"),
        ("build", "direct_extension_cflags"),
        ("build", "special_kernel_cflags"),
        ("build", "extra_ldflags"),
        ("build", "torch_include_paths"),
    ),
)
def test_request_identity_rejects_constructor_normalized_lists(
    owner,
    field_name,
) -> None:
    artifact = scheduled(build_matmul, BASE_SCHEDULE)
    options = CompileOptions.from_environment()
    carrier = options if owner == "options" else options.build
    object.__setattr__(carrier, field_name, list(getattr(carrier, field_name)))

    with pytest.raises(VerificationError, match="stored as an exact tuple"):
        request_identity(artifact, compile_options=options)


def test_request_identity_does_not_consume_forged_option_iterators() -> None:
    artifact = scheduled(build_matmul, BASE_SCHEDULE)
    options = CompileOptions.from_environment()
    original = options.build.extra_cflags
    forged = iter(original)
    object.__setattr__(options.build, "extra_cflags", forged)

    with pytest.raises(VerificationError, match="stored as an exact tuple"):
        request_identity(artifact, compile_options=options)
    assert tuple(forged) == original


def test_request_identity_does_not_reprobe_darwin_snapshot() -> None:
    artifact = scheduled(build_matmul, BASE_SCHEDULE)
    options = CompileOptions.from_environment()
    if options.build.darwin_toolchain is None:
        pytest.skip("Darwin snapshot required for the host-probe regression")

    with patch(
        "scorch.compiler.compile_options.os.path.isdir",
        side_effect=AssertionError("identity must not re-probe the host"),
    ):
        request_identity(artifact, compile_options=options)


def test_request_identity_rejects_hostile_container_subclasses_without_calls() -> None:
    artifact = scheduled(build_matmul, BASE_SCHEDULE)

    class HostileList(list):
        calls = 0

        def __iter__(self):
            self.calls += 1
            raise RuntimeError("caller-controlled iteration")

        def __len__(self):
            self.calls += 1
            raise RuntimeError("caller-controlled length")

    hostile_result = HostileList((4, 6))
    hostile_bindings = HostileList(MATMUL_BINDINGS)
    hostile_pair = HostileList(MATMUL_BINDINGS[0])
    hostile_shape = HostileList((4, 5))
    probes = (
        (hostile_result, MATMUL_BINDINGS),
        ((4, 6), hostile_bindings),
        ((4, 6), (hostile_pair, MATMUL_BINDINGS[1])),
        ((4, 6), ((hostile_shape, F32), MATMUL_BINDINGS[1])),
    )
    for result_shape, bindings in probes:
        with pytest.raises(VerificationError):
            loopir_request_dump(
                artifact.normalized_cin,
                artifact.verified_loop_plan,
                result_shape,
                bindings,
                compile_options=IDENTITY_OPTIONS,
            )
    assert (
        sum(
            probe.calls
            for probe in (
                hostile_result,
                hostile_bindings,
                hostile_pair,
                hostile_shape,
            )
        )
        == 0
    )


def test_compile_options_are_part_of_the_request_identity() -> None:
    artifact = scheduled(build_matmul, BASE_SCHEDULE)
    comments = IDENTITY_OPTIONS
    no_comments = replace(comments, emit_comments=False)
    assert request_identity(artifact, compile_options=comments) != request_identity(
        artifact, compile_options=no_comments
    )

    schedule = Schedule(loop_order=("i", "j"))
    comments = replace(comments, requested_schedule=schedule)
    no_comments = replace(no_comments, requested_schedule=schedule)
    first = compile_cin_via_loopir(
        build_elementwise(),
        (4, 5),
        ELEMENTWISE_BINDINGS,
        compile_options=comments,
    )
    second = compile_cin_via_loopir(
        build_elementwise(),
        (4, 5),
        ELEMENTWISE_BINDINGS,
        compile_options=no_comments,
    )
    assert first.cpp_source != second.cpp_source
    assert first.request_dump != second.request_dump
    assert first.request_identity != second.request_identity


def test_identity_failure_is_owned_by_the_frontend_request_stage() -> None:
    context = CompilationContext(IDENTITY_OPTIONS)
    huge = 1 << 20000
    with pytest.raises(VerificationError, match="nonnegative int64"):
        compile_cin_via_loopir(
            build_elementwise(),
            (huge, 5),
            (((huge, 5), F32), ((huge, 5), F32)),
            compile_options=IDENTITY_OPTIONS,
            compilation_context=context,
        )
    assert (
        context._failed_stage_id
        is CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION
    )
    assert context._active_stage_tokens == ()
    assert [record.stage_id for record in context.stage_run_records] == [
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION,
    ]


def test_identity_is_the_hash_of_the_retained_dump() -> None:
    """Collision handling: the dump is authoritative, the digest is derived."""

    artifact = scheduled(build_matmul, BASE_SCHEDULE)
    dump = loopir_request_dump(
        artifact.normalized_cin,
        artifact.verified_loop_plan,
        (4, 6),
        MATMUL_BINDINGS,
        compile_options=IDENTITY_OPTIONS,
    )
    payload = json.loads(dump)
    assert payload["schema"] == CANONICAL_REQUEST_SCHEMA
    assert (
        request_identity(artifact) == hashlib.sha256(dump.encode("utf-8")).hexdigest()
    )


def test_pipeline_owns_the_request_identity_at_the_compile_boundary() -> None:
    """Repeated strangler builds carry equal identities; stages unchanged."""

    options = CompileOptions.from_environment(
        requested_schedule=Schedule(loop_order=("i", "j", "k"))
    )
    context = CompilationContext(options)
    first = compile_cin_via_loopir(
        build_matmul(),
        (4, 6),
        MATMUL_BINDINGS,
        compile_options=options,
        compilation_context=context,
    )
    second = compile_cin_via_loopir(
        build_matmul(),
        (4, 6),
        MATMUL_BINDINGS,
        compile_options=CompileOptions.from_environment(
            requested_schedule=Schedule(loop_order=("i", "j", "k"))
        ),
    )
    assert first.request_identity == second.request_identity
    assert len(first.request_identity) == 64
    assert first.request_dump == second.request_dump
    assert (
        first.request_identity
        == hashlib.sha256(first.request_dump.encode("utf-8")).hexdigest()
    )
    # Identity computation adds no compiler stage.
    stages = [record.stage_id for record in context.stage_run_records]
    assert stages == [
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION,
        CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION,
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        CompilerStageId.CIN_TO_LOOPIR_LOWERING,
        CompilerStageId.LOOPIR_SCHEDULE_APPLICATION,
        CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
        CompilerStageId.LLIR_TO_CPP_GENERATION,
    ]

    unscheduled = compile_cin_via_loopir(
        build_elementwise(),
        (4, 5),
        ELEMENTWISE_BINDINGS,
    )
    assert unscheduled.request_identity != first.request_identity
    auto = compile_cin_via_loopir(
        build_spmm(),
        (4, 6),
        SPMM_BINDINGS,
        compile_options=CompileOptions.from_environment(requested_schedule=Schedule()),
    )
    assert auto.schedule is not None
    assert auto.schedule.plan.provenance == "auto"
    assert auto.request_identity not in (
        first.request_identity,
        unscheduled.request_identity,
    )
    with pytest.raises(TypeError, match="must hash the retained request_dump"):
        replace(first, request_identity="0" * 64)
