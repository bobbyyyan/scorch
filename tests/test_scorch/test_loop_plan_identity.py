"""Canonical LoopPlan serialization and the strangler request identity."""

import hashlib
import json
from dataclasses import replace

import pytest
import torch

from scorch.compiler.cin import (
    BinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
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
from scorch.compiler.scheduler import Schedule, Scheduler, TileSpec
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir

F32 = torch.float32
MATMUL_BINDINGS = (((4, 5), F32), ((5, 6), F32))
ELEMENTWISE_BINDINGS = (((4, 5), F32), ((4, 5), F32))
SPMM_BINDINGS = (((4, 5), F32), ((5, 6), F32))


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


def request_identity(artifact, result_shape=(4, 6), bindings=MATMUL_BINDINGS):
    return loopir_request_identity(
        artifact.normalized_cin,
        artifact.verified_loop_plan,
        result_shape,
        bindings,
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


def test_panel_relayout_result_tile_and_parallel_change_the_identity() -> None:
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
    identities = {
        request_identity(reorder),
        request_identity(panel),
        request_identity(anchored),
    }
    assert len(identities) == 3
    panel_payload = json.loads(
        canonical_plan_dump(panel.normalized_cin, panel.verified_loop_plan)
    )
    assert panel_payload["panel_bounds"], "panel bound must enter the dump"
    assert panel_payload["parallel_loop"] is not None


def test_workspace_fact_changes_the_identity() -> None:
    auto = scheduled(build_matmul, Schedule())
    assert auto.verified_loop_plan.workspace is not None
    payload = json.loads(
        canonical_plan_dump(auto.normalized_cin, auto.verified_loop_plan)
    )
    assert payload["workspace"] is not None
    assert payload["workspace"]["dense"] is True

    stripped = replace(
        auto.verified_loop_plan,
        tiles=(),
        workspace=None,
    )
    assert canonical_plan_dump(
        auto.normalized_cin, auto.verified_loop_plan
    ) != canonical_plan_dump(auto.normalized_cin, stripped)


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
        artifact.normalized_cin, None, (4, 6), MATMUL_BINDINGS
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
        loopir_request_dump(cin, plan, (4, True), MATMUL_BINDINGS)
    with pytest.raises(VerificationError):
        loopir_request_dump(cin, plan, (4, 6), (((4, "x"), F32), ((5, 6), F32)))
    with pytest.raises(VerificationError):
        loopir_request_dump(object(), plan, (4, 6), MATMUL_BINDINGS)


def test_identity_is_the_hash_of_the_retained_dump() -> None:
    """Collision handling: the dump is authoritative, the digest is derived."""

    artifact = scheduled(build_matmul, BASE_SCHEDULE)
    dump = loopir_request_dump(
        artifact.normalized_cin,
        artifact.verified_loop_plan,
        (4, 6),
        MATMUL_BINDINGS,
    )
    payload = json.loads(dump)
    assert payload["schema"] == CANONICAL_REQUEST_SCHEMA
    assert (
        request_identity(artifact) == hashlib.sha256(dump.encode("utf-8")).hexdigest()
    )


def test_pipeline_owns_the_request_identity_at_the_compile_boundary() -> None:
    """Repeated strangler builds carry equal identities; stages unchanged."""

    from scorch.compiler.compilation_context import (
        CompilationContext,
        CompilerStageId,
    )

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
