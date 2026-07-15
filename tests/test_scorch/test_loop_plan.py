from dataclasses import FrozenInstanceError, replace
from typing import Tuple, cast
from unittest.mock import patch

import pytest

from scorch.compiler.cin import (
    ForAll,
    IndexStmt,
    IndexVar,
    TensorAssign,
    TensorVar,
    Where,
)
from scorch.compiler.cin_analysis import analyze_cin, canonical_cin_dump, normalize_cin
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.diagnostics import (
    InvalidSchedule,
    UnsupportedFeature,
    VerificationError,
)
from scorch.compiler.identity import IndexId, SymbolId
from scorch.compiler.loop_plan import (
    LoopPart,
    LoopPlan,
    LoopPlacement,
    LoopRef,
    LoopTile,
    OperandRelayout,
    PanelBound,
    PlacementKind,
    ResultTile,
    ScheduledCIN,
    verify_loop_plan,
    verify_scheduled_cin,
)
from scorch.compiler.legacy_cin_adapter import legacy_cin_working_copy
from scorch.compiler.scheduler import (
    RelayoutSpec,
    Schedule,
    Scheduler,
    TileSpec,
    regblock_force,
)


def _build_spmm() -> ForAll:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    result = TensorVar("C", fmt="dd")
    sparse = TensorVar("A", fmt="ds")
    dense = TensorVar("B", fmt="dd")
    result[i, k] = sparse[i, j] * dense[j, k]
    return ForAll(i, ForAll(j, ForAll(k, result._assignment)))


def _build_dense_matmul() -> ForAll:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    result = TensorVar("C", fmt="dd")
    left = TensorVar("A", fmt="dd")
    right = TensorVar("B", fmt="dd")
    result[i, k] = left[i, j] * right[j, k]
    return ForAll(i, ForAll(j, ForAll(k, result._assignment)))


def _build_nondefault_dense_result() -> ForAll:
    i, k = IndexVar("i"), IndexVar("k")
    result = TensorVar("C", fmt="dd", mode_order=[1, 0])
    source = TensorVar("A", fmt="dd")
    result[i, k] = source[i, k]
    return ForAll(i, ForAll(k, result._assignment))


def _build_nondefault_sparse_result(fmt: str) -> ForAll:
    i, k = IndexVar("i"), IndexVar("k")
    result = TensorVar("C", fmt=fmt, mode_order=[1, 0])
    source = TensorVar("A", fmt="dd")
    result[i, k] = source[i, k]
    return ForAll(i, ForAll(k, result._assignment))


def _build_nested_sparse_access(fmt: str) -> ForAll:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    result = TensorVar("C", fmt="d")
    source = TensorVar("A", fmt=fmt, mode_order=[2, 0, 1])
    result[k] = source[i, j, k]
    return ForAll(k, ForAll(i, ForAll(j, result._assignment)))


def _build_singleton_access() -> ForAll:
    i, k = IndexVar("i"), IndexVar("k")
    result = TensorVar("C", fmt="d")
    source = TensorVar("A", fmt=["d", "singleton"])
    result[k] = source[i, k]
    return ForAll(i, ForAll(k, result._assignment))


def _build_sparse_vector_access() -> ForAll:
    i = IndexVar("i")
    result = TensorVar("C", fmt="d")
    source = TensorVar("A", fmt="s")
    result[i] = source[i]
    return ForAll(i, result._assignment)


def _packed_schedule(accum: str) -> Schedule:
    return Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                4,
                placement="outermost",
                accum=accum,
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
        tag=f"legality-relayout-{accum}",
    )


def _index_ids_by_name(cin: IndexStmt) -> dict[str, IndexId]:
    analysis = analyze_cin(cin)
    return {
        definition.display_name: index_id
        for index_id, definition in analysis.index_definitions.items()
        if definition.bindings
    }


def _symbol_ids_by_name(cin: IndexStmt) -> dict[str, SymbolId]:
    analysis = analyze_cin(cin)
    return {
        definition.display_name: symbol_id
        for symbol_id, definition in analysis.symbol_definitions.items()
    }


def _affine_tile(
    index_id: IndexId,
    placement: LoopPlacement,
    *,
    accumulation: str = "direct",
) -> LoopTile:
    return LoopTile(
        loop=LoopRef(index_id),
        width=4,
        placement=placement,
        parallel=False,
        kind="affine",
        accumulation=accumulation,
        unroll=False,
    )


def _assert_plan_rejected(
    cin: IndexStmt,
    plan: LoopPlan,
    expected: type[Exception],
    diagnostic_code: str,
) -> None:
    with pytest.raises(expected, match=diagnostic_code):
        verify_loop_plan(cin, plan)
    with pytest.raises(expected, match=diagnostic_code):
        verify_scheduled_cin(ScheduledCIN(cin, plan))


def _lower_to_cpp(scheduled: ScheduledCIN) -> str:
    llir = CINLowerer().lower_IndexStmt(scheduled)
    return LLIRLowerer().lower_llir(llir)


def test_logical_entities_have_stable_typed_identity() -> None:
    first_index = IndexVar("same")
    second_index = IndexVar("same")
    first_tensor = TensorVar("same", fmt="d")
    second_tensor = TensorVar("same", fmt="d")

    assert first_index.index_id != second_index.index_id
    assert first_tensor.symbol_id != second_tensor.symbol_id
    assert isinstance(first_index.index_id, IndexId)


def test_schedule_returns_frozen_verified_identity_based_boundary() -> None:
    cin = _build_spmm()
    scheduled = Scheduler.apply_schedule(
        cin,
        Schedule(loop_order=("i", "k", "j"), tag="reordered"),
    )

    assert isinstance(scheduled, ScheduledCIN)
    assert verify_scheduled_cin(scheduled) is scheduled
    source_ids = {index_var.index_id for index_var in cin.index_vars}
    assert set(scheduled.verified_loop_plan.loop_order) == source_ids
    assert not hasattr(scheduled.normalized_cin, "explicit_schedule")
    assert not hasattr(scheduled.normalized_cin, "panel_bounds")
    assert not hasattr(scheduled.normalized_cin, "relayout_plan")
    assert not hasattr(scheduled.normalized_cin, "result_tile_plan")
    assert all(
        access.tensor._assignment is None
        for access in scheduled.normalized_cin.tensor_accesses
    )
    assert all(
        not index_var.tensor_accesses
        for index_var in scheduled.normalized_cin.index_vars
    )

    with pytest.raises(FrozenInstanceError):
        scheduled.verified_loop_plan.tag = "mutated"  # type: ignore[misc]


def test_loop_plan_verifier_rejects_dangling_loop_reference() -> None:
    scheduled = Scheduler.apply_schedule(_build_spmm(), Schedule())
    invalid_plan = replace(
        scheduled.verified_loop_plan,
        parallel_loop=LoopRef(IndexId(10**9)),
    )
    invalid = ScheduledCIN(scheduled.normalized_cin, invalid_plan)

    with pytest.raises(VerificationError, match="unknown IndexId"):
        verify_scheduled_cin(invalid)


def test_loop_plan_verifier_rejects_dangling_symbol_reference() -> None:
    scheduled = Scheduler.apply_schedule(_build_spmm(), Schedule())
    first_loop = LoopRef(scheduled.verified_loop_plan.loop_order[0])
    invalid_plan = replace(
        scheduled.verified_loop_plan,
        result_tile=ResultTile(SymbolId(10**9), first_loop, 0, (), ()),
    )
    invalid = ScheduledCIN(scheduled.normalized_cin, invalid_plan)

    with pytest.raises(VerificationError, match="unknown SymbolId"):
        verify_scheduled_cin(invalid)


def test_loop_plan_verifier_requires_complete_loop_order() -> None:
    scheduled = Scheduler.apply_schedule(_build_spmm(), Schedule())
    incomplete_plan = replace(
        scheduled.verified_loop_plan,
        loop_order=scheduled.verified_loop_plan.loop_order[:-1],
    )

    with pytest.raises(
        VerificationError,
        match="loop_order must contain every bound IndexId exactly once",
    ):
        verify_scheduled_cin(ScheduledCIN(scheduled.normalized_cin, incomplete_plan))


def test_loop_plan_rejects_unordered_loop_order_input() -> None:
    scheduled = Scheduler.apply_schedule(_build_spmm(), Schedule())

    with pytest.raises(TypeError, match="must be a tuple or list"):
        LoopPlan(
            loop_order=cast(
                Tuple[IndexId, ...],
                set(scheduled.verified_loop_plan.loop_order),
            )
        )


def test_loop_plan_verifier_fails_closed_on_malformed_panel_bound() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(TileSpec("j", 32, kind="panel", accum="direct"),),
            parallel_loop="i",
        ),
    )
    plan = scheduled.verified_loop_plan
    malformed_bound = replace(
        plan.panel_bounds[0],
        loop=cast(LoopRef, object()),
    )
    malformed_plan = replace(plan, panel_bounds=(malformed_bound,))

    with pytest.raises(
        VerificationError,
        match=r"panel_bounds\[0\]\.loop must be a LoopRef",
    ):
        verify_scheduled_cin(ScheduledCIN(scheduled.normalized_cin, malformed_plan))


def test_tuple_valued_loop_plan_inputs_are_detached_from_mutable_callers() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(TileSpec("j", 32, kind="panel", accum="direct"),),
            parallel_loop="i",
        ),
    )
    source = scheduled.verified_loop_plan
    loop_order = list(source.loop_order)
    tiles = list(source.tiles)
    panel_bounds = list(source.panel_bounds)
    plan = LoopPlan(
        loop_order=cast(Tuple[IndexId, ...], loop_order),
        tiles=cast(Tuple[LoopTile, ...], tiles),
        panel_bounds=cast(Tuple[PanelBound, ...], panel_bounds),
        relayout=source.relayout,
        result_tile=source.result_tile,
        parallel_loop=source.parallel_loop,
        provenance=source.provenance,
        tag=source.tag,
    )

    loop_order.reverse()
    tiles.clear()
    panel_bounds.clear()

    assert plan.loop_order == source.loop_order
    assert plan.tiles == source.tiles
    assert plan.panel_bounds == source.panel_bounds
    replay = ScheduledCIN(scheduled.normalized_cin, plan)
    assert verify_scheduled_cin(replay) is replay


def test_nested_loop_plan_sequences_are_detached_from_mutable_callers() -> None:
    scheduled = Scheduler.apply_schedule(_build_spmm(), Schedule())
    loop_order = scheduled.verified_loop_plan.loop_order
    relayout_indices = list(loop_order)
    result_prefix = list(loop_order[:1])
    result_indices = list(loop_order)

    relayout = OperandRelayout(
        operand_id=SymbolId(1),
        pack_loop=LoopRef(loop_order[0]),
        panel_loop=LoopRef(loop_order[1]),
        scope_loop=LoopRef(loop_order[0]),
        row_loop=LoopRef(loop_order[0]),
        strip_width=4,
        access_indices=cast(Tuple[IndexId, ...], relayout_indices),
        operand_panel_level=0,
        operand_pack_level=1,
    )
    result_tile = ResultTile(
        result_id=SymbolId(2),
        tile_loop=LoopRef(loop_order[0]),
        result_level=0,
        result_prefix=cast(Tuple[IndexId, ...], result_prefix),
        access_indices=cast(Tuple[IndexId, ...], result_indices),
    )

    relayout_indices.clear()
    result_prefix.clear()
    result_indices.clear()

    assert relayout.access_indices == loop_order
    assert result_tile.result_prefix == loop_order[:1]
    assert result_tile.access_indices == loop_order


def test_scheduling_is_deterministic_and_does_not_mutate_input() -> None:
    cin = _build_spmm()
    original_text = str(cin)
    original_assignment = cin.stmt.stmt.stmt
    original_parent = original_assignment.parent
    original_indices = tuple(
        (index_var.index_id, index_var.name, index_var._expr, index_var.is_tiled)
        for index_var in cin.index_vars
    )

    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(TileSpec("k", 4, placement="child_of:i", accum="direct"),),
        tag="deterministic",
    )
    first = Scheduler.apply_schedule(cin, schedule)
    second = Scheduler.apply_schedule(cin, schedule)

    assert first.verified_loop_plan == second.verified_loop_plan
    assert _lower_to_cpp(first) == _lower_to_cpp(second)
    assert str(cin) == original_text
    assert original_assignment.parent is original_parent
    assert (
        tuple(
            (index_var.index_id, index_var.name, index_var._expr, index_var.is_tiled)
            for index_var in cin.index_vars
        )
        == original_indices
    )


def test_one_cin_can_be_scheduled_independently_two_ways() -> None:
    cin = _build_spmm()
    source_dump = canonical_cin_dump(cin)
    reordered = Scheduler.apply_schedule(
        cin, Schedule(loop_order=("i", "k", "j"), tag="reordered")
    )
    tiled = Scheduler.apply_schedule(
        cin,
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(TileSpec("k", 4, accum="direct"),),
            tag="tiled",
        ),
    )

    assert reordered.verified_loop_plan != tiled.verified_loop_plan
    assert canonical_cin_dump(reordered.normalized_cin) == source_dump
    assert canonical_cin_dump(tiled.normalized_cin) == source_dump
    assert "k_out" not in str(reordered.normalized_cin)
    assert "k_out" not in str(tiled.normalized_cin)
    assert "k_out" not in str(cin)
    assert _lower_to_cpp(reordered) != _lower_to_cpp(tiled)


def test_auto_plan_replay_does_not_rerun_loop_order_policy() -> None:
    cin = _build_spmm()

    def select_ikj(working, costs):
        del costs
        by_name = {index_var.name: index_var for index_var in working.index_vars}
        return [by_name["i"], by_name["k"], by_name["j"]]

    with patch.object(Scheduler, "select_loop_order", side_effect=select_ikj):
        scheduled = Scheduler.apply_schedule(cin, Schedule())

    with patch.object(
        Scheduler,
        "select_loop_order",
        side_effect=AssertionError("auto policy must not run during replay"),
    ):
        working = legacy_cin_working_copy(
            scheduled.normalized_cin,
            scheduled.verified_loop_plan,
        )

    replayed_ids = []
    current = working
    while isinstance(current, ForAll):
        replayed_ids.append(current.index_var.index_id)
        current = current.stmt
    assert tuple(replayed_ids) == scheduled.verified_loop_plan.loop_order


def test_auto_plan_replay_is_independent_of_regblock_context() -> None:
    with regblock_force(True):
        direct = Scheduler.auto_schedule(_build_spmm())
        scheduled = Scheduler.apply_schedule(_build_spmm(), Schedule())

    assert scheduled.verified_loop_plan.tiles
    with regblock_force(False):
        working = legacy_cin_working_copy(
            scheduled.normalized_cin,
            scheduled.verified_loop_plan,
        )

    assert str(working) == str(direct)
    assert tuple(tile.size for tile in working.get_tile_size_vars()) == tuple(
        tile.width for tile in scheduled.verified_loop_plan.tiles
    )


def test_forged_result_order_fails_both_artifact_verifiers() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_spmm(), Schedule(loop_order=("i", "j", "k"))
    )
    ids = _index_ids_by_name(scheduled.normalized_cin)
    forged = replace(
        scheduled.verified_loop_plan,
        loop_order=(ids["k"], ids["i"], ids["j"]),
    )

    _assert_plan_rejected(
        scheduled.normalized_cin,
        forged,
        InvalidSchedule,
        "result_storage_order",
    )


def test_forged_sparse_child_before_parent_fails_both_artifact_verifiers() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_spmm(), Schedule(loop_order=("i", "j", "k"))
    )
    ids = _index_ids_by_name(scheduled.normalized_cin)
    forged = replace(
        scheduled.verified_loop_plan,
        loop_order=(ids["j"], ids["i"], ids["k"]),
    )

    _assert_plan_rejected(
        scheduled.normalized_cin,
        forged,
        InvalidSchedule,
        "sparse_parent_dominance",
    )


def test_forged_parallel_reduction_fails_both_artifact_verifiers() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_spmm(), Schedule(loop_order=("i", "j", "k"))
    )
    ids = _index_ids_by_name(scheduled.normalized_cin)
    forged = replace(
        scheduled.verified_loop_plan,
        parallel_loop=LoopRef(ids["j"]),
    )

    _assert_plan_rejected(
        scheduled.normalized_cin,
        forged,
        InvalidSchedule,
        "parallel_reduction",
    )


def test_parallel_loop_must_partition_every_result_write() -> None:
    i, k = IndexVar("i"), IndexVar("k")
    first_result = TensorVar("C", fmt="d")
    second_result = TensorVar("D", fmt="d")
    first_source = TensorVar("A", fmt="d")
    second_source = TensorVar("B", fmt="d")
    cin = normalize_cin(
        ForAll(
            i,
            ForAll(
                k,
                Where(
                    producer=TensorAssign(first_result[i], first_source[i]),
                    consumer=TensorAssign(second_result[k], second_source[k]),
                ),
            ),
        )
    )
    plan = LoopPlan(
        loop_order=(i.index_id, k.index_id),
        parallel_loop=LoopRef(i.index_id),
        provenance="explicit",
    )

    _assert_plan_rejected(
        cin,
        plan,
        InvalidSchedule,
        "parallel_result_race",
    )


@pytest.mark.parametrize(
    "schedule",
    (
        Schedule(loop_order=("k", "i", "j")),
        Schedule(loop_order=("j", "i", "k")),
        Schedule(loop_order=("i", "j", "k"), parallel_loop="j"),
    ),
)
def test_public_schedule_and_direct_plans_share_semantic_legality(
    schedule: Schedule,
) -> None:
    with pytest.raises(InvalidSchedule):
        Scheduler.apply_schedule(_build_spmm(), schedule)


def test_public_and_direct_paths_reject_sparse_affine_target() -> None:
    cin = _build_sparse_vector_access()
    with pytest.raises(UnsupportedFeature):
        Scheduler.apply_schedule(
            cin,
            Schedule(
                loop_order=("i",),
                tiles=(TileSpec("i", 4, accum="direct"),),
            ),
        )

    scheduled = Scheduler.apply_schedule(cin, Schedule(loop_order=("i",)))
    ids = _index_ids_by_name(scheduled.normalized_cin)
    forged = replace(
        scheduled.verified_loop_plan,
        tiles=(
            _affine_tile(
                ids["i"],
                LoopPlacement(PlacementKind.OUTERMOST),
            ),
        ),
    )
    _assert_plan_rejected(
        scheduled.normalized_cin,
        forged,
        UnsupportedFeature,
        "sparse_affine_tile",
    )


def test_nondefault_dense_result_order_is_physical() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_nondefault_dense_result(),
        Schedule(loop_order=("k", "i")),
    )
    assert verify_scheduled_cin(scheduled) is scheduled
    ids = _index_ids_by_name(scheduled.normalized_cin)
    forged = replace(
        scheduled.verified_loop_plan,
        loop_order=(ids["i"], ids["k"]),
    )

    _assert_plan_rejected(
        scheduled.normalized_cin,
        forged,
        InvalidSchedule,
        "result_storage_order",
    )


@pytest.mark.parametrize("fmt", ("ds", "do"))
def test_nondefault_sparse_result_order_is_physical(fmt: str) -> None:
    scheduled = Scheduler.apply_schedule(
        _build_nondefault_sparse_result(fmt),
        Schedule(loop_order=("k", "i")),
    )
    assert verify_scheduled_cin(scheduled) is scheduled
    ids = _index_ids_by_name(scheduled.normalized_cin)
    forged = replace(
        scheduled.verified_loop_plan,
        loop_order=(ids["i"], ids["k"]),
    )

    _assert_plan_rejected(
        scheduled.normalized_cin,
        forged,
        InvalidSchedule,
        "result_storage_order",
    )


@pytest.mark.parametrize("fmt", ("dss", "doo"))
def test_nondefault_nested_sparse_storage_requires_every_parent(
    fmt: str,
) -> None:
    scheduled = Scheduler.apply_schedule(
        _build_nested_sparse_access(fmt),
        Schedule(loop_order=("k", "i", "j")),
    )
    assert verify_scheduled_cin(scheduled) is scheduled
    ids = _index_ids_by_name(scheduled.normalized_cin)
    forged = replace(
        scheduled.verified_loop_plan,
        loop_order=(ids["i"], ids["k"], ids["j"]),
    )

    _assert_plan_rejected(
        scheduled.normalized_cin,
        forged,
        InvalidSchedule,
        "sparse_parent_dominance",
    )


def test_singleton_storage_is_rejected_as_unsupported() -> None:
    cin = normalize_cin(_build_singleton_access())
    ids = _index_ids_by_name(cin)
    plan = LoopPlan(
        loop_order=(ids["i"], ids["k"]),
        provenance="explicit",
    )

    _assert_plan_rejected(
        cin,
        plan,
        UnsupportedFeature,
        "singleton_level",
    )


def test_incomplete_free_reduction_classification_is_a_verification_error() -> None:
    i, unused = IndexVar("i"), IndexVar("unused")
    result = TensorVar("C", fmt="d")
    source = TensorVar("A", fmt="d")
    result[i] = source[i]
    cin = ForAll(i, ForAll(unused, result._assignment))
    plan = LoopPlan(
        loop_order=(i.index_id, unused.index_id),
        provenance="explicit",
    )

    _assert_plan_rejected(
        cin,
        plan,
        VerificationError,
        "unused_index_binding",
    )


def test_reduction_tiles_and_accumulator_lifetimes_are_artifact_verified() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_dense_matmul(), Schedule(loop_order=("i", "j", "k"))
    )
    cin = scheduled.normalized_cin
    ids = _index_ids_by_name(cin)
    base = scheduled.verified_loop_plan

    reduction_tile = replace(
        base,
        tiles=(
            _affine_tile(
                ids["j"],
                LoopPlacement(PlacementKind.OUTERMOST),
            ),
        ),
    )
    _assert_plan_rejected(
        cin,
        reduction_tile,
        UnsupportedFeature,
        "affine_reduction_tile",
    )

    invalid_stack_lifetime = replace(
        base,
        tiles=(
            _affine_tile(
                ids["i"],
                LoopPlacement(PlacementKind.OUTERMOST),
                accumulation="stack",
            ),
        ),
    )
    _assert_plan_rejected(
        cin,
        invalid_stack_lifetime,
        UnsupportedFeature,
        "stack_accumulator_lifetime",
    )

    invalid_auto_lifetime = replace(
        base,
        loop_order=(ids["i"], ids["k"], ids["j"]),
        tiles=(
            _affine_tile(
                ids["i"],
                LoopPlacement(PlacementKind.OUTERMOST),
            ),
        ),
        provenance="auto",
    )
    _assert_plan_rejected(
        cin,
        invalid_auto_lifetime,
        InvalidSchedule,
        "auto_accumulator_lifetime",
    )

    valid_stack = Scheduler.apply_schedule(
        _build_dense_matmul(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(
                TileSpec(
                    "k",
                    4,
                    placement="child_of:i",
                    accum="stack",
                ),
            ),
        ),
    )
    assert verify_scheduled_cin(valid_stack) is valid_stack

    valid_auto = Scheduler.apply_schedule(_build_dense_matmul(), Schedule())
    reduction_ids = set(analyze_cin(valid_auto.normalized_cin).reduction_index_ids)
    assert any(
        tile.kind == "affine" and tile.loop.index_id in reduction_ids
        for tile in valid_auto.verified_loop_plan.tiles
    )
    assert verify_scheduled_cin(valid_auto) is valid_auto


def test_parallel_post_reduction_workspace_scope_is_rejected() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_dense_matmul(), Schedule(loop_order=("i", "j", "k"))
    )
    cin = scheduled.normalized_cin
    ids = _index_ids_by_name(cin)
    forged = replace(
        scheduled.verified_loop_plan,
        tiles=(
            _affine_tile(
                ids["i"],
                LoopPlacement(PlacementKind.OUTERMOST),
            ),
        ),
        parallel_loop=LoopRef(ids["k"]),
        provenance="auto",
    )

    _assert_plan_rejected(
        cin,
        forged,
        InvalidSchedule,
        "parallel_workspace_scope",
    )


def test_affine_placement_rejects_self_future_and_out_of_range_parents() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_spmm(), Schedule(loop_order=("i", "j", "k"))
    )
    cin = scheduled.normalized_cin
    ids = _index_ids_by_name(cin)
    base = scheduled.verified_loop_plan

    self_parent = replace(
        base,
        tiles=(
            _affine_tile(
                ids["k"],
                LoopPlacement(PlacementKind.CHILD_OF, parent=LoopRef(ids["k"])),
            ),
        ),
    )
    _assert_plan_rejected(
        cin,
        self_parent,
        InvalidSchedule,
        "tile_outer_dominance",
    )

    future_parent = replace(
        base,
        tiles=(
            _affine_tile(
                ids["k"],
                LoopPlacement(
                    PlacementKind.CHILD_OF,
                    parent=LoopRef(ids["i"], LoopPart.OUTER),
                ),
            ),
            _affine_tile(
                ids["i"],
                LoopPlacement(PlacementKind.OUTERMOST),
            ),
        ),
    )
    _assert_plan_rejected(
        cin,
        future_parent,
        InvalidSchedule,
        "placement_parent_scope",
    )

    out_of_range = replace(
        base,
        tiles=(
            _affine_tile(
                ids["k"],
                LoopPlacement(PlacementKind.AT_DEPTH, depth=99),
            ),
        ),
    )
    _assert_plan_rejected(
        cin,
        out_of_range,
        InvalidSchedule,
        "placement_depth_scope",
    )


def test_affine_placement_accepts_logical_and_prior_derived_parents() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_spmm(), Schedule(loop_order=("i", "j", "k"))
    )
    cin = scheduled.normalized_cin
    ids = _index_ids_by_name(cin)
    base = scheduled.verified_loop_plan
    logical_parent = replace(
        base,
        tiles=(
            _affine_tile(
                ids["k"],
                LoopPlacement(PlacementKind.CHILD_OF, parent=LoopRef(ids["i"])),
            ),
        ),
    )
    derived_parent = replace(
        base,
        tiles=(
            _affine_tile(
                ids["i"],
                LoopPlacement(PlacementKind.OUTERMOST),
            ),
            _affine_tile(
                ids["k"],
                LoopPlacement(
                    PlacementKind.CHILD_OF,
                    parent=LoopRef(ids["i"], LoopPart.OUTER),
                ),
            ),
        ),
    )

    assert verify_loop_plan(cin, logical_parent) is logical_parent
    assert verify_loop_plan(cin, derived_parent) is derived_parent


def test_panel_plan_checks_bound_parallelism_and_placement() -> None:
    with pytest.raises(InvalidSchedule, match="outside its parallel"):
        Scheduler.apply_schedule(
            _build_spmm(),
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(
                    TileSpec("i", 4, placement="outermost", accum="direct"),
                    TileSpec(
                        "j",
                        4,
                        placement="child_of:i_out",
                        kind="panel",
                        accum="direct",
                    ),
                ),
                parallel_loop="i",
            ),
        )

    scheduled = Scheduler.apply_schedule(
        _build_spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(TileSpec("j", 4, kind="panel", accum="direct"),),
            parallel_loop="i",
        ),
    )
    assert verify_scheduled_cin(scheduled) is scheduled
    cin = scheduled.normalized_cin
    ids = _index_ids_by_name(cin)
    symbols = _symbol_ids_by_name(cin)
    plan = scheduled.verified_loop_plan
    bound = plan.panel_bounds[0]

    wrong_tensor = replace(
        plan,
        panel_bounds=(replace(bound, tensor_id=symbols["A"]),),
    )
    _assert_plan_rejected(
        cin,
        wrong_tensor,
        InvalidSchedule,
        "panel_bound_access",
    )

    wrong_level = replace(
        plan,
        panel_bounds=(replace(bound, level=1),),
    )
    _assert_plan_rejected(
        cin,
        wrong_level,
        InvalidSchedule,
        "panel_bound_access",
    )

    wrong_parallel = replace(plan, parallel_loop=LoopRef(ids["k"]))
    _assert_plan_rejected(
        cin,
        wrong_parallel,
        InvalidSchedule,
        "panel_parallel_parent",
    )

    panel_tile = plan.tiles[0]
    wrong_placement = replace(
        plan,
        tiles=(
            replace(
                panel_tile,
                placement=LoopPlacement(
                    PlacementKind.CHILD_OF,
                    parent=LoopRef(ids["i"]),
                ),
            ),
        ),
    )
    _assert_plan_rejected(
        cin,
        wrong_placement,
        InvalidSchedule,
        "panel_parent_placement",
    )

    row_scoped_panel = replace(
        plan,
        tiles=(
            _affine_tile(
                ids["i"],
                LoopPlacement(PlacementKind.OUTERMOST),
            ),
            replace(
                panel_tile,
                placement=LoopPlacement(
                    PlacementKind.CHILD_OF,
                    parent=LoopRef(ids["i"], LoopPart.OUTER),
                ),
            ),
        ),
    )
    _assert_plan_rejected(
        cin,
        row_scoped_panel,
        InvalidSchedule,
        "panel_parallel_scope",
    )


def test_heap_result_tile_pairing_and_access_metadata() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_spmm(),
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
    assert verify_scheduled_cin(scheduled) is scheduled
    cin = scheduled.normalized_cin
    plan = scheduled.verified_loop_plan
    result_tile = plan.result_tile
    assert result_tile is not None

    missing = replace(plan, result_tile=None)
    _assert_plan_rejected(
        cin,
        missing,
        InvalidSchedule,
        "heap_result_tile_pairing",
    )

    extra = replace(plan, tiles=())
    _assert_plan_rejected(
        cin,
        extra,
        InvalidSchedule,
        "heap_result_tile_pairing",
    )

    corruptions = (
        replace(result_tile, result_prefix=()),
        replace(
            result_tile, access_indices=tuple(reversed(result_tile.access_indices))
        ),
        replace(result_tile, result_level=0),
    )
    for corrupted in corruptions:
        _assert_plan_rejected(
            cin,
            replace(plan, result_tile=corrupted),
            InvalidSchedule,
            "result_tile_access",
        )

    ids = _index_ids_by_name(cin)
    wrong_parallel_owner = replace(plan, parallel_loop=LoopRef(ids["k"]))
    _assert_plan_rejected(
        cin,
        wrong_parallel_owner,
        InvalidSchedule,
        "heap_parallel_tile",
    )


@pytest.mark.parametrize("accum", ("direct", "heap"))
def test_valid_public_relayout_plans_pass_artifact_verification(accum: str) -> None:
    scheduled = Scheduler.apply_schedule(_build_spmm(), _packed_schedule(accum))

    assert (
        verify_loop_plan(
            scheduled.normalized_cin,
            scheduled.verified_loop_plan,
        )
        is scheduled.verified_loop_plan
    )
    assert verify_scheduled_cin(scheduled) is scheduled


def test_relayout_plan_checks_operand_access_levels_and_scope() -> None:
    scheduled = Scheduler.apply_schedule(_build_spmm(), _packed_schedule("direct"))
    cin = scheduled.normalized_cin
    symbols = _symbol_ids_by_name(cin)
    ids = _index_ids_by_name(cin)
    plan = scheduled.verified_loop_plan
    relayout = plan.relayout
    assert relayout is not None

    wrong_operand = replace(
        plan,
        relayout=replace(relayout, operand_id=symbols["C"]),
    )
    _assert_plan_rejected(
        cin,
        wrong_operand,
        InvalidSchedule,
        "relayout_operand_access",
    )

    wrong_access = replace(
        plan,
        relayout=replace(
            relayout,
            access_indices=tuple(reversed(relayout.access_indices)),
        ),
    )
    _assert_plan_rejected(
        cin,
        wrong_access,
        InvalidSchedule,
        "relayout_access_indices",
    )

    wrong_levels = replace(
        plan,
        relayout=replace(
            relayout,
            operand_panel_level=1,
            operand_pack_level=0,
        ),
    )
    _assert_plan_rejected(
        cin,
        wrong_levels,
        InvalidSchedule,
        "relayout_operand_levels",
    )

    wrong_scope = replace(
        plan,
        relayout=replace(relayout, scope_loop=LoopRef(ids["i"])),
    )
    _assert_plan_rejected(
        cin,
        wrong_scope,
        InvalidSchedule,
        "relayout_scope",
    )


def test_verification_is_pure_repeatable_and_diagnostics_are_nonsemantic() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_spmm(), Schedule(loop_order=("i", "j", "k"))
    )
    cin = scheduled.normalized_cin
    plan = scheduled.verified_loop_plan
    plan_snapshot = replace(plan)
    source_dump = canonical_cin_dump(cin)
    analysis_before = analyze_cin(cin)
    plan_hash = hash(plan)

    assert verify_loop_plan(cin, plan) is plan
    assert verify_loop_plan(cin, plan) is plan
    assert analyze_cin(cin) == analysis_before
    assert canonical_cin_dump(cin) == source_dump
    assert plan == plan_snapshot
    assert hash(plan) == plan_hash
    assert not hasattr(plan, "diagnostics")

    ids = _index_ids_by_name(cin)
    invalid = replace(plan, loop_order=(ids["k"], ids["i"], ids["j"]))
    invalid_snapshot = replace(invalid)
    invalid_hash = hash(invalid)
    with pytest.raises(InvalidSchedule) as first:
        verify_loop_plan(cin, invalid)
    with pytest.raises(InvalidSchedule) as second:
        verify_loop_plan(cin, invalid)

    assert first.value is not second.value
    assert first.value.diagnostics is not second.value.diagnostics
    assert first.value.diagnostics == second.value.diagnostics
    assert invalid == invalid_snapshot
    assert hash(invalid) == invalid_hash
    assert analyze_cin(cin) == analysis_before
    assert canonical_cin_dump(cin) == source_dump
