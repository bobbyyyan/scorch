from dataclasses import FrozenInstanceError, replace
from typing import Tuple, cast
from unittest.mock import patch

import pytest

from scorch.compiler.cin import ForAll, IndexVar, TensorVar
from scorch.compiler.cin_analysis import canonical_cin_dump
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.diagnostics import VerificationError
from scorch.compiler.identity import IndexId, SymbolId
from scorch.compiler.loop_plan import (
    LoopPlan,
    LoopRef,
    LoopTile,
    OperandRelayout,
    PanelBound,
    ResultTile,
    ScheduledCIN,
    verify_scheduled_cin,
)
from scorch.compiler.legacy_cin_adapter import legacy_cin_working_copy
from scorch.compiler.scheduler import (
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
