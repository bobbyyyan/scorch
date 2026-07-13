from dataclasses import FrozenInstanceError, replace

import pytest

from scorch.compiler.cin import ForAll, IndexVar, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.diagnostics import VerificationError
from scorch.compiler.identity import IndexId, SymbolId
from scorch.compiler.loop_plan import (
    LoopRef,
    ResultTile,
    ScheduledCIN,
    verify_scheduled_cin,
)
from scorch.compiler.scheduler import Schedule, Scheduler, TileSpec


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
    assert "k_out" not in str(reordered.normalized_cin)
    assert "k_out" in str(tiled.normalized_cin)
    assert "k_out" not in str(cin)
