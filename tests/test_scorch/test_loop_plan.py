import copy
from dataclasses import FrozenInstanceError, replace
from typing import Tuple, cast
from unittest.mock import patch

import pytest

from scorch.compiler.cin import (
    ForAll,
    IndexStmt,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
    Workspace,
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
    MAX_AFFINE_TILE_WIDTH,
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
    WorkspaceInsertion,
    verify_loop_plan,
    verify_scheduled_cin,
)
from scorch.compiler.legacy_cin_adapter import legacy_cin_working_copy
from scorch.compiler.compile_options import CompileOptions
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


def _build_nonadditive_dense_matmul() -> ForAll:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    result = TensorVar("C", fmt="dd")
    left = TensorVar("A", fmt="dd")
    right = TensorVar("B", fmt="dd")
    assignment = TensorAssign(
        result[i, k],
        left[i, j] * right[j, k],
        op=Operation.MUL,
    )
    return ForAll(i, ForAll(j, ForAll(k, assignment)))


def _build_sparse_output_reduction(*, op: Operation | None = None) -> ForAll:
    i, j = IndexVar("i"), IndexVar("j")
    result = TensorVar("C", fmt="s")
    source = TensorVar("A", fmt="dd")
    assignment = TensorAssign(result[i], source[j, i], op=op)
    return ForAll(j, ForAll(i, assignment))


def _build_existing_workspace() -> ForAll:
    i = IndexVar("i")
    result = TensorVar("C", fmt="d")
    source = TensorVar("A", fmt="d")
    workspace = Workspace("tmp", dim=1, dense=True)
    return ForAll(
        i,
        Where(
            producer=TensorAssign(workspace[i], source[i]),
            consumer=TensorAssign(result[i], workspace[i]),
        ),
    )


def _build_multi_assignment_where() -> ForAll:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    first_result = TensorVar("C", fmt="dd")
    second_result = TensorVar("D", fmt="dd")
    first_source = TensorVar("A", fmt="ddd")
    second_source = TensorVar("B", fmt="ddd")
    return ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                Where(
                    producer=TensorAssign(
                        first_result[i, k],
                        first_source[i, j, k],
                        op=Operation.ADD,
                    ),
                    consumer=TensorAssign(
                        second_result[i, k],
                        second_source[i, j, k],
                        op=Operation.ADD,
                    ),
                ),
            ),
        ),
    )


def _build_scalar_assignment() -> TensorAssign:
    result = TensorVar("C", shape=())
    source = TensorVar("A", shape=())
    return TensorAssign(result[[]], source[[]])


def _build_root_reduction() -> ForAll:
    j, k = IndexVar("j"), IndexVar("k")
    result = TensorVar("C", fmt="d")
    source = TensorVar("A", fmt="dd")
    assignment = TensorAssign(result[k], source[j, k], op=Operation.ADD)
    return ForAll(j, ForAll(k, assignment))


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


def test_loop_plan_verifier_rejects_unrepresentable_tile_width():
    scheduled = Scheduler.apply_schedule(
        _build_spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(TileSpec("k", 4, accum="direct"),),
        ),
    )
    plan = scheduled.verified_loop_plan
    oversized = replace(
        plan,
        tiles=(
            replace(
                plan.tiles[0],
                width=MAX_AFFINE_TILE_WIDTH + 1,
            ),
        ),
    )
    with pytest.raises(VerificationError, match="constexpr int target"):
        verify_loop_plan(scheduled.normalized_cin, oversized)


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


def test_loop_plan_verifier_rejects_primitive_subclasses_before_dispatch() -> None:
    """Stored plan primitives are exact values, not executable subclasses."""

    class ExplosiveStr(str):
        def __eq__(self, other):
            raise RuntimeError("hostile string comparison escaped")

        __hash__ = str.__hash__

    class ExplosiveInt(int):
        def __eq__(self, other):
            raise RuntimeError("hostile integer comparison escaped")

        def __lt__(self, other):
            raise RuntimeError("hostile integer comparison escaped")

        def __gt__(self, other):
            raise RuntimeError("hostile integer comparison escaped")

        __hash__ = int.__hash__

    scheduled = Scheduler.apply_schedule(
        _build_spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(TileSpec("k", 4, accum="direct"),),
        ),
    )
    cin = scheduled.normalized_cin
    plan = scheduled.verified_loop_plan
    tile = plan.tiles[0]
    malformed = (
        (
            replace(
                plan,
                tiles=(replace(tile, kind=ExplosiveStr("affine")),),
            ),
            "tile kind",
        ),
        (
            replace(
                plan,
                tiles=(replace(tile, accumulation=ExplosiveStr("direct")),),
            ),
            "tile accumulation",
        ),
        (
            replace(plan, tiles=(replace(tile, width=ExplosiveInt(4)),)),
            "tile widths",
        ),
        (
            replace(
                plan,
                tiles=(
                    replace(
                        tile,
                        loop=LoopRef(IndexId(ExplosiveInt(tile.loop.index_id.value))),
                    ),
                ),
            ),
            "well-formed IndexId",
        ),
        (replace(plan, provenance=ExplosiveStr("explicit")), "provenance"),
        (replace(plan, tag=ExplosiveStr("tag")), "tag"),
    )
    for forged, message in malformed:
        with pytest.raises(VerificationError, match=message):
            verify_loop_plan(cin, forged)


def test_loop_plan_verifier_requires_exact_stored_carrier_fields() -> None:
    """Deleted defaults may not silently erase or reinterpret a schedule."""

    scheduled = Scheduler.apply_schedule(
        _build_spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(TileSpec("k", 4, accum="direct"),),
        ),
    )
    cin = scheduled.normalized_cin
    source = scheduled.verified_loop_plan
    source_tile = source.tiles[0]

    def reject_missing(plan, carrier, field):
        object.__delattr__(carrier, field)
        with pytest.raises(VerificationError, match="invalid stored fields"):
            verify_loop_plan(cin, plan)

    missing_tiles = replace(source)
    reject_missing(missing_tiles, missing_tiles, "tiles")

    missing_provenance = replace(source, provenance="auto")
    reject_missing(missing_provenance, missing_provenance, "provenance")

    missing_loop_order = replace(source)
    reject_missing(missing_loop_order, missing_loop_order, "loop_order")

    tile = replace(source_tile)
    reject_missing(replace(source, tiles=(tile,)), tile, "accumulation")

    loop = replace(source_tile.loop)
    reject_missing(
        replace(source, tiles=(replace(source_tile, loop=loop),)),
        loop,
        "part",
    )

    placement = replace(source_tile.placement)
    reject_missing(
        replace(source, tiles=(replace(source_tile, placement=placement),)),
        placement,
        "parent",
    )

    bound = PanelBound(LoopRef(source.loop_order[0]), SymbolId(100), 0)
    reject_missing(replace(source, panel_bounds=(bound,)), bound, "level")

    relayout = OperandRelayout(
        operand_id=SymbolId(101),
        pack_loop=LoopRef(source.loop_order[0]),
        panel_loop=LoopRef(source.loop_order[1]),
        scope_loop=LoopRef(source.loop_order[0]),
        row_loop=LoopRef(source.loop_order[0]),
        strip_width=4,
        access_indices=source.loop_order,
        operand_panel_level=0,
        operand_pack_level=1,
    )
    reject_missing(replace(source, relayout=relayout), relayout, "access_indices")

    result_tile = ResultTile(
        result_id=SymbolId(102),
        tile_loop=LoopRef(source.loop_order[-1]),
        result_level=1,
        result_prefix=source.loop_order[:1],
        access_indices=source.loop_order,
    )
    reject_missing(
        replace(source, result_tile=result_tile),
        result_tile,
        "result_prefix",
    )

    index_id = IndexId(103)
    index_loop = LoopRef(index_id)
    reject_missing(
        replace(source, tiles=(replace(source_tile, loop=index_loop),)),
        index_id,
        "value",
    )

    symbol_id = SymbolId(104)
    symbol_bound = PanelBound(LoopRef(source.loop_order[0]), symbol_id, 0)
    reject_missing(
        replace(source, panel_bounds=(symbol_bound,)),
        symbol_id,
        "value",
    )

    extra_state = replace(source)
    object.__setattr__(extra_state, "shadow_tiles", source.tiles)
    with pytest.raises(VerificationError, match="invalid stored fields"):
        verify_loop_plan(cin, extra_state)


def test_scheduled_cin_verifier_requires_exact_stored_carrier_fields() -> None:
    scheduled = Scheduler.apply_schedule(_build_spmm(), Schedule())
    missing_plan = ScheduledCIN(
        scheduled.normalized_cin,
        scheduled.verified_loop_plan,
    )
    object.__delattr__(missing_plan, "verified_loop_plan")

    with pytest.raises(VerificationError, match="invalid stored fields"):
        verify_scheduled_cin(missing_plan)


def test_loop_plan_verifier_rejects_forged_enum_members() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(TileSpec("k", 4, accum="direct"),),
        ),
    )
    cin = scheduled.normalized_cin
    source = scheduled.verified_loop_plan
    source_tile = source.tiles[0]

    forged_part = object.__new__(LoopPart)
    object.__setattr__(forged_part, "_name_", "FORGED")
    object.__setattr__(forged_part, "_value_", "forged")
    forged_loop = LoopRef(source_tile.loop.index_id, forged_part)
    with pytest.raises(VerificationError, match="part must be a LoopPart"):
        verify_loop_plan(cin, replace(source, parallel_loop=forged_loop))

    forged_kind = object.__new__(PlacementKind)
    object.__setattr__(forged_kind, "_name_", "FORGED")
    object.__setattr__(forged_kind, "_value_", "forged")
    forged_placement = LoopPlacement(forged_kind)
    with pytest.raises(VerificationError, match="kind must be a PlacementKind"):
        verify_loop_plan(
            cin,
            replace(
                source,
                tiles=(replace(source_tile, placement=forged_placement),),
            ),
        )


def test_loop_plan_diagnostics_handle_unrenderably_large_identities() -> None:
    scheduled = Scheduler.apply_schedule(_build_spmm(), Schedule())
    source = scheduled.verified_loop_plan
    huge = 10**5000

    with pytest.raises(
        VerificationError,
        match="unknown IndexId <integer too large to render>",
    ):
        verify_loop_plan(
            scheduled.normalized_cin,
            replace(source, loop_order=(IndexId(huge),) + source.loop_order[1:]),
        )

    bound = PanelBound(
        LoopRef(source.loop_order[0]),
        SymbolId(huge),
        0,
    )
    panel = LoopTile(
        loop=LoopRef(source.loop_order[0]),
        width=4,
        placement=LoopPlacement(PlacementKind.OUTERMOST),
        parallel=False,
        kind="panel",
        accumulation="direct",
        unroll=False,
    )
    with pytest.raises(
        VerificationError,
        match="unknown SymbolId <integer too large to render>",
    ):
        verify_loop_plan(
            scheduled.normalized_cin,
            replace(source, tiles=(panel,), panel_bounds=(bound,)),
        )


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


def test_parallel_multi_result_writes_fail_closed() -> None:
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
        UnsupportedFeature,
        "multi_assignment_schedule",
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


def test_direct_artifacts_require_one_unambiguous_root_scheduling_scope() -> None:
    shared = IndexVar("i")
    first_result = TensorVar("C", fmt="d")
    first_source = TensorVar("A", fmt="d")
    second_result = TensorVar("D", fmt="d")
    second_source = TensorVar("B", fmt="d")
    ambiguous = normalize_cin(
        Where(
            producer=ForAll(
                shared,
                TensorAssign(first_result[shared], first_source[shared]),
            ),
            consumer=ForAll(
                shared,
                TensorAssign(second_result[shared], second_source[shared]),
            ),
        )
    )
    ambiguous_ids = _index_ids_by_name(ambiguous)
    _assert_plan_rejected(
        ambiguous,
        LoopPlan(
            loop_order=(ambiguous_ids["i"],),
            provenance="explicit",
        ),
        UnsupportedFeature,
        "ambiguous_scheduling_scope",
    )

    i, k = IndexVar("i"), IndexVar("k")
    left_result = TensorVar("Left", fmt="d")
    left_source = TensorVar("LeftSource", fmt="d")
    right_result = TensorVar("Right", fmt="d")
    right_source = TensorVar("RightSource", fmt="d")
    non_prefix = normalize_cin(
        Where(
            producer=ForAll(
                i,
                TensorAssign(left_result[i], left_source[i]),
            ),
            consumer=ForAll(
                k,
                TensorAssign(right_result[k], right_source[k]),
            ),
        )
    )
    non_prefix_ids = _index_ids_by_name(non_prefix)
    _assert_plan_rejected(
        non_prefix,
        LoopPlan(
            loop_order=(non_prefix_ids["i"], non_prefix_ids["k"]),
            provenance="explicit",
        ),
        UnsupportedFeature,
        "non_prefix_scheduling_scope",
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
    assert legacy_cin_working_copy(
        valid_auto.normalized_cin,
        valid_auto.verified_loop_plan,
    ).inserted_workspace


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
        workspace=WorkspaceInsertion(
            reduction_loop=LoopRef(ids["j"]),
            axis_loops=(LoopRef(ids["k"]),),
            dense=True,
        ),
        provenance="auto",
    )

    _assert_plan_rejected(
        cin,
        forged,
        InvalidSchedule,
        "parallel_workspace_scope",
    )


def test_stack_accumulation_rejects_nonadditive_reductions() -> None:
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                4,
                placement="child_of:i",
                accum="stack",
            ),
        ),
    )
    with pytest.raises(UnsupportedFeature, match="stack_reduction_operator"):
        Scheduler.apply_schedule(_build_nonadditive_dense_matmul(), schedule)

    scheduled = Scheduler.apply_schedule(
        _build_nonadditive_dense_matmul(),
        Schedule(loop_order=("i", "j", "k")),
    )
    cin = scheduled.normalized_cin
    ids = _index_ids_by_name(cin)
    forged = replace(
        scheduled.verified_loop_plan,
        tiles=(
            _affine_tile(
                ids["k"],
                LoopPlacement(
                    PlacementKind.CHILD_OF,
                    parent=LoopRef(ids["i"]),
                ),
                accumulation="stack",
            ),
        ),
    )
    _assert_plan_rejected(
        cin,
        forged,
        UnsupportedFeature,
        "stack_reduction_operator",
    )


def test_derived_sparse_workspace_rejects_nonadditive_reductions() -> None:
    explicit = Schedule(loop_order=("j", "i"))
    valid = Scheduler.apply_schedule(_build_sparse_output_reduction(), explicit)
    assert verify_scheduled_cin(valid) is valid
    assert legacy_cin_working_copy(
        valid.normalized_cin,
        valid.verified_loop_plan,
    ).inserted_workspace
    valid_auto = Scheduler.apply_schedule(_build_sparse_output_reduction(), Schedule())
    assert verify_scheduled_cin(valid_auto) is valid_auto
    assert legacy_cin_working_copy(
        valid_auto.normalized_cin,
        valid_auto.verified_loop_plan,
    ).inserted_workspace

    with pytest.raises(UnsupportedFeature, match="workspace_reduction_operator"):
        Scheduler.apply_schedule(
            _build_sparse_output_reduction(op=Operation.MUL),
            explicit,
        )
    with pytest.raises(UnsupportedFeature, match="auto_reduction_operator"):
        Scheduler.apply_schedule(
            _build_sparse_output_reduction(op=Operation.MUL),
            Schedule(),
        )

    cin = normalize_cin(_build_sparse_output_reduction(op=Operation.MUL))
    ids = _index_ids_by_name(cin)
    forged_explicit = LoopPlan(
        loop_order=(ids["j"], ids["i"]),
        provenance="explicit",
    )
    _assert_plan_rejected(
        cin,
        forged_explicit,
        UnsupportedFeature,
        "workspace_reduction_operator",
    )
    forged_auto = replace(forged_explicit, provenance="auto")
    _assert_plan_rejected(
        cin,
        forged_auto,
        UnsupportedFeature,
        "auto_reduction_operator",
    )


def test_existing_workspace_rejects_unsupported_replay_decisions() -> None:
    auto = Scheduler.apply_schedule(_build_existing_workspace(), Schedule())
    assert verify_scheduled_cin(auto) is auto

    explicit_requests = (
        Schedule(loop_order=("i",)),
        Schedule(
            loop_order=("i",),
            tiles=(TileSpec("i", 4, accum="direct"),),
        ),
        Schedule(loop_order=("i",), parallel_loop="i"),
    )
    for schedule in explicit_requests:
        with pytest.raises(UnsupportedFeature, match="existing workspace"):
            Scheduler.apply_schedule(_build_existing_workspace(), schedule)

    cin = auto.normalized_cin
    ids = _index_ids_by_name(cin)
    for provenance in ("explicit", "tuned", "fallback", "direct"):
        forged = replace(auto.verified_loop_plan, provenance=provenance)
        _assert_plan_rejected(
            cin,
            forged,
            UnsupportedFeature,
            "workspace_plan_provenance",
        )

    forged_tile = replace(
        auto.verified_loop_plan,
        tiles=(
            _affine_tile(
                ids["i"],
                LoopPlacement(PlacementKind.OUTERMOST),
            ),
        ),
    )
    _assert_plan_rejected(
        cin,
        forged_tile,
        UnsupportedFeature,
        "multi_assignment_schedule",
    )

    forged_parallel = replace(
        auto.verified_loop_plan,
        parallel_loop=LoopRef(ids["i"]),
    )
    _assert_plan_rejected(
        cin,
        forged_parallel,
        UnsupportedFeature,
        "multi_assignment_schedule",
    )


def test_multi_assignment_scope_rejects_tile_and_parallel_decisions() -> None:
    with pytest.raises(UnsupportedFeature, match="Derived workspace"):
        Scheduler.apply_schedule(_build_multi_assignment_where(), Schedule())

    explicit_requests = (
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(TileSpec("k", 4, accum="direct"),),
        ),
        Schedule(loop_order=("i", "j", "k"), parallel_loop="i"),
    )
    for schedule in explicit_requests:
        with pytest.raises(UnsupportedFeature):
            Scheduler.apply_schedule(_build_multi_assignment_where(), schedule)

    cin = normalize_cin(_build_multi_assignment_where())
    ids = _index_ids_by_name(cin)
    base = LoopPlan(
        loop_order=(ids["i"], ids["j"], ids["k"]),
        provenance="auto",
    )
    _assert_plan_rejected(
        cin,
        base,
        UnsupportedFeature,
        "multi_assignment_schedule",
    )
    _assert_plan_rejected(
        cin,
        replace(base, provenance="explicit"),
        UnsupportedFeature,
        "multi_assignment_schedule",
    )
    forged_tile = replace(
        base,
        tiles=(
            _affine_tile(
                ids["k"],
                LoopPlacement(PlacementKind.OUTERMOST),
            ),
        ),
    )
    _assert_plan_rejected(
        cin,
        forged_tile,
        UnsupportedFeature,
        "multi_assignment_schedule",
    )
    forged_parallel = replace(
        base,
        parallel_loop=LoopRef(ids["i"]),
        provenance="explicit",
    )
    _assert_plan_rejected(
        cin,
        forged_parallel,
        UnsupportedFeature,
        "multi_assignment_schedule",
    )


def test_loop_free_cin_supports_only_auto_plan_provenance() -> None:
    auto = Scheduler.apply_schedule(_build_scalar_assignment(), Schedule())
    assert verify_scheduled_cin(auto) is auto

    with pytest.raises(UnsupportedFeature):
        Scheduler.apply_schedule(
            _build_scalar_assignment(),
            Schedule(loop_order=()),
        )

    for provenance in ("explicit", "tuned", "fallback", "direct"):
        forged = replace(auto.verified_loop_plan, provenance=provenance)
        _assert_plan_rejected(
            auto.normalized_cin,
            forged,
            UnsupportedFeature,
            "scalar_plan_provenance",
        )


def test_root_workspace_scope_rejects_tiling_replay() -> None:
    auto = Scheduler.apply_schedule(_build_root_reduction(), Schedule())
    assert verify_scheduled_cin(auto) is auto
    assert not legacy_cin_working_copy(
        auto.normalized_cin,
        auto.verified_loop_plan,
    ).inserted_workspace

    with pytest.raises(UnsupportedFeature, match="workspace inserted at the root"):
        Scheduler.apply_schedule(
            _build_root_reduction(),
            Schedule(
                loop_order=("j", "k"),
                tiles=(TileSpec("k", 4, accum="stack"),),
            ),
        )

    cin = normalize_cin(_build_root_reduction())
    ids = _index_ids_by_name(cin)
    base = LoopPlan(
        loop_order=(ids["j"], ids["k"]),
        tiles=(
            _affine_tile(
                ids["k"],
                LoopPlacement(PlacementKind.OUTERMOST),
            ),
        ),
        workspace=WorkspaceInsertion(
            reduction_loop=LoopRef(ids["j"]),
            axis_loops=(LoopRef(ids["k"]),),
            dense=True,
        ),
        provenance="auto",
    )
    _assert_plan_rejected(
        cin,
        base,
        UnsupportedFeature,
        "root_workspace_tiling",
    )
    _assert_plan_rejected(
        cin,
        replace(
            base,
            tiles=(replace(base.tiles[0], accumulation="stack"),),
            workspace=None,
            provenance="explicit",
        ),
        UnsupportedFeature,
        "root_workspace_tiling",
    )

    direct = replace(base, tiles=(), workspace=None)
    assert verify_loop_plan(cin, direct) is direct
    assert not legacy_cin_working_copy(cin, direct).inserted_workspace


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
    panel_tile = plan.tiles[0]

    not_unrolled = Scheduler.apply_schedule(
        _build_spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(
                TileSpec(
                    "j",
                    4,
                    kind="panel",
                    accum="direct",
                    unroll=False,
                ),
            ),
            parallel_loop="i",
        ),
    )
    assert verify_scheduled_cin(not_unrolled) is not_unrolled
    assert not_unrolled.verified_loop_plan.tiles[0].unroll is False
    not_unrolled_plan = replace(
        plan,
        tiles=(replace(panel_tile, unroll=False),),
    )
    assert verify_loop_plan(cin, not_unrolled_plan) is not_unrolled_plan

    duplicated_bound = replace(plan, panel_bounds=(bound, bound))
    _assert_plan_rejected(
        cin,
        duplicated_bound,
        VerificationError,
        "unique loops",
    )

    multiple_panels = replace(
        plan,
        tiles=(panel_tile, replace(panel_tile, loop=LoopRef(ids["k"]))),
        panel_bounds=(bound, replace(bound, loop=LoopRef(ids["k"]))),
    )
    _assert_plan_rejected(
        cin,
        multiple_panels,
        UnsupportedFeature,
        "multiple_panel_tiles",
    )

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
    heap_tile = plan.tiles[0]
    invalid_lifetime = replace(
        plan,
        tiles=(
            replace(
                heap_tile,
                placement=LoopPlacement(
                    PlacementKind.CHILD_OF,
                    parent=LoopRef(ids["i"]),
                ),
            ),
        ),
    )
    _assert_plan_rejected(
        cin,
        invalid_lifetime,
        InvalidSchedule,
        "heap_tile_lifetime",
    )

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

    wrong_pack = replace(
        plan,
        relayout=replace(relayout, pack_loop=LoopRef(ids["i"])),
    )
    _assert_plan_rejected(
        cin,
        wrong_pack,
        InvalidSchedule,
        "relayout_storage_axes",
    )

    wrong_row = replace(
        plan,
        relayout=replace(relayout, row_loop=LoopRef(ids["k"])),
    )
    _assert_plan_rejected(
        cin,
        wrong_row,
        InvalidSchedule,
        "relayout_sparse_axes",
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


# ---------------------------------------------------------------------------
# The explicit automatic workspace-insertion fact
# ---------------------------------------------------------------------------


def _auto_plan(cin: ForAll) -> Tuple[ScheduledCIN, LoopPlan]:
    scheduled = Scheduler.apply_schedule(cin, Schedule())
    return scheduled, scheduled.verified_loop_plan


def test_auto_plan_records_the_workspace_insertion_decision() -> None:
    """The F2 identity path records the standalone workspace fact exactly."""

    scheduled, plan = _auto_plan(_build_dense_matmul())
    ids = _index_ids_by_name(scheduled.normalized_cin)
    workspace = plan.workspace
    assert workspace is not None
    assert workspace.dense is True
    assert workspace.reduction_loop == LoopRef(ids["j"])
    assert workspace.axis_loops == (LoopRef(ids["k"]),)
    assert len(plan.tiles) == 2

    _, spmm_plan = _auto_plan(_build_spmm())
    assert spmm_plan.workspace is None
    assert spmm_plan.tiles == ()

    sparse_scheduled, sparse_plan = _auto_plan(_build_sparse_output_reduction())
    sparse_ids = _index_ids_by_name(sparse_scheduled.normalized_cin)
    sparse_workspace = sparse_plan.workspace
    assert sparse_workspace is not None
    assert sparse_workspace.dense is False
    assert sparse_workspace.reduction_loop == LoopRef(sparse_ids["j"])
    assert sparse_workspace.axis_loops == (LoopRef(sparse_ids["i"]),)
    assert sparse_plan.tiles == ()


def test_auto_workspace_replay_consumes_the_recorded_fact() -> None:
    """Replay materializes the recorded workspace without re-deriving policy."""

    scheduled, plan = _auto_plan(_build_dense_matmul())
    with patch.object(
        Scheduler,
        "should_insert_workspace",
        side_effect=AssertionError("replay must not re-derive the workspace decision"),
    ):
        replayed = legacy_cin_working_copy(scheduled.normalized_cin, plan)
    assert replayed.inserted_workspace

    direct = Scheduler._auto_schedule_owned(
        copy.deepcopy(scheduled.normalized_cin),
        CompileOptions.from_environment(),
    )
    assert str(replayed) == str(direct)


def test_root_workspace_auto_plan_records_the_replay_contract() -> None:
    """A dense root-scope insertion whose tiles never materialize records None.

    The private surgery inserts a pure-overhead root workspace (the Where
    root makes the tiling heuristics bail, so no tiles are recorded); the
    replayable plan contract deliberately omits it, exactly as the
    established ScheduledCIN replay behavior does.  The surgery-versus-
    replay divergence is a documented legacy observation, not new state.
    """

    scheduled, plan = _auto_plan(_build_root_reduction())
    assert plan.workspace is None
    assert plan.tiles == ()
    replayed = legacy_cin_working_copy(scheduled.normalized_cin, plan)
    assert not replayed.inserted_workspace

    surgery = Scheduler._auto_schedule_owned(
        copy.deepcopy(scheduled.normalized_cin),
        CompileOptions.from_environment(),
    )
    assert surgery.inserted_workspace
    assert str(replayed) != str(surgery)


def test_plan_free_root_workspace_origin_fails_closed() -> None:
    """F4 must not return a plan that silently drops a real workspace decision."""

    with pytest.raises(
        UnsupportedFeature,
        match="dense root-workspace materialization is not represented",
    ):
        Scheduler.auto_schedule_plan(_build_root_reduction())


def test_release_auto_schedule_does_not_depend_on_plan_recording() -> None:
    """The ordinary release mutation path must not call recording-only helpers."""

    expected = Scheduler.auto_schedule(_build_dense_matmul())
    with patch.object(
        Scheduler,
        "_workspace_insertion_record",
        side_effect=AssertionError("release scheduling must not record a plan"),
    ):
        actual = Scheduler.auto_schedule(_build_dense_matmul())
    assert str(actual) == str(expected)


def test_auto_schedule_plan_originates_the_production_decisions() -> None:
    """The F4 origination replays identically to the plan-free scheduler."""

    for build in (_build_dense_matmul, _build_spmm):
        for arm in (None, True, False):
            scheduled = Scheduler.auto_schedule_plan(
                build(),
                regblock_enabled=arm,
            )
            plan = scheduled.verified_loop_plan
            assert plan.provenance == "auto"
            replayed = legacy_cin_working_copy(scheduled.normalized_cin, plan)
            if arm is None:
                direct = Scheduler.auto_schedule(build())
            else:
                direct = Scheduler._auto_schedule_regblock_arm(
                    build(),
                    enabled=arm,
                    compile_options=CompileOptions.from_environment(),
                )
            assert str(replayed) == str(direct), (build.__name__, arm)

    regblock = Scheduler.auto_schedule_plan(
        _build_spmm(),
        regblock_enabled=True,
    )
    regblock_plan = regblock.verified_loop_plan
    assert regblock_plan.workspace is not None
    assert len(regblock_plan.tiles) == 1
    assert regblock_plan.tiles[0].placement.kind is PlacementKind.CHILD_OF


def test_auto_schedule_plan_scalar_cin_records_the_empty_plan() -> None:
    loop_free = Scheduler.auto_schedule_plan(_build_scalar_assignment())
    assert loop_free.verified_loop_plan.provenance == "auto"
    assert loop_free.verified_loop_plan.loop_order == ()


def test_complete_auto_plan_helper_requires_exact_owned_sinks() -> None:
    options = CompileOptions.from_environment()
    with pytest.raises(TypeError, match="require_complete_plan must be an exact bool"):
        Scheduler._apply_auto_order_owned(
            _build_scalar_assignment(),
            [],
            options,
            require_complete_plan=1,
        )

    normalized = normalize_cin(_build_dense_matmul())
    with pytest.raises(TypeError, match="needs empty exact plan"):
        Scheduler._apply_auto_order_owned(
            normalized,
            [],
            options,
            require_complete_plan=True,
        )
    with pytest.raises(TypeError, match="needs empty exact plan"):
        Scheduler._apply_auto_order_owned(
            normalized,
            [],
            options,
            plan_tiles=[
                LoopTile(
                    loop=LoopRef(normalized.index_var.index_id),
                    width=4,
                    placement=LoopPlacement(PlacementKind.OUTERMOST),
                    parallel=False,
                    kind="affine",
                    accumulation="direct",
                    unroll=True,
                )
            ],
            plan_workspace=[],
            require_complete_plan=True,
        )
    shared_sinks = []
    with pytest.raises(TypeError, match="needs empty exact plan"):
        Scheduler._apply_auto_order_owned(
            normalized,
            [],
            options,
            plan_tiles=shared_sinks,
            plan_workspace=shared_sinks,
            require_complete_plan=True,
        )


def test_auto_workspace_fact_must_equal_the_derived_decision() -> None:
    """Forged, dropped, or mutated workspace facts are rejected exactly."""

    scheduled, plan = _auto_plan(_build_dense_matmul())
    cin = scheduled.normalized_cin
    ids = _index_ids_by_name(cin)
    assert plan.workspace is not None

    _assert_plan_rejected(
        cin,
        replace(plan, workspace=None),
        InvalidSchedule,
        "auto_workspace_decision",
    )
    _assert_plan_rejected(
        cin,
        replace(
            plan,
            workspace=replace(plan.workspace, reduction_loop=LoopRef(ids["i"])),
        ),
        InvalidSchedule,
        "auto_workspace_decision",
    )
    _assert_plan_rejected(
        cin,
        replace(
            plan,
            workspace=replace(plan.workspace, axis_loops=(LoopRef(ids["i"]),)),
        ),
        InvalidSchedule,
        "auto_workspace_decision",
    )
    _assert_plan_rejected(
        cin,
        replace(plan, workspace=replace(plan.workspace, dense=False)),
        InvalidSchedule,
        "auto_workspace_decision",
    )

    spmm_scheduled, spmm_plan = _auto_plan(_build_spmm())
    spmm_ids = _index_ids_by_name(spmm_scheduled.normalized_cin)
    assert spmm_plan.workspace is None
    _assert_plan_rejected(
        spmm_scheduled.normalized_cin,
        replace(
            spmm_plan,
            workspace=WorkspaceInsertion(
                reduction_loop=LoopRef(spmm_ids["j"]),
                axis_loops=(LoopRef(spmm_ids["k"]),),
                dense=True,
            ),
        ),
        InvalidSchedule,
        "auto_workspace_decision",
    )


def test_workspace_fact_is_an_automatic_decision_only() -> None:
    scheduled = Scheduler.apply_schedule(
        _build_dense_matmul(), Schedule(loop_order=("i", "j", "k"))
    )
    cin = scheduled.normalized_cin
    ids = _index_ids_by_name(cin)
    _assert_plan_rejected(
        cin,
        replace(
            scheduled.verified_loop_plan,
            workspace=WorkspaceInsertion(
                reduction_loop=LoopRef(ids["j"]),
                axis_loops=(LoopRef(ids["k"]),),
                dense=True,
            ),
        ),
        InvalidSchedule,
        "workspace_provenance",
    )


def test_workspace_fact_structure_is_exact() -> None:
    scheduled, plan = _auto_plan(_build_dense_matmul())
    cin = scheduled.normalized_cin
    assert plan.workspace is not None

    with pytest.raises(VerificationError, match="exactly one axis"):
        verify_loop_plan(
            cin,
            replace(
                plan,
                workspace=replace(
                    plan.workspace,
                    axis_loops=plan.workspace.axis_loops * 2,
                ),
            ),
        )
    with pytest.raises(VerificationError, match="non-empty tuple"):
        verify_loop_plan(
            cin,
            replace(plan, workspace=replace(plan.workspace, axis_loops=())),
        )
    with pytest.raises(VerificationError, match="must be a bool"):
        verify_loop_plan(
            cin,
            replace(
                plan,
                workspace=replace(plan.workspace, dense=cast(bool, 1)),
            ),
        )
    with pytest.raises(VerificationError, match="WorkspaceInsertion or None"):
        verify_loop_plan(
            cin,
            replace(plan, workspace=cast(WorkspaceInsertion, object())),
        )
    forged = replace(plan.workspace)
    object.__setattr__(forged, "ghost", True)
    with pytest.raises(VerificationError, match="invalid stored fields"):
        verify_loop_plan(cin, replace(plan, workspace=forged))
    deleted = replace(plan.workspace)
    object.__delattr__(deleted, "dense")
    with pytest.raises(VerificationError, match="invalid stored fields"):
        verify_loop_plan(cin, replace(plan, workspace=deleted))
