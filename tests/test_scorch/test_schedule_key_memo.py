"""A schedule's cache key is validated and formatted once per distinct state.

``Schedule.cache_key`` revalidated the schedule's stored state on every read by
rebuilding a canonical Schedule from it, and a warm ``einsum`` call with a schedule
read the key twice: about 174,000 instructions per call on sapling (jobs 85220 and
85225), 46% of the call.  The key of each validated state is now memoized under a
typed key of that state (``scheduler._schedule_memo_key``), which every read rebuilds
with the reads and refusals the full validation starts with, and the call reads the
key once (``ops._DeferredOptions.schedule_key``).  These tests pin:

* a second read returns the memoized string and builds no Schedule, and two equal
  schedules built apart share one entry;
* a state changed with ``object.__setattr__`` after a read -- in the Schedule or in a
  TileSpec or RelayoutSpec it holds -- is revalidated on the next read: refused when
  invalid, keyed afresh when valid;
* values that compare equal across types (``True`` and ``1``, ``False`` and ``0``) are
  different states, each judged as the full validation judges it;
* a state the memo cannot key is refused on every read, and no method of a stored
  value runs;
* the module state the validation reads (the assembly and accumulator registries and
  the tile width bound) is part of the state;
* the memo stays within its bound;
* a warm call reads the key once, and a schedule changed between the two points that
  used to read it gives the parse memo and the dispatch key one state's key, and
  keeps no kernel under a state the schedule left during the call.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest
import torch

import scorch
import scorch.ops as ops  # type: ignore[import-untyped]
from scorch import STensor  # type: ignore[import-untyped]
from scorch.compiler import (  # type: ignore[import-untyped]
    compile_options,
    scheduler as scheduler_module,
    sparse_accumulator,
    sparse_assembly,
)
from scorch.compiler.diagnostics import InvalidSchedule  # type: ignore[import-untyped]
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    RelayoutSpec,
    Schedule,
    TileSpec,
)


@pytest.fixture(autouse=True)
def _empty_memo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(scheduler_module, "_SCHEDULE_KEY_MEMO", {})
    compile_options.clear_build_snapshot_cache()
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
    yield
    compile_options.clear_build_snapshot_cache()


@pytest.fixture
def validations(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Every schedule the full validation ran on, in order."""

    seen: list[object] = []
    full = scheduler_module._canonical_schedule

    def counting(schedule: object) -> Schedule:
        seen.append(schedule)
        return full(schedule)

    monkeypatch.setattr(scheduler_module, "_canonical_schedule", counting)
    return seen


def _fresh_key(schedule: Schedule) -> str:
    """The key the full validation and format give, bypassing the memo."""

    canonical = scheduler_module._canonical_schedule(schedule)
    return scheduler_module._format_schedule_key(canonical)


class _HostileStr(str):
    def __hash__(self):
        raise AssertionError("caller hash executed")

    def __eq__(self, other):
        raise AssertionError("caller eq executed")

    def __repr__(self):
        raise AssertionError("caller repr executed")


def test_a_second_read_returns_the_memoized_key_and_builds_no_schedule(
    validations: list[object],
) -> None:
    schedule = Schedule(loop_order=("i", "j"), tiles=(TileSpec("j", 4),))
    first = schedule.cache_key
    assert len(validations) == 1
    second = schedule.cache_key
    assert second is first
    assert len(validations) == 1
    assert first == _fresh_key(schedule)


def test_equal_schedules_built_apart_share_one_entry(
    validations: list[object],
) -> None:
    built = Schedule(loop_order=("i", "j"), vector_loops=("j",))
    from_lists = Schedule(loop_order=["i", "j"], vector_loops=["j"])
    assert built.cache_key == from_lists.cache_key
    assert len(validations) == 1


def test_a_changed_schedule_field_is_revalidated_on_the_next_read(
    validations: list[object],
) -> None:
    schedule = Schedule(tag="a")
    before = schedule.cache_key
    object.__setattr__(schedule, "tag", "b")
    after = schedule.cache_key
    assert after != before
    assert after == Schedule(tag="b").cache_key
    assert len(validations) == 2

    tiled = Schedule(tiles=(TileSpec("k", 4),))
    _ = tiled.cache_key
    object.__setattr__(tiled, "tiles", [TileSpec("k", 4)])
    with pytest.raises(InvalidSchedule, match="owned tuple"):
        _ = tiled.cache_key
    with pytest.raises(InvalidSchedule, match="owned tuple"):
        _ = tiled.cache_key

    tagged = Schedule(tag="a")
    _ = tagged.cache_key
    object.__setattr__(tagged, "tag", _HostileStr("a"))
    with pytest.raises(InvalidSchedule, match="exact str"):
        _ = tagged.cache_key


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("width", 8, None),
        ("width", 0, "invalid stored values"),
        ("kind", "panel", None),
        ("accum", "heap", None),
        ("accum", "nowhere", "invalid stored values"),
        ("index_var", _HostileStr("k"), "invalid stored values"),
    ],
)
def test_a_changed_tile_is_revalidated_on_the_next_read(
    field: str, value: object, match: str | None
) -> None:
    tile = TileSpec("k", 4)
    schedule = Schedule(tiles=(tile,))
    before = schedule.cache_key
    object.__setattr__(tile, field, value)
    if match is None:
        after = schedule.cache_key
        assert after != before
        rebuilt = TileSpec(**{**TileSpec("k", 4).__dict__, field: value})
        assert after == Schedule(tiles=(rebuilt,)).cache_key
    else:
        for _ in range(2):
            with pytest.raises(InvalidSchedule, match=match):
                _ = schedule.cache_key


def test_a_tile_with_a_field_added_or_removed_is_refused_after_a_read() -> None:
    added = TileSpec("k", 4)
    schedule = Schedule(tiles=(added,))
    _ = schedule.cache_key
    object.__setattr__(added, "extra", 1)
    with pytest.raises(InvalidSchedule, match="stored fields"):
        _ = schedule.cache_key

    removed = TileSpec("k", 4)
    other = Schedule(tiles=(removed,))
    _ = other.cache_key
    object.__delattr__(removed, "unroll")
    with pytest.raises(InvalidSchedule, match="stored fields"):
        _ = other.cache_key


def test_a_changed_relayout_is_revalidated_on_the_next_read() -> None:
    relayout = RelayoutSpec("B", "k", 4, scope_var="j")
    schedule = Schedule(relayout=relayout)
    before = schedule.cache_key
    object.__setattr__(relayout, "strip_width", 8)
    after = schedule.cache_key
    assert after != before
    widened = Schedule(relayout=RelayoutSpec("B", "k", 8, scope_var="j"))
    assert after == widened.cache_key
    object.__setattr__(relayout, "strip_width", 0)
    with pytest.raises(InvalidSchedule, match="invalid stored values"):
        _ = schedule.cache_key


@pytest.mark.parametrize(
    "field,valid,equal_value",
    [("width", 1, True), ("parallel", False, 0), ("unroll", True, 1)],
)
def test_values_equal_across_types_are_different_states(
    field: str, valid: object, equal_value: object
) -> None:
    # True == 1 and hash(True) == hash(1): a key without the exact type would hand
    # the second state the first one's verdict.
    assert valid == equal_value and hash(valid) == hash(equal_value)
    tile = TileSpec(**{"index_var": "k", "width": 4, field: valid})
    schedule = Schedule(tiles=(tile,))
    _ = schedule.cache_key
    object.__setattr__(tile, field, equal_value)
    with pytest.raises(InvalidSchedule, match="invalid stored values"):
        _ = schedule.cache_key


def test_an_unkeyable_state_is_refused_on_every_read_and_runs_nothing() -> None:
    schedule = Schedule(loop_order=("i", "j"))
    _ = schedule.cache_key
    object.__setattr__(schedule, "loop_order", (_HostileStr("i"), _HostileStr("j")))
    entries = len(scheduler_module._SCHEDULE_KEY_MEMO)
    for _ in range(2):
        with pytest.raises(InvalidSchedule):
            _ = schedule.cache_key
    assert len(scheduler_module._SCHEDULE_KEY_MEMO) == entries


def test_a_shadowing_class_attribute_is_refused_after_a_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule()
    _ = schedule.cache_key
    monkeypatch.setattr(Schedule, "tag", property(lambda self: "descriptor value"))
    with pytest.raises(InvalidSchedule, match="diverges"):
        _ = schedule.cache_key


def test_the_module_state_the_validation_reads_is_part_of_the_state(
    monkeypatch: pytest.MonkeyPatch, validations: list[object]
) -> None:
    # Each value the validation reads besides the schedule (the assembly and
    # accumulator registries, the tile width bound) enters the memo key, so a read
    # after one changes runs the full validation again.  A value that makes the state
    # invalid is refused by the check that reads it: the registries by
    # ``_canonical_schedule``'s registry checks, which run before the Schedule it
    # rebuilds would read them again, and the tile width bound by the TileSpec it
    # rebuilds.  A value that keeps the state valid gives the same key.  Restoring the
    # value finds the memoized key again, which the full validation also gives.
    cases = (
        (
            Schedule(assembly="two_pass_serial"),
            sparse_assembly,
            "SPARSE_ASSEMBLY_STRATEGIES",
            ("single_pass_serial",),
            sparse_assembly.SPARSE_ASSEMBLY_STRATEGIES + ("unused_strategy",),
            "Schedule.assembly must name a sparse assembly strategy",
        ),
        (
            Schedule(accumulator="linked_list"),
            sparse_accumulator,
            "SPARSE_ACCUMULATOR_STRUCTURES",
            ("coordinate_list",),
            sparse_accumulator.SPARSE_ACCUMULATOR_STRUCTURES + ("unused_structure",),
            "Schedule.accumulator must name a sparse accumulation structure",
        ),
        (
            Schedule(tiles=(TileSpec("k", 64),)),
            scheduler_module,
            "MAX_AFFINE_TILE_WIDTH",
            32,
            128,
            r"Schedule\.tiles\[0\] has invalid stored values",
        ),
    )
    for schedule, module, name, refusing, keeping, message in cases:
        key = schedule.cache_key
        with monkeypatch.context() as patch:
            patch.setattr(module, name, refusing)
            validated = len(validations)
            with pytest.raises(InvalidSchedule, match=message) as refused:
                _ = schedule.cache_key
            assert refused.value.code == "schedule_state_invalid"
            assert len(validations) == validated + 1

            patch.setattr(module, name, keeping)
            validated = len(validations)
            assert schedule.cache_key == key
            assert len(validations) == validated + 1

        validated = len(validations)
        assert schedule.cache_key == key
        assert len(validations) == validated
        assert key == _fresh_key(schedule)


def test_the_memo_stays_within_its_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler_module, "_SCHEDULE_KEY_MEMO_LIMIT", 3)
    for number in range(10):
        schedule = Schedule(tag=f"t{number}")
        assert schedule.cache_key == _fresh_key(schedule)
        assert len(scheduler_module._SCHEDULE_KEY_MEMO) <= 3


def _operands(rows: int) -> tuple[STensor, STensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(rows)
    dense = torch.rand(rows, rows, generator=generator, dtype=torch.float64)
    dense[dense < 0.7] = 0.0
    stored = dense.nonzero()
    matrix = STensor.from_components(
        (rows, rows),
        "oo",
        [[stored[:, 0].to(torch.int32)], [stored[:, 1].to(torch.int32)]],
        dense[stored[:, 0], stored[:, 1]],
        name="A",
    )
    x = torch.rand(rows, generator=generator, dtype=torch.float64)
    return matrix, STensor.from_torch(x, "x"), dense @ x


def _call(matrix: STensor, vector: STensor, **kwargs: Any) -> Any:
    return scorch.einsum("ij,j->i", matrix, vector, format="d", **kwargs)


def _check(result: Any, expected: torch.Tensor) -> None:
    got = result.to_torch() if isinstance(result, STensor) else result
    assert bool(torch.isfinite(got).all())
    assert torch.allclose(got, expected, rtol=1e-12, atol=1e-12)


def _counting_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    reads: list[str] = []
    read = Schedule.__dict__["cache_key"].fget

    def counting(schedule: Schedule) -> str:
        key = read(schedule)
        reads.append(key)
        return key

    monkeypatch.setattr(Schedule, "cache_key", property(counting))
    return reads


def test_a_warm_call_reads_the_schedule_key_once(
    monkeypatch: pytest.MonkeyPatch, validations: list[object]
) -> None:
    matrix, vector, expected = _operands(12)
    schedule = Schedule(loop_order=("i", "j"))
    _check(_call(matrix, vector, schedule=schedule), expected)
    reads = _counting_reads(monkeypatch)
    validated = len(validations)
    for _ in range(3):
        _check(_call(matrix, vector, schedule=schedule), expected)
    assert len(reads) == 3
    assert len(validations) == validated


def _parsed_fingerprint(schedule_key: str) -> object:
    """The fingerprint the parse memo holds for ``schedule_key`` (one environment)."""

    found = [
        fingerprint
        for (_, key), fingerprint in compile_options._parsed_options_keys.items()
        if key == schedule_key
    ]
    assert len(found) == 1
    return found[0]


def _recording_dispatch_keys(
    monkeypatch: pytest.MonkeyPatch, after_key: Callable[[], None] | None = None
) -> list[tuple]:
    keys: list[tuple] = []
    make_key = ops._einsum_cache_key

    def recording(*args: Any, **kwargs: Any) -> tuple:
        key = make_key(*args, **kwargs)
        keys.append(key)
        if after_key is not None:
            after_key()
        return key

    monkeypatch.setattr(ops, "_einsum_cache_key", recording)
    return keys


@pytest.mark.parametrize("warm", [False, True])
def test_a_schedule_changed_after_the_calls_read_gives_one_states_keys(
    monkeypatch: pytest.MonkeyPatch, warm: bool
) -> None:
    # The call reads the key once, then the schedule changes before the point where
    # the dispatch key read it again until step 31.  The dispatch key's schedule part
    # and options part must then describe one state: the state read when the parse
    # memo held it (warm), the state the options were parsed from when it did not.
    # The change is made at the call's first read of ``Schedule.cache_key``, wherever
    # the call makes it (through ``ops._schedule_cache_key`` or, as the warm path
    # does, from the property directly).
    matrix, vector, expected = _operands(12)
    schedule = Schedule(loop_order=("i", "j"), tag="before")
    before = Schedule(loop_order=("i", "j"), tag="before").cache_key
    after = Schedule(loop_order=("i", "j"), tag="after").cache_key
    if warm:
        _check(_call(matrix, vector, schedule=schedule), expected)

    with monkeypatch.context() as patch:
        first_reads: list[object] = []
        read = Schedule.__dict__["cache_key"].fget

        def changing(value: Schedule) -> str:
            key = read(value)
            if value is schedule and not first_reads:
                first_reads.append(key)
                object.__setattr__(schedule, "tag", "after")
            return key

        patch.setattr(Schedule, "cache_key", property(changing))
        dispatch_keys = _recording_dispatch_keys(patch)
        _check(_call(matrix, vector, schedule=schedule), expected)

    assert first_reads == [before]
    assert len(dispatch_keys) == 1
    state = before if warm else after
    assert dispatch_keys[0][5] == state
    assert dispatch_keys[0][6] == _parsed_fingerprint(state)
    if not warm:
        assert {key[5] for key in ops._einsum_dispatch_cache} == {after}


def test_a_schedule_changed_after_the_dispatch_key_keeps_no_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix, vector, expected = _operands(12)
    schedule = Schedule(loop_order=("i", "j"), tag="before")
    after = Schedule(loop_order=("i", "j"), tag="after").cache_key

    def change() -> None:
        object.__setattr__(schedule, "tag", "after")

    with monkeypatch.context() as patch:
        _recording_dispatch_keys(patch, change)
        _check(_call(matrix, vector, schedule=schedule), expected)
    # The kernel may have been compiled from the changed state, so it is not kept
    # under the key of the state the call read.
    assert len(ops._einsum_dispatch_cache) == 0

    _check(_call(matrix, vector, schedule=schedule), expected)
    assert {key[5] for key in ops._einsum_dispatch_cache} == {after}
