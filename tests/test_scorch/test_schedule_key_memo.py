"""A schedule's cache key is validated and formatted once per distinct state.

``Schedule.cache_key`` revalidated the schedule's stored state on every read by
rebuilding a canonical Schedule from it, and a warm ``einsum`` call with a schedule
read the key twice: about 174,000 instructions per call on sapling (jobs 85220 and
85225), 46% of the call.  The key of each validated state is now memoized under a
key of that state's values (``scheduler._schedule_state_key``), which every read
rebuilds with the reads and refusals the full validation starts with, and the call
reads the key once (``ops._DeferredOptions.schedule_key``).  These tests pin:

* a second read returns the memoized string and builds no Schedule, and two equal
  schedules built apart share one entry;
* every schedule built the normal ways (from lists, ``dataclasses.replace``,
  ``copy``, ``deepcopy``, ``pickle``, the canonical copy) is keyed, and the memo
  holds no reference to it;
* the types each position admits are one table per carrier, in
  ``dataclasses.fields`` order, and the key functions admit exactly those;
* a state changed with ``object.__setattr__`` after a read -- in the Schedule or in a
  TileSpec or RelayoutSpec it holds -- is revalidated on the next read: refused when
  invalid, keyed afresh when valid;
* values that compare equal across types (``True`` and ``1``, ``False`` and ``0``) are
  different states, each judged as the full validation judges it;
* a state the memo cannot key is judged by the full validation on every read, with
  nothing memoized for it, and no method of a stored value or of its type runs;
* the key's C version (``scorch_ops._schedule_state_key``) and its Python version
  give one answer on every state above and after every change to a carrier's class
  (a data descriptor at a field name leaves a state unkeyed, whatever it returns),
  and the read builds the key in Python when scorch_ops lacks the C version;
* the module state the validation reads (the assembly and accumulator registries and
  the tile width bound) is part of the state, and is keyed only while each part is
  of an exact immutable type;
* the memo stays within its bound, counting its per-state dicts and their entries;
* a warm call reads the key once, and a schedule changed between the two points that
  used to read it gives the parse memo and the dispatch key one state's key, and
  keeps no kernel under a state the schedule left during the call.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import operator
import pickle
import weakref
from typing import Any, Callable, cast

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
def _empty_memo(fresh_kernel_caches: None):
    """The memo, the build snapshots and the kernel caches start empty
    (fresh_kernel_caches)."""


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


def _memo_size() -> int:
    """What the memo's bound counts: its per-state dicts and their entries."""

    memo = scheduler_module._SCHEDULE_KEY_MEMO
    return len(memo) + sum(map(len, memo.values()))


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


def _normal_builds() -> list[tuple[str, Schedule]]:
    """Schedules built each way callers and the compiler build them."""

    # One panel tile and a relayout without scope_var: __post_init__ rebuilds the
    # relayout through dataclasses.replace to name the panel variable.
    full = Schedule(
        loop_order=["i", "j", "k"],
        tiles=[TileSpec("k", 4, kind="panel"), TileSpec("j", 8, parallel=True)],
        relayout=RelayoutSpec("B", "j", 8),
        tag="full",
        vector_loops=["j_in"],
    )
    return [
        ("defaults", Schedule()),
        ("tuples", Schedule(loop_order=("i", "j"), tiles=(TileSpec("j", 4),))),
        ("lists", Schedule(loop_order=["i", "j"], tiles=[TileSpec("j", 4)])),
        (
            "names",
            Schedule(
                parallel_loop="i",
                assembly="two_pass_serial",
                accumulator="linked_list",
                vector_loops=(),
                entries_of="A",
            ),
        ),
        ("relayout", Schedule(relayout=RelayoutSpec("B", "k", 4, scope_var="j"))),
        ("full", full),
        ("replace", dataclasses.replace(full, tag="replaced")),
        ("copy", copy.copy(full)),
        ("deepcopy", copy.deepcopy(full)),
        ("pickle", pickle.loads(pickle.dumps(full))),
        ("canonical", scheduler_module._canonical_schedule(full)),
    ]


@pytest.mark.parametrize("label", [label for label, _ in _normal_builds()])
def test_every_schedule_built_normally_is_keyed(
    label: str, validations: list[object]
) -> None:
    schedule = dict(_normal_builds())[label]
    validations.clear()
    # The memo key compares the stored names by identity with the interned field
    # names, on whichever Python runs this.
    carriers = [schedule, *schedule.tiles]
    if schedule.relayout is not None:
        carriers.append(schedule.relayout)
    for carrier in carriers:
        stored = object.__getattribute__(carrier, "__dict__")
        names = [field.name for field in dataclasses.fields(carrier)]
        assert len(stored) == len(names)
        assert all(map(operator.is_, stored, names))
    assert scheduler_module._schedule_state_key(schedule) is not None
    first = schedule.cache_key
    assert len(validations) == 1
    assert schedule.cache_key is first
    assert len(validations) == 1
    assert first == _fresh_key(schedule)


def test_the_memo_holds_no_reference_to_a_schedule() -> None:
    tile = TileSpec("k", 4)
    relayout = RelayoutSpec("B", "k", 4, scope_var="j")
    schedule = Schedule(tiles=(tile,), relayout=relayout)
    held = [weakref.ref(schedule), weakref.ref(tile), weakref.ref(relayout)]
    for _ in range(2):
        _ = schedule.cache_key
    del schedule, tile, relayout
    gc.collect()
    assert [ref() for ref in held] == [None, None, None]


def test_each_key_table_follows_its_carriers_fields() -> None:
    for carrier, table in (
        (Schedule, scheduler_module._SCHEDULE_KEY_TYPES),
        (TileSpec, scheduler_module._TILE_SPEC_KEY_TYPES),
        (RelayoutSpec, scheduler_module._RELAYOUT_SPEC_KEY_TYPES),
    ):
        names = [field.name for field in dataclasses.fields(carrier)]
        assert [name for name, _, _ in table] == names
        assert all(row[0] is name for row, name in zip(table, names))


def _hostile_meta_str(calls: list[str]) -> type:
    """A str subclass whose type's ``__hash__`` and ``__eq__`` record each call."""

    class _HostileMeta(type):
        def __hash__(cls) -> int:
            calls.append("hash")
            return 0

        def __eq__(cls, other: object) -> bool:
            calls.append("eq")
            return NotImplemented

    class _MetaStr(str, metaclass=_HostileMeta):
        pass

    return _MetaStr


class _TupleSubclass(tuple):
    pass


class _IntSubclass(int):
    pass


class _TileSpecSubclass(TileSpec):
    pass


class _RelayoutSpecSubclass(RelayoutSpec):
    pass


def _admitted(value: object, kinds: tuple, items: type | None) -> bool:
    """Whether a key table admits ``value`` at a position, by identity alone."""

    kind = type(value)
    if not any(kind is admitted for admitted in kinds):
        return False
    if kind is tuple:
        return all(type(item) is items for item in cast(tuple, value))
    return True


def _key_candidates(meta_str: type) -> list[object]:
    """Values set into each field of each carrier: some admitted, most refused."""

    return [
        None,
        "x",
        4,
        True,
        1.5,
        b"x",
        bytearray(b"x"),
        ["x"],
        {"x": 1},
        frozenset({"x"}),
        (),
        ("x",),
        ("x", 4),
        (["x"],),
        (TileSpec("k", 4),),
        (TileSpec("k", 4), "x"),
        TileSpec("k", 4),
        RelayoutSpec("B", "k", 4, scope_var="j"),
        _HostileStr("x"),
        meta_str("x"),
        (meta_str("x"),),
        _TupleSubclass(("x",)),
        _TupleSubclass(),
        _IntSubclass(4),
        (_TileSpecSubclass("k", 4),),
        _TileSpecSubclass("k", 4),
        _RelayoutSpecSubclass("B", "k", 4, scope_var="j"),
    ]


def test_the_key_functions_admit_exactly_their_tables_types() -> None:
    calls: list[str] = []
    meta_str = _hostile_meta_str(calls)
    candidates = _key_candidates(meta_str)
    calls.clear()  # only what the key functions do below counts
    for build, table, key_of in (
        (
            Schedule,
            scheduler_module._SCHEDULE_KEY_TYPES,
            scheduler_module._schedule_state_key,
        ),
        (
            lambda: TileSpec("k", 4),
            scheduler_module._TILE_SPEC_KEY_TYPES,
            scheduler_module._tile_state_key,
        ),
        (
            lambda: RelayoutSpec("B", "k", 4, scope_var="j"),
            scheduler_module._RELAYOUT_SPEC_KEY_TYPES,
            scheduler_module._relayout_state_key,
        ),
    ):
        for name, kinds, items in table:
            for position, value in enumerate(candidates):
                carrier = build()
                assert key_of(carrier) is not None
                object.__setattr__(carrier, name, value)
                keyed = key_of(carrier) is not None
                assert keyed == _admitted(value, kinds, items), (name, position)
    assert calls == []


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
    entries = _memo_size()
    for _ in range(2):
        with pytest.raises(InvalidSchedule):
            _ = schedule.cache_key
    assert _memo_size() == entries


def test_a_shadowing_class_attribute_is_refused_after_a_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedule = Schedule()
    _ = schedule.cache_key
    monkeypatch.setattr(Schedule, "tag", property(lambda self: "descriptor value"))
    with pytest.raises(InvalidSchedule, match="diverges"):
        _ = schedule.cache_key


def _set(name: str, value: object) -> Callable[[Schedule], None]:
    return lambda schedule: object.__setattr__(schedule, name, value)


def _add_field(schedule: Schedule) -> None:
    object.__setattr__(schedule, "extra", 1)


def _remove_field(schedule: Schedule) -> None:
    object.__delattr__(schedule, "entries_of")


def _reorder_fields(schedule: Schedule) -> None:
    # The same fields and values, with ``tag`` stored last.
    object.__delattr__(schedule, "tag")
    object.__setattr__(schedule, "tag", "")


def _add_relayout_field(schedule: Schedule) -> None:
    object.__setattr__(schedule.relayout, "extra", 1)


#: Changes after which the memo does not key a schedule, each with the full
#: validation's refusal (None: it accepts the state) and an id.
_UNKEYED_CHANGES: list[tuple[Callable[[Schedule], None], str | None, str]] = [
    (_set("loop_order", ["i", "j"]), "owned tuple or None", "list"),
    (_set("tag", {"a": 1}), "exact str", "dict"),
    (_set("tag", bytearray(b"a")), "exact str", "bytearray"),
    (_set("parallel_loop", _HostileStr("i")), "exact str or None", "str subclass"),
    (
        _set("vector_loops", (["j"],)),
        "owned tuple of exact str",
        "tuple holding a list",
    ),
    (_set("relayout", "B"), "exact RelayoutSpec or None", "str for a carrier"),
    (_add_field, "stored fields", "extra field"),
    (_remove_field, "stored fields", "missing field"),
    (_add_relayout_field, "stored fields", "carrier extra field"),
    (_reorder_fields, None, "field order"),
]


@pytest.mark.parametrize(
    "change,match",
    [(change, match) for change, match, _ in _UNKEYED_CHANGES],
    ids=[label for _, _, label in _UNKEYED_CHANGES],
)
def test_an_unkeyed_state_is_judged_in_full_on_every_read(
    change: Callable[[Schedule], None], match: str | None, validations: list[object]
) -> None:
    schedule = Schedule(
        loop_order=("i", "j"), relayout=RelayoutSpec("B", "k", 4, scope_var="j")
    )
    key = schedule.cache_key
    change(schedule)
    assert scheduler_module._schedule_state_key(schedule) is None
    entries = _memo_size()
    validated = len(validations)
    for read in range(1, 3):
        if match is None:
            assert schedule.cache_key == key
        else:
            with pytest.raises(InvalidSchedule, match=match):
                _ = schedule.cache_key
        assert len(validations) == validated + read
    assert _memo_size() == entries


def test_each_carrier_class_has_only_object_as_its_base() -> None:
    # The key's class check reads the class's own attributes alone, which is exact
    # only while object is its one base.
    for carrier in (Schedule, TileSpec, RelayoutSpec):
        assert carrier.__mro__ == (carrier, object)


@pytest.mark.parametrize("hook", ["__getattribute__", "__getattr__"])
def test_a_class_level_attribute_hook_leaves_its_carriers_unkeyed(
    monkeypatch: pytest.MonkeyPatch, validations: list[object], hook: str
) -> None:
    # The key reads fields through the class, which is the full validation's
    # object.__getattribute__ read only while the class defines neither hook;
    # either one could return anything.
    schedule = Schedule(tiles=(TileSpec("k", 4),))
    key = schedule.cache_key

    def reading(self: object, name: str) -> Any:
        return object.__getattribute__(self, name)

    # raising=False: TileSpec has no __getattr__ to replace, and undo deletes the
    # one set here.
    monkeypatch.setattr(TileSpec, hook, reading, raising=False)
    assert hook in TileSpec.__dict__
    assert scheduler_module._tile_state_key(schedule.tiles[0]) is None
    entries = _memo_size()
    validated = len(validations)
    for read in range(1, 3):
        assert schedule.cache_key == key
        assert len(validations) == validated + read
    assert _memo_size() == entries


def test_no_method_of_a_stored_values_type_runs() -> None:
    # The memo key checks each type by identity before anything hashes a value or
    # its type: a metaclass's __hash__ and __eq__ are caller code too.
    calls: list[str] = []
    meta_str = _hostile_meta_str(calls)
    tagged = Schedule(tag="a")
    _ = tagged.cache_key
    object.__setattr__(tagged, "tag", meta_str("a"))
    renamed = Schedule(tag="a")
    _ = renamed.cache_key
    stored = object.__getattribute__(renamed, "__dict__")
    stored[meta_str("tag")] = stored.pop("tag")
    calls.clear()  # only what the reads below do counts
    for _ in range(2):
        with pytest.raises(InvalidSchedule, match="exact str"):
            _ = tagged.cache_key
        with pytest.raises(InvalidSchedule, match="stored fields"):
            _ = renamed.cache_key
    assert calls == []


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


@pytest.mark.parametrize(
    "module,name,mutable",
    [
        (sparse_assembly, "SPARSE_ASSEMBLY_STRATEGIES", list),
        (sparse_accumulator, "SPARSE_ACCUMULATOR_STRUCTURES", list),
        (scheduler_module, "_SCHEDULE_KEY_WHEN_SET", set),
    ],
)
def test_module_state_of_a_mutable_type_is_not_keyed(
    monkeypatch: pytest.MonkeyPatch,
    validations: list[object],
    module: object,
    name: str,
    mutable: type,
) -> None:
    # One object could hold two values over time, so identity could not tell them
    # apart: every read is judged by the full validation.
    schedule = Schedule(assembly="two_pass_serial", accumulator="linked_list")
    key = schedule.cache_key
    monkeypatch.setattr(module, name, mutable(getattr(module, name)))
    assert scheduler_module._current_context() is None
    entries = _memo_size()
    validated = len(validations)
    for read in range(1, 3):
        assert schedule.cache_key == key
        assert len(validations) == validated + read
    assert _memo_size() == entries


def test_module_state_changed_in_place_is_judged_afresh_on_each_read(
    monkeypatch: pytest.MonkeyPatch, validations: list[object]
) -> None:
    schedule = Schedule(assembly="two_pass_serial")
    key = schedule.cache_key
    registry = list(sparse_assembly.SPARSE_ASSEMBLY_STRATEGIES)
    monkeypatch.setattr(sparse_assembly, "SPARSE_ASSEMBLY_STRATEGIES", registry)
    validated = len(validations)
    assert schedule.cache_key == key
    registry.remove("two_pass_serial")
    for _ in range(2):
        with pytest.raises(InvalidSchedule, match="must name a sparse assembly"):
            _ = schedule.cache_key
    registry.append("two_pass_serial")
    assert schedule.cache_key == key
    assert len(validations) == validated + 4

    # The fields a key leaves out when unset change the key's text, not its verdict.
    when_set = set(scheduler_module._SCHEDULE_KEY_WHEN_SET)
    monkeypatch.setattr(scheduler_module, "_SCHEDULE_KEY_WHEN_SET", when_set)
    assert schedule.cache_key == key
    when_set.add("parallel_loop")
    shorter = schedule.cache_key
    assert "parallel_loop=None" in key and "parallel_loop" not in shorter
    assert shorter == _fresh_key(schedule)


def test_the_memo_stays_within_its_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler_module, "_SCHEDULE_KEY_MEMO_LIMIT", 3)
    for number in range(10):
        schedule = Schedule(tag=f"t{number}")
        assert schedule.cache_key == _fresh_key(schedule)
        assert 0 < _memo_size() <= 3


def test_the_bound_counts_the_per_state_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each tile width bound is another module state, so each read below adds a dict
    # as well as an entry; a bound on the entries alone would let the dicts grow.
    monkeypatch.setattr(scheduler_module, "_SCHEDULE_KEY_MEMO_LIMIT", 3)
    schedule = Schedule(tiles=(TileSpec("k", 4),))
    for width in range(64, 74):
        monkeypatch.setattr(scheduler_module, "MAX_AFFINE_TILE_WIDTH", width)
        assert schedule.cache_key == _fresh_key(schedule)
        assert 0 < _memo_size() <= 3
        assert len(scheduler_module._SCHEDULE_KEY_MEMO) <= 1


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


# The C version of the memo key (scorch_ops._schedule_state_key, csrc/schedule_key.cpp)
# and the Python version give one answer on every state; the read uses the C version
# when scorch_ops has it, and the Python version otherwise.


class _ScheduleSubclass(Schedule):
    pass


class _DictSubclass(dict):
    pass


class _DeleteOnly:
    """A data descriptor through ``__delete__`` alone."""

    def __get__(self, instance: object, owner: object = None) -> object:
        return "descriptor value"

    def __delete__(self, instance: object) -> None:
        raise AttributeError("not deletable")


class _PropertySubclass(property):
    pass


def _native_key(value: object) -> tuple | None:
    native = scheduler_module._native_state_key
    assert native is not None, "scorch_ops was built without _schedule_state_key"
    return cast(tuple | None, native(value, scheduler_module._NATIVE_KEY_TABLE))


def _same_key(first: object, second: object) -> bool:
    """One answer: None for None, or tuples of one length holding the same objects,
    a nested carrier's key compared the same way."""

    if first is None or second is None:
        return first is None and second is None
    if type(first) is not tuple or type(second) is not tuple:
        return False
    if len(first) != len(second):
        return False
    return all(
        a is b or (type(a) is tuple and type(b) is tuple and _same_key(a, b))
        for a, b in zip(first, second)
    )


def _one_answer(schedule: object) -> tuple | None:
    """Both versions' key of ``schedule``, required to be one answer."""

    python = scheduler_module._schedule_state_key(schedule)
    native = _native_key(schedule)
    assert _same_key(python, native), (python, native)
    return native


def _all_carriers() -> Schedule:
    return Schedule(
        loop_order=("i", "j", "k"),
        tiles=(TileSpec("j", 4),),
        relayout=RelayoutSpec("B", "k", 4, scope_var="j"),
    )


def test_scorch_ops_has_the_c_version() -> None:
    assert scheduler_module._native_state_key is not None
    assert _one_answer(_all_carriers()) is not None


def test_both_versions_give_one_answer_on_every_admission_candidate() -> None:
    calls: list[str] = []
    meta_str = _hostile_meta_str(calls)
    candidates = _key_candidates(meta_str)
    calls.clear()  # only what the key functions do below counts
    keyed = refused = 0
    for make, carrier_of, table in (
        (Schedule, lambda s: s, scheduler_module._SCHEDULE_KEY_TYPES),
        (
            lambda: Schedule(tiles=(TileSpec("k", 4),)),
            lambda s: s.tiles[0],
            scheduler_module._TILE_SPEC_KEY_TYPES,
        ),
        (
            lambda: Schedule(relayout=RelayoutSpec("B", "k", 4, scope_var="j")),
            lambda s: s.relayout,
            scheduler_module._RELAYOUT_SPEC_KEY_TYPES,
        ),
    ):
        for name, _, _ in table:
            for value in candidates:
                schedule = make()
                object.__setattr__(carrier_of(schedule), name, value)
                if _one_answer(schedule) is None:
                    refused += 1
                else:
                    keyed += 1
    assert calls == []
    assert keyed > 0 and refused > 0


@pytest.mark.parametrize(
    "change",
    [change for change, _, _ in _UNKEYED_CHANGES],
    ids=[label for _, _, label in _UNKEYED_CHANGES],
)
def test_both_versions_give_one_answer_on_every_unkeyed_change(
    change: Callable[[Schedule], None],
) -> None:
    schedule = Schedule(
        loop_order=("i", "j"), relayout=RelayoutSpec("B", "k", 4, scope_var="j")
    )
    assert _one_answer(schedule) is not None
    change(schedule)
    assert _one_answer(schedule) is None


def _changed(change: Callable[[Schedule], None]) -> Callable[[], Schedule]:
    def make() -> Schedule:
        schedule = Schedule(loop_order=("i", "j"))
        change(schedule)
        return schedule

    return make


def _replace_dict(rebuild: Callable[[dict], dict]) -> Callable[[Schedule], None]:
    def change(schedule: Schedule) -> None:
        state = object.__getattribute__(schedule, "__dict__")
        object.__setattr__(schedule, "__dict__", rebuild(state))

    return change


def _uninterned_keys(state: dict) -> dict:
    # Equal names, built at run time, so not the interned field names.
    return {"".join(list(name)): value for name, value in state.items()}


def _meta_str_name(schedule: Schedule) -> None:
    stored = object.__getattribute__(schedule, "__dict__")
    stored[_hostile_meta_str([])("tag")] = stored.pop("tag")


_INSTANCE_CASES: list[tuple[Callable[[], object], bool, str]] = [
    (
        _changed(_set("loop_order", (_HostileStr("i"), _HostileStr("j")))),
        False,
        "str subclass in a tuple",
    ),
    (_changed(_replace_dict(dict)), True, "dict replaced by an equal dict"),
    (_changed(_replace_dict(_uninterned_keys)), False, "equal names not interned"),
    (_changed(_replace_dict(_DictSubclass)), False, "dict subclass"),
    (_changed(_meta_str_name), False, "stored name of a str subclass"),
    (_ScheduleSubclass, False, "Schedule subclass"),
]


@pytest.mark.parametrize(
    "make,keyed",
    [(make, keyed) for make, keyed, _ in _INSTANCE_CASES],
    ids=[label for _, _, label in _INSTANCE_CASES],
)
def test_both_versions_give_one_answer_on_a_changed_instance(
    make: Callable[[], object], keyed: bool
) -> None:
    assert (_one_answer(make()) is not None) == keyed


def _stored(name: str) -> property:
    """A property at ``name`` that returns the stored object itself."""

    return property(lambda self: object.__getattribute__(self, "__dict__")[name])


def _hook(self: object, name: str) -> Any:
    return object.__getattribute__(self, name)


def _another_object(name: str) -> object:
    return property(lambda self: "descriptor value")


def _delete_only(name: str) -> object:
    return _DeleteOnly()


def _property_subclass(name: str) -> object:
    return _PropertySubclass(lambda self: "descriptor value")


def _function(name: str) -> object:
    return lambda self: None


def _plain_default(name: str) -> object:
    return "other"


def _hook_for(name: str) -> object:
    return _hook


#: Each carrier with a field that has a class default and, where one exists, a
#: field that has none.
_CARRIER_FIELDS = [
    (Schedule, "tag", None),
    (TileSpec, "placement", "width"),
    (RelayoutSpec, "scope_var", "operand"),
]

#: Changes to a carrier's class: (id, attribute (a field name chosen per carrier, or
#: a hook), the replacement (None: the entry is deleted), whether the state stays
#: keyed).  A data descriptor at a field name leaves the state unkeyed, whatever it
#: returns (design section 1b); a plain entry or a function does not.
_CLASS_CHANGES: list[tuple[str, str, Callable[[str], object] | None, bool]] = [
    ("property returning another object", "default", _another_object, False),
    ("property returning the stored object", "default", _stored, False),
    ("descriptor with only __delete__", "default", _delete_only, False),
    ("property subclass", "default", _property_subclass, False),
    ("function", "default", _function, True),
    ("plain default replaced", "default", _plain_default, True),
    ("default deleted", "default", None, True),
    ("property where the class had no entry", "no default", _stored, False),
    ("__getattr__ hook", "__getattr__", _hook_for, False),
    ("__getattribute__ hook", "__getattribute__", _hook_for, False),
]


def _class_change_cases() -> list[tuple[type, str, Any, bool, str]]:
    """Each class change on each carrier it applies to (Schedule has a class default
    at every field, so no change at a field without one)."""

    cases = []
    for carrier, with_default, without_default in _CARRIER_FIELDS:
        names = {"default": with_default, "no default": without_default}
        for label, attribute, replacement, keyed in _CLASS_CHANGES:
            name = names.get(attribute, attribute)
            if name is None:
                continue
            case_id = f"{carrier.__name__}-{label}"
            cases.append((carrier, name, replacement, keyed, case_id))
    return cases


@pytest.mark.parametrize(
    "carrier,name,replacement,keyed",
    [case[:4] for case in _class_change_cases()],
    ids=[case[4] for case in _class_change_cases()],
)
def test_both_versions_give_one_answer_after_a_class_change(
    monkeypatch: pytest.MonkeyPatch,
    carrier: type,
    name: str,
    replacement: Callable[[str], object] | None,
    keyed: bool,
) -> None:
    schedule = _all_carriers()  # built first: a property without a setter refuses it
    assert _one_answer(schedule) is not None
    if replacement is None:
        monkeypatch.delattr(carrier, name)
    else:
        monkeypatch.setattr(carrier, name, replacement(name), raising=False)
    assert (_one_answer(schedule) is not None) == keyed


def test_without_the_c_version_the_read_builds_the_key_in_python(
    monkeypatch: pytest.MonkeyPatch, validations: list[object]
) -> None:
    built: list[object] = []
    python = scheduler_module._schedule_state_key

    def spying(value: object) -> tuple | None:
        built.append(value)
        return cast(tuple | None, python(value))

    monkeypatch.setattr(scheduler_module, "_schedule_state_key", spying)
    schedule = _all_carriers()
    first = schedule.cache_key  # the C version builds the key
    assert built == []
    entries = _memo_size()
    monkeypatch.setattr(scheduler_module, "_native_state_key", None)
    for _ in range(2):  # the Python version's key finds the same entry
        assert schedule.cache_key is first
    assert built == [schedule, schedule]
    assert _memo_size() == entries
    assert len(validations) == 1


def test_the_c_versions_table_is_the_key_tables() -> None:
    root, nested = scheduler_module._NATIVE_KEY_TABLE
    assert len(nested) == 2
    for description, carrier_class, table in zip(
        (root, *nested),
        (
            scheduler_module._SCHEDULE_CLASS,
            scheduler_module._TILE_SPEC_CLASS,
            scheduler_module._RELAYOUT_SPEC_CLASS,
        ),
        (
            scheduler_module._SCHEDULE_KEY_TYPES,
            scheduler_module._TILE_SPEC_KEY_TYPES,
            scheduler_module._RELAYOUT_SPEC_KEY_TYPES,
        ),
    ):
        cls, mro, fields, allowed, items = description
        assert cls is carrier_class[0] and mro is carrier_class[1]
        assert mro == (cls, object)
        assert all(map(operator.is_, fields, (name for name, _, _ in table)))
        assert len(fields) == len(table)
        assert allowed == tuple(types for _, types, _ in table)
        assert items == tuple(item for _, _, item in table)


def _malformed_tables() -> list[tuple[str, Callable[[tuple], object]]]:
    def root_with(position: int, part: object) -> Callable[[tuple], object]:
        def change(table: tuple) -> object:
            root = list(table[0])
            root[position] = part
            return (tuple(root), table[1])

        return change

    return [
        ("a list", list),
        ("one part", lambda table: (table[0],)),
        ("nested a list", lambda table: (table[0], list(table[1]))),
        ("a description of four", lambda table: (table[0][:4], table[1])),
        ("a class that is not a type", root_with(0, "Schedule")),
        ("fields of another length", root_with(2, ("loop_order",))),
        ("an items rule with no description", lambda table: (table[0], ())),
    ]


@pytest.mark.parametrize(
    "change",
    [change for _, change in _malformed_tables()],
    ids=[label for label, _ in _malformed_tables()],
)
def test_a_malformed_table_raises_type_error(change: Callable[[tuple], object]) -> None:
    native = scheduler_module._native_state_key
    assert native is not None
    table = change(scheduler_module._NATIVE_KEY_TABLE)
    with pytest.raises(TypeError, match="malformed table"):
        native(_all_carriers(), table)


def test_the_c_version_takes_a_value_and_a_table() -> None:
    native = scheduler_module._native_state_key
    assert native is not None
    with pytest.raises(TypeError, match="takes a value and a table"):
        native(Schedule())


def test_a_class_entry_at_a_field_name_must_be_an_immutable_default() -> None:
    class _Carrier:
        x = property(lambda self: 1)

    record = (_Carrier, _Carrier.__mro__, _Carrier.__dict__)
    with pytest.raises(TypeError, match="immutable builtin default"):
        scheduler_module._class_entries(record, ("x",))

    class _Plain:
        pass

    record = (_Plain, _Plain.__mro__, _Plain.__dict__)
    with pytest.raises(TypeError, match="attribute of object"):
        scheduler_module._class_entries(record, ("__class__",))


def test_the_data_descriptor_test_is_the_interpreters() -> None:
    import functools
    import inspect

    class _Slotted:
        __slots__ = ("a",)

    entries = [
        property(lambda self: 1),
        _DeleteOnly(),
        _PropertySubclass(lambda self: 1),
        lambda self: 1,
        staticmethod(len),
        classmethod(len),
        _Slotted.__dict__["a"],
        type.__dict__["__dict__"],
        functools.cached_property(lambda self: 1),
        None,
        "",
        (),
        1,
        True,
    ]
    for entry in entries:
        expected = inspect.isdatadescriptor(entry)
        assert scheduler_module._is_data_descriptor(entry) == expected, entry
