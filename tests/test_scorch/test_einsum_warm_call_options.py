"""A warm ``einsum`` call runs its cached kernel without making an options snapshot.

Making the snapshot cost 85 us of a 329 us warm ``einsum`` on sapling (job 82682), with
the parse itself kept, because the kept parse's key reads the host: 19 us for the 22
environment variables through ``os.environ.get`` and 16 us of ``stat`` on the
preamble's header files over NFS (job 82687).  A warm call now matches the earlier call
on ``environment_reuse_key`` instead: the environment variables the parse reads, read
from ``os.environ``'s own store, and the context overrides.  These tests pin:

* the fast reader sees every variable the parse reads, set, changed or removed, and
  agrees with ``os.environ.get``;
* a warm call makes no snapshot and reads no header file;
* the first call under an environment still parses, so a refused environment is refused
  on the call that meets it, cached kernel or not;
* a variable or context override that changes selects its own kernel entry;
* a schedule named while the context forces another selects the named one, as the
  parse does, and its kernel;
* a kernel is kept under the cache fingerprint of the snapshot it was built from, and a
  call handed a snapshot selects the kernel a call that made the same snapshot built;
* ``clear_build_snapshot_cache()`` makes the next call parse again, and a snapshot made
  after a header change is made from the changed header;
* a change to any one of the variables misses the parse memo, and the call's key is
  then the one a fresh parse gives;
* nothing a snapshot holds can be assigned, and a forged write to one a caller was
  handed reaches no other call;
* the dispatch-cache hit raises a kernel's coded refusal as its structured error.
"""

from __future__ import annotations

import dataclasses
import enum
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import torch

import scorch
import scorch.ops as ops  # type: ignore[import-untyped]
import scorch.utils as utils  # type: ignore[import-untyped]
from scorch import STensor  # type: ignore[import-untyped]
from scorch.compiler import (  # type: ignore[import-untyped]
    cin_analysis as cin_analysis_module,
    compile_options,
    scheduler as scheduler_module,
)
from scorch.compiler.cin_analysis import (  # type: ignore[import-untyped]
    full_cin_verification,
)
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    CompileOptionsError,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    Schedule,
    TileSpec,
    regblock_force,
    schedule_force,
)
from scorch.exceptions import IndexCapacityError  # type: ignore[import-untyped]

_KEYS = compile_options._ENVIRONMENT_OPTION_KEYS


@pytest.fixture(autouse=True)
def _nothing_kept_between_tests(monkeypatch: pytest.MonkeyPatch):
    compile_options.clear_build_snapshot_cache()
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
    yield
    compile_options.clear_build_snapshot_cache()


@pytest.fixture
def made(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The keyword arguments of every snapshot an ``einsum`` call makes, in order."""

    calls: list[dict[str, Any]] = []
    make = ops._compile_options_from_kwargs

    def counted(kwargs: dict[str, Any], **keywords: Any) -> CompileOptions:
        calls.append(kwargs)
        return make(kwargs, **keywords)

    monkeypatch.setattr(ops, "_compile_options_from_kwargs", counted)
    return calls


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
    assert torch.allclose(got, expected, rtol=1e-12, atol=1e-12)


def _decoded(values: tuple[object, ...]) -> tuple[object, ...]:
    tag, read = values
    if tag == "decoded":
        return read  # type: ignore[return-value]
    decode = os.environ.decodevalue  # type: ignore[attr-defined]
    return tuple(None if value is None else decode(value) for value in read)


def test_the_fast_reader_agrees_with_os_environ_for_every_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = compile_options._environment_values(os.environ)
    if os.name == "posix":
        assert before[0] == "encoded"
    for key in _KEYS:
        assert _decoded(before) == tuple(map(os.environ.get, _KEYS))
        for value in (f"set-{key}", "caf\udce9", ""):
            monkeypatch.setenv(key, value)
            now = compile_options._environment_values(os.environ)
            assert now != before, key
            assert _decoded(now) == tuple(map(os.environ.get, _KEYS))
            before = now
        monkeypatch.delenv(key)
        now = compile_options._environment_values(os.environ)
        assert now != before, key
        assert _decoded(now) == tuple(map(os.environ.get, _KEYS))
        before = now


def test_another_mapping_is_read_through_get_and_never_equals_os_environ() -> None:
    copy = dict(os.environ)
    values = compile_options._environment_values(copy)
    assert values == ("decoded", tuple(map(copy.get, _KEYS)))
    if os.name == "posix":
        assert values != compile_options._environment_values(os.environ)


def test_a_warm_call_makes_no_options_and_reads_no_header_file(
    made: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix, vector, expected = _operands(40)
    _check(_call(matrix, vector), expected)
    assert len(made) == 1

    reads: list[int] = []
    identity = utils.jit_preamble_file_identity

    def counted() -> Any:
        reads.append(1)
        return identity()

    monkeypatch.setattr(utils, "jit_preamble_file_identity", counted)
    for _ in range(3):
        _check(_call(matrix, vector), expected)
    assert len(made) == 1
    assert reads == []
    assert len(ops._einsum_dispatch_cache) == 1


@pytest.mark.parametrize(
    ("name", "value", "code"),
    [
        ("CPATH", "/usr/include", "unsupported_compiler_environment"),
        ("TORCH_DONT_CHECK_COMPILER_ABI", "1", "unsupported_compiler_abi_policy"),
        ("SCORCH_REGBLOCK", "yes", "invalid_environment_boolean"),
    ],
)
def test_a_refused_environment_is_refused_on_a_warm_call(
    made: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    code: str,
) -> None:
    monkeypatch.delenv(name, raising=False)
    matrix, vector, expected = _operands(24)
    _call(matrix, vector)
    monkeypatch.setenv(name, value)
    for _ in range(2):
        with pytest.raises(CompileOptionsError) as error:
            _call(matrix, vector)
        assert error.value.diagnostics[0].code == code
    monkeypatch.delenv(name)
    before = len(made)
    _check(_call(matrix, vector), expected)
    assert len(made) == before


def test_a_changed_variable_or_override_selects_its_own_kernel_entry(
    made: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCORCH_REGBLOCK_MAX_N", raising=False)
    matrix, vector, expected = _operands(32)
    _check(_call(matrix, vector), expected)
    entries = len(ops._einsum_dispatch_cache)

    monkeypatch.setenv("SCORCH_REGBLOCK_MAX_N", "3")
    for _ in range(2):
        _check(_call(matrix, vector), expected)
    assert len(ops._einsum_dispatch_cache) == entries + 1
    monkeypatch.delenv("SCORCH_REGBLOCK_MAX_N")

    plain = CompileOptions.from_environment()
    with regblock_force(not plain.scheduler.regblock_enabled):
        for _ in range(2):
            _check(_call(matrix, vector), expected)
    assert len(ops._einsum_dispatch_cache) == entries + 2
    with full_cin_verification(True):
        for _ in range(2):
            _check(_call(matrix, vector), expected)
    assert len(ops._einsum_dispatch_cache) == entries + 3

    _check(_call(matrix, vector), expected)
    assert len(made) == 4


def test_a_schedule_named_while_the_context_forces_another_selects_the_named_one(
    made: list[dict[str, Any]],
) -> None:
    matrix, vector, expected = _operands(16)
    named = Schedule(loop_order=("i", "j"))
    _check(_call(matrix, vector, schedule=named), expected)
    entries = len(ops._einsum_dispatch_cache)
    with schedule_force(Schedule(loop_order=("j", "i"))):
        for _ in range(2):
            _check(_call(matrix, vector, schedule=named), expected)
    assert len(ops._einsum_dispatch_cache) == entries
    assert len(made) == 1


def test_a_kernel_is_kept_under_the_snapshot_it_was_built_from(
    made: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SCORCH_REGBLOCK_MAX_N", raising=False)
    matrix, vector, _ = _operands(20)
    other, other_vector, other_expected = _operands(28)
    _call(matrix, vector)
    assert len(ops._einsum_dispatch_cache) == 1

    make = ops._compile_options_from_kwargs

    def changing(kwargs: dict[str, Any], **keywords: Any) -> CompileOptions:
        options = make(kwargs, **keywords)
        monkeypatch.setenv("SCORCH_REGBLOCK_MAX_N", "3")
        return options

    monkeypatch.setattr(ops, "_compile_options_from_kwargs", changing)
    _check(_call(other, other_vector), other_expected)
    assert len(made) == 2
    assert len(ops._einsum_dispatch_cache) == 2

    # The environment changed after the parse the kernel was built from, so the next
    # call parses under the new one and selects a kernel of its own.
    monkeypatch.setattr(ops, "_compile_options_from_kwargs", make)
    _check(_call(other, other_vector), other_expected)
    assert len(made) == 3
    assert len(ops._einsum_dispatch_cache) == 3


def test_clearing_the_caches_makes_the_next_call_parse_again(
    made: list[dict[str, Any]],
) -> None:
    matrix, vector, expected = _operands(24)
    _call(matrix, vector)
    _call(matrix, vector)
    assert len(made) == 1
    compile_options.clear_build_snapshot_cache()
    for _ in range(2):
        _check(_call(matrix, vector), expected)
    assert len(made) == 2
    assert len(ops._einsum_dispatch_cache) == 1


def test_a_snapshot_made_after_a_header_change_reads_the_changed_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import resources

    packaged = resources.files("scorch").joinpath("csrc")
    directory = tmp_path / "csrc"
    directory.mkdir()
    for name in ("header.h", "native_abi.h", "scorch_policy.h"):
        (directory / name).write_text(
            packaged.joinpath(name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    monkeypatch.setattr(utils, "_NATIVE_RESOURCES", directory)
    monkeypatch.setattr(utils, "_preamble_text_memo", None)
    monkeypatch.setattr(utils, "_preamble_paths_memo", None)

    first = ops._DeferredOptions({})
    before = first.options()
    with (directory / "native_abi.h").open("a", encoding="utf-8") as stream:
        stream.write("// changed by test_einsum_warm_call_options\n")
    second = ops._DeferredOptions({})
    assert second._options is None
    assert second.reuse_key == first.reuse_key
    after = second.options()
    assert not second.still_reusable()
    assert ops._DeferredOptions({}).reuse_key == after.cache_fingerprint

    fresh = CompileOptions._parse_environment()
    assert after.cache_key != before.cache_key
    assert after.cache_key == fresh.cache_key
    assert after.build.preamble_source == fresh.build.preamble_source


def test_a_call_handed_a_snapshot_keys_its_kernel_by_the_snapshot() -> None:
    options = CompileOptions.from_environment()
    deferred = ops._DeferredOptions({"_compile_options": options})
    assert deferred.reuse_key == options.cache_fingerprint
    assert deferred.options() is options
    assert deferred.still_reusable()


@pytest.mark.parametrize("regblock", [None, True, False])
@pytest.mark.parametrize("verify", [True, False])
def test_the_warm_key_holds_what_the_readers_return(
    regblock: Any, verify: bool
) -> None:
    """``environment_reuse_key`` reads the two context variables itself.

    It holds what ``get_forced_regblock`` and ``get_full_cin_verification`` return, and
    the environment values ``_environment_values`` reads.
    """

    with regblock_force(regblock), full_cin_verification(verify):
        key = compile_options.environment_reuse_key()
        assert key is not None
        values, read_regblock, read_verify = key
        assert values == compile_options._environment_values(os.environ)
        assert read_regblock is scheduler_module.get_forced_regblock()
        assert read_verify is cin_analysis_module.get_full_cin_verification()


@pytest.mark.parametrize(
    ("variable", "reader", "value"),
    [
        ("_REGBLOCK_FORCE", "get_forced_regblock", 1),
        ("_VERIFY_CIN_CONTEXT", "get_full_cin_verification", "yes"),
        ("_VERIFY_CIN_CONTEXT", "get_full_cin_verification", None),
    ],
)
def test_an_override_its_reader_refuses_gives_no_warm_key(
    variable: str, reader: str, value: object
) -> None:
    """A value the reader refuses gives None, so the call makes its options in full."""

    module = scheduler_module if variable == "_REGBLOCK_FORCE" else cin_analysis_module
    context_variable = getattr(module, variable)
    token = context_variable.set(value)
    try:
        with pytest.raises(TypeError):
            getattr(module, reader)()
        assert compile_options.environment_reuse_key() is None
    finally:
        context_variable.reset(token)


def test_a_call_without_keywords_gets_the_holder_one_with_them_gets(
    made: list[dict[str, Any]],
) -> None:
    """The holder's path for no keyword arguments resolves what the general one does."""

    forced = Schedule(loop_order=("j", "i"))
    for context in (None, forced):
        with schedule_force(context):
            # ``time_dict`` changes nothing the options are made from; it only sends
            # the holder down its general path.
            general = ops._DeferredOptions({"time_dict": {}})
            plain = ops._DeferredOptions({})
            assert plain._options is None
            assert plain.reuse_key is general.reuse_key
            assert plain.requested_schedule is general.requested_schedule
            assert plain.requested_schedule is context
    assert len(made) == 2


def test_the_options_part_of_a_dispatch_key_is_one_string() -> None:
    """A dict lookup hashes its key on every call, and CPython keeps no tuple's hash.

    The options' cache key is a nested tuple of about 100 objects; the fingerprint is
    one string, which keeps its hash, and a warm call is handed the same string object
    the parse memo holds.  Equal snapshots give equal fingerprints, unequal ones
    different fingerprints.
    """

    first = ops._DeferredOptions({})
    first.options()
    second = ops._DeferredOptions({})
    assert second._options is None
    assert type(second.reuse_key) is str
    assert second.reuse_key is ops._DeferredOptions({}).reuse_key
    options = second.options()
    assert second.reuse_key == options.cache_fingerprint
    assert options.cache_fingerprint == compile_options.canonical_cache_digest(
        options.cache_key
    )
    with regblock_force(not options.scheduler.regblock_enabled):
        other = ops._DeferredOptions({})
        assert other.options().cache_key != options.cache_key
        assert other.reuse_key != second.reuse_key


@pytest.mark.parametrize("key", _KEYS)
def test_a_changed_variable_makes_the_warm_holder_parse_again(
    made: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    """Every variable the parse reads: a change misses the parse memo.

    The holder then parses, and its key is the fingerprint a fresh parse of the changed
    environment gives, or it raises the refusal that parse raises.
    """

    # From the environment the test runs in: without PATH there may be no compiler.
    original = os.environ.get(key)
    first = ops._DeferredOptions({})
    assert ops._DeferredOptions({}).reuse_key == first.reuse_key
    parsed = len(made)
    assert parsed == 1
    for value in (f"set-{key}", ""):
        if value == original:
            continue
        monkeypatch.setenv(key, value)
        try:
            fresh = CompileOptions._parse_environment()
        except CompileOptionsError as refused:
            with pytest.raises(CompileOptionsError) as error:
                ops._DeferredOptions({})
            assert error.value.diagnostics[0].code == refused.diagnostics[0].code
        else:
            assert ops._DeferredOptions({}).reuse_key == fresh.cache_fingerprint
        assert len(made) == parsed + 1, value
        parsed = len(made)


def _parts(value: object, path: str) -> Iterator[tuple[str, object]]:
    """``value`` and everything it holds: each attribute of a dataclass instance (its
    fields and anything else it stores), and each item of a tuple or frozenset."""

    yield path, value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for name, held in vars(value).items():
            yield from _parts(held, f"{path}.{name}")
    elif type(value) in (tuple, frozenset):
        for index, held in enumerate(value):  # type: ignore[arg-type]
            yield from _parts(held, f"{path}[{index}]")


_IMMUTABLE_LEAVES = (type(None), bool, int, float, str, bytes)


@pytest.mark.parametrize(
    "schedule",
    [
        None,
        Schedule(
            loop_order=("i", "k", "j"),
            tiles=(TileSpec("j", 8, placement="child_of:i"),),
        ),
    ],
)
def test_nothing_a_snapshot_holds_can_be_assigned(schedule: Any) -> None:
    """A fingerprint is computed once per kept snapshot (``_keep_cache_keys``) and every
    copy of it inherits it, so it is the snapshot's only while nothing the snapshot
    holds can change: every object it holds is a frozen dataclass, which refuses
    assignment and deletion of any attribute, or an immutable value."""

    snapshot = CompileOptions.from_environment(requested_schedule=schedule)
    instances = 0
    for path, value in _parts(snapshot, "options"):
        if dataclasses.is_dataclass(value):
            instances += 1
            assert type(value).__dataclass_params__.frozen, path  # type: ignore
            for name in vars(value):
                with pytest.raises(dataclasses.FrozenInstanceError):
                    setattr(value, name, None)
                with pytest.raises(dataclasses.FrozenInstanceError):
                    delattr(value, name)
        elif type(value) not in (tuple, frozenset):
            immutable = type(value) in _IMMUTABLE_LEAVES or isinstance(value, enum.Enum)
            assert immutable, (path, type(value))
    # The snapshot, its build, scheduler, cost model, verification and pass options.
    assert instances >= 6 + (0 if schedule is None else 2)


def test_a_forged_write_to_a_handed_out_snapshot_reaches_no_other_call(
    made: list[dict[str, Any]],
) -> None:
    """``object.__setattr__`` is the one way past a frozen dataclass.  A write made that
    way to a snapshot a caller was handed changes no other call's snapshot or key:
    each call is handed its own copy of every policy object (``_own_copy_of_options``),
    and the parse memo holds a string.  The forged copy itself keeps the fingerprint it
    was handed with, as it kept its cache key before the fingerprint replaced the key.
    """

    matrix, vector, expected = _operands(28)
    _check(_call(matrix, vector), expected)
    warm = ops._DeferredOptions({}).reuse_key
    handed = CompileOptions.from_environment()
    assert handed.cache_fingerprint == warm
    object.__setattr__(handed, "emit_comments", not handed.emit_comments)
    object.__setattr__(handed.build, "extra_cflags", ("-DFORGED",))
    object.__setattr__(handed.scheduler, "regblock_max_n", 1)
    object.__setattr__(handed.scheduler.cost_model, "alpha", -1.0)
    object.__setattr__(handed.verification, "verify_cin", True)

    assert ops._DeferredOptions({}).reuse_key == warm
    again = CompileOptions.from_environment()
    fresh = CompileOptions._parse_environment()
    assert again.cache_key == fresh.cache_key
    assert again.cache_fingerprint == fresh.cache_fingerprint == warm
    for _ in range(2):
        _check(_call(matrix, vector), expected)
    assert len(made) == 1
    assert len(ops._einsum_dispatch_cache) == 1


def test_a_warm_call_raises_a_coded_native_refusal_as_its_structured_error() -> None:
    """The dispatch-cache hit maps a kernel's coded refusal itself, as every other call
    of a kernel does (``raise_if_native_coded``); any other error leaves unchanged."""

    matrix, vector, expected = _operands(18)
    _check(_call(matrix, vector), expected)
    ((key, entry),) = ops._einsum_dispatch_cache.items()
    line = "a result level passes the 32-bit position bound"

    def refusing(*args: Any) -> Any:
        raise RuntimeError(f"{line} [index_position_exceeds_int32]")

    def failing(*args: Any) -> Any:
        raise RuntimeError("not a coded refusal")

    ops._einsum_dispatch_cache[key] = (SimpleNamespace(evaluate=refusing), *entry[1:])
    with pytest.raises(IndexCapacityError) as refused:
        _call(matrix, vector)
    assert line in str(refused.value)
    assert "[index_position_exceeds_int32]" not in str(refused.value)
    assert type(refused.value.__cause__) is RuntimeError

    ops._einsum_dispatch_cache[key] = (SimpleNamespace(evaluate=failing), *entry[1:])
    with pytest.raises(RuntimeError) as other:
        _call(matrix, vector)
    assert type(other.value) is RuntimeError
    assert str(other.value) == "not a coded refusal"

    ops._einsum_dispatch_cache[key] = entry
    _check(_call(matrix, vector), expected)
