"""``CompileOptions.from_environment`` parses once per distinct set of inputs.

Every public call takes an options snapshot, and parsing one (about twenty environment
values validated one by one, four validated objects built) cost 120 us of a 390 us warm
``einsum`` on sapling.  The parsed snapshot is kept under a key of everything the parse
reads.  These tests pin what keeps that exact:

* the key reads exactly the variables the parse reads, in its order;
* under a change of any variable, context override, schedule object, header file or
  working directory, the kept snapshot equals a fresh parse, cache key included;
* an input the parse refuses is refused on every call and never kept;
* every call receives its own snapshot, with its own nested policies and build snapshot;
* ``clear_build_snapshot_cache()`` forgets the parsed snapshots too.

The second half pins the other part of the same change: a warm ``einsum`` call that was
handed no compilation context makes none, and a frontend stage begun from an earlier
reading is timed from that reading.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import gc
import os
from pathlib import Path
import time
from typing import Any, Optional

import pytest
import torch

import scorch
import scorch.ops as ops  # type: ignore[import-untyped]
from scorch import STensor  # type: ignore[import-untyped]
from scorch.compiler import compile_options  # type: ignore[import-untyped]
from scorch.compiler.cin_analysis import (  # type: ignore[import-untyped]
    full_cin_verification,
)
from scorch.compiler.compilation_context import (  # type: ignore[import-untyped]
    CompilationContext,
    CompilationContextError,
    CompilerStageId,
)
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    CompileOptionsError,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    Schedule,
    regblock_force,
    schedule_force,
)


@pytest.fixture(autouse=True)
def _nothing_kept_between_tests():
    compile_options.clear_build_snapshot_cache()
    yield
    compile_options.clear_build_snapshot_cache()


@pytest.fixture
def parses(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The arguments of every parse made during the test, in order."""

    made: list[dict[str, Any]] = []
    parse = CompileOptions._parse_environment.__func__

    def recorded(cls: type, environ: Optional[Mapping[str, str]] = None, **kw: Any):
        made.append({"environ": environ, **kw})
        return parse(cls, environ, **kw)

    monkeypatch.setattr(CompileOptions, "_parse_environment", classmethod(recorded))
    return made


def _environment(**changes: Optional[str]) -> dict[str, str]:
    environ = dict(os.environ)
    for key, value in changes.items():
        if value is None:
            environ.pop(key, None)
        else:
            environ[key] = value
    return environ


# The parse itself, captured before any test replaces it with a recorder.
_FRESH_PARSE = CompileOptions._parse_environment


def _assert_same_as_a_fresh_parse(kept: CompileOptions, **arguments: Any) -> None:
    fresh = _FRESH_PARSE(**arguments)
    assert kept == fresh
    assert kept.cache_key == fresh.cache_key
    assert kept.cache_fingerprint == fresh.cache_fingerprint


class _RecordingMapping(dict):
    """A mapping that records the keys read through ``get``, as the parse reads."""

    def __init__(self, values: Mapping[str, str]) -> None:
        super().__init__(values)
        self.reads: list[str] = []

    def get(self, key: str, default: object = None) -> Any:  # type: ignore[override]
        self.reads.append(key)
        return super().get(key, default)


def test_the_key_reads_exactly_the_variables_the_parse_reads() -> None:
    recording = _RecordingMapping(os.environ)
    CompileOptions._parse_environment(recording)
    assert tuple(recording.reads) == compile_options._ENVIRONMENT_OPTION_KEYS


def test_repeated_calls_parse_once_and_each_receives_its_own_snapshot(
    parses: list[dict[str, Any]],
) -> None:
    snapshots = [CompileOptions.from_environment() for _ in range(4)]

    assert len(parses) == 1
    for snapshot in snapshots:
        _assert_same_as_a_fresh_parse(snapshot)
    for first, second in zip(snapshots, snapshots[1:]):
        assert first is not second
        assert first.scheduler is not second.scheduler
        assert first.scheduler.cost_model is not second.scheduler.cost_model
        assert first.verification is not second.verification
        assert first.build is not second.build


_VALUES = [
    ("SCORCH_REGBLOCK", "1"),
    ("SCORCH_REGBLOCK", "0"),
    ("SCORCH_REGBLOCK_MAX_N", "4"),
    ("SCORCH_REGBLOCK_T", "2"),
    ("SCORCH_VERIFY_CIN", "1"),
    ("SCORCH_REGBLOCK_DUAL", "0"),
    ("SCORCH_JIT_TUNE_HOOKS", "1"),
    ("TORCH_DONT_CHECK_COMPILER_ABI", "0"),
    ("TORCH_NO_COMPILER_WRAPPER", "1"),
]


@pytest.mark.parametrize(("key", "value"), _VALUES)
def test_a_changed_variable_parses_again_and_equals_a_fresh_parse(
    key: str,
    value: str,
    parses: list[dict[str, Any]],
) -> None:
    base = _environment(**{key: None})
    changed = _environment(**{key: value})

    first = CompileOptions.from_environment(base)
    second = CompileOptions.from_environment(changed)
    third = CompileOptions.from_environment(base)
    fourth = CompileOptions.from_environment(changed)

    assert len(parses) == 2
    _assert_same_as_a_fresh_parse(first, environ=base)
    _assert_same_as_a_fresh_parse(second, environ=changed)
    _assert_same_as_a_fresh_parse(third, environ=base)
    _assert_same_as_a_fresh_parse(fourth, environ=changed)


def test_a_changed_compiler_and_search_path_parse_again(
    tmp_path: Path,
    parses: list[dict[str, Any]],
) -> None:
    compiler = CompileOptions.from_environment().build.cxx_compiler_path
    directory = tmp_path / "bin"
    directory.mkdir()
    (directory / "c++").symlink_to(compiler)
    by_path = _environment(PATH=str(directory), CXX=None)
    by_name = _environment(CXX=compiler)

    for environ in (by_path, by_name, by_path):
        _assert_same_as_a_fresh_parse(
            CompileOptions.from_environment(environ), environ=environ
        )
    assert CompileOptions.from_environment(by_path).build.cxx_compiler_path == str(
        directory / "c++"
    )
    # One parse for the plain call that found the compiler, one for each mapping.
    assert len(parses) == 3


def test_the_working_directory_is_part_of_the_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parses: list[dict[str, Any]],
) -> None:
    CompileOptions.from_environment()
    monkeypatch.chdir(tmp_path)
    moved = CompileOptions.from_environment()
    assert len(parses) == 2
    _assert_same_as_a_fresh_parse(moved)


def test_context_overrides_are_part_of_the_key(
    parses: list[dict[str, Any]],
) -> None:
    plain = CompileOptions.from_environment()
    schedule = Schedule(loop_order=("i", "j"))
    with regblock_force(not plain.scheduler.regblock_enabled):
        flipped = CompileOptions.from_environment()
        _assert_same_as_a_fresh_parse(flipped)
    with schedule_force(schedule):
        forced = CompileOptions.from_environment()
        _assert_same_as_a_fresh_parse(forced)
        assert forced.requested_schedule is schedule
    with full_cin_verification(True):
        verified = CompileOptions.from_environment()
        _assert_same_as_a_fresh_parse(verified)
    again = CompileOptions.from_environment()

    assert flipped.scheduler.regblock_enabled is not plain.scheduler.regblock_enabled
    assert verified.verification.verify_cin is True
    assert again == plain
    assert len(parses) == 4


def test_explicit_overrides_are_part_of_the_key(
    parses: list[dict[str, Any]],
) -> None:
    arguments = [
        {"regblock_override": True},
        {"regblock_override": False},
        {"regblock_override": None},
        {"verify_cin_override": True},
        {"verify_cin_override": False},
        {"forced_schedule": None},
    ]
    for keywords in arguments:
        _assert_same_as_a_fresh_parse(
            CompileOptions.from_environment(**keywords), **keywords
        )
        _assert_same_as_a_fresh_parse(
            CompileOptions.from_environment(**keywords), **keywords
        )
    assert len(parses) == len(arguments)


def test_a_schedule_is_keyed_by_identity(parses: list[dict[str, Any]]) -> None:
    first = Schedule(loop_order=("i", "k", "j"))
    equal = Schedule(loop_order=("i", "k", "j"))

    one = CompileOptions.from_environment(requested_schedule=first)
    two = CompileOptions.from_environment(requested_schedule=first)
    three = CompileOptions.from_environment(requested_schedule=equal)

    assert one.requested_schedule is first
    assert two.requested_schedule is first
    assert three.requested_schedule is equal
    assert len(parses) == 2


def test_a_conflicting_forced_schedule_is_refused_every_time() -> None:
    requested = Schedule(loop_order=("i", "k", "j"))
    forced = Schedule(loop_order=("i", "j", "k"))
    for _ in range(2):
        with pytest.raises(CompileOptionsError) as error:
            CompileOptions.from_environment(
                requested_schedule=requested, forced_schedule=forced
            )
        assert error.value.diagnostics[0].code == "conflicting_schedule"
    assert not compile_options._environment_options_cache


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"CPATH": "/usr/include"}, "unsupported_compiler_environment"),
        ({"TORCH_DONT_CHECK_COMPILER_ABI": "1"}, "unsupported_compiler_abi_policy"),
        ({"SCORCH_REGBLOCK": "yes"}, "invalid_environment_boolean"),
    ],
)
def test_a_refused_environment_is_refused_every_time_and_never_kept(
    changes: dict[str, str],
    code: str,
) -> None:
    CompileOptions.from_environment()
    environ = _environment(**changes)
    for _ in range(2):
        with pytest.raises(CompileOptionsError) as error:
            CompileOptions.from_environment(environ)
        assert error.value.diagnostics[0].code == code
    assert len(compile_options._environment_options_cache) == 1


def test_a_value_that_is_not_a_string_is_refused_by_the_parse() -> None:
    environ: dict[str, object] = dict(os.environ)
    environ["SCORCH_REGBLOCK"] = 1
    for _ in range(2):
        with pytest.raises(CompileOptionsError) as error:
            CompileOptions.from_environment(environ)  # type: ignore[arg-type]
        assert error.value.diagnostics[0].code == "invalid_environment_value"
    assert not compile_options._environment_options_cache


def test_an_environment_change_during_the_parse_does_not_reach_the_kept_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The parse reads the values its key holds, not the environment again, so the
    # kept snapshot is the parse of its key even when a variable changes meanwhile.
    # The next call reads the changed value, misses and parses again.
    parse = CompileOptions._parse_environment.__func__
    made: list[Optional[Mapping[str, str]]] = []

    def changing(cls: type, environ: Optional[Mapping[str, str]] = None, **kw: Any):
        made.append(environ)
        parsed = parse(cls, environ, **kw)
        if len(made) == 1:
            monkeypatch.setenv("SCORCH_REGBLOCK_MAX_N", "3")
        return parsed

    monkeypatch.delenv("SCORCH_REGBLOCK_MAX_N", raising=False)
    monkeypatch.setattr(CompileOptions, "_parse_environment", classmethod(changing))
    first = CompileOptions.from_environment()
    assert made[0] is not os.environ
    assert "SCORCH_REGBLOCK_MAX_N" not in made[0]
    assert first.scheduler.regblock_max_n == 8
    assert len(compile_options._environment_options_cache) == 1
    second = CompileOptions.from_environment()
    assert len(made) == 2
    assert second.scheduler.regblock_max_n == 3
    assert CompileOptions.from_environment().scheduler.regblock_max_n == 3
    assert len(made) == 2


def test_a_build_input_changed_during_the_parse_is_not_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The working directory is in the build snapshot's key and is not an environment
    # variable, so the parse cannot be given a copy of it: a change while it parses
    # leaves a snapshot that is used once and not kept.
    parse = CompileOptions._parse_environment.__func__

    def changing(cls: type, environ: Optional[Mapping[str, str]] = None, **kw: Any):
        parsed = parse(cls, environ, **kw)
        monkeypatch.chdir(tmp_path)
        return parsed

    monkeypatch.setattr(CompileOptions, "_parse_environment", classmethod(changing))
    CompileOptions.from_environment()
    assert not compile_options._environment_options_cache


def test_a_changed_header_parses_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parses: list[dict[str, Any]],
) -> None:
    import scorch.utils as utils  # type: ignore[import-untyped]
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

    before = CompileOptions.from_environment()
    with (directory / "native_abi.h").open("a", encoding="utf-8") as stream:
        stream.write("// changed by test_environment_options_cache\n")
    after = CompileOptions.from_environment()

    assert len(parses) == 2
    assert after.cache_key != before.cache_key
    _assert_same_as_a_fresh_parse(after)


def test_clearing_the_build_snapshots_forgets_the_parsed_ones(
    parses: list[dict[str, Any]],
) -> None:
    CompileOptions.from_environment()
    compile_options.clear_build_snapshot_cache()
    assert not compile_options._environment_options_cache
    CompileOptions.from_environment()
    assert len(parses) == 2


def test_the_cache_is_bounded(parses: list[dict[str, Any]]) -> None:
    limit = compile_options._BUILD_SNAPSHOT_CACHE_LIMIT
    for width in range(1, limit + 4):
        CompileOptions.from_environment(_environment(SCORCH_REGBLOCK_MAX_N=str(width)))
        assert len(compile_options._environment_options_cache) <= limit
    assert len(parses) == limit + 3


def test_a_replaced_snapshot_computes_its_own_key() -> None:
    kept = CompileOptions.from_environment()
    other = kept.with_regblock_enabled(not kept.scheduler.regblock_enabled)
    assert id(other) not in compile_options._kept_cache_keys
    assert other.cache_key != kept.cache_key
    assert other.cache_key == (
        other.semantic_cache_key,
        other.verification.cache_key,
        other.build_cache_key,
    )


# --- The compilation context of a warm einsum call ---------------------------------


def _coo_operand(rows: int) -> tuple[STensor, torch.Tensor, torch.Tensor]:
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
    return matrix, x, dense @ x


def _dense(result: Any) -> torch.Tensor:
    return result.to_torch() if isinstance(result, STensor) else result


def test_a_warm_einsum_call_makes_no_compilation_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix, x, expected = _coo_operand(40)
    vector = STensor.from_torch(x, "x")
    made: list[CompilationContext] = []

    def counted(*args: Any, **kwargs: Any) -> CompilationContext:
        context = CompilationContext(*args, **kwargs)
        made.append(context)
        return context

    monkeypatch.setattr(ops, "CompilationContext", counted)
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    first = scorch.einsum("ij,j->i", matrix, vector, format="d")
    assert torch.allclose(_dense(first), expected, rtol=1e-12, atol=1e-12)
    compiled = len(made)
    assert compiled >= 1
    records = made[0].stage_run_records
    frontend = CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION
    assert records[0].stage_id is frontend
    assert records[0].sequence_index == 0

    for _ in range(3):
        warm = scorch.einsum("ij,j->i", matrix, vector, format="d")
        assert torch.allclose(_dense(warm), expected, rtol=1e-12, atol=1e-12)
    assert len(made) == compiled


def test_a_warm_einsum_call_handed_a_context_cancels_its_frontend_stage() -> None:
    matrix, x, expected = _coo_operand(24)
    vector = STensor.from_torch(x, "x")
    scorch.einsum("ij,j->i", matrix, vector, format="d")

    options = CompileOptions.from_environment()
    context = CompilationContext(compile_options=options)
    warm = scorch.einsum(
        "ij,j->i",
        matrix,
        vector,
        format="d",
        _compile_options=options,
        _compilation_context=context,
    )

    assert torch.allclose(_dense(warm), expected, rtol=1e-12, atol=1e-12)
    assert context.stage_run_records == ()
    assert context._active_stage_tokens == ()
    assert len(context._retired_stage_token_identities) == 1


def test_a_stage_begun_from_an_earlier_reading_is_timed_from_it() -> None:
    options = CompileOptions.from_environment()
    context = CompilationContext(compile_options=options)
    started = time.perf_counter_ns()
    time.sleep(0.01)
    token = context.begin_stage(
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        compile_options=options,
        started_ns=started,
    )
    context.complete_stage(token)
    assert context.stage_run_records[0].duration_ns >= 10_000_000


@pytest.mark.parametrize("started", [True, -1, 1.5, "0", "later"])
def test_a_stage_start_that_is_not_an_earlier_reading_is_refused(
    started: object,
) -> None:
    options = CompileOptions.from_environment()
    context = CompilationContext(compile_options=options)
    if started == "later":
        started = time.perf_counter_ns() + 10**12
    with pytest.raises(CompilationContextError) as error:
        context.begin_stage(
            CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
            compile_options=options,
            started_ns=started,  # type: ignore[arg-type]
        )
    assert error.value.diagnostic.code == "invalid_stage_start"


def test_kept_snapshots_and_their_copies_hold_only_their_fields() -> None:
    # The new compiler's request identity refuses a snapshot whose instance dictionary
    # holds anything but its fields, so the kept key lives beside the instances.
    names = {field.name for field in dataclasses.fields(CompileOptions)}
    first = CompileOptions.from_environment()
    second = CompileOptions.from_environment()
    fresh = CompileOptions._parse_environment()
    for snapshot in (first, second):
        assert set(vars(snapshot)) == names
        assert snapshot.cache_key == fresh.cache_key
        assert snapshot.cache_fingerprint == fresh.cache_fingerprint
    # By its id: collecting may also drop entries of other tests' copies.
    ident = id(second)
    assert ident in compile_options._kept_cache_keys
    del snapshot, second
    gc.collect()
    assert ident not in compile_options._kept_cache_keys
