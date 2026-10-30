"""The build snapshot is resolved once per distinct set of inputs, never stale.

``CompileOptions.from_environment`` runs on every public call.  Resolving its build
snapshot reads the kernel preamble's header files, lists Torch's library directory and
searches ``PATH``; hashing the preamble for the kernel cache key reads all of it again.
Both are kept per process.  These tests pin what keeps them correct:

* a changed header file, environment variable, working directory or compiler changes
  the snapshot and its cache key at the next call;
* a kept snapshot equals one resolved from scratch;
* every call receives its own instance, so one caller cannot change what the next
  receives;
* what is deliberately not read again (the contents of the ``PATH`` directories) is
  read after ``clear_build_snapshot_cache()``.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, replace
import hashlib
from importlib import resources
import io
import os
from pathlib import Path
import pickle

import pytest
import torch

import scorch
import scorch.utils as utils  # type: ignore[import-untyped]
from scorch import STensor  # type: ignore[import-untyped]
from scorch.compiler import compile_options  # type: ignore[import-untyped]
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
    CompilerWrapperPolicy,
    DarwinToolchainOptions,
    KernelBuildOptions,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    CompileOptionsError,
)

_HEADERS = ("header.h", "native_abi.h", "scorch_policy.h")
_MARKER = "// changed by test_build_snapshot_cache\n"


def _snapshot(environ: dict[str, str] | None = None) -> CompileOptions:
    return CompileOptions.from_environment(
        environ={} if environ is None else environ,
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )


@pytest.fixture(autouse=True)
def _nothing_kept_between_tests(
    fresh_kernel_caches: None, monkeypatch: pytest.MonkeyPatch
):
    # The build snapshots and the kernel caches start empty (fresh_kernel_caches); the
    # preamble memos, which these tests change the files of, do too.
    monkeypatch.setattr(utils, "_preamble_text_memo", None)
    monkeypatch.setattr(utils, "_preamble_paths_memo", None)


@pytest.fixture
def resolutions(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """The arguments of every host resolution made during the test, in order."""

    made: list[dict[str, object]] = []
    resolve = compile_options._resolve_kernel_build_options

    def recorded(**arguments: object) -> KernelBuildOptions:
        made.append(arguments)
        return resolve(**arguments)

    monkeypatch.setattr(compile_options, "_resolve_kernel_build_options", recorded)
    return made


@pytest.fixture
def headers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A private copy of the preamble's header files, which a test may change."""

    packaged = resources.files("scorch").joinpath("csrc")
    directory = tmp_path / "csrc"
    directory.mkdir()
    for name in _HEADERS:
        text = packaged.joinpath(name).read_text(encoding="utf-8")
        (directory / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(utils, "_NATIVE_RESOURCES", directory)
    return directory


def _append(path: Path) -> None:
    """Change the file's size."""

    with path.open("a", encoding="utf-8") as stream:
        stream.write(_MARKER)


def _other_last_byte(data: bytes) -> bytes:
    return data[:-1] + (b" " if data.endswith(b"\n") else b"\n")


def _rewrite_in_place(path: Path) -> None:
    """Change the content at the same size and inode; the modification time moves."""

    status = path.stat()
    path.write_bytes(_other_last_byte(path.read_bytes()))
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000))


def _replace_by_rename(path: Path) -> None:
    """Change the content at the same size and modification time; the inode moves."""

    status = path.stat()
    replacement = path.with_name(path.name + ".new")
    replacement.write_bytes(_other_last_byte(path.read_bytes()))
    os.utime(replacement, ns=(status.st_atime_ns, status.st_mtime_ns))
    os.replace(replacement, path)


def test_preamble_text_is_joined_once_while_its_files_are_unchanged(
    headers: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joins: list[str] = []
    read = utils._read_jit_preamble_text

    def recorded() -> str:
        joins.append("join")
        return read()

    monkeypatch.setattr(utils, "_read_jit_preamble_text", recorded)

    first = utils.jit_preamble_text()
    again = utils.jit_preamble_text()

    assert again is first
    assert len(joins) == 1
    assert first == read()
    assert utils.jit_preamble_file_identity() == utils.jit_preamble_file_identity()


@pytest.mark.parametrize("name", _HEADERS)
@pytest.mark.parametrize(
    "change",
    (_append, _rewrite_in_place, _replace_by_rename),
    ids=("size", "modification-time", "inode"),
)
def test_changed_header_changes_the_text_and_the_snapshot_key(
    headers: Path,
    name: str,
    change: Callable[[Path], None],
) -> None:
    text_before = utils.jit_preamble_text()
    identity_before = utils.jit_preamble_file_identity()
    before = _snapshot()

    change(headers / name)

    text_after = utils.jit_preamble_text()
    after = _snapshot()

    assert utils.jit_preamble_file_identity() != identity_before
    assert text_after != text_before
    assert text_after == utils._read_jit_preamble_text()
    assert after.build.preamble_source == text_after
    assert after.build_cache_key != before.build_cache_key
    assert after.cache_key != before.cache_key
    assert after.cache_fingerprint != before.cache_fingerprint
    # The earlier snapshot stays what it was when it was taken.
    assert before.build.preamble_source == text_before


def test_tuning_header_appearing_and_disappearing_changes_the_snapshot(
    headers: Path,
) -> None:
    without = _snapshot()
    tuning = headers / "scorch_policy_tuned.h"
    tuning.write_text("#define SCORCH_TEST_TUNING 1\n", encoding="utf-8")
    with_tuning = _snapshot()
    tuning.unlink()
    without_again = _snapshot()

    assert "SCORCH_TEST_TUNING" not in without.build.preamble_source
    assert "SCORCH_TEST_TUNING" in with_tuning.build.preamble_source
    assert with_tuning.build_cache_key != without.build_cache_key
    assert without_again.build == without.build
    assert without_again.cache_key == without.cache_key


def test_environment_variable_changed_in_process_changes_the_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    resolutions: list[dict[str, object]],
) -> None:
    monkeypatch.delenv("SCORCH_JIT_TUNE_HOOKS", raising=False)
    plain = CompileOptions.from_environment()
    plain_again = CompileOptions.from_environment()
    monkeypatch.setenv("SCORCH_JIT_TUNE_HOOKS", "1")
    hooked = CompileOptions.from_environment()
    monkeypatch.delenv("SCORCH_JIT_TUNE_HOOKS")
    plain_after = CompileOptions.from_environment()

    assert not plain.build.jit_tune_hooks
    assert hooked.build.jit_tune_hooks
    assert "-DSCORCH_TUNE_HOOKS" not in plain.build.extra_cflags
    assert "-DSCORCH_TUNE_HOOKS" in hooked.build.extra_cflags
    assert hooked.cache_key != plain.cache_key
    assert plain_again.cache_key == plain.cache_key
    assert plain_after.build == plain.build
    # One resolution for each of the two environments, none for the repeats.
    assert len(resolutions) == 2


def test_unchanged_second_call_opens_no_file(
    monkeypatch: pytest.MonkeyPatch,
    resolutions: list[dict[str, object]],
) -> None:
    first = CompileOptions.from_environment()
    opened: list[object] = []
    real_open = io.open

    def recorded(file: object, *arguments: object, **keywords: object) -> object:
        opened.append(file)
        return real_open(file, *arguments, **keywords)  # type: ignore[call-overload]

    monkeypatch.setattr(io, "open", recorded)
    monkeypatch.setattr(builtins, "open", recorded)

    second = CompileOptions.from_environment()
    second_key = second.cache_key

    assert opened == []
    assert len(resolutions) == 1
    assert second_key == first.cache_key

    # The recorder does see the reads of a call that has to resolve: three headers.
    compile_options.clear_build_snapshot_cache()
    monkeypatch.setattr(utils, "_preamble_text_memo", None)
    third = CompileOptions.from_environment()

    assert len(resolutions) == 2
    assert len(opened) >= len(_HEADERS)
    assert third.cache_key == first.cache_key


def test_compiler_wrapper_policy_is_part_of_the_key() -> None:
    automatic = _snapshot()
    disabled = _snapshot({"TORCH_NO_COMPILER_WRAPPER": "1"})
    automatic_again = _snapshot()

    assert automatic.build.compiler_wrapper_policy is CompilerWrapperPolicy.AUTO
    assert disabled.build.compiler_wrapper_policy is CompilerWrapperPolicy.DISABLED
    assert disabled.build_cache_key != automatic.build_cache_key
    assert automatic_again.build == automatic.build


def test_the_programs_a_compiler_path_reaches_are_the_key_not_the_path(
    tmp_path: Path,
) -> None:
    """Two directories holding a link to the same compiler, put in front of ``PATH``,
    resolve different compiler paths but reach the same programs, so they share the
    key (``build.toolchain_identity``); naming one through ``CXX`` changes the name the
    compiler is invoked as, which the key holds."""

    base = _snapshot()
    rest = base.build.executable_search_path
    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        directory.mkdir()
        (directory / "c++").symlink_to(base.build.cxx_compiler_path)

    found_in_first = _snapshot({"PATH": f"{first}{os.pathsep}{rest}"})
    found_in_second = _snapshot({"PATH": f"{second}{os.pathsep}{rest}"})
    named = _snapshot(
        {"CXX": str(second / "c++"), "PATH": f"{first}{os.pathsep}{rest}"}
    )

    assert found_in_first.build.cxx_compiler_path == str(first / "c++")
    assert found_in_second.build.cxx_compiler_path == str(second / "c++")
    assert named.build.cxx_compiler_path == str(second / "c++")
    assert named.build.cxx_compiler_from_environment
    assert not found_in_second.build.cxx_compiler_from_environment
    assert (
        found_in_first.build_cache_key
        == found_in_second.build_cache_key
        == base.build_cache_key
    )
    assert named.build_cache_key != base.build_cache_key


def test_working_directory_is_part_of_the_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = _snapshot()
    environ = {"CXX": "./relative-cxx", "PATH": base.build.executable_search_path}
    resolved: list[str] = []
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "relative-cxx").symlink_to(base.build.cxx_compiler_path)
        monkeypatch.chdir(directory)
        resolved.append(_snapshot(environ).build.cxx_compiler_path)

    assert resolved == [
        str(tmp_path / "first" / "relative-cxx"),
        str(tmp_path / "second" / "relative-cxx"),
    ]


@pytest.mark.parametrize(
    "environ",
    (
        {},
        {"SCORCH_JIT_TUNE_HOOKS": "1"},
        {"TORCH_NO_COMPILER_WRAPPER": "1"},
        {"SCORCH_REGBLOCK": "1", "SCORCH_REGBLOCK_T": "4"},
    ),
    ids=("default", "tune-hooks", "no-wrapper", "register-blocking"),
)
def test_kept_snapshot_equals_a_fresh_resolution(
    environ: dict[str, str],
    resolutions: list[dict[str, object]],
) -> None:
    resolved = _snapshot(environ)
    kept = _snapshot(environ)
    compile_options.clear_build_snapshot_cache()
    resolved_again = _snapshot(environ)

    assert len(resolutions) == 2
    assert kept == resolved == resolved_again
    assert kept.cache_key == resolved.cache_key == resolved_again.cache_key
    assert kept.cache_fingerprint == resolved_again.cache_fingerprint
    fresh = compile_options._resolve_kernel_build_options(**resolutions[0])
    assert kept.build == fresh
    assert kept.build.cache_key == fresh.cache_key
    for field in fields(KernelBuildOptions):
        assert type(getattr(kept.build, field.name)) is type(getattr(fresh, field.name))


def test_every_call_receives_its_own_instance() -> None:
    first = _snapshot()
    second = _snapshot()

    assert first.build is not second.build
    assert first.build == second.build
    assert type(second.build) is KernelBuildOptions
    with pytest.raises(FrozenInstanceError):
        second.build.target_arch = "changed"  # type: ignore[misc]
    if first.build.darwin_toolchain is not None:
        assert first.build.darwin_toolchain is not second.build.darwin_toolchain
        assert first.build.darwin_toolchain == second.build.darwin_toolchain

    # A write past the frozen dataclass stays in the object it was made on.
    object.__setattr__(first.build, "extra_cflags", ["-DFORGED"])
    object.__setattr__(first.build, "preamble_source", "forged")
    third = _snapshot()

    assert third.build == second.build
    assert type(third.build.extra_cflags) is tuple
    assert "-DFORGED" not in third.build.extra_cflags


def test_own_copy_also_copies_a_darwin_toolchain() -> None:
    # Built without the constructors so that the Darwin branch runs on every host.
    toolchain = object.__new__(DarwinToolchainOptions)
    toolchain.__dict__.update(
        developer_dir="/developer", sdk_root="/developer/sdk", deployment_target=None
    )
    kept = object.__new__(KernelBuildOptions)
    kept.__dict__.update(_snapshot().build.__dict__)
    kept.__dict__["darwin_toolchain"] = toolchain

    own = compile_options._own_copy_of_build_options(kept)

    assert type(own) is KernelBuildOptions
    assert own is not kept
    assert own == kept
    assert type(own.darwin_toolchain) is DarwinToolchainOptions
    assert own.darwin_toolchain is not toolchain
    assert own.darwin_toolchain == toolchain
    assert kept.darwin_toolchain is toolchain


def test_returned_snapshot_pickles_and_validates_like_a_constructed_one() -> None:
    build = _snapshot().build

    assert pickle.loads(pickle.dumps(build)) == build
    # ``replace`` runs the constructor's validation on the copied fields.
    assert replace(build) == build


def test_preamble_digest_is_computed_once_per_text() -> None:
    build = _snapshot().build
    digest = hashlib.sha256(build.preamble_source.encode()).hexdigest()
    compile_options._preamble_digest.cache_clear()

    keys = [build.cache_key for _ in range(3)] + [_snapshot().build.cache_key]
    counts = compile_options._preamble_digest.cache_info()

    assert all(key == keys[0] for key in keys)
    assert keys[0][-1] == digest
    assert counts.misses == 1
    assert counts.hits >= 3

    other = replace(build, preamble_source=build.preamble_source + _MARKER)
    other_digest = hashlib.sha256(other.preamble_source.encode()).hexdigest()
    assert other.cache_key[-1] == other_digest
    assert other.cache_key != build.cache_key


def test_forged_preamble_is_not_looked_up_in_the_digest_cache() -> None:
    class Hostile(str):
        def __hash__(self) -> int:
            raise AssertionError("a forged preamble was used as a cache key")

    build = _snapshot().build
    object.__setattr__(build, "preamble_source", Hostile("forged"))

    assert build.cache_key[-1] == hashlib.sha256(b"forged").hexdigest()


def test_failed_resolution_is_not_kept(
    tmp_path: Path,
    resolutions: list[dict[str, object]],
) -> None:
    base = _snapshot()
    compiler = tmp_path / "missing-compiler"
    environ = {"CXX": str(compiler), "PATH": str(tmp_path)}

    for _ in range(2):
        with pytest.raises(CompileOptionsError) as error:
            _snapshot(environ)
        assert error.value.diagnostics[0].code == "compiler_not_found"
    assert len(resolutions) == 3

    compiler.symlink_to(base.build.cxx_compiler_path)
    assert _snapshot(environ).build.cxx_compiler_path == str(compiler)


def test_path_directory_contents_are_read_again_after_clearing(
    tmp_path: Path,
) -> None:
    base = _snapshot()
    (tmp_path / "c++").symlink_to(base.build.cxx_compiler_path)
    environ = {"PATH": str(tmp_path)}

    before = _snapshot(environ)
    (tmp_path / "ccache").symlink_to(base.build.cxx_compiler_path)
    same_key = _snapshot(environ)
    compile_options.clear_build_snapshot_cache()
    after_clearing = _snapshot(environ)

    assert before.build.compiler_wrapper_name is None
    # The documented limit: a file that appears in a directory already on PATH is
    # not part of the key.
    assert same_key.build.compiler_wrapper_name is None
    assert after_clearing.build.compiler_wrapper_name == "ccache"
    assert after_clearing.build.compiler_wrapper_path == str(tmp_path / "ccache")


def test_cache_is_bounded(tmp_path: Path) -> None:
    base = _snapshot()
    limit = compile_options._BUILD_SNAPSHOT_CACHE_LIMIT

    for number in range(limit + 3):
        directory = tmp_path / f"path{number}"
        directory.mkdir()
        (directory / "c++").symlink_to(base.build.cxx_compiler_path)
        build = _snapshot({"PATH": str(directory)}).build
        assert build.cxx_compiler_path == str(directory / "c++")
        assert len(compile_options._build_snapshot_cache) <= limit


class _ReadableButNotAPath:
    """A resource directory that can be read but has no file-system path."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def joinpath(self, name: str) -> "_ReadableButNotAPath":
        return _ReadableButNotAPath(self._path / name)

    def is_file(self) -> bool:
        return self._path.is_file()

    def read_text(self, encoding: str) -> str:
        return self._path.read_text(encoding=encoding)


def test_without_file_paths_every_call_reads_and_resolves(
    headers: Path,
    monkeypatch: pytest.MonkeyPatch,
    resolutions: list[dict[str, object]],
) -> None:
    expected = utils.jit_preamble_text()
    monkeypatch.setattr(utils, "_NATIVE_RESOURCES", _ReadableButNotAPath(headers))

    first_text = utils.jit_preamble_text()
    second_text = utils.jit_preamble_text()
    first = _snapshot()
    _append(headers / "native_abi.h")
    second = _snapshot()

    assert utils.jit_preamble_file_identity() is None
    assert first_text == second_text == expected
    assert first_text is not second_text
    assert len(resolutions) == 2
    assert not compile_options._build_snapshot_cache
    assert first.build.preamble_source == expected
    assert second.build.preamble_source != expected
    assert second.cache_key != first.cache_key


@pytest.mark.parametrize("rows", (5, 300))
def test_repeated_public_calls_resolve_the_host_once(
    rows: int,
    resolutions: list[dict[str, object]],
) -> None:
    generator = torch.Generator().manual_seed(rows)
    dense = torch.rand(rows, rows, generator=generator, dtype=torch.float64)
    dense[dense < 0.7] = 0.0
    dense[rows // 2] = 0.0  # one row stores nothing
    stored = dense.nonzero()
    matrix = STensor.from_components(
        (rows, rows),
        "oo",
        [[stored[:, 0].to(torch.int32)], [stored[:, 1].to(torch.int32)]],
        dense[stored[:, 0], stored[:, 1]],
        name="A",
    )
    x = torch.rand(rows, generator=generator, dtype=torch.float64)
    vector = STensor.from_torch(x, "x")
    expected = dense @ x

    for _ in range(3):
        result = scorch.einsum("ij,j->i", matrix, vector, format="d")
        got = result.to_torch() if isinstance(result, STensor) else result
        assert torch.allclose(got, expected, rtol=1e-12, atol=1e-12)

    assert len(resolutions) == 1
