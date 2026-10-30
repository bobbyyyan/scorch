"""The JIT build child's environment, in a sanitizer run and outside one.

A kernel builds in a child process (``scorch.utils._build_and_load_extension``) whose
environment is closed: ``KernelBuildOptions.build_child_environment``, which the
toolchain identity keys on. Under tests/sanitizers/run_sanitizers.sh the child loads
the ASan-built scorch_ops and the instrumented kernel it builds, so it needs the
parent's sanitizer runtime: ``LD_PRELOAD`` (``DYLD_INSERT_LIBRARIES`` on macOS) and the
``*SAN_OPTIONS``. The parent passes them only when its own process has the runtime
loaded, and the child deletes them before it starts the compiler, so the toolchain
always runs in the closed environment.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

import pytest

import scorch.utils as utils  # type: ignore[import-untyped]
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)


def _options() -> CompileOptions:
    return CompileOptions.from_environment(
        forced_schedule=None, regblock_override=None, verify_cin_override=False
    )


def _request(tmp_path: Path, build, name: str) -> utils._JITBuildRequest:
    return utils._JITBuildRequest(
        name=name,
        cpp_sources=("int evaluate() { return 0; }",),
        functions=("evaluate",),
        extra_cflags=build.extra_cflags,
        extra_ldflags=build.extra_ldflags,
        build_directory=str(tmp_path / "build"),
        build_options=build,
    )


def _child_environment(
    request: utils._JITBuildRequest,
) -> tuple[dict[str, str], dict[str, str]]:
    """``build_child_environment`` and the environment the child is started with."""
    toolchain_directory = utils._prepare_compiler_toolchain(request)
    closed = request.build_options.build_child_environment(toolchain_directory)
    child = utils._jit_build_environment_from_request(
        request, toolchain_directory=toolchain_directory
    )
    return closed, child


def test_outside_a_sanitizer_run_the_child_gets_the_closed_environment(
    tmp_path, monkeypatch
):
    # Every runtime variable set in this process, as a user's LD_PRELOAD of an
    # allocator would be: none reaches the child.
    for name in utils._SANITIZER_RUNTIME_VARIABLES:
        monkeypatch.setenv(name, f"planted-{name}")
    monkeypatch.setattr(utils, "_sanitizer_runtime_loaded", lambda: False)
    closed, child = _child_environment(
        _request(tmp_path, _options().build, "jit_child_environment_closed")
    )
    assert child == closed
    assert not set(utils._SANITIZER_RUNTIME_VARIABLES) & set(child)


@pytest.mark.parametrize(
    "parent_asan, child_asan",
    [
        # A leak pass's parent: the child's own setting comes last, so it wins.
        (
            "halt_on_error=1:detect_leaks=1",
            "halt_on_error=1:detect_leaks=1:detect_leaks=0",
        ),
        # No ASAN_OPTIONS in the parent (ASan's Linux default detects leaks).
        (None, "detect_leaks=0"),
    ],
)
def test_in_a_sanitizer_run_the_child_also_gets_the_runtime_variables_set(
    tmp_path, monkeypatch, parent_asan, child_asan
):
    for name in utils._SANITIZER_RUNTIME_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    planted = {
        "LD_PRELOAD": "/planted/libasan.so",
        "LSAN_OPTIONS": "leak_check_at_exit=1",
        "UBSAN_OPTIONS": "print_stacktrace=1",
    }
    if parent_asan is not None:
        planted["ASAN_OPTIONS"] = parent_asan
    for name, value in planted.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(utils, "_sanitizer_runtime_loaded", lambda: True)
    closed, child = _child_environment(
        _request(tmp_path, _options().build, "jit_child_environment_sanitized")
    )
    # Exactly the ones set, an unset DYLD_INSERT_LIBRARIES not added, and the child
    # never running LeakSanitizer.
    assert child == {**closed, **planted, "ASAN_OPTIONS": child_asan}


def test_the_runtime_is_found_only_where_it_was_preloaded(monkeypatch):
    # This process has the ASan runtime only when the harness preloaded it.
    preloaded = any(
        "asan" in os.environ.get(name, "")
        for name in ("LD_PRELOAD", "DYLD_INSERT_LIBRARIES")
    )
    monkeypatch.setattr(utils, "_SANITIZER_RUNTIME_LOADED", None)
    assert utils._sanitizer_runtime_loaded() is preloaded


def test_the_child_deletes_the_runtime_variables_before_it_starts_the_compiler(
    tmp_path, monkeypatch
):
    # The test plants its own LD_PRELOAD and options for the child, so under a real
    # ASan parent the child would start without libasan and die. Asked before any
    # monkeypatch, of the real lookup.
    if utils._sanitizer_runtime_loaded():
        pytest.skip("this process has the ASan runtime, which the planted ones replace")
    # The compiler is a wrapper that records its arguments and its whole environment
    # on every invocation, then runs the real compiler. Its paths are written into its
    # text, since no variable reaches it.
    record = tmp_path / "compiler_environment.txt"
    wrapper = tmp_path / "cxx"
    real = _options().build.cxx_compiler_path
    wrapper.write_text(
        "#!/bin/sh\n"
        f"{{ printf '== %s\\n' \"$*\"; /usr/bin/env; }} >> {shlex.quote(str(record))}\n"
        f'exec {shlex.quote(real)} "$@"\n'
    )
    wrapper.chmod(0o755)
    monkeypatch.setenv("CXX", str(wrapper))
    build = _options().build
    assert os.path.samefile(build.cxx_compiler_path, wrapper)
    record.write_text("")  # the toolchain probe ran the wrapper; only the build counts

    # LD_PRELOAD of a library every process already has; on macOS no library is
    # inserted, and the options alone are planted.
    planted = {
        "ASAN_OPTIONS": "verbosity=0",
        "LSAN_OPTIONS": "verbosity=0",
        "UBSAN_OPTIONS": "verbosity=0",
    }
    if sys.platform.startswith("linux"):
        planted["LD_PRELOAD"] = "libc.so.6"
    for name in utils._SANITIZER_RUNTIME_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name, value in planted.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(utils, "_sanitizer_runtime_loaded", lambda: True)
    passed: list[dict[str, str]] = []
    environment_for = utils._jit_build_environment_from_request

    def recorded(request, *, toolchain_directory):
        environment = environment_for(request, toolchain_directory=toolchain_directory)
        passed.append(environment)
        return environment

    monkeypatch.setattr(utils, "_jit_build_environment_from_request", recorded)

    utils._build_and_load_extension(
        _request(tmp_path, build, "jit_child_environment_deleted")
    )

    # The child got them (ASAN_OPTIONS with detect_leaks=0 added) ...
    assert len(passed) == 1
    assert set(planted) <= set(passed[0])
    # ... and the compiler, which compiled the kernel's main.cpp, saw none of them.
    invocations = record.read_text().split("== ")[1:]
    assert any("main.cpp" in invocation.splitlines()[0] for invocation in invocations)
    for invocation in invocations:
        names = set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)=", invocation, re.M))
        seen = names & set(utils._SANITIZER_RUNTIME_VARIABLES)
        assert not seen, (invocation.splitlines()[0], seen)
