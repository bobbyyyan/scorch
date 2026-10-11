"""A failed JIT build raises what an in-process ``load_inline`` would raise.

Kernels build in a child process (``scorch.utils._build_and_load_extension``).
Before this fix a failed build surfaced as ``subprocess.CalledProcessError``
whose message named only the child's command line: the compiler log went to the
terminal, not into the exception, and code that caught torch's ``RuntimeError``
missed it.  The child now reports the exception that stopped it, and the parent
re-raises its type and message.  None of these tests runs a compiler.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

import scorch.utils as utils  # type: ignore[import-untyped]
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)


def _request(tmp_path: Path, **build_changes: object) -> utils._JITBuildRequest:
    options = CompileOptions.from_environment(
        forced_schedule=None, regblock_override=None, verify_cin_override=False
    )
    build = dataclasses.replace(options.build, **build_changes)
    return utils._JITBuildRequest(
        name="jit_build_error_test",
        cpp_sources=("int evaluate() { return 0; }",),
        functions=("evaluate",),
        extra_cflags=build.extra_cflags,
        extra_ldflags=build.extra_ldflags,
        build_directory=str(tmp_path / "build"),
        build_options=build,
    )


def _completed(returncode: int, stderr: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(["child"], returncode, None, stderr)


def _report(error_type: str, message: str) -> bytes:
    payload = json.dumps({"type": error_type, "message": message})
    return (utils._JIT_BUILD_ERROR_MARKER + payload).encode()


def test_a_reported_build_error_is_raised_with_its_type_and_message(capsys):
    message = "Error building extension 'k': main.cpp:3:5: error: use of undeclared identifier 'k'"
    completed = _completed(
        1, b"a warning the child printed\n" + _report("RuntimeError", message)
    )
    with pytest.raises(RuntimeError) as raised:
        utils._raise_for_jit_build_child("k", completed)
    assert type(raised.value) is RuntimeError
    assert str(raised.value) == message
    assert capsys.readouterr().err == "a warning the child printed\n"


@pytest.mark.parametrize("error_type", sorted(utils._JIT_BUILD_CHILD_ERRORS))
def test_every_forwarded_type_is_raised_as_itself(error_type):
    with pytest.raises(utils._JIT_BUILD_CHILD_ERRORS[error_type]) as raised:
        utils._raise_for_jit_build_child("k", _completed(1, _report(error_type, "m")))
    assert type(raised.value).__name__ == error_type
    assert str(raised.value) == "m"


def test_an_unlisted_type_becomes_a_runtime_error_with_the_same_message():
    with pytest.raises(RuntimeError, match="^m$") as raised:
        utils._raise_for_jit_build_child("k", _completed(1, _report("KeyError", "m")))
    assert type(raised.value) is RuntimeError


def test_a_child_that_dies_without_a_report_still_raises_with_its_stderr(capsys):
    with pytest.raises(RuntimeError, match="status -9") as raised:
        utils._raise_for_jit_build_child("k", _completed(-9, b"Killed\n"))
    assert "Killed" in str(raised.value)
    assert capsys.readouterr().err == "Killed\n"


def test_a_successful_build_forwards_its_stderr_and_raises_nothing(capsys):
    utils._raise_for_jit_build_child("k", _completed(0, b"note\n"))
    assert capsys.readouterr().err == "note\n"


def test_the_real_child_reports_a_runtime_snapshot_mismatch(tmp_path):
    # The child checks its runtime against the request's snapshot before it
    # compiles, so a wrong torch version stops it with no compiler invocation.
    request = _request(tmp_path, torch_version="0.0.0+not-this-torch")
    with pytest.raises(RuntimeError) as raised:
        utils._build_and_load_extension(request)
    assert type(raised.value) is RuntimeError
    assert str(raised.value) == (
        "JIT build runtime differs from CompileOptions snapshot: torch_version"
    )
    assert not (Path(request.build_directory) / f"{request.name}.so").exists()


def test_the_child_reports_through_the_marker(tmp_path):
    # Runs the child entry point exactly as the parent does and reads its raw
    # stderr: the report is the last thing written, after the marker.
    request = _request(tmp_path, torch_version="0.0.0+not-this-torch")
    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            "from scorch.utils import _run_jit_build_request; _run_jit_build_request()",
        ],
        input=__import__("pickle").dumps(request),
        env=utils._jit_build_environment_from_request(
            request, toolchain_directory=utils._prepare_compiler_toolchain(request)
        ),
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 1
    _, marker, payload = completed.stderr.decode().partition(
        utils._JIT_BUILD_ERROR_MARKER
    )
    assert marker
    assert json.loads(payload) == {
        "type": "RuntimeError",
        "message": "JIT build runtime differs from CompileOptions snapshot: torch_version",
    }
