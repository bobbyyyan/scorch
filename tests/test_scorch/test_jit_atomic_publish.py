"""A kernel's ``.so`` appears on its final path whole, or not at all.

Many processes can share one kernel directory (``TORCH_EXTENSIONS_DIR``): DataLoader
workers, torchrun ranks on one node, a parallel compile step.  ``_load_validated_prepared_kernel``
imports any ``.so`` it finds on the final path without a lock, so before patch 0064 a
process could import a ``.so`` that another process was still linking and fail with
``ImportError: ... file too short`` (4 of 2,824 entries in one 48-process run on mkt1).
The build now runs in a private directory and the finished ``.so`` is moved onto its
final path with ``os.replace``.

Every test here runs its builders and readers as separate Python processes.  A stub
stands in for the build child: it writes a real kernel's bytes in two halves to the
directory the parent gave it, and waits between them, so the window in which the file
is half written is as long as the test needs.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import scorch  # type: ignore[import-untyped]

NAME = "kernel_atomic_publish_test"
SOURCE = "int evaluate() { return 7; }"
TIMEOUT = 600

# A kernel that does not compile: a failed build's files and error (patch 0064b).
BROKEN_NAME = "kernel_atomic_publish_broken"
BROKEN_SOURCE = "int evaluate() { return name_no_scope_declares; }"


def _prelude(name: str, source: str) -> str:
    """Every process prepares the same request; the build identity holds the working
    directory, so all of a test's processes run in the same one."""

    return textwrap.dedent(f"""
        import json, os, pickle, subprocess, sys, time
        import scorch.utils as utils

        options = utils._resolve_compile_options(None)
        prepared = utils._prepare_jit_build(
            {name!r}, ({source!r},), ("evaluate",),
            options.build.extra_cflags, options.build.extra_ldflags,
        )
        """)


PRELUDE = _prelude(NAME, SOURCE)

# Replaces the build child in this process: writes the payload's first half to
# <the directory the parent named>/<name>.so, creates the marker "half", waits for the
# marker "go" (forever when there is none), then writes the rest.
STUB = textwrap.dedent("""
    completed_process = subprocess.CompletedProcess
    payload = open(os.environ["PUBLISH_PAYLOAD"], "rb").read()
    markers = os.environ["PUBLISH_MARKERS"]

    def stub_run(command, *, input, env, stderr, check):
        request = pickle.loads(input)
        target = os.path.join(request.build_directory, request.name + ".so")
        with open(target, "wb") as out:
            out.write(payload[: len(payload) // 2])
            out.flush()
            os.fsync(out.fileno())
            open(os.path.join(markers, "half"), "w").close()
            while not os.path.exists(os.path.join(markers, "go")):
                time.sleep(0.01)
            out.write(payload[len(payload) // 2 :])
        return completed_process(command, 0, None, b"")

    subprocess.run = stub_run
    """)

LOAD = textwrap.dedent("""
    try:
        module = utils._load_validated_prepared_kernel(prepared)
        result = {"value": module.evaluate()}
    except Exception as error:
        result = {"error": type(error).__name__ + ": " + str(error)[:300]}
    print(json.dumps(result))
    """)

# Waits for the builder's first half, looks at the final path, and imports from it if
# a file is there (as the lock-free check in _load_validated_prepared_kernel would);
# then lets the builder finish.  With nothing there it calls the loader, which waits
# for the builder to publish.
READER = textwrap.dedent("""
    markers = os.environ["PUBLISH_MARKERS"]
    while not os.path.exists(os.path.join(markers, "half")):
        time.sleep(0.01)
    seen = os.path.isfile(prepared.so_path)
    result = {"seen_while_half_written": seen}
    if not seen:
        open(os.path.join(markers, "go"), "w").close()
    try:
        module = utils._load_validated_prepared_kernel(prepared)
        result["value"] = module.evaluate()
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)[:300]
    open(os.path.join(markers, "go"), "w").close()
    print(json.dumps(result))
    """)


def _environment(extensions: Path, **extra: str) -> dict:
    environment = dict(os.environ)
    tree = str(Path(scorch.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        [tree] + [p for p in environment.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    environment["TORCH_EXTENSIONS_DIR"] = str(extensions)
    environment.update(extra)
    return environment


def _start(
    body: str, cwd: Path, environment: dict, prelude: str = PRELUDE
) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", prelude + body],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _result(process: subprocess.Popen) -> dict:
    out, err = process.communicate(timeout=TIMEOUT)
    assert process.returncode == 0, err[-3000:]
    return json.loads(out.strip().splitlines()[-1])


def _final_path(cwd: Path, environment: dict, prelude: str = PRELUDE) -> Path:
    shown = subprocess.run(
        [sys.executable, "-c", prelude + "print(prepared.so_path)"],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=True,
    )
    return Path(shown.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def payload(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real build of the test kernel, whose bytes the stubbed builders write."""

    root = tmp_path_factory.mktemp("publish_payload")
    environment = _environment(root / "extensions")
    assert _result(_start(LOAD, root, environment)) == {"value": 7}
    built = _final_path(root, environment)
    copy = root / "payload.so"
    copy.write_bytes(built.read_bytes())
    return copy


def _stubbed(tmp_path: Path, payload: Path) -> dict:
    markers = tmp_path / "markers"
    markers.mkdir()
    return _environment(
        tmp_path / "extensions",
        PUBLISH_PAYLOAD=str(payload),
        PUBLISH_MARKERS=str(markers),
    )


def test_a_reader_never_imports_a_kernel_still_being_written(tmp_path, payload):
    environment = _stubbed(tmp_path, payload)
    builder = _start(STUB + LOAD, tmp_path, environment)
    reader = _start(READER, tmp_path, environment)
    seen = _result(reader)
    built = _result(builder)
    assert seen == {"seen_while_half_written": False, "value": 7}
    assert built == {"value": 7}
    final = _final_path(tmp_path, environment)
    assert final.read_bytes() == payload.read_bytes()


def test_a_builder_killed_mid_write_leaves_nothing_on_the_final_path(tmp_path, payload):
    environment = _stubbed(tmp_path, payload)
    builder = _start(STUB + LOAD, tmp_path, environment)
    half = Path(environment["PUBLISH_MARKERS"]) / "half"
    deadline = time.monotonic() + TIMEOUT
    while not half.exists():
        assert builder.poll() is None, builder.communicate()[1][-3000:]
        assert time.monotonic() < deadline
        time.sleep(0.01)
    builder.send_signal(signal.SIGKILL)
    builder.communicate(timeout=TIMEOUT)
    final = _final_path(tmp_path, environment)
    assert not final.exists()
    # The next call, a real build, succeeds and clears the killed builder's directory.
    assert _result(_start(LOAD, tmp_path, _environment(tmp_path / "extensions"))) == {
        "value": 7
    }
    assert final.read_bytes()[:4] == payload.read_bytes()[:4]
    assert [
        p.name for p in final.parent.iterdir() if p.name.startswith(".scorch-staging-")
    ] == []


def test_leftover_build_directories_and_a_stale_torch_lock_do_not_stop_a_build(
    tmp_path,
):
    environment = _environment(tmp_path / "extensions")
    final = _final_path(tmp_path, environment)
    final.parent.mkdir(parents=True)
    for index in range(3):
        leftover = final.parent / f".scorch-staging-left{index}"
        leftover.mkdir()
        (leftover / final.name).write_bytes(b"\x7fELF half")
        (leftover / "main.cpp").write_text("not this kernel")
        (leftover / "lock").write_text("")
    # torch's own build lock, left by a build killed under the code before 0064.
    (final.parent / "lock").write_text("")
    assert _result(_start(LOAD, tmp_path, environment)) == {"value": 7}
    assert final.is_file()
    assert SOURCE in (final.parent / "main.cpp").read_text()
    # The build moved its sources beside the .so, as an in-place build left them.
    assert (final.parent / "build.ninja").is_file()


# ------------------------------------------------------------------------------------
# Patch 0064b: a kernel built once is imported, not rebuilt, by every later process; a
# failed build keeps its files; a lock holder always builds in .scorch-staging-build.
# ------------------------------------------------------------------------------------

STAGING = ".scorch-staging-build"

# Prints the error a failed build raises, in full.
FAIL_LOAD = textwrap.dedent("""
    try:
        utils._load_validated_prepared_kernel(prepared)
        result = {"built": True}
    except Exception as error:
        result = {
            "type": type(error).__name__,
            "code": getattr(error, "code", None),
            "message": str(error),
        }
    print(json.dumps(result))
    """)

# Replaces the build child in this process with one that reports an error, as a compiler
# error would, and exits with CHILD_STATUS (1 when unset). With CHILD_WRITES set it first
# writes a whole kernel (the payload) where the parent would publish from.
FAILING_CHILD = textwrap.dedent("""
    completed_process = subprocess.CompletedProcess

    def failing_run(command, *, input, env, stderr, check):
        request = pickle.loads(input)
        if os.environ.get("CHILD_WRITES"):
            target = os.path.join(request.build_directory, request.name + ".so")
            with open(target, "wb") as out:
                out.write(open(os.environ["PUBLISH_PAYLOAD"], "rb").read())
        report = json.dumps({"type": "RuntimeError", "message": "stub build failed"})
        status = int(os.environ.get("CHILD_STATUS", "1"))
        return completed_process(
            command, status, None, (utils._JIT_BUILD_ERROR_MARKER + report).encode()
        )

    subprocess.run = failing_run
    """)


def _kept_directory(message: str) -> Path:
    """The directory a failed build's error says keeps its files."""

    kept = re.search(r"files are kept in (\S+?)(?: \(main\.cpp also in |; )", message)
    assert kept, message[-600:]
    return Path(kept.group(1))


def _logging_compiler(tmp_path: Path) -> tuple[Path, Path]:
    """A C++ compiler that logs each call and runs the one this environment uses.

    The scorch environment sets ``CXX`` to an absolute path, which the build snapshot
    takes over ``PATH``, so the wrapper is passed as ``CXX`` to every process.
    """

    real = os.environ.get("CXX") or shutil.which("c++")
    assert real, "no C++ compiler to wrap"
    log = tmp_path / "compiler_calls.log"
    log.write_text("")
    wrapper = tmp_path / "logging-cxx"
    wrapper.write_text(f'#!/bin/sh\necho "$@" >> {log}\nexec {real} "$@"\n')
    wrapper.chmod(0o755)
    return wrapper, log


def _calls(log: Path) -> int:
    return len(log.read_text().splitlines())


def _identity(path: Path) -> tuple[int, int]:
    status = path.stat()
    return status.st_ino, status.st_mtime_ns


def test_a_published_kernel_is_imported_by_later_processes_without_compiling(tmp_path):
    wrapper, log = _logging_compiler(tmp_path)
    environment = _environment(tmp_path / "extensions", CXX=str(wrapper))
    assert _result(_start(LOAD, tmp_path, environment)) == {"value": 7}
    built = _calls(log)
    assert built >= 2, log.read_text()  # positive control: a compile and a link
    final = _final_path(tmp_path, environment)
    published = _identity(final)
    build = final.parent

    def again(step: str) -> None:
        before = _calls(log)
        assert _result(_start(LOAD, tmp_path, environment)) == {"value": 7}, step
        assert _calls(log) == before, f"{step}: {log.read_text()}"
        assert _identity(final) == published, step

    again("a second process")
    for leftover in build.glob(".scorch-staging*"):
        shutil.rmtree(leftover)
    again("after the staging directories were removed")
    (build / "build.ninja").rename(build / "build.ninja.moved")
    again("without build.ninja")
    (build / "build.ninja").write_text("rule cxx\n  command = /bin/false\n")
    again("with a build.ninja that cannot build")
    # Positive control: without the .so, the next process builds the kernel again.
    final.unlink()
    assert _result(_start(LOAD, tmp_path, environment)) == {"value": 7}
    assert _calls(log) > built, log.read_text()


def test_every_build_of_a_kernel_writes_the_same_build_ninja(tmp_path):
    environment = _environment(tmp_path / "extensions")
    texts = []
    for _ in range(2):
        assert _result(_start(LOAD, tmp_path, environment)) == {"value": 7}
        build = _final_path(tmp_path, environment).parent
        texts.append((build / "build.ninja").read_bytes())
        shutil.rmtree(build)
    assert texts[0] == texts[1]
    assert f"/{STAGING}/".encode() in texts[0]


def test_a_failed_build_keeps_its_files_and_its_error_names_them(tmp_path):
    broken = _prelude(BROKEN_NAME, BROKEN_SOURCE)
    environment = _environment(tmp_path / "extensions")
    final = _final_path(tmp_path, environment, broken)
    for attempt in range(2):  # a second failure replaces the first one's files
        failed = _result(_start(FAIL_LOAD, tmp_path, environment, broken))
        assert failed["type"] == "ScorchRuntimeError", failed
        assert failed["code"] == "jit_build_failed", failed
        kept = final.parent / STAGING
        assert _kept_directory(failed["message"]) == kept
        assert f"(main.cpp also in {final.parent});" in failed["message"]
        assert BROKEN_SOURCE in (kept / "main.cpp").read_text(), attempt
        assert BROKEN_SOURCE in (final.parent / "main.cpp").read_text(), attempt
        # Every main.cpp the compiler's messages name still exists.
        named = set(re.findall(r"(/[^\s'\":]*main\.cpp)", failed["message"]))
        assert named and all(Path(path).is_file() for path in named), named
        assert not final.exists()
        assert not (final.parent / f"{BROKEN_NAME}.so").exists()


def test_a_failed_build_without_the_lock_keeps_its_own_directory(tmp_path):
    unlocked = "utils._lock_jit_build_directory = lambda directory: (None, False)\n"
    broken = _prelude(BROKEN_NAME, BROKEN_SOURCE)
    environment = _environment(tmp_path / "extensions")
    failed = _result(_start(unlocked + FAIL_LOAD, tmp_path, environment, broken))
    kept_path = _kept_directory(failed["message"])
    assert kept_path.name.startswith(".scorch-staging-") and kept_path.name != STAGING
    assert BROKEN_SOURCE in (kept_path / "main.cpp").read_text()


def test_a_stale_staging_directory_is_never_published(tmp_path, payload):
    environment = _environment(tmp_path / "extensions")
    final = _final_path(tmp_path, environment)
    stale = final.parent / STAGING
    stale.mkdir(parents=True)
    (stale / final.name).write_bytes(payload.read_bytes())  # a whole, working kernel
    (stale / "main.cpp").write_text("not this build")
    failed = _result(_start(FAILING_CHILD + FAIL_LOAD, tmp_path, environment))
    assert failed.get("code") == "jit_build_failed", failed
    assert not final.exists()
    assert not (final.parent / "main.cpp").exists()
    assert not (stale / final.name).exists() and not (stale / "main.cpp").exists()
    assert _result(_start(LOAD, tmp_path, environment)) == {"value": 7}
    assert SOURCE in (final.parent / "main.cpp").read_text()
    assert [p.name for p in final.parent.glob(".scorch-staging*")] == []


def test_a_build_whose_child_reported_an_error_is_never_published(tmp_path, payload):
    # The build child exits 1 after reporting an error; a child that reported one and
    # exited 0 anyway is still a failed build, so its kernel is never published.
    environment = _environment(
        tmp_path / "extensions",
        CHILD_STATUS="0",
        CHILD_WRITES="1",
        PUBLISH_PAYLOAD=str(payload),
    )
    final = _final_path(tmp_path, environment)
    failed = _result(_start(FAILING_CHILD + FAIL_LOAD, tmp_path, environment))
    assert failed.get("code") == "jit_build_failed", failed
    assert _kept_directory(failed["message"]) == final.parent / STAGING
    assert "main.cpp also in" not in failed["message"]  # it left only the .so
    assert not final.exists()
    assert (final.parent / STAGING / final.name).is_file()  # kept, never published


def test_a_failed_build_that_left_no_file_keeps_no_directory(tmp_path):
    # A child that stops before writing anything (as on a runtime-snapshot mismatch):
    # nothing to keep, so the error is the child's own and no directory stays.
    environment = _environment(tmp_path / "extensions")
    final = _final_path(tmp_path, environment)
    failed = _result(_start(FAILING_CHILD + FAIL_LOAD, tmp_path, environment))
    assert failed == {
        "type": "ScorchRuntimeError",
        "code": "jit_build_failed",
        "message": "stub build failed",
    }
    assert [p.name for p in final.parent.glob(".scorch-staging*")] == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root removes a read-only directory")
def test_a_staging_directory_that_cannot_be_removed_does_not_stop_a_build(tmp_path):
    environment = _environment(tmp_path / "extensions")
    final = _final_path(tmp_path, environment)
    stuck = final.parent / STAGING
    stuck.mkdir(parents=True)
    (stuck / "main.cpp").write_text("not this build")
    stuck.chmod(0o500)  # its file cannot be removed, so neither can it
    try:
        assert _result(_start(LOAD, tmp_path, environment)) == {"value": 7}
        assert SOURCE in (final.parent / "main.cpp").read_text()
        # The build ran in a directory of its own and removed it.
        assert [p.name for p in final.parent.glob(".scorch-staging*")] == [STAGING]
    finally:
        stuck.chmod(0o700)


# Calls a generated kernel through each public entry that builds one, then calls them
# again with the build path replaced by one that raises.
PUBLIC_ENTRIES = textwrap.dedent("""
    import torch
    import scorch

    def entries():
        g = torch.Generator().manual_seed(64)
        matrix = (torch.rand(40, 30, generator=g) < 0.2) * torch.rand(40, 30, generator=g)
        x = torch.rand(30, 8, generator=g)
        a = scorch.from_torch(matrix.to_sparse_csr(), "A").to_sparse("ss")
        b = scorch.from_torch(matrix.t().contiguous().to_sparse_csr(), "B").to_sparse("ss")
        out = {}
        out["matmul ss @ dense"] = scorch.matmul(a, x)
        out["matmul ss @ ss"] = scorch.matmul(a, b)
        out["einsum ss ss -> ss"] = scorch.einsum("ij,jk->ik", a, b, format="ss")
        out["elementwise add"] = a + a
        out["einsum ss ss -> ss, elementwise"] = scorch.einsum("ij,ij->ij", a, a, format="ss")
        out["conversion to oo"] = a.to_sparse("oo")
        out["change_mode_order"] = a.change_mode_order([1, 0])
        out["to_torch"] = a.to_torch()
        return {k: dense(v) for k, v in out.items()}

    def dense(value):
        value = value.to_torch() if hasattr(value, "to_torch") else value
        return value.to_dense() if value.layout != torch.strided else value

    def build_refused(*args, **kwargs):
        raise AssertionError("a call that hit a built kernel tried to build one")

    first = entries() if os.environ.get("ENTRIES_BUILD") == "1" else None
    utils._run_jit_build_child_privately = build_refused
    cold = entries()  # in a new process: every kernel imported from disk
    warm = entries()  # every kernel from this process's caches
    for key in cold:
        assert torch.equal(cold[key], warm[key]), key
        if first is not None:
            assert torch.equal(first[key], cold[key]), key
    print(json.dumps({"entries": sorted(cold)}))
    """)


def test_a_call_that_hits_a_built_kernel_never_enters_the_build(tmp_path):
    environment = _environment(tmp_path / "extensions")
    built = _result(
        _start(PUBLIC_ENTRIES, tmp_path, dict(environment, ENTRIES_BUILD="1"))
    )
    assert len(built["entries"]) == 8
    assert _result(_start(PUBLIC_ENTRIES, tmp_path, environment)) == built
    # Positive control: in a directory without the kernels, the guard does stop a build.
    empty = _environment(tmp_path / "no_kernels")
    refused = _start(PUBLIC_ENTRIES, tmp_path, empty)
    _, err = refused.communicate(timeout=TIMEOUT)
    assert refused.returncode != 0 and "tried to build one" in err, err[-2000:]
