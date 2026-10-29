"""The kernel build key holds the programs the build runs, not the ``PATH`` string.

The build key (``KernelBuildOptions.cache_key``) names each kernel's build directory and
enters every kept kernel's options key.  It used to hold the raw ``PATH``, so a profiler
that put its own directory first, or an environment activation, built every kernel
again with the same compiler (POST_LANDING item 14; sapling job 85671, about 102G
instructions a kernel).  It now holds ``build.toolchain_identity``
(``scorch/compiler/toolchain_identity.py``): the wrapper, the driver by the name it is
invoked as, every program ``-###`` shows the driver running for a compile and a link,
with its arguments, collect2's linker on gcc, and the build child's closed environment
without ``PATH``.  These tests pin:

1. a directory holding none of those programs, put in front of ``PATH``, keeps the key,
   and a kernel built under one ``PATH`` is not built again under the other;
2. another driver, a wrapper, or an assembler or linker the driver runs, put in front of
   ``PATH``, changes the key; an assembler or linker the driver does not run (Apple
   clang takes ld from its own toolchain) does not, checked against ``-###`` directly;
3. the identity holds the driver's arguments, with the probe's own paths masked;
4. a kept probe is read without asking the driver, in this process and in a new one,
   also from another working directory; with a relative ``PATH`` entry each working
   directory asks once; a changed ``PATH`` directory, a replaced program or a corrupt
   kept probe asks again; a program the driver names by a relative path fails the probe;
5. a driver that cannot be asked leaves the key on the ``PATH`` string, as before;
6. each variable a compiler reads from its environment is refused by the options parse
   or held in the key, and the build child's environment holds nothing else;
7. two paths to one driver share a build directory, each with its own launcher;
8. a build whose programs changed since its snapshot refuses to run.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import textwrap
from typing import Dict, List, Optional

import pytest

import scorch.utils as utils  # type: ignore[import-untyped]
from scorch.compiler import compile_options  # type: ignore[import-untyped]
from scorch.compiler import toolchain_identity  # type: ignore[import-untyped]
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    CompileOptionsError,
)
from scorch.exceptions import ScorchRuntimeError  # type: ignore[import-untyped]

_DARWIN = platform.system() == "Darwin"


def _snapshot(environ: Dict[str, str]) -> CompileOptions:
    return CompileOptions.from_environment(
        environ=environ,
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )


@pytest.fixture(autouse=True)
def _own_memo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A memo directory of the test's own, and no snapshot kept across tests."""

    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))
    compile_options.clear_build_snapshot_cache()
    yield
    compile_options.clear_build_snapshot_cache()


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """The driver path of every probe that asked the driver, in order."""

    asked: List[str] = []
    probe = toolchain_identity._probe

    def counted(**arguments: object):
        asked.append(str(arguments["cxx_compiler_path"]))
        return probe(**arguments)

    monkeypatch.setattr(toolchain_identity, "_probe", counted)
    return asked


def _path() -> str:
    return os.environ["PATH"]


def _base() -> CompileOptions:
    return _snapshot({"PATH": _path()})


def _stand_in(directory: Path, name: str, target: str, *, refuse: str = "") -> Path:
    """An executable ``name`` in ``directory`` that runs ``target``.

    With ``refuse``, it exits 1 when that argument is passed.
    """

    directory.mkdir(exist_ok=True)
    script = directory / name
    guard = (
        f'for a in "$@"; do [ "$a" = "{refuse}" ] && exit 1; done\n' if refuse else ""
    )
    script.write_text(f'#!/bin/sh\n{guard}exec "{target}" "$@"\n')
    script.chmod(0o755)
    return script


def _real(name: str, base: CompileOptions) -> str:
    """The real path of the program ``name`` the base driver runs (as or ld)."""

    for step in _steps(base, "compile") + _steps(base, "link"):
        if os.path.basename(str(step[0])) == name:
            return str(step[0])
    found = subprocess.run(
        [base.build.cxx_compiler, f"-print-prog-name={name}"],
        executable=base.build.cxx_compiler_path,
        env=base.build.build_child_environment(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return os.path.realpath(
        found if os.path.isabs(found) else str(_which(found, _path()))
    )


def _which(name: str, search_path: str) -> Optional[str]:
    import shutil

    return shutil.which(name, path=search_path)


def _steps(options: CompileOptions, part: str) -> list:
    identity = options.build.toolchain_identity
    assert options.build.toolchain_probe == toolchain_identity.PROBED, identity
    return list(dict((entry[0], entry[1:]) for entry in identity)[part][0])


def _driver_runs(options: CompileOptions) -> List[str]:
    """What ``-###`` says the driver runs for a compile and a link, asked directly."""

    build = options.build
    probe = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / "direct-probe"
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "probe.cpp").write_text("")
    (probe / "probe.o").write_bytes(b"")
    ran: List[str] = []
    for arguments in (
        [*build.extra_cflags, "-x", "c++", "-c", "probe.cpp", "-o", "probe.o", "-###"],
        ["probe.o", "-shared", *build.extra_ldflags, "-o", "probe.so", "-###"],
    ):
        done = subprocess.run(
            [build.cxx_compiler, *arguments],
            executable=build.cxx_compiler_path,
            cwd=probe,
            env=build.build_child_environment(),
            capture_output=True,
            text=True,
            check=True,
        )
        for line in (done.stdout + done.stderr).splitlines():
            # A command's line starts with one space. clang quotes every word; GCC only
            # a word with a character outside [A-Za-z0-9_/.-], so a path may be bare.
            if not line.startswith(" ") or line.strip() in ("", "(in-process)"):
                continue
            words = line[1:]
            if words.startswith('"'):
                program = words[1 : words.index('"', 1)]
            else:
                program = words.split(" ")[0]
                found = (
                    program
                    if os.path.isabs(program)
                    else _which(program, build.executable_search_path)
                )
                ran.append(os.path.realpath(str(found)))
                if os.path.basename(program) == "collect2":
                    linker = subprocess.run(
                        [build.cxx_compiler, "-print-prog-name=ld"],
                        executable=build.cxx_compiler_path,
                        env=build.build_child_environment(),
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout.strip()
                    ran.append(
                        os.path.realpath(
                            linker
                            if os.path.isabs(linker)
                            else str(_which(linker, build.executable_search_path))
                        )
                    )
    return ran


# 1. A directory without programs keeps the key.


def test_a_directory_without_programs_in_front_of_path_keeps_the_key(
    tmp_path: Path,
) -> None:
    base = _base()
    empty = tmp_path / "empty"
    empty.mkdir()
    other = _snapshot({"PATH": f"{empty}{os.pathsep}{_path()}"})
    assert other.build.executable_search_path != base.build.executable_search_path
    assert other.build.toolchain_probe == toolchain_identity.PROBED
    assert other.build.toolchain_identity == base.build.toolchain_identity
    assert other.build_cache_key == base.build_cache_key
    assert other.cache_key == base.cache_key


_CHILD = """
import os, sys
import torch
import scorch
import scorch.utils as utils
from scorch import STensor

built = []
run = utils._run_jit_build_child_privately


def counted(request, **arguments):
    built.append(request.name)
    return run(request, **arguments)


utils._run_jit_build_child_privately = counted
torch.manual_seed(0)
dense = torch.rand(30, 20, dtype=torch.float64) * (torch.rand(30, 20) < 0.3)
vector = torch.rand(20, dtype=torch.float64)
result = scorch.matmul(STensor.from_torch(dense, "A").to_sparse("ss"), vector)
error = (result - dense @ vector).abs().max().item()
print(len(built), error)
"""


def _child_builds(environment: Dict[str, str]) -> int:
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_CHILD)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    built, error = done.stdout.split()
    assert float(error) < 1e-12
    return int(built)


def test_a_kernel_built_under_one_path_is_not_built_again_under_another(
    tmp_path: Path,
) -> None:
    """The defect itself: a new process whose ``PATH`` has a directory in front (as
    perf stat's has) finds every kernel the first process built."""

    empty = tmp_path / "empty"
    empty.mkdir()
    environment = dict(os.environ)
    first = _child_builds(environment)
    environment["PATH"] = f"{empty}{os.pathsep}{environment['PATH']}"
    second = _child_builds(environment)
    assert first > 0
    assert second == 0


# 2. Another program the build runs changes the key.


def test_another_driver_in_front_of_path_changes_the_key(tmp_path: Path) -> None:
    base = _base()
    driver = _stand_in(
        tmp_path / "bin", "c++", os.path.realpath(base.build.cxx_compiler_path)
    )
    other = _snapshot({"PATH": f"{driver.parent}{os.pathsep}{_path()}"})
    assert other.build.cxx_compiler_path == str(driver)
    assert other.build.toolchain_probe == toolchain_identity.PROBED
    entries = dict((entry[0], entry[1:]) for entry in other.build.toolchain_identity)
    assert entries["driver"][1] == str(driver.resolve())
    assert other.build_cache_key != base.build_cache_key


def test_a_wrapper_in_front_of_path_changes_the_key(tmp_path: Path) -> None:
    base = _base()
    wrapper = _stand_in(tmp_path / "bin", "ccache", "/usr/bin/env")
    other = _snapshot({"PATH": f"{wrapper.parent}{os.pathsep}{_path()}"})
    assert other.build.compiler_wrapper_path == str(wrapper)
    entries = dict((entry[0], entry[1:]) for entry in other.build.toolchain_identity)
    assert entries["wrapper"][0][0] == str(wrapper.resolve())
    assert other.build_cache_key != base.build_cache_key


@pytest.mark.parametrize("name", ["as", "ld"])
def test_an_assembler_or_linker_in_front_of_path_enters_the_key_when_it_is_run(
    tmp_path: Path, name: str
) -> None:
    """Whether the driver takes ``name`` from ``PATH`` depends on the driver (gcc takes
    ``as`` from it; Apple clang takes ``ld`` from its own toolchain), so the expected
    outcome comes from asking the driver with ``-###`` directly, not from this test."""

    base = _base()
    program = _stand_in(tmp_path / "bin", name, _real(name, base))
    other = _snapshot({"PATH": f"{program.parent}{os.pathsep}{_path()}"})
    runs_it = str(program.resolve()) in _driver_runs(other)
    assert (other.build_cache_key != base.build_cache_key) is runs_it


# 3. The identity holds the driver's arguments, the probe's own paths masked.


def test_the_identity_holds_the_driver_s_arguments_with_the_probe_masked() -> None:
    base = _base()
    compile_steps, link_steps = _steps(base, "compile"), _steps(base, "link")
    every = [
        str(argument) for step in compile_steps + link_steps for argument in step[2]
    ]
    assert any(toolchain_identity._MASK in argument for argument in every)
    assert not any("scorch-toolchain-probe-" in argument for argument in every)
    assert all(os.path.isabs(str(step[0])) for step in compile_steps + link_steps)
    assert all(len(str(step[1])) == 64 for step in compile_steps + link_steps)
    if _DARWIN:
        sdk = base.build.darwin_toolchain.sdk_root
        assert any(sdk in argument for argument in every)


# 4. The kept probe.


def test_a_kept_probe_is_read_without_asking_the_driver(probes: List[str]) -> None:
    first = _base()
    compile_options.clear_build_snapshot_cache()
    second = _base()
    assert len(probes) == 1
    assert second.build.toolchain_identity == first.build.toolchain_identity


def _ask_in_a_new_process(cwd: Optional[Path] = None) -> List[str]:
    """How many probes a new process asks for, and its build key digest."""

    script = textwrap.dedent("""
        import json, os
        from scorch.compiler import compile_options, toolchain_identity
        asked = []
        probe = toolchain_identity._probe
        def counted(**arguments):
            asked.append(1)
            return probe(**arguments)
        toolchain_identity._probe = counted
        options = compile_options.CompileOptions.from_environment(
            environ={"PATH": os.environ["PATH"]}, forced_schedule=None,
            regblock_override=None, verify_cin_override=False)
        print(len(asked), compile_options.canonical_cache_digest(options.build_cache_key))
        """)
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.split()


def test_a_new_process_reads_the_kept_probe(probes: List[str]) -> None:
    first = _base()
    asked, digest = _ask_in_a_new_process()
    assert len(probes) == 1
    assert int(asked) == 0
    assert digest == compile_options.canonical_cache_digest(first.build_cache_key)


def test_a_new_process_in_another_directory_reads_the_kept_probe(
    probes: List[str], tmp_path: Path
) -> None:
    assert all(os.path.isabs(entry) for entry in _path().split(os.pathsep))
    first = _base()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    asked, digest = _ask_in_a_new_process(elsewhere)
    assert len(probes) == 1
    assert int(asked) == 0
    assert digest == compile_options.canonical_cache_digest(first.build_cache_key)
    memo = compile_options.toolchain_memo_directory()
    assert len(glob.glob(os.path.join(memo, "probe-*.json"))) == 1


def test_another_working_directory_reads_the_kept_probe(
    probes: List[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _base()
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    compile_options.clear_build_snapshot_cache()
    second = _base()
    assert len(probes) == 1
    assert second.build_cache_key == first.build_cache_key


def test_a_relative_path_entry_asks_once_per_working_directory(
    probes: List[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ = {"PATH": f"bin{os.pathsep}{_path()}"}
    keys = []
    for name in ("one", "two", "one"):
        (tmp_path / name).mkdir(exist_ok=True)
        monkeypatch.chdir(tmp_path / name)
        compile_options.clear_build_snapshot_cache()
        keys.append(_snapshot(environ).build_cache_key)
    assert len(probes) == 2
    assert keys[0] == keys[1] == keys[2]


@pytest.mark.parametrize("name", ["bin/ld", "./ld", "../bin/as"])
def test_a_program_named_by_a_relative_path_fails_the_probe(name: str) -> None:
    with pytest.raises(toolchain_identity._ProbeFailed, match="a relative path"):
        toolchain_identity._resolve(name, _path())


def test_a_changed_directory_on_path_asks_again(
    probes: List[str], tmp_path: Path
) -> None:
    directory = tmp_path / "bin"
    directory.mkdir()
    environ = {"PATH": f"{directory}{os.pathsep}{_path()}"}
    first = _snapshot(environ)
    (directory / "unrelated").write_text("")
    os.utime(directory, ns=(1, 1))  # a new entry changes the directory's mtime
    compile_options.clear_build_snapshot_cache()
    second = _snapshot(environ)
    assert len(probes) == 2
    assert second.build_cache_key == first.build_cache_key


def test_a_replaced_program_asks_again(probes: List[str], tmp_path: Path) -> None:
    base = _base()
    target = os.path.realpath(base.build.cxx_compiler_path)
    driver = _stand_in(tmp_path / "bin", "c++", target)
    environ = {"PATH": f"{driver.parent}{os.pathsep}{_path()}"}
    first = _snapshot(environ)
    driver.write_text(f'#!/bin/sh\n# replaced\nexec "{target}" "$@"\n')
    compile_options.clear_build_snapshot_cache()
    second = _snapshot(environ)
    assert len(probes) == 3
    assert second.build_cache_key != first.build_cache_key


def test_a_corrupt_kept_probe_asks_again(probes: List[str]) -> None:
    first = _base()
    memo = compile_options.toolchain_memo_directory()
    kept = glob.glob(os.path.join(memo, "probe-*.json"))
    assert len(kept) == 1
    Path(kept[0]).write_text('{"inputs": [')
    compile_options.clear_build_snapshot_cache()
    second = _base()
    assert len(probes) == 2
    assert second.build.toolchain_identity == first.build.toolchain_identity
    assert json.loads(Path(kept[0]).read_text())["identity"]


# 5. A driver that cannot be asked.


def test_a_driver_that_cannot_be_asked_leaves_the_key_on_path(tmp_path: Path) -> None:
    base = _base()
    driver = _stand_in(
        tmp_path / "bin",
        "c++",
        os.path.realpath(base.build.cxx_compiler_path),
        refuse="-###",
    )
    environ = {"PATH": f"{driver.parent}{os.pathsep}{_path()}"}
    options = _snapshot(environ)
    assert options.build.toolchain_probe.startswith("unprobed: ")
    assert "exited 1" in options.build.toolchain_probe
    assert options.build.toolchain_identity == toolchain_identity.unprobed_identity(
        environ["PATH"], str(driver), options.build.compiler_wrapper_path
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    compile_options.clear_build_snapshot_cache()
    moved = _snapshot({"PATH": f"{empty}{os.pathsep}{environ['PATH']}"})
    assert moved.build_cache_key != options.build_cache_key


# 6. What a compiler reads from its environment.


@pytest.mark.parametrize("name", compile_options._UNSUPPORTED_COMPILER_ENVIRONMENT)
def test_a_compiler_search_variable_is_refused_and_never_reaches_a_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    with pytest.raises(CompileOptionsError) as raised:
        _snapshot({"PATH": _path(), name: str(tmp_path)})
    assert raised.value.diagnostics[0].code == "unsupported_compiler_environment"
    options = _base()
    monkeypatch.setenv(name, str(tmp_path))
    assert name not in options.build.build_child_environment()
    entries = dict((entry[0], entry[1:]) for entry in options.build.toolchain_identity)
    assert name not in dict(entries["environment"][0])


@pytest.mark.skipif(not _DARWIN, reason="Darwin toolchain variables")
@pytest.mark.parametrize("name", ["SDKROOT", "DEVELOPER_DIR"])
def test_a_darwin_toolchain_variable_is_refused_unless_it_is_the_policy_s(
    name: str,
) -> None:
    base = _base()
    toolchain = base.build.darwin_toolchain
    policy = {"SDKROOT": toolchain.sdk_root, "DEVELOPER_DIR": toolchain.developer_dir}
    same = _snapshot({"PATH": _path(), name: policy[name]})
    assert same.build_cache_key == base.build_cache_key
    with pytest.raises(CompileOptionsError):
        _snapshot({"PATH": _path(), name: "/nonexistent"})


@pytest.mark.skipif(not _DARWIN, reason="Darwin deployment target")
def test_a_deployment_target_enters_the_key_and_the_linker_s_arguments() -> None:
    base = _base()
    other = _snapshot({"PATH": _path(), "MACOSX_DEPLOYMENT_TARGET": "13.0"})
    entries = dict((entry[0], entry[1:]) for entry in other.build.toolchain_identity)
    assert ("MACOSX_DEPLOYMENT_TARGET", "13.0") in entries["environment"][0]
    assert other.build_cache_key != base.build_cache_key
    assert _steps(other, "link") != _steps(base, "link")


def test_the_build_child_environment_is_closed() -> None:
    environment = _base().build.build_child_environment()
    allowed = {
        "PATH",
        "PYTHONPATH",
        "TORCH_NO_COMPILER_WRAPPER",
        "CXX",
        "DEVELOPER_DIR",
        "SDKROOT",
        "MACOSX_DEPLOYMENT_TARGET",
    }
    assert set(environment) <= allowed


# 7. Two paths to one driver.


def test_two_paths_to_one_driver_share_a_build_directory(tmp_path: Path) -> None:
    base = _base()
    directory = tmp_path / "bin"
    directory.mkdir()
    (directory / "c++").symlink_to(base.build.cxx_compiler_path)
    other = _snapshot({"PATH": f"{directory}{os.pathsep}{_path()}"})
    assert other.build.cxx_compiler_path == str(directory / "c++")
    assert other.build_cache_key == base.build_cache_key
    requests = [
        utils._prepare_jit_build(
            "kernel_toolchain_identity_test",
            ("int f() { return 1; }",),
            ("f",),
            options.build.extra_cflags,
            options.build.extra_ldflags,
            compile_options=options,
        ).request
        for options in (base, other)
    ]
    assert requests[0].build_directory == requests[1].build_directory
    launchers = [utils._prepare_compiler_toolchain(request) for request in requests]
    assert launchers[0] != launchers[1]
    for launcher, options in zip(launchers, (base, other)):
        assert os.readlink(os.path.join(launcher, "c++")) == (
            options.build.cxx_compiler_path
        )


# 8. A build whose programs changed since its snapshot.


def test_a_build_refuses_programs_changed_since_its_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base()
    request = utils._prepare_jit_build(
        "kernel_toolchain_identity_refused",
        ("int f() { return 1; }",),
        ("f",),
        base.build.extra_cflags,
        base.build.extra_ldflags,
        compile_options=base,
    ).request
    os.makedirs(request.build_directory, exist_ok=True)
    monkeypatch.setattr(
        compile_options,
        "_probe_toolchain",
        lambda build: (("changed",), toolchain_identity.PROBED),
    )

    def no_child(*arguments: object, **keywords: object) -> None:
        raise AssertionError("a build child ran")

    monkeypatch.setattr(utils.subprocess, "run", no_child)
    with pytest.raises(ScorchRuntimeError) as raised:
        utils._run_jit_build_child_privately(request, toolchain_directory=None)
    assert raised.value.code == "jit_build_runtime_mismatch"
    assert "toolchain_identity" in str(raised.value)


# 9. Reading the driver's -### output.

# Apple clang 21's -### output on the M5, its command cut to a few words.
_CLANG_TRANSCRIPT = (
    "Apple clang version 21.0.0 (clang-2100.1.1.101)\n"
    "Target: arm64-apple-darwin25.4.0\n"
    "Thread model: posix\n"
    "InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/"
    "XcodeDefault.xctoolchain/usr/bin\n"
    ' "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain'
    '/usr/bin/clang" "-cc1" "-triple" "arm64-apple-macosx26.0.0" "-O3" "-o" "probe.o"'
    ' "-x" "c++" "probe.cpp"\n'
)

# GCC's -### output, written to gcc.cc's execute(): a word is quoted only when it has a
# character other than a letter, a digit, '_', '/', '-' or '.', and '"', '\\' and '$'
# are escaped inside quotes. The lines that are not commands start with no space.
_GCC_TRANSCRIPT = (
    "Using built-in specs.\n"
    "COLLECT_GCC=/usr/bin/c++\n"
    "Target: x86_64-linux-gnu\n"
    "Configured with: ../src/configure -v --enable-languages=c,c++\n"
    "Thread model: posix\n"
    "gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04) \n"
    "COLLECT_GCC_OPTIONS='-O3' '-c' '-o' 'probe.o' '-mtune=generic'\n"
    " /usr/lib/gcc/x86_64-linux-gnu/11/cc1plus -quiet -D_GNU_SOURCE probe.cpp"
    ' "-mtune=generic" -O3 "-DX=\\$HOME" "-DQ=\\"a b\\"" "" -o /tmp/ccAbCd12.s\n'
    "COLLECT_GCC_OPTIONS='-O3' '-c' '-o' 'probe.o' '-mtune=generic'\n"
    " as --64 -o probe.o /tmp/ccAbCd12.s\n"
)


def test_clang_s_commands_are_read_word_by_word() -> None:
    (command,) = toolchain_identity._subcommands_listed(_CLANG_TRANSCRIPT)
    assert command[0].endswith("/usr/bin/clang")
    assert command[1:4] == ["-cc1", "-triple", "arm64-apple-macosx26.0.0"]
    assert command[-1] == "probe.cpp"


def test_clang_s_in_process_line_is_not_a_command() -> None:
    lines = _CLANG_TRANSCRIPT.splitlines()
    transcript = "\n".join(lines[:4] + [" (in-process)"] + lines[4:]) + "\n"
    assert toolchain_identity._subcommands_listed(
        transcript
    ) == toolchain_identity._subcommands_listed(_CLANG_TRANSCRIPT)


def test_gcc_s_commands_are_read_with_bare_and_quoted_words() -> None:
    compile_step, assemble_step = toolchain_identity._subcommands_listed(
        _GCC_TRANSCRIPT
    )
    assert compile_step == [
        "/usr/lib/gcc/x86_64-linux-gnu/11/cc1plus",
        "-quiet",
        "-D_GNU_SOURCE",
        "probe.cpp",
        "-mtune=generic",
        "-O3",
        "-DX=$HOME",
        '-DQ="a b"',
        "",
        "-o",
        "/tmp/ccAbCd12.s",
    ]
    assert assemble_step == ["as", "--64", "-o", "probe.o", "/tmp/ccAbCd12.s"]


def test_a_dollar_escaped_in_quotes_is_read_as_the_drivers_meant_it() -> None:
    """Both drivers write $ as \\$ inside quotes, as a POSIX shell reads it; shlex keeps
    the backslash, which is why the probe does not use it."""

    line = ' "/usr/bin/clang" "-DX=\\$HOME"'
    assert shlex.split(line)[1] == "-DX=\\$HOME"
    assert toolchain_identity._subcommands_listed(line) == [
        ["/usr/bin/clang", "-DX=$HOME"]
    ]


@pytest.mark.parametrize(
    "line",
    [' "/usr/bin/clang" "-cc1', ' "/usr/bin/clang""-cc1"', ' /usr/bin/c"c'],
    ids=["unterminated", "glued", "quote_in_a_bare_word"],
)
def test_a_line_that_does_not_parse_is_refused(line: str) -> None:
    with pytest.raises(toolchain_identity._ProbeFailed, match="does not parse"):
        toolchain_identity._subcommands_listed(line)


def test_an_unprobed_reason_carries_what_the_driver_printed(tmp_path: Path) -> None:
    """A driver whose -### output lists no command: the reason says what it printed, so
    a failure on another host can be read without running the driver there."""

    base = _base()
    directory = tmp_path / "bin"
    directory.mkdir()
    driver = directory / "c++"
    target = os.path.realpath(base.build.cxx_compiler_path)
    driver.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do [ "$a" = "-###" ] && '
        '{ echo "planted first line"; echo "planted second line"; exit 0; }; done\n'
        f'exec "{target}" "$@"\n'
    )
    driver.chmod(0o755)
    options = _snapshot({"PATH": f"{directory}{os.pathsep}{_path()}"})
    assert options.build.toolchain_probe == (
        "unprobed: -### printed no subcommand; it printed: "
        "planted first line | planted second line"
    )
