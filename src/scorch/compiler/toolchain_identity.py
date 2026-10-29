"""The identity of the programs a kernel build runs.

A kernel's build directory is named by a digest of its build key
(``KernelBuildOptions.cache_key``).  That key used to hold the ``PATH`` string, so any
change to it -- a profiler that puts its own directory first, a conda or module
activation, a job script -- built every kernel again, though the same compiler ran
and the C++ was the same.  The key now holds what ``PATH`` is used for: the programs the
build runs.  They are

- the compiler wrapper (ccache or sccache) when one is found;
- the compiler driver, by the name it is invoked as (clang and clang++ behave
  differently) and by its contents;
- every program the driver runs for a compile and for a link with the snapshot's flags,
  as ``-###`` prints them, with each one's arguments (the system root, the include and
  library directories and the target the driver adds); paths into the probe's own
  directory are masked, and so are the random names of the temporary files a driver
  names there (the probe sets ``TMPDIR`` to it; the build child has no ``TMPDIR``,
  which changes only where those files go);
- for a gcc driver that links through collect2, the linker collect2 runs
  (``-print-prog-name=ld``);
- the build child's environment, which is closed (``utils._jit_build_environment_from_request``),
  except ``PATH`` itself.

Each program is named by its real path and the SHA-256 of its contents.  ``PATH`` is
still passed to the build unchanged.  ninja, which torch's builder finds on ``PATH``,
is not part of the identity: it schedules the commands and does not change what they
produce.

Probing runs the driver two or three times and hashes the programs, tens of
milliseconds, so the result is kept on disk, under the extensions directory, one file
per probe input.  A kept result is used only while every file it names and every
directory a lookup went through has the same ``stat`` identity it had then; any change
probes again.  The probe input holds the working directory only when ``PATH`` has a
relative or empty entry, the one way that directory changes which programs are found, so
a process started in another directory reads the kept result.  A program the driver
names by a relative path with a directory part cannot be placed (it is relative to the
driver's directory, not this process's), so it fails the probe.  Content digests are
kept the same way as kept results, by (real path, inode, size, modification time).  A
probe that cannot run, or whose output cannot be read, leaves
the identity as it was before this module: the ``PATH`` string and the two resolved
paths, so nothing that used to be built apart is shared.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Changes when the probe or the identity's layout changes, so an old kept result is
# never read as a new one.
PROBE_VERSION = 2
PROBED = "probed"
_PROBE_TIMEOUT_SECONDS = 60
_MASK = "<probe>"
# A file in the probe directory other than the probe's own: a temporary file the driver
# names (gcc's ccXXXXXX.s), whose name is random.  TMPDIR points into the directory.
_TEMPORARY = re.compile(re.escape(_MASK) + r"/(?!probe\.(?:cpp|o|so)\b)[^/\s,=]+")
_DIGEST_CHUNK = 1 << 20

# Content digests computed in this process, by (real path, inode, size, mtime_ns).
_digests: Dict[Tuple[str, int, int, int], str] = {}


class _ProbeFailed(Exception):
    """The driver could not be asked, or its answer could not be read."""


def _subcommands_listed(text: str) -> List[List[str]]:
    """The commands a driver's ``-###`` output lists, each as its words.

    clang and GCC both print each command on a line of its own that starts with one
    space; their other lines (the version, the target, GCC's COLLECT_GCC_OPTIONS) start
    with no space. clang also prints `` (in-process)`` above a command it runs in its own
    process, which is not a command. The two quote differently:

    - clang quotes every word (``Command::Print``): `` "/usr/bin/clang" "-cc1" ...``;
    - GCC quotes only a word holding a character other than a letter, a digit, ``_``,
      ``/``, ``-`` or ``.`` (gcc.cc, ``execute``), so a program's path is bare:
      `` /usr/lib/gcc/x86_64-linux-gnu/11/cc1plus -quiet "-mtune=generic" ...``.

    Inside quotes both escape ``"``, ``\\`` and ``$`` with a backslash. shlex would keep
    the backslash before ``$``, so the words are read here.
    """

    commands = []
    for line in text.splitlines():
        if not line.startswith(" ") or line.strip() in ("", "(in-process)"):
            continue
        commands.append(_words_of(line))
    return commands


def _first_lines(text: str) -> str:
    """The start of a driver's output, for an unprobed result's reason: a reader can tell
    from it what the driver printed without running it again."""

    lines = [line[:160] for line in text.splitlines()[:8]]
    return " | ".join(lines)[:1000] or "nothing"


def _words_of(line: str) -> List[str]:
    words: List[str] = []
    index, end = 0, len(line)
    while index < end:
        if line[index] == " ":
            index += 1
            continue
        if line[index] != '"':
            stop = line.find(" ", index)
            stop = end if stop < 0 else stop
            word = line[index:stop]
            if '"' in word or "\\" in word:
                raise _ProbeFailed(f"a -### line does not parse: {line[:200]!r}")
            words.append(word)
            index = stop
            continue
        index += 1
        characters = []
        while True:
            if index >= end:
                raise _ProbeFailed(f"a -### line does not parse: {line[:200]!r}")
            character = line[index]
            if character == "\\" and index + 1 < end and line[index + 1] in '"\\$':
                characters.append(line[index + 1])
                index += 2
            elif character == '"':
                index += 1
                break
            else:
                characters.append(character)
                index += 1
        if index < end and line[index] != " ":
            raise _ProbeFailed(f"a -### line does not parse: {line[:200]!r}")
        words.append("".join(characters))
    return words


def unprobed_identity(
    executable_search_path: str,
    cxx_compiler_path: str,
    compiler_wrapper_path: Optional[str],
) -> Tuple[object, ...]:
    """The identity of a build whose programs were not probed: the strings that find them."""

    return (
        "unprobed",
        executable_search_path,
        cxx_compiler_path,
        compiler_wrapper_path,
    )


def toolchain_identity(
    *,
    cxx_compiler: str,
    cxx_compiler_path: str,
    compiler_wrapper_path: Optional[str],
    environment: Mapping[str, str],
    cflags: Sequence[str],
    ldflags: Sequence[str],
    memo_directory: Optional[str],
) -> Tuple[Tuple[object, ...], str]:
    """Return the build's program identity and how it was found.

    ``environment`` is the build child's environment without the launcher directory
    ``utils._prepare_compiler_toolchain`` puts in front of ``PATH``.  The second value
    is ``PROBED``, or ``"unprobed: <reason>"`` with the fallback identity.
    """

    search_path = environment.get("PATH", os.defpath)
    try:
        working_directory: Optional[str] = os.getcwd()
    except OSError:
        working_directory = None
    relative_search = any(
        not os.path.isabs(entry) for entry in search_path.split(os.pathsep)
    )
    inputs = (
        PROBE_VERSION,
        cxx_compiler,
        cxx_compiler_path,
        compiler_wrapper_path,
        tuple(sorted(environment.items())),
        working_directory if relative_search else None,
        tuple(cflags),
        tuple(ldflags),
        platform.system(),
        platform.machine(),
        _stat_key(cxx_compiler_path),
        None if compiler_wrapper_path is None else _stat_key(compiler_wrapper_path),
    )
    memo_path = None
    if memo_directory is not None:
        name = hashlib.sha256(json.dumps(inputs).encode()).hexdigest()
        memo_path = os.path.join(memo_directory, f"probe-{name}.json")
        kept = _read_kept_probe(memo_path, inputs)
        if kept is not None:
            return kept, PROBED
    try:
        identity, checks = _probe(
            cxx_compiler=cxx_compiler,
            cxx_compiler_path=cxx_compiler_path,
            compiler_wrapper_path=compiler_wrapper_path,
            environment=environment,
            search_path=search_path,
            working_directory=working_directory,
            cflags=cflags,
            ldflags=ldflags,
            memo_directory=memo_directory,
        )
    except _ProbeFailed as failure:
        return (
            unprobed_identity(search_path, cxx_compiler_path, compiler_wrapper_path),
            f"unprobed: {failure}",
        )
    if memo_path is not None:
        _write_atomically(
            memo_path,
            json.dumps({"inputs": inputs, "checks": checks, "identity": identity}),
        )
    return identity, PROBED


def _probe(
    *,
    cxx_compiler: str,
    cxx_compiler_path: str,
    compiler_wrapper_path: Optional[str],
    environment: Mapping[str, str],
    search_path: str,
    working_directory: Optional[str],
    cflags: Sequence[str],
    ldflags: Sequence[str],
    memo_directory: Optional[str],
) -> Tuple[Tuple[object, ...], List[List[object]]]:
    checks: List[List[object]] = []

    def program(lookup_path: str) -> Tuple[str, str]:
        real = os.path.realpath(lookup_path)
        status = _stat_key(real)
        if status is None:
            raise _ProbeFailed(f"cannot stat {lookup_path!r}")
        checks.append(["file", lookup_path, real, *status])
        checks.append(["dir", *_directory_check(os.path.dirname(real))])
        return real, _content_digest(real, status, memo_directory)

    probe_directory = tempfile.mkdtemp(prefix="scorch-toolchain-probe-")
    try:
        real_probe = os.path.realpath(probe_directory)
        for name in ("probe.cpp", "probe.o"):
            with open(os.path.join(probe_directory, name), "wb"):
                pass

        def ask(arguments: Sequence[str]) -> str:
            try:
                done = subprocess.run(
                    [cxx_compiler, *arguments],
                    executable=cxx_compiler_path,
                    cwd=probe_directory,
                    env={**environment, "TMPDIR": probe_directory},
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=_PROBE_TIMEOUT_SECONDS,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise _ProbeFailed(f"{type(error).__name__} running the driver")
            output = done.stdout.decode(errors="replace") + done.stderr.decode(
                errors="replace"
            )
            if done.returncode != 0:
                raise _ProbeFailed(
                    f"the driver exited {done.returncode} for -###; it printed: "
                    + _first_lines(output)
                )
            return output

        def subcommands(text: str) -> Tuple[Tuple[object, ...], ...]:
            found = []
            for tokens in _subcommands_listed(text):
                masked = tuple(
                    _TEMPORARY.sub(
                        lambda found: _MASK
                        + "/<temporary>"
                        + os.path.splitext(found.group(0))[1],
                        token.replace(real_probe, _MASK).replace(
                            probe_directory, _MASK
                        ),
                    )
                    for token in tokens[1:]
                )
                found.append((*program(_resolve(tokens[0], search_path)), masked))
            if not found:
                raise _ProbeFailed(
                    "-### printed no subcommand; it printed: " + _first_lines(text)
                )
            return tuple(found)

        compile_steps = subcommands(
            ask([*cflags, "-x", "c++", "-c", "probe.cpp", "-o", "probe.o", "-###"])
        )
        link_steps = subcommands(
            ask(["probe.o", "-shared", *ldflags, "-o", "probe.so", "-###"])
        )
        collect2_ld: Optional[Tuple[str, str]] = None
        if any(os.path.basename(str(step[0])) == "collect2" for step in link_steps):
            collect2_ld = program(
                _resolve(ask(["-print-prog-name=ld"]).strip(), search_path)
            )
            for line in ask(["-print-search-dirs"]).splitlines():
                if line.startswith("programs: ="):
                    for entry in line[len("programs: =") :].split(os.pathsep):
                        if entry:
                            checks.append(["dir", *_directory_check(entry)])
    finally:
        shutil.rmtree(probe_directory, ignore_errors=True)

    for entry in search_path.split(os.pathsep):
        directory = entry or "."
        if not os.path.isabs(directory) and working_directory is not None:
            directory = os.path.join(working_directory, directory)
        checks.append(["dir", *_directory_check(directory)])
    wrapper = None if compiler_wrapper_path is None else program(compiler_wrapper_path)
    identity = (
        ("probe", PROBE_VERSION),
        ("wrapper", wrapper),
        ("driver", cxx_compiler, *program(cxx_compiler_path)),
        (
            "environment",
            tuple(sorted(item for item in environment.items() if item[0] != "PATH")),
        ),
        ("compile", compile_steps),
        ("link", link_steps),
        ("collect2_ld", collect2_ld),
    )
    return identity, checks


def _resolve(name: str, search_path: str) -> str:
    if os.path.isabs(name):
        return name
    if os.path.dirname(name):
        raise _ProbeFailed(f"the driver runs {name!r}, a relative path")
    found = shutil.which(name, path=search_path)
    if found is None:
        raise _ProbeFailed(f"the driver runs {name!r}, which is not on PATH")
    return os.path.abspath(found)


def _stat_key(path: str) -> Optional[Tuple[int, int, int]]:
    try:
        status = os.stat(path)
    except OSError:
        return None
    return (status.st_ino, status.st_size, status.st_mtime_ns)


def _directory_check(path: str) -> List[object]:
    try:
        status = os.stat(path)
    except OSError:
        return [path, None, None]
    return [path, status.st_ino, status.st_mtime_ns]


def _content_digest(
    real: str, status: Tuple[int, int, int], memo_directory: Optional[str]
) -> str:
    key = (real, *status)
    digest = _digests.get(key)
    if digest is not None:
        return digest
    memo_path = None
    if memo_directory is not None:
        name = hashlib.sha256(json.dumps(key).encode()).hexdigest()
        memo_path = os.path.join(memo_directory, f"digest-{name}.json")
        try:
            with open(memo_path) as handle:
                kept = json.load(handle)
            if kept["key"] == list(key) and type(kept["sha256"]) is str:
                digest = kept["sha256"]
        except (OSError, ValueError, KeyError, TypeError):
            digest = None
    if digest is None:
        hasher = hashlib.sha256()
        try:
            with open(real, "rb") as handle:
                for chunk in iter(lambda: handle.read(_DIGEST_CHUNK), b""):
                    hasher.update(chunk)
        except OSError:
            raise _ProbeFailed(f"cannot read {real!r}")
        if _stat_key(real) != status:
            raise _ProbeFailed(f"{real!r} changed while it was read")
        digest = hasher.hexdigest()
        if memo_path is not None:
            _write_atomically(memo_path, json.dumps({"key": key, "sha256": digest}))
    _digests[key] = digest
    return digest


def _read_kept_probe(
    memo_path: str, inputs: Tuple[object, ...]
) -> Optional[Tuple[object, ...]]:
    """A kept probe result for ``inputs``, if every file and directory it saw is unchanged."""

    try:
        with open(memo_path) as handle:
            kept = json.load(handle)
        if _freeze(kept["inputs"]) != _freeze(json.loads(json.dumps(inputs))):
            return None
        for check in kept["checks"]:
            if check[0] == "file":
                _, lookup_path, real, *status = check
                if os.path.realpath(lookup_path) != real or _stat_key(real) != tuple(
                    status
                ):
                    return None
            elif check[0] == "dir":
                if _directory_check(check[1]) != check[1:]:
                    return None
            else:
                return None
        identity = _freeze(kept["identity"])
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None
    return identity if type(identity) is tuple else None


def _freeze(value: object) -> object:
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


def _write_atomically(path: str, text: str) -> None:
    """Publish ``text`` at ``path`` by rename; a failure leaves nothing kept."""

    directory = os.path.dirname(path)
    try:
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    except OSError:
        return
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
