import hashlib
import importlib.util
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from importlib import resources
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import torch
from .compiler.llir import DataType
from .exceptions import (
    EinsumLoopOrderError,
    ScorchImportError,
    ScorchOSError,
    ScorchRuntimeError,
    ScorchTypeError,
    ScorchValueError,
    defect_code,
)
from .format import parse_format  # noqa: F401 - compatibility re-export

if TYPE_CHECKING:
    from .compiler.compile_options import CompileOptions, KernelBuildOptions

_NATIVE_RESOURCES = resources.files("scorch").joinpath("csrc")


def native_resource_text(filename: str) -> str:
    """Read a packaged native source or header as UTF-8 text.

    A name the package does not ship raises ``ScorchOSError`` with code
    ``native_resource_missing``: still an ``OSError``, and a refusal with a code
    rather than the bare ``FileNotFoundError`` of the resource reader.
    """
    if filename in {"", ".", ".."} or "/" in filename or "\\" in filename:
        raise ScorchValueError(
            "filename must name a file directly under scorch/csrc",
            code="internal_argument_type_invalid",
        )
    try:
        return _NATIVE_RESOURCES.joinpath(filename).read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ScorchOSError(
            f"scorch/csrc/{filename} is not in the installed package",
            code="native_resource_missing",
        ) from error


def _policy_header_text() -> str:
    """Return packaged policy text, with optional local tuning overrides first."""
    base = native_resource_text("scorch_policy.h")
    tuned = _NATIVE_RESOURCES.joinpath("scorch_policy_tuned.h")
    tuned_text = tuned.read_text(encoding="utf-8") if tuned.is_file() else ""
    return tuned_text + base


# The files ``jit_preamble_text`` joins.  The last one is the optional local tuning
# header, which usually does not exist.
_PREAMBLE_RESOURCE_NAMES = (
    "header.h",
    "native_abi.h",
    "scorch_policy.h",
    "scorch_policy_tuned.h",
)
# (resource root, paths of the files above under it), so the paths are built once.
_preamble_paths_memo: Optional[Tuple[Any, Optional[Tuple[str, ...]]]] = None
# (file identity, preamble text): the text last joined and the files it was joined from.
_preamble_text_memo: Optional[Tuple[Tuple[object, ...], str]] = None


def _preamble_resource_paths() -> Optional[Tuple[str, ...]]:
    """Paths of the preamble's files, or None when the package is not plain files."""
    global _preamble_paths_memo
    root = _NATIVE_RESOURCES
    memo = _preamble_paths_memo
    if memo is not None and memo[0] is root:
        return memo[1]
    paths: Optional[Tuple[str, ...]] = None
    if isinstance(root, os.PathLike):
        directory = os.fspath(root)
        paths = tuple(
            os.path.join(directory, name) for name in _PREAMBLE_RESOURCE_NAMES
        )
    _preamble_paths_memo = (root, paths)
    return paths


def jit_preamble_file_identity() -> Optional[Tuple[object, ...]]:
    """Identify the files behind ``jit_preamble_text`` without reading them.

    Each file contributes its path, size, modification time in nanoseconds and inode
    number, or its path alone when it does not exist.  The preamble text is a function
    of these files, so a caller may reuse anything it derived from the text for as long
    as this value compares equal.  The cost is one ``stat`` per file.

    Returns None when the package is not stored as plain files (a zip archive, for
    example): nothing short of reading the text identifies it then.
    """
    paths = _preamble_resource_paths()
    if paths is None:
        return None
    identity: List[Tuple[object, ...]] = []
    optional = len(paths) - 1
    for position, path in enumerate(paths):
        # The tuning header is usually absent.  Asking first keeps every public call
        # from raising and catching an exception for it.
        if position == optional and not os.access(path, os.F_OK):
            identity.append((path,))
            continue
        try:
            status = os.stat(path)
        except OSError:
            identity.append((path,))
        else:
            identity.append((path, status.st_size, status.st_mtime_ns, status.st_ino))
    return tuple(identity)


def jit_preamble_text() -> str:
    """Return the complete packaged C++ preamble used by generated kernels.

    ``load_inline`` writes its source into a separate cache directory, so quote
    includes in ``header.h`` cannot reliably find adjacent installed headers.
    Expanding the packaged policy header into the template keeps each generated
    translation unit self-contained and also works for non-filesystem resource
    loaders.

    The files are read and joined again only when ``jit_preamble_file_identity``
    changes; otherwise the same ``str`` object is returned.  Without a file identity
    the text is read on every call, as it was before the text was kept.
    """
    global _preamble_text_memo
    identity = jit_preamble_file_identity()
    memo = _preamble_text_memo
    if identity is not None and memo is not None and memo[0] == identity:
        return memo[1]
    text = _read_jit_preamble_text()
    # Keep the text only if no file changed while it was being read: otherwise the
    # text could be newer than the identity it would be stored under.
    if identity is not None and identity == jit_preamble_file_identity():
        _preamble_text_memo = (identity, text)
    return text


def _read_jit_preamble_text() -> str:
    """Read and join the packaged preamble files."""
    template = native_resource_text("header.h")
    abi_include = '#include "native_abi.h"'
    if template.count(abi_include) != 1:
        raise ScorchRuntimeError(
            "packaged header.h must include native_abi.h once",
            code="packaged_header_invalid",
        )
    template = template.replace(abi_include, native_resource_text("native_abi.h"), 1)

    include = '#include "scorch_policy.h"'
    if template.count(include) != 1:
        raise ScorchRuntimeError(
            "packaged header.h must include scorch_policy.h once",
            code="packaged_header_invalid",
        )
    return template.replace(include, _policy_header_text(), 1)


def _resolve_compile_options(
    compile_options: Optional["CompileOptions"] = None,
) -> "CompileOptions":
    """Resolve one typed snapshot at a direct utility compilation boundary."""

    from .compiler.compile_options import CompileOptions

    if compile_options is None:
        return CompileOptions.from_environment()
    if type(compile_options) is not CompileOptions:
        raise ScorchTypeError(
            "compile_options must be a CompileOptions instance",
            code="internal_argument_type_invalid",
        )
    return compile_options


def _kernel_name(
    *sources: str, compile_options: Optional["CompileOptions"] = None
) -> str:
    """Deterministic name from kernel source so torch's disk cache persists.

    Includes torch version in the hash so a PyTorch upgrade invalidates all
    cached .so files (they link against libtorch). ``jit_preamble_text`` expands
    the policy header into the source, so a local policy retune is covered too.
    """
    options = _resolve_compile_options(compile_options)
    keyed = "".join(sources) + options.build.torch_version
    if options.build.jit_tune_hooks:
        # Instrumented sweep kernels carry extra getenv branches (same text, -D flag),
        # so key them apart from clean kernels to avoid serving one for the other.
        keyed += "|scorch_tune_hooks"
    h = hashlib.md5(keyed.encode()).hexdigest()[:12]
    return f"kernel_{h}"


_so_cache: Dict[tuple[str, tuple[object, ...]], Any] = {}


@dataclass(frozen=True)
class _JITBuildRequest:
    """Detached, typed payload for one out-of-process Torch extension build."""

    name: str
    cpp_sources: Tuple[str, ...]
    functions: Tuple[str, ...]
    extra_cflags: Tuple[str, ...]
    extra_ldflags: Tuple[str, ...]
    build_directory: str
    build_options: "KernelBuildOptions"


@dataclass(frozen=True)
class _PreparedJITBuild:
    """Validated build request and identities at the pre-native boundary."""

    request: _JITBuildRequest
    cache_key: tuple[str, tuple[object, ...]]
    so_path: str


def _validate_prepared_jit_build(prepared: object) -> _PreparedJITBuild:
    """Validate that one pre-native carrier has a coherent frozen identity."""

    if type(prepared) is not _PreparedJITBuild:
        raise ScorchTypeError(
            "prepared JIT build must be an exact _PreparedJITBuild",
            code="jit_build_request_invalid",
        )
    typed_prepared = prepared
    request = _validate_jit_build_request(typed_prepared.request)
    expected_cache_key = (request.name, _request_cache_key(request))
    if typed_prepared.cache_key != expected_cache_key:
        raise ScorchValueError(
            "prepared JIT cache key disagrees with its build request",
            code="jit_build_request_invalid",
        )
    expected_so_path = os.path.join(
        request.build_directory,
        f"{request.name}.so",
    )
    if typed_prepared.so_path != expected_so_path:
        raise ScorchValueError(
            "prepared JIT path disagrees with its build request",
            code="jit_build_request_invalid",
        )
    return typed_prepared


def _validate_jit_build_request(request: object) -> _JITBuildRequest:
    from .compiler.compile_options import KernelBuildOptions

    if type(request) is not _JITBuildRequest:
        raise ScorchTypeError(
            "JIT build payload must be an exact _JITBuildRequest",
            code="jit_build_request_invalid",
        )
    typed_request = request
    if type(typed_request.build_options) is not KernelBuildOptions:
        raise ScorchTypeError(
            "JIT build payload must own exact KernelBuildOptions",
            code="jit_build_request_invalid",
        )
    string_fields = (
        ("name", typed_request.name),
        ("build_directory", typed_request.build_directory),
    )
    for field_name, string_value in string_fields:
        if type(string_value) is not str or not string_value:
            raise ScorchTypeError(
                f"JIT build {field_name} must be a non-empty string",
                code="jit_build_request_invalid",
            )
    sequence_fields = (
        ("cpp_sources", typed_request.cpp_sources),
        ("functions", typed_request.functions),
        ("extra_cflags", typed_request.extra_cflags),
        ("extra_ldflags", typed_request.extra_ldflags),
    )
    for field_name, sequence_value in sequence_fields:
        if type(sequence_value) is not tuple or any(
            type(item) is not str or not item for item in sequence_value
        ):
            raise ScorchTypeError(
                f"JIT build {field_name} must be an exact tuple of strings",
                code="jit_build_request_invalid",
            )
    if not typed_request.cpp_sources or not typed_request.functions:
        raise ScorchValueError(
            "JIT build sources and functions must be non-empty",
            code="jit_build_request_invalid",
        )
    if typed_request.extra_cflags not in (
        typed_request.build_options.extra_cflags,
        typed_request.build_options.special_kernel_cflags,
    ):
        raise ScorchValueError(
            "JIT build compiler flags disagree with KernelBuildOptions",
            code="jit_build_request_invalid",
        )
    if typed_request.extra_ldflags != typed_request.build_options.extra_ldflags:
        raise ScorchValueError(
            "JIT build linker flags disagree with KernelBuildOptions",
            code="jit_build_request_invalid",
        )
    if not os.path.isabs(typed_request.build_directory):
        raise ScorchValueError(
            "JIT build directory must be absolute", code="jit_build_request_invalid"
        )
    return typed_request


def _verify_snapshotted_build_runtime(request: _JITBuildRequest) -> None:
    """Fail closed if the build child does not match its frozen ABI snapshot."""

    import platform
    import sysconfig

    from .compiler.compile_options import CompilerWrapperPolicy

    options = request.build_options
    current_torch_path = torch.__file__
    current_torch_root = os.path.dirname(current_torch_path)
    current_torch_include = os.path.join(current_torch_root, "include")
    current_torch_include_paths = (
        current_torch_include,
        os.path.join(current_torch_include, "torch", "csrc", "api", "include"),
    )
    current_torch_library_path = os.path.join(current_torch_root, "lib")
    current_cache_tag = sys.implementation.cache_tag
    current_python_include = sysconfig.get_path("include", scheme="posix_prefix")
    observed = (
        ("torch_version", str(torch.__version__), options.torch_version),
        ("torch_path", current_torch_path, options.torch_path),
        (
            "torch_include_paths",
            current_torch_include_paths,
            options.torch_include_paths,
        ),
        (
            "torch_library_path",
            current_torch_library_path,
            options.torch_library_path,
        ),
        (
            "torch_cxx11_abi",
            bool(torch._C._GLIBCXX_USE_CXX11_ABI),
            options.torch_cxx11_abi,
        ),
        ("python_executable", sys.executable, options.python_executable),
        ("python_version", platform.python_version(), options.python_version),
        ("python_cache_tag", current_cache_tag, options.python_cache_tag),
        (
            "python_include_path",
            current_python_include,
            options.python_include_path,
        ),
    )
    mismatches = [name for name, current, expected in observed if current != expected]

    wrapper_disabled = bool(os.environ.get("TORCH_NO_COMPILER_WRAPPER"))
    if options.compiler_wrapper_policy is CompilerWrapperPolicy.DISABLED:
        if not wrapper_disabled:
            mismatches.append("compiler_wrapper_policy")
    else:
        if wrapper_disabled:
            mismatches.append("compiler_wrapper_policy")
        observed_wrapper_name: Optional[str] = None
        observed_wrapper_path: Optional[str] = None
        for wrapper_name in ("ccache", "sccache"):
            resolved_wrapper = shutil.which(wrapper_name)
            if resolved_wrapper is not None:
                observed_wrapper_name = wrapper_name
                observed_wrapper_path = os.path.abspath(resolved_wrapper)
                break
        if observed_wrapper_name != options.compiler_wrapper_name:
            mismatches.append("compiler_wrapper_name")
        if observed_wrapper_path != options.compiler_wrapper_path:
            mismatches.append("compiler_wrapper_path")
    if mismatches:
        raise ScorchRuntimeError(
            "JIT build runtime differs from CompileOptions snapshot: "
            + ", ".join(mismatches),
            code="jit_build_runtime_mismatch",
        )


# The build child writes this line, then a JSON object naming the exception that
# stopped its build, as the last thing on its stderr.
_JIT_BUILD_ERROR_MARKER = "@@scorch-jit-build-error@@\n"
# Exception types the parent re-raises as themselves; any other becomes a
# RuntimeError with the same message.
# The type a build child reported, by name, and the Scorch variant the parent raises:
# the same built-in, carrying the child's code.
_JIT_BUILD_CHILD_ERRORS = {
    name: variant
    for builtin, variant in (
        (RuntimeError, ScorchRuntimeError),
        (ImportError, ScorchImportError),
        (OSError, ScorchOSError),
        (ValueError, ScorchValueError),
        (TypeError, ScorchTypeError),
    )
    for name in (builtin.__name__, variant.__name__)
}


def _run_jit_build_request() -> None:
    """Child-process entry point; build without mutating the parent environment.

    The exception that stops a build -- ``load_inline``'s ``RuntimeError``,
    whose message carries the compiler log, or a runtime-snapshot mismatch --
    is reported after ``_JIT_BUILD_ERROR_MARKER`` on stderr, so the parent
    raises the type and message an in-process ``load_inline`` would.
    """

    from torch.utils.cpp_extension import load_inline

    request = _validate_jit_build_request(pickle.load(sys.stdin.buffer))
    try:
        _verify_snapshotted_build_runtime(request)
        load_inline(
            name=request.name,
            cpp_sources=list(request.cpp_sources),
            functions=list(request.functions),
            extra_cflags=list(request.extra_cflags),
            extra_ldflags=list(request.extra_ldflags),
            build_directory=request.build_directory,
        )
    except Exception as error:
        sys.stderr.write(
            _JIT_BUILD_ERROR_MARKER
            + json.dumps(
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "code": defect_code(error),
                }
            )
        )
        sys.stderr.flush()
        raise SystemExit(1)


def _raise_for_jit_build_child(
    name: str,
    completed: "subprocess.CompletedProcess[bytes]",
    *,
    where: str = "",
) -> None:
    """Forward the build child's stderr and re-raise the error that stopped it.

    ``where`` is added to the error's message: the sentence ``_keep_failed_build``
    returns, saying where a failed build's files are kept.
    """

    stderr = (completed.stderr or b"").decode("utf-8", errors="replace")
    log, marker, payload = stderr.partition(_JIT_BUILD_ERROR_MARKER)
    if log:
        sys.stderr.write(log)
        sys.stderr.flush()
    if completed.returncode == 0 and not marker:
        return
    if marker:
        error = json.loads(payload)
        error_type = _JIT_BUILD_CHILD_ERRORS.get(error["type"], ScorchRuntimeError)
        raise error_type(
            error["message"] + where, code=error.get("code") or "jit_build_failed"
        )
    raise ScorchRuntimeError(
        f"JIT build child for {name!r} exited with status "
        f"{completed.returncode} before reporting an error:\n{log}{where}",
        code="jit_build_failed",
    )


def _request_cache_key(request: _JITBuildRequest) -> tuple[object, ...]:
    from .compiler.compile_options import canonical_cache_digest

    content_digest = canonical_cache_digest(
        (
            request.cpp_sources,
            request.functions,
            request.extra_cflags,
            request.extra_ldflags,
        )
    )
    return request.build_options.cache_key + (content_digest,)


def _build_identity_digest(
    cpp_sources: Sequence[str],
    functions: Sequence[str],
    extra_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    build_options: "KernelBuildOptions",
) -> str:
    from .compiler.compile_options import canonical_cache_digest

    return canonical_cache_digest(
        (
            tuple(cpp_sources),
            tuple(functions),
            tuple(extra_cflags),
            tuple(extra_ldflags),
            build_options.cache_key,
        )
    )


def _load_extension_file(name: str, so_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, so_path)
    if spec is None or spec.loader is None:
        raise ScorchRuntimeError(
            f"could not load compiled extension {so_path}",
            code="jit_extension_load_failed",
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A build's private directory is made inside the build directory, so that it is on the
# same filesystem as the final path; the lock file lets one process at a time build there.
# A builder that holds the lock builds in _JIT_STAGING_NAME, so every build of a kernel
# writes the same paths (into build.ninja and the compiler's messages); without the lock
# each builder makes a directory of its own, whose 8-character random suffix is never
# "build".  Both names start with _JIT_STAGING_PREFIX.
_JIT_STAGING_PREFIX = ".scorch-staging-"
_JIT_STAGING_NAME = _JIT_STAGING_PREFIX + "build"
_JIT_BUILD_LOCK_NAME = ".scorch-build.lock"


def _lock_jit_build_directory(build_directory: str) -> Tuple[Optional[int], bool]:
    """Wait for the build directory's lock; return its descriptor and whether it is held.

    ``flock`` is advisory and belongs to the open file, so the kernel releases it when
    the process holding it dies: a killed builder never leaves a lock behind, as torch's
    build lock (a file created with ``O_EXCL``) does. Where there is no ``flock`` the
    lock is not held and builds may overlap; publishing stays atomic either way.
    """

    try:
        import fcntl
    except ImportError:  # no fcntl on this platform
        return None, False
    descriptor = os.open(
        os.path.join(build_directory, _JIT_BUILD_LOCK_NAME),
        os.O_RDWR | os.O_CREAT,
        0o644,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError:  # the filesystem has no flock
        return descriptor, False
    return descriptor, True


def _publish_staged_build(staging: str, build_directory: str, name: str) -> None:
    """Move a finished build's files into the build directory, the ``.so`` last.

    ``os.replace`` is atomic on one filesystem, so ``<name>.so`` appears on its final
    path whole or not at all. The sources and the ninja file come first, so the build
    directory holds what an in-place build left there (``main.cpp``, ``build.ninja``).
    """

    library = f"{name}.so"
    for entry in os.scandir(staging):
        if entry.name not in (library, "lock") and entry.is_file(follow_symlinks=False):
            os.replace(entry.path, os.path.join(build_directory, entry.name))
    staged_library = os.path.join(staging, library)
    if os.path.isfile(staged_library):
        os.replace(staged_library, os.path.join(build_directory, library))


def _keep_failed_build(staging: str, build_directory: str, name: str) -> str:
    """Leave a failed build's files where they can be read; say where they are.

    The staging directory stays, so the paths in the compiler's messages name files
    that exist, and its files are also copied into the build directory, where an
    in-place build left them before 0064 (``main.cpp`` above all). The ``.so`` is
    never copied: a failed link can leave part of one. The next build of the kernel
    removes the staging directory and replaces the copies.

    Returns the sentence the build's error adds, or ``""`` when the build left no
    file (the child stopped before writing one, as on a runtime-snapshot mismatch);
    the caller then removes the empty directory and the error is unchanged.
    """

    library = f"{name}.so"
    left = [
        entry
        for entry in os.scandir(staging)
        if entry.name != "lock" and entry.is_file(follow_symlinks=False)
    ]
    if not left:
        return ""
    copied = []
    for entry in left:
        if entry.name == library:
            continue
        try:
            shutil.copy2(entry.path, os.path.join(build_directory, entry.name))
            copied.append(entry.name)
        except OSError:  # the build's own error, raised next, matters more
            pass
    also = f" (main.cpp also in {build_directory})" if "main.cpp" in copied else ""
    return (
        f"\nThe failed build's files are kept in {staging}{also}; the next build of "
        "this kernel replaces them."
    )


def _new_staging_directory(build_directory: str, held: bool) -> str:
    """Make the empty directory a build runs in.

    With the lock held, the earlier builds' directories were just removed, and
    ``_JIT_STAGING_NAME`` is made afresh. When one of them could not be removed (so
    ``os.mkdir`` finds it), or the lock is not held, a new directory with a random name
    is made instead. No directory is ever built in twice, so nothing an earlier build
    left there can be published.
    """

    if held:
        staging = os.path.join(build_directory, _JIT_STAGING_NAME)
        try:
            os.mkdir(staging)
            return staging
        except FileExistsError:
            pass
    return tempfile.mkdtemp(prefix=_JIT_STAGING_PREFIX, dir=build_directory)


def _run_jit_build_child_privately(
    request: _JITBuildRequest, *, toolchain_directory: Optional[str]
) -> "subprocess.CompletedProcess[bytes]":
    """Run the build child in a private directory and publish what it builds.

    ``_load_validated_prepared_kernel`` imports any ``<name>.so`` it finds without a
    lock, and torch's build lock is taken only by builders, so a child that linked the
    ``.so`` in place let other processes import a file still being written ("file too
    short"). The child now builds in a directory of its own inside the build directory
    (``_new_staging_directory``) and the finished files are published with
    ``_publish_staged_build``; a builder killed before that leaves nothing on the final
    path. One process at a time builds in a build directory: one that waited for the
    lock finds the published ``.so`` and runs no child, and with the lock held, the
    directories that killed or failed builds left behind are removed first. A failed
    build keeps its files (``_keep_failed_build``), and the error it raises names them.
    """

    build_directory = request.build_directory
    so_path = os.path.join(build_directory, f"{request.name}.so")
    descriptor, held = _lock_jit_build_directory(build_directory)
    try:
        if os.path.isfile(so_path):  # published while this process waited
            return subprocess.CompletedProcess([], 0, None, b"")
        _refuse_changed_toolchain(request)
        if held:
            for entry in os.scandir(build_directory):
                if entry.name.startswith(_JIT_STAGING_PREFIX) and entry.is_dir(
                    follow_symlinks=False
                ):
                    shutil.rmtree(entry.path, ignore_errors=True)
        staging = _new_staging_directory(build_directory, held)
        failed, where = False, ""
        try:
            completed = subprocess.run(
                [
                    request.build_options.python_executable,
                    "-P",
                    "-c",
                    "from scorch.utils import _run_jit_build_request; "
                    "_run_jit_build_request()",
                ],
                input=pickle.dumps(replace(request, build_directory=staging)),
                env=_jit_build_environment_from_request(
                    request,
                    toolchain_directory=toolchain_directory,
                ),
                stderr=subprocess.PIPE,
                check=False,
            )
            # A failure as _raise_for_jit_build_child reads one: a nonzero exit, or an
            # error the child reported (it exits 1 after reporting, so both hold).
            reported = _JIT_BUILD_ERROR_MARKER.encode() in (completed.stderr or b"")
            if completed.returncode == 0 and not reported:
                _publish_staged_build(staging, build_directory, request.name)
            else:
                failed = True
                where = _keep_failed_build(staging, build_directory, request.name)
        finally:
            if not where:  # published, empty, or abandoned before the child finished
                shutil.rmtree(staging, ignore_errors=True)
        if failed:
            _raise_for_jit_build_child(request.name, completed, where=where)
        return completed
    finally:
        if descriptor is not None:
            os.close(descriptor)  # releases the lock


def _refuse_changed_toolchain(request: _JITBuildRequest) -> None:
    """Refuse to build when the programs the build would run are not the snapshot's.

    The build directory is keyed on the snapshot's toolchain identity, and a process
    keeps its snapshot, so a compiler, assembler or linker replaced since then would
    otherwise put a kernel it built under the old programs' key.  Asking again reads
    the kept probe, so this costs a few ``stat`` calls before a build.
    """

    from .compiler.compile_options import _probe_toolchain
    from .compiler.toolchain_identity import PROBED

    build_options = request.build_options
    if build_options.toolchain_probe != PROBED:
        return  # keyed on the PATH string and the resolved paths, as before
    if _probe_toolchain(build_options)[0] != build_options.toolchain_identity:
        raise ScorchRuntimeError(
            "JIT build runtime differs from CompileOptions snapshot: "
            "toolchain_identity",
            code="jit_build_runtime_mismatch",
        )


def _build_and_load_extension(request: _JITBuildRequest) -> Any:
    """Compile in a child with frozen process state, then import the result.

    The child builds privately and its ``.so`` is published atomically
    (``_run_jit_build_child_privately``), so the ``.so`` is never seen half written.
    """

    request = _validate_jit_build_request(request)
    os.makedirs(request.build_directory, exist_ok=True)
    toolchain_directory = _prepare_compiler_toolchain(request)
    so_path = os.path.join(request.build_directory, f"{request.name}.so")
    if not os.path.isfile(so_path):
        completed = _run_jit_build_child_privately(
            request, toolchain_directory=toolchain_directory
        )
        _raise_for_jit_build_child(request.name, completed)
    if not os.path.isfile(so_path):
        raise ScorchRuntimeError(
            f"JIT build did not produce expected extension {so_path}",
            code="jit_build_output_missing",
        )
    return _load_extension_file(request.name, so_path)


def _prepare_compiler_toolchain(request: _JITBuildRequest) -> Optional[str]:
    """Pin a bare compiler spelling to the snapshotted absolute executable."""

    build_options = request.build_options
    if os.path.isabs(build_options.cxx_compiler):
        return None
    # The build directory is keyed on the programs the build runs, not on the path
    # that found the driver, so two snapshots that reach the same driver through
    # different paths share it.  Each such path gets a launcher directory of its own.
    toolchain_directory = os.path.join(
        request.build_directory,
        ".scorch-toolchain",
        hashlib.sha256(build_options.cxx_compiler_path.encode()).hexdigest()[:16],
    )
    os.makedirs(toolchain_directory, exist_ok=True)
    launcher = os.path.join(toolchain_directory, build_options.cxx_compiler)
    if not os.path.lexists(launcher):
        try:
            os.symlink(build_options.cxx_compiler_path, launcher)
        except FileExistsError:
            # A concurrent identical build may have installed the same launcher.
            pass
    if not os.path.islink(launcher) or os.readlink(launcher) != (
        build_options.cxx_compiler_path
    ):
        raise ScorchRuntimeError(
            "JIT compiler launcher conflicts with the CompileOptions snapshot",
            code="jit_build_request_invalid",
        )
    return toolchain_directory


def _jit_build_environment_from_request(
    request: _JITBuildRequest,
    *,
    toolchain_directory: Optional[str],
) -> Dict[str, str]:
    """Build a child environment without requiring a second option snapshot."""

    # One definition, so the toolchain probe asks the driver under this environment.
    return request.build_options.build_child_environment(toolchain_directory)


def _prepare_jit_build(
    name: str,
    cpp_sources: Sequence[str],
    functions: Sequence[str],
    extra_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    *,
    compile_options: Optional["CompileOptions"] = None,
) -> _PreparedJITBuild:
    """Assemble one frozen request and cache identity before native work."""

    options = _resolve_compile_options(compile_options)
    cpp_sources_snapshot = tuple(cpp_sources)
    functions_snapshot = tuple(functions)
    cflags_snapshot = tuple(extra_cflags)
    ldflags_snapshot = tuple(extra_ldflags)
    if cflags_snapshot != options.build.extra_cflags:
        raise ScorchValueError(
            "extra_cflags must match the CompileOptions snapshot",
            code="jit_build_request_invalid",
        )
    if ldflags_snapshot != options.build.extra_ldflags:
        raise ScorchValueError(
            "extra_ldflags must match the CompileOptions snapshot",
            code="jit_build_request_invalid",
        )

    from torch.utils.cpp_extension import _get_build_directory

    build_root = _get_build_directory(name, verbose=False)
    build_digest = _build_identity_digest(
        cpp_sources_snapshot,
        functions_snapshot,
        cflags_snapshot,
        ldflags_snapshot,
        options.build,
    )
    build_dir = os.path.join(build_root, f"scorch_{build_digest}")
    so_path = os.path.join(build_dir, f"{name}.so")
    request = _validate_jit_build_request(
        _JITBuildRequest(
            name=name,
            cpp_sources=cpp_sources_snapshot,
            functions=functions_snapshot,
            extra_cflags=cflags_snapshot,
            extra_ldflags=ldflags_snapshot,
            build_directory=build_dir,
            build_options=options.build,
        )
    )
    cache_key = (name, _request_cache_key(request))
    return _validate_prepared_jit_build(
        _PreparedJITBuild(
            request=request,
            cache_key=cache_key,
            so_path=so_path,
        )
    )


def _load_validated_prepared_kernel(prepared: _PreparedJITBuild) -> Any:
    """Begin cache/native work from a production-validated frozen request."""

    request = prepared.request
    cache_key = prepared.cache_key
    if cache_key in _so_cache:
        return _so_cache[cache_key]

    if os.path.isfile(prepared.so_path):
        module = _load_extension_file(request.name, prepared.so_path)
    else:
        module = _build_and_load_extension(request)

    _so_cache[cache_key] = module
    return module


def _load_prepared_kernel(prepared: _PreparedJITBuild) -> Any:
    """Compatibility boundary validating a detached pre-native carrier."""

    return _load_validated_prepared_kernel(_validate_prepared_jit_build(prepared))


def _load_kernel(
    name: str,
    cpp_sources: Sequence[str],
    functions: Sequence[str],
    extra_cflags: Sequence[str],
    extra_ldflags: Sequence[str],
    *,
    compile_options: Optional["CompileOptions"] = None,
) -> Any:
    """Compatibility boundary that prepares, caches, and loads one kernel.

    PyTorch's JIT_EXTENSION_VERSIONER is in-memory only, so load_inline
    always recompiles on the first call in each process (~7s).  We bypass
    it by checking if the .so already exists on disk and loading it directly.
    """

    return _load_validated_prepared_kernel(
        _prepare_jit_build(
            name,
            cpp_sources,
            functions,
            extra_cflags,
            extra_ldflags,
            compile_options=compile_options,
        )
    )


def get_extra_cflags(
    base_flags: Optional[List[str]] = None,
    *,
    compile_options: Optional["CompileOptions"] = None,
) -> List[str]:
    """Get platform-specific extra compiler flags for torch cpp_extension.

    On macOS, adds the active SDK's C++ standard library include path needed
    for compilation, plus the OpenMP flags.

    Args:
        base_flags: Base compiler flags to include. Defaults to ["-O3"].

    Returns:
        List of compiler flags including platform-specific additions.
    """
    options = _resolve_compile_options(compile_options)
    default_base = ("-O3", "-march=native", "-ffast-math", "-funroll-loops")
    if base_flags is None:
        return list(options.build.direct_extension_cflags)

    # Compatibility callers may replace only the optimization prefix. Target,
    # OpenMP, and instrumentation flags still come from the same snapshot.
    return list(base_flags) + list(
        options.build.direct_extension_cflags[len(default_base) :]
    )


def get_extra_ldflags(
    *, compile_options: Optional["CompileOptions"] = None
) -> List[str]:
    """Get platform-specific extra linker flags for torch cpp_extension.

    On macOS, links against PyTorch's bundled libomp to avoid runtime conflicts
    with Homebrew's libomp.

    Returns:
        List of linker flags.
    """
    options = _resolve_compile_options(compile_options)
    return list(options.build.extra_ldflags)


def _letters_in_order_of_first_appearance(substrings):
    return list(dict.fromkeys("".join(substrings)))


def topo_sort_characters(substrings, tensors):
    # Create a directed graph
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    nodes = set()

    # Initialize in_degree for all characters
    for substring in substrings:
        for char in substring:
            nodes.add(char)
            if char not in in_degree:
                in_degree[char] = 0

    # Add edges to the graph.  A letter beside itself ("ii") orders nothing, and an edge
    # from a letter to itself would be a cycle that no inversion can remove.
    for substring in substrings:
        for i in range(len(substring) - 1):
            if substring[i] == substring[i + 1]:
                continue
            if substring[i + 1] not in graph[substring[i]]:
                graph[substring[i]].append(substring[i + 1])
                in_degree[substring[i + 1]] += 1

    def topo_sort(letters):
        zero_in_degree_nodes = [node for node in letters if in_degree[node] == 0]
        zero_in_degree_nodes = deque(zero_in_degree_nodes)

        result_ = []
        while zero_in_degree_nodes:
            node = zero_in_degree_nodes.popleft()
            result_.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree_nodes.append(neighbor)

            # Sort newly zero in-degree nodes according to priority
            zero_in_degree_nodes = list(zero_in_degree_nodes)
            zero_in_degree_nodes = deque(zero_in_degree_nodes)
        return result_

    # Run topological sort
    result = topo_sort(nodes)

    if len(result) < len(nodes):
        if resolve_cycles(nodes, graph, in_degree, substrings, tensors):
            # Re-run topo_sort with new graph, in_degree
            result = topo_sort(nodes)
        else:
            # The second rule resolved the graph (see resolve_cycles).  Like that rule,
            # this sort takes the letters in order of first appearance, not in the order
            # of the set, so the loop order does not follow the hash seed.
            result = topo_sort(_letters_in_order_of_first_appearance(substrings))

        if len(result) < len(nodes):
            raise EinsumLoopOrderError(
                "einsum loop order: indices "
                + ", ".join(sorted(nodes - set(result)))
                + " are still on a cycle after cycle resolution"
            )

    return result


def resolve_cycles(nodes, graph, in_degree, substrings, tensors):
    """Remove or invert edges of ``graph``, in place, until it has no cycle.

    Each pass takes one cycle and gives up its cheapest edge: the edge is removed, and
    its reverse is put in unless the graph already has it.  That rule does not finish on
    every graph.  An edge it put in belongs to no operand, so a later cycle through one
    has an edge with no cost to compare (``max()`` of an empty list).  And nothing
    bounds its passes: an edge it removed can come back as the reverse of another, so
    the graphs it visits need not all differ, and on a graph with an edge from a letter
    to itself they do not.

    So the rule runs first exactly as it always has, which keeps every loop order it has
    ever selected, and ``True`` is returned.  Only if it meets a cycle through an edge
    of no operand, or arrives at a state it has already had (its passes would then
    repeat for ever), is the graph restored and resolved again by a second rule, and
    ``False`` returned.  The second rule differs in two ways.  An edge the loop put in,
    or one no operand has, is an order no operand asked for, so one found on a cycle is
    dropped, and is never inverted: each pass then either takes an operand edge out for
    good or drops an edge of that kind, so the rule ends within twice as many passes as
    there are operand edges (plus the edges of no operand it was given), with no cycle
    left.  And it looks for cycles from the letters in order of first appearance, not in
    the order of the set, so what it leaves does not follow the hash seed.
    """

    # Finds a cycle in the graph (if any) and then returns cycle's edges
    def find_cycle(letters):
        visited = set()
        stack = []
        in_stack = set()

        def dfs(curr_node):
            visited.add(curr_node)
            stack.append(curr_node)
            in_stack.add(curr_node)

            for neighbor in graph[curr_node]:
                if neighbor not in visited:
                    result = dfs(neighbor)
                    if result:  # If cycle edges were found, propagate them up
                        return result
                elif neighbor in in_stack:
                    # Cycle found, collect the cycle edges
                    cycle_start = stack.index(neighbor)
                    edges = [
                        (stack[i], stack[i + 1])
                        for i in range(cycle_start, len(stack) - 1)
                    ]
                    edges.append((stack[-1], neighbor))
                    return edges

            stack.pop()
            in_stack.remove(curr_node)
            return []

        for node in letters:
            if node not in visited:
                cycle_edges_ = dfs(node)
                if cycle_edges_:
                    return cycle_edges_

        return []

    # Finds the cheapest edge to invert based on tensor shape and already inverted edges
    def invert_cheapest_edge(edges):
        inverted_tensor_indices = list(inverted_edges.keys())

        # Find edge associated with the smallest tensor, tiebreaker goes to result tensor. Represents cost as a tuple.
        def edge_cost(edge):
            edge_tensor_indices = edges_to_tensor_indices[edge]
            return (
                (0, -(max(edge_tensor_indices)))
                if set(edge_tensor_indices).issubset(set(inverted_tensor_indices))
                else (
                    sum(tensor_index_to_size[index] for index in edge_tensor_indices),
                    -max(edge_tensor_indices),
                )
            )

        min_cost_edge = min(edges, key=edge_cost)
        min_cost_tensor_indices = edges_to_tensor_indices[min_cost_edge]
        return min_cost_edge, min_cost_tensor_indices

    inverted_edges: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    edges_to_tensor_indices = defaultdict(list)

    # Build dictionary from edges (tuples) in graph to indices of tensors they appear in
    for tensor_index, substring in enumerate(substrings):
        for i in range(len(substring) - 1):
            edges_to_tensor_indices[(substring[i], substring[i + 1])].append(
                tensor_index
            )

    # Build dictionary from tensor index to size using tensor shapes for operands and result
    tensor_index_to_size = {}
    for i in range(len(tensors)):
        tensor_index_to_size[i] = math.prod(tensors[i].shape)

    result_size = 1
    for char in substrings[-1]:
        shape_index = substrings[0].find(char)
        if shape_index == -1:
            shape_index = substrings[1].find(char)
            result_size *= tensors[1].shape[shape_index]
        else:
            result_size *= tensors[0].shape[shape_index]

    tensor_index_to_size[len(tensors)] = result_size

    node_order = sorted(nodes)

    # Loop while there are cycles, resolving them by inverting edges and storing them in
    # inverted_edges.  Returns False, under the shipped rule only, when that rule cannot
    # finish on this graph.
    def break_cycles(shipped_rule, letters):
        inverted_edges.clear()
        added_edges: Set[Tuple[str, str]] = set()
        seen_states: Set[Tuple[Any, ...]] = set()
        while True:
            cycle_edges = find_cycle(letters)
            if not cycle_edges:
                return True
            if shipped_rule:
                # The next pass is a function of the adjacency lists and of which
                # operands already have an inverted edge, so a state seen before means
                # the passes repeat for ever.
                state = (
                    tuple(tuple(graph[node]) for node in node_order),
                    frozenset(inverted_edges),
                )
                if state in seen_states or not all(
                    edge in edges_to_tensor_indices for edge in cycle_edges
                ):
                    return False
                seen_states.add(state)
            else:
                unasked = next(
                    (
                        edge
                        for edge in cycle_edges
                        if edge in added_edges or edge not in edges_to_tensor_indices
                    ),
                    None,
                )
                if unasked is not None:
                    graph[unasked[0]].remove(unasked[1])
                    in_degree[unasked[1]] -= 1
                    continue
            edge_to_invert, tensor_indices = invert_cheapest_edge(cycle_edges)
            graph[edge_to_invert[0]].remove(edge_to_invert[1])
            in_degree[edge_to_invert[1]] -= 1

            # If the cycle is just two edges, we will remove one edge, not invert it
            if edge_to_invert[0] not in graph[edge_to_invert[1]]:
                graph[edge_to_invert[1]].append(edge_to_invert[0])
                in_degree[edge_to_invert[0]] += 1
                added_edges.add((edge_to_invert[1], edge_to_invert[0]))

            for tensor_index in tensor_indices:
                inverted_edges[tensor_index].append(edge_to_invert)

    graph_at_entry = {node: list(successors) for node, successors in graph.items()}
    if break_cycles(True, nodes):
        return True
    graph.clear()
    graph.update(graph_at_entry)
    break_cycles(False, _letters_in_order_of_first_appearance(substrings))
    # The counts the caller passed in no longer include the edges out of the letters its
    # first sort placed, and the sort after the shipped rule takes those edges out a
    # second time.  The sort after this rule is to be exact, so every edge of the
    # resolved graph is counted afresh.
    in_degree.clear()
    in_degree.update({node: 0 for node in nodes})
    for successors in graph.values():
        for successor in successors:
            in_degree[successor] += 1
    return False


PYTORCH_DTYPE_TO_C_PYTORCH_DTYPE_NAME: Dict[torch.dtype, str] = {
    torch.float32: "kFloat32",
    torch.float64: "kFloat64",
    torch.int32: "kInt32",
    torch.int64: "kInt64",
    torch.int8: "kInt8",
    torch.uint8: "kUInt8",
}

PYTORCH_DTYPE_TO_C_PYTORCH_DTYPE: Dict[torch.dtype, str] = {
    dtype: f"torch::{name}"
    for dtype, name in PYTORCH_DTYPE_TO_C_PYTORCH_DTYPE_NAME.items()
}

PYTORCH_DTYPE_TO_DATATYPE: Dict[torch.dtype, DataType] = {
    torch.float32: DataType.TORCH_FLOAT32,
    torch.float64: DataType.TORCH_FLOAT64,
    torch.int32: DataType.TORCH_INT32,
    torch.int64: DataType.TORCH_INT64,
    torch.int8: DataType.TORCH_INT8,
    torch.uint8: DataType.TORCH_UINT8,
}

PYTORCH_DTYPE_TO_C_DATATYPE: Dict[torch.dtype, DataType] = {
    torch.float32: DataType.FLOAT32,
    torch.float64: DataType.FLOAT64,
    torch.int32: DataType.INT32,
    torch.int64: DataType.INT64,
    torch.int8: DataType.INT8,
    torch.uint8: DataType.UINT8,
}


def dtype_to_c_datatype(dtype: torch.dtype) -> DataType:
    """Convert a pytorch dtype to a C++ DataType.

    Args:
        dtype (torch.dtype): Pytorch dtype.

    Returns:
        DataType: C++ DataType object.
    """
    return PYTORCH_DTYPE_TO_C_DATATYPE[dtype]


def dtype_to_datatype(dtype: torch.dtype) -> DataType:
    """Convert a pytorch dtype to a DataType.

    Args:
        dtype (torch.dtype): Pytorch dtype.

    Returns:
        DataType: DataType object.
    """
    return PYTORCH_DTYPE_TO_DATATYPE[dtype]


def get_pytorch_c_dtype_str(dtype: torch.dtype) -> str:
    """Get the C++ pytorch dtype string for a given pytorch dtype.

    Args:
        dtype (torch.dtype): Pytorch dtype.

    Returns:
        str: C++ pytorch dtype string.
    """
    return PYTORCH_DTYPE_TO_C_PYTORCH_DTYPE[dtype]


def get_pytorch_c_dtype_name(dtype: torch.dtype) -> str:
    """Get the unqualified C++ PyTorch dtype constant name."""

    return PYTORCH_DTYPE_TO_C_PYTORCH_DTYPE_NAME[dtype]


def flatten_2d_list(lst: Iterable[List[Any]]) -> List[Any]:
    return list(chain(*lst))
