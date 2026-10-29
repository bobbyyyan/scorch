"""Immutable configuration snapshot for one legacy compiler invocation.

Only the public compilation/runtime boundary constructs :class:`CompileOptions`.
Compiler stages receive that value explicitly and must not consult environment
variables, context-local overrides, platform probes, or mutable build-flag lists.
"""

from __future__ import annotations

import functools
import glob
import hashlib
import math
import os
import platform
import re
import shutil
import sys
import subprocess
import sysconfig
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, NoReturn, Optional, Tuple, cast

from .diagnostics import CompileOptionsDiagnostic, CompileOptionsError
from .llir_pass_manager import (
    CURRENT_LLIR_PASSES,
    DEBUG_LLIR_PASS_OPTIONS,
    LLIRPassId,
    LLIRPassOptions,
    PRODUCTION_LLIR_PASS_OPTIONS,
)
from ..exceptions import ScorchTypeError

if TYPE_CHECKING:
    from .scheduler import Schedule


class TargetOS(Enum):
    """Host operating systems supported by the current JIT build path."""

    DARWIN = "Darwin"
    LINUX = "Linux"


class ISAPolicy(Enum):
    """Instruction-set selection supported by the current build flags."""

    NATIVE = "native"


class ParallelBackend(Enum):
    """Target parallel runtime used by generated kernels."""

    OPENMP = "openmp"


class LegacyParallelPolicy(Enum):
    """Versioned legacy parallel-lowering behavior."""

    CURRENT = "legacy_parallel_v1"


class LegacyLoweringPolicy(Enum):
    """Versioned CIN-to-current-LLIR lowering behavior."""

    CURRENT = "legacy_lowering_v1"


class IndexWidthPolicy(Enum):
    """Index width assumed by the current generated Torch/C++ ABI."""

    INT32 = "int32"


class ABIPolicy(Enum):
    """Native wrapper ABI emitted by the current JIT compiler."""

    TORCH_CPP_EXTENSION = "torch_cpp_extension_v1"


class CompilerABICheckPolicy(Enum):
    """PyTorch compiler-compatibility validation allowed by production builds."""

    REQUIRED = "required"


class CompilerWrapperPolicy(Enum):
    """Implicit compiler-wrapper behavior allowed by the current JIT."""

    AUTO = "auto"
    DISABLED = "disabled"


_BASE_CFLAGS = ("-O3", "-march=native", "-ffast-math", "-funroll-loops")
_SPECIAL_KERNEL_BASE_CFLAGS = (
    "-O3",
    "-march=native",
    "-ffast-math",
    "-fno-signed-zeros",
)
_USE_CONTEXT = object()
_POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*\Z")
_DARWIN_DEPLOYMENT_TARGET = re.compile(r"[1-9][0-9]*(?:\.[0-9]+){0,2}\Z")
_INT32_MAX = 2**31 - 1
_DARWIN_FALLBACK_DEVELOPER_DIR = "/Library/Developer/CommandLineTools"


def _ask_toolchain(argv: Tuple[str, ...]) -> Optional[str]:
    """One line of output from a toolchain query, or None if it cannot answer."""

    try:
        found = subprocess.run(
            list(argv), capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None  # the tool is absent, or it hung
    if found.returncode != 0:
        return None
    answer = found.stdout.strip()
    return answer or None


def _resolve_darwin_toolchain() -> Tuple[str, str]:
    """The ACTIVE developer directory and SDK root, asked rather than assumed.

    These must come from the active toolchain. Pinning them to CommandLineTools,
    which this module used to do, breaks every generated kernel on a host whose
    ``xcode-select -p`` is Xcode: the compiler torch invokes is then Xcode's clang,
    and handing it a different toolchain's libc++ fails inside the headers
    themselves ("reference to unresolved using declaration" in
    ``<__type_traits/is_trivially_copyable.h>``). Both toolchains can be installed
    at once, so the directory check below passes either way and the breakage shows
    up later as a compile error in generated code, looking like a codegen defect
    rather than a toolchain one.

    ``SCORCH_MACOS_SDK`` overrides the SDK lookup. Resolved once per process, so a
    mid-process ``xcode-select`` switch is not picked up; set ``SCORCH_MACOS_SDK``
    if that is ever wanted. Non-Darwin hosts never run the queries.
    """

    fallback_dir = _DARWIN_FALLBACK_DEVELOPER_DIR
    fallback_sdk = f"{fallback_dir}/SDKs/MacOSX.sdk"
    if sys.platform != "darwin":
        return fallback_dir, fallback_sdk

    override = os.environ.get("SCORCH_MACOS_SDK")
    if override:
        sdk_candidates = [override, fallback_sdk]
    else:
        sdk_candidates = [
            _ask_toolchain(("xcrun", "--show-sdk-path")),
            fallback_sdk,
        ]
    sdk_root = fallback_sdk
    for candidate in sdk_candidates:
        if candidate and os.path.isdir(
            os.path.join(candidate, "usr", "include", "c++", "v1")
        ):
            sdk_root = candidate
            break

    developer_dir = _ask_toolchain(("xcode-select", "-p")) or fallback_dir
    if not os.path.isdir(developer_dir):
        developer_dir = fallback_dir
    return developer_dir, sdk_root


_DARWIN_DEVELOPER_DIR, _DARWIN_SDK_ROOT = _resolve_darwin_toolchain()
_DARWIN_CXX_INCLUDE = f"{_DARWIN_SDK_ROOT}/usr/include/c++/v1"
_UNSUPPORTED_COMPILER_ENVIRONMENT = (
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "OBJCPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "COMPILER_PATH",
    "GCC_EXEC_PREFIX",
    "CCC_OVERRIDE_OPTIONS",
)


def _raise_options_error(code: str, field_name: str, message: str) -> NoReturn:
    raise CompileOptionsError(
        (CompileOptionsDiagnostic(code=code, field=field_name, message=message),)
    )


def _require_exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        _raise_options_error("invalid_type", field_name, "expected an exact bool value")
    return cast(bool, value)


def _require_positive_int(
    value: object,
    field_name: str,
    *,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int or cast(int, value) <= 0:
        _raise_options_error(
            "invalid_positive_integer",
            field_name,
            "expected a positive exact integer",
        )
    if maximum is not None and cast(int, value) > maximum:
        _raise_options_error(
            "integer_out_of_range",
            field_name,
            f"expected a value no greater than {maximum}",
        )
    return cast(int, value)


def canonical_cache_digest(value: object) -> str:
    """Hash supported typed cache values without human ``str``/``repr`` forms."""

    digest = hashlib.sha256()

    def write_payload(tag: bytes, payload: bytes) -> None:
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def visit(item: object) -> None:
        if item is None:
            digest.update(b"n")
        elif type(item) is bool:
            digest.update(b"b1" if item else b"b0")
        elif type(item) is int:
            magnitude = abs(item)
            payload = magnitude.to_bytes(
                max(1, (magnitude.bit_length() + 7) // 8), "big"
            )
            write_payload(b"i-" if item < 0 else b"i+", payload)
        elif type(item) is float:
            write_payload(b"f", item.hex().encode("ascii"))
        elif type(item) is str:
            write_payload(b"s", item.encode("utf-8"))
        elif type(item) is tuple:
            digest.update(b"t")
            digest.update(len(item).to_bytes(8, "big"))
            for child in item:
                visit(child)
        else:
            raise ScorchTypeError(
                "cache identity supports only None, exact scalars, and tuples",
                code="internal_argument_type_invalid",
            )

    visit(value)
    return digest.hexdigest()


@functools.lru_cache(maxsize=8)
def _preamble_digest(preamble_source: str) -> str:
    """SHA-256 of one kernel preamble text, computed once per distinct text.

    ``KernelBuildOptions.cache_key`` is read on every public call and the preamble
    is several hundred kilobytes, so hashing it there cost more than a small kernel
    takes to run.  The digest is keyed by the text itself, so it can never describe
    a different text.
    """

    return hashlib.sha256(preamble_source.encode()).hexdigest()


def _require_nonempty_string(value: object, field_name: str) -> str:
    if type(value) is not str or not cast(str, value).strip():
        _raise_options_error(
            "invalid_string", field_name, "expected a non-empty exact string"
        )
    return cast(str, value)


def _snapshot_string_tuple(value: object, field_name: str) -> Tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray, set, frozenset, dict)):
        _raise_options_error(
            "invalid_sequence",
            field_name,
            "expected an ordered sequence of strings",
        )
    if not isinstance(value, Iterable):
        _raise_options_error(
            "invalid_sequence",
            field_name,
            "expected an ordered sequence of strings",
        )
    snapshot: Tuple[object, ...] = tuple(value)
    if any(type(item) is not str or not item for item in snapshot):
        _raise_options_error(
            "invalid_sequence_item",
            field_name,
            "all entries must be non-empty exact strings",
        )
    return cast(Tuple[str, ...], snapshot)


@dataclass(frozen=True)
class DarwinToolchainOptions:
    """Coherent Apple developer-directory, SDK, and deployment snapshot."""

    developer_dir: str = _DARWIN_DEVELOPER_DIR
    sdk_root: str = _DARWIN_SDK_ROOT
    deployment_target: Optional[str] = None

    def _validate_stored_policy(self) -> None:
        """Validate immutable policy values without probing the host again."""

        if type(self.developer_dir) is not str:
            _raise_options_error(
                "invalid_type",
                "build.darwin_toolchain.developer_dir",
                "expected an exact string",
            )
        if self.developer_dir != _DARWIN_DEVELOPER_DIR:
            _raise_options_error(
                "unsupported_darwin_toolchain",
                "build.darwin_toolchain.developer_dir",
                "the current build policy requires CommandLineTools",
            )
        if type(self.sdk_root) is not str:
            _raise_options_error(
                "invalid_type",
                "build.darwin_toolchain.sdk_root",
                "expected an exact string",
            )
        if self.sdk_root != _DARWIN_SDK_ROOT:
            _raise_options_error(
                "unsupported_darwin_toolchain",
                "build.darwin_toolchain.sdk_root",
                "the SDK must match the emitted CommandLineTools libc++ path",
            )
        if self.deployment_target is not None:
            if type(self.deployment_target) is not str:
                _raise_options_error(
                    "invalid_type",
                    "build.darwin_toolchain.deployment_target",
                    "expected an exact string or None",
                )
            if (
                len(self.deployment_target) > 32
                or _DARWIN_DEPLOYMENT_TARGET.fullmatch(self.deployment_target) is None
            ):
                _raise_options_error(
                    "invalid_deployment_target",
                    "build.darwin_toolchain.deployment_target",
                    "expected a numeric macOS version with at most three components",
                )

    def __post_init__(self) -> None:
        self._validate_stored_policy()
        if not os.path.isdir(self.developer_dir):
            _raise_options_error(
                "darwin_toolchain_not_found",
                "build.darwin_toolchain.developer_dir",
                "the CommandLineTools developer directory does not exist",
            )
        if not os.path.isdir(self.sdk_root) or not os.path.isdir(_DARWIN_CXX_INCLUDE):
            _raise_options_error(
                "darwin_toolchain_not_found",
                "build.darwin_toolchain.sdk_root",
                "the CommandLineTools SDK or its libc++ headers do not exist",
            )

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (
            self.developer_dir,
            self.sdk_root,
            self.deployment_target,
        )


@dataclass(frozen=True)
class SchedulerCostModel:
    """Exact current scheduler cost-model constants."""

    alpha: float = 2.975
    beta: float = 0.1005
    gamma: float = 43.55
    c_insert: float = 85.34
    c_sort: float = 1.741
    c_trans: float = 40.61
    rho: float = 0.0014
    default_dim_size: int = 1024

    def __post_init__(self) -> None:
        for field_name, value in (
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("gamma", self.gamma),
            ("c_insert", self.c_insert),
            ("c_sort", self.c_sort),
            ("c_trans", self.c_trans),
            ("rho", self.rho),
        ):
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                _raise_options_error(
                    "invalid_cost_model_value",
                    f"scheduler.cost_model.{field_name}",
                    "expected a positive finite exact float",
                )
        _require_positive_int(
            self.default_dim_size, "scheduler.cost_model.default_dim_size"
        )

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (
            self.alpha,
            self.beta,
            self.gamma,
            self.c_insert,
            self.c_sort,
            self.c_trans,
            self.rho,
            self.default_dim_size,
        )


@dataclass(frozen=True)
class SchedulerPolicy:
    """Immutable scheduling-policy inputs used by the legacy scheduler."""

    regblock_enabled: bool = False
    regblock_max_n: int = 8
    regblock_tile_width: int = 8
    auto_tile_width: int = 32
    cost_model: SchedulerCostModel = field(default_factory=SchedulerCostModel)

    def __post_init__(self) -> None:
        _require_exact_bool(self.regblock_enabled, "scheduler.regblock_enabled")
        _require_positive_int(
            self.regblock_max_n,
            "scheduler.regblock_max_n",
            maximum=_INT32_MAX,
        )
        _require_positive_int(
            self.regblock_tile_width,
            "scheduler.regblock_tile_width",
            maximum=_INT32_MAX,
        )
        _require_positive_int(
            self.auto_tile_width,
            "scheduler.auto_tile_width",
            maximum=_INT32_MAX,
        )
        if type(self.cost_model) is not SchedulerCostModel:
            _raise_options_error(
                "invalid_type",
                "scheduler.cost_model",
                "expected an exact SchedulerCostModel",
            )

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (
            self.regblock_enabled,
            self.regblock_max_n,
            self.regblock_tile_width,
            self.auto_tile_width,
            self.cost_model.cache_key,
        )


@dataclass(frozen=True)
class VerificationPolicy:
    """Immutable production/debug verification and pass instrumentation policy."""

    verify_cin: bool = False
    llir_pass_options: LLIRPassOptions = field(default_factory=LLIRPassOptions)

    def __post_init__(self) -> None:
        _require_exact_bool(self.verify_cin, "verification.verify_cin")
        if type(self.llir_pass_options) is not LLIRPassOptions:
            _raise_options_error(
                "invalid_type",
                "verification.llir_pass_options",
                "expected exact LLIRPassOptions",
            )
        for field_name, value in (
            ("verify_before_pass", self.llir_pass_options.verify_before_pass),
            ("verify_after_pass", self.llir_pass_options.verify_after_pass),
            ("record_timing", self.llir_pass_options.record_timing),
        ):
            _require_exact_bool(
                value,
                f"verification.llir_pass_options.{field_name}",
            )
        if self.llir_pass_options not in (
            PRODUCTION_LLIR_PASS_OPTIONS,
            DEBUG_LLIR_PASS_OPTIONS,
        ):
            _raise_options_error(
                "unsupported_verification_policy",
                "verification.llir_pass_options",
                "only the current production and debug policies are supported",
            )

    @property
    def cache_key(self) -> tuple[object, ...]:
        options = self.llir_pass_options
        return (
            self.verify_cin,
            options.verify_before_pass,
            options.verify_after_pass,
            options.record_timing,
        )


@dataclass(frozen=True)
class KernelBuildOptions:
    """Complete immutable snapshot of current JIT target and build inputs."""

    target_os: TargetOS
    target_arch: str
    optimization_level: int
    isa_policy: ISAPolicy
    fast_math: bool
    unroll_loops: bool
    parallel_backend: ParallelBackend
    legacy_parallel_policy: LegacyParallelPolicy
    legacy_lowering_policy: LegacyLoweringPolicy
    index_width: IndexWidthPolicy
    abi_policy: ABIPolicy
    compiler_abi_check_policy: CompilerABICheckPolicy
    compiler_wrapper_policy: CompilerWrapperPolicy
    compiler_wrapper_name: Optional[str]
    compiler_wrapper_path: Optional[str]
    darwin_toolchain: Optional[DarwinToolchainOptions]
    extra_cflags: Tuple[str, ...]
    direct_extension_cflags: Tuple[str, ...]
    special_kernel_cflags: Tuple[str, ...]
    extra_ldflags: Tuple[str, ...]
    jit_tune_hooks: bool
    torch_version: str
    torch_path: str
    torch_include_paths: Tuple[str, ...]
    torch_library_path: str
    torch_cxx11_abi: bool
    python_executable: str
    python_version: str
    python_cache_tag: str
    python_include_path: str
    scorch_python_path: str
    cxx_compiler: str
    cxx_compiler_path: str
    cxx_compiler_from_environment: bool
    executable_search_path: str
    preamble_source: str
    # What the build key holds in place of the PATH string and the two paths resolved
    # through it: the programs the build runs (toolchain_identity.py).  A snapshot made
    # without a probe holds those three strings instead, as the key did before.
    toolchain_identity: Tuple[object, ...] = ()
    toolchain_probe: str = ""

    def __post_init__(self) -> None:
        enum_fields = (
            ("target_os", self.target_os, TargetOS),
            ("isa_policy", self.isa_policy, ISAPolicy),
            ("parallel_backend", self.parallel_backend, ParallelBackend),
            (
                "legacy_parallel_policy",
                self.legacy_parallel_policy,
                LegacyParallelPolicy,
            ),
            (
                "legacy_lowering_policy",
                self.legacy_lowering_policy,
                LegacyLoweringPolicy,
            ),
            ("index_width", self.index_width, IndexWidthPolicy),
            ("abi_policy", self.abi_policy, ABIPolicy),
            (
                "compiler_abi_check_policy",
                self.compiler_abi_check_policy,
                CompilerABICheckPolicy,
            ),
            (
                "compiler_wrapper_policy",
                self.compiler_wrapper_policy,
                CompilerWrapperPolicy,
            ),
        )
        for field_name, enum_value, expected_type in enum_fields:
            if type(enum_value) is not expected_type:
                _raise_options_error(
                    "invalid_type",
                    f"build.{field_name}",
                    f"expected exact {expected_type.__name__}",
                )

        if self.target_os is TargetOS.DARWIN:
            if type(self.darwin_toolchain) is not DarwinToolchainOptions:
                _raise_options_error(
                    "invalid_darwin_toolchain",
                    "build.darwin_toolchain",
                    "Darwin builds require exact DarwinToolchainOptions",
                )
        elif self.darwin_toolchain is not None:
            _raise_options_error(
                "invalid_darwin_toolchain",
                "build.darwin_toolchain",
                "non-Darwin builds must not carry Darwin toolchain state",
            )

        if self.compiler_wrapper_policy is CompilerWrapperPolicy.DISABLED:
            if (
                self.compiler_wrapper_name is not None
                or self.compiler_wrapper_path is not None
            ):
                _raise_options_error(
                    "inconsistent_compiler_wrapper",
                    "build.compiler_wrapper_policy",
                    "a disabled wrapper policy must not carry a wrapper",
                )
        else:
            if (self.compiler_wrapper_name is None) != (
                self.compiler_wrapper_path is None
            ):
                _raise_options_error(
                    "inconsistent_compiler_wrapper",
                    "build.compiler_wrapper_path",
                    "wrapper name and path must both be present or both be absent",
                )
            if self.compiler_wrapper_name is not None:
                if type(self.compiler_wrapper_name) is not str:
                    _raise_options_error(
                        "invalid_type",
                        "build.compiler_wrapper_name",
                        "expected an exact string or None",
                    )
                if self.compiler_wrapper_name not in {"ccache", "sccache"}:
                    _raise_options_error(
                        "unsupported_compiler_wrapper",
                        "build.compiler_wrapper_name",
                        "only PyTorch's ccache and sccache wrappers are supported",
                    )
                if (
                    type(self.compiler_wrapper_path) is not str
                    or not os.path.isabs(self.compiler_wrapper_path)
                    or os.path.basename(self.compiler_wrapper_path)
                    != self.compiler_wrapper_name
                ):
                    _raise_options_error(
                        "invalid_compiler_wrapper_path",
                        "build.compiler_wrapper_path",
                        "the wrapper path must be absolute and match its name",
                    )

        _require_nonempty_string(self.target_arch, "build.target_arch")
        if type(self.optimization_level) is not int or self.optimization_level != 3:
            _raise_options_error(
                "unsupported_optimization_level",
                "build.optimization_level",
                "the current JIT supports exactly optimization level 3",
            )
        _require_exact_bool(self.fast_math, "build.fast_math")
        _require_exact_bool(self.unroll_loops, "build.unroll_loops")
        _require_exact_bool(self.jit_tune_hooks, "build.jit_tune_hooks")
        _require_exact_bool(self.torch_cxx11_abi, "build.torch_cxx11_abi")
        _require_exact_bool(
            self.cxx_compiler_from_environment,
            "build.cxx_compiler_from_environment",
        )
        if not self.fast_math or not self.unroll_loops:
            _raise_options_error(
                "unsupported_build_policy",
                "build",
                "the current JIT requires fast-math and loop unrolling",
            )
        for field_name, string_value in (
            ("torch_version", self.torch_version),
            ("torch_path", self.torch_path),
            ("torch_library_path", self.torch_library_path),
            ("python_executable", self.python_executable),
            ("python_version", self.python_version),
            ("python_cache_tag", self.python_cache_tag),
            ("python_include_path", self.python_include_path),
            ("scorch_python_path", self.scorch_python_path),
            ("cxx_compiler", self.cxx_compiler),
            ("cxx_compiler_path", self.cxx_compiler_path),
            ("executable_search_path", self.executable_search_path),
            ("preamble_source", self.preamble_source),
        ):
            _require_nonempty_string(string_value, f"build.{field_name}")
        if not os.path.isabs(self.cxx_compiler_path):
            _raise_options_error(
                "invalid_compiler_path",
                "build.cxx_compiler_path",
                "the snapshotted compiler path must be absolute",
            )
        if not os.path.isabs(self.scorch_python_path):
            _raise_options_error(
                "invalid_package_path",
                "build.scorch_python_path",
                "the Scorch import root must be absolute",
            )
        if not self.cxx_compiler_from_environment and self.cxx_compiler != "c++":
            _raise_options_error(
                "inconsistent_compiler_identity",
                "build.cxx_compiler_from_environment",
                "an implicit compiler command must be exactly 'c++'",
            )
        expected_compiler_name = self.cxx_compiler
        actual_compiler_name = (
            self.cxx_compiler_path
            if os.path.isabs(self.cxx_compiler)
            else os.path.basename(self.cxx_compiler_path)
        )
        if actual_compiler_name != expected_compiler_name:
            _raise_options_error(
                "inconsistent_compiler_identity",
                "build.cxx_compiler_path",
                "compiler command and resolved path must identify the same command",
            )

        cflags = _snapshot_string_tuple(self.extra_cflags, "build.extra_cflags")
        direct_extension_cflags = _snapshot_string_tuple(
            self.direct_extension_cflags,
            "build.direct_extension_cflags",
        )
        special_cflags = _snapshot_string_tuple(
            self.special_kernel_cflags,
            "build.special_kernel_cflags",
        )
        ldflags = _snapshot_string_tuple(self.extra_ldflags, "build.extra_ldflags")
        torch_include_paths = _snapshot_string_tuple(
            self.torch_include_paths,
            "build.torch_include_paths",
        )
        object.__setattr__(self, "extra_cflags", cflags)
        object.__setattr__(
            self,
            "direct_extension_cflags",
            direct_extension_cflags,
        )
        object.__setattr__(self, "special_kernel_cflags", special_cflags)
        object.__setattr__(self, "extra_ldflags", ldflags)
        object.__setattr__(self, "torch_include_paths", torch_include_paths)
        for field_name, path_value in (
            ("torch_path", self.torch_path),
            ("torch_library_path", self.torch_library_path),
            ("python_executable", self.python_executable),
            ("python_include_path", self.python_include_path),
        ):
            if not os.path.isabs(path_value):
                _raise_options_error(
                    "invalid_build_path",
                    f"build.{field_name}",
                    "snapshotted build paths must be absolute",
                )
        if any(not os.path.isabs(path) for path in torch_include_paths):
            _raise_options_error(
                "invalid_build_path",
                "build.torch_include_paths",
                "snapshotted Torch include paths must be absolute",
            )
        has_tune_define = "-DSCORCH_TUNE_HOOKS" in cflags
        direct_has_tune_define = "-DSCORCH_TUNE_HOOKS" in direct_extension_cflags
        if (
            has_tune_define != self.jit_tune_hooks
            or direct_has_tune_define != self.jit_tune_hooks
        ):
            _raise_options_error(
                "inconsistent_tune_hooks",
                "build.extra_cflags",
                "SCORCH_TUNE_HOOKS define must match jit_tune_hooks",
            )
        _validate_supported_build_flags(
            self,
            cflags,
            direct_extension_cflags,
            special_cflags,
            ldflags,
        )
        if (
            type(self.toolchain_identity) is not tuple
            or type(self.toolchain_probe) is not str
        ):
            _raise_options_error(
                "invalid_type",
                "build.toolchain_identity",
                "expected an exact tuple and an exact string",
            )
        if not self.toolchain_identity:
            from .toolchain_identity import unprobed_identity

            object.__setattr__(
                self,
                "toolchain_identity",
                unprobed_identity(
                    self.executable_search_path,
                    self.cxx_compiler_path,
                    self.compiler_wrapper_path,
                ),
            )
            object.__setattr__(
                self, "toolchain_probe", "unprobed: made without a probe"
            )
        elif not self.toolchain_probe:
            _raise_options_error(
                "invalid_toolchain_identity",
                "build.toolchain_probe",
                "a toolchain identity must say how it was found",
            )

    def build_child_environment(
        self, toolchain_directory: Optional[str] = None
    ) -> dict[str, str]:
        """The whole environment a build child runs in; nothing else reaches it.

        ``toolchain_directory`` (the launcher directory) goes in front of ``PATH``.
        The toolchain probe asks the driver under this environment without it.
        """

        return _build_child_environment(
            executable_search_path=self.executable_search_path,
            scorch_python_path=self.scorch_python_path,
            compiler_wrapper_policy=self.compiler_wrapper_policy,
            cxx_compiler=self.cxx_compiler,
            cxx_compiler_from_environment=self.cxx_compiler_from_environment,
            darwin_toolchain=self.darwin_toolchain,
            toolchain_directory=toolchain_directory,
        )

    @property
    def cache_key(self) -> tuple[object, ...]:
        # A forged non-``str`` preamble keeps the uncached spelling, so it fails
        # exactly as it did before the digest was kept.
        preamble_digest = (
            _preamble_digest(self.preamble_source)
            if type(self.preamble_source) is str
            else hashlib.sha256(self.preamble_source.encode()).hexdigest()
        )
        darwin_toolchain_key = (
            None if self.darwin_toolchain is None else self.darwin_toolchain.cache_key
        )
        return (
            self.target_os.value,
            self.target_arch,
            self.optimization_level,
            self.isa_policy.value,
            self.fast_math,
            self.unroll_loops,
            self.parallel_backend.value,
            self.legacy_parallel_policy.value,
            self.legacy_lowering_policy.value,
            self.index_width.value,
            self.abi_policy.value,
            self.compiler_abi_check_policy.value,
            self.compiler_wrapper_policy.value,
            self.compiler_wrapper_name,
            darwin_toolchain_key,
            self.extra_cflags,
            self.direct_extension_cflags,
            self.special_kernel_cflags,
            self.extra_ldflags,
            self.jit_tune_hooks,
            self.torch_version,
            self.torch_path,
            self.torch_include_paths,
            self.torch_library_path,
            self.torch_cxx11_abi,
            self.python_executable,
            self.python_version,
            self.python_cache_tag,
            self.python_include_path,
            self.scorch_python_path,
            self.cxx_compiler,
            self.cxx_compiler_from_environment,
            # PATH, and the driver's and wrapper's paths found through it, enter only
            # as the programs they resolve to.
            self.toolchain_identity,
            preamble_digest,
        )


@dataclass(frozen=True)
class CompileOptions:
    """One detached configuration snapshot shared by every compiler stage."""

    build: KernelBuildOptions
    requested_schedule: Optional["Schedule"] = None
    scheduler: SchedulerPolicy = field(default_factory=SchedulerPolicy)
    verification: VerificationPolicy = field(default_factory=VerificationPolicy)
    enabled_llir_passes: Tuple[LLIRPassId, ...] = CURRENT_LLIR_PASSES
    regblock_dual: bool = True
    emit_comments: bool = True

    def __post_init__(self) -> None:
        if self.requested_schedule is not None:
            from .scheduler import Schedule

            if type(self.requested_schedule) is not Schedule:
                _raise_options_error(
                    "invalid_type",
                    "requested_schedule",
                    "expected an exact Schedule or None",
                )
        if type(self.scheduler) is not SchedulerPolicy:
            _raise_options_error(
                "invalid_type", "scheduler", "expected an exact SchedulerPolicy"
            )
        if type(self.verification) is not VerificationPolicy:
            _raise_options_error(
                "invalid_type",
                "verification",
                "expected an exact VerificationPolicy",
            )
        if isinstance(
            self.enabled_llir_passes,
            (str, bytes, bytearray, set, frozenset, dict),
        ):
            _raise_options_error(
                "invalid_sequence",
                "enabled_llir_passes",
                "expected the ordered current pass sequence",
            )
        try:
            enabled_passes = tuple(self.enabled_llir_passes)
        except TypeError:
            _raise_options_error(
                "invalid_sequence",
                "enabled_llir_passes",
                "expected the ordered current pass sequence",
            )
        object.__setattr__(self, "enabled_llir_passes", enabled_passes)
        if (
            any(type(pass_id) is not LLIRPassId for pass_id in enabled_passes)
            or enabled_passes != CURRENT_LLIR_PASSES
        ):
            _raise_options_error(
                "unsupported_pass_pipeline",
                "enabled_llir_passes",
                "all seven current passes must be enabled in their current order",
            )
        _require_exact_bool(self.regblock_dual, "regblock_dual")
        _require_exact_bool(self.emit_comments, "emit_comments")
        if type(self.build) is not KernelBuildOptions:
            _raise_options_error(
                "invalid_type", "build", "expected exact KernelBuildOptions"
            )

    @classmethod
    def from_environment(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        *,
        requested_schedule: Optional["Schedule"] = None,
        forced_schedule: object = _USE_CONTEXT,
        regblock_override: object = _USE_CONTEXT,
        verify_cin_override: object = _USE_CONTEXT,
        llir_pass_options: LLIRPassOptions = PRODUCTION_LLIR_PASS_OPTIONS,
    ) -> "CompileOptions":
        """Parse process and context state exactly once at the public boundary.

        The parse (``_parse_environment``) is a function of what it reads: the
        environment variables it names, the three context overrides it consults when an
        argument says so, its arguments, and what the build snapshot is keyed by
        (``_snapshot_kernel_build_options``).  Every public call takes a snapshot, and
        the parse validates each value and builds the nested policy objects: 120 us of a
        390 us warm ``einsum`` call on sapling, whose kernel ran about 146 us (job
        82239).  So the parsed snapshot is kept under a key of exactly those inputs
        (``_environment_options_key``) and every call receives its own copy of it, as
        the build snapshot does.  A change in any input parses again; an input that the
        parse would refuse is not keyed at all, so the parse raises as it always did.
        ``clear_build_snapshot_cache()`` forgets these snapshots as well.
        """

        key, held_objects = _environment_options_key(
            cls,
            environ,
            requested_schedule,
            forced_schedule,
            regblock_override,
            verify_cin_override,
            llir_pass_options,
        )
        if key is None:
            return cls._parse_environment(
                environ,
                requested_schedule=requested_schedule,
                forced_schedule=forced_schedule,
                regblock_override=regblock_override,
                verify_cin_override=verify_cin_override,
                llir_pass_options=llir_pass_options,
            )
        held = _environment_options_cache.get(key)
        if held is not None and _same_objects(held[1], held_objects):
            return _own_copy_of_options(held[0])
        # The parse reads the values the key holds, not the mapping again: each variable
        # is read once, and the kept snapshot is the parse of its own key.
        _, values, *_, build_inputs = key
        kept = cls._parse_environment(
            _environment_copy(environ, values),
            requested_schedule=requested_schedule,
            forced_schedule=forced_schedule,
            regblock_override=regblock_override,
            verify_cin_override=verify_cin_override,
            llir_pass_options=llir_pass_options,
        )
        if _build_snapshot_inputs() != build_inputs:
            # The working directory or a header file changed while it parsed, so the
            # snapshot may belong to neither reading.  Use it, do not keep it.
            return kept
        _keep_cache_keys(kept)
        if len(_environment_options_cache) >= _BUILD_SNAPSHOT_CACHE_LIMIT:
            _environment_options_cache.clear()
        _environment_options_cache[key] = (kept, held_objects)
        return _own_copy_of_options(kept)

    @classmethod
    def _parse_environment(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        *,
        requested_schedule: Optional["Schedule"] = None,
        forced_schedule: object = _USE_CONTEXT,
        regblock_override: object = _USE_CONTEXT,
        verify_cin_override: object = _USE_CONTEXT,
        llir_pass_options: LLIRPassOptions = PRODUCTION_LLIR_PASS_OPTIONS,
    ) -> "CompileOptions":
        """The parse behind ``from_environment``, with nothing kept."""

        if environ is not None and not isinstance(environ, Mapping):
            _raise_options_error(
                "invalid_type", "environ", "expected a string-to-string Mapping"
            )
        source = os.environ if environ is None else environ

        def read_once(key: str) -> Optional[str]:
            value = source.get(key)
            if value is not None and type(value) is not str:
                _raise_options_error(
                    "invalid_environment_value",
                    key,
                    "environment values must be exact strings",
                )
            return value

        regblock_raw = read_once("SCORCH_REGBLOCK")
        max_n_raw = read_once("SCORCH_REGBLOCK_MAX_N")
        tile_width_raw = read_once("SCORCH_REGBLOCK_T")
        verify_cin_raw = read_once("SCORCH_VERIFY_CIN")
        regblock_dual_raw = read_once("SCORCH_REGBLOCK_DUAL")
        jit_tune_hooks_raw = read_once("SCORCH_JIT_TUNE_HOOKS")
        cxx_compiler_raw = read_once("CXX")
        executable_search_path_raw = read_once("PATH")
        sdk_root_raw = read_once("SDKROOT")
        developer_dir_raw = read_once("DEVELOPER_DIR")
        deployment_target_raw = read_once("MACOSX_DEPLOYMENT_TARGET")
        unsupported_compiler_environment = tuple(
            (key, read_once(key)) for key in _UNSUPPORTED_COMPILER_ENVIRONMENT
        )
        torch_disable_abi_check_raw = read_once("TORCH_DONT_CHECK_COMPILER_ABI")
        torch_no_wrapper_raw = read_once("TORCH_NO_COMPILER_WRAPPER")

        for key, value in unsupported_compiler_environment:
            if value is not None:
                _raise_options_error(
                    "unsupported_compiler_environment",
                    key,
                    "compiler search-path overrides are not supported",
                )
        if torch_disable_abi_check_raw is not None:
            disable_abi_check = _parse_environment_bool(
                torch_disable_abi_check_raw,
                "TORCH_DONT_CHECK_COMPILER_ABI",
                default=False,
            )
            if disable_abi_check:
                _raise_options_error(
                    "unsupported_compiler_abi_policy",
                    "TORCH_DONT_CHECK_COMPILER_ABI",
                    "production compiler ABI validation cannot be disabled",
                )
        compiler_wrapper_policy = (
            CompilerWrapperPolicy.DISABLED
            if torch_no_wrapper_raw
            else CompilerWrapperPolicy.AUTO
        )

        env_regblock = _parse_environment_bool(
            regblock_raw, "SCORCH_REGBLOCK", default=False
        )
        regblock_max_n = _parse_environment_positive_int(
            max_n_raw, "SCORCH_REGBLOCK_MAX_N", default=8
        )
        regblock_tile_width = (
            regblock_max_n
            if tile_width_raw is None
            else _parse_environment_positive_int(
                tile_width_raw, "SCORCH_REGBLOCK_T", default=regblock_max_n
            )
        )
        env_verify_cin = _parse_environment_bool(
            verify_cin_raw, "SCORCH_VERIFY_CIN", default=False
        )
        regblock_dual = _parse_environment_bool(
            regblock_dual_raw, "SCORCH_REGBLOCK_DUAL", default=True
        )
        jit_tune_hooks = _parse_environment_bool(
            jit_tune_hooks_raw, "SCORCH_JIT_TUNE_HOOKS", default=False
        )
        cxx_compiler = (
            "c++"
            if cxx_compiler_raw is None
            else _require_nonempty_string(cxx_compiler_raw, "CXX")
        )
        executable_search_path = (
            os.defpath
            if executable_search_path_raw is None
            else executable_search_path_raw
        )

        if regblock_override is _USE_CONTEXT:
            from .scheduler import get_forced_regblock

            context_regblock: Optional[bool] = get_forced_regblock()
        else:
            if regblock_override is None:
                context_regblock = None
            elif type(regblock_override) is bool:
                context_regblock = cast(bool, regblock_override)
            else:
                _raise_options_error(
                    "invalid_override",
                    "regblock_override",
                    "expected bool, None, or the context sentinel",
                )
        regblock_enabled = (
            env_regblock if context_regblock is None else cast(bool, context_regblock)
        )

        resolved_schedule: Optional["Schedule"]
        if requested_schedule is not None:
            resolved_schedule = requested_schedule
            if (
                forced_schedule is not _USE_CONTEXT
                and forced_schedule is not None
                and forced_schedule != requested_schedule
            ):
                _raise_options_error(
                    "conflicting_schedule",
                    "forced_schedule",
                    "requested and forced schedules disagree",
                )
        elif forced_schedule is _USE_CONTEXT:
            from .scheduler import get_forced_schedule

            resolved_schedule = get_forced_schedule()
        else:
            resolved_schedule = cast(Optional["Schedule"], forced_schedule)

        if verify_cin_override is _USE_CONTEXT:
            from .cin_analysis import get_full_cin_verification

            context_verify_cin = get_full_cin_verification()
            verify_cin = env_verify_cin or context_verify_cin
        else:
            verify_cin = _require_exact_bool(verify_cin_override, "verify_cin_override")

        build = _snapshot_kernel_build_options(
            jit_tune_hooks=jit_tune_hooks,
            cxx_compiler=cxx_compiler,
            cxx_compiler_from_environment=cxx_compiler_raw is not None,
            compiler_wrapper_policy=compiler_wrapper_policy,
            executable_search_path=executable_search_path,
            sdk_root_raw=sdk_root_raw,
            developer_dir_raw=developer_dir_raw,
            deployment_target_raw=deployment_target_raw,
        )
        return cls(
            requested_schedule=cast(Optional["Schedule"], resolved_schedule),
            scheduler=SchedulerPolicy(
                regblock_enabled=regblock_enabled,
                regblock_max_n=regblock_max_n,
                regblock_tile_width=regblock_tile_width,
            ),
            verification=VerificationPolicy(
                verify_cin=verify_cin,
                llir_pass_options=llir_pass_options,
            ),
            regblock_dual=regblock_dual,
            build=build,
        )

    def with_regblock_enabled(self, enabled: bool) -> "CompileOptions":
        """Return an independent typed snapshot for one explicit scheduler arm."""

        _require_exact_bool(enabled, "scheduler.regblock_enabled")
        return replace(
            self,
            scheduler=replace(self.scheduler, regblock_enabled=enabled),
        )

    @property
    def semantic_cache_key(self) -> tuple[object, ...]:
        schedule_key = (
            self.requested_schedule.cache_key
            if self.requested_schedule is not None
            else None
        )
        return (
            "compile_options_v1",
            schedule_key,
            self.scheduler.cache_key,
            tuple(pass_id.value for pass_id in self.enabled_llir_passes),
            self.regblock_dual,
            self.emit_comments,
            self.build.legacy_parallel_policy.value,
            self.build.legacy_lowering_policy.value,
            self.build.index_width.value,
            self.build.abi_policy.value,
        )

    @property
    def build_cache_key(self) -> tuple[object, ...]:
        return self.build.cache_key

    @property
    def cache_fingerprint(self) -> str:
        kept = _kept_cache_keys.get(id(self))
        if kept is not None and kept[0]() is self:
            return kept[2]
        return canonical_cache_digest(self.cache_key)

    @property
    def cache_key(self) -> tuple[object, ...]:
        # A snapshot kept by `from_environment` has its key, computed once when it was
        # kept (`_keep_cache_keys`), and its copies inherit it.
        kept = _kept_cache_keys.get(id(self))
        if kept is not None and kept[0]() is self:
            return kept[1]
        return (
            self.semantic_cache_key,
            self.verification.cache_key,
            self.build_cache_key,
        )


def _parse_environment_bool(
    raw: Optional[str], field_name: str, *, default: bool
) -> bool:
    if raw is None:
        return default
    if raw == "0":
        return False
    if raw == "1":
        return True
    _raise_options_error(
        "invalid_environment_boolean",
        field_name,
        "expected exactly '0' or '1'",
    )


def _validate_supported_build_flags(
    build: KernelBuildOptions,
    cflags: Tuple[str, ...],
    direct_extension_cflags: Tuple[str, ...],
    special_cflags: Tuple[str, ...],
    ldflags: Tuple[str, ...],
) -> None:
    """Reject flag spellings outside the current immutable build policy."""

    tune_suffix = ("-DSCORCH_TUNE_HOOKS",) if build.jit_tune_hooks else ()
    if build.target_os is TargetOS.DARWIN:
        production_tail = (
            f"-isystem{_DARWIN_CXX_INCLUDE}",
            "-Xpreprocessor",
            "-fopenmp",
        )
        direct_tail = (
            f"-isystem{_DARWIN_CXX_INCLUDE}",
            "-isysroot",
            _DARWIN_SDK_ROOT,
            "-Xpreprocessor",
            "-fopenmp",
        )
        include_tails = (
            (),
            ("-I/opt/homebrew/opt/libomp/include",),
            ("-I/usr/local/opt/libomp/include",),
        )
        supported_cflags = {
            _BASE_CFLAGS + production_tail + include_tail + tune_suffix
            for include_tail in include_tails
        }
        supported_direct_cflags = {
            _BASE_CFLAGS + direct_tail + include_tail + tune_suffix
            for include_tail in include_tails
        }
    else:
        supported_cflags = {_BASE_CFLAGS + ("-fopenmp",) + tune_suffix}
        supported_direct_cflags = supported_cflags
    if cflags not in supported_cflags:
        _raise_options_error(
            "unsupported_build_flags",
            "build.extra_cflags",
            "compiler flags must match one current target policy exactly",
        )
    if direct_extension_cflags not in supported_direct_cflags:
        _raise_options_error(
            "unsupported_build_flags",
            "build.direct_extension_cflags",
            "direct extension flags must match one current target policy exactly",
        )
    supported_special_cflags = {
        _SPECIAL_KERNEL_BASE_CFLAGS + flags[len(_BASE_CFLAGS) :]
        for flags in supported_cflags
    }
    if special_cflags not in supported_special_cflags:
        _raise_options_error(
            "unsupported_build_flags",
            "build.special_kernel_cflags",
            "special-kernel flags must match one current target policy exactly",
        )

    torch_lib_path = os.path.join(os.path.dirname(build.torch_path), "lib")
    rpath = f"-Wl,-rpath,{torch_lib_path}"
    if build.target_os is TargetOS.DARWIN:
        supported_ldflags = {
            (os.path.join(torch_lib_path, "libomp.dylib"), rpath),
            ("-lomp",),
            ("-lomp", "-L/opt/homebrew/opt/libomp/lib"),
            ("-lomp", "-L/usr/local/opt/libomp/lib"),
        }
        supported = ldflags in supported_ldflags
    else:
        supported = ldflags == ("-fopenmp",)
        if len(ldflags) == 2 and ldflags[1] == rpath:
            library_path = ldflags[0]
            supported = os.path.dirname(
                library_path
            ) == torch_lib_path and os.path.basename(library_path).startswith("libgomp")
    if not supported:
        _raise_options_error(
            "unsupported_link_policy",
            "build.extra_ldflags",
            "linker flags must match one current OpenMP target policy exactly",
        )


def _parse_environment_positive_int(
    raw: Optional[str],
    field_name: str,
    *,
    default: int,
    maximum: int = _INT32_MAX,
) -> int:
    if raw is None:
        return default
    if _POSITIVE_DECIMAL.fullmatch(raw) is None:
        _raise_options_error(
            "invalid_environment_integer",
            field_name,
            "expected a positive base-10 integer",
        )
    maximum_text = str(maximum)
    if len(raw) > len(maximum_text) or (
        len(raw) == len(maximum_text) and raw > maximum_text
    ):
        _raise_options_error(
            "integer_out_of_range",
            field_name,
            f"expected a value no greater than {maximum}",
        )
    return int(raw)


def _snapshot_darwin_toolchain(
    target_os: TargetOS,
    *,
    sdk_root_raw: Optional[str],
    developer_dir_raw: Optional[str],
    deployment_target_raw: Optional[str],
) -> Optional[DarwinToolchainOptions]:
    """Validate the one coherent Darwin toolchain supported by current flags."""

    if target_os is not TargetOS.DARWIN:
        for field_name, value in (
            ("SDKROOT", sdk_root_raw),
            ("DEVELOPER_DIR", developer_dir_raw),
            ("MACOSX_DEPLOYMENT_TARGET", deployment_target_raw),
        ):
            if value is not None:
                _raise_options_error(
                    "unsupported_target_environment",
                    field_name,
                    "Darwin target variables are not supported on this host",
                )
        return None

    sdk_root = (
        _DARWIN_SDK_ROOT
        if sdk_root_raw is None
        else _require_nonempty_string(sdk_root_raw, "SDKROOT")
    )
    developer_dir = (
        _DARWIN_DEVELOPER_DIR
        if developer_dir_raw is None
        else _require_nonempty_string(developer_dir_raw, "DEVELOPER_DIR")
    )
    deployment_target = (
        None
        if deployment_target_raw is None
        else _require_nonempty_string(
            deployment_target_raw,
            "MACOSX_DEPLOYMENT_TARGET",
        )
    )
    return DarwinToolchainOptions(
        developer_dir=developer_dir,
        sdk_root=sdk_root,
        deployment_target=deployment_target,
    )


# Resolved build snapshots, keyed by everything their resolution reads that is cheap to
# read again (see ``_snapshot_kernel_build_options``).  One entry in ordinary use; the
# limit bounds a process that keeps changing its compiler environment.
_BUILD_SNAPSHOT_CACHE_LIMIT = 16
_build_snapshot_cache: dict[tuple[object, ...], KernelBuildOptions] = {}


def clear_build_snapshot_cache() -> None:
    """Forget the resolved build snapshots, so the next public call probes the host.

    The parsed option snapshots that hold them (``CompileOptions.from_environment``)
    are forgotten with them, so the next public call parses again as well.  So are the
    cache keys of the parses that succeeded (``parsed_options_key``), so a call that
    skipped the parse makes it again.

    Needed only after a change the cache key cannot see: a compiler or compiler
    wrapper installed, removed or replaced in a directory already on ``PATH``, or a
    change to the files in Torch's library directory, inside one process.
    """

    _build_snapshot_cache.clear()
    _environment_options_cache.clear()
    _parsed_options_keys.clear()


def _own_copy_of_build_options(kept: KernelBuildOptions) -> KernelBuildOptions:
    """Return a new instance equal to ``kept`` without validating its fields again.

    The fields are immutable values that were validated when ``kept`` was built.
    Every public call runs this, so it copies the instance dictionary directly
    instead of going through the ``copy`` protocol.
    """

    own = object.__new__(KernelBuildOptions)
    own.__dict__.update(kept.__dict__)
    toolchain = kept.darwin_toolchain
    if toolchain is not None:
        own_toolchain = object.__new__(DarwinToolchainOptions)
        own_toolchain.__dict__.update(toolchain.__dict__)
        own.__dict__["darwin_toolchain"] = own_toolchain
    return own


# Kept results of ``CompileOptions.from_environment``: its key
# (``_environment_options_key``) -> (the snapshot, the objects whose identity is in the
# key).  One entry in ordinary use; bounded like the build snapshots, which these
# contain.
_environment_options_cache: dict[
    tuple[object, ...], tuple["CompileOptions", tuple[object, ...]]
] = {}

# Every variable ``CompileOptions._parse_environment`` reads from its mapping, in its
# order.
_ENVIRONMENT_OPTION_KEYS = (
    "SCORCH_REGBLOCK",
    "SCORCH_REGBLOCK_MAX_N",
    "SCORCH_REGBLOCK_T",
    "SCORCH_VERIFY_CIN",
    "SCORCH_REGBLOCK_DUAL",
    "SCORCH_JIT_TUNE_HOOKS",
    "CXX",
    "PATH",
    "SDKROOT",
    "DEVELOPER_DIR",
    "MACOSX_DEPLOYMENT_TARGET",
    *_UNSUPPORTED_COMPILER_ENVIRONMENT,
    "TORCH_DONT_CHECK_COMPILER_ABI",
    "TORCH_NO_COMPILER_WRAPPER",
)

# (scheduler, cin_analysis, utils, torch), imported at the first public call.  The
# modules, not their functions, so that a function replaced on its module is the one
# called.  ``environment_reuse_key`` reads the two context variables on the first two
# itself rather than through their reader functions.
_late_modules_memo: Optional[tuple[object, ...]] = None


def _late_modules() -> tuple:
    global _late_modules_memo
    if _late_modules_memo is None:
        import torch

        from .. import utils
        from . import cin_analysis, scheduler

        _late_modules_memo = (scheduler, cin_analysis, utils, torch)
    return _late_modules_memo


# (os.environ's key encoder, ``_ENVIRONMENT_OPTION_KEYS`` encoded by it).
_encoded_option_keys_memo: Optional[tuple[object, tuple[object, ...]]] = None


def _environment_values(source: Mapping[str, str]) -> Optional[tuple[object, ...]]:
    """The value of every variable the parse reads from ``source``, or None.

    Equal values give equal results.  None when a value is not a string: the parse
    refuses that mapping, and the caller then lets the parse raise.

    ``os.environ.get`` decodes the value and, for a variable that is not set, raises
    and catches ``KeyError`` inside ``Mapping.get``.  Reading the 22 variables that
    way cost 19.4 us on sapling (job 82687), most of them unset.  ``os.environ`` keeps
    its variables encoded in a dict (``os._Environ._data``), and its encoding is one to
    one, so for ``os.environ`` the encoded values are read from that dict instead.  They
    are tagged, so they never compare equal to decoded ones.  Any other mapping is read
    through ``get``, as the parse reads it.
    """

    global _encoded_option_keys_memo
    if type(source) is os._Environ:
        data = getattr(source, "_data", None)
        encode = getattr(source, "encodekey", None)
        if type(data) is dict and encode is not None:
            memo = _encoded_option_keys_memo
            if memo is None or memo[0] is not encode:
                memo = (encode, tuple(map(encode, _ENVIRONMENT_OPTION_KEYS)))
                _encoded_option_keys_memo = memo
            return ("encoded", tuple(map(data.get, memo[1])))
    try:
        values = tuple(map(source.get, _ENVIRONMENT_OPTION_KEYS))
    except Exception:
        return None
    for value in values:
        if value is not None and type(value) is not str:
            return None
    return ("decoded", values)


def _environment_copy(
    environ: Optional[Mapping[str, str]], values: object
) -> dict[str, str]:
    """The variables ``values`` holds (``_environment_values``) as a mapping to parse.

    The parse reads exactly ``_ENVIRONMENT_OPTION_KEYS``, so it reads the same values
    from this copy as from the mapping they were read from.  Values read from
    ``os.environ``'s store are decoded as ``os.environ`` decodes them.
    """

    tag, read = cast(tuple[str, tuple[Optional[str], ...]], values)
    if tag == "encoded":
        source = os.environ if environ is None else environ
        decode = source.decodevalue  # type: ignore[attr-defined]
        read = tuple(None if value is None else decode(value) for value in read)
    return {
        key: value
        for key, value in zip(_ENVIRONMENT_OPTION_KEYS, read)
        if value is not None
    }


def environment_reuse_key() -> Optional[tuple[object, ...]]:
    """What a call must match to run a kernel built under an earlier call's options.

    ``CompileOptions.from_environment`` reads three kinds of input:
    - environment variables;
    - the scheduler's and the verifier's context overrides;
    - the host: the working directory, the interpreter, Torch, the files on ``PATH``
      and the preamble's header files.
    The first two decide which kernel a call selects and whether its options are
    refused.  The host decides only how a new kernel is built: a kernel that is built
    and loaded runs the same whatever the host holds now.  So a call whose operation
    and these values equal an earlier call's may run that call's kernel without making
    options, and this key holds exactly these values.  The next build makes its
    options in full, so it sees any change to the host.

    Making the options on such a call cost 85 us of a 329 us warm ``einsum`` on sapling
    (job 82682), even with the parse kept (``from_environment``), because the parse's
    key reads the host too.  This key took about 2 us there (job 82687's parts).

    None when a value is one the parse would refuse to read.  The caller then makes
    its options in full and lets the parse raise.

    Every public call that reaches a generated kernel reads this key, so on its usual
    path it calls no other Python function: once ``_environment_values`` has encoded
    the variable names, ``os.environ``'s store is read here as that function reads it,
    and the two overrides are read from their context variables
    (``scheduler._REGBLOCK_FORCE``, ``cin_analysis._VERIFY_CIN_CONTEXT``), with the
    checks their readers (``get_forced_regblock``, ``get_full_cin_verification``)
    make.  A value a reader would refuse gives None here, as the reader's TypeError
    did.  The four helper calls this replaces were four of the frames a warm call made
    beyond 864745d's.
    """

    source = os.environ
    memo = _encoded_option_keys_memo
    data = getattr(source, "_data", None)
    if (
        memo is not None
        and type(source) is os._Environ
        and type(data) is dict
        and getattr(source, "encodekey", None) is memo[0]
    ):
        read = tuple(map(data.get, memo[1]))
        values: Optional[tuple[object, ...]] = ("encoded", read)
    else:
        values = _environment_values(source)
        if values is None:
            return None
    modules = _late_modules_memo
    if modules is None:
        modules = _late_modules()
    regblock = modules[0]._REGBLOCK_FORCE.get()  # type: ignore[attr-defined]
    verify = modules[1]._VERIFY_CIN_CONTEXT.get()  # type: ignore[attr-defined]
    if regblock is not None and type(regblock) is not bool:
        return None
    if type(verify) is not bool:
        return None
    return (values, regblock, verify)


# The cache fingerprint of the snapshot each successful parse made, by the reuse key
# (``environment_reuse_key``) it was begun under and the cache key of the schedule it
# resolved.  Bounded as the snapshot caches are, and forgotten with them.
_parsed_options_keys: dict[tuple[object, ...], str] = {}


def parsed_options_key(
    key: tuple[object, ...], schedule_key: Optional[str]
) -> Optional[str]:
    """The cache fingerprint of the snapshot a parse under ``key`` made here, or None.

    ``schedule_key`` is the cache key of the schedule the parse resolved, or None.  A
    parse under such a key would succeed again and make a snapshot with this
    fingerprint: its other inputs are the host's, which this process treats as fixed
    until ``clear_build_snapshot_cache()``.  A call may then name its options by this
    fingerprint and skip the parse until it needs the snapshot itself.  The fingerprint
    is kept rather than the cache key so that the call's dispatch key hashes one string,
    whose hash Python keeps, and not a nested tuple (``ops._DeferredOptions``).
    """

    return _parsed_options_keys.get((key, schedule_key))


def note_environment_parsed(key: tuple[object, ...], options: "CompileOptions") -> None:
    """Record the cache fingerprint of ``options``, made by a parse begun under ``key``.

    Nothing is recorded when the key changed while the parse ran: the parse may then
    have read either value.
    """

    if environment_reuse_key() != key:
        return
    if len(_parsed_options_keys) >= _BUILD_SNAPSHOT_CACHE_LIMIT:
        _parsed_options_keys.clear()
    schedule = options.requested_schedule
    schedule_key = schedule.cache_key if schedule is not None else None
    _parsed_options_keys[(key, schedule_key)] = options.cache_fingerprint


def _build_snapshot_inputs() -> Optional[tuple[object, ...]]:
    """The parts of the build snapshot's key that are not environment variables.

    The values ``_snapshot_kernel_build_options`` puts in its key besides its arguments,
    read the same way.  None when the preamble's files have no identity: that function
    then resolves on every call, so the parse is not kept either.
    """

    _, _, utils, torch = _late_modules()
    preamble_identity = utils.jit_preamble_file_identity()
    if preamble_identity is None:
        return None
    try:
        working_directory: Optional[str] = os.getcwd()
    except OSError:
        working_directory = None
    return (
        working_directory,
        platform.system(),
        platform.machine(),
        sys.executable,
        str(torch.__version__),
        torch.__file__,
        preamble_identity,
    )


def _environment_options_key(
    cls: type,
    environ: Optional[Mapping[str, str]],
    requested_schedule: object,
    forced_schedule: object,
    regblock_override: object,
    verify_cin_override: object,
    llir_pass_options: object,
) -> Tuple[Optional[tuple[object, ...]], tuple[object, ...]]:
    """What ``_parse_environment`` reads, as a key, and the objects named by identity.

    The key holds the value of every variable the parse reads, each context override it
    consults (only when its argument says to, as the parse does), the identity of each
    object argument, and the rest of the build snapshot's key
    (``_build_snapshot_inputs``).  The second value holds the objects whose identity is
    in the key, so that a hit can check the key's ids still name them.  An input the
    parse would refuse (a mapping that is not one, a value that is not an exact string,
    an override of the wrong type, a context reader that raises) gives no key, and the
    parse then raises as it did, in its own order.
    """

    if environ is None:
        source: Mapping[str, str] = os.environ
    elif isinstance(environ, Mapping):
        source = environ
    else:
        return None, ()
    values = _environment_values(source)
    if values is None:
        return None, ()
    scheduler, cin_analysis, _, _ = _late_modules()
    schedule_object: object = None
    try:
        if regblock_override is _USE_CONTEXT:
            forced_regblock = scheduler.get_forced_regblock()
            if forced_regblock is not None and type(forced_regblock) is not bool:
                return None, ()
            regblock: tuple[object, ...] = ("context", forced_regblock)
        elif regblock_override is None or type(regblock_override) is bool:
            regblock = ("given", regblock_override)
        else:
            return None, ()
        if requested_schedule is not None:
            schedule: tuple[object, ...] = (
                ("requested",)
                if forced_schedule is _USE_CONTEXT
                else ("given", id(forced_schedule))
            )
            if forced_schedule is not _USE_CONTEXT:
                schedule_object = forced_schedule
        elif forced_schedule is _USE_CONTEXT:
            schedule_object = scheduler.get_forced_schedule()
            schedule = ("context", id(schedule_object))
        else:
            schedule_object = forced_schedule
            schedule = ("given", id(forced_schedule))
        if verify_cin_override is _USE_CONTEXT:
            context_verify = cin_analysis.get_full_cin_verification()
            if type(context_verify) is not bool:
                return None, ()
            verify: tuple[object, ...] = ("context", context_verify)
        elif type(verify_cin_override) is bool:
            verify = ("given", verify_cin_override)
        else:
            return None, ()
        build_inputs = _build_snapshot_inputs()
    except Exception:
        return None, ()
    if build_inputs is None:
        return None, ()
    key = (
        cls,
        values,
        regblock,
        schedule,
        verify,
        id(requested_schedule),
        id(llir_pass_options),
        build_inputs,
    )
    return key, (requested_schedule, schedule_object, llir_pass_options)


def _same_objects(held: tuple[object, ...], now: tuple[object, ...]) -> bool:
    """Whether the key's ids still name the objects the kept entry was made from."""

    return len(held) == len(now) and all(a is b for a, b in zip(held, now))


# The cache key and fingerprint of each snapshot ``from_environment`` kept, and of each
# copy it handed out, by the instance's id.  Held beside the instances, not in their
# dictionaries: a snapshot's instance dictionary holds exactly its fields, which the
# new compiler's request identity requires (``plan_identity.require_exact_carrier``).
# An entry goes when its instance is collected.
_kept_cache_keys: dict[
    int, tuple["weakref.ref[CompileOptions]", tuple[object, ...], str]
] = {}


def _note_kept_keys(
    options: "CompileOptions", key: tuple[object, ...], fingerprint: str
) -> None:
    ident = id(options)

    def forget(ref: "weakref.ref[CompileOptions]") -> None:
        entry = _kept_cache_keys.get(ident)
        if entry is not None and entry[0] is ref:
            del _kept_cache_keys[ident]

    _kept_cache_keys[ident] = (weakref.ref(options, forget), key, fingerprint)


def _keep_cache_keys(options: "CompileOptions") -> None:
    """Record the snapshot's cache key and fingerprint, for it and its copies.

    Called once, on a snapshot about to be kept.  Its fields are frozen and nothing
    replaces them afterwards, so the key computed now stays its key.
    """

    key = options.cache_key
    _note_kept_keys(options, key, canonical_cache_digest(key))


def _own_copy_of_options(kept: "CompileOptions") -> "CompileOptions":
    """Return a new snapshot equal to ``kept`` without parsing or validating again.

    As ``_own_copy_of_build_options``: every field was validated when ``kept`` was built
    and is an immutable value.  The scheduler policy (with its cost model), the
    verification policy and the build snapshot are copied as well, so no two callers
    share an instance the snapshot owns.  The schedule and the pass options stay the
    objects the caller passed, as the parse leaves them.
    """

    own = object.__new__(type(kept))
    state = own.__dict__
    state.update(kept.__dict__)
    scheduler = object.__new__(SchedulerPolicy)
    scheduler.__dict__.update(kept.scheduler.__dict__)
    cost_model = object.__new__(SchedulerCostModel)
    cost_model.__dict__.update(kept.scheduler.cost_model.__dict__)
    scheduler.__dict__["cost_model"] = cost_model
    state["scheduler"] = scheduler
    verification = object.__new__(VerificationPolicy)
    verification.__dict__.update(kept.verification.__dict__)
    state["verification"] = verification
    state["build"] = _own_copy_of_build_options(kept.build)
    entry = _kept_cache_keys.get(id(kept))
    if entry is not None and entry[0]() is kept:
        _note_kept_keys(own, entry[1], entry[2])
    return own


def _snapshot_kernel_build_options(
    *,
    jit_tune_hooks: bool,
    cxx_compiler: str,
    cxx_compiler_from_environment: bool,
    compiler_wrapper_policy: CompilerWrapperPolicy,
    executable_search_path: str,
    sdk_root_raw: Optional[str],
    developer_dir_raw: Optional[str],
    deployment_target_raw: Optional[str],
) -> KernelBuildOptions:
    """Return the build snapshot for these inputs, probing the host only when needed.

    Resolving a snapshot reads the preamble's header files, lists Torch's library
    directory and searches ``PATH`` three times.  That took milliseconds on every
    public call, where a small cached kernel runs in a fraction of one.  A resolved
    snapshot is therefore kept for the process under a key that holds everything the
    resolution reads that is cheap to read again: the arguments (they carry every
    environment variable involved), the working directory (a relative ``CXX`` and
    relative ``PATH`` entries resolve against it), the platform, the interpreter,
    Torch's version and location, and the identity of the preamble's files (path,
    size, modification time, inode).  A change in any of them resolves again.

    Not read again while the key is unchanged: which files the ``PATH`` directories
    and Torch's library directory contain, and Python's include directory.  A
    compiler or compiler wrapper that appears, disappears or is replaced under an
    unchanged ``PATH`` is seen by the next process, or after
    ``clear_build_snapshot_cache()``.  Until then a kernel that is already built
    still runs, and a new build uses the compiler path the snapshot holds.  The build
    child still compares the snapshot with its own host before it compiles
    (``utils._verify_snapshotted_build_runtime``) and refuses the build if the
    wrapper lookup, Torch or Python differ.  The Darwin toolchain above is resolved
    once per process in the same way.

    Every call returns its own copy of the kept snapshot.  No two callers share an
    instance, so what one caller does to its object cannot reach the next caller.
    """

    import torch

    from ..utils import jit_preamble_file_identity

    def resolve() -> KernelBuildOptions:
        return _resolve_kernel_build_options(
            jit_tune_hooks=jit_tune_hooks,
            cxx_compiler=cxx_compiler,
            cxx_compiler_from_environment=cxx_compiler_from_environment,
            compiler_wrapper_policy=compiler_wrapper_policy,
            executable_search_path=executable_search_path,
            sdk_root_raw=sdk_root_raw,
            developer_dir_raw=developer_dir_raw,
            deployment_target_raw=deployment_target_raw,
        )

    preamble_identity = jit_preamble_file_identity()
    if preamble_identity is None:
        # The package is not stored as plain files, so nothing short of reading the
        # preamble says whether it changed: resolve every time, as before.
        return resolve()
    try:
        working_directory: Optional[str] = os.getcwd()
    except OSError:
        working_directory = None
    key = (
        jit_tune_hooks,
        cxx_compiler,
        cxx_compiler_from_environment,
        compiler_wrapper_policy,
        executable_search_path,
        sdk_root_raw,
        developer_dir_raw,
        deployment_target_raw,
        working_directory,
        platform.system(),
        platform.machine(),
        sys.executable,
        str(torch.__version__),
        torch.__file__,
        preamble_identity,
    )
    kept = _build_snapshot_cache.get(key)
    if kept is None:
        kept = resolve()
        if jit_preamble_file_identity() != preamble_identity:
            # A preamble file changed while the snapshot was being resolved, so its
            # text may be newer than the identity in the key.  Use it, do not keep it.
            return kept
        if len(_build_snapshot_cache) >= _BUILD_SNAPSHOT_CACHE_LIMIT:
            _build_snapshot_cache.clear()
        _build_snapshot_cache[key] = kept
    return _own_copy_of_build_options(kept)


def _resolve_kernel_build_options(
    *,
    jit_tune_hooks: bool,
    cxx_compiler: str,
    cxx_compiler_from_environment: bool,
    compiler_wrapper_policy: CompilerWrapperPolicy,
    executable_search_path: str,
    sdk_root_raw: Optional[str],
    developer_dir_raw: Optional[str],
    deployment_target_raw: Optional[str],
) -> KernelBuildOptions:
    """Resolve platform, package, and Torch build inputs without global storage."""

    import torch

    from ..utils import jit_preamble_text

    system = platform.system()
    if system == TargetOS.DARWIN.value:
        target_os = TargetOS.DARWIN
    elif system == TargetOS.LINUX.value:
        target_os = TargetOS.LINUX
    else:
        _raise_options_error(
            "unsupported_target_os",
            "build.target_os",
            f"unsupported target operating system {system!r}",
        )
    darwin_toolchain = _snapshot_darwin_toolchain(
        target_os,
        sdk_root_raw=sdk_root_raw,
        developer_dir_raw=developer_dir_raw,
        deployment_target_raw=deployment_target_raw,
    )
    target_arch = _require_nonempty_string(platform.machine(), "build.target_arch")
    if any(character.isspace() for character in cxx_compiler):
        _raise_options_error(
            "unsupported_compiler_command",
            "CXX",
            "compiler wrappers or commands with whitespace are not supported",
        )
    if "\x00" in cxx_compiler or "\x00" in executable_search_path:
        _raise_options_error(
            "unsupported_compiler_command",
            "CXX",
            "compiler command and PATH must not contain NUL bytes",
        )

    compiler_wrapper_name: Optional[str] = None
    compiler_wrapper_path: Optional[str] = None
    if compiler_wrapper_policy is CompilerWrapperPolicy.AUTO:
        for wrapper_name in ("ccache", "sccache"):
            try:
                resolved_wrapper = shutil.which(
                    wrapper_name,
                    path=executable_search_path,
                )
            except (OSError, ValueError):
                _raise_options_error(
                    "unsupported_compiler_command",
                    "PATH",
                    "PATH must be a valid process string",
                )
            if resolved_wrapper is not None:
                compiler_wrapper_name = wrapper_name
                compiler_wrapper_path = os.path.abspath(resolved_wrapper)
                break
    normalized_cxx = (
        os.path.abspath(cxx_compiler) if os.path.dirname(cxx_compiler) else cxx_compiler
    )
    try:
        resolved_cxx = shutil.which(normalized_cxx, path=executable_search_path)
    except (OSError, ValueError):
        _raise_options_error(
            "unsupported_compiler_command",
            "CXX",
            "compiler command and PATH must be valid process strings",
        )
    if resolved_cxx is None:
        _raise_options_error(
            "compiler_not_found",
            "CXX",
            f"could not resolve compiler command {cxx_compiler!r}",
        )

    raw_torch_version = torch.__version__
    torch_version = _require_nonempty_string(
        str(raw_torch_version),
        "build.torch_version",
    )
    torch_path = _require_nonempty_string(
        torch.__file__,
        "build.torch_path",
    )
    torch_lib_path = os.path.join(os.path.dirname(torch_path), "lib")
    torch_include_path = os.path.join(os.path.dirname(torch_path), "include")
    torch_include_paths = (
        torch_include_path,
        os.path.join(torch_include_path, "torch", "csrc", "api", "include"),
    )
    python_executable = _require_nonempty_string(
        sys.executable,
        "build.python_executable",
    )
    python_version = _require_nonempty_string(
        platform.python_version(),
        "build.python_version",
    )
    raw_cache_tag = sys.implementation.cache_tag
    python_cache_tag = _require_nonempty_string(
        raw_cache_tag,
        "build.python_cache_tag",
    )
    raw_python_include_path = sysconfig.get_path(
        "include",
        scheme="posix_prefix",
    )
    python_include_path = _require_nonempty_string(
        raw_python_include_path,
        "build.python_include_path",
    )
    scorch_python_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    torch_cxx11_abi = bool(torch._C._GLIBCXX_USE_CXX11_ABI)

    cflags = list(_BASE_CFLAGS)
    direct_extension_cflags = list(_BASE_CFLAGS)
    ldflags: list[str] = []
    if target_os is TargetOS.DARWIN:
        if darwin_toolchain is None:
            _raise_options_error(
                "invalid_darwin_toolchain",
                "build.darwin_toolchain",
                "Darwin build options require a toolchain snapshot",
            )
        cflags.append(f"-isystem{darwin_toolchain.sdk_root}/usr/include/c++/v1")
        direct_extension_cflags.append(
            f"-isystem{darwin_toolchain.sdk_root}/usr/include/c++/v1"
        )
        cflags.extend(("-Xpreprocessor", "-fopenmp"))
        direct_extension_cflags.extend(
            (
                "-isysroot",
                darwin_toolchain.sdk_root,
                "-Xpreprocessor",
                "-fopenmp",
            )
        )
        for header_path in (
            "/opt/homebrew/opt/libomp/include",
            "/usr/local/opt/libomp/include",
        ):
            if os.path.exists(header_path):
                cflags.append(f"-I{header_path}")
                direct_extension_cflags.append(f"-I{header_path}")
                break

        torch_omp = os.path.join(torch_lib_path, "libomp.dylib")
        if os.path.exists(torch_omp):
            ldflags.extend((torch_omp, f"-Wl,-rpath,{torch_lib_path}"))
        else:
            for library_path in (
                "/opt/homebrew/opt/libomp/lib",
                "/usr/local/opt/libomp/lib",
            ):
                if os.path.exists(library_path):
                    ldflags.extend(("-lomp", f"-L{library_path}"))
                    break
            else:
                ldflags.append("-lomp")
    else:
        cflags.append("-fopenmp")
        direct_extension_cflags.append("-fopenmp")
        gomp_libraries = glob.glob(os.path.join(torch_lib_path, "libgomp*.so*"))
        if gomp_libraries:
            ldflags.extend((gomp_libraries[0], f"-Wl,-rpath,{torch_lib_path}"))
        else:
            ldflags.append("-fopenmp")

    if jit_tune_hooks:
        cflags.append("-DSCORCH_TUNE_HOOKS")
        direct_extension_cflags.append("-DSCORCH_TUNE_HOOKS")

    special_cflags = list(_SPECIAL_KERNEL_BASE_CFLAGS)
    special_cflags.extend(cflags[len(_BASE_CFLAGS) :])

    preamble_source = jit_preamble_text()
    from .toolchain_identity import toolchain_identity

    cxx_compiler_path = os.path.abspath(resolved_cxx)
    identity, probe = toolchain_identity(
        cxx_compiler=normalized_cxx,
        cxx_compiler_path=cxx_compiler_path,
        compiler_wrapper_path=compiler_wrapper_path,
        environment=_build_child_environment(
            executable_search_path=executable_search_path,
            scorch_python_path=scorch_python_path,
            compiler_wrapper_policy=compiler_wrapper_policy,
            cxx_compiler=normalized_cxx,
            cxx_compiler_from_environment=cxx_compiler_from_environment,
            darwin_toolchain=darwin_toolchain,
        ),
        cflags=tuple(cflags),
        ldflags=tuple(ldflags),
        memo_directory=toolchain_memo_directory(),
    )
    return KernelBuildOptions(
        target_os=target_os,
        target_arch=target_arch,
        optimization_level=3,
        isa_policy=ISAPolicy.NATIVE,
        fast_math=True,
        unroll_loops=True,
        parallel_backend=ParallelBackend.OPENMP,
        legacy_parallel_policy=LegacyParallelPolicy.CURRENT,
        legacy_lowering_policy=LegacyLoweringPolicy.CURRENT,
        index_width=IndexWidthPolicy.INT32,
        abi_policy=ABIPolicy.TORCH_CPP_EXTENSION,
        compiler_abi_check_policy=CompilerABICheckPolicy.REQUIRED,
        compiler_wrapper_policy=compiler_wrapper_policy,
        compiler_wrapper_name=compiler_wrapper_name,
        compiler_wrapper_path=compiler_wrapper_path,
        darwin_toolchain=darwin_toolchain,
        extra_cflags=tuple(cflags),
        direct_extension_cflags=tuple(direct_extension_cflags),
        special_kernel_cflags=tuple(special_cflags),
        extra_ldflags=tuple(ldflags),
        jit_tune_hooks=jit_tune_hooks,
        torch_version=torch_version,
        torch_path=torch_path,
        torch_include_paths=torch_include_paths,
        torch_library_path=torch_lib_path,
        torch_cxx11_abi=torch_cxx11_abi,
        python_executable=python_executable,
        python_version=python_version,
        python_cache_tag=python_cache_tag,
        python_include_path=python_include_path,
        scorch_python_path=scorch_python_path,
        cxx_compiler=normalized_cxx,
        cxx_compiler_path=cxx_compiler_path,
        cxx_compiler_from_environment=cxx_compiler_from_environment,
        executable_search_path=executable_search_path,
        preamble_source=preamble_source,
        toolchain_identity=identity,
        toolchain_probe=probe,
    )


def _build_child_environment(
    *,
    executable_search_path: str,
    scorch_python_path: str,
    compiler_wrapper_policy: CompilerWrapperPolicy,
    cxx_compiler: str,
    cxx_compiler_from_environment: bool,
    darwin_toolchain: Optional[DarwinToolchainOptions],
    toolchain_directory: Optional[str] = None,
) -> dict[str, str]:
    """See ``KernelBuildOptions.build_child_environment``."""

    search_path = executable_search_path
    if toolchain_directory is not None:
        search_path = toolchain_directory + os.pathsep + search_path
    environment = {"PATH": search_path, "PYTHONPATH": scorch_python_path}
    if compiler_wrapper_policy is CompilerWrapperPolicy.DISABLED:
        environment["TORCH_NO_COMPILER_WRAPPER"] = "1"
    if cxx_compiler_from_environment:
        environment["CXX"] = cxx_compiler
    if darwin_toolchain is not None:
        environment["DEVELOPER_DIR"] = darwin_toolchain.developer_dir
        environment["SDKROOT"] = darwin_toolchain.sdk_root
        if darwin_toolchain.deployment_target is not None:
            environment["MACOSX_DEPLOYMENT_TARGET"] = darwin_toolchain.deployment_target
    return environment


def _probe_toolchain(build: KernelBuildOptions) -> Tuple[Tuple[object, ...], str]:
    """The toolchain identity of ``build``'s programs, asked under its build environment."""

    from .toolchain_identity import toolchain_identity

    return toolchain_identity(
        cxx_compiler=build.cxx_compiler,
        cxx_compiler_path=build.cxx_compiler_path,
        compiler_wrapper_path=build.compiler_wrapper_path,
        environment=build.build_child_environment(),
        cflags=build.extra_cflags,
        ldflags=build.extra_ldflags,
        memo_directory=toolchain_memo_directory(),
    )


def toolchain_memo_directory() -> str:
    """Where kept toolchain probes live: in the extensions root torch builds under.

    It is a cache, so its place changes no result.  It is found the way torch finds
    the root (``TORCH_EXTENSIONS_DIR``, else torch's user cache directory) without
    importing ``torch.utils.cpp_extension``, which costs about 80 ms and which a
    process that builds and loads no kernel does not otherwise import.
    """

    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root is None:
        import torch._appdirs

        root = os.path.realpath(
            torch._appdirs.user_cache_dir(appname="torch_extensions")
        )
    return os.path.join(root, "scorch_toolchain")
