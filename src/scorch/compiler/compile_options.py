"""Immutable configuration snapshot for one legacy compiler invocation.

Only the public compilation/runtime boundary constructs :class:`CompileOptions`.
Compiler stages receive that value explicitly and must not consult environment
variables, context-local overrides, platform probes, or mutable build-flag lists.
"""

from __future__ import annotations

import glob
import hashlib
import math
import os
import platform
import re
import shutil
import sys
import sysconfig
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
_DARWIN_DEVELOPER_DIR = "/Library/Developer/CommandLineTools"
_DARWIN_SDK_ROOT = f"{_DARWIN_DEVELOPER_DIR}/SDKs/MacOSX.sdk"
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
            raise TypeError(
                "cache identity supports only None, exact scalars, and tuples"
            )

    visit(value)
    return digest.hexdigest()


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

    @property
    def cache_key(self) -> tuple[object, ...]:
        preamble_digest = hashlib.sha256(self.preamble_source.encode()).hexdigest()
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
            self.compiler_wrapper_path,
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
            self.cxx_compiler_path,
            self.cxx_compiler_from_environment,
            self.executable_search_path,
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
        if enabled_passes != CURRENT_LLIR_PASSES or any(
            type(pass_id) is not LLIRPassId for pass_id in enabled_passes
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
        """Parse process and context state exactly once at the public boundary."""

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
        return canonical_cache_digest(self.cache_key)

    @property
    def cache_key(self) -> tuple[object, ...]:
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
        cxx_compiler_path=os.path.abspath(resolved_cxx),
        cxx_compiler_from_environment=cxx_compiler_from_environment,
        executable_search_path=executable_search_path,
        preamble_source=preamble_source,
    )
