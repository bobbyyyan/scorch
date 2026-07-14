from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import hashlib
import os
import pickle
import platform
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any, cast

import pytest

import scorch.compiler.llir as llir  # type: ignore[import-untyped]
import scorch.ops as ops  # type: ignore[import-untyped]
import scorch.utils as utils  # type: ignore[import-untyped]
from scorch.compiler.cin import (  # type: ignore[import-untyped]
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_analysis import (  # type: ignore[import-untyped]
    canonical_cin_dump,
    full_cin_verification,
    normalize_cin,
)
from scorch.compiler.cin_lowerer import CINLowerer  # type: ignore[import-untyped]
from scorch.compiler.codegen import LLIRLowerer  # type: ignore[import-untyped]
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    ABIPolicy,
    CompilerABICheckPolicy,
    CompilerWrapperPolicy,
    CURRENT_LLIR_PASSES,
    CompileOptions,
    DarwinToolchainOptions,
    IndexWidthPolicy,
    ISAPolicy,
    KernelBuildOptions,
    LegacyLoweringPolicy,
    LegacyParallelPolicy,
    LLIRPassId,
    ParallelBackend,
    SchedulerCostModel,
    SchedulerPolicy,
    TargetOS,
    VerificationPolicy,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    CompileOptionsDiagnostic,
    CompileOptionsError,
    VerificationError,
)
from scorch.compiler.llir_pass_manager import (  # type: ignore[import-untyped]
    CURRENT_LLIR_PASS_DESCRIPTORS,
    DEBUG_LLIR_PASS_OPTIONS,
    DENSE_POINTER_HOIST_PASS,
    DensePointerHoistPassSpec,
    LLIRBodyAssembler,
    LLIRPassManager,
    LLIRPassOptions,
    LLIRPassPipeline,
    LLIRProductionPipelineResult,
    LLIRStatementListArtifact,
    PRODUCTION_LLIR_PASS_OPTIONS,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    Schedule,
    Scheduler,
    TileSpec,
    regblock_force,
    schedule_force,
)
from scorch.layout import TensorSpec  # type: ignore[import-untyped]
from scorch.utils import (  # type: ignore[import-untyped]
    _kernel_name,
    get_extra_cflags,
    get_extra_ldflags,
)

_ENVIRONMENT_KEYS = (
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
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "OBJCPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
    "COMPILER_PATH",
    "GCC_EXEC_PREFIX",
    "CCC_OVERRIDE_OPTIONS",
    "TORCH_DONT_CHECK_COMPILER_ABI",
    "TORCH_NO_COMPILER_WRAPPER",
)


class _CountingEnvironment(Mapping[str, str]):
    def __init__(self, values: Mapping[str, str]) -> None:
        self._contents = dict(values)
        self.reads: Counter[str] = Counter()

    def __getitem__(self, key: str) -> str:
        self.reads[key] += 1
        return self._contents[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._contents)

    def __len__(self) -> int:
        return len(self._contents)

    def get(self, key: str, default: Any = None) -> Any:
        self.reads[key] += 1
        return self._contents.get(key, default)


def _default_options(
    *,
    verify_cin: bool = False,
    llir_pass_options: LLIRPassOptions = PRODUCTION_LLIR_PASS_OPTIONS,
) -> CompileOptions:
    return CompileOptions.from_environment(
        environ={},
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=verify_cin,
        llir_pass_options=llir_pass_options,
    )


def _build_spmm_source() -> ForAll:
    row, reduction, column = IndexVar("i"), IndexVar("k"), IndexVar("j")
    result = TensorVar("C", fmt="dd")
    left = TensorVar("A", fmt="ds")
    right = TensorVar("B", fmt="dd")
    assignment = TensorAssign(
        result[row, column],
        left[row, reduction] * right[reduction, column],
        op=Operation.ADD,
    )
    return ForAll(row, ForAll(reduction, ForAll(column, assignment)))


def _build_dss_source() -> ForAll:
    batch, row, reduction, column = (
        IndexVar("batch"),
        IndexVar("row"),
        IndexVar("reduction"),
        IndexVar("column"),
    )
    result = TensorVar("Result", fmt="dss")
    left = TensorVar("Left", fmt="dss")
    right = TensorVar("Right", fmt="dss")
    result[batch, row, column] = (
        left[batch, row, reduction] * right[batch, reduction, column]
    )
    assignment = result._assignment
    assert assignment is not None
    return ForAll(
        batch,
        ForAll(row, ForAll(reduction, ForAll(column, assignment))),
    )


def _structural_snapshot(value: object) -> object:
    if isinstance(value, llir.Node):
        return (
            type(value).__name__,
            tuple(
                (name, _structural_snapshot(child))
                for name, child in sorted(vars(value).items())
            ),
        )
    if isinstance(value, list):
        return ("list", tuple(_structural_snapshot(child) for child in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_structural_snapshot(child) for child in value))
    return value


def _compile_spmm(
    source: ForAll, options: CompileOptions
) -> tuple[object, object, CINLowerer, str]:
    scheduled = Scheduler.auto_schedule(source, compile_options=options)
    lowerer = CINLowerer(compile_options=options)
    lowered = lowerer.lower_IndexStmt(scheduled)
    cpp = LLIRLowerer(compile_options=options).lower_llir(lowered)
    return scheduled, lowered, lowerer, cpp


def _assert_compile_options_error(
    error: pytest.ExceptionInfo[CompileOptionsError],
    *,
    code: str,
    field: str,
) -> None:
    assert type(error.value.diagnostics) is tuple
    assert len(error.value.diagnostics) == 1
    diagnostic = error.value.diagnostics[0]
    assert type(diagnostic) is CompileOptionsDiagnostic
    assert diagnostic.code == code
    assert diagnostic.field == field
    with pytest.raises(FrozenInstanceError):
        diagnostic.code = "changed"


def test_compile_options_and_nested_policies_are_frozen_and_structurally_typed() -> (
    None
):
    options = _default_options()

    assert type(options) is CompileOptions
    assert type(options.build) is KernelBuildOptions
    assert type(options.scheduler) is SchedulerPolicy
    assert type(options.scheduler.cost_model) is SchedulerCostModel
    assert type(options.verification) is VerificationPolicy
    assert type(options.verification.llir_pass_options) is LLIRPassOptions
    assert type(options.build.target_os) is TargetOS
    assert type(options.build.isa_policy) is ISAPolicy
    assert type(options.build.parallel_backend) is ParallelBackend
    assert type(options.build.legacy_parallel_policy) is LegacyParallelPolicy
    assert type(options.build.legacy_lowering_policy) is LegacyLoweringPolicy
    assert type(options.build.index_width) is IndexWidthPolicy
    assert type(options.build.abi_policy) is ABIPolicy
    assert type(options.build.compiler_abi_check_policy) is CompilerABICheckPolicy
    assert type(options.build.compiler_wrapper_policy) is CompilerWrapperPolicy
    assert (
        options.build.compiler_wrapper_name is None
        or type(options.build.compiler_wrapper_name) is str
    )
    assert (
        options.build.compiler_wrapper_path is None
        or type(options.build.compiler_wrapper_path) is str
    )
    if options.build.target_os is TargetOS.DARWIN:
        assert type(options.build.darwin_toolchain) is DarwinToolchainOptions
        assert options.build.darwin_toolchain is not None
        assert type(options.build.darwin_toolchain.developer_dir) is str
        assert type(options.build.darwin_toolchain.sdk_root) is str
        assert options.build.darwin_toolchain.deployment_target is None
    else:
        assert options.build.darwin_toolchain is None
    assert type(options.enabled_llir_passes) is tuple
    assert all(type(pass_id) is LLIRPassId for pass_id in options.enabled_llir_passes)
    assert type(options.build.extra_cflags) is tuple
    assert type(options.build.direct_extension_cflags) is tuple
    assert type(options.build.special_kernel_cflags) is tuple
    assert options.build.special_kernel_cflags[:4] == (
        "-O3",
        "-march=native",
        "-ffast-math",
        "-fno-signed-zeros",
    )
    assert type(options.build.extra_ldflags) is tuple
    assert type(options.build.torch_include_paths) is tuple
    assert type(options.build.torch_cxx11_abi) is bool
    assert type(options.build.cxx_compiler_from_environment) is bool
    assert type(options.build.scorch_python_path) is str

    frozen_updates = (
        (options, "regblock_dual", False),
        (options.build, "target_arch", "changed"),
        (options.scheduler, "regblock_enabled", True),
        (options.scheduler.cost_model, "alpha", 1.0),
        (options.verification, "verify_cin", True),
        (options.verification.llir_pass_options, "record_timing", False),
    )
    for value, name, replacement in frozen_updates:
        with pytest.raises(FrozenInstanceError):
            setattr(value, name, replacement)

    if options.build.darwin_toolchain is not None:
        with pytest.raises(FrozenInstanceError):
            options.build.darwin_toolchain.sdk_root = "changed"


def test_compile_options_detaches_mutable_build_pass_and_schedule_inputs() -> None:
    base = _default_options()
    cflags = list(base.build.extra_cflags)
    direct_cflags = list(base.build.direct_extension_cflags)
    special_cflags = list(base.build.special_kernel_cflags)
    ldflags = list(base.build.extra_ldflags)
    enabled_passes = list(base.enabled_llir_passes)
    loop_order = ["i", "k", "j"]
    tiles = [TileSpec("j", 8, placement="child_of:i")]

    build = replace(
        base.build,
        extra_cflags=cflags,
        direct_extension_cflags=direct_cflags,
        special_kernel_cflags=special_cflags,
        extra_ldflags=ldflags,
    )
    schedule = Schedule(loop_order=loop_order, tiles=tiles)
    options = replace(
        base,
        build=build,
        requested_schedule=schedule,
        enabled_llir_passes=enabled_passes,
    )

    cflags.append("-DCHANGED")
    direct_cflags.append("-DCHANGED")
    special_cflags.append("-DCHANGED")
    ldflags.append("-lchanged")
    enabled_passes.clear()
    loop_order.reverse()
    tiles.clear()

    assert options.build.extra_cflags == base.build.extra_cflags
    assert options.build.direct_extension_cflags == base.build.direct_extension_cflags
    assert options.build.special_kernel_cflags == base.build.special_kernel_cflags
    assert options.build.extra_ldflags == base.build.extra_ldflags
    assert options.enabled_llir_passes == CURRENT_LLIR_PASSES
    assert options.requested_schedule is not None
    assert options.requested_schedule.loop_order == ("i", "k", "j")
    assert options.requested_schedule.tiles == (
        TileSpec("j", 8, placement="child_of:i"),
    )


def test_environment_keys_are_each_parsed_once_at_snapshot_boundary() -> None:
    compiler_path = shutil.which("c++")
    assert compiler_path is not None
    environ = _CountingEnvironment(
        {
            "SCORCH_REGBLOCK": "1",
            "SCORCH_REGBLOCK_MAX_N": "16",
            "SCORCH_REGBLOCK_T": "4",
            "SCORCH_VERIFY_CIN": "1",
            "SCORCH_REGBLOCK_DUAL": "0",
            "SCORCH_JIT_TUNE_HOOKS": "0",
            "CXX": compiler_path,
            "PATH": os.environ["PATH"],
        }
    )

    options = CompileOptions.from_environment(
        environ=environ,
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )

    assert environ.reads == Counter({key: 1 for key in _ENVIRONMENT_KEYS})
    assert options.scheduler.regblock_enabled
    assert options.scheduler.regblock_max_n == 16
    assert options.scheduler.regblock_tile_width == 4
    assert not options.regblock_dual
    assert options.build.cxx_compiler == compiler_path


@pytest.mark.parametrize(
    ("key", "value", "code"),
    (
        ("SCORCH_REGBLOCK", "true", "invalid_environment_boolean"),
        ("SCORCH_VERIFY_CIN", "", "invalid_environment_boolean"),
        ("SCORCH_REGBLOCK_DUAL", "2", "invalid_environment_boolean"),
        ("SCORCH_JIT_TUNE_HOOKS", "yes", "invalid_environment_boolean"),
        ("SCORCH_REGBLOCK_MAX_N", "0", "invalid_environment_integer"),
        ("SCORCH_REGBLOCK_T", "-1", "invalid_environment_integer"),
    ),
)
def test_malformed_environment_values_fail_with_typed_diagnostics(
    key: str, value: str, code: str
) -> None:
    with pytest.raises(CompileOptionsError) as error:
        CompileOptions.from_environment(
            environ={key: value},
            forced_schedule=None,
            regblock_override=None,
            verify_cin_override=False,
        )

    _assert_compile_options_error(error, code=code, field=key)


@pytest.mark.parametrize(
    ("environ", "code"),
    (
        ({"CXX": "c++ --wrapper"}, "unsupported_compiler_command"),
        ({"CXX": "scorch-compiler-that-does-not-exist"}, "compiler_not_found"),
        ({"CXX": "bad\x00compiler"}, "unsupported_compiler_command"),
        ({"PATH": ""}, "compiler_not_found"),
        ({"SCORCH_REGBLOCK_MAX_N": "2147483648"}, "integer_out_of_range"),
        ({"SCORCH_REGBLOCK_T": "9" * 5000}, "integer_out_of_range"),
    ),
)
def test_invalid_compiler_environment_fails_with_typed_diagnostics(
    environ: dict[str, str],
    code: str,
) -> None:
    with pytest.raises(CompileOptionsError) as error:
        CompileOptions.from_environment(
            environ=environ,
            forced_schedule=None,
            regblock_override=None,
            verify_cin_override=False,
        )

    field = next(
        (
            key
            for key in ("SCORCH_REGBLOCK_MAX_N", "SCORCH_REGBLOCK_T")
            if key in environ
        ),
        "CXX",
    )
    _assert_compile_options_error(error, code=code, field=field)


@pytest.mark.parametrize(
    ("key", "value", "code", "field"),
    (
        (
            "CPATH",
            "/tmp/include",
            "unsupported_compiler_environment",
            "CPATH",
        ),
        (
            "OBJCPLUS_INCLUDE_PATH",
            "/tmp/include",
            "unsupported_compiler_environment",
            "OBJCPLUS_INCLUDE_PATH",
        ),
        (
            "CCC_OVERRIDE_OPTIONS",
            "+-O0",
            "unsupported_compiler_environment",
            "CCC_OVERRIDE_OPTIONS",
        ),
        (
            "TORCH_DONT_CHECK_COMPILER_ABI",
            "1",
            "unsupported_compiler_abi_policy",
            "TORCH_DONT_CHECK_COMPILER_ABI",
        ),
        (
            "MACOSX_DEPLOYMENT_TARGET",
            "latest",
            "invalid_deployment_target",
            "build.darwin_toolchain.deployment_target",
        ),
        (
            "SDKROOT",
            "/tmp/unsupported-sdk",
            "unsupported_darwin_toolchain",
            "build.darwin_toolchain.sdk_root",
        ),
        (
            "DEVELOPER_DIR",
            "/Applications/Xcode.app/Contents/Developer",
            "unsupported_darwin_toolchain",
            "build.darwin_toolchain.developer_dir",
        ),
    ),
)
def test_unsupported_toolchain_environment_fails_closed(
    key: str,
    value: str,
    code: str,
    field: str,
) -> None:
    if key in {"DEVELOPER_DIR", "MACOSX_DEPLOYMENT_TARGET", "SDKROOT"} and (
        platform.system() != "Darwin"
    ):
        pytest.skip("Darwin toolchain validation is Darwin-specific")

    with pytest.raises(CompileOptionsError) as error:
        CompileOptions.from_environment(
            environ={key: value},
            forced_schedule=None,
            regblock_override=None,
            verify_cin_override=False,
        )

    _assert_compile_options_error(error, code=code, field=field)


def test_darwin_toolchain_snapshot_is_detached_and_cache_independent() -> None:
    if platform.system() != "Darwin":
        pytest.skip("Darwin toolchain policy is Darwin-specific")

    environ = {"MACOSX_DEPLOYMENT_TARGET": "13.0"}
    first = CompileOptions.from_environment(
        environ=environ,
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )
    environ["MACOSX_DEPLOYMENT_TARGET"] = "14.0"
    second = CompileOptions.from_environment(
        environ=environ,
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )

    assert first.build.darwin_toolchain is not None
    assert second.build.darwin_toolchain is not None
    assert first.build.darwin_toolchain.deployment_target == "13.0"
    assert second.build.darwin_toolchain.deployment_target == "14.0"
    assert first.build_cache_key != second.build_cache_key
    assert first.cache_fingerprint != second.cache_fingerprint


def test_darwin_toolchain_and_target_combinations_fail_closed() -> None:
    base = _default_options()
    if base.build.target_os is not TargetOS.DARWIN:
        pytest.skip("Darwin toolchain policy is Darwin-specific")

    with pytest.raises(CompileOptionsError) as missing_error:
        replace(base.build, darwin_toolchain=None)
    _assert_compile_options_error(
        missing_error,
        code="invalid_darwin_toolchain",
        field="build.darwin_toolchain",
    )

    assert base.build.darwin_toolchain is not None
    with pytest.raises(CompileOptionsError) as sdk_error:
        replace(base.build.darwin_toolchain, sdk_root="/tmp/unsupported-sdk")
    _assert_compile_options_error(
        sdk_error,
        code="unsupported_darwin_toolchain",
        field="build.darwin_toolchain.sdk_root",
    )


def test_invalid_option_combinations_fail_closed() -> None:
    base = _default_options()

    with pytest.raises(CompileOptionsError) as pass_error:
        replace(base, enabled_llir_passes=base.enabled_llir_passes[:-1])
    _assert_compile_options_error(
        pass_error,
        code="unsupported_pass_pipeline",
        field="enabled_llir_passes",
    )

    with pytest.raises(CompileOptionsError) as build_error:
        replace(base.build, jit_tune_hooks=True)
    _assert_compile_options_error(
        build_error,
        code="inconsistent_tune_hooks",
        field="build.extra_cflags",
    )

    with pytest.raises(CompileOptionsError) as verification_error:
        replace(
            base.verification,
            llir_pass_options=LLIRPassOptions(record_timing=False),
        )
    _assert_compile_options_error(
        verification_error,
        code="unsupported_verification_policy",
        field="verification.llir_pass_options",
    )

    with pytest.raises(CompileOptionsError) as bool_error:
        replace(
            base.verification,
            llir_pass_options=LLIRPassOptions(verify_before_pass=1),
        )
    _assert_compile_options_error(
        bool_error,
        code="invalid_type",
        field="verification.llir_pass_options.verify_before_pass",
    )

    with pytest.raises(CompileOptionsError) as width_error:
        replace(base.scheduler, regblock_tile_width=2**31)
    _assert_compile_options_error(
        width_error,
        code="integer_out_of_range",
        field="scheduler.regblock_tile_width",
    )

    with pytest.raises(CompileOptionsError) as flags_error:
        replace(
            base.build,
            extra_cflags=base.build.extra_cflags + ("-O0",),
        )
    _assert_compile_options_error(
        flags_error,
        code="unsupported_build_flags",
        field="build.extra_cflags",
    )

    with pytest.raises(CompileOptionsError) as special_flags_error:
        replace(
            base.build,
            special_kernel_cflags=base.build.special_kernel_cflags + ("-O0",),
        )
    _assert_compile_options_error(
        special_flags_error,
        code="unsupported_build_flags",
        field="build.special_kernel_cflags",
    )

    with pytest.raises(CompileOptionsError) as direct_flags_error:
        replace(
            base.build,
            direct_extension_cflags=base.build.direct_extension_cflags + ("-O0",),
        )
    _assert_compile_options_error(
        direct_flags_error,
        code="unsupported_build_flags",
        field="build.direct_extension_cflags",
    )

    with pytest.raises(CompileOptionsError) as build_path_error:
        replace(base.build, python_include_path="relative/include")
    _assert_compile_options_error(
        build_path_error,
        code="invalid_build_path",
        field="build.python_include_path",
    )

    requested = Schedule(loop_order=("i", "k", "j"))
    forced = Schedule(loop_order=("i", "j", "k"))
    with pytest.raises(CompileOptionsError) as schedule_error:
        CompileOptions.from_environment(
            environ={},
            requested_schedule=requested,
            forced_schedule=forced,
            regblock_override=None,
            verify_cin_override=False,
        )
    _assert_compile_options_error(
        schedule_error,
        code="conflicting_schedule",
        field="forced_schedule",
    )


def test_environment_mutation_after_snapshot_does_not_change_compilation() -> None:
    environ = {"SCORCH_REGBLOCK": "0", "SCORCH_REGBLOCK_T": "4"}
    snapshot = CompileOptions.from_environment(
        environ=environ,
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )
    environ["SCORCH_REGBLOCK"] = "1"
    environ["SCORCH_REGBLOCK_T"] = "16"

    source = _build_spmm_source()
    _, _, _, snapshotted_cpp = _compile_spmm(source, snapshot)
    changed = CompileOptions.from_environment(
        environ=environ,
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )
    _, _, _, changed_cpp = _compile_spmm(source, changed)

    assert not snapshot.scheduler.regblock_enabled
    assert snapshot.scheduler.regblock_tile_width == 4
    assert changed.scheduler.regblock_enabled
    assert changed.scheduler.regblock_tile_width == 16
    assert snapshotted_cpp != changed_cpp


def test_distinct_snapshots_compile_independently_without_mutating_owned_ir() -> None:
    source = _build_spmm_source()
    source_before = canonical_cin_dump(source)
    base = _default_options()
    baseline = base.with_regblock_enabled(False)
    register_blocked = base.with_regblock_enabled(True)

    scheduled_off, llir_off, _, cpp_off = _compile_spmm(source, baseline)
    scheduled_off_before = canonical_cin_dump(cast(ForAll, scheduled_off))
    llir_off_before = _structural_snapshot(llir_off)
    scheduled_on, _, _, cpp_on = _compile_spmm(source, register_blocked)
    scheduled_off_again, _, _, cpp_off_again = _compile_spmm(source, baseline)

    assert cpp_on != cpp_off
    assert cpp_off_again == cpp_off
    assert canonical_cin_dump(cast(ForAll, scheduled_on)) != scheduled_off_before
    assert canonical_cin_dump(cast(ForAll, scheduled_off_again)) == (
        scheduled_off_before
    )
    assert canonical_cin_dump(cast(ForAll, scheduled_off)) == scheduled_off_before
    assert _structural_snapshot(llir_off) == llir_off_before
    assert canonical_cin_dump(source) == source_before


def test_default_explicit_options_preserve_csr_dense_generated_cpp_anchor() -> None:
    options = _default_options()
    _, _, _, cpp = _compile_spmm(_build_spmm_source(), options)

    assert len(cpp) == 2505
    assert hashlib.sha256(cpp.encode()).hexdigest() == (
        "36a8599c59f06b2cb060e27af26b7c9196716be88f666282d83b1ec2dc9d6151"
    )


def test_production_and_debug_snapshots_preserve_verification_policy() -> None:
    production = _default_options()
    debug = _default_options(
        verify_cin=True,
        llir_pass_options=DEBUG_LLIR_PASS_OPTIONS,
    )
    i = IndexVar("i")
    invalid = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="dd")[i], TensorVar("A", fmt="dd")[i]),
    )

    normalize_cin(invalid, compile_options=production)
    with pytest.raises(VerificationError, match="tensor_access_rank_mismatch"):
        normalize_cin(invalid, compile_options=debug)

    _, _, production_lowerer, production_cpp = _compile_spmm(
        _build_spmm_source(), production
    )
    _, _, debug_lowerer, debug_cpp = _compile_spmm(_build_spmm_source(), debug)

    assert production_cpp == debug_cpp
    assert production_lowerer.llir_pass_run_records
    assert debug_lowerer.llir_pass_run_records
    assert all(
        not record.verified_before and not record.verified_after
        for record in production_lowerer.llir_pass_run_records
    )
    assert all(
        record.verified_before and record.verified_after
        for record in debug_lowerer.llir_pass_run_records
    )


def test_explicit_snapshot_prevents_nested_environment_resnapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _default_options()

    def fail_resnapshot(
        cls: type[CompileOptions], *args: object, **kwargs: object
    ) -> None:
        raise AssertionError("a nested compiler stage attempted to resnapshot state")

    monkeypatch.setattr(
        CompileOptions,
        "from_environment",
        classmethod(fail_resnapshot),
    )

    _, _, _, cpp = _compile_spmm(_build_spmm_source(), options)
    assert cpp
    assert get_extra_cflags(compile_options=options) == list(
        options.build.direct_extension_cflags
    )
    assert get_extra_ldflags(compile_options=options) == list(
        options.build.extra_ldflags
    )
    assert _kernel_name("source", compile_options=options).startswith("kernel_")


def test_production_lowering_delegates_once_to_each_snapshotted_manager_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = _default_options()
    debug = _default_options(
        verify_cin=True,
        llir_pass_options=DEBUG_LLIR_PASS_OPTIONS,
    )
    observed_calls: list[
        tuple[
            LLIRPassManager,
            LLIRStatementListArtifact,
            DensePointerHoistPassSpec,
            LLIRBodyAssembler,
            LLIRProductionPipelineResult,
        ]
    ] = []
    original_pipeline_entry = LLIRPassManager.run_production_pipeline

    def record_pipeline_entry(
        self: LLIRPassManager,
        artifact: LLIRStatementListArtifact,
        *,
        compressed_where_pass_spec: object,
        dense_pointer_pass_spec: DensePointerHoistPassSpec,
        body_assembler: LLIRBodyAssembler,
    ) -> LLIRProductionPipelineResult:
        assert compressed_where_pass_spec is None
        result = original_pipeline_entry(
            self,
            artifact,
            compressed_where_pass_spec=None,
            dense_pointer_pass_spec=dense_pointer_pass_spec,
            body_assembler=body_assembler,
        )
        observed_calls.append(
            (
                self,
                artifact,
                dense_pointer_pass_spec,
                body_assembler,
                result,
            )
        )
        return result

    def fail_resnapshot(
        cls: type[CompileOptions], *args: object, **kwargs: object
    ) -> CompileOptions:
        raise AssertionError("production lowering attempted to resnapshot state")

    monkeypatch.setattr(
        LLIRPassManager,
        "run_production_pipeline",
        record_pipeline_entry,
    )
    monkeypatch.setattr(
        CompileOptions,
        "from_environment",
        classmethod(fail_resnapshot),
    )
    monkeypatch.setenv("SCORCH_REGBLOCK", "1")
    monkeypatch.setenv("SCORCH_VERIFY_CIN", "1")
    conflicting_schedule = Schedule(loop_order=("i", "j", "k"))

    with (
        regblock_force(True),
        schedule_force(conflicting_schedule),
        full_cin_verification(True),
    ):
        _, _, production_lowerer, production_cpp = _compile_spmm(
            _build_spmm_source(), production
        )
    with (
        regblock_force(True),
        schedule_force(conflicting_schedule),
        full_cin_verification(False),
    ):
        _, _, debug_lowerer, debug_cpp = _compile_spmm(_build_spmm_source(), debug)

    assert len(observed_calls) == 2
    expected_options = (production, debug)
    expected_lowerers = (production_lowerer, debug_lowerer)
    expected_record_names = tuple(pass_id.value for pass_id in CURRENT_LLIR_PASSES[2:])
    for call, options, lowerer in zip(
        observed_calls, expected_options, expected_lowerers
    ):
        manager, artifact, dense_spec, body_assembler, result = call
        pipeline = manager.pipeline

        assert type(pipeline) is LLIRPassPipeline
        assert pipeline is lowerer.llir_pass_pipeline
        assert pipeline.compile_options is options
        assert pipeline.pass_ids is options.enabled_llir_passes
        assert pipeline.pass_ids == CURRENT_LLIR_PASSES
        assert pipeline.pass_descriptors is CURRENT_LLIR_PASS_DESCRIPTORS
        assert pipeline.options is options.verification.llir_pass_options
        assert manager.options is pipeline.options
        assert type(artifact) is LLIRStatementListArtifact
        assert type(artifact.statements) is list
        assert type(dense_spec) is DensePointerHoistPassSpec
        assert dense_spec.descriptor is DENSE_POINTER_HOIST_PASS
        assert dense_spec.context.value_array_ctypes == (
            ("A_val", "float"),
            ("B_val", "float"),
        )
        assert callable(body_assembler)
        assert type(result) is LLIRProductionPipelineResult
        assert tuple(record.sequence_index for record in result.run_records) == tuple(
            range(len(result.run_records))
        )
        assert tuple(record.pass_name for record in result.run_records) == (
            expected_record_names
        )
        assert result.run_records == lowerer.llir_pass_run_records

    first_result = observed_calls[0][-1]
    second_result = observed_calls[1][-1]
    assert production_lowerer.llir_pass_pipeline is not debug_lowerer.llir_pass_pipeline
    assert first_result is not second_result
    assert first_result.artifact is not second_result.artifact
    assert first_result.artifact.value is not second_result.artifact.value
    assert production_cpp == debug_cpp
    assert len(production_cpp) == 2505
    assert hashlib.sha256(production_cpp.encode()).hexdigest() == (
        "36a8599c59f06b2cb060e27af26b7c9196716be88f666282d83b1ec2dc9d6151"
    )


def test_compressed_output_routes_exact_snapshot_through_nested_renderers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = replace(_default_options(), emit_comments=False)
    observed_options: list[object] = []
    original_codegen_init = LLIRLowerer.__init__

    def record_codegen_init(
        self: LLIRLowerer,
        compile_options: CompileOptions | None = None,
    ) -> None:
        observed_options.append(compile_options)
        original_codegen_init(self, compile_options=compile_options)

    monkeypatch.setattr(LLIRLowerer, "__init__", record_codegen_init)

    scheduled = Scheduler.auto_schedule(
        _build_dss_source(),
        compile_options=options,
    )
    lowerer = CINLowerer(compile_options=options)
    lowered = lowerer.lower_IndexStmt(scheduled)
    cpp = LLIRLowerer(compile_options=options).lower_llir(lowered)

    assert cpp
    assert len(observed_options) == 5
    assert all(observed is options for observed in observed_options)
    assert [
        record.configuration_name for record in lowerer.llir_pass_run_records[:3]
    ] == ["compressed_where_openmp", "count", "fill"]


def test_public_einsum_snapshots_once_and_routes_one_identity_to_all_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _default_options().with_regblock_enabled(True)
    snapshot_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    scheduler_options: list[object] = []
    cin_lowerer_options: list[object] = []
    codegen_options: list[object] = []
    build_calls: list[utils._PreparedJITBuild] = []
    original_auto_schedule = Scheduler.auto_schedule
    original_regblock_arm = Scheduler._auto_schedule_regblock_arm
    original_cin_init = CINLowerer.__init__
    original_codegen_init = LLIRLowerer.__init__

    def snapshot_once(
        cls: type[CompileOptions], *args: object, **kwargs: object
    ) -> CompileOptions:
        snapshot_calls.append((args, kwargs))
        return snapshot

    def record_schedule(*args: object, **kwargs: object) -> object:
        scheduler_options.append(kwargs.get("compile_options"))
        return original_auto_schedule(*args, **kwargs)

    def record_regblock_arm(*args: object, **kwargs: object) -> object:
        scheduler_options.append(kwargs.get("compile_options"))
        return original_regblock_arm(*args, **kwargs)

    def record_cin_init(self: CINLowerer, *args: object, **kwargs: object) -> None:
        cin_lowerer_options.append(kwargs.get("compile_options"))
        original_cin_init(self, *args, **kwargs)

    def record_codegen_init(self: LLIRLowerer, *args: object, **kwargs: object) -> None:
        codegen_options.append(kwargs.get("compile_options"))
        original_codegen_init(self, *args, **kwargs)

    def record_build(prepared: utils._PreparedJITBuild) -> object:
        build_calls.append(prepared)
        return object()

    monkeypatch.setattr(
        CompileOptions,
        "from_environment",
        classmethod(snapshot_once),
    )
    monkeypatch.setattr(Scheduler, "auto_schedule", staticmethod(record_schedule))
    monkeypatch.setattr(
        Scheduler,
        "_auto_schedule_regblock_arm",
        staticmethod(record_regblock_arm),
    )
    monkeypatch.setattr(CINLowerer, "__init__", record_cin_init)
    monkeypatch.setattr(LLIRLowerer, "__init__", record_codegen_init)
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", record_build)
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})

    result = ops.einsum(
        "ik,kj->ij",
        TensorSpec("dd", (2, 3), name="A"),
        TensorSpec("dd", (3, 4), name="B"),
        compile_only=True,
        format="dd",
    )

    assert isinstance(result, TensorSpec)
    assert len(snapshot_calls) == 1
    assert scheduler_options and all(value is snapshot for value in scheduler_options)
    assert cin_lowerer_options and all(
        value is snapshot for value in cin_lowerer_options
    )
    assert codegen_options and all(value is snapshot for value in codegen_options)
    assert len(build_calls) == 1
    request = build_calls[0].request
    assert request.cpp_sources[0] == snapshot.build.preamble_source
    assert request.extra_cflags == snapshot.build.extra_cflags
    assert request.extra_ldflags == snapshot.build.extra_ldflags
    assert request.build_options is snapshot.build


def test_build_and_emission_variants_are_explicit_and_cache_independent() -> None:
    base = _default_options()
    custom_compiler = CompileOptions.from_environment(
        environ={
            "CXX": base.build.cxx_compiler_path,
            "PATH": base.build.executable_search_path,
        },
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )
    without_comments = replace(base, emit_comments=False)

    assert _kernel_name("source", compile_options=base) == _kernel_name(
        "source",
        compile_options=custom_compiler,
    )
    assert base.build_cache_key != custom_compiler.build_cache_key
    assert LLIRLowerer(compile_options=base).lower_llir(llir.Comment("kept")) == (
        "// kept"
    )
    assert (
        LLIRLowerer(compile_options=without_comments).lower_llir(
            llir.Comment("removed")
        )
        == ""
    )
    with pytest.raises(CompileOptionsError) as emission_error:
        LLIRLowerer(compile_options=base).lower_llir(
            llir.Comment("conflict"),
            no_comments=True,
        )
    _assert_compile_options_error(
        emission_error,
        code="conflicting_emission_policy",
        field="no_comments",
    )


def test_build_boundary_uses_snapshotted_compiler_without_parent_environment_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    implicit_options = _default_options()
    explicit_options = CompileOptions.from_environment(
        environ={
            "CXX": implicit_options.build.cxx_compiler_path,
            "PATH": implicit_options.build.executable_search_path,
        },
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )
    barrier = threading.Barrier(2)
    observed_compilers: list[str | None] = []
    observed_paths: list[str] = []
    observed_lock = threading.Lock()

    def request_for(
        options: CompileOptions,
        directory_name: str,
    ) -> utils._JITBuildRequest:
        return utils._JITBuildRequest(
            name="compile_options_cxx_test",
            cpp_sources=("source",),
            functions=("evaluate",),
            extra_cflags=options.build.extra_cflags,
            extra_ldflags=options.build.extra_ldflags,
            build_directory=str(tmp_path / directory_name),
            build_options=options.build,
        )

    requests = (
        request_for(implicit_options, "implicit"),
        request_for(explicit_options, "explicit"),
    )

    def fake_run(
        command: list[str],
        *,
        input: bytes,
        env: dict[str, str],
        check: bool,
    ) -> object:
        request = pickle.loads(input)
        assert type(request) is utils._JITBuildRequest
        assert command[0] == request.build_options.python_executable
        assert command[1] == "-P"
        assert check
        assert env["PYTHONPATH"] == request.build_options.scorch_python_path
        if (
            request.build_options.compiler_wrapper_policy
            is CompilerWrapperPolicy.DISABLED
        ):
            assert env["TORCH_NO_COMPILER_WRAPPER"] == "1"
        else:
            assert "TORCH_NO_COMPILER_WRAPPER" not in env
        darwin_toolchain = request.build_options.darwin_toolchain
        if darwin_toolchain is None:
            assert "DEVELOPER_DIR" not in env
            assert "SDKROOT" not in env
            assert "MACOSX_DEPLOYMENT_TARGET" not in env
        else:
            assert env["DEVELOPER_DIR"] == darwin_toolchain.developer_dir
            assert env["SDKROOT"] == darwin_toolchain.sdk_root
            assert "MACOSX_DEPLOYMENT_TARGET" not in env
        with observed_lock:
            observed_compilers.append(env.get("CXX"))
            observed_paths.append(env["PATH"])
        barrier.wait(timeout=5)
        output = Path(request.build_directory) / f"{request.name}.so"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        return object()

    monkeypatch.setattr(utils.subprocess, "run", fake_run)
    monkeypatch.setattr(utils, "_load_extension_file", lambda name, path: path)
    monkeypatch.setenv("CXX", "ambient-after-snapshot")
    monkeypatch.setenv("DEVELOPER_DIR", "ambient-after-snapshot")
    monkeypatch.setenv("SDKROOT", "ambient-after-snapshot")
    monkeypatch.setenv("MACOSX_DEPLOYMENT_TARGET", "99.0")
    monkeypatch.setenv("TORCH_NO_COMPILER_WRAPPER", "ambient-after-snapshot")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(utils._build_and_load_extension, requests))

    assert results == tuple(
        str(Path(request.build_directory) / f"{request.name}.so")
        for request in requests
    )
    assert set(observed_compilers) == {
        None,
        explicit_options.build.cxx_compiler,
    }
    implicit_toolchain = str(Path(requests[0].build_directory) / ".scorch-toolchain")
    assert set(observed_paths) == {
        implicit_toolchain + os.pathsep + implicit_options.build.executable_search_path,
        explicit_options.build.executable_search_path,
    }
    assert (Path(implicit_toolchain) / "c++").readlink() == Path(
        implicit_options.build.cxx_compiler_path
    )
    assert os.environ["CXX"] == "ambient-after-snapshot"
    assert os.environ["DEVELOPER_DIR"] == "ambient-after-snapshot"
    assert os.environ["SDKROOT"] == "ambient-after-snapshot"
    assert os.environ["MACOSX_DEPLOYMENT_TARGET"] == "99.0"
    assert os.environ["TORCH_NO_COMPILER_WRAPPER"] == "ambient-after-snapshot"


def test_compiler_wrapper_discovery_is_snapshotted_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = _default_options()
    compiler = tmp_path / "c++"
    wrapper = tmp_path / "ccache"
    compiler.symlink_to(base.build.cxx_compiler_path)
    wrapper.symlink_to(base.build.cxx_compiler_path)

    automatic = CompileOptions.from_environment(
        environ={"PATH": str(tmp_path)},
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )
    disabled = CompileOptions.from_environment(
        environ={
            "PATH": str(tmp_path),
            "TORCH_NO_COMPILER_WRAPPER": "0",
        },
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )

    assert automatic.build.compiler_wrapper_policy is CompilerWrapperPolicy.AUTO
    assert automatic.build.compiler_wrapper_name == "ccache"
    assert automatic.build.compiler_wrapper_path == str(wrapper)
    assert disabled.build.compiler_wrapper_policy is CompilerWrapperPolicy.DISABLED
    assert disabled.build.compiler_wrapper_name is None
    assert disabled.build.compiler_wrapper_path is None
    assert automatic.build_cache_key != disabled.build_cache_key

    request = utils._JITBuildRequest(
        name="compile_options_wrapper_test",
        cpp_sources=("source",),
        functions=("evaluate",),
        extra_cflags=automatic.build.extra_cflags,
        extra_ldflags=automatic.build.extra_ldflags,
        build_directory=str(tmp_path / "build"),
        build_options=automatic.build,
    )
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv("TORCH_NO_COMPILER_WRAPPER", raising=False)
    utils._verify_snapshotted_build_runtime(request)

    wrapper.unlink()
    with pytest.raises(RuntimeError, match="compiler_wrapper"):
        utils._verify_snapshotted_build_runtime(request)


def test_build_boundary_snapshots_caller_owned_sequences_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch.utils.cpp_extension as cpp_extension

    options = _default_options()
    sources = ["source"]
    functions = ["evaluate"]
    cflags = list(options.build.extra_cflags)
    ldflags = list(options.build.extra_ldflags)
    captured: list[utils._JITBuildRequest] = []

    def fake_build(request: utils._JITBuildRequest) -> object:
        captured.append(request)
        sources.append("changed")
        functions.append("changed")
        cflags.append("-DCHANGED")
        ldflags.append("-lchanged")
        return object()

    monkeypatch.setattr(
        cpp_extension,
        "_get_build_directory",
        lambda name, verbose: str(tmp_path),
    )
    monkeypatch.setattr(utils, "_build_and_load_extension", fake_build)
    utils._so_cache.clear()

    utils._load_kernel(
        "compile_options_detached_build_inputs",
        sources,
        functions,
        cflags,
        ldflags,
        compile_options=options,
    )

    assert len(captured) == 1
    request = captured[0]
    assert request.cpp_sources == ("source",)
    assert request.functions == ("evaluate",)
    assert request.extra_cflags == options.build.extra_cflags
    assert request.extra_ldflags == options.build.extra_ldflags


def test_darwin_direct_extension_flags_compile_without_ambient_target_state(
    tmp_path: Path,
) -> None:
    options = _default_options()
    if options.build.target_os is not TargetOS.DARWIN:
        pytest.skip("Darwin SDK coherence regression is Darwin-specific")
    assert options.build.darwin_toolchain is not None

    source = tmp_path / "toolchain_probe.cpp"
    source.write_text(
        "#include <cstdint>\n"
        "#include <valarray>\n"
        "static_assert(sizeof(std::int32_t) == 4);\n",
        encoding="utf-8",
    )
    request = utils._JITBuildRequest(
        name="compile_options_toolchain_probe",
        cpp_sources=("source",),
        functions=("evaluate",),
        extra_cflags=options.build.extra_cflags,
        extra_ldflags=options.build.extra_ldflags,
        build_directory=str(tmp_path / "build"),
        build_options=options.build,
    )
    toolchain_directory = utils._prepare_compiler_toolchain(request)
    child_environment = utils._jit_build_environment_from_request(
        request,
        toolchain_directory=toolchain_directory,
    )
    child_environment.pop("DEVELOPER_DIR")
    child_environment.pop("SDKROOT")
    completed = subprocess.run(
        [
            options.build.cxx_compiler,
            "-std=c++20",
            *options.build.direct_extension_cflags,
            "-fsyntax-only",
            str(source),
        ],
        env=child_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_generated_and_direct_extension_target_flags_are_explicit() -> None:
    options = _default_options()
    if options.build.target_os is TargetOS.DARWIN:
        assert "-isysroot" not in options.build.extra_cflags
        assert "-isysroot" in options.build.direct_extension_cflags
        assert options.build.darwin_toolchain is not None
        sysroot_index = options.build.direct_extension_cflags.index("-isysroot")
        assert options.build.direct_extension_cflags[sysroot_index + 1] == (
            options.build.darwin_toolchain.sdk_root
        )
    else:
        assert options.build.direct_extension_cflags == options.build.extra_cflags


def test_relative_cxx_path_is_resolved_and_detached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = _default_options()
    relative_compiler = tmp_path / "relative-cxx"
    relative_compiler.symlink_to(base.build.cxx_compiler_path)
    monkeypatch.chdir(tmp_path)

    options = CompileOptions.from_environment(
        environ={"CXX": "./relative-cxx", "PATH": base.build.executable_search_path},
        forced_schedule=None,
        regblock_override=None,
        verify_cin_override=False,
    )

    assert options.build.cxx_compiler == str(relative_compiler)
    assert options.build.cxx_compiler_path == str(relative_compiler)


def test_legacy_schedule_and_cost_arguments_cannot_conflict_with_snapshot() -> None:
    base = _default_options()
    requested = Schedule(loop_order=("i", "k", "j"))
    scheduled_options = replace(base, requested_schedule=requested)
    different_costs = replace(base.scheduler.cost_model, alpha=3.0)

    with pytest.raises(CompileOptionsError) as auto_error:
        Scheduler.auto_schedule(
            _build_spmm_source(),
            compile_options=scheduled_options,
        )
    _assert_compile_options_error(
        auto_error,
        code="conflicting_schedule",
        field="requested_schedule",
    )

    with pytest.raises(CompileOptionsError) as cost_error:
        Scheduler.auto_schedule(
            _build_spmm_source(),
            costs=different_costs,
            compile_options=base,
        )
    _assert_compile_options_error(
        cost_error,
        code="conflicting_scheduler_cost_model",
        field="scheduler.cost_model",
    )

    with pytest.raises(CompileOptionsError) as apply_error:
        Scheduler.apply_schedule(
            _build_spmm_source(),
            requested,
            compile_options=base,
        )
    _assert_compile_options_error(
        apply_error,
        code="conflicting_schedule",
        field="requested_schedule",
    )

    scheduled = Scheduler.apply_schedule(
        _build_spmm_source(),
        requested,
        compile_options=scheduled_options,
    )
    assert scheduled.verified_loop_plan.provenance == "explicit"


def test_compile_entry_without_schedule_support_fails_closed() -> None:
    requested = Schedule(loop_order=("i", "k", "j"))
    options = replace(_default_options(), requested_schedule=requested)

    with pytest.raises(CompileOptionsError) as error:
        ops.spmv(None, None, _compile_options=options)  # type: ignore[arg-type]
    _assert_compile_options_error(
        error,
        code="unsupported_requested_schedule",
        field="requested_schedule",
    )


def test_plain_cin_lowering_cannot_ignore_snapshotted_schedule() -> None:
    requested = Schedule(loop_order=("i", "k", "j"))
    options = replace(_default_options(), requested_schedule=requested)
    source = _build_spmm_source()

    with pytest.raises(CompileOptionsError) as error:
        CINLowerer(compile_options=options).lower_IndexStmt(source)
    _assert_compile_options_error(
        error,
        code="unsupported_requested_schedule",
        field="requested_schedule",
    )

    scheduled = Scheduler.apply_schedule(
        source,
        requested,
        compile_options=options,
    )
    lowered = CINLowerer(compile_options=options).lower_IndexStmt(scheduled)
    assert lowered
