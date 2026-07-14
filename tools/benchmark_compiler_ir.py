#!/usr/bin/env python3
"""Phase 0 compiler-IR latency corpus and generated-kernel noise control.

The latency command measures the production Python compiler path from the public,
validated operation boundary through the frozen JIT build request and cache identity.
It records the predecessor-compatible marker before request assembly and intercepts
the pre-native loader, so native cache lookup, C++ compilation, dynamic loading, and
kernel execution are excluded.

The kernel-aa command measures the legacy generated SpMM kernel twice in alternating
lanes.  Both lanes call the same loaded module; their ratios are the per-machine A/A
noise control for a later generated-kernel before/after comparison.

Results are printed but are written only when ``--output`` is supplied.  Result files
are benchmark artifacts and must not be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

import torch

import scorch  # type: ignore[import-untyped]
from scorch import ops
from scorch.compiler.compilation_context import (  # type: ignore[import-untyped]
    CompilationContext,
)
from scorch.layout import TensorSpec  # type: ignore[import-untyped]
from scorch.stensor import STensor  # type: ignore[import-untyped]
from scorch.utils import _PreparedJITBuild  # type: ignore[import-untyped]

SCHEMA_VERSION = 1
LATENCY_CORPUS_VERSION = "phase0-v1"
KERNEL_CORPUS_VERSION = "codegen-parity-spmm-v1"
LATENCY_MAX_RATIO = 1.10

FULL_ROWS = (512, 4096, 20000)
FULL_FREE_DIMS = (1, 3, 4, 8, 16, 64, 256)
FULL_DENSITIES = (0.02, 0.1)
QUICK_ROWS = (64, 256)
QUICK_FREE_DIMS = (4, 16)
QUICK_DENSITIES = (0.05,)


class _CompilationCaptured(Exception):
    """Internal sentinel raised exactly where the native build would begin."""

    def __init__(
        self,
        boundary_ns: int,
        legacy_boundary_ns: int,
        prepared: _PreparedJITBuild,
    ) -> None:
        super().__init__("native build boundary reached")
        self.boundary_ns = boundary_ns
        self.legacy_boundary_ns = legacy_boundary_ns
        self.prepared = prepared


@dataclass(frozen=True)
class LatencyCase:
    name: str
    operation: str
    formats: Sequence[str]
    output_format: str
    invoke: Callable[[], object]


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _metadata() -> Dict[str, Any]:
    return {
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_status_short": _git_value("status", "--short"),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "scorch": getattr(scorch, "__version__", "unknown"),
        "scorch_source_root": str(Path(scorch.__file__).resolve().parent),
        "torch_threads": torch.get_num_threads(),
    }


def _require_imported_scorch_from_worktree() -> None:
    """Fail closed if Git metadata and imported compiler source can disagree."""

    worktree = _git_value("rev-parse", "--show-toplevel")
    if worktree == "unknown":
        raise RuntimeError("compiler benchmark requires a Git worktree")
    expected = (Path(worktree) / "src" / "scorch").resolve()
    actual = Path(scorch.__file__).resolve().parent
    if actual != expected:
        raise RuntimeError(
            "compiler benchmark imported Scorch from a different worktree: "
            f"expected {expected}, got {actual}"
        )


def _percentile(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("cannot compute a percentile of no samples")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _source_summary(load_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    sources = [str(source) for source in load_kwargs.get("cpp_sources", ())]
    source_text = "\n".join(sources)
    return {
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "source_bytes": len(source_text.encode()),
        "kernel_name": load_kwargs.get("name"),
        "functions": list(load_kwargs.get("functions", ())),
        "extra_cflags": list(load_kwargs.get("extra_cflags", ())),
        "extra_ldflags": list(load_kwargs.get("extra_ldflags", ())),
    }


def _prepared_source_summary(prepared: _PreparedJITBuild) -> Dict[str, Any]:
    request = prepared.request
    return _source_summary(
        {
            "name": request.name,
            "cpp_sources": request.cpp_sources,
            "functions": request.functions,
            "extra_cflags": request.extra_cflags,
            "extra_ldflags": request.extra_ldflags,
        }
    )


def _write_result(result: Mapping[str, Any], output: Optional[Path]) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


def _machine_identity(result: Mapping[str, Any]) -> tuple[str, str]:
    metadata = result.get("metadata", {})
    hostname = metadata.get("hostname")
    machine = metadata.get("machine")
    if not isinstance(hostname, str) or not isinstance(machine, str):
        raise ValueError("benchmark result is missing machine identity metadata")
    return hostname, machine


def _build_latency_cases() -> List[LatencyCase]:
    dense_left = TensorSpec("dd", (4, 4), name="dense_left")
    dense_right = TensorSpec("dd", (4, 4), name="dense_right")
    reduction_matrix = TensorSpec("dd", (8, 16), name="reduction_matrix")
    reduction_vector = TensorSpec("d", (16,), name="reduction_vector")
    intersection_left = TensorSpec("ds", (8, 16), name="intersection_left")
    intersection_right = TensorSpec("ds", (8, 16), name="intersection_right")

    union_left = STensor.from_torch(
        torch.tensor(
            [
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "union_left",
    ).to_sparse("ds")
    union_right = STensor.from_torch(
        torch.tensor(
            [
                [0.0, 6.0, 2.0, 0.0],
                [7.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 5.0],
                [0.0, 9.0, 0.0, 0.0],
            ]
        ),
        "union_right",
    ).to_sparse("ds")

    return [
        LatencyCase(
            "small_dense",
            "ij,ij->ij",
            ("dd", "dd"),
            "dd",
            lambda: ops.einsum(
                "ij,ij->ij",
                dense_left,
                dense_right,
                compile_only=True,
                format="dd",
            ),
        ),
        LatencyCase(
            "reduction",
            "ij,j->i",
            ("dd", "d"),
            "d",
            lambda: ops.einsum(
                "ij,j->i",
                reduction_matrix,
                reduction_vector,
                compile_only=True,
                format="d",
            ),
        ),
        LatencyCase(
            "csr_intersection",
            "ij,ij->ij (multiply/intersection)",
            ("ds", "ds"),
            "ds",
            lambda: ops.einsum(
                "ij,ij->ij",
                intersection_left,
                intersection_right,
                compile_only=True,
                format="ds",
            ),
        ),
        LatencyCase(
            "sparse_union",
            "STensor.__add__",
            ("ds", "ds"),
            "ds",
            lambda: union_left + union_right,
        ),
    ]


@contextmanager
def _intercept_native_builds(
    captures: List[Dict[str, Any]],
    legacy_boundaries: List[int],
    compilation_contexts: List[CompilationContext],
    *,
    stop_before_build: bool,
) -> Iterator[None]:
    stensor_module = importlib.import_module("scorch.stensor")
    original_ops_prepare = ops._prepare_jit_build
    original_stensor_prepare = getattr(stensor_module, "_prepare_jit_build")
    original_ops_load = ops._load_validated_prepared_kernel
    original_stensor_load = getattr(stensor_module, "_load_validated_prepared_kernel")
    original_context_post_init = CompilationContext.__post_init__

    def intercept_prepare(*args: Any, **kwargs: Any) -> _PreparedJITBuild:
        legacy_boundaries.append(time.perf_counter_ns())
        return original_ops_prepare(*args, **kwargs)

    def intercept_load(prepared: _PreparedJITBuild) -> object:
        if stop_before_build:
            if len(legacy_boundaries) != 1:
                raise RuntimeError("compilation reached an ambiguous legacy boundary")
            raise _CompilationCaptured(
                time.perf_counter_ns(),
                legacy_boundaries[0],
                prepared,
            )
        captures.append(_prepared_source_summary(prepared))
        return original_ops_load(prepared)

    def intercept_context_post_init(context: CompilationContext) -> None:
        original_context_post_init(context)
        compilation_contexts.append(context)

    ops._prepare_jit_build = intercept_prepare
    setattr(stensor_module, "_prepare_jit_build", intercept_prepare)
    ops._load_validated_prepared_kernel = intercept_load
    setattr(stensor_module, "_load_validated_prepared_kernel", intercept_load)
    CompilationContext.__post_init__ = intercept_context_post_init
    try:
        yield
    finally:
        ops._prepare_jit_build = original_ops_prepare
        setattr(stensor_module, "_prepare_jit_build", original_stensor_prepare)
        ops._load_validated_prepared_kernel = original_ops_load
        setattr(
            stensor_module,
            "_load_validated_prepared_kernel",
            original_stensor_load,
        )
        CompilationContext.__post_init__ = original_context_post_init


def _clear_compiler_caches() -> None:
    ops._kernel_cache.clear()
    ops._einsum_dispatch_cache.clear()


def _time_captured_compilation(
    case: LatencyCase,
) -> tuple[float, float, Dict[str, Any], List[Dict[str, Any]]]:
    _clear_compiler_caches()
    captures: List[Dict[str, Any]] = []
    legacy_boundaries: List[int] = []
    compilation_contexts: List[CompilationContext] = []
    with _intercept_native_builds(
        captures,
        legacy_boundaries,
        compilation_contexts,
        stop_before_build=True,
    ):
        start = time.perf_counter_ns()
        try:
            case.invoke()
        except _CompilationCaptured as captured:
            compatible_elapsed_ms = (captured.legacy_boundary_ns - start) / 1_000_000.0
            canonical_elapsed_ms = (captured.boundary_ns - start) / 1_000_000.0
            build = _prepared_source_summary(captured.prepared)
        else:
            raise RuntimeError(f"{case.name} did not reach the native build boundary")
    if captures:
        raise RuntimeError(f"{case.name} unexpectedly entered the native build")
    if len(compilation_contexts) != 1:
        raise RuntimeError(
            f"{case.name} must publish exactly one compilation timing owner; "
            f"observed {len(compilation_contexts)}"
        )
    if not compilation_contexts[0].stage_run_records:
        raise RuntimeError(f"{case.name} published no compiler-stage records")
    stage_runs = [
        {
            "sequence_index": record.sequence_index,
            "stage_id": record.stage_id.value,
            "duration_ns": record.duration_ns,
            "nested_within": (
                record.nested_within.value if record.nested_within is not None else None
            ),
        }
        for context in compilation_contexts
        for record in context.stage_run_records
    ]
    return compatible_elapsed_ms, canonical_elapsed_ms, build, stage_runs


def _stage_timing_summary(
    samples: Sequence[Sequence[Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    stage_ids = sorted({str(run["stage_id"]) for sample in samples for run in sample})
    summary: Dict[str, Dict[str, Any]] = {}
    for stage_id in stage_ids:
        durations_ms: List[float] = []
        runs_per_sample: List[int] = []
        nested_within: set[Optional[str]] = set()
        for sample in samples:
            matching = [run for run in sample if run["stage_id"] == stage_id]
            runs_per_sample.append(len(matching))
            durations_ms.append(
                sum(float(run["duration_ns"]) for run in matching) / 1_000_000.0
            )
            nested_within.update(run["nested_within"] for run in matching)
        summary[stage_id] = {
            "samples_ms": durations_ms,
            "p50_ms": _percentile(durations_ms, 0.50),
            "p95_ms": _percentile(durations_ms, 0.95),
            "runs_per_sample": runs_per_sample,
            "nested_within": sorted(
                value for value in nested_within if value is not None
            ),
        }
    return summary


def run_latency(args: argparse.Namespace) -> int:
    _require_imported_scorch_from_worktree()
    if args.warmup < 0 or args.samples < 1:
        raise ValueError("warmup must be nonnegative and samples must be positive")
    cases = _build_latency_cases()
    case_results: List[Dict[str, Any]] = []

    print("Phase 0 Python compiler latency (native build and execution excluded)")
    print(f"warmup={args.warmup} samples={args.samples}")
    print(
        f"{'case':<20} {'compat50':>10} {'compat95':>10} "
        f"{'canon50':>10} {'canon95':>10} {'source_sha256':>16}"
    )
    for case in cases:
        for _ in range(args.warmup):
            _time_captured_compilation(case)
        samples: List[float] = []
        canonical_samples: List[float] = []
        builds: List[Dict[str, Any]] = []
        stage_samples: List[List[Dict[str, Any]]] = []
        for _ in range(args.samples):
            compatible_ms, canonical_ms, build, stage_runs = _time_captured_compilation(
                case
            )
            samples.append(compatible_ms)
            canonical_samples.append(canonical_ms)
            builds.append(build)
            stage_samples.append(stage_runs)
        if any(build != builds[0] for build in builds[1:]):
            raise RuntimeError(
                f"{case.name} emitted nondeterministic compiler build inputs"
            )
        p50_ms = _percentile(samples, 0.50)
        p95_ms = _percentile(samples, 0.95)
        canonical_p50_ms = _percentile(canonical_samples, 0.50)
        canonical_p95_ms = _percentile(canonical_samples, 0.95)
        endpoint_extension_samples = [
            canonical - compatible
            for compatible, canonical in zip(samples, canonical_samples)
        ]
        print(
            f"{case.name:<20} {p50_ms:10.3f} {p95_ms:10.3f} "
            f"{canonical_p50_ms:10.3f} {canonical_p95_ms:10.3f} "
            f"{builds[0]['source_sha256'][:16]:>16}"
        )
        case_results.append(
            {
                "name": case.name,
                "operation": case.operation,
                "formats": list(case.formats),
                "output_format": case.output_format,
                "samples_ms": samples,
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "canonical_samples_ms": canonical_samples,
                "canonical_p50_ms": canonical_p50_ms,
                "canonical_p95_ms": canonical_p95_ms,
                "canonical_endpoint_extension_samples_ms": endpoint_extension_samples,
                "canonical_endpoint_extension_p50_ms": _percentile(
                    endpoint_extension_samples, 0.50
                ),
                "canonical_endpoint_extension_p95_ms": _percentile(
                    endpoint_extension_samples, 0.95
                ),
                "stage_timing": _stage_timing_summary(stage_samples),
                "build": builds[0],
            }
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "compiler-ir-phase0-latency",
        "corpus_version": LATENCY_CORPUS_VERSION,
        "metadata": _metadata(),
        "configuration": {"warmup": args.warmup, "samples": args.samples},
        "cases": case_results,
    }
    _write_result(result, args.output)
    return 0


def _load_latency_result(path: Path) -> Mapping[str, Any]:
    result = json.loads(path.read_text())
    if result.get("kind") != "compiler-ir-phase0-latency":
        raise ValueError(f"{path} is not a compiler latency result")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} uses an unsupported result schema")
    return result


def run_compare_latency(args: argparse.Namespace) -> int:
    baseline = _load_latency_result(args.baseline)
    candidate = _load_latency_result(args.candidate)
    if baseline["corpus_version"] != candidate["corpus_version"]:
        raise ValueError("baseline and candidate use different latency corpora")
    if baseline["configuration"] != candidate["configuration"]:
        raise ValueError("baseline and candidate use different latency settings")
    if _machine_identity(baseline) != _machine_identity(candidate):
        raise ValueError("baseline and candidate were measured on different machines")

    baseline_cases = {case["name"]: case for case in baseline["cases"]}
    candidate_cases = {case["name"]: case for case in candidate["cases"]}
    if baseline_cases.keys() != candidate_cases.keys():
        raise ValueError("baseline and candidate latency cases differ")

    print("Python compiler latency comparison")
    print(f"{'case':<20} {'p50 new/old':>13} {'p95 new/old':>13} {'status':>12}")
    investigation_required = False
    for name in baseline_cases:
        old = baseline_cases[name]
        new = candidate_cases[name]
        p50_ratio = new["p50_ms"] / old["p50_ms"]
        p95_ratio = new["p95_ms"] / old["p95_ms"]
        within_target = (
            p50_ratio <= LATENCY_MAX_RATIO and p95_ratio <= LATENCY_MAX_RATIO
        )
        investigation_required = investigation_required or not within_target
        print(
            f"{name:<20} {p50_ratio:13.3f} {p95_ratio:13.3f} "
            f"{'TARGET' if within_target else 'INVESTIGATE':>12}"
        )
    print(f"target: {LATENCY_MAX_RATIO:.2f}x for both p50 and p95 in every category")
    print(
        "A crossing requires attribution under the compiler-latency policy; "
        "it is not an automatic rejection."
    )
    return 1 if investigation_required else 0


def _build_csr(
    rows: int, cols: int, density: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    dense = torch.randn(rows, cols, generator=generator, dtype=torch.float32)
    dense *= torch.rand(rows, cols, generator=generator) < density
    return dense, dense.to_sparse_csr()


def _generated_spmm(
    sparse_left: torch.Tensor, dense_right: torch.Tensor
) -> tuple[torch.Tensor, float]:
    timing: Dict[str, float] = {}
    output = ops.matmul(sparse_left, dense_right, use_cache=False, time_dict=timing)
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("generated dense SpMM did not return a torch.Tensor")
    return output, timing["eval_time"] * 1_000.0


def _lane_median(
    sparse_left: torch.Tensor, dense_right: torch.Tensor, calls: int
) -> tuple[float, List[float]]:
    samples = [_generated_spmm(sparse_left, dense_right)[1] for _ in range(calls)]
    return statistics.median(samples), samples


def _symmetric_band(ratios: Sequence[float]) -> Dict[str, float]:
    symmetric = list(ratios) + [1.0 / ratio for ratio in ratios]
    return {"low": min(symmetric), "high": max(symmetric)}


def _geomean(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def run_kernel_aa(args: argparse.Namespace) -> int:
    _require_imported_scorch_from_worktree()
    if args.warmup < 0 or args.rounds < 2 or args.calls < 1:
        raise ValueError("warmup must be nonnegative, rounds >= 2, and calls positive")
    if args.threads:
        torch.set_num_threads(args.threads)
    rows_grid = QUICK_ROWS if args.quick else FULL_ROWS
    free_grid = QUICK_FREE_DIMS if args.quick else FULL_FREE_DIMS
    density_grid = QUICK_DENSITIES if args.quick else FULL_DENSITIES

    _clear_compiler_caches()
    build_captures: List[Dict[str, Any]] = []
    builds_by_shape: Dict[tuple[int, int, int], Dict[str, Any]] = {}
    cells: List[Dict[str, Any]] = []
    print("Generated-kernel same-binary A/A control")
    print(
        f"grid={len(rows_grid) * len(free_grid) * len(density_grid)} cells "
        f"warmup={args.warmup} rounds={args.rounds} calls={args.calls}"
    )
    print(f"{'M':>7} {'N':>5} {'density':>8} {'median_ms':>11} {'band':>19}")

    with _intercept_native_builds(
        build_captures,
        [],
        [],
        stop_before_build=False,
    ):
        for rows in rows_grid:
            cols = rows
            for density in density_grid:
                dense_left, sparse_left = _build_csr(rows, cols, density, args.seed)
                nnz = int(sparse_left.values().numel())
                for free_dim in free_grid:
                    generator = torch.Generator().manual_seed(args.seed + 1)
                    dense_right = torch.randn(
                        cols, free_dim, generator=generator, dtype=torch.float32
                    )
                    reference = dense_left @ dense_right
                    capture_start = len(build_captures)
                    output, _ = _generated_spmm(sparse_left, dense_right)
                    new_builds = build_captures[capture_start:]
                    if len(new_builds) > 1:
                        raise RuntimeError(
                            "one kernel cell reached multiple native build boundaries"
                        )
                    shape_key = (rows, cols, free_dim)
                    if new_builds:
                        cell_build = new_builds[0]
                        builds_by_shape[shape_key] = cell_build
                    elif shape_key in builds_by_shape:
                        # The second density for a shape reuses the already-loaded
                        # generated module.
                        cell_build = builds_by_shape[shape_key]
                    else:
                        raise RuntimeError("kernel execution produced no build capture")
                    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-3)
                    for _ in range(args.warmup):
                        _generated_spmm(sparse_left, dense_right)

                    lane_a: List[float] = []
                    lane_b: List[float] = []
                    lane_a_raw: List[List[float]] = []
                    lane_b_raw: List[List[float]] = []
                    for round_index in range(args.rounds):
                        if round_index % 2 == 0:
                            a_median, a_raw = _lane_median(
                                sparse_left, dense_right, args.calls
                            )
                            b_median, b_raw = _lane_median(
                                sparse_left, dense_right, args.calls
                            )
                        else:
                            b_median, b_raw = _lane_median(
                                sparse_left, dense_right, args.calls
                            )
                            a_median, a_raw = _lane_median(
                                sparse_left, dense_right, args.calls
                            )
                        lane_a.append(a_median)
                        lane_b.append(b_median)
                        lane_a_raw.append(a_raw)
                        lane_b_raw.append(b_raw)

                    ratios = [right / left for left, right in zip(lane_a, lane_b)]
                    band = _symmetric_band(ratios)
                    all_samples = [*lane_a, *lane_b]
                    median_ms = statistics.median(all_samples)
                    print(
                        f"{rows:7d} {free_dim:5d} {density:8.3f} "
                        f"{median_ms:11.3f} [{band['low']:.3f}, {band['high']:.3f}]"
                    )
                    cells.append(
                        {
                            "M": rows,
                            "K": cols,
                            "N": free_dim,
                            "density": density,
                            "nnz": nnz,
                            "correct": True,
                            "lane_a_round_medians_ms": lane_a,
                            "lane_b_round_medians_ms": lane_b,
                            "lane_a_samples_ms": lane_a_raw,
                            "lane_b_samples_ms": lane_b_raw,
                            "control_ratios_b_over_a": ratios,
                            "control_band": band,
                            "baseline_median_ms": median_ms,
                            "build": cell_build,
                        }
                    )

    round_geomeans = [
        _geomean([cell["control_ratios_b_over_a"][round_index] for cell in cells])
        for round_index in range(args.rounds)
    ]
    machine_band = _symmetric_band(round_geomeans)
    unique_builds = {capture["source_sha256"]: capture for capture in build_captures}
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "compiler-ir-generated-kernel-aa",
        "corpus_version": KERNEL_CORPUS_VERSION,
        "metadata": _metadata(),
        "configuration": {
            "quick": args.quick,
            "warmup": args.warmup,
            "rounds": args.rounds,
            "calls": args.calls,
            "seed": args.seed,
            "threads": torch.get_num_threads(),
            "rows": list(rows_grid),
            "free_dims": list(free_grid),
            "densities": list(density_grid),
        },
        "builds": [unique_builds[digest] for digest in sorted(unique_builds)],
        "machine_control_round_geomeans": round_geomeans,
        "machine_control_band": machine_band,
        "cells": cells,
    }
    print(
        "machine geomean A/A band: "
        f"[{machine_band['low']:.3f}, {machine_band['high']:.3f}]"
    )
    _write_result(result, args.output)
    return 0


def _cell_key(cell: Mapping[str, Any]) -> tuple[int, int, int, float]:
    return (
        int(cell["M"]),
        int(cell["K"]),
        int(cell["N"]),
        float(cell["density"]),
    )


def _load_kernel_result(path: Path) -> Mapping[str, Any]:
    result = json.loads(path.read_text())
    if result.get("kind") != "compiler-ir-generated-kernel-aa":
        raise ValueError(f"{path} is not a generated-kernel A/A result")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} uses an unsupported result schema")
    return result


def run_compare(args: argparse.Namespace) -> int:
    baseline = _load_kernel_result(args.baseline)
    candidate = _load_kernel_result(args.candidate)
    if baseline["corpus_version"] != candidate["corpus_version"]:
        raise ValueError("baseline and candidate use different corpus versions")
    if baseline["configuration"] != candidate["configuration"]:
        raise ValueError("baseline and candidate use different benchmark settings")
    if _machine_identity(baseline) != _machine_identity(candidate):
        raise ValueError("baseline and candidate were measured on different machines")

    baseline_cells = {_cell_key(cell): cell for cell in baseline["cells"]}
    candidate_cells = {_cell_key(cell): cell for cell in candidate["cells"]}
    if baseline_cells.keys() != candidate_cells.keys():
        raise ValueError("baseline and candidate grids differ")

    same_source = all(
        baseline_cells[key]["build"] == candidate_cells[key]["build"]
        for key in baseline_cells
    )
    print("Generated-kernel baseline/candidate comparison")
    print(f"byte-identical build input: {same_source}")
    if same_source:
        print("runtime gate waived; structural activation tests remain required")
        return 0

    print(f"{'M':>7} {'N':>5} {'density':>8} {'new/old':>9} {'band':>19} {'status':>7}")
    ratios: List[float] = []
    failures: List[str] = []
    for key in sorted(baseline_cells):
        old = baseline_cells[key]
        new = candidate_cells[key]
        ratio = new["baseline_median_ms"] / old["baseline_median_ms"]
        low = new["control_band"]["low"]
        high = new["control_band"]["high"]
        passed = low <= ratio <= high
        status = "PASS" if passed else "FAIL"
        print(
            f"{key[0]:7d} {key[2]:5d} {key[3]:8.3f} {ratio:9.3f} "
            f"[{low:.3f}, {high:.3f}] {status:>7}"
        )
        ratios.append(ratio)
        if not passed:
            failures.append(f"cell {key}: ratio {ratio:.6f} outside [{low}, {high}]")

    ratio_geomean = _geomean(ratios)
    machine_low = candidate["machine_control_band"]["low"]
    machine_high = candidate["machine_control_band"]["high"]
    machine_passed = machine_low <= ratio_geomean <= machine_high
    print(
        f"machine geomean new/old={ratio_geomean:.3f} "
        f"band=[{machine_low:.3f}, {machine_high:.3f}] "
        f"{'PASS' if machine_passed else 'FAIL'}"
    )
    if not machine_passed:
        failures.append("machine geomean is outside the A/A-calibrated band")
    return 1 if failures else 0


def _output_path(value: str) -> Optional[Path]:
    return None if not value else Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    latency = subparsers.add_parser("latency", help="measure Python compile latency")
    latency.add_argument("--warmup", type=int, default=5)
    latency.add_argument("--samples", type=int, default=30)
    latency.add_argument("--output", type=_output_path, default=None)
    latency.set_defaults(handler=run_latency)

    compare_latency = subparsers.add_parser(
        "compare-latency", help="compare latency results to the Phase 0 target"
    )
    compare_latency.add_argument("baseline", type=Path)
    compare_latency.add_argument("candidate", type=Path)
    compare_latency.set_defaults(handler=run_compare_latency)

    kernel = subparsers.add_parser(
        "kernel-aa", help="measure generated-kernel same-binary A/A noise"
    )
    kernel.add_argument("--quick", action="store_true")
    kernel.add_argument("--warmup", type=int, default=3)
    kernel.add_argument("--rounds", type=int, default=5)
    kernel.add_argument("--calls", type=int, default=3)
    kernel.add_argument("--seed", type=int, default=0)
    kernel.add_argument("--threads", type=int, default=0)
    kernel.add_argument("--output", type=_output_path, default=None)
    kernel.set_defaults(handler=run_kernel_aa)

    compare = subparsers.add_parser(
        "compare", help="compare two runs against their A/A control bands"
    )
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.set_defaults(handler=run_compare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
