#!/usr/bin/env python3
"""Steady-state before/after benchmark for Scorch native ownership changes.

Run with a source snapshot on PYTHONPATH and an isolated TORCH_EXTENSIONS_DIR.
The timed functions are the native pybind entry points, so Python scheduling,
lowering, validation, and STensor wrapping are excluded.  Compilation is done
once per generated-kernel family and reported separately.

The fixed-seed suite covers 139 cases across dynamic sparse assembly, format
conversion, 1-D/2-D/3-D expressions, reductions, SpGEMM, SDDMM, and SpMM.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

import scorch
import scorch_ops as native_extension
import scorch.utils as scorch_utils
from scorch import STensor
from scorch.prebuilt_kernels import resolve_prebuilt_matmul

NativeFn = Callable[[], Any]


def git_value(args: Sequence[str], cwd: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def make_dense(shape: Sequence[int], density: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    if density < 1.0:
        mask = torch.rand(tuple(shape), generator=generator) < density
        values.mul_(mask)
    return values


def make_csr(
    rows: int,
    cols: int,
    degree: int,
    seed: int,
    *,
    irregular: bool = False,
) -> tuple[STensor, torch.Tensor]:
    """Build sorted, duplicate-free CSR without materializing a dense mask."""
    rng = np.random.default_rng(seed)
    degree = max(0, min(int(degree), cols))
    row_degrees: list[int] = []
    columns: list[np.ndarray] = []
    for row in range(rows):
        if irregular and degree:
            # A bounded heavy tail: many short rows and a few much longer rows.
            scale = min(8.0, max(0.125, float(rng.lognormal(0.0, 1.15))))
            row_degree = min(cols, max(0, int(round(degree * scale))))
        else:
            row_degree = degree
        row_degrees.append(row_degree)
        if row_degree:
            columns.append(np.sort(rng.choice(cols, row_degree, replace=False)))
        else:
            columns.append(np.empty(0, dtype=np.int64))
    crow_np = np.empty(rows + 1, dtype=np.int32)
    crow_np[0] = 0
    np.cumsum(np.asarray(row_degrees, dtype=np.int64), out=crow_np[1:])
    col_np = (
        np.concatenate(columns).astype(np.int32, copy=False)
        if columns
        else np.empty(0, dtype=np.int32)
    )
    val_gen = torch.Generator().manual_seed(seed + 10_000)
    values = torch.randn(col_np.size, generator=val_gen, dtype=torch.float32)
    tensor = torch.sparse_csr_tensor(
        torch.from_numpy(crow_np),
        torch.from_numpy(col_np),
        values,
        size=(rows, cols),
    )
    return STensor.from_csr(tensor, f"csr_{seed}"), tensor


def to_dcsr(tensor: STensor) -> STensor:
    crow, col = tensor.storage.index.mode_indices[1]
    counts = crow[1:] - crow[:-1]
    nonempty = torch.nonzero(counts, as_tuple=False).flatten().to(torch.int32)
    outer_pos = torch.tensor([0, nonempty.numel()], dtype=torch.int32)
    if nonempty.numel():
        selected_counts = counts[nonempty.to(torch.int64)].to(torch.int32)
        inner_pos = torch.empty(nonempty.numel() + 1, dtype=torch.int32)
        inner_pos[0] = 0
        inner_pos[1:] = torch.cumsum(selected_counts, dim=0)
    else:
        inner_pos = torch.zeros(1, dtype=torch.int32)
    return STensor.from_components(
        tensor.shape,
        "ss",
        [[outer_pos, nonempty], [inner_pos, col.to(torch.int32)]],
        tensor.values,
        name="dcsr",
        index_dtype=torch.int32,
    )


def to_coo(tensor: STensor) -> STensor:
    crow, col = tensor.storage.index.mode_indices[1]
    counts = (crow[1:] - crow[:-1]).to(torch.int64)
    rows = torch.repeat_interleave(
        torch.arange(tensor.shape[0], dtype=torch.int32), counts
    )
    return STensor.from_components(
        tensor.shape,
        "oo",
        [[rows], [col.to(torch.int32)]],
        tensor.values,
        name="coo",
        index_dtype=torch.int32,
    )


def unary_args(result_shape: Sequence[int], tensor: STensor) -> tuple[Any, ...]:
    return (
        tuple(result_shape),
        tensor.shape,
        tensor._native_mode_indices(),
        tensor.values,
    )


def binary_args(
    result_shape: Sequence[int], left: STensor, right: STensor
) -> tuple[Any, ...]:
    return (
        tuple(result_shape),
        left.shape,
        left._native_mode_indices(),
        left.values,
        right.shape,
        right._native_mode_indices(),
        right.values,
    )


def ternary_args(
    result_shape: Sequence[int], first: STensor, second: STensor, third: STensor
) -> tuple[Any, ...]:
    return (
        tuple(result_shape),
        first.shape,
        first._native_mode_indices(),
        first.values,
        second.shape,
        second._native_mode_indices(),
        second.values,
        third.shape,
        third._native_mode_indices(),
        third.values,
    )


def generated_tail_info(kernel_name: str) -> dict[str, Any]:
    ext_root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", ""))
    main_cpp = ext_root / kernel_name / "main.cpp"
    if not main_cpp.exists():
        return {"kernel": kernel_name, "source": str(main_cpp), "source_missing": True}
    source = main_cpp.read_text(encoding="utf-8")
    start = source.find("Tensor evaluate(")
    tail = source[start:] if start >= 0 else ""
    return {
        "kernel": kernel_name,
        "source": str(main_cpp),
        "cvector_instances": tail.count("cvector<"),
        "coo_workspace_instances": tail.count("coo_workspace_1d<"),
        "linked_workspace_instances": tail.count("linked_list_workspace_1d<"),
        "malloc_calls": tail.count("malloc("),
        "calloc_calls": tail.count("calloc("),
        "from_blob_calls": tail.count("torch::from_blob("),
        "torch_empty_calls": tail.count("torch::empty("),
    }


def capture_generated(
    label: str, compile_call: Callable[[], Any]
) -> tuple[Any, dict[str, Any], Any]:
    before = list(scorch_utils._so_cache)
    start = time.perf_counter()
    result = compile_call()
    compile_seconds = time.perf_counter() - start
    new_names = [name for name in scorch_utils._so_cache if name not in before]
    if not new_names:
        raise RuntimeError(f"{label}: compile call did not load a generated module")
    # Relayout helpers, if any, load before the final expression module.
    kernel_name = new_names[-1]
    module = scorch_utils._so_cache[kernel_name]
    info = generated_tail_info(kernel_name)
    info.update(
        {
            "label": label,
            "compile_ms": compile_seconds * 1e3,
            "all_new_modules": new_names,
        }
    )
    print(
        f"COMPILE {label:24s} {compile_seconds:8.3f}s  {kernel_name}  "
        f"cvector={info.get('cvector_instances')} "
        f"coo_wksp={info.get('coo_workspace_instances')} "
        f"linked_wksp={info.get('linked_workspace_instances')}",
        flush=True,
    )
    return module, info, result


def percentile(values: Sequence[float], fraction: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def measure(
    fn: NativeFn,
    *,
    samples: int,
    target_seconds: float,
    max_batch: int,
) -> tuple[dict[str, Any], int]:
    # Prime lazy OpenMP teams and allocator state before calibration.
    warmup_output = fn()
    del warmup_output
    once: list[float] = []
    for _ in range(3):
        start = time.perf_counter_ns()
        sample_output = fn()
        once.append((time.perf_counter_ns() - start) * 1e-9)
        del sample_output
    per_call = max(statistics.median(once), 1e-9)
    batch = max(1, min(max_batch, int(math.ceil(target_seconds / per_call))))

    def run_batch() -> float:
        last_result = None
        start = time.perf_counter_ns()
        for _ in range(batch):
            last_result = fn()
        elapsed = (time.perf_counter_ns() - start) * 1e-9 / batch
        del last_result
        return elapsed

    for _ in range(3):
        run_batch()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        seconds = [run_batch() for _ in range(samples)]
    finally:
        if gc_was_enabled:
            gc.enable()
    microseconds = [value * 1e6 for value in seconds]
    median = statistics.median(microseconds)
    mad = statistics.median(abs(value - median) for value in microseconds)
    mean = statistics.mean(microseconds)
    stdev = statistics.stdev(microseconds) if len(microseconds) > 1 else 0.0
    summary = {
        "median_us": median,
        "p10_us": percentile(microseconds, 0.10),
        "p90_us": percentile(microseconds, 0.90),
        "min_us": min(microseconds),
        "max_us": max(microseconds),
        "mad_us": mad,
        "cv_pct": (stdev / mean * 100.0) if mean else 0.0,
        "samples_us": microseconds,
    }
    return summary, batch


class Runner:
    def __init__(self, args: argparse.Namespace, metadata: dict[str, Any]):
        self.args = args
        self.metadata = metadata
        self.rows: list[dict[str, Any]] = []
        self.kernels: dict[str, dict[str, Any]] = {}
        self.csv_path = Path(f"{args.output_prefix}.csv")
        self.json_path = Path(f"{args.output_prefix}.json")

    def add_kernel(self, family: str, info: dict[str, Any]) -> None:
        self.kernels[family] = info

    def run(
        self,
        family: str,
        case: str,
        fn: NativeFn,
        *,
        backend: str,
        rank: int,
        shape: str,
        input_nnz: int,
        output_format: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        gc.collect()
        output = fn()
        output_nnz = int(output.storage.value.numel())
        del output
        summary, batch = measure(
            fn,
            samples=self.args.samples,
            target_seconds=self.args.target_ms / 1000.0,
            max_batch=self.args.max_batch,
        )
        row: dict[str, Any] = {
            "phase": self.args.phase,
            "family": family,
            "case": case,
            "backend": backend,
            "rank": rank,
            "shape": shape,
            "input_nnz": input_nnz,
            "output_nnz": output_nnz,
            "output_format": output_format,
            "batch": batch,
            **summary,
        }
        if extra:
            row.update(extra)
        self.rows.append(row)
        print(
            f"RESULT  {family:24s} {case:30s} "
            f"{summary['median_us']:11.3f} us  "
            f"p10/p90={summary['p10_us']:.3f}/{summary['p90_us']:.3f} "
            f"batch={batch} nnz={input_nnz}->{output_nnz}",
            flush=True,
        )
        self.save()

    def save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_rows = []
        for row in self.rows:
            csv_rows.append(
                {key: value for key, value in row.items() if key != "samples_us"}
            )
        fieldnames = (
            sorted({key for row in csv_rows for key in row}) if csv_rows else []
        )
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        payload = {
            "metadata": self.metadata,
            "kernels": self.kernels,
            "results": self.rows,
        }
        temp_path = self.json_path.with_suffix(self.json_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(self.json_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark native ownership-sensitive Scorch kernels across a "
            "fixed 139-case synthetic suite."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("before", "after"),
        default="before",
        help="label stored in every result row (default: before)",
    )
    parser.add_argument(
        "--output-prefix",
        default="/tmp/scorch-raii-before",
        help="path prefix for the generated .json and .csv files",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=13,
        help="number of independently timed samples per case (default: 13)",
    )
    parser.add_argument(
        "--target-ms",
        type=float,
        default=20.0,
        help="target duration of each adaptively batched sample (default: 20)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=2048,
        help="maximum native calls in one timed sample (default: 2048)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=6,
        help="Torch/OpenMP thread count used by the suite (default: 6)",
    )
    parser.add_argument(
        "--source-commit",
        default=os.environ.get("SCORCH_SOURCE_COMMIT", ""),
        help=(
            "source revision recorded in metadata; defaults to "
            "SCORCH_SOURCE_COMMIT or the current Git HEAD"
        ),
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260712)
    np.random.seed(20260712)
    source_root = str(Path(scorch.__file__).resolve().parents[2])
    native_path = Path(native_extension.__file__).resolve()
    native_sha256 = hashlib.sha256(native_path.read_bytes()).hexdigest()
    metadata = {
        "phase": args.phase,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": source_root,
        "git_commit": args.source_commit
        or git_value(["rev-parse", "HEAD"], source_root),
        "git_status": git_value(["status", "--short"], source_root),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "scorch_file": scorch.__file__,
        "native_extension": str(native_path),
        "native_extension_sha256": native_sha256,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR"),
        "benchmark": {
            "samples": args.samples,
            "target_ms": args.target_ms,
            "max_batch": args.max_batch,
            "seed": 20260712,
            "timing_scope": "direct native pybind evaluate call",
        },
        "parallel_info": torch.__config__.parallel_info(),
    }
    runner = Runner(args, metadata)
    generated_modules: dict[tuple[str, str], Any] = {}

    def module_for(family: str, shape_key: str, compile_call: Callable[[], Any]) -> Any:
        cache_key = (family, shape_key)
        if cache_key not in generated_modules:
            label = f"{family}[{shape_key}]"
            module, info, _ = capture_generated(label, compile_call)
            generated_modules[cache_key] = module
            runner.add_kernel(label, info)
        return generated_modules[cache_key]

    # ------------------------------------------------------------------
    # Rank-2 dense -> sparse conversions: square, tall, wide, tiny through 2M cells.
    # ------------------------------------------------------------------
    conversion_cases = [
        ("tiny_16x16_p05", (16, 16), 0.05),
        ("small_64x64_p50", (64, 64), 0.50),
        ("mid_256x256_p01", (256, 256), 0.01),
        ("mid_256x256_p50", (256, 256), 0.50),
        ("tall_4096x16_p02", (4096, 16), 0.02),
        ("wide_16x4096_p10", (16, 4096), 0.10),
        ("large_1024x1024_p001", (1024, 1024), 0.001),
        ("large_1024x1024_p10", (1024, 1024), 0.10),
        ("large_1024x1024_p50", (1024, 1024), 0.50),
        ("wide_512x4096_p01", (512, 4096), 0.01),
    ]
    for fmt, family in (
        ("ds", "dense2_to_csr"),
        ("ss", "dense2_to_dcsr"),
        ("oo", "dense2_to_coo"),
    ):
        for index, (name, shape, density) in enumerate(conversion_cases):
            dense = make_dense(shape, density, 1000 + index)
            tensor = STensor.from_torch(dense, "dense")
            shape_key = "x".join(map(str, shape))
            module = module_for(
                family, shape_key, lambda t=tensor, f=fmt: t.copy().to_sparse(f)
            )
            native_args = unary_args(shape, tensor)
            runner.run(
                family,
                name,
                lambda m=module, a=native_args: m.evaluate(*a),
                backend="generated",
                rank=2,
                shape="x".join(map(str, shape)),
                input_nnz=int(torch.count_nonzero(dense)),
                output_format=fmt,
                extra={"density": density, "cells": int(np.prod(shape))},
            )

    # COO -> CSR/DCSR conversion exercises sparse-input dynamic assembly.
    coo_convert_cases = [
        ("tiny_32x32_d4", 32, 32, 4),
        ("square_512_d4", 512, 512, 4),
        ("square_512_d64", 512, 512, 64),
        ("tall_4096x64_d4", 4096, 64, 4),
        ("wide_64x4096_d16", 64, 4096, 16),
    ]
    for fmt, family in (("ds", "coo2_to_csr"), ("ss", "coo2_to_dcsr")):
        for index, (name, rows, cols, degree) in enumerate(coo_convert_cases):
            csr, _ = make_csr(rows, cols, degree, 210 + index)
            coo = to_coo(csr)
            shape_key = f"{rows}x{cols}"
            module = module_for(
                family, shape_key, lambda t=coo, f=fmt: t.copy().to_sparse(f)
            )
            native_args = unary_args(coo.shape, coo)
            runner.run(
                family,
                name,
                lambda m=module, a=native_args: m.evaluate(*a),
                backend="generated",
                rank=2,
                shape=f"{rows}x{cols}",
                input_nnz=int(coo.values.numel()),
                output_format=fmt,
                extra={"degree": degree},
            )

    # ------------------------------------------------------------------
    # Rank-1 sparse union and rank-2 sparse union/intersection.
    # ------------------------------------------------------------------
    for index, (size, density) in enumerate(
        ((64, 0.1), (4096, 0.01), (262144, 0.001), (262144, 0.1), (1048576, 0.01))
    ):
        left_dense = make_dense((size,), density, 310 + index)
        right_dense = make_dense((size,), density, 320 + index)
        left = STensor.from_torch(left_dense, "a").to_sparse("s")
        right = STensor.from_torch(right_dense, "b").to_sparse("s")
        one_module = module_for(
            "sparse1_add", str(size), lambda left_arg=left, r=right: left_arg + r
        )
        native_args = binary_args((size,), left, right)
        runner.run(
            "sparse1_add",
            f"n{size}_p{density:g}",
            lambda m=one_module, a=native_args: m.evaluate(*a),
            backend="generated",
            rank=1,
            shape=str(size),
            input_nnz=int(left.values.numel() + right.values.numel()),
            output_format="s",
            extra={"density": density},
        )

    sparse2_cases = [
        ("tiny_32_d4", 32, 32, 4, False),
        ("square_256_d4", 256, 256, 4, False),
        ("square_256_d64", 256, 256, 64, False),
        ("square_1024_d2", 1024, 1024, 2, False),
        ("square_1024_d32", 1024, 1024, 32, False),
        ("tall_4096x64_d4", 4096, 64, 4, False),
        ("wide_64x4096_d32", 64, 4096, 32, False),
        ("irregular_2048_d16", 2048, 2048, 16, True),
    ]
    for index, (name, rows, cols, degree, irregular) in enumerate(sparse2_cases):
        left, _ = make_csr(rows, cols, degree, 410 + 2 * index, irregular=irregular)
        right, _ = make_csr(rows, cols, degree, 411 + 2 * index, irregular=irregular)
        shape_key = f"{rows}x{cols}"
        add_module = module_for(
            "csr2_add", shape_key, lambda left_arg=left, r=right: left_arg + r
        )
        mul_module = module_for(
            "csr2_intersection",
            shape_key,
            lambda left_arg=left, r=right: scorch.einsum(
                "ij,ij->ij", left_arg, r, format="ds"
            ),
        )
        native_args = binary_args((rows, cols), left, right)
        input_nnz = int(left.values.numel() + right.values.numel())
        common = {
            "backend": "generated",
            "rank": 2,
            "shape": f"{rows}x{cols}",
            "input_nnz": input_nnz,
            "extra": {"degree": degree, "irregular": irregular},
        }
        runner.run(
            "csr2_add",
            name,
            lambda m=add_module, a=native_args: m.evaluate(*a),
            output_format="ds",
            **common,
        )
        runner.run(
            "csr2_intersection",
            name,
            lambda m=mul_module, a=native_args: m.evaluate(*a),
            output_format="ds",
            **common,
        )
        dleft, dright = to_dcsr(left), to_dcsr(right)
        dadd_module = module_for(
            "dcsr2_add",
            shape_key,
            lambda left_arg=dleft, r=dright: left_arg + r,
        )
        dargs = binary_args((rows, cols), dleft, dright)
        runner.run(
            "dcsr2_add",
            name,
            lambda m=dadd_module, a=dargs: m.evaluate(*a),
            output_format="ss",
            **common,
        )

    # ------------------------------------------------------------------
    # Rank-3 conversion and intersection (CSF and coordinate output).
    # ------------------------------------------------------------------
    conversion3_cases = [
        ("tiny_8x8x8_p10", (8, 8, 8), 0.10),
        ("rect_16x32x64_p01", (16, 32, 64), 0.01),
        ("rect_16x32x64_p20", (16, 32, 64), 0.20),
        ("wide_4x64x512_p01", (4, 64, 512), 0.01),
        ("cube_64_p001", (64, 64, 64), 0.001),
        ("cube_64_p10", (64, 64, 64), 0.10),
    ]
    for fmt, family in (("sss", "dense3_to_csf"), ("ooo", "dense3_to_coo")):
        for index, (name, shape, density) in enumerate(conversion3_cases):
            dense = make_dense(shape, density, 510 + index)
            tensor = STensor.from_torch(dense, "dense3")
            shape_key = "x".join(map(str, shape))
            module = module_for(
                family, shape_key, lambda t=tensor, f=fmt: t.copy().to_sparse(f)
            )
            native_args = unary_args(shape, tensor)
            runner.run(
                family,
                name,
                lambda m=module, a=native_args: m.evaluate(*a),
                backend="generated",
                rank=3,
                shape="x".join(map(str, shape)),
                input_nnz=int(torch.count_nonzero(dense)),
                output_format=fmt,
                extra={"density": density, "cells": int(np.prod(shape))},
            )
    for index, (name, shape, density) in enumerate(conversion3_cases):
        left = STensor.from_torch(
            make_dense(shape, density, 610 + index), "a"
        ).to_sparse("sss")
        right = STensor.from_torch(
            make_dense(shape, density, 620 + index), "b"
        ).to_sparse("sss")
        shape_key = "x".join(map(str, shape))
        csf_mul_module = module_for(
            "csf3_intersection",
            shape_key,
            lambda left_arg=left, r=right: scorch.einsum(
                "ijk,ijk->ijk", left_arg, r, format="sss"
            ),
        )
        native_args = binary_args(shape, left, right)
        runner.run(
            "csf3_intersection",
            name,
            lambda m=csf_mul_module, a=native_args: m.evaluate(*a),
            backend="generated",
            rank=3,
            shape="x".join(map(str, shape)),
            input_nnz=int(left.values.numel() + right.values.numel()),
            output_format="sss",
            extra={"density": density},
        )

    # ------------------------------------------------------------------
    # Reductions: row/column CSR plus rank-3 CSF -> dense matrix.
    # ------------------------------------------------------------------
    reduction_cases = [
        ("tiny_32x64_d4", 32, 64, 4, False),
        ("square_1024_d4", 1024, 1024, 4, False),
        ("square_1024_d128", 1024, 1024, 128, False),
        ("tall_16384x64_d4", 16384, 64, 4, False),
        ("wide_64x16384_d32", 64, 16384, 32, False),
        ("irregular_4096_d32", 4096, 4096, 32, True),
    ]
    for index, (name, rows, cols, degree, irregular) in enumerate(reduction_cases):
        tensor, _ = make_csr(rows, cols, degree, 710 + index, irregular=irregular)
        shape_key = f"{rows}x{cols}"
        red_row_module = module_for(
            "csr_reduce_rows",
            shape_key,
            lambda t=tensor: scorch.einsum("ij->i", t, format="d"),
        )
        red_col_module = module_for(
            "csr_reduce_cols",
            shape_key,
            lambda t=tensor: scorch.einsum("ij->j", t, format="d"),
        )
        for family, module, out_shape in (
            ("csr_reduce_rows", red_row_module, (rows,)),
            ("csr_reduce_cols", red_col_module, (cols,)),
        ):
            native_args = unary_args(out_shape, tensor)
            runner.run(
                family,
                name,
                lambda m=module, a=native_args: m.evaluate(*a),
                backend="generated",
                rank=2,
                shape=f"{rows}x{cols}",
                input_nnz=int(tensor.values.numel()),
                output_format="d",
                extra={"degree": degree, "irregular": irregular},
            )

    # ------------------------------------------------------------------
    # SpGEMM: default CSR two-pass, generic compiler linked workspace, and
    # native COO dynamic-output path. The historical family identifier is kept
    # stable so pre-RAII and post-RAII result files pair without a migration map.
    # ------------------------------------------------------------------
    spgemm_cases = [
        ("tiny_32_d4", 32, 32, 32, 4, 4, False),
        ("square_256_d4", 256, 256, 256, 4, 4, False),
        ("square_256_d16", 256, 256, 256, 16, 16, False),
        ("square_1024_d4", 1024, 1024, 1024, 4, 4, False),
        ("square_1024_d16", 1024, 1024, 1024, 16, 16, False),
        ("tall_4096x256x64", 4096, 256, 64, 8, 8, False),
        ("wide_64x1024x4096", 64, 1024, 4096, 8, 8, False),
        ("irregular_2048_d12", 2048, 2048, 2048, 12, 12, True),
    ]
    sg_a, _ = make_csr(32, 32, 4, 800)
    sg_b, _ = make_csr(32, 32, 4, 801)
    resolved = resolve_prebuilt_matmul(sg_a, sg_b)
    if resolved is None:
        raise RuntimeError("CSR SpGEMM prebuilt kernel unavailable")
    runner.add_kernel(
        "spgemm_csr_prebuilt",
        {
            "label": "spgemm_csr_prebuilt",
            "kernel": resolved.symbol_name,
            "compile_ms": 0.0,
            "ownership": "native prebuilt",
        },
    )
    for index, (name, rows, inner, cols, da, db, irregular) in enumerate(spgemm_cases):
        left, _ = make_csr(rows, inner, da, 810 + 2 * index, irregular=irregular)
        right, _ = make_csr(inner, cols, db, 811 + 2 * index, irregular=irregular)
        shape_key = f"{rows}x{inner}x{cols}"
        sg_generic_module = module_for(
            "spgemm_csr_generic",
            shape_key,
            lambda left_arg=left, r=right: scorch.matmul(left_arg, r, use_cache=False),
        )
        native_args = binary_args((rows, cols), left, right)
        common = {
            "rank": 2,
            "shape": f"{rows}x{inner}x{cols}",
            "input_nnz": int(left.values.numel() + right.values.numel()),
            "output_format": "ds",
            "extra": {"degree_a": da, "degree_b": db, "irregular": irregular},
        }
        runner.run(
            "spgemm_csr_prebuilt",
            name,
            lambda f=resolved.fn, a=native_args: f(*a),
            backend="prebuilt",
            **common,
        )
        runner.run(
            "spgemm_csr_generic",
            name,
            lambda m=sg_generic_module, a=native_args: m.evaluate(*a),
            backend="generated",
            **common,
        )

    coo_a, _ = make_csr(32, 32, 4, 900)
    coo_b, _ = make_csr(32, 32, 4, 901)
    coo_a, coo_b = to_coo(coo_a), to_coo(coo_b)
    coo_resolved = resolve_prebuilt_matmul(coo_a, coo_b)
    if coo_resolved is None:
        raise RuntimeError("COO SpGEMM prebuilt kernel unavailable")
    runner.add_kernel(
        "spgemm_coo_cvector",
        {
            "label": "spgemm_coo_cvector",
            "kernel": coo_resolved.symbol_name,
            "compile_ms": 0.0,
            "ownership": "native std::vector move output",
        },
    )
    for index, (size, degree) in enumerate(((16, 2), (32, 4), (128, 4), (256, 8))):
        left, _ = make_csr(size, size, degree, 910 + 2 * index)
        right, _ = make_csr(size, size, degree, 911 + 2 * index)
        left, right = to_coo(left), to_coo(right)
        native_args = binary_args((size, size), left, right)
        runner.run(
            "spgemm_coo_cvector",
            f"n{size}_d{degree}",
            lambda f=coo_resolved.fn, a=native_args: f(*a),
            backend="prebuilt",
            rank=2,
            shape=f"{size}x{size}x{size}",
            input_nnz=int(left.values.numel() + right.values.numel()),
            output_format="oo",
            extra={"degree": degree},
        )

    # ------------------------------------------------------------------
    # SDDMM: generic CSR mask emits coordinate builders; COO mask takes the
    # native fixed-nnz output path. Feature widths cover 4..256.
    # ------------------------------------------------------------------
    import scorch_ops as native_ops

    sddmm_prebuilt = native_ops.sddmm_coo_float_prebuilt
    runner.add_kernel(
        "sddmm_coo_prebuilt",
        {
            "label": "sddmm_coo_prebuilt",
            "kernel": "sddmm_coo_float_prebuilt",
            "compile_ms": 0.0,
            "ownership": "native fixed-nnz output",
        },
    )
    sddmm_cases = [
        ("tiny_32x64_d4_k4", 32, 64, 4, 4, False),
        ("square_512_d4_k16", 512, 512, 4, 16, False),
        ("square_512_d32_k64", 512, 512, 32, 64, False),
        ("tall_4096x256_d8_k32", 4096, 256, 8, 32, False),
        ("wide_256x4096_d16_k128", 256, 4096, 16, 128, False),
        ("irregular_2048_d16_k256", 2048, 2048, 16, 256, True),
    ]
    for index, (name, rows, cols, degree, width, irregular) in enumerate(sddmm_cases):
        mask, _ = make_csr(rows, cols, degree, 1010 + index, irregular=irregular)
        left = STensor.from_torch(torch.randn(rows, width), "a")
        right = STensor.from_torch(torch.randn(cols, width), "b")
        shape_key = f"{rows}x{cols}x{width}"
        sd_module = module_for(
            "sddmm_csr_generic_cvector",
            shape_key,
            lambda s=mask, left_arg=left, r=right: scorch.einsum(
                "ij,ik,jk->ij", s, left_arg, r, format="oo"
            ),
        )
        generic_args = ternary_args((rows, cols), mask, left, right)
        runner.run(
            "sddmm_csr_generic_cvector",
            name,
            lambda m=sd_module, a=generic_args: m.evaluate(*a),
            backend="generated",
            rank=2,
            shape=f"{rows}x{cols}x{width}",
            input_nnz=int(mask.values.numel()),
            output_format="oo",
            extra={"degree": degree, "feature_width": width, "irregular": irregular},
        )
        coo_mask = to_coo(mask)
        prebuilt_args = ternary_args((rows, cols), coo_mask, left, right)
        runner.run(
            "sddmm_coo_prebuilt",
            name,
            lambda f=sddmm_prebuilt, a=prebuilt_args: f(*a),
            backend="prebuilt",
            rank=2,
            shape=f"{rows}x{cols}x{width}",
            input_nnz=int(coo_mask.values.numel()),
            output_format="oo",
            extra={"degree": degree, "feature_width": width, "irregular": irregular},
        )

    # ------------------------------------------------------------------
    # CSR x dense: output buffer ownership across tiny/large rows and widths 1..1024.
    # ------------------------------------------------------------------
    spmm_cases = [
        ("tiny_32_d4_w1", 32, 32, 4, 1, False),
        ("small_256_d8_w4", 256, 256, 8, 4, False),
        ("square_1024_d4_w16", 1024, 1024, 4, 16, False),
        ("square_1024_d32_w64", 1024, 1024, 32, 64, False),
        ("tall_16384x256_d8_w16", 16384, 256, 8, 16, False),
        ("wide_features_512_d16_w256", 512, 512, 16, 256, False),
        ("very_wide_128_d32_w1024", 128, 1024, 32, 1024, False),
        ("irregular_4096_d16_w128", 4096, 4096, 16, 128, True),
    ]
    sm_a, _ = make_csr(32, 32, 4, 1100)
    sm_b = STensor.from_torch(torch.randn(32, 1), "b")
    sm_resolved = resolve_prebuilt_matmul(sm_a, sm_b, "dd")
    if sm_resolved is None:
        raise RuntimeError("CSR SpMM prebuilt unavailable")
    runner.add_kernel(
        "spmm_csr_prebuilt",
        {
            "label": "spmm_csr_prebuilt",
            "kernel": sm_resolved.symbol_name,
            "compile_ms": 0.0,
            "ownership": "native dense output",
        },
    )
    for index, (name, rows, inner, degree, width, irregular) in enumerate(spmm_cases):
        left, _ = make_csr(rows, inner, degree, 1110 + index, irregular=irregular)
        right = STensor.from_torch(torch.randn(inner, width), "b")
        native_args = binary_args((rows, width), left, right)
        runner.run(
            "spmm_csr_prebuilt",
            name,
            lambda f=sm_resolved.fn, a=native_args: f(*a),
            backend="prebuilt",
            rank=2,
            shape=f"{rows}x{inner}x{width}",
            input_nnz=int(left.values.numel()),
            output_format="dd",
            extra={"degree": degree, "feature_width": width, "irregular": irregular},
        )

    runner.save()
    print(f"\nWROTE {runner.csv_path}\nWROTE {runner.json_path}", flush=True)


if __name__ == "__main__":
    main()
