#!/usr/bin/env python3
"""Benchmark generated tile-ijk storage schedules against the native kernel.

This benchmark compares, at identical ``Nc`` and ``Jc``:

* an untiled compiler-generated CSR x dense kernel;
* all four crosses of packed-operand scope (panel/full reduction axis) and
  result storage (direct global output/compact heap tile); and
* ``scorch_ops.spmm_csr_float_tileijk``, the handwritten full-pack/compact-C
  implementation.

JIT compilation happens before correctness checks, warmup, or measurement. Each
candidate is checked against ``torch.sparse.mm``. Timed rounds visit every
candidate once in a newly shuffled order and report the median, so allocation,
packing, compact-tile initialization, and copy-out costs remain inside the timed
kernel call while compilation and Python scheduling do not.

Examples
--------
Run the controlled panel-count sweep from the storage-schedule investigation::

    python bench/bench_codegen_tileijk_storage.py --suite panels

Sweep ``Nc`` with one full-J panel::

    python bench/bench_codegen_tileijk_storage.py --suite nc \
        --nc-values 64,128,256,448

Run policy-derived named cases, or select a subset::

    python bench/bench_codegen_tileijk_storage.py --suite policy
    python bench/bench_codegen_tileijk_storage.py \
        --case llc-edge --case ragged-tail

The default ``quick`` suite uses a small ragged case suitable for a smoke test.
Use ``--csv PATH`` to retain machine-readable results.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import math
import random
import statistics
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_NAMED_CASE_DESCRIPTIONS = {
    "tiny-fit": "small B/output working sets that fit comfortably in cache",
    "llc-edge": "B is just larger than the queried last-level cache",
    "wide-n": "very wide dense free axis with several k strips",
    "high-degree": "high-reuse scattered rows",
    "low-degree": "low-reuse scattered rows",
    "tall": "many output rows and a moderate free axis",
    "short-wide": "few output rows with a large reduction and free axis",
    "ragged-tail": "ragged i/j/k sizes and empty rows",
}


@dataclass(frozen=True)
class Workload:
    """One synthetic CSR x dense problem, independent of schedule widths."""

    name: str
    m: int
    j: int
    n: int
    degree: int
    empty_every: int = 0

    @property
    def key(self) -> Tuple[int, int, int, int, int]:
        return (self.m, self.j, self.n, self.degree, self.empty_every)


@dataclass(frozen=True)
class RunConfig:
    """One workload plus the widths shared by every tiled candidate."""

    label: str
    workload: Workload
    nc: int
    jc: int


@dataclass
class Variant:
    """A compiled callable and its storage-strategy labels."""

    name: str
    stage_scope: str
    result_storage: str
    fn: Callable[[], Any]
    max_abs_error: float = 0.0
    relative_l2_error: float = 0.0


def _positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}")
    return value


def _parse_int_csv(value: str, name: str) -> List[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be comma-separated integers"
        ) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return parsed


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark untiled and four packed tile-ijk generated schedules "
            "against the handwritten full-pack/compact-C kernel."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("quick", "policy", "panels", "nc", "all"),
        default="quick",
        help=(
            "case group to run (default: quick); explicit --case values replace "
            "this selection"
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(_NAMED_CASE_DESCRIPTIONS),
        help="run one named policy case; repeat to select several",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="print named policy cases and exit",
    )
    parser.add_argument(
        "--panel-counts",
        default="1,2,4,8,16",
        help="requested P values for --suite panels (default: 1,2,4,8,16)",
    )
    parser.add_argument(
        "--nc-values",
        default="64,128,256,448",
        help="Nc values for --suite nc (default: 64,128,256,448)",
    )
    parser.add_argument("--m", type=int, default=1024, help="controlled-sweep rows")
    parser.add_argument(
        "--j", type=int, default=8192, help="controlled-sweep reduction extent"
    )
    parser.add_argument(
        "--n", type=int, default=513, help="controlled-sweep free-axis extent"
    )
    parser.add_argument(
        "--degree", type=int, default=64, help="controlled-sweep entries per row"
    )
    parser.add_argument(
        "--panel-nc",
        type=int,
        default=256,
        help="fixed Nc for --suite panels (default: 256)",
    )
    parser.add_argument(
        "--nc-jc",
        type=int,
        default=0,
        help="fixed Jc for --suite nc; 0 means one full-J panel (default)",
    )
    parser.add_argument(
        "--warmup", type=int, default=2, help="unmeasured calls per candidate"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=9,
        help="randomized timed rounds per candidate (default: 9)",
    )
    parser.add_argument("--seed", type=int, default=0, help="input/timing RNG seed")
    parser.add_argument(
        "--native-threads",
        type=int,
        default=-1,
        help=(
            "native nthreads_override (default: -1, matching generated "
            "scorch_nthreads policy)"
        ),
    )
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument(
        "--csv", type=Path, help="optional path for machine-readable result rows"
    )
    return parser


def _named_workloads(llc_bytes: int) -> Dict[str, Workload]:
    # Keep J fixed for the LLC-edge case and choose a ragged N one element beyond
    # the fit boundary. This makes the meaning portable across machines.
    edge_j = 8192
    edge_n = max(65, llc_bytes // (4 * edge_j) + 1)
    wide_n = max(2049, 2 * (llc_bytes // (4 * 4096)) + 1)
    return {
        "tiny-fit": Workload("tiny-fit", 128, 512, 65, 16),
        "llc-edge": Workload("llc-edge", 1024, edge_j, edge_n, 64),
        "wide-n": Workload("wide-n", 1024, 4096, wide_n, 64),
        "high-degree": Workload("high-degree", 2048, 4096, 513, 256),
        "low-degree": Workload("low-degree", 2048, 4096, 513, 8),
        "tall": Workload("tall", 8192, 2048, 257, 32),
        "short-wide": Workload("short-wide", 128, 8192, 2049, 64),
        "ragged-tail": Workload("ragged-tail", 513, 2053, 515, 37, 17),
    }


def _policy_config(workload: Workload, tiling: Any, llc_bytes: int) -> RunConfig:
    nc, jc = tiling._ijk_params(workload.n, workload.m, workload.j, llc_bytes)
    return RunConfig(f"policy:{workload.name}", workload, nc, jc)


def _controlled_workload(args: argparse.Namespace) -> Workload:
    for name in ("m", "j", "n", "degree"):
        _positive_int(f"--{name}", getattr(args, name))
    if args.degree > args.j:
        raise ValueError(f"--degree ({args.degree}) cannot exceed --j ({args.j})")
    return Workload("controlled", args.m, args.j, args.n, args.degree)


def _build_configs(
    args: argparse.Namespace, tiling: Any, llc_bytes: int
) -> List[RunConfig]:
    named = _named_workloads(llc_bytes)
    if args.case:
        return [_policy_config(named[name], tiling, llc_bytes) for name in args.case]

    configs: List[RunConfig] = []
    if args.suite == "quick":
        smoke = Workload("quick-ragged", 31, 67, 35, 9, 7)
        return [RunConfig("quick-ragged", smoke, nc=16, jc=17)]

    if args.suite in ("policy", "all"):
        configs.extend(
            _policy_config(workload, tiling, llc_bytes) for workload in named.values()
        )

    if args.suite in ("panels", "nc", "all"):
        controlled = _controlled_workload(args)
    if args.suite in ("panels", "all"):
        _positive_int("--panel-nc", args.panel_nc)
        if args.panel_nc > controlled.n:
            raise ValueError(
                f"--panel-nc ({args.panel_nc}) cannot exceed --n ({controlled.n})"
            )
        panel_counts = _parse_int_csv(args.panel_counts, "--panel-counts")
        for requested_p in panel_counts:
            jc = max(1, math.ceil(controlled.j / requested_p))
            actual_p = math.ceil(controlled.j / jc)
            configs.append(
                RunConfig(
                    f"panels:P{requested_p}-actual{actual_p}",
                    controlled,
                    nc=args.panel_nc,
                    jc=jc,
                )
            )

    if args.suite in ("nc", "all"):
        nc_values = _parse_int_csv(args.nc_values, "--nc-values")
        invalid = [nc for nc in nc_values if nc > controlled.n]
        if invalid:
            raise ValueError(f"Nc values cannot exceed --n ({controlled.n}): {invalid}")
        jc = controlled.j if args.nc_jc == 0 else args.nc_jc
        _positive_int("--nc-jc", jc)
        if jc > controlled.j:
            raise ValueError(f"--nc-jc ({jc}) cannot exceed --j ({controlled.j})")
        configs.extend(
            RunConfig(f"nc:Nc{nc}", controlled, nc=nc, jc=jc) for nc in nc_values
        )

    return configs


def _make_inputs(workload: Workload, seed: int, np: Any, torch: Any) -> tuple:
    """Build sorted int32 CSR, dense RHS, Scorch tensors, and PyTorch reference."""
    if workload.degree > workload.j:
        raise ValueError(
            f"case {workload.name}: degree {workload.degree} exceeds J={workload.j}"
        )
    rng = np.random.default_rng(seed)
    counts = np.full(workload.m, workload.degree, dtype=np.int64)
    if workload.empty_every:
        counts[:: workload.empty_every] = 0
    crow64 = np.empty(workload.m + 1, dtype=np.int64)
    crow64[0] = 0
    np.cumsum(counts, out=crow64[1:])
    nnz = int(crow64[-1])
    columns = np.empty(nnz, dtype=np.int32)
    for row in range(workload.m):
        begin = int(crow64[row])
        end = int(crow64[row + 1])
        if begin == end:
            continue
        # Unique coordinates make the requested degree exact. Sorting is required
        # by sparse-panel lower_bound windowing in generated and native kernels.
        columns[begin:end] = np.sort(
            rng.choice(workload.j, size=end - begin, replace=False)
        ).astype(np.int32)
    values = rng.standard_normal(nnz).astype(np.float32)
    crow = torch.from_numpy(crow64.astype(np.int32))
    col = torch.from_numpy(columns)
    val = torch.from_numpy(values)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta state"
        )
        csr = torch.sparse_csr_tensor(
            crow,
            col,
            val,
            size=(workload.m, workload.j),
            check_invariants=True,
        )
    rhs_generator = torch.Generator().manual_seed(seed + 1)
    rhs = torch.randn(
        workload.j, workload.n, generator=rhs_generator, dtype=torch.float32
    )

    import scorch

    a_st = scorch.STensor.from_csr(csr, "A")
    b_st = scorch.STensor.from_torch(rhs, "B")
    reference = torch.sparse.mm(csr.to(torch.float64), rhs.to(torch.float64))
    return csr, rhs, a_st, b_st, reference, nnz


def _untiled_schedule(scorch: Any) -> Any:
    return scorch.Schedule(
        loop_order=("i", "j", "k"),
        tag="bench-generated-untiled",
        parallel_loop="i",
    )


def _packed_schedule(
    scorch: Any, nc: int, jc: int, stage_scope: str, result_storage: str
) -> Any:
    if stage_scope not in ("panel", "full"):
        raise ValueError(f"unknown stage scope {stage_scope!r}")
    if result_storage not in ("direct", "compact"):
        raise ValueError(f"unknown result storage {result_storage!r}")
    return scorch.Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            scorch.TileSpec(
                "k",
                nc,
                placement="outermost",
                accum="direct" if result_storage == "direct" else "heap",
                unroll=False,
            ),
            scorch.TileSpec(
                "j",
                jc,
                placement="child_of:k_out",
                kind="panel",
                accum="direct",
            ),
        ),
        relayout=scorch.RelayoutSpec(
            operand="B",
            pack_var="k",
            strip_width=nc,
            scope_var="j" if stage_scope == "panel" else "k",
        ),
        tag=f"bench-packed-{stage_scope}-{result_storage}",
        parallel_loop="i",
    )


def _extract_values(result: Any, m: int, n: int) -> Any:
    storage = getattr(result, "storage", None)
    if storage is not None:
        return storage.value.reshape(m, n)
    private_storage = getattr(result, "_storage", None)
    if private_storage is not None:
        return private_storage._value.reshape(m, n)
    raise TypeError(
        f"cannot extract generated/native values from {type(result).__name__}"
    )


def _compile_generated(
    scorch: Any,
    python_ops: Any,
    a_st: Any,
    b_st: Any,
    workload: Workload,
    schedule: Any,
) -> Callable[[], Any]:
    """Compile once, then return a direct module.evaluate callable."""
    expression = "ij,jk->ik"
    output_format = "dd"
    scorch.einsum(
        expression,
        a_st,
        b_st,
        format=output_format,
        schedule=schedule,
    )
    dispatch_key = python_ops._einsum_cache_key(
        expression,
        (a_st, b_st),
        output_format,
        None,
        schedule,
    )
    cached = python_ops._einsum_dispatch_cache.get(dispatch_key)
    if cached is None:
        raise RuntimeError(
            "generated kernel was compiled but is absent from the einsum dispatch cache"
        )
    module = cached[0]
    args: Sequence[Any] = (
        (workload.m, workload.n),
        a_st.shape,
        a_st.index.mode_indices,
        a_st.values,
        b_st.shape,
        b_st.index.mode_indices,
        b_st.values,
    )

    def evaluate() -> Any:
        return _extract_values(module.evaluate(*args), workload.m, workload.n)

    return evaluate


def _native_variant(
    native_ops: Any,
    a_st: Any,
    b_st: Any,
    workload: Workload,
    nc: int,
    jc: int,
    nthreads: int,
) -> Variant:
    args = (
        [workload.m, workload.n],
        list(a_st.shape),
        a_st.index.mode_indices,
        a_st.values,
        list(b_st.shape),
        b_st.index.mode_indices,
        b_st.values,
        nc,
        jc,
        nthreads,
    )

    def evaluate() -> Any:
        return _extract_values(
            native_ops.spmm_csr_float_tileijk(*args),
            workload.m,
            workload.n,
        )

    return Variant("native_tileijk", "full", "compact", evaluate)


def _build_variants(
    scorch: Any,
    python_ops: Any,
    native_ops: Any,
    a_st: Any,
    b_st: Any,
    config: RunConfig,
    native_threads: int,
) -> List[Variant]:
    workload = config.workload
    variants: List[Variant] = []
    schedules = [("generated_untiled", "none", "direct", _untiled_schedule(scorch))]
    for stage_scope in ("panel", "full"):
        for result_storage in ("direct", "compact"):
            schedules.append(
                (
                    f"generated_{stage_scope}_{result_storage}",
                    stage_scope,
                    result_storage,
                    _packed_schedule(
                        scorch,
                        config.nc,
                        config.jc,
                        stage_scope,
                        result_storage,
                    ),
                )
            )

    for name, stage_scope, result_storage, schedule in schedules:
        print(f"    precompile {name} ...", flush=True)
        fn = _compile_generated(scorch, python_ops, a_st, b_st, workload, schedule)
        variants.append(Variant(name, stage_scope, result_storage, fn))
    variants.append(
        _native_variant(
            native_ops,
            a_st,
            b_st,
            workload,
            config.nc,
            config.jc,
            native_threads,
        )
    )
    return variants


def _verify_variants(
    variants: Iterable[Variant], reference: Any, atol: float, rtol: float, torch: Any
) -> None:
    reference_norm = float(torch.linalg.vector_norm(reference)) + 1e-30
    for variant in variants:
        actual = variant.fn().to(torch.float64)
        difference = actual - reference
        variant.max_abs_error = float(difference.abs().max()) if actual.numel() else 0.0
        variant.relative_l2_error = (
            float(torch.linalg.vector_norm(difference)) / reference_norm
        )
        if not torch.allclose(actual, reference, atol=atol, rtol=rtol):
            raise RuntimeError(
                f"{variant.name} failed correctness: max_abs="
                f"{variant.max_abs_error:.3e}, rel_l2="
                f"{variant.relative_l2_error:.3e}"
            )


def _measure(
    variants: Sequence[Variant], warmup: int, repetitions: int, seed: int
) -> Dict[str, List[float]]:
    if warmup < 0:
        raise ValueError("--warmup cannot be negative")
    _positive_int("--repetitions", repetitions)
    rng = random.Random(seed)
    order = list(variants)
    for _ in range(warmup):
        rng.shuffle(order)
        for variant in order:
            variant.fn()

    gc.collect()
    samples = {variant.name: [] for variant in variants}
    for _ in range(repetitions):
        rng.shuffle(order)
        for variant in order:
            start = time.perf_counter_ns()
            result = variant.fn()
            elapsed = (time.perf_counter_ns() - start) * 1e-9
            samples[variant.name].append(elapsed)
            # The CPU extension call is synchronous. Retain no output between
            # candidates, so output allocation/deallocation remains per invocation.
            del result
    return samples


def _result_rows(
    config: RunConfig,
    variants: Sequence[Variant],
    samples: Dict[str, List[float]],
    nnz: int,
    warmup: int,
    repetitions: int,
) -> List[dict]:
    medians = {name: statistics.median(times) for name, times in samples.items()}
    native_time = medians["native_tileijk"]
    workload = config.workload
    j_panels = math.ceil(workload.j / config.jc)
    k_strips = math.ceil(workload.n / config.nc)
    flop = 2.0 * nnz * workload.n
    rows = []
    for variant in variants:
        median_s = medians[variant.name]
        rows.append(
            {
                "case": config.label,
                "M": workload.m,
                "J": workload.j,
                "N": workload.n,
                "nnz": nnz,
                "degree": workload.degree,
                "Nc": config.nc,
                "Jc": config.jc,
                "j_panels": j_panels,
                "k_strips": k_strips,
                "variant": variant.name,
                "stage_scope": variant.stage_scope,
                "result_storage": variant.result_storage,
                "median_ms": median_s * 1e3,
                "gflops": flop / median_s / 1e9,
                "speedup_vs_native": native_time / median_s,
                "max_abs_error": variant.max_abs_error,
                "relative_l2_error": variant.relative_l2_error,
                "warmup": warmup,
                "repetitions": repetitions,
            }
        )
    return rows


def _print_rows(rows: Sequence[dict]) -> None:
    print(
        "    "
        f"{'variant':29s} {'stage':>7s} {'result':>8s} "
        f"{'median ms':>10s} {'GF/s':>9s} {'vs native':>10s} {'relerr':>10s}"
    )
    for row in rows:
        print(
            "    "
            f"{row['variant']:29s} {row['stage_scope']:>7s} "
            f"{row['result_storage']:>8s} {row['median_ms']:10.3f} "
            f"{row['gflops']:9.1f} {row['speedup_vs_native']:10.3f}x "
            f"{row['relative_l2_error']:10.2e}"
        )


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_case_list() -> None:
    for name, description in _NAMED_CASE_DESCRIPTIONS.items():
        print(f"{name:12s} {description}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.list_cases:
        _print_case_list()
        return 0

    # Heavy imports occur after argument handling so --help/--list-cases remain
    # useful even outside a configured Scorch environment.
    import numpy as np
    import torch
    import scorch
    import scorch_ops as native_ops

    python_ops = importlib.import_module("scorch.ops")
    tiling = importlib.import_module("scorch.tiling")
    llc_bytes = tiling.query_llc()
    configs = _build_configs(args, tiling, llc_bytes)
    if not configs:
        parser.error("selected suite produced no benchmark configurations")

    print(
        f"[config] LLC={llc_bytes / (1 << 20):.1f} MiB, "
        f"torch_threads={torch.get_num_threads()}, warmup={args.warmup}, "
        f"repetitions={args.repetitions}, seed={args.seed}"
    )
    print(
        "[timing] direct cached evaluate calls; JIT excluded; allocation/pack/"
        "zero/copy included; randomized candidate order"
    )

    input_cache: Dict[Tuple[int, int, int, int, int], tuple] = {}
    all_rows: List[dict] = []
    for config_index, config in enumerate(configs):
        workload = config.workload
        if workload.key not in input_cache:
            print(
                f"\n[input] {workload.name}: M={workload.m}, J={workload.j}, "
                f"N={workload.n}, degree={workload.degree}, "
                f"empty_every={workload.empty_every or 'none'}",
                flush=True,
            )
            input_cache[workload.key] = _make_inputs(workload, args.seed, np, torch)
        _, _, a_st, b_st, reference, nnz = input_cache[workload.key]
        j_panels = math.ceil(workload.j / config.jc)
        k_strips = math.ceil(workload.n / config.nc)
        print(
            f"\n[case] {config.label}: Nc={config.nc}, Jc={config.jc}, "
            f"P={j_panels}, k_strips={k_strips}, nnz={nnz}",
            flush=True,
        )
        variants = _build_variants(
            scorch,
            python_ops,
            native_ops,
            a_st,
            b_st,
            config,
            args.native_threads,
        )
        _verify_variants(variants, reference, args.atol, args.rtol, torch)
        print("    correctness: all variants passed", flush=True)
        samples = _measure(
            variants,
            args.warmup,
            args.repetitions,
            args.seed + config_index,
        )
        rows = _result_rows(
            config,
            variants,
            samples,
            nnz,
            args.warmup,
            args.repetitions,
        )
        _print_rows(rows)
        all_rows.extend(rows)

    if args.csv is not None:
        _write_csv(args.csv, all_rows)
        print(f"\n[csv] wrote {args.csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
