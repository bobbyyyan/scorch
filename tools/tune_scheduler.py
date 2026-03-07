#!/usr/bin/env python3
"""
Tune scheduler cost-model constants against measured kernel runtimes.

This harness intentionally generates workloads with different tensor mode orders
and expression templates so loop-order choices are not always fixed by
constraints.

Example:
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  conda run -n scorch python tools/tune_scheduler.py \
    --method random --trials 40 --workloads 8 --benchmark-repeats 3 \
    --output-json /tmp/scheduler_tuning.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scorch import STensor
import scorch
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.scheduler import Scheduler, _CostModelConstants
from scorch.utils import PROJECT_ROOT_DIR, get_extra_cflags, get_extra_ldflags


LOOP_VARS: Tuple[str, str, str] = ("i", "j", "k")
LOOP_PERMUTATIONS: Tuple[Tuple[str, str, str], ...] = tuple(
    itertools.permutations(LOOP_VARS)
)

SUPPORTED_TEMPLATES: Tuple[str, ...] = (
    "spmm",
    "spgemm",
    "spmm_transposed_rhs",
    "broadcast_rhs_vec",
    "broadcast_lhs_vec",
)
DEFAULT_TEMPLATES: Tuple[str, ...] = (
    "spmm",
    "spgemm",
    "broadcast_rhs_vec",
    "broadcast_lhs_vec",
)
MODE_ORDERS_2D: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0))


@dataclass(frozen=True)
class InputTensorSpec:
    name: str
    rank: int
    fmt: str
    zero_frac: float
    mode_order: Tuple[int, ...]


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    template: str
    n: int
    fmt_out: str
    output_mode_order: Tuple[int, int]
    inputs: Tuple[InputTensorSpec, ...]


@dataclass
class WorkloadRuntimeTable:
    spec: WorkloadSpec
    base_cin: ForAll
    candidate_runtimes: Dict[Tuple[str, str, str], float]
    oracle_perm: Tuple[str, str, str]
    oracle_runtime: float
    penalty_runtime: float


@dataclass
class TuneResult:
    best_params: _CostModelConstants
    best_score: float
    method: str
    trials: int
    per_workload_ratio: Dict[str, float]


def _is_dense_format(rank: int, fmt: str) -> bool:
    return (rank == 1 and fmt == "d") or (rank == 2 and fmt == "dd")


def _normalize_templates(raw_templates: Sequence[str]) -> Tuple[str, ...]:
    tokens: List[str] = []
    for raw in raw_templates:
        for token in raw.split(","):
            stripped = token.strip()
            if stripped:
                tokens.append(stripped)

    if not tokens:
        return DEFAULT_TEMPLATES

    if "all" in tokens:
        return SUPPORTED_TEMPLATES

    invalid = [token for token in tokens if token not in SUPPORTED_TEMPLATES]
    if invalid:
        raise ValueError(
            f"Unknown templates: {invalid}. Supported templates: {list(SUPPORTED_TEMPLATES)}"
        )

    # Preserve order while dropping duplicates.
    return tuple(dict.fromkeys(tokens))


def _make_dense_tensor(
    n: int,
    rank: int,
    zero_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if rank == 1:
        tensor = torch.rand((n,), generator=generator, dtype=torch.float32)
    elif rank == 2:
        tensor = torch.rand((n, n), generator=generator, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported rank: {rank}")

    if zero_fraction <= 0.0:
        return tensor

    mask = torch.rand(tensor.shape, generator=generator, dtype=torch.float32) > zero_fraction
    return tensor * mask


def _to_stensor(tensor: torch.Tensor, spec: InputTensorSpec) -> STensor:
    mode_order = list(spec.mode_order) if spec.rank == 2 else [0]
    stensor = STensor.from_torch(tensor, name=spec.name, mode_order=mode_order)
    if _is_dense_format(spec.rank, spec.fmt):
        return stensor
    return stensor.to_sparse(spec.fmt)


def _build_workload_cin(spec: WorkloadSpec, loop_order: Sequence[str]) -> ForAll:
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    ivars = {"i": i, "j": j, "k": k}

    C = TensorVar("C", fmt=spec.fmt_out, mode_order=list(spec.output_mode_order))
    tensor_vars = {
        inp.name: TensorVar(inp.name, fmt=inp.fmt, mode_order=list(inp.mode_order))
        for inp in spec.inputs
    }

    if spec.template in {"spmm", "spgemm"}:
        # C[i,k] += A[i,j] * B[j,k]
        rhs = tensor_vars["A"][i, j] * tensor_vars["B"][j, k]
    elif spec.template == "spmm_transposed_rhs":
        # C[i,k] += A[i,j] * B[k,j]
        rhs = tensor_vars["A"][i, j] * tensor_vars["B"][k, j]
    elif spec.template == "broadcast_rhs_vec":
        # C[i,k] += A[i,j] * b[j]
        rhs = tensor_vars["A"][i, j] * tensor_vars["b"][j]
    elif spec.template == "broadcast_lhs_vec":
        # C[i,k] += a[i] * B[j,k]
        rhs = tensor_vars["a"][i] * tensor_vars["B"][j, k]
    else:
        raise ValueError(f"Unknown template: {spec.template}")

    stmt = TensorAssign(C[i, k], rhs, op=Operation.ADD)
    for loop_var in reversed(loop_order):
        stmt = ForAll(ivars[loop_var], stmt)
    assert isinstance(stmt, ForAll)
    return stmt


def _compile_module_from_cin(
    cin_stmt: ForAll,
    module_cache: Dict[str, object],
    build_dir: Path,
) -> object:
    cache_key = str(cin_stmt)
    if cache_key in module_cache:
        return module_cache[cache_key]

    lowerer = CINLowerer()
    lowered_llir = lowerer.lower_IndexStmt(cin_stmt)
    llir_lowerer = LLIRLowerer()
    cpp_code = llir_lowerer.lower_llir(lowered_llir)

    with open(PROJECT_ROOT_DIR / "csrc/header.cpp", "r") as f:
        header_cpp_code = f.read()

    module_hash = hashlib.sha1((header_cpp_code + cpp_code).encode()).hexdigest()[:16]
    module_name = f"scheduler_tune_{module_hash}"

    module = torch.utils.cpp_extension.load_inline(
        name=module_name,
        cpp_sources=[header_cpp_code, cpp_code],
        functions=["evaluate"],
        extra_cflags=get_extra_cflags(),
        extra_ldflags=get_extra_ldflags(),
        build_directory=str(build_dir),
    )
    module_cache[cache_key] = module
    return module


def _build_module_args(result_shape: Tuple[int, int], inputs: Sequence[STensor]) -> List[object]:
    args: List[object] = [result_shape]
    for tensor in inputs:
        args.append(tensor.shape)
        args.append(tensor.index.mode_indices)
        args.append(tensor.values)
    return args


def _benchmark_cin_eval_time(
    cin_stmt: ForAll,
    result_shape: Tuple[int, int],
    inputs: Sequence[STensor],
    module_cache: Dict[str, object],
    build_dir: Path,
    repeats: int,
    warmup: int,
) -> float:
    try:
        module = _compile_module_from_cin(
            cin_stmt=cin_stmt,
            module_cache=module_cache,
            build_dir=build_dir,
        )
    except Exception:
        return float("inf")

    module_args = _build_module_args(result_shape, inputs)

    try:
        for _ in range(max(warmup, 0)):
            module.evaluate(*module_args)
    except Exception:
        return float("inf")

    times: List[float] = []
    for _ in range(max(repeats, 1)):
        start = time.perf_counter()
        try:
            module.evaluate(*module_args)
        except Exception:
            return float("inf")
        times.append(time.perf_counter() - start)

    return statistics.median(times)


@contextlib.contextmanager
def _force_select_loop_order(loop_order_names: Sequence[str]):
    original_select_loop_order = Scheduler.select_loop_order

    def _forced_select_loop_order(
        cin: ForAll,
        costs: _CostModelConstants = Scheduler._DEFAULT_COSTS,
    ):
        all_index_vars = Scheduler.get_index_variables(cin)
        index_var_by_name = {index_var.name: index_var for index_var in all_index_vars}
        selected: List[IndexVar] = []
        for name in loop_order_names:
            index_var = index_var_by_name.get(name)
            if index_var is not None and index_var not in selected:
                selected.append(index_var)
        for index_var in all_index_vars:
            if index_var not in selected:
                selected.append(index_var)
        return selected

    Scheduler.select_loop_order = staticmethod(_forced_select_loop_order)
    try:
        yield
    finally:
        Scheduler.select_loop_order = original_select_loop_order


def _benchmark_forced_loop_matmul_eval_time(
    spec: WorkloadSpec,
    inputs: Sequence[STensor],
    loop_order: Sequence[str],
    repeats: int,
    warmup: int,
) -> float:
    if len(inputs) != 2:
        return float("inf")

    a, b = inputs
    matmul_kwargs = {
        "format": spec.fmt_out,
        "output_mode_order": list(spec.output_mode_order),
        "use_cache": False,
    }

    try:
        for _ in range(max(warmup, 0)):
            with _force_select_loop_order(loop_order):
                scorch.matmul(a, b, **matmul_kwargs)
    except Exception:
        return float("inf")

    times: List[float] = []
    for _ in range(max(repeats, 1)):
        time_dict: Dict[str, float] = {}
        try:
            with _force_select_loop_order(loop_order):
                scorch.matmul(a, b, time_dict=time_dict, **matmul_kwargs)
        except Exception:
            return float("inf")
        eval_time = time_dict.get("eval_time", float("inf"))
        if not math.isfinite(eval_time):
            return float("inf")
        times.append(eval_time)

    return statistics.median(times)


def _generate_workload_specs(
    count: int,
    min_n: int,
    max_n: int,
    seed: int,
    templates: Sequence[str],
) -> List[WorkloadSpec]:
    rng = random.Random(seed)
    specs: List[WorkloadSpec] = []

    def sparse_zero_frac(fmt: str, rank: int) -> float:
        return 0.0 if _is_dense_format(rank, fmt) else rng.uniform(0.75, 0.995)

    template_pool = list(templates)
    rng.shuffle(template_pool)

    for idx in range(count):
        if idx > 0 and idx % len(template_pool) == 0:
            rng.shuffle(template_pool)

        template = template_pool[idx % len(template_pool)]
        n = rng.randint(min_n, max_n)
        output_mode_order = rng.choice(MODE_ORDERS_2D)

        if template == "spmm":
            fmt_out = "dd"
            a_fmt = rng.choice(["ds", "oo"])
            b_fmt = "dd"
            inputs = (
                InputTensorSpec(
                    "A", 2, a_fmt, sparse_zero_frac(a_fmt, 2), rng.choice(MODE_ORDERS_2D)
                ),
                InputTensorSpec(
                    "B", 2, b_fmt, sparse_zero_frac(b_fmt, 2), rng.choice(MODE_ORDERS_2D)
                ),
            )
        elif template == "spgemm":
            fmt_out = "dd"
            a_fmt = rng.choice(["ds", "oo"])
            b_fmt = rng.choice(["ds", "oo"])
            inputs = (
                InputTensorSpec(
                    "A", 2, a_fmt, sparse_zero_frac(a_fmt, 2), rng.choice(MODE_ORDERS_2D)
                ),
                InputTensorSpec(
                    "B", 2, b_fmt, sparse_zero_frac(b_fmt, 2), rng.choice(MODE_ORDERS_2D)
                ),
            )
        elif template == "spmm_transposed_rhs":
            fmt_out = "dd"
            a_fmt = rng.choice(["ds", "oo"])
            b_fmt = rng.choice(["dd", "ds", "oo"])
            inputs = (
                InputTensorSpec(
                    "A", 2, a_fmt, sparse_zero_frac(a_fmt, 2), rng.choice(MODE_ORDERS_2D)
                ),
                InputTensorSpec(
                    "B", 2, b_fmt, sparse_zero_frac(b_fmt, 2), rng.choice(MODE_ORDERS_2D)
                ),
            )
        elif template == "broadcast_rhs_vec":
            fmt_out = "dd"
            a_fmt = rng.choice(["dd", "ds", "oo"])
            b_fmt = rng.choice(["d", "s"])
            inputs = (
                InputTensorSpec(
                    "A", 2, a_fmt, sparse_zero_frac(a_fmt, 2), rng.choice(MODE_ORDERS_2D)
                ),
                InputTensorSpec("b", 1, b_fmt, sparse_zero_frac(b_fmt, 1), (0,)),
            )
        elif template == "broadcast_lhs_vec":
            fmt_out = "dd"
            a_fmt = rng.choice(["d", "s"])
            b_fmt = rng.choice(["dd", "ds", "oo"])
            inputs = (
                InputTensorSpec("a", 1, a_fmt, sparse_zero_frac(a_fmt, 1), (0,)),
                InputTensorSpec(
                    "B", 2, b_fmt, sparse_zero_frac(b_fmt, 2), rng.choice(MODE_ORDERS_2D)
                ),
            )
        else:
            raise ValueError(f"Unknown template: {template}")

        input_sig = "_".join(
            f"{inp.name}{inp.fmt}_mo{''.join(str(v) for v in inp.mode_order)}"
            for inp in inputs
        )
        name = (
            f"{template}_n{n}_{idx}"
            f"_out{fmt_out}_mo{output_mode_order[0]}{output_mode_order[1]}"
            f"_{input_sig}"
        )

        specs.append(
            WorkloadSpec(
                name=name,
                template=template,
                n=n,
                fmt_out=fmt_out,
                output_mode_order=output_mode_order,
                inputs=inputs,
            )
        )

    return specs


def _required_workload_specs(
    min_n: int,
    max_n: int,
    seed: int,
    templates: Sequence[str],
) -> List[WorkloadSpec]:
    """
    Deterministic calibration cases that should always be present in tuning.
    This keeps tuning data-driven while ensuring canonical kernels are represented.
    """
    rng = random.Random(seed + 1729)
    n = max(min(128, max_n), min_n)
    specs: List[WorkloadSpec] = []

    if "spmm" in templates:
        specs.append(
            WorkloadSpec(
                name=f"required_spmm_csr_dense_n{n}_0",
                template="spmm",
                n=n,
                fmt_out="dd",
                output_mode_order=(0, 1),
                inputs=(
                    InputTensorSpec("A", 2, "ds", rng.uniform(0.75, 0.995), (0, 1)),
                    InputTensorSpec("B", 2, "dd", 0.0, (0, 1)),
                ),
            )
        )
        specs.append(
            WorkloadSpec(
                name=f"required_spmm_csr_dense_n{n}_1",
                template="spmm",
                n=n,
                fmt_out="dd",
                output_mode_order=(1, 0),
                inputs=(
                    InputTensorSpec("A", 2, "ds", rng.uniform(0.75, 0.995), (1, 0)),
                    InputTensorSpec("B", 2, "dd", 0.0, (1, 0)),
                ),
            )
        )

    if "spgemm" in templates:
        specs.append(
            WorkloadSpec(
                name=f"required_spgemm_csr_csr_n{n}_0",
                template="spgemm",
                n=n,
                fmt_out="dd",
                output_mode_order=(0, 1),
                inputs=(
                    InputTensorSpec("A", 2, "ds", rng.uniform(0.75, 0.995), (0, 1)),
                    InputTensorSpec("B", 2, "ds", rng.uniform(0.75, 0.995), (0, 1)),
                ),
            )
        )
        specs.append(
            WorkloadSpec(
                name=f"required_spgemm_coord_csr_n{n}_1",
                template="spgemm",
                n=n,
                fmt_out="dd",
                output_mode_order=(1, 0),
                inputs=(
                    InputTensorSpec("A", 2, "oo", rng.uniform(0.75, 0.995), (1, 0)),
                    InputTensorSpec("B", 2, "ds", rng.uniform(0.75, 0.995), (0, 1)),
                ),
            )
        )

    return specs


def _instantiate_inputs_for_spec(spec: WorkloadSpec, seed: int) -> List[STensor]:
    generator = torch.Generator().manual_seed(seed)
    tensors: List[STensor] = []
    for input_spec in spec.inputs:
        dense = _make_dense_tensor(
            n=spec.n,
            rank=input_spec.rank,
            zero_fraction=input_spec.zero_frac,
            generator=generator,
        )
        tensors.append(_to_stensor(dense, input_spec))
    return tensors


def _normalize_spec_for_scoring(spec: WorkloadSpec) -> WorkloadSpec:
    """Normalize sparse 2D input mode orders to (0,1), mirroring matmul()'s normalization."""
    default_mo = (0, 1)
    has_sparse_input = any(
        not _is_dense_format(inp.rank, inp.fmt) for inp in spec.inputs
    )
    has_non_default = any(
        inp.rank == 2 and inp.mode_order != default_mo for inp in spec.inputs
    )
    if not (has_sparse_input and has_non_default):
        return spec

    new_inputs = []
    for inp in spec.inputs:
        if inp.rank == 2 and inp.mode_order != default_mo:
            inp = dataclasses.replace(inp, mode_order=default_mo)
        new_inputs.append(inp)

    return dataclasses.replace(
        spec,
        output_mode_order=default_mo,
        inputs=tuple(new_inputs),
    )


def _build_runtime_table_for_workload(
    spec: WorkloadSpec,
    benchmark_repeats: int,
    benchmark_warmup: int,
    module_cache: Dict[str, object],
    build_dir: Path,
    seed: int,
) -> Optional[WorkloadRuntimeTable]:
    inputs = _instantiate_inputs_for_spec(spec, seed)
    result_shape = (spec.n, spec.n)
    use_matmul_backend = spec.template in {"spmm", "spgemm"}

    candidate_runtimes: Dict[Tuple[str, str, str], float] = {}
    for perm in LOOP_PERMUTATIONS:
        if use_matmul_backend:
            runtime = _benchmark_forced_loop_matmul_eval_time(
                spec=spec,
                inputs=inputs,
                loop_order=perm,
                repeats=benchmark_repeats,
                warmup=benchmark_warmup,
            )
        else:
            cin_stmt = _build_workload_cin(spec, loop_order=perm)

            loop_order_vars, _ = Scheduler._extract_loop_chain(cin_stmt)
            if Scheduler.should_insert_workspace(cin_stmt, loop_order_vars):
                cin_stmt = Scheduler.insert_workspace(cin_stmt, allow_dense=True)

            runtime = _benchmark_cin_eval_time(
                cin_stmt=cin_stmt,
                result_shape=result_shape,
                inputs=inputs,
                module_cache=module_cache,
                build_dir=build_dir,
                repeats=benchmark_repeats,
                warmup=benchmark_warmup,
            )
        candidate_runtimes[perm] = runtime

    finite_runtimes = {
        perm: runtime
        for perm, runtime in candidate_runtimes.items()
        if math.isfinite(runtime)
    }
    if len(finite_runtimes) < 1:
        return None

    oracle_perm = min(finite_runtimes, key=finite_runtimes.get)
    oracle_runtime = finite_runtimes[oracle_perm]
    penalty_runtime = max(finite_runtimes.values()) * 20.0

    scoring_spec = _normalize_spec_for_scoring(spec) if use_matmul_backend else spec
    base_cin = _build_workload_cin(scoring_spec, loop_order=LOOP_VARS)

    return WorkloadRuntimeTable(
        spec=spec,
        base_cin=base_cin,
        candidate_runtimes=candidate_runtimes,
        oracle_perm=oracle_perm,
        oracle_runtime=oracle_runtime,
        penalty_runtime=penalty_runtime,
    )


def _selected_perm_for_stage(
    table: WorkloadRuntimeTable,
    params: _CostModelConstants,
    score_stage: str,
) -> Tuple[str, str, str]:
    if score_stage == "optimize":
        init = Scheduler.init_loop_order(table.base_cin, costs=params)
        selected = Scheduler.optimize_loop_order(table.base_cin, init, costs=params)
    elif score_stage == "full":
        selected = Scheduler.select_loop_order(table.base_cin, costs=params)
    else:
        raise ValueError(f"Unknown score stage: {score_stage}")
    return tuple(index_var.name for index_var in selected)


def _score_params(
    params: _CostModelConstants,
    runtime_tables: List[WorkloadRuntimeTable],
    score_stage: str,
) -> Tuple[float, Dict[str, float]]:
    ratios: List[float] = []
    per_workload_ratio: Dict[str, float] = {}

    for table in runtime_tables:
        selected_perm = _selected_perm_for_stage(
            table=table,
            params=params,
            score_stage=score_stage,
        )
        selected_runtime = table.candidate_runtimes.get(selected_perm, table.penalty_runtime)
        if not math.isfinite(selected_runtime):
            selected_runtime = table.penalty_runtime
        if table.oracle_runtime <= 0:
            return float("inf"), {}

        ratio = selected_runtime / table.oracle_runtime
        ratios.append(ratio)
        per_workload_ratio[table.spec.name] = ratio

    if not ratios:
        return float("inf"), {}
    return float(sum(ratios) / len(ratios)), per_workload_ratio


def _sample_random_params(rng: random.Random) -> _CostModelConstants:
    def log_uniform(lo: float, hi: float) -> float:
        return 10 ** rng.uniform(math.log10(lo), math.log10(hi))

    return _CostModelConstants(
        alpha=log_uniform(1e-2, 1e2),
        beta=log_uniform(1e-2, 1e2),
        gamma=log_uniform(1e-2, 1e2),
        c_insert=log_uniform(1e-2, 1e2),
        c_sort=log_uniform(1e-3, 1e1),
        c_trans=log_uniform(1e-2, 1e2),
        rho=log_uniform(1e-4, 5e-1),
        default_dim_size=Scheduler._DEFAULT_COSTS.default_dim_size,
    )


def _run_random_search(
    runtime_tables: List[WorkloadRuntimeTable],
    trials: int,
    seed: int,
    score_stage: str,
) -> TuneResult:
    rng = random.Random(seed)
    best_score = float("inf")
    best_params = Scheduler._DEFAULT_COSTS
    best_per_workload_ratio: Dict[str, float] = {}

    for trial_idx in range(1, trials + 1):
        params = _sample_random_params(rng)
        score, per_workload_ratio = _score_params(
            params=params,
            runtime_tables=runtime_tables,
            score_stage=score_stage,
        )
        if score < best_score:
            best_score = score
            best_params = params
            best_per_workload_ratio = per_workload_ratio
        print(
            f"[trial {trial_idx:03d}] score={score:.6f} best={best_score:.6f}",
            flush=True,
        )

    return TuneResult(
        best_params=best_params,
        best_score=best_score,
        method="random",
        trials=trials,
        per_workload_ratio=best_per_workload_ratio,
    )


def _run_optuna_search(
    runtime_tables: List[WorkloadRuntimeTable],
    trials: int,
    seed: int,
    score_stage: str,
) -> TuneResult:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed. Re-run with --method random or install optuna."
        ) from exc

    def objective(trial) -> float:
        params = _CostModelConstants(
            alpha=trial.suggest_float("alpha", 1e-2, 1e2, log=True),
            beta=trial.suggest_float("beta", 1e-2, 1e2, log=True),
            gamma=trial.suggest_float("gamma", 1e-2, 1e2, log=True),
            c_insert=trial.suggest_float("c_insert", 1e-2, 1e2, log=True),
            c_sort=trial.suggest_float("c_sort", 1e-3, 1e1, log=True),
            c_trans=trial.suggest_float("c_trans", 1e-2, 1e2, log=True),
            rho=trial.suggest_float("rho", 1e-4, 5e-1, log=True),
            default_dim_size=Scheduler._DEFAULT_COSTS.default_dim_size,
        )
        score, _ = _score_params(
            params=params,
            runtime_tables=runtime_tables,
            score_stage=score_stage,
        )
        return score if math.isfinite(score) else 1e30

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials)

    best_params = _CostModelConstants(
        alpha=study.best_params["alpha"],
        beta=study.best_params["beta"],
        gamma=study.best_params["gamma"],
        c_insert=study.best_params["c_insert"],
        c_sort=study.best_params["c_sort"],
        c_trans=study.best_params["c_trans"],
        rho=study.best_params["rho"],
        default_dim_size=Scheduler._DEFAULT_COSTS.default_dim_size,
    )
    best_score, per_workload_ratio = _score_params(
        params=best_params,
        runtime_tables=runtime_tables,
        score_stage=score_stage,
    )
    return TuneResult(
        best_params=best_params,
        best_score=best_score,
        method="optuna",
        trials=trials,
        per_workload_ratio=per_workload_ratio,
    )


def _dump_result_json(
    result: TuneResult,
    runtime_tables: List[WorkloadRuntimeTable],
    score_stage: str,
    templates: Sequence[str],
    output_path: Path,
) -> None:
    payload = {
        "method": result.method,
        "trials": result.trials,
        "score_stage": score_stage,
        "templates": list(templates),
        "best_score_mean_runtime_ratio": result.best_score,
        "best_params": dataclasses.asdict(result.best_params),
        "workloads": [
            {
                "name": table.spec.name,
                "template": table.spec.template,
                "n": table.spec.n,
                "fmt_out": table.spec.fmt_out,
                "output_mode_order": list(table.spec.output_mode_order),
                "inputs": [dataclasses.asdict(inp) for inp in table.spec.inputs],
                "oracle_perm": list(table.oracle_perm),
                "oracle_runtime_sec": table.oracle_runtime,
                "penalty_runtime_sec": table.penalty_runtime,
                "finite_perms": [
                    list(perm)
                    for perm, runtime in table.candidate_runtimes.items()
                    if math.isfinite(runtime)
                ],
                "best_selected_to_oracle_ratio": result.per_workload_ratio.get(
                    table.spec.name, float("inf")
                ),
            }
            for table in runtime_tables
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune scheduler cost-model constants")
    parser.add_argument("--method", choices=["random", "optuna"], default="random")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--workloads", type=int, default=8)
    parser.add_argument(
        "--templates",
        nargs="+",
        default=list(DEFAULT_TEMPLATES),
        help=(
            "Templates to tune over. "
            f"Supported: {', '.join(SUPPORTED_TEMPLATES)}. "
            "Use 'all' to include every supported template. "
            "Comma-separated values are also accepted."
        ),
    )
    parser.add_argument("--min-n", type=int, default=256)
    parser.add_argument("--max-n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--score-stage",
        choices=["optimize", "full"],
        default="full",
        help=(
            "optimize: tune pre-constraint greedy loop-order model only; "
            "full: tune full selection including mode-order constraints"
        ),
    )
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--benchmark-warmup", type=int, default=1)
    parser.add_argument("--torch-extensions-dir", type=str, default="/tmp/torch_extensions")
    parser.add_argument("--build-dir", type=str, default="/tmp/torch_extensions/scheduler_tune")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.min_n <= 0 or args.max_n < args.min_n:
        raise ValueError("Invalid dimension bounds: require 0 < min_n <= max_n")
    if args.workloads <= 0:
        raise ValueError("workloads must be > 0")
    if args.trials <= 0:
        raise ValueError("trials must be > 0")

    templates = _normalize_templates(args.templates)

    os.environ.setdefault("TORCH_EXTENSIONS_DIR", args.torch_extensions_dir)
    torch.set_num_threads(max(args.threads, 1))

    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    module_cache: Dict[str, object] = {}

    print(f"Tuning templates: {', '.join(templates)}", flush=True)

    required_specs = _required_workload_specs(
        min_n=args.min_n,
        max_n=args.max_n,
        seed=args.seed,
        templates=templates,
    )
    random_specs = _generate_workload_specs(
        count=max(args.workloads * max(4, len(templates) * 3), args.workloads),
        min_n=args.min_n,
        max_n=args.max_n,
        seed=args.seed,
        templates=templates,
    )
    specs = required_specs + random_specs

    runtime_tables: List[WorkloadRuntimeTable] = []
    for idx, spec in enumerate(specs):
        if len(runtime_tables) >= args.workloads:
            break
        inputs_desc = ", ".join(
            f"{inp.name}:{inp.fmt}@{list(inp.mode_order)}" for inp in spec.inputs
        )
        print(
            f"[workload {idx + 1}/{len(specs)}] {spec.name} "
            f"template={spec.template} out={spec.fmt_out}@{list(spec.output_mode_order)} "
            f"inputs=[{inputs_desc}]",
            flush=True,
        )
        table = _build_runtime_table_for_workload(
            spec=spec,
            benchmark_repeats=args.benchmark_repeats,
            benchmark_warmup=args.benchmark_warmup,
            module_cache=module_cache,
            build_dir=build_dir,
            seed=args.seed + idx + 1,
        )
        if table is None:
            print(f"  skipped {spec.name}: no finite candidate runtime", flush=True)
            continue
        num_finite = sum(math.isfinite(v) for v in table.candidate_runtimes.values())
        runtime_tables.append(table)
        print(
            f"  oracle={table.oracle_perm} runtime={table.oracle_runtime:.6f}s "
            f"finite_perms={num_finite}/{len(table.candidate_runtimes)}",
            flush=True,
        )

    if len(runtime_tables) < max(1, args.workloads // 2):
        raise RuntimeError(
            "Too few usable workloads were benchmarked. "
            "Try increasing --workloads or tightening workload templates."
        )

    baseline_score, _ = _score_params(
        params=Scheduler._DEFAULT_COSTS,
        runtime_tables=runtime_tables,
        score_stage=args.score_stage,
    )
    print(
        f"\nBaseline default-constants score "
        f"({args.score_stage} stage): {baseline_score:.6f}"
    )

    if args.method == "optuna":
        result = _run_optuna_search(
            runtime_tables=runtime_tables,
            trials=args.trials,
            seed=args.seed,
            score_stage=args.score_stage,
        )
    else:
        result = _run_random_search(
            runtime_tables=runtime_tables,
            trials=args.trials,
            seed=args.seed,
            score_stage=args.score_stage,
        )

    print("\nBest tuning result")
    print(f"method: {result.method}")
    print(f"trials: {result.trials}")
    print(f"mean selected/oracle runtime ratio: {result.best_score:.6f}")
    for key, value in dataclasses.asdict(result.best_params).items():
        print(f"{key}: {value}")

    if args.output_json:
        output_path = Path(args.output_json)
        _dump_result_json(
            result,
            runtime_tables,
            args.score_stage,
            templates,
            output_path,
        )
        print(f"\nWrote tuning report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
