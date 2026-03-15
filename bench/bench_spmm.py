#!/usr/bin/env python3
"""Benchmark SpMM: A_sparse @ B_dense.

Supports running on the curated 21-matrix "quick" set or the full
SuiteSparse collection available on disk ("full").  Use --continue
(on by default) to skip matrices already recorded in the CSV.

Each matrix is benchmarked in a subprocess so that C-level crashes
(segfaults) don't kill the entire run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _utils import (
    MATRIX_SETS,
    add_common_args,
    completed_matrices,
    discover_local_matrices,
    plot_scatter_loglog,
    ResultsCollector,
    suppress_torch_warnings,
)


def _get_matrices(matrix_set: str):
    if matrix_set == "quick":
        return MATRIX_SETS["spmm"]
    elif matrix_set == "full":
        return discover_local_matrices()
    else:
        raise ValueError(f"Unknown matrix set: {matrix_set!r}")


# -----------------------------------------------------------------------
# Subprocess worker – benchmarks a single matrix
# -----------------------------------------------------------------------

_WORKER_SCRIPT = r"""
import json, sys, gc, traceback
import torch
import scorch
from _utils import (
    benchmark_fn, check_correctness, download_matrix, load_matrix,
    scipy_to_torch, suppress_torch_warnings, to_scorch_csr, to_scorch_dense,
)

def run(params):
    suppress_torch_warnings()
    ssid    = params["ssid"]
    group   = params["group"]
    name    = params["name"]
    k       = params["k"]
    warmup  = params["warmup"]
    repeats = params["repeats"]
    seed    = params["seed"]

    torch.manual_seed(seed)

    if group == "local":
        csr = load_matrix(name)
    else:
        csr = download_matrix(ssid, group, name)

    n_rows, n_cols = csr.shape
    nnz = csr.nnz

    b_dense = torch.rand(n_cols, k, dtype=torch.float32)

    # PyTorch
    torch_csr = scipy_to_torch(csr, fmt="csr")
    ref, pt_timing = benchmark_fn(
        lambda: torch.sparse.mm(torch_csr, b_dense),
        warmup=warmup, repeats=repeats,
    )

    # Scorch
    a_st = to_scorch_csr(csr, "A")
    b_st = to_scorch_dense(b_dense, "B")
    sc_result, sc_timing = benchmark_fn(
        lambda: scorch.matmul(a_st, b_st, format="dd"),
        warmup=warmup, repeats=repeats,
    )

    sc_dense = sc_result if isinstance(sc_result, torch.Tensor) else sc_result.to_torch()
    correct = check_correctness(sc_dense, ref, name)

    result = {
        "name": name, "ssid": ssid,
        "rows": n_rows, "cols": n_cols, "nnz": nnz,
        "pt_times": list(pt_timing.times),
        "sc_times": list(sc_timing.times),
        "pt_median_ms": pt_timing.median_ms,
        "sc_median_ms": sc_timing.median_ms,
        "correct": correct,
    }
    print(json.dumps(result))

params = json.loads(sys.argv[1])
run(params)
"""


def _run_single(params: dict, timeout: int | None) -> dict | None:
    """Run the worker in a subprocess; return parsed result or None on failure."""
    cmd = [sys.executable, "-c", _WORKER_SCRIPT, json.dumps(params)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout,
            cwd=Path(__file__).parent,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s — skipping")
        return None

    if proc.returncode != 0:
        sig = -proc.returncode if proc.returncode < 0 else proc.returncode
        print(f"  CRASHED (exit {sig}) — skipping")
        stderr = proc.stderr.strip()
        if stderr:
            # Print last few lines of stderr for context
            for line in stderr.splitlines()[-5:]:
                print(f"    {line}")
        return None

    # The last line of stdout is the JSON result
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass

    print(f"  FAILED — no result in output")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="SpMM benchmark: A_sparse @ B_dense")
    add_common_args(parser)
    parser.add_argument("--k", type=int, default=128,
                        help="Number of columns in dense matrix B (default: 128)")
    parser.add_argument("--matrix-set", type=str, default="quick",
                        choices=["quick", "full"],
                        help="Matrix set: 'quick' (curated 21) or 'full' (all local) (default: quick)")
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        default=True,
                        help="Skip matrices already in CSV (default: on)")
    parser.add_argument("--no-continue", dest="continue_run", action="store_false",
                        help="Re-run all matrices even if already in CSV")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-matrix timeout in seconds (default: 300)")
    args = parser.parse_args()

    suppress_torch_warnings()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / (args.csv or "spmm_results.csv")
    plot_path = out_dir / f"spmm.{args.format}"

    if args.plot_only:
        df = ResultsCollector.load(csv_path)
        plot_scatter_loglog(df, f"SpMM Performance (k={args.k})", plot_path)
        return

    matrices = _get_matrices(args.matrix_set)
    done = completed_matrices(csv_path) if args.continue_run else set()

    collector = ResultsCollector(csv_path)

    total = len(matrices)
    skipped = 0
    failed = 0
    ran = 0

    print(f"SpMM Benchmark  (k={args.k}, warmup={args.warmup}, repeats={args.repeats})")
    print(f"Matrix set: {args.matrix_set} ({total} matrices)")
    if done:
        print(f"Continue mode: {len(done)} matrices already completed, will skip")
    print("=" * 80)

    for i, (ssid, group, name, desc) in enumerate(matrices, 1):
        if name in done:
            skipped += 1
            continue

        print(f"\n[{i}/{total}] {name}" + (f" (id={ssid}): {desc}" if desc else ""))

        params = {
            "ssid": ssid, "group": group, "name": name,
            "k": args.k, "warmup": args.warmup, "repeats": args.repeats,
            "seed": args.seed,
        }
        result = _run_single(params, timeout=args.timeout)

        if result is None:
            failed += 1
            continue

        n_rows = result["rows"]
        n_cols = result["cols"]
        nnz = result["nnz"]
        k = args.k

        print(f"  shape=({n_rows}, {n_cols}), nnz={nnz:,}, k={k}")
        print(f"  PyTorch  median={result['pt_median_ms']:.3f} ms")
        print(f"  Scorch   median={result['sc_median_ms']:.3f} ms")

        for t in result["pt_times"]:
            collector.append("PyTorch", name, ssid, n_rows, n_cols, nnz, t, k=k)
        for t in result["sc_times"]:
            collector.append("Scorch", name, ssid, n_rows, n_cols, nnz, t, k=k)

        ran += 1

    print("\n" + "=" * 80)
    print(f"Done. ran={ran}, skipped={skipped}, failed={failed}, total={total}")
    print(f"CSV saved to {csv_path.resolve()}")

    # Final plot from all data (including previous runs)
    try:
        df = ResultsCollector.load(csv_path)
        plot_scatter_loglog(df, f"SpMM Performance (k={args.k})", plot_path)
    except Exception:
        print("Warning: could not generate plot")


if __name__ == "__main__":
    main()
