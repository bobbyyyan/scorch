#!/usr/bin/env python3
"""Benchmark all SpMM kernel variants on SuiteSparse matrices.

Compares Scorch kernel variants (tiled, untiled, direct, etc.) and
PyTorch MKL across a representative sample of matrices at different
sizes.  Each matrix runs in a subprocess for crash isolation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from _utils import (
    CACHE_DIR,
    MATRIX_SETS,
    discover_local_matrices,
    load_matrix,
    suppress_torch_warnings,
)


# Representative matrices spanning small → very large
_SAMPLE_MATRICES = [
    # Small
    "494_bus",       # 1.7K nnz
    "bcspwr09",      # 6.5K
    "bcsstk08",      # 13K
    # Medium
    "bcsstk14",      # 63K
    "bcsstk17",      # 429K
    "crystk02",      # 969K
    # Large
    "bcsstk30",      # 2M
    "ct20stif",      # 2.6M
    "gupta2",        # 4.2M
    "pre2",          # 5.8M
    "pkustk11",      # 5.2M
    # Very large
    "mouse_gene",    # 29M
    # Different sparsity patterns
    "minsurfo",      # 204K - regular grid
    "mosfet2",       # 1.5M - circuit sim
    "mixtank_new",   # 2M - CFD
    "amazon0312",    # 1.2M - graph
    "ca-CondMat",    # 186K - small graph
    "cfd2",          # 3M - CFD
    "parabolic_fem", # 3.7M - FEM
    "thermal2",      # 8.6M - thermal
]


# -----------------------------------------------------------------------
# Subprocess worker
# -----------------------------------------------------------------------

_WORKER_SCRIPT = r"""
import json, sys, time, statistics
import torch
import numpy as np
import scipy.io, scipy.sparse
import scorch_ops as ops
from scorch import STensor

def load(name, cache_dir, on_redwood):
    from pathlib import Path
    if on_redwood:
        p = Path(cache_dir) / name / f"{name}.mtx"
    else:
        p = Path(cache_dir) / f"{name}.mtx"
    mat = scipy.io.mmread(str(p))
    return scipy.sparse.csr_matrix(mat, dtype=np.float32)

def to_kernel_args(csr, k):
    n_rows, n_cols = csr.shape
    indptr = torch.from_numpy(csr.indptr.astype(np.int32))
    indices = torch.from_numpy(csr.indices.astype(np.int32))
    values = torch.from_numpy(csr.data.astype(np.float32))
    torch_csr = torch.sparse_csr_tensor(indptr, indices, values, size=csr.shape)
    a_st = STensor.from_csr(torch_csr, "A")
    b_dense = torch.rand(n_cols, k, dtype=torch.float32)
    b_st = STensor.from_torch(b_dense, "B")
    result_shape = [n_rows, k]
    args = [result_shape]
    for t in (a_st, b_st):
        args.append(list(t.shape))
        args.append(t.index.mode_indices)
        args.append(t.values)
    return args, b_dense, torch_csr

def bench(fn, args, kwargs, warmup, repeats):
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return result, times

def run(params):
    name     = params["name"]
    k        = params["k"]
    warmup   = params["warmup"]
    repeats  = params["repeats"]
    cache_dir = params["cache_dir"]
    on_redwood = params["on_redwood"]

    torch.manual_seed(42)
    csr = load(name, cache_dir, on_redwood)
    n_rows, n_cols = csr.shape
    nnz = csr.nnz

    kernel_args, b_dense, torch_csr = to_kernel_args(csr, k)

    variants = [
        ("baseline",      ops.prebuilt_spmm_csr_f32, {}),
        ("untiled",       ops.spmm_csr_float_untiled, {}),
        ("optimized",     ops.spmm_csr_float_optimized, {}),
        ("turbo",         ops.spmm_csr_float_turbo, {}),
        ("ultra",         ops.spmm_csr_float_ultra, {}),
        ("apex",          ops.spmm_csr_float_apex, {}),
        ("tiled_i_k",    ops.spmm_csr_float_tiled_i_k, {}),
        ("direct",        ops.spmm_csr_float_direct, {}),
        ("neon",          ops.spmm_csr_float_neon, {}),
        ("row_panel",     ops.spmm_csr_float_row_panel, {}),
        ("k_parallel",    ops.spmm_csr_float_k_parallel, {}),
        ("sorted_rows",   ops.spmm_csr_float_sorted_rows, {}),
        ("neon2",         ops.spmm_csr_float_neon2, {}),
        ("neon4",         ops.spmm_csr_float_neon4, {}),
        ("tiled_neon",    ops.spmm_csr_float_tiled_neon, {}),
        ("v2",            ops.spmm_csr_float_v2, {}),
    ]

    # PyTorch reference
    for _ in range(warmup):
        torch.sparse.mm(torch_csr, b_dense)
    pt_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ref = torch.sparse.mm(torch_csr, b_dense)
        pt_times.append(time.perf_counter() - t0)

    results = {
        "name": name, "rows": n_rows, "cols": n_cols, "nnz": nnz,
        "pytorch": {
            "median_ms": statistics.median(pt_times) * 1e3,
            "min_ms": min(pt_times) * 1e3,
            "times": pt_times,
        },
        "variants": {},
    }

    for vname, fn, kwargs in variants:
        try:
            out, times = bench(fn, kernel_args, kwargs, warmup, repeats)
            out_tensor = out.storage.value.view(n_rows, k)
            correct = torch.allclose(out_tensor, ref, atol=1e-2, rtol=1e-2)
            results["variants"][vname] = {
                "median_ms": statistics.median(times) * 1e3,
                "min_ms": min(times) * 1e3,
                "times": times,
                "correct": correct,
            }
        except Exception as e:
            results["variants"][vname] = {"error": str(e)}

    # --- Scorch compiler-generated path ---
    import scorch
    a_st = STensor.from_csr(torch_csr, "A")
    b_st = STensor.from_torch(b_dense, "B")
    try:
        for _ in range(warmup):
            scorch.ops.matmul_wksp(a_st, b_st, output_format="dd", output_mode_order=[0,1], use_cache=True)
        comp_times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            scorch.ops.matmul_wksp(a_st, b_st, output_format="dd", output_mode_order=[0,1], use_cache=True)
            comp_times.append(time.perf_counter() - t0)
        comp_out = scorch.ops.matmul_wksp(a_st, b_st, output_format="dd", output_mode_order=[0,1], use_cache=True)
        comp_dense = comp_out if isinstance(comp_out, torch.Tensor) else comp_out.to_torch()
        correct = torch.allclose(comp_dense, ref, atol=1e-2, rtol=1e-2)
        results["variants"]["compiler"] = {
            "median_ms": statistics.median(comp_times) * 1e3,
            "min_ms": min(comp_times) * 1e3,
            "times": comp_times,
            "correct": correct,
        }
    except Exception as e:
        results["variants"]["compiler"] = {"error": str(e)}

    print(json.dumps(results))

params = json.loads(sys.argv[1])
run(params)
"""


def _run_single(params: dict, timeout: int) -> dict | None:
    cmd = [sys.executable, "-c", _WORKER_SCRIPT, json.dumps(params)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=Path(__file__).parent,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return None
    if proc.returncode != 0:
        sig = -proc.returncode if proc.returncode < 0 else proc.returncode
        print(f"  CRASHED (exit {sig})")
        stderr = proc.stderr.strip()
        if stderr:
            for line in stderr.splitlines()[-3:]:
                print(f"    {line}")
        return None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    print("  FAILED — no result")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SpMM kernel variants on SuiteSparse")
    parser.add_argument("--k", type=int, default=128, help="Dense columns (default: 128)")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iters (default: 3)")
    parser.add_argument("--repeats", type=int, default=10, help="Timed iters (default: 10)")
    parser.add_argument("--timeout", type=int, default=300, help="Per-matrix timeout (default: 300s)")
    parser.add_argument("--output-dir", type=str, default="bench_results")
    args = parser.parse_args()

    suppress_torch_warnings()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "spmm_variants.csv"

    # Verify sample matrices exist
    matrices = []
    for name in _SAMPLE_MATRICES:
        try:
            load_matrix(name)
            matrices.append(name)
        except FileNotFoundError:
            print(f"  Skipping {name} — not found locally")

    from _utils import _ON_REDWOOD

    all_rows = []

    print(f"SpMM Variant Benchmark  (k={args.k}, warmup={args.warmup}, repeats={args.repeats})")
    print(f"Matrices: {len(matrices)}")
    print("=" * 100)

    for i, name in enumerate(matrices, 1):
        print(f"\n[{i}/{len(matrices)}] {name}")

        params = {
            "name": name, "k": args.k,
            "warmup": args.warmup, "repeats": args.repeats,
            "cache_dir": str(CACHE_DIR), "on_redwood": _ON_REDWOOD,
        }
        result = _run_single(params, timeout=args.timeout)
        if result is None:
            continue

        n_rows = result["rows"]
        n_cols = result["cols"]
        nnz = result["nnz"]
        pt = result["pytorch"]

        print(f"  shape=({n_rows}, {n_cols}), nnz={nnz:,}")
        print(f"  {'pytorch (MKL)':20s}  median={pt['median_ms']:9.3f} ms")

        # Collect rows
        all_rows.append({
            "Matrix": name, "Rows": n_rows, "Cols": n_cols, "NNZ": nnz,
            "Variant": "pytorch", "MedianMs": pt["median_ms"], "MinMs": pt["min_ms"],
            "Correct": True,
        })

        for vname, vdata in result["variants"].items():
            if "error" in vdata:
                print(f"  {vname:20s}  ERROR: {vdata['error']}")
                continue
            tag = "" if vdata.get("correct", True) else " [WRONG]"
            speedup = pt["median_ms"] / vdata["median_ms"] if vdata["median_ms"] > 0 else 0
            print(f"  {vname:20s}  median={vdata['median_ms']:9.3f} ms  ({speedup:.2f}x vs MKL){tag}")
            all_rows.append({
                "Matrix": name, "Rows": n_rows, "Cols": n_cols, "NNZ": nnz,
                "Variant": vname, "MedianMs": vdata["median_ms"], "MinMs": vdata["min_ms"],
                "Correct": vdata.get("correct", False),
            })

    # Save CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to {csv_path.resolve()}")

    # Print summary: best variant per matrix
    print("\n" + "=" * 100)
    print("SUMMARY — Best variant per matrix (median ms)")
    print("-" * 100)
    correct_df = df[df["Correct"] == True]
    for name in matrices:
        sub = correct_df[correct_df["Matrix"] == name]
        if sub.empty:
            continue
        best = sub.loc[sub["MedianMs"].idxmin()]
        pt_row = sub[sub["Variant"] == "pytorch"]
        pt_ms = pt_row["MedianMs"].values[0] if not pt_row.empty else float("inf")
        speedup = pt_ms / best["MedianMs"] if best["MedianMs"] > 0 else 0
        print(f"  {name:20s}  nnz={int(best['NNZ']):>12,}  "
              f"best={best['Variant']:20s}  {best['MedianMs']:9.3f} ms  "
              f"({speedup:.2f}x vs MKL)")


if __name__ == "__main__":
    main()
