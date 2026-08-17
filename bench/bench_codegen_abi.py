#!/usr/bin/env python3
"""Per-call cost of the ABI boundary on the JIT codegen path.

The prebuilt kernels reach their kernel through ``validate_binary_inputs`` /
``checked_csr_view``. Generated kernels do not: CINLowerer emits a
``scorch_native::validate_jit_tensor`` call per right-hand-side operand at the top of
every ``evaluate()``, and that validator walked the whole index structure with a
``TORCH_CHECK`` per element on every call. This measures what that costs and what
removing it is worth.

Two routes, because they charge different parts of the boundary:

``wksp_ds``
    ``matmul_wksp(A, B, output_format="ds")``. ``matmul_wksp`` *always* goes through the
    CIN compiler — never a prebuilt kernel, never the tiling selector — and the operand
    keeps the int64 indices ``torch.sparse_csr_tensor`` gave it, so this pays the
    narrowing AND the structural scans.
``dcsr_dd``
    ``scorch.matmul(A.to_sparse("ss"), B_dense)``: DCSR x dense with a dense result,
    which has no prebuilt kernel and so lowers through codegen. ``to_sparse`` emits
    int32 indices, so unless ``--index-dtype int64`` is given this pays the structural
    scans only. This is the combination the codegen zero-fill work used as its gate.

``output_format="dd"`` through ``matmul_wksp`` is NOT one of the routes: it fails to
compile with ``no matching function for call to min(int, int64_t&)`` in the generated
body, identically on both trees measured here. A pre-existing codegen defect, unrelated
to the ABI boundary, but it is why the dense-output cell goes through ``matmul``.

Method, same as the SpMM-vs-MKL grid:

* Arms are visited in a freshly shuffled order every round; the median over rounds is
  reported, so per-round drift hits every arm equally.
* ``aa`` is the ``codegen`` arm entered under a second name. |aa/codegen - 1| is that
  cell's in-process noise floor, and nothing smaller than it counts as a result.
* ``torch`` is ``torch.sparse.mm``, which is byte-identical between the two trees being
  compared, so it doubles as the CROSS-PROCESS control for a base/candidate A/B.
* Every cell is checked against a float64 reference.

Usage
-----
    python bench/bench_codegen_abi.py --csv out.csv                  # default cells
    python bench/bench_codegen_abi.py --ns 8 32 128 --reps 11
    SCORCH_ABI_VALIDATE_MEMO=0 python bench/bench_codegen_abi.py     # screens, no memo
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
import time

import numpy as np
import torch


def build_csr(name, M, J, deg, seed=0):
    """A CSR matrix with sorted columns, either synthetic or from the npz cache."""
    cache = os.environ.get("SCORCH_MCACHE")
    if cache and name not in ("band", "scatter"):
        path = os.path.join(cache, f"{name}.npz")
        if os.path.exists(path):
            z = np.load(path)
            return (z["indptr"].astype(np.int64), z["indices"].astype(np.int64),
                    z["data"].astype(np.float32), int(z["shape"][0]),
                    int(z["shape"][1]))
    rng = np.random.default_rng(seed)
    if name == "band":  # contiguous columns: the cheap-locality end
        cols = np.concatenate([((np.arange(deg) + i) % J) for i in range(M)])
        cols = np.sort(cols.reshape(M, deg), axis=1).reshape(-1)
    else:               # scattered columns: the expensive-locality end
        cols = np.concatenate([np.sort(rng.choice(J, size=deg, replace=False))
                               for _ in range(M)])
    indptr = np.arange(M + 1, dtype=np.int64) * deg
    data = rng.standard_normal(M * deg).astype(np.float32)
    return indptr, cols.astype(np.int64), data, M, J


def median_time(fn, reps, inner):
    """Median per-call seconds, holding results so the allocator behaves normally."""
    out = []
    for _ in range(reps):
        held = []
        t0 = time.perf_counter()
        for _ in range(inner):
            held.append(fn())
        out.append((time.perf_counter() - t0) / inner)
        del held
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+",
                    default=["band", "scatter", "pubmed", "bcsstk17"])
    ap.add_argument("--ns", nargs="+", type=int, default=[8, 32, 128])
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--deg", type=int, default=24)
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--tag", default="cand")
    ap.add_argument("--route", default="wksp_ds", choices=["wksp_ds", "dcsr_dd"])
    ap.add_argument("--index-dtype", default="int64", choices=["int32", "int64"])
    args = ap.parse_args()

    import scorch
    from scorch.stensor import STensor
    from scorch.ops import matmul_wksp


    memo = os.environ.get("SCORCH_ABI_VALIDATE_MEMO", "1")
    idtype = torch.int32 if args.index_dtype == "int32" else torch.int64
    print(f"# tag={args.tag} route={args.route} index_dtype={args.index_dtype} "
          f"SCORCH_ABI_VALIDATE_MEMO={memo} threads={torch.get_num_threads()}")
    print(f"{'matrix':12s} {'M':>8s} {'nnz':>10s} {'N':>5s} "
          f"{'codegen_ms':>11s} {'aa_ms':>9s} {'torch_ms':>9s} "
          f"{'floor%':>7s} {'relerr':>9s}")

    rows_out = []
    for name in args.matrices:
        indptr, indices, data, M, J = build_csr(name, args.rows, args.rows, args.deg)
        nnz = int(indices.size)
        t_csr = torch.sparse_csr_tensor(torch.from_numpy(indptr).to(idtype),
                                        torch.from_numpy(indices).to(idtype),
                                        torch.from_numpy(data), size=(M, J))
        if args.route == "wksp_ds":
            # from_torch inherits the source tensor's index dtype, so --index-dtype is
            # honoured here.
            A = STensor.from_torch(t_csr)
        else:
            # to_sparse("ss") builds a layout that declares int32 indices and scorch's
            # own storage validator rejects anything else, so this route is int32 no
            # matter what --index-dtype says: it charges the structural scans only.
            A = STensor.from_torch(t_csr).to_sparse("ss")
        for N in args.ns:
            rng = np.random.default_rng(7)
            Bd = torch.from_numpy(rng.standard_normal((J, N)).astype(np.float32))
            B = STensor.from_torch(Bd)

            if args.route == "wksp_ds":
                def codegen():
                    return matmul_wksp(A, B, output_format="ds")
            else:
                def codegen():
                    return scorch.matmul(A, Bd)

            def torch_mm():
                return torch.sparse.mm(t_csr, Bd)

            try:
                first = codegen()          # JIT compile before anything is timed
            except Exception as exc:       # noqa: BLE001
                print(f"{name:12s} {M:8d} {nnz:10d} {N:5d}   SKIP {type(exc).__name__}:"
                      f" {str(exc)[:90]}")
                continue
            ref_csr = torch.sparse_csr_tensor(
                torch.from_numpy(indptr), torch.from_numpy(indices),
                torch.from_numpy(data.astype(np.float64)), size=(M, J))
            ref = torch.sparse.mm(ref_csr, Bd.to(torch.float64)).numpy()
            got = first.to_torch() if hasattr(first, "to_torch") else first
            if got.is_sparse or got.layout != torch.strided:
                got = got.to_dense()
            g = got.detach().numpy()
            relerr = float(np.abs(g.reshape(ref.shape) - ref).max()
                           / max(np.abs(ref).max(), 1e-30))

            for _ in range(2):             # warm caches and the allocator
                codegen()
                torch_mm()

            arms = {"codegen": codegen, "aa": codegen, "torch": torch_mm}
            acc = {k: [] for k in arms}
            for _ in range(args.reps):
                order = list(arms)
                random.shuffle(order)
                for k in order:
                    acc[k].append(median_time(arms[k], 1, args.inner))
            ms = {k: statistics.median(v) * 1e3 for k, v in acc.items()}
            floor = abs(ms["aa"] / ms["codegen"] - 1.0) * 100
            print(f"{name:12s} {M:8d} {nnz:10d} {N:5d} "
                  f"{ms['codegen']:11.4f} {ms['aa']:9.4f} {ms['torch']:9.4f} "
                  f"{floor:7.2f} {relerr:9.2e}")
            rows_out.append(dict(tag=args.tag, memo=memo, matrix=name, M=M, J=J,
                                 nnz=nnz, N=N, codegen_ms=ms["codegen"],
                                 aa_ms=ms["aa"], torch_ms=ms["torch"],
                                 floor_pct=floor, relerr=relerr))

    if args.csv and rows_out:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nwrote {args.csv} ({len(rows_out)} cells)")
    bad = [r for r in rows_out if r["relerr"] > 1e-4]
    if bad:
        print(f"CORRECTNESS FAILURES: {len(bad)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
