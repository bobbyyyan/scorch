"""tile-ijk's inner accumulation: register-resident vs the scalar per-nonzero form.

spmm_csr_float_tileijk relayouts a strip of B, then for each contraction panel and
each row accumulates that panel's nonzeros into the row's Nc-wide slice of a
cache-resident output panel Cp. The accumulation was

    for (; pb < pe; ++pb) { a = A_val[pb]; ... for (k = 0; k < w; ++k) C_row[k] += a * B_row[k]; }

which issues w L1 loads and w L1 stores PER NONZERO. Nc exists precisely so Cp fits
the cache, and at the widths the cost model actually picks the slice fits REGISTERS --
so the row can be loaded once, accumulated over the panel's nonzeros in registers, and
stored once. This prices that.

Nc and Jc come from production (`tiling._ijk_params` with `tiling.query_llc()`), not
from a copy of the formula here: a harness that restates a production policy drifts
from it silently and flatteringly, which has already happened three times in this
project.

Both arms are the SAME binary, selected per call by SCORCH_TILEIJK_SCALAR, so they
interleave in one process.

  scalar    SCORCH_TILEIJK_SCALAR=1, the loop that shipped
  register  =0, an accumulating pass of scorch_spmm_row_neon
  aa        a second entry of `register`, the control
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch

from scorch import tiling


def synth(kind, rows, cols, deg, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "powerlaw":
        raw = rng.pareto(1.5, size=rows) + 1.0
        lens = np.maximum(1, np.rint(raw * deg / raw.mean())).astype(np.int64)
        lens = np.minimum(lens, cols)
    else:
        lens = np.full(rows, min(deg, cols), dtype=np.int64)
    indptr = np.zeros(rows + 1, dtype=np.int64)
    np.cumsum(lens, out=indptr[1:])
    total = int(indptr[-1])
    crd = np.empty(total, dtype=np.int64)
    for i in range(rows):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        d = hi - lo
        if kind == "banded":
            centre = int(i * cols / max(rows, 1))
            start = min(max(centre - d // 2, 0), max(cols - d, 0))
            crd[lo:hi] = np.arange(start, start + d)
        else:
            crd[lo:hi] = np.sort(rng.choice(cols, size=d, replace=False))
    val = (rng.standard_normal(total) * 0.05).astype(np.float32)
    return indptr.astype(np.int32), crd.astype(np.int32), val, total


def calibrate_reps(call, target_s, cap=512):
    reps = 1
    while reps < cap:
        t0 = time.perf_counter()
        for _ in range(reps):
            call()
        dt = time.perf_counter() - t0
        if dt >= target_s:
            return reps
        reps = min(cap, max(reps * 2, int(reps * target_s / max(dt, 1e-9) * 1.3) + 1))
    return cap


def timed(specs, rounds, reps, seed=0):
    best = [float("inf")] * len(specs)
    rng = random.Random(seed)
    for setup, call in specs:
        setup()
        call()
    for _ in range(rounds):
        for j in rng.sample(range(len(specs)), len(specs)):
            setup, call = specs[j]
            setup()
            t0 = time.perf_counter()
            for _ in range(reps):
                call()
            best[j] = min(best[j], (time.perf_counter() - t0) / reps)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mats", default="uniform:40000:40000:16,uniform:200000:200000:8,"
                                      "powerlaw:100000:100000:24,banded:40000:40000:33,"
                                      "uniform:20000:20000:64")
    ap.add_argument("--ns", default="512,1024,2048")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--batch-ms", type=float, default=8.0)
    args = ap.parse_args()

    import scorch_ops as SO
    if getattr(SO, "spmm_tune_hooks", None) is False:
        raise SystemExit("rebuild with SCORCH_BUILD_TUNE_HOOKS=1")
    with open(SO.__file__, "rb") as f:
        if b"SCORCH_TILEIJK_SCALAR" not in f.read():
            raise SystemExit(
                f"{SO.__file__} has no SCORCH_TILEIJK_SCALAR: both arms would be the "
                "same loop. Rebuild that tree with SCORCH_BUILD_TUNE_HOOKS=1.")

    C = tiling.query_llc()
    nt = torch.get_num_threads()
    print(f"threads={nt}  rounds={args.rounds}  LLC={C/1e6:.1f} MB  "
          f"batch floor {args.batch_ms:.1f} ms")
    print(f"\n{'matrix':<26}{'N':>6}{'Nc':>5}{'Jc':>8}{'nnz':>10}"
          f"{'scalar ms':>11}{'reg ms':>10}{'scl/reg':>9}{'A/A':>7}{'relerr':>10}")
    rows_out = []
    for spec in args.mats.split(","):
        kind, M, J, deg = spec.split(":")
        M, J, deg = int(M), int(J), int(deg)
        pos, crd, val, nnz = synth(kind, M, J, deg)
        tp, tc, tv = torch.from_numpy(pos), torch.from_numpy(crd), torch.from_numpy(val)
        Aidx = [[], [tp, tc]]
        for N in (int(x) for x in args.ns.split(",")):
            if (M * N + J * N) * 4 > 6e9:
                continue
            Nc, Jc = tiling._ijk_params(N, M, J, C)   # production's own choice
            B = torch.randn(J, N, dtype=torch.float32)
            shapes = ([M, N], [M, J], Aidx, tv, [J, N], [[], []], B.reshape(-1))

            def call():
                return SO.spmm_csr_float_tileijk(*shapes, Nc, Jc, nt)

            def set_scalar():
                os.environ["SCORCH_TILEIJK_SCALAR"] = "1"

            def set_reg():
                os.environ["SCORCH_TILEIJK_SCALAR"] = "0"

            set_scalar()
            ref = call().storage.value.reshape(M, N).clone()
            set_reg()
            got = call().storage.value.reshape(M, N)
            scale = max(ref.abs().max().item(), 1e-30)
            relerr = (got - ref).abs().max().item() / scale
            if relerr > 1e-4:
                raise SystemExit(f"{spec}@{N}: arms disagree, relerr {relerr:.3e}")

            reps = calibrate_reps(call, args.batch_ms * 1e-3)
            t = timed([(set_scalar, call), (set_reg, call), (set_reg, call)],
                      args.rounds, reps)
            t_s, t_r = t[0] * 1e3, t[1] * 1e3
            aa = max(t[1], t[2]) / min(t[1], t[2])
            rows_out.append((spec, N, Nc, t_s / t_r, aa))
            print(f"{spec:<26}{N:>6}{Nc:>5}{Jc:>8}{nnz:>10}"
                  f"{t_s:>11.3f}{t_r:>10.3f}{t_s/t_r:>9.3f}{aa:>7.3f}{relerr:>10.2e}")
            del B

    if not rows_out:
        return
    r = [x[3] for x in rows_out]
    aas = sorted(x[4] for x in rows_out)
    g = math.exp(sum(map(math.log, r)) / len(r))
    print("\n" + "=" * 78)
    print(f"n={len(rows_out)} cells   scalar/register geomean {g:.3f}  min {min(r):.3f}  "
          f"max {max(r):.3f}  cells where the register pass loses: "
          f"{sum(1 for x in r if x < 1.0)}")
    print(f"  A/A control: {aas[0]:.3f}-{aas[-1]:.3f}  median {aas[len(aas)//2]:.3f}")
    # Nc > 32 needs more than one register strip, so the row is walked more than once
    # per panel; that is where the change should pay least.
    one = [x[3] for x in rows_out if x[2] <= 32]
    many = [x[3] for x in rows_out if x[2] > 32]
    if one:
        print(f"  Nc <= 32 (one strip, one pass):   n={len(one)} "
              f"geomean {math.exp(sum(map(math.log, one))/len(one)):.3f}")
    if many:
        print(f"  Nc  > 32 (multi-strip, re-walks): n={len(many)} "
              f"geomean {math.exp(sum(map(math.log, many))/len(many)):.3f}")


if __name__ == "__main__":
    main()
