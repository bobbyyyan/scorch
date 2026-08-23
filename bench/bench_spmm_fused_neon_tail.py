"""The fused Linear path's NEON row kernel against the one it replaced, on ARM.

`spmm_csr_linear_fused_float` had its own NEON row kernel, `scorch_spmm_row_neon_
regtile`, written beside its row loop rather than shared with the drop-in SpMM. Its
32-wide strip body was fine. Its tail was not: it walked the row's nonzeros ONCE PER
REMAINING COLUMN, outside the nonzero loop, so a free dimension under 32 read the row
that many times instead of once and a ragged width paid extra full walks for its last
few columns. The fused path now calls `scorch_spmm_row_neon`, which carries the
remainder in scalar accumulators updated in the same pass over the row.

Here the free dimension is the BATCH -- the sparse operand is the weight -- so the
widths that matter are small batches and batches that are not multiples of 32, which is
every batch that is a dataset's last incomplete one.

Both arms are the SAME binary, built with SCORCH_BUILD_TUNE_HOOKS=1, selected per call
by SCORCH_FUSED_LEGACY_TAIL, which the kernel reads once per op. Interleaved in one
process because cross-run drift on a laptop is larger than the effect.

  legacy   SCORCH_FUSED_LEGACY_TAIL=1, the kernel that shipped
  shared   =0, scorch_spmm_row_neon
  aa       a second entry of `shared`, the control; `shared`'s reported time is its
           FIRST entry, so every arm is estimated from the same number of draws

Timed regions are batched up to --batch-ms, because an AE-sized fused call at a small
batch is a few tens of microseconds -- shorter than the thread-wake noise it sits in,
which is how an earlier grid on this host produced a control of 1.4 and a result that
was a coin. Both arms are batched identically.
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch

from scorch import ops
from scorch.stensor import STensor


def require_hook(SO):
    """Fail closed if the A/B hook is not compiled into this build.

    Without -DSCORCH_TUNE_HOOKS both arms take the same path and every ratio reads
    ~1.000 with tight controls, which is indistinguishable from "the replacement
    changed nothing". See tests/test_scorch/test_prebuilt_kernel_registry.py.
    """
    flagged = getattr(SO, "spmm_tune_hooks", None)
    if flagged is False:
        raise SystemExit(
            "this extension reports spmm_tune_hooks=False: rebuild with "
            "SCORCH_BUILD_TUNE_HOOKS=1, or both arms are the same code")
    so = getattr(SO, "__file__", None)
    if not so or not os.path.exists(so):
        raise SystemExit("cannot locate scorch_ops to verify the hook")
    with open(so, "rb") as f:
        if b"SCORCH_FUSED_LEGACY_TAIL" not in f.read():
            raise SystemExit(
                f"{so} has no SCORCH_FUSED_LEGACY_TAIL: the hook is not in the binary, "
                "so both arms would run the same kernel. Rebuild that tree with "
                "SCORCH_BUILD_TUNE_HOOKS=1 or point PYTHONPATH at the tree that has it.")


def sparse_weight(out_dim, in_dim, density, seed=0):
    """A CSR weight with a fixed number of nonzeros per output row.

    Fixed degree rather than Bernoulli so the row length is the swept variable and not
    a per-row random one: the tail defect costs per nonzero per row, and a degree that
    varies row to row would average it away.
    """
    rng = np.random.default_rng(seed)
    deg = max(1, int(round(in_dim * density)))
    pos = np.arange(out_dim + 1, dtype=np.int64) * deg
    crd = np.empty(out_dim * deg, dtype=np.int64)
    for r in range(out_dim):
        crd[r * deg:(r + 1) * deg] = np.sort(rng.choice(in_dim, size=deg, replace=False))
    val = rng.standard_normal(out_dim * deg).astype(np.float32) * 0.05
    dense = torch.zeros(out_dim, in_dim, dtype=torch.float32)
    rows = np.repeat(np.arange(out_dim), deg)
    dense[torch.from_numpy(rows), torch.from_numpy(crd)] = torch.from_numpy(val)
    return dense, deg


def calibrate_reps(call, target_s, cap=4096):
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
    ap.add_argument("--shapes", default="256:784:0.10,64:256:0.10,784:256:0.05,"
                                        "1024:4096:0.01,4096:1024:0.01")
    ap.add_argument("--batches", default="1,4,8,16,24,31,32,33,48,64,96,100,128,256,512")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--batch-ms", type=float, default=3.0)
    ap.add_argument("--act", default="relu",
                    choices=("none", "relu", "sigmoid"))
    args = ap.parse_args()

    import scorch_ops as SO
    require_hook(SO)
    act_arg = None if args.act == "none" else args.act

    print(f"threads={torch.get_num_threads()}  rounds={args.rounds}  act={args.act}  "
          f"batch floor {args.batch_ms:.1f} ms")
    print(f"\n{'weight':<16}{'deg':>6}{'batch':>7}{'legacy ms':>11}{'shared ms':>11}"
          f"{'lgc/shr':>9}{'A/A':>7}{'reps':>6}{'relerr':>10}")
    rows = []
    for spec in args.shapes.split(","):
        o, i, d = spec.split(":")
        dense, deg = sparse_weight(int(o), int(i), float(d))
        w = STensor.from_torch(dense.to_sparse_csr())
        bias = torch.randn(int(o), dtype=torch.float32) * 0.1
        for batch in (int(x) for x in args.batches.split(",")):
            x_fm = torch.randn(int(i), batch, dtype=torch.float32)

            def call():
                return ops.sparse_linear_fm(x_fm, w, bias, activation=act_arg)

            def set_legacy():
                os.environ["SCORCH_FUSED_LEGACY_TAIL"] = "1"

            def set_shared():
                os.environ["SCORCH_FUSED_LEGACY_TAIL"] = "0"

            # Both arms against a dense reference, not just against each other: two
            # arms can agree and both be wrong, and the tail is exactly where an
            # off-by-one would hide.
            ref = torch.matmul(dense, x_fm) + bias.view(-1, 1)
            if args.act == "relu":
                ref = torch.relu(ref)
            elif args.act == "sigmoid":
                ref = torch.sigmoid(ref)
            scale = max(ref.abs().max().item(), 1e-30)
            errs = []
            for setter in (set_legacy, set_shared):
                setter()
                errs.append((call() - ref).abs().max().item() / scale)
            if max(errs) > 1e-4:
                raise SystemExit(f"{spec}@{batch}: wrong result, relerr {errs}")

            set_shared()
            reps = calibrate_reps(call, args.batch_ms * 1e-3)
            t = timed([(set_legacy, call), (set_shared, call), (set_shared, call)],
                      args.rounds, reps)
            t_l, t_s = t[0] * 1e3, t[1] * 1e3
            aa = max(t[1], t[2]) / min(t[1], t[2])
            rows.append((spec, batch, t_l / t_s, aa))
            print(f"{spec:<16}{deg:>6}{batch:>7}{t_l:>11.4f}{t_s:>11.4f}"
                  f"{t_l/t_s:>9.3f}{aa:>7.3f}{reps:>6}{max(errs):>10.2e}")

    if not rows:
        return
    r = [x[2] for x in rows]
    aas = sorted(x[3] for x in rows)
    g = math.exp(sum(map(math.log, r)) / len(r))
    print("\n" + "=" * 78)
    print(f"n={len(rows)} cells   legacy/shared geomean {g:.3f}  min {min(r):.3f}  "
          f"max {max(r):.3f}  cells where the replacement loses: "
          f"{sum(1 for x in r if x < 1.0)}")
    print(f"  A/A control: {aas[0]:.3f}-{aas[-1]:.3f}  (median "
          f"{aas[len(aas)//2]:.3f})")
    # Two changes ride together here and the batch width separates them. At an exact
    # multiple of 32 there is no remainder, so the tail fix cannot be doing anything
    # and what is left is the 2-nonzero unroll the old fused kernel never had. Every
    # other width pays for the remainder as well, so the difference between the two
    # rows below is what the tail defect actually cost.
    exact = [x[2] for x in rows if x[1] % 32 == 0]
    ragged = [x[2] for x in rows if x[1] % 32 != 0]
    if exact:
        print(f"  batch % 32 == 0 -- the 2-nnz unroll alone:  n={len(exact)} "
              f"geomean {math.exp(sum(map(math.log, exact))/len(exact)):.3f}")
    if ragged:
        print(f"  batch % 32 != 0 -- unroll plus the tail:    n={len(ragged)} "
              f"geomean {math.exp(sum(map(math.log, ragged))/len(ragged)):.3f}")
    small = [x[2] for x in rows if x[1] < 32]
    if small:
        print(f"  batch < 32 -- the row was re-walked once per column: n={len(small)} "
              f"geomean {math.exp(sum(map(math.log, small))/len(small)):.3f}")


if __name__ == "__main__":
    main()
