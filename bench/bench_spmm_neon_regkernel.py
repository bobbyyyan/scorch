"""The NEON register row kernel against the workspace loop it replaces, on ARM.

Until this change the drop-in SpMM's only register-resident path was guarded on
``__AVX2__``, so on ARM every row went through the workspace loop: memset a tile,
load-modify-store into it once per nonzero, memcpy it out. This prices the
replacement.

Both arms are the SAME binary, built with ``SCORCH_BUILD_TUNE_HOOKS=1``, and selected
per call by ``SCORCH_SPMM_WORKSPACE`` -- which ``spmm_csr_v2_core`` reads with getenv
on every call, so the two paths interleave inside one process. That matters more than
usual here: cross-run drift on a laptop under thermal control is larger than the
effect, and a two-process comparison could not see it.

  ws     SCORCH_SPMM_WORKSPACE=1, the path that shipped on ARM
  neon   SCORCH_SPMM_WORKSPACE=0, scorch_spmm_row_neon
  aa     a second entry of `neon`, the control; `neon`'s reported time is its FIRST
         entry, so every arm is estimated from the same number of draws

The hook build carries per-row hook overhead on BOTH arms, which pushes the ratio
toward 1 -- so a win measured here is a lower bound on the shipped one.

The environment is set OUTSIDE the timed region. Setting it inside cost 0.5% in an
earlier harness in this project, which is the same order as some of the effects here.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Synthetic operands, so the grid can be swept along one axis at a time.
#
# The real-matrix corpus is all square-ish and its k values are all multiples of 4,
# which leaves two things the register kernel cares about completely unmeasured: the
# ragged tail (k not a whole number of NEON lanes -- the scalar-remainder code) and
# any aspect ratio other than square. Degree distribution matters too: a power-law
# matrix puts wildly different row lengths through a kernel whose accumulators are
# sized per row.
#
# Spec grammar: kind:rows:cols:deg
#   uniform   deg columns per row drawn uniformly -- no locality, worst case for B
#   banded    deg columns clustered around the diagonal -- B stays cache-resident
#   powerlaw  row degrees Zipf-ish with mean deg -- a few very long rows, many short
#   blocked   rows in groups of 8 share a column block -- FEM/mesh-like reuse
# ---------------------------------------------------------------------------
def synth(spec, seed=0):
    kind, rows, cols, deg = spec.split(":")
    rows, cols, deg = int(rows), int(cols), int(deg)
    if deg > cols:
        raise SystemExit(f"{spec}: degree {deg} exceeds {cols} columns")
    rng = np.random.default_rng(seed)
    if kind == "powerlaw":
        # Zipf-ish row lengths, rescaled to the requested mean and clamped to cols.
        raw = rng.pareto(1.5, size=rows) + 1.0
        lens = np.maximum(1, np.rint(raw * deg / raw.mean())).astype(np.int64)
        lens = np.minimum(lens, cols)
    else:
        lens = np.full(rows, deg, dtype=np.int64)
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
        elif kind == "blocked":
            block = (i // 8) * d
            start = block % max(cols - d + 1, 1)
            crd[lo:hi] = np.arange(start, start + d)
        else:  # uniform, powerlaw
            crd[lo:hi] = np.sort(rng.choice(cols, size=d, replace=False))
    val = rng.standard_normal(total).astype(np.float32)
    return rows, cols, total, indptr.astype(np.int32), crd.astype(np.int32), val


def load_bin(path):
    with open(path, "rb") as f:
        M, J, nnz = np.frombuffer(f.read(24), dtype=np.int64)
        pos = np.frombuffer(f.read(4 * (int(M) + 1)), dtype=np.int32).copy()
        crd = np.frombuffer(f.read(4 * int(nnz)), dtype=np.int32).copy()
        val = np.frombuffer(f.read(4 * int(nnz)), dtype=np.float32).copy()
    return int(M), int(J), int(nnz), pos, crd, val


def calibrate_reps(call, target_s, cap=4096):
    """Back-to-back calls needed for one timed region to clear `target_s`.

    The cells this kernel most wants to win -- short rows, narrow k -- are also the
    fastest: gcn__cora at k=8 is a 20 microsecond call.  A 20us region cannot be
    A/B-tested one call at a time on a laptop, because it is shorter than the OpenMP
    thread-wake and scheduling noise it sits inside; min-over-rounds never finds a
    clean draw and the A/A control blows out past 1.4, which is what happened on the
    first attempt at this grid.  Timing a batch and dividing lifts the region above
    that floor.  Both arms get the same batch size, so the comparison stays fair.  What
    it changes is that the second call onward finds C and B warm, i.e. this measures
    the steady state rather than a cold first touch -- the right regime for comparing
    two row kernels, and stated here because it is a choice and not a detail.
    """
    reps = 1
    while reps < cap:
        t0 = time.perf_counter()
        for _ in range(reps):
            call()
        dt = time.perf_counter() - t0
        if dt >= target_s:
            return reps
        nxt = int(reps * target_s / max(dt, 1e-9) * 1.3) + 1
        reps = min(cap, max(reps * 2, nxt))
    return cap


def timed(specs, rounds, reps=1, seed=0):
    """Min per arm over `rounds`, arms in a fresh permutation each round.

    `specs` entries are (setup, call): setup runs before the clock starts, and the
    whole batch of `reps` calls runs under one arm, which is what makes batching safe
    here -- the kernel reads its arm from the environment on every call.
    """
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
            dt = (time.perf_counter() - t0) / reps
            best[j] = min(best[j], dt)
    return best


def require_hook(SO):
    """Refuse to run against a build where the A/B hook is compiled out.

    Without ``-DSCORCH_TUNE_HOOKS`` the getenv is not in the binary, both arms take the
    same path, and this harness reports a tidy ~1.000 with tight controls -- a result
    shaped exactly like "the change does nothing".  An earlier version of this file
    carried a comment promising this check and did not implement it, and a run against
    the wrong tree duly produced 0.98-1.03 across 400 cells before anyone noticed.  So
    the check is real now, and it fails closed.

    Two ways to establish it, in order of directness: a module attribute the extension
    sets under the same #ifdef, and failing that the presence of the string literal
    ``SCORCH_SPMM_WORKSPACE`` in the shared object -- if the literal is absent, the
    getenv that names it was not compiled.
    """
    flagged = getattr(SO, "spmm_tune_hooks", None)
    if flagged is True:
        return
    if flagged is False:
        raise SystemExit(
            "this extension reports spmm_tune_hooks=False: rebuild with "
            "SCORCH_BUILD_TUNE_HOOKS=1, or both arms are the same code")
    so = getattr(SO, "__file__", None)
    if not so or not os.path.exists(so):
        raise SystemExit("cannot locate the scorch_ops shared object to verify the hook")
    with open(so, "rb") as f:
        if b"SCORCH_SPMM_WORKSPACE" not in f.read():
            raise SystemExit(
                f"{so} was built without -DSCORCH_TUNE_HOOKS: the workspace/NEON hook "
                "is not in the binary, so both arms would run the same path and every "
                "ratio would be identical code against itself. Rebuild that tree with "
                "SCORCH_BUILD_TUNE_HOOKS=1, or point PYTHONPATH at the tree that has it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtxdir", default=".")
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--ks", default="8,16,32,64,128")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--bytes-cap", type=float, default=3e9)
    ap.add_argument("--batch-ms", type=float, default=3.0,
                    help="minimum duration of one timed region; cells "
                         "faster than this are batched up to it")
    args = ap.parse_args()

    import scorch_ops as SO

    if not hasattr(SO, "spmm_csr_float_v2"):
        raise SystemExit("this extension has no drop-in SpMM")
    require_hook(SO)
    # The 2-nonzero unroll gets its own arm when the binary can turn it off, so the
    # unroll is priced against its own predecessor in the same process rather than
    # across two builds -- cross-run drift on this host is larger than the effect.
    with open(SO.__file__, "rb") as f:
        has_nodual = b"SCORCH_SPMM_NEON_NODUAL" in f.read()
    os.environ["SCORCH_SPMM_NEON_NODUAL"] = "0"
    os.environ["SCORCH_SPMM_WORKSPACE"] = "1"
    probe_np = np.float32 if args.dtype == "float32" else np.float64
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    symbol = (SO.spmm_csr_float_v2 if args.dtype == "float32"
              else SO.spmm_csr_double_v2)

    nt = torch.get_num_threads()
    print(f"dtype={args.dtype}  threads={nt}  rounds={args.rounds}")
    print(f"batch floor {args.batch_ms:.1f} ms per timed region")
    print("2-nonzero unroll arm: " + ("present" if has_nodual else
          "ABSENT (binary predates SCORCH_SPMM_NEON_NODUAL)"))
    if has_nodual:
        print(f"\n{'matrix':<26}{'k':>5}{'rows':>9}{'deg':>7}"
              f"{'ws ms':>10}{'dual ms':>10}{'1nnz ms':>10}"
              f"{'dual/ws':>9}{'1nnz/ws':>9}{'dual/1nnz':>11}{'A/A':>7}"
              f"{'reps':>6}{'relerr':>10}")
    else:
        print(f"\n{'matrix':<26}{'k':>5}{'rows':>9}{'deg':>7}"
              f"{'ws ms':>10}{'neon ms':>10}{'neon/ws':>9}{'A/A':>7}"
              f"{'reps':>6}{'relerr':>10}")
    rows = []
    for name in args.matrices.split(","):
        if ":" in name:                      # a synthetic spec, not a file
            M, J, nnz, pos, crd, val = synth(name)
        else:
            path = os.path.join(args.mtxdir,
                                name if name.endswith(".bin") else name + ".bin")
            M, J, nnz, pos, crd, val = load_bin(path)
        tp = torch.from_numpy(pos)
        tc = torch.from_numpy(crd)
        tv = torch.from_numpy(val.astype(probe_np))
        Aidx = [[], [tp, tc]]
        itemsize = 4 if args.dtype == "float32" else 8
        for k in (int(x) for x in args.ks.split(",")):
            if (M * k + J * k) * itemsize > args.bytes_cap:
                continue
            B = torch.randn(J, k, dtype=torch_dtype)   # J rows: A is M x J
            shapes = ([M, k], [M, J], Aidx, tv, [J, k], [[], []], B.reshape(-1))

            def call():
                return symbol(*shapes, nthreads_override=nt, atparallel=True)

            def set_ws():
                os.environ["SCORCH_SPMM_WORKSPACE"] = "1"

            def set_neon():
                os.environ["SCORCH_SPMM_WORKSPACE"] = "0"
                os.environ["SCORCH_SPMM_NEON_NODUAL"] = "0"

            def set_nodual():
                os.environ["SCORCH_SPMM_WORKSPACE"] = "0"
                os.environ["SCORCH_SPMM_NEON_NODUAL"] = "1"

            set_ws()
            ref = call().storage.value.reshape(M, k).clone()
            set_neon()
            got = call().storage.value.reshape(M, k)
            scale = max(ref.abs().max().item(), 1e-30)
            relerr = (got - ref).abs().max().item() / scale
            tol = 1e-4 if args.dtype == "float32" else 1e-12
            if relerr > tol:
                raise SystemExit(f"{name}@{k}: arms disagree, relerr {relerr:.3e}")
            if has_nodual:
                # Check the third arm too. It is the kernel as it was before the
                # 2-nonzero unroll, and a harness that times a path it never
                # validated is one silent wrong answer away from a fast result.
                set_nodual()
                gnd = call().storage.value.reshape(M, k)
                rnd = (gnd - ref).abs().max().item() / scale
                if rnd > tol:
                    raise SystemExit(
                        f"{name}@{k}: the 1-nnz arm disagrees, relerr {rnd:.3e}")

            set_neon()
            reps = calibrate_reps(call, args.batch_ms * 1e-3)
            specs = [(set_ws, call), (set_neon, call), (set_neon, call)]
            if has_nodual:
                specs.append((set_nodual, call))
            t = timed(specs, args.rounds, reps=reps)
            t_ws, t_neon = t[0] * 1e3, t[1] * 1e3
            aa = max(t[1], t[2]) / min(t[1], t[2])
            if has_nodual:
                t_nd = t[3] * 1e3
                rows.append((name, k, t_ws / t_neon, aa, t_ws / t_nd, t_nd / t_neon))
                print(f"{name:<26}{k:>5}{M:>9}{nnz/M:>7.1f}"
                      f"{t_ws:>10.4f}{t_neon:>10.4f}{t_nd:>10.4f}"
                      f"{t_ws/t_neon:>9.3f}{t_ws/t_nd:>9.3f}{t_nd/t_neon:>11.3f}"
                      f"{aa:>7.3f}{reps:>6}{relerr:>10.2e}", flush=True)
            else:
                rows.append((name, k, t_ws / t_neon, aa))
                print(f"{name:<26}{k:>5}{M:>9}{nnz/M:>7.1f}"
                      f"{t_ws:>10.4f}{t_neon:>10.4f}{t_ws/t_neon:>9.3f}{aa:>7.3f}"
                      f"{reps:>6}{relerr:>10.2e}", flush=True)

    if not rows:
        return
    ratios = [r[2] for r in rows]
    geo = float(np.exp(np.mean(np.log(ratios))))
    aas = sorted(r[3] for r in rows)
    print("\n" + "=" * 78)
    print(f"n={len(rows)} cells   neon/ws geomean {geo:.3f}  "
          f"min {min(ratios):.3f}  max {max(ratios):.3f}  "
          f"cells below 1.0: {sum(1 for x in ratios if x < 1.0)}")
    print(f"  A/A control on the neon arm: {aas[0]:.3f}-{aas[-1]:.3f}")
    if rows and len(rows[0]) > 4:
        def geo_of(idx):
            v = [r[idx] for r in rows]
            return float(np.exp(np.mean(np.log(v)))), min(v), max(v)
        g, lo, hi = geo_of(4)
        print(f"  1-nnz kernel vs workspace: geomean {g:.3f}  min {lo:.3f}  max {hi:.3f}")
        g, lo, hi = geo_of(5)
        below = sum(1 for r in rows if r[5] < 1.0)
        print(f"  the 2-nnz unroll's own effect (1nnz/dual): geomean {g:.3f}  "
              f"min {lo:.3f}  max {hi:.3f}  cells where the unroll loses: {below}")
        aa_hi = max(r[3] for r in rows)
        print(f"  read that against the A/A ceiling of {aa_hi:.3f}: an effect inside "
              f"it is not an effect")


if __name__ == "__main__":
    main()
