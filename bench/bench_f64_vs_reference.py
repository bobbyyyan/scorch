"""float64 CSR x dense: the new register-resident kernel against MKL and against the
reference kernel float64 used to resolve to.

Three arms plus a control, all in ONE process against ONE binary, so the new-vs-old
comparison carries no build or process variance:

  torch  torch.sparse.mm on a float64 CSR tensor. On x86 that is
         mkl_sparse_d_mm; on Apple silicon there is no MKL and it is
         torch's own sparse path, so the column is labelled for what it
         actually calls rather than for what it calls on one host.
  ref    spmm_csr_double  -- spmm_csr_typed_core<double>, what float64 resolved to.
         Dispatch actually named `prebuilt_spmm_csr_f64` first, but that symbol is
         `spmm_csr_typed<double>` too, with the same default tile_size, and
         spmm_csr_typed_core ignores tile_size -- so this arm is the same machine
         code the old route ran, not a stand-in for it.
  new    spmm_csr_double_v2 -- spmm_csr_v2_core<double>
  aa     a second entry of `new`, at the other end of the arm list. It is the
         control ONLY -- `new`'s reported time is its first entry's minimum, so all
         three arms are estimated from the same number of draws.

Arms are drawn in a fresh random permutation every round, so a slow neighbour is
variance the A/A control can see rather than a fixed per-arm offset.

The `new` arm's thread count and launch mode come from ``ops._composition_hints``,
the function production itself calls -- not from a value written here. An earlier
version of this harness passed ``atparallel=False`` while production ships
``atparallel=True``; on a hybrid P+E part those launch through different pools with
different core counts, so the grid was timing a configuration nobody runs. A harness
that restates a production policy agrees with it only until the policy moves.
"""

import argparse, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import scorch_ops as SO  # noqa: E402
from scorch import ops as scorch_ops  # noqa: E402
from scorch.prebuilt_kernels import resolve_prebuilt_matmul  # noqa: E402
from scorch.stensor import STensor  # noqa: E402


def production_hints(csr):
    """The (nthreads_override, atparallel) production passes for this operand.

    Asked of ``ops._composition_hints`` rather than reproduced, so this harness
    cannot drift from the dispatch it is supposed to be measuring. ``None`` means
    the hints are off, and production then calls the kernel with no override at all
    -- which is the kernel's own default, so pass its default here too.
    """
    a = STensor.from_torch(csr)
    # Both operands must be STensors: resolution reads `.format` off each.
    b = STensor.from_torch(torch.zeros(csr.size(1), 1, dtype=csr.dtype))
    resolved = resolve_prebuilt_matmul(a, b, output_format="dd")
    if resolved is None:
        raise SystemExit(
            "float64 CSR x dense resolves no prebuilt kernel; this harness would be "
            "timing a symbol production does not call"
        )
    nthreads, atparallel = scorch_ops._composition_hints(resolved)
    return resolved.symbol_name, (-1 if nthreads is None else int(nthreads)), atparallel


def load_bin(path):
    with open(path, "rb") as f:
        M, J, nnz = np.frombuffer(f.read(24), dtype=np.int64)
        pos = np.frombuffer(f.read(4 * (int(M) + 1)), dtype=np.int32).copy()
        crd = np.frombuffer(f.read(4 * int(nnz)), dtype=np.int32).copy()
        val = np.frombuffer(f.read(4 * int(nnz)), dtype=np.float32).copy()
    return int(M), int(J), int(nnz), pos, crd, val


def timed(specs, rounds, seed=0):
    n = len(specs)
    best = [float("inf")] * n
    rng = random.Random(seed)
    for fn in specs:
        fn()
    for _ in range(rounds):
        for j in rng.sample(range(n), n):
            t0 = time.perf_counter()
            specs[j]()
            dt = time.perf_counter() - t0
            if dt < best[j]:
                best[j] = dt
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtxdir", required=True)
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--ks", default="8,16,32,64,128")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--bytes-cap", type=float, default=3e9)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    nt = torch.get_num_threads()
    print(f"threads={nt}  rounds={args.rounds}  dtype=float64")
    print(
        "the 'torch' arm is torch.sparse.mm: mkl_sparse_d_mm on x86, torch's own "
        "sparse path on Apple silicon -- there is no MKL to compare against there, "
        "so on ARM this column is not a parity claim"
    )
    hint_symbol = hint_nt = hint_atp = None
    print(
        f"\n{'matrix':<20}{'k':>5}{'rows':>9}{'nnz':>11}"
        f"{'torch ms':>10}{'ref ms':>10}{'new ms':>10}"
        f"{'new/tch':>9}{'ref/tch':>9}{'new/ref':>9}{'A/A':>7}"
    )
    rows = []
    for name in args.matrices.split(","):
        p = os.path.join(args.mtxdir, name if name.endswith(".bin") else name + ".bin")
        M, J, nnz, pos, crd, val = load_bin(p)
        tp64 = torch.from_numpy(pos.astype(np.int64))
        tc64 = torch.from_numpy(crd.astype(np.int64))
        tv64 = torch.from_numpy(val.astype(np.float64))
        csr = torch.sparse_csr_tensor(tp64, tc64, tv64, size=(M, J))
        tp = torch.from_numpy(pos)
        tc = torch.from_numpy(crd)
        Aidx = [[], [tp, tc]]
        for k in (int(x) for x in args.ks.split(",")):
            if (M * k + J * k) * 8 > args.bytes_cap:
                continue
            B = torch.randn(J, k, dtype=torch.float64)
            shapes = ([M, k], [M, J], Aidx, tv64, [J, k], [[], []], B.reshape(-1))
            mkl = lambda: torch.sparse.mm(csr, B)
            ref = lambda: SO.spmm_csr_double(*shapes)
            if hint_symbol is None:
                hint_symbol, hint_nt, hint_atp = production_hints(csr)
                print(
                    f"production dispatch: {hint_symbol}"
                    f"(nthreads_override={hint_nt}, atparallel={hint_atp})"
                )
                if hint_symbol != "spmm_csr_double_v2":
                    raise SystemExit(
                        f"production resolves {hint_symbol}, not the symbol this "
                        f"harness times"
                    )
            new = lambda: SO.spmm_csr_double_v2(
                *shapes, nthreads_override=hint_nt, atparallel=hint_atp
            )
            t = timed([mkl, ref, new, new], args.rounds)
            # Every reported time is ONE arm's min over the rounds. `new` is entered
            # twice, but the second entry is only the A/A control -- reporting
            # min(t[2], t[3]) for `new` would give it twice as many draws as mkl and
            # ref, and a minimum over more draws is biased low. Measured: that bias
            # was +0.7% on a host where both arms run identical machine code.
            t_mkl, t_ref = t[0] * 1e3, t[1] * 1e3
            t_new = t[2] * 1e3
            aa = max(t[2], t[3]) / min(t[2], t[3])
            rows.append((name, k, t_mkl / t_new, t_mkl / t_ref, t_ref / t_new, aa))
            print(
                f"{name:<20}{k:>5}{M:>9}{nnz:>11}"
                f"{t_mkl:>10.3f}{t_ref:>10.3f}{t_new:>10.3f}"
                f"{t_mkl/t_new:>9.3f}{t_mkl/t_ref:>9.3f}{t_ref/t_new:>9.3f}{aa:>7.3f}",
                flush=True,
            )
    if not rows:
        return

    def geo(v):
        return float(np.exp(np.mean(np.log(v))))

    nm = [r[2] for r in rows]
    rm = [r[3] for r in rows]
    nr = [r[4] for r in rows]
    aas = sorted(r[5] for r in rows)
    print("\n" + "=" * 78)
    print(f"n={len(rows)} cells")
    print(
        f"  NEW vs torch: geomean {geo(nm):.3f}  min {min(nm):.3f}  max {max(nm):.3f}  "
        f"cells below parity: {sum(1 for x in nm if x < 1.0)}"
    )
    print(
        f"  ref vs torch: geomean {geo(rm):.3f}  min {min(rm):.3f}  max {max(rm):.3f}  "
        f"cells below parity: {sum(1 for x in rm if x < 1.0)}"
    )
    print(f"  NEW vs ref: geomean {geo(nr):.3f}  min {min(nr):.3f}  max {max(nr):.3f}")
    print(f"  A/A control on the new kernel: {aas[0]:.3f}-{aas[-1]:.3f}")


if __name__ == "__main__":
    main()
