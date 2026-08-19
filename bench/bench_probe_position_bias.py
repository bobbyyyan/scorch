"""Does the tiling probe's candidate ORDER change which kernel it picks?

`tiling.maybe_dispatch` times each candidate to completion before starting the
next, and the caller's baseline is always `cands[0]`. So the baseline is timed on
the coldest machine of the run -- clocks still ramping, OpenMP team not yet
settled -- and every tiled candidate after it is timed on a machine the earlier
candidates warmed. If that matters, the routine whose entire purpose is to
guarantee "never slower than the baseline" is systematically tilted against the
baseline.

The instrument is a position-swapped A/A control: the baseline is entered into
the candidate list TWICE, first and last, calling the identical function with the
identical arguments. Under a position-free timing scheme the two entries measure
the same number. Whatever gap opens between them is position, measured directly,
with no model of turbo ramp or thread settling in between.

Three schemes are compared on the same list:
  seq     what ships -- per candidate: one warmup, then the min of 2 timed reps
  seq_rev the same, candidate list reversed (the bias should reverse with it)
  inter   warm every candidate, then `rounds` rotated rounds, min per candidate

What makes a finding actionable is not the A/A ratio on its own but a
*position-induced pick*: a tiled candidate that beats baseline@first (so
production routes to it) while losing to baseline@last (so it does not actually
beat the baseline). That is a shipped regression the guarantee was supposed to
exclude.

Usage:
  python bench/bench_probe_position_bias.py [--Ns 512,1024] [--rounds 2]
                                            [--reps 3] [--matrices a,b,c]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Callable, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import scorch  # noqa: E402
from scorch.stensor import STensor  # noqa: E402
from scorch import tiling  # noqa: E402

_ops = tiling._ops

SS_DIRS = [
    os.path.expanduser("~/.cache/scorch/ss_full_envelope"),
    os.path.expanduser("~/.cache/scorch/ss_redwood_panel"),
    os.path.expanduser("~/.cache/scorch_suitesparse"),
]


# --------------------------------------------------------------------------
# operands
# --------------------------------------------------------------------------
def synthetic_scatter(rows: int, degree: int, cols: int, seed: int) -> "torch.Tensor":
    """Uniform-random column choices: maximal scatter, so the locality gate opens.

    Uniform-random is the adversarial case for tile-j width (see
    tilej_width_never_tuned) but that is irrelevant here -- this harness asks
    about candidate ORDER, and every candidate sees the same matrix.
    """
    g = np.random.default_rng(seed)
    indptr = np.arange(rows + 1, dtype=np.int64) * degree
    # Distinct, sorted columns within each row: CSR requires sorted-within-parent,
    # and duplicate coordinates are not a legal compressed level either.
    indices = np.empty((rows, degree), dtype=np.int64)
    for r in range(rows):
        indices[r] = np.sort(g.choice(cols, size=degree, replace=False))
    indices = indices.reshape(-1)
    values = g.standard_normal(rows * degree).astype(np.float32)
    return torch.sparse_csr_tensor(
        torch.from_numpy(indptr), torch.from_numpy(indices),
        torch.from_numpy(values), size=(rows, cols),
    )


def load_bin(path: str):
    """The flat CSR dump used by the SpMM-vs-MKL harnesses: three int64 header
    words (M, J, nnz), then int32 indptr, int32 indices, float32 values."""
    with open(path, "rb") as f:
        M, J, nnz = np.frombuffer(f.read(24), dtype=np.int64)
        pos = np.frombuffer(f.read(4 * (int(M) + 1)), dtype=np.int32).copy()
        crd = np.frombuffer(f.read(4 * int(nnz)), dtype=np.int32).copy()
        val = np.frombuffer(f.read(4 * int(nnz)), dtype=np.float32).copy()
    return torch.sparse_csr_tensor(
        torch.from_numpy(pos.astype(np.int64)),
        torch.from_numpy(crd.astype(np.int64)),
        torch.from_numpy(val), size=(int(M), int(J)),
    )


def load_mtx(name: str):
    from scipy.io import mmread
    for d in SS_DIRS:
        p = os.path.join(d, name, f"{name}.mtx")
        if not os.path.exists(p):
            p = os.path.join(d, f"{name}.mtx")
        if os.path.exists(p):
            m = mmread(p).tocsr().astype(np.float32)
            return torch.sparse_csr_tensor(
                torch.from_numpy(m.indptr.astype(np.int64)),
                torch.from_numpy(m.indices.astype(np.int64)),
                torch.from_numpy(m.data), size=m.shape,
            )
    raise FileNotFoundError(name)


# --------------------------------------------------------------------------
# the candidate list, built exactly as maybe_dispatch builds it
# --------------------------------------------------------------------------
def build_candidates(a, b, nthreads) -> Tuple[List[Tuple[str, object, Callable]], dict]:
    J, N, M = int(a.shape[1]), int(b.shape[1]), int(a.shape[0])
    idx = a._native_mode_indices()
    nnz = int(idx[1][1].numel())
    C = tiling.query_llc()
    result_shape = [M, N]
    nt = nthreads if nthreads is not None else -1
    Jc = tiling._panel_width(N, C)

    info = {
        "M": M, "J": J, "N": N, "nnz": nnz, "C": C, "Jc": Jc,
        "eligible": tiling._eligible(J, nnz, N, C),
        "scattered": tiling._scattered(a, J),
    }

    def baseline():
        return _ops.spmm_csr_float_v2(
            result_shape, a.shape, a._native_mode_indices(), a.values,
            b.shape, b._native_mode_indices(), b.values, nthreads, False,
        )

    cands: list = [("base@first", None, baseline)]
    for jc in tiling._jc_ladder(Jc):
        cands.append(("tilej", jc, lambda jc=jc: _ops.spmm_csr_float_tilej(
            *tiling._tilej_args(a, b, result_shape, jc, nt))))
    if tiling._HAS_TILEIJK and N >= tiling._NIJK_MIN:
        Nc, Jc_ijk = tiling._ijk_params(N, M, J, C)
        if Nc < N:
            info["ijk"] = (Nc, Jc_ijk)
            cands.append(("tileijk", (Nc, Jc_ijk),
                          lambda: _ops.spmm_csr_float_tileijk(
                              *tiling._tileijk_args(
                                  a, b, result_shape, Nc, Jc_ijk, nt))))
    # The A/A twin: identical function, identical arguments, last position.
    cands.append(("base@last", None, baseline))
    return cands, info


# --------------------------------------------------------------------------
# timing schemes
# --------------------------------------------------------------------------
def time_sequential(fns: List[Callable], reps: int = 2) -> List[float]:
    """What ships: each candidate warmed and timed to completion, in order."""
    out = []
    for fn in fns:
        fn()
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        out.append(best)
    return out


def time_interleaved(fns: List[Callable], rounds: int = 2) -> List[float]:
    """Warm every candidate, then rotate the starting point each round."""
    n = len(fns)
    best = [float("inf")] * n
    for fn in fns:
        fn()
    for r in range(rounds):
        for i in range(n):
            j = (i + r) % n
            t0 = time.perf_counter()
            fns[j]()
            dt = time.perf_counter() - t0
            if dt < best[j]:
                best[j] = dt
    return best


# --------------------------------------------------------------------------
def run_cell(label, a_t, N, nthreads, rounds, reps):
    a = STensor.from_torch(a_t)
    b = torch.randn(int(a.shape[1]), N, dtype=torch.float32)
    bst = STensor.from_torch(b)
    cands, info = build_candidates(a, bst, nthreads)
    if not info["eligible"] or not info["scattered"]:
        print(f"{label:<22}{N:>6}   gate closed "
              f"(eligible={info['eligible']} scattered={info['scattered']})")
        return None
    fns = [fn for _, _, fn in cands]
    names = [f"{k}{'' if p is None else '@' + str(p)}" for k, p, _ in cands]

    schemes = {
        "seq": lambda: time_sequential(fns, reps),
        "seq_rev": lambda: [t for t in reversed(time_sequential(
            list(reversed(fns)), reps))],
        "inter": lambda: time_interleaved(fns, rounds),
    }
    rows = {}
    for sname, run in schemes.items():
        rows[sname] = run()

    first_i, last_i = 0, len(cands) - 1
    print(f"\n### {label}  M={info['M']} J={info['J']} nnz={info['nnz']} "
          f"N={N} Jc={info['Jc']} C={info['C'] >> 20}MiB")
    hdr = f"{'candidate':<20}" + "".join(f"{s:>12}" for s in schemes)
    print(hdr)
    for i, nm in enumerate(names):
        print(f"{nm:<20}" + "".join(f"{rows[s][i] * 1e6:12.1f}" for s in schemes))
    print(f"{'A/A last/first':<20}"
          + "".join(f"{rows[s][last_i] / rows[s][first_i]:12.3f}" for s in schemes))

    out = {"label": label, "N": N, "info": info}
    for s in schemes:
        t = rows[s]
        aa = t[last_i] / t[first_i]
        tiled = [(t[i], names[i]) for i in range(1, last_i)]
        bt, bn = min(tiled)
        # production's comparison: tiled vs baseline@first
        picks_tiled = bt < t[first_i]
        # the honest comparison: does that pick also beat the OTHER baseline copy?
        survives = bt < t[last_i]
        out[s] = {
            "aa": aa, "pick": bn if picks_tiled else "base",
            "position_induced": bool(picks_tiled and not survives),
            "ratio_vs_first": t[first_i] / bt,
            "ratio_vs_last": t[last_i] / bt,
        }
    # Which way each rule votes, from the interleaved (position-free) times: the
    # old rule ships any tiled candidate faster than the baseline as timed; the new
    # one requires the win to exceed the gap between the two identical baseline
    # arms. A disagreement is a cell where the old selector shipped a margin it
    # could not distinguish from its own noise.
    ti = rows["inter"]
    t_first_i, t_last_i = ti[first_i], ti[last_i]
    best_tiled_t = min(ti[i] for i in range(1, last_i))
    old_rule = best_tiled_t < t_first_i
    new_rule = tiling._clears_noise(best_tiled_t, t_first_i, t_last_i)
    floor = max(t_first_i, t_last_i) / min(t_first_i, t_last_i)
    margin = min(t_first_i, t_last_i) / best_tiled_t
    out["rules"] = {"old": old_rule, "new": new_rule,
                    "floor": floor, "margin": margin}
    print(f"  floor {floor:.3f}  margin {margin:.3f}  "
          f"old rule={'tiled' if old_rule else 'base'}  "
          f"new rule={'tiled' if new_rule else 'base'}"
          + ("   <-- RULES DISAGREE" if old_rule != new_rule else ""))

    flag = ""
    if out["seq"]["position_induced"]:
        flag = "  <-- POSITION-INDUCED PICK under the shipped scheme"
    print(f"  seq pick={out['seq']['pick']:<14} "
          f"vs first {out['seq']['ratio_vs_first']:.3f}x  "
          f"vs last {out['seq']['ratio_vs_last']:.3f}x{flag}")
    print(f"  inter pick={out['inter']['pick']:<12} "
          f"vs first {out['inter']['ratio_vs_first']:.3f}x  "
          f"vs last {out['inter']['ratio_vs_last']:.3f}x")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", default="512,1024")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--matrices", default="")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mtxdir", default="",
                    help="directory of flat .bin CSR dumps; --matrices names them")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    nthreads = torch.get_num_threads()
    print(f"host threads={nthreads}  LLC={tiling.query_llc() >> 20}MiB  "
          f"rounds={args.rounds} reps={args.reps}")

    work = []
    if args.matrices and args.mtxdir:
        for nm in args.matrices.split(","):
            p = os.path.join(args.mtxdir, nm if nm.endswith(".bin") else nm + ".bin")
            try:
                work.append((nm.replace(".bin", ""), load_bin(p)))
            except Exception as e:  # noqa: BLE001
                print(f"skip {nm}: {e}")
    elif args.matrices:
        for nm in args.matrices.split(","):
            try:
                work.append((nm, load_mtx(nm)))
            except Exception as e:  # noqa: BLE001
                print(f"skip {nm}: {e}")
    else:
        # Synthetic scatter across a range of CALL DURATIONS, because a fixed
        # turbo/thread-settling cost is a larger share of a short call.
        for rows, deg in ((20_000, 200), (60_000, 200), (200_000, 100)):
            work.append((f"scatter{deg}-{rows // 1000}k",
                         synthetic_scatter(rows, deg, rows, seed=7)))

    results = []
    for label, a_t in work:
        for N in (int(x) for x in args.Ns.split(",")):
            r = run_cell(label, a_t, N, nthreads, args.rounds, args.reps)
            if r:
                results.append(r)

    if not results:
        print("\nno cells ran")
        return
    print("\n" + "=" * 78)
    print(f"{'cell':<28}{'aa_seq':>9}{'aa_rev':>9}{'aa_int':>9}"
          f"{'seq pick':>16}{'inter pick':>16}")
    for r in results:
        print(f"{r['label'] + '/' + str(r['N']):<28}"
              f"{r['seq']['aa']:9.3f}{r['seq_rev']['aa']:9.3f}"
              f"{r['inter']['aa']:9.3f}{r['seq']['pick']:>16}"
              f"{r['inter']['pick']:>16}")
    print(f"\n{'cell':<28}{'floor':>8}{'margin':>8}{'old':>7}{'new':>7}")
    for r in results:
        ru = r["rules"]
        print(f"{r['label'] + '/' + str(r['N']):<28}{ru['floor']:8.3f}"
              f"{ru['margin']:8.3f}{'tiled' if ru['old'] else 'base':>7}"
              f"{'tiled' if ru['new'] else 'base':>7}"
              + ("   DISAGREE" if ru['old'] != ru['new'] else ""))
    n_dis = sum(1 for r in results if r["rules"]["old"] != r["rules"]["new"])
    marg = sorted(r["rules"]["margin"] for r in results)
    print(f"\nrules disagree: {n_dis}/{len(results)}   "
          f"margin min {marg[0]:.3f} median {marg[len(marg) // 2]:.3f} "
          f"max {marg[-1]:.3f}")
    n_ind = sum(1 for r in results if r["seq"]["position_induced"])
    n_flip = sum(1 for r in results if r["seq"]["pick"] != r["inter"]["pick"])
    print(f"\nposition-induced picks under the shipped scheme: "
          f"{n_ind}/{len(results)}")
    print(f"winner differs seq vs interleaved:                {n_flip}/{len(results)}")
    aa_seq = [r["seq"]["aa"] for r in results]
    aa_int = [r["inter"]["aa"] for r in results]
    print(f"A/A spread  seq   {min(aa_seq):.3f}-{max(aa_seq):.3f}   "
          f"inter {min(aa_int):.3f}-{max(aa_int):.3f}")


if __name__ == "__main__":
    main()
