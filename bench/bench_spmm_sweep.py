r"""bench_spmm_sweep.py -- scorch's SpMM against MKL over a whole matrix collection
and a ladder of free-dimension widths.

This is a *screening* harness, not a verdict harness. The grids we have run so far
answer "how does scorch do on the 20-odd matrices we chose", which cannot distinguish
"scorch wins" from "we chose well". This one runs the SuiteSparse collection and the
DLMC deep-learning pruning collection end to end, at every k from 1 to 512, and its
output is a per-cell table whose point is to *find* the cells below MKL parity so a
second, slower pass can confirm or refute each one.

Arms, all in one process against one binary:

  mkl32  torch.sparse.mm on a float32 CSR with int32 index arrays. This is MKL's
         own preferred layout on x86 and the faster of the two MKL arms.
  mkl64  the same with int64 indices -- what a user gets from
         torch.sparse_csr_tensor without thinking about it.
  sc     scorch.matmul: the whole production path, dispatch and selector included.
         That is the honest comparison for the question "is the library slower",
         and time_dict["eval_time"] separates the kernel from the Python on top
         of it so a loss can be attributed rather than guessed at.
  aa     a second entry of `sc`, identical code. It is the control: any cell whose
         sc/aa ratio departs from 1 by more than the margin under test has not
         measured anything, and the analyzer screens on it before pooling.

`best MKL` is the *faster* of mkl32 and mkl64 at every cell, which is the choice
that makes scorch look worst.

``--dtype float64`` reruns the whole thing against a different pair of kernels --
scorch's ``spmm_csr_double_v2`` against ``mkl_sparse_d_mm`` -- because float64 was
the half of this comparison scorch used to lose, and a claim about one dtype says
nothing about the other.

Scheduling. A whole-collection sweep cannot afford a fixed rep count: the same
number that gives a stable reading on a 20 us cell would spend an hour on a
100 M-nonzero one. Two mechanisms bound it.

  * A per-cell work cap. A cell runs only if nnz*k is under --work-cap, so the k
    ladder truncates on large matrices instead of the sweep stalling on them. Which
    (matrix, k) pairs this drops is written to the CSV as a `skipped` row with the
    reason -- a bounded sweep that does not say what it bounded reads as a complete
    one.
  * Adaptive reps and batching. One untimed call measures the cell. A DLMC weight
    at k=4 is a 30 us call -- shorter than OpenMP thread-wake noise, so timing it
    once measures the pool and not the kernel. So each sample times a *batch* of
    back-to-back calls big enough to clear --batch-ms and divides, and the number of
    batches is then chosen to spend about --target-ms per arm, clamped to
    [--min-reps, --max-reps]. Batching does change what is measured -- the first
    call's cold first-touch becomes steady state -- but it does so identically for
    every arm, which is the property the comparison needs. Large cells get batch=1
    and three reps.

Rounds draw the arms in a fresh random permutation, so a slow neighbour -- another
job, a thermal excursion, a page fault -- lands on whichever arm happened to be next
rather than always on the same one, and the A/A control can see it.

Results are appended and flushed per cell, and a rerun skips (key, k) pairs already
present, so a sweep this long can be interrupted, resumed, and extended with more k
values without redoing work.

usage:
  python bench/spmm_corpus.py --suitesparse /scratch/suitesparse --out ss.csv
  python bench/bench_spmm_sweep.py --manifest ss.csv --cache /scratch/bobbyy/csrcache \
      --csv sweep.csv --ks 1,2,4,8,16,32,64,128,256,512
"""

import argparse
import csv
import gc
import math
import os
import random
import statistics
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

import spmm_corpus as corpus  # noqa: E402

# The CSR cache is written once in float32 -- the structure is what costs time to
# parse, and a float64 run recasts the values on load rather than keeping a second
# copy of 18 GB of index arrays on disk.
TDT = torch.float32
NPDT = np.float32
ITEMSIZE = 4.0
import scorch  # noqa: E402
from scorch import tiling  # noqa: E402

FIELDS = [
    "key",
    "family",
    "kind",
    "dtype",
    "primary",
    "rows",
    "cols",
    "nnz",
    "density",
    "mean_row",
    "std_row",
    "max_row",
    "empty_rows",
    "k",
    "mkl32_ms",
    "mkl64_ms",
    "sc_ms",
    "aa_ms",
    "sc_kernel_ms",
    "mkl32_min_ms",
    "mkl64_min_ms",
    "sc_min_ms",
    "aa_min_ms",
    "vs_mkl",
    "vs_mkl_min",
    "vs_mkl_kernel",
    "aa_ratio",
    "route",
    "route_param",
    "reps",
    "batch",
    "settle",
    "relerr_sc",
    "relerr_mkl",
    "ref_rows",
    "status",
    "note",
]


def as_tensor(x):
    """A dense torch tensor from whatever an arm returned."""
    if isinstance(x, torch.Tensor):
        return x.to_dense() if x.layout != torch.strided else x
    return x.to_torch()


def load_or_convert(rec, cache_dir, nnz_cap):
    """Cached canonical CSR for one manifest row, converting on first sight."""
    dest = os.path.join(cache_dir, rec["key"].replace(":", "/") + ".npz")
    if os.path.exists(dest):
        try:
            return corpus.load_cache(dest), None
        except Exception as ex:  # a truncated cache from a killed run
            os.remove(dest)
            note = f"cache reread failed ({type(ex).__name__}), reconverting"
            csr, why = corpus.convert(rec["path"], rec["kind"], dest, nnz_cap)
            return csr, (why or note if csr is None else None)
    return corpus.convert(rec["path"], rec["kind"], dest, nnz_cap)


def torch_csr(csr, idx_dtype, val_dtype):
    return torch.sparse_csr_tensor(
        torch.from_numpy(csr.indptr.astype(idx_dtype)),
        torch.from_numpy(csr.indices.astype(idx_dtype)),
        torch.from_numpy(csr.data.astype(val_dtype)),
        size=csr.shape,
    )


def measure(arms, reps, seed):
    """Interleaved random-order timing. Returns {name: (median_s, min_s)}."""
    names = list(arms)
    samples = {n: [] for n in names}
    rng = random.Random(seed)
    for _ in range(reps):
        for n in rng.sample(names, len(names)):
            t0 = time.perf_counter()
            out = arms[n]()
            dt = time.perf_counter() - t0
            del out
            samples[n].append(dt)
    return {n: (statistics.median(v), min(v)) for n, v in samples.items()}


def open_matrix(rec, a):
    """(csr, None) for a matrix the sweep can time, or (None, reason).

    Both ways a matrix drops out -- outside the nnz window, or unreadable and
    unconvertible -- come back as a reason string, so the caller records why rather
    than the matrix vanishing from the corpus silently.
    """
    if rec["nnz_pred"] > a.nnz_cap or rec["nnz_pred"] < a.nnz_floor:
        return None, (
            f"predicted nnz {rec['nnz_pred']} outside "
            f"[{a.nnz_floor:.0f}, {a.nnz_cap:.0f}]"
        )
    try:
        return load_or_convert(rec, a.cache, a.nnz_cap)
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}"[:200]


def cell_over_cap(nnz, k, M, J, a):
    """Why this (matrix, k) cell is too big to time, or None.

    A bounded sweep that does not say what it bounded reads as a complete one, so the
    caller writes this reason into the CSV instead of dropping the cell quietly.
    """
    if nnz * k > a.work_cap:
        return f"nnz*k = {nnz * k:.3g} over work cap {a.work_cap:.3g}"
    nbytes = ITEMSIZE * k * (J + M)
    if nbytes > a.bytes_cap:
        return f"dense operands {nbytes:.3g} B over cap {a.bytes_cap:.3g}"
    return None


def read_only_cells(path, rows):
    """(cell set, filtered manifest rows) for a --only-cells re-measurement."""
    if not path:
        return None, rows
    only = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            only.add((r["key"], str(int(float(r["k"])))))
    keys = {k for k, _ in only}
    rows = [r for r in rows if r["key"] in keys]
    print(f"restricted to {len(only)} cells over {len(rows)} matrices")
    return only, rows


def convert_corpus(rows, a):
    """Populate the CSR cache and time nothing.

    Parsing Matrix Market text is single-threaded and dominates a whole-collection
    sweep, so it is run first, sharded across processes, and the timing process then
    only ever opens .npz.
    """
    t0 = time.time()
    n_ok = n_bad = 0
    for mi, rec in enumerate(rows):
        if rec["nnz_pred"] > a.nnz_cap or rec["nnz_pred"] < a.nnz_floor:
            continue
        try:
            csr, why = load_or_convert(rec, a.cache, a.nnz_cap)
        except Exception as ex:
            csr, why = None, f"{type(ex).__name__}: {ex}"[:200]
        if csr is None:
            n_bad += 1
            print(f"  [skip] {rec['key']}: {why}")
        else:
            n_ok += 1
        csr = None
        if (mi + 1) % a.progress_every == 0:
            print(
                f"[{mi + 1}/{len(rows)}] {(time.time() - t0) / 60:6.1f} min "
                f"converted={n_ok} skipped={n_bad}"
            )
            sys.stdout.flush()
    print(
        f"CONVERT_DONE converted={n_ok} skipped={n_bad} "
        f"{(time.time() - t0) / 60:.1f} min"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True, help="canonical CSR cache directory")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ks", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument(
        "--nnz-cap",
        type=float,
        default=3e7,
        help="skip matrices with more nonzeros than this",
    )
    ap.add_argument("--nnz-floor", type=float, default=0.0)
    ap.add_argument(
        "--work-cap",
        type=float,
        default=2e9,
        help="skip a (matrix, k) cell whose nnz*k exceeds this",
    )
    ap.add_argument(
        "--bytes-cap",
        type=float,
        default=8e9,
        help="skip a cell whose dense operands would exceed this",
    )
    ap.add_argument(
        "--target-ms",
        type=float,
        default=30.0,
        help="measurement budget per arm per cell",
    )
    ap.add_argument(
        "--batch-ms",
        type=float,
        default=2.0,
        help="calls are batched until a sample takes at least this long",
    )
    ap.add_argument("--max-batch", type=int, default=4096)
    ap.add_argument(
        "--settle",
        type=int,
        default=1,
        help="untimed calls of an arm immediately before timing it, so an "
        "interleaved neighbour cannot leave its thread team parked",
    )
    ap.add_argument(
        "--only-cells",
        default=None,
        help="CSV with key,k columns -- run only those cells. Re-measures a region a "
        "previous pass got wrong without redoing the whole sweep.",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="untimed calls per arm before probing; 1 is not enough on "
        "allocation-dominated cells",
    )
    ap.add_argument("--min-reps", type=int, default=5)
    ap.add_argument("--max-reps", type=int, default=25)
    ap.add_argument("--threads", type=int, default=0, help="0 => torch default")
    ap.add_argument("--level", default=None, help="autotune level; None => default")
    ap.add_argument(
        "--ref-rows",
        type=int,
        default=2048,
        help="rows sampled for the float64 correctness reference",
    )
    ap.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="float64 exercises a different kernel on both sides: "
        "spmm_csr_double_v2 against mkl_sparse_d_mm",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N manifest rows")
    ap.add_argument("--shard", default="", help="i/n -- take every n-th row")
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument(
        "--convert-only",
        action="store_true",
        help="populate the CSR cache and exit, timing nothing. Parsing 41 GB of "
        "Matrix Market text would otherwise happen inside the single timing "
        "process; run this sharded across processes first, then time.",
    )
    a = ap.parse_args()

    global TDT, NPDT, ITEMSIZE
    TDT = torch.float32 if a.dtype == "float32" else torch.float64
    NPDT = np.float32 if a.dtype == "float32" else np.float64
    ITEMSIZE = 4.0 if a.dtype == "float32" else 8.0
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.level:
        tiling.set_autotune(a.level)
    ks = [int(x) for x in a.ks.split(",")]
    os.makedirs(a.cache, exist_ok=True)

    rows = corpus.read_manifest(a.manifest)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        rows = rows[i::n]
    if a.limit:
        rows = rows[: a.limit]

    only, rows = read_only_cells(a.only_cells, rows)

    done = set()
    exists = os.path.exists(a.csv)
    if exists:
        with open(a.csv) as f:
            for r in csv.DictReader(f):
                done.add((r["key"], r["k"]))
    out = open(a.csv, "a", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS)
    if not exists:
        w.writeheader()
        out.flush()

    print(f"host            : {os.uname().nodename}  {os.uname().machine}")
    print(
        f"torch           : {torch.__version__}  MKL={torch.backends.mkl.is_available()}"
    )
    print(f"threads         : {torch.get_num_threads()}")
    print(f"autotune level  : {tiling.get_autotune()}")
    print(f"matrices        : {len(rows)}   k ladder: {ks}")
    print(
        f"caps            : nnz<={a.nnz_cap:.0f}  nnz*k<={a.work_cap:.0f}  "
        f"operand bytes<={a.bytes_cap:.0f}"
    )
    print(f"already in csv  : {len(done)} cells")
    sys.stdout.flush()

    t_start = time.time()
    n_cell = n_skip = n_err = 0

    if a.convert_only:
        convert_corpus(rows, a)
        return

    def emit(rec):
        w.writerow(rec)
        out.flush()

    for mi, rec in enumerate(rows):
        key = rec["key"]
        if all((key, str(k)) in done for k in ks):
            continue
        csr, skip = open_matrix(rec, a)
        if csr is None:
            emit(
                dict.fromkeys(FIELDS, "")
                | dict(
                    key=key,
                    family=rec["family"],
                    kind=rec["kind"],
                    k="",
                    rows=rec["rows"],
                    cols=rec["cols"],
                    nnz=rec["nnz_pred"],
                    status="skipped",
                    note=skip,
                )
            )
            n_skip += 1
            continue

        if NPDT is not np.float32:
            csr = csr.astype(NPDT)
        M, J = csr.shape
        nnz = csr.nnz
        mean_row, std_row, max_row, empty_rows = corpus.row_stats(csr)
        base = dict(
            dtype=a.dtype,
            primary=rec.get("primary", 1),
            key=key,
            family=rec["family"],
            kind=rec["kind"],
            rows=M,
            cols=J,
            nnz=nnz,
            density=nnz / max(1, M * J),
            mean_row=round(mean_row, 3),
            std_row=round(std_row, 3),
            max_row=max_row,
            empty_rows=empty_rows,
        )
        A32 = torch_csr(csr, np.int32, NPDT)
        A64 = torch_csr(csr, np.int64, NPDT)
        A_st = scorch.STensor.from_torch(A64)
        gen = torch.Generator().manual_seed(a.seed + (hash(key) & 0xFFFF))

        for k in ks:
            if (key, str(k)) in done:
                continue
            if only is not None and (key, str(k)) not in only:
                continue
            why = cell_over_cap(nnz, k, M, J, a)
            if why:
                emit(
                    dict.fromkeys(FIELDS, "")
                    | base
                    | dict(k=k, status="skipped", note=why)
                )
                n_skip += 1
                continue
            try:
                B = torch.rand(J, k, generator=gen, dtype=TDT) + 0.5
                B_st = scorch.STensor.from_torch(B)
                td = {}

                def f_mkl32():
                    return torch.sparse.mm(A32, B)

                def f_mkl64():
                    return torch.sparse.mm(A64, B)

                def f_sc():
                    return scorch.matmul(A_st, B_st, time_dict=td)

                arms = {"mkl32": f_mkl32, "mkl64": f_mkl64, "sc": f_sc, "aa": f_sc}

                tiling._decision.clear()
                f_sc()  # prime the selector so its memo can be read back
                route = list(tiling._decision.values())
                # An empty memo means the selector declined to tile, which is the
                # untiled v2 kernel -- the same default the older grid reports.
                route = route[0] if route else ("v2", None)

                # Warm every arm several times before probing. On a cell whose
                # output is large relative to its work -- a 22692x1 matrix with one
                # nonzero, times k=512, so a 46 MB output built from one
                # multiply-add -- the time is first-touch page faulting, and whether
                # a call faults fresh pages or gets a recycled buffer from torch's
                # caching allocator is bimodal. Measured on mkt1 before this warmup
                # existed: the same code timed twice read 1.21 ms and 9.52 ms. Three
                # warm calls per arm put the allocator in the state every later call
                # will see, which removes the bimodality rather than averaging over
                # it.
                for fn in arms.values():
                    for _ in range(a.warmup):
                        fn()

                # ONE batch count, shared by every arm, sized off the slowest so the
                # cell stays affordable. Per-arm batches would give each arm a
                # different number of averaged calls and -- worse -- a different
                # amount of allocator recycling, which biases the ratio toward
                # whichever arm happened to get the deeper batch.
                probe = {}
                for n, fn in arms.items():
                    for _ in range(a.settle):
                        fn()
                    t0 = time.perf_counter()
                    fn()
                    probe[n] = max(time.perf_counter() - t0, 1e-9)
                slow = max(probe.values())
                batch = int(
                    min(a.max_batch, max(1, math.ceil(a.batch_ms * 1e-3 / slow)))
                )
                per_rep_ms = batch * sum(probe.values()) * 1e3
                reps = int(
                    min(
                        a.max_reps,
                        max(a.min_reps, a.target_ms * len(arms) / per_rep_ms),
                    )
                )
                td.clear()
                kern = []
                names = list(arms)
                rng = random.Random(a.seed + k)
                samples = {n: [] for n in names}
                for _ in range(reps):
                    for n in rng.sample(names, len(names)):
                        fn = arms[n]
                        # Settle THIS arm before timing it. Arms are interleaved, so
                        # whatever ran previously may have left this arm's OpenMP team
                        # parked and the next call then pays a wake-up. On mkt1 -- 64
                        # threads across two sockets -- that produced a cell measured
                        # at 11.961 ms which re-measures at 0.132 ms when settled: a
                        # 90x error the A/A control could not see, because it hit both
                        # draws of the same arm equally. One untimed call suffices,
                        # and it costs 1/batch of a sample.
                        for _ in range(a.settle):
                            fn()
                        t0 = time.perf_counter()
                        for _ in range(batch):
                            o = fn()
                        dt = (time.perf_counter() - t0) / batch
                        del o
                        samples[n].append(dt)
                        if n == "sc" and "eval_time" in td:
                            kern.append(td["eval_time"])
                res = {n: (statistics.median(v), min(v)) for n, v in samples.items()}

                # Correctness against a float64 scipy reference, not against the
                # other arm: MKL is the thing under comparison, so using it as the
                # oracle could only ever confirm that two arms agree. A full
                # reference is a single-threaded 100+ GFLOP job on the large
                # matrices, so check a RANDOM ROW SAMPLE -- random rather than the
                # leading block so a bug anywhere in the row space is still caught.
                nref = min(M, a.ref_rows)
                if nref < M:
                    rr = np.sort(
                        np.random.default_rng(a.seed).choice(M, nref, replace=False)
                    )
                    csr_ref = csr[rr]
                else:
                    rr, csr_ref = None, csr
                ref = csr_ref.astype(np.float64) @ B.numpy().astype(np.float64)
                refn = np.linalg.norm(ref) + 1e-30
                errs = {}
                for an, fn in (("sc", f_sc), ("mkl32", f_mkl32)):
                    r = as_tensor(fn()).reshape(M, k)
                    if rr is not None:
                        r = r[rr]
                    errs[an] = float(np.linalg.norm(r.double().numpy() - ref) / refn)
                    del r
                del ref, csr_ref

                bm = min(res["mkl32"][0], res["mkl64"][0])
                # A median over samples is the wrong estimator on a hybrid-core host:
                # which cores the OpenMP team lands on varies run to run and the two
                # kinds differ about 2x, so the same code timed twice reads up to 40%
                # apart in the 0.1-5 ms band. The minimum picks the same (uncontended)
                # placement for every arm, so it is reported alongside the median and
                # the A/A control says which one to trust at a given cell size.
                bm_min = min(res["mkl32"][1], res["mkl64"][1])
                kms = statistics.median(kern) if kern else float("nan")
                emit(
                    base
                    | dict(
                        k=k,
                        mkl32_ms=res["mkl32"][0] * 1e3,
                        mkl64_ms=res["mkl64"][0] * 1e3,
                        sc_ms=res["sc"][0] * 1e3,
                        aa_ms=res["aa"][0] * 1e3,
                        sc_kernel_ms=(kms * 1e3 if math.isfinite(kms) else ""),
                        mkl32_min_ms=res["mkl32"][1] * 1e3,
                        mkl64_min_ms=res["mkl64"][1] * 1e3,
                        sc_min_ms=res["sc"][1] * 1e3,
                        aa_min_ms=res["aa"][1] * 1e3,
                        vs_mkl_min=bm_min / res["sc"][1],
                        vs_mkl=bm / res["sc"][0],
                        vs_mkl_kernel=(
                            bm / kms if math.isfinite(kms) and kms > 0 else ""
                        ),
                        aa_ratio=res["sc"][0] / res["aa"][0],
                        route=route[0],
                        route_param=("" if route[1] is None else route[1]),
                        reps=reps,
                        batch=batch,
                        settle=a.settle,
                        relerr_sc=errs["sc"],
                        relerr_mkl=errs["mkl32"],
                        ref_rows=nref,
                        status="ok",
                        note="",
                    )
                )
                n_cell += 1
                B = B_st = None
            except Exception as ex:
                emit(
                    dict.fromkeys(FIELDS, "")
                    | base
                    | dict(k=k, status="error", note=f"{type(ex).__name__}: {ex}"[:300])
                )
                n_err += 1
                if n_err <= 5:
                    traceback.print_exc()

        # Rebound rather than `del`d: a 30 M-nonzero matrix is held three times
        # over here (int32 CSR, int64 CSR, STensor) and the next matrix must not be
        # loaded on top of it.
        A32 = A64 = A_st = csr = None
        gc.collect()
        if (mi + 1) % a.progress_every == 0:
            el = time.time() - t_start
            print(
                f"[{mi + 1}/{len(rows)}] {el / 60:7.1f} min  "
                f"cells={n_cell} skipped={n_skip} errors={n_err}  last={key}"
            )
            sys.stdout.flush()

    print(
        f"\nDONE  {(time.time() - t_start) / 60:.1f} min  "
        f"cells={n_cell} skipped={n_skip} errors={n_err}"
    )
    out.close()


if __name__ == "__main__":
    main()
