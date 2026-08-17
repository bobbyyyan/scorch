#!/usr/bin/env python3
"""phase0_attrib.py — per-cell hardware attribution for CSR x dense SpMM.

Times ONE (matrix, N, arm) cell and, when run under

    perf stat -D -1 --control fifo:<ctl>,<ack> -e ... -- python phase0_attrib.py ...

enables the counters for exactly the timed region and disables them after, so the
counts describe the kernel and not the matrix load, the B allocation, or the warmup.

Emits one `ATTRIB {json}` line on stdout. Compulsory traffic is modelled here (it is
a property of the cell, not of the arm) so the analysis step can divide measured
DRAM bytes by it and get a traffic amplification factor.

arms:
  sc_off   scorch.matmul with autotune "off"  -> spmm_csr_float_v2, untiled
  sc_tilej scorch.matmul with autotune "balanced" -> whatever the probe picks
  mkl32    torch.sparse.mm on CSR with int32 indices (MKL's native index width)
  mkl64    torch.sparse.mm on CSR with int64 indices (what scipy hands a user)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time

if os.path.isdir("/scratch/bobbyy"):
    os.environ.setdefault("HOME", "/scratch/bobbyy")
    os.environ.setdefault("MPLCONFIGDIR", "/scratch/bobbyy/.mplcache")

import numpy as np
import scipy.sparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_spmm_vs_mkl as H  # matrix loaders, to_st  # noqa: E402
import scorch  # noqa: E402


# --------------------------------------------------------------------------- #
# perf stat region control
# --------------------------------------------------------------------------- #
class PerfCtl:
    """Toggle a parent `perf stat --control fifo:ctl,ack` around a region.

    A no-op when the env vars are absent, so the harness also runs bare.
    """

    def __init__(self):
        self.ctl = self.ack = None
        c, a = os.environ.get("PERF_CTL_FIFO"), os.environ.get("PERF_ACK_FIFO")
        if c and a and os.path.exists(c) and os.path.exists(a):
            # Open write end first: perf holds the read end open.
            self.ctl = open(c, "w")
            self.ack = open(a, "r")

    def _cmd(self, s):
        if not self.ctl:
            return
        self.ctl.write(s + "\n")
        self.ctl.flush()
        self.ack.readline()  # perf writes "ack\n"

    def enable(self):
        self._cmd("enable")

    def disable(self):
        self._cmd("disable")


# --------------------------------------------------------------------------- #
# matrices
# --------------------------------------------------------------------------- #
MCACHE = os.environ.get("SCORCH_MCACHE", "/scratch/bobbyy/spmm-beat-mkl/mcache")


def load_matrix(spec):
    """Cached matrix load. Parsing audikw_1/inline_1 from Matrix Market costs a
    minute; a scipy .npz round-trip costs a second, and every probe in this study
    reloads the same handful of matrices dozens of times."""
    if MCACHE:
        try:
            os.makedirs(MCACHE, exist_ok=True)
            path = os.path.join(MCACHE, spec.replace(":", "__") + ".npz")
            if os.path.exists(path):
                return scipy.sparse.load_npz(path)
            m = _load_matrix_uncached(spec)
            tmp = path + ".tmp%d" % os.getpid()
            scipy.sparse.save_npz(tmp, m)
            os.replace(tmp, path)
            return m
        except OSError:
            pass
    return _load_matrix_uncached(spec)


def _load_matrix_uncached(spec):
    """spec like 'gcn:reddit', 'ss:inline_1', 'syn:band16', 'syn:scatter200'."""
    kind, _, name = spec.partition(":")
    if kind == "ss":
        m = H.m_suitesparse(name)
        if m is None:
            raise SystemExit(f"suitesparse matrix not found: {name}")
        return m
    if kind == "gcn":
        if name == "reddit":
            return H.m_reddit()
        if name.startswith("ogbn"):
            return H.m_ogbn(name)
        return H.m_planetoid(name)
    if kind == "syn":
        if name == "band16":
            return H.m_band(40000, 16)
        if name == "scatter16":
            return H.m_scatter(40000, 16)
        if name == "scatter200":
            return H.m_scatter(30000, 200)
        if name == "scatter200-big":
            return H.m_scatter(300000, 200)
        if name == "scatter120":
            return H.m_scatter(30000, 120, seed=2)
    raise SystemExit(f"unknown matrix spec: {spec}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--arm", required=True,
                    choices=["sc_off", "sc_balanced", "sc_analytic", "sc_off32",
                             "sc_balanced32", "mkl32", "mkl64"])
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float64"],
                    help="value dtype; float64 resolves a different prebuilt symbol "
                         "(prebuilt_spmm_csr_f64) that gets no tiling at all, but "
                         "goes through the same ABI validator")
    ap.add_argument("--extra", default=None,
                    help="JSON object merged into the emitted record (run labels)")
    ap.add_argument("--hold-window", type=int, default=2,
                    help="live outputs kept inside the counted region; 2 reproduces "
                         "the steady-state allocator behaviour of the baseline study "
                         "(one alloc + one free per rep) without letting the first "
                         "rep's page faults dominate a short region")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    csr = load_matrix(args.matrix)
    M, J = csr.shape
    nnz = int(csr.nnz)
    N = args.n

    npdt = np.float32 if args.dtype == "float32" else np.float64
    rng = np.random.default_rng(args.seed)
    B = torch.from_numpy(rng.standard_normal((J, N), dtype=npdt))

    if args.arm.startswith("sc_"):
        # The "...32" arms build the STensor from int32 index arrays instead of the
        # int64 ones scipy/torch hand a user. Same kernel, same data — the only
        # difference is that native_abi.h's checked_index_tensor then takes its
        # cheap path instead of rescanning and re-casting every index per call.
        lvl = args.arm[3:]
        if lvl.endswith("32"):
            lvl = lvl[:-2]
            it = np.int32
        else:
            it = np.int64
        A = scorch.STensor.from_torch(
            torch.sparse_csr_tensor(
                torch.from_numpy(csr.indptr.astype(it)),
                torch.from_numpy(csr.indices.astype(it)),
                torch.from_numpy(csr.data.astype(npdt)),
                size=csr.shape,
            )
        )
        scorch.set_autotune(lvl)
        td = {}

        def run():
            return scorch.matmul(A, B, time_dict=td)
    else:
        it = np.int32 if args.arm == "mkl32" else np.int64
        A = torch.sparse_csr_tensor(
            torch.from_numpy(csr.indptr.astype(it)),
            torch.from_numpy(csr.indices.astype(it)),
            torch.from_numpy(csr.data.astype(npdt)),
            size=csr.shape,
        )

        td = None

        def run():
            return torch.sparse.mm(A, B)

    # Warmup outside the counted region: first-touch of the output pages, MKL handle
    # creation, and (for sc_balanced) the whole micro-probe all land here.
    out = None
    for _ in range(max(1, args.warmup)):
        out = run()
    del out
    gc.collect()

    ctl = PerfCtl()
    times, ktimes = [], []
    keep = []
    ctl.enable()
    t_region0 = time.perf_counter()
    for _ in range(args.reps):
        t0 = time.perf_counter()
        out = run()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if td is not None and "eval_time" in td:
            ktimes.append(td["eval_time"])
        keep.append(out)
        if len(keep) > max(0, args.hold_window):
            keep.pop(0)
    t_region = time.perf_counter() - t_region0
    ctl.disable()
    del keep

    # Compulsory DRAM traffic for this cell: A read once (4B index + 4B value per
    # nonzero, plus the row pointer), B read once, C written once (a write costs a
    # line fill plus a writeback unless the store is non-temporal, hence 2x).
    esz = 4.0 if args.dtype == "float32" else 8.0
    a_bytes = (4.0 + esz) * nnz + 4.0 * (M + 1)
    b_bytes = esz * J * N
    c_bytes = esz * M * N
    rec = dict(
        matrix=args.matrix, n=N, arm=args.arm, dtype=args.dtype, M=M, J=J, nnz=nnz,
        degree=nnz / max(1, J), reps=args.reps,
        t_med=statistics.median(times), t_min=min(times), t_max=max(times),
        t_sum=sum(times), t_region=t_region, hold_window=args.hold_window,
        overhead_frac=1.0 - sum(times) / t_region,
        t_kernel_med=(statistics.median(ktimes) if ktimes else None),
        compulsory_read=a_bytes + b_bytes,
        compulsory_rw=a_bytes + b_bytes + 2.0 * c_bytes,
        a_bytes=a_bytes, b_bytes=b_bytes, c_bytes=c_bytes,
        threads=torch.get_num_threads(),
    )
    if args.extra:
        rec.update(json.loads(args.extra))
    print("ATTRIB " + json.dumps(rec))


if __name__ == "__main__":
    main()
