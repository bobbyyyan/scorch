#!/usr/bin/env python3
"""Measurements for the performance section of the README.

Each measurement compares two calls that compute the same result:

* the call a PyTorch user writes with ``import torch``, and
* the same call under ``import scorch as torch``, which sends ``torch.matmul`` and
  ``torch.einsum`` to Scorch (``scorch.matmul`` and ``scorch.einsum`` are the same
  function objects).

Operands are ordinary ``torch`` tensors: a float32 sparse CSR matrix with int64
indices, as ``Tensor.to_sparse_csr()`` builds it, and dense float32 operands.

Subcommands
-----------
``select``
    Print the matrix set drawn from the SuiteSparse index (``~/.ssgetpy/index.db``,
    the index the ``ssgetpy`` package downloads) by the rule in ``select_matrices``
    and check that it equals ``MATRICES`` below.
``kernels``
    Time SpMV, SpMM (k = 16, 64, 256), SDDMM (k = 64) and SpGEMM (A @ A) on every
    matrix of ``MATRICES`` that is found under ``--matrix-dir``.
``report``
    Summarize one or more ``kernels`` result directories (one per host) into
    ``summary.md`` and a figure in a light and a dark version.

Method (``kernels``)
--------------------
* Each cell (matrix, operation) is timed in two passes over the matrix set: the
  first in the order of ``MATRICES``, the second in reverse order.
* Within a pass, three arms are timed in rounds: the PyTorch call, the Scorch call,
  and the Scorch call again as a second, identical arm.  The arm order is shuffled
  in every round (seeded).  The ratio of the two Scorch arms is an A/A control: its
  spread over the matrix set is the noise of the measurement setup.
* The first call of each arm is recorded separately (``first_call_s``); it includes
  any kernel compilation and any per-operand setup.  One more untimed call per arm
  follows before the rounds.  A cell runs at least ``--min-rounds`` rounds and keeps
  going until each arm has run for ``--budget`` seconds or ``--max-rounds`` is hit.
* The reported time of an arm is the median of its rounds.  The speedup of a cell is
  the PyTorch median divided by the Scorch median, per pass; ``report`` takes the
  geometric mean of the two passes.
* Correctness: every result is compared with a float64 reference computed by SciPy.
  An entry passes when ``|y - y_ref| <= gamma_n * (|A| |X|)`` elementwise, with
  ``gamma_n = n u / (1 - n u)``, ``u = 2**-24`` and ``n`` the number of products
  summed into that entry (Higham, *Accuracy and Stability of Numerical Algorithms*,
  2nd ed., section 3.1).  The bound holds for any summation order.
* Threads are the defaults a user gets: nothing is set.  ``meta.json`` records
  ``torch.get_num_threads()``, the CPU model and the PyTorch build.

Usage
-----
    python bench/readme_benchmarks.py select
    python bench/readme_benchmarks.py kernels --matrix-dir ~/suitesparse --out results/m5
    python bench/readme_benchmarks.py report results/m5 results/mkt1 --out docs/assets/perf

Matrix files are read as ``<dir>/<group>/<name>/<name>.mtx`` (the layout of a
SuiteSparse tarball), ``<dir>/<group>/<name>.npz`` or ``<dir>/<name>/<name>.npz``
(``indptr``, ``indices``, ``data``, ``shape``), or ``<dir>/<name>.mtx``.  Download
them with ``ssgetpy`` or from https://sparse.tamu.edu.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import sqlite3
import statistics
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# The matrix set
# ---------------------------------------------------------------------------

# Stratification of the draw: nonzeros by decade, and mean nonzeros per row.
NNZ_BINS: Tuple[Tuple[float, float], ...] = (
    (1e4, 1e5),
    (1e5, 1e6),
    (1e6, 1e7),
    (1e7, 2.9e7),
)
ROW_BINS: Tuple[Tuple[float, float], ...] = ((0, 4), (4, 16), (16, 64), (64, math.inf))
PER_STRATUM = 6
SEED = 2026

# (SuiteSparse id, group, name, rows, cols, nnz), filled from ``select`` output.
MATRICES: List[Tuple[int, str, str, int, int, int]] = [
    (1430, "Oberwolfach", "filter2D", 1668, 1668, 10750),
    (1822, "Meszaros", "plddb", 3069, 5049, 10839),
    (1506, "Pajek", "Journals", 124, 124, 12068),
    (2189, "JGD_SL6", "D_7", 1270, 971, 12714),
    (2418, "AG-Monien", "grid2", 3296, 3296, 12864),
    (1932, "JGD_BIBD", "bibd_17_4", 136, 2380, 14280),
    (2006, "JGD_Groebner", "robot24c1_mat5", 404, 302, 15118),
    (336, "Bai", "rw5151", 5151, 5151, 20199),
    (2311, "SNAP", "p2p-Gnutella08", 6301, 6301, 20777),
    (2062, "JGD_Homology", "mk11-b2", 6930, 990, 20790),
    (2677, "VDOL", "freeFlyingRobot_5", 2878, 2878, 24582),
    (698, "LPnetlib", "lp_truss", 1000, 8806, 27836),
    (157, "HB", "gemat12", 4929, 4929, 33044),
    (34, "HB", "bcsstk12", 1473, 1473, 34241),
    (660, "LPnetlib", "lp_qap12", 3192, 8856, 38304),
    (363, "Boeing", "nasa1824", 1824, 1824, 39208),
    (1190, "Rajat", "rajat06", 10922, 10922, 46983),
    (78, "HB", "beacxc", 497, 506, 50409),
    (2850, "Negre", "dendrimer", 730, 730, 63024),
    (919, "Hohn", "fd18", 16428, 16428, 63406),
    (412, "FIDAP", "ex14", 3251, 3251, 65875),
    (701, "LPnetlib", "lp_wood1p", 244, 2595, 70216),
    (1680, "Meszaros", "aa03", 825, 8627, 70806),
    (1726, "Meszaros", "model5", 1888, 11802, 89925),
    (1624, "Bai", "mhd4800a", 4800, 4800, 102252),
    (2569, "Bodendiek", "CurlCurl_0", 11083, 11083, 113343),
    (791, "Okunbor", "aft01", 8205, 8205, 125567),
    (1218, "GHS_indef", "aug3dcqp", 35543, 35543, 128115),
    (2434, "AG-Monien", "shock-9", 36476, 36476, 142580),
    (2052, "JGD_Homology", "D6-6", 120576, 23740, 146520),
    (1821, "Meszaros", "nsir", 4453, 10057, 154939),
    (818, "Simon", "raefsky5", 6316, 6316, 167178),
    (1459, "Pajek", "dictionary28", 52652, 52652, 178076),
    (1721, "Meszaros", "mod2", 34774, 66409, 199810),
    (578, "Hollinger", "jan99jac120", 41374, 41374, 229385),
    (2829, "TAMU_SmartGridCenter", "ACTIVSg70K", 69999, 69999, 238627),
    (1332, "Toledo", "deltaX", 68600, 21961, 247424),
    (2568, "Davis", "FX_March2010", 1319, 9498, 301899),
    (346, "Boeing", "bcsstm36", 23052, 23052, 320606),
    (354, "Boeing", "crystm02", 13965, 13965, 322905),
    (768, "Nemeth", "nemeth04", 9506, 9506, 394808),
    (236, "HB", "psmigr_2", 3140, 3140, 540022),
    (899, "Shen", "e40r0100", 17281, 17281, 553562),
    (781, "Nemeth", "nemeth17", 9506, 9506, 629620),
    (1394, "Andrianov", "pf2177", 9728, 9728, 725144),
    (2598, "DIMACS10", "ky2010", 161672, 161672, 787778),
    (930, "FEMLAB", "sme3Da", 12504, 12504, 874887),
    (2004, "JGD_Groebner", "HFE18_96_in", 2372, 4096, 933343),
    (1435, "Oberwolfach", "gyro", 17361, 17361, 1021159),
    (2604, "DIMACS10", "mn2010", 259777, 259777, 1227102),
    (1297, "GHS_indef", "boyd2", 466316, 466316, 1500397),
    (1670, "Mittelmann", "neos", 479119, 515905, 1526794),
    (1308, "GHS_indef", "sparsine", 50000, 50000, 1548988),
    (2038, "JGD_Homology", "ch7-9-b4", 317520, 105840, 1587600),
    (1859, "QLi", "majorbasis", 160000, 160000, 1750416),
    (2222, "TSOPF", "TSOPF_FS_b162_c3", 30798, 30798, 1801300),
    (2788, "SNAP", "loc-Gowalla", 196591, 196591, 1900654),
    (1235, "GHS_indef", "exdata_1", 6001, 6001, 2269500),
    (1419, "Sandia", "ASIC_680k", 682862, 682862, 2638997),
    (761, "Nasa", "nasasrb", 54870, 54870, 2677324),
    (2510, "DIMACS10", "belgium_osm", 1441295, 1441295, 3099940),
    (932, "FEMLAB", "sme3Dc", 42930, 42930, 3148656),
    (803, "Rothberg", "3dtube", 45330, 45330, 3213618),
    (1774, "Meszaros", "stat96v3", 33841, 1113780, 3317736),
    (1396, "Rajat", "rajat29", 643994, 643994, 3760246),
    (537, "Gupta", "gupta2", 62064, 62064, 4248286),
    (2239, "TSOPF", "TSOPF_RS_b300_c3", 42138, 42138, 4413449),
    (1423, "GHS_psdef", "apache2", 715176, 715176, 4817870),
    (2291, "SNAP", "wiki-Talk", 2394385, 2394385, 5021410),
    (1201, "Hamrle", "Hamrle3", 1447360, 1447360, 5514242),
    (2336, "Fluorem", "PR02R", 161070, 161070, 8185136),
    (1995, "JGD_GL7d", "GL7d22", 349443, 822922, 8251000),
    (2268, "Bourchtein", "atmosmodm", 1489752, 1489752, 10319760),
    (1257, "GHS_psdef", "crankseg_1", 52804, 52804, 10614210),
    (1658, "Mittelmann", "rail4284", 4284, 1096894, 11284032),
    (1285, "DNVS", "troll", 213453, 213453, 11985111),
    (1876, "Zaoui", "kkt_power", 2063494, 2063494, 12771361),
    (2786, "SNAP", "higgs-twitter", 456626, 456626, 14855842),
    (2277, "Freescale", "circuit5M_dc", 3523317, 3523317, 14865409),
    (1514, "Pajek", "patents", 3774768, 3774768, 14970767),
    (1362, "PARSEC", "Si41Ge41H72", 185639, 185639, 15011265),
    (2294, "SNAP", "cit-Patents", 3774768, 3774768, 16518948),
    (2647, "Janna", "CoupCons3D", 416800, 416800, 17277420),
    (943, "Schenk_AFE", "af_shell4", 504855, 504855, 17562051),
    (2281, "Belcastro", "human_gene2", 14340, 14340, 18068388),
    (2547, "Janna", "StocF-1465", 1465137, 1465137, 21005389),
    (2491, "DIMACS10", "kron_g500-logn18", 262144, 262144, 21165908),
    (2329, "Dziekonski", "gsm_106857", 589446, 589446, 21758924),
    (2577, "DIMACS10", "333SP", 3712815, 3712815, 22217266),
    (2178, "JGD_Relat", "rel9", 9888048, 274669, 23667183),
    (2512, "DIMACS10", "germany_osm", 11548845, 11548845, 24738362),
    (1990, "JGD_GL7d", "GL7d17", 1548650, 955128, 25978098),
    (2896, "FlowIPM22", "Spielman_k300", 9045202, 9045202, 27226204),
    (2648, "Janna", "ML_Laplace", 377002, 377002, 27582698),
]


def select_matrices(index_db: Path) -> List[Tuple[int, str, str, int, int, int]]:
    """Draw the matrix set from the SuiteSparse index.

    Candidates are the real and binary (pattern) matrices with 10^4 to 2.9 * 10^7
    stored nonzeros.  They are split into 16 strata by ``NNZ_BINS`` x ``ROW_BINS``;
    from each stratum ``PER_STRATUM`` matrices are drawn at random with seed ``SEED``,
    at most one per SuiteSparse group, so that one family cannot fill a stratum.
    """

    con = sqlite3.connect(str(index_db))
    rows = con.execute(
        "select id, matrixgroup, name, rows, cols, nnz, dtype from matrices"
    ).fetchall()
    con.close()
    rng = random.Random(SEED)
    chosen: List[Tuple[int, str, str, int, int, int]] = []
    for lo_n, hi_n in NNZ_BINS:
        for lo_r, hi_r in ROW_BINS:
            stratum = sorted(
                r
                for r in rows
                if r[6] in ("real", "binary")
                and lo_n <= r[5] < hi_n
                and lo_r <= r[5] / r[3] < hi_r
            )
            rng.shuffle(stratum)
            groups: set = set()
            for r in stratum:
                if len(groups) == PER_STRATUM:
                    break
                if r[1] in groups:
                    continue
                groups.add(r[1])
                chosen.append((r[0], r[1], r[2], r[3], r[4], r[5]))
    return sorted(chosen, key=lambda m: m[5])


# ---------------------------------------------------------------------------
# Loading matrices
# ---------------------------------------------------------------------------


def _candidate_paths(root: Path, group: str, name: str) -> List[Path]:
    return [
        root / group / name / f"{name}.mtx",
        root / group / f"{name}.npz",
        root / group / name / f"{name}.npz",
        root / name / f"{name}.npz",
        root / name / f"{name}.mtx",
        root / f"{name}.mtx",
    ]


def find_matrix(roots: Sequence[Path], group: str, name: str) -> Optional[Path]:
    for root in roots:
        for p in _candidate_paths(root, group, name):
            if p.exists():
                return p
    return None


def load_csr(path: Path):
    """Read a matrix as a SciPy CSR matrix: float32 values, sorted, no duplicates,
    no explicit zeros.  A pattern matrix gets the value 1 at every entry."""

    import scipy.io
    import scipy.sparse

    if path.suffix == ".npz":
        z = np.load(str(path))
        a = scipy.sparse.csr_matrix(
            (z["data"], z["indices"], z["indptr"]),
            shape=tuple(int(s) for s in z["shape"]),
        )
    else:
        a = scipy.sparse.csr_matrix(scipy.io.mmread(str(path)))
    if a.dtype.kind == "c":
        a = a.real
    if a.dtype.kind not in "f":
        a = scipy.sparse.csr_matrix(
            (np.ones(a.nnz), a.indices, a.indptr), shape=a.shape
        )
    a = a.astype(np.float32)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    return a


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

U32 = 2.0**-24


def gamma(n: np.ndarray) -> np.ndarray:
    nu = n.astype(np.float64) * U32
    return nu / (1.0 - nu)


def check_dense(
    y, ref: np.ndarray, absref: np.ndarray, n_terms: np.ndarray
) -> Tuple[bool, float]:
    """Elementwise error bound for a dense result.  ``n_terms`` broadcasts against
    ``ref`` (one count per row).  Returns (passes, largest error / bound)."""

    y64 = (
        y.detach().to_dense().double().numpy()
        if hasattr(y, "detach")
        else np.asarray(y)
    )
    bound = gamma(n_terms) * absref
    err = np.abs(y64 - ref)
    tiny = np.finfo(np.float64).tiny
    ratio = float(np.max(err / np.maximum(bound, tiny))) if err.size else 0.0
    ok = bool(np.all(err <= bound + tiny)) and bool(np.all(np.isfinite(y64)))
    return ok, ratio


def csr_parts(t) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """(indptr, indices, values, shape) of a torch sparse result, any layout."""

    import torch

    if hasattr(t, "storage") and hasattr(t, "to_torch"):  # an STensor
        # STensor.to_torch() always densifies, so read the CSR arrays directly.
        index = t.storage.index
        if str(t.format) != "d,s" or list(index.mode_order or [0, 1]) != [0, 1]:
            raise TypeError(f"expected a CSR STensor, got format {t.format}")
        pos, crd = index.mode_indices[1][0], index.mode_indices[1][1]
        return (
            pos.numpy().astype(np.int64),
            crd.numpy().astype(np.int64),
            t.storage.value.detach().double().numpy(),
            tuple(t.shape),
        )
    if t.layout == torch.sparse_csr:
        return (
            t.crow_indices().numpy(),
            t.col_indices().numpy(),
            t.values().detach().double().numpy(),
            tuple(t.shape),
        )
    if t.layout == torch.sparse_coo:
        t = t.coalesce().to_sparse_csr()
        return csr_parts(t)
    if t.layout == torch.strided:
        return csr_parts(t.to_sparse_csr())
    raise TypeError(f"unexpected result layout {t.layout}")


def check_sparse(
    result, ref, absref, n_terms_per_row: np.ndarray
) -> Tuple[bool, float]:
    """Elementwise bound for a sparse result against SciPy references with the same
    shape.  Entries missing from ``result`` count as 0."""

    import scipy.sparse

    indptr, indices, values, shape = csr_parts(result)
    got = scipy.sparse.csr_matrix((values, indices, indptr), shape=shape)
    got.sum_duplicates()
    diff = (got - ref).tocsr()
    diff.sum_duplicates()
    if diff.nnz == 0:
        return True, 0.0
    bound = absref.tocsr()
    rows = np.repeat(np.arange(shape[0]), np.diff(diff.indptr))
    b = np.asarray(bound[rows, diff.indices]).ravel() * gamma(n_terms_per_row[rows])
    err = np.abs(diff.data)
    tiny = np.finfo(np.float64).tiny
    ok = bool(np.all(err <= b + tiny)) and bool(np.all(np.isfinite(values)))
    return ok, float(np.max(err / np.maximum(b, tiny)))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_arms(
    arms: Dict[str, Callable[[], Any]],
    rng: random.Random,
    min_rounds: int,
    max_rounds: int,
    budget: float,
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """First call of each arm, one untimed call, then shuffled rounds."""

    first: Dict[str, float] = {}
    for name, fn in arms.items():
        t0 = time.perf_counter()
        fn()
        first[name] = time.perf_counter() - t0
    for fn in arms.values():
        fn()
    times: Dict[str, List[float]] = {name: [] for name in arms}
    order = list(arms)
    rounds = 0
    while rounds < max_rounds:
        if rounds >= min_rounds and all(sum(v) >= budget for v in times.values()):
            break
        rng.shuffle(order)
        for name in order:
            t0 = time.perf_counter()
            arms[name]()
            times[name].append(time.perf_counter() - t0)
        rounds += 1
    return first, times


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

OPS = ("spmv", "spmm16", "spmm64", "spmm256", "sddmm64", "spgemm")

# Cells whose dense operands would exceed this many bytes are skipped (recorded).
DENSE_BYTES_CAP = 4 << 30
# SpGEMM: square matrices with at most this many scalar multiply-adds.
SPGEMM_FLOPS_CAP = 1e9


def build_cell(op: str, a_sp, name: str):
    """Return (torch_fn, scorch_fn, checker, info) or a skip reason string."""

    import torch

    import scorch

    m, n = a_sp.shape
    gen = torch.Generator().manual_seed(zlib.crc32(f"{name}:{op}".encode()))
    A = torch.sparse_csr_tensor(
        torch.from_numpy(a_sp.indptr.astype(np.int64)),
        torch.from_numpy(a_sp.indices.astype(np.int64)),
        torch.from_numpy(a_sp.data),
        size=(m, n),
    )
    row_nnz = np.diff(a_sp.indptr)
    a64 = a_sp.astype(np.float64)
    abs_a = abs(a64)
    info: Dict[str, Any] = {"baseline": ""}

    if op == "spmv" or op.startswith("spmm"):
        k = 1 if op == "spmv" else int(op[4:])
        if 4 * (m + n) * k > DENSE_BYTES_CAP:
            return f"dense operands over {DENSE_BYTES_CAP >> 30} GiB"
        X = torch.rand((n,) if op == "spmv" else (n, k), generator=gen)
        x64 = X.double().numpy()
        ref = a64 @ x64
        absref = abs_a @ np.abs(x64)
        n_terms = row_nnz if op == "spmv" else row_nnz[:, None]
        info["k"] = k
        info["baseline"] = "torch.matmul(A_csr, X)"

        def checker(y):
            return check_dense(y, ref, absref, n_terms)

        return (
            (lambda: torch.matmul(A, X)),
            (lambda: scorch.matmul(A, X)),
            checker,
            info,
        )

    if op.startswith("sddmm"):
        k = int(op[5:])
        if 4 * (m + n) * k > DENSE_BYTES_CAP:
            return f"dense operands over {DENSE_BYTES_CAP >> 30} GiB"
        U = torch.rand((m, k), generator=gen)
        V = torch.rand((k, n), generator=gen)
        vals = A.values()

        def torch_sddmm():
            s = torch.sparse.sampled_addmm(A, U, V, beta=0.0)
            return torch.sparse_csr_tensor(
                s.crow_indices(), s.col_indices(), s.values() * vals, size=(m, n)
            )

        def scorch_sddmm():
            return scorch.einsum("ij,ik,kj->ij", A, U, V, format="ds")

        # Reference in blocks of nonzeros: v_ij * (U[i, :] . V[:, j]).
        u64, v64 = U.double().numpy(), V.double().numpy().T
        rows_of = np.repeat(np.arange(m), row_nnz)
        ref_vals = np.empty(a_sp.nnz)
        abs_vals = np.empty(a_sp.nnz)
        step = max(1, (1 << 27) // (8 * k))
        for s0 in range(0, a_sp.nnz, step):
            r = rows_of[s0 : s0 + step]
            c = a_sp.indices[s0 : s0 + step]
            dots = np.einsum("ij,ij->i", u64[r], v64[c])
            ref_vals[s0 : s0 + step] = a64.data[s0 : s0 + step] * dots
            abs_vals[s0 : s0 + step] = (
                np.abs(a64.data[s0 : s0 + step]) * dots
            )  # U, V >= 0
        import scipy.sparse

        ref = scipy.sparse.csr_matrix(
            (ref_vals, a_sp.indices, a_sp.indptr), shape=(m, n)
        )
        absref = scipy.sparse.csr_matrix(
            (abs_vals, a_sp.indices, a_sp.indptr), shape=(m, n)
        )
        n_terms = np.full(m, k + 1)
        info["k"] = k
        info["baseline"] = "torch.sparse.sampled_addmm(A, U, V, beta=0) * A.values()"

        def checker(y):
            return check_sparse(y, ref, absref, n_terms)

        return torch_sddmm, scorch_sddmm, checker, info

    if op == "spgemm":
        if m != n:
            return "not square"
        col_nnz = np.bincount(a_sp.indices, minlength=n)
        flops = float(np.dot(col_nnz.astype(np.float64), row_nnz.astype(np.float64)))
        info["flops"] = flops
        if flops > SPGEMM_FLOPS_CAP:
            return f"{flops:.2e} multiply-adds, over {SPGEMM_FLOPS_CAP:.0e}"
        try:
            torch_fn = lambda: torch.matmul(A, A)  # noqa: E731
            torch_fn()
            info["baseline"] = "torch.matmul(A_csr, A_csr)"
        except RuntimeError as e:
            if "MKL" not in str(e):
                raise
            A_coo = A.to_sparse_coo().coalesce()
            torch_fn = lambda: torch.sparse.mm(A_coo, A_coo)  # noqa: E731
            info["baseline"] = "torch.sparse.mm(A_coo, A_coo) (CSR x CSR needs MKL)"
        ref = (a64 @ a64).tocsr()
        absref = (abs_a @ abs_a).tocsr()

        def checker(y):
            return check_sparse(
                y, ref, absref, np.full(m, int(row_nnz.max(initial=0)) + 1)
            )

        return torch_fn, (lambda: scorch.matmul(A, A)), checker, info

    raise ValueError(op)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

FIELDS = [
    "pass",
    "op",
    "ssid",
    "group",
    "name",
    "rows",
    "cols",
    "nnz",
    "k",
    "status",
    "baseline",
    "torch_ok",
    "torch_err_ratio",
    "scorch_ok",
    "scorch_err_ratio",
    "torch_first_s",
    "scorch_first_s",
    "torch_median_s",
    "scorch_median_s",
    "scorch_b_median_s",
    "rounds",
    "torch_times",
    "scorch_times",
    "scorch_b_times",
    "wall_s",
]


def host_meta(tree: Path) -> Dict[str, Any]:
    import torch

    import scorch

    cpu = platform.processor()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        else:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    commit = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "host": platform.node(),
        "cpu": cpu,
        "machine": platform.machine(),
        "os": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_mkl": bool(torch.backends.mkl.is_available()),
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        "scorch_file": scorch.__file__,
        "scorch_commit": commit,
        "scorch_tree_modified": bool(dirty),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def cmd_kernels(args: argparse.Namespace) -> int:
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tree = Path(__file__).resolve().parent.parent
    meta = host_meta(tree)
    meta.update(
        {
            "ops": args.ops,
            "min_rounds": args.min_rounds,
            "max_rounds": args.max_rounds,
            "budget_s": args.budget,
            "matrix_dirs": [str(d) for d in args.matrix_dir],
        }
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)

    roots = [Path(d).expanduser() for d in args.matrix_dir]
    entries = MATRICES if not args.only else [m for m in MATRICES if m[2] in args.only]
    found = [(m, find_matrix(roots, m[1], m[2])) for m in entries]
    missing = [m for m, p in found if p is None]
    if missing:
        (out / "missing.txt").write_text("".join(f"{m[1]}/{m[2]}\n" for m in missing))
        print(f"{len(missing)} of {len(entries)} matrices not found; see missing.txt")
        if args.require_all:
            return 2
    present = [(m, p) for m, p in found if p is not None]
    rng = random.Random(SEED)

    for pass_no in range(1, args.passes + 1):
        order = present if pass_no % 2 == 1 else list(reversed(present))
        path = out / f"pass{pass_no}.csv"
        done = set()
        if path.exists():
            with open(path) as fh:
                done = {(r["op"], r["name"]) for r in csv.DictReader(fh)}
        new = not path.exists()
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            for m, p in order:
                ssid, group, name, rows, cols, nnz = m
                todo = [op for op in args.ops if (op, name) not in done]
                if not todo:
                    continue
                a_sp = load_csr(p)
                for op in todo:
                    t_cell = time.perf_counter()
                    row: Dict[str, Any] = {
                        "pass": pass_no,
                        "op": op,
                        "ssid": ssid,
                        "group": group,
                        "name": name,
                        "rows": a_sp.shape[0],
                        "cols": a_sp.shape[1],
                        "nnz": a_sp.nnz,
                    }
                    try:
                        cell = build_cell(op, a_sp, name)
                        if isinstance(cell, str):
                            row["status"] = "skipped: " + cell
                        else:
                            torch_fn, scorch_fn, checker, info = cell
                            row["k"] = info.get("k", "")
                            row["baseline"] = info["baseline"]
                            row["torch_ok"], row["torch_err_ratio"] = checker(
                                torch_fn()
                            )
                            row["scorch_ok"], row["scorch_err_ratio"] = checker(
                                scorch_fn()
                            )
                            first, times = time_arms(
                                {
                                    "torch": torch_fn,
                                    "scorch": scorch_fn,
                                    "scorch_b": scorch_fn,
                                },
                                rng,
                                args.min_rounds,
                                args.max_rounds,
                                args.budget,
                            )
                            row["torch_first_s"] = first["torch"]
                            row["scorch_first_s"] = first["scorch"]
                            for arm in ("torch", "scorch", "scorch_b"):
                                row[f"{arm}_median_s"] = statistics.median(times[arm])
                                row[f"{arm}_times"] = json.dumps(
                                    [round(t, 9) for t in times[arm]]
                                )
                            row["rounds"] = len(times["torch"])
                            row["status"] = "ok"
                    except Exception as e:  # recorded, never hidden
                        row["status"] = f"error: {type(e).__name__}: {str(e)[:300]}"
                    row["wall_s"] = round(time.perf_counter() - t_cell, 3)
                    w.writerow(row)
                    fh.flush()
                    sp = (
                        f"{row['torch_median_s'] / row['scorch_median_s']:.2f}x"
                        if row.get("status") == "ok"
                        else row["status"]
                    )
                    print(
                        f"pass {pass_no} {op:8s} {name:28s} nnz={a_sp.nnz:>10,}  {sp}",
                        flush=True,
                    )
                    gc.collect()
                del a_sp
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return 0


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def cmd_select(args: argparse.Namespace) -> int:
    drawn = select_matrices(Path(args.index).expanduser())
    for m in drawn:
        print(f"    {m!r},")
    print(f"# {len(drawn)} matrices", file=sys.stderr)
    if MATRICES and drawn != MATRICES:
        print("the draw differs from MATRICES", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

OP_LABELS = {
    "spmv": "SpMV",
    "spmm16": "SpMM k=16",
    "spmm64": "SpMM k=64",
    "spmm256": "SpMM k=256",
    "sddmm64": "SDDMM k=64",
    "spgemm": "SpGEMM",
}


def _read_results(
    d: Path,
) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], List[Dict[str, str]]]]:
    meta = json.loads((d / "meta.json").read_text())
    cells: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for p in sorted(d.glob("pass*.csv")):
        with open(p) as fh:
            for r in csv.DictReader(fh):
                cells.setdefault((r["op"], r["name"]), []).append(r)
    return meta, cells


def summarize(
    cells: Dict[Tuple[str, str], List[Dict[str, str]]],
) -> Dict[str, Dict[str, Any]]:
    by_op: Dict[str, Dict[str, Any]] = {}
    for (op, name), rows in cells.items():
        ok = [r for r in rows if r["status"] == "ok"]
        s = by_op.setdefault(
            op,
            {
                "speedups": [],
                "aa": [],
                "skipped": 0,
                "errors": 0,
                "wrong": 0,
                "torch_wrong": 0,
                "names": [],
            },
        )
        if not ok:
            if any(r["status"].startswith("error") for r in rows):
                s["errors"] += 1
            else:
                s["skipped"] += 1
            continue
        if any(r["scorch_ok"] != "True" for r in ok):
            s["wrong"] += 1
        if any(r["torch_ok"] != "True" for r in ok):
            s["torch_wrong"] += 1
        ratios = [float(r["torch_median_s"]) / float(r["scorch_median_s"]) for r in ok]
        aas = [float(r["scorch_median_s"]) / float(r["scorch_b_median_s"]) for r in ok]
        s["speedups"].append(math.exp(statistics.fmean(math.log(x) for x in ratios)))
        s["aa"].append(math.exp(statistics.fmean(math.log(x) for x in aas)))
        s["names"].append(name)
    for s in by_op.values():
        v = sorted(s["speedups"])
        if v:
            q = (
                statistics.quantiles(v, n=4, method="inclusive")
                if len(v) > 1
                else [v[0]] * 3
            )
            s.update(
                n=len(v),
                geomean=math.exp(statistics.fmean(math.log(x) for x in v)),
                q1=q[0],
                median=q[1],
                q3=q[2],
                lo=v[0],
                hi=v[-1],
                slower=sum(1 for x in v if x < 1.0),
                aa_p95=sorted(abs(math.log(a)) for a in s["aa"])[
                    max(0, math.ceil(0.95 * len(s["aa"])) - 1)
                ],
            )
    return by_op


def cmd_report(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hosts = []
    for d in args.results:
        meta, cells = _read_results(Path(d))
        hosts.append((Path(d).name, meta, summarize(cells)))

    lines = [
        "| Host | Operation | Matrices | Geometric mean | Median | Quartiles | Range | PyTorch faster | Wrong results |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for label, meta, by_op in hosts:
        for op in OPS:
            s = by_op.get(op)
            if not s or not s.get("n"):
                continue
            lines.append(
                f"| {label} | {OP_LABELS[op]} | {s['n']} | {s['geomean']:.2f}× | {s['median']:.2f}× "
                f"| {s['q1']:.2f}–{s['q3']:.2f}× | {s['lo']:.2f}–{s['hi']:.2f}× | {s['slower']} "
                f"| {s['wrong']} |"
            )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    meta_out = {label: meta for label, meta, _ in hosts}
    for label, _, by_op in hosts:
        meta_out[label]["aa_p95_abs_log"] = {
            op: s.get("aa_p95") for op, s in by_op.items()
        }
        meta_out[label]["skipped"] = {op: s["skipped"] for op, s in by_op.items()}
        meta_out[label]["errors"] = {op: s["errors"] for op, s in by_op.items()}
        meta_out[label]["torch_wrong"] = {
            op: s["torch_wrong"] for op, s in by_op.items()
        }
    (out / "summary.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    print("\n".join(lines))
    for theme in ("light", "dark"):
        plot(hosts, out / f"speedup-{theme}.svg", theme)
    return 0


THEMES = {
    "light": {
        "ink": "#1D1D1F",
        "muted": "#59636E",
        "grid": "#D1D9E0",
        "hosts": ["#E8431F", "#2F6FB0"],
        "band": "#E8E8E6",
    },
    "dark": {
        "ink": "#F4F4F5",
        "muted": "#9198A1",
        "grid": "#3D444D",
        "hosts": ["#FF6A3D", "#6CA8E8"],
        "band": "#20262E",
    },
}


def plot(hosts, path: Path, theme: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "svg.fonttype": "path",
            "axes.edgecolor": t["grid"],
            "axes.labelcolor": t["ink"],
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "text.color": t["ink"],
        }
    )
    ops = [op for op in OPS if any(op in h[2] and h[2][op].get("n") for h in hosts)]
    fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    width = 0.8 / max(1, len(hosts))
    rng = random.Random(0)
    for hi, (label, meta, by_op) in enumerate(hosts):
        color = t["hosts"][hi % len(t["hosts"])]
        for oi, op in enumerate(ops):
            s = by_op.get(op)
            if not s or not s.get("n"):
                continue
            x0 = oi - 0.4 + width * (hi + 0.5)
            xs = [x0 + (rng.random() - 0.5) * width * 0.7 for _ in s["speedups"]]
            ax.scatter(
                xs, s["speedups"], s=9, color=color, alpha=0.55, linewidths=0, zorder=3
            )
            ax.plot(
                [x0 - width * 0.42, x0 + width * 0.42],
                [s["geomean"]] * 2,
                color=t["ink"],
                lw=1.6,
                zorder=4,
                solid_capstyle="butt",
            )
        ax.scatter([], [], s=24, color=color, label=f"{label}: {meta.get('cpu', '')}")
    ax.axhline(1.0, color=t["muted"], lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels([OP_LABELS[o] for o in ops])
    ax.set_xlim(-0.6, len(ops) - 0.4)
    ax.set_ylabel("PyTorch time / Scorch time")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×")
    )
    ax.grid(axis="y", color=t["grid"], lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=10, handletextpad=0.3)
    for txt in leg.get_texts():
        txt.set_color(t["ink"])
    fig.tight_layout()
    fig.savefig(str(path), transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="draw the matrix set from the SuiteSparse index")
    s.add_argument("--index", default="~/.ssgetpy/index.db")

    k = sub.add_parser("kernels", help="time the kernels on the matrix set")
    k.add_argument("--matrix-dir", nargs="+", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--ops", nargs="+", default=list(OPS), choices=OPS)
    k.add_argument(
        "--only", nargs="+", default=None, help="matrix names to run (testing)"
    )
    k.add_argument("--passes", type=int, default=2)
    k.add_argument("--min-rounds", type=int, default=7)
    k.add_argument("--max-rounds", type=int, default=60)
    k.add_argument(
        "--budget", type=float, default=0.25, help="seconds per arm per cell"
    )
    k.add_argument("--require-all", action="store_true")

    r = sub.add_parser("report", help="summarize result directories")
    r.add_argument("results", nargs="+")
    r.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    return {"select": cmd_select, "kernels": cmd_kernels, "report": cmd_report}[
        args.cmd
    ](args)


if __name__ == "__main__":
    sys.exit(main())
