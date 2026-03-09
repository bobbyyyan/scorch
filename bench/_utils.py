"""Shared utilities for Scorch benchmark scripts."""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import subprocess
import tarfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
import seaborn as sns
import torch

import scorch
from scorch import STensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "scorch_suitesparse"

COLORS: Dict[str, str] = {
    "Scorch": "#fc764a",
    "PyTorch": "#19526c",
}
EXTRA_COLORS: List[str] = ["#1AACAC", "#E7B10A", "#ED5AB3"]

# ---------------------------------------------------------------------------
# Matrix sets – curated (ssid, name, description) tuples per kernel.
# Spanning NNZ from ~1K to ~50M.
# ---------------------------------------------------------------------------

# Each entry: (ssid, group, name, description)
MatrixEntry = Tuple[int, str, str, str]

_GENERAL_MATRICES: List[MatrixEntry] = [
    # Small (~400–2K NNZ)
    (6, "HB", "arc130", "130x130, nnz=1037"),
    (2, "HB", "494_bus", "494x494, nnz=1666"),
    (8, "HB", "ash292", "292x292, nnz=2208"),
    # Small-medium (~5K–13K NNZ)
    (18, "HB", "bcspwr06", "1454x1454, nnz=5300"),
    (21, "HB", "bcspwr09", "1723x1723, nnz=6511"),
    (30, "HB", "bcsstk08", "1074x1074, nnz=12960"),
    # Medium (~60K–430K NNZ)
    (36, "HB", "bcsstk14", "1806x1806, nnz=63454"),
    (35, "HB", "bcsstk13", "2003x2003, nnz=83883"),
    (37, "HB", "bcsstk15", "3948x3948, nnz=117816"),
    (38, "HB", "bcsstk16", "4884x4884, nnz=290378"),
    (39, "HB", "bcsstk17", "10974x10974, nnz=428650"),
    # Large (~600K–4.2M NNZ)
    (55, "HB", "bcsstk33", "8738x8738, nnz=591904"),
    (51, "HB", "bcsstk29", "13992x13992, nnz=619488"),
    (351, "Boeing", "crystk02", "13965x13965, nnz=968583"),
    (53, "HB", "bcsstk31", "35588x35588, nnz=1181416"),
    (352, "Boeing", "crystk03", "24696x24696, nnz=1751178"),
    (52, "HB", "bcsstk30", "28924x28924, nnz=2043492"),
    (356, "Boeing", "ct20stif", "52329x52329, nnz=2600295"),
    (537, "Gupta", "gupta2", "62064x62064, nnz=4248286"),
    # Very large (~5M–12M NNZ)
    (285, "ATandT", "pre2", "659033x659033, nnz=5834044"),
    (857, "Chen", "pkustk11", "87804x87804, nnz=5217912"),
]

# SpMSpM: bounded to max(rows,cols)<100K and nnz<10M
_SPMSPM_MATRICES: List[MatrixEntry] = [
    (6, "HB", "arc130", "130x130, nnz=1037"),
    (2, "HB", "494_bus", "494x494, nnz=1666"),
    (8, "HB", "ash292", "292x292, nnz=2208"),
    (18, "HB", "bcspwr06", "1454x1454, nnz=5300"),
    (30, "HB", "bcsstk08", "1074x1074, nnz=12960"),
    (36, "HB", "bcsstk14", "1806x1806, nnz=63454"),
    (35, "HB", "bcsstk13", "2003x2003, nnz=83883"),
    (38, "HB", "bcsstk16", "4884x4884, nnz=290378"),
    (39, "HB", "bcsstk17", "10974x10974, nnz=428650"),
    (55, "HB", "bcsstk33", "8738x8738, nnz=591904"),
    (351, "Boeing", "crystk02", "13965x13965, nnz=968583"),
    (352, "Boeing", "crystk03", "24696x24696, nnz=1751178"),
    (52, "HB", "bcsstk30", "28924x28924, nnz=2043492"),
    (356, "Boeing", "ct20stif", "52329x52329, nnz=2600295"),
    (537, "Gupta", "gupta2", "62064x62064, nnz=4248286"),
]

MATRIX_SETS: Dict[str, List[MatrixEntry]] = {
    "spmv": _GENERAL_MATRICES,
    "spmm": _GENERAL_MATRICES,
    "spmspm": _SPMSPM_MATRICES,
    "sddmm": _GENERAL_MATRICES,
}


# ---------------------------------------------------------------------------
# Matrix download & conversion
# ---------------------------------------------------------------------------

SUITESPARSE_URL = "https://suitesparse-collection-website.herokuapp.com/MM"


def download_matrix(ssid: int, group: str, name: str) -> scipy.sparse.csr_matrix:
    """Download a SuiteSparse matrix via curl, cache locally, return as scipy CSR."""
    cache_path = CACHE_DIR / f"{name}.mtx"
    if cache_path.exists():
        mat = scipy.io.mmread(str(cache_path))
        return scipy.sparse.csr_matrix(mat, dtype=np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{SUITESPARSE_URL}/{group}/{name}.tar.gz"
    tar_path = CACHE_DIR / f"{name}.tar.gz"

    print(f"  Downloading {url} ... ", end="", flush=True)
    subprocess.run(
        ["curl", "-fSL", "-o", str(tar_path), url],
        check=True,
    )
    print("OK")

    # Extract the .mtx file from the tarball
    with tarfile.open(str(tar_path), "r:gz") as tf:
        tf.extractall(path=str(CACHE_DIR))
    tar_path.unlink()

    extracted = CACHE_DIR / name / f"{name}.mtx"
    if extracted.exists():
        mat = scipy.io.mmread(str(extracted))
        shutil.copy2(str(extracted), str(cache_path))
    elif cache_path.exists():
        mat = scipy.io.mmread(str(cache_path))
    else:
        raise RuntimeError(f"Could not find downloaded matrix for {name} (id={ssid})")
    return scipy.sparse.csr_matrix(mat, dtype=np.float32)


def scipy_to_torch(matrix: scipy.sparse.spmatrix, fmt: str = "csr") -> torch.Tensor:
    """Convert scipy sparse matrix to PyTorch sparse tensor."""
    if fmt == "coo":
        coo = matrix.tocoo()
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(i, v, torch.Size(coo.shape))
    elif fmt == "csr":
        csr = matrix.tocsr()
        crow = torch.from_numpy(csr.indptr.astype(np.int32))
        cols = torch.from_numpy(csr.indices.astype(np.int32))
        vals = torch.from_numpy(csr.data.astype(np.float32))
        return torch.sparse_csr_tensor(crow, cols, vals, size=csr.shape)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")


# ---------------------------------------------------------------------------
# Scorch tensor helpers
# ---------------------------------------------------------------------------

def to_scorch_csr(csr: scipy.sparse.csr_matrix, name: str = "A") -> STensor:
    """Build an STensor (CSR) from a scipy CSR matrix."""
    indptr = torch.from_numpy(csr.indptr.astype(np.int32))
    indices = torch.from_numpy(csr.indices.astype(np.int32))
    values = torch.from_numpy(csr.data.astype(np.float32))
    torch_csr = torch.sparse_csr_tensor(indptr, indices, values, size=csr.shape)
    return STensor.from_csr(torch_csr, name)


def to_scorch_dense(tensor: torch.Tensor, name: str = "B") -> STensor:
    """Wrap a dense torch tensor as an STensor."""
    return STensor.from_torch(tensor, name)


def to_scorch_coo(torch_coo: torch.Tensor, name: str = "A") -> STensor:
    """Build an STensor (COO) from a torch COO tensor."""
    return STensor.from_coo(torch_coo, name)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimingSummary:
    """Summary statistics for a set of timing measurements (in seconds)."""
    times: Tuple[float, ...]

    @property
    def min_s(self) -> float:
        return min(self.times)

    @property
    def median_s(self) -> float:
        return statistics.median(self.times)

    @property
    def mean_s(self) -> float:
        return statistics.mean(self.times)

    @property
    def std_s(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return self.min_s * 1e3

    @property
    def median_ms(self) -> float:
        return self.median_s * 1e3

    @property
    def mean_ms(self) -> float:
        return self.mean_s * 1e3

    @property
    def std_ms(self) -> float:
        return self.std_s * 1e3


def benchmark_fn(
    fn: Callable[[], Any],
    warmup: int = 3,
    repeats: int = 10,
) -> Tuple[Any, TimingSummary]:
    """Run *fn* with warmup, then time *repeats* iterations."""
    result = None
    for _ in range(warmup):
        result = fn()
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return result, TimingSummary(times=tuple(times))


# ---------------------------------------------------------------------------
# Results & CSV
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "Framework", "MatrixName", "MatrixID", "Rows", "Cols", "NNZ", "K", "Runtime",
]


class ResultsCollector:
    """Collect benchmark rows and save incrementally to CSV."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.rows: List[Dict[str, Any]] = []
        self._wrote_header = False

    def append(
        self,
        framework: str,
        matrix_name: str,
        matrix_id: int,
        rows: int,
        cols: int,
        nnz: int,
        runtime: float,
        k: int = 0,
    ) -> None:
        row = {
            "Framework": framework,
            "MatrixName": matrix_name,
            "MatrixID": matrix_id,
            "Rows": rows,
            "Cols": cols,
            "NNZ": nnz,
            "K": k,
            "Runtime": runtime,
        }
        self.rows.append(row)
        self._write_row(row)

    def _write_row(self, row: Dict[str, Any]) -> None:
        write_header = not self.csv_path.exists() or not self._wrote_header
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=CSV_COLUMNS)

    @classmethod
    def load(cls, csv_path: Path) -> pd.DataFrame:
        return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def setup_plot_style() -> None:
    """Configure seaborn / matplotlib for publication-quality plots."""
    sns.set(style="white", context="talk")
    plt.rcParams.update({
        "grid.linestyle": " ",
        "font.size": 17,
        "axes.labelsize": 22,
        "axes.titlesize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 18,
        "legend.fontsize": 22,
        "legend.title_fontsize": 24,
        "legend.markerscale": 5,
    })


def plot_scatter_loglog(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """NNZ vs Runtime log-log scatter plot."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(15, 6))

    for framework in ["Scorch", "PyTorch"]:
        sub = df[df["Framework"] == framework]
        if sub.empty:
            continue
        color = COLORS.get(framework, EXTRA_COLORS[0])
        ax.scatter(
            sub["NNZ"], sub["Runtime"],
            label=framework, s=2, alpha=0.7, color=color,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Non-Zeros (NNZ)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path.resolve()}")


# ---------------------------------------------------------------------------
# Common argparse
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add standard benchmark arguments."""
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations (default: 3)")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of timed iterations (default: 10)")
    parser.add_argument("--output-dir", type=str, default="bench_results",
                        help="Directory for CSV and plot output (default: bench_results)")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV filename override (default: <kernel>_results.csv)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip benchmarking; load CSV and regenerate plot")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg"],
                        help="Plot output format (default: png)")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def suppress_torch_warnings() -> None:
    """Suppress PyTorch sparse CSR beta warnings."""
    warnings.filterwarnings(
        "ignore", category=UserWarning,
        message="Sparse CSR tensor support is in beta state.*",
    )


def check_correctness(
    result: torch.Tensor,
    reference: torch.Tensor,
    label: str,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> bool:
    """Check two tensors are close; print warning if not."""
    if torch.allclose(result, reference, atol=atol, rtol=rtol):
        return True
    max_diff = (result.float() - reference.float()).abs().max().item()
    print(f"  WARNING: {label} correctness check failed (max_diff={max_diff:.6f})")
    return False
