#!/usr/bin/env python3
"""Per-matrix speedup view of SpMM benchmark results.

The default SpMM plot (``plot_scatter_loglog`` in ``_utils.py``) draws two
overlapping NNZ-vs-runtime clouds; the dominant visual signal there is matrix
size, not the Scorch-vs-PyTorch gap. This script instead plots the quantity the
benchmark is actually arguing about -- the per-matrix speedup
(PyTorch runtime / Scorch runtime) against NNZ -- so every point reads directly
against a parity line, and a geomean-by-size trend line shows how the advantage
scales. A right-hand marginal histogram summarises the speedup distribution.

Reads the CSV written by ``bench_spmm.py`` (columns: Framework, MatrixName, NNZ,
Runtime, ...). Runtimes are aggregated to one median per (matrix, framework).

Usage:
    python bench/plot_spmm_speedup.py \
        --csv bench_results/spmm_full_fast.csv \
        --out bench_results/spmm_speedup.pdf

Output format is inferred from --out's extension (.pdf/.svg give vector output).
Defaults to a transparent background so the figure composits onto any slide or
paper background; pass --opaque for a white background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Scorch / PyTorch house colours (mirrors COLORS in bench/_utils.py).
ORANGE, NAVY = "#fc764a", "#19526c"
INK, MUTED = "#1a1a1a", "#8a8a8a"


def _load_paired(csv_path: Path) -> pd.DataFrame:
    """One row per matrix with median Scorch/PyTorch runtime, NNZ and speedup."""
    df = pd.read_csv(csv_path)
    med = df.groupby(["MatrixName", "Framework"]).Runtime.median().unstack()
    nnz = df.groupby("MatrixName").NNZ.first()
    m = med.join(nnz).dropna(subset=["Scorch", "PyTorch"])
    m = m[(m["NNZ"] > 0) & (m["Scorch"] > 0) & (m["PyTorch"] > 0)].copy()
    m["speedup"] = m["PyTorch"] / m["Scorch"]  # > 1  => Scorch faster
    return m


def _geomean_trend(m: pd.DataFrame, step: float = 0.5, min_count: int = 5):
    """Geometric-mean speedup per ``step``-decade bin of NNZ."""
    lb = np.log10(m["NNZ"])
    edges = np.arange(np.floor(lb.min() / step) * step,
                      np.ceil(lb.max() / step) * step + step, step)
    cx, cy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = m[(lb >= lo) & (lb < hi)]
        if len(g) >= min_count:
            cx.append(10 ** ((lo + hi) / 2))
            cy.append(np.exp(np.log(g["speedup"]).mean()))
    return cx, cy


def plot_speedup(m: pd.DataFrame, title: str, out: Path, transparent: bool) -> None:
    win = m["speedup"] >= 1
    n = len(m)
    pct = win.mean() * 100
    geo = np.exp(np.log(m["speedup"]).mean())
    medd = m["speedup"].median()
    hi_mask = m["NNZ"] >= 1e7
    geo_hi = np.exp(np.log(m.loc[hi_mask, "speedup"]).mean()) if hi_mask.any() else float("nan")
    cx, cy = _geomean_trend(m)

    sns.set(style="white", context="talk")
    plt.rcParams.update({
        "grid.linestyle": " ", "font.size": 17,
        "axes.labelsize": 22, "axes.titlesize": 24,
        "xtick.labelsize": 20, "ytick.labelsize": 18, "legend.fontsize": 18,
    })

    fig, (ax, axm) = plt.subplots(
        1, 2, figsize=(15, 6.5), sharey=True,
        gridspec_kw=dict(width_ratios=[5, 1], wspace=0.04),
    )

    # Faint bands: above parity = Scorch faster, below = slower.
    ax.axhspan(1, 1e3, color=ORANGE, alpha=0.05, zorder=0)
    ax.axhspan(1e-3, 1, color=NAVY, alpha=0.05, zorder=0)
    ax.axhline(1.0, color=MUTED, lw=2, zorder=1)
    ax.text(0.985, 1.045, "parity", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", color=MUTED, fontsize=15)

    ax.scatter(m.loc[win, "NNZ"], m.loc[win, "speedup"], s=7, alpha=0.55,
               color=ORANGE, linewidths=0, zorder=2,
               label=f"Scorch faster  ({pct:.0f}%)")
    ax.scatter(m.loc[~win, "NNZ"], m.loc[~win, "speedup"], s=7, alpha=0.55,
               color=NAVY, linewidths=0, zorder=2,
               label=f"PyTorch faster  ({100 - pct:.0f}%)")
    ax.plot(cx, cy, color=INK, lw=2.5, marker="o", ms=6, zorder=4,
            label="geomean by size")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.08, 13)
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.1, 0.25, 0.5, 1, 2, 4, 8]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Number of Non-Zeros (NNZ)")
    ax.set_ylabel("Speedup  (PyTorch ÷ Scorch runtime)")
    ax.set_title(title)

    ann = (f"Scorch faster on {pct:.0f}% of {n:,} matrices\n"
           f"geomean {geo:.2f}×   median {medd:.2f}×\n"
           f"grows with size → {geo_hi:.1f}× at ≥10⁷ NNZ")
    ax.text(0.025, 0.965, ann, transform=ax.transAxes, va="top", ha="left",
            fontsize=16, color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=MUTED, alpha=0.9))
    ax.legend(loc="lower right", framealpha=0.9, markerscale=2.2)

    # Right marginal: distribution of speedups on the shared log y-axis.
    bins = np.logspace(np.log10(0.08), np.log10(13), 46)
    axm.hist(m.loc[win, "speedup"], bins=bins, orientation="horizontal",
             color=ORANGE, alpha=0.85)
    axm.hist(m.loc[~win, "speedup"], bins=bins, orientation="horizontal",
             color=NAVY, alpha=0.85)
    axm.axhline(1.0, color=MUTED, lw=2)
    axm.axhline(medd, color=INK, lw=2, ls="--")
    axm.text(0.94, medd * 1.06, f"median {medd:.2f}×",
             transform=axm.get_yaxis_transform(), ha="right", va="bottom",
             color=INK, fontsize=13)
    axm.set_xticks([])
    axm.set_xlabel("# matrices", fontsize=15)
    for spine in ("top", "right", "bottom"):
        axm.spines[spine].set_visible(False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
    print(f"Plot saved to {out.resolve()}  "
          f"(n={n}, {pct:.0f}% faster, geomean {geo:.2f}x)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path,
                        default=Path("bench_results/spmm_results.csv"),
                        help="Input CSV from bench_spmm.py "
                             "(default: bench_results/spmm_results.csv)")
    parser.add_argument("--out", type=Path,
                        default=Path("bench_results/spmm_speedup.pdf"),
                        help="Output path; format inferred from extension "
                             "(default: bench_results/spmm_speedup.pdf)")
    parser.add_argument("--k", type=int, default=128,
                        help="k (dense width) shown in the title (default: 128)")
    parser.add_argument("--opaque", dest="transparent", action="store_false",
                        help="Use a white background instead of transparent")
    parser.set_defaults(transparent=True)
    args = parser.parse_args()

    m = _load_paired(args.csv)
    title = f"Scorch vs PyTorch SpMM — per-matrix speedup (k={args.k})"
    plot_speedup(m, title, args.out, args.transparent)


if __name__ == "__main__":
    main()
