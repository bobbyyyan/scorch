#!/usr/bin/env python3
"""Benchmark weight-sparse autoencoder inference across frameworks.

Train a dense autoencoder on one of several image datasets, apply magnitude
pruning to create sparse weight matrices, then benchmark inference with sparse
weights. Demonstrates Scorch's SpMM (sparse weight x dense activation)
performance vs PyTorch (dense + native sparse) and PyG (torch_sparse).

Frameworks compared (default):
    PyTorch Dense   -- F.linear(x, W_dense, b)                  (baseline, no sparsity)
    PyTorch Sparse  -- torch.sparse.mm(W_csr, x.T)              (native sparse)
    Scorch          -- scorch.matmul(W_stensor, x.T)           (prebuilt SpMM fast-path)
    PyG             -- torch_sparse.matmul(SparseTensor(W), x.T)

Notes on frameworks that were evaluated but are NOT in the default set:
  * DGL: dropped. dgl.sparse's C++ backend is unavailable in this env, and
    GraphConv is graph-message-passing over an adjacency, not a weight-sparse
    linear layer -- no honest mapping for W @ x.
  * Scorch (fused): available via `--frameworks ... scorch-fused` but off by
    default. Fuses the whole Linear epilogue -- SpMM + per-output-channel bias +
    activation -- into ONE prebuilt parallel region via scorch.sparse_linear
    (the prebuilt spmm_csr_linear_fused_float kernel: v2's fast regtile inner
    loops with the bias+act folded in, per-OUTPUT-channel bias, correct for a
    Linear layer). One warm parallel region per layer with no cross-op handoff
    beats the unfused path's SpMM->torch-bias->torch-act composition tax under
    passive OpenMP; it flips the sole @0.99 loss (svhn) to a win and widens the
    others (see bench/README_sparse_autoencoder.md). It is left off by default
    only so the default set matches the other frameworks' unfused SpMM shape;
    pass `scorch-fused` to measure the fused path.

Usage:
    conda run -n scorch python bench/bench_sparse_autoencoder.py train --model all
    conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model all
    conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model mnist --sparsity 0.9 0.95 0.99
    conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model all --frameworks scorch pytorch-dense
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import scorch
from scorch import STensor

from _utils import (
    COLORS,
    EXTRA_COLORS,
    TimingSummary,
    benchmark_fn,
    check_correctness,
    setup_plot_style,
    suppress_torch_warnings,
)

# Optional: torch_sparse (PyG ecosystem) for the PyG framework.
try:
    from torch_sparse import SparseTensor
    from torch_sparse import matmul as ts_matmul
    HAS_TORCH_SPARSE = True
except Exception:  # pragma: no cover - env dependent
    HAS_TORCH_SPARSE = False

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    tv_class: str      # torchvision.datasets class name
    input_dim: int     # flattened image dimension
    uses_split: bool   # True -> constructor takes split="train"/"test"; else train=bool


DATASET_SPECS: Dict[str, DatasetSpec] = {
    # 28x28 grayscale -> 784
    "mnist":    DatasetSpec("MNIST",        784,   False),
    "fashion":  DatasetSpec("FashionMNIST", 784,   False),
    "kmnist":   DatasetSpec("KMNIST",       784,   False),
    # 32x32x3 -> 3072
    "cifar10":  DatasetSpec("CIFAR10",      3072,  False),
    "cifar100": DatasetSpec("CIFAR100",     3072,  False),
    "svhn":     DatasetSpec("SVHN",         3072,  True),
    # 96x96x3 -> 27648 (large first/last layers)
    "stl10":    DatasetSpec("STL10",        27648, True),
}


def _make_tv_dataset(dataset_name: str, train: bool) -> Any:
    spec = DATASET_SPECS[dataset_name]
    cls = getattr(torchvision.datasets, spec.tv_class)
    transform = transforms.ToTensor()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if spec.uses_split:
        split = "train" if train else "test"
        return cls(root=str(DATA_DIR), split=split, download=True, transform=transform)
    return cls(root=str(DATA_DIR), train=train, download=True, transform=transform)


def load_dataset(dataset_name: str, batch_size: int = 256) -> torch.utils.data.DataLoader:
    """Training DataLoader of flattened float32 images."""
    dataset = _make_tv_dataset(dataset_name, train=True)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_input_dim(dataset_name: str) -> int:
    return DATASET_SPECS[dataset_name].input_dim


def get_test_batch(dataset_name: str, batch_size: int) -> torch.Tensor:
    """A single flattened batch from the test set for benchmarking."""
    dataset = _make_tv_dataset(dataset_name, train=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images, _ = next(iter(loader))
    return images.view(images.size(0), -1)  # flatten


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden_dims: List[int]  # [h1, h2]
    dataset: str
    lr: float
    epochs: int
    batch_size: int


# Broad set: spans input dims 784 / 3072 / 27648 and hidden sizes small->large.
# Epochs are deliberately low -- pruned weights just need realistic magnitudes,
# not converged reconstruction, for a weight-sparse SpMM benchmark.
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "mnist":     ModelConfig("mnist",     [1024, 512],  "mnist",    lr=1e-3, epochs=3, batch_size=256),
    "mnist_big": ModelConfig("mnist_big", [4096, 2048], "mnist",    lr=1e-3, epochs=3, batch_size=256),
    "fashion":   ModelConfig("fashion",   [2048, 1024], "fashion",  lr=1e-3, epochs=3, batch_size=256),
    "kmnist":    ModelConfig("kmnist",    [2048, 1024], "kmnist",   lr=1e-3, epochs=3, batch_size=256),
    # NOTE: CIFAR-10/100 dropped -- the Toronto mirror (cs.toronto.edu) is
    # globally throttled to ~25 kB/s (verified from both redwood and a home
    # link). SVHN covers the same 3072-dim input tier at 117 MB/s (Stanford-
    # hosted), so svhn + svhn_big recover the exact [2048,1024] and [4096,2048]
    # 3072-input shapes the CIFAR configs would have provided.
    "svhn":      ModelConfig("svhn",      [2048, 1024], "svhn",     lr=1e-3, epochs=4, batch_size=256),
    "svhn_big":  ModelConfig("svhn_big",  [4096, 2048], "svhn",     lr=1e-3, epochs=4, batch_size=256),
    "stl10":     ModelConfig("stl10",     [4096, 2048], "stl10",    lr=1e-3, epochs=5, batch_size=256),
}

ALL_MODELS = list(MODEL_CONFIGS.keys())

SPARSITY_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

BENCH_BATCH_SIZE = 256

# Order (and identity) of frameworks. "Scorch (fused)" is available but excluded
# from DEFAULT_FRAMEWORKS (see module docstring).
FRAMEWORK_ORDER = ["PyTorch Dense", "PyTorch Sparse", "Scorch", "PyG", "Scorch (fused)"]
DEFAULT_FRAMEWORKS = ["pytorch-dense", "pytorch-sparse", "scorch", "pyg"]

FRAMEWORK_NAME_MAP = {
    "pytorch-dense": "PyTorch Dense",
    "pytorch-sparse": "PyTorch Sparse",
    "scorch": "Scorch",
    "pyg": "PyG",
    "scorch-fused": "Scorch (fused)",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WEIGHT_DIR = Path(__file__).resolve().parent.parent / "weights"

LAYER_NAMES = ["enc1", "enc2", "dec1", "dec2"]
ACTIVATIONS: List[Callable[[torch.Tensor], torch.Tensor]] = [
    F.relu, F.relu, F.relu, torch.sigmoid,
]

# ---------------------------------------------------------------------------
# Autoencoder model
# ---------------------------------------------------------------------------

class DenseAutoencoder(nn.Module):
    """4-layer symmetric autoencoder (2 encoder + 2 decoder)."""

    def __init__(self, input_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.enc1 = nn.Linear(input_dim, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.dec1 = nn.Linear(h2, h1)
        self.dec2 = nn.Linear(h1, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.dec1(x))
        x = torch.sigmoid(self.dec2(x))
        return x


# ---------------------------------------------------------------------------
# Magnitude pruning
# ---------------------------------------------------------------------------

def magnitude_prune(
    state_dict: Dict[str, torch.Tensor], sparsity: float
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Global unstructured magnitude pruning (Han et al. 2015)."""
    weight_keys = [k for k in state_dict if k.endswith(".weight")]
    all_abs = torch.cat([state_dict[k].abs().flatten() for k in weight_keys])
    # torch.quantile fails on tensors > 2^24 elements; use kthvalue instead
    k = max(1, int(sparsity * all_abs.numel()))
    k = min(k, all_abs.numel())
    threshold = all_abs.float().kthvalue(k).values.item()

    pruned = dict(state_dict)
    stats: Dict[str, Any] = {}
    for key in weight_keys:
        w = state_dict[key]
        mask = w.abs() >= threshold
        pruned[key] = w * mask
        total = w.numel()
        nnz = int(mask.sum().item())
        stats[key] = {
            "nnz": nnz,
            "total": total,
            "actual_sparsity": 1.0 - nnz / total,
        }

    return pruned, stats


# ---------------------------------------------------------------------------
# Per-framework inference runners
# ---------------------------------------------------------------------------

def _dense_autoencoder_forward(
    x: torch.Tensor, state_dict: Dict[str, torch.Tensor]
) -> torch.Tensor:
    x = F.relu(F.linear(x, state_dict["enc1.weight"], state_dict["enc1.bias"]))
    x = F.relu(F.linear(x, state_dict["enc2.weight"], state_dict["enc2.bias"]))
    x = F.relu(F.linear(x, state_dict["dec1.weight"], state_dict["dec1.bias"]))
    x = torch.sigmoid(F.linear(x, state_dict["dec2.weight"], state_dict["dec2.bias"]))
    return x


def run_pytorch_dense(
    x: torch.Tensor, state_dict: Dict[str, torch.Tensor], warmup: int, repeats: int,
) -> Tuple[torch.Tensor, TimingSummary]:
    with torch.no_grad():
        def fn() -> torch.Tensor:
            return _dense_autoencoder_forward(x, state_dict)
        return benchmark_fn(fn, warmup=warmup, repeats=repeats)


def run_pytorch_sparse(
    x: torch.Tensor, state_dict: Dict[str, torch.Tensor], warmup: int, repeats: int,
) -> Tuple[torch.Tensor, TimingSummary]:
    """CSR sparse weights via torch.sparse.mm."""
    csr_weights = {
        k: state_dict[k].to_sparse_csr()
        for k in state_dict if k.endswith(".weight")
    }
    with torch.no_grad():
        def fn() -> torch.Tensor:
            h = x
            for ln, act in zip(LAYER_NAMES, ACTIVATIONS):
                w_csr = csr_weights[f"{ln}.weight"]
                b = state_dict[f"{ln}.bias"]
                h = act(torch.sparse.mm(w_csr, h.T).T + b)
            return h
        return benchmark_fn(fn, warmup=warmup, repeats=repeats)


def run_scorch(
    x: torch.Tensor, state_dict: Dict[str, torch.Tensor], warmup: int, repeats: int,
) -> Tuple[torch.Tensor, TimingSummary]:
    """Scorch STensor CSR weights via scorch.matmul (prebuilt SpMM fast-path)."""
    scorch_weights = {
        k: STensor.from_csr(state_dict[k].to_sparse_csr(), k.replace(".weight", ""))
        for k in state_dict if k.endswith(".weight")
    }
    with torch.no_grad():
        def fn() -> torch.Tensor:
            h = x
            for ln, act in zip(LAYER_NAMES, ACTIVATIONS):
                w_st = scorch_weights[f"{ln}.weight"]
                b = state_dict[f"{ln}.bias"]
                h_t = h.T.contiguous()
                h = act(scorch.matmul(w_st, h_t, format="dd").T + b)
            return h
        return benchmark_fn(fn, warmup=warmup, repeats=repeats)


def run_pyg(
    x: torch.Tensor, state_dict: Dict[str, torch.Tensor], warmup: int, repeats: int,
) -> Tuple[torch.Tensor, TimingSummary]:
    """torch_sparse (PyG ecosystem) SpMM: torch_sparse.matmul(SparseTensor, dense)."""
    if not HAS_TORCH_SPARSE:
        raise RuntimeError("torch_sparse not installed")
    sp_weights: Dict[str, Any] = {}
    for key in [k for k in state_dict if k.endswith(".weight")]:
        csr = state_dict[key].to_sparse_csr()
        out_dim, in_dim = state_dict[key].shape
        sp_weights[key] = SparseTensor(
            rowptr=csr.crow_indices().to(torch.long),
            col=csr.col_indices().to(torch.long),
            value=csr.values(),
            sparse_sizes=(out_dim, in_dim),
        )
    with torch.no_grad():
        def fn() -> torch.Tensor:
            h = x
            for ln, act in zip(LAYER_NAMES, ACTIVATIONS):
                w_sp = sp_weights[f"{ln}.weight"]
                b = state_dict[f"{ln}.bias"]
                h_t = h.T.contiguous()
                h = act(ts_matmul(w_sp, h_t).T + b)
            return h
        return benchmark_fn(fn, warmup=warmup, repeats=repeats)


# Fused Linear epilogue via scorch.sparse_linear: each layer's SpMM + per-output
# bias + activation run in ONE prebuilt parallel region (spmm_csr_linear_fused_
# float). Off by default -- see module docstring -- but the fastest Scorch path.
_FUSED_ACTS = ["relu", "relu", "relu", "sigmoid"]


def run_scorch_fused(
    x: torch.Tensor, state_dict: Dict[str, torch.Tensor], warmup: int, repeats: int,
) -> Tuple[torch.Tensor, TimingSummary]:
    """Scorch fused Linear chain: each layer's SpMM + per-output-channel bias +
    activation fused into ONE prebuilt parallel region (spmm_csr_linear_fused_float).

    Runs the chain FEATURE-MAJOR: transpose the input ONCE (via the fast cache-
    blocked scorch.fast_transpose) into [in, batch], keep every intermediate in
    [feat, batch] layout through scorch.sparse_linear_fm, and take a single lazy
    transpose out. This is the documented best path for a multi-layer forward --
    the per-call scorch.sparse_linear drop-in would re-transpose at every layer
    boundary (each layer already yields feature-major output), so it is NOT used
    here. Mirrors torch.sparse.mm(W, h.T).T, which likewise transposes once."""
    scorch_weights = {
        ln: STensor.from_csr(state_dict[f"{ln}.weight"].to_sparse_csr(), ln)
        for ln in LAYER_NAMES
    }
    with torch.no_grad():
        def fn() -> torch.Tensor:
            g = scorch.fast_transpose(x)  # [batch, in] -> [in, batch], once
            for ln, act in zip(LAYER_NAMES, _FUSED_ACTS):
                b = state_dict[f"{ln}.bias"]
                g = scorch.sparse_linear_fm(g, scorch_weights[ln], b, act)
            return g.T  # [out, batch] -> [batch, out] (lazy view), once
        return benchmark_fn(fn, warmup=warmup, repeats=repeats)


FRAMEWORK_RUNNERS: Dict[str, Callable[..., Tuple[torch.Tensor, TimingSummary]]] = {
    "PyTorch Dense": run_pytorch_dense,
    "PyTorch Sparse": run_pytorch_sparse,
    "Scorch": run_scorch,
    "PyG": run_pyg,
    "Scorch (fused)": run_scorch_fused,
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_autoencoder(config: ModelConfig) -> DenseAutoencoder:
    input_dim = get_input_dim(config.dataset)
    model = DenseAutoencoder(input_dim, config.hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loader = load_dataset(config.dataset, config.batch_size)
    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for images, _ in loader:
            x = images.view(images.size(0), -1)
            optimizer.zero_grad()
            recon = model(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        print(f"  Epoch {epoch:3d}/{config.epochs}  loss={avg_loss:.6f}", flush=True)

    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHT_DIR / f"autoencoder_{config.name}.pt"
    torch.save(model.state_dict(), path)
    print(f"  Weights saved to {path}", flush=True)
    return model


# ---------------------------------------------------------------------------
# CSV results
# ---------------------------------------------------------------------------

AE_CSV_COLUMNS = [
    "Model", "Dataset", "InputDim", "Hidden", "Sparsity", "Framework",
    "Median_ms", "Min_ms", "Std_ms",
]


class AEResultsCollector:
    """Collect autoencoder benchmark rows and save incrementally to CSV."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.rows: List[Dict[str, Any]] = []
        self._wrote_header = False

    def append(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        self._write_row(row)

    def _write_row(self, row: Dict[str, Any]) -> None:
        write_header = not self.csv_path.exists() or not self._wrote_header
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=AE_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def benchmark_model(
    config_name: str,
    sparsity_levels: List[float],
    frameworks: List[str],
    warmup: int,
    repeats: int,
    collector: Optional[AEResultsCollector] = None,
) -> List[Dict[str, Any]]:
    config = MODEL_CONFIGS[config_name]
    input_dim = get_input_dim(config.dataset)

    weight_path = WEIGHT_DIR / f"autoencoder_{config_name}.pt"
    if not weight_path.exists():
        print(f"  SKIP: weights not found at {weight_path}. Run 'train --model {config_name}' first.")
        return []

    base_state_dict = torch.load(weight_path, weights_only=True)
    x = get_test_batch(config.dataset, BENCH_BATCH_SIZE)

    hidden_str = "x".join(str(h) for h in config.hidden_dims)
    print(f"\nModel: {config_name}  (dataset={config.dataset}, input={input_dim}, hidden={config.hidden_dims})")
    print(f"  Bench input: ({BENCH_BATCH_SIZE}, {input_dim})", flush=True)

    all_results: List[Dict[str, Any]] = []

    for sparsity in sparsity_levels:
        print(f"\n  Sparsity: {sparsity:.0%}", flush=True)
        pruned_sd, stats = magnitude_prune(base_state_dict, sparsity)
        for key, s in stats.items():
            print(f"    {key}: nnz={s['nnz']:,}/{s['total']:,}  sparsity={s['actual_sparsity']:.2%}")

        reference_output: Optional[torch.Tensor] = None
        results: List[Tuple[str, TimingSummary]] = []

        for fw in FRAMEWORK_ORDER:
            if fw not in frameworks:
                continue
            print(f"\n    {fw}:", flush=True)
            try:
                runner = FRAMEWORK_RUNNERS[fw]
                output, timing = runner(x, pruned_sd, warmup, repeats)
                if fw == "PyTorch Dense":
                    reference_output = output
                elif reference_output is not None:
                    check_correctness(output, reference_output, f"{fw} vs Dense")

                results.append((fw, timing))
                print(
                    f"      median={timing.median_ms:.3f} ms  "
                    f"min={timing.min_ms:.3f} ms  std={timing.std_ms:.3f} ms",
                    flush=True,
                )

                row = {
                    "Model": config_name,
                    "Dataset": config.dataset,
                    "InputDim": input_dim,
                    "Hidden": hidden_str,
                    "Sparsity": sparsity,
                    "Framework": fw,
                    "Median_ms": timing.median_ms,
                    "Min_ms": timing.min_ms,
                    "Std_ms": timing.std_ms,
                }
                if collector is not None:
                    collector.append(row)
                all_results.append(row)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"      SKIPPED (OOM: {e})")
                else:
                    print(f"      FAILED: {e}")
            except Exception as e:
                print(f"      FAILED: {e}")

        if results:
            _print_results_table(config_name, sparsity, results)

    return all_results


def _print_results_table(
    model_name: str, sparsity: float, results: List[Tuple[str, TimingSummary]],
) -> None:
    print(f"\n    {'=' * 62}\n    {model_name} @ {sparsity:.0%} sparsity\n    {'=' * 62}")
    print(f"    {'Framework':<20}| {'Median (ms)':>11} | {'Min (ms)':>9} | {'Std (ms)':>9}")
    print(f"    {'-' * 58}")
    for fw, timing in results:
        print(f"    {fw:<20}| {timing.median_ms:>11.3f} | {timing.min_ms:>9.3f} | {timing.std_ms:>9.3f}")
    print(flush=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

FW_COLORS = {
    "PyTorch Dense": COLORS["PyTorch"],
    "PyTorch Sparse": EXTRA_COLORS[0],
    "Scorch": COLORS["Scorch"],
    "PyG": EXTRA_COLORS[1],
    "Scorch (fused)": EXTRA_COLORS[2],
}
FW_MARKERS = {
    "PyTorch Dense": "s",
    "PyTorch Sparse": "^",
    "Scorch": "o",
    "PyG": "D",
    "Scorch (fused)": "v",
}


def plot_results(
    csv_path: Path, output_path: Path, models: Optional[List[str]] = None,
) -> None:
    """Grid of subplots (one per model): x=sparsity, y=median runtime (ms)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if models is not None:
        df = df[df["Model"].isin(models)]

    model_names = [m for m in MODEL_CONFIGS if m in set(df["Model"].unique())]
    n_models = len(model_names)
    if n_models == 0:
        print("No data to plot.")
        return

    setup_plot_style()
    ncols = min(4, n_models)
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5 * nrows), squeeze=False)

    handles_labels: Optional[Tuple[List[Any], List[str]]] = None
    for idx, model_name in enumerate(model_names):
        ax = axes[idx // ncols, idx % ncols]
        sub = df[df["Model"] == model_name]
        for fw in FRAMEWORK_ORDER:
            fw_data = sub[sub["Framework"] == fw].sort_values("Sparsity")
            if fw_data.empty:
                continue
            ax.plot(
                fw_data["Sparsity"], fw_data["Median_ms"],
                label=fw, color=FW_COLORS.get(fw, "gray"),
                marker=FW_MARKERS.get(fw, "o"), linewidth=2, markersize=7,
            )
        # log y: runtimes span ~0.5ms .. ~1000ms; keeps all 4 frameworks legible
        ax.set_yscale("log")
        dataset = sub["Dataset"].iloc[0]
        hidden = sub["Hidden"].iloc[0]
        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Median Runtime (ms)")
        ax.set_title(f"{model_name}\n{dataset}, {hidden}", fontsize=15)
        ax.grid(True, which="both", alpha=0.3)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    # Hide any unused axes
    for j in range(n_models, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    # Single shared legend (markerscale=1 overrides the scatter-tuned rcParam)
    if handles_labels is not None:
        fig.legend(*handles_labels, loc="upper center", ncol=len(handles_labels[1]),
                   fontsize=14, markerscale=1, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Weight-Sparse Autoencoder Inference (x86, redwood)", fontsize=18, y=1.07)
    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path.resolve()}")


def print_speedup_summary(csv_path: Path) -> None:
    """Print Scorch speedups vs each other framework at the highest sparsity."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty or "Scorch" not in set(df["Framework"]):
        return
    print(f"\n{'=' * 70}\nScorch speedup summary (at max sparsity per model)\n{'=' * 70}")
    print(f"{'Model':<12}{'Sparsity':>9}  {'vs Dense':>10}{'vs Sparse':>11}{'vs PyG':>9}")
    for model_name in [m for m in MODEL_CONFIGS if m in set(df["Model"].unique())]:
        sub = df[df["Model"] == model_name]
        sp = sub["Sparsity"].max()
        s = sub[sub["Sparsity"] == sp]
        piv = dict(zip(s["Framework"], s["Median_ms"]))
        if "Scorch" not in piv:
            continue
        sc = piv["Scorch"]
        def ratio(fw: str) -> str:
            return f"{piv[fw] / sc:.2f}x" if fw in piv and sc > 0 else "-"
        print(f"{model_name:<12}{sp:>8.0%}  {ratio('PyTorch Dense'):>10}"
              f"{ratio('PyTorch Sparse'):>11}{ratio('PyG'):>9}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weight-sparse autoencoder inference benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train autoencoder(s) and save weights")
    train_parser.add_argument("--model", type=str, default="all",
                              help="Model config name or 'all' (default: all)")

    bench_parser = subparsers.add_parser("bench", help="Benchmark sparse autoencoder inference")
    bench_parser.add_argument("--model", type=str, default="all",
                              help="Model config name or 'all' (default: all)")
    bench_parser.add_argument("--sparsity", nargs="+", type=float, default=None,
                              help=f"Sparsity levels (default: {SPARSITY_LEVELS})")
    bench_parser.add_argument("--frameworks", nargs="+", default=DEFAULT_FRAMEWORKS,
                              help=f"Frameworks (default: {DEFAULT_FRAMEWORKS}). "
                                   "Extra: scorch-fused")
    bench_parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    bench_parser.add_argument("--repeats", type=int, default=20, help="Timed iterations (default: 20)")
    bench_parser.add_argument("--output-dir", type=str, default="bench_results",
                              help="Directory for CSV and plot output")
    bench_parser.add_argument("--csv", type=str, default="sparse_autoencoder_results.csv",
                              help="CSV filename")
    bench_parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    bench_parser.add_argument("--plot-only", action="store_true",
                              help="Skip benchmarking; load CSV and regenerate plot")

    args = parser.parse_args()
    suppress_torch_warnings()
    torch.manual_seed(42)
    print(f"torch threads: {torch.get_num_threads()}", flush=True)

    if args.command == "train":
        models = ALL_MODELS if args.model.lower() == "all" else [args.model]
        for model_name in models:
            if model_name not in MODEL_CONFIGS:
                print(f"Unknown model config: {model_name}")
                continue
            config = MODEL_CONFIGS[model_name]
            print(f"\n{'=' * 62}")
            print(f"Training: {model_name} (dataset={config.dataset}, hidden={config.hidden_dims}, epochs={config.epochs})")
            print(f"{'=' * 62}", flush=True)
            train_autoencoder(config)

    elif args.command == "bench":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / args.csv
        plot_path = output_dir / f"sparse_autoencoder.{args.format}"

        models = ALL_MODELS if args.model.lower() == "all" else [args.model]
        frameworks = [FRAMEWORK_NAME_MAP.get(f.lower(), f) for f in args.frameworks]
        sparsity_levels = args.sparsity if args.sparsity is not None else SPARSITY_LEVELS

        if not args.plot_only:
            if csv_path.exists():
                csv_path.unlink()
            collector = AEResultsCollector(csv_path)
            for model_name in models:
                if model_name not in MODEL_CONFIGS:
                    print(f"Unknown model config: {model_name}")
                    continue
                benchmark_model(
                    model_name, sparsity_levels=sparsity_levels, frameworks=frameworks,
                    warmup=args.warmup, repeats=args.repeats, collector=collector,
                )

        if csv_path.exists():
            plot_results(csv_path, plot_path, models=models)
            print_speedup_summary(csv_path)
        else:
            print(f"No CSV found at {csv_path}; nothing to plot.")


if __name__ == "__main__":
    main()
